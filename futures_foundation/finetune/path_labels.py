"""Dense, causal forward-path labels for cross-family futures evaluation.

The decision row is bar ``i`` and uses the close at ``i`` as the path origin.  Every target uses
bars strictly after ``i`` through an exact wall-clock endpoint.  Inputs may be dense, but a label is
masked when its complete future interval is unavailable, crosses a contract roll, or violates the
declared bar cadence.

Barrier states deliberately distinguish OHLC ambiguity from an adverse-first executable score:

``NEITHER``
    Neither barrier was touched during the horizon.
``FAVORABLE_FIRST``
    The favorable barrier was touched on an earlier bar.
``ADVERSE_FIRST``
    The adverse barrier was touched on an earlier bar.
``AMBIGUOUS``
    The first favorable and adverse touches occur in the same OHLC bar, so their order is unknown.

This module does not build model features.  Its ATR scale is Wilder-smoothed and strictly causal;
the value stored for row ``i`` depends only on bars at or before ``i``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np
import pandas as pd

from futures_foundation.pipeline._primitives import compute_atr
from futures_foundation.session_gap import (
    VerifiedSessionGapCapability,
    require_session_gap_capability,
    verified_session_edge_mask,
)


SCHEMA_VERSION = "ffm_dense_path_labels_v4"

BARRIER_NEITHER = np.int8(0)
BARRIER_FAVORABLE_FIRST = np.int8(1)
BARRIER_ADVERSE_FIRST = np.int8(2)
BARRIER_AMBIGUOUS = np.int8(3)

TREND_TERMINATION = np.int8(0)
TREND_CONTINUATION = np.int8(1)
TREND_REVERSAL = np.int8(2)
TREND_UNDEFINED = np.int8(-1)

INVALID_NONE = np.uint8(0)
INVALID_INSUFFICIENT_FUTURE = np.uint8(1)
INVALID_CADENCE = np.uint8(2)
INVALID_CONTRACT_ROLL = np.uint8(3)
INVALID_SCALE_OR_PRICE = np.uint8(4)


@dataclass(frozen=True)
class PathLabelConfig:
    """Versioned semantics for dense path labels.

    All durations are elapsed UTC minutes.  They must be exact multiples of the stream timeframe;
    this prevents model families from receiving economically different fixed-bar horizons.
    """

    horizons_minutes: tuple[int, ...] = (60, 180, 360)
    targets_r: tuple[float, ...] = (1.0, 2.0, 3.0)
    adverse_r: float = 1.0
    atr_period: int = 20
    context_minutes: int = 60
    context_deadband_r: float = 0.25
    barrier_chunk_rows: int = 8192

    def validate(self, timeframe_minutes: int) -> None:
        timeframe_minutes = int(timeframe_minutes)
        if timeframe_minutes <= 0:
            raise ValueError("timeframe_minutes must be positive")
        if not self.horizons_minutes or any(int(x) <= 0 for x in self.horizons_minutes):
            raise ValueError("horizons_minutes must contain positive values")
        if tuple(sorted(set(self.horizons_minutes))) != tuple(self.horizons_minutes):
            raise ValueError("horizons_minutes must be unique and increasing")
        if any(int(x) % timeframe_minutes for x in self.horizons_minutes):
            raise ValueError("every horizon must be an exact multiple of the timeframe")
        if self.context_minutes <= 0 or self.context_minutes % timeframe_minutes:
            raise ValueError("context_minutes must be a positive multiple of the timeframe")
        if not self.targets_r or any(float(x) <= 0 for x in self.targets_r):
            raise ValueError("targets_r must contain positive values")
        if tuple(sorted(set(self.targets_r))) != tuple(self.targets_r):
            raise ValueError("targets_r must be unique and increasing")
        if self.adverse_r <= 0 or self.atr_period <= 0 or self.barrier_chunk_rows <= 0:
            raise ValueError("adverse_r, atr_period and barrier_chunk_rows must be positive")


def _required_frame(frame: pd.DataFrame) -> tuple[np.ndarray, ...]:
    required = {"datetime", "open", "high", "low", "close", "contract_id"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"path-label frame is missing columns: {missing}")
    ts = pd.DatetimeIndex(pd.to_datetime(frame["datetime"], utc=True)).asi8
    if len(ts) and np.any(np.diff(ts) <= 0):
        raise ValueError("datetime must be strictly increasing and duplicate-free")
    o, h, l, c = (
        np.asarray(frame[name], dtype=np.float64)
        for name in ("open", "high", "low", "close")
    )
    if not (len(ts) == len(o) == len(h) == len(l) == len(c)):
        raise ValueError("OHLC arrays must have equal lengths")
    finite = np.isfinite(o) & np.isfinite(h) & np.isfinite(l) & np.isfinite(c)
    if np.any(finite & ((h < np.maximum(o, c)) | (l > np.minimum(o, c)) | (h < l))):
        raise ValueError("invalid OHLC geometry")
    if frame["contract_id"].isna().any():
        raise ValueError("contract_id must be non-empty")
    contract = frame["contract_id"].astype(str).str.strip().to_numpy(dtype=str)
    if np.any(np.char.str_len(contract) == 0):
        raise ValueError("contract_id must be non-empty")
    segment = np.r_[0, np.cumsum(contract[1:] != contract[:-1])].astype(np.int64)
    return ts, o, h, l, c, segment


def _segmented_atr(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, segment: np.ndarray, period: int,
) -> np.ndarray:
    """Compute causal ATR independently inside every contiguous contract segment."""
    output = np.full(len(close), np.nan, dtype=np.float64)
    for segment_value in np.unique(segment):
        rows = np.flatnonzero(segment == segment_value)
        output[rows] = compute_atr(high[rows], low[rows], close[rows], int(period))
    return output


def _prefix_sums(values: np.ndarray) -> np.ndarray:
    return np.r_[0.0, np.cumsum(values, dtype=np.float64)]


def _first_touch(mask: np.ndarray) -> np.ndarray:
    """Return one-based bar offsets and zero when no element is true."""
    touched = np.any(mask, axis=1)
    first = np.zeros(len(mask), dtype=np.int32)
    first[touched] = np.argmax(mask[touched], axis=1).astype(np.int32) + 1
    return first


def _barrier_labels(
    high_windows: np.ndarray,
    low_windows: np.ndarray,
    rows: np.ndarray,
    base: np.ndarray,
    scale: np.ndarray,
    terminal_move_r: np.ndarray,
    targets: np.ndarray,
    adverse_r: float,
    timeframe_minutes: int,
    chunk_rows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized first-touch labels for long (+1) and short (-1) paths."""
    m, t = len(rows), len(targets)
    state = np.full((m, 2, t), BARRIER_NEITHER, dtype=np.int8)
    time_fav = np.full((m, 2, t), -1, dtype=np.int32)
    time_adv = np.full((m, 2, t), -1, dtype=np.int32)
    policy_r = np.empty((m, 2, t), dtype=np.float32)
    if not m:
        return state, time_fav, time_adv, policy_r

    for start in range(0, m, int(chunk_rows)):
        stop = min(start + int(chunk_rows), m)
        source_rows = rows[start:stop]
        hw = high_windows[source_rows]
        lw = low_windows[source_rows]
        b = base[source_rows, None]
        s = scale[source_rows, None]

        for direction_i, direction in enumerate((1.0, -1.0)):
            if direction > 0:
                adverse_touch = lw <= b - float(adverse_r) * s
            else:
                adverse_touch = hw >= b + float(adverse_r) * s
            adverse_at = _first_touch(adverse_touch)

            for target_i, target in enumerate(targets):
                if direction > 0:
                    favorable_touch = hw >= b + float(target) * s
                else:
                    favorable_touch = lw <= b - float(target) * s
                favorable_at = _first_touch(favorable_touch)

                neither = (favorable_at == 0) & (adverse_at == 0)
                favorable = (favorable_at > 0) & ((adverse_at == 0) | (favorable_at < adverse_at))
                adverse = (adverse_at > 0) & ((favorable_at == 0) | (adverse_at < favorable_at))
                ambiguous = (favorable_at > 0) & (favorable_at == adverse_at)

                out = state[start:stop, direction_i, target_i]
                out[favorable] = BARRIER_FAVORABLE_FIRST
                out[adverse] = BARRIER_ADVERSE_FIRST
                out[ambiguous] = BARRIER_AMBIGUOUS

                time_fav[start:stop, direction_i, target_i] = np.where(
                    favorable_at > 0, favorable_at * int(timeframe_minutes), -1
                )
                time_adv[start:stop, direction_i, target_i] = np.where(
                    adverse_at > 0, adverse_at * int(timeframe_minutes), -1
                )

                gross = policy_r[start:stop, direction_i, target_i]
                gross[neither] = direction * terminal_move_r[source_rows][neither]
                gross[favorable] = float(target)
                # Executable OHLC score is deliberately conservative for ambiguous bars.
                gross[adverse | ambiguous] = -float(adverse_r)
    return state, time_fav, time_adv, policy_r


def build_dense_path_labels(
    frame: pd.DataFrame,
    *,
    timeframe_minutes: int,
    config: PathLabelConfig | None = None,
    causal_scale: np.ndarray | None = None,
    context_direction: np.ndarray | None = None,
    session_gap_capability: VerifiedSessionGapCapability | None = None,
) -> dict[str, object]:
    """Materialize dense path targets for one sorted, roll-identified OHLC stream.

    Returned path arrays have shape ``[rows, horizons]``.  Barrier arrays have shape
    ``[rows, horizons, directions, targets]``, with direction order ``(+1, -1)``.  Invalid floating
    labels are NaN, invalid class/time labels are -1, and ``valid``/``invalid_reason`` are the
    authoritative masks.
    """
    config = config or PathLabelConfig()
    config.validate(timeframe_minutes)
    ts, o, h, l, c, segment = _required_frame(frame)
    n = len(ts)
    if n == 0:
        raise ValueError("path-label frame must contain at least one row")
    horizons = np.asarray(config.horizons_minutes, dtype=np.int32)
    targets = np.asarray(config.targets_r, dtype=np.float32)
    directions = np.asarray((1, -1), dtype=np.int8)
    h_count, target_count = len(horizons), len(targets)
    expected_ns = int(timeframe_minutes) * 60 * 1_000_000_000

    admitted_scale = _segmented_atr(h, l, c, segment, int(config.atr_period))
    if causal_scale is not None:
        supplied_scale = np.asarray(causal_scale, dtype=np.float64)
        if supplied_scale.shape != (n,):
            raise ValueError(f"causal_scale must have shape {(n,)}, got {supplied_scale.shape}")
        if not np.array_equal(supplied_scale, admitted_scale, equal_nan=True):
            raise ValueError(
                "external causal_scale is not identical to contract-segmented causal ATR"
            )
    scale = admitted_scale

    valid = np.zeros((n, h_count), dtype=bool)
    invalid_reason = np.full((n, h_count), INVALID_INSUFFICIENT_FUTURE, dtype=np.uint8)
    label_end_time_ns = np.full((n, h_count), -1, dtype=np.int64)
    float_names = (
        "terminal_move_r", "forward_abs_move_r",
        "forward_realized_vol", "upside_mfe_r", "downside_mae_r", "forward_trend_eff",
    )
    result: dict[str, object] = {
        name: np.full((n, h_count), np.nan, dtype=np.float32) for name in float_names
    }
    trend_path_class = np.full((n, h_count), TREND_UNDEFINED, dtype=np.int8)
    barrier_state = np.full((n, h_count, 2, target_count), -1, dtype=np.int8)
    time_to_favorable_minutes = np.full_like(barrier_state, -1, dtype=np.int32)
    time_to_adverse_minutes = np.full_like(barrier_state, -1, dtype=np.int32)
    policy_r_gross = np.full(barrier_state.shape, np.nan, dtype=np.float32)

    if session_gap_capability is None:
        edge_valid = np.diff(ts) == expected_ns
        session_manifest = None
        horizon_basis = "elapsed_utc_minutes_exact_cadence_v1"
    else:
        session_gap_capability = require_session_gap_capability(session_gap_capability)
        edge_valid = verified_session_edge_mask(
            pd.to_datetime(ts, utc=True),
            expected_delta=pd.Timedelta(expected_ns, unit="ns"),
            capability=session_gap_capability,
        )
        session_manifest = session_gap_capability.manifest()
        horizon_basis = "admitted_bar_minutes_across_verified_sessions_v1"
    bad_cadence = ~edge_valid
    bad_cadence_prefix = np.r_[0, np.cumsum(bad_cadence, dtype=np.int64)]
    bar_ok = np.isfinite(o) & np.isfinite(h) & np.isfinite(l) & np.isfinite(c)
    bad_bar_prefix = np.r_[0, np.cumsum(~bar_ok, dtype=np.int64)]
    price_ok = bar_ok & np.isfinite(scale) & (scale > 0)
    # Log returns are undefined for legitimate futures prices at or below zero.  Normalize raw
    # increments by the decision-time causal ATR below so targets remain translation-safe and
    # never use a future-derived denominator.
    price_change = np.diff(c)
    change_prefix = _prefix_sums(np.nan_to_num(price_change, nan=0.0))
    change_sq_prefix = _prefix_sums(np.nan_to_num(price_change * price_change, nan=0.0))
    abs_change_prefix = _prefix_sums(np.nan_to_num(np.abs(price_change), nan=0.0))

    context_steps = int(config.context_minutes) // int(timeframe_minutes)
    if context_direction is None:
        context_dir = np.zeros(n, dtype=np.int8)
        context_valid = np.zeros(n, dtype=bool)
        rows = np.arange(context_steps, n, dtype=np.int64)
        starts = rows - context_steps
        context_valid[rows] = (
            (bad_cadence_prefix[rows] - bad_cadence_prefix[starts] == 0)
            & (segment[rows] == segment[starts])
            & price_ok[rows]
            & np.isfinite(c[starts])
        )
        net_r = np.full(n, np.nan, dtype=np.float64)
        good_context_rows = rows[context_valid[rows]]
        good_context_starts = starts[context_valid[rows]]
        net_r[good_context_rows] = (
            (c[good_context_rows] - c[good_context_starts]) / scale[good_context_rows]
        )
        active = context_valid & (np.abs(net_r) >= float(config.context_deadband_r))
        context_dir[active] = np.sign(net_r[active]).astype(np.int8)
    else:
        context_dir = np.asarray(context_direction, dtype=np.int8)
        if context_dir.shape != (n,) or not np.isin(context_dir, (-1, 0, 1)).all():
            raise ValueError("context_direction must have shape [rows] and values in {-1, 0, 1}")

    for horizon_i, horizon_minutes in enumerate(horizons):
        steps = int(horizon_minutes) // int(timeframe_minutes)
        if n <= steps:
            continue
        rows = np.arange(0, n - steps, dtype=np.int64)
        ends = rows + steps
        label_end_time_ns[rows, horizon_i] = ts[ends]

        cadence_ok = bad_cadence_prefix[ends] - bad_cadence_prefix[rows] == 0
        roll_ok = segment[rows] == segment[ends]
        future_bars_ok = bad_bar_prefix[ends + 1] - bad_bar_prefix[rows] == 0
        scale_ok = price_ok[rows] & future_bars_ok
        good = cadence_ok & roll_ok & scale_ok
        valid[rows, horizon_i] = good
        invalid_reason[rows, horizon_i] = INVALID_NONE
        invalid_reason[rows[~cadence_ok], horizon_i] = INVALID_CADENCE
        invalid_reason[rows[cadence_ok & ~roll_ok], horizon_i] = INVALID_CONTRACT_ROLL
        invalid_reason[rows[cadence_ok & roll_ok & ~scale_ok], horizon_i] = INVALID_SCALE_OR_PRICE
        good_rows = rows[good]
        if not len(good_rows):
            continue

        terminal_r = (c[ends] - c[rows]) / scale[rows]
        sum_change = change_prefix[ends] - change_prefix[rows]
        sum_change_sq = change_sq_prefix[ends] - change_sq_prefix[rows]
        mean_change = sum_change / float(steps)
        variance = np.maximum(
            sum_change_sq / float(steps) - mean_change * mean_change, 0.0,
        )
        path_length = abs_change_prefix[ends] - abs_change_prefix[rows]
        trend_eff = np.divide(
            np.abs(c[ends] - c[rows]), path_length,
            out=np.zeros_like(terminal_r), where=path_length > 0,
        )

        high_windows = np.lib.stride_tricks.sliding_window_view(h[1:], steps)
        low_windows = np.lib.stride_tricks.sliding_window_view(l[1:], steps)
        max_high = np.max(high_windows, axis=1)
        min_low = np.min(low_windows, axis=1)
        # Excursion is measured from the decision price with zero as an admitted
        # baseline.  A full future window may gap below (or above) the decision
        # price, but maximum favorable/adverse excursion cannot be negative.
        up_mfe = np.maximum(max_high - c[rows], 0.0) / scale[rows]
        down_mae = np.maximum(c[rows] - min_low, 0.0) / scale[rows]

        values = {
            "terminal_move_r": terminal_r,
            "forward_abs_move_r": np.abs(terminal_r),
            "forward_realized_vol": np.sqrt(variance) / scale[rows],
            "upside_mfe_r": up_mfe,
            "downside_mae_r": down_mae,
            "forward_trend_eff": trend_eff,
        }
        for name, value in values.items():
            result[name][good_rows, horizon_i] = value[good].astype(np.float32)

        active_context = good & (context_dir[rows] != 0)
        active_rows = rows[active_context]
        signed_terminal = context_dir[active_rows] * terminal_r[active_context]
        classes = np.full(len(active_rows), TREND_TERMINATION, dtype=np.int8)
        decisive = np.abs(terminal_r[active_context]) >= float(config.context_deadband_r)
        classes[decisive & (signed_terminal > 0)] = TREND_CONTINUATION
        classes[decisive & (signed_terminal < 0)] = TREND_REVERSAL
        trend_path_class[active_rows, horizon_i] = classes

        states, fav_time, adv_time, gross_r = _barrier_labels(
            high_windows, low_windows, good_rows, c, scale, terminal_r, targets,
            float(config.adverse_r), int(timeframe_minutes), int(config.barrier_chunk_rows),
        )
        barrier_state[good_rows, horizon_i] = states
        time_to_favorable_minutes[good_rows, horizon_i] = fav_time
        time_to_adverse_minutes[good_rows, horizon_i] = adv_time
        policy_r_gross[good_rows, horizon_i] = gross_r

    result.update({
        "schema_version": SCHEMA_VERSION,
        "target_semantics": {
            "price_change_basis": "raw_price_increment_over_decision_time_causal_atr_v1",
            "forward_realized_vol": "std_future_raw_price_increments_over_decision_time_causal_atr_v1",
            "forward_trend_eff": "absolute_terminal_price_change_over_path_absolute_price_change_v1",
            "horizon_basis": horizon_basis,
            "barrier_time_basis": "admitted_bar_minutes_v1",
            "negative_prices_supported": True,
        },
        "session_gap_capability": session_manifest,
        "config": asdict(config),
        "timeframe_minutes": int(timeframe_minutes),
        "horizons_minutes": horizons,
        "targets_r": targets,
        "directions": directions,
        "decision_time_ns": ts.copy(),
        "label_end_time_ns": label_end_time_ns,
        "causal_scale": scale.astype(np.float32),
        "context_direction": context_dir,
        "valid": valid,
        "invalid_reason": invalid_reason,
        "trend_path_class": trend_path_class,
        "barrier_state": barrier_state,
        "time_to_favorable_minutes": time_to_favorable_minutes,
        "time_to_adverse_minutes": time_to_adverse_minutes,
        "policy_r_gross": policy_r_gross,
    })
    return result


def path_label_fingerprint(labels: dict[str, object]) -> str:
    """Content fingerprint for deterministic dense-label artifacts."""
    digest = hashlib.sha256()
    scalar = {
        "schema_version": labels["schema_version"],
        "config": labels["config"],
        "timeframe_minutes": labels["timeframe_minutes"],
        "target_semantics": labels["target_semantics"],
        "session_gap_capability": labels.get("session_gap_capability"),
    }
    digest.update(json.dumps(scalar, sort_keys=True, separators=(",", ":")).encode())
    for key in sorted(labels):
        value = labels[key]
        if not isinstance(value, np.ndarray):
            continue
        array = np.ascontiguousarray(value)
        digest.update(key.encode())
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


__all__ = [
    "SCHEMA_VERSION", "PathLabelConfig", "build_dense_path_labels",
    "path_label_fingerprint", "BARRIER_NEITHER", "BARRIER_FAVORABLE_FIRST",
    "BARRIER_ADVERSE_FIRST", "BARRIER_AMBIGUOUS", "TREND_TERMINATION",
    "TREND_CONTINUATION", "TREND_REVERSAL", "TREND_UNDEFINED",
    "INVALID_NONE", "INVALID_INSUFFICIENT_FUTURE", "INVALID_CADENCE",
    "INVALID_CONTRACT_ROLL", "INVALID_SCALE_OR_PRICE",
]
