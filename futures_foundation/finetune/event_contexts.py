"""Deduplicated, causal market-context shards for the frozen downstream gate."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from futures_foundation.finetune.path_labels import (
    BARRIER_ADVERSE_FIRST, BARRIER_AMBIGUOUS, BARRIER_FAVORABLE_FIRST,
    BARRIER_NEITHER, PathLabelConfig, build_dense_path_labels,
)
from futures_foundation.finetune.trend_strategy_eval import TICK_SIZES, executable_risk
from futures_foundation.pipeline._primitives import compute_atr, compute_supertrend
from futures_foundation.pivots import causal_htf_dir
from futures_foundation.primitives.detection import (
    detect_atr_zigzag_pivots_v2,
    detect_fractal_pivots,
    detect_fractal_zigzag_pivots,
)


SCHEMA_VERSION = "ffm_event_context_shard_v1"
TAG_NAMES = ("atr_zigzag_v2", "fractal_k2", "supertrend_flip", "fractal_zigzag")
BASELINE_LOOKBACKS = (4, 16, 64, 256)
PATH_ROW_KEYS = (
    "terminal_log_return", "terminal_move_r", "forward_abs_move_r",
    "forward_realized_vol", "upside_mfe_r", "downside_mae_r", "forward_trend_eff",
    "label_end_time_ns", "trend_path_class", "barrier_state",
    "time_to_favorable_minutes", "time_to_adverse_minutes", "policy_r_gross",
)


@dataclass(frozen=True)
class EventContextConfig:
    eval_start: str = "2024-07-01"
    eval_end: str = "2025-07-01"
    context_bars: int = 256
    atr_period: int = 20
    atr_stop: float = 0.5
    structural_buffer_atr: float = 0.05
    round_trip_cost_ticks: float = 1.0
    path: PathLabelConfig = field(default_factory=PathLabelConfig)

    def validate(self) -> None:
        start = pd.Timestamp(self.eval_start)
        end = pd.Timestamp(self.eval_end)
        if end <= start:
            raise ValueError("eval_end must follow eval_start")
        if self.context_bars < max(BASELINE_LOOKBACKS) or self.atr_period < 1:
            raise ValueError(
                f"context_bars must be >= {max(BASELINE_LOOKBACKS)} and atr_period positive"
            )
        if self.atr_stop <= 0 or self.structural_buffer_atr < 0:
            raise ValueError("risk parameters are invalid")
        if self.round_trip_cost_ticks < 0:
            raise ValueError("round_trip_cost_ticks must be nonnegative")


def _utc(value: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def _timeframe_minutes(timeframe: str) -> int:
    value = str(timeframe)
    if not value.endswith("min"):
        raise ValueError(f"unsupported timeframe: {timeframe}")
    minutes = int(value[:-3])
    if minutes < 1:
        raise ValueError("timeframe must be positive")
    return minutes


def _frame_arrays(frame: pd.DataFrame):
    required = {"datetime", "open", "high", "low", "close", "volume", "contract_id"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"event-context frame is missing columns: {missing}")
    ts = pd.DatetimeIndex(pd.to_datetime(frame["datetime"], utc=True))
    if len(ts) == 0 or np.any(np.diff(ts.asi8) <= 0):
        raise ValueError("event-context timestamps must be non-empty and strictly increasing")
    values = frame[["open", "high", "low", "close", "volume"]].to_numpy(np.float64)
    if not np.isfinite(values).all():
        raise ValueError("event-context OHLCV must be finite")
    o, h, l, c, v = values.T
    if np.any((h < np.maximum(o, c)) | (l > np.minimum(o, c)) | (h < l) | (v < 0)):
        raise ValueError("invalid OHLCV geometry")
    contract = frame["contract_id"].astype(str).to_numpy(dtype=str)
    if np.any(np.char.str_len(contract) == 0):
        raise ValueError("contract_id must be non-empty")
    segment = np.r_[0, np.cumsum(contract[1:] != contract[:-1])].astype(np.int64)
    source_row = (
        np.asarray(frame["source_row_idx"], np.int64)
        if "source_row_idx" in frame else np.arange(len(frame), dtype=np.int64)
    )
    if source_row.shape != (len(frame),) or np.any(np.diff(source_row) <= 0):
        raise ValueError("source_row_idx must be strictly increasing")
    return ts, o, h, l, c, v, contract, segment, source_row


def causal_baseline_features(
    frame: pd.DataFrame,
    *,
    causal_scale: np.ndarray,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Small context-only control using no bars before the declared 256-bar context."""
    _, o, h, l, c, v, _, _, _ = _frame_arrays(frame)
    n = len(c)
    scale = np.asarray(causal_scale, np.float64)
    if scale.shape != (n,):
        raise ValueError("causal_scale must align with frame")
    safe_scale = np.where(np.isfinite(scale) & (scale > 0), scale, np.nan)
    log_c = np.log(c)
    log_return = np.diff(log_c)
    ret_prefix = np.r_[0.0, np.cumsum(log_return)]
    ret_sq_prefix = np.r_[0.0, np.cumsum(log_return * log_return)]
    abs_ret_prefix = np.r_[0.0, np.cumsum(np.abs(log_return))]
    vol_prefix = np.r_[0.0, np.cumsum(v)]

    bar_range = h - l
    body = c - o
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - l
    safe_range = np.where(bar_range > 0, bar_range, np.nan)
    columns = [
        bar_range / safe_scale,
        body / safe_scale,
        upper / safe_scale,
        lower / safe_scale,
        np.divide(c - l, safe_range, out=np.full(n, 0.5), where=np.isfinite(safe_range)),
        safe_scale / c,
        np.log1p(v),
    ]
    names = [
        "bar_range_atr", "bar_body_atr", "upper_wick_atr", "lower_wick_atr",
        "close_position", "atr_fraction", "log1p_volume",
    ]

    for steps in BASELINE_LOOKBACKS:
        net = np.full(n, np.nan)
        realized = np.full(n, np.nan)
        efficiency = np.full(n, np.nan)
        range_atr = np.full(n, np.nan)
        volume_ratio = np.full(n, np.nan)
        rows = np.arange(steps, n)
        net[rows] = log_c[rows] - log_c[rows - steps]
        total = ret_prefix[rows] - ret_prefix[rows - steps]
        total_sq = ret_sq_prefix[rows] - ret_sq_prefix[rows - steps]
        mean = total / float(steps)
        realized[rows] = np.sqrt(np.maximum(total_sq / float(steps) - mean * mean, 0.0))
        path = abs_ret_prefix[rows] - abs_ret_prefix[rows - steps]
        efficiency[rows] = np.divide(
            np.abs(net[rows]), path, out=np.zeros(len(rows)), where=path > 0,
        )
        high_window = np.lib.stride_tricks.sliding_window_view(h, steps + 1)
        low_window = np.lib.stride_tricks.sliding_window_view(l, steps + 1)
        range_atr[rows] = (high_window.max(axis=1) - low_window.min(axis=1)) / safe_scale[rows]
        trailing_volume = vol_prefix[rows + 1] - vol_prefix[rows - steps + 1]
        volume_ratio[rows] = np.divide(
            v[rows] * float(steps), trailing_volume,
            out=np.zeros(len(rows)), where=trailing_volume > 0,
        )
        columns.extend((net, realized, efficiency, range_atr, volume_ratio))
        names.extend((
            f"log_return_{steps}bar", f"realized_vol_{steps}bar",
            f"trend_eff_{steps}bar", f"range_atr_{steps}bar", f"volume_ratio_{steps}bar",
        ))

    ts = pd.DatetimeIndex(pd.to_datetime(frame["datetime"], utc=True))
    minute_of_week = (ts.dayofweek * 1440 + ts.hour * 60 + ts.minute).to_numpy(float)
    phase = 2.0 * np.pi * minute_of_week / (7.0 * 1440.0)
    columns.extend((np.sin(phase), np.cos(phase)))
    names.extend(("minute_of_week_sin", "minute_of_week_cos"))
    return np.column_stack(columns).astype(np.float32), tuple(names)


def detect_context_tags(
    frame: pd.DataFrame,
    *,
    timeframe: str,
    atr_period: int = 20,
) -> dict[str, np.ndarray]:
    """Detect every causal trigger without one-active-trade suppression."""
    ts, o, h, l, c, _, _, segment, source_row = _frame_arrays(frame)
    n, tag_count = len(frame), len(TAG_NAMES)
    tags = np.zeros((n, tag_count), dtype=bool)
    direction = np.zeros((n, tag_count), dtype=np.int8)
    origin_source_idx = np.full((n, tag_count), -1, dtype=np.int64)
    htf_direction = np.zeros(n, dtype=np.int8)

    for segment_value in np.unique(segment):
        rows = np.flatnonzero(segment == segment_value)
        po, ph, pl, pc = o[rows], h[rows], l[rows], c[rows]
        st_direction, _, _ = compute_supertrend(ph, pl, pc, 10, 3.0)
        events = {
            "atr_zigzag_v2": detect_atr_zigzag_pivots_v2(
                po, ph, pl, pc, atr_period=atr_period, rev_atr=1.25,
            ),
            "fractal_k2": detect_fractal_pivots(ph, pl, k=2),
            "supertrend_flip": [
                {"confirm": i, "direction": int(st_direction[i]), "origin": i}
                for i in range(1, len(rows))
                if int(st_direction[i]) != int(st_direction[i - 1])
            ],
            "fractal_zigzag": detect_fractal_zigzag_pivots(
                po, ph, pl, pc, k=2, min_leg_atr=1.25, atr_period=atr_period,
            ),
        }
        local_ts = ts[rows].tz_convert("UTC").tz_localize(None).to_numpy()
        htf_direction[rows] = causal_htf_dir(
            {"ts": local_ts, "o": po, "h": ph, "l": pl, "c": pc},
            timeframe, local_ts, atr_period,
        )
        for tag_i, tag_name in enumerate(TAG_NAMES):
            for event in events[tag_name]:
                local = int(event["confirm"])
                origin = int(event.get("origin", local))
                if not (0 <= origin <= local < len(rows)):
                    raise ValueError(f"invalid {tag_name} event indices")
                global_row = int(rows[local])
                event_direction = 1 if int(event["direction"]) > 0 else -1
                if tags[global_row, tag_i] and direction[global_row, tag_i] != event_direction:
                    raise ValueError(f"conflicting {tag_name} events at row {global_row}")
                tags[global_row, tag_i] = True
                direction[global_row, tag_i] = event_direction
                origin_source_idx[global_row, tag_i] = int(source_row[rows[origin]])
    agreement = tags & (direction == htf_direction[:, None]) & (htf_direction[:, None] != 0)
    return {
        "tags": tags,
        "tag_direction": direction,
        "tag_origin_source_idx": origin_source_idx,
        "tag_htf_agreement": agreement,
        "htf_direction": htf_direction,
    }


def _block_weights(block_id: np.ndarray) -> np.ndarray:
    _, inverse, counts = np.unique(block_id, return_inverse=True, return_counts=True)
    weight = 1.0 / counts[inverse].astype(np.float64)
    return (weight / weight.mean()).astype(np.float32)


def _context_edge_is_valid(ts: pd.DatetimeIndex, expected_ns: int) -> np.ndarray:
    """Allow regular bars plus scheduled CME maintenance/weekend context edges.

    The exception applies to past context only. Future path labels retain exact cadence. Weekend
    gaps, arbitrary missing bars and gaps longer than two hours remain invalid.
    """
    delta = np.diff(ts.asi8)
    valid = delta == int(expected_ns)
    if len(delta) == 0:
        return valid
    local = ts.tz_convert("America/Chicago")
    left, right = local[:-1], local[1:]
    daily_maintenance = (
        (delta > int(expected_ns))
        & (delta <= 2 * 60 * 60 * 1_000_000_000)
        & (right.hour.to_numpy() == 17)
        & np.isin(left.hour.to_numpy(), (15, 16))
        & (right.minute.to_numpy() == 0)
    )
    weekend = (
        (delta >= 47 * 60 * 60 * 1_000_000_000)
        & (delta <= 52 * 60 * 60 * 1_000_000_000)
        & (left.dayofweek.to_numpy() == 4)
        & (right.dayofweek.to_numpy() == 6)
        & (right.hour.to_numpy() == 17)
    )
    return valid | daily_maintenance | weekend


def _single_policy_path(
    h: np.ndarray,
    l: np.ndarray,
    c: np.ndarray,
    *,
    decision: int,
    direction: int,
    entry: float,
    risk: float,
    steps: int,
    targets: np.ndarray,
    cost_r: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Conservative next-open execution labels with explicit OHLC ambiguity."""
    states = np.full(len(targets), BARRIER_NEITHER, dtype=np.int8)
    realized = np.empty(len(targets), dtype=np.float32)
    reached = np.zeros(len(targets), dtype=bool)
    exit_idx = np.full(len(targets), decision + steps, dtype=np.int64)
    adverse_at = 0
    favorable_at = np.zeros(len(targets), dtype=np.int64)
    direction = 1 if int(direction) > 0 else -1
    for j in range(decision + 1, decision + int(steps) + 1):
        favorable = (
            (float(h[j]) - entry) / risk if direction > 0
            else (entry - float(l[j])) / risk
        )
        adverse = (
            (entry - float(l[j])) / risk if direction > 0
            else (float(h[j]) - entry) / risk
        )
        if adverse_at == 0 and adverse >= 1.0:
            adverse_at = j
        for target_i, target in enumerate(targets):
            if favorable_at[target_i] == 0 and favorable >= float(target):
                favorable_at[target_i] = j

    terminal = direction * (float(c[decision + int(steps)]) - entry) / risk
    for target_i, target in enumerate(targets):
        fav_at = int(favorable_at[target_i])
        if fav_at and adverse_at and fav_at == adverse_at:
            states[target_i] = BARRIER_AMBIGUOUS
            realized[target_i] = -1.0 - float(cost_r)
            exit_idx[target_i] = adverse_at
        elif fav_at and (not adverse_at or fav_at < adverse_at):
            states[target_i] = BARRIER_FAVORABLE_FIRST
            realized[target_i] = float(target) - float(cost_r)
            reached[target_i] = True
            exit_idx[target_i] = fav_at
        elif adverse_at:
            states[target_i] = BARRIER_ADVERSE_FIRST
            realized[target_i] = -1.0 - float(cost_r)
            exit_idx[target_i] = adverse_at
        else:
            realized[target_i] = terminal - float(cost_r)
    return states, realized, reached, exit_idx


def event_policy_labels(
    frame: pd.DataFrame,
    *,
    ticker: str,
    selected: np.ndarray,
    selected_tags: np.ndarray,
    selected_tag_direction: np.ndarray,
    selected_tag_origin_source_idx: np.ndarray,
    causal_scale: np.ndarray,
    horizons_minutes: np.ndarray,
    targets_r: np.ndarray,
    timeframe_minutes: int,
    config: EventContextConfig,
) -> dict[str, np.ndarray]:
    """Attach ATR/structural policies to primary trigger tags without copying contexts."""
    if ticker not in TICK_SIZES:
        raise ValueError(f"missing explicit tick size for {ticker}")
    _, o, h, l, c, _, _, _, source_row = _frame_arrays(frame)
    source_to_local = {int(value): i for i, value in enumerate(source_row)}
    event_context, event_tag = np.nonzero(selected_tags[:, :3])
    event_count = len(event_context)
    modes = ("atr_stop", "structural_stop")
    horizon_count, target_count = len(horizons_minutes), len(targets_r)
    shape = (event_count, len(modes), horizon_count, target_count)
    valid = np.zeros((event_count, len(modes)), dtype=bool)
    risk_price = np.full((event_count, len(modes)), np.nan, dtype=np.float32)
    risk_ticks = np.full_like(risk_price, np.nan)
    cost_r = np.full_like(risk_price, np.nan)
    state = np.full(shape, -1, dtype=np.int8)
    realized = np.full(shape, np.nan, dtype=np.float32)
    reached = np.zeros(shape, dtype=bool)
    exit_time_ns = np.full(shape, -1, dtype=np.int64)
    tick = float(TICK_SIZES[ticker])
    ts_ns = pd.DatetimeIndex(pd.to_datetime(frame["datetime"], utc=True)).asi8
    event_decision = selected[event_context].astype(np.int64)
    event_entry = o[event_decision + 1].astype(np.float64)
    event_direction_values = selected_tag_direction[event_context, event_tag].astype(np.int8)

    for event_i, (context_row, tag_i) in enumerate(zip(event_context, event_tag)):
        decision = int(event_decision[event_i])
        direction = int(event_direction_values[event_i])
        origin_source = int(selected_tag_origin_source_idx[context_row, tag_i])
        origin = source_to_local.get(origin_source)
        if origin is None or not 0 <= origin <= decision or decision + 1 >= len(frame):
            continue
        entry = float(o[decision + 1])
        atr = float(causal_scale[decision])
        raw_risks = [float(config.atr_stop) * atr]
        structural_stop = (
            float(l[origin]) - float(config.structural_buffer_atr) * atr
            if direction > 0 else
            float(h[origin]) + float(config.structural_buffer_atr) * atr
        )
        raw_risks.append(
            entry - structural_stop if direction > 0 else structural_stop - entry
        )
        for mode_i, raw_risk in enumerate(raw_risks):
            risk = executable_risk(raw_risk, tick)
            if not np.isfinite(risk):
                continue
            valid[event_i, mode_i] = True
            risk_price[event_i, mode_i] = risk
            risk_ticks[event_i, mode_i] = risk / tick
            event_cost = float(config.round_trip_cost_ticks) * tick / risk
            cost_r[event_i, mode_i] = event_cost

    def first_touch(mask: np.ndarray) -> np.ndarray:
        touched = mask.any(axis=1)
        first = np.zeros(len(mask), dtype=np.int32)
        first[touched] = np.argmax(mask[touched], axis=1).astype(np.int32) + 1
        return first

    chunk_rows = int(config.path.barrier_chunk_rows)
    for mode_i in range(len(modes)):
        mode_events = np.flatnonzero(valid[:, mode_i])
        for horizon_i, horizon in enumerate(horizons_minutes):
            steps = int(horizon) // int(timeframe_minutes)
            high_windows = np.lib.stride_tricks.sliding_window_view(h[1:], steps)
            low_windows = np.lib.stride_tricks.sliding_window_view(l[1:], steps)
            for start in range(0, len(mode_events), chunk_rows):
                event_rows = mode_events[start:start + chunk_rows]
                decisions = event_decision[event_rows]
                hw = high_windows[decisions]
                lw = low_windows[decisions]
                entry = event_entry[event_rows, None]
                risk = risk_price[event_rows, mode_i, None].astype(np.float64)
                direction = event_direction_values[event_rows]
                long = direction > 0
                favorable_move = np.where(long[:, None], hw - entry, entry - lw) / risk
                adverse_move = np.where(long[:, None], entry - lw, hw - entry) / risk
                adverse_at = first_touch(adverse_move >= 1.0)
                terminal = direction * (c[decisions + steps] - event_entry[event_rows]) / risk[:, 0]
                event_cost = cost_r[event_rows, mode_i]

                for target_i, target in enumerate(targets_r):
                    favorable_at = first_touch(favorable_move >= float(target))
                    neither = (favorable_at == 0) & (adverse_at == 0)
                    favorable = (
                        (favorable_at > 0)
                        & ((adverse_at == 0) | (favorable_at < adverse_at))
                    )
                    adverse = (
                        (adverse_at > 0)
                        & ((favorable_at == 0) | (adverse_at < favorable_at))
                    )
                    ambiguous = (favorable_at > 0) & (favorable_at == adverse_at)
                    output_state = state[event_rows, mode_i, horizon_i, target_i]
                    output_state[neither] = BARRIER_NEITHER
                    output_state[favorable] = BARRIER_FAVORABLE_FIRST
                    output_state[adverse] = BARRIER_ADVERSE_FIRST
                    output_state[ambiguous] = BARRIER_AMBIGUOUS
                    state[event_rows, mode_i, horizon_i, target_i] = output_state
                    output_realized = realized[event_rows, mode_i, horizon_i, target_i]
                    output_realized[neither] = terminal[neither] - event_cost[neither]
                    output_realized[favorable] = float(target) - event_cost[favorable]
                    output_realized[adverse | ambiguous] = -1.0 - event_cost[adverse | ambiguous]
                    realized[event_rows, mode_i, horizon_i, target_i] = output_realized
                    reached[event_rows[favorable], mode_i, horizon_i, target_i] = True
                    exit_offset = np.full(len(event_rows), steps, dtype=np.int64)
                    exit_offset[favorable] = favorable_at[favorable]
                    exit_offset[adverse | ambiguous] = adverse_at[adverse | ambiguous]
                    exit_time_ns[event_rows, mode_i, horizon_i, target_i] = ts_ns[
                        decisions + exit_offset
                    ]

    return {
        "policy_mode_names": np.asarray(modes),
        "policy_event_context_row": event_context.astype(np.int64),
        "policy_event_tag_index": event_tag.astype(np.int8),
        "policy_event_direction": event_direction_values,
        "policy_valid": valid,
        "policy_risk_price": risk_price,
        "policy_risk_ticks": risk_ticks,
        "policy_cost_r": cost_r,
        "policy_barrier_state": state,
        "policy_realized_r": realized,
        "policy_reached": reached,
        "policy_exit_time_ns": exit_time_ns,
    }


def materialize_context_stream(
    frame: pd.DataFrame,
    *,
    ticker: str,
    timeframe: str,
    config: EventContextConfig | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Build one row per eligible decision context for one stream."""
    config = config or EventContextConfig()
    config.validate()
    minutes = _timeframe_minutes(timeframe)
    config.path.validate(minutes)
    ts, o, h, l, c, _, contract, segment, source_row = _frame_arrays(frame)
    atr = compute_atr(h, l, c, config.atr_period)
    labels = build_dense_path_labels(
        frame, timeframe_minutes=minutes, config=config.path, causal_scale=atr,
    )
    features, feature_names = causal_baseline_features(frame, causal_scale=atr)
    tag_data = detect_context_tags(frame, timeframe=timeframe, atr_period=config.atr_period)

    n = len(frame)
    context = int(config.context_bars)
    expected_ns = minutes * 60 * 1_000_000_000
    bad_gap_prefix = np.r_[
        0, np.cumsum(~_context_edge_is_valid(ts, expected_ns), dtype=np.int64)
    ]
    context_ok = np.zeros(n, dtype=bool)
    rows = np.arange(context - 1, n, dtype=np.int64)
    starts = rows - context + 1
    context_ok[rows] = (
        (bad_gap_prefix[rows] - bad_gap_prefix[starts] == 0)
        & (segment[rows] == segment[starts])
    )
    label_ok = np.asarray(labels["valid"]).all(axis=1)
    feature_ok = np.isfinite(features).all(axis=1)
    start_ns, end_ns = _utc(config.eval_start).value, _utc(config.eval_end).value
    interval_ok = (ts.asi8 >= start_ns) & (ts.asi8 < end_ns)
    label_within_interval = np.asarray(labels["label_end_time_ns"])[:, -1] < end_ns
    eligible = context_ok & label_ok & feature_ok & interval_ok & label_within_interval
    selected = np.flatnonzero(eligible)
    if not len(selected):
        raise ValueError(f"no eligible contexts for {ticker}@{timeframe}")

    local_in_segment = np.zeros(n, dtype=np.int64)
    for segment_value in np.unique(segment):
        segment_rows = np.flatnonzero(segment == segment_value)
        local_in_segment[segment_rows] = np.arange(len(segment_rows))
    block_id = (segment[selected] << np.int64(32)) + (
        local_in_segment[selected] // context
    )

    arrays: dict[str, np.ndarray] = {
        "ticker": np.full(len(selected), str(ticker)),
        "timeframe": np.full(len(selected), str(timeframe)),
        "contract_id": contract[selected],
        "context_start_source_idx": source_row[selected - context + 1],
        "decision_source_idx": source_row[selected],
        "decision_time_ns": ts.asi8[selected],
        "block_id": block_id,
        "sample_weight": _block_weights(block_id),
        "features": features[selected],
        "feature_names": np.asarray(feature_names),
        "tag_names": np.asarray(TAG_NAMES),
        "horizons_minutes": np.asarray(labels["horizons_minutes"], np.int32),
        "targets_r": np.asarray(labels["targets_r"], np.float32),
        "directions": np.asarray(labels["directions"], np.int8),
        "causal_scale": np.asarray(labels["causal_scale"])[selected],
        "context_direction": np.asarray(labels["context_direction"])[selected],
    }
    local_time = ts.tz_convert("America/Chicago")
    local_minute = local_time.hour.to_numpy() * 60 + local_time.minute.to_numpy()
    arrays["cme_session_minute"] = ((local_minute[selected] - 17 * 60) % 1440).astype(np.int16)
    arrays["contract_segment_id"] = segment[selected]
    arrays["bars_since_contract_start"] = local_in_segment[selected]
    for key in PATH_ROW_KEYS:
        arrays[key] = np.asarray(labels[key])[selected]
    for key, value in tag_data.items():
        arrays[key] = np.asarray(value)[selected]
    arrays.update(event_policy_labels(
        frame, ticker=ticker, selected=selected, selected_tags=arrays["tags"],
        selected_tag_direction=arrays["tag_direction"],
        selected_tag_origin_source_idx=arrays["tag_origin_source_idx"],
        causal_scale=atr, horizons_minutes=arrays["horizons_minutes"],
        targets_r=arrays["targets_r"], timeframe_minutes=minutes, config=config,
    ))

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "ticker": str(ticker),
        "timeframe": str(timeframe),
        "config": asdict(config),
        "rows": int(len(selected)),
        "source_rows": int(n),
        "event_rows": int(np.any(arrays["tags"], axis=1).sum()),
        "policy_events": int(len(arrays["policy_event_context_row"])),
        "tag_counts": {
            name: int(arrays["tags"][:, i].sum()) for i, name in enumerate(TAG_NAMES)
        },
        "split": {
            "eval_start": config.eval_start, "eval_end": config.eval_end,
            "oos_read": False,
        },
        "detectors": {
            "atr_zigzag": "prefix_invariant_v2", "fractal": "k2_confirmed",
            "supertrend": "10x3_flip", "fractal_zigzag": "k2_leg1.25_metadata",
        },
        "context_gap_policy": "exact_cadence_or_scheduled_cme_maintenance_or_weekend",
        "future_gap_policy": "exact_cadence_only_mask_never_truncate",
    }
    return arrays, metadata


def context_shard_fingerprint(arrays: dict[str, np.ndarray], metadata: dict[str, object]) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode())
    for key in sorted(arrays):
        value = np.ascontiguousarray(arrays[key])
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save_context_shard(
    path: str | Path,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, object],
    *,
    source: dict[str, object] | None = None,
) -> dict[str, object]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fingerprint = context_shard_fingerprint(arrays, metadata)
    temporary = Path(str(path) + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "content_fingerprint": fingerprint,
        "artifact": {"path": str(path.resolve()), "sha256": _sha256(path),
                     "bytes": path.stat().st_size},
        "source": dict(source or {}),
        "metadata": metadata,
    }
    manifest_path = Path(str(path) + ".manifest.json")
    manifest_temp = Path(str(manifest_path) + ".tmp")
    manifest_temp.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    os.replace(manifest_temp, manifest_path)
    return manifest


def load_context_shard(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    path = Path(path)
    manifest = json.loads(Path(str(path) + ".manifest.json").read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("status") != "complete":
        raise ValueError("unsupported or incomplete event-context shard")
    if _sha256(path) != manifest.get("artifact", {}).get("sha256"):
        raise ValueError("event-context artifact hash mismatch")
    with np.load(path, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    if context_shard_fingerprint(arrays, manifest["metadata"]) != manifest["content_fingerprint"]:
        raise ValueError("event-context content fingerprint mismatch")
    return arrays, manifest


__all__ = [
    "SCHEMA_VERSION", "TAG_NAMES", "BASELINE_LOOKBACKS", "EventContextConfig",
    "causal_baseline_features", "detect_context_tags", "materialize_context_stream",
    "event_policy_labels", "context_shard_fingerprint", "save_context_shard",
    "load_context_shard",
]
