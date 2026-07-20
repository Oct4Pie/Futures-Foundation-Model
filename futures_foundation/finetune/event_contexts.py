"""Deduplicated, causal market-context shards for the frozen downstream gate."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from futures_foundation.corpus_v3_request_authority import (
    VerifiedRequestAuthorityV1,
    load_request_authority_manifest_v1,
    request_segment_ids_v1,
    require_request_authority_v1,
)
from futures_foundation.execution_economics import (
    ExecutionEconomics, load_execution_economics, require_execution_economics,
)
from futures_foundation.finetune.path_labels import (
    BARRIER_ADVERSE_FIRST, BARRIER_AMBIGUOUS, BARRIER_FAVORABLE_FIRST,
    BARRIER_NEITHER, PathLabelConfig, build_dense_path_labels,
)
from futures_foundation.finetune.trend_strategy_eval import executable_risk
from futures_foundation.pipeline._primitives import compute_atr, compute_supertrend
from futures_foundation.pivots import causal_htf_dir
from futures_foundation.session_gap import (
    VerifiedSessionGapCapability,
    advance_admitted_bars,
    load_session_gap_capability,
    require_session_gap_capability,
    verified_session_edge_mask,
)
from futures_foundation.primitives.detection import (
    detect_atr_zigzag_pivots_v2,
    detect_fractal_pivots,
    detect_fractal_zigzag_pivots,
)


SCHEMA_VERSION = "ffm_event_context_shard_v6"
COLLECTION_SCHEMA_VERSION = "ffm_event_context_collection_v5"
LEGACY_SCHEMA_VERSIONS = {
    "ffm_event_context_shard_v1", "ffm_event_context_shard_v2",
    "ffm_event_context_shard_v3", "ffm_event_context_shard_v4",
    "ffm_event_context_shard_v5",
}
TAG_NAMES = (
    "atr_zigzag_v2", "fractal_k2", "supertrend_flip", "fractal_zigzag",
    "pullback_continuation", "compression_breakout",
)
POLICY_TAG_NAMES = (
    "atr_zigzag_v2", "fractal_k2", "supertrend_flip",
    "pullback_continuation", "compression_breakout",
)
BASELINE_LOOKBACKS = (4, 16, 64, 256)
PATH_ROW_KEYS = (
    "terminal_move_r", "forward_abs_move_r",
    "forward_realized_vol", "upside_mfe_r", "downside_mae_r", "forward_trend_eff",
    "label_end_time_ns", "trend_path_class", "barrier_state",
    "time_to_favorable_minutes", "time_to_adverse_minutes", "policy_r_gross",
)
_EVENT_ARRAY_KEYS = {
    "ticker", "timeframe", "contract_id", "request_segment_id",
    "context_start_source_idx", "decision_source_idx", "decision_time_ns",
    "block_id", "sample_weight",
    "features", "feature_names", "tag_names", "horizons_minutes", "targets_r",
    "directions", "causal_scale", "context_direction", "contract_segment_id",
    "bars_since_contract_start", *PATH_ROW_KEYS, "tags", "tag_direction",
    "tag_origin_source_idx", "tag_htf_agreement", "htf_direction",
    "policy_mode_names", "policy_event_context_row", "policy_event_tag_index",
    "policy_event_direction", "policy_valid", "policy_risk_price",
    "policy_risk_ticks", "policy_barrier_state", "policy_gross_r",
    "policy_reached", "policy_exit_time_ns",
}
_EVENT_METADATA_KEYS = {
    "schema_version", "ticker", "timeframe", "config", "rows", "source_rows",
    "event_rows", "policy_events", "tag_counts", "split", "detectors",
    "context_gap_policy", "session_gap_capability", "future_gap_policy",
    "request_authority", "requested_use", "execution_economics",
}


@dataclass(frozen=True)
class EventContextConfig:
    eval_start: str = "2024-07-01"
    eval_end: str = "2025-07-01"
    context_bars: int = 256
    atr_period: int = 20
    atr_stop: float = 0.5
    structural_buffer_atr: float = 0.05
    pullback_fast: int = 20
    pullback_slow: int = 50
    pullback_trend_lookback: int = 64
    pullback_leg_bars: int = 8
    pullback_min_efficiency: float = 0.25
    pullback_min_trend_atr: float = 1.5
    pullback_min_depth_atr: float = 0.25
    pullback_max_depth_atr: float = 2.5
    compression_lookback: int = 20
    compression_max_range_atr: float = 4.0
    breakout_min_range_atr: float = 0.75
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
        if not (1 <= self.pullback_fast < self.pullback_slow):
            raise ValueError("pullback EMA periods must satisfy 1 <= fast < slow")
        if min(self.pullback_trend_lookback, self.pullback_leg_bars,
               self.compression_lookback) < 2:
            raise ValueError("event lookbacks must be at least two bars")
        if not (0 <= self.pullback_min_efficiency <= 1):
            raise ValueError("pullback_min_efficiency must be in [0, 1]")
        if not (0 <= self.pullback_min_depth_atr < self.pullback_max_depth_atr):
            raise ValueError("pullback depth bounds are invalid")
        if min(self.pullback_min_trend_atr, self.compression_max_range_atr,
               self.breakout_min_range_atr) <= 0:
            raise ValueError("event ATR thresholds must be positive")


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
    if frame["contract_id"].isna().any():
        raise ValueError("contract_id must be non-empty")
    contract = frame["contract_id"].astype(str).str.strip().to_numpy(dtype=str)
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


def _causal_ema(values: np.ndarray, period: int) -> np.ndarray:
    """Recursive EMA whose value at row i consumes rows no later than i."""
    values = np.asarray(values, np.float64)
    if values.ndim != 1 or not len(values) or int(period) < 1:
        raise ValueError("EMA input must be a non-empty vector and period positive")
    alpha = 2.0 / (int(period) + 1.0)
    output = np.empty_like(values)
    output[0] = values[0]
    for row in range(1, len(values)):
        output[row] = alpha * values[row] + (1.0 - alpha) * output[row - 1]
    return output


def _fixed_context_ema(values: np.ndarray, period: int, context: int) -> np.ndarray:
    """EMA reset at the start of each fixed trailing context, without an O(N*context) loop."""
    values = np.asarray(values, np.float64)
    context = int(context)
    if context < 1:
        raise ValueError("EMA context must be positive")
    output = np.full(len(values), np.nan, dtype=np.float64)
    if len(values) < context:
        return output
    global_ema = _causal_ema(values, int(period))
    rows = np.arange(context - 1, len(values), dtype=np.int64)
    starts = rows - context + 1
    output[rows] = global_ema[rows]
    later = starts > 0
    beta_power = (1.0 - 2.0 / (int(period) + 1.0)) ** context
    output[rows[later]] += beta_power * (
        values[starts[later]] - global_ema[starts[later] - 1]
    )
    return output


def _context_range_scale(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int,
) -> np.ndarray:
    """Finite-window mean true range; unlike Wilder ATR it has no pre-context state."""
    period = int(period)
    true_range = np.empty(len(close), dtype=np.float64)
    true_range[0] = high[0] - low[0]
    if len(close) > 1:
        true_range[1:] = np.maximum.reduce((
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1]),
        ))
    output = np.full(len(close), np.nan, dtype=np.float64)
    if len(close) >= period:
        prefix = np.r_[0.0, np.cumsum(true_range, dtype=np.float64)]
        rows = np.arange(period - 1, len(close), dtype=np.int64)
        output[rows] = (prefix[rows + 1] - prefix[rows + 1 - period]) / float(period)
    return output


def _detect_pullback_continuation(
    o: np.ndarray,
    h: np.ndarray,
    l: np.ndarray,
    c: np.ndarray,
    atr: np.ndarray,
    config: EventContextConfig,
) -> list[dict[str, int]]:
    """Past-trend, controlled-pullback and current-bar reclaim events.

    The established-trend and pullback tests end at i-1. Only the reclaim decision consumes bar i.
    The returned origin is the causal pullback extreme used by the structural-stop policy.
    """
    fast = _causal_ema(c, config.pullback_fast)
    slow = _causal_ema(c, config.pullback_slow)
    trend_n = int(config.pullback_trend_lookback)
    leg_n = int(config.pullback_leg_bars)
    slope_n = min(10, max(config.pullback_slow // 5, 2))
    events: list[dict[str, int]] = []
    first = max(trend_n + 1, config.pullback_slow + slope_n, leg_n + 1)
    for i in range(first, len(c)):
        scale = float(atr[i - 1])
        if not np.isfinite(scale) or scale <= 0:
            continue
        net = float(c[i - 1] - c[i - 1 - trend_n])
        path = float(np.abs(np.diff(c[i - 1 - trend_n:i])).sum())
        efficiency = abs(net) / path if path > 0 else 0.0
        if (efficiency < config.pullback_min_efficiency
                or abs(net) / scale < config.pullback_min_trend_atr):
            continue
        leg = slice(i - leg_n, i)
        if net > 0:
            trend = (fast[i - 1] > slow[i - 1]
                     and slow[i - 1] > slow[i - 1 - slope_n]
                     and c[i - 1] >= slow[i - 1] - 0.25 * scale)
            depth = (float(np.max(h[leg])) - float(c[i - 1])) / scale
            reclaim = c[i - 1] <= fast[i - 1] and c[i] > fast[i] and c[i] > o[i]
            if (trend and reclaim
                    and config.pullback_min_depth_atr <= depth <= config.pullback_max_depth_atr):
                origin = i - leg_n + int(np.argmin(l[leg]))
                events.append({"confirm": i, "direction": 1, "origin": origin})
        elif net < 0:
            trend = (fast[i - 1] < slow[i - 1]
                     and slow[i - 1] < slow[i - 1 - slope_n]
                     and c[i - 1] <= slow[i - 1] + 0.25 * scale)
            depth = (float(c[i - 1]) - float(np.min(l[leg]))) / scale
            reclaim = c[i - 1] >= fast[i - 1] and c[i] < fast[i] and c[i] < o[i]
            if (trend and reclaim
                    and config.pullback_min_depth_atr <= depth <= config.pullback_max_depth_atr):
                origin = i - leg_n + int(np.argmax(h[leg]))
                events.append({"confirm": i, "direction": -1, "origin": origin})
    return events


def _detect_compression_breakout(
    o: np.ndarray,
    h: np.ndarray,
    l: np.ndarray,
    c: np.ndarray,
    atr: np.ndarray,
    config: EventContextConfig,
) -> list[dict[str, int]]:
    """Close-confirmed break of a bounded, strictly prior compression range."""
    lookback = int(config.compression_lookback)
    events: list[dict[str, int]] = []
    for i in range(max(lookback, config.atr_period) + 1, len(c)):
        scale = float(atr[i - 1])
        if not np.isfinite(scale) or scale <= 0:
            continue
        prior = slice(i - lookback, i)
        upper, lower = float(np.max(h[prior])), float(np.min(l[prior]))
        if (upper - lower) / scale > config.compression_max_range_atr:
            continue
        true_range = max(float(h[i] - l[i]), abs(float(h[i] - c[i - 1])),
                         abs(float(l[i] - c[i - 1])))
        if true_range / scale < config.breakout_min_range_atr:
            continue
        if c[i] > upper and c[i] > o[i]:
            origin = i - lookback + int(np.argmin(l[prior]))
            events.append({"confirm": i, "direction": 1, "origin": origin})
        elif c[i] < lower and c[i] < o[i]:
            origin = i - lookback + int(np.argmax(h[prior]))
            events.append({"confirm": i, "direction": -1, "origin": origin})
    return events


def causal_baseline_features(
    frame: pd.DataFrame,
    *,
    event_config: EventContextConfig | None = None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Small context-only control using no bars before the declared 256-bar context."""
    event_config = event_config or EventContextConfig()
    event_config.validate()
    _, o, h, l, c, v, _, segment, _ = _frame_arrays(frame)
    n = len(c)
    # Stateful transforms must never inherit EMA/prefix/volume state from an expiring contract.
    # Segment recursively before computing any statistic; each recursive frame contains exactly
    # one contiguous contract and therefore reaches the scalar implementation below.
    if int(segment[-1]) > 0:
        output: np.ndarray | None = None
        feature_names: tuple[str, ...] | None = None
        for segment_value in np.unique(segment):
            rows = np.flatnonzero(segment == segment_value)
            local, local_names = causal_baseline_features(
                frame.iloc[rows].reset_index(drop=True),
                event_config=event_config,
            )
            if output is None:
                output = np.full((n, local.shape[1]), np.nan, dtype=np.float32)
                feature_names = local_names
            elif local_names != feature_names:
                raise RuntimeError("contract-segment feature schemas differ")
            output[rows] = local
        if output is None or feature_names is None:  # pragma: no cover - nonempty frame is enforced
            raise RuntimeError("contract segmentation produced no feature rows")
        return output, feature_names
    scale = _context_range_scale(h, l, c, event_config.atr_period)
    safe_scale = np.where(np.isfinite(scale) & (scale > 0), scale, np.nan)
    raw_change = np.diff(c)
    change_prefix = np.r_[0.0, np.cumsum(raw_change)]
    change_sq_prefix = np.r_[0.0, np.cumsum(raw_change * raw_change)]
    abs_change_prefix = np.r_[0.0, np.cumsum(np.abs(raw_change))]
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
        safe_scale / np.maximum(np.abs(c), safe_scale),
        np.log1p(v),
    ]
    names = [
        "bar_range_context_scale", "bar_body_context_scale",
        "upper_wick_context_scale", "lower_wick_context_scale",
        "close_position", "context_scale_to_abs_close", "log1p_volume",
    ]

    # Event-geometry controls. These are available at the decision close and give the classical
    # ruler the same causal setup geometry that an embedding could infer from its input window.
    fast_n, slow_n = event_config.pullback_fast, event_config.pullback_slow
    range_n = event_config.compression_lookback
    feature_context = max(BASELINE_LOOKBACKS)
    ema_fast = _fixed_context_ema(c, fast_n, feature_context)
    ema_slow = _fixed_context_ema(c, slow_n, feature_context)
    slow_slope = np.full(n, np.nan)
    if n >= feature_context:
        rows = np.arange(feature_context - 1, n, dtype=np.int64)
        starts = rows - feature_context + 1
        lag_rows = rows - 10
        global_slow = _causal_ema(c, slow_n)
        lagged = global_slow[lag_rows].copy()
        later = starts > 0
        beta_power = (1.0 - 2.0 / (slow_n + 1.0)) ** (feature_context - 10)
        lagged[later] += beta_power * (
            c[starts[later]] - global_slow[starts[later] - 1]
        )
        slow_slope[rows] = (ema_slow[rows] - lagged) / safe_scale[rows]
    prior_range = np.full(n, np.nan)
    break_above = np.full(n, np.nan)
    break_below = np.full(n, np.nan)
    if n > range_n:
        prior_high = np.lib.stride_tricks.sliding_window_view(h[:-1], range_n).max(axis=1)
        prior_low = np.lib.stride_tricks.sliding_window_view(l[:-1], range_n).min(axis=1)
        range_rows = np.arange(range_n, n)
        prior_range[range_rows] = (prior_high - prior_low) / safe_scale[range_rows]
        break_above[range_rows] = (c[range_rows] - prior_high) / safe_scale[range_rows]
        break_below[range_rows] = (prior_low - c[range_rows]) / safe_scale[range_rows]
    columns.extend((
        (c - ema_fast) / safe_scale,
        (ema_fast - ema_slow) / safe_scale,
        slow_slope,
        prior_range,
        break_above,
        break_below,
    ))
    names.extend((
        f"close_minus_ema{fast_n}_context_scale",
        f"ema{fast_n}_minus_ema{slow_n}_context_scale",
        f"ema{slow_n}_slope_10bar_context_scale",
        f"prior_range_{range_n}bar_context_scale",
        f"break_above_{range_n}bar_context_scale",
        f"break_below_{range_n}bar_context_scale",
    ))

    for steps in BASELINE_LOOKBACKS:
        # ``steps`` denotes bars, not return intervals. A 256-bar deployment context spans 255
        # close-to-close returns and begins at i-255; it must never read i-256.
        intervals = steps - 1
        net = np.full(n, np.nan)
        realized = np.full(n, np.nan)
        efficiency = np.full(n, np.nan)
        range_atr = np.full(n, np.nan)
        volume_ratio = np.full(n, np.nan)
        rows = np.arange(intervals, n)
        starts = rows - intervals
        net[rows] = (c[rows] - c[starts]) / safe_scale[rows]
        total = change_prefix[rows] - change_prefix[starts]
        total_sq = change_sq_prefix[rows] - change_sq_prefix[starts]
        mean = total / float(intervals)
        realized[rows] = np.sqrt(
            np.maximum(total_sq / float(intervals) - mean * mean, 0.0)
        ) / safe_scale[rows]
        path = abs_change_prefix[rows] - abs_change_prefix[starts]
        efficiency[rows] = np.divide(
            np.abs(c[rows] - c[starts]), path,
            out=np.zeros(len(rows)), where=path > 0,
        )
        high_window = np.lib.stride_tricks.sliding_window_view(h, steps)
        low_window = np.lib.stride_tricks.sliding_window_view(l, steps)
        range_atr[rows] = (high_window.max(axis=1) - low_window.min(axis=1)) / safe_scale[rows]
        trailing_volume = vol_prefix[rows + 1] - vol_prefix[starts]
        volume_ratio[rows] = np.divide(
            v[rows] * float(steps), trailing_volume,
            out=np.zeros(len(rows)), where=trailing_volume > 0,
        )
        columns.extend((net, realized, efficiency, range_atr, volume_ratio))
        names.extend((
            f"net_change_context_scale_{steps}bar",
            f"realized_change_vol_context_scale_{steps}bar",
            f"trend_eff_{steps}bar", f"range_context_scale_{steps}bar",
            f"volume_ratio_{steps}bar",
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
    config: EventContextConfig | None = None,
) -> dict[str, np.ndarray]:
    """Detect every causal trigger without one-active-trade suppression."""
    config = config or EventContextConfig(atr_period=int(atr_period))
    if int(atr_period) != int(config.atr_period):
        raise ValueError("atr_period and event config disagree")
    config.validate()
    ts, o, h, l, c, _, _, segment, source_row = _frame_arrays(frame)
    n, tag_count = len(frame), len(TAG_NAMES)
    tags = np.zeros((n, tag_count), dtype=bool)
    direction = np.zeros((n, tag_count), dtype=np.int8)
    origin_source_idx = np.full((n, tag_count), -1, dtype=np.int64)
    htf_direction = np.zeros(n, dtype=np.int8)

    for segment_value in np.unique(segment):
        rows = np.flatnonzero(segment == segment_value)
        po, ph, pl, pc = o[rows], h[rows], l[rows], c[rows]
        local_atr = compute_atr(ph, pl, pc, config.atr_period)
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
            "pullback_continuation": _detect_pullback_continuation(
                po, ph, pl, pc, local_atr, config,
            ),
            "compression_breakout": _detect_compression_breakout(
                po, ph, pl, pc, local_atr, config,
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


def _context_edge_is_valid(
    ts: pd.DatetimeIndex,
    expected_ns: int,
    session_gap_capability: VerifiedSessionGapCapability | None = None,
) -> np.ndarray:
    """Accept exact cadence or an independently verified adjacent session edge."""
    if session_gap_capability is None:
        return np.diff(ts.asi8) == int(expected_ns)
    return verified_session_edge_mask(
        ts,
        expected_delta=pd.Timedelta(int(expected_ns), unit="ns"),
        capability=session_gap_capability,
    )


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Conservative next-open gross-R labels with explicit OHLC ambiguity."""
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
            realized[target_i] = -1.0
            exit_idx[target_i] = adverse_at
        elif fav_at and (not adverse_at or fav_at < adverse_at):
            states[target_i] = BARRIER_FAVORABLE_FIRST
            realized[target_i] = float(target)
            reached[target_i] = True
            exit_idx[target_i] = fav_at
        elif adverse_at:
            states[target_i] = BARRIER_ADVERSE_FIRST
            realized[target_i] = -1.0
            exit_idx[target_i] = adverse_at
        else:
            realized[target_i] = terminal
    return states, realized, reached, exit_idx


def event_policy_labels(
    frame: pd.DataFrame,
    *,
    ticker: str,
    selected: np.ndarray,
    selected_tags: np.ndarray,
    selected_tag_direction: np.ndarray,
    selected_tag_origin_source_idx: np.ndarray,
    selected_tag_names: np.ndarray | tuple[str, ...] = TAG_NAMES,
    causal_scale: np.ndarray,
    horizons_minutes: np.ndarray,
    targets_r: np.ndarray,
    timeframe_minutes: int,
    config: EventContextConfig,
    execution_economics: ExecutionEconomics,
) -> dict[str, np.ndarray]:
    """Attach ATR/structural policies to primary trigger tags without copying contexts."""
    execution_economics = require_execution_economics(execution_economics)
    _, o, h, l, c, _, _, _, source_row = _frame_arrays(frame)
    source_to_local = {int(value): i for i, value in enumerate(source_row)}
    tag_names = tuple(str(value) for value in selected_tag_names)
    if selected_tags.shape[1] != len(tag_names):
        raise ValueError("selected tags and tag-name contract disagree")
    policy_tag_indices = np.asarray([
        tag_names.index(name) for name in POLICY_TAG_NAMES if name in tag_names
    ], dtype=np.int64)
    if not len(policy_tag_indices):
        raise ValueError("no policy-producing tag exists in the selected tag contract")
    local_context, local_tag = np.nonzero(selected_tags[:, policy_tag_indices])
    event_context = local_context
    event_tag = policy_tag_indices[local_tag]
    event_count = len(event_context)
    modes = ("atr_stop", "structural_stop")
    horizon_count, target_count = len(horizons_minutes), len(targets_r)
    shape = (event_count, len(modes), horizon_count, target_count)
    valid = np.zeros((event_count, len(modes)), dtype=bool)
    risk_price = np.full((event_count, len(modes)), np.nan, dtype=np.float32)
    risk_ticks = np.full_like(risk_price, np.nan)
    state = np.full(shape, -1, dtype=np.int8)
    realized = np.full(shape, np.nan, dtype=np.float32)
    reached = np.zeros(shape, dtype=bool)
    exit_time_ns = np.full(shape, -1, dtype=np.int64)
    tick = execution_economics.instrument(ticker).tick_size
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
                    output_realized[neither] = terminal[neither]
                    output_realized[favorable] = float(target)
                    output_realized[adverse | ambiguous] = -1.0
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
        "policy_barrier_state": state,
        "policy_gross_r": realized,
        "policy_reached": reached,
        "policy_exit_time_ns": exit_time_ns,
    }


def materialize_context_stream(
    frame: pd.DataFrame,
    *,
    ticker: str,
    timeframe: str,
    execution_economics: ExecutionEconomics,
    config: EventContextConfig | None = None,
    session_gap_capability: VerifiedSessionGapCapability | None = None,
    request_authority: VerifiedRequestAuthorityV1 | None = None,
    requested_use: str = "validation",
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Build one row per eligible decision context for one stream."""
    config = config or EventContextConfig()
    config.validate()
    execution_economics = require_execution_economics(execution_economics)
    execution_economics.assert_covers(
        _utc(config.eval_start).isoformat(), _utc(config.eval_end).isoformat(),
    )
    minutes = _timeframe_minutes(timeframe)
    config.path.validate(minutes)
    if session_gap_capability is not None:
        session_gap_capability = require_session_gap_capability(session_gap_capability)
        if session_gap_capability.root != str(ticker).strip().upper():
            raise ValueError("session-gap capability root differs from the event stream")
        if session_gap_capability.expected_delta_ns != minutes * 60 * 1_000_000_000:
            raise ValueError("session-gap capability bar size differs from the event stream")
    ts, o, h, l, c, _, contract, segment, source_row = _frame_arrays(frame)
    request_manifest = None
    requested_use_value = None
    request_segments = np.full(len(frame), -1, dtype=np.int64)
    if request_authority is not None:
        request_authority = require_request_authority_v1(request_authority)
        requested_use_value = str(requested_use).strip()
        if not requested_use_value:
            raise ValueError("requested_use must be non-empty when request authority is supplied")
        request_segments = request_segment_ids_v1(
            request_authority,
            root=str(ticker).strip().upper(),
            requested_use=requested_use_value,
            timestamps_ns=ts.asi8,
            contract_ids=contract,
        )
        request_manifest = request_authority.manifest()
    labels = build_dense_path_labels(
        frame,
        timeframe_minutes=minutes,
        config=config.path,
        session_gap_capability=session_gap_capability,
    )
    atr = np.asarray(labels["causal_scale"], dtype=np.float64)
    features, feature_names = causal_baseline_features(
        frame, event_config=config,
    )
    tag_data = detect_context_tags(
        frame, timeframe=timeframe, atr_period=config.atr_period, config=config,
    )

    n = len(frame)
    context = int(config.context_bars)
    expected_ns = minutes * 60 * 1_000_000_000
    bad_gap_prefix = np.r_[
        0, np.cumsum(
            ~_context_edge_is_valid(ts, expected_ns, session_gap_capability),
            dtype=np.int64,
        )
    ]
    if request_authority is None:
        bad_request_prefix = np.zeros(n, dtype=np.int64)
    else:
        bad_request_edge = (
            (request_segments[1:] != request_segments[:-1])
            | (request_segments[1:] < 0)
            | (request_segments[:-1] < 0)
        )
        bad_request_prefix = np.r_[
            0, np.cumsum(bad_request_edge, dtype=np.int64)
        ]
    context_ok = np.zeros(n, dtype=bool)
    rows = np.arange(context - 1, n, dtype=np.int64)
    starts = rows - context + 1
    context_ok[rows] = (
        (bad_gap_prefix[rows] - bad_gap_prefix[starts] == 0)
        & (segment[rows] == segment[starts])
    )
    if request_authority is not None:
        context_ok[rows] &= (
            (request_segments[rows] >= 0)
            & (bad_request_prefix[rows] - bad_request_prefix[starts] == 0)
        )
    label_valid = np.asarray(labels["valid"]).copy()
    if request_authority is not None:
        for horizon_i, horizon_minutes in enumerate(labels["horizons_minutes"]):
            steps = int(horizon_minutes) // minutes
            if n <= steps:
                continue
            decision_rows = np.arange(0, n - steps, dtype=np.int64)
            end_rows = decision_rows + steps
            label_valid[decision_rows, horizon_i] &= (
                (request_segments[decision_rows] >= 0)
                & (request_segments[end_rows] == request_segments[decision_rows])
                & (
                    bad_request_prefix[end_rows]
                    - bad_request_prefix[decision_rows]
                    == 0
                )
            )
    label_ok = label_valid.all(axis=1)
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
    block_segment = segment if request_authority is None else request_segments
    local_in_block_segment = np.zeros(n, dtype=np.int64)
    for segment_value in np.unique(block_segment[block_segment >= 0]):
        segment_rows = np.flatnonzero(block_segment == segment_value)
        local_in_block_segment[segment_rows] = np.arange(len(segment_rows))
    block_id = (block_segment[selected] << np.int64(32)) + (
        local_in_block_segment[selected] // context
    )

    arrays: dict[str, np.ndarray] = {
        "ticker": np.full(len(selected), str(ticker)),
        "timeframe": np.full(len(selected), str(timeframe)),
        "contract_id": contract[selected],
        "request_segment_id": request_segments[selected],
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
        selected_tag_names=arrays["tag_names"],
        causal_scale=atr, horizons_minutes=arrays["horizons_minutes"],
        targets_r=arrays["targets_r"], timeframe_minutes=minutes, config=config,
        execution_economics=execution_economics,
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
            "pullback_continuation": "ema20_50_reclaim_v1",
            "compression_breakout": "prior20_atr_bounded_close_break_v1",
        },
        "context_gap_policy": (
            "exact_timestamp_cadence_and_contract_segments_v1"
            if session_gap_capability is None
            else "verified_session_edges_and_contract_segments_v1"
        ),
        "session_gap_capability": (
            None if session_gap_capability is None else session_gap_capability.manifest()
        ),
        "future_gap_policy": (
            "exact_cadence_only_mask_never_truncate"
            if session_gap_capability is None
            else "verified_session_edges_mask_never_truncate_v1"
        ),
        "request_authority": request_manifest,
        "requested_use": requested_use_value,
        "execution_economics": execution_economics.manifest(),
    }
    return arrays, metadata


def validate_context_shard(
    arrays: Mapping[str, np.ndarray], metadata: Mapping[str, object],
) -> None:
    """Validate the complete current event-shard contract.

    A content hash proves only byte consistency.  This validator proves that those
    bytes have the exact shapes, dtypes, relationships, split semantics, event
    geometry, request authority, and authenticated economics required by
    ``ffm_event_context_shard_v6``.
    """
    if not isinstance(arrays, Mapping) or set(arrays) != _EVENT_ARRAY_KEYS:
        missing = sorted(_EVENT_ARRAY_KEYS - set(arrays)) if isinstance(arrays, Mapping) else []
        unknown = sorted(set(arrays) - _EVENT_ARRAY_KEYS) if isinstance(arrays, Mapping) else []
        raise ValueError(f"event-context array keys mismatch; missing={missing}, unknown={unknown}")
    if not isinstance(metadata, Mapping) or set(metadata) != _EVENT_METADATA_KEYS:
        missing = sorted(_EVENT_METADATA_KEYS - set(metadata)) if isinstance(metadata, Mapping) else []
        unknown = sorted(set(metadata) - _EVENT_METADATA_KEYS) if isinstance(metadata, Mapping) else []
        raise ValueError(f"event-context metadata keys mismatch; missing={missing}, unknown={unknown}")
    if metadata["schema_version"] != SCHEMA_VERSION:
        raise ValueError("event-context metadata schema mismatch")

    def count(value: object, field: str, *, minimum: int = 0) -> int:
        if type(value) is not int or value < minimum:
            raise ValueError(f"{field} must be an integer >= {minimum}")
        return value

    def array(
        name: str,
        *,
        dtype: str | np.dtype | None = None,
        kind: str | None = None,
        shape: tuple[int, ...] | None = None,
    ) -> np.ndarray:
        value = arrays[name]
        if not isinstance(value, np.ndarray) or value.dtype.hasobject:
            raise ValueError(f"event-context array {name} must be a non-object ndarray")
        if dtype is not None and value.dtype != np.dtype(dtype):
            raise ValueError(f"event-context array {name} has dtype {value.dtype}, expected {np.dtype(dtype)}")
        if kind is not None and value.dtype.kind != kind:
            raise ValueError(f"event-context array {name} must have dtype kind {kind}")
        if shape is not None and value.shape != shape:
            raise ValueError(f"event-context array {name} has shape {value.shape}, expected {shape}")
        return value

    rows = count(metadata["rows"], "metadata.rows", minimum=1)
    source_rows = count(metadata["source_rows"], "metadata.source_rows", minimum=rows)
    ticker = str(metadata["ticker"])
    timeframe = str(metadata["timeframe"])
    if not ticker or ticker != ticker.strip().upper():
        raise ValueError("event-context ticker must be non-empty uppercase text")
    minutes = _timeframe_minutes(timeframe)

    ticker_values = array("ticker", kind="U", shape=(rows,)).astype(str)
    timeframe_values = array("timeframe", kind="U", shape=(rows,)).astype(str)
    contract = array("contract_id", kind="U", shape=(rows,)).astype(str)
    request_segment = array("request_segment_id", dtype=np.int64, shape=(rows,))
    if not np.all(ticker_values == ticker) or not np.all(timeframe_values == timeframe):
        raise ValueError("event-context stream identity arrays disagree with metadata")
    if np.any(np.char.str_len(np.char.strip(contract)) == 0):
        raise ValueError("event-context contract_id contains blank values")
    request_manifest = metadata["request_authority"]
    requested_use = metadata["requested_use"]
    request_capability = None
    if request_manifest is None:
        if requested_use is not None or np.any(request_segment != -1):
            raise ValueError("event-context rows claim request membership without authority")
    else:
        if not isinstance(requested_use, str) or not requested_use.strip():
            raise ValueError("event-context requested_use must be non-empty")
        try:
            request_capability = load_request_authority_manifest_v1(request_manifest)
        except (TypeError, ValueError, FileNotFoundError) as exc:
            raise ValueError("event-context request authority cannot be reverified") from exc
        if np.any(request_segment < 0):
            raise ValueError("event-context authorized rows contain missing request membership")

    config_value = metadata["config"]
    if not isinstance(config_value, Mapping):
        raise ValueError("event-context config must be a mapping")
    try:
        raw_config = dict(config_value)
        raw_path = raw_config.pop("path")
        if not isinstance(raw_path, Mapping):
            raise TypeError("path config is not a mapping")
        path_config = PathLabelConfig(**dict(raw_path))
        config = EventContextConfig(path=path_config, **raw_config)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("event-context config does not match the current dataclass contract") from exc
    config.validate()
    config.path.validate(minutes)
    if int(config.context_bars) > source_rows:
        raise ValueError("event-context source row count cannot contain the declared context")

    split = metadata["split"]
    if not isinstance(split, Mapping) or set(split) != {"eval_start", "eval_end", "oos_read"}:
        raise ValueError("event-context split contract is malformed")
    if (
        split["eval_start"] != config.eval_start
        or split["eval_end"] != config.eval_end
        or split["oos_read"] is not False
    ):
        raise ValueError("event-context split contract differs from the materialization config")
    session_manifest = metadata["session_gap_capability"]
    session_capability = None
    if session_manifest is None:
        if metadata["context_gap_policy"] != "exact_timestamp_cadence_and_contract_segments_v1":
            raise ValueError("event-context exact-cadence context-gap policy is unsupported")
        if metadata["future_gap_policy"] != "exact_cadence_only_mask_never_truncate":
            raise ValueError("event-context exact-cadence future-gap policy is unsupported")
    else:
        if metadata["context_gap_policy"] != "verified_session_edges_and_contract_segments_v1":
            raise ValueError("event-context verified-session context-gap policy is unsupported")
        if metadata["future_gap_policy"] != "verified_session_edges_mask_never_truncate_v1":
            raise ValueError("event-context verified-session future-gap policy is unsupported")
        try:
            session_capability = load_session_gap_capability(session_manifest)
        except (TypeError, ValueError, FileNotFoundError) as exc:
            raise ValueError("event-context session-gap capability cannot be reverified") from exc
        if session_capability.root != ticker:
            raise ValueError("event-context session-gap root differs from the stream")
        if session_capability.expected_delta_ns != minutes * 60 * 1_000_000_000:
            raise ValueError("event-context session-gap bar size differs from the stream")

    feature_names = array("feature_names", kind="U")
    if feature_names.ndim != 1 or not len(feature_names):
        raise ValueError("event-context feature names must be a non-empty vector")
    if len(np.unique(feature_names.astype(str))) != len(feature_names):
        raise ValueError("event-context feature names must be unique")
    features = array("features", dtype=np.float32, shape=(rows, len(feature_names)))
    if not np.isfinite(features).all():
        raise ValueError("event-context features must be finite")

    tag_names = array("tag_names", kind="U", shape=(len(TAG_NAMES),)).astype(str)
    if tuple(tag_names.tolist()) != TAG_NAMES:
        raise ValueError("event-context tag-name contract differs from the canonical order")
    horizons = array("horizons_minutes", dtype=np.int32)
    targets = array("targets_r", dtype=np.float32)
    directions = array("directions", dtype=np.int8, shape=(2,))
    if horizons.ndim != 1 or not len(horizons) or np.any(horizons <= 0):
        raise ValueError("event-context horizons must be a positive vector")
    if targets.ndim != 1 or not len(targets) or not np.isfinite(targets).all() or np.any(targets <= 0):
        raise ValueError("event-context targets must be a positive finite vector")
    if not np.array_equal(horizons, np.asarray(config.path.horizons_minutes, np.int32)):
        raise ValueError("event-context horizons differ from the config")
    if not np.array_equal(targets, np.asarray(config.path.targets_r, np.float32)):
        raise ValueError("event-context targets differ from the config")
    if not np.array_equal(directions, np.asarray((1, -1), np.int8)):
        raise ValueError("event-context direction order is not canonical")
    horizon_count, target_count = len(horizons), len(targets)

    context_start = array("context_start_source_idx", dtype=np.int64, shape=(rows,))
    decision_source = array("decision_source_idx", dtype=np.int64, shape=(rows,))
    decision_time = array("decision_time_ns", dtype=np.int64, shape=(rows,))
    if (
        np.any(np.diff(decision_source) <= 0)
        or np.any(np.diff(decision_time) <= 0)
        or np.any(context_start < 0)
        or np.any(decision_source < context_start)
        or np.any(decision_source - context_start != int(config.context_bars) - 1)
    ):
        raise ValueError("event-context source/timestamp ordering or context geometry is invalid")
    eval_start_ns = _utc(config.eval_start).value
    eval_end_ns = _utc(config.eval_end).value
    if np.any(decision_time < eval_start_ns) or np.any(decision_time >= eval_end_ns):
        raise ValueError("event-context decision timestamps escape the declared split")
    if request_capability is not None:
        expected_request_segment = request_segment_ids_v1(
            request_capability,
            root=ticker,
            requested_use=requested_use,
            timestamps_ns=decision_time,
            contract_ids=contract,
        )
        if not np.array_equal(request_segment, expected_request_segment):
            raise ValueError("event-context decision request membership is stale or invalid")

    block_id = array("block_id", dtype=np.int64, shape=(rows,))
    sample_weight = array("sample_weight", dtype=np.float32, shape=(rows,))
    if not np.isfinite(sample_weight).all() or np.any(sample_weight <= 0):
        raise ValueError("event-context sample weights must be positive and finite")
    try:
        np.testing.assert_allclose(
            sample_weight, _block_weights(block_id), rtol=0.0, atol=1e-6,
            err_msg="event-context block weights are not canonical",
        )
    except AssertionError as exc:
        raise ValueError("event-context block weights are not canonical") from exc

    segment = array("contract_segment_id", dtype=np.int64, shape=(rows,))
    bars_since = array("bars_since_contract_start", dtype=np.int64, shape=(rows,))
    if np.any(segment < 0) or np.any(np.diff(segment) < 0) or np.any(bars_since < 0):
        raise ValueError("event-context contract-segment geometry is invalid")
    if rows > 1:
        segment_change = segment[1:] != segment[:-1]
        contract_change = contract[1:] != contract[:-1]
        if not np.array_equal(segment_change, contract_change):
            raise ValueError("event-context contract IDs and segment IDs disagree")
        same = ~segment_change
        if not np.array_equal(
            np.diff(bars_since)[same], np.diff(decision_source)[same]
        ):
            raise ValueError("event-context bars-since-contract geometry drifted")

    causal_scale = array("causal_scale", dtype=np.float32, shape=(rows,))
    context_direction = array("context_direction", dtype=np.int8, shape=(rows,))
    htf_direction = array("htf_direction", dtype=np.int8, shape=(rows,))
    if not np.isfinite(causal_scale).all() or np.any(causal_scale <= 0):
        raise ValueError("event-context causal scale must be positive and finite")
    if not np.isin(context_direction, (-1, 0, 1)).all() or not np.isin(htf_direction, (-1, 0, 1)).all():
        raise ValueError("event-context direction arrays contain invalid states")

    float_paths = (
        "terminal_move_r", "forward_abs_move_r", "forward_realized_vol",
        "upside_mfe_r", "downside_mae_r", "forward_trend_eff",
    )
    for name in float_paths:
        value = array(name, dtype=np.float32, shape=(rows, horizon_count))
        if not np.isfinite(value).all():
            raise ValueError(f"event-context path array {name} must be finite")
    if (
        np.any(arrays["forward_abs_move_r"] < 0)
        or np.any(arrays["forward_realized_vol"] < 0)
        or np.any(arrays["upside_mfe_r"] < 0)
        or np.any(arrays["downside_mae_r"] < 0)
        or np.any((arrays["forward_trend_eff"] < 0) | (arrays["forward_trend_eff"] > 1 + 1e-6))
    ):
        raise ValueError("event-context path magnitudes violate their declared domains")
    label_end = array("label_end_time_ns", dtype=np.int64, shape=(rows, horizon_count))
    if session_capability is None:
        expected_end = (
            decision_time[:, None]
            + horizons.astype(np.int64)[None, :] * 60_000_000_000
        )
    else:
        expected_end = np.column_stack([
            advance_admitted_bars(
                decision_time,
                np.full(rows, int(horizon) // minutes, dtype=np.int64),
                capability=session_capability,
            )
            for horizon in horizons
        ])
    if not np.array_equal(label_end, expected_end) or np.any(label_end >= eval_end_ns):
        raise ValueError("event-context label endpoints are incomplete or escape the split")
    if request_capability is not None:
        endpoint_segments = request_segment_ids_v1(
            request_capability,
            root=ticker,
            requested_use=requested_use,
            timestamps_ns=label_end.reshape(-1),
            contract_ids=np.repeat(contract, horizon_count),
        ).reshape(rows, horizon_count)
        if not np.array_equal(
            endpoint_segments,
            np.broadcast_to(request_segment[:, None], endpoint_segments.shape),
        ):
            raise ValueError("event-context label endpoints escape their planned request")
    trend_class = array("trend_path_class", dtype=np.int8, shape=(rows, horizon_count))
    if not np.isin(trend_class, (-1, 0, 1, 2)).all():
        raise ValueError("event-context trend-path class contains invalid states")
    path_shape = (rows, horizon_count, 2, target_count)
    barrier_state = array("barrier_state", dtype=np.int8, shape=path_shape)
    favorable_time = array("time_to_favorable_minutes", dtype=np.int32, shape=path_shape)
    adverse_time = array("time_to_adverse_minutes", dtype=np.int32, shape=path_shape)
    path_gross = array("policy_r_gross", dtype=np.float32, shape=path_shape)
    if not np.isin(barrier_state, (0, 1, 2, 3)).all() or not np.isfinite(path_gross).all():
        raise ValueError("event-context dense barrier labels are incomplete")
    horizon_limit = horizons[None, :, None, None]
    if np.any(favorable_time < -1) or np.any(adverse_time < -1):
        raise ValueError("event-context barrier times use an invalid negative sentinel")
    if np.any(favorable_time > horizon_limit) or np.any(adverse_time > horizon_limit):
        raise ValueError("event-context barrier times exceed their horizon")
    neither = barrier_state == BARRIER_NEITHER
    favorable = barrier_state == BARRIER_FAVORABLE_FIRST
    adverse = barrier_state == BARRIER_ADVERSE_FIRST
    ambiguous = barrier_state == BARRIER_AMBIGUOUS
    if (
        np.any(neither & ((favorable_time != -1) | (adverse_time != -1)))
        or np.any(favorable & ((favorable_time < 0) | ((adverse_time >= 0) & (favorable_time >= adverse_time))))
        or np.any(adverse & ((adverse_time < 0) | ((favorable_time >= 0) & (adverse_time >= favorable_time))))
        or np.any(ambiguous & ((favorable_time < 0) | (favorable_time != adverse_time)))
    ):
        raise ValueError("event-context barrier times disagree with first-touch states")

    tags = array("tags", dtype=np.bool_, shape=(rows, len(TAG_NAMES)))
    tag_direction = array("tag_direction", dtype=np.int8, shape=(rows, len(TAG_NAMES)))
    tag_origin = array("tag_origin_source_idx", dtype=np.int64, shape=(rows, len(TAG_NAMES)))
    tag_agreement = array("tag_htf_agreement", dtype=np.bool_, shape=(rows, len(TAG_NAMES)))
    if not np.isin(tag_direction, (-1, 0, 1)).all():
        raise ValueError("event-context tag directions contain invalid states")
    if np.any(tag_direction[~tags] != 0) or np.any(tag_direction[tags] == 0):
        raise ValueError("event-context tag presence and direction disagree")
    if np.any(tag_origin[~tags] != -1) or np.any(tag_origin[tags] < 0):
        raise ValueError("event-context tag origins are malformed")
    if np.any(tag_origin[tags] > np.broadcast_to(decision_source[:, None], tags.shape)[tags]):
        raise ValueError("event-context tag origin occurs after its decision")
    expected_agreement = tags & (tag_direction == htf_direction[:, None]) & (htf_direction[:, None] != 0)
    if not np.array_equal(tag_agreement, expected_agreement):
        raise ValueError("event-context HTF agreement is not derivable from tag directions")

    event_rows = int(np.any(tags, axis=1).sum())
    if count(metadata["event_rows"], "metadata.event_rows") != event_rows:
        raise ValueError("event-context event-row count mismatch")
    tag_counts = metadata["tag_counts"]
    expected_tag_counts = {name: int(tags[:, i].sum()) for i, name in enumerate(TAG_NAMES)}
    if not isinstance(tag_counts, Mapping) or dict(tag_counts) != expected_tag_counts:
        raise ValueError("event-context tag counts mismatch")

    modes = array("policy_mode_names", kind="U", shape=(2,)).astype(str)
    if tuple(modes.tolist()) != ("atr_stop", "structural_stop"):
        raise ValueError("event-context policy-mode order is not canonical")
    policy_rows = array("policy_event_context_row", dtype=np.int64)
    event_count = len(policy_rows)
    if policy_rows.shape != (event_count,):
        raise ValueError("event-context policy row vector is malformed")
    policy_tag = array("policy_event_tag_index", dtype=np.int8, shape=(event_count,))
    policy_direction = array("policy_event_direction", dtype=np.int8, shape=(event_count,))
    if (
        np.any(policy_rows < 0) or np.any(policy_rows >= rows)
        or np.any(policy_tag < 0) or np.any(policy_tag >= len(TAG_NAMES))
        or not np.isin(policy_direction, (-1, 1)).all()
    ):
        raise ValueError("event-context policy-event identity is invalid")
    if event_count:
        keys = np.column_stack((policy_rows, policy_tag))
        if len(np.unique(keys, axis=0)) != event_count:
            raise ValueError("event-context policy events are duplicated")
        policy_names = tag_names[policy_tag.astype(np.int64)]
        if any(name not in POLICY_TAG_NAMES for name in policy_names):
            raise ValueError("event-context policy event references a non-policy tag")
        if not tags[policy_rows, policy_tag].all():
            raise ValueError("event-context policy event references an absent tag")
        if not np.array_equal(policy_direction, tag_direction[policy_rows, policy_tag]):
            raise ValueError("event-context policy-event direction disagrees with its tag")
    if count(metadata["policy_events"], "metadata.policy_events") != event_count:
        raise ValueError("event-context policy-event count mismatch")

    policy_valid = array("policy_valid", dtype=np.bool_, shape=(event_count, 2))
    risk_price = array("policy_risk_price", dtype=np.float32, shape=(event_count, 2))
    risk_ticks = array("policy_risk_ticks", dtype=np.float32, shape=(event_count, 2))
    policy_shape = (event_count, 2, horizon_count, target_count)
    policy_state = array("policy_barrier_state", dtype=np.int8, shape=policy_shape)
    policy_gross = array("policy_gross_r", dtype=np.float32, shape=policy_shape)
    policy_reached = array("policy_reached", dtype=np.bool_, shape=policy_shape)
    policy_exit = array("policy_exit_time_ns", dtype=np.int64, shape=policy_shape)

    economics_manifest = metadata["execution_economics"]
    if not isinstance(economics_manifest, Mapping):
        raise ValueError("event-context execution economics must be a mapping")
    try:
        reopened_economics = load_execution_economics(
            str(economics_manifest["schedule_path"]),
            evaluation_start=str(economics_manifest["evaluation_start_utc"]),
            evaluation_end=str(economics_manifest["evaluation_end_exclusive_utc"]),
            required_roots=(ticker,),
        )
    except (KeyError, TypeError, ValueError, FileNotFoundError) as exc:
        raise ValueError("event-context execution economics cannot be reverified") from exc
    if reopened_economics.manifest() != dict(economics_manifest):
        raise ValueError("event-context execution economics differs from canonical re-verification")
    tick_size = reopened_economics.instrument(ticker).tick_size
    if (
        np.any(~np.isfinite(risk_price[policy_valid]))
        or np.any(risk_price[policy_valid] <= 0)
        or np.any(~np.isfinite(risk_ticks[policy_valid]))
        or np.any(risk_ticks[policy_valid] <= 0)
        or np.any(~np.isnan(risk_price[~policy_valid]))
        or np.any(~np.isnan(risk_ticks[~policy_valid]))
    ):
        raise ValueError("event-context policy risk validity mask is inconsistent")
    try:
        np.testing.assert_allclose(
            risk_ticks[policy_valid], risk_price[policy_valid] / tick_size,
            rtol=1e-6, atol=1e-6,
            err_msg="event-context policy risk/tick geometry differs",
        )
    except AssertionError as exc:
        raise ValueError("event-context policy risk/tick geometry differs") from exc
    valid_policy = np.broadcast_to(policy_valid[:, :, None, None], policy_shape)
    if (
        not np.isin(policy_state[valid_policy], (0, 1, 2, 3)).all()
        or np.any(~np.isfinite(policy_gross[valid_policy]))
        or np.any(policy_exit[valid_policy] < 0)
        or np.any(policy_state[~valid_policy] != -1)
        or np.any(~np.isnan(policy_gross[~valid_policy]))
        or np.any(policy_reached[~valid_policy])
        or np.any(policy_exit[~valid_policy] != -1)
    ):
        raise ValueError("event-context policy outcome mask is inconsistent")
    if np.any(policy_reached != (policy_state == BARRIER_FAVORABLE_FIRST)):
        raise ValueError("event-context policy reached flags disagree with barrier states")
    if event_count:
        decision_for_event = decision_time[policy_rows][:, None, None, None]
        end_for_event = label_end[policy_rows][:, None, :, None]
        if np.any(policy_exit[valid_policy] <= np.broadcast_to(decision_for_event, policy_shape)[valid_policy]):
            raise ValueError("event-context policy exit is not after the decision")
        if np.any(policy_exit[valid_policy] > np.broadcast_to(end_for_event, policy_shape)[valid_policy]):
            raise ValueError("event-context policy exit exceeds its label endpoint")

    detectors = metadata["detectors"]
    if not isinstance(detectors, Mapping) or set(detectors) != {
        "atr_zigzag", "fractal", "supertrend", "fractal_zigzag",
        "pullback_continuation", "compression_breakout",
    }:
        raise ValueError("event-context detector provenance is incomplete")


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
    validate_context_shard(arrays, metadata)
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


def load_context_shard(
    path: str | Path, *, allow_legacy: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    path = Path(path)
    manifest = json.loads(Path(str(path) + ".manifest.json").read_text())
    admitted = {SCHEMA_VERSION} | (LEGACY_SCHEMA_VERSIONS if allow_legacy else set())
    if (manifest.get("schema_version") not in admitted or manifest.get("status") != "complete"):
        raise ValueError("unsupported or incomplete event-context shard")
    if _sha256(path) != manifest.get("artifact", {}).get("sha256"):
        raise ValueError("event-context artifact hash mismatch")
    with np.load(path, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    if context_shard_fingerprint(arrays, manifest["metadata"]) != manifest["content_fingerprint"]:
        raise ValueError("event-context content fingerprint mismatch")
    if manifest["schema_version"] == SCHEMA_VERSION:
        validate_context_shard(arrays, manifest["metadata"])
    for value in arrays.values():
        value.setflags(write=False)
    return arrays, manifest


__all__ = [
    "SCHEMA_VERSION", "COLLECTION_SCHEMA_VERSION", "TAG_NAMES", "POLICY_TAG_NAMES",
    "BASELINE_LOOKBACKS",
    "EventContextConfig",
    "causal_baseline_features", "detect_context_tags", "materialize_context_stream",
    "event_policy_labels", "validate_context_shard", "context_shard_fingerprint",
    "save_context_shard", "load_context_shard",
]
