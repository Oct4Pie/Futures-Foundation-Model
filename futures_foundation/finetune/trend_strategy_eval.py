"""Matched, causal event-level ruler for trend strategy candidates.

This module deliberately evaluates trigger quality before fitting a foundation-model head.  Every
candidate shares entry timing, costs, horizon, same-bar policy, roll segmentation, and R accounting.
It is a development ruler, not a portfolio simulator: simultaneous trades across different streams
are allowed, while overlapping trades inside one strategy/stream are suppressed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from futures_foundation.execution_economics import (
    ExecutionEconomics, require_execution_economics,
)
from futures_foundation.pipeline._primitives import compute_atr, compute_supertrend
from futures_foundation.primitives.detection import (
    detect_atr_zigzag_pivots_v2,
    detect_fractal_pivots,
    detect_fractal_zigzag_pivots,
)
from futures_foundation.pivots import HTF_MAP, causal_htf_dir


@dataclass(frozen=True)
class RulerConfig:
    eval_start: str = "2024-07-01"
    eval_end: str = "2025-07-01"
    warmup_days: int = 180
    context: int = 256
    horizon_hours: float = 6.0
    atr_period: int = 20
    atr_stop: float = 0.5
    structural_buffer_atr: float = 0.05
    targets: tuple[float, ...] = (2.0, 3.0, 4.0, 6.0)
    primary_target: float = 3.0
    added_slippage_ticks_round_trip: float = 0.0
    same_bar_policy: str = "stop_first"

    def __post_init__(self):
        if pd.Timestamp(self.eval_end) <= pd.Timestamp(self.eval_start):
            raise ValueError("eval_end must follow eval_start")
        if self.warmup_days < 1 or self.context < 2 or self.horizon_hours <= 0:
            raise ValueError("warmup, context, and horizon must be positive")
        if (self.atr_stop <= 0 or self.structural_buffer_atr < 0
                or self.added_slippage_ticks_round_trip < 0):
            raise ValueError("risk and cost parameters must be nonnegative")
        if self.same_bar_policy != "stop_first":
            raise ValueError("matched ruler requires conservative stop_first handling")
        if not self.targets or tuple(sorted(self.targets)) != tuple(self.targets):
            raise ValueError("targets must be non-empty and sorted")
        if self.primary_target not in self.targets:
            raise ValueError("primary_target must be present in targets")


def timeframe_minutes(value: str) -> int:
    value = str(value)
    if not value.endswith("min"):
        raise ValueError(f"unsupported timeframe: {value}")
    minutes = int(value[:-3])
    if minutes < 1:
        raise ValueError("timeframe minutes must be positive")
    return minutes


def horizon_bars(timeframe: str, hours: float) -> int:
    return max(1, int(round(float(hours) * 60.0 / timeframe_minutes(timeframe))))


def executable_risk(raw_risk: float, tick_size: float) -> float:
    """Round a positive proposed stop distance outward to at least one exchange tick."""
    raw_risk, tick_size = float(raw_risk), float(tick_size)
    if not (np.isfinite(raw_risk) and raw_risk > 0 and np.isfinite(tick_size) and tick_size > 0):
        return float("nan")
    return float(max(1, int(np.ceil(raw_risk / tick_size - 1e-12))) * tick_size)


def load_stream(path: Path, eval_start: str, eval_end: str, *, warmup_days: int = 180,
                chunksize: int = 500_000) -> pd.DataFrame:
    """Read only the warmup+evaluation interval from one sorted continuous-futures CSV."""
    path = Path(path)
    columns = set(pd.read_csv(path, nrows=0).columns)
    contract_col = "contract_id" if "contract_id" in columns else (
        "instrument_id" if "instrument_id" in columns else None
    )
    if contract_col is None:
        raise ValueError(f"{path} lacks contract_id/instrument_id; roll-safe ruler refuses it")
    usecols = ["datetime", "open", "high", "low", "close", "volume", contract_col]
    lo = pd.Timestamp(eval_start, tz="UTC") - pd.Timedelta(days=int(warmup_days))
    hi = pd.Timestamp(eval_end, tz="UTC")
    pieces = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=int(chunksize)):
        ts = pd.to_datetime(chunk["datetime"], utc=True)
        keep = (ts >= lo) & (ts < hi)
        if keep.any():
            part = chunk.loc[keep].copy()
            part["datetime"] = ts[keep]
            part.rename(columns={contract_col: "contract_id"}, inplace=True)
            pieces.append(part)
    if not pieces:
        raise ValueError(f"{path} has no rows in requested interval")
    out = pd.concat(pieces, ignore_index=True)
    out.sort_values("datetime", inplace=True)
    out.drop_duplicates("datetime", keep="last", inplace=True)
    out.reset_index(drop=True, inplace=True)
    numeric = ["open", "high", "low", "close", "volume"]
    out[numeric] = out[numeric].apply(pd.to_numeric, errors="coerce")
    out.dropna(subset=["datetime", "open", "high", "low", "close"], inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out


def _trade_outcomes(o, h, l, c, *, signal_idx: int, direction: int, risk: float,
                    horizon: int, targets, cost_r: float):
    """Conservative first-touch outcomes for separate fixed-R exits.

    Entry is the next bar's open.  A bar touching both stop and target is a stop.  If neither is
    touched, the position is marked to the final horizon close.  Returns one outcome per target and
    the primary caller may choose any target's exit index for overlap suppression.
    """
    entry_idx = int(signal_idx) + 1
    n = len(c)
    if entry_idx >= n or entry_idx + int(horizon) > n:
        return None
    entry = float(o[entry_idx])
    risk = float(risk)
    if not (np.isfinite(entry) and np.isfinite(risk) and risk > 0):
        return None
    direction = 1 if int(direction) > 0 else -1
    targets = tuple(float(value) for value in targets)
    stop_at = None
    hit_at = [None] * len(targets)
    peak = 0.0
    end = entry_idx + int(horizon)
    for j in range(entry_idx, end):
        favorable = ((float(h[j]) - entry) / risk if direction == 1
                     else (entry - float(l[j])) / risk)
        adverse = ((float(l[j]) - entry) / risk if direction == 1
                   else (entry - float(h[j])) / risk)
        peak = max(peak, favorable)
        if adverse <= -1.0:  # conservative same-bar ordering
            stop_at = j
            break
        for ti, target in enumerate(targets):
            if hit_at[ti] is None and favorable >= target:
                hit_at[ti] = j
    final_r = direction * (float(c[end - 1]) - entry) / risk
    realized = np.empty(len(targets), np.float32)
    reached = np.zeros(len(targets), bool)
    exit_idx = np.empty(len(targets), np.int64)
    for ti, target in enumerate(targets):
        if hit_at[ti] is not None:
            reached[ti] = True
            realized[ti] = target - float(cost_r)
            exit_idx[ti] = int(hit_at[ti])
        elif stop_at is not None:
            realized[ti] = -1.0 - float(cost_r)
            exit_idx[ti] = int(stop_at)
        else:
            realized[ti] = final_r - float(cost_r)
            exit_idx[ti] = int(end - 1)
    return {
        "entry_idx": entry_idx, "label_end_idx": end - 1,
        "realized": realized, "reached": reached, "exit_idx": exit_idx,
        "peak_r": float(peak), "entry": entry, "risk": risk,
    }


def _detector_events(o, h, l, c, st_direction):
    st = []
    for i in range(1, len(st_direction) - 1):
        if int(st_direction[i]) != int(st_direction[i - 1]) and int(st_direction[i]) != 0:
            st.append({"confirm": i, "direction": int(st_direction[i]), "origin": i})
    return {
        "supertrend": st,
        "atr_zigzag_v2": detect_atr_zigzag_pivots_v2(
            o, h, l, c, atr_period=20, rev_atr=1.25,
        ),
        "fractal_k2": detect_fractal_pivots(h, l, k=2),
        "fractal_zigzag": detect_fractal_zigzag_pivots(
            o, h, l, c, k=2, min_leg_atr=1.25, atr_period=20,
        ),
    }


def _strategy_specs(has_htf: bool):
    specs = []
    for detector in ("supertrend", "atr_zigzag_v2", "fractal_k2", "fractal_zigzag"):
        specs.append((f"{detector}__atr", detector, "atr", False))
        if detector != "supertrend":
            specs.append((f"{detector}__structural", detector, "structural", False))
    if has_htf:
        for detector in ("supertrend", "fractal_zigzag"):
            specs.append((f"{detector}_htf__atr", detector, "atr", True))
            if detector != "supertrend":
                specs.append((f"{detector}_htf__structural", detector, "structural", True))
    return specs


def evaluate_stream(
    df: pd.DataFrame,
    ticker: str,
    timeframe: str,
    cfg: RulerConfig,
    execution_economics: ExecutionEconomics,
):
    """Evaluate every registered strategy on one stream, resetting at every contract roll."""
    execution_economics = require_execution_economics(execution_economics)
    execution_economics.assert_covers(
        pd.Timestamp(cfg.eval_start, tz="UTC").isoformat(),
        pd.Timestamp(cfg.eval_end, tz="UTC").isoformat(),
    )
    instrument = execution_economics.instrument(ticker)
    tick_size = instrument.tick_size
    slippage_ticks = execution_economics.validate_added_slippage(
        cfg.added_slippage_ticks_round_trip,
    )
    ts_all = pd.DatetimeIndex(df["datetime"])
    eval_lo = pd.Timestamp(cfg.eval_start, tz="UTC")
    eval_hi = pd.Timestamp(cfg.eval_end, tz="UTC")
    horizon = horizon_bars(timeframe, cfg.horizon_hours)
    primary_i = cfg.targets.index(cfg.primary_target)
    events_out = []
    contract = df["contract_id"].astype(str).to_numpy()
    segment_id = np.r_[0, np.cumsum(contract[1:] != contract[:-1])].astype(np.int64)
    for segment in np.unique(segment_id):
        rows = np.flatnonzero(segment_id == segment)
        if len(rows) < cfg.context + horizon + cfg.atr_period:
            continue
        part = df.iloc[rows]
        ts = pd.DatetimeIndex(part["datetime"])
        o, h, l, c = (part[name].to_numpy(float) for name in ("open", "high", "low", "close"))
        atr = compute_atr(h, l, c, cfg.atr_period)
        st_direction, _, _ = compute_supertrend(h, l, c, 10, 3.0)
        detectors = _detector_events(o, h, l, c, st_direction)
        htf = None
        if timeframe in HTF_MAP:
            htf = causal_htf_dir(
                {"ts": ts.tz_convert("UTC").tz_localize(None).to_numpy(),
                 "o": o, "h": h, "l": l, "c": c},
                timeframe, ts.tz_convert("UTC").tz_localize(None).to_numpy(), cfg.atr_period,
            )
        for strategy, detector, stop_mode, gated in _strategy_specs(htf is not None):
            active_until = -1
            for event in sorted(detectors[detector], key=lambda value: int(value["confirm"])):
                signal = int(event["confirm"])
                direction = int(event["direction"])
                if signal < cfg.context - 1 or signal + 1 <= active_until:
                    continue
                if signal + 1 + horizon > len(part):
                    continue
                signal_time = ts[signal]
                if not (eval_lo <= signal_time < eval_hi):
                    continue
                if ts[signal + horizon] >= eval_hi:
                    continue
                if gated and int(htf[signal]) != direction:
                    continue
                a = float(atr[signal])
                if not (np.isfinite(a) and a > 0):
                    continue
                entry = float(o[signal + 1])
                if stop_mode == "atr":
                    raw_risk = cfg.atr_stop * a
                else:
                    origin = int(event.get("origin", signal))
                    if not 0 <= origin <= signal:
                        continue
                    extreme = float(l[origin]) if direction == 1 else float(h[origin])
                    stop = (extreme - cfg.structural_buffer_atr * a if direction == 1
                            else extreme + cfg.structural_buffer_atr * a)
                    raw_risk = (entry - stop) if direction == 1 else (stop - entry)
                risk = executable_risk(raw_risk, tick_size)
                if not np.isfinite(risk):
                    continue
                risk_ticks = risk / tick_size
                one_tick_r = 1.0 / risk_ticks
                fee_r = instrument.fee_rt_usd / (risk_ticks * instrument.tick_value_usd)
                cost_r = fee_r + slippage_ticks * one_tick_r
                outcome = _trade_outcomes(
                    o, h, l, c, signal_idx=signal, direction=direction, risk=risk,
                    horizon=horizon, targets=cfg.targets, cost_r=cost_r,
                )
                if outcome is None:
                    continue
                primary_exit = int(outcome["exit_idx"][primary_i])
                active_until = primary_exit
                global_signal = int(rows[signal])
                events_out.append({
                    "strategy": strategy, "ticker": str(ticker), "timeframe": str(timeframe),
                    "signal_time_ns": int(ts[signal].value),
                    "entry_time_ns": int(ts[outcome["entry_idx"]].value),
                    "label_end_time_ns": int(ts[outcome["label_end_idx"]].value),
                    "exit_time_ns": int(ts[primary_exit].value),
                    "source_signal_idx": global_signal, "direction": direction,
                    "contract_id": str(contract[rows[signal]]), "stop_mode": stop_mode,
                    "raw_risk_price": float(raw_risk), "risk_price": float(outcome["risk"]),
                    "risk_ticks": float(risk_ticks), "one_tick_r": float(one_tick_r),
                    "fee_r": float(fee_r), "cost_r": float(cost_r),
                    "risk_atr": float(outcome["risk"] / a),
                    "peak_r": outcome["peak_r"],
                    "realized": outcome["realized"], "reached": outcome["reached"],
                })
    return events_out


def events_to_arrays(events, targets):
    if not events:
        raise ValueError("no strategy events")
    text_keys = ("strategy", "ticker", "timeframe", "contract_id", "stop_mode")
    int_keys = ("signal_time_ns", "entry_time_ns", "label_end_time_ns", "exit_time_ns",
                "source_signal_idx", "direction")
    out = {key: np.asarray([event[key] for event in events]) for key in text_keys + int_keys}
    out["risk_atr"] = np.asarray([event["risk_atr"] for event in events], np.float32)
    for key in ("raw_risk_price", "risk_price", "risk_ticks", "one_tick_r", "fee_r", "cost_r"):
        out[key] = np.asarray([event[key] for event in events], np.float32)
    out["peak_r"] = np.asarray([event["peak_r"] for event in events], np.float32)
    out["realized"] = np.stack([event["realized"] for event in events]).astype(np.float32)
    out["reached"] = np.stack([event["reached"] for event in events]).astype(bool)
    out["targets"] = np.asarray(targets, np.float32)
    order = np.argsort(out["signal_time_ns"], kind="stable")
    for key in tuple(out):
        if key != "targets":
            out[key] = out[key][order]
    return out


def _max_drawdown(values):
    equity = np.r_[0.0, np.cumsum(np.asarray(values, float))]
    return float(np.max(np.maximum.accumulate(equity) - equity))


def metric_summary(realized, reached):
    realized = np.asarray(realized, float)
    reached = np.asarray(reached, bool)
    if len(realized) == 0:
        return {"signals": 0, "wr": None, "mean_r": None, "median_r": None,
                "profit_factor": None, "total_r": None, "max_drawdown_r": None}
    wins = realized[realized > 0]
    losses = realized[realized < 0]
    gross_loss = -float(losses.sum())
    pf = float(wins.sum() / gross_loss) if gross_loss > 0 else None
    return {
        "signals": int(len(realized)), "wr": float(reached.mean()),
        "mean_r": float(realized.mean()), "median_r": float(np.median(realized)),
        "profit_factor": pf, "total_r": float(realized.sum()),
        "max_drawdown_r": _max_drawdown(realized),
        "average_win_r": float(wins.mean()) if len(wins) else None,
        "average_loss_r": float(losses.mean()) if len(losses) else None,
    }


def summarize_events(arrays, cfg: RulerConfig, *, folds: int = 6):
    primary_i = cfg.targets.index(cfg.primary_target)
    strategies = sorted(np.unique(arrays["strategy"]).tolist())
    start_ns = pd.Timestamp(cfg.eval_start, tz="UTC").value
    end_ns = pd.Timestamp(cfg.eval_end, tz="UTC").value
    edges = np.linspace(start_ns, end_ns, int(folds) + 1).astype(np.int64)
    report = {}
    for strategy in strategies:
        rows = arrays["strategy"] == strategy
        realized = arrays["realized"][rows, primary_i]
        reached = arrays["reached"][rows, primary_i]
        item = metric_summary(realized, reached)
        fold_rows = []
        for fold in range(int(folds)):
            selected = rows & (arrays["signal_time_ns"] >= edges[fold]) & (
                arrays["signal_time_ns"] < edges[fold + 1]
            )
            metrics = metric_summary(
                arrays["realized"][selected, primary_i], arrays["reached"][selected, primary_i],
            )
            metrics.update({"fold": fold + 1,
                            "start": pd.Timestamp(edges[fold], tz="UTC").isoformat(),
                            "end": pd.Timestamp(edges[fold + 1], tz="UTC").isoformat()})
            fold_rows.append(metrics)
        valid_means = [value["mean_r"] for value in fold_rows if value["mean_r"] is not None]
        item["folds"] = fold_rows
        item["positive_fold_fraction"] = float(np.mean(np.asarray(valid_means) > 0))
        item["worst_fold_mean_r"] = float(min(valid_means))
        item["by_timeframe"] = {}
        for timeframe in sorted(np.unique(arrays["timeframe"][rows]),
                                key=timeframe_minutes):
            selected = rows & (arrays["timeframe"] == timeframe)
            item["by_timeframe"][str(timeframe)] = metric_summary(
                arrays["realized"][selected, primary_i], arrays["reached"][selected, primary_i],
            )
        item["by_ticker"] = {}
        for ticker in sorted(np.unique(arrays["ticker"][rows])):
            selected = rows & (arrays["ticker"] == ticker)
            item["by_ticker"][str(ticker)] = metric_summary(
                arrays["realized"][selected, primary_i], arrays["reached"][selected, primary_i],
            )
        item["by_stream"] = {}
        stream_names = np.char.add(np.char.add(arrays["ticker"].astype(str), "@"),
                                   arrays["timeframe"].astype(str))
        for stream in sorted(np.unique(stream_names[rows])):
            selected = rows & (stream_names == stream)
            item["by_stream"][str(stream)] = metric_summary(
                arrays["realized"][selected, primary_i], arrays["reached"][selected, primary_i],
            )
        item["by_target"] = {}
        for target_i, target in enumerate(cfg.targets):
            item["by_target"][str(target)] = metric_summary(
                arrays["realized"][rows, target_i], arrays["reached"][rows, target_i],
            )
        item["cost_tick_sensitivity"] = {}
        gross = realized + arrays["cost_r"][rows]
        fee_r = arrays["fee_r"][rows]
        one_tick_r = arrays["one_tick_r"][rows]
        for cost_ticks in (0.0, 1.0, 2.0, 3.0):
            adjusted = gross - fee_r - one_tick_r * float(cost_ticks)
            item["cost_tick_sensitivity"][f"{cost_ticks:.1f}"] = metric_summary(
                adjusted, reached,
            )
        item["risk_ticks"] = {
            "median": float(np.median(arrays["risk_ticks"][rows])),
            "p10": float(np.quantile(arrays["risk_ticks"][rows], 0.10)),
            "p90": float(np.quantile(arrays["risk_ticks"][rows], 0.90)),
        }
        item["development_promising"] = bool(
            item["signals"] >= 500 and item["mean_r"] > 0.05
            and item["profit_factor"] is not None and item["profit_factor"] > 1.10
            and item["positive_fold_fraction"] >= 2 / 3 and item["worst_fold_mean_r"] > -0.10
        )
        report[strategy] = item
    return report
