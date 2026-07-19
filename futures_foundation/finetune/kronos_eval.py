"""Leak-audited data and metrics for the external Kronos forecast benchmark.

This module deliberately contains no torch or Kronos imports.  Corpus selection and
metric calculation remain unit-testable without downloading a model.  The executable
adapter lives in :mod:`scripts.benchmark_kronos`.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from . import ssl_data


OHLCV = ("open", "high", "low", "close", "volume")
KRONOS_PRETRAIN_END = pd.Timestamp("2024-06-30 23:59:59.999999999", tz="UTC")
EARLIEST_CLEAN_EVAL = pd.Timestamp("2024-07-01", tz="UTC")


def _utc(value) -> pd.Timestamp:
    out = pd.Timestamp(value)
    return out.tz_localize("UTC") if out.tzinfo is None else out.tz_convert("UTC")


def validate_eval_period(eval_start, eval_end) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Require a strict post-Kronos-pretraining evaluation interval."""
    start, end = _utc(eval_start), _utc(eval_end)
    if start < EARLIEST_CLEAN_EVAL:
        raise ValueError(
            f"Kronos evaluation must start on/after {EARLIEST_CLEAN_EVAL.date()}; "
            f"its published pretraining data extends through June 2024 (got {start})"
        )
    if end <= start:
        raise ValueError(f"eval_end must follow eval_start: {end} <= {start}")
    return start, end


def _read_slice(path: Path, lower: pd.Timestamp, upper: pd.Timestamp,
                chunksize: int = 250_000) -> pd.DataFrame:
    available = set(pd.read_csv(path, nrows=0).columns)
    required = {"datetime", *OHLCV, "contract_id"}
    missing = required - available
    if missing:
        raise ValueError(f"{path}: missing required columns {sorted(missing)}")
    pieces = []
    for chunk in pd.read_csv(path, usecols=["datetime", *OHLCV, "contract_id"],
                             chunksize=int(chunksize)):
        ts = pd.to_datetime(chunk.pop("datetime"), utc=True, errors="coerce")
        if ts.isna().any():
            raise ValueError(f"{path}: invalid timestamps")
        chunk.insert(0, "datetime", ts)
        keep = (ts >= lower) & (ts < upper)
        if keep.any():
            pieces.append(chunk.loc[keep].copy())
        if len(ts) and ts.iloc[-1] >= upper:
            break
    if not pieces:
        return pd.DataFrame(columns=["datetime", *OHLCV, "contract_id"])
    out = pd.concat(pieces, ignore_index=True).sort_values("datetime").reset_index(drop=True)
    if out["datetime"].duplicated().any():
        raise ValueError(f"{path}: duplicate timestamps in evaluation slice")
    values = out[list(OHLCV)].to_numpy(np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: non-finite OHLCV values")
    o, h, l, c, v = values.T
    invalid = (h < np.maximum(o, c)) | (l > np.minimum(o, c)) | (h < l) | (v < 0)
    if invalid.any():
        raise ValueError(f"{path}: {int(invalid.sum())} invalid OHLCV rows")
    contract_id = out["contract_id"]
    if contract_id.isna().any() or (contract_id.astype(str).str.len() == 0).any():
        raise ValueError(f"{path}: missing contract_id values")
    out["contract_id"] = contract_id.astype(str)
    return out


def _separated(starts: np.ndarray, separation: int) -> np.ndarray:
    """Greedy deterministic subset whose future-label starts cannot overlap."""
    starts = np.asarray(starts, np.int64)
    if not len(starts):
        return starts
    separation = max(1, int(separation))
    keep, next_ok = [], None
    for start in starts:
        if next_ok is None or int(start) >= next_ok:
            keep.append(int(start))
            next_ok = int(start) + separation
    return np.asarray(keep, np.int64)


def build_forecast_windows(data_dir, tickers, timeframes, *, context=512, horizon=16,
                           eval_start="2024-07-01", eval_end="2025-07-01",
                           max_per_stream=200, separation_bars=None, seed=0,
                           chunksize=250_000, verbose=True):
    """Build balanced causal contexts and immediately following forecast targets.

    Every ``[context, future]`` parent window stays within one contract and one accepted
    timestamp segment.  The target begins on or after ``eval_start`` and finishes strictly
    before ``eval_end``.  Sampling happens only after those conditions are established.
    """
    start_cut, end_cut = validate_eval_period(eval_start, eval_end)
    context, horizon = int(context), int(horizon)
    if context < 8 or horizon < 1:
        raise ValueError("context must be >= 8 and horizon must be positive")
    if max_per_stream is not None and int(max_per_stream) < 1:
        raise ValueError("max_per_stream must be positive or None")
    separation = int(separation_bars or horizon)
    rng = np.random.default_rng(int(seed))
    acc = {k: [] for k in ("context", "future", "context_time_ns", "future_time_ns",
                           "ticker", "timeframe", "source_start")}
    counts = {}
    data_dir = Path(data_dir)
    for ticker in tickers:
        for timeframe in timeframes:
            path = data_dir / f"{ticker}_{timeframe}.csv"
            if not path.is_file():
                if verbose:
                    print(f"[kronos-data] skip missing {ticker}@{timeframe}", flush=True)
                continue
            delta = pd.Timedelta(timeframe)
            # Extra seven days covers exchange closures around the longest requested context.
            lower = start_cut - context * delta - pd.Timedelta("7D")
            df = _read_slice(path, lower, end_cut, chunksize=chunksize)
            total = context + horizon
            if len(df) < total:
                if verbose:
                    print(f"[kronos-data] skip short {ticker}@{timeframe}", flush=True)
                continue
            ts = pd.DatetimeIndex(df["datetime"])
            segment = df["contract_id"].to_numpy(dtype=str)
            valid = ssl_data.window_starts(
                np.arange(len(df)), total, timestamps=ts, expected_delta=delta,
                segment_ids=segment,
            )
            target_start = valid + context
            target_end = valid + total - 1
            valid = valid[(ts[target_start] >= start_cut) & (ts[target_end] < end_cut)]
            # ``start`` and ``start + separation`` have target starts separated by the same
            # amount.  With separation >= horizon, future labels never share raw bars.
            valid = _separated(valid, separation)
            if max_per_stream is not None and len(valid) > int(max_per_stream):
                valid = np.sort(rng.choice(valid, int(max_per_stream), replace=False))
            if not len(valid):
                if verbose:
                    print(f"[kronos-data] skip empty {ticker}@{timeframe}", flush=True)
                continue
            values = df[list(OHLCV)].to_numpy(np.float32)
            time_ns = ts.asi8
            ci = valid[:, None] + np.arange(context)[None, :]
            fi = (valid + context)[:, None] + np.arange(horizon)[None, :]
            acc["context"].append(values[ci])
            acc["future"].append(values[fi])
            acc["context_time_ns"].append(time_ns[ci])
            acc["future_time_ns"].append(time_ns[fi])
            acc["ticker"].append(np.full(len(valid), ticker, dtype="U8"))
            acc["timeframe"].append(np.full(len(valid), timeframe, dtype="U8"))
            acc["source_start"].append(valid.astype(np.int64))
            counts[f"{ticker}@{timeframe}"] = int(len(valid))
            if verbose:
                print(f"[kronos-data] {ticker}@{timeframe}: {len(valid)} windows", flush=True)
    if not acc["context"]:
        raise ValueError("no valid Kronos evaluation windows")
    out = {key: np.concatenate(parts, axis=0) for key, parts in acc.items()}
    out["counts"] = counts
    out["eval_start"] = start_cut.isoformat()
    out["eval_end"] = end_cut.isoformat()
    out["context_length"] = context
    out["horizon"] = horizon
    out["separation_bars"] = separation
    if np.any(out["context_time_ns"][:, -1] >= out["future_time_ns"][:, 0]):
        raise RuntimeError("context/target timestamp overlap")
    return out


def window_fingerprint(windows) -> str:
    """Content hash binding cached forecasts to exact values, rows and timestamps."""
    h = hashlib.sha256()
    for key in ("context", "future", "context_time_ns", "future_time_ns",
                "ticker", "timeframe"):
        value = np.ascontiguousarray(windows[key])
        h.update(key.encode())
        h.update(str(value.dtype).encode())
        h.update(str(value.shape).encode())
        h.update(value.view(np.uint8))
    return h.hexdigest()


def _r2(actual, predicted):
    from sklearn.metrics import r2_score
    return float(r2_score(actual, predicted)) if len(actual) >= 2 else None


def _auc(actual_positive, score):
    from sklearn.metrics import roc_auc_score
    return (float(roc_auc_score(actual_positive, score))
            if len(np.unique(actual_positive)) == 2 else None)


def _corr(a, b, rank=False):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if rank:
        a = pd.Series(a).rank(method="average").to_numpy()
        b = pd.Series(b).rank(method="average").to_numpy()
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _trend_eff(path):
    step = np.diff(np.log(np.clip(path, 1e-9, None)), axis=1)
    return np.abs(step.sum(1)) / (np.abs(step).sum(1) + 1e-12)


def _mean_series_corr(actual, predicted, rank=False):
    """Kronos paper price metric: correlation over horizon, then sample/channel mean."""
    values = []
    for row in range(actual.shape[0]):
        for channel in range(actual.shape[2]):
            value = _corr(actual[row, :, channel], predicted[row, :, channel], rank=rank)
            if value is not None:
                values.append(value)
    return float(np.mean(values)) if values else None


def _core_metrics(context, future, prediction):
    context = np.asarray(context, np.float64)
    future = np.asarray(future, np.float64)
    prediction = np.asarray(prediction, np.float64)
    if prediction.shape[:2] != future.shape[:2] or prediction.shape[2] < 5:
        raise ValueError(f"prediction shape {prediction.shape} incompatible with {future.shape}")
    base = context[:, -1, 3]
    actual_close, pred_close = future[:, :, 3], prediction[:, :, 3]
    actual_path = np.log(np.clip(actual_close, 1e-9, None) / np.clip(base[:, None], 1e-9, None))
    pred_path = np.log(np.clip(pred_close, 1e-9, None) / np.clip(base[:, None], 1e-9, None))
    path_mse = float(np.mean((pred_path - actual_path) ** 2))
    persistence_mse = float(np.mean(actual_path ** 2))
    terminal, pred_terminal = actual_path[:, -1], pred_path[:, -1]
    actual_series = np.concatenate([base[:, None], actual_close], axis=1)
    pred_series = np.concatenate([base[:, None], pred_close], axis=1)
    # Match Kronos Appendix D: realized volatility is the sum of squared log returns.
    actual_rv = np.sum(np.diff(np.log(np.clip(actual_series, 1e-9, None)), axis=1) ** 2, axis=1)
    pred_rv = np.sum(np.diff(np.log(np.clip(pred_series, 1e-9, None)), axis=1) ** 2, axis=1)
    actual_eff, pred_eff = _trend_eff(actual_series), _trend_eff(pred_series)
    ref_n = min(future.shape[1], context.shape[1])
    ref = context[:, -ref_n:, :]
    ref_range = ref[:, :, 1].max(1) - ref[:, :, 2].min(1)
    actual_range = future[:, :, 1].max(1) - future[:, :, 2].min(1)
    pred_range = prediction[:, :, 1].max(1) - prediction[:, :, 2].min(1)
    actual_expand = np.log((actual_range + 1e-9) / (ref_range + 1e-9))
    pred_expand = np.log((np.clip(pred_range, 0, None) + 1e-9) / (ref_range + 1e-9))
    po, ph, pl, pc, pv = (prediction[:, :, i] for i in range(5))
    candle_valid = (ph >= np.maximum(po, pc)) & (pl <= np.minimum(po, pc)) & (ph >= pl)
    result = {
        "n": int(len(future)),
        "path_log_return_mse": path_mse,
        "persistence_path_mse": persistence_mse,
        "path_skill_vs_persistence": (1.0 - path_mse / persistence_mse
                                       if persistence_mse > 0 else None),
        "price_series_ic": _mean_series_corr(future[:, :, :4], prediction[:, :, :4]),
        "price_series_rank_ic": _mean_series_corr(
            future[:, :, :4], prediction[:, :, :4], rank=True),
        "terminal_return_ic": _corr(terminal, pred_terminal),
        "terminal_return_rank_ic": _corr(terminal, pred_terminal, rank=True),
        "fwd_dir_auc": _auc(terminal > 0, pred_terminal),
        "fwd_dir_accuracy": float(np.mean((terminal > 0) == (pred_terminal > 0))),
        "fwd_absmove_r2": _r2(np.abs(terminal), np.abs(pred_terminal)),
        "vol_r2": _r2(actual_rv, pred_rv),
        "vol_mae": float(np.mean(np.abs(actual_rv - pred_rv))),
        "trend_eff_r2": _r2(actual_eff, pred_eff),
        "range_expand_r2": _r2(actual_expand, pred_expand),
        "valid_candle_fraction": float(np.mean(candle_valid)),
        "nonnegative_volume_fraction": float(np.mean(pv >= 0)),
    }
    return result


def evaluate_predictions(windows, prediction):
    """Return overall, per-stream, and macro-stream scale-free forecast metrics."""
    prediction = np.asarray(prediction, np.float32)
    overall = _core_metrics(windows["context"], windows["future"], prediction)
    groups = {}
    labels = np.char.add(np.char.add(windows["ticker"].astype(str), "@"),
                         windows["timeframe"].astype(str))
    for label in sorted(np.unique(labels)):
        rows = labels == label
        groups[str(label)] = _core_metrics(windows["context"][rows], windows["future"][rows],
                                           prediction[rows])
    macro = {}
    for key in overall:
        if key == "n":
            continue
        values = [metric[key] for metric in groups.values() if metric[key] is not None]
        macro[key] = float(np.mean(values)) if values else None
    macro["stream_count"] = len(groups)
    return {"overall": overall, "macro_stream": macro, "per_stream": groups}
