"""Shared leak-safe sampling primitives for foundation-model adaptation workers."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from . import ssl, ssl_data
from .tournament import OOS_START, TRAIN_START, VALIDATION_START


CACHE_MANIFEST = "TOURNAMENT_CACHE.json"


def _sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _atomic_npy(path, value):
    path = Path(path)
    tmp = Path(str(path) + ".tmp")
    with tmp.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
    os.replace(tmp, path)


def build_cache(source_dir, cache_dir, tickers, timeframes, *, verbose=True):
    """Materialize the train+validation date slice once as mmap-friendly arrays."""
    source_dir, cache_dir = Path(source_dir), Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    entries = {}
    for ticker in tickers:
        for timeframe in timeframes:
            streams = ssl_data.load_ohlcv(
                source_dir, (ticker,), (timeframe,), verbose=verbose,
                start=TRAIN_START, end=OOS_START,
            )
            if not streams:
                continue
            stream = streams[0]
            stem = f"{ticker}_{timeframe}"
            paths = {
                "ohlcv": cache_dir / f"{stem}.ohlcv.npy",
                "timestamps": cache_dir / f"{stem}.timestamps.npy",
            }
            _atomic_npy(paths["ohlcv"], stream["ohlcv"])
            timestamp_ns = pd.DatetimeIndex(stream["ts"]).asi8.astype(np.int64)
            _atomic_npy(paths["timestamps"], timestamp_ns)
            if stream.get("contract_id") is not None:
                paths["contract_id"] = cache_dir / f"{stem}.contract_id.npy"
                _atomic_npy(paths["contract_id"], np.asarray(stream["contract_id"], dtype=str))
            entries[f"{ticker}@{timeframe}"] = {
                "ticker": ticker, "timeframe": timeframe, "rows": int(len(timestamp_ns)),
                "files": {
                    key: {"path": path.name, "sha256": _sha256(path),
                          "bytes": path.stat().st_size}
                    for key, path in paths.items()
                },
            }
    source_manifest = source_dir / "MANIFEST.json"
    report = {
        "schema_version": "ffm_foundation_tournament_cache_v1",
        "interval": {"start": TRAIN_START, "end_exclusive": OOS_START,
                     "contains_oos": False},
        "source_dir": str(source_dir.resolve()),
        "source_manifest_sha256": (_sha256(source_manifest)
                                   if source_manifest.is_file() else None),
        "entries": entries,
    }
    target = cache_dir / CACHE_MANIFEST
    tmp = Path(str(target) + ".tmp")
    tmp.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(tmp, target)
    return report


def load_cache(cache_dir, tickers, timeframes, *, verbose=True):
    cache_dir = Path(cache_dir)
    manifest_path = cache_dir / CACHE_MANIFEST
    if not manifest_path.is_file():
        raise FileNotFoundError(f"tournament cache manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != "ffm_foundation_tournament_cache_v1":
        raise ValueError("unsupported tournament cache schema")
    interval = manifest.get("interval") or {}
    if (interval.get("start"), interval.get("end_exclusive"), interval.get("contains_oos")) != (
            TRAIN_START, OOS_START, False):
        raise ValueError("tournament cache interval does not match the locked protocol")
    streams = []
    for ticker in tickers:
        for timeframe in timeframes:
            entry = (manifest.get("entries") or {}).get(f"{ticker}@{timeframe}")
            if entry is None:
                if verbose:
                    print(f"  [tournament-cache] skip missing {ticker}@{timeframe}", flush=True)
                continue
            files = entry["files"]
            values = np.load(cache_dir / files["ohlcv"]["path"], mmap_mode="r",
                             allow_pickle=False)
            timestamp_ns = np.load(cache_dir / files["timestamps"]["path"], mmap_mode="r",
                                   allow_pickle=False)
            contract = (np.load(cache_dir / files["contract_id"]["path"], mmap_mode="r",
                                allow_pickle=False)
                        if "contract_id" in files else None)
            if values.shape != (int(entry["rows"]), 5) or len(timestamp_ns) != len(values):
                raise ValueError(f"invalid cached array shape for {ticker}@{timeframe}")
            streams.append({
                "sid": f"{ticker}@{timeframe}", "ticker": ticker, "tf": timeframe,
                "ohlcv": values, "ts": np.asarray(timestamp_ns).astype("datetime64[ns]"),
                "contract_id": contract,
            })
            if verbose:
                print(f"  [tournament-cache] {ticker}@{timeframe} bars={len(values)}", flush=True)
    if not streams:
        raise FileNotFoundError("no requested streams in tournament cache")
    return streams


def load_adaptation_data(data_dir, tickers, timeframes, *, parent_length, verbose=True):
    """Load the immutable train/validation universe without exposing OOS rows."""
    data_dir = Path(data_dir)
    streams = (load_cache(data_dir, tickers, timeframes, verbose=verbose)
               if (data_dir / CACHE_MANIFEST).is_file() else
               ssl_data.load_ohlcv(
                   data_dir, tickers, timeframes, verbose=verbose,
                   start=TRAIN_START, end=OOS_START,
               ))
    big, train, validation, groups = ssl.assemble(
        streams, seq=int(parent_length), max_jitter=0, val_frac=0.0,
        train_start=TRAIN_START, val_start=VALIDATION_START, holdout_start=OOS_START,
        return_groups=True, verbose=verbose, allow_aligned_market_gaps=True,
    )
    if len(groups["train_bounds"]) != len(streams) or len(groups["val_bounds"]) != len(streams):
        raise ValueError("every requested stream must have train and validation windows")
    return streams, big, train, validation, groups


def balanced_schedule(starts, bounds, examples, seed):
    """Sample exactly ``examples`` anchors with uniform stream probability."""
    starts = np.asarray(starts, np.int64)
    bounds = np.asarray(bounds, np.int64)
    examples = int(examples)
    if examples < 1 or bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError("examples must be positive and bounds must have shape [G,2]")
    if np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("every stream must contain eligible anchors")
    rng = np.random.default_rng(int(seed))
    group = rng.integers(0, len(bounds), size=examples)
    draw = rng.random(examples)
    lo, hi = bounds[group, 0], bounds[group, 1]
    row = lo + np.floor(draw * (hi - lo)).astype(np.int64)
    return starts[row], group.astype(np.int64)


def schedule_fingerprint(starts, groups):
    h = hashlib.sha256()
    for value in (np.asarray(starts, np.int64), np.asarray(groups, np.int64)):
        h.update(str(value.shape).encode())
        h.update(np.ascontiguousarray(value).view(np.uint8))
    return h.hexdigest()


def gather_contexts(big, starts, context, *, parent_length=None):
    """Gather a causal suffix from common parent anchors as ``[B,C,T]``."""
    big = np.asarray(big)
    starts = np.asarray(starts, np.int64)
    context = int(context)
    parent_length = int(parent_length or context)
    if context < 1 or context > parent_length:
        raise ValueError("context must lie in [1,parent_length]")
    shifted = starts + parent_length - context
    rows = shifted[:, None] + np.arange(context, dtype=np.int64)[None, :]
    return np.transpose(big[rows], (0, 2, 1)).astype(np.float32, copy=False)


def gather_parent(big, starts, length):
    """Gather exact common parent rows as ``[B,T,C]``."""
    starts = np.asarray(starts, np.int64)
    rows = starts[:, None] + np.arange(int(length), dtype=np.int64)[None, :]
    return np.asarray(big)[rows].astype(np.float32, copy=False)


def gather_time_features(starts, group_ids, streams, row_bounds, length):
    """Kronos minute/hour/weekday/day/month features aligned to common parent rows."""
    starts = np.asarray(starts, np.int64)
    group_ids = np.asarray(group_ids, np.int64)
    row_bounds = np.asarray(row_bounds, np.int64)
    if len(starts) != len(group_ids):
        raise ValueError("starts and group_ids must align")
    output = np.empty((len(starts), int(length), 5), np.float32)
    for group in np.unique(group_ids):
        rows = np.flatnonzero(group_ids == group)
        local = starts[rows] - int(row_bounds[group, 0])
        offsets = local[:, None] + np.arange(int(length), dtype=np.int64)[None, :]
        times = pd.DatetimeIndex(streams[int(group)]["ts"])
        if offsets.min() < 0 or offsets.max() >= len(times):
            raise ValueError(f"time-feature rows escape stream {group}")
        selected = pd.DatetimeIndex(times.asi8[offsets.reshape(-1)]).tz_localize("UTC")
        values = np.stack([
            selected.minute, selected.hour, selected.weekday, selected.day, selected.month,
        ], axis=1).astype(np.float32)
        output[rows] = values.reshape(len(rows), int(length), 5)
    return output
