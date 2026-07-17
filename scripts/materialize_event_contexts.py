#!/usr/bin/env python3
"""Build resumable, hash-bound dense context shards from the tournament cache."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

from futures_foundation.finetune.event_contexts import (
    EventContextConfig, load_context_shard, materialize_context_stream, save_context_shard,
)
from futures_foundation.finetune.path_labels import PathLabelConfig
from futures_foundation.finetune.tournament_data import CACHE_MANIFEST


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _load_stream(cache_dir: Path, entry: dict, *, start: pd.Timestamp, end: pd.Timestamp):
    files = entry["files"]
    loaded = {}
    for key in ("ohlcv", "timestamps", "contract_id"):
        if key not in files:
            raise ValueError(f"{entry['ticker']}@{entry['timeframe']} lacks {key}")
        path = cache_dir / files[key]["path"]
        actual = _sha256(path)
        if actual != files[key]["sha256"]:
            raise ValueError(f"source cache hash mismatch: {path}")
        loaded[key] = np.load(path, mmap_mode="r", allow_pickle=False)
    ts_ns = np.asarray(loaded["timestamps"], np.int64)
    keep = (ts_ns >= start.value) & (ts_ns < end.value)
    source_rows = np.flatnonzero(keep)
    if not len(source_rows):
        raise ValueError("stream has no rows in requested materialization interval")
    values = np.asarray(loaded["ohlcv"])[source_rows]
    return pd.DataFrame({
        "datetime": pd.to_datetime(ts_ns[source_rows], utc=True),
        "open": values[:, 0], "high": values[:, 1], "low": values[:, 2],
        "close": values[:, 3], "volume": values[:, 4],
        "contract_id": np.asarray(loaded["contract_id"])[source_rows].astype(str),
        "source_row_idx": source_rows,
    }), {
        key: {"path": str((cache_dir / files[key]["path"]).resolve()),
              "sha256": files[key]["sha256"]}
        for key in ("ohlcv", "timestamps", "contract_id")
    }


def _verified_source_files(cache_dir: Path, entry: dict) -> dict:
    files = entry["files"]
    verified = {}
    for key in ("ohlcv", "timestamps", "contract_id"):
        if key not in files:
            raise ValueError(f"{entry['ticker']}@{entry['timeframe']} lacks {key}")
        path = cache_dir / files[key]["path"]
        actual = _sha256(path)
        if actual != files[key]["sha256"]:
            raise ValueError(f"source cache hash mismatch: {path}")
        verified[key] = {"path": str(path.resolve()), "sha256": actual}
    return verified


def run(args) -> dict:
    cache_dir = Path(args.cache_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_manifest_path = cache_dir / CACHE_MANIFEST
    cache_manifest = json.loads(cache_manifest_path.read_text())
    if cache_manifest.get("schema_version") != "ffm_foundation_tournament_cache_v1":
        raise ValueError("unsupported tournament cache")
    if cache_manifest.get("interval", {}).get("contains_oos") is not False:
        raise ValueError("event contexts require a development-only source cache")

    tickers = tuple(value.strip().upper() for value in args.tickers.split(",") if value.strip())
    timeframes = tuple(value.strip() for value in args.timeframes.split(",") if value.strip())
    path_config = PathLabelConfig(
        horizons_minutes=tuple(int(value) for value in args.horizons.split(",")),
        targets_r=tuple(float(value) for value in args.targets.split(",")),
        adverse_r=args.adverse_r,
        atr_period=args.atr_period,
        context_minutes=args.context_minutes,
        context_deadband_r=args.context_deadband_r,
        barrier_chunk_rows=args.barrier_chunk_rows,
    )
    config = EventContextConfig(
        eval_start=args.eval_start, eval_end=args.eval_end, context_bars=args.context_bars,
        atr_period=args.atr_period, atr_stop=args.atr_stop,
        structural_buffer_atr=args.structural_buffer_atr,
        round_trip_cost_ticks=args.round_trip_cost_ticks, path=path_config,
    )
    config.validate()
    config_json = json.loads(json.dumps(asdict(config)))
    eval_start, eval_end = pd.Timestamp(args.eval_start, tz="UTC"), pd.Timestamp(args.eval_end, tz="UTC")
    load_start = eval_start - pd.Timedelta(days=int(args.warmup_days))

    shards = {}
    for ticker in tickers:
        for timeframe in timeframes:
            sid = f"{ticker}@{timeframe}"
            entry = cache_manifest.get("entries", {}).get(sid)
            if entry is None:
                raise KeyError(f"source cache is missing {sid}")
            output = output_dir / f"{ticker}_{timeframe}.npz"
            if output.exists() and not args.overwrite:
                _, existing = load_context_shard(output)
                if existing["metadata"].get("config") != config_json:
                    raise ValueError(f"existing shard config differs for {sid}; use --overwrite")
                current_source = _verified_source_files(cache_dir, entry)
                if existing.get("source", {}).get("files") != current_source:
                    raise ValueError(f"existing shard source differs for {sid}; use --overwrite")
                print(f"[resume] {sid}: verified {existing['metadata']['rows']:,} rows", flush=True)
                shards[sid] = existing
                continue
            started = time.perf_counter()
            frame, source_files = _load_stream(
                cache_dir, entry, start=load_start, end=eval_end,
            )
            arrays, metadata = materialize_context_stream(
                frame, ticker=ticker, timeframe=timeframe, config=config,
            )
            source = {
                "cache_manifest": str(cache_manifest_path),
                "cache_manifest_sha256": _sha256(cache_manifest_path),
                "stream": sid, "files": source_files,
                "loaded_start": load_start.isoformat(), "loaded_end": eval_end.isoformat(),
            }
            manifest = save_context_shard(output, arrays, metadata, source=source)
            manifest["elapsed_seconds"] = time.perf_counter() - started
            shards[sid] = manifest
            print(
                f"[done] {sid}: rows={metadata['rows']:,} events={metadata['event_rows']:,} "
                f"seconds={manifest['elapsed_seconds']:.2f} bytes={manifest['artifact']['bytes']:,}",
                flush=True,
            )

    summary = {
        "schema_version": "ffm_event_context_collection_v1",
        "status": "complete", "oos_read": False,
        "config": config_json,
        "cache_manifest": str(cache_manifest_path),
        "cache_manifest_sha256": _sha256(cache_manifest_path),
        "shards": {
            sid: {
                "path": manifest["artifact"]["path"],
                "sha256": manifest["artifact"]["sha256"],
                "content_fingerprint": manifest["content_fingerprint"],
                "rows": manifest["metadata"]["rows"],
                "event_rows": manifest["metadata"]["event_rows"],
                "policy_events": manifest["metadata"]["policy_events"],
                "tag_counts": manifest["metadata"]["tag_counts"],
                "bytes": manifest["artifact"]["bytes"],
            }
            for sid, manifest in sorted(shards.items())
        },
    }
    summary["totals"] = {
        "streams": len(summary["shards"]),
        "rows": int(sum(item["rows"] for item in summary["shards"].values())),
        "event_rows": int(sum(item["event_rows"] for item in summary["shards"].values())),
        "policy_events": int(sum(item["policy_events"] for item in summary["shards"].values())),
        "bytes": int(sum(item["bytes"] for item in summary["shards"].values())),
    }
    _atomic_json(output_dir / "MANIFEST.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default="output/foundation_tournament/data_cache")
    parser.add_argument("--output-dir", default="output/foundation_tournament/event_contexts_v1")
    parser.add_argument("--tickers", default="ES")
    parser.add_argument("--timeframes", default="5min")
    parser.add_argument("--eval-start", default="2024-07-01")
    parser.add_argument("--eval-end", default="2025-07-01")
    parser.add_argument("--warmup-days", type=int, default=30)
    parser.add_argument("--context-bars", type=int, default=256)
    parser.add_argument("--horizons", default="60,180,360")
    parser.add_argument("--targets", default="1,2,3")
    parser.add_argument("--adverse-r", type=float, default=1.0)
    parser.add_argument("--atr-period", type=int, default=20)
    parser.add_argument("--atr-stop", type=float, default=0.5)
    parser.add_argument("--structural-buffer-atr", type=float, default=0.05)
    parser.add_argument("--round-trip-cost-ticks", type=float, default=1.0)
    parser.add_argument("--context-minutes", type=int, default=60)
    parser.add_argument("--context-deadband-r", type=float, default=0.25)
    parser.add_argument("--barrier-chunk-rows", type=int, default=8192)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.warmup_days < 1:
        parser.error("--warmup-days must be positive")
    summary = run(args)
    print(json.dumps({"status": summary["status"], "shards": len(summary["shards"])}, indent=2))


if __name__ == "__main__":
    main()
