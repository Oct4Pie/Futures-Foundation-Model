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

from futures_foundation.corpus_v3_request_authority import (
    load_and_verify_request_authority_v1,
)
from futures_foundation.execution_economics import load_execution_economics
from futures_foundation.finetune.event_contexts import (
    COLLECTION_SCHEMA_VERSION,
    EventContextConfig, load_context_shard, materialize_context_stream, save_context_shard,
)
from futures_foundation.finetune.path_labels import PathLabelConfig
from futures_foundation.finetune.tournament_data import (
    CACHE_MANIFEST,
    load_cache_entry,
    load_cache_manifest,
)
from futures_foundation.session_gap import load_session_gap_capability_set


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_arg(value: str) -> str:
    timestamp = pd.Timestamp(value)
    timestamp = timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")
    return timestamp.isoformat()


def _atomic_json(path: Path, value: dict) -> None:
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _load_stream(
    cache_dir: Path, cache_manifest: dict, entry: dict, *, start: pd.Timestamp, end: pd.Timestamp,
):
    stream, verified = load_cache_entry(
        cache_dir, cache_manifest, entry["ticker"], entry["timeframe"],
    )
    ts_ns = np.asarray(stream["ts"]).astype("datetime64[ns]").astype(np.int64)
    keep = (ts_ns >= start.value) & (ts_ns < end.value)
    source_rows = np.flatnonzero(keep)
    if not len(source_rows):
        raise ValueError("stream has no rows in requested materialization interval")
    values = np.asarray(stream["ohlcv"])[source_rows]
    return pd.DataFrame({
        "datetime": pd.to_datetime(ts_ns[source_rows], utc=True),
        "open": values[:, 0], "high": values[:, 1], "low": values[:, 2],
        "close": values[:, 3], "volume": values[:, 4],
        "contract_id": np.asarray(stream["contract_id"])[source_rows].astype(str),
        "source_row_idx": source_rows,
    }), verified


def run(args) -> dict:
    cache_dir = Path(args.cache_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_manifest_path = cache_dir / CACHE_MANIFEST
    cache_manifest = load_cache_manifest(
        cache_dir, expected_manifest_sha256=args.cache_manifest_sha256,
    )

    tickers = tuple(value.strip().upper() for value in args.tickers.split(",") if value.strip())
    timeframes = tuple(value.strip() for value in args.timeframes.split(",") if value.strip())
    if bool(args.session_gap_capability_set) != bool(args.session_gap_capability_set_sha256):
        raise ValueError(
            "session-gap capability set path and SHA-256 must be supplied together"
        )
    session_capabilities = (
        load_session_gap_capability_set(
            args.session_gap_capability_set,
            expected_sha256=args.session_gap_capability_set_sha256,
        )
        if args.session_gap_capability_set else {}
    )
    request_values = (
        args.request_expected,
        args.request_expected_sha256,
        args.request_inventory,
        args.request_inventory_sha256,
        args.request_plan,
        args.request_plan_sha256,
    )
    if any(request_values) and not all(request_values):
        raise ValueError(
            "request expected/inventory/plan paths and SHA-256 values must all be supplied"
        )
    request_authority = (
        load_and_verify_request_authority_v1(
            expected_path=args.request_expected,
            expected_physical_sha256=args.request_expected_sha256,
            inventory_path=args.request_inventory,
            inventory_physical_sha256=args.request_inventory_sha256,
            plan_path=args.request_plan,
            plan_physical_sha256=args.request_plan_sha256,
        )
        if all(request_values) else None
    )
    request_manifest = None if request_authority is None else request_authority.manifest()
    requested_streams = {
        f"{ticker}@{timeframe}" for ticker in tickers for timeframe in timeframes
    }
    if session_capabilities and set(session_capabilities) != requested_streams:
        raise ValueError(
            "session-gap capability set must exactly cover requested streams: "
            f"missing={sorted(requested_streams - set(session_capabilities))}, "
            f"unknown={sorted(set(session_capabilities) - requested_streams)}"
        )
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
        structural_buffer_atr=args.structural_buffer_atr, path=path_config,
    )
    config.validate()
    economics = load_execution_economics(
        args.execution_costs,
        evaluation_start=_utc_arg(args.eval_start),
        evaluation_end=_utc_arg(args.eval_end),
        required_roots=tickers,
    )
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
            capability = session_capabilities.get(sid)
            if output.exists() and not args.overwrite:
                _, existing = load_context_shard(output)
                if existing["metadata"].get("config") != config_json:
                    raise ValueError(f"existing shard config differs for {sid}; use --overwrite")
                expected_session = None if capability is None else capability.manifest()
                if existing["metadata"].get("session_gap_capability") != expected_session:
                    raise ValueError(
                        f"existing shard session authority differs for {sid}; use --overwrite"
                    )
                if (
                    existing["metadata"].get("request_authority") != request_manifest
                    or existing["metadata"].get("requested_use")
                    != (None if request_authority is None else args.requested_use)
                ):
                    raise ValueError(
                        f"existing shard request authority differs for {sid}; use --overwrite"
                    )
                _, current_source = load_cache_entry(
                    cache_dir, cache_manifest, ticker, timeframe,
                )
                if existing.get("source", {}).get("files") != current_source:
                    raise ValueError(f"existing shard source differs for {sid}; use --overwrite")
                print(f"[resume] {sid}: verified {existing['metadata']['rows']:,} rows", flush=True)
                shards[sid] = existing
                continue
            started = time.perf_counter()
            frame, source_files = _load_stream(
                cache_dir, cache_manifest, entry, start=load_start, end=eval_end,
            )
            arrays, metadata = materialize_context_stream(
                frame, ticker=ticker, timeframe=timeframe, config=config,
                execution_economics=economics,
                session_gap_capability=capability,
                request_authority=request_authority,
                requested_use=args.requested_use,
            )
            source = {
                "cache_manifest": str(cache_manifest_path),
                "cache_manifest_sha256": _sha256(cache_manifest_path),
                "stream": sid, "files": source_files,
                "loaded_start": load_start.isoformat(), "loaded_end": eval_end.isoformat(),
                "session_gap_capability_set": (
                    None if not args.session_gap_capability_set else {
                        "path": str(Path(args.session_gap_capability_set).resolve()),
                        "sha256": args.session_gap_capability_set_sha256,
                    }
                ),
                "request_authority": request_manifest,
                "requested_use": (
                    None if request_authority is None else args.requested_use
                ),
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
        "schema_version": COLLECTION_SCHEMA_VERSION,
        "status": "complete", "oos_read": False,
        "config": config_json,
        "cache_manifest": str(cache_manifest_path),
        "cache_manifest_sha256": _sha256(cache_manifest_path),
        "session_gap_capability_set": (
            None if not args.session_gap_capability_set else {
                "path": str(Path(args.session_gap_capability_set).resolve()),
                "sha256": args.session_gap_capability_set_sha256,
            }
        ),
        "request_authority": request_manifest,
        "requested_use": None if request_authority is None else args.requested_use,
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
    parser.add_argument("--cache-manifest-sha256", required=True)
    parser.add_argument("--session-gap-capability-set")
    parser.add_argument("--session-gap-capability-set-sha256")
    parser.add_argument("--request-expected")
    parser.add_argument("--request-expected-sha256")
    parser.add_argument("--request-inventory")
    parser.add_argument("--request-inventory-sha256")
    parser.add_argument("--request-plan")
    parser.add_argument("--request-plan-sha256")
    parser.add_argument("--requested-use", default="validation")
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
    parser.add_argument("--execution-costs", default="config/execution_costs.yaml")
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
