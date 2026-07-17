#!/usr/bin/env python3
"""Run the matched raw-event ruler across trend strategy candidates."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from futures_foundation.finetune.trend_strategy_eval import (
    RulerConfig, evaluate_stream, events_to_arrays, load_stream, summarize_events,
)


TICKERS = ("ES", "NQ", "RTY", "YM", "GC", "SI", "CL", "ZB", "ZN")
TIMEFRAMES = ("1min", "3min", "5min", "15min", "30min", "60min")
OOS_START = "2025-07-01"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _atomic_npz(path, arrays):
    path = Path(path)
    temporary = Path(str(path) + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _dataset_fingerprint(data_dir, tickers, timeframes):
    digest = hashlib.sha256()
    manifest = Path(data_dir) / "MANIFEST.json"
    if manifest.is_file():
        digest.update(_sha256(manifest).encode())
    for ticker in tickers:
        for timeframe in timeframes:
            path = Path(data_dir) / f"{ticker}_{timeframe}.csv"
            stat = path.stat()
            digest.update(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()


def _source_fingerprint():
    root = Path(__file__).resolve().parents[1]
    relative = (
        "scripts/benchmark_trend_strategies.py",
        "futures_foundation/finetune/trend_strategy_eval.py",
        "futures_foundation/pivots.py",
        "futures_foundation/primitives/detection.py",
        "futures_foundation/pipeline/_primitives.py",
    )
    files = {name: _sha256(root / name) for name in relative}
    digest = hashlib.sha256()
    for name, value in files.items():
        digest.update(f"{name}:{value}".encode())
    return {"sha256": digest.hexdigest(), "files": files}


def _render(report):
    lines = [
        "# Matched trend-strategy event benchmark", "",
        f"Status: `{report['status']}`. OOS read: `{str(report['oos_read']).lower()}`.", "",
        (f"Period: `{report['config']['eval_start']}` to `{report['config']['eval_end']}`; "
         f"primary target {report['config']['primary_target']}R; "
         f"horizon {report['config']['horizon_hours']} hours."), "",
        "| Strategy | Signals | WR@3R | Mean R (1 tick RT) | Mean R (2 ticks RT) | PF | Median risk ticks | Positive folds | Worst fold | Promising |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    ranking = sorted(report["strategies"].items(), key=lambda item: item[1]["mean_r"], reverse=True)
    for name, value in ranking:
        lines.append(
            f"| {name} | {value['signals']:,} | {value['wr']:.4f} | "
            f"{value['mean_r']:.4f} | {value['cost_tick_sensitivity']['2.0']['mean_r']:.4f} | "
            f"{value['profit_factor']:.3f} | "
            f"{value['risk_ticks']['median']:.1f} | {value['positive_fold_fraction']:.2f} | "
            f"{value['worst_fold_mean_r']:.4f} | "
            f"{'yes' if value['development_promising'] else 'no'} |"
        )
    lines.extend([
        "", "The promising flag is a development triage rule, not an OOS or deployment verdict. "
        "Maximum drawdown is computed over chronologically ordered event R and is not a capital-"
        "constrained portfolio simulation.", "",
    ])
    return "\n".join(lines)


def run(args):
    tickers = tuple(value for value in args.tickers.split(",") if value)
    timeframes = tuple(value for value in args.timeframes.split(",") if value)
    if args.eval_end > OOS_START:
        raise ValueError(f"development ruler refuses eval_end after OOS boundary {OOS_START}")
    cfg = RulerConfig(
        eval_start=args.eval_start, eval_end=args.eval_end, warmup_days=args.warmup_days,
        context=args.context, horizon_hours=args.horizon_hours, atr_stop=args.atr_stop,
        structural_buffer_atr=args.structural_buffer_atr,
        targets=tuple(float(value) for value in args.targets.split(",") if value),
        primary_target=args.primary_target, round_trip_cost_ticks=args.round_trip_cost_ticks,
    )
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    events_path = output_dir / "events.npz"
    all_events = []
    for ticker in tickers:
        for timeframe in timeframes:
            path = Path(args.data_dir) / f"{ticker}_{timeframe}.csv"
            frame = load_stream(
                path, cfg.eval_start, cfg.eval_end, warmup_days=cfg.warmup_days,
                chunksize=args.csv_chunksize,
            )
            events = evaluate_stream(frame, ticker, timeframe, cfg)
            all_events.extend(events)
            print(f"[events] {ticker}@{timeframe}: {len(events):,}", flush=True)
    arrays = events_to_arrays(all_events, cfg.targets)
    _atomic_npz(events_path, arrays)
    strategies = summarize_events(arrays, cfg, folds=args.folds)
    config = {
        "eval_start": cfg.eval_start, "eval_end": cfg.eval_end,
        "warmup_days": cfg.warmup_days, "context": cfg.context,
        "horizon_hours": cfg.horizon_hours, "atr_period": cfg.atr_period,
        "atr_stop": cfg.atr_stop, "structural_buffer_atr": cfg.structural_buffer_atr,
        "targets": list(cfg.targets), "primary_target": cfg.primary_target,
        "round_trip_cost_ticks": cfg.round_trip_cost_ticks,
        "same_bar_policy": cfg.same_bar_policy,
        "entry": "next_bar_open", "overlap": "one active trade per strategy/stream",
        "tickers": list(tickers), "timeframes": list(timeframes), "folds": args.folds,
    }
    report = {
        "schema_version": "ffm_matched_trend_strategy_events_v2",
        "created_utc": datetime.now(timezone.utc).isoformat(), "status": "complete",
        "oos_read": False, "dataset_fingerprint": _dataset_fingerprint(
            args.data_dir, tickers, timeframes,
        ),
        "source": _source_fingerprint(),
        "config": config,
        "events": {"path": str(events_path), "sha256": _sha256(events_path),
                   "rows": int(len(arrays["strategy"]))},
        "strategies": strategies,
        "limitations": [
            "raw event ruler only; no learned filter or classical-feature control",
            "one-tick round-trip cost is an optimistic executable lower bound; broker fees are not modeled",
            "event-sequence drawdown is not a capital-constrained portfolio drawdown",
            "development period has been inspected and is not pristine OOS",
        ],
    }
    _atomic_json(output_dir / "report.json", report)
    (output_dir / "report.md").write_text(_render(report))
    print(f"[done] {output_dir / 'report.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/ssl_corpus_v2_6tf")
    parser.add_argument("--output-dir", default="output/trend_strategy_benchmark_dev")
    parser.add_argument("--tickers", default=",".join(TICKERS))
    parser.add_argument("--timeframes", default=",".join(TIMEFRAMES))
    parser.add_argument("--eval-start", default="2024-07-01")
    parser.add_argument("--eval-end", default=OOS_START)
    parser.add_argument("--warmup-days", type=int, default=180)
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--horizon-hours", type=float, default=6.0)
    parser.add_argument("--atr-stop", type=float, default=0.5)
    parser.add_argument("--structural-buffer-atr", type=float, default=0.05)
    parser.add_argument("--targets", default="2,3,4,6")
    parser.add_argument("--primary-target", type=float, default=3.0)
    parser.add_argument("--round-trip-cost-ticks", type=float, default=1.0)
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--csv-chunksize", type=int, default=500000)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
