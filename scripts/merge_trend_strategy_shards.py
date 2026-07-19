#!/usr/bin/env python3
"""Hash-check and merge restart-safe trend-strategy benchmark shards."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from futures_foundation.finetune.trend_strategy_eval import RulerConfig, summarize_events


SCHEMA = "ffm_matched_trend_strategy_events_v2"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path, value):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _atomic_npz(path, arrays):
    temporary = Path(str(path) + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


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
    for name, value in sorted(report["strategies"].items(),
                              key=lambda item: item[1]["mean_r"], reverse=True):
        lines.append(
            f"| {name} | {value['signals']:,} | {value['wr']:.4f} | "
            f"{value['mean_r']:.4f} | {value['cost_tick_sensitivity']['2.0']['mean_r']:.4f} | "
            f"{value['profit_factor']:.3f} | "
            f"{value['risk_ticks']['median']:.1f} | {value['positive_fold_fraction']:.2f} | "
            f"{value['worst_fold_mean_r']:.4f} | "
            f"{'yes' if value['development_promising'] else 'no'} |"
        )
    lines.extend(["", "Promising is a development triage flag, not an OOS verdict. Event-R "
                  "drawdown is not a capital-constrained portfolio drawdown.", ""])
    return "\n".join(lines)


def run(args):
    shard_dirs = [Path(value).resolve() for value in args.shards]
    reports, loaded = [], []
    for directory in shard_dirs:
        report = json.loads((directory / "report.json").read_text())
        if report.get("schema_version") != SCHEMA or report.get("status") != "complete":
            raise ValueError(f"incomplete or incompatible shard: {directory}")
        if report.get("oos_read") is not False:
            raise ValueError(f"shard is not development-only: {directory}")
        events_path = directory / "events.npz"
        if _sha256(events_path) != report["events"]["sha256"]:
            raise ValueError(f"event hash mismatch: {directory}")
        with np.load(events_path, allow_pickle=False) as saved:
            loaded.append({key: saved[key] for key in saved.files})
        reports.append(report)
    base = dict(reports[0]["config"])
    comparable = {key: value for key, value in base.items() if key != "tickers"}
    tickers = []
    for report in reports:
        if report.get("source") != reports[0].get("source"):
            raise ValueError("shard source fingerprint mismatch")
        current = {key: value for key, value in report["config"].items() if key != "tickers"}
        if current != comparable:
            raise ValueError("shard configuration mismatch")
        overlap = set(tickers) & set(report["config"]["tickers"])
        if overlap:
            raise ValueError(f"duplicate ticker shards: {sorted(overlap)}")
        tickers.extend(report["config"]["tickers"])
    targets = loaded[0]["targets"]
    if any(not np.array_equal(item["targets"], targets) for item in loaded[1:]):
        raise ValueError("target mismatch across shards")
    keys = tuple(key for key in loaded[0] if key != "targets")
    arrays = {key: np.concatenate([item[key] for item in loaded]) for key in keys}
    arrays["targets"] = targets
    order = np.argsort(arrays["signal_time_ns"], kind="stable")
    for key in keys:
        arrays[key] = arrays[key][order]
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    events_path = output / "events.npz"
    _atomic_npz(events_path, arrays)
    cfg = RulerConfig(
        eval_start=base["eval_start"], eval_end=base["eval_end"],
        warmup_days=base["warmup_days"], context=base["context"],
        horizon_hours=base["horizon_hours"], atr_period=base["atr_period"],
        atr_stop=base["atr_stop"], structural_buffer_atr=base["structural_buffer_atr"],
        targets=tuple(base["targets"]), primary_target=base["primary_target"],
        added_slippage_ticks_round_trip=base["added_slippage_ticks_round_trip"],
        same_bar_policy=base["same_bar_policy"],
    )
    base["tickers"] = sorted(tickers)
    dataset_fingerprint = hashlib.sha256()
    artifact_fingerprint = hashlib.sha256()
    for report in sorted(reports, key=lambda value: value["config"]["tickers"]):
        dataset_fingerprint.update(report["dataset_fingerprint"].encode())
        artifact_fingerprint.update(report["dataset_fingerprint"].encode())
        artifact_fingerprint.update(report["events"]["sha256"].encode())
    report = {
        "schema_version": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(), "status": "complete",
        "oos_read": False, "dataset_fingerprint": dataset_fingerprint.hexdigest(),
        "artifact_fingerprint": artifact_fingerprint.hexdigest(), "config": base,
        "source": reports[0]["source"],
        "scorer_source": {
            "merge_script_sha256": _sha256(Path(__file__).resolve()),
            "strategy_eval_sha256": _sha256(
                Path(__file__).resolve().parents[1]
                / "futures_foundation/finetune/trend_strategy_eval.py"
            ),
        },
        "events": {"path": str(events_path), "sha256": _sha256(events_path),
                   "rows": int(len(arrays["strategy"]))},
        "shards": [{"report": str(directory / "report.json"),
                    "events_sha256": value["events"]["sha256"]}
                   for directory, value in zip(shard_dirs, reports)],
        "strategies": summarize_events(arrays, cfg, folds=base["folds"]),
        "limitations": reports[0]["limitations"],
    }
    _atomic_json(output / "report.json", report)
    (output / "report.md").write_text(_render(report))
    print(f"[done] {output / 'report.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="output/trend_strategy_benchmark_dev")
    parser.add_argument("shards", nargs="+")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
