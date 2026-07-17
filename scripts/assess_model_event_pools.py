#!/usr/bin/env python3
"""Assess causal event pools for model supervision, not standalone profitability."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np


CANONICAL_ARMS = (
    "supertrend__atr",
    "atr_zigzag__atr",
    "fractal_k2__atr",
    "fractal_zigzag__atr",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _binary_entropy(probability: float) -> float:
    p = float(probability)
    if p <= 0 or p >= 1:
        return 0.0
    return float(-(p * math.log2(p) + (1 - p) * math.log2(1 - p)))


def _stream_names(arrays: dict[str, np.ndarray]) -> np.ndarray:
    return np.char.add(
        np.char.add(arrays["ticker"].astype(str), "@"), arrays["timeframe"].astype(str)
    )


def _independent_context_count(
    arrays: dict[str, np.ndarray], rows: np.ndarray, context_bars: int
) -> int:
    streams = _stream_names(arrays)
    total = 0
    for stream in np.unique(streams[rows]):
        source = np.sort(arrays["source_signal_idx"][rows & (streams == stream)])
        last = -10**18
        for index in source:
            if int(index) - last >= int(context_bars):
                total += 1
                last = int(index)
    return total


def assess(arrays: dict[str, np.ndarray], *, context_bars: int = 256) -> dict:
    streams = _stream_names(arrays)
    targets = arrays["targets"].astype(float).tolist()
    strategies: dict[str, dict] = {}
    for strategy in sorted(np.unique(arrays["strategy"]).tolist()):
        rows = arrays["strategy"] == strategy
        counts = np.asarray(
            [np.count_nonzero(rows & (streams == stream)) for stream in np.unique(streams[rows])],
            dtype=float,
        )
        shares = counts / counts.sum()
        entropy = (
            1.0 if len(counts) == 1
            else float(-(shares * np.log(shares)).sum() / math.log(len(counts)))
        )
        target_rows = {}
        for target_i, target in enumerate(targets):
            positive_rate = float(arrays["reached"][rows, target_i].mean())
            target_rows[str(target)] = {
                "positive_rate": positive_rate,
                "positive_count": int(arrays["reached"][rows, target_i].sum()),
                "binary_entropy": _binary_entropy(positive_rate),
            }
        independent = _independent_context_count(arrays, rows, context_bars)
        strategies[str(strategy)] = {
            "events": int(rows.sum()),
            "symbols": int(len(np.unique(arrays["ticker"][rows]))),
            "timeframes": int(len(np.unique(arrays["timeframe"][rows]))),
            "streams": int(len(counts)),
            "median_events_per_stream": float(np.median(counts)),
            "minimum_events_per_stream": int(counts.min()),
            "normalized_stream_entropy": entropy,
            "one_minute_share": float((arrays["timeframe"][rows] == "1min").mean()),
            "long_rate": float((arrays["direction"][rows] > 0).mean()),
            "independent_contexts_approx": int(independent),
            "independent_fraction_approx": float(independent / rows.sum()),
            "risk_ticks_median": float(np.median(arrays["risk_ticks"][rows])),
            "peak_r_median": float(np.median(arrays["peak_r"][rows])),
            "peak_r_p90": float(np.quantile(arrays["peak_r"][rows], 0.90)),
            "targets": target_rows,
        }

    event_sets = {}
    for strategy in CANONICAL_ARMS:
        rows = arrays["strategy"] == strategy
        event_sets[strategy] = {
            (str(ticker), str(timeframe), int(timestamp), int(direction))
            for ticker, timeframe, timestamp, direction in zip(
                arrays["ticker"][rows], arrays["timeframe"][rows],
                arrays["signal_time_ns"][rows], arrays["direction"][rows],
            )
        }
    overlap = []
    for i, left in enumerate(CANONICAL_ARMS):
        for right in CANONICAL_ARMS[i + 1 :]:
            intersection = len(event_sets[left] & event_sets[right])
            union = len(event_sets[left] | event_sets[right])
            overlap.append({
                "left": left, "right": right, "exact_overlap": intersection,
                "jaccard": float(intersection / union),
                "fraction_of_smaller": float(
                    intersection / min(len(event_sets[left]), len(event_sets[right]))
                ),
            })

    selected = ("supertrend__atr", "atr_zigzag__atr", "fractal_k2__atr")
    selected_rows = np.isin(arrays["strategy"], selected)
    records = {}
    for ticker, timeframe, timestamp, direction, source_index in zip(
        arrays["ticker"][selected_rows], arrays["timeframe"][selected_rows],
        arrays["signal_time_ns"][selected_rows], arrays["direction"][selected_rows],
        arrays["source_signal_idx"][selected_rows],
    ):
        records[(str(ticker), str(timeframe), int(timestamp), int(direction))] = int(source_index)
    union_streams: dict[tuple[str, str], list[int]] = {}
    for (ticker, timeframe, _, _), source_index in records.items():
        union_streams.setdefault((ticker, timeframe), []).append(source_index)
    union_independent = 0
    independent_by_stream = []
    for source in union_streams.values():
        last, count = -10**18, 0
        for index in sorted(set(source)):
            if index - last >= context_bars:
                count += 1
                last = index
        union_independent += count
        independent_by_stream.append(count)

    return {
        "context_bars": int(context_bars),
        "strategies": strategies,
        "canonical_exact_overlap": overlap,
        "recommended_trigger_union_diagnostic": {
            "arms": list(selected),
            "deduplicated_events": int(len(records)),
            "streams": int(len(union_streams)),
            "independent_contexts_approx": int(union_independent),
            "median_independent_contexts_per_stream": float(np.median(independent_by_stream)),
            "minimum_independent_contexts_per_stream": int(min(independent_by_stream)),
        },
        "limitations": [
            "independent_contexts is a greedy 256-source-bar spacing diagnostic, not an IID claim",
            "events were materialized with one-active-trade suppression for the execution ruler",
            "raw event count must not be used as effective sample size",
            "profitability is intentionally not a model-pool selection metric",
        ],
    }


def _atomic_json(path: Path, value: dict) -> None:
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def run(args) -> None:
    source_report = Path(args.source_report).resolve()
    report = json.loads(source_report.read_text())
    if report.get("status") != "complete" or report.get("oos_read") is not False:
        raise ValueError("source event report must be complete and development-only")
    events = Path(report["events"]["path"])
    if _sha256(events) != report["events"]["sha256"]:
        raise ValueError("source event artifact hash mismatch")
    with np.load(events, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    result = {
        "schema_version": "ffm_model_event_pool_assessment_v1",
        "status": "complete",
        "oos_read": False,
        "source_report": str(source_report),
        "source_report_sha256": _sha256(source_report),
        "source_events": str(events),
        "source_events_sha256": _sha256(events),
        **assess(arrays, context_bars=args.context_bars),
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(output, result)
    print(f"[done] {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-report", default="output/trend_strategy_benchmark_tick_dev_v1/report.json"
    )
    parser.add_argument("--output", default="output/model_event_pool_assessment/report.json")
    parser.add_argument("--context-bars", type=int, default=256)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
