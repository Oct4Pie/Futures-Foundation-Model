#!/usr/bin/env python3
"""Paired calendar-block comparison of two sealed downstream prediction artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from futures_foundation.finetune.downstream_probe import paired_calendar_block_bootstrap
from futures_foundation.finetune.downstream_probe import target_specs
from futures_foundation.finetune.downstream_sample import load_balanced_sample


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def load_predictions(results_path: str | Path, arm: str | None = None) -> tuple[dict, dict]:
    results_path = Path(results_path).resolve()
    report = json.loads(results_path.read_text())
    if report.get("status") != "complete" or report.get("oos_read") is not False:
        raise ValueError("prediction report is incomplete or not development-only")
    prediction_path = Path(report["predictions"]["path"])
    if _sha256(prediction_path) != report["predictions"]["sha256"]:
        raise ValueError("prediction artifact hash mismatch")
    with np.load(prediction_path, allow_pickle=False) as saved:
        values = {key: saved[key] for key in saved.files}
    names_key = "probe_arms" if "probe_arms" in values else "arm_names"
    names = [str(value) for value in values[names_key]]
    if arm is None:
        if len(names) != 1:
            raise ValueError("--arm is required when a prediction artifact contains multiple arms")
        arm = names[0]
    if arm not in names:
        raise ValueError(f"arm {arm!r} is absent from {prediction_path}")
    arm_index = names.index(arm)
    keep = values["arm_index"] == arm_index
    selected = {
        key: np.asarray(values[key][keep])
        for key in ("row_index", "target_index", "fold", "y_true", "prediction")
    }
    selected["target_names"] = np.asarray(values["target_names"])
    selected["arm"] = arm
    return selected, {
        "results": str(results_path), "results_sha256": _sha256(results_path),
        "predictions": str(prediction_path), "predictions_sha256": _sha256(prediction_path),
        "arm": arm,
    }


def aligned_rows(left: dict, right: dict, target_index: int, timeframe_mask: np.ndarray):
    def select(values):
        rows = np.flatnonzero(
            (values["target_index"] == target_index)
            & timeframe_mask[values["row_index"]]
        )
        order = np.lexsort((values["fold"][rows], values["row_index"][rows]))
        return rows[order]

    left_rows, right_rows = select(left), select(right)
    if not len(left_rows) and not len(right_rows):
        return None
    if not len(left_rows) or not len(right_rows):
        raise ValueError("only one side contains prediction rows")
    for key in ("row_index", "fold", "y_true"):
        if not np.array_equal(left[key][left_rows], right[key][right_rows]):
            raise ValueError(f"paired prediction mismatch for {key}")
    return left_rows, right_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-results", required=True)
    parser.add_argument("--left-arm")
    parser.add_argument("--right-results", required=True)
    parser.add_argument("--right-arm")
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--sample", default="output/foundation_tournament/downstream_gate_v1/balanced_sample.npz",
    )
    parser.add_argument("--block-days", type=int, default=7)
    parser.add_argument("--repetitions", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260716)
    args = parser.parse_args()

    sample, sample_manifest = load_balanced_sample(args.sample)
    left, left_identity = load_predictions(args.left_results, args.left_arm)
    right, right_identity = load_predictions(args.right_results, args.right_arm)
    left_targets = [str(value) for value in left["target_names"]]
    right_targets = [str(value) for value in right["target_names"]]
    if left_targets != right_targets:
        raise ValueError("target-name mismatch between prediction artifacts")

    comparisons, skipped = [], []
    timeframes = sorted(str(value) for value in np.unique(sample["timeframe"]))
    kinds = {spec.name: spec.kind for spec in target_specs(sample)}
    for target_index, target in enumerate(left_targets):
        if target not in kinds:
            raise ValueError(f"target {target!r} is absent from the current target contract")
        kind = kinds[target]
        for timeframe_index, timeframe in enumerate(timeframes):
            timeframe_mask = np.asarray(sample["timeframe"] == timeframe)
            aligned = aligned_rows(
                left, right, target_index, timeframe_mask,
            )
            if aligned is None:
                skipped.append({
                    "timeframe": timeframe, "target": target,
                    "reason": "both_artifacts_have_no_valid_rows",
                })
                continue
            left_rows, right_rows = aligned
            row_index = left["row_index"][left_rows]
            interval = paired_calendar_block_bootstrap(
                left["y_true"][left_rows], left["prediction"][left_rows],
                right["prediction"][right_rows], sample["decision_time_ns"][row_index],
                kind=kind, block_days=args.block_days, repetitions=args.repetitions,
                seed=args.seed + target_index * 1009 + timeframe_index,
            )
            comparisons.append({
                "timeframe": timeframe, "target": target, "kind": kind,
                "rows": int(len(left_rows)), **interval,
            })

    report = {
        "schema_version": "ffm_downstream_paired_comparison_v1",
        "status": "complete", "oos_read": False, "label": args.label,
        "sample_sha256": sample_manifest["artifact"]["sha256"],
        "left": left_identity, "right": right_identity,
        "configuration": {
            "block_days": args.block_days, "repetitions": args.repetitions,
            "seed": args.seed, "resampling_unit": "UTC_calendar_block",
            "delta_definition": "left_metric_minus_right_metric",
        },
        "comparisons": comparisons, "skipped": skipped,
    }
    _atomic_json(args.output, report)
    print(json.dumps({
        "status": "complete", "comparisons": len(comparisons),
        "output": str(args.output.resolve()), "sha256": _sha256(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
