#!/usr/bin/env python3
"""Add paired weekly-block uncertainty to the causal Gate-3 controls."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from futures_foundation.finetune.downstream_probe import paired_calendar_block_bootstrap, target_specs
from futures_foundation.finetune.downstream_sample import load_balanced_sample


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample",
        default="output/foundation_tournament/downstream_gate_v1/balanced_sample.npz",
    )
    parser.add_argument(
        "--results",
        default="output/foundation_tournament/downstream_gate_v1/causal_baseline_results.json",
    )
    parser.add_argument(
        "--predictions",
        default="output/foundation_tournament/downstream_gate_v1/causal_baseline_predictions.npz",
    )
    parser.add_argument(
        "--output",
        default="output/foundation_tournament/downstream_gate_v1/causal_baseline_bootstrap.json",
    )
    parser.add_argument("--block-days", type=int, default=7)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--controls", default="time_destroyed")
    parser.add_argument("--seed", type=int, default=20260716)
    args = parser.parse_args()
    controls = tuple(value.strip() for value in args.controls.split(",") if value.strip())
    allowed_controls = {"shuffled_label", "random_feature", "time_destroyed"}
    if not controls or any(value not in allowed_controls for value in controls):
        raise ValueError(f"controls must be drawn from {sorted(allowed_controls)}")

    sample, sample_manifest = load_balanced_sample(args.sample)
    results_path, predictions_path = Path(args.results), Path(args.predictions)
    report = json.loads(results_path.read_text())
    if report.get("status") != "complete" or report.get("oos_read") is not False:
        raise ValueError("causal baseline report is incomplete or not development-only")
    if _sha256(predictions_path) != report["predictions"]["sha256"]:
        raise ValueError("causal prediction hash mismatch")
    if report["sample"]["sha256"] != sample_manifest["artifact"]["sha256"]:
        raise ValueError("causal report/sample identity mismatch")
    with np.load(predictions_path, allow_pickle=False) as saved:
        predictions = {key: saved[key] for key in saved.files}
    arms = [str(value) for value in predictions["arm_names"]]
    targets = [str(value) for value in predictions["target_names"]]
    target_kinds = {spec.name: spec.kind for spec in target_specs(sample)}
    comparisons = []
    for head in ("linear", "xgb"):
        real_name = f"causal_{head}_real"
        real_i = arms.index(real_name)
        for target_i, target in enumerate(targets):
            if target not in target_kinds:
                raise ValueError(f"target {target!r} is absent from the current target contract")
            kind = target_kinds[target]
            for timeframe in sorted(str(value) for value in np.unique(sample["timeframe"])):
                base_mask = (
                    (predictions["target_index"] == target_i)
                    & (sample["timeframe"][predictions["row_index"]] == timeframe)
                )
                real_rows = np.flatnonzero(base_mask & (predictions["arm_index"] == real_i))
                if not len(real_rows):
                    continue
                real_order = np.lexsort((
                    predictions["fold"][real_rows], predictions["row_index"][real_rows],
                ))
                real_rows = real_rows[real_order]
                for control in controls:
                    control_name = f"causal_{head}_{control}"
                    control_i = arms.index(control_name)
                    control_rows = np.flatnonzero(
                        base_mask & (predictions["arm_index"] == control_i)
                    )
                    control_order = np.lexsort((
                        predictions["fold"][control_rows], predictions["row_index"][control_rows],
                    ))
                    control_rows = control_rows[control_order]
                    if not np.array_equal(
                        predictions["row_index"][real_rows],
                        predictions["row_index"][control_rows],
                    ) or not np.array_equal(
                        predictions["fold"][real_rows], predictions["fold"][control_rows],
                    ):
                        raise ValueError(f"prediction rows differ for {real_name}/{control_name}")
                    if not np.array_equal(
                        predictions["y_true"][real_rows], predictions["y_true"][control_rows],
                    ):
                        raise ValueError(f"truth rows differ for {real_name}/{control_name}")
                    row_index = predictions["row_index"][real_rows]
                    interval = paired_calendar_block_bootstrap(
                        predictions["y_true"][real_rows],
                        predictions["prediction"][real_rows],
                        predictions["prediction"][control_rows],
                        sample["decision_time_ns"][row_index], kind=kind,
                        block_days=args.block_days, repetitions=args.repetitions,
                        seed=args.seed + target_i * 1009 + arms.index(control_name),
                    )
                    comparisons.append({
                        "timeframe": timeframe, "target": target, "kind": kind,
                        "arm": real_name, "control": control_name, "rows": int(len(real_rows)),
                        **interval,
                    })
    output = {
        "schema_version": "ffm_causal_downstream_bootstrap_v1",
        "status": "complete", "oos_read": False,
        "sample_sha256": sample_manifest["artifact"]["sha256"],
        "results_sha256": _sha256(results_path),
        "predictions_sha256": _sha256(predictions_path),
        "configuration": {
            "block_days": args.block_days, "repetitions": args.repetitions,
            "seed": args.seed, "controls": list(controls),
            "resampling_unit": "UTC_calendar_block",
        },
        "comparisons": comparisons,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_path, output)
    print(json.dumps({
        "status": "complete", "comparisons": len(comparisons),
        "output": str(output_path), "sha256": _sha256(output_path),
    }, indent=2))


if __name__ == "__main__":
    main()
