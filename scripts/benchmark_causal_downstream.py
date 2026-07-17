#!/usr/bin/env python3
"""Run leakage-safe causal-feature controls on the sealed Gate-3 sample."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from futures_foundation.finetune.downstream_probe import (
    causal_feature_matrix,
    fold_target_issue,
    fit_predict_fold,
    prediction_metrics,
    target_specs,
    target_values,
)
from futures_foundation.finetune.downstream_sample import (
    load_balanced_sample,
    load_row_selection,
    purged_calendar_splits,
)


SCHEMA_VERSION = "ffm_causal_downstream_benchmark_v1"


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


def _atomic_npz(path: Path, **values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp.npz")
    np.savez_compressed(temporary, **values)
    os.replace(temporary, path)


def _aggregate(records: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for record in records:
        key = (record["timeframe"], record["target"], record["kind"], record["arm"])
        groups.setdefault(key, []).append(record)
    output = []
    for (timeframe, target, kind, arm), rows in sorted(groups.items()):
        metric_names = ("r2", "mae", "spearman") if kind == "reg" else (
            "auc", "pr_auc", "brier", "prevalence",
        )
        metrics = {}
        for name in metric_names:
            values = np.asarray([row["metrics"][name] for row in rows], float)
            metrics[name] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "fold_values": values.tolist(),
            }
        output.append({
            "timeframe": timeframe, "target": target, "kind": kind, "arm": arm,
            "folds": len(rows), "test_rows": int(sum(row["test_rows"] for row in rows)),
            "metrics": metrics,
        })
    return output


def run(args) -> dict:
    sample_path = Path(args.sample).resolve()
    output_dir = Path(args.output_dir).resolve()
    arrays, sample_manifest = load_balanced_sample(sample_path)
    if args.row_selection:
        selection, selection_manifest = load_row_selection(
            args.row_selection, sample_manifest=sample_manifest,
        )
        eligible_rows = np.asarray(selection["row_index"], np.int64)
    else:
        selection_manifest = None
        eligible_rows = np.arange(len(arrays["stream_id"]), dtype=np.int64)
    heads = tuple(value.strip() for value in args.heads.split(",") if value.strip())
    controls = tuple(value.strip() for value in args.controls.split(",") if value.strip())
    if not heads or any(value not in {"linear", "xgb"} for value in heads):
        raise ValueError("heads must be drawn from linear,xgb")
    valid_controls = {"real", "shuffled_label", "random_feature", "time_destroyed"}
    if not controls or any(value not in valid_controls for value in controls):
        raise ValueError(f"controls must be drawn from {sorted(valid_controls)}")

    fold_records, scores, skipped_targets = {}, [], []
    pred_row, pred_target, pred_arm, pred_fold, pred_y, pred_value = [], [], [], [], [], []
    target_names = [spec.name for spec in target_specs(arrays)]
    arm_names = [f"causal_{head}_{control}" for head in heads for control in controls]
    for timeframe in sorted(str(value) for value in np.unique(arrays["timeframe"])):
        timeframe_rows = eligible_rows[arrays["timeframe"][eligible_rows] == timeframe]
        minutes = int(timeframe[:-3])
        local_splits, contract = purged_calendar_splits(
            arrays["decision_time_ns"][timeframe_rows],
            arrays["label_end_time_ns"][timeframe_rows],
            arrays["ticker"][timeframe_rows], folds=args.folds,
            embargo_ns=args.context_bars * minutes * 60 * 1_000_000_000,
        )
        fold_records[timeframe] = contract
        X, feature_names = causal_feature_matrix(arrays, timeframe_rows)
        groups = arrays["ticker"][timeframe_rows]
        weights = arrays["sample_weight"][timeframe_rows] if args.block_weights else None
        if not args.quiet:
            print(
                f"[{timeframe}] rows={len(timeframe_rows):,} features={X.shape[1]} "
                f"fold={contract['contract_sha256'][:12]}", flush=True,
            )
        for spec_i, spec in enumerate(target_specs(arrays)):
            y_all, valid_all = target_values(arrays, spec)
            y = y_all[timeframe_rows]
            valid = valid_all[timeframe_rows]
            if not np.any(valid):
                skipped_targets.append({
                    "timeframe": timeframe, "target": spec.name,
                    "reason": "fewer_than_two_forward_returns",
                })
                if not args.quiet:
                    print(f"  {spec.name}: skipped (fewer than two forward returns)", flush=True)
                continue
            for head in heads:
                for control in controls:
                    arm = f"causal_{head}_{control}"
                    arm_i = arm_names.index(arm)
                    for fold_i, (train, test) in enumerate(local_splits, start=1):
                        train = train[valid[train]]
                        test = test[valid[test]]
                        issue = fold_target_issue(y, train, test, spec.kind)
                        if issue:
                            skipped_targets.append({
                                "timeframe": timeframe, "target": spec.name,
                                "arm": arm, "fold": fold_i,
                                "train_rows": int(len(train)), "test_rows": int(len(test)),
                                "reason": issue,
                            })
                            continue
                        prediction = fit_predict_fold(
                            X, y, groups, train, test, kind=spec.kind, head=head,
                            control=control, sample_weight=weights,
                            seed=args.seed + spec_i * 100 + fold_i,
                        )
                        metrics = prediction_metrics(y[test], prediction, spec.kind)
                        scores.append({
                            "timeframe": timeframe, "target": spec.name, "kind": spec.kind,
                            "horizon_minutes": spec.horizon_minutes, "arm": arm,
                            "fold": fold_i, "train_rows": int(len(train)),
                            "test_rows": int(len(test)), "metrics": metrics,
                        })
                        global_test = timeframe_rows[test]
                        pred_row.append(global_test.astype(np.int32))
                        pred_target.append(np.full(len(test), spec_i, np.int16))
                        pred_arm.append(np.full(len(test), arm_i, np.int8))
                        pred_fold.append(np.full(len(test), fold_i, np.int8))
                        pred_y.append(np.asarray(y[test], np.float32))
                        pred_value.append(np.asarray(prediction, np.float32))
            primary = "r2" if spec.kind == "reg" else "auc"
            real = [
                row["metrics"][primary] for row in scores
                if row["timeframe"] == timeframe and row["target"] == spec.name
                and row["arm"] == f"causal_{heads[0]}_real"
            ]
            if real and not args.quiet:
                print(f"  {spec.name}: {heads[0]} {primary}={np.mean(real):.4f}", flush=True)

    prediction_path = output_dir / "causal_baseline_predictions.npz"
    _atomic_npz(
        prediction_path,
        row_index=np.concatenate(pred_row), target_index=np.concatenate(pred_target),
        arm_index=np.concatenate(pred_arm), fold=np.concatenate(pred_fold),
        y_true=np.concatenate(pred_y), prediction=np.concatenate(pred_value),
        target_names=np.asarray(target_names), arm_names=np.asarray(arm_names),
        sample_sha256=np.asarray(sample_manifest["artifact"]["sha256"]),
    )
    repo = Path(__file__).resolve().parents[1]
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "oos_read": False,
        "sample": {
            "path": str(sample_path), "sha256": sample_manifest["artifact"]["sha256"],
            "content_fingerprint": sample_manifest["content_fingerprint"],
        },
        "row_selection": (None if selection_manifest is None else {
            "path": selection_manifest["artifact"]["path"],
            "sha256": selection_manifest["artifact"]["sha256"],
            "content_fingerprint": selection_manifest["content_fingerprint"],
        }),
        "configuration": {
            "folds": args.folds, "context_bars": args.context_bars,
            "heads": list(heads), "controls": list(controls), "seed": args.seed,
            "block_weights": bool(args.block_weights),
            "linear": {"regression": "ridge_lsqr_alpha1", "binary": "logistic_C1"},
            "xgb": {
                "n_estimators": 120, "max_depth": 3, "learning_rate": 0.04,
                "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 10.0,
                "min_child_weight": 20.0,
            },
            "features": list(feature_names),
        },
        "source": {
            "git_revision": subprocess.check_output(
                ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True,
            ).strip(),
            "working_tree_dirty": bool(subprocess.check_output(
                ["git", "-C", str(repo), "status", "--porcelain"], text=True,
            ).strip()),
            "implementation_sha256": {
                "probe": _sha256(repo / "futures_foundation/finetune/downstream_probe.py"),
                "sample": _sha256(repo / "futures_foundation/finetune/downstream_sample.py"),
                "runner": _sha256(Path(__file__)),
            },
        },
        "fold_contracts": fold_records,
        "skipped_targets": skipped_targets,
        "fold_scores": scores,
        "summary": _aggregate(scores),
        "predictions": {
            "path": str(prediction_path), "sha256": _sha256(prediction_path),
            "rows": int(sum(len(value) for value in pred_row)),
        },
    }
    report_path = output_dir / "causal_baseline_results.json"
    _atomic_json(report_path, report)
    print(json.dumps({
        "status": "complete", "report": str(report_path),
        "predictions": str(prediction_path), "prediction_rows": report["predictions"]["rows"],
    }, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample",
        default="output/foundation_tournament/downstream_gate_v1/balanced_sample.npz",
    )
    parser.add_argument(
        "--output-dir", default="output/foundation_tournament/downstream_gate_v1",
    )
    parser.add_argument("--row-selection")
    parser.add_argument("--heads", default="linear,xgb")
    parser.add_argument(
        "--controls", default="real,shuffled_label,random_feature,time_destroyed",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--context-bars", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--block-weights", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
