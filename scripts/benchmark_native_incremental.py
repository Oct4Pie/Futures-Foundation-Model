#!/usr/bin/env python3
"""Benchmark one native feature table against causal and negative-control rulers.

Primary arms:
  * causal-only linear ruler;
  * model-only linear ruler;
  * causal-plus-model linear ruler;
  * residual-over-causal linear correction.

Model outputs receive one train-fold-fitted, target-independent 32-component PCA
bottleneck.  Model-only negative controls use shuffled labels, random features, and
time-destroyed features.  A passing screen may fund nonlinear sensitivity only; it
cannot grant promotion, full training, OOS access, deployment, or trading.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from futures_foundation.finetune.downstream_probe import (
    causal_feature_matrix,
    fold_target_issue,
    prediction_metrics,
    target_specs,
    target_values,
)
from futures_foundation.finetune.downstream_sample import (
    load_balanced_sample,
    load_row_selection,
    purged_calendar_splits,
)
from futures_foundation.finetune.native_downstream_features import load_feature_table
from futures_foundation.finetune.native_downstream_ruler import (
    PRIMARY_TARGETS,
    SCREEN_POLICY,
    fit_fold_arms,
    fit_model_bottleneck,
    root_calendar_block_bootstrap,
    screen_verdict,
)


SCHEMA_VERSION = "ffm_native_incremental_benchmark_v1"
ARM_NAMES = (
    "causal", "model", "causal_plus_model", "residual_over_causal",
    "model_shuffled_label", "model_random_feature", "model_time_destroyed",
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: str | Path, value: object) -> Path:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(target) + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)
    return target


def _atomic_npz(path: str | Path, **arrays: np.ndarray) -> Path:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(target) + f".{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, target)
    return target


def _target_names(value: str, available: list[str]) -> tuple[str, ...]:
    if value == "primary":
        selected = tuple(PRIMARY_TARGETS)
    elif value == "all":
        selected = tuple(available)
    else:
        selected = tuple(item.strip() for item in value.split(",") if item.strip())
    missing = sorted(set(selected) - set(available))
    if not selected or missing:
        raise ValueError(f"target selection is empty or unknown; missing={missing}")
    return selected


def _safe_bootstrap(
    y: np.ndarray,
    candidate: np.ndarray,
    causal: np.ndarray,
    time_ns: np.ndarray,
    roots: np.ndarray,
    *,
    kind: str,
    seed: int,
    repetitions: int,
) -> dict[str, Any]:
    score_name = "r2" if kind == "reg" else "auc"
    point = prediction_metrics(y, candidate, kind)[score_name] - prediction_metrics(
        y, causal, kind,
    )[score_name]
    try:
        return root_calendar_block_bootstrap(
            y,
            candidate,
            causal,
            time_ns,
            roots,
            kind=kind,
            block_days=int(SCREEN_POLICY["bootstrap_block_days"]),
            repetitions=int(repetitions),
            confidence=float(SCREEN_POLICY["bootstrap_confidence"]),
            seed=seed,
        )
    except ValueError as exc:
        return {
            "metric": score_name,
            "delta": float(point),
            "ci_low_99": None,
            "ci_high_99": None,
            "positive_probability": 0.0,
            "root_calendar_blocks": 0,
            "block_days": int(SCREEN_POLICY["bootstrap_block_days"]),
            "valid_repetitions": 0,
            "confidence": float(SCREEN_POLICY["bootstrap_confidence"]),
            "error": str(exc),
        }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sample, sample_manifest = load_balanced_sample(args.sample)
    selection, selection_manifest = load_row_selection(
        args.row_selection, sample_manifest=sample_manifest,
    )
    feature_arrays, feature_manifest = load_feature_table(args.features)
    row_index = np.asarray(selection["row_index"], np.int64)
    if not np.array_equal(feature_arrays["row_index"], row_index):
        raise ValueError("native feature table and frozen row selection differ")
    if len(np.unique(sample["timeframe"][row_index])) != 1:
        raise ValueError("native incremental benchmark requires one common timeframe")
    timeframe = str(sample["timeframe"][row_index][0])
    minutes = int(timeframe[:-3])
    folds, fold_contract = purged_calendar_splits(
        sample["decision_time_ns"][row_index],
        sample["label_end_time_ns"][row_index],
        sample["ticker"][row_index],
        folds=args.folds,
        embargo_ns=args.context_bars * minutes * 60 * 1_000_000_000,
    )
    if fold_contract["contract_sha256"] != args.expected_fold_sha256:
        raise ValueError("native incremental fold contract differs from the external receipt")

    causal, causal_names = causal_feature_matrix(sample, row_index)
    model_features = np.asarray(feature_arrays["features"], np.float32)
    groups = np.asarray(sample["ticker"][row_index]).astype(str)
    weights = (
        np.asarray(sample["sample_weight"][row_index], np.float32)
        if args.block_weights else None
    )
    all_specs = target_specs(sample)
    available = [spec.name for spec in all_specs]
    selected_names = _target_names(args.targets, available)
    selected_specs = [spec for spec in all_specs if spec.name in selected_names]
    if tuple(spec.name for spec in selected_specs) != tuple(
        name for name in available if name in selected_names
    ):
        raise RuntimeError("target selection ordering drifted")

    bottlenecks = []
    for fold_i, (train, test) in enumerate(folds, start=1):
        real_train, real_test, real_meta = fit_model_bottleneck(
            model_features, groups, train, test,
            control="real", components=args.pca_components,
            seed=args.seed + fold_i * 10,
        )
        random_train, random_test, random_meta = fit_model_bottleneck(
            model_features, groups, train, test,
            control="random_feature", components=args.pca_components,
            seed=args.seed + fold_i * 10 + 1,
        )
        destroyed_train, destroyed_test, destroyed_meta = fit_model_bottleneck(
            model_features, groups, train, test,
            control="time_destroyed", components=args.pca_components,
            seed=args.seed + fold_i * 10 + 2,
        )
        train_position = np.full(len(row_index), -1, np.int64)
        test_position = np.full(len(row_index), -1, np.int64)
        train_position[train] = np.arange(len(train), dtype=np.int64)
        test_position[test] = np.arange(len(test), dtype=np.int64)
        bottlenecks.append({
            "train": train, "test": test,
            "train_position": train_position, "test_position": test_position,
            "real_train": real_train, "real_test": real_test,
            "random_train": random_train, "random_test": random_test,
            "destroyed_train": destroyed_train, "destroyed_test": destroyed_test,
            "metadata": {
                "real": real_meta, "random_feature": random_meta,
                "time_destroyed": destroyed_meta,
            },
        })
        print(
            f"[bottleneck] fold={fold_i} train={len(train):,} test={len(test):,} "
            f"components={real_train.shape[1]}",
            flush=True,
        )

    prediction_rows: dict[str, list[np.ndarray]] = {
        name: [] for name in (
            "row_index", "target_index", "fold", "y_true", *ARM_NAMES,
        )
    }
    fold_scores = []
    skipped = []
    pooled: dict[str, dict[str, list[np.ndarray]]] = {}
    spec_index = {spec.name: available.index(spec.name) for spec in selected_specs}
    for local_spec_i, spec in enumerate(selected_specs):
        y_all, valid_all = target_values(sample, spec)
        y = y_all[row_index]
        valid = valid_all[row_index]
        pooled[spec.name] = {
            name: [] for name in ("y", "row", "time", "root", *ARM_NAMES)
        }
        for fold_i, item in enumerate(bottlenecks, start=1):
            train = item["train"]
            test = item["test"]
            target_train = train[valid[train]]
            target_test = test[valid[test]]
            issue = fold_target_issue(y, target_train, target_test, spec.kind)
            if issue:
                skipped.append({
                    "target": spec.name, "fold": fold_i, "reason": issue,
                    "train_rows": int(len(target_train)),
                    "test_rows": int(len(target_test)),
                })
                continue
            train_pos = item["train_position"][target_train]
            test_pos = item["test_position"][target_test]
            predictions = fit_fold_arms(
                causal,
                item["real_train"][train_pos],
                item["real_test"][test_pos],
                y,
                groups,
                target_train,
                target_test,
                kind=spec.kind,
                sample_weight=weights,
                seed=args.seed + local_spec_i * 100 + fold_i,
                random_train=item["random_train"][train_pos],
                random_test=item["random_test"][test_pos],
                destroyed_train=item["destroyed_train"][train_pos],
                destroyed_test=item["destroyed_test"][test_pos],
            )
            metrics = {
                name: prediction_metrics(y[target_test], value, spec.kind)
                for name, value in predictions.items()
            }
            fold_scores.append({
                "target": spec.name,
                "kind": spec.kind,
                "horizon_minutes": spec.horizon_minutes,
                "fold": fold_i,
                "train_rows": int(len(target_train)),
                "test_rows": int(len(target_test)),
                "metrics": metrics,
            })
            global_rows = row_index[target_test]
            prediction_rows["row_index"].append(global_rows.astype(np.int32))
            prediction_rows["target_index"].append(
                np.full(len(target_test), spec_index[spec.name], np.int16)
            )
            prediction_rows["fold"].append(
                np.full(len(target_test), fold_i, np.int8)
            )
            prediction_rows["y_true"].append(np.asarray(y[target_test], np.float32))
            for name in ARM_NAMES:
                prediction_rows[name].append(np.asarray(predictions[name], np.float32))
            target_pool = pooled[spec.name]
            target_pool["y"].append(np.asarray(y[target_test]))
            target_pool["row"].append(global_rows)
            target_pool["time"].append(sample["decision_time_ns"][global_rows])
            target_pool["root"].append(sample["ticker"][global_rows])
            for name in ARM_NAMES:
                target_pool[name].append(np.asarray(predictions[name], np.float32))
        print(f"[target] {spec.name}", flush=True)

    summaries = []
    for spec_i, spec in enumerate(selected_specs):
        target_pool = pooled[spec.name]
        if not target_pool["y"]:
            continue
        y = np.concatenate(target_pool["y"])
        rows = np.concatenate(target_pool["row"])
        times = np.concatenate(target_pool["time"])
        roots = np.concatenate(target_pool["root"])
        predictions = {
            name: np.concatenate(target_pool[name]) for name in ARM_NAMES
        }
        metrics = {
            name: prediction_metrics(y, value, spec.kind)
            for name, value in predictions.items()
        }
        bootstrap = {
            name: _safe_bootstrap(
                y,
                predictions[name],
                predictions["causal"],
                times,
                roots,
                kind=spec.kind,
                seed=args.seed + spec_i * 1000 + index,
                repetitions=args.bootstrap_repetitions,
            )
            for index, name in enumerate(
                ("model", "causal_plus_model", "residual_over_causal"), start=1
            )
        }
        summaries.append({
            "target": spec.name,
            "kind": spec.kind,
            "horizon_minutes": spec.horizon_minutes,
            "test_rows": int(len(y)),
            "unique_sample_rows": int(len(np.unique(rows))),
            "metrics": metrics,
            "bootstrap": bootstrap,
        })

    summary_by_target = {row["target"]: row for row in summaries}
    if set(PRIMARY_TARGETS).issubset(summary_by_target):
        verdict = screen_verdict([summary_by_target[name] for name in PRIMARY_TARGETS])
    else:
        verdict = {
            "policy": dict(SCREEN_POLICY),
            "downstream_screen_survived": False,
            "nonlinear_sensitivity_funded": False,
            "promotion_admitted": False,
            "full_training_admitted": False,
            "reason": "primary_target_closure_not_scored",
        }

    prediction_path = _atomic_npz(
        output_dir / "native_incremental_predictions.npz",
        **{
            name: np.concatenate(values) if values else np.asarray([])
            for name, values in prediction_rows.items()
        },
        target_names=np.asarray(available),
        arm_names=np.asarray(ARM_NAMES),
        feature_table_sha256=np.asarray(feature_manifest["artifact"]["sha256"]),
        fold_contract_sha256=np.asarray(fold_contract["contract_sha256"]),
    )
    repo = ROOT
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "oos_read": False,
        "route_key": feature_manifest["metadata"]["route_key"],
        "sample": {
            "path": sample_manifest["artifact"]["path"],
            "sha256": sample_manifest["artifact"]["sha256"],
            "content_fingerprint": sample_manifest["content_fingerprint"],
        },
        "row_selection": {
            "path": selection_manifest["artifact"]["path"],
            "sha256": selection_manifest["artifact"]["sha256"],
            "content_fingerprint": selection_manifest["content_fingerprint"],
        },
        "feature_table": {
            "path": feature_manifest["artifact"]["path"],
            "sha256": feature_manifest["artifact"]["sha256"],
            "content_fingerprint": feature_manifest["content_fingerprint"],
            "feature_kind": feature_manifest["metadata"]["feature_kind"],
            "feature_count": feature_manifest["metadata"]["feature_count"],
            "information_view": feature_manifest["metadata"]["information_view"],
        },
        "configuration": {
            "targets": list(selected_names),
            "folds": int(args.folds),
            "context_bars": int(args.context_bars),
            "expected_fold_sha256": args.expected_fold_sha256,
            "pca_components": int(args.pca_components),
            "seed": int(args.seed),
            "block_weights": bool(args.block_weights),
            "bootstrap_repetitions": int(args.bootstrap_repetitions),
            "linear_heads": {
                "regression": "ridge_lsqr_alpha1",
                "binary": "logistic_C1",
                "binary_residual": "probability_residual_ridge_clip_1e6",
            },
        },
        "fold_contract": fold_contract,
        "bottlenecks": [item["metadata"] for item in bottlenecks],
        "causal_feature_names": list(causal_names),
        "skipped_targets": skipped,
        "fold_scores": fold_scores,
        "summary": summaries,
        "screen_verdict": verdict,
        "predictions": {
            "path": str(prediction_path),
            "sha256": _sha256(prediction_path),
            "rows": int(sum(len(value) for value in prediction_rows["y_true"])),
        },
        "source": {
            "git_revision": subprocess.check_output(
                ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True,
            ).strip(),
            "working_tree_dirty": bool(subprocess.check_output(
                ["git", "-C", str(repo), "status", "--porcelain"], text=True,
            ).strip()),
            "implementation_sha256": {
                "ruler": _sha256(
                    repo / "futures_foundation/finetune/native_downstream_ruler.py"
                ),
                "probe": _sha256(
                    repo / "futures_foundation/finetune/downstream_probe.py"
                ),
                "runner": _sha256(Path(__file__).resolve()),
            },
        },
        "nonlinear_sensitivity_funded": bool(
            verdict.get("nonlinear_sensitivity_funded", False)
        ),
        "promotion_admitted": False,
        "full_training_admitted": False,
        "live_trading_ready": False,
    }
    report_path = _atomic_json(output_dir / "native_incremental_results.json", report)
    result = {
        "status": "complete",
        "route_key": report["route_key"],
        "targets": len(summaries),
        "prediction_rows": report["predictions"]["rows"],
        "downstream_screen_survived": verdict.get("downstream_screen_survived", False),
        "nonlinear_sensitivity_funded": report["nonlinear_sensitivity_funded"],
        "promotion_admitted": False,
        "report": str(report_path),
        "predictions": str(prediction_path),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--row-selection", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--targets", default="primary")
    parser.add_argument("--folds", type=int, default=2)
    parser.add_argument("--context-bars", type=int, default=512)
    parser.add_argument("--expected-fold-sha256", required=True)
    parser.add_argument("--pca-components", type=int, default=32)
    parser.add_argument("--bootstrap-repetitions", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--block-weights", action="store_true")
    args = parser.parse_args()
    if args.folds < 1 or args.context_bars < 1 or args.pca_components < 1:
        parser.error("folds, context-bars, and pca-components must be positive")
    if args.bootstrap_repetitions != SCREEN_POLICY["bootstrap_repetitions"]:
        parser.error(
            "bootstrap-repetitions must equal the frozen screen policy value "
            f"{SCREEN_POLICY['bootstrap_repetitions']}"
        )
    run(args)


if __name__ == "__main__":
    main()
