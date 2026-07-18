#!/usr/bin/env python3
"""Score one row-bound frozen embedding on the sealed Gate-3 targets and folds."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

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
from futures_foundation.finetune.native_contracts import verify_admission_report


SCHEMA_VERSION = "ffm_downstream_embedding_benchmark_v1"


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


def load_bound_embedding(
    path: str | Path,
    *,
    selection_manifest: dict,
    expected_rows: np.ndarray,
) -> tuple[np.ndarray, dict]:
    path = Path(path).resolve()
    manifest = json.loads(Path(str(path) + ".manifest.json").read_text())
    if manifest.get("oos_read") is not False or manifest.get("artifact", {}).get("sha256") != _sha256(path):
        raise ValueError("embedding artifact hash/OOS guard failed")
    if manifest.get("row_selection", {}).get("sha256") != selection_manifest["artifact"]["sha256"]:
        raise ValueError("embedding row-selection identity mismatch")
    with np.load(path, allow_pickle=False) as saved:
        if "embedding" not in saved.files or "row_index" not in saved.files:
            raise ValueError("embedding artifact lacks row identity")
        embedding = np.asarray(saved["embedding"], np.float32)
        row_index = np.asarray(saved["row_index"], np.int32)
        embedded_metadata = json.loads(str(saved["metadata"].item()))
    if not np.array_equal(row_index, np.asarray(expected_rows, np.int32)):
        raise ValueError("embedding rows differ from the sealed selection")
    if embedding.ndim != 2 or len(embedding) != len(row_index) or not np.isfinite(embedding).all():
        raise ValueError("embedding values are invalid or misaligned")
    for key in ("arm", "stage", "window_fingerprint"):
        if embedded_metadata.get(key) != manifest.get(key):
            raise ValueError(f"embedded/sidecar metadata differs for {key}")
    return embedding, manifest


def reduce_embedding_fold(
    embedding: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    *,
    max_components: int = 128,
    seed: int = 0,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Fit scaling/PCA on the outer training fold and transform only its later test fold."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    embedding = np.asarray(embedding, np.float32)
    train, test = np.asarray(train, np.int64), np.asarray(test, np.int64)
    components = min(int(max_components), embedding.shape[1], len(train) - 1)
    if components < 1:
        raise ValueError("embedding fold has too few rows/components")
    scaler = StandardScaler().fit(embedding[train])
    train_scaled = scaler.transform(embedding[train])
    test_scaled = scaler.transform(embedding[test])
    if components == embedding.shape[1]:
        train_value, test_value = train_scaled, test_scaled
        explained = 1.0
    else:
        pca = PCA(n_components=components, svd_solver="randomized", random_state=int(seed))
        train_value = pca.fit_transform(train_scaled)
        test_value = pca.transform(test_scaled)
        explained = float(pca.explained_variance_ratio_.sum())
    output = np.zeros((len(embedding), components), dtype=np.float32)
    output[train] = train_value
    output[test] = test_value
    return output, {"components": components, "explained_variance": explained}


def _aggregate(records: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for record in records:
        key = (record["timeframe"], record["target"], record["kind"], record["probe_arm"])
        groups.setdefault(key, []).append(record)
    output = []
    for (timeframe, target, kind, arm), rows in sorted(groups.items()):
        metric_names = ("r2", "mae", "spearman") if kind == "reg" else (
            "auc", "pr_auc", "brier", "prevalence",
        )
        metrics = {}
        for metric in metric_names:
            values = np.asarray([row["metrics"][metric] for row in rows], float)
            metrics[metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "fold_values": values.tolist(),
            }
        output.append({
            "timeframe": timeframe, "target": target, "kind": kind,
            "probe_arm": arm, "folds": len(rows),
            "test_rows": int(sum(row["test_rows"] for row in rows)), "metrics": metrics,
        })
    return output


def run(args) -> dict:
    embedding_sidecar = json.loads(Path(str(Path(args.embedding).resolve()) + ".manifest.json").read_text())
    stamped_admission = embedding_sidecar.get("admission")
    if not isinstance(stamped_admission, dict):
        raise ValueError("embedding lacks native-admission provenance")
    admission = verify_admission_report(
        args.admission_report,
        arm_key=str(embedding_sidecar.get("arm")),
        track=str(stamped_admission.get("track")),
        route=stamped_admission.get("route"),
        require_training=False,
    )
    if admission["integrity"] != stamped_admission.get("integrity"):
        raise ValueError("embedding admission provenance differs from the supplied current report")
    sample, sample_manifest = load_balanced_sample(args.sample)
    selection, selection_manifest = load_row_selection(
        args.row_selection, sample_manifest=sample_manifest,
    )
    selected_rows = np.asarray(selection["row_index"], np.int32)
    embedding, embedding_manifest = load_bound_embedding(
        args.embedding, selection_manifest=selection_manifest, expected_rows=selected_rows,
    )
    heads = tuple(value.strip() for value in args.heads.split(",") if value.strip())
    inputs = tuple(value.strip() for value in args.inputs.split(",") if value.strip())
    controls = tuple(value.strip() for value in args.controls.split(",") if value.strip())
    if not heads or set(heads) - {"linear", "xgb"}:
        raise ValueError("heads must be drawn from linear,xgb")
    if not inputs or set(inputs) - {"embedding", "embedding_plus_features"}:
        raise ValueError("inputs must be embedding and/or embedding_plus_features")
    if not controls or set(controls) - {"real", "time_destroyed"}:
        raise ValueError("controls must be real and/or time_destroyed")

    global_to_embedding = np.full(len(sample["stream_id"]), -1, np.int32)
    global_to_embedding[selected_rows] = np.arange(len(selected_rows), dtype=np.int32)
    target_names = [spec.name for spec in target_specs(sample)]
    probe_arms = [
        f"{embedding_manifest['arm']}:{embedding_manifest['stage']}:{input_name}:{head}:{control}"
        for input_name in inputs for head in heads for control in controls
    ]
    scores, folds_by_timeframe, skipped, reductions = [], {}, [], {}
    pred_row, pred_target, pred_arm, pred_fold, pred_y, pred_value = [], [], [], [], [], []
    for timeframe in sorted(str(value) for value in np.unique(sample["timeframe"])):
        timeframe_rows = selected_rows[sample["timeframe"][selected_rows] == timeframe]
        positions = global_to_embedding[timeframe_rows]
        if np.any(positions < 0):
            raise RuntimeError("timeframe rows are absent from the embedding")
        minutes = int(timeframe[:-3])
        splits, fold_contract = purged_calendar_splits(
            sample["decision_time_ns"][timeframe_rows],
            sample["label_end_time_ns"][timeframe_rows],
            sample["ticker"][timeframe_rows], folds=args.folds,
            embargo_ns=args.context_bars * minutes * 60 * 1_000_000_000,
        )
        folds_by_timeframe[timeframe] = fold_contract
        causal, _ = causal_feature_matrix(sample, timeframe_rows)
        local_embedding = embedding[positions]
        fold_matrices = {name: [] for name in inputs}
        reductions[timeframe] = []
        for fold_i, (train, test) in enumerate(splits, start=1):
            reduced, reduction = reduce_embedding_fold(
                local_embedding, train, test, max_components=args.max_components,
                seed=args.seed + fold_i,
            )
            reductions[timeframe].append({"fold": fold_i, **reduction})
            if "embedding" in inputs:
                fold_matrices["embedding"].append(reduced)
            if "embedding_plus_features" in inputs:
                fold_matrices["embedding_plus_features"].append(
                    np.column_stack((reduced, causal)).astype(np.float32),
                )
        groups = sample["ticker"][timeframe_rows]
        if not args.quiet:
            print(
                f"[{embedding_manifest['arm']}:{embedding_manifest['stage']}:{timeframe}] "
                f"rows={len(timeframe_rows):,} dim={embedding.shape[1]}", flush=True,
            )
        for spec_i, spec in enumerate(target_specs(sample)):
            y_global, valid_global = target_values(sample, spec)
            y, valid = y_global[timeframe_rows], valid_global[timeframe_rows]
            if not np.any(valid):
                skipped.append({
                    "timeframe": timeframe, "target": spec.name,
                    "reason": "fewer_than_two_forward_returns",
                })
                continue
            for input_name in inputs:
                for head in heads:
                    for control in controls:
                        probe_arm = (
                            f"{embedding_manifest['arm']}:{embedding_manifest['stage']}:"
                            f"{input_name}:{head}:{control}"
                        )
                        arm_i = probe_arms.index(probe_arm)
                        for fold_i, (train, test) in enumerate(splits, start=1):
                            X = fold_matrices[input_name][fold_i - 1]
                            train, test = train[valid[train]], test[valid[test]]
                            issue = fold_target_issue(y, train, test, spec.kind)
                            if issue:
                                skipped.append({
                                    "timeframe": timeframe, "target": spec.name,
                                    "probe_arm": probe_arm, "fold": fold_i,
                                    "train_rows": int(len(train)), "test_rows": int(len(test)),
                                    "reason": issue,
                                })
                                continue
                            prediction = fit_predict_fold(
                                X, y, groups, train, test, kind=spec.kind, head=head,
                                control=control,
                                seed=args.seed + spec_i * 100 + fold_i,
                            )
                            metrics = prediction_metrics(y[test], prediction, spec.kind)
                            scores.append({
                                "timeframe": timeframe, "target": spec.name, "kind": spec.kind,
                                "horizon_minutes": spec.horizon_minutes,
                                "probe_arm": probe_arm, "fold": fold_i,
                                "train_rows": int(len(train)), "test_rows": int(len(test)),
                                "metrics": metrics,
                            })
                            pred_row.append(timeframe_rows[test].astype(np.int32))
                            pred_target.append(np.full(len(test), spec_i, np.int16))
                            pred_arm.append(np.full(len(test), arm_i, np.int8))
                            pred_fold.append(np.full(len(test), fold_i, np.int8))
                            pred_y.append(np.asarray(y[test], np.float32))
                            pred_value.append(np.asarray(prediction, np.float32))
            primary = "r2" if spec.kind == "reg" else "auc"
            current = [
                row["metrics"][primary] for row in scores
                if row["timeframe"] == timeframe and row["target"] == spec.name
                and row["probe_arm"] == probe_arms[0]
            ]
            if current and not args.quiet:
                print(f"  {spec.name}: {primary}={np.mean(current):.4f}", flush=True)

    output_dir = Path(args.output_dir).resolve()
    stem = f"{embedding_manifest['arm']}__{embedding_manifest['stage']}"
    prediction_path = output_dir / f"{stem}.predictions.npz"
    _atomic_npz(
        prediction_path, row_index=np.concatenate(pred_row),
        target_index=np.concatenate(pred_target), arm_index=np.concatenate(pred_arm),
        fold=np.concatenate(pred_fold), y_true=np.concatenate(pred_y),
        prediction=np.concatenate(pred_value), target_names=np.asarray(target_names),
        probe_arms=np.asarray(probe_arms),
        embedding_sha256=np.asarray(embedding_manifest["artifact"]["sha256"]),
    )
    report = {
        "schema_version": SCHEMA_VERSION, "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(), "oos_read": False,
        "sample_sha256": sample_manifest["artifact"]["sha256"],
        "row_selection_sha256": selection_manifest["artifact"]["sha256"],
        "admission": {
            "integrity": admission["integrity"],
            "registry_sha256": admission["registry_sha256"],
            "dossier_sha256": admission["dossier_sha256"],
        },
        "embedding": {
            "path": str(Path(args.embedding).resolve()),
            "sha256": embedding_manifest["artifact"]["sha256"],
            "arm": embedding_manifest["arm"], "stage": embedding_manifest["stage"],
            "shape": list(embedding.shape),
        },
        "configuration": {
            "heads": list(heads), "inputs": list(inputs), "controls": list(controls),
            "folds": args.folds, "context_bars": args.context_bars, "seed": args.seed,
            "embedding_preprocessing": {
                "method": "train_fold_standardize_then_pca",
                "max_components": args.max_components,
            },
        },
        "fold_contracts": folds_by_timeframe, "reductions": reductions,
        "skipped_targets": skipped,
        "fold_scores": scores, "summary": _aggregate(scores),
        "predictions": {
            "path": str(prediction_path), "sha256": _sha256(prediction_path),
            "rows": int(sum(len(value) for value in pred_row)),
        },
    }
    report_path = output_dir / f"{stem}.results.json"
    _atomic_json(report_path, report)
    print(json.dumps({
        "status": "complete", "report": str(report_path),
        "prediction_rows": report["predictions"]["rows"],
    }, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding", required=True)
    parser.add_argument("--admission-report", required=True)
    parser.add_argument(
        "--sample", default="output/foundation_tournament/downstream_gate_v1/balanced_sample.npz",
    )
    parser.add_argument(
        "--row-selection",
        default="output/foundation_tournament/downstream_gate_v1/representation_rows.npz",
    )
    parser.add_argument(
        "--output-dir", default="output/foundation_tournament/downstream_gate_v1/screen/scores",
    )
    parser.add_argument("--heads", default="linear")
    parser.add_argument("--inputs", default="embedding")
    parser.add_argument("--controls", default="real")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--context-bars", type=int, default=256)
    parser.add_argument("--max-components", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
