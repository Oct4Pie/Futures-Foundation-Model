#!/usr/bin/env python3
"""Matched realized-R screen for causal features and finalist representation fusions."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from futures_foundation.finetune.downstream_probe import causal_feature_matrix, fit_predict_fold
from futures_foundation.finetune.downstream_sample import (
    load_balanced_sample,
    load_row_selection,
    purged_calendar_splits,
)
from futures_foundation.finetune.downstream_trading import load_policy_events
from scripts.benchmark_downstream_embedding import load_bound_embedding, reduce_embedding_fold


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


def apply_concurrency(events: dict, rows: np.ndarray, selected: np.ndarray) -> np.ndarray:
    """Execute selected signals under one active trade per ticker/policy/timeframe."""
    rows = np.asarray(rows, np.int64)
    selected = np.asarray(selected, bool)
    if len(rows) != len(selected):
        raise ValueError("rows/selection length mismatch")
    executed = np.zeros(len(rows), bool)
    order = np.argsort(events["signal_time_ns"][rows], kind="stable")
    active_until: dict[str, int] = {}
    for position in order:
        if not selected[position]:
            continue
        event = int(rows[position])
        ticker = str(events["ticker"][event])
        signal = int(events["signal_time_ns"][event])
        if signal <= active_until.get(ticker, -1):
            continue
        executed[position] = True
        active_until[ticker] = int(events["exit_time_ns"][event])
    return executed


def trade_metrics(realized: np.ndarray, reached: np.ndarray, signal_time_ns: np.ndarray) -> dict:
    realized = np.asarray(realized, float)
    reached = np.asarray(reached, bool)
    time = np.asarray(signal_time_ns, np.int64)
    if not len(realized):
        return {
            "trades": 0, "win_rate": None, "target_hit_rate": None, "mean_r": None,
            "median_r": None, "profit_factor": None, "total_r": None,
            "max_drawdown_r": None,
        }
    order = np.argsort(time, kind="stable")
    values = realized[order]
    equity = np.r_[0.0, np.cumsum(values)]
    losses, wins = values[values < 0], values[values > 0]
    gross_loss = -float(losses.sum())
    return {
        "trades": int(len(values)), "win_rate": float(np.mean(values > 0)),
        "target_hit_rate": float(reached.mean()), "mean_r": float(values.mean()),
        "median_r": float(np.median(values)),
        "profit_factor": float(wins.sum() / gross_loss) if gross_loss > 0 else None,
        "total_r": float(values.sum()),
        "max_drawdown_r": float(np.max(np.maximum.accumulate(equity) - equity)),
    }


def policy_feature_matrix(events: dict, rows: np.ndarray) -> tuple[np.ndarray, tuple[str, ...]]:
    """Known-at-decision execution geometry shared by every learned selector arm."""
    rows = np.asarray(rows, np.int64)
    names = (
        "event_direction", "log1p_risk_ticks", "slippage_r", "fee_r", "total_cost_r",
    )
    values = np.column_stack((
        np.asarray(events["direction"][rows], np.float32),
        np.log1p(np.asarray(events["risk_ticks"][rows], np.float32)),
        np.asarray(events["slippage_r"][rows], np.float32),
        np.asarray(events["fee_r"][rows], np.float32),
        np.asarray(events["total_cost_r"][rows], np.float32),
    )).astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("non-finite policy execution feature")
    return values, names


def _parse_csv_numbers(value: str, cast):
    return tuple(cast(item.strip()) for item in value.split(",") if item.strip())


def stable_policy_seed(base_seed: int, policy: str, fold: int) -> int:
    """Derive a policy/fold seed that is invariant to filtering and list order."""
    payload = f"{int(base_seed)}\0{policy}\0{int(fold)}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    return int(value % (2**31 - 1))


def inner_calibration_rows(
    events: dict,
    policy_event_rows: np.ndarray,
    outer_train: np.ndarray,
    *,
    fraction: float = 0.2,
    min_fit: int = 100,
    min_calibration: int = 50,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Split an outer-training fold chronologically and purge labels crossing calibration."""
    policy_event_rows = np.asarray(policy_event_rows, np.int64)
    outer_train = np.asarray(outer_train, np.int64)
    fraction = float(fraction)
    if not 0 < fraction < 0.5 or len(outer_train) < min_fit + min_calibration:
        raise ValueError("insufficient rows or invalid inner calibration fraction")
    signal = np.asarray(events["signal_time_ns"][policy_event_rows[outer_train]], np.int64)
    ordered = np.sort(signal, kind="stable")
    position = min(len(ordered) - 1, max(1, int(np.floor((1.0 - fraction) * len(ordered)))))
    calibration_start = int(ordered[position])
    exit_time = np.asarray(events["exit_time_ns"][policy_event_rows[outer_train]], np.int64)
    fit = outer_train[exit_time < calibration_start]
    calibration = outer_train[signal >= calibration_start]
    if len(fit) < min_fit or len(calibration) < min_calibration:
        raise ValueError("inner calibration split is too small after label-end purge")
    if np.any(events["exit_time_ns"][policy_event_rows[fit]] >= calibration_start):
        raise RuntimeError("inner calibration label purge failed")
    if np.any(events["signal_time_ns"][policy_event_rows[calibration]] < calibration_start):
        raise RuntimeError("inner calibration chronology failed")
    return fit, calibration, calibration_start


def choose_calibrated_threshold(
    events: dict,
    policy_event_rows: np.ndarray,
    calibration_rows: np.ndarray,
    score: np.ndarray,
    *,
    quantiles: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9),
    min_executed: int = 20,
    floor_threshold: float = 0.0,
) -> dict:
    """Choose net-R threshold on an earlier calibration slice using R per opportunity."""
    policy_event_rows = np.asarray(policy_event_rows, np.int64)
    calibration_rows = np.asarray(calibration_rows, np.int64)
    score = np.asarray(score, np.float64)
    if len(calibration_rows) != len(score) or not len(score) or not np.isfinite(score).all():
        raise ValueError("calibration scores must be finite, non-empty, and aligned")
    quantiles = tuple(float(value) for value in quantiles)
    if any(not 0 < value < 1 for value in quantiles) or min_executed < 1:
        raise ValueError("invalid threshold calibration configuration")
    floor_threshold = float(floor_threshold)
    if not np.isfinite(floor_threshold):
        raise ValueError("threshold floor must be finite")
    candidates = np.unique(np.r_[floor_threshold, np.quantile(score, quantiles)])
    candidates = candidates[candidates >= floor_threshold]
    event_rows = policy_event_rows[calibration_rows]
    outcomes = np.asarray(events["realized_r"][event_rows], np.float64)
    viable = []
    for threshold in candidates:
        selected = score > threshold
        executed = apply_concurrency(events, event_rows, selected)
        count = int(executed.sum())
        if count < min_executed:
            continue
        total = float(outcomes[executed].sum())
        viable.append({
            "threshold": float(threshold),
            "selected": int(selected.sum()),
            "executed": count,
            "total_r": total,
            "r_per_candidate": total / len(score),
            "mean_executed_r": total / count,
        })
    if not viable:
        raise ValueError("no threshold candidate meets the calibration execution floor")
    return max(
        viable,
        key=lambda row: (row["r_per_candidate"], row["executed"], -abs(row["threshold"])),
    )


def run(args) -> dict:
    sample, sample_manifest = load_balanced_sample(args.sample)
    selection, selection_manifest = load_row_selection(
        args.row_selection, sample_manifest=sample_manifest,
    )
    selected_rows = np.asarray(selection["row_index"], np.int32)
    events, policy_manifest = load_policy_events(args.policy_events)
    if policy_manifest.get("sample_sha256") != sample_manifest["artifact"]["sha256"]:
        raise ValueError("policy/sample identity mismatch")
    if policy_manifest.get("row_selection_sha256") != selection_manifest["artifact"]["sha256"]:
        raise ValueError("policy/row-selection identity mismatch")
    if not np.all(np.isin(events["context_row"], selected_rows)):
        raise ValueError("policy event is outside the sealed row selection")

    embedding_paths = tuple(Path(value).resolve() for value in args.embedding)
    embeddings, embedding_manifests = {}, {}
    for path in embedding_paths:
        value, manifest = load_bound_embedding(
            path, selection_manifest=selection_manifest, expected_rows=selected_rows,
        )
        name = f"{manifest['arm']}:{manifest['stage']}"
        if name in embeddings:
            raise ValueError(f"duplicate embedding arm: {name}")
        embeddings[name], embedding_manifests[name] = value, manifest
    arm_names = ["raw_all", "causal_xgb"] + [f"{name}:fusion_xgb" for name in embeddings]

    horizons = set(_parse_csv_numbers(args.horizons, int))
    targets = set(_parse_csv_numbers(args.targets, float))
    event_horizon = sample["horizons_minutes"][events["horizon_index"]]
    event_target = sample["targets_r"][events["target_index"]]
    eligible_event = np.isin(event_horizon, tuple(horizons)) & np.isin(event_target, tuple(targets))
    global_to_embedding = np.full(len(sample["stream_id"]), -1, np.int32)
    global_to_embedding[selected_rows] = np.arange(len(selected_rows), dtype=np.int32)

    prediction_parts = {key: [] for key in (
        "event_row", "policy_index", "arm_index", "fold", "score", "decision_threshold",
        "selected", "executed",
    )}
    fold_scores, skipped, fold_contracts = [], [], {}
    policy_names = sorted(str(value) for value in np.unique(events["policy_key"][eligible_event]))
    for timeframe in sorted(str(value) for value in np.unique(sample["timeframe"][selected_rows])):
        context_rows = selected_rows[sample["timeframe"][selected_rows] == timeframe]
        minutes = int(timeframe[:-3])
        splits, contract = purged_calendar_splits(
            sample["decision_time_ns"][context_rows], sample["label_end_time_ns"][context_rows],
            sample["ticker"][context_rows], folds=args.folds,
            embargo_ns=args.context_bars * minutes * 60 * 1_000_000_000,
        )
        fold_contracts[timeframe] = contract
        causal, _ = causal_feature_matrix(sample, context_rows)
        global_to_local = np.full(len(sample["stream_id"]), -1, np.int32)
        global_to_local[context_rows] = np.arange(len(context_rows), dtype=np.int32)
        event_pool = np.flatnonzero(eligible_event & (events["timeframe"] == timeframe))
        event_local = global_to_local[events["context_row"][event_pool]]
        if np.any(event_local < 0):
            raise RuntimeError("event/context timeframe alignment failed")

        matrices = {"causal_xgb": []}
        matrices.update({f"{name}:fusion_xgb": [] for name in embeddings})
        for fold_index, (train, test) in enumerate(splits, start=1):
            matrices["causal_xgb"].append(causal)
            for name, embedding in embeddings.items():
                positions = global_to_embedding[context_rows]
                reduced, _ = reduce_embedding_fold(
                    embedding[positions], train, test, max_components=args.max_components,
                    seed=args.seed + fold_index,
                )
                matrices[f"{name}:fusion_xgb"].append(
                    np.column_stack((reduced, causal)).astype(np.float32)
                )

        for policy_index, policy in enumerate(policy_names):
            local_policy_rows = np.flatnonzero(events["policy_key"][event_pool] == policy)
            if not len(local_policy_rows):
                continue
            policy_event_rows = event_pool[local_policy_rows]
            policy_context_local = event_local[local_policy_rows]
            policy_features, policy_feature_names = policy_feature_matrix(events, policy_event_rows)
            y = np.asarray(events["realized_r"][policy_event_rows], np.float32)
            groups = events["ticker"][policy_event_rows]
            for fold_index, (train_context, test_context) in enumerate(splits, start=1):
                role = np.zeros(len(context_rows), np.int8)
                role[train_context], role[test_context] = 1, 2
                train = np.flatnonzero(role[policy_context_local] == 1)
                test = np.flatnonzero(role[policy_context_local] == 2)
                if len(train) < args.min_train or len(test) < args.min_test:
                    skipped.append({
                        "timeframe": timeframe, "policy": policy, "fold": fold_index,
                        "train_rows": int(len(train)), "test_rows": int(len(test)),
                        "reason": "insufficient_policy_rows",
                    })
                    continue

                raw_selected = np.ones(len(test), bool)
                raw_executed = apply_concurrency(events, policy_event_rows[test], raw_selected)
                test_event_rows = policy_event_rows[test]
                for arm_index, arm in enumerate(arm_names):
                    calibration_record = {
                        "threshold_mode": "raw" if arm == "raw_all" else args.threshold_mode,
                        "decision_threshold": None,
                        "calibration_fit_rows": 0,
                        "calibration_rows": 0,
                        "calibration_executed": 0,
                        "calibration_r_per_candidate": None,
                    }
                    if arm == "raw_all":
                        score = np.full(len(test), np.nan, np.float32)
                        selected, executed = raw_selected, raw_executed
                        decision_threshold = np.nan
                    else:
                        matrix = np.column_stack((
                            matrices[arm][fold_index - 1][policy_context_local], policy_features,
                        )).astype(np.float32)
                        decision_threshold = float(args.threshold)
                        if args.threshold_mode == "inner_calibrated":
                            try:
                                inner_fit, calibration, calibration_start = inner_calibration_rows(
                                    events, policy_event_rows, train,
                                    fraction=args.calibration_fraction,
                                    min_fit=args.min_calibration_fit,
                                    min_calibration=args.min_calibration_rows,
                                )
                                calibration_score = fit_predict_fold(
                                    matrix, y, groups, inner_fit, calibration,
                                    kind="reg", head="xgb", control="real",
                                    seed=stable_policy_seed(
                                        args.seed, policy + ":calibration", fold_index,
                                    ),
                                )
                                calibrated = choose_calibrated_threshold(
                                    events, policy_event_rows, calibration, calibration_score,
                                    quantiles=args.threshold_quantiles,
                                    min_executed=args.min_calibration_executed,
                                    floor_threshold=args.threshold,
                                )
                                decision_threshold = calibrated["threshold"]
                                calibration_record.update({
                                    "decision_threshold": decision_threshold,
                                    "calibration_start_ns": calibration_start,
                                    "calibration_fit_rows": int(len(inner_fit)),
                                    "calibration_rows": int(len(calibration)),
                                    "calibration_executed": calibrated["executed"],
                                    "calibration_r_per_candidate": calibrated["r_per_candidate"],
                                })
                            except ValueError as error:
                                calibration_record["threshold_mode"] = "fixed_fallback"
                                calibration_record["fallback_reason"] = str(error)
                        score = fit_predict_fold(
                            matrix, y, groups, train, test, kind="reg", head="xgb",
                            control="real",
                            seed=stable_policy_seed(args.seed, policy, fold_index),
                        )
                        selected = np.asarray(score > decision_threshold)
                        executed = apply_concurrency(events, test_event_rows, selected)
                    executed_rows = test_event_rows[executed]
                    metrics = trade_metrics(
                        events["realized_r"][executed_rows], events["reached"][executed_rows],
                        events["signal_time_ns"][executed_rows],
                    )
                    metrics.update({
                        "candidates": int(len(test)), "selected": int(selected.sum()),
                        "executed": int(executed.sum()),
                        "selection_rate": float(selected.mean()),
                        **calibration_record,
                    })
                    fold_scores.append({
                        "timeframe": timeframe, "policy": policy, "arm": arm,
                        "fold": fold_index, **metrics,
                    })
                    prediction_parts["event_row"].append(test_event_rows.astype(np.int32))
                    prediction_parts["policy_index"].append(
                        np.full(len(test), policy_index, np.int16)
                    )
                    prediction_parts["arm_index"].append(np.full(len(test), arm_index, np.int8))
                    prediction_parts["fold"].append(np.full(len(test), fold_index, np.int8))
                    prediction_parts["score"].append(np.asarray(score, np.float32))
                    prediction_parts["decision_threshold"].append(
                        np.full(len(test), decision_threshold, np.float32)
                    )
                    prediction_parts["selected"].append(np.asarray(selected, bool))
                    prediction_parts["executed"].append(np.asarray(executed, bool))
        print(f"[{timeframe}] events={len(event_pool):,}", flush=True)

    output_dir = Path(args.output_dir).resolve()
    prediction_path = output_dir / "trading_predictions.npz"
    _atomic_npz(
        prediction_path,
        **{key: np.concatenate(value) for key, value in prediction_parts.items()},
        policy_names=np.asarray(policy_names), arm_names=np.asarray(arm_names),
        policy_events_sha256=np.asarray(policy_manifest["artifact"]["sha256"]),
    )
    with np.load(prediction_path, allow_pickle=False) as saved:
        saved_predictions = {key: saved[key] for key in saved.files}
    summaries = []
    for policy_index, policy in enumerate(policy_names):
        for arm_index, arm in enumerate(arm_names):
            rows = np.flatnonzero(
                (saved_predictions["policy_index"] == policy_index)
                & (saved_predictions["arm_index"] == arm_index)
            )
            if not len(rows):
                continue
            event_rows = saved_predictions["event_row"][rows]
            executed = saved_predictions["executed"][rows]
            executed_events = event_rows[executed]
            metrics = trade_metrics(
                events["realized_r"][executed_events], events["reached"][executed_events],
                events["signal_time_ns"][executed_events],
            )
            tickers = np.unique(events["ticker"][executed_events]) if len(executed_events) else []
            timeframes = np.unique(events["timeframe"][executed_events]) if len(executed_events) else []
            metrics.update({
                "policy": policy, "arm": arm, "candidates": int(len(rows)),
                "selected": int(saved_predictions["selected"][rows].sum()),
                "executed": int(executed.sum()),
                "selection_rate": float(saved_predictions["selected"][rows].mean()),
                "ticker_breadth": int(len(tickers)), "timeframe_breadth": int(len(timeframes)),
            })
            summaries.append(metrics)
    report = {
        "schema_version": "ffm_downstream_trading_benchmark_v1",
        "status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
        "oos_read": False,
        "sample_sha256": sample_manifest["artifact"]["sha256"],
        "row_selection_sha256": selection_manifest["artifact"]["sha256"],
        "policy_events_sha256": policy_manifest["artifact"]["sha256"],
        "configuration": {
            "horizons_minutes": sorted(horizons), "targets_r": sorted(targets),
            "folds": args.folds, "context_bars": args.context_bars,
            "threshold": args.threshold,
            "threshold_contract": (
                "predicted net R > fixed floor; inner calibration may only raise the threshold"
                if args.threshold_mode == "inner_calibrated"
                else "fixed predicted net R > threshold"
            ),
            "threshold_mode": args.threshold_mode,
            "calibration_fraction": args.calibration_fraction,
            "min_calibration_fit": args.min_calibration_fit,
            "min_calibration_rows": args.min_calibration_rows,
            "min_calibration_executed": args.min_calibration_executed,
            "threshold_quantiles": list(args.threshold_quantiles),
            "execution": "one active trade per policy/ticker/timeframe",
            "head": "constrained_xgb_regression", "min_train": args.min_train,
            "min_test": args.min_test, "max_components": args.max_components,
            "policy_features": list(policy_feature_names),
            "screen_sampling_limit": (
                "chronologically sampled contexts; relative lift/breadth only, not absolute annual quantity"
            ),
        },
        "embeddings": {
            name: {"path": str(embedding_paths[index]),
                   "sha256": embedding_manifests[name]["artifact"]["sha256"]}
            for index, name in enumerate(embeddings)
        },
        "fold_contracts": fold_contracts, "skipped": skipped,
        "fold_scores": fold_scores, "summary": summaries,
        "predictions": {"path": str(prediction_path), "sha256": _sha256(prediction_path),
                        "rows": int(len(saved_predictions["event_row"]))},
    }
    report_path = output_dir / "trading_results.json"
    _atomic_json(report_path, report)
    print(json.dumps({
        "status": "complete", "policies": len(policy_names), "arms": len(arm_names),
        "prediction_rows": report["predictions"]["rows"], "report": str(report_path),
    }, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample", default="output/foundation_tournament/downstream_gate_v1/balanced_sample.npz",
    )
    parser.add_argument(
        "--row-selection", default="output/foundation_tournament/downstream_gate_v1/representation_rows.npz",
    )
    parser.add_argument(
        "--policy-events", default="output/foundation_tournament/downstream_gate_v1/screen/policy_events.npz",
    )
    parser.add_argument("--embedding", action="append", default=[])
    parser.add_argument(
        "--output-dir", default="output/foundation_tournament/downstream_gate_v1/screen/trading",
    )
    parser.add_argument("--horizons", default="360")
    parser.add_argument("--targets", default="3")
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument(
        "--threshold-mode", choices=("fixed", "inner_calibrated"), default="fixed",
    )
    parser.add_argument("--calibration-fraction", type=float, default=0.2)
    parser.add_argument("--min-calibration-fit", type=int, default=100)
    parser.add_argument("--min-calibration-rows", type=int, default=50)
    parser.add_argument("--min-calibration-executed", type=int, default=20)
    parser.add_argument("--threshold-quantiles", default="0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--context-bars", type=int, default=256)
    parser.add_argument("--max-components", type=int, default=128)
    parser.add_argument("--min-train", type=int, default=100)
    parser.add_argument("--min-test", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260716)
    args = parser.parse_args()
    args.threshold_quantiles = _parse_csv_numbers(args.threshold_quantiles, float)
    run(args)


if __name__ == "__main__":
    main()
