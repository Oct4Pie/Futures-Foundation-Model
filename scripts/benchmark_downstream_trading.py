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
from futures_foundation.finetune.calibration import (
    apply_isotonic_expected_value,
    fit_isotonic_expected_value,
)
from futures_foundation.finetune.downstream_sample import (
    load_balanced_sample,
    load_row_selection,
    purged_calendar_splits,
    purged_interval_splits,
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


def _parse_csv_strings(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(value).split(",") if item.strip())


def stable_policy_seed(base_seed: int, policy: str, fold: int) -> int:
    """Derive a policy/fold seed that is invariant to filtering and list order."""
    payload = f"{int(base_seed)}\0{policy}\0{int(fold)}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    return int(value % (2**31 - 1))


def nested_context_splits(
    decision_time_ns: np.ndarray,
    label_end_time_ns: np.ndarray,
    group_ids: np.ndarray,
    outer_train: np.ndarray,
    *,
    folds: int,
    embargo_ns: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict]:
    """Build expanding inner folds entirely inside one outer-training fold."""
    outer_train = np.asarray(outer_train, np.int64)
    if len(outer_train) == 0:
        raise ValueError("outer training fold is empty")
    local, contract = purged_calendar_splits(
        np.asarray(decision_time_ns)[outer_train],
        np.asarray(label_end_time_ns)[outer_train],
        np.asarray(group_ids)[outer_train],
        folds=int(folds), embargo_ns=int(embargo_ns),
    )
    mapped = [(outer_train[train], outer_train[test]) for train, test in local]
    outer_set = set(outer_train.tolist())
    if any(
        not set(train.tolist()).issubset(outer_set) or not set(test.tolist()).issubset(outer_set)
        for train, test in mapped
    ):
        raise RuntimeError("inner fold escaped the outer training set")
    return mapped, contract


def context_matrix_for_fold(
    causal: np.ndarray,
    embedding: np.ndarray | None,
    train_context: np.ndarray,
    test_context: np.ndarray,
    *,
    max_components: int,
    seed: int,
) -> np.ndarray:
    """Fit any embedding reduction on the declared training contexts only."""
    causal = np.asarray(causal, np.float32)
    if embedding is None:
        return causal
    reduced, _ = reduce_embedding_fold(
        np.asarray(embedding, np.float32), train_context, test_context,
        max_components=max_components, seed=seed,
    )
    return np.column_stack((reduced, causal)).astype(np.float32)


def nested_oof_predictions(
    context_matrices: list[np.ndarray],
    context_splits: list[tuple[np.ndarray, np.ndarray]],
    policy_context_rows: np.ndarray,
    policy_features: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    head: str,
    seed: int,
    min_train: int,
    min_test: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Generate strictly earlier-fold predictions for calibrator and threshold fitting."""
    policy_context_rows = np.asarray(policy_context_rows, np.int64)
    policy_features = np.asarray(policy_features, np.float32)
    y, groups = np.asarray(y), np.asarray(groups)
    if len(context_matrices) != len(context_splits):
        raise ValueError("inner matrices/splits are misaligned")
    rows_out, score_out, fold_out, records = [], [], [], []
    for inner_fold, ((train_context, test_context), context_matrix) in enumerate(
        zip(context_splits, context_matrices), start=1,
    ):
        role = np.zeros(len(context_matrix), np.int8)
        role[np.asarray(train_context, np.int64)] = 1
        role[np.asarray(test_context, np.int64)] = 2
        train = np.flatnonzero(role[policy_context_rows] == 1)
        test = np.flatnonzero(role[policy_context_rows] == 2)
        record = {
            "fold": inner_fold, "train_rows": int(len(train)), "test_rows": int(len(test)),
        }
        if len(train) < int(min_train) or len(test) < int(min_test):
            record["status"] = "skipped_insufficient_rows"
            records.append(record)
            continue
        matrix = np.column_stack((context_matrix[policy_context_rows], policy_features)).astype(
            np.float32,
        )
        score = fit_predict_fold(
            matrix, y, groups, train, test, kind="reg", head=head, control="real",
            seed=int(seed) + inner_fold,
        )
        rows_out.append(test)
        score_out.append(score)
        fold_out.append(np.full(len(test), inner_fold, np.int8))
        record["status"] = "complete"
        records.append(record)
    if not rows_out:
        raise ValueError("no inner fold produced calibration predictions")
    rows = np.concatenate(rows_out)
    if len(np.unique(rows)) != len(rows):
        raise RuntimeError("inner OOF rows overlap")
    return rows, np.concatenate(score_out), np.concatenate(fold_out), records


def choose_stable_calibrated_threshold(
    events: dict,
    policy_event_rows: np.ndarray,
    calibration_rows: np.ndarray,
    calibrated_score: np.ndarray,
    calibration_folds: np.ndarray,
    *,
    quantiles: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9),
    min_executed: int = 20,
    min_coverage: float = 0.02,
    floor_threshold: float = 0.0,
    lcb_z: float = 1.0,
) -> dict:
    """Select a calibrated expected-R cutoff using chronological-fold stability.

    The objective is R per candidate, so sparse lucky subsets are penalized. If no cutoff has a
    positive lower confidence bound, the declared action is no-trade rather than falling back to an
    unvalidated zero cutoff.
    """
    policy_event_rows = np.asarray(policy_event_rows, np.int64)
    calibration_rows = np.asarray(calibration_rows, np.int64)
    score = np.asarray(calibrated_score, np.float64)
    fold_id = np.asarray(calibration_folds, np.int64)
    if (
        len(calibration_rows) != len(score) or len(score) != len(fold_id) or not len(score)
        or not np.isfinite(score).all()
    ):
        raise ValueError("calibrated scores, rows, and folds must be finite and aligned")
    if not 0 <= min_coverage < 1 or min_executed < 1 or lcb_z < 0:
        raise ValueError("invalid stable-threshold constraints")
    candidates = np.unique(np.r_[float(floor_threshold), np.quantile(score, quantiles)])
    candidates = candidates[candidates >= float(floor_threshold)]
    event_rows = policy_event_rows[calibration_rows]
    outcomes = np.asarray(events["realized_r"][event_rows], np.float64)
    viable = []
    for threshold in candidates:
        fold_values, executed_total, selected_total = [], 0, 0
        for fold in np.unique(fold_id):
            rows = np.flatnonzero(fold_id == fold)
            selected = score[rows] > threshold
            executed = apply_concurrency(events, event_rows[rows], selected)
            selected_total += int(selected.sum())
            executed_total += int(executed.sum())
            fold_values.append(float(outcomes[rows][executed].sum()) / len(rows))
        coverage = executed_total / len(score)
        if executed_total < int(min_executed) or coverage < float(min_coverage):
            continue
        values = np.asarray(fold_values, np.float64)
        mean = float(values.mean())
        se = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else float("inf")
        viable.append({
            "threshold": float(threshold), "selected": selected_total,
            "executed": executed_total, "coverage": float(coverage),
            "fold_r_per_candidate": values.tolist(), "mean_r_per_candidate": mean,
            "standard_error": se, "lcb_r_per_candidate": mean - float(lcb_z) * se,
            "positive_fold_fraction": float(np.mean(values > 0)),
        })
    positive = [row for row in viable if row["lcb_r_per_candidate"] > 0]
    if positive:
        return max(
            positive,
            key=lambda row: (
                row["lcb_r_per_candidate"], row["mean_r_per_candidate"], row["executed"],
            ),
        ) | {"no_trade": False}
    # The isotonic map clips future scores to its learned range, so the next finite float above the
    # calibration maximum is an explicit no-trade threshold for the outer fold.
    no_trade_threshold = float(np.nextafter(float(score.max()), np.inf))
    return {
        "threshold": no_trade_threshold, "selected": 0, "executed": 0,
        "coverage": 0.0, "fold_r_per_candidate": [], "mean_r_per_candidate": 0.0,
        "standard_error": None, "lcb_r_per_candidate": None,
        "positive_fold_fraction": 0.0, "no_trade": True,
        "reason": "no candidate threshold has positive chronological-fold LCB",
        "viable_candidates": len(viable),
    }


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
        "event_row", "policy_index", "arm_index", "fold", "raw_score", "score",
        "decision_threshold", "selected", "executed",
    )}
    fold_scores, skipped, fold_contracts, inner_fold_contracts = [], [], {}, {}
    available_policies = set(str(value) for value in np.unique(events["policy_key"][eligible_event]))
    requested_policies = set(args.policies)
    missing_policies = sorted(requested_policies - available_policies)
    if missing_policies:
        raise ValueError(f"requested policies are absent after horizon/target filtering: {missing_policies}")
    policy_names = sorted(requested_policies)
    for timeframe in sorted(str(value) for value in np.unique(sample["timeframe"][selected_rows])):
        context_rows = selected_rows[sample["timeframe"][selected_rows] == timeframe]
        minutes = int(timeframe[:-3])
        split_args = {
            "folds": args.folds,
            "embargo_ns": args.context_bars * minutes * 60 * 1_000_000_000,
        }
        if args.outer_eval_start is not None:
            splits, contract = purged_interval_splits(
                sample["decision_time_ns"][context_rows],
                sample["label_end_time_ns"][context_rows],
                sample["ticker"][context_rows],
                eval_start_ns=args.outer_eval_start, eval_end_ns=args.outer_eval_end,
                **split_args,
            )
        else:
            splits, contract = purged_calendar_splits(
                sample["decision_time_ns"][context_rows],
                sample["label_end_time_ns"][context_rows],
                sample["ticker"][context_rows], **split_args,
            )
        fold_contracts[timeframe] = contract
        causal, _ = causal_feature_matrix(sample, context_rows)
        global_to_local = np.full(len(sample["stream_id"]), -1, np.int32)
        global_to_local[context_rows] = np.arange(len(context_rows), dtype=np.int32)
        event_pool = np.flatnonzero(eligible_event & (events["timeframe"] == timeframe))
        event_local = global_to_local[events["context_row"][event_pool]]
        if np.any(event_local < 0):
            raise RuntimeError("event/context timeframe alignment failed")

        positions = global_to_embedding[context_rows]
        if embeddings and np.any(positions < 0):
            raise RuntimeError("timeframe rows are absent from one or more embeddings")
        timeframe_embeddings = {name: value[positions] for name, value in embeddings.items()}
        outer_matrices = {"causal_xgb": []}
        outer_matrices.update({f"{name}:fusion_xgb": [] for name in embeddings})
        inner_splits_by_outer, inner_matrices_by_outer = [], []
        for fold_index, (train, test) in enumerate(splits, start=1):
            outer_matrices["causal_xgb"].append(causal)
            for name, embedding in timeframe_embeddings.items():
                outer_matrices[f"{name}:fusion_xgb"].append(context_matrix_for_fold(
                    causal, embedding, train, test, max_components=args.max_components,
                    seed=args.seed + fold_index,
                ))
            if args.threshold_mode == "nested_isotonic":
                try:
                    inner_splits, inner_contract = nested_context_splits(
                        sample["decision_time_ns"][context_rows],
                        sample["label_end_time_ns"][context_rows],
                        sample["ticker"][context_rows], train, folds=args.inner_folds,
                        embargo_ns=args.context_bars * minutes * 60 * 1_000_000_000,
                    )
                    inner_fold_contracts[f"{timeframe}:outer_{fold_index}"] = inner_contract
                except ValueError as error:
                    inner_splits = []
                    inner_fold_contracts[f"{timeframe}:outer_{fold_index}"] = {
                        "status": "unavailable", "reason": str(error),
                    }
                inner_splits_by_outer.append(inner_splits)
                arm_inner = {"causal_xgb": [causal for _ in inner_splits]}
                for name, embedding in timeframe_embeddings.items():
                    arm_inner[f"{name}:fusion_xgb"] = [
                        context_matrix_for_fold(
                            causal, embedding, inner_train, inner_test,
                            max_components=args.max_components,
                            seed=args.seed + 1000 * fold_index + inner_index,
                        )
                        for inner_index, (inner_train, inner_test) in enumerate(
                            inner_splits, start=1,
                        )
                    ]
                inner_matrices_by_outer.append(arm_inner)
            else:
                inner_splits_by_outer.append([])
                inner_matrices_by_outer.append({})

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
                        raw_score = score = np.full(len(test), np.nan, np.float32)
                        selected, executed = raw_selected, raw_executed
                        decision_threshold = np.nan
                    else:
                        matrix = np.column_stack((
                            outer_matrices[arm][fold_index - 1][policy_context_local], policy_features,
                        )).astype(np.float32)
                        raw_score = fit_predict_fold(
                            matrix, y, groups, train, test, kind="reg", head="xgb",
                            control="real",
                            seed=stable_policy_seed(args.seed, policy, fold_index),
                        )
                        score = raw_score
                        decision_threshold = float(args.threshold)
                        if args.threshold_mode == "nested_isotonic":
                            try:
                                inner_rows, inner_raw, inner_fold_ids, inner_records = (
                                    nested_oof_predictions(
                                        inner_matrices_by_outer[fold_index - 1][arm],
                                        inner_splits_by_outer[fold_index - 1],
                                        policy_context_local, policy_features, y, groups,
                                        head="xgb",
                                        seed=stable_policy_seed(
                                            args.seed, policy + ":nested", fold_index,
                                        ),
                                        min_train=args.min_calibration_fit,
                                        min_test=args.min_calibration_rows,
                                    )
                                )
                                calibrator = fit_isotonic_expected_value(
                                    inner_raw, y[inner_rows],
                                )
                                inner_calibrated = apply_isotonic_expected_value(
                                    inner_raw, calibrator,
                                )
                                calibrated = choose_stable_calibrated_threshold(
                                    events, policy_event_rows, inner_rows, inner_calibrated,
                                    inner_fold_ids, quantiles=args.threshold_quantiles,
                                    min_executed=args.min_calibration_executed,
                                    min_coverage=args.min_calibration_coverage,
                                    floor_threshold=args.threshold, lcb_z=args.calibration_lcb_z,
                                )
                                score = apply_isotonic_expected_value(raw_score, calibrator)
                                decision_threshold = calibrated["threshold"]
                                calibration_record.update({
                                    "decision_threshold": decision_threshold,
                                    "score_scale": "calibrated_expected_net_r",
                                    "calibration_fit_rows": int(len(inner_rows)),
                                    "calibration_rows": int(len(inner_rows)),
                                    "calibration_executed": calibrated["executed"],
                                    "calibration_r_per_candidate": calibrated[
                                        "mean_r_per_candidate"
                                    ],
                                    "calibration_lcb_r_per_candidate": calibrated[
                                        "lcb_r_per_candidate"
                                    ],
                                    "calibration_no_trade": calibrated["no_trade"],
                                    "calibration_inner_folds": inner_records,
                                    "calibrator": {
                                        "method": calibrator["method"],
                                        "x": calibrator["x"].tolist(),
                                        "y": calibrator["y"].tolist(),
                                    },
                                })
                            except ValueError as error:
                                # Calibration failure cannot authorize trades. Fixed-threshold
                                # behavior remains available as a separately declared benchmark.
                                decision_threshold = float(np.nextafter(float(raw_score.max()), np.inf))
                                calibration_record.update({
                                    "threshold_mode": "nested_no_trade_fallback",
                                    "score_scale": "raw_uncalibrated",
                                    "decision_threshold": decision_threshold,
                                    "calibration_no_trade": True,
                                    "fallback_reason": str(error),
                                })
                        elif args.threshold_mode == "inner_calibrated":
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
                    prediction_parts["raw_score"].append(np.asarray(raw_score, np.float32))
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
            "policies": policy_names,
            "folds": args.folds, "context_bars": args.context_bars,
            "outer_evaluation": (
                {
                    "mode": "fixed_interval",
                    "start_ns": args.outer_eval_start,
                    "end_ns": args.outer_eval_end,
                }
                if args.outer_eval_start is not None
                else {"mode": "shared_calendar_support"}
            ),
            "threshold": args.threshold,
            "threshold_contract": (
                "nested OOF isotonic expected-net-R; trade only above a positive-LCB cutoff"
                if args.threshold_mode == "nested_isotonic"
                else (
                    "predicted net R > fixed floor; inner calibration may only raise the threshold"
                    if args.threshold_mode == "inner_calibrated"
                    else "fixed predicted net R > threshold"
                )
            ),
            "threshold_mode": args.threshold_mode,
            "inner_folds": args.inner_folds,
            "calibration_fraction": args.calibration_fraction,
            "min_calibration_fit": args.min_calibration_fit,
            "min_calibration_rows": args.min_calibration_rows,
            "min_calibration_executed": args.min_calibration_executed,
            "min_calibration_coverage": args.min_calibration_coverage,
            "calibration_lcb_z": args.calibration_lcb_z,
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
        "fold_contracts": fold_contracts, "inner_fold_contracts": inner_fold_contracts,
        "skipped": skipped,
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
    parser.add_argument(
        "--policies",
        default=(
            "atr_zigzag_v2__structural_stop__360m__3R,"
            "fractal_k2__structural_stop__360m__3R,"
            "fractal_k2__atr_stop__360m__3R,"
            "supertrend_flip__atr_stop__360m__3R"
        ),
        help="explicit comma-separated policy keys; defaults to two primaries and two controls",
    )
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument(
        "--threshold-mode", choices=("fixed", "inner_calibrated", "nested_isotonic"),
        default="nested_isotonic",
    )
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--calibration-fraction", type=float, default=0.2)
    parser.add_argument("--min-calibration-fit", type=int, default=100)
    parser.add_argument("--min-calibration-rows", type=int, default=50)
    parser.add_argument("--min-calibration-executed", type=int, default=20)
    parser.add_argument("--min-calibration-coverage", type=float, default=0.02)
    parser.add_argument("--calibration-lcb-z", type=float, default=1.0)
    parser.add_argument("--threshold-quantiles", default="0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--outer-eval-start",
        help="optional inclusive UTC date/time for fixed outer evaluation (for example 2024-07-01)",
    )
    parser.add_argument(
        "--outer-eval-end",
        help="optional exclusive UTC date/time for fixed outer evaluation (for example 2025-07-01)",
    )
    parser.add_argument("--context-bars", type=int, default=256)
    parser.add_argument("--max-components", type=int, default=128)
    parser.add_argument("--min-train", type=int, default=100)
    parser.add_argument("--min-test", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260716)
    args = parser.parse_args()
    args.threshold_quantiles = _parse_csv_numbers(args.threshold_quantiles, float)
    args.policies = _parse_csv_strings(args.policies)
    if (args.outer_eval_start is None) != (args.outer_eval_end is None):
        parser.error("--outer-eval-start and --outer-eval-end must be provided together")
    if args.outer_eval_start is not None:
        try:
            args.outer_eval_start = int(np.datetime64(args.outer_eval_start, "ns").astype(np.int64))
            args.outer_eval_end = int(np.datetime64(args.outer_eval_end, "ns").astype(np.int64))
        except (TypeError, ValueError) as error:
            parser.error(f"invalid outer evaluation interval: {error}")
        if args.outer_eval_end <= args.outer_eval_start:
            parser.error("outer evaluation end must be after start")
    run(args)


if __name__ == "__main__":
    main()
