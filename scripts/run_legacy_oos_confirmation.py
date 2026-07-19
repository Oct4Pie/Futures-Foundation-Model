#!/usr/bin/env python3
"""One-shot, full-breadth legacy OOS confirmation for the frozen pullback finalists.

This runner is intentionally separate from the development benchmark.  It reconstructs the
predeclared pullback event pool on an OOS-only interval, fits every selector and calibration rule
using development artifacts only, freezes those objects, and applies them once to OOS.  No OOS
label is passed to a scaler, PCA, estimator, calibrator, or threshold selector.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from futures_foundation.finetune.calibration import (
    apply_isotonic_expected_value,
    fit_isotonic_expected_value,
)
from futures_foundation.finetune.downstream_probe import causal_feature_matrix
from futures_foundation.execution_economics import load_execution_economics
from futures_foundation.finetune.downstream_sample import load_balanced_sample, load_row_selection
from futures_foundation.finetune.downstream_trading import (
    build_policy_events,
    load_policy_events,
)
from futures_foundation.finetune.event_contexts import (
    EventContextConfig,
    materialize_context_stream,
    save_context_shard,
)
from futures_foundation.finetune.native_contracts import verify_admission_report
from futures_foundation.finetune.path_labels import PathLabelConfig
from futures_foundation.finetune.pretext._torch.common import embed_windows
from futures_foundation.finetune.ssl_data import TFS_ALL, TICKERS_9, load_ohlcv
from scripts.benchmark_downstream_embedding import reduce_embedding_fold
from scripts.benchmark_downstream_trading import (
    apply_concurrency,
    barrier_outcome_classes,
    choose_stable_calibrated_threshold,
    expected_net_r_from_barrier,
    policy_feature_matrix,
    stable_policy_seed,
    trade_metrics,
)


POLICY = "pullback_continuation__structural_stop__360m__3R"
OOS_START = "2025-07-01"
DEFAULT_OOS_END = "2026-04-14"
CONTEXT_BARS = 256


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


def _load_embedding(path: str | Path, selection_manifest: dict) -> tuple[np.ndarray, dict]:
    path = Path(path).resolve()
    manifest = json.loads(Path(str(path) + ".manifest.json").read_text())
    if (
        manifest.get("oos_read") is not False
        or manifest.get("artifact", {}).get("sha256") != _sha256(path)
        or manifest.get("row_selection", {}).get("sha256")
        != selection_manifest["artifact"]["sha256"]
    ):
        raise ValueError(f"development embedding guard failed: {path}")
    with np.load(path, allow_pickle=False) as saved:
        embedding = np.asarray(saved["embedding"], np.float32)
        row_index = np.asarray(saved["row_index"], np.int32)
    expected = np.asarray(selection_manifest["metadata"]["rows"], np.int64)
    if embedding.ndim != 2 or len(embedding) != int(expected) or len(row_index) != len(embedding):
        raise ValueError(f"development embedding shape mismatch: {path}")
    if not np.isfinite(embedding).all() or len(np.unique(row_index)) != len(row_index):
        raise ValueError(f"invalid development embedding: {path}")
    return embedding, manifest


def _event_config(path: str | Path, *, start: str, end: str) -> tuple[EventContextConfig, dict]:
    collection_path = Path(path).resolve()
    collection = json.loads(collection_path.read_text())
    if collection.get("status") != "complete" or collection.get("oos_read") is not False:
        raise ValueError("development event collection guard failed")
    raw = dict(collection["config"])
    raw["eval_start"], raw["eval_end"] = start, end
    raw["path"] = PathLabelConfig(**raw["path"])
    config = EventContextConfig(**raw)
    config.validate()
    return config, collection


def _build_oos_artifacts(args, config: EventContextConfig, output_dir: Path):
    """Build OOS candidates and raw contexts without exposing outcomes to any fitted object."""
    tag_index = None
    sample_chunks: dict[str, list[np.ndarray]] = {
        key: [] for key in (
            "ticker", "timeframe", "decision_time_ns", "features", "feature_names",
            "tag_names", "shard_row", "stream_id",
        )
    }
    contexts, context_times = [], []
    source_shards = {}
    coverage = {}
    load_start = (
        pd.Timestamp(args.oos_start, tz="UTC") - pd.Timedelta(days=args.warmup_days)
    ).isoformat()
    eval_end = pd.Timestamp(args.oos_end, tz="UTC")
    economics = load_execution_economics(
        args.execution_costs,
        evaluation_start=pd.Timestamp(args.oos_start, tz="UTC").isoformat(),
        evaluation_end=eval_end.isoformat(),
        required_roots=TICKERS_9,
    )

    for ticker in TICKERS_9:
        for timeframe in TFS_ALL:
            stream = load_ohlcv(
                args.source_dir, (ticker,), (timeframe,), verbose=False,
                start=load_start, end=args.oos_end,
            )[0]
            ts = pd.DatetimeIndex(pd.to_datetime(stream["ts"], utc=True))
            if stream.get("contract_id") is None:
                raise ValueError(f"{ticker}@{timeframe} has no contract identity")
            last = ts[-1]
            # The source must include the complete final UTC date requested.  Coarser bars end
            # before 23:59 by construction, so require only that they reach that date.
            required_date = eval_end - pd.Timedelta(days=1)
            if last.normalize() < required_date.normalize():
                raise ValueError(
                    f"{ticker}@{timeframe} ends {last.isoformat()}, before {required_date.date()}"
                )
            frame = pd.DataFrame({
                "datetime": ts,
                "open": stream["ohlcv"][:, 0], "high": stream["ohlcv"][:, 1],
                "low": stream["ohlcv"][:, 2], "close": stream["ohlcv"][:, 3],
                "volume": stream["ohlcv"][:, 4],
                "contract_id": np.asarray(stream["contract_id"]).astype(str),
                "source_row_idx": np.arange(len(ts), dtype=np.int64),
            })
            arrays, metadata = materialize_context_stream(
                frame, ticker=ticker, timeframe=timeframe, config=config,
                execution_economics=economics,
            )
            metadata["split"]["oos_read"] = True
            metadata["split"]["role"] = "legacy_confirmation_only"
            shard_path = output_dir / "event_shards" / f"{ticker}_{timeframe}.npz"
            manifest = save_context_shard(
                shard_path, arrays, metadata,
                source={
                    "corpus": str(Path(args.source_dir).resolve()),
                    "corpus_manifest_sha256": _sha256(Path(args.source_dir) / "MANIFEST.json"),
                    "loaded_start": load_start, "loaded_end": args.oos_end,
                },
            )
            names = [str(value) for value in arrays["tag_names"]]
            current_tag = names.index("pullback_continuation")
            if tag_index is None:
                tag_index = current_tag
            elif current_tag != tag_index:
                raise ValueError("pullback tag index differs across streams")
            selected = np.flatnonzero(np.asarray(arrays["tags"])[:, current_tag])
            source_shards[f"{ticker}@{timeframe}"] = {
                "path": str(shard_path.resolve()), "sha256": manifest["artifact"]["sha256"],
            }
            coverage[f"{ticker}@{timeframe}"] = {
                "loaded_first": ts[0].isoformat(), "loaded_last": last.isoformat(),
                "dense_rows": int(metadata["rows"]), "pullback_candidates": int(len(selected)),
            }
            if not len(selected):
                print(
                    f"[oos] {ticker}@{timeframe}: candidates=0 last={last}", flush=True,
                )
                continue
            starts = np.asarray(arrays["context_start_source_idx"])[selected].astype(np.int64)
            decisions = np.asarray(arrays["decision_source_idx"])[selected].astype(np.int64)
            if np.any(decisions - starts + 1 != CONTEXT_BARS):
                raise ValueError(f"{ticker}@{timeframe} context length mismatch")
            gather = starts[:, None] + np.arange(CONTEXT_BARS, dtype=np.int64)[None, :]
            value = np.asarray(stream["ohlcv"])[gather].astype(np.float32)
            time_value = ts.asi8[gather]
            contract = np.asarray(stream["contract_id"])[gather].astype(str)
            if np.any(contract != contract[:, :1]):
                raise ValueError(f"{ticker}@{timeframe} gathered context crosses a roll")
            if not np.array_equal(time_value[:, -1], arrays["decision_time_ns"][selected]):
                raise ValueError(f"{ticker}@{timeframe} gathered context identity mismatch")

            sample_chunks["ticker"].append(np.asarray(arrays["ticker"])[selected])
            sample_chunks["timeframe"].append(np.asarray(arrays["timeframe"])[selected])
            sample_chunks["decision_time_ns"].append(
                np.asarray(arrays["decision_time_ns"])[selected].astype(np.int64)
            )
            sample_chunks["features"].append(np.asarray(arrays["features"])[selected])
            sample_chunks["feature_names"].append(np.asarray(arrays["feature_names"]))
            sample_chunks["tag_names"].append(np.asarray(arrays["tag_names"]))
            sample_chunks["shard_row"].append(selected.astype(np.int64))
            sample_chunks["stream_id"].append(
                np.full(len(selected), f"{ticker}@{timeframe}")
            )
            contexts.append(value)
            context_times.append(time_value.astype(np.int64))
            print(
                f"[oos] {ticker}@{timeframe}: candidates={len(selected):,} last={last}",
                flush=True,
            )

    # Static schema arrays must be identical, not concatenated.
    feature_names = sample_chunks.pop("feature_names")
    tag_names = sample_chunks.pop("tag_names")
    if any(not np.array_equal(feature_names[0], value) for value in feature_names[1:]):
        raise ValueError("feature schema differs across OOS streams")
    if any(not np.array_equal(tag_names[0], value) for value in tag_names[1:]):
        raise ValueError("tag schema differs across OOS streams")
    sample = {key: np.concatenate(value) for key, value in sample_chunks.items()}
    sample["feature_names"], sample["tag_names"] = feature_names[0], tag_names[0]
    context = np.concatenate(contexts)
    context_time_ns = np.concatenate(context_times)
    if len(sample["stream_id"]) != len(context):
        raise RuntimeError("OOS sample/context alignment failed")

    events, event_metadata = build_policy_events(
        sample, np.arange(len(context), dtype=np.int64), source_shards, economics,
        slippage_ticks=0.0,
    )
    event_metadata["oos_read"] = True
    event_metadata["split"] = {
        "role": "legacy_confirmation_only", "start": args.oos_start, "end": args.oos_end,
    }
    sample_path = output_dir / "oos_pullback_sample.npz"
    context_path = output_dir / "oos_pullback_contexts.npz"
    event_path = output_dir / "oos_policy_events_fees_only.npz"
    _atomic_npz(sample_path, **sample)
    _atomic_npz(
        context_path, context=context, context_time_ns=context_time_ns,
        row_index=np.arange(len(context), dtype=np.int32),
    )
    _atomic_npz(event_path, **events)
    event_metadata["artifact"] = {
        "path": str(event_path.resolve()), "sha256": _sha256(event_path),
    }
    _atomic_json(Path(str(event_path) + ".manifest.json"), event_metadata)
    manifest = {
        "schema_version": "ffm_legacy_oos_candidates_v1", "status": "complete",
        "oos_read": True, "created_utc": datetime.now(timezone.utc).isoformat(),
        "split": {"start": args.oos_start, "end": args.oos_end},
        "config": asdict(config), "coverage": coverage,
        "sample": {"path": str(sample_path.resolve()), "sha256": _sha256(sample_path)},
        "contexts": {
            "path": str(context_path.resolve()), "sha256": _sha256(context_path),
            "shape": list(context.shape),
        },
        "events": event_metadata["artifact"],
        "source_manifest_sha256": _sha256(Path(args.source_dir) / "MANIFEST.json"),
    }
    _atomic_json(output_dir / "oos_candidates_manifest.json", manifest)
    return sample, context, context_time_ns, events, event_metadata, manifest


def _extract_oos_embeddings(args, context: np.ndarray, output_dir: Path) -> dict[str, np.ndarray]:
    specifications = {
        "mantis_v1:vanilla": ("paris-noah/Mantis-8M", "v1", None),
        "mantis_v1:stage2": ("paris-noah/Mantis-8M", "v1", args.v1_stage2_checkpoint),
        "mantis_v2:vanilla": ("paris-noah/MantisV2", "v2", None),
        "mantis_v2:stage2": ("paris-noah/MantisV2", "v2", args.v2_stage2_checkpoint),
    }
    output = {}
    for name, (model_id, version, checkpoint) in specifications.items():
        checkpoint_path = None if checkpoint is None else Path(checkpoint).resolve()
        if checkpoint_path is not None and not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        value = embed_windows(
            context.transpose(0, 2, 1), ckpt=checkpoint_path,
            model_id=model_id, model_version=version, device=args.device,
            batch=args.mantis_batch, preprocessing="per_window_per_channel_zscore_v1",
        )
        if value.ndim != 2 or len(value) != len(context) or not np.isfinite(value).all():
            raise ValueError(f"invalid OOS embedding for {name}")
        path = output_dir / "embeddings" / f"{name.replace(':', '_')}.npz"
        _atomic_npz(path, embedding=np.asarray(value, np.float32))
        metadata = {
            "schema_version": "ffm_legacy_oos_embedding_v1", "status": "complete",
            "oos_read": True, "arm": name, "shape": list(value.shape),
            "checkpoint": str(checkpoint_path) if checkpoint_path else None,
            "checkpoint_sha256": _sha256(checkpoint_path) if checkpoint_path else None,
            "preprocessing": "per_window_per_channel_zscore_v1",
            "artifact": {"path": str(path.resolve()), "sha256": _sha256(path)},
        }
        _atomic_json(Path(str(path) + ".manifest.json"), metadata)
        output[name] = np.asarray(value, np.float32)
        print(f"[embed] {name}: {value.shape}", flush=True)
    return output


def _xgb_regression(train_x, train_y, test_x, *, seed):
    import xgboost as xgb
    model = xgb.XGBRegressor(
        n_estimators=120, max_depth=3, learning_rate=0.04, subsample=0.8,
        colsample_bytree=0.8, reg_lambda=10.0, min_child_weight=20.0,
        objective="reg:squarederror", tree_method="hist", random_state=int(seed),
        n_jobs=1, verbosity=0,
    )
    model.fit(np.asarray(train_x, np.float32), np.asarray(train_y, np.float32))
    return np.asarray(model.predict(np.asarray(test_x, np.float32)), np.float32)


def _barrier_prediction(train_x, train_events, train_rows, test_x, test_events, test_rows, *, seed):
    import xgboost as xgb
    classes = barrier_outcome_classes(train_events["barrier_state"][train_rows])
    if set(np.unique(classes).tolist()) != {0, 1, 2}:
        raise ValueError("development barrier labels lack an outcome class")
    common = dict(
        n_estimators=120, max_depth=3, learning_rate=0.04, subsample=0.8,
        colsample_bytree=0.8, reg_lambda=10.0, min_child_weight=20.0,
        tree_method="hist", random_state=int(seed), n_jobs=1, verbosity=0,
    )
    classifier = xgb.XGBClassifier(
        objective="multi:softprob", num_class=3, eval_metric="mlogloss", **common,
    )
    classifier.fit(train_x, classes)
    probabilities = np.asarray(classifier.predict_proba(test_x), np.float64)
    neither = np.flatnonzero(classes == 2)
    if len(neither) < 20:
        raise ValueError("too few development neither outcomes")
    terminal = xgb.XGBRegressor(objective="reg:squarederror", **common)
    terminal.fit(train_x[neither], train_events["gross_r"][train_rows][neither])
    terminal_r = np.clip(np.asarray(terminal.predict(test_x), np.float64), -1.0, 3.0)
    return expected_net_r_from_barrier(
        probabilities, terminal_r, test_events["total_cost_r"][test_rows], target_r=3.0,
    )


def _calibration_from_development(source_dir: Path, arm_name: str, dev_events: dict, timeframe: str):
    result_path, expected_arm = {
        "causal_barrier": (
            source_dir / "mantis_barrier_decomposed_pullback", "causal_xgb",
        ),
        "mantis_v1:vanilla": (
            source_dir / "mantis_v1_v2_vanilla_direct", "mantis_v1:vanilla:fusion_xgb",
        ),
        "mantis_v2:vanilla": (
            source_dir / "mantis_v1_v2_vanilla_direct", "mantis_v2:vanilla:fusion_xgb",
        ),
        "mantis_v1:stage2": (
            source_dir / "mantis_v1_stage2_direct", "mantis_v1:stage2:fusion_xgb",
        ),
        "mantis_v2:stage2": (
            source_dir / "mantis_v1_v2_vanilla_direct", "mantis_v2:stage2:fusion_xgb",
        ),
    }[arm_name]
    report = json.loads((result_path / "trading_results.json").read_text())
    if report.get("oos_read") is not False:
        raise ValueError("calibration source is not development-only")
    prediction_path = result_path / "trading_predictions.npz"
    if report["predictions"]["sha256"] != _sha256(prediction_path):
        raise ValueError("calibration prediction hash mismatch")
    with np.load(prediction_path, allow_pickle=False) as saved:
        prediction = {key: saved[key] for key in saved.files}
    arms = [str(value) for value in prediction["arm_names"]]
    policies = [str(value) for value in prediction["policy_names"]]
    arm_index, policy_index = arms.index(expected_arm), policies.index(POLICY)
    rows = np.flatnonzero(
        (prediction["arm_index"] == arm_index)
        & (prediction["policy_index"] == policy_index)
    )
    event_rows = np.asarray(prediction["event_row"])[rows].astype(np.int64)
    rows = rows[np.asarray(dev_events["timeframe"])[event_rows] == timeframe]
    event_rows = np.asarray(prediction["event_row"])[rows].astype(np.int64)
    raw = np.asarray(prediction["raw_score"])[rows].astype(np.float64)
    folds = np.asarray(prediction["fold"])[rows].astype(np.int64)
    if not len(rows) or not np.isfinite(raw).all():
        raise ValueError(f"missing finite development OOF predictions for {arm_name}/{timeframe}")
    calibration = fit_isotonic_expected_value(raw, dev_events["realized_r"][event_rows])
    calibrated = apply_isotonic_expected_value(raw, calibration)
    threshold = choose_stable_calibrated_threshold(
        dev_events, event_rows, np.arange(len(event_rows)), calibrated, folds,
        quantiles=(0.5, 0.6, 0.7, 0.8, 0.9), min_executed=20,
        min_coverage=0.02, floor_threshold=0.0, lcb_z=1.0,
    )
    return calibration, threshold, {
        "source": str(result_path.resolve()), "source_sha256": _sha256(result_path / "trading_results.json"),
        "oof_rows": int(len(rows)), "threshold": threshold,
    }


def _paired_block_interval(delta: np.ndarray, time_ns: np.ndarray, *, repetitions: int, seed: int):
    delta = np.asarray(delta, np.float64)
    time_ns = np.asarray(time_ns, np.int64)
    if len(delta) != len(time_ns) or not len(delta):
        raise ValueError("paired interval inputs are empty or misaligned")
    day = time_ns // (24 * 60 * 60 * 1_000_000_000)
    blocks = day // 7
    unique = np.unique(blocks)
    block_values = [np.flatnonzero(blocks == value) for value in unique]
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(repetitions), np.float64)
    for i in range(int(repetitions)):
        chosen = rng.integers(0, len(block_values), size=len(block_values))
        values = np.concatenate([delta[block_values[index]] for index in chosen])
        draws[i] = values.mean()
    return {
        "delta_r_per_candidate": float(delta.mean()),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "bootstrap_positive_probability": float(np.mean(draws > 0)),
        "calendar_blocks": int(len(unique)), "repetitions": int(repetitions),
    }


def _run_confirmation(args, oos_sample, oos_events, oos_embeddings, output_dir: Path):
    dev_sample, dev_sample_manifest = load_balanced_sample(args.dev_sample)
    dev_selection, dev_selection_manifest = load_row_selection(
        args.dev_row_selection, sample_manifest=dev_sample_manifest,
    )
    dev_selected = np.asarray(dev_selection["row_index"], np.int32)
    dev_events, dev_event_manifest = load_policy_events(args.dev_policy_events)
    if dev_event_manifest["sample_sha256"] != dev_sample_manifest["artifact"]["sha256"]:
        raise ValueError("development policy/sample mismatch")
    oos_start_ns = int(pd.Timestamp(args.oos_start, tz="UTC").value)
    oos_end_ns = int(pd.Timestamp(args.oos_end, tz="UTC").value)
    if (
        np.any(np.asarray(dev_sample["decision_time_ns"], np.int64) >= oos_start_ns)
        or np.any(np.asarray(dev_events["signal_time_ns"], np.int64) >= oos_start_ns)
        or np.any(np.asarray(dev_events["exit_time_ns"], np.int64) >= oos_start_ns)
    ):
        raise ValueError("development fitting artifact touches the OOS interval")
    if (
        np.any(np.asarray(oos_sample["decision_time_ns"], np.int64) < oos_start_ns)
        or np.any(np.asarray(oos_sample["decision_time_ns"], np.int64) >= oos_end_ns)
        or np.any(np.asarray(oos_events["signal_time_ns"], np.int64) < oos_start_ns)
        or np.any(np.asarray(oos_events["exit_time_ns"], np.int64) >= oos_end_ns)
    ):
        raise ValueError("OOS scoring artifact escapes its frozen interval")
    dev_embedding_paths = {
        "mantis_v1:vanilla": args.dev_v1_vanilla,
        "mantis_v1:stage2": args.dev_v1_stage2,
        "mantis_v2:vanilla": args.dev_v2_vanilla,
        "mantis_v2:stage2": args.dev_v2_stage2,
    }
    dev_embeddings = {
        name: _load_embedding(path, dev_selection_manifest)[0]
        for name, path in dev_embedding_paths.items()
    }
    global_to_dev_embedding = np.full(len(dev_sample["stream_id"]), -1, np.int32)
    global_to_dev_embedding[dev_selected] = np.arange(len(dev_selected), dtype=np.int32)

    dev_policy_rows = np.flatnonzero(dev_events["policy_key"] == POLICY)
    oos_policy_rows = np.flatnonzero(oos_events["policy_key"] == POLICY)
    if not len(dev_policy_rows) or not len(oos_policy_rows):
        raise ValueError("pullback policy is absent from development or OOS events")
    arms = ["raw_all", "causal_barrier", *dev_embeddings]
    utilities = {arm: np.zeros(len(oos_policy_rows), np.float64) for arm in arms}
    executed_flags = {arm: np.zeros(len(oos_policy_rows), bool) for arm in arms}
    calibration_records = {}

    for timeframe in TFS_ALL:
        dev_context_rows = dev_selected[dev_sample["timeframe"][dev_selected] == timeframe]
        oos_context_rows = np.flatnonzero(oos_sample["timeframe"] == timeframe)
        dev_event_positions = np.flatnonzero(
            dev_events["timeframe"][dev_policy_rows] == timeframe
        )
        oos_event_positions = np.flatnonzero(
            oos_events["timeframe"][oos_policy_rows] == timeframe
        )
        if not len(dev_event_positions) or not len(oos_event_positions):
            raise ValueError(f"{timeframe} lacks development or OOS policy candidates")
        dev_rows = dev_policy_rows[dev_event_positions]
        oos_rows = oos_policy_rows[oos_event_positions]

        dev_causal, dev_causal_names = causal_feature_matrix(dev_sample, dev_context_rows)
        oos_causal, oos_causal_names = causal_feature_matrix(oos_sample, oos_context_rows)
        if dev_causal_names != oos_causal_names:
            raise ValueError(f"{timeframe} causal feature schema mismatch")
        dev_global_to_local = np.full(len(dev_sample["stream_id"]), -1, np.int32)
        dev_global_to_local[dev_context_rows] = np.arange(len(dev_context_rows), dtype=np.int32)
        oos_global_to_local = np.full(len(oos_sample["stream_id"]), -1, np.int32)
        oos_global_to_local[oos_context_rows] = np.arange(len(oos_context_rows), dtype=np.int32)
        dev_event_context = dev_global_to_local[dev_events["context_row"][dev_rows]]
        oos_event_context = oos_global_to_local[oos_events["context_row"][oos_rows]]
        if np.any(dev_event_context < 0) or np.any(oos_event_context < 0):
            raise RuntimeError(f"{timeframe} event/context alignment failed")
        dev_policy_features, names = policy_feature_matrix(dev_events, dev_rows)
        oos_policy_features, oos_names = policy_feature_matrix(oos_events, oos_rows)
        if names != oos_names:
            raise ValueError("policy feature schema mismatch")

        # Raw selection has no fitted object.
        raw_selected = np.ones(len(oos_rows), bool)
        raw_executed = apply_concurrency(oos_events, oos_rows, raw_selected)
        executed_flags["raw_all"][oos_event_positions] = raw_executed
        utilities["raw_all"][oos_event_positions[raw_executed]] = oos_events["realized_r"][
            oos_rows[raw_executed]
        ]

        # Causal barrier decomposition, trained only on development outcomes.
        train_matrix = np.column_stack((
            dev_causal[dev_event_context], dev_policy_features,
        )).astype(np.float32)
        test_matrix = np.column_stack((
            oos_causal[oos_event_context], oos_policy_features,
        )).astype(np.float32)
        raw_score = _barrier_prediction(
            train_matrix, dev_events, dev_rows, test_matrix, oos_events, oos_rows,
            seed=stable_policy_seed(args.seed, POLICY, int(timeframe[:-3])),
        )
        calibration, threshold, record = _calibration_from_development(
            Path(args.dev_results_root), "causal_barrier", dev_events, timeframe,
        )
        score = apply_isotonic_expected_value(raw_score, calibration)
        selected = score > float(threshold["threshold"])
        executed = apply_concurrency(oos_events, oos_rows, selected)
        executed_flags["causal_barrier"][oos_event_positions] = executed
        utilities["causal_barrier"][oos_event_positions[executed]] = oos_events["realized_r"][
            oos_rows[executed]
        ]
        calibration_records[f"{timeframe}:causal_barrier"] = record

        for arm, dev_embedding in dev_embeddings.items():
            dev_positions = global_to_dev_embedding[dev_context_rows]
            if np.any(dev_positions < 0):
                raise ValueError(f"development embedding is missing {timeframe} rows")
            train_embedding = dev_embedding[dev_positions]
            test_embedding = oos_embeddings[arm][oos_context_rows]
            joined = np.concatenate((train_embedding, test_embedding))
            train_index = np.arange(len(train_embedding), dtype=np.int64)
            test_index = np.arange(len(train_embedding), len(joined), dtype=np.int64)
            reduced, reduction = reduce_embedding_fold(
                joined, train_index, test_index, max_components=128,
                seed=args.seed + int(timeframe[:-3]),
            )
            train_context = np.column_stack((reduced[train_index], dev_causal)).astype(np.float32)
            test_context = np.column_stack((reduced[test_index], oos_causal)).astype(np.float32)
            train_matrix = np.column_stack((
                train_context[dev_event_context], dev_policy_features,
            )).astype(np.float32)
            test_matrix = np.column_stack((
                test_context[oos_event_context], oos_policy_features,
            )).astype(np.float32)
            raw_score = _xgb_regression(
                train_matrix, dev_events["realized_r"][dev_rows], test_matrix,
                seed=stable_policy_seed(args.seed, POLICY + arm, int(timeframe[:-3])),
            )
            calibration, threshold, record = _calibration_from_development(
                Path(args.dev_results_root), arm, dev_events, timeframe,
            )
            score = apply_isotonic_expected_value(raw_score, calibration)
            selected = score > float(threshold["threshold"])
            executed = apply_concurrency(oos_events, oos_rows, selected)
            executed_flags[arm][oos_event_positions] = executed
            utilities[arm][oos_event_positions[executed]] = oos_events["realized_r"][
                oos_rows[executed]
            ]
            calibration_records[f"{timeframe}:{arm}"] = record | {"reduction": reduction}
        print(f"[confirm] {timeframe}: candidates={len(oos_rows):,}", flush=True)

    summaries, one_tick = [], []
    candidate_events = oos_policy_rows
    for arm in arms:
        executed = executed_flags[arm]
        rows = candidate_events[executed]
        metrics = trade_metrics(
            oos_events["realized_r"][rows], oos_events["reached"][rows],
            oos_events["signal_time_ns"][rows],
        )
        metrics.update({
            "arm": arm, "policy": POLICY, "candidates": int(len(candidate_events)),
            "selected_and_executed": int(executed.sum()),
            "ticker_breadth": int(len(np.unique(oos_events["ticker"][rows]))) if len(rows) else 0,
            "timeframe_breadth": int(len(np.unique(oos_events["timeframe"][rows]))) if len(rows) else 0,
        })
        summaries.append(metrics)
        one_tick_r = (
            oos_events["gross_r"][rows] - oos_events["fee_r"][rows]
            - 1.0 / oos_events["risk_ticks"][rows]
        )
        stress = trade_metrics(
            one_tick_r, oos_events["reached"][rows], oos_events["signal_time_ns"][rows],
        )
        stress.update({"arm": arm, "policy": POLICY, "round_trip_slippage_ticks": 1.0})
        one_tick.append(stress)

    comparisons = []
    times = oos_events["signal_time_ns"][candidate_events]
    comparison_pairs = [
        (arm, baseline)
        for baseline in ("raw_all", "causal_barrier")
        for arm in arms if arm != baseline
    ] + [
        ("mantis_v1:stage2", "mantis_v1:vanilla"),
        ("mantis_v2:stage2", "mantis_v2:vanilla"),
    ]
    for arm, baseline in comparison_pairs:
        comparisons.append({
            "arm": arm, "baseline": baseline,
            **_paired_block_interval(
                utilities[arm] - utilities[baseline], times,
                repetitions=args.bootstrap_repetitions,
                seed=args.seed + len(comparisons),
            ),
        })

    utility_path = output_dir / "oos_frozen_execution_utilities.npz"
    _atomic_npz(
        utility_path,
        event_row=candidate_events.astype(np.int64),
        signal_time_ns=np.asarray(times, np.int64),
        arm_names=np.asarray(arms),
        utility=np.column_stack([utilities[arm] for arm in arms]).astype(np.float32),
        executed=np.column_stack([executed_flags[arm] for arm in arms]),
    )

    report = {
        "schema_version": "ffm_legacy_oos_confirmation_v1", "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(), "oos_read": True,
        "split": {
            "role": "legacy_oos_confirmation", "start": args.oos_start,
            "end_exclusive": args.oos_end,
            "training_or_calibration_on_oos": False,
            "prior_inspection_caveat": True,
        },
        "policy": POLICY,
        "admission": args.admissions,
        "execution": {
            "primary_slippage_ticks_round_trip": 0.0, "delay_bars": 0,
            "entry": "next bar open", "same_bar_ambiguity": "adverse_first",
            "concurrency": "one active trade per ticker/policy/timeframe",
            "fee_schedule_sha256": _sha256(args.execution_costs),
        },
        "development_sources": {
            "sample_sha256": dev_sample_manifest["artifact"]["sha256"],
            "row_selection_sha256": dev_selection_manifest["artifact"]["sha256"],
            "policy_events_sha256": dev_event_manifest["artifact"]["sha256"],
            "embedding_sha256": {
                name: json.loads(Path(str(path) + ".manifest.json").read_text())["artifact"]["sha256"]
                for name, path in dev_embedding_paths.items()
            },
        },
        "calibration": calibration_records,
        "summaries_zero_tick": summaries,
        "summaries_one_tick_frozen_reprice": one_tick,
        "paired_oos_comparisons": comparisons,
        "frozen_execution_utilities": {
            "path": str(utility_path.resolve()), "sha256": _sha256(utility_path),
        },
    }
    report_path = output_dir / "legacy_oos_confirmation.json"
    _atomic_json(report_path, report)
    print(json.dumps({
        "status": "complete", "oos_read": True, "report": str(report_path.resolve()),
        "summaries": summaries,
    }, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", default="data/ssl_corpus_v2_6tf")
    parser.add_argument("--development-collection", default="output/foundation_tournament/event_contexts_conditional_v2/MANIFEST.json")
    parser.add_argument("--dev-sample", default="output/foundation_tournament/conditional_event_gate_v2/candidate_sample.npz")
    parser.add_argument("--dev-row-selection", default="output/foundation_tournament/conditional_event_gate_v2/all_candidate_rows.npz")
    parser.add_argument("--dev-policy-events", default="output/foundation_tournament/conditional_event_gate_v2/policy_events_fees_only.npz")
    parser.add_argument("--dev-results-root", default="output/foundation_tournament/conditional_event_gate_v2")
    parser.add_argument("--dev-v1-vanilla", default="output/foundation_tournament/conditional_event_gate_v2/representations/embeddings/mantis_v1/vanilla.npz")
    parser.add_argument("--dev-v1-stage2", default="output/foundation_tournament/conditional_event_gate_v2/representations/embeddings/mantis_v1/stage2.npz")
    parser.add_argument("--dev-v2-vanilla", default="output/foundation_tournament/conditional_event_gate_v2/representations/embeddings/mantis_v2/vanilla.npz")
    parser.add_argument("--dev-v2-stage2", default="output/foundation_tournament/conditional_event_gate_v2/representations/embeddings/mantis_v2/stage2.npz")
    parser.add_argument("--v1-stage2-checkpoint", default="output/mantis_v1_ssl_pilot_256_v1/mantis_v1_stage2_vicreg_v1_seed17_256bar_dev.pt")
    parser.add_argument("--v2-stage2-checkpoint", default="output/mantis_v2_ssl_pilot_256_v1/mantis_v2_stage2_vicreg_v1_seed17_256bar_dev.pt")
    parser.add_argument("--v1-admission-report", required=True)
    parser.add_argument("--v2-admission-report", required=True)
    parser.add_argument("--execution-costs", default="config/execution_costs.yaml")
    parser.add_argument("--output-dir", default="output/foundation_tournament/legacy_oos_confirmation_v1")
    parser.add_argument("--oos-start", default=OOS_START)
    parser.add_argument("--oos-end", default=DEFAULT_OOS_END)
    parser.add_argument("--warmup-days", type=int, default=45)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mantis-batch", type=int, default=256)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260717)
    args = parser.parse_args()
    if args.oos_start != OOS_START:
        parser.error(f"confirmation start is frozen at {OOS_START}")
    if pd.Timestamp(args.oos_end, tz="UTC") > pd.Timestamp(DEFAULT_OOS_END, tz="UTC"):
        parser.error(f"common-coverage end cannot exceed {DEFAULT_OOS_END}")
    if args.warmup_days < 30 or args.bootstrap_repetitions < 1000:
        parser.error("warmup must be >=30 days and bootstrap repetitions >=1000")
    admissions = {
        "mantis_v1": verify_admission_report(
            args.v1_admission_report, arm_key="mantis_v1", track="B",
            route="supervised_barrier_experimental_task", require_training=False,
            required_artifacts={"stage2_checkpoint": args.v1_stage2_checkpoint},
        ),
        "mantis_v2": verify_admission_report(
            args.v2_admission_report, arm_key="mantis_v2", track="B",
            route="supervised_barrier_experimental_task", require_training=False,
            required_artifacts={"stage2_checkpoint": args.v2_stage2_checkpoint},
        ),
    }
    args.admissions = {
        key: {
            "integrity": value["integrity"],
            "registry_sha256": value["registry_sha256"],
            "dossier_sha256": value["dossier_sha256"],
        }
        for key, value in admissions.items()
    }
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config, _ = _event_config(
        args.development_collection, start=args.oos_start, end=args.oos_end,
    )
    sample, context, _, events, _, _ = _build_oos_artifacts(args, config, output_dir)
    embeddings = _extract_oos_embeddings(args, context, output_dir)
    _run_confirmation(args, sample, events, embeddings, output_dir)


if __name__ == "__main__":
    main()
