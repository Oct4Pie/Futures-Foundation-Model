#!/usr/bin/env python3
"""Extract one surviving route's native output on the frozen common-information rows.

The script infers all model, source-runtime, tokenizer, parent, and deployment paths
from verified pilot/smoke evidence.  It accepts no replacement checkpoint arguments.
Native outputs are preserved in full and exposed to the downstream ruler by an exact,
nonlearned C-order flattening.  The resulting table cannot grant route promotion,
full training, OOS access, deployment, or trading.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from futures_foundation.finetune.downstream_contexts import load_downstream_contexts
from futures_foundation.finetune.downstream_sample import (
    load_balanced_sample,
    load_row_selection,
)
from futures_foundation.finetune.native_downstream_features import (
    build_metadata,
    save_feature_table,
)
from futures_foundation.finetune.native_route_pilot import load_route_pilot_evidence
from futures_foundation.finetune.native_route_smoke import load_route_smoke_evidence
from futures_foundation.finetune.routes import (
    chronos_bolt,
    chronos_v1,
    kronos_predictor,
    moment_reconstruction,
)


CHUNK_SCHEMA = "ffm_native_downstream_feature_chunk_v1"

SUPPORTED_ROUTES = (
    chronos_bolt.ROUTE_KEY,
    chronos_v1.ROUTE_KEY,
    moment_reconstruction.ROUTE_KEY,
    kronos_predictor.ROUTE_KEY,
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: str | Path, value: object) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(target) + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)
    return target


def _tensor_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _save_chunk(
    path: str | Path,
    *,
    route_key: str,
    start: int,
    stop: int,
    total_rows: int,
    row_index: np.ndarray,
    native_output: np.ndarray,
    batch_size: int,
    seed: int,
    selection_manifest: dict[str, Any],
    context_manifest: dict[str, Any],
    pilot: dict[str, Any],
    executor_path: str | Path,
) -> dict[str, Any]:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(target) + f".{os.getpid()}.tmp.npz")
    np.savez_compressed(
        temporary,
        row_index=np.asarray(row_index, np.int64),
        native_output=np.asarray(native_output),
    )
    os.replace(temporary, target)
    document = {
        "schema_version": CHUNK_SCHEMA,
        "route_key": route_key,
        "start": int(start),
        "stop": int(stop),
        "total_rows": int(total_rows),
        "batch_size": int(batch_size),
        "seed": int(seed),
        "selection_sha256": selection_manifest["artifact"]["sha256"],
        "contexts_sha256": context_manifest["artifact"]["sha256"],
        "pilot_evidence_sha256": pilot["evidence_sha256"],
        "executor_sha256": _sha256(executor_path),
        "extractor_sha256": _sha256(Path(__file__).resolve()),
        "native_dtype": str(np.asarray(native_output).dtype),
        "native_shape": list(np.asarray(native_output).shape),
        "artifact": {
            "path": str(target),
            "sha256": _sha256(target),
            "bytes": int(target.stat().st_size),
        },
    }
    manifest_path = _atomic_json(Path(str(target) + ".manifest.json"), document)
    return {
        "status": "chunk_complete",
        "route_key": route_key,
        "start": int(start),
        "stop": int(stop),
        "artifact": document["artifact"],
        "manifest": {"path": str(manifest_path), "sha256": _sha256(manifest_path)},
    }


def _load_chunk(
    path: str | Path,
    *,
    expected_route: str,
    expected_total_rows: int,
    selection_manifest: dict[str, Any],
    context_manifest: dict[str, Any],
    pilot: dict[str, Any],
    executor_path: str | Path,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    source = Path(path).expanduser().resolve()
    manifest_path = Path(str(source) + ".manifest.json")
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        document.get("schema_version") != CHUNK_SCHEMA
        or document.get("route_key") != expected_route
        or int(document.get("total_rows", -1)) != int(expected_total_rows)
        or document.get("selection_sha256")
        != selection_manifest["artifact"]["sha256"]
        or document.get("contexts_sha256") != context_manifest["artifact"]["sha256"]
        or document.get("pilot_evidence_sha256") != pilot["evidence_sha256"]
        or document.get("executor_sha256") != _sha256(executor_path)
        or document.get("extractor_sha256") != _sha256(Path(__file__).resolve())
        or document.get("artifact", {}).get("sha256") != _sha256(source)
    ):
        raise ValueError(f"native downstream feature chunk is stale: {source}")
    with np.load(source, allow_pickle=False) as saved:
        if set(saved.files) != {"row_index", "native_output"}:
            raise ValueError(f"native downstream feature chunk closure is invalid: {source}")
        row_index = np.asarray(saved["row_index"], np.int64)
        native_output = saved["native_output"]
    start, stop = int(document["start"]), int(document["stop"])
    if (
        stop <= start
        or row_index.shape != (stop - start,)
        or native_output.shape[0] != stop - start
        or str(native_output.dtype) != document["native_dtype"]
        or list(native_output.shape) != document["native_shape"]
    ):
        raise ValueError(f"native downstream feature chunk geometry is invalid: {source}")
    return document, row_index, native_output


def _causal_parent(context: np.ndarray, *, stub_bars: int) -> np.ndarray:
    values = np.asarray(context, np.float32)
    if values.ndim != 3 or values.shape[1:] != (512, 5):
        raise ValueError("common downstream context must have shape [B,512,5]")
    if int(stub_bars) < 1:
        raise ValueError("causal parent stub_bars must be positive")
    stub = np.repeat(values[:, -1:, :], int(stub_bars), axis=1)
    return np.concatenate((values, stub), axis=1).astype(np.float32, copy=False)


def _future_timestamps(context_time_ns: np.ndarray) -> np.ndarray:
    timestamps = np.asarray(context_time_ns, np.int64)
    if timestamps.ndim != 2 or timestamps.shape[1] != 512:
        raise ValueError("common downstream timestamps must have shape [B,512]")
    delta = timestamps[:, -1] - timestamps[:, -2]
    if np.any(delta <= 0):
        raise ValueError("common downstream timestamps have invalid terminal cadence")
    return timestamps[:, -1, None] + delta[:, None] * np.arange(
        1, 17, dtype=np.int64,
    )[None, :]


def _flatten(output: np.ndarray) -> np.ndarray:
    values = np.asarray(output)
    return np.ascontiguousarray(values.reshape(len(values), -1), dtype=np.float32)


def _bolt_names() -> np.ndarray:
    return np.asarray([
        f"quantile_{channel}_h{horizon + 1:02d}_q{quantile:g}"
        for channel in chronos_bolt.CHANNELS
        for horizon in range(chronos_bolt.HORIZON_LENGTH)
        for quantile in chronos_bolt.QUANTILES
    ])


def _v1_names() -> np.ndarray:
    return np.asarray([
        f"sample_{channel}_s{sample:02d}_h{horizon + 1:02d}"
        for channel in chronos_v1.CHANNELS
        for sample in range(chronos_v1.NUM_SAMPLES)
        for horizon in range(chronos_v1.EFFECTIVE_HORIZON)
    ])


def _moment_names() -> np.ndarray:
    return np.asarray([f"embedding_{index:03d}" for index in range(512)])


def _predictor_names() -> np.ndarray:
    return np.asarray([
        f"forecast_{channel}_h{horizon + 1:02d}"
        for horizon in range(kronos_predictor.HORIZON_LENGTH)
        for channel in kronos_predictor.NATIVE_CHANNELS
    ])


def _extract_batches(
    rows: int,
    batch_size: int,
    function: Callable[[int, int], np.ndarray],
) -> np.ndarray:
    chunks = []
    for start in range(0, rows, batch_size):
        stop = min(start + batch_size, rows)
        chunk = np.asarray(function(start, stop))
        if chunk.shape[0] != stop - start:
            raise RuntimeError("native downstream extractor returned wrong batch cardinality")
        chunks.append(chunk)
        print(f"[native-features] rows={stop:,}/{rows:,}", flush=True)
    output = np.concatenate(chunks, axis=0)
    if output.shape[0] != rows:
        raise RuntimeError("native downstream extraction row count drifted")
    return output


def _finalize_v1_chunks(
    *,
    output_path: Path,
    chunk_paths: list[str],
    row_index: np.ndarray,
    sample_manifest: dict[str, Any],
    selection_manifest: dict[str, Any],
    context_manifest: dict[str, Any],
    pilot_path: Path,
    pilot: dict[str, Any],
    bundle_path: Path,
    overwrite: bool,
) -> dict[str, Any]:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"native downstream output already exists: {output_path}")
    if not chunk_paths:
        raise ValueError("Chronos V1 finalization requires at least one chunk")
    loaded = [
        _load_chunk(
            path,
            expected_route=chronos_v1.ROUTE_KEY,
            expected_total_rows=len(row_index),
            selection_manifest=selection_manifest,
            context_manifest=context_manifest,
            pilot=pilot,
            executor_path=chronos_v1.__file__,
        )
        for path in chunk_paths
    ]
    loaded.sort(key=lambda item: int(item[0]["start"]))
    seed_values = {int(item[0]["seed"]) for item in loaded}
    if len(seed_values) != 1:
        raise ValueError("Chronos V1 feature chunks use different extraction seeds")
    extraction_seed = seed_values.pop()
    cursor = 0
    native_parts = []
    ranges = []
    for document, chunk_rows, native in loaded:
        start, stop = int(document["start"]), int(document["stop"])
        if start != cursor or stop > len(row_index):
            raise ValueError("Chronos V1 feature chunks have a gap, overlap, or overflow")
        if not np.array_equal(chunk_rows, row_index[start:stop]):
            raise ValueError("Chronos V1 feature chunk row identity changed")
        if native.shape[1:] != (
            len(chronos_v1.CHANNELS), chronos_v1.NUM_SAMPLES,
            chronos_v1.HORIZON_LENGTH,
        ) or native.dtype != np.float32:
            raise ValueError("Chronos V1 feature chunk native geometry changed")
        native_parts.append(native)
        ranges.append({
            "start": start, "stop": stop,
            "batch_size": int(document["batch_size"]),
            "artifact_sha256": document["artifact"]["sha256"],
        })
        cursor = stop
    if cursor != len(row_index):
        raise ValueError("Chronos V1 feature chunks do not cover every frozen row")
    native_output = np.concatenate(native_parts, axis=0)
    features = _flatten(native_output[..., :chronos_v1.EFFECTIVE_HORIZON])
    names = _v1_names()
    if len(names) != features.shape[1]:
        raise RuntimeError("Chronos V1 feature names do not match finalized samples")
    arrays = {
        "row_index": row_index.astype(np.int64, copy=False),
        "features": features,
        "feature_names": names,
        "native_output": native_output,
    }
    metadata = build_metadata(
        route_key=chronos_v1.ROUTE_KEY,
        feature_kind="native_forecast_sample_tensor",
        information_view="common_information_512_v1",
        native_output=native_output,
        native_axes=["row", "channel", "sample", "horizon"],
        native_semantics="adapted_chronos_v1_seeded_native_64_step_forecast_samples_v1",
        feature_construction_id="admitted_first_16_positions_then_c_order_flatten_v1",
        feature_construction_parameters={
            "seed": extraction_seed,
            "batch_seed_rule": "base_seed_plus_global_batch_start_position_v1",
            "num_samples": int(chronos_v1.NUM_SAMPLES),
            "native_horizon": int(chronos_v1.HORIZON_LENGTH),
            "admitted_feature_horizon": int(chronos_v1.EFFECTIVE_HORIZON),
            "future_stub": (
                "repeat_last_observed_bar_64_native_parent_"
                "not_consumed_by_forecast_context_v1"
            ),
            "channel_order": list(chronos_v1.CHANNELS),
            "chunks": ranges,
        },
        features=features,
        sample_manifest=sample_manifest,
        selection_manifest=selection_manifest,
        context_manifest=context_manifest,
        pilot_evidence_path=pilot_path,
        deployment_bundle_path=bundle_path,
        executor_path=chronos_v1.__file__,
        source_path=Path(__file__).resolve(),
    )
    manifest = save_feature_table(output_path, arrays, metadata)
    result = {
        "status": "complete",
        "route_key": chronos_v1.ROUTE_KEY,
        "rows": int(len(row_index)),
        "feature_count": int(features.shape[1]),
        "native_shape": list(native_output.shape),
        "chunks": len(ranges),
        "artifact": manifest["artifact"],
        "content_fingerprint": manifest["content_fingerprint"],
        "promotion_admitted": False,
        "full_training_admitted": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    output_path = Path(args.output).expanduser().resolve()
    if (
        output_path.exists()
        and not args.overwrite
        and not args.chunk_output
        and not args.finalize_chunk
    ):
        raise FileExistsError(f"native downstream output already exists: {output_path}")

    sample, sample_manifest = load_balanced_sample(args.sample)
    selection, selection_manifest = load_row_selection(
        args.row_selection, sample_manifest=sample_manifest,
    )
    contexts, context_manifest = load_downstream_contexts(
        args.contexts, sample_manifest=sample_manifest,
    )
    row_index = np.asarray(selection["row_index"], np.int64)
    total_rows = len(row_index)
    if not np.array_equal(
        np.asarray(contexts["row_index"], np.int64),
        np.arange(len(sample["stream_id"]), dtype=np.int64),
    ):
        raise ValueError("downstream context row identity is not canonical")
    common_context = np.asarray(contexts["context"])[row_index]
    common_time = np.asarray(contexts["context_time_ns"])[row_index]
    pilot_path = Path(args.pilot_evidence).expanduser().resolve()
    pilot = load_route_pilot_evidence(pilot_path)
    if pilot["route_key"] != args.route_key or pilot["native_objective_survived"] is not True:
        raise ValueError("native downstream route/pilot identity mismatch")
    smoke_path = Path(pilot["artifacts"]["smoke_evidence"]["path"]).resolve()
    smoke = load_route_smoke_evidence(smoke_path)
    if smoke["route_key"] != args.route_key or smoke["smoke_admitted"] is not True:
        raise ValueError("native downstream route smoke is not admitted")
    bundle_path = Path(pilot["artifacts"]["deployment_bundle"]["path"]).resolve()
    model_snapshot = Path(pilot["artifacts"]["model_snapshot"]["path"]).resolve()
    source_runtime = Path(smoke["artifacts"]["source_runtime"]["path"]).resolve()

    if args.finalize_chunk:
        if args.route_key != chronos_v1.ROUTE_KEY:
            raise ValueError("chunk finalization is currently supported only for Chronos V1")
        return _finalize_v1_chunks(
            output_path=output_path,
            chunk_paths=list(args.finalize_chunk),
            row_index=row_index,
            sample_manifest=sample_manifest,
            selection_manifest=selection_manifest,
            context_manifest=context_manifest,
            pilot_path=pilot_path,
            pilot=pilot,
            bundle_path=bundle_path,
            overwrite=args.overwrite,
        )

    chunk_start = 0 if args.chunk_start is None else int(args.chunk_start)
    chunk_stop = total_rows if args.chunk_stop is None else int(args.chunk_stop)
    if args.chunk_output:
        if args.route_key != chronos_v1.ROUTE_KEY:
            raise ValueError("chunk extraction is currently supported only for Chronos V1")
        if not 0 <= chunk_start < chunk_stop <= total_rows:
            raise ValueError("native downstream chunk bounds are invalid")
        row_index = row_index[chunk_start:chunk_stop]
        common_context = common_context[chunk_start:chunk_stop]
        common_time = common_time[chunk_start:chunk_stop]
    elif args.chunk_start is not None or args.chunk_stop is not None:
        raise ValueError("chunk bounds require --chunk-output")

    extraction_offset = chunk_start if args.chunk_output else 0
    parameters: dict[str, Any] = {
        "batch_size": int(args.batch_size),
        "device": str(args.device),
    }
    if args.route_key == chronos_bolt.ROUTE_KEY:
        _, model, _ = chronos_bolt.load_export_bundle(
            bundle_path, snapshot=model_snapshot, device=args.device,
        )
        native_output = _extract_batches(
            len(row_index), args.batch_size,
            lambda start, stop: _tensor_numpy(
                chronos_bolt.direct_quantiles(
                    model,
                    _causal_parent(common_context[start:stop], stub_bars=16),
                    device=args.device,
                )
            ).astype(np.float32),
        )
        axes = ["row", "channel", "horizon", "quantile"]
        semantics = "adapted_chronos_bolt_native_quantiles_first_16_positions_v1"
        feature_kind = "native_forecast_quantile_tensor"
        names = _bolt_names()
        parameters.update({
            "future_stub": "repeat_last_observed_bar_16_not_consumed_v1",
            "channel_order": list(chronos_bolt.CHANNELS),
            "quantile_levels": list(chronos_bolt.QUANTILES),
        })
        executor_path = chronos_bolt.__file__
    elif args.route_key == chronos_v1.ROUTE_KEY:
        pipeline, _, _ = chronos_v1.load_export_bundle(
            bundle_path, snapshot=model_snapshot, device=args.device,
        )
        native_output = _extract_batches(
            len(row_index), args.batch_size,
            lambda start, stop: _tensor_numpy(
                chronos_v1.forecast_samples(
                    pipeline,
                    _causal_parent(common_context[start:stop], stub_bars=64),
                    device=args.device,
                    seed=int(args.seed + extraction_offset + start),
                    num_samples=chronos_v1.NUM_SAMPLES,
                )
            ).astype(np.float32),
        )
        axes = ["row", "channel", "sample", "horizon"]
        semantics = "adapted_chronos_v1_seeded_native_64_step_forecast_samples_v1"
        feature_kind = "native_forecast_sample_tensor"
        names = _v1_names()
        parameters.update({
            "seed": int(args.seed),
            "batch_seed_rule": "base_seed_plus_global_batch_start_position_v1",
            "num_samples": int(chronos_v1.NUM_SAMPLES),
            "native_horizon": int(chronos_v1.HORIZON_LENGTH),
            "admitted_feature_horizon": int(chronos_v1.EFFECTIVE_HORIZON),
            "future_stub": "repeat_last_observed_bar_64_native_parent_not_consumed_by_forecast_context_v1",
            "channel_order": list(chronos_v1.CHANNELS),
        })
        executor_path = chronos_v1.__file__
    elif args.route_key == moment_reconstruction.ROUTE_KEY:
        model, _ = moment_reconstruction.load_export_bundle(
            bundle_path,
            snapshot=model_snapshot,
            source_runtime=source_runtime,
            device=args.device,
        )
        native_output = _extract_batches(
            len(row_index), args.batch_size,
            lambda start, stop: _tensor_numpy(
                moment_reconstruction.mean_embedding(
                    model, common_context[start:stop], device=args.device,
                )
            ).astype(np.float32),
        )
        axes = ["row", "embedding"]
        semantics = "adapted_moment_official_masked_mean_embedding_v1"
        feature_kind = "official_embedding"
        names = _moment_names()
        parameters.update({"embedding_dimension": 512, "reduction": "official_mean"})
        executor_path = moment_reconstruction.__file__
    elif args.route_key == kronos_predictor.ROUTE_KEY:
        tokenizer_snapshot = Path(
            smoke["artifacts"]["tokenizer_snapshot"]["path"]
        ).resolve()
        parent_evidence = Path(
            smoke["artifacts"]["parent_route_evidence"]["path"]
        ).resolve()
        parent_bundle = Path(
            smoke["artifacts"]["parent_route_bundle"]["path"]
        ).resolve()
        loaded, _ = kronos_predictor.load_export_bundle(
            bundle_path,
            model_snapshot=model_snapshot,
            tokenizer_snapshot=tokenizer_snapshot,
            source_runtime=source_runtime,
            parent_pilot_evidence=parent_evidence,
            parent_tokenizer_bundle=parent_bundle,
            device=args.device,
        )
        future_time = _future_timestamps(common_time)
        native_output = _extract_batches(
            len(row_index), args.batch_size,
            lambda start, stop: kronos_predictor.public_greedy_forecast(
                loaded,
                common_context[start:stop],
                common_time[start:stop],
                future_time[start:stop],
            ).astype(np.float32),
        )
        axes = ["row", "horizon", "channel"]
        semantics = "adapted_kronos_mini_public_greedy_joint_ohlcva_forecast_v1"
        feature_kind = "native_joint_forecast_tensor"
        names = _predictor_names()
        parameters.update({
            "channel_order": list(kronos_predictor.NATIVE_CHANNELS),
            "venue_timezone": kronos_predictor.VENUE_TIMEZONE,
            "temperature": 1.0, "top_k": 1, "top_p": 1.0, "sample_count": 1,
        })
        executor_path = kronos_predictor.__file__
    else:  # pragma: no cover - argparse closes this branch
        raise ValueError(f"unsupported route: {args.route_key}")

    if args.chunk_output:
        result = _save_chunk(
            args.chunk_output,
            route_key=args.route_key,
            start=chunk_start,
            stop=chunk_stop,
            total_rows=total_rows,
            row_index=row_index,
            native_output=native_output,
            batch_size=args.batch_size,
            seed=args.seed,
            selection_manifest=selection_manifest,
            context_manifest=context_manifest,
            pilot=pilot,
            executor_path=executor_path,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return result

    if args.route_key == chronos_v1.ROUTE_KEY:
        feature_source = native_output[..., :chronos_v1.EFFECTIVE_HORIZON]
        feature_construction_id = "admitted_first_16_positions_then_c_order_flatten_v1"
    else:
        feature_source = native_output
        feature_construction_id = "exact_native_tensor_c_order_flatten_v1"
    features = _flatten(feature_source)
    if len(names) != features.shape[1]:
        raise RuntimeError("native downstream feature names do not match flattened output")
    arrays = {
        "row_index": row_index.astype(np.int64, copy=False),
        "features": features,
        "feature_names": names,
        "native_output": native_output,
    }
    metadata = build_metadata(
        route_key=args.route_key,
        feature_kind=feature_kind,
        information_view="common_information_512_v1",
        native_output=native_output,
        native_axes=axes,
        native_semantics=semantics,
        feature_construction_id=feature_construction_id,
        feature_construction_parameters=parameters,
        features=features,
        sample_manifest=sample_manifest,
        selection_manifest=selection_manifest,
        context_manifest=context_manifest,
        pilot_evidence_path=pilot_path,
        deployment_bundle_path=bundle_path,
        executor_path=executor_path,
        source_path=Path(__file__).resolve(),
    )
    manifest = save_feature_table(output_path, arrays, metadata)
    result = {
        "status": "complete",
        "route_key": args.route_key,
        "rows": int(len(row_index)),
        "feature_count": int(features.shape[1]),
        "native_shape": list(native_output.shape),
        "artifact": manifest["artifact"],
        "content_fingerprint": manifest["content_fingerprint"],
        "promotion_admitted": False,
        "full_training_admitted": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-key", choices=SUPPORTED_ROUTES, required=True)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--row-selection", required=True)
    parser.add_argument("--contexts", required=True)
    parser.add_argument("--pilot-evidence", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--chunk-output")
    parser.add_argument("--chunk-start", type=int)
    parser.add_argument("--chunk-stop", type=int)
    parser.add_argument("--finalize-chunk", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    run(args)


if __name__ == "__main__":
    main()
