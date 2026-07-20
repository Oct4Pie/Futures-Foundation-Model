#!/usr/bin/env python3
"""Extract current native outputs for the four newly surviving bounded pilots.

All inputs are fixed to the existing common-information 512-bar development screen.  Model,
source-runtime, smoke, pilot, and deployment paths are recovered only from verified pilot
lineage.  Native tensors are retained in full and exposed through exact C-order flattening;
no learned feature transform is used here.  This command cannot grant promotion, full
training, OOS access, deployment, paper trading, or live trading.
"""
from __future__ import annotations

import argparse
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
from futures_foundation.finetune.downstream_sample import load_balanced_sample, load_row_selection
from futures_foundation.finetune.native_downstream_features import build_metadata, save_feature_table
from futures_foundation.finetune.native_route_pilot import load_route_pilot_evidence
from futures_foundation.finetune.native_route_smoke import load_route_smoke_evidence
from futures_foundation.finetune.routes import mantis_native, moment_tasks

SAMPLE = ROOT / "output/foundation_tournament/downstream_current_1min_v4/balanced_sample.npz"
ROW_SELECTION = ROOT / "output/foundation_tournament/downstream_current_1min_v4/representation_rows_300.npz"
CONTEXTS = ROOT / "output/foundation_tournament/downstream_current_1min_v4/contexts.npz"
OUTPUT_ROOT = ROOT / "output/foundation_tournament/downstream_current_1min_v4/representation_screen"

ROUTES = {
    "y01": {
        "route_key": "mantis_v1:R:official_crop_resize_contrastive",
        "pilot": ROOT / "output/native_training_pilot/mantis_v1_official_crop_resize_contrastive_1min/pilot_evidence.json",
        "output": OUTPUT_ROOT / "features_mantis_v1_contrastive.npz",
        "batch_size": 64,
    },
    "y02": {
        "route_key": "mantis_v2:R:official_crop_resize_contrastive",
        "pilot": ROOT / "output/native_training_pilot/mantis_v2_official_crop_resize_contrastive_1min/pilot_evidence.json",
        "output": OUTPUT_ROOT / "features_mantis_v2_contrastive.npz",
        "batch_size": 32,
    },
    "y03": {
        "route_key": "moment_small:F:forecast_full_raw_mse",
        "pilot": ROOT / "output/native_training_pilot/moment_small_forecast_full_raw_mse_1min/pilot_evidence.json",
        "output": OUTPUT_ROOT / "features_moment_forecast_full.npz",
        "batch_size": 64,
    },
    "y04": {
        "route_key": "moment_small:F:forecast_head_only_raw_mse",
        "pilot": ROOT / "output/native_training_pilot/moment_small_forecast_head_only_raw_mse_1min/pilot_evidence.json",
        "output": OUTPUT_ROOT / "features_moment_forecast_head.npz",
        "batch_size": 64,
    },
}


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().float().cpu().numpy()
    return np.asarray(value)


def _extract_batches(
    values: np.ndarray,
    *,
    batch_size: int,
    run: Callable[[np.ndarray], Any],
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for start in range(0, len(values), batch_size):
        stop = min(start + batch_size, len(values))
        chunk = _numpy(run(values[start:stop])).astype(np.float32, copy=False)
        if chunk.shape[0] != stop - start or not np.isfinite(chunk).all():
            raise RuntimeError("extended downstream native output is malformed or non-finite")
        chunks.append(chunk)
        print(f"[extended-features] rows={stop:,}/{len(values):,}", flush=True)
    return np.concatenate(chunks, axis=0)


def _mantis_names(dimension: int) -> np.ndarray:
    return np.asarray([
        f"mantis_channel_{channel}_embedding_{index:04d}"
        for channel in mantis_native.CHANNELS
        for index in range(int(dimension))
    ])


def _moment_names() -> np.ndarray:
    return np.asarray([
        f"moment_forecast_{channel}_horizon_{horizon:02d}"
        for channel in moment_tasks.CHANNELS
        for horizon in range(1, moment_tasks.HORIZON_LENGTH + 1)
    ])


def run(alias: str, *, overwrite: bool, device: str) -> dict[str, Any]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    spec = ROUTES[alias]
    route_key = str(spec["route_key"])
    output = Path(spec["output"]).resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(f"extended downstream feature table already exists: {output}")

    sample, sample_manifest = load_balanced_sample(SAMPLE)
    selection, selection_manifest = load_row_selection(
        ROW_SELECTION, sample_manifest=sample_manifest,
    )
    contexts, context_manifest = load_downstream_contexts(
        CONTEXTS, sample_manifest=sample_manifest,
    )
    row_index = np.asarray(selection["row_index"], np.int64)
    if not np.array_equal(
        np.asarray(contexts["row_index"], np.int64),
        np.arange(len(sample["stream_id"]), dtype=np.int64),
    ):
        raise ValueError("extended downstream context row identity is not canonical")
    common_context = np.asarray(contexts["context"], np.float32)[row_index]
    if common_context.shape != (len(row_index), 512, 5):
        raise ValueError("extended downstream common context geometry changed")

    pilot_path = Path(spec["pilot"]).resolve()
    pilot = load_route_pilot_evidence(pilot_path)
    if pilot["route_key"] != route_key or pilot["native_objective_survived"] is not True:
        raise ValueError("extended downstream route requires a surviving current pilot")
    smoke_path = Path(pilot["artifacts"]["smoke_evidence"]["path"]).resolve()
    smoke = load_route_smoke_evidence(smoke_path)
    if smoke["route_key"] != route_key or smoke["smoke_admitted"] is not True:
        raise ValueError("extended downstream smoke lineage is not admitted")
    bundle_path = Path(pilot["artifacts"]["deployment_bundle"]["path"]).resolve()
    model_snapshot = Path(pilot["artifacts"]["model_snapshot"]["path"]).resolve()
    source_runtime = Path(smoke["artifacts"]["source_runtime"]["path"]).resolve()

    if route_key in mantis_native.ROUTES:
        loaded = mantis_native.load_route(
            route_key,
            model_snapshot=model_snapshot,
            source_runtime=source_runtime,
            device=device,
        )
        mantis_native.load_export_bundle(bundle_path, loaded=loaded)
        native_output = _extract_batches(
            common_context,
            batch_size=int(spec["batch_size"]),
            run=lambda batch: mantis_native.deployment_output(loaded, batch, device=device),
        )
        if native_output.ndim != 3 or native_output.shape[1] != len(mantis_native.CHANNELS):
            raise RuntimeError("Mantis downstream representation geometry changed")
        axes = ["row", "channel", "embedding"]
        semantics = (
            "adapted_mantis_v1_final_cls_per_channel_v1"
            if mantis_native.ROUTES[route_key]["version"] == 1
            else "adapted_mantis_v2_layer2_combined_per_channel_v1"
        )
        feature_kind = "official_per_channel_representation"
        names = _mantis_names(native_output.shape[2])
        parameters = {
            "channel_order": list(mantis_native.CHANNELS),
            "embedding_dimension": int(native_output.shape[2]),
            "reduction": (
                "final_cls" if mantis_native.ROUTES[route_key]["version"] == 1
                else "layer2_cls_mean_combined"
            ),
            "batch_size": int(spec["batch_size"]),
            "device": device,
        }
        executor_path = mantis_native.__file__
    elif route_key in moment_tasks.ROUTES:
        loaded = moment_tasks.load_route(
            route_key,
            model_snapshot=model_snapshot,
            source_runtime=source_runtime,
            device=device,
        )
        moment_tasks.load_export_bundle(bundle_path, loaded=loaded)
        native_output = _extract_batches(
            common_context,
            batch_size=int(spec["batch_size"]),
            run=lambda batch: moment_tasks.deployment_output(loaded, batch),
        )
        if native_output.shape != (
            len(common_context), len(moment_tasks.CHANNELS), moment_tasks.HORIZON_LENGTH,
        ):
            raise RuntimeError("MOMENT downstream forecast geometry changed")
        axes = ["row", "channel", "horizon"]
        semantics = "adapted_moment_raw_16_bar_multichannel_forecast_v1"
        feature_kind = "native_raw_forecast_tensor"
        names = _moment_names()
        parameters = {
            "channel_order": list(moment_tasks.CHANNELS),
            "horizon": moment_tasks.HORIZON_LENGTH,
            "surface": moment_tasks.ROUTES[route_key]["surface"],
            "batch_size": int(spec["batch_size"]),
            "device": device,
        }
        executor_path = moment_tasks.__file__
    else:  # pragma: no cover
        raise ValueError(route_key)

    features = np.ascontiguousarray(native_output.reshape(len(native_output), -1), dtype=np.float32)
    if len(names) != features.shape[1]:
        raise RuntimeError("extended downstream feature-name closure changed")
    arrays = {
        "row_index": row_index,
        "features": features,
        "feature_names": names,
        "native_output": native_output,
    }
    metadata = build_metadata(
        route_key=route_key,
        feature_kind=feature_kind,
        information_view="common_information_512_v1",
        native_output=native_output,
        native_axes=axes,
        native_semantics=semantics,
        feature_construction_id="exact_native_tensor_c_order_flatten_v1",
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
    manifest = save_feature_table(output, arrays, metadata)
    return {
        "status": "complete",
        "route_key": route_key,
        "rows": int(len(row_index)),
        "feature_count": int(features.shape[1]),
        "native_shape": list(native_output.shape),
        "artifact": manifest["artifact"],
        "content_fingerprint": manifest["content_fingerprint"],
        "promotion_admitted": False,
        "full_training_admitted": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("alias", choices=sorted(ROUTES))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.alias, overwrite=args.overwrite, device=args.device), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
