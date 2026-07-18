#!/usr/bin/env python3
"""Extract only technically admitted native representation outputs.

This command intentionally does not produce a common 2-D feature matrix.  Mantis keeps
its channel axis, MOMENT returns its official masked mean, and every admitted Chronos
track preserves token embeddings plus its native preprocessing state.  Pooling or
channel fusion belongs to Track C.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from futures_foundation.finetune.foundation_eval import load_window_artifact
from futures_foundation.finetune.foundation_roster import get_arm
from futures_foundation.finetune.native_adapters import (
    chronos2_native_embedding,
    chronos_native_embedding,
    left_pad_channel_first,
    mantis_native_representation,
    moment_native_embedding,
)
from futures_foundation.finetune.native_contracts import (
    add_admission_argument,
    require_admission_from_args,
    technical_runtime_contract,
    validate_runtime_contract,
)
from futures_foundation.finetune.native_runtime_binding import (
    require_distribution_record,
    require_import_origin,
    require_module_within,
    require_same_path,
)
from futures_foundation.finetune.native_parity_runtime import (
    install_python_network_guard,
    runtime_profile_for_arm,
    validate_runtime_profile,
)


SCHEMA = "ffm_native_representation_artifact_v1"


RUNTIME_FACTS = {
    "mantis_v1": {
        "context_length": 512,
        "dtype": "float32",
        "output_layout": "B,C,D",
        "channel_fusion": "forbidden_in_track_R",
    },
    "mantis_v2": {
        "context_length": 512,
        "dtype": "float32",
        "return_transf_layer": 2,
        "output_token": "combined",
        "output_layout": "B,C,D",
        "channel_fusion": "forbidden_in_track_R",
    },
    "moment_small": {
        "context_length": 512,
        "dtype": "float32",
        "reduction": "mean",
        "output_layout": "B,D",
    },
    "chronos_v1": {
        "context_length": 512,
        "dtype": "float32",
        "output": "unpooled_embeddings_and_tokenizer_state",
    },
    "chronos_bolt": {
        "context_length": 512,
        "dtype": "float32",
        "output": "unpooled_embeddings_and_location_scale",
    },
    "chronos_v2": {
        "context_length": 512,
        "dtype": "float32",
        "output": "tokens_and_scaling_state_unpooled",
    },
}


def _representation_artifact_paths(args) -> dict[str, Path]:
    """Build the admission map from the exact paths every extractor consumes."""
    return {
        "model": Path(args.model_snapshot).expanduser().resolve(),
        "source": Path(args.source_repo).expanduser().resolve(),
    }


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: str | Path, value: dict[str, Any]) -> None:
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(destination) + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, destination)


def _atomic_npz(path: str | Path, **values: Any) -> None:
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(destination) + ".tmp.npz")
    np.savez_compressed(temporary, **values)
    os.replace(temporary, destination)


def _git_revision(path: str | Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(Path(path).resolve()), "rev-parse", "HEAD"], text=True
    ).strip()


def _require_source_checkout(path: str | Path | None, expected_revision: str) -> Path:
    if not path:
        raise ValueError("this arm requires --source-repo")
    source = Path(path).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    actual = _git_revision(source)
    if actual != expected_revision:
        raise ValueError(
            f"source revision mismatch: expected {expected_revision}, got {actual}"
        )
    return source


def _module_within(module_file: str | None, source: Path, label: str) -> None:
    if not module_file:
        raise ValueError(f"cannot resolve imported {label} module file")
    resolved = Path(module_file).resolve()
    try:
        resolved.relative_to(source)
    except ValueError as exc:
        raise ValueError(
            f"imported {label} from {resolved}, outside pinned source {source}"
        ) from exc


def _contexts(windows: dict[str, Any]) -> np.ndarray:
    values = np.asarray(windows["context"], dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (512, 5):
        raise ValueError(f"native representations require [B,512,5], got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("native representation contexts contain non-finite values")
    return values


def _extract_mantis(args, contexts: np.ndarray) -> dict[str, np.ndarray]:
    source = _require_source_checkout(
        args.native_artifacts["source"], get_arm(args.arm).source_revision
    )
    require_import_origin("mantis", source, "mantis")
    import mantis
    from mantis.architecture import MantisV1, MantisV2

    arm = get_arm(args.arm)
    require_same_path(args.source_repo, source, "--source-repo")
    _module_within(mantis.__file__, source, "mantis")
    if args.arm == "mantis_v1":
        model = MantisV1(device=args.device).from_pretrained(
            str(args.native_artifacts["model"])
        )
    else:
        model = MantisV2(
            device=args.device, return_transf_layer=2, output_token="combined"
        ).from_pretrained(str(args.native_artifacts["model"]))
    model.eval()
    representation = mantis_native_representation(
        model, contexts, batch_size=args.batch_size, target_length=512
    )
    return {"representation": representation}


def _extract_moment(args, contexts: np.ndarray) -> dict[str, np.ndarray]:
    import torch
    arm = get_arm("moment_small")
    source = _require_source_checkout(args.native_artifacts["source"], arm.source_revision)
    require_import_origin("momentfm", source, "momentfm")
    import momentfm
    from scripts.benchmark_moment import _load_moment

    require_same_path(args.source_repo, source, "--source-repo")
    _module_within(momentfm.__file__, source, "momentfm")
    model = _load_moment(
        source, str(args.native_artifacts["model"]), arm.model_revision, args.device
    )
    values, mask = left_pad_channel_first(contexts, target_length=512)
    pieces = []
    with torch.inference_mode():
        for low in range(0, len(values), args.batch_size):
            high = min(len(values), low + args.batch_size)
            output = moment_native_embedding(
                model,
                torch.as_tensor(values[low:high], device=args.device),
                torch.as_tensor(mask[low:high], device=args.device),
            )
            pieces.append(output.float().cpu().numpy())
    return {
        "representation": np.concatenate(pieces, axis=0).astype(np.float32, copy=False),
        "input_mask": mask,
    }


def _load_chronos_pipeline(args):
    import torch
    from chronos import BaseChronosPipeline

    arm = get_arm(args.arm)
    return BaseChronosPipeline.from_pretrained(
        str(args.native_artifacts["model"]),
        device_map=args.device,
        dtype=torch.float32,
        local_files_only=True,
    )


def _native_numpy(value: Any, *, name: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    array = np.asarray(value, dtype=np.float32)
    if array.ndim < 1:
        raise ValueError(f"{name} must preserve the flattened batch axis")
    return array


def _restore_channel_axis(
    value: Any,
    *,
    batch_size: int,
    channels: int,
    name: str,
) -> np.ndarray:
    array = _native_numpy(value, name=name)
    expected = batch_size * channels
    if array.shape[0] != expected:
        raise ValueError(
            f"{name} row mismatch: expected {expected}, got {array.shape[0]}"
        )
    return array.reshape(batch_size, channels, *array.shape[1:])


def _extract_chronos_v1_bolt(args, contexts: np.ndarray) -> dict[str, np.ndarray]:
    """Preserve concrete Chronos token and channel axes without Track-C pooling."""
    import torch

    pipeline = _load_chronos_pipeline(args)
    channel_first = contexts.transpose(0, 2, 1)
    channels = int(channel_first.shape[1])
    representations: list[np.ndarray] = []
    tokenizer_states: list[np.ndarray] = []
    locations: list[np.ndarray] = []
    scales: list[np.ndarray] = []

    with torch.inference_mode():
        for low in range(0, len(channel_first), args.batch_size):
            high = min(len(channel_first), low + args.batch_size)
            batch_size = high - low
            flattened = np.ascontiguousarray(
                channel_first[low:high].reshape(
                    batch_size * channels, channel_first.shape[-1]
                )
            )
            embeddings, state = chronos_native_embedding(
                pipeline, torch.as_tensor(flattened)
            )
            representations.append(
                _restore_channel_axis(
                    embeddings,
                    batch_size=batch_size,
                    channels=channels,
                    name="representation",
                )
            )
            if args.arm == "chronos_v1":
                if isinstance(state, tuple):
                    raise ValueError("Chronos V1 embed returned unexpected tuple state")
                tokenizer_states.append(
                    _restore_channel_axis(
                        state,
                        batch_size=batch_size,
                        channels=channels,
                        name="tokenizer_state",
                    )
                )
            else:
                if not isinstance(state, tuple) or len(state) != 2:
                    raise ValueError(
                        "Chronos-Bolt embed must return (location, scale) state"
                    )
                locations.append(
                    _restore_channel_axis(
                        state[0],
                        batch_size=batch_size,
                        channels=channels,
                        name="scaling_location",
                    )
                )
                scales.append(
                    _restore_channel_axis(
                        state[1],
                        batch_size=batch_size,
                        channels=channels,
                        name="scaling_scale",
                    )
                )

    result = {
        "representation": np.concatenate(representations, axis=0).astype(
            np.float32, copy=False
        )
    }
    if args.arm == "chronos_v1":
        result["tokenizer_state"] = np.concatenate(tokenizer_states, axis=0).astype(
            np.float32, copy=False
        )
    else:
        result["scaling_location"] = np.concatenate(locations, axis=0).astype(
            np.float32, copy=False
        )
        result["scaling_scale"] = np.concatenate(scales, axis=0).astype(
            np.float32, copy=False
        )
    return result


def _extract_chronos2(args, contexts: np.ndarray) -> dict[str, np.ndarray]:
    import torch
    from chronos import Chronos2Pipeline

    arm = get_arm("chronos_v2")
    pipeline = Chronos2Pipeline.from_pretrained(
        str(args.native_artifacts["model"]),
        device_map=args.device,
        dtype=torch.float32,
        local_files_only=True,
    )
    embeddings: list[np.ndarray] = []
    locations: list[np.ndarray] = []
    scales: list[np.ndarray] = []
    channel_first = contexts.transpose(0, 2, 1)
    for low in range(0, len(channel_first), args.batch_size):
        high = min(len(channel_first), low + args.batch_size)
        batch_embeddings, batch_state = chronos2_native_embedding(
            pipeline,
            torch.as_tensor(channel_first[low:high]),
            batch_size=args.batch_size,
            context_length=512,
        )
        embeddings.extend(item.float().cpu().numpy() for item in batch_embeddings)
        for location, scale in batch_state:
            locations.append(location.float().cpu().numpy())
            scales.append(scale.float().cpu().numpy())
    return {
        "representation": np.stack(embeddings).astype(np.float32, copy=False),
        "scaling_location": np.stack(locations).astype(np.float32, copy=False),
        "scaling_scale": np.stack(scales).astype(np.float32, copy=False),
    }


EXTRACTORS = {
    "mantis_v1": _extract_mantis,
    "mantis_v2": _extract_mantis,
    "moment_small": _extract_moment,
    "chronos_v1": _extract_chronos_v1_bolt,
    "chronos_bolt": _extract_chronos_v1_bolt,
    "chronos_v2": _extract_chronos2,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=tuple(EXTRACTORS), required=True)
    parser.add_argument("--windows", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-repo", required=True)
    parser.add_argument("--model-snapshot", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dtype", choices=("float32",), default="float32")
    add_admission_argument(parser)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    profile = runtime_profile_for_arm(args.arm)
    validate_runtime_profile(profile, args.arm)
    install_python_network_guard("python_socket_deny")
    args.native_artifacts = _representation_artifact_paths(args)
    admission = require_admission_from_args(
        args, arm_key=args.arm, track="R", route=None, require_training=False,
        required_artifacts=args.native_artifacts,
        runtime_controls={
            "device": args.device, "dtype": args.dtype,
            "network_policy": "python_socket_deny", "profile": profile,
        },
    )
    if args.arm.startswith("chronos"):
        package_root = require_distribution_record(
            args.native_artifacts["source"],
            distribution_name="chronos-forecasting",
            package_prefix="chronos",
        )
        require_import_origin("chronos", package_root, "Chronos")
        import chronos
        require_module_within(chronos.__file__, package_root, "Chronos")
    import futures_foundation.finetune.native_adapters as native_adapters_module
    require_module_within(
        native_adapters_module.__file__, ROOT, "native adapter runner"
    )
    runtime_facts = {**RUNTIME_FACTS[args.arm], "dtype": args.dtype}
    runtime = validate_runtime_contract(args.arm, "R", runtime_facts)
    windows, windows_manifest = load_window_artifact(args.windows)
    contexts = _contexts(windows)
    arrays = EXTRACTORS[args.arm](args, contexts)
    if any(len(value) != len(contexts) for value in arrays.values()):
        raise ValueError("native representation arrays do not preserve row cardinality")
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("native representation artifact contains non-finite values")

    metadata = {
        "schema_version": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "arm": get_arm(args.arm).manifest(),
        "track": "R",
        "runtime_contract": runtime,
        "technical_runtime_contract": technical_runtime_contract(args.arm, "R"),
        "windows": {
            "path": str(Path(args.windows).resolve()),
            "sha256": windows_manifest["artifact"]["sha256"],
            "window_fingerprint": windows_manifest["window_fingerprint"],
        },
        "admission": {
            "schema_version": admission["schema_version"],
            "integrity": admission["integrity"],
            "registry_sha256": admission["registry_sha256"],
            "dossier_sha256": admission["dossier_sha256"],
            "evidence_registry_sha256": admission["evidence_registry_sha256"],
            "technical_evidence_id": admission["technical_evidence_id"],
        },
        "array_shapes": {name: list(value.shape) for name, value in arrays.items()},
        "environment": {
            "python": ".".join(map(str, sys.version_info[:3])),
            "numpy": importlib.metadata.version("numpy"),
            "dtype": args.dtype,
            "device": args.device,
        },
        "oos_read": False,
        "custom_pooling": False,
    }
    signature = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output = Path(args.output).resolve()
    _atomic_npz(
        output,
        **arrays,
        signature=np.array(signature),
        metadata=np.array(json.dumps(metadata, sort_keys=True)),
    )
    manifest = {
        **metadata,
        "signature": signature,
        "artifact": {"path": str(output), "sha256": _sha256(output)},
    }
    _atomic_json(str(output) + ".manifest.json", manifest)
    print(json.dumps({"artifact": manifest["artifact"], "shapes": metadata["array_shapes"]}, indent=2))


if __name__ == "__main__":
    main()
