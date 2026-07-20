"""Hash-bound native-output tables for the common downstream ruler.

A feature table preserves each route's native output tensor and separately declares
how that tensor is exposed to the low-variance ruler.  The artifact binds exact sample
rows, raw contexts, route smoke/pilot evidence, deployment bundle, and executor bytes.
It cannot grant route promotion, full training, OOS access, deployment, or trading.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from futures_foundation.finetune.downstream_contexts import load_downstream_contexts
from futures_foundation.finetune.downstream_sample import (
    load_balanced_sample,
    load_row_selection,
)
from futures_foundation.finetune.native_contracts import NativeContractError
from futures_foundation.finetune.native_route_pilot import load_route_pilot_evidence


SCHEMA_VERSION = "ffm_native_downstream_feature_table_v1"
MANIFEST_VERSION = "ffm_native_downstream_feature_manifest_v1"
_ARRAY_KEYS = {"row_index", "features", "feature_names", "native_output"}
_METADATA_KEYS = {
    "schema_version", "status", "oos_read", "route_key", "feature_kind",
    "information_view", "native_output", "feature_construction", "rows",
    "feature_count", "sample", "row_selection", "contexts", "pilot_evidence",
    "deployment_bundle", "executor", "source",
}


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _fingerprint(arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    )
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _identity(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"native downstream source must be a regular file: {source}")
    return {
        "path": str(source),
        "sha256": _sha256(source),
        "bytes": int(source.stat().st_size),
    }


def validate_feature_table(
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    *,
    verify_sources: bool = True,
) -> None:
    if not isinstance(arrays, Mapping) or set(arrays) != _ARRAY_KEYS:
        raise ValueError("native downstream feature array closure is invalid")
    if not isinstance(metadata, Mapping) or set(metadata) != _METADATA_KEYS:
        raise ValueError("native downstream feature metadata closure is invalid")
    if (
        metadata["schema_version"] != SCHEMA_VERSION
        or metadata["status"] != "complete"
        or metadata["oos_read"] is not False
    ):
        raise ValueError("native downstream feature status/scope is invalid")
    route_key = metadata["route_key"]
    if not isinstance(route_key, str) or route_key.count(":") != 2:
        raise ValueError("native downstream route key is malformed")
    if metadata["information_view"] not in {
        "common_information_512_v1", "native_best_528_past_v1",
    }:
        raise ValueError("native downstream information view is unsupported")
    if not isinstance(metadata["feature_kind"], str) or not metadata["feature_kind"]:
        raise ValueError("native downstream feature kind is required")

    row_index = np.asarray(arrays["row_index"])
    features = np.asarray(arrays["features"])
    feature_names = np.asarray(arrays["feature_names"])
    native_output = np.asarray(arrays["native_output"])
    rows = int(metadata["rows"])
    feature_count = int(metadata["feature_count"])
    if row_index.dtype != np.int64 or row_index.shape != (rows,):
        raise ValueError("native downstream row index must be int64 [rows]")
    if len(row_index) == 0 or np.any(np.diff(row_index) <= 0):
        raise ValueError("native downstream row index must be increasing and unique")
    if features.dtype != np.float32 or features.shape != (rows, feature_count):
        raise ValueError("native downstream features must be float32 [rows,features]")
    if not np.isfinite(features).all():
        raise ValueError("native downstream features contain non-finite values")
    if feature_names.ndim != 1 or len(feature_names) != feature_count:
        raise ValueError("native downstream feature-name vector is misaligned")
    names = feature_names.astype(str)
    if np.any(np.char.str_len(names) == 0) or len(np.unique(names)) != len(names):
        raise ValueError("native downstream feature names must be non-empty and unique")
    if native_output.shape[0] != rows or native_output.dtype.kind not in "iufb":
        raise ValueError("native downstream output must be aligned numeric rows")
    if native_output.dtype.kind in "f" and not np.isfinite(native_output).all():
        raise ValueError("native downstream output contains non-finite values")

    native_contract = metadata["native_output"]
    if not isinstance(native_contract, Mapping) or set(native_contract) != {
        "dtype", "shape", "axes", "semantics",
    }:
        raise ValueError("native downstream output contract is malformed")
    if (
        native_contract["dtype"] != str(native_output.dtype)
        or list(native_output.shape) != list(native_contract["shape"])
        or not isinstance(native_contract["axes"], list)
        or len(native_contract["axes"]) != native_output.ndim
        or native_contract["axes"][0] != "row"
        or not isinstance(native_contract["semantics"], str)
        or not native_contract["semantics"]
    ):
        raise ValueError("native downstream output contract disagrees with the array")
    construction = metadata["feature_construction"]
    if not isinstance(construction, Mapping) or set(construction) != {
        "id", "input_axes", "output_shape", "learned_parameters", "parameters",
    }:
        raise ValueError("native downstream feature construction is malformed")
    if (
        not isinstance(construction["id"], str)
        or not construction["id"]
        or construction["input_axes"] != native_contract["axes"]
        or construction["output_shape"] != [rows, feature_count]
        or construction["learned_parameters"] is not False
        or not isinstance(construction["parameters"], Mapping)
    ):
        raise ValueError("native downstream feature construction is inconsistent")

    for name in (
        "sample", "row_selection", "contexts", "pilot_evidence",
        "deployment_bundle", "executor", "source",
    ):
        value = metadata[name]
        if not isinstance(value, Mapping) or not isinstance(value.get("path"), str):
            raise ValueError(f"native downstream source identity is malformed: {name}")
        if not isinstance(value.get("sha256"), str) or len(value["sha256"]) != 64:
            raise ValueError(f"native downstream source SHA-256 is malformed: {name}")

    if not verify_sources:
        return
    sample, sample_manifest = load_balanced_sample(metadata["sample"]["path"])
    if (
        sample_manifest["artifact"]["sha256"] != metadata["sample"]["sha256"]
        or sample_manifest["content_fingerprint"]
        != metadata["sample"]["content_fingerprint"]
    ):
        raise ValueError("native downstream sample identity changed")
    selection, selection_manifest = load_row_selection(
        metadata["row_selection"]["path"], sample_manifest=sample_manifest,
    )
    if (
        selection_manifest["artifact"]["sha256"] != metadata["row_selection"]["sha256"]
        or selection_manifest["content_fingerprint"]
        != metadata["row_selection"]["content_fingerprint"]
        or not np.array_equal(np.asarray(selection["row_index"], np.int64), row_index)
    ):
        raise ValueError("native downstream row selection changed")
    contexts, context_manifest = load_downstream_contexts(
        metadata["contexts"]["path"], sample_manifest=sample_manifest,
    )
    del sample, contexts
    if (
        context_manifest["artifact"]["sha256"] != metadata["contexts"]["sha256"]
        or context_manifest["content_fingerprint"]
        != metadata["contexts"]["content_fingerprint"]
    ):
        raise ValueError("native downstream context identity changed")
    pilot_path = Path(metadata["pilot_evidence"]["path"]).resolve()
    if _sha256(pilot_path) != metadata["pilot_evidence"]["sha256"]:
        raise ValueError("native downstream pilot evidence bytes changed")
    pilot = load_route_pilot_evidence(pilot_path)
    if (
        pilot["route_key"] != route_key
        or pilot["pilot_completed"] is not True
        or pilot["native_objective_survived"] is not True
        or pilot["evidence_sha256"] != metadata["pilot_evidence"]["content_sha256"]
    ):
        raise ValueError("native downstream pilot evidence is not an admitted survivor")
    declared_bundle = pilot["artifacts"]["deployment_bundle"]
    if (
        Path(str(declared_bundle["path"])).resolve()
        != Path(metadata["deployment_bundle"]["path"]).resolve()
        or declared_bundle["sha256"] != metadata["deployment_bundle"]["sha256"]
        or _sha256(metadata["deployment_bundle"]["path"])
        != metadata["deployment_bundle"]["sha256"]
    ):
        raise ValueError("native downstream deployment bundle changed")
    for name in ("executor", "source"):
        if _sha256(metadata[name]["path"]) != metadata[name]["sha256"]:
            raise ValueError(f"native downstream implementation bytes changed: {name}")


def build_metadata(
    *,
    route_key: str,
    feature_kind: str,
    information_view: str,
    native_output: np.ndarray,
    native_axes: list[str],
    native_semantics: str,
    feature_construction_id: str,
    feature_construction_parameters: Mapping[str, Any],
    features: np.ndarray,
    sample_manifest: Mapping[str, Any],
    selection_manifest: Mapping[str, Any],
    context_manifest: Mapping[str, Any],
    pilot_evidence_path: str | Path,
    deployment_bundle_path: str | Path,
    executor_path: str | Path,
    source_path: str | Path,
) -> dict[str, Any]:
    pilot_path = Path(pilot_evidence_path).expanduser().resolve()
    pilot = load_route_pilot_evidence(pilot_path)
    if pilot["route_key"] != route_key or pilot["native_objective_survived"] is not True:
        raise ValueError("native downstream metadata requires a surviving route pilot")
    bundle_identity = pilot["artifacts"]["deployment_bundle"]
    bundle_path = Path(deployment_bundle_path).expanduser().resolve()
    if (
        Path(str(bundle_identity["path"])).resolve() != bundle_path
        or bundle_identity["sha256"] != _sha256(bundle_path)
    ):
        raise ValueError("native downstream deployment bundle differs from pilot evidence")
    output = np.asarray(native_output)
    matrix = np.asarray(features)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "oos_read": False,
        "route_key": route_key,
        "feature_kind": feature_kind,
        "information_view": information_view,
        "native_output": {
            "dtype": str(output.dtype),
            "shape": list(output.shape),
            "axes": list(native_axes),
            "semantics": native_semantics,
        },
        "feature_construction": {
            "id": feature_construction_id,
            "input_axes": list(native_axes),
            "output_shape": list(matrix.shape),
            "learned_parameters": False,
            "parameters": deepcopy(dict(feature_construction_parameters)),
        },
        "rows": int(matrix.shape[0]),
        "feature_count": int(matrix.shape[1]),
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
        "contexts": {
            "path": context_manifest["artifact"]["path"],
            "sha256": context_manifest["artifact"]["sha256"],
            "content_fingerprint": context_manifest["content_fingerprint"],
        },
        "pilot_evidence": {
            "path": str(pilot_path),
            "sha256": _sha256(pilot_path),
            "content_sha256": pilot["evidence_sha256"],
        },
        "deployment_bundle": {
            "path": str(bundle_path),
            "sha256": bundle_identity["sha256"],
            "bytes": int(bundle_identity.get("bytes", bundle_identity.get("size_bytes", bundle_path.stat().st_size))),
        },
        "executor": _identity(executor_path),
        "source": _identity(source_path),
    }


def save_feature_table(
    path: str | Path,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    validate_feature_table(arrays, metadata, verify_sources=True)
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(target) + f".{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, target)
    fingerprint = _fingerprint(arrays, metadata)
    manifest = {
        "schema_version": MANIFEST_VERSION,
        "status": "complete",
        "oos_read": False,
        "artifact": {
            "path": str(target),
            "sha256": _sha256(target),
            "bytes": int(target.stat().st_size),
        },
        "content_fingerprint": fingerprint,
        "metadata": deepcopy(dict(metadata)),
    }
    manifest_path = Path(str(target) + ".manifest.json")
    temporary_manifest = Path(str(manifest_path) + f".{os.getpid()}.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_manifest, manifest_path)
    return manifest


def load_feature_table(
    path: str | Path,
    *,
    verify_sources: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    manifest_path = Path(str(source) + ".manifest.json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NativeContractError(f"cannot read native downstream feature manifest: {manifest_path}") from exc
    if (
        manifest.get("schema_version") != MANIFEST_VERSION
        or manifest.get("status") != "complete"
        or manifest.get("oos_read") is not False
    ):
        raise NativeContractError("native downstream feature manifest is unsupported")
    if _sha256(source) != manifest.get("artifact", {}).get("sha256"):
        raise NativeContractError("native downstream feature artifact hash mismatch")
    with np.load(source, allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in saved.files}
    metadata = manifest.get("metadata")
    if _fingerprint(arrays, metadata) != manifest.get("content_fingerprint"):
        raise NativeContractError("native downstream feature fingerprint mismatch")
    validate_feature_table(arrays, metadata, verify_sources=verify_sources)
    for value in arrays.values():
        value.setflags(write=False)
    return arrays, manifest


__all__ = [
    "MANIFEST_VERSION", "SCHEMA_VERSION", "build_metadata", "load_feature_table",
    "save_feature_table", "validate_feature_table",
]
