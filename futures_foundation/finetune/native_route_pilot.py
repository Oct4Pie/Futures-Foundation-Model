"""Hash-bound bounded-pilot evidence for exact architecture-native routes.

A pilot proves one route can improve its own native forward objective on an
externally bound, non-OOS development corpus under fixed exposure.  It does not
prove residual value over causal features and therefore cannot authorize model
promotion, full training, deployment, OOS access, or trading.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .native_contracts import NativeContractError, content_sha256, file_sha256
from .native_evidence_bundle import _tree_description, _verify_tree
from .native_family_route_catalog_v2 import catalog_sha256, load_family_route_catalog
from .native_route_smoke import load_route_smoke_evidence
from .native_training_schema_v2 import canonical_route_profile_sha256
from .tournament_data import CACHE_MANIFEST, load_cache_manifest


PILOT_EVIDENCE_SCHEMA = "ffm_native_route_pilot_evidence_v1"
PILOT_POLICY = "bounded_non_oos_native_objective_elimination_v1"
REQUIRED_PILOT_ARTIFACTS = frozenset({
    "model_snapshot",
    "smoke_evidence",
    "cache_manifest",
    "exposure_schedule",
    "training_state",
    "deployment_bundle",
    "raw_report",
    "pilot_runner",
})


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _artifacts(values: Mapping[str, str | Path]) -> dict[str, Any]:
    if not isinstance(values, Mapping) or set(values) != REQUIRED_PILOT_ARTIFACTS:
        raise NativeContractError(
            "route pilot artifact closure must exactly match the required artifacts"
        )
    return {
        name: _tree_description(path)
        for name, path in sorted(values.items())
    }


def _exposure(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "sampling_kind", "train_schedule_sha256", "validation_schedule_sha256",
        "train_examples", "validation_examples", "train_stream_counts",
        "validation_stream_counts", "seed", "validation_seed",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise NativeContractError("route pilot exposure fields mismatch")
    if value["sampling_kind"] != "uniform_stream_then_uniform_window_v1":
        raise NativeContractError("route pilot sampling kind is unsupported")
    for name in ("train_schedule_sha256", "validation_schedule_sha256"):
        item = value[name]
        if not isinstance(item, str) or len(item) != 64:
            raise NativeContractError(f"route pilot {name} is malformed")
    for name in ("train_examples", "validation_examples", "seed", "validation_seed"):
        if type(value[name]) is not int or value[name] < 0:
            raise NativeContractError(f"route pilot {name} must be a nonnegative integer")
    for name in ("train_stream_counts", "validation_stream_counts"):
        counts = value[name]
        if (
            not isinstance(counts, Mapping)
            or not counts
            or any(not isinstance(key, str) or type(count) is not int or count < 0
                   for key, count in counts.items())
        ):
            raise NativeContractError(f"route pilot {name} is malformed")
        expected_total = value[
            "train_examples" if name.startswith("train") else "validation_examples"
        ]
        if sum(counts.values()) != expected_total:
            raise NativeContractError(f"route pilot {name} does not conserve examples")
    return deepcopy(dict(value))


def _metrics(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "vanilla_validation_loss", "adapted_validation_loss",
        "relative_validation_improvement", "required_relative_improvement",
        "best_step", "history",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise NativeContractError("route pilot metric fields mismatch")
    numbers = np.asarray([
        value["vanilla_validation_loss"], value["adapted_validation_loss"],
        value["relative_validation_improvement"], value["required_relative_improvement"],
    ], dtype=np.float64)
    if not np.isfinite(numbers).all():
        raise NativeContractError("route pilot losses/improvements are invalid")
    if not 0.0 <= float(value["required_relative_improvement"]) < 1.0:
        raise NativeContractError("route pilot required improvement is invalid")
    expected = float(
        (numbers[0] - numbers[1]) / max(abs(numbers[0]), 1e-12)
    )
    if not np.isclose(expected, numbers[2], rtol=0.0, atol=1e-12):
        raise NativeContractError("route pilot relative improvement is inconsistent")
    if type(value["best_step"]) is not int or value["best_step"] < 1:
        raise NativeContractError("route pilot best_step is invalid")
    if not isinstance(value["history"], list) or not value["history"]:
        raise NativeContractError("route pilot history is empty")
    return deepcopy(dict(value))


def build_route_pilot_evidence(
    *,
    route_key: str,
    smoke_evidence_path: str | Path,
    executor_path: str | Path,
    executor_entrypoint: str,
    cache_dir: str | Path,
    cache_manifest_sha256: str,
    stream_ids: list[str],
    exposure: Mapping[str, Any],
    metrics: Mapping[str, Any],
    artifacts: Mapping[str, str | Path],
    created_utc: str | None = None,
) -> dict[str, Any]:
    catalog = load_family_route_catalog()
    route = catalog["routes"].get(route_key)
    if not isinstance(route, Mapping):
        raise NativeContractError(f"route pilot references an unknown route: {route_key}")
    smoke = load_route_smoke_evidence(smoke_evidence_path)
    if smoke["route_key"] != route_key or smoke["smoke_admitted"] is not True:
        raise NativeContractError("route pilot requires passing smoke for the same route")
    executor = Path(executor_path).expanduser().resolve()
    if (
        not executor.is_file() or executor.is_symlink()
        or smoke["executor"] != {
            "path": str(executor),
            "sha256": file_sha256(executor),
            "entrypoint": executor_entrypoint,
        }
    ):
        raise NativeContractError("route pilot executor differs from smoke evidence")
    cache_root = Path(cache_dir).expanduser().resolve()
    cache = load_cache_manifest(
        cache_root, expected_manifest_sha256=cache_manifest_sha256,
    )
    available = set(cache["entries"])
    streams = sorted(set(stream_ids))
    if not streams or set(streams) - available:
        raise NativeContractError("route pilot stream set is empty or outside the cache")
    interval = cache["interval"]
    if interval.get("contains_oos") is not False:
        raise NativeContractError("route pilot cache exposes OOS")
    measured = _metrics(metrics)
    survived = (
        measured["relative_validation_improvement"]
        >= measured["required_relative_improvement"]
    )
    document = {
        "schema_version": PILOT_EVIDENCE_SCHEMA,
        "policy": PILOT_POLICY,
        "created_utc": created_utc or _utc_now(),
        "route_key": route_key,
        "route_profile_sha256": canonical_route_profile_sha256(
            str(route["arm_key"]), str(route["track"]), str(route["route_id"]),
        ),
        "catalog_sha256": catalog_sha256(catalog),
        "smoke_evidence": {
            "path": str(Path(smoke_evidence_path).expanduser().resolve()),
            "evidence_sha256": smoke["evidence_sha256"],
        },
        "executor": deepcopy(dict(smoke["executor"])),
        "model_identity": deepcopy(dict(smoke["model_identity"])),
        "data_capability": {
            "cache_manifest_path": str(cache_root / CACHE_MANIFEST),
            "cache_manifest_sha256": cache_manifest_sha256,
            "cache_schema_version": cache["schema_version"],
            "interval": deepcopy(dict(interval)),
            "stream_ids": streams,
            "window_gap_policy": "exact_cadence_hard_boundary_no_session_inference_v1",
            "session_gap_capability": None,
            "oos_read": False,
        },
        "exposure": _exposure(exposure),
        "metrics": measured,
        "artifacts": _artifacts(artifacts),
        "pilot_completed": True,
        "native_objective_survived": bool(survived),
        "promotion_admitted": False,
        "promotion_blockers": [
            "causal_feature_baseline_missing",
            "residual_over_causal_gate_missing",
            "nested_downstream_calibration_missing",
        ],
        "full_training_admitted": False,
        "oos_admitted": False,
        "live_trading_ready": False,
    }
    document["evidence_sha256"] = content_sha256(document)
    return document


def validate_route_pilot_evidence(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise NativeContractError("route pilot evidence must be a mapping")
    expected_fields = {
        "schema_version", "policy", "created_utc", "route_key",
        "route_profile_sha256", "catalog_sha256", "smoke_evidence", "executor",
        "model_identity", "data_capability", "exposure", "metrics", "artifacts",
        "pilot_completed", "native_objective_survived", "promotion_admitted",
        "promotion_blockers", "full_training_admitted", "oos_admitted",
        "live_trading_ready", "evidence_sha256",
    }
    if set(value) != expected_fields:
        raise NativeContractError("route pilot evidence fields mismatch")
    payload = deepcopy(dict(value))
    supplied = payload.pop("evidence_sha256", None)
    if supplied != content_sha256(payload):
        raise NativeContractError("route pilot evidence integrity mismatch")
    if value["schema_version"] != PILOT_EVIDENCE_SCHEMA or value["policy"] != PILOT_POLICY:
        raise NativeContractError("route pilot evidence schema or policy is invalid")
    catalog = load_family_route_catalog()
    route = catalog["routes"].get(value["route_key"])
    if not isinstance(route, Mapping):
        raise NativeContractError("route pilot route is no longer canonical")
    if value["catalog_sha256"] != catalog_sha256(catalog):
        raise NativeContractError("route pilot catalog binding is stale")
    if value["route_profile_sha256"] != canonical_route_profile_sha256(
        str(route["arm_key"]), str(route["track"]), str(route["route_id"]),
    ):
        raise NativeContractError("route pilot profile binding is stale")
    smoke = load_route_smoke_evidence(value["smoke_evidence"]["path"])
    if (
        smoke["route_key"] != value["route_key"]
        or smoke["evidence_sha256"] != value["smoke_evidence"]["evidence_sha256"]
        or smoke["smoke_admitted"] is not True
        or smoke["executor"] != value["executor"]
        or smoke["model_identity"] != value["model_identity"]
    ):
        raise NativeContractError("route pilot smoke parent is stale or substituted")
    executor = value["executor"]
    path = Path(str(executor["path"])).resolve()
    if not path.is_file() or path.is_symlink() or file_sha256(path) != executor["sha256"]:
        raise NativeContractError("route pilot executor bytes changed")
    data = value["data_capability"]
    required_data = {
        "cache_manifest_path", "cache_manifest_sha256", "cache_schema_version",
        "interval", "stream_ids", "window_gap_policy", "session_gap_capability",
        "oos_read",
    }
    if not isinstance(data, Mapping) or set(data) != required_data:
        raise NativeContractError("route pilot data capability fields mismatch")
    if (
        data["window_gap_policy"] != "exact_cadence_hard_boundary_no_session_inference_v1"
        or data["session_gap_capability"] is not None
        or data["oos_read"] is not False
    ):
        raise NativeContractError("route pilot data gap/OOS policy is invalid")
    manifest_path = Path(str(data["cache_manifest_path"])).resolve()
    if manifest_path.name != CACHE_MANIFEST:
        raise NativeContractError("route pilot cache manifest name is invalid")
    cache = load_cache_manifest(
        manifest_path.parent,
        expected_manifest_sha256=str(data["cache_manifest_sha256"]),
    )
    if (
        cache["schema_version"] != data["cache_schema_version"]
        or cache["interval"] != data["interval"]
        or not set(data["stream_ids"]).issubset(cache["entries"])
        or cache["interval"].get("contains_oos") is not False
    ):
        raise NativeContractError("route pilot cache identity or stream closure changed")
    exposure = _exposure(value["exposure"])
    metrics = _metrics(value["metrics"])
    survived = metrics["relative_validation_improvement"] >= metrics["required_relative_improvement"]
    if value["pilot_completed"] is not True or value["native_objective_survived"] is not survived:
        raise NativeContractError("route pilot outcome differs from measured metrics")
    if (
        value["promotion_admitted"] is not False
        or value["full_training_admitted"] is not False
        or value["oos_admitted"] is not False
        or value["live_trading_ready"] is not False
        or value["promotion_blockers"] != [
            "causal_feature_baseline_missing",
            "residual_over_causal_gate_missing",
            "nested_downstream_calibration_missing",
        ]
    ):
        raise NativeContractError("route pilot cannot grant promotion/training/OOS/trading admission")
    artifacts = value["artifacts"]
    if not isinstance(artifacts, Mapping) or set(artifacts) != REQUIRED_PILOT_ARTIFACTS:
        raise NativeContractError("route pilot artifact closure is invalid")
    for name, description in artifacts.items():
        _verify_tree(description, f"route pilot artifact {name}")
    return deepcopy(dict(value))


def load_route_pilot_evidence(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NativeContractError(f"cannot read route pilot evidence: {source}") from exc
    return validate_route_pilot_evidence(value)


__all__ = [
    "PILOT_EVIDENCE_SCHEMA", "PILOT_POLICY", "REQUIRED_PILOT_ARTIFACTS",
    "build_route_pilot_evidence", "load_route_pilot_evidence",
    "validate_route_pilot_evidence",
]
