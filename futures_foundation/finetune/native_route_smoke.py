"""Hash-bound route-specific synthetic smoke evidence.

A smoke report may admit one exact route to the *pilot candidate* queue.  It never
admits market-data training, model promotion, OOS access, deployment, or trading.
Every report is bound to the current catalog route/profile, executor bytes, model
snapshot, raw checkpoint/export artifacts, and the complete required check set.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
from typing import Any, Mapping

from .native_contracts import (
    NativeContractError,
    content_sha256,
    file_sha256,
    load_registry,
)
from .native_evidence_bundle import (
    _normalize_git_origin,
    _tree_description,
    _verify_tree,
)
from .native_family_route_catalog_v2 import catalog_sha256, load_family_route_catalog
from .native_smoke_contract import REQUIRED_SMOKE_CHECKS
from .native_training_schema_v2 import canonical_route_profile_sha256


SMOKE_EVIDENCE_SCHEMA = "ffm_native_route_smoke_evidence_v1"
SMOKE_POLICY = "synthetic_non_oos_non_authorizing_route_smoke_v1"
REQUIRED_SMOKE_ARTIFACTS = frozenset({
    "model_snapshot",
    "source_runtime",
    "synthetic_fixture",
    "synthetic_fixture_manifest",
    "interrupted_state",
    "training_state",
    "deployment_bundle",
    "raw_checks",
    "smoke_runner",
})


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _runtime_environment() -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": platform.python_version(),
        "executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
    }
    try:
        import torch

        result["torch"] = str(torch.__version__)
        result["cuda_runtime"] = str(torch.version.cuda)
        result["cuda_available"] = bool(torch.cuda.is_available())
        result["devices"] = [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": list(torch.cuda.get_device_capability(index)),
                "total_memory": int(torch.cuda.get_device_properties(index).total_memory),
            }
            for index in range(torch.cuda.device_count())
        ]
    except Exception as exc:  # pragma: no cover - runtime capture must not hide evidence
        result["torch_error"] = f"{type(exc).__name__}: {exc}"
    return result


def _check_contract(checks: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(checks, Mapping) or set(checks) != set(REQUIRED_SMOKE_CHECKS):
        raise NativeContractError(
            "route smoke checks must exactly match the required check closure"
        )
    output: dict[str, Any] = {}
    for name in REQUIRED_SMOKE_CHECKS:
        raw = checks[name]
        if not isinstance(raw, Mapping) or set(raw) != {"status", "metrics", "reason"}:
            raise NativeContractError(f"route smoke check {name!r} is malformed")
        status = raw["status"]
        if status not in {"pass", "fail"}:
            raise NativeContractError(f"route smoke check {name!r} has invalid status")
        if not isinstance(raw["metrics"], Mapping):
            raise NativeContractError(f"route smoke check {name!r} metrics must be a mapping")
        reason = raw["reason"]
        if reason is not None and not isinstance(reason, str):
            raise NativeContractError(f"route smoke check {name!r} reason must be text or null")
        output[name] = {
            "status": status,
            "metrics": deepcopy(dict(raw["metrics"])),
            "reason": reason,
        }
    return output


def _validate_source_runtime(
    description: Mapping[str, Any],
    dossier: Mapping[str, Any],
) -> None:
    revision = str(dossier["source_revision"])
    if description.get("kind") == "git_checkout":
        if description.get("head_revision") != revision:
            raise NativeContractError(
                "route smoke Git source revision differs from the inference dossier"
            )
        expected_origin = _normalize_git_origin(str(dossier["source_url"]))
        if description.get("origin") != expected_origin:
            raise NativeContractError(
                "route smoke Git source origin differs from the inference dossier"
            )
        return
    if "==" in revision and description.get("kind") == "directory":
        distribution, version = revision.split("==", 1)
        expected = f"{distribution.replace('-', '_')}-{version}.dist-info".lower()
        actual = Path(str(description.get("path", ""))).name.lower()
        if actual != expected:
            raise NativeContractError(
                "route smoke installed source distribution differs from the inference dossier"
            )
        return
    raise NativeContractError(
        "route smoke source runtime cannot be matched to the inference dossier"
    )


def _parent_route_keys(
    catalog: Mapping[str, Any], route: Mapping[str, Any],
) -> tuple[str, ...]:
    profile = catalog["constraint_profiles"][route["constraint_profile"]]
    lineage = profile["lineage"]
    if lineage["state"] != "resolved":
        return ()
    parent_tags = tuple(lineage["value"].get("parent_artifacts", ()))
    if not parent_tags:
        return ()
    resolved: list[str] = []
    for tag in parent_tags:
        matches = []
        for route_key, candidate in catalog["routes"].items():
            if candidate["arm_key"] != route["arm_key"]:
                continue
            candidate_profile = catalog["constraint_profiles"][
                candidate["constraint_profile"]
            ]
            export = candidate_profile["export"]
            if (
                export["state"] == "resolved"
                and export["value"].get("bundle_tag") == tag
            ):
                matches.append(route_key)
        if len(matches) != 1:
            raise NativeContractError(
                f"route parent artifact {tag!r} does not resolve to one canonical route"
            )
        resolved.append(matches[0])
    return tuple(resolved)


def required_smoke_artifacts(
    dossier: Mapping[str, Any],
    *,
    catalog: Mapping[str, Any] | None = None,
    route: Mapping[str, Any] | None = None,
) -> frozenset[str]:
    required = set(REQUIRED_SMOKE_ARTIFACTS)
    tokenizer = dossier.get("tokenizer")
    if isinstance(tokenizer, Mapping):
        tokenizer_id = tokenizer.get("id")
        tokenizer_revision = tokenizer.get("revision")
        if (
            isinstance(tokenizer_id, str)
            and tokenizer_id
            and tokenizer_id != dossier.get("model_id")
            and tokenizer_revision != "model_revision"
        ):
            required.add("tokenizer_snapshot")
    if catalog is not None and route is not None and _parent_route_keys(catalog, route):
        required.update({"parent_route_evidence", "parent_route_bundle"})
    return frozenset(required)


def _validate_parent_artifacts(
    artifacts: Mapping[str, Any],
    *,
    catalog: Mapping[str, Any],
    route: Mapping[str, Any],
) -> None:
    parent_routes = _parent_route_keys(catalog, route)
    if not parent_routes:
        return
    if len(parent_routes) != 1:
        raise NativeContractError(
            "route smoke currently requires exactly one canonical parent route"
        )
    from .native_route_pilot import load_route_pilot_evidence

    evidence_path = Path(str(artifacts["parent_route_evidence"]["path"])).resolve()
    bundle_path = Path(str(artifacts["parent_route_bundle"]["path"])).resolve()
    parent = load_route_pilot_evidence(evidence_path)
    if (
        parent["route_key"] != parent_routes[0]
        or parent["pilot_completed"] is not True
        or parent["native_objective_survived"] is not True
    ):
        raise NativeContractError(
            "route smoke parent evidence is not a surviving canonical parent pilot"
        )
    declared_bundle = parent["artifacts"]["deployment_bundle"]
    if (
        Path(str(declared_bundle["path"])).resolve() != bundle_path
        or declared_bundle.get("sha256")
        != artifacts["parent_route_bundle"].get("sha256")
    ):
        raise NativeContractError(
            "route smoke parent bundle differs from the parent pilot evidence"
        )


def _artifact_contract(
    artifacts: Mapping[str, str | Path],
    *,
    required: frozenset[str],
) -> dict[str, Any]:
    if not isinstance(artifacts, Mapping) or set(artifacts) != required:
        raise NativeContractError(
            "route smoke evidence artifact closure must exactly match the required artifacts"
        )
    output = {}
    for name, path in sorted(artifacts.items()):
        if not isinstance(name, str) or not name or " " in name:
            raise NativeContractError("route smoke artifact names must be structured tags")
        output[name] = _tree_description(path)
    return output


def build_route_smoke_evidence(
    *,
    route_key: str,
    executor_path: str | Path,
    executor_entrypoint: str,
    checks: Mapping[str, Any],
    artifacts: Mapping[str, str | Path],
    metrics: Mapping[str, Any],
    created_utc: str | None = None,
) -> dict[str, Any]:
    catalog = load_family_route_catalog()
    route = catalog["routes"].get(route_key)
    if not isinstance(route, Mapping):
        raise NativeContractError(f"route smoke references an unknown route: {route_key}")
    registry = load_registry()
    dossier = registry["models"][str(route["arm_key"])]
    executor = Path(executor_path).expanduser().resolve()
    if not executor.is_file() or executor.is_symlink():
        raise NativeContractError("route smoke executor must be a regular file")
    if not isinstance(executor_entrypoint, str) or not executor_entrypoint:
        raise NativeContractError("route smoke executor entrypoint is required")
    checked = _check_contract(checks)
    required_artifacts = required_smoke_artifacts(
        dossier, catalog=catalog, route=route,
    )
    bound_artifacts = _artifact_contract(artifacts, required=required_artifacts)
    _validate_source_runtime(bound_artifacts["source_runtime"], dossier)
    _validate_parent_artifacts(
        bound_artifacts, catalog=catalog, route=route,
    )
    snapshot_path = Path(str(bound_artifacts["model_snapshot"]["path"]))
    if snapshot_path.name != dossier["model_revision"]:
        raise NativeContractError(
            "route smoke model snapshot path does not match the inference dossier revision"
        )
    if "tokenizer_snapshot" in bound_artifacts:
        tokenizer = dossier.get("tokenizer") or {}
        tokenizer_path = Path(str(bound_artifacts["tokenizer_snapshot"]["path"]))
        if tokenizer_path.name != tokenizer.get("revision"):
            raise NativeContractError(
                "route smoke tokenizer snapshot path does not match the inference dossier revision"
            )
    passed = all(item["status"] == "pass" for item in checked.values())
    document = {
        "schema_version": SMOKE_EVIDENCE_SCHEMA,
        "policy": SMOKE_POLICY,
        "created_utc": created_utc or _utc_now(),
        "route_key": route_key,
        "route_profile_sha256": canonical_route_profile_sha256(
            str(route["arm_key"]), str(route["track"]), str(route["route_id"]),
        ),
        "catalog_sha256": catalog_sha256(catalog),
        "executor": {
            "path": str(executor),
            "sha256": file_sha256(executor),
            "entrypoint": executor_entrypoint,
        },
        "model_identity": {
            "model_id": dossier["model_id"],
            "model_revision": dossier["model_revision"],
            "source_revision": dossier["source_revision"],
            "dossier_sha256": content_sha256(dossier),
        },
        "environment": _runtime_environment(),
        "data_scope": {
            "kind": "deterministic_synthetic_only",
            "market_data_read": False,
            "oos_read": False,
        },
        "checks": checked,
        "metrics": deepcopy(dict(metrics)),
        "artifacts": bound_artifacts,
        "smoke_admitted": bool(passed),
        "pilot_admitted": False,
        "training_admitted": False,
        "live_trading_ready": False,
    }
    document["evidence_sha256"] = content_sha256(document)
    return document


def validate_route_smoke_evidence(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise NativeContractError("route smoke evidence must be a mapping")
    expected_fields = {
        "schema_version", "policy", "created_utc", "route_key",
        "route_profile_sha256", "catalog_sha256", "executor", "model_identity",
        "environment", "data_scope", "checks", "metrics", "artifacts",
        "smoke_admitted", "pilot_admitted", "training_admitted",
        "live_trading_ready", "evidence_sha256",
    }
    if set(value) != expected_fields:
        raise NativeContractError("route smoke evidence fields mismatch")
    document = deepcopy(dict(value))
    supplied = document.pop("evidence_sha256", None)
    if supplied != content_sha256(document):
        raise NativeContractError("route smoke evidence integrity mismatch")
    if value.get("schema_version") != SMOKE_EVIDENCE_SCHEMA or value.get("policy") != SMOKE_POLICY:
        raise NativeContractError("route smoke evidence schema or policy is invalid")
    if value.get("data_scope") != {
        "kind": "deterministic_synthetic_only", "market_data_read": False, "oos_read": False,
    }:
        raise NativeContractError("route smoke evidence data scope is invalid")
    catalog = load_family_route_catalog()
    registry = load_registry()
    route_key = value.get("route_key")
    route = catalog["routes"].get(route_key)
    if not isinstance(route, Mapping):
        raise NativeContractError("route smoke evidence route is no longer canonical")
    if value.get("catalog_sha256") != catalog_sha256(catalog):
        raise NativeContractError("route smoke evidence catalog binding is stale")
    dossier = registry["models"][str(route["arm_key"])]
    expected_model_identity = {
        "model_id": dossier["model_id"],
        "model_revision": dossier["model_revision"],
        "source_revision": dossier["source_revision"],
        "dossier_sha256": content_sha256(dossier),
    }
    if value.get("model_identity") != expected_model_identity:
        raise NativeContractError("route smoke model identity is stale or substituted")
    if value.get("route_profile_sha256") != canonical_route_profile_sha256(
        str(route["arm_key"]), str(route["track"]), str(route["route_id"]),
    ):
        raise NativeContractError("route smoke evidence profile binding is stale")
    executor = value.get("executor")
    if not isinstance(executor, Mapping) or set(executor) != {"path", "sha256", "entrypoint"}:
        raise NativeContractError("route smoke executor identity is malformed")
    executor_path = Path(str(executor["path"])).resolve()
    if not executor_path.is_file() or executor_path.is_symlink() or file_sha256(executor_path) != executor["sha256"]:
        raise NativeContractError("route smoke executor bytes changed")
    checks = _check_contract(value.get("checks"))
    passed = all(item["status"] == "pass" for item in checks.values())
    if value.get("smoke_admitted") is not passed:
        raise NativeContractError("route smoke admission flag differs from measured checks")
    if any(bool(value.get(name)) for name in (
        "pilot_admitted", "training_admitted", "live_trading_ready",
    )):
        raise NativeContractError("route smoke evidence cannot grant pilot/training/trading admission")
    artifacts = value.get("artifacts")
    required_artifacts = required_smoke_artifacts(
        dossier, catalog=catalog, route=route,
    )
    if not isinstance(artifacts, Mapping) or set(artifacts) != required_artifacts:
        raise NativeContractError("route smoke bound artifact closure is invalid")
    _validate_source_runtime(artifacts["source_runtime"], dossier)
    snapshot_path = Path(str(artifacts["model_snapshot"].get("path", "")))
    if snapshot_path.name != dossier["model_revision"]:
        raise NativeContractError("route smoke model snapshot revision is substituted")
    if "tokenizer_snapshot" in artifacts:
        tokenizer = dossier.get("tokenizer") or {}
        tokenizer_path = Path(str(artifacts["tokenizer_snapshot"].get("path", "")))
        if tokenizer_path.name != tokenizer.get("revision"):
            raise NativeContractError("route smoke tokenizer snapshot revision is substituted")
    _validate_parent_artifacts(
        artifacts, catalog=catalog, route=route,
    )
    for name, description in artifacts.items():
        if not isinstance(description, Mapping):
            raise NativeContractError(f"route smoke artifact {name!r} is malformed")
        _verify_tree(description, f"route smoke artifact {name}")
    return deepcopy(dict(value))


def load_route_smoke_evidence(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NativeContractError(f"cannot read route smoke evidence: {source}") from exc
    return validate_route_smoke_evidence(value)


__all__ = [
    "SMOKE_EVIDENCE_SCHEMA",
    "SMOKE_POLICY",
    "REQUIRED_SMOKE_ARTIFACTS",
    "required_smoke_artifacts",
    "build_route_smoke_evidence",
    "load_route_smoke_evidence",
    "validate_route_smoke_evidence",
]
