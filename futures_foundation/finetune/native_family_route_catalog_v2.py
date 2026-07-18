"""Strict loader for the blocked native family training-route catalog.

The packaged JSON is the sole source of truth for audited training semantics.  Model,
source, tokenizer, and license identity remain owned by ``native_contracts.json``;
catalog arms bind those inference dossiers by exact content hash.  This module validates
but does not construct or authorize routes.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

from .native_contracts import (
    REGISTRY_PATH,
    REGISTRY_SCHEMA,
    NativeContractError,
    content_sha256,
    load_registry,
)


SOURCE_CATALOG_PATH = (
    Path(__file__).resolve().parents[2]
    / "config"
    / "foundation_models"
    / "native_family_route_catalog_v2.json"
)
CATALOG_SCHEMA = "ffm_native_family_route_catalog_v2"
CATALOG_POLICY = "blocked_non_authorizing_audit_inventory_v1"
METHODS = frozenset({"upstream_native", "native_derived", "project_extension", "unsupported"})
PATHWAYS = frozenset({"optimizer_training", "in_context_fit", "unsupported"})
TASK_KINDS = frozenset({
    "contrastive_representation", "masked_reconstruction", "classification",
    "continuous_forecast", "quantile_forecast", "token_forecast",
    "tokenizer_reconstruction", "path_supervision", "generative_forecast",
    "support_query_forecast", "support_query_downstream", "unsupported",
})
USE_SCOPES = frozenset({"production", "research_noncommercial"})
BLOCKERS = frozenset({
    "raw_route_evidence_missing", "optimization_hyperparameters_unresolved",
    "project_objective_unresolved", "exact_resume_unverified", "export_unverified",
    "no_upstream_training_api", "nonfinite_hidden_states", "terms_unaccepted",
    "checkpoint_unavailable", "checkpoint_hash_unavailable", "model_identity_unresolved",
    "native_output_parity_missing", "family_catalog_review_pending",
    "input_contract_unresolved", "time_contract_unresolved",
    "preprocessing_contract_unresolved", "target_contract_unresolved",
    "objective_contract_unresolved", "optimization_surface_unresolved",
    "lineage_contract_unresolved", "export_contract_unresolved",
})
CONSTRAINT_FIELDS = frozenset({
    "input", "time", "preprocessing", "target", "objective",
    "optimization_surface", "optimization_hyperparameters", "lineage", "export",
})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def installed_catalog_candidates() -> tuple[Path, ...]:
    """Return catalog paths colocated with the resolved inference registry."""
    return (REGISTRY_PATH.with_name("native_family_route_catalog_v2.json"),)


def resolve_catalog_path(candidates: Iterable[str | Path] | None = None) -> Path:
    """Resolve the source or installed declarative training catalog."""
    paths = tuple(Path(value) for value in (
        candidates or (SOURCE_CATALOG_PATH, *installed_catalog_candidates())
    ))
    for candidate in paths:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "native family-route catalog not found; checked "
        + ", ".join(str(path) for path in paths)
    )


CATALOG_PATH = resolve_catalog_path()


def _exact(value: Any, fields: set[str], field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise NativeContractError(f"{field} must be an object")
    if set(value) != fields:
        raise NativeContractError(
            f"{field} fields mismatch: missing={sorted(fields-set(value))}, "
            f"unknown={sorted(set(value)-fields)}"
        )
    return value


_CONSTRAINT_ALLOWED_FIELDS = {
    "input": {"layout_tag", "context_length", "horizon_length", "parent_length",
              "channel_order", "grouping_tag", "dtype", "stamp_shape"},
    "time": {"horizon_unit", "timestamp_tag", "timezone_tag"},
    "preprocessing": {"scaler_tag", "statistics_interval_tag", "mask_tag",
                      "frequency_prefix_tag", "three_minute_prefix_id",
                      "padding_side_tag", "input_mask_tag",
                      "frequency_tokens_by_timeframe", "selector_id"},
    "target": {"target_tag", "interval"},
    "objective": {"loss_tag", "temperature", "direction_tag", "resize_length",
                  "training_projector_dim", "deployment_embedding_dim", "fusion_tag",
                  "reduction_tag", "n_channels", "patch_length", "effective_horizon",
                  "missing_label_id", "native_horizon", "prediction_filter_length",
                  "negative_eligibility_tag", "sibling_parent_policy_tag",
                  "encoder_embedding_tag", "quantile_crossing_tag",
                  "output_patch_audit_tag", "force_flip_invariance",
                  "truncate_negative", "fix_quantile_crossing"},
    "optimization_surface": {"trainable_tag", "frozen_tag", "precision",
                             "adapter_target_tag"},
    "optimization_hyperparameters": {"status"},
    "lineage": {"initialization_tag", "parent_artifacts"},
    "export": {"bundle_tag", "output_tag", "deployment_filter_tag"},
}
_CONSTRAINT_REQUIRED_FIELDS = {
    "input": {"layout_tag", "context_length", "horizon_length", "channel_order"},
    "time": {"horizon_unit", "timestamp_tag"},
    "preprocessing": {"scaler_tag", "mask_tag"},
    "target": {"target_tag", "interval"},
    "objective": {"loss_tag"},
    "optimization_surface": {"trainable_tag", "precision"},
    "optimization_hyperparameters": {"status"},
    "lineage": {"initialization_tag", "parent_artifacts"},
    "export": {"bundle_tag", "output_tag"},
}


def _validate_known_constraint(value: Any, constraint_name: str, field: str) -> None:
    if not isinstance(value, Mapping):
        raise NativeContractError(f"{field} resolved constraint must be an object")
    allowed = _CONSTRAINT_ALLOWED_FIELDS[constraint_name]
    required = _CONSTRAINT_REQUIRED_FIELDS[constraint_name]
    if set(value) - allowed or required - set(value):
        raise NativeContractError(
            f"{field} nested fields mismatch: missing={sorted(required-set(value))}, "
            f"unknown={sorted(set(value)-allowed)}"
        )
    for name, item in value.items():
        if name.endswith("_tag") or name in {
            "layout_tag", "loss_tag", "bundle_tag", "output_tag",
        }:
            if not isinstance(item, str) or not item or " " in item:
                raise NativeContractError(f"{field}.{name} must be a structured tag")


def _validate_constraint(value: Any, constraint_name: str, field: str) -> str:
    item = _exact(value, {"state", "value"}, field)
    if item["state"] not in {"resolved", "partial", "unresolved"}:
        raise NativeContractError(f"{field}.state is invalid")
    if item["state"] == "unresolved" and item["value"] is not None:
        raise NativeContractError(f"{field} unresolved value must be null")
    if item["state"] == "resolved" and item["value"] is None:
        raise NativeContractError(f"{field} resolved value cannot be null")
    if item["state"] == "resolved":
        _validate_known_constraint(item["value"], constraint_name, field)
    if item["state"] == "partial":
        partial = _exact(item["value"], {"known", "unresolved_fields"}, field)
        _validate_known_constraint(partial["known"], constraint_name, f"{field}.known")
        unresolved = partial["unresolved_fields"]
        if (
            not isinstance(unresolved, list) or not unresolved
            or not all(isinstance(name, str) and name and " " not in name for name in unresolved)
            or len(unresolved) != len(set(unresolved))
        ):
            raise NativeContractError(f"{field}.unresolved_fields is malformed")
    return str(item["state"])


def _dossier_scopes(dossier: Mapping[str, Any], arm_key: str) -> frozenset[str]:
    license_item = dossier.get("license")
    if not isinstance(license_item, Mapping):
        raise NativeContractError(f"inference dossier {arm_key} license is malformed")
    deployment = license_item.get("deployment")
    if deployment == "allowed_after_admission":
        return frozenset({"production", "research_noncommercial"})
    if deployment == "research_only_noncommercial":
        return frozenset({"research_noncommercial"})
    if deployment == "blocked_until_terms_and_version_verified":
        return frozenset()
    raise NativeContractError(
        f"inference dossier {arm_key} has unknown license deployment {deployment!r}"
    )


def _validate_catalog(value: Any, registry: Mapping[str, Any]) -> dict[str, Any]:
    catalog = _exact(value, {
        "schema_version", "policy", "non_authorizing", "inference_registry_schema",
        "arms", "constraint_profiles", "routes",
    }, "catalog")
    if catalog["schema_version"] != CATALOG_SCHEMA or catalog["policy"] != CATALOG_POLICY:
        raise NativeContractError("family-route catalog schema or policy is invalid")
    if catalog["non_authorizing"] is not True:
        raise NativeContractError("family-route catalog must remain non-authorizing")
    if catalog["inference_registry_schema"] != REGISTRY_SCHEMA:
        raise NativeContractError("family-route catalog inference registry schema is invalid")
    if registry.get("schema_version") != REGISTRY_SCHEMA:
        raise NativeContractError("inference registry schema does not match catalog")
    dossiers = registry.get("models")
    if not isinstance(dossiers, Mapping):
        raise NativeContractError("inference registry models must be an object")
    registry_tracks = registry.get("tracks")
    if not isinstance(registry_tracks, Mapping) or not registry_tracks:
        raise NativeContractError("inference registry tracks must be a nonempty object")

    arms = catalog["arms"]
    if not isinstance(arms, Mapping) or set(arms) != set(dossiers):
        raise NativeContractError(
            "family-route catalog arms must exactly match the inference registry"
        )
    arm_scopes: dict[str, frozenset[str]] = {}
    for arm_key, raw in arms.items():
        arm = _exact(raw, {"status", "dossier_ref"}, f"arms.{arm_key}")
        if arm["status"] != "blocked":
            raise NativeContractError(f"arm {arm_key} must remain blocked")
        ref = _exact(
            arm["dossier_ref"], {"registry_schema", "arm_key", "content_sha256"},
            f"arms.{arm_key}.dossier_ref",
        )
        if ref["registry_schema"] != REGISTRY_SCHEMA or ref["arm_key"] != arm_key:
            raise NativeContractError(f"arm {arm_key} dossier reference identity is invalid")
        if not isinstance(ref["content_sha256"], str) or not _SHA256.fullmatch(
            ref["content_sha256"]
        ):
            raise NativeContractError(f"arm {arm_key} dossier content hash is malformed")
        dossier = dossiers.get(arm_key)
        if not isinstance(dossier, Mapping):
            raise NativeContractError(f"arm {arm_key} inference dossier is missing")
        if content_sha256(dossier) != ref["content_sha256"]:
            raise NativeContractError(f"arm {arm_key} inference dossier hash mismatch")
        arm_scopes[arm_key] = _dossier_scopes(dossier, arm_key)

    profiles = catalog["constraint_profiles"]
    if not isinstance(profiles, Mapping) or not profiles:
        raise NativeContractError("constraint profiles must be nonempty")
    for profile_id, raw in profiles.items():
        profile = _exact(raw, set(CONSTRAINT_FIELDS), f"profiles.{profile_id}")
        for field, constraint in profile.items():
            _validate_constraint(constraint, field, f"profiles.{profile_id}.{field}")

    routes = catalog["routes"]
    if not isinstance(routes, Mapping) or not routes:
        raise NativeContractError("routes must be nonempty")
    route_fields = {
        "arm_key", "track", "route_id", "method_provenance", "pathway_kind", "status",
        "task_kind", "permitted_use_scopes", "constraint_profile", "evidence_id",
        "blocker_tags",
    }
    referenced_profiles: set[str] = set()
    represented_arms: set[str] = set()
    for key, raw in routes.items():
        route = _exact(raw, route_fields, f"routes.{key}")
        expected_key = f"{route['arm_key']}:{route['track']}:{route['route_id']}"
        if (
            key != expected_key or route["arm_key"] not in arms
            or route["track"] not in registry_tracks
        ):
            raise NativeContractError(f"route {key} identity is invalid")
        if route["method_provenance"] not in METHODS or route["pathway_kind"] not in PATHWAYS:
            raise NativeContractError(f"route {key} methodology tag is invalid")
        if route["task_kind"] not in TASK_KINDS:
            raise NativeContractError(f"route {key} task kind is invalid")
        allowed_tasks_by_track = {
            "B": {"path_supervision"},
            "C": {"classification"},
            "D": {"support_query_downstream"},
            "R": {
                "contrastive_representation", "masked_reconstruction", "unsupported",
            },
            "F": {
                "continuous_forecast", "quantile_forecast", "token_forecast",
                "tokenizer_reconstruction", "generative_forecast",
                "support_query_forecast", "unsupported",
            },
        }
        if route["task_kind"] not in allowed_tasks_by_track[route["track"]]:
            raise NativeContractError(f"route {key} task kind does not match its track")
        if (route["pathway_kind"] == "unsupported") != (
            route["task_kind"] == "unsupported"
        ):
            raise NativeContractError(f"route {key} unsupported task/pathway mismatch")
        if route["pathway_kind"] == "in_context_fit" and route["task_kind"] not in {
            "support_query_forecast", "support_query_downstream",
        }:
            raise NativeContractError(f"route {key} in-context task kind is invalid")
        if route["status"] != "blocked" or route["evidence_id"] is not None:
            raise NativeContractError(f"route {key} must remain blocked without evidence")
        scopes = route["permitted_use_scopes"]
        if (
            not isinstance(scopes, list) or not set(scopes).issubset(USE_SCOPES)
            or not set(scopes).issubset(arm_scopes[route["arm_key"]])
        ):
            raise NativeContractError(f"route {key} exceeds inference dossier license scopes")
        blockers = route["blocker_tags"]
        if not isinstance(blockers, list) or not blockers or not set(blockers).issubset(BLOCKERS):
            raise NativeContractError(f"route {key} blockers are invalid")
        profile_id = route["constraint_profile"]
        if profile_id not in profiles:
            raise NativeContractError(f"route {key} references an unknown profile")
        unresolved = {
            field for field, constraint in profiles[profile_id].items()
            if constraint["state"] != "resolved"
        }
        if unresolved and not set(blockers) & {
            "optimization_hyperparameters_unresolved", "project_objective_unresolved",
            "no_upstream_training_api", "nonfinite_hidden_states",
            "checkpoint_unavailable", "checkpoint_hash_unavailable",
            "model_identity_unresolved", "native_output_parity_missing",
        }:
            raise NativeContractError(f"route {key} hides unresolved constraints")
        if route["pathway_kind"] == "unsupported" and (
            route["method_provenance"] != "unsupported" or scopes
        ):
            raise NativeContractError(f"unsupported route {key} cannot claim use")
        referenced_profiles.add(str(profile_id))
        represented_arms.add(str(route["arm_key"]))
    if represented_arms != set(arms):
        raise NativeContractError("every inference-registry arm must have at least one route")
    if referenced_profiles != set(profiles):
        raise NativeContractError("constraint profile closure is not exact")
    return deepcopy(dict(catalog))


def _read_catalog(path: str | Path) -> Any:
    resolved = Path(path).resolve()
    try:
        return json.loads(resolved.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NativeContractError(f"family-route catalog is not valid JSON: {resolved}") from exc


_CANONICAL_CATALOG = _validate_catalog(_read_catalog(CATALOG_PATH), load_registry())
CATALOG = deepcopy(_CANONICAL_CATALOG)
EXPECTED_ARMS = frozenset(_CANONICAL_CATALOG["arms"])


def load_family_route_catalog(
    path: str | Path = CATALOG_PATH,
    *,
    registry_path: str | Path = REGISTRY_PATH,
) -> dict[str, Any]:
    """Load and validate an exact serialized copy of the canonical catalog."""
    value = _validate_catalog(_read_catalog(path), load_registry(registry_path))
    if value != _CANONICAL_CATALOG:
        raise NativeContractError("family-route catalog differs from canonical route semantics")
    return value


def validate_family_route_catalog(
    value: Any | None = None,
    *,
    registry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a catalog value and reject any semantic mutation."""
    candidate = _CANONICAL_CATALOG if value is None else value
    validated = _validate_catalog(candidate, registry or load_registry())
    if validated != _CANONICAL_CATALOG:
        raise NativeContractError("family-route catalog differs from canonical route semantics")
    return validated


def catalog_sha256(value: Any | None = None) -> str:
    """Return the stable hash of a validated catalog."""
    return content_sha256(validate_family_route_catalog(value))
