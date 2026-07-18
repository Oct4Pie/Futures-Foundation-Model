"""Fail-closed compatibility view over the native family-route catalog.

``native_family_route_catalog_v2`` is the sole source of truth for training-route
inventory and methodology.  This module intentionally contains no admission parser,
evidence loader, or file-backed route registry.  It exists only for callers that have
not yet migrated from the v1 API.

The catalog is an audited, non-authorizing inventory.  Consequently every optimizer
and evidence/authorization request is rejected before any model, data, or caller-
supplied filesystem path can be touched.
"""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

from .native_contracts import REGISTRY_PATH, NativeContractError
from .native_family_route_catalog_v2 import (
    CATALOG_PATH,
    CATALOG_POLICY,
    CATALOG_SCHEMA,
    catalog_sha256,
    load_family_route_catalog,
)


# Compatibility constants only.  They do not describe an executable admission format.
TRAINING_ROUTE_SCHEMA = CATALOG_SCHEMA
TRAINING_REPORT_SCHEMA = "ffm_native_training_admission_report_retired_v2"
ADMISSION_POLICY = CATALOG_POLICY

_RETIRED_MESSAGE = (
    "training admission is disabled: the v1 route/evidence authority was retired; "
    "the v2 family-route catalog is a non-authorizing inventory and no route has a "
    "verified route instance, raw smoke evidence, or runtime approval"
)


def block_unadmitted_optimizer(entrypoint: str) -> None:
    """Reject an optimizer-capable entrypoint before it can perform any work."""
    raise NativeContractError(
        f"optimizer entrypoint {entrypoint!r} is disabled: {_RETIRED_MESSAGE}"
    )


def route_key(arm_key: str, track: str, route_id: str) -> str:
    """Return the catalog's canonical route identifier."""
    return f"{arm_key}:{track}:{route_id}"


def route_registry_path(path: str | Path = REGISTRY_PATH) -> Path:
    """Return the catalog source location for legacy display-only callers.

    ``path`` is deliberately ignored.  A caller-provided directory must never select
    an alternative training authority.
    """
    del path
    return Path(CATALOG_PATH).resolve()


def route_evidence_path(path: str | Path = REGISTRY_PATH) -> Path:
    """Reject access to the retired file-backed evidence registry."""
    del path
    raise NativeContractError(_RETIRED_MESSAGE)


@lru_cache(maxsize=1)
def _catalog_snapshot() -> dict[str, Any]:
    return load_family_route_catalog(CATALOG_PATH, registry_path=REGISTRY_PATH)


def load_route_registry(path: str | Path = REGISTRY_PATH) -> dict[str, Any]:
    """Return a defensive copy of the canonical non-authorizing catalog."""
    del path
    return deepcopy(_catalog_snapshot())


def load_route_evidence(path: str | Path = REGISTRY_PATH) -> dict[str, Any]:
    """Reject access; the retired facade has no evidence authority."""
    del path
    raise NativeContractError(_RETIRED_MESSAGE)


# Preserve the display-cache hook used by diagnostic tooling.
load_route_registry.cache_clear = _catalog_snapshot.cache_clear  # type: ignore[attr-defined]


def registry_sha256(path: str | Path = REGISTRY_PATH) -> str:
    del path
    return catalog_sha256()


def evidence_sha256(path: str | Path = REGISTRY_PATH) -> str:
    del path
    raise NativeContractError(_RETIRED_MESSAGE)


def get_route(
    arm_key: str,
    track: str,
    route_id: str,
    path: str | Path = REGISTRY_PATH,
) -> dict[str, Any]:
    """Resolve one display-only catalog route; this grants no authority."""
    key = route_key(arm_key, track, route_id)
    route = load_route_registry(path)["routes"].get(key)
    if not isinstance(route, dict):
        raise NativeContractError(f"undeclared training route: {key}")
    return deepcopy(route)


def evidence_for_route(
    arm_key: str,
    track: str,
    route_id: str,
    path: str | Path = REGISTRY_PATH,
) -> tuple[str, dict[str, Any]]:
    """Always reject before consulting the catalog, path, model, or data."""
    del arm_key, track, route_id, path
    raise NativeContractError(_RETIRED_MESSAGE)


def admitted_routes_for_arm(
    arm_key: str,
    path: str | Path = REGISTRY_PATH,
    *,
    include_research: bool = False,
) -> tuple[dict[str, Any], ...]:
    """Return no executable routes while the catalog is non-authorizing."""
    del arm_key, path, include_research
    return ()


def authorize_route(
    *,
    arm_key: str,
    track: str,
    route_id: str | None,
    use_scope: str | None,
    path: str | Path = REGISTRY_PATH,
) -> tuple[dict[str, Any], str, dict[str, Any], str, str]:
    """Always reject before filesystem, registry, model, or data access."""
    del arm_key, track, route_id, use_scope, path
    raise NativeContractError(_RETIRED_MESSAGE)
