"""Route-scoped, fail-closed training admission layered over native inference contracts.

Inference parity and training admission are deliberately separate authorities.  A native
forecast or representation track cannot authorize an optimizer, objective, channel mixer,
or custom task.  Every training decision is keyed by ``arm:track:route`` and binds the
complete route record plus mandatory route-specific evidence.
"""
from __future__ import annotations

from functools import lru_cache
import json
import math
import os
from pathlib import Path
import stat
from typing import Any, Mapping

from .native_contracts import (
    REGISTRY_PATH,
    NativeContractError,
    _parse_utc,
    _require_string,
    content_sha256,
    load_registry,
)


TRAINING_ROUTE_SCHEMA = "ffm_native_training_route_registry_v1"
TRAINING_EVIDENCE_SCHEMA = "ffm_native_training_route_evidence_v1"
TRAINING_REPORT_SCHEMA = "ffm_native_training_admission_report_v1"
TRAINING_ROUTE_STATUSES = frozenset({"blocked", "admitted", "research_only"})
TRAINING_ROUTE_DISPOSITIONS = frozenset({
    "upstream_native", "native_derived", "custom_research", "unsupported",
})
TRAINING_USE_SCOPES = frozenset({"production", "research_noncommercial"})
TRAINING_EVIDENCE_STATUSES = frozenset({"pass", "research_only_pass"})
TRAINING_CHECK_STATUSES = frozenset({"pass", "fail", "not_run"})
ADMISSION_POLICY = "phase_a_blocked_pending_raw_bundle_v1"
FROZEN_ROUTE_MATRIX_SHA256 = (
    "c9c36de2a63cdfe349046be6754b8b5d530a47f6d0e1fe3e18a3ed904e909a0f"
)
MAX_CONTRACT_BYTES = 2 * 1024 * 1024
MAX_JSON_NODES = 100_000
MAX_JSON_DEPTH = 32
TRAINING_CHECKS = frozenset({
    "upstream_wrapper_input_target_loss_parity",
    "causal_prefix_invariance",
    "channel_group_semantics",
    "context_horizon_boundaries",
    "scaling_mask_behavior",
    "batch_partition_parity",
    "gradient_freeze_surface",
    "repeated_batch_loss_decrease",
    "shuffled_label_control",
    "corrupted_target_control",
    "constant_input_control",
    "exact_full_trajectory_sampler_resume",
    "save_reload_state",
    "deployment_preprocessing_parity",
    "deployment_output_parity",
    "fp32_finite",
})


def block_unadmitted_optimizer(entrypoint: str) -> None:
    """Phase-A kill switch for optimizer-capable paths lacking route-bundle admission."""
    raise NativeContractError(
        f"optimizer entrypoint {entrypoint!r} is disabled: no route has raw training-evidence "
        "bundle admission; inference parity or a legacy checkpoint cannot authorize training"
    )


def route_registry_path(path: str | Path = REGISTRY_PATH) -> Path:
    return Path(path).resolve().with_name("native_training_routes.json")


def route_evidence_path(path: str | Path = REGISTRY_PATH) -> Path:
    return Path(path).resolve().with_name("native_training_route_evidence.json")


def route_key(arm_key: str, track: str, route_id: str) -> str:
    return f"{arm_key}:{track}:{route_id}"


def _identifier(value: Any, field: str) -> str:
    text = _require_string(value, field)
    if (
        text != text.strip().lower()
        or not text[0].isalpha()
        or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in text)
    ):
        raise NativeContractError(f"{field} must be lowercase snake-case")
    return text


def _validate_source(value: Any, *, admitted: bool) -> None:
    if not isinstance(value, Mapping) or set(value) != {"kind", "revision"}:
        raise NativeContractError(
            "training_methodology_source must define exactly kind and revision"
        )
    if value.get("kind") == "pending_git_commit":
        if value.get("revision") is not None:
            raise NativeContractError("pending training methodology cannot claim a revision")
        if admitted:
            raise NativeContractError("an admitted route requires a frozen methodology commit")
        return
    revision = value.get("revision")
    if value.get("kind") != "git_commit" or (
        not isinstance(revision, str)
        or len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
    ):
        raise NativeContractError("training methodology must be a full Git SHA-1")


def _validate_registry(value: Mapping[str, Any], inference: Mapping[str, Any]) -> None:
    expected = {
        "schema_version", "evidence_schema", "inference_methodology_commit",
        "training_methodology_source", "status_vocabulary", "disposition_vocabulary",
        "use_scope_vocabulary", "required_checks", "contract_profiles", "routes",
        "admission_policy",
    }
    if set(value) != expected:
        raise NativeContractError(
            "training-route registry fields mismatch: "
            f"missing={sorted(expected - set(value))}, unknown={sorted(set(value) - expected)}"
        )
    if value.get("schema_version") != TRAINING_ROUTE_SCHEMA:
        raise NativeContractError(f"training-route schema must be {TRAINING_ROUTE_SCHEMA!r}")
    if value.get("evidence_schema") != TRAINING_EVIDENCE_SCHEMA:
        raise NativeContractError(
            f"training-route evidence schema must be {TRAINING_EVIDENCE_SCHEMA!r}"
        )
    if value.get("inference_methodology_commit") != inference.get("methodology_commit"):
        raise NativeContractError("training routes reference stale inference methodology")
    if set(value.get("status_vocabulary") or ()) != set(TRAINING_ROUTE_STATUSES):
        raise NativeContractError("training-route status vocabulary drifted")
    if set(value.get("disposition_vocabulary") or ()) != set(
        TRAINING_ROUTE_DISPOSITIONS
    ):
        raise NativeContractError("training-route disposition vocabulary drifted")
    if set(value.get("use_scope_vocabulary") or ()) != set(TRAINING_USE_SCOPES):
        raise NativeContractError("training-route use-scope vocabulary drifted")
    checks = value.get("required_checks")
    if (
        not isinstance(checks, list)
        or len(checks) != len(set(checks))
        or set(checks) != set(TRAINING_CHECKS)
    ):
        raise NativeContractError("training-route mandatory checks drifted")

    routes = value.get("routes")
    if not isinstance(routes, Mapping) or not routes:
        raise NativeContractError("training routes must be a nonempty object")
    admitted = any(
        isinstance(route, Mapping) and route.get("status") in {"admitted", "research_only"}
        for route in routes.values()
    )
    if value.get("admission_policy") != ADMISSION_POLICY:
        raise NativeContractError("training-route admission policy drifted")
    if admitted:
        raise NativeContractError(
            "Phase A forbids nonblocked training routes until raw route-bundle "
            "verification is implemented"
        )
    _validate_source(value.get("training_methodology_source"), admitted=admitted)

    profiles = value.get("contract_profiles")
    if not isinstance(profiles, Mapping) or not profiles:
        raise NativeContractError("training contract profiles must be nonempty")
    profile_fields = {
        "objective", "input_layout", "channel_semantics", "scaling_masking",
        "trainable_surface", "deployment_output",
    }
    for profile_id, profile in profiles.items():
        _identifier(profile_id, "training contract profile id")
        if not isinstance(profile, Mapping) or set(profile) != profile_fields:
            raise NativeContractError(f"training profile {profile_id!r} fields mismatch")
        for field in profile_fields:
            _require_string(profile.get(field), f"training profile {profile_id}.{field}")

    route_fields = {
        "arm_key", "track", "route_id", "disposition", "status",
        "allowed_use_scopes", "contract_profile", "base_inference_track",
        "evidence_id", "reason",
    }
    referenced_profiles: set[str] = set()
    for key, route in routes.items():
        _require_string(key, "training route key")
        if not isinstance(route, Mapping) or set(route) != route_fields:
            raise NativeContractError(f"training route {key!r} fields mismatch")
        arm_key = _identifier(route.get("arm_key"), f"{key}.arm_key")
        track = _require_string(route.get("track"), f"{key}.track")
        route_id = _identifier(route.get("route_id"), f"{key}.route_id")
        if key != route_key(arm_key, track, route_id):
            raise NativeContractError(f"training route key {key!r} is not canonical")
        if arm_key not in inference["models"] or track not in inference["tracks"]:
            raise NativeContractError(f"training route {key!r} has unknown arm/track")
        if route.get("base_inference_track") not in inference["tracks"]:
            raise NativeContractError(f"training route {key!r} has unknown base track")
        disposition = route.get("disposition")
        status = route.get("status")
        if disposition not in TRAINING_ROUTE_DISPOSITIONS:
            raise NativeContractError(f"training route {key!r} has invalid disposition")
        if status not in TRAINING_ROUTE_STATUSES:
            raise NativeContractError(f"training route {key!r} has invalid status")
        scopes = route.get("allowed_use_scopes")
        if (
            not isinstance(scopes, list)
            or len(scopes) != len(set(scopes))
            or not set(scopes).issubset(TRAINING_USE_SCOPES)
        ):
            raise NativeContractError(f"training route {key!r} has invalid use scopes")
        if disposition == "unsupported" and (status != "blocked" or scopes):
            raise NativeContractError(
                f"unsupported route {key!r} must remain blocked with no use scope"
            )
        if disposition == "custom_research" and (
            status == "admitted" or set(scopes) != {"research_noncommercial"}
        ):
            raise NativeContractError(
                f"custom research route {key!r} cannot authorize production"
            )
        if status == "research_only" and set(scopes) != {"research_noncommercial"}:
            raise NativeContractError(
                f"research-only route {key!r} must be noncommercial-only"
            )
        if status == "admitted" and "production" not in scopes:
            raise NativeContractError(
                f"production-admitted route {key!r} must explicitly allow production"
            )
        if status in {"admitted", "research_only"}:
            base_capability = inference["models"][arm_key]["tracks"][
                route["base_inference_track"]
            ]
            admitted_base_statuses = {
                "native_valid", "native_valid_experimental_task", "research_only",
            }
            if base_capability.get("status") not in admitted_base_statuses:
                raise NativeContractError(
                    f"training route {key!r} has no admitted base inference track"
                )
            if status == "admitted" and base_capability.get("status") == "research_only":
                raise NativeContractError(
                    f"production route {key!r} cannot inherit a research-only base"
                )
        if status == "blocked":
            if route.get("evidence_id") is not None:
                raise NativeContractError(
                    f"blocked route {key!r} cannot claim admission evidence"
                )
        else:
            _require_string(route.get("evidence_id"), f"{key}.evidence_id")
        profile_id = _identifier(route.get("contract_profile"), f"{key}.contract_profile")
        if profile_id not in profiles:
            raise NativeContractError(f"training route {key!r} has unknown profile")
        referenced_profiles.add(profile_id)
        _require_string(route.get("reason"), f"{key}.reason")
    if referenced_profiles != set(profiles):
        raise NativeContractError(
            "training contract profiles must all be referenced: "
            f"unreferenced={sorted(set(profiles) - referenced_profiles)}"
        )
    matrix_fields = (
        "arm_key", "track", "route_id", "disposition", "contract_profile",
        "base_inference_track",
    )
    matrix = {
        "contract_profiles": profiles,
        "routes": {
            key: {field: route[field] for field in matrix_fields}
            for key, route in routes.items()
        },
    }
    if content_sha256(matrix) != FROZEN_ROUTE_MATRIX_SHA256:
        raise NativeContractError("training route/profile/base matrix drifted from code")


def _validate_checks(value: Any, field: str) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping) or set(value) != set(TRAINING_CHECKS):
        raise NativeContractError(f"{field} must define every training check exactly once")
    output: dict[str, dict[str, Any]] = {}
    for name, raw in value.items():
        if not isinstance(raw, Mapping) or set(raw) - {"status", "evidence", "metrics"}:
            raise NativeContractError(f"{field}.{name} is malformed")
        item = dict(raw)
        if item.get("status") not in TRAINING_CHECK_STATUSES:
            raise NativeContractError(f"{field}.{name} has invalid status")
        if item["status"] in {"pass", "fail"} and not (
            str(item.get("evidence", "")).strip() or item.get("metrics")
        ):
            raise NativeContractError(f"{field}.{name} needs concrete evidence")
        output[name] = item
    return output


def _validate_evidence(value: Mapping[str, Any], registry: Mapping[str, Any]) -> None:
    expected = {
        "schema_version", "inference_methodology_commit", "training_methodology_source",
        "route_registry_schema", "route_registry_sha256", "generated_utc", "records",
        "admission_policy",
    }
    if set(value) != expected:
        raise NativeContractError(
            "training evidence fields mismatch: "
            f"missing={sorted(expected - set(value))}, unknown={sorted(set(value) - expected)}"
        )
    if value.get("schema_version") != TRAINING_EVIDENCE_SCHEMA:
        raise NativeContractError(f"training evidence schema must be {TRAINING_EVIDENCE_SCHEMA!r}")
    if value.get("route_registry_schema") != TRAINING_ROUTE_SCHEMA:
        raise NativeContractError("training evidence route schema drifted")
    if value.get("route_registry_sha256") != content_sha256(registry):
        raise NativeContractError("training evidence is not hash-bound to its route registry")
    if value.get("admission_policy") != registry.get("admission_policy"):
        raise NativeContractError("training evidence admission policy drifted")
    for field in ("inference_methodology_commit", "training_methodology_source"):
        if value.get(field) != registry.get(field):
            raise NativeContractError(f"training evidence {field} drifted")
    _parse_utc(value.get("generated_utc"), "training evidence generated_utc")
    records = value.get("records")
    if not isinstance(records, Mapping):
        raise NativeContractError("training evidence records must be an object")

    fields = {
        "arm_key", "track", "route_id", "route_key", "route_sha256", "status",
        "checks", "environment", "artifacts", "reason",
    }
    claimed: set[str] = set()
    for evidence_id, record in records.items():
        _require_string(evidence_id, "training evidence id")
        if not isinstance(record, Mapping) or set(record) != fields:
            raise NativeContractError(f"training evidence {evidence_id!r} fields mismatch")
        key = route_key(
            str(record.get("arm_key")), str(record.get("track")), str(record.get("route_id"))
        )
        route = registry["routes"].get(key)
        if not isinstance(route, Mapping) or record.get("route_key") != key:
            raise NativeContractError(f"training evidence {evidence_id!r} route mismatch")
        if (
            record.get("arm_key") != route["arm_key"]
            or record.get("track") != route["track"]
            or record.get("route_id") != route["route_id"]
            or record.get("route_sha256") != content_sha256(route)
        ):
            raise NativeContractError(
                f"training evidence {evidence_id!r} is stale or profile-swapped"
            )
        status = record.get("status")
        if status not in TRAINING_EVIDENCE_STATUSES:
            raise NativeContractError(f"training evidence {evidence_id!r} status is invalid")
        checks = _validate_checks(record.get("checks"), f"training evidence {evidence_id}.checks")
        nonpassing = [name for name, item in checks.items() if item["status"] != "pass"]
        if nonpassing:
            raise NativeContractError(
                f"training evidence {evidence_id!r} lacks mandatory passes: {nonpassing}"
            )
        if not isinstance(record.get("environment"), Mapping) or not record["environment"]:
            raise NativeContractError(f"training evidence {evidence_id!r} needs environment")
        if not isinstance(record.get("artifacts"), Mapping) or not record["artifacts"]:
            raise NativeContractError(f"training evidence {evidence_id!r} needs artifacts")
        _require_string(record.get("reason"), f"training evidence {evidence_id}.reason")
        claimed.add(evidence_id)

    referenced: set[str] = set()
    for key, route in registry["routes"].items():
        if route["status"] == "blocked":
            continue
        evidence_id = route["evidence_id"]
        referenced.add(evidence_id)
        record = records.get(evidence_id)
        if not isinstance(record, Mapping):
            raise NativeContractError(f"admitted route {key!r} has no training evidence")
        expected_status = "research_only_pass" if route["status"] == "research_only" else "pass"
        if record.get("status") != expected_status:
            raise NativeContractError(
                f"training route {key!r} requires evidence status {expected_status!r}"
            )
    if claimed != referenced:
        raise NativeContractError(
            "training evidence references must be exact: "
            f"orphaned={sorted(claimed - referenced)}, missing={sorted(referenced - claimed)}"
        )


@lru_cache(maxsize=8)
def _load_display_state(
    path: str | Path = REGISTRY_PATH,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Cache one jointly validated snapshot for non-authorizing display/reporting only."""
    return _load_current_authorization_state(path)


def load_route_registry(path: str | Path = REGISTRY_PATH) -> dict[str, Any]:
    return _load_display_state(path)[0]


def load_route_evidence(path: str | Path = REGISTRY_PATH) -> dict[str, Any]:
    return _load_display_state(path)[1]


# Preserve the cache-clear surface used by existing tooling while preventing split caches.
load_route_registry.cache_clear = _load_display_state.cache_clear  # type: ignore[attr-defined]
load_route_evidence.cache_clear = _load_display_state.cache_clear  # type: ignore[attr-defined]


def registry_sha256(path: str | Path = REGISTRY_PATH) -> str:
    return content_sha256(load_route_registry(path))


def evidence_sha256(path: str | Path = REGISTRY_PATH) -> str:
    return content_sha256(load_route_evidence(path))


def _open_absolute_directory(path: Path) -> tuple[int, tuple[int, int]]:
    """Traverse an absolute directory one component at a time without following links."""
    absolute = Path(os.path.abspath(path))
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open("/", flags)
    try:
        for component in absolute.parts[1:]:
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
            current = os.fstat(descriptor)
            if not stat.S_ISDIR(current.st_mode):
                raise NativeContractError(
                    f"training contract path component {component!r} is not a directory"
                )
        current = os.fstat(descriptor)
        return descriptor, (current.st_dev, current.st_ino)
    except Exception:
        os.close(descriptor)
        raise


def _strict_json_object(payload: bytes, name: str) -> dict[str, Any]:
    if len(payload) > MAX_CONTRACT_BYTES:
        raise NativeContractError(
            f"training contract {name!r} exceeds {MAX_CONTRACT_BYTES} bytes"
        )

    def reject_duplicates(pairs):
        output = {}
        for key, value in pairs:
            if key in output:
                raise NativeContractError(
                    f"training contract {name!r} contains duplicate key {key!r}"
                )
            output[key] = value
        return output

    def reject_constant(value):
        raise NativeContractError(
            f"training contract {name!r} contains non-finite number {value!r}"
        )

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except NativeContractError:
        raise
    except (UnicodeDecodeError, ValueError, OverflowError, RecursionError) as exc:
        raise NativeContractError(
            f"training contract {name!r} is not strict UTF-8 JSON"
        ) from exc
    if not isinstance(value, dict):
        raise NativeContractError(f"training contract {name!r} must contain a JSON object")

    nodes = 0
    stack = [(value, 1)]
    while stack:
        item, depth = stack.pop()
        nodes += 1
        if nodes > MAX_JSON_NODES:
            raise NativeContractError(
                f"training contract {name!r} exceeds {MAX_JSON_NODES} JSON nodes"
            )
        if depth > MAX_JSON_DEPTH:
            raise NativeContractError(
                f"training contract {name!r} exceeds JSON depth {MAX_JSON_DEPTH}"
            )
        if isinstance(item, dict):
            stack.extend((key, depth + 1) for key in item)
            stack.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, list):
            stack.extend((child, depth + 1) for child in item)
        elif isinstance(item, float) and not math.isfinite(item):
            raise NativeContractError(
                f"training contract {name!r} contains a non-finite float"
            )
    return value


def _secure_read_at(directory_fd: int, name: str) -> dict[str, Any]:
    """Read one regular, non-symlink file and detect mutation during the read."""
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        namespace_before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except OSError as exc:
        raise NativeContractError(f"cannot inspect training contract {name!r}: {exc}") from exc
    if (
        not stat.S_ISREG(namespace_before.st_mode)
        or stat.S_ISLNK(namespace_before.st_mode)
        or namespace_before.st_nlink != 1
    ):
        raise NativeContractError(
            f"training contract {name!r} must be a single-link regular file"
        )
    if namespace_before.st_size > MAX_CONTRACT_BYTES:
        raise NativeContractError(
            f"training contract {name!r} exceeds {MAX_CONTRACT_BYTES} bytes"
        )
    try:
        file_fd = os.open(name, flags, dir_fd=directory_fd)
    except OSError as exc:
        raise NativeContractError(f"cannot securely open training contract {name!r}: {exc}") from exc
    try:
        before = os.fstat(file_fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or (before.st_dev, before.st_ino)
            != (namespace_before.st_dev, namespace_before.st_ino)
        ):
            raise NativeContractError(
                f"training contract {name!r} identity changed before open"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(file_fd, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(file_fd)
        identity_before = (
            before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns,
        )
        payload = b"".join(chunks)
        if identity_before != identity_after or len(payload) != before.st_size:
            raise NativeContractError(f"training contract {name!r} changed during authorization")
        namespace_after = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if (
            namespace_after.st_nlink != 1
            or (namespace_after.st_dev, namespace_after.st_ino, namespace_after.st_size,
                namespace_after.st_mtime_ns, namespace_after.st_ctime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        ):
            raise NativeContractError(
                f"training contract {name!r} namespace changed during authorization"
            )
    finally:
        os.close(file_fd)
    return _strict_json_object(payload, name)


def _load_current_authorization_state(
    path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Uncached, same-directory route/evidence snapshot for an authorization decision."""
    inference = load_registry(path)
    parent = Path(path).expanduser().absolute().parent
    try:
        directory_fd, opened_identity = _open_absolute_directory(parent)
    except OSError as exc:
        raise NativeContractError(f"cannot securely open training contract directory: {exc}") from exc
    try:
        registry = _secure_read_at(directory_fd, "native_training_routes.json")
        evidence = _secure_read_at(directory_fd, "native_training_route_evidence.json")
        after_directory = os.fstat(directory_fd)
        reopened_fd, reopened_identity = _open_absolute_directory(parent)
        os.close(reopened_fd)
        if (
            opened_identity != (after_directory.st_dev, after_directory.st_ino)
            or opened_identity != reopened_identity
        ):
            raise NativeContractError("training contract directory changed during authorization")
    except OSError as exc:
        raise NativeContractError(f"training contract directory changed: {exc}") from exc
    finally:
        os.close(directory_fd)
    _validate_registry(registry, inference)
    _validate_evidence(evidence, registry)
    return registry, evidence


def get_route(
    arm_key: str, track: str, route_id: str, path: str | Path = REGISTRY_PATH
) -> dict[str, Any]:
    key = route_key(arm_key, track, route_id)
    route = load_route_registry(path)["routes"].get(key)
    if not isinstance(route, Mapping):
        raise NativeContractError(f"undeclared training route: {key}")
    return dict(route)


def evidence_for_route(
    arm_key: str, track: str, route_id: str, path: str | Path = REGISTRY_PATH
) -> tuple[str, dict[str, Any]]:
    route = get_route(arm_key, track, route_id, path)
    if route["status"] == "blocked":
        raise NativeContractError(
            f"training route {route_key(arm_key, track, route_id)} is blocked: {route['reason']}"
        )
    evidence_id = route["evidence_id"]
    record = load_route_evidence(path)["records"].get(evidence_id)
    if not isinstance(record, Mapping):
        raise NativeContractError(f"training evidence not found: {evidence_id}")
    return str(evidence_id), dict(record)


def admitted_routes_for_arm(
    arm_key: str, path: str | Path = REGISTRY_PATH, *, include_research: bool = False
) -> tuple[dict[str, Any], ...]:
    statuses = {"admitted", "research_only"} if include_research else {"admitted"}
    return tuple(
        dict(route)
        for route in load_route_registry(path)["routes"].values()
        if route["arm_key"] == arm_key and route["status"] in statuses
    )


def authorize_route(
    *, arm_key: str, track: str, route_id: str | None, use_scope: str | None,
    path: str | Path = REGISTRY_PATH,
) -> tuple[dict[str, Any], str, dict[str, Any], str, str]:
    """Authorize only a concrete, independently evidenced training route."""
    if route_id is None:
        raise NativeContractError("training admission requires a non-null route")
    registry, evidence_registry = _load_current_authorization_state(path)
    key = route_key(arm_key, track, route_id)
    route = registry["routes"].get(key)
    if not isinstance(route, Mapping):
        raise NativeContractError(f"undeclared training route: {key}")
    route = dict(route)
    if route["status"] == "blocked":
        raise NativeContractError(
            f"training route {route_key(arm_key, track, route_id)} is blocked: {route['reason']}"
        )
    if use_scope not in TRAINING_USE_SCOPES:
        raise NativeContractError("training admission requires an explicit valid use_scope")
    if use_scope not in route["allowed_use_scopes"]:
        raise NativeContractError(
            f"training route {route_key(arm_key, track, route_id)} does not allow {use_scope!r}"
        )
    if route["status"] == "research_only" and use_scope != "research_noncommercial":
        raise NativeContractError("research-only training cannot authorize production")
    evidence_id = str(route["evidence_id"])
    evidence = evidence_registry["records"].get(evidence_id)
    if not isinstance(evidence, Mapping):
        raise NativeContractError(f"training evidence not found: {evidence_id}")
    return (
        route,
        evidence_id,
        dict(evidence),
        content_sha256(registry),
        content_sha256(evidence_registry),
    )
