"""Executable native-contract registry and fail-closed admission gates.

The JSON registry is the single source of truth for cross-family model identity,
capabilities, historical disposition, and admission status.  A model is not safe to
benchmark or train merely because its package imports or a historical checkpoint exists.
An immutable admission report must bind the current registry/dossier hashes, the requested
track, all applicable parity checks, and two independent approvals.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import lru_cache
import base64
import hashlib
import json
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path
import platform
import sys
import sysconfig
from typing import Any, Iterable, Mapping


SOURCE_REGISTRY_PATH = (
    Path(__file__).resolve().parents[2]
    / "config"
    / "foundation_models"
    / "native_contracts.json"
)
SOURCE_EVIDENCE_PATH = SOURCE_REGISTRY_PATH.with_name("native_contract_evidence.json")
SOURCE_TRUST_PATH = SOURCE_REGISTRY_PATH.with_name("trusted_approvers.json")


def evidence_path_for_registry(path: str | Path) -> Path:
    """Return the evidence document colocated with a source or installed registry."""
    return Path(path).resolve().with_name("native_contract_evidence.json")


def trust_path_for_registry(path: str | Path) -> Path:
    """Return the trusted-reviewer key registry colocated with a model registry."""
    return Path(path).resolve().with_name("trusted_approvers.json")


def installed_registry_candidates() -> tuple[Path, ...]:
    """Return installation-scheme-aware registry locations, ordered by authority."""
    candidates: list[Path] = []
    try:
        package = distribution("futures-foundation-model")
        for item in package.files or ():
            if str(item).replace("\\", "/").endswith(
                "config/foundation_models/native_contracts.json"
            ):
                candidates.append(Path(package.locate_file(item)).resolve())
    except PackageNotFoundError:
        pass
    candidates.extend((
        Path(sysconfig.get_path("data"))
        / "config"
        / "foundation_models"
        / "native_contracts.json",
        Path(sys.prefix) / "config" / "foundation_models" / "native_contracts.json",
    ))
    return tuple(dict.fromkeys(candidates))


def resolve_registry_path(candidates: Iterable[str | Path] | None = None) -> Path:
    """Resolve the source-checkout registry or a wheel-installed data-file fallback."""
    paths = tuple(Path(value) for value in (
        candidates or (SOURCE_REGISTRY_PATH, *installed_registry_candidates())
    ))
    for candidate in paths:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "native-contract registry not found; checked "
        + ", ".join(str(path) for path in paths)
    )


REGISTRY_PATH = resolve_registry_path()
REPORT_SCHEMA = "ffm_native_admission_report_v3"
REGISTRY_SCHEMA = "ffm_native_contract_registry_v1"
EVIDENCE_SCHEMA = "ffm_native_contract_evidence_v1"
TRUST_SCHEMA = "ffm_trusted_approvers_v1"
VALID_STATUSES = frozenset({
    "native_valid",
    "native_valid_experimental_task",
    "configuration_specific",
    "invalid_contract",
    "blocked",
    "research_only",
})
ADMITTED_STATUSES = frozenset({
    "native_valid",
    "native_valid_experimental_task",
    "research_only",
})
VALID_CHECK_STATUSES = frozenset({"pass", "fail", "not_applicable"})
TRAINING_CHECKS = frozenset({
    "gradient_freeze_surface",
    "repeated_batch_loss_decrease",
    "exact_resume",
    "save_reload_export",
})
MANDATORY_TECHNICAL_CHECKS = frozenset({
    "official_example",
    "adapter_public_api_parity",
    "fp32_finite",
    "license_governance",
})
VALID_EVIDENCE_STATUSES = frozenset({"pass", "research_only_pass", "blocked"})
RUNTIME_CONTROL_FIELDS = frozenset({
    "device", "dtype", "network_policy", "profile", "use_scope",
})


class NativeContractError(ValueError):
    """Raised when a model identity or admission contract is unsafe."""


def canonical_json(value: Any) -> bytes:
    """Return the stable UTF-8 representation used by every contract hash."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def content_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_trusted_approvers(path: str | Path = REGISTRY_PATH) -> dict[str, Any]:
    """Load the explicit Ed25519 trust root used for operational approvals.

    The trust store is deliberately separate from an admission report.  A report cannot
    introduce its own reviewer key, and an absent or empty store always fails closed.
    """
    trust_path = trust_path_for_registry(path)
    if not trust_path.is_file():
        raise NativeContractError(f"trusted-approver registry not found: {trust_path}")
    try:
        value = json.loads(trust_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NativeContractError(
            f"trusted-approver registry is not valid JSON: {trust_path}"
        ) from exc
    if not isinstance(value, dict) or value.get("schema_version") != TRUST_SCHEMA:
        raise NativeContractError(
            f"trusted-approver registry schema must be {TRUST_SCHEMA!r}"
        )
    keys = value.get("keys")
    if not isinstance(keys, dict):
        raise NativeContractError("trusted-approver registry keys must be an object")
    normalized_reviewers: set[str] = set()
    public_key_fingerprints: set[str] = set()
    for key_id, raw in keys.items():
        _require_string(key_id, "trusted key id")
        if not isinstance(raw, Mapping):
            raise NativeContractError(f"trusted key {key_id!r} must be an object")
        reviewer = _require_string(raw.get("reviewer"), f"trusted key {key_id}.reviewer")
        if raw.get("algorithm") != "ed25519":
            raise NativeContractError(f"trusted key {key_id!r} must use ed25519")
        public_key_pem = _require_string(
            raw.get("public_key_pem"), f"trusted key {key_id}.public_key_pem"
        )
        try:
            from cryptography.hazmat.primitives import serialization
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
            public_key = serialization.load_pem_public_key(public_key_pem.encode("ascii"))
        except (ImportError, ValueError, TypeError, UnicodeEncodeError) as exc:
            raise NativeContractError(f"trusted key {key_id!r} is not valid Ed25519 PEM") from exc
        if not isinstance(public_key, Ed25519PublicKey):
            raise NativeContractError(f"trusted key {key_id!r} is not Ed25519")
        fingerprint = hashlib.sha256(public_key.public_bytes(
            serialization.Encoding.DER,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )).hexdigest()
        if fingerprint in public_key_fingerprints:
            raise NativeContractError(
                "trusted-approver registry cannot alias one public key to multiple reviewers"
            )
        public_key_fingerprints.add(fingerprint)
        normalized = reviewer.strip().casefold()
        if normalized in normalized_reviewers:
            raise NativeContractError(
                "trusted-approver registry must contain at most one active key per reviewer"
            )
        normalized_reviewers.add(normalized)
    return value


def trusted_approvers_sha256(path: str | Path = REGISTRY_PATH) -> str:
    return content_sha256(load_trusted_approvers(path))


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise NativeContractError(f"{field} must be a nonempty string")
    return value


def _parse_utc(value: Any, field: str) -> datetime:
    """Require an explicit, timezone-aware UTC timestamp."""
    text = _require_string(value, field)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise NativeContractError(f"{field} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise NativeContractError(f"{field} must be timezone-aware UTC")
    return parsed


def _validate_registry(value: Mapping[str, Any]) -> None:
    if value.get("schema_version") != REGISTRY_SCHEMA:
        raise NativeContractError(
            f"native-contract registry schema must be {REGISTRY_SCHEMA!r}"
        )
    if value.get("evidence_schema") != EVIDENCE_SCHEMA:
        raise NativeContractError(
            f"native-contract evidence schema must be {EVIDENCE_SCHEMA!r}"
        )
    tolerances = value.get("native_parity_tolerances")
    if not isinstance(tolerances, dict) or set(tolerances) != {"atol", "rtol"}:
        raise NativeContractError(
            "registry native_parity_tolerances must define exactly atol and rtol"
        )
    if any(
        not isinstance(tolerances[name], (int, float))
        or isinstance(tolerances[name], bool)
        or not 0 <= float(tolerances[name]) <= 0.001
        for name in ("atol", "rtol")
    ):
        raise NativeContractError(
            "registry native parity tolerances must be finite numbers in [0, 0.001]"
        )
    statuses = set(value.get("status_vocabulary") or ())
    if statuses != set(VALID_STATUSES):
        raise NativeContractError("registry status vocabulary drifted")
    tracks = value.get("tracks")
    if not isinstance(tracks, dict) or set(tracks) != {"F", "R", "C", "B", "D"}:
        raise NativeContractError("registry must define exactly tracks F/R/C/B/D")
    required_checks = value.get("required_checks")
    if not isinstance(required_checks, list) or len(required_checks) != len(set(required_checks)):
        raise NativeContractError("registry required_checks must be a unique list")
    models = value.get("models")
    if not isinstance(models, dict) or not models:
        raise NativeContractError("registry models must be a nonempty object")
    dispositions = value.get("historical_dispositions")
    if not isinstance(dispositions, dict) or set(dispositions) != set(models):
        raise NativeContractError("every registered model needs a historical disposition")

    for key, dossier in models.items():
        _require_string(key, "model key")
        if not isinstance(dossier, dict):
            raise NativeContractError(f"dossier {key} must be an object")
        for field in (
            "family", "model_id", "model_revision", "source_url", "source_revision",
            "overall_status", "role", "ohlcv_mode", "native_preprocessing",
        ):
            _require_string(dossier.get(field), f"{key}.{field}")
        if dossier["overall_status"] not in VALID_STATUSES:
            raise NativeContractError(f"invalid overall status for {key}")
        if not isinstance(dossier.get("pin_complete"), bool):
            raise NativeContractError(f"{key}.pin_complete must be boolean")
        capabilities = dossier.get("tracks")
        if not isinstance(capabilities, dict) or set(capabilities) != set(tracks):
            raise NativeContractError(f"{key} must define every track exactly once")
        for track, capability in capabilities.items():
            if not isinstance(capability, dict):
                raise NativeContractError(f"{key}.{track} capability must be an object")
            if capability.get("status") not in VALID_STATUSES:
                raise NativeContractError(f"invalid {key}.{track} status")
            _require_string(capability.get("reason"), f"{key}.{track}.reason")
            if capability.get("status") in ADMITTED_STATUSES:
                _require_string(capability.get("evidence_id"), f"{key}.{track}.evidence_id")
        admitted_tracks = [
            track for track, capability in capabilities.items()
            if capability["status"] in ADMITTED_STATUSES
        ]
        if admitted_tracks and dossier["overall_status"] not in ADMITTED_STATUSES:
            raise NativeContractError(
                f"{key} has admitted tracks {admitted_tracks} but blocked overall status"
            )
        if dossier["overall_status"] in ADMITTED_STATUSES and not admitted_tracks:
            raise NativeContractError(f"{key} overall status is admitted without an admitted track")
        if admitted_tracks and not dossier["pin_complete"]:
            raise NativeContractError(f"{key} cannot admit tracks with incomplete pins")
        tokenizer = dossier.get("tokenizer")
        if tokenizer is not None:
            if not isinstance(tokenizer, dict):
                raise NativeContractError(f"{key}.tokenizer must be null or an object")
            _require_string(tokenizer.get("id"), f"{key}.tokenizer.id")
            _require_string(tokenizer.get("revision"), f"{key}.tokenizer.revision")
        native_parity = dossier.get("native_parity") or {}
        if not isinstance(native_parity, dict):
            raise NativeContractError(f"{key}.native_parity must be an object")
        required_artifacts = native_parity.get("required_artifacts") or []
        if (
            not isinstance(required_artifacts, list)
            or len(required_artifacts) != len(set(required_artifacts))
            or any(
                not isinstance(name, str)
                or not name
                or not name.replace("_", "a").isalnum()
                or not name[0].isalpha()
                or not name.islower()
                for name in required_artifacts
            )
        ):
            raise NativeContractError(
                f"{key}.native_parity.required_artifacts must contain unique "
                "lowercase snake-case names"
            )
        if "reference_model" in required_artifacts:
            _require_string(
                native_parity.get("reference_model_id"),
                f"{key}.native_parity.reference_model_id",
            )
            _require_string(
                native_parity.get("reference_model_revision"),
                f"{key}.native_parity.reference_model_revision",
            )
        if "execution_source" in required_artifacts:
            execution_source = native_parity.get("execution_source_distribution")
            if not isinstance(execution_source, dict):
                raise NativeContractError(
                    f"{key}.native_parity.execution_source_distribution must be an object"
                )
            _require_string(
                execution_source.get("name"),
                f"{key}.native_parity.execution_source_distribution.name",
            )
            _require_string(
                execution_source.get("version"),
                f"{key}.native_parity.execution_source_distribution.version",
            )
        disposition = dispositions[key]
        if disposition.get("default_status") not in VALID_STATUSES:
            raise NativeContractError(f"invalid historical disposition for {key}")
        _require_string(disposition.get("reason"), f"{key}.historical.reason")


def _validate_evidence_checks(
    checks: Mapping[str, Any],
    *,
    required_checks: Iterable[str],
    field: str,
    require_exact: bool,
) -> dict[str, dict[str, Any]]:
    required = tuple(required_checks)
    if not isinstance(checks, Mapping):
        raise NativeContractError(f"{field} must be an object")
    unknown = sorted(set(checks) - set(required))
    missing = sorted(set(required) - set(checks)) if require_exact else []
    if unknown or missing:
        raise NativeContractError(
            f"{field} checks mismatch: missing={missing}, unknown={unknown}"
        )
    output: dict[str, dict[str, Any]] = {}
    for name, raw in checks.items():
        if not isinstance(raw, Mapping):
            raise NativeContractError(f"{field}.{name} must be an object")
        item = dict(raw)
        status = item.get("status")
        if status not in VALID_CHECK_STATUSES:
            raise NativeContractError(f"{field}.{name} has invalid status {status!r}")
        if status == "not_applicable" and not str(item.get("reason", "")).strip():
            raise NativeContractError(f"{field}.{name} needs a not-applicable reason")
        if status in {"pass", "fail"} and not str(item.get("evidence", "")).strip():
            raise NativeContractError(f"{field}.{name} needs concrete evidence")
        output[name] = item
    return output


def _resolved_evidence_checks(
    evidence: Mapping[str, Any], record: Mapping[str, Any], registry: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    profile_name = _require_string(record.get("profile"), "evidence.profile")
    profiles = evidence.get("check_profiles") or {}
    try:
        profile = profiles[profile_name]
    except KeyError as exc:
        raise NativeContractError(f"unknown evidence check profile {profile_name!r}") from exc
    checks = _validate_evidence_checks(
        profile,
        required_checks=registry["required_checks"],
        field=f"evidence profile {profile_name}",
        require_exact=True,
    )
    overrides = _validate_evidence_checks(
        record.get("checks") or {},
        required_checks=registry["required_checks"],
        field=f"evidence record {record.get('arm_key')}:{record.get('track')}",
        require_exact=False,
    )
    checks.update(overrides)
    return checks


def _validate_evidence(value: Mapping[str, Any], registry: Mapping[str, Any]) -> None:
    if value.get("schema_version") != EVIDENCE_SCHEMA:
        raise NativeContractError(f"native-contract evidence schema must be {EVIDENCE_SCHEMA!r}")
    if value.get("methodology_commit") != registry.get("methodology_commit"):
        raise NativeContractError("native-contract evidence methodology revision is stale")
    _require_string(value.get("generated_utc"), "evidence.generated_utc")
    profiles = value.get("check_profiles")
    records = value.get("records")
    if not isinstance(profiles, Mapping) or not profiles:
        raise NativeContractError("evidence check_profiles must be a nonempty object")
    if not isinstance(records, Mapping) or not records:
        raise NativeContractError("evidence records must be a nonempty object")
    for name, profile in profiles.items():
        _require_string(name, "evidence profile name")
        _validate_evidence_checks(
            profile,
            required_checks=registry["required_checks"],
            field=f"evidence profile {name}",
            require_exact=True,
        )

    for evidence_id, record in records.items():
        _require_string(evidence_id, "evidence id")
        if not isinstance(record, Mapping):
            raise NativeContractError(f"evidence record {evidence_id} must be an object")
        arm_key = _require_string(record.get("arm_key"), f"{evidence_id}.arm_key")
        track = _require_string(record.get("track"), f"{evidence_id}.track")
        if arm_key not in registry["models"] or track not in registry["tracks"]:
            raise NativeContractError(f"evidence record {evidence_id} has unknown arm/track")
        if record.get("status") not in VALID_EVIDENCE_STATUSES:
            raise NativeContractError(f"evidence record {evidence_id} has invalid status")
        if record.get("status") in {"pass", "research_only_pass"}:
            runtime = record.get("admitted_runtime")
            if not isinstance(runtime, Mapping) or not runtime:
                raise NativeContractError(
                    f"evidence record {evidence_id} needs an admitted_runtime contract"
                )
        identity = record.get("identity")
        if not isinstance(identity, Mapping):
            raise NativeContractError(f"evidence record {evidence_id} needs identity")
        dossier = registry["models"][arm_key]
        for field in ("model_id", "model_revision", "source_revision"):
            if identity.get(field) != dossier[field]:
                raise NativeContractError(
                    f"evidence record {evidence_id} {field} mismatch: "
                    f"expected {dossier[field]!r}, got {identity.get(field)!r}"
                )
        tokenizer = dossier.get("tokenizer") or {}
        external_tokenizer = tokenizer and tokenizer.get("revision") != "model_revision"
        if external_tokenizer and (
            identity.get("tokenizer_id") != tokenizer.get("id")
            or identity.get("tokenizer_revision") != tokenizer.get("revision")
        ):
            raise NativeContractError(f"evidence record {evidence_id} tokenizer mismatch")
        if not isinstance(record.get("environment"), Mapping) or not record["environment"]:
            raise NativeContractError(f"evidence record {evidence_id} needs an environment")
        if record.get("runtime_lock") is not None:
            from .native_parity_runtime import NativeParityRuntimeError, validate_runtime_lock

            try:
                validate_runtime_lock(record["runtime_lock"])
            except NativeParityRuntimeError as exc:
                raise NativeContractError(
                    f"evidence record {evidence_id} runtime lock is invalid: {exc}"
                ) from exc
        if record.get("profile") == "generated_bundle":
            bundle = record.get("bundle")
            if not isinstance(bundle, Mapping):
                raise NativeContractError(
                    f"generated evidence record {evidence_id} needs a raw bundle binding"
                )
            if bundle.get("path_base") != "evidence_registry_parent":
                raise NativeContractError(
                    f"generated evidence record {evidence_id} has invalid bundle path base"
                )
            for field in (
                "path", "bundle_sha256", "fixture_sha256", "command_sha256",
                "result_sha256", "stdout_sha256", "stderr_sha256",
            ):
                _require_string(bundle.get(field), f"{evidence_id}.bundle.{field}")
        _resolved_evidence_checks(value, record, registry)

    for arm_key, dossier in registry["models"].items():
        for track, capability in dossier["tracks"].items():
            if capability["status"] not in ADMITTED_STATUSES:
                continue
            evidence_id = capability["evidence_id"]
            record = records.get(evidence_id)
            if not isinstance(record, Mapping):
                raise NativeContractError(
                    f"admitted {arm_key}.{track} references missing evidence {evidence_id!r}"
                )
            if record.get("arm_key") != arm_key or record.get("track") != track:
                raise NativeContractError(f"evidence {evidence_id!r} arm/track mismatch")
            expected_status = (
                "research_only_pass" if capability["status"] == "research_only" else "pass"
            )
            if record.get("status") != expected_status:
                raise NativeContractError(
                    f"admitted {arm_key}.{track} requires evidence status {expected_status!r}"
                )
            checks = _resolved_evidence_checks(value, record, registry)
            failed = [name for name, item in checks.items() if item["status"] == "fail"]
            if failed:
                raise NativeContractError(
                    f"admitted {arm_key}.{track} has failed technical checks: {failed}"
                )
            mandatory = [
                name for name in sorted(MANDATORY_TECHNICAL_CHECKS)
                if checks[name]["status"] != "pass"
            ]
            if mandatory:
                raise NativeContractError(
                    f"admitted {arm_key}.{track} lacks mandatory technical passes: {mandatory}"
                )


@lru_cache(maxsize=8)
def load_registry(path: str | Path = REGISTRY_PATH) -> dict[str, Any]:
    path = Path(path).resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    _validate_registry(value)
    evidence_path = evidence_path_for_registry(path)
    if not evidence_path.is_file():
        raise NativeContractError(
            f"native-contract evidence not found beside registry: {evidence_path}"
        )
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    _validate_evidence(evidence, value)
    return value


@lru_cache(maxsize=8)
def load_evidence(path: str | Path = REGISTRY_PATH) -> dict[str, Any]:
    registry_path = Path(path).resolve()
    load_registry(registry_path)
    return json.loads(evidence_path_for_registry(registry_path).read_text(encoding="utf-8"))


def registry_sha256(path: str | Path = REGISTRY_PATH) -> str:
    return content_sha256(load_registry(path))


def evidence_sha256(path: str | Path = REGISTRY_PATH) -> str:
    return content_sha256(load_evidence(path))


def technical_evidence(
    arm_key: str, track: str, path: str | Path = REGISTRY_PATH
) -> tuple[str, dict[str, Any], dict[str, dict[str, Any]]]:
    registry = load_registry(path)
    dossier = registry["models"].get(str(arm_key))
    if not isinstance(dossier, Mapping) or track not in registry["tracks"]:
        raise NativeContractError(f"unknown foundation arm/track: {arm_key}.{track}")
    capability = dossier["tracks"][track]
    evidence_id = _require_string(
        capability.get("evidence_id"), f"{arm_key}.{track}.evidence_id"
    )
    evidence = load_evidence(path)
    record = evidence["records"].get(evidence_id)
    if not isinstance(record, Mapping):
        raise NativeContractError(f"technical evidence not found: {evidence_id}")
    value = dict(record)
    return evidence_id, value, _resolved_evidence_checks(evidence, value, registry)


def technical_runtime_contract(
    arm_key: str, track: str, path: str | Path = REGISTRY_PATH
) -> dict[str, Any]:
    """Return the exact runtime surface covered by current technical evidence."""
    _, record, _ = technical_evidence(arm_key, track, path)
    runtime = record.get("admitted_runtime")
    if not isinstance(runtime, Mapping) or not runtime:
        raise NativeContractError(f"{arm_key}.{track} has no admitted runtime contract")
    return dict(runtime)


def verify_technical_evidence_bundle(
    arm_key: str,
    track: str,
    path: str | Path = REGISTRY_PATH,
) -> dict[str, Any] | None:
    """Reopen and verify the raw bundle bound by installed technical evidence.

    Transitional hand-reviewed records have no bundle and return ``None``.  Generated
    records must resolve relative to the canonical registry/evidence directory.  This
    function is intentionally called by admission-report build and verification so an
    installed JSON claim cannot authorize execution after its raw proof is moved,
    deleted, or tampered with.
    """
    evidence_id, record, _ = technical_evidence(arm_key, track, path)
    bundle = record.get("bundle")
    if bundle is None:
        if record.get("profile") == "generated_bundle":
            raise NativeContractError(
                f"generated technical evidence {evidence_id!r} has no raw bundle binding"
            )
        return None
    if not isinstance(bundle, Mapping):
        raise NativeContractError(f"technical evidence {evidence_id!r} bundle is invalid")
    if bundle.get("path_base") != "evidence_registry_parent":
        raise NativeContractError(
            f"technical evidence {evidence_id!r} has unsupported bundle path base"
        )
    relative = _require_string(bundle.get("path"), f"{evidence_id}.bundle.path")
    relative_path = Path(relative)
    if relative_path.is_absolute():
        raise NativeContractError(f"technical evidence {evidence_id!r} bundle path is absolute")
    bundle_path = (evidence_path_for_registry(path).parent / relative_path).resolve()
    if not (bundle_path / "bundle_manifest.json").is_file():
        raise NativeContractError(
            f"technical evidence {evidence_id!r} raw bundle is unavailable: {bundle_path}"
        )
    # Lazy import avoids a module cycle: the bundle verifier itself consumes this
    # registry module.
    from .native_evidence_bundle import NativeEvidenceError, verify_parity_bundle

    try:
        manifest, result = verify_parity_bundle(
            bundle_path,
            registry_path=path,
            # The bundle records the clean source/model trees used to produce it.
            # Installing reviewed evidence advances the repository, so canonical
            # replay verifies the immutable archived files.  Admission reports bind
            # and verify the current execution artifacts separately.
            verify_external_artifacts=False,
        )
    except NativeEvidenceError as exc:
        raise NativeContractError(
            f"technical evidence {evidence_id!r} raw bundle failed verification: {exc}"
        ) from exc
    expected = {
        "bundle_sha256": manifest.get("bundle_sha256"),
        "fixture_sha256": (manifest.get("fixture") or {}).get("fixture_sha256"),
        "command_sha256": (manifest.get("command") or {}).get("command_sha256"),
        "result_sha256": (manifest.get("result") or {}).get("sha256"),
        "stdout_sha256": ((manifest.get("logs") or {}).get("stdout") or {}).get("sha256"),
        "stderr_sha256": ((manifest.get("logs") or {}).get("stderr") or {}).get("sha256"),
    }
    mismatches = {
        key: {"expected": value, "recorded": bundle.get(key)}
        for key, value in expected.items()
        if bundle.get(key) != value
    }
    if mismatches:
        raise NativeContractError(
            f"technical evidence {evidence_id!r} bundle binding drift: {mismatches}"
        )
    if result.get("admitted_runtime") != record.get("admitted_runtime"):
        raise NativeContractError(
            f"technical evidence {evidence_id!r} runtime differs from its raw result"
        )
    return {"path": str(bundle_path), "manifest": manifest, "result": result}


def validate_runtime_contract(
    arm_key: str,
    track: str,
    actual: Mapping[str, Any],
    *,
    path: str | Path = REGISTRY_PATH,
) -> dict[str, Any]:
    """Require the caller's complete runtime surface to equal technical evidence.

    A subset comparison is unsafe here: omitting a field such as a frequency token,
    pooling choice, sampling policy, or channel mode would silently authorize a runtime
    that was never exercised.  Callers must therefore state every evidence-bound fact.
    """
    contract = technical_runtime_contract(arm_key, track, path)
    supplied = dict(actual)
    missing = sorted(set(contract) - set(supplied))
    unexpected = sorted(set(supplied) - set(contract))
    mismatches = {
        key: {"expected": contract[key], "actual": supplied[key]}
        for key in sorted(set(contract) & set(supplied))
        if contract[key] != supplied[key]
    }
    if missing or unexpected or mismatches:
        raise NativeContractError(
            f"{arm_key}.{track} runtime contract mismatch: "
            f"missing={missing}, unexpected={unexpected}, values={mismatches}"
        )
    return contract


@dataclass(frozen=True)
class TrackCapability:
    track: str
    status: str
    reason: str
    evidence_id: str | None = None
    training_admitted: bool = False


@dataclass(frozen=True)
class FoundationArm:
    """Compatibility view backed by the authoritative per-track dossier."""

    key: str
    family: str
    model_id: str
    model_revision: str
    source_url: str
    source_revision: str
    license: str
    role: str
    adaptation: str
    ohlcv_mode: str
    supported_training: bool
    overall_status: str
    pin_complete: bool
    tokenizer_id: str | None
    tokenizer_revision: str | None
    tracks: tuple[TrackCapability, ...]
    adaptation_routes: tuple[str, ...]

    def manifest(self) -> dict[str, Any]:
        value = asdict(self)
        value["tracks"] = {
            item.track: {
                "status": item.status,
                "reason": item.reason,
                "evidence_id": item.evidence_id,
                "training_admitted": item.training_admitted,
            }
            for item in self.tracks
        }
        value["adaptation_routes"] = list(self.adaptation_routes)
        return value

    def capability(self, track: str) -> TrackCapability:
        for capability in self.tracks:
            if capability.track == track:
                return capability
        raise NativeContractError(f"unknown track {track!r}")

    @property
    def dossier_sha256(self) -> str:
        return dossier_sha256(self.key)

    @property
    def training_admitted(self) -> bool:
        return self.overall_status in ADMITTED_STATUSES and any(
            capability.training_admitted for capability in self.tracks
        )


def _arm_from_dossier(
    key: str, dossier: Mapping[str, Any], path: str | Path
) -> FoundationArm:
    tokenizer = dossier.get("tokenizer") or {}
    routes = tuple(str(value) for value in dossier.get("adaptation_routes") or ())
    return FoundationArm(
        key=key,
        family=str(dossier["family"]),
        model_id=str(dossier["model_id"]),
        model_revision=str(dossier["model_revision"]),
        source_url=str(dossier["source_url"]),
        source_revision=str(dossier["source_revision"]),
        license=str((dossier.get("license") or {}).get("id", "unresolved")),
        role=str(dossier["role"]),
        adaptation=(routes[0] if routes else "none"),
        ohlcv_mode=str(dossier["ohlcv_mode"]),
        supported_training=bool(routes),
        overall_status=str(dossier["overall_status"]),
        pin_complete=bool(dossier["pin_complete"]),
        tokenizer_id=tokenizer.get("id"),
        tokenizer_revision=tokenizer.get("revision"),
        tracks=tuple(
            TrackCapability(
                track=track,
                status=str(capability["status"]),
                reason=str(capability["reason"]),
                evidence_id=capability.get("evidence_id"),
                training_admitted=(
                    capability["status"] in ADMITTED_STATUSES
                    and all(
                        item["status"] == "pass"
                        for name, item in technical_evidence(key, track, path)[2].items()
                        if name in TRAINING_CHECKS
                    )
                ),
            )
            for track, capability in dossier["tracks"].items()
        ),
        adaptation_routes=routes,
    )


def all_arms(path: str | Path = REGISTRY_PATH) -> dict[str, FoundationArm]:
    registry = load_registry(path)
    return {
        key: _arm_from_dossier(key, dossier, path)
        for key, dossier in registry["models"].items()
    }


def get_arm(key: str, path: str | Path = REGISTRY_PATH) -> FoundationArm:
    try:
        return all_arms(path)[str(key)]
    except KeyError as exc:
        raise NativeContractError(f"unknown foundation arm: {key}") from exc


def get_dossier(key: str, path: str | Path = REGISTRY_PATH) -> dict[str, Any]:
    registry = load_registry(path)
    try:
        return dict(registry["models"][str(key)])
    except KeyError as exc:
        raise NativeContractError(f"unknown foundation arm: {key}") from exc


def dossier_sha256(key: str, path: str | Path = REGISTRY_PATH) -> str:
    return content_sha256(get_dossier(key, path))


def historical_disposition(key: str, path: str | Path = REGISTRY_PATH) -> dict[str, str]:
    registry = load_registry(path)
    try:
        return dict(registry["historical_dispositions"][str(key)])
    except KeyError as exc:
        raise NativeContractError(f"unknown foundation arm: {key}") from exc


def validate_identity(
    key: str,
    *,
    model_id: str,
    model_revision: str,
    source_revision: str,
    tokenizer_id: str | None = None,
    tokenizer_revision: str | None = None,
    path: str | Path = REGISTRY_PATH,
) -> FoundationArm:
    """Reject model, source, or tokenizer drift before any package/model loading."""
    arm = get_arm(key, path)
    actual = (str(model_id), str(model_revision), str(source_revision))
    expected = (arm.model_id, arm.model_revision, arm.source_revision)
    if actual != expected:
        raise NativeContractError(
            f"{key} model/source identity mismatch: expected {expected}, got {actual}"
        )
    expected_tokenizer = (arm.tokenizer_id, arm.tokenizer_revision)
    supplied_tokenizer = (tokenizer_id, tokenizer_revision)
    if expected_tokenizer != (None, None) and supplied_tokenizer != expected_tokenizer:
        raise NativeContractError(
            f"{key} tokenizer mismatch: expected {expected_tokenizer}, "
            f"got {supplied_tokenizer}"
        )
    if expected_tokenizer == (None, None) and supplied_tokenizer != (None, None):
        raise NativeContractError(f"{key} does not declare an external tokenizer")
    return arm


def normalize_check_results(
    checks: Mapping[str, bool | Mapping[str, Any]],
    *,
    path: str | Path = REGISTRY_PATH,
) -> dict[str, dict[str, Any]]:
    """Normalize harness results and reject missing, unknown, or vague checks."""
    required = tuple(load_registry(path)["required_checks"])
    unknown = sorted(set(checks) - set(required))
    missing = sorted(set(required) - set(checks))
    if unknown or missing:
        raise NativeContractError(
            f"admission checks mismatch: missing={missing}, unknown={unknown}"
        )
    normalized: dict[str, dict[str, Any]] = {}
    for name in required:
        value = checks[name]
        if isinstance(value, bool):
            item = {"status": "pass" if value else "fail"}
        elif isinstance(value, Mapping):
            item = dict(value)
        else:
            raise NativeContractError(f"check {name} must be boolean or object")
        status = item.get("status")
        if status not in VALID_CHECK_STATUSES:
            raise NativeContractError(f"check {name} has invalid status {status!r}")
        if status == "not_applicable" and not str(item.get("reason", "")).strip():
            raise NativeContractError(f"check {name} needs a not-applicable reason")
        if status in {"pass", "fail"}:
            evidence = str(item.get("evidence", "")).strip()
            metrics = item.get("metrics")
            if not evidence and not (isinstance(metrics, Mapping) and metrics):
                raise NativeContractError(
                    f"check {name} needs concrete evidence or nonempty metrics"
                )
        normalized[name] = item
    return normalized


def measure_runtime_environment(
    expected: Mapping[str, Any],
    requested: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Measure Python/package identity and validate non-measurable runtime controls.

    Package and interpreter fields are never copied from caller JSON.  Fields such as
    dtype and device are execution controls rather than package identity; callers may
    declare them, but they must exactly match the evidence-covered contract.
    """
    if not isinstance(expected, Mapping) or not expected:
        raise NativeContractError("technical evidence requires an environment contract")
    supplied = dict(requested or {})
    allowed = set(expected) | {"use_scope"}
    extras = sorted(set(supplied) - allowed)
    if extras:
        raise NativeContractError(f"admission environment has unsupported fields: {extras}")
    measured: dict[str, Any] = {}
    for name, expected_value in expected.items():
        if name in RUNTIME_CONTROL_FIELDS:
            continue
        if name == "python":
            actual: Any = platform.python_version()
        elif name == "executable":
            actual = str(Path(sys.executable).resolve())
        else:
            try:
                actual = version(name)
            except PackageNotFoundError as exc:
                raise NativeContractError(
                    f"required runtime package {name!r} is not installed"
                ) from exc
        measured[name] = actual
        if actual != expected_value:
            raise NativeContractError(
                f"measured runtime environment drift for {name!r}: "
                f"expected {expected_value!r}, got {actual!r}"
            )
        if name in supplied and supplied[name] != actual:
            raise NativeContractError(
                f"reported runtime environment disagrees with measurement for {name!r}: "
                f"reported {supplied[name]!r}, measured {actual!r}"
            )
    return measured


def _validated_runtime_controls(
    expected_environment: Mapping[str, Any],
    actual: Mapping[str, Any] | None,
) -> dict[str, Any]:
    expected = {
        name: expected_environment[name]
        for name in RUNTIME_CONTROL_FIELDS
        if name in expected_environment
    }
    supplied = dict(actual or {})
    if supplied != expected:
        raise NativeContractError(
            "runtime controls must be supplied by the execution consumer and match "
            f"technical evidence exactly: expected={expected}, actual={supplied}"
        )
    return expected


def _verified_runtime_lock(record: Mapping[str, Any]) -> dict[str, Any]:
    expected = record.get("runtime_lock")
    if not isinstance(expected, Mapping):
        raise NativeContractError(
            "technical evidence predates the complete runtime lock and cannot authorize execution"
        )
    from .native_parity_runtime import (
        NativeParityRuntimeError,
        measure_runtime_lock,
        validate_runtime_lock,
    )

    try:
        expected_value = validate_runtime_lock(expected)
        actual_value = validate_runtime_lock(measure_runtime_lock())
    except NativeParityRuntimeError as exc:
        raise NativeContractError(f"runtime lock measurement is invalid: {exc}") from exc
    if actual_value != expected_value:
        raise NativeContractError(
            "complete runtime lock drifted from technical evidence"
        )
    return actual_value


def _runtime_artifact_description(name: str, artifact: str | Path) -> dict[str, Any]:
    artifact_path = Path(artifact).expanduser().absolute()
    if name == "source" and artifact_path.name.endswith(".dist-info"):
        from .native_parity_runtime import validate_distribution_record

        try:
            validate_distribution_record(artifact_path)
        except RuntimeError as exc:
            raise NativeContractError(
                f"runtime artifact {name!r} has invalid installed-package bytes: {exc}"
            ) from exc
    # Lazy import avoids the native-evidence module's dependency back on this module.
    from .native_evidence_bundle import NativeEvidenceError, _tree_description

    try:
        return _tree_description(
            artifact_path,
            # Executable source trees must not contain untracked importable code.
            git_untracked_policy="reject" if name == "source" else "ignore",
        )
    except NativeEvidenceError as exc:
        raise NativeContractError(f"runtime artifact {name!r} is unsafe: {exc}") from exc


def _current_execution_artifact() -> dict[str, Any]:
    """Hash the predeclared executable Python surface, excluding mutable evidence/docs.

    Binding the entire Git checkout would be self-referential because installing evidence
    changes the repository commit.  This manifest instead covers the package and every
    executable consumer script while excluding registries, evidence, outputs, and docs.
    """
    from .native_evidence_bundle import NativeEvidenceError, execution_code_description

    checkout = Path(__file__).resolve().parents[2]
    if not (checkout / ".git").exists():
        raise NativeContractError(
            "operational admission currently requires a clean source checkout; "
            "wheel installations support archive inspection only"
        )
    try:
        return execution_code_description(checkout)
    except NativeEvidenceError as exc:
        raise NativeContractError(f"executing code cannot be sealed: {exc}") from exc


def _technical_execution_artifact(raw: Mapping[str, Any]) -> dict[str, Any]:
    if raw.get("kind") == "execution_code":
        return dict(raw)
    entries = []
    for item in raw.get("entries") or ():
        path = str(item.get("path", ""))
        if not path.endswith(".py") or not path.startswith(("futures_foundation/", "scripts/")):
            continue
        entries.append({
            "path": path,
            "sha256": item.get("sha256"),
            "size_bytes": item.get("size_bytes"),
        })
    entries.sort(key=lambda item: item["path"])
    if not entries:
        raise NativeContractError(
            "technical runner evidence does not bind executable package/script sources"
        )
    return {
        "path": raw.get("path"),
        "kind": "execution_code",
        "sha256": content_sha256(entries),
        "file_count": len(entries),
        "size_bytes": sum(int(item["size_bytes"]) for item in entries),
        "entries": entries,
    }


def _technical_runtime_artifacts(
    arm_key: str,
    track: str,
    path: str | Path,
) -> dict[str, dict[str, Any]]:
    verified = verify_technical_evidence_bundle(arm_key, track, path)
    if not verified:
        raise NativeContractError(
            f"{arm_key}.{track} has no generated bundle to bind runtime artifacts"
        )
    artifacts = (verified["manifest"].get("bound_artifacts") or {})
    if not isinstance(artifacts, Mapping):
        raise NativeContractError("technical evidence has no bound artifact manifest")
    required = {
        str(name): dict(item)
        for name, item in artifacts.items()
    }
    if not {"model", "source", "runner"}.issubset(required):
        raise NativeContractError(
            "technical evidence must bind current model, source, and execution-code artifacts"
        )
    required["runner"] = _technical_execution_artifact(required["runner"])
    return required


def _assert_technical_artifact_match(
    name: str,
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    if name == "source" and expected.get("kind") == "git_checkout":
        fields = ("kind", "head_revision", "origin", "entries", "size_bytes")
    else:
        fields = ("kind", "sha256", "size_bytes")
    mismatches = {
        field: {"expected": expected.get(field), "actual": actual.get(field)}
        for field in fields
        if actual.get(field) != expected.get(field)
    }
    if mismatches:
        raise NativeContractError(
            f"runtime artifact {name!r} differs from technical evidence: {mismatches}"
        )


def _normalize_runtime_artifacts(
    artifacts: Mapping[str, str | Path] | None,
    *,
    technical: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    supplied = dict(artifacts or {})
    if "runner" in supplied:
        raise NativeContractError("runner artifact is measured from executing code, not supplied")
    missing = sorted((set(technical) - {"runner"}) - set(supplied))
    if missing:
        raise NativeContractError(
            f"admission requires current runtime artifacts: missing={missing}"
        )
    output: dict[str, dict[str, Any]] = {}
    runner = _current_execution_artifact()
    _assert_technical_artifact_match("runner", runner, technical["runner"])
    output["runner"] = runner
    for name, artifact in supplied.items():
        _require_string(name, "artifact name")
        if not isinstance(artifact, (str, Path)):
            raise NativeContractError(
                f"runtime artifact {name!r} must be a current file or directory path"
            )
        description = _runtime_artifact_description(name, artifact)
        if name in technical:
            _assert_technical_artifact_match(name, description, technical[name])
        output[name] = description
    return output


def _report_integrity_payload(report: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(report)
    value.pop("integrity", None)
    return value


def _approval_target_payload(report: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(report)
    value.pop("integrity", None)
    value.pop("approvals", None)
    value.pop("approval_target_sha256", None)
    value.pop("finalized_utc", None)
    return value


def approval_signature_payload(
    approval_target_sha256: str,
    approval: Mapping[str, Any],
) -> bytes:
    """Return the exact canonical message an independent reviewer must sign."""
    return canonical_json({
        "schema_version": "ffm_native_approval_signature_v1",
        "approval_target_sha256": _require_string(
            approval_target_sha256, "approval_target_sha256"
        ),
        "reviewer": _require_string(approval.get("reviewer"), "approval.reviewer"),
        "key_id": _require_string(approval.get("key_id"), "approval.key_id"),
        "algorithm": "ed25519",
        "decision": approval.get("decision"),
        "approved_utc": approval.get("approved_utc"),
    })


def attach_integrity(report: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(report)
    value["integrity"] = {
        "algorithm": "sha256",
        "digest": content_sha256(_report_integrity_payload(value)),
    }
    return value


def _validated_approval_identities(
    approvals: Any,
    *,
    request_created: datetime,
    approval_target_sha256: str,
    trust_store: Mapping[str, Any],
    minimum_approvals: int = 2,
) -> set[str]:
    if minimum_approvals < 2:
        raise NativeContractError("admission approval floor cannot be lower than 2")
    if not isinstance(approvals, list):
        raise NativeContractError("admission approvals must be a list")
    trusted_keys = trust_store.get("keys") if isinstance(trust_store, Mapping) else None
    if not isinstance(trusted_keys, Mapping) or not trusted_keys:
        raise NativeContractError("trusted-approver registry is empty; admission fails closed")
    identities: set[str] = set()
    for approval in approvals:
        if not isinstance(approval, Mapping):
            raise NativeContractError("each approval must be an object")
        reviewer = _require_string(approval.get("reviewer"), "approval.reviewer")
        approved = _parse_utc(approval.get("approved_utc"), "approval.approved_utc")
        if approved < request_created:
            raise NativeContractError(
                f"reviewer {reviewer} approval predates the signed admission request"
            )
        if approved > datetime.now(timezone.utc):
            raise NativeContractError(f"reviewer {reviewer} approval is in the future")
        if approval.get("decision") != "approve":
            raise NativeContractError(f"reviewer {reviewer} did not approve")
        key_id = _require_string(approval.get("key_id"), "approval.key_id")
        trusted = trusted_keys.get(key_id)
        if not isinstance(trusted, Mapping):
            raise NativeContractError(
                f"reviewer {reviewer} used untrusted approval key {key_id!r}"
            )
        if trusted.get("algorithm") != "ed25519" or approval.get("algorithm") != "ed25519":
            raise NativeContractError(f"reviewer {reviewer} approval must use ed25519")
        if reviewer.strip().casefold() != str(trusted.get("reviewer", "")).strip().casefold():
            raise NativeContractError(
                f"approval key {key_id!r} is not trusted for reviewer {reviewer!r}"
            )
        signature_text = _require_string(approval.get("signature"), "approval.signature")
        try:
            signature = base64.b64decode(signature_text, validate=True)
        except (ValueError, TypeError) as exc:
            raise NativeContractError(
                f"reviewer {reviewer} approval signature is not valid base64"
            ) from exc
        try:
            from cryptography.exceptions import InvalidSignature
            from cryptography.hazmat.primitives import serialization
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        except ImportError as exc:
            raise NativeContractError(
                "cryptography is required to authenticate admission approvals"
            ) from exc
        try:
            public_key = serialization.load_pem_public_key(
                _require_string(
                    trusted.get("public_key_pem"),
                    f"trusted key {key_id}.public_key_pem",
                ).encode("ascii")
            )
        except (ValueError, TypeError, UnicodeEncodeError) as exc:
            raise NativeContractError(f"trusted approval key {key_id!r} is invalid") from exc
        if not isinstance(public_key, Ed25519PublicKey):
            raise NativeContractError(f"trusted approval key {key_id!r} is not Ed25519")
        try:
            public_key.verify(
                signature,
                approval_signature_payload(approval_target_sha256, approval),
            )
        except InvalidSignature as exc:
            raise NativeContractError(
                f"reviewer {reviewer} approval signature is invalid"
            ) from exc
        identities.add(reviewer.strip().casefold())
    if len(identities) < minimum_approvals:
        raise NativeContractError(
            f"admission requires {minimum_approvals} independent approvals, "
            f"got {len(identities)}"
        )
    return identities


def build_admission_request(
    *,
    arm_key: str,
    track: str,
    status: str,
    checks: Mapping[str, bool | Mapping[str, Any]],
    environment: Mapping[str, Any],
    route: str | None = None,
    artifacts: Mapping[str, str | Path] | None = None,
    created_utc: str,
    path: str | Path = REGISTRY_PATH,
) -> dict[str, Any]:
    """Build the immutable report body that trusted reviewers independently sign."""
    _parse_utc(created_utc, "created_utc")
    arm = get_arm(arm_key, path)
    if track not in {"F", "R", "C", "B", "D"}:
        raise NativeContractError(f"unknown track {track!r}")
    if status not in ADMITTED_STATUSES:
        raise NativeContractError(f"admission report status {status!r} is not admissible")
    normalized = normalize_check_results(checks, path=path)
    evidence_id, evidence_record, evidence_checks = technical_evidence(arm.key, track, path)
    technical_artifacts = _technical_runtime_artifacts(arm.key, track, path)
    expected_evidence_status = (
        "research_only_pass" if status == "research_only" else "pass"
    )
    if evidence_record.get("status") != expected_evidence_status:
        raise NativeContractError(
            f"technical evidence {evidence_id!r} is not admissible for status {status!r}"
        )
    evidence_environment = dict(evidence_record["environment"])
    runtime_lock = _verified_runtime_lock(evidence_record)
    environment_value = measure_runtime_environment(evidence_environment, environment)
    runtime_controls = _validated_runtime_controls(
        evidence_environment,
        {name: environment[name] for name in RUNTIME_CONTROL_FIELDS if name in environment},
    )
    use_scope = environment.get("use_scope")
    if status == "research_only" and use_scope != "research_noncommercial":
        raise NativeContractError(
            "research-only admission requires environment.use_scope='research_noncommercial'"
        )
    normalized_artifacts = _normalize_runtime_artifacts(
        artifacts, technical=technical_artifacts
    )
    trust_store = load_trusted_approvers(path)
    value = {
        "schema_version": REPORT_SCHEMA,
        "request_created_utc": created_utc,
        "arm_key": arm.key,
        "track": track,
        "route": route,
        "status": status,
        "methodology_commit": load_registry(path)["methodology_commit"],
        "registry_sha256": registry_sha256(path),
        "dossier_sha256": dossier_sha256(arm.key, path),
        "evidence_registry_sha256": evidence_sha256(path),
        "technical_evidence_id": evidence_id,
        "technical_evidence_sha256": content_sha256(evidence_record),
        "technical_checks_sha256": content_sha256(evidence_checks),
        "technical_runtime_sha256": content_sha256(evidence_record["admitted_runtime"]),
        "technical_environment_sha256": content_sha256(evidence_environment),
        "technical_runtime_lock_sha256": content_sha256(evidence_record["runtime_lock"]),
        "trusted_approvers_sha256": content_sha256(trust_store),
        "checks": normalized,
        "environment": environment_value,
        "runtime_lock": runtime_lock,
        "runtime_controls": runtime_controls,
        "use_scope": use_scope,
        "artifacts": normalized_artifacts,
    }
    value["approval_target_sha256"] = content_sha256(_approval_target_payload(value))
    return value


def build_admission_report(
    *,
    arm_key: str,
    track: str,
    status: str,
    checks: Mapping[str, bool | Mapping[str, Any]],
    approvals: list[Mapping[str, Any]],
    environment: Mapping[str, Any],
    route: str | None = None,
    artifacts: Mapping[str, str | Path] | None = None,
    created_utc: str,
    path: str | Path = REGISTRY_PATH,
) -> dict[str, Any]:
    """Build a signed, runtime-bound report; blocked dossiers remain blocked."""
    value = build_admission_request(
        arm_key=arm_key,
        track=track,
        status=status,
        checks=checks,
        environment=environment,
        route=route,
        artifacts=artifacts,
        created_utc=created_utc,
        path=path,
    )
    _validated_approval_identities(
        approvals,
        request_created=_parse_utc(created_utc, "created_utc"),
        approval_target_sha256=value["approval_target_sha256"],
        trust_store=load_trusted_approvers(path),
        minimum_approvals=2,
    )
    value["approvals"] = [dict(item) for item in approvals]
    value["finalized_utc"] = max(
        _parse_utc(item.get("approved_utc"), "approval.approved_utc")
        for item in approvals
    ).isoformat().replace("+00:00", "Z")
    return attach_integrity(value)


def _read_report(report: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(report, Mapping):
        return dict(report)
    path = Path(report)
    if not path.is_file():
        raise NativeContractError(f"admission report not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def verify_admission_report(
    report: str | Path | Mapping[str, Any],
    *,
    arm_key: str,
    track: str,
    route: str | None = None,
    require_training: bool = False,
    required_artifacts: Mapping[str, str | Path] | None = None,
    runtime_controls: Mapping[str, Any] | None = None,
    minimum_approvals: int = 2,
    path: str | Path = REGISTRY_PATH,
) -> dict[str, Any]:
    """Verify an admission report against the current registry and requested execution path."""
    value = _read_report(report)
    if value.get("schema_version") != REPORT_SCHEMA:
        raise NativeContractError(f"admission report schema must be {REPORT_SCHEMA!r}")
    integrity = value.get("integrity") or {}
    if integrity.get("algorithm") != "sha256":
        raise NativeContractError("admission report must use sha256 integrity")
    expected_digest = content_sha256(_report_integrity_payload(value))
    if integrity.get("digest") != expected_digest:
        raise NativeContractError("admission report integrity mismatch")
    target_digest = content_sha256(_approval_target_payload(value))
    if value.get("approval_target_sha256") != target_digest:
        raise NativeContractError("admission report approval target mismatch")

    arm = get_arm(arm_key, path)
    capability = arm.capability(track)
    if value.get("arm_key") != arm.key or value.get("track") != track:
        raise NativeContractError("admission report arm/track mismatch")
    if value.get("route") != route:
        raise NativeContractError(
            f"admission report route mismatch: expected {route!r}, got {value.get('route')!r}"
        )
    if value.get("methodology_commit") != load_registry(path)["methodology_commit"]:
        raise NativeContractError("admission report methodology revision is stale")
    if value.get("registry_sha256") != registry_sha256(path):
        raise NativeContractError("admission report registry hash is stale")
    if value.get("dossier_sha256") != dossier_sha256(arm.key, path):
        raise NativeContractError("admission report dossier hash is stale")
    if value.get("evidence_registry_sha256") != evidence_sha256(path):
        raise NativeContractError("admission report technical-evidence registry is stale")
    evidence_id, evidence_record, evidence_checks = technical_evidence(arm.key, track, path)
    verify_technical_evidence_bundle(arm.key, track, path)
    if value.get("technical_evidence_id") != evidence_id:
        raise NativeContractError("admission report technical-evidence ID mismatch")
    if value.get("technical_evidence_sha256") != content_sha256(evidence_record):
        raise NativeContractError("admission report technical-evidence record is stale")
    if value.get("technical_checks_sha256") != content_sha256(evidence_checks):
        raise NativeContractError("admission report resolved technical checks are stale")
    if value.get("technical_runtime_sha256") != content_sha256(evidence_record["admitted_runtime"]):
        raise NativeContractError("admission report technical runtime contract is stale")
    if value.get("technical_environment_sha256") != content_sha256(evidence_record["environment"]):
        raise NativeContractError("admission report technical environment contract is stale")
    if value.get("technical_runtime_lock_sha256") != content_sha256(
        evidence_record.get("runtime_lock")
    ):
        raise NativeContractError("admission report technical runtime lock is stale")
    trust_store = load_trusted_approvers(path)
    if value.get("trusted_approvers_sha256") != content_sha256(trust_store):
        raise NativeContractError("admission report trusted-approver registry is stale")
    if not arm.pin_complete:
        raise NativeContractError(f"{arm.key} has unresolved package/model/source pins")
    if arm.overall_status not in ADMITTED_STATUSES:
        raise NativeContractError(
            f"{arm.key} registry status is {arm.overall_status!r}; update the reviewed dossier "
            "only after parity evidence passes"
        )
    if capability.status not in ADMITTED_STATUSES:
        raise NativeContractError(
            f"{arm.key} track {track} is {capability.status!r}: {capability.reason}"
        )
    if value.get("status") != capability.status:
        raise NativeContractError("report status does not match the admitted track status")
    if route is not None and route not in arm.adaptation_routes:
        raise NativeContractError(f"route {route!r} is not declared for {arm.key}")

    checks = normalize_check_results(value.get("checks") or {}, path=path)
    failed = [name for name, item in checks.items() if item["status"] == "fail"]
    if failed:
        raise NativeContractError(f"admission report has failed checks: {failed}")
    if require_training:
        nonpassing = [
            name for name in sorted(TRAINING_CHECKS)
            if checks[name]["status"] != "pass"
        ]
        technical_nonpassing = [
            name for name in sorted(TRAINING_CHECKS)
            if evidence_checks[name]["status"] != "pass"
        ]
        if nonpassing or technical_nonpassing:
            raise NativeContractError(
                "training requires explicit report and technical-evidence passes: "
                f"report={nonpassing}, technical={technical_nonpassing}"
            )

    request_created = _parse_utc(
        value.get("request_created_utc"), "request_created_utc"
    )
    approvals = value.get("approvals")
    _validated_approval_identities(
        approvals,
        request_created=request_created,
        approval_target_sha256=target_digest,
        trust_store=trust_store,
        minimum_approvals=minimum_approvals,
    )
    expected_finalized = max(
        _parse_utc(item.get("approved_utc"), "approval.approved_utc")
        for item in approvals
    )
    finalized = _parse_utc(value.get("finalized_utc"), "finalized_utc")
    if finalized != expected_finalized or finalized < request_created:
        raise NativeContractError(
            "admission report finalization must equal its latest authenticated approval"
        )
    if not isinstance(value.get("environment"), Mapping) or not value["environment"]:
        raise NativeContractError("admission report requires a pinned environment manifest")
    measured_environment = measure_runtime_environment(
        evidence_record["environment"], value["environment"]
    )
    if measured_environment != value["environment"]:
        raise NativeContractError("admission report environment differs from measured runtime")
    measured_lock = _verified_runtime_lock(evidence_record)
    if value.get("runtime_lock") != measured_lock:
        raise NativeContractError("admission report complete runtime lock drifted")
    actual_controls = _validated_runtime_controls(
        evidence_record["environment"], runtime_controls
    )
    if value.get("runtime_controls") != actual_controls:
        raise NativeContractError("admission report runtime controls are stale")
    if capability.status == "research_only" and value.get("use_scope") != "research_noncommercial":
        raise NativeContractError(
            "research-only admission requires environment.use_scope='research_noncommercial'"
        )
    admitted_artifacts = value.get("artifacts") or {}
    if not isinstance(admitted_artifacts, Mapping):
        raise NativeContractError("admission report artifacts must be an object")
    technical_artifacts = _technical_runtime_artifacts(arm.key, track, path)
    supplied_artifacts = dict(required_artifacts or {})
    expected_names = set(admitted_artifacts)
    provided_names = set(supplied_artifacts) | {"runner"}
    if provided_names != expected_names:
        raise NativeContractError(
            "execution artifacts must exactly match the admission report: "
            f"missing={sorted(expected_names - provided_names)}, "
            f"unknown={sorted(provided_names - expected_names)}"
        )
    actual_artifacts = {
        name: _runtime_artifact_description(name, artifact)
        for name, artifact in supplied_artifacts.items()
    }
    actual_artifacts["runner"] = _current_execution_artifact()
    for name, actual in actual_artifacts.items():
        expected = admitted_artifacts.get(name)
        if not isinstance(expected, Mapping):
            raise NativeContractError(f"admission report artifact {name!r} is invalid")
        expected_identity = dict(expected)
        # Paths may move; the exact tree/file identity may not.
        actual_identity = dict(actual)
        expected_identity.pop("path", None)
        actual_identity.pop("path", None)
        if actual_identity != expected_identity:
            raise NativeContractError(f"admission artifact {name!r} tree hash mismatch")
        if name in technical_artifacts:
            _assert_technical_artifact_match(name, actual, technical_artifacts[name])
    return value


def require_admission_from_args(
    args: Any,
    *,
    arm_key: str,
    track: str,
    route: str | None = None,
    require_training: bool = False,
    required_artifacts: Mapping[str, str | Path] | None = None,
    runtime_controls: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    report = getattr(args, "admission_report", None)
    if not report:
        raise NativeContractError(
            f"{arm_key} track {track} is blocked without --admission-report"
        )
    return verify_admission_report(
        report,
        arm_key=arm_key,
        track=track,
        route=route,
        require_training=require_training,
        required_artifacts=required_artifacts,
        runtime_controls=runtime_controls,
    )


def add_admission_argument(parser: Any) -> None:
    parser.add_argument(
        "--admission-report",
        help=(
            "Path to a current ffm_native_admission_report_v3 bound to this model, "
            "track, route, registry, and dossier. Missing or stale reports fail closed."
        ),
    )
