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
import hashlib
import json
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
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


def evidence_path_for_registry(path: str | Path) -> Path:
    """Return the evidence document colocated with a source or installed registry."""
    return Path(path).resolve().with_name("native_contract_evidence.json")


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
REPORT_SCHEMA = "ffm_native_admission_report_v2"
REGISTRY_SCHEMA = "ffm_native_contract_registry_v1"
EVIDENCE_SCHEMA = "ffm_native_contract_evidence_v1"
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


def _report_integrity_payload(report: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(report)
    value.pop("integrity", None)
    return value


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
    report_created: datetime,
    minimum_approvals: int = 2,
) -> set[str]:
    if minimum_approvals < 2:
        raise NativeContractError("admission approval floor cannot be lower than 2")
    if not isinstance(approvals, list):
        raise NativeContractError("admission approvals must be a list")
    identities: set[str] = set()
    for approval in approvals:
        if not isinstance(approval, Mapping):
            raise NativeContractError("each approval must be an object")
        reviewer = _require_string(approval.get("reviewer"), "approval.reviewer")
        approved = _parse_utc(approval.get("approved_utc"), "approval.approved_utc")
        if approved > report_created:
            raise NativeContractError(
                f"reviewer {reviewer} approval is later than report creation"
            )
        if approval.get("decision") != "approve":
            raise NativeContractError(f"reviewer {reviewer} did not approve")
        identities.add(reviewer.strip().casefold())
    if len(identities) < minimum_approvals:
        raise NativeContractError(
            f"admission requires {minimum_approvals} independent approvals, "
            f"got {len(identities)}"
        )
    return identities


def build_admission_report(
    *,
    arm_key: str,
    track: str,
    status: str,
    checks: Mapping[str, bool | Mapping[str, Any]],
    approvals: list[Mapping[str, Any]],
    environment: Mapping[str, Any],
    route: str | None = None,
    artifacts: Mapping[str, Mapping[str, Any] | str | Path] | None = None,
    created_utc: str,
    path: str | Path = REGISTRY_PATH,
) -> dict[str, Any]:
    """Build a deterministic, hash-bound report; this does not auto-admit a blocked dossier."""
    report_created = _parse_utc(created_utc, "created_utc")
    _validated_approval_identities(
        approvals, report_created=report_created, minimum_approvals=2
    )
    arm = get_arm(arm_key, path)
    if track not in {"F", "R", "C", "B", "D"}:
        raise NativeContractError(f"unknown track {track!r}")
    if status not in ADMITTED_STATUSES:
        raise NativeContractError(f"admission report status {status!r} is not admissible")
    normalized = normalize_check_results(checks, path=path)
    evidence_id, evidence_record, evidence_checks = technical_evidence(arm.key, track, path)
    verify_technical_evidence_bundle(arm.key, track, path)
    expected_evidence_status = (
        "research_only_pass" if status == "research_only" else "pass"
    )
    if evidence_record.get("status") != expected_evidence_status:
        raise NativeContractError(
            f"technical evidence {evidence_id!r} is not admissible for status {status!r}"
        )
    environment_value = dict(environment)
    evidence_environment = dict(evidence_record["environment"])
    environment_mismatches = {
        key: {"expected": expected, "actual": environment_value.get(key)}
        for key, expected in evidence_environment.items()
        if environment_value.get(key) != expected
    }
    if environment_mismatches:
        raise NativeContractError(
            "admission environment does not match technical evidence: "
            f"{environment_mismatches}"
        )
    if status == "research_only" and environment_value.get("use_scope") != "research_noncommercial":
        raise NativeContractError(
            "research-only admission requires environment.use_scope='research_noncommercial'"
        )
    normalized_artifacts: dict[str, dict[str, Any]] = {}
    for name, artifact in (artifacts or {}).items():
        _require_string(name, "artifact name")
        if isinstance(artifact, (str, Path)):
            artifact_path = Path(artifact)
            if not artifact_path.is_file():
                raise NativeContractError(f"admission artifact not found: {artifact_path}")
            normalized_artifacts[name] = {
                "path": str(artifact_path.resolve()),
                "sha256": file_sha256(artifact_path),
            }
        elif isinstance(artifact, Mapping):
            item = dict(artifact)
            digest = _require_string(item.get("sha256"), f"artifact {name}.sha256")
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest.lower()):
                raise NativeContractError(f"artifact {name}.sha256 must be a hexadecimal SHA-256")
            item["sha256"] = digest.lower()
            normalized_artifacts[name] = item
        else:
            raise NativeContractError(f"artifact {name} must be a path or object")
    value = {
        "schema_version": REPORT_SCHEMA,
        "created_utc": created_utc,
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
        "checks": normalized,
        "environment": environment_value,
        "artifacts": normalized_artifacts,
        "approvals": [dict(item) for item in approvals],
    }
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

    report_created = _parse_utc(value.get("created_utc"), "created_utc")
    approvals = value.get("approvals")
    _validated_approval_identities(
        approvals,
        report_created=report_created,
        minimum_approvals=minimum_approvals,
    )
    if not isinstance(value.get("environment"), Mapping) or not value["environment"]:
        raise NativeContractError("admission report requires a pinned environment manifest")
    environment_mismatches = {
        key: {"expected": expected, "actual": value["environment"].get(key)}
        for key, expected in evidence_record["environment"].items()
        if value["environment"].get(key) != expected
    }
    if environment_mismatches:
        raise NativeContractError(
            f"admission report environment drift: {environment_mismatches}"
        )
    if capability.status == "research_only" and value["environment"].get("use_scope") != "research_noncommercial":
        raise NativeContractError(
            "research-only admission requires environment.use_scope='research_noncommercial'"
        )
    admitted_artifacts = value.get("artifacts") or {}
    if not isinstance(admitted_artifacts, Mapping):
        raise NativeContractError("admission report artifacts must be an object")
    for name, artifact in (required_artifacts or {}).items():
        expected = admitted_artifacts.get(name)
        if not isinstance(expected, Mapping):
            raise NativeContractError(f"admission report does not bind required artifact {name!r}")
        artifact_path = Path(artifact)
        if not artifact_path.is_file():
            raise NativeContractError(f"required artifact not found: {artifact_path}")
        actual_digest = file_sha256(artifact_path)
        if expected.get("sha256") != actual_digest:
            raise NativeContractError(
                f"admission artifact {name!r} hash mismatch: expected "
                f"{expected.get('sha256')!r}, got {actual_digest!r}"
            )
    return value


def require_admission_from_args(
    args: Any,
    *,
    arm_key: str,
    track: str,
    route: str | None = None,
    require_training: bool = False,
    required_artifacts: Mapping[str, str | Path] | None = None,
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
    )


def add_admission_argument(parser: Any) -> None:
    parser.add_argument(
        "--admission-report",
        help=(
            "Path to a current ffm_native_admission_report_v2 bound to this model, "
            "track, route, registry, and dossier. Missing or stale reports fail closed."
        ),
    )
