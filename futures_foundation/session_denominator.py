"""Independent, outcome-blind verification of AlphaForge session denominators.

This module deliberately does not import AlphaForge or a market-calendar package.  FFM
accepts a denominator only when canonical, externally hash-pinned inputs prove the exact
root x session-day coverage and the UTC geometry can be reconstructed from official rule
artifacts with the pinned TZif bytes.  Calendar-library observations are diagnostics: they
may demand a sourced override, but never define session geometry.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping
from zoneinfo import TZPATH, ZoneInfo

from ._authority_bundle_io import (
    AuthorityBundleFileSizeError,
    AuthorityBundleIOError,
    read_canonical_json_file,
    sha256_regular_file,
)
from .corpus_v3 import verify_contract


RULES_SCHEMA = "alphaforge_market_calendar_rules_v2"
SCOPE_SCHEMA = "alphaforge_session_denominator_scope_v1"
DENOMINATOR_SCHEMA = "alphaforge_session_denominator_v1"
CONSUMER_SCOPE_SCHEMA = "alphaforge_denominator_consumer_scope_v1"
SCOPE_SCHEMA_V2 = "alphaforge_session_denominator_scope_v2"
DENOMINATOR_SCHEMA_V2 = "alphaforge_session_denominator_v2"
CONSUMER_SCHEMA = "ffm_corpus_v3_contract_v1"
SCOPE_PURPOSE = "session_denominator_no_outcomes"
CONSUMER_SCOPE_PURPOSE = "denominator_consumer_scope_no_market_inputs"
SCOPE_V2_PURPOSE = "session_denominator_scope_from_narrow_consumer_contract"
CONSUMER_SCOPE_BLOCKERS = (
    "detached_split_declarations_unproven",
    "production_admission_unavailable",
)
SCOPE_V2_BLOCKERS = (
    "calendar_rules_native_source_bundle_single_dirfd_unavailable",
    "narrow_consumer_scope_detached_unproven",
    "production_admission_unavailable",
)

_RULES_TOKEN = object()
_CONSUMER_TOKEN = object()
_SCOPE_TOKEN = object()
_DENOMINATOR_TOKEN = object()
_CONSUMER_SCOPE_V2_TOKEN = object()
_SCOPE_V2_TOKEN = object()
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_V2_MAX_BYTES = 64 * 1024 * 1024
_V2_MAX_NODES = 5_000_000
_V2_MAX_DEPTH = 20


class SessionDenominatorVerificationError(ValueError):
    """Raised when a denominator trust-chain or session invariant fails."""


def _hash_source_artifact(path: Path, artifact: Mapping[str, Any]) -> str:
    """Hash one declared calendar source with stable public integrity semantics."""
    try:
        _, physical = sha256_regular_file(
            path,
            label=f"calendar source artifact {artifact['path']}",
            expected_size=artifact["size"],
        )
    except AuthorityBundleFileSizeError as exc:
        raise SessionDenominatorVerificationError(
            f"source artifact bytes differ: {artifact['path']}"
        ) from exc
    except AuthorityBundleIOError as exc:
        raise SessionDenominatorVerificationError(str(exc)) from exc
    return physical


@dataclass(frozen=True)
class VerifiedCalendarRules:
    document: Mapping[str, Any]
    path: Path
    physical_sha256: str
    _token: object


@dataclass(frozen=True)
class VerifiedConsumerContract:
    document: Mapping[str, Any]
    path: Path
    physical_sha256: str
    semantic_sha256: str
    _token: object


@dataclass(frozen=True)
class VerifiedDenominatorScope:
    document: Mapping[str, Any]
    path: Path
    physical_sha256: str
    _token: object


@dataclass(frozen=True)
class VerifiedSessionDenominator:
    path: Path
    physical_sha256: str
    semantic_sha256: str
    _token: object


@dataclass(frozen=True)
class VerifiedDenominatorConsumerScope:
    document: Mapping[str, Any]
    path: Path
    physical_sha256: str
    semantic_sha256: str
    _token: object


@dataclass(frozen=True)
class VerifiedDenominatorScopeV2:
    document: Mapping[str, Any]
    raw_document: Mapping[str, Any]
    path: Path
    physical_sha256: str
    semantic_sha256: str
    rules_path: Path
    consumer_scope_path: Path
    consumer_scope_physical_sha256: str
    consumer_scope_semantic_sha256: str
    _token: object


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SessionDenominatorVerificationError("value is not canonical-JSON encodable") from exc


def content_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_json(
    path: Path, name: str, *, canonical: bool
) -> tuple[dict[str, Any], bytes]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise SessionDenominatorVerificationError(
                    f"duplicate JSON key in {name}: {key}"
                )
            result[key] = value
        return result

    if path.is_symlink() or not path.is_file():
        raise SessionDenominatorVerificationError(f"{name} is unavailable or a symlink")
    try:
        raw = path.read_bytes()
        value = json.loads(
            raw,
            object_pairs_hook=pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                SessionDenominatorVerificationError(
                    f"non-finite JSON constant in {name}: {value}"
                )
            ),
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise SessionDenominatorVerificationError(f"cannot load {name}: {path}") from exc
    if not isinstance(value, dict):
        raise SessionDenominatorVerificationError(f"{name} must be a JSON object")
    if canonical and raw != _canonical_bytes(value):
        raise SessionDenominatorVerificationError(f"{name} must be canonical JSON")
    return value, raw


def _strict_canonical_json(path: Path, name: str) -> tuple[dict[str, Any], bytes]:
    return _strict_json(path, name, canonical=True)


def _strict_canonical_json_v2(
    path: str | Path, name: str
) -> tuple[Path, dict[str, Any], str]:
    """Use the authority transport SSOT for new v2 parent capabilities."""
    try:
        return read_canonical_json_file(
            path,
            label=name,
            max_bytes=_V2_MAX_BYTES,
            max_nodes=_V2_MAX_NODES,
            max_depth=_V2_MAX_DEPTH,
        )
    except AuthorityBundleIOError as exc:
        raise SessionDenominatorVerificationError(str(exc)) from exc


def _safe_lexical_path(path: str | Path, name: str) -> Path:
    """Return an absolute path without resolving away symlink evidence."""
    candidate = Path(os.path.abspath(Path(path).expanduser()))
    for parent in reversed(candidate.parents):
        if parent == parent.parent:
            continue
        if parent.is_symlink():
            raise SessionDenominatorVerificationError(
                f"{name} parent directory is a symlink: {parent}"
            )
        if not parent.is_dir():
            raise SessionDenominatorVerificationError(
                f"{name} parent directory is unavailable: {parent}"
            )
    return candidate


def _require_expected_sha(actual: str, expected: str, name: str) -> None:
    _require_sha(expected, f"expected {name} SHA-256")
    if actual != expected:
        raise SessionDenominatorVerificationError(f"{name} physical SHA-256 mismatch")


def _require_sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise SessionDenominatorVerificationError(f"{name} must be a lowercase SHA-256")
    return value


def _exact_keys(value: Any, keys: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SessionDenominatorVerificationError(f"{name} must be an object")
    if set(value) != keys:
        raise SessionDenominatorVerificationError(
            f"{name} keys differ: missing={sorted(keys - set(value))}, "
            f"extra={sorted(set(value) - keys)}"
        )
    return value


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise SessionDenominatorVerificationError(f"{name} is not a constrained identifier")
    return value


def _day(value: Any, name: str) -> date:
    if not isinstance(value, str):
        raise SessionDenominatorVerificationError(f"{name} must be an ISO-date string")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise SessionDenominatorVerificationError(f"{name} must be an ISO date") from exc
    if parsed.isoformat() != value:
        raise SessionDenominatorVerificationError(f"{name} must be canonical YYYY-MM-DD")
    return parsed


def _source_path(base: Path, relative_text: Any, name: str) -> Path:
    if not isinstance(relative_text, str) or not relative_text:
        raise SessionDenominatorVerificationError(f"{name} must be a nonempty path")
    relative = Path(relative_text)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise SessionDenominatorVerificationError(f"{name} is unsafe")
    path = base / relative
    if path.is_symlink() or not path.is_file() or path.resolve() != Path(os.path.abspath(path)):
        raise SessionDenominatorVerificationError(f"source artifact is unavailable or unsafe: {relative}")
    return path


_DEPENDENCY_KEYS = {
    "pandas_market_calendars_version",
    "pandas_market_calendars_distribution_sha256",
    "exchange_calendars_version",
    "exchange_calendars_distribution_sha256",
    "calendar_dependency_closure_sha256",
    "environment_lock_sha256",
    "python_implementation",
    "python_version",
    "python_executable_sha256",
    "timezone_key",
    "tzif_sha256",
    "tzdata_zi_sha256",
}


def _tzif_path(key: str) -> Path:
    relative = Path(key)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise SessionDenominatorVerificationError("timezone key is unsafe")
    for directory in TZPATH:
        candidate = Path(directory) / relative
        if candidate.is_file() and not candidate.is_symlink():
            return candidate
    raise SessionDenominatorVerificationError(f"pinned TZif file is unavailable: {key}")


def _tzdata_zi_path(tzif: Path) -> Path:
    candidate = tzif.parents[1] / "tzdata.zi"
    if candidate.is_file() and not candidate.is_symlink():
        return candidate
    raise SessionDenominatorVerificationError("system tzdata.zi is unavailable")


def _verify_dependency_bytes(dependencies: Mapping[str, Any]) -> None:
    _exact_keys(dependencies, _DEPENDENCY_KEYS, "rules.dependencies")
    key = dependencies["timezone_key"]
    if not isinstance(key, str) or not key:
        raise SessionDenominatorVerificationError("timezone_key must be a nonempty string")
    _identifier(key.replace("/", ":"), "rules.dependencies.timezone_key")
    for field in (
        "pandas_market_calendars_distribution_sha256",
        "exchange_calendars_distribution_sha256",
        "calendar_dependency_closure_sha256",
        "environment_lock_sha256",
        "python_executable_sha256",
        "tzif_sha256",
        "tzdata_zi_sha256",
    ):
        _require_sha(dependencies[field], f"rules.dependencies.{field}")
    for field in (
        "pandas_market_calendars_version",
        "exchange_calendars_version",
        "python_implementation",
        "python_version",
    ):
        if not isinstance(dependencies[field], str) or not dependencies[field]:
            raise SessionDenominatorVerificationError(
                f"rules.dependencies.{field} must be a nonempty string"
            )
    tzif = _tzif_path(key)
    if sha256_file(tzif) != dependencies["tzif_sha256"]:
        raise SessionDenominatorVerificationError("pinned TZif bytes changed")
    if sha256_file(_tzdata_zi_path(tzif)) != dependencies["tzdata_zi_sha256"]:
        raise SessionDenominatorVerificationError("pinned tzdata.zi bytes changed")


def _validate_rule_segment(segment: Any, name: str) -> tuple[int, int]:
    segment = _exact_keys(
        segment,
        {"start_day_offset", "start_time_s", "end_day_offset", "end_time_s"},
        name,
    )
    for field in ("start_day_offset", "end_day_offset"):
        if type(segment[field]) is not int or segment[field] not in {-1, 0, 1}:
            raise SessionDenominatorVerificationError(f"{name}.{field} is invalid")
    for field in ("start_time_s", "end_time_s"):
        if type(segment[field]) is not int or not 0 <= segment[field] < 86_400:
            raise SessionDenominatorVerificationError(f"{name}.{field} is invalid")
    return (
        segment["start_day_offset"] * 86_400 + segment["start_time_s"],
        segment["end_day_offset"] * 86_400 + segment["end_time_s"],
    )


def _validate_segments(segments: Any, name: str, *, allow_empty: bool = False) -> None:
    if not isinstance(segments, list) or (not segments and not allow_empty):
        raise SessionDenominatorVerificationError(f"{name} must be a nonempty list")
    previous_end: int | None = None
    for index, segment in enumerate(segments):
        start, end = _validate_rule_segment(segment, f"{name}[{index}]")
        if start >= end or (previous_end is not None and start < previous_end):
            raise SessionDenominatorVerificationError(f"{name} overlaps, regresses or is empty")
        previous_end = end


def _validate_rules(document: Mapping[str, Any], base: Path) -> None:
    _exact_keys(
        document,
        {"schema_version", "coverage", "dependencies", "source_artifacts", "products", "roots", "overrides"},
        "rules",
    )
    if document["schema_version"] != RULES_SCHEMA:
        raise SessionDenominatorVerificationError("unsupported calendar-rules schema")
    coverage = _exact_keys(document["coverage"], {"start", "end_exclusive"}, "rules.coverage")
    coverage_start = _day(coverage["start"], "rules.coverage.start")
    coverage_end = _day(coverage["end_exclusive"], "rules.coverage.end_exclusive")
    if coverage_start >= coverage_end:
        raise SessionDenominatorVerificationError("calendar-rules coverage is empty")
    _verify_dependency_bytes(document["dependencies"])

    artifacts = document["source_artifacts"]
    if not isinstance(artifacts, list) or not artifacts:
        raise SessionDenominatorVerificationError("rules require source artifacts")
    source_ids: set[str] = set()
    source_paths: set[str] = set()
    source_hashes: set[str] = set()
    stable_identifiers: set[str] = set()
    for index, artifact in enumerate(artifacts):
        artifact = _exact_keys(
            artifact,
            {"source_id", "path", "artifact_type", "stable_identifier", "size", "sha256"},
            f"rules.source_artifacts[{index}]",
        )
        source_id = _identifier(artifact["source_id"], f"source_artifacts[{index}].source_id")
        artifact_type = _identifier(artifact["artifact_type"], f"source_artifacts[{index}].artifact_type")
        if not artifact_type.startswith("official_"):
            raise SessionDenominatorVerificationError("calendar rule source is not declared official")
        if not isinstance(artifact["stable_identifier"], str) or not artifact["stable_identifier"]:
            raise SessionDenominatorVerificationError("source stable_identifier is empty")
        if type(artifact["size"]) is not int or artifact["size"] < 0:
            raise SessionDenominatorVerificationError("source artifact size is invalid")
        artifact_hash = _require_sha(artifact["sha256"], "source artifact SHA-256")
        path = _source_path(base, artifact["path"], "source artifact path")
        artifact_physical = _hash_source_artifact(path, artifact)
        if artifact_physical != artifact["sha256"]:
            raise SessionDenominatorVerificationError(
                f"source artifact bytes differ: {artifact['path']}"
            )
        if (
            source_id in source_ids
            or artifact["path"] in source_paths
            or artifact_hash in source_hashes
            or artifact["stable_identifier"] in stable_identifiers
        ):
            raise SessionDenominatorVerificationError(
                "source artifact IDs, paths, hashes and stable identifiers must be unique"
            )
        source_ids.add(source_id)
        source_paths.add(artifact["path"])
        source_hashes.add(artifact_hash)
        stable_identifiers.add(artifact["stable_identifier"])

    products = document["products"]
    roots = document["roots"]
    if not isinstance(products, dict) or not products or not isinstance(roots, dict) or not roots:
        raise SessionDenominatorVerificationError("rules require products and roots")
    global_rule_ids: set[str] = set()
    for product, spec_value in products.items():
        _identifier(product, "product ID")
        spec = _exact_keys(
            spec_value,
            {"exchange_calendar", "open_weekdays", "weekday_source_id", "rules"},
            f"rules.products.{product}",
        )
        _identifier(spec["exchange_calendar"], f"products.{product}.exchange_calendar")
        if spec["weekday_source_id"] not in source_ids:
            raise SessionDenominatorVerificationError("weekday source ID is unknown")
        weekdays = spec["open_weekdays"]
        if (
            not isinstance(weekdays, list)
            or weekdays != sorted(set(weekdays))
            or not weekdays
            or any(type(value) is not int or not 0 <= value <= 6 for value in weekdays)
        ):
            raise SessionDenominatorVerificationError("product open weekdays are invalid")
        rules = spec["rules"]
        if not isinstance(rules, list) or not rules:
            raise SessionDenominatorVerificationError("product has no effective rules")
        previous_end: date | None = None
        for index, rule_value in enumerate(rules):
            rule = _exact_keys(
                rule_value,
                {"rule_id", "effective_start", "effective_end_exclusive", "segments", "source_id"},
                f"products.{product}.rules[{index}]",
            )
            rule_id = _identifier(rule["rule_id"], "rule_id")
            if rule_id in global_rule_ids:
                raise SessionDenominatorVerificationError("rule IDs must be globally unique")
            global_rule_ids.add(rule_id)
            start = _day(rule["effective_start"], "rule effective_start")
            end = _day(rule["effective_end_exclusive"], "rule effective_end_exclusive")
            if start >= end or (previous_end is not None and start < previous_end):
                raise SessionDenominatorVerificationError("product rules overlap or are empty")
            previous_end = end
            if rule["source_id"] not in source_ids:
                raise SessionDenominatorVerificationError("rule source ID is unknown")
            _validate_segments(rule["segments"], "rule segments")

    for root, spec_value in roots.items():
        _identifier(root, "root ID")
        spec = _exact_keys(
            spec_value,
            {"product", "effective_start", "effective_end_exclusive", "source_id"},
            f"rules.roots.{root}",
        )
        if spec["product"] not in products or spec["source_id"] not in source_ids:
            raise SessionDenominatorVerificationError("root references an unknown product or source")
        start = _day(spec["effective_start"], "root effective_start")
        end = _day(spec["effective_end_exclusive"], "root effective_end_exclusive")
        if not coverage_start <= start < end <= coverage_end:
            raise SessionDenominatorVerificationError("root effective interval is invalid")
        cursor = start
        for rule in products[spec["product"]]["rules"]:
            rule_start = _day(rule["effective_start"], "rule effective_start")
            rule_end = _day(rule["effective_end_exclusive"], "rule effective_end_exclusive")
            if rule_end <= cursor or rule_start >= end:
                continue
            if rule_start > cursor:
                raise SessionDenominatorVerificationError("root has an uncovered rule interval")
            cursor = min(end, max(cursor, rule_end))
        if cursor != end:
            raise SessionDenominatorVerificationError("root has an uncovered rule interval")

    overrides = document["overrides"]
    if not isinstance(overrides, list):
        raise SessionDenominatorVerificationError("rules.overrides must be a list")
    override_ids: set[str] = set()
    affected: set[tuple[str, str]] = set()
    for index, override_value in enumerate(overrides):
        override = _exact_keys(
            override_value,
            {"override_id", "session_day", "products", "roots", "status", "segments", "source_id"},
            f"rules.overrides[{index}]",
        )
        override_id = _identifier(override["override_id"], "override_id")
        if override_id in override_ids:
            raise SessionDenominatorVerificationError("override IDs must be unique")
        override_ids.add(override_id)
        override_day = _day(override["session_day"], "override session_day")
        if not coverage_start <= override_day < coverage_end:
            raise SessionDenominatorVerificationError("override is outside calendar coverage")
        for field, known in (("products", products), ("roots", roots)):
            values = override[field]
            if not isinstance(values, list) or values != sorted(set(values)):
                raise SessionDenominatorVerificationError(f"override {field} must be sorted and unique")
            if any(value not in known for value in values):
                raise SessionDenominatorVerificationError(f"override contains an unknown {field[:-1]}")
        if not override["products"] and not override["roots"]:
            raise SessionDenominatorVerificationError("override has no scope")
        if override["source_id"] not in source_ids:
            raise SessionDenominatorVerificationError("override source ID is unknown")
        if override["status"] not in {"open", "closed"}:
            raise SessionDenominatorVerificationError("override status is invalid")
        if override["status"] == "closed":
            if override["segments"] != []:
                raise SessionDenominatorVerificationError("closed override must have no segments")
        else:
            _validate_segments(override["segments"], "override segments")
        scoped_roots = {
            root
            for root, root_spec in roots.items()
            if root in override["roots"] or root_spec["product"] in override["products"]
        }
        for root in scoped_roots:
            root_spec = roots[root]
            if root in override["roots"] and not (
                _day(root_spec["effective_start"], "root effective_start")
                <= override_day
                < _day(root_spec["effective_end_exclusive"], "root effective_end_exclusive")
            ):
                raise SessionDenominatorVerificationError("override is outside a root active interval")
            key = (root, override_day.isoformat())
            if key in affected:
                raise SessionDenominatorVerificationError("overlapping override scopes")
            affected.add(key)


def _verify_source_artifacts(rules: VerifiedCalendarRules) -> None:
    for artifact in rules.document["source_artifacts"]:
        path = _source_path(rules.path.parent, artifact["path"], "source artifact path")
        physical = _hash_source_artifact(path, artifact)
        if physical != artifact["sha256"]:
            raise SessionDenominatorVerificationError(
                f"source artifact bytes differ: {artifact['path']}"
            )


def load_calendar_rules(path: str | Path, *, expected_sha256: str) -> VerifiedCalendarRules:
    source, document, physical = _strict_canonical_json_v2(path, "calendar rules")
    _require_expected_sha(physical, expected_sha256, "calendar rules")
    _validate_rules(document, source.parent)
    return VerifiedCalendarRules(document, source, physical, _RULES_TOKEN)


def load_consumer_contract(
    path: str | Path, *, expected_sha256: str
) -> VerifiedConsumerContract:
    source = _safe_lexical_path(path, "FFM consumer contract")
    # The checked-in FFM contract is intentionally pretty-printed and its exact bytes are
    # already bound by producer receipts.  Reject ambiguous JSON, but do not rewrite it or
    # substitute a canonical-content hash for that physical identity.
    document, raw = _strict_json(source, "FFM consumer contract", canonical=False)
    physical = hashlib.sha256(raw).hexdigest()
    _require_expected_sha(physical, expected_sha256, "FFM consumer contract")
    try:
        verify_contract(document, verify_artifacts=False)
    except Exception as exc:
        raise SessionDenominatorVerificationError("FFM consumer contract is invalid") from exc
    if document.get("schema_version") != CONSUMER_SCHEMA:
        raise SessionDenominatorVerificationError("unsupported FFM consumer-contract schema")
    return VerifiedConsumerContract(
        document, source, physical, content_sha256(document), _CONSUMER_TOKEN
    )


def load_denominator_consumer_scope_v1(
    path: str | Path, *, expected_sha256: str
) -> VerifiedDenominatorConsumerScope:
    """Verify AlphaForge's narrow, outcome-blind denominator consumer contract.

    This capability is deliberately production-blocked.  It exists only to let FFM
    independently consume the exact synthetic mechanism emitted by AlphaForge.
    """
    source, document, physical = _strict_canonical_json_v2(
        path, "denominator consumer scope"
    )
    _require_expected_sha(physical, expected_sha256, "denominator consumer scope")
    _exact_keys(
        document,
        {
            "schema_version", "purpose", "production_admission",
            "admission_blockers", "admitted_roots", "splits",
            "reserved_oos_excluded", "consumer_scope_semantic_sha256",
        },
        "denominator consumer scope",
    )
    if (
        document["schema_version"] != CONSUMER_SCOPE_SCHEMA
        or document["purpose"] != CONSUMER_SCOPE_PURPOSE
        or document["production_admission"] is not False
        or document["admission_blockers"] != list(CONSUMER_SCOPE_BLOCKERS)
        or document["reserved_oos_excluded"] is not True
    ):
        raise SessionDenominatorVerificationError(
            "denominator consumer scope must remain narrow and production-blocked"
        )
    roots = document["admitted_roots"]
    if (
        not isinstance(roots, list)
        or not roots
        or any(not isinstance(root, str) or _IDENTIFIER_RE.fullmatch(root) is None for root in roots)
        or roots != sorted(set(roots))
    ):
        raise SessionDenominatorVerificationError(
            "denominator consumer roots must be sorted unique identifiers"
        )
    splits = document["splits"]
    if not isinstance(splits, list) or len(splits) < 2:
        raise SessionDenominatorVerificationError(
            "denominator consumer scope requires usable and holdout splits"
        )
    normalized: list[tuple[str, date, date, tuple[str, ...]]] = []
    for index, value in enumerate(splits):
        split = _exact_keys(
            value,
            {"partition_id", "start", "end_exclusive", "permitted_uses"},
            f"denominator consumer split {index}",
        )
        partition = _identifier(
            split["partition_id"], f"denominator consumer split {index}.partition_id"
        )
        start = _day(split["start"], f"denominator consumer split {index}.start")
        end = _day(
            split["end_exclusive"],
            f"denominator consumer split {index}.end_exclusive",
        )
        uses = split["permitted_uses"]
        if not isinstance(uses, list) or any(
            not isinstance(use, str) or _IDENTIFIER_RE.fullmatch(use) is None
            for use in uses
        ):
            raise SessionDenominatorVerificationError(
                f"denominator consumer split {index} is invalid"
            )
        if start >= end or len(uses) != len(set(uses)):
            raise SessionDenominatorVerificationError(
                f"denominator consumer split {index} is invalid"
            )
        normalized.append((partition, start, end, tuple(uses)))
    if len({row[0] for row in normalized}) != len(normalized):
        raise SessionDenominatorVerificationError(
            "denominator consumer partition IDs must be unique"
        )
    if normalized != sorted(normalized, key=lambda row: (row[1], row[2], row[0])):
        raise SessionDenominatorVerificationError(
            "denominator consumer splits must be chronologically sorted"
        )
    for previous, current in zip(normalized, normalized[1:]):
        if previous[2] != current[1]:
            raise SessionDenominatorVerificationError(
                "denominator consumer splits must be contiguous and nonoverlapping"
            )
    holdout = [row for row in normalized if row[0] == "legacy_holdout"]
    if (
        len(holdout) != 1
        or holdout[0] != normalized[-1]
        or holdout[0][3]
        or any(not row[3] for row in normalized[:-1])
    ):
        raise SessionDenominatorVerificationError(
            "denominator consumer scope requires one final zero-use legacy_holdout"
        )
    semantic = _require_sha(
        document["consumer_scope_semantic_sha256"],
        "denominator consumer scope semantic SHA-256",
    )
    payload = dict(document)
    payload.pop("consumer_scope_semantic_sha256")
    if semantic != content_sha256(payload):
        raise SessionDenominatorVerificationError(
            "denominator consumer scope semantic SHA-256 mismatch"
        )
    return VerifiedDenominatorConsumerScope(
        document, source, physical, semantic, _CONSUMER_SCOPE_V2_TOKEN
    )


def denominator_consumer_scope_document(
    capability: VerifiedDenominatorConsumerScope,
) -> Mapping[str, Any]:
    if (
        type(capability) is not VerifiedDenominatorConsumerScope
        or capability._token is not _CONSUMER_SCOPE_V2_TOKEN
    ):
        raise SessionDenominatorVerificationError(
            "a verified denominator-consumer-scope capability is required"
        )
    reopened = load_denominator_consumer_scope_v1(
        capability.path, expected_sha256=capability.physical_sha256
    )
    if reopened != capability:
        raise SessionDenominatorVerificationError(
            "denominator consumer scope changed on reopen"
        )
    return reopened.document


def load_denominator_scope_v2(
    path: str | Path,
    *,
    expected_sha256: str,
    rules: VerifiedCalendarRules,
    consumer_scope: VerifiedDenominatorConsumerScope,
) -> VerifiedDenominatorScopeV2:
    source, document, physical = _strict_canonical_json_v2(
        path, "denominator scope v2"
    )
    _require_expected_sha(physical, expected_sha256, "denominator scope v2")
    consumer_document = denominator_consumer_scope_document(consumer_scope)
    rules_document = _rules_document(rules)
    _exact_keys(
        document,
        {
            "schema_version", "purpose", "production_admission",
            "admission_blockers", "parent_calendar_rules",
            "parent_consumer_scope", "scope_semantic_sha256",
        },
        "denominator scope v2",
    )
    if (
        document["schema_version"] != SCOPE_SCHEMA_V2
        or document["purpose"] != SCOPE_V2_PURPOSE
        or document["production_admission"] is not False
        or document["admission_blockers"] != list(SCOPE_V2_BLOCKERS)
    ):
        raise SessionDenominatorVerificationError(
            "denominator scope v2 must remain production-blocked"
        )
    expected_rules_ref = {
        "path": str(rules.path),
        "physical_sha256": rules.physical_sha256,
        "semantic_sha256": rules.physical_sha256,
    }
    expected_consumer_ref = {
        "path": str(consumer_scope.path),
        "physical_sha256": consumer_scope.physical_sha256,
        "semantic_sha256": consumer_scope.semantic_sha256,
    }
    if document["parent_calendar_rules"] != expected_rules_ref:
        raise SessionDenominatorVerificationError(
            "denominator scope v2 calendar-rules binding mismatch"
        )
    if document["parent_consumer_scope"] != expected_consumer_ref:
        raise SessionDenominatorVerificationError(
            "denominator scope v2 consumer-scope binding mismatch"
        )
    semantic = _require_sha(
        document["scope_semantic_sha256"], "denominator scope v2 semantic SHA-256"
    )
    payload = dict(document)
    payload.pop("scope_semantic_sha256")
    if semantic != content_sha256(payload):
        raise SessionDenominatorVerificationError(
            "denominator scope v2 semantic SHA-256 mismatch"
        )
    usable = [split for split in consumer_document["splits"] if split["permitted_uses"]]
    holdout = consumer_document["splits"][-1]
    normalized = {
        "schema_version": SCOPE_SCHEMA_V2,
        "purpose": SCOPE_PURPOSE,
        "calendar_rules_sha256": rules.physical_sha256,
        "split_uses": [split["partition_id"] for split in usable],
        "roots": list(consumer_document["admitted_roots"]),
        "start": usable[0]["start"],
        "end_exclusive": holdout["start"],
        "reserved_oos_excluded": True,
    }
    coverage = rules_document["coverage"]
    if not (
        _day(coverage["start"], "rules coverage start")
        <= _day(normalized["start"], "scope v2 start")
        < _day(normalized["end_exclusive"], "scope v2 end")
        <= _day(coverage["end_exclusive"], "rules coverage end")
    ):
        raise SessionDenominatorVerificationError(
            "denominator scope v2 is outside calendar coverage"
        )
    if sorted(normalized["roots"]) != sorted(rules_document["roots"]):
        raise SessionDenominatorVerificationError(
            "denominator scope v2 does not conserve full root authority"
        )
    return VerifiedDenominatorScopeV2(
        normalized,
        document,
        source,
        physical,
        semantic,
        rules.path,
        consumer_scope.path,
        consumer_scope.physical_sha256,
        consumer_scope.semantic_sha256,
        _SCOPE_V2_TOKEN,
    )


def denominator_scope_v2_document(
    capability: VerifiedDenominatorScopeV2,
    *,
    rules: VerifiedCalendarRules,
    consumer_scope: VerifiedDenominatorConsumerScope,
) -> Mapping[str, Any]:
    if (
        type(capability) is not VerifiedDenominatorScopeV2
        or capability._token is not _SCOPE_V2_TOKEN
    ):
        raise SessionDenominatorVerificationError(
            "a verified denominator-scope-v2 capability is required"
        )
    reopened = load_denominator_scope_v2(
        capability.path,
        expected_sha256=capability.physical_sha256,
        rules=rules,
        consumer_scope=consumer_scope,
    )
    if reopened != capability:
        raise SessionDenominatorVerificationError(
            "denominator scope v2 changed on reopen"
        )
    return reopened.document


def _rules_document(rules: VerifiedCalendarRules) -> Mapping[str, Any]:
    if type(rules) is not VerifiedCalendarRules or rules._token is not _RULES_TOKEN:
        raise SessionDenominatorVerificationError("a verified calendar-rules capability is required")
    if sha256_file(rules.path) != rules.physical_sha256 or content_sha256(rules.document) != rules.physical_sha256:
        raise SessionDenominatorVerificationError("verified calendar rules changed")
    _verify_dependency_bytes(rules.document["dependencies"])
    _verify_source_artifacts(rules)
    return rules.document


def _consumer_document(contract: VerifiedConsumerContract) -> Mapping[str, Any]:
    if type(contract) is not VerifiedConsumerContract or contract._token is not _CONSUMER_TOKEN:
        raise SessionDenominatorVerificationError("a verified FFM consumer capability is required")
    if (
        sha256_file(contract.path) != contract.physical_sha256
        or content_sha256(contract.document) != contract.semantic_sha256
    ):
        raise SessionDenominatorVerificationError("verified FFM consumer contract changed")
    return contract.document


_SCOPE_KEYS = {
    "schema_version",
    "consumer_contract_sha256",
    "purpose",
    "calendar_rules_sha256",
    "split_uses",
    "roots",
    "start",
    "end_exclusive",
    "reserved_oos_excluded",
}


def load_denominator_scope(
    path: str | Path,
    *,
    expected_sha256: str,
    rules: VerifiedCalendarRules,
    consumer: VerifiedConsumerContract,
) -> VerifiedDenominatorScope:
    rules_document = _rules_document(rules)
    consumer_document = _consumer_document(consumer)
    source = _safe_lexical_path(path, "denominator scope")
    document, raw = _strict_canonical_json(source, "denominator scope")
    physical = hashlib.sha256(raw).hexdigest()
    _require_expected_sha(physical, expected_sha256, "denominator scope")
    _exact_keys(document, _SCOPE_KEYS, "denominator scope")
    if document["schema_version"] != SCOPE_SCHEMA or document["purpose"] != SCOPE_PURPOSE:
        raise SessionDenominatorVerificationError("unsupported denominator scope or purpose")
    if document["consumer_contract_sha256"] != consumer.physical_sha256:
        raise SessionDenominatorVerificationError("scope consumer-contract hash mismatch")
    if document["calendar_rules_sha256"] != rules.physical_sha256:
        raise SessionDenominatorVerificationError("scope calendar-rules hash mismatch")
    if document["reserved_oos_excluded"] is not True:
        raise SessionDenominatorVerificationError("scope must exclude reserved OOS")
    roots = document["roots"]
    if not isinstance(roots, list) or not roots or roots != sorted(set(roots)):
        raise SessionDenominatorVerificationError("scope roots must be sorted and unique")
    if roots != consumer_document["admitted_roots"]:
        raise SessionDenominatorVerificationError("scope must contain every admitted root exactly")
    if any(root not in rules_document["roots"] for root in roots):
        raise SessionDenominatorVerificationError("scope contains a root absent from calendar rules")

    split_uses = document["split_uses"]
    if (
        not isinstance(split_uses, list)
        or not split_uses
        or split_uses != sorted(set(split_uses))
        or "legacy_holdout_excluded" in split_uses
    ):
        raise SessionDenominatorVerificationError("scope split uses are invalid")
    splits = consumer_document["splits"]
    intervals: list[tuple[date, date]] = []
    for split_name in split_uses:
        split = splits.get(split_name)
        if not isinstance(split, Mapping) or split.get("use") not in {
            "training_only",
            "validation_model_selection",
        }:
            raise SessionDenominatorVerificationError(f"split is not denominator-admitted: {split_name}")
        start = _day(split.get("start"), f"splits.{split_name}.start")
        end = _day(split.get("end_exclusive"), f"splits.{split_name}.end_exclusive")
        if start >= end:
            raise SessionDenominatorVerificationError("scope includes an empty split")
        intervals.append((start, end))
    intervals.sort()
    merged_start, merged_end = intervals[0]
    for interval_start, interval_end in intervals[1:]:
        if interval_start > merged_end:
            raise SessionDenominatorVerificationError("selected splits contain a date gap")
        merged_end = max(merged_end, interval_end)
    start = _day(document["start"], "scope.start")
    end = _day(document["end_exclusive"], "scope.end_exclusive")
    if (start, end) != (merged_start, merged_end):
        raise SessionDenominatorVerificationError("scope does not exactly cover selected splits")
    holdout_start = _day(
        consumer_document["splits"]["legacy_holdout_excluded"]["start"],
        "legacy holdout start",
    )
    if end > holdout_start:
        raise SessionDenominatorVerificationError("scope overlaps reserved OOS")
    coverage = rules_document["coverage"]
    if not _day(coverage["start"], "coverage.start") <= start < end <= _day(
        coverage["end_exclusive"], "coverage.end_exclusive"
    ):
        raise SessionDenominatorVerificationError("scope is outside calendar-rule coverage")
    return VerifiedDenominatorScope(document, source, physical, _SCOPE_TOKEN)


def _scope_document(
    scope: VerifiedDenominatorScope,
    rules: VerifiedCalendarRules,
    consumer: VerifiedConsumerContract,
) -> Mapping[str, Any]:
    if type(scope) is not VerifiedDenominatorScope or scope._token is not _SCOPE_TOKEN:
        raise SessionDenominatorVerificationError("a verified denominator-scope capability is required")
    if sha256_file(scope.path) != scope.physical_sha256 or content_sha256(scope.document) != scope.physical_sha256:
        raise SessionDenominatorVerificationError("verified denominator scope changed")
    if scope.document["calendar_rules_sha256"] != rules.physical_sha256:
        raise SessionDenominatorVerificationError("scope no longer matches calendar rules")
    if scope.document["consumer_contract_sha256"] != consumer.physical_sha256:
        raise SessionDenominatorVerificationError("scope no longer matches consumer contract")
    return scope.document


def _verified_timezone(dependencies: Mapping[str, Any]) -> ZoneInfo:
    _verify_dependency_bytes(dependencies)
    path = _tzif_path(dependencies["timezone_key"])
    with path.open("rb") as handle:
        return ZoneInfo.from_file(handle, key=dependencies["timezone_key"])


def _aware_local(day: date, offset: int, seconds: int, tz: ZoneInfo) -> datetime:
    target = day + timedelta(days=offset)
    naive = datetime(target.year, target.month, target.day) + timedelta(seconds=seconds)
    first = naive.replace(tzinfo=tz, fold=0)
    second = naive.replace(tzinfo=tz, fold=1)
    if first.utcoffset() != second.utcoffset():
        raise SessionDenominatorVerificationError(f"ambiguous local boundary: {naive}")
    if first.astimezone(timezone.utc).astimezone(tz).replace(tzinfo=None) != naive:
        raise SessionDenominatorVerificationError(f"nonexistent local boundary: {naive}")
    return first


def _resolve_segments(segments: Iterable[Mapping[str, Any]], day: date, tz: ZoneInfo) -> list[list[int]]:
    result: list[list[int]] = []
    previous_end: int | None = None
    for segment in segments:
        start = _aware_local(day, segment["start_day_offset"], segment["start_time_s"], tz)
        end = _aware_local(day, segment["end_day_offset"], segment["end_time_s"], tz)
        pair = [
            int(start.astimezone(timezone.utc).timestamp() * 1_000_000_000),
            int(end.astimezone(timezone.utc).timestamp() * 1_000_000_000),
        ]
        if pair[0] >= pair[1] or (previous_end is not None and pair[0] < previous_end):
            raise SessionDenominatorVerificationError("resolved segments overlap, regress or are empty")
        result.append(pair)
        previous_end = pair[1]
    if not result:
        raise SessionDenominatorVerificationError("open session has no resolved segments")
    return result


def _rule_for(product: Mapping[str, Any], day: date) -> Mapping[str, Any]:
    matches = [
        rule
        for rule in product["rules"]
        if _day(rule["effective_start"], "rule effective_start")
        <= day
        < _day(rule["effective_end_exclusive"], "rule effective_end_exclusive")
    ]
    if len(matches) != 1:
        raise SessionDenominatorVerificationError("active root does not have exactly one rule")
    return matches[0]


def _override_for(
    rules: Mapping[str, Any], root: str, product: str, day: date
) -> Mapping[str, Any] | None:
    matches = [
        override
        for override in rules["overrides"]
        if override["session_day"] == day.isoformat()
        and (root in override["roots"] or product in override["products"])
    ]
    if len(matches) > 1:
        raise SessionDenominatorVerificationError("multiple overrides match a root/session")
    return matches[0] if matches else None


def _segments_subset(candidate: list[list[int]], container: list[list[int]]) -> bool:
    index = 0
    for start, end in candidate:
        while index < len(container) and container[index][1] <= start:
            index += 1
        if index == len(container) or start < container[index][0] or end > container[index][1]:
            return False
    return True


_ROW_KEYS = {
    "root",
    "product",
    "session_day",
    "root_source_id",
    "weekday_source_id",
    "status",
    "segments_utc_ns",
    "rule_id",
    "rule_source_id",
    "override_id",
    "override_source_id",
    "calendar_observation",
    "calendar_exception_types",
    "segment_semantic_sha256",
}


def _expected_row(
    rules: Mapping[str, Any], row: Mapping[str, Any], tz: ZoneInfo
) -> dict[str, Any]:
    _exact_keys(row, _ROW_KEYS, "denominator row")
    root = row["root"]
    if root not in rules["roots"]:
        raise SessionDenominatorVerificationError("denominator row has an unknown root")
    day = _day(row["session_day"], "row.session_day")
    root_spec = rules["roots"][root]
    product_name = root_spec["product"]
    product = rules["products"][product_name]
    base = {
        "root": root,
        "product": product_name,
        "session_day": day.isoformat(),
        "root_source_id": root_spec["source_id"],
        "weekday_source_id": product["weekday_source_id"],
    }
    observation = row["calendar_observation"]
    exceptions = row["calendar_exception_types"]
    if observation not in {None, "open", "closed"}:
        raise SessionDenominatorVerificationError("calendar observation is invalid")
    if (
        not isinstance(exceptions, list)
        or exceptions != sorted(set(exceptions))
        or any(value not in {"early_close", "late_open"} for value in exceptions)
    ):
        raise SessionDenominatorVerificationError("calendar exception flags are invalid")
    effective_start = _day(root_spec["effective_start"], "root effective_start")
    effective_end = _day(root_spec["effective_end_exclusive"], "root effective_end_exclusive")
    if day < effective_start or day >= effective_end:
        expected_status = "prelisting" if day < effective_start else "delisted"
        if observation is not None or exceptions:
            raise SessionDenominatorVerificationError("inactive-root observation flags must be empty")
        return {
            **base,
            "status": expected_status,
            "segments_utc_ns": [],
            "rule_id": None,
            "rule_source_id": None,
            "override_id": None,
            "override_source_id": None,
            "calendar_observation": None,
            "calendar_exception_types": [],
            "segment_semantic_sha256": content_sha256([]),
        }

    if observation not in {"open", "closed"}:
        raise SessionDenominatorVerificationError(
            "active-root calendar observation must be open or closed"
        )

    rule = _rule_for(product, day)
    normal = _resolve_segments(rule["segments"], day, tz)
    override = _override_for(rules, root, product_name, day)
    official_weekday_open = day.weekday() in product["open_weekdays"]
    if override is None:
        if exceptions:
            raise SessionDenominatorVerificationError("calendar exception lacks a source-backed override")
        if official_weekday_open and observation == "closed":
            raise SessionDenominatorVerificationError("weekday closure lacks a source-backed override")
        if not official_weekday_open and observation == "open":
            raise SessionDenominatorVerificationError("closed-weekday opening lacks a source-backed override")
        selected = normal if official_weekday_open else []
        status = "regular" if official_weekday_open else "closed"
    elif override["status"] == "closed":
        selected = []
        status = "closed"
    else:
        selected = _resolve_segments(override["segments"], day, tz)
        if selected == normal and observation == "open" and not exceptions:
            raise SessionDenominatorVerificationError(
                "stale open override duplicates a normal session"
            )
        if selected == normal:
            status = "regular"
        else:
            selected_duration = sum(end - start for start, end in selected)
            normal_duration = sum(end - start for start, end in normal)
            if _segments_subset(selected, normal) and selected_duration < normal_duration:
                status = "shortened"
            elif _segments_subset(normal, selected) and selected_duration > normal_duration:
                status = "extended"
            else:
                status = "irregular"
    expected = {
        **base,
        "status": status,
        "segments_utc_ns": selected,
        "rule_id": rule["rule_id"],
        "rule_source_id": rule["source_id"],
        "override_id": override["override_id"] if override is not None else None,
        "override_source_id": override["source_id"] if override is not None else None,
        "calendar_observation": observation,
        "calendar_exception_types": exceptions,
        "segment_semantic_sha256": content_sha256(selected),
    }
    return expected


def _scope_keys(scope: Mapping[str, Any]) -> list[tuple[str, str]]:
    start = _day(scope["start"], "scope.start")
    end = _day(scope["end_exclusive"], "scope.end_exclusive")
    days = [
        (start + timedelta(days=offset)).isoformat()
        for offset in range((end - start).days)
    ]
    return [(root, day) for root in scope["roots"] for day in days]


_DENOMINATOR_KEYS = {
    "schema_version",
    "calendar_rules_sha256",
    "denominator_scope_sha256",
    "dependencies",
    "row_count",
    "rows",
    "denominator_semantic_sha256",
}


def verify_session_denominator(
    artifact: Mapping[str, Any],
    *,
    rules: VerifiedCalendarRules,
    scope: VerifiedDenominatorScope,
    consumer: VerifiedConsumerContract,
) -> str:
    rules_document = _rules_document(rules)
    _consumer_document(consumer)
    scope_document = _scope_document(scope, rules, consumer)
    _exact_keys(artifact, _DENOMINATOR_KEYS, "session denominator")
    if artifact["schema_version"] != DENOMINATOR_SCHEMA:
        raise SessionDenominatorVerificationError("unsupported denominator schema")
    if artifact["calendar_rules_sha256"] != rules.physical_sha256:
        raise SessionDenominatorVerificationError("denominator rules hash mismatch")
    if artifact["denominator_scope_sha256"] != scope.physical_sha256:
        raise SessionDenominatorVerificationError("denominator scope hash mismatch")
    if artifact["dependencies"] != rules_document["dependencies"]:
        raise SessionDenominatorVerificationError("denominator dependency identity mismatch")
    payload = dict(artifact)
    supplied_semantic = _require_sha(
        payload.pop("denominator_semantic_sha256"), "denominator semantic SHA-256"
    )
    if content_sha256(payload) != supplied_semantic:
        raise SessionDenominatorVerificationError("denominator semantic SHA-256 mismatch")
    rows = artifact["rows"]
    if not isinstance(rows, list) or type(artifact["row_count"]) is not int:
        raise SessionDenominatorVerificationError("denominator row/count types are invalid")
    actual_keys = [
        (row.get("root"), row.get("session_day")) if isinstance(row, Mapping) else (None, None)
        for row in rows
    ]
    if artifact["row_count"] != len(rows) or actual_keys != _scope_keys(scope_document):
        raise SessionDenominatorVerificationError(
            "denominator does not exactly cover admitted root x selected non-OOS dates"
        )
    tz = _verified_timezone(rules_document["dependencies"])
    for row in rows:
        expected = _expected_row(rules_document, row, tz)
        if expected != row:
            raise SessionDenominatorVerificationError(
                f"denominator row differs from official rules: {row.get('root')}/{row.get('session_day')}"
            )
    return supplied_semantic


def load_and_verify_session_denominator(
    path: str | Path,
    *,
    expected_sha256: str,
    rules: VerifiedCalendarRules,
    scope: VerifiedDenominatorScope,
    consumer: VerifiedConsumerContract,
) -> VerifiedSessionDenominator:
    source = _safe_lexical_path(path, "session denominator")
    document, raw = _strict_canonical_json(source, "session denominator")
    physical = hashlib.sha256(raw).hexdigest()
    _require_expected_sha(physical, expected_sha256, "session denominator")
    semantic = verify_session_denominator(
        document, rules=rules, scope=scope, consumer=consumer
    )
    return VerifiedSessionDenominator(
        source, physical, semantic, _DENOMINATOR_TOKEN
    )


def session_denominator_document(
    denominator: VerifiedSessionDenominator,
    *,
    rules: VerifiedCalendarRules,
    scope: VerifiedDenominatorScope,
    consumer: VerifiedConsumerContract,
) -> Mapping[str, Any]:
    """Reopen and reverify a denominator at its point of consumption."""
    if (
        type(denominator) is not VerifiedSessionDenominator
        or denominator._token is not _DENOMINATOR_TOKEN
    ):
        raise SessionDenominatorVerificationError(
            "a verified session-denominator capability is required"
        )
    source = _safe_lexical_path(denominator.path, "session denominator")
    if sha256_file(source) != denominator.physical_sha256:
        raise SessionDenominatorVerificationError(
            "verified session-denominator bytes changed"
        )
    document, _ = _strict_canonical_json(source, "session denominator")
    semantic = verify_session_denominator(
        document, rules=rules, scope=scope, consumer=consumer
    )
    if semantic != denominator.semantic_sha256:
        raise SessionDenominatorVerificationError(
            "verified session-denominator semantics changed"
        )
    return document


__all__ = [
    "CONSUMER_SCOPE_BLOCKERS",
    "CONSUMER_SCOPE_PURPOSE",
    "CONSUMER_SCOPE_SCHEMA",
    "CONSUMER_SCHEMA",
    "DENOMINATOR_SCHEMA",
    "DENOMINATOR_SCHEMA_V2",
    "RULES_SCHEMA",
    "SCOPE_PURPOSE",
    "SCOPE_SCHEMA",
    "SCOPE_SCHEMA_V2",
    "SCOPE_V2_BLOCKERS",
    "SCOPE_V2_PURPOSE",
    "SessionDenominatorVerificationError",
    "VerifiedCalendarRules",
    "VerifiedConsumerContract",
    "VerifiedDenominatorConsumerScope",
    "VerifiedDenominatorScope",
    "VerifiedDenominatorScopeV2",
    "VerifiedSessionDenominator",
    "content_sha256",
    "denominator_consumer_scope_document",
    "denominator_scope_v2_document",
    "load_and_verify_session_denominator",
    "load_calendar_rules",
    "load_consumer_contract",
    "load_denominator_consumer_scope_v1",
    "load_denominator_scope",
    "load_denominator_scope_v2",
    "session_denominator_document",
    "sha256_file",
    "verify_session_denominator",
]
