"""Independent FFM consumer for AlphaForge partitioned session bundles v2.

The accepted schema is intentionally synthetic and production-blocked.  Successful
verification proves producer/consumer parity only; it never grants training admission.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
import re
from typing import Any, Iterator, Mapping

from ._authority_bundle_io import (
    AuthorityBundleIOError,
    VerifiedDirectoryReader,
    content_sha256,
    require_sha256,
)
from .session_denominator import (
    SCOPE_V2_BLOCKERS,
    SessionDenominatorVerificationError,
    VerifiedCalendarRules,
    VerifiedDenominatorConsumerScope,
    VerifiedDenominatorScopeV2,
    _day,
    _expected_row,
    _verified_timezone,
    denominator_consumer_scope_document,
    denominator_scope_v2_document,
    load_calendar_rules,
    load_denominator_consumer_scope_v1,
    load_denominator_scope_v2,
)


MANIFEST_SCHEMA = "alphaforge_session_denominator_bundle_v2"
SHARD_SCHEMA = "alphaforge_session_denominator_shard_v2"
MANIFEST_NAME = "manifest.json"
# Live producer parity is tested against this exact revision when its independent
# worktree is present.  Absence of that optional evidence can never remove BLOCKERS.
BLOCKERS = SCOPE_V2_BLOCKERS
PRODUCER_COMPATIBILITY_COMMIT = "b84925763459c2f1a7f4300d11e9760867083629"
MAX_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_SHARD_BYTES = 16 * 1024 * 1024
MAX_SHARD_NODES = 750_000
MAX_SHARD_ROWS = 10_000
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_CAPABILITY_TOKEN = object()

_MANIFEST_KEYS = {
    "schema_version", "production_admission", "admission_blockers",
    "calendar_rules", "scope_v2", "narrow_consumer_scope", "roots",
    "partitions", "shard_count", "row_count", "shards",
    "manifest_semantic_sha256",
}
_ENTRY_KEYS = {
    "partition_id", "root", "leaf", "physical_sha256", "semantic_sha256",
    "row_count", "start", "end_exclusive",
}
_SHARD_KEYS = {
    "schema_version", "partition_id", "root", "start", "end_exclusive",
    "permitted_uses", "row_count", "rows", "shard_semantic_sha256",
}


class SessionDenominatorBundleVerificationError(ValueError):
    """Raised when a partitioned session authority fails independent verification."""


@dataclass(frozen=True)
class VerifiedSessionDenominatorBundleV2:
    bundle_path: Path
    manifest_physical_sha256: str
    manifest_semantic_sha256: str
    calendar_rules_path: Path
    calendar_rules_sha256: str
    scope_v2_path: Path
    scope_v2_sha256: str
    consumer_scope_path: Path
    consumer_scope_sha256: str
    shard_count: int
    row_count: int
    production_admitted: bool
    _token: object


def _fail(message: str) -> SessionDenominatorBundleVerificationError:
    return SessionDenominatorBundleVerificationError(message)


def _exact_keys(value: Any, expected: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise _fail(f"{label} has an invalid exact schema")
    return value


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise _fail(f"{label} is not a constrained identifier")
    return value


def _count(value: Any, label: str, *, maximum: int | None = None) -> int:
    if type(value) is not int or value < 0 or (maximum is not None and value > maximum):
        raise _fail(f"{label} must be a bounded nonnegative integer")
    return value


def _leaf(partition: str, root: str) -> str:
    return f"shard--{_identifier(partition, 'partition ID')}--{_identifier(root, 'root')}.json"


def _parents(
    *,
    calendar_rules_path: str | Path,
    calendar_rules_sha256: str,
    scope_v2_path: str | Path,
    scope_v2_sha256: str,
    consumer_scope_path: str | Path,
    consumer_scope_sha256: str,
) -> tuple[
    VerifiedCalendarRules, VerifiedDenominatorScopeV2,
    VerifiedDenominatorConsumerScope,
]:
    rules = load_calendar_rules(
        calendar_rules_path, expected_sha256=calendar_rules_sha256
    )
    consumer = load_denominator_consumer_scope_v1(
        consumer_scope_path, expected_sha256=consumer_scope_sha256
    )
    scope = load_denominator_scope_v2(
        scope_v2_path,
        expected_sha256=scope_v2_sha256,
        rules=rules,
        consumer_scope=consumer,
    )
    return rules, scope, consumer


def _verify_shard(
    document: Mapping[str, Any],
    *,
    split: Mapping[str, Any],
    root: str,
    rules: VerifiedCalendarRules,
) -> int:
    _exact_keys(document, _SHARD_KEYS, "session shard")
    if (
        document["schema_version"] != SHARD_SCHEMA
        or document["partition_id"] != split["partition_id"]
        or document["root"] != root
        or document["start"] != split["start"]
        or document["end_exclusive"] != split["end_exclusive"]
        or document["permitted_uses"] != split["permitted_uses"]
    ):
        raise _fail("session shard partition binding mismatch")
    rows = document["rows"]
    row_count = _count(document["row_count"], "session shard row_count", maximum=MAX_SHARD_ROWS)
    start = _day(document["start"], "session shard start")
    end = _day(document["end_exclusive"], "session shard end")
    expected_count = (end - start).days
    if (
        not isinstance(rows, list)
        or row_count != len(rows)
        or row_count != expected_count
    ):
        raise _fail("session shard does not exactly close its calendar-day interval")
    timezone = _verified_timezone(rules.document["dependencies"])
    for offset, row in enumerate(rows):
        expected_day = (start + timedelta(days=offset)).isoformat()
        if (
            not isinstance(row, Mapping)
            or row.get("root") != root
            or row.get("session_day") != expected_day
            or _expected_row(rules.document, row, timezone) != row
        ):
            raise _fail("session shard row differs from independent calendar resolution")
    semantic = require_sha256(
        document["shard_semantic_sha256"], "session shard semantic SHA-256"
    )
    if semantic != content_sha256(dict(document), "shard_semantic_sha256"):
        raise _fail("session shard semantic SHA-256 mismatch")
    return row_count


def _verify_once(
    bundle_path: str | Path,
    *,
    expected_manifest_sha256: str,
    calendar_rules_path: str | Path,
    calendar_rules_sha256: str,
    scope_v2_path: str | Path,
    scope_v2_sha256: str,
    consumer_scope_path: str | Path,
    consumer_scope_sha256: str,
) -> VerifiedSessionDenominatorBundleV2:
    require_sha256(expected_manifest_sha256, "expected session manifest SHA-256")
    rules, scope, consumer = _parents(
        calendar_rules_path=calendar_rules_path,
        calendar_rules_sha256=calendar_rules_sha256,
        scope_v2_path=scope_v2_path,
        scope_v2_sha256=scope_v2_sha256,
        consumer_scope_path=consumer_scope_path,
        consumer_scope_sha256=consumer_scope_sha256,
    )
    scope_document = denominator_scope_v2_document(
        scope, rules=rules, consumer_scope=consumer
    )
    consumer_document = denominator_consumer_scope_document(consumer)
    usable_splits = [
        split for split in consumer_document["splits"] if split["permitted_uses"]
    ]
    expected_keys = [
        (split["partition_id"], root)
        for split in usable_splits
        for root in sorted(scope_document["roots"])
    ]
    with VerifiedDirectoryReader(bundle_path, label="session denominator bundle") as reader:
        manifest, manifest_physical = reader.read_json(
            MANIFEST_NAME,
            label="session bundle manifest",
            max_bytes=MAX_MANIFEST_BYTES,
            max_nodes=100_000,
        )
        if manifest_physical != expected_manifest_sha256:
            raise _fail("session bundle manifest physical SHA-256 mismatch")
        _exact_keys(manifest, _MANIFEST_KEYS, "session bundle manifest")
        if (
            manifest["schema_version"] != MANIFEST_SCHEMA
            or manifest["production_admission"] is not False
            or manifest["admission_blockers"] != list(BLOCKERS)
        ):
            raise _fail("session bundle must remain exactly production-blocked")
        if manifest["calendar_rules"] != {
            "path": str(rules.path), "physical_sha256": rules.physical_sha256,
        }:
            raise _fail("session bundle calendar-rules parent binding mismatch")
        if manifest["scope_v2"] != {
            "path": str(scope.path),
            "physical_sha256": scope.physical_sha256,
            "semantic_sha256": scope.semantic_sha256,
        }:
            raise _fail("session bundle scope-v2 parent binding mismatch")
        if manifest["narrow_consumer_scope"] != {
            "path": str(consumer.path),
            "physical_sha256": consumer.physical_sha256,
            "semantic_sha256": consumer.semantic_sha256,
        }:
            raise _fail("session bundle consumer-scope parent binding mismatch")
        if (
            manifest["roots"] != sorted(scope_document["roots"])
            or manifest["partitions"] != [split["partition_id"] for split in usable_splits]
        ):
            raise _fail("session bundle root/partition closure mismatch")
        entries = manifest["shards"]
        if not isinstance(entries, list):
            raise _fail("session bundle shard index must be a list")
        actual_keys = []
        for index, value in enumerate(entries):
            entry = _exact_keys(value, _ENTRY_KEYS, f"session shard entry {index}")
            partition = _identifier(
                entry["partition_id"], f"session shard entry {index}.partition_id"
            )
            root = _identifier(entry["root"], f"session shard entry {index}.root")
            if not isinstance(entry["leaf"], str):
                raise _fail(f"session shard entry {index}.leaf must be a string")
            require_sha256(
                entry["physical_sha256"],
                f"session shard entry {index} physical SHA-256",
            )
            require_sha256(
                entry["semantic_sha256"],
                f"session shard entry {index} semantic SHA-256",
            )
            _count(
                entry["row_count"],
                f"session shard entry {index}.row_count",
                maximum=MAX_SHARD_ROWS,
            )
            start = _day(entry["start"], f"session shard entry {index}.start")
            end = _day(
                entry["end_exclusive"],
                f"session shard entry {index}.end_exclusive",
            )
            if start >= end:
                raise _fail(f"session shard entry {index} has an empty interval")
            actual_keys.append((partition, root))
        shard_count = _count(manifest["shard_count"], "session bundle shard_count")
        if actual_keys != expected_keys or shard_count != len(expected_keys):
            raise _fail("session bundle shard index is not exact and ordered")
        expected_names = sorted([MANIFEST_NAME, *[entry["leaf"] for entry in entries]])
        if reader.names() != expected_names or len(expected_names) != len(set(expected_names)):
            raise _fail("session bundle directory closure mismatch")
        splits = {split["partition_id"]: split for split in usable_splits}
        total = 0
        for entry in entries:
            partition, root = entry["partition_id"], entry["root"]
            if entry["leaf"] != _leaf(partition, root):
                raise _fail("session shard leaf is not canonical")
            document, physical = reader.read_json(
                entry["leaf"],
                label="session denominator shard",
                max_bytes=MAX_SHARD_BYTES,
                max_nodes=MAX_SHARD_NODES,
            )
            count = _verify_shard(
                document, split=splits[partition], root=root, rules=rules
            )
            expected_entry = {
                "partition_id": partition,
                "root": root,
                "leaf": entry["leaf"],
                "physical_sha256": physical,
                "semantic_sha256": document["shard_semantic_sha256"],
                "row_count": count,
                "start": document["start"],
                "end_exclusive": document["end_exclusive"],
            }
            if entry != expected_entry:
                raise _fail("session shard manifest binding mismatch")
            total += count
        manifest_row_count = _count(manifest["row_count"], "session bundle row_count")
        if manifest_row_count != total:
            raise _fail("session bundle aggregate row count mismatch")
        manifest_semantic = require_sha256(
            manifest["manifest_semantic_sha256"], "session manifest semantic SHA-256"
        )
        if manifest_semantic != content_sha256(manifest, "manifest_semantic_sha256"):
            raise _fail("session manifest semantic SHA-256 mismatch")
        reader.assert_unchanged()
        return VerifiedSessionDenominatorBundleV2(
            reader.path,
            manifest_physical,
            manifest_semantic,
            rules.path,
            rules.physical_sha256,
            scope.path,
            scope.physical_sha256,
            consumer.path,
            consumer.physical_sha256,
            len(entries),
            total,
            False,
            _CAPABILITY_TOKEN,
        )


def load_and_verify_session_denominator_bundle_v2(
    bundle_path: str | Path,
    *,
    expected_manifest_sha256: str,
    calendar_rules_path: str | Path,
    calendar_rules_sha256: str,
    scope_v2_path: str | Path,
    scope_v2_sha256: str,
    consumer_scope_path: str | Path,
    consumer_scope_sha256: str,
) -> VerifiedSessionDenominatorBundleV2:
    """Load twice and return a non-forgeable, production-blocked capability."""
    arguments = dict(
        expected_manifest_sha256=expected_manifest_sha256,
        calendar_rules_path=calendar_rules_path,
        calendar_rules_sha256=calendar_rules_sha256,
        scope_v2_path=scope_v2_path,
        scope_v2_sha256=scope_v2_sha256,
        consumer_scope_path=consumer_scope_path,
        consumer_scope_sha256=consumer_scope_sha256,
    )
    try:
        first = _verify_once(bundle_path, **arguments)
        second = _verify_once(bundle_path, **arguments)
    except (AuthorityBundleIOError, SessionDenominatorVerificationError) as exc:
        raise _fail(str(exc)) from exc
    if first != second:
        raise _fail("session bundle changed across mandatory reopen")
    return first


def iter_verified_session_shards(
    capability: VerifiedSessionDenominatorBundleV2,
) -> Iterator[dict[str, Any]]:
    """Reverify at use, then yield shards bound to the verified capability."""
    if (
        type(capability) is not VerifiedSessionDenominatorBundleV2
        or capability._token is not _CAPABILITY_TOKEN
        or capability.production_admitted is not False
    ):
        raise _fail("a verified production-blocked session-bundle capability is required")
    reopened = load_and_verify_session_denominator_bundle_v2(
        capability.bundle_path,
        expected_manifest_sha256=capability.manifest_physical_sha256,
        calendar_rules_path=capability.calendar_rules_path,
        calendar_rules_sha256=capability.calendar_rules_sha256,
        scope_v2_path=capability.scope_v2_path,
        scope_v2_sha256=capability.scope_v2_sha256,
        consumer_scope_path=capability.consumer_scope_path,
        consumer_scope_sha256=capability.consumer_scope_sha256,
    )
    if reopened != capability:
        raise _fail("session bundle capability changed before use")
    primary_error = False
    try:
        with VerifiedDirectoryReader(
            capability.bundle_path, label="session denominator bundle"
        ) as reader:
            manifest, physical = reader.read_json(
                MANIFEST_NAME,
                label="session bundle manifest",
                max_bytes=MAX_MANIFEST_BYTES,
                max_nodes=100_000,
            )
            if physical != capability.manifest_physical_sha256:
                raise _fail("session manifest changed before iteration")
            for entry in manifest["shards"]:
                shard, shard_physical = reader.read_json(
                    entry["leaf"],
                    label="session denominator shard",
                    max_bytes=MAX_SHARD_BYTES,
                    max_nodes=MAX_SHARD_NODES,
                )
                if shard_physical != entry["physical_sha256"]:
                    raise _fail("session shard changed before iteration")
                yield shard
            reader.assert_unchanged()
    except GeneratorExit:
        raise
    except BaseException:
        primary_error = True
        raise
    finally:
        if not primary_error:
            after = load_and_verify_session_denominator_bundle_v2(
                capability.bundle_path,
                expected_manifest_sha256=capability.manifest_physical_sha256,
                calendar_rules_path=capability.calendar_rules_path,
                calendar_rules_sha256=capability.calendar_rules_sha256,
                scope_v2_path=capability.scope_v2_path,
                scope_v2_sha256=capability.scope_v2_sha256,
                consumer_scope_path=capability.consumer_scope_path,
                consumer_scope_sha256=capability.consumer_scope_sha256,
            )
            if after != capability:
                raise _fail("session bundle capability changed during use")


__all__ = [
    "BLOCKERS",
    "MANIFEST_SCHEMA",
    "PRODUCER_COMPATIBILITY_COMMIT",
    "SHARD_SCHEMA",
    "SessionDenominatorBundleVerificationError",
    "VerifiedSessionDenominatorBundleV2",
    "iter_verified_session_shards",
    "load_and_verify_session_denominator_bundle_v2",
]
