"""Split-scoped source inventory and non-executable materialization plan.

The inventory accounts for every expected contract-session request exactly once using
metadata-only source-file identities.  It cannot add requests, cross frozen split
boundaries, or read market content.  The derived plan selects only requests with exact,
nonoverlapping, gap-free source coverage and remains execution-blocked pending an
independent production approval and immutable-storage authority.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping, Sequence

from ._authority_bundle_io import (
    AuthorityBundleIOError,
    canonical_absolute_path,
    canonical_json_bytes,
    content_sha256,
    read_canonical_json_file,
    require_sha256,
)
from .corpus_v3_expected_requests import SCHEMA_VERSION as EXPECTED_SCHEMA


OBSERVATION_SCHEMA = "ffm_corpus_v3_inventory_observations_v1"
OBSERVATION_PURPOSE = "split_scoped_expected_request_source_metadata_observations_v1"
INVENTORY_SCHEMA = "ffm_corpus_v3_split_scoped_inventory_v1"
INVENTORY_POLICY = "expected_request_exact_closure_metadata_only_v1"
PLAN_SCHEMA = "ffm_corpus_v3_materialization_plan_v1"
PLAN_POLICY = "available_exact_requests_only_non_executable_v1"
INVENTORY_STATUSES = {
    "available_exact", "missing", "ambiguous", "boundary_blocked",
}
MAX_REQUESTS = 5_000_000
MAX_SOURCE_FILES_PER_REQUEST = 100_000
MAX_CANONICAL_BYTES = 768 * 1024 * 1024
MAX_OBSERVATION_BYTES = 512 * 1024 * 1024
MAX_OBSERVATION_NODES = 8_000_000
_PATH_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/@:+-]*(?:/[A-Za-z0-9][A-Za-z0-9._/@:+-]*)*$")
_INPUT_FIELDS = {"request_semantic_sha256", "status", "reason", "source_files"}
_OBSERVATION_FIELDS = {
    "schema_version", "purpose", "expected_request_denominator_sha256",
    "rows", "observations_semantic_sha256",
}
_SOURCE_FIELDS = {
    "relative_path", "interval_start_utc_ns", "interval_end_exclusive_utc_ns",
    "size", "sha256",
}
_REQUEST_FIELDS = {
    "candidate_index", "provider_instrument_id", "provider_symbol", "contract_id",
    "venue", "session_day", "session_segment_index", "request_start_utc_ns",
    "request_end_exclusive_utc_ns",
}


class CorpusV3MaterializationPlanError(ValueError):
    """Raised when inventory or plan closure is unsafe or inconsistent."""


def _fail(message: str) -> CorpusV3MaterializationPlanError:
    return CorpusV3MaterializationPlanError(message)


def _exact(value: Any, fields: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        missing = sorted(fields - set(value)) if isinstance(value, Mapping) else sorted(fields)
        unknown = sorted(set(value) - fields) if isinstance(value, Mapping) else []
        raise _fail(f"{label} fields mismatch; missing={missing}, unknown={unknown}")
    return value


def _count(value: Any, label: str, *, maximum: int | None = None) -> int:
    if type(value) is not int or value < 0 or (maximum is not None and value > maximum):
        raise _fail(f"{label} is outside the admitted nonnegative range")
    return value


def _ns(value: Any, label: str) -> int:
    if type(value) is not int or value < 0 or value > 9_223_372_036_854_775_807:
        raise _fail(f"{label} must be a nonnegative signed UTC-ns integer")
    return value


def _relative_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise _fail(f"{label} must be a canonical relative POSIX path")
    parsed = PurePosixPath(value)
    if (
        parsed.is_absolute()
        or parsed.as_posix() != value
        or any(part in {"", ".", ".."} for part in parsed.parts)
        or _PATH_RE.fullmatch(value) is None
    ):
        raise _fail(f"{label} must be a canonical relative POSIX path")
    return value


def _expected_integrity(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _fail("expected-request denominator must be an object")
    document = deepcopy(dict(value))
    supplied = document.get("expected_request_denominator_sha256")
    payload = deepcopy(document)
    payload.pop("expected_request_denominator_sha256", None)
    if (
        document.get("schema_version") != EXPECTED_SCHEMA
        or supplied != content_sha256(payload, "expected_request_denominator_sha256")
        or document.get("production_admitted") is not False
        or document.get("materialization_admitted") is not False
        or document.get("training_admitted") is not False
        or document.get("oos_admitted") is not False
    ):
        raise _fail("expected-request denominator is stale, unsupported, or authorizing")
    return document


def _expected_requests(value: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    shards = value.get("request_shards")
    if not isinstance(shards, list):
        raise _fail("expected-request denominator shard closure is invalid")
    for shard_index, shard_raw in enumerate(shards):
        if not isinstance(shard_raw, Mapping):
            raise _fail(f"expected request shard {shard_index} is invalid")
        partition = shard_raw.get("partition_id")
        root = shard_raw.get("root")
        permitted = shard_raw.get("permitted_uses")
        requests = shard_raw.get("requests")
        if (
            not isinstance(partition, str) or not partition
            or not isinstance(root, str) or not root
            or not isinstance(permitted, list) or not permitted
            or not isinstance(requests, list)
            or shard_raw.get("request_count") != len(requests)
        ):
            raise _fail("expected request shard identity/count is invalid")
        for request_index, request_raw in enumerate(requests):
            request = _exact(
                request_raw, _REQUEST_FIELDS,
                f"expected request {partition}/{root}/{request_index}",
            )
            start = _ns(request["request_start_utc_ns"], "expected request start")
            end = _ns(
                request["request_end_exclusive_utc_ns"], "expected request end"
            )
            if start >= end:
                raise _fail("expected request interval must be positive half-open")
            identity = {
                "partition_id": partition,
                "root": root,
                "permitted_uses": list(permitted),
                **dict(request),
            }
            identity["request_semantic_sha256"] = hashlib.sha256(
                canonical_json_bytes(identity)
            ).hexdigest()
            rows.append(identity)
    if len(rows) > MAX_REQUESTS:
        raise _fail("expected request count exceeds inventory resource limit")
    identifiers = [row["request_semantic_sha256"] for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise _fail("expected-request denominator contains duplicate request identities")
    return rows


def _source_files(
    value: Any,
    *,
    request_start: int,
    request_end: int,
    status: str,
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) > MAX_SOURCE_FILES_PER_REQUEST:
        raise _fail("inventory source_files must be a bounded list")
    files: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for index, raw in enumerate(value):
        row = _exact(raw, _SOURCE_FIELDS, f"inventory source file {index}")
        path = _relative_path(row["relative_path"], "inventory source relative_path")
        if path in seen_paths:
            raise _fail("inventory source files contain a duplicate relative path")
        seen_paths.add(path)
        start = _ns(row["interval_start_utc_ns"], "inventory source interval start")
        end = _ns(
            row["interval_end_exclusive_utc_ns"], "inventory source interval end"
        )
        if start >= end:
            raise _fail("inventory source interval must be positive half-open")
        size = _count(row["size"], "inventory source size")
        digest = require_sha256(row["sha256"], "inventory source SHA-256")
        files.append({
            "relative_path": path,
            "interval_start_utc_ns": start,
            "interval_end_exclusive_utc_ns": end,
            "size": size,
            "sha256": digest,
        })
    files.sort(key=lambda row: (
        row["interval_start_utc_ns"], row["interval_end_exclusive_utc_ns"],
        row["relative_path"],
    ))
    if status == "available_exact":
        if not files:
            raise _fail("available_exact inventory rows require source files")
        cursor = request_start
        previous_end: int | None = None
        for row in files:
            start = row["interval_start_utc_ns"]
            end = row["interval_end_exclusive_utc_ns"]
            if end <= request_start or start >= request_end:
                raise _fail("available_exact source file does not overlap its request")
            if previous_end is not None and start < previous_end:
                raise _fail("available_exact source intervals overlap")
            if start > cursor:
                raise _fail("available_exact source intervals leave an internal gap")
            cursor = max(cursor, end)
            previous_end = end
        if cursor < request_end:
            raise _fail("available_exact source intervals do not cover request end")
    elif files and status in {"missing", "boundary_blocked"}:
        raise _fail(f"{status} inventory rows cannot carry source files")
    return files


def load_inventory_observations_v1(
    path: str | Path,
    *,
    expected_physical_sha256: str,
    expected_request_denominator: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load one exact metadata-only inventory-observation authority artifact."""
    expected = _expected_integrity(expected_request_denominator)
    try:
        source = canonical_absolute_path(path, "inventory observations path")
        reopened, document, physical = read_canonical_json_file(
            source,
            label="Corpus-v3 inventory observations",
            max_bytes=MAX_OBSERVATION_BYTES,
            max_nodes=MAX_OBSERVATION_NODES,
            max_depth=20,
        )
    except AuthorityBundleIOError as exc:
        raise _fail(str(exc)) from exc
    if (
        reopened != source
        or physical != require_sha256(
            expected_physical_sha256, "expected inventory observations SHA-256"
        )
    ):
        raise _fail("inventory observations physical SHA-256 differs from authority")
    _exact(document, _OBSERVATION_FIELDS, "inventory observations")
    if (
        document["schema_version"] != OBSERVATION_SCHEMA
        or document["purpose"] != OBSERVATION_PURPOSE
        or document["expected_request_denominator_sha256"]
        != expected["expected_request_denominator_sha256"]
    ):
        raise _fail("inventory observations schema, purpose, or expected parent differs")
    semantic = require_sha256(
        document["observations_semantic_sha256"],
        "inventory observations semantic SHA-256",
    )
    if content_sha256(
        dict(document), "observations_semantic_sha256"
    ) != semantic:
        raise _fail("inventory observations semantic hash mismatch")
    rows = document["rows"]
    if not isinstance(rows, list) or len(rows) > MAX_REQUESTS:
        raise _fail("inventory observation rows must be a bounded list")
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        row = _exact(raw, _INPUT_FIELDS, f"inventory observation row {index}")
        normalized.append({
            "request_semantic_sha256": require_sha256(
                row["request_semantic_sha256"],
                f"inventory observation row {index} request SHA-256",
            ),
            "status": row["status"],
            "reason": row["reason"],
            "source_files": deepcopy(row["source_files"]),
        })
    return normalized, {
        "path": source.as_posix(),
        "physical_sha256": physical,
        "semantic_sha256": semantic,
        "row_count": len(normalized),
    }


def build_split_scoped_inventory_v1(
    *,
    expected_request_denominator: Mapping[str, Any],
    inventory_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    expected = _expected_integrity(expected_request_denominator)
    requests = _expected_requests(expected)
    if not isinstance(inventory_rows, Sequence) or isinstance(
        inventory_rows, (str, bytes, bytearray)
    ):
        raise _fail("inventory_rows must be a sequence")
    supplied: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(inventory_rows):
        row = _exact(raw, _INPUT_FIELDS, f"inventory input row {index}")
        digest = require_sha256(
            row["request_semantic_sha256"],
            f"inventory input row {index} request SHA-256",
        )
        if digest in supplied:
            raise _fail("inventory contains duplicate request rows")
        supplied[digest] = row
    expected_ids = [row["request_semantic_sha256"] for row in requests]
    if set(supplied) != set(expected_ids) or len(supplied) != len(expected_ids):
        raise _fail("inventory request closure differs from the expected denominator")
    normalized: list[dict[str, Any]] = []
    status_counts = {status: 0 for status in sorted(INVENTORY_STATUSES)}
    for request in requests:
        raw = supplied[request["request_semantic_sha256"]]
        status = raw["status"]
        if status not in INVENTORY_STATUSES:
            raise _fail("inventory status is unsupported")
        reason = raw["reason"]
        if status == "available_exact":
            if reason is not None:
                raise _fail("available_exact inventory rows cannot carry an exclusion reason")
        elif not isinstance(reason, str) or not reason or reason.strip() != reason:
            raise _fail("nonavailable inventory rows require a canonical reason")
        files = _source_files(
            raw["source_files"],
            request_start=request["request_start_utc_ns"],
            request_end=request["request_end_exclusive_utc_ns"],
            status=status,
        )
        normalized.append({
            **request,
            "status": status,
            "reason": reason,
            "source_files": files,
        })
        status_counts[status] += 1
    document: dict[str, Any] = {
        "schema_version": INVENTORY_SCHEMA,
        "policy": INVENTORY_POLICY,
        "purpose": "split_scoped_expected_request_source_inventory_metadata_only",
        "expected_request_denominator": {
            "sha256": expected["expected_request_denominator_sha256"],
            "request_count": expected["counts"]["expected_requests"],
        },
        "counts": {
            "inventory_rows": len(normalized),
            **{f"status_{key}": value for key, value in status_counts.items()},
        },
        "complete_against_expected_requests": len(normalized) == len(requests),
        "all_requests_available_exact": (
            len(normalized) == len(requests)
            and status_counts["available_exact"] == len(requests)
        ),
        "rows": normalized,
        "data_access": {
            "expected_requests_only": True,
            "reserved_oos_requests_present": False,
            "source_metadata_read": True,
            "source_content_read": False,
            "prices_volume_rows_or_labels_read": False,
            "source_file_bytes_reopened_by_this_verifier": False,
        },
        "production_admitted": False,
        "materialization_admitted": False,
        "training_admitted": False,
        "oos_admitted": False,
    }
    if len(canonical_json_bytes(document)) > MAX_CANONICAL_BYTES:
        raise _fail("split-scoped inventory exceeds canonical byte limit")
    document["inventory_sha256"] = content_sha256(document, "inventory_sha256")
    return document


def validate_split_scoped_inventory_v1(
    value: Any,
    *,
    expected_request_denominator: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _fail("split-scoped inventory must be an object")
    supplied = value.get("inventory_sha256")
    payload = deepcopy(dict(value))
    payload.pop("inventory_sha256", None)
    if supplied != content_sha256(payload, "inventory_sha256"):
        raise _fail("split-scoped inventory integrity mismatch")
    rows = value.get("rows")
    if not isinstance(rows, list):
        raise _fail("split-scoped inventory rows are invalid")
    inputs = [
        {
            "request_semantic_sha256": row.get("request_semantic_sha256"),
            "status": row.get("status"),
            "reason": row.get("reason"),
            "source_files": row.get("source_files"),
        }
        for row in rows
    ]
    expected = build_split_scoped_inventory_v1(
        expected_request_denominator=expected_request_denominator,
        inventory_rows=inputs,
    )
    if dict(value) != expected:
        raise _fail("split-scoped inventory is stale or non-canonical")
    return deepcopy(expected)


def build_materialization_plan_v1(
    *,
    expected_request_denominator: Mapping[str, Any],
    inventory: Mapping[str, Any],
) -> dict[str, Any]:
    expected = _expected_integrity(expected_request_denominator)
    verified_inventory = validate_split_scoped_inventory_v1(
        inventory,
        expected_request_denominator=expected,
    )
    selected = [
        {
            key: value
            for key, value in row.items()
            if key not in {"status", "reason"}
        }
        for row in verified_inventory["rows"]
        if row["status"] == "available_exact"
    ]
    excluded = {
        status: verified_inventory["counts"][f"status_{status}"]
        for status in sorted(INVENTORY_STATUSES - {"available_exact"})
    }
    document: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA,
        "policy": PLAN_POLICY,
        "purpose": "non_executable_materialization_selection_from_verified_inventory",
        "expected_request_denominator_sha256": expected[
            "expected_request_denominator_sha256"
        ],
        "inventory_sha256": verified_inventory["inventory_sha256"],
        "counts": {
            "expected_requests": expected["counts"]["expected_requests"],
            "selected_requests": len(selected),
            "excluded_requests": sum(excluded.values()),
            "excluded_by_status": excluded,
        },
        "selected_requests": selected,
        "selection_rule": "status_equals_available_exact_only",
        "execution_status": (
            "blocked_pending_source_byte_reopen_immutable_storage_and_independent_approval"
        ),
        "production_admitted": False,
        "materialization_admitted": False,
        "training_admitted": False,
        "oos_admitted": False,
    }
    if len(canonical_json_bytes(document)) > MAX_CANONICAL_BYTES:
        raise _fail("materialization plan exceeds canonical byte limit")
    document["plan_sha256"] = content_sha256(document, "plan_sha256")
    return document


def validate_materialization_plan_v1(
    value: Any,
    *,
    expected_request_denominator: Mapping[str, Any],
    inventory: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _fail("materialization plan must be an object")
    supplied = value.get("plan_sha256")
    payload = deepcopy(dict(value))
    payload.pop("plan_sha256", None)
    if supplied != content_sha256(payload, "plan_sha256"):
        raise _fail("materialization plan integrity mismatch")
    expected = build_materialization_plan_v1(
        expected_request_denominator=expected_request_denominator,
        inventory=inventory,
    )
    if dict(value) != expected:
        raise _fail("materialization plan is stale or non-canonical")
    return deepcopy(expected)


def write_materialization_artifact(value: Mapping[str, Any], path: str | Path) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(target) + ".tmp")
    temporary.write_bytes(canonical_json_bytes(dict(value)))
    temporary.replace(target)
    return target


__all__ = [
    "INVENTORY_POLICY", "INVENTORY_SCHEMA", "INVENTORY_STATUSES",
    "OBSERVATION_PURPOSE", "OBSERVATION_SCHEMA", "PLAN_POLICY", "PLAN_SCHEMA",
    "CorpusV3MaterializationPlanError", "build_materialization_plan_v1",
    "build_split_scoped_inventory_v1", "load_inventory_observations_v1",
    "validate_materialization_plan_v1", "validate_split_scoped_inventory_v1",
    "write_materialization_artifact",
]
