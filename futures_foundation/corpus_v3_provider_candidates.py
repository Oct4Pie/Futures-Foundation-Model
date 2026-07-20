"""Independent FFM consumer for synthetic provider candidate-evidence bundles.

The verifier consumes metadata-only query scope, page receipts and responses through
one held directory descriptor.  It is joined to a reverified producer/split capability
by the scope's claimed semantic hashes.  No market-data namespace is opened or listed.

The evidence remains production-blocked because provider authenticity, transport and
an immutable evidence mount are not established by this schema.  Candidate metadata
is not availability, liquidity, lifecycle, materialization or training authority.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import hashlib
import os
import re
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

from ._authority_bundle_io import (
    AuthorityBundleIOError,
    VerifiedDirectoryReader,
    canonical_absolute_path,
    canonical_json_bytes,
    content_sha256,
    require_sha256,
)
from .corpus_v3_producer_governance import (
    VerifiedFrozenSplitUseContractV1,
    reopen_and_verify_frozen_split_use_contract_v1,
)


SCOPE_SCHEMA = "alphaforge_provider_candidate_query_scope_v1"
RESPONSE_SCHEMA = "alphaforge_provider_candidate_response_v1"
RECEIPT_SCHEMA = "alphaforge_provider_candidate_receipt_v1"
BUNDLE_MANIFEST_SCHEMA = "alphaforge_provider_candidate_evidence_bundle_v1"
PURPOSE = "provider_metadata_only_contract_candidate_universe"
SOURCE_KIND = "synthetic_fixture"
PRODUCTION_BLOCKER = "provider_authenticity_and_transport_evidence_unavailable"
UPSTREAM_AUTHORITY_BLOCKER = "producer_governance_and_split_use_capabilities_unavailable"
IMMUTABLE_MOUNT_BLOCKER = "immutable_evidence_mount_and_single_dirfd_consumption_unavailable"
FFM_BLOCKERS = (
    PRODUCTION_BLOCKER,
    IMMUTABLE_MOUNT_BLOCKER,
    "synthetic_fixture_not_provider_attestation",
)
PRODUCER_COMPATIBILITY_COMMIT = "b84925763459c2f1a7f4300d11e9760867083629"
MAX_EVIDENCE_JSON_BYTES = 16 * 1024 * 1024
MAX_EVIDENCE_NODES = 2_000_000
MAX_PAGES = 10_000
MAX_TOTAL_CANDIDATES = 1_000_000
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_SYMBOL_RE = re.compile(r"^[A-Z0-9][A-Z0-9._:/-]{0,63}$")
_CANDIDATE_FIELDS = {
    "provider_instrument_id", "provider_symbol", "root_symbol", "venue",
    "instrument_class",
}
_FORBIDDEN_FIELD_TERMS = {
    "path", "lake_path", "file", "file_path", "filename", "availability", "available",
    "row", "rows", "row_count", "tick", "ticks", "tick_count", "price", "prices",
    "open_price", "high", "low", "close", "volume", "labels", "label", "outcome",
    "outcomes", "return", "returns", "pnl", "profit", "loss", "mfe", "mae", "r_multiple",
}
_CAPABILITY_TOKEN = object()

_MANIFEST_FIELDS = {
    "schema_version", "purpose", "source_kind", "entries", "production_admitted",
    "admission_blockers", "manifest_semantic_sha256",
}
_MANIFEST_ENTRY_FIELDS = {"role", "path", "sha256", "size"}
_SCOPE_FIELDS = {
    "schema_version", "purpose", "provider_id", "provider_api_revision", "query",
    "claimed_producer_governance_sha256", "claimed_frozen_split_use_contract_sha256",
    "production_admission_requested", "scope_semantic_sha256",
}
_QUERY_FIELDS = {
    "method", "endpoint", "venue", "instrument_class", "root_filter", "page_size",
    "pagination_mode", "cursor_parameter", "requested_fields",
}
_RECEIPT_FIELDS = {
    "schema_version", "source_kind", "provider_id", "provider_api_revision",
    "scope_sha256", "query_semantic_sha256", "page_index", "cursor_in",
    "request_semantic_sha256", "response_path", "response_sha256", "response_size",
    "http_status", "content_type", "captured_utc", "receipt_semantic_sha256",
}
_RESPONSE_FIELDS = {
    "schema_version", "provider_id", "provider_api_revision", "scope_sha256",
    "query_semantic_sha256", "page_index", "cursor_in", "next_cursor", "candidates",
    "response_semantic_sha256",
}


class CorpusV3ProviderCandidateError(ValueError):
    """Raised when provider metadata evidence is unsafe, incomplete or ambiguous."""


def _fail(message: str) -> CorpusV3ProviderCandidateError:
    return CorpusV3ProviderCandidateError(message)


@dataclass(frozen=True)
class ProviderCandidateV1:
    provider_instrument_id: str
    provider_symbol: str
    root_symbol: str
    venue: str
    instrument_class: str


@dataclass(frozen=True)
class VerifiedProviderCandidateUniverseV1:
    evidence_root: Path
    manifest_path: Path
    manifest_physical_sha256: str
    manifest_semantic_sha256: str
    scope_physical_sha256: str
    scope_semantic_sha256: str
    provider_id: str
    provider_api_revision: str
    query_semantic_sha256: str
    pagination_semantic_sha256: str
    page_receipt_sha256s: tuple[str, ...]
    provider_response_sha256s: tuple[str, ...]
    candidates: tuple[ProviderCandidateV1, ...]
    split: VerifiedFrozenSplitUseContractV1
    evidence_status: str
    production_admitted: bool
    admission_blockers: tuple[str, ...]
    _token: object

    def document(self) -> dict[str, Any]:
        return {
            "schema_version": "ffm_verified_provider_candidate_universe_v1",
            "evidence_root": self.evidence_root.as_posix(),
            "manifest_path": self.manifest_path.as_posix(),
            "manifest_physical_sha256": self.manifest_physical_sha256,
            "manifest_semantic_sha256": self.manifest_semantic_sha256,
            "scope_physical_sha256": self.scope_physical_sha256,
            "scope_semantic_sha256": self.scope_semantic_sha256,
            "provider_id": self.provider_id,
            "provider_api_revision": self.provider_api_revision,
            "producer_governance_semantic_sha256": self.split.producer.semantic_sha256,
            "frozen_split_use_semantic_sha256": self.split.semantic_sha256,
            "query_semantic_sha256": self.query_semantic_sha256,
            "pagination_semantic_sha256": self.pagination_semantic_sha256,
            "page_receipt_sha256s": list(self.page_receipt_sha256s),
            "provider_response_sha256s": list(self.provider_response_sha256s),
            "candidate_count": len(self.candidates),
            "candidates": [
                {
                    "provider_instrument_id": item.provider_instrument_id,
                    "provider_symbol": item.provider_symbol,
                    "root_symbol": item.root_symbol,
                    "venue": item.venue,
                    "instrument_class": item.instrument_class,
                }
                for item in self.candidates
            ],
            "evidence_status": self.evidence_status,
            "production_admitted": self.production_admitted,
            "admission_blockers": list(self.admission_blockers),
            "market_namespace_opened": False,
            "availability_claimed": False,
            "lifecycle_claimed": False,
            "materialization_admitted": False,
            "training_admitted": False,
        }


def _exact_mapping(value: Any, fields: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        missing = sorted(fields - set(value)) if isinstance(value, Mapping) else sorted(fields)
        unknown = sorted(set(value) - fields) if isinstance(value, Mapping) else []
        raise _fail(f"{label} fields mismatch; missing={missing}, unknown={unknown}")
    return value


def _sha(value: Any, label: str) -> str:
    try:
        return require_sha256(value, label)
    except AuthorityBundleIOError as exc:
        raise _fail(str(exc)) from exc


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise _fail(f"{label} is not a constrained identifier")
    return value


def _symbol(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SYMBOL_RE.fullmatch(value) is None:
        raise _fail(f"{label} is not a canonical provider symbol")
    return value


def _safe_leaf(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or PurePosixPath(value).name != value
        or "/" in value
        or "\\" in value
    ):
        raise _fail(f"{label} is not a safe evidence leaf")
    return value


def _reject_forbidden_fields(value: Any, label: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower().replace("-", "_")
            if normalized in _FORBIDDEN_FIELD_TERMS:
                raise _fail(f"{label} contains forbidden observed-data field: {key}")
            _reject_forbidden_fields(child, label)
    elif isinstance(value, list):
        for child in value:
            _reject_forbidden_fields(child, label)


def _timestamp(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise _fail(f"{label} must be a canonical second-resolution UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise _fail(f"{label} must be a canonical second-resolution UTC timestamp") from exc
    if (
        parsed.tzinfo != timezone.utc
        or parsed.isoformat(timespec="seconds").replace("+00:00", "Z") != value
    ):
        raise _fail(f"{label} must be a canonical second-resolution UTC timestamp")
    return value


def _request_hash(
    scope_sha256: str,
    query_sha256: str,
    page_index: int,
    cursor_in: str | None,
) -> str:
    return hashlib.sha256(canonical_json_bytes({
        "scope_sha256": scope_sha256,
        "query_semantic_sha256": query_sha256,
        "page_index": page_index,
        "cursor_in": cursor_in,
    })).hexdigest()


def _lexically_within(path: Path, root: str) -> bool:
    try:
        return os.path.commonpath((path.as_posix(), root)) == root
    except ValueError:
        return False


def _candidate(
    value: Any,
    *,
    label: str,
    roots: set[str],
    venue: str,
) -> ProviderCandidateV1:
    candidate = _exact_mapping(value, _CANDIDATE_FIELDS, label)
    _reject_forbidden_fields(candidate, label)
    result = ProviderCandidateV1(
        provider_instrument_id=_identifier(
            candidate["provider_instrument_id"], f"{label}.provider_instrument_id"
        ),
        provider_symbol=_symbol(candidate["provider_symbol"], f"{label}.provider_symbol"),
        root_symbol=_symbol(candidate["root_symbol"], f"{label}.root_symbol"),
        venue=_identifier(candidate["venue"], f"{label}.venue"),
        instrument_class=_identifier(
            candidate["instrument_class"], f"{label}.instrument_class"
        ),
    )
    if result.root_symbol not in roots:
        raise _fail(f"{label} root is outside the frozen query scope")
    if result.venue != venue or result.instrument_class != "future":
        raise _fail(f"{label} venue or instrument class differs from the query scope")
    return result


def _manifest(
    reader: VerifiedDirectoryReader,
    *,
    manifest_leaf: str,
    expected_sha256: str,
) -> tuple[dict[str, Any], str, str, list[tuple[str, str, str, int]]]:
    if manifest_leaf != "bundle-manifest.json":
        raise _fail("provider manifest must be bundle-manifest.json")
    try:
        document, physical = reader.read_json(
            manifest_leaf,
            label="provider candidate bundle manifest",
            max_bytes=MAX_EVIDENCE_JSON_BYTES,
            max_nodes=MAX_EVIDENCE_NODES,
            max_depth=12,
        )
    except AuthorityBundleIOError as exc:
        raise _fail(str(exc)) from exc
    if physical != _sha(expected_sha256, "expected provider manifest SHA-256"):
        raise _fail("provider candidate manifest physical SHA-256 changed")
    _exact_mapping(document, _MANIFEST_FIELDS, "provider candidate manifest")
    if (
        document["schema_version"] != BUNDLE_MANIFEST_SCHEMA
        or document["purpose"] != PURPOSE
        or document["source_kind"] != SOURCE_KIND
        or document["production_admitted"] is not False
        or document["admission_blockers"] != [
            PRODUCTION_BLOCKER, UPSTREAM_AUTHORITY_BLOCKER, IMMUTABLE_MOUNT_BLOCKER,
        ]
    ):
        raise _fail("provider candidate manifest must remain synthetic and blocked")
    semantic = _sha(
        document["manifest_semantic_sha256"], "provider manifest semantic SHA-256"
    )
    if content_sha256(dict(document), "manifest_semantic_sha256") != semantic:
        raise _fail("provider candidate manifest semantic hash mismatch")
    entries = document["entries"]
    if not isinstance(entries, list) or not entries or len(entries) > 1 + 2 * MAX_PAGES:
        raise _fail("provider candidate manifest entry count is invalid")
    normalized: list[tuple[str, str, str, int]] = []
    for index, raw in enumerate(entries):
        entry = _exact_mapping(raw, _MANIFEST_ENTRY_FIELDS, f"manifest entry {index}")
        role = _identifier(entry["role"], f"manifest entry {index}.role")
        leaf = _safe_leaf(entry["path"], f"manifest entry {index}.path")
        if leaf == manifest_leaf:
            raise _fail("provider manifest cannot list itself")
        digest = _sha(entry["sha256"], f"manifest entry {index}.sha256")
        size = entry["size"]
        if type(size) is not int or not 0 <= size <= MAX_EVIDENCE_JSON_BYTES:
            raise _fail(f"manifest entry {index}.size is invalid")
        normalized.append((role, leaf, digest, size))
    if normalized != sorted(normalized):
        raise _fail("provider manifest entries must be role-sorted")
    if len({row[0] for row in normalized}) != len(normalized):
        raise _fail("provider manifest roles must be unique")
    if len({row[1] for row in normalized}) != len(normalized):
        raise _fail("provider manifest paths must be unique")
    receipt_rows = [row for row in normalized if row[0].startswith("receipt:")]
    response_rows = [row for row in normalized if row[0].startswith("response:")]
    scope_rows = [row for row in normalized if row[0] == "scope"]
    if len(scope_rows) != 1 or not receipt_rows or len(receipt_rows) != len(response_rows):
        raise _fail("provider manifest requires one scope and paired page evidence")
    expected_receipts = [f"receipt:{index:06d}" for index in range(len(receipt_rows))]
    expected_responses = [f"response:{index:06d}" for index in range(len(response_rows))]
    if [row[0] for row in receipt_rows] != expected_receipts:
        raise _fail("provider receipt roles are not contiguous")
    if [row[0] for row in response_rows] != expected_responses:
        raise _fail("provider response roles are not contiguous")
    if {row[0] for row in normalized} != {"scope", *expected_receipts, *expected_responses}:
        raise _fail("provider manifest role closure is not exact")
    expected_names = sorted([manifest_leaf, *(row[1] for row in normalized)])
    if reader.names() != expected_names:
        raise _fail("provider evidence directory closure differs from the manifest")
    for role, leaf, digest, size in normalized:
        try:
            value, actual = reader.read_json(
                leaf,
                label=f"provider evidence {role}",
                max_bytes=MAX_EVIDENCE_JSON_BYTES,
                max_nodes=MAX_EVIDENCE_NODES,
                max_depth=20,
            )
        except AuthorityBundleIOError as exc:
            raise _fail(str(exc)) from exc
        if actual != digest or len(canonical_json_bytes(value)) != size:
            raise _fail(f"provider evidence bytes differ from manifest: {role}")
    return dict(document), physical, semantic, normalized


def _derive(
    reader: VerifiedDirectoryReader,
    *,
    rows: Sequence[tuple[str, str, str, int]],
    split: VerifiedFrozenSplitUseContractV1,
) -> dict[str, Any]:
    by_role = {row[0]: row for row in rows}
    scope_leaf = by_role["scope"][1]
    try:
        scope, scope_sha = reader.read_json(
            scope_leaf,
            label="provider query scope",
            max_bytes=MAX_EVIDENCE_JSON_BYTES,
            max_nodes=MAX_EVIDENCE_NODES,
            max_depth=12,
        )
    except AuthorityBundleIOError as exc:
        raise _fail(str(exc)) from exc
    _reject_forbidden_fields(scope, "provider query scope")
    _exact_mapping(scope, _SCOPE_FIELDS, "provider query scope")
    if scope["schema_version"] != SCOPE_SCHEMA or scope["purpose"] != PURPOSE:
        raise _fail("unsupported provider query scope")
    provider_id = _identifier(scope["provider_id"], "scope.provider_id")
    provider_revision = _identifier(
        scope["provider_api_revision"], "scope.provider_api_revision"
    )
    if (
        _sha(
            scope["claimed_producer_governance_sha256"],
            "scope claimed producer-governance SHA-256",
        )
        != split.producer.semantic_sha256
        or _sha(
            scope["claimed_frozen_split_use_contract_sha256"],
            "scope claimed split/use SHA-256",
        )
        != split.semantic_sha256
    ):
        raise _fail("provider query scope claims differ from reverified upstream authority")
    if scope["production_admission_requested"] is not False:
        raise _fail("synthetic provider scope cannot request production admission")
    query = _exact_mapping(scope["query"], _QUERY_FIELDS, "provider scope query")
    if query["method"] != "GET" or query["pagination_mode"] != "opaque_cursor":
        raise _fail("provider query method or pagination mode is unsupported")
    endpoint = query["endpoint"]
    parsed = urlsplit(endpoint) if isinstance(endpoint, str) else None
    if (
        parsed is None
        or parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise _fail("provider endpoint must be a plain HTTPS URL")
    venue = _identifier(query["venue"], "provider query venue")
    if query["instrument_class"] != "future":
        raise _fail("provider query must enumerate futures")
    roots = query["root_filter"]
    if (
        not isinstance(roots, list)
        or not roots
        or roots != sorted(set(roots))
        or any(not isinstance(root, str) or _SYMBOL_RE.fullmatch(root) is None for root in roots)
    ):
        raise _fail("provider query roots must be sorted unique symbols")
    if type(query["page_size"]) is not int or not 1 <= query["page_size"] <= 10_000:
        raise _fail("provider query page_size is invalid")
    _identifier(query["cursor_parameter"], "provider query cursor_parameter")
    if query["requested_fields"] != sorted(_CANDIDATE_FIELDS):
        raise _fail("provider query requested_fields are not metadata-only")
    scope_semantic = _sha(
        scope["scope_semantic_sha256"], "provider scope semantic SHA-256"
    )
    if content_sha256(dict(scope), "scope_semantic_sha256") != scope_semantic:
        raise _fail("provider query scope semantic hash mismatch")
    query_sha = hashlib.sha256(canonical_json_bytes(query)).hexdigest()

    receipt_rows = [row for row in rows if row[0].startswith("receipt:")]
    response_rows = [row for row in rows if row[0].startswith("response:")]
    expected_cursor: str | None = None
    seen_cursors: set[str] = set()
    previous_capture: str | None = None
    pages: list[dict[str, Any]] = []
    receipt_hashes: list[str] = []
    response_hashes: list[str] = []
    candidates: list[ProviderCandidateV1] = []
    root_set = set(roots)
    for page_index, (receipt_row, response_row) in enumerate(
        zip(receipt_rows, response_rows)
    ):
        receipt_leaf = receipt_row[1]
        response_leaf = response_row[1]
        try:
            receipt, receipt_sha = reader.read_json(
                receipt_leaf,
                label=f"provider receipt {page_index}",
                max_bytes=MAX_EVIDENCE_JSON_BYTES,
                max_nodes=MAX_EVIDENCE_NODES,
                max_depth=12,
            )
            response, response_sha = reader.read_json(
                response_leaf,
                label=f"provider response {page_index}",
                max_bytes=MAX_EVIDENCE_JSON_BYTES,
                max_nodes=MAX_EVIDENCE_NODES,
                max_depth=20,
            )
        except AuthorityBundleIOError as exc:
            raise _fail(str(exc)) from exc
        _reject_forbidden_fields(receipt, f"provider receipt {page_index}")
        _reject_forbidden_fields(response, f"provider response {page_index}")
        _exact_mapping(receipt, _RECEIPT_FIELDS, f"provider receipt {page_index}")
        _exact_mapping(response, _RESPONSE_FIELDS, f"provider response {page_index}")
        if (
            receipt["schema_version"] != RECEIPT_SCHEMA
            or receipt["source_kind"] != SOURCE_KIND
            or receipt["provider_id"] != provider_id
            or receipt["provider_api_revision"] != provider_revision
            or receipt["scope_sha256"] != scope_sha
            or receipt["query_semantic_sha256"] != query_sha
            or type(receipt["page_index"]) is not int
            or receipt["page_index"] != page_index
        ):
            raise _fail("provider receipt identity mismatch")
        cursor = receipt["cursor_in"]
        if cursor is not None:
            _identifier(cursor, f"provider receipt {page_index}.cursor_in")
        if cursor != expected_cursor:
            raise _fail("provider receipt cursor chain is inconsistent")
        if receipt["request_semantic_sha256"] != _request_hash(
            scope_sha, query_sha, page_index, cursor
        ):
            raise _fail("provider receipt request hash mismatch")
        if (
            type(receipt["http_status"]) is not int
            or receipt["http_status"] != 200
            or receipt["content_type"] != "application/json"
        ):
            raise _fail("provider receipt was not successful JSON")
        captured = _timestamp(receipt["captured_utc"], f"receipt {page_index}.captured_utc")
        if previous_capture is not None and captured < previous_capture:
            raise _fail("provider receipt capture timestamps are not ordered")
        previous_capture = captured
        if _safe_leaf(receipt["response_path"], "receipt response_path") != response_leaf:
            raise _fail("provider receipt response_path differs from manifest pairing")
        if (
            receipt["response_sha256"] != response_sha
            or type(receipt["response_size"]) is not int
            or receipt["response_size"] != len(canonical_json_bytes(response))
        ):
            raise _fail("provider receipt does not bind exact response bytes")
        if (
            response["schema_version"] != RESPONSE_SCHEMA
            or response["provider_id"] != provider_id
            or response["provider_api_revision"] != provider_revision
            or response["scope_sha256"] != scope_sha
            or response["query_semantic_sha256"] != query_sha
            or type(response["page_index"]) is not int
            or response["page_index"] != page_index
            or response["cursor_in"] != cursor
        ):
            raise _fail("provider response identity mismatch")
        next_cursor = response["next_cursor"]
        if next_cursor is not None:
            _identifier(next_cursor, f"provider response {page_index}.next_cursor")
            if next_cursor == cursor or next_cursor in seen_cursors:
                raise _fail("provider pagination cursor did not advance uniquely")
            seen_cursors.add(next_cursor)
        page_candidates = response["candidates"]
        if (
            not isinstance(page_candidates, list)
            or len(page_candidates) > query["page_size"]
        ):
            raise _fail("provider response candidate page is invalid")
        normalized = [
            _candidate(
                value,
                label=f"response {page_index}.candidate {offset}",
                roots=root_set,
                venue=venue,
            )
            for offset, value in enumerate(page_candidates)
        ]
        if content_sha256(dict(response), "response_semantic_sha256") != _sha(
            response["response_semantic_sha256"],
            f"provider response {page_index} semantic SHA-256",
        ):
            raise _fail("provider response semantic hash mismatch")
        if content_sha256(dict(receipt), "receipt_semantic_sha256") != _sha(
            receipt["receipt_semantic_sha256"],
            f"provider receipt {page_index} semantic SHA-256",
        ):
            raise _fail("provider receipt semantic hash mismatch")
        pages.append({
            "page_index": page_index,
            "cursor_in": cursor,
            "next_cursor": next_cursor,
            "request_semantic_sha256": receipt["request_semantic_sha256"],
            "receipt_sha256": receipt_sha,
            "response_sha256": response_sha,
        })
        receipt_hashes.append(receipt_sha)
        response_hashes.append(response_sha)
        candidates.extend(normalized)
        if len(candidates) > MAX_TOTAL_CANDIDATES:
            raise _fail("provider candidate count exceeds the resource limit")
        expected_cursor = next_cursor
        if expected_cursor is None and page_index != len(receipt_rows) - 1:
            raise _fail("provider pagination terminated before the final page")
    if expected_cursor is not None:
        raise _fail("provider pagination proof lacks a terminal page")
    ids = [item.provider_instrument_id for item in candidates]
    symbols = [item.provider_symbol for item in candidates]
    if len(ids) != len(set(ids)) or len(symbols) != len(set(symbols)):
        raise _fail("provider candidate IDs and symbols must be globally unique")
    candidates.sort(
        key=lambda item: (item.root_symbol, item.provider_symbol, item.provider_instrument_id)
    )
    return {
        "scope_physical_sha256": scope_sha,
        "scope_semantic_sha256": scope_semantic,
        "provider_id": provider_id,
        "provider_api_revision": provider_revision,
        "query_semantic_sha256": query_sha,
        "pagination_semantic_sha256": hashlib.sha256(
            canonical_json_bytes(pages)
        ).hexdigest(),
        "page_receipt_sha256s": tuple(receipt_hashes),
        "provider_response_sha256s": tuple(response_hashes),
        "candidates": tuple(candidates),
    }


def load_and_verify_provider_candidate_universe_v1(
    evidence_root: str | Path,
    *,
    manifest_path: str | Path,
    manifest_sha256: str,
    split_capability: VerifiedFrozenSplitUseContractV1,
) -> VerifiedProviderCandidateUniverseV1:
    """Verify the complete synthetic bundle and return a blocked capability."""
    split = reopen_and_verify_frozen_split_use_contract_v1(split_capability)
    try:
        root = canonical_absolute_path(evidence_root, "provider evidence root")
        manifest = canonical_absolute_path(manifest_path, "provider manifest path")
    except AuthorityBundleIOError as exc:
        raise _fail(str(exc)) from exc
    if manifest.parent != root or manifest.name != "bundle-manifest.json":
        raise _fail("provider manifest must be the fixed direct child of evidence root")
    if _lexically_within(root, split.producer.namespace_root):
        raise _fail("provider evidence root is lexically beneath the market namespace")
    try:
        with VerifiedDirectoryReader(root, label="provider candidate evidence root") as reader:
            _, manifest_physical, manifest_semantic, rows = _manifest(
                reader,
                manifest_leaf=manifest.name,
                expected_sha256=manifest_sha256,
            )
            derived = _derive(reader, rows=rows, split=split)
            reader.assert_unchanged()
    except AuthorityBundleIOError as exc:
        raise _fail(str(exc)) from exc
    return VerifiedProviderCandidateUniverseV1(
        evidence_root=root,
        manifest_path=manifest,
        manifest_physical_sha256=manifest_physical,
        manifest_semantic_sha256=manifest_semantic,
        scope_physical_sha256=derived["scope_physical_sha256"],
        scope_semantic_sha256=derived["scope_semantic_sha256"],
        provider_id=derived["provider_id"],
        provider_api_revision=derived["provider_api_revision"],
        query_semantic_sha256=derived["query_semantic_sha256"],
        pagination_semantic_sha256=derived["pagination_semantic_sha256"],
        page_receipt_sha256s=derived["page_receipt_sha256s"],
        provider_response_sha256s=derived["provider_response_sha256s"],
        candidates=derived["candidates"],
        split=split,
        evidence_status="synthetic_provider_metadata_pagination_only",
        production_admitted=False,
        admission_blockers=FFM_BLOCKERS,
        _token=_CAPABILITY_TOKEN,
    )


def reopen_and_verify_provider_candidate_universe_v1(
    capability: VerifiedProviderCandidateUniverseV1,
) -> VerifiedProviderCandidateUniverseV1:
    if (
        type(capability) is not VerifiedProviderCandidateUniverseV1
        or capability._token is not _CAPABILITY_TOKEN
        or capability.production_admitted is not False
    ):
        raise _fail("a verified production-blocked provider candidate capability is required")
    reopened = load_and_verify_provider_candidate_universe_v1(
        capability.evidence_root,
        manifest_path=capability.manifest_path,
        manifest_sha256=capability.manifest_physical_sha256,
        split_capability=capability.split,
    )
    if reopened != capability:
        raise _fail("provider candidate evidence changed before use")
    return reopened


__all__ = [
    "BUNDLE_MANIFEST_SCHEMA", "FFM_BLOCKERS", "IMMUTABLE_MOUNT_BLOCKER",
    "PRODUCTION_BLOCKER", "PRODUCER_COMPATIBILITY_COMMIT", "PURPOSE",
    "RECEIPT_SCHEMA", "RESPONSE_SCHEMA", "SCOPE_SCHEMA", "SOURCE_KIND",
    "UPSTREAM_AUTHORITY_BLOCKER", "CorpusV3ProviderCandidateError",
    "ProviderCandidateV1", "VerifiedProviderCandidateUniverseV1",
    "load_and_verify_provider_candidate_universe_v1",
    "reopen_and_verify_provider_candidate_universe_v1",
]
