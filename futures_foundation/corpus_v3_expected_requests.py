"""Deterministic expected contract-session requests without market observation.

Dependency direction is one-way::

    frozen split/use + verified session denominator + verified lifecycle
        -> expected request denominator

No provider lake, file inventory, tick, price, volume, availability, label, outcome or
materialization plan is accepted.  Open session segments are intersected with admitted
contract lifecycle intervals.  Quarantined candidates remain explicit zero-request
rows rather than disappearing from the denominator.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ._authority_bundle_io import canonical_json_bytes, content_sha256
from .corpus_v3_contract_lifecycle import (
    LifecycleRowV2,
    VerifiedContractLifecycleV2,
    reopen_and_verify_contract_lifecycle_v2,
)
from .corpus_v3_producer_governance import (
    VerifiedFrozenSplitUseContractV1,
    reopen_and_verify_frozen_split_use_contract_v1,
)
from .session_denominator_bundle import (
    VerifiedSessionDenominatorBundleV2,
    iter_verified_session_shards,
)


SCHEMA_VERSION = "ffm_corpus_v3_expected_request_denominator_v1"
POLICY = "split_session_lifecycle_intersection_no_market_observation_v1"
OPEN_STATUSES = {"regular", "shortened", "extended", "irregular"}
MAX_REQUEST_ROWS = 5_000_000
MAX_CANDIDATES = 1_000_000
MAX_CANONICAL_BYTES = 512 * 1024 * 1024


class CorpusV3ExpectedRequestError(ValueError):
    """Raised when expected-request authority is incomplete or inconsistent."""


def _fail(message: str) -> CorpusV3ExpectedRequestError:
    return CorpusV3ExpectedRequestError(message)


def _exact(value: Any, fields: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        missing = sorted(fields - set(value)) if isinstance(value, Mapping) else sorted(fields)
        unknown = sorted(set(value) - fields) if isinstance(value, Mapping) else []
        raise _fail(f"{label} fields mismatch; missing={missing}, unknown={unknown}")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise _fail(f"{label} must be a nonnegative integer")
    return value


def _segments(value: Any, label: str) -> list[list[int]]:
    if not isinstance(value, list):
        raise _fail(f"{label} must be a list")
    result: list[list[int]] = []
    previous_end: int | None = None
    for index, raw in enumerate(value):
        if (
            not isinstance(raw, list)
            or len(raw) != 2
            or any(type(item) is not int or item < 0 for item in raw)
            or raw[0] >= raw[1]
            or (previous_end is not None and raw[0] < previous_end)
        ):
            raise _fail(f"{label} segment {index} is invalid")
        result.append([int(raw[0]), int(raw[1])])
        previous_end = int(raw[1])
    return result


def _candidate_key(row: LifecycleRowV2) -> tuple[str, str, str]:
    return row.venue, row.provider_symbol, row.provider_instrument_id


def _split_maps(
    split: VerifiedFrozenSplitUseContractV1,
) -> tuple[dict[str, tuple[str, str]], dict[str, tuple[str, ...]]]:
    intervals = {
        row.partition_id: (row.start, row.end_exclusive)
        for row in split.partitions
    }
    uses = dict(split.permitted_uses)
    return intervals, uses


def _candidate_reason(row: LifecycleRowV2, emitted: int) -> list[str]:
    if row.disposition == "quarantine":
        if row.eligibility_start_utc_ns is None:
            return ["missing_lifecycle_evidence"]
        return ["lifecycle_interval_outside_protocol_scope"]
    if emitted == 0:
        return ["no_open_session_segment_lifecycle_intersection"]
    return []


def _derive(
    *,
    split: VerifiedFrozenSplitUseContractV1,
    session: VerifiedSessionDenominatorBundleV2,
    lifecycle: VerifiedContractLifecycleV2,
    session_shards: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    if lifecycle.split != split:
        raise _fail("lifecycle and expected-request split capabilities differ")
    rows = tuple(lifecycle.rows)
    if len(rows) > MAX_CANDIDATES:
        raise _fail("lifecycle candidate count exceeds expected-request limit")
    grouped: dict[str, list[LifecycleRowV2]] = {}
    for row in rows:
        grouped.setdefault(row.root, []).append(row)
    for root in grouped:
        grouped[root].sort(key=_candidate_key)
        identifiers = [row.provider_instrument_id for row in grouped[root]]
        if len(identifiers) != len(set(identifiers)):
            raise _fail("lifecycle candidates contain duplicate provider identities")
    candidate_roots = sorted(grouped)
    split_intervals, split_uses = _split_maps(split)
    emitted = {row.provider_instrument_id: 0 for row in rows}
    seen_shards: set[tuple[str, str]] = set()
    request_shards: list[dict[str, Any]] = []
    total_sessions = 0
    total_requests = 0
    for shard_index, raw in enumerate(session_shards):
        shard = _exact(
            raw,
            {
                "schema_version", "partition_id", "root", "permitted_uses", "start",
                "end_exclusive", "row_count", "rows", "shard_semantic_sha256",
            },
            f"session shard {shard_index}",
        )
        partition = shard["partition_id"]
        root = shard["root"]
        if not isinstance(partition, str) or not partition:
            raise _fail("session shard partition_id is invalid")
        if not isinstance(root, str) or not root:
            raise _fail("session shard root is invalid")
        key = (partition, root)
        if key in seen_shards:
            raise _fail("session denominator contains a duplicate partition/root shard")
        seen_shards.add(key)
        if partition not in split_intervals:
            raise _fail("session denominator partition is outside frozen split/use")
        partition_start, partition_end = split_intervals[partition]
        if shard["start"] != partition_start or shard["end_exclusive"] != partition_end:
            raise _fail("session shard interval differs from frozen split partition")
        permitted = shard["permitted_uses"]
        allowed = split_uses[partition]
        if (
            not isinstance(permitted, list)
            or not permitted
            or permitted != sorted(set(permitted))
            or any(value not in allowed for value in permitted)
        ):
            raise _fail("session shard uses are empty, noncanonical, or outside frozen split/use")
        if root not in grouped:
            raise _fail("session denominator root lacks a lifecycle candidate disposition")
        session_rows = shard["rows"]
        row_count = _nonnegative_int(shard["row_count"], "session shard row_count")
        if not isinstance(session_rows, list) or row_count != len(session_rows):
            raise _fail("session shard row count is inconsistent")
        candidates = grouped[root]
        requests: list[dict[str, Any]] = []
        for session_index, session_raw in enumerate(session_rows):
            session_row = _exact(
                session_raw,
                {
                    "root", "session_day", "status", "segments_utc_ns",
                    "source_ids", "segment_semantic_sha256",
                },
                f"session shard {partition}/{root} row {session_index}",
            )
            if session_row["root"] != root:
                raise _fail("session row/root identity mismatch")
            day = session_row["session_day"]
            if not isinstance(day, str) or not partition_start <= day < partition_end:
                raise _fail("session day is outside the frozen split partition")
            status = session_row["status"]
            if not isinstance(status, str) or not status:
                raise _fail("session status is invalid")
            segments = _segments(
                session_row["segments_utc_ns"],
                f"session {root}/{day} segments",
            )
            if (status in OPEN_STATUSES) != bool(segments):
                raise _fail("session status and segment closure disagree")
            total_sessions += 1
            if status not in OPEN_STATUSES:
                continue
            for candidate_index, candidate in enumerate(candidates):
                if candidate.disposition != "admit":
                    continue
                if (
                    candidate.eligibility_start_utc_ns is None
                    or candidate.trading_end_exclusive_utc_ns is None
                ):
                    raise _fail("admitted lifecycle candidate lacks a complete interval")
                for segment_index, (segment_start, segment_end) in enumerate(segments):
                    start = max(segment_start, candidate.eligibility_start_utc_ns)
                    end = min(segment_end, candidate.trading_end_exclusive_utc_ns)
                    if start >= end:
                        continue
                    requests.append({
                        "candidate_index": candidate_index,
                        "provider_instrument_id": candidate.provider_instrument_id,
                        "provider_symbol": candidate.provider_symbol,
                        "contract_id": candidate.contract_id,
                        "venue": candidate.venue,
                        "session_day": day,
                        "session_segment_index": segment_index,
                        "request_start_utc_ns": start,
                        "request_end_exclusive_utc_ns": end,
                    })
                    emitted[candidate.provider_instrument_id] += 1
                    total_requests += 1
                    if total_requests > MAX_REQUEST_ROWS:
                        raise _fail("expected-request count exceeds the resource limit")
        requests.sort(key=lambda row: (
            row["candidate_index"], row["session_day"],
            row["session_segment_index"], row["request_start_utc_ns"],
        ))
        request_shard = {
            "partition_id": partition,
            "root": root,
            "permitted_uses": list(permitted),
            "parent_session_shard_semantic_sha256": shard["shard_semantic_sha256"],
            "session_disposition_count": row_count,
            "request_count": len(requests),
            "requests": requests,
        }
        request_shard["request_shard_semantic_sha256"] = content_sha256(
            request_shard, "request_shard_semantic_sha256"
        )
        request_shards.append(request_shard)
    partition_rank = {
        row.partition_id: index for index, row in enumerate(split.partitions)
    }
    request_shards.sort(
        key=lambda row: (partition_rank[row["partition_id"]], row["root"])
    )
    observed_roots = sorted({row["root"] for row in request_shards})
    if observed_roots != candidate_roots:
        raise _fail("session denominator and lifecycle root closures differ")
    expected_shard_keys = [
        (partition.partition_id, root)
        for partition in split.partitions
        if split_uses[partition.partition_id]
        for root in candidate_roots
    ]
    actual_shard_keys = [
        (row["partition_id"], row["root"])
        for row in request_shards
    ]
    if actual_shard_keys != expected_shard_keys:
        raise _fail("session denominator partition/root closure differs from frozen split/use")
    candidate_rows: list[dict[str, Any]] = []
    for root in candidate_roots:
        for candidate_index, candidate in enumerate(grouped[root]):
            evidence_state = (
                "complete" if candidate.eligibility_start_utc_ns is not None
                else "unestablished"
            )
            count = emitted[candidate.provider_instrument_id]
            candidate_rows.append({
                "root": root,
                "candidate_index": candidate_index,
                "provider_instrument_id": candidate.provider_instrument_id,
                "provider_symbol": candidate.provider_symbol,
                "venue": candidate.venue,
                "contract_id": candidate.contract_id,
                "lifecycle_disposition": candidate.disposition,
                "lifecycle_evidence_state": evidence_state,
                "lifecycle_start_utc_ns": candidate.eligibility_start_utc_ns,
                "lifecycle_end_exclusive_utc_ns": (
                    candidate.trading_end_exclusive_utc_ns
                ),
                "official_source_ids": list(candidate.official_source_ids),
                "emitted_request_count": count,
                "exclusion_reasons": _candidate_reason(candidate, count),
            })
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "policy": POLICY,
        "purpose": "expected_contract_session_requests_no_market_observation",
        "parent_capabilities": {
            "producer_governance": {
                "path": split.producer.path.as_posix(),
                "physical_sha256": split.producer.physical_sha256,
                "semantic_sha256": split.producer.semantic_sha256,
            },
            "frozen_split_use_contract": {
                "path": split.path.as_posix(),
                "physical_sha256": split.physical_sha256,
                "semantic_sha256": split.semantic_sha256,
            },
            "session_denominator_bundle": {
                "path": session.bundle_path.as_posix(),
                "manifest_physical_sha256": session.manifest_physical_sha256,
                "manifest_semantic_sha256": session.manifest_semantic_sha256,
            },
            "contract_lifecycle": {
                "path": lifecycle.lifecycle_path.as_posix(),
                "physical_sha256": lifecycle.lifecycle_physical_sha256,
                "semantic_sha256": lifecycle.lifecycle_semantic_sha256,
            },
        },
        "split_assignment_basis": "exchange_session_day_only",
        "interval_semantics": "half_open_start_inclusive_end_exclusive",
        "roots": candidate_roots,
        "partitions": [
            row.partition_id for row in split.partitions
            if split_uses[row.partition_id]
        ],
        "counts": {
            "candidate_dispositions": len(candidate_rows),
            "session_dispositions": total_sessions,
            "request_shards": len(request_shards),
            "expected_requests": total_requests,
        },
        "candidate_dispositions": candidate_rows,
        "request_shards": request_shards,
        "data_access": {
            "market_namespace_opened": False,
            "market_files_enumerated": False,
            "market_content_read": False,
            "availability_or_liquidity_read": False,
            "materialization_plan_read": False,
            "labels_or_outcomes_read": False,
        },
        "production_admitted": False,
        "materialization_admitted": False,
        "training_admitted": False,
        "oos_admitted": False,
    }
    encoded = canonical_json_bytes(document)
    if len(encoded) > MAX_CANONICAL_BYTES:
        raise _fail("expected-request denominator exceeds canonical byte limit")
    document["expected_request_denominator_sha256"] = content_sha256(
        document, "expected_request_denominator_sha256"
    )
    return document


def build_expected_request_denominator_v1(
    *,
    split_capability: VerifiedFrozenSplitUseContractV1,
    session_denominator_capability: VerifiedSessionDenominatorBundleV2,
    lifecycle_capability: VerifiedContractLifecycleV2,
) -> dict[str, Any]:
    """Reopen all parent capabilities and derive exact expected requests."""
    split = reopen_and_verify_frozen_split_use_contract_v1(split_capability)
    lifecycle = reopen_and_verify_contract_lifecycle_v2(lifecycle_capability)
    if lifecycle.split != split:
        raise _fail("expected-request parents do not share one split/use capability")
    if (
        type(session_denominator_capability) is not VerifiedSessionDenominatorBundleV2
        or session_denominator_capability.production_admitted is not False
    ):
        raise _fail("a verified production-blocked session denominator capability is required")
    shards = list(iter_verified_session_shards(session_denominator_capability))
    return _derive(
        split=split,
        session=session_denominator_capability,
        lifecycle=lifecycle,
        session_shards=shards,
    )


def validate_expected_request_denominator_v1(
    value: Any,
    *,
    split_capability: VerifiedFrozenSplitUseContractV1,
    session_denominator_capability: VerifiedSessionDenominatorBundleV2,
    lifecycle_capability: VerifiedContractLifecycleV2,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _fail("expected-request denominator must be an object")
    supplied = value.get("expected_request_denominator_sha256")
    payload = deepcopy(dict(value))
    payload.pop("expected_request_denominator_sha256", None)
    if supplied != content_sha256(
        payload, "expected_request_denominator_sha256"
    ):
        raise _fail("expected-request denominator integrity mismatch")
    expected = build_expected_request_denominator_v1(
        split_capability=split_capability,
        session_denominator_capability=session_denominator_capability,
        lifecycle_capability=lifecycle_capability,
    )
    if dict(value) != expected:
        raise _fail("expected-request denominator is stale or non-canonical")
    return deepcopy(expected)


def write_expected_request_denominator_v1(
    value: Mapping[str, Any], path: str | Path,
) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(target) + ".tmp")
    temporary.write_bytes(canonical_json_bytes(dict(value)))
    temporary.replace(target)
    return target


__all__ = [
    "OPEN_STATUSES", "POLICY", "SCHEMA_VERSION", "CorpusV3ExpectedRequestError",
    "build_expected_request_denominator_v1", "validate_expected_request_denominator_v1",
    "write_expected_request_denominator_v1",
]
