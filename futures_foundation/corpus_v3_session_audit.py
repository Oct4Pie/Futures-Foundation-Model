"""Outcome-blind sessionized coverage and yield audit for Corpus v3.

The audit has two independent authority parents:

* a verified session-denominator bundle defines the expected root/session rows;
* verified contract-day exports define observed activity and physical lineage.

No raw path, inventory bucket, strategy event, label, prediction, return, or holdout
outcome can enter this module.  The resulting report is always non-authorizing: it
cannot select roots, admit materialization, or admit model training.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ._authority_bundle_io import (
    AuthorityBundleIOError,
    canonical_absolute_path,
    canonical_json_bytes,
    content_sha256 as authority_content_sha256,
    read_canonical_json_file,
    require_sha256,
)
from .corpus_v3 import CorpusV3Error, content_sha256, load_contract, sha256_file
from .corpus_v3_export import VerifiedContractDayExport, verify_contract_day_export
from .session_denominator_bundle import (
    VerifiedSessionDenominatorBundleV2,
    iter_verified_session_shards,
)


EXPORT_INDEX_SCHEMA = "ffm_corpus_v3_sessionized_export_index_v1"
EXPORT_INDEX_PURPOSE = "outcome_blind_verified_contract_day_export_inventory_v1"
AUDIT_SCHEMA = "ffm_corpus_v3_sessionized_coverage_audit_v1"
AUDIT_POLICY = "verified_denominator_plus_verified_contract_day_exports_v1"
MAX_INDEX_BYTES = 32 * 1024 * 1024
MAX_INDEX_NODES = 750_000

_INDEX_KEYS = {
    "schema_version", "purpose", "contract_sha256", "entries",
    "index_semantic_sha256",
}
_ENTRY_KEYS = {
    "export_path", "expected_request", "receipt_sha256", "output_shard_sha256",
}
_REQUEST_KEYS = {"root", "contract_id", "session_day", "split_use", "purpose"}
_INDEX_IDENTITY_KEYS = {
    "path", "physical_sha256", "semantic_sha256", "entry_count",
}

_SPLIT_TO_DENOMINATOR_USE = {
    "foundation_pretraining": "self_supervised_training",
    "supervised_training": "supervised_training",
    "development": "validation",
}
_CLOSED_STATUSES = {"closed", "prelisting", "delisted"}


class CorpusV3SessionAuditError(ValueError):
    """Raised when the sessionized audit authority chain fails closed."""


def _fail(message: str) -> CorpusV3SessionAuditError:
    return CorpusV3SessionAuditError(message)


def _exact_mapping(value: Any, expected: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise _fail(f"{label} has an invalid exact schema")
    return value


def _strict_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise _fail(f"{label} must be a nonempty canonical string")
    return value


def _strict_count(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise _fail(f"{label} must be a nonnegative integer")
    return value


def _request_key(request: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(request["root"]), str(request["session_day"]),
        str(request["contract_id"]), str(request["split_use"]),
    )


def _validate_request(value: Any, label: str) -> dict[str, str]:
    request = _exact_mapping(value, _REQUEST_KEYS, label)
    result = {name: _strict_text(request[name], f"{label}.{name}") for name in _REQUEST_KEYS}
    if result["purpose"] != "foundation_training":
        raise _fail(f"{label}.purpose must be foundation_training")
    if result["split_use"] not in _SPLIT_TO_DENOMINATOR_USE:
        raise _fail(f"{label}.split_use is outside the admitted audit partitions")
    return result


def load_sessionized_export_index(
    path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load one canonical, exact-closure export index through no-follow transport."""
    try:
        source, document, physical = read_canonical_json_file(
            Path(path).expanduser().resolve(),
            label="Corpus-v3 sessionized export index",
            max_bytes=MAX_INDEX_BYTES,
            max_nodes=MAX_INDEX_NODES,
            max_depth=12,
        )
    except AuthorityBundleIOError as exc:
        raise _fail(str(exc)) from exc
    _exact_mapping(document, _INDEX_KEYS, "sessionized export index")
    if (
        document["schema_version"] != EXPORT_INDEX_SCHEMA
        or document["purpose"] != EXPORT_INDEX_PURPOSE
    ):
        raise _fail("sessionized export index schema or purpose is unsupported")
    contract_sha = require_sha256(
        document["contract_sha256"], "sessionized export index contract SHA-256"
    )
    entries = document["entries"]
    if not isinstance(entries, list):
        raise _fail("sessionized export index entries must be a list")
    normalized: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    seen_requests: set[tuple[str, str, str, str]] = set()
    for index, value in enumerate(entries):
        entry = _exact_mapping(value, _ENTRY_KEYS, f"export index entry {index}")
        try:
            export_path = canonical_absolute_path(
                _strict_text(entry["export_path"], f"entry {index} export_path"),
                f"entry {index} export path",
            )
        except AuthorityBundleIOError as exc:
            raise _fail(str(exc)) from exc
        request = _validate_request(entry["expected_request"], f"entry {index} request")
        receipt_sha = require_sha256(entry["receipt_sha256"], f"entry {index} receipt SHA-256")
        output_sha = require_sha256(
            entry["output_shard_sha256"], f"entry {index} output shard SHA-256"
        )
        path_text = export_path.as_posix()
        request_key = _request_key(request)
        if path_text in seen_paths:
            raise _fail("sessionized export index contains a duplicate export path")
        if request_key in seen_requests:
            raise _fail("sessionized export index contains a duplicate request")
        seen_paths.add(path_text)
        seen_requests.add(request_key)
        normalized.append({
            "export_path": path_text,
            "expected_request": request,
            "receipt_sha256": receipt_sha,
            "output_shard_sha256": output_sha,
        })
    expected_order = sorted(
        normalized,
        key=lambda row: (*_request_key(row["expected_request"]), row["export_path"]),
    )
    if normalized != expected_order:
        raise _fail("sessionized export index entries must use canonical request order")
    semantic = require_sha256(
        document["index_semantic_sha256"], "sessionized export index semantic SHA-256"
    )
    if authority_content_sha256(document, "index_semantic_sha256") != semantic:
        raise _fail("sessionized export index semantic hash mismatch")
    normalized_document = {
        "schema_version": EXPORT_INDEX_SCHEMA,
        "purpose": EXPORT_INDEX_PURPOSE,
        "contract_sha256": contract_sha,
        "entries": normalized,
        "index_semantic_sha256": semantic,
    }
    identity = {
        "path": source.as_posix(),
        "physical_sha256": physical,
        "semantic_sha256": semantic,
        "entry_count": len(normalized),
    }
    return normalized_document, identity


def load_verified_exports_from_index(
    index_path: str | Path,
    *,
    contract_path: str | Path,
    allow_test_contract: bool = False,
) -> tuple[tuple[VerifiedContractDayExport, ...], dict[str, Any]]:
    """Reopen every indexed bundle through the existing independent export verifier."""
    index, identity = load_sessionized_export_index(index_path)
    contract_file = Path(contract_path).expanduser().resolve()
    contract_sha = sha256_file(contract_file)
    if contract_sha != index["contract_sha256"]:
        raise _fail("sessionized export index binds a different Corpus-v3 contract")
    verified: list[VerifiedContractDayExport] = []
    for entry in index["entries"]:
        export_path = Path(entry["export_path"])
        if export_path.is_symlink() or not export_path.is_dir():
            raise _fail("indexed export path is not a non-symlink directory")
        try:
            capability = verify_contract_day_export(
                export_path,
                contract_path=contract_file,
                expected_request=entry["expected_request"],
                allow_test_contract=allow_test_contract,
            )
        except CorpusV3Error as exc:
            raise _fail(f"indexed export failed independent verification: {exc}") from exc
        if (
            capability.receipt_sha256 != entry["receipt_sha256"]
            or capability.output_shard_sha256 != entry["output_shard_sha256"]
            or capability.contract_sha256 != contract_sha
        ):
            raise _fail("indexed export identity differs from the verified capability")
        verified.append(capability)
    return tuple(verified), identity


@dataclass(frozen=True)
class _ObservedContractDay:
    timestamps_utc_ns: np.ndarray
    root: str
    contract_id: str
    session_day: str
    split_use: str
    receipt_sha256: str
    output_shard_sha256: str
    source_file_table_sha256: str
    environment_receipt_sha256: str
    instrument_spec_sha256: str
    trade_rows: int
    quote_valid_rows: int
    negative_trade_rows: int
    zero_trade_rows: int
    total_volume: float
    coverage_start_utc_ns: int
    coverage_end_utc_ns: int
    session_start_utc_ns: int
    session_end_utc_ns: int
    source_file_count: int


def _observation(capability: VerifiedContractDayExport) -> _ObservedContractDay:
    if (
        type(capability) is not VerifiedContractDayExport
        or not capability.is_authentic()
    ):
        raise _fail("a verified contract-day export capability is required")
    arrays = capability.arrays
    timestamps = np.asarray(arrays["timestamp_utc_ns"], dtype=np.int64)
    prices = np.asarray(arrays["price"], dtype=np.float64)
    quote_valid = np.asarray(arrays["quote_valid"], dtype=np.bool_)
    volume = np.asarray(arrays["volume"], dtype=np.float64)
    files = (capability.receipt.get("source_file_table") or {}).get("files") or []
    return _ObservedContractDay(
        timestamps_utc_ns=timestamps,
        root=capability.root,
        contract_id=capability.contract_id,
        session_day=capability.session_day,
        split_use=capability.split_use,
        receipt_sha256=capability.receipt_sha256,
        output_shard_sha256=capability.output_shard_sha256,
        source_file_table_sha256=capability.source_file_table_sha256,
        environment_receipt_sha256=capability.environment_receipt_sha256,
        instrument_spec_sha256=capability.instrument_spec_sha256,
        trade_rows=len(timestamps),
        quote_valid_rows=int(np.count_nonzero(quote_valid)),
        negative_trade_rows=int(np.count_nonzero(prices < 0)),
        zero_trade_rows=int(np.count_nonzero(prices == 0)),
        total_volume=float(volume.sum(dtype=np.float64)),
        coverage_start_utc_ns=int(timestamps[0]),
        coverage_end_utc_ns=int(timestamps[-1]),
        session_start_utc_ns=capability.session_start_utc_ns,
        session_end_utc_ns=capability.session_end_utc_ns,
        source_file_count=len(files),
    )


def _segments(value: Any, label: str) -> list[list[int]]:
    if not isinstance(value, list):
        raise _fail(f"{label} segments must be a list")
    result: list[list[int]] = []
    previous_end: int | None = None
    for index, segment in enumerate(value):
        if (
            not isinstance(segment, list)
            or len(segment) != 2
            or any(type(item) is not int for item in segment)
            or segment[0] >= segment[1]
            or (previous_end is not None and segment[0] < previous_end)
        ):
            raise _fail(f"{label} segment {index} is invalid")
        result.append([int(segment[0]), int(segment[1])])
        previous_end = int(segment[1])
    return result


def _p10(values: Sequence[int]) -> float:
    if not values:
        return 0.0
    ordered = sorted(int(value) for value in values)
    index = max(0, int(0.10 * (len(ordered) - 1)))
    return float(ordered[index])


def _max_missing_run(expected_days: Sequence[str], observed_days: set[str]) -> int:
    longest = current = 0
    for day in expected_days:
        if day in observed_days:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def _aggregate_sessionized_coverage(
    *,
    denominator_shards: Iterable[Mapping[str, Any]],
    observations: Sequence[_ObservedContractDay],
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected: dict[tuple[str, str], dict[str, Any]] = {}
    partition_roots: dict[tuple[str, str], list[str]] = defaultdict(list)
    denominator_rows = 0
    for shard in denominator_shards:
        if not isinstance(shard, Mapping):
            raise _fail("verified denominator yielded a malformed shard")
        partition = _strict_text(shard.get("partition_id"), "denominator partition")
        root = _strict_text(shard.get("root"), "denominator root")
        permitted = shard.get("permitted_uses")
        if (
            not isinstance(permitted, list)
            or not permitted
            or any(not isinstance(value, str) or not value for value in permitted)
            or permitted != sorted(set(permitted))
        ):
            raise _fail("denominator permitted uses are malformed")
        rows = shard.get("rows")
        if not isinstance(rows, list):
            raise _fail("denominator shard rows must be a list")
        for row in rows:
            if not isinstance(row, Mapping) or row.get("root") != root:
                raise _fail("denominator row/root identity mismatch")
            day = _strict_text(row.get("session_day"), "denominator session day")
            key = (root, day)
            if key in expected:
                raise _fail("denominator contains a duplicate root/session day")
            segments = _segments(row.get("segments_utc_ns"), f"denominator {root}/{day}")
            status = _strict_text(row.get("status"), "denominator status")
            if (status in _CLOSED_STATUSES) != (not segments):
                raise _fail("denominator status and segment closure disagree")
            expected[key] = {
                "partition_id": partition,
                "root": root,
                "session_day": day,
                "permitted_uses": list(permitted),
                "status": status,
                "segments_utc_ns": segments,
                "segment_semantic_sha256": require_sha256(
                    row.get("segment_semantic_sha256"),
                    "denominator segment semantic SHA-256",
                ),
            }
            partition_roots[(partition, root)].append(day)
            denominator_rows += 1

    observed_contract_keys: set[tuple[str, str, str]] = set()
    by_session: dict[tuple[str, str], list[_ObservedContractDay]] = defaultdict(list)
    contract_days: list[dict[str, Any]] = []
    for observation in observations:
        contract_key = (
            observation.root, observation.contract_id, observation.session_day,
        )
        if contract_key in observed_contract_keys:
            raise _fail("observed exports contain a duplicate contract/session day")
        observed_contract_keys.add(contract_key)
        key = (observation.root, observation.session_day)
        row = expected.get(key)
        if row is None:
            raise _fail("observed export is outside the verified session denominator")
        segments = row["segments_utc_ns"]
        if not segments or row["status"] in _CLOSED_STATUSES:
            raise _fail("observed export targets a closed, prelisting, or delisted session")
        required_use = _SPLIT_TO_DENOMINATOR_USE.get(observation.split_use)
        if required_use not in row["permitted_uses"]:
            raise _fail("observed export split is not permitted by the denominator partition")
        if (
            observation.session_start_utc_ns != int(segments[0][0])
            or observation.session_end_utc_ns != int(segments[-1][1])
        ):
            raise _fail("observed export envelope differs from denominator segments")
        timestamp_mask = np.zeros(len(observation.timestamps_utc_ns), dtype=np.bool_)
        for segment_start, segment_end in segments:
            timestamp_mask |= (
                (observation.timestamps_utc_ns >= int(segment_start))
                & (observation.timestamps_utc_ns < int(segment_end))
            )
        if not bool(np.all(timestamp_mask)):
            raise _fail("observed export contains events outside denominator segments")
        by_session[key].append(observation)
        contract_days.append({
            "partition_id": row["partition_id"],
            "root": observation.root,
            "contract_id": observation.contract_id,
            "session_day": observation.session_day,
            "split_use": observation.split_use,
            "trade_rows": observation.trade_rows,
            "quote_valid_rows": observation.quote_valid_rows,
            "negative_trade_rows": observation.negative_trade_rows,
            "zero_trade_rows": observation.zero_trade_rows,
            "total_volume": observation.total_volume,
            "coverage_start_utc_ns": observation.coverage_start_utc_ns,
            "coverage_end_utc_ns": observation.coverage_end_utc_ns,
            "first_event_lag_ns": observation.coverage_start_utc_ns - int(segments[0][0]),
            "last_event_lead_ns": int(segments[-1][1]) - observation.coverage_end_utc_ns,
            "expected_segment_count": len(segments),
            "source_file_count": observation.source_file_count,
            "receipt_sha256": observation.receipt_sha256,
            "output_shard_sha256": observation.output_shard_sha256,
            "source_file_table_sha256": observation.source_file_table_sha256,
            "environment_receipt_sha256": observation.environment_receipt_sha256,
            "instrument_spec_sha256": observation.instrument_spec_sha256,
        })

    roots: dict[str, Any] = {}
    total_open = total_observed = total_missing = 0
    total_contract_days = total_trade_rows = total_quote_valid = 0
    for partition_root in sorted(partition_roots):
        partition, root = partition_root
        days = sorted(partition_roots[partition_root])
        open_days = [
            day for day in days
            if expected[(root, day)]["segments_utc_ns"]
        ]
        observed_days = {day for day in open_days if (root, day) in by_session}
        missing_days = [day for day in open_days if day not in observed_days]
        top_rows: list[int] = []
        contract_count = rows = quote_rows = negative = zero = source_files = 0
        volume = 0.0
        session_rows: list[dict[str, Any]] = []
        for day in open_days:
            values = sorted(
                by_session.get((root, day), []),
                key=lambda item: (-item.trade_rows, item.contract_id),
            )
            if values:
                top = values[0]
                top_rows.append(top.trade_rows)
                contract_count += len(values)
                rows += sum(value.trade_rows for value in values)
                quote_rows += sum(value.quote_valid_rows for value in values)
                negative += sum(value.negative_trade_rows for value in values)
                zero += sum(value.zero_trade_rows for value in values)
                source_files += sum(value.source_file_count for value in values)
                volume += sum(value.total_volume for value in values)
                session_rows.append({
                    "session_day": day,
                    "status": "observed",
                    "contract_count": len(values),
                    "top_contract_id": top.contract_id,
                    "top_contract_trade_rows": top.trade_rows,
                    "total_trade_rows": sum(value.trade_rows for value in values),
                })
            else:
                session_rows.append({
                    "session_day": day,
                    "status": "missing",
                    "contract_count": 0,
                    "top_contract_id": None,
                    "top_contract_trade_rows": 0,
                    "total_trade_rows": 0,
                })
        expected_open_count = len(open_days)
        observed_count = len(observed_days)
        metrics = {
            "partition_id": partition,
            "root": root,
            "denominator_days": len(days),
            "expected_open_sessions": expected_open_count,
            "observed_sessions": observed_count,
            "missing_open_sessions": len(missing_days),
            "coverage_fraction": (
                round(observed_count / expected_open_count, 8)
                if expected_open_count else 0.0
            ),
            "all_expected_open_sessions_observed": (
                expected_open_count > 0 and not missing_days
            ),
            "max_consecutive_missing_open_sessions": _max_missing_run(
                open_days, observed_days
            ),
            "first_observed_session": min(observed_days) if observed_days else None,
            "last_observed_session": max(observed_days) if observed_days else None,
            "contract_days_observed": contract_count,
            "total_trade_rows": rows,
            "total_quote_valid_rows": quote_rows,
            "quote_valid_fraction": round(quote_rows / rows, 8) if rows else 0.0,
            "negative_trade_rows": negative,
            "zero_trade_rows": zero,
            "total_volume": volume,
            "source_files_observed": source_files,
            "median_top_contract_trade_rows": float(median(top_rows)) if top_rows else 0.0,
            "p10_top_contract_trade_rows": _p10(top_rows),
            "missing_session_days": missing_days,
            "sessions": session_rows,
        }
        roots[f"{partition}:{root}"] = metrics
        total_open += expected_open_count
        total_observed += observed_count
        total_missing += len(missing_days)
        total_contract_days += contract_count
        total_trade_rows += rows
        total_quote_valid += quote_rows

    contract_days.sort(
        key=lambda row: (
            row["partition_id"], row["root"], row["session_day"], row["contract_id"],
        )
    )
    counts = {
        "denominator_rows": denominator_rows,
        "partition_root_pairs": len(roots),
        "expected_open_sessions": total_open,
        "observed_sessions": total_observed,
        "missing_open_sessions": total_missing,
        "contract_days_observed": total_contract_days,
        "trade_rows_observed": total_trade_rows,
        "quote_valid_rows_observed": total_quote_valid,
    }
    return {
        "counts": counts,
        "complete_against_denominator": total_open > 0 and total_missing == 0,
        "roots": roots,
        "contract_days": contract_days,
    }, expected


def _validate_index_identity(
    value: Any,
    *,
    exports: Sequence[VerifiedContractDayExport],
    contract_sha256: str,
) -> dict[str, Any]:
    identity = _exact_mapping(value, _INDEX_IDENTITY_KEYS, "export index identity")
    try:
        path = canonical_absolute_path(identity["path"], "export index identity path")
    except AuthorityBundleIOError as exc:
        raise _fail(str(exc)) from exc
    reopened, actual = load_sessionized_export_index(path)
    declared = {
        "path": path.as_posix(),
        "physical_sha256": require_sha256(
            identity["physical_sha256"], "export index physical SHA-256"
        ),
        "semantic_sha256": require_sha256(
            identity["semantic_sha256"], "export index semantic SHA-256"
        ),
        "entry_count": _strict_count(
            identity["entry_count"], "export index entry_count"
        ),
    }
    if declared != actual:
        raise _fail("export index identity changed before audit construction")
    if reopened["contract_sha256"] != contract_sha256:
        raise _fail("export index binds a different Corpus-v3 contract")
    expected_entries = [
        {
            "export_path": capability.export_path.as_posix(),
            "expected_request": {
                "root": capability.root,
                "contract_id": capability.contract_id,
                "session_day": capability.session_day,
                "split_use": capability.split_use,
                "purpose": "foundation_training",
            },
            "receipt_sha256": capability.receipt_sha256,
            "output_shard_sha256": capability.output_shard_sha256,
        }
        for capability in exports
    ]
    expected_entries.sort(
        key=lambda row: (*_request_key(row["expected_request"]), row["export_path"])
    )
    if reopened["entries"] != expected_entries:
        raise _fail("export index entries differ from verified export capabilities")
    return actual


def build_sessionized_coverage_audit(
    *,
    contract_path: str | Path,
    denominator: VerifiedSessionDenominatorBundleV2,
    exports: Sequence[VerifiedContractDayExport],
    export_index_identity: Mapping[str, Any],
    allow_test_contract: bool = False,
) -> dict[str, Any]:
    """Build a deterministic, non-authorizing audit from verified capabilities only."""
    if type(denominator) is not VerifiedSessionDenominatorBundleV2:
        raise _fail("a verified session-denominator-bundle capability is required")
    if denominator.production_admitted is not False:
        raise _fail("session denominator must remain explicitly production-blocked")
    contract_file = Path(contract_path).expanduser().resolve()
    contract = load_contract(contract_file)
    contract_sha = sha256_file(contract_file)
    observations: list[_ObservedContractDay] = []
    reverified_exports: list[VerifiedContractDayExport] = []
    for capability in exports:
        if (
            type(capability) is not VerifiedContractDayExport
            or not capability.is_authentic()
        ):
            raise _fail("a verified contract-day export capability is required")
        expected_request = {
            "root": capability.root,
            "contract_id": capability.contract_id,
            "session_day": capability.session_day,
            "split_use": capability.split_use,
            "purpose": "foundation_training",
        }
        try:
            reopened = verify_contract_day_export(
                capability.export_path,
                contract_path=contract_file,
                expected_request=expected_request,
                allow_test_contract=allow_test_contract,
            )
        except CorpusV3Error as exc:
            raise _fail(f"verified export changed before audit use: {exc}") from exc
        if (
            reopened.receipt_sha256 != capability.receipt_sha256
            or reopened.output_shard_sha256 != capability.output_shard_sha256
            or reopened.contract_sha256 != contract_sha
        ):
            raise _fail("verified export identity changed before audit use")
        reverified_exports.append(reopened)
        observations.append(_observation(reopened))
    identity = _validate_index_identity(
        export_index_identity,
        exports=reverified_exports,
        contract_sha256=contract_sha,
    )
    denominator_shards = list(iter_verified_session_shards(denominator))
    aggregate, _ = _aggregate_sessionized_coverage(
        denominator_shards=denominator_shards,
        observations=observations,
    )
    document: dict[str, Any] = {
        "schema_version": AUDIT_SCHEMA,
        "policy": AUDIT_POLICY,
        "purpose": "sessionized_coverage_and_liquidity_only_no_labels_predictions_or_outcomes",
        "contract": {
            "path": contract_file.as_posix(),
            "physical_sha256": contract_sha,
            "semantic_sha256": content_sha256(contract),
            "contract_id": contract["contract_id"],
        },
        "session_denominator": {
            "bundle_path": denominator.bundle_path.as_posix(),
            "manifest_physical_sha256": denominator.manifest_physical_sha256,
            "manifest_semantic_sha256": denominator.manifest_semantic_sha256,
            "calendar_rules_sha256": denominator.calendar_rules_sha256,
            "scope_v2_sha256": denominator.scope_v2_sha256,
            "consumer_scope_sha256": denominator.consumer_scope_sha256,
            "production_admitted": False,
        },
        "export_index": identity,
        "counts": aggregate["counts"],
        "complete_against_denominator": aggregate["complete_against_denominator"],
        "candidate_roots": sorted(contract["admitted_roots"]),
        "selected_roots": [],
        "selection_status": "blocked_pending_contract_lifecycle_inventory_and_scale_admission",
        "roots": aggregate["roots"],
        "contract_days": aggregate["contract_days"],
        "data_access": {
            "raw_paths_opened_by_audit": False,
            "verified_contract_day_capabilities_only": True,
            "strategy_events_read": False,
            "labels_read": False,
            "predictions_read": False,
            "returns_or_pnl_read": False,
            "legacy_holdout_outcomes_read": False,
        },
        "root_selection_authorized": False,
        "materialization_admitted": False,
        "training_admitted": False,
        "oos_admitted": False,
        "deployment_admitted": False,
    }
    document["audit_sha256"] = content_sha256(document)
    return document


def validate_sessionized_coverage_audit(
    value: Any,
    *,
    contract_path: str | Path,
    denominator: VerifiedSessionDenominatorBundleV2,
    exports: Sequence[VerifiedContractDayExport],
    export_index_identity: Mapping[str, Any],
    allow_test_contract: bool = False,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _fail("sessionized coverage audit must be an object")
    candidate = deepcopy(dict(value))
    supplied = candidate.pop("audit_sha256", None)
    if supplied != content_sha256(candidate):
        raise _fail("sessionized coverage audit integrity mismatch")
    expected = build_sessionized_coverage_audit(
        contract_path=contract_path,
        denominator=denominator,
        exports=exports,
        export_index_identity=export_index_identity,
        allow_test_contract=allow_test_contract,
    )
    if dict(value) != expected:
        raise _fail("sessionized coverage audit is stale or non-canonical")
    if any(bool(value.get(name)) for name in (
        "root_selection_authorized", "materialization_admitted", "training_admitted",
        "oos_admitted", "deployment_admitted",
    )):
        raise _fail("sessionized coverage audit cannot authorize downstream use")
    return deepcopy(expected)


def write_sessionized_coverage_audit(value: Mapping[str, Any], path: str | Path) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(target) + ".tmp")
    temporary.write_bytes(canonical_json_bytes(dict(value)))
    temporary.replace(target)
    return target


__all__ = [
    "AUDIT_POLICY", "AUDIT_SCHEMA", "CorpusV3SessionAuditError",
    "EXPORT_INDEX_PURPOSE", "EXPORT_INDEX_SCHEMA",
    "build_sessionized_coverage_audit", "load_sessionized_export_index",
    "load_verified_exports_from_index", "validate_sessionized_coverage_audit",
    "write_sessionized_coverage_audit",
]
