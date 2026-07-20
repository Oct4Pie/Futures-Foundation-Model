"""Independent FFM consumer for detached Corpus-v3 producer governance.

The accepted producer artifacts prove only canonical self-consistency, a frozen
exchange-session-day split/use protocol, and explicit denial of market-content
access.  They do not prove provider authenticity, source immutability, lifecycle,
session geometry, inventory, materialization admission, or model-training authority.

Every public use reopens its parent artifacts through the shared no-follow,
single-link authority transport.  Parsed dataclass values are never accepted as a
substitute for the physical artifacts that created them.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
import re
from typing import Any, Mapping

from ._authority_bundle_io import (
    AuthorityBundleIOError,
    canonical_absolute_path,
    content_sha256,
    read_canonical_json_file,
    require_sha256,
)


PRODUCER_SCHEMA_VERSION = "alphaforge_producer_governance_v1"
SPLIT_SCHEMA_VERSION = "alphaforge_frozen_split_use_contract_v1"
PRODUCER_PURPOSE = "readiness_governance_only"
SPLIT_PURPOSE = "frozen_split_and_use_readiness_contract"
PROTOCOL_ID = "ffm_corpus_v3_2011_2026_v1"
PROTOCOL_SOURCE_START = "2011-01-01"
PROTOCOL_SOURCE_MAX_DATE_EXCLUSIVE = "2026-07-01"
PARTITION_IDS = ("pretrain", "shared_train", "development", "legacy_holdout")
PROTOCOL_PARTITIONS = (
    ("pretrain", "2011-01-01", "2019-07-01"),
    ("shared_train", "2019-07-01", "2024-07-01"),
    ("development", "2024-07-01", "2025-07-01"),
    ("legacy_holdout", "2025-07-01", "2026-07-01"),
)
PERMITTED_USE_MATRIX = {
    "pretrain": ("foundation_pretraining", "self_supervised_training"),
    "shared_train": (
        "foundation_pretraining", "self_supervised_training", "supervised_training",
        "downstream_head_training", "train_only_calibration",
    ),
    "development": ("validation", "model_selection", "threshold_selection"),
    "legacy_holdout": (),
}
PRODUCER_COMPATIBILITY_COMMIT = "b84925763459c2f1a7f4300d11e9760867083629"
MAX_ARTIFACT_BYTES = 262_144
MAX_ARTIFACT_NODES = 1_024
_IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_PRODUCER_TOKEN = object()
_SPLIT_TOKEN = object()

_PRODUCER_FIELDS = {
    "schema_version", "purpose", "evidence_status", "production_admission",
    "filesystem_immutability_claimed", "signature_status", "source_namespace",
    "access_policy", "governance_semantic_sha256",
}
_NAMESPACE_FIELDS = {"provider_id", "source_id", "data_mode", "namespace_root"}
_ACCESS_FIELDS = {
    "prohibited_oos_content_access", "market_content_enumeration",
    "market_content_read", "allowed_readiness_operations",
    "boundary_container_policy", "separate_materialization_admission_required",
}
_SPLIT_FIELDS = {
    "schema_version", "purpose", "evidence_status", "production_admission",
    "parent_producer_governance", "source_namespace", "interval_semantics",
    "protocol_scope", "partitions", "permitted_use_matrix", "content_access_policy",
    "boundary_leaf_policy", "contract_semantic_sha256",
}
_PARENT_FIELDS = {"path", "physical_sha256", "semantic_sha256"}
_INTERVAL_FIELDS = {"calendar", "closure", "ordering", "gap_policy"}
_PROTOCOL_FIELDS = {"protocol_id", "source_start", "source_max_date_exclusive"}
_PARTITION_FIELDS = {"partition_id", "start", "end_exclusive"}
_CONTENT_ACCESS_FIELDS = {
    "verifier_market_content_access", "legacy_holdout_content_access",
    "legacy_holdout_metadata_access", "other_partition_content_access",
    "oos_training_validation_calibration_selection",
}
_BOUNDARY_FIELDS = {
    "policy", "whole_container_interval_required", "cross_partition_leaf_access",
    "partial_decode_or_hash",
}


class CorpusV3ProducerGovernanceError(ValueError):
    """Raised when detached producer governance fails independent verification."""


def _fail(message: str) -> CorpusV3ProducerGovernanceError:
    return CorpusV3ProducerGovernanceError(message)


@dataclass(frozen=True)
class PartitionV1:
    partition_id: str
    start: str
    end_exclusive: str


@dataclass(frozen=True)
class VerifiedProducerGovernanceV1:
    path: Path
    physical_sha256: str
    semantic_sha256: str
    provider_id: str
    source_id: str
    data_mode: str
    namespace_root: str
    evidence_status: str
    production_admitted: bool
    _token: object


@dataclass(frozen=True)
class VerifiedFrozenSplitUseContractV1:
    path: Path
    physical_sha256: str
    semantic_sha256: str
    producer: VerifiedProducerGovernanceV1
    partitions: tuple[PartitionV1, ...]
    permitted_uses: tuple[tuple[str, tuple[str, ...]], ...]
    boundary_leaf_policy: str
    evidence_status: str
    production_admitted: bool
    _token: object


def _exact_mapping(value: Any, fields: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        missing = sorted(fields - set(value)) if isinstance(value, Mapping) else sorted(fields)
        unknown = sorted(set(value) - fields) if isinstance(value, Mapping) else []
        raise _fail(f"{label} fields mismatch; missing={missing}, unknown={unknown}")
    return value


def _require_exact(value: Any, expected: Any, label: str) -> None:
    if type(value) is not type(expected) or value != expected:
        raise _fail(f"{label} must equal {expected!r}")


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise _fail(f"{label} is not a canonical identifier")
    return value


def _canonical_date(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise _fail(f"{label} must be a canonical ISO date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise _fail(f"{label} must be a canonical ISO date") from exc
    if parsed.isoformat() != value:
        raise _fail(f"{label} must be a canonical ISO date")
    return value


def _absolute_path(value: Any, label: str) -> Path:
    try:
        return canonical_absolute_path(value, label)
    except AuthorityBundleIOError as exc:
        raise _fail(str(exc)) from exc


def _sha(value: Any, label: str) -> str:
    try:
        return require_sha256(value, label)
    except AuthorityBundleIOError as exc:
        raise _fail(str(exc)) from exc


def _read(path: Path, *, label: str, expected_sha256: str) -> tuple[dict[str, Any], str]:
    expected = _sha(expected_sha256, f"expected {label} SHA-256")
    try:
        source, document, physical = read_canonical_json_file(
            path,
            label=label,
            max_bytes=MAX_ARTIFACT_BYTES,
            max_nodes=MAX_ARTIFACT_NODES,
            max_depth=12,
        )
    except AuthorityBundleIOError as exc:
        raise _fail(str(exc)) from exc
    if source != path or physical != expected:
        raise _fail(f"{label} physical SHA-256 differs from the expected authority")
    return document, physical


def _namespace(value: Any, label: str) -> tuple[str, str, str, str]:
    namespace = _exact_mapping(value, _NAMESPACE_FIELDS, label)
    return (
        _identifier(namespace["provider_id"], f"{label}.provider_id"),
        _identifier(namespace["source_id"], f"{label}.source_id"),
        _identifier(namespace["data_mode"], f"{label}.data_mode"),
        _absolute_path(namespace["namespace_root"], f"{label}.namespace_root").as_posix(),
    )


def _validate_producer(
    document: Mapping[str, Any],
    *,
    path: Path,
    physical_sha256: str,
) -> VerifiedProducerGovernanceV1:
    _exact_mapping(document, _PRODUCER_FIELDS, "producer governance")
    _require_exact(document["schema_version"], PRODUCER_SCHEMA_VERSION, "producer schema_version")
    _require_exact(document["purpose"], PRODUCER_PURPOSE, "producer purpose")
    _require_exact(
        document["evidence_status"], "detached_self_consistency_only",
        "producer evidence_status",
    )
    _require_exact(document["production_admission"], False, "producer production_admission")
    _require_exact(
        document["filesystem_immutability_claimed"], False,
        "producer filesystem_immutability_claimed",
    )
    _require_exact(document["signature_status"], "unsigned", "producer signature_status")
    provider_id, source_id, data_mode, namespace_root = _namespace(
        document["source_namespace"], "producer source_namespace"
    )
    access = _exact_mapping(document["access_policy"], _ACCESS_FIELDS, "producer access_policy")
    _require_exact(
        access["prohibited_oos_content_access"], True,
        "producer prohibited_oos_content_access",
    )
    _require_exact(
        access["market_content_enumeration"], "prohibited",
        "producer market_content_enumeration",
    )
    _require_exact(access["market_content_read"], "prohibited", "producer market_content_read")
    _require_exact(
        access["allowed_readiness_operations"],
        ["verify_detached_governance_artifacts"],
        "producer allowed_readiness_operations",
    )
    _require_exact(
        access["boundary_container_policy"], "boundary_blocked",
        "producer boundary_container_policy",
    )
    _require_exact(
        access["separate_materialization_admission_required"], True,
        "producer separate_materialization_admission_required",
    )
    semantic = _sha(
        document["governance_semantic_sha256"],
        "producer governance semantic SHA-256",
    )
    if content_sha256(dict(document), "governance_semantic_sha256") != semantic:
        raise _fail("producer governance semantic hash mismatch")
    return VerifiedProducerGovernanceV1(
        path=path,
        physical_sha256=physical_sha256,
        semantic_sha256=semantic,
        provider_id=provider_id,
        source_id=source_id,
        data_mode=data_mode,
        namespace_root=namespace_root,
        evidence_status="detached_self_consistency_only",
        production_admitted=False,
        _token=_PRODUCER_TOKEN,
    )


def _producer_once(path: Path, expected_sha256: str) -> VerifiedProducerGovernanceV1:
    document, physical = _read(
        path, label="Corpus-v3 producer governance", expected_sha256=expected_sha256
    )
    return _validate_producer(document, path=path, physical_sha256=physical)


def load_and_verify_producer_governance_v1(
    path: str | Path,
    *,
    expected_sha256: str,
) -> VerifiedProducerGovernanceV1:
    """Load twice and return a non-forgeable, production-blocked capability."""
    source = _absolute_path(path, "producer governance path")
    first = _producer_once(source, expected_sha256)
    second = _producer_once(source, expected_sha256)
    if first != second:
        raise _fail("producer governance changed across mandatory reopen")
    return first


def reopen_and_verify_producer_governance_v1(
    capability: VerifiedProducerGovernanceV1,
) -> VerifiedProducerGovernanceV1:
    if (
        type(capability) is not VerifiedProducerGovernanceV1
        or capability._token is not _PRODUCER_TOKEN
        or capability.production_admitted is not False
    ):
        raise _fail("a verified production-blocked producer-governance capability is required")
    reopened = load_and_verify_producer_governance_v1(
        capability.path, expected_sha256=capability.physical_sha256
    )
    if reopened != capability:
        raise _fail("producer governance changed before use")
    return reopened


def _validate_split(
    document: Mapping[str, Any],
    *,
    path: Path,
    physical_sha256: str,
    producer: VerifiedProducerGovernanceV1,
) -> VerifiedFrozenSplitUseContractV1:
    _exact_mapping(document, _SPLIT_FIELDS, "frozen split/use contract")
    _require_exact(document["schema_version"], SPLIT_SCHEMA_VERSION, "split schema_version")
    _require_exact(document["purpose"], SPLIT_PURPOSE, "split purpose")
    _require_exact(
        document["evidence_status"], "detached_self_consistency_only",
        "split evidence_status",
    )
    _require_exact(document["production_admission"], False, "split production_admission")
    parent = _exact_mapping(
        document["parent_producer_governance"], _PARENT_FIELDS,
        "split parent_producer_governance",
    )
    if _absolute_path(parent["path"], "split parent path") != producer.path:
        raise _fail("split contract parent path mismatch")
    if (
        _sha(parent["physical_sha256"], "split parent physical SHA-256")
        != producer.physical_sha256
        or _sha(parent["semantic_sha256"], "split parent semantic SHA-256")
        != producer.semantic_sha256
    ):
        raise _fail("split contract parent hash mismatch")
    observed_namespace = _namespace(document["source_namespace"], "split source_namespace")
    expected_namespace = (
        producer.provider_id, producer.source_id, producer.data_mode,
        producer.namespace_root,
    )
    if observed_namespace != expected_namespace:
        raise _fail("split contract source namespace differs from producer governance")
    interval = _exact_mapping(
        document["interval_semantics"], _INTERVAL_FIELDS, "split interval_semantics"
    )
    _require_exact(interval["calendar"], "exchange_session_day", "split interval calendar")
    _require_exact(
        interval["closure"], "half_open_start_inclusive_end_exclusive",
        "split interval closure",
    )
    _require_exact(interval["ordering"], "strict_ascending", "split interval ordering")
    _require_exact(interval["gap_policy"], "no_gaps", "split interval gap_policy")
    scope = _exact_mapping(document["protocol_scope"], _PROTOCOL_FIELDS, "split protocol_scope")
    _require_exact(scope["protocol_id"], PROTOCOL_ID, "split protocol_id")
    _require_exact(scope["source_start"], PROTOCOL_SOURCE_START, "split source_start")
    _require_exact(
        scope["source_max_date_exclusive"], PROTOCOL_SOURCE_MAX_DATE_EXCLUSIVE,
        "split source_max_date_exclusive",
    )
    rows = document["partitions"]
    if not isinstance(rows, list) or len(rows) != len(PROTOCOL_PARTITIONS):
        raise _fail("split partitions must contain exactly four rows")
    partitions: list[PartitionV1] = []
    prior_end: str | None = None
    for index, (row, expected) in enumerate(zip(rows, PROTOCOL_PARTITIONS)):
        expected_id, expected_start, expected_end = expected
        parsed = _exact_mapping(row, _PARTITION_FIELDS, f"split partition {index}")
        _require_exact(parsed["partition_id"], expected_id, f"partition {index}.partition_id")
        start = _canonical_date(parsed["start"], f"partition {expected_id}.start")
        end = _canonical_date(parsed["end_exclusive"], f"partition {expected_id}.end_exclusive")
        if start >= end:
            raise _fail(f"partition {expected_id} is not positive half-open")
        if prior_end is not None and start != prior_end:
            raise _fail("split partitions must be contiguous and nonoverlapping")
        if (start, end) != (expected_start, expected_end):
            raise _fail(f"partition {expected_id} differs from the frozen protocol boundary")
        prior_end = end
        partitions.append(PartitionV1(expected_id, start, end))
    if (
        partitions[0].start != PROTOCOL_SOURCE_START
        or partitions[-1].end_exclusive != PROTOCOL_SOURCE_MAX_DATE_EXCLUSIVE
    ):
        raise _fail("split partition coverage differs from the frozen protocol scope")
    matrix = _exact_mapping(
        document["permitted_use_matrix"], set(PARTITION_IDS),
        "split permitted_use_matrix",
    )
    for partition_id in PARTITION_IDS:
        _require_exact(
            matrix[partition_id], list(PERMITTED_USE_MATRIX[partition_id]),
            f"split permitted_use_matrix.{partition_id}",
        )
    content = _exact_mapping(
        document["content_access_policy"], _CONTENT_ACCESS_FIELDS,
        "split content_access_policy",
    )
    _require_exact(
        content["verifier_market_content_access"], "prohibited",
        "split verifier_market_content_access",
    )
    _require_exact(
        content["legacy_holdout_content_access"], "prohibited",
        "split legacy_holdout_content_access",
    )
    _require_exact(
        content["legacy_holdout_metadata_access"], "contract_dates_only",
        "split legacy_holdout_metadata_access",
    )
    _require_exact(
        content["other_partition_content_access"],
        "requires_separate_materialization_admission",
        "split other_partition_content_access",
    )
    _require_exact(
        content["oos_training_validation_calibration_selection"], "prohibited",
        "split oos_training_validation_calibration_selection",
    )
    boundary = _exact_mapping(
        document["boundary_leaf_policy"], _BOUNDARY_FIELDS,
        "split boundary_leaf_policy",
    )
    _require_exact(boundary["policy"], "boundary_blocked", "split boundary policy")
    _require_exact(
        boundary["whole_container_interval_required"], True,
        "split whole_container_interval_required",
    )
    _require_exact(
        boundary["cross_partition_leaf_access"], "prohibited",
        "split cross_partition_leaf_access",
    )
    _require_exact(
        boundary["partial_decode_or_hash"], "prohibited",
        "split partial_decode_or_hash",
    )
    semantic = _sha(
        document["contract_semantic_sha256"], "split contract semantic SHA-256"
    )
    if content_sha256(dict(document), "contract_semantic_sha256") != semantic:
        raise _fail("split/use contract semantic hash mismatch")
    return VerifiedFrozenSplitUseContractV1(
        path=path,
        physical_sha256=physical_sha256,
        semantic_sha256=semantic,
        producer=producer,
        partitions=tuple(partitions),
        permitted_uses=tuple(
            (partition_id, PERMITTED_USE_MATRIX[partition_id])
            for partition_id in PARTITION_IDS
        ),
        boundary_leaf_policy="boundary_blocked",
        evidence_status="detached_self_consistency_only",
        production_admitted=False,
        _token=_SPLIT_TOKEN,
    )


def _split_once(
    path: Path,
    *,
    expected_sha256: str,
    producer: VerifiedProducerGovernanceV1,
) -> VerifiedFrozenSplitUseContractV1:
    document, physical = _read(
        path, label="Corpus-v3 frozen split/use contract", expected_sha256=expected_sha256
    )
    return _validate_split(
        document, path=path, physical_sha256=physical, producer=producer
    )


def load_and_verify_frozen_split_use_contract_v1(
    path: str | Path,
    *,
    expected_sha256: str,
    producer_governance_path: str | Path,
    producer_governance_sha256: str,
) -> VerifiedFrozenSplitUseContractV1:
    """Load both artifacts twice and return a non-forgeable blocked capability."""
    split_path = _absolute_path(path, "frozen split/use contract path")
    producer = load_and_verify_producer_governance_v1(
        producer_governance_path,
        expected_sha256=producer_governance_sha256,
    )
    first = _split_once(
        split_path, expected_sha256=expected_sha256, producer=producer
    )
    reopened_producer = reopen_and_verify_producer_governance_v1(producer)
    second = _split_once(
        split_path, expected_sha256=expected_sha256, producer=reopened_producer
    )
    if first != second:
        raise _fail("frozen split/use contract changed across mandatory reopen")
    return first


def reopen_and_verify_frozen_split_use_contract_v1(
    capability: VerifiedFrozenSplitUseContractV1,
) -> VerifiedFrozenSplitUseContractV1:
    if (
        type(capability) is not VerifiedFrozenSplitUseContractV1
        or capability._token is not _SPLIT_TOKEN
        or capability.production_admitted is not False
    ):
        raise _fail("a verified production-blocked split/use capability is required")
    reopened = load_and_verify_frozen_split_use_contract_v1(
        capability.path,
        expected_sha256=capability.physical_sha256,
        producer_governance_path=capability.producer.path,
        producer_governance_sha256=capability.producer.physical_sha256,
    )
    if reopened != capability:
        raise _fail("frozen split/use contract changed before use")
    return reopened


def evaluate_session_request_v1(
    capability: VerifiedFrozenSplitUseContractV1,
    *,
    partition_id: str,
    requested_use: str,
    session_day: str,
) -> str:
    """Evaluate split/use eligibility after mandatory artifact reopen."""
    verified = reopen_and_verify_frozen_split_use_contract_v1(capability)
    day = _canonical_date(session_day, "session_day")
    actual = next(
        (
            partition for partition in verified.partitions
            if partition.start <= day < partition.end_exclusive
        ),
        None,
    )
    uses = dict(verified.permitted_uses)
    if (
        actual is None
        or partition_id != actual.partition_id
        or requested_use not in uses.get(partition_id, ())
    ):
        return "boundary_blocked"
    return "eligible_by_split_use_contract_not_content_authorized"


def evaluate_boundary_leaf_v1(
    capability: VerifiedFrozenSplitUseContractV1,
    *,
    partition_id: str,
    requested_use: str,
    session_day: str,
    interval_start_utc_ns: int,
    interval_end_exclusive_utc_ns: int,
) -> str:
    """Reject or require an official session denominator; never inspect a leaf."""
    if (
        type(interval_start_utc_ns) is not int
        or type(interval_end_exclusive_utc_ns) is not int
        or interval_start_utc_ns < 0
        or interval_start_utc_ns >= interval_end_exclusive_utc_ns
    ):
        raise _fail("leaf interval must be positive half-open")
    status = evaluate_session_request_v1(
        capability,
        partition_id=partition_id,
        requested_use=requested_use,
        session_day=session_day,
    )
    if status != "eligible_by_split_use_contract_not_content_authorized":
        return "boundary_blocked"
    return "requires_session_denominator"


__all__ = [
    "PARTITION_IDS", "PERMITTED_USE_MATRIX", "PRODUCER_COMPATIBILITY_COMMIT",
    "PRODUCER_SCHEMA_VERSION", "PROTOCOL_ID", "PROTOCOL_PARTITIONS",
    "PROTOCOL_SOURCE_MAX_DATE_EXCLUSIVE", "PROTOCOL_SOURCE_START",
    "SPLIT_SCHEMA_VERSION", "CorpusV3ProducerGovernanceError", "PartitionV1",
    "VerifiedFrozenSplitUseContractV1", "VerifiedProducerGovernanceV1",
    "evaluate_boundary_leaf_v1", "evaluate_session_request_v1",
    "load_and_verify_frozen_split_use_contract_v1",
    "load_and_verify_producer_governance_v1",
    "reopen_and_verify_frozen_split_use_contract_v1",
    "reopen_and_verify_producer_governance_v1",
]
