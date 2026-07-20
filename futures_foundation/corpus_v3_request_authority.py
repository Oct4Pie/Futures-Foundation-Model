"""Non-forgeable request membership derived from the Corpus-v3 plan chain.

The capability binds three canonical artifacts:

    expected-request denominator -> split-scoped inventory -> materialization plan

It does not authorize source reads, materialization, training, OOS use, or deployment.
Consumers may use it only to prove that already supplied rows belong to one exact
planned ``(partition, root, contract, session segment, use)`` interval.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ._authority_bundle_io import (
    AuthorityBundleIOError,
    read_canonical_json_file,
    require_sha256,
)
from .corpus_v3_expected_requests import SCHEMA_VERSION as EXPECTED_SCHEMA
from .corpus_v3_materialization_plan import (
    INVENTORY_SCHEMA,
    PLAN_SCHEMA,
    validate_materialization_plan_v1,
    validate_split_scoped_inventory_v1,
)


SCHEMA_VERSION = "ffm_corpus_v3_request_authority_v1"
MAX_ARTIFACT_BYTES = 768 * 1024 * 1024
MAX_ARTIFACT_NODES = 10_000_000
_CAPABILITY_TOKEN = object()


class CorpusV3RequestAuthorityError(ValueError):
    """Raised when the request authority chain is stale, unsafe, or ambiguous."""


def _fail(message: str) -> CorpusV3RequestAuthorityError:
    return CorpusV3RequestAuthorityError(message)


@dataclass(frozen=True)
class PlannedRequestV1:
    request_semantic_sha256: str
    partition_id: str
    root: str
    permitted_uses: tuple[str, ...]
    provider_instrument_id: str
    provider_symbol: str
    contract_id: str
    venue: str
    session_day: str
    session_segment_index: int
    request_start_utc_ns: int
    request_end_exclusive_utc_ns: int


@dataclass(frozen=True)
class VerifiedRequestAuthorityV1:
    expected_path: Path
    expected_physical_sha256: str
    expected_semantic_sha256: str
    inventory_path: Path
    inventory_physical_sha256: str
    inventory_semantic_sha256: str
    plan_path: Path
    plan_physical_sha256: str
    plan_semantic_sha256: str
    requests: tuple[PlannedRequestV1, ...]
    production_admitted: bool
    materialization_admitted: bool
    training_admitted: bool
    oos_admitted: bool
    _token: object

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "expected_request_denominator": {
                "path": self.expected_path.as_posix(),
                "physical_sha256": self.expected_physical_sha256,
                "semantic_sha256": self.expected_semantic_sha256,
            },
            "split_scoped_inventory": {
                "path": self.inventory_path.as_posix(),
                "physical_sha256": self.inventory_physical_sha256,
                "semantic_sha256": self.inventory_semantic_sha256,
            },
            "materialization_plan": {
                "path": self.plan_path.as_posix(),
                "physical_sha256": self.plan_physical_sha256,
                "semantic_sha256": self.plan_semantic_sha256,
            },
            "selected_request_count": len(self.requests),
            "production_admitted": False,
            "materialization_admitted": False,
            "training_admitted": False,
            "oos_admitted": False,
        }


def _read(
    path: str | Path,
    *,
    expected_sha256: str,
    label: str,
) -> tuple[Path, dict[str, Any], str]:
    try:
        source, document, physical = read_canonical_json_file(
            Path(path).expanduser().resolve(),
            label=label,
            max_bytes=MAX_ARTIFACT_BYTES,
            max_nodes=MAX_ARTIFACT_NODES,
            max_depth=24,
        )
    except AuthorityBundleIOError as exc:
        raise _fail(str(exc)) from exc
    expected = require_sha256(expected_sha256, f"expected {label} SHA-256")
    if physical != expected:
        raise _fail(f"{label} physical SHA-256 differs from authority")
    return source, document, physical


def _expected_document(value: Mapping[str, Any]) -> dict[str, Any]:
    if (
        value.get("schema_version") != EXPECTED_SCHEMA
        or value.get("production_admitted") is not False
        or value.get("materialization_admitted") is not False
        or value.get("training_admitted") is not False
        or value.get("oos_admitted") is not False
    ):
        raise _fail("expected-request denominator is unsupported or authorizing")
    semantic = require_sha256(
        value.get("expected_request_denominator_sha256"),
        "expected-request denominator semantic SHA-256",
    )
    from ._authority_bundle_io import content_sha256

    if content_sha256(dict(value), "expected_request_denominator_sha256") != semantic:
        raise _fail("expected-request denominator semantic hash mismatch")
    return dict(value)


def _request(value: Mapping[str, Any], index: int) -> PlannedRequestV1:
    fields = {
        "request_semantic_sha256", "partition_id", "root", "permitted_uses",
        "candidate_index", "provider_instrument_id", "provider_symbol", "contract_id",
        "venue", "session_day", "session_segment_index", "request_start_utc_ns",
        "request_end_exclusive_utc_ns", "source_files",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise _fail(f"materialization plan request {index} has an invalid exact schema")
    uses = value["permitted_uses"]
    if (
        not isinstance(uses, list)
        or not uses
        or uses != sorted(set(uses))
        or any(not isinstance(item, str) or not item for item in uses)
    ):
        raise _fail("materialization plan request permitted_uses are invalid")
    text_fields = (
        "partition_id", "root", "provider_instrument_id", "provider_symbol",
        "contract_id", "venue", "session_day",
    )
    if any(not isinstance(value[field], str) or not value[field] for field in text_fields):
        raise _fail("materialization plan request identity contains blank text")
    segment = value["session_segment_index"]
    start = value["request_start_utc_ns"]
    end = value["request_end_exclusive_utc_ns"]
    if (
        type(segment) is not int
        or segment < 0
        or type(start) is not int
        or type(end) is not int
        or start < 0
        or start >= end
    ):
        raise _fail("materialization plan request interval is invalid")
    return PlannedRequestV1(
        request_semantic_sha256=require_sha256(
            value["request_semantic_sha256"],
            f"materialization plan request {index} SHA-256",
        ),
        partition_id=value["partition_id"],
        root=value["root"],
        permitted_uses=tuple(uses),
        provider_instrument_id=value["provider_instrument_id"],
        provider_symbol=value["provider_symbol"],
        contract_id=value["contract_id"],
        venue=value["venue"],
        session_day=value["session_day"],
        session_segment_index=segment,
        request_start_utc_ns=start,
        request_end_exclusive_utc_ns=end,
    )


def _load_once(
    *,
    expected_path: str | Path,
    expected_physical_sha256: str,
    inventory_path: str | Path,
    inventory_physical_sha256: str,
    plan_path: str | Path,
    plan_physical_sha256: str,
) -> VerifiedRequestAuthorityV1:
    expected_source, expected_raw, expected_physical = _read(
        expected_path,
        expected_sha256=expected_physical_sha256,
        label="Corpus-v3 expected-request denominator",
    )
    expected = _expected_document(expected_raw)
    inventory_source, inventory_raw, inventory_physical = _read(
        inventory_path,
        expected_sha256=inventory_physical_sha256,
        label="Corpus-v3 split-scoped inventory",
    )
    if inventory_raw.get("schema_version") != INVENTORY_SCHEMA:
        raise _fail("split-scoped inventory schema mismatch")
    inventory = validate_split_scoped_inventory_v1(
        inventory_raw,
        expected_request_denominator=expected,
    )
    plan_source, plan_raw, plan_physical = _read(
        plan_path,
        expected_sha256=plan_physical_sha256,
        label="Corpus-v3 materialization plan",
    )
    if plan_raw.get("schema_version") != PLAN_SCHEMA:
        raise _fail("materialization plan schema mismatch")
    plan = validate_materialization_plan_v1(
        plan_raw,
        expected_request_denominator=expected,
        inventory=inventory,
    )
    if any(
        plan.get(field) is not False
        for field in (
            "production_admitted", "materialization_admitted", "training_admitted",
            "oos_admitted",
        )
    ):
        raise _fail("request authority parents cannot grant admission")
    selected = plan.get("selected_requests")
    if not isinstance(selected, list):
        raise _fail("materialization plan selected_requests must be a list")
    requests = tuple(_request(row, index) for index, row in enumerate(selected))
    keys = [row.request_semantic_sha256 for row in requests]
    if len(keys) != len(set(keys)):
        raise _fail("materialization plan selected requests are duplicated")
    canonical_order = sorted(
        requests,
        key=lambda row: (
            row.partition_id, row.root, row.contract_id, row.session_day,
            row.session_segment_index, row.request_start_utc_ns,
            row.request_semantic_sha256,
        ),
    )
    if list(requests) != canonical_order:
        raise _fail("materialization plan selected requests are not canonically ordered")
    by_contract: dict[tuple[str, str], list[PlannedRequestV1]] = {}
    for request in requests:
        by_contract.setdefault((request.root, request.contract_id), []).append(request)
    for contract_requests in by_contract.values():
        ordered = sorted(
            contract_requests,
            key=lambda row: (
                row.request_start_utc_ns, row.request_end_exclusive_utc_ns,
                row.request_semantic_sha256,
            ),
        )
        for previous, current in zip(ordered, ordered[1:]):
            if (
                current.request_start_utc_ns < previous.request_end_exclusive_utc_ns
                and set(current.permitted_uses) & set(previous.permitted_uses)
            ):
                raise _fail(
                    "materialization plan selected requests overlap for one contract/use"
                )
    return VerifiedRequestAuthorityV1(
        expected_path=expected_source,
        expected_physical_sha256=expected_physical,
        expected_semantic_sha256=expected["expected_request_denominator_sha256"],
        inventory_path=inventory_source,
        inventory_physical_sha256=inventory_physical,
        inventory_semantic_sha256=inventory["inventory_sha256"],
        plan_path=plan_source,
        plan_physical_sha256=plan_physical,
        plan_semantic_sha256=plan["plan_sha256"],
        requests=requests,
        production_admitted=False,
        materialization_admitted=False,
        training_admitted=False,
        oos_admitted=False,
        _token=_CAPABILITY_TOKEN,
    )


def load_and_verify_request_authority_v1(
    *,
    expected_path: str | Path,
    expected_physical_sha256: str,
    inventory_path: str | Path,
    inventory_physical_sha256: str,
    plan_path: str | Path,
    plan_physical_sha256: str,
) -> VerifiedRequestAuthorityV1:
    """Load every parent twice and return a non-forgeable membership capability."""
    kwargs = dict(
        expected_path=expected_path,
        expected_physical_sha256=expected_physical_sha256,
        inventory_path=inventory_path,
        inventory_physical_sha256=inventory_physical_sha256,
        plan_path=plan_path,
        plan_physical_sha256=plan_physical_sha256,
    )
    first = _load_once(**kwargs)
    second = _load_once(**kwargs)
    if first != second:
        raise _fail("request authority changed across mandatory reopen")
    return first


def require_request_authority_v1(
    capability: VerifiedRequestAuthorityV1,
) -> VerifiedRequestAuthorityV1:
    if (
        type(capability) is not VerifiedRequestAuthorityV1
        or capability._token is not _CAPABILITY_TOKEN
        or any((
            capability.production_admitted,
            capability.materialization_admitted,
            capability.training_admitted,
            capability.oos_admitted,
        ))
    ):
        raise _fail("a verified production-blocked request authority is required")
    reopened = load_and_verify_request_authority_v1(
        expected_path=capability.expected_path,
        expected_physical_sha256=capability.expected_physical_sha256,
        inventory_path=capability.inventory_path,
        inventory_physical_sha256=capability.inventory_physical_sha256,
        plan_path=capability.plan_path,
        plan_physical_sha256=capability.plan_physical_sha256,
    )
    if reopened != capability:
        raise _fail("request authority changed before use")
    return reopened


def load_request_authority_manifest_v1(value: Any) -> VerifiedRequestAuthorityV1:
    if not isinstance(value, Mapping) or value.get("schema_version") != SCHEMA_VERSION:
        raise _fail("request authority manifest schema mismatch")
    expected = value.get("expected_request_denominator")
    inventory = value.get("split_scoped_inventory")
    plan = value.get("materialization_plan")
    if not all(isinstance(item, Mapping) for item in (expected, inventory, plan)):
        raise _fail("request authority manifest parents are malformed")
    capability = load_and_verify_request_authority_v1(
        expected_path=expected.get("path"),
        expected_physical_sha256=expected.get("physical_sha256"),
        inventory_path=inventory.get("path"),
        inventory_physical_sha256=inventory.get("physical_sha256"),
        plan_path=plan.get("path"),
        plan_physical_sha256=plan.get("physical_sha256"),
    )
    if capability.manifest() != dict(value):
        raise _fail("request authority manifest is stale or non-canonical")
    return capability


def request_segment_ids_v1(
    capability: VerifiedRequestAuthorityV1,
    *,
    root: str,
    requested_use: str,
    timestamps_ns: Any,
    contract_ids: Any,
) -> np.ndarray:
    """Return canonical selected-request IDs, with ``-1`` for unauthorized rows."""
    verified = require_request_authority_v1(capability)
    root = str(root).strip().upper()
    requested_use = str(requested_use).strip()
    if not root or not requested_use:
        raise _fail("root and requested_use must be non-empty")
    timestamps = np.asarray(timestamps_ns)
    contracts = np.asarray(contract_ids)
    if timestamps.ndim != 1 or contracts.ndim != 1 or timestamps.shape != contracts.shape:
        raise _fail("timestamps and contract_ids must be aligned one-dimensional arrays")
    if timestamps.dtype.kind not in "iu":
        raise _fail("timestamps must be integer UTC nanoseconds")
    timestamps = timestamps.astype(np.int64, copy=False)
    if np.any(timestamps < 0):
        raise _fail("timestamps must be nonnegative UTC nanoseconds")
    contracts = contracts.astype(str)
    if np.any(np.char.str_len(np.char.strip(contracts)) == 0):
        raise _fail("contract_ids contain blank values")
    result = np.full(len(timestamps), -1, dtype=np.int64)
    for segment_id, request in enumerate(verified.requests):
        if (
            request.root != root
            or requested_use not in request.permitted_uses
        ):
            continue
        match = (
            (contracts == request.contract_id)
            & (timestamps >= request.request_start_utc_ns)
            & (timestamps < request.request_end_exclusive_utc_ns)
        )
        if np.any(match & (result >= 0)):
            raise _fail("one stream row matches multiple selected request intervals")
        result[match] = segment_id
    return result


__all__ = [
    "SCHEMA_VERSION", "CorpusV3RequestAuthorityError", "PlannedRequestV1",
    "VerifiedRequestAuthorityV1", "load_and_verify_request_authority_v1",
    "load_request_authority_manifest_v1", "request_segment_ids_v1",
    "require_request_authority_v1",
]
