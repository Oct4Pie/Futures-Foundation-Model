"""Independent FFM consumer for synthetic contract-lifecycle evidence.

The lifecycle chain joins four independently reverified parents:

* detached producer governance and frozen split/use capability;
* metadata-only provider candidate pagination capability;
* a canonical provider-universe artifact reproducing that capability;
* a claim-scoped official lifecycle registry.

The registry and lifecycle artifact are mechanism evidence only.  Official-source
authenticity is explicitly detached and unproven, so every result remains
production-blocked.  No market file, observed activity, plan, materialization,
liquidity or month-code inference is accepted.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import hashlib
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
from .corpus_v3_producer_governance import (
    VerifiedFrozenSplitUseContractV1,
    reopen_and_verify_frozen_split_use_contract_v1,
)
from .corpus_v3_provider_candidates import (
    IMMUTABLE_MOUNT_BLOCKER,
    PRODUCTION_BLOCKER,
    UPSTREAM_AUTHORITY_BLOCKER,
    ProviderCandidateV1,
    VerifiedProviderCandidateUniverseV1,
    reopen_and_verify_provider_candidate_universe_v1,
)


REGISTRY_SCHEMA = "alphaforge_official_lifecycle_evidence_registry_v1"
LIFECYCLE_SCHEMA = "alphaforge_contract_lifecycle_capability_v2"
PROVIDER_UNIVERSE_SCHEMA = "alphaforge_provider_candidate_universe_v1"
SCOPE_SEMANTICS = "frozen_protocol_date_midnight_utc_envelope_not_session_denominator"
PRODUCER_COMPATIBILITY_COMMIT = "b84925763459c2f1a7f4300d11e9760867083629"
MAX_BYTES = 4 * 1024 * 1024
MAX_NODES = 100_000
MAX_ROWS = 1_000_000
_PROVIDER_BLOCKERS = (
    PRODUCTION_BLOCKER,
    UPSTREAM_AUTHORITY_BLOCKER,
    IMMUTABLE_MOUNT_BLOCKER,
)
LIFECYCLE_BLOCKERS = (
    "producer_governance_unsigned_and_not_immutable",
    "provider_candidate_authenticity_unproven",
    "official_lifecycle_evidence_authenticity_unproven",
    "production_admission_unavailable",
)
_FORBIDDEN_TERMS = {
    "plan", "plan_hash", "materialization", "materialization_plan", "file",
    "filename", "file_path", "lake", "tick", "ticks", "price", "volume",
    "activity", "observed", "first_trade", "last_trade", "first_file",
    "last_file", "row_count_observed",
}
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_SYMBOL_RE = re.compile(r"^[A-Z0-9][A-Z0-9._:/-]{0,63}$")
_CAPABILITY_TOKEN = object()

_UNIVERSE_FIELDS = {
    "schema_version", "purpose", "provider_id", "provider_api_revision", "scope_sha256",
    "claimed_producer_governance_sha256", "claimed_frozen_split_use_contract_sha256",
    "bundle_manifest_sha256", "bundle_manifest_semantic_sha256",
    "prohibited_root_boundary_sha256", "query_semantic_sha256",
    "pagination_semantic_sha256", "page_receipt_sha256s", "provider_response_sha256s",
    "candidate_count", "candidates", "production_admitted", "admission_blockers",
    "universe_semantic_sha256",
}
_CANDIDATE_FIELDS = {
    "provider_instrument_id", "provider_symbol", "root_symbol", "venue",
    "instrument_class",
}
_REGISTRY_FIELDS = {
    "schema_version", "purpose", "evidence_status", "production_admission",
    "provider_id", "scope_semantics", "scope_start_utc_ns", "scope_end_utc_ns",
    "official_sources", "claims", "registry_semantic_sha256",
}
_SOURCE_FIELDS = {
    "source_id", "authority", "evidence_physical_sha256",
    "evidence_semantic_sha256", "authenticity_status",
}
_CLAIM_FIELDS = {
    "claim_id", "source_id", "provider_instrument_id", "provider_symbol",
    "root", "venue", "claim_kind", "claim_utc_ns",
}
_LIFECYCLE_FIELDS = {
    "schema_version", "purpose", "evidence_status", "production_admission",
    "admission_blockers", "parent_artifacts", "scope_semantics",
    "scope_start_utc_ns", "scope_end_utc_ns", "row_count", "rows",
    "lifecycle_semantic_sha256",
}
_PARENT_FIELDS = {"path", "physical_sha256", "semantic_sha256"}
_PARENT_NAMES = {
    "producer_governance", "frozen_split_use_contract",
    "provider_candidate_universe", "official_lifecycle_evidence_registry",
}
_ROW_FIELDS = {
    "provider_instrument_id", "provider_symbol", "root", "contract_id", "venue",
    "start_kind", "eligibility_start_utc_ns", "end_kind",
    "trading_end_exclusive_utc_ns", "official_source_ids", "disposition",
}


class CorpusV3ContractLifecycleError(ValueError):
    """Raised when lifecycle evidence is unsafe, incomplete or non-canonical."""


def _fail(message: str) -> CorpusV3ContractLifecycleError:
    return CorpusV3ContractLifecycleError(message)


@dataclass(frozen=True)
class LifecycleRowV2:
    provider_instrument_id: str
    provider_symbol: str
    root: str
    venue: str
    contract_id: str
    start_kind: str | None
    eligibility_start_utc_ns: int | None
    end_kind: str | None
    trading_end_exclusive_utc_ns: int | None
    official_source_ids: tuple[str, ...]
    disposition: str


@dataclass(frozen=True)
class VerifiedContractLifecycleV2:
    lifecycle_path: Path
    lifecycle_physical_sha256: str
    lifecycle_semantic_sha256: str
    registry_path: Path
    registry_physical_sha256: str
    registry_semantic_sha256: str
    provider_universe_path: Path
    provider_universe_physical_sha256: str
    provider_universe_semantic_sha256: str
    split: VerifiedFrozenSplitUseContractV1
    provider_candidates: VerifiedProviderCandidateUniverseV1
    rows: tuple[LifecycleRowV2, ...]
    evidence_status: str
    production_admitted: bool
    admission_blockers: tuple[str, ...]
    _token: object

    def document(self) -> dict[str, Any]:
        return {
            "schema_version": "ffm_verified_contract_lifecycle_v2",
            "lifecycle": {
                "path": self.lifecycle_path.as_posix(),
                "physical_sha256": self.lifecycle_physical_sha256,
                "semantic_sha256": self.lifecycle_semantic_sha256,
            },
            "registry": {
                "path": self.registry_path.as_posix(),
                "physical_sha256": self.registry_physical_sha256,
                "semantic_sha256": self.registry_semantic_sha256,
            },
            "provider_universe": {
                "path": self.provider_universe_path.as_posix(),
                "physical_sha256": self.provider_universe_physical_sha256,
                "semantic_sha256": self.provider_universe_semantic_sha256,
            },
            "row_count": len(self.rows),
            "rows": [
                {
                    "provider_instrument_id": row.provider_instrument_id,
                    "provider_symbol": row.provider_symbol,
                    "root": row.root,
                    "venue": row.venue,
                    "contract_id": row.contract_id,
                    "start_kind": row.start_kind,
                    "eligibility_start_utc_ns": row.eligibility_start_utc_ns,
                    "end_kind": row.end_kind,
                    "trading_end_exclusive_utc_ns": row.trading_end_exclusive_utc_ns,
                    "official_source_ids": list(row.official_source_ids),
                    "disposition": row.disposition,
                }
                for row in self.rows
            ],
            "evidence_status": self.evidence_status,
            "production_admitted": self.production_admitted,
            "admission_blockers": list(self.admission_blockers),
            "market_data_read": False,
            "materialization_admitted": False,
            "training_admitted": False,
        }


def _exact(value: Any, fields: set[str], label: str) -> Mapping[str, Any]:
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
    if not isinstance(value, str) or _ID_RE.fullmatch(value) is None:
        raise _fail(f"{label} is not a constrained identifier")
    return value


def _symbol(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SYMBOL_RE.fullmatch(value) is None:
        raise _fail(f"{label} is not a canonical symbol")
    return value


def _ns(value: Any, label: str) -> int:
    if type(value) is not int or value < 0 or value > 9_223_372_036_854_775_807:
        raise _fail(f"{label} must be a nonnegative signed UTC-ns integer")
    return value


def _protocol_date_utc_ns(value: str) -> int:
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except (TypeError, ValueError) as exc:
        raise _fail("verified split contains an invalid protocol date") from exc
    if parsed.strftime("%Y-%m-%d") != value:
        raise _fail("verified split contains a noncanonical protocol date")
    return int(parsed.timestamp()) * 1_000_000_000


def _reject_forbidden(value: Any, label: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower().replace("-", "_") in _FORBIDDEN_TERMS:
                raise _fail(f"{label} contains forbidden field: {key}")
            _reject_forbidden(child, label)
    elif isinstance(value, list):
        for child in value:
            _reject_forbidden(child, label)


def _read(
    path: str | Path,
    *,
    expected_sha256: str,
    label: str,
) -> tuple[Path, dict[str, Any], str]:
    try:
        source = canonical_absolute_path(path, f"{label} path")
        reopened, document, physical = read_canonical_json_file(
            source,
            label=label,
            max_bytes=MAX_BYTES,
            max_nodes=MAX_NODES,
            max_depth=20,
        )
    except AuthorityBundleIOError as exc:
        raise _fail(str(exc)) from exc
    if reopened != source or physical != _sha(expected_sha256, f"expected {label} SHA-256"):
        raise _fail(f"{label} physical SHA-256 differs from expected authority")
    _reject_forbidden(document, label)
    return source, document, physical


def _provider_universe(
    path: str | Path,
    *,
    expected_sha256: str,
    split: VerifiedFrozenSplitUseContractV1,
    capability: VerifiedProviderCandidateUniverseV1,
) -> tuple[Path, str, str]:
    source, document, physical = _read(
        path, expected_sha256=expected_sha256, label="provider candidate universe artifact"
    )
    _exact(document, _UNIVERSE_FIELDS, "provider candidate universe artifact")
    if (
        document["schema_version"] != PROVIDER_UNIVERSE_SCHEMA
        or document["purpose"] != "provider_metadata_only_contract_candidate_universe"
        or document["production_admitted"] is not False
        or document["admission_blockers"] != list(_PROVIDER_BLOCKERS)
    ):
        raise _fail("provider candidate universe artifact must remain synthetic and blocked")
    semantic = _sha(
        document["universe_semantic_sha256"], "provider universe semantic SHA-256"
    )
    if content_sha256(dict(document), "universe_semantic_sha256") != semantic:
        raise _fail("provider candidate universe semantic hash mismatch")
    if (
        document["provider_id"] != capability.provider_id
        or document["provider_api_revision"] != capability.provider_api_revision
        or document["scope_sha256"] != capability.scope_physical_sha256
        or document["bundle_manifest_sha256"] != capability.manifest_physical_sha256
        or document["bundle_manifest_semantic_sha256"]
        != capability.manifest_semantic_sha256
        or document["query_semantic_sha256"] != capability.query_semantic_sha256
        or document["pagination_semantic_sha256"]
        != capability.pagination_semantic_sha256
        or document["page_receipt_sha256s"] != list(capability.page_receipt_sha256s)
        or document["provider_response_sha256s"]
        != list(capability.provider_response_sha256s)
        or document["claimed_producer_governance_sha256"]
        != split.producer.semantic_sha256
        or document["claimed_frozen_split_use_contract_sha256"]
        != split.semantic_sha256
    ):
        raise _fail("provider universe artifact differs from reverified provider capability")
    for field in ("prohibited_root_boundary_sha256",):
        _sha(document[field], f"provider universe {field}")
    candidates = document["candidates"]
    if (
        not isinstance(candidates, list)
        or type(document["candidate_count"]) is not int
        or document["candidate_count"] != len(candidates)
        or len(candidates) > MAX_ROWS
    ):
        raise _fail("provider universe candidate count is invalid")
    normalized: list[ProviderCandidateV1] = []
    for index, raw in enumerate(candidates):
        row = _exact(raw, _CANDIDATE_FIELDS, f"provider universe candidate {index}")
        if row["instrument_class"] != "future":
            raise _fail("provider universe candidate must be a future")
        normalized.append(ProviderCandidateV1(
            provider_instrument_id=_identifier(
                row["provider_instrument_id"], "provider instrument ID"
            ),
            provider_symbol=_symbol(row["provider_symbol"], "provider symbol"),
            root_symbol=_symbol(row["root_symbol"], "provider root"),
            venue=_identifier(row["venue"], "provider candidate venue"),
            instrument_class="future",
        ))
    if tuple(normalized) != capability.candidates:
        raise _fail("provider universe candidate rows differ from reverified capability")
    if normalized != sorted(
        normalized,
        key=lambda row: (row.root_symbol, row.provider_symbol, row.provider_instrument_id),
    ):
        raise _fail("provider universe candidates are not canonically sorted")
    return source, physical, semantic


def _registry(
    path: str | Path,
    *,
    expected_sha256: str,
    provider_id: str,
) -> tuple[Path, dict[str, Any], str, str]:
    source, document, physical = _read(
        path, expected_sha256=expected_sha256,
        label="official lifecycle evidence registry",
    )
    _exact(document, _REGISTRY_FIELDS, "official lifecycle evidence registry")
    if (
        document["schema_version"] != REGISTRY_SCHEMA
        or document["purpose"] != "claim_scoped_official_lifecycle_evidence_only"
        or document["evidence_status"] != "synthetic_fixture_claims_unproven"
        or document["production_admission"] is not False
        or document["scope_semantics"] != SCOPE_SEMANTICS
        or document["provider_id"] != provider_id
    ):
        raise _fail("official lifecycle registry must remain synthetic and blocked")
    start = _ns(document["scope_start_utc_ns"], "registry scope start")
    end = _ns(document["scope_end_utc_ns"], "registry scope end")
    if start >= end:
        raise _fail("registry scope must be positive half-open")
    sources = document["official_sources"]
    claims = document["claims"]
    if not isinstance(sources, list) or not isinstance(claims, list):
        raise _fail("registry sources and claims must be lists")
    source_ids: list[str] = []
    for index, raw in enumerate(sources):
        row = _exact(raw, _SOURCE_FIELDS, f"official source {index}")
        source_ids.append(_identifier(row["source_id"], "official source ID"))
        if row["authority"] not in {"official_exchange", "official_provider"}:
            raise _fail("official source authority is invalid")
        _sha(row["evidence_physical_sha256"], "source physical SHA-256")
        _sha(row["evidence_semantic_sha256"], "source semantic SHA-256")
        if row["authenticity_status"] != "detached_unproven":
            raise _fail("official source authenticity must remain detached and unproven")
    if source_ids != sorted(set(source_ids)):
        raise _fail("official source IDs must be sorted and unique")
    normalized_claims: list[tuple[str, str, str, str, str, str, str, int]] = []
    for index, raw in enumerate(claims):
        row = _exact(raw, _CLAIM_FIELDS, f"official claim {index}")
        kind = row["claim_kind"]
        if kind not in {
            "eligibility_start_exact", "trading_end_exclusive_exact",
            "continues_before_scope_start", "continues_after_scope_end",
        }:
            raise _fail("official lifecycle claim kind is invalid")
        normalized_claims.append((
            _identifier(row["claim_id"], "claim ID"),
            _identifier(row["source_id"], "claim source ID"),
            _identifier(row["provider_instrument_id"], "claim provider instrument ID"),
            _symbol(row["provider_symbol"], "claim provider symbol"),
            _symbol(row["root"], "claim root"),
            _identifier(row["venue"], "claim venue"),
            kind,
            _ns(row["claim_utc_ns"], "claim UTC ns"),
        ))
    if len({row[0] for row in normalized_claims}) != len(normalized_claims):
        raise _fail("official lifecycle claim IDs must be unique")
    semantics = [row[1:] for row in normalized_claims]
    if len(semantics) != len(set(semantics)):
        raise _fail("official lifecycle claim semantics must be unique")
    if normalized_claims != sorted(normalized_claims, key=lambda row: row[0]):
        raise _fail("official lifecycle claims must be sorted by claim_id")
    if any(row[1] not in source_ids for row in normalized_claims):
        raise _fail("official lifecycle claim references nonexistent source")
    if {row[1] for row in normalized_claims} != set(source_ids):
        raise _fail("official lifecycle registry contains unused source IDs")
    semantic = _sha(document["registry_semantic_sha256"], "registry semantic SHA-256")
    if content_sha256(dict(document), "registry_semantic_sha256") != semantic:
        raise _fail("official lifecycle registry semantic hash mismatch")
    return source, dict(document), physical, semantic


def _parent_ref(value: Any, label: str) -> tuple[Path, str, str]:
    row = _exact(value, _PARENT_FIELDS, label)
    try:
        path = canonical_absolute_path(row["path"], f"{label}.path")
    except AuthorityBundleIOError as exc:
        raise _fail(str(exc)) from exc
    return (
        path,
        _sha(row["physical_sha256"], f"{label}.physical_sha256"),
        _sha(row["semantic_sha256"], f"{label}.semantic_sha256"),
    )


def _lifecycle(
    path: str | Path,
    *,
    expected_sha256: str,
    split: VerifiedFrozenSplitUseContractV1,
    candidates: VerifiedProviderCandidateUniverseV1,
    universe_identity: tuple[Path, str, str],
    registry_identity: tuple[Path, dict[str, Any], str, str],
) -> tuple[Path, str, str, tuple[LifecycleRowV2, ...]]:
    source, document, physical = _read(
        path, expected_sha256=expected_sha256,
        label="contract lifecycle capability artifact",
    )
    _exact(document, _LIFECYCLE_FIELDS, "contract lifecycle capability artifact")
    if (
        document["schema_version"] != LIFECYCLE_SCHEMA
        or document["purpose"] != "contract_lifecycle_eligibility_mechanism_only"
        or document["evidence_status"] != "synthetic_mechanism_only"
        or document["production_admission"] is not False
        or document["scope_semantics"] != SCOPE_SEMANTICS
    ):
        raise _fail("contract lifecycle artifact must remain synthetic and blocked")
    blockers = document["admission_blockers"]
    if not isinstance(blockers, list) or blockers != [
        "producer_governance_unsigned_and_not_immutable",
        "provider_candidate_query_scope_and_evidence_not_reopened",
        "provider_candidate_authenticity_unproven",
        "official_lifecycle_evidence_authenticity_unproven",
        "production_admission_unavailable",
    ]:
        raise _fail("contract lifecycle artifact blocker closure changed")
    parents = _exact(document["parent_artifacts"], _PARENT_NAMES, "lifecycle parent_artifacts")
    refs = {name: _parent_ref(value, f"parent_artifacts.{name}") for name, value in parents.items()}
    expected_refs = {
        "producer_governance": (
            split.producer.path, split.producer.physical_sha256,
            split.producer.semantic_sha256,
        ),
        "frozen_split_use_contract": (
            split.path, split.physical_sha256, split.semantic_sha256,
        ),
        "provider_candidate_universe": universe_identity,
        "official_lifecycle_evidence_registry": (
            registry_identity[0], registry_identity[2], registry_identity[3],
        ),
    }
    if refs != expected_refs:
        raise _fail("contract lifecycle parent path/hash binding mismatch")
    scope_start = _ns(document["scope_start_utc_ns"], "lifecycle scope start")
    scope_end = _ns(document["scope_end_utc_ns"], "lifecycle scope end")
    expected_start = _protocol_date_utc_ns(split.partitions[0].start)
    expected_end = _protocol_date_utc_ns(split.partitions[-1].end_exclusive)
    registry = registry_identity[1]
    if (
        scope_start >= scope_end
        or (scope_start, scope_end) != (expected_start, expected_end)
        or scope_start != registry["scope_start_utc_ns"]
        or scope_end != registry["scope_end_utc_ns"]
    ):
        raise _fail("lifecycle scope differs from the frozen protocol UTC date envelope")
    candidate_map = {
        item.provider_instrument_id: (
            item.provider_symbol, item.root_symbol, item.venue,
        )
        for item in candidates.candidates
    }
    claims_by_candidate: dict[str, list[Mapping[str, Any]]] = {}
    for claim in registry["claims"]:
        candidate_id = claim["provider_instrument_id"]
        if candidate_id not in candidate_map:
            raise _fail("official claim references nonexistent provider candidate")
        symbol, root, venue = candidate_map[candidate_id]
        if (
            claim["provider_symbol"] != symbol
            or claim["root"] != root
            or claim["venue"] != venue
        ):
            raise _fail("official claim identity differs from provider candidate")
        claims_by_candidate.setdefault(candidate_id, []).append(claim)
    rows = document["rows"]
    if (
        not isinstance(rows, list)
        or type(document["row_count"]) is not int
        or document["row_count"] != len(rows)
        or len(rows) > MAX_ROWS
    ):
        raise _fail("contract lifecycle row count is invalid")
    normalized: list[LifecycleRowV2] = []
    for index, raw in enumerate(rows):
        row = _exact(raw, _ROW_FIELDS, f"contract lifecycle row {index}")
        candidate_id = _identifier(row["provider_instrument_id"], "provider instrument ID")
        if candidate_id not in candidate_map:
            raise _fail("lifecycle row references nonexistent provider candidate")
        provider_symbol = _symbol(row["provider_symbol"], "provider symbol")
        root = _symbol(row["root"], "root")
        venue = _identifier(row["venue"], "venue")
        if (provider_symbol, root, venue) != candidate_map[candidate_id]:
            raise _fail("lifecycle row identity differs from provider candidate")
        contract_id = _symbol(row["contract_id"], "contract_id")
        if contract_id != provider_symbol:
            raise _fail("contract_id must equal provider_symbol until mapping authority exists")
        sources = row["official_source_ids"]
        if (
            not isinstance(sources, list)
            or sources != sorted(set(sources))
            or any(not isinstance(value, str) or _ID_RE.fullmatch(value) is None for value in sources)
        ):
            raise _fail("official_source_ids must be sorted and unique")
        disposition = row["disposition"]
        if disposition not in {"admit", "quarantine"}:
            raise _fail("lifecycle disposition must be admit or quarantine")
        claims = claims_by_candidate.get(candidate_id, [])
        if sources != sorted({claim["source_id"] for claim in claims}):
            raise _fail("lifecycle official_source_ids do not close candidate claims")
        start_kind = row["start_kind"]
        start_raw = row["eligibility_start_utc_ns"]
        end_kind = row["end_kind"]
        end_raw = row["trading_end_exclusive_utc_ns"]
        unestablished = (
            start_kind is None and start_raw is None and end_kind is None and end_raw is None
            and not sources and not claims
        )
        if unestablished:
            if disposition != "quarantine":
                raise _fail("unestablished lifecycle evidence may only quarantine")
            normalized.append(LifecycleRowV2(
                candidate_id, provider_symbol, root, venue, contract_id,
                None, None, None, None, (), "quarantine",
            ))
            continue
        if any(value is None for value in (start_kind, start_raw, end_kind, end_raw)):
            raise _fail("lifecycle evidence must be complete or wholly unestablished quarantine")
        if start_kind not in {"official_exact", "left_censored_at_scope_start"}:
            raise _fail("lifecycle start_kind is invalid")
        if end_kind not in {"official_exact", "right_censored_at_scope_end"}:
            raise _fail("lifecycle end_kind is invalid")
        start = _ns(start_raw, "eligibility start")
        end = _ns(end_raw, "trading end exclusive")
        if start >= end:
            raise _fail("lifecycle interval must be positive half-open")
        if start_kind == "left_censored_at_scope_start" and start != scope_start:
            raise _fail("left censoring is allowed only at scope start")
        if end_kind == "right_censored_at_scope_end" and end != scope_end:
            raise _fail("right censoring is allowed only at scope end")
        expected_disposition = "admit" if start < scope_end and end > scope_start else "quarantine"
        if disposition != expected_disposition:
            raise _fail("lifecycle disposition is not derived from evidence and scope overlap")
        expected_start_claim = (
            "eligibility_start_exact" if start_kind == "official_exact"
            else "continues_before_scope_start"
        )
        expected_end_claim = (
            "trading_end_exclusive_exact" if end_kind == "official_exact"
            else "continues_after_scope_end"
        )
        start_claims = [claim for claim in claims if claim["claim_kind"] == expected_start_claim]
        end_claims = [claim for claim in claims if claim["claim_kind"] == expected_end_claim]
        if not start_claims or any(claim["claim_utc_ns"] != start for claim in start_claims):
            raise _fail("lifecycle start is not established by official claim")
        if not end_claims or any(claim["claim_utc_ns"] != end for claim in end_claims):
            raise _fail("lifecycle end is not established by official claim")
        if any(
            claim["claim_kind"] not in {expected_start_claim, expected_end_claim}
            for claim in claims
        ):
            raise _fail("official registry contains unused lifecycle claims")
        normalized.append(LifecycleRowV2(
            candidate_id, provider_symbol, root, venue, contract_id,
            start_kind, start, end_kind, end, tuple(sources), disposition,
        ))
    expected_ids = set(candidate_map)
    observed_ids = [row.provider_instrument_id for row in normalized]
    if set(observed_ids) != expected_ids or len(observed_ids) != len(expected_ids):
        raise _fail("lifecycle rows omit or duplicate provider candidate dispositions")
    if len({row.provider_symbol for row in normalized}) != len(normalized):
        raise _fail("lifecycle provider symbols must be unique")
    if len({row.contract_id for row in normalized}) != len(normalized):
        raise _fail("lifecycle contract mappings must be unique")
    if normalized != sorted(
        normalized,
        key=lambda row: (row.venue, row.root, row.provider_symbol, row.provider_instrument_id),
    ):
        raise _fail("lifecycle rows are not canonically sorted")
    semantic = _sha(document["lifecycle_semantic_sha256"], "lifecycle semantic SHA-256")
    if content_sha256(dict(document), "lifecycle_semantic_sha256") != semantic:
        raise _fail("contract lifecycle semantic hash mismatch")
    return source, physical, semantic, tuple(normalized)


def _load_once(
    *,
    lifecycle_path: str | Path,
    lifecycle_sha256: str,
    registry_path: str | Path,
    registry_sha256: str,
    provider_universe_path: str | Path,
    provider_universe_sha256: str,
    split_capability: VerifiedFrozenSplitUseContractV1,
    provider_candidate_capability: VerifiedProviderCandidateUniverseV1,
) -> VerifiedContractLifecycleV2:
    split = reopen_and_verify_frozen_split_use_contract_v1(split_capability)
    candidates = reopen_and_verify_provider_candidate_universe_v1(
        provider_candidate_capability
    )
    if candidates.split != split:
        raise _fail("provider candidate capability and lifecycle split capability differ")
    universe = _provider_universe(
        provider_universe_path,
        expected_sha256=provider_universe_sha256,
        split=split,
        capability=candidates,
    )
    registry = _registry(
        registry_path,
        expected_sha256=registry_sha256,
        provider_id=candidates.provider_id,
    )
    lifecycle = _lifecycle(
        lifecycle_path,
        expected_sha256=lifecycle_sha256,
        split=split,
        candidates=candidates,
        universe_identity=universe,
        registry_identity=registry,
    )
    return VerifiedContractLifecycleV2(
        lifecycle_path=lifecycle[0],
        lifecycle_physical_sha256=lifecycle[1],
        lifecycle_semantic_sha256=lifecycle[2],
        registry_path=registry[0],
        registry_physical_sha256=registry[2],
        registry_semantic_sha256=registry[3],
        provider_universe_path=universe[0],
        provider_universe_physical_sha256=universe[1],
        provider_universe_semantic_sha256=universe[2],
        split=split,
        provider_candidates=candidates,
        rows=lifecycle[3],
        evidence_status="synthetic_lifecycle_mechanism_with_reverified_candidate_chain",
        production_admitted=False,
        admission_blockers=LIFECYCLE_BLOCKERS,
        _token=_CAPABILITY_TOKEN,
    )


def load_and_verify_contract_lifecycle_v2(
    lifecycle_path: str | Path,
    *,
    lifecycle_sha256: str,
    registry_path: str | Path,
    registry_sha256: str,
    provider_universe_path: str | Path,
    provider_universe_sha256: str,
    split_capability: VerifiedFrozenSplitUseContractV1,
    provider_candidate_capability: VerifiedProviderCandidateUniverseV1,
) -> VerifiedContractLifecycleV2:
    """Load the complete lifecycle chain twice and return a blocked capability."""
    arguments = dict(
        lifecycle_path=lifecycle_path,
        lifecycle_sha256=lifecycle_sha256,
        registry_path=registry_path,
        registry_sha256=registry_sha256,
        provider_universe_path=provider_universe_path,
        provider_universe_sha256=provider_universe_sha256,
        split_capability=split_capability,
        provider_candidate_capability=provider_candidate_capability,
    )
    first = _load_once(**arguments)
    second = _load_once(**arguments)
    if first != second:
        raise _fail("contract lifecycle chain changed across mandatory reopen")
    return first


def reopen_and_verify_contract_lifecycle_v2(
    capability: VerifiedContractLifecycleV2,
) -> VerifiedContractLifecycleV2:
    if (
        type(capability) is not VerifiedContractLifecycleV2
        or capability._token is not _CAPABILITY_TOKEN
        or capability.production_admitted is not False
    ):
        raise _fail("a verified production-blocked lifecycle capability is required")
    reopened = load_and_verify_contract_lifecycle_v2(
        capability.lifecycle_path,
        lifecycle_sha256=capability.lifecycle_physical_sha256,
        registry_path=capability.registry_path,
        registry_sha256=capability.registry_physical_sha256,
        provider_universe_path=capability.provider_universe_path,
        provider_universe_sha256=capability.provider_universe_physical_sha256,
        split_capability=capability.split,
        provider_candidate_capability=capability.provider_candidates,
    )
    if reopened != capability:
        raise _fail("contract lifecycle capability changed before use")
    return reopened


__all__ = [
    "LIFECYCLE_BLOCKERS", "LIFECYCLE_SCHEMA", "PRODUCER_COMPATIBILITY_COMMIT",
    "PROVIDER_UNIVERSE_SCHEMA", "REGISTRY_SCHEMA", "SCOPE_SEMANTICS",
    "CorpusV3ContractLifecycleError", "LifecycleRowV2",
    "VerifiedContractLifecycleV2", "load_and_verify_contract_lifecycle_v2",
    "reopen_and_verify_contract_lifecycle_v2",
]
