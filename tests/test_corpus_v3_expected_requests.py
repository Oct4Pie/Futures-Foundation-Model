from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from futures_foundation._authority_bundle_io import canonical_json_bytes
from futures_foundation.corpus_v3_contract_lifecycle import (
    LifecycleRowV2,
    VerifiedContractLifecycleV2,
)
from futures_foundation.corpus_v3_expected_requests import (
    CorpusV3ExpectedRequestError,
    _derive,
    build_expected_request_denominator_v1,
    validate_expected_request_denominator_v1,
    write_expected_request_denominator_v1,
)
from futures_foundation.corpus_v3_producer_governance import (
    PartitionV1,
    VerifiedFrozenSplitUseContractV1,
    VerifiedProducerGovernanceV1,
)
from futures_foundation.session_denominator_bundle import (
    VerifiedSessionDenominatorBundleV2,
)


SHA = "a" * 64


def _split() -> VerifiedFrozenSplitUseContractV1:
    producer = VerifiedProducerGovernanceV1(
        path=Path("/authority/producer.json"), physical_sha256="1" * 64,
        semantic_sha256="2" * 64, provider_id="provider", source_id="source",
        data_mode="metadata_only", namespace_root="/market/namespace",
        evidence_status="detached_self_consistency_only", production_admitted=False,
        _token=object(),
    )
    return VerifiedFrozenSplitUseContractV1(
        path=Path("/authority/split.json"), physical_sha256="3" * 64,
        semantic_sha256="4" * 64, producer=producer,
        partitions=(
            PartitionV1("pretrain", "2011-01-01", "2019-07-01"),
            PartitionV1("shared_train", "2019-07-01", "2024-07-01"),
            PartitionV1("development", "2024-07-01", "2025-07-01"),
            PartitionV1("legacy_holdout", "2025-07-01", "2026-07-01"),
        ),
        permitted_uses=(
            ("pretrain", ("foundation_pretraining", "self_supervised_training")),
            ("shared_train", ("self_supervised_training", "supervised_training")),
            ("development", ("validation",)),
            ("legacy_holdout", ()),
        ),
        boundary_leaf_policy="boundary_blocked",
        evidence_status="detached_self_consistency_only",
        production_admitted=False,
        _token=object(),
    )


def _session() -> VerifiedSessionDenominatorBundleV2:
    return VerifiedSessionDenominatorBundleV2(
        bundle_path=Path("/authority/session-bundle"),
        manifest_physical_sha256="5" * 64,
        manifest_semantic_sha256="6" * 64,
        calendar_rules_path=Path("/authority/rules.json"),
        calendar_rules_sha256="7" * 64,
        scope_v2_path=Path("/authority/scope.json"),
        scope_v2_sha256="8" * 64,
        consumer_scope_path=Path("/authority/consumer.json"),
        consumer_scope_sha256="9" * 64,
        shard_count=3, row_count=4, production_admitted=False, _token=object(),
    )


def _lifecycle(split: VerifiedFrozenSplitUseContractV1) -> VerifiedContractLifecycleV2:
    rows = (
        LifecycleRowV2(
            "cl-202409", "CLU24", "CL", "GLBX", "CLU24",
            "official_exact", 150, "official_exact", 350,
            ("source_cl",), "admit",
        ),
        LifecycleRowV2(
            "cl-202412", "CLZ24", "CL", "GLBX", "CLZ24",
            None, None, None, None, (), "quarantine",
        ),
    )
    return VerifiedContractLifecycleV2(
        lifecycle_path=Path("/authority/lifecycle.json"),
        lifecycle_physical_sha256="b" * 64,
        lifecycle_semantic_sha256="c" * 64,
        registry_path=Path("/authority/registry.json"),
        registry_physical_sha256="d" * 64,
        registry_semantic_sha256="e" * 64,
        provider_universe_path=Path("/authority/universe.json"),
        provider_universe_physical_sha256="f" * 64,
        provider_universe_semantic_sha256="0" * 64,
        split=split, provider_candidates=None, rows=rows,
        evidence_status="synthetic_lifecycle_mechanism_with_reverified_candidate_chain",
        production_admitted=False, admission_blockers=("blocked",), _token=object(),
    )


def _row(day: str, segments: list[list[int]], status: str = "regular") -> dict:
    return {
        "root": "CL", "session_day": day, "status": status,
        "segments_utc_ns": segments, "source_ids": ["calendar"],
        "segment_semantic_sha256": SHA,
    }


def _shard(partition: str, start: str, end: str, uses: list[str], rows: list[dict]) -> dict:
    return {
        "schema_version": "alphaforge_session_denominator_shard_v2",
        "partition_id": partition, "root": "CL", "start": start,
        "end_exclusive": end, "permitted_uses": uses,
        "row_count": len(rows), "rows": rows,
        "shard_semantic_sha256": SHA,
    }


def _shards() -> list[dict]:
    return [
        _shard(
            "pretrain", "2011-01-01", "2019-07-01",
            ["self_supervised_training"],
            [_row("2018-06-01", [[100, 200]])],
        ),
        _shard(
            "shared_train", "2019-07-01", "2024-07-01",
            ["supervised_training"],
            [
                _row("2020-01-02", [[180, 300]]),
                _row("2020-01-03", [], status="closed"),
            ],
        ),
        _shard(
            "development", "2024-07-01", "2025-07-01",
            ["validation"],
            [_row("2024-07-02", [[300, 400]])],
        ),
    ]


def _derive_fixture() -> dict:
    split = _split()
    return _derive(
        split=split, session=_session(), lifecycle=_lifecycle(split),
        session_shards=_shards(),
    )


def test_expected_requests_clip_exact_segments_and_preserve_protocol_order():
    report = _derive_fixture()
    assert report["partitions"] == ["pretrain", "shared_train", "development"]
    assert [row["partition_id"] for row in report["request_shards"]] == [
        "pretrain", "shared_train", "development",
    ]
    requests = [
        request
        for shard in report["request_shards"]
        for request in shard["requests"]
    ]
    assert [
        (row["request_start_utc_ns"], row["request_end_exclusive_utc_ns"])
        for row in requests
    ] == [(150, 200), (180, 300), (300, 350)]
    assert report["counts"] == {
        "candidate_dispositions": 2,
        "session_dispositions": 4,
        "request_shards": 3,
        "expected_requests": 3,
    }
    assert report["production_admitted"] is False
    assert report["materialization_admitted"] is False
    assert report["training_admitted"] is False


def test_quarantined_candidate_remains_explicit_zero_request_row():
    report = _derive_fixture()
    quarantined = report["candidate_dispositions"][1]
    assert quarantined["provider_instrument_id"] == "cl-202412"
    assert quarantined["lifecycle_disposition"] == "quarantine"
    assert quarantined["emitted_request_count"] == 0
    assert quarantined["exclusion_reasons"] == ["missing_lifecycle_evidence"]


def test_request_identity_and_candidate_index_are_not_inferred_from_month_codes():
    report = _derive_fixture()
    first = report["request_shards"][0]["requests"][0]
    assert first == {
        "candidate_index": 0,
        "provider_instrument_id": "cl-202409",
        "provider_symbol": "CLU24",
        "contract_id": "CLU24",
        "venue": "GLBX",
        "session_day": "2018-06-01",
        "session_segment_index": 0,
        "request_start_utc_ns": 150,
        "request_end_exclusive_utc_ns": 200,
    }


def test_missing_partition_root_shard_extra_root_and_use_escalation_reject():
    split = _split(); session = _session(); lifecycle = _lifecycle(split)
    with pytest.raises(CorpusV3ExpectedRequestError, match="closure"):
        _derive(
            split=split, session=session, lifecycle=lifecycle,
            session_shards=_shards()[:-1],
        )
    extra = deepcopy(_shards())
    extra[0]["root"] = "ES"
    extra[0]["rows"][0]["root"] = "ES"
    with pytest.raises(CorpusV3ExpectedRequestError, match="lacks a lifecycle"):
        _derive(split=split, session=session, lifecycle=lifecycle, session_shards=extra)
    escalated = deepcopy(_shards())
    escalated[0]["permitted_uses"] = ["validation"]
    with pytest.raises(CorpusV3ExpectedRequestError, match="outside frozen split/use"):
        _derive(
            split=split, session=session, lifecycle=lifecycle,
            session_shards=escalated,
        )


def test_session_day_status_segments_and_lifecycle_parent_mismatch_reject():
    split = _split(); session = _session(); lifecycle = _lifecycle(split)
    outside = deepcopy(_shards())
    outside[0]["rows"][0]["session_day"] = "2020-01-01"
    with pytest.raises(CorpusV3ExpectedRequestError, match="outside the frozen split"):
        _derive(split=split, session=session, lifecycle=lifecycle, session_shards=outside)
    bad_status = deepcopy(_shards())
    bad_status[0]["rows"][0]["status"] = "closed"
    with pytest.raises(CorpusV3ExpectedRequestError, match="status and segment"):
        _derive(split=split, session=session, lifecycle=lifecycle, session_shards=bad_status)
    other_split = _split()
    mismatched = _lifecycle(other_split)
    with pytest.raises(CorpusV3ExpectedRequestError, match="split capabilities differ"):
        _derive(split=split, session=session, lifecycle=mismatched, session_shards=_shards())


def test_public_builder_reopens_all_parents_and_validator_recomputes(monkeypatch, tmp_path):
    import futures_foundation.corpus_v3_expected_requests as module

    split = _split(); session = _session(); lifecycle = _lifecycle(split)
    monkeypatch.setattr(
        module, "reopen_and_verify_frozen_split_use_contract_v1", lambda value: split,
    )
    monkeypatch.setattr(
        module, "reopen_and_verify_contract_lifecycle_v2", lambda value: lifecycle,
    )
    monkeypatch.setattr(module, "iter_verified_session_shards", lambda value: iter(_shards()))
    report = build_expected_request_denominator_v1(
        split_capability=split,
        session_denominator_capability=session,
        lifecycle_capability=lifecycle,
    )
    reopened = validate_expected_request_denominator_v1(
        report,
        split_capability=split,
        session_denominator_capability=session,
        lifecycle_capability=lifecycle,
    )
    assert reopened == report
    output = write_expected_request_denominator_v1(report, tmp_path / "expected.json")
    assert output.read_bytes() == canonical_json_bytes(report)


def test_validator_rejects_tampered_count_even_with_stale_hash(monkeypatch):
    import futures_foundation.corpus_v3_expected_requests as module

    split = _split(); session = _session(); lifecycle = _lifecycle(split)
    monkeypatch.setattr(
        module, "reopen_and_verify_frozen_split_use_contract_v1", lambda value: split,
    )
    monkeypatch.setattr(
        module, "reopen_and_verify_contract_lifecycle_v2", lambda value: lifecycle,
    )
    monkeypatch.setattr(module, "iter_verified_session_shards", lambda value: iter(_shards()))
    report = build_expected_request_denominator_v1(
        split_capability=split,
        session_denominator_capability=session,
        lifecycle_capability=lifecycle,
    )
    forged = deepcopy(report)
    forged["counts"]["expected_requests"] = 4
    with pytest.raises(CorpusV3ExpectedRequestError, match="integrity"):
        validate_expected_request_denominator_v1(
            forged,
            split_capability=split,
            session_denominator_capability=session,
            lifecycle_capability=lifecycle,
        )
