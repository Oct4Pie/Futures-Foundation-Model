from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys

import numpy as np
import pytest

from futures_foundation._authority_bundle_io import (
    canonical_json_bytes,
    content_sha256 as authority_content_sha256,
)
from futures_foundation.corpus_v3_export import verify_contract_day_export
from futures_foundation.corpus_v3_session_audit import (
    AUDIT_SCHEMA,
    CorpusV3SessionAuditError,
    EXPORT_INDEX_PURPOSE,
    EXPORT_INDEX_SCHEMA,
    _ObservedContractDay,
    _aggregate_sessionized_coverage,
    build_sessionized_coverage_audit,
    load_sessionized_export_index,
    load_verified_exports_from_index,
    validate_sessionized_coverage_audit,
)
from futures_foundation.session_denominator_bundle import (
    load_and_verify_session_denominator_bundle_v2,
)


ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(path: Path, value: object) -> None:
    path.write_bytes(canonical_json_bytes(value))


def _index(path: Path, contract: Path, export: Path, request: dict) -> Path:
    document = {
        "schema_version": EXPORT_INDEX_SCHEMA,
        "purpose": EXPORT_INDEX_PURPOSE,
        "contract_sha256": _sha(contract),
        "entries": [{
            "export_path": export.resolve().as_posix(),
            "expected_request": dict(request),
            "receipt_sha256": _sha(export / "receipt.json"),
            "output_shard_sha256": _sha(export / "ticks.parquet"),
        }],
    }
    document["index_semantic_sha256"] = authority_content_sha256(
        document, "index_semantic_sha256"
    )
    _canonical(path, document)
    return path


def _export_fixture(tmp_path: Path):
    namespace = runpy.run_path(str(ROOT / "tests/test_corpus_v3_export.py"))
    export, contract, _ = namespace["_fixture"](tmp_path)
    request = namespace["_request"]()
    return export, contract, request


def _denominator_fixture(tmp_path: Path):
    namespace = runpy.run_path(str(ROOT / "tests/test_session_denominator_bundle.py"))
    fixture = namespace["_producer_fixture"](tmp_path)
    capability = load_and_verify_session_denominator_bundle_v2(
        fixture["bundle_path"],
        expected_manifest_sha256=fixture["manifest_sha256"],
        calendar_rules_path=fixture["rules_path"],
        calendar_rules_sha256=fixture["rules_sha256"],
        scope_v2_path=fixture["scope_path"],
        scope_v2_sha256=fixture["scope_sha256"],
        consumer_scope_path=fixture["consumer_path"],
        consumer_scope_sha256=fixture["consumer_sha256"],
    )
    return capability, fixture


def _denominator_shards():
    return [{
        "partition_id": "shared_train",
        "root": "CL",
        "permitted_uses": ["self_supervised_training"],
        "rows": [{
            "root": "CL",
            "session_day": "2024-01-03",
            "status": "regular",
            "segments_utc_ns": [[100, 200], [300, 400]],
            "segment_semantic_sha256": "a" * 64,
        }],
    }]


def _observation(*, timestamps=(110, 150, 310, 390), split_use="foundation_pretraining"):
    values = np.asarray(timestamps, dtype=np.int64)
    return _ObservedContractDay(
        timestamps_utc_ns=values,
        root="CL",
        contract_id="CLH24",
        session_day="2024-01-03",
        split_use=split_use,
        receipt_sha256="1" * 64,
        output_shard_sha256="2" * 64,
        source_file_table_sha256="3" * 64,
        environment_receipt_sha256="4" * 64,
        instrument_spec_sha256="5" * 64,
        trade_rows=len(values),
        quote_valid_rows=max(0, len(values) - 1),
        negative_trade_rows=1,
        zero_trade_rows=0,
        total_volume=float(len(values)),
        coverage_start_utc_ns=int(values[0]),
        coverage_end_utc_ns=int(values[-1]),
        session_start_utc_ns=100,
        session_end_utc_ns=400,
        source_file_count=2,
    )


def test_sessionized_aggregation_uses_exact_denominator_and_reports_missing():
    aggregate, expected = _aggregate_sessionized_coverage(
        denominator_shards=_denominator_shards(),
        observations=[_observation()],
    )
    assert set(expected) == {("CL", "2024-01-03")}
    assert aggregate["complete_against_denominator"] is True
    assert aggregate["counts"] == {
        "denominator_rows": 1,
        "partition_root_pairs": 1,
        "expected_open_sessions": 1,
        "observed_sessions": 1,
        "missing_open_sessions": 0,
        "contract_days_observed": 1,
        "trade_rows_observed": 4,
        "quote_valid_rows_observed": 3,
    }
    root = aggregate["roots"]["shared_train:CL"]
    assert root["coverage_fraction"] == 1.0
    assert root["median_top_contract_trade_rows"] == 4.0
    assert root["sessions"] == [{
        "session_day": "2024-01-03",
        "status": "observed",
        "contract_count": 1,
        "top_contract_id": "CLH24",
        "top_contract_trade_rows": 4,
        "total_trade_rows": 4,
    }]

    missing, _ = _aggregate_sessionized_coverage(
        denominator_shards=_denominator_shards(), observations=[],
    )
    assert missing["complete_against_denominator"] is False
    assert missing["counts"]["missing_open_sessions"] == 1
    assert missing["roots"]["shared_train:CL"]["missing_session_days"] == ["2024-01-03"]


def test_sessionized_aggregation_rejects_break_events_split_substitution_and_duplicates():
    with pytest.raises(CorpusV3SessionAuditError, match="outside denominator segments"):
        _aggregate_sessionized_coverage(
            denominator_shards=_denominator_shards(),
            observations=[_observation(timestamps=(110, 250, 310))],
        )
    with pytest.raises(CorpusV3SessionAuditError, match="not permitted"):
        _aggregate_sessionized_coverage(
            denominator_shards=_denominator_shards(),
            observations=[_observation(split_use="development")],
        )
    with pytest.raises(CorpusV3SessionAuditError, match="duplicate contract/session"):
        _aggregate_sessionized_coverage(
            denominator_shards=_denominator_shards(),
            observations=[_observation(), _observation()],
        )


def test_export_index_reopens_real_bundle_and_binds_exact_hashes(tmp_path):
    export, contract, request = _export_fixture(tmp_path)
    index_path = _index(tmp_path / "index.json", contract, export, request)
    document, identity = load_sessionized_export_index(index_path)
    exports, reopened_identity = load_verified_exports_from_index(
        index_path, contract_path=contract, allow_test_contract=True,
    )
    assert document["entries"][0]["expected_request"] == request
    assert identity == reopened_identity
    assert len(exports) == 1
    assert exports[0].receipt_sha256 == document["entries"][0]["receipt_sha256"]


def test_export_index_rejects_duplicate_request_hash_swap_and_noncanonical_json(tmp_path):
    export, contract, request = _export_fixture(tmp_path)
    index_path = _index(tmp_path / "index.json", contract, export, request)
    document = json.loads(index_path.read_text(encoding="ascii"))
    document["entries"].append(dict(document["entries"][0]))
    document["index_semantic_sha256"] = authority_content_sha256(
        document, "index_semantic_sha256"
    )
    _canonical(index_path, document)
    with pytest.raises(CorpusV3SessionAuditError, match="duplicate export path|duplicate request"):
        load_sessionized_export_index(index_path)

    index_path = _index(tmp_path / "index-hash.json", contract, export, request)
    document = json.loads(index_path.read_text(encoding="ascii"))
    document["entries"][0]["receipt_sha256"] = "f" * 64
    document["index_semantic_sha256"] = authority_content_sha256(
        document, "index_semantic_sha256"
    )
    _canonical(index_path, document)
    with pytest.raises(CorpusV3SessionAuditError, match="identity differs"):
        load_verified_exports_from_index(
            index_path, contract_path=contract, allow_test_contract=True,
        )

    pretty = tmp_path / "pretty.json"
    pretty.write_text(json.dumps(document, indent=2), encoding="utf-8")
    with pytest.raises(CorpusV3SessionAuditError, match="canonical JSON"):
        load_sessionized_export_index(pretty)


def test_audit_reopens_capability_and_rejects_dataclass_copy_substitution(tmp_path):
    export, contract, request = _export_fixture(tmp_path / "export-fixture")
    index_path = _index(tmp_path / "index.json", contract, export, request)
    exports, identity = load_verified_exports_from_index(
        index_path, contract_path=contract, allow_test_contract=True,
    )
    denominator, _ = _denominator_fixture(tmp_path / "denominator-fixture")
    forged = replace(exports[0], root="ES")
    with pytest.raises(CorpusV3SessionAuditError, match="changed before audit use"):
        build_sessionized_coverage_audit(
            contract_path=contract,
            denominator=denominator,
            exports=[forged],
            export_index_identity=identity,
            allow_test_contract=True,
        )


def test_real_denominator_with_empty_index_is_explicitly_incomplete(tmp_path):
    denominator, fixture = _denominator_fixture(tmp_path / "denominator")
    contract = ROOT / "config/corpus_v3/contract.json"
    index_path = tmp_path / "empty-index.json"
    document = {
        "schema_version": EXPORT_INDEX_SCHEMA,
        "purpose": EXPORT_INDEX_PURPOSE,
        "contract_sha256": _sha(contract),
        "entries": [],
    }
    document["index_semantic_sha256"] = authority_content_sha256(
        document, "index_semantic_sha256"
    )
    _canonical(index_path, document)
    exports, identity = load_verified_exports_from_index(
        index_path, contract_path=contract,
    )
    report = build_sessionized_coverage_audit(
        contract_path=contract,
        denominator=denominator,
        exports=exports,
        export_index_identity=identity,
    )
    assert report["schema_version"] == AUDIT_SCHEMA
    assert report["complete_against_denominator"] is False
    assert report["counts"]["expected_open_sessions"] == 6
    assert report["counts"]["missing_open_sessions"] == 6
    assert report["selected_roots"] == []
    assert report["root_selection_authorized"] is False
    assert report["materialization_admitted"] is False
    assert report["training_admitted"] is False
    reopened = validate_sessionized_coverage_audit(
        report,
        contract_path=contract,
        denominator=denominator,
        exports=exports,
        export_index_identity=identity,
    )
    assert reopened["audit_sha256"] == report["audit_sha256"]

    output = tmp_path / "cli-audit.json"
    completed = subprocess.run([
        sys.executable,
        str(ROOT / "scripts/audit_corpus_v3_sessionized_coverage.py"),
        "--contract", str(contract),
        "--denominator-bundle", str(fixture["bundle_path"]),
        "--denominator-manifest-sha256", str(fixture["manifest_sha256"]),
        "--calendar-rules", str(fixture["rules_path"]),
        "--calendar-rules-sha256", str(fixture["rules_sha256"]),
        "--scope-v2", str(fixture["scope_path"]),
        "--scope-v2-sha256", str(fixture["scope_sha256"]),
        "--consumer-scope", str(fixture["consumer_path"]),
        "--consumer-scope-sha256", str(fixture["consumer_sha256"]),
        "--export-index", str(index_path),
        "--output", str(output),
    ], cwd=ROOT, check=True, capture_output=True, text=True)
    payload = json.loads(completed.stdout)
    assert payload["counts"]["missing_open_sessions"] == 6
    assert output.read_bytes() == canonical_json_bytes(report)


def test_setup_packages_checked_in_sessionized_export_index():
    setup_text = (ROOT / "setup.py").read_text(encoding="utf-8")
    assert '"config/corpus_v3/sessionized_export_index_v1.json"' in setup_text


def test_checked_in_export_index_explicitly_admits_zero_scale_exports():
    contract = ROOT / "config/corpus_v3/contract.json"
    index_path = ROOT / "config/corpus_v3/sessionized_export_index_v1.json"
    document, identity = load_sessionized_export_index(index_path)
    assert document["contract_sha256"] == _sha(contract)
    assert document["entries"] == []
    assert identity == {
        "path": index_path.resolve().as_posix(),
        "physical_sha256": _sha(index_path),
        "semantic_sha256": document["index_semantic_sha256"],
        "entry_count": 0,
    }
