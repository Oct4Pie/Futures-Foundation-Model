from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from futures_foundation.corpus_v3 import CorpusV3Error
from futures_foundation.corpus_v3_export import (
    ARRAY_FIELDS,
    EXPORT_SCHEMA_VERSION,
    RECEIPT_SCHEMA_VERSION,
    _content_sha,
    _output_schema,
    _schema_sha256,
    _semantic_sha256,
    verify_contract_day_export,
)
from futures_foundation.tick_path_labels import (
    TickPathLabelConfig,
    VerifiedTickPathLabels,
    _VERIFIED_LABEL_TOKEN,
    build_tick_path_labels,
    decision_manifest_sha256,
    write_tick_label_bundle,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")), encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    arrays = {
        "timestamp_utc_ns": np.array([1_704_236_400_000_000_000, 1_704_236_401_000_000_000], dtype=np.int64),
        "time_us": np.array([0, 1_000_000], dtype=np.int64),
        "event_seq": np.array([0, 1], dtype=np.uint64),
        "price": np.array([-5.0, -4.99]),
        "bid": np.array([-5.01, np.nan]),
        "ask": np.array([-5.0, -4.98]),
        "quote_valid": np.array([True, False]),
        "volume": np.array([1.0, 2.0]),
        "bid_volume": np.array([1.0, 1.0]),
        "ask_volume": np.array([0.0, 1.0]),
        "source_file_index": np.array([0, 0], dtype=np.uint32),
        "source_row_ordinal": np.array([0, 1], dtype=np.uint64),
    }
    raw_root = tmp_path / "raw"
    raw_filename = "CLH24_20240102T000000Z_20240104T000000Z_ticks.parquet"
    raw_file = raw_root / "CLH24" / "20240102" / raw_filename
    raw_file.parent.mkdir(parents=True)
    raw_schema = pa.schema([
        pa.field("timestamp", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("event_seq", pa.uint64(), nullable=False),
        pa.field("price", pa.float64(), nullable=False),
        pa.field("bid", pa.float64(), nullable=False),
        pa.field("ask", pa.float64(), nullable=False),
        pa.field("volume", pa.float64(), nullable=False),
        pa.field("bid_volume", pa.float64(), nullable=False),
        pa.field("ask_volume", pa.float64(), nullable=False),
    ], metadata={
        b"event_sequence_scope": b"file_order",
        b"source_system": b"sierra_chart_dtc",
        b"dtc_endpoint": b"historical_price_data",
        b"dtc_record_interval": b"0",
    })
    raw_values = [
        pa.array(arrays["timestamp_utc_ns"] // 1_000, type=pa.timestamp("us", tz="UTC")),
        pa.array(arrays["event_seq"]), pa.array(arrays["price"]), pa.array(arrays["bid"]),
        pa.array(arrays["ask"]), pa.array(arrays["volume"]),
        pa.array(arrays["bid_volume"]), pa.array(arrays["ask_volume"]),
    ]
    pq.write_table(pa.Table.from_arrays(raw_values, schema=raw_schema), raw_file)
    manifest = tmp_path / "leaf.jsonl.gz"
    logical = f"sc_v2_ticks/raw/CLH24/20240102/{raw_filename}"
    with gzip.open(manifest, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "path": logical, "size": raw_file.stat().st_size, "sha256": _sha(raw_file),
        }, separators=(",", ":")) + "\n")
    instruments = tmp_path / "instruments.yaml"
    instruments.write_text(yaml.safe_dump({
        "instruments": {"CL": {"tick_size": 0.01, "tick_value_usd": 10.0}},
    }))
    registry = tmp_path / "data_sources.yaml"
    registry.write_text(yaml.safe_dump({
        "sources": {"sierra_chart": {"dataset_governance": {"ticks": {
            "price_normalization": {"multiplier_by_root": {"CL": 1.0}},
        }}}},
    }))
    base = json.loads((Path(__file__).parents[1] / "config/corpus_v3/contract.json").read_text())
    base["source"]["physical_root"] = str(raw_root)
    base["source"]["hash_of_hashes_sha256"] = "a" * 64
    base["admitted_roots"] = ["CL"]
    base["blocked_roots"] = {}
    base["required_export_seam"]["producer_exporter_sha256"] = "b" * 64
    base["required_export_seam"]["producer_git_commit"] = "c" * 40
    base["required_export_seam"]["pilot_request"] = {
        "root": "CL", "contract_id": "CLH24", "session_day": "2024-01-03",
        "split_use": "foundation_pretraining", "purpose": "foundation_training",
    }
    base["current_admission"] = {
        "status": "representative_shard_pilot",
        "materialization": "representative_shard_only",
    }
    base["artifacts"]["lake_leaf_manifest"] = {"path": str(manifest), "sha256": _sha(manifest)}
    base["artifacts"]["instrument_economics"] = {
        "path": str(instruments), "sha256": _sha(instruments),
    }
    base["artifacts"]["data_source_registry"] = {
        "path": str(registry), "sha256": _sha(registry),
    }
    contract = tmp_path / "contract.json"
    _canonical(contract, base)

    export = tmp_path / "export"
    export.mkdir()
    table = pa.Table.from_arrays(
        [pa.array(arrays[name], type=_output_schema().field(name).type) for name in ARRAY_FIELDS],
        schema=_output_schema(),
    )
    pq.write_table(table, export / "ticks.parquet", compression="zstd", use_dictionary=False)
    request = {
        "root": "CL", "contract_id": "CLH24", "session_day": "2024-01-03",
        "split_use": "foundation_pretraining", "purpose": "foundation_training",
    }
    source_table = {"schema_version": "alphaforge_foundation_source_file_table_v2", "files": [{
        "source_file_index": 0,
        "path": logical,
        "physical_relative_path": f"CLH24/20240102/{raw_filename}",
        "size": raw_file.stat().st_size,
        "sha256": _sha(raw_file),
        "rows": 2,
        "filename_interval_start_utc_ns": 1_704_153_600_000_000_000,
        "filename_interval_end_utc_ns": 1_704_326_400_000_000_000,
    }]}
    instrument = {
        "schema_version": "alphaforge_foundation_instrument_spec_v1", "root": "CL",
        "tick_size": "0.01", "tick_value_usd": "10",
        "source_price_multiplier": "1",
        "source_artifact_sha256": _sha(instruments),
        "price_normalization_artifact_sha256": _sha(registry),
    }
    producer = {
        "schema_version": "alphaforge_foundation_producer_sources_v1",
        "repository": {"git_commit": "c" * 40, "git_clean": True},
        "files": [{
        "path": "src/alphaforge/foundation_export.py", "sha256": "b" * 64, "size": 123,
    }]}
    environment = {"schema_version": "test_environment"}
    governance = {
        "data_admission_sha256": base["artifacts"]["data_admission"]["sha256"],
        "qa_artifact_sha256": base["artifacts"]["tick_admission"]["sha256"],
        "loader_smoke_sha256": base["artifacts"]["loader_smoke"]["sha256"],
        "registry_sha256": base["artifacts"]["data_source_registry"]["sha256"],
    }
    semantic_metadata = {
        "schema_version": EXPORT_SCHEMA_VERSION, "root": "CL", "contract_id": "CLH24",
        "session_day": "2024-01-03",
        "session_start_utc_ns": int(arrays["timestamp_utc_ns"][0]),
        "session_end_utc_ns": 1_704_319_200_000_000_000,
        "source_file_table_sha256": _content_sha(source_table),
        "consumer_contract_sha256": _sha(contract),
    }
    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION, "status": "complete",
        "purpose": "foundation_training", "request": request,
        "request_sha256": _content_sha(request),
        "roots": ["CL"], "date_range": ["2024-01-03", "2024-01-03"],
        "window_contract": "exchange_session_half_open_unspliced_contract_day",
        "output_format": "parquet_zstd_numeric_no_dictionary_v1",
        "output_file": "ticks.parquet",
        "consumer_contract_sha256": _sha(contract),
        "lake_hash_of_hashes_sha256": "a" * 64,
        "leaf_manifest_sha256": _sha(manifest),
        "producer_source_manifest": producer,
        "producer_source_manifest_sha256": _content_sha(producer),
        "environment": environment,
        "environment_receipt_sha256": _content_sha(environment),
        "instrument_spec": instrument,
        "instrument_spec_sha256": _content_sha(instrument),
        "governance": governance, "governance_sha256": _content_sha(governance),
        "source_file_table": source_table,
        "source_file_table_sha256": _content_sha(source_table),
        "output_schema_sha256": _schema_sha256(),
        "output_shard_sha256": _sha(export / "ticks.parquet"),
        "semantic_metadata": semantic_metadata,
        "semantic_shard_sha256": _semantic_sha256(arrays, semantic_metadata),
        "output_row_counts": {"trade_rows": 2},
        "excluded_row_counts": {"nonfinite_trade": 0, "invalid_quote_not_excluded": 1},
        "source_rows_read": 2,
        "selected_source_file_sha256": [_sha(raw_file)],
        "session_bounds_and_internal_gap_evidence": {
            "session_start_utc_ns": int(arrays["timestamp_utc_ns"][0]),
            "session_end_utc_ns": 1_704_319_200_000_000_000,
            "internal_source_interval_gaps": [],
            "market_calendar_sha256": base["artifacts"]["market_calendar"]["sha256"],
        },
        "negative_price_preservation": {
            "policy": "finite_tick_aligned_trade_prices_are_preserved_without_positive_filter",
            "negative_trade_rows": 2, "zero_trade_rows": 0,
        },
        "trade_row_preservation_independent_of_quote_validity": {
            "policy": "invalid_bbo_at_trade_sets_quote_valid_false_without_dropping_trade",
            "invalid_quote_rows_preserved": 1,
        },
        "source_ordering_evidence": {
            "timestamp_inversion_count": 0,
            "max_source_timestamp_regression_ns": 0,
            "output_order": ["timestamp_utc_ns", "event_seq"],
            "ordering_semantics": "historical_event_time_not_live_arrival_or_zero_delay_proof",
        },
        "continuous_market_completeness": False,
        "market_path_completeness_claim": "complete_among_verified_observed_source_records_only",
    }
    receipt["receipt_payload_sha256"] = _content_sha(receipt)
    _canonical(export / "receipt.json", receipt)
    return export, contract, raw_file


def _request() -> dict:
    return {
        "root": "CL", "contract_id": "CLH24", "session_day": "2024-01-03",
        "split_use": "foundation_pretraining", "purpose": "foundation_training",
    }


def test_verified_export_is_the_production_capability_for_path_labels(tmp_path):
    export, contract, _ = _fixture(tmp_path)
    verified = verify_contract_day_export(
        export, contract_path=contract, expected_request=_request(), allow_test_contract=True,
    )
    assert verified.receipt_sha256 == _sha(export / "receipt.json")
    rows = verified.label_rows()
    assert rows["export_receipt_sha256"] == verified.receipt_sha256
    assert rows["source_shard_sha256"] == _sha(export / "ticks.parquet")
    assert rows["price"].tolist() == [-5.0, -4.99]
    assert verified.build_path_index().contract_id == "CLH24"


def test_verifier_rejects_raw_source_and_output_mutation(tmp_path):
    export, contract, raw = _fixture(tmp_path)
    raw.write_bytes(raw.read_bytes() + b"tamper")
    with pytest.raises(CorpusV3Error, match="source bytes"):
        verify_contract_day_export(
            export, contract_path=contract, expected_request=_request(), allow_test_contract=True,
        )
    export, contract, _ = _fixture(tmp_path / "output")
    shard = export / "ticks.parquet"
    data = bytearray(shard.read_bytes())
    data[-1] ^= 1
    shard.write_bytes(data)
    with pytest.raises(CorpusV3Error, match="byte hash"):
        verify_contract_day_export(
            export, contract_path=contract, expected_request=_request(), allow_test_contract=True,
        )


def test_verifier_rejects_receipt_swap_and_holdout_split(tmp_path):
    export, contract, _ = _fixture(tmp_path)
    receipt_path = export / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["request"]["session_day"] = "2025-07-02"
    receipt["request"]["split_use"] = "legacy_holdout_excluded"
    receipt["request_sha256"] = _content_sha(receipt["request"])
    receipt.pop("receipt_payload_sha256")
    receipt["receipt_payload_sha256"] = _content_sha(receipt)
    _canonical(receipt_path, receipt)
    with pytest.raises(CorpusV3Error, match="identity|holdout|contract"):
        verify_contract_day_export(
            export, contract_path=contract, expected_request=_request(), allow_test_contract=True,
        )


def test_genuine_leaf_cannot_cover_forged_rehashed_output(tmp_path):
    export, contract, _ = _fixture(tmp_path)
    shard = export / "ticks.parquet"
    table = pq.read_table(shard)
    columns = {
        name: table.column(name).combine_chunks().to_numpy(zero_copy_only=False)
        for name in ARRAY_FIELDS
    }
    columns["price"] = columns["price"].copy()
    columns["price"][1] = -4.98  # still tick-aligned, but no longer the cited raw row
    forged = pa.Table.from_arrays(
        [pa.array(columns[name], type=_output_schema().field(name).type) for name in ARRAY_FIELDS],
        schema=_output_schema(),
    )
    pq.write_table(forged, shard, compression="zstd", use_dictionary=False)
    receipt_path = export / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["output_shard_sha256"] = _sha(shard)
    receipt["semantic_shard_sha256"] = _semantic_sha256(columns, receipt["semantic_metadata"])
    receipt.pop("receipt_payload_sha256")
    receipt["receipt_payload_sha256"] = _content_sha(receipt)
    _canonical(receipt_path, receipt)
    with pytest.raises(CorpusV3Error, match="reconstruct from raw source lineage"):
        verify_contract_day_export(
            export, contract_path=contract, expected_request=_request(), allow_test_contract=True,
        )


def test_complete_bundle_transplant_and_nested_receipt_mutation_are_blocked(tmp_path):
    export, contract, _ = _fixture(tmp_path)
    wrong = dict(_request())
    wrong["contract_id"] = "CLK24"
    with pytest.raises(CorpusV3Error, match="bundle identity"):
        verify_contract_day_export(
            export, contract_path=contract, expected_request=wrong, allow_test_contract=True,
        )
    verified = verify_contract_day_export(
        export, contract_path=contract, expected_request=_request(), allow_test_contract=True,
    )
    with pytest.raises(TypeError):
        verified.receipt["request"]["root"] = "ES"


def test_duck_typed_fake_cannot_impersonate_verified_export():
    Fake = type("VerifiedContractDayExport", (), {
        "is_authentic": lambda self: True,
        "build_path_index": lambda self, **kwargs: None,
    })
    with pytest.raises(ValueError, match="VerifiedContractDayExport"):
        build_tick_path_labels(
            Fake(),
            decision_time_utc_ns=np.array([], dtype=np.int64),
            decision_event_seq=np.array([], dtype=np.uint64),
            risk_ticks=np.array([], dtype=np.int64),
            risk_known_time_utc_ns=np.array([], dtype=np.int64),
            risk_known_event_seq=np.array([], dtype=np.uint64),
            decision_manifest_sha256_value="0" * 64,
        )


def test_verified_labels_are_immutable_and_writer_cross_binds_full_identity(tmp_path):
    export, contract, _ = _fixture(tmp_path)
    verified = verify_contract_day_export(
        export, contract_path=contract, expected_request=_request(), allow_test_contract=True,
    )
    config = TickPathLabelConfig(
        horizons_seconds=(1,), targets_r=(1.0,),
        entry_tolerance_seconds=1, endpoint_tolerance_seconds=1,
    )
    index = verified.build_path_index(config=config)
    decision_ts = np.array([index.timestamp_utc_ns[0]], dtype=np.int64)
    decision_seq = np.array([index.event_seq[0]], dtype=np.uint64)
    risk = np.array([1], dtype=np.int64)
    manifest = decision_manifest_sha256(
        index, decision_ts, decision_seq, risk, decision_ts, decision_seq,
    )
    labels = build_tick_path_labels(
        verified,
        decision_time_utc_ns=decision_ts,
        decision_event_seq=decision_seq,
        risk_ticks=risk,
        risk_known_time_utc_ns=decision_ts,
        risk_known_event_seq=decision_seq,
        decision_manifest_sha256_value=manifest,
        config=config,
    )
    with pytest.raises(TypeError):
        labels["root"] = "ES"
    forged_values = dict(labels)
    forged_values["root"] = "ES"
    forged = VerifiedTickPathLabels(forged_values, _token=_VERIFIED_LABEL_TOKEN)
    with pytest.raises(ValueError, match="matching verified export"):
        write_tick_label_bundle(
            forged, tmp_path / "forged_labels", verified_export=verified,
        )
