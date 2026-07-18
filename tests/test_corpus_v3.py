import gzip
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from futures_foundation.corpus_v3 import (
    CorpusV3Error,
    build_coverage_audit,
    content_sha256,
    contract_root,
    verify_contract,
)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> dict:
    tmp_path.mkdir(parents=True, exist_ok=True)
    artifact_names = (
        "data_source_registry", "data_admission", "tick_admission", "loader_smoke",
        "instrument_economics", "market_calendar", "lake_hash_summary",
        "lake_leaf_manifest",
    )
    artifacts = {}
    for name in artifact_names:
        path = tmp_path / f"{name}.txt"
        path.write_text(name)
        artifacts[name] = {"path": str(path), "sha256": _hash(path)}
    manifest = tmp_path / "coverage.jsonl.gz"
    rows = []
    for year in range(2019, 2026):
        for month in range(1, 13):
            for day in (2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20):
                value = f"{year:04d}{month:02d}{day:02d}"
                rows.append({"sym": f"ESH{year % 100:02d}", "kind": "ticks", "day": value,
                             "files": 1, "rows": 2000})
                rows.append({"sym": f"MESH{year % 100:02d}", "kind": "ticks", "day": value,
                             "files": 1, "rows": 2500})
    with gzip.open(manifest, "wt") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    artifacts["coverage_manifest"] = {"path": str(manifest), "sha256": _hash(manifest)}
    loader = tmp_path / "loader.py"
    loader.write_text("# pinned\n")
    tick_root = tmp_path / "ticks"
    tick_root.mkdir()
    contract = {
        "schema_version": "ffm_corpus_v3_contract_v1",
        "contract_id": "test",
        "current_admission": {"status": "coverage_audit_only", "materialization": "blocked"},
        "source": {"source_id": "ticks", "physical_root": str(tick_root),
                   "max_date_exclusive": "2026-01-01",
                   "hash_of_hashes_sha256": "a" * 64,
                   "inventory_totals": {
                       "all_tick_files": len(rows),
                       "all_tick_rows": sum(row["rows"] for row in rows),
                       "admitted_pre_cutoff_files": len(rows),
                       "admitted_pre_cutoff_rows": sum(row["rows"] for row in rows),
                       "admitted_contract_symbols_all_dates": len({row["sym"] for row in rows}),
                       "admitted_contract_symbols_pre_cutoff": len({row["sym"] for row in rows}),
                   }},
        "loader": {"path": str(loader), "sha256": _hash(loader), "schema_version": "test_loader",
                   "disposition": "reference_only_not_authorized_as_corpus_v3_export"},
        "required_export_seam": {"owner": "alphaforge", "purpose_token": "foundation_training",
                                 "missing_event_seq_policy": "reject"},
        "artifacts": artifacts,
        "admitted_roots": ["ES", "MES"],
        "blocked_roots": {},
        "splits": {
            "foundation_pretraining": {"start": "2019-01-01", "end_exclusive": "2024-07-01"},
            "supervised_training": {"start": "2019-07-01", "end_exclusive": "2024-07-01"},
            "development": {"start": "2024-07-01", "end_exclusive": "2025-07-01"},
            "legacy_holdout_excluded": {
                "start": "2025-07-01", "end_exclusive": "2026-01-01",
                "use": "coverage_report_only_never_training_validation_calibration_or_selection",
            },
        },
        "universe_screen": {
            "uses_strategy_outcomes": False,
            "authorizes_universe_selection": False,
            "eligible_periods": ["supervised_training", "development"],
            "provisional_diagnostic_thresholds": {
                "min_training_utc_buckets": 1,
                "min_training_years_with_150_utc_buckets": 1,
                "min_development_utc_buckets": 1,
                "min_median_top_contract_ticks_per_utc_bucket": 1500,
                "max_training_utc_bucket_gap_days": 60,
            },
        },
        "execution_ruler": {"primary_added_slippage_ticks_round_trip": 0},
    }
    Path(artifacts["data_admission"]["path"]).write_text(yaml.safe_dump({
        "admissions": [{"source_id": "ticks", "status": "admitted_limited",
                        "roots": ["ES", "MES"], "admitted_data_modes": ["raw_ticks"],
                        "max_date_exclusive": "2026-01-01"}],
    }))
    Path(artifacts["tick_admission"]["path"]).write_text(yaml.safe_dump({
        "source_id": "ticks", "decision": "admitted_limited",
        "registry": {"sha256": artifacts["data_source_registry"]["sha256"]},
        "corpus": {"hash_of_hashes_sha256": "a" * 64},
        "scope": {"max_date_exclusive": "2026-01-01"},
        "loader_policy": {"schema_version": "test_loader"},
        "summary": {"admitted": ["ES", "MES"]},
    }))
    Path(artifacts["loader_smoke"]["path"]).write_text(yaml.safe_dump({
        "source_id": "ticks", "admission_artifact_sha256": "pending",
        "summary": {"roots": 2, "passed": 2, "failed": 0},
    }))
    Path(artifacts["instrument_economics"]["path"]).write_text(yaml.safe_dump({
        "instruments": {"ES": {}, "MES": {}},
    }))
    Path(artifacts["market_calendar"]["path"]).write_text(yaml.safe_dump({
        "coverage": {"end": "2025-12-31"},
        "products": {"test": {"roots": ["ES", "MES"]}},
    }))
    Path(artifacts["lake_hash_summary"]["path"]).write_text(json.dumps({
        "manifest_jsonl": Path(artifacts["lake_leaf_manifest"]["path"]).name,
        "hash_of_hashes_sha256": "a" * 64,
    }))
    for record in artifacts.values():
        record["sha256"] = _hash(Path(record["path"]))
    smoke_path = Path(artifacts["loader_smoke"]["path"])
    smoke = yaml.safe_load(smoke_path.read_text())
    smoke["admission_artifact_sha256"] = artifacts["tick_admission"]["sha256"]
    smoke_path.write_text(yaml.safe_dump(smoke))
    artifacts["loader_smoke"]["sha256"] = _hash(smoke_path)
    return contract


def test_contract_root_is_exact_and_prefix_safe():
    roots = ["ES", "MES", "M6E", "6E"]
    assert contract_root("ESH25", roots) == "ES"
    assert contract_root("MESH25", roots) == "MES"
    assert contract_root("M6EZ25", roots) == "M6E"
    assert contract_root("6EZ25", roots) == "6E"
    assert contract_root("ES_CONT", roots) is None


def test_coverage_audit_is_deterministic_and_holdout_cannot_select(tmp_path):
    contract = _fixture(tmp_path)
    first = build_coverage_audit(contract)
    second = build_coverage_audit(contract)
    assert first == second
    assert first["report_sha256"] == content_sha256({
        key: value for key, value in first.items() if key != "report_sha256"
    })
    assert first["candidate_roots"] == ["ES", "MES"]
    assert first["selected_roots"] == []
    assert first["selection_status"].startswith("blocked")
    assert first["roots"]["ES"]["periods"]["legacy_holdout_excluded"]["active_days"] > 0
    assert "legacy_holdout_excluded" not in first["screen"]["eligible_periods"]


def test_artifact_drift_and_holdout_eligibility_fail_closed(tmp_path):
    contract = _fixture(tmp_path)
    verify_contract(contract, verify_artifacts=True)
    Path(contract["artifacts"]["market_calendar"]["path"]).write_text("changed")
    with pytest.raises(CorpusV3Error, match="artifact hash drift"):
        build_coverage_audit(contract)
    contract = _fixture(tmp_path / "again")
    contract["universe_screen"]["eligible_periods"].append("legacy_holdout_excluded")
    with pytest.raises(CorpusV3Error, match="holdout cannot influence"):
        verify_contract(contract)


def test_manifest_audit_cannot_authorize_root_selection(tmp_path):
    contract = _fixture(tmp_path)
    contract["universe_screen"]["authorizes_universe_selection"] = True
    with pytest.raises(CorpusV3Error, match="cannot authorize"):
        verify_contract(contract)


def test_future_inventory_rows_are_ignored_not_admitted(tmp_path):
    contract = _fixture(tmp_path)
    manifest = Path(contract["artifacts"]["coverage_manifest"]["path"])
    with gzip.open(manifest, "at") as handle:
        handle.write(json.dumps({"sym": "ESH26", "kind": "ticks", "day": "20260102",
                                 "files": 1, "rows": 999999}) + "\n")
    contract["artifacts"]["coverage_manifest"]["sha256"] = _hash(manifest)
    contract["source"]["inventory_totals"]["all_tick_files"] += 1
    contract["source"]["inventory_totals"]["all_tick_rows"] += 999999
    contract["source"]["inventory_totals"]["admitted_contract_symbols_all_dates"] += 1
    report = build_coverage_audit(contract)
    assert report["source_scope"]["ignored"]["at_or_after_source_end"] == 1
    assert report["roots"]["ES"]["periods"]["legacy_holdout_excluded"]["total_ticks"] < 999999


def test_duplicate_and_coerced_inventory_counts_fail_closed(tmp_path):
    contract = _fixture(tmp_path)
    manifest = Path(contract["artifacts"]["coverage_manifest"]["path"])
    with gzip.open(manifest, "rt") as handle:
        rows = [json.loads(line) for line in handle]
    rows.append(dict(rows[0]))
    with gzip.open(manifest, "wt") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    contract["artifacts"]["coverage_manifest"]["sha256"] = _hash(manifest)
    with pytest.raises(CorpusV3Error, match="duplicate admitted"):
        build_coverage_audit(contract)

    contract = _fixture(tmp_path / "typed")
    manifest = Path(contract["artifacts"]["coverage_manifest"]["path"])
    with gzip.open(manifest, "rt") as handle:
        rows = [json.loads(line) for line in handle]
    rows[0]["rows"] = str(rows[0]["rows"])
    with gzip.open(manifest, "wt") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    contract["artifacts"]["coverage_manifest"]["sha256"] = _hash(manifest)
    with pytest.raises(CorpusV3Error, match="strict integers"):
        build_coverage_audit(contract)


def test_semantic_governance_drift_fails_even_when_hash_is_updated(tmp_path):
    contract = _fixture(tmp_path)
    path = Path(contract["artifacts"]["data_admission"]["path"])
    value = yaml.safe_load(path.read_text())
    value["admissions"][0]["roots"] = ["ES"]
    path.write_text(yaml.safe_dump(value))
    contract["artifacts"]["data_admission"]["sha256"] = _hash(path)
    with pytest.raises(CorpusV3Error, match="semantic cross-binding"):
        build_coverage_audit(contract)


def test_holdout_inventory_cannot_change_training_diagnostics(tmp_path):
    contract = _fixture(tmp_path)
    baseline = build_coverage_audit(contract)
    manifest = Path(contract["artifacts"]["coverage_manifest"]["path"])
    addition = {"sym": "ESH25", "kind": "ticks", "day": "20250801",
                "files": 1, "rows": 999999}
    with gzip.open(manifest, "at") as handle:
        handle.write(json.dumps(addition) + "\n")
    contract["artifacts"]["coverage_manifest"]["sha256"] = _hash(manifest)
    totals = contract["source"]["inventory_totals"]
    totals["all_tick_files"] += 1
    totals["all_tick_rows"] += addition["rows"]
    totals["admitted_pre_cutoff_files"] += 1
    totals["admitted_pre_cutoff_rows"] += addition["rows"]
    changed = build_coverage_audit(contract)
    assert changed["roots"]["ES"]["periods"]["supervised_training"] == (
        baseline["roots"]["ES"]["periods"]["supervised_training"]
    )
    assert changed["roots"]["ES"]["periods"]["development"] == (
        baseline["roots"]["ES"]["periods"]["development"]
    )
    assert changed["roots"]["ES"]["screen"] == baseline["roots"]["ES"]["screen"]
    assert changed["selected_roots"] == baseline["selected_roots"] == []
