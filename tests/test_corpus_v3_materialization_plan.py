from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys

import pytest

from futures_foundation._authority_bundle_io import canonical_json_bytes, content_sha256
from futures_foundation.corpus_v3_materialization_plan import (
    OBSERVATION_PURPOSE,
    OBSERVATION_SCHEMA,
    CorpusV3MaterializationPlanError,
    _expected_requests,
    build_materialization_plan_v1,
    build_split_scoped_inventory_v1,
    load_inventory_observations_v1,
    validate_materialization_plan_v1,
    validate_split_scoped_inventory_v1,
    write_materialization_artifact,
)


ROOT = Path(__file__).resolve().parents[1]


def _expected() -> dict:
    namespace = runpy.run_path(str(ROOT / "tests/test_corpus_v3_expected_requests.py"))
    return namespace["_derive_fixture"]()


def _file(path: str, start: int, end: int, marker: str) -> dict:
    return {
        "relative_path": path,
        "interval_start_utc_ns": start,
        "interval_end_exclusive_utc_ns": end,
        "size": 100,
        "sha256": marker * 64,
    }


def _rows(expected: dict) -> list[dict]:
    requests = _expected_requests(expected)
    return [
        {
            "request_semantic_sha256": requests[0]["request_semantic_sha256"],
            "status": "available_exact",
            "reason": None,
            "source_files": [_file("CLU24/2018-06-01/a.parquet", 100, 200, "1")],
        },
        {
            "request_semantic_sha256": requests[1]["request_semantic_sha256"],
            "status": "available_exact",
            "reason": None,
            "source_files": [
                _file("CLU24/2020-01-02/a.parquet", 180, 240, "2"),
                _file("CLU24/2020-01-02/b.parquet", 240, 310, "3"),
            ],
        },
        {
            "request_semantic_sha256": requests[2]["request_semantic_sha256"],
            "status": "missing",
            "reason": "no_source_files_for_expected_request",
            "source_files": [],
        },
    ]


def test_inventory_has_exact_request_closure_and_metadata_only_scope():
    expected = _expected()
    inventory = build_split_scoped_inventory_v1(
        expected_request_denominator=expected,
        inventory_rows=_rows(expected),
    )
    assert inventory["counts"] == {
        "inventory_rows": 3,
        "status_ambiguous": 0,
        "status_available_exact": 2,
        "status_boundary_blocked": 0,
        "status_missing": 1,
    }
    assert inventory["complete_against_expected_requests"] is True
    assert inventory["all_requests_available_exact"] is False
    assert inventory["data_access"] == {
        "expected_requests_only": True,
        "reserved_oos_requests_present": False,
        "source_metadata_read": True,
        "source_content_read": False,
        "prices_volume_rows_or_labels_read": False,
        "source_file_bytes_reopened_by_this_verifier": False,
    }
    assert inventory["materialization_admitted"] is False
    assert inventory["training_admitted"] is False
    assert validate_split_scoped_inventory_v1(
        inventory, expected_request_denominator=expected,
    ) == inventory


def test_inventory_rejects_missing_duplicate_and_unknown_request_rows():
    expected = _expected()
    rows = _rows(expected)
    with pytest.raises(CorpusV3MaterializationPlanError, match="closure"):
        build_split_scoped_inventory_v1(
            expected_request_denominator=expected, inventory_rows=rows[:-1],
        )
    with pytest.raises(CorpusV3MaterializationPlanError, match="duplicate request"):
        build_split_scoped_inventory_v1(
            expected_request_denominator=expected,
            inventory_rows=[*rows, deepcopy(rows[0])],
        )
    unknown = deepcopy(rows)
    unknown[0]["request_semantic_sha256"] = "f" * 64
    with pytest.raises(CorpusV3MaterializationPlanError, match="closure"):
        build_split_scoped_inventory_v1(
            expected_request_denominator=expected, inventory_rows=unknown,
        )


def test_available_exact_source_geometry_rejects_gap_overlap_and_nonoverlap():
    expected = _expected()
    rows = _rows(expected)
    gap = deepcopy(rows)
    gap[1]["source_files"][1]["interval_start_utc_ns"] = 250
    with pytest.raises(CorpusV3MaterializationPlanError, match="internal gap"):
        build_split_scoped_inventory_v1(
            expected_request_denominator=expected, inventory_rows=gap,
        )
    overlap = deepcopy(rows)
    overlap[1]["source_files"][1]["interval_start_utc_ns"] = 230
    with pytest.raises(CorpusV3MaterializationPlanError, match="overlap"):
        build_split_scoped_inventory_v1(
            expected_request_denominator=expected, inventory_rows=overlap,
        )
    outside = deepcopy(rows)
    outside[0]["source_files"][0].update(
        interval_start_utc_ns=1, interval_end_exclusive_utc_ns=99,
    )
    with pytest.raises(CorpusV3MaterializationPlanError, match="does not overlap"):
        build_split_scoped_inventory_v1(
            expected_request_denominator=expected, inventory_rows=outside,
        )


def test_source_path_hash_and_status_contracts_fail_closed():
    expected = _expected()
    traversal = _rows(expected)
    traversal[0]["source_files"][0]["relative_path"] = "../escape.parquet"
    with pytest.raises(CorpusV3MaterializationPlanError, match="relative POSIX path"):
        build_split_scoped_inventory_v1(
            expected_request_denominator=expected, inventory_rows=traversal,
        )
    bad_hash = _rows(expected)
    bad_hash[0]["source_files"][0]["sha256"] = "not-a-hash"
    with pytest.raises(ValueError, match="SHA-256"):
        build_split_scoped_inventory_v1(
            expected_request_denominator=expected, inventory_rows=bad_hash,
        )
    missing_with_file = _rows(expected)
    missing_with_file[2]["source_files"] = [
        _file("CLU24/2024-07-02/a.parquet", 300, 400, "4")
    ]
    with pytest.raises(CorpusV3MaterializationPlanError, match="cannot carry"):
        build_split_scoped_inventory_v1(
            expected_request_denominator=expected,
            inventory_rows=missing_with_file,
        )
    available_with_reason = _rows(expected)
    available_with_reason[0]["reason"] = "caller_override"
    with pytest.raises(CorpusV3MaterializationPlanError, match="cannot carry"):
        build_split_scoped_inventory_v1(
            expected_request_denominator=expected,
            inventory_rows=available_with_reason,
        )


def test_materialization_plan_selects_only_available_exact_and_remains_blocked(tmp_path):
    expected = _expected()
    inventory = build_split_scoped_inventory_v1(
        expected_request_denominator=expected,
        inventory_rows=_rows(expected),
    )
    plan = build_materialization_plan_v1(
        expected_request_denominator=expected, inventory=inventory,
    )
    assert plan["counts"] == {
        "expected_requests": 3,
        "selected_requests": 2,
        "excluded_requests": 1,
        "excluded_by_status": {
            "ambiguous": 0, "boundary_blocked": 0, "missing": 1,
        },
    }
    assert len(plan["selected_requests"]) == 2
    assert all("status" not in row and "reason" not in row for row in plan["selected_requests"])
    assert plan["selection_rule"] == "status_equals_available_exact_only"
    assert plan["execution_status"].startswith("blocked_pending")
    assert plan["materialization_admitted"] is False
    assert plan["training_admitted"] is False
    assert validate_materialization_plan_v1(
        plan, expected_request_denominator=expected, inventory=inventory,
    ) == plan
    output = write_materialization_artifact(plan, tmp_path / "plan.json")
    assert output.read_bytes() == canonical_json_bytes(plan)


def test_inventory_observation_authority_and_cli_are_physically_bound(tmp_path):
    expected = _expected()
    expected_path = tmp_path / "expected.json"
    expected_path.write_bytes(canonical_json_bytes(expected))
    observations = {
        "schema_version": OBSERVATION_SCHEMA,
        "purpose": OBSERVATION_PURPOSE,
        "expected_request_denominator_sha256": expected[
            "expected_request_denominator_sha256"
        ],
        "rows": _rows(expected),
    }
    observations["observations_semantic_sha256"] = content_sha256(
        observations, "observations_semantic_sha256"
    )
    observations_path = tmp_path / "observations.json"
    observations_path.write_bytes(canonical_json_bytes(observations))
    rows, identity = load_inventory_observations_v1(
        observations_path,
        expected_physical_sha256=hashlib.sha256(
            observations_path.read_bytes()
        ).hexdigest(),
        expected_request_denominator=expected,
    )
    assert rows == _rows(expected)
    assert identity["row_count"] == 3

    inventory_output = tmp_path / "inventory.json"
    plan_output = tmp_path / "plan.json"
    completed = subprocess.run([
        sys.executable,
        str(ROOT / "scripts/build_corpus_v3_materialization_plan.py"),
        "--expected-requests", str(expected_path),
        "--expected-requests-sha256", hashlib.sha256(
            expected_path.read_bytes()
        ).hexdigest(),
        "--inventory-observations", str(observations_path),
        "--inventory-observations-sha256", hashlib.sha256(
            observations_path.read_bytes()
        ).hexdigest(),
        "--inventory-output", str(inventory_output),
        "--plan-output", str(plan_output),
    ], cwd=ROOT, check=True, capture_output=True, text=True)
    payload = json.loads(completed.stdout)
    inventory = json.loads(inventory_output.read_text(encoding="ascii"))
    plan = json.loads(plan_output.read_text(encoding="ascii"))
    assert payload["inventory_sha256"] == inventory["inventory_sha256"]
    assert payload["plan_sha256"] == plan["plan_sha256"]
    assert payload["materialization_admitted"] is False
    assert plan["selected_requests"] == build_materialization_plan_v1(
        expected_request_denominator=expected, inventory=inventory,
    )["selected_requests"]


def test_inventory_observation_hash_parent_and_canonical_transport_reject(tmp_path):
    expected = _expected()
    document = {
        "schema_version": OBSERVATION_SCHEMA,
        "purpose": OBSERVATION_PURPOSE,
        "expected_request_denominator_sha256": expected[
            "expected_request_denominator_sha256"
        ],
        "rows": _rows(expected),
    }
    document["observations_semantic_sha256"] = content_sha256(
        document, "observations_semantic_sha256"
    )
    path = tmp_path / "observations.json"
    path.write_bytes(canonical_json_bytes(document))
    with pytest.raises(CorpusV3MaterializationPlanError, match="physical SHA-256"):
        load_inventory_observations_v1(
            path, expected_physical_sha256="f" * 64,
            expected_request_denominator=expected,
        )
    wrong_parent = deepcopy(document)
    wrong_parent["expected_request_denominator_sha256"] = "f" * 64
    wrong_parent["observations_semantic_sha256"] = content_sha256(
        wrong_parent, "observations_semantic_sha256"
    )
    path.write_bytes(canonical_json_bytes(wrong_parent))
    with pytest.raises(CorpusV3MaterializationPlanError, match="expected parent"):
        load_inventory_observations_v1(
            path,
            expected_physical_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            expected_request_denominator=expected,
        )
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    with pytest.raises(CorpusV3MaterializationPlanError, match="canonical JSON"):
        load_inventory_observations_v1(
            path,
            expected_physical_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            expected_request_denominator=expected,
        )


def test_inventory_and_plan_integrity_rejects_tampering():
    expected = _expected()
    inventory = build_split_scoped_inventory_v1(
        expected_request_denominator=expected,
        inventory_rows=_rows(expected),
    )
    forged_inventory = deepcopy(inventory)
    forged_inventory["materialization_admitted"] = True
    with pytest.raises(CorpusV3MaterializationPlanError, match="integrity"):
        validate_split_scoped_inventory_v1(
            forged_inventory, expected_request_denominator=expected,
        )
    plan = build_materialization_plan_v1(
        expected_request_denominator=expected, inventory=inventory,
    )
    forged_plan = deepcopy(plan)
    forged_plan["selected_requests"].append(deepcopy(plan["selected_requests"][0]))
    with pytest.raises(CorpusV3MaterializationPlanError, match="integrity"):
        validate_materialization_plan_v1(
            forged_plan, expected_request_denominator=expected, inventory=inventory,
        )
