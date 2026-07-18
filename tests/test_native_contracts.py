import copy
import json
from pathlib import Path

import pytest

from futures_foundation.finetune.native_contracts import (
    NativeContractError,
    REGISTRY_PATH,
    attach_integrity,
    build_admission_report,
    evidence_sha256,
    get_arm,
    historical_disposition,
    load_evidence,
    load_registry,
    file_sha256,
    registry_sha256,
    resolve_registry_path,
    technical_evidence,
    technical_runtime_contract,
    validate_identity,
    validate_runtime_contract,
    verify_admission_report,
)
from scripts.snapshot_foundation_history import build_snapshot, snapshot_json_bytes


def _all_passing_checks(registry):
    return {name: {"status": "pass", "evidence": f"test:{name}"}
            for name in registry["required_checks"]}


def _admitted_registry(tmp_path, arm_key="kronos_small", track="F"):
    registry = copy.deepcopy(load_registry(REGISTRY_PATH))
    evidence = copy.deepcopy(load_evidence(REGISTRY_PATH))
    evidence_id = registry["models"][arm_key]["tracks"][track]["evidence_id"]
    checks = evidence["records"][evidence_id].setdefault("checks", {})
    for name in (
        "gradient_freeze_surface", "repeated_batch_loss_decrease",
        "exact_resume", "save_reload_export",
    ):
        checks[name] = {"status": "pass", "evidence": f"test:{name}"}
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(registry), encoding="utf-8")
    path.with_name("native_contract_evidence.json").write_text(
        json.dumps(evidence), encoding="utf-8"
    )
    load_registry.cache_clear(); load_evidence.cache_clear()
    return path, registry


def test_registry_resolution_uses_installed_fallback_when_source_is_absent(tmp_path):
    missing = tmp_path / "missing.json"
    installed = tmp_path / "installed" / "native_contracts.json"
    installed.parent.mkdir(parents=True)
    installed.write_text("{}", encoding="utf-8")
    assert resolve_registry_path((missing, installed)) == installed
    with pytest.raises(FileNotFoundError, match="native-contract registry not found"):
        resolve_registry_path((missing,))


def test_kronos_identity_rejects_wrong_native_tokenizer():
    mini = get_arm("kronos_mini")
    with pytest.raises(NativeContractError, match="tokenizer mismatch"):
        validate_identity(
            mini.key,
            model_id=mini.model_id,
            model_revision=mini.model_revision,
            source_revision=mini.source_revision,
            tokenizer_id="NeoQuasar/Kronos-Tokenizer-base",
            tokenizer_revision="0e0117387f39004a9016484a186a908917e22426",
        )
    assert validate_identity(
        mini.key,
        model_id=mini.model_id,
        model_revision=mini.model_revision,
        source_revision=mini.source_revision,
        tokenizer_id=mini.tokenizer_id,
        tokenizer_revision=mini.tokenizer_revision,
    ) == mini


def test_default_registry_cannot_admit_training_without_training_evidence():
    registry = load_registry(REGISTRY_PATH)
    report = build_admission_report(
        arm_key="kronos_small", track="F", status="native_valid",
        checks=_all_passing_checks(registry),
        approvals=[
            {"reviewer": "reviewer-a", "decision": "approve", "approved_utc": "2026-07-17T00:00:00Z"},
            {"reviewer": "reviewer-b", "decision": "approve", "approved_utc": "2026-07-17T00:01:00Z"},
        ],
        environment={
            **technical_evidence("kronos_small", "F")[1]["environment"],
            "lock_sha256": "a" * 64,
        },
        created_utc="2026-07-17T00:02:00Z",
    )
    with pytest.raises(NativeContractError, match="technical=.*exact_resume"):
        verify_admission_report(report, arm_key="kronos_small", track="F", require_training=True)


def test_hash_bound_report_requires_current_registry_two_approvals_and_all_training_checks(tmp_path):
    registry_path, registry = _admitted_registry(tmp_path)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"admitted checkpoint")
    report = build_admission_report(
        arm_key="kronos_small", track="F", status="native_valid",
        checks=_all_passing_checks(registry),
        approvals=[
            {"reviewer": "reviewer-a", "decision": "approve", "approved_utc": "2026-07-17T00:00:00Z"},
            {"reviewer": "reviewer-b", "decision": "approve", "approved_utc": "2026-07-17T00:01:00Z"},
        ],
        environment={
            **technical_evidence("kronos_small", "F", registry_path)[1]["environment"],
            "lock_sha256": "b" * 64,
        },
        artifacts={"checkpoint": checkpoint},
        created_utc="2026-07-17T00:02:00Z", path=registry_path,
    )
    verified = verify_admission_report(
        report, arm_key="kronos_small", track="F", require_training=True,
        required_artifacts={"checkpoint": checkpoint}, path=registry_path,
    )
    assert verified["status"] == "native_valid"

    replacement = tmp_path / "replacement.pt"
    replacement.write_bytes(b"different checkpoint")
    with pytest.raises(NativeContractError, match="artifact 'checkpoint' hash mismatch"):
        verify_admission_report(
            report, arm_key="kronos_small", track="F", require_training=True,
            required_artifacts={"checkpoint": replacement}, path=registry_path,
        )

    tampered = copy.deepcopy(report)
    tampered["checks"]["fp32_finite"]["status"] = "fail"
    with pytest.raises(NativeContractError, match="integrity mismatch"):
        verify_admission_report(
            tampered, arm_key="kronos_small", track="F", require_training=True,
            path=registry_path,
        )

    one_reviewer = copy.deepcopy(report)
    one_reviewer["approvals"] = one_reviewer["approvals"][:1]
    one_reviewer = attach_integrity(one_reviewer)
    with pytest.raises(NativeContractError, match="independent approvals"):
        verify_admission_report(
            one_reviewer, arm_key="kronos_small", track="F", require_training=True,
            path=registry_path,
        )

    aliased_reviewer = copy.deepcopy(report)
    aliased_reviewer["approvals"][1]["reviewer"] = " REVIEWER-A "
    aliased_reviewer = attach_integrity(aliased_reviewer)
    with pytest.raises(NativeContractError, match="independent approvals"):
        verify_admission_report(
            aliased_reviewer, arm_key="kronos_small", track="F",
            require_training=True, path=registry_path,
        )

    with pytest.raises(NativeContractError, match="floor cannot be lower than 2"):
        verify_admission_report(
            report, arm_key="kronos_small", track="F", require_training=True,
            minimum_approvals=0, path=registry_path,
        )


def test_admission_report_rejects_unsupported_check_claims_and_bad_approval_time(tmp_path):
    registry_path, registry = _admitted_registry(tmp_path)
    unsupported = _all_passing_checks(registry)
    unsupported["fp32_finite"] = {"status": "pass"}
    with pytest.raises(NativeContractError, match="needs concrete evidence"):
        build_admission_report(
            arm_key="kronos_small", track="F", status="native_valid",
            checks=unsupported,
            approvals=[
                {"reviewer": "reviewer-a", "decision": "approve", "approved_utc": "2026-07-17T00:00:00Z"},
                {"reviewer": "reviewer-b", "decision": "approve", "approved_utc": "2026-07-17T00:01:00Z"},
            ],
            environment={
                **technical_evidence("kronos_small", "F", registry_path)[1]["environment"],
                "lock_sha256": "c" * 64,
            },
            created_utc="2026-07-17T00:02:00Z", path=registry_path,
        )

    with pytest.raises(NativeContractError, match="later than report creation"):
        build_admission_report(
            arm_key="kronos_small", track="F", status="native_valid",
            checks=_all_passing_checks(registry),
            approvals=[
                {"reviewer": "reviewer-a", "decision": "approve", "approved_utc": "2026-07-17T00:00:00Z"},
                {"reviewer": "reviewer-b", "decision": "approve", "approved_utc": "2026-07-17T00:03:00Z"},
            ],
            environment={
                **technical_evidence("kronos_small", "F", registry_path)[1]["environment"],
                "lock_sha256": "d" * 64,
            },
            created_utc="2026-07-17T00:02:00Z", path=registry_path,
        )


def test_runtime_contracts_are_exact_and_reject_uncovered_shapes():
    contract = technical_runtime_contract("chronos_v1", "F")
    assert contract["context_length"] == 512
    assert contract["prediction_length"] == 16
    assert contract["num_samples"] == 20
    assert validate_runtime_contract(
        "chronos_v1", "F",
        {"context_length": 512, "prediction_length": 16,
         "dtype": "float32", "num_samples": 20,
         "quantile_levels": [0.1, 0.5, 0.9]},
    ) == contract
    with pytest.raises(NativeContractError, match="missing=.*quantile_levels"):
        validate_runtime_contract(
            "chronos_v1", "F",
            {"context_length": 512, "prediction_length": 16,
             "dtype": "float32", "num_samples": 20},
        )
    with pytest.raises(NativeContractError, match="runtime contract mismatch"):
        validate_runtime_contract(
            "chronos_v1", "F", {"context_length": 256}
        )


def test_report_environment_must_match_technical_evidence():
    registry = load_registry(REGISTRY_PATH)
    with pytest.raises(NativeContractError, match="environment does not match"):
        build_admission_report(
            arm_key="kronos_small", track="F", status="native_valid",
            checks=_all_passing_checks(registry),
            approvals=[
                {"reviewer": "reviewer-a", "decision": "approve", "approved_utc": "2026-07-17T00:00:00Z"},
                {"reviewer": "reviewer-b", "decision": "approve", "approved_utc": "2026-07-17T00:01:00Z"},
            ],
            environment={"python": "3.11", "dtype": "float32"},
            created_utc="2026-07-17T00:02:00Z",
        )


def test_historical_dispositions_cover_known_invalid_contracts():
    assert historical_disposition("kronos_mini")["default_status"] == "invalid_contract"
    assert "artificial_normalized_zeros" in historical_disposition("ttm_r2")["reason"]
    assert historical_disposition("toto2_22m")["default_status"] == "invalid_contract"
    assert historical_disposition("moirai2_small")["default_status"] == "research_only"


def test_tracked_historical_snapshot_is_bound_to_current_registry_and_index():
    snapshot_path = Path(
        "config/foundation_models/historical_native_contract_snapshot.json"
    )
    snapshot = json.loads(snapshot_path.read_text())
    source = Path(snapshot["source_index"]["path"])
    assert snapshot["registry_sha256"] == registry_sha256()
    assert snapshot["evidence_sha256"] == evidence_sha256()
    assert snapshot["source_index"]["sha256"] == file_sha256(source)
    assert snapshot["coverage"]["registered_model_count"] == len(load_registry()["models"])
    assert snapshot["coverage"]["native_ranking_eligible_count"] == 0
    assert all(not row["native_ranking_eligible"] for row in snapshot["models"].values())
    assert sum(
        any(capability["status"] in {"native_valid", "research_only"}
            for capability in dossier["tracks"].values())
        for dossier in load_registry()["models"].values()
    ) == 13
    rebuilt = build_snapshot(source, Path(snapshot["artifact_root"]))
    assert snapshot_path.read_bytes() == snapshot_json_bytes(rebuilt)


def test_historical_snapshot_is_overlay_only_and_never_native_ranking_eligible(tmp_path):
    artifact_root = tmp_path / "artifacts"
    (artifact_root / "kronos_mini").mkdir(parents=True)
    (artifact_root / "kronos_mini" / "stage1.pt").write_bytes(b"historical")
    index_path = artifact_root / "STAGE_RESULTS_INDEX.json"
    index_path.write_text(json.dumps({
        "schema_version": "ffm_stage_results_index_v1",
        "created_utc": "2026-07-16T00:00:00Z",
        "models": {"kronos_mini": {"complete_chain": True, "shared_validation": {"x": 1}}},
    }), encoding="utf-8")
    snapshot = build_snapshot(index_path, artifact_root)
    assert snapshot_json_bytes(snapshot) == snapshot_json_bytes(
        build_snapshot(index_path, artifact_root)
    )
    row = snapshot["models"]["kronos_mini"]
    assert row["status"] == "invalid_contract"
    assert row["artifact_file_count"] == 1
    assert row["complete_chain_claim"] is True
    assert row["native_ranking_eligible"] is False
    assert snapshot["coverage"]["native_ranking_eligible_count"] == 0
