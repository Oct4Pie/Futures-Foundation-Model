import base64
import copy
import json
from pathlib import Path
import platform
import subprocess
import sys

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from futures_foundation.finetune import native_contracts
from futures_foundation.finetune.native_contracts import (
    NativeContractError,
    REGISTRY_PATH,
    approval_signature_payload,
    attach_integrity,
    build_admission_request,
    build_admission_report,
    evidence_sha256,
    get_arm,
    historical_disposition,
    load_evidence,
    load_registry,
    file_sha256,
    measure_runtime_environment,
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


def _measured_test_environment():
    return {
        "python": platform.python_version(),
        "executable": str(Path(sys.executable).resolve()),
        "dtype": "float32",
        "profile": "test",
        "network_policy": "python_socket_deny",
    }


def _test_runtime_controls():
    environment = _measured_test_environment()
    return {
        name: environment[name]
        for name in native_contracts.RUNTIME_CONTROL_FIELDS
        if name in environment
    }


def _test_runtime_lock():
    return {
        "schema_version": "ffm_native_runtime_lock_v1",
        "comparison_policy": {
            "portable_software": "exact",
            "hardware_runtime": "exact_when_measurable_explicit_when_unavailable",
        },
        "portable_software": {
            "python_executable": str(Path(sys.executable).resolve()),
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "distributions": [{"name": "test-runtime", "version": "1.0"}],
        },
        "hardware_runtime": {
            "torch_importable": False,
            "cuda_available": False,
            "visible_devices": None,
            "torch_cuda_runtime": None,
            "cudnn_version": None,
            "devices": [],
            "driver_probe": {"status": "unavailable", "rows": []},
        },
    }


def _write_trust_store(registry_path: Path):
    private_keys = {}
    records = {}
    for reviewer in ("reviewer-a", "reviewer-b"):
        key = Ed25519PrivateKey.generate()
        key_id = f"{reviewer}-key"
        public_pem = key.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        ).decode("ascii")
        records[key_id] = {
            "reviewer": reviewer,
            "algorithm": "ed25519",
            "public_key_pem": public_pem,
        }
        private_keys[key_id] = key
    registry_path.with_name("trusted_approvers.json").write_text(
        json.dumps({"schema_version": "ffm_trusted_approvers_v1", "keys": records}),
        encoding="utf-8",
    )
    return private_keys


def _sign_approvals(request, private_keys):
    approvals = []
    for offset, reviewer in enumerate(("reviewer-a", "reviewer-b")):
        key_id = f"{reviewer}-key"
        approval = {
            "reviewer": reviewer,
            "key_id": key_id,
            "algorithm": "ed25519",
            "decision": "approve",
            "approved_utc": f"2026-07-17T00:0{offset}:00Z",
        }
        approval["signature"] = base64.b64encode(
            private_keys[key_id].sign(
                approval_signature_payload(request["approval_target_sha256"], approval)
            )
        ).decode("ascii")
        approvals.append(approval)
    return approvals


def _runtime_artifacts(tmp_path: Path, monkeypatch):
    from futures_foundation.finetune import native_parity_runtime

    monkeypatch.setattr(
        native_parity_runtime, "measure_runtime_lock",
        lambda: copy.deepcopy(_test_runtime_lock()),
    )
    model = tmp_path / "model.bin"
    model.write_bytes(b"exact-model")
    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(source), "config", "user.name", "Test"], check=True)
    (source / "model.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(source), "add", "model.py"], check=True)
    subprocess.run(["git", "-C", str(source), "commit", "-q", "-m", "fixture"], check=True)
    artifacts = {"model": model, "source": source}
    technical = {
        name: native_contracts._runtime_artifact_description(name, path)
        for name, path in artifacts.items()
    }
    runner = copy.deepcopy(technical["source"])
    runner["path"] = str(source)
    technical["runner"] = runner
    monkeypatch.setattr(
        native_contracts,
        "_technical_runtime_artifacts",
        lambda *args, **kwargs: copy.deepcopy(technical),
    )
    monkeypatch.setattr(
        native_contracts, "_current_execution_artifact",
        lambda: copy.deepcopy(runner),
    )
    return artifacts


def _signed_report(
    *, registry_path, registry, artifacts, private_keys, approvals_mutator=None,
    environment=None, extra_artifacts=None, route=None, require_training=False,
    status="native_valid", arm_key="kronos_small", track="F",
):
    all_artifacts = {**artifacts, **(extra_artifacts or {})}
    kwargs = dict(
        arm_key=arm_key, track=track, status=status,
        checks=_all_passing_checks(registry),
        environment=environment or _measured_test_environment(),
        artifacts=all_artifacts,
        route=route,
        require_training=require_training,
        created_utc="2026-07-17T00:00:00Z", path=registry_path,
    )
    request = build_admission_request(**kwargs)
    approvals = _sign_approvals(request, private_keys)
    if approvals_mutator:
        approvals_mutator(approvals)
    return build_admission_report(**kwargs, approvals=approvals)


def _admitted_registry(tmp_path, arm_key="kronos_small", track="F"):
    registry = copy.deepcopy(load_registry(REGISTRY_PATH))
    evidence = copy.deepcopy(load_evidence(REGISTRY_PATH))
    evidence_id = registry["models"][arm_key]["tracks"][track]["evidence_id"]
    # Admission-report unit tests mutate technical training checks deliberately.
    # Keep that synthetic record transitional so it does not pretend to retain the
    # canonical raw-bundle binding after being copied into a temporary directory.
    evidence["check_profiles"]["test_transitional"] = copy.deepcopy(
        evidence["check_profiles"][evidence["records"][evidence_id]["profile"]]
    )
    evidence["records"][evidence_id]["profile"] = "test_transitional"
    evidence["records"][evidence_id].pop("bundle", None)
    checks = evidence["records"][evidence_id].setdefault("checks", {})
    evidence["records"][evidence_id]["environment"] = _measured_test_environment()
    evidence["records"][evidence_id]["runtime_lock"] = _test_runtime_lock()
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
    private_keys = _write_trust_store(path)
    load_registry.cache_clear(); load_evidence.cache_clear()
    return path, registry, private_keys


def _copy_training_route_contracts(registry_path: Path):
    from futures_foundation.finetune import native_training_routes

    registry_path.with_name("native_training_routes.json").write_bytes(
        native_training_routes.route_registry_path(REGISTRY_PATH).read_bytes()
    )
    registry_path.with_name("native_training_route_evidence.json").write_bytes(
        native_training_routes.route_evidence_path(REGISTRY_PATH).read_bytes()
    )
    native_training_routes.load_route_registry.cache_clear()
    native_training_routes.load_route_evidence.cache_clear()


def _admit_training_route(registry_path: Path, key: str):
    from futures_foundation.finetune import native_training_routes

    _copy_training_route_contracts(registry_path)
    route_path = registry_path.with_name("native_training_routes.json")
    evidence_path = registry_path.with_name("native_training_route_evidence.json")
    routes = json.loads(route_path.read_text(encoding="utf-8"))
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    source = {"kind": "git_commit", "revision": "a" * 40}
    routes["training_methodology_source"] = source
    route = routes["routes"][key]
    route["status"] = "admitted"
    evidence_id = f"{key}:fixture"
    route["evidence_id"] = evidence_id
    evidence["training_methodology_source"] = source
    evidence["records"] = {
        evidence_id: {
            "arm_key": route["arm_key"],
            "track": route["track"],
            "route_id": route["route_id"],
            "route_key": key,
            "route_sha256": native_contracts.content_sha256(route),
            "status": "pass",
            "checks": {
                name: {"status": "pass", "evidence": f"fixture:{name}"}
                for name in native_training_routes.TRAINING_CHECKS
            },
            "environment": {"python": "fixture", "dtype": "float32"},
            "artifacts": {"trainer": {"sha256": "b" * 64}},
            "reason": "synthetic admitted-route fixture",
        }
    }
    evidence["route_registry_sha256"] = native_contracts.content_sha256(routes)
    route_path.write_text(json.dumps(routes), encoding="utf-8")
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    native_training_routes.load_route_registry.cache_clear()
    native_training_routes.load_route_evidence.cache_clear()
    return route["route_id"]


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
    assert get_arm("kronos_small").training_admitted is False
    _, record, checks = technical_evidence("kronos_small", "F")
    assert native_contracts._verified_runtime_lock(record) == record["runtime_lock"]
    assert {
        name for name in native_contracts.TRAINING_CHECKS
        if checks[name]["status"] != "pass"
    } == set(native_contracts.TRAINING_CHECKS)


def test_training_gate_rejects_previous_null_and_flat_custom_false_admissions(
    tmp_path, monkeypatch
):
    registry_path, registry, private_keys = _admitted_registry(tmp_path)
    _copy_training_route_contracts(registry_path)
    artifacts = _runtime_artifacts(tmp_path, monkeypatch)
    report = _signed_report(
        registry_path=registry_path,
        registry=registry,
        artifacts=artifacts,
        private_keys=private_keys,
    )

    with pytest.raises(NativeContractError, match="requires a non-null route"):
        verify_admission_report(
            report,
            arm_key="kronos_small",
            track="F",
            route=None,
            require_training=True,
            path=registry_path,
        )
    with pytest.raises(NativeContractError, match="undeclared training route"):
        verify_admission_report(
            report,
            arm_key="kronos_small",
            track="F",
            route="adjacent_half_contrastive",
            require_training=True,
            path=registry_path,
        )
    with pytest.raises(NativeContractError, match="is blocked"):
        verify_admission_report(
            report,
            arm_key="kronos_small",
            track="C",
            route="adjacent_half_contrastive",
            require_training=True,
            path=registry_path,
        )


def test_phase_a_rejects_self_authored_route_before_training_report_build(tmp_path):
    registry_path, registry, _ = _admitted_registry(
        tmp_path, arm_key="moment_small", track="R"
    )
    route = _admit_training_route(
        registry_path, "moment_small:C:classification"
    )
    environment = {**_measured_test_environment(), "use_scope": "production"}
    with pytest.raises(NativeContractError, match="Phase A forbids nonblocked"):
        build_admission_request(
            arm_key="moment_small", track="C", status="admitted",
            checks=_all_passing_checks(registry), environment=environment,
            route=route,
            require_training=True,
            artifacts={}, created_utc="2026-07-17T00:00:00Z", path=registry_path,
        )


def test_hash_bound_inference_report_requires_current_registry_and_two_approvals(tmp_path, monkeypatch):
    registry_path, registry, private_keys = _admitted_registry(tmp_path)
    artifacts = _runtime_artifacts(tmp_path, monkeypatch)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"admitted checkpoint")
    report = _signed_report(
        registry_path=registry_path, registry=registry, artifacts=artifacts,
        private_keys=private_keys, extra_artifacts={"checkpoint": checkpoint},
    )
    verified = verify_admission_report(
        report, arm_key="kronos_small", track="F", require_training=False,
        required_artifacts={**artifacts, "checkpoint": checkpoint},
        runtime_controls=_test_runtime_controls(), path=registry_path,
    )
    assert verified["status"] == "native_valid"

    with pytest.raises(NativeContractError, match="runtime controls must be supplied"):
        verify_admission_report(
            report, arm_key="kronos_small", track="F", require_training=False,
            required_artifacts={**artifacts, "checkpoint": checkpoint},
            path=registry_path,
        )
    with pytest.raises(NativeContractError, match="runtime controls must be supplied"):
        verify_admission_report(
            report, arm_key="kronos_small", track="F", require_training=False,
            required_artifacts={**artifacts, "checkpoint": checkpoint},
            runtime_controls={**_test_runtime_controls(), "device": "cpu"},
            path=registry_path,
        )

    replacement = tmp_path / "replacement.pt"
    replacement.write_bytes(b"different checkpoint")
    with pytest.raises(NativeContractError, match="artifact 'checkpoint' tree hash mismatch"):
        verify_admission_report(
                report, arm_key="kronos_small", track="F", require_training=False,
                required_artifacts={**artifacts, "checkpoint": replacement},
                runtime_controls=_test_runtime_controls(), path=registry_path,
        )

    tampered = copy.deepcopy(report)
    tampered["checks"]["fp32_finite"]["status"] = "fail"
    with pytest.raises(NativeContractError, match="integrity mismatch"):
        verify_admission_report(
            tampered, arm_key="kronos_small", track="F", require_training=False,
            path=registry_path,
        )

    one_reviewer = attach_integrity({
        **copy.deepcopy(report), "approvals": report["approvals"][:1]
    })
    with pytest.raises(NativeContractError, match="independent approvals"):
        verify_admission_report(
            one_reviewer, arm_key="kronos_small", track="F", require_training=False,
            path=registry_path,
        )

    aliased_reviewer = copy.deepcopy(report)
    aliased_reviewer["approvals"][1]["reviewer"] = " REVIEWER-A "
    aliased_reviewer = attach_integrity(aliased_reviewer)
    with pytest.raises(NativeContractError, match="not trusted for reviewer"):
        verify_admission_report(
            aliased_reviewer, arm_key="kronos_small", track="F",
            require_training=False, path=registry_path,
        )

    with pytest.raises(NativeContractError, match="floor cannot be lower than 2"):
        verify_admission_report(
            report, arm_key="kronos_small", track="F", require_training=False,
            minimum_approvals=0, path=registry_path,
        )

    invalid_finalization = copy.deepcopy(report)
    invalid_finalization["finalized_utc"] = "2026-07-17T00:02:00Z"
    invalid_finalization = attach_integrity(invalid_finalization)
    with pytest.raises(NativeContractError, match="latest authenticated approval"):
        verify_admission_report(
            invalid_finalization, arm_key="kronos_small", track="F",
            require_training=False, path=registry_path,
        )


def test_invented_reviewer_and_empty_trust_store_fail_closed(tmp_path, monkeypatch):
    registry_path, registry, private_keys = _admitted_registry(tmp_path)
    artifacts = _runtime_artifacts(tmp_path, monkeypatch)
    with pytest.raises(NativeContractError, match="not trusted for reviewer"):
        _signed_report(
            registry_path=registry_path, registry=registry,
            artifacts=artifacts, private_keys=private_keys,
            approvals_mutator=lambda values: values[0].update({"reviewer": "invented-a"}),
        )

    trust_path = registry_path.with_name("trusted_approvers.json")
    aliased = json.loads(trust_path.read_text(encoding="utf-8"))
    aliased["keys"]["reviewer-b-key"]["public_key_pem"] = (
        aliased["keys"]["reviewer-a-key"]["public_key_pem"]
    )
    trust_path.write_text(json.dumps(aliased), encoding="utf-8")
    with pytest.raises(NativeContractError, match="alias one public key"):
        _signed_report(
            registry_path=registry_path, registry=registry,
            artifacts=artifacts, private_keys=private_keys,
        )

    trust_path.write_text(
        json.dumps({"schema_version": "ffm_trusted_approvers_v1", "keys": {}}),
        encoding="utf-8",
    )
    with pytest.raises(NativeContractError, match="registry is empty"):
        _signed_report(
            registry_path=registry_path, registry=registry,
            artifacts=artifacts, private_keys=private_keys,
        )


def test_runner_is_measured_and_distribution_record_binds_imported_code(tmp_path, monkeypatch):
    registry_path, registry, private_keys = _admitted_registry(tmp_path)
    artifacts = _runtime_artifacts(tmp_path, monkeypatch)
    with pytest.raises(NativeContractError, match="runner artifact is measured"):
        _signed_report(
            registry_path=registry_path, registry=registry,
            artifacts={**artifacts, "runner": tmp_path}, private_keys=private_keys,
        )
    monkeypatch.undo()
    site = tmp_path / "site"
    dist_info = site / "example-1.0.dist-info"
    dist_info.mkdir(parents=True)
    module = site / "example.py"
    module.write_bytes(b"VALUE = 1\n")
    digest = base64.urlsafe_b64encode(
        __import__("hashlib").sha256(module.read_bytes()).digest()
    ).decode("ascii").rstrip("=")
    (dist_info / "RECORD").write_text(
        f"example.py,sha256={digest},{module.stat().st_size}\n"
        "example-1.0.dist-info/RECORD,,\n",
        encoding="utf-8",
    )
    assert native_contracts._runtime_artifact_description("source", dist_info)["kind"] == "directory"
    module.write_bytes(b"VALUE = 2\n")
    with pytest.raises(NativeContractError, match="RECORD hash mismatch"):
        native_contracts._runtime_artifact_description("source", dist_info)


def test_verification_remeasures_environment_and_runtime_artifact_trees(tmp_path, monkeypatch):
    registry_path, registry, private_keys = _admitted_registry(tmp_path)
    artifacts = _runtime_artifacts(tmp_path, monkeypatch)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    report = _signed_report(
        registry_path=registry_path, registry=registry, artifacts=artifacts,
        private_keys=private_keys, extra_artifacts={"checkpoint": checkpoint},
    )
    execution_artifacts = {**artifacts, "checkpoint": checkpoint}

    monkeypatch.setattr(platform, "python_version", lambda: "0.0.0")
    with pytest.raises(NativeContractError, match="measured runtime environment drift"):
        verify_admission_report(
            report, arm_key="kronos_small", track="F", require_training=False,
            required_artifacts=execution_artifacts,
            runtime_controls=_test_runtime_controls(), path=registry_path,
        )
    monkeypatch.undo()
    # Restore the technical-artifact test seam after undoing all monkeypatches.
    from futures_foundation.finetune import native_parity_runtime
    monkeypatch.setattr(
        native_parity_runtime, "measure_runtime_lock",
        lambda: copy.deepcopy(_test_runtime_lock()),
    )
    technical = {
        name: native_contracts._runtime_artifact_description(name, path)
        for name, path in artifacts.items()
    }
    runner = copy.deepcopy(technical["source"])
    runner["path"] = str(artifacts["source"])
    technical["runner"] = runner
    monkeypatch.setattr(
        native_contracts, "_technical_runtime_artifacts",
        lambda *args, **kwargs: copy.deepcopy(technical),
    )
    monkeypatch.setattr(
        native_contracts, "_current_execution_artifact",
        lambda: copy.deepcopy(runner),
    )

    drifted_lock = _test_runtime_lock()
    drifted_lock["portable_software"]["distributions"][0]["version"] = "2.0"
    monkeypatch.setattr(
        native_parity_runtime, "measure_runtime_lock",
        lambda: copy.deepcopy(drifted_lock),
    )
    with pytest.raises(NativeContractError, match="complete runtime lock drifted"):
        verify_admission_report(
            report, arm_key="kronos_small", track="F", require_training=False,
            required_artifacts=execution_artifacts,
            runtime_controls=_test_runtime_controls(), path=registry_path,
        )
    monkeypatch.setattr(
        native_parity_runtime, "measure_runtime_lock",
        lambda: copy.deepcopy(_test_runtime_lock()),
    )

    (artifacts["source"] / "untracked.py").write_text("raise RuntimeError\n")
    with pytest.raises(NativeContractError, match="untracked files"):
        verify_admission_report(
            report, arm_key="kronos_small", track="F", require_training=False,
            required_artifacts=execution_artifacts,
            runtime_controls=_test_runtime_controls(), path=registry_path,
        )
    (artifacts["source"] / "untracked.py").unlink()

    artifacts["model"].write_bytes(b"tampered-model")
    with pytest.raises(NativeContractError, match="tree hash mismatch"):
        verify_admission_report(
            report, arm_key="kronos_small", track="F", require_training=False,
            required_artifacts=execution_artifacts,
            runtime_controls=_test_runtime_controls(), path=registry_path,
        )


def test_admission_report_rejects_unsupported_check_claims_and_bad_approval_time(tmp_path, monkeypatch):
    registry_path, registry, private_keys = _admitted_registry(tmp_path)
    artifacts = _runtime_artifacts(tmp_path, monkeypatch)
    unsupported = _all_passing_checks(registry)
    unsupported["fp32_finite"] = {"status": "pass"}
    with pytest.raises(NativeContractError, match="needs concrete evidence"):
        build_admission_report(
            arm_key="kronos_small", track="F", status="native_valid",
            checks=unsupported,
            approvals=[],
            environment=_measured_test_environment(),
            artifacts=artifacts,
            created_utc="2026-07-17T00:00:00Z", path=registry_path,
        )

    with pytest.raises(NativeContractError, match="predates the signed admission request"):
        _signed_report(
            registry_path=registry_path, registry=registry,
            artifacts=artifacts, private_keys=private_keys,
            approvals_mutator=lambda values: values[1].update(
                {"approved_utc": "2026-07-16T23:59:00Z"}
            ),
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


def test_report_environment_must_match_measured_runtime():
    expected = _measured_test_environment()
    with pytest.raises(NativeContractError, match="disagrees with measurement"):
        measure_runtime_environment(expected, {**expected, "python": "0.0.0"})


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
