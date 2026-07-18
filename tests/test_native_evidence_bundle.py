import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from futures_foundation.finetune.native_contracts import (
    EVIDENCE_SCHEMA,
    load_registry,
)
from futures_foundation.finetune.native_evidence_bundle import (
    AGGREGATE_SCHEMA,
    NativeEvidenceError,
    aggregate_parity_bundles,
    create_shared_fixture,
    run_parity_bundle,
    verify_parity_bundle,
    verify_shared_fixture,
    _tree_description,
)


def _registry_copy(tmp_path: Path) -> Path:
    source = Path("config/foundation_models")
    target = tmp_path / "config"
    target.mkdir()
    shutil.copy2(source / "native_contracts.json", target / "native_contracts.json")
    shutil.copy2(
        source / "native_contract_evidence.json", target / "native_contract_evidence.json"
    )
    return target / "native_contracts.json"


def _runner(path: Path, required_checks: list[str]) -> None:
    checks = {
        name: (
            {"status": "pass", "evidence": f"raw:{name}"}
            if name in {
                "official_example", "adapter_public_api_parity", "fp32_finite",
                "license_governance",
            }
            else {"status": "not_applicable", "reason": f"not required by test {name}"}
        )
        for name in required_checks
    }
    payload = {
        'schema_version': 'ffm_native_parity_result_v1',
        'arm_key': 'mantis_v1',
        'track': 'R',
        'status': 'pass',
        'environment': {'python': 'test', 'dtype': 'float32'},
        'admitted_runtime': {'context_length': 512, 'dtype': 'float32'},
        'metrics': {'finite': True, 'shape': [4, 16]},
        'checks': checks,
        'output_files': ['official_output.npy'],
    }
    source = f"""
import json, os
from pathlib import Path
import numpy as np
values = np.load(os.environ['FFM_NATIVE_PARITY_VALUES'], allow_pickle=False)
assert Path(os.environ['FFM_NATIVE_PARITY_ARTIFACT_MODEL']).exists()
assert Path(os.environ['FFM_NATIVE_PARITY_ARTIFACT_SOURCE']).exists()
out = Path(os.environ['FFM_NATIVE_PARITY_RESULT_DIR'])
np.save(out / 'official_output.npy', values[:, -16:, 3])
result = {payload!r}
(out / 'result.json').write_text(json.dumps(result, sort_keys=True), encoding='utf-8')
print('real parity fixture consumed')
"""
    path.write_text(source, encoding="utf-8")


def _bundle(tmp_path: Path):
    registry_path = _registry_copy(tmp_path)
    registry = load_registry(registry_path)
    runner = tmp_path / "parity_runner.py"
    _runner(runner, registry["required_checks"])
    model = tmp_path / "model.bin"
    model.write_bytes(b"exact-model-weights")
    source = tmp_path / "source.py"
    source.write_text("# exact source\n", encoding="utf-8")
    bundle = tmp_path / "bundle"
    manifest = run_parity_bundle(
        arm_key="mantis_v1",
        track="R",
        command=[sys.executable, str(runner)],
        output_directory=bundle,
        artifacts={"model": model, "source": source},
        environment={"CUDA_VISIBLE_DEVICES": ""},
        created_utc="2026-07-17T12:00:00Z",
        registry_path=registry_path,
    )
    return registry_path, bundle, manifest


def test_shared_fixture_is_byte_deterministic_and_tamper_evident(tmp_path):
    left = create_shared_fixture(tmp_path / "left")
    right = create_shared_fixture(tmp_path / "right")
    assert left == right
    assert (tmp_path / "left/ohlcv_f32.npy").read_bytes() == (
        tmp_path / "right/ohlcv_f32.npy"
    ).read_bytes()
    assert verify_shared_fixture(tmp_path / "left") == left
    values = np.load(tmp_path / "left/ohlcv_f32.npy", allow_pickle=False)
    values[0, 0, 0] += 1
    np.save(tmp_path / "left/ohlcv_f32.npy", values, allow_pickle=False)
    with pytest.raises(NativeEvidenceError, match="bytes do not match"):
        verify_shared_fixture(tmp_path / "left")


def test_bundle_binds_command_fixture_artifacts_outputs_and_logs(tmp_path):
    registry_path, bundle, manifest = _bundle(tmp_path)
    verified, result = verify_parity_bundle(bundle, registry_path=registry_path)
    assert verified == manifest
    assert result["status"] == "pass"
    assert set(manifest["bound_artifacts"]) == {"model", "source"}
    assert len(manifest["command"]["command_sha256"]) == 64
    assert len(manifest["fixture"]["fixture_sha256"]) == 64
    assert len(manifest["raw_outputs"]["official_output.npy"]["sha256"]) == 64
    assert manifest["raw_outputs"]["official_output.npy"]["path"] == (
        "raw/official_output.npy"
    )
    assert manifest["command"]["file_arguments"][0]["argv_index"] == 1
    assert b"real parity fixture consumed" in (bundle / "stdout.log").read_bytes()


def test_bundle_verification_rejects_output_and_model_tampering(tmp_path):
    registry_path, bundle, _ = _bundle(tmp_path)
    output = bundle / "raw/official_output.npy"
    output.write_bytes(output.read_bytes() + b"tamper")
    with pytest.raises(NativeEvidenceError, match="raw_outputs"):
        verify_parity_bundle(bundle, registry_path=registry_path)

    other = tmp_path / "other"
    other.mkdir()
    registry_path, bundle, _ = _bundle(other)
    Path(json.loads((bundle / "bundle_manifest.json").read_text())["bound_artifacts"]["model"]["path"]).write_bytes(b"changed")
    with pytest.raises(NativeEvidenceError, match="bound_artifacts.model drifted"):
        verify_parity_bundle(bundle, registry_path=registry_path)


def test_bundle_binds_hf_file_symlinks_and_worker_source(tmp_path):
    registry_path, bundle, manifest = _bundle(tmp_path)
    runner_path = Path(manifest["command"]["file_arguments"][0]["artifact"]["path"])
    runner_path.write_text("# drifted worker", encoding="utf-8")
    with pytest.raises(NativeEvidenceError, match="command.file_arguments"):
        verify_parity_bundle(bundle, registry_path=registry_path)

    other = tmp_path / "hf"
    other.mkdir()
    registry_path = _registry_copy(other)
    registry = load_registry(registry_path)
    runner = other / "runner.py"
    _runner(runner, registry["required_checks"])
    source = other / "source.py"
    source.write_text("# source", encoding="utf-8")
    blob = other / "blob.bin"
    blob.write_bytes(b"checkpoint shard")
    snapshot = other / "snapshot"
    snapshot.mkdir()
    (snapshot / "model.safetensors").symlink_to(blob)
    bundle = other / "bundle"
    manifest = run_parity_bundle(
        arm_key="mantis_v1", track="R", command=[sys.executable, str(runner)],
        output_directory=bundle, artifacts={"model": snapshot, "source": source},
        registry_path=registry_path,
    )
    entry = manifest["bound_artifacts"]["model"]["entries"][0]
    assert entry["kind"] == "file_symlink"
    assert entry["link_target"] == str(blob)
    verify_parity_bundle(bundle, registry_path=registry_path)
    blob.write_bytes(b"changed shard")
    with pytest.raises(NativeEvidenceError, match="bound_artifacts.model drifted"):
        verify_parity_bundle(bundle, registry_path=registry_path)


def test_registry_declared_reference_artifact_is_required_and_passed_to_worker(tmp_path):
    registry_path = _registry_copy(tmp_path)
    registry_json = json.loads(registry_path.read_text(encoding="utf-8"))
    registry_json["models"]["mantis_v1"]["native_parity"] = {
        "required_artifacts": ["reference_model"],
        "reference_model_id": "example/reference-model",
        "reference_model_revision": "reference-revision",
    }
    registry_path.write_text(json.dumps(registry_json), encoding="utf-8")
    load_registry.cache_clear()
    registry = load_registry(registry_path)
    runner = tmp_path / "runner.py"
    _runner(runner, registry["required_checks"])
    # Make the real worker prove it received the extra artifact path.
    runner.write_text(
        runner.read_text(encoding="utf-8").replace(
            "out = Path(os.environ['FFM_NATIVE_PARITY_RESULT_DIR'])",
            "assert Path(os.environ['FFM_NATIVE_PARITY_ARTIFACT_REFERENCE_MODEL']).exists()\n"
            "out = Path(os.environ['FFM_NATIVE_PARITY_RESULT_DIR'])",
        ),
        encoding="utf-8",
    )
    model = tmp_path / "model.bin"
    source = tmp_path / "source.py"
    reference = tmp_path / "reference.bin"
    model.write_bytes(b"model")
    source.write_text("# source", encoding="utf-8")
    reference.write_bytes(b"reference")
    with pytest.raises(NativeEvidenceError, match="reference_model"):
        run_parity_bundle(
            arm_key="mantis_v1", track="R", command=[sys.executable, str(runner)],
            output_directory=tmp_path / "missing", artifacts={"model": model, "source": source},
            registry_path=registry_path,
        )
    manifest = run_parity_bundle(
        arm_key="mantis_v1", track="R", command=[sys.executable, str(runner)],
        output_directory=tmp_path / "bundle",
        artifacts={"model": model, "source": source, "reference_model": reference},
        registry_path=registry_path,
    )
    assert set(manifest["bound_artifacts"]) == {"model", "source", "reference_model"}


def test_git_checkout_hash_ignores_clone_metadata_and_detects_tracked_drift(tmp_path):
    source = tmp_path / "source_repo"
    source.mkdir()
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(source), "config", "user.name", "Test"], check=True)
    (source / "adapter.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(source), "add", "adapter.py"], check=True)
    subprocess.run(["git", "-C", str(source), "commit", "-q", "-m", "fixture"], check=True)
    origin = "https://github.com/example/native-model.git"
    subprocess.run(["git", "-C", str(source), "remote", "add", "origin", origin], check=True)
    left = tmp_path / "left"
    right = tmp_path / "right"
    subprocess.run(["git", "clone", "-q", str(source), str(left)], check=True)
    subprocess.run(["git", "clone", "-q", str(source), str(right)], check=True)
    for clone in (left, right):
        subprocess.run(["git", "-C", str(clone), "remote", "set-url", "origin", origin], check=True)
    # Clone-local logs, hooks, caches and untracked files must not affect tracked identity.
    (left / ".git/hooks/local-hook.sample").write_text("left", encoding="utf-8")
    (right / ".git/hooks/local-hook.sample").write_text("right", encoding="utf-8")
    (left / "__pycache__").mkdir()
    (left / "__pycache__/adapter.pyc").write_bytes(b"generated-left")
    (right / "scratch.log").write_text("generated-right", encoding="utf-8")
    left_description = _tree_description(left)
    right_description = _tree_description(right)
    assert left_description["kind"] == "git_checkout"
    assert left_description["sha256"] == right_description["sha256"]
    assert left_description["entries"] == right_description["entries"]
    assert left_description["origin"] == "https://github.com/example/native-model"
    with pytest.raises(NativeEvidenceError, match="untracked files"):
        _tree_description(left, git_untracked_policy="reject")
    (left / "adapter.py").write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(NativeEvidenceError, match="tracked/index drift"):
        _tree_description(left)


def test_run_is_fail_closed_on_missing_artifact_and_incomplete_checks(tmp_path):
    registry_path = _registry_copy(tmp_path)
    registry = load_registry(registry_path)
    runner = tmp_path / "runner.py"
    _runner(runner, registry["required_checks"])
    source = tmp_path / "source.py"
    source.write_text("# source", encoding="utf-8")
    with pytest.raises(NativeEvidenceError, match=r"missing=\['model'\]"):
        run_parity_bundle(
            arm_key="mantis_v1", track="R", command=[sys.executable, str(runner)],
            output_directory=tmp_path / "bundle", artifacts={"source": source},
            registry_path=registry_path,
        )
    incomplete = tmp_path / "incomplete.py"
    checks = list(registry["required_checks"][:-1])
    payload = {
        "schema_version": "ffm_native_parity_result_v1",
        "arm_key": "mantis_v1",
        "track": "R",
        "status": "pass",
        "environment": {"python": "test"},
        "admitted_runtime": {"context_length": 512},
        "metrics": {"finite": True},
        "checks": {
            name: {"status": "pass", "evidence": f"raw:{name}"} for name in checks
        },
        "output_files": ["output.npy"],
    }
    incomplete.write_text(
        "import json, os\n"
        "from pathlib import Path\n"
        "out=Path(os.environ['FFM_NATIVE_PARITY_RESULT_DIR'])\n"
        "(out/'output.npy').write_bytes(b'raw')\n"
        f"(out/'result.json').write_text(json.dumps({payload!r}), encoding='utf-8')\n",
        encoding="utf-8",
    )
    model = tmp_path / "model.bin"
    model.write_bytes(b"model")
    with pytest.raises(NativeEvidenceError, match="checks must exactly match"):
        run_parity_bundle(
            arm_key="mantis_v1", track="R", command=[sys.executable, str(incomplete)],
            output_directory=tmp_path / "incomplete_bundle",
            artifacts={"model": model, "source": source}, registry_path=registry_path,
        )


def test_aggregate_generates_candidate_but_cannot_install_it(tmp_path):
    registry_path, bundle, _ = _bundle(tmp_path)
    destination = tmp_path / "candidate.json"
    aggregate = aggregate_parity_bundles(
        [bundle], output_path=destination,
        generated_utc="2026-07-17T13:00:00Z", require_all_current=False,
        registry_path=registry_path,
    )
    assert aggregate["schema_version"] == AGGREGATE_SCHEMA
    assert aggregate["candidate_evidence"]["schema_version"] == EVIDENCE_SCHEMA
    assert set(aggregate["candidate_evidence"]["records"]) == {
        "mantis_v1:R:2026-07-18"
    }
    assert destination.is_file()
    assert aggregate["candidate_evidence"]["records"][
        "mantis_v1:R:2026-07-18"
    ]["bundle"]["path"] == "bundle"
    assert aggregate["candidate_evidence"]["records"][
        "mantis_v1:R:2026-07-18"
    ]["bundle"]["path_base"] == "aggregate_parent"
    with pytest.raises(NativeEvidenceError, match="may not overwrite canonical evidence"):
        aggregate_parity_bundles(
            [bundle], output_path=registry_path.with_name("native_contract_evidence.json"),
            require_all_current=False, registry_path=registry_path,
        )


def test_aggregate_requires_complete_current_coverage_by_default(tmp_path):
    registry_path, bundle, _ = _bundle(tmp_path)
    with pytest.raises(NativeEvidenceError, match="bundle coverage differs"):
        aggregate_parity_bundles(
            [bundle], output_path=tmp_path / "aggregate.json",
            registry_path=registry_path,
        )
