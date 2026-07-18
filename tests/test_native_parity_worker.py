import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import pytest

from futures_foundation.finetune.native_adapters import (
    NativeAdapterError,
    chronos_native_embedding,
)
from scripts import native_parity_worker as worker
from futures_foundation.finetune.native_parity_runtime import (
    NativeParityRuntimeError,
    validate_runtime_lock,
)


EXPECTED_RUNNERS = {
    "mantis_v1", "mantis_v2", "moment_small", "kronos_mini", "kronos_small",
    "chronos_v1", "chronos_bolt", "chronos_v2", "timesfm25", "ttm_r2",
    "moirai2_small", "toto2_22m", "sundial_base",
}


def _runtime_lock():
    return {
        "schema_version": "ffm_native_runtime_lock_v1",
        "comparison_policy": {
            "portable_software": "exact",
            "hardware_runtime": "exact_when_measurable_explicit_when_unavailable",
        },
        "portable_software": {
            "python_executable": sys.executable,
            "python_version": "test",
            "python_implementation": "CPython",
            "platform": "test",
            "distributions": [{"name": "torch", "version": "test"}],
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


def test_every_real_arm_has_one_explicit_profile_and_dispatch():
    assert set(worker.RUNNERS) == EXPECTED_RUNNERS
    assert set().union(*worker.PROFILE_ARMS.values()) == EXPECTED_RUNNERS
    assert sum(map(len, worker.PROFILE_ARMS.values())) == len(EXPECTED_RUNNERS)
    for arm in EXPECTED_RUNNERS:
        assert worker.runtime_profile_for_arm(arm) in worker.PROFILE_ARMS


def test_runtime_lock_rejects_nested_drift_and_duplicate_normalized_names():
    assert validate_runtime_lock(_runtime_lock()) == _runtime_lock()
    duplicate = copy.deepcopy(_runtime_lock())
    duplicate["portable_software"]["distributions"].append(
        {"name": "torch", "version": "other"}
    )
    with pytest.raises(NativeParityRuntimeError, match="unique and sorted"):
        validate_runtime_lock(duplicate)
    nested = copy.deepcopy(_runtime_lock())
    nested["hardware_runtime"]["unknown"] = True
    with pytest.raises(NativeParityRuntimeError, match="hardware fields drifted"):
        validate_runtime_lock(nested)


def test_affine_contract_allows_quantization_but_rejects_missing_inverse_scale():
    expected = np.array([132.0, 3544.0], dtype=np.float32)
    quantized = expected + np.array([0.0005, 1.5], dtype=np.float32)
    assert worker._affine_evidence(
        quantized, expected, label="test"
    )["passed"] is True
    assert worker._affine_evidence(
        expected / 1.25, expected, label="test"
    )["passed"] is False


def test_profile_selection_rejects_wrong_family(monkeypatch):
    monkeypatch.setattr(worker.sys, "version_info", (3, 12, 0))
    with pytest.raises(worker.WorkerError, match="requires runtime profile"):
        worker.validate_runtime_profile("common", "ttm_r2")


def test_bound_artifact_rejects_unsealed_or_different_path(tmp_path, monkeypatch):
    supplied = tmp_path / "supplied"
    supplied.write_text("x")
    with pytest.raises(worker.WorkerError, match="requires bound artifact"):
        worker.bound_artifact("model", supplied)
    bound = tmp_path / "bound"
    bound.write_text("x")
    monkeypatch.setenv("FFM_NATIVE_PARITY_ARTIFACT_MODEL", str(bound))
    with pytest.raises(worker.WorkerError, match="differs from sealed"):
        worker.bound_artifact("model", supplied)
    assert worker.bound_artifact("model", bound) == bound.resolve()


def _git(path: Path, *args: str):
    return subprocess.run(
        ["git", "-C", str(path), *args], check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    ).stdout.strip()


def test_source_validation_requires_exact_clean_head_and_remote(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _git(source, "init")
    _git(source, "config", "user.email", "test@example.com")
    _git(source, "config", "user.name", "Test")
    (source / "module.py").write_text("# pinned\n")
    _git(source, "add", "module.py")
    _git(source, "commit", "-m", "pinned")
    revision = _git(source, "rev-parse", "HEAD")
    _git(source, "remote", "add", "origin", "git@github.com:owner/repo.git")
    assert worker.validate_source_checkout(
        source, revision=revision, source_url="https://github.com/owner/repo"
    ) == source.resolve()
    (source / "dirty.txt").write_text("dirty")
    with pytest.raises(worker.WorkerError, match="dirty"):
        worker.validate_source_checkout(
            source, revision=revision, source_url="https://github.com/owner/repo"
        )


def test_execute_dispatches_sealed_fixture_and_writes_native_result(tmp_path, monkeypatch):
    registry = worker.load_registry()
    dossier = registry["models"]["chronos_v1"]
    snapshot = tmp_path / dossier["model_revision"]
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}")
    result_dir = tmp_path / "result"
    monkeypatch.setenv("FFM_NATIVE_PARITY_ARM", "chronos_v1")
    monkeypatch.setenv("FFM_NATIVE_PARITY_TRACK", "F")
    monkeypatch.setenv("FFM_NATIVE_PARITY_ARTIFACT_MODEL", str(snapshot))
    monkeypatch.setenv("FFM_NATIVE_PARITY_RESULT_DIR", str(result_dir))
    monkeypatch.setattr(worker, "GIT_SOURCE_ARMS", set())
    monkeypatch.setattr(worker, "PACKAGE_SOURCE_ARMS", set())
    monkeypatch.setattr(
        worker, "validate_runtime_profile",
        lambda profile, arm: {"torch": "test"},
    )
    monkeypatch.setattr(worker, "install_python_network_guard", lambda policy: None)
    monkeypatch.setattr(worker, "measure_runtime_lock", _runtime_lock)
    fixture = (
        np.ones((4, 512, 5), np.float32),
        np.arange(4 * 512, dtype=np.int64).reshape(4, 512),
    )
    monkeypatch.setattr(worker, "_load_fixture", lambda: fixture)
    monkeypatch.setattr(
        worker,
        "_license_evidence",
        lambda args, dossier: (
            worker._invariant(True, "mock bound license artifacts"),
            {"mock": True},
        ),
    )
    seen = {}

    def fake_runner(args, values, *, track):
        seen.update(arm=args.arm, track=track, shape=values.shape)
        output = np.ones((4, 16, 3), np.float32)
        return {
            "arrays": {
                "official": output,
                "adapter": output.copy(),
                "partitioned": output.copy(),
            },
            "parity_error": 0.0,
            "batch_error": 0.0,
            "runtime": {"context_length": 512, "prediction_length": 16},
            "metrics": {"finite": True},
            "channel": worker._invariant(True, "mock native channels"),
                "padding": worker._invariant(True, "mock native mask"),
                "frequency": worker._invariant(True, "mock native frequency"),
            "scaling": worker._invariant(True, "mock native scaling"),
        }

    monkeypatch.setitem(worker.RUNNERS, "chronos_v1", fake_runner)
    args = argparse.Namespace(
        arm="chronos_v1", track="F", profile="common",
        model_snapshot=str(snapshot), tokenizer_snapshot=None,
        reference_model_snapshot=None, source_repo=None, device="cpu",
        batch_size=4, samples=2, seed=7,
        network_policy="python_socket_deny",
    )
    worker.execute(args)
    assert seen == {"arm": "chronos_v1", "track": "F", "shape": (4, 512, 5)}
    result = json.loads((result_dir / "result.json").read_text())
    assert result["status"] == "pass"
    assert result["metrics"]["adapter_public_api_max_abs"] == 0.0
    assert result["metrics"]["adapter_public_api_allclose"] is True
    assert result["metrics"]["native_parity_atol"] == 1e-5
    assert result["runtime_lock"] == _runtime_lock()
    with np.load(result_dir / "native_outputs.npz") as arrays:
        assert set(arrays.files) == {"official", "adapter", "partitioned"}


def test_execute_fails_and_records_out_of_tolerance_public_output(tmp_path, monkeypatch):
    registry = worker.load_registry()
    dossier = registry["models"]["chronos_v1"]
    snapshot = tmp_path / dossier["model_revision"]
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}")
    result_dir = tmp_path / "result"
    monkeypatch.setenv("FFM_NATIVE_PARITY_ARM", "chronos_v1")
    monkeypatch.setenv("FFM_NATIVE_PARITY_TRACK", "F")
    monkeypatch.setenv("FFM_NATIVE_PARITY_ARTIFACT_MODEL", str(snapshot))
    monkeypatch.setenv("FFM_NATIVE_PARITY_RESULT_DIR", str(result_dir))
    monkeypatch.setattr(worker, "GIT_SOURCE_ARMS", set())
    monkeypatch.setattr(worker, "PACKAGE_SOURCE_ARMS", set())
    monkeypatch.setattr(
        worker, "validate_runtime_profile", lambda profile, arm: {"torch": "test"},
    )
    monkeypatch.setattr(worker, "install_python_network_guard", lambda policy: None)
    monkeypatch.setattr(worker, "measure_runtime_lock", _runtime_lock)
    monkeypatch.setattr(worker, "_load_fixture", lambda: (
        np.ones((4, 512, 5), np.float32),
        np.arange(4 * 512, dtype=np.int64).reshape(4, 512),
    ))

    monkeypatch.setattr(
        worker,
        "_license_evidence",
        lambda args, dossier: (
            worker._invariant(True, "mock bound license artifacts"),
            {"mock": True},
        ),
    )

    def bad_runner(args, values, *, track):
        official = np.ones((4, 16, 3), np.float32)
        adapter = official.copy()
        adapter[0, 0, 0] += 1.0
        return {
            "arrays": {"official": official, "adapter": adapter},
            "parity_error": 1.0,
            "batch_error": None,
            "runtime": {"context_length": 512, "prediction_length": 16},
            "metrics": {},
                "channel": worker._invariant(True, "mock native channels"),
            "padding": None,
            "frequency": None,
                "scaling": worker._invariant(True, "mock native scaling"),
        }

    monkeypatch.setitem(worker.RUNNERS, "chronos_v1", bad_runner)
    args = argparse.Namespace(
        arm="chronos_v1", track="F", profile="common",
        model_snapshot=str(snapshot), tokenizer_snapshot=None,
        reference_model_snapshot=None, source_repo=None, device="cpu",
        batch_size=4, samples=2, seed=7,
    )
    with pytest.raises(worker.WorkerError, match="parity checks failed"):
        worker.execute(args)
    result = json.loads((result_dir / "result.json").read_text())
    assert result["status"] == "fail"
    assert result["metrics"]["adapter_public_api_allclose"] is False
    assert result["checks"]["adapter_public_api_parity"]["status"] == "fail"


def test_native_parity_report_rejects_out_of_tolerance_partition():
    registry = worker.load_registry()
    official = np.ones((2, 3), np.float32)
    partitioned = official.copy()
    partitioned[0, 0] += 0.1
    report = worker._native_parity_report(
        {
            "official": official,
            "adapter": official.copy(),
            "partitioned": partitioned,
        },
        registry=registry,
        require_partition=True,
    )
    assert report["public_pass"] is True
    assert report["batch_pass"] is False
    assert report["batch_pairs"][0]["allclose"] is False


def test_invariant_checks_cannot_pass_from_prose_or_failed_measurement():
    with pytest.raises(worker.WorkerError, match="passed/evidence"):
        worker._check_invariant("descriptive prose", not_applicable="unused")
    failed = worker._check_invariant(
        worker._invariant(False, "masked-value max_abs=0.5"),
        not_applicable="unused",
    )
    assert failed == {
        "status": "fail", "evidence": "masked-value max_abs=0.5",
    }


@pytest.mark.parametrize("arm", ["chronos_v1", "chronos_bolt", "chronos_v2"])
def test_chronos_forecast_runner_measures_scale_and_missing_values(arm, monkeypatch):
    import torch

    class FakePipeline:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def predict_quantiles(
            self, inputs, *, prediction_length, quantile_levels, **kwargs
        ):
            tensor = torch.as_tensor(inputs, dtype=torch.float32)
            center = torch.nanmean(tensor, dim=-1)
            output = center[..., None, None].repeat(
                *([1] * center.ndim), prediction_length, len(quantile_levels)
            )
            if arm == "chronos_v2":
                return [row for row in output], None
            return output, None

    monkeypatch.setitem(
        sys.modules, "chronos",
        types.SimpleNamespace(
            BaseChronosPipeline=FakePipeline,
            Chronos2Pipeline=FakePipeline,
        ),
    )
    args = argparse.Namespace(
        arm=arm,
        model_snapshot="unused",
        device="cpu",
        batch_size=2,
        samples=4,
        seed=7,
    )
    values = np.linspace(10.0, 50.0, 4 * 512 * 5, dtype=np.float32).reshape(
        4, 512, 5
    )
    outcome = worker._run_chronos(args, values, track="F")
    assert outcome["scaling"]["passed"] is True
    assert outcome["padding"]["passed"] is True
    assert outcome["metrics"]["affine_scale"] == 1.25
    assert outcome["metrics"]["affine_shift"] == 0.0
    assert outcome["metrics"]["missing_prefix_length"] == 16
    assert outcome["metrics"]["missing_outputs_finite"] is True
    assert {
        "scaling_observed", "scaling_expected",
        "missing_reference", "missing_adapter",
    }.issubset(outcome["arrays"])


class _ChronosEmbeddingPipeline:
    def embed(self, context):
        import torch
        values = torch.as_tensor(context, dtype=torch.float32)
        embedding = values[:, :, None].repeat(1, 1, 3)
        state = (values.mean(dim=1, keepdim=True), values.std(dim=1, keepdim=True))
        return embedding, state


def test_chronos_native_embedding_preserves_unpooled_tokens_and_state():
    import torch
    context = torch.arange(24, dtype=torch.float32).reshape(2, 12)
    embedding, state = chronos_native_embedding(_ChronosEmbeddingPipeline(), context)
    assert embedding.shape == (2, 12, 3)
    assert state[0].shape == (2, 1)
    assert state[1].shape == (2, 1)


def test_chronos_native_embedding_rejects_nonfinite_state():
    import torch

    class Bad(_ChronosEmbeddingPipeline):
        def embed(self, context):
            embedding, state = super().embed(context)
            state[0][0] = torch.nan
            return embedding, state

    with pytest.raises(NativeAdapterError, match=r"state\[0\].*non-finite"):
        chronos_native_embedding(Bad(), torch.ones(2, 12))


def test_python_network_guard_denies_deliberate_socket_attempt():
    source = (
        "import socket;"
        "from scripts.native_parity_worker import install_python_network_guard,WorkerError;"
        "install_python_network_guard('python_socket_deny');"
        "\ntry: socket.getaddrinfo('example.com',443)\n"
        "except WorkerError: raise SystemExit(0)\n"
        "raise SystemExit(9)"
    )
    completed = subprocess.run([sys.executable, "-c", source], check=False)
    assert completed.returncode == 0
