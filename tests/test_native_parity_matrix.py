import json
import base64
import hashlib
from pathlib import Path
import subprocess
import sys

import pytest

from futures_foundation.finetune.native_contracts import load_registry
from futures_foundation.finetune import native_parity_matrix as matrix
from futures_foundation.finetune.native_parity_runtime import validate_distribution_record


def _snapshot(root: Path, model_id: str, revision: str) -> Path:
    path = root / ("models--" + model_id.replace("/", "--")) / "snapshots" / revision
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"sealed weights")
    return path


def _chronos_registry() -> dict:
    registry = load_registry()
    return {
        **registry,
        "models": {"chronos_v1": registry["models"]["chronos_v1"]},
    }


def test_registry_matrix_covers_every_admitted_native_track_including_chronos_r():
    pairs = set(matrix.admitted_native_pairs(load_registry()))
    assert len(pairs) == 16
    assert {
        ("chronos_v1", "F"), ("chronos_v1", "R"),
        ("chronos_bolt", "F"), ("chronos_bolt", "R"),
        ("chronos_v2", "F"), ("chronos_v2", "R"),
    }.issubset(pairs)
    assert not any(arm == "tabpfn_ts" for arm, _ in pairs)


def test_offline_snapshot_resolution_requires_one_exact_materialized_commit(tmp_path):
    revision = "a" * 40
    expected = _snapshot(tmp_path / "left", "owner/model", revision)
    assert matrix.resolve_offline_snapshot(
        [tmp_path / "left"], model_id="owner/model", revision=revision
    ) == expected.resolve()
    with pytest.raises(matrix.NativeParityMatrixError, match="found 0 matches"):
        matrix.resolve_offline_snapshot(
            [tmp_path / "missing"], model_id="owner/model", revision=revision
        )
    _snapshot(tmp_path / "right", "owner/model", revision)
    with pytest.raises(matrix.NativeParityMatrixError, match="found 2 matches"):
        matrix.resolve_offline_snapshot(
            [tmp_path / "left", tmp_path / "right"],
            model_id="owner/model", revision=revision,
        )
    with pytest.raises(matrix.NativeParityMatrixError, match="40-hex"):
        matrix.resolve_offline_snapshot(
            [tmp_path / "left"], model_id="owner/model", revision="main"
        )


def test_snapshot_preflight_rejects_missing_index_shard(tmp_path):
    revision = "b" * 40
    snapshot = _snapshot(tmp_path, "owner/sharded", revision)
    (snapshot / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"layer": "model-00002-of-00002.safetensors"}
    }), encoding="utf-8")
    with pytest.raises(matrix.NativeParityMatrixError, match="missing shards"):
        matrix.resolve_offline_snapshot(
            [tmp_path], model_id="owner/sharded", revision=revision
        )


def test_snapshot_preflight_rejects_missing_remote_code_module(tmp_path):
    revision = "c" * 40
    snapshot = _snapshot(tmp_path, "owner/remote", revision)
    (snapshot / "config.json").write_text(json.dumps({
        "auto_map": {"AutoModel": "modeling_remote.RemoteModel"}
    }), encoding="utf-8")
    with pytest.raises(matrix.NativeParityMatrixError, match="missing Python modules"):
        matrix.resolve_offline_snapshot(
            [tmp_path], model_id="owner/remote", revision=revision
        )


def test_build_dry_run_plan_preserves_both_tracks_and_requires_exact_sources(tmp_path):
    registry = _chronos_registry()
    dossier = registry["models"]["chronos_v1"]
    cache = tmp_path / "hub"
    snapshot = _snapshot(cache, dossier["model_id"], dossier["model_revision"])
    source = tmp_path / "chronos"
    source.mkdir()
    runner = tmp_path / "worker.py"
    runner.write_text("# worker", encoding="utf-8")
    entries = matrix.build_matrix_plan(
        registry=registry,
        runtime_pythons={"common": sys.executable},
        source_roots={"chronos_v1": source},
        hf_cache_roots=[cache], output_directory=tmp_path / "out",
        runner=runner, runner_source=tmp_path, validate_environments=False,
    )
    assert [entry.key for entry in entries] == [
        ("chronos_v1", "F"), ("chronos_v1", "R")
    ]
    assert all(entry.model == snapshot.resolve() for entry in entries)
    dry_run = matrix.plan_record(entries)
    assert dry_run["mode"] == "dry_run_no_models_loaded_no_bundles_written"
    assert dry_run["coverage_count"] == 2
    with pytest.raises(matrix.NativeParityMatrixError, match="source-root mapping"):
        matrix.build_matrix_plan(
            registry=registry, runtime_pythons={"common": sys.executable},
            source_roots={}, hf_cache_roots=[cache],
            output_directory=tmp_path / "out", runner=runner,
            runner_source=tmp_path,
            validate_environments=False,
        )


def test_worker_command_binds_all_paths_and_track(tmp_path):
    entry = matrix.MatrixEntry(
        arm_key="timesfm25", track="F", profile="timesfm",
        python=Path(sys.executable), source=tmp_path / "source",
        model=tmp_path / "model", tokenizer=None,
        extra_artifacts=(("reference_model", tmp_path / "reference"),),
        runner_source=tmp_path,
        bundle=tmp_path / "bundle",
    )
    command = matrix.worker_command(
        entry, runner=tmp_path / "worker.py", device="cpu",
        batch_size=4, samples=7, seed=11,
    )
    assert command[:3] == [
        str(Path(sys.executable)), "-I", str((tmp_path / "worker.py").resolve())
    ]
    assert command[command.index("--track") + 1] == "F"
    assert command[command.index("--reference-model-snapshot") + 1] == str(
        tmp_path / "reference"
    )


def test_execute_is_sequential_resumes_only_verified_and_aggregates_complete(
    tmp_path, monkeypatch
):
    runner = tmp_path / "worker.py"
    runner.write_text("# worker", encoding="utf-8")
    entries = []
    for track in ("F", "R"):
        bundle = tmp_path / f"chronos_v1__{track}"
        entries.append(matrix.MatrixEntry(
            arm_key="chronos_v1", track=track, profile="common",
            python=Path(sys.executable), source=tmp_path / "source",
            model=tmp_path / "model", tokenizer=None, extra_artifacts=(),
            runner_source=tmp_path,
            bundle=bundle,
        ))
    entries[0].bundle.mkdir()
    (entries[0].bundle / "bundle_manifest.json").write_text("{}", encoding="utf-8")
    entries[1].bundle.mkdir()
    (entries[1].bundle / "stderr.log").write_text("interrupted", encoding="utf-8")
    calls = []
    expected_env = {
        "CUDA_VISIBLE_DEVICES": "", **matrix.OFFLINE_ENVIRONMENT,
    }

    def fake_verify(path, registry_path=None):
        path = Path(path)
        track = "F" if path.name.endswith("__F") else "R"
        entry = next(item for item in entries if item.track == track)
        calls.append(("verify", track))
        return {
            "arm_key": "chronos_v1", "track": track,
            "command": {"argv": matrix.worker_command(
                entry, runner=runner, device="cpu", batch_size=4,
                samples=20, seed=20260717,
            )},
            "declared_environment": expected_env,
            "bound_artifacts": {
                name: {"path": str(path.resolve())}
                for name, path in entry.artifacts.items()
            },
        }, {}

    def fake_run(**kwargs):
        calls.append(("run", kwargs["track"]))
        Path(kwargs["output_directory"]).mkdir(parents=True)
        (Path(kwargs["output_directory"]) / "bundle_manifest.json").write_text("{}")

    def fake_aggregate(paths, **kwargs):
        calls.append(("aggregate", tuple(Path(path).name for path in paths)))
        assert kwargs["require_all_current"] is True
        return {"complete": True}

    monkeypatch.setattr(matrix, "verify_parity_bundle", fake_verify)
    monkeypatch.setattr(matrix, "run_parity_bundle", fake_run)
    monkeypatch.setattr(matrix, "aggregate_parity_bundles", fake_aggregate)
    monkeypatch.setattr(matrix, "load_registry", lambda **kwargs: _chronos_registry())
    result = matrix.execute_matrix(
        entries, runner=runner, aggregate_output=tmp_path / "aggregate.json",
        environment={"CUDA_VISIBLE_DEVICES": ""}, device="cpu",
    )
    assert result == {"complete": True}
    assert ("run", "F") not in calls
    assert ("run", "R") in calls
    assert (tmp_path / "chronos_v1__R.incomplete-001/stderr.log").is_file()
    assert calls[-1] == (
        "aggregate", ("chronos_v1__F", "chronos_v1__R")
    )


def test_execute_rejects_network_enabled_override(tmp_path, monkeypatch):
    monkeypatch.setattr(matrix, "admitted_native_pairs", lambda registry: ())
    with pytest.raises(matrix.NativeParityMatrixError, match="may not be overridden"):
        matrix.execute_matrix(
            [], runner=tmp_path / "worker.py",
            aggregate_output=tmp_path / "aggregate.json",
            environment={"HF_HUB_OFFLINE": "0"},
        )


def test_execute_rejects_partial_and_duplicate_coverage_before_loading(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(matrix, "load_registry", lambda **kwargs: _chronos_registry())
    entry = matrix.MatrixEntry(
        arm_key="chronos_v1", track="F", profile="common",
        python=Path(sys.executable), source=tmp_path, model=tmp_path,
        tokenizer=None, extra_artifacts=(), runner_source=tmp_path,
        bundle=tmp_path / "F",
    )
    with pytest.raises(matrix.NativeParityMatrixError, match="missing"):
        matrix.execute_matrix(
            [entry], runner=tmp_path / "worker.py",
            aggregate_output=tmp_path / "aggregate.json",
        )
    duplicate = matrix.MatrixEntry(**{**entry.__dict__, "bundle": tmp_path / "F2"})
    with pytest.raises(matrix.NativeParityMatrixError, match="duplicates"):
        matrix.execute_matrix(
            [entry, duplicate], runner=tmp_path / "worker.py",
            aggregate_output=tmp_path / "aggregate.json",
        )


def test_resume_rejects_execution_identity_drift(tmp_path, monkeypatch):
    registry = _chronos_registry()
    entries = []
    for track in ("F", "R"):
        bundle = tmp_path / track
        bundle.mkdir()
        (bundle / "bundle_manifest.json").write_text("{}")
        entries.append(matrix.MatrixEntry(
            arm_key="chronos_v1", track=track, profile="common",
            python=Path(sys.executable), source=tmp_path, model=tmp_path,
            tokenizer=None, extra_artifacts=(), runner_source=tmp_path,
            bundle=bundle,
        ))
    monkeypatch.setattr(matrix, "load_registry", lambda **kwargs: registry)
    monkeypatch.setattr(matrix, "verify_parity_bundle", lambda *args, **kwargs: ({
        "arm_key": "chronos_v1", "track": Path(args[0]).name,
        "command": {"argv": ["stale"]},
        "declared_environment": matrix.OFFLINE_ENVIRONMENT,
        "bound_artifacts": {},
    }, {}))
    with pytest.raises(matrix.NativeParityMatrixError, match="command/settings drift"):
        matrix.execute_matrix(
            entries, runner=tmp_path / "worker.py",
            aggregate_output=tmp_path / "aggregate.json", device="cpu",
        )


def test_config_expands_paths_without_committing_host_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("PARITY_HOME", str(tmp_path))
    config = tmp_path / "matrix.json"
    config.write_text(json.dumps({
        "schema_version": matrix.MATRIX_CONFIG_SCHEMA,
        "runtime_profiles": {"common": "${PARITY_HOME}/env/bin/python"},
        "source_roots": {"chronos_v1": "sources/chronos"},
        "hf_cache_roots": ["${PARITY_HOME}/hf"],
    }), encoding="utf-8")
    value, base = matrix.load_matrix_config(config)
    assert value["runtime_profiles"]["common"].startswith("${")
    assert base == tmp_path


def test_distribution_record_binds_installed_transitive_source(tmp_path):
    package = tmp_path / "demo/module.py"
    package.parent.mkdir()
    package.write_bytes(b"VALUE = 1\n")
    dist = tmp_path / "demo-1.0.dist-info"
    dist.mkdir()
    digest = base64.urlsafe_b64encode(hashlib.sha256(package.read_bytes()).digest()).decode().rstrip("=")
    (dist / "RECORD").write_text(
        f"demo/module.py,sha256={digest},{package.stat().st_size}\n"
        "demo-1.0.dist-info/RECORD,,\n",
        encoding="utf-8",
    )
    assert validate_distribution_record(dist) == dist.resolve()
    package.write_bytes(b"VALUE = 2\n")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        validate_distribution_record(dist)


def test_worker_and_matrix_are_packaged_console_surfaces():
    from setuptools import find_packages

    assert "scripts" in find_packages()
    setup_text = Path("setup.py").read_text(encoding="utf-8")
    assert (
        "ffm-native-parity-matrix=futures_foundation.finetune.native_parity_matrix_cli:main"
        in setup_text
    )
    from futures_foundation.finetune import native_parity_matrix_cli
    assert callable(native_parity_matrix_cli.main)


def test_matrix_package_module_is_directly_executable():
    completed = subprocess.run(
        [sys.executable, "-m", "futures_foundation.finetune.native_parity_matrix_cli", "--help"],
        check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--runtime-profile" in completed.stdout
