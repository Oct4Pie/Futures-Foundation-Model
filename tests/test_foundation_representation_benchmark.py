import json
from pathlib import Path

import numpy as np
import pytest

from scripts import benchmark_foundation_representations as benchmark


def test_stage_contract_is_explicit():
    assert benchmark.STAGE_NAMES == {
        "stage1": "stage1_reconstruction",
        "stage2": "stage2_contrastive",
        "stage3": "stage3_forecast",
    }
    assert "toto2_22m" in benchmark.EXTRACTORS
    assert "sundial_base" not in benchmark.EXTRACTORS


def test_load_windows_rejects_oos_and_hash_drift(tmp_path):
    path = tmp_path / "windows.npz"
    np.savez_compressed(
        path, context=np.zeros((2, 256, 5), np.float32),
        future=np.zeros((2, 16, 5), np.float32),
        context_time_ns=np.zeros((2, 256), np.int64),
        future_time_ns=np.zeros((2, 16), np.int64),
        ticker=np.array(["ES", "ES"]), timeframe=np.array(["1min", "1min"]),
        source_start=np.array([0, 272]),
    )
    manifest = {
        "schema_version": benchmark.SCHEMA,
        "artifact": {"sha256": benchmark._sha256(path)},
        "window_fingerprint": "test",
        "split": {"oos_read": True},
    }
    Path(str(path) + ".manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="validation-only"):
        benchmark._load_windows(path)
    manifest["split"]["oos_read"] = False
    manifest["artifact"]["sha256"] = "bad"
    Path(str(path) + ".manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="hash mismatch"):
        benchmark._load_windows(path)


def test_save_embedding_excludes_nonfinite(tmp_path):
    args = type("Args", (), {"output_dir": str(tmp_path)})()
    manifest = {"window_fingerprint": "w", "artifact": {"sha256": "x"}}
    with pytest.raises(ValueError, match="invalid embeddings"):
        benchmark._save_embedding(
            args, "arm", "stage1", None, np.array([[np.nan]], np.float32), {}, manifest,
        )


def test_save_embedding_rejects_unbound_row_identity(tmp_path):
    args = type("Args", (), {
        "output_dir": str(tmp_path), "row_index": np.array([1]),
        "row_selection_manifest": None, "context_manifest": None,
    })()
    manifest = {"window_fingerprint": "w", "artifact": {"sha256": "x"}}
    with pytest.raises(ValueError, match="require selection and context"):
        benchmark._save_embedding(
            args, "arm", "stage1", None, np.ones((1, 2), np.float32), {}, manifest,
        )


def test_mantis_diagnostic_checkpoint_fallback_is_explicit(tmp_path):
    args = type("Args", (), {"checkpoint_root": str(tmp_path)})()
    root = tmp_path / "mantis_v2"
    root.mkdir()
    diagnostic = root / "stage2_diagnostic.pt"
    diagnostic.write_bytes(b"checkpoint")
    assert benchmark._checkpoint_path(args, "mantis_v2", "stage2") == diagnostic

    canonical = root / "stage2.pt"
    canonical.write_bytes(b"canonical")
    assert benchmark._checkpoint_path(args, "mantis_v2", "stage2") == canonical


def test_markdown_renders_all_six_metrics():
    metrics = {
        target: {"mean": float(index)}
        for index, target in enumerate(
            ("vol", "trend_eff", "range_expand", "fwd_absmove", "direction", "fwd_dir")
        )
    }
    report = {
        "status": "complete", "oos_read": False,
        "windows": {"rows": 1, "context": 256, "horizon": 16},
        "results": {"m:stage1": {"arm": "m", "stage": "stage1",
                                   "checkpoint": "/x.pt", "metrics": metrics}},
        "coverage": {"m": {"stage1": "complete", "stage2": "missing",
                             "stage3": "missing"}},
    }
    rendered = benchmark._render_markdown(report)
    assert "fwd_absmove R²" in rendered
    assert "fwd_dir AUC" in rendered
    assert "| m | S1 | 0.0000 | 1.0000 | 2.0000 | 3.0000 | 4.0000 | 5.0000 |" in rendered


def test_report_source_declares_oos_not_read():
    source = Path(benchmark.__file__).read_text()
    assert '"oos_read": False' in source


def test_timesfm_extraction_promotes_training_bf16_model_to_float32():
    source = Path(benchmark.__file__).read_text()
    assert "model = _load_model(ns).float().eval()" in source
    assert '"context": CONTEXT, "precision": "float32"' in source


def test_coverage_status_does_not_hide_blockers():
    coverage = {
        arm: {stage: "complete" for stage in ("stage1", "stage2", "stage3")}
        for arm in benchmark.EXTRACTORS
    }
    coverage["sundial_base"] = {
        stage: "blocked_nonfinite_native_hidden_states"
        for stage in ("stage1", "stage2", "stage3")
    }
    assert benchmark._coverage_status(coverage, {"row": {}}) == "complete_with_declared_blockers"
