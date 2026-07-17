import subprocess
import sys
from pathlib import Path

import numpy as np

from futures_foundation.finetune.foundation_eval import (
    load_window_artifact, persistence_prediction, save_prediction_artifact,
    save_window_artifact, score_prediction_artifact,
)
from futures_foundation.finetune.tournament import OOS_START, VALIDATION_START
from scripts.predict_foundation_forecasts import _causal_suffix, _windows_with_causal_suffix


def test_prediction_script_bootstraps_repo_root_for_direct_execution():
    script = Path(__file__).resolve().parents[1] / "scripts" / "predict_foundation_forecasts.py"
    code = (
        "import runpy,sys; "
        f"d=runpy.run_path({str(script)!r}); "
        "assert str(d['ROOT']) in sys.path"
    )
    subprocess.run([sys.executable, "-c", code], cwd="/tmp", check=True)


def _windows():
    rng = np.random.default_rng(4)
    context = rng.uniform(10, 11, (3, 8, 5)).astype(np.float32)
    future = rng.uniform(10, 11, (3, 2, 5)).astype(np.float32)
    # Make OHLC internally valid for the metric path.
    for values in (context, future):
        values[:, :, 1] = np.maximum.reduce([values[:, :, 0], values[:, :, 1], values[:, :, 3]])
        values[:, :, 2] = np.minimum.reduce([values[:, :, 0], values[:, :, 2], values[:, :, 3]])
    return {
        "context": context, "future": future,
        "context_time_ns": np.arange(24, dtype=np.int64).reshape(3, 8),
        "future_time_ns": (100 + np.arange(6, dtype=np.int64)).reshape(3, 2),
        "ticker": np.array(["ES", "NQ", "GC"]),
        "timeframe": np.array(["1min", "5min", "60min"]),
        "source_start": np.arange(3, dtype=np.int64),
        "counts": {"ES@1min": 1, "NQ@5min": 1, "GC@60min": 1},
        "eval_start": VALIDATION_START, "eval_end": OOS_START,
    }


def test_shared_window_and_prediction_round_trip(tmp_path):
    window_path = tmp_path / "windows.npz"
    manifest = save_window_artifact(window_path, _windows(), config={"seed": 1})
    windows, loaded_manifest = load_window_artifact(window_path)
    assert loaded_manifest["window_fingerprint"] == manifest["window_fingerprint"]
    prediction_path = tmp_path / "prediction.npz"
    prediction = persistence_prediction(windows)
    save_prediction_artifact(
        prediction_path, prediction, windows_manifest=loaded_manifest, arm="persistence_copy",
    )
    prediction_manifest, metrics = score_prediction_artifact(
        windows, loaded_manifest, prediction_path,
    )
    assert prediction_manifest["arm"] == "persistence_copy"
    assert metrics["overall"]["path_skill_vs_persistence"] == 0.0


def test_causal_suffix_uses_only_the_latest_supported_context():
    context = np.arange(2 * 8 * 5, dtype=np.float32).reshape(2, 8, 5)
    suffix = _causal_suffix(context, 4, "test-arm")
    np.testing.assert_array_equal(suffix, context[:, 4:])


def test_causal_suffix_rejects_insufficient_context():
    context = np.zeros((2, 3, 5), dtype=np.float32)
    with np.testing.assert_raises_regex(ValueError, "requires 4"):
        _causal_suffix(context, 4, "test-arm")


def test_window_suffix_keeps_future_and_aligns_context_times():
    windows = _windows()
    native = _windows_with_causal_suffix(windows, 4, "test-arm")
    np.testing.assert_array_equal(native["context"], windows["context"][:, -4:])
    np.testing.assert_array_equal(
        native["context_time_ns"], windows["context_time_ns"][:, -4:]
    )
    assert native["future"] is windows["future"]
    assert native["context_length"] == 4
