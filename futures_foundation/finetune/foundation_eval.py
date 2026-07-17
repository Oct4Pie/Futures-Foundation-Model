"""Immutable shared-window artifacts for foundation-model forecast comparisons."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .kronos_eval import evaluate_predictions, window_fingerprint
from .tournament import OOS_START, VALIDATION_START


WINDOW_KEYS = (
    "context", "future", "context_time_ns", "future_time_ns", "ticker", "timeframe",
    "source_start",
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _calendar_date(value):
    """Canonicalize the ISO timestamps emitted by the CSV window builder."""
    return str(value)[:10]


def save_window_artifact(path, windows, *, config):
    """Persist exact validation inputs/labels and bind them to a content fingerprint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    missing = [key for key in WINDOW_KEYS if key not in windows]
    if missing:
        raise ValueError(f"window artifact is missing keys: {missing}")
    if (_calendar_date(windows.get("eval_start")) != VALIDATION_START
            or _calendar_date(windows.get("eval_end")) != OOS_START):
        raise ValueError("shared evaluation artifact must contain validation rows only")
    fingerprint = window_fingerprint(windows)
    temporary = Path(str(path) + ".tmp.npz")
    np.savez_compressed(temporary, **{key: windows[key] for key in WINDOW_KEYS})
    os.replace(temporary, path)
    manifest = {
        "schema_version": "ffm_shared_validation_windows_v1",
        "window_fingerprint": fingerprint,
        "artifact": {"path": str(path.resolve()), "sha256": _sha256(path)},
        "split": {
            "validation_start": VALIDATION_START, "oos_start": OOS_START,
            "oos_read": False,
        },
        "shape": {
            "windows": int(len(windows["context"])),
            "context": int(windows["context"].shape[1]),
            "horizon": int(windows["future"].shape[1]),
            "channels": int(windows["future"].shape[2]),
        },
        "counts": dict(windows.get("counts", {})),
        "config": dict(config),
    }
    _atomic_json(str(path) + ".manifest.json", manifest)
    return manifest


def load_window_artifact(path):
    path = Path(path)
    manifest_path = Path(str(path) + ".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != "ffm_shared_validation_windows_v1":
        raise ValueError("unsupported shared-window artifact schema")
    if manifest.get("split") != {
        "validation_start": VALIDATION_START, "oos_start": OOS_START, "oos_read": False,
    }:
        raise ValueError("shared-window split guard failed")
    if _sha256(path) != manifest["artifact"]["sha256"]:
        raise ValueError("shared-window artifact hash mismatch")
    with np.load(path, allow_pickle=False) as saved:
        windows = {key: saved[key] for key in WINDOW_KEYS}
    windows.update({
        "eval_start": VALIDATION_START, "eval_end": OOS_START,
        "context_length": int(windows["context"].shape[1]),
        "horizon": int(windows["future"].shape[1]),
    })
    if window_fingerprint(windows) != manifest["window_fingerprint"]:
        raise ValueError("shared-window content fingerprint mismatch")
    return windows, manifest


def save_prediction_artifact(path, prediction, *, windows_manifest, arm, metadata=None):
    """Persist point forecasts only when they match the locked validation artifact."""
    prediction = np.asarray(prediction, np.float32)
    expected = windows_manifest["shape"]
    shape = (expected["windows"], expected["horizon"], expected["channels"])
    if prediction.shape != shape:
        raise ValueError(f"prediction shape mismatch: expected {shape}, got {prediction.shape}")
    if not np.isfinite(prediction).all():
        raise ValueError("predictions contain non-finite values")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp.npz")
    np.savez_compressed(
        temporary, prediction=prediction,
        window_fingerprint=np.array(windows_manifest["window_fingerprint"]),
    )
    os.replace(temporary, path)
    manifest = {
        "schema_version": "ffm_shared_forecast_predictions_v1",
        "arm": str(arm),
        "window_fingerprint": windows_manifest["window_fingerprint"],
        "artifact": {"path": str(path.resolve()), "sha256": _sha256(path)},
        "metadata": dict(metadata or {}),
    }
    _atomic_json(str(path) + ".manifest.json", manifest)
    return manifest


def score_prediction_artifact(windows, windows_manifest, path):
    path = Path(path)
    manifest = json.loads(Path(str(path) + ".manifest.json").read_text())
    if manifest.get("schema_version") != "ffm_shared_forecast_predictions_v1":
        raise ValueError("unsupported prediction artifact schema")
    fingerprint = windows_manifest["window_fingerprint"]
    if manifest.get("window_fingerprint") != fingerprint:
        raise ValueError("prediction/window fingerprint mismatch")
    if _sha256(path) != manifest["artifact"]["sha256"]:
        raise ValueError("prediction artifact hash mismatch")
    with np.load(path, allow_pickle=False) as saved:
        if str(saved["window_fingerprint"].item()) != fingerprint:
            raise ValueError("embedded prediction/window fingerprint mismatch")
        prediction = saved["prediction"]
    return manifest, evaluate_predictions(windows, prediction)


def persistence_prediction(windows):
    return np.repeat(windows["context"][:, -1:, :], windows["future"].shape[1], axis=1)
