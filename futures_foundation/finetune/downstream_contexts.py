"""Shared raw OHLCV contexts for cross-family downstream representation scoring."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from futures_foundation.finetune.downstream_sample import load_balanced_sample


SCHEMA_VERSION = "ffm_downstream_contexts_v1"


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _fingerprint(arrays: dict[str, np.ndarray], metadata: dict[str, object]) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode())
    for key in sorted(arrays):
        value = np.ascontiguousarray(arrays[key])
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def build_downstream_contexts(
    sample: dict[str, np.ndarray],
    sample_manifest: dict[str, object],
    cache_manifest_path: str | Path,
    *,
    context_bars: int = 256,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Reconstruct exact source contexts for every row in a verified balanced sample."""
    cache_path = Path(cache_manifest_path).resolve()
    cache = json.loads(cache_path.read_text())
    if (
        cache.get("schema_version") != "ffm_foundation_tournament_cache_v1"
        or cache.get("interval", {}).get("contains_oos") is not False
    ):
        raise ValueError("downstream contexts require the development-only tournament cache")
    if sample_manifest.get("status") != "complete" or sample_manifest.get("oos_read") is not False:
        raise ValueError("downstream contexts require a completed development-only sample")
    context_bars = int(context_bars)
    if context_bars < 1:
        raise ValueError("context_bars must be positive")
    rows = len(sample["stream_id"])
    if rows < 1:
        raise ValueError("sample is empty")
    contexts = np.empty((rows, context_bars, 5), dtype=np.float32)
    context_time_ns = np.empty((rows, context_bars), dtype=np.int64)
    source_files: dict[str, dict[str, object]] = {}
    cache_dir = cache_path.parent

    for stream_id in sorted(str(value) for value in np.unique(sample["stream_id"])):
        selected = np.flatnonzero(sample["stream_id"] == stream_id)
        entry = cache.get("entries", {}).get(stream_id)
        if entry is None:
            raise KeyError(f"tournament cache lacks {stream_id}")
        loaded = {}
        verified = {}
        for key in ("ohlcv", "timestamps", "contract_id"):
            declared = entry["files"][key]
            path = cache_dir / declared["path"]
            actual = _sha256(path)
            if actual != declared["sha256"]:
                raise ValueError(f"source hash mismatch for {stream_id}/{key}")
            loaded[key] = np.load(path, mmap_mode="r", allow_pickle=False)
            verified[key] = {
                "path": str(path.resolve()), "sha256": actual,
                "bytes": int(path.stat().st_size),
            }
        starts = np.asarray(sample["context_start_source_idx"][selected], np.int64)
        decisions = np.asarray(sample["decision_source_idx"][selected], np.int64)
        if np.any(decisions - starts + 1 != context_bars):
            raise ValueError(f"{stream_id} sample does not contain {context_bars}-bar contexts")
        index = starts[:, None] + np.arange(context_bars, dtype=np.int64)[None, :]
        if index.min() < 0 or index.max() >= len(loaded["ohlcv"]):
            raise ValueError(f"{stream_id} source context index is out of range")
        value = np.asarray(loaded["ohlcv"][index], np.float32)
        timestamp = np.asarray(loaded["timestamps"][index], np.int64)
        contract = np.asarray(loaded["contract_id"][index]).astype(str)
        if value.shape != (len(selected), context_bars, 5) or not np.isfinite(value).all():
            raise ValueError(f"{stream_id} returned invalid OHLCV contexts")
        o, h, l, c, v = value.transpose(2, 0, 1)
        if np.any((h < np.maximum(o, c)) | (l > np.minimum(o, c)) | (h < l) | (v < 0)):
            raise ValueError(f"{stream_id} source has invalid OHLCV geometry")
        if np.any(np.diff(timestamp, axis=1) <= 0):
            raise ValueError(f"{stream_id} context timestamps are not increasing")
        if np.any(contract != contract[:, :1]):
            raise ValueError(f"{stream_id} context crosses a contract roll")
        if not np.array_equal(timestamp[:, -1], sample["decision_time_ns"][selected]):
            raise ValueError(f"{stream_id} decision timestamp/source identity mismatch")
        contexts[selected] = value
        context_time_ns[selected] = timestamp
        source_files[stream_id] = verified

    arrays = {
        "context": contexts,
        "context_time_ns": context_time_ns,
        "row_index": np.arange(rows, dtype=np.int32),
    }
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "oos_read": False,
        "shape": {"rows": rows, "context_bars": context_bars, "channels": 5},
        "channels": ["open", "high", "low", "close", "volume"],
        "sample": {
            "path": sample_manifest["artifact"]["path"],
            "sha256": sample_manifest["artifact"]["sha256"],
            "content_fingerprint": sample_manifest["content_fingerprint"],
        },
        "cache": {
            "path": str(cache_path), "sha256": _sha256(cache_path),
            "schema_version": cache["schema_version"],
        },
        "source_files": source_files,
    }
    return arrays, metadata


def save_downstream_contexts(
    path: str | Path,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, object],
) -> dict[str, object]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fingerprint = _fingerprint(arrays, metadata)
    temporary = Path(str(path) + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "oos_read": False,
        "content_fingerprint": fingerprint,
        "artifact": {
            "path": str(path.resolve()), "sha256": _sha256(path),
            "bytes": int(path.stat().st_size),
        },
        "metadata": metadata,
    }
    manifest_path = Path(str(path) + ".manifest.json")
    temporary_manifest = Path(str(manifest_path) + ".tmp")
    temporary_manifest.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    os.replace(temporary_manifest, manifest_path)
    return manifest


def load_downstream_contexts(
    path: str | Path,
    *,
    sample_manifest: dict[str, object] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    path = Path(path)
    manifest = json.loads(Path(str(path) + ".manifest.json").read_text())
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("status") != "complete"
        or manifest.get("oos_read") is not False
    ):
        raise ValueError("unsupported or incomplete downstream context artifact")
    if _sha256(path) != manifest.get("artifact", {}).get("sha256"):
        raise ValueError("downstream context artifact hash mismatch")
    with np.load(path, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    if _fingerprint(arrays, manifest["metadata"]) != manifest["content_fingerprint"]:
        raise ValueError("downstream context content fingerprint mismatch")
    if sample_manifest is not None:
        expected = sample_manifest["artifact"]["sha256"]
        if manifest["metadata"]["sample"]["sha256"] != expected:
            raise ValueError("downstream context/sample identity mismatch")
    shape = manifest["metadata"]["shape"]
    if arrays["context"].shape != (
        shape["rows"], shape["context_bars"], shape["channels"],
    ) or arrays["context_time_ns"].shape != (shape["rows"], shape["context_bars"]):
        raise ValueError("downstream context shape contract mismatch")
    return arrays, manifest


def load_sample_and_contexts(
    sample_path: str | Path,
    context_path: str | Path,
) -> tuple[dict[str, np.ndarray], dict[str, object], dict[str, np.ndarray], dict[str, object]]:
    sample, sample_manifest = load_balanced_sample(sample_path)
    contexts, context_manifest = load_downstream_contexts(
        context_path, sample_manifest=sample_manifest,
    )
    if not np.array_equal(contexts["row_index"], np.arange(len(sample["stream_id"]))):
        raise ValueError("downstream context row identity mismatch")
    return sample, sample_manifest, contexts, context_manifest


__all__ = [
    "SCHEMA_VERSION", "build_downstream_contexts", "save_downstream_contexts",
    "load_downstream_contexts", "load_sample_and_contexts",
]
