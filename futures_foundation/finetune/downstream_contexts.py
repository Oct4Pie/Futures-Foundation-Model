"""Shared raw OHLCV contexts for cross-family downstream representation scoring."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from futures_foundation.finetune.downstream_sample import load_balanced_sample
from futures_foundation.finetune.tournament_data import (
    CACHE_MANIFEST,
    load_cache_entry,
    load_cache_manifest,
)
from futures_foundation.session_gap import (
    load_session_gap_capability,
    verified_session_edge_mask,
)


SCHEMA_VERSION = "ffm_downstream_contexts_v2"


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
    cache_manifest_sha256: str,
    context_bars: int = 256,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Reconstruct exact source contexts for every row in a verified balanced sample."""
    cache_path = Path(cache_manifest_path).resolve()
    if cache_path.name != CACHE_MANIFEST:
        raise ValueError(f"cache manifest must be named {CACHE_MANIFEST}")
    cache = load_cache_manifest(
        cache_path.parent, expected_manifest_sha256=cache_manifest_sha256,
    )
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
    context_contract_id = np.empty((rows, context_bars), dtype="U64")
    source_files: dict[str, dict[str, object]] = {}
    session_manifests: dict[str, object] = {}
    cache_dir = cache_path.parent
    sample_metadata = sample_manifest.get("metadata")
    source_shards = (
        sample_metadata.get("source_shards")
        if isinstance(sample_metadata, dict) else None
    )
    requested_streams = {str(value) for value in np.unique(sample["stream_id"])}
    if not isinstance(source_shards, dict) or set(source_shards) != requested_streams:
        raise ValueError("sample manifest lacks exact source-shard authority closure")

    for stream_id in sorted(str(value) for value in np.unique(sample["stream_id"])):
        selected = np.flatnonzero(sample["stream_id"] == stream_id)
        ticker, timeframe = stream_id.split("@", 1)
        stream, verified = load_cache_entry(cache_dir, cache, ticker, timeframe)
        loaded = {
            "ohlcv": stream["ohlcv"],
            "timestamps": np.asarray(stream["ts"]).astype("datetime64[ns]").astype(np.int64),
            "contract_id": stream["contract_id"],
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
        source_info = source_shards[stream_id]
        if not isinstance(source_info, dict):
            raise ValueError(f"{stream_id} source-shard authority is malformed")
        session_manifest = source_info.get("session_gap_capability")
        if session_manifest is None:
            expected_ns = int(pd.Timedelta(timeframe).value)
            edge_valid = np.diff(np.asarray(loaded["timestamps"], np.int64)) == expected_ns
        else:
            capability = load_session_gap_capability(session_manifest)
            if capability.root != ticker:
                raise ValueError(f"{stream_id} session-gap root identity mismatch")
            edge_valid = verified_session_edge_mask(
                pd.to_datetime(loaded["timestamps"], utc=True),
                expected_delta=pd.Timedelta(timeframe),
                capability=capability,
            )
        bad_prefix = np.r_[0, np.cumsum(~edge_valid, dtype=np.int64)]
        if np.any(bad_prefix[decisions] - bad_prefix[starts] != 0):
            raise ValueError(f"{stream_id} context crosses an unexplained timestamp gap")
        if np.any(contract != contract[:, :1]):
            raise ValueError(f"{stream_id} context crosses a contract roll")
        if not np.array_equal(timestamp[:, -1], sample["decision_time_ns"][selected]):
            raise ValueError(f"{stream_id} decision timestamp/source identity mismatch")
        contexts[selected] = value
        context_time_ns[selected] = timestamp
        context_contract_id[selected] = contract
        source_files[stream_id] = verified
        session_manifests[stream_id] = session_manifest

    arrays = {
        "context": contexts,
        "context_time_ns": context_time_ns,
        "context_contract_id": context_contract_id,
        "stream_id": np.asarray(sample["stream_id"]).astype(str),
        "decision_time_ns": np.asarray(sample["decision_time_ns"], np.int64),
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
            "path": str(cache_path), "sha256": cache_manifest_sha256,
            "schema_version": cache["schema_version"],
        },
        "source_files": source_files,
        "session_gap_capabilities": session_manifests,
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
    rows = int(shape["rows"])
    context_bars = int(shape["context_bars"])
    if set(arrays) != {
        "context", "context_time_ns", "context_contract_id", "stream_id",
        "decision_time_ns", "row_index",
    }:
        raise ValueError("downstream context array contract mismatch")
    if (
        arrays["context"].shape != (rows, context_bars, shape["channels"])
        or arrays["context_time_ns"].shape != (rows, context_bars)
        or arrays["context_contract_id"].shape != (rows, context_bars)
        or arrays["stream_id"].shape != (rows,)
        or arrays["decision_time_ns"].shape != (rows,)
        or arrays["row_index"].shape != (rows,)
    ):
        raise ValueError("downstream context shape contract mismatch")
    if not np.isfinite(arrays["context"]).all():
        raise ValueError("downstream contexts contain non-finite OHLCV")
    o, h, l, c, v = arrays["context"].transpose(2, 0, 1)
    if np.any((h < np.maximum(o, c)) | (l > np.minimum(o, c)) | (h < l) | (v < 0)):
        raise ValueError("downstream contexts contain invalid OHLCV geometry")
    if np.any(np.diff(arrays["context_time_ns"], axis=1) <= 0):
        raise ValueError("downstream context timestamps are not increasing")
    contract = arrays["context_contract_id"].astype(str)
    if np.any(np.char.str_len(np.char.strip(contract)) == 0) or np.any(contract != contract[:, :1]):
        raise ValueError("downstream context crosses a contract roll or has blank identity")
    if not np.array_equal(arrays["context_time_ns"][:, -1], arrays["decision_time_ns"]):
        raise ValueError("downstream context decision timestamp identity mismatch")
    if not np.array_equal(arrays["row_index"], np.arange(rows, dtype=np.int32)):
        raise ValueError("downstream context row identity mismatch")
    session_manifests = manifest["metadata"].get("session_gap_capabilities")
    stream_ids = arrays["stream_id"].astype(str)
    if not isinstance(session_manifests, dict) or set(session_manifests) != set(stream_ids):
        raise ValueError("downstream context session authority closure mismatch")
    for stream_id in sorted(set(stream_ids)):
        selected = np.flatnonzero(stream_ids == stream_id)
        ticker, timeframe = stream_id.split("@", 1)
        session_manifest = session_manifests[stream_id]
        if session_manifest is None:
            valid = np.diff(arrays["context_time_ns"][selected], axis=1) == int(
                pd.Timedelta(timeframe).value
            )
        else:
            capability = load_session_gap_capability(session_manifest)
            if capability.root != ticker:
                raise ValueError("downstream context session root identity mismatch")
            valid = np.vstack([
                verified_session_edge_mask(
                    pd.to_datetime(row, utc=True),
                    expected_delta=pd.Timedelta(timeframe),
                    capability=capability,
                )
                for row in arrays["context_time_ns"][selected]
            ])
        if not bool(np.all(valid)):
            raise ValueError("downstream context crosses an unexplained timestamp gap")
    for value in arrays.values():
        value.setflags(write=False)
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
    if not np.array_equal(contexts["stream_id"].astype(str), sample["stream_id"].astype(str)):
        raise ValueError("downstream context stream identity mismatch")
    if not np.array_equal(contexts["decision_time_ns"], sample["decision_time_ns"]):
        raise ValueError("downstream context decision identity mismatch")
    return sample, sample_manifest, contexts, context_manifest


__all__ = [
    "SCHEMA_VERSION", "build_downstream_contexts", "save_downstream_contexts",
    "load_downstream_contexts", "load_sample_and_contexts",
]
