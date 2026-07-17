import json
from pathlib import Path

import numpy as np
import pytest

from futures_foundation.finetune.downstream_contexts import (
    build_downstream_contexts,
    load_downstream_contexts,
    save_downstream_contexts,
)


def _sha(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_context_artifact_reconstructs_exact_rows_and_binds_sample(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    n = 20
    close = 100 + np.arange(n, dtype=np.float32)
    ohlcv = np.column_stack((close, close + 1, close - 1, close, np.ones(n))).astype(np.float32)
    timestamp = np.arange(n, dtype=np.int64) * 60_000_000_000
    contract = np.full(n, "ESZ4")
    files = {}
    for key, value in (("ohlcv", ohlcv), ("timestamps", timestamp), ("contract_id", contract)):
        path = cache_dir / f"ES_1min.{key}.npy"
        np.save(path, value)
        files[key] = {"path": path.name, "sha256": _sha(path), "bytes": path.stat().st_size}
    cache = cache_dir / "TOURNAMENT_CACHE.json"
    cache.write_text(json.dumps({
        "schema_version": "ffm_foundation_tournament_cache_v1",
        "interval": {"contains_oos": False},
        "entries": {"ES@1min": {"files": files}},
    }))
    sample = {
        "stream_id": np.array(["ES@1min", "ES@1min"]),
        "context_start_source_idx": np.array([2, 10]),
        "decision_source_idx": np.array([5, 13]),
        "decision_time_ns": timestamp[[5, 13]],
    }
    sample_manifest = {
        "status": "complete", "oos_read": False,
        "artifact": {"path": "sample.npz", "sha256": "sample-sha"},
        "content_fingerprint": "sample-fingerprint",
    }
    arrays, metadata = build_downstream_contexts(
        sample, sample_manifest, cache, context_bars=4,
    )
    assert arrays["context"].shape == (2, 4, 5)
    np.testing.assert_array_equal(arrays["context"][0], ohlcv[2:6])
    np.testing.assert_array_equal(arrays["context_time_ns"][1], timestamp[10:14])
    output = tmp_path / "contexts.npz"
    manifest = save_downstream_contexts(output, arrays, metadata)
    loaded, loaded_manifest = load_downstream_contexts(
        output, sample_manifest=sample_manifest,
    )
    assert loaded_manifest["content_fingerprint"] == manifest["content_fingerprint"]
    np.testing.assert_array_equal(loaded["context"], arrays["context"])
    with output.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_downstream_contexts(output)


def test_context_builder_rejects_roll_crossing(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    ohlcv = np.tile(np.array([100, 101, 99, 100, 1], np.float32), (5, 1))
    timestamp = np.arange(5, dtype=np.int64)
    contract = np.array(["A", "A", "B", "B", "B"])
    files = {}
    for key, value in (("ohlcv", ohlcv), ("timestamps", timestamp), ("contract_id", contract)):
        path = cache_dir / f"x.{key}.npy"
        np.save(path, value)
        files[key] = {"path": path.name, "sha256": _sha(path)}
    cache = cache_dir / "manifest.json"
    cache.write_text(json.dumps({
        "schema_version": "ffm_foundation_tournament_cache_v1",
        "interval": {"contains_oos": False},
        "entries": {"ES@1min": {"files": files}},
    }))
    sample = {
        "stream_id": np.array(["ES@1min"]),
        "context_start_source_idx": np.array([0]), "decision_source_idx": np.array([3]),
        "decision_time_ns": timestamp[[3]],
    }
    manifest = {
        "status": "complete", "oos_read": False,
        "artifact": {"path": "x", "sha256": "x"}, "content_fingerprint": "x",
    }
    with pytest.raises(ValueError, match="crosses a contract roll"):
        build_downstream_contexts(sample, manifest, cache, context_bars=4)
