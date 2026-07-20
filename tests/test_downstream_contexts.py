import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from futures_foundation.finetune.downstream_contexts import (
    build_downstream_contexts,
    load_downstream_contexts,
    save_downstream_contexts,
)
from futures_foundation.finetune.tournament_cache_authority import (
    SOURCE_AUTHORITY_SCHEMA_VERSION,
    canonical_authority_document,
)
from futures_foundation.finetune.tournament_data import (
    CACHE_MANIFEST,
    build_cache,
    cache_manifest_sha256,
)


def _sha(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_cache(
    tmp_path: Path,
    ohlcv: np.ndarray,
    timestamp: np.ndarray,
    contract: np.ndarray,
) -> tuple[Path, str]:
    source_dir = tmp_path / "source"
    cache_dir = tmp_path / "cache"
    source_dir.mkdir()
    csv = source_dir / "ES_1min.csv"
    pd.DataFrame({
        "datetime": pd.to_datetime(timestamp, utc=True),
        "open": ohlcv[:, 0], "high": ohlcv[:, 1], "low": ohlcv[:, 2],
        "close": ohlcv[:, 3], "volume": ohlcv[:, 4], "contract_id": contract,
    }).to_csv(csv, index=False)
    source_manifest = source_dir / "MANIFEST.json"
    source_manifest.write_text(json.dumps({
        "schema_version": "ffm_ssl_corpus_v1",
        "created_utc": "2026-07-18T00:00:00+00:00",
        "purpose": "self-supervised OHLCV only; no labels or outcomes read",
        "source_root": str(source_dir.resolve()),
        "source_snapshot_sha256": _sha(csv),
        "roots": ["ES"],
        "timeframes_minutes": [1],
        "resample": {
            "closed": "left", "label": "left", "origin": "epoch",
            "forward_fill": False, "within_contract_only": True,
        },
        "roots_report": {},
        "outputs": {"ES_1min": {
            "path": str(csv.resolve()), "bytes": csv.stat().st_size,
            "sha256": _sha(csv), "rows": len(ohlcv),
        }},
    }, indent=2) + "\n")
    authority = tmp_path / "source-authority.json"
    authority.write_bytes(canonical_authority_document({
        "schema_version": SOURCE_AUTHORITY_SCHEMA_VERSION,
        "authority_id": "downstream-context-test",
        "purpose": "tournament_cache_source_admission",
        "source_manifest": {
            "path": str(source_manifest.resolve()), "sha256": _sha(source_manifest),
            "bytes": source_manifest.stat().st_size,
            "schema_version": "ffm_ssl_corpus_v1",
        },
        "admitted_streams": ["ES@1min"],
        "cache_construction_admitted": True,
        "training_admitted": False,
    }))
    build_cache(
        source_dir, cache_dir, ("ES",), ("1min",),
        source_authority_path=authority,
        source_authority_sha256=_sha(authority),
        verbose=False,
    )
    return cache_dir / CACHE_MANIFEST, cache_manifest_sha256(cache_dir)


def test_context_artifact_reconstructs_exact_rows_and_binds_sample(tmp_path):
    n = 20
    close = 100 + np.arange(n, dtype=np.float32)
    ohlcv = np.column_stack((close, close + 1, close - 1, close, np.ones(n))).astype(np.float32)
    timestamp = pd.date_range(
        "2020-01-01", periods=n, freq="1min", tz="UTC",
    ).asi8
    contract = np.full(n, "ESZ4")
    cache, cache_sha = _build_cache(tmp_path, ohlcv, timestamp, contract)
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
        "metadata": {
            "source_shards": {
                "ES@1min": {"session_gap_capability": None},
            },
        },
    }
    arrays, metadata = build_downstream_contexts(
        sample, sample_manifest, cache,
        cache_manifest_sha256=cache_sha, context_bars=4,
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
    ohlcv = np.tile(np.array([100, 101, 99, 100, 1], np.float32), (5, 1))
    timestamp = pd.date_range(
        "2020-01-01", periods=5, freq="1min", tz="UTC",
    ).asi8
    contract = np.array(["A", "A", "B", "B", "B"])
    cache, cache_sha = _build_cache(tmp_path, ohlcv, timestamp, contract)
    sample = {
        "stream_id": np.array(["ES@1min"]),
        "context_start_source_idx": np.array([0]), "decision_source_idx": np.array([3]),
        "decision_time_ns": timestamp[[3]],
    }
    manifest = {
        "status": "complete", "oos_read": False,
        "artifact": {"path": "x", "sha256": "x"}, "content_fingerprint": "x",
        "metadata": {
            "source_shards": {
                "ES@1min": {"session_gap_capability": None},
            },
        },
    }
    with pytest.raises(ValueError, match="crosses a contract roll"):
        build_downstream_contexts(
            sample, manifest, cache,
            cache_manifest_sha256=cache_sha, context_bars=4,
        )
