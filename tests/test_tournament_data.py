import json
import numpy as np

import pandas as pd

from futures_foundation.finetune.tournament_cache_authority import (
    SOURCE_AUTHORITY_SCHEMA_VERSION,
    canonical_authority_document,
)
from futures_foundation.finetune.tournament_data import (
    balanced_schedule, build_cache, cache_manifest_sha256, gather_contexts, gather_parent,
    gather_time_features, load_cache, schedule_fingerprint,
)


def _sha(path):
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_source_authority(source, *, ticker="ES", timeframe="1D"):
    csv = source / f"{ticker}_{timeframe}.csv"
    rows = sum(1 for _ in csv.open()) - 1
    output_id = f"{ticker}_{timeframe}"
    manifest = {
        "schema_version": "ffm_ssl_corpus_v1",
        "created_utc": "2026-07-18T00:00:00+00:00",
        "purpose": "self-supervised OHLCV only; no labels or outcomes read",
        "source_root": str(source.resolve()),
        "source_snapshot_sha256": _sha(csv),
        "roots": [ticker],
        "timeframes_minutes": [1440],
        "resample": {
            "closed": "left", "label": "left", "origin": "epoch",
            "forward_fill": False, "within_contract_only": True,
        },
        "roots_report": {},
        "outputs": {
            output_id: {
                "path": str(csv.resolve()),
                "bytes": csv.stat().st_size,
                "sha256": _sha(csv),
                "rows": rows,
            },
        },
    }
    manifest_path = source / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    authority = source.parent / f"{source.name}-source-authority.json"
    authority.write_bytes(canonical_authority_document({
        "schema_version": SOURCE_AUTHORITY_SCHEMA_VERSION,
        "authority_id": "test-authority",
        "purpose": "tournament_cache_source_admission",
        "source_manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": _sha(manifest_path),
            "bytes": manifest_path.stat().st_size,
            "schema_version": "ffm_ssl_corpus_v1",
        },
        "admitted_streams": [f"{ticker}@{timeframe}"],
        "cache_construction_admitted": True,
        "training_admitted": False,
    }))
    return authority, _sha(authority)


def test_balanced_schedule_is_deterministic_and_stream_uniform():
    starts = np.arange(120)
    bounds = np.array([[0, 100], [100, 110], [110, 120]])
    first, groups = balanced_schedule(starts, bounds, examples=3000, seed=9)
    second, groups2 = balanced_schedule(starts, bounds, examples=3000, seed=9)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(groups, groups2)
    assert schedule_fingerprint(first, groups) == schedule_fingerprint(second, groups2)
    counts = np.bincount(groups, minlength=3)
    assert np.max(np.abs(counts - 1000)) < 100
    assert ((first[groups == 0] >= 0) & (first[groups == 0] < 100)).all()
    assert ((first[groups == 1] >= 100) & (first[groups == 1] < 110)).all()


def test_contexts_are_suffixes_of_common_parent():
    big = np.stack([np.arange(40), np.arange(40) + 100], axis=1).astype(np.float32)
    starts = np.array([2, 20])
    short = gather_contexts(big, starts, 4, parent_length=8)
    full = gather_contexts(big, starts, 8, parent_length=8)
    np.testing.assert_array_equal(short, full[:, :, -4:])
    np.testing.assert_array_equal(gather_parent(big, starts, 8).transpose(0, 2, 1), full)


def test_time_features_align_with_stream_local_global_offsets():
    streams = [
        {"ts": pd.date_range("2024-01-01", periods=20, freq="1min", tz="UTC")},
        {"ts": pd.date_range("2024-02-02 12:00", periods=20, freq="1h", tz="UTC")},
    ]
    features = gather_time_features(
        starts=np.array([2, 23]), group_ids=np.array([0, 1]), streams=streams,
        row_bounds=np.array([[0, 20], [20, 40]]), length=3,
    )
    assert features[0, :, 0].tolist() == [2, 3, 4]
    assert features[1, :, 1].tolist() == [15, 16, 17]
    assert features[1, 0, 3] == 2 and features[1, 0, 4] == 2


def test_time_features_convert_to_explicit_cme_venue_timezone_across_dst():
    stream = {
        "ts": pd.to_datetime(
            ["2024-03-10 07:59:00+00:00", "2024-03-10 08:00:00+00:00"],
            utc=True,
        )
    }
    features = gather_time_features(
        starts=np.array([0]),
        group_ids=np.array([0]),
        streams=[stream],
        row_bounds=np.array([[0, 2]]),
        length=2,
        timezone="America/Chicago",
    )
    assert features[0, :, 0].tolist() == [59, 0]
    assert features[0, :, 1].tolist() == [1, 3]
    assert features[0, :, 2].tolist() == [6, 6]
    assert features[0, :, 3].tolist() == [10, 10]
    assert features[0, :, 4].tolist() == [3, 3]


def test_time_features_reject_invalid_timezone():
    with np.testing.assert_raises_regex(ValueError, "IANA timezone"):
        gather_time_features(
            starts=np.array([0]),
            group_ids=np.array([0]),
            streams=[{"ts": pd.date_range("2024-01-01", periods=2, tz="UTC")}],
            row_bounds=np.array([[0, 2]]),
            length=2,
            timezone="not/a_timezone",
        )


def test_binary_cache_is_bounded_and_mmap_loadable(tmp_path):
    source, cache = tmp_path / "source", tmp_path / "cache"
    source.mkdir()
    ts = pd.date_range("2019-06-28", "2025-07-03", freq="1D", tz="UTC")
    close = np.arange(len(ts), dtype=float) + 100
    pd.DataFrame({
        "datetime": ts, "open": close, "high": close + 1, "low": close - 1,
        "close": close, "volume": 10, "contract_id": "X",
    }).to_csv(source / "ES_1D.csv", index=False)
    authority, authority_sha = _write_source_authority(source)
    report = build_cache(
        source, cache, ("ES",), ("1D",),
        source_authority_path=authority,
        source_authority_sha256=authority_sha,
        verbose=False,
    )
    assert not report["interval"]["contains_oos"]
    assert report["training_admitted"] is False
    receipt_sha = cache_manifest_sha256(cache)
    streams = load_cache(
        cache, ("ES",), ("1D",),
        expected_manifest_sha256=receipt_sha,
        verbose=False,
    )
    times = pd.DatetimeIndex(streams[0]["ts"]).tz_localize("UTC")
    assert times.min() == pd.Timestamp("2019-07-01", tz="UTC")
    assert times.max() == pd.Timestamp("2025-06-30", tz="UTC")
    assert isinstance(streams[0]["ohlcv"], np.memmap)


def test_cache_requires_and_revalidates_source_and_array_hashes(tmp_path):
    source, cache = tmp_path / "source", tmp_path / "cache"
    source.mkdir()
    ts = pd.date_range("2019-06-28", "2025-07-03", freq="1D", tz="UTC")
    close = np.arange(len(ts), dtype=float) + 100
    pd.DataFrame({
        "datetime": ts, "open": close, "high": close + 1, "low": close - 1,
        "close": close, "volume": 10, "contract_id": "X",
    }).to_csv(source / "ES_1D.csv", index=False)
    with np.testing.assert_raises(TypeError):
        build_cache(source, cache, ("ES",), ("1D",), verbose=False)
    authority, authority_sha = _write_source_authority(source)
    build_cache(
        source, cache, ("ES",), ("1D",),
        source_authority_path=authority,
        source_authority_sha256=authority_sha,
        verbose=False,
    )
    receipt_sha = cache_manifest_sha256(cache)
    with (cache / "ES_1D.ohlcv.npy").open("ab") as stream:
        stream.write(b"tamper")
    with np.testing.assert_raises_regex(ValueError, "differs from its declared size"):
        load_cache(
            cache, ("ES",), ("1D",),
            expected_manifest_sha256=receipt_sha,
            verbose=False,
        )


def test_cache_authority_and_source_reject_symlink_and_hardlink_transport(tmp_path):
    def make_source(root):
        root.mkdir()
        ts = pd.date_range("2019-06-28", "2025-07-03", freq="1D", tz="UTC")
        close = np.arange(len(ts), dtype=float) + 100
        csv = root / "ES_1D.csv"
        pd.DataFrame({
            "datetime": ts, "open": close, "high": close + 1, "low": close - 1,
            "close": close, "volume": 10, "contract_id": "X",
        }).to_csv(csv, index=False)
        authority, authority_sha = _write_source_authority(root)
        return csv, authority, authority_sha

    source = tmp_path / "symlink-authority-source"
    _, authority, authority_sha = make_source(source)
    authority_link = tmp_path / "authority-link.json"
    authority_link.symlink_to(authority)
    with np.testing.assert_raises_regex(ValueError, "symlink"):
        build_cache(
            source, tmp_path / "cache-a", ("ES",), ("1D",),
            source_authority_path=authority_link,
            source_authority_sha256=authority_sha,
            verbose=False,
        )

    source = tmp_path / "hardlink-authority-source"
    _, authority, authority_sha = make_source(source)
    authority_hardlink = tmp_path / "authority-hardlink.json"
    authority_hardlink.hardlink_to(authority)
    with np.testing.assert_raises_regex(ValueError, "bounded regular file"):
        build_cache(
            source, tmp_path / "cache-b", ("ES",), ("1D",),
            source_authority_path=authority,
            source_authority_sha256=authority_sha,
            verbose=False,
        )

    source = tmp_path / "symlink-source-file"
    csv, authority, authority_sha = make_source(source)
    real_csv = csv.with_name("ES_1D.real.csv")
    csv.rename(real_csv)
    csv.symlink_to(real_csv)
    with np.testing.assert_raises_regex(ValueError, "symlink"):
        build_cache(
            source, tmp_path / "cache-c", ("ES",), ("1D",),
            source_authority_path=authority,
            source_authority_sha256=authority_sha,
            verbose=False,
        )


def test_cache_revalidates_source_file_and_external_manifest_receipt(tmp_path):
    source, cache = tmp_path / "source", tmp_path / "cache"
    source.mkdir()
    ts = pd.date_range("2019-06-28", "2025-07-03", freq="1D", tz="UTC")
    close = np.arange(len(ts), dtype=float) + 100
    csv = source / "ES_1D.csv"
    pd.DataFrame({
        "datetime": ts, "open": close, "high": close + 1, "low": close - 1,
        "close": close, "volume": 10, "contract_id": "X",
    }).to_csv(csv, index=False)
    authority, authority_sha = _write_source_authority(source)
    build_cache(
        source, cache, ("ES",), ("1D",),
        source_authority_path=authority,
        source_authority_sha256=authority_sha,
        verbose=False,
    )
    receipt_sha = cache_manifest_sha256(cache)
    manifest_path = cache / "TOURNAMENT_CACHE.json"
    original = manifest_path.read_bytes()
    manifest_path.write_bytes(original + b" ")
    with np.testing.assert_raises_regex(ValueError, "canonical JSON|physical SHA-256"):
        load_cache(
            cache, ("ES",), ("1D",),
            expected_manifest_sha256=receipt_sha,
            verbose=False,
        )
    manifest_path.write_bytes(original)
    with csv.open("ab") as stream:
        stream.write(b"tamper")
    with np.testing.assert_raises_regex(ValueError, "declared size|identity mismatch"):
        load_cache(
            cache, ("ES",), ("1D",),
            expected_manifest_sha256=receipt_sha,
            verbose=False,
        )
