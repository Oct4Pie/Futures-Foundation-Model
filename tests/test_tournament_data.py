import numpy as np

import pandas as pd

from futures_foundation.finetune.tournament_data import (
    balanced_schedule, build_cache, gather_contexts, gather_parent, gather_time_features,
    load_cache, schedule_fingerprint,
)


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


def test_binary_cache_is_bounded_and_mmap_loadable(tmp_path):
    source, cache = tmp_path / "source", tmp_path / "cache"
    source.mkdir()
    ts = pd.date_range("2019-06-28", "2025-07-03", freq="1D", tz="UTC")
    close = np.arange(len(ts), dtype=float) + 100
    pd.DataFrame({
        "datetime": ts, "open": close, "high": close + 1, "low": close - 1,
        "close": close, "volume": 10, "contract_id": "X",
    }).to_csv(source / "ES_1D.csv", index=False)
    report = build_cache(source, cache, ("ES",), ("1D",), verbose=False)
    assert not report["interval"]["contains_oos"]
    streams = load_cache(cache, ("ES",), ("1D",), verbose=False)
    times = pd.DatetimeIndex(streams[0]["ts"]).tz_localize("UTC")
    assert times.min() == pd.Timestamp("2019-07-01", tz="UTC")
    assert times.max() == pd.Timestamp("2025-06-30", tz="UTC")
    assert isinstance(streams[0]["ohlcv"], np.memmap)
