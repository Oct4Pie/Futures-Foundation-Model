import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from futures_foundation.finetune.downstream_sample import (
    build_balanced_sample,
    build_balanced_row_selection,
    load_balanced_sample,
    load_row_selection,
    purged_calendar_splits,
    purged_interval_splits,
    save_balanced_sample,
    save_row_selection,
    temporally_distributed_rows,
)
from futures_foundation.execution_economics import load_execution_economics
from futures_foundation.finetune.event_contexts import (
    COLLECTION_SCHEMA_VERSION,
    POLICY_TAG_NAMES,
    TAG_NAMES,
    EventContextConfig,
    materialize_context_stream,
    save_context_shard,
)
from futures_foundation.finetune.path_labels import PathLabelConfig


def _event_frame(ticker: str, rows: int = 900) -> pd.DataFrame:
    seed = sum(ord(character) for character in ticker)
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0.0, 0.25, rows))
    open_ = np.r_[close[0], close[:-1]]
    width = rng.uniform(0.05, 0.25, rows)
    return pd.DataFrame({
        "datetime": pd.date_range("2024-01-01", periods=rows, freq="1min", tz="UTC"),
        "open": open_, "high": np.maximum(open_, close) + width,
        "low": np.minimum(open_, close) - width, "close": close,
        "volume": rng.integers(1, 1000, rows),
        "contract_id": f"{ticker}H4", "source_row_idx": np.arange(rows),
    })


def _shard(path: Path, ticker: str, timeframe: str, n: int = 20, event_rows=()):
    frame = _event_frame(ticker)
    config = EventContextConfig(
        eval_start=str(frame["datetime"].iloc[0]),
        eval_end=str(frame["datetime"].iloc[-1] + pd.Timedelta(minutes=1)),
        context_bars=256,
        path=PathLabelConfig(
            horizons_minutes=(10, 20), targets_r=(1.0, 2.0),
            atr_period=20, context_minutes=10, barrier_chunk_rows=64,
        ),
    )
    economics = load_execution_economics(
        Path(__file__).resolve().parents[1] / "config/execution_costs.yaml",
        evaluation_start="2024-01-01T00:00:00Z",
        evaluation_end="2025-01-01T00:00:00Z",
        required_roots=(ticker,),
    )
    arrays, metadata = materialize_context_stream(
        frame, ticker=ticker, timeframe=timeframe,
        config=config, execution_economics=economics,
    )
    source_rows = int(metadata["rows"])
    assert source_rows >= n
    arrays = {
        name: (value[:n].copy() if value.ndim and value.shape[0] == source_rows else value.copy())
        for name, value in arrays.items()
    }

    tag_index = TAG_NAMES.index("fractal_zigzag")
    tags = np.zeros((n, len(TAG_NAMES)), dtype=bool)
    selected = np.asarray(event_rows, np.int64)
    tags[selected, tag_index] = True
    _, inverse, counts = np.unique(
        arrays["block_id"], return_inverse=True, return_counts=True,
    )
    block_weight = 1.0 / counts[inverse].astype(np.float64)
    arrays["sample_weight"] = (block_weight / block_weight.mean()).astype(np.float32)
    arrays["tags"] = tags
    arrays["tag_direction"] = np.zeros(tags.shape, np.int8)
    arrays["tag_direction"][selected, tag_index] = 1
    arrays["tag_origin_source_idx"] = np.full(tags.shape, -1, np.int64)
    arrays["tag_origin_source_idx"][selected, tag_index] = arrays["decision_source_idx"][selected]
    arrays["htf_direction"] = np.ones(n, np.int8)
    arrays["tag_htf_agreement"] = tags.copy()

    horizon_count = len(arrays["horizons_minutes"])
    target_count = len(arrays["targets_r"])
    arrays["policy_mode_names"] = np.asarray(("atr_stop", "structural_stop"))
    arrays["policy_event_context_row"] = np.empty(0, np.int64)
    arrays["policy_event_tag_index"] = np.empty(0, np.int8)
    arrays["policy_event_direction"] = np.empty(0, np.int8)
    arrays["policy_valid"] = np.empty((0, 2), bool)
    arrays["policy_risk_price"] = np.empty((0, 2), np.float32)
    arrays["policy_risk_ticks"] = np.empty((0, 2), np.float32)
    policy_shape = (0, 2, horizon_count, target_count)
    arrays["policy_barrier_state"] = np.empty(policy_shape, np.int8)
    arrays["policy_gross_r"] = np.empty(policy_shape, np.float32)
    arrays["policy_reached"] = np.empty(policy_shape, bool)
    arrays["policy_exit_time_ns"] = np.empty(policy_shape, np.int64)

    metadata["rows"] = n
    metadata["event_rows"] = int(tags.any(axis=1).sum())
    metadata["policy_events"] = 0
    metadata["tag_counts"] = {
        name: int(tags[:, index].sum()) for index, name in enumerate(TAG_NAMES)
    }
    assert "fractal_zigzag" not in POLICY_TAG_NAMES
    return save_context_shard(path, arrays, metadata)


def test_temporal_rows_are_unique_ordered_and_cover_stream():
    rows = temporally_distributed_rows(100, 12)
    assert len(rows) == len(np.unique(rows)) == 12
    assert np.all(np.diff(rows) > 0)
    assert rows[0] < 10 and rows[-1] > 89
    with pytest.raises(ValueError):
        temporally_distributed_rows(3, 4)


def test_balanced_sample_is_hash_bound_and_equal_per_stream(tmp_path):
    shards = {}
    for ticker in ("ES", "NQ"):
        path = tmp_path / f"{ticker}_1min.npz"
        manifest = _shard(path, ticker, "1min")
        shards[f"{ticker}@1min"] = {
            "path": str(path), "sha256": manifest["artifact"]["sha256"],
            "content_fingerprint": manifest["content_fingerprint"], "rows": 20,
        }
    collection = tmp_path / "MANIFEST.json"
    collection.write_text(json.dumps({
        "schema_version": COLLECTION_SCHEMA_VERSION, "status": "complete",
        "oos_read": False, "shards": shards,
    }))
    arrays, metadata = build_balanced_sample(collection, rows_per_stream=8)
    assert len(arrays["stream_id"]) == 16
    unique, counts = np.unique(arrays["stream_id"], return_counts=True)
    assert unique.tolist() == ["ES@1min", "NQ@1min"] and counts.tolist() == [8, 8]
    assert np.isclose(arrays["sample_weight"].mean(), 1.0)
    output = tmp_path / "sample.npz"
    manifest = save_balanced_sample(output, arrays, metadata)
    loaded, loaded_manifest = load_balanced_sample(output)
    assert loaded_manifest["content_fingerprint"] == manifest["content_fingerprint"]
    np.testing.assert_array_equal(loaded["shard_row"], arrays["shard_row"])
    with output.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_balanced_sample(output)


def test_legacy_sample_schema_requires_explicit_opt_in(tmp_path):
    arrays = {"stream_id": np.asarray(["ES@1min"])}
    metadata = {"schema_version": "ffm_downstream_sample_v3", "source_shards": {}}
    output = tmp_path / "sample.npz"
    save_balanced_sample(output, arrays, metadata)
    manifest_path = tmp_path / "sample.npz.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["schema_version"] = "ffm_downstream_sample_v1"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unsupported"):
        load_balanced_sample(output)
    loaded, _ = load_balanced_sample(output, allow_legacy=True)
    assert loaded["stream_id"].item() == "ES@1min"


def test_event_sample_retains_rare_events_and_caps_common_events_per_stream(tmp_path):
    shards = {}
    rows_by_ticker = {"ES": [1, 7], "NQ": list(range(12))}
    for ticker, event_rows in rows_by_ticker.items():
        path = tmp_path / f"{ticker}_1min.npz"
        manifest = _shard(path, ticker, "1min", event_rows=event_rows)
        shards[f"{ticker}@1min"] = {
            "path": str(path), "sha256": manifest["artifact"]["sha256"],
            "content_fingerprint": manifest["content_fingerprint"], "rows": 20,
        }
    collection = tmp_path / "MANIFEST.json"
    collection.write_text(json.dumps({
        "schema_version": COLLECTION_SCHEMA_VERSION, "status": "complete",
        "oos_read": False, "shards": shards,
    }))

    arrays, metadata = build_balanced_sample(
        collection, rows_per_stream=4, event_tags=("fractal_zigzag",),
    )

    selected = {
        stream: arrays["shard_row"][arrays["stream_id"] == stream].tolist()
        for stream in np.unique(arrays["stream_id"])
    }
    assert selected["ES@1min"] == [1, 7]
    assert len(selected["NQ@1min"]) == 4
    tag_index = np.flatnonzero(arrays["tag_names"] == "fractal_zigzag").item()
    assert arrays["tags"][:, tag_index].all()
    assert metadata["selection"]["method"] == "event_tag_stratified_midpoint_quantiles"
    total_weight = {
        stream: float(arrays["sample_weight"][arrays["stream_id"] == stream].sum())
        for stream in np.unique(arrays["stream_id"])
    }
    assert total_weight["ES@1min"] == pytest.approx(total_weight["NQ@1min"])


def test_purged_calendar_splits_use_actual_label_end_and_embargo():
    decision = np.arange(240, dtype=np.int64) * 1_000
    groups = np.repeat(np.arange(2), 120)
    # Give both groups identical calendar coverage.
    decision[120:] = decision[:120]
    label_end = np.column_stack((decision + 50, decision + 100))
    splits, contract = purged_calendar_splits(
        decision, label_end, groups, folds=3, embargo_ns=200,
    )
    assert len(splits) == 3 and contract["groups"] == 2
    for (train, test), record in zip(splits, contract["records"]):
        assert not np.intersect1d(train, test).size
        assert np.all(label_end[train].max(axis=1) + 200 < record["test_start_ns"])
        assert set(groups[train]) == {0, 1} and set(groups[test]) == {0, 1}


def test_purged_calendar_splits_use_common_group_coverage():
    first = np.arange(0, 120, dtype=np.int64) * 1_000
    second = np.arange(10, 130, dtype=np.int64) * 1_000
    decision = np.r_[first, second]
    groups = np.repeat(["ES", "ZB"], 120)
    splits, contract = purged_calendar_splits(
        decision, decision + 100, groups, folds=3, embargo_ns=100,
    )
    assert contract["shared_start_ns"] == 10_000
    assert contract["shared_end_ns"] == 119_001
    for train, test in splits:
        assert set(groups[train]) == {"ES", "ZB"}
        assert set(groups[test]) == {"ES", "ZB"}


def test_purged_interval_splits_train_on_history_and_test_only_declared_interval():
    timeline = np.arange(100, dtype=np.int64) * 1_000
    decision = np.r_[timeline, timeline]
    groups = np.repeat(["ES", "NQ"], len(timeline))
    label_end = np.column_stack((decision + 50, decision + 100))
    splits, contract = purged_interval_splits(
        decision, label_end, groups,
        eval_start_ns=50_000, eval_end_ns=90_000, folds=4, embargo_ns=200,
    )
    assert contract["eval_start_ns"] == 50_000
    assert contract["eval_end_ns"] == 90_000
    assert len(splits) == 4
    for (train, test), record in zip(splits, contract["records"]):
        assert decision[test].min() >= 50_000
        assert decision[test].max() < 90_000
        assert np.any(decision[train] < 50_000)
        assert np.all(label_end[train].max(axis=1) + 200 < record["test_start_ns"])
        assert set(groups[train]) == {"ES", "NQ"}
        assert set(groups[test]) == {"ES", "NQ"}


def test_purged_interval_splits_ignore_post_evaluation_perturbations():
    timeline = np.arange(100, dtype=np.int64) * 1_000
    decision = np.r_[timeline, timeline]
    groups = np.repeat(["ES", "NQ"], len(timeline))
    label_end = decision + 100
    kwargs = {
        "eval_start_ns": 50_000, "eval_end_ns": 90_000,
        "folds": 4, "embargo_ns": 200,
    }
    original, original_contract = purged_interval_splits(
        decision, label_end, groups, **kwargs,
    )
    changed_decision, changed_label_end = decision.copy(), label_end.copy()
    future = decision >= 90_000
    changed_decision[future] += 10_000_000
    changed_label_end[future] += 10_000_000
    changed, changed_contract = purged_interval_splits(
        changed_decision, changed_label_end, groups, **kwargs,
    )
    assert original_contract["contract_sha256"] == changed_contract["contract_sha256"]
    for (original_train, original_test), (changed_train, changed_test) in zip(original, changed):
        np.testing.assert_array_equal(original_train, changed_train)
        np.testing.assert_array_equal(original_test, changed_test)


def test_purged_interval_split_excludes_label_ending_exactly_at_test_boundary():
    decision = np.r_[np.arange(8), np.arange(8)].astype(np.int64)
    groups = np.repeat(["ES", "NQ"], 8)
    label_end = decision.copy()
    splits, contract = purged_interval_splits(
        decision, label_end, groups,
        eval_start_ns=4, eval_end_ns=8, folds=1, embargo_ns=0,
    )
    train, test = splits[0]
    test_start = contract["records"][0]["test_start_ns"]
    assert test_start == 4
    assert np.all(label_end[train] < test_start)
    assert not np.any(label_end[train] == test_start)
    assert np.all(decision[test] >= test_start)


def test_nested_row_selection_is_balanced_and_hash_bound(tmp_path):
    sample = {
        "stream_id": np.repeat(["ES@1min", "NQ@1min"], 20),
        "decision_time_ns": np.r_[np.arange(20), np.arange(20)],
    }
    sample_manifest = {
        "artifact": {"path": "sample.npz", "sha256": "sample-sha"},
        "content_fingerprint": "sample-fingerprint",
    }
    arrays, metadata = build_balanced_row_selection(
        sample, sample_manifest, rows_per_stream=8,
    )
    assert len(arrays["row_index"]) == 16
    selected_streams, counts = np.unique(
        sample["stream_id"][arrays["row_index"]], return_counts=True,
    )
    assert selected_streams.tolist() == ["ES@1min", "NQ@1min"]
    assert counts.tolist() == [8, 8]
    output = tmp_path / "rows.npz"
    manifest = save_row_selection(output, arrays, metadata)
    loaded, loaded_manifest = load_row_selection(output, sample_manifest=sample_manifest)
    assert loaded_manifest["content_fingerprint"] == manifest["content_fingerprint"]
    np.testing.assert_array_equal(loaded["row_index"], arrays["row_index"])

    all_rows, all_metadata = build_balanced_row_selection(
        sample, sample_manifest, rows_per_stream=None,
    )
    np.testing.assert_array_equal(all_rows["row_index"], np.arange(40))
    assert all_metadata["method"] == "all_sample_rows"
