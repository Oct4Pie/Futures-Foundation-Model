import json
from pathlib import Path

import numpy as np
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
from futures_foundation.finetune.event_contexts import SCHEMA_VERSION as CONTEXT_SCHEMA
from futures_foundation.finetune.event_contexts import save_context_shard


def _shard(path: Path, ticker: str, timeframe: str, n: int = 20):
    horizons = np.array([60, 180, 360], np.int32)
    decision = np.arange(n, dtype=np.int64) * 1_000 + 10_000
    arrays = {
        "ticker": np.full(n, ticker), "timeframe": np.full(n, timeframe),
        "contract_id": np.full(n, f"{ticker}Z4"),
        "context_start_source_idx": np.arange(n),
        "decision_source_idx": np.arange(n) + 255,
        "decision_time_ns": decision, "block_id": np.arange(n) // 4,
        "sample_weight": np.ones(n, np.float32), "features": np.ones((n, 2), np.float32),
        "feature_names": np.array(["a", "b"]), "tag_names": np.array(["tag"]),
        "horizons_minutes": horizons, "targets_r": np.array([1, 2], np.float32),
        "directions": np.array([1, -1], np.int8), "causal_scale": np.ones(n, np.float32),
        "context_direction": np.ones(n, np.int8), "cme_session_minute": np.arange(n),
        "contract_segment_id": np.zeros(n, np.int64),
        "bars_since_contract_start": np.arange(n),
        "terminal_log_return": np.ones((n, 3), np.float32),
        "terminal_move_r": np.ones((n, 3), np.float32),
        "forward_abs_move_r": np.ones((n, 3), np.float32),
        "forward_realized_vol": np.ones((n, 3), np.float32),
        "upside_mfe_r": np.ones((n, 3), np.float32),
        "downside_mae_r": np.ones((n, 3), np.float32),
        "forward_trend_eff": np.ones((n, 3), np.float32),
        "label_end_time_ns": decision[:, None] + np.array([100, 200, 300]),
        "trend_path_class": np.ones((n, 3), np.int8),
        "barrier_state": np.zeros((n, 3, 2, 2), np.int8),
        "time_to_favorable_minutes": np.zeros((n, 3, 2, 2), np.int32),
        "time_to_adverse_minutes": np.zeros((n, 3, 2, 2), np.int32),
        "policy_r_gross": np.zeros((n, 3, 2, 2), np.float32),
        "tags": np.zeros((n, 1), bool), "tag_direction": np.zeros((n, 1), np.int8),
        "tag_origin_source_idx": np.full((n, 1), -1, np.int64),
        "tag_htf_agreement": np.zeros((n, 1), bool), "htf_direction": np.zeros(n, np.int8),
    }
    metadata = {
        "schema_version": CONTEXT_SCHEMA, "ticker": ticker, "timeframe": timeframe,
        "rows": n, "source_rows": n, "event_rows": 0, "policy_events": 0,
        "config": {}, "split": {"oos_read": False}, "tag_counts": {"tag": 0},
    }
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
        "schema_version": "ffm_event_context_collection_v1", "status": "complete",
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
        assert np.all(label_end[train].max(axis=1) + 200 <= record["test_start_ns"])
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
        assert np.all(label_end[train].max(axis=1) + 200 <= record["test_start_ns"])
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
