import hashlib
import json
import numpy as np
import pandas as pd
import pytest
import subprocess
import sys
from pathlib import Path

from futures_foundation.execution_economics import load_execution_economics
from futures_foundation.finetune.downstream_trading import (
    build_policy_events, load_policy_events, save_policy_events,
)
from futures_foundation.finetune.event_contexts import (
    TAG_NAMES,
    EventContextConfig,
    materialize_context_stream,
    save_context_shard,
)
from futures_foundation.finetune.path_labels import PathLabelConfig
from futures_foundation.finetune.calibration import (
    apply_isotonic_expected_value,
    fit_isotonic_expected_value,
)
from scripts.benchmark_downstream_trading import (
    apply_concurrency,
    barrier_outcome_classes,
    choose_calibrated_threshold,
    choose_stable_calibrated_threshold,
    expected_net_r_from_barrier,
    fit_predict_residual_fold,
    inner_calibration_rows,
    nested_context_splits,
    nested_oof_predictions,
    policy_feature_matrix,
    stable_policy_seed,
    trade_metrics,
)
from scripts.analyze_downstream_trading import slippage_r_per_round_trip_tick


def _policy_source(tmp_path, *, gross_r=2.0):
    shard = tmp_path / "ES_1min.npz"
    rows = 900
    rng = np.random.default_rng(12)
    close = 100.0 + np.cumsum(rng.normal(0.0, 0.25, rows))
    open_ = np.r_[close[0], close[:-1]]
    width = rng.uniform(0.05, 0.25, rows)
    frame = pd.DataFrame({
        "datetime": pd.date_range("2024-01-01", periods=rows, freq="1min", tz="UTC"),
        "open": open_, "high": np.maximum(open_, close) + width,
        "low": np.minimum(open_, close) - width, "close": close,
        "volume": rng.integers(1, 1000, rows),
        "contract_id": "ESH4", "source_row_idx": np.arange(rows),
    })
    config = EventContextConfig(
        eval_start=str(frame["datetime"].iloc[0]),
        eval_end=str(frame["datetime"].iloc[-1] + pd.Timedelta(minutes=1)),
        context_bars=256,
        path=PathLabelConfig(
            horizons_minutes=(60,), targets_r=(2.0,),
            atr_period=20, context_minutes=10, barrier_chunk_rows=64,
        ),
    )
    arrays, metadata = materialize_context_stream(
        frame, ticker="ES", timeframe="1min",
        config=config, execution_economics=_execution_costs(),
    )
    context_row = 4
    tag_index = TAG_NAMES.index("supertrend_flip")
    arrays["tags"][:] = False
    arrays["tag_direction"][:] = 0
    arrays["tag_origin_source_idx"][:] = -1
    arrays["tag_htf_agreement"][:] = False
    arrays["tags"][context_row, tag_index] = True
    arrays["tag_direction"][context_row, tag_index] = 1
    arrays["tag_origin_source_idx"][context_row, tag_index] = arrays["decision_source_idx"][context_row]
    arrays["htf_direction"][context_row] = 1
    arrays["tag_htf_agreement"][context_row, tag_index] = True

    decision_ns = int(arrays["decision_time_ns"][context_row])
    exit_ns = decision_ns + 100
    arrays["policy_mode_names"] = np.asarray(("atr_stop", "structural_stop"))
    arrays["policy_event_context_row"] = np.asarray([context_row], np.int64)
    arrays["policy_event_tag_index"] = np.asarray([tag_index], np.int8)
    arrays["policy_event_direction"] = np.asarray([1], np.int8)
    arrays["policy_valid"] = np.asarray([[True, False]])
    arrays["policy_risk_price"] = np.asarray([[1.0, np.nan]], np.float32)
    arrays["policy_risk_ticks"] = np.asarray([[4.0, np.nan]], np.float32)
    arrays["policy_barrier_state"] = np.asarray([[[[1]], [[-1]]]], np.int8)
    arrays["policy_gross_r"] = np.asarray([[[[gross_r]], [[np.nan]]]], np.float32)
    arrays["policy_reached"] = np.asarray([[[[True]], [[False]]]])
    arrays["policy_exit_time_ns"] = np.asarray([[[[exit_ns]], [[-1]]]], np.int64)
    metadata["event_rows"] = 1
    metadata["policy_events"] = 1
    metadata["tag_counts"] = {
        name: int(arrays["tags"][:, index].sum())
        for index, name in enumerate(TAG_NAMES)
    }
    manifest = save_context_shard(shard, arrays, metadata)
    manifest_path = tmp_path / "ES_1min.npz.manifest.json"
    source_info = {
        "path": str(shard),
        "sha256": manifest["artifact"]["sha256"],
        "manifest_path": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "content_fingerprint": manifest["content_fingerprint"],
    }
    sample = {
        "stream_id": np.asarray(["ES@1min"]),
        "shard_row": np.asarray([context_row]),
        "decision_time_ns": np.asarray([decision_ns]),
        "ticker": np.asarray(["ES"]),
        "timeframe": np.asarray(["1min"]),
        "tag_names": np.asarray(TAG_NAMES),
    }
    return shard, manifest_path, source_info, sample, exit_ns


def _execution_costs():
    return load_execution_economics(
        Path(__file__).resolve().parents[1] / "config/execution_costs.yaml",
        evaluation_start="2024-01-01T00:00:00Z",
        evaluation_end="2025-01-01T00:00:00Z",
        required_roots=("ES",),
    )


def test_build_policy_events_applies_capability_costs_to_gross_outcomes(tmp_path):
    _, _, source_info, sample, exit_ns = _policy_source(tmp_path)

    arrays, metadata = build_policy_events(
        sample, np.asarray([0]), {"ES@1min": source_info}, _execution_costs(),
        slippage_ticks=1.0,
    )

    assert metadata["rows"] == 1
    assert arrays["policy_key"].item() == "supertrend_flip__atr_stop__60m__2R"
    assert arrays["gross_r"].item() == 2.0
    assert arrays["barrier_state"].item() == 1
    assert arrays["slippage_r"].item() == 0.25
    assert arrays["fee_r"].item() == pytest.approx(4.36 / 50.0)
    assert arrays["realized_r"].item() == pytest.approx(2.0 - 0.25 - 4.36 / 50.0)
    assert arrays["exit_time_ns"].item() == exit_ns


def test_build_policy_events_rejects_manifest_tampering_even_if_caller_rehashes_it(tmp_path):
    _, manifest_path, source_info, sample, _ = _policy_source(tmp_path)
    document = json.loads(manifest_path.read_text())
    document["metadata"]["config"]["context_bars"] = 128
    manifest_path.write_text(json.dumps(document))
    source_info["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="content fingerprint mismatch"):
        build_policy_events(
            sample, np.asarray([0]), {"ES@1min": source_info}, _execution_costs(),
            slippage_ticks=0.0,
        )


def test_build_policy_events_rejects_missing_gross_outcome_array(tmp_path):
    shard, _, source_info, sample, _ = _policy_source(tmp_path)
    with np.load(shard, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files if key != "policy_gross_r"}
    np.savez_compressed(shard, **arrays)
    source_info["sha256"] = hashlib.sha256(shard.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        build_policy_events(
            sample, np.asarray([0]), {"ES@1min": source_info}, _execution_costs(),
            slippage_ticks=0.0,
        )


def test_policy_artifact_binds_cost_and_lineage_metadata_and_legacy_is_explicit(tmp_path):
    _, _, source_info, sample, _ = _policy_source(tmp_path)
    arrays, metadata = build_policy_events(
        sample, np.asarray([0]), {"ES@1min": source_info}, _execution_costs(),
        slippage_ticks=0.0,
    )
    path = tmp_path / "policy.npz"
    save_policy_events(path, arrays, metadata)
    manifest_path = tmp_path / "policy.npz.manifest.json"
    document = json.loads(manifest_path.read_text())
    document["execution_economics"]["instruments"]["ES"]["fee_rt_usd"] = 0.0
    manifest_path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="content fingerprint mismatch"):
        load_policy_events(path)

    save_policy_events(path, arrays, metadata)
    document = json.loads(manifest_path.read_text())
    document["schema_version"] = "ffm_downstream_policy_events_v2"
    manifest_path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="unsupported"):
        load_policy_events(path)
    loaded, legacy = load_policy_events(path, allow_legacy=True)
    assert len(loaded["context_row"]) == 1
    assert legacy["schema_version"] == "ffm_downstream_policy_events_v2"


def test_concurrency_suppresses_overlapping_same_ticker_trades():
    events = {
        "ticker": np.asarray(["ES", "ES", "NQ"]),
        "signal_time_ns": np.asarray([10, 15, 15]),
        "exit_time_ns": np.asarray([20, 30, 25]),
    }
    executed = apply_concurrency(events, np.asarray([0, 1, 2]), np.ones(3, bool))
    assert executed.tolist() == [True, False, True]


def test_trade_metrics_use_positive_net_r_as_win_rate():
    result = trade_metrics(
        np.asarray([2.0, -1.0, 0.5]), np.asarray([True, False, False]),
        np.asarray([1, 2, 3]),
    )
    assert result["trades"] == 3
    assert result["win_rate"] == 2 / 3
    assert result["target_hit_rate"] == 1 / 3
    assert result["profit_factor"] == 2.5


def test_barrier_outcome_decomposition_is_mutually_exclusive_and_fee_aware():
    classes = barrier_outcome_classes(np.asarray([0, 1, 2, 3], np.int8))
    assert classes.tolist() == [2, 0, 1, 1]
    score = expected_net_r_from_barrier(
        np.asarray([[0.5, 0.25, 0.25], [0.0, 0.0, 1.0]]),
        np.asarray([0.4, -0.2]), np.asarray([0.1, 0.05]), target_r=3.0,
    )
    np.testing.assert_allclose(score, [1.25, -0.25])


def test_slippage_sensitivity_uses_risk_geometry_when_primary_slippage_is_zero():
    values = slippage_r_per_round_trip_tick({"risk_ticks": np.asarray([4.0, 10.0])})
    np.testing.assert_allclose(values, [0.25, 0.1])


def test_policy_features_include_known_cost_burden():
    events = {
        "direction": np.asarray([1]), "risk_ticks": np.asarray([4.0]),
        "slippage_r": np.asarray([0.25]), "fee_r": np.asarray([0.10]),
        "total_cost_r": np.asarray([0.35]),
    }
    values, names = policy_feature_matrix(events, np.asarray([0]))
    assert values.shape == (1, 5)
    assert names[-1] == "total_cost_r"
    assert values[0, -1] == pytest.approx(0.35)


def test_trading_benchmark_direct_entrypoint_loads():
    completed = subprocess.run(
        [sys.executable, "scripts/benchmark_downstream_trading.py", "--help"],
        check=False, capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_policy_seed_is_order_invariant_and_policy_specific():
    first = stable_policy_seed(7, "fractal__atr__360m__3R", 2)

    assert first == stable_policy_seed(7, "fractal__atr__360m__3R", 2)
    assert first != stable_policy_seed(7, "fractal__atr__360m__3R", 3)
    assert first != stable_policy_seed(7, "supertrend__atr__360m__3R", 2)


def test_inner_calibration_is_chronological_and_label_purged():
    signal = np.arange(200, dtype=np.int64) * 10
    events = {"signal_time_ns": signal, "exit_time_ns": signal + 15}

    fit, calibration, start = inner_calibration_rows(
        events, np.arange(200), np.arange(200), fraction=0.2,
        min_fit=100, min_calibration=30,
    )

    assert len(fit) >= 100 and len(calibration) >= 30
    assert np.all(events["exit_time_ns"][fit] < start)
    assert np.all(events["signal_time_ns"][calibration] >= start)


def test_calibrated_threshold_uses_net_r_per_candidate_and_concurrency():
    events = {
        "ticker": np.asarray(["ES"] * 10),
        "signal_time_ns": np.arange(10, dtype=np.int64) * 10,
        "exit_time_ns": np.arange(10, dtype=np.int64) * 10 + 5,
        "realized_r": np.asarray([-1.0] * 5 + [1.0] * 5),
    }
    score = np.arange(10, dtype=np.float64)

    result = choose_calibrated_threshold(
        events, np.arange(10), np.arange(10), score,
        quantiles=(0.5, 0.8), min_executed=2,
    )

    assert result["threshold"] == pytest.approx(4.5)
    assert result["executed"] == 5
    assert result["r_per_candidate"] == pytest.approx(0.5)


def test_calibrated_threshold_never_drops_below_economic_floor():
    events = {
        "ticker": np.asarray(["ES"] * 6),
        "signal_time_ns": np.arange(6, dtype=np.int64),
        "exit_time_ns": np.arange(6, dtype=np.int64),
        "realized_r": np.ones(6),
    }
    result = choose_calibrated_threshold(
        events, np.arange(6), np.arange(6), np.linspace(-2.0, 0.2, 6),
        quantiles=(0.5, 0.8), min_executed=1, floor_threshold=0.0,
    )

    assert result["threshold"] >= 0.0


def test_isotonic_expected_value_is_monotonic_and_clipped():
    raw = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0])
    realized = np.asarray([-1.0, -0.5, 0.2, 0.1, 1.0])
    calibration = fit_isotonic_expected_value(raw, realized)
    applied = apply_isotonic_expected_value(np.asarray([-99.0, *raw, 99.0]), calibration)

    assert np.all(np.diff(applied) >= 0)
    assert applied[0] == applied[1]
    assert applied[-1] == applied[-2]


def test_nested_context_splits_cannot_read_outside_outer_training_rows():
    times = np.repeat(np.arange(120, dtype=np.int64) * 100, 2)
    groups = np.tile(np.asarray(["ES", "NQ"]), 120)
    label_end = times + 10
    outer = np.flatnonzero(times < 9_000)
    first, _ = nested_context_splits(
        times, label_end, groups, outer, folds=2, embargo_ns=20,
    )
    changed_end = label_end.copy()
    changed_end[times >= 9_000] += 1_000_000
    second, _ = nested_context_splits(
        times, changed_end, groups, outer, folds=2, embargo_ns=20,
    )

    for (train_a, test_a), (train_b, test_b) in zip(first, second):
        np.testing.assert_array_equal(train_a, train_b)
        np.testing.assert_array_equal(test_a, test_b)
        assert set(train_a).issubset(set(outer)) and set(test_a).issubset(set(outer))
        assert label_end[train_a].max() + 20 < times[test_a].min()


def test_nested_oof_predictions_use_disjoint_later_rows():
    context = np.arange(80, dtype=np.float32)[:, None]
    splits = [(np.arange(30), np.arange(30, 45)), (np.arange(45), np.arange(45, 60))]
    y = (0.05 * context[:, 0]).astype(np.float32)
    events = {
        "realized_r": y, "ticker": np.asarray(["ES"] * 80),
    }
    rows, score, folds, records = nested_oof_predictions(
        [context, context], splits, np.arange(80), np.arange(80),
        np.ones((80, 1), np.float32),
        events, head="xgb", objective="direct_r", target_r=3.0, seed=3,
        min_train=20, min_test=10,
    )

    assert len(rows) == len(score) == 30
    assert len(np.unique(rows)) == len(rows)
    assert set(np.unique(folds)) == {1, 2}
    assert all(record["status"] == "complete" for record in records)


def test_residual_fusion_predictions_ignore_test_and_future_outcomes():
    n = 400
    signal = np.arange(n, dtype=np.int64) * 10
    base = np.column_stack((np.sin(np.arange(n) / 17), np.arange(n) % 5)).astype(np.float32)
    residual = np.column_stack((np.cos(np.arange(n) / 11), np.arange(n) % 7)).astype(np.float32)
    y = (0.2 * base[:, 0] + 0.1 * residual[:, 0]).astype(np.float32)
    events = {
        "realized_r": y.copy(), "ticker": np.asarray(["ES"] * n),
        "signal_time_ns": signal, "exit_time_ns": signal + 1,
    }
    train, test = np.arange(300), np.arange(300, 350)
    first = fit_predict_residual_fold(
        residual, base, events, np.arange(n), train, test, seed=9,
        min_base_fit=100, min_residual_fit=50,
    )
    changed = dict(events)
    changed["realized_r"] = y.copy()
    changed["realized_r"][300:] += 1000.0
    second = fit_predict_residual_fold(
        residual, base, changed, np.arange(n), train, test, seed=9,
        min_base_fit=100, min_residual_fit=50,
    )
    np.testing.assert_allclose(first, second, rtol=0, atol=0)


def test_stable_threshold_returns_no_trade_without_positive_fold_lcb():
    n = 60
    events = {
        "ticker": np.asarray(["ES"] * n),
        "signal_time_ns": np.arange(n, dtype=np.int64) * 10,
        "exit_time_ns": np.arange(n, dtype=np.int64) * 10 + 1,
        "realized_r": -np.ones(n),
    }
    result = choose_stable_calibrated_threshold(
        events, np.arange(n), np.arange(n), np.linspace(0.0, 1.0, n),
        np.repeat(np.arange(1, 4), 20), quantiles=(0.5, 0.8),
        min_executed=5, min_coverage=0.05,
    )

    assert result["no_trade"] is True
    assert result["threshold"] > 1.0
