import numpy as np
import pytest
import subprocess
import sys

from futures_foundation.finetune.downstream_trading import build_policy_events
from futures_foundation.finetune.calibration import (
    apply_isotonic_expected_value,
    fit_isotonic_expected_value,
)
from scripts.benchmark_downstream_trading import (
    apply_concurrency,
    choose_calibrated_threshold,
    choose_stable_calibrated_threshold,
    inner_calibration_rows,
    nested_context_splits,
    nested_oof_predictions,
    policy_feature_matrix,
    stable_policy_seed,
    trade_metrics,
)
from scripts.analyze_downstream_trading import slippage_r_per_round_trip_tick


def test_build_policy_events_expands_net_tick_costed_outcomes(tmp_path):
    shard = tmp_path / "ES_1min.npz"
    np.savez_compressed(
        shard,
        tag_names=np.asarray(["supertrend_flip"]), horizons_minutes=np.asarray([60]),
        targets_r=np.asarray([2.0], np.float32), policy_mode_names=np.asarray(["atr"]),
        policy_event_context_row=np.asarray([4]), policy_event_tag_index=np.asarray([0], np.int8),
        policy_event_direction=np.asarray([1], np.int8),
        policy_valid=np.asarray([[True]]), policy_risk_ticks=np.asarray([[4.0]], np.float32),
        policy_cost_r=np.asarray([[0.25]], np.float32),
        policy_realized_r=np.asarray([[[[1.75]]]], np.float32),
        policy_reached=np.asarray([[[[True]]]]),
        policy_exit_time_ns=np.asarray([[[[200]]]], np.int64),
    )
    import hashlib
    digest = hashlib.sha256(shard.read_bytes()).hexdigest()
    sample = {
        "stream_id": np.asarray(["ES@1min"]), "shard_row": np.asarray([4]),
        "decision_time_ns": np.asarray([100]), "ticker": np.asarray(["ES"]),
        "timeframe": np.asarray(["1min"]), "tag_names": np.asarray(["supertrend_flip"]),
    }

    arrays, metadata = build_policy_events(
        sample, np.asarray([0]), {"ES@1min": {"path": str(shard), "sha256": digest}},
        {"ES": {"tick_size": 0.25, "tick_value_usd": 12.5, "fee_rt_usd": 5.0}},
    )

    assert metadata["rows"] == 1
    assert arrays["policy_key"].item() == "supertrend_flip__atr__60m__2R"
    assert arrays["gross_r"].item() == 2.0
    assert arrays["slippage_r"].item() == 0.25
    assert arrays["fee_r"].item() == pytest.approx(0.1)
    assert arrays["realized_r"].item() == pytest.approx(1.65)
    assert arrays["exit_time_ns"].item() == 200


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
        assert label_end[train_a].max() + 20 <= times[test_a].min()


def test_nested_oof_predictions_use_disjoint_later_rows():
    context = np.arange(80, dtype=np.float32)[:, None]
    splits = [(np.arange(30), np.arange(30, 45)), (np.arange(45), np.arange(45, 60))]
    y = (0.05 * context[:, 0]).astype(np.float32)
    rows, score, folds, records = nested_oof_predictions(
        [context, context], splits, np.arange(80), np.ones((80, 1), np.float32),
        y, np.asarray(["ES"] * 80), head="linear", seed=3,
        min_train=20, min_test=10,
    )

    assert len(rows) == len(score) == 30
    assert len(np.unique(rows)) == len(rows)
    assert set(np.unique(folds)) == {1, 2}
    assert all(record["status"] == "complete" for record in records)


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
