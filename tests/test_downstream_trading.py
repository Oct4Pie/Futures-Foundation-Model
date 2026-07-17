import numpy as np
import pytest
import subprocess
import sys

from futures_foundation.finetune.downstream_trading import build_policy_events
from scripts.benchmark_downstream_trading import (
    apply_concurrency,
    choose_calibrated_threshold,
    inner_calibration_rows,
    policy_feature_matrix,
    stable_policy_seed,
    trade_metrics,
)


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
