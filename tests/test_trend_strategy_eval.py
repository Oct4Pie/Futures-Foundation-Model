import numpy as np
import pandas as pd

from futures_foundation.finetune.trend_strategy_eval import (
    RulerConfig, _trade_outcomes, events_to_arrays, executable_risk, horizon_bars,
    metric_summary, summarize_events, timeframe_minutes,
)


def test_timeframe_and_fixed_wall_clock_horizon():
    assert timeframe_minutes("3min") == 3
    assert horizon_bars("1min", 6) == 360
    assert horizon_bars("3min", 6) == 120
    assert horizon_bars("60min", 6) == 6
    assert executable_risk(.01, .03125) == .03125
    assert executable_risk(.04, .03125) == .0625


def test_stop_first_same_bar_is_loss():
    o = np.array([100, 100, 100], float)
    h = np.array([100, 104, 100], float)
    l = np.array([100, 98, 100], float)
    c = np.array([100, 101, 100], float)
    result = _trade_outcomes(
        o, h, l, c, signal_idx=0, direction=1, risk=1, horizon=1,
        targets=(2.0, 3.0), cost_r=.03,
    )
    assert not result["reached"].any()
    np.testing.assert_allclose(result["realized"], [-1.03, -1.03])


def test_next_open_target_and_horizon_mark():
    o = np.array([90, 100, 100, 100], float)
    h = np.array([90, 101, 103.1, 100], float)
    l = np.array([90, 99.5, 100, 100], float)
    c = np.array([90, 100.5, 102.5, 100], float)
    result = _trade_outcomes(
        o, h, l, c, signal_idx=0, direction=1, risk=1, horizon=2,
        targets=(2.0, 4.0), cost_r=.03,
    )
    assert result["reached"].tolist() == [True, False]
    np.testing.assert_allclose(result["realized"], [1.97, 2.47], atol=1e-6)
    assert result["exit_idx"].tolist() == [2, 2]


def _event(strategy, day, value, reached):
    ts = pd.Timestamp(day, tz="UTC").value
    return {
        "strategy": strategy, "ticker": "ES", "timeframe": "3min",
        "signal_time_ns": ts, "entry_time_ns": ts + 1,
        "label_end_time_ns": ts + 3, "exit_time_ns": ts + 2,
        "source_signal_idx": 10, "direction": 1, "contract_id": "ESH5",
        "stop_mode": "atr", "raw_risk_price": .4, "risk_price": .5,
        "risk_ticks": 2.0, "one_tick_r": .25, "fee_r": .25,
        "cost_r": .5, "risk_atr": .5, "peak_r": 3.0,
        "realized": np.array([1.97, value, value], np.float32),
        "reached": np.array([True, reached, reached]),
    }


def test_event_arrays_sort_and_summary():
    events = [
        _event("a", "2024-11-01", -1.03, False),
        _event("a", "2024-08-01", 2.97, True),
    ]
    arrays = events_to_arrays(events, (2.0, 3.0, 4.0))
    assert arrays["signal_time_ns"].tolist() == sorted(arrays["signal_time_ns"].tolist())
    cfg = RulerConfig(targets=(2.0, 3.0, 4.0), primary_target=3.0)
    report = summarize_events(arrays, cfg, folds=2)["a"]
    assert report["signals"] == 2
    assert report["wr"] == .5
    np.testing.assert_allclose(report["mean_r"], .97, atol=1e-6)
    assert report["profit_factor"] > 2.8
    assert set(report["by_target"]) == {"2.0", "3.0", "4.0"}
    np.testing.assert_allclose(report["cost_tick_sensitivity"]["0.0"]["mean_r"],
                               report["mean_r"] + .25, atol=1e-6)
    np.testing.assert_allclose(report["cost_tick_sensitivity"]["2.0"]["mean_r"],
                               report["mean_r"] - .25, atol=1e-6)


def test_metric_summary_empty_and_drawdown():
    assert metric_summary([], [])["signals"] == 0
    result = metric_summary([1.0, -2.0, 1.0], [True, False, True])
    assert result["max_drawdown_r"] == 2.0
    assert result["profit_factor"] == 1.0
