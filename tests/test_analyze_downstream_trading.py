import numpy as np
import pytest

from scripts.analyze_downstream_trading import (
    _aligned_arm_rows,
    benjamini_hochberg,
    fixed_cost_metrics,
    paired_utility_interval,
)


def test_paired_utility_interval_detects_positive_lift():
    delta = np.full(40, 0.2)
    days = np.arange(40, dtype=np.int64) * 86_400 * 1_000_000_000

    result = paired_utility_interval(delta, days, repetitions=100, seed=7)

    assert result["delta_r_per_candidate"] == pytest.approx(0.2)
    assert result["ci95_low"] > 0
    assert result["bootstrap_positive_probability"] == 1.0
    assert result["bootstrap_two_sided_p"] == 0.0
    assert result["calendar_blocks"] == 6


def test_paired_utility_interval_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="equal non-empty"):
        paired_utility_interval(np.ones(2), np.ones(1, dtype=np.int64))
    with pytest.raises(ValueError, match="invalid"):
        paired_utility_interval(np.asarray([np.nan]), np.asarray([1], np.int64))


def test_aligned_arm_rows_aligns_storage_order_and_rejects_mismatch():
    predictions = {
        "policy_index": np.asarray([0, 0, 0, 0], np.int16),
        "arm_index": np.asarray([0, 0, 1, 1], np.int8),
        "event_row": np.asarray([3, 1, 1, 3], np.int32),
        "fold": np.asarray([2, 1, 1, 2], np.int8),
    }

    left, right = _aligned_arm_rows(predictions, 0, 0, 1)

    assert predictions["event_row"][left].tolist() == [1, 3]
    assert predictions["event_row"][right].tolist() == [1, 3]
    predictions["fold"][3] = 1
    with pytest.raises(ValueError, match="fold"):
        _aligned_arm_rows(predictions, 0, 0, 1)


def test_benjamini_hochberg_is_monotone_in_rank():
    adjusted = benjamini_hochberg(np.asarray([0.04, 0.001, 0.03, 0.5]))

    assert adjusted.tolist() == pytest.approx([0.0533333333, 0.004, 0.0533333333, 0.5])
    with pytest.raises(ValueError, match="p-values"):
        benjamini_hochberg(np.asarray([1.1]))


def test_fixed_cost_metrics_reprices_exact_executions():
    result = fixed_cost_metrics(
        gross_r=np.asarray([1.0, -0.5]),
        fee_r=np.asarray([0.1, 0.1]),
        slippage_r_per_tick=np.asarray([0.2, 0.2]),
        slippage_ticks=1.0,
    )

    assert result["executed"] == 2
    assert result["mean_r"] == pytest.approx(-0.05)
    assert result["total_r"] == pytest.approx(-0.1)
    assert result["profit_factor"] == pytest.approx(0.7 / 0.8)
