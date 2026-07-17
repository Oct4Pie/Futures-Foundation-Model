import numpy as np

from scripts.assess_model_event_pools import _binary_entropy, assess


def test_binary_entropy():
    assert _binary_entropy(0.0) == 0.0
    assert _binary_entropy(0.5) == 1.0


def test_assessment_counts_overlap_and_independent_contexts():
    rows = 8
    arrays = {
        "strategy": np.array(["supertrend__atr"] * 2 + ["atr_zigzag__atr"] * 2
                             + ["fractal_k2__atr"] * 2 + ["fractal_zigzag__atr"] * 2),
        "ticker": np.array(["ES"] * rows),
        "timeframe": np.array(["1min"] * rows),
        "signal_time_ns": np.array([1, 300, 1, 600, 1, 900, 1, 1200]),
        "source_signal_idx": np.array([1, 300, 1, 600, 1, 900, 1, 1200]),
        "direction": np.array([1, -1, 1, -1, 1, -1, 1, -1]),
        "targets": np.array([2.0, 3.0]),
        "reached": np.array([[True, False], [False, True]] * 4),
        "risk_ticks": np.ones(rows),
        "peak_r": np.arange(rows, dtype=float),
    }
    result = assess(arrays, context_bars=256)
    assert result["strategies"]["supertrend__atr"]["events"] == 2
    assert result["strategies"]["supertrend__atr"]["independent_contexts_approx"] == 2
    pair = result["canonical_exact_overlap"][0]
    assert pair["left"] == "supertrend__atr"
    assert pair["exact_overlap"] == 1
    assert result["recommended_trigger_union_diagnostic"]["deduplicated_events"] == 4
