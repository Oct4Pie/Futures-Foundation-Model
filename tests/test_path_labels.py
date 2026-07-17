import numpy as np
import pandas as pd

from futures_foundation.finetune.path_labels import (
    BARRIER_ADVERSE_FIRST,
    BARRIER_AMBIGUOUS,
    BARRIER_FAVORABLE_FIRST,
    BARRIER_NEITHER,
    INVALID_CADENCE,
    INVALID_CONTRACT_ROLL,
    PathLabelConfig,
    TREND_CONTINUATION,
    TREND_REVERSAL,
    build_dense_path_labels,
    path_label_fingerprint,
)


def _frame(close, *, freq="1min", contract=None, high=None, low=None):
    close = np.asarray(close, dtype=float)
    n = len(close)
    ts = pd.date_range("2024-01-02 00:00", periods=n, freq=freq, tz="UTC")
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) if high is None else np.asarray(high, float)
    low = np.minimum(open_, close) if low is None else np.asarray(low, float)
    return pd.DataFrame({
        "datetime": ts,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "contract_id": contract if contract is not None else ["ESH4"] * n,
    })


def _cfg(**kwargs):
    values = dict(
        horizons_minutes=(2,), targets_r=(1.0, 2.0), adverse_r=1.0,
        atr_period=2, context_minutes=2, context_deadband_r=0.1,
        barrier_chunk_rows=4,
    )
    values.update(kwargs)
    return PathLabelConfig(**values)


def test_monotone_paths_and_trend_classes():
    up = _frame([100, 101, 102, 103, 104, 105])
    up_labels = build_dense_path_labels(
        up, timeframe_minutes=1, config=_cfg(), causal_scale=np.ones(len(up)),
        context_direction=np.ones(len(up), np.int8),
    )
    assert up_labels["valid"][0, 0]
    np.testing.assert_allclose(up_labels["terminal_move_r"][0, 0], 2.0)
    np.testing.assert_allclose(up_labels["upside_mfe_r"][0, 0], 2.0)
    np.testing.assert_allclose(up_labels["downside_mae_r"][0, 0], 0.0)
    assert up_labels["trend_path_class"][0, 0] == TREND_CONTINUATION

    down = _frame([100, 99, 98, 97, 96, 95])
    down_labels = build_dense_path_labels(
        down, timeframe_minutes=1, config=_cfg(), causal_scale=np.ones(len(down)),
        context_direction=np.ones(len(down), np.int8),
    )
    assert down_labels["trend_path_class"][0, 0] == TREND_REVERSAL


def test_chop_and_volatility_expansion_targets():
    quiet = _frame([100.0] * 10)
    volatile = _frame([100, 104, 96, 105, 95, 106, 94, 107, 93, 108])
    cfg = _cfg(horizons_minutes=(6,))
    quiet_labels = build_dense_path_labels(
        quiet, timeframe_minutes=1, config=cfg, causal_scale=np.ones(len(quiet)),
    )
    volatile_labels = build_dense_path_labels(
        volatile, timeframe_minutes=1, config=cfg, causal_scale=np.ones(len(volatile)),
    )
    assert quiet_labels["forward_realized_vol"][0, 0] == 0.0
    assert quiet_labels["forward_trend_eff"][0, 0] == 0.0
    assert volatile_labels["forward_realized_vol"][0, 0] > 0.05
    assert volatile_labels["forward_trend_eff"][0, 0] < 0.2


def test_same_bar_ambiguity_is_preserved_but_policy_is_adverse_first():
    frame = _frame(
        [100, 100, 100, 100],
        high=[100, 102.5, 100, 100], low=[100, 98.5, 100, 100],
    )
    labels = build_dense_path_labels(
        frame, timeframe_minutes=1, config=_cfg(), causal_scale=np.ones(len(frame)),
    )
    long = 0
    assert labels["barrier_state"][0, 0, long, 0] == BARRIER_AMBIGUOUS
    assert labels["barrier_state"][0, 0, long, 1] == BARRIER_AMBIGUOUS
    np.testing.assert_allclose(labels["policy_r_gross"][0, 0, long], [-1.0, -1.0])
    assert labels["time_to_favorable_minutes"][0, 0, long, 0] == 1
    assert labels["time_to_adverse_minutes"][0, 0, long, 0] == 1


def test_barrier_first_touch_and_neither_states():
    frame = _frame(
        [100, 100, 100, 100, 100],
        high=[100, 101.2, 102.2, 100, 100], low=[100, 99.5, 98.5, 100, 100],
    )
    labels = build_dense_path_labels(
        frame, timeframe_minutes=1, config=_cfg(), causal_scale=np.ones(len(frame)),
    )
    long, short = 0, 1
    assert labels["barrier_state"][0, 0, long, 0] == BARRIER_FAVORABLE_FIRST
    assert labels["barrier_state"][0, 0, long, 1] == BARRIER_AMBIGUOUS
    assert labels["barrier_state"][0, 0, short, 0] == BARRIER_ADVERSE_FIRST
    assert labels["barrier_state"][2, 0, long, 0] == BARRIER_NEITHER


def test_roll_and_cadence_crossings_are_masked_not_truncated():
    roll = _frame([100, 101, 102, 103, 104], contract=["ESH4", "ESH4", "ESM4", "ESM4", "ESM4"])
    roll_labels = build_dense_path_labels(
        roll, timeframe_minutes=1, config=_cfg(), causal_scale=np.ones(len(roll)),
    )
    assert not roll_labels["valid"][0, 0]
    assert roll_labels["invalid_reason"][0, 0] == INVALID_CONTRACT_ROLL

    gap = _frame([100, 101, 102, 103, 104])
    gap.loc[2:, "datetime"] += pd.Timedelta(minutes=1)
    gap_labels = build_dense_path_labels(
        gap, timeframe_minutes=1, config=_cfg(), causal_scale=np.ones(len(gap)),
    )
    assert not gap_labels["valid"][0, 0]
    assert gap_labels["invalid_reason"][0, 0] == INVALID_CADENCE

    missing = _frame([100, 101, 102, 103, 104])
    missing.loc[1, ["open", "high", "low", "close"]] = np.nan
    missing_labels = build_dense_path_labels(
        missing, timeframe_minutes=1, config=_cfg(), causal_scale=np.ones(len(missing)),
    )
    assert not missing_labels["valid"][0, 0]


def test_prefix_invariance_and_causal_scale():
    prefix = _frame(np.linspace(100, 120, 20))
    full = _frame(np.linspace(100, 130, 30))
    # Preserve the exact prefix, then append an extreme future that must not revise completed rows.
    full.loc[:19, ["open", "high", "low", "close"]] = prefix[["open", "high", "low", "close"]].to_numpy()
    full.loc[20:, "high"] += 1000
    prefix_labels = build_dense_path_labels(prefix, timeframe_minutes=1, config=_cfg())
    full_labels = build_dense_path_labels(full, timeframe_minutes=1, config=_cfg())
    complete = len(prefix) - 2
    for name in (
        "terminal_log_return", "terminal_move_r", "forward_realized_vol", "upside_mfe_r",
        "downside_mae_r", "forward_trend_eff", "causal_scale",
    ):
        a = prefix_labels[name][:complete]
        b = full_labels[name][:complete]
        np.testing.assert_allclose(a, b, equal_nan=True)
    np.testing.assert_array_equal(
        prefix_labels["barrier_state"][:complete], full_labels["barrier_state"][:complete]
    )


def test_elapsed_horizon_and_fingerprint_are_deterministic():
    one_minute = _frame(np.linspace(100, 107, 8))
    one_cfg = _cfg(horizons_minutes=(6,), context_minutes=2)
    one = build_dense_path_labels(
        one_minute, timeframe_minutes=1, config=one_cfg, causal_scale=np.ones(len(one_minute)),
    )
    sixty_minute = _frame(np.linspace(100, 107, 8), freq="60min")
    sixty_cfg = _cfg(horizons_minutes=(360,), context_minutes=120)
    sixty = build_dense_path_labels(
        sixty_minute, timeframe_minutes=60, config=sixty_cfg,
        causal_scale=np.ones(len(sixty_minute)),
    )
    assert one["label_end_time_ns"][0, 0] - one["decision_time_ns"][0] == 6 * 60 * 1_000_000_000
    assert sixty["label_end_time_ns"][0, 0] - sixty["decision_time_ns"][0] == 360 * 60 * 1_000_000_000
    assert path_label_fingerprint(one) == path_label_fingerprint(one)
    changed_frame = one_minute.copy()
    changed_frame.loc[7, ["high", "close"]] += 1.0
    changed = build_dense_path_labels(
        changed_frame,
        timeframe_minutes=1, config=one_cfg, causal_scale=np.ones(len(one_minute)),
    )
    assert path_label_fingerprint(one) != path_label_fingerprint(changed)


def test_all_declared_timeframes_use_exact_elapsed_horizons():
    for timeframe in (1, 3, 5, 15, 30, 60):
        frame = _frame(np.linspace(100, 120, 400), freq=f"{timeframe}min")
        labels = build_dense_path_labels(
            frame,
            timeframe_minutes=timeframe,
            config=PathLabelConfig(
                horizons_minutes=(60, 180, 360), context_minutes=60,
                targets_r=(1.0,), barrier_chunk_rows=32,
            ),
            causal_scale=np.ones(len(frame)),
        )
        elapsed = labels["label_end_time_ns"][0] - labels["decision_time_ns"][0]
        np.testing.assert_array_equal(
            elapsed,
            np.asarray((60, 180, 360), np.int64) * 60 * 1_000_000_000,
        )
        assert not labels["valid"][-1].any()
