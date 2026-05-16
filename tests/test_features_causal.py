"""
Train/serve causality tests for features.py.

Root cause of the CRT Sweep look-ahead skew (CRT_LOOKAHEAD_BUGREPORT.md):
`derive_features` computed swing pivots with a centered window and
`sess_bars_elapsed` with the realized (future) session length. Batch
(training) saw hindsight values; live computed them causally → different
model inputs.

Acceptance criterion: for every column the per-bar value of
`derive_features(full_df)` must equal `derive_features(full_df[:i+1])`
(streaming / live path) at bar `i`. These tests pin that invariant for the
previously-offending columns and, as a guardrail, for the whole model
feature set.
"""
import importlib.util
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

_CP_PATH = Path(__file__).parent.parent / "futures_foundation" / "candle_psychology.py"
_cp_spec = importlib.util.spec_from_file_location("futures_foundation.candle_psychology", _CP_PATH)
_cp_mod = importlib.util.module_from_spec(_cp_spec)
sys.modules["futures_foundation.candle_psychology"] = _cp_mod
_cp_spec.loader.exec_module(_cp_mod)

_FEATURES_PATH = Path(__file__).parent.parent / "futures_foundation" / "features.py"
_spec = importlib.util.spec_from_file_location("ffm_features", _FEATURES_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

derive_features = _mod.derive_features
get_model_feature_columns = _mod.get_model_feature_columns

# The four columns called out in the bug report as look-ahead.
OFFENDING_COLS = [
    "str_swing_high_dist",
    "str_swing_low_dist",
    "str_structure_state",
    "sess_bars_elapsed",
]

_DATA = Path(__file__).parent.parent / "data" / "GC_3min.csv"


def _synthetic(n=900, seed=7, bar_freq_min=3):
    np.random.seed(seed)
    dates = pd.date_range("2024-01-02 00:00", periods=n, freq=f"{bar_freq_min}min")
    close = 2000 + np.cumsum(np.random.randn(n) * 1.2)
    df = pd.DataFrame({
        "datetime": dates,
        "open": close + np.random.randn(n),
        "high": close + np.abs(np.random.randn(n)) * 2,
        "low": close - np.abs(np.random.randn(n)) * 2,
        "close": close,
        "volume": np.random.randint(100, 5000, n).astype(float),
    })
    df["high"] = df[["open", "close", "high"]].max(axis=1)
    df["low"] = df[["open", "close", "low"]].min(axis=1)
    return df


def _assert_batch_equals_streaming(df, cols, sample_idx, tol=1e-6):
    """Core acceptance check: batch value at bar i == causal recompute on df[:i+1]."""
    batch = derive_features(df, "GC")
    for i in sample_idx:
        causal = derive_features(df.iloc[: i + 1].copy(), "GC")
        assert len(causal) == i + 1
        for col in cols:
            b = batch[col].iloc[i]
            c = causal[col].iloc[i]
            if pd.isna(b) and pd.isna(c):
                continue
            assert abs(float(b) - float(c)) <= tol, (
                f"LOOK-AHEAD SKEW in '{col}' at bar {i}: "
                f"batch={b!r} streaming={c!r}"
            )


# =============================================================================
# Acceptance criterion — the offending columns
# =============================================================================

def test_offending_cols_batch_equals_streaming_synthetic():
    df = _synthetic()
    # Sample across the series; stay clear of the warm-up region.
    idx = list(range(400, len(df), 73))
    _assert_batch_equals_streaming(df, OFFENDING_COLS, idx)


@pytest.mark.skipif(not _DATA.exists(), reason="GC_3min.csv not present")
def test_offending_cols_batch_equals_streaming_real_data():
    # Tail slice of real gold 3-min data — the exact instrument/timeframe class
    # the live CRT bot trades (MGC 3-min).
    df = pd.read_csv(_DATA).tail(1600).reset_index(drop=True)
    idx = list(range(800, len(df), 97))
    _assert_batch_equals_streaming(df, OFFENDING_COLS, idx)


# =============================================================================
# Guardrail — the WHOLE model feature set must be causal, on real data
# =============================================================================

@pytest.mark.skipif(not _DATA.exists(), reason="GC_3min.csv not present")
def test_all_model_features_batch_equals_streaming_real_data():
    df = pd.read_csv(_DATA).tail(1600).reset_index(drop=True)
    # htf_daily_structure needs >=1 prior trading day of history to be
    # populated; with full history up to bar i (the live path) it is causal
    # and matches batch (per the bug report's BATCH==CAUSALfull finding).
    cols = get_model_feature_columns()
    idx = list(range(900, len(df), 151))
    _assert_batch_equals_streaming(df, cols, idx)


# =============================================================================
# Regression: the centered-window leakage must be gone
# =============================================================================

def test_swings_do_not_use_future_bars():
    """
    Detect a single dominant pivot, then verify it is not 'known' before its
    confirmation bar. Pre-fix (center=True) the swing distance reacted to the
    pivot on the pivot bar itself; post-fix it can only react >= lookback bars
    later.
    """
    n, lookback = 300, 10
    dates = pd.date_range("2024-01-02 00:00", periods=n, freq="3min")
    close = np.full(n, 2000.0)
    spike = 150  # lone, unambiguous swing high
    close[spike] = 2050.0
    df = pd.DataFrame({
        "datetime": dates,
        "open": close, "high": close, "low": close,
        "close": close, "volume": np.full(n, 1000.0),
    })
    df["high"] = df["close"] + 0.5
    df["low"] = df["close"] - 0.5
    df.loc[spike, "high"] = 2051.0

    feats = derive_features(df, "GC", structure_lookback=lookback)
    shd = feats["str_swing_high_dist"]
    # On and just after the pivot (before confirmation) the new pivot must NOT
    # yet influence the swing-high distance.
    pre = shd.iloc[spike: spike + lookback].dropna()
    if len(pre):
        assert (pre.abs() > 1e-9).all() or pre.isna().all(), (
            "swing-high distance reacted to a pivot before it was confirmable "
            "(centered/future window leak)"
        )
