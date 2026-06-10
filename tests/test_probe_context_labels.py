"""Tests — context-head label generators (scripts/probe_context_heads.py).

The Step-2 label test spec: known-answer synthetic series per head,
causality (changing FUTURE bars changes labels, not past-dependent values;
changing PAST bars must not change forward-looking targets' future inputs),
and tail-NaN handling (forward windows never filled).

These generators are the probe's ground truth and get promoted into
futures_foundation in Phase 2 — they earn real tests now.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_spec = importlib.util.spec_from_file_location(
    'probe_context_heads',
    Path(__file__).resolve().parents[1] / 'scripts' / 'probe_context_heads.py')
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)

N = 600
RNG = np.random.default_rng(7)


def _series(values):
    return pd.Series(np.asarray(values, dtype=float))


@pytest.fixture
def random_close():
    return _series(100 * np.exp(np.cumsum(RNG.normal(0, 0.001, N))))


def test_all_heads_present_and_tail_nan(random_close):
    lab = probe.compute_labels(random_close)
    # every probed head has a label column (order is not a contract)
    assert set(lab.columns) == {name for name, _ in probe.HEADS}
    # forward-looking: the last `horizon` rows must be NaN, never filled
    assert lab['fwd_return'].iloc[-20:].isna().all()
    assert lab['vol_expansion'].iloc[-20:].isna().all()
    assert lab['volatility'].iloc[-10:].isna().all()
    assert lab['structure'].iloc[-20:].isna().all()
    assert lab['range_bound'].iloc[-10:].isna().all()
    # head of series (trailing windows unavailable) also NaN
    assert lab['fwd_return'].iloc[:49].isna().all()      # needs 50 obs of sigma
    # and there IS a populated middle
    mid = lab.iloc[250:350]
    assert mid['fwd_return'].notna().all()
    assert mid['volatility'].notna().all()


def test_fwd_return_known_answer():
    """Steady exponential uptrend -> strongly positive z, clipped at +4."""
    close = _series(100 * np.exp(0.001 * np.arange(N)))
    z = probe.compute_labels(close)['fwd_return']
    # constant log-growth: 20-bar fwd return is constant, trailing std ~ 0
    # -> z hits the +4 clip wherever defined
    assert (z.dropna() > 0).all()
    assert z.dropna().max() <= 4.0 + 1e-9
    down = probe.compute_labels(
        _series(100 * np.exp(-0.001 * np.arange(N))))['fwd_return']
    assert (down.dropna() < 0).all()
    assert down.dropna().min() >= -4.0 - 1e-9


def test_vol_expansion_known_answer():
    """Quiet regime then a volatility burst: bars whose forward window
    enters the burst must label 1; deep-quiet bars 0."""
    quiet = RNG.normal(0, 0.0002, 500)
    burst = RNG.normal(0, 0.005, 100)
    close = _series(100 * np.exp(np.cumsum(np.r_[quiet, burst])))
    ve = probe.compute_labels(close)['vol_expansion']
    assert (ve.iloc[300:450].dropna() == 0).all()        # quiet stays quiet
    assert (ve.iloc[490:520].dropna() == 1).all()        # fwd window in burst


def test_structure_known_answer():
    """Monotonic rise -> structure 1 (bull); monotonic fall -> 0 (bear);
    a flat series -> mixed/NaN (neither leg extends)."""
    up = probe.compute_labels(_series(np.arange(100, 100 + N)))['structure']
    assert (up.dropna() == 1.0).all()
    dn = probe.compute_labels(_series(np.arange(100 + N, 100, -1)))['structure']
    assert (dn.dropna() == 0.0).all()
    flat = probe.compute_labels(_series(np.full(N, 100.0)))['structure']
    assert flat.isna().all()                              # mixed = sentinel


def test_volatility_percentile_bounds_and_response(random_close):
    v = probe.compute_labels(random_close)['volatility'].dropna()
    assert ((v >= 0) & (v <= 1)).all()
    # forward vol burst -> percentile near 1 for bars just before it
    quiet = RNG.normal(0, 0.0002, 500)
    burst = RNG.normal(0, 0.01, 100)
    close = _series(100 * np.exp(np.cumsum(np.r_[quiet, burst])))
    vol = probe.compute_labels(close)['volatility']
    assert vol.iloc[495:505].dropna().min() > 0.9


def test_labels_are_forward_looking_causality():
    """Changing ONLY future bars (t >= 400) must change labels at t < 400
    whose forward window reaches the change, and must NOT change labels
    whose forward window ends before it (t <= 400 - 21)."""
    base = 100 * np.exp(np.cumsum(RNG.normal(0, 0.001, N)))
    a = probe.compute_labels(_series(base))
    mod = base.copy()
    mod[400:] = mod[400:] * np.exp(0.01 * np.arange(N - 400))  # future shock
    b = probe.compute_labels(_series(mod))
    # labels with forward window fully BEFORE the change: identical
    pd.testing.assert_frame_equal(a.iloc[250:370], b.iloc[250:370])
    # labels whose forward window crosses the change: must differ somewhere
    cross = slice(390, 400)
    assert not a['fwd_return'].iloc[cross].equals(b['fwd_return'].iloc[cross])


def test_trivial_features_are_strictly_trailing():
    """Trivial baseline features at t must not depend on bars > t."""
    base = 100 * np.exp(np.cumsum(RNG.normal(0, 0.001, N)))
    a = probe.trivial_features(_series(base))
    mod = base.copy()
    mod[400:] *= 1.5
    b = probe.trivial_features(_series(mod))
    pd.testing.assert_frame_equal(a.iloc[:400], b.iloc[:400])


# ---------------------------------------------------------------------------
# Trend/range/chop heads (promoted 2026-06-10, FF68 arm) — known answers +
# conditionality
# ---------------------------------------------------------------------------

def test_trendiness_known_answer():
    """Monotonic line -> efficiency ratio ~1; alternating chop -> ~0."""
    trend = probe.compute_labels(_series(np.arange(100, 100 + N)))
    assert (trend['trendiness'].dropna() > 0.95).all()
    osc = 100 + 0.5 * (np.arange(N) % 2)          # +/- alternating closes
    chop = probe.compute_labels(_series(osc))
    assert (chop['trendiness'].dropna() < 0.10).all()


def test_range_bound_known_answer():
    """Oscillation inside a fixed band -> 1; strong trend -> 0."""
    osc = 100 + 0.5 * (np.arange(N) % 2)
    rb = probe.compute_labels(_series(osc))['range_bound']
    assert (rb.dropna() == 1.0).all()
    trend = probe.compute_labels(_series(np.arange(100, 100 + N)))
    assert (trend['range_bound'].dropna() == 0.0).all()


def test_quiet_persist_conditional_and_known_answer():
    """Defined ONLY on currently-quiet bars; 1 when quiet persists,
    0 just before a vol burst."""
    quiet = RNG.normal(0, 0.0002, 500)
    burst = RNG.normal(0, 0.01, 100)
    close = _series(100 * np.exp(np.cumsum(np.r_[quiet, burst])))
    qp = probe.compute_labels(close)['quiet_persist']
    # uniform quiet: overwhelmingly 1 (the label is statistical — rare
    # sampling-noise flips are inherent; discrimination is the probe's job)
    assert qp.iloc[300:450].dropna().mean() > 0.85
    # any bar defined just before the burst (fwd window inside it) must be 0
    pre_burst = qp.iloc[478:500].dropna()
    assert (pre_burst == 0.0).all()
    assert qp.iloc[530:560].isna().all()               # not quiet -> NaN
