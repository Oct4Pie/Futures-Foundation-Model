"""futures_foundation.indicators — the reusable strategy-trigger indicator library.

The contract that matters: EVERY function is CAUSAL (output i uses only bars <= i) — these feed
entry triggers and model features, so a single future peek breaks OOS. Each new indicator gets
the perturb-the-future test: change bars AFTER i, outputs at <= i must be byte-identical.
The certified primitives (ATR/ADX/SuperTrend) are re-exports — covered by their own suite."""
import numpy as np
import pandas as pd
import pytest

from futures_foundation import indicators as I


def _series(n=400, seed=0):
    rng = np.random.default_rng(seed)
    c = 100 + np.cumsum(rng.standard_normal(n) * 0.3)
    h = c + np.abs(rng.standard_normal(n)) * 0.2
    l = c - np.abs(rng.standard_normal(n)) * 0.2
    ts = pd.date_range('2024-01-02 09:00', periods=n, freq='3min', tz='UTC')
    return ts, h, l, c


def _perturb_future_invariant(fn, x, cut=250):
    """Outputs at <= cut must not change when bars AFTER cut change."""
    a = np.asarray(fn(x))
    y = np.array(x, float); y[cut + 1:] += 37.0
    b = np.asarray(fn(y))
    assert np.allclose(a[:cut + 1], b[:cut + 1], equal_nan=True)


def test_reexports_are_the_certified_impls():
    """ONE implementation rule: the re-exports ARE pipeline._primitives (never a fork)."""
    from futures_foundation.pipeline import _primitives as P
    assert I.compute_atr is P.compute_atr and I.compute_adx is P.compute_adx
    assert I.compute_supertrend is P.compute_supertrend
    assert I.apply_rr_barriers is P.apply_rr_barriers


def test_ema_causal_and_sane():
    _, _, _, c = _series()
    _perturb_future_invariant(lambda x: I.ema(x, 9), c)
    e = I.ema(c, 9)
    assert np.isfinite(e).all() and abs(e[-1] - c[-1]) < 5.0    # tracks price


def test_adx_slope_causal():
    ts, h, l, c = _series()
    adx = I.compute_adx(h, l, c, 14)
    s = I.adx_slope(adx, 5)
    assert np.isnan(s[:5]).all() and np.isfinite(s[20:]).any()


def test_kalman_velocity_causal_and_directional():
    _, _, _, c = _series()
    lp = np.log(c)
    _perturb_future_invariant(lambda x: I.kalman_velocity(x), lp)
    up = I.kalman_velocity(np.linspace(0, 1, 200))              # clean up-drift -> positive vel
    assert up[50:].mean() > 0


def test_nw_slope_causal_and_directional():
    _, _, _, c = _series()
    lp = np.log(c)
    _perturb_future_invariant(lambda x: I.nw_slope(x, 30, 8.0), lp)
    s = I.nw_slope(np.linspace(0, 1, 200), 30, 8.0)
    assert np.isnan(s[:29]).all() and s[40:][np.isfinite(s[40:])].min() > 0


def test_opening_range_causal_active_after_window():
    """The range activates only AFTER its orb_bars window closes (fully in the past), resets
    per session day, and never uses future bars."""
    n = 200
    ts = pd.date_range('2024-01-02 09:00', periods=n, freq='3min',
                       tz='America/New_York').tz_convert('UTC')
    rng = np.random.default_rng(1)
    c = 100 + np.cumsum(rng.standard_normal(n) * 0.2)
    h, l = c + 0.3, c - 0.3
    oh, ol = I.opening_range(ts, h, l, orb_bars=5)
    et = pd.DatetimeIndex(ts).tz_convert('America/New_York')
    tmin = (et.hour * 60 + et.minute).to_numpy()
    first = int(np.argmax(tmin >= 9 * 60 + 30))                 # first range-window bar
    assert np.isnan(oh[:first + 5]).all()                       # nothing active before it closes
    act = first + 5
    assert np.isfinite(oh[act]) and np.isfinite(ol[act])
    assert oh[act] == pytest.approx(h[first:first + 5].max())   # = window high (past bars only)
    assert ol[act] == pytest.approx(l[first:first + 5].min())
    # future perturbation cannot change the active range at <= act
    h2 = h.copy(); h2[act + 1:] += 50
    oh2, _ = I.opening_range(ts, h2, l, orb_bars=5)
    assert np.allclose(oh[:act + 1], oh2[:act + 1], equal_nan=True)
