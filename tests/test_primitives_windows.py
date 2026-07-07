"""Multi-scale OHLCV window primitives (primitives/windows.py) — the multi-TF embedding input.

Locks: factor=1 is the raw window (byte-identical baseline), aggregation matches native resample
semantics (O=first/H=max/L=min/C=last/V=sum), the window is CAUSAL and anchored (last candle ends
exactly at bar i — future bars can never leak in), insufficient history fails fast, and the
multi-scale stack has the [5*nF, seq] channel layout the channel-independent encoder expects.
"""
import numpy as np
import pytest

from futures_foundation.primitives import aggregate_ohlcv_window, multi_scale_ohlcv_window


def _bars(n, seed=0):
    """Valid random-walk OHLCV arrays (H >= max(O,C), L <= min(O,C), V > 0)."""
    rng = np.random.default_rng(seed)
    c = 100 + np.cumsum(rng.normal(0, 1, n))
    o = np.roll(c, 1); o[0] = 100.0
    h = np.maximum(o, c) + rng.uniform(0, 0.5, n)
    l = np.minimum(o, c) - rng.uniform(0, 0.5, n)
    v = rng.uniform(1, 100, n)
    return o, h, l, c, v


def test_factor_one_is_identity():
    o, h, l, c, v = _bars(100)
    i, seq = 99, 64
    w = aggregate_ohlcv_window(o, h, l, c, v, i, seq, factor=1)
    assert w.shape == (5, seq) and w.dtype == np.float32
    sl = slice(i - seq + 1, i + 1)
    assert np.allclose(w, np.stack([o[sl], h[sl], l[sl], c[sl], v[sl]]).astype(np.float32))


def test_aggregation_matches_native_resample_semantics():
    o, h, l, c, v = _bars(400)
    i, seq, f = 399, 64, 5
    w = aggregate_ohlcv_window(o, h, l, c, v, i, seq, factor=f)
    assert w.shape == (5, seq)
    # spot-check every block against the definition (O=first, H=max, L=min, C=last, V=sum)
    start = i - seq * f + 1
    for b in range(seq):
        blk = slice(start + b * f, start + (b + 1) * f)
        assert w[0, b] == np.float32(o[blk][0])          # open  = first
        assert w[1, b] == np.float32(h[blk].max())       # high  = max
        assert w[2, b] == np.float32(l[blk].min())       # low   = min
        assert w[3, b] == np.float32(c[blk][-1])         # close = last
        assert np.isclose(w[4, b], v[blk].sum(), rtol=1e-6)   # volume = sum


def test_anchored_and_causal():
    # the LAST aggregated candle must END exactly at bar i, and bars AFTER i must never matter
    o, h, l, c, v = _bars(400, seed=1)
    i, seq, f = 350, 64, 5
    w = aggregate_ohlcv_window(o, h, l, c, v, i, seq, factor=f)
    assert w[3, -1] == np.float32(c[i])                  # anchored: final close == close at i
    o2, h2, l2, c2, v2 = (a.copy() for a in (o, h, l, c, v))
    o2[i + 1:] = 9e9; h2[i + 1:] = 9e9; l2[i + 1:] = -9e9; c2[i + 1:] = 9e9; v2[i + 1:] = 9e9
    w2 = aggregate_ohlcv_window(o2, h2, l2, c2, v2, i, seq, factor=f)
    assert np.array_equal(w, w2)                         # future bars cannot leak in


def test_aggregated_candles_stay_valid_ohlc():
    o, h, l, c, v = _bars(400, seed=2)
    w = aggregate_ohlcv_window(o, h, l, c, v, 399, 64, factor=5)
    assert (w[1] >= np.maximum(w[0], w[3]) - 1e-6).all()  # H >= max(O, C)
    assert (w[2] <= np.minimum(w[0], w[3]) + 1e-6).all()  # L <= min(O, C)
    assert (w[4] > 0).all()


def test_insufficient_history_raises():
    o, h, l, c, v = _bars(100)
    aggregate_ohlcv_window(o, h, l, c, v, i=63, seq=64, factor=1)        # exactly enough: 64 bars, OK
    with pytest.raises(ValueError):
        aggregate_ohlcv_window(o, h, l, c, v, i=62, seq=64, factor=1)    # one bar short
    with pytest.raises(ValueError):
        aggregate_ohlcv_window(o, h, l, c, v, i=99, seq=64, factor=5)    # needs 320, has 100


def test_multi_scale_stack_layout():
    # [5*nF, seq]: first 5 channels = the raw window, next 5 = the factor-5 window
    o, h, l, c, v = _bars(400, seed=3)
    i, seq = 399, 64
    m = multi_scale_ohlcv_window(o, h, l, c, v, i, seq, factors=(1, 5))
    assert m.shape == (10, seq)
    assert np.array_equal(m[:5], aggregate_ohlcv_window(o, h, l, c, v, i, seq, 1))
    assert np.array_equal(m[5:], aggregate_ohlcv_window(o, h, l, c, v, i, seq, 5))
    assert np.array_equal(multi_scale_ohlcv_window(o, h, l, c, v, i, seq, (1,)),
                          aggregate_ohlcv_window(o, h, l, c, v, i, seq, 1))   # (1,) == plain


def test_nan_bars_are_zeroed():
    o, h, l, c, v = _bars(400, seed=4)
    h[380] = np.nan
    w = aggregate_ohlcv_window(o, h, l, c, v, 399, 64, factor=5)
    assert np.isfinite(w).all()
