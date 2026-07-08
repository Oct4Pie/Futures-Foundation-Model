"""Williams-fractal pivots (detect_fractal_pivots) — the time-confirmation trigger variant.

Contract: same schema as detect_atr_zigzag_pivots (drop-in), strictly causal (confirm = extreme+k,
entry bar exists), extremes unique in their 2k+1 window, byte-exact detection.
"""
import numpy as np

from futures_foundation.primitives.detection import (
    detect_atr_zigzag_pivots, detect_fractal_pivots)


def _v(center=30, n=64, depth=5.0):
    """A clean V: strictly falling into `center`, strictly rising after — one unambiguous low."""
    t = np.arange(n, dtype=float)
    c = 100 + np.abs(t - center) * (depth / center)
    h = c + 0.2
    l = c - 0.2
    return h, l


def test_finds_the_v_bottom():
    h, l = _v(center=30)
    piv = detect_fractal_pivots(h, l, k=2)
    lows = [p for p in piv if p['direction'] == 1]
    assert any(p['origin'] == 30 for p in lows)            # THE bottom is detected
    p = next(p for p in lows if p['origin'] == 30)
    assert p['confirm'] == 32                              # confirm = extreme + k
    assert p['leg_end'] == 30


def test_schema_matches_atr_zigzag():
    # independent wick noise — derived h/l from rolled closes creates systematic TIES, which the
    # uniqueness rule correctly rejects (a tied extreme is not a fractal)
    rng = np.random.default_rng(0)
    c = 100 + rng.normal(0, 1, 400).cumsum()
    o = np.roll(c, 1); o[0] = c[0]
    h = np.maximum(o, c) + np.abs(rng.normal(0, 0.3, 400))
    l = np.minimum(o, c) - np.abs(rng.normal(0, 0.3, 400))
    zz = detect_atr_zigzag_pivots(o, h, l, c)
    fr = detect_fractal_pivots(h, l, k=2)
    assert len(fr) > 0 and len(zz) > 0
    assert set(fr[0]) == set(zz[0])                        # identical keys = drop-in trigger


def test_causal_confirm_and_unique_extreme():
    rng = np.random.default_rng(1)
    c = 100 + rng.normal(0, 1, 500).cumsum()
    h = c + np.abs(rng.normal(0, 0.3, 500))
    l = c - np.abs(rng.normal(0, 0.3, 500))
    n = len(c)
    for p in detect_fractal_pivots(h, l, k=3):
        i, cf = p['origin'], p['confirm']
        assert cf == i + 3 and cf + 1 < n                  # time confirmation + causal entry exists
        if p['direction'] == 1:
            seg = l[i - 3:i + 4]
            assert l[i] == seg.min() and (seg == l[i]).sum() == 1   # unique lowest low
        else:
            seg = h[i - 3:i + 4]
            assert h[i] == seg.max() and (seg == h[i]).sum() == 1


def test_flat_series_has_no_pivots():
    z = np.full(100, 100.0)
    assert detect_fractal_pivots(z + 0.1, z - 0.1, k=2) == []   # ties are never pivots


def test_min_bars_apart_suppresses_clusters():
    rng = np.random.default_rng(2)
    c = 100 + rng.normal(0, 1, 600).cumsum()
    h = c + np.abs(rng.normal(0, 0.3, 600))
    l = c - np.abs(rng.normal(0, 0.3, 600))
    dense = detect_fractal_pivots(h, l, k=2)
    sparse = detect_fractal_pivots(h, l, k=2, min_bars_apart=10)
    assert len(sparse) < len(dense)
    for d in (1, -1):
        idx = [p['origin'] for p in sparse if p['direction'] == d]
        assert all(b - a >= 10 for a, b in zip(idx, idx[1:]))


def test_deterministic():
    rng = np.random.default_rng(3)
    c = 100 + rng.normal(0, 1, 300).cumsum()
    h, l = c + 0.3, c - 0.3
    assert detect_fractal_pivots(h, l, k=2) == detect_fractal_pivots(h, l, k=2)


# ---------------------------------------------------------------- fractal-zigzag hybrid
from futures_foundation.primitives.detection import detect_fractal_zigzag_pivots


def _rw(n=2000, seed=5):
    rng = np.random.default_rng(seed)
    c = 100 + rng.normal(0, 1, n).cumsum()
    o = np.roll(c, 1); o[0] = c[0]
    h = np.maximum(o, c) + np.abs(rng.normal(0, 0.3, n))
    l = np.minimum(o, c) - np.abs(rng.normal(0, 0.3, n))
    return o, h, l, c


def test_hybrid_alternates_and_respects_leg_rule():
    from futures_foundation.pipeline._primitives import compute_atr
    o, h, l, c = _rw()
    piv = detect_fractal_zigzag_pivots(o, h, l, c, k=2, min_leg_atr=1.25)
    assert len(piv) > 3
    dirs = [p['direction'] for p in piv]
    assert all(a != b for a, b in zip(dirs, dirs[1:]))     # strict alternation
    atr = compute_atr(h, l, c, 20)
    for a, b in zip(piv, piv[1:]):
        pa = l[a['origin']] if a['direction'] == 1 else h[a['origin']]
        pb = l[b['origin']] if b['direction'] == 1 else h[b['origin']]
        assert abs(pb - pa) >= 1.25 * atr[b['origin']] - 1e-9   # significance rule


def test_hybrid_keep_first_no_hindsight():
    # two same-direction fractals, second deeper: the FIRST must be kept (live-tradeable rule);
    # swapping in the deeper one is the measured ~9-WR-pt lookahead.
    o, h, l, c = _rw(seed=6)
    piv = detect_fractal_zigzag_pivots(o, h, l, c, k=2, min_leg_atr=1.0)
    from futures_foundation.primitives.detection import detect_fractal_pivots
    raw = detect_fractal_pivots(h, l, k=2)
    kept = {p['origin'] for p in piv}
    # walk the causal state machine independently and confirm identical keeps
    from futures_foundation.pipeline._primitives import compute_atr
    atr = compute_atr(h, l, c, 20)
    last_d, last_px, expect = 0, None, set()
    for p in raw:
        i, d = p['origin'], p['direction']
        px = l[i] if d == 1 else h[i]
        if not (np.isfinite(atr[i]) and atr[i] > 0) or d == last_d:
            continue
        if last_px is not None and abs(px - last_px) < 1.0 * atr[i]:
            continue
        expect.add(i); last_d, last_px = d, px
    assert kept == expect


def test_hybrid_stricter_leg_fewer_pivots():
    o, h, l, c = _rw(seed=7)
    loose = detect_fractal_zigzag_pivots(o, h, l, c, k=2, min_leg_atr=0.5)
    tight = detect_fractal_zigzag_pivots(o, h, l, c, k=2, min_leg_atr=2.5)
    assert len(tight) < len(loose)
