"""ATR-zigzag pivot-confirm tests — the SHARED pivot detection (used by trend_scan
AND any 3rd-party consumer, e.g. the live bot). Load-bearing guarantee: CAUSALITY
(a pivot is only confirmed after price retraces rev_atr*ATR) + determinism."""
import numpy as np

from futures_foundation.primitives.detection import (
    atr_zigzag_legs, detect_atr_zigzag_pivots)


def _bars(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.standard_normal(n) * 0.003))
    h = c * (1 + np.abs(rng.standard_normal(n)) * 0.002)
    l = c * (1 - np.abs(rng.standard_normal(n)) * 0.002)
    return c.copy(), h, l, c


# ---- CAUSALITY: future bars can't change already-confirmed pivots ----------
def test_zigzag_pivots_causal():
    o, h, l, c = _bars()
    p1 = detect_atr_zigzag_pivots(o, h, l, c)
    k = 2000
    o2, h2, l2, c2 = o.copy(), h.copy(), l.copy(), c.copy()
    for a in (o2, h2, l2, c2):
        a[k:] *= 1.1
    p2 = detect_atr_zigzag_pivots(o2, h2, l2, c2)
    a = [p for p in p1 if p['confirm'] < k - 200]
    b = [p for p in p2 if p['confirm'] < k - 200]
    assert a == b and len(a) > 20


# ---- structure: valid, causal, tradeable pivots ---------------------------
def test_zigzag_pivots_structure():
    o, h, l, c = _bars()
    piv = detect_atr_zigzag_pivots(o, h, l, c)
    assert len(piv) > 50
    for p in piv:
        assert p['direction'] in (1, -1)
        assert p['confirm'] > p['origin']        # confirmed AFTER the origin pivot
        assert p['confirm'] + 1 < len(c)         # a causal entry (confirm+1) exists
        assert p['R'] >= 0
        assert isinstance(p['is_trend'], bool)


# ---- determinism: same bars -> same legs ----------------------------------
def test_zigzag_legs_deterministic():
    o, h, l, c = _bars(seed=2)
    atr = np.full(len(c), 0.5)
    a = atr_zigzag_legs(o, h, l, c, atr, 1.25, 0.25)
    b = atr_zigzag_legs(o, h, l, c, atr, 1.25, 0.25)
    assert a == b and len(a) > 50
