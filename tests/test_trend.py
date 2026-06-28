"""Trend-primitive tests — the load-bearing guarantee is CAUSALITY (these feed the
regime HMM and models; one future peek breaks OOS), plus correctness of the
vectorized efficiency ratio and Kalman direction tracking."""
import numpy as np
import pandas as pd

from futures_foundation import trend as T


def _walk(n=2000, seed=0, drift=0.0):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.standard_normal(n) * 0.002 + drift) + np.log(100)


# ---- CAUSALITY: perturb the future, past rows MUST NOT change --------------
def test_trend_primitives_causal():
    lp = _walk()
    k = 1500
    for fn in (lambda a: T.kalman_velocity(a)[1],
               lambda a: T.efficiency_ratio(a, 20),
               lambda a: T.decycler_slope(a, 60, 5)):
        x1 = fn(lp)
        a2 = lp.copy()
        a2[k:] += 0.5
        x2 = fn(a2)
        assert np.allclose(np.nan_to_num(x1[:k - 100]), np.nan_to_num(x2[:k - 100]))


# ---- efficiency ratio: vectorized == reference loop, range [0,1] -----------
def test_efficiency_ratio_correct():
    lp = _walk(500, seed=3)
    w = 20
    er = T.efficiency_ratio(lp, w)
    assert np.isnan(er[:w]).all()
    fin = er[w:]
    assert (fin >= -1e-9).all() and (fin <= 1 + 1e-9).all()
    for i in range(w, len(lp)):
        net = abs(lp[i] - lp[i - w])
        denom = np.abs(np.diff(lp[i - w:i + 1])).sum()
        ref = net / denom if denom > 1e-12 else 0.0
        assert abs(er[i] - ref) < 1e-9


# ---- efficiency ratio: clean trend ≈ 1, chop ≈ 0 --------------------------
def test_efficiency_ratio_trend_vs_chop():
    trend = np.linspace(np.log(100), np.log(110), 500)
    chop = np.log(100) + 0.001 * np.array([(-1) ** i for i in range(500)], float)
    assert np.nanmean(T.efficiency_ratio(trend, 20)) > 0.95
    assert np.nanmean(T.efficiency_ratio(chop, 20)) < 0.2


# ---- kalman velocity: sign tracks trend direction -------------------------
def test_kalman_velocity_direction():
    up = np.linspace(np.log(100), np.log(110), 1000)
    dn = np.linspace(np.log(110), np.log(100), 1000)
    _, vu = T.kalman_velocity(up)
    _, vd = T.kalman_velocity(dn)
    assert vu[200:].mean() > 0 and vd[200:].mean() < 0


# ---- TRIGGER GATE: truth table (positive / negative / edge) ---------------
def test_gate_truth_table():
    """The trigger rule: aligned = (sign(htf) == sign(dir)) AND dir != 0.
    POSITIVE = same nonzero sign; NEGATIVE = opposite; EDGE = htf==0 or dir==0."""
    htf = np.array([1.,  1., -1., -1.,  0.,  0.,  1., -1.,  0.])
    dr = np.array([1., -1., -1.,  1.,  1., -1.,  0.,  0.,  0.])
    exp = np.array([1,   0,   1,   0,   0,   0,   0,   0,   0], bool)
    got = (np.sign(htf) == np.sign(dr)) & (dr != 0)
    assert np.array_equal(got, exp)


def _swing_bars(n=4000, seed=0):
    ts = pd.date_range('2024-01-01', periods=n, freq='1min', tz='UTC').values
    rng = np.random.default_rng(seed)
    c = 100 + 8 * np.sin(np.arange(n) / 250.0) + rng.standard_normal(n) * 0.05
    return dict(ts=ts, o=c, h=c + 0.1, l=c - 0.1, c=c), rng


# ---- trend_aligned: positive / negative / edge (end-to-end, derived htf) ---
def test_trend_aligned_positive_negative_edge():
    from futures_foundation.pivots import causal_htf_dir
    bars, rng = _swing_bars()
    n = len(bars['c'])
    htf = np.sign(causal_htf_dir(bars, '1min', bars['ts'], 20))
    assert (htf == 1).any() and (htf == -1).any() and (htf == 0).any()   # all 3 regions present

    al_long = T.trend_aligned(bars, np.ones(n, int), '1min')
    al_short = T.trend_aligned(bars, -np.ones(n, int), '1min')
    # POSITIVE: long aligned EXACTLY where htf==+1; short EXACTLY where htf==-1
    assert np.array_equal(al_long, htf == 1) and al_long.any()
    assert np.array_equal(al_short, htf == -1) and al_short.any()
    # NEGATIVE: never aligned against the trend (or when flat)
    assert not al_long[htf != 1].any()
    assert not al_short[htf != -1].any()
    # EDGE: signal_dir == 0 -> never aligned anywhere
    assert not T.trend_aligned(bars, np.zeros(n, int), '1min').any()
    # EDGE: flat HTF (htf==0) -> not aligned for long OR short
    flat = htf == 0
    assert not al_long[flat].any() and not al_short[flat].any()
    # MIXED signal vector matches the elementwise rule exactly
    sd = rng.choice([-1, 0, 1], n)
    assert np.array_equal(T.trend_aligned(bars, sd, '1min'), (htf == np.sign(sd)) & (sd != 0))


# ---- trend_aligned: causal (future can't change past gate decisions) -------
def test_trend_aligned_causal():
    bars, _ = _swing_bars()
    n = len(bars['c'])
    al = T.trend_aligned(bars, np.ones(n, int), '1min')
    b2 = {k: (v.copy() if hasattr(v, 'copy') else v) for k, v in bars.items()}
    b2['h'][3000:] *= 1.2; b2['l'][3000:] *= 1.2; b2['c'][3000:] *= 1.2
    al2 = T.trend_aligned(b2, np.ones(n, int), '1min')
    assert np.array_equal(al[:2900], al2[:2900])


# ---- decycler slope: causal sign tracks trend -----------------------------
def test_decycler_slope_direction():
    up = np.linspace(np.log(100), np.log(112), 1000)
    ds = T.decycler_slope(up, 60, 5)
    assert np.nanmean(ds[200:]) > 0


# ---- swing_pivots: causal + correct geometry ------------------------------
def test_swing_pivots_causal():
    rng = np.random.default_rng(1)
    c = 100 + np.cumsum(rng.standard_normal(2000) * 0.3)
    h = c + np.abs(rng.standard_normal(2000)) * 0.1
    l = c - np.abs(rng.standard_normal(2000)) * 0.1
    p1 = T.swing_pivots(h, l, c, 3)
    k = 1500
    h2, l2, c2 = h.copy(), l.copy(), c.copy()
    h2[k:] += 5; l2[k:] += 5; c2[k:] += 5
    p2 = T.swing_pivots(h2, l2, c2, 3)
    assert np.array_equal(p1[:k], p2[:k])              # future cannot change past pivots


def test_swing_pivots_geometry():
    # construct a clean swing low at i=5 then an up candle at i=6 -> LONG at 6
    c = np.array([10, 9, 8, 7, 6, 5, 7, 8, 9, 10], float)   # trough at idx5
    h = c + 0.1
    l = c - 0.1
    piv = T.swing_pivots(h, l, c, 3)
    assert piv[6] == 1                                 # long confirmed at the up bar
    # mirror: a peak then a down candle -> SHORT
    c2 = np.array([5, 6, 7, 8, 9, 10, 8, 7, 6, 5], float)   # peak at idx5
    piv2 = T.swing_pivots(c2 + 0.1, c2 - 0.1, c2, 3)
    assert piv2[6] == -1
