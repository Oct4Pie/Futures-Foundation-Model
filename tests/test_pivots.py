"""Pivot/HTF feature tests — the load-bearing guarantee is CAUSALITY (these are
model inputs; one future peek breaks OOS) plus numpy==pandas parity (the fast
path must match the reference)."""
import numpy as np
import pandas as pd

from futures_foundation import pivots as P


def _series(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.standard_normal(n) * 0.002))
    h = c * (1 + np.abs(rng.standard_normal(n)) * 0.001)
    l = c * (1 - np.abs(rng.standard_normal(n)) * 0.001)
    return dict(h=h, l=l, c=c, atr=np.full(n, 0.5))


# ---- numpy == pandas parity (fast path matches reference) ------------------
def test_roll_and_since_match_pandas():
    a = _series(5000)['c']
    for n in (20, 50):
        pm = pd.Series(a).rolling(n, min_periods=1).max().to_numpy()
        pn = pd.Series(a).rolling(n, min_periods=1).min().to_numpy()
        np.testing.assert_allclose(P._roll_max(a, n), pm)
        np.testing.assert_allclose(P._roll_min(a, n), pn)
        for mx in (True, False):
            f = np.argmax if mx else np.argmin
            ref = pd.Series(a).rolling(n, min_periods=1).apply(
                lambda x: len(x) - 1 - f(x), raw=True).to_numpy()
            np.testing.assert_array_equal(P._bars_since(a, n, mx), ref)


# ---- CAUSALITY: perturb the future, past rows MUST NOT change --------------
def test_pivot_features_causal():
    B = _series()
    X1 = P.pivot_features(B)
    k = 1500
    B2 = {kk: v.copy() for kk, v in B.items()}
    B2['h'][k:] += 50; B2['l'][k:] += 50; B2['c'][k:] += 50
    X2 = P.pivot_features(B2)
    assert np.abs(X1[:k] - X2[:k]).max() == 0.0          # no leakage from future
    assert np.abs(X1[k:] - X2[k:]).max() > 0             # but it DOES use the data


def test_causal_htf_dir_causal():
    n = 6000
    ts = pd.date_range('2024-01-01', periods=n, freq='1min', tz='UTC').values
    rng = np.random.default_rng(2)
    c = 100 * np.exp(np.cumsum(rng.standard_normal(n) * 0.001))
    B1 = dict(ts=ts, o=c, h=c * 1.0005, l=c * 0.9995, c=c)
    d1 = P.causal_htf_dir(B1, '1min', ts, atr_p=20)
    k = 4000
    B2 = {kk: (v.copy() if hasattr(v, 'copy') else v) for kk, v in B1.items()}
    B2['h'][k:] *= 1.05; B2['l'][k:] *= 1.05; B2['c'][k:] *= 1.05
    d2 = P.causal_htf_dir(B2, '1min', ts, atr_p=20)
    assert np.array_equal(d1[:k], d2[:k])                 # every past bar is stable


def test_causal_htf_dir_matches_prefix_terminal_values():
    """Batch output at a bar must equal output computed live through that bar."""
    n = 1200
    i = np.arange(n, dtype=float)
    c = 100 + 0.015 * i + 2.2 * np.sin(i / 13) + 0.7 * np.sin(i / 3.1)
    o = c - 0.2 * np.cos(i / 4.7)
    h = np.maximum(o, c) + 0.6 + 0.1 * np.sin(i / 5)
    l = np.minimum(o, c) - 0.5 - 0.08 * np.cos(i / 7)
    ts = pd.date_range('2026-01-01', periods=n, freq='3min', tz='UTC').values
    bars = dict(ts=ts, o=o, h=h, l=l, c=c)

    batch = P.causal_htf_dir(bars, '3min', ts, atr_p=20)
    endpoints = np.arange(299, n, 10)
    live = np.array([
        P.causal_htf_dir(
            {name: values[:endpoint + 1] for name, values in bars.items()},
            '3min',
            ts[:endpoint + 1],
            atr_p=20,
        )[-1]
        for endpoint in endpoints
    ])

    np.testing.assert_array_equal(batch[endpoints], live)


def test_causal_htf_dir_prefix_invariant():
    """Appending bars must never revise any direction already emitted at a prefix."""
    n = 3000
    ts = pd.date_range('2024-01-01', periods=n, freq='1min', tz='UTC').values
    rng = np.random.default_rng(7)
    c = 100 + np.cumsum(rng.normal(0, .2, n))
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) + rng.uniform(0, .1, n)
    l = np.minimum(o, c) - rng.uniform(0, .1, n)
    bars = dict(ts=ts, o=o, h=h, l=l, c=c)
    full = P.causal_htf_dir(bars, '1min', ts, atr_p=20)
    for end in range(1000, n, 200):
        prefix = {key: value[:end] for key, value in bars.items()}
        observed = P.causal_htf_dir(prefix, '1min', ts[:end], atr_p=20)
        np.testing.assert_array_equal(observed, full[:end])


def test_pivot_names_match_width():
    B = _series()
    X = P.pivot_features(B)
    # pivot_features = all PIVOT_NAMES except the htf_dir (appended separately)
    assert X.shape[1] == len(P.PIVOT_NAMES) - 1
