"""Causal pivot + HTF feature functions — GENERIC FFM capability.

Reusable by any strategy AND the live bot (same code -> train/serve parity). EVERYTHING here is CAUSAL (bar i uses
only bars <= i) — these are MODEL INPUTS, so a single future peek breaks OOS.
(The trend-scan LABEL is separate and may use the future; these features may not.)
"""
import numpy as np
import pandas as pd

from futures_foundation.pipeline._primitives import compute_atr

PIVOT_NS = (20, 50)        # rolling lookbacks
PIVOT_KS = (5, 20)         # velocity lags
HTF_MAP = {'1min': '5min', '3min': '15min', '5min': '30min', '15min': '60min'}

PIVOT_NAMES = []
for _N in PIVOT_NS:
    PIVOT_NAMES += [f'pos_in_range_{_N}', f'pull_from_hi_{_N}', f'pull_from_lo_{_N}',
                    f'bars_since_hi_{_N}', f'bars_since_lo_{_N}', f'range_{_N}']
PIVOT_NAMES += [f'ret_{_k}' for _k in PIVOT_KS] + ['htf_dir']


from numpy.lib.stride_tricks import sliding_window_view as _swv

# NUMPY-vectorized rolling (no pandas rolling.apply) -> fast at LIVE inference
# (per-bar = compute on the recent window, microseconds) and in batch.


def _roll_max(a, n):
    a = np.asarray(a, float); T = len(a)
    out = np.maximum.accumulate(a).astype(float)        # warmup (min_periods=1)
    if T >= n:
        out[n - 1:] = _swv(a, n).max(1)
    return out


def _roll_min(a, n):
    a = np.asarray(a, float); T = len(a)
    out = np.minimum.accumulate(a).astype(float)
    if T >= n:
        out[n - 1:] = _swv(a, n).min(1)
    return out


def _bars_since(a, n, want_max):
    a = np.asarray(a, float); T = len(a)
    out = np.empty(T, float)
    for i in range(min(n - 1, T)):                      # tiny warmup loop (<n)
        seg = a[:i + 1]
        out[i] = i - (np.argmax(seg) if want_max else np.argmin(seg))
    if T >= n:                                          # vectorized bulk (C-fast)
        sw = _swv(a, n)
        arg = sw.argmax(1) if want_max else sw.argmin(1)
        out[n - 1:] = (n - 1) - arg
    return out


def pivot_features(B):
    """Causal pivot-structure features [T, len(PIVOT_NS)*6 + len(PIVOT_KS)]
    (htf appended separately by causal_htf_dir)."""
    h, l, c, atr = B['h'], B['l'], B['c'], np.clip(B['atr'], 1e-9, None)
    feats = []
    for N in PIVOT_NS:
        hi, lo = _roll_max(h, N), _roll_min(l, N)
        rng = np.clip(hi - lo, 1e-9, None)
        feats += [(c - lo) / rng, (hi - c) / atr, (c - lo) / atr,
                  _bars_since(h, N, True) / N, _bars_since(l, N, False) / N,
                  (hi - lo) / atr]
    for k in PIVOT_KS:
        r = np.zeros_like(c); r[k:] = (c[k:] - c[:-k]) / atr[k:]
        feats.append(r)
    return np.column_stack(feats).astype(np.float32)


def _to_ns(x):
    idx = pd.DatetimeIndex(pd.to_datetime(x, utc=True))
    return idx.tz_convert('UTC').tz_localize(None).asi8


def _resample(ts, o, h, l, c, freq):
    df = pd.DataFrame({'o': o, 'h': h, 'l': l, 'c': c}, index=pd.to_datetime(ts))
    r = df.resample(freq, label='right', closed='right').agg(
        {'o': 'first', 'h': 'max', 'l': 'min', 'c': 'last'}).dropna()
    return (r.index.values, r['o'].to_numpy(), r['h'].to_numpy(),
            r['l'].to_numpy(), r['c'].to_numpy())


def _zigzag_dir(o, h, l, c, atr, rev_atr, aflr):
    """Forward-only ATR-zigzag direction known at each HTF bar close."""
    n = len(c)
    d = np.zeros(n, np.int8)
    floor = np.asarray(aflr, dtype=float)
    if floor.ndim == 0:
        floor = np.full(n, float(floor))
    if len(floor) != n:
        raise ValueError('aflr must be a scalar or match the ATR length')
    i0 = next((k for k in range(n)
               if np.isfinite(atr[k]) and atr[k] > 0
               and np.isfinite(floor[k]) and floor[k] > 0), None)
    if i0 is None:
        return d
    ext_idx = i0
    ext_px = c[i0]
    direction = 0
    for j in range(i0 + 1, n):
        a = max(atr[ext_idx] if np.isfinite(atr[ext_idx]) and atr[ext_idx] > 0
                else 0.0, floor[j])
        rev = rev_atr * a
        if direction >= 0 and h[j] > ext_px:
            ext_px, ext_idx, direction = h[j], j, 1
        if direction <= 0 and l[j] < ext_px and direction != 1:
            ext_px, ext_idx, direction = l[j], j, -1
        if direction == 1 and (ext_px - l[j]) >= rev:
            ext_idx, ext_px, direction = j, l[j], -1
        elif direction == -1 and (h[j] - ext_px) >= rev:
            ext_idx, ext_px, direction = j, h[j], 1
        d[j] = direction
    return d


def causal_htf_dir(B1, tf, base_ts, atr_p, rev_atr=2.0):
    """CAUSAL HTF direction per base bar: resample 1min->the TF's HTF, zigzag,
    map each base bar to the LAST HTF bar already CLOSED at its time (no future).
    HTF-dir as a FEATURE must be causal -> last-closed mapping (side='right'-1)."""
    htf = HTF_MAP.get(tf)
    if htf is None:
        return np.zeros(len(base_ts), np.int8)
    ts_h, o, h, l, c = _resample(B1['ts'], B1['o'], B1['h'], B1['l'], B1['c'], htf)
    atr_h = compute_atr(h, l, c, atr_p)
    # A full-series median changes past thresholds when future ATR values are
    # appended. The expanding median is available only after 50 causal ATR
    # observations and is prefix-stable at every later HTF bar.
    aflr = 0.5 * pd.Series(atr_h).expanding(min_periods=50).median().to_numpy()
    dh = _zigzag_dir(o, h, l, c, atr_h, rev_atr, aflr)
    idx = np.searchsorted(_to_ns(ts_h), _to_ns(base_ts), side='right') - 1
    out = np.zeros(len(base_ts), np.int8)
    ok = idx >= 0
    out[ok] = dh[idx[ok]]
    return out
