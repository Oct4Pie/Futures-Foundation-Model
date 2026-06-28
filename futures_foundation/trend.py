"""Fast, causal trend-detection primitives — reusable across strategies and the
regime HMM. Low-lag by design: the point is to confirm the trend AT the pivot,
not bars late like ADX (double-smoothed → laggy). Every value at bar i uses only
bars <= i (strictly causal — these are model inputs; one future peek breaks OOS).

  - efficiency_ratio : Kaufman Efficiency Ratio (1 = clean trend, 0 = chop) over a
                       trailing window. Vectorized.
  - kalman_velocity  : constant-velocity Kalman on a series -> (level, velocity);
                       sign(velocity) = direction, |velocity| = trend strength.
                       Adaptive + low-lag (the chop-adaptive gain is why it's
                       responsive). Ported from the validated kalman_nw strategy.
  - ehlers_decycler  : re-exported from pipeline._primitives (THE certified impl
                       the live strategies use — single source of truth).
  - decycler_slope   : signed low-lag trend slope derived from the decycler.

These are the trend confluence the fractal pivot model was missing: a pivot is
the location; these confirm the trend is real before entry.
"""
import numpy as np

from futures_foundation.pipeline._primitives import ehlers_decycler  # certified, reuse

__all__ = ['efficiency_ratio', 'kalman_velocity', 'ehlers_decycler', 'decycler_slope',
           'swing_pivots']


def swing_pivots(high, low, close, lookback=3):
    """Causal swing-pivot LOCATOR (no displacement magnitude — location only):

      SHORT pivot: high[i] is the highest of the trailing `lookback` bars
                   (i-lookback+1 .. i) AND the next bar closes against it
                   (close[i+1] < close[i]).
      LONG  pivot: low[i] is the lowest of the trailing `lookback` bars AND the
                   next bar closes up (close[i+1] > close[i]).

    The pivot is CONFIRMED at the reversal bar i+1 (enter at i+1 / its next open).
    Returns int8[n]: +1 (long) / -1 (short) / 0 at the confirm bar. Strictly
    causal — out[i+1] uses only h/l<=i and closes<=i+1. Vectorized.

    Lighter and earlier than a centered fractal (which must wait `lookback` bars
    after the extreme); here the reversal candle is the confirmation. The trend
    HMM provides the confidence, so the locator carries no displacement magnitude."""
    import pandas as pd
    h = np.asarray(high, np.float64)
    l = np.asarray(low, np.float64)
    c = np.asarray(close, np.float64)
    n = len(c)
    out = np.zeros(n, np.int8)
    if n < lookback + 2:
        return out
    roll_hi = pd.Series(h).rolling(lookback).max().to_numpy()   # max of trailing `lookback` ending at i
    roll_lo = pd.Series(l).rolling(lookback).min().to_numpy()
    is_high = h >= roll_hi
    is_low = l <= roll_lo
    dn = c[1:] < c[:-1]                                         # close[i+1] < close[i]
    up = c[1:] > c[:-1]
    out[1:][is_high[:-1] & dn] = -1
    out[1:][is_low[:-1] & up] = 1
    return out


def efficiency_ratio(values, window=20):
    """Kaufman Efficiency Ratio over a TRAILING window:

        ER[i] = |value[i] - value[i-window]|  /  sum_{j=i-window+1..i} |value[j]-value[j-1]|

    1 = perfectly directional (clean trend), ~0 = choppy back-and-forth. Causal
    (bar i uses bars i-window..i); NaN until `window` bars exist. Pure numpy,
    vectorized."""
    v = np.asarray(values, np.float64)
    n = len(v)
    out = np.full(n, np.nan)
    if n <= window:
        return out
    absdiff = np.abs(np.diff(v))                          # |Δ| per step, len n-1
    csum = np.concatenate([[0.0], np.cumsum(absdiff)])    # csum[i] = sum(absdiff[:i])
    idx = np.arange(window, n)
    denom = csum[idx] - csum[idx - window]                # path length over window
    net = np.abs(v[idx] - v[idx - window])                # net displacement
    out[idx] = np.where(denom > 1e-12, net / denom, 0.0)
    return out


def kalman_velocity(values, q=1e-5, r=1e-3):
    """Causal constant-velocity Kalman (local level + velocity). Returns
    (level, velocity) arrays; estimate[i] uses only observations <= i (forward
    recursion). sign(velocity) = trend direction, |velocity| = strength.

    q = process noise (trend agility), r = measurement noise; the ratio q/r sets
    responsiveness. Feed LOG-price for scale-free behavior. Ported verbatim from
    the kalman_nw strategy's `kalman_trend` (the validated live impl)."""
    v = np.asarray(values, np.float64)
    n = len(v)
    level = np.zeros(n)
    vel = np.zeros(n)
    if n == 0:
        return level, vel
    x = np.array([v[0], 0.0])                             # state [level, velocity]
    P = np.eye(2) * 1.0
    Q = np.array([[q, 0.0], [0.0, q]])
    F = np.array([[1.0, 1.0], [0.0, 1.0]])                # constant-velocity model
    for i in range(n):
        x = F @ x                                         # predict
        P = F @ P @ F.T + Q
        y = v[i] - x[0]                                   # innovation
        S = P[0, 0] + r
        Kg = P[:, 0] / S                                  # Kalman gain
        x = x + Kg * y                                    # update
        P = P - np.outer(Kg, P[0, :])
        level[i] = x[0]
        vel[i] = x[1]
    return level, vel


def decycler_slope(values, period=60, k=5):
    """Signed low-lag trend slope: decycler[i] - decycler[i-k]. Positive = rising
    trend, negative = falling. Causal; NaN for the first k bars. `period` is the
    decycler high-pass cutoff (cycles shorter than this are removed → trend)."""
    dec = np.asarray(ehlers_decycler(values, period), np.float64)
    n = len(dec)
    out = np.full(n, np.nan)
    if n > k:
        out[k:] = dec[k:] - dec[:-k]
    return out
