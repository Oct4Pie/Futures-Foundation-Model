"""Reusable STRATEGY-TRIGGER INDICATORS — one library, every strategy imports from here.

CAUSAL BY CONTRACT: every function uses only bars <= i for output i (these feed entry triggers
and model features; a single future peek breaks OOS). New strategies must take indicators from
this module — never re-implement them in a script (train/serve parity + one certified impl).

The battle-tested primitives (ATR / ADX / SuperTrend / RR barriers) are RE-EXPORTED from the
certified pipeline._primitives implementations — the live bot was certified on those exact
functions, so there is exactly ONE implementation (never fork them; see the primitives-
divergence trap). The additions here (EMA, Kalman velocity, Nadaraya-Watson slope, opening
range) are ports of the strategy-script implementations, moved to the lib verbatim.
"""
import numpy as np
import pandas as pd

from futures_foundation.pipeline._primitives import (      # noqa: F401  certified — ONE impl
    compute_atr, compute_adx, compute_supertrend, apply_rr_barriers, ehlers_decycler)


def ema(x, span):
    """Causal exponential moving average (pandas ewm, adjust=False — the strategy-certified
    form). Returns float array aligned to x."""
    return pd.Series(np.asarray(x, float)).ewm(span=int(span), adjust=False).mean().to_numpy()


def adx_slope(adx, lag=5):
    """Causal ADX slope: adx[i] - adx[i-lag] (NaN for the first `lag` bars)."""
    adx = np.asarray(adx, float)
    out = np.full_like(adx, np.nan)
    out[lag:] = adx[lag:] - adx[:-lag]
    return out


def kalman_velocity(lp, q=1e-5, r=1e-3):
    """Causal constant-velocity Kalman filter on log-price -> velocity (trend slope).
    The Kalman-NW strategy's trend direction; also a data-confirmed lifter feature."""
    lp = np.asarray(lp, float)
    n = len(lp)
    vel = np.zeros(n)
    x = np.array([lp[0], 0.0]); P = np.eye(2)
    F = np.array([[1.0, 1.0], [0.0, 1.0]]); Q = np.array([[q, 0.0], [0.0, q]])
    for i in range(n):
        x = F @ x; P = F @ P @ F.T + Q
        y = lp[i] - x[0]; S = P[0, 0] + r; K = P[:, 0] / S
        x = x + K * y; P = P - np.outer(K, P[0, :])
        vel[i] = x[1]
    return vel


def nw_slope(lp, win=30, bw=8.0):
    """Causal Nadaraya-Watson local slope over the trailing `win` bars of log-price (Gaussian
    kernel anchored at the window end). NaN until `win` bars exist. The Kalman-NW entry signal."""
    lp = np.asarray(lp, float)
    n = len(lp)
    out = np.full(n, np.nan)
    t = np.arange(win, dtype=float)
    w = np.exp(-0.5 * ((t - (win - 1)) / bw) ** 2)
    tw = t * w; sw = w.sum(); stw = tw.sum()
    denom = sw * (t * tw).sum() - stw * stw
    if not denom:
        return out
    for i in range(win - 1, n):
        seg = lp[i - win + 1:i + 1]
        out[i] = (sw * (tw * seg).sum() - stw * (w * seg).sum()) / denom
    return out


def opening_range(ts, h, l, orb_bars=5, session_tz='America/New_York', open_min=9 * 60 + 30):
    """Causal opening range. Returns (or_high, or_low): at bar i, the ACTIVE opening-range
    high/low for i's session day — the high/low of the first `orb_bars` bars at/after the
    session open, active only from the bar AFTER the window closes (the range is fully in the
    past). NaN before activation. Uses only bars <= i by construction."""
    h = np.asarray(h, float); l = np.asarray(l, float)
    et = pd.DatetimeIndex(ts).tz_convert(session_tz)
    day = et.normalize().asi8
    tmin = (et.hour * 60 + et.minute).to_numpy()
    n = len(h)
    or_high = np.full(n, np.nan)
    or_low = np.full(n, np.nan)
    cur_day = None
    oh = ol = np.nan
    count, done = 0, False
    for i in range(n):
        if day[i] != cur_day:                               # new session day
            cur_day = day[i]; oh = ol = np.nan; count = 0; done = False
        if (not done) and tmin[i] >= open_min:              # building the range
            oh = h[i] if count == 0 else max(oh, h[i])
            ol = l[i] if count == 0 else min(ol, l[i])
            count += 1
            if count >= orb_bars:
                done = True                                 # known AFTER this bar
        elif done:                                          # range active
            or_high[i] = oh; or_low[i] = ol
    return or_high, or_low
