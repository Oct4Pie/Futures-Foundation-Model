"""Signal detection — pivot highs/lows, CISD signals, OTE zones."""

from collections import deque

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def detect_pivots(highs, lows, period):
    """
    Vectorised pivot high / pivot low detection.

    A bar at index b is a pivot high if highs[b] is the unique maximum of
    the (2*period+1) window centred on b. Same for pivot lows.

    Returns:
        pivot_high_bars : int array — bar indices of pivot highs
        pivot_low_bars  : int array — bar indices of pivot lows
    """
    highs = np.asarray(highs, dtype=np.float64)
    lows  = np.asarray(lows,  dtype=np.float64)
    n     = len(highs)

    if n < 2 * period + 1:
        return np.array([], dtype=int), np.array([], dtype=int)

    win_h   = sliding_window_view(highs, 2 * period + 1)
    win_l   = sliding_window_view(lows,  2 * period + 1)
    win_max = win_h.max(axis=1)
    win_min = win_l.min(axis=1)
    ch      = win_h[:, period]
    cl      = win_l[:, period]

    is_ph = (ch == win_max) & (np.sum(win_h == ch[:, None], axis=1) == 1)
    is_pl = (cl == win_min) & (np.sum(win_l == cl[:, None], axis=1) == 1)

    return np.where(is_ph)[0] + period, np.where(is_pl)[0] + period


def detect_atr_zigzag_pivots_v2(o, h, l, c, atr_period=20, rev_atr=1.25):
    """Prefix-invariant online ATR-reversal pivots for new research artifacts.

    This version emits a pivot only on the bar that *currently* confirms a reversal.  It never
    walks completed future legs and backfills their earlier origin-confirm times.  A confirmed
    swing low emits ``direction=+1``; a confirmed swing high emits ``direction=-1``.  ``R`` is the
    observed reversal distance in ATR units at confirmation, not a future leg outcome.

    The removed legacy detector walked completed future legs and exposed backfilled fields. New
    causal datasets must use this function and record the detector version explicitly.
    """
    from futures_foundation.pipeline._primitives import compute_atr

    o = np.asarray(o, float)
    h = np.asarray(h, float)
    l = np.asarray(l, float)
    c = np.asarray(c, float)
    if not (len(o) == len(h) == len(l) == len(c)):
        raise ValueError("OHLC arrays must align")
    if atr_period < 1 or rev_atr <= 0:
        raise ValueError("atr_period and rev_atr must be positive")
    n = len(c)
    atr = compute_atr(h, l, c, int(atr_period))
    first = next((i for i in range(n) if np.isfinite(atr[i]) and atr[i] > 0), None)
    if first is None:
        return []

    direction = 0
    extreme_idx = int(first)
    extreme_price = float(c[first])
    out = []
    for j in range(first + 1, n):
        if direction >= 0 and float(h[j]) > extreme_price:
            direction = 1
            extreme_idx = int(j)
            extreme_price = float(h[j])
        elif direction <= 0 and float(l[j]) < extreme_price:
            direction = -1
            extreme_idx = int(j)
            extreme_price = float(l[j])

        scale = float(atr[extreme_idx])
        if not (np.isfinite(scale) and scale > 0):
            continue
        reversal = float(rev_atr) * scale
        if direction == 1 and extreme_price - float(l[j]) >= reversal:
            observed_r = (extreme_price - float(l[j])) / scale
            out.append({
                "confirm": int(j), "direction": -1, "origin": extreme_idx,
                "leg_end": extreme_idx, "R": float(observed_r), "is_trend": False,
            })
            direction = -1
            extreme_idx = int(j)
            extreme_price = float(l[j])
        elif direction == -1 and float(h[j]) - extreme_price >= reversal:
            observed_r = (float(h[j]) - extreme_price) / scale
            out.append({
                "confirm": int(j), "direction": 1, "origin": extreme_idx,
                "leg_end": extreme_idx, "R": float(observed_r), "is_trend": False,
            })
            direction = 1
            extreme_idx = int(j)
            extreme_price = float(h[j])
    return out


def detect_fractal_pivots(h, l, k=2, min_bars_apart=0, live_edge=False):
    """WILLIAMS-FRACTAL pivots — TIME-based confirmation (the trigger-scan winner, 2026-07-09).

    A bar is a pivot LOW when its low is strictly the lowest of the k bars on EACH side (unique in
    the 2k+1 window); pivot HIGH symmetric. CONFIRM = extreme + k (the k right-side bars must have
    closed) -> ENTER at confirm+1. Confirmation is paid in TIME (k bars), not price distance — the
    naive-floor scan measured k=2/3 at WR@3R 36.2-36.5% / meanR +0.42 vs the ATR-zigzag incumbent's
    33.8% / +0.32 at comparable entry tax: the "held for k bars" rule selects cleaner turns than a
    fixed-ATR reversal.

    Returns the shared pivot schema (confirm, direction, origin, leg_end, R,
    is_trend) so every consumer accepts it as a drop-in trigger variant: origin = leg_end = the
    fractal extreme (stop reference); R/is_trend are placeholders (0/False — no leg semantics here).
    min_bars_apart > 0 optionally suppresses a pivot within that many bars of the previous SAME-
    direction pivot (fractals can cluster on flat stretches). Strictly causal at the confirm bar.

    live_edge: training keeps the cf+1 < n guard (every emitted pivot has an entry bar, needed for
    labeling); a bar-by-bar LIVE/backtest consumer passes live_edge=True so a pivot confirming on
    the NEWEST bar is emitted (its entry bar is the next bar, which doesn't exist yet)."""
    h = np.asarray(h, float)
    l = np.asarray(l, float)
    n = len(h)
    out = []
    last = {1: -10**9, -1: -10**9}
    for i in range(int(k), n - int(k)):
        seg_l = l[i - k:i + k + 1]
        seg_h = h[i - k:i + k + 1]
        d = 0
        if l[i] == seg_l.min() and (seg_l == l[i]).sum() == 1:
            d = 1                                        # unique lowest low -> long pivot
        elif h[i] == seg_h.max() and (seg_h == h[i]).sum() == 1:
            d = -1                                       # unique highest high -> short pivot
        if d == 0:
            continue
        if min_bars_apart and i - last[d] < int(min_bars_apart):
            continue
        cf = i + int(k)
        if not live_edge and cf + 1 >= n:    # training-only: require the entry bar to exist
            continue
        last[d] = i
        out.append({'confirm': int(cf), 'direction': int(d), 'origin': int(i),
                    'leg_end': int(i), 'R': 0.0, 'is_trend': False})
    return out


def detect_fractal_zigzag_pivots(o, h, l, c, k=2, min_leg_atr=1.25, atr_period=20,
                                 live_edge=False):
    """FRACTAL-ZIGZAG hybrid — fractal TIME confirmation + zigzag STRUCTURE (the trigger-scan
    causal winner, 2026-07-09: WR@3R 37.0% / meanR +0.449 @ leg=1.25 vs the ATR-zigzag incumbent's
    33.8% / +0.323 on ES+NQ 3min pre-2026).

    Pivots = Williams fractals (unique extreme of ±k bars, confirm = extreme+k) filtered by the
    zigzag significance rule, STRICTLY CAUSALLY (keep-FIRST state machine — a pivot is kept iff at
    ITS confirm it (a) alternates with the last KEPT pivot and (b) the leg from that pivot's extreme
    is >= min_leg_atr * ATR(extreme). No hindsight replacement: a later deeper same-direction
    fractal is SKIPPED, never swapped in — the naive floor of the swap variant is inflated ~9 WR pts
    by lookahead (measured). Shared pivot schema; R = leg size in ATRs."""
    from futures_foundation.pipeline._primitives import compute_atr
    o = np.asarray(o, float); h = np.asarray(h, float)
    l = np.asarray(l, float); c = np.asarray(c, float)
    atr = compute_atr(h, l, c, atr_period)
    out = []
    last_d, last_px, last_i = 0, None, None
    for p in detect_fractal_pivots(h, l, k=k, live_edge=live_edge):
        i, d = p['origin'], p['direction']
        px = l[i] if d == 1 else h[i]
        a = atr[i]
        if not (np.isfinite(a) and a > 0):
            continue
        if d == last_d:
            continue                                     # keep-first: no hindsight replace
        if last_px is not None and abs(px - last_px) < float(min_leg_atr) * a:
            continue                                     # leg too small = noise, not a swing
        leg_r = 0.0 if last_px is None else float(abs(px - last_px) / a)
        out.append({'confirm': p['confirm'], 'direction': d, 'origin': int(i),
                    'leg_end': int(i), 'R': leg_r, 'is_trend': False})
        last_d, last_px, last_i = d, px, i
    return out


def detect_cisd_signals(o, h, l, c, tolerance=0.70, expiry_bars=50,
                        body_ratio_min=0.50, close_str_min=0.60):
    """
    Detect CISD (Change In State of Delivery) displacement candles.

    A bearish CISD forms when a prior bearish candle's open (pot_price) is
    breached to the upside and closes back below it on a strong bear bar.
    A bullish CISD forms symmetrically.

    Args:
        o, h, l, c       : float64 arrays of OHLC (equal length n)
        tolerance        : minimum sweep ratio for a valid CISD (default 0.70)
        expiry_bars      : potential zones expire after this many bars
        body_ratio_min   : minimum body/range for displacement candle
        close_str_min    : minimum close-strength for displacement candle
                           (bear: upper wick / range; bull: lower wick / range)

    Returns dict of arrays (all length n):
        cisd_signal      : int8    — 0=none, 1=bearish CISD, 2=bullish CISD
        displacement_str : float32 — sweep ratio of displacement candle
        disp_body_ratio  : float32 — body/range of displacement candle
        disp_close_str   : float32 — close strength of displacement candle
        origin_level     : float64 — price level of the triggering potential
    """
    o = np.asarray(o, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    l = np.asarray(l, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    n = len(c)

    cisd_signal     = np.zeros(n, dtype=np.int8)
    disp_strength   = np.zeros(n, dtype=np.float32)
    disp_body_ratio = np.zeros(n, dtype=np.float32)
    disp_close_str  = np.zeros(n, dtype=np.float32)
    origin_level    = np.full(n, np.nan, dtype=np.float64)

    bear_pots = deque()
    bull_pots = deque()

    for bar in range(1, n):
        if c[bar - 1] < o[bar - 1] and c[bar] > o[bar]:
            bear_pots.append((o[bar], bar))
        if c[bar - 1] > o[bar - 1] and c[bar] < o[bar]:
            bull_pots.append((o[bar], bar))

        while bear_pots and bar - bear_pots[0][1] >= expiry_bars:
            bear_pots.popleft()
        while bull_pots and bar - bull_pots[0][1] >= expiry_bars:
            bull_pots.popleft()

        # ── Bearish CISD ──
        while bear_pots:
            pot_price, pot_bar = bear_pots[0]
            if c[bar] < pot_price:
                highest_c = c[pot_bar:bar + 1].max()
                top_level = 0.0
                idx = pot_bar + 1
                while idx < bar and c[idx] < o[idx]:
                    top_level = o[idx]; idx += 1
                if top_level > 0 and (top_level - pot_price) > 0:
                    ratio = (highest_c - pot_price) / (top_level - pot_price)
                    if ratio > tolerance:
                        full_range = h[bar] - l[bar]
                        body = abs(c[bar] - o[bar])
                        br   = body / full_range if full_range > 0 else 0.0
                        cs   = (h[bar] - c[bar]) / full_range if full_range > 0 else 0.0
                        if br >= body_ratio_min and cs >= close_str_min:
                            cisd_signal[bar]     = 1
                            origin_level[bar]    = pot_price
                            disp_strength[bar]   = ratio
                            disp_body_ratio[bar] = br
                            disp_close_str[bar]  = cs
                            bear_pots.clear(); break
                        else:
                            bear_pots.popleft(); continue
                    else:
                        bear_pots.popleft(); continue
                else:
                    bear_pots.popleft(); continue
            else:
                break

        # ── Bullish CISD ──
        while bull_pots:
            pot_price, pot_bar = bull_pots[0]
            if c[bar] > pot_price:
                lowest_c     = c[pot_bar:bar + 1].min()
                bottom_level = 0.0
                idx = pot_bar + 1
                while idx < bar and c[idx] > o[idx]:
                    bottom_level = o[idx]; idx += 1
                if bottom_level > 0 and (pot_price - bottom_level) > 0:
                    ratio = (pot_price - lowest_c) / (pot_price - bottom_level)
                    if ratio > tolerance:
                        full_range = h[bar] - l[bar]
                        body = abs(c[bar] - o[bar])
                        br   = body / full_range if full_range > 0 else 0.0
                        cs   = (c[bar] - l[bar]) / full_range if full_range > 0 else 0.0
                        if br >= body_ratio_min and cs >= close_str_min:
                            cisd_signal[bar]     = 2
                            origin_level[bar]    = pot_price
                            disp_strength[bar]   = ratio
                            disp_body_ratio[bar] = br
                            disp_close_str[bar]  = cs
                            bull_pots.clear(); break
                        else:
                            bull_pots.popleft(); continue
                    else:
                        bull_pots.popleft(); continue
                else:
                    bull_pots.popleft(); continue
            else:
                break

    return {
        'cisd_signal':      cisd_signal,
        'displacement_str': disp_strength,
        'disp_body_ratio':  disp_body_ratio,
        'disp_close_str':   disp_close_str,
        'origin_level':     origin_level,
    }


def compute_ote_zones(cisd_result, h, l, c, fib_1=0.618, fib_2=0.786):
    """
    Compute OTE (Optimal Trade Entry) fibonacci zones from CISD signals.

    For bearish CISD (signal=1): uses highest high up to signal bar.
    For bullish CISD (signal=2): uses lowest low up to signal bar.

    Returns:
        fib_top : float64 array (n,) — upper boundary of OTE zone
        fib_bot : float64 array (n,) — lower boundary of OTE zone
    """
    h = np.asarray(h, dtype=np.float64)
    l = np.asarray(l, dtype=np.float64)
    n = len(h)

    cisd_signal = cisd_result['cisd_signal']
    fib_top = np.full(n, np.nan, dtype=np.float64)
    fib_bot = np.full(n, np.nan, dtype=np.float64)

    for bar in range(n):
        sig = cisd_signal[bar]
        if sig == 1:
            h_max = h[:bar + 1].max()
            diff  = h_max - l[bar]
            if diff > 0:
                f1 = h_max - diff * fib_1
                f2 = h_max - diff * fib_2
                fib_top[bar] = max(f1, f2)
                fib_bot[bar] = min(f1, f2)
        elif sig == 2:
            l_min = l[:bar + 1].min()
            diff  = h[bar] - l_min
            if diff > 0:
                f1 = l_min + diff * fib_1
                f2 = l_min + diff * fib_2
                fib_top[bar] = max(f1, f2)
                fib_bot[bar] = min(f1, f2)

    return fib_top, fib_bot
