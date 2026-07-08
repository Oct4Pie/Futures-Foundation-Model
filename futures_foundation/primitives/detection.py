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


def atr_zigzag_legs(o, h, l, c, atr, rev_atr, aflr):
    """ATR-ZigZag swing/pivot detection — the fractal pivot-confirm, SHARED so
    3rd-party consumers (e.g. the live bot) derive the SAME pivots from bars (no
    drift). Pure OHLC, strictly causal: a pivot is only emitted once price has
    retraced `rev_atr * ATR` off the extreme (the moment it's KNOWN).

    Returns list of legs: (origin_idx, extreme_idx, dir, R, confirm_idx). A leg
    runs from origin (a swing extreme) to extreme, in `dir` (+1 up / -1 down).
    confirm_idx = bar at which the leg's ORIGIN pivot was confirmed causally; a
    causal entry for the leg is confirm_idx+1. The first seed leg has
    confirm_idx == origin (not a real signal; drop downstream).

    rev_atr : reversal threshold = retrace this * ATR off the extreme (granularity).
    aflr    : ATR floor for the R-normalization (kills low-ATR R explosions)."""
    def _r(move, oidx):
        return move / (0.5 * max(atr[oidx], aflr))
    n = len(c)
    legs = []
    i0 = next((k for k in range(n) if np.isfinite(atr[k]) and atr[k] > 0), None)
    if i0 is None:
        return legs
    origin = i0
    direction = 0                       # 0 = unknown until first reversal sets it
    ext_idx = i0
    ext_px = c[i0]
    confirm = i0                        # confirm bar of the CURRENT leg's origin pivot
    for j in range(i0 + 1, n):
        a = atr[origin] if np.isfinite(atr[origin]) and atr[origin] > 0 else atr[j]
        if not (np.isfinite(a) and a > 0):
            continue
        rev = rev_atr * a
        if direction >= 0 and h[j] > ext_px:              # extend up-extreme
            ext_px, ext_idx, direction = h[j], j, 1
        if direction <= 0 and l[j] < ext_px and direction != 1:
            ext_px, ext_idx, direction = l[j], j, -1
        if direction == 1 and (ext_px - l[j]) >= rev:     # retrace down -> up-pivot confirmed
            legs.append((origin, ext_idx, 1, _r(ext_px - c[origin], origin), confirm))
            origin, ext_idx, ext_px, direction = ext_idx, j, l[j], -1
            confirm = j
        elif direction == -1 and (h[j] - ext_px) >= rev:  # retrace up -> down-pivot confirmed
            legs.append((origin, ext_idx, -1, _r(c[origin] - ext_px, origin), confirm))
            origin, ext_idx, ext_px, direction = ext_idx, j, h[j], 1
            confirm = j
    # final in-progress leg (current trend at the data edge); filtered downstream
    if direction != 0 and ext_idx > origin:
        move = (ext_px - c[origin]) if direction == 1 else (c[origin] - ext_px)
        legs.append((origin, ext_idx, direction, _r(move, origin), confirm))
    return legs


def detect_atr_zigzag_pivots(o, h, l, c, atr_period=20, rev_atr=1.25,
                             min_r=3.0, min_bars=5):
    """FULL self-contained pivot-confirm — bars → confirmed swing pivots ready to
    trade. THE shared entry-trigger method so any consumer (live bot, other
    strategies) derives IDENTICAL pivots from OHLC, no drift. Byte-identical to the
    fractal label scan (`trend_scan`): Wilder ATR(atr_period) → aflr = 0.5·median(ATR)
    → ATR-zigzag legs → confirmed pivots (seed/edge legs dropped).

    Returns list of dict(confirm, direction, origin, leg_end, R, is_trend):
      confirm   : causal confirm bar — ENTER at confirm+1
      direction : +1 long / -1 short
      origin    : swing extreme (stop reference)
      leg_end   : leg extreme
      R         : |leg R| (ATR-normalized)
      is_trend  : |R| >= min_r AND (leg_end - origin) >= min_bars  (good vs chop pivot)"""
    from futures_foundation.pipeline._primitives import compute_atr  # the certified ATR
    o = np.asarray(o, float); h = np.asarray(h, float)
    l = np.asarray(l, float); c = np.asarray(c, float)
    atr = compute_atr(h, l, c, atr_period)
    fin = np.isfinite(atr) & (atr > 0)
    if fin.sum() < 1:
        return []
    aflr = 0.5 * float(np.nanmedian(atr[fin]))
    n = len(c)
    pivots = []
    for (oi, ei, d, R, cf) in atr_zigzag_legs(o, h, l, c, atr, rev_atr, aflr):
        if cf <= oi or cf + 1 >= n:                  # seed leg / no causal entry
            continue
        pivots.append({'confirm': int(cf), 'direction': int(d), 'origin': int(oi),
                       'leg_end': int(ei), 'R': float(abs(R)),
                       'is_trend': bool(abs(R) >= min_r and (ei - oi) >= min_bars)})
    return pivots


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
