"""Pure-numpy primitives — vendored locally so the chronos pipeline has
NO futures_foundation import chain (importing futures_foundation pulls in
the FFM PyTorch transformer, which loads libomp into the parent and then
segfaults when XGBoost runs — see project_chronos_openmp_isolation).
Internal: same contracts as futures_foundation.primitives versions.
"""
import numpy as np


def compute_atr(h, l, c, period):
    """Wilder ATR (true range RMA). Returns float64[n]; NaN before period."""
    h = np.asarray(h, np.float64)
    l = np.asarray(l, np.float64)
    c = np.asarray(c, np.float64)
    n = len(c)
    if n == 0:
        return np.zeros(0, np.float64)
    tr = np.empty(n, np.float64)
    tr[0] = h[0] - l[0]
    prev_c = c[:-1]
    tr[1:] = np.maximum.reduce([h[1:] - l[1:],
                                np.abs(h[1:] - prev_c),
                                np.abs(l[1:] - prev_c)])
    atr = np.full(n, np.nan, np.float64)
    if n < period:
        return atr
    atr[period - 1] = tr[:period].mean()
    inv_p = 1.0 / period
    for i in range(period, n):
        atr[i] = atr[i - 1] + (tr[i] - atr[i - 1]) * inv_p   # Wilder RMA
    return atr


def _rma(arr, period):
    """Wilder smoothing (RMA). Returns float64[n]; NaN before warm-up."""
    n = len(arr)
    out = np.full(n, np.nan)
    if n < period:
        return out
    out[period - 1] = arr[:period].mean()
    inv_p = 1.0 / period
    prev = out[period - 1]
    for i in range(period, n):
        prev = prev + (arr[i] - prev) * inv_p
        out[i] = prev
    return out


def compute_supertrend(h, l, c, period, mult):
    """Wilder-ATR SuperTrend. Returns (direction int8 +/-1, st_line, atr).
    direction starts at +1 (bull) so the state machine kicks off."""
    h = np.asarray(h, np.float64)
    l = np.asarray(l, np.float64)
    c = np.asarray(c, np.float64)
    n = len(c)
    atr = compute_atr(h, l, c, period)
    hl2 = (h + l) * 0.5
    upper = hl2 + mult * atr
    lower = hl2 - mult * atr
    f_up = upper.copy()
    f_lo = lower.copy()
    for i in range(1, n):
        if not (np.isfinite(f_up[i - 1]) and np.isfinite(f_lo[i - 1])):
            continue
        if upper[i] < f_up[i - 1] or c[i - 1] > f_up[i - 1]:
            pass                                       # f_up[i] = upper[i]
        else:
            f_up[i] = f_up[i - 1]
        if lower[i] > f_lo[i - 1] or c[i - 1] < f_lo[i - 1]:
            pass                                       # f_lo[i] = lower[i]
        else:
            f_lo[i] = f_lo[i - 1]
    direction = np.ones(n, dtype=np.int8)
    st = np.full(n, np.nan, np.float64)
    for i in range(1, n):
        if direction[i - 1] == 1 and c[i] < f_lo[i]:
            direction[i] = -1
        elif direction[i - 1] == -1 and c[i] > f_up[i]:
            direction[i] = 1
        else:
            direction[i] = direction[i - 1]
        st[i] = f_lo[i] if direction[i] == 1 else f_up[i]
    return direction, st, atr


def compute_adx(h, l, c, period=14):
    """Wilder ADX (default 14). Pure numpy. Returns float64[n]; NaN
    until ~2*period warm-up."""
    h = np.asarray(h, np.float64)
    l = np.asarray(l, np.float64)
    c = np.asarray(c, np.float64)
    n = len(c)
    if n < 2 * period:
        return np.full(n, np.nan)
    tr = np.empty(n)
    tr[0] = h[0] - l[0]
    prev_c = c[:-1]
    tr[1:] = np.maximum.reduce([h[1:] - l[1:],
                                np.abs(h[1:] - prev_c),
                                np.abs(l[1:] - prev_c)])
    up = np.empty(n); up[0] = 0.0; up[1:] = h[1:] - h[:-1]
    dn = np.empty(n); dn[0] = 0.0; dn[1:] = l[:-1] - l[1:]
    plus_dm = np.where((up > dn) & (up > 0.0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0.0), dn, 0.0)
    tr_s = _rma(tr, period)
    pdm_s = _rma(plus_dm, period)
    mdm_s = _rma(minus_dm, period)
    with np.errstate(divide='ignore', invalid='ignore'):
        plus_di = 100.0 * pdm_s / tr_s
        minus_di = 100.0 * mdm_s / tr_s
        denom = plus_di + minus_di
        dx = np.where(denom > 0,
                      100.0 * np.abs(plus_di - minus_di) / denom, 0.0)
    return _rma(dx, period)


def apply_rr_barriers(h, l, c, entry_idx, is_long, entry_price, sl_price,
                      rr_targets, lookahead=None, is_session_end=None):
    """Walk forward from entry_idx measuring R:R outcomes. Stops early on
    stop hit, session end, lookahead exhaustion, or data end. Returns
    dict[float -> {hit:bool, outcome:str, realized_rr:float}]; outcome in
    {target_hit, stopped, session_end, data_end, invalid}."""
    h = np.asarray(h, np.float64)
    l = np.asarray(l, np.float64)
    c = np.asarray(c, np.float64)
    n = len(c)
    stop_dist = abs(entry_price - sl_price)
    if stop_dist <= 0:
        return {rr: {'hit': False, 'outcome': 'invalid', 'realized_rr': 0.0}
                for rr in rr_targets}
    targets = {rr: entry_price + stop_dist * rr * (1 if is_long else -1)
               for rr in rr_targets}
    results = {rr: {'hit': False, 'outcome': None, 'realized_rr': 0.0}
               for rr in rr_targets}
    end = n if lookahead is None else min(entry_idx + lookahead + 1, n)
    for j in range(entry_idx + 1, end):
        if is_session_end is not None and is_session_end[j]:
            fr = ((c[j] - entry_price) / stop_dist if is_long
                  else (entry_price - c[j]) / stop_dist)
            for rr in rr_targets:
                if results[rr]['outcome'] is None:
                    results[rr]['outcome'] = 'session_end'
                    results[rr]['realized_rr'] = fr
            break
        if (is_long and l[j] <= sl_price) or (not is_long and h[j] >= sl_price):
            for rr in rr_targets:
                if results[rr]['outcome'] is None:
                    results[rr]['outcome'] = 'stopped'
                    results[rr]['realized_rr'] = -1.0
            break
        for rr in sorted(rr_targets):
            if results[rr]['outcome'] is not None:
                continue
            if ((is_long and h[j] >= targets[rr])
                    or (not is_long and l[j] <= targets[rr])):
                results[rr]['hit'] = True
                results[rr]['outcome'] = 'target_hit'
                results[rr]['realized_rr'] = rr
        if all(r['outcome'] is not None for r in results.values()):
            break
    for rr in rr_targets:
        if results[rr]['outcome'] is None:
            results[rr]['outcome'] = 'data_end'
            results[rr]['realized_rr'] = 0.0
    return results
