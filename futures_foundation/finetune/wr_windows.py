"""Portable pivot-window cache — the shared backbone-EVAL ruler (build once, score many).

A backbone A/B (the Optuna sweep, the 2026 benchmark, any future checkpoint compare) needs the same
thing: raw OHLCV windows ending at each gated pivot's confirm bar, plus that pivot's triple-barrier
outcome, cached to a portable .npz so every checkpoint is scored on IDENTICAL windows with only the
encoder weights changing. This module is that builder, extracted from the sweep so it lives in ONE
certified place and every Colab eval imports it (pip-installed FFM is importable however the cell is
run — a sibling colab file is not).

Pure price/ATR, no labels, no leak, no private-repo dependency: compute_atr -> DETECTOR
(zigzag == trend_scan | fractal_zigzag == the deployed live trigger) -> causal HTF-alignment gate ->
forward triple-barrier. The npz is self-describing (v2 schema incl. per-pivot direction + trailing
trend for the counter-trend split). load_wr_cache round-trips it.

    build_wr_cache(path, data_dir=..., tickers=[...], tfs=[...], seq=256, detector='fractal_zigzag')
    d = load_wr_cache(path)   # dict: win[N,5,seq], peak, r3, ts, tk, tf, dir, trend, meta
"""
import os

import numpy as np
import pandas as pd

FIXED_TARGETS = (2.0, 3.0, 4.0, 6.0)     # triple-barrier R targets (r3 = index 1)


def _triple_barrier(o, h, l, i, d, n, *, risk, cost_r, vert, targets):
    """Forward triple-barrier from entry i+1: first-touch TP=+X / SL=-1R per target ->
    (realized R per targets, peak R). risk = the 1R price distance (caller sets the stop model:
    ATR-multiple or structural). Pure price, no labels, no leak."""
    entry = o[i + 1]
    peak, last, t_stop = 0.0, i, None
    t_hit = [None] * len(targets)
    for j in range(i + 1, min(i + 1 + vert, n)):
        last = j
        fav = (h[j] - entry) / risk if d == 1 else (entry - l[j]) / risk
        adv = (l[j] - entry) / risk if d == 1 else (entry - h[j]) / risk
        if fav > peak:
            peak = fav
        for ti, X in enumerate(targets):
            if t_hit[ti] is None and fav >= X:
                t_hit[ti] = j
        if t_stop is None and adv <= -1.0:
            t_stop = j
        if t_stop is not None and all(t is not None for t in t_hit):
            break
    mid = (h[last] + l[last]) / 2.0
    rclose = (mid - entry) / risk if d == 1 else (entry - mid) / risk
    out = np.empty(len(targets), np.float32)
    for ti, X in enumerate(targets):
        if t_hit[ti] is not None and (t_stop is None or t_hit[ti] <= t_stop):
            out[ti] = X - cost_r                     # hit target first = win
        elif t_stop is not None and (t_hit[ti] is None or t_stop < t_hit[ti]):
            out[ti] = -1.0 - cost_r                  # stopped first = loss
        else:
            out[ti] = rclose - cost_r                # neither -> close
    return out, peak


def build_wr_cache(path, *, data_dir, tickers, tfs, seq=64, detector='zigzag',
                   atr_period=20, min_history=128, vert=150, stop_atr=0.5, cost_r=0.03,
                   targets=FIXED_TARGETS, data_months=72, htf_gate=True, trend_n=480,
                   rev_atr=1.25, fractal_k=2, fractal_leg_atr=1.25,
                   stop_mode='atr', stop_buffer_atr=0.05, stop_cap_atr=0.0, verbose=True):
    """Build the portable window cache from RAW {ticker}_{tf}.csv OHLCV under data_dir.

    seq       : window length (64 = Mantis-native; 256/512 = the long-context ruler for a backbone
                trained on whole-trend windows — embed_windows interpolates to Mantis's 512).
    detector  : 'zigzag' (ATR-zigzag == trend_scan, the historical ruler) | 'fractal_zigzag'
                (the DEPLOYED live trigger — use this to score a backbone on the pool it will trade).
    stop_mode : 'atr' (1R = stop_atr*ATR, DEFAULT/backward-compat) | 'structural' (1R = distance from
                entry to the pivot extreme + stop_buffer_atr*ATR — the deployed pivot-structure stop;
                stop_cap_atr>0 caps the tail). Match this to the strategy's STOP_MODE so the benchmark
                scores backbones under the SAME triple-barrier the strategy trades.
    Windows are raw OHLCV [N,5,seq] ending at each pivot's confirm bar; the encoder standardizes.
    Writes an ATOMIC v2 npz: win, peak, r3, ts, tk, tf, dir, trend (+ meta: seq/detector/tickers/tfs).
    """
    from futures_foundation.pipeline._primitives import compute_atr
    from futures_foundation.primitives.detection import (
        detect_atr_zigzag_pivots, detect_fractal_zigzag_pivots)
    from futures_foundation.pivots import causal_htf_dir
    seq = int(seq)
    r3i = list(targets).index(3.0)
    Ws, PK, R3, TS, TK, TFa, DR, TND = [], [], [], [], [], [], [], []
    for tk in tickers:
        for tf in tfs:
            csv = os.path.join(data_dir, f'{tk}_{tf}.csv')
            if not os.path.exists(csv):
                if verbose:
                    print(f"[cache] {tk}@{tf}: no CSV ({csv}) — skip", flush=True)
                continue
            df = pd.read_csv(csv, usecols=['datetime', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
            df = df[df['datetime'] >= df['datetime'].max()
                    - pd.DateOffset(months=data_months)].reset_index(drop=True)
            ts = df['datetime'].dt.tz_localize(None).to_numpy()   # naive-UTC datetime64 (npz-storable;
            #                                          load_wr_cache re-attaches UTC — clean round-trip)
            o, h, l, c = (df[k].to_numpy(float) for k in ('open', 'high', 'low', 'close'))
            v = df['volume'].to_numpy(float); n = len(c)
            atr = compute_atr(h, l, c, atr_period)
            htf = causal_htf_dir({'ts': ts, 'o': o, 'h': h, 'l': l, 'c': c}, tf, ts, atr_period)
            pivots = (detect_fractal_zigzag_pivots(o, h, l, c, k=fractal_k, min_leg_atr=fractal_leg_atr)
                      if detector == 'fractal_zigzag'
                      else detect_atr_zigzag_pivots(o, h, l, c, atr_period=atr_period, rev_atr=rev_atr))
            w, pk, r3, tsp, dr, tnd = [], [], [], [], [], []
            min_hist = max(min_history, seq)                     # long windows need seq bars of history
            for p in pivots:
                i, d = p['confirm'], p['direction']
                if i < min_hist or i + 1 + vert >= n:            # need history + full outcome window
                    continue
                a = atr[i]
                if not (np.isfinite(a) and a > 0):
                    continue
                if htf_gate and int(htf[i]) != d:                # pivot must align with the HTF trend
                    continue
                # 1R price distance — the stop model (PARITY with pivot_trend_mantis _fixed_outcomes)
                if stop_mode == 'structural':
                    oi = int(p['origin'])                        # the pivot extreme bar (< i)
                    if oi < 0:
                        continue
                    entry = o[i + 1]
                    ext = l[oi] if d == 1 else h[oi]             # pivot low (long) / high (short)
                    buf = stop_buffer_atr * a
                    risk = (entry - (ext - buf)) if d == 1 else ((ext + buf) - entry)
                    if not (risk > 0):                           # entry already beyond the pivot -> skip
                        continue
                    if stop_cap_atr > 0:
                        risk = min(risk, stop_cap_atr * a)
                else:
                    risk = stop_atr * a                          # backward-compat default
                realized, peak = _triple_barrier(o, h, l, i, d, n, risk=risk,
                                                 cost_r=cost_r, vert=vert, targets=targets)
                sl = slice(i - seq + 1, i + 1)                   # raw OHLCV window (no direction flip)
                w.append(np.stack([o[sl], h[sl], l[sl], c[sl], v[sl]]).astype(np.float32))
                pk.append(peak); r3.append(float(realized[r3i])); tsp.append(ts[i])
                # counter-trend metadata: entry direction + ATR-normalized net move over trailing
                # trend_n bars (~a day of 3min at 480) — the daily context the gate can't see.
                dr.append(d)
                tnd.append(float((c[i] - c[max(0, i - trend_n)]) / a))
            if not w:
                if verbose:
                    print(f"[cache] {tk}@{tf}: 0 pivots", flush=True)
                continue
            Ws.append(np.nan_to_num(np.stack(w))); PK.append(np.asarray(pk, np.float32))
            R3.append(np.asarray(r3, np.float32)); TS.append(np.asarray(tsp, dtype='datetime64[ns]'))
            TK.append(np.array([tk] * len(w))); TFa.append(np.array([tf] * len(w)))
            DR.append(np.asarray(dr, np.int8)); TND.append(np.asarray(tnd, np.float32))
            if verbose:
                print(f"[cache] {tk}@{tf}: {len(w)} pivots  win({len(w)}, 5, {seq})  trig={detector}",
                      flush=True)
    if not Ws:
        raise RuntimeError(f"build_wr_cache: no pivots for any {tickers} x {tfs} under {data_dir}")
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    tmp = str(path) + '.tmp.npz'                     # ATOMIC write: a disconnect mid-save can never
    np.savez(tmp, win=np.concatenate(Ws), peak=np.concatenate(PK), r3=np.concatenate(R3),
             ts=np.concatenate(TS), tk=np.concatenate(TK), tf=np.concatenate(TFa),
             dir=np.concatenate(DR), trend=np.concatenate(TND),
             meta_seq=np.int64(seq), meta_detector=np.array(detector),
             meta_stop=np.array(stop_mode))              # self-describing v2 (seq/detector/stop)
    os.replace(tmp, path)                            # never leaves a half-written npz at the final path
    total = sum(len(x) for x in R3)
    if verbose:
        print(f"[cache] wrote {path}  total pivots={total}  seq={seq} trig={detector} stop={stop_mode}",
              flush=True)
    return path


def load_wr_cache(path):
    """Load a portable window cache -> dict. Pure numpy (no private-repo dependency). Older v1/v2
    caches without meta/dir/trend load with those keys as None."""
    d = np.load(path, allow_pickle=True)
    out = {
        'win': d['win'], 'peak': d['peak'].astype(np.float32), 'r3': d['r3'].astype(np.float32),
        'ts': pd.to_datetime(d['ts'], utc=True), 'tk': d['tk'], 'tf': d['tf'],
        'dir': d['dir'].astype(np.int8) if 'dir' in d.files else None,
        'trend': d['trend'].astype(np.float32) if 'trend' in d.files else None,
        'seq': int(d['meta_seq']) if 'meta_seq' in d.files else int(d['win'].shape[2]),
        'detector': str(d['meta_detector']) if 'meta_detector' in d.files else 'zigzag',
        'stop': str(d['meta_stop']) if 'meta_stop' in d.files else 'atr',
    }
    return out
