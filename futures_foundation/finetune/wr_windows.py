"""Portable pivot-window cache — the shared backbone-EVAL ruler (build once, score many).

A backbone A/B (the Optuna sweep, the 2026 benchmark, any future checkpoint compare) needs the same
thing: raw OHLCV windows ending at each gated pivot's confirm bar, plus that pivot's triple-barrier
outcomes and explicit first-touch flags, cached to a portable .npz so every checkpoint is scored on
IDENTICAL windows with only the encoder weights changing. This module is that builder, extracted
from the sweep so it lives in ONE
certified place and every Colab eval imports it (pip-installed FFM is importable however the cell is
run — a sibling colab file is not).

Pure price/ATR, no labels, no leak, no private-repo dependency: compute_atr -> DETECTOR
(zigzag == trend_scan | fractal_zigzag == the deployed live trigger) -> causal HTF-alignment gate ->
forward triple-barrier. The npz is self-describing (v4 schema including explicit reach flags,
label-end timestamps, per-pivot direction, and trailing trend). load_wr_cache round-trips it.

    build_wr_cache(path, data_dir=..., tickers=[...], tfs=[...], seq=256, detector='fractal_zigzag')
    d = load_wr_cache(path)   # dict: win, realized[N,T], reached[N,T], peak, r3, ts, ...
"""
import os

import numpy as np
import pandas as pd

FIXED_TARGETS = (2.0, 3.0, 4.0, 6.0, 8.0)     # triple-barrier R targets (r3 = index 1)


def _triple_barrier(o, h, l, i, d, n, *, risk, cost_r, vert, targets,
                    same_bar_policy='tp_first'):
    """Forward triple-barrier from entry i+1: first-touch TP=+X / SL=-1R per target ->
    (realized R per targets, peak R before stop, explicit target-first flags). ``reached`` is kept
    separate from realized R because a positive vertical-barrier close does NOT imply that the
    target was touched. ``same_bar_policy`` resolves bars whose high touches TP while low touches
    SL; ``stop_first`` is the conservative choice when intrabar ordering is unavailable.

    risk = the 1R price distance (caller sets the stop model: ATR-multiple or structural). Pure
    price, no labels, no leak."""
    if same_bar_policy not in {'tp_first', 'stop_first'}:
        raise ValueError("same_bar_policy must be 'tp_first' or 'stop_first'")
    entry = o[i + 1]
    peak, last, t_stop = 0.0, i, None
    t_hit = [None] * len(targets)
    for j in range(i + 1, min(i + 1 + vert, n)):
        last = j
        fav = (h[j] - entry) / risk if d == 1 else (entry - l[j]) / risk
        adv = (l[j] - entry) / risk if d == 1 else (entry - h[j]) / risk
        bar_stops = adv <= -1.0
        # With unknown intrabar order, stop_first means the favorable excursion on a stop bar is
        # not observable before exit either. Keep peak aligned with the same conservative policy.
        if fav > peak and not (bar_stops and same_bar_policy == 'stop_first'):
            peak = fav
        for ti, X in enumerate(targets):
            if t_hit[ti] is None and fav >= X:
                t_hit[ti] = j
        if t_stop is None and bar_stops:
            t_stop = j
        if t_stop is not None:
            break                                           # the trade ended; never inspect later bars
    mid = (h[last] + l[last]) / 2.0
    rclose = (mid - entry) / risk if d == 1 else (entry - mid) / risk
    out = np.empty(len(targets), np.float32)
    reached = np.zeros(len(targets), np.bool_)
    for ti, X in enumerate(targets):
        hit_first = (t_hit[ti] is not None and
                     (t_stop is None or t_hit[ti] < t_stop or
                      (t_hit[ti] == t_stop and same_bar_policy == 'tp_first')))
        reached[ti] = hit_first
        if hit_first:
            out[ti] = X - cost_r                     # hit target first = win
        elif t_stop is not None:
            out[ti] = -1.0 - cost_r                  # stopped first = loss
        else:
            out[ti] = rclose - cost_r                # neither -> close
    return out, peak, reached


def build_wr_cache(path, *, data_dir, tickers, tfs, seq=64, detector='fractal_zigzag',
                   atr_period=20, min_history=128, vert=150, stop_atr=0.5, cost_r=0.03,
                   targets=FIXED_TARGETS, data_months=72, htf_gate=True, trend_n=480,
                   rev_atr=1.25, fractal_k=2, fractal_leg_atr=1.25,
                   stop_mode='atr', stop_buffer_atr=0.05, stop_cap_atr=0.0,
                   roll_guard=False, same_bar_policy='tp_first', verbose=True):
    """Build the portable window cache from RAW {ticker}_{tf}.csv OHLCV under data_dir.

    seq       : window length (64 = Mantis-native; 256/512 = the long-context ruler for a backbone
                trained on whole-trend windows — embed_windows interpolates to Mantis's 512).
    detector  : only 'fractal_zigzag' is accepted. The legacy ATR-zigzag exposed future-completed
                leg fields and was removed from this deployable cache contract.
    stop_mode : 'atr' (1R = stop_atr*ATR, DEFAULT/backward-compat) | 'structural' (1R = distance from
                entry to the pivot extreme + stop_buffer_atr*ATR — the deployed pivot-structure stop;
                stop_cap_atr>0 caps the tail). Match this to the strategy's STOP_MODE so the benchmark
                scores backbones under the SAME triple-barrier the strategy trades.
    Windows are raw OHLCV [N,5,seq] ending at each pivot's confirm bar; the encoder standardizes.
    roll_guard: require ``contract_id`` and reject any candidate whose input or full label horizon
                crosses a contract boundary. Enable for unadjusted continuous futures.
    Writes an ATOMIC v4 npz including every target's realized return and explicit reach flag plus
    each label's conservative end timestamp, which downstream splits must use to purge boundaries.
    """
    from futures_foundation.pipeline._primitives import compute_atr
    from futures_foundation.primitives.detection import detect_fractal_zigzag_pivots
    from futures_foundation.pivots import causal_htf_dir
    seq = int(seq)
    r3i = list(targets).index(3.0)
    Ws, PK, REAL, REACH, R3, TS, LE, TK, TFa, DR, TND = [], [], [], [], [], [], [], [], [], [], []
    for tk in tickers:
        for tf in tfs:
            csv = os.path.join(data_dir, f'{tk}_{tf}.csv')
            if not os.path.exists(csv):
                if verbose:
                    print(f"[cache] {tk}@{tf}: no CSV ({csv}) — skip", flush=True)
                continue
            available = set(pd.read_csv(csv, nrows=0).columns)
            contract_col = ('contract_id' if 'contract_id' in available else
                            'instrument_id' if 'instrument_id' in available else None)
            if roll_guard and contract_col is None:
                raise ValueError(f'{csv}: roll_guard requires contract_id or instrument_id')
            cols = ['datetime', 'open', 'high', 'low', 'close', 'volume']
            if contract_col:
                cols.append(contract_col)
            df = pd.read_csv(csv, usecols=cols)
            df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
            df = df[df['datetime'] >= df['datetime'].max()
                    - pd.DateOffset(months=data_months)]
            df = df.sort_values('datetime').drop_duplicates('datetime', keep='last').reset_index(drop=True)
            ts = df['datetime'].dt.tz_localize(None).to_numpy()   # naive-UTC datetime64 (npz-storable;
            #                                          load_wr_cache re-attaches UTC — clean round-trip)
            o, h, l, c = (df[k].to_numpy(float) for k in ('open', 'high', 'low', 'close'))
            v = df['volume'].to_numpy(float); n = len(c)
            segment = None
            if contract_col:
                contract = df[contract_col].astype(str).to_numpy()
                segment = np.r_[0, np.cumsum(contract[1:] != contract[:-1])].astype(np.int32)
            atr = compute_atr(h, l, c, atr_period)
            htf = causal_htf_dir({'ts': ts, 'o': o, 'h': h, 'l': l, 'c': c}, tf, ts, atr_period)
            if detector != 'fractal_zigzag':      # atr_zigzag DELETED 2026-07-16 (lookahead fields)
                raise ValueError(f'unknown detector {detector!r}; only fractal_zigzag is certified')
            pivots = detect_fractal_zigzag_pivots(o, h, l, c, k=fractal_k, min_leg_atr=fractal_leg_atr)
            w, pk, real, reach, r3, tsp, label_end, dr, tnd = [], [], [], [], [], [], [], [], []
            min_hist = max(min_history, seq)                     # long windows need seq bars of history
            for p in pivots:
                i, d = p['confirm'], p['direction']
                if i < min_hist or i + 1 + vert >= n:            # need history + full outcome window
                    continue
                if roll_guard and (segment[i - seq + 1] != segment[i] or
                                   segment[i] != segment[i + 1 + vert]):
                    continue                                     # no synthetic roll pattern/outcome
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
                realized, peak, reached = _triple_barrier(
                    o, h, l, i, d, n, risk=risk, cost_r=cost_r, vert=vert,
                    targets=targets, same_bar_policy=same_bar_policy)
                sl = slice(i - seq + 1, i + 1)                   # raw OHLCV window (no direction flip)
                w.append(np.stack([o[sl], h[sl], l[sl], c[sl], v[sl]]).astype(np.float32))
                pk.append(peak); real.append(realized); reach.append(reached)
                r3.append(float(realized[r3i])); tsp.append(ts[i])
                label_end.append(ts[i + 1 + vert])                # conservative split embargo
                # counter-trend metadata: entry direction + ATR-normalized net move over trailing
                # trend_n bars (~a day of 3min at 480) — the daily context the gate can't see.
                dr.append(d)
                tnd.append(float((c[i] - c[max(0, i - trend_n)]) / a))
            if not w:
                if verbose:
                    print(f"[cache] {tk}@{tf}: 0 pivots", flush=True)
                continue
            Ws.append(np.nan_to_num(np.stack(w))); PK.append(np.asarray(pk, np.float32))
            REAL.append(np.asarray(real, np.float32)); REACH.append(np.asarray(reach, np.bool_))
            R3.append(np.asarray(r3, np.float32)); TS.append(np.asarray(tsp, dtype='datetime64[ns]'))
            LE.append(np.asarray(label_end, dtype='datetime64[ns]'))
            TK.append(np.array([tk] * len(w))); TFa.append(np.array([tf] * len(w)))
            DR.append(np.asarray(dr, np.int8)); TND.append(np.asarray(tnd, np.float32))
            if verbose:
                print(f"[cache] {tk}@{tf}: {len(w)} pivots  win({len(w)}, 5, {seq})  trig={detector}",
                      flush=True)
    if not Ws:
        raise RuntimeError(f"build_wr_cache: no pivots for any {tickers} x {tfs} under {data_dir}")
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    tmp = str(path) + '.tmp.npz'                     # ATOMIC write: a disconnect mid-save can never
    np.savez(tmp, win=np.concatenate(Ws), peak=np.concatenate(PK),
             realized=np.concatenate(REAL), reached=np.concatenate(REACH),
             targets=np.asarray(targets, np.float32), r3=np.concatenate(R3),
             ts=np.concatenate(TS), label_end=np.concatenate(LE),
             tk=np.concatenate(TK), tf=np.concatenate(TFa),
             dir=np.concatenate(DR), trend=np.concatenate(TND),
             meta_version=np.int64(4),
             meta_seq=np.int64(seq), meta_detector=np.array(detector),
             meta_stop=np.array(stop_mode), meta_tickers=np.asarray(tickers),
             meta_tfs=np.asarray(tfs), meta_vert=np.int64(vert),
             meta_data_months=np.int64(data_months), meta_roll_guard=np.bool_(roll_guard),
             meta_htf_gate=np.bool_(htf_gate), meta_atr_period=np.int64(atr_period),
             meta_min_history=np.int64(min_history), meta_trend_n=np.int64(trend_n),
             meta_rev_atr=np.float64(rev_atr), meta_fractal_k=np.int64(fractal_k),
             meta_fractal_leg_atr=np.float64(fractal_leg_atr),
             meta_stop_atr=np.float64(stop_atr), meta_stop_buffer_atr=np.float64(stop_buffer_atr),
             meta_stop_cap_atr=np.float64(stop_cap_atr), meta_cost_r=np.float64(cost_r),
             meta_same_bar_policy=np.array(same_bar_policy))
    os.replace(tmp, path)                            # never leaves a half-written npz at the final path
    total = sum(len(x) for x in R3)
    if verbose:
        print(f"[cache] wrote {path}  total pivots={total}  seq={seq} trig={detector} stop={stop_mode}",
              flush=True)
    return path


def load_wr_cache(path):
    """Load a portable window cache -> dict. Pure numpy (no private-repo dependency). Older v1/v2
    caches without meta/dir/trend/label_end load with those keys as None."""
    d = np.load(path, allow_pickle=True)
    out = {
        'win': d['win'], 'peak': d['peak'].astype(np.float32), 'r3': d['r3'].astype(np.float32),
        'realized': d['realized'].astype(np.float32) if 'realized' in d.files else None,
        'reached': d['reached'].astype(bool) if 'reached' in d.files else None,
        'targets': d['targets'].astype(np.float32) if 'targets' in d.files else None,
        'ts': pd.to_datetime(d['ts'], utc=True), 'tk': d['tk'], 'tf': d['tf'],
        'label_end': (pd.to_datetime(d['label_end'], utc=True) if 'label_end' in d.files else None),
        'dir': d['dir'].astype(np.int8) if 'dir' in d.files else None,
        'trend': d['trend'].astype(np.float32) if 'trend' in d.files else None,
        'seq': int(d['meta_seq']) if 'meta_seq' in d.files else int(d['win'].shape[2]),
        'detector': str(d['meta_detector']) if 'meta_detector' in d.files else 'zigzag',
        'stop': str(d['meta_stop']) if 'meta_stop' in d.files else 'atr',
        'meta': {
            'version': int(d['meta_version']) if 'meta_version' in d.files else 1,
            'tickers': d['meta_tickers'].tolist() if 'meta_tickers' in d.files else None,
            'tfs': d['meta_tfs'].tolist() if 'meta_tfs' in d.files else None,
            'vert': int(d['meta_vert']) if 'meta_vert' in d.files else None,
            'data_months': int(d['meta_data_months']) if 'meta_data_months' in d.files else None,
            'roll_guard': bool(d['meta_roll_guard']) if 'meta_roll_guard' in d.files else False,
            'htf_gate': bool(d['meta_htf_gate']) if 'meta_htf_gate' in d.files else None,
            'atr_period': int(d['meta_atr_period']) if 'meta_atr_period' in d.files else None,
            'min_history': int(d['meta_min_history']) if 'meta_min_history' in d.files else None,
            'trend_n': int(d['meta_trend_n']) if 'meta_trend_n' in d.files else None,
            'rev_atr': float(d['meta_rev_atr']) if 'meta_rev_atr' in d.files else None,
            'fractal_k': int(d['meta_fractal_k']) if 'meta_fractal_k' in d.files else None,
            'fractal_leg_atr': (float(d['meta_fractal_leg_atr'])
                                if 'meta_fractal_leg_atr' in d.files else None),
            'stop_atr': float(d['meta_stop_atr']) if 'meta_stop_atr' in d.files else None,
            'stop_buffer_atr': (float(d['meta_stop_buffer_atr'])
                                if 'meta_stop_buffer_atr' in d.files else None),
            'stop_cap_atr': (float(d['meta_stop_cap_atr'])
                             if 'meta_stop_cap_atr' in d.files else None),
            'cost_r': float(d['meta_cost_r']) if 'meta_cost_r' in d.files else None,
            'same_bar_policy': (str(d['meta_same_bar_policy'])
                                if 'meta_same_bar_policy' in d.files else 'tp_first'),
        },
    }
    return out
