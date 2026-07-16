"""Final-model training + held-out OOS — the 'produce' half (mirrors pipeline.produce).

Two modes:
  stream=False (default): featurize the train/val/oos windows into RAM arrays. Fine for
    small/medium sets and the torch-free tests.
  stream=True (the REAL full-data run): featurize to DISK memmaps in chunks (parent RAM
    = one chunk) and train PER-BATCH from the memmaps (worker RAM = one batch), with
    per-channel standardize stats applied per-batch. Lets a transformer train on the
    FULL aligned-pivot set (no 5.6GB array in RAM) — the data it needs to not overfit.

Either way: train on all data < holdout_start (inner val + early-stop), score the
held-out 2026 OOS once with a SHUFFLE control, and (export_onnx) emit <base>.onnx +
<base>_signal.json (input spec, channel names, standardize mu/sd, oos metrics, sha).
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .classifier import get_classifier
from .wf import (_pct_threshold, _arm_R, _meanR, _standardize_on_train,
                 OP_PERCENTILE, PASS_LIFT_MARGIN_R)


def _oos_days(ts):
    return max(1, int(pd.DatetimeIndex(ts).normalize().nunique())) if len(ts) else 1


def operating_points(eval_lab, keys, proba, ts, rates=(5, 3, 2, 1)):
    """OOS quality at DEPLOY operating points (CUMULATIVE): rank all OOS pivots by score, take the
    top N = rate * trading_days, report WR@3R + meanR + count at each per-day trade rate. The honest
    read of the signal-count floor ('1-2 A+ trades/day, not per week') — the pooled top-50% meanR
    proves the edge; THIS shows the quality at the volume you deploy at. Returns list of dicts."""
    proba = np.asarray(proba, float)
    if len(proba) == 0:
        return []
    R_all = np.asarray(eval_lab.evaluate(list(keys), np.ones(len(keys), int)), float)  # take-all R
    days = _oos_days(ts)
    order = np.argsort(-proba)                             # best pivots first
    rows = []
    for r in rates:
        n = int(min(len(proba), max(1, round(r * days))))
        sel = order[:n]
        Rs = R_all[sel]
        rows.append(dict(rate=r, n=n, days=days, wr3R=float((Rs > 0).mean()),
                         meanR=float(Rs.mean()),
                         thresh=float(proba[sel[-1]])))     # the score cutoff -> 'enter if score>=thresh'
    return rows


def wr_by_score(eval_lab, keys, proba, ts, edges=(0.90, 0.75, 0.50, 0.25, 0.0)):
    """OOS WR@3R broken down by MODEL SCORE band (NON-cumulative) — 'is a higher score actually a
    better trade?'. Splits pivots into score quantile bands (top 10% / 10-25% / 25-50% / 50-75% /
    bottom 25%) and reports per-band WR@3R + meanR + count + trades/day. A monotone WR down the
    bands = the ranking is real and the top band is where the A+ signals live. Score = the ranking
    output (calibrated proba for the single head; expected-R for the ladder). Returns list of dicts."""
    proba = np.asarray(proba, float)
    if len(proba) == 0:
        return []
    R_all = np.asarray(eval_lab.evaluate(list(keys), np.ones(len(keys), int)), float)
    days = _oos_days(ts)
    qs = [float(np.quantile(proba, e)) for e in edges]     # score thresholds (desc)
    rows, hi = [], np.inf
    labels = ['top10%', '10-25%', '25-50%', '50-75%', 'bot25%'][:len(qs)]
    for lab, lo in zip(labels, qs):
        m = (proba < hi) & (proba >= lo)
        Rs = R_all[m]
        rows.append(dict(band=lab, lo=lo, n=int(m.sum()), per_day=float(m.sum()) / days,
                         wr3R=(float((Rs > 0).mean()) if len(Rs) else float('nan')),
                         meanR=(float(Rs.mean()) if len(Rs) else float('nan'))))
        hi = lo
    return rows


def alignment_breakdown(eval_lab, keys, proba, ts):
    """THE sighted-counter-trend readout: split the OOS pivots by HTF alignment (labeler's
    htf_alignment: -1 = counter the HTF trend — the fades a hard gate blindly discards, +1 = with)
    and report each group's BASE WR@3R vs its WR at the model's top scores (per-day operating
    points on the group). The hypothesis check: WR rising steeply with score AMONG counter pivots
    = the model can now tell the fade-that-reverses from the fade-that-gets-run-over. None if the
    labeler doesn't expose htf_alignment."""
    fn = getattr(eval_lab, 'htf_alignment', None)
    if fn is None or ts is None or not len(keys):
        return None
    a = np.asarray(fn(list(keys)))
    proba = np.asarray(proba, float)
    out = {}
    for name, m in (('counter', a < 0), ('aligned', a > 0)):
        if int(m.sum()) < 50:
            continue
        ks = [k for k, mm in zip(keys, m) if mm]
        ts_s = [t for t, mm in zip(ts, m) if mm]
        R = np.asarray(eval_lab.evaluate(ks, np.ones(len(ks), int)), float)
        out[name] = dict(n=int(m.sum()), base_wr3R=float((R > 0).mean()),
                         base_meanR=float(R.mean()),
                         ops=operating_points(eval_lab, ks, proba[m], ts_s))
    return out or None


def _print_alignment(ab):
    if not ab:
        return
    print("  2026 OOS — COUNTER-TREND readout (WR@3R by model score WITHIN each HTF-alignment "
          "group; counter rising with score = SIGHTED soft-gating):", flush=True)
    for name, g in ab.items():
        print(f"    {name:>7}: n={g['n']:,}  base WR@3R={g['base_wr3R']:.1%}  "
              f"base meanR={g['base_meanR']:+.3f}", flush=True)
        for r in g['ops']:
            print(f"      ~{r['rate']}/day: n={r['n']:>5}  score>={r.get('thresh', float('nan')):.3f}  "
                  f"WR@3R={r['wr3R']:6.1%}  meanR={r['meanR']:+.3f}", flush=True)


def _print_operating_points(op_rows, band_rows, title='2026 OOS'):
    if band_rows:
        print(f"  {title} — WR@3R by score band (non-cumulative; monotone WR down the bands = the "
              "ranking is real, top band = the A+ signals):", flush=True)
        print(f"    {'band':>7} {'n':>6} {'trades/day':>11} {'WR@3R':>8} {'meanR':>8}", flush=True)
        for b in band_rows:
            print(f"    {b['band']:>7} {b['n']:>6} {b['per_day']:>11.2f} "
                  f"{b['wr3R']:>8.1%} {b['meanR']:>+8.3f}", flush=True)
    if op_rows:
        print(f"  {title} — CUMULATIVE operating points (enter if score>=thresh, {op_rows[0]['days']} "
              "OOS days):", flush=True)
        for r in op_rows:
            print(f"    ~{r['rate']}/day: n={r['n']:>5}  score>={r.get('thresh', float('nan')):.3f}  "
                  f"WR@3R={r['wr3R']:6.1%}  meanR={r['meanR']:+.3f}", flush=True)


def _contract(labeler, classifier, ck, C, seq, mu, sd, out, onnx_path, sha,
              holdout_start, n_train, n_oos):
    tfs = sorted({k[1] for k in getattr(labeler, '_b', {})}) or None
    tks = sorted({k[0] for k in getattr(labeler, '_b', {})}) or None
    return {
        'contract_version': '1.0', 'role': 'signal', 'classifier': classifier,
        'input': {'channels': int(C), 'seq_len': int(seq),
                  'mv_mode': getattr(labeler, 'MV_MODE', None)},
        'channel_names': (labeler.mv_feature_names()
                          if hasattr(labeler, 'mv_feature_names') else None),
        'standardize': ({'mu': np.asarray(mu).tolist(), 'sd': np.asarray(sd).tolist()}
                        if mu is not None else None),
        'mv_contexts_fn': 'strategy.mv_contexts (direction-normalized causal window)',
        'nan_policy': {'posinf': 0, 'neginf': 0, 'nan': 0},
        'ft_config': {k: v for k, v in (ck or {}).items()
                      if k not in ('standardize_mu', 'standardize_sd', 'log_path')},
        'n_classes': 2, 'proba_meaning': 'P(good trend pivot reaches target before stop)',
        'output_fn': 'softmax(logits)[:,1]',
        'train_scope': {'tickers': tks, 'timeframes': tfs, 'holdout_start': holdout_start,
                        'n_train': int(n_train), 'n_oos': int(n_oos)},
        'oos_metrics': {k: out.get(k) for k in
                        ('oos_auc', 'oos_meanR', 'shuffle_meanR', 'edge_shuffle', 'oos_trades')},
        'onnx': (Path(onnx_path).name if sha else None), 'content_sha': sha,
    }


def _bundle_files(base):
    """Existing ONNX artifact files for a given export base (single-file classifiers write
    <base>.onnx; the frozen bundle writes <base>_encoder.onnx + <base>_signal_head.onnx)."""
    return [p for p in (str(base) + '.onnx', str(base) + '_encoder.onnx',
                        str(base) + '_signal_head.onnx') if Path(p).exists()]


def _emit(out, classifier, ck, eval_lab, mu, sd, C, seq,
          channel_names, tks, tfs, holdout_start, export_onnx, output_path, verbose):
    """Write the deploy contract for the ONNX exported DURING the REAL fit (no refit — a
    third full fit doubled peak RAM and OOM-killed the Colab session at produce scale)."""
    if not (export_onnx and output_path):
        return out
    base = Path(output_path).with_suffix('')
    bundle = _bundle_files(base)
    sha = (hashlib.sha256(b''.join(Path(p).read_bytes() for p in bundle)).hexdigest()
           if bundle else None)
    dist = (ck or {}).get('rank') == 'expected_reach'     # reach-LADDER head (p_3r entry signal)
    mv_agg = tuple(getattr(eval_lab, 'MV_AGG', ()) or ())  # multi-TF factors the labeler declares
    contract = {
        'contract_version': '1.0', 'role': 'signal', 'classifier': classifier,
        'input': {'channels': int(C), 'seq_len': int(seq),
                  'mv_mode': getattr(eval_lab, 'MV_MODE', None)},
        # MULTI-TF WINDOW RECIPE — how the bot must BUILD the encoder input. For each factor f the
        # window is the last MV_SEQ*f bars ENDING at the signal bar, each consecutive f-bar block
        # aggregated to one candle (O=first, H=max, L=min, C=last, V=sum — anchored at the signal
        # bar, NOT clock-aligned), stacked on the channel axis in factor order. factors=[1] or
        # absent = the plain single-TF window (backward-compatible).
        'window_recipe': ({'factors': [int(f) for f in mv_agg],
                           'mv_seq': int(getattr(eval_lab, 'MV_SEQ', seq) or seq),
                           'aggregation': 'O=first H=max L=min C=last V=sum; anchored at signal bar',
                           'channel_order': 'per factor: [O,H,L,C,V], factors ascending',
                           'ref': 'futures_foundation.primitives.multi_scale_ohlcv_window'}
                          if mv_agg and tuple(mv_agg) != (1,) else None),
        'channel_names': channel_names,
        'standardize': ({'mu': np.asarray(mu).tolist(), 'sd': np.asarray(sd).tolist()}
                        if mu is not None else None),
        'mv_contexts_fn': 'strategy.mv_contexts (direction-normalized causal window)',
        'nan_policy': {'posinf': 0, 'neginf': 0, 'nan': 0},
        'ft_config': {k: v for k, v in (ck or {}).items()
                      if k not in ('standardize_mu', 'standardize_sd', 'log_path')},
        'n_classes': 2,
        'train_scope': {'tickers': tks, 'timeframes': tfs, 'holdout_start': holdout_start,
                        'n_train': int(out['n_train']), 'n_oos': int(out['n_oos'])},
        'oos_metrics': {k: out.get(k) for k in
                        ('oos_auc', 'oos_meanR', 'shuffle_meanR', 'edge_shuffle', 'oos_trades')},
        'onnx': ([Path(p).name for p in bundle] if sha else None), 'content_sha': sha,
    }
    if dist:
        # reach-LADDER head: the signal_head.onnx emits p_3r (calibrated P(reach>=3R) = the ENTRY
        # signal the bot thresholds) + expected_reach (area-under-survival ranking). Per-rung Platt
        # is BAKED INTO the graph, so the bot reads p_3r directly (no post-calibration step).
        contract.update({
            'head_type': 'reach_ladder',
            'reach_targets': list(ck.get('reach_targets', [])),
            'head_outputs': ['p_3r', 'expected_reach'],
            # DEPLOY: the ENTRY signal is expected_reach (what the WF/produce validated -> the 80%
            # WR@3R tiers). The bot ENTERS when expected_reach >= T; entry_thresholds gives ready T's
            # per quality tier (val-derived, leak-free). p_3r is exported too (calibrated P(>=3R)) but
            # is NOT the validated entry signal — it's ~ the single-head proba.
            'entry_signal': 'expected_reach',
            'entry_rule': 'enter if expected_reach >= entry_thresholds[tier]',
            'entry_thresholds': out.get('entry_thresholds'),
            'proba_meaning': 'expected_reach = E[peak favorable R] (area under the calibrated '
                             'survival curve) = the ENTRY ranking score; p_3r = calibrated P(reach>=3R)',
            'calibration': {'method': 'platt', 'baked_into_onnx': True,
                            'note': 'per-rung Platt baked into signal_head.onnx; read outputs as-is'},
            'output_fn': 'signal_head.onnx -> {p_3r, expected_reach}',
        })
    else:
        contract.update({
            'proba_meaning': ('P(reach target before stop) — Platt-CALIBRATED to the empirical hit '
                              'rate' if out.get('platt') else 'P(good trend pivot reaches target)'),
            # 2026-07-16 (the standard-proba-range fix): calibration is BAKED INTO the head graph
            # (same convention as the ladder head) — signal_head.onnx's 'probabilities' output IS
            # the calibrated proba; the bot reads it as-is, NO post-step. A/B kept for reference.
            'calibration': ({'method': 'platt', 'baked_into_onnx': True,
                             'formula': 'sigmoid(A*logit(p_raw)+B)',
                             'A': out['platt'][0], 'B': out['platt'][1],
                             'note': 'baked into signal_head.onnx; read the output as-is'}
                            if out.get('platt') else None),
            'output_fn': ('signal_head.onnx -> probabilities (CALIBRATED)'
                          if out.get('platt') else 'softmax(logits)[:,1]'),
            # DEPLOY THRESHOLDS (the remap) — calibrated-proba cutoffs per quality tier, from the
            # VAL distribution (leak-free). The bot enters when calibrated P >= T.
            'entry_rule': 'enter if calibrated_proba >= entry_thresholds[tier]',
            'entry_thresholds': out.get('entry_thresholds'),
        })
    cpath = str(base) + '_signal.json'
    Path(cpath).write_text(json.dumps(contract, indent=2))
    out['artifacts'] = {'onnx': (bundle if sha else None), 'contract': cpath, 'content_sha': sha}
    if verbose:
        print(f"  artifacts: onnx={out['artifacts']['onnx']} contract={cpath}")
    return out


def _fit_score(classifier, ck, eval_lab, Xtr, Ytr_tr, Xval, Ytr_va, Xte, Kte, Yte, seed, verbose,
               onnx_path=None, keys_tr=None, keys_val=None, oos_ts=None):
    """Fit REAL (exporting ONNX during that fit when onnx_path is set) + SHUFFLE control.
    Two fits total — the export must ride the REAL fit; a separate export refit doubles
    peak RAM (fresh standardized copies on top of allocator creep) and OOMs at full scale.

    keys_tr/keys_val (distributional reach-ladder produce) carry the per-target labels; the
    SHUFFLE control permutes them in lockstep with the label (else the ladder is untouched and the
    control collapses onto REAL). oos_ts (OOS key timestamps) -> the per-day operating-point table
    (WR@3R at 5/3/2/1 trades/day) so the deploy signal-count floor is read directly."""
    import gc
    dist = (ck or {}).get('rank') == 'expected_reach'
    rng = np.random.default_rng(seed)
    if verbose:
        print(f"  [produce 1/2] fit REAL head{' (+ ONNX export)' if onnx_path else ''}",
              flush=True)
    ck_real = dict(ck, export_onnx_path=onnx_path) if onnx_path else ck
    clf_real = get_classifier(classifier, **ck_real)          # keep the instance: it holds _platt
    p_val, p_te, ba = clf_real.fit_predict(Xtr, Ytr_tr, Xval, Ytr_va, Xte, seed,
                                           keys_tr=keys_tr, keys_val=keys_val)
    gc.collect()
    thr = _pct_threshold(p_val, OP_PERCENTILE)
    R = _arm_R(eval_lab, Kte, p_te, thr)
    # SHUFFLE control = the produce-side honest ruler. SKIP_SHUFFLE=1 skips it (halves produce time)
    # when the WF ALREADY ran the honest ruler on this config; REAL + calibration are unaffected.
    if os.environ.get('SKIP_SHUFFLE') == '1':
        Rs = None
        if verbose:
            print("  [produce 2/2] SKIPPED (SKIP_SHUFFLE=1; honest ruler already run in WF)", flush=True)
    else:
        if dist and keys_tr is not None:
            perm = rng.permutation(len(Ytr_tr))
            ysh = np.asarray(Ytr_tr)[perm]; Ksh = [keys_tr[i] for i in perm]
        else:
            ysh = np.asarray(Ytr_tr).copy(); rng.shuffle(ysh); Ksh = None
        if verbose:
            print("  [produce 2/2] fit SHUFFLE control (honest ruler)", flush=True)
        psv, ps, _ = get_classifier(classifier, **ck).fit_predict(Xtr, ysh, Xval, Ytr_va, Xte, seed,
                                                                  keys_tr=Ksh, keys_val=keys_val)
        gc.collect()
        Rs = _arm_R(eval_lab, Kte, ps, _pct_threshold(psv, OP_PERCENTILE))
    auc = None
    if len(np.unique(Yte)) == 2:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(Yte, p_te))
    edge = (_meanR(R) - _meanR(Rs)) if Rs is not None else None
    # WR@3R by score band + trades/day at deploy operating points (the '1-2 A+ trades/day' read).
    bands = wr_by_score(eval_lab, Kte, p_te, oos_ts) if oos_ts is not None else []
    ops = operating_points(eval_lab, Kte, p_te, oos_ts) if oos_ts is not None else []
    align = alignment_breakdown(eval_lab, Kte, p_te, oos_ts)  # sighted-counter-trend readout
    # TIER-LEVEL SHUFFLE RULER (2026-07-16): the pool-level beats_shuffle compares REAL vs
    # SHUFFLE over the top ~half of the pool — a scale where fixed-R constructions cap the edge
    # below the pass margin for ANY model (oracle-proven), so it fails unconditionally for
    # top-tier instruments. The honest ruler compares the arms WHERE THE MODEL TRADES: the same
    # per-day operating points. PASS = REAL meanR beats SHUFFLE meanR by PASS_LIFT_MARGIN_R at
    # EVERY rate (and a shuffle tier sitting near the take-all floor = no artifact rescue).
    ops_sh = (operating_points(eval_lab, Kte, ps, oos_ts)
              if (Rs is not None and oos_ts is not None) else [])
    beats_shuffle_tiers = None
    if ops and ops_sh:
        beats_shuffle_tiers = all(
            (r['meanR'] - s['meanR']) >= PASS_LIFT_MARGIN_R
            for r, s in zip(ops, ops_sh) if r['rate'] == s['rate'])
    # PER-TICKER operating points (the deploy question is per-instrument: "NQ at N/day") —
    # rank WITHIN each ticker's candidates, same forward-threshold semantics.
    per_tk = {}
    if oos_ts is not None and len(Kte):
        _tks = np.array([str(k[0]).split('@')[0] for k in Kte])
        _pa = np.asarray(p_te, float)
        for _t in sorted(set(_tks)):
            m = _tks == _t
            if int(m.sum()) < 50:
                continue
            ks = [k for k, mm in zip(Kte, m) if mm]
            ts_s = [t for t, mm in zip(oos_ts, m) if mm]
            per_tk[_t] = operating_points(eval_lab, ks, _pa[m], ts_s, rates=(1, 2, 3, 4))
    out = dict(oos_auc=auc, best_val_auc=ba, oos_meanR=_meanR(R),
               shuffle_meanR=(_meanR(Rs) if Rs is not None else None), edge_shuffle=edge,
               n_train=len(Ytr_tr), n_oos=len(Kte), oos_trades=int(len(R)),
               beats_shuffle=(bool(edge >= PASS_LIFT_MARGIN_R) if edge is not None else None),
               wr_by_score=bands, operating_points=ops, wr_by_alignment=align,
               per_ticker_ops=per_tk, shuffle_operating_points=ops_sh,
               beats_shuffle_tiers=beats_shuffle_tiers,
               entry_thresholds=getattr(clf_real, '_entry_thresholds', None),   # val-derived T's
               platt=getattr(clf_real, '_platt', None))       # Platt (A,B) -> deploy contract
    if verbose:
        _print_operating_points(ops, bands)
        if per_tk:
            print("  OOS — PER-TICKER operating points (rank within the ticker; the "
                  "'N/day per instrument' deploy read):", flush=True)
            for _t, rows in per_tk.items():
                cells = '  '.join(f"{r['rate']}/d {r['wr3R']:5.1%} {r['meanR']:+.2f}" for r in rows)
                print(f"    {_t:>4}: {cells}", flush=True)
        _print_alignment(align)
        print(f"  OOS AUC {auc:.4f}" if auc is not None else "  OOS AUC n/a")
        if edge is not None:
            print(f"  OOS meanR REAL {out['oos_meanR']:+.3f} SHUFFLE {out['shuffle_meanR']:+.3f} "
                  f"edge {edge:+.3f} (trades={out['oos_trades']})  -> "
                  f"{'beats SHUFFLE' if out['beats_shuffle'] else 'does NOT beat SHUFFLE'}"
                  f"  [POOL-scale ruler: construction-capped; deploy verdict = the TIER ruler]")
        else:
            print(f"  OOS meanR REAL {out['oos_meanR']:+.3f} (SHUFFLE skipped; trades={out['oos_trades']})")
        if ops and ops_sh:
            print("  TIER RULER — REAL vs SHUFFLE at the SAME per-day operating points "
                  "(where the model actually trades):", flush=True)
            for r, s in zip(ops, ops_sh):
                if r['rate'] != s['rate']:
                    continue
                lift = r['meanR'] - s['meanR']
                print(f"    ~{r['rate']}/day: REAL {r['wr3R']:5.1%} {r['meanR']:+.3f}  vs  "
                      f"SHUFFLE {s['wr3R']:5.1%} {s['meanR']:+.3f}   lift {lift:+.3f} "
                      f"{'PASS' if lift >= PASS_LIFT_MARGIN_R else 'FAIL'}", flush=True)
            print(f"  -> TIER VERDICT: "
                  f"{'BEATS SHUFFLE at every deploy tier' if out['beats_shuffle_tiers'] else 'FAIL — tier lift below margin'}",
                  flush=True)
    return out


def train_final_streamed(make_labeler, streams, classifier, clf_kwargs=None,
                         holdout_start='2026-01-01', val_frac=0.15, seed=0, chunk=2000,
                         export_onnx=False, output_path=None, verbose=True, oos_end=None):
    """Run on ALL data across many (ticker, timeframe) streams with bounded memory:
    load each stream sequentially, featurize its train/val/oos pivots to part memmaps,
    RELEASE its bars, next stream; concat parts into full memmaps; train per-batch.
    Peak RAM = one stream + one batch. This is the full 3/5/15 (or all-tickers) run.

    oos_end: optional upper bound on the OOS window (default None = end of data). The
    ANCHORED walk-forward primitive: train <holdout_start, score [holdout_start, oos_end)
    — one anchored fold per (holdout_start, oos_end) pair, bars/caches shared across folds."""
    import gc
    from ._memmap import featurize_to_memmap, concat_memmaps, memmap_standardize_stats
    clf = get_classifier(classifier, **dict(clf_kwargs or {}))
    hs = pd.Timestamp(holdout_start, tz='UTC')
    oe = pd.Timestamp(oos_end, tz='UTC') if oos_end else None
    rng = np.random.default_rng(seed)
    rundir = (Path(output_path).parent if output_path else Path('.'))
    rundir.mkdir(parents=True, exist_ok=True)

    tr_parts, va_parts, te_parts = [], [], []
    Ytr_tr, Ytr_va, all_Kte, all_Yte, all_te_ts = [], [], [], [], []
    Ktr_tr, Ktr_va = [], []                          # per-subset keys (distributional ladder labels)
    channel_names = None; C = seq = None; eval_lab = None
    for i, (tk, tf) in enumerate(streams):
        lab = make_labeler(tk, tf)                       # loads ONLY this stream's bars
        cal = lab.calendar(); lo, hi = cal['timestamp'].min(), cal['timestamp'].max()
        _, Ytr, Ktr = lab.build(lo, hs, hs)
        _hi = min(hi + pd.Timedelta('1ns'), oe) if oe is not None else hi + pd.Timedelta('1ns')
        _, Yte, Kte = lab.build(hs, _hi, None)
        Ytr = np.asarray(Ytr).astype(int); Yte = np.asarray(Yte).astype(int)
        if channel_names is None and hasattr(lab, 'mv_feature_names'):
            channel_names = lab.mv_feature_names()
        if len(Ktr) >= 2:
            idx = rng.permutation(len(Ktr)); nv = max(1, int(len(Ktr) * val_frac))
            va_i, tr_i = idx[:nv], idx[nv:]
            p = str(rundir / f'_tr{i}.npy')
            _, sh = featurize_to_memmap(clf, lab, [Ktr[j] for j in tr_i], p, chunk)
            tr_parts.append((p, len(tr_i))); Ytr_tr += list(Ytr[tr_i]); C, seq = sh[1], sh[2]
            Ktr_tr += [Ktr[j] for j in tr_i]
            p = str(rundir / f'_va{i}.npy')
            featurize_to_memmap(clf, lab, [Ktr[j] for j in va_i], p, chunk)
            va_parts.append((p, len(va_i))); Ytr_va += list(Ytr[va_i])
            Ktr_va += [Ktr[j] for j in va_i]
        if len(Kte):
            p = str(rundir / f'_te{i}.npy')
            featurize_to_memmap(clf, lab, list(Kte), p, chunk)
            te_parts.append((p, len(Kte))); all_Kte += list(Kte); all_Yte += list(Yte)
            try:                                          # OOS timestamps (per-day operating points)
                all_te_ts += [lab._b[(tk, tf)]['ts'][int(k[1])] for k in Kte]
            except Exception:
                all_te_ts += [pd.NaT] * len(Kte)
        if verbose:
            print(f"  [stream {tk}@{tf}] train={len(Ktr)} oos={len(Kte)}", flush=True)
        for attr in ('_b', '_labels'):                   # release bars (evaluate uses keys)
            if hasattr(lab, attr):
                try:
                    getattr(lab, attr).clear()
                except Exception:
                    pass
        eval_lab = lab                                   # keep one (R from key tuples)
        gc.collect()

    Xtr, _ = concat_memmaps(tr_parts, str(rundir / '_Xtr.npy'))
    Xval, _ = concat_memmaps(va_parts, str(rundir / '_Xval.npy'))
    Xte, _ = concat_memmaps(te_parts, str(rundir / '_Xte.npy'))
    Ytr_tr = np.array(Ytr_tr); Ytr_va = np.array(Ytr_va); Yte = np.array(all_Yte)
    ck = dict(clf_kwargs or {}); mu = sd = None
    if clf.needs_standardize:
        mu, sd = memmap_standardize_stats(Xtr)
        ck['standardize_mu'] = mu.tolist(); ck['standardize_sd'] = sd.tolist()
    tks = sorted({s[0] for s in streams}); tfs = sorted({s[1] for s in streams})
    if verbose:
        print(f"=== PRODUCE STREAMED ({classifier}: {len(streams)} streams, 2026 OOS) ===")
        print(f"  train={len(Ytr_tr)} val={len(Ytr_va)} oos={len(all_Kte)} C={C} seq={seq} "
              f"good(train)={Ytr_tr.mean():.3f} good(oos)={Yte.mean():.3f}", flush=True)
    onnx_path = (str(Path(output_path).with_suffix('')) + '.onnx'
                 if (export_onnx and output_path) else None)
    dist = ck.get('rank') == 'expected_reach'
    out = _fit_score(classifier, ck, eval_lab, Xtr, Ytr_tr, Xval, Ytr_va, Xte, all_Kte, Yte,
                     seed, verbose, onnx_path=onnx_path,
                     keys_tr=(Ktr_tr if dist else None), keys_val=(Ktr_va if dist else None),
                     oos_ts=all_te_ts)
    return _emit(out, classifier, ck, eval_lab, mu, sd, C, seq,
                 channel_names, tks, tfs, holdout_start, export_onnx, output_path, verbose)


def train_final(labeler, classifier, clf_kwargs=None, holdout_start='2026-01-01',
                val_frac=0.15, seed=0, max_train=None, stream=False, chunk=2000,
                export_onnx=False, output_path=None, verbose=True):
    clf = get_classifier(classifier, **dict(clf_kwargs or {}))
    hs = pd.Timestamp(holdout_start, tz='UTC')
    cal = labeler.calendar()
    lo, hi = cal['timestamp'].min(), cal['timestamp'].max()
    Ctr, Ytr, Ktr = labeler.build(lo, hs, hs)
    Cte, Yte, Kte = labeler.build(hs, hi + pd.Timedelta('1ns'), None)
    Ytr = np.asarray(Ytr).astype(int); Yte = np.asarray(Yte).astype(int)
    oos_ts = []                                          # OOS timestamps for per-day operating points
    try:
        oos_ts = [labeler._b[tuple(k[0].split('@'))]['ts'][int(k[1])] for k in Kte]
    except Exception:
        oos_ts = []
    if len(Ytr) < 50 or len(Kte) < 20:
        raise ValueError(f"insufficient data: train={len(Ytr)} oos={len(Kte)}")
    if max_train and len(Ktr) > max_train:
        sub = np.random.default_rng(seed).choice(len(Ktr), max_train, replace=False)
        Ktr = [Ktr[j] for j in sub]; Ytr = Ytr[sub]
    rng = np.random.default_rng(seed)
    idx = np.arange(len(Ktr)); rng.shuffle(idx)
    nv = max(10, int(len(idx) * val_frac))
    va_i, tr_i = idx[:nv], idx[nv:]
    Ktr_tr = [Ktr[j] for j in tr_i]; Ytr_tr = Ytr[tr_i]
    Ktr_va = [Ktr[j] for j in va_i]; Ytr_va = Ytr[va_i]

    ck = dict(clf_kwargs or {})
    mu = sd = None
    if stream:
        from ._memmap import featurize_to_memmap, memmap_standardize_stats
        rundir = (Path(output_path).parent if output_path else Path('.'))
        rundir.mkdir(parents=True, exist_ok=True)
        Xtr = str(rundir / '_Xtr.npy'); Xval = str(rundir / '_Xval.npy'); Xte = str(rundir / '_Xte.npy')
        _, (ntr, C, seq) = featurize_to_memmap(clf, labeler, Ktr_tr, Xtr, chunk)
        featurize_to_memmap(clf, labeler, Ktr_va, Xval, chunk)
        featurize_to_memmap(clf, labeler, Kte, Xte, chunk)
        if clf.needs_standardize:
            mu, sd = memmap_standardize_stats(Xtr)
            ck['standardize_mu'] = mu.tolist(); ck['standardize_sd'] = sd.tolist()
        # features are on disk now; evaluate() reads R from the key tuples (not _b), so
        # drop the labeler's bars to free RAM before the worker trains (stream is the
        # memory-critical full-data path — always free).
        import gc
        for attr in ('_b', '_labels'):
            if hasattr(labeler, attr):
                try:
                    getattr(labeler, attr).clear()
                except Exception:
                    pass
        gc.collect()
        if verbose:
            print("  [mem] freed labeler bars after featurize (memmap holds features)",
                  flush=True)
    else:
        Xtr = clf.featurize(labeler, Ktr_tr)
        Xval = clf.featurize(labeler, Ktr_va)
        Xte = clf.featurize(labeler, Kte)
        C, seq = int(Xtr.shape[1]), int(Xtr.shape[2])
        if clf.needs_standardize:
            Xtr, Xval, Xte, mu, sd = _standardize_on_train(Xtr, Xval, Xte)

    if verbose:
        print(f"=== PRODUCE ({classifier}{' STREAM' if stream else ''}: train < "
              f"{holdout_start}, 2026 OOS) ===")
        print(f"  train={len(tr_i)} val={len(va_i)} oos={len(Kte)} C={C} seq={seq} "
              f"good(train)={Ytr_tr.mean():.3f} good(oos)={Yte.mean():.3f}", flush=True)

    onnx_path = (str(Path(output_path).with_suffix('')) + '.onnx'
                 if (export_onnx and output_path) else None)
    if verbose:
        print(f"  [produce 1/2] fit REAL head{' (+ ONNX export)' if onnx_path else ''}",
              flush=True)
    dist = ck.get('rank') == 'expected_reach'
    kt, kv = (Ktr_tr, Ktr_va) if dist else (None, None)   # distributional ladder labels
    ck_real = dict(ck, export_onnx_path=onnx_path) if onnx_path else ck
    p_val, p_te, ba = get_classifier(classifier, **ck_real).fit_predict(
        Xtr, Ytr_tr, Xval, Ytr_va, Xte, seed, keys_tr=kt, keys_val=kv)
    thr = _pct_threshold(p_val, OP_PERCENTILE)
    R = _arm_R(labeler, Kte, p_te, thr)
    if dist:
        perm = rng.permutation(len(Ytr_tr))               # permute label + ladder keys together
        ysh, Ksh = Ytr_tr[perm], [Ktr_tr[i] for i in perm]
    else:
        ysh = Ytr_tr.copy(); rng.shuffle(ysh); Ksh = None
    if verbose:
        print("  [produce 2/2] fit SHUFFLE control (honest ruler)", flush=True)
    psv, ps, _ = get_classifier(classifier, **ck).fit_predict(Xtr, ysh, Xval, Ytr_va, Xte, seed,
                                                              keys_tr=Ksh, keys_val=kv)
    Rs = _arm_R(labeler, Kte, ps, _pct_threshold(psv, OP_PERCENTILE))

    auc = None
    if len(np.unique(Yte)) == 2:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(Yte, p_te))
    edge = _meanR(R) - _meanR(Rs)
    bands = wr_by_score(labeler, Kte, p_te, oos_ts) if oos_ts else []
    ops = operating_points(labeler, Kte, p_te, oos_ts) if oos_ts else []
    out = dict(oos_auc=auc, best_val_auc=ba, oos_meanR=_meanR(R), shuffle_meanR=_meanR(Rs),
               edge_shuffle=edge, n_train=len(tr_i), n_oos=len(Kte), oos_trades=int(len(R)),
               beats_shuffle=bool(edge >= PASS_LIFT_MARGIN_R),
               wr_by_score=bands, operating_points=ops)
    if verbose:
        print(f"  OOS AUC {auc:.4f}" if auc is not None else "  OOS AUC n/a")
        print(f"  OOS meanR REAL {out['oos_meanR']:+.3f} SHUFFLE {out['shuffle_meanR']:+.3f} "
              f"edge {edge:+.3f} (trades={out['oos_trades']})")
        print(f"  -> {'beats SHUFFLE' if out['beats_shuffle'] else 'does NOT beat SHUFFLE'}")
        _print_operating_points(ops, bands)

    if export_onnx and output_path:
        base = Path(output_path).with_suffix('')
        bundle = _bundle_files(base)                  # exported during the REAL fit (no refit)
        sha = (hashlib.sha256(b''.join(Path(p).read_bytes() for p in bundle)).hexdigest()
               if bundle else None)
        contract = _contract(labeler, classifier, ck, C, seq, mu, sd, out, onnx_path, sha,
                             holdout_start, len(tr_i), len(Kte))
        contract['onnx'] = [Path(p).name for p in bundle] if sha else None
        cpath = str(base) + '_signal.json'
        Path(cpath).write_text(json.dumps(contract, indent=2))
        out['artifacts'] = {'onnx': (bundle if sha else None), 'contract': cpath,
                            'content_sha': sha}
        if verbose:
            print(f"  artifacts: onnx={out['artifacts']['onnx']} contract={cpath}")
    return out
