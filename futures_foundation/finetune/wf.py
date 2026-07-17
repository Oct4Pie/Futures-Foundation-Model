"""Walk-forward honest ruler — Classifier-agnostic (the generic, reusable core).

Concepts ported from pipeline.evaluate (the Chronos+XGBoost harness) but freed of
any model specifics: leak-free walk-forward × {REAL, SHUFFLE, RANDOM} × seeds, a
number believed ONLY if REAL clearly beats the controls on realized R, plus the
VAL→TEST generalization gate and per-fold health monitoring. The MODEL is any
`Classifier` (Mantis, logistic, future backbones) — featurize + fit_predict + an
optional train-stat standardize. Swap models by name; the ruler is identical.

The labeler satisfies the pipeline StrategyLabeler protocol (calendar/build/evaluate)
plus whatever featurization the chosen Classifier needs (e.g. mv_contexts).
"""
import numpy as np
import pandas as pd

from futures_foundation.pipeline.data import walk_forward_folds
from .classifier import get_classifier
from .health import FoldHealthMonitor

PASS_LIFT_MARGIN_R = 0.10     # REAL must beat each control by this (realized R)
GEN_GAP_TOL = 0.30            # VAL->TEST meanR gap above this = does NOT generalize
OP_PERCENTILE = 0.50          # trade the top 50% by proba (usable volume)


def _pct_threshold(proba, top_pct):
    proba = np.asarray(proba, float)
    if proba.size == 0:
        return 1.0
    return float(np.quantile(proba, 1.0 - top_pct))


def _meanR(R):
    R = np.asarray(R, float)
    return float(R.mean()) if len(R) else 0.0


def _arm_R(labeler, keys, proba, thr):
    """Top-`thr` proba decisions -> realized per-trade R (cost in the strategy)."""
    preds = (np.asarray(proba) >= thr).astype(int)
    if preds.sum() == 0:
        return np.array([])
    return np.asarray(labeler.evaluate(list(keys), preds), float)


def _standardize_on_train(Xtr, Xval, Xeval):
    """Per-channel standardize [N,C,seq] on TRAIN stats only (no leak).
    Returns (Xtr, Xval, Xeval, mu, sd) — mu/sd are the per-channel train stats the
    serve path (ONNX consumer) must reproduce."""
    C = Xtr.shape[1]
    flat = Xtr.transpose(0, 2, 1).reshape(-1, C)
    mu, sd = flat.mean(0), flat.std(0) + 1e-6
    def s(A):
        return ((A - mu[None, :, None]) / sd[None, :, None]).astype(np.float32)
    return s(Xtr), s(Xval), s(Xeval), mu, sd


def _health_metrics(p_te, Yte, p_val, Yval, thr=0.80):
    def prec(p, y):
        m = np.asarray(p) >= thr
        return (float((np.asarray(y)[m] == 1).mean()) if m.sum() else 0.0, int(m.sum()))
    pte, nte = prec(p_te, Yte)
    pva, _ = prec(p_val, Yval)
    return dict(all_conf=np.asarray(p_te), all_labels=np.asarray(Yte),
                prec_at_80=pte, n_at_80=nte, val_p80=pva)


def run(labeler, classifier, clf_kwargs=None, seeds=(0,), train_m=3, val_m=1,
        test_m=1, max_folds=None, holdout_start='2026-01-01', verbose=True,
        health_monitor=None):
    """Returns a verdict dict. clf_kwargs forwarded to get_classifier."""
    clf = get_classifier(classifier, **dict(clf_kwargs or {}))
    monitor = health_monitor or FoldHealthMonitor()
    pool = {'REAL': [], 'SHUFFLE': [], 'RANDOM': []}
    auc_real, val_meanR, test_meanR = [], [], []
    n_folds = 0

    for fold, tr, val, te in walk_forward_folds(labeler.calendar(), train_m, val_m,
                                                test_m, holdout_start=holdout_start):
        if max_folds is not None and n_folds >= max_folds:
            break
        val0, te0 = val['timestamp'].min(), te['timestamp'].min()
        Ctr, Ytr, Ktr = labeler.build(tr['timestamp'].min(), tr['timestamp'].max(), val0)
        Cval, Yval, Kval = labeler.build(val0, val['timestamp'].max(), te0)
        Cte, Yte, Kte = labeler.build(te['timestamp'].min(),
                                      te['timestamp'].max() + np.timedelta64(1, 'ns'), None)
        Ytr, Yval, Yte = map(lambda a: np.asarray(a).astype(int), (Ytr, Yval, Yte))
        if len(Ytr) < 50 or len(Kte) < 50 or len(Cval) < 10:
            continue
        Xtr = clf.featurize(labeler, Ktr)
        Xval = clf.featurize(labeler, Kval)
        Xte = clf.featurize(labeler, Kte)
        if clf.needs_standardize:
            Xtr, Xval, Xte, _, _ = _standardize_on_train(Xtr, Xval, Xte)
        n_folds += 1
        if verbose:
            print(f"\n[fold {fold}] train={len(Ytr)} val={len(Yval)} test={len(Yte)} "
                  f"X={tuple(Xtr.shape[1:])} good={Ytr.mean():.3f}", flush=True)

        dist = (clf_kwargs or {}).get('rank') == 'expected_reach'
        Ktr = list(Ktr) if dist else None                 # ladder labels (distributional only)
        fold_p_te = fold_p_val = None
        for seed in seeds:
            rng = np.random.default_rng(seed)
            # REAL
            p_val, p_te, ba = clf.fit_predict(Xtr, Ytr, Xval, Yval, Xte, seed,
                                              keys_tr=Ktr, keys_val=list(Kval))
            thr = _pct_threshold(p_val, OP_PERCENTILE)
            R_te = _arm_R(labeler, Kte, p_te, thr)
            pool['REAL'].append(R_te)
            auc_real.append((Yte, p_te))
            val_meanR.append(_meanR(_arm_R(labeler, Kval, p_val, thr)))
            test_meanR.append(_meanR(R_te))
            if fold_p_te is None:
                fold_p_te, fold_p_val = p_te, p_val
            if verbose:
                from sklearn.metrics import roc_auc_score
                ta = (roc_auc_score(Yte, p_te) if len(np.unique(Yte)) == 2 else float('nan'))
                print(f"  seed{seed} REAL best_val_auc={ba:.4f} test_auc={ta:.4f} "
                      f"meanR={_meanR(R_te):+.3f}", flush=True)
            # SHUFFLE — permute label + ladder keys together (dist) so the control isn't a no-op
            if dist:
                perm = rng.permutation(len(Ytr))
                ysh, Ksh = Ytr[perm], [Ktr[i] for i in perm]
            else:
                ysh = Ytr.copy(); rng.shuffle(ysh); Ksh = None
            psv, ps, _ = clf.fit_predict(Xtr, ysh, Xval, Yval, Xte, seed,
                                         keys_tr=Ksh, keys_val=list(Kval))
            pool['SHUFFLE'].append(_arm_R(labeler, Kte, ps, _pct_threshold(psv, OP_PERCENTILE)))
            # RANDOM — threshold from a val-sized draw (same forward protocol as REAL/SHUFFLE;
            # thresholding pr on itself reads the test distribution, an asymmetric ruler)
            pr = rng.random(len(Kte))
            pool['RANDOM'].append(_arm_R(labeler, Kte, pr,
                                         _pct_threshold(rng.random(len(Kval)), OP_PERCENTILE)))

        # per-fold health (REAL, first seed)
        monitor.check(f'F{n_folds}', _health_metrics(fold_p_te, Yte, fold_p_val, Yval))

    def cat(arm):
        return np.concatenate(pool[arm]) if pool[arm] else np.array([])
    real_m, shuf_m, rand_m = _meanR(cat('REAL')), _meanR(cat('SHUFFLE')), _meanR(cat('RANDOM'))
    gap = (np.mean(val_meanR) - np.mean(test_meanR)) if val_meanR else None
    auc = None
    if auc_real:
        ys = np.concatenate([y for y, _ in auc_real])
        ps = np.concatenate([p for _, p in auc_real])
        if len(np.unique(ys)) == 2:
            from sklearn.metrics import roc_auc_score
            auc = float(roc_auc_score(ys, ps))

    generalizes = gap is not None and gap <= GEN_GAP_TOL
    checks = [
        (real_m - shuf_m >= PASS_LIFT_MARGIN_R,
         f"REAL-SHUFFLE >={PASS_LIFT_MARGIN_R}R ({real_m-shuf_m:+.2f}R)"),
        (real_m - rand_m >= PASS_LIFT_MARGIN_R,
         f"REAL-RANDOM >={PASS_LIFT_MARGIN_R}R ({real_m-rand_m:+.2f}R)"),
        (generalizes, f"GENERALIZES VAL->TEST gap <={GEN_GAP_TOL}R "
         + (f"({gap:+.2f}R)" if gap is not None else "(no val/test)")),
    ]
    all_pass = all(ok for ok, _ in checks)
    verdict = dict(all_pass=all_pass, generalizes=generalizes, auc=auc, real_meanR=real_m,
                   shuffle_meanR=shuf_m, random_meanR=rand_m, gap=gap, n_folds=n_folds,
                   real_trades=len(cat('REAL')), edge_shuffle=real_m - shuf_m)
    if verbose:
        print(f"\n=== WF HONEST RULER ({classifier}, folds={n_folds}) ===")
        print(f"  pooled TEST AUC {auc:.4f}" if auc is not None else "  AUC n/a")
        print(f"  meanR REAL {real_m:+.3f} SHUFFLE {shuf_m:+.3f} RANDOM {rand_m:+.3f} "
              f"(trades={len(cat('REAL'))})")
        for ok, msg in checks:
            print(f"  [{'PASS' if ok else 'FAIL'}] {msg}")
        print(f"  -> {'ALL PASS' if all_pass else 'FAIL'}")
    return verdict


# ===========================================================================
# Streamed walk-forward: featurize ALL pivots ONCE (per-stream, bounded RAM) ->
# rolling 3/1/1 folds slice the full memmap by month window -> optimized per-fold
# train (small fold = all-on-device + big batch). The generalize-over-time gate
# for full multi-TF/all-ticker data without re-loading the labeler per fold.
# ===========================================================================
def _verdict(classifier, pool, auc_real, val_meanR, test_meanR, n_folds, verbose):
    def cat(arm):
        return np.concatenate(pool[arm]) if pool[arm] else np.array([])
    real_m, shuf_m, rand_m = _meanR(cat('REAL')), _meanR(cat('SHUFFLE')), _meanR(cat('RANDOM'))
    gap = (np.mean(val_meanR) - np.mean(test_meanR)) if val_meanR else None
    auc = None
    if auc_real:
        ys = np.concatenate([y for y, _ in auc_real])
        ps = np.concatenate([p for _, p in auc_real])
        if len(np.unique(ys)) == 2:
            from sklearn.metrics import roc_auc_score
            auc = float(roc_auc_score(ys, ps))
    generalizes = gap is not None and gap <= GEN_GAP_TOL
    checks = [
        (real_m - shuf_m >= PASS_LIFT_MARGIN_R,
         f"REAL-SHUFFLE >={PASS_LIFT_MARGIN_R}R ({real_m-shuf_m:+.2f}R)"),
        (real_m - rand_m >= PASS_LIFT_MARGIN_R,
         f"REAL-RANDOM >={PASS_LIFT_MARGIN_R}R ({real_m-rand_m:+.2f}R)"),
        (generalizes, f"GENERALIZES VAL->TEST gap <={GEN_GAP_TOL}R "
         + (f"({gap:+.2f}R)" if gap is not None else "(no val/test)")),
    ]
    all_pass = all(ok for ok, _ in checks)
    verdict = dict(all_pass=all_pass, generalizes=generalizes, auc=auc, real_meanR=real_m,
                   shuffle_meanR=shuf_m, random_meanR=rand_m, gap=gap, n_folds=n_folds,
                   real_trades=len(cat('REAL')), edge_shuffle=real_m - shuf_m)
    if verbose:
        print(f"\n=== WF-STREAM HONEST RULER ({classifier}, folds={n_folds}) ===")
        print(f"  pooled TEST AUC {auc:.4f}" if auc is not None else "  AUC n/a")
        print(f"  meanR REAL {real_m:+.3f} SHUFFLE {shuf_m:+.3f} RANDOM {rand_m:+.3f} "
              f"(trades={len(cat('REAL'))})")
        for ok, msg in checks:
            print(f"  [{'PASS' if ok else 'FAIL'}] {msg}")
        print(f"  -> {'ALL PASS (generalizes over time)' if all_pass else 'FAIL'}")
    return verdict


def _rolling_folds(ts, train_m=3, val_m=1, test_m=1, max_folds=None, holdout_start=None,
                   label_end_ts=None):
    """Rolling month windows -> list of (tr_rows, va_rows, te_rows) row-index arrays
    into the featurize-once full set. Stride = test_m (default 1).

    holdout_start (e.g. '2026-01-01'): months >= this are EXCLUDED from the rolling
    folds entirely — the WF never sees the holdout, which stays the untouched final
    OOS for produce.

    label_end_ts is the timestamp of the final bar consumed by each forward label.
    When supplied, every arm is purged so its labels finish before the next temporal
    boundary. Signal timestamps alone are insufficient for forward-outcome labels."""
    ts_idx = pd.DatetimeIndex(ts)
    per = ts_idx.tz_localize(None) if ts_idx.tz is not None else ts_idx
    perM = per.to_period('M')
    end_idx = None
    if label_end_ts is not None:
        end_idx = pd.DatetimeIndex(label_end_ts)
        if len(end_idx) != len(ts_idx):
            raise ValueError(f'label_end_ts length {len(end_idx)} != timestamps {len(ts_idx)}')
        if ts_idx.tz is not None:
            end_idx = (end_idx.tz_localize(ts_idx.tz) if end_idx.tz is None
                       else end_idx.tz_convert(ts_idx.tz))
        elif end_idx.tz is not None:
            end_idx = end_idx.tz_localize(None)

    def boundary(period):
        out = period.to_timestamp(how='start')
        return out.tz_localize(ts_idx.tz) if ts_idx.tz is not None else out
    months = perM.unique().sort_values()
    if holdout_start is not None:
        hs = pd.Period(pd.Timestamp(holdout_start).tz_localize(None)
                       if pd.Timestamp(holdout_start).tzinfo else pd.Timestamp(holdout_start),
                       freq='M')
        months = months[months < hs]          # 2026+ never in any fold
    span = train_m + val_m + test_m
    starts = list(range(0, len(months) - span + 1, test_m))    # every 3/1/1 window (stride)
    if max_folds and len(starts) > max_folds:
        # SPACE max_folds evenly across ALL years (not the first N) — each fold is still a
        # true train_m/val_m/test_m window; the start points are spread over the full range.
        sel = np.linspace(0, len(starts) - 1, max_folds).round().astype(int)
        starts = [starts[i] for i in dict.fromkeys(sel.tolist())]
    folds = []
    for s in starts:
        tr = np.flatnonzero(perM.isin(months[s:s + train_m]))
        va = np.flatnonzero(perM.isin(months[s + train_m:s + train_m + val_m]))
        te = np.flatnonzero(perM.isin(months[s + train_m + val_m:s + span]))
        if end_idx is not None:
            val_start = boundary(months[s + train_m])
            test_start = boundary(months[s + train_m + val_m])
            test_end = boundary(months[s + span - 1] + 1)
            tr = tr[np.asarray(end_idx[tr] < val_start)]
            va = va[np.asarray(end_idx[va] < test_start)]
            te = te[np.asarray(end_idx[te] < test_end)]
        if len(tr) >= 50 and len(va) >= 10 and len(te) >= 10:
            folds.append((tr, va, te))
    return folds


def _label_end_times(labeler, keys):
    """Return the final timestamp consumed by every candidate's forward label.

    A labeler may expose ``label_end_times(keys)``. The common Mantis strategy
    contract is also supported directly via ``VERT`` and ``_b``. Anything else
    fails closed: streamed WF cannot honestly purge an unknown label horizon.
    """
    fn = getattr(labeler, 'label_end_times', None)
    if callable(fn):
        out = pd.DatetimeIndex(fn(list(keys)))
        if len(out) != len(keys):
            raise ValueError('label_end_times returned the wrong number of timestamps')
        return out
    vert = getattr(labeler, 'VERT', None)
    bars = getattr(labeler, '_b', None)
    if vert is None or not isinstance(bars, dict):
        raise ValueError(
            'streamed walk-forward requires label_end_times(keys), or a Mantis-style '
            'labeler with VERT and _b, so forward labels can be purged')
    out = []
    for key in keys:
        sid, i = key[0], int(key[1])
        tk, tf = sid.split('@')
        bts = bars[(tk, tf)]['ts']
        end_i = i + 1 + int(vert)
        if end_i >= len(bts):
            raise ValueError(f'label horizon exceeds bars for {sid} candidate {i}')
        out.append(bts[end_i])
    return pd.DatetimeIndex(out)


def featurize_all_streams(make_labeler, streams, clf, rundir, chunk=2000, verbose=True):
    """Per-stream sequential featurize of ALL pivots -> full memmap + per-pivot timestamps
    + keys + labels. Loads/releases one stream at a time (bounded RAM)."""
    import gc
    from ._memmap import featurize_to_memmap, concat_memmaps
    from pathlib import Path
    rundir = Path(rundir)
    parts, all_keys, all_y, all_ts, all_label_end_ts = [], [], [], [], []
    channel_names = C = seq = eval_lab = None
    for i, (tk, tf) in enumerate(streams):
        lab = make_labeler(tk, tf)
        cal = lab.calendar(); lo, hi = cal['timestamp'].min(), cal['timestamp'].max()
        _, Y, K = lab.build(lo, hi + np.timedelta64(1, 'ns'), None)
        Y = np.asarray(Y).astype(int)
        if channel_names is None and hasattr(lab, 'mv_feature_names'):
            channel_names = lab.mv_feature_names()
        if len(K):
            ts = [lab._b[tuple(k[0].split('@'))]['ts'][int(k[1])] for k in K]
            label_end_ts = _label_end_times(lab, K)
            p = str(rundir / f'_all{i}.npy')
            _, sh = featurize_to_memmap(clf, lab, list(K), p, chunk)
            parts.append((p, len(K))); all_keys += list(K); all_y += list(Y)
            all_ts += list(ts); all_label_end_ts += list(label_end_ts); C, seq = sh[1], sh[2]
        if verbose:
            print(f"  [featurize {tk}@{tf}] pivots={len(K)}", flush=True)
        for attr in ('_b', '_labels'):
            if hasattr(lab, attr):
                try:
                    getattr(lab, attr).clear()
                except Exception:
                    pass
        eval_lab = lab
        gc.collect()
    Xpath, _ = concat_memmaps(parts, str(rundir / '_Xall.npy'))
    return (Xpath, np.array(all_y), all_keys, pd.DatetimeIndex(all_ts),
            pd.DatetimeIndex(all_label_end_ts),
            channel_names, eval_lab, C, seq)


def _run_folds(classifier, ck, Xall, Y, keys, eval_lab, rundir, folds, seed, verbose, monitor,
               fold_ckpt=None):
    """Per-fold: slice the full memmap -> fit_predict (REAL/SHUFFLE/RANDOM) -> R + health.
    Given a featurized full set + a config; reused across overfit-loop iterations (no
    re-featurize).

    DISCONNECT-RESUME: fold_ckpt (optional durable path, e.g. on Drive) -> per-fold results are
    saved atomically after EVERY fold via the shared resume toolkit (finetune/resume.py, the same
    machinery as the sweep's DB snapshot); on restart, completed folds reload and only the
    remaining ones run. Keyed on a CONFIG SIGNATURE (classifier + clf_kwargs + fold layout +
    seed) so a changed config auto-invalidates; the control RNG stream is replayed for skipped
    folds so a resumed run is byte-identical to a single-session run."""
    import os as _os
    import warnings
    from ._memmap import slice_memmap
    from .resume import KeyedCheckpoint, config_signature
    # Silence any residual lbfgs ConvergenceWarning. With train-stat standardize applied at fit
    # (mantis_frozen.fit_predict) the head converges well inside the iter cap; a rare fold that
    # still warns is a stable-ranking edge case, and the arms stay comparable either way.
    try:
        from sklearn.exceptions import ConvergenceWarning
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
    except Exception:
        pass
    clf_run = get_classifier(classifier, **ck)
    needs_standardize = bool(clf_run.needs_standardize)
    cfg = {k: v for k, v in (ck or {}).items() if not str(k).startswith('standardize_')}
    cfg['_wf_preprocess_version'] = 2
    cfg['_fold_train_standardize'] = needs_standardize
    import hashlib as _hashlib
    def _rowsig(rows):
        a = np.asarray(rows, np.int64)
        return [len(a), _hashlib.sha1(a.tobytes()).hexdigest()[:16]]
    ckpt = KeyedCheckpoint(fold_ckpt, config_signature(
        classifier, cfg, int(seed),
        [[_rowsig(a), _rowsig(b), _rowsig(c)] for a, b, c in folds]))
    res = ckpt.load(dict(real=[], shuf=[], rand=[], yte=[], pte=[], valm=[], testm=[]))
    if verbose and res['real']:
        print(f"  [fold-ckpt] resumed {len(res['real'])}/{len(folds)} folds from {fold_ckpt}",
              flush=True)
    # DISTRIBUTIONAL reach-ladder mode threads the strategy keys (the ladder's labels) into
    # fit_predict. The SHUFFLE control must then permute the KEYS in lockstep with the label — the
    # ladder reads its per-target labels from the keys, so shuffling only the label vector would
    # leave the ladder intact and collapse SHUFFLE onto REAL (dead control). The `dist` branch uses
    # rng.permutation (label+keys together); the single-head path keeps its exact rng.shuffle stream
    # so existing fold-ckpts stay byte-identical (distributional adds rank= to the sig -> new ckpt).
    dist = (ck or {}).get('rank') == 'expected_reach'
    rng = np.random.default_rng(seed)
    for fi, (tr, va, te) in enumerate(folds):
        tr, va, te = np.sort(tr), np.sort(va), np.sort(te)
        if fi < len(res['real']):                         # done in a previous session ->
            if dist:                                      # replay the control RNG stream so the
                rng.permutation(len(tr))                  # remaining folds match a 1-session run
            else:
                d = np.empty(len(tr)); rng.shuffle(d)
            rng.random(len(te))
            continue
        ftr, fva, fte = (str(rundir / '_ftr.npy'), str(rundir / '_fva.npy'),
                         str(rundir / '_fte.npy'))
        slice_memmap(Xall, tr, ftr); slice_memmap(Xall, va, fva); slice_memmap(Xall, te, fte)
        fold_clf = clf_run
        if needs_standardize:
            from ._memmap import memmap_standardize_stats
            mu, sd = memmap_standardize_stats(ftr)
            fold_ck = dict(ck)
            fold_ck['standardize_mu'] = mu.tolist()
            fold_ck['standardize_sd'] = sd.tolist()
            fold_clf = get_classifier(classifier, **fold_ck)
        Ytr, Yva, Yte = Y[tr], Y[va], Y[te]
        Kte = [keys[i] for i in te]; Kva = [keys[i] for i in va]
        Ktr = [keys[i] for i in tr] if dist else None     # ladder labels (distributional only)
        p_val, p_te, ba = fold_clf.fit_predict(ftr, Ytr, fva, Yva, fte, seed,
                                               keys_tr=Ktr, keys_val=Kva)
        thr = _pct_threshold(p_val, OP_PERCENTILE)
        R_te = _arm_R(eval_lab, Kte, p_te, thr)
        res['real'].append(R_te); res['yte'].append(np.asarray(Yte)); res['pte'].append(np.asarray(p_te))
        res['valm'].append(_meanR(_arm_R(eval_lab, Kva, p_val, thr))); res['testm'].append(_meanR(R_te))
        if dist:
            perm = rng.permutation(len(Ytr))              # permute label + ladder-keys together
            ysh, Ksh = Ytr[perm], [Ktr[i] for i in perm]
        else:
            ysh = Ytr.copy(); rng.shuffle(ysh); Ksh = None
        psv, ps, _ = fold_clf.fit_predict(ftr, ysh, fva, Yva, fte, seed,
                                          keys_tr=Ksh, keys_val=Kva)
        res['shuf'].append(_arm_R(eval_lab, Kte, ps, _pct_threshold(psv, OP_PERCENTILE)))
        pr = rng.random(len(Kte))                        # val-sized threshold draw: same forward
        res['rand'].append(_arm_R(eval_lab, Kte, pr,     # protocol as the REAL/SHUFFLE arms
                                  _pct_threshold(rng.random(len(Kva)), OP_PERCENTILE)))
        if monitor:                                       # legacy P@80 monitor (opt-in; callers
            monitor.check(f'F{fi}', _health_metrics(p_te, Yte, p_val, Yva))   # pass None to skip)
        if verbose:
            from sklearn.metrics import roc_auc_score
            ta = roc_auc_score(Yte, p_te) if len(np.unique(Yte)) == 2 else float('nan')
            # Per-fold OVERFIT read, epoch-style: VAL is the tuning proxy, TEST is this fold's OOS.
            # A big VAL->TEST drop = overfit. gap = val meanR - test meanR (same metric + tolerance
            # as the GENERALIZES verdict, so a fold flagged OVERFIT is what fails the honest ruler).
            vmR, tmR = res['valm'][-1], res['testm'][-1]
            gap = vmR - tmR
            flag = 'OVERFIT' if gap > GEN_GAP_TOL else 'ok'
            print(f"  fold {fi:>2} | n train={len(tr)} val={len(va)} test={len(te)}"
                  f"  good={Ytr.mean():.2f} | AUC val={ba:.3f} test={ta:.3f}"
                  f" | meanR val={vmR:+.3f} test={tmR:+.3f} | val->test gap={gap:+.3f} [{flag}]",
                  flush=True)
        ckpt.save(res)                                    # durable after EVERY fold (atomic)
        for f in (ftr, fva, fte):
            try:
                _os.remove(f)
            except OSError:
                pass
    pool = {'REAL': res['real'], 'SHUFFLE': res['shuf'], 'RANDOM': res['rand']}
    auc_real = list(zip(res['yte'], res['pte']))
    return _verdict(classifier, pool, auc_real, res['valm'], res['testm'], len(folds), verbose)


def _featurize_and_folds(make_labeler, streams, clf, clf_kwargs, train_m, val_m, test_m,
                         max_folds, output_path, chunk, verbose, holdout_start='2026-01-01'):
    from pathlib import Path
    rundir = (Path(output_path).parent if output_path else Path('.'))
    rundir.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"[wf-stream] featurize-once over {len(streams)} streams ...", flush=True)
    Xall, Y, keys, ts, label_end_ts, channel_names, eval_lab, C, seq = featurize_all_streams(
        make_labeler, streams, clf, rundir, chunk, verbose)
    ck = dict(clf_kwargs or {})
    mu = sd = None
    folds = _rolling_folds(ts, train_m, val_m, test_m, max_folds,
                           holdout_start=holdout_start, label_end_ts=label_end_ts)
    if verbose:
        print(f"[wf-stream] {len(folds)} folds (rolling {train_m}/{val_m}/{test_m}, stride "
              f"{test_m}); holdout {holdout_start} EXCLUDED (reserved OOS)", flush=True)
    return rundir, Xall, Y, keys, eval_lab, ck, mu, sd, folds


def run_streamed(make_labeler, streams, classifier, clf_kwargs=None, train_m=3,
                 val_m=1, test_m=1, max_folds=None, holdout_start='2026-01-01',
                 output_path=None, chunk=2000, seed=0, verbose=True, health_monitor=None,
                 fold_ckpt=None):
    """Walk-forward over time on FULL multi-stream data, SINGLE config. Featurize once ->
    rolling folds (holdout_start EXCLUDED = untouched OOS) -> per-fold REAL/SHUFFLE/RANDOM
    + VAL->TEST gap + health. Overfit guards: per-fold val early-stop + gap + health.
    fold_ckpt (durable path) -> disconnect-resume (see _run_folds)."""
    clf = get_classifier(classifier, **dict(clf_kwargs or {}))
    rundir, Xall, Y, keys, eval_lab, ck, mu, sd, folds = _featurize_and_folds(
        make_labeler, streams, clf, clf_kwargs, train_m, val_m, test_m, max_folds,
        output_path, chunk, verbose, holdout_start=holdout_start)
    return _run_folds(classifier, ck, Xall, Y, keys, eval_lab, rundir, folds, seed,
                      verbose, health_monitor, fold_ckpt=fold_ckpt)   # None default = no P@80


def loop_streamed(make_labeler, streams, classifier, clf_kwargs=None, train_m=3,
                  val_m=1, test_m=1, max_folds=None, holdout_start='2026-01-01', max_iters=2,
                  n_trials=8, output_path=None, chunk=2000, seed=0, verbose=True,
                  fold_ckpt=None):
    """Walk-forward + OVERFIT->OPTUNA guard (the full overfit-guarded process). Run folds
    (holdout_start EXCLUDED = untouched OOS) with defaults; if they don't generalize
    (VAL->TEST gap), Optuna-search a generalizing config on fold-0 and re-run; repeat until
    it generalizes or max_iters. Featurize ONCE (overfit loop re-runs folds, not featurize)."""
    from . import tune as TUNE
    from ._memmap import slice_memmap
    clf = get_classifier(classifier, **dict(clf_kwargs or {}))
    rundir, Xall, Y, keys, eval_lab, ck, mu, sd, folds = _featurize_and_folds(
        make_labeler, streams, clf, clf_kwargs, train_m, val_m, test_m, max_folds,
        output_path, chunk, verbose, holdout_start=holdout_start)
    if verbose:
        print("[wf-loop] iter 0 · default config", flush=True)
    v = _run_folds(classifier, ck, Xall, Y, keys, eval_lab, rundir, folds, seed, verbose,
                   None, fold_ckpt=fold_ckpt)          # None = no legacy P@80 monitor; the
    #                                     per-fold val/test/overfit readout replaces it
    history = [dict(it=0, source='default', generalizes=v['generalizes'], gap=v['gap'])]

    if not v['generalizes'] and folds:
        import os as _os
        tr, va = np.sort(folds[0][0]), np.sort(folds[0][1])
        ftr, fva = str(rundir / '_otr.npy'), str(rundir / '_ova.npy')
        slice_memmap(Xall, tr, ftr); slice_memmap(Xall, va, fva)
        Xtr = np.asarray(np.load(ftr, mmap_mode='r'), np.float32)
        Xva = np.asarray(np.load(fva, mmap_mode='r'), np.float32)
        if clf.needs_standardize:                            # fold-0 TRAIN stats only
            from ._memmap import memmap_standardize_stats
            mu, sd = memmap_standardize_stats(ftr)
            Xtr = (Xtr - mu[None, :, None]) / sd[None, :, None]
            Xva = (Xva - mu[None, :, None]) / sd[None, :, None]
        base = dict(clf_kwargs or {})
        for it in range(1, max_iters + 1):
            if verbose:
                print(f"\n[wf-loop] OVERFIT (gap {v['gap']}) -> Optuna iter {it}", flush=True)
            scan = TUNE.tune(eval_lab, classifier, Xtr, Y[tr], Xva, Y[va], n_trials=n_trials,
                             base_kwargs=base, seed=seed + it, verbose=verbose)
            if not scan['generalizes']:
                if verbose:
                    print(f"  Optuna found nothing better (iter {it}); stop.", flush=True)
                break
            ck = dict(scan['params'])
            v = _run_folds(classifier, ck, Xall, Y, keys, eval_lab, rundir, folds, seed,
                           verbose, None, fold_ckpt=fold_ckpt)
            history.append(dict(it=it, source=f'tuned{it}', generalizes=v['generalizes'],
                                gap=v['gap'], params=scan['params']))
            base = dict(scan['params'])
            if v['generalizes']:
                if verbose:
                    print(f"  ** tuned config generalizes (iter {it})", flush=True)
                break
        for f in (ftr, fva):
            try:
                _os.remove(f)
            except OSError:
                pass

    v['history'] = history
    v['final_config'] = {k: val for k, val in ck.items()
                         if k not in ('standardize_mu', 'standardize_sd')}
    return v
