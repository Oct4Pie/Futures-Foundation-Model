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


def run(labeler, classifier='mantis', clf_kwargs=None, seeds=(0,), train_m=3, val_m=1,
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

        fold_p_te = fold_p_val = None
        for seed in seeds:
            rng = np.random.default_rng(seed)
            # REAL
            p_val, p_te, ba = clf.fit_predict(Xtr, Ytr, Xval, Yval, Xte, seed)
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
            # SHUFFLE
            ysh = Ytr.copy(); rng.shuffle(ysh)
            psv, ps, _ = clf.fit_predict(Xtr, ysh, Xval, Yval, Xte, seed)
            pool['SHUFFLE'].append(_arm_R(labeler, Kte, ps, _pct_threshold(psv, OP_PERCENTILE)))
            # RANDOM
            pr = rng.random(len(Kte))
            pool['RANDOM'].append(_arm_R(labeler, Kte, pr, _pct_threshold(pr, OP_PERCENTILE)))

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


def _rolling_folds(ts, train_m=3, val_m=1, test_m=1, max_folds=None, holdout_start=None):
    """Rolling month windows -> list of (tr_rows, va_rows, te_rows) row-index arrays
    into the featurize-once full set. Stride = test_m (default 1).

    holdout_start (e.g. '2026-01-01'): months >= this are EXCLUDED from the rolling
    folds entirely — the WF never sees the holdout, which stays the untouched final
    OOS for produce."""
    per = pd.DatetimeIndex(ts)
    per = per.tz_localize(None) if per.tz is not None else per
    perM = per.to_period('M')
    months = perM.unique().sort_values()
    if holdout_start is not None:
        hs = pd.Period(pd.Timestamp(holdout_start).tz_localize(None)
                       if pd.Timestamp(holdout_start).tzinfo else pd.Timestamp(holdout_start),
                       freq='M')
        months = months[months < hs]          # 2026+ never in any fold
    span = train_m + val_m + test_m
    folds, s = [], 0
    while s + span <= len(months):
        tr = np.flatnonzero(perM.isin(months[s:s + train_m]))
        va = np.flatnonzero(perM.isin(months[s + train_m:s + train_m + val_m]))
        te = np.flatnonzero(perM.isin(months[s + train_m + val_m:s + span]))
        if len(tr) >= 50 and len(va) >= 10 and len(te) >= 10:
            folds.append((tr, va, te))
        s += test_m
        if max_folds and len(folds) >= max_folds:
            break
    return folds


def featurize_all_streams(make_labeler, streams, clf, rundir, chunk=2000, verbose=True):
    """Per-stream sequential featurize of ALL pivots -> full memmap + per-pivot timestamps
    + keys + labels. Loads/releases one stream at a time (bounded RAM)."""
    import gc
    from ._memmap import featurize_to_memmap, concat_memmaps
    from pathlib import Path
    rundir = Path(rundir)
    parts, all_keys, all_y, all_ts = [], [], [], []
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
            p = str(rundir / f'_all{i}.npy')
            _, sh = featurize_to_memmap(clf, lab, list(K), p, chunk)
            parts.append((p, len(K))); all_keys += list(K); all_y += list(Y)
            all_ts += list(ts); C, seq = sh[1], sh[2]
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
            channel_names, eval_lab, C, seq)


def _run_folds(classifier, ck, Xall, Y, keys, eval_lab, rundir, folds, seed, verbose, monitor):
    """Per-fold: slice the full memmap -> fit_predict (REAL/SHUFFLE/RANDOM) -> R + health.
    Given a featurized full set + a config; reused across overfit-loop iterations (no
    re-featurize)."""
    import os as _os
    from ._memmap import slice_memmap
    clf_run = get_classifier(classifier, **ck)
    pool = {'REAL': [], 'SHUFFLE': [], 'RANDOM': []}
    auc_real, val_meanR, test_meanR = [], [], []
    rng = np.random.default_rng(seed)
    for fi, (tr, va, te) in enumerate(folds):
        tr, va, te = np.sort(tr), np.sort(va), np.sort(te)
        ftr, fva, fte = (str(rundir / '_ftr.npy'), str(rundir / '_fva.npy'),
                         str(rundir / '_fte.npy'))
        slice_memmap(Xall, tr, ftr); slice_memmap(Xall, va, fva); slice_memmap(Xall, te, fte)
        Ytr, Yva, Yte = Y[tr], Y[va], Y[te]
        Kte = [keys[i] for i in te]; Kva = [keys[i] for i in va]
        p_val, p_te, ba = clf_run.fit_predict(ftr, Ytr, fva, Yva, fte, seed)
        thr = _pct_threshold(p_val, OP_PERCENTILE)
        R_te = _arm_R(eval_lab, Kte, p_te, thr)
        pool['REAL'].append(R_te); auc_real.append((Yte, p_te))
        val_meanR.append(_meanR(_arm_R(eval_lab, Kva, p_val, thr))); test_meanR.append(_meanR(R_te))
        ysh = Ytr.copy(); rng.shuffle(ysh)
        psv, ps, _ = clf_run.fit_predict(ftr, ysh, fva, Yva, fte, seed)
        pool['SHUFFLE'].append(_arm_R(eval_lab, Kte, ps, _pct_threshold(psv, OP_PERCENTILE)))
        pr = rng.random(len(Kte))
        pool['RANDOM'].append(_arm_R(eval_lab, Kte, pr, _pct_threshold(pr, OP_PERCENTILE)))
        if monitor:
            monitor.check(f'F{fi}', _health_metrics(p_te, Yte, p_val, Yva))
        if verbose:
            from sklearn.metrics import roc_auc_score
            ta = roc_auc_score(Yte, p_te) if len(np.unique(Yte)) == 2 else float('nan')
            print(f"  fold {fi}: train={len(tr)} test={len(te)} test_auc={ta:.4f} "
                  f"meanR={_meanR(R_te):+.3f}", flush=True)
        for f in (ftr, fva, fte):
            try:
                _os.remove(f)
            except OSError:
                pass
    return _verdict(classifier, pool, auc_real, val_meanR, test_meanR, len(folds), verbose)


def _featurize_and_folds(make_labeler, streams, clf, clf_kwargs, train_m, val_m, test_m,
                         max_folds, output_path, chunk, verbose, holdout_start='2026-01-01'):
    from pathlib import Path
    from ._memmap import memmap_standardize_stats
    rundir = (Path(output_path).parent if output_path else Path('.'))
    rundir.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"[wf-stream] featurize-once over {len(streams)} streams ...", flush=True)
    Xall, Y, keys, ts, channel_names, eval_lab, C, seq = featurize_all_streams(
        make_labeler, streams, clf, rundir, chunk, verbose)
    ck = dict(clf_kwargs or {})
    mu = sd = None
    if clf.needs_standardize:
        mu, sd = memmap_standardize_stats(Xall)
        ck['standardize_mu'] = mu.tolist(); ck['standardize_sd'] = sd.tolist()
    folds = _rolling_folds(ts, train_m, val_m, test_m, max_folds, holdout_start=holdout_start)
    if verbose:
        print(f"[wf-stream] {len(folds)} folds (rolling {train_m}/{val_m}/{test_m}, stride "
              f"{test_m}); holdout {holdout_start} EXCLUDED (reserved OOS)", flush=True)
    return rundir, Xall, Y, keys, eval_lab, ck, mu, sd, folds


def run_streamed(make_labeler, streams, classifier='mantis', clf_kwargs=None, train_m=3,
                 val_m=1, test_m=1, max_folds=None, holdout_start='2026-01-01',
                 output_path=None, chunk=2000, seed=0, verbose=True, health_monitor=None):
    """Walk-forward over time on FULL multi-stream data, SINGLE config. Featurize once ->
    rolling folds (holdout_start EXCLUDED = untouched OOS) -> per-fold REAL/SHUFFLE/RANDOM
    + VAL->TEST gap + health. Overfit guards: per-fold val early-stop + gap + health."""
    clf = get_classifier(classifier, **dict(clf_kwargs or {}))
    rundir, Xall, Y, keys, eval_lab, ck, mu, sd, folds = _featurize_and_folds(
        make_labeler, streams, clf, clf_kwargs, train_m, val_m, test_m, max_folds,
        output_path, chunk, verbose, holdout_start=holdout_start)
    monitor = health_monitor or FoldHealthMonitor()
    return _run_folds(classifier, ck, Xall, Y, keys, eval_lab, rundir, folds, seed,
                      verbose, monitor)


def loop_streamed(make_labeler, streams, classifier='mantis', clf_kwargs=None, train_m=3,
                  val_m=1, test_m=1, max_folds=None, holdout_start='2026-01-01', max_iters=2,
                  n_trials=8, output_path=None, chunk=2000, seed=0, verbose=True):
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
                   FoldHealthMonitor())
    history = [dict(it=0, source='default', generalizes=v['generalizes'], gap=v['gap'])]

    if not v['generalizes'] and folds:
        import os as _os
        tr, va = np.sort(folds[0][0]), np.sort(folds[0][1])
        ftr, fva = str(rundir / '_otr.npy'), str(rundir / '_ova.npy')
        slice_memmap(Xall, tr, ftr); slice_memmap(Xall, va, fva)
        Xtr = np.asarray(np.load(ftr, mmap_mode='r'), np.float32)
        Xva = np.asarray(np.load(fva, mmap_mode='r'), np.float32)
        if mu is not None:                                   # pre-standardize for tuning
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
            if mu is not None:
                ck['standardize_mu'] = mu.tolist(); ck['standardize_sd'] = sd.tolist()
            v = _run_folds(classifier, ck, Xall, Y, keys, eval_lab, rundir, folds, seed,
                           verbose, FoldHealthMonitor())
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
