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
