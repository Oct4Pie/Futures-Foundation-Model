"""Walk-forward honest ruler for the fine-tune pipeline — Classifier-agnostic.

Mirrors the pipeline's discipline (leak-free walk-forward x {REAL, SHUFFLE,
RANDOM} x seeds; a number is believed ONLY if REAL clearly beats the controls)
but the MODEL is a pluggable `Classifier` fine-tuned end-to-end on MULTIVARIATE
windows — no frozen embedding, no XGBoost head. Swap Mantis/UniShape/axial via
the classifier name; the ruler is identical.

The labeler must satisfy the pipeline StrategyLabeler protocol (calendar / build /
evaluate) AND expose `mv_contexts(keys) -> np.ndarray [N, C, seq]` (the causal
multivariate window per decision — the input the classifier fine-tunes on).

ISOLATION: imports only walk_forward_folds from pipeline.data (torch/xgboost-free).
The classifier (torch) is loaded lazily via get_classifier — never alongside
xgboost in one process (libomp segfault).
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from futures_foundation.pipeline.data import walk_forward_folds

PASS_LIFT_MARGIN_R = 0.10     # REAL must beat each control by this (realized R)
GEN_GAP_TOL = 0.30            # VAL->TEST meanR gap above this = does not generalize
OP_PERCENTILE = 0.50          # trade the top 50% by proba (usable volume)


def _fit_predict(classifier, kwargs, Xtr, ytr, Xval, yval, Xte, seed):
    """Fit the classifier in an ISOLATED torch subprocess (no xgboost) and return
    (p_val, p_te, best_val_auc). Each call is a fresh process — MPS/RAM freed on
    exit."""
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        for name, arr in [('Xtr', Xtr), ('ytr', ytr), ('Xval', Xval),
                          ('yval', yval), ('Xte', Xte)]:
            np.save(d / f'{name}.npy', arr)
        (d / 'meta.json').write_text(json.dumps(
            dict(classifier=classifier, kwargs=kwargs, seed=int(seed))))
        r = subprocess.run(
            [sys.executable, '-m', 'futures_foundation.finetune.classifiers._worker', str(d)],
            capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"classifier worker failed:\n{r.stderr[-3000:]}")
        ba = float(np.load(d / 'best_val_auc.npy')[0])
        return np.load(d / 'p_val.npy'), np.load(d / 'p_te.npy'), ba


def _pct_threshold(proba, top_pct):
    proba = np.asarray(proba, float)
    if proba.size == 0:
        return 1.0
    return float(np.quantile(proba, 1.0 - top_pct))


def _meanR(R):
    R = np.asarray(R, float)
    return float(R.mean()) if len(R) else 0.0


def _arm_R(labeler, keys, proba, thr):
    """Take the top-`thr` proba decisions -> realized per-trade R via the
    strategy's own evaluate (cost included). preds: 1=take, 0=skip."""
    preds = (np.asarray(proba) >= thr).astype(int)
    if preds.sum() == 0:
        return np.array([])
    return np.asarray(labeler.evaluate(list(keys), preds), float)


def run(labeler, classifier='mantis', clf_kwargs=None, seeds=(0,),
        train_m=3, val_m=1, test_m=1, max_folds=None, holdout_start=None,
        verbose=True):
    """Returns a verdict dict. clf_kwargs forwarded to get_classifier (n_channels
    is injected from the data)."""
    clf_kwargs = dict(clf_kwargs or {})
    pool = {'REAL': [], 'SHUFFLE': [], 'RANDOM': []}     # realized R per arm (test)
    auc_real = []                                        # pooled (yte, proba) REAL
    val_meanR, test_meanR = [], []                       # for the gen-gap
    n_folds = 0

    for fold, tr, val, te in walk_forward_folds(labeler.calendar(), train_m,
                                                val_m, test_m,
                                                holdout_start=holdout_start):
        if max_folds is not None and n_folds >= max_folds:
            break
        val0, te0 = val['timestamp'].min(), te['timestamp'].min()
        Ctr, Ytr, Ktr = labeler.build(tr['timestamp'].min(), tr['timestamp'].max(), val0)
        Cval, Yval, Kval = labeler.build(val0, val['timestamp'].max(), te0)
        Cte, Yte, Kte = labeler.build(te['timestamp'].min(),
                                      te['timestamp'].max() + np.timedelta64(1, 'ns'), None)
        Ytr, Yval, Yte = map(lambda a: np.asarray(a).astype(int), (Ytr, Yval, Yte))
        if len(Ytr) < 50 or len(Cte) < 50 or len(Cval) < 10:
            continue
        Xtr = np.asarray(labeler.mv_contexts(Ktr), np.float32)
        Xval = np.asarray(labeler.mv_contexts(Kval), np.float32)
        Xte = np.asarray(labeler.mv_contexts(Kte), np.float32)
        C = Xtr.shape[1]
        # per-channel standardize on TRAIN stats only (no leak across the split)
        flat = Xtr.transpose(0, 2, 1).reshape(-1, C)
        mu, sd = flat.mean(0), flat.std(0) + 1e-6
        def _std(A):
            return ((A - mu[None, :, None]) / sd[None, :, None]).astype(np.float32)
        Xtr, Xval, Xte = _std(Xtr), _std(Xval), _std(Xte)
        n_folds += 1
        if verbose:
            print(f"\n[fold {fold}] train={len(Ytr)} val={len(Yval)} test={len(Yte)} "
                  f"C={C} seq={Xtr.shape[2]} good={Ytr.mean():.3f}", flush=True)

        for seed in seeds:
            rng = np.random.default_rng(seed)
            # ---- REAL ---- (isolated torch subprocess)
            p_val, p_te, ba = _fit_predict(classifier, clf_kwargs, Xtr, Ytr,
                                           Xval, Yval, Xte, seed)
            thr = _pct_threshold(p_val, OP_PERCENTILE)
            R_te = _arm_R(labeler, Kte, p_te, thr)
            R_val = _arm_R(labeler, Kval, p_val, thr)
            pool['REAL'].append(R_te)
            auc_real.append((Yte, p_te))
            val_meanR.append(_meanR(R_val)); test_meanR.append(_meanR(R_te))
            if verbose:
                from sklearn.metrics import roc_auc_score
                te_auc = (roc_auc_score(Yte, p_te) if len(np.unique(Yte)) == 2 else float('nan'))
                print(f"  seed{seed} REAL: best_val_auc={ba:.4f} test_auc={te_auc:.4f} "
                      f"meanR={_meanR(R_te):+.3f}", flush=True)
            # ---- SHUFFLE (train labels permuted) ----
            ysh = Ytr.copy(); rng.shuffle(ysh)
            psv, ps, _ = _fit_predict(classifier, clf_kwargs, Xtr, ysh, Xval, Yval, Xte, seed)
            pool['SHUFFLE'].append(_arm_R(labeler, Kte, ps, _pct_threshold(psv, OP_PERCENTILE)))
            # ---- RANDOM (proba ~ U) ----
            pr = rng.random(len(Kte))
            pool['RANDOM'].append(_arm_R(labeler, Kte, pr, _pct_threshold(pr, OP_PERCENTILE)))

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

    checks = [
        (real_m - shuf_m >= PASS_LIFT_MARGIN_R, f"REAL-SHUFFLE >={PASS_LIFT_MARGIN_R}R "
         f"({real_m-shuf_m:+.2f}R)"),
        (real_m - rand_m >= PASS_LIFT_MARGIN_R, f"REAL-RANDOM >={PASS_LIFT_MARGIN_R}R "
         f"({real_m-rand_m:+.2f}R)"),
        (gap is not None and gap <= GEN_GAP_TOL, f"GENERALIZES VAL->TEST gap <={GEN_GAP_TOL}R "
         f"({gap:+.2f}R)" if gap is not None else "no val/test"),
    ]
    all_pass = all(ok for ok, _ in checks)
    verdict = dict(all_pass=all_pass, auc=auc, real_meanR=real_m, shuffle_meanR=shuf_m,
                   random_meanR=rand_m, gap=gap, n_folds=n_folds,
                   real_trades=len(cat('REAL')))
    if verbose:
        print(f"\n=== WF HONEST RULER ({classifier}, folds={n_folds}) ===")
        print(f"  pooled TEST AUC {auc:.4f}" if auc is not None else "  AUC n/a")
        print(f"  meanR  REAL {real_m:+.3f}  SHUFFLE {shuf_m:+.3f}  RANDOM {rand_m:+.3f}  "
              f"(trades={len(cat('REAL'))})")
        for ok, msg in checks:
            print(f"  [{'PASS' if ok else 'FAIL'}] {msg}")
        print(f"  -> {'ALL PASS' if all_pass else 'FAIL'}")
    return verdict
