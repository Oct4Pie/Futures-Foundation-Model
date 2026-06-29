"""Final-model training + 2026 OOS — the 'produce' half (mirrors pipeline.produce).

Two-step, same as the Chronos process:
  1. wf.run(..., holdout_start='2026-01-01')  -> walk-forward honest ruler over
     PRE-2026 folds: does the edge generalize across time? (research/validation)
  2. produce.train_final(..., holdout_start='2026-01-01') -> train the FINAL model
     on ALL data < 2026 (inner val for early-stop) and score the held-out 2026 OOS
     ONCE, with a SHUFFLE control. 2026 is touched exactly once, at the end.

Classifier-agnostic; the classifier fits in an isolated torch subprocess (no
xgboost). Returns the OOS verdict dict.
"""
import numpy as np
import pandas as pd

from .wf import (_fit_predict, _pct_threshold, _arm_R, _meanR,
                 OP_PERCENTILE, PASS_LIFT_MARGIN_R)


def train_final(labeler, classifier='mantis', clf_kwargs=None, holdout_start='2026-01-01',
                val_frac=0.15, seed=0, max_train=None, verbose=True):
    clf_kwargs = dict(clf_kwargs or {})
    hs = pd.Timestamp(holdout_start, tz='UTC')
    cal = labeler.calendar()
    lo, hi = cal['timestamp'].min(), cal['timestamp'].max()
    # train < 2026 (purged so labels can't reach the holdout); 2026 = OOS
    Ctr, Ytr, Ktr = labeler.build(lo, hs, hs)
    Cte, Yte, Kte = labeler.build(hs, hi + pd.Timedelta('1ns'), None)
    Ytr = np.asarray(Ytr).astype(int); Yte = np.asarray(Yte).astype(int)
    if len(Ytr) < 50 or len(Kte) < 20:
        raise ValueError(f"insufficient data: train={len(Ytr)} oos={len(Kte)}")
    # cap train BEFORE building contexts (parent stays light — don't hold 67k windows)
    if max_train and len(Ktr) > max_train:
        sub = np.random.default_rng(seed).choice(len(Ktr), max_train, replace=False)
        Ktr = [Ktr[j] for j in sub]; Ytr = Ytr[sub]
    Xtr = np.asarray(labeler.mv_contexts(Ktr), np.float32)
    Xte = np.asarray(labeler.mv_contexts(Kte), np.float32)
    C = Xtr.shape[1]
    # standardize on TRAIN stats only (no leak into OOS)
    flat = Xtr.transpose(0, 2, 1).reshape(-1, C)
    mu, sd = flat.mean(0), flat.std(0) + 1e-6
    def _std(A):
        return ((A - mu[None, :, None]) / sd[None, :, None]).astype(np.float32)
    Xtr, Xte = _std(Xtr), _std(Xte)
    # inner train/val carve for early-stop (val is NEVER the OOS)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(Xtr)); rng.shuffle(idx)
    nv = max(10, int(len(idx) * val_frac))
    va_i, tr_i = idx[:nv], idx[nv:]

    if verbose:
        print(f"=== PRODUCE ({classifier}: final train < {holdout_start}, 2026 OOS) ===")
        print(f"  train={len(tr_i)} val={len(va_i)} oos={len(Kte)} C={C} "
              f"good(train)={Ytr[tr_i].mean():.3f} good(oos)={Yte.mean():.3f}", flush=True)

    # REAL
    p_val, p_te, ba = _fit_predict(classifier, clf_kwargs, Xtr[tr_i], Ytr[tr_i],
                                   Xtr[va_i], Ytr[va_i], Xte, seed)
    thr = _pct_threshold(p_val, OP_PERCENTILE)
    R = _arm_R(labeler, Kte, p_te, thr)
    # SHUFFLE control (train labels permuted)
    ysh = Ytr[tr_i].copy(); rng.shuffle(ysh)
    psv, ps, _ = _fit_predict(classifier, clf_kwargs, Xtr[tr_i], ysh,
                              Xtr[va_i], Ytr[va_i], Xte, seed)
    Rs = _arm_R(labeler, Kte, ps, _pct_threshold(psv, OP_PERCENTILE))

    auc = None
    if len(np.unique(Yte)) == 2:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(Yte, p_te))
    edge = _meanR(R) - _meanR(Rs)
    out = dict(oos_auc=auc, best_val_auc=ba, oos_meanR=_meanR(R),
               shuffle_meanR=_meanR(Rs), edge_shuffle=edge,
               n_train=len(tr_i), n_oos=len(Kte), oos_trades=int(len(R)),
               beats_shuffle=bool(edge >= PASS_LIFT_MARGIN_R))
    if verbose:
        print(f"  OOS AUC {auc:.4f}" if auc is not None else "  OOS AUC n/a")
        print(f"  OOS meanR  REAL {out['oos_meanR']:+.3f}  SHUFFLE {out['shuffle_meanR']:+.3f}  "
              f"edge {edge:+.3f}  (trades={out['oos_trades']})")
        print(f"  -> {'beats SHUFFLE' if out['beats_shuffle'] else 'does NOT beat SHUFFLE'}")
    return out
