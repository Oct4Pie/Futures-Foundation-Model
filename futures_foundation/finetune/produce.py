"""Final-model training + held-out OOS — the 'produce' half (mirrors pipeline.produce).

Two-step, same as the Chronos process:
  1. wf.run(holdout_start='2026-01-01')  -> walk-forward honest ruler over PRE-2026
     folds (does the edge generalize?).  loop.train_loop wraps this with the
     overfit→Optuna→rerun guard.
  2. produce.train_final(holdout_start='2026-01-01') -> train the FINAL model on ALL
     data < 2026 (inner val + early-stop) and score the held-out 2026 OOS ONCE, with
     a SHUFFLE control.  2026 is touched exactly once, at the end.

Classifier-agnostic (Mantis/logistic/future backbones via the Classifier seam).
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .classifier import get_classifier
from .wf import (_pct_threshold, _arm_R, _meanR, _standardize_on_train,
                 OP_PERCENTILE, PASS_LIFT_MARGIN_R)


def _write_artifacts(labeler, classifier, clf_kwargs, Xtr, ytr, Xval, yval, Xte,
                     mu, sd, out, output_path, holdout_start, n_train, n_oos, seed):
    """Export the final model to ONNX and write the self-describing signal contract
    `<base>_signal.json` next to it (mirrors pipeline.produce's signal contract). The
    contract tells the bot how to reproduce the input (channel features + standardize
    stats) and validate integrity (sha) before serving the ONNX."""
    base = Path(output_path).with_suffix('')
    onnx_path = str(base) + '.onnx'
    # final export fit (trains + exports inside the isolated worker, if supported)
    clf_exp = get_classifier(classifier, **dict(clf_kwargs or {}, export_onnx_path=onnx_path))
    clf_exp.fit_predict(Xtr, ytr, Xval, yval, Xte, seed)
    sha = (hashlib.sha256(Path(onnx_path).read_bytes()).hexdigest()
           if Path(onnx_path).exists() else None)
    tfs = sorted({k[1] for k in getattr(labeler, '_b', {})}) or None
    tks = sorted({k[0] for k in getattr(labeler, '_b', {})}) or None
    contract = {
        'contract_version': '1.0',
        'role': 'mantis_signal',
        'classifier': classifier,
        'input': {'channels': int(Xtr.shape[1]), 'seq_len': int(Xtr.shape[2]),
                  'mv_mode': getattr(labeler, 'MV_MODE', None)},
        'channel_names': (labeler.mv_feature_names()
                          if hasattr(labeler, 'mv_feature_names') else None),
        'standardize': ({'mu': np.asarray(mu).tolist(), 'sd': np.asarray(sd).tolist()}
                        if mu is not None else None),
        'mv_contexts_fn': 'strategy.mv_contexts (direction-normalized causal window)',
        'nan_policy': {'posinf': 0, 'neginf': 0, 'nan': 0},
        'ft_config': dict(clf_kwargs or {}),
        'n_classes': 2,
        'proba_meaning': 'P(good trend pivot reaches target before stop)',
        'output_fn': 'softmax(logits)[:,1]',
        'train_scope': {'tickers': tks, 'timeframes': tfs, 'holdout_start': holdout_start,
                        'n_train': int(n_train), 'n_oos': int(n_oos)},
        'oos_metrics': {k: out.get(k) for k in
                        ('oos_auc', 'oos_meanR', 'shuffle_meanR', 'edge_shuffle', 'oos_trades')},
        'onnx': (Path(onnx_path).name if sha else None),
        'content_sha': sha,
    }
    cpath = str(base) + '_signal.json'
    Path(cpath).write_text(json.dumps(contract, indent=2))
    return {'onnx': onnx_path if sha else None, 'contract': cpath, 'content_sha': sha}


def train_final(labeler, classifier='mantis', clf_kwargs=None, holdout_start='2026-01-01',
                val_frac=0.15, seed=0, max_train=None, export_onnx=False,
                output_path=None, verbose=True):
    clf = get_classifier(classifier, **dict(clf_kwargs or {}))
    hs = pd.Timestamp(holdout_start, tz='UTC')
    cal = labeler.calendar()
    lo, hi = cal['timestamp'].min(), cal['timestamp'].max()
    Ctr, Ytr, Ktr = labeler.build(lo, hs, hs)
    Cte, Yte, Kte = labeler.build(hs, hi + pd.Timedelta('1ns'), None)
    Ytr = np.asarray(Ytr).astype(int); Yte = np.asarray(Yte).astype(int)
    if len(Ytr) < 50 or len(Kte) < 20:
        raise ValueError(f"insufficient data: train={len(Ytr)} oos={len(Kte)}")
    # cap train BEFORE featurizing (lean parent)
    if max_train and len(Ktr) > max_train:
        sub = np.random.default_rng(seed).choice(len(Ktr), max_train, replace=False)
        Ktr = [Ktr[j] for j in sub]; Ytr = Ytr[sub]
    # inner train/val carve for early-stop (val is NEVER the OOS)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(Ktr)); rng.shuffle(idx)
    nv = max(10, int(len(idx) * val_frac))
    va_i, tr_i = idx[:nv], idx[nv:]
    Ktr_tr = [Ktr[j] for j in tr_i]; Ytr_tr = Ytr[tr_i]
    Ktr_va = [Ktr[j] for j in va_i]; Ytr_va = Ytr[va_i]

    Xtr = clf.featurize(labeler, Ktr_tr)
    Xval = clf.featurize(labeler, Ktr_va)
    Xte = clf.featurize(labeler, Kte)
    mu = sd = None
    if clf.needs_standardize:
        Xtr, Xval, Xte, mu, sd = _standardize_on_train(Xtr, Xval, Xte)
    if verbose:
        print(f"=== PRODUCE ({classifier}: train < {holdout_start}, 2026 OOS) ===")
        print(f"  train={len(tr_i)} val={len(va_i)} oos={len(Kte)} X={tuple(Xtr.shape[1:])} "
              f"good(train)={Ytr_tr.mean():.3f} good(oos)={Yte.mean():.3f}", flush=True)

    p_val, p_te, ba = clf.fit_predict(Xtr, Ytr_tr, Xval, Ytr_va, Xte, seed)
    thr = _pct_threshold(p_val, OP_PERCENTILE)
    R = _arm_R(labeler, Kte, p_te, thr)
    ysh = Ytr_tr.copy(); rng.shuffle(ysh)
    psv, ps, _ = clf.fit_predict(Xtr, ysh, Xval, Ytr_va, Xte, seed)
    Rs = _arm_R(labeler, Kte, ps, _pct_threshold(psv, OP_PERCENTILE))

    auc = None
    if len(np.unique(Yte)) == 2:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(Yte, p_te))
    edge = _meanR(R) - _meanR(Rs)
    out = dict(oos_auc=auc, best_val_auc=ba, oos_meanR=_meanR(R), shuffle_meanR=_meanR(Rs),
               edge_shuffle=edge, n_train=len(tr_i), n_oos=len(Kte), oos_trades=int(len(R)),
               beats_shuffle=bool(edge >= PASS_LIFT_MARGIN_R))
    if verbose:
        print(f"  OOS AUC {auc:.4f}" if auc is not None else "  OOS AUC n/a")
        print(f"  OOS meanR REAL {out['oos_meanR']:+.3f} SHUFFLE {out['shuffle_meanR']:+.3f} "
              f"edge {edge:+.3f} (trades={out['oos_trades']})")
        print(f"  -> {'beats SHUFFLE' if out['beats_shuffle'] else 'does NOT beat SHUFFLE'}")
    if export_onnx and output_path:
        out['artifacts'] = _write_artifacts(labeler, classifier, clf_kwargs, Xtr, Ytr_tr,
                                            Xval, Ytr_va, Xte, mu, sd, out, output_path,
                                            holdout_start, len(tr_i), len(Kte), seed)
        if verbose:
            print(f"  artifacts: onnx={out['artifacts']['onnx']} "
                  f"contract={out['artifacts']['contract']}")
    return out
