"""MantisFrozenClassifier — FROZEN backbone, only a head trains (the head-only path).

featurize() runs the strategy's multivariate windows through the FROZEN Mantis encoder ONCE
(via the isolated _embed_worker, encoder-only + interpolated to native length) -> a cached
embedding [N, C*hidden]. fit_predict() then trains a cheap HEAD on those embeddings per fold
(torch-free sklearn) — the backbone is never updated. This is the Chronos+XGBoost "embed
once -> head per fold" pattern, but with the frozen masked-SSL Mantis encoder.

backbone_ckpt -> the masked-SSL encoder (the A/B "SSL" arm); None -> vanilla Mantis (the
"vanilla" arm). head='logistic' (linear probe of the frozen rep, default) or 'mlp'.
Registered as 'mantis_frozen'.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from ..classifier import Classifier, register_classifier

_EMBED_KEYS = ('model_id', 'device', 'batch')


@register_classifier('mantis_frozen')
class MantisFrozenClassifier(Classifier):
    needs_standardize = True            # harness standardizes the cached embeddings on train

    def __init__(self, **cfg):
        self.cfg = cfg

    def featurize(self, labeler, keys):
        windows = np.asarray(labeler.mv_contexts(keys), np.float32)    # [N, C, seq]
        ecfg = {k: self.cfg[k] for k in _EMBED_KEYS if k in self.cfg}
        ecfg['ckpt'] = self.cfg.get('backbone_ckpt')                   # SSL ckpt or None
        cmd = [sys.executable, '-u', '-m',
               'futures_foundation.finetune.classifiers._embed_worker']
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            np.save(d / 'w.npy', windows)
            (d / 'cfg.json').write_text(json.dumps(dict(ecfg, _windows=str(d / 'w.npy'))))
            r = subprocess.run(cmd + [str(d)], capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(f"embed worker failed:\n{r.stderr[-2000:]}")
            return np.load(d / 'emb.npy')

    def fit_predict(self, Xtr, ytr, Xval, yval, Xeval, seed=0):
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier
        from sklearn.metrics import roc_auc_score

        def arr(a):
            return np.asarray(np.load(a, mmap_mode='r') if isinstance(a, str) else a)
        Xtr, Xval, Xeval = arr(Xtr), arr(Xval), arr(Xeval)
        ytr = np.asarray(ytr).astype(int); yval = np.asarray(yval).astype(int)
        if len(np.unique(ytr)) < 2:
            return np.full(len(Xval), .5), np.full(len(Xeval), .5), 0.5
        if self.cfg.get('head', 'logistic') == 'mlp':
            clf = MLPClassifier(hidden_layer_sizes=tuple(self.cfg.get('hidden', (128,))),
                                max_iter=int(self.cfg.get('max_iter', 300)),
                                early_stopping=True, random_state=seed)
        else:
            clf = LogisticRegression(max_iter=1000, C=float(self.cfg.get('C', 1.0)))
        clf.fit(Xtr, ytr)
        p_val = clf.predict_proba(Xval)[:, 1]
        p_eval = clf.predict_proba(Xeval)[:, 1]
        auc = roc_auc_score(yval, p_val) if len(np.unique(yval)) == 2 else 0.5
        return p_val, p_eval, float(auc)
