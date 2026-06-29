"""Subprocess worker — fit a Classifier (torch) in ISOLATION and write predictions.

Run as: python -m futures_foundation.finetune.classifiers._worker <workdir>
Reads  : Xtr.npy, ytr.npy, Xval.npy, yval.npy, Xte.npy, meta.json{classifier,kwargs,seed}
Writes : p_val.npy, p_te.npy, best_val_auc.npy

WHY: the strategy labeler pulls xgboost (pipeline.evaluate) and Mantis pulls torch
— the two segfault in one macOS process (libomp). The parent (wf.py) stays
torch-free; this child loads ONLY torch + the classifier. Each invocation is a
fresh process, so MPS/RAM is fully released on exit (no accumulation/freeze).
"""
import json
import sys
from pathlib import Path

import numpy as np


def main(wd):
    wd = Path(wd)
    meta = json.loads((wd / 'meta.json').read_text())
    Xtr, ytr = np.load(wd / 'Xtr.npy'), np.load(wd / 'ytr.npy')
    Xval, yval = np.load(wd / 'Xval.npy'), np.load(wd / 'yval.npy')
    Xte = np.load(wd / 'Xte.npy')

    from futures_foundation.finetune.classifier import get_classifier
    clf = get_classifier(meta['classifier'], n_channels=Xtr.shape[1], **meta.get('kwargs', {}))
    clf.fit(Xtr, ytr, Xval, yval, seed=meta.get('seed', 0))

    np.save(wd / 'p_val.npy', clf.predict_proba(Xval))
    np.save(wd / 'p_te.npy', clf.predict_proba(Xte))
    np.save(wd / 'best_val_auc.npy',
            np.array([float(getattr(clf, 'best_val_auc', float('nan')))]))


if __name__ == '__main__':
    main(sys.argv[1])
