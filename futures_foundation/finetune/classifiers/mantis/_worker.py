"""Subprocess worker — runs the Mantis torch fine-tune in ISOLATION.

Run as: python -m futures_foundation.finetune.classifiers.mantis._worker <workdir>
Reads  : cfg.json (incl _paths to the X memmaps/npys + ft params + standardize mu/sd
         + seed), ytr.npy, yval.npy
Writes : p_val.npy, p_eval.npy, meta.npy = [best_val_auc, best_epoch]

X arrays are opened mmap_mode='r' so a full-data feature memmap never loads into RAM;
the trainer pages batches on demand. Loads only torch + Mantis (never xgboost).
"""
import json
import sys
from pathlib import Path

import numpy as np


def main(wd):
    wd = Path(wd)
    cfg = json.loads((wd / 'cfg.json').read_text())
    paths = cfg.pop('_paths')
    Xtr = np.load(paths['Xtr'], mmap_mode='r')
    Xval = np.load(paths['Xval'], mmap_mode='r')
    Xeval = np.load(paths['Xeval'], mmap_mode='r')
    ytr = np.load(wd / 'ytr.npy')
    yval = np.load(wd / 'yval.npy')
    from futures_foundation.finetune.classifiers.mantis._torch import fit_predict_torch
    p_val, p_eval, ba, be = fit_predict_torch(Xtr, ytr, Xval, yval, Xeval, **cfg)
    np.save(wd / 'p_val.npy', p_val)
    np.save(wd / 'p_eval.npy', p_eval)
    np.save(wd / 'meta.npy', np.array([ba, be], float))


if __name__ == '__main__':
    main(sys.argv[1])
