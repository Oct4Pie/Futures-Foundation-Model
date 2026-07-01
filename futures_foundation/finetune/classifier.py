"""Classifier seam — the single pluggable model interface for the unified
walk-forward harness (wf.py), overfit loop (loop.py), and Optuna tuner (tune.py).

ONE harness, any model. A Classifier owns its featurization and its fit/predict,
so Chronos+XGBoost and Mantis (and logistic, and future backbones) all run through
the *same* leak-free walk-forward × {REAL, SHUFFLE, RANDOM} × health × overfit-loop
× Optuna — no per-model duplication.

Contract:
  featurize(labeler, keys) -> X            parent-side, numpy (no torch in parent).
                                           Mantis: mv_contexts [N,C,seq].
                                           Chronos: embed+features [N,D].
  fit_predict(Xtr, ytr, Xval, yval, Xeval, seed) -> (p_val, p_eval, best_val_auc)
                                           in-process (XGBoost/logistic) OR an
                                           isolated torch subprocess (Mantis).
  needs_standardize : bool                 harness standardizes featurized X on
                                           TRAIN stats before fit_predict (Mantis
                                           yes; Chronos embeddings already scaled).

IMPORT CONTRACT: torch-free at import time. Concrete impls that need torch keep it
in a subprocess worker; their parent-side adapter stays torch-free so it never
shares a process with xgboost (libomp segfault).
"""
from abc import ABC, abstractmethod

import numpy as np

# ── BASE FOUNDATION CHECKPOINT — the default starting point for ANY new strategy finetune ──
# New strategies finetune ON TOP of the best SSL foundation, never from vanilla Mantis. This is
# the SINGLE source of truth: point it at whichever SSL checkpoint currently wins the OHLCV-only
# WR@3R benchmark, and every strategy inherits the lift automatically. Update in ONE place.
#   stage-2 (ctx200 candle seq2seq)      = mantis_ssl_seq2seq.pt     ← current base (+4.8 vs vanilla)
#   stage-3 (trend contrastive)          = mantis_ssl_contrastive.pt ← promote here IFF it beats stage-2
BASE_BACKBONE_CKPT = 'checkpoints/mantis_ssl_seq2seq.pt'


class Classifier(ABC):
    n_classes: int = 2
    needs_standardize: bool = False

    @abstractmethod
    def featurize(self, labeler, keys) -> np.ndarray:
        """Parent-side featurization (numpy). Returns X aligned 1:1 to keys."""
        ...

    @abstractmethod
    def fit_predict(self, Xtr, ytr, Xval, yval, Xeval, seed=0):
        """Fit on (Xtr,ytr) with (Xval,yval) for early-stop/guard, return
        (p_val, p_eval, best_val_auc) where p_* are P(class 1) for Xval/Xeval."""
        ...


_REGISTRY = {}


def register_classifier(name):
    def deco(cls):
        _REGISTRY[name] = cls
        return cls
    return deco


def get_classifier(name, **kwargs) -> Classifier:
    """Instantiate a registered classifier by name (concrete impls imported lazily
    so this module + the finetune parent stay torch-free)."""
    if name not in _REGISTRY:
        if name == 'mantis':
            from .classifiers.mantis import MantisClassifier        # noqa: F401
        elif name == 'mantis_frozen':
            from .classifiers.mantis_frozen import MantisFrozenClassifier   # noqa: F401
        elif name == 'logistic':
            from .classifiers.logistic import LogisticClassifier    # noqa: F401
        else:
            raise KeyError(f"unknown classifier '{name}'. registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)
