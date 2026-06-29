"""Classifier seam — the single pluggable model interface for the fine-tune
pipeline.

A Classifier maps multivariate windows X:[N, C, seq] + labels y:[N] to a trained
model that emits P(class 1). Implementations OWN their training (Mantis fine-tune,
UniShape, a fresh transformer, ...); the walk-forward honest ruler (`wf.py`) only
ever calls `fit` / `predict_proba`, so swapping backends is a one-line change.

IMPORT CONTRACT: this module is torch-free at import time — concrete impls (which
import torch) are lazy-loaded in `get_classifier`, so the torch-free `finetune`
parent can import the seam eagerly (libomp isolation — torch+xgboost segfault in
one macOS process).
"""
from abc import ABC, abstractmethod

import numpy as np


class Classifier(ABC):
    """End-to-end classifier over multivariate windows.

    fit(X, y, X_val, y_val, seed): train (impls SHOULD early-stop on the val
        split if given; else carve one from train). X:[N,C,seq] float32, y:[N] int.
    predict_proba(X) -> [N] float  P(class 1).
    """

    n_classes: int = 2

    @abstractmethod
    def fit(self, X, y, X_val=None, y_val=None, seed=0):
        ...

    @abstractmethod
    def predict_proba(self, X) -> np.ndarray:
        ...

    # optional — override where the backend supports it
    def pretrain(self, X_unlabeled):
        """Optional domain-adaptation (e.g. contrastive pretrain on raw OHLCV)
        BEFORE the supervised fit. No-op by default."""
        return self

    def save(self, path):
        raise NotImplementedError

    @classmethod
    def load(cls, path):
        raise NotImplementedError


_REGISTRY = {}


def register_classifier(name):
    def deco(cls):
        _REGISTRY[name] = cls
        return cls
    return deco


def get_classifier(name, **kwargs) -> Classifier:
    """Instantiate a registered classifier by name. Concrete impls are imported
    lazily here (keeps this module + the finetune parent torch-free)."""
    if name not in _REGISTRY:
        if name == 'mantis':
            from .classifiers.mantis import MantisClassifier      # noqa: F401
        elif name == 'logistic':
            from .classifiers.logistic import LogisticClassifier  # noqa: F401
        else:
            raise KeyError(f"unknown classifier '{name}'. "
                           f"registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)
