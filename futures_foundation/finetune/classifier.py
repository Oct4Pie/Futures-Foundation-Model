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


def get_classifier_class(name):
    """Return the registered Classifier CLASS, importing its backbone from the plugin manifest if
    needed. For introspection (e.g. a tuner reading a backbone's `suggest_params`); build an
    instance with get_classifier(). Keeps this interface torch-free + backbone-name-free."""
    if name not in _REGISTRY:
        import importlib
        from .classifiers import LAZY_BACKBONES
        mod = LAZY_BACKBONES.get(name)
        if mod is None:
            raise KeyError(f"unknown classifier '{name}'. registered: {sorted(_REGISTRY)}; "
                           f"available: {sorted(LAZY_BACKBONES)}")
        importlib.import_module(mod)                    # self-registers via @register_classifier
    return _REGISTRY[name]


def get_classifier(name, **kwargs) -> Classifier:
    """Instantiate a registered classifier by name (backbone imported lazily from the manifest)."""
    return get_classifier_class(name)(**kwargs)


def base_backbone_ckpt(backbone=None):
    """DI accessor: the default SSL foundation checkpoint a new strategy finetunes on top of,
    resolved from the ACTIVE backbone (no hard-coded foundation path in generic code). Each
    backbone package exposes `BASE_CKPT`; default = the manifest's DEFAULT_BACKBONE."""
    import importlib
    from .classifiers import LAZY_BACKBONES, DEFAULT_BACKBONE
    pkg = importlib.import_module(LAZY_BACKBONES[backbone or DEFAULT_BACKBONE])
    return pkg.BASE_CKPT


def __getattr__(name):
    # Back-compat (PEP 562): `from ...classifier import BASE_BACKBONE_CKPT` resolves LAZILY to the
    # active backbone's base ckpt, so existing strategy colabs keep working without naming a
    # backbone in generic code. Deferred so importing this module stays torch-free.
    if name == 'BASE_BACKBONE_CKPT':
        return base_backbone_ckpt()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
