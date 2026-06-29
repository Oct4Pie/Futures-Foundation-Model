"""Classifier seam + registry + torch-free LogisticClassifier.

Torch-free — runs in the default suite (no CHRONOS_TORCH_TESTS gate). Mantis is
tested separately in test_mantis_classifier.py (torch-gated).
"""
import numpy as np
import pytest

from futures_foundation.finetune.classifier import (
    Classifier, get_classifier, register_classifier, _REGISTRY)


def test_abc_cannot_instantiate():
    with pytest.raises(TypeError):
        Classifier()                       # abstract: fit/predict_proba unimplemented


def test_registry_and_get_classifier():
    @register_classifier('_unit_dummy')
    class _Dummy(Classifier):
        def __init__(self, n_channels, **kw):
            self.n_channels = n_channels
        def fit(self, X, y, X_val=None, y_val=None, seed=0):
            return self
        def predict_proba(self, X):
            return np.zeros(len(X))
    assert '_unit_dummy' in _REGISTRY
    clf = get_classifier('_unit_dummy', n_channels=3)
    assert isinstance(clf, Classifier) and clf.n_channels == 3


def test_get_classifier_unknown_raises():
    with pytest.raises(KeyError):
        get_classifier('does_not_exist', n_channels=1)


def test_get_classifier_lazy_loads_logistic():
    clf = get_classifier('logistic', n_channels=4)
    assert isinstance(clf, Classifier)


def test_logistic_fit_predict_shapes_and_range():
    rng = np.random.default_rng(0)
    N, C, T = 200, 4, 16
    X = rng.standard_normal((N, C, T)).astype(np.float32)
    y = rng.integers(0, 2, N)
    clf = get_classifier('logistic', n_channels=C)
    clf.fit(X[:150], y[:150], X[150:], y[150:])
    p = clf.predict_proba(X[150:])
    assert p.shape == (50,)
    assert np.all((p >= 0) & (p <= 1))


def test_logistic_learns_separable_signal():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(1)
    N, C, T = 400, 4, 16
    y = rng.integers(0, 2, N)
    X = rng.standard_normal((N, C, T)).astype(np.float32)
    X[y == 1, 0, -4:] += 2.0                # class-1 signal in channel 0, last bars
    tr = slice(0, 300)
    clf = get_classifier('logistic', n_channels=C)
    clf.fit(X[tr], y[tr], X[300:], y[300:])
    auc = roc_auc_score(y[300:], clf.predict_proba(X[300:]))
    assert auc > 0.85
    assert np.isfinite(clf.best_val_auc) and clf.best_val_auc > 0.85
