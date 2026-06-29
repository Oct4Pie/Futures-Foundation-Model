"""MantisClassifier (torch-gated).

torch + the Mantis backbone load here, so the suite is gated behind
CHRONOS_TORCH_TESTS=1 (libomp isolation — see tests/conftest.py) and torch is
imported inside test bodies. Kept tiny (few epochs, small N, CPU) — these check
WIRING (fit/predict shapes, freeze policy, separability), not production AUC.

Run: CHRONOS_TORCH_TESTS=1 pytest tests/test_mantis_classifier.py
"""
import os

import numpy as np
import pytest

torch_test = pytest.mark.skipif(
    os.environ.get('CHRONOS_TORCH_TESTS') != '1',
    reason='torch test — set CHRONOS_TORCH_TESTS=1 (libomp isolation)')


def _toy(seed=0, N=120, C=4, T=64):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, N)
    X = rng.standard_normal((N, C, T)).astype(np.float32)
    X[y == 1, 0, -8:] += 2.0                       # separable signal
    return X, y


@torch_test
def test_fit_predict_shapes_and_range():
    from futures_foundation.finetune.classifier import get_classifier
    X, y = _toy()
    clf = get_classifier('mantis', n_channels=4, ft_mode='partial', epochs=2,
                         batch=32, threads=2, device='cpu', verbose=False)
    clf.fit(X[:90], y[:90], X[90:], y[90:], seed=0)
    p = clf.predict_proba(X[90:])
    assert p.shape == (30,)
    assert np.all((p >= 0) & (p <= 1))
    assert np.isfinite(getattr(clf, 'best_val_auc', np.nan))


@torch_test
def test_partial_mode_freezes_backbone():
    import torch                                    # noqa: F401
    from futures_foundation.finetune.classifiers.mantis import MantisClassifier
    clf = MantisClassifier(n_channels=4, ft_mode='partial', unfreeze_blocks=2,
                           device='cpu', verbose=False)
    model = clf._build_model()
    layers = model.encoder.vit_unit.transformer.layers
    # last 2 blocks trainable, an earlier block frozen
    assert all(p.requires_grad for p in layers[-1].parameters())
    assert not any(p.requires_grad for p in layers[0].parameters())
    # head + adapter always trainable
    assert all(p.requires_grad for p in model.head.parameters())


@torch_test
def test_head_mode_freezes_all_backbone():
    from futures_foundation.finetune.classifiers.mantis import MantisClassifier
    clf = MantisClassifier(n_channels=4, ft_mode='head', device='cpu', verbose=False)
    model = clf._build_model()
    assert not any(p.requires_grad for p in model.encoder.parameters())
    assert all(p.requires_grad for p in model.head.parameters())


@torch_test
def test_max_train_subsamples():
    from futures_foundation.finetune.classifiers.mantis import MantisClassifier
    X, y = _toy(N=200)
    clf = MantisClassifier(n_channels=4, ft_mode='head', epochs=1, batch=32,
                           device='cpu', max_train=50, verbose=False)
    clf.fit(X, y, seed=0)                           # internal val split + cap to 50
    assert clf.predict_proba(X[:10]).shape == (10,)


@torch_test
def test_learns_separable_signal():
    from sklearn.metrics import roc_auc_score
    from futures_foundation.finetune.classifier import get_classifier
    X, y = _toy(N=300, seed=2)
    clf = get_classifier('mantis', n_channels=4, ft_mode='partial', epochs=15,
                         batch=64, threads=2, device='cpu', verbose=False)
    clf.fit(X[:220], y[:220], X[220:], y[220:], seed=0)
    auc = roc_auc_score(y[220:], clf.predict_proba(X[220:]))
    assert auc > 0.7                                # learns the planted signal
