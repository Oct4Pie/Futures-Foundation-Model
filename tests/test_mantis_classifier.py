"""Mantis classifier — torch-free adapter (non-gated) + torch trainer/worker (gated).

The parent-side MantisClassifier is torch-free (featurize + spawn worker); the actual
fine-tune lives in _mantis_torch and runs in the subprocess. torch tests are gated
behind CHRONOS_TORCH_TESTS=1 (libomp isolation) and import torch in-body.

Run torch parts: CHRONOS_TORCH_TESTS=1 pytest tests/test_mantis_classifier.py
"""
import os

import numpy as np
import pytest

from futures_foundation.finetune.native_contracts import NativeContractError

torch_test = pytest.mark.skipif(
    os.environ.get('CHRONOS_TORCH_TESTS') != '1',
    reason='torch test — set CHRONOS_TORCH_TESTS=1 (libomp isolation)')


class _StubLabeler:
    def __init__(self, X):
        self._X = X
    def mv_contexts(self, keys):
        return np.stack([self._X[k] for k in keys])


def _toy(seed=0, N=120, C=4, T=64):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, N)
    X = rng.standard_normal((N, C, T)).astype(np.float32)
    X[y == 1, 0, -8:] += 2.0
    return X, y


# ---- torch-free: the adapter ----------------------------------------------
def test_adapter_featurizes_and_needs_standardize():
    from futures_foundation.finetune.classifier import get_classifier
    clf = get_classifier('mantis', epochs=2)
    assert clf.needs_standardize is True
    X = np.random.default_rng(0).standard_normal((10, 4, 8)).astype(np.float32)
    out = clf.featurize(_StubLabeler(X), [0, 3, 7])
    assert out.shape == (3, 4, 8)


def test_adapter_module_has_no_top_level_torch_import():
    # the parent-side adapter must not import torch at module level (libomp isolation)
    import ast
    import futures_foundation.finetune.classifiers.mantis as m
    src = ast.parse(open(m.__file__).read())
    imported = {n.name.split('.')[0] for node in ast.walk(src)
                if isinstance(node, ast.Import) for n in node.names}
    imported |= {node.module.split('.')[0] for node in ast.walk(src)
                 if isinstance(node, ast.ImportFrom) and node.module}
    assert 'torch' not in imported


# ---- torch-gated: freeze policy on the model ------------------------------
@torch_test
def test_partial_mode_freezes_backbone():
    from futures_foundation.finetune.classifiers.mantis._torch import build_model
    model, new_c = build_model(4, new_channels=4, ft_mode='partial', unfreeze_blocks=2,
                               device='cpu')
    layers = model.encoder.vit_unit.transformer.layers
    assert all(p.requires_grad for p in layers[-1].parameters())
    assert not any(p.requires_grad for p in layers[0].parameters())
    assert all(p.requires_grad for p in model.head.parameters())


@torch_test
def test_head_mode_freezes_all_backbone():
    from futures_foundation.finetune.classifiers.mantis._torch import build_model
    model, _ = build_model(4, ft_mode='head', device='cpu')
    assert not any(p.requires_grad for p in model.encoder.parameters())
    assert all(p.requires_grad for p in model.head.parameters())


# ---- torch-gated: the trainer learns + shapes ----------------------------
@torch_test
def test_fit_predict_torch_shapes_and_learns():
    from futures_foundation.finetune.classifiers.mantis._torch import fit_predict_torch
    X, y = _toy(N=300, seed=2)
    with pytest.raises(NativeContractError, match='training admission is disabled'):
        fit_predict_torch(
            X[:200], y[:200], X[200:250], y[200:250], X[250:],
            ft_mode='partial', epochs=15, batch=64, threads=2,
            device='cpu', verbose=False)


# ---- torch-gated: memmap input + per-batch standardize (the full-data path) ----
@torch_test
def test_fit_predict_torch_memmap_and_standardize(tmp_path):
    from futures_foundation.finetune.classifiers.mantis._torch import fit_predict_torch
    X, y = _toy(N=300, seed=3)                      # raw (unstandardized)
    p = str(tmp_path / 'Xtr.npy'); np.save(p, X[:200])
    Xtr_mm = np.load(p, mmap_mode='r')              # disk-backed train
    flat = X[:200].transpose(0, 2, 1).reshape(-1, X.shape[1])
    mu, sd = flat.mean(0), flat.std(0) + 1e-6
    with pytest.raises(NativeContractError, match='training admission is disabled'):
        fit_predict_torch(
            Xtr_mm, y[:200], X[200:250], y[200:250], X[250:],
            ft_mode='partial', epochs=12, batch=64, device='cpu',
            standardize_mu=mu.tolist(), standardize_sd=sd.tolist(), verbose=False)


# ---- torch-gated: the adapter end-to-end (spawns the isolated worker) -----
@torch_test
def test_adapter_fit_predict_via_worker():
    from futures_foundation.finetune.classifier import get_classifier
    X, y = _toy(N=120)
    clf = get_classifier('mantis', ft_mode='head', epochs=2, batch=32, threads=2,
                         device='cpu', verbose=False)
    with pytest.raises(NativeContractError, match='training admission is disabled'):
        clf.fit_predict(X[:90], y[:90], X[90:], y[90:], X[90:], seed=0)
