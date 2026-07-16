"""ONNX export for the frozen-Mantis fractal — deployable bundle (encoder + head).

Torch-free: head (skl2onnx) parity. Torch-gated (CHRONOS_TORCH_TESTS=1, libomp isolation):
encoder parity vs embed_windows + the full _export_frozen_bundle two-file output.
"""
import os

import numpy as np
import pytest

torch_test = pytest.mark.skipif(
    os.environ.get('CHRONOS_TORCH_TESTS') != '1',
    reason='torch test — set CHRONOS_TORCH_TESTS=1 (libomp isolation)')


def _onnx_proba(path, X):
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    outs = sess.run(None, {'input': np.asarray(X, np.float32)})
    return [o for o in outs if getattr(o, 'ndim', 0) == 2 and o.shape[1] == 2][0]


@pytest.mark.parametrize('head', ['logistic', 'mlp'])
def test_head_onnx_parity(tmp_path, head):
    """skl2onnx head matches the sklearn head's predict_proba (torch-free)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from futures_foundation.finetune.classifiers.mantis.frozen import export_head_onnx
    rng = np.random.default_rng(0)
    X = rng.standard_normal((400, 40)).astype(np.float32)
    y = (rng.random(400) < 0.3).astype(int)
    clf = (MLPClassifier((32,), max_iter=200, random_state=0) if head == 'mlp'
           else LogisticRegression(max_iter=500))
    clf.fit(X, y)
    p = str(tmp_path / 'head.onnx')
    export_head_onnx(clf, X.shape[1], p)
    onnx_p1 = _onnx_proba(p, X)[:, 1]
    assert np.abs(clf.predict_proba(X)[:, 1] - onnx_p1).max() < 1e-4


@pytest.mark.parametrize('head', ['logistic', 'mlp'])
def test_head_onnx_baked_platt_parity(tmp_path, head):
    """_bake_platt_into_head: the exported 'probabilities' output IS the CALIBRATED proba
    (sigmoid(A*logit(p_raw)+B)) — the standard-proba-range fix (2026-07-16). Verifies exact
    parity vs apply_platt(sklearn proba), the output name/shape contract is unchanged, and
    the two columns still sum to 1."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from futures_foundation.finetune.calibration import apply_platt
    from futures_foundation.finetune.classifiers.mantis.frozen import (
        export_head_onnx, _bake_platt_into_head)
    rng = np.random.default_rng(1)
    X = rng.standard_normal((400, 40)).astype(np.float32)
    y = (rng.random(400) < 0.3).astype(int)
    clf = (MLPClassifier((32,), max_iter=200, random_state=0) if head == 'mlp'
           else LogisticRegression(max_iter=500))
    clf.fit(X, y)
    p = str(tmp_path / 'head.onnx')
    export_head_onnx(clf, X.shape[1], p)
    platt = (0.8180077893606709, -0.2410633800430193)  # the 2026 production bundle's (A, B)
    _bake_platt_into_head(p, platt)
    proba = _onnx_proba(p, X)                          # same output-finder as every consumer
    ref = apply_platt(clf.predict_proba(X)[:, 1], platt)
    assert np.abs(ref - proba[:, 1]).max() < 1e-4
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)   # [1-p, p] columns coherent
    assert proba.min() >= 0.0 and proba.max() <= 1.0


def test_baked_platt_identity_when_neutral(tmp_path):
    """A=1, B=0 is the identity transform — baked output must equal the raw head proba
    (guards the graph surgery itself, independent of any particular calibration values)."""
    from sklearn.linear_model import LogisticRegression
    from futures_foundation.finetune.classifiers.mantis.frozen import (
        export_head_onnx, _bake_platt_into_head)
    rng = np.random.default_rng(2)
    X = rng.standard_normal((300, 20)).astype(np.float32)
    y = (rng.random(300) < 0.4).astype(int)
    clf = LogisticRegression(max_iter=500).fit(X, y)
    p = str(tmp_path / 'head.onnx')
    export_head_onnx(clf, X.shape[1], p)
    _bake_platt_into_head(p, (1.0, 0.0))
    assert np.abs(clf.predict_proba(X)[:, 1] - _onnx_proba(p, X)[:, 1]).max() < 1e-4


@torch_test
def test_encoder_onnx_parity(tmp_path):
    """export_encoder_onnx reproduces embed_windows numerically (vanilla Mantis, no ckpt)."""
    from futures_foundation.finetune._ssl_torch import embed_windows, export_encoder_onnx
    rng = np.random.default_rng(0)
    W = (rng.standard_normal((6, 5, 64)).astype(np.float32) * 3 + 100)
    ref = embed_windows(W, ckpt=None, device='cpu')
    p = export_encoder_onnx(str(tmp_path / 'enc.onnx'), ckpt=None, C=5, seq=64, device='cpu')
    import onnxruntime as ort
    sess = ort.InferenceSession(p, providers=['CPUExecutionProvider'])
    onnx_emb = sess.run(['embedding'], {'window': W})[0]
    assert onnx_emb.shape == ref.shape
    assert np.abs(ref - onnx_emb).max() < 1e-3


@torch_test
def test_export_frozen_bundle_writes_both(tmp_path):
    """_export_frozen_bundle writes <base>_encoder.onnx + <base>_signal_head.onnx, head parity."""
    from sklearn.linear_model import LogisticRegression
    from futures_foundation.finetune.classifiers.mantis.frozen import _export_frozen_bundle
    D = 5 * 256 + 10
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, D)).astype(np.float32)
    y = (rng.random(300) < 0.3).astype(int)
    clf = LogisticRegression(max_iter=500).fit(X, y)
    cfg = dict(export_onnx_path=str(tmp_path / 'fractal.onnx'), backbone_ckpt=None,
               raw_C=5, raw_seq=64)
    enc, head = _export_frozen_bundle(cfg, clf, D, X)
    assert os.path.exists(enc) and os.path.exists(head)
    assert np.abs(clf.predict_proba(X)[:, 1] - _onnx_proba(head, X)[:, 1]).max() < 1e-4
