"""ShapeAwareAdapter tests (UniShape port over frozen Chronos tokens).

torch-gated: the adapter is an nn.Module, so torch must be imported INSIDE each
test body and the suite gated behind CHRONOS_TORCH_TESTS=1 (libomp isolation —
see tests/conftest.py). Run: CHRONOS_TORCH_TESTS=1 pytest tests/test_shape_adapter.py
"""
import os

import numpy as np
import pytest

torch_test = pytest.mark.skipif(
    os.environ.get('CHRONOS_TORCH_TESTS') != '1',
    reason='torch test — set CHRONOS_TORCH_TESTS=1 (libomp isolation; import torch in body)')


# ---- shapes: forward + encode ---------------------------------------------
@torch_test
def test_forward_and_encode_shapes():
    import torch
    from futures_foundation.extractors.chronos.shape_adapter import ShapeAwareAdapter
    m = ShapeAwareAdapter(d=16, n_tokens=5, depth=1, heads=2, mlp=32, proto=True)
    x = torch.randn(8, 5, 16)
    logits, loss = m(x)                          # no labels -> no loss
    assert logits.shape == (8, 2) and loss is None
    clsf, lg = m.encode(x)
    assert clsf.shape == (8, 16) and lg.shape == (8, 2)


# ---- loss is finite scalar + grads flow (it trains) -----------------------
@torch_test
def test_loss_and_backward():
    import torch
    from futures_foundation.extractors.chronos.shape_adapter import ShapeAwareAdapter
    m = ShapeAwareAdapter(d=16, n_tokens=5, depth=1, heads=2, mlp=32, proto=True)
    x = torch.randn(8, 5, 16)
    y = torch.randint(0, 2, (8,))
    logits, loss = m(x, y)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    # the prototype centers receive gradient (shape-clustering is active)
    assert m.centers.grad is not None


# ---- prototype toggle off: CE-only path still valid -----------------------
@torch_test
def test_proto_off():
    import torch
    from futures_foundation.extractors.chronos.shape_adapter import ShapeAwareAdapter
    m = ShapeAwareAdapter(d=16, n_tokens=5, depth=1, heads=2, proto=False)
    assert not hasattr(m, 'centers')
    _, loss = m(torch.randn(4, 5, 16), torch.randint(0, 2, (4,)))
    assert torch.isfinite(loss)


# ---- works at the Bolt token shape (9 tokens, d=256) ----------------------
@torch_test
def test_bolt_token_shape():
    import torch
    from futures_foundation.extractors.chronos.shape_adapter import ShapeAwareAdapter
    m = ShapeAwareAdapter(d=256, n_tokens=9, depth=2, heads=4)
    logits, _ = m(torch.randn(4, 9, 256))
    assert logits.shape == (4, 2)


# ---- in_proj: ingest RAW per-bar feature sequences (handcraft over time) ---
@torch_test
def test_in_proj_raw_feature_sequence():
    import torch
    from futures_foundation.extractors.chronos.shape_adapter import (
        ShapeAwareAdapter, fit_and_infer)
    m = ShapeAwareAdapter(d=32, in_dim=10, n_tokens=20, depth=1, heads=2)
    logits, _ = m(torch.randn(6, 20, 10))        # raw [B, T, in_dim] -> projected
    assert logits.shape == (6, 2)
    # fit_and_infer with proj_dim learns a SPARSE feature/bar signal — but only
    # with the right config: prototype loss OFF (it caps sparse-signal learning)
    # + adequate proj/lr/epochs. (Diagnosed: proto-on/lr1e-3/20ep -> 0.57; this -> 0.9+.)
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    N, T, k = 600, 12, 10
    Y = rng.integers(0, 2, N)
    X = rng.standard_normal((N, T, k)).astype(np.float32)
    X[Y == 1, -1, 0] += 2.0                       # class-1 signal in ONE feature, last bar
    tr = np.zeros(N, bool); tr[:420] = True
    p, c, val_auc = fit_and_infer(X, Y, tr, epochs=80, device='cpu', proj_dim=32,
                                  depth=1, heads=2, proto=False, lr=2e-3)
    assert c.shape == (N, 32)
    assert 0.0 <= val_auc <= 1.0                  # early-stopping returns best val AUC
    assert roc_auc_score(Y[~tr], p[~tr]) > 0.8


# ---- axial adapter: cross-feature + temporal, detects a FEATURE LIFE-CYCLE ----
@torch_test
def test_axial_adapter_lifecycle():
    import torch
    from sklearn.metrics import roc_auc_score
    from futures_foundation.extractors.chronos.shape_adapter import (
        AxialShapeAdapter, fit_and_infer)
    m = AxialShapeAdapter(T=20, F=9, d=32, depth=1, heads=2)
    logits, _ = m(torch.randn(6, 20, 9))
    assert logits.shape == (6, 2)
    z, lg = m.encode(torch.randn(6, 20, 9))
    assert z.shape == (6, 64)                     # 2*d: time stream ++ feature stream
    # learns a FEATURE LIFE-CYCLE: feature 2 GROWS over the window -> class 1
    rng = np.random.default_rng(0)
    N, T, Fdim = 500, 20, 9
    Y = rng.integers(0, 2, N)
    X = rng.standard_normal((N, T, Fdim)).astype(np.float32)
    X[Y == 1, :, 2] += np.linspace(0, 2, T)[None, :]
    tr = np.zeros(N, bool); tr[:340] = True
    p, c, va = fit_and_infer(X, Y, tr, epochs=40, device='cpu', proj_dim=32,
                             arch='axial', depth=1, heads=2)
    assert c.shape == (N, 64)
    assert roc_auc_score(Y[~tr], p[~tr]) > 0.8   # sees the growing feature's life-cycle


# ---- pipeline: OOF-stacked adapter feature (leak-safe) + val/test monitor --
@torch_test
def test_oof_adapter_feature():
    from sklearn.metrics import roc_auc_score
    from futures_foundation.pipeline.shape_stack import (
        oof_adapter_feature, val_test_gap)
    rng = np.random.default_rng(0)
    N, T, Fdim = 500, 8, 6
    Y = rng.integers(0, 2, N)
    S = rng.standard_normal((N, T, Fdim)).astype(np.float32)
    S[Y == 1, -2:, :] += 1.5                          # easy: all features, last 2 bars
    tr = np.zeros(N, bool); tr[:340] = True
    cls, info = oof_adapter_feature(S, Y, tr, n_folds=2, proj_dim=16, epochs=40,
                                    device='cpu', patience=8)
    assert cls.shape == (N, 16)
    assert info['adapter_probs'].shape == (N,)
    assert 0.0 <= info['mean_val_auc'] <= 1.0
    # adapter generalizes on the held/test rows (OOF, leak-safe)
    assert roc_auc_score(Y[~tr], info['adapter_probs'][~tr]) > 0.65
    gap, overfit = val_test_gap(info['mean_val_auc'],
                                roc_auc_score(Y[~tr], info['adapter_probs'][~tr]))
    assert isinstance(bool(overfit), bool)


# ---- fit_and_infer LEARNS a separable shape (end-to-end) ------------------
@torch_test
def test_fit_and_infer_learns_separable_shape():
    from sklearn.metrics import roc_auc_score
    from futures_foundation.extractors.chronos.shape_adapter import fit_and_infer
    rng = np.random.default_rng(0)
    N, T, d = 600, 6, 16
    Y = rng.integers(0, 2, N)
    X = rng.standard_normal((N, T, d)).astype(np.float32)
    X[Y == 1, -1, :] += 1.5                      # class-1 'shape' = bump in last token
    tr = np.zeros(N, bool); tr[:400] = True
    probs, cls, val_auc = fit_and_infer(X, Y, tr, epochs=60, device='cpu',
                                        depth=1, heads=2)
    assert probs.shape == (N,) and cls.shape == (N, d)
    assert np.isfinite(probs).all()
    assert roc_auc_score(Y[~tr], probs[~tr]) > 0.7   # learned the shape, OOS
