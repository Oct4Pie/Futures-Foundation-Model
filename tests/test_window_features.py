"""Unit tests for the Chronos extractor's window_features + CHRONOS_RETURN_SHAPE
fusion (pure numpy, no torch / no market data)."""
import numpy as np
import pytest

from futures_foundation.extractors.chronos import window_features as wf
from futures_foundation.extractors.chronos import backbone


def test_log_return_window_values_and_shape():
    W = np.array([[1.0, 2.0, 4.0, 7.0]], np.float32)      # diffs: 1,2,3
    R = wf.log_return_window(W)
    assert R.shape == W.shape
    assert R.dtype == np.float32
    np.testing.assert_allclose(R[0], [0.0, 1.0, 2.0, 3.0], rtol=1e-6)


def test_return_shape_features_shape_and_names():
    W = np.cumsum(np.random.default_rng(0).normal(0, 0.01, (5, 64)), axis=1).astype(np.float32)
    F = wf.return_shape_features(W)
    assert F.shape == (5, 7)
    assert F.dtype == np.float32
    assert len(wf.return_shape_feature_names()) == 7
    assert np.isfinite(F).all()


def test_return_shape_features_known_drift():
    # pure linear ramp -> constant returns of +1: mean=1, std=0, signrun=1
    W = np.arange(8, dtype=np.float32)[None, :]
    F = wf.return_shape_features(W)
    cols = wf.return_shape_feature_names()
    assert F[0, cols.index("ret_mean")] == pytest.approx(1.0, abs=1e-5)
    assert F[0, cols.index("ret_std")] == pytest.approx(0.0, abs=1e-5)
    assert F[0, cols.index("ret_signrun")] == pytest.approx(1.0, abs=1e-5)
    assert np.isfinite(F).all()                            # autocorr of constant -> finite (0)


def test_return_shape_features_constant_window_no_nan():
    W = np.full((3, 16), 5.0, np.float32)                  # zero returns everywhere
    F = wf.return_shape_features(W)
    assert np.isfinite(F).all()
    assert F.sum() == pytest.approx(0.0, abs=1e-5)         # all-zero, never NaN/inf


def test_return_shape_features_rejects_too_short():
    with pytest.raises(ValueError):
        wf.return_shape_features(np.zeros((2, 4), np.float32))   # need >=5 bars


def test_return_shape_features_deterministic():
    W = np.cumsum(np.random.default_rng(1).normal(0, 0.01, (4, 32)), axis=1).astype(np.float32)
    np.testing.assert_array_equal(wf.return_shape_features(W), wf.return_shape_features(W))


def test_subwindow_musig_values_and_width():
    W = np.concatenate([np.zeros((1, 16), np.float32),
                        np.full((1, 16), 2.0, np.float32)], axis=1)   # [1,32]
    M = wf.subwindow_musig(W, sub=16)
    assert M.shape == (1, 4)                               # 2 sub-windows x (mean,std)
    np.testing.assert_allclose(M[0], [0.0, 0.0, 2.0, 0.0], atol=1e-6)
    assert len(wf.subwindow_musig_names(32, 16)) == 4


def test_subwindow_musig_128_width_matches_names():
    W = np.random.default_rng(2).normal(0, 1, (2, 128)).astype(np.float32)
    M = wf.subwindow_musig(W, sub=16)
    assert M.shape[1] == len(wf.subwindow_musig_names(128, 16)) == 16
    assert np.isfinite(M).all()


# --- CHRONOS_RETURN_SHAPE fusion: BUILT IN (on by default), =0 disables ---

def test_return_shape_on_by_default(monkeypatch):
    monkeypatch.delenv('CHRONOS_RETURN_SHAPE', raising=False)
    assert backbone._return_shape_on() is True            # default = ON, not opt-in
    monkeypatch.setenv('CHRONOS_RETURN_SHAPE', '0')
    assert backbone._return_shape_on() is False           # only =0 disables


def test_pooled_dim_includes_return_shape_by_default(monkeypatch):
    monkeypatch.delenv('CHRONOS_POOL_LOCSCALE', raising=False)
    monkeypatch.setenv('CHRONOS_RETURN_SHAPE', '0')
    off = backbone.pooled_dim('mean')
    monkeypatch.delenv('CHRONOS_RETURN_SHAPE', raising=False)   # default on
    assert backbone.pooled_dim('mean') == off + wf.RETURN_SHAPE_DIM


def test_pooled_dim_stacks_locscale_and_return_shape(monkeypatch):
    monkeypatch.delenv('CHRONOS_POOL_LOCSCALE', raising=False)
    monkeypatch.setenv('CHRONOS_RETURN_SHAPE', '0')
    base = backbone.pooled_dim('mean')
    monkeypatch.setenv('CHRONOS_POOL_LOCSCALE', '1')
    monkeypatch.delenv('CHRONOS_RETURN_SHAPE', raising=False)   # default on
    assert backbone.pooled_dim('mean') == base + 2 + wf.RETURN_SHAPE_DIM


def test_maybe_return_shape_default_on_appends(monkeypatch):
    monkeypatch.delenv('CHRONOS_RETURN_SHAPE', raising=False)   # default on
    E = np.ones((4, 8), np.float32)
    ctx = np.cumsum(np.random.default_rng(0).normal(0, 0.01, (4, 64)), axis=1).astype(np.float32)
    out = backbone._maybe_return_shape(E, ctx)
    assert out.shape == (4, 8 + wf.RETURN_SHAPE_DIM)
    np.testing.assert_allclose(out[:, 8:], wf.return_shape_features(ctx), rtol=1e-5)


def test_maybe_return_shape_disabled_is_identity(monkeypatch):
    monkeypatch.setenv('CHRONOS_RETURN_SHAPE', '0')
    E = np.ones((4, 8), np.float32)
    ctx = np.cumsum(np.random.default_rng(0).normal(0, 0.01, (4, 64)), axis=1).astype(np.float32)
    out = backbone._maybe_return_shape(E, ctx)
    assert out.shape == (4, 8)
    np.testing.assert_array_equal(out, E)


def test_maybe_return_shape_empty_safe(monkeypatch):
    monkeypatch.delenv('CHRONOS_RETURN_SHAPE', raising=False)
    E = np.zeros((0, 8), np.float32)
    out = backbone._maybe_return_shape(E, np.zeros((0, 64), np.float32))
    assert out.shape == (0, 8)
