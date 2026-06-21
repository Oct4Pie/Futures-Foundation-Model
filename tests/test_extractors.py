"""Torch-free unit tests for the swappable feature-extractor interface."""
import numpy as np
import pytest

from futures_foundation import extractors as X


def test_registry_and_specs():
    ch = X.get_extractor('chronos')
    assert ch.name == 'chronos' and ch.ctx == 128 and ch.dim == 256


def test_chronos_meanreg_dim():
    ch = X.get_extractor('chronos', pool='meanreg')
    assert ch.dim == 512                       # 2 * D_MODEL


def test_protocol_conformance():
    assert isinstance(X.get_extractor('chronos'), X.FeatureExtractor)


def test_bad_name_rejected():
    with pytest.raises(ValueError):
        X.get_extractor('bogus')


def test_windows_full_no_pad():
    close = 100.0 + np.arange(300.0)
    w = X._windows(close, [299], ctx=128)
    assert w.shape == (1, 128)
    assert abs(float(w[0, -1]) - float(np.log(399.0))) < 1e-4
    assert abs(float(w[0, 0]) - float(np.log(272.0))) < 1e-4


def test_windows_left_pad_short():
    close = 100.0 + np.arange(50.0)            # only 50 bars, ctx 128
    w = X._windows(close, [49], ctx=128)
    assert w.shape == (1, 128)
    assert abs(float(w[0, 0]) - float(np.log(100.0))) < 1e-4
    assert abs(float(w[0, -1]) - float(np.log(149.0))) < 1e-4
    assert np.allclose(w[0, :78], np.float32(np.log(100.0)), atol=1e-4)


def test_embed_bars_empty_is_torch_free():
    # empty indices must short-circuit to zeros WITHOUT importing torch
    ch = X.get_extractor('chronos')
    assert ch.embed_bars([], []).shape == (0, 256)
