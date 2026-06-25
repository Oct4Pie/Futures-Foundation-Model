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


# ---------------------------------------------------------------------------
# from_pretrained — HF-style checkpoint loader (torch-free: resolves + configs)
# ---------------------------------------------------------------------------
import os as _os
import json as _json


@pytest.fixture(autouse=True)
def _iso_ckpt_env():
    keys = ('CHRONOS_POOL_LOCSCALE', 'CHRONOS_FT_CKPT', 'CHRONOS_RETURN_SHAPE')
    saved = {k: _os.environ.get(k) for k in keys}
    _os.environ['CHRONOS_RETURN_SHAPE'] = '0'   # isolate base dims; default-on tested in test_window_features
    yield
    for k, v in saved.items():
        _os.environ.pop(k, None) if v is None else _os.environ.__setitem__(k, v)


def test_from_pretrained_vanilla(monkeypatch):
    monkeypatch.delenv('CHRONOS_FT_CKPT', raising=False)
    ext = X.ChronosExtractor.from_pretrained(None)
    assert 'CHRONOS_FT_CKPT' not in _os.environ and ext.dim == 256
    for v in ('vanilla', 'frozen', 'base'):
        X.ChronosExtractor.from_pretrained(v)
        assert 'CHRONOS_FT_CKPT' not in _os.environ


def test_from_pretrained_name_resolves_local(monkeypatch, tmp_path):
    from futures_foundation.extractors.chronos import backbone as b
    monkeypatch.setattr(b, '_ROOT', tmp_path)
    ck = tmp_path / 'checkpoints' / 'my_ft'; ck.mkdir(parents=True)
    (ck / 'PROVENANCE.json').write_text(_json.dumps({'config': {'locscale': False}}))
    monkeypatch.delenv('CHRONOS_POOL_LOCSCALE', raising=False)
    ext = X.ChronosExtractor.from_pretrained('my_ft')
    assert _os.environ['CHRONOS_FT_CKPT'] == str(ck) and ext.dim == 256


def test_from_pretrained_locscale_ckpt_autoconfigs_258(monkeypatch, tmp_path):
    from futures_foundation.extractors.chronos import backbone as b
    monkeypatch.setattr(b, '_ROOT', tmp_path)
    ck = tmp_path / 'checkpoints' / 'ls_ft'; ck.mkdir(parents=True)
    (ck / 'PROVENANCE.json').write_text(_json.dumps({'config': {'locscale': True}}))
    monkeypatch.delenv('CHRONOS_POOL_LOCSCALE', raising=False)
    ext = X.ChronosExtractor.from_pretrained('ls_ft')
    assert _os.environ.get('CHRONOS_POOL_LOCSCALE') == '1' and ext.dim == 258
