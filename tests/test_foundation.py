"""Tests — futures_foundation.foundation (the Chronos-Bolt seam).

Honors the OpenMP isolation conventions (see test_chronos_framework.py):
no torch/chronos import at module top; the real subprocess embed is
covered by the existing gated framework tests. Here we test the torch-free
parent surface: import contract, window construction, validation, shims.
"""
import subprocess
import sys

import numpy as np
import pytest

import futures_foundation as ffm
from futures_foundation.extractors.chronos import backbone as foundation


@pytest.fixture(autouse=True)
def _iso_return_shape():
    """Base-mechanic tests (pool/locscale/empty dims) run with return-shape OFF;
    the default-on (+7) contract is verified in test_window_features."""
    import os as _o
    saved = _o.environ.get('CHRONOS_RETURN_SHAPE')
    _o.environ['CHRONOS_RETURN_SHAPE'] = '0'
    yield
    (_o.environ.pop('CHRONOS_RETURN_SHAPE', None) if saved is None
     else _o.environ.__setitem__('CHRONOS_RETURN_SHAPE', saved))


# ---------------------------------------------------------------------------
# Import contract — the parent must stay torch-free (macOS libomp segfault)
# ---------------------------------------------------------------------------

def test_package_import_is_torch_free():
    """`import futures_foundation` must NOT pull torch/transformers into
    the process. Run in a fresh interpreter so this suite's other imports
    can't mask a leak."""
    code = (
        "import sys; import futures_foundation; "
        "import futures_foundation.pipeline; "
        "leaked = [m for m in ('torch', 'transformers', 'chronos') "
        "if m in sys.modules]; "
        "sys.exit(1 if leaked else 0)"
    )
    r = subprocess.run([sys.executable, '-c', code], capture_output=True,
                       text=True)
    assert r.returncode == 0, f"torch leaked into parent: {r.stderr[-500:]}"


def test_dir_names_and_unknown_raises():
    # All names are eager + torch-free since the FFM transformer was retired
    # (tag ffm-transformer-final) — no lazy torch-side names remain.
    assert 'run_labeling' in dir(ffm)         # finetune name (eager, torch-free)
    assert 'embed_bars' in dir(ffm)           # eager name visible
    with pytest.raises(AttributeError):
        ffm.definitely_not_a_real_name


def test_torch_free_constants():
    assert foundation.D_MODEL == 256
    assert foundation.CTX == 128
    assert foundation.MODEL == 'amazon/chronos-bolt-tiny'


def test_active_source_env_override(monkeypatch):
    monkeypatch.delenv('CHRONOS_FT_CKPT', raising=False)
    assert foundation.active_source() == foundation.MODEL
    monkeypatch.setenv('CHRONOS_FT_CKPT', '/tmp/some_ckpt')
    assert foundation.active_source() == '/tmp/some_ckpt'


# ---------------------------------------------------------------------------
# embed_bars — causal window construction (embed() monkeypatched: the
# subprocess/torch path is covered by the gated framework tests)
# ---------------------------------------------------------------------------

def _capture_embed(monkeypatch):
    captured = {}

    def fake_embed(windows, batch=64, pool='mean', return_loc_scale=False):
        captured['windows'] = np.asarray(windows)
        captured['pool'] = pool
        captured['return_loc_scale'] = return_loc_scale
        dim = 2 * foundation.D_MODEL if pool == 'meanreg' else foundation.D_MODEL
        E = np.zeros((len(windows), dim), np.float32)
        if return_loc_scale:
            return E, np.zeros((len(windows), 2), np.float32)
        return E

    monkeypatch.setattr(foundation, 'embed', fake_embed)
    return captured


def test_embed_bars_windows_are_causal_log_close(monkeypatch):
    captured = _capture_embed(monkeypatch)
    close = np.arange(1, 301, dtype=np.float64)        # 300 bars
    idx = [127, 200, 299]
    out = foundation.embed_bars(close, idx, ctx=128)
    assert out.shape == (3, foundation.D_MODEL)
    W = captured['windows']
    assert W.shape == (3, 128)
    lp = np.log(close)
    for row, i in zip(W, idx):
        np.testing.assert_allclose(row, lp[i - 127:i + 1], rtol=1e-6)
    # last element of each window IS the decision bar (bars <= t, causal)
    np.testing.assert_allclose(W[:, -1], lp[idx], rtol=1e-6)


def test_embed_bars_rejects_short_history_and_oob(monkeypatch):
    _capture_embed(monkeypatch)
    close = np.arange(1, 301, dtype=np.float64)
    with pytest.raises(ValueError):
        foundation.embed_bars(close, [126], ctx=128)    # not enough history
    with pytest.raises(ValueError):
        foundation.embed_bars(close, [300], ctx=128)    # past end


def test_embed_bars_empty_indices_short_circuits():
    # no monkeypatch needed — must not touch the subprocess at all
    out = foundation.embed_bars(np.arange(1, 301, dtype=float), [])
    assert out.shape == (0, foundation.D_MODEL)
    assert out.dtype == np.float32


def test_embed_empty_contexts_short_circuits():
    out = foundation.embed(np.zeros((0, 128), np.float32))
    assert out.shape == (0, foundation.D_MODEL)


# ---------------------------------------------------------------------------
# Tier-1 levers (torch-free surface): pooled_dim, empty-path shapes for each
# pool mode + loc_scale, and embed_bars forwarding. Real pooling correctness is
# in the gated framework tests (it needs the torch subprocess + model).
# ---------------------------------------------------------------------------

def test_pooled_dim_modes():
    d = foundation.D_MODEL
    assert foundation.pooled_dim('mean') == d
    assert foundation.pooled_dim('reg') == d
    assert foundation.pooled_dim('meanreg') == 2 * d


def test_embed_empty_pool_modes_short_circuit():
    d = foundation.D_MODEL
    assert foundation.embed(np.zeros((0, 128), np.float32), pool='reg').shape == (0, d)
    out = foundation.embed(np.zeros((0, 128), np.float32), pool='meanreg')
    assert out.shape == (0, 2 * d)


def test_embed_empty_return_loc_scale_short_circuit():
    E, ls = foundation.embed(np.zeros((0, 128), np.float32),
                             pool='meanreg', return_loc_scale=True)
    assert E.shape == (0, 2 * foundation.D_MODEL)
    assert ls.shape == (0, 2)


def test_embed_bars_forwards_tier1_kwargs(monkeypatch):
    captured = _capture_embed(monkeypatch)
    close = np.arange(1, 301, dtype=np.float64)
    out = foundation.embed_bars(close, [200], ctx=128, pool='meanreg',
                                return_loc_scale=True)
    assert captured['pool'] == 'meanreg'
    assert captured['return_loc_scale'] is True
    E, ls = out                                     # tuple when loc_scale requested
    assert E.shape == (1, 2 * foundation.D_MODEL) and ls.shape == (1, 2)


def test_embed_bars_empty_meanreg_loc_scale_shapes():
    E, ls = foundation.embed_bars(np.arange(1, 301, dtype=float), [],
                                  pool='meanreg', return_loc_scale=True)
    assert E.shape == (0, 2 * foundation.D_MODEL) and ls.shape == (0, 2)


def test_embed_rejects_bad_pool():
    # fail-fast in the torch-free parent, not a cryptic worker RuntimeError
    with pytest.raises(ValueError):
        foundation.embed(np.zeros((2, 128), np.float32), pool='bogus')
    with pytest.raises(ValueError):
        foundation.embed_bars(np.arange(1, 301, dtype=float), [200], pool='nope')


def test_stamp_active_source_prints_frozen_tag(monkeypatch, capsys):
    monkeypatch.delenv('CHRONOS_FT_CKPT', raising=False)
    src = foundation.stamp_active_source(context='unit test')
    out = capsys.readouterr().out
    assert src == foundation.MODEL
    assert 'FROZEN' in out and 'unit test' in out


def test_stamp_active_source_local_ckpt(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv('CHRONOS_FT_CKPT', str(tmp_path))
    src = foundation.stamp_active_source()
    out = capsys.readouterr().out
    assert src == str(tmp_path)
    assert 'FINE-TUNED' in out


# ---------------------------------------------------------------------------
# Checkpoint-resolution flow — HF-style name resolution + self-describing
# config (loc_scale) from PROVENANCE.json. All torch-free.
# ---------------------------------------------------------------------------
import os as _os
import json as _json


@pytest.fixture(autouse=True)
def _isolate_ckpt_env():
    """active_source()/_apply_ckpt_config() mutate os.environ directly (by
    design — to propagate loc_scale to the worker subprocess). Snapshot +
    restore the two keys around every test so the side-effect can't leak."""
    keys = ('CHRONOS_POOL_LOCSCALE', 'CHRONOS_FT_CKPT')
    saved = {k: _os.environ.get(k) for k in keys}
    yield
    for k, v in saved.items():
        if v is None:
            _os.environ.pop(k, None)
        else:
            _os.environ[k] = v


def test_resolve_ckpt_name_path_and_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(foundation, '_ROOT', tmp_path)
    ck = tmp_path / 'checkpoints' / 'my_ft'
    ck.mkdir(parents=True)
    assert foundation.resolve_ckpt('my_ft') == str(ck)        # bare name -> checkpoints/<name>
    assert foundation.resolve_ckpt(str(ck)) == str(ck)        # existing path -> as-is
    assert foundation.resolve_ckpt('nope') == 'nope'          # unknown -> passthrough (HF hub)
    assert foundation.resolve_ckpt('') is None                # empty -> None


def test_active_source_vanilla_aliases(monkeypatch):
    for v in ('vanilla', 'frozen', 'base', 'VANILLA'):
        monkeypatch.setenv('CHRONOS_FT_CKPT', v)
        assert foundation.active_source() == foundation.MODEL
    monkeypatch.delenv('CHRONOS_FT_CKPT', raising=False)
    assert foundation.active_source() == foundation.MODEL      # unset -> vanilla


def test_active_source_name_resolves(monkeypatch, tmp_path):
    monkeypatch.setattr(foundation, '_ROOT', tmp_path)
    ck = tmp_path / 'checkpoints' / 'chronos_bolt_ft'
    ck.mkdir(parents=True)
    (ck / 'PROVENANCE.json').write_text(_json.dumps({'config': {'locscale': False}}))
    monkeypatch.delenv('CHRONOS_POOL_LOCSCALE', raising=False)
    monkeypatch.setenv('CHRONOS_FT_CKPT', 'chronos_bolt_ft')
    assert foundation.active_source() == str(ck)
    assert 'CHRONOS_POOL_LOCSCALE' not in _os.environ          # locscale:false -> not set


def test_apply_ckpt_config_autoenables_locscale(monkeypatch, tmp_path):
    monkeypatch.setattr(foundation, '_ROOT', tmp_path)
    ck = tmp_path / 'checkpoints' / 'ls_ft'
    ck.mkdir(parents=True)
    (ck / 'PROVENANCE.json').write_text(_json.dumps({'config': {'locscale': True, 'ctx': 128}}))
    monkeypatch.delenv('CHRONOS_POOL_LOCSCALE', raising=False)
    monkeypatch.setenv('CHRONOS_FT_CKPT', 'ls_ft')
    assert foundation.active_source() == str(ck)
    assert _os.environ.get('CHRONOS_POOL_LOCSCALE') == '1'     # auto-applied from PROVENANCE
    assert foundation.pooled_dim('mean') == foundation.D_MODEL + 2   # 258 follows


def test_explicit_locscale_env_wins(monkeypatch, tmp_path):
    monkeypatch.setattr(foundation, '_ROOT', tmp_path)
    ck = tmp_path / 'checkpoints' / 'base_ft'
    ck.mkdir(parents=True)
    (ck / 'PROVENANCE.json').write_text(_json.dumps({'config': {'locscale': False}}))
    monkeypatch.setenv('CHRONOS_POOL_LOCSCALE', '1')           # explicit override
    monkeypatch.setenv('CHRONOS_FT_CKPT', 'base_ft')
    foundation.active_source()
    assert _os.environ.get('CHRONOS_POOL_LOCSCALE') == '1'     # not clobbered by locscale:false


def test_pooled_dim_locscale_modes(monkeypatch):
    monkeypatch.delenv('CHRONOS_POOL_LOCSCALE', raising=False)
    assert foundation.pooled_dim('mean') == foundation.D_MODEL
    assert foundation.pooled_dim('meanreg') == 2 * foundation.D_MODEL
    monkeypatch.setenv('CHRONOS_POOL_LOCSCALE', '1')
    assert foundation.pooled_dim('mean') == foundation.D_MODEL + 2
    assert foundation.pooled_dim('meanreg') == 2 * foundation.D_MODEL + 2
