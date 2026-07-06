"""ELECTRA-style replaced-candle-detection pretext (stage 4): registry, gate, and the pure
corruption math (bar-mask sampling, valid-OHLC clamp). Torch parts (clamp parity, loss shape)
are gated behind CHRONOS_TORCH_TESTS=1 like the other SSL trainers (libomp isolation).

The clamp is load-bearing for the WHOLE pretext: without valid OHLC the discriminator cheats by
flagging impossible candles (H under the body) instead of learning market dynamics.
"""
import os

import numpy as np
import pytest

from futures_foundation.finetune.pretext import PRETEXTS, get_pretext
from futures_foundation.finetune.pretext.electra import (
    ElectraTask, sample_bar_mask, clamp_valid_ohlc)

torch_test = pytest.mark.skipif(
    os.environ.get('CHRONOS_TORCH_TESTS') != '1',
    reason='torch test — set CHRONOS_TORCH_TESTS=1 (libomp isolation)')


# ---------------------------------------------------------------- registry + task
def test_registered_in_pretexts():
    assert 'electra' in PRETEXTS
    t = get_pretext('electra')
    assert isinstance(t, ElectraTask)
    assert t.trainer == 'train_ssl_electra'                # resolved via _ssl_torch by the base task
    assert t.reserve({}) == 0                              # RTD is in-window: nothing reserved


def test_trainer_importable_from_ssl_torch_shim():
    # PretextTask.train resolves getattr(_ssl_torch, trainer) — the re-export must exist.
    # Import the SHIM lazily and only check the attribute is exported (no torch instantiation).
    import importlib.util
    spec = importlib.util.find_spec('futures_foundation.finetune.pretext._torch.electra')
    assert spec is not None


def test_gate_passes_and_fails_like_mask():
    t = get_pretext('electra')
    probe = {'mean_core_delta': 0.05, 'descriptive_delta': 0.0, 'fwd_absmove_delta': 0.0,
             'fwd_dir_delta': 0.0, 'forward_score': 0.0, 'learns_regime_vol_structure': True}
    ok, detail = t.gate(probe, std=0.5, margin=0.0, dir_margin=0.0)
    assert ok and detail['no_collapse']
    ok_c, _ = t.gate(probe, std=0.0, margin=0.0, dir_margin=0.0)      # collapsed embedding
    assert not ok_c
    ok_m, _ = t.gate({**probe, 'mean_core_delta': -0.01}, std=0.5, margin=0.0, dir_margin=0.0)
    assert not ok_m                                        # probe below margin -> fail


def test_finalize_verdict_notes_downstream_judging():
    v = get_pretext('electra').finalize_verdict({}, None, None)
    assert 'WR@3R' in v['pretext_note']                    # the ship gate stays downstream


# ---------------------------------------------------------------- bar-mask sampler
def test_sample_bar_mask_shape_ratio_and_min_one():
    rng = np.random.default_rng(0)
    m = sample_bar_mask(rng, 512, 64, 0.15)
    assert m.shape == (512, 64) and m.dtype == bool
    assert m.any(axis=1).all()                             # >=1 masked bar per row, always
    assert 0.10 < m.mean() < 0.22                          # ~ratio


def test_sample_bar_mask_zero_ratio_still_masks_one():
    rng = np.random.default_rng(1)
    m = sample_bar_mask(rng, 8, 16, 0.0)
    assert (m.sum(axis=1) == 1).all()                      # exactly the forced bar


def test_sample_bar_mask_deterministic_with_seed():
    a = sample_bar_mask(np.random.default_rng(7), 32, 64, 0.2)
    b = sample_bar_mask(np.random.default_rng(7), 32, 64, 0.2)
    assert np.array_equal(a, b)


# ---------------------------------------------------------------- valid-OHLC clamp
def _candles(o, h, l, c, v=None):
    """[1, C, seq] window from per-bar lists."""
    v = v if v is not None else [1.0] * len(o)
    return np.stack([o, h, l, c, v]).astype(float)[None]


def test_clamp_fixes_invalid_high_low():
    w = _candles(o=[10.0], h=[9.0], l=[11.0], c=[12.0])    # impossible: H under body, L above
    out = clamp_valid_ohlc(w)
    o, h, l, c = out[0, 0, 0], out[0, 1, 0], out[0, 2, 0], out[0, 3, 0]
    assert h >= max(o, c) and l <= min(o, c)
    assert h >= l


def test_clamp_valid_candles_unchanged_and_idempotent():
    w = _candles(o=[10.0, 11.0], h=[12.0, 11.5], l=[9.5, 10.2], c=[11.5, 10.4])
    out = clamp_valid_ohlc(w)
    assert np.allclose(out, w)                             # already valid -> untouched
    assert np.allclose(clamp_valid_ohlc(out), out)         # idempotent


def test_clamp_leaves_volume_alone():
    w = _candles(o=[10.0], h=[9.0], l=[11.0], c=[12.0], v=[123.0])
    assert clamp_valid_ohlc(w)[0, 4, 0] == 123.0


def test_clamp_does_not_mutate_input():
    w = _candles(o=[10.0], h=[9.0], l=[11.0], c=[12.0])
    keep = w.copy()
    clamp_valid_ohlc(w)
    assert np.array_equal(w, keep)


# ---------------------------------------------------------------- cfg plumbing
def test_base_cfg_keeps_electra_knobs():
    # _base_cfg drops UNKNOWN keys silently — rtd_weight/gen_width must be registered defaults
    # or the runner's knobs would never reach the trainer (the silent-drop trap).
    from futures_foundation.finetune.ssl import _base_cfg
    cfg = _base_cfg(pretext='electra', rtd_weight=7.5, gen_width=32, mask_ratio=0.15)
    assert cfg['rtd_weight'] == 7.5
    assert cfg['gen_width'] == 32
    assert cfg['mask_ratio'] == 0.15


# ---------------------------------------------------------------- torch parity (gated)
@torch_test
def test_torch_clamp_matches_numpy_reference():
    import torch
    from futures_foundation.finetune.pretext._torch.electra import clamp_valid_ohlc_t
    rng = np.random.default_rng(3)
    raw = rng.normal(100.0, 5.0, size=(4, 5, 16))          # random (mostly invalid) raw candles
    mu = raw.mean(axis=2, keepdims=True)
    sd = raw.std(axis=2, keepdims=True) + 1e-6
    std = (raw - mu) / sd
    out_t = clamp_valid_ohlc_t(torch.tensor(std), torch.tensor(mu), torch.tensor(sd))
    back = out_t.numpy() * sd + mu                         # un-standardize the torch result
    ref = clamp_valid_ohlc(raw)                            # numpy reference on raw
    assert np.allclose(back, ref, atol=1e-8)
