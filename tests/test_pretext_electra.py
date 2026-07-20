"""TURN-ELECTRA pretext (stage 4, the discriminative slot): registry, gate, cfg plumbing, and the
pure corruption math — turn detection (local_turns), the TURN-BIASED span sampler (spans land on
swings), and the valid-OHLC clamp. Torch parts (network shapes, encoder-anchor gradient, clamp
parity) are gated behind CHRONOS_TORCH_TESTS=1 like the other SSL trainers (libomp isolation).

THE POINT UNDER TEST: the corruption must land ON THE TURNS (the swing event a pivot entry trades)
— that placement is the entire difference from the uniform span/candle ELECTRAs that came before.
If the mask isn't turn-focused (turn_cov low), the pretext degenerates to the old uniform variant.
"""
import os

import numpy as np
import pytest

from futures_foundation.finetune.native_contracts import NativeContractError
from futures_foundation.finetune.pretext import PRETEXTS, get_pretext
from futures_foundation.finetune.pretext.electra import TurnElectraTask, clamp_valid_ohlc
from futures_foundation.finetune.pretext.spans import local_turns, sample_turn_span_mask

torch_test = pytest.mark.skipif(
    os.environ.get('CHRONOS_TORCH_TESTS') != '1',
    reason='torch test — set CHRONOS_TORCH_TESTS=1 (libomp isolation)')


# ---------------------------------------------------------------- registry + task
def test_registered_in_pretexts():
    assert 'electra' in PRETEXTS                           # the discriminative slot, now turn-electra
    t = get_pretext('electra')
    assert isinstance(t, TurnElectraTask)
    assert t.trainer == 'train_ssl_electra'               # resolved via _ssl_torch by the base task
    assert t.reserve({'seq': 64}) == 0                    # in-window corruption: nothing reserved


def test_trainer_importable_from_ssl_torch_shim():
    import importlib.util
    spec = importlib.util.find_spec('futures_foundation.finetune.pretext._torch.electra')
    assert spec is not None


def test_gate_passes_and_fails_like_mask():
    t = get_pretext('electra')
    deltas = {'vol': 0.01, 'trend_eff': 0.03, 'range_expand': 0.02,
              'fwd_absmove': 0.01, 'direction': 0.0, 'fwd_dir': 0.001}
    per = {name: {'delta': delta, 'fold_delta': [delta] * 5}
           for name, delta in deltas.items()}
    probe = {'per_target': per, 'mean_core_delta': 0.02, 'descriptive_delta': 0.02,
             'fwd_absmove_delta': 0.01, 'fwd_dir_delta': 0.001,
             'forward_score': 0.011, 'learns_regime_vol_structure': True}
    ok, detail = t.gate(probe, std=0.5, margin=0.0, dir_margin=0.0)
    assert ok and detail['no_collapse']
    ok_c, _ = t.gate(probe, std=0.0, margin=0.0, dir_margin=0.0)      # collapsed embedding
    assert not ok_c
    regressed = {**probe, 'per_target': {**per, 'vol': {'delta': -0.01,
                                                         'fold_delta': [-0.01] * 5}}}
    ok_m, _ = t.gate(regressed, std=0.5, margin=0.0, dir_margin=0.0)
    assert not ok_m                                       # one target regression -> fail


def test_finalize_verdict_notes_the_specific_metric():
    v = get_pretext('electra').finalize_verdict({}, None, None)
    # the ONLY valid verdict for this pretext is fakeout discrimination on the counter-trend/turn
    # subset — the note must say so (aggregate is the forecasting objective's home turf).
    assert 'counter-trend' in v['pretext_note']
    assert 'aggregate' in v['pretext_note']


# ---------------------------------------------------------------- turn detection
def _tri(seq=64, peak=32):
    """Triangle window: strictly rising into `peak`, strictly falling after — ONE swing high at
    `peak` (plus trough turns at the edges by construction)."""
    t = np.arange(seq, dtype=float)
    h = -np.abs(t - peak)                                 # max (peak) exactly at `peak`
    l = h - 1.0
    return h, l


def test_local_turns_finds_the_swing_high():
    h, l = _tri(peak=32)
    turns = local_turns(h, l, w=3)
    assert 32 in turns                                    # THE swing high is detected
    assert 0 in turns and 63 in turns                     # edge troughs (lowest lows) detected
    assert 20 not in turns and 45 not in turns            # mid-leg bars are NOT turns


def test_local_turns_on_monotonic_ramp_only_edges():
    h = np.arange(64, dtype=float)
    l = h - 1.0
    turns = local_turns(h, l, w=3)
    assert set(turns) == {0, 63}                          # a pure trend has no interior turn


def test_local_turns_v_shape_finds_the_bottom():
    # V-shape = the pivot-entry event itself (a swing LOW): must be detected via the LOW channel
    t = np.arange(64, dtype=float)
    l = np.abs(t - 40)                                    # min (bottom) exactly at 40
    h = l + 1.0
    assert 40 in local_turns(h, l, w=3)


# ---------------------------------------------------------------- turn-biased span sampler
def test_turn_span_mask_shape_coverage_min_one():
    rng = np.random.default_rng(0)
    h, l = _tri()
    H, L = np.tile(h, (32, 1)), np.tile(l, (32, 1))
    m, cov = sample_turn_span_mask(rng, H, L, ratio=0.2, mean_span=4, max_span=10)
    assert m.shape == (32, 64) and m.dtype == bool
    assert m.any(axis=1).all()                            # >=1 masked bar per row, always
    assert 0.10 < m.mean() < 0.40                         # ~ratio (spans overshoot a little)
    assert 0.0 <= cov <= 1.0


def test_turn_span_mask_lands_on_turns_when_biased():
    # THE core property: with turn_bias=1 the masked bars concentrate AROUND the detected turns.
    rng = np.random.default_rng(1)
    h, l = _tri(peak=32)
    H, L = np.tile(h, (64, 1)), np.tile(l, (64, 1))
    m, cov = sample_turn_span_mask(rng, H, L, ratio=0.15, mean_span=4, max_span=8,
                                   turn_w=3, turn_bias=1.0)
    assert cov >= 0.6                                     # corruption is turn-focused
    # every span was CENTERED on a turn (0, 32, or 63) -> masked bars live within max_span of one
    turns = np.array([0, 32, 63])
    for b in range(m.shape[0]):
        for t in np.flatnonzero(m[b]):
            assert np.abs(turns - t).min() <= 8, f'masked bar {t} is far from every turn'


def test_turn_bias_zero_is_uniform_ablation():
    # turn_bias=0 = the uniform span-ELECTRA ablation: coverage of the (small) turn region should
    # be well BELOW the biased run's — the knob genuinely changes placement.
    h, l = _tri(peak=32)
    H, L = np.tile(h, (64, 1)), np.tile(l, (64, 1))
    _, cov_biased = sample_turn_span_mask(np.random.default_rng(2), H, L, 0.15, 4, 8,
                                          turn_w=3, turn_bias=1.0)
    _, cov_uniform = sample_turn_span_mask(np.random.default_rng(2), H, L, 0.15, 4, 8,
                                           turn_w=3, turn_bias=0.0)
    assert cov_biased > cov_uniform + 0.2


def test_turn_span_mask_deterministic_with_seed():
    h, l = _tri()
    H, L = np.tile(h, (16, 1)), np.tile(l, (16, 1))
    a, ca = sample_turn_span_mask(np.random.default_rng(7), H, L, 0.2, 4, 10)
    b, cb = sample_turn_span_mask(np.random.default_rng(7), H, L, 0.2, 4, 10)
    assert np.array_equal(a, b) and ca == cb


def test_turn_span_mask_flat_window_still_valid():
    # flat H/L = every bar ties as a "turn" — degenerate but must stay valid (mask non-empty, no crash)
    H = np.ones((8, 64)); L = np.zeros((8, 64))
    m, cov = sample_turn_span_mask(np.random.default_rng(3), H, L, 0.2, 4, 10)
    assert m.any(axis=1).all()
    assert cov == 1.0                                     # everything is "near a turn" in a flat window


# ---------------------------------------------------------------- valid-OHLC clamp
def _candles(o, h, l, c, v=None):
    """[1, C, seq] window from per-bar lists."""
    v = v if v is not None else [1.0] * len(o)
    return np.stack([o, h, l, c, v]).astype(float)[None]


def test_clamp_fixes_invalid_high_low():
    w = _candles(o=[10.0], h=[9.0], l=[11.0], c=[12.0])   # impossible: H under body, L above
    out = clamp_valid_ohlc(w)
    o, h, l, c = out[0, 0, 0], out[0, 1, 0], out[0, 2, 0], out[0, 3, 0]
    assert h >= max(o, c) and l <= min(o, c)
    assert h >= l


def test_clamp_valid_candles_unchanged_and_idempotent():
    w = _candles(o=[10.0, 11.0], h=[12.0, 11.5], l=[9.5, 10.2], c=[11.5, 10.4])
    out = clamp_valid_ohlc(w)
    assert np.allclose(out, w)                            # already valid -> untouched
    assert np.allclose(clamp_valid_ohlc(out), out)        # idempotent


def test_clamp_leaves_volume_alone_and_no_mutation():
    w = _candles(o=[10.0], h=[9.0], l=[11.0], c=[12.0], v=[123.0])
    keep = w.copy()
    out = clamp_valid_ohlc(w)
    assert out[0, 4, 0] == 123.0
    assert np.array_equal(w, keep)                        # input untouched


# ---------------------------------------------------------------- cfg plumbing
def test_base_cfg_keeps_turn_electra_knobs():
    # _base_cfg drops UNKNOWN keys silently — every runner knob must be a registered default or it
    # never reaches the trainer (the silent-drop trap).
    from futures_foundation.finetune.ssl import _base_cfg
    cfg = _base_cfg(pretext='electra', turn_w=5, turn_bias=0.5, rtd_weight=7.5, gen_width=32,
                    recon_weight=2.0, span_mean=4.0, span_max=8, mask_ratio=0.2)
    assert cfg['turn_w'] == 5 and cfg['turn_bias'] == 0.5
    assert cfg['rtd_weight'] == 7.5 and cfg['gen_width'] == 32 and cfg['recon_weight'] == 2.0
    assert cfg['span_mean'] == 4.0 and cfg['span_max'] == 8 and cfg['mask_ratio'] == 0.2


def test_base_cfg_defaults_are_turn_biased():
    from futures_foundation.finetune.ssl import _base_cfg
    cfg = _base_cfg(pretext='electra')
    assert cfg['turn_bias'] > 0.5                         # turn placement ON by default
    assert cfg['recon_weight'] == 1.0                     # anchored by default (drift guard)


def test_base_cfg_keeps_std_guard():
    # the IN-LOOP drift halt (added after emb_std marched 1.2->1.6 while val kept micro-improving
    # — early-stop alone kept crowning drifted epochs as "best"). Must survive the silent-drop
    # filter and default ON at 1.6.
    from futures_foundation.finetune.ssl import _base_cfg
    assert _base_cfg(pretext='electra')['std_guard'] == 1.6
    assert _base_cfg(pretext='electra', std_guard=1.4)['std_guard'] == 1.4
    assert _base_cfg(pretext='electra', std_guard=0)['std_guard'] == 0   # 0 = off


# ---------------------------------------------------------------- torch parity (gated)
@torch_test
def test_std_guard_halts_and_keeps_pre_breach_best():
    # BEHAVIORAL: a trainer whose emb_std drifts past the guard must HALT at the breach epoch and
    # NOT save that epoch as best — even though val keeps improving (the exact trap: early-stop
    # alone would have kept crowning drifted epochs).
    import torch
    import torch.nn as nn
    from futures_foundation.finetune.pretext._torch.common import BaseTrainer

    class _Drifty(BaseTrainer):
        """Minimal trainer: val improves EVERY epoch; std crosses the guard at epoch 3."""
        def build_net(self):
            class _N(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.encoder = nn.Linear(4, 4)
            self.net = _N()
            self._ep = 0

        def make_batch(self, starts):
            return None

        def compute_loss(self, batch):
            return (self.net.encoder.weight ** 2).mean()   # any differentiable scalar

        def val_eval(self):
            ep = self._ep; self._ep += 1
            return 10.0 - ep, {'std': 1.0 + 0.3 * ep}      # val always improves; std 1.0,1.3,1.6,1.9

    big = np.zeros((64, 4), np.float32)
    tr = va = np.arange(4, dtype=np.int64)
    t = _Drifty(big, tr, va, epochs=10, steps_per_epoch=1, batch=2, patience=8,
                device='cpu', verbose=False, std_guard=1.6)
    with pytest.raises(NativeContractError, match='training admission is disabled'):
        t.fit()


@torch_test
def test_network_heads_shapes_and_encoder_anchor_gradient():
    import torch
    from futures_foundation.finetune.pretext._torch.electra import ElectraNetwork
    net = ElectraNetwork(C=5, new_channels=3, seq=64, gen_width=16)
    x = torch.randn(4, 5, 64)
    rtd, rec = net.heads(x)
    assert rtd.shape == (4, 64)                           # per-bar real/replaced logits
    assert rec.shape == (4, 5, 64)                        # reconstructed [C, seq] window
    assert torch.isfinite(rtd).all() and torch.isfinite(rec).all()
    net.zero_grad()
    torch.nn.functional.mse_loss(rec, x).backward()       # enc_recon anchor only (no RTD)
    enc_grad = sum(float(p.grad.abs().sum()) for p in net.encoder.parameters() if p.grad is not None)
    recon_grad = sum(float(p.grad.abs().sum()) for p in net.recon.parameters() if p.grad is not None)
    assert not hasattr(net, 'adapter')
    assert enc_grad > 0 and recon_grad > 0                # encoder IS anchored by the recon head


@torch_test
def test_torch_clamp_matches_numpy_reference():
    import torch
    from futures_foundation.finetune.pretext._torch.electra import clamp_valid_ohlc_t
    rng = np.random.default_rng(3)
    raw = rng.normal(100.0, 5.0, size=(4, 5, 16))         # random (mostly invalid) raw candles
    mu = raw.mean(axis=2, keepdims=True)
    sd = raw.std(axis=2, keepdims=True) + 1e-6
    std = (raw - mu) / sd
    out_t = clamp_valid_ohlc_t(torch.tensor(std), torch.tensor(mu), torch.tensor(sd))
    back = out_t.numpy() * sd + mu                        # un-standardize the torch result
    ref = clamp_valid_ohlc(raw)                           # numpy reference on raw
    assert np.allclose(back, ref, atol=1e-8)
