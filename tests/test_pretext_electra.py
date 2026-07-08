"""BREAK-HOLD discriminative pretext (stage 4, the rewritten electra slot): registry, gate, cfg
plumbing, and the pure break-detection + hold/fail labeling math (unit-testable, torch-free). The
torch parity (vectorized labeler matches the numpy reference, encoder-recon anchor gradient) is
gated behind CHRONOS_TORCH_TESTS=1 like the other SSL trainers (libomp isolation).

The MOTIVATING SCENARIO (see test_strong_downtrend_fake_pullback_is_fail): a strong downtrend with a
relief pullback that pokes above a recent swing high then rolls over is the LIVE loser — the label
MUST call that break a FAIL so the downstream head learns to ignore pivot longs there.
"""
import os

import numpy as np
import pytest

from futures_foundation.finetune.pretext import PRETEXTS, get_pretext
from futures_foundation.finetune.pretext.electra import (
    BreakHoldTask, detect_break, label_hold)

torch_test = pytest.mark.skipif(
    os.environ.get('CHRONOS_TORCH_TESTS') != '1',
    reason='torch test — set CHRONOS_TORCH_TESTS=1 (libomp isolation)')

_O, _H, _L, _C, _V = 0, 1, 2, 3, 4


def _win(o, h, l, c, v=None):
    """[C, seq] window from per-bar lists."""
    v = v if v is not None else [1.0] * len(o)
    return np.stack([o, h, l, c, v]).astype(float)


# ---------------------------------------------------------------- registry + task
def test_registered_in_pretexts():
    assert 'electra' in PRETEXTS                            # the discriminative slot, now break-hold
    t = get_pretext('electra')
    assert isinstance(t, BreakHoldTask)
    assert t.trainer == 'train_ssl_electra'                # resolved via _ssl_torch by the base task
    assert t.reserve({'seq': 64}) == 64 + 12               # seq + hold_k future bars (label only)
    assert t.reserve({'seq': 64, 'hold_k': 20}) == 84


def test_trainer_importable_from_ssl_torch_shim():
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


# ---------------------------------------------------------------- break detection
def test_detect_upside_break():
    # prior 20 bars top out at high=10; the anchor closes at 11 -> upside break of the swing high
    o = [9.0] * 20 + [10.0]
    h = [10.0] * 20 + [11.2]
    l = [8.0] * 20 + [9.5]
    c = [9.5] * 20 + [11.0]
    d, level = detect_break(_win(o, h, l, c), anchor=20, lookback=20)
    assert d == 1 and level == 10.0


def test_detect_downside_break():
    o = [9.0] * 20 + [8.0]
    h = [10.0] * 20 + [8.5]
    l = [8.0] * 20 + [6.8]
    c = [9.5] * 20 + [7.0]
    d, level = detect_break(_win(o, h, l, c), anchor=20, lookback=20)
    assert d == -1 and level == 8.0                        # broke the swing low


def test_detect_no_break_inside_range():
    o = [9.0] * 20 + [9.4]
    h = [10.0] * 20 + [9.8]
    l = [8.0] * 20 + [9.1]
    c = [9.5] * 20 + [9.5]                                 # closes inside [8,10] -> no break
    d, level = detect_break(_win(o, h, l, c), anchor=20, lookback=20)
    assert d == 0 and np.isnan(level)


# ---------------------------------------------------------------- hold / fail labeling
def test_hold_when_break_extends():
    # up-break of level=10, anchor close 11, atr=1: target = 11 + 1*1 = 12. Future extends to 12.5
    # without ever dipping below 10 -> HOLD.
    fut = _win(o=[11.5, 12.2], h=[11.8, 12.6], l=[11.1, 12.0], c=[11.6, 12.5])
    assert label_hold(fut, anchor_close=11.0, direction=1, level=10.0, atr=1.0, theta=1.0) == 1


def test_fail_when_break_retraces_through_level():
    # up-break of level=10; price falls back BELOW 10 before reaching target -> FAIL (bull trap)
    fut = _win(o=[10.5, 9.5], h=[10.8, 9.8], l=[10.1, 9.2], c=[10.4, 9.4])
    assert label_hold(fut, anchor_close=11.0, direction=1, level=10.0, atr=1.0, theta=1.0) == 0


def test_fail_when_break_stalls():
    # up-break but price just chops between the level and the target for all k bars -> stall = FAIL
    fut = _win(o=[11.1, 11.2, 11.0], h=[11.4, 11.5, 11.3], l=[10.6, 10.7, 10.5],
               c=[11.2, 11.1, 11.0])
    assert label_hold(fut, anchor_close=11.0, direction=1, level=10.0, atr=1.0, theta=1.0) == 0


def test_same_bar_tie_is_fail():
    # a bar that both pokes the target AND wicks back through the level = not a hold (invalidation wins)
    fut = _win(o=[11.5], h=[12.5], l=[9.5], c=[10.0])       # hi>=12 target AND lo<10 level, same bar
    assert label_hold(fut, anchor_close=11.0, direction=1, level=10.0, atr=1.0, theta=1.0) == 0


def test_no_break_is_zero():
    fut = _win(o=[9.5], h=[9.8], l=[9.2], c=[9.5])
    assert label_hold(fut, anchor_close=9.5, direction=0, level=float('nan'), atr=1.0, theta=1.0) == 0


def test_strong_downtrend_fake_pullback_is_fail():
    """THE MOTIVATING CASE (user, 2026-07-08): a strong downtrend, a relief pullback pokes ABOVE the
    recent swing high (a long-side break — exactly where a counter-trend pivot long would fire), then
    the downtrend resumes. The label MUST be FAIL so the downstream head learns to IGNORE those longs.

    Construct 20 falling bars, then an anchor bar whose close breaks just above the local swing high,
    then future bars that roll back below the broken level (the trend reasserts)."""
    # falling swing: highs drift down from 30 to 12; the last few bars' swing high ~ 12.5
    highs = list(np.linspace(30, 12.5, 20))
    lows = [h - 2 for h in highs]
    closes = [h - 1 for h in highs]
    opens = [h - 0.5 for h in highs]
    swing_hi = max(highs[-20:])
    # anchor: a fake-pullback bar that closes just ABOVE the swing high (the bull-trap poke)
    opens.append(12.0); highs.append(swing_hi + 1.0); lows.append(11.8); closes.append(swing_hi + 0.5)
    w = _win(opens, highs, lows, closes)
    d, level = detect_break(w, anchor=20, lookback=20)
    assert d == 1                                          # it IS an upside break (the trap trigger)
    # future: downtrend resumes — price falls back through the broken swing high almost immediately
    fut = _win(o=[swing_hi - 0.2, swing_hi - 2],
               h=[swing_hi + 0.1, swing_hi - 1],
               l=[swing_hi - 1.5, swing_hi - 3],
               c=[swing_hi - 1.0, swing_hi - 2.5])
    atr = float((w[_H] - w[_L]).mean())
    assert label_hold(fut, anchor_close=closes[-1], direction=d, level=level, atr=atr, theta=1.0) == 0


def test_real_breakout_holds():
    """Contrast: a genuine breakout that keeps going must be HOLD — the label isn't just 'always fail
    counter-trend', it separates the two. Rising bars, anchor breaks the swing high, price extends."""
    highs = list(np.linspace(10, 20, 20))
    lows = [h - 2 for h in highs]
    closes = [h - 0.5 for h in highs]
    opens = [h - 1 for h in highs]
    swing_hi = max(highs[-20:])
    opens.append(swing_hi - 0.5); highs.append(swing_hi + 1.5)
    lows.append(swing_hi - 1.0); closes.append(swing_hi + 1.0)
    w = _win(opens, highs, lows, closes)
    d, level = detect_break(w, anchor=20, lookback=20)
    assert d == 1
    atr = float((w[_H] - w[_L]).mean())
    target = closes[-1] + atr                              # theta=1
    fut = _win(o=[target + 0.2, target + 1], h=[target + 0.5, target + 2],
               l=[closes[-1] + 0.1, target], c=[target + 0.3, target + 1.5])
    assert label_hold(fut, anchor_close=closes[-1], direction=d, level=level, atr=atr, theta=1.0) == 1


# ---------------------------------------------------------------- cfg plumbing
def test_base_cfg_keeps_break_hold_knobs():
    # _base_cfg drops UNKNOWN keys silently — the break-hold knobs must be registered defaults or the
    # runner's knobs would never reach the trainer (the silent-drop trap).
    from futures_foundation.finetune.ssl import _base_cfg
    cfg = _base_cfg(pretext='electra', hold_k=20, break_lookback=30, hold_theta=1.5, hold_weight=3.0)
    assert cfg['hold_k'] == 20 and cfg['break_lookback'] == 30
    assert cfg['hold_theta'] == 1.5 and cfg['hold_weight'] == 3.0


def test_base_cfg_keeps_recon_weight():
    # the encoder-side anchor knob MUST thread through the silent-drop filter (default 1.0), else the
    # runner's RECON_WEIGHT never reaches the trainer and pure discrimination drifts the encoder.
    from futures_foundation.finetune.ssl import _base_cfg
    assert _base_cfg(pretext='electra')['recon_weight'] == 1.0        # default = anchored
    assert _base_cfg(pretext='electra', recon_weight=0.0)['recon_weight'] == 0.0   # pure-discrim ablation


# ---------------------------------------------------------------- torch parity (gated)
@torch_test
def test_vectorized_labeler_matches_numpy_reference():
    # the trainer's vectorized break_hold_labels must equal the per-window numpy detect_break+label_hold
    import torch
    from futures_foundation.finetune.pretext._torch.electra import break_hold_labels
    rng = np.random.default_rng(3)
    seq, k, B, lookback, theta = 64, 12, 32, 20, 1.0
    base = np.cumsum(rng.normal(0, 1, size=(B, seq + k)), axis=1) + 100
    O = base + rng.normal(0, 0.1, (B, seq + k))
    Cl = base + rng.normal(0, 0.1, (B, seq + k))
    H = np.maximum(O, Cl) + np.abs(rng.normal(0, 0.5, (B, seq + k)))
    L = np.minimum(O, Cl) - np.abs(rng.normal(0, 0.5, (B, seq + k)))
    V = np.abs(rng.normal(1000, 100, (B, seq + k)))
    full = np.stack([O, H, L, Cl, V], axis=1).astype(np.float32)      # [B, C, seq+k]
    w, fut = full[:, :, :seq], full[:, :, seq:]
    lab_t, brk_t = break_hold_labels(torch.tensor(w), torch.tensor(fut), lookback, theta)
    for b in range(B):
        d, level = detect_break(w[b], anchor=seq - 1, lookback=lookback)
        atr = float((w[b, _H] - w[b, _L]).mean())
        ref = label_hold(fut[b], float(w[b, _C, seq - 1]), d, level, atr, theta) if d != 0 else 0
        assert bool(brk_t[b]) == (d != 0), f'break mismatch at {b}'
        if d != 0:
            assert int(lab_t[b]) == ref, f'hold mismatch at {b}: {int(lab_t[b])} vs {ref}'


@torch_test
def test_encoder_recon_head_shapes_and_gradient():
    # the break-hold head + encoder-recon anchor: shapes, and the anchor DOES gradient the encoder
    # (the piece that keeps emb_std ~1 while it learns to discriminate).
    import torch
    from futures_foundation.finetune.pretext._torch.electra import BreakHoldNetwork
    net = BreakHoldNetwork(C=5, new_channels=3, seq=64)
    x = torch.randn(4, 5, 64)
    hold, rec = net.heads(x)
    assert hold.shape == (4,)                              # per-window hold/fail logit
    assert rec.shape == (4, 5, 64)                         # reconstructed [C, seq] window
    assert torch.isfinite(hold).all() and torch.isfinite(rec).all()
    net.zero_grad()
    torch.nn.functional.mse_loss(rec, x).backward()        # enc_recon anchor only
    enc_grad = sum(float(p.grad.abs().sum()) for p in net.encoder.parameters() if p.grad is not None)
    adapt_grad = sum(float(p.grad.abs().sum()) for p in net.adapter.parameters() if p.grad is not None)
    assert enc_grad > 0 and adapt_grad > 0                 # encoder IS anchored by the recon head
