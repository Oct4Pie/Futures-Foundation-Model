"""TURN-ELECTRA (2026-07-08): replaced-TURN detection — ELECTRA aimed at the swings.

THE PROBLEM (live-validated): the pivot strategy enters AT the swing bottom/top, and its losers are
FAKE turns — relief bounces in a grinding trend that look like reversals and die. The prior
discriminative attempts missed because their ANCHOR didn't match the entry: replaced-candle ELECTRA
corrupted uniform bars (generic dynamics, not turns); break-hold anchored on the BREAK bar (an event
a few bars AFTER the pivot entry — detection, where the pivot needs the turn itself).

THE FIX = keep the objective 100% self-supervised, change WHERE the corruption lands (the
salient-span-masking insight: don't change WHAT you predict, change WHERE): span-mask the region
AROUND DETECTED TURNS (local swing highs/lows — the same structural event the pivot trades), let the
deliberately-weak generator fill each masked turn with a PLAUSIBLE alternative development, and train
the encoder to label every bar real-vs-replaced. A generator-filled turn is functionally a SYNTHETIC
FAKE TURN — a bounce that continues where the real market reversed, a reversal where it rolled over.
To tell real from replaced AT THE SWINGS the encoder must learn how GENUINE turns develop (bar
sequencing, volume signature, follow-through formation) vs plausible imposters — the fakeout-vs-real
skill, learned with zero labels, no outcomes, fully generic (any reversal/breakout/regime consumer
reads it). Non-adversarial (generator trains on recon only), encoder-side recon anchor kept (the
piece that mechanically held emb_std ~1 across every discriminative run).

    loss = gen_recon(masked) + rtd_weight*BCE(real/replaced, all bars) + recon_weight*enc_recon(clean)

Warm from the promoted base (ctr_seq2seq). SHIP GATE unchanged: judged DOWNSTREAM — and per the
user's rule, on the metric specific to what this tests: fakeout discrimination among COUNTER-TREND /
turn pivots (the alignment table), not the aggregate (the aggregate is the forecasting objective's
home turf and says nothing about turn discrimination). rtd_bal_acc = learning diagnostic only.

This module is torch-free: the task + the pure corruption math (unit-testable). Turn detection +
the turn-biased span sampler live in pretext/spans.py; the torch trainer in _torch/electra.py.
"""
import numpy as np

from .base import PretextTask

# OHLCV channel indices in the mv window layout [B, C, seq] (see strategy.mv_feature_names)
_O, _H, _L, _C = 0, 1, 2, 3


def clamp_valid_ohlc(candles):
    """Force generated candles to be VALID OHLC in raw space: H >= max(O,C,H), L <= min(O,C,L).
    Without this the discriminator cheats by flagging impossible candles (H below the body) instead
    of learning turn dynamics. candles: [..., C, seq] with channels (O,H,L,C[,V...]); returns a
    corrected copy, non-OHLC channels (volume) untouched. Idempotent on already-valid candles."""
    out = np.array(candles, dtype=float, copy=True)
    body_hi = np.maximum(out[..., _O, :], out[..., _C, :])
    body_lo = np.minimum(out[..., _O, :], out[..., _C, :])
    out[..., _H, :] = np.maximum(out[..., _H, :], body_hi)
    out[..., _L, :] = np.minimum(out[..., _L, :], body_lo)
    return out


class TurnElectraTask(PretextTask):
    """The discriminative slot (registered under 'electra'), now TURN-ELECTRA. In-window corruption
    (nothing reserved) and the same representation gate as every refine: the probe must show the
    encoder still encodes regime/vol/structure better than vanilla (mean_core_delta > margin) and
    hasn't collapsed — a refine must not destroy the reconstruction lineage it warms from."""
    name, trainer = 'electra', 'train_ssl_electra'

    def _decide(self, probe_res, no_collapse, margin, dir_margin, detail):
        return bool(probe_res['mean_core_delta'] > margin and no_collapse), detail

    def finalize_verdict(self, verdict, fc_skill, probe_res):
        verdict['pretext_note'] = ('turn-electra (replaced-TURN detection): judge DOWNSTREAM on '
                                   'fakeout discrimination among counter-trend/turn pivots (the '
                                   'alignment table), NOT the aggregate; rtd_bal_acc + turn_cov are '
                                   'learning diagnostics only')
        return verdict
