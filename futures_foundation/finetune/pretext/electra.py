"""ELECTRA-style pretext: REPLACED-CANDLE DETECTION (RTD) — discriminative SSL refine.

A small, deliberately-weak GENERATOR fills masked bars with plausible fake candles; the Mantis
encoder (the DISCRIMINATOR — the foundation we keep) labels EVERY bar real-vs-replaced. Because
every bar carries a training signal (not just the ~15-20% masked ones), it's far more
sample-efficient than generative masking — the ELECTRA insight, and the highest-leverage axis for
our data-limited financial regime. To spot a SUBTLY wrong candle the encoder must model normal
dynamics (momentum/vol/structure in context) — that internal model IS the representation the
downstream heads (trend classification, forecasting, regime) read. The generator is thrown away.

Non-adversarial (no GAN instability): the generator trains on reconstruction only; the encoder
trains on BCE over all bars. loss = recon(gen, masked) + rtd_weight * BCE(disc, all bars).
Warm-start = the promoted base (ctr_seq2seq) so the reconstruction lineage is kept and RTD adds
NEW discriminative signal on top. SHIP GATE unchanged: downstream WR@3R one-shot 2026 vs the
stage-2 bar + generality probes (forecast skill, regime separation) — never the pretext loss.

This module is torch-free: the PretextTask + the pure corruption math (unit-testable). The torch
trainer lives in pretext/_torch/electra.py.
"""
import numpy as np

from .base import PretextTask

# OHLCV channel indices in the mv window layout [B, C, seq] (see strategy.mv_feature_names)
_O, _H, _L, _C = 0, 1, 2, 3


def sample_bar_mask(rng, batch, seq, ratio):
    """Per-sample bar mask [batch, seq] bool with >=1 masked bar per row (a row with nothing to
    discriminate/reconstruct is a wasted sample). rng = np.random.Generator (deterministic)."""
    m = rng.random((batch, seq)) < ratio
    none = ~m.any(axis=1)
    m[none, 0] = True
    return m


def clamp_valid_ohlc(candles):
    """Force generated candles to be VALID OHLC in raw space: H >= max(O,C,H), L <= min(O,C,L).
    Without this the discriminator cheats by flagging impossible candles (H below the body) instead
    of learning market dynamics. candles: [..., C, seq] with channels (O,H,L,C[,V...]); returns a
    corrected copy, non-OHLC channels (volume) untouched. Idempotent on already-valid candles."""
    out = np.array(candles, dtype=float, copy=True)
    body_hi = np.maximum(out[..., _O, :], out[..., _C, :])
    body_lo = np.minimum(out[..., _O, :], out[..., _C, :])
    out[..., _H, :] = np.maximum(out[..., _H, :], body_hi)
    out[..., _L, :] = np.minimum(out[..., _L, :], body_lo)
    return out


class ElectraTask(PretextTask):
    """Stage-4 refine on the promoted base. Same reserve as mask (none — RTD is in-window) and the
    same representation gate: the probe must show the encoder still encodes regime/vol/structure
    better than vanilla (mean_core_delta > margin) and hasn't collapsed."""
    name, trainer = 'electra', 'train_ssl_electra'

    def _decide(self, probe_res, no_collapse, margin, dir_margin, detail):
        return bool(probe_res['mean_core_delta'] > margin and no_collapse), detail

    def finalize_verdict(self, verdict, fc_skill, probe_res):
        verdict['pretext_note'] = ('electra RTD: judge on downstream WR@3R + generality probes; '
                                   'rtd_acc is a task-calibration diagnostic only')
        return verdict
