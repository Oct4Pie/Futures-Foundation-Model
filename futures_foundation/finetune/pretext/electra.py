"""ELECTRA slot, REWRITTEN (2026-07-08) as the BREAK-HOLD discriminative pretext — teach the
foundation to tell a REAL break from a FAKEOUT, self-supervised and DIRECTLY.

The original electra (replaced-candle detection) asked "is this candle synthetic?" — an INDIRECT
proxy that HOPED fakeout-discrimination would emerge as a byproduct. This rewrite makes it the
OBJECTIVE. At each window's ANCHOR (the last bar — strictly causal), we detect a structural break
(close through the recent swing high/low) and label whether it HOLDS (price extends in the break
direction) or FAILS (price reverses back through the broken level, or stalls) over the next k bars.
That hold-vs-fail label is the atomic unit of a fakeout — computed self-supervised from raw OHLCV,
millions of examples across every instrument/TF, NO strategy outcome, NO leak: the k future bars
that define the label are RESERVED and never shown to the encoder.

Discriminative like electra (the encoder is the keeper; a binary head reads it out), but NO
generator — the "fake" is a REAL failed break in the data, not a synthesized candle. So there is no
GAN instability, no OHLC-clamp cheat, no generator-strength knob. The one piece of electra-v2 that
worked MECHANICALLY — the encoder-side RECON anchor that held emb_std ~1 — is KEPT:
    loss = recon_weight * enc_recon(clean window) + hold_weight * BCE(hold | is_break).
Warm from the promoted base (ctr_seq2seq); ship gate unchanged (downstream WR@3R one-shot 2026).
The pretext balanced-accuracy is a LEARNING diagnostic only, never the verdict.

DIAGNOSTIC VALUE (why this is worth running even skeptically): if a DIRECT hold/fail objective
cannot beat ~0.5 balanced accuracy, that is the cleanest evidence yet that the fakeout discriminator
is NOT in OHLCV -> the answer is order flow, not another pretext. If it clears ~0.55, there is real
fakeout signal every prior (indirect) objective was leaving on the table.

torch-free: the break detection + hold/fail labeling math (unit-testable) lives here; the torch
trainer is in pretext/_torch/electra.py.
"""
import numpy as np

from .base import PretextTask

# OHLCV channel indices in the mv window layout [C, seq] (see strategy.mv_feature_names)
_O, _H, _L, _C = 0, 1, 2, 3


def detect_break(w, anchor, lookback):
    """Break state at bar `anchor` of a window w [C, seq]: does the anchor CLOSE break the swing
    extreme of the prior `lookback` bars? Returns (direction, level):
      +1, prior_high  -> upside break (close above the highest high of [anchor-lookback : anchor])
      -1, prior_low   -> downside break (close below the lowest low)
       0, nan         -> no break.
    Strictly CAUSAL — uses only bars strictly BEFORE the anchor. `level` is the broken swing extreme
    (the line a real break must stay beyond and a fakeout falls back through)."""
    lo = max(0, int(anchor) - int(lookback))
    if anchor <= lo:                                       # not enough prior bars to define a swing
        return 0, float('nan')
    prior_hi = float(np.max(w[_H, lo:anchor]))
    prior_lo = float(np.min(w[_L, lo:anchor]))
    c = float(w[_C, anchor])
    if c > prior_hi:
        return 1, prior_hi
    if c < prior_lo:
        return -1, prior_lo
    return 0, float('nan')


def label_hold(future, anchor_close, direction, level, atr, theta):
    """HOLD/FAIL for a break, from the k FUTURE bars `future` [C, k] AFTER the anchor. A barrier race:
    HOLD (1) if price EXTENDS by >= theta*atr beyond the anchor close in the break direction BEFORE it
    retraces back through the broken `level`; FAIL (0) otherwise — a retrace-first (bull/bear trap) OR
    a stall that never extends within k bars (the dead-bounce fakeout). direction 0 -> 0.

    Same-bar tie = FAIL: if a bar both extends AND falls back through the level, the invalidation
    wins (a wick through the level is not a hold). Causal by construction — `future` is disjoint from
    the encoder window."""
    if direction == 0 or future.shape[1] == 0 or not (atr > 0):
        return 0
    target = anchor_close + direction * theta * atr
    hi, lo = future[_H], future[_L]
    for j in range(future.shape[1]):
        if direction == 1:
            hit_fail = lo[j] < level                       # back below the broken swing high
            hit_target = hi[j] >= target and not hit_fail
        else:
            hit_fail = hi[j] > level                       # back above the broken swing low
            hit_target = lo[j] <= target and not hit_fail
        if hit_fail:
            return 0                                        # invalidation first (or same bar) = trap
        if hit_target:
            return 1                                        # clean extension = a real break
    return 0                                                # neither barrier in k bars = stall = fail


class BreakHoldTask(PretextTask):
    """The discriminative slot (registered under 'electra'), now BREAK-HOLD. Same representation gate
    as the original electra — the probe must show the encoder still encodes regime/vol/structure
    better than vanilla (mean_core_delta > margin) and hasn't collapsed — because break-hold is a
    REFINE of the promoted base and must not destroy the reconstruction lineage it warms from."""
    name, trainer = 'electra', 'train_ssl_electra'

    def reserve(self, cfg):
        # parent window = seq (encoder sees this; anchor = last bar) + hold_k FUTURE bars (label only,
        # never encoded). Break detection uses break_lookback bars that sit INSIDE the seq window.
        return int(cfg['seq']) + int(cfg.get('hold_k', 12))

    def _decide(self, probe_res, no_collapse, margin, dir_margin, detail):
        return bool(probe_res['mean_core_delta'] > margin and no_collapse), detail

    def finalize_verdict(self, verdict, fc_skill, probe_res):
        verdict['pretext_note'] = (
            'break-hold: judge downstream WR@3R + generality probes; hold_bal_acc is a LEARNING '
            'diagnostic (>~0.55 = fakeout signal exists in OHLCV; ~0.50 = it is order-flow, not price)')
        return verdict
