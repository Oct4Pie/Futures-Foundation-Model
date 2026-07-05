"""Stage-3 pretext: TEMPORAL-NEIGHBORHOOD CONTRASTIVE (regime geometry, label-free).

Positives = temporally-nearby windows at multiple scales + augmented views; negatives =
far-in-time windows; per-anchor volatility DOWN-weighting (data-driven, not a label). Teaches
the encoder a smooth "market state geometry": nearby-in-time / structurally-similar windows
cluster, different structures separate — the regime representation the FFM vision wants the
foundation to own.

Replaces the outcome-keyed contrastive (v1-v3, dropped 2026-07-02 — ~90 trials, no arm beat
stage-2): the key here is TIME PROXIMITY, never a future path statistic — a fundamentally
different supervision source.

GATE (this stage) = the requirement doc's structural metrics A-E on the embedding space
(temporal consistency, emergent clusters, multi-scale ordering, noise robustness, temporal
stability — `regime_gate` in the torch module). The SHIP gate is unchanged: stage-2 seq2seq
stays the shipped base; a stage-3 checkpoint must beat it on the one-shot 2026 WR@3R benchmark
before promotion (feedback_holdout_offlimits discipline)."""
from .base import PretextTask


class ContrastiveTask(PretextTask):
    name, trainer = 'contrastive', 'train_ssl_contrastive'

    def reserve(self, cfg):
        # positives live at anchor+delta: reserve the largest delta so every positive window fits
        return max(int(d) for d in cfg.get('pos_deltas', (2, 16, 64)))

    def _decide(self, probe_res, no_collapse, margin, dir_margin, detail):
        desc_ok = bool(probe_res.get('descriptive_delta', probe_res['mean_core_delta']) >= -1e-9)
        detail.update({'descriptive_ok': desc_ok})
        return bool(no_collapse and desc_ok), detail
