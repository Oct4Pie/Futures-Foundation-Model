"""Stage-3 pretext (EXPERIMENT): trend contrastive — multi-positive InfoNCE grouped by a
self-supervised causal trend key, to sharpen trend-vs-chop separation. Reserves the context.
Gate is report-only (descriptive content doesn't regress + no collapse); the REAL gate =
trend-AUC + WR@3R vs the ctx200 baseline, judged offline. Fallback = stage-2."""
from .base import PretextTask


class ContrastiveTask(PretextTask):
    name, trainer = 'contrastive', 'train_ssl_contrastive'

    def reserve(self, cfg):
        return max(int(x) for x in cfg['context_lengths'])

    def _decide(self, probe_res, no_collapse, margin, dir_margin, detail):
        desc_ok = bool(probe_res.get('descriptive_delta', probe_res['mean_core_delta']) >= -1e-9)
        detail.update({'descriptive_ok': desc_ok})
        return bool(no_collapse and desc_ok), detail
