"""Stage-2 pretext: TEMPORAL-NEIGHBORHOOD CONTRASTIVE (regime geometry, label-free).

The current objective uses elapsed-time neighborhoods, independently augmented observations,
equal anchor weighting, and synchronized-negative exclusion. The original bar-offset / volatility-
weighted objective remains available only as ``bar_offset_v1`` for controlled comparisons. Teaches
the encoder a smooth "market state geometry": nearby-in-time / structurally-similar windows
cluster, different structures separate — the regime representation the FFM vision wants the
foundation to own.

In the canonical V2 lineage it warm-starts from masked checkpoint 1 and emits checkpoint 2;
seq2seq then warm-starts from this encoder. The key is TIME PROXIMITY, never a future outcome."""
from .base import PretextTask


class ContrastiveTask(PretextTask):
    name, trainer = 'contrastive', 'train_ssl_contrastive'
    primary_targets = ('trend_eff', 'range_expand')

    def reserve(self, cfg):
        common = cfg.get('contrastive_reserve_contexts')
        if common is not None:
            # Comparison mode holds the eligible anchor universe fixed across objective versions.
            common_len = int(round(int(cfg['seq']) * float(common)))
            if cfg.get('contrastive_objective', 'elapsed_time_v2') == 'elapsed_time_v2':
                max_gap = max(float(x) for x in
                              cfg.get('positive_gap_fractions', (0.6, 1.0, 2.0)))
                natural = int(round(int(cfg['seq']) * (1.0 + max_gap)))
            else:
                natural = int(cfg['seq']) + max(
                    int(d) for d in cfg.get('pos_deltas', (2, 16, 64)))
            if common_len < natural:
                raise ValueError('contrastive_reserve_contexts is smaller than objective reach')
            return common_len
        if cfg.get('contrastive_objective', 'elapsed_time_v2') == 'elapsed_time_v2':
            # Context-relative elapsed-time gaps are equivalent to this many nominal context
            # lengths. Reserve the positive context as well so no snap can cross a temporal split.
            gap = max(float(x) for x in cfg.get('positive_gap_fractions', (0.6, 1.0, 2.0)))
            return int(round(int(cfg['seq']) * (1.0 + gap)))
        # Legacy bar-offset baseline.
        return int(cfg['seq']) + max(int(d) for d in cfg.get('pos_deltas', (2, 16, 64)))
