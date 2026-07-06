"""Pretext-task REGISTRY — pluggable SSL pretraining objectives (stages).

Add a new pretrain experiment by dropping a module here (subclass PretextTask) and registering
its instance in PRETEXTS below; the orchestrator (ssl.py) stays untouched. Keeping each task in
its own file keeps ssl.py a clean orchestrator.

  mask          (stage 1)   — BERT-style masked modeling
  forecast      (stage 2)   — multi-horizon / variable-context candle seq2seq
  forecast_dist (stage 2.5) — DISTRIBUTIONAL forecast refine ON stage-2 (Chronos-style
                              quantile/bin objectives; own modules, stage-2 untouched)
  contrastive   (stage 3)   — TEMPORAL-NEIGHBORHOOD contrastive: regime geometry from
                              multi-scale time proximity + augmentations, sigma-weighted
                              (replaced the outcome-keyed v1-v3, dropped 2026-07-02)
  electra       (stage 4)   — ELECTRA-style REPLACED-CANDLE DETECTION (RTD): a weak generator
                              plants plausible fakes, the encoder labels EVERY bar real/fake —
                              discriminative every-bar signal = sample-efficiency for our
                              data-limited regime; warm from the promoted base
"""
from .base import PretextTask
from .mask import MaskTask
from .forecast import ForecastTask
from .forecast_dist import ForecastDistTask
from .contrastive import ContrastiveTask
from .electra import ElectraTask

PRETEXTS = {t.name: t for t in (MaskTask(), ForecastTask(), ForecastDistTask(),
                                ContrastiveTask(), ElectraTask())}


def get_pretext(name):
    """Resolve the pretext task by name (None -> 'mask'). Unknown name -> KeyError (fail fast)."""
    return PRETEXTS[name or 'mask']


__all__ = ['PretextTask', 'MaskTask', 'ForecastTask', 'ForecastDistTask', 'ContrastiveTask',
           'ElectraTask', 'PRETEXTS', 'get_pretext']
