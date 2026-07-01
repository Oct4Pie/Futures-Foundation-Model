"""Pretext-task REGISTRY — pluggable SSL pretraining objectives (stages).

Add a new pretrain experiment by dropping a module here (subclass PretextTask) and registering
its instance in PRETEXTS below; the orchestrator (ssl.py) stays untouched. Keeping each task in
its own file keeps ssl.py a clean orchestrator.

  mask        (stage 1) — BERT-style masked modeling
  forecast    (stage 2) — multi-horizon / variable-context candle seq2seq
  contrastive (stage 3) — trend contrastive (multi-positive InfoNCE by self-supervised trend key)
"""
from .base import PretextTask
from .mask import MaskTask
from .forecast import ForecastTask
from .contrastive import ContrastiveTask

PRETEXTS = {t.name: t for t in (MaskTask(), ForecastTask(), ContrastiveTask())}


def get_pretext(name):
    """Resolve the pretext task by name (None -> 'mask'). Unknown name -> KeyError (fail fast)."""
    return PRETEXTS[name or 'mask']


__all__ = ['PretextTask', 'MaskTask', 'ForecastTask', 'ContrastiveTask', 'PRETEXTS', 'get_pretext']
