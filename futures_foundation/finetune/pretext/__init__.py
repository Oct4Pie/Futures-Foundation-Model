"""Pretext-task REGISTRY — pluggable SSL pretraining objectives (stages).

Add a new pretrain experiment by dropping a module here (subclass PretextTask) and registering
its instance in PRETEXTS below; the orchestrator (ssl.py) stays untouched. Keeping each task in
its own file keeps ssl.py a clean orchestrator.

  mask          (stage 1)   — BERT-style masked modeling
  contrastive   (stage 2)   — TEMPORAL-NEIGHBORHOOD contrastive: regime geometry from
                              multi-scale time proximity + augmentations, sigma-weighted
  forecast      (stage 3)   — multi-horizon / variable-context candle seq2seq
  forecast_dist (optional)  — distributional forecast experiment, outside the canonical 3 stages
  electra       (stage 4)   — TURN-ELECTRA (replaced-TURN detection): span-mask the regions around
                              DETECTED SWINGS (the event a pivot entry trades), a weak generator
                              fills each masked turn with a plausible alternative development (a
                              SYNTHETIC FAKE TURN), the encoder labels every bar real/replaced —
                              so it must learn how GENUINE turns develop vs plausible imposters
                              (fakeout-vs-real), pure SSL, zero labels, fully generic; warm from
                              the promoted base. (Prior slot occupants in git history: replaced-
                              candle RTD, break-hold.)
"""
from .base import PretextTask
from .mask import MaskTask
from .structure_mask import StructureMaskTask
from .forecast import ForecastTask
from .path import PathTask
from .forecast_dist import ForecastDistTask
from .contrastive import ContrastiveTask
from .electra import TurnElectraTask
from .nextleg import NextLegTask

PRETEXTS = {t.name: t for t in (MaskTask(), StructureMaskTask(), ForecastTask(), PathTask(), ForecastDistTask(),
                                ContrastiveTask(), TurnElectraTask(), NextLegTask())}


def get_pretext(name):
    """Resolve the pretext task by name (None -> 'mask'). Unknown name -> KeyError (fail fast)."""
    return PRETEXTS[name or 'mask']


__all__ = ['PretextTask', 'MaskTask', 'StructureMaskTask', 'ForecastTask', 'PathTask',
           'ForecastDistTask', 'ContrastiveTask',
           'TurnElectraTask', 'NextLegTask', 'PRETEXTS', 'get_pretext']
