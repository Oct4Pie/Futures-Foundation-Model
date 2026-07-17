"""Revised Stage 1: shared-timestamp structural span reconstruction."""
from .base import PretextTask


class StructureMaskTask(PretextTask):
    name, trainer = 'structure_mask', 'train_ssl_structure_mask'
    primary_targets = ('trend_eff', 'range_expand')
