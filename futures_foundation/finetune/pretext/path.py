"""Revised Stage 3: fixed-wall-clock forward path distribution."""
from .base import PretextTask


class PathTask(PretextTask):
    name, trainer = 'path', 'train_ssl_path'
    primary_targets = ('fwd_absmove',)

    def reserve(self, cfg):
        # Eligibility is filtered per stream inside the trainer because one fixed wall-clock
        # horizon is 360 bars at 1m but only 6 bars at 60m. A global 616-bar parent would erase
        # coarse-stream validation coverage and make the tournament incomparable.
        return 0
