"""Chronos-Bolt backbone extractor.

`backbone` holds the frozen Chronos-Bolt embedding seam (subprocess-isolated;
moved here from futures_foundation/foundation.py). `_worker` is its torch
subprocess. `ChronosExtractor` adapts it to the FeatureExtractor interface.
"""
import numpy as np

from . import backbone
from ..base import _windows


class ChronosExtractor:
    name = 'chronos'

    def __init__(self, pool: str = 'mean'):
        self._fb = backbone
        self.pool = pool
        self.ctx = backbone.CTX                          # 128
        self.dim = backbone.pooled_dim(pool)             # 256 (mean)

    def embed(self, contexts, batch: int = 64) -> np.ndarray:
        return self._fb.embed(contexts, batch=batch, pool=self.pool)

    def embed_bars(self, close, indices, batch: int = 64) -> np.ndarray:
        if len(indices) == 0:
            return np.zeros((0, self.dim), np.float32)
        return self.embed(_windows(close, indices, self.ctx), batch=batch)


__all__ = ['ChronosExtractor', 'backbone']
