"""Feature-extractor interface + the swappable runner.

`FeatureExtractor` is the single protocol the pipeline depends on; concrete
backbones (chronos.py, moment.py) implement it. `evaluate_with_extractor` runs a
strategy through the proven `evaluate.run` with ANY extractor by injecting its
embeddings via ev.run's `embed_cache` seam — so swapping the backbone needs ZERO
change to the core evaluator, and the default (Chronos) is byte-identical to the
standard path.
"""
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class FeatureExtractor(Protocol):
    name: str
    ctx: int                       # causal window length it consumes
    dim: int                       # embedding dimensionality

    def embed(self, contexts, batch: int = 64) -> np.ndarray:
        """Embed pre-built equal-length (=ctx) log-price windows -> [N, dim]."""
        ...

    def embed_bars(self, close, indices, batch: int = 64) -> np.ndarray:
        """Build the causal ctx-window ending at each decision index (left-pad
        if a bar lacks ctx history) and embed -> [len(indices), dim]."""
        ...


def _windows(close, indices, ctx):
    """Causal log-price windows ending at each index, left-padded to ctx."""
    lp = np.log(np.asarray(close, dtype=np.float64))
    out = []
    for i in indices:
        w = lp[max(0, i - ctx + 1):i + 1]
        if len(w) < ctx:
            w = np.concatenate([np.full(ctx - len(w), w[0]), w])
        out.append(w)
    return np.asarray(out, np.float32)


def evaluate_with_extractor(labeler, extractor=None, *, seeds=(0, 1, 2),
                            train_m=3, val_m=1, test_m=1, max_folds=None,
                            **runkw):
    """Run a strategy through evaluate.run with ANY FeatureExtractor.

    Builds the extractor's embeddings at every signal bar the evaluator will see
    (per-ticker, from calendar()'s close series + the keys' bar-indices) and
    injects them via ev.run's embed_cache. Default extractor = Chronos
    (byte-identical to the standard path). Returns ev.run's verdict dict."""
    from .chronos import ChronosExtractor
    from ..pipeline import evaluate as ev
    from ..pipeline.data import walk_forward_folds
    extractor = extractor or ChronosExtractor()

    cal = labeler.calendar()
    close_by_tk = {tk: g['target'].to_numpy(np.float64)
                   for tk, g in cal.groupby('item_id', sort=False)}

    folds = list(walk_forward_folds(cal, train_m, val_m, test_m))
    if max_folds is not None:
        folds = folds[:max_folds]
    idx_by_tk = {}
    for _f, tr, val, te in folds:
        v0, t0 = val['timestamp'].min(), te['timestamp'].min()
        for lo, hi, ts in ((tr['timestamp'].min(), tr['timestamp'].max(), v0),
                           (v0, val['timestamp'].max(), t0),
                           (te['timestamp'].min(),
                            te['timestamp'].max() + np.timedelta64(1, 'ns'), None)):
            _C, _Y, K = labeler.build(lo, hi, ts)
            for k in K:
                idx_by_tk.setdefault(k[0], set()).add(k[1])

    cache = {}
    for tk, idxset in idx_by_tk.items():
        idxs = sorted(idxset)
        embs = extractor.embed_bars(close_by_tk[tk], idxs)
        cache[tk] = {i: e for i, e in zip(idxs, embs)}

    return ev.run(labeler, embed_cache=cache, seeds=seeds, train_m=train_m,
                  val_m=val_m, test_m=test_m, max_folds=max_folds, **runkw)
