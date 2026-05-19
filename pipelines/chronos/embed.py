"""Frozen Chronos-Bolt feature extractor (Route C, POC).

The supported, clean "backbone -> our head" surface: ChronosBoltPipeline
exposes `.embed(context) -> (emb[1, n_tokens, 256], scale)`. We FREEZE the
backbone (no grad, no fine-tune) and mean-pool the patch tokens into a
256-d vector. Causal by construction: the embedding/scale derive ONLY from
the context series we pass (always bars <= t).

Heavy deps (chronos, torch) imported lazily so data-contract tests import
without them.
"""
import numpy as np

MODEL = "amazon/chronos-bolt-tiny"
_PIPE = None


def _pipe():
    global _PIPE
    if _PIPE is None:
        import torch
        from chronos import BaseChronosPipeline
        _PIPE = BaseChronosPipeline.from_pretrained(
            MODEL, device_map="cpu", dtype=torch.float32)
    return _PIPE


def embed_contexts(contexts) -> np.ndarray:
    """contexts: iterable of 1-D causal series (each = bars <= decision t).
    Returns [N, 256] frozen mean-pooled Chronos-Bolt embeddings."""
    import torch
    p = _pipe()
    out = []
    with torch.no_grad():
        for c in contexts:
            t = torch.tensor(np.asarray(c, dtype=np.float32))
            emb, _ = p.embed(t)                 # [1, n_tokens, 256]
            out.append(emb[0].mean(0).cpu().numpy())   # -> [256]
    return np.stack(out) if out else np.empty((0, 256), np.float32)
