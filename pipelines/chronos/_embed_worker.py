"""Subprocess worker: frozen Chronos embeddings.

Isolates torch's OpenMP from xgboost's — they segfault in one process on
macOS. NOT imported by the parent; invoked as
`python -m pipelines.chronos._embed_worker IN.npy OUT.npy BATCH`.

Model source resolves in priority:
  $CHRONOS_FT_CKPT  -> a local fine-tuned checkpoint directory (T5 or bolt)
  else              -> backbone.MODEL (default 'amazon/chronos-bolt-tiny')

Uses the official BaseChronosPipeline.embed() API (works for bolt, t5,
chronos-2 uniformly), then mean-pools the encoder tokens.
"""
import os
import sys

import numpy as np


def main(inp, outp, batch):
    import torch

    from chronos import BaseChronosPipeline

    from . import backbone
    src = os.environ.get('CHRONOS_FT_CKPT') or backbone.MODEL
    pipe = BaseChronosPipeline.from_pretrained(
        src, device_map='cpu', dtype=torch.float32)
    X = np.load(inp).astype(np.float32)
    out = []
    with torch.no_grad():
        for s in range(0, len(X), batch):
            emb, _ = pipe.embed(torch.tensor(X[s:s + batch]))
            out.append(emb.mean(1).cpu().numpy())
    np.save(outp, np.concatenate(out).astype(np.float32))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
