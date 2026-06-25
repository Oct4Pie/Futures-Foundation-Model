"""Subprocess worker: frozen Chronos embeddings.

Isolates torch's OpenMP from xgboost's — they segfault in one process on
macOS. NOT imported by the parent; invoked as
`python -m futures_foundation._embed_worker IN.npy OUT.npy BATCH [POOL] [LOCSCALE_OUT]`.

Model source resolves in priority:
  $CHRONOS_FT_CKPT  -> a local fine-tuned checkpoint directory (T5 or bolt)
  else              -> foundation.MODEL (default 'amazon/chronos-bolt-tiny')

Uses the official BaseChronosPipeline.embed() API. POOL (Tier-1 lever):
  'mean'    masked-mean over tokens (legacy default — byte-identical)
  'reg'     the [REG] summary token (Bolt APPENDS it → last position)
  'meanreg' concat([mean, reg]) -> 2*d_model
If LOCSCALE_OUT is given, also writes [N,2] (loc, scale) = the window mean/std
that instance_norm strips before the encoder (Chronos's magnitude blind spot).
"""
import os
import sys

import numpy as np


def _resolve_embed_device(requested: str, mps_available: bool) -> str:
    """Embed device: honor 'mps' only if Metal is actually available, else fall
    back to 'cpu' (the parity-safe default). Pure/torch-free -> unit-testable."""
    if requested == 'mps' and not mps_available:
        return 'cpu'
    return requested


def main(inp, outp, batch, pool='mean', locscale_out=None):
    import torch

    from chronos import BaseChronosPipeline

    from . import backbone as foundation
    # device: default CPU (byte-identical / parity-safe for produced bundles);
    # CHRONOS_EMBED_DEVICE=mps opts into Metal (no help for tiny bolt, but safe).
    dev = _resolve_embed_device(os.environ.get('CHRONOS_EMBED_DEVICE', 'cpu'),
                                torch.backends.mps.is_available())
    src = foundation.active_source()
    is_local = os.path.isabs(src) or os.path.exists(src)
    tag = 'FINE-TUNED' if is_local else 'FROZEN-VANILLA'
    print(f"[chronos worker] loading {tag} backbone: {src} (pool={pool}, dev={dev})",
          flush=True, file=sys.stderr)
    pipe = BaseChronosPipeline.from_pretrained(
        src, device_map=dev, dtype=torch.float32)
    X = np.load(inp).astype(np.float32)
    out, lss = [], []
    with torch.no_grad():
        for s in range(0, len(X), batch):
            emb, ls = pipe.embed(torch.tensor(X[s:s + batch]).to(dev))
            if pool == 'mean':
                v = emb.mean(1)
            elif pool == 'reg':
                v = emb[:, -1, :]
            elif pool == 'meanreg':
                v = torch.cat([emb.mean(1), emb[:, -1, :]], dim=-1)
            else:
                raise ValueError(f"pool {pool!r} not in mean|reg|meanreg")
            if os.environ.get('CHRONOS_POOL_LOCSCALE') == '1':
                # append bolt's own loc+scale (instance-norm de-norm terms) ->
                # +2 dims, restoring level/vol. Must match the FT pool.
                if isinstance(ls, (tuple, list)):
                    loc, scale = ls
                    lsv = torch.stack([loc.reshape(-1), scale.reshape(-1)], dim=-1)
                else:
                    lsv = torch.as_tensor(ls).reshape(len(v), -1)
                v = torch.cat([v, lsv.to(v.dtype)], dim=-1)
            out.append(v.cpu().numpy())
            if locscale_out:
                if isinstance(ls, (tuple, list)):
                    loc, scale = ls
                    lsv = torch.stack([loc.reshape(-1), scale.reshape(-1)], dim=-1)
                else:
                    lsv = torch.as_tensor(ls).reshape(len(v), -1)
                lss.append(lsv.cpu().numpy())
    np.save(outp, np.concatenate(out).astype(np.float32))
    if locscale_out:
        np.save(locscale_out, np.concatenate(lss).astype(np.float32))


if __name__ == '__main__':
    # argv: IN OUT BATCH [POOL] [LOCSCALE_OUT]
    pool = sys.argv[4] if len(sys.argv) > 4 else 'mean'
    ls_out = sys.argv[5] if len(sys.argv) > 5 else None
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), pool, ls_out)
