"""Subprocess worker: frozen-encoder embedding of windows (torch isolated from the parent).

Reads cfg.json (ckpt/model_id/device/batch) + a windows .npy (memmap), runs
_ssl_torch.embed_windows (frozen Mantis encoder, interpolated to native length), writes
emb.npy. Keeps torch out of the torch-free parent (libomp isolation), same pattern as the
mantis fine-tune worker.
"""
import json
import sys
from pathlib import Path

import numpy as np


def main(d):
    d = Path(d)
    cfg = json.loads((d / 'cfg.json').read_text())
    windows = np.load(cfg.pop('_windows'), mmap_mode='r')
    from futures_foundation.finetune._ssl_torch import embed_windows
    emb = embed_windows(windows, **cfg)
    np.save(d / 'emb.npy', emb)


if __name__ == '__main__':
    main(sys.argv[1])
