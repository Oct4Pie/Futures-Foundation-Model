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
    if cfg.get('_export_encoder'):                    # ONNX encoder export (torch isolated here)
        from futures_foundation.finetune._ssl_torch import export_encoder_onnx
        export_encoder_onnx(cfg['_export_encoder'], ckpt=cfg.get('ckpt'),
                            C=int(cfg.get('C', 5)), seq=int(cfg.get('seq', 64)),
                            model_id=cfg.get('model_id', 'paris-noah/Mantis-8M'),
                            model_version=cfg.get('model_version'), device='cpu',
                            preprocessing=cfg.get('preprocessing'))
        return
    windows = np.load(cfg.pop('_windows'), mmap_mode='r')
    from futures_foundation.finetune._ssl_torch import embed_windows
    emb = embed_windows(windows, **cfg)
    np.save(d / 'emb.npy', emb)


if __name__ == '__main__':
    main(sys.argv[1])
