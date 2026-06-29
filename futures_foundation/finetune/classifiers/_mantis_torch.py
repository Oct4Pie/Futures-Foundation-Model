"""Mantis torch fine-tune — runs ONLY inside the subprocess worker. MPS-optimized.

Trains on arrays OR disk memmaps without starving the GPU on I/O:
  * BIG BATCH (512-1024): amortizes MPS per-launch overhead, fills the GPU.
  * CHUNK-TO-DEVICE + PREFETCH: load a large chunk (chunk_rows) onto the device ONCE
    and train many batches from it (no per-batch disk read / host->device copy); a
    background thread prefetches+standardizes the next chunk while the GPU trains the
    current one (overlap I/O with compute). Small folds (< chunk_rows) become a single
    chunk = all-on-device (the fast path) for free.
  * bf16 AUTOCAST (opt-in, amp=True): ~2x throughput + half activation memory on MPS,
    with fp32 fallback.
  * No per-batch loss.item() sync; val AUC syncs once/epoch; empty_cache per epoch.

torch loads here only (parent adapter is torch-free -> no xgboost collision).
"""
import os
import queue
import threading

import numpy as np
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn


def build_model(C, *, new_channels=10, ft_mode='partial', unfreeze_blocks=2,
                device='cpu', model_id='paris-noah/Mantis-8M', backbone_ckpt=None):
    """Mantis backbone + channel adapter + head with the freeze policy applied:
    'full' = all trainable; 'partial' = last `unfreeze_blocks` blocks + adapter + head;
    'head' = adapter + head only. backbone_ckpt = a contrastive-SSL-adapted encoder
    state_dict (from finetune.ssl) to initialize the backbone from instead of the
    vanilla pretrained weights. Returns (model, new_c)."""
    from mantis.architecture import Mantis8M
    from mantis.adapters import LinearChannelCombiner
    from mantis.trainer.trainer_utils.architecture import FineTuningNetwork
    new_c = min(new_channels, C)
    net = Mantis8M.from_pretrained(model_id)
    if backbone_ckpt:                                   # start from SSL-adapted encoder
        net.load_state_dict(torch.load(backbone_ckpt, map_location='cpu'))
    adapter = LinearChannelCombiner(num_channels=C, new_num_channels=new_c)
    head = nn.Sequential(nn.LayerNorm(net.hidden_dim * new_c),
                         nn.Linear(net.hidden_dim * new_c, 2))
    model = FineTuningNetwork(net, head, adapter).to(device)
    if ft_mode in ('partial', 'head'):
        for p in net.parameters():
            p.requires_grad = False
    if ft_mode == 'partial':
        for blk in net.vit_unit.transformer.layers[-unfreeze_blocks:]:
            for p in blk.parameters():
                p.requires_grad = True
    return model, new_c


def export_model_onnx(model, C, seq, path, device='cpu'):
    """Export the fitted model to ONNX. Input = standardized window [batch, C, seq];
    output logits [batch, 2]. Serve: mv_contexts -> standardize (contract mu/sd) ->
    softmax(logits)[:,1]."""
    model.eval()
    dummy = torch.zeros(1, C, seq, device=device)
    torch.onnx.export(model, dummy, path, input_names=['window'], output_names=['logits'],
                      dynamic_axes={'window': {0: 'batch'}, 'logits': {0: 'batch'}},
                      opset_version=17)
    return path


def _chunk_iter(X, rows, chunk_rows, mu, sd):
    """Yield (sorted_idx, standardized_chunk_np) — each chunk read from X (memmap/array)
    + standardized on a BACKGROUND thread (prefetch), so disk I/O overlaps GPU compute.
    Rows are sorted within a chunk for fast memmap reads; the row PERMUTATION across
    chunks provides SGD shuffling."""
    chunks = [np.sort(rows[i:i + chunk_rows]) for i in range(0, len(rows), chunk_rows)]

    def read(idx):
        xb = np.asarray(X[idx], np.float32)
        if mu is not None:
            xb = (xb - mu[None, :, None]) / sd[None, :, None]
        return idx, xb

    q = queue.Queue(maxsize=2)

    def producer():
        for ch in chunks:
            q.put(read(ch))
        q.put(None)

    threading.Thread(target=producer, daemon=True).start()
    while True:
        item = q.get()
        if item is None:
            break
        yield item


def fit_predict_torch(Xtr, ytr, Xval, yval, Xeval, *, new_channels=10, ft_mode='partial',
                      unfreeze_blocks=2, epochs=40, batch=64, chunk_rows=65536, amp=False,
                      amp_dtype='fp16', lr=3e-4, weight_decay=0.05, patience=10, threads=2,
                      device=None, model_id='paris-noah/Mantis-8M', backbone_ckpt=None,
                      max_train=None, standardize_mu=None, standardize_sd=None,
                      export_onnx_path=None, seed=0, verbose=True):
    """Returns (p_val, p_eval, best_val_auc, best_epoch). Xtr/Xval/Xeval: arrays or
    memmaps. standardize_mu/sd: per-channel arrays applied per-chunk (None = already
    standardized)."""
    os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
    torch.set_num_threads(int(threads))
    dev = device or ('mps' if torch.backends.mps.is_available() else 'cpu')
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    ytr = np.asarray(ytr).astype(np.int64)
    yval = np.asarray(yval).astype(int)
    n = len(Xtr)
    tr_rows = np.arange(n)
    if max_train and n > max_train:
        tr_rows = np.sort(rng.choice(n, max_train, replace=False))
    C = int(Xtr.shape[1]); seq = int(Xtr.shape[2])
    mu = None if standardize_mu is None else np.asarray(standardize_mu, np.float32)
    sd = None if standardize_sd is None else np.asarray(standardize_sd, np.float32)
    _adt = torch.float16 if str(amp_dtype).lower() in ('fp16', 'float16') else torch.bfloat16
    amp_ctx = (lambda: torch.autocast(device_type=dev, dtype=_adt)) if amp \
        else (lambda: _nullctx())

    model, new_c = build_model(C, new_channels=new_channels, ft_mode=ft_mode,
                               unfreeze_blocks=unfreeze_blocks, device=dev,
                               model_id=model_id, backbone_ckpt=backbone_ckpt)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()

    @torch.no_grad()
    def pred(M, X):
        M.eval(); out = []
        for idx, ch_np in _chunk_iter(X, np.arange(len(X)), chunk_rows, mu, sd):
            ch = torch.tensor(ch_np, device=dev)
            for b in range(0, len(ch), batch):
                with amp_ctx():
                    out.append(torch.softmax(M(ch[b:b + batch]), 1)[:, 1].float().cpu().numpy())
            del ch
        return np.concatenate(out) if len(X) else np.array([])

    best, best_state, bad, best_epoch = -1.0, None, 0, 0
    for ep in range(epochs):
        model.train()
        perm = tr_rows[rng.permutation(len(tr_rows))]
        for idx, ch_np in _chunk_iter(Xtr, perm, chunk_rows, mu, sd):
            ch = torch.tensor(ch_np, device=dev)
            yb_all = torch.tensor(ytr[idx], dtype=torch.long, device=dev)
            for b in range(0, len(ch), batch):
                with amp_ctx():
                    loss = crit(model(ch[b:b + batch]), yb_all[b:b + batch])
                opt.zero_grad(); loss.backward(); opt.step()
            del ch, yb_all
        sched.step()
        if dev == 'mps':
            torch.mps.empty_cache()
        va = roc_auc_score(yval, pred(model, Xval)) if len(np.unique(yval)) == 2 else 0.5
        if va > best + 1e-4:
            best, bad, best_epoch = va, 0, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if verbose:
            print(f"    ep{ep:>2} val_auc={va:.4f}{'  *' if bad == 0 else ''}", flush=True)
        if bad >= patience:
            break
    if best_state:
        model.load_state_dict(best_state)
    if export_onnx_path:
        export_model_onnx(model, C, seq, export_onnx_path, device=dev)
    return pred(model, Xval), pred(model, Xeval), float(best), int(best_epoch)


class _nullctx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False
