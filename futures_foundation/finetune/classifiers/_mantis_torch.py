"""Mantis torch fine-tune — runs ONLY inside the subprocess worker.

torch + the Mantis backbone load here. The parent-side adapter
(classifiers/mantis.py) is torch-free and spawns this via _worker, so torch never
shares a process with xgboost (libomp segfault). Our own loop (validated in
colabs/mantis_ft.py): partial FT of the last K transformer blocks + channel adapter
+ head, all data on-device once, on-device loss accumulation, val early-stop, MPS
cache clearing, thread cap — fast and won't OOM/freeze.
"""
import os

import numpy as np
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn


def build_model(C, *, new_channels=10, ft_mode='partial', unfreeze_blocks=2,
                device='cpu', model_id='paris-noah/Mantis-8M'):
    """Mantis backbone + channel adapter + head, with the freeze policy applied:
    'full' = all trainable; 'partial' = last `unfreeze_blocks` transformer blocks +
    adapter + head; 'head' = adapter + head only. Returns (model, new_c)."""
    from mantis.architecture import Mantis8M
    from mantis.adapters import LinearChannelCombiner
    from mantis.trainer.trainer_utils.architecture import FineTuningNetwork
    new_c = min(new_channels, C)
    net = Mantis8M.from_pretrained(model_id)
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
    """Export the fitted FineTuningNetwork (adapter+backbone+head) to ONNX. Input is
    a standardized window [batch, C, seq]; output logits [batch, 2]. The serve path
    builds the window via the labeler's mv_contexts + standardizes with the contract's
    mu/sd, then softmax over the logits for P(class 1)."""
    model.eval()
    dummy = torch.zeros(1, C, seq, device=device)
    torch.onnx.export(model, dummy, path, input_names=['window'], output_names=['logits'],
                      dynamic_axes={'window': {0: 'batch'}, 'logits': {0: 'batch'}},
                      opset_version=17)
    return path


def fit_predict_torch(Xtr, ytr, Xval, yval, Xeval, *, new_channels=10, ft_mode='partial',
                      unfreeze_blocks=2, epochs=40, batch=64, lr=3e-4, weight_decay=0.05,
                      patience=10, threads=2, device=None, model_id='paris-noah/Mantis-8M',
                      max_train=None, seed=0, export_onnx_path=None, verbose=True):
    """Returns (p_val, p_eval, best_val_auc, best_epoch). p_* are P(class 1)."""
    os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
    torch.set_num_threads(int(threads))
    dev = device or ('mps' if torch.backends.mps.is_available() else 'cpu')
    torch.manual_seed(seed)

    Xtr = np.asarray(Xtr, np.float32); ytr = np.asarray(ytr).astype(np.int64)
    if max_train and len(Xtr) > max_train:
        sub = np.random.default_rng(seed).choice(len(Xtr), max_train, replace=False)
        Xtr, ytr = Xtr[sub], ytr[sub]
    C = Xtr.shape[1]
    model, new_c = build_model(C, new_channels=new_channels, ft_mode=ft_mode,
                               unfreeze_blocks=unfreeze_blocks, device=dev, model_id=model_id)

    Xtr_t = torch.tensor(Xtr, device=dev)
    ytr_t = torch.tensor(ytr, dtype=torch.long, device=dev)
    Xva_t = torch.tensor(np.asarray(Xval, np.float32), device=dev)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    n = len(Xtr_t)
    yval_arr = np.asarray(yval).astype(int)

    @torch.no_grad()
    def pred(M, A):
        M.eval(); out = []
        for s in range(0, len(A), batch):
            out.append(torch.softmax(M(A[s:s + batch]), 1)[:, 1].float().cpu().numpy())
        return np.concatenate(out) if len(A) else np.array([])

    best, best_state, bad, best_epoch = -1.0, None, 0, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=dev)
        for s in range(0, n, batch):
            bid = perm[s:s + batch]
            loss = crit(model(Xtr_t[bid]), ytr_t[bid])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if dev == 'mps':
            torch.mps.empty_cache()
        va = roc_auc_score(yval_arr, pred(model, Xva_t)) if len(np.unique(yval_arr)) == 2 else 0.5
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
        export_model_onnx(model, C, Xtr.shape[2], export_onnx_path, device=dev)
    p_val = pred(model, Xva_t)
    p_eval = pred(model, torch.tensor(np.asarray(Xeval, np.float32), device=dev))
    return p_val, p_eval, float(best), int(best_epoch)
