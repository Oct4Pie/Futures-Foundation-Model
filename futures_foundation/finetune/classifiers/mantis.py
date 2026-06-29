"""MantisClassifier — fine-tunes the pretrained Mantis-8M backbone on multivariate
windows, with OUR OWN training loop (bypasses the slow MantisTrainer.fit).

Why our own loop (validated in colabs/mantis_ft.py): Mantis's fit() is CPU-bound
(per-batch loss.item() sync + num_workers=0 loader) and full-FT OOMs MPS (it runs
each of the C channels through the ViT, so batch*C activations explode). Our loop:

  * partial FT by default — last 2 of 6 transformer blocks + channel adapter + head
    (most autograd memory is backprop through ALL layers; the top blocks adapt the
    representation at a fraction of the cost — full-FT OOM'd MPS at 18GB)
  * all data moved to device ONCE; on-device batch indexing (no DataLoader)
  * on-device loss accumulation (no per-batch sync — the big MPS killer)
  * small batch + torch.mps.empty_cache()/epoch + thread cap -> won't freeze the box
  * val-based EARLY-STOPPING (restore best) + cosine LR

Registered as 'mantis'. torch is imported here (module loaded lazily via
get_classifier — never from the torch-free finetune parent).
"""
import os

import numpy as np
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn

from ..classifier import Classifier, register_classifier


@register_classifier('mantis')
class MantisClassifier(Classifier):
    def __init__(self, n_channels, new_channels=6, ft_mode='partial', unfreeze_blocks=2,
                 epochs=40, batch=64, lr=3e-4, weight_decay=0.05, patience=10,
                 val_frac=0.15, threads=2, device=None, model_id='paris-noah/Mantis-8M',
                 max_train=None, verbose=True):
        os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
        torch.set_num_threads(int(threads))
        self.n_channels = n_channels
        self.new_channels = min(new_channels, n_channels)
        self.ft_mode = ft_mode
        self.unfreeze_blocks = unfreeze_blocks
        self.epochs, self.batch, self.lr = epochs, batch, lr
        self.weight_decay, self.patience, self.val_frac = weight_decay, patience, val_frac
        self.model_id, self.verbose, self.max_train = model_id, verbose, max_train
        self.device = device or ('mps' if torch.backends.mps.is_available() else 'cpu')
        self.model = None

    # ---- internals --------------------------------------------------------
    def _build_model(self):
        from mantis.architecture import Mantis8M
        from mantis.adapters import LinearChannelCombiner
        from mantis.trainer.trainer_utils.architecture import FineTuningNetwork
        net = Mantis8M.from_pretrained(self.model_id)
        adapter = LinearChannelCombiner(num_channels=self.n_channels,
                                        new_num_channels=self.new_channels)
        head = nn.Sequential(nn.LayerNorm(net.hidden_dim * self.new_channels),
                             nn.Linear(net.hidden_dim * self.new_channels, 2))
        model = FineTuningNetwork(net, head, adapter).to(self.device)
        if self.ft_mode in ('partial', 'head'):
            for p in net.parameters():
                p.requires_grad = False
        if self.ft_mode == 'partial':
            for blk in net.vit_unit.transformer.layers[-self.unfreeze_blocks:]:
                for p in blk.parameters():
                    p.requires_grad = True
        # 'full' leaves all trainable
        return model

    @torch.no_grad()
    def _predict_idx(self, model, Xt, idx):
        model.eval()
        out = []
        for s in range(0, len(idx), self.batch):
            xb = Xt[torch.as_tensor(idx[s:s + self.batch], device=self.device)]
            out.append(torch.softmax(model(xb), 1)[:, 1].float().cpu().numpy())
        return np.concatenate(out) if out else np.array([])

    # ---- Classifier API ---------------------------------------------------
    def fit(self, X, y, X_val=None, y_val=None, seed=0):
        torch.manual_seed(seed)
        X = np.asarray(X, np.float32); y = np.asarray(y).astype(np.int64)
        if self.max_train and len(X) > self.max_train:
            sub = np.random.default_rng(seed).choice(len(X), self.max_train, replace=False)
            X, y = X[sub], y[sub]
        if X_val is None:
            rng = np.random.default_rng(seed)
            idx = np.arange(len(X)); rng.shuffle(idx)
            nv = max(1, int(len(idx) * self.val_frac))
            va_i, tr_i = idx[:nv], idx[nv:]
            Xtr, ytr, Xva, yva = X[tr_i], y[tr_i], X[va_i], y[va_i]
        else:
            Xtr, ytr = X, y
            Xva, yva = np.asarray(X_val, np.float32), np.asarray(y_val).astype(np.int64)

        Xtr_t = torch.tensor(Xtr, device=self.device)
        ytr_t = torch.tensor(ytr, dtype=torch.long, device=self.device)
        Xva_t = torch.tensor(Xva, device=self.device)

        model = self._build_model()
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        crit = nn.CrossEntropyLoss()
        n = len(Xtr_t)
        best_auc, best_state, bad = -1.0, None, 0
        va_idx = np.arange(len(Xva_t))
        for ep in range(self.epochs):
            model.train()
            perm = torch.randperm(n, device=self.device)
            ep_loss = torch.zeros((), device=self.device); nb = 0
            for s in range(0, n, self.batch):
                bid = perm[s:s + self.batch]
                out = model(Xtr_t[bid])
                loss = crit(out, ytr_t[bid])
                opt.zero_grad(); loss.backward(); opt.step()
                ep_loss += loss.detach(); nb += 1
            sched.step()
            if self.device == 'mps':
                torch.mps.empty_cache()
            va = roc_auc_score(yva, self._predict_idx(model, Xva_t, va_idx))
            if va > best_auc + 1e-4:
                best_auc, bad = va, 0
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
            else:
                bad += 1
            if self.verbose:
                print(f"    ep{ep:>2} loss={float(ep_loss)/max(nb,1):.4f} "
                      f"val_auc={va:.4f}{'  *' if bad == 0 else ''}", flush=True)
            if bad >= self.patience:
                break
        if best_state:
            model.load_state_dict(best_state)
        self.model = model
        self.best_val_auc = best_auc
        return self

    def predict_proba(self, X):
        X = np.asarray(X, np.float32)
        Xt = torch.tensor(X, device=self.device)
        return self._predict_idx(self.model, Xt, np.arange(len(Xt)))
