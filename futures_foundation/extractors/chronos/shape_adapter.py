"""Shape-aware adapter — UniShape's TransformerEnc mechanism over FROZEN Chronos
per-patch tokens. Reusable FFM capability: a learnable CLS token
attends (with positional encoding) over the token sequence, aggregating the
discriminative developing shape into a class representation, supervised by CE + a
prototype (shape-clustering) loss.

UniShape learns shape with a conv tokenizer it trains end-to-end; we keep its
shape-LEARNING head and swap the tokenizer for Chronos Bolt's frozen per-patch
tokens (pool='seq' -> [N, n_patches+1, d_model]). Backbone stays frozen -> small,
trainable, ONNX-deployable.

torch is imported at MODULE TOP (needed for the nn.Module class defs), so — like
`_worker.py` — this module is ONLY loaded in a SUBPROCESS (run as __main__) or in
a CHRONOS_TORCH_TESTS-gated test. It is NOT imported by the package __init__, so
the torch-free xgboost parent never pulls it (libomp-collision safe).
"""
import math
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d, max_len=64):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):                        # x: [B, T, d]
        return x + self.pe[:, :x.size(1)]


class ShapeAwareAdapter(nn.Module):
    """UniShape TransformerEnc adapter over frozen per-patch tokens (+ prototype).
    Input  x: [B, T, d] frozen tokens.  Output: class logits [B, n_classes].
    `encode(x)` returns (cls_shape_embedding [B, d], logits)."""

    def __init__(self, d=256, n_tokens=9, depth=2, heads=4, mlp=512, n_classes=2,
                 dropout=0.1, proto=True, proto_dim=128, proto_alpha=0.3,
                 proto_temp=0.1, in_dim=None):
        super().__init__()
        # optional input projection: lets the adapter run on RAW per-bar feature
        # sequences (e.g. handcraft features over time, in_dim=k) -> work dim d,
        # not just pretrained tokens (in_dim=None -> tokens already at d).
        self.in_proj = nn.Linear(in_dim, d) if in_dim else None
        self.cls = nn.Parameter(torch.randn(d) * 0.02)
        self.pos = PositionalEncoding(d, max_len=n_tokens + 1)
        layer = nn.TransformerEncoderLayer(d, heads, mlp, dropout, batch_first=True)
        self.tr = nn.TransformerEncoder(layer, depth)
        self.head = nn.Sequential(
            nn.Linear(d, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(256, n_classes))
        self.proto = proto
        if proto:
            self.proto_proj = nn.Linear(d, proto_dim)
            self.centers = nn.Parameter(torch.randn(n_classes, proto_dim) * 0.02)
            self.proto_alpha, self.proto_temp = proto_alpha, proto_temp

    def encode(self, x):                         # x: [B, T, d_in] -> (cls_emb, logits)
        if self.in_proj is not None:             # raw feature seq -> work dim
            x = self.in_proj(x)
        b = x.shape[0]
        cls = self.cls.unsqueeze(0).expand(b, 1, -1)
        h = self.pos(torch.cat([cls, x], dim=1))
        h = self.tr(h)
        clsf = h[:, 0]                            # aggregated shape representation
        return clsf, self.head(clsf)

    def forward(self, x, labels=None):
        clsf, logits = self.encode(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            if self.proto:                        # shape-clustering supervision
                z = F.normalize(self.proto_proj(clsf), dim=-1)
                c = F.normalize(self.centers, dim=-1)
                loss = loss + self.proto_alpha * F.cross_entropy(
                    (z @ c.T) / self.proto_temp, labels)
        return logits, loss


class AxialShapeAdapter(nn.Module):
    """Cross-feature + temporal progression learner (for RAW per-bar feature seqs).
    Input x: [B, T, F]. Two streams fused into the class representation:
      TIME    — per-bar tokens (Linear F->d) + positional enc + self-attention
                -> learns the temporal pattern (how the blended state evolves).
      FEATURE — each FEATURE's whole time-series as a token (Linear T->d) + self-
                attention -> learns how each field PROGRESSES and how fields develop
                TOGETHER (mom contracts WHILE vol dries) — the cross-feature axis the
                plain Linear+time adapter cannot represent.
    encode(x) -> (cls [B, 2d], logits)."""

    def __init__(self, T, F, d=64, depth=2, heads=4, mlp=256, n_classes=2,
                 dropout=0.1, proto=False, proto_dim=128, proto_alpha=0.3,
                 proto_temp=0.1):
        super().__init__()
        self.time_proj = nn.Linear(F, d)
        self.feat_proj = nn.Linear(T, d)
        self.cls_t = nn.Parameter(torch.randn(d) * 0.02)
        self.cls_f = nn.Parameter(torch.randn(d) * 0.02)
        self.pos_t = PositionalEncoding(d, max_len=T + 1)
        mk = lambda: nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, heads, mlp, dropout, batch_first=True), depth)
        self.tr_t, self.tr_f = mk(), mk()
        self.head = nn.Sequential(
            nn.Linear(2 * d, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(256, n_classes))
        self.proto = proto
        if proto:
            self.proto_proj = nn.Linear(2 * d, proto_dim)
            self.centers = nn.Parameter(torch.randn(n_classes, proto_dim) * 0.02)
            self.proto_alpha, self.proto_temp = proto_alpha, proto_temp

    def encode(self, x):                         # x: [B, T, F]
        b = x.shape[0]
        t = self.pos_t(torch.cat([self.cls_t.expand(b, 1, -1),
                                  self.time_proj(x)], dim=1))      # time tokens
        clst = self.tr_t(t)[:, 0]
        f = torch.cat([self.cls_f.expand(b, 1, -1),
                       self.feat_proj(x.transpose(1, 2))], dim=1)  # feature tokens
        clsf = self.tr_f(f)[:, 0]                                  # (features unordered: no pos)
        z = torch.cat([clst, clsf], dim=-1)
        return z, self.head(z)

    def forward(self, x, labels=None):
        z, logits = self.encode(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            if self.proto:
                zp = F.normalize(self.proto_proj(z), dim=-1)
                c = F.normalize(self.centers, dim=-1)
                loss = loss + self.proto_alpha * F.cross_entropy(
                    (zp @ c.T) / self.proto_temp, labels)
        return logits, loss


def fit_and_infer(tokens, y, train_mask, *, depth=2, heads=4, mlp=512, epochs=80,
                  device='cpu', proto=False, lr=2e-3, weight_decay=1e-4,
                  batch_size=512, seed=0, proj_dim=None, val_frac=0.15, patience=10,
                  arch='shape'):
    """Train a ShapeAwareAdapter with OVERFIT GUARDS; return (probs[N], cls[N,work_d],
    val_auc). An inner train/val split is carved off train_mask:
      - EARLY STOPPING on val AUC (patience) + restore best-val weights
      - LR SCHEDULING (ReduceLROnPlateau on val AUC)
      - returns best val_auc so the caller can gate the VAL->TEST gap (reject overfit)
    Backbone-agnostic: tokens [N,T,d], y [N] in {0,1}, train_mask [N] bool. proj_dim
    projects raw per-bar feature seqs -> work dim. Minority oversampled. Defaults
    (proto OFF, lr 2e-3) are the diagnosed-best for raw feature sequences."""
    import copy
    from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
    from sklearn.metrics import roc_auc_score
    torch.manual_seed(seed)
    X = np.asarray(tokens, np.float32)
    Y = np.asarray(y).astype(np.int64)
    tr = np.asarray(train_mask, bool)
    dev = torch.device(
        'mps' if device == 'mps' and torch.backends.mps.is_available()
        else 'cuda' if device == 'cuda' and torch.cuda.is_available() else 'cpu')
    Xt, Yt = torch.tensor(X), torch.tensor(Y)
    # inner train/val split off train_mask (early stopping + LR scheduling)
    tr_idx = np.flatnonzero(tr)
    rs = np.random.default_rng(seed); rs.shuffle(tr_idx)
    nval = max(64, int(len(tr_idx) * val_frac))
    val_idx, in_idx = tr_idx[:nval], tr_idx[nval:]
    cnt = np.bincount(Y[in_idx], minlength=2)
    wts = (1.0 / np.clip(cnt, 1, None))[Y[in_idx]]
    sampler = WeightedRandomSampler(torch.tensor(wts, dtype=torch.double), len(wts),
                                    replacement=True)
    dl = DataLoader(TensorDataset(Xt[in_idx], Yt[in_idx]), batch_size=batch_size,
                    sampler=sampler)
    work_d = int(proj_dim) if proj_dim else X.shape[2]
    if arch == 'axial':                          # cross-feature + temporal streams
        model = AxialShapeAdapter(T=X.shape[1], F=X.shape[2], d=work_d, depth=depth,
                                  heads=heads, mlp=mlp, n_classes=2, proto=proto).to(dev)
    else:
        in_dim = X.shape[2] if proj_dim else None
        model = ShapeAwareAdapter(d=work_d, in_dim=in_dim, n_tokens=X.shape[1],
                                  depth=depth, heads=heads, mlp=mlp, n_classes=2,
                                  proto=proto).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='max', factor=0.5, patience=max(2, patience // 3))
    Xval = Xt[val_idx].to(dev)
    best_auc, best_state, bad, ep = -1.0, None, 0, 0
    for ep in range(int(epochs)):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev)
            _, loss = model(xb, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vp = torch.softmax(model.encode(Xval)[1], 1)[:, 1].cpu().numpy()
        vauc = float(roc_auc_score(Y[val_idx], vp))
        sched.step(vauc)
        if vauc > best_auc + 1e-4:
            best_auc, best_state, bad = vauc, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
        if ep % 10 == 0:
            print(f"[shape_adapter] ep {ep} val_auc {vauc:.4f} (best {best_auc:.4f})",
                  file=sys.stderr, flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"[shape_adapter] early-stop @ ep {ep}  best val_auc {best_auc:.4f}",
          file=sys.stderr, flush=True)
    model.eval()
    probs, cls = [], []
    with torch.no_grad():
        for s in range(0, len(Xt), 1024):
            clsf, logits = model.encode(Xt[s:s + 1024].to(dev))
            probs.append(torch.softmax(logits, 1)[:, 1].cpu().numpy())
            cls.append(clsf.cpu().numpy())
    return (np.concatenate(probs).astype(np.float32),
            np.concatenate(cls).astype(np.float32), best_auc)


def _main(argv):
    """Subprocess worker: TOKENS.npy Y.npy TRAINMASK.npy OUT_PREFIX [device] [epochs].
    Writes OUT_PREFIX_probs.npy [N] + OUT_PREFIX_cls.npy [N, d]."""
    import os
    tok_p, y_p, tr_p, out_prefix = argv[:4]
    device = argv[4] if len(argv) > 4 else 'cpu'
    epochs = int(argv[5]) if len(argv) > 5 else 40
    proj = int(os.environ.get('ADAPTER_PROJ', '0')) or None    # raw-seq projection dim
    proto = os.environ.get('ADAPTER_PROTO', '0') == '1'        # prototype loss (off=diagnosed best)
    lr = float(os.environ.get('ADAPTER_LR', '2e-3'))
    patience = int(os.environ.get('ADAPTER_PATIENCE', '10'))
    val_frac = float(os.environ.get('ADAPTER_VAL_FRAC', '0.15'))
    arch = os.environ.get('ADAPTER_ARCH', 'shape')             # 'shape' | 'axial'
    probs, cls, val_auc = fit_and_infer(np.load(tok_p), np.load(y_p), np.load(tr_p),
                                        device=device, epochs=epochs, proj_dim=proj,
                                        proto=proto, lr=lr, patience=patience,
                                        val_frac=val_frac, arch=arch)
    np.save(out_prefix + '_probs.npy', probs)
    np.save(out_prefix + '_cls.npy', cls)
    np.save(out_prefix + '_valauc.npy', np.array([val_auc], np.float32))
    print(f"[shape_adapter] done: val_auc {val_auc:.4f} probs {probs.shape} cls {cls.shape}",
          file=sys.stderr, flush=True)


if __name__ == '__main__':
    _main(sys.argv[1:])
