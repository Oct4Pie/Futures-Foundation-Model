"""Shape-aware adapter — UniShape's TransformerEnc mechanism over FROZEN Chronos
per-patch tokens. Reusable FFM capability (NOT strategy IP): a learnable CLS token
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


def fit_and_infer(tokens, y, train_mask, *, depth=2, heads=4, mlp=512, epochs=40,
                  device='cpu', proto=True, lr=1e-3, weight_decay=1e-4,
                  batch_size=512, seed=0, proj_dim=None):
    """Train a ShapeAwareAdapter on the train split; return (probs[N], cls[N, work_d]).
    Backbone-agnostic: tokens [N, T, d] float, y [N] in {0,1}, train_mask [N] bool.
    proj_dim: if set, project each timestep d -> proj_dim (use for RAW per-bar
    feature sequences, e.g. all handcraft features over time); None -> tokens are
    already at the work dim (pretrained tokens). Minority class oversampled."""
    from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
    torch.manual_seed(seed)
    X = np.asarray(tokens, np.float32)
    Y = np.asarray(y).astype(np.int64)
    tr = np.asarray(train_mask, bool)
    dev = torch.device(
        'mps' if device == 'mps' and torch.backends.mps.is_available()
        else 'cuda' if device == 'cuda' and torch.cuda.is_available() else 'cpu')
    Xt, Yt = torch.tensor(X), torch.tensor(Y)
    cnt = np.bincount(Y[tr], minlength=2)
    w = (1.0 / np.clip(cnt, 1, None))[Y[tr]]
    sampler = WeightedRandomSampler(torch.tensor(w, dtype=torch.double), len(w),
                                    replacement=True)
    dl = DataLoader(TensorDataset(Xt[tr], Yt[tr]), batch_size=batch_size, sampler=sampler)
    work_d = int(proj_dim) if proj_dim else X.shape[2]
    in_dim = X.shape[2] if proj_dim else None
    model = ShapeAwareAdapter(d=work_d, in_dim=in_dim, n_tokens=X.shape[1], depth=depth,
                              heads=heads, mlp=mlp, n_classes=2, proto=proto).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    for ep in range(int(epochs)):
        model.train()
        last = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev)
            _, loss = model(xb, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            last = float(loss.item())
        if ep % 10 == 0 or ep == int(epochs) - 1:
            print(f"[shape_adapter] epoch {ep} loss {last:.4f}", file=sys.stderr, flush=True)
    model.eval()
    probs, cls = [], []
    with torch.no_grad():
        for s in range(0, len(Xt), 1024):
            clsf, logits = model.encode(Xt[s:s + 1024].to(dev))
            probs.append(torch.softmax(logits, 1)[:, 1].cpu().numpy())
            cls.append(clsf.cpu().numpy())
    return (np.concatenate(probs).astype(np.float32),
            np.concatenate(cls).astype(np.float32))


def _main(argv):
    """Subprocess worker: TOKENS.npy Y.npy TRAINMASK.npy OUT_PREFIX [device] [epochs].
    Writes OUT_PREFIX_probs.npy [N] + OUT_PREFIX_cls.npy [N, d]."""
    import os
    tok_p, y_p, tr_p, out_prefix = argv[:4]
    device = argv[4] if len(argv) > 4 else 'cpu'
    epochs = int(argv[5]) if len(argv) > 5 else 40
    proj = int(os.environ.get('ADAPTER_PROJ', '0')) or None    # raw-seq projection dim
    probs, cls = fit_and_infer(np.load(tok_p), np.load(y_p), np.load(tr_p),
                               device=device, epochs=epochs, proj_dim=proj)
    np.save(out_prefix + '_probs.npy', probs)
    np.save(out_prefix + '_cls.npy', cls)
    print(f"[shape_adapter] done: probs {probs.shape} cls {cls.shape}",
          file=sys.stderr, flush=True)


if __name__ == '__main__':
    _main(sys.argv[1:])
