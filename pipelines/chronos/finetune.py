"""Generic supervised fine-tune of the Chronos backbone + a small head.

Strategy-agnostic: consumes (contexts, integer labels), returns a trained
(model, head). Deterministic given `seed` (torch + numpy seeded, backbone
reset to pretrained each call). No strategy or evaluation logic here.
"""
from dataclasses import dataclass

import numpy as np

from . import backbone


@dataclass
class FTConfig:
    steps: int = 150                   # POC-scale short fine-tune
    batch: int = 16
    lr_head: float = 1e-3
    lr_back: float = 2e-5
    n_classes: int = 3


def _seed(seed):
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)


def fresh_head(d, n_classes):
    import torch.nn as nn
    return nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, n_classes))


def train(contexts, labels, cfg=FTConfig(), seed=0):
    """Joint fine-tune: pristine backbone + fresh head, CE on `labels`."""
    import torch
    _seed(seed)
    m = backbone.fresh_model()                    # reset -> independent run
    head = fresh_head(backbone.d_model(), cfg.n_classes)
    for p in m.parameters():
        p.requires_grad_(True)
    opt = torch.optim.Adam(
        [{'params': head.parameters(), 'lr': cfg.lr_head},
         {'params': m.parameters(), 'lr': cfg.lr_back}])
    lossf = torch.nn.CrossEntropyLoss()
    X = np.asarray(contexts, np.float32)
    Y = torch.tensor(np.asarray(labels), dtype=torch.long)
    n = len(Y)
    rng = np.random.default_rng(seed)
    m.train(); head.train()
    for _ in range(cfg.steps):
        b = rng.choice(n, size=min(cfg.batch, n), replace=False)
        logits = head(backbone.pool(m, torch.tensor(X[b])))
        loss = lossf(logits, Y[b])
        opt.zero_grad(); loss.backward(); opt.step()
    return m, head


def predict(m, head, contexts):
    """Argmax class per context. {0,1,2,...} — meaning is the strategy's."""
    import torch
    m.eval(); head.eval()
    X = np.asarray(contexts, np.float32)
    out = []
    with torch.no_grad():
        for s in range(0, len(X), 64):
            z = head(backbone.pool(m, torch.tensor(X[s:s + 64])))
            out.append(z.argmax(-1).numpy())
    return np.concatenate(out) if out else np.array([], int)
