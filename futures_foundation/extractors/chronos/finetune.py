"""Generic supervised fine-tune of the Chronos backbone + a small head.

Strategy-agnostic: consumes (contexts, integer labels), returns a trained
(model, head). Deterministic given `seed` (torch + numpy seeded, backbone
reset to pretrained each call). No strategy or evaluation logic here.

OBJECTIVE (UniShape-inspired, 2026-06) — turn bolt-tiny into a CLASSIFIER, not a
forecaster. The point of fine-tuning the BACKBONE is a DISCRIMINATIVE embedding
(so the downstream frozen-embed + XGBoost improves), NOT next-value forecasting
and NOT label-CE alone (the prior TB-direction FT used a direction/CE objective
and didn't transfer). UniShape (AAAI-26) shows discriminative TS embeddings come
from CONTRASTIVE + PROTOTYPE objectives with shape preserved (no instance-norm).
We HAVE labels, so we use the supervised analog of MoCo-contrastive — Supervised
Contrastive (SupCon) — plus a prototype loss, on top of a light CE head:

    loss = CE(head) + λ_supcon·SupCon(embed,label) + λ_proto·ProtoCE(embed,label)

  • SupCon  : pulls same-class embeddings together / pushes classes apart — the
              discriminative structure that makes the frozen embedding separable.
  • Proto   : learnable class centers; embeddings pulled toward their class center
              (UniShape's prototype head), CE over cosine-sim to centers.
  • balanced batches: each batch is class-balanced so SupCon/Proto have positives
              even when the positive class is rare (~8% trends). Without this,
              contrastive is degenerate.
  • shape   : set CHRONOS_POOL_LOCSCALE=1 to append bolt's loc/scale (the magnitude
              it instance-norms away — its direction blind spot). Head dim
              auto-adapts to the pooled width.

LEAK NOTE: generic trainer — leak-cleanliness is the CALLER's job. Fit on data
strictly BEFORE the holdout (e.g. <=2025) so the embedding never sees the test
period. The prior TB-direction FT leaked by training across the walk-forward test
folds (great walk-forward, dead clean-holdout). Validate on a strict forward block.
"""
from dataclasses import dataclass

import numpy as np

from futures_foundation.extractors.chronos import backbone


@dataclass
class FTConfig:
    steps: int = 150                   # POC-scale short fine-tune
    batch: int = 16                    # balanced sampling gives SupCon positives; bump (128-256) for real runs
    lr_head: float = 1e-3
    lr_back: float = 2e-5
    n_classes: int = 3
    lambda_supcon: float = 1.0         # SupCon weight (0 -> legacy CE-only behaviour)
    lambda_proto: float = 0.3          # prototype-CE weight (0 -> off)
    tau: float = 0.1                   # SupCon temperature (UniShape: 0.1)
    proto_tau: float = 0.1             # prototype temperature
    balanced: bool = True              # class-balanced batches (positives for SupCon)


def _seed(seed):
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)


def fresh_head(d, n_classes):
    import torch.nn as nn
    return nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, n_classes))


def _supcon(z, y, tau):
    """Supervised Contrastive loss (Khosla et al.). z:[B,d], y:[B]. Same-label
    samples are positives; non-self all-class are the denominator. Returns 0 if
    no anchor has an in-batch positive (e.g. a class absent that batch)."""
    import torch
    import torch.nn.functional as F
    z = F.normalize(z, dim=1)
    sim = z @ z.t() / tau                                  # [B,B] cosine/τ
    B = z.shape[0]
    eye = torch.eye(B, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, float('-inf'))             # drop self-similarity
    pos = (y.unsqueeze(0) == y.unsqueeze(1)) & ~eye       # same-label, not self
    logprob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pcnt = pos.sum(1)
    valid = pcnt > 0
    if int(valid.sum()) == 0:
        return z.new_zeros(())
    loss = -(logprob * pos).sum(1)[valid] / pcnt[valid].clamp(min=1)
    return loss.mean()


def _balanced_batch(idx_by_c, batch, rng):
    """~batch/nc indices per non-empty class (with replacement if class small)."""
    picks = []
    nz = [ix for ix in idx_by_c if len(ix) > 0]
    if not nz:
        return np.array([], int)
    per = max(1, batch // len(nz))
    for idxs in nz:
        picks.append(rng.choice(idxs, per, replace=len(idxs) < per))
    return np.concatenate(picks)


def train(contexts, labels, cfg=FTConfig(), seed=0):
    """Joint fine-tune: pristine backbone + fresh head + class prototypes.
    loss = CE + λ_supcon·SupCon + λ_proto·ProtoCE (discriminative embedding)."""
    import torch
    import torch.nn.functional as F
    _seed(seed)
    m = backbone.fresh_model()                    # reset -> independent run
    for p in m.parameters():
        p.requires_grad_(True)
    X = np.asarray(contexts, np.float32)
    yl = np.asarray(labels)
    Y = torch.tensor(yl, dtype=torch.long)
    n = len(Y); nc = cfg.n_classes
    # probe pooled width (auto-adapts to CHRONOS_POOL_LOCSCALE shape-preservation)
    with torch.no_grad():
        d = backbone.pool(m, torch.tensor(X[:2])).shape[1]
    head = fresh_head(d, nc)
    centers = torch.nn.Parameter(torch.randn(nc, d) * 0.01)
    opt = torch.optim.Adam(
        [{'params': head.parameters(), 'lr': cfg.lr_head},
         {'params': [centers], 'lr': cfg.lr_head},
         {'params': m.parameters(), 'lr': cfg.lr_back}])
    ce = torch.nn.CrossEntropyLoss()
    idx_by_c = [np.where(yl == c)[0] for c in range(nc)]
    rng = np.random.default_rng(seed)
    m.train(); head.train()
    for _ in range(cfg.steps):
        b = (_balanced_batch(idx_by_c, cfg.batch, rng) if cfg.balanced
             else rng.choice(n, size=min(cfg.batch, n), replace=False))
        z = backbone.pool(m, torch.tensor(X[b]))
        yb = Y[b]
        loss = ce(head(z), yb)
        if cfg.lambda_supcon > 0:
            loss = loss + cfg.lambda_supcon * _supcon(z, yb, cfg.tau)
        if cfg.lambda_proto > 0:
            proto_logits = (F.normalize(z, dim=1)
                            @ F.normalize(centers, dim=1).t()) / cfg.proto_tau
            loss = loss + cfg.lambda_proto * ce(proto_logits, yb)
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
