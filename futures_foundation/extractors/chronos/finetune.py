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

import os

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
    log_every: int = 0                 # >0: print step/loss every N steps (long runs)
    self_supervised: bool = False      # Option A: MoCo/SimCLR contrastive, NO labels
    aug_jitter: float = 0.10           # jitter σ as fraction of per-window std
    aug_mask: float = 0.15             # fraction of bars time-masked per view
    lora: bool = False                 # LoRA: freeze base, learn low-rank deltas (no drift)
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0


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
    pos = (y.unsqueeze(0) == y.unsqueeze(1)) & ~eye       # same-label, not self
    # MPS-safe log-prob: row-max stabilize + multiplicative self-mask (NO -inf,
    # which yields NaN under MPS logsumexp). Denominator excludes self (Khosla).
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()
    exp_sim = torch.exp(sim) * (~eye).to(sim.dtype)       # zero self in denom
    logprob = sim - torch.log(exp_sim.sum(1, keepdim=True) + 1e-12)
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


def _augment_view(X, rng, jitter, mask_frac):
    """Two-way TS augmentation for SimCLR/MoCo. X:[B,T] log-price windows. Only
    SHAPE-changing augs (level/scale are stripped by bolt's instance_norm anyway):
    per-window jitter + contiguous time-masking (hold-last). Returns a new [B,T]."""
    B, T = X.shape
    sd = X.std(axis=1, keepdims=True) + 1e-8
    out = X + rng.normal(0.0, 1.0, X.shape).astype(np.float32) * (jitter * sd)
    span = max(1, int(mask_frac * T))
    for r in range(B):                                # contiguous hold-last mask
        s = int(rng.integers(0, max(1, T - span)))
        out[r, s:s + span] = out[r, max(0, s - 1)]
    return out


def _ntxent(z1, z2, tau):
    """NT-Xent (SimCLR). z1,z2:[B,d] are two views of the same B windows. Positive
    pair = (i, i+B); all other 2B-2 are negatives. MPS-safe: row-max stabilize +
    multiplicative self-mask (NO -inf -> no NaN on Metal)."""
    import torch
    import torch.nn.functional as F
    z = torch.cat([z1, z2], dim=0)                    # [2B,d]
    z = F.normalize(z, dim=1)
    n2 = z.shape[0]; B = z1.shape[0]
    sim = z @ z.t() / tau                             # [2B,2B]
    eye = torch.eye(n2, dtype=torch.bool, device=z.device)
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()
    exp_sim = torch.exp(sim) * (~eye).to(sim.dtype)   # drop self from denominator
    logprob = sim - torch.log(exp_sim.sum(1, keepdim=True) + 1e-12)
    pos = torch.arange(n2, device=z.device)
    pos = (pos + B) % n2                              # i<->i+B partner index
    return -logprob[torch.arange(n2, device=z.device), pos].mean()


def _wrap_lora(m, cfg):
    """Freeze base, inject LoRA adapters (official Chronos targets: attn q/k/v/o +
    output projection). Preserves the pretrained embedding -> no full-FT drift."""
    from peft import LoraConfig, get_peft_model
    lc = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
                    lora_dropout=cfg.lora_dropout, bias='none',
                    target_modules=['SelfAttention.q', 'SelfAttention.v',
                                    'SelfAttention.k', 'SelfAttention.o',
                                    'output_patch_embedding.output_layer'])
    m = get_peft_model(m, lc)
    if cfg.log_every:
        m.print_trainable_parameters()
    return m


def _finalize(m, cfg):
    """Merge LoRA into base (so backbone.embed loads a plain ckpt) + move to cpu."""
    if cfg.lora:
        m = m.merge_and_unload()
    m.to('cpu')
    return m


def _train_ssl(contexts, cfg, seed):
    """Option A: self-supervised contrastive FT (NO labels). Returns (m, None).
    The FT only adapts bolt-tiny into a discriminative feature extractor; the
    downstream classifier (XGBoost) owns the trend label. Leak-immune."""
    import torch
    _seed(seed)
    dev = os.environ.get('FFM_FT_DEVICE') or (
        'mps' if torch.backends.mps.is_available()
        else 'cuda' if torch.cuda.is_available() else 'cpu')
    m = backbone.fresh_model(); m.to(dev)
    if cfg.lora:
        m = _wrap_lora(m, cfg); m.to(dev)         # freeze base, train adapters only
    else:
        for p in m.parameters():
            p.requires_grad_(True)
    X = np.asarray(contexts, np.float32)
    n = len(X)
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=cfg.lr_back)
    rng = np.random.default_rng(seed)
    m.train()
    if cfg.log_every:
        print(f"[finetune] device={dev} | SSL contrastive | {cfg.steps} steps "
              f"batch={cfg.batch} | {n:,} ctx (NO labels)", flush=True)
    for step in range(cfg.steps):
        b = rng.choice(n, size=min(cfg.batch, n), replace=False)
        xb = X[b]
        v1 = _augment_view(xb.copy(), rng, cfg.aug_jitter, cfg.aug_mask)
        v2 = _augment_view(xb.copy(), rng, cfg.aug_jitter, cfg.aug_mask)
        z1 = backbone.pool(m, torch.tensor(v1, device=dev))
        z2 = backbone.pool(m, torch.tensor(v2, device=dev))
        loss = _ntxent(z1, z2, cfg.tau)
        opt.zero_grad(); loss.backward(); opt.step()
        if cfg.log_every and (step % cfg.log_every == 0 or step == cfg.steps - 1):
            print(f"[finetune] step {step+1}/{cfg.steps}  ntxent={float(loss):.4f}",
                  flush=True)
    return _finalize(m, cfg), None


def train(contexts, labels, cfg=FTConfig(), seed=0):
    """Joint fine-tune. Two modes:
      - cfg.self_supervised (Option A): MoCo/SimCLR contrastive, NO labels ->
        bolt-tiny becomes a discriminative extractor; XGBoost owns the label.
      - else (Option B): loss = CE + λ_supcon·SupCon + λ_proto·ProtoCE."""
    if cfg.self_supervised:
        return _train_ssl(contexts, cfg, seed)             # labels ignored
    import torch
    import torch.nn.functional as F
    _seed(seed)
    dev = os.environ.get('FFM_FT_DEVICE') or (    # auto: local GPU if available
        'mps' if torch.backends.mps.is_available()
        else 'cuda' if torch.cuda.is_available() else 'cpu')
    m = backbone.fresh_model()                    # reset -> independent run
    m.to(dev)
    X = np.asarray(contexts, np.float32)
    yl = np.asarray(labels)
    Y = torch.tensor(yl, dtype=torch.long, device=dev)
    n = len(Y); nc = cfg.n_classes
    # probe pooled width (auto-adapts to CHRONOS_POOL_LOCSCALE shape-preservation)
    with torch.no_grad():
        d = backbone.pool(m, torch.tensor(X[:2], device=dev)).shape[1]
    if cfg.lora:
        m = _wrap_lora(m, cfg); m.to(dev)         # freeze base, train adapters only
    else:
        for p in m.parameters():
            p.requires_grad_(True)
    head = fresh_head(d, nc).to(dev)
    centers = torch.nn.Parameter(torch.randn(nc, d, device=dev) * 0.01)
    opt = torch.optim.Adam(
        [{'params': head.parameters(), 'lr': cfg.lr_head},
         {'params': [centers], 'lr': cfg.lr_head},
         {'params': [p for p in m.parameters() if p.requires_grad], 'lr': cfg.lr_back}])
    ce = torch.nn.CrossEntropyLoss()
    idx_by_c = [np.where(yl == c)[0] for c in range(nc)]
    rng = np.random.default_rng(seed)
    m.train(); head.train()
    if cfg.log_every:
        print(f"[finetune] device={dev} | {cfg.steps} steps batch={cfg.batch} "
              f"| {n:,} ctx | CE+λsc{cfg.lambda_supcon}+λpr{cfg.lambda_proto}", flush=True)
    for step in range(cfg.steps):
        b = (_balanced_batch(idx_by_c, cfg.batch, rng) if cfg.balanced
             else rng.choice(n, size=min(cfg.batch, n), replace=False))
        z = backbone.pool(m, torch.tensor(X[b], device=dev))
        yb = Y[b]
        loss = ce(head(z), yb)
        if cfg.lambda_supcon > 0:
            loss = loss + cfg.lambda_supcon * _supcon(z, yb, cfg.tau)
        if cfg.lambda_proto > 0:
            proto_logits = (F.normalize(z, dim=1)
                            @ F.normalize(centers, dim=1).t()) / cfg.proto_tau
            loss = loss + cfg.lambda_proto * ce(proto_logits, yb)
        opt.zero_grad(); loss.backward(); opt.step()
        if cfg.log_every and (step % cfg.log_every == 0 or step == cfg.steps - 1):
            print(f"[finetune] step {step+1}/{cfg.steps}  loss={float(loss):.4f}",
                  flush=True)
    m = _finalize(m, cfg); head.to('cpu')         # merge LoRA + save/embed from cpu
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
