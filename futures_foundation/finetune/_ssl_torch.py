"""Temporal contrastive SSL of the Mantis backbone — GPU-maximized (Colab).

Continues pretraining the Mantis-8M encoder on raw OHLCV windows with a SimCLR-style
NT-Xent objective: two augmented views of the SAME window are pulled together, all
other windows in the (large) batch are pushed apart. The backbone learns futures
price-action structure (momentum / volume / volatility shifts) that downstream
classifier finetuning starts from.

GPU-maximizing choices:
  * ALL bars resident on the GPU once; each step gathers a big batch of windows by
    index (no host<->device copies, no DataLoader workers).
  * Augmentations run as vectorized GPU tensor ops (no CPU bottleneck).
  * Large contrastive batch (more negatives = better NT-Xent) + CUDA AMP (fp16 +
    GradScaler). AMP is enabled on CUDA only (MPS bf16/fp16 autocast is slow).
  * optional torch.compile of the encoder.

Output: the adapted encoder state_dict saved to a checkpoint (Drive on Colab), used
as the init for supervised finetuning (build_model(..., backbone_ckpt=...)).

torch imports live here only (kept out of the torch-free orchestrator + tests).
"""
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------- model
class SSLNetwork(nn.Module):
    """Mantis encoder + channel adapter + projection head. forward -> L2-normalized
    projection z [B, proj_dim] for the contrastive loss."""

    def __init__(self, C=5, new_channels=8, proj_dim=128, hidden_dim=256,
                 model_id='paris-noah/Mantis-8M'):
        super().__init__()
        from mantis.architecture import Mantis8M
        from mantis.adapters import LinearChannelCombiner
        self.encoder = Mantis8M.from_pretrained(model_id)
        hidden_dim = getattr(self.encoder, 'hidden_dim', hidden_dim)
        self.new_c = min(new_channels, C)
        self.adapter = LinearChannelCombiner(num_channels=C, new_num_channels=self.new_c)
        emb_dim = hidden_dim * self.new_c
        self.proj = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.BatchNorm1d(emb_dim),
                                  nn.ReLU(inplace=True), nn.Linear(emb_dim, proj_dim))

    def embed(self, x):                                   # [B, C, seq] -> [B, new_c*hidden]
        a = self.adapter(x)
        return torch.cat([self.encoder(a[:, [i], :]) for i in range(a.shape[1])], dim=-1)

    def forward(self, x):
        return F.normalize(self.proj(self.embed(x)), dim=1)


def build_ssl_net(C=5, new_channels=8, proj_dim=128, device='cpu',
                  model_id='paris-noah/Mantis-8M', backbone_ckpt=None):
    net = SSLNetwork(C=C, new_channels=new_channels, proj_dim=proj_dim, model_id=model_id)
    if backbone_ckpt:                                    # resume / continue adaptation
        net.encoder.load_state_dict(torch.load(backbone_ckpt, map_location='cpu'))
    return net.to(device)


@torch.no_grad()
def embed_encoder(big, starts, seq, *, ckpt=None, model_id='paris-noah/Mantis-8M',
                  device=None, batch=512, max_windows=20000, seed=0):
    """Frozen ENCODER-ONLY embeddings of clean (un-augmented, per-window z-scored)
    windows — the quantity that transfers downstream via backbone_ckpt. Each OHLCV
    channel is encoded independently and concatenated -> [M, C*hidden]. ckpt=None ->
    vanilla Mantis (the probe baseline); ckpt=path -> the SSL-adapted encoder.
    """
    from mantis.architecture import Mantis8M
    dev = device or ('cuda' if torch.cuda.is_available()
                     else 'mps' if torch.backends.mps.is_available() else 'cpu')
    enc = Mantis8M.from_pretrained(model_id)
    if ckpt:
        enc.load_state_dict(torch.load(ckpt, map_location='cpu'))
    enc = enc.to(dev).eval()
    big_t = torch.as_tensor(np.asarray(big, np.float32), device=dev)
    s = np.asarray(starts, np.int64)
    if len(s) > max_windows:
        s = np.sort(np.random.default_rng(seed).choice(s, max_windows, replace=False))
    s_t = torch.as_tensor(s, device=dev)
    out = []
    for b in range(0, len(s_t), batch):
        win = _gather_batch(big_t, s_t, torch.arange(b, min(b + batch, len(s_t)), device=dev), seq)
        win = _standardize(win)                          # [B, C, seq]
        emb = torch.cat([enc(win[:, [i], :]) for i in range(win.shape[1])], dim=-1)
        out.append(emb.float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 0), np.float32), s


# ----------------------------------------------------------------- augmentations (GPU)
def _two_views(parent, seq, *, max_jitter, resize=(0.7, 1.0), jitter=0.05, scale=0.1,
               warp=0.1, gen=None):
    """parent [B, C, parent_len] -> (v1, v2) each [B, C, seq], independently augmented.
    The two views differ by a random TIME OFFSET (the temporal-contrastive positive
    pair) plus resize / jitter / scale / magnitude-warp."""
    return (_one_view(parent, seq, max_jitter, resize, jitter, scale, warp, gen),
            _one_view(parent, seq, max_jitter, resize, jitter, scale, warp, gen))


def _one_view(parent, seq, max_jitter, resize, jitter, scale, warp, gen):
    B, C, P = parent.shape
    dev = parent.device
    # (1) random time offset per sample -> crop length seq (the "slight time shift")
    off = torch.randint(0, max(1, max_jitter + 1), (B,), device=dev, generator=gen)
    cols = off[:, None] + torch.arange(seq, device=dev)[None, :]          # [B, seq]
    idx = cols[:, None, :].expand(B, C, seq)
    x = torch.gather(parent, 2, idx)                                      # [B, C, seq]
    # (2) resize: crop a sub-length then interpolate back to seq ("different window size")
    if resize is not None:
        lo, hi = resize
        s = float(torch.empty(1).uniform_(lo, hi).item())
        L = max(4, int(round(seq * s)))
        if L < seq:
            st = (seq - L) // 2
            x = F.interpolate(x[:, :, st:st + L], size=seq, mode='linear',
                              align_corners=False)
    # (3) magnitude warp: smooth random per-sample gain curve along time
    if warp and warp > 0:
        knots = torch.randn(B, 1, 5, device=dev, generator=gen) * warp + 1.0
        curve = F.interpolate(knots, size=seq, mode='linear', align_corners=False)
        x = x * curve
    # (4) per-sample/channel scale + (5) additive jitter
    if scale and scale > 0:
        x = x * (1.0 + torch.randn(B, C, 1, device=dev, generator=gen) * scale)
    if jitter and jitter > 0:
        x = x + torch.randn_like(x) * (jitter * x.std(dim=2, keepdim=True))
    return _standardize(x)


def _standardize(x):                                     # per-window per-channel z-score
    m = x.mean(dim=2, keepdim=True)
    s = x.std(dim=2, keepdim=True)
    return (x - m) / (s + 1e-6)


# --------------------------------------------------------------------------- loss/metrics
def nt_xent(z1, z2, temp=0.2):
    """SimCLR NT-Xent over a batch. z1,z2: [B, D] L2-normalized."""
    B = z1.shape[0]
    z = torch.cat([z1, z2], 0)                           # [2B, D]
    sim = (z @ z.t()) / temp                             # [2B, 2B]
    sim.fill_diagonal_(-1e9)                             # mask self
    targets = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z.device)
    return F.cross_entropy(sim, targets)


@torch.no_grad()
def collapse_metrics(z1, z2):
    """Representation-collapse diagnostics (the contrastive 'overfit'/degeneracy):
    embed std (->0 = collapse), alignment (positives close, lower=better), uniformity
    (Wang&Isola, lower=more spread). Returns dict of floats."""
    z = torch.cat([z1, z2], 0)
    std = z.std(0).mean().item()
    align = (z1 - z2).pow(2).sum(1).mean().item()
    d = torch.pdist(z).pow(2)
    unif = d.mul(-2).exp().mean().clamp_min(1e-12).log().item() if len(d) else 0.0
    return {'std': std, 'align': align, 'uniformity': unif}


# ------------------------------------------------------------------------------- train
def _gather_batch(big, starts, b_idx, parent_len):
    """big [T, C] -> parent windows [B, C, parent_len] for start positions starts[b_idx]."""
    s = starts[b_idx]                                    # [B]
    rows = s[:, None] + torch.arange(parent_len, device=big.device)[None, :]   # [B,P]
    return big[rows].permute(0, 2, 1).contiguous()       # [B, C, parent_len]


def train_ssl(big, train_starts, val_starts, *, seq=64, max_jitter=8, new_channels=8,
              proj_dim=128, temp=0.2, epochs=50, steps_per_epoch=200, batch=512, lr=1e-4,
              weight_decay=0.05, patience=8, device=None, model_id='paris-noah/Mantis-8M',
              backbone_ckpt=None, compile_model=False, control='real', seed=0,
              aug=None, verbose=True):
    """Contrastive-pretrain the Mantis encoder on OHLCV windows.

    big: float32 [T, C] all bars concatenated. train_starts/val_starts: int parent-window
    start positions. control: 'real' | 'shuffle' (shuffle time within each window ->
    destroys temporal structure) | 'random' (gaussian-noise windows) — the SSL analogue
    of the WF REAL/SHUFFLE/RANDOM controls. Returns (best_encoder_state, history).
    """
    dev = device or ('cuda' if torch.cuda.is_available()
                     else 'mps' if torch.backends.mps.is_available() else 'cpu')
    torch.manual_seed(seed)
    gen = torch.Generator(device=dev); gen.manual_seed(seed)
    C = int(big.shape[1]); parent_len = seq + max_jitter
    aug = aug or {}
    use_amp = (dev == 'cuda')                            # AMP on CUDA only (MPS amp is slow)

    big_t = torch.as_tensor(np.asarray(big, np.float32), device=dev)
    tr = torch.as_tensor(np.asarray(train_starts, np.int64), device=dev)
    va = torch.as_tensor(np.asarray(val_starts, np.int64), device=dev)

    net = build_ssl_net(C=C, new_channels=new_channels, proj_dim=proj_dim, device=dev,
                        model_id=model_id, backbone_ckpt=backbone_ckpt)
    if compile_model and hasattr(torch, 'compile'):
        net = torch.compile(net)
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],
                            lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    amp_ctx = (lambda: torch.autocast('cuda', dtype=torch.float16)) if use_amp \
        else _nullctx

    def _prep(parent):
        if control == 'shuffle':                         # destroy temporal order within window
            perm = torch.argsort(torch.rand(parent.shape[0], parent.shape[2], device=dev), 1)
            parent = torch.gather(parent, 2, perm[:, None, :].expand_as(parent))
        elif control == 'random':
            parent = torch.randn_like(parent)
        v1, v2 = _two_views(parent, seq, max_jitter=max_jitter, gen=gen, **aug)
        return v1, v2

    def run_batch(starts):
        b_idx = torch.randint(0, len(starts), (batch,), device=dev, generator=gen)
        parent = _gather_batch(big_t, starts, b_idx, parent_len)
        v1, v2 = _prep(parent)
        with amp_ctx():
            z1, z2 = net(v1), net(v2)
            loss = nt_xent(z1, z2, temp)
        return loss, z1, z2

    @torch.no_grad()
    def val_loss():
        net.eval(); tot = 0.0; nb = max(1, len(va) // batch); nb = min(nb, 20)
        last = None
        for _ in range(nb):
            loss, z1, z2 = run_batch(va); tot += loss.item(); last = (z1, z2)
        net.train()
        return tot / nb, collapse_metrics(*last)

    best, best_state, bad, history = 1e18, None, 0, []
    for ep in range(epochs):
        net.train(); tr_tot = 0.0
        for _ in range(steps_per_epoch):
            opt.zero_grad(set_to_none=True)
            loss, _, _ = run_batch(tr)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tr_tot += loss.item()
        sched.step()
        if dev == 'cuda':
            torch.cuda.empty_cache()
        vloss, cm = val_loss()
        history.append({'epoch': ep, 'train_loss': tr_tot / steps_per_epoch,
                        'val_loss': vloss, **cm})
        improved = vloss < best - 1e-4
        if improved:
            best, bad = vloss, 0
            enc = net.encoder if not hasattr(net, '_orig_mod') else net._orig_mod.encoder
            best_state = {k: v.detach().cpu().clone() for k, v in enc.state_dict().items()}
        else:
            bad += 1
        if verbose:
            print(f"  ep{ep:>3} train={tr_tot / steps_per_epoch:.4f} val={vloss:.4f} "
                  f"std={cm['std']:.4f} align={cm['align']:.3f} unif={cm['uniformity']:.3f}"
                  f"{'  *' if improved else ''}", flush=True)
        if bad >= patience:
            break
    return best_state, history


class _nullctx:
    def __enter__(self): return None
    def __exit__(self, *a): return False
