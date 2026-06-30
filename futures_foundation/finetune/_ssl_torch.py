"""Masked-modeling SSL of the Mantis backbone — "BERT for futures", GPU-maximized (Colab).

Continues pretraining the Mantis-8M encoder on raw OHLCV windows with a BERT-style masked
objective: a fraction of bars are masked (replaced with noise) and the model reconstructs
them from the surrounding context. To reconstruct a masked bar the encoder MUST model
regime/volatility (bar size), temporal dynamics (trend continuation) and cross-channel
coupling — i.e. the market-context the downstream buy/sell classifier needs. Unlike
contrastive instance-discrimination, masked modeling is NOT gameable by a distributional
shortcut, and its REAL/SHUFFLE/RANDOM controls are meaningful (real has predictable context
to reconstruct from; time-scrambled and noise windows do not).

GPU-maximizing choices:
  * ALL bars resident on the GPU once; each step gathers a big batch of windows by index
    (no host<->device copies, no DataLoader workers).
  * Masking + standardization run as vectorized GPU tensor ops.
  * Large batch + CUDA AMP (fp16). AMP is enabled on CUDA only (MPS amp is slow).
  * optional torch.compile of the network.

Output: the adapted encoder state_dict saved to a checkpoint (Drive on Colab), used as the
init for supervised finetuning (build_model(..., backbone_ckpt=...)).

torch imports live here only (kept out of the torch-free orchestrator + tests).
"""
import os

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def _enc(encoder, x1):
    """Encode one channel [B,1,L] -> [B, hidden], interpolating the window to Mantis's
    native seq_len (512) first so it ALWAYS sees its pretrained patch size (patch_size =
    seq_len/num_patches = 16). Without this, a short window gives tiny patches (e.g. seq=64
    -> patch 2, off-distribution; seq=32 -> patch 1 -> per-patch std=0 -> NaN)."""
    L = int(getattr(encoder, 'seq_len', 512))
    if x1.shape[-1] != L:
        x1 = F.interpolate(x1, size=L, mode='linear', align_corners=False)
    return encoder(x1)


def _standardize(x):                                     # per-window per-channel z-score
    m = x.mean(dim=2, keepdim=True)
    s = x.std(dim=2, keepdim=True)
    return (x - m) / (s + 1e-6)


def _time_shuffle(x):
    """Permute the time axis independently per sample -> destroys temporal order, keeps the
    exact value set. Used for the SHUFFLE control (reconstruction should get much worse)."""
    B, C, T = x.shape
    perm = torch.argsort(torch.rand(B, T, device=x.device), 1)
    return torch.gather(x, 2, perm[:, None, :].expand(B, C, T))


def _gather_batch(big, starts, b_idx, length):
    """big [T, C] -> windows [B, C, length] for the start positions starts[b_idx]."""
    s = starts[b_idx]                                    # [B]
    rows = s[:, None] + torch.arange(length, device=big.device)[None, :]   # [B, length]
    return big[rows].permute(0, 2, 1).contiguous()       # [B, C, length]


# ----------------------------------------------------------------- frozen embedding (probe)
@torch.no_grad()
def embed_encoder(big, starts, seq, *, ckpt=None, model_id='paris-noah/Mantis-8M',
                  device=None, batch=512, max_windows=20000, seed=0):
    """Frozen ENCODER-ONLY embeddings of clean (per-window z-scored) windows — the quantity
    that transfers downstream via backbone_ckpt. Each OHLCV channel is encoded independently
    (interpolated to Mantis's native length) and concatenated -> [M, C*hidden]. ckpt=None ->
    vanilla Mantis (the probe baseline); ckpt=path -> the masked-adapted encoder."""
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
        emb = torch.cat([_enc(enc, win[:, [i], :]) for i in range(win.shape[1])], dim=-1)
        out.append(emb.float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 0), np.float32), s


@torch.no_grad()
def embed_windows(windows, *, ckpt=None, model_id='paris-noah/Mantis-8M', device=None,
                  batch=512):
    """Frozen ENCODER-ONLY embeddings of pre-extracted windows [N, C, seq] -> [N, C*hidden].
    Each channel is per-window standardized, interpolated to Mantis's native length, encoded,
    and concatenated. ckpt=None -> vanilla Mantis; ckpt=path -> the masked-adapted encoder.
    This is the head-only/cached downstream primitive: backbone frozen, embed ONCE, then a
    cheap head trains on the cache."""
    os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')   # any unsupported op -> CPU
    from mantis.architecture import Mantis8M
    dev = device or ('cuda' if torch.cuda.is_available()
                     else 'mps' if torch.backends.mps.is_available() else 'cpu')
    enc = Mantis8M.from_pretrained(model_id)
    if ckpt:
        enc.load_state_dict(torch.load(ckpt, map_location='cpu'))
    enc = enc.to(dev).eval()
    X = torch.as_tensor(np.asarray(windows, np.float32))
    out = []
    for b in range(0, len(X), batch):
        w = _standardize(X[b:b + batch].to(dev))
        emb = torch.cat([_enc(enc, w[:, [i], :]) for i in range(w.shape[1])], dim=-1)
        out.append(emb.float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 0), np.float32)


class _EncoderONNX(nn.Module):
    """ONNX-exportable wrapper that reproduces embed_windows EXACTLY: per-window standardize ->
    per-channel interpolate to native length -> encode -> concat. Input raw window [B,C,seq],
    output embedding [B, C*hidden]. The per-window standardize is baked in so the bot feeds
    RAW OHLCV windows (no external preprocessing for the encoder)."""

    def __init__(self, encoder, C):
        super().__init__()
        self.encoder = encoder
        self.C = int(C)

    def forward(self, w):                                     # [B, C, seq] raw OHLCV
        w = _standardize(w)
        return torch.cat([_enc(self.encoder, w[:, [i], :]) for i in range(self.C)], dim=-1)


def export_encoder_onnx(path, *, ckpt=None, C=5, seq=64,
                        model_id='paris-noah/Mantis-8M', device='cpu'):
    """Export the frozen encoder (standardize+interp+encode) to ONNX: raw window [B,C,seq] ->
    embedding [B, C*hidden]. Matches embed_windows numerically (parity-tested)."""
    from mantis.architecture import Mantis8M
    enc = Mantis8M.from_pretrained(model_id)
    if ckpt:
        enc.load_state_dict(torch.load(ckpt, map_location='cpu'))
    enc = enc.to(device).eval()
    m = _EncoderONNX(enc, C).to(device).eval()
    dummy = torch.randn(2, int(C), int(seq), device=device)   # >1 row so std is well-defined
    # Mantis calls torch.diff internally (aten::diff has no ONNX symbolic) -> swap it for an
    # equivalent slice-subtract during export so the traced graph is exportable. Restored after.
    _orig_diff = torch.diff

    def _diff_traceable(x, n=1, dim=-1, *, axis=None, prepend=None, append=None):
        d = axis if axis is not None else dim
        for _ in range(int(n)):
            x = x.narrow(d, 1, x.size(d) - 1) - x.narrow(d, 0, x.size(d) - 1)
        return x
    torch.diff = _diff_traceable
    try:
        torch.onnx.export(m, dummy, path, input_names=['window'], output_names=['embedding'],
                          dynamic_axes={'window': {0: 'batch'}, 'embedding': {0: 'batch'}},
                          opset_version=17, dynamo=False)      # legacy tracer (dynamo chokes on Mantis)
    finally:
        torch.diff = _orig_diff
    return path


# ============================================================ MASKED MODELING (BERT pretext)
class MaskNetwork(nn.Module):
    """Mantis encoder + channel adapter + a light reconstruction decoder. Masked OHLCV bars
    go in; the decoder reconstructs the full (standardized) window from the pooled embedding.
    To reconstruct a masked bar the encoder MUST model regime/vol (bar size), temporal
    dynamics (trend continuation) and cross-channel coupling — i.e. the market-context the
    downstream classifier needs."""

    def __init__(self, C=5, new_channels=8, seq=64, model_id='paris-noah/Mantis-8M'):
        super().__init__()
        from mantis.architecture import Mantis8M
        from mantis.adapters import LinearChannelCombiner
        self.encoder = Mantis8M.from_pretrained(model_id)
        hidden = getattr(self.encoder, 'hidden_dim', 256)
        self.new_c = min(new_channels, C)
        self.adapter = LinearChannelCombiner(num_channels=C, new_num_channels=self.new_c)
        self.C, self.seq = C, seq
        emb = hidden * self.new_c
        self.decoder = nn.Sequential(nn.Linear(emb, emb), nn.GELU(), nn.Linear(emb, C * seq))

    def embed(self, x):                                   # [B, C, seq] -> [B, new_c*hidden]
        a = self.adapter(x)
        return torch.cat([_enc(self.encoder, a[:, [i], :]) for i in range(a.shape[1])], dim=-1)

    def forward(self, x):                                 # masked [B,C,seq] -> recon [B,C,seq]
        return self.decoder(self.embed(x)).view(-1, self.C, self.seq)


def train_ssl_mask(big, train_starts, val_starts, *, seq=64, new_channels=8, mask_ratio=0.4,
                   epochs=60, steps_per_epoch=200, batch=512, lr=1e-4, weight_decay=0.05,
                   patience=8, device=None, model_id='paris-noah/Mantis-8M', backbone_ckpt=None,
                   compile_model=False, control='real', seed=0, amp_dtype='fp16',
                   verbose=True, **_ignore):
    """BERT-style masked modeling: mask a fraction of bars, reconstruct them from context
    (MSE on masked positions). Returns (best_encoder_state, history).

    The REAL/SHUFFLE/RANDOM controls are MEANINGFUL: REAL reconstructs from temporal context,
    SHUFFLE (time-scrambled) and RANDOM (noise) have no predictable context -> their val MSE
    should be clearly WORSE. history carries 'val_loss' (recon MSE) + 'std' (embedding std,
    for the collapse guard)."""
    os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
    dev = device or ('cuda' if torch.cuda.is_available()
                     else 'mps' if torch.backends.mps.is_available() else 'cpu')
    torch.manual_seed(seed); gen = torch.Generator(device=dev); gen.manual_seed(seed)
    C = int(big.shape[1])
    use_amp = (dev == 'cuda')
    _adt = torch.float16 if str(amp_dtype).lower() in ('fp16', 'float16') else torch.bfloat16
    amp_ctx = (lambda: torch.autocast('cuda', dtype=_adt)) if use_amp else (lambda: _nullctx())

    big_t = torch.as_tensor(np.asarray(big, np.float32), device=dev)
    tr = torch.as_tensor(np.asarray(train_starts, np.int64), device=dev)
    va = torch.as_tensor(np.asarray(val_starts, np.int64), device=dev)

    net = MaskNetwork(C=C, new_channels=new_channels, seq=seq, model_id=model_id).to(dev)
    if backbone_ckpt:
        net.encoder.load_state_dict(torch.load(backbone_ckpt, map_location='cpu'))
    if compile_model and hasattr(torch, 'compile'):
        net = torch.compile(net)
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],
                            lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    def _win(starts):
        b_idx = torch.randint(0, len(starts), (batch,), device=dev, generator=gen)
        w = _gather_batch(big_t, starts, b_idx, seq)         # [B,C,seq] raw
        if control == 'shuffle':
            w = _time_shuffle(w)
        elif control == 'random':
            w = torch.randn_like(w)
        return _standardize(w)                               # per-window z-score

    def _recon_loss(w):
        m = torch.rand(w.shape[0], seq, device=dev, generator=gen) < mask_ratio   # [B,seq]
        none = ~m.any(1); m[none, 0] = True                  # >=1 masked bar per sample
        me = m[:, None, :].expand_as(w)
        corrupted = torch.where(me, torch.randn_like(w), w)  # fill masked bars w/ noise so
        recon = net(corrupted)                               # patches keep variance (Mantis
        diff = (recon - w) ** 2                              # instance-norm would /0 on zeros)
        return diff[me].mean()                               # MSE on masked positions only

    @torch.no_grad()
    def val_eval():
        net.eval(); tot = 0.0; nb = min(20, max(1, len(va) // batch))
        for _ in range(nb):
            with amp_ctx():
                tot += float(_recon_loss(_win(va)))
        estd = float(net.embed(_win(va)).std(0).mean())
        net.train()
        return tot / nb, estd

    best, best_state, bad, history = 1e18, None, 0, []
    for ep in range(epochs):
        net.train(); tr_tot = 0.0
        for _ in range(steps_per_epoch):
            opt.zero_grad(set_to_none=True)
            with amp_ctx():
                loss = _recon_loss(_win(tr))
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tr_tot += float(loss.detach())
        sched.step()
        if dev == 'cuda':
            torch.cuda.empty_cache()
        vloss, estd = val_eval()
        history.append({'epoch': ep, 'train_loss': tr_tot / steps_per_epoch,
                        'val_loss': vloss, 'std': estd})
        improved = vloss < best - 1e-5
        if improved:
            best, bad = vloss, 0
            enc = net.encoder if not hasattr(net, '_orig_mod') else net._orig_mod.encoder
            best_state = {k: v.detach().cpu().clone() for k, v in enc.state_dict().items()}
        else:
            bad += 1
        if verbose:
            print(f"  ep{ep:>3} train={tr_tot / steps_per_epoch:.4f} val={vloss:.4f} "
                  f"emb_std={estd:.4f}{'  *' if improved else ''}", flush=True)
        if bad >= patience:
            break
    return best_state, history


# ====================================== SEQ2SEQ FORECASTING (causal pretext, SSL stage 2)
def _standardize_ctx(ctx, fut, clamp=10.0):
    """z-score the CONTEXT window per channel, apply the SAME shift/scale to the future bars.
    Standardizing the future by its own stats would leak its mean/scale into the target —
    using the context's stats keeps the forecast target strictly causal.

    CLAMP (anti-blowup): a FLAT/compressed context window has near-zero std, so a real future
    move divided by that tiny std explodes to astronomical standardized values -> exploding
    gradients -> training diverges (train loss 5e4, val/persist blow up). Compressed windows
    before a breakout are exactly the setups we care about, so this is the common case, not an
    edge case. Clamping to +/-clamp bounds it: a big move out of compression saturates at
    'large move' (direction preserved, magnitude capped) instead of detonating the loss."""
    m = ctx.mean(dim=2, keepdim=True)
    s = ctx.std(dim=2, keepdim=True) + 1e-6
    cs = ((ctx - m) / s).clamp(-clamp, clamp)
    fs = ((fut - m) / s).clamp(-clamp, clamp)
    return cs, fs


class ForecastNetwork(nn.Module):
    """Mantis encoder + channel adapter + a forecast decoder. A CONTEXT window of past bars is
    encoded; the decoder predicts the NEXT `horizon` bars (full OHLCV) from the pooled
    embedding. To forecast the future the encoder MUST model forward dynamics — trend
    continuation, momentum, volatility persistence — i.e. the trend-prediction the downstream
    buy/sell classifier needs. Warm-start the encoder from the masked-SSL ckpt (stage 1)."""

    def __init__(self, C=5, new_channels=8, seq=64, horizon=16, model_id='paris-noah/Mantis-8M'):
        super().__init__()
        from mantis.architecture import Mantis8M
        from mantis.adapters import LinearChannelCombiner
        self.encoder = Mantis8M.from_pretrained(model_id)
        hidden = getattr(self.encoder, 'hidden_dim', 256)
        self.new_c = min(new_channels, C)
        self.adapter = LinearChannelCombiner(num_channels=C, new_num_channels=self.new_c)
        self.C, self.seq, self.horizon = C, seq, horizon
        emb = hidden * self.new_c
        self.decoder = nn.Sequential(nn.Linear(emb, emb), nn.GELU(), nn.Linear(emb, C * horizon))

    def embed(self, x):                                   # [B,C,seq] context -> [B, new_c*hidden]
        a = self.adapter(x)
        return torch.cat([_enc(self.encoder, a[:, [i], :]) for i in range(a.shape[1])], dim=-1)

    def forward(self, ctx):                               # [B,C,seq] -> forecast [B,C,horizon]
        return self.decoder(self.embed(ctx)).view(-1, self.C, self.horizon)


def train_ssl_forecast(big, train_starts, val_starts, *, seq=64, horizon=16, new_channels=8,
                       epochs=60, steps_per_epoch=200, batch=512, lr=1e-4, weight_decay=0.05,
                       patience=8, device=None, model_id='paris-noah/Mantis-8M',
                       backbone_ckpt=None, compile_model=False, control='real', seed=0,
                       amp_dtype='fp16', grad_clip=1.0, clamp=10.0, channel_weights=None,
                       verbose=True, **_ignore):
    """Causal seq2seq forecasting: encode `seq` context bars, predict the next `horizon` bars
    (MSE, context-standardized). Returns (best_encoder_state, history). Warm-start the encoder
    from the masked-SSL stage-1 ckpt via backbone_ckpt.

    channel_weights: optional length-C per-channel loss weights (O,H,L,C,V order), e.g.
    [1,1,1,2,0] = price-path (emphasize close, ignore volume). None = equal (default; unchanged).
    The SAME weights apply to the persistence baseline, so 'skill' stays consistent: with
    volume zeroed, skill becomes pure PRICE skill. This focuses capacity on the trend-relevant
    price path instead of diluting it with near-unpredictable volume.

    REAL/SHUFFLE/RANDOM controls are MEANINGFUL: only REAL has a future that follows from its
    past — SHUFFLE (time-scrambled) and RANDOM (noise) have no predictable continuation, so
    their val MSE should be clearly WORSE. history carries 'val_loss' + 'std' (collapse guard)."""
    os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
    dev = device or ('cuda' if torch.cuda.is_available()
                     else 'mps' if torch.backends.mps.is_available() else 'cpu')
    torch.manual_seed(seed); gen = torch.Generator(device=dev); gen.manual_seed(seed)
    C = int(big.shape[1])
    parent = seq + horizon
    use_amp = (dev == 'cuda')
    _adt = torch.float16 if str(amp_dtype).lower() in ('fp16', 'float16') else torch.bfloat16
    amp_ctx = (lambda: torch.autocast('cuda', dtype=_adt)) if use_amp else (lambda: _nullctx())

    big_t = torch.as_tensor(np.asarray(big, np.float32), device=dev)
    tr = torch.as_tensor(np.asarray(train_starts, np.int64), device=dev)
    va = torch.as_tensor(np.asarray(val_starts, np.int64), device=dev)

    net = ForecastNetwork(C=C, new_channels=new_channels, seq=seq, horizon=horizon,
                          model_id=model_id).to(dev)
    if backbone_ckpt:                                    # warm-start from stage-1 masked SSL
        net.encoder.load_state_dict(torch.load(backbone_ckpt, map_location='cpu'))
    if compile_model and hasattr(torch, 'compile'):
        net = torch.compile(net)
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],
                            lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    # per-channel loss weights (O,H,L,C,V). None -> equal. Applied to BOTH forecast and persist
    # so 'skill' stays a like-for-like ratio under any weighting (volume=0 -> pure price skill).
    w_t = (None if channel_weights is None else
           torch.as_tensor(np.asarray(channel_weights, np.float32), device=dev).view(1, C, 1))

    def _ctx_fut(starts):
        b_idx = torch.randint(0, len(starts), (batch,), device=dev, generator=gen)
        w = _gather_batch(big_t, starts, b_idx, parent)      # [B,C,seq+horizon] raw
        if control == 'shuffle':
            w = _time_shuffle(w)                             # no causal continuation survives
        elif control == 'random':
            w = torch.randn_like(w)
        return _standardize_ctx(w[:, :, :seq], w[:, :, seq:], clamp=clamp)

    def _target(ctx, fut):
        """Forecast the forward PATH relative to 'now' = future bars minus the last context bar.
        This DELETES the persistence shortcut by construction: 'copy the last bar' (the trivial
        MSE minimizer on absolute levels) becomes 'predict zero' here, which the loss punishes —
        so the encoder is FORCED to learn signed forward movement (direction + magnitude), the
        trend content we actually want. ctx/fut are already context-z-scored."""
        return fut - ctx[:, :, -1:]                          # [B,C,horizon] forward delta path

    def _wmean(se):                                          # (weighted) mean of a squared-error tensor
        return (se * w_t).mean() if w_t is not None else se.mean()

    def _fc_loss(ctx, fut):
        return _wmean((net(ctx) - _target(ctx, fut)) ** 2)   # (weighted) MSE on the forward DELTA path

    def _persist_loss(ctx, fut):
        return _wmean(_target(ctx, fut) ** 2)                # predict-zero = copy-last-bar baseline

    @torch.no_grad()
    def val_eval():
        net.eval(); tot = 0.0; ptot = 0.0; nb = min(20, max(1, len(va) // batch))
        for _ in range(nb):
            ctx, fut = _ctx_fut(va)
            with amp_ctx():
                tot += float(_fc_loss(ctx, fut))
            ptot += float(_persist_loss(ctx, fut))
        estd = float(net.embed(_ctx_fut(va)[0]).std(0).mean())
        net.train()
        return tot / nb, ptot / nb, estd

    best, best_state, bad, history = 1e18, None, 0, []
    for ep in range(epochs):
        net.train(); tr_tot = 0.0
        for _ in range(steps_per_epoch):
            opt.zero_grad(set_to_none=True)
            with amp_ctx():
                loss = _fc_loss(*_ctx_fut(tr))
            scaler.scale(loss).backward()
            scaler.unscale_(opt)                              # unscale before clipping (AMP)
            torch.nn.utils.clip_grad_norm_(                   # stability: cap gradient norm
                [p for p in net.parameters() if p.requires_grad], grad_clip)
            scaler.step(opt); scaler.update()
            tr_tot += float(loss.detach())
        sched.step()
        if dev == 'cuda':
            torch.cuda.empty_cache()
        vloss, ploss, estd = val_eval()
        skill = float(1.0 - vloss / ploss) if ploss > 1e-12 else 0.0   # >0 => beats copy-last-bar
        history.append({'epoch': ep, 'train_loss': tr_tot / steps_per_epoch,
                        'val_loss': vloss, 'persist_loss': ploss, 'skill': skill, 'std': estd})
        improved = vloss < best - 1e-5
        if improved:
            best, bad = vloss, 0
            enc = net.encoder if not hasattr(net, '_orig_mod') else net._orig_mod.encoder
            best_state = {k: v.detach().cpu().clone() for k, v in enc.state_dict().items()}
        else:
            bad += 1
        if verbose:
            print(f"  ep{ep:>3} train={tr_tot / steps_per_epoch:.4f} val={vloss:.4f} "
                  f"persist={ploss:.4f} skill={skill:+.3f} emb_std={estd:.4f}"
                  f"{'  *' if improved else ''}", flush=True)
        if bad >= patience:
            break
    return best_state, history


class _nullctx:
    def __enter__(self): return None
    def __exit__(self, *a): return False
