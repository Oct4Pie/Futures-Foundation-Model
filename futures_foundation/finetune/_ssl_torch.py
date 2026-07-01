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


def _apply_control(x, control):
    """Corrupt ONLY the model INPUT per the apples-to-apples control: 'shuffle' scrambles the time
    axis, 'random' replaces with noise, else (real) passes through. The target/trend-key is always
    computed from the REAL context by the caller -> real must beat shuffle/random. Shared by every
    pretext trainer (mask/forecast/contrastive) so the control logic lives in one place."""
    if control == 'shuffle':
        return _time_shuffle(x)
    if control == 'random':
        return torch.randn_like(x)
    return x


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
        return _standardize(_apply_control(w, control))      # corrupt input per control, then z-score

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


# ===== SEQ2SEQ (stage 2): multi-horizon, variable-context CANDLE forecasting =====
class MultiHorizonForecastNet(nn.Module):
    """Mantis encoder + channel adapter + a MULTI-HORIZON candle decoder. A variable-length
    CONTEXT of past bars is encoded (interpolated to Mantis's native length, so short/long context
    both work); the decoder predicts the future CANDLE (OHLCV) at EACH horizon in `horizons` (e.g.
    5/10/20/25/50 bars ahead) -> [B, C, n_horizons]. Forecasting near AND far from short AND long
    context forces the encoder to model price-action dynamics at multiple scales — the trend
    understanding the downstream catcher needs. Warm-start from the stage-1 masked-SSL ckpt."""

    def __init__(self, C=5, new_channels=8, horizons=(5, 10, 20, 25, 50),
                 model_id='paris-noah/Mantis-8M'):
        super().__init__()
        from mantis.architecture import Mantis8M
        from mantis.adapters import LinearChannelCombiner
        self.encoder = Mantis8M.from_pretrained(model_id)
        hidden = getattr(self.encoder, 'hidden_dim', 256)
        self.new_c = min(new_channels, C)
        self.adapter = LinearChannelCombiner(num_channels=C, new_num_channels=self.new_c)
        self.C, self.horizons = C, tuple(int(h) for h in horizons)
        self.nH = len(self.horizons)
        emb = hidden * self.new_c
        self.decoder = nn.Sequential(nn.Linear(emb, emb), nn.GELU(), nn.Linear(emb, C * self.nH))

    def embed(self, x):                                   # [B,C,L] context (all OHLCV) -> [B, new_c*hidden]
        a = self.adapter(x)
        return torch.cat([_enc(self.encoder, a[:, [i], :]) for i in range(a.shape[1])], dim=-1)

    def forward(self, ctx):                               # [B,C,L] -> [B, C, n_horizons] (candles)
        return self.decoder(self.embed(ctx)).view(-1, self.C, self.nH)


def train_ssl_forecast(big, train_starts, val_starts, *, horizons=(5, 10, 20, 25, 50),
                       context_lengths=(64, 100, 150, 200), new_channels=8,
                       epochs=60, steps_per_epoch=200, batch=512, lr=1e-4, weight_decay=0.05,
                       patience=8, device=None, model_id='paris-noah/Mantis-8M',
                       backbone_ckpt=None, compile_model=False, control='real', seed=0,
                       amp_dtype='fp16', grad_clip=1.0, clamp=10.0, verbose=True, **_ignore):
    """MULTI-HORIZON, VARIABLE-CONTEXT CANDLE forecasting — the stage-2 pretext. Warm-start the
    encoder from the stage-1 masked-SSL ckpt (backbone_ckpt). Returns (best_encoder_state, history).

    Each step: sample a context length L from `context_lengths` (short↔long) + encode it
    (interpolated to Mantis's native length -> scale-invariance), and predict the future CANDLE
    (OHLCV) at every horizon in `horizons` (near↔far, e.g. 5/10/20/25/50 bars ahead). The target is
    the future candle CONTEXT-STANDARDIZED (per-channel z-score by the context's own mean/std — no
    ATR/R, no cross-instrument leak) as a move FROM 'now'. Forecasting near+far from short+long
    context forces multi-scale price-action understanding — the trend signal.

    Anti-shortcut: target = move FROM now, so 'copy now' == predict-zero (punished) -> the encoder
    must learn signed movement. clamp bounds flat-context blow-ups; grad_clip stabilizes.
    APPLES-TO-APPLES controls: target/persist are ALWAYS real; only the model's INPUT context is
    corrupted (shuffle=scramble time order, random=noise) -> real>shuffle>random~0 on skill.
    All channels weighted equally — the model uses/predicts all OHLCV (incl volume)."""
    os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
    dev = device or ('cuda' if torch.cuda.is_available()
                     else 'mps' if torch.backends.mps.is_available() else 'cpu')
    torch.manual_seed(seed); gen = torch.Generator(device=dev); gen.manual_seed(seed)
    C = int(big.shape[1])
    hlist = [int(h) for h in horizons]
    clens = [int(x) for x in context_lengths]
    max_ctx, h_max = max(clens), max(hlist)
    parent = max_ctx + h_max
    h_off = torch.as_tensor([h - 1 for h in hlist], dtype=torch.long, device=dev)   # bar offsets in fut
    clens_t = torch.as_tensor(clens, dtype=torch.long, device=dev)
    use_amp = (dev == 'cuda')
    _adt = torch.float16 if str(amp_dtype).lower() in ('fp16', 'float16') else torch.bfloat16
    amp_ctx = (lambda: torch.autocast('cuda', dtype=_adt)) if use_amp else (lambda: _nullctx())

    big_t = torch.as_tensor(np.asarray(big, np.float32), device=dev)
    tr = torch.as_tensor(np.asarray(train_starts, np.int64), device=dev)
    va = torch.as_tensor(np.asarray(val_starts, np.int64), device=dev)

    net = MultiHorizonForecastNet(C=C, new_channels=new_channels, horizons=hlist,
                                  model_id=model_id).to(dev)
    if backbone_ckpt:                                    # warm-start from stage-1 masked SSL
        net.encoder.load_state_dict(torch.load(backbone_ckpt, map_location='cpu'))
    if compile_model and hasattr(torch, 'compile'):
        net = torch.compile(net)
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],
                            lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    def _batch(starts):
        """(model_ctx [B,C,L], target [B,C,nH]) for a batch. Sample ONE context length L this batch
        (variable context); TARGET = future CANDLE (OHLCV) at each horizon, CONTEXT-STANDARDIZED
        (per-channel z-score by the context's mean/std) as a move FROM now. Corrupt only the
        model's INPUT per control (apples-to-apples)."""
        b_idx = torch.randint(0, len(starts), (batch,), device=dev, generator=gen)
        w = _gather_batch(big_t, starts, b_idx, parent)      # [B,C,max_ctx+h_max] REAL raw
        L = int(clens_t[torch.randint(0, len(clens_t), (1,), device=dev, generator=gen)].item())
        ctx_raw = w[:, :, max_ctx - L:max_ctx]               # [B,C,L] real context ending at 'now'
        fut_raw = w[:, :, max_ctx:]                          # [B,C,h_max] real future candles
        # standardize BOTH by the CONTEXT's stats (causal — no future leak, no ATR/R).
        m = ctx_raw.mean(2, keepdim=True); s = ctx_raw.std(2, keepdim=True) + 1e-6
        cs = ((ctx_raw - m) / s).clamp(-clamp, clamp)        # standardized context (model input)
        fs = ((fut_raw - m) / s).clamp(-clamp, clamp)        # standardized future candles
        # TARGET = future candle at each horizon, as a move FROM 'now' (anti-shortcut: copy-now == 0).
        target = fs[:, :, h_off] - cs[:, :, -1:]             # [B,C,nH]
        model_ctx = _apply_control(cs, control)              # corrupt ONLY the input (apples-to-apples)
        return model_ctx, target

    def _fc_loss(model_ctx, target):
        return ((net(model_ctx) - target) ** 2).mean()       # MSE on the multi-horizon candle targets

    def _persist_loss(target):
        return (target ** 2).mean()                          # predict-zero = 'copy now' (same baseline)

    @torch.no_grad()
    def val_eval():
        net.eval(); tot = 0.0; ptot = 0.0; nb = min(20, max(1, len(va) // batch))
        toth = torch.zeros(len(hlist), device=dev)       # per-horizon accum: is 20/25 learning?
        ptoth = torch.zeros(len(hlist), device=dev)
        for _ in range(nb):
            mc, tg = _batch(va)
            with amp_ctx():
                pred = net(mc)                            # [B,C,nH] — one forward, split by horizon
            se = (pred.float() - tg) ** 2                 # [B,C,nH]
            tot += float(se.mean()); ptot += float((tg ** 2).mean())
            toth += se.mean(dim=(0, 1)); ptoth += (tg ** 2).mean(dim=(0, 1))
        estd = float(net.embed(_batch(va)[0]).std(0).mean())
        net.train()
        skill_h = (1.0 - toth / ptoth.clamp_min(1e-12)).cpu().tolist()   # skill per horizon
        return tot / nb, ptot / nb, estd, skill_h

    best, best_state, bad, history = 1e18, None, 0, []
    for ep in range(epochs):
        net.train(); tr_tot = 0.0
        for _ in range(steps_per_epoch):
            opt.zero_grad(set_to_none=True)
            with amp_ctx():
                loss = _fc_loss(*_batch(tr))
            scaler.scale(loss).backward()
            scaler.unscale_(opt)                              # unscale before clipping (AMP)
            torch.nn.utils.clip_grad_norm_(                   # stability: cap gradient norm
                [p for p in net.parameters() if p.requires_grad], grad_clip)
            scaler.step(opt); scaler.update()
            tr_tot += float(loss.detach())
        sched.step()
        if dev == 'cuda':
            torch.cuda.empty_cache()
        vloss, ploss, estd, skill_h = val_eval()
        skill = float(1.0 - vloss / ploss) if ploss > 1e-12 else 0.0   # >0 => beats copy-last-bar
        history.append({'epoch': ep, 'train_loss': tr_tot / steps_per_epoch,
                        'val_loss': vloss, 'persist_loss': ploss, 'skill': skill,
                        'skill_per_h': dict(zip(hlist, skill_h)), 'std': estd})
        improved = vloss < best - 1e-5
        if improved:
            best, bad = vloss, 0
            enc = net.encoder if not hasattr(net, '_orig_mod') else net._orig_mod.encoder
            best_state = {k: v.detach().cpu().clone() for k, v in enc.state_dict().items()}
        else:
            bad += 1
        if verbose:
            ph = ' '.join(f"h{h}={s:+.2f}" for h, s in zip(hlist, skill_h))   # per-horizon: 20/25?
            print(f"  ep{ep:>3} train={tr_tot / steps_per_epoch:.4f} val={vloss:.4f} "
                  f"persist={ploss:.4f} skill={skill:+.3f} [{ph}] emb_std={estd:.4f}"
                  f"{'  *' if improved else ''}", flush=True)
        if bad >= patience:
            break
    return best_state, history


class _nullctx:
    def __enter__(self): return None
    def __exit__(self, *a): return False


# ============================================================ STAGE-3 (experiment): TREND CONTRASTIVE
# Reuses Mantis's InfoNCE machinery (normalized-sim + temperature + projection head +
# RandomCropResize) but ADAPTS the objective for TREND DETECTION: single-positive instance
# discrimination -> MULTI-POSITIVE (SupCon mechanics) grouped by a SELF-SUPERVISED CAUSAL trend
# key. Same-trend windows become POSITIVES (pulled together), not negatives -> fixes the
# false-negative problem that makes plain InfoNCE fail at grouping. Warm-start from stage-2
# (ctx200). Fallback = stage-2 if it doesn't beat the ctx200 trend-AUC/WR@3R baseline (judged
# offline via scratchpad/trend_learn_analysis.py + ridgeR_2026_wr_lean.py).

def _random_crop_resize(x, crop_max=0.2):
    """Mantis RandomCropResize (multichannel): crop a random 0..crop_max fraction off the time
    axis (random start), interpolate back to the original length. Preserves trend SHAPE while
    varying crop position/scale = the nuisance we want invariance to. One augmented VIEW."""
    B, C, L = x.shape
    cr = float(torch.empty(1, device=x.device).uniform_(0.0, float(crop_max)).item())
    cl = max(8, int(L * (1.0 - cr)))
    start = int(torch.randint(0, L - cl + 1, (1,)).item())
    return F.interpolate(x[:, :, start:start + cl], size=L, mode='linear', align_corners=False)


def _trend_key(ctx, close_ch=3):
    """SELF-SUPERVISED, CAUSAL trend key per window from PAST bars ONLY (no future, no label).
    Signature = (direction bucket, magnitude bucket) of the trailing least-squares slope on the
    CLOSE channel over the window the model sees (ends at 'now'). Windows sharing a key = 'same
    trend character' = POSITIVES for the multi-positive InfoNCE. Leak-free: uses only context."""
    B, C, L = ctx.shape
    close = ctx[:, min(close_ch, C - 1), :]                # [B, L] standardized close
    t = torch.linspace(-1, 1, L, device=ctx.device); tc = t - t.mean()
    slope = (close * tc).sum(1) / (tc * tc).sum().clamp_min(1e-6)     # [B] causal trailing slope
    dz = slope.abs().median().clamp_min(1e-6) * 0.5        # deadzone -> a flat/chop bucket
    dir_b = (torch.sign(slope) * (slope.abs() > dz)).long() + 1       # 0/1/2 = down/flat/up
    mag = slope.abs()
    q1 = torch.quantile(mag, 0.5); q2 = torch.quantile(mag, 0.85)
    mag_b = (mag > q1).long() + (mag > q2).long()          # 0/1/2 = weak/med/strong
    return (dir_b * 3 + mag_b).long()                      # [B] key in [0,9)


def _multi_positive_infonce(z, key, inst, temperature):
    """SupCon-style multi-positive InfoNCE over a 2-view batch. z:[N,D] L2-normalized. key:[N]
    self-supervised trend key; inst:[N] instance id (the two crop views share it). Positive(i) =
    {j!=i : key[j]==key[i] OR inst[j]==inst[i]}. Trend-key positives PULL same-trend windows
    together (fixes false-negatives); instance positives keep the crop-invariance. Anchors with
    no positive contribute 0."""
    N = z.shape[0]
    sim = (z @ z.t()) / temperature
    eye = torch.eye(N, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, -1e9)                       # drop self-similarity
    logp = sim - torch.logsumexp(sim, dim=1, keepdim=True) # log-softmax over the N-1 others
    pos = ((key[:, None] == key[None, :]) | (inst[:, None] == inst[None, :])) & ~eye
    cnt = pos.sum(1)
    loss = -(logp * pos).sum(1) / cnt.clamp_min(1)         # avg positive log-prob per anchor
    valid = cnt > 0
    return loss[valid].mean() if valid.any() else (z.sum() * 0.0)


class ContrastiveTrendNet(nn.Module):
    """Mantis encoder + channel adapter + SimCLR-style projection head (Mantis 'prj'). embed(x) =
    per-channel encode + concat (the SAME embedding downstream consumes via backbone_ckpt);
    forward(x) = L2-normalized projection for the contrastive loss (head discarded after)."""
    def __init__(self, C=5, new_channels=8, proj_dim=128, model_id='paris-noah/Mantis-8M'):
        super().__init__()
        from mantis.architecture import Mantis8M
        from mantis.adapters import LinearChannelCombiner
        self.encoder = Mantis8M.from_pretrained(model_id)
        hidden = getattr(self.encoder, 'hidden_dim', 256)
        self.new_c = min(new_channels, C)
        self.adapter = LinearChannelCombiner(num_channels=C, new_num_channels=self.new_c)
        self.C = C
        emb = hidden * self.new_c
        self.prj = nn.Sequential(nn.LayerNorm(emb), nn.Linear(emb, emb), nn.GELU(),
                                 nn.Linear(emb, proj_dim))

    def embed(self, x):                                    # [B,C,L] -> [B, new_c*hidden]
        a = self.adapter(x)
        return torch.cat([_enc(self.encoder, a[:, [i], :]) for i in range(a.shape[1])], dim=-1)

    def forward(self, x):                                  # [B,C,L] -> [B, proj_dim] (normalized)
        return F.normalize(self.prj(self.embed(x)), dim=1)


def train_ssl_contrastive(big, train_starts, val_starts, *, context_lengths=(64, 100, 150, 200),
                          new_channels=8, proj_dim=128, temperature=0.1, crop_max=0.2,
                          epochs=60, steps_per_epoch=200, batch=512, lr=2e-3, weight_decay=0.05,
                          patience=8, device=None, model_id='paris-noah/Mantis-8M',
                          backbone_ckpt=None, control='real', seed=0, clamp=10.0, grad_clip=1.0,
                          verbose=True, **_ignore):
    """STAGE-3 (experiment): TREND CONTRASTIVE. Multi-positive InfoNCE (SupCon mechanics) with a
    SELF-SUPERVISED causal trend key -> groups same-trend windows, sharpens trend/chop separation.
    Warm-start the encoder from stage-2 (backbone_ckpt=ctx200). Returns (best_encoder_state, history).
    APPLES-TO-APPLES controls: trend key is ALWAYS from the real context; only the model INPUT is
    corrupted (shuffle=scramble time, random=noise) -> real should cluster, shuffle/random shouldn't."""
    os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
    dev = device or ('cuda' if torch.cuda.is_available()
                     else 'mps' if torch.backends.mps.is_available() else 'cpu')
    torch.manual_seed(seed); gen = torch.Generator(device=dev); gen.manual_seed(seed)
    C = int(big.shape[1]); clens = [int(x) for x in context_lengths]
    max_ctx = max(clens); clens_t = torch.as_tensor(clens, dtype=torch.long, device=dev)
    big_t = torch.as_tensor(np.asarray(big, np.float32), device=dev)
    tr = torch.as_tensor(np.asarray(train_starts, np.int64), device=dev)
    va = torch.as_tensor(np.asarray(val_starts, np.int64), device=dev)

    net = ContrastiveTrendNet(C=C, new_channels=new_channels, proj_dim=proj_dim,
                              model_id=model_id).to(dev)
    if backbone_ckpt:                                      # warm-start from stage-2 (ctx200)
        net.encoder.load_state_dict(torch.load(backbone_ckpt, map_location='cpu'))
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],
                            lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    def _batch(starts):
        b_idx = torch.randint(0, len(starts), (batch,), device=dev, generator=gen)
        L = int(clens_t[torch.randint(0, len(clens_t), (1,), device=dev, generator=gen)].item())
        w = _gather_batch(big_t, starts, b_idx, max_ctx)   # [B,C,max_ctx] real, ends at 'now'
        cs = _standardize(w[:, :, max_ctx - L:]).clamp(-clamp, clamp)   # [B,C,L] per-window z-score
        key = _trend_key(cs)                               # SS causal trend key from REAL context
        model_in = _apply_control(cs, control)             # corrupt ONLY the input (apples-to-apples)
        return model_in, key

    def _step_loss(model_in, key):
        z1, z2 = net(_random_crop_resize(model_in, crop_max)), net(_random_crop_resize(model_in, crop_max))
        z = torch.cat([z1, z2], 0)
        key2 = torch.cat([key, key], 0)
        ids = torch.arange(len(key), device=dev)
        inst = torch.cat([ids, ids], 0)                    # crop-pair shares instance id
        return _multi_positive_infonce(z, key2, inst, temperature)

    @torch.no_grad()
    def val_eval():
        net.eval(); tot = 0.0; nb = min(10, max(1, len(va) // batch))
        for _ in range(nb):
            tot += float(_step_loss(*_batch(va)))
        estd = float(net.embed(_batch(va)[0]).std(0).mean())
        net.train()
        return tot / nb, estd

    best, best_state, bad, history = 1e18, None, 0, []
    for ep in range(epochs):
        net.train(); tr_tot = 0.0
        for _ in range(steps_per_epoch):
            opt.zero_grad(set_to_none=True)
            loss = _step_loss(*_batch(tr))
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad], grad_clip)
            opt.step(); tr_tot += float(loss.detach())
        sched.step()
        vloss, estd = val_eval()
        history.append({'epoch': ep, 'train_loss': tr_tot / steps_per_epoch,
                        'val_loss': vloss, 'std': estd})
        improved = vloss < best - 1e-5
        if improved:
            best, bad = vloss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.encoder.state_dict().items()}
        else:
            bad += 1
        if verbose:
            print(f"  ep{ep:>3} train={tr_tot / steps_per_epoch:.4f} val={vloss:.4f} "
                  f"emb_std={estd:.4f}{'  *' if improved else ''}", flush=True)
        if bad >= patience:
            break
    return best_state, history
