"""Shared TORCH layer for the SSL pretext trainers.

Window helpers (encode / standardize / control-corrupt / gather), the frozen-embedding + ONNX
primitives, and a BaseTrainer that owns the SHARED training loop (epoch loop, AMP, grad-clip,
cosine LR, early-stop on val loss, best-ENCODER-state snapshot). Each pretext trainer
(mask / forecast / contrastive) subclasses BaseTrainer and provides only build_net / make_batch /
compute_loss / val_eval — no copied loops. torch imports live under this subpackage only (loaded
lazily), so the orchestrator + task registry stay torch-free.
"""
import os

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class _nullctx:
    def __enter__(self): return None
    def __exit__(self, *a): return False


def _enc(encoder, x1):
    """Encode one channel [B,1,L] -> [B, hidden], interpolating the window to Mantis's native
    seq_len (512) first so it ALWAYS sees its pretrained patch size."""
    L = int(getattr(encoder, 'seq_len', 512))
    if x1.shape[-1] != L:
        x1 = F.interpolate(x1, size=L, mode='linear', align_corners=False)
    return encoder(x1)


def _standardize(x):                                     # per-window per-channel z-score
    m = x.mean(dim=2, keepdim=True)
    s = x.std(dim=2, keepdim=True)
    return (x - m) / (s + 1e-6)


def _time_shuffle(x):
    """Permute the time axis independently per sample -> destroys temporal order, keeps the exact
    value set. Used for the SHUFFLE control."""
    B, C, T = x.shape
    perm = torch.argsort(torch.rand(B, T, device=x.device), 1)
    return torch.gather(x, 2, perm[:, None, :].expand(B, C, T))


def _apply_control(x, control):
    """Corrupt ONLY the model INPUT per the apples-to-apples control: 'shuffle' scrambles the time
    axis, 'random' replaces with noise, else (real) passes through. The target/trend-key is always
    computed from the REAL context by the caller -> real must beat shuffle/random. Shared by every
    pretext trainer so the control logic lives in one place."""
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


# ----------------------------------------------------------------- frozen embedding (probe / cache)
@torch.no_grad()
def embed_encoder(big, starts, seq, *, ckpt=None, model_id='paris-noah/Mantis-8M',
                  device=None, batch=512, max_windows=20000, seed=0):
    """Frozen ENCODER-ONLY embeddings of clean (per-window z-scored) windows — the quantity that
    transfers downstream via backbone_ckpt. Each OHLCV channel is encoded independently and
    concatenated -> [M, C*hidden]. ckpt=None -> vanilla Mantis (probe baseline)."""
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
def embed_windows(windows, *, ckpt=None, model_id='paris-noah/Mantis-8M', device=None, batch=512):
    """Frozen ENCODER-ONLY embeddings of pre-extracted windows [N, C, seq] -> [N, C*hidden]. The
    head-only/cached downstream primitive: backbone frozen, embed ONCE, then a cheap head trains
    on the cache."""
    os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
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
    """ONNX-exportable wrapper reproducing embed_windows EXACTLY: per-window standardize ->
    per-channel interpolate to native length -> encode -> concat. Raw window [B,C,seq] in,
    embedding [B, C*hidden] out (standardize baked in so the bot feeds RAW OHLCV)."""

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
    # Mantis calls torch.diff internally (aten::diff has no ONNX symbolic) -> swap for an
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


# ================================================================= BASE TRAINER (shared loop)
class BaseTrainer:
    """Shared SSL training loop for every pretext. Subclass and implement:
      * build_net()            -> set self.net (+ warm-start from backbone_ckpt)
      * make_batch(starts)     -> a batch object for compute_loss / accumulated over steps
      * compute_loss(batch)    -> scalar loss (called inside the AMP context)
      * val_eval()             -> (val_loss: float, extra: dict incl 'std' for the collapse guard)
    fit() owns: the epoch/step loop, AMP (cuda) + grad-clip, cosine LR, early-stop on val_loss,
    and the best-ENCODER-state snapshot. Subclasses may override make_optimizer / log_line."""

    def __init__(self, big, train_starts, val_starts, *, epochs=60, steps_per_epoch=200, batch=512,
                 lr=1e-4, weight_decay=0.05, patience=8, device=None, seed=0, grad_clip=None,
                 amp=True, amp_dtype='fp16', verbose=True, control='real'):
        os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
        self.dev = device or ('cuda' if torch.cuda.is_available()
                              else 'mps' if torch.backends.mps.is_available() else 'cpu')
        torch.manual_seed(seed)
        self.gen = torch.Generator(device=self.dev); self.gen.manual_seed(seed)
        self.big_t = torch.as_tensor(np.asarray(big, np.float32), device=self.dev)
        self.tr = torch.as_tensor(np.asarray(train_starts, np.int64), device=self.dev)
        self.va = torch.as_tensor(np.asarray(val_starts, np.int64), device=self.dev)
        self.epochs, self.steps_per_epoch, self.batch = epochs, steps_per_epoch, batch
        self.lr, self.weight_decay, self.patience = lr, weight_decay, patience
        self.grad_clip, self.verbose, self.control = grad_clip, verbose, control
        self.use_amp = (self.dev == 'cuda') and amp                # contrastive runs fp32 (amp=False)
        _adt = torch.float16 if str(amp_dtype).lower() in ('fp16', 'float16') else torch.bfloat16
        self.amp_ctx = (lambda: torch.autocast('cuda', dtype=_adt)) if self.use_amp else (lambda: _nullctx())
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        self.net = None

    # ---- hooks (subclass implements) ----
    def build_net(self):
        raise NotImplementedError

    def make_batch(self, starts):
        raise NotImplementedError

    def compute_loss(self, batch):
        raise NotImplementedError

    def val_eval(self):
        raise NotImplementedError

    def make_optimizer(self):
        return torch.optim.AdamW([p for p in self.net.parameters() if p.requires_grad],
                                 lr=self.lr, weight_decay=self.weight_decay)

    def log_line(self, ep, tr_loss, vloss, extra, improved):
        if self.verbose:
            print(f"  ep{ep:>3} train={tr_loss:.4f} val={vloss:.4f} "
                  f"emb_std={extra.get('std', 0.0):.4f}{'  *' if improved else ''}", flush=True)

    def _encoder(self):
        net = self.net
        return net.encoder if not hasattr(net, '_orig_mod') else net._orig_mod.encoder

    def _params(self):
        return [p for p in self.net.parameters() if p.requires_grad]

    def fit(self):
        """Run the shared loop -> (best_encoder_state, history)."""
        self.build_net()
        opt = self.make_optimizer()
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        best, best_state, bad, history = 1e18, None, 0, []
        for ep in range(self.epochs):
            self.net.train(); tr_tot = 0.0
            for _ in range(self.steps_per_epoch):
                opt.zero_grad(set_to_none=True)
                with self.amp_ctx():
                    loss = self.compute_loss(self.make_batch(self.tr))
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    if self.grad_clip:
                        self.scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(self._params(), self.grad_clip)
                    self.scaler.step(opt); self.scaler.update()
                else:
                    loss.backward()
                    if self.grad_clip:
                        torch.nn.utils.clip_grad_norm_(self._params(), self.grad_clip)
                    opt.step()
                tr_tot += float(loss.detach())
            sched.step()
            if self.dev == 'cuda':
                torch.cuda.empty_cache()
            vloss, extra = self.val_eval()
            history.append({'epoch': ep, 'train_loss': tr_tot / self.steps_per_epoch,
                            'val_loss': vloss, **extra})
            improved = vloss < best - 1e-5
            if improved:
                best, bad = vloss, 0
                best_state = {k: v.detach().cpu().clone() for k, v in self._encoder().state_dict().items()}
            else:
                bad += 1
            self.log_line(ep, tr_tot / self.steps_per_epoch, vloss, extra, improved)
            if bad >= self.patience:
                break
        return best_state, history
