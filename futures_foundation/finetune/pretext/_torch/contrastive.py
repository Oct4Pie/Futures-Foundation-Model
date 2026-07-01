"""Stage-3 (EXPERIMENT) trainer: TREND CONTRASTIVE. Reuses Mantis's InfoNCE machinery
(normalized-sim + temperature + projection head + RandomCropResize) but ADAPTS single-positive
instance-discrimination -> MULTI-POSITIVE (SupCon mechanics) grouped by a SELF-SUPERVISED CAUSAL
trend key. Same-trend windows become POSITIVES (pulled together) -> groups trends / separates chop.
Runs fp32 (amp=False) — the normalized-similarity InfoNCE is fp16-sensitive."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import _enc, _standardize, _apply_control, _gather_batch, BaseTrainer


def _random_crop_resize(x, crop_max=0.2):
    """Mantis RandomCropResize (multichannel): crop a random 0..crop_max fraction off the time axis
    (random start), interpolate back to the original length. Preserves trend SHAPE. One view."""
    B, C, L = x.shape
    cr = float(torch.empty(1, device=x.device).uniform_(0.0, float(crop_max)).item())
    cl = max(8, int(L * (1.0 - cr)))
    start = int(torch.randint(0, L - cl + 1, (1,)).item())
    return F.interpolate(x[:, :, start:start + cl], size=L, mode='linear', align_corners=False)


def _trend_key(ctx, close_ch=3):
    """SELF-SUPERVISED, CAUSAL trend key per window from PAST bars ONLY (no future, no label).
    Signature = (direction bucket, magnitude bucket) of the trailing least-squares slope on the
    CLOSE channel. Windows sharing a key = 'same trend character' = POSITIVES."""
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
    """SupCon-style multi-positive InfoNCE over a 2-view batch. z:[N,D] L2-normalized. Positive(i) =
    {j!=i : key[j]==key[i] OR inst[j]==inst[i]}. Trend-key positives PULL same-trend windows
    together (fixes false-negatives); instance positives keep crop-invariance."""
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
    per-channel encode + concat (the SAME embedding downstream consumes); forward(x) = L2-normalized
    projection for the contrastive loss (head discarded after)."""

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


class _ContrastiveTrainer(BaseTrainer):
    def __init__(self, big, tr, va, *, context_lengths=(64, 100, 150, 200), new_channels=8,
                 proj_dim=128, temperature=0.1, crop_max=0.2, model_id='paris-noah/Mantis-8M',
                 backbone_ckpt=None, clamp=10.0, **base):
        super().__init__(big, tr, va, amp=False, **base)             # InfoNCE runs fp32
        self.clens = [int(x) for x in context_lengths]
        self.max_ctx = max(self.clens)
        self.clens_t = torch.as_tensor(self.clens, dtype=torch.long, device=self.dev)
        self.new_channels, self.proj_dim = new_channels, proj_dim
        self.temperature, self.crop_max, self.clamp = temperature, crop_max, clamp
        self.model_id, self.backbone_ckpt = model_id, backbone_ckpt
        self.C = int(self.big_t.shape[1])

    def build_net(self):
        net = ContrastiveTrendNet(C=self.C, new_channels=self.new_channels, proj_dim=self.proj_dim,
                                  model_id=self.model_id).to(self.dev)
        if self.backbone_ckpt:                                       # warm-start from stage-2 (ctx200)
            net.encoder.load_state_dict(torch.load(self.backbone_ckpt, map_location='cpu'))
        self.net = net

    def make_batch(self, starts):
        b_idx = torch.randint(0, len(starts), (self.batch,), device=self.dev, generator=self.gen)
        L = int(self.clens_t[torch.randint(0, len(self.clens_t), (1,), device=self.dev,
                                           generator=self.gen)].item())
        w = _gather_batch(self.big_t, starts, b_idx, self.max_ctx)   # [B,C,max_ctx] real, ends at 'now'
        cs = _standardize(w[:, :, self.max_ctx - L:]).clamp(-self.clamp, self.clamp)   # [B,C,L] z-score
        key = _trend_key(cs)                                        # SS causal trend key from REAL context
        return _apply_control(cs, self.control), key                # corrupt ONLY the input

    def compute_loss(self, batch):
        model_in, key = batch
        z1 = self.net(_random_crop_resize(model_in, self.crop_max))
        z2 = self.net(_random_crop_resize(model_in, self.crop_max))
        z = torch.cat([z1, z2], 0)
        key2 = torch.cat([key, key], 0)
        ids = torch.arange(len(key), device=self.dev)
        inst = torch.cat([ids, ids], 0)                             # crop-pair shares instance id
        return _multi_positive_infonce(z, key2, inst, self.temperature)

    @torch.no_grad()
    def val_eval(self):
        self.net.eval(); tot = 0.0; nb = min(10, max(1, len(self.va) // self.batch))
        for _ in range(nb):
            tot += float(self.compute_loss(self.make_batch(self.va)))
        estd = float(self.net.embed(self.make_batch(self.va)[0]).std(0).mean())
        self.net.train()
        return tot / nb, {'std': estd}


def train_ssl_contrastive(big, train_starts, val_starts, *, context_lengths=(64, 100, 150, 200),
                          new_channels=8, proj_dim=128, temperature=0.1, crop_max=0.2, epochs=60,
                          steps_per_epoch=200, batch=512, lr=2e-3, weight_decay=0.05, patience=8,
                          device=None, model_id='paris-noah/Mantis-8M', backbone_ckpt=None,
                          control='real', seed=0, clamp=10.0, grad_clip=1.0, verbose=True, **_ignore):
    """Trend contrastive (multi-positive InfoNCE by self-supervised trend key). Warm-start from
    stage-2 (ctx200). Returns (best_encoder_state, history) with 'val_loss' + 'std'."""
    return _ContrastiveTrainer(big, train_starts, val_starts, context_lengths=context_lengths,
                               new_channels=new_channels, proj_dim=proj_dim, temperature=temperature,
                               crop_max=crop_max, model_id=model_id, backbone_ckpt=backbone_ckpt,
                               clamp=clamp, epochs=epochs, steps_per_epoch=steps_per_epoch, batch=batch,
                               lr=lr, weight_decay=weight_decay, patience=patience, device=device,
                               seed=seed, grad_clip=grad_clip, verbose=verbose, control=control).fit()
