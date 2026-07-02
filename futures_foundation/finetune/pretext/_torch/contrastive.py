"""Stage-3 trainer v3: FORWARD BARRIER-EXCURSION CONTRASTIVE. Reuses Mantis's own InfoNCE machinery
(normalized-sim + temperature + projection head + RandomCropResize) adapted multi-positive (SupCon
mechanics) — the ONLY thing that changes across versions is the POSITIVE-PAIR DEFINITION, the lever
that makes or breaks contrastive trend detection.

WHY v1 WASHED (benchmarked 2026-07: WR@3R 42.3 vs 42.8, spread 14.2 vs 14.8): its key was the
TRAILING slope of the input — computable from the window (shortcut; taught nothing forward) and
past-tense (pulled together windows whose futures differ, erasing the tell).

WHY v2 WASHED (Optuna sweep 2026-07-02 on the promoted stage-2 base: 11 trials, best 6.667 < the
6.720 baseline, key_gap pinned at ~0.003 across EVERY config): its key — future direction x path
EFFICIENCY (|net|/sum|steps|) — is a statistic of the future path that the stage-2 FORECAST
objective already learned (stage-2 regresses the future candles; net-move efficiency is implicit in
that). Refining stage-2 with a key it already encodes gave the loss no new structure to carve:
frozen configs reproduced stage-2, unfrozen ones only forgot it.

THE v3 KEY IS THE BARRIER EXCURSION — the asymmetric path statistic the forecast does NOT regress
and the exact shape of the ship metric (WR@3R = reach 3R before -1R): how far did the future close
path RUN in one direction (in calibrated candle-sigma units) before RETRACING one unit against
that run? key = direction bucket x excursion bucket (low excursion = chop: price went nowhere
cleanly). Same anti-shortcut property as v2 (the key is computed from the FUTURE segment,
target-side like the stage-2 forecast target, never model input) and same SSL discipline: raw
close path only, standardized by the context's own mean/std (the pipeline's universal z-score) —
NO ATR / NO R / NO indicators / NO labels; reserve ctx+horizon; holdout excluded upstream.

Bucket edges are FIXED (calibrated once from a seeded sample of train windows: the retrace unit =
|net| tercile, excursion edges = run terciles) — batch-relative quantiles would key the same window
differently across batches. Runs fp32 (amp=False) — the normalized-similarity InfoNCE is
fp16-sensitive. Val runs on FIXED batches (seeded, RNG state save/restored) so early-stop/best-save
track a stable signal.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import _enc, _apply_control, _gather_batch, BaseTrainer


def _random_crop_resize(x, crop_max=0.2):
    """Mantis RandomCropResize (multichannel): crop a random 0..crop_max fraction off the time axis
    (random start), interpolate back to the original length. Preserves trend SHAPE. One view."""
    B, C, L = x.shape
    cr = float(torch.empty(1, device=x.device).uniform_(0.0, float(crop_max)).item())
    cl = max(8, int(L * (1.0 - cr)))
    start = int(torch.randint(0, L - cl + 1, (1,)).item())
    return F.interpolate(x[:, :, start:start + cl], size=L, mode='linear', align_corners=False)


def _future_path_stats(cs, fs, close_ch=3):
    """(net, efficiency) of the FUTURE close path, in context-standardized units. Path = last
    context close -> the `horizon` future closes; net = end-start; efficiency = |net| / sum|steps|
    (1 = perfectly directional, ~0 = whipsaw). Both scale-free across TFs/tickers."""
    cch = min(close_ch, cs.shape[1] - 1)
    p = torch.cat([cs[:, cch, -1:], fs[:, cch, :]], dim=1)          # [B, h+1] close path from 'now'
    steps = p[:, 1:] - p[:, :-1]                                    # [B, h]
    net = p[:, -1] - p[:, 0]
    eff = net.abs() / steps.abs().sum(1).clamp_min(1e-9)
    return net, eff


def _future_barrier_stats(cs, fs, unit, close_ch=3):
    """(net, run) of the FUTURE close path — the v3 BARRIER-EXCURSION statistic, in context-
    standardized units. run = the largest excursion (in `unit`s) the path achieved in EITHER
    direction BEFORE first retracing `unit` against that direction from 'now' — the self-supervised
    analogue of 'reached K-R before -1R' computed purely from raw closes (no ATR/R/labels; ordering
    matters: a clean runner scores high, a whipsaw that gives back a unit before running scores
    low). net = end-start (for the direction bucket)."""
    cch = min(close_ch, cs.shape[1] - 1)
    p = torch.cat([cs[:, cch, -1:], fs[:, cch, :]], dim=1)          # [B, h+1] close path from 'now'
    d = p - p[:, :1]                                                # excursion from entry ('now')
    u = max(float(unit), 1e-6)
    hit_dn = (d <= -u).to(torch.int8).cummax(1).values.bool()       # been a unit BELOW entry by t
    hit_up = (d >= u).to(torch.int8).cummax(1).values.bool()        # been a unit ABOVE entry by t
    run_up = d.masked_fill(hit_dn, float('-inf')).max(1).values.clamp_min(0.0)   # MFE before -1u
    run_dn = (-d).masked_fill(hit_up, float('-inf')).max(1).values.clamp_min(0.0)
    net = p[:, -1] - p[:, 0]
    return net, torch.maximum(run_up, run_dn) / u                   # run in retrace-units


def _future_key(cs, fs, dz, e1, e2, close_ch=3):
    """SELF-SUPERVISED FORWARD trend-vs-chop key (v3): direction bucket (net vs FIXED deadzone dz)
    x BARRIER-EXCURSION bucket (run vs FIXED edges e1<e2, retrace unit = dz). key = dir*3 + run_b
    in [0,9); run bucket 0 = future CHOP (price ran nowhere before retracing). Same-key windows =
    'same future character' = POSITIVES. NOT computable from the model input (the future is
    target-side) — the anti-shortcut property v1 lacked. Unlike v2's path-efficiency (already
    implicit in the stage-2 forecast -> key_gap ~0), the barrier excursion is an ORDERING statistic
    the candle-MSE forecast does not regress — and it is the shape of WR@3R itself."""
    net, run = _future_barrier_stats(cs, fs, dz, close_ch)
    dir_b = (torch.sign(net) * (net.abs() > dz)).long() + 1         # 0/1/2 = down/flat/up
    run_b = (run > e1).long() + (run > e2).long()                   # 0/1/2 = chop/mixed/runner
    return dir_b * 3 + run_b                                        # [B] in [0, 9)


def _multi_positive_infonce(z, key, inst, temperature, pos_cap=None):
    """SupCon-style multi-positive InfoNCE over a 2-view batch. z:[N,D] L2-normalized. Positive(i) =
    {j!=i : key[j]==key[i] OR inst[j]==inst[i]}. Key positives PULL same-future-character windows
    together; instance positives keep crop-invariance. pos_cap (optional) randomly subsamples the
    KEY positives per anchor (instance positives always kept) so a few huge buckets can't reduce
    the loss to bucket-centroid averaging."""
    N = z.shape[0]
    sim = (z @ z.t()) / temperature
    eye = torch.eye(N, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, -1e9)                       # drop self-similarity
    logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)  # log-softmax over the N-1 others
    key_pos = (key[:, None] == key[None, :]) & ~eye
    if pos_cap is not None:
        kcnt = key_pos.sum(1, keepdim=True).clamp_min(1)
        keep = torch.rand(key_pos.shape, device=z.device) < (float(pos_cap) / kcnt)
        key_pos = key_pos & keep                            # expected <= pos_cap key-positives/anchor
    pos = key_pos | ((inst[:, None] == inst[None, :]) & ~eye)
    cnt = pos.sum(1)
    loss = -(logp * pos).sum(1) / cnt.clamp_min(1)          # avg positive log-prob per anchor
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
    def __init__(self, big, tr, va, *, context_lengths=(64, 100, 150, 200), contrast_horizon=25,
                 new_channels=8, proj_dim=128, temperature=0.1, crop_max=0.2, pos_cap=64,
                 model_id='paris-noah/Mantis-8M', backbone_ckpt=None, clamp=10.0, **base):
        super().__init__(big, tr, va, amp=False, **base)             # InfoNCE runs fp32
        self.clens = [int(x) for x in context_lengths]
        self.max_ctx, self.h = max(self.clens), int(contrast_horizon)
        self.parent = self.max_ctx + self.h                          # context + FUTURE (key source)
        self.clens_t = torch.as_tensor(self.clens, dtype=torch.long, device=self.dev)
        self.new_channels, self.proj_dim = new_channels, proj_dim
        self.temperature, self.crop_max, self.clamp = temperature, crop_max, clamp
        self.pos_cap = int(pos_cap) if pos_cap else None
        self.model_id, self.backbone_ckpt = model_id, backbone_ckpt
        self.C = int(self.big_t.shape[1])
        self.dz, self.e1, self.e2 = self._calibrate_edges()          # FIXED key edges (stable keys)

    def _calibrate_edges(self, n=4096, seed_off=1):
        """FIXED bucket edges from a seeded sample of TRAIN windows: dz = |net| tercile (direction
        deadzone -> ~1/3 of futures 'flat'; doubles as the v3 RETRACE UNIT); e1/e2 = terciles of
        the barrier-excursion run computed with that unit (chop/mixed/runner). Data-derived, never
        hand-tuned. v1 bucketed per batch -> the same window keyed differently across batches."""
        g = torch.Generator(device=self.dev); g.manual_seed(int(self.gen.initial_seed()) + seed_off)
        b_idx = torch.randint(0, len(self.tr), (min(n, len(self.tr)),), device=self.dev, generator=g)
        cs, fs = self._split_standardize(_gather_batch(self.big_t, self.tr, b_idx, self.parent),
                                         self.max_ctx)
        net, _ = _future_barrier_stats(cs, fs, 1.0)                  # net is unit-independent
        net_c = net.abs().float().cpu()                              # quantiles on CPU (MPS-safe)
        dz = float(torch.quantile(net_c, 1 / 3))
        _, run = _future_barrier_stats(cs, fs, dz)                   # run in dz retrace-units
        run_c = run.float().cpu()
        e1, e2 = float(torch.quantile(run_c, 1 / 3)), float(torch.quantile(run_c, 2 / 3))
        if self.verbose:
            print(f"  [key-edges] dz={dz:.3f} run=({e1:.3f},{e2:.3f}) from {len(net_c)} windows",
                  flush=True)
        return dz, e1, e2

    def _split_standardize(self, w, ctx_end, L=None):
        """Parent window [B,C,parent] -> (cs [B,C,L], fs [B,C,h]) both standardized by the CONTEXT's
        per-channel stats (the forecast pretext's convention: future in context-sigma units)."""
        L = L or self.max_ctx
        ctx_raw = w[:, :, ctx_end - L:ctx_end]
        fut_raw = w[:, :, ctx_end:]
        m = ctx_raw.mean(2, keepdim=True); s = ctx_raw.std(2, keepdim=True) + 1e-6
        cs = ((ctx_raw - m) / s).clamp(-self.clamp, self.clamp)
        fs = ((fut_raw - m) / s).clamp(-self.clamp, self.clamp)
        return cs, fs

    def build_net(self):
        net = ContrastiveTrendNet(C=self.C, new_channels=self.new_channels, proj_dim=self.proj_dim,
                                  model_id=self.model_id).to(self.dev)
        if self.backbone_ckpt:                                       # warm-start from the stage-2 winner
            net.encoder.load_state_dict(torch.load(self.backbone_ckpt, map_location='cpu'))
        self.net = net

    def make_batch(self, starts, gen=None):
        gen = gen or self.gen
        b_idx = torch.randint(0, len(starts), (self.batch,), device=self.dev, generator=gen)
        L = int(self.clens_t[torch.randint(0, len(self.clens_t), (1,), device=self.dev,
                                           generator=gen)].item())
        w = _gather_batch(self.big_t, starts, b_idx, self.parent)    # [B,C,ctx+h]; ctx ends at 'now'
        cs, fs = self._split_standardize(w, self.max_ctx, L)
        key = _future_key(cs, fs, self.dz, self.e1, self.e2)         # key from the REAL future
        return _apply_control(cs, self.control), key                 # corrupt ONLY the input

    def compute_loss(self, batch):
        model_in, key = batch
        z1 = self.net(_random_crop_resize(model_in, self.crop_max))
        z2 = self.net(_random_crop_resize(model_in, self.crop_max))
        z = torch.cat([z1, z2], 0)
        key2 = torch.cat([key, key], 0)
        ids = torch.arange(len(key), device=self.dev)
        inst = torch.cat([ids, ids], 0)                              # crop-pair shares instance id
        return _multi_positive_infonce(z, key2, inst, self.temperature, self.pos_cap)

    @torch.no_grad()
    def val_eval(self):
        """FIXED-batch val (seeded generator + global-RNG save/restore, so crops/pos_cap noise is
        identical every epoch) -> a stable early-stop/best-save signal. Also reports 'key_gap' =
        mean same-key minus diff-key cosine sim in projection space — the trend/chop separation
        actually forming (the quantity v1 optimized blindly via its loss)."""
        self.net.eval()
        cpu_state = torch.random.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        torch.manual_seed(20260701)
        vgen = torch.Generator(device=self.dev); vgen.manual_seed(20260701)
        try:
            tot = 0.0; nb = min(10, max(1, len(self.va) // self.batch))
            for _ in range(nb):
                tot += float(self.compute_loss(self.make_batch(self.va, gen=vgen)))
            ctx, key = self.make_batch(self.va, gen=vgen)            # one batch for diagnostics
            z = self.net(ctx)
            sim = z @ z.t()
            eye = torch.eye(len(z), dtype=torch.bool, device=z.device)
            same = (key[:, None] == key[None, :]) & ~eye
            diff = ~same & ~eye
            gap = float(sim[same].mean() - sim[diff].mean()) if same.any() and diff.any() else 0.0
            estd = float(self.net.embed(ctx).std(0).mean())
        finally:
            torch.random.set_rng_state(cpu_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)
        self.net.train()
        return tot / nb, {'std': estd, 'key_gap': gap}

    def log_line(self, ep, tr_loss, vloss, extra, improved):
        if self.verbose:
            print(f"  ep{ep:>3} train={tr_loss:.4f} val={vloss:.4f} "
                  f"key_gap={extra['key_gap']:+.4f} emb_std={extra['std']:.4f}"
                  f"{'  *' if improved else ''}", flush=True)


def train_ssl_contrastive(big, train_starts, val_starts, *, context_lengths=(64, 100, 150, 200),
                          contrast_horizon=25, new_channels=8, proj_dim=128, temperature=0.1,
                          crop_max=0.2, pos_cap=64, epochs=60, steps_per_epoch=200, batch=512,
                          lr=2e-4, weight_decay=0.05, patience=8, device=None,
                          model_id='paris-noah/Mantis-8M', backbone_ckpt=None, control='real',
                          seed=0, clamp=10.0, grad_clip=1.0, verbose=True, ckpt_path=None,
                          resume=False, freeze_encoder_layers=0, **_ignore):
    """Forward trend-vs-chop contrastive (multi-positive InfoNCE keyed by the FUTURE window's
    direction x path-efficiency — self-supervised, anti-shortcut). Warm-start from the best stage-2
    checkpoint. Returns (best_encoder_state, history) with 'val_loss' + 'std' + 'key_gap'."""
    return _ContrastiveTrainer(big, train_starts, val_starts, context_lengths=context_lengths,
                               contrast_horizon=contrast_horizon, new_channels=new_channels,
                               proj_dim=proj_dim, temperature=temperature, crop_max=crop_max,
                               pos_cap=pos_cap, model_id=model_id, backbone_ckpt=backbone_ckpt,
                               clamp=clamp, epochs=epochs, steps_per_epoch=steps_per_epoch,
                               batch=batch, lr=lr, weight_decay=weight_decay, patience=patience,
                               device=device, seed=seed, grad_clip=grad_clip, verbose=verbose,
                               control=control, ckpt_path=ckpt_path, resume=resume,
                               freeze_encoder_layers=freeze_encoder_layers).fit()
