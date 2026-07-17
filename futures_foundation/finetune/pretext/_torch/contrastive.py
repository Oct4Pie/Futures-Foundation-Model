"""Stage-2 temporal-neighborhood contrastive trainer.

``elapsed_time_v2`` removes the proven shortcuts in the original objective: positive/negative
distance is measured in wall-clock time relative to each context span, crop and mask parameters
are sampled independently per observation, every regime has equal loss weight, and synchronized
cross-stream observations are excluded from the negative set. ``bar_offset_v1`` is retained only
as an explicit reproducibility baseline for existing studies.

Teaches: windows close in time / structurally similar -> nearby embeddings; different market
structures -> far apart. The intended product is a smooth "market state geometry" in the
DOWNSTREAM embedding space (net.embed — what mantis_frozen consumes), validated by the spec's
structural metrics (temporal consistency / emergent clusters / multi-scale ordering / noise
robustness / temporal stability) — NOT by loss and NOT by trade outcomes. Its encoder checkpoint
becomes the warm start for stage-3 seq2seq.

Mechanics: fp32 InfoNCE (fp16-sensitive), fixed seeded val batches (stable early-stop),
_apply_control corrupts ONLY the input (SHUFFLE/RANDOM controls stay honest — they destroy
exactly the temporal structure this objective feeds on).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import (_apply_control, BaseTrainer, encode_independent,
                     load_encoder_checkpoint, preprocess_windows, resolve_preprocessing)
from .backbone import load_mantis

ELAPSED_TIME_OBJECTIVE = 'elapsed_time_v2'
LEGACY_BAR_OBJECTIVE = 'bar_offset_v1'
CONTRASTIVE_OBJECTIVES = (ELAPSED_TIME_OBJECTIVE, LEGACY_BAR_OBJECTIVE)
MINUTE_NS = 60 * 1_000_000_000

def _random_crop_resize(x, crop_max=0.2, *, gen=None, independent=True, return_params=False):
    """Crop-resize along time.

    V2 samples crop fraction and start independently for every observation and performs one
    vectorized grid-sample. ``independent=False`` reproduces the shared-batch legacy behavior.
    """
    B, C, L = x.shape
    n = B if independent else 1
    crop = torch.rand(n, device=x.device, generator=gen) * float(crop_max)
    # At least eight input samples remain, matching the old implementation's floor.
    crop = torch.minimum(crop, torch.full_like(crop, max(0.0, 1.0 - 8.0 / max(L, 8))))
    span = (1.0 - crop) * max(L - 1, 1)
    start = torch.rand(n, device=x.device, generator=gen) * (max(L - 1, 1) - span)
    if not independent:
        crop, span, start = crop.expand(B), span.expand(B), start.expand(B)
    u = torch.linspace(0.0, 1.0, L, device=x.device)[None, :]
    xcoord = 2.0 * (start[:, None] + u * span[:, None]) / max(L - 1, 1) - 1.0
    grid = torch.stack([xcoord, torch.zeros_like(xcoord)], dim=-1)[:, None, :, :]
    out = F.grid_sample(x[:, :, None, :], grid, mode='bilinear', padding_mode='border',
                        align_corners=True)[:, :, 0, :]
    return (out, {'crop_fraction': crop, 'crop_start': start}) if return_params else out


class ContrastiveTrendNet(nn.Module):
    """Channel-independent Mantis plus a cross-channel SimCLR projection head."""

    def __init__(self, C=5, new_channels=8, proj_dim=128, model_id='paris-noah/Mantis-8M',
                 model_version=None):
        super().__init__()
        self.encoder = load_mantis(model_id, model_version=model_version, device='cpu')
        hidden = getattr(self.encoder, 'hidden_dim', 256)
        self.new_c = C                 # new_channels retained only for legacy config compatibility
        self.C = C
        emb = hidden * self.new_c
        self.prj = nn.Sequential(nn.LayerNorm(emb), nn.Linear(emb, emb), nn.GELU(),
                                 nn.Linear(emb, proj_dim))

    def embed(self, x):                                    # [B,C,L] -> [B, new_c*hidden]
        return encode_independent(self.encoder, x)

    def forward(self, x):                                  # [B,C,L] -> [B, proj_dim] (normalized)
        return F.normalize(self.prj(self.embed(x)), dim=1)


def _snap_to_starts(starts, target):
    """Nearest valid window start for each target position (both 1-D int64, starts SORTED).
    Returns (snapped_starts, |snapped-target| distance) — the caller decides tolerance. Snapping
    handles stream boundaries in the concatenated corpus: a target that falls in a gap/next
    stream snaps far away and gets dropped by the tolerance check."""
    j = torch.searchsorted(starts, target).clamp(0, len(starts) - 1)
    jm = (j - 1).clamp_min(0)
    pick_lo = (target - starts[jm]).abs() < (starts[j] - target).abs()
    j = torch.where(pick_lo, jm, j)
    s = starts[j]
    return s, (s - target).abs()


def _snap_to_times(starts, times_ns, target_ns):
    """Nearest valid start by elapsed timestamp; all inputs are sorted int64 tensors."""
    j = torch.searchsorted(times_ns, target_ns).clamp(0, len(times_ns) - 1)
    jm = (j - 1).clamp_min(0)
    pick_lo = (target_ns - times_ns[jm]).abs() < (times_ns[j] - target_ns).abs()
    j = torch.where(pick_lo, jm, j)
    return starts[j], times_ns[j], (times_ns[j] - target_ns).abs()


def _zscore_clamp(w, clamp, preprocessing=None):
    """Versioned preprocessing + clamp (name retained for backward imports)."""
    return preprocess_windows(w, resolve_preprocessing(preprocessing), clamp=clamp)


def _vol_sigma(w_raw, close_ch=3):
    """Data-driven per-window volatility: mean |Δclose| / mean |close| of the RAW window.
    Scale-free across tickers/TFs; high = chaotic/noisy window. A weight source, NOT a label."""
    c = w_raw[:, min(close_ch, w_raw.shape[1] - 1), :]
    return c.diff(dim=1).abs().mean(1) / c.abs().mean(1).clamp_min(1e-9)


def _augment(x, gen, noise=0.10, scale=0.20, tmask=0.15, crop_max=0.2, *,
             independent=True, return_params=False):
    """One stochastic view: crop-resize (trend-shape preserving) + gaussian noise + per-channel
    scale jitter + contiguous time-mask. Input is standardized (unit-ish sigma), so `noise` is
    in sigma units."""
    v, params = _random_crop_resize(x, crop_max, gen=gen, independent=independent,
                                    return_params=True)
    if noise > 0:
        v = v + noise * torch.randn(v.shape, device=v.device, generator=gen)
    if scale > 0:
        sc = 1.0 + scale * (2 * torch.rand((v.shape[0], v.shape[1], 1), device=v.device,
                                           generator=gen) - 1)
        v = v * sc
    if tmask > 0:
        L = v.shape[2]
        mlen = max(1, int(L * tmask))
        n = v.shape[0] if independent else 1
        t0 = torch.randint(0, L - mlen + 1, (n,), device=v.device, generator=gen)
        if not independent:
            t0 = t0.expand(v.shape[0])
        ti = torch.arange(L, device=v.device)[None, :]
        masked = (ti >= t0[:, None]) & (ti < (t0 + mlen)[:, None])
        v = v.masked_fill(masked[:, None, :], 0.0)
        params['mask_start'] = t0
    else:
        params['mask_start'] = torch.full((v.shape[0],), -1, device=v.device)
    return (v, params) if return_params else v


def _weighted_supcon(z, group, pos_ok, positions, w_row, temperature, far_min,
                     stream_ids=None, *, timestamps_ns=None, context_ns=None,
                     negative_min_contexts=None, sync_exclusion_ns=None,
                     min_valid_negatives=1, return_diagnostics=False):
    """SupCon over the stacked batch. z:[N,D] L2-normalized; group: rows of the same anchor
    family (views + its temporal positives) are mutual POSITIVES; pos_ok=False rows (failed
    snaps) act as plain negatives. Pairs from DIFFERENT groups closer than `far_min` bars are
    EXCLUDED (neither positive nor negative). w_row: per-row anchor σ-weight."""
    N = z.shape[0]
    sim = (z @ z.t()) / temperature
    eye = torch.eye(N, dtype=torch.bool, device=z.device)
    same_group = (group[:, None] == group[None, :]) & ~eye
    elapsed = timestamps_ns is not None
    if elapsed:
        if stream_ids is None or context_ns is None or negative_min_contexts is None:
            raise ValueError('elapsed-time SupCon requires stream_ids, context_ns, and threshold')
        dt = (timestamps_ns[:, None] - timestamps_ns[None, :]).abs()
        threshold = torch.maximum(context_ns[:, None], context_ns[None, :]).float()
        threshold = threshold * float(negative_min_contexts)
        same_stream = stream_ids[:, None] == stream_ids[None, :]
        near = same_stream & (dt.float() < threshold)
        synchronized = (~same_stream & (dt <= int(sync_exclusion_ns or 0)))
    else:
        near = (positions[:, None] - positions[None, :]).abs() < far_min
        synchronized = torch.zeros_like(near)
    if stream_ids is not None and not elapsed:
        # Absolute row offsets from two concatenated streams may be numerically close but are not
        # temporal neighbors.  The near-pair exclusion applies only inside one market stream.
        near &= stream_ids[:, None] == stream_ids[None, :]
    pos = same_group & pos_ok[None, :] & pos_ok[:, None]
    # Failed temporal snaps from one anchor family are neither positives nor accidental negatives.
    invalid_family = same_group & ~pos
    excluded = eye | ((near | synchronized) & ~same_group) | invalid_family
    sim = sim.masked_fill(excluded, -1e9)
    logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    cnt = pos.sum(1)
    loss_row = -(logp * pos).sum(1) / cnt.clamp_min(1)
    negatives = (~excluded & ~same_group).sum(1)
    valid = (cnt > 0) & pos_ok & (negatives >= int(min_valid_negatives))
    if not valid.any():
        loss = z.sum() * 0.0
    else:
        w = w_row[valid]
        loss = (loss_row[valid] * w).sum() / w.sum().clamp_min(1e-9)
    if not return_diagnostics:
        return loss
    possible_cross = ((stream_ids[:, None] != stream_ids[None, :]) & ~eye
                      if stream_ids is not None else torch.zeros_like(eye))
    diag = {
        'valid_rows_fraction': float(valid.float().mean()),
        'valid_negatives_mean': float(negatives.float().mean()),
        'valid_negatives_min': int(negatives.min()),
        'sync_excluded_fraction': float(
            synchronized.sum().float() / possible_cross.sum().clamp_min(1)),
        'weight_min': float(w_row.min()), 'weight_max': float(w_row.max()),
    }
    return loss, diag


class _ContrastiveTrainer(BaseTrainer):
    def __init__(self, big, tr, va, *, seq=64, pos_deltas=(2, 16, 64), far_min=512,
                 new_channels=8, proj_dim=128, temperature=0.1, aug_noise=0.10, aug_scale=0.20,
                 aug_tmask=0.15, crop_max=0.2, vol_weight=None, w_clip=4.0, metrics_n=768,
                 contrastive_objective=ELAPSED_TIME_OBJECTIVE,
                 positive_gap_fractions=(0.6, 1.0, 2.0), max_positive_overlap=0.5,
                 positive_tolerance_fraction=0.20, negative_min_contexts=4.0,
                 sync_exclusion_minutes=60.0, min_valid_negatives=1,
                 train_start_times_ns=None, val_start_times_ns=None, stream_bar_ns=None,
                 model_id='paris-noah/Mantis-8M', model_version=None, backbone_ckpt=None,
                 clamp=10.0, preprocessing=None, **base):
        super().__init__(big, tr, va, amp=False, **base)        # InfoNCE runs fp32
        self.seq = int(seq)
        self.pos_deltas = [int(d) for d in pos_deltas]
        self.far_min = int(far_min)
        self.tol = [max(4, d // 2) for d in self.pos_deltas]    # snap tolerance per scale
        self.contrastive_objective = str(contrastive_objective)
        if self.contrastive_objective not in CONTRASTIVE_OBJECTIVES:
            raise ValueError(f'unknown contrastive objective: {self.contrastive_objective}')
        self.positive_gap_fractions = tuple(float(x) for x in positive_gap_fractions)
        if (not self.positive_gap_fractions or any(x <= 0 for x in self.positive_gap_fractions)
                or tuple(sorted(self.positive_gap_fractions)) != self.positive_gap_fractions):
            raise ValueError('positive_gap_fractions must be positive and sorted')
        self.max_positive_overlap = float(max_positive_overlap)
        if not 0.0 <= self.max_positive_overlap < 1.0:
            raise ValueError('max_positive_overlap must be in [0,1)')
        min_gap = 1.0 - self.max_positive_overlap
        if self.contrastive_objective == ELAPSED_TIME_OBJECTIVE and \
                self.positive_gap_fractions[0] + 1e-12 < min_gap:
            raise ValueError('smallest positive gap violates max_positive_overlap')
        self.positive_tolerance_fraction = float(positive_tolerance_fraction)
        self.negative_min_contexts = float(negative_min_contexts)
        self.sync_exclusion_ns = int(float(sync_exclusion_minutes) * MINUTE_NS)
        self.min_valid_negatives = int(min_valid_negatives)
        if self.positive_tolerance_fraction <= 0 or self.negative_min_contexts <= 0:
            raise ValueError('elapsed-time tolerances must be positive')
        if self.min_valid_negatives < 1:
            raise ValueError('min_valid_negatives must be >=1')
        self.new_channels, self.proj_dim = new_channels, proj_dim
        self.temperature, self.crop_max, self.clamp = temperature, crop_max, clamp
        self.aug_noise, self.aug_scale, self.aug_tmask = aug_noise, aug_scale, aug_tmask
        default_vol_weight = 0.0 if self.contrastive_objective == ELAPSED_TIME_OBJECTIVE else 1.0
        self.vol_weight = float(default_vol_weight if vol_weight is None else vol_weight)
        self.w_clip = float(w_clip)
        self.metrics_n = int(metrics_n)
        self.model_id = model_id
        self.model_version = model_version
        self.backbone_ckpt = backbone_ckpt
        self.preprocessing = resolve_preprocessing(preprocessing, backbone_ckpt)
        self.C = int(self.big_t.shape[1])
        if (len(self.tr) > 1 and not bool(torch.all(self.tr[1:] >= self.tr[:-1]))) or \
           (len(self.va) > 1 and not bool(torch.all(self.va[1:] >= self.va[:-1]))):
            raise ValueError("contrastive starts must be sorted so stream bounds remain valid")
        self.tr_sorted = self.tr                               # assembled starts are already sorted
        self.va_sorted = self.va
        self.tr_times = self._validate_start_times(train_start_times_ns, self.tr, 'train')
        self.va_times = self._validate_start_times(val_start_times_ns, self.va, 'validation')
        n_groups = (len(self.tr_groups) if self.tr_groups is not None else 1)
        bars = ([MINUTE_NS] * n_groups if stream_bar_ns is None else
                [int(x) for x in stream_bar_ns])
        if len(bars) != n_groups or any(x <= 0 for x in bars):
            raise ValueError('stream_bar_ns must contain one positive cadence per stream group')
        self.stream_bar_ns = torch.as_tensor(bars, dtype=torch.long, device=self.dev)
        self._last_loss_diagnostics = {}

    def _validate_start_times(self, values, starts, name):
        if values is None:
            times = starts * MINUTE_NS                 # deterministic synthetic/test fallback
        else:
            if len(values) != len(starts):
                raise ValueError(f'{name} timestamps must align one-to-one with starts')
            times = torch.as_tensor(values, dtype=torch.long, device=self.dev)
        bounds = self._groups_for(starts)
        slices = [(0, len(times))] if bounds is None else [(int(a), int(b)) for a, b in bounds]
        if any(bool(torch.any(times[a + 1:b] <= times[a:b - 1])) for a, b in slices if b - a > 1):
            raise ValueError(f'{name} timestamps must be strictly increasing inside each stream')
        return times

    def _resume_signature(self):
        sig = super()._resume_signature()
        sig.update({
            'contrastive_objective': self.contrastive_objective,
            'positive_gap_fractions': self.positive_gap_fractions,
            'max_positive_overlap': self.max_positive_overlap,
            'positive_tolerance_fraction': self.positive_tolerance_fraction,
            'negative_min_contexts': self.negative_min_contexts,
            'sync_exclusion_ns': self.sync_exclusion_ns,
            'min_valid_negatives': self.min_valid_negatives,
            'pos_deltas': tuple(self.pos_deltas), 'far_min': self.far_min,
            'vol_weight': self.vol_weight,
        })
        return sig

    def build_net(self):
        net = ContrastiveTrendNet(C=self.C, new_channels=self.new_channels,
                                  proj_dim=self.proj_dim, model_id=self.model_id,
                                  model_version=self.model_version).to(self.dev)
        if self.backbone_ckpt:                                  # warm-start from stage-1 mask
            load_encoder_checkpoint(net.encoder, self.backbone_ckpt, model_id=self.model_id,
                                    model_version=self.model_version, expected_channels=self.C)
        self.net = net

    # ------------------------------------------------------------------ batch construction
    def _windows_at(self, pos):
        """Gather raw windows [n, C, seq] ending-exclusive at pos+seq for absolute positions."""
        rows = pos[:, None] + torch.arange(self.seq, device=self.dev)[None, :]
        return self.big_t[rows].permute(0, 2, 1).contiguous()

    def _sorted(self, starts):
        return starts

    def _snap_within_groups(self, starts, target, stream_groups):
        """Nearest valid start inside each anchor's own symbol/timeframe stream."""
        bounds = self._groups_for(starts)
        if bounds is None:
            return _snap_to_starts(starts, target)
        snapped = torch.empty_like(target)
        distance = torch.empty_like(target)
        for gid in torch.unique(stream_groups):
            rows = stream_groups == gid
            lo, hi = bounds[gid, 0], bounds[gid, 1]
            sg, dg = _snap_to_starts(starts[lo:hi], target[rows])
            snapped[rows], distance[rows] = sg, dg
        return snapped, distance

    def _snap_times_within_groups(self, starts, times, target_times, stream_groups):
        """Nearest valid start by timestamp inside each anchor's symbol/timeframe stream."""
        bounds = self._groups_for(starts)
        if bounds is None:
            return _snap_to_times(starts, times, target_times)
        snapped = torch.empty_like(target_times)
        snapped_times = torch.empty_like(target_times)
        distance = torch.empty_like(target_times)
        for gid in torch.unique(stream_groups):
            rows = stream_groups == gid
            lo, hi = bounds[gid, 0], bounds[gid, 1]
            sg, tg, dg = _snap_to_times(starts[lo:hi], times[lo:hi], target_times[rows])
            snapped[rows], snapped_times[rows], distance[rows] = sg, tg, dg
        return snapped, snapped_times, distance

    def make_batch(self, starts, gen=None):
        gen = gen or self.gen
        ss = self._sorted(starts)
        b_idx, stream_groups = self.sample_start_indices(
            starts, generator=gen, return_groups=True)
        s = starts[b_idx]                                       # [B] anchor start positions
        raw_a = self._windows_at(s)
        sigma = _vol_sigma(raw_a)                               # data-driven anchor volatility
        pos_s, pos_ok = [], []
        pos_times = []
        anchor_times = self.tr_times[b_idx] if starts is self.tr else self.va_times[b_idx]
        context_ns = self.stream_bar_ns[stream_groups] * max(self.seq - 1, 1)
        if self.contrastive_objective == ELAPSED_TIME_OBJECTIVE:
            all_times = self.tr_times if starts is self.tr else self.va_times
            for fraction in self.positive_gap_fractions:
                gap_ns = torch.round(context_ns.float() * fraction).long()
                target_times = anchor_times + gap_ns
                ps, pt, dist = self._snap_times_within_groups(
                    ss, all_times, target_times, stream_groups)
                tolerance = torch.maximum(
                    torch.round(gap_ns.float() * self.positive_tolerance_fraction).long(),
                    self.stream_bar_ns[stream_groups])
                actual_gap = (pt - anchor_times).abs()
                overlap = (1.0 - actual_gap.float() / context_ns.float().clamp_min(1)).clamp(0, 1)
                pos_s.append(ps); pos_times.append(pt)
                pos_ok.append((dist <= tolerance) & (overlap <= self.max_positive_overlap + 1e-6))
        else:
            for d, tol in zip(self.pos_deltas, self.tol):
                ps, dist = self._snap_within_groups(ss, s + d, stream_groups)
                pos_s.append(ps); pos_times.append(anchor_times + d * self.stream_bar_ns[stream_groups])
                pos_ok.append(dist <= tol)
        anchors = _zscore_clamp(raw_a, self.clamp, self.preprocessing)
        positives = [_zscore_clamp(self._windows_at(p), self.clamp, self.preprocessing) for p in pos_s]
        # corrupt ONLY the input (controls destroy the temporal structure the loss feeds on)
        anchors = _apply_control(anchors, self.control)
        positives = [_apply_control(p, self.control) for p in positives]
        return (anchors, positives, torch.stack(pos_ok), torch.stack(pos_s), s, sigma,
                stream_groups, anchor_times, torch.stack(pos_times), context_ns)

    def _sigma_weights(self, sigma):
        """σ_t -> per-anchor weight: high-vol DOWN-weighted (w = (med/σ)^vol_weight, mean-1
        normalized, clipped). vol_weight=0 disables (all-equal)."""
        if self.contrastive_objective == ELAPSED_TIME_OBJECTIVE or self.vol_weight <= 0:
            return torch.ones_like(sigma)
        med = sigma.median().clamp_min(1e-9)
        w = (med / sigma.clamp_min(1e-9)) ** self.vol_weight
        w = w.clamp(1.0 / self.w_clip, self.w_clip)
        return w / w.mean().clamp_min(1e-9)

    def compute_loss(self, batch):
        (anchors, positives, pos_ok, pos_s, s, sigma, stream_groups,
         anchor_times, pos_times, context_ns) = batch
        B = anchors.shape[0]
        g = self.gen
        independent = self.contrastive_objective == ELAPSED_TIME_OBJECTIVE
        v1 = _augment(anchors, g, self.aug_noise, self.aug_scale, self.aug_tmask, self.crop_max,
                      independent=independent)
        v2 = _augment(anchors, g, self.aug_noise, self.aug_scale, self.aug_tmask, self.crop_max,
                      independent=independent)
        X = torch.cat([v1, v2] + positives, 0)                  # [(2+K)B, C, seq]
        z = self.net(X)                                         # L2-normalized projections
        ids = torch.arange(B, device=self.dev)
        K = len(positives)
        group = torch.cat([ids] * (2 + K), 0)
        ok = torch.cat([torch.ones(2 * B, dtype=torch.bool, device=self.dev),
                        pos_ok.reshape(-1)], 0)
        positions = torch.cat([s, s] + [p for p in pos_s], 0)
        timestamps = torch.cat([anchor_times, anchor_times] + [p for p in pos_times], 0)
        streams = stream_groups.repeat(2 + K)
        contexts = context_ns.repeat(2 + K)
        w_row = self._sigma_weights(sigma).repeat(2 + K)
        result = _weighted_supcon(
            z, group, ok, positions, w_row, self.temperature, self.far_min,
            stream_ids=streams,
            timestamps_ns=(timestamps if independent else None),
            context_ns=(contexts if independent else None),
            negative_min_contexts=(self.negative_min_contexts if independent else None),
            sync_exclusion_ns=(self.sync_exclusion_ns if independent else None),
            min_valid_negatives=(self.min_valid_negatives if independent else 1),
            return_diagnostics=True)
        loss, diag = result
        actual_gap = (pos_times - anchor_times[None, :]).abs().float() / MINUTE_NS
        overlap = (1.0 - actual_gap * MINUTE_NS /
                   context_ns[None, :].float().clamp_min(1)).clamp(0, 1)
        valid = pos_ok
        diag.update({
            'positive_valid_fraction': float(valid.float().mean()),
            'positive_gap_minutes_mean': float(actual_gap[valid].mean()) if valid.any() else 0.0,
            'positive_overlap_max': float(overlap[valid].max()) if valid.any() else 0.0,
            'vol_weight_sigma_corr': 0.0 if independent else float(
                torch.corrcoef(torch.stack([sigma.float(), self._sigma_weights(sigma).float()]))[0, 1]
            ) if len(sigma) > 1 and sigma.std() > 0 else 0.0,
        })
        for i in range(len(positives)):
            diag[f'positive_valid_fraction_scale_{i}'] = float(valid[i].float().mean())
            diag[f'positive_gap_minutes_scale_{i}'] = (
                float(actual_gap[i, valid[i]].mean()) if valid[i].any() else 0.0)
        self._last_loss_diagnostics = diag
        return loss

    # ------------------------------------------------------------------ spec validation (A-E)
    @torch.no_grad()
    def _regime_metrics(self, n=768):
        """The requirement doc's structural metrics, on the DOWNSTREAM embedding (net.embed):
          A smooth     temporal consistency: cos(z_t, z_nearest) - cos(z_t, z_random)  (want >0)
          B sil        emergent structure: k-means silhouette of the embedding cloud   (want >0)
          C scale_span multi-scale: sim(short Δ) - sim(long Δ) with monotone ordering  (want >=0)
          D vol_ratio  noise robustness: dispersion(high-σ) / dispersion(low-σ)        (want ~1)
          E drift      temporal stability: early-vs-late centroid shift / cloud radius (want <1)
        Computed on a seeded fixed sample of VAL windows — diagnostics + the stage gate, never a
        trading metric (the downstream/OOS gate is judged elsewhere)."""
        ss = self.va_sorted
        g = torch.Generator(device=self.dev)
        g.manual_seed(20260704)
        idx, stream_groups = self.sample_start_indices(
            self.va, generator=g, return_groups=True, count=min(n, len(ss)))
        order = torch.argsort(self.va[idx])
        s, stream_groups = self.va[idx][order], stream_groups[order]
        sample_times = self.va_times[idx][order]
        context_ns = self.stream_bar_ns[stream_groups] * max(self.seq - 1, 1)
        z = F.normalize(self.net.embed(_zscore_clamp(
            self._windows_at(s), self.clamp, self.preprocessing)), dim=1)
        # A: nearest valid neighbor vs random pair
        if self.contrastive_objective == ELAPSED_TIME_OBJECTIVE:
            s1, _, d1 = self._snap_times_within_groups(
                ss, self.va_times, sample_times + self.stream_bar_ns[stream_groups], stream_groups)
            okA = d1 <= self.stream_bar_ns[stream_groups]
        else:
            s1, d1 = self._snap_within_groups(ss, s + 1, stream_groups)
            okA = d1 <= 4
        z1 = F.normalize(self.net.embed(_zscore_clamp(
            self._windows_at(s1), self.clamp, self.preprocessing)), dim=1)
        perm = torch.randperm(len(z), device=self.dev, generator=g)
        smooth = float((z[okA] * z1[okA]).sum(1).mean() - (z * z[perm]).sum(1).mean())
        # C: similarity across the positive scales (short/medium/long)
        sims = []
        scales = (self.positive_gap_fractions if self.contrastive_objective == ELAPSED_TIME_OBJECTIVE
                  else self.pos_deltas)
        for i, scale in enumerate(scales):
            if self.contrastive_objective == ELAPSED_TIME_OBJECTIVE:
                target_gap = torch.round(context_ns.float() * float(scale)).long()
                pd, pt, dd = self._snap_times_within_groups(
                    ss, self.va_times, sample_times + target_gap, stream_groups)
                tolerance = torch.maximum(
                    torch.round(target_gap.float() * self.positive_tolerance_fraction).long(),
                    self.stream_bar_ns[stream_groups])
                actual_gap = (pt - sample_times).abs()
                overlap = (1.0 - actual_gap.float() / context_ns.float().clamp_min(1)).clamp(0, 1)
                okd = (dd <= tolerance) & (overlap <= self.max_positive_overlap + 1e-6)
            else:
                pd, dd = self._snap_within_groups(ss, s + int(scale), stream_groups)
                okd = dd <= self.tol[i]
            zk = F.normalize(self.net.embed(_zscore_clamp(
                self._windows_at(pd), self.clamp, self.preprocessing)),
                             dim=1)
            sims.append(float((z[okd] * zk[okd]).sum(1).mean()) if okd.any() else float('nan'))
        finite = [x for x in sims if x == x]
        scale_mono = bool(all(a >= b - 1e-6 for a, b in zip(finite, finite[1:])))
        scale_span = (finite[0] - finite[-1]) if len(finite) >= 2 else float('nan')
        # B: emergent cluster structure (CPU sklearn on the sample)
        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
            zc = z.float().cpu().numpy()
            lab = KMeans(n_clusters=6, n_init=4, random_state=0).fit_predict(zc)
            sil = float(silhouette_score(zc, lab)) if len(set(lab)) > 1 else 0.0
        except Exception:
            sil = float('nan')
        # D: high-vol vs low-vol dispersion (collapse/domination check)
        sig = _vol_sigma(self._windows_at(s))
        q1, q2 = torch.quantile(sig.float(), torch.tensor([1 / 3, 2 / 3], device=sig.device))
        lo, hi = z[sig <= q1], z[sig >= q2]

        def _disp(a):
            return float((a - a.mean(0, keepdim=True)).norm(dim=1).mean()) if len(a) > 1 else 0.0
        vol_ratio = _disp(hi) / max(_disp(lo), 1e-9)
        # E: early-vs-late half stability
        half = len(z) // 2
        c_all = z.mean(0, keepdim=True)
        radius = float((z - c_all).norm(dim=1).mean())
        drift = float((z[:half].mean(0) - z[half:].mean(0)).norm()) / max(radius, 1e-9)
        return {'smooth': smooth, 'sil': sil, 'scale_span': float(scale_span),
                'scale_mono': scale_mono, 'vol_ratio': float(vol_ratio), 'drift': drift}

    @torch.no_grad()
    def val_eval(self):
        """FIXED-batch val loss (seeded, RNG save/restored -> stable early-stop) + the spec's
        A-E regime-geometry metrics on the downstream embedding."""
        self.net.eval()
        cpu_state = torch.random.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        torch.manual_seed(20260704)
        vgen = torch.Generator(device=self.dev)
        vgen.manual_seed(20260704)
        try:
            tot = 0.0
            pair_diags = []
            nb = min(self.val_batches or 10, max(1, len(self.va) // self.batch))
            for _ in range(nb):
                tot += float(self.compute_loss(self.make_batch(self.va, gen=vgen)))
                pair_diags.append(dict(self._last_loss_diagnostics))
            mx = self._regime_metrics(self.metrics_n)
            for key in ('valid_rows_fraction', 'valid_negatives_mean', 'valid_negatives_min',
                        'sync_excluded_fraction', 'weight_min', 'weight_max',
                        'positive_valid_fraction', 'positive_gap_minutes_mean',
                        'positive_overlap_max', 'vol_weight_sigma_corr'):
                values = [float(d[key]) for d in pair_diags if key in d]
                if values:
                    mx[key] = min(values) if key in ('valid_negatives_min',) else sum(values) / len(values)
            for key in sorted({k for d in pair_diags for k in d
                               if k.startswith('positive_valid_fraction_scale_') or
                               k.startswith('positive_gap_minutes_scale_')}):
                values = [float(d[key]) for d in pair_diags if key in d]
                mx[key] = sum(values) / len(values)
            b = self.make_batch(self.va, gen=vgen)
            mx['std'] = float(self.net.embed(b[0]).std(0).mean())
        finally:
            torch.random.set_rng_state(cpu_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)
        self.net.train()
        return tot / nb, mx

    def log_line(self, ep, tr_loss, vloss, extra, improved):
        if self.verbose:
            print(f"  ep{ep:>3} train={tr_loss:.4f} val={vloss:.4f} "
                  f"A.smooth={extra['smooth']:+.3f} B.sil={extra['sil']:+.3f} "
                  f"C.span={extra['scale_span']:+.3f}{'✓' if extra['scale_mono'] else '✗'} "
                  f"D.vol={extra['vol_ratio']:.2f} E.drift={extra['drift']:.2f} "
                  f"pos={extra.get('positive_valid_fraction', 0):.2f} "
                  f"neg={extra.get('valid_negatives_mean', 0):.1f} "
                  f"std={extra['std']:.3f}{'  *' if improved else ''}", flush=True)


def regime_gate(extra):
    """The requirement doc's success definition as an explicit gate on the A-E metrics.
    Thresholds are stated heuristics for 'smooth, structured market-state geometry' vs noise:
      A smooth > 0.05 (adjacent windows measurably closer than random)
      B sil    > 0.05 (clusters emerge above the no-structure ~0 line)
      C monotone scale ordering with span >= 0 (multi-scale similarity preserved)
      D vol_ratio in [0.5, 2.0] (high-vol neither collapsed nor dominating)
      E drift  < 1.0 (early/late centroids shift less than the cloud radius)"""
    checks = {
        'A_temporal_consistency': extra['smooth'] > 0.05,
        'B_emergent_structure': extra['sil'] == extra['sil'] and extra['sil'] > 0.05,
        'C_multi_scale': bool(extra['scale_mono']) and extra['scale_span'] >= 0,
        'D_noise_robustness': 0.5 <= extra['vol_ratio'] <= 2.0,
        'E_temporal_stability': extra['drift'] < 1.0,
    }
    return all(checks.values()), checks


def train_ssl_contrastive(big, train_starts, val_starts, *, seq=64, pos_deltas=(2, 16, 64),
                          far_min=512, new_channels=8, proj_dim=128, temperature=0.1,
                          aug_noise=0.10, aug_scale=0.20, aug_tmask=0.15, crop_max=0.2,
                          vol_weight=None, w_clip=4.0, metrics_n=768, epochs=60,
                          contrastive_objective=ELAPSED_TIME_OBJECTIVE,
                          positive_gap_fractions=(0.6, 1.0, 2.0), max_positive_overlap=0.5,
                          positive_tolerance_fraction=0.20, negative_min_contexts=4.0,
                          sync_exclusion_minutes=60.0, min_valid_negatives=1,
                          steps_per_epoch=200, batch=256, lr=2e-4, weight_decay=0.05,
                          patience=8, device=None, model_id='paris-noah/Mantis-8M',
                          model_version=None, backbone_ckpt=None, control='real', seed=0,
                          clamp=10.0, grad_clip=1.0,
                          verbose=True, ckpt_path=None, resume=False, freeze_encoder_layers=0,
                          **_ignore):
    """Temporal-neighborhood contrastive regime refinement. The default v2 objective uses
    elapsed-time positives/negatives, independent augmentation, and equal regime weighting.
    Returns (best_encoder_state, history); history extras carry the spec's A-E regime metrics
    (see regime_gate for the pass/fail)."""
    return _ContrastiveTrainer(
        big, train_starts, val_starts, seq=seq, pos_deltas=pos_deltas, far_min=far_min,
        new_channels=new_channels, proj_dim=proj_dim, temperature=temperature,
        aug_noise=aug_noise, aug_scale=aug_scale, aug_tmask=aug_tmask, crop_max=crop_max,
        vol_weight=vol_weight, w_clip=w_clip, metrics_n=metrics_n, model_id=model_id,
        contrastive_objective=contrastive_objective,
        positive_gap_fractions=positive_gap_fractions,
        max_positive_overlap=max_positive_overlap,
        positive_tolerance_fraction=positive_tolerance_fraction,
        negative_min_contexts=negative_min_contexts,
        sync_exclusion_minutes=sync_exclusion_minutes,
        min_valid_negatives=min_valid_negatives,
        train_start_times_ns=_ignore.get('train_start_times_ns'),
        val_start_times_ns=_ignore.get('val_start_times_ns'),
        stream_bar_ns=_ignore.get('stream_bar_ns'),
        model_version=model_version,
        backbone_ckpt=backbone_ckpt, clamp=clamp, epochs=epochs,
        preprocessing=_ignore.get('preprocessing'),
        steps_per_epoch=steps_per_epoch, batch=batch, lr=lr, weight_decay=weight_decay,
        patience=patience, device=device, seed=seed, grad_clip=grad_clip, verbose=verbose,
        control=control, ckpt_path=ckpt_path, resume=resume,
        freeze_encoder_layers=freeze_encoder_layers,
        train_group_bounds=_ignore.get('train_group_bounds'),
        val_group_bounds=_ignore.get('val_group_bounds'),
        val_batches=_ignore.get('val_batches')).fit()
