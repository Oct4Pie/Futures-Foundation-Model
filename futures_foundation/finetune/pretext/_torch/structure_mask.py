"""Stage-1 structural span reconstruction from raw OHLCV-derived, scale-stable targets."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..spans import sample_span_mask
from .backbone import load_mantis
from .common import (_apply_control, _gather_batch, BaseTrainer, encode_independent,
                     load_encoder_checkpoint, preprocess_windows, resolve_preprocessing)


STRUCTURAL_TARGET_NAMES = (
    'close_log_return_x100', 'log_true_range_atr', 'body_atr',
    'upper_wick_atr', 'lower_wick_atr', 'log_volume_change',
)


def structural_targets(raw):
    """Causal-within-window structural targets; no row after the input window is read."""
    if raw.ndim != 3 or raw.shape[1] < 5:
        raise ValueError('structural targets require [B,C>=5,T] OHLCV')
    o, h, l, c, v = (raw[:, index, :] for index in range(5))
    prev_c = torch.cat((c[:, :1], c[:, :-1]), dim=1)
    true_range = torch.maximum(h - l, torch.maximum((h - prev_c).abs(), (l - prev_c).abs()))
    # One scale per complete context preserves relative volatility inside the window. This target
    # is self-supervised/bidirectional; it never inspects a deployment-future row.
    scale = true_range.median(dim=1, keepdim=True).values.clamp_min(1e-6)
    ret = torch.log(c.clamp_min(1e-9) / prev_c.clamp_min(1e-9)) * 100.0
    ret[:, 0] = 0.0
    body = (c - o) / scale
    upper = (h - torch.maximum(o, c)).clamp_min(0.0) / scale
    lower = (torch.minimum(o, c) - l).clamp_min(0.0) / scale
    logv = torch.log1p(v.clamp_min(0.0))
    volume_change = torch.cat((torch.zeros_like(logv[:, :1]), logv[:, 1:] - logv[:, :-1]), 1)
    return torch.stack((ret.clamp(-10, 10), torch.log1p(true_range / scale).clamp(0, 5),
                        body.clamp(-10, 10), upper.clamp(0, 10), lower.clamp(0, 10),
                        volume_change.clamp(-5, 5)), dim=1)


class StructuralMaskNetwork(nn.Module):
    def __init__(self, C=5, seq=256, model_id='paris-noah/Mantis-8M', model_version=None):
        super().__init__()
        self.encoder = load_mantis(model_id, model_version=model_version, device='cpu')
        hidden = getattr(self.encoder, 'hidden_dim', 256)
        emb = hidden * C
        self.decoder = nn.Sequential(
            nn.Linear(emb, emb), nn.GELU(), nn.Linear(emb, len(STRUCTURAL_TARGET_NAMES) * seq),
        )
        self.C, self.seq = C, seq

    def embed(self, x):
        return encode_independent(self.encoder, x)

    def forward(self, x):
        return self.decoder(self.embed(x)).view(-1, len(STRUCTURAL_TARGET_NAMES), self.seq)


class _StructuralMaskTrainer(BaseTrainer):
    def __init__(self, big, tr, va, *, seq=256, mask_ratio=0.3, span_mean=16.0,
                 span_max=64, feature_anchor_weight=0.1, model_id='paris-noah/Mantis-8M',
                 model_version=None, backbone_ckpt=None, preprocessing=None,
                 compile_model=False, **base):
        super().__init__(big, tr, va, **base)
        if span_mean <= 0 or span_max < 1 or not 0 < mask_ratio < 1:
            raise ValueError('structural masking requires positive spans and mask_ratio in (0,1)')
        self.seq, self.mask_ratio = int(seq), float(mask_ratio)
        self.span_mean, self.span_max = float(span_mean), int(span_max)
        self.feature_anchor_weight = float(feature_anchor_weight)
        if self.feature_anchor_weight < 0:
            raise ValueError('feature_anchor_weight must be nonnegative')
        self.model_id, self.model_version = model_id, model_version
        self.backbone_ckpt, self.compile_model = backbone_ckpt, compile_model
        self.preprocessing = resolve_preprocessing(preprocessing, backbone_ckpt)
        self.C = int(self.big_t.shape[1])
        self._nprng = np.random.default_rng(base.get('seed', 0))

    def build_net(self):
        net = StructuralMaskNetwork(self.C, self.seq, self.model_id, self.model_version).to(self.dev)
        if self.backbone_ckpt:
            load_encoder_checkpoint(net.encoder, self.backbone_ckpt, model_id=self.model_id,
                                    model_version=self.model_version, expected_channels=self.C)
        self.teacher = None
        if self.feature_anchor_weight > 0:
            self.teacher = load_mantis(self.model_id, model_version=self.model_version, device='cpu')
            self.teacher.load_state_dict(net.encoder.state_dict())
            self.teacher = self.teacher.to(self.dev).eval()
            for parameter in self.teacher.parameters():
                parameter.requires_grad = False
        if self.compile_model and hasattr(torch, 'compile'):
            net = torch.compile(net)
        self.net = net

    def make_batch(self, starts):
        indices = self.sample_start_indices(starts)
        raw = _gather_batch(self.big_t, starts, indices, self.seq)
        return raw, structural_targets(raw)

    def compute_loss(self, batch):
        raw, target = batch
        mask = torch.from_numpy(sample_span_mask(
            self._nprng, raw.shape[0], self.seq, self.mask_ratio,
            self.span_mean, self.span_max,
        )).to(raw.device)
        clean = preprocess_windows(raw, self.preprocessing)
        corrupted = torch.where(mask[:, None, :], torch.randn_like(clean), clean)
        prediction = self.net(_apply_control(corrupted, self.control))
        selected = mask[:, None, :].expand_as(prediction)
        loss = F.smooth_l1_loss(prediction[selected].float(), target[selected].float())
        if self.teacher is not None:
            with torch.no_grad():
                teacher = encode_independent(self.teacher, clean)
            loss = loss + self.feature_anchor_weight * F.mse_loss(
                self.net.embed(clean).float(), teacher.float())
        return loss

    @torch.no_grad()
    def val_eval(self):
        self.net.eval(); total = 0.0
        batches = min(self.val_batches or 20, max(1, len(self.va) // self.batch))
        with self.fixed_validation_rng():
            for _ in range(batches):
                with self.amp_ctx():
                    total += float(self.compute_loss(self.make_batch(self.va)))
            raw, _ = self.make_batch(self.va)
            clean = preprocess_windows(raw, self.preprocessing)
            std = float(self.net.embed(clean).std(0).mean())
        self.net.train()
        return total / batches, {'std': std}

    def _resume_signature(self):
        return {**super()._resume_signature(), 'objective': 'structure_mask_v1',
                'seq': self.seq, 'mask_ratio': self.mask_ratio,
                'span_mean': self.span_mean, 'span_max': self.span_max,
                'feature_anchor_weight': self.feature_anchor_weight}


def train_ssl_structure_mask(big, train_starts, val_starts, *, seq=256, mask_ratio=0.3,
                             span_mean=16.0, span_max=64, feature_anchor_weight=0.1,
                             epochs=60, steps_per_epoch=200, batch=256, lr=1e-4,
                             weight_decay=0.05, patience=8, device=None,
                             model_id='paris-noah/Mantis-8M', model_version=None,
                             backbone_ckpt=None, compile_model=False, control='real', seed=0,
                             amp_dtype='fp16', verbose=True, ckpt_path=None, resume=False,
                             freeze_encoder_layers=0, **extra):
    return _StructuralMaskTrainer(
        big, train_starts, val_starts, seq=seq, mask_ratio=mask_ratio,
        span_mean=span_mean, span_max=span_max, feature_anchor_weight=feature_anchor_weight,
        model_id=model_id, model_version=model_version, backbone_ckpt=backbone_ckpt,
        preprocessing=extra.get('preprocessing'), compile_model=compile_model,
        epochs=epochs, steps_per_epoch=steps_per_epoch, batch=batch, lr=lr,
        weight_decay=weight_decay, patience=patience, device=device, seed=seed,
        grad_clip=1.0, amp_dtype=amp_dtype, verbose=verbose, control=control,
        ckpt_path=ckpt_path, resume=resume, freeze_encoder_layers=freeze_encoder_layers,
        train_group_bounds=extra.get('train_group_bounds'),
        val_group_bounds=extra.get('val_group_bounds'), val_batches=extra.get('val_batches'),
    ).fit()
