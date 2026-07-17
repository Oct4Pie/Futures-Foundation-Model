"""Stage-1 masked-modeling trainer: mask a fraction of bars, reconstruct them from context (MSE
on masked positions). REAL/SHUFFLE/RANDOM controls are meaningful — REAL reconstructs from
temporal context; time-scrambled / noise inputs cannot."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..spans import sample_span_mask
from .common import (_apply_control, _gather_batch, BaseTrainer, encode_independent,
                     load_encoder_checkpoint, preprocess_windows, resolve_preprocessing)
from .backbone import load_mantis


class MaskNetwork(nn.Module):
    """Channel-independent Mantis + a light cross-channel reconstruction decoder."""

    def __init__(self, C=5, new_channels=8, seq=64, model_id='paris-noah/Mantis-8M',
                 model_version=None):
        super().__init__()
        self.encoder = load_mantis(model_id, model_version=model_version, device='cpu')
        hidden = getattr(self.encoder, 'hidden_dim', 256)
        self.new_c = C                 # new_channels retained only for legacy config compatibility
        self.C, self.seq = C, seq
        emb = hidden * self.new_c
        self.decoder = nn.Sequential(nn.Linear(emb, emb), nn.GELU(), nn.Linear(emb, C * seq))

    def embed(self, x):                                   # [B, C, seq] -> [B, new_c*hidden]
        return encode_independent(self.encoder, x)

    def forward(self, x):                                 # masked [B,C,seq] -> recon [B,C,seq]
        return self.decoder(self.embed(x)).view(-1, self.C, self.seq)


class _MaskTrainer(BaseTrainer):
    def __init__(self, big, tr, va, *, seq=64, new_channels=8, mask_ratio=0.4, span_mean=0.0,
                 span_max=10, feature_anchor_weight=0.0,
                 model_id='paris-noah/Mantis-8M', backbone_ckpt=None,
                 model_version=None, compile_model=False, preprocessing=None, **base):
        super().__init__(big, tr, va, **base)
        self.seq, self.new_channels, self.mask_ratio = seq, new_channels, mask_ratio
        # SpanBERT move: span_mean>0 masks CONTIGUOUS multi-bar spans instead of scattered bars,
        # so reconstruction must infer a whole missing MOVE from context (trend development), not
        # interpolate a hole from neighbors. 0 = original BERT-style single-bar masking.
        self.span_mean, self.span_max = float(span_mean), int(span_max)
        self.feature_anchor_weight = float(feature_anchor_weight)
        if self.feature_anchor_weight < 0:
            raise ValueError('feature_anchor_weight must be nonnegative')
        self._nprng = np.random.default_rng(base.get('seed', 0))
        self.model_id, self.model_version = model_id, model_version
        self.preprocessing = resolve_preprocessing(preprocessing, backbone_ckpt)
        self.backbone_ckpt, self.compile_model = backbone_ckpt, compile_model
        self.C = int(self.big_t.shape[1])

    def build_net(self):
        net = MaskNetwork(C=self.C, new_channels=self.new_channels, seq=self.seq,
                          model_id=self.model_id, model_version=self.model_version).to(self.dev)
        if self.backbone_ckpt:
            load_encoder_checkpoint(net.encoder, self.backbone_ckpt, model_id=self.model_id,
                                    model_version=self.model_version, expected_channels=self.C)
        # Frozen copy of the representation at stage entry. It is deliberately outside ``net``
        # so it never enters the optimizer or deployment/training checkpoint contracts.
        self.teacher = None
        if self.feature_anchor_weight > 0:
            self.teacher = load_mantis(
                self.model_id, model_version=self.model_version, device='cpu')
            self.teacher.load_state_dict(net.encoder.state_dict())
            self.teacher = self.teacher.to(self.dev).eval()
            for parameter in self.teacher.parameters():
                parameter.requires_grad = False
        if self.compile_model and hasattr(torch, 'compile'):
            net = torch.compile(net)
        self.net = net

    def make_batch(self, starts):
        b_idx = self.sample_start_indices(starts)
        w = _gather_batch(self.big_t, starts, b_idx, self.seq)        # [B,C,seq] raw
        return preprocess_windows(_apply_control(w, self.control), self.preprocessing)

    def compute_loss(self, w):
        if self.span_mean > 0:                                       # SpanBERT: contiguous spans
            m = torch.from_numpy(sample_span_mask(
                self._nprng, w.shape[0], self.seq, self.mask_ratio,
                self.span_mean, self.span_max)).to(w.device)
        else:                                                        # BERT-style single-bar
            m = torch.rand(w.shape[0], self.seq, device=self.dev, generator=self.gen) < self.mask_ratio
            none = ~m.any(1); m[none, 0] = True                      # >=1 masked bar per sample
        me = m[:, None, :].expand_as(w)
        corrupted = torch.where(me, torch.randn_like(w), w)          # noise-fill masked bars
        diff = (self.net(corrupted) - w) ** 2
        loss = diff[me].mean()                                       # masked reconstruction
        if self.teacher is not None:
            with torch.no_grad():
                target_embedding = encode_independent(self.teacher, w)
            clean_embedding = self.net.embed(w)
            loss = loss + self.feature_anchor_weight * F.mse_loss(
                clean_embedding.float(), target_embedding.float())
        return loss

    @torch.no_grad()
    def val_eval(self):
        self.net.eval(); tot = 0.0
        nb = min(self.val_batches or 20, max(1, len(self.va) // self.batch))
        with self.fixed_validation_rng():
            for _ in range(nb):
                with self.amp_ctx():
                    tot += float(self.compute_loss(self.make_batch(self.va)))
            estd = float(self.net.embed(self.make_batch(self.va)).std(0).mean())
        self.net.train()
        return tot / nb, {'std': estd}


def train_ssl_mask(big, train_starts, val_starts, *, seq=64, new_channels=8, mask_ratio=0.4,
                   span_mean=0.0, span_max=10, feature_anchor_weight=0.0,
                   epochs=60, steps_per_epoch=200, batch=512, lr=1e-4,
                   weight_decay=0.05, patience=8, device=None, model_id='paris-noah/Mantis-8M',
                   model_version=None, backbone_ckpt=None, compile_model=False, control='real', seed=0,
                   amp_dtype='fp16', verbose=True, ckpt_path=None, resume=False,
                   freeze_encoder_layers=0, train_group_bounds=None, val_group_bounds=None,
                   preprocessing=None, **_ignore):
    """BERT-style masked modeling (span_mean>0 = SpanBERT-style contiguous-span reconstruction).
    Returns (best_encoder_state, history) with 'val_loss' (recon MSE) + 'std' (collapse guard)."""
    return _MaskTrainer(big, train_starts, val_starts, seq=seq, new_channels=new_channels,
                        mask_ratio=mask_ratio, span_mean=span_mean, span_max=span_max,
                        feature_anchor_weight=feature_anchor_weight,
                        model_id=model_id, model_version=model_version, backbone_ckpt=backbone_ckpt,
                        preprocessing=preprocessing,
                        compile_model=compile_model, epochs=epochs, steps_per_epoch=steps_per_epoch,
                        batch=batch, lr=lr, weight_decay=weight_decay, patience=patience,
                        device=device, seed=seed, grad_clip=None, amp_dtype=amp_dtype,
                        verbose=verbose, control=control, ckpt_path=ckpt_path, resume=resume,
                        freeze_encoder_layers=freeze_encoder_layers,
                        train_group_bounds=train_group_bounds,
                        val_group_bounds=val_group_bounds,
                        val_batches=_ignore.get('val_batches')).fit()
