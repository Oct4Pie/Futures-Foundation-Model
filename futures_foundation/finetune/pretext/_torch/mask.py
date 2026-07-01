"""Stage-1 masked-modeling trainer: mask a fraction of bars, reconstruct them from context (MSE
on masked positions). REAL/SHUFFLE/RANDOM controls are meaningful — REAL reconstructs from
temporal context; time-scrambled / noise inputs cannot."""
import torch
import torch.nn as nn

from .common import _enc, _standardize, _apply_control, _gather_batch, BaseTrainer


class MaskNetwork(nn.Module):
    """Mantis encoder + channel adapter + a light reconstruction decoder. Masked OHLCV bars go in;
    the decoder reconstructs the full (standardized) window from the pooled embedding."""

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


class _MaskTrainer(BaseTrainer):
    def __init__(self, big, tr, va, *, seq=64, new_channels=8, mask_ratio=0.4,
                 model_id='paris-noah/Mantis-8M', backbone_ckpt=None, compile_model=False, **base):
        super().__init__(big, tr, va, **base)
        self.seq, self.new_channels, self.mask_ratio = seq, new_channels, mask_ratio
        self.model_id, self.backbone_ckpt, self.compile_model = model_id, backbone_ckpt, compile_model
        self.C = int(self.big_t.shape[1])

    def build_net(self):
        net = MaskNetwork(C=self.C, new_channels=self.new_channels, seq=self.seq,
                          model_id=self.model_id).to(self.dev)
        if self.backbone_ckpt:
            net.encoder.load_state_dict(torch.load(self.backbone_ckpt, map_location='cpu'))
        if self.compile_model and hasattr(torch, 'compile'):
            net = torch.compile(net)
        self.net = net

    def make_batch(self, starts):
        b_idx = torch.randint(0, len(starts), (self.batch,), device=self.dev, generator=self.gen)
        w = _gather_batch(self.big_t, starts, b_idx, self.seq)        # [B,C,seq] raw
        return _standardize(_apply_control(w, self.control))         # corrupt input per control, z-score

    def compute_loss(self, w):
        m = torch.rand(w.shape[0], self.seq, device=self.dev, generator=self.gen) < self.mask_ratio
        none = ~m.any(1); m[none, 0] = True                          # >=1 masked bar per sample
        me = m[:, None, :].expand_as(w)
        corrupted = torch.where(me, torch.randn_like(w), w)          # noise-fill masked bars
        diff = (self.net(corrupted) - w) ** 2
        return diff[me].mean()                                       # MSE on masked positions only

    @torch.no_grad()
    def val_eval(self):
        self.net.eval(); tot = 0.0; nb = min(20, max(1, len(self.va) // self.batch))
        for _ in range(nb):
            with self.amp_ctx():
                tot += float(self.compute_loss(self.make_batch(self.va)))
        estd = float(self.net.embed(self.make_batch(self.va)).std(0).mean())
        self.net.train()
        return tot / nb, {'std': estd}


def train_ssl_mask(big, train_starts, val_starts, *, seq=64, new_channels=8, mask_ratio=0.4,
                   epochs=60, steps_per_epoch=200, batch=512, lr=1e-4, weight_decay=0.05,
                   patience=8, device=None, model_id='paris-noah/Mantis-8M', backbone_ckpt=None,
                   compile_model=False, control='real', seed=0, amp_dtype='fp16',
                   verbose=True, **_ignore):
    """BERT-style masked modeling. Returns (best_encoder_state, history) with 'val_loss' (recon MSE)
    + 'std' (embedding std, collapse guard)."""
    return _MaskTrainer(big, train_starts, val_starts, seq=seq, new_channels=new_channels,
                        mask_ratio=mask_ratio, model_id=model_id, backbone_ckpt=backbone_ckpt,
                        compile_model=compile_model, epochs=epochs, steps_per_epoch=steps_per_epoch,
                        batch=batch, lr=lr, weight_decay=weight_decay, patience=patience,
                        device=device, seed=seed, grad_clip=None, amp_dtype=amp_dtype,
                        verbose=verbose, control=control).fit()
