"""ELECTRA-style replaced-candle-detection (RTD) trainer — the torch half of pretext/electra.py.

Two networks, non-adversarial:
  GENERATOR (weak, thrown away)  — small causal-ish conv stack that fills noise-masked bars with
                                   plausible candles. Deliberately small (ELECTRA: a strong
                                   generator makes fakes undetectable; a weak one keeps the task
                                   learnable). Trained by masked-recon MSE only — never to fool.
  DISCRIMINATOR (the foundation) — Mantis encoder (+ adapter) + a per-bar head that labels EVERY
                                   bar real(0)/replaced(1). Trained by BCE over ALL bars — the
                                   every-bar signal is the sample-efficiency win.

loss = recon_mse(masked) + rtd_weight * bce(all bars).  Fake candles are detached (generator gets
no gradient from the discriminator) and OHLC-clamped in RAW space (mu/sd un-standardize -> clamp
H>=body_hi, L<=body_lo -> re-standardize) so "impossible candle" is never the tell.

Diagnostics (per epoch, in history): rtd_bal_acc — BALANCED accuracy, the honest learning signal
(a lazy all-real predictor scores 0.5 here, vs 85% raw acc at mask_ratio=0.15). Target band
~0.60-0.95: ~0.5 = not learning / generator too strong; ~1.0 = fakes trivially detectable (weak
generator or a shortcut tell) => tune gen_width/mask_ratio. fake_recall/real_acc split the two
error modes; gen_mse tracks fake plausibility (the generator-strength knob); std guards collapse.
Anti-cheat: OHLC clamp (no impossible-candle tell), BCE pos_weight (loss can't be gamed by
all-real), detached fakes (no adversarial loop). Ship gate stays downstream (WR@3R + probes).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import _enc, _apply_control, _gather_batch, BaseTrainer


def _standardize_stats(x):
    """Per-window per-channel z-score that RETURNS (z, mu, sd) so generated candles can be
    un-standardized for the raw-space OHLC clamp (common._standardize discards the stats)."""
    mu = x.mean(dim=2, keepdim=True)
    sd = x.std(dim=2, keepdim=True).clamp_min(1e-6)
    return (x - mu) / sd, mu, sd


def clamp_valid_ohlc_t(cand_std, mu, sd):
    """Torch mirror of pretext.electra.clamp_valid_ohlc, operating on STANDARDIZED candles via
    their window stats: un-standardize -> H>=max(O,C,H), L<=min(O,C,L) -> re-standardize.
    cand_std/mu/sd: [B, C, seq]; channels (O,H,L,C[,V])."""
    raw = cand_std * sd + mu
    o, h, l, c = raw[:, 0], raw[:, 1], raw[:, 2], raw[:, 3]
    body_hi = torch.maximum(o, c)
    body_lo = torch.minimum(o, c)
    raw = raw.clone()
    raw[:, 1] = torch.maximum(h, body_hi)
    raw[:, 2] = torch.minimum(l, body_lo)
    return (raw - mu) / sd


class ElectraNetwork(nn.Module):
    """Weak conv generator + Mantis discriminator (encoder + adapter + per-bar RTD head).
    `.encoder` is the Mantis backbone (BaseTrainer freezes/saves exactly that attribute)."""

    def __init__(self, C=5, new_channels=8, seq=64, gen_width=48,
                 model_id='paris-noah/Mantis-8M'):
        super().__init__()
        from mantis.architecture import Mantis8M
        from mantis.adapters import LinearChannelCombiner
        self.encoder = Mantis8M.from_pretrained(model_id)
        hidden = getattr(self.encoder, 'hidden_dim', 256)
        self.new_c = min(new_channels, C)
        self.adapter = LinearChannelCombiner(num_channels=C, new_num_channels=self.new_c)
        self.C, self.seq = C, seq
        emb = hidden * self.new_c
        # GENERATOR — small on purpose (weak): 3-layer conv over the noise-masked window
        self.gen = nn.Sequential(
            nn.Conv1d(C, gen_width, 5, padding=2), nn.GELU(),
            nn.Conv1d(gen_width, gen_width, 5, padding=2), nn.GELU(),
            nn.Conv1d(gen_width, C, 3, padding=1))
        # DISCRIMINATOR head — pooled embedding -> per-bar real/fake logits (decoder-style)
        self.disc = nn.Sequential(nn.Linear(emb, emb), nn.GELU(), nn.Linear(emb, seq))

    def embed(self, x):                                   # [B, C, seq] -> [B, new_c*hidden]
        a = self.adapter(x)
        return torch.cat([_enc(self.encoder, a[:, [i], :]) for i in range(a.shape[1])], dim=-1)

    def forward(self, x):                                 # corrupted [B,C,seq] -> rtd logits [B,seq]
        return self.disc(self.embed(x))


class _ElectraTrainer(BaseTrainer):
    def __init__(self, big, tr, va, *, seq=64, new_channels=8, mask_ratio=0.15, rtd_weight=5.0,
                 gen_width=48, model_id='paris-noah/Mantis-8M', backbone_ckpt=None,
                 compile_model=False, **base):
        super().__init__(big, tr, va, **base)
        self.seq, self.new_channels = seq, new_channels
        self.mask_ratio, self.rtd_weight, self.gen_width = mask_ratio, rtd_weight, gen_width
        self.model_id, self.backbone_ckpt, self.compile_model = model_id, backbone_ckpt, compile_model
        self.C = int(self.big_t.shape[1])
        self._last_rtd = {'rtd_bal_acc': float('nan'), 'fake_recall': float('nan'),
                          'real_acc': float('nan')}

    def build_net(self):
        net = ElectraNetwork(C=self.C, new_channels=self.new_channels, seq=self.seq,
                             gen_width=self.gen_width, model_id=self.model_id).to(self.dev)
        if self.backbone_ckpt:                            # warm = the promoted base (lineage kept)
            net.encoder.load_state_dict(torch.load(self.backbone_ckpt, map_location='cpu'))
        if self.compile_model and hasattr(torch, 'compile'):
            net = torch.compile(net)
        self.net = net

    def make_batch(self, starts):
        b_idx = torch.randint(0, len(starts), (self.batch,), device=self.dev, generator=self.gen)
        w = _gather_batch(self.big_t, starts, b_idx, self.seq)         # [B,C,seq] raw
        z, mu, sd = _standardize_stats(_apply_control(w, self.control))
        self._mu, self._sd = mu, sd                       # window stats for the raw-space clamp
        return z

    def compute_loss(self, w):
        net = self.net if not hasattr(self.net, '_orig_mod') else self.net._orig_mod
        B = w.shape[0]
        m = torch.rand(B, self.seq, device=self.dev, generator=self.gen) < self.mask_ratio
        none = ~m.any(1); m[none, 0] = True                            # >=1 masked bar per sample
        me = m[:, None, :].expand_as(w)
        # 1) generator fills the noise-masked bars; trained by masked-recon MSE only
        gen_out = net.gen(torch.where(me, torch.randn_like(w), w))
        recon = ((gen_out - w) ** 2)[me].mean()
        # 2) plant DETACHED, OHLC-valid fakes at the masked positions
        fake = clamp_valid_ohlc_t(gen_out.detach(), self._mu, self._sd)
        corrupted = torch.where(me, fake, w)
        # 3) discriminator labels EVERY bar real/replaced (the every-bar signal)
        logits = self.net(corrupted)                                   # [B, seq]
        # pos_weight balances the 15/85 fake/real imbalance so the LOSS can't be gamed by a lazy
        # all-real predictor (anti-cheat #2; the balanced-acc diagnostic below would expose it,
        # pos_weight makes the gradient itself push toward detecting fakes).
        pw = torch.tensor((1.0 - self.mask_ratio) / max(self.mask_ratio, 1e-3), device=w.device)
        rtd = F.binary_cross_entropy_with_logits(logits, m.float(), pos_weight=pw)
        with torch.no_grad():
            pred = logits > 0
            # BALANCED accuracy — the honest "is it learning" signal. With mask_ratio=0.15 a lazy
            # all-real predictor scores 85% RAW accuracy while learning nothing; balanced acc
            # (mean of fake-recall + real-acc) is 0.5 for that same lazy model.
            fake_rec = float(pred[m].float().mean()) if m.any() else 0.0
            real_acc = float((~pred[~m]).float().mean()) if (~m).any() else 0.0
            self._last_rtd = {'rtd_bal_acc': 0.5 * (fake_rec + real_acc),
                              'fake_recall': fake_rec, 'real_acc': real_acc,
                              'gen_mse': float(recon)}   # generator plausibility (strength knob)
        return recon + self.rtd_weight * rtd

    @torch.no_grad()
    def val_eval(self):
        self.net.eval(); tot = 0.0
        agg = {'rtd_bal_acc': 0.0, 'fake_recall': 0.0, 'real_acc': 0.0, 'gen_mse': 0.0}
        nb = min(20, max(1, len(self.va) // self.batch))
        for _ in range(nb):
            with self.amp_ctx():
                tot += float(self.compute_loss(self.make_batch(self.va)))
            for k in agg:
                agg[k] += self._last_rtd[k]
        net = self.net if not hasattr(self.net, '_orig_mod') else self.net._orig_mod
        estd = float(net.embed(self.make_batch(self.va)).std(0).mean())
        self.net.train()
        return tot / nb, {'std': estd, **{k: v / nb for k, v in agg.items()}}


def train_ssl_electra(big, train_starts, val_starts, *, seq=64, new_channels=8, mask_ratio=0.15,
                      rtd_weight=5.0, gen_width=48, epochs=60, steps_per_epoch=200, batch=512,
                      lr=1e-4, weight_decay=0.05, patience=8, device=None,
                      model_id='paris-noah/Mantis-8M', backbone_ckpt=None, compile_model=False,
                      control='real', seed=0, amp_dtype='fp16', verbose=True, ckpt_path=None,
                      resume=False, freeze_encoder_layers=0, **_ignore):
    """ELECTRA-style replaced-candle detection. Returns (best_encoder_state, history) with
    'val_loss' (recon + rtd_weight*bce), 'rtd_bal_acc'/'fake_recall'/'real_acc' (learning
    diagnostics — balanced acc, NOT raw acc: a lazy all-real predictor is 0.5 here) + 'std'."""
    return _ElectraTrainer(big, train_starts, val_starts, seq=seq, new_channels=new_channels,
                           mask_ratio=mask_ratio, rtd_weight=rtd_weight, gen_width=gen_width,
                           model_id=model_id, backbone_ckpt=backbone_ckpt,
                           compile_model=compile_model, epochs=epochs,
                           steps_per_epoch=steps_per_epoch, batch=batch, lr=lr,
                           weight_decay=weight_decay, patience=patience, device=device, seed=seed,
                           grad_clip=None, amp_dtype=amp_dtype, verbose=verbose, control=control,
                           ckpt_path=ckpt_path, resume=resume,
                           freeze_encoder_layers=freeze_encoder_layers).fit()
