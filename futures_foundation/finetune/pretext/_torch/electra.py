"""ELECTRA-style replaced-candle-detection (RTD) trainer — the torch half of pretext/electra.py.

Two networks, non-adversarial:
  GENERATOR (weak, thrown away)  — small causal-ish conv stack that fills noise-masked bars with
                                   plausible candles. Deliberately small (ELECTRA: a strong
                                   generator makes fakes undetectable; a weak one keeps the task
                                   learnable). Trained by masked-recon MSE only — never to fool.
  DISCRIMINATOR (the foundation) — Mantis encoder (+ adapter) + a per-bar head that labels EVERY
                                   bar real(0)/replaced(1). Trained by BCE over ALL bars — the
                                   every-bar signal is the sample-efficiency win.

loss = gen_recon_mse(masked) + rtd_weight*bce(all bars) + recon_weight*enc_recon_mse(clean window).
Fake candles are detached (generator gets no gradient from the discriminator) and OHLC-clamped in
RAW space (mu/sd un-standardize -> clamp H>=body_hi, L<=body_lo -> re-standardize) so "impossible
candle" is never the tell.

ENCODER-SIDE JOINT LOSS (v2 fix — the load-bearing anchor). v1 (pure RTD) FAILED: the encoder's
ONLY gradient was the BCE, so a reconstruction-lineage backbone fine-tuned on pure discrimination
DRIFTED (emb_std 1.0->2.35) and lost the base's trend signal (-12pt @1/d). Fix: give the ENCODER
its own reconstruction head (hidden states -> the ORIGINAL uncorrupted OHLCV window) and add
recon_weight * enc_recon_mse to the loss, so the gradient pulling the encoder toward the physical
data is as strong as the one pushing it to discriminate. This is a DENOISING objective (reconstruct
clean from corrupted) that keeps the forecast/reconstruction lineage while RTD adds discrimination.
Ablation knobs: recon_weight=0 -> v1 pure-RTD (the failed control); rtd_weight=0 -> denoising-AE
only (isolates whether RTD adds anything over the anchor).

Diagnostics (per epoch, in history): rtd_bal_acc — BALANCED accuracy, the honest learning signal
(a lazy all-real predictor scores 0.5 here, vs 85% raw acc at mask_ratio=0.15). Target band
~0.60-0.95: ~0.5 = not learning / generator too strong; ~1.0 = fakes trivially detectable (weak
generator or a shortcut tell) => tune gen_width/mask_ratio. fake_recall/real_acc split the two
error modes; gen_mse tracks fake plausibility (the generator-strength knob); std guards collapse.
Anti-cheat: OHLC clamp (no impossible-candle tell), BCE pos_weight (loss can't be gamed by
all-real), detached fakes (no adversarial loop). Ship gate stays downstream (WR@3R + probes).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..spans import sample_span_mask
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
        # ENCODER RECONSTRUCTION head (v2 anchor) — pooled embedding -> the ORIGINAL uncorrupted
        # [C, seq] window. This is what gives the ENCODER a reconstruction gradient (v1 had none),
        # keeping it anchored to the physical data while it learns to discriminate.
        self.recon = nn.Sequential(nn.Linear(emb, emb), nn.GELU(), nn.Linear(emb, C * seq))

    def embed(self, x):                                   # [B, C, seq] -> [B, new_c*hidden]
        a = self.adapter(x)
        return torch.cat([_enc(self.encoder, a[:, [i], :]) for i in range(a.shape[1])], dim=-1)

    def forward(self, x):                                 # corrupted [B,C,seq] -> rtd logits [B,seq]
        return self.disc(self.embed(x))

    def heads(self, x):
        """corrupted [B,C,seq] -> (rtd_logits [B,seq], enc_recon [B,C,seq]) from ONE encoder pass —
        both heads share the embedding so the joint loss is a single forward."""
        emb = self.embed(x)
        return self.disc(emb), self.recon(emb).view(-1, self.C, self.seq)


class _ElectraTrainer(BaseTrainer):
    def __init__(self, big, tr, va, *, seq=64, new_channels=8, mask_ratio=0.15, rtd_weight=5.0,
                 recon_weight=1.0, gen_width=48, span_mean=0.0, span_max=10,
                 model_id='paris-noah/Mantis-8M',
                 backbone_ckpt=None, compile_model=False, **base):
        super().__init__(big, tr, va, **base)
        self.seq, self.new_channels = seq, new_channels
        self.mask_ratio, self.rtd_weight, self.gen_width = mask_ratio, rtd_weight, gen_width
        self.recon_weight = float(recon_weight)            # lambda on the encoder-recon anchor (v2)
        # span-ELECTRA (SpanBERT move): span_mean>0 = corrupt CONTIGUOUS multi-bar spans instead
        # of scattered single bars — the generator must fake a plausible move, the encoder must
        # detect the fake SPAN (models development-over-bars). 0 = original bar mode.
        self.span_mean, self.span_max = float(span_mean), int(span_max)
        self._nprng = np.random.default_rng(base.get('seed', 0))       # span sampler (CPU, cheap)
        self.model_id, self.backbone_ckpt, self.compile_model = model_id, backbone_ckpt, compile_model
        self.C = int(self.big_t.shape[1])
        self._last_rtd = {'rtd_bal_acc': float('nan'), 'fake_recall': float('nan'),
                          'real_acc': float('nan'), 'enc_recon': float('nan')}

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
        if self.span_mean > 0:                                         # span-ELECTRA (SpanBERT)
            m = torch.from_numpy(sample_span_mask(
                self._nprng, B, self.seq, self.mask_ratio,
                self.span_mean, self.span_max)).to(w.device)
        else:                                                          # original bar mode
            m = torch.rand(B, self.seq, device=self.dev, generator=self.gen) < self.mask_ratio
            none = ~m.any(1); m[none, 0] = True                        # >=1 masked bar per sample
        me = m[:, None, :].expand_as(w)
        # 1) generator fills the noise-masked bars; trained by masked-recon MSE only
        gen_out = net.gen(torch.where(me, torch.randn_like(w), w))
        recon = ((gen_out - w) ** 2)[me].mean()
        # 2) plant DETACHED, OHLC-valid fakes at the masked positions
        fake = clamp_valid_ohlc_t(gen_out.detach(), self._mu, self._sd)
        corrupted = torch.where(me, fake, w)
        # 3) ONE encoder pass -> discriminator (real/replaced per bar) + encoder RECONSTRUCTION of
        #    the ORIGINAL uncorrupted window (the v2 anchor). enc_recon is a DENOISING target: from
        #    the corrupted embedding, rebuild the clean candles -> keeps the encoder tied to the data.
        logits, enc_rec = net.heads(corrupted)                         # [B,seq], [B,C,seq]
        # pos_weight balances the 15/85 fake/real imbalance so the LOSS can't be gamed by a lazy
        # all-real predictor (anti-cheat #2; the balanced-acc diagnostic below would expose it,
        # pos_weight makes the gradient itself push toward detecting fakes).
        pw = torch.tensor((1.0 - self.mask_ratio) / max(self.mask_ratio, 1e-3), device=w.device)
        rtd = F.binary_cross_entropy_with_logits(logits, m.float(), pos_weight=pw)
        enc_recon = F.mse_loss(enc_rec, w)                             # encoder-side anchor (vs clean w)
        with torch.no_grad():
            pred = logits > 0
            # BALANCED accuracy — the honest "is it learning" signal. With mask_ratio=0.15 a lazy
            # all-real predictor scores 85% RAW accuracy while learning nothing; balanced acc
            # (mean of fake-recall + real-acc) is 0.5 for that same lazy model.
            fake_rec = float(pred[m].float().mean()) if m.any() else 0.0
            real_acc = float((~pred[~m]).float().mean()) if (~m).any() else 0.0
            self._last_rtd = {'rtd_bal_acc': 0.5 * (fake_rec + real_acc),
                              'fake_recall': fake_rec, 'real_acc': real_acc,
                              'gen_mse': float(recon),     # generator plausibility (strength knob)
                              'enc_recon': float(enc_recon)}   # the ANCHOR (should DROP as it learns)
        return recon + self.rtd_weight * rtd + self.recon_weight * enc_recon

    def log_line(self, ep, tr_loss, vloss, extra, improved):
        # live learning verification: show the RTD diagnostics per epoch (base prints only std)
        if self.verbose:
            print(f"  ep{ep:>3} train={tr_loss:.4f} val={vloss:.4f} "
                  f"bal_acc={extra.get('rtd_bal_acc', float('nan')):.3f} "
                  f"(fake={extra.get('fake_recall', float('nan')):.3f}/"
                  f"real={extra.get('real_acc', float('nan')):.3f}) "
                  f"gen_mse={extra.get('gen_mse', float('nan')):.4f} "
                  f"enc_recon={extra.get('enc_recon', float('nan')):.4f} "
                  f"emb_std={extra.get('std', 0.0):.4f}{'  *' if improved else ''}", flush=True)

    @torch.no_grad()
    def val_eval(self):
        self.net.eval(); tot = 0.0
        agg = {'rtd_bal_acc': 0.0, 'fake_recall': 0.0, 'real_acc': 0.0, 'gen_mse': 0.0,
               'enc_recon': 0.0}
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
                      rtd_weight=5.0, recon_weight=1.0, gen_width=48, span_mean=0.0, span_max=10,
                      epochs=60, steps_per_epoch=200, batch=512,
                      lr=1e-4, weight_decay=0.05, patience=8, device=None,
                      model_id='paris-noah/Mantis-8M', backbone_ckpt=None, compile_model=False,
                      control='real', seed=0, amp_dtype='fp16', verbose=True, ckpt_path=None,
                      resume=False, freeze_encoder_layers=0, **_ignore):
    """ELECTRA-style replaced-candle detection with the v2 ENCODER-SIDE JOINT LOSS. Returns
    (best_encoder_state, history) with 'val_loss' (gen_recon + rtd_weight*bce + recon_weight*
    enc_recon), 'rtd_bal_acc'/'fake_recall'/'real_acc' (balanced-acc learning diagnostics, NOT raw
    acc), 'enc_recon' (the encoder anchor — should DROP as it learns), + 'std'. recon_weight=0 =
    v1 pure-RTD (failed control); rtd_weight=0 = denoising-AE only (RTD-adds-anything ablation)."""
    return _ElectraTrainer(big, train_starts, val_starts, seq=seq, new_channels=new_channels,
                           mask_ratio=mask_ratio, rtd_weight=rtd_weight, recon_weight=recon_weight,
                           gen_width=gen_width, span_mean=span_mean, span_max=span_max,
                           model_id=model_id, backbone_ckpt=backbone_ckpt,
                           compile_model=compile_model, epochs=epochs,
                           steps_per_epoch=steps_per_epoch, batch=batch, lr=lr,
                           weight_decay=weight_decay, patience=patience, device=device, seed=seed,
                           grad_clip=None, amp_dtype=amp_dtype, verbose=verbose, control=control,
                           ckpt_path=ckpt_path, resume=resume,
                           freeze_encoder_layers=freeze_encoder_layers).fit()
