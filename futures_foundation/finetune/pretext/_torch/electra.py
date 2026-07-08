"""TURN-ELECTRA trainer — the torch half of pretext/electra.py (replaced-TURN detection).

Two networks, non-adversarial (the proven ELECTRA mechanics, corruption re-aimed at the swings):
  GENERATOR (weak, thrown away)  — small conv stack that fills the masked TURN-regions with
                                   plausible candles. Deliberately small (a strong generator makes
                                   fakes undetectable; a weak one keeps the task learnable).
                                   Trained by masked-recon MSE only — never to fool.
  DISCRIMINATOR (the foundation) — Mantis encoder (+ adapter) + a per-bar head that labels EVERY
                                   bar real(0)/replaced(1). Trained by BCE over ALL bars.

WHERE the corruption lands is the whole point: spans are CENTERED ON DETECTED TURNS (local swing
highs/lows via pretext.spans.local_turns — the event a pivot entry trades) with prob turn_bias,
uniform otherwise. A generator-filled turn = a SYNTHETIC FAKE TURN; discriminating it from the real
development teaches the encoder how genuine reversals form — the fakeout-vs-real skill — as pure SSL.

loss = gen_recon_mse(masked) + rtd_weight*bce(all bars) + recon_weight*enc_recon_mse(clean window).
Fakes are detached (no adversarial loop) and OHLC-clamped in RAW space (mu/sd un-standardize ->
clamp -> re-standardize) so "impossible candle" is never the tell. The ENCODER-SIDE recon anchor
(recon_weight) keeps the embedding tied to the physical data while it learns to discriminate — the
one piece that held emb_std ~1 across every discriminative run. Diagnostics per epoch: rtd_bal_acc
(BALANCED acc — a lazy all-real predictor scores 0.5), fake_recall/real_acc (the two error modes),
turn_cov (fraction of masked bars near a turn — how turn-focused the corruption really is), gen_mse
(generator plausibility), enc_recon (the anchor — should DROP), std (collapse guard).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..spans import sample_turn_span_mask
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
    """Weak conv generator + Mantis discriminator (encoder + adapter + per-bar RTD head + encoder
    recon anchor head). `.encoder` is the Mantis backbone (BaseTrainer freezes/saves exactly that)."""

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
        # DISCRIMINATOR head — pooled embedding -> per-bar real/replaced logits
        self.disc = nn.Sequential(nn.Linear(emb, emb), nn.GELU(), nn.Linear(emb, seq))
        # ENCODER RECONSTRUCTION anchor — pooled embedding -> the ORIGINAL uncorrupted [C, seq]
        # window: the reconstruction gradient that keeps the encoder tied to the physical data
        # while it learns to discriminate (without it, pure discrimination drifts the embedding).
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


class _TurnElectraTrainer(BaseTrainer):
    def __init__(self, big, tr, va, *, seq=64, new_channels=8, mask_ratio=0.2, rtd_weight=5.0,
                 recon_weight=1.0, gen_width=48, span_mean=4.0, span_max=10, turn_w=3,
                 turn_bias=0.85, model_id='paris-noah/Mantis-8M',
                 backbone_ckpt=None, compile_model=False, **base):
        super().__init__(big, tr, va, **base)
        self.seq, self.new_channels = seq, new_channels
        self.mask_ratio, self.rtd_weight, self.gen_width = mask_ratio, rtd_weight, gen_width
        self.recon_weight = float(recon_weight)            # lambda on the encoder-recon anchor
        # span_mean is a SHARED cfg knob whose mask-pretext default is 0 (bar mode); turn-electra is
        # span-native, so 0/negative coerces to the span default (4) instead of degenerate 1-bar spans.
        self.span_mean = float(span_mean) if float(span_mean) > 0 else 4.0
        self.span_max = int(span_max)
        self.turn_w, self.turn_bias = int(turn_w), float(turn_bias)
        self._nprng = np.random.default_rng(base.get('seed', 0))   # turn-span sampler (CPU, cheap)
        self.model_id, self.backbone_ckpt, self.compile_model = model_id, backbone_ckpt, compile_model
        self.C = int(self.big_t.shape[1])
        self._last = {'rtd_bal_acc': float('nan'), 'fake_recall': float('nan'),
                      'real_acc': float('nan'), 'gen_mse': float('nan'),
                      'enc_recon': float('nan'), 'turn_cov': float('nan')}

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
        w = _apply_control(_gather_batch(self.big_t, starts, b_idx, self.seq), self.control)
        # TURN-BIASED span mask from the window's own H/L (extrema are invariant under the
        # per-window standardize, so placement on raw == placement on z). CPU numpy is trivial
        # here (512x64 floats) and keeps ONE tested sampler implementation — no torch fork.
        hl = w[:, 1:3, :].detach().cpu().numpy()
        m, self._turn_cov = sample_turn_span_mask(
            self._nprng, hl[:, 0], hl[:, 1], self.mask_ratio, self.span_mean, self.span_max,
            turn_w=self.turn_w, turn_bias=self.turn_bias)
        self._mask = torch.from_numpy(m).to(w.device)
        z, mu, sd = _standardize_stats(w)
        self._mu, self._sd = mu, sd                       # window stats for the raw-space clamp
        return z

    def compute_loss(self, w):
        net = self.net if not hasattr(self.net, '_orig_mod') else self.net._orig_mod
        m = self._mask
        me = m[:, None, :].expand_as(w)
        # 1) generator fills the noise-masked TURN regions; trained by masked-recon MSE only
        gen_out = net.gen(torch.where(me, torch.randn_like(w), w))
        recon = ((gen_out - w) ** 2)[me].mean()
        # 2) plant DETACHED, OHLC-valid fakes at the masked turns -> SYNTHETIC FAKE TURNS
        fake = clamp_valid_ohlc_t(gen_out.detach(), self._mu, self._sd)
        corrupted = torch.where(me, fake, w)
        # 3) ONE encoder pass -> per-bar real/replaced logits + encoder RECONSTRUCTION of the
        #    ORIGINAL uncorrupted window (the anchor that keeps the embedding on the data).
        logits, enc_rec = net.heads(corrupted)                         # [B,seq], [B,C,seq]
        # pos_weight balances the masked/unmasked imbalance so the LOSS can't be gamed by a lazy
        # all-real predictor (the balanced-acc diagnostic would expose it; pos_weight makes the
        # gradient itself push toward detecting the fake turns).
        pw = torch.tensor((1.0 - self.mask_ratio) / max(self.mask_ratio, 1e-3), device=w.device)
        rtd = F.binary_cross_entropy_with_logits(logits, m.float(), pos_weight=pw)
        enc_recon = F.mse_loss(enc_rec, w)                             # anchor (vs the clean window)
        with torch.no_grad():
            pred = logits > 0
            fake_rec = float(pred[m].float().mean()) if bool(m.any()) else 0.0
            real_acc = float((~pred[~m]).float().mean()) if bool((~m).any()) else 0.0
            self._last = {'rtd_bal_acc': 0.5 * (fake_rec + real_acc),
                          'fake_recall': fake_rec, 'real_acc': real_acc,
                          'gen_mse': float(recon),         # generator plausibility (strength knob)
                          'enc_recon': float(enc_recon),   # the ANCHOR (should DROP as it learns)
                          'turn_cov': float(self._turn_cov)}   # how turn-focused the masking is
        return recon + self.rtd_weight * rtd + self.recon_weight * enc_recon

    def log_line(self, ep, tr_loss, vloss, extra, improved):
        if self.verbose:
            print(f"  ep{ep:>3} train={tr_loss:.4f} val={vloss:.4f} "
                  f"bal_acc={extra.get('rtd_bal_acc', float('nan')):.3f} "
                  f"(fake={extra.get('fake_recall', float('nan')):.3f}/"
                  f"real={extra.get('real_acc', float('nan')):.3f}) "
                  f"turn_cov={extra.get('turn_cov', float('nan')):.2f} "
                  f"gen_mse={extra.get('gen_mse', float('nan')):.4f} "
                  f"enc_recon={extra.get('enc_recon', float('nan')):.4f} "
                  f"emb_std={extra.get('std', 0.0):.4f}{'  *' if improved else ''}", flush=True)

    @torch.no_grad()
    def val_eval(self):
        self.net.eval()
        tot = 0.0
        agg = {'rtd_bal_acc': 0.0, 'fake_recall': 0.0, 'real_acc': 0.0, 'gen_mse': 0.0,
               'enc_recon': 0.0, 'turn_cov': 0.0}
        nb = min(20, max(1, len(self.va) // self.batch))
        for _ in range(nb):
            with self.amp_ctx():
                tot += float(self.compute_loss(self.make_batch(self.va)))
            for k in agg:
                agg[k] += self._last[k]
        net = self.net if not hasattr(self.net, '_orig_mod') else self.net._orig_mod
        estd = float(net.embed(self.make_batch(self.va)).std(0).mean())
        self.net.train()
        return tot / nb, {'std': estd, **{k: v / nb for k, v in agg.items()}}


def train_ssl_electra(big, train_starts, val_starts, *, seq=64, new_channels=8, mask_ratio=0.2,
                      rtd_weight=5.0, recon_weight=1.0, gen_width=48, span_mean=4.0, span_max=10,
                      turn_w=3, turn_bias=0.85, epochs=60, steps_per_epoch=200, batch=512,
                      lr=1e-4, weight_decay=0.05, patience=8, device=None,
                      model_id='paris-noah/Mantis-8M', backbone_ckpt=None, compile_model=False,
                      control='real', seed=0, amp_dtype='fp16', verbose=True, ckpt_path=None,
                      resume=False, freeze_encoder_layers=0, **_ignore):
    """TURN-ELECTRA: replaced-TURN detection (spans centered on detected swings, turn_bias of the
    time). Returns (best_encoder_state, history) with 'val_loss' (gen_recon + rtd_weight*bce +
    recon_weight*enc_recon), 'rtd_bal_acc'/'fake_recall'/'real_acc' (balanced-acc diagnostics),
    'turn_cov' (masked-bars-near-a-turn fraction), 'enc_recon' (anchor — should DROP), + 'std'.
    turn_bias=0 = uniform span-ELECTRA (the ablation: does TURN placement earn the lift?);
    recon_weight=0 = pure discrimination (drift risk); rtd_weight=0 = denoising-AE only."""
    return _TurnElectraTrainer(big, train_starts, val_starts, seq=seq, new_channels=new_channels,
                               mask_ratio=mask_ratio, rtd_weight=rtd_weight,
                               recon_weight=recon_weight, gen_width=gen_width, span_mean=span_mean,
                               span_max=span_max, turn_w=turn_w, turn_bias=turn_bias,
                               model_id=model_id, backbone_ckpt=backbone_ckpt,
                               compile_model=compile_model, epochs=epochs,
                               steps_per_epoch=steps_per_epoch, batch=batch, lr=lr,
                               weight_decay=weight_decay, patience=patience, device=device,
                               seed=seed, grad_clip=None, amp_dtype=amp_dtype, verbose=verbose,
                               control=control, ckpt_path=ckpt_path, resume=resume,
                               freeze_encoder_layers=freeze_encoder_layers).fit()
