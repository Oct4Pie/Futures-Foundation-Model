"""BREAK-HOLD discriminative trainer — the torch half of pretext/electra.py (the rewritten
discriminative slot). GENERATOR-FREE: the labels are REAL break outcomes, not synthesized candles.

Per parent window (seq + hold_k bars): the encoder sees the standardized [C, seq] window (anchor =
the last bar, strictly causal); a per-WINDOW head predicts whether the anchor's structural break
HOLDS or FAILS over the hold_k RESERVED future bars (which the encoder NEVER sees). Only break
windows carry the BCE; ALL windows carry the encoder-recon anchor (the electra-v2 piece that held
emb_std ~1 while the encoder learned to discriminate). No generator, no fakes, no OHLC clamp — the
discrimination target is real market structure.

    loss = recon_weight * enc_recon_mse(clean seq window) + hold_weight * BCE(hold_logit | is_break)

Warm from the promoted base (ctr_seq2seq) so the reconstruction lineage is kept. Ship gate stays
downstream (WR@3R one-shot 2026). Diagnostics per epoch: hold_bal_acc (BALANCED accuracy over break
windows — a lazy predictor scores 0.5), hold_recall / fail_recall (the two error modes), break_rate
(fraction of windows that are breaks), hold_rate (fraction of breaks that held), enc_recon (the
anchor — should DROP), std (collapse guard). This module is imported lazily (torch under _torch/).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import _enc, _apply_control, _standardize, _gather_batch, BaseTrainer

_O, _H, _L, _C = 0, 1, 2, 3


class BreakHoldNetwork(nn.Module):
    """Mantis encoder (+ channel adapter) + a per-window HOLD head and an encoder-RECON anchor head.
    `.encoder` is the Mantis backbone (BaseTrainer freezes/saves exactly that attribute)."""

    def __init__(self, C=5, new_channels=3, seq=64, model_id='paris-noah/Mantis-8M'):
        super().__init__()
        from mantis.architecture import Mantis8M
        from mantis.adapters import LinearChannelCombiner
        self.encoder = Mantis8M.from_pretrained(model_id)
        hidden = getattr(self.encoder, 'hidden_dim', 256)
        self.new_c = min(new_channels, C)
        self.adapter = LinearChannelCombiner(num_channels=C, new_num_channels=self.new_c)
        self.C, self.seq = C, seq
        emb = hidden * self.new_c
        # HOLD head — pooled embedding -> a single real/fake (hold/fail) logit per WINDOW.
        self.hold = nn.Sequential(nn.Linear(emb, emb), nn.GELU(), nn.Linear(emb, 1))
        # ENCODER RECONSTRUCTION anchor — pooled embedding -> the clean [C, seq] window. Gives the
        # encoder a reconstruction gradient so it stays tied to the physical data while it learns to
        # discriminate (the electra-v2 lesson: without it, pure discrimination drifts emb_std off).
        self.recon = nn.Sequential(nn.Linear(emb, emb), nn.GELU(), nn.Linear(emb, C * seq))

    def embed(self, x):                                    # [B, C, seq] -> [B, new_c*hidden]
        a = self.adapter(x)
        return torch.cat([_enc(self.encoder, a[:, [i], :]) for i in range(a.shape[1])], dim=-1)

    def heads(self, x):
        """standardized [B,C,seq] -> (hold_logit [B], enc_recon [B,C,seq]) from ONE encoder pass."""
        emb = self.embed(x)
        return self.hold(emb).squeeze(-1), self.recon(emb).view(-1, self.C, self.seq)


def break_hold_labels(w, fut, lookback, theta):
    """VECTORIZED break-hold labels for a batch. Torch mirror of pretext.electra.detect_break +
    label_hold (unit-tested against them). Strictly causal: break state from the seq window, HOLD/FAIL
    from the disjoint future bars.
      w    [B, C, seq]     — the encoder window (RAW); anchor = last bar (seq-1)
      fut  [B, C, k]       — the k reserved future bars (RAW), never encoded
    Returns (label [B] float {0,1}, is_break [B] bool). atr = mean bar range over the seq window."""
    B, _, seq = w.shape
    k = fut.shape[2]
    anchor = seq - 1
    lo = max(0, anchor - int(lookback))
    prior_hi = w[:, _H, lo:anchor].max(dim=1).values          # [B] swing high before the anchor
    prior_lo = w[:, _L, lo:anchor].min(dim=1).values          # [B] swing low
    c_anchor = w[:, _C, anchor]                               # [B]
    up = c_anchor > prior_hi
    down = c_anchor < prior_lo
    is_break = up | down
    direction = up.float() - down.float()                    # +1 / -1 / 0
    level = torch.where(up, prior_hi, prior_lo)              # broken swing extreme (only meaningful if break)
    atr = (w[:, _H, :] - w[:, _L, :]).mean(dim=1).clamp_min(1e-9)   # [B] mean bar range
    target = c_anchor + direction * theta * atr             # [B] continuation barrier

    fh, fl = fut[:, _H, :], fut[:, _L, :]                    # [B, k]
    lvl, tgt, dr = level[:, None], target[:, None], direction[:, None]
    # per future bar: did it fall back through the level (fail), or extend to target (hold)?
    hit_fail = torch.where(dr >= 0, fl < lvl, fh > lvl)      # [B, k] bool
    hit_ext = torch.where(dr >= 0, fh >= tgt, fl <= tgt)
    hit_target = hit_ext & ~hit_fail                        # same-bar tie -> fail wins
    idx = torch.arange(k, device=w.device)[None, :]
    BIG = k + 1
    first_fail = torch.where(hit_fail, idx, torch.full_like(idx, BIG)).min(dim=1).values
    first_target = torch.where(hit_target, idx, torch.full_like(idx, BIG)).min(dim=1).values
    hold = is_break & (first_target < first_fail) & (first_target < BIG)   # extend strictly before invalidation
    return hold.float(), is_break


class _BreakHoldTrainer(BaseTrainer):
    def __init__(self, big, tr, va, *, seq=64, new_channels=3, hold_k=12, break_lookback=20,
                 hold_theta=1.0, hold_weight=5.0, recon_weight=1.0,
                 model_id='paris-noah/Mantis-8M', backbone_ckpt=None, compile_model=False, **base):
        super().__init__(big, tr, va, **base)
        self.seq, self.new_channels = seq, new_channels
        self.hold_k, self.lookback, self.theta = int(hold_k), int(break_lookback), float(hold_theta)
        self.hold_weight, self.recon_weight = float(hold_weight), float(recon_weight)
        self.model_id, self.backbone_ckpt, self.compile_model = model_id, backbone_ckpt, compile_model
        self.C = int(self.big_t.shape[1])
        self._last = {'hold_bal_acc': float('nan'), 'hold_recall': float('nan'),
                      'fail_recall': float('nan'), 'break_rate': float('nan'),
                      'hold_rate': float('nan'), 'enc_recon': float('nan')}

    def build_net(self):
        net = BreakHoldNetwork(C=self.C, new_channels=self.new_channels, seq=self.seq,
                               model_id=self.model_id).to(self.dev)
        if self.backbone_ckpt:                             # warm = the promoted base (lineage kept)
            net.encoder.load_state_dict(torch.load(self.backbone_ckpt, map_location='cpu'))
        if self.compile_model and hasattr(torch, 'compile'):
            net = torch.compile(net)
        self.net = net

    def make_batch(self, starts):
        b_idx = torch.randint(0, len(starts), (self.batch,), device=self.dev, generator=self.gen)
        full = _gather_batch(self.big_t, starts, b_idx, self.seq + self.hold_k)   # [B,C,seq+k] RAW
        w = full[:, :, :self.seq]                          # encoder window (RAW; anchor = last bar)
        fut = full[:, :, self.seq:]                        # k future bars (RAW) — LABEL ONLY, never encoded
        # labels from REAL bars (controls corrupt only the encoder INPUT -> real must beat shuffle/random)
        self._label, self._isbreak = break_hold_labels(w, fut, self.lookback, self.theta)
        return _standardize(_apply_control(w, self.control))    # [B, C, seq] standardized encoder input

    def compute_loss(self, z):
        net = self.net if not hasattr(self.net, '_orig_mod') else self.net._orig_mod
        logit, enc_rec = net.heads(z)                      # [B], [B,C,seq]
        recon = F.mse_loss(enc_rec, z)                     # encoder anchor: rebuild the clean window
        m = self._isbreak
        if bool(m.any()):
            lab, lg = self._label[m], logit[m]
            n_hold = lab.sum()
            n_break = m.sum().float()
            # pos_weight balances hold/fail so the BCE can't be gamed by predicting the majority class
            pw = ((n_break - n_hold) / n_hold.clamp_min(1.0)).clamp(0.1, 10.0)
            hold_bce = F.binary_cross_entropy_with_logits(lg, lab, pos_weight=pw)
            with torch.no_grad():
                pred = lg > 0
                hpos, hneg = lab > 0.5, lab < 0.5
                hold_rec = float(pred[hpos].float().mean()) if bool(hpos.any()) else 0.0
                fail_rec = float((~pred[hneg]).float().mean()) if bool(hneg.any()) else 0.0
                self._last = {'hold_bal_acc': 0.5 * (hold_rec + fail_rec),
                              'hold_recall': hold_rec, 'fail_recall': fail_rec,
                              'break_rate': float(m.float().mean()),
                              'hold_rate': float(lab.mean()), 'enc_recon': float(recon)}
        else:
            hold_bce = torch.zeros((), device=z.device)
            self._last = {**self._last, 'break_rate': 0.0, 'enc_recon': float(recon)}
        return self.recon_weight * recon + self.hold_weight * hold_bce

    def log_line(self, ep, tr_loss, vloss, extra, improved):
        if self.verbose:
            print(f"  ep{ep:>3} train={tr_loss:.4f} val={vloss:.4f} "
                  f"bal_acc={extra.get('hold_bal_acc', float('nan')):.3f} "
                  f"(hold={extra.get('hold_recall', float('nan')):.3f}/"
                  f"fail={extra.get('fail_recall', float('nan')):.3f}) "
                  f"break_rate={extra.get('break_rate', float('nan')):.3f} "
                  f"hold_rate={extra.get('hold_rate', float('nan')):.3f} "
                  f"enc_recon={extra.get('enc_recon', float('nan')):.4f} "
                  f"emb_std={extra.get('std', 0.0):.4f}{'  *' if improved else ''}", flush=True)

    @torch.no_grad()
    def val_eval(self):
        self.net.eval()
        tot = 0.0
        agg = {'hold_bal_acc': 0.0, 'hold_recall': 0.0, 'fail_recall': 0.0, 'break_rate': 0.0,
               'hold_rate': 0.0, 'enc_recon': 0.0}
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


def train_ssl_electra(big, train_starts, val_starts, *, seq=64, new_channels=3, hold_k=12,
                      break_lookback=20, hold_theta=1.0, hold_weight=5.0, recon_weight=1.0,
                      epochs=60, steps_per_epoch=200, batch=512, lr=1e-4, weight_decay=0.05,
                      patience=8, device=None, model_id='paris-noah/Mantis-8M', backbone_ckpt=None,
                      compile_model=False, control='real', seed=0, amp_dtype='fp16', verbose=True,
                      ckpt_path=None, resume=False, freeze_encoder_layers=0, **_ignore):
    """BREAK-HOLD discriminative refine (the rewritten electra slot). Returns (best_encoder_state,
    history) with 'val_loss' (recon_weight*enc_recon + hold_weight*BCE), 'hold_bal_acc'/'hold_recall'/
    'fail_recall' (balanced-acc learning diagnostics over break windows), 'break_rate'/'hold_rate',
    'enc_recon' (the encoder anchor — should DROP), + 'std'. hold_weight=0 = denoising-AE only (does
    break-hold add anything over the anchor?); recon_weight=0 = pure discrimination (drift risk)."""
    return _BreakHoldTrainer(big, train_starts, val_starts, seq=seq, new_channels=new_channels,
                             hold_k=hold_k, break_lookback=break_lookback, hold_theta=hold_theta,
                             hold_weight=hold_weight, recon_weight=recon_weight, model_id=model_id,
                             backbone_ckpt=backbone_ckpt, compile_model=compile_model, epochs=epochs,
                             steps_per_epoch=steps_per_epoch, batch=batch, lr=lr,
                             weight_decay=weight_decay, patience=patience, device=device, seed=seed,
                             grad_clip=None, amp_dtype=amp_dtype, verbose=verbose, control=control,
                             ckpt_path=ckpt_path, resume=resume,
                             freeze_encoder_layers=freeze_encoder_layers).fit()
