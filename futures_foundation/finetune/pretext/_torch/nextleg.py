"""Stage-2.6 trainer: NEXT-LEG forecasting — candle-seq2seq ANCHOR + a leg head, windows anchored
at CONFIRMED fractal pivots. Targets in BARS (instrument/TF-agnostic — the fractal unit; NO ATR,
no derived indicators anywhere): t1 = bars until the newborn leg's end extreme, t2 = bars the
counter-leg (the retest) lasts. Pure SSL: targets computed by the deterministic PURE fractal
detector (k-bar candle comparisons) on future price. Direction is not predicted (legs alternate).

Anti-drift: the candle seq2seq loss stays as the ANCHOR (mse_weight, banked load-bearing);
warm-start + freeze_encoder_layers via the shared BaseTrainer; std_guard active.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import _apply_control, _gather_batch
from .forecast import MultiHorizonForecastNet, _ForecastTrainer


class NextLegNet(MultiHorizonForecastNet):
    """Forecast net + a LEG head (2 outputs: log1p bars of newborn leg / counter-leg) off the
    same embedding — the leg gradient shapes the ENCODER; the head is discarded after training."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        emb = self.decoder[0].in_features
        self.leg_head = nn.Sequential(nn.Linear(emb, emb // 4), nn.GELU(), nn.Linear(emb // 4, 2))

    def forward_all(self, ctx):                            # -> (candles [B,C,nH], legs [B,2])
        e = self.embed(ctx)
        return self.decoder(e).view(-1, self.C, self.nH), self.leg_head(e)


def _alternating_fractals(h, l, k):
    """PURE fractal pivots (raw candle comparisons, no ATR), keep-FIRST alternation ->
    list of (extreme_idx, confirm_idx, direction). Torch-free numpy."""
    from ....primitives.detection import detect_fractal_pivots
    piv = detect_fractal_pivots(np.asarray(h, float), np.asarray(l, float), k=int(k))
    out, last_d = [], 0
    for p in piv:                                          # keep-first: skip same-type repeats
        d = int(p['direction'])
        if d == last_d:
            continue
        out.append((int(p['origin']), int(p['confirm']), d))
        last_d = d
    return out


def _leg_targets(big, k, leg_cap, objective_row_bounds=None):
    """Per-confirm leg targets computed independently inside stream/contract segments:
    at confirm c_i -> t1 = extreme_{i+1} - c_i (bars until the newborn leg ends),
                      t2 = extreme_{i+2} - extreme_{i+1} (the counter-leg's bars).
    Returns (confirms [N], targets [N,2] log1p, ok [N] resolved mask). Stream-concat boundary
    pivots are a negligible fraction (~36 of ~2M) and mostly excluded by the resolve cap."""
    big = np.asarray(big)
    if objective_row_bounds is None:
        objective_row_bounds = np.asarray([[0, len(big)]], np.int64)
    bounds = np.asarray(objective_row_bounds, np.int64).reshape(-1, 2)
    confirms, tgts, oks = [], [], []
    for lo, hi in bounds:
        lo, hi = int(lo), int(hi)
        if lo < 0 or hi > len(big) or hi <= lo:
            raise ValueError('invalid next-leg objective segment bounds')
        seq = _alternating_fractals(big[lo:hi, 1], big[lo:hi, 2], k)
        for i in range(len(seq) - 2):
            _o_i, c_i, _ = seq[i]
            o_n, _, _ = seq[i + 1]
            o_nn, _, _ = seq[i + 2]
            t1, t2 = o_n - c_i, o_nn - o_n
            ok = (t1 > 0) and (t2 > 0) and (t1 <= leg_cap) and (t2 <= leg_cap)
            confirms.append(lo + c_i)
            tgts.append((np.log1p(max(t1, 0)), np.log1p(max(t2, 0))))
            oks.append(ok)
    return (np.asarray(confirms, np.int64), np.asarray(tgts, np.float32).reshape(-1, 2),
            np.asarray(oks, bool))


class _NextLegTrainer(_ForecastTrainer):
    def __init__(self, big, tr, va, *, leg_cap=256, leg_w=1.0, leg_k=2, mse_weight=1.0,
                 objective_row_bounds=None, **fw):
        super().__init__(big, tr, va, **fw)
        self.leg_w, self.mse_weight = float(leg_w), float(mse_weight)
        confirms, tgts, ok = _leg_targets(
            np.asarray(big, np.float32), int(leg_k), int(leg_cap), objective_row_bounds,
        )
        confirms, tgts = confirms[ok], tgts[ok]            # resolved-only anchors
        starts = confirms - self.max_ctx + 1               # window start s.t. ctx ENDS at confirm
        # anchor sets = pivot-anchored starts that are LEGAL train/val window starts (leak-safe
        # split + in-stream reserve both inherited from the orchestrator's start sets)
        tr_np, va_np = tr if isinstance(tr, np.ndarray) else np.asarray(tr), \
                       va if isinstance(va, np.ndarray) else np.asarray(va)
        m_tr, m_va = np.isin(starts, tr_np), np.isin(starts, va_np)
        if m_tr.sum() < 1000 or m_va.sum() < 200:
            raise ValueError(f'nextleg: too few resolved pivot anchors '
                             f'(train={int(m_tr.sum())}, val={int(m_va.sum())})')
        self.tr = torch.as_tensor(starts[m_tr], device=self.dev)          # REPLACE start pools
        self.va = torch.as_tensor(starts[m_va], device=self.dev)          # with pivot anchors
        self._tgt_tr = torch.as_tensor(tgts[m_tr], device=self.dev)
        self._tgt_va = torch.as_tensor(tgts[m_va], device=self.dev)
        if self.verbose:
            print(f"  [nextleg] anchors train={len(self.tr):,} val={len(self.va):,} "
                  f"(k={leg_k}, cap={leg_cap} bars, resolved-only) | "
                  f"t1 med={float(np.expm1(np.median(tgts[m_tr][:, 0]))):.0f} bars "
                  f"t2 med={float(np.expm1(np.median(tgts[m_tr][:, 1]))):.0f} bars", flush=True)

    def build_net(self):
        net = NextLegNet(C=self.C, new_channels=self.new_channels, horizons=self.hlist,
                         model_id=self.model_id, aux_dim=0).to(self.dev)
        if self.backbone_ckpt:
            net.encoder.load_state_dict(torch.load(self.backbone_ckpt, map_location='cpu'))
        self.net = net

    def make_batch(self, starts):
        tgt_all = self._tgt_tr if starts is self.tr else self._tgt_va
        b_idx = torch.randint(0, len(starts), (self.batch,), device=self.dev, generator=self.gen)
        w = _gather_batch(self.big_t, starts, b_idx, self.parent)
        L = int(self.clens_t[torch.randint(0, len(self.clens_t), (1,), device=self.dev,
                                           generator=self.gen)].item())
        ctx_raw = w[:, :, self.max_ctx - L:self.max_ctx]   # ends at the CONFIRM bar
        fut_raw = w[:, :, self.max_ctx:]
        m = ctx_raw.mean(2, keepdim=True); s = ctx_raw.std(2, keepdim=True) + 1e-6
        cs = ((ctx_raw - m) / s).clamp(-self.clamp, self.clamp)
        fs = ((fut_raw - m) / s).clamp(-self.clamp, self.clamp)
        cand_t = fs[:, :, self.h_off] - cs[:, :, -1:]      # candle ANCHOR target (move from now)
        return _apply_control(cs, self.control), cand_t, tgt_all[b_idx]

    def compute_loss(self, batch):
        ctx, cand_t, leg_t = batch
        candles, legs = self.net.forward_all(ctx)
        cand_loss = F.mse_loss(candles.float(), cand_t)
        leg_loss = F.smooth_l1_loss(legs.float(), leg_t)
        return self.mse_weight * cand_loss + self.leg_w * leg_loss

    @torch.no_grad()
    def val_eval(self):
        self.net.eval()
        tot_c = tot_p = tot_l = 0.0
        preds, tgts = [], []
        nb = min(20, max(1, len(self.va) // self.batch))
        for _ in range(nb):
            ctx, cand_t, leg_t = self.make_batch(self.va)
            candles, legs = self.net.forward_all(ctx)
            tot_c += float(F.mse_loss(candles.float(), cand_t))
            tot_p += float((cand_t ** 2).mean())
            tot_l += float(F.smooth_l1_loss(legs.float(), leg_t))
            preds.append(legs.float().cpu()); tgts.append(leg_t.cpu())
        P, T = torch.cat(preds), torch.cat(tgts)
        corr = [float(np.corrcoef(P[:, j].numpy(), T[:, j].numpy())[0, 1]) for j in (0, 1)]
        mae = [float(np.expm1(P[:, j].numpy()).mean() - np.expm1(T[:, j].numpy()).mean())
               for j in (0, 1)]                                     # bars, bias diagnostic
        estd = float(self.net.embed(self.make_batch(self.va)[0]).std(0).mean())
        skill = 1.0 - (tot_c / nb) / max(tot_p / nb, 1e-12)
        vloss = self.mse_weight * (tot_c / nb) + self.leg_w * (tot_l / nb)
        return vloss, {'skill': skill, 'leg_corr1': corr[0], 'leg_corr2': corr[1],
                       'leg_bias1': mae[0], 'leg_bias2': mae[1], 'std': estd}

    def log_line(self, ep, tr_loss, vloss, extra, improved):
        if self.verbose:
            print(f"  ep{ep:>3} train={tr_loss:.4f} val={vloss:.4f} skill={extra['skill']:+.3f} "
                  f"legR={extra['leg_corr1']:+.3f}/{extra['leg_corr2']:+.3f} "
                  f"emb_std={extra['std']:.4f}{'  *' if improved else ''}", flush=True)


def train_ssl_nextleg(big, train_starts, val_starts, *, horizons=(5, 10, 20, 25),
                      context_lengths=(64, 100, 150, 200), new_channels=8, epochs=60,
                      steps_per_epoch=200, batch=512, lr=1e-4, weight_decay=0.05, patience=8,
                      device=None, model_id='paris-noah/Mantis-8M', backbone_ckpt=None,
                      control='real', seed=0, clamp=10.0, grad_clip=1.0, verbose=True,
                      ckpt_path=None, resume=False, freeze_encoder_layers=0, std_guard=1.6,
                      leg_cap=256, leg_w=1.0, leg_k=2, mse_weight=1.0,
                      objective_row_bounds=None, **_ignore):
    """NEXT-LEG SSL -> (best_encoder_state, history) with 'val_loss', 'skill' (candle anchor),
    'leg_corr1/2' (bars-to-leg-end / counter-leg correlation — the learning diagnostics), 'std'."""
    t = _NextLegTrainer(big, train_starts, val_starts,
                        horizons=horizons, context_lengths=context_lengths,
                        new_channels=new_channels, model_id=model_id, backbone_ckpt=backbone_ckpt,
                        clamp=clamp, leg_cap=leg_cap, leg_w=leg_w, leg_k=leg_k,
                        objective_row_bounds=objective_row_bounds,
                        mse_weight=mse_weight, epochs=epochs, steps_per_epoch=steps_per_epoch,
                        batch=batch, lr=lr, weight_decay=weight_decay, patience=patience,
                        device=device, seed=seed, grad_clip=grad_clip, verbose=verbose,
                        control=control, ckpt_path=ckpt_path, resume=resume,
                        freeze_encoder_layers=freeze_encoder_layers, std_guard=std_guard)
    return t.fit()
