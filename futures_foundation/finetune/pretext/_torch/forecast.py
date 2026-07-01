"""Stage-2 trainer: MULTI-HORIZON, VARIABLE-CONTEXT candle seq2seq (ANTI-SHORTCUT). Predict the
future CANDLE (OHLCV) at each horizon as a move FROM 'now' (context-standardized), so 'copy now'
== predict-zero (punished). Reports per-horizon skill so we can see whether the far horizons learn."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import _enc, _apply_control, _gather_batch, BaseTrainer


class MultiHorizonForecastNet(nn.Module):
    """Mantis encoder + channel adapter + a MULTI-HORIZON candle decoder. A variable-length context
    is encoded; the decoder predicts the future CANDLE (OHLCV) at EACH horizon -> [B, C, n_horizons]."""

    def __init__(self, C=5, new_channels=8, horizons=(5, 10, 20, 25),
                 model_id='paris-noah/Mantis-8M', use_direction=False):
        super().__init__()
        from mantis.architecture import Mantis8M
        from mantis.adapters import LinearChannelCombiner
        self.encoder = Mantis8M.from_pretrained(model_id)
        hidden = getattr(self.encoder, 'hidden_dim', 256)
        self.new_c = min(new_channels, C)
        self.adapter = LinearChannelCombiner(num_channels=C, new_num_channels=self.new_c)
        self.C, self.horizons = C, tuple(int(h) for h in horizons)
        self.nH = len(self.horizons)
        emb = hidden * self.new_c
        self.decoder = nn.Sequential(nn.Linear(emb, emb), nn.GELU(), nn.Linear(emb, C * self.nH))
        # OPTIONAL aux direction head (LINEAR on purpose) — predicts sign(forward close move) per
        # horizon. Forces the ENCODER to encode direction linearly (BCE-on-sign has no mean-regression
        # to wash out the sign, unlike the candle MSE). Discarded after training. None (default) =
        # backward-compatible pure-candle-MSE forecast.
        self.dir_head = nn.Linear(emb, self.nH) if use_direction else None

    def embed(self, x):                                   # [B,C,L] -> [B, new_c*hidden]
        a = self.adapter(x)
        return torch.cat([_enc(self.encoder, a[:, [i], :]) for i in range(a.shape[1])], dim=-1)

    def forward(self, ctx):                               # -> candles [B,C,nH]  (+ dir logits [B,nH] if on)
        e = self.embed(ctx)
        candles = self.decoder(e).view(-1, self.C, self.nH)
        if self.dir_head is None:
            return candles                                # backward-compat: candles only
        return candles, self.dir_head(e)                  # direction enabled: (candles, dir_logits)


class _ForecastTrainer(BaseTrainer):
    def __init__(self, big, tr, va, *, horizons=(5, 10, 20, 25), context_lengths=(64, 100, 150, 200),
                 new_channels=8, model_id='paris-noah/Mantis-8M', backbone_ckpt=None,
                 compile_model=False, clamp=10.0, dir_weight=0.0, dir_close_ch=3, **base):
        super().__init__(big, tr, va, **base)
        self.hlist = [int(h) for h in horizons]
        self.clens = [int(x) for x in context_lengths]
        self.max_ctx, self.h_max = max(self.clens), max(self.hlist)
        self.parent = self.max_ctx + self.h_max
        self.h_off = torch.as_tensor([h - 1 for h in self.hlist], dtype=torch.long, device=self.dev)
        self.clens_t = torch.as_tensor(self.clens, dtype=torch.long, device=self.dev)
        self.new_channels, self.model_id = new_channels, model_id
        self.backbone_ckpt, self.compile_model, self.clamp = backbone_ckpt, compile_model, clamp
        self.C = int(self.big_t.shape[1])
        # optional direction-head squeeze (backward-compat: dir_weight=0 -> off, pure candle MSE)
        self.dir_weight = float(dir_weight); self.use_direction = self.dir_weight > 0
        self.close_ch = min(int(dir_close_ch), self.C - 1)           # OHLCV close = index 3

    def build_net(self):
        net = MultiHorizonForecastNet(C=self.C, new_channels=self.new_channels, horizons=self.hlist,
                                      model_id=self.model_id, use_direction=self.use_direction).to(self.dev)
        if self.backbone_ckpt:
            net.encoder.load_state_dict(torch.load(self.backbone_ckpt, map_location='cpu'))
        if self.compile_model and hasattr(torch, 'compile'):
            net = torch.compile(net)
        self.net = net

    def make_batch(self, starts):
        b_idx = torch.randint(0, len(starts), (self.batch,), device=self.dev, generator=self.gen)
        w = _gather_batch(self.big_t, starts, b_idx, self.parent)     # [B,C,max_ctx+h_max] real
        L = int(self.clens_t[torch.randint(0, len(self.clens_t), (1,), device=self.dev,
                                           generator=self.gen)].item())
        ctx_raw = w[:, :, self.max_ctx - L:self.max_ctx]             # [B,C,L] context ending at 'now'
        fut_raw = w[:, :, self.max_ctx:]                            # [B,C,h_max] future candles
        m = ctx_raw.mean(2, keepdim=True); s = ctx_raw.std(2, keepdim=True) + 1e-6
        cs = ((ctx_raw - m) / s).clamp(-self.clamp, self.clamp)     # standardized context (input)
        fs = ((fut_raw - m) / s).clamp(-self.clamp, self.clamp)     # standardized future candles
        target = fs[:, :, self.h_off] - cs[:, :, -1:]               # [B,C,nH] move FROM now (anti-shortcut)
        return _apply_control(cs, self.control), target             # corrupt ONLY the input

    def compute_loss(self, batch):
        model_ctx, target = batch
        out = self.net(model_ctx)
        if not self.use_direction:
            return ((out - target) ** 2).mean()                     # backward-compat: pure candle MSE
        candles, dir_logits = out
        mse = ((candles - target) ** 2).mean()                      # "how far" (magnitude)
        dir_tgt = (target[:, self.close_ch, :] > 0).float()         # "which way": sign of fwd close move
        dir_loss = F.binary_cross_entropy_with_logits(dir_logits, dir_tgt)
        return mse + self.dir_weight * dir_loss

    @torch.no_grad()
    def val_eval(self):
        self.net.eval(); tot = 0.0; ptot = 0.0
        toth = torch.zeros(len(self.hlist), device=self.dev)        # per-horizon: is 20/25 learning?
        ptoth = torch.zeros(len(self.hlist), device=self.dev)
        dhits = torch.zeros(len(self.hlist), device=self.dev); dn = 0   # per-horizon direction hits
        nb = min(20, max(1, len(self.va) // self.batch))
        for _ in range(nb):
            mc, tg = self.make_batch(self.va)
            with self.amp_ctx():
                out = self.net(mc)                                  # candles [B,C,nH] (+ dir if enabled)
            candles = out[0] if self.use_direction else out
            se = (candles.float() - tg) ** 2
            tot += float(se.mean()); ptot += float((tg ** 2).mean())
            toth += se.mean(dim=(0, 1)); ptoth += (tg ** 2).mean(dim=(0, 1))
            if self.use_direction:                                  # per-horizon directional ACCURACY
                dtgt = tg[:, self.close_ch, :] > 0
                dhits += ((out[1] > 0) == dtgt).float().sum(0); dn += dtgt.shape[0]
        estd = float(self.net.embed(self.make_batch(self.va)[0]).std(0).mean())
        self.net.train()
        vloss, ploss = tot / nb, ptot / nb
        skill = float(1.0 - vloss / ploss) if ploss > 1e-12 else 0.0
        skill_h = (1.0 - toth / ptoth.clamp_min(1e-12)).cpu().tolist()
        extra = {'persist_loss': ploss, 'skill': skill,
                 'skill_per_h': dict(zip(self.hlist, skill_h)), 'std': estd}
        if self.use_direction:                                     # dir_acc>0.5 = learning direction
            dir_h = (dhits / max(dn, 1)).cpu().tolist()
            extra['dir_acc'] = float(sum(dir_h) / len(dir_h))
            extra['dir_acc_per_h'] = dict(zip(self.hlist, dir_h))
        return vloss, extra

    def log_line(self, ep, tr_loss, vloss, extra, improved):
        if self.verbose:
            ph = ' '.join(f"h{h}={s:+.2f}" for h, s in extra['skill_per_h'].items())
            da = f" dir={extra['dir_acc']:.3f}" if 'dir_acc' in extra else ''
            print(f"  ep{ep:>3} train={tr_loss:.4f} val={vloss:.4f} "
                  f"persist={extra['persist_loss']:.4f} skill={extra['skill']:+.3f} [{ph}]{da} "
                  f"emb_std={extra['std']:.4f}{'  *' if improved else ''}", flush=True)


def train_ssl_forecast(big, train_starts, val_starts, *, horizons=(5, 10, 20, 25),
                       context_lengths=(64, 100, 150, 200), new_channels=8, epochs=60,
                       steps_per_epoch=200, batch=512, lr=1e-4, weight_decay=0.05, patience=8,
                       device=None, model_id='paris-noah/Mantis-8M', backbone_ckpt=None,
                       compile_model=False, control='real', seed=0, amp_dtype='fp16',
                       grad_clip=1.0, clamp=10.0, verbose=True,
                       ckpt_path=None, resume=False, freeze_encoder_layers=0,
                       dir_weight=0.0, dir_close_ch=3, **_ignore):
    """Multi-horizon / variable-context candle seq2seq. Returns (best_encoder_state, history) with
    'val_loss', 'persist_loss', 'skill', 'skill_per_h', 'std' (+ 'dir_acc' if dir_weight>0). Warm-start
    from stage-1. OPTIONAL: dir_weight>0 adds a direction-head BCE term (sign of the fwd close move) to
    the candle MSE — trains the encoder to be direction-aware (WR-relevant); 0 = original behavior."""
    return _ForecastTrainer(big, train_starts, val_starts, horizons=horizons,
                            context_lengths=context_lengths, new_channels=new_channels,
                            model_id=model_id, backbone_ckpt=backbone_ckpt, compile_model=compile_model,
                            clamp=clamp, epochs=epochs, steps_per_epoch=steps_per_epoch, batch=batch,
                            lr=lr, weight_decay=weight_decay, patience=patience, device=device,
                            seed=seed, grad_clip=grad_clip, amp_dtype=amp_dtype, verbose=verbose,
                            control=control, ckpt_path=ckpt_path, resume=resume,
                            freeze_encoder_layers=freeze_encoder_layers,
                            dir_weight=dir_weight, dir_close_ch=dir_close_ch).fit()
