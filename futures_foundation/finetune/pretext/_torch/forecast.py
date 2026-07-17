"""Stage-3 trainer: MULTI-HORIZON, VARIABLE-CONTEXT candle seq2seq (ANTI-SHORTCUT). Predict the
future CANDLE (OHLCV) at each horizon as a move FROM 'now' (context-standardized), so 'copy now'
== predict-zero (punished). Reports per-horizon skill so we can see whether the far horizons learn."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import (_apply_control, _gather_batch, BaseTrainer, encode_independent,
                     load_encoder_checkpoint, preprocess_context_and_future,
                     resolve_preprocessing)
from .backbone import load_mantis


class MultiHorizonForecastNet(nn.Module):
    """Channel-independent Mantis plus a cross-channel multi-horizon candle decoder."""

    def __init__(self, C=5, new_channels=8, horizons=(5, 10, 20, 25),
                 model_id='paris-noah/Mantis-8M', model_version=None, aux_dim=0):
        super().__init__()
        self.encoder = load_mantis(model_id, model_version=model_version, device='cpu')
        hidden = getattr(self.encoder, 'hidden_dim', 256)
        self.new_c = C                 # new_channels retained only for legacy config compatibility
        self.C, self.horizons = C, tuple(int(h) for h in horizons)
        self.nH = len(self.horizons)
        emb = hidden * self.new_c
        self.decoder = nn.Sequential(nn.Linear(emb, emb), nn.GELU(), nn.Linear(emb, C * self.nH))
        # OPTIONAL aux head sized by the forecast OBJECTIVE (e.g. direction logits) — a LINEAR readout
        # off the same embedding, so the objective's gradient shapes the ENCODER. Discarded after
        # training. aux_dim=0 = candle-only.
        self.aux_head = nn.Linear(emb, aux_dim) if aux_dim > 0 else None

    def embed(self, x):                                   # [B,C,L] -> [B, new_c*hidden]
        return encode_independent(self.encoder, x)

    def forward(self, ctx):                               # -> (candles [B,C,nH], aux [B,aux_dim] or None)
        e = self.embed(ctx)
        candles = self.decoder(e).view(-1, self.C, self.nH)
        return candles, (self.aux_head(e) if self.aux_head is not None else None)


class _ForecastTrainer(BaseTrainer):
    def __init__(self, big, tr, va, *, horizons=(5, 10, 20, 25), context_lengths=(64, 100, 150, 200),
                 new_channels=8, model_id='paris-noah/Mantis-8M', backbone_ckpt=None,
                 model_version=None, compile_model=False, clamp=10.0, objective='candle_mse', dir_weight=0.0,
                 dir_close_ch=3, preprocessing=None, **base):
        super().__init__(big, tr, va, **base)
        self.hlist = [int(h) for h in horizons]
        self.clens = [int(x) for x in context_lengths]
        self.max_ctx, self.h_max = max(self.clens), max(self.hlist)
        self.parent = self.max_ctx + self.h_max
        self.h_off = torch.as_tensor([h - 1 for h in self.hlist], dtype=torch.long, device=self.dev)
        self.clens_t = torch.as_tensor(self.clens, dtype=torch.long, device=self.dev)
        self.new_channels, self.model_id, self.model_version = new_channels, model_id, model_version
        self.backbone_ckpt, self.compile_model, self.clamp = backbone_ckpt, compile_model, clamp
        self.preprocessing = resolve_preprocessing(preprocessing, backbone_ckpt)
        self.C = int(self.big_t.shape[1])
        from .forecast_objectives import get_forecast_objective
        self.obj = get_forecast_objective(objective)                # pluggable forecast supervision
        self.dir_weight = float(dir_weight)                         # aux-head loss weight (0 = candle-only)
        self.close_ch = min(int(dir_close_ch), self.C - 1)          # OHLCV close = index 3

    def build_net(self):
        net = MultiHorizonForecastNet(C=self.C, new_channels=self.new_channels, horizons=self.hlist,
                                      model_id=self.model_id, model_version=self.model_version,
                                      aux_dim=self.obj.aux_dim(len(self.hlist))).to(self.dev)
        if self.backbone_ckpt:
            load_encoder_checkpoint(net.encoder, self.backbone_ckpt, model_id=self.model_id,
                                    model_version=self.model_version, expected_channels=self.C)
        if self.compile_model and hasattr(torch, 'compile'):
            net = torch.compile(net)
        self.net = net

    def make_batch(self, starts):
        b_idx = self.sample_start_indices(starts)
        w = _gather_batch(self.big_t, starts, b_idx, self.parent)     # [B,C,max_ctx+h_max] real
        L = int(self.clens_t[torch.randint(0, len(self.clens_t), (1,), device=self.dev,
                                           generator=self.gen)].item())
        ctx_raw = w[:, :, self.max_ctx - L:self.max_ctx]             # [B,C,L] context ending at 'now'
        fut_raw = w[:, :, self.max_ctx:]                            # [B,C,h_max] future candles
        cs, fs = preprocess_context_and_future(
            ctx_raw, fut_raw, self.preprocessing, clamp=self.clamp)
        target = fs[:, :, self.h_off] - cs[:, :, -1:]               # [B,C,nH] move FROM now (anti-shortcut)
        return _apply_control(cs, self.control), target             # corrupt ONLY the input

    def compute_loss(self, batch):
        model_ctx, target = batch
        candles, aux = self.net(model_ctx)
        return self.obj.loss(candles, aux, target, self.close_ch, self.dir_weight)

    @torch.no_grad()
    def val_eval(self):
        from .forecast_objectives import dir_acc as _dir_acc
        self.net.eval(); tot = 0.0; ptot = 0.0
        toth = torch.zeros(len(self.hlist), device=self.dev)        # per-horizon: is 20/25 learning?
        ptoth = torch.zeros(len(self.hlist), device=self.dev)
        dacc = torch.zeros(len(self.hlist), device=self.dev)        # per-horizon directional accuracy
        nb = min(self.val_batches or 20, max(1, len(self.va) // self.batch))
        with self.fixed_validation_rng():
            for _ in range(nb):
                mc, tg = self.make_batch(self.va)
                with self.amp_ctx():
                    candles, _aux = self.net(mc)                    # net ALWAYS returns (candles, aux)
                se = (candles.float() - tg) ** 2
                tot += float(se.mean()); ptot += float((tg ** 2).mean())
                toth += se.mean(dim=(0, 1)); ptoth += (tg ** 2).mean(dim=(0, 1))
                dacc += _dir_acc(candles, tg, self.close_ch)        # universal across objectives
            estd = float(self.net.embed(self.make_batch(self.va)[0]).std(0).mean())
        self.net.train()
        vloss, ploss = tot / nb, ptot / nb
        skill = float(1.0 - vloss / ploss) if ploss > 1e-12 else 0.0
        skill_h = (1.0 - toth / ptoth.clamp_min(1e-12)).cpu().tolist()
        dir_h = (dacc / nb).cpu().tolist()                          # dir_acc>0.5 = learning direction
        return vloss, {'persist_loss': ploss, 'skill': skill,
                       'skill_per_h': dict(zip(self.hlist, skill_h)),
                       'dir_acc': float(sum(dir_h) / len(dir_h)),
                       'dir_acc_per_h': dict(zip(self.hlist, dir_h)), 'std': estd}

    def log_line(self, ep, tr_loss, vloss, extra, improved):
        if self.verbose:
            ph = ' '.join(f"h{h}={s:+.2f}" for h, s in extra['skill_per_h'].items())
            print(f"  ep{ep:>3} train={tr_loss:.4f} val={vloss:.4f} "
                  f"persist={extra['persist_loss']:.4f} skill={extra['skill']:+.3f} [{ph}] "
                  f"dir={extra['dir_acc']:.3f} emb_std={extra['std']:.4f}{'  *' if improved else ''}",
                  flush=True)


def train_ssl_forecast(big, train_starts, val_starts, *, horizons=(5, 10, 20, 25),
                       context_lengths=(64, 100, 150, 200), new_channels=8, epochs=60,
                       steps_per_epoch=200, batch=512, lr=1e-4, weight_decay=0.05, patience=8,
                       device=None, model_id='paris-noah/Mantis-8M', backbone_ckpt=None,
                       model_version=None, compile_model=False, control='real', seed=0, amp_dtype='fp16',
                       grad_clip=1.0, clamp=10.0, verbose=True,
                       ckpt_path=None, resume=False, freeze_encoder_layers=0,
                       objective='candle_mse', dir_weight=0.0, dir_close_ch=3, **_ignore):
    """Multi-horizon / variable-context candle seq2seq. Returns (best_encoder_state, history) with
    'val_loss', 'persist_loss', 'skill', 'skill_per_h', 'std' (+ 'dir_acc' if dir_weight>0). Warm-start
    from stage-1. OPTIONAL: dir_weight>0 adds a direction-head BCE term (sign of the fwd close move) to
    the candle MSE — trains the encoder to be direction-aware (WR-relevant); 0 = original behavior."""
    return _ForecastTrainer(big, train_starts, val_starts, horizons=horizons,
                            context_lengths=context_lengths, new_channels=new_channels,
                            model_id=model_id, model_version=model_version,
                            backbone_ckpt=backbone_ckpt, compile_model=compile_model,
                            clamp=clamp, epochs=epochs, steps_per_epoch=steps_per_epoch, batch=batch,
                            lr=lr, weight_decay=weight_decay, patience=patience, device=device,
                            seed=seed, grad_clip=grad_clip, amp_dtype=amp_dtype, verbose=verbose,
                            control=control, ckpt_path=ckpt_path, resume=resume,
                            freeze_encoder_layers=freeze_encoder_layers, objective=objective,
                            dir_weight=dir_weight, dir_close_ch=dir_close_ch,
                            preprocessing=_ignore.get('preprocessing'),
                            train_group_bounds=_ignore.get('train_group_bounds'),
                            val_group_bounds=_ignore.get('val_group_bounds'),
                            val_batches=_ignore.get('val_batches')).fit()
