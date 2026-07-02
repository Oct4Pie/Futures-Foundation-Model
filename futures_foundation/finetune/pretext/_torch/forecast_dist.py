"""Stage-2.5 trainer: DISTRIBUTIONAL forecast refine (the Chronos insight) — builds ON stage-2.

Both Chronos variants refuse to learn a conditional MEAN: classic learns the full next-value
DISTRIBUTION by bin classification (uniform quantization -> cross-entropy over the vocab), Bolt by
direct multi-step quantile heads (pinball loss, quantiles 0.1..0.9). Our ship metric WR@3R is a
TAIL question ("which pivots run 3R before -1R") — a mean-regressor (candle_mse) is structurally
blind to exactly that. This pretext warm-starts from the PROMOTED stage-2 seq2seq and refines it
with a distributional objective: same SSL targets (context-standardized future close move — raw
OHLCV, no labels/ATR/leak), same net/trainer machinery (imported, NOT modified), only the LOSS
GEOMETRY changes. Unlike stage-3's contrastive key (redundant with what the forecast had already
learned -> key_gap flat), the distributional term has gradient exactly where MSE provably has
none: the shape of the outcome distribution around the mean.

A SEPARATE pretext ('forecast_dist') so the original stage-2 ('forecast') stays byte-untouched:
its own local objective registry, its own study namespace, forecast.py never edited.

  candle_quantile  Chronos-BOLT:   candle head = median; aux = lo/hi close-move quantiles
                   (t=0.1/0.9) per horizon; loss = candle MSE + Bolt's exact pinball
                   2*|(t-q)*((t<=q)-tau)| over the three quantiles. The learned SPREAD is
                   uncertainty — a wide upper quantile at a pivot ~ "this can run".
  candle_bins      Chronos-CLASSIC: forecasting as CLASSIFICATION. Future close move (context-
                   sigma units, clamped +-clamp by the trainer) -> K uniform bins; aux = K logits
                   per horizon; loss = candle MSE + cross-entropy. Native fit: Mantis is
                   classification-pretrained — this speaks its loss; the logit vector is a learned
                   per-horizon distribution over how far price moves, tails included.

`dir_weight` mixes the distributional term with the candle MSE (the already-plumbed weight knob);
0/unset defaults to 1.0 — these objectives are meaningless without their term, so there is no
silent fall-through to plain MSE.
"""
from .forecast import _ForecastTrainer
from .forecast_objectives import ForecastObjective


class CandleQuantile(ForecastObjective):
    """Chronos-Bolt style direct multi-step QUANTILE supervision (see module docstring)."""
    name = 'candle_quantile'
    TAUS = (0.1, 0.9)                                     # + median from the candle head (t=0.5)

    def aux_dim(self, nH):
        return nH * len(self.TAUS)                        # lo/hi close-move quantile per horizon

    @staticmethod
    def _pinball(t, q, tau):
        return (2.0 * ((t - q) * ((t <= q).float() - tau)).abs()).mean()   # Bolt's exact form

    def loss(self, candles, aux, target, close_ch, weight):
        w = weight if weight and weight > 0 else 1.0
        mse = ((candles - target) ** 2).mean()
        t = target[:, close_ch, :]                                       # [B, nH] close move
        q = aux.view(aux.shape[0], t.shape[1], len(self.TAUS))           # [B, nH, |TAUS|]
        pin = self._pinball(t, candles[:, close_ch, :], 0.5)             # median = candle head
        for k, tau in enumerate(self.TAUS):
            pin = pin + self._pinball(t, q[:, :, k], tau)
        return mse + w * pin


class CandleBins(ForecastObjective):
    """Chronos-classic style bin-CLASSIFICATION supervision (see module docstring)."""
    name = 'candle_bins'
    K, BIN_RANGE = 41, 10.0                               # K uniform bins over [-clamp, clamp]

    def aux_dim(self, nH):
        return nH * self.K

    def loss(self, candles, aux, target, close_ch, weight):
        import torch
        import torch.nn.functional as F
        w = weight if weight and weight > 0 else 1.0
        mse = ((candles - target) ** 2).mean()
        t = target[:, close_ch, :]                                       # [B, nH] close move
        edges = torch.linspace(-self.BIN_RANGE, self.BIN_RANGE, self.K + 1,
                               device=t.device)[1:-1]                    # inner edges -> K bins
        idx = torch.bucketize(t.contiguous(), edges)                     # [B, nH] in [0, K)
        logits = aux.view(aux.shape[0], t.shape[1], self.K)              # [B, nH, K]
        ce = F.cross_entropy(logits.reshape(-1, self.K), idx.reshape(-1))
        return mse + w * ce


DIST_OBJECTIVES = {o.name: o for o in (CandleQuantile(), CandleBins())}


def get_dist_objective(name):
    """Resolve a DISTRIBUTIONAL objective by name (None -> 'candle_quantile'). KeyError = fail fast."""
    return DIST_OBJECTIVES[name or 'candle_quantile']


class _DistForecastTrainer(_ForecastTrainer):
    """The stage-2 trainer with the objective swapped to a distributional one. Pure subclass —
    forecast.py is imported, never modified; net/batching/val (universal dir_acc) all inherited."""

    def __init__(self, big, tr, va, *, objective='candle_quantile', **kw):
        super().__init__(big, tr, va, objective='candle_mse', **kw)      # base init (placeholder obj)
        self.obj = get_dist_objective(objective)                         # swap BEFORE build_net()


def train_ssl_forecast_dist(big, train_starts, val_starts, *, horizons=(5, 10, 20, 25),
                            context_lengths=(64, 100, 150, 200), new_channels=8, epochs=60,
                            steps_per_epoch=200, batch=512, lr=1e-4, weight_decay=0.05, patience=8,
                            device=None, model_id='paris-noah/Mantis-8M', backbone_ckpt=None,
                            compile_model=False, control='real', seed=0, amp_dtype='fp16',
                            grad_clip=1.0, clamp=10.0, verbose=True,
                            ckpt_path=None, resume=False, freeze_encoder_layers=0,
                            objective='candle_quantile', dir_weight=1.0, dir_close_ch=3, **_ignore):
    """Distributional forecast refine (stage-2.5). Warm-start = the PROMOTED stage-2 encoder
    (backbone_ckpt). Returns (best_encoder_state, history) with the same metrics as stage-2
    ('skill', 'skill_per_h', 'dir_acc', 'std') — comparable across objectives (dir_acc is
    universal, read off the candle head)."""
    return _DistForecastTrainer(big, train_starts, val_starts, horizons=horizons,
                                context_lengths=context_lengths, new_channels=new_channels,
                                model_id=model_id, backbone_ckpt=backbone_ckpt,
                                compile_model=compile_model, clamp=clamp, epochs=epochs,
                                steps_per_epoch=steps_per_epoch, batch=batch, lr=lr,
                                weight_decay=weight_decay, patience=patience, device=device,
                                seed=seed, grad_clip=grad_clip, amp_dtype=amp_dtype,
                                verbose=verbose, control=control, ckpt_path=ckpt_path,
                                resume=resume, freeze_encoder_layers=freeze_encoder_layers,
                                objective=objective, dir_weight=dir_weight,
                                dir_close_ch=dir_close_ch).fit()
