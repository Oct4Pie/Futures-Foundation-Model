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
  candle_mixture   MOIRAI (Salesforce uni2ts): MIXTURE-DENSITY head trained by mixture NLL.
                   Student-t (LEARNED fat tails = winR) + Normal + low-variance Normal (chop
                   mode), adapted REAL-VALUED for signed moves (Moirai's neg-binomial/log-normal
                   components are for positive data — copying them would be wrong here). The
                   mixture weights can express the pivot's BIMODAL future (runs vs chops).

`dir_weight` mixes the distributional term with the candle MSE (the already-plumbed weight knob);
0/unset defaults to 1.0 — these objectives are meaningless without their term, so there is no
silent fall-through to plain MSE.
"""
from .forecast import _ForecastTrainer
from .forecast_objectives import ForecastObjective


"""FAITHFULNESS KNOBS (all default to the ORIGINAL refine-study behavior — the running study's
objectives stay byte-identical; the knobs only activate when explicitly configured):
  mse_weight     1.0 = original mixed loss; 0.0 = PURE Chronos (no MSE anchor anywhere — classic
                 is pure CE, Bolt is pure pinball; the mean lives inside the distribution).
  quantile_taus  'lohi' = the original 2-quantile head; 'bolt9' = Bolt's full 9-level weather
                 report (0.1..0.9, median from the candle head) — the spread SHAPE is the signal.
  bins_k         41 = original; raise (129/257) toward Chronos-classic's ~4094-bin resolution so
                 the tails stop being one blurry 'big move' bucket."""

_TAU_PRESETS = {'lohi': (0.1, 0.9),
                'bolt9': (0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9)}   # +median = Bolt's 9 levels


class CandleQuantile(ForecastObjective):
    """Chronos-Bolt style direct multi-step QUANTILE supervision (see module docstring)."""
    name = 'candle_quantile'

    def __init__(self, mse_weight=1.0, quantile_taus='lohi', **_):
        self.mse_w = float(mse_weight)
        self.TAUS = _TAU_PRESETS[quantile_taus or 'lohi']

    def aux_dim(self, nH):
        return nH * len(self.TAUS)                        # close-move quantiles per horizon

    @staticmethod
    def _pinball(t, q, tau):
        return (2.0 * ((t - q) * ((t <= q).float() - tau)).abs()).mean()   # Bolt's exact form

    def loss(self, candles, aux, target, close_ch, weight):
        w = weight if weight and weight > 0 else 1.0
        t = target[:, close_ch, :]                                       # [B, nH] close move
        q = aux.view(aux.shape[0], t.shape[1], len(self.TAUS))           # [B, nH, |TAUS|]
        pin = self._pinball(t, candles[:, close_ch, :], 0.5)             # median = candle head
        for k, tau in enumerate(self.TAUS):
            pin = pin + self._pinball(t, q[:, :, k], tau)
        mse = ((candles - target) ** 2).mean() if self.mse_w > 0 else 0.0
        return self.mse_w * mse + w * pin


class CandleBins(ForecastObjective):
    """Chronos-classic style bin-CLASSIFICATION supervision (see module docstring)."""
    name = 'candle_bins'
    BIN_RANGE = 10.0                                      # uniform bins over [-clamp, clamp]

    def __init__(self, mse_weight=1.0, bins_k=41, **_):
        self.mse_w = float(mse_weight)
        self.K = int(bins_k or 41)

    def aux_dim(self, nH):
        return nH * self.K

    def loss(self, candles, aux, target, close_ch, weight):
        import torch
        import torch.nn.functional as F
        w = weight if weight and weight > 0 else 1.0
        t = target[:, close_ch, :]                                       # [B, nH] close move
        edges = torch.linspace(-self.BIN_RANGE, self.BIN_RANGE, self.K + 1,
                               device=t.device)[1:-1]                    # inner edges -> K bins
        idx = torch.bucketize(t.contiguous(), edges)                     # [B, nH] in [0, K)
        logits = aux.view(aux.shape[0], t.shape[1], self.K)              # [B, nH, K]
        ce = F.cross_entropy(logits.reshape(-1, self.K), idx.reshape(-1))
        mse = ((candles - target) ** 2).mean() if self.mse_w > 0 else 0.0
        return self.mse_w * mse + w * ce


class CandleMixture(ForecastObjective):
    """MOIRAI-style MIXTURE-DENSITY supervision: per horizon, predict a full parametric
    DISTRIBUTION of the future close move as a K=3 mixture — Student-t (LEARNED fat tails =
    'how far can this run', the winR signal), Normal (the body), and a low-variance Normal
    (the confident/chop mode) — trained by mixture NEGATIVE LOG-LIKELIHOOD (Moirai's exact
    loss). ADAPTED, not copied: Moirai's neg-binomial/log-normal components are for POSITIVE
    data; our target is the SIGNED context-standardized close move, so all components are
    real-valued. The mixture WEIGHTS can express bimodality (breaks-out vs chops) that a
    single-mode head blurs — the weight on the running mode ~ P(reach target).

    aux per horizon (P=9): 3 mixture logits + Student-t(mu, scale, df) + Normal(mu, scale)
    + low-variance Normal(mu). Scales/df via softplus with stability clamps."""
    name = 'candle_mixture'
    K, P = 3, 9
    LOWVAR_SCALE = 0.1                                     # fixed tight sigma (Moirai's 4th comp)

    def __init__(self, mse_weight=1.0, **_):
        self.mse_w = float(mse_weight)

    def aux_dim(self, nH):
        return nH * self.P

    def loss(self, candles, aux, target, close_ch, weight):
        import torch
        import torch.nn.functional as F
        from torch.distributions import Normal, StudentT
        w = weight if weight and weight > 0 else 1.0
        t = target[:, close_ch, :]                                       # [B, nH] close move
        p = aux.view(aux.shape[0], t.shape[1], self.P).float()           # NLL wants fp32
        logw = F.log_softmax(p[..., 0:3], dim=-1)                        # mixture weights
        mu_t, sc_t = p[..., 3], F.softplus(p[..., 4]).clamp(1e-3, 100.0)
        df = (2.1 + F.softplus(p[..., 5])).clamp(max=100.0)              # df>2 -> finite variance
        mu_n, sc_n = p[..., 6], F.softplus(p[..., 7]).clamp(1e-3, 100.0)
        mu_l = p[..., 8]
        tf = t.float()
        lp = torch.stack([StudentT(df, mu_t, sc_t).log_prob(tf),
                          Normal(mu_n, sc_n).log_prob(tf),
                          Normal(mu_l, self.LOWVAR_SCALE).log_prob(tf)], dim=-1)
        nll = -(torch.logsumexp(logw + lp, dim=-1)).mean()               # mixture NLL (Moirai)
        mse = ((candles - target) ** 2).mean() if self.mse_w > 0 else 0.0
        return self.mse_w * mse + w * nll


DIST_OBJECTIVE_CLASSES = {c.name: c for c in (CandleQuantile, CandleBins, CandleMixture)}


def get_dist_objective(name, **params):
    """Resolve + CONFIGURE a DISTRIBUTIONAL objective by name (None -> 'candle_quantile').
    params (mse_weight / quantile_taus / bins_k) default to the original refine-study behavior.
    KeyError = fail fast."""
    return DIST_OBJECTIVE_CLASSES[name or 'candle_quantile'](**params)


class _DistForecastTrainer(_ForecastTrainer):
    """The stage-2 trainer with the objective swapped to a distributional one. Pure subclass —
    forecast.py is imported, never modified; net/batching/val (universal dir_acc) all inherited."""

    def __init__(self, big, tr, va, *, objective='candle_quantile', mse_weight=1.0,
                 quantile_taus='lohi', bins_k=41, **kw):
        super().__init__(big, tr, va, objective='candle_mse', **kw)      # base init (placeholder obj)
        self.obj = get_dist_objective(objective, mse_weight=mse_weight,  # swap BEFORE build_net()
                                      quantile_taus=quantile_taus, bins_k=bins_k)


def train_ssl_forecast_dist(big, train_starts, val_starts, *, horizons=(5, 10, 20, 25),
                            context_lengths=(64, 100, 150, 200), new_channels=8, epochs=60,
                            steps_per_epoch=200, batch=512, lr=1e-4, weight_decay=0.05, patience=8,
                            device=None, model_id='paris-noah/Mantis-8M', backbone_ckpt=None,
                            compile_model=False, control='real', seed=0, amp_dtype='fp16',
                            grad_clip=1.0, clamp=10.0, verbose=True,
                            ckpt_path=None, resume=False, freeze_encoder_layers=0,
                            objective='candle_quantile', dir_weight=1.0, dir_close_ch=3,
                            mse_weight=1.0, quantile_taus='lohi', bins_k=41, **_ignore):
    """Distributional forecast (stage-2.5). TWO experiment shapes, same trainer:
      REFINE  backbone_ckpt = the promoted stage-2 encoder; mixed loss (mse_weight=1).
      PRIMARY backbone_ckpt = the stage-1 masked encoder; PURE Chronos loss (mse_weight=0,
              quantile_taus='bolt9' / bins_k 129+) — distributional pretraining from the ground
              up, the faithful head-to-head vs the MSE stage-2.
    Returns (best_encoder_state, history) with the same metrics as stage-2 ('skill',
    'skill_per_h', 'dir_acc', 'std') — comparable across objectives (dir_acc is universal,
    read off the candle head)."""
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
                                dir_close_ch=dir_close_ch, mse_weight=mse_weight,
                                quantile_taus=quantile_taus, bins_k=bins_k).fit()
