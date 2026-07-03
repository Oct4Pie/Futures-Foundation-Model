"""Pluggable FORECAST OBJECTIVES — how the multi-horizon candle forecast is supervised.

Add a variant (direction head, quantile, excursion, ...) as ONE class registered below; the forecast
net + trainer never change (no per-variant if-chains — the trainer just calls the resolved objective).
Each objective declares how many EXTRA head outputs it needs beyond the candle decoder (`aux_dim`) and
its `loss`. A universal per-horizon DIRECTIONAL accuracy is shared — computed off the candle prediction
so it's comparable across every variant regardless of aux head.

Torch-free at IMPORT (torch is pulled in lazily inside the one loss that needs it) so the registry —
resolve-by-name + aux_dim — is testable without the torch/xgboost stack (see tests/conftest.py: torch
imported at module top segfaults the shared xgboost suite via the macOS libomp collision).
"""


def dir_acc(candles, target, close_ch):
    """Universal per-horizon directional accuracy: sign of the predicted close move vs actual, read
    off the CANDLE prediction so it's available for every objective and comparable across configs."""
    return ((candles[:, close_ch, :] > 0) == (target[:, close_ch, :] > 0)).float().mean(0)


class ForecastObjective:
    """A forecast-supervision strategy. Subclass -> set `name`, optionally override `aux_dim` + `loss`."""
    name = 'base'

    def aux_dim(self, nH):
        return 0                                          # extra aux-head outputs (0 = candle-only)

    def loss(self, candles, aux, target, close_ch, weight):
        raise NotImplementedError

    def diagnostics(self, aux, target, close_ch):
        """Optional per-objective VAL diagnostics (health reads, e.g. mixture collapse). Default
        none. Called under no_grad in the trainer's val_eval; must return a {str: float} dict."""
        return {}


class CandleMSE(ForecastObjective):
    """Original stage-2: MSE on the future candle move (all channels). No aux head."""
    name = 'candle_mse'

    def loss(self, candles, aux, target, close_ch, weight):
        return ((candles - target) ** 2).mean()


class CandleDirection(ForecastObjective):
    """Candle MSE + a LINEAR direction head: BCE on sign(fwd close move) per horizon. MSE says 'how
    far' (and mean-regresses the sign toward 0); BCE says 'which way' with no mean-regression — so it
    forces the ENCODER to encode direction (the WR-relevant signal). `weight` = the BCE mix (dir_weight)."""
    name = 'candle_direction'

    def aux_dim(self, nH):
        return nH                                         # one direction logit per horizon

    def loss(self, candles, aux, target, close_ch, weight):
        import torch.nn.functional as F                    # lazy: keep the registry torch-free at import
        mse = ((candles - target) ** 2).mean()
        dtgt = (target[:, close_ch, :] > 0).float()
        return mse + weight * F.binary_cross_entropy_with_logits(aux, dtgt)


_OBJECTIVES = {o.name: o for o in (CandleMSE(), CandleDirection())}


def get_forecast_objective(name):
    """Resolve a forecast objective by name (None/'candle_mse' = original). Unknown -> KeyError (fail fast)."""
    return _OBJECTIVES[name or 'candle_mse']
