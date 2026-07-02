"""Stage-2.5 pretext: DISTRIBUTIONAL forecast refine (Chronos-style quantile / bin objectives) —
builds ON the promoted stage-2 seq2seq (warm-start = backbone_ckpt), original 'forecast' pretext
untouched. Same reserve (context + horizon) and the same anti-shortcut probe gate as stage-2 —
the targets are identical; only the loss geometry (mean -> distribution) changes."""
from .forecast import ForecastTask


class ForecastDistTask(ForecastTask):
    name, trainer = 'forecast_dist', 'train_ssl_forecast_dist'
