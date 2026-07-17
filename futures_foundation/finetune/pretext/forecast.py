"""Stage-3 pretext: multi-horizon / variable-context candle seq2seq (ANTI-SHORTCUT). Reserves
context+horizon per window. Gate additionally requires forward-move size up. Forward direction is
reported but not gated because its pooled AUC is inside the current probe's noise floor."""
from .base import PretextTask


class ForecastTask(PretextTask):
    name, trainer = 'forecast', 'train_ssl_forecast'
    primary_targets = ('fwd_absmove',)

    def reserve(self, cfg):
        return max(int(x) for x in cfg['context_lengths']) + max(int(h) for h in cfg['horizons'])

    def finalize_verdict(self, verdict, fc_skill, probe_res):
        verdict['forecast_skill'] = fc_skill
        if probe_res is not None:
            verdict['fwd_absmove_delta'] = float(probe_res.get('fwd_absmove_delta', 0.0))
            verdict['fwd_dir_delta'] = float(probe_res.get('fwd_dir_delta', 0.0))
        return verdict
