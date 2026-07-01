"""Stage-2 pretext: multi-horizon / variable-context candle seq2seq (ANTI-SHORTCUT). Reserves
context+horizon per window. Gate additionally requires forward-move size up + forward-direction
non-regress (a shortcut embedding can lift easy descriptive stats while the predictive forward
targets barely move, so the descriptive average alone is not enough)."""
from .base import PretextTask


class ForecastTask(PretextTask):
    name, trainer = 'forecast', 'train_ssl_forecast'

    def reserve(self, cfg):
        return max(int(x) for x in cfg['context_lengths']) + max(int(h) for h in cfg['horizons'])

    def _decide(self, probe_res, no_collapse, margin, dir_margin, detail):
        desc_ok = bool(probe_res.get('descriptive_delta', 0.0) >= -1e-9)
        fwd_size_ok = bool(probe_res.get('fwd_absmove_delta', 0.0) > margin)
        fwd_dir_ok = bool(probe_res.get('fwd_dir_delta', 0.0) >= dir_margin)
        detail.update({'descriptive_ok': desc_ok, 'fwd_size_ok': fwd_size_ok, 'fwd_dir_ok': fwd_dir_ok})
        return bool(no_collapse and desc_ok and fwd_size_ok and fwd_dir_ok), detail

    def finalize_verdict(self, verdict, fc_skill, probe_res):
        verdict['forecast_skill'] = fc_skill
        if probe_res is not None:
            verdict['fwd_absmove_delta'] = float(probe_res.get('fwd_absmove_delta', 0.0))
            verdict['fwd_dir_delta'] = float(probe_res.get('fwd_dir_delta', 0.0))
        return verdict
