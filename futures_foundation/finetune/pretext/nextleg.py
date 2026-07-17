"""Stage-2.6 pretext: NEXT-LEG forecasting — the seq2seq family, prediction unit = THE TREND LEG.

From a context window ending at a CONFIRMED fractal pivot, predict the market's own next pattern
element in BARS (user spec 2026-07-16 — bar-counts are instrument- and TF-agnostic, the fractal
unit; NO ATR anywhere — ATR is derived/volatile, raw candles only): (1) how many bars the newborn
leg runs, (2) how many bars the counter-leg (the retest) lasts. Pivots = the PURE fractal detector
(k-bar swing high/low comparisons on raw candles, detect_fractal_pivots — no ATR filter), legs =
alternating swings (keep-first). Targets come from that same deterministic structure applied to
future price — pure self-supervision, zero labels, zero strategy, zero derived indicators.
Direction is NOT predicted (legs alternate: known at confirm — degenerate target).

Why this survives where turn-electra/lc512 died: it stays a FORECASTER (generic objective, the
input/prediction aimed at the pattern unit — the banked design law), keeps the candle-seq2seq
ANCHOR (mse_weight, the anti-drift piece proven load-bearing), warm-starts from the promoted
base with early blocks frozen (the reorder recipe). GATE: the trend-lifecycle scorecard — the
new checkpoint must BEAT the banked per-direction probes (0.7689/0.7635 start, 0.7519/0.7603
end) BEFORE any WR pipeline touches it; the probe gate here is the anti-shortcut forecast gate.
"""
from .forecast import ForecastTask


class NextLegTask(ForecastTask):
    name, trainer = 'nextleg', 'train_ssl_nextleg'

    def reserve(self, cfg):
        # Both future legs may independently reach leg_cap. Reserving only one cap lets a training
        # target read across the train/validation boundary when t1+t2 > leg_cap.
        return max(int(x) for x in cfg['context_lengths']) + 2 * int(cfg.get('leg_cap', 256))

    def finalize_verdict(self, verdict, fc_skill, probe_res):
        verdict = super().finalize_verdict(verdict, fc_skill, probe_res)
        verdict['pretext_note'] = ('next-leg forecaster (bars): judge on the trend-lifecycle '
                                   'scorecard FIRST (must beat the banked 0.75-0.77 per-direction '
                                   'probes), then the standard WR pipeline; leg-skill is a '
                                   'learning diagnostic only')
        return verdict
