# Revised Mantis SSL Objectives

## Decision

The legacy masked-candle and candle-MSE forecast objectives are retained for reproducibility but
are not used for the new MantisV1/V2 pilots. Two new versioned pretexts isolate the intended work:

- `structure_mask`: revised Stage 1 structural span reconstruction.
- `path`: revised Stage 3 fixed-wall-clock future-path supervision.

Neither task is a trading policy. Their heads are discarded; only the encoder is exported. The
existing sealed pullback ruler remains unchanged and is used only to measure incremental frozen
representation value.

## Stage 1 `structure_mask`

- 256-bar OHLCV context.
- Contiguous timestamp spans shared by every channel pass.
- Raw-price-derived targets computed only from the input context:
  close log return, log true-range/scale, body/scale, upper and lower wick/scale, and log-volume
  change.
- Smooth-L1 loss only on masked timestamps.
- Frozen entry-backbone feature anchoring.
- Stream-balanced sampling across all 54 symbol/timeframe streams.

This avoids reconstructing independently standardized absolute candles and prevents one channel
pass from seeing a timestamp hidden from another.

## Stage 3 `path`

- 256-bar OHLCV context.
- Exact elapsed horizons of 60, 180 and 360 minutes.
- Per-stream step conversion: 360 bars at 1-minute through 6 bars at 60-minute.
- Every context is filtered so its complete label remains inside its temporal split and contract.
- Future rows are targets only and never enter the encoder.
- Initial core targets:
  forward realized volatility, favorable MFE quantiles, adverse MAE quantiles, and
  termination/continuation/reversal.
- Monotone nonnegative excursion quantiles with pinball loss, robust log-volatility loss,
  cross-entropy path-class loss, and frozen entry-backbone anchoring.
- BF16 mixed precision; FP16 was rejected after a V2 smoke gradient-scale skip.

Barrier policy outcomes, fees, stops, entries, net R and strategy-specific utility are deliberately
excluded from the pretext objective.

## Leakage and parity controls

- Training `[2019-07-01, 2024-07-01)`.
- Validation `[2024-07-01, 2025-07-01)`.
- Rows at or after 2025-07-01 physically excluded.
- Future-path tests prove changing rows after one horizon cannot alter that horizon's labels.
- Price-scale tests prove structural targets preserve geometry under a common price scaling.
- Exact-resume signatures bind objective versions, horizons, weights, masking and anchoring.
- Checkpoint and deployment bundle behavior remains encoder-only and versioned.

## Upstream audit

Upstream `johnamcruz/Futures-Foundation-Model` was fetched and reviewed on 2026-07-17.

- `ssl/stage-2.7-nextleg-path` adds unitless within-leg retrace. The idea is useful as a later
  one-factor roughness target, but the implementation is pivot-specific, bar-horizon based and
  tied to the upstream next-leg lineage. It is not merged into the fixed-wall-clock core.
- `feature/exhaustion-head-in-produce` enforces coherent checkpoint/head bundles and predicts
  structural trend exhaustion. It requires external lifecycle labels and uses a separate 64-bar,
  2025-calibration contract. It belongs to downstream/production policy work, not this pretext.
- Upstream per-stream score scaling and deploy ladders are production-head changes and are outside
  the frozen foundation-model experiment.

No upstream commit was merged wholesale.

## Verification completed before full pilots

- Revised objective tests: 5 passed.
- Torch-enabled SSL suite: 90 passed.
- Full default suite: 908 passed, 90 skipped, 8 pre-existing warnings.
- V1/V2 structure and path smoke runs passed on CUDA.
- Initial global path reservation failed safely by removing CL@60-minute validation coverage; it
  was replaced with per-stream future filtering, after which all 54 streams remained eligible.

## Pilot order

Run one seed for four direct-from-vanilla arms:

1. MantisV1 `structure_mask`.
2. MantisV2 `structure_mask`.
3. MantisV1 `path`.
4. MantisV2 `path`.

An objective is chained only if it beats its own vanilla backbone on the paired frozen pullback
ruler without material safety-target regression. A second seed is funded only after that gate.
