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

Upstream `main` was merged as commit `b359022`; the experimental next-leg and exhaustion branches
were deliberately not merged. The pilot below was trained from the merged, frozen source revision.

## Verification completed before full pilots

- Revised objective tests: 5 passed.
- Torch-enabled SSL suite: 90 passed.
- Full default suite after the upstream-main merge: 916 passed, 90 skipped, 8 pre-existing warnings.
- V1/V2 structure and path smoke runs passed on CUDA.
- Initial global path reservation failed safely by removing CL@60-minute validation coverage; it
  was replaced with per-stream future filtering, after which all 54 streams remained eligible.

## Frozen pilot decision

The earlier four-arm draft is retired before the canonical run. Generic descriptive adaptation has
already failed to produce reliable trading lift twice. The only funded branch is:

- MantisV1 vanilla -> direct `path`, seed 17.
- Five epochs, 50 steps/epoch, batch 64, 256 bars.
- Matched shuffle control and five-fold representation probe.
- Training/validation/holdout dates remain unchanged.

`structure_mask` remains implemented but unfunded. One pre-merge V1 structure diagnostic passed the
generic probe with only a small real-minus-shuffle margin; it is exploratory, is not a finalist and
will not be rerun on the sealed trading ruler.

Before training, promotion is fixed as all of:

1. Positive paired R-per-candidate lift over vanilla MantisV1 on the unchanged pullback ruler.
2. Positive lift in a declared majority of chronological folds.
3. Paired 95% weekly-block interval excluding zero after the declared comparison correction.
4. No material compression-control regression versus vanilla V1.
5. Positive fee-adjusted economics under the user-declared zero-tick primary ruler and survival of
   the frozen one-tick sensitivity.

Failure stops this objective. No loss-weight, horizon, event, head or threshold iteration is
permitted on the same development rows. A successful checkpoint remains exploratory and cannot
rewrite the already frozen confirmation finalists; it may join a future confirmation only under a
separately predeclared protocol before confirmation data are read.

## MantisV1 direct-path pilot result

The one-shot seed-17 pilot completed on 2026-07-17 and failed the locked promotion gate. It did not
read the OOS holdout beginning at `2025-07-01`.

Training and generic probe diagnostics:

- Best validation objective: `2.9797` at epoch 3.
- Real mean-core delta over vanilla: `+0.09103`.
- Shuffle-control mean-core delta: `+0.02348`.
- Temporal signal above shuffle: `+0.06755`.
- Target deltas were vol `+0.0694`, trend efficiency `-0.0016`, range expansion `+0.0350`,
  forward absolute move `+0.2613`, direction `+0.0141`, and forward direction `+0.0055`.
- The generic runner verdict was `all_pass=false`; descriptive/probe gains were not treated as an
  economic promotion.

The frozen encoder was then extracted on the identical 147,309-row development candidate artifact
and scored with the same nested chronological calibration, fee schedule, next-open execution,
adverse-first ambiguity and one-active-trade rule as vanilla MantisV1.

| Policy | Arm | Trades | Mean R | PF | Paired R/candidate vs vanilla | 95% weekly-block interval | Positive folds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Pullback continuation | V1 vanilla | 686 | +0.0961 | 1.139 | baseline | — | — |
| Pullback continuation | V1 path | 615 | +0.0081 | 1.012 | -0.01338 | [-0.03002, +0.00374] | 2/5 |
| Compression breakout | V1 vanilla | 426 | -0.0551 | 0.906 | baseline | — | — |
| Compression breakout | V1 path | 1,485 | +0.00043 | 1.001 | +0.00105 | [-0.00481, +0.00698] | 1/5 |

The frozen one-tick repricing sensitivity reduced V1 path to `-0.0802R`, PF `0.895` on pullbacks
and `-0.0404R`, PF `0.941` on compression. No arm was refit for this sensitivity.

### Decision

Reject this `path_v1` checkpoint. It improves several generic forward probes and changes which
compression candidates are selected, but it destroys most of the stronger vanilla pullback point
estimate and produces no statistically reliable paired lift on either control. It receives no
second seed and must not be evaluated on OOS.

The result does **not** establish that path supervision is categorically useless. It establishes
that this broad, policy-independent multi-horizon objective and weighting do not preserve the
useful vanilla representation under the locked budget. Per the precommitment, its loss weights,
horizons and class construction will not be tuned on the same development ruler. The frozen
MantisV1 vanilla checkpoint remains the lead representation for this lane.
