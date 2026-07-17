# Legacy OOS Confirmation Protocol

## Frozen decision

This protocol was committed before reading policy outcomes from the confirmation interval.

The original full-year endpoint cannot be reconstructed from the canonical bar corpus for all nine
roots. The protocol therefore uses the complete common-coverage interval determined only from
source timestamps:

- Start inclusive: `2025-07-01T00:00:00Z`
- End exclusive: `2026-04-14T00:00:00Z`
- Instruments: ES, NQ, RTY, YM, GC, SI, CL, ZB and ZN
- Timeframes: 1, 3, 5, 15, 30 and 60 minutes
- Policy: `pullback_continuation__structural_stop__360m__3R`

The endpoint is not selected from outcomes. It is the end of the final complete UTC date shared by
the original canonical source: CL, GC and SI end on April 13, 2026.

## Frozen arms

1. All causal pullback candidates under the execution concurrency rule.
2. Causal features with the barrier-decomposed selector.
3. Vanilla MantisV1 plus causal features, direct-R selector.
4. Vanilla MantisV2 plus causal features, direct-R selector.
5. MantisV1 VICReg Stage 2 plus causal features as a negative adaptation control.
6. MantisV2 VICReg Stage 2 plus causal features as a negative adaptation control.

No path-objective checkpoint is eligible because it failed development promotion.

## Fitting and leakage contract

- SSL checkpoints remain unchanged.
- Final selector heads are fit only on the existing pre-OOS development candidate artifact.
- PCA/scaling is fit only on pre-OOS embeddings.
- Isotonic mappings and operating thresholds are fit only from saved pre-OOS out-of-fold
  predictions.
- Every development label exit must precede `2025-07-01`.
- Every confirmation signal and exit must remain inside the confirmation interval.
- No OOS target, realized R, barrier state, exit or future bar may affect preprocessing, PCA,
  selector fitting, calibration, thresholds or selection.
- Outcomes are loaded for measurement only after scores and selections are frozen.

## Execution contract

- Signal at completed decision-bar close.
- Entry at the next bar open with no added delay.
- Structural stop and 3R target over a fixed 360-minute horizon.
- Same-bar target/stop ambiguity resolves adverse-first.
- One active trade per ticker, policy and timeframe.
- Instrument-specific cash fees are charged.
- Primary slippage is zero round-trip ticks, per the user-declared research contract.
- One round-trip tick is a frozen-selection repricing sensitivity; no head, calibration, threshold
  or trade selection is refit for it.

## Interpretation fixed before the read

- The event pool confirms at the primary cost point only if mean R is positive and PF exceeds one.
- It is execution-robust only if the same frozen executions remain positive with PF above one at
  one tick.
- A representation adds confirmed incremental value only if its paired weekly-block R-per-candidate
  interval versus the causal selector excludes zero on the positive side.
- An adapted representation beats its vanilla family only if the corresponding paired interval
  excludes zero on the positive side.
- Point-estimate improvement with an interval crossing zero is unconfirmed, not a promotion.
- Failure does not authorize threshold, event, objective or finalist changes on this OOS interval.

The interval has a known prior-inspection caveat, but it was not used for model fitting,
validation, calibration or selection. It is treated as legacy OOS under the user's definition.
