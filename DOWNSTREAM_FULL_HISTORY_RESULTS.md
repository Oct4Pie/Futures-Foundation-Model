# Full-History Calibrated Downstream Results

Date: 2026-07-17

## Decision

The corrected full-history ruler found one economically positive development cell, but it did not
show stable incremental value from a foundation representation. Broad SSL retraining remains
unauthorized.

On the ATR-zigzag structural-stop pool, Chronos Bolt Stage 1 produced `+0.1199 R/trade` with profit
factor `1.2179`. The causal-feature head produced `+0.1105 R/trade` with profit factor `1.1967`.
The apparent Chronos advantage disappears on the required matched-candidate comparison:
`-0.00050 R/candidate`, weekly-block 95% interval `[-0.00300, +0.00202]`, with positive lift in
only two of five chronological folds. Almost all Chronos executions came from one timeframe and
one fold. This is a useful lead, not a promotable result.

The fractal-k2 structural-stop pool remained economically negative for every arm. MantisV2 Stage 2
also remained negative on both primary pools. Current evidence is therefore consistent with a mix
of Cases A, C and D: causal features remain competitive, existing adaptation is inconsistent, and
event-pool quality still matters. Case E—representation quality demonstrably binding—has not been
established.

## Locked protocol

- Context/head history: eligible rows from `[2019-07-01, outer-test-start)`.
- Outer development evaluation: `[2024-07-01, 2025-07-01)` in five expanding folds.
- Legacy 2025-07 to 2026-07 data: not read.
- Context: 256 OHLCV bars at 1/3/5/15/30/60-minute cadence.
- Costs: instrument-specific round-trip fees retained; primary tick slippage `0`; no added entry
  delay. One- and two-tick slippage are frozen-selection sensitivity checks.
- Fills: causal strategy signal followed by the existing next-open execution contract.
- Same-bar barrier ambiguity: adverse-first for executable outcomes.
- Concurrency: one active trade per ticker, policy and timeframe.
- Calibration: nested expanding-fold raw predictions, train-only isotonic expected-net-R mapping,
  and a threshold requiring a positive chronological-fold lower confidence bound with minimum
  coverage and trade count. A failed calibration selects no trade.
- Leakage controls: every training row's complete label end plus a 256-context-bar embargo must
  precede the outer test boundary. PCA, XGBoost, isotonic calibration and threshold selection are
  all fit before the outer test fold.

The sealed full-history sample has 162,000 balanced contexts. The policy artifact has 1,833,912
events. The benchmark saved 104,706 matched outer-test predictions for two policies and six arms.
Every serialized fold contract passed the label-end/embargo and fixed-interval checks.

## Primary results

### ATR-zigzag v2, structural stop, 360 minutes, 3R

| Arm | Executed | Mean R | PF | WR | Instruments | Timeframes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Chronos Bolt Stage 1 + causal | 188 | +0.1199 | 1.2179 | 40.43% | 9 | 2 |
| Causal XGBoost | 255 | +0.1105 | 1.1967 | 40.39% | 9 | 3 |
| MantisV2 vanilla + causal | 224 | +0.0079 | 1.0125 | 33.04% | 9 | 4 |
| Chronos Bolt vanilla + causal | 219 | -0.0448 | 0.9296 | 32.42% | 9 | 5 |
| Raw event pool | 10,839 | -0.0784 | 0.8927 | 28.19% | 9 | 6 |
| MantisV2 Stage 2 + causal | 58 | -0.2336 | 0.6835 | 25.86% | 8 | 6 |

The aggregate Chronos result is concentrated: 187/188 executions are 30-minute signals, and
178 executions occur in fold 4 (`2025-02-05` to `2025-04-19`). Its paired lift over causal-only is
not positive. Its paired lift over vanilla Chronos is `+0.00285 R/candidate`, but the 95% interval
`[-0.00066, +0.00643]` crosses zero and only three of five folds are positive.

### Fractal k=2, structural stop, 360 minutes, 3R

| Arm | Executed | Mean R | PF | WR |
| --- | ---: | ---: | ---: | ---: |
| Raw event pool | 5,917 | -0.0269 | 0.9609 | 30.81% |
| Chronos Bolt vanilla + causal | 817 | -0.0582 | 0.9109 | 32.93% |
| MantisV2 Stage 2 + causal | 431 | -0.0674 | 0.8984 | 32.02% |
| Causal XGBoost | 1,129 | -0.1212 | 0.8236 | 30.20% |
| MantisV2 vanilla + causal | 719 | -0.1238 | 0.8054 | 32.13% |
| Chronos Bolt Stage 1 + causal | 686 | -0.1268 | 0.8112 | 31.20% |

MantisV2 Stage 2 has statistically positive utility lift over causal-only on this negative pool,
but it remains economically negative. Becoming less bad is not promotion evidence.

## Cost sensitivity

Selections are frozen from the zero-slippage primary run and only exact executed outcomes are
repriced.

| ATR arm | 0 ticks | 1 tick | 2 ticks |
| --- | ---: | ---: | ---: |
| Chronos Bolt Stage 1 + causal | +0.1199 R / PF 1.2179 | +0.0812 / 1.1404 | +0.0426 / 1.0701 |
| Causal XGBoost | +0.1105 / 1.1967 | +0.0731 / 1.1242 | +0.0358 / 1.0580 |
| MantisV2 vanilla + causal | +0.0079 / 1.0125 | -0.0358 / 0.9457 | -0.0796 / 0.8850 |

This sensitivity is informative but not a substitute for a slippage model. The requested primary
assumption remains fees-only and zero ticks.

## Audit of the author's `mantis_ssl_nextleg.pt` claim

The reported Mantis atlas log is promising on `pred_runner_6R` (AUC 0.7235),
`pred_vol_expand` (0.8420) and `pred_real_trend_start` (0.6465). The retention AUCs primarily show
that the embedding preserves contemporaneous state. `pred_reach_4R` (0.5347) and
`pred_stopped_out` (0.5473) are weak.

The log is not sufficient to establish an elite or OOS representation:

1. The checked-in training script sets `HOLDOUT_START=2026-01-01` and chooses the checkpoint using
   the last 10% of pre-2026 data. A probe reported as “eval 2025” therefore evaluates an encoder
   already selected with 2025 SSL validation data and possibly trained on part of 2025, depending
   on corpus coverage.
2. The historical next-leg reserve covered only one `leg_cap`, although the target reads two legs
   each allowed to reach that cap. Targets could cross the train/validation boundary.
3. Pivots were detected on the fully concatenated array, allowing artificial legs across stream or
   contract boundaries.
4. Bar-count targets are not timeframe-invariant. The same 20-bar label spans 20 minutes at one
   minute and five hours at 15 minutes.
5. The log supplies no paired causal-feature or vanilla-Mantis control, fold dispersion,
   per-symbol/timeframe breakdown, artifact hashes or auditable label implementation.

The code now reserves both future legs and computes next-leg targets independently inside each
stream/contract segment. Historical `mantis_ssl_nextleg.pt` weights were trained under the old
contract and are not retroactively repaired. A clean replication also needs fixed elapsed-time
targets or explicit per-timeframe heads, strict train/validation dates, and the same downstream
ruler used here.

## Artifact identities

- Balanced sample SHA-256: `28642ed0d0965bcfc4eda86fbd7ef7e1bc6a8529521ff8a9c9ffe9ab2e89e201`
- Row selection SHA-256: `372ce068513413795dccd2b8608aa41bd49bd5918162c475f2b1e89a4619e13f`
- Policy events SHA-256: `415fc0eb8ed63159edd91b85fbd5e50a5a84e63730e938d84e1e954cec9879d2`
- Trading results SHA-256: `819dddf69386f951fa3754c59be8de527d957b094569031fe31d6fa94a4ccbaa`
- Analysis versus causal SHA-256: `6c7cfe961e2fa9346e9d61186d014bf5fb7ceeb7856083735dadf20151404881`
- Analysis versus Chronos vanilla SHA-256: `77e5b12ca1a241c707fb6ebbce4b61ffc473feab57a8dcbef094c7e1ec678ebb`
- Analysis versus Mantis vanilla SHA-256: `66ff0429e21e48177482c90dbc870d07d3b23bc37c7095dfdb8234067ad74170`

Canonical local report:
`output/foundation_tournament/downstream_full_history_v1/trading_nested_isotonic_primary_seed20260716/trading_results.json`.

## Next decision

Do not launch a broad Stage 1→3 sweep. The next bounded downstream experiment is decomposed barrier
outcomes plus residual embedding fusion on the ATR-zigzag structural pool. That test asks whether
Mantis or Chronos adds information beyond the causal head without allowing a large embedding to
destabilize it. Only stable paired lift over causal and vanilla controls can authorize revised SSL.
