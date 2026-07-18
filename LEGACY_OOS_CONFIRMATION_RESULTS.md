# Legacy OOS Confirmation Results

> **Historical adapter-contract warning:** the Mantis vectors and VICReg checkpoints in this
> confirmation were frozen under the historical adapter contract. The OOS measurements remain
> valid for those exact artifacts; they do not establish native representation, classification,
> training or deployment admission. New work is governed by
> [FOUNDATION_MODEL_NATIVE_CONTRACT_PLAN.md](FOUNDATION_MODEL_NATIVE_CONTRACT_PLAN.md).

## Decision

The frozen pullback-continuation lane produced positive conditional-selector economics on the
legacy OOS interval, but the foundation-model hypothesis did not confirm.

- The raw event pool was approximately break-even at zero slippage and failed at one tick.
- The causal barrier selector was the strongest robust arm: `+0.1253R/trade`, PF `1.185` at the
  primary zero-tick contract and `+0.0646R/trade`, PF `1.090` under the frozen one-tick repricing.
- Vanilla MantisV1 and MantisV2 were positive, but neither added statistically resolved utility
  over the causal selector.
- Both VICReg Stage 2 adaptations were worse than their vanilla family point estimates and became
  negative under one-tick repricing.

The correct conclusion is not that Mantis won. The conditional pullback lane survived this OOS
read, while incremental foundation-model value remains unproven and the tested SSL adaptation is
not deployable.

## Frozen protocol

The protocol was committed before the outcome read in commit `691cd80` and the zero-event stream
handling fix was committed in `36d3be6` before the completed scoring run.

| Field | Frozen value |
|---|---|
| Role | Legacy OOS confirmation; no training, validation, calibration or threshold fitting |
| Interval | `[2025-07-01T00:00:00Z, 2026-04-14T00:00:00Z)` |
| Instruments | ES, NQ, RTY, YM, GC, SI, CL, ZB, ZN |
| Timeframes | 1, 3, 5, 15, 30, 60 minutes |
| Event/policy | Pullback continuation, structural stop, 360-minute horizon, 3R target |
| Entry | Next bar open; no added delay |
| Ambiguous bar | Adverse-first |
| Concurrency | One active trade per ticker/policy/timeframe |
| Primary slippage | Zero round-trip ticks |
| Sensitivity | One round-trip tick, with frozen selections and no refit |
| Fees | Instrument-specific round-trip cash fees from `config/execution_costs.yaml` |

The interval was selected from common source timestamps, not performance. It has a known
prior-inspection caveat, but no OOS target or outcome entered training, validation, PCA, model
fitting, calibration, threshold selection or trade selection.

## How the causal selector works

“Causal” means all inputs are known at the completed decision-bar close. It does not mean that the
model estimates causal effects in the econometric sense. The selector is a development-trained
XGBoost filter over the causal pullback event pool; it contains no Mantis embedding.

### Candidate and feature contract

The pullback detector first emits a directional candidate from completed bars. It requires an
established trend, minimum directional efficiency and displacement, a controlled countertrend
retracement, and executable structural risk. Entry remains the next bar open.

The selector receives the following known-at-decision context features:

- Current candle range, body, upper/lower wick and close position.
- ATR fraction, log volume, close-to-EMA20, EMA20-to-EMA50 and normalized EMA50 slope.
- Prior 20-bar range and normalized distance above/below that range.
- Log return, realized volatility, trend efficiency, range and volume ratio over 4, 16, 64 and
  256 bars.
- Sine/cosine minute-of-week encoding.
- One-hot ticker identity.
- Event direction, log structural-risk ticks, and known fee/slippage/total-cost geometry.

No MFE, MAE, future return, future volatility, barrier outcome, exit price or realized R is an
input.

### Barrier-decomposed objective

For each timeframe, a three-class XGBoost classifier is trained on development events to estimate:

1. The 3R favorable barrier is reached first.
2. The 1R adverse barrier is reached first; same-bar ambiguity is included here.
3. Neither barrier is reached during the fixed 360-minute horizon.

A second XGBoost regressor is fit only on development “neither” events to estimate their terminal
gross R. The selector composes the outputs as:

```text
expected net R =
    P(favorable first) × 3R
  - P(adverse or ambiguous first) × 1R
  + P(neither) × predicted terminal R
  - known fees
  - declared slippage
```

The classifier and terminal regressor use 120 depth-3 trees, learning rate `0.04`, 80% row and
column subsampling, L2 regularization `10`, and minimum child weight `20`. These were frozen before
the OOS read.

### Calibration, threshold and execution

Raw development scores are converted to expected realized R with an isotonic map fit only to
chronological out-of-fold predictions. A separate threshold is frozen for each timeframe from
development folds. A threshold is admissible only when it has:

- At least 20 executed development trades.
- At least 2% executed coverage.
- A positive one-standard-error lower confidence bound for R per original candidate.

If no candidate threshold satisfies the rule, the declared action is no trade. That is why the
causal OOS arm took no 1-minute trades. After selection, the concurrency layer allows one active
trade per ticker/policy/timeframe and suppresses overlapping signals until the active event exits.

### Important comparison limitation

The causal arm and the Mantis arms do not differ only by representation. The causal arm uses the
barrier-decomposed head above, while the frozen Mantis comparison arms used PCA-reduced embeddings
plus the same causal features with direct realized-R regression. Therefore this OOS comparison
establishes that the complete causal-selector pipeline was stronger; it does not attribute the
entire difference to hand features versus Mantis hidden states. A future representation experiment
must hold the barrier head and rows identical across causal-only and causal-plus-embedding arms.

## Data and breadth

The run materialized 3,660 causal pullback contexts of shape `256 × 5` from all nine roots. Four
Mantis representation matrices were extracted, each with shape `3,660 × 1,280`.

| Timeframe | Candidates | Share |
|---|---:|---:|
| 1 minute | 1,488 | 40.7% |
| 3 minutes | 838 | 22.9% |
| 5 minutes | 635 | 17.3% |
| 15 minutes | 474 | 13.0% |
| 30 minutes | 185 | 5.1% |
| 60 minutes | 40 | 1.1% |

| Root | Candidates |
|---|---:|
| ES | 765 |
| NQ | 744 |
| RTY | 375 |
| YM | 383 |
| GC | 511 |
| SI | 303 |
| CL | 381 |
| ZB | 89 |
| ZN | 109 |

The source reaches April 13 for all 54 requested streams, but timestamp endpoint coverage is not
the same as dense coverage. In particular, ZB has only 25 materialized dense rows at 1 minute and
1,461 at 3 minutes, producing no pullback candidates in those two streams. The global conclusions
do not depend on fabricated or forward-filled ZB events, but the run is not evidence of strong
short-timeframe ZB breadth.

## Primary results: fees and zero slippage

Every row includes the configured instrument-specific cash fee. No tick slippage was added at the
primary point, as predeclared.

| Arm | Trades | WR | Mean R/trade | PF | Total R | Max DD (R) | Roots / TFs |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw candidates | 2,705 | 27.87% | -0.0029 | 0.996 | -7.71 | 79.53 | 9 / 6 |
| Causal barrier selector | 773 | 32.86% | **+0.1253** | **1.185** | +96.84 | 32.81 | 9 / 5 |
| MantisV1 vanilla + causal | 996 | 31.43% | +0.1052 | 1.152 | +104.82 | 36.75 | 9 / 5 |
| MantisV1 VICReg S2 + causal | 1,064 | 30.26% | +0.0648 | 1.092 | +68.95 | 48.75 | 9 / 5 |
| MantisV2 vanilla + causal | 1,281 | 30.68% | +0.0809 | 1.115 | +103.65 | 32.77 | 9 / 5 |
| MantisV2 VICReg S2 + causal | 1,323 | 30.01% | +0.0492 | 1.069 | +65.03 | 46.78 | 9 / 5 |

The higher total R for vanilla V1 is caused by taking more trades; it does not beat the causal
selector on R per trade or on the paired incremental-value test.

## Frozen one-tick repricing

This is a sensitivity analysis over exactly the same selected executions. Nothing was retrained,
recalibrated or reselected.

| Arm | Trades | Mean R/trade | PF | Total R | Max DD (R) |
|---|---:|---:|---:|---:|---:|
| Raw candidates | 2,705 | -0.0809 | 0.898 | -218.75 | 233.97 |
| Causal barrier selector | 773 | **+0.0646** | **1.090** | +49.93 | 49.67 |
| MantisV1 vanilla + causal | 996 | +0.0300 | 1.040 | +29.88 | 54.85 |
| MantisV1 VICReg S2 + causal | 1,064 | -0.0067 | 0.991 | -7.09 | 66.00 |
| MantisV2 vanilla + causal | 1,281 | +0.0078 | 1.010 | +9.99 | 50.14 |
| MantisV2 VICReg S2 + causal | 1,323 | -0.0282 | 0.963 | -37.30 | 88.41 |

The causal selector is the only arm with a meaningful one-tick buffer. Vanilla MantisV1 is
marginal; vanilla V2 is effectively break-even. Both adapted arms fail.

## Paired weekly-block inference

Utility is measured per original candidate, so rejected candidates contribute zero and cannot be
hidden by reporting only selected-trade averages. Intervals use 42 calendar-week blocks and 5,000
bootstrap repetitions.

### Incremental utility over raw candidates

| Arm | Delta R/candidate | 95% interval | P(delta > 0) |
|---|---:|---:|---:|
| Causal barrier | +0.02857 | [-0.00671, +0.06294] | 0.9448 |
| MantisV1 vanilla | +0.03075 | [-0.00824, +0.07023] | 0.9342 |
| MantisV1 VICReg S2 | +0.02095 | [-0.01751, +0.05791] | 0.8630 |
| MantisV2 vanilla | +0.03043 | [-0.00460, +0.06439] | 0.9570 |
| MantisV2 VICReg S2 | +0.01988 | [-0.01227, +0.05092] | 0.8884 |

All intervals cross zero. The positive point economics are useful evidence, but the weekly-block
ruler does not resolve the improvement over the raw pool at the predeclared 95% level.

### Foundation-model value over causal features

| Arm | Delta R/candidate vs causal | 95% interval | P(delta > 0) |
|---|---:|---:|---:|
| MantisV1 vanilla | +0.00218 | [-0.02254, +0.02653] | 0.5712 |
| MantisV1 VICReg S2 | -0.00762 | [-0.02934, +0.01597] | 0.2654 |
| MantisV2 vanilla | +0.00186 | [-0.02220, +0.02569] | 0.5598 |
| MantisV2 VICReg S2 | -0.00869 | [-0.03526, +0.01689] | 0.2502 |

The vanilla embeddings add nearly zero paired utility over the causal selector. The intervals are
wide enough to include modest benefit or harm, so “no confirmed incremental value” is the correct
claim—not proof that the hidden states contain no information.

### Adapted versus vanilla family

| Comparison | Delta R/candidate | 95% interval | P(delta > 0) |
|---|---:|---:|---:|
| MantisV1 VICReg S2 - V1 vanilla | -0.00980 | [-0.02883, +0.01055] | 0.1604 |
| MantisV2 VICReg S2 - V2 vanilla | -0.01055 | [-0.03644, +0.01563] | 0.2174 |

Neither adaptation beats vanilla. Both point estimates are negative and both lose their positive
economics at one tick. The tested descriptive VICReg objective is rejected for this trading task.

## Timeframe diagnostic

This table is descriptive only; no timeframe was selected or removed after the OOS read.

| Arm | 1m | 3m | 5m | 15m | 30m | 60m |
|---|---:|---:|---:|---:|---:|---:|
| Raw | -0.095 | +0.155 | +0.051 | +0.045 | -0.204 | -0.127 |
| Causal | no trade | +0.304 | +0.060 | +0.066 | -0.227 | -0.186 |
| V1 vanilla | +0.292 | +0.172 | -0.112 | +0.045 | no trade | -0.127 |
| V1 VICReg S2 | +0.124 | +0.152 | -0.054 | +0.027 | no trade | -0.127 |
| V2 vanilla | -0.119 | +0.171 | +0.048 | +0.051 | no trade | -0.100 |
| V2 VICReg S2 | +0.036 | +0.108 | +0.010 | +0.045 | no trade | -0.127 |

Values are zero-tick mean R per executed trade. The 60-minute slice has only 40 candidates and is
not inferentially useful. The concentration of gains at 3 minutes and the negative coarse slices
are monitoring risks, not authorization to retune this read.

## Leakage and integrity verification

The completed runner enforced these conditions before scoring:

- Every development decision, signal and exit timestamp is before `2025-07-01`.
- Every OOS decision and signal is on or after `2025-07-01`, and every OOS decision, signal and exit
  is before `2026-04-14`.
- Development embedding manifests are hash-bound to the saved development row selection.
- OOS embedding rows are extracted in the frozen OOS context order.
- PCA is fit on development embeddings, then applied to OOS embeddings.
- Heads are fit on development events only.
- Isotonic calibrators and thresholds are reconstructed from saved development OOF predictions.
- OOS outcomes are used only after scores and selections are fixed.
- The one-tick result reprices frozen executions without refitting.

Key artifacts:

| Artifact | SHA-256 |
|---|---|
| Canonical source manifest | `d7cb5f4273e72e30642019f91a0581da7aafc27c342813c09be0fb5c4fc97865` |
| OOS sample | `28027533d000007d0540b619c9d6b85d47a63e9611fb152039fbd17b615aac37` |
| OOS contexts | `95f35fbb634279726f13a189e3c251d5578c7f200fcbae91c6317ce03ea1b0e1` |
| Fee-only policy events | `64e37ec40dac5076f9a723cc87c52826f0698202cea59c1c41c3546cb43e5d72` |
| Frozen execution utilities | `f6adee23b63163e90173f3ea7b37f9119fea3a743fa789368ba03e91f764585c` |
| Final JSON report | `5e1fa9743d136e44a5c23863f988218d6ddd32cd435513beb7646faec117a02e` |

Canonical local report:
`output/foundation_tournament/legacy_oos_confirmation_v1/legacy_oos_confirmation.json`.

## Final status and next decision

1. Preserve the causal barrier selector and sealed ruler as the confirmed research deliverable.
2. Retain vanilla MantisV1/V2 as compatibility controls, not as proven value-adding components.
3. Do not deploy or further fund the current VICReg adaptations.
4. Freeze further SSL variants on this nine-root, 2019–2024 contract and its spent evaluation
   ruler. Generic representation-probe improvements repeatedly failed to transfer into paired
   trading utility at that scale.
5. Do not tune symbols, timeframes, thresholds, policies or objectives on this read.
6. Any new foundation-model hypothesis must use a newly predeclared development contract and earn
   a genuinely untouched prospective confirmation after it freezes.

The trading-policy and risk-management work remains outside this repository. This repository's
job is to preserve the representation artifacts, causal event/outcome contract and honest ruler.

This result is not a universal scaling verdict. The separate [data-scale audit](DATA_SCALE_AUDIT.md)
shows that substantially more governed historical tick data exists. Any scaled experiment is a
new contract, not permission to tune this completed OOS read.
