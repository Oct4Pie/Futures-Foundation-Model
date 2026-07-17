# Gate 3 Causal-Feature Baseline Results

Status: **complete causal ruler; foundation-representation comparison still pending**
Evaluation interval: `[2024-07-01, 2025-07-01)` only
OOS read: **false**

This document reports the fixed classical-feature ruler that every frozen foundation representation
must beat on the same rows. It is not a foundation-model ranking and does not authorize retraining.

## Sealed inputs and outputs

- Balanced sample: 64,800 rows, exactly 1,200 chronologically distributed contexts from each of
  54 symbol/timeframe streams.
- Symbols: ES, NQ, RTY, YM, GC, SI, CL, ZB and ZN.
- Timeframes: 1/3/5/15/30/60 minutes; 10,800 rows per timeframe.
- Context: 256 bars ending at the decision row.
- Horizons: fixed elapsed 1/3/6 hours.
- Sample SHA-256: `adbb643758bf706a16de4899ef3d38335cf146ea68df140ac5a5cb41aebdfb4f`.
- Score report SHA-256: `160e5af668b6c949dc674e57f1645a5bdf0ff201b8ac6332c6c67c1835b3a57a`.
- Prediction SHA-256: `19c31e9ad4d179f23b64d06b26ee46d449a6826d5e44d9f3d6a36ca423d466c9`.
- Weekly-bootstrap SHA-256: `f2d7e64d4d50b534ca50b45e15b157b09be45d64a9a058babdc5c47a6dcbefdb`.
- Saved row-level predictions: 6,550,592.

Every source shard was hash-verified before sampling. The saved sample also records the source shard
row, source decision row and stream identity for every selected context.

## Evaluation contract

- Five expanding calendar walk-forward folds per timeframe.
- Calendar support is restricted to the intersection covered by all nine symbols in that timeframe.
- A training row is retained only when its actual maximum label-end timestamp plus one complete
  cadence-specific 256-bar context embargo is no later than the next test block.
- The scaler is fit on the training fold only.
- Linear arm: Ridge for regression and logistic regression for binary targets.
- Tree arm: fixed constrained XGBoost; 120 trees, depth 3, learning rate 0.04, strong L2 and
  minimum-child regularization. No tuning was performed against these results.
- Inputs: 29 causal context features plus nine known-at-decision ticker indicators.
- Negative controls: shuffled training labels, random features and features time-destroyed within
  ticker. The latter two preserve/destroy different information by design; time-destroyed features
  retain static ticker effects.
- Paired uncertainty: 100 deterministic resamples of seven-day UTC calendar blocks for the primary
  real-versus-time-destroyed comparison. Finalists will use the larger predeclared resample count.

On 60-minute bars, the 1-hour horizon contains one forward return. Forward realized volatility and
forward trend efficiency are undefined with one return, so those two cells are explicitly skipped.
All other 60-minute/1-hour targets remain valid.

## Real causal-feature results

Each cell below is the unweighted mean of the five fold scores and then the available 1/3/6-hour
horizon scores within that timeframe. This is a compact diagnostic, not a promotion scalar.

### Linear head

| Timeframe | Abs move R2 | Fwd vol R2 | Fwd trend-eff R2 | Fwd direction AUC | Continuation AUC | Termination AUC |
|---|---:|---:|---:|---:|---:|---:|
| 1 min | 0.0568 | 0.5815 | -0.0150 | 0.5356 | 0.5015 | 0.5429 |
| 3 min | 0.0067 | 0.5607 | -0.0124 | 0.5005 | 0.5039 | 0.5406 |
| 5 min | 0.0273 | 0.5880 | -0.0129 | 0.5037 | 0.5065 | 0.5368 |
| 15 min | 0.0101 | 0.4788 | -0.0169 | 0.5177 | 0.5052 | 0.5725 |
| 30 min | 0.0737 | 0.3856 | -0.0162 | 0.5275 | 0.5178 | 0.5890 |
| 60 min | 0.0484 | 0.3380* | -0.0197* | 0.5301 | 0.5439 | 0.5863 |

### Constrained XGBoost head

| Timeframe | Abs move R2 | Fwd vol R2 | Fwd trend-eff R2 | Fwd direction AUC | Continuation AUC | Termination AUC |
|---|---:|---:|---:|---:|---:|---:|
| 1 min | 0.0979 | 0.5889 | -0.0151 | 0.5286 | 0.5032 | 0.5280 |
| 3 min | 0.0514 | 0.5915 | -0.0158 | 0.4913 | 0.5010 | 0.5284 |
| 5 min | 0.0501 | 0.5950 | -0.0127 | 0.5015 | 0.5067 | 0.5402 |
| 15 min | 0.0781 | 0.5200 | -0.0108 | 0.5135 | 0.5005 | 0.5952 |
| 30 min | 0.1323 | 0.4227 | -0.0142 | 0.5165 | 0.5095 | 0.6136 |
| 60 min | 0.0824 | 0.3547* | -0.0141* | 0.5259 | 0.5395 | 0.6102 |

`*` The 60-minute average uses only 3-hour and 6-hour cells for forward volatility/trend
efficiency because the 1-hour cell is undefined.

## Negative-control and uncertainty findings

Random features behave as a proper null: aggregate R2 is negative and AUC is approximately 0.50.
Time-destroyed and label-shuffled controls retain static ticker-level base rates; they are therefore
not expected to be exactly zero. Their purpose is to measure whether row-aligned market context adds
information beyond those static differences.

Paired weekly-block results for real context versus time-destroyed context:

| Target | Linear mean delta | Linear cells with 95% CI > 0 | XGBoost mean delta | XGBoost cells with 95% CI > 0 |
|---|---:|---:|---:|---:|
| Forward absolute move R2 | +0.0445 | 11/18 | +0.0900 | 18/18 |
| Forward realized-vol R2 | +0.3561 | 17/17 | +0.3292 | 17/17 |
| Forward trend-eff R2 | -0.0054 | 0/17 | -0.0048 | 0/17 |
| Forward direction AUC | -0.0019 | 0/18 | +0.0021 | 1/18 |
| Trend continuation AUC | +0.0109 | 4/18 | +0.0080 | 3/18 |
| Trend termination AUC | +0.0490 | 11/18 | +0.0661 | 14/18 |

## Findings

1. Forward realized volatility is a valid positive control. The old single-terminal-move target
   understated how much forward information exists in causal market context.
2. Absolute move size is weakly but consistently learnable by the constrained tree.
3. Trend termination is materially easier than trend continuation under the current target.
4. Unconditional forward direction remains too close to chance to rank models reliably.
5. Forward trend efficiency is not learned by either classical ruler and may be dominated by path
   noise/nonstationarity under this one-year fold contract.
6. A foundation representation is useful only if it adds stable paired lift over these exact rulers,
   especially on continuation/termination or the later executable-R scorecard. High descriptive
   trend decoding alone is insufficient.

## Decision

Do **not** retrain the foundation models yet. The next funded work is to extract frozen vanilla and
existing Stage 1/2/3 representations on a single predeclared subset of this sealed sample, fit the
same heads/folds, and compare row-level paired deltas against this ruler. Only that comparison can
identify Case A-E and decide whether revised stages deserve GPU time.
