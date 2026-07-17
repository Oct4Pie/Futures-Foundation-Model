# Trend Strategy Benchmark Results

Date: 2026-07-16
Status: complete for raw event strategies; learned foundation-model filtering not yet run
Legacy confirmatory interval read by this benchmark: **false**. The interval is not globally
pristine because dates in it were inspected by earlier project research.

## Scope correction

This document originally treated positive standalone expectancy as the main promotion criterion.
That is not the project's actual use case. The strategies are candidate event generators and label
anchors for foundation-model training. Their relevant criteria are breadth, effective quantity,
outcome diversity, causality, and non-redundancy. Raw economics remain a useful execution diagnostic,
but a negative unconditional pool is not grounds to discard a conditionally learnable setup.

The corrected model-pool assessment is in `MODEL_EVENT_POOL_ASSESSMENT.md` and its reproducible JSON
artifact. The raw execution findings below remain valid within their narrower scope.

## Raw-strategy verdict

None of the trend strategies currently wired into this repository is a promising standalone
futures strategy under executable assumptions.

The author's SuperTrend(10,3) flip is **not** the promising strategy suggested by the earlier
prototype report. Its apparent edge depended on a 0.5-ATR stop that was often smaller than one
exchange tick and a flat 0.03R cost assumption. After stops and targets were placed on executable
tick increments and only one optimistic round-trip tick of cost was charged, SuperTrend was the
worst family in the matched validation ruler.

The least-bad family is pivot/zigzag with structural stops. It is still negative overall. A few NQ
15/30-minute slices are weak research leads, not validated edges: only three of six calendar folds
were positive.

This does **not** prove that foundation-model filtering cannot turn a causal event pool into a useful
strategy. It does prove that raw SuperTrend or raw pivots should not be called profitable baselines,
and it prevents a model head from receiving credit for an impossible execution contract.

## Scope

The repository currently has two complete trading event families:

1. SuperTrend(10,3) direction flips.
2. Pivot events: ATR zigzag, k=2 fractal, and fractal-zigzag, with ATR and structural stops.

The matched tournament tested these ten variants:

- `supertrend__atr`
- `supertrend_htf__atr`
- `atr_zigzag__atr`
- `atr_zigzag__structural`
- `fractal_k2__atr`
- `fractal_k2__structural`
- `fractal_zigzag__atr`
- `fractal_zigzag__structural`
- `fractal_zigzag_htf__atr`
- `fractal_zigzag_htf__structural`

No claim is made here about strategy families that have not been implemented and benchmarked, such
as Donchian breakouts, pullback continuation, opening-range breakout, or time-series momentum.

## Matched execution contract

Every strategy used the same ruler:

- Symbols: ES, NQ, RTY, YM, GC, SI, CL, ZB, ZN.
- Timeframes: 1, 3, 5, 15, 30, and 60 minutes.
- OHLCV is segmented at contract rolls; detector state and active trades reset at every roll.
- Signal is known at bar close; entry occurs at the next bar open.
- Parent context requirement: 256 bars.
- Fixed wall-clock holding horizon: six hours, converted to 360/120/72/24/12/6 bars.
- Separate first-touch outcomes at 2R, 3R, 4R, and 6R; 3R is primary.
- If stop and target are both present in one OHLC bar, the stop wins.
- One active trade per strategy and stream; overlapping signals are suppressed.
- Proposed risk is rounded outward to at least one valid outright exchange tick.
- Baseline execution cost is one tick round trip. Zero-, two-, and three-tick sensitivities are
  recorded. Broker commissions are not included, so one tick is an optimistic lower bound.
- Six chronological reporting folds.
- The reserved period starting 2025-07-01 was not read.

The explicit tick sizes are ES/NQ 0.25, RTY 0.10, YM 1.0, GC 0.10, SI 0.005, CL 0.01,
ZB 1/32, and ZN 1/64. These are outright increments, not spread increments.

## Correctness defects found

### Retroactive HTF zigzag direction

The previous `_zigzag_dir` implementation filled the direction of a completed leg backward over
earlier bars. A historical feature therefore changed after a future reversal was confirmed. Mapping
the last closed HTF bar back to the lower timeframe did not remove this lookahead.

A prefix-invariance diagnostic failed on all 20 tested truncations. The implementation was replaced
with an online state output that only assigns the direction known at each bar. The same diagnostic
now reports zero mismatches, and a regression test protects it.

All HTF-gated results produced before this fix are invalid.

### Sub-tick risk and flat R cost

The original benchmark used `risk = 0.5 * ATR20`, raw floating-point stop/target prices, and a flat
0.03R charge. On quiet short bars, especially Treasury futures, risk was often one tick or less.
The previous positive ZB one-minute result was therefore not executable.

The corrected ruler rounds risk outward to a valid tick multiple and computes cost in R from the
actual number of risk ticks. In the corrected ZB one-minute SuperTrend arm, median risk is only one
tick. A one-tick round-trip cost is therefore 1R, not 0.03R.

## Validation-year results

Period: 2024-07-01 through 2025-06-30.
Configuration: author's 0.5-ATR stop for ATR-stop variants.
Artifact: `output/trend_strategy_benchmark_tick_dev_v1/report.json`
Events: 3,126,959.

| Strategy | Signals | WR@3R | Mean R, 1 tick RT | Mean R, 2 ticks RT | PF | Median risk ticks | Positive folds |
|---|---:|---:|---:|---:|---:|---:|---:|
| fractal_k2 structural | 162,066 | 0.2046 | -0.1344 | -0.2503 | 0.830 | 16 | 0/6 |
| fractal-zigzag structural | 136,170 | 0.2128 | -0.1353 | -0.2549 | 0.831 | 15 | 0/6 |
| ATR-zigzag structural | 171,166 | 0.2064 | -0.1396 | -0.2595 | 0.824 | 16 | 0/6 |
| fractal-zigzag HTF structural | 116,237 | 0.2287 | -0.1407 | -0.2710 | 0.830 | 13 | 0/6 |
| fractal-zigzag ATR | 408,093 | 0.2465 | -0.2744 | -0.5416 | 0.712 | 6 | 0/6 |
| fractal-zigzag HTF ATR | 225,790 | 0.2479 | -0.2749 | -0.5433 | 0.712 | 5 | 0/6 |
| fractal k=2 ATR | 705,181 | 0.2445 | -0.3003 | -0.5851 | 0.691 | 5 | 0/6 |
| ATR-zigzag ATR | 997,830 | 0.2389 | -0.3998 | -0.7590 | 0.616 | 4 | 0/6 |
| SuperTrend ATR | 151,649 | 0.2369 | -0.4639 | -0.8778 | 0.573 | 3 | 0/6 |
| SuperTrend HTF ATR | 52,777 | 0.2370 | -0.4791 | -0.9072 | 0.564 | 3 | 0/6 |

All ten variants fail the predeclared development gate. More importantly, all ten have negative
mean R in every aggregate calendar fold.

The result is not merely a fee objection. At zero tick cost, the best family is still slightly
negative: fractal-zigzag HTF ATR is -0.0065R/trade and PF 0.991. SuperTrend is approximately
-0.05R/trade before execution costs.

## Five-year calibration

Period: 2019-07-01 through 2024-06-30. This precedes the validation year.
ATR stop grid: 1.0, 1.5, and 2.0 ATR.
Artifacts:

- `output/trend_strategy_calibration_atr1.0_merged/report.json`
- `output/trend_strategy_calibration_atr1.5_merged/report.json`
- `output/trend_strategy_calibration_atr2.0_merged/report.json`

Wider stops improved the ATR-stop variants but did not produce a viable calibration candidate.
The best ATR-stop result was fractal-zigzag HTF at 2.0 ATR:

| Calibration arm | Signals | Mean R, 1 tick RT | PF | Positive folds | Worst fold |
|---|---:|---:|---:|---:|---:|
| fractal-zigzag HTF, 2.0 ATR | 460,197 | -0.1290 | 0.840 | 0/6 | -0.1527 |
| fractal-zigzag, 2.0 ATR | 590,391 | -0.1320 | 0.831 | 0/6 | -0.1584 |
| fractal k=2, 2.0 ATR | 744,249 | -0.1350 | 0.826 | 0/6 | -0.1482 |
| ATR-zigzag, 2.0 ATR | 839,618 | -0.1526 | 0.810 | 0/6 | -0.1618 |
| SuperTrend, 2.0 ATR | 475,845 | -0.1830 | 0.782 | 0/6 | -0.2054 |
| SuperTrend HTF, 2.0 ATR | 221,968 | -0.2072 | 0.761 | 0/6 | -0.2236 |

The 1.0- and 1.5-ATR variants were worse. Because no wider-stop configuration passed even the
pre-validation calibration period, no parameter was promoted to validation. This avoids selecting a
stop on the validation year after inspecting it.

## Slice findings

The strongest validation-year stream was ATR-zigzag with a structural stop on NQ 15-minute:

- 1,233 signals
- +0.0598R/trade
- PF 1.106
- Positive in only 3/6 folds
- Fold mean R: +0.1313, -0.0407, +0.1096, -0.0083, +0.2483, -0.0710

NQ 30-minute was similar (+0.0547R, PF 1.114) and also positive in only 3/6 folds. These are unstable
post-hoc slices and do not qualify as promising. They can be retained as diagnostic strata in a
future conditional model test, but not traded or sent to reserved OOS.

The earlier ZB one-minute lead disappeared completely after tick-valid execution. SuperTrend on that
stream has median one-tick risk and approximately -1.33R/trade after one round-trip tick.

## What is promising now

### As a standalone strategy

Nothing tested.

### As an event pool for a learned trade/no-trade filter

The following are worth one bounded downstream test, in this order:

1. ATR-zigzag with structural stop: best NQ slices, causal confirmation, executable risk, and many
   events.
2. Fractal-zigzag with structural stop: similar aggregate behavior and a cleaner structural pivot
   interpretation.
3. SuperTrend(10,3): control arm only. It is simple and common, but the raw economics are materially
   worse.

For each event pool, the test must compare:

- Unfiltered raw events.
- A causal hand-feature logistic baseline.
- Frozen foundation representations plus the same logistic head.
- A shuffled-label control.

The metric that decides is realized R after tick-aware costs, with WR, PF, maximum drawdown, signal
count, fold dispersion, ticker/timeframe slices, and paired confidence intervals. A representation
probe or unconditional forward AUC cannot promote a trading strategy.

## Implications for trend detection

The foundation-model tournament showed that trend state *description* is easy to decode. This ruler
shows that entering every detected flip or confirmed pivot is not profitable. Those findings are
consistent: knowing that a trend exists is different from deciding whether the current setup will
continue far enough to pay for risk and execution.

The next head should therefore estimate conditional continuation, termination, and payoff—not repeat
the descriptive trend-efficiency task. Useful event-level targets include maximum favorable excursion,
maximum adverse excursion, continuation before reversal, and survival to 2R/3R/4R under the exact
executable barrier contract.

## Recommended next experiment

Build the cross-family downstream ruler for the two structural pivot pools above. Use only the
pre-2025-07 development data, chronological walk-forward fits, and the same tick-aware outcomes.
Compare the causal-feature control against the frozen/adapted finalist representations. Do not read
reserved OOS unless one arm clears a predeclared economic gate across folds and is demonstrably better
than the causal-feature baseline.

If no model-filtered arm clears that gate, stop improving the current three SSL stages for this
strategy lane. Implement genuinely forward strategy candidates—breakout after compression, pullback
continuation, and time-series momentum—and rerun the same raw-event ruler before spending more GPU.

## Verification record

- Prefix-invariance diagnostic after HTF fix: 0/20 mismatches.
- Targeted tests: 20 passed.
- Full repository suite: 798 passed, 82 skipped, 0 failed.
- OOS guard: all reports record `oos_read=false`; evaluation ends at 2025-07-01.
- Shards are hash-checked before merge.
- Event artifacts, data fingerprints, source fingerprints, configuration, and scorer hashes are
  stored in the JSON reports.

Final `git diff --check` also passed. The worktree contains substantial pre-existing uncommitted
foundation-model work; these benchmark changes do not claim ownership of those unrelated edits.
