# Foundation-Model Tournament Review and Next-Experiment Specification

Status: decision document for the next agent.
Date: 2026-07-16 America/Los_Angeles.
Evidence source: sealed validation artifacts; no reserved-OOS read.

## Purpose

This document reconciles the completed foundation-model tournament with the subsequent methodological review. It is intentionally separate from `FOUNDATION_MODEL_RESULTS.md`:

- `FOUNDATION_MODEL_RESULTS.md` records what was run and the resulting scores.
- This document explains what those scores do and do not establish, corrects overstatements, and specifies the next experiment.

Do not overwrite the original result artifacts when implementing this plan.

## Terminal objective

Everything in this program exists to trade futures. The final unit of account is realized trading performance — the R distribution per trade, win rate, profit factor, trade count/coverage, and drawdown in R, after declared costs — measured by the honest-ruler strategy benchmarks (`wf.py` walk-forward, `benchmark_supertrend_mantis.py`, `benchmark_fractal_mantis.py`, and the cross-family successors specified below).

Every other number in this program — probe R², probe AUC, forecast skill, contrastive gate scores — is an intermediate diagnostic. A representation change that improves probes but does not improve realized R at fixed, predeclared selection rules is not progress. Representations are promoted, retired, or deployed on trading utility; probe metrics only explain *why* a result happened and select which candidates are worth the cost of a trading benchmark.

## Executive decision

Do **not** launch another full Stage 1 → Stage 2 → Stage 3 training sweep yet.

The highest-value next work is:

1. Re-score the existing cached representations with better forward targets, timeframe slices, causal controls, and paired uncertainty estimates.
2. Test whether frozen representations improve conditional trade selection on strategy-generated events.
3. Redesign and retrain Stage 2 or Stage 3 only if the downstream experiment shows that representation quality is the binding constraint.

The tournament supports a narrower conclusion than the original results report stated:

> Descriptive market state is strongly decodable from several representations. The current unconditional linear probes do not demonstrate stable forward prediction. They do not establish that the representations are useless as conditional regime filters for strategy setups.

That distinction matters. The repository’s stated architecture is “BERT for futures”: the backbone learns regime, structure, and volatility; strategy-specific heads decide whether to trade. The next experiment must test that division of labor directly.

## Verified tournament facts

### Data and split contract

| Item | Verified value |
|---|---|
| Training interval | `[2019-07-01, 2024-07-01)` |
| Validation interval | `[2024-07-01, 2025-07-01)` |
| Reserved OOS boundary | `2025-07-01` |
| OOS read by representation benchmark | `false` |
| Symbols | ES, NQ, RTY, YM, GC, SI, CL, ZB, ZN |
| Timeframes | 1, 3, 5, 15, 30, and 60 minutes |
| Streams | 54 |
| Cached validation windows | 6,554 |
| Context | 256 bars |
| Future horizon | 16 bars |
| Per-stream window separation | 272 bars, so complete context+future parents do not overlap |
| Probe folds | Five expanding calendar walk-forward folds |
| Embargo | Maximum cross-timeframe parent duration plus an equally long embargo |
| Regression head | Standardized Ridge, `alpha=1`, `solver=lsqr` |
| Classification head | Standardized Logistic Regression, `C=1` |

Canonical hashes:

- Window artifact: `646b1dbeb0d45d074a1f7bd14f8f751488530630c36b59e47309206387158716`
- Window fingerprint: `fe1abb816fa22665b0250206e1636de788c59a1efd88a38393bdd3e5d5f6e2f4`
- Fold contract: `2348c5c9df335b931130af2c5278b8969754d80fa739858cc037715f68f9ab55`

### Artifact coverage

- 12 trainable encoder arms have vanilla, Stage 1, Stage 2, and Stage 3 representations scored: 48 representations total.
- The trained-stage table contains 36 rows.
- Sundial is correctly blocked rather than assigned a fake score: its native hidden states became non-finite on real OHLCV smoke windows.
- TabPFN-TS is correctly excluded from the staged-encoder table: it is an in-context downstream model, not a persistent Stage 1 → 2 → 3 encoder.
- Mantis V1/V2 Stage 2 and Stage 3 results are diagnostic and non-promotable because the locked Stage 1 lineage failed its canonical gate.

### Headline representation results

| Property | Best relevant result | Important qualification |
|---|---:|---|
| Realized in-window volatility | Chronos V1 S2, R² 0.9298 | Descriptive and stable; fold std 0.0208. |
| Trend efficiency | Chronos Bolt S1, R² 0.8171 | Descriptive; fold std 0.0960. |
| Range expansion | Chronos Bolt S2, R² 0.7877 | Descriptive; fold std 0.1036. |
| In-window direction | MOMENT S1, AUC 0.9932 | Descriptive, not forward evidence. |
| Future terminal absolute move | Moirai-2 S2, R² -0.0766 | Best result is still below the fold-mean baseline and changes sign across folds. |
| Future direction, trained | Chronos Bolt S1, AUC 0.5190 | Small effect with fold std 0.0109; no formal significance analysis. |
| Future direction, any representation | Chronos V2 vanilla, AUC 0.5207 | Fold std 0.0255; its 0.0017 lead over Bolt S1 is not a meaningful ranking. |

The exact 36 trained-stage rows and 12 vanilla rows remain in `FOUNDATION_MODEL_RESULTS.md` and `output/foundation_tournament/representation_apples/representation_results.json`.

## Corrections to the original interpretation

### Correction 1: forward-direction arms are not meaningfully ranked

The original report stated that Chronos V2 vanilla was the best forward-direction representation and that training reduced its score at every stage. The raw means are correct, but the ranking language is too strong.

- Chronos V2 vanilla: 0.5207 AUC, fold std 0.0255.
- Chronos Bolt S1: 0.5190 AUC, fold std 0.0109.
- Difference: 0.0017 AUC.

These folds are expanding walk-forward folds with temporal dependence and overlapping training histories. Therefore `fold_std / sqrt(5)` is, at best, a rough dispersion heuristic—not a valid standard error or significance test. Cross-model predictions are also paired on the same events, so independent-error comparisons are inappropriate.

The honest conclusion is:

> No representation currently has demonstrated a statistically reliable forward-direction advantage under this evaluation.

Chronos V2 Stage 2 deserves separate scrutiny because its paired delta versus its own vanilla baseline is -0.0184 and is negative in four of five folds. That is evidence of possible forgetting, but it is still not a formal paired significance result.

Required correction to future reports: show paired fold deltas and time-clustered confidence intervals; do not shortlist models using differences of a few AUC thousandths.

### Correction 2: terminal absolute return is an inadequate sole forward-magnitude diagnostic

Current `fwd_absmove` is:

```text
abs(log(final future close) - log(final context close))
```

It is one terminal displacement over 16 bars. Trend cancellation and path noise make it a noisy target. Negative R² still has a valid literal meaning—the fitted probe failed to outperform the test fold’s mean—but it does not identify the cause. Possible causes include:

- the representation lacks conditional forward-scale information;
- the linear head is insufficient;
- target calibration shifts across calendar folds;
- a terminal absolute return is too noisy despite predictable volatility clustering;
- pooling timeframes with different economic horizons destroys a conditional relationship.

Add a forward realized-volatility positive control before concluding that forward transfer is absent.

Recommended target:

```text
future_returns = diff(log([last_context_close] + future_closes))
fwd_realized_vol = sqrt(mean(future_returns ** 2))
```

This uses all 16 causal future returns and does not allow up/down cancellation. It remains a label only; it must never enter an input feature set.

Also report:

- Spearman rank IC for `fwd_absmove` and `fwd_realized_vol`;
- AUC for whether forward magnitude is in the top validation-train quantile;
- calibration-sensitive R² separately from ranking metrics;
- an EWMA/past-volatility forecast as a positive-control predictor.

Interpretation rules:

- If the encoder predicts forward realized volatility but not terminal absolute return, it contains conditional-scale information but cannot predict a single terminal displacement.
- If neither the encoder nor a simple past-volatility control predicts forward realized volatility, first audit target construction, time slices, and fold nonstationarity.
- If the classical volatility control succeeds and all encoders fail, representation extraction or probe alignment is suspect.

### Correction 3: pooled timeframes are a material confound, not a footnote

A 16-bar future means:

- 16 minutes on 1-minute bars;
- 48 minutes on 3-minute bars;
- 80 minutes on 5-minute bars;
- 4 hours on 15-minute bars;
- 8 hours on 30-minute bars;
- 16 hours on 60-minute bars.

The current aggregate is also heavily weighted toward short bars:

| Timeframe | Rows | Share |
|---|---:|---:|
| 1 minute | 1,800 | 27.46% |
| 3 minutes | 1,790 | 27.31% |
| 5 minutes | 1,664 | 25.39% |
| 15 minutes | 763 | 11.64% |
| 30 minutes | 366 | 5.58% |
| 60 minutes | 171 | 2.61% |

The 1/3/5-minute streams constitute 80.16% of the rows. The pooled table is therefore mostly a short-timeframe benchmark.

Required reports:

1. Per-timeframe scores fitted and evaluated within each timeframe.
2. Micro average over all rows, retaining the current result for compatibility.
3. Macro average that weights each timeframe equally.
4. Per-symbol/timeframe diagnostics for instability—not 54 separate promotion decisions.
5. A fixed-wall-clock target comparison, such as 60-minute, 4-hour, and 16-hour forward horizons, where data availability allows it.

The cached embeddings are sufficient for timeframe slicing and the new 16-bar target. Fixed-wall-clock targets longer than the cached 16 future bars may require a new sealed label artifact. Do not silently read source data beyond the cached future contract and still claim the old artifact hash.

## Correction to the proposed classical-feature ablation

The suggestion to probe “the six hand-computed target statistics themselves as a six-feature representation” is unsafe as written.

Two of the six values—`fwd_absmove` and `fwd_dir`—are future labels. Supplying either as an input is direct leakage.

Use two causal controls instead.

### Minimal context-statistics control

Computed solely from the 256-bar context:

- realized volatility;
- trend efficiency;
- range expansion;
- net log return;
- sign of net log return;
- log volume level/change using a past-only robust normalization.

This asks whether a tiny set of obvious context summaries matches the frozen encoders.

### Strong causal OHLCV control

Also context-only, with all transforms fitted on past training rows:

- multi-horizon returns;
- realized volatility and downside/upside semivolatility;
- high-low range and ATR-normalized range;
- candle body and wick geometry;
- close location within recent range;
- volume change, volume z-score, and price-volume interaction;
- compression/expansion ratios at multiple causal windows;
- labeler-provided setup geometry available at decision time.

Freeze this feature specification before comparing models. It must use the same folds, embargo, scaler-fitting scope, and downstream events as the encoder arms.

Required downstream arms:

1. Labeler/setup features only.
2. Frozen embedding only.
3. Labeler/setup features plus frozen embedding.
4. Minimal context-statistics control.
5. Strong causal OHLCV control.
6. Strong causal control plus frozen embedding.

The foundation representation adds value only if the combined arm improves consistently over the causal control, not merely over a constant baseline.

## Why downstream testing moves ahead of retraining

The repository explicitly separates market understanding from strategy decisions. A generic unconditional direction probe is not the final use case. The relevant question is conditional:

> Given a causal strategy setup at time `t`, does the frozen representation improve trade/no-trade ranking, calibration, or realized-R selection beyond the setup and classical OHLCV features already known at `t`?

An encoder can fail unconditional direction prediction and still be useful for:

- filtering a setup in hostile volatility regimes;
- distinguishing trend continuation from chop;
- calibrating trade size or target reach;
- identifying compression before expansion;
- conditioning a strategy-specific head without directly forecasting direction.

Therefore the downstream honest-ruler test becomes the promotion gate for representation usefulness. It should no longer be postponed until an unconditional forward-AUC gate passes.

## Downstream experiment specification

### Finalists

Start with four frozen representation candidates and controls:

| Arm | Reason for inclusion |
|---|---|
| Chronos Bolt S1 | Best trained trend representation and highest trained mean `fwd_dir` AUC. |
| Chronos V1 S2 | Strongest and most stable volatility representation. |
| Chronos V2 vanilla | Tests whether pretrained signal is better preserved without futures adaptation. |
| Moirai-2 Small S2 | Best mean `fwd_absmove` R² and clearest forward-magnitude improvement versus its vanilla control. |
| Causal classical features | Required falsification baseline. |
| TabPFN-TS | Downstream in-context head control, not a staged encoder. |

Mantis V2 S1 may be retained as a compatibility/incumbent control because the existing strategy scripts support it, but it is not a promoted lineage.

### Strategies

Use both existing causal event families:

- SuperTrend flip meta-labeling;
- fractal/zigzag pivot meta-labeling.

Do not use the reserved OOS interval for development. Rebuild or configure development folds entirely inside `[2019-07-01, 2025-07-01)` with forward-label purge at every boundary. Any dates already inspected by prior research remain development evidence, not pristine OOS.

### Heads

First representation comparison:

- standardized logistic regression as the low-variance primary head;
- one fixed small MLP as a nonlinear sensitivity check;
- identical head hyperparameter budget for every embedding arm;
- TabPFN-TS as a separately labeled downstream-head control.

Do not mix representation selection and broad head tuning. Determine whether the representation changes results before optimizing complex heads.

### Required metrics

Trading utility at fixed, predeclared selection rules — **primary; this is the deliverable**:

- mean realized R per event and selected trade;
- profit factor;
- win rate and trade count/coverage;
- maximum drawdown in R;
- results after the same cost/slippage assumption;
- target-reach ranking where the labeler exposes multiple R targets.

Prediction quality — diagnostic, used to explain trading deltas, never to overrule them:

- ROC AUC and PR AUC;
- log loss and Brier score;
- calibration slope/intercept or reliability error;
- fold-by-fold paired deltas versus the causal-feature control.

Controls:

- real labels;
- shuffled labels within the legally defined training set;
- random-input or time-destroyed representation control;
- setup-only and causal-feature-only baselines.

### Statistical comparison

Do not treat five fold scores as independent observations.

- Save row-level predictions for every arm on identical test events.
- Compute paired deltas on those events.
- Use calendar block bootstrap intervals, resampling shared weeks or sessions across all synchronized streams so cross-symbol market dependence is preserved.
- Predeclare the block unit and number of bootstrap repetitions.
- Report confidence intervals and the fraction of folds with the same delta sign.
- Treat tiny AUC gaps without stable paired intervals as ties.

### Promotion gate

The primary metric is trading utility: predeclare one headline number (recommended: mean realized R per event at the declared selection rule, after costs) plus profit factor as a co-primary sanity check. AUC, log loss, and calibration are supporting diagnostics only.

A representation advances only if, across both strategies or one explicitly predeclared target strategy:

1. Real labels beat shuffle/random controls.
2. Embedding plus causal features beats causal features alone on the predeclared primary trading metric.
3. The improvement is positive across a declared majority of folds and its paired block-bootstrap interval does not include a material regression.
4. Win rate, trade coverage, drawdown in R, and calibration do not materially regress — a higher mean R bought with collapsed trade count or deeper drawdown is not a pass.
5. Results repeat under at least one additional seed.

No scalar average of unlike metrics controls promotion, and no probe metric can overrule the trading result in either direction.

## Required implementation work

### 1. Preserve experiment source

The worktree currently contains approximately 2,630 changed lines across 30 tracked files plus many untracked tournament scripts, tests, and reports. Do not make one blind bulk commit.

Before new experiments:

1. Review the diff for unrelated/user-owned work.
2. Separate coherent source, tests, and documentation commits.
3. Exclude caches, model weights, and large generated embeddings unless repository policy explicitly tracks them.
4. Record the exact revision, dirty state, dependency environment, corpus fingerprint, and command in every new artifact.

The date in the result artifact is UTC and can legitimately be one calendar day ahead of America/Los_Angeles near midnight. Future reports should label the timezone rather than implying a date error.

### 2. Repair result indexing/documentation drift

`FOUNDATION_MODEL_TODO.md` is stale relative to completed artifacts. For example, it still marks several shared representation probes as pending and Toto’s training adapter as incomplete even though the tournament contains verified Toto Stage 1/2/3 artifacts.

Do not use that TODO as the source of truth until it is reconciled from the canonical result JSON and checkpoint reports.

### 3. Extend the scorer without re-extracting embeddings

Add a versioned scoring schema rather than mutating the existing result JSON in place.

Suggested outputs:

```text
output/foundation_tournament/representation_apples_v2/
  representation_results_v2.json
  representation_results_v2.md
  target_manifest.json
  slice_manifest.json
  predictions/<arm>/<stage>/<target>.npz
```

The v2 result must reference:

- the original window SHA and fingerprint;
- every embedding SHA;
- the exact target formula/version;
- the original fold contract or a new hash if slicing changes fold membership;
- `oos_read=false`;
- row-level prediction hashes for paired comparisons.

For per-timeframe probes, fit scalers and heads inside that timeframe’s training rows and score only its test rows. Do not fit one pooled head and merely partition its predictions unless that is reported as a separate diagnostic.

### 4. Build a cross-family embedding seam

The current SuperTrend and fractal scripts are Mantis-specific:

- `scripts/benchmark_supertrend_mantis.py` accepts only Mantis V1/V2 arms and constructs a Mantis classifier.
- `scripts/benchmark_fractal_mantis.py` calls the Mantis `embed_windows` path directly.

The tournament’s extractor registry already knows how to export multiple model families. Reuse that registry behind a generic interface such as:

```text
embed(arm, checkpoint, contexts, window_contract) -> [N, D]
```

The interface must:

- preserve model-native preprocessing;
- bind caches to model/checkpoint/window hashes;
- reject mismatched context/channel contracts;
- export only deployment-available backbone states;
- produce deterministic batch-parity tests;
- keep task projectors/decoders out of the frozen representation unless explicitly versioned as part of deployment.

Do not duplicate family-specific extraction logic inside each strategy benchmark.

## Decision tree after re-scoring and downstream evaluation

### Case A: causal classical features match every encoder

Conclusion: the tested foundation representations do not add enough conditional information for these strategies.

Action:

- stop broad foundation-model sweeps;
- keep the cheapest classical system as incumbent;
- investigate different data modalities, event definitions, or objectives only with a concrete downstream hypothesis.

### Case B: embeddings add value, but stage choice barely matters

Conclusion: pretrained representations help, while futures adaptation is not the bottleneck.

Action:

- deploy or continue validating the simplest vanilla/frozen finalist;
- do not spend compute on Stage 2/3 redesign;
- focus on head calibration, event labeling, and execution robustness.

### Case C: one adapted stage consistently beats vanilla and classical controls

Conclusion: futures adaptation can add useful conditional information.

Action:

- repeat with a second seed;
- test the already implemented `elapsed_time_v2` Stage 2 against its legacy objective;
- redesign Stage 3 only if Stage 3 itself adds downstream value.

### Case D: forward realized volatility is predictable, but no strategy improves

Conclusion: the representation contains conditional scale information that the current labelers or heads do not use effectively.

Action:

- test volatility-conditioned sizing/abstention rather than direction prediction;
- audit whether setup geometry dominates the market-state representation;
- do not infer that more pretraining will fix the strategy.

### Case E: neither embeddings nor classical controls predict forward realized volatility

Conclusion: measurement, horizon definition, regime shift, or event construction may be the bottleneck.

Action:

- audit targets and folds;
- add fixed-wall-clock horizons;
- increase the short-timeframe sample cap for finalists only if power analysis justifies new extraction;
- do not launch a full-family retraining sweep.

## What remains valid from the original redesign proposal

If downstream evidence eventually justifies new representation training, retain these requirements:

- elapsed-time rather than raw bar-offset sampling;
- nontrivial positive separation and reduced overlap shortcuts;
- independent augmentations per observation;
- synchronized cross-stream exclusion where appropriate;
- vanilla-feature anchoring or distillation to limit forgetting;
- Stage 3 return-distribution, realized-volatility, absolute-move, and direction auxiliaries;
- per-target non-inferiority gates;
- multiple seeds and time-destroyed controls;
- exact training-state resume and deployment parity.

Part of this work already exists in `futures_foundation/finetune/pretext/_torch/contrastive.py` as `elapsed_time_v2`. Do not claim that the completed cross-family tournament used this revised objective unless each checkpoint report proves it; existing code after the run is not evidence about historical checkpoint construction.

## Explicit non-goals for the next experiment

- Do not repair Sundial hidden states before higher-value comparisons are complete.
- Do not launch broad Optuna studies.
- Do not add more foundation-model families.
- Do not read the reserved OOS interval.
- Do not tune thresholds separately for each model on evaluation folds.
- Do not use forward targets as features.
- Do not rank models by native training loss across families.
- Do not treat tiny aggregate AUC differences as evidence.

## Verification checklist for the next agent

Before reporting v2 results, verify all of the following:

- [ ] Original window SHA and fingerprint match.
- [ ] Original embedding hashes match.
- [ ] New target formulas have unit tests with synthetic trend/chop/volatility paths.
- [ ] No target-derived future value enters any feature matrix.
- [ ] Every scaler, quantile threshold, and feature transform is fit on training rows only.
- [ ] Per-timeframe folds remain chronological and embargoed.
- [ ] Row-level predictions are saved for paired analysis.
- [ ] Confidence intervals use calendar blocks, not iid row bootstrap.
- [ ] Cross-family embeddings pass deterministic batch-parity tests.
- [ ] Strategy labels are purged by label-end time at every boundary.
- [ ] REAL, SHUFFLE, and RANDOM/time-destroyed controls use identical event rows.
- [ ] `oos_read=false` is asserted in every development artifact.
- [ ] Full tests and `git diff --check` pass.

## Final recommendation

The immediate work should be a versioned, GPU-free representation re-score followed by a cross-family, frozen-embedding downstream honest-ruler comparison. That directly tests the repository’s intended architecture and preserves the current checkpoints as controls.

The deliverable of the downstream comparison is a trading scorecard per arm — realized R distribution, win rate, profit factor, coverage, and drawdown after costs, with paired intervals versus the causal-feature control. That scorecard, not any probe table, decides which of Cases A–E applies and whether any retraining is funded.

Only retrain Stage 2 or Stage 3 after the downstream experiment identifies representation quality—not target definition, timeframe pooling, event labeling, head calibration, or classical features—as the limiting factor.
