# Futures Trading Foundation-Model Task List

This checklist executes [FUTURES_TRADING_FOUNDATION_PLAN.md](FUTURES_TRADING_FOUNDATION_PLAN.md).
Items are dependency ordered. A checked implementation item is not evidence of success unless its
verification item is also checked.

## Gate 0 — freeze and reconcile

- [x] Write the canonical execution plan and define its authority over stale execution/OOS wording.
- [x] Preserve historical tournament documents and checkpoints as evidence.
- [ ] Commit and publish the exact source patch and dependency environment.
- [x] Add the canonical-plan link to README and mark the older tournament TODO as historical.
- [x] Inventory hashes for the sealed windows, current embeddings and all checkpoints used by the
  frozen downstream comparison. Verified manifest:
  `output/foundation_tournament/frozen_downstream_inventory.json` (48 embeddings, 36 checkpoints,
  one sealed window artifact; 85/85 files passed).
- [x] Correct every active document that calls 2025-07 to 2026-07 pristine OOS.

## Gate 1 — dense causal target contract

- [x] Implement fixed-wall-clock path targets for 1/3/6-hour horizons.
- [x] Emit terminal return, forward realized volatility, MFE, MAE and forward trend efficiency.
- [x] Emit continuation/termination relative to a causal context direction.
- [x] Emit long/short target-before-adverse state with explicit `ambiguous` and `neither` states.
- [x] Emit bar-resolution time-to-favorable/adverse and complete label-end timestamps.
- [x] Normalize price-path magnitudes using a causal, decision-time ATR/R scale.
- [x] Reject/mask horizons crossing a contract roll, missing endpoint or excessive cadence gap.
- [x] Apply adverse-first only to executable policy outcomes; retain ambiguity in representation
  labels.
- [x] Add deterministic schema/version and content fingerprinting.

### Gate 1 verification

- [x] Synthetic monotone-up, monotone-down, chop and volatility-expansion tests.
- [x] Same-bar favorable/adverse ambiguity test.
- [x] Conservative adverse-first executable-R test.
- [x] Prefix-invariance test: appending later bars cannot change already complete labels.
- [x] Decision-prefix perturbation test: future perturbations cannot alter context features/scales.
- [x] Contract-roll crossing is masked.
- [x] Maintenance/session gap and missing endpoint are masked, not truncated.
- [x] 1/3/5/15/30/60-minute streams describe identical elapsed horizons.
- [x] Performance benchmark on representative 1-minute data before full materialization: 100,000
  rows, three horizons and both barrier directions at about 269k rows/s, 34.7 MiB label arrays and
  234 MiB peak RSS on the development host.

## Gate 2 — context/event materialization

- [x] Deduplicate one row per context and attach multi-hot ATR-zigzag, k=2-fractal and SuperTrend
  tags.
- [x] Store fractal-zigzag, HTF agreement, symbol, timeframe, session and causal roll-state metadata
  without duplicating rows.
- [x] Attach ATR-stop and structural-stop policy outcomes to the same context.
- [x] Compute causal classical features through decision time only.
- [x] Store block-aware weights for overlapping dense contexts.
- [x] Implement the Gate 3 scheduler that balances symbol/timeframe shards during fitting.
- [x] Store context start, decision time, label end, contract segment and source row identifiers.

### Gate 2 verification

- [x] Exact event/context deduplication tests.
- [x] Event confirmation-time causality and prefix-invariance tests.
- [x] No context or outcome crosses a contract roll.
- [x] No future-derived value appears in classical features or preprocessing.
- [x] Artifact hashes, counts and per-stream coverage report are reproducible. Canonical collection:
  `output/foundation_tournament/event_contexts_v1/MANIFEST.json`; 54/54 streams, 2,061,548 contexts,
  1,140,467 tagged contexts, 1,348,780 policy events and 481,948,746 compressed bytes.

## Gate 3 — frozen representation re-score

- [x] Reuse sealed embeddings where rows match; extract from frozen checkpoints only where needed.
- [x] Evaluate causal-feature logistic and constrained-XGBoost controls.
- [x] Evaluate vanilla, Stage 1, Stage 2 and Stage 3 embeddings for every available encoder family
  in the sealed 48-representation screen.
- [x] Evaluate embedding-only and embedding-plus-causal-feature arms on the predeclared downstream
  finalist set. The expanded path-target re-score is intentionally not another 48-arm sweep.
- [x] Run shuffle, random-feature and time-destroyed controls on identical causal-baseline rows.
- [x] Add forward realized volatility, forward trend efficiency and continuation/termination metrics.
- [x] Add direction-relative MFE/MAE, barrier-state, reversal, time-to-barrier and policy-R targets
  to the cached-embedding scorer.
- [x] Report pooled and per-timeframe results with per-slice fold contracts.
- [x] Treat 30/60-minute slices as diagnostic unless sample power is increased under a predeclared
  protocol.

### Gate 3 verification

- [x] Every causal-baseline scaler and feature transform is fit on earlier rows only; repeat this
  verification for representation arms.
- [x] Split purge uses label end plus the declared embargo.
- [x] Cross-family embedding extraction passes deterministic batch-parity tests for admitted arms;
  Sundial remains explicitly blocked by non-finite hidden states.
- [x] Causal-baseline and representation-arm row-level predictions and fold assignments are saved.
- [x] Causal-baseline paired uncertainty uses calendar blocks, not iid rows; finalist comparisons
  will use the larger predeclared resample count.

## Gate 4 — conditional trading honest ruler

- [x] Score trend continuation/termination within active-trend contexts.
- [x] Score trend birth after compression and tagged trigger subsets.
- [ ] Replace the bounded screen's raw zero threshold with fully nested, earlier-fold-only monotonic
  calibration and a predeclared coverage/stability rule.
- [x] Apply identical cost, fill, ambiguity and concurrency rules to every arm.
- [x] Default to one active position per symbol/strategy stream.
- [x] Report realized R, WR, PF, coverage, drawdown, breadth and calendar stability.
- [x] Report paired lift over causal features and the same family's vanilla backbone for the bounded
  screen.
- [ ] Build the full-history 2019-2024 downstream fitting artifact for the predeclared finalists.
- [ ] Test residual embedding fusion and decomposed barrier outcomes under the corrected ruler.
- [ ] Record the Case A-E diagnosis and training-funding decision.

## Gate 5 — bounded revised-stage pilots, blocked until Case E

- [ ] **BLOCKED:** numerical Stage 0 adapter/export/context parity.
- [ ] **BLOCKED:** MantisV2 Stage 1 implementation/debug pilot.
- [ ] **BLOCKED:** Chronos Bolt Stage 1 decision pilot.
- [ ] **BLOCKED:** direct Stage 2-from-vanilla ablation.
- [x] Train and evaluate the two-seed `elapsed_time_v2` correction against `bar_offset_v1`; it
  improved mean core behavior but failed the locked promotion rule and remains a baseline.
- [ ] **BLOCKED:** implement and compare one negative-free Stage 2 objective such as VICReg.
- [ ] **BLOCKED:** minimal Stage 3 core: MFE/MAE quantiles + forward vol + continuation + anchor.
- [ ] **BLOCKED:** add one Stage 3 objective at a time only after the core is measured.
- [ ] **BLOCKED:** second architecture if MantisV2 and Chronos Bolt disagree.

### Gate 5 verification

- [x] Full training-state resume matches uninterrupted training on the verified exact-resume test.
- [ ] Parent hashes and backbone changes are verified at every transition.
- [ ] Per-target non-inferiority and downstream paired gates pass.
- [ ] At least two seeds pass for any promoted pilot.

## Gate 6 — full cross-family training, blocked until pilots pass

- [ ] **BLOCKED:** Kronos-mini.
- [ ] **BLOCKED:** Kronos-small.
- [ ] **BLOCKED:** MOMENT-small.
- [ ] **BLOCKED:** MantisV1.
- [ ] **BLOCKED:** MantisV2.
- [ ] **BLOCKED:** Chronos V1.
- [ ] **BLOCKED:** Chronos Bolt.
- [ ] **BLOCKED:** Chronos V2.
- [ ] **BLOCKED:** TimesFM 2.5.
- [ ] **BLOCKED:** TTM-R2.
- [ ] **BLOCKED:** Moirai-2 small.
- [ ] Toto only after a verified native training adapter exists.
- [ ] Sundial excluded from representation training while hidden states remain non-finite.
- [ ] TabPFN-TS retained as a downstream head/control, not forced into an artificial Stage 1-3 chain.

## Gate 7 — confirmation and deployment

- [ ] Freeze finalist checkpoints, bundles, labels, folds, thresholds and execution policy.
- [ ] Read the legacy 2025-07 to 2026-07 interval once for qualified confirmatory evidence.
- [ ] Report its known prior-inspection caveat prominently.
- [ ] Run prospective confirmation on data arriving after the full plan/finalist freeze.
- [ ] Require monitoring for drift, calibration, costs, capacity and contract-roll behavior before
  production use.
