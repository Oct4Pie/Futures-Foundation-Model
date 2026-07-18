# Futures Trading Foundation-Model Task List

This checklist executes [FUTURES_TRADING_FOUNDATION_PLAN.md](FUTURES_TRADING_FOUNDATION_PLAN.md).
Items are dependency ordered. A checked implementation item is not evidence of success unless its
verification item is also checked.

## Gate 0 — freeze and reconcile

- [x] Write the canonical execution plan and define its authority over stale execution/OOS wording.
- [x] Preserve historical tournament documents and checkpoints as evidence.
- [x] Commit and publish the exact source patch and dependency environment.
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
- [x] Official representation extraction passes deterministic batch-parity tests for the admitted
  Mantis, MOMENT and Chronos tracks. Sundial forecasting is separately native-valid; Sundial hidden
  states remain explicitly excluded because they are non-finite.
- [x] Causal-baseline and representation-arm row-level predictions and fold assignments are saved.
- [x] Causal-baseline paired uncertainty uses calendar blocks, not iid rows; finalist comparisons
  will use the larger predeclared resample count.

## Gate 4 — conditional trading honest ruler

- [x] Score trend continuation/termination within active-trend contexts.
- [x] Score trend birth after compression and tagged trigger subsets.
- [x] Replace the bounded screen's raw zero threshold with fully nested, earlier-fold-only monotonic
  calibration and a predeclared coverage/stability rule.
- [x] Apply identical cost, fill, ambiguity and concurrency rules to every arm.
- [x] Default to one active position per symbol/strategy stream.
- [x] Report realized R, WR, PF, coverage, drawdown, breadth and calendar stability.
- [x] Report paired lift over causal features and the same family's vanilla backbone for the bounded
  screen.
- [x] Build the full-history 2019-2024 downstream fitting artifact for the predeclared finalists.
  The fixed outer development interval is 2024-07 to 2025-07; see
  [DOWNSTREAM_FULL_HISTORY_RESULTS.md](DOWNSTREAM_FULL_HISTORY_RESULTS.md).
- [x] Test residual embedding fusion and decomposed barrier outcomes under the corrected ruler.
  Both are head-sensitive but fail to add stable paired utility over causal features.
- [x] Record the Case A-E diagnosis and training-funding decision. Current decision: Case A for the
  ATR lane, Case C for inconsistent/damaging adaptation and Case D for the fractal pool; Case E is
  rejected, so broad SSL funding remains blocked.

## Gate 4B — conditional trend-event pools

- [x] Implement versioned, prefix-invariant pullback-continuation and compression-breakout
  detectors with causal structural stops.
- [x] Add event-stratified sampling that retains rare candidates, caps common candidates
  chronologically and equalizes stream weight.
- [x] Materialize the 2019-07 to 2025-07 development-only collection across all 54 streams.
- [x] Run raw and causal-only structural/ATR-stop rulers with nested calibration, fees, zero primary
  slippage, next-open entry and adverse-first ambiguity.
- [x] Admit only viable structural pools to the vanilla/VICReg MantisV2 comparison.
- [x] Run concatenated, residual and barrier-decomposed head comparisons on identical rows.
- [x] Record canonical definitions, hashes, commands/artifacts, results and frozen decisions in
  [CONDITIONAL_EVENT_GATE_RESULTS.md](CONDITIONAL_EVENT_GATE_RESULTS.md).
- [x] Reject both ATR-stop lanes, residual fusion and VICReg promotion for this task.
- [x] Freeze raw pullback, causal barrier pullback, vanilla direct-head pullback and VICReg negative
  control as the only legacy-confirmation arms.
- [x] Run all nine symbols and all six timeframes on the predeclared common-timestamp interval
  `[2025-07-01, 2026-04-14)`. SSL, heads, PCA, calibration and thresholds remained pre-OOS-only.
  The causal selector produced `+0.1253R/trade`, PF `1.185` at zero tick and `+0.0646R/trade`, PF
  `1.090` under frozen one-tick repricing. Neither vanilla Mantis family added resolved paired
  utility over causal features, and both VICReg adaptations degraded their vanilla point estimate.
  See [LEGACY_OOS_CONFIRMATION_RESULTS.md](LEGACY_OOS_CONFIRMATION_RESULTS.md).

## Gate 5 — bounded MantisV2 revised-stage research exception

- [x] Numerical Stage 0 adapter/export/context parity passed at 64/128/256 bars; ONNX maximum
  absolute error is `5.72e-6`. CUDA batch-shape sensitivity is recorded separately.
- [ ] **NOT FUNDED:** MantisV2 Stage 1 structural-reconstruction pilot; direct Stage 2 failed the
  downstream economic gate.
- [ ] **BLOCKED:** new Chronos Bolt training; retain its frozen representation as a control only.
- [x] Direct MantisV2 Stage 2-from-vanilla ablation completed at 256 bars on the locked
  2019-07/2024-07/2025-07 calendar protocol.
- [x] Train and evaluate the two-seed `elapsed_time_v2` correction against `bar_offset_v1`; it
  improved mean core behavior but failed the locked promotion rule and remains a baseline.
- [x] Implemented and compared `vicreg_v1` against `elapsed_time_v2`, with the same preprocessing,
  anchor universe, seed and optimizer budget. VICReg passed the representation/control gate.
- [ ] **NOT FUNDED:** minimal Stage 3 core: Stage 2 improved vanilla but did not add stable
  forward/economic utility over causal features.
- [ ] **BLOCKED:** add one Stage 3 objective at a time only after the core is measured.
- [ ] **BLOCKED:** second trainable architecture; this exception is MantisV2-only.

### Gate 5 verification

- [x] Full training-state resume matches uninterrupted training on the verified exact-resume test.
- [x] Audit the historical next-leg objective. Its one-cap reserve and concatenated-stream target
  construction were unsafe; the code now reserves both legs and segments targets by stream and
  contract. Historical checkpoints remain non-promotable under the corrected contract.
- [x] Artifact hashes and direct-vanilla lineage are recorded; this branch has no parent checkpoint.
- [x] Per-target and downstream paired gates were evaluated. Representation non-inferiority passed,
  but paired forward/economic lift over causal features failed.
- [ ] **NOT FUNDED:** second seed; no checkpoint passed the downstream funding gate.

## Gate 5B — bounded MantisV1/V2 objective and transfer branch

- [x] Extract and evaluate vanilla MantisV1 on the identical 147,309-row conditional-event
  contract. It leads the pullback point estimates but its paired lift over causal crosses zero.
- [x] Train direct MantisV1 `vicreg_v1` with the exact V2 data, context, seed, budget, anchor and
  shuffle-control contract. It passed the generic representation/control gate.
- [x] Score V1 VICReg against V1 vanilla on the same full-history pullback ruler. It failed paired
  promotion: `-0.00098` R/candidate, one of five folds positive, and damaged compression.
- [x] Build a hash-bound frozen V1+V2 late-fusion arm and compare it with each single backbone.
  Fusion was negative (`-0.0130R`, PF 0.982) and beat V1 in only one of five folds.
- [x] Reject cross-version feature/teacher distillation under the current evidence and retain
  specialized frozen backbones. Prediction ensembling requires a separate declared hypothesis.
- [x] Implement revised structural Stage 1 at 256 bars with shared-timestamp spans, structural
  targets and entry-backbone anchoring as the separate `structure_mask` pretext.
- [x] Implement minimal fixed-wall-clock `path` Stage 3 with forward volatility, MFE/MAE quantiles
  and continuation/termination/reversal; candle MSE is not reused.
- [x] Preserve all 54 streams through per-timeframe future filtering; reject uniform 360-bar
  reservation because it removed CL@60-minute validation coverage.
- [x] Verify both objectives on V1/V2 CUDA smokes, the 90-test torch SSL suite and the full suite;
  after the upstream-main merge it reports 916 passed and 90 skipped. See
  [REVISED_MANTIS_SSL_OBJECTIVES.md](REVISED_MANTIS_SSL_OBJECTIVES.md).
- [x] Run the single predeclared MantisV1 direct-`path` pilot from the merged source revision.
  `structure_mask`, V2 path and all chained lineages remain unfunded.
- [x] Apply the locked paired promotion gate. The checkpoint failed: pullback lift was `-0.01338`
  R/candidate with interval `[-0.03002,+0.00374]` and two of five positive folds; compression lift
  was `+0.00105` with interval `[-0.00481,+0.00698]` and one of five positive folds. It was not
  exposed to OOS. See [REVISED_MANTIS_SSL_OBJECTIVES.md](REVISED_MANTIS_SSL_OBJECTIVES.md).
- [x] Deny a second seed because the branch failed to beat its vanilla backbone and failed the
  one-tick sensitivity (`-0.0802R`, PF `0.895` on pullbacks).
- [ ] **NOT FUNDED:** further path-objective diagnosis or adaptation. The confirmation read found
  no resolved vanilla-Mantis lift over the causal selector and the adapted Stage 2 controls were
  worse. Reopen only for a genuinely new, predeclared representation hypothesis and fresh ruler.

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

- [x] Freeze finalist checkpoints, bundles, labels, folds, thresholds and execution policy.
- [x] Read the predeclared legacy common-timestamp interval `[2025-07-01, 2026-04-14)` once for
  qualified confirmatory evidence; the original full-year all-stream endpoint was unavailable in
  the canonical source.
- [x] Report its known prior-inspection and sparse-ZB-short-timeframe caveats prominently.
- [ ] Run prospective confirmation on data arriving after the full plan/finalist freeze.
- [ ] Require monitoring for drift, calibration, costs, capacity and contract-roll behavior before
  production use.

## Gate 8 — Corpus v3 scale program

- [x] Audit the local data lake instead of inferring availability from the narrow training corpus.
  The current evidence is 45.0M one-minute rows in the nine-root FFM corpus and 15.59B admitted
  tick rows across 43 roots before 2026. General depth and Databento use remains blocked. See
  [DATA_SCALE_AUDIT.md](DATA_SCALE_AUDIT.md).
- [x] Complete real-checkpoint native forecast/representation parity for the locally verifiable
  families in [FOUNDATION_MODEL_NATIVE_CONTRACT_TASKS.md](FOUNDATION_MODEL_NATIVE_CONTRACT_TASKS.md).
  This unblocks governed Corpus v3 design/materialization, not model training. Training remains
  prohibited until the family-specific training surface and Phase 3/4/5 gates pass.
- [ ] Pin a Corpus v3 development contract to the admitted Sierra tick source, registry,
  admission artifact, loader, instrument economics and lake hash-of-hashes.
- [ ] Produce a root-by-year liquidity, continuity, roll and gap matrix; select roots without
  reading strategy outcomes.
- [ ] Materialize a representative tick-derived multi-resolution subset with exact ordered barrier,
  MFE, MAE and time-to-barrier labels.
- [ ] Measure event yield, barrier balance, effective calendar-block sample size, cross-root
  dependence, purge loss and build throughput before claiming a 20–50× scale increase.
- [ ] Run causal-only, frozen-Mantis, causal-plus-frozen-Mantis and end-to-end-Mantis arms with the
  identical barrier head and identical rows. This supervised pilot comes before scaled SSL.
- [ ] Fund one scaled SSL branch only if the measured corpus passes the scale gate and the
  identical-head experiment shows representation learning is a plausible binding constraint.
- [ ] Keep the read legacy OOS interval out of all Corpus v3 training, validation, calibration and
  finalist selection. Require new unused prospective data for confirmation.
