# Foundation Model Native-Contract Tasks

This checklist is governed by
[FOUNDATION_MODEL_NATIVE_CONTRACT_PLAN.md](FOUNDATION_MODEL_NATIVE_CONTRACT_PLAN.md). No training or
new ranking is authorized until the relevant admission items pass.

## Phase 0 — historical evidence and registry

- [x] Snapshot and hash the current historical tournament index in
  [`config/foundation_models/historical_native_contract_snapshot.json`](config/foundation_models/historical_native_contract_snapshot.json).
- [x] Add `native_valid`, `configuration_specific`, `invalid_contract`, `blocked` and
  `research_only` statuses without modifying old predictions or scores.
- [x] Quarantine Kronos Mini wrong-tokenizer arms, TTM padded representations and Toto trained
  stages; record Sundial hidden-state extraction as a failed attempt without implying valid staged
  artifacts exist.
- [x] Relabel noncanonical Mantis, custom MOMENT pooling and all custom forecast-model hidden states.
- [ ] Add a warning to every active historical ranking document.
- [x] Replace fragmented rosters with one per-track capability registry at
  [`config/foundation_models/native_contracts.json`](config/foundation_models/native_contracts.json).
- [ ] Record source patches, dependency locks, model/weight/tokenizer revisions, preprocessing and
  split/OOS status.

## Phase 1 — dossiers and independent approval

### MantisV1

- [x] Pin source and weight revision/hash.
- [x] Document raw univariate input, fixed channel order and interpolation contract.
- [x] Prove final-CLS compatibility with the pinned official implementation.
- [x] Declare official contrastive, frozen representation and supervised-barrier roles separately.
- [ ] Obtain independent dossier approval.

### MantisV2

- [x] Pin source and weight revision/hash.
- [x] Prove raw-input, fixed-512 and channel-folding parity for the admitted surface.
- [ ] Test shorter-input interpolation as a separate runtime surface.
- [x] Prove layer-index-2 combined CLS+mean extraction and output dimension.
- [x] Retain final CLS only as a named compatibility sensitivity.
- [ ] Obtain independent dossier approval.

### MOMENT Small

- [x] Pin source, weight, package and dependency revisions.
- [x] Prove raw left-padding, input-mask and internal RevIN parity.
- [x] Prove official embedding-mean contract.
- [ ] Prove an official classification-head contract after task-specific fine-tuning.
- [x] Separate masked reconstruction, classification and forecast task branches.
- [ ] Obtain independent dossier approval.

### Kronos Mini

- [x] Mark all Tokenizer-base historical arms invalid.
- [x] Pin Mini with `Kronos-Tokenizer-2k` fail-closed.
- [x] Document OHLCVA amount fallback, stamps, timezone, normalization and clip ±5.
- [x] Reproduce the official greedy forecast surface and inverse normalization across
  1/3/5/15/30/60-minute UTC timestamps.
- [ ] Admit stochastic sampling separately before describing the executable surface as probabilistic.
- [x] Label hidden states custom.
- [ ] Obtain independent dossier approval.

### Kronos Small

- [x] Pin Small with `Kronos-Tokenizer-base` fail-closed.
- [x] Document OHLCVA amount fallback, stamps, timezone, normalization and clip ±5.
- [x] Reproduce the official greedy forecast surface and inverse normalization across
  1/3/5/15/30/60-minute UTC timestamps.
- [ ] Admit stochastic sampling separately before describing the executable surface as probabilistic.
- [x] Label hidden states custom.
- [ ] Obtain independent dossier approval.

### Chronos V1

- [x] Pin model/package/source revisions and resolve the stale embedding-API registry claim.
- [x] Reproduce official sampled forecast distribution.
- [x] Reproduce the concrete `ChronosPipeline` public `embed()` tokens and tokenizer state without pooling.
- [x] Document univariate independent-channel semantics and native adaptation support.
- [ ] Obtain independent dossier approval.

### Chronos Bolt

- [x] Pin model/package/source revisions.
- [x] Reproduce official quantile forecast and native scaling behavior.
- [x] Reproduce the concrete `ChronosBoltPipeline` public `embed()` tokens and location/scale state without pooling.
- [x] Document the separate, still-blocked native tuning surface.
- [ ] Obtain independent dossier approval.

### Chronos-2

- [x] Pin model/package/source revisions.
- [x] Reproduce joint multivariate forecasts, grouping and covariate behavior.
- [x] Reproduce official `embed()` tokens and scaling state without custom pooling.
- [ ] Reproduce official `fit()` full/LoRA baseline before custom tuning.
- [ ] Obtain independent dossier approval.

### TimesFM 2.5

- [x] Pin model/package/source revisions.
- [x] Prove Transformers-wrapper parity with the official wrapper.
- [x] Preserve native point/raw-quantile outputs and the frequency-agnostic raw-series
  normalization contract.
- [ ] Verify official LoRA adaptation; hidden states remain custom.
- [ ] Obtain independent dossier approval.

### TTM-R2

- [x] Write a regression test for artificial padding and checkpoint selection.
- [x] Select and assert the context/horizon/frequency checkpoint through the official API.
- [x] Remove trailing-zero padding from executable TTM paths.
- [x] Prove scaler, all 1/3/5/15/30/60-minute frequency-prefix tokens
  (`1/0/3/5/6/7`; 3-minute is OOV), and channel-mixer initialization behavior.
- [x] Keep representation and adaptation blocked until forecast parity passes.
- [ ] Obtain independent dossier approval.

### Moirai-2 Small

- [x] Pin model/package/source revisions and record CC-BY-NC research-only restriction.
- [x] Reproduce official packed multivariate quantile forecasts with masked-value sanitization.
- [ ] Verify custom token indexing/loss before any tuning.
- [x] Label hidden states custom.
- [ ] Obtain independent dossier approval.

### Toto-2 22M

- [x] Pin model/package/source revisions.
- [x] Reproduce joint masks, group IDs, scaling and quantile outputs.
- [x] Set `decode_block_size=None` for the short-horizon primary contract.
- [x] Mark historical trained stages invalid; retain zero-shot forecast role only.
- [ ] Obtain independent dossier approval.

### Sundial Base

- [x] Build an isolated pinned compatible inference environment.
- [x] Reproduce finite official samples and inverse normalization.
- [x] Keep hidden states and tuning excluded in the executable registry and legacy gates.
- [ ] Obtain independent dossier approval.

### TabPFN-TS

- [ ] Pin package/model/license revisions.
- [x] Prove support sets remain inside each training fold.
- [x] Define in-context forecast and downstream-control roles only.
- [ ] Obtain independent dossier approval.

## Phase 2 — shared contract harness

- [x] Official-example parity fixture for every technically admitted family.
- [x] Fail-closed measured invariants for scaling/raw-output parity, or an explicit
  not-applicable disposition. This proves adapter/public-output parity, not forecast quality.
- [x] Numerically gated padding/mask/missing-value tests or an explicit finite-input contract.
- [x] Frequency/timezone/covariate tests where the native interface exposes them.
- [ ] Complete joint-versus-independent channel perturbation tests for every applicable family;
  current shape/group assertions do not prove perturbation isolation.
- [x] Single/batch/repeated/partitioned batch parity harness.
- [x] FP32 finite reference for every technically admitted track.
- [ ] Reduced-precision tolerance; current admitted runtimes are FP32-only.
- [x] Exact admitted 512-context/16-horizon execution-surface assertion.
- [ ] Add lower/upper boundary rejection and sensitivity tests where the upstream API exposes
  variable context or horizon.
- [x] Structural no-future/no-label input-surface assertion for the synthetic parity worker.
- [ ] Add numerical prefix-invariance tests where an API accepts a larger composite input.
- [ ] Gradient/freeze-surface tests.
- [ ] Native repeated-batch loss-decrease smoke.
- [ ] Exact resume, save/reload and deployment/export parity.
- [x] Evidence-bound machine-readable v2 admission report with route, exact runtime/environment,
  optional artifact hashes and two independent approvals.
- [x] Enforce a non-bypassable two-approval floor, normalized reviewer identity comparison, and
  valid approval/report timestamp ordering.
- [ ] Authenticate reviewer identity and organizational independence through external governance
  or a trusted-signature mechanism before any operational authorization.

## Phase 3 — data and label contract

- [ ] Complete governed Corpus v3 coverage/yield audit before materialization.
- [ ] Freeze common-information and native-best views separately.
- [ ] Seal regular bars, contract IDs, source rows and timestamps.
- [ ] Seal native future-series forecast targets separately from trading labels.
- [ ] Build post-entry ordered tick barrier, MFE, MAE and time-to-barrier labels.
- [ ] Store exact/ambiguous coverage and deterministic conservative fallback.
- [ ] Validate wall-clock cadence, session gaps, rolls and complete label-end timestamps.
- [ ] Purge/embargo by label end, not fixed bar count.
- [ ] Version tick size/value, fees and zero/one-tick scoring contracts.

## Phase 4 — frozen baselines

- [ ] Save native quantiles/samples without premature point reduction.
- [ ] Score MAE, RMSE, MASE, WQL/pinball, coverage, width, crossing and CRPS where valid.
- [ ] Extract only admitted official representations into the native table.
- [ ] Store hidden states and forecast summaries in separate custom-transfer tables.
- [ ] Run deterministic, shuffled and time-destroyed controls.

## Phase 5 — identical downstream ruler

- [ ] Causal-only baseline.
- [ ] Official-embedding-only baseline.
- [ ] Native-forecast-feature-only baseline.
- [ ] Causal + embedding.
- [ ] Causal + forecast features.
- [ ] Residual-over-causal primary incremental-information test.
- [ ] Low-variance linear ruler plus capacity-controlled nonlinear sensitivity.
- [ ] Nested chronological preprocessing, calibration and threshold selection.
- [ ] Identical concurrency, fees, zero-tick primary and frozen one-tick sensitivity.
- [ ] Paired calendar/root blocks, stream macro, per-timeframe and effective-sample-size reports.

## Phase 6 — family-native adaptation

- [ ] Fail closed unless the family passed Phase 2, Phase 3 is sealed, and its Phase 4/5 frozen
  baseline and identical downstream ruler are complete.
- [ ] Match unique parent-window exposure and record token/patch/compute budgets.
- [ ] Run every branch directly from pinned vanilla.
- [ ] Use only supported native objectives and trainable surfaces.
- [ ] Compare every adapted arm with its own frozen/native parent.
- [ ] Do not promote from descriptive probe gains.
- [ ] Require three seeds for any promotable result.

## Phase 7 — supervised barrier and custom SSL

- [ ] Run frozen → partial → full barrier tuning only after frozen incremental value is measured.
- [ ] Keep identical barrier outputs, head family/search budget and train-fold-fitted bottleneck
  across compatible families; record actual parameter counts.
- [ ] Fund at most one architecture-specific custom SSL hypothesis at a time.
- [ ] Require paired forward/path and economic lift over vanilla and causal controls.
- [ ] Allow chaining only after a direct-parent ablation proves incremental value.

## Phase 8 — selection and prospective confirmation

- [ ] Predeclare primary contract and family/role selection rule.
- [ ] Select inside nested development with hierarchical multiplicity correction.
- [ ] Freeze one representative per family/role with complete deployment bundle.
- [ ] Treat all 2024–2026 results as spent for selection.
- [ ] Confirm once on prospectively arriving data after the complete freeze.
