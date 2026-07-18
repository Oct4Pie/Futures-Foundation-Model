# Foundation Model Native-Contract Tasks

This checklist is governed by
[FOUNDATION_MODEL_NATIVE_CONTRACT_PLAN.md](FOUNDATION_MODEL_NATIVE_CONTRACT_PLAN.md). No training or
new ranking is authorized until the relevant admission items pass.

## Phase 0 — historical evidence and registry

- [ ] Snapshot and hash the current historical tournament index.
- [ ] Add `native_valid`, `configuration_specific`, `invalid_contract`, `blocked` and
  `research_only` statuses without modifying old predictions or scores.
- [ ] Quarantine Kronos Mini wrong-tokenizer arms, TTM padded representations and Toto trained
  stages; record Sundial hidden-state extraction as a failed attempt without implying valid staged
  artifacts exist.
- [ ] Relabel noncanonical Mantis, custom MOMENT pooling and all custom forecast-model hidden states.
- [ ] Add a warning to every active historical ranking document.
- [ ] Replace fragmented rosters with one per-track capability registry.
- [ ] Record source patches, dependency locks, model/weight/tokenizer revisions, preprocessing and
  split/OOS status.

## Phase 1 — dossiers and independent approval

### MantisV1

- [ ] Pin source and weight revision/hash.
- [ ] Document raw univariate input, fixed channel order and interpolation contract.
- [ ] Prove final-CLS compatibility with the pinned official implementation.
- [ ] Declare official contrastive, frozen representation and supervised-barrier roles separately.
- [ ] Obtain independent dossier approval.

### MantisV2

- [ ] Pin source and weight revision/hash.
- [ ] Prove raw-input, native-length/interpolation and channel-folding parity.
- [ ] Prove layer-index-2 combined CLS+mean extraction and output dimension.
- [ ] Retain final CLS only as a named compatibility sensitivity.
- [ ] Obtain independent dossier approval.

### MOMENT Small

- [ ] Pin source, weight, package and dependency revisions.
- [ ] Prove raw left-padding, input-mask and internal RevIN parity.
- [ ] Prove official embedding-mean contract.
- [ ] Prove official classification-concat contract.
- [ ] Separate masked reconstruction, classification and forecast task branches.
- [ ] Obtain independent dossier approval.

### Kronos Mini

- [ ] Mark all Tokenizer-base historical arms invalid.
- [ ] Pin Mini with `Kronos-Tokenizer-2k` fail-closed.
- [ ] Document OHLCVA amount fallback, stamps, timezone, normalization and clip ±5.
- [ ] Reproduce official probabilistic forecast and inverse normalization.
- [ ] Label hidden states custom.
- [ ] Obtain independent dossier approval.

### Kronos Small

- [ ] Pin Small with `Kronos-Tokenizer-base` fail-closed.
- [ ] Document OHLCVA amount fallback, stamps, timezone, normalization and clip ±5.
- [ ] Reproduce official probabilistic forecast and inverse normalization.
- [ ] Label hidden states custom.
- [ ] Obtain independent dossier approval.

### Chronos V1

- [ ] Pin model/package/source revisions and resolve the stale embedding-API registry claim.
- [ ] Reproduce official sampled forecast distribution.
- [ ] If `embed()` exists in the pinned contract, reproduce its exact output before pooling/fusion.
- [ ] Document univariate independent-channel semantics and native adaptation support.
- [ ] Obtain independent dossier approval.

### Chronos Bolt

- [ ] Pin model/package/source revisions.
- [ ] Reproduce official quantile forecast and scaling state.
- [ ] Reproduce official `embed()` output and declare pooling/channel fusion separately.
- [ ] Document supported native tuning surface.
- [ ] Obtain independent dossier approval.

### Chronos-2

- [ ] Pin model/package/source revisions.
- [ ] Reproduce joint multivariate forecasts, grouping and covariate behavior.
- [ ] Reproduce official `embed()` including context, REG and output-patch token semantics.
- [ ] Reproduce official `fit()` full/LoRA baseline before custom tuning.
- [ ] Obtain independent dossier approval.

### TimesFM 2.5

- [ ] Pin model/package/source revisions.
- [ ] Prove Transformers-wrapper parity with the official wrapper.
- [ ] Preserve native point/quantile outputs and raw-series normalization contract.
- [ ] Verify official LoRA adaptation; label hidden states custom.
- [ ] Obtain independent dossier approval.

### TTM-R2

- [ ] Write a failing test for current artificial padding/checkpoint selection.
- [ ] Select the correct context/horizon/frequency checkpoint through the official API.
- [ ] Remove trailing-zero padding.
- [ ] Prove scaler, frequency-prefix and channel-mixer initialization behavior.
- [ ] Keep representation and adaptation blocked until forecast parity passes.
- [ ] Obtain independent dossier approval.

### Moirai-2 Small

- [ ] Pin model/package/source revisions and record CC-BY-NC research-only restriction.
- [ ] Reproduce official packed multivariate quantile forecasts.
- [ ] Verify custom token indexing/loss before any tuning.
- [ ] Label hidden states custom.
- [ ] Obtain independent dossier approval.

### Toto-2 22M

- [ ] Pin model/package/source revisions.
- [ ] Reproduce joint masks, group IDs, scaling and quantile outputs.
- [ ] Set `decode_block_size=None` for the short-horizon primary contract.
- [ ] Mark historical trained stages invalid; retain zero-shot forecast role only.
- [ ] Obtain independent dossier approval.

### Sundial Base

- [ ] Build an isolated pinned compatible inference environment.
- [ ] Reproduce finite official samples and inverse normalization.
- [ ] Keep hidden states and tuning excluded.
- [ ] Obtain independent dossier approval.

### TabPFN-TS

- [ ] Pin package/model/license revisions.
- [ ] Prove support sets remain inside each training fold.
- [ ] Define in-context forecast and downstream-control roles only.
- [ ] Obtain independent dossier approval.

## Phase 2 — shared contract harness

- [ ] Official-example parity fixture for every admitted family.
- [ ] Scaling/inverse-scaling tests.
- [ ] Padding/mask/missing-value tests.
- [ ] Frequency/timezone/covariate tests.
- [ ] Joint-versus-independent channel perturbation tests.
- [ ] Single/batch/repeated/partitioned batch parity.
- [ ] FP32 finite reference and reduced-precision tolerance.
- [ ] Context/horizon boundary tests.
- [ ] Prefix-invariance and no-future preprocessing tests.
- [ ] Gradient/freeze-surface tests.
- [ ] Native repeated-batch loss-decrease smoke.
- [ ] Exact resume, save/reload and deployment/export parity.
- [ ] Signed machine-readable admission report.

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
