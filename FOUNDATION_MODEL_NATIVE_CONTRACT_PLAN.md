# Foundation Model Native-Contract Methodology

## Authority and decision

This document governs all new cross-family foundation-model work. It supersedes any requirement
that every model complete a universal Stage 1 → Stage 2 → Stage 3 chain.

No new model training, Corpus v3 materialization, cross-family ranking or prospective evaluation
may begin merely because a model dossier exists. Training remains blocked until the relevant model
passes the shared parity harness, Phase 3 data contract is sealed, and the corresponding frozen
Phase 4/5 baselines are complete in
[FOUNDATION_MODEL_NATIVE_CONTRACT_TASKS.md](FOUNDATION_MODEL_NATIVE_CONTRACT_TASKS.md).
The executable family dossiers and current blocked/admitted status are indexed in
[FOUNDATION_MODEL_NATIVE_DOSSIERS.md](FOUNDATION_MODEL_NATIVE_DOSSIERS.md). The machine sources of
truth are [`config/foundation_models/native_contracts.json`](config/foundation_models/native_contracts.json)
and [`config/foundation_models/native_contract_evidence.json`](config/foundation_models/native_contract_evidence.json).

The historical tournament remains immutable evidence about its exact adapters. It is not a valid
ranking of the best correctly configured version of every foundation-model family.

## Judgment

The previous definition of “apples to apples” was wrong. Equal dates, windows and optimizer budgets
do not create fairness when models receive incorrect tokenizers, non-native normalization, custom
pooling presented as canonical extraction, unsupported training objectives or artificial padding.

The correct standard is:

> Equal causal information, eligible rows, labels, splits, downstream heads, execution rules,
> costs and statistical tests; model-native preprocessing, tokenization, outputs and supported
> adaptation.

Correct configuration does not require every model to train. A legitimate outcome is
`forecast-only`, `frozen-only`, `research-only` or `blocked`.

## Status vocabulary

Every model artifact and result must have one of these statuses:

| Status | Meaning |
|---|---|
| `native_valid` | Reproduces a pinned official input/output or embedding contract and passes parity |
| `native_valid_experimental_task` | Native model contract with a clearly project-specific downstream task |
| `configuration_specific` | Finite, reproducible project adapter without an official native extraction claim |
| `invalid_contract` | Violates a documented tokenizer, padding, input, output or adaptation contract |
| `blocked` | Required parity, dependency, numerical, licensing or governance condition has not passed |
| `research_only` | Technically admissible for research but prohibited from deployment by license/governance |

No invalid or blocked arm receives a zero score. It is excluded with a reason.

## Historical artifact disposition

Preserve every historical file and hash. Add status metadata rather than deleting or overwriting
artifacts.

Immediate reclassification:

- Kronos Mini vanilla and trained arms using `Kronos-Tokenizer-base` are `invalid_contract`; the
  official pairing is Mini with `Kronos-Tokenizer-2k`.
- Mantis arms using external per-window channel z-scoring are `configuration_specific`, not
  canonical Mantis.
- MantisV2 final-CLS extraction is a compatibility arm, not the enhanced V2 primary extraction.
- TTM representations built from 256 real bars plus 256 artificial trailing zeros are
  `invalid_contract`.
- Toto-2 trained stages are `invalid_contract` as native adaptation because the pinned upstream
  path does not support that training contract.
- Sundial's failed hidden-state extraction attempts remain blocked evidence; no valid Sundial
  representation or trained-stage artifacts are implied. Zero-shot forecasting is separately
  eligible only after pinned-environment parity.
- MOMENT pooled-per-channel concatenation, Kronos hidden-state mean, TimesFM hidden-state mean,
  Moirai token mean and Toto internal-state mean are `configuration_specific` until an official
  representation contract proves otherwise.
- Chronos hidden-state/embedding outputs must be classified per pinned model/API. An `embed()` API
  authorizes feature extraction, not a claim of native classification capability.

Claims in historical reports must be qualified with “under the historical adapter contract.”

## Model-role matrix

`F` is native forecasting, `R` is official representation/embedding, `C` is custom representation
transfer, `B` is the project-specific supervised barrier task, and `D` is downstream-only.

| Family | F | R | C | B/D | Current technical status and role |
|---|---:|---:|---:|---:|---|
| MantisV1 | — | native valid | blocked | blocked | Official final-CLS per-channel representation |
| MantisV2 | — | native valid | blocked | blocked | Official layer-2 combined per-channel representation |
| MOMENT Small | blocked | native valid | blocked | blocked | Pretrained masked embedding mean; task heads require separate evidence |
| Kronos Mini | native valid | — | blocked | blocked | Joint OHLCVA forecast with Tokenizer-2k |
| Kronos Small | native valid | — | blocked | blocked | Joint OHLCVA forecast with Tokenizer-base |
| Chronos V1 | native valid | native valid | blocked | blocked | Seeded forecast plus unpooled public tokens/tokenizer state |
| Chronos Bolt | native valid | native valid | blocked | blocked | Quantile forecast plus unpooled public tokens/location-scale state |
| Chronos-2 | native valid | native valid | blocked | blocked | Grouped forecast plus unpooled public tokens/scaling state |
| TimesFM 2.5 | native valid | — | blocked | blocked | Official-wrapper-equivalent point/raw-quantile forecast |
| TTM-R2 | native valid | — | blocked | blocked | Repaired selector, real 512-bar input, native scaler/prefix |
| Moirai-2 Small | research only | — | blocked | blocked | Packed native forecast, noncommercial only |
| Toto-2 22M | native valid | — | blocked | blocked | Zero-shot forecast only |
| Sundial Base | native valid | excluded | blocked | blocked | Isolated forecast-only environment |
| TabPFN-TS | — | — | — | D blocked | Fold guard repaired; model terms/artifact unavailable |

A technically valid cell still requires a current evidence-bound report and two independent
approvals before execution. Every custom/adaptation cell remains blocked until its own evidence
passes; no current arm is training-admitted.

The current verifier enforces a hard minimum of two normalized, distinct reviewer labels and
valid approval timestamps. It does **not** cryptographically authenticate reviewer identity or
organizational independence. Operational authorization must remain blocked until that external
governance step is completed or a trusted-signature mechanism is implemented.

## Three separate leaderboards

### Track F — native forecasting

Use only pinned official forecast APIs. Save distributions before reducing them.

Required artifact contents:

- Point forecast.
- Native quantile levels and values, or seeded forecast samples.
- Native normalization/inverse-transform metadata.
- Context, horizon, frequency and covariate settings.
- Joint versus independent variate semantics.
- Sample count, RNG seed and decoding controls.
- Model, package, source and weight revisions.

Common metrics:

- MAE, RMSE and MASE.
- Mean pinball/weighted quantile loss on a declared common quantile grid.
- Empirical interval coverage, interval width and quantile-crossing rate.
- CRPS when native samples are available, or a declared quantile approximation.
- Return IC/rank IC, forward-volatility skill and path metrics as secondary trading diagnostics.

Native family metrics may be retained, but native training losses are never compared across
families.

### Track R — official frozen representation

Only a documented embedding/classification API or an exact upstream parity implementation may be
called official. Record exact layer, tokens, pooling and output dimension.

Every arm receives identical downstream rulers:

- Low-variance ridge/logistic diagnostics.
- A separately declared capacity-controlled nonlinear sensitivity.
- Train-fold-only scaling/PCA.
- Causal-only, representation-only, causal+representation and residual-over-causal comparisons.
- Random, shuffled and time-destroyed controls.

### Track C — custom feature transfer

Custom hidden states and forecast-distribution summaries are legitimate experiments but live in a
separate table. They cannot determine the “best native representation.”

Feature types remain separate:

1. Official embeddings.
2. Custom hidden states.
3. Native forecast-distribution features.

### Track B — supervised barrier/path adaptation

This is the primary trading-aligned track, not a native pretraining claim.

All compatible models use identical rows and output decomposition. A backbone arm must use an
admitted official interface; otherwise it is Track C custom transfer and cannot silently enter the
native Track B comparison:

```text
P(favorable first)
P(adverse or ambiguous first)
P(neither)
E[terminal R | neither]
```

Compare, on identical rows:

1. Causal features only.
2. Official embedding only.
3. Forecast-distribution features only.
4. Causal + official embedding.
5. Causal + forecast features.
6. Frozen backbone with the common neural barrier head.
7. Partially tuned backbone with the same head.
8. Full tuning only if partial tuning passes.

Use the same head family and train-only search budget, not blindly equal raw parameter counts across
unequal input widths. A common train-fold-fitted bottleneck controls input dimension; actual head
parameters are reported. Calibration, threshold rule, concurrency, fees and slippage contract
remain equal. Projection/PCA is fit inside each training fold.

## Two fairness views

### Common-information view

Equal across families:

- Decision rows and timestamps.
- Maximum causal history available.
- Labels and complete label-end timestamps.
- Folds, purge, embargo and costs.
- Downstream ruler and statistics.

Allowed to differ:

- Native causal normalization.
- Padding and masks.
- Tokenization and frequency features.
- Joint or channel-independent processing.

Primary paired comparisons use the intersection of valid rows.

### Native-best view

Each family may use a context length, official covariates and native settings selected entirely
inside training-only folds. These results are reported separately because information exposure may
differ. They cannot be presented as information-matched rankings.

## Native-contract dossier

Every model/version requires an immutable machine-readable dossier containing:

- Model ID, exact weight revision and weight hashes.
- Upstream repository, source commit, package revision and dependency lock.
- License and deployment restriction.
- Native task and pretraining loss.
- Input tensor shape, channel order and feature semantics.
- Context and prediction limits.
- Normalization, clipping and inverse transform.
- Padding, missing-value masks and frequency/covariate behavior.
- Joint, grouped or channel-independent multivariate semantics.
- Official output type.
- Embedding API, layer, token and pooling contract, if supported.
- Supported fine-tuning method and exact trainable parameters.
- Precision/dtype policy.
- Save/resume/export/deployment contract.
- Known incompatibilities and the result that would falsify admission.

The unified registry must expose capabilities per track. One `supported_training` Boolean is
insufficient.

## Contract admission tests

A family cannot be benchmarked until all applicable tests pass:

1. Reproduce a pinned official example in its compatible dependency environment.
2. Official public API and project adapter agree on fixed raw fixtures.
3. Native scaling and inverse scaling match.
4. Padding and missing masks exclude unavailable history.
5. Frequency, timestamps, timezone and covariates match the official contract.
6. Joint models respond appropriately to another channel; independent models remain independent.
7. Single, repeated and differently partitioned batches agree within tolerance.
8. FP32 reference outputs are finite; reduced precision passes a declared tolerance.
9. Context/horizon boundary behavior matches documentation.
10. Prefix invariance proves no future-dependent preprocessing.
11. Intended parameters receive gradients; frozen parameters do not.
12. Native loss decreases on a repeated toy batch for trainable arms.
13. Exact resume reproduces the uninterrupted trajectory.
14. Save/reload and deployment export reproduce the selected checkpoint.
15. License and data governance permit the declared role.

A synthetic and small calibration-only qualification dataset may test numerical behavior and
resource feasibility. It is excluded from every ranking and may not select economic variants.

## Family-specific P0 corrections

### MantisV1/V2

- Pin external source and weight revisions.
- Add raw-input canonical paths; external z-score remains a sensitivity only.
- Make native length/interpolation explicit.
- Verify V1 final-CLS compatibility.
- Verify V2 layer-index-2 combined CLS+mean extraction as primary.
- Fold channels through independent encoder calls and preserve fixed OHLCV order.
- Treat official crop/resize contrastive learning as the native SSL control.
- Treat barrier/path tuning as project-specific supervised transfer.

### MOMENT

- Verify raw left-padding and mask behavior against the pinned official pipeline.
- Implement and label official embedding-mean and classification-concat contracts separately.
- Keep masked patch reconstruction as a native branch.
- Keep adjacent-half contrastive learning custom and non-mandatory.

### Kronos Mini/Small

- Enforce Mini ↔ Tokenizer-2k and Small ↔ Tokenizer-base fail-closed pairing.
- Use joint OHLCVA, declared amount fallback, calendar stamps, official normalization and clip ±5.
- Admit the official greedy output/inverse-transform surface first across all six project
  timeframes; treat stochastic/probabilistic decoding as a separate evidence expansion.
- Keep `decode_s1` hidden-state pooling custom.
- Use tokenizer reconstruction and autoregressive hierarchical-token training only as native
  adaptation.

### Chronos V1/Bolt/2

- Verify the exact pinned package/API rather than trusting the stale local registry.
- Reproduce official forecast distributions and official `embed()` behavior where present.
- Preserve Chronos-2 REG/output tokens and grouped multivariate semantics.
- Establish official `fit()` or supported tuning baselines before custom objectives.
- Treat embedding availability as feature support, not proof of classification capability.

### TimesFM 2.5

- Prove Transformers-wrapper parity with the pinned official wrapper.
- Treat the pinned forward API as frequency-agnostic; it has no external frequency input.
- Preserve raw-series LoRA/native forecast behavior and probabilistic outputs.
- Keep hidden-state pooling custom.

### TTM-R2

- Bind project timeframes to resolution-prefix tokens `1/0/3/5/6/7`, with 3-minute explicitly OOV.

- Select the correct context/horizon/frequency checkpoint branch through the official API.
- Remove artificial trailing-zero padding.
- Prove scaler, frequency-prefix and channel-mixer initialization behavior.
- Remain blocked until forecast parity passes; representation pooling remains custom afterward.

### Moirai-2

- Prove official packed multivariate forecast/output parity.
- Verify custom token indexing and loss before any tuning.
- Keep hidden-state pooling custom and deployment research-only.

### Toto-2

- Use official joint multivariate masks/group IDs and internal scaling.
- Use `decode_block_size=None` for the short-horizon primary contract.
- Remain zero-shot forecast-only until upstream supports a verified training surface.

### Sundial

- Run in an isolated pinned compatible environment.
- Admit only finite official forecast samples with normalization/inverse-transform parity.
- Exclude hidden-state and trained-stage arms.

### TabPFN-TS

- Keep the support set entirely inside each training fold.
- Use only as a nested in-context forecast/downstream control.

## Data and label contract

Ticks and bars have different jobs.

- Approved ordered ticks construct contract-safe regular bars and exact observed path labels.
- Current foundation models receive regular scalar time-series inputs unless they have a verified
  irregular-event tokenizer.
- Tick labels begin strictly after the decision and declared entry time.
- Ordered ticks determine observed barrier order, MFE, MAE and time-to-barrier; they do not prove a
  fill without the declared bid/ask/marketability contract.
- When exact tick order is unavailable, preserve an ambiguity flag and use the conservative rule
  only for executable scoring.
- Native future-series forecast targets remain separate from tick-derived trading labels.
- Fixed wall-clock labels require actual timestamp/cadence checks.
- Purge and embargo use each row's complete label-end timestamp.
- No window crosses a contract roll; causal roll selection cannot use future volume.

The primary bar set remains 1/3/5/15/30/60 minutes. Sub-minute inputs are separate experiments.

## Hyperparameter and exposure policy

- Official defaults are always the first baseline.
- Predeclare a small family/task-specific search space.
- Perform all selection in inner chronological folds.
- Extractor layer, pooling, preprocessing, context, PCA dimension and forecast-feature summaries
  are hyperparameters and multiplicity entries.
- If two official contracts are legitimate, declare the primary from upstream guidance before
  economic scoring; report the other as sensitivity.
- No retuning after outer development results.
- Equalize unique parent-window presentations, not optimizer steps or flattened channel count.
- Record parent presentations, channel-series presentations, tokens/patches, trainable/total
  parameters, FLOPs, GPU-hours, memory and early-stop epoch.
- Report an accuracy-versus-compute frontier separately.

## Adaptation graph

There are no universal ordinal stages:

```text
Pinned vanilla
├── native forecast adaptation
├── native representation adaptation, if supported
├── architecture-compatible custom SSL hypothesis
└── supervised barrier/path adaptation
```

Rules:

- Every branch begins directly from vanilla.
- A chained child is allowed only after an identical-row ablation proves the parent adds value.
- Failed parents cannot be used to complete a sequence.
- Objective names describe tasks, not progress stages.
- Training heads/projectors used at deployment must be included in the deployed bundle.
- Custom SSL cannot overwrite or replace native leaderboard results.

## Statistics and promotion

- One seed may reject a catastrophic pilot; at least three training seeds are required to promote.
- Use paired calendar- and root-aware block inference.
- Report pooled, stream-macro, root and per-timeframe results plus effective sample size.
- Select within family/track inside nested development folds.
- Apply hierarchical multiplicity before comparing one declared representative per family/role.
- Require minimum trade count, coverage and breadth.
- Primary execution uses instrument fees, zero added delay and the declared zero-tick slippage
  assumption; frozen one-tick sensitivity is mandatory and cannot trigger retuning.
- Representation promotion requires paired incremental value over both vanilla and causal-only
  controls, not descriptive probe improvement.
- Forecast promotion requires native forecast/calibration improvement and downstream incremental
  path value where trading use is claimed.
- Deployment parity and licensing must pass.

All 2024–2026 evidence is spent for model/extractor/head selection. It remains training-excluded
under the project definition, but final confirmation for this new program must use prospectively
arriving data after every model, extractor, head and threshold is frozen.

## Execution phases

### Phase 0 — stop, preserve and relabel

- Freeze historical artifacts and hashes.
- Add per-arm native/configuration/invalid status.
- Correct historical claims without rewriting recorded numbers.
- Create the unified capability registry.

### Phase 1 — native contract dossiers

- Complete and independently review every model/version dossier.
- No family owner approves its own dossier.

### Phase 2 — reference parity harness

- Implement official-example, transform, output, embedding, batching, precision, gradient,
  resume and deployment parity tests.
- Block every failing arm.

### Phase 3 — Corpus v3 development contract

- Select and freeze roots using governance, liquidity, continuity, gaps and coverage only. Strategy
  event or outcome yield cannot influence universe membership.
- After the universe freezes, measure event/label yield and decide only whether materialization is
  computationally and statistically viable—not which roots to retain.
- Seal common raw rows, native transforms, native future-series targets and separate tick-derived
  path labels.
- Store complete row, contract, source and label-end identity.

### Phase 4 — frozen native baselines

- Run official forecasts and official embeddings without training.
- Save probabilistic outputs and all contract metadata.
- Keep custom-transfer arms in a separate table.

### Phase 5 — identical downstream ruler

- Compare causal, embedding, forecast-feature and fused arms on identical rows.
- Use nested chronological calibration and paired block inference.
- Determine whether any foundation information adds value before adaptation.

### Phase 6 — family-native adaptation

- Begin only after the model passed Phase 2, Phase 3 is sealed, and its Phase 4/5 frozen baselines
  are complete.
- Train supported objectives directly from vanilla.
- Match parent exposure and report compute.
- Do not chain or broaden until a branch passes its own vanilla control.

### Phase 7 — bounded supervised barrier and custom SSL work

- Run frozen → partial → full supervised tuning only for surviving compatible families.
- Fund one architecture-specific SSL hypothesis only when representation quality remains a proven
  binding constraint.

### Phase 8 — selection and freeze

- Select one declared representative per family/role inside nested development.
- Apply multiplicity correction, require three-seed stability and freeze complete bundles.

### Phase 9 — prospective confirmation

- Read only newly arriving data after the complete freeze.
- Do not change objectives, extractors, heads, thresholds or policies after the read.

## Stop conditions

Stop a family or track when:

- Official parity cannot be reproduced.
- Required outputs are non-finite or batch-sensitive beyond tolerance.
- License/governance blocks the intended role.
- Native frozen/forecast features add no incremental value over causal controls.
- Adaptation fails its own vanilla parent in two independent seeds.
- Improvement exists only after selecting among undeclared extractor/head variants.
- Deployment cannot reproduce the evaluated function.

This process is deliberately front-loaded. Correctness is cheaper than another invalid full sweep.
