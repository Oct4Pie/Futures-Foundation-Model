# Foundation Model Native-Contract Tasks

This checklist is governed by
[FOUNDATION_MODEL_NATIVE_CONTRACT_PLAN.md](FOUNDATION_MODEL_NATIVE_CONTRACT_PLAN.md). No training or
new ranking is authorized until the relevant admission items pass.

## Historical native F/R evidence checkpoint — 2026-07-17 PDT / 2026-07-18 UTC

This checkpoint predates the mandatory 15-arm TabPFN identity split. Its bundles remain historical
technical evidence, but its registry-bound reports do not authorize the refrozen registry.

- Methodology source: `8e2bd47d8fd6dc333dfa74ad0eea3d3613e63469`.
- Registry content SHA-256: `6134f06aa223671f521099f35332dab23a240fc3db7cd9a6a3b18650b06fa233`.
- Canonical evidence content SHA-256:
  `88c0d1c594747668d57354a900ffa42a3e90ceec89be0daa8bacd7f072003d2a`.
- All 16 admitted native forecast/representation tracks passed canonical execution and an
  independent replay; every output key, dtype, shape and array byte matched. The replay comparison
  is sealed in
  [`output/native_parity_evidence_v2_replay/replay_attestation.json`](output/native_parity_evidence_v2_replay/replay_attestation.json).
- A fresh wheel unpacked outside the source checkout verified all 16 archived bundles. A fresh clone
  from the project remote reproduced the registry/evidence hashes and replay comparison.
- Verification suite: 1,035 passed, 96 skipped and 8 existing warnings.
- Trusted reviewer keys: 0. Runtime-authorized tracks: 0. Training-admitted tracks: 0. Independent
  agent reviews are technical rubber-duck evidence, not organizational approvals and were not
  converted into signatures.
- Scope is deliberately narrow: 512-context, 16-horizon, FP32 synthetic native F/R parity. This is
  not classification, custom pooling, fine-tuning, SSL, barrier-task, market-data, trading-quality,
  resume/export, deployment, or OOS evidence.

## Phase 0 — historical evidence and registry

- [x] Snapshot and hash the current historical tournament index in
  [`config/foundation_models/historical_native_contract_snapshot.json`](config/foundation_models/historical_native_contract_snapshot.json).
- [x] Add `native_valid`, `configuration_specific`, `invalid_contract`, `blocked` and
  `research_only` statuses without modifying old predictions or scores.
- [x] Quarantine Kronos Mini wrong-tokenizer arms, TTM padded representations and Toto trained
  stages; record Sundial hidden-state extraction as a failed attempt without implying valid staged
  artifacts exist.
- [x] Relabel noncanonical Mantis, custom MOMENT pooling and all custom forecast-model hidden states.
- [x] Add the historical-adapter-contract warning to the active tournament review, full-history
  downstream result, conditional-event gate and legacy-OOS confirmation documents; retain the
  existing warning in `FOUNDATION_MODEL_RESULTS.md`.
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

### TabPFN-TS3 forecast

- [ ] Pin package/model/license revisions.
- [x] Prove support sets remain inside each training fold.
- [x] Separate the in-context forecast identity from generic TabPFN downstream fitting.
- [ ] Obtain the gated checkpoint, accept its terms and verify native forecast outputs.
- [ ] Obtain independent dossier approval.

### TabPFN V3 downstream

- [ ] Pin the exact generic model, source, package and license revisions.
- [x] Prove support sets remain inside each training fold.
- [x] Keep this identity out of persistent encoder/SSL training.
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
- [ ] Add numerical prefix-invariance tests where an API accepts a larger composite input. Base
  native APIs receive context only and therefore report this check as not applicable, not passed.
- [ ] Gradient/freeze-surface tests.
- [ ] Native repeated-batch loss-decrease smoke.
- [ ] Exact resume, save/reload and deployment/export parity.
- [x] Evidence-bound machine-readable v3 admission report with route, measured Python/package
  environment, consumer-supplied execution controls, and exact runtime artifact trees.
- [x] Add a complete deterministic runtime lock over all installed distributions, interpreter,
  platform, Torch CUDA/cuDNN, visible devices, GPU identity and measurable driver details; remeasure
  it at report build and execution. Legacy evidence without this lock cannot authorize execution.
- [x] Enforce a non-bypassable two-approval floor, distinct reviewer/key fingerprints, Ed25519
  signatures, and request/approval/finalization timestamp ordering.
- [x] Install an explicit trusted-public-key registry that fails closed while empty. Independent
  organizational assignment of those keys remains an external governance responsibility.
- [x] Install a durable canonical raw-bundle archive and require admission-report build/verification
  to reopen its fixture, result, log, manifest and raw-output hashes. Current runtime artifacts are
  bound separately because installing reviewed evidence necessarily advances the source checkout.
  The executable-code binding uses a stable package/script source manifest rather than Git HEAD.
- [x] Restrict operational consumers to a clean source checkout. Wheel installs may inspect the
  canonical archive but cannot authorize model execution in this phase.

## Phase 2B — training-route authority

- [x] Refreeze the inference roster as 15 arms and split TabPFN forecast/downstream identities.
- [x] Install one packaged, declarative 42-route/29-profile training catalog bound to inference
  dossiers by content hash; remove duplicated model/license pins from training semantics.
- [x] Delete the v1 free-text route/evidence JSON authority and retain only an always-blocked,
  catalog-derived compatibility facade.
- [x] Reject the 17 demonstrated coherent semantic corruptions.
- [x] Gate every discovered direct Torch optimizer/backward script before model or data access.
- [x] Install a minimal blocked training-data authority bound to the Corpus v3 contract. Remove
  arbitrary digest bags from route instances; unresolved materialized data dependencies remain
  explicit blockers rather than placeholder hashes.
- [x] Bind strict route templates to exact resolved catalog routes and authoritative data contracts.
  The current data authority remains blocked, so no concrete route instance can be built.
- [x] Admit route-specific synthetic smoke, exact-resume and export evidence for the five exact
  development routes that passed all 20 mandatory checks: Chronos Bolt forecast, Chronos V1
  forecast, MOMENT masked reconstruction, Kronos Mini tokenizer and the parent-bound Kronos Mini
  predictor. Kronos Small tokenizer remains explicitly failed on both chronology controls. Smoke
  admission remains synthetic-only and does not imply pilot, full-training or deployment admission.

### Phase 2B execution-readiness checkpoint — 2026-07-18

- [x] Add a deterministic, non-authorizing execution-readiness audit at
  `scripts/audit_native_training_readiness.py`. It binds the 42-route catalog, route-profile hashes,
  current launcher bytes, blocked training-data authority, inference-registry state and Git
  methodology state.
- [x] Distinguish catalog inventory from exact executable training routes. Current measured counts
  are 42 catalog routes, 23 exact executors, 17 fully closed unsupported routes and two externally
  blocked in-context routes; zero candidate mismatches or unspecified implementations remain.
  Twenty-one exact routes pass all 20 smoke checks, Kronos Small tokenizer has complete failed-smoke
  evidence and its predictor is parent-blocked. All 23 exact routes have terminal pilot and
  downstream dispositions: 15 bounded pilots completed, nine survive their native objective, six
  classification routes are blocked on governed labels, eight compatible downstream screens fail,
  and zero routes are training-admitted.
- [x] Expand the shared smoke harness with one-batch gradient/update, controlled learnable loss
  decrease, shuffled/time-destroyed control rejection, exact interruption/resume comparison,
  future-corruption invariance, invalid-boundary rejection, multivariate layout, bounded
  throughput/memory and negative-price behavior checks. These are reusable test primitives, not
  route evidence.
- [x] Reclassify legacy optimizer tests as kill-switch tests. Model construction, pure native-loss
  seams and export parity remain testable, while every retired optimizer fails before checkpoint
  writes or data access.
- [x] Build exact architecture-native executors for all 23 internally supported routes and
  route-specific raw smoke bundles for every executable route whose parent lineage is available.
  The readiness audit reopens every raw artifact, records a complete failed smoke without admitting
  it, and intentionally cannot convert generic harness checks or historical inference parity into
  training admission.

### Cross-model configuration audit checkpoint — 2026-07-19

- [x] Add `scripts/audit_native_configuration.py` and the fail-closed
  `ffm_native_configuration_audit_v1` contract. It compares all 15 inference identities, 42
  routes and 29 profiles, and independently inspects all 23 exact executors' Python constants,
  route defaults, optimizer construction, scheduler runtime, preprocessing/time contract and
  export surface.
- [x] Correct four configuration-contract defects found by that audit rather than grandfathering
  prior evidence: Kronos parity now derives calendar features in `America/Chicago` from the UTC
  source timeline; exact executors advertise only the implemented accumulation value `1`; cosine
  no-warmup routes advertise exactly zero warmup; Kronos tokenizer profiles no longer claim stamp
  inputs and Chronos forecast bundles no longer claim unexported hidden states/tokens.
- [x] Record PyTorch OneCycle's native Kronos behavior explicitly: AdamW starts at betas
  `(0.9,0.95)`, while the official/default OneCycle scheduler cycles beta1 over `[0.85,0.95]` with
  cosine annealing, `pct_start=0.03`, `div_factor=10` and `final_div_factor=1e4`.
- [x] Build clean parity wheel
  `117e103a451751f5f1a0b2cd278ab8745e42373f41380a124521ade1b674e06e`
  and regenerate all 16 currently admitted native F/R parity tracks against the current 15-arm
  registry. Aggregate SHA-256:
  `c9fe0cd39fed22297ca52fab2cf42074d3e6f7034608326674c88db6734f015d`.
  Build the final sealed wheel at
  `13540578b3fd6050b78572978fd341dd8e3d0bf605c6901ce6e7419099378672`;
  its parity-critical registry/catalog/worker/adapter/route bytes match the tested checkout exactly.
- [x] Regenerate every exact-route smoke, every eligible pilot and the common-information downstream
  ruler after the catalog/profile/executor hashes changed. Twenty-one smokes pass; Kronos Small
  tokenizer remains eliminated by both chronology controls and its predictor remains parent-blocked.
  Fifteen pilots complete and nine survive; all eight compatible common-information routes fail the
  downstream funding screen. Current screen collection:
  `e74d0b5476e7dead9bfcecb2430abc8b53ee64a8bf1ee90bf020f71084e29597`.
- [x] Seal configuration audit
  `a32d731ed5b7e521ee003e18aaf982b954f856f8a8595dc230138108fca7b374`
  with zero cross-layer discrepancies, zero unresolved constraints and current inference parity
  complete. This does **not** say all models are admitted: two TabPFN identities remain externally
  blocked, 17 routes are explicitly unsupported, zero routes pass the downstream funding gate, and
  training, OOS, deployment and trading admission remain false. Current readiness SHA-256:
  `5583b6d9dc975dd62a88fe1102fbb38fc9567447c773a4497574165b3d5c126f`.

## Phase 3 — data and label contract

- [x] Complete the outcome-blind, manifest-level Corpus v3 inventory audit and pin all governance
  and leaf-manifest inputs. Root selection remains blocked pending a sessionized export audit.
- [ ] Complete the governed sessionized Corpus v3 coverage/yield audit before materialization.
  - [x] Implement the independent FFM consumer audit and bound CLI. It accepts only a reverified
    session-denominator-bundle capability plus reverified contract-day export capabilities, checks
    every event against exact denominator segments, preserves missing sessions explicitly and can
    never select roots or authorize materialization/training.
  - [x] Freeze the current export authority as the exact empty canonical index
    `config/corpus_v3/sessionized_export_index_v1.json` (physical SHA-256
    `07d95cc48ead4d8a6f45b9f7824ea2ca386fb7bdabbb729c30da7214129e461d`; semantic SHA-256
    `a4b8181fe0a54db5d99000b535f8f9ef2ebfea4cf1d0a97c2885f4078ee72d72`). This states that zero
    scale exports are currently admitted; it does not infer inventory from the filesystem.
  - [x] Implement independent FFM consumers for detached producer governance/frozen split-use,
    joined metadata-only provider pagination, official-claim lifecycle/quarantine, exact expected
    requests and a split-scoped inventory/non-executable plan. The two production-facing CLIs are
    `scripts/build_corpus_v3_expected_requests.py` and
    `scripts/build_corpus_v3_materialization_plan.py`.
  - [ ] Populate the production 43-root producer governance, provider evidence, lifecycle,
    session denominator, expected-request denominator, inventory observations and nonempty verified
    export index; obtain immutable-storage and independent materialization approval.
- [ ] Freeze common-information and native-best views separately.
- [ ] Seal regular bars, contract IDs, source rows and timestamps.
- [ ] Seal native future-series forecast targets separately from trading labels.
- [ ] Materialize post-entry ordered tick barrier, MFE, MAE and time-to-barrier artifacts from the
  admitted streaming export.
- [x] Implement and adversarially test the strict contract-day tick-label semantics on synthetic
  governed rows; real label artifacts remain blocked on the admitted streaming export.
- [ ] Store exact/ambiguous coverage and deterministic conservative fallback.
- [ ] Validate wall-clock cadence, session gaps, rolls and complete label-end timestamps.
- [x] Purge/embargo by inclusive label end and, for legacy produce paths, against the first
  validation context start rather than its decision timestamp.
- [ ] Version tick size/value, fees and zero/one-tick scoring contracts.

### Phase 3 integrity checkpoint — 2026-07-18

- [x] Require nonempty contract IDs in SSL, event-context, path-label and standalone Kronos window
  ingestion. Exact cadence and contract segments are mandatory.
- [x] Remove the four-day gap escape hatch and handwritten CME maintenance/weekend continuity
  rules. SSL and event contexts now share one conservative exact-cadence policy while session
  authority is unresolved.
- [x] Reset path ATR and all event-control state at contract changes; make the classical control's
  EMA/range scale genuinely bounded to the declared 256-bar context.
- [x] Seal policy arrays and economics/lineage metadata together. Event shards own gross R only;
  the typed economics capability applies fees and the declared slippage scenario exactly once.
- [x] Increment active event, collection, sample, fold, row-selection, dense-label and policy
  schemas. Legacy v1/v2 artifacts require explicit archival opt-in.
- [x] Require and revalidate tournament source-manifest and array identities; stale cache prose no
  longer describes a removed gap policy.
- [x] Independently consume and semantically verify AlphaForge session-bundle v2 with bounded,
  root-dirfd/no-follow transport and adversarial FIFO, symlink, hardlink, mutation, orphan and
  bool-as-int rejection. This is synthetic producer/consumer parity only and is hard-coded
  `production_admitted=false`.
- [x] Add `ffm_corpus_v3_sessionized_coverage_audit_v1` and
  `scripts/audit_corpus_v3_sessionized_coverage.py`. The consumer reopens the exact export index,
  independently reruns raw-leaf contract-day verification at use, reopens the denominator at use,
  rejects events in declared market breaks and records exact missing-session/liquidity/yield rows.
  The original Corpus-v3/export/denominator compatibility sweep is incorporated into the current
  `190 passed` joined authority/materialization sweep; production completion stays blocked because
  the checked-in index has zero entries and the denominator evidence is synthetic.
- [x] Build independent FFM consumers for producer governance, frozen split-use, provider
  candidates, contract lifecycle and official lifecycle evidence before accepting expected
  requests. Every consumer requires exact physical hashes, mandatory reopen-at-use and
  non-forgeable capability continuity; hash-only acceptance is prohibited.
- [x] Derive `ffm_corpus_v3_expected_request_denominator_v1` only from frozen split/use, session
  denominator and lifecycle capabilities. Candidate quarantine, closed sessions and zero
  intersections remain explicit; no month-code inference, availability or market observation is
  permitted.
- [x] Add canonical `ffm_corpus_v3_inventory_observations_v1`,
  `ffm_corpus_v3_split_scoped_inventory_v1` and `ffm_corpus_v3_materialization_plan_v1` contracts.
  Exact request closure, source-path/hash metadata, gap/overlap checks and available-only selection
  are enforced, while source-byte reopening, plan execution and production admission remain false.
- [x] Connect the verified plan-derived request capability to both SSL and event materialization.
  `ffm_corpus_v3_request_authority_v1` reopens the expected-request, inventory and plan artifacts,
  rejects overlapping contract/use intervals and returns exact per-row planned-request IDs. SSL
  independently authorizes train and validation uses and rejects parent windows crossing request
  segments. Event-context v6 binds the authority manifest and request ID into every row, rejects
  context/future paths crossing request segments and reverifies decision plus label endpoints on
  reload. This is mechanism closure only: current production artifacts select zero requests and
  neither materialization nor training is admitted.
- [x] Define and version nonpositive-price-safe path/probe/structural targets using raw price
  changes normalized by decision/context-only causal scale. Log-price preprocessing now rejects
  nonpositive inputs instead of clipping them. Family forecast-score parity for negative prices
  remains a route-level Phase 4 blocker.
- [x] Replace duplicated ticker tick-size tables with one hash/source/date-scoped typed
  instrument-economics capability. Zero added slippage is the primary research ruler and one tick
  is the frozen sensitivity. The fee schedule is explicitly a static research estimate, not a
  historical broker ledger; wider/effective-dated production evidence remains pending.
- [x] Centralize tournament-cache schema, source-manifest and array verification in one loader used
  by adaptation, event materialization and downstream context reconstruction.
- [x] Verify the pre-red-team integrated source with `1249 passed, 98 skipped`, build the wheel, and import the
  new authority/economics/target modules from the extracted wheel outside the checkout.
- [x] Red-team the integrated seam and repair the P0 canonical-gap contradiction, trend-efficiency
  units, Stage-3 future-gap segmentation, economics capability forgery, and probe-result semantic
  versioning. These repairs require a new full-suite count before checkpointing.
- [x] Reverify the P0 repairs with `1250 passed, 98 skipped, 8 warnings`.
- [x] Replace the self-hashed tournament cache with cache schema v3 and a typed, externally
  SHA-bound source authority. Construction and every load revalidate the pinned corpus manifest,
  each source CSV, interval, bar size, transformation/code bytes, array bytes/content, contract set
  and full contract-ID sequence. The cache remains explicitly `training_admitted=false`; a real
  production source authority has not been published.
- [x] Add a strict current-schema event-shard validator at save and load boundaries. Generic arrays,
  repaired-hash semantic tampering, invalid split/label endpoints, malformed tag/policy geometry and
  unverifiable economics cannot be upgraded into a canonical v4 shard; loaded arrays are immutable.
- [ ] Bind and validate gross-shard instrument geometry and the full decision-through-exit
  economics interval. The first implementation is present; adversarial coverage remains required.
- [x] Replace unbounded/path-following YAML reads with shared bounded no-follow authority transport.
  Economics files are single-link regular files with stable descriptor/parent identities, bounded
  bytes/tokens/nodes/depth, UTF-8, duplicate-key rejection and no YAML aliases/anchors. Production
  effective-dated broker/economic evidence remains separate and pending.
- [x] Reverify the completed end-to-end checkout with `1392 passed, 98 skipped, 8 warnings`; enable
  the optional Torch matrix for `1479 passed, 11 skipped, 37 warnings`; run the isolated
  Chronos/XGBoost file with both isolation gates for `17 passed`. Remaining suite skips are optional
  dependency or process gates, not training admissions.

### End-to-end native-route/downstream checkpoint — 2026-07-19

- [x] Publish and reverify the real 54-stream cache-v3 receipt over
  `[2019-07-01,2025-07-01)`: `41281860ff1ef3474e226d22a2df504e97f8d348839fce8d2438689d160b9e0a`.
  It remains `training_admitted=false`; authentic production session/request/lifecycle/roll/label
  authority is still absent.
- [x] Complete 15 one-minute native-objective pilots on identical non-OOS cache authority. Nine
  survive: Chronos Bolt `+2.51%`, Chronos V1 `+1.13%`, MOMENT reconstruction `+40.02%`, Kronos Mini
  tokenizer `+3.18%`, Mini predictor `+1.20%`, Mantis V1 contrastive `+97.32%`, Mantis V2
  contrastive `+66.83%`, MOMENT forecast full `+22.95%` and forecast head-only `+22.87%`.
  TTM full/head, TimesFM LoRA, Chronos-2 full/LoRA and research-only Moirai are eliminated by the
  frozen 1% gate. These are native-loss dispositions only; Kronos Small tokenizer separately fails
  both chronology-control checks and six classification routes remain label-authority blocked.
- [x] Materialize a fresh strict v4 one-minute development collection for all nine roots: 416,191
  decision rows. Build a 10,800-row equal-root sample and a frozen 2,700-row representation screen
  with exactly 300 rows per root and verified 512×5 contexts. No OOS rows were read.
- [x] Preserve native outputs without premature reduction: Chronos Bolt `2700×5×16×9` quantiles,
  Chronos V1 `2700×5×20×64` samples with only the admitted first 16 positions exposed to the common
  ruler, MOMENT reconstruction `2700×512` embeddings, Kronos Mini predictor `2700×16×6` forecasts,
  Mantis V1 `2700×5×256` and V2 `2700×5×512` per-channel representations, and MOMENT full/head
  forecast tensors at `2700×5×16`. The 528-past-bar tokenizer route remains explicitly outside the
  primary 512-bar common-information view.
- [x] Freeze the maximum viable common purged contract at two folds because a third fold empties
  sparse ZB after the required 512-minute embargo. Fold SHA-256:
  `d460f60d442407cb207a63d907bba5b85a84a8ee362f78c959d46de4cc52cdf1`.
- [x] Run the predeclared five-target linear incremental ruler with train-fold-only 32-component
  PCA, causal-only/model-only/combined/residual arms, three model controls and 500 root×week
  bootstrap repetitions at 99% confidence. All eight common-information routes fail. Results as
  `(controls, point wins, degradations, adjusted-CI wins)` are: MOMENT reconstruction `(3,2,3,0)`,
  Bolt `(1,1,4,0)`, Chronos V1 `(1,1,4,0)`, Kronos Mini predictor `(3,0,5,0)`, Mantis V1
  `(4,3,2,0)`, Mantis V2 `(3,4,1,0)`, MOMENT forecast full `(0,0,5,0)` and forecast head-only
  `(0,0,5,0)`. No route funds nonlinear sensitivity.
- [x] Seal the verified aggregate screen collection at
  `e74d0b5476e7dead9bfcecb2430abc8b53ee64a8bf1ee90bf020f71084e29597` and the evidence-aware
  readiness report at `5583b6d9dc975dd62a88fe1102fbb38fc9567447c773a4497574165b3d5c126f`.
  Current counts: 23 exact executors, 21 passing smokes, one failed smoke, one parent-blocked route,
  15 completed native pilots, nine native survivors, eight completed downstream screens, zero
  screen survivors, zero nonlinear sensitivities funded and zero training-admitted routes. Both
  exact-route pilot disposition and surviving-pilot downstream disposition closures are complete.
- [x] Stop the program at the declared gate. No full training, OOS evaluation, deployment, paper
  trading or live trading was run. The wheel built at 37,735,883 bytes with SHA-256
  `13540578b3fd6050b78572978fd341dd8e3d0bf605c6901ce6e7419099378672` and imported all new
  authority/route/downstream modules from an unpacked location outside the checkout.

## Phase 4 — frozen baselines

- [x] Save native quantiles/samples without premature point reduction.
- [ ] Score MAE, RMSE, MASE, WQL/pinball, coverage, width, crossing and CRPS where valid. The
  downstream funding decision did not require a broader forecast-metric tournament after every
  route failed incremental value.
- [x] Extract only exact-route official representations/native forecasts into typed native tables.
- [x] Keep official embeddings, forecast tensors and custom/native-best views separate; no custom
  hidden-state pooling was relabeled as an official representation.
- [x] Run deterministic, shuffled-label, random-feature and time-destroyed controls.

## Phase 5 — identical downstream ruler

- [x] Causal-only baseline.
- [x] Official-embedding-only baseline.
- [x] Native-forecast-feature-only baseline.
- [x] Causal + embedding.
- [x] Causal + forecast features.
- [x] Residual-over-causal primary incremental-information test.
- [x] Low-variance linear ruler with target-independent train-fold PCA.
- [ ] **NOT FUNDED:** capacity-controlled nonlinear sensitivity; no route passed the frozen linear
  screen policy.
- [ ] Nested chronological calibration and trading-threshold selection; no route reached this gate.
- [ ] Identical concurrency, fees, zero-tick primary and frozen one-tick sensitivity; no route
  reached the trading ruler.
- [x] Paired root×calendar-week blocks and equal-root macro sampling for the primary one-minute
  screen. Broader per-timeframe/effective-sample-size reports remain blocked on authentic session
  authority.

## Phase 6 — family-native adaptation

- [x] Fail closed unless the family passed Phase 2, the available development data authority was
  externally receipt-bound, and its frozen common-information downstream ruler completed. No route
  passed the final downstream gate.
- [x] Replace flat `adaptation_routes` strings with hash-bound, route-specific training contracts;
  explicitly classify every route as upstream-native, native-derived, custom research-only or
  unsupported. A listed route is not an admission.
- [x] Version the training-evidence gate so prefix invariance, channel/group semantics,
  context/horizon bounds, scaling/masks, batch partitioning, corrupted/shuffled controls,
  sampler/next-batch continuity and deployment preprocessing parity cannot be bypassed by the four
  legacy training checks alone.
- [x] Match bounded pilot parent-window exposure and record exact schedules, stream counts and
  deployment/checkpoint bundles.
- [x] Run every implemented branch directly from pinned vanilla; the Kronos Mini predictor alone
  consumes the same-arm tokenizer pilot required by its catalog lineage.
- [x] Use only supported native objectives and trainable surfaces.
- [x] Compare every adapted arm with its own frozen/native parent at the pilot objective and common
  downstream ruler.
- [x] Do not promote from descriptive native-loss gains. All five pilot survivors failed the
  incremental-information gate and were stopped.
- [ ] Require three seeds for any promotable result. No route became promotable, so additional seeds
  were not funded.

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
