# Native Training Pathway Audit

## Decision

No model is training-admitted yet. The historical universal `Stage 1 → Stage 2 → Stage 3`
curriculum is not a valid cross-family methodology and is excluded from native ranking.

Apples-to-apples comparison means the same sealed rows, chronological splits, context/horizon
contract, causal preprocessing boundary, costs and scorecards. It does **not** mean forcing models
with different pretraining tasks through the same objectives.

This audit is source- and architecture-bound. It does not authorize data access or training.

## Current implementation gap

The executable native registry currently proves narrowly scoped forecast/representation parity.
Its `adaptation_routes` arrays are only candidate route names; they are **not** training
admissions. Some names still describe historical custom work (for example adjacent-half
contrastive branches) and must not be presented as native routes merely because they are listed.

Before any training report can be admitted, the registry must replace that flat route vocabulary
with separately hash-bound route records containing the exact native/custom disposition, input and
target contract, loss, trainable parameter surface, checkpoint/resume bundle, deployment output,
and applicable falsification tests. Unsupported upstream routes remain excluded rather than being
filled with a common three-stage recipe.

The current executable training gate also treats only gradient/freeze behavior, repeated-batch
loss decrease, exact resume and save/reload/export as mandatory. That is insufficient. A versioned
training-evidence schema must additionally make causal prefix invariance, channel/group semantics,
context/horizon boundaries, scaling and mask behavior, batch-partition parity, corrupted-target and
shuffled-label controls, sampler/next-batch continuity and deployment preprocessing parity mandatory
where applicable. Existing inference evidence cannot satisfy those route-specific checks.

## Evidence-bound route matrix

| Arm | Correct candidate route | Native OHLCV semantics | Training disposition |
| --- | --- | --- | --- |
| MantisV1 | Official crop/resize two-view InfoNCE; supervised classification is a separate branch | Raw univariate channel passes through shared weights; concatenate only at the declared downstream seam | Candidate, blocked pending route tests |
| MantisV2 | Official compatible-output crop/resize InfoNCE; deployment uses layer-2 CLS+mean; classification separate | Same independent-channel contract; input length divisible by 32 and official interpolation policy | Candidate, blocked pending route tests |
| MOMENT Small | Continued masked-patch reconstruction, classification and forecast are three separate native branches | API accepts `[B,C,512]`, but the encoder folds to `B*C`; native RevIN and valid-timestamp mask | Candidates, blocked pending branch-specific tests |
| Kronos Mini | Tokenizer reconstruction/BSQ → hierarchical autoregressive predictor | Joint ordered OHLCVA plus timestamps; Mini requires Tokenizer-2k | Candidate, blocked pending route tests |
| Kronos Small | Tokenizer reconstruction/BSQ → hierarchical autoregressive predictor | Joint ordered OHLCVA plus timestamps; Small requires Tokenizer-base | Candidate, blocked pending route tests |
| Chronos v1 | Direct T5 token-forecast cross-entropy | Five independent univariate channel passes | Candidate native-derived manual loop; blocked |
| Chronos Bolt | Direct native quantile/pinball forecast tuning | Five independent univariate channel passes | Candidate native-derived manual loop; blocked |
| Chronos-2 | Direct official grouped-multivariate quantile `full` or `lora` tuning | Joint OHLCV with shared group IDs | Candidate, blocked pending fail-closed LoRA and resume tests |
| TimesFM 2.5 | Direct official 512-context LoRA forecast tuning | Raw channel-independent passes; internal RevIN | Candidate, blocked pending FP32 route tests |
| TTM-R2 | Direct raw-value forecast fine-tuning with official HF Trainer | `[B,512,C]`, shared channel-independent weights, cross-channel mixer off | Candidate, blocked pending route tests |
| Moirai-2 Small | No official pinned Moirai-2 trainer; possible custom scaled-pinball research branch | Packed joint multivariate input with sample/time/variate IDs | Official training blocked; custom noncommercial research candidate only |
| Toto-2 22M | No released Toto-2 fine-tuning pathway | Joint multivariate; all OHLCV variates must share the semantic group | Training excluded |
| Sundial Base | No released fine-tuning pathway | Univariate/channel-independent, internal ReVIN, stochastic TimeFlow forecast | Training and hidden-state representation excluded |
| TabPFN-TS | Nested support/query in-context downstream control only | Support-set task, not persistent SSL pretraining | Fully blocked pending terms/checkpoint/hash/parity |

## Proven upstream constraints

### Mantis

- Native pretraining is univariate crop/resize contrastive learning.
- Multivariate series are encoded channel by channel.
- Raw patch mean and standard deviation are part of the pretrained representation; external
  per-window z-scoring changes the native input contract and is only a separately named
  sensitivity.
- Official checkpoints are encoder weights, not exact trajectory-resume state.
- Historical mask, candle-forecast, path, cross-channel adapter and universal stage-chain
  checkpoints are custom and cannot inherit native status.

### MOMENT

- Masked reconstruction, classification and forecasting have distinct task heads and objectives.
- Native RevIN operates per sample/channel over valid timestamps.
- Adjacent-half contrastive pooling is custom.
- The released lightweight repository does not by itself reproduce full upstream pretraining;
  continued reconstruction must still receive its own parity evidence.

### Kronos

- The supported sequence is tokenizer reconstruction/BSQ followed by a frozen-tokenizer
  autoregressive predictor.
- Context-only normalization is required for causal FFM training. Upstream CSV code that
  normalizes context plus future cannot be copied.
- The intervening hidden-state contrastive stage is foreign to the native route.
- Upstream best-weight saves do not provide exact interruption/resume trajectory state.

### Chronos family

- Chronos v1 uses mean-scale tokenization and T5 sequence cross-entropy. The public label contract
  is 64 steps; the historical private 16-step tokenizer route is custom unless separately proven.
- Chronos Bolt uses internal normalization and native short-horizon quantile/pinball loss.
- Chronos-2 is genuinely grouped multivariate. `cross_learning=False` does not disable interaction
  among variates within one OHLCV group.
- Chronos-2 official `fit()` does not provide exact resume and may silently fall back from LoRA to
  full tuning if PEFT is unavailable; FFM must fail closed instead.
- Package 2.3.1 exposes a public Chronos v1 `embed()` API. This supports a representation surface;
  it does not make Chronos a native classifier.

### TimesFM and TTM

- TimesFM uses direct forecast tuning, internal RevIN and an official LoRA example. Historical
  Stage 1 and Stage 3 duplicate the forecast task at different contexts; Stage 2 is custom.
- TTM must receive raw values with its native scaler. External context z-scoring changes the loss.
- TTM's historical zero-filled observed suffix is not a mask and is invalid as native training.
- TTM timeframe prefixes are `1/0/3/5/6/7` for `1/3/5/15/30/60` minutes; 3-minute is deliberately
  the official OOV value `0`.

### Moirai-2, Toto-2, Sundial and TabPFN

- Pinned Uni2TS contains Moirai-2 inference modules but its official training configurations target
  Moirai 1.x. A differentiable pinball seam is not an official training pathway.
- Toto-2 upstream explicitly states fine-tuning is planned. Toto 1 training code does not admit
  Toto-2. Historical distinct `series_ids` for each OHLCV channel disabled the intended joint
  variate attention and is invalid.
- Sundial upstream states fine-tuning code is forthcoming. A forward loss surface does not define
  the missing data, mask, optimizer, validation and checkpoint contract.
- TabPFN fold containment must eventually purge by complete support-label end and parent interval,
  not only row timestamps. Its separate terms and missing checkpoint remain hard blockers.

## Required evidence before any route becomes training-admitted

Each route requires its own machine-readable technical profile and two independent approvals.
Evidence is not transferable between a family's forecast, reconstruction, representation,
classification or custom trading-objective branches.

1. Exact source/model/weight/tokenizer/package/environment hashes.
2. Upstream-versus-wrapper input, target, mask, grouping, scaling, loss and output parity.
3. Prefix invariance: mutating future data cannot change context tensors, scaling, tokens or masks.
4. Exact OHLCV/OHLCVA order and independent-versus-joint channel perturbation behavior.
5. Declared parameter surface: intended finite nonzero gradients; frozen parameters unchanged.
6. Repeated-batch loss decrease plus shuffled-label/corrupted-target and constant-input controls.
7. Batch-size and batch-partition parity without sample-dependent stochastic leakage.
8. Exact interruption/resume equality for model/head/adapter/projector, optimizer, scheduler, AMP
   scaler, epoch/step, best state, all RNG streams, sampler, next batch, next loss and history.
9. Resume binding to the exact corpus manifest, row schedule and preprocessing fingerprints.
10. Save/reload and deployment-output parity at the declared 512-bar common context.
11. FP32 reference admission before any reduced-precision profile.
12. No validation, calibration, selection or reserved-OOS access during route qualification.

## Model-specific falsification tests

- Mantis: official view sampling/loss parity; V2 projector/output compatibility; deployment
  layer-2 CLS+mean parity.
- MOMENT: reconstruction mask and valid-position loss; separately initialized task-head parity.
- Kronos: tokenizer pairing, BSQ loss, frozen tokenizer during predictor training, forecast
  save/reload of both components.
- Chronos v1: EOS and label-mask contract.
- Chronos Bolt: short-horizon mask and pinball formula.
- Chronos-2: group IDs, exact LoRA target modules, hard failure when PEFT is absent.
- TimesFM: explicit flip-invariance/truncation settings and FP32 first.
- TTM: selector/config completeness, `scaling=False`, frequency derivation and mixer state.
- Any custom Moirai-2 route: quantile/mask loss audit across all output patches before training.
- TabPFN: label-end purge, support-only preprocessing and no support/query parent overlap.

## Frozen conclusion

The earlier model tournament can still be retained as historical evidence about the exact custom
pipelines that produced it. It cannot rank the families' native trainability or answer how their
proper native adaptation would perform.

Full training begins only after the common data authorization chain is admitted and each funded
route independently passes the tests above. Unsupported models remain forecast controls; they are
not assigned invented stages merely to fill a table.
