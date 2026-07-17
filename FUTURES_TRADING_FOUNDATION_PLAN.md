# Futures Trading Foundation-Model Execution Plan

Status: **active source of truth**
Frozen on: 2026-07-16
Scope: ES, NQ, RTY, YM, GC, SI, CL, ZB and ZN at 1/3/5/15/30/60-minute bars

This plan supersedes conflicting execution-order and OOS-language in older TODO documents. It does
not erase completed tournament evidence or authorize new full-model training. Training is funded
only after the downstream decision gate below shows that representation quality is the binding
constraint.

## 1. Trading objective

The project is not optimizing a generic forecasting leaderboard. Its purpose is to produce market
representations that improve executable futures decisions across a broad, sufficiently large and
non-redundant set of contexts.

The final question is:

> Relative to causal classical features and the vanilla backbone, does a representation improve
> out-of-sample trade selection, sizing or rejection after costs under the same execution rules?

Descriptive probe metrics remain diagnostics. They cannot promote a model without downstream
trading evidence.

## 2. Locked data policy

- Foundation training: `[2019-07-01, 2024-07-01)`.
- Development validation: `[2024-07-01, 2025-07-01)`.
- Legacy confirmatory interval: `[2025-07-01, 2026-07-01)`.
- The legacy confirmatory interval was not read by the sealed representation tournament, but dates
  in it were inspected during earlier project research. It is therefore **not globally pristine**.
- Truly pristine confirmation begins with data arriving after this plan and the finalist are frozen.
- No legacy-confirmatory or newly arriving data may be used for fitting, tuning, threshold selection,
  checkpoint selection or deciding objectives.
- Every artifact records source revision/patch, corpus and window fingerprints, split contract,
  configuration, seeds and parent checkpoint hashes.

## 3. Label and event contract

### 3.1 Dense decision rows

Build one eligible decision row per causal market context. Triggered rows are tags and sampling
strata, not separate copies of the same context.

Primary trigger tags:

- Prefix-invariant ATR-zigzag v2. The legacy detector walks completed future legs and can backfill
  earlier trigger metadata, so it is historical evidence only and is prohibited in new artifacts.
- Confirmed k=2 fractal.
- SuperTrend flip.

Fractal-zigzag is retained as metadata but not independently weighted because it is highly redundant
with k=2 fractals. ATR and structural stops are alternative policies attached to one row. HTF
agreement is causal metadata, not a duplicated event pool.

### 3.2 Context-only inputs

- Raw causal OHLCV context under each backbone's declared input contract.
- Causal classical features computed only through decision time.
- Symbol, timeframe, session clock and causal roll-state metadata where allowed by the declared
  arm. Forward distance to the realized continuous-contract roll is prohibited unless it comes
  from a schedule that was genuinely known at decision time.
- No future-derived target, scaler, threshold, event outcome or target statistic may enter a feature
  matrix.

### 3.3 Forward targets

At fixed wall-clock horizons, materialize:

- Terminal log return and ATR/R-normalized terminal move.
- Forward realized volatility.
- Upside MFE and downside MAE in causal ATR/R units.
- Forward trend efficiency.
- Continuation, reversal and termination labels relative to a causal context direction.
- Target-before-adverse state and time-to-barrier.
- Executable policy outcomes for the declared ATR and structural stops.

Primary horizons are 1, 3 and 6 elapsed hours. A label is masked, never shortened, if its horizon:

- crosses a contract roll;
- lacks endpoint coverage;
- contains an undeclared maintenance/session gap; or
- otherwise violates the stream's expected cadence contract.

### 3.4 Intrabar ambiguity

- Use tick ordering when an approved tick path is present.
- With OHLC bars, a bar touching both favorable and adverse barriers has unknown ordering.
- Store that representation target as `ambiguous`; never label it target-first.
- Score executable R conservatively as adverse-first.
- Time-to-barrier is bar-resolution/interval-censored when no tick path exists.
- MFE/MAE over an unstopped fixed horizon remain observable from high/low. Excursion-until-exit is
  ambiguous when the exit bar touches both sides and follows the same conservative policy.

## 4. Evaluation levels

### Level A: representation diagnostics

Report the following with fold dispersion and paired uncertainty:

| Target | Metric | Purpose |
|---|---|---|
| realized volatility | R2 | current volatility state |
| trend efficiency | R2 | trend versus chop |
| range expansion | R2 | compression/expansion state |
| forward absolute move | R2 | future movement magnitude |
| in-window direction | AUC | descriptive direction only |
| forward direction | AUC | unconditional future direction |
| forward realized volatility | R2 | predictable forward-volatility positive control |
| forward trend efficiency | R2 | future trend quality |
| continuation/termination | AUC/PR-AUC | conditional trend persistence |

Always report by timeframe. The 30/60-minute slices in the current one-year, non-overlapping
artifact are diagnostic rather than promotion-blocking because they are underpowered. Each
timeframe slice receives its own fold contract and cadence-appropriate embargo.

### Level B: downstream honest ruler

Use identical decision rows and chronological folds for:

1. Causal classical features with logistic regression.
2. Causal classical features with constrained XGBoost. XGBoost is already pinned and used by the
   project; adding LightGBM solely for this control would create avoidable environment drift.
3. Vanilla foundation embeddings with the same light heads.
4. Existing Stage 1, Stage 2 and Stage 3 embeddings with the same heads.
5. Embeddings plus causal classical features.
6. Shuffle, random-feature and time-destroyed controls.

All preprocessing, calibration and thresholds are fit on earlier folds only. Save row-level
predictions for paired comparisons.

### Level C: executable trading scorecard

For every arm report:

- realized R distribution and mean/median R after costs;
- win rate, profit factor and coverage;
- maximum drawdown and calendar stability;
- signal count and breadth by symbol, timeframe and trigger tag;
- calibration and selection curves;
- paired calendar-block confidence intervals versus causal features and vanilla;
- sensitivity to costs, thresholds and execution capacity.

Use one predeclared concurrency rule across all arms. The default is one active position per
symbol/strategy stream. If portfolio concurrency is studied, declare sizing, exposure and collision
rules before scoring. Never sum overlapping event outcomes as if they were independent trades.

## 5. Retraining decision gate

The frozen downstream comparison decides which case applies:

- **Case A — classical features match all embeddings:** do not fund broad retraining; simplify the
  system or use foundation models only as optional controls.
- **Case B — embeddings add value and the head is limiting:** improve calibration/head design
  without changing the backbone.
- **Case C — vanilla beats trained stages:** redesign or remove the damaging stages.
- **Case D — labels, event construction or execution dominate:** fix those contracts first.
- **Case E — representation quality is demonstrably binding:** authorize bounded revised-stage
  pilots.

Advancing to training requires paired evidence, not a small mean-AUC ranking. At minimum, an adapted
representation must add stable value over both its vanilla backbone and the causal-feature control
on conditional trend/path tasks or executable R, without material degradation of declared safety
metrics.

## 6. Revised training methodology: bounded MantisV2 research exception

The locked ruler did not establish Case E, so broad SSL funding remains blocked. The project owner
has nevertheless authorized one narrow MantisV2 experiment to determine whether a corrected
objective can beat the current adaptation recipe. This is a research exception, not a reversal of
the Case A/C/D diagnosis and not authorization for a cross-family sweep.

### Stage 0 — vanilla parity

Purpose: prove the data adapter, normalization, context length and exported embedding contract do
not damage the pretrained model.

- Compare native/frozen inference, training clean-input inference and exported inference.
- Require deterministic batch parity and exact artifact lineage.
- Evaluate 64/128/256-bar context parity before selecting deployment context.

### Stage 1 — structural reconstruction

Purpose: learn local and multi-scale futures price structure without teaching a trivial interpolation
shortcut.

- Contiguous patch/span masking, not independent point noise.
- Mask the same timestamps across all OHLCV channel passes.
- For jointly multivariate models, mask those timestamps jointly.
- For channel-independent models, reuse the same mask across separate channel encodes.
- Prefer reconstructing normalized returns, ranges, candle geometry, volume changes and patch
  statistics over independent absolute OHLC values.
- Stage 1 is optional: Stage 2-from-vanilla is a mandatory ablation.

### Stage 2 — market-state invariance

Purpose: group genuinely related views of the same market state without solving the task through
near-identical overlapping bars.

- Sample positives by elapsed time and nontrivial context crops.
- Draw augmentations independently per observation while preserving OHLC consistency.
- Do not systematically downweight the high-volatility states most relevant to trading.
- Control synchronized cross-symbol/timeframe false negatives.
- Compare exactly two bounded candidates initially: corrected contrastive learning and one
  non-negative-pair objective such as VICReg.
- Anchor to vanilla representations and require per-target non-inferiority.

### Stage 3 — conditional future path distribution

Purpose: estimate whether a current state is likely to produce useful continuation, reversal,
volatility and excursion—not reconstruct a point candle path.

Initial pilot core:

- MFE/MAE quantiles.
- Forward realized volatility.
- Continuation/termination.
- Stage 2 representation anchoring.

Add terminal return, tail, barrier-ordering and policy losses one at a time. Do not start with a
seven-loss mixture or broad hyperparameter search. Stage 3 is optional and must preserve Stage 2's
descriptive/conditional utility.

### Pilot and family policy

- MantisV2 is the only trainable pilot in this exception.
- Chronos Bolt and the other frozen tournament representations remain historical controls; they are
  not funded for new SSL training in this branch.
- Compare direct Stage 2 from vanilla at the 256-bar deployment context: `elapsed_time_v2` versus
  one negative-free `vicreg_v1` objective, on the same anchor universe and training budget.
- Keep the deployed preprocessing contract fixed during the objective comparison. Test
  normalization as a separate one-factor ablation so objective and input changes are attributable.
- Stage 1 is funded only if the direct Stage 2 comparison produces a viable arm. Stage 3 is funded
  only after a Stage 2 arm adds forward-path or downstream economic information over vanilla.
- Sundial is excluded from representation-stage work until non-finite hidden-state extraction is
  independently repaired; it may remain a forecast-only diagnostic.
- Do not extrapolate a MantisV2 result to another architecture without a new native-method audit.

## 7. Promotion and stopping rules

- One versioned promotion schema controls all reports.
- Use per-target non-inferiority limits; never average unrelated R2 and AUC deltas into a gate.
- Require multiple seeds for finalists and paired calendar-block uncertainty.
- Training loss is not comparable across model families.
- A failed stage is reported and is not silently used as a parent.
- A full tournament is funded only after bounded pilots beat vanilla and causal-feature controls.
- The legacy confirmatory interval is opened only for frozen finalists; newly arriving data is the
  final untouched confirmation.

## 8. Verification required before GPU training

- Source revision/patch and environment are reproducible.
- Full-state checkpoint resume reproduces the uninterrupted trajectory.
- Deployment, training clean-input and exported embeddings agree numerically.
- Dense labels pass synthetic path, no-lookahead, prefix-invariance, roll, gap and same-bar tests.
- Split purging uses each row's complete label-end timestamp.
- Cached/current embeddings and causal baselines have completed the downstream honest ruler.
- The Case A-E decision is recorded with row-level evidence.

## 9. Current execution status

Gate 2 materialization is complete for all 54 streams. The hash-bound collection manifest is
`output/foundation_tournament/event_contexts_v1/MANIFEST.json` and contains:

- 2,061,548 deduplicated decision contexts;
- 1,140,467 contexts carrying at least one primary/metadata trigger tag;
- 1,348,780 primary-trigger policy events linked back to context rows;
- ATR and structural policies across 1/3/6-hour horizons and 1R/2R/3R targets;
- 24,216,948 valid policy labels before downstream fold filtering;
- 54/54 nonempty symbol/timeframe shards.

The Gate 3 causal ruler and the original 48-arm frozen representation screen are complete. See
[DOWNSTREAM_GATE_BASELINE_RESULTS.md](DOWNSTREAM_GATE_BASELINE_RESULTS.md). It uses a sealed,
stream-balanced 64,800-row sample, six timeframe-specific purged fold contracts, linear and
constrained-XGBoost heads, three negative-control types, 6,550,592 saved row predictions and paired
weekly-block uncertainty. The sealed cross-family screen contains 48 representations from 12
trainable encoder arms; see [FOUNDATION_MODEL_RESULTS.md](FOUNDATION_MODEL_RESULTS.md).

The first 54-policy cost-aware screen remains diagnostic because it fit heads on only part of the
development year and used a raw zero threshold. The corrected full-history ruler is now complete;
see [DOWNSTREAM_FULL_HISTORY_RESULTS.md](DOWNSTREAM_FULL_HISTORY_RESULTS.md). It fits eligible
history from 2019 onward, confines outer tests to 2024-07 through 2025-07, uses nested train-only
isotonic calibration and applies a predeclared coverage/stability rule. No 2025-07 onward data was
read.

The ATR-zigzag structural pool produced positive fee-only aggregate results for causal XGBoost and
some representation arms, but no embedding added significant paired utility over causal-only. The
result was often concentrated by fold/timeframe. The fractal pool remained negative. Decomposed
barrier outcomes and honest residual embedding fusion are also complete; both changed selected-arm
economics but neither established incremental representation value over causal features. See
[DOWNSTREAM_FULL_HISTORY_RESULTS.md](DOWNSTREAM_FULL_HISTORY_RESULTS.md).

The locked diagnosis is Case A for the current ATR lane, Case C for inconsistent/damaging
adaptation and Case D for the fractal pool. Case E is rejected on current evidence, so broad SSL
retraining remains blocked. The separately authorized MantisV2-only direct-Stage-2 pilot is now
complete; see [MANTIS_V2_SSL_PILOT_RESULTS.md](MANTIS_V2_SSL_PILOT_RESULTS.md). Negative-free
VICReg improved the representation probes and beat its shuffle control, but did not beat causal
features on the sealed forward-path or full-history realized-R rulers. Concatenated and residual
fusion both failed, so Stage 1, Stage 3, a second seed and other trainable families remain unfunded.

The subsequent conditional-event gate is complete; see
[CONDITIONAL_EVENT_GATE_RESULTS.md](CONDITIONAL_EVENT_GATE_RESULTS.md). Causal, prefix-invariant
pullback-continuation and compression-breakout detectors were evaluated across all 54 streams.
Structural-stop pullbacks are the strongest current event pool (`+0.0511R`, PF 1.071 raw), while a
causal selector rescued the otherwise near-break-even structural compression pool. Vanilla
MantisV2 produced promising direct-head pullback results (`+0.0843R`, PF 1.124), but its paired
interval crossed zero and the lift disappeared under barrier decomposition. VICReg trailed
vanilla. The frozen next step is legacy confirmation of the declared pullback finalists; event,
threshold, objective and checkpoint tuning are prohibited before that read.
The read is blocked until every symbol/timeframe source reaches 2026-07-01. Current common coverage
ends in April 2026; a partial interval or NQ-only result is not an acceptable confirmation.
The matched development extension now includes MantisV1. Vanilla V1 produced the strongest
pullback point estimate (`+0.0961R`, PF 1.139), but its paired lift over causal crossed zero. A
matched V1 VICReg run passed generic representation and shuffle-control gates yet failed paired
trading promotion versus vanilla (`-0.00098` R/candidate; one of five folds positive) and damaged
compression. Therefore the existing VICReg recipe is not a promoted adaptation for either Mantis
version. Frozen V1+V2 late fusion has now failed: it produced `-0.0130R`, PF 0.982, trailed V1 by
`-0.01613` R per candidate and was positive in only one of five folds. Cross-version
feature/teacher distillation is not funded; retain specialized frozen backbones.
The author's next-leg atlas is treated as an external targeted hypothesis, not a result reproduced
by this repository: its exact script, checkpoint, report/hash and corpus manifest are required
before a frozen sealed evaluation. The primary execution sensitivity remains fee-only with zero
modeled tick slippage and no added delay; one or more ticks remain frozen-selection stress tests.
