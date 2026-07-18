# Conditional Trend-Event Gate Results

> **Historical adapter-contract warning:** the Mantis vectors and adapted checkpoints in this
> report were evaluated under the historical adapter contract. The findings remain valid for those
> exact artifacts, but do not constitute native Track-R evidence or admit custom pooling/fusion,
> classification/barrier training or deployment. New work is governed by
> [FOUNDATION_MODEL_NATIVE_CONTRACT_PLAN.md](FOUNDATION_MODEL_NATIVE_CONTRACT_PLAN.md).

## Decision

The causal pullback-continuation event family is promoted as a trading-research candidate. The
compression-breakout family remains a secondary structural-stop candidate. ATR-stop variants are
rejected. Vanilla MantisV1 and MantisV2 are promising but unconfirmed pullback selectors;
MantisV1 has the stronger point estimate. The matched `vicreg_v1` checkpoints are not promoted
because neither produced stable paired lift over its own vanilla backbone on this task.

Do not rerun this experiment with the same rows, objectives or checkpoint. The next admissible
step is a frozen confirmation of the declared finalists, not more development tuning.

This is development evidence, not untouched OOS evidence. The experiment artifacts read no row at
or after 2025-07-01. A later source-coverage audit inspected terminal timestamps (and printed two
terminal ES bars) but computed no OOS event, label, trade, threshold or model prediction.

## Versioned event definitions

Both events are confirmed at the decision-bar close and enter at the next bar open.

### Pullback continuation `ema20_50_reclaim_v1`

- Established trend is measured through bar `i-1`, never through the future path.
- EMA(20) must be on the trend side of EMA(50), and EMA(50) must slope in that direction.
- The trailing 64-bar close path must have directional efficiency at least 0.25 and net movement
  at least 1.5 causal ATR.
- Bar `i-1` must pull back to/beyond EMA(20) without materially invalidating EMA(50).
- Pullback depth over the prior eight bars must be between 0.25 and 2.5 ATR.
- Bar `i` must close back through EMA(20) in the established trend direction.
- The causal eight-bar pullback extreme defines the structural stop.

### Compression breakout `prior20_atr_bounded_close_break_v1`

- The setup range uses only the 20 bars ending at `i-1`.
- That range must be no wider than 4.0 causal ATR.
- Decision bar `i` must have true range at least 0.75 ATR.
- Its close must break the prior range and agree with the candle direction.
- The opposite causal 20-bar range extreme defines the structural stop.

The event thresholds were fixed before any realized-R result was inspected. Detection is reset at
contract boundaries. Prefix-invariance tests compare each full-series detector with multiple
truncated prefixes.

## Locked protocol

- Symbols: ES, NQ, RTY, YM, GC, SI, CL, ZB and ZN.
- Timeframes: 1/3/5/15/30/60 minute.
- Downstream history: `[2019-07-01, 2024-07-01)`.
- Outer development tests: `[2024-07-01, 2025-07-01)` in five calendar folds.
- Context: 256 OHLCV bars.
- Outcome: structural or 0.5-ATR stop, 360-minute horizon, 3R favorable barrier.
- Entry: next-bar open with zero additional delay.
- Primary slippage: zero round-trip ticks.
- Costs: instrument-specific declared round-trip cash fees.
- Same-bar ambiguity: adverse-first.
- Concurrency: one active trade per policy/ticker/timeframe.
- Thresholds: nested earlier-fold-only isotonic expected-net-R calibration with a positive-LCB
  operating rule.
- Fusion: causal features alone, causal plus vanilla MantisV2, and causal plus the MantisV2
  `vicreg_v1` checkpoint; residual fusion and barrier decomposition were declared sensitivity
  tests.

## Coverage and sealed artifacts

The 54-stream collection contains 11,669,560 eligible contexts. The new detectors produced:

| Timeframe | Pullback continuation | Compression breakout |
| --- | ---: | ---: |
| 1 minute | 9,792 | 180,178 |
| 3 minute | 5,308 | 92,848 |
| 5 minute | 4,037 | 70,573 |
| 15 minute | 3,006 | 30,228 |
| 30 minute | 1,366 | 16,909 |
| 60 minute | 492 | 3,378 |
| **Total** | **24,001** | **394,114** |

Both detectors are nonempty in all 54 symbol/timeframe streams.

The candidate sampler retains every sparse event until a per-tag/per-stream cap is reached and
thins common events with chronological midpoint quantiles. It produced 147,309 contexts. Stream
weights have equal total mass, so short timeframes cannot dominate only through row count.

Canonical hashes:

- Collection manifest: `78bdd7219abaf978b2854dc564237319963d5afa052857675de3a6f7f174554e`.
- Candidate sample: `9fcdc6966d4a084deac5ad25d2c2c0ae69d084f597f66fbc3afc71eba3f2879e`.
- All-candidate row contract: `5e848797d223578575db6d72e1385ccabca20f2ef5eea524af37c8892e400cb1`.
- Raw 256-bar contexts: `70bf5f1c2607cde441011ea13a398a2a1de9232cba9de0d9f3d2a4f8eb6597db`.
- Fee-only policy events: `5f12e385ea8863b4452387d3eb8c141ee3f145e06f1f62c49e4cada4c99f9678`.
- Vanilla MantisV2 embeddings: `d1819c8d908d9b3b2746986aeb15d5548a63715d78c61ef79cc5cf4dac57f2d3`.
- VICReg MantisV2 embeddings: `7b186d426ac998a091eab80fa63d41b4828a37a93e6c003f3265df678dea4d97`.
- VICReg checkpoint: `755e56ee4d7308218ad861b31f183a4b8e3b3c25279d522f5f39ef3aa3a60cf5`.
- Vanilla MantisV1 embeddings: `04bfc263225df176113548c39b43082b900650448a2f44e05303c6f8e2d65eef`.
- VICReg MantisV1 embeddings: `9c023207080981659b5352d92c76d04a31df3660bd02a923f91b2cbdab15a7de`.
- VICReg MantisV1 checkpoint: `d1f031d59e54e5c7b7c5c481d80f8ad46760804f20a39ca02ad48e2a5a0c9e03`.
- Frozen V1+V2 fusion embeddings: `4566c081fb8151cf9faae190749d245e849fcc6de5e8a3e3c964a6887e084b79`.

## Causal event-family results

These are aggregate outer-test results after instrument fees and concurrency.

| Event/policy | Arm | Trades | Mean net R | PF | WR | Total R | Breadth |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Pullback, structural | Raw pool | 3,434 | +0.0511 | 1.071 | 29.7% | +175.64 | 9 symbols / 6 TF |
| Pullback, structural | Causal direct-R | 335 | +0.0220 | 1.032 | 31.0% | +7.38 | 9 / 4 |
| Pullback, structural | Causal barrier | 419 | +0.1005 | 1.149 | 33.4% | +42.11 | 9 / 4 |
| Compression, structural | Raw pool | 15,949 | -0.0080 | 0.986 | 36.4% | -127.50 | 9 / 6 |
| Compression, structural | Causal direct-R | 259 | +0.0602 | 1.122 | 40.5% | +15.60 | 9 / 4 |
| Pullback, ATR stop | Raw pool | 4,197 | -0.0065 | 0.992 | 27.0% | -27.39 | 9 / 6 |
| Compression, ATR stop | Raw pool | 20,936 | -0.1368 | 0.834 | 23.7% | -2,863.57 | 9 / 6 |

The raw pullback structural pool was positive in four of five aggregate calendar folds. The
compression causal selector was positive in four of five aggregate folds, but it is sparse and
does not yet justify a production claim. ATR stops are the wrong risk geometry for these events.
Compression rows were capped chronologically per tag/stream, so its trade count is a balanced
screen count rather than an estimate of total annual venue activity. The cap does not change the
matched arm comparisons.

## MantisV1 and MantisV2 comparison

### Direct realized-R head with concatenated fusion

| Event | Arm | Trades | Mean net R | PF | Total R |
| --- | --- | ---: | ---: | ---: | ---: |
| Pullback | Causal | 335 | +0.0220 | 1.032 | +7.38 |
| Pullback | Vanilla MantisV2 | 652 | +0.0843 | 1.124 | +54.99 |
| Pullback | VICReg MantisV2 | 414 | +0.0362 | 1.055 | +15.00 |
| Compression | Causal | 259 | +0.0602 | 1.122 | +15.60 |
| Compression | Vanilla MantisV2 | 1,332 | -0.0575 | 0.915 | -76.65 |
| Compression | VICReg MantisV2 | 1,166 | -0.0064 | 0.991 | -7.48 |

Vanilla MantisV2 versus causal on pullbacks improved R per candidate by `+0.01045`, was positive in
four of five folds, and had a 95% weekly-block interval `[-0.00251, +0.02355]`. The interval crosses
zero and the FDR q-value is 0.288, so this is a promising hypothesis, not confirmed lift.

VICReg versus causal improved pullback R per candidate by only `+0.00167`, was positive in two of
five folds, and had interval `[-0.01239, +0.01609]`. VICReg also trailed vanilla by `-0.00878` R per
candidate. The SSL checkpoint is not promoted for this lane.

### Matched MantisV1 result

MantisV1 was extracted on the same 147,309-row contract and evaluated with identical heads, folds,
costs, threshold calibration and execution rules.

| Event | Arm | Trades | Mean net R | PF | Total R |
| --- | --- | ---: | ---: | ---: | ---: |
| Pullback | Vanilla MantisV1 | 686 | +0.0961 | 1.139 | +65.90 |
| Pullback | MantisV1 VICReg | 566 | +0.1085 | 1.162 | +61.42 |
| Compression | Vanilla MantisV1 | 426 | -0.0551 | 0.906 | -23.46 |
| Compression | MantisV1 VICReg | 487 | -0.1648 | 0.775 | -80.28 |

Vanilla MantisV1 had the best pullback point estimate of the two vanilla backbones. Its paired lift
over causal was `+0.01285` R per candidate with 95% weekly-block interval
`[-0.00279, +0.02807]`; it was positive in three of five folds and was not significant after the
declared comparison correction.

The V1 VICReg checkpoint passed its generic representation/control gate: mean-core delta was
`+0.1316` versus `+0.0397` for the matched shuffle control, and forward-direction AUC delta was
`+0.0067`. That did not become stable economic lift. Against vanilla V1, VICReg changed R per
candidate by `-0.00098`, with interval `[-0.01740, +0.01527]`, and beat vanilla in only one of five
folds. The higher selected-trade mean R came from a smaller selection and produced less total R.
It also materially damaged compression selection. Generic probe gains are not sufficient for
trading promotion.

### Sensitivity tests

- Residual embedding fusion failed: vanilla pullback residual PF was 1.004 and VICReg PF was 0.950;
  both compression residual arms were negative.
- Barrier decomposition improved the causal pullback selector to `+0.1005R` and PF 1.149.
- Under barrier decomposition, vanilla embedding lift over causal was only `+0.00238` R per
  candidate with interval `[-0.01778, +0.02476]`; VICReg lift was negative.

The direct-head vanilla improvement is therefore head-sensitive. It cannot yet be attributed to a
robust foundation representation advantage.

## Findings that must not be rediscovered

1. Pullback continuation with a structural stop is the strongest current event pool.
2. Compression breakout needs causal selection and structural risk; its unconditional and ATR-stop
   forms are not viable.
3. Vanilla MantisV2 contains potentially useful conditional pullback information.
4. The current VICReg adaptation improves generic representation probes but damages or fails to
   improve the valuable conditional trading task.
5. Residual fusion is not the answer under the current head implementation.
6. Barrier decomposition improves the causal head but removes the apparent Mantis advantage.
7. No result in this document is untouched OOS or production evidence.
8. MantisV1 is the strongest current vanilla pullback representation by point estimate, but its
   advantage is not statistically established.
9. The current direct VICReg objective is not a promoted adaptation for either Mantis version.
10. Frozen V1+V2 late fusion must beat the best single backbone before any cross-version feature or
    teacher distillation is funded. It did not, so cross-version distillation is not funded.

### Cross-version transfer gate

The required frozen late-fusion control concatenated aligned V1 and V2 embeddings before fold-local
scaling/PCA. It used a hash-bound 147,309-row artifact and the same pullback ruler.

| Arm | Trades | Mean net R | PF | Total R |
| --- | ---: | ---: | ---: | ---: |
| Vanilla MantisV1 | 686 | +0.0961 | 1.139 | +65.90 |
| Vanilla MantisV2 | 652 | +0.0843 | 1.124 | +54.99 |
| Frozen V1+V2 feature fusion | 582 | -0.0130 | 0.982 | -7.54 |

Fusion versus V1 changed R per candidate by `-0.01613`, with 95% weekly-block interval
`[-0.03548, +0.00283]`, and was positive in only one of five folds. The current representations do
not demonstrate complementary value under this head. Feature/teacher distillation is rejected for
the present branch; it would add complexity without an empirical transfer target.

## Artifact locations

- Event collection: `output/foundation_tournament/event_contexts_conditional_v2/`.
- Candidate sample and row contract: `output/foundation_tournament/conditional_event_gate_v2/`.
- Causal/ATR screen: `conditional_event_gate_v2/causal_nested_isotonic/`.
- Concatenated Mantis comparison: `conditional_event_gate_v2/mantis_nested_isotonic_structural/`.
- Residual comparison: `conditional_event_gate_v2/mantis_residual_structural/`.
- Barrier-decomposed pullback comparison:
  `conditional_event_gate_v2/mantis_barrier_decomposed_pullback/`.
- Matched V1/V2 vanilla comparison:
  `conditional_event_gate_v2/mantis_v1_v2_vanilla_direct/`.
- MantisV1 VICReg comparison:
  `conditional_event_gate_v2/mantis_v1_stage2_vs_vanilla_pullback/`.
- Frozen V1+V2 late-fusion comparison:
  `conditional_event_gate_v2/mantis_v1_v2_late_fusion_pullback/`.

## Reproduction commands

These commands describe the canonical run. They are recorded for auditability, not as a request
to spend compute rerunning completed cells.

```bash
.venv/bin/python scripts/materialize_event_contexts.py \
  --output-dir output/foundation_tournament/event_contexts_conditional_v2 \
  --tickers ES,NQ,RTY,YM,GC,SI,CL,ZB,ZN \
  --timeframes 1min,3min,5min,15min,30min,60min \
  --eval-start 2019-07-01 --eval-end 2025-07-01 --warmup-days 60

.venv/bin/python scripts/build_downstream_sample.py \
  --collection output/foundation_tournament/event_contexts_conditional_v2/MANIFEST.json \
  --output output/foundation_tournament/conditional_event_gate_v2/candidate_sample.npz \
  --rows-per-stream 3000 \
  --event-tags pullback_continuation,compression_breakout

.venv/bin/python scripts/build_downstream_representation_rows.py \
  --sample output/foundation_tournament/conditional_event_gate_v2/candidate_sample.npz \
  --output output/foundation_tournament/conditional_event_gate_v2/all_candidate_rows.npz \
  --all-rows

.venv/bin/python scripts/build_downstream_policy_events.py \
  --sample output/foundation_tournament/conditional_event_gate_v2/candidate_sample.npz \
  --row-selection output/foundation_tournament/conditional_event_gate_v2/all_candidate_rows.npz \
  --output output/foundation_tournament/conditional_event_gate_v2/policy_events_fees_only.npz \
  --slippage-ticks 0
```

The representation extraction and three trading report directories bind their exact row,
checkpoint, sample and policy hashes in their manifests. Use those manifests rather than an
unversioned checkpoint filename when auditing or extending the comparison.

## Frozen next decision

Freeze these development finalists without further threshold or event-definition changes:

1. Raw pullback-continuation structural 360m/3R.
2. Causal barrier-decomposed pullback selector.
3. Vanilla MantisV1 direct-R pullback fusion as the lead, explicitly unconfirmed representation
   arm.
4. Vanilla MantisV2 direct-R pullback fusion as the second unconfirmed representation arm.
5. Both VICReg direct-R pullback fusions as negative adaptation controls.

The next evaluation is the legacy 2025-07 to 2026-07 confirmation interval, carrying its known
prior-inspection caveat. It is currently blocked by incomplete common source coverage: GC, SI and
CL end on 2026-04-13; ES, RTY, YM, ZB and ZN end on 2026-05-04; only NQ reaches 2026-07-10. Do not
run a partial-symbol or partial-period substitute. Supply hash-bound data through at least
2026-07-01 for all nine symbols and all six timeframes, then execute the frozen arms exactly
once. A truly untouched verdict requires subsequently arriving data. Do not train Stage 3 or tune
the pullback definition as part of the frozen confirmation. Separately versioned objective
research may continue on the development interval, but it cannot replace or alter these frozen
finalists.
