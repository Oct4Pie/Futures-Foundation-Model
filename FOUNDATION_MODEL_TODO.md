# Foundation-model tournament TODO

> **Historical tournament record.** This file preserves the work that was completed under the
> original tournament protocol. The active execution order and corrected evidence-boundary policy
> live in [FUTURES_TRADING_FOUNDATION_PLAN.md](FUTURES_TRADING_FOUNDATION_PLAN.md) and
> [FUTURES_TRADING_TASKS.md](FUTURES_TRADING_TASKS.md). Do not use unchecked items below to bypass
> the frozen downstream decision gate.

## Locked protocol

- [x] Training interval: `[2019-07-01, 2024-07-01)`
- [x] Validation interval: `[2024-07-01, 2025-07-01)`
- [x] Legacy confirmatory interval: `[2025-07-01, 2026-07-01)`; not read by the sealed
  representation tournament, but not globally pristine because earlier project research inspected
  dates in this interval
- [x] Corpus: ES, NQ, RTY, YM, GC, SI, CL, ZB, ZN
- [x] Timeframes: 1/3/5/15/30/60 minutes
- [x] Inputs: causal OHLCV from all 54 symbol/timeframe streams
- [x] Final exposure: 262,144 balanced training examples per arm
- [x] Rough tuning: reuse completed studies; do not launch another broad Optuna search
- [x] Final training seed: 8400; validation schedule seed: 5400
- [ ] Require exact checkpoint/resume state, source archive, corpus hash, and validation fingerprint

## Required three-stage contract

Every trainable arm must complete all three stages in order. A native forecast-only checkpoint is
only a preliminary Stage 3 control and is not a finalist.

- [ ] Stage 1: native reconstruction/tokenizer/domain pretraining
- [ ] Stage 2: contrastive market-regime representation training with backbone gradients
- [ ] Stage 3: native causal forecasting refinement warm-started from the Stage 2 backbone
- [ ] Verify the parent checkpoint hash at both stage transitions
- [ ] Verify that each stage changes backbone weights, not only a disposable head

## Trainable arms

- [ ] Kronos-mini (`NeoQuasar/Kronos-mini`)
  - [x] Rough native tuning completed
  - [x] Stage 1 tokenizer/reconstruction (`final_staged/kronos_mini/stage1.pt`)
  - [x] Stage 2 contrastive representation (`final_staged/kronos_mini/stage2.pt`)
  - [x] Preliminary Stage 3-only control (`final_5y1y1y/kronos_mini`, seed 8400)
  - [x] Final Stage 3 warm-started from Stage 2 (`final_staged/kronos_mini/stage3.pt`)
  - [x] Parent hashes verified and predictor changed at both transitions
  - [x] Shared validation forecast (`final_staged/kronos_mini/validation_scorecard.json`)
  - [x] Stage 3 promotion failed: macro path skill −13.3151; valid-candle fraction 0.8661
  - [ ] Shared Stage 1/2 representation probe
- [ ] Kronos-small (`NeoQuasar/Kronos-small`)
  - [x] Rough native tuning completed
  - [x] Stage 1 tokenizer/reconstruction (`final_staged/kronos_small/stage1.pt`)
  - [x] Stage 2 contrastive representation (`final_staged/kronos_small/stage2.pt`)
  - [x] Final Stage 3 warm-started from Stage 2 (`final_staged/kronos_small/stage3.pt`)
  - [x] Parent hashes verified and predictor changed at both transitions
  - [x] Shared validation forecast (`final_staged/kronos_small/validation_scorecard.json`)
  - [x] Stage 3 promotion failed: macro path skill −0.1713; direction AUC 0.4907
  - [ ] Shared Stage 1/2 representation probe
- [ ] MOMENT-small (`AutonLab/MOMENT-1-small`)
  - [x] Rough masked-reconstruction tuning completed
  - [x] Stage 1 masked reconstruction (`final_staged/moment/stage1.pt`)
  - [x] Stage 2 contrastive representation (`final_staged/moment/stage2.pt`)
  - [x] Stage 3 causal forecasting (`final_staged/moment/stage3.pt`)
  - [x] Parent hashes verified; 76/78 backbone tensors changed at both transitions
  - [x] Shared representation probe (`final_staged/moment/representation_probe.json`)
  - [x] Stage 2 promotion failed: mean core Δ −0.0902; forward score −0.4049
  - [x] Shared validation forecast (`final_staged/moment/validation_scorecard.json`)
  - [x] Stage 3 promotion failed: macro path skill −0.0511; valid candles 0.7867
- [ ] MantisV1 (`paris-noah/Mantis-8M`)
  - [x] Rough contrastive tuning completed
  - [x] Stage 1 masked reconstruction attempted (`final_staged/mantis_v1/stage1.pt`)
  - [x] Stage 1 promotion failed; do not use it as a parent
  - [ ] Stage 2 contrastive representation (blocked for the strict chain; direct ablation exists)
  - [ ] Stage 3 causal forecasting (blocked for the strict chain)
  - [ ] Shared representation probe
- [ ] MantisV2 (`paris-noah/MantisV2`)
  - [x] Stage 1/2/3 tuning and controls completed
  - [x] Final-seed Stage 1 attempted (`final_staged/mantis_v2/stage1.pt`)
  - [x] Stage 1 promotion failed; do not use it as a parent
  - [ ] Strict Stage 2/3 chain blocked; direct Stage 2 and Stage 3 remain ablations
  - [ ] Final baseline is vanilla unless an adapted checkpoint passes independent confirmation
  - [ ] Shared representation probe
- [ ] Chronos V1 (`amazon/chronos-t5-tiny`)
  - [x] Rough native forecast tuning completed
  - [x] Stage 1 token reconstruction (`final_staged/chronos_v1/stage1.pt`)
  - [x] Stage 2 contrastive representation (`final_staged/chronos_v1/stage2.pt`)
  - [x] Final Stage 3 warm-started from Stage 2 (`final_staged/chronos_v1/stage3.pt`)
  - [x] Parent hashes verified; all 35 encoder tensors changed at both transitions
  - [x] Shared validation forecast (`final_staged/chronos_v1/validation_scorecard.json`)
  - [x] Stage 3 promotion failed: macro path skill −3.0469; direction AUC 0.4975
  - [ ] Shared Stage 1/2 representation probe
- [ ] Chronos Bolt (`amazon/chronos-bolt-tiny`)
  - [x] Rough native forecast tuning completed
  - [x] Stage 1 patch reconstruction (`final_staged/chronos_bolt/stage1.pt`)
  - [x] Stage 2 contrastive representation (`final_staged/chronos_bolt/stage2.pt`)
  - [x] Final Stage 3 warm-started from Stage 2 (`final_staged/chronos_bolt/stage3.pt`)
  - [x] Parent hashes verified; 43/103 shared tensors changed in Stage 1→2 and 103/103 in Stage 2→3
  - [x] Shared validation forecast (`final_staged/chronos_bolt/validation_scorecard.json`)
  - [x] Stage 3 promotion failed: macro path skill −1.0393; direction AUC 0.4967
  - [ ] Shared Stage 1/2 representation probe
- [ ] Chronos V2 (`autogluon/chronos-2-small`)
  - [x] Rough native multivariate forecast tuning completed
  - [x] Stage 1 patch reconstruction (`final_staged/chronos_v2/stage1.pt`)
  - [x] Stage 2 contrastive representation (`final_staged/chronos_v2/stage2.pt`)
  - [x] Final Stage 3 warm-started from Stage 2 (`final_staged/chronos_v2/stage3.pt`)
  - [x] Parent hashes verified; 86/92 shared tensors changed in Stage 1→2 and 92/92 in Stage 2→3
  - [x] Shared validation forecast (`final_staged/chronos_v2/validation_scorecard.json`)
  - [x] Stage 3 promotion failed: macro path skill −1.0317; direction AUC 0.5101
  - [ ] Shared Stage 1/2 representation probe
- [ ] TimesFM 2.5 (`google/timesfm-2.5-200m-transformers`)
  - [x] Rough LoRA tuning completed
  - [x] Stage 1 causal historical-patch reconstruction (`final_staged/timesfm25/stage1.pt`)
  - [x] Stage 2 contrastive representation (`final_staged/timesfm25/stage2.pt`)
  - [x] Final Stage 3 LoRA warm-started from Stage 2 (`final_staged/timesfm25/stage3.pt`)
  - [x] Parent hashes verified; 246/258 LoRA tensors changed in Stage 1→2 and 258/258 in Stage 2→3
  - [x] Shared validation forecast (`final_staged/timesfm25/validation_scorecard.json`)
  - [x] Stage 3 promotion failed: macro path skill −0.00045; direction AUC 0.5052; valid candles 0.7470
  - [ ] Shared Stage 1/2 representation probe
- [ ] TTM-R2 (`ibm-granite/granite-timeseries-ttm-r2`)
  - [x] Rough native tuning completed
  - [x] Stage 1 suffix-masked reconstruction (`final_staged/ttm_r2/stage1.pt`)
  - [x] Stage 2 contrastive representation (`final_staged/ttm_r2/stage2.pt`)
  - [x] Preliminary Stage 3-only control (`final_5y1y1y/ttm_r2`, seed 8400)
  - [x] Final Stage 3 warm-started from Stage 2 (`final_staged/ttm_r2/stage3.pt`)
  - [x] Parent hashes verified; all 103 backbone tensors changed at both transitions
  - [x] Shared validation forecast (`final_staged/ttm_r2/validation_scorecard.json`)
  - [x] Stage 3 promotion failed: macro path skill −0.0315; direction AUC 0.4983; valid candles 0.8442
  - [ ] Shared Stage 1/2 representation probe
- [ ] Moirai-2 small (`Salesforce/moirai-2.0-R-small`)
  - [x] Rough native tuning completed
  - [x] Stage 1 causal historical-patch reconstruction (`final_staged/moirai2_small/stage1.pt`)
  - [x] Stage 2 contrastive representation (`final_staged/moirai2_small/stage2.pt`)
  - [x] Preliminary Stage 3-only control (`final_5y1y1y/moirai2_small`, seed 8400)
  - [x] Final Stage 3 warm-started from Stage 2 (`final_staged/moirai2_small/stage3.pt`)
  - [x] Parent hashes verified; all 73 encoder tensors changed at both transitions
  - [x] Shared validation forecast (`final_staged/moirai2_small/validation_scorecard.json`)
  - [x] Stage 3 promotion failed: macro path skill −0.0165; direction AUC 0.5016
  - [ ] Shared Stage 1/2 representation probe

## Control/integration arms

- [ ] Toto 2.0 22M (`Datadog/Toto-2.0-22m`)
  - [x] Frozen shared-validation forecast exists
  - [ ] Implement and verify a native training adapter before calling it trained
- [ ] Sundial base (`thuml/sundial-base-128m`)
  - [x] Frozen shared-validation forecast exists
  - [ ] Implement and verify a native training adapter before calling it trained
- [ ] TabPFN-TS (`PriorLabs/TabPFN-TS-3`)
  - [ ] Run as an in-context/downstream control; it has no comparable backbone-training phase

## Final comparison

- [ ] Confirm every finalist uses the same corpus and validation-window fingerprints
- [ ] Run a second seed/control check for promoted finalists
- [ ] Freeze the selected checkpoints and thresholds before OOS
- [ ] Adapt and run the fractal-pivot one-shot benchmark for every compatible model
- [ ] Adapt and run the SuperTrend walk-forward/produce benchmark for every compatible model
- [ ] Read the 2025-07-01 to 2026-07-01 legacy confirmatory interval once for a frozen finalist and
  report the prior-inspection caveat
- [ ] Produce one forecast, representation, and strategy scorecard with training cost

## Non-negotiable failure rules

- A frozen checkpoint is never labeled as trained.
- A model that fails its independent-seed/control check is reported, not promoted.
- Native validation losses from different families are not compared directly.
- No model gets access to OOS during tuning, checkpoint selection, or threshold selection.
