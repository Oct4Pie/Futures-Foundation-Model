# MantisV2 256-Bar SSL Pilot Results

## Decision

The bounded MantisV2 SSL experiment is complete. `vicreg_v1` materially improved the frozen
representation over vanilla MantisV2 and over the matched shuffle control, but it did not add
stable forward-path or realized-R utility over causal features. Do not fund a second seed, revised
Stage 1, or Stage 3 from this checkpoint under the locked plan.

This is a development result, not OOS evidence. No row at or after 2025-07-01 was read.

## Locked protocol

- Backbone: `paris-noah/MantisV2`.
- Context: 256 OHLCV bars; channels encoded independently and concatenated.
- Timeframes: 1/3/5/15/30/60 minute.
- Symbols: ES, NQ, RTY, YM, GC, SI, CL, ZB and ZN.
- SSL training: `[2019-07-01, 2024-07-01)`.
- Development evaluation: `[2024-07-01, 2025-07-01)`.
- Holdout boundary: 2025-07-01; physically excluded by the runner.
- Preprocessing: `per_window_per_channel_zscore_v1`, held fixed across objectives.
- Lineage: direct Stage 2 from vanilla, no Stage 1 parent.
- Budget: seed 17, five epochs, 50 steps/epoch, batch 32.
- Anti-forgetting: cosine feature anchor to the vanilla entry encoder, weight 0.1.
- Objective comparison: `elapsed_time_v2` versus negative-free `vicreg_v1`.
- Negative control: identically trained time-shuffled input.
- Source and corpus hashes are recorded in `comparison.json`.

The assembled corpus contained 30,840,372 bars, 24,960,930 eligible training windows and
4,946,676 eligible development windows across 54 streams.

## Stage 0 parity

Stage 0 passed on eight real rows from the sealed representation artifact with SHA-256
`646b1dbeb0d45d074a1f7bd14f8f751488530630c36b59e47309206387158716`.

- Training-clean, legacy frozen Python and versioned bundle embeddings were identical at
  64/128/256 bars.
- Same-batch repeated CUDA extraction was identical.
- CPU batch-versus-singleton extraction was identical.
- CUDA batch-shape sensitivity was bounded at 0.00803 maximum absolute difference at 256 bars.
- Python-bundle versus ONNX maximum absolute difference was `5.72e-6`.

The CUDA batch-shape result is floating-point kernel sensitivity, not nondeterminism. Deployment
must retain a declared extraction batch contract; CPU/ONNX is the deterministic reference.

## Direct Stage 2 representation probe

All values are adapted-minus-vanilla deltas on a five-fold expanding calendar walk-forward probe.
Unconditional direction remains report-only under promotion schema `ffm_ssl_promotion_v3`.

| Objective | vol R2 | trend_eff R2 | range_expand R2 | fwd_absmove R2 | direction AUC | fwd_dir AUC | Mean core |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| elapsed-time | +0.1102 | +0.1167 | +0.0384 | +0.0549 | +0.0663 | -0.0144 | +0.0801 |
| elapsed shuffle | +0.0872 | -0.0017 | +0.0451 | +0.0937 | +0.0211 | -0.0048 | +0.0561 |
| VICReg | +0.1714 | +0.1650 | +0.2190 | +0.2444 | +0.0506 | -0.0085 | +0.1999 |
| VICReg shuffle | +0.1019 | +0.0842 | +0.0582 | +0.1481 | +0.0291 | -0.0119 | +0.0981 |

Elapsed-time failed the matched-control rule: real training did not beat shuffle on range expansion
or forward absolute move. VICReg beat shuffle on every reported target and passed the corrected
representation gate. Its checkpoint SHA-256 is
`755e56ee4d7308218ad861b31f183a4b8e3b3c25279d522f5f39ef3aa3a60cf5`.

The shared five-fold calendar ruler excludes all nine 60-minute streams from promotion scoring:
after non-overlap and the maximum-duration embargo they cannot populate every fold. This exclusion
is determined only from timestamps and is recorded in each report. The 60-minute results remain
diagnostic.

## Economically aligned forward-path screen

The VICReg checkpoint was extracted on the sealed 21,600-row Gate-3 screen and compared using the
same folds, labels and XGBoost settings as vanilla and causal controls. Across 405 non-coarse
timeframe/target cells:

| Comparison | Mean metric delta | Median delta | Positive-cell fraction |
| --- | ---: | ---: | ---: |
| VICReg embedding vs vanilla embedding | +0.00544 | +0.00529 | 63.2% |
| VICReg+causal vs vanilla+causal | +0.00277 | +0.00371 | 60.0% |
| VICReg+causal vs causal-only | -0.00057 | -0.00009 | 49.1% |

VICReg therefore improves the vanilla representation but does not establish incremental value over
causal features. The useful cells are target/timeframe-specific:

- 3/5-minute contexts contain forward realized-volatility information.
- 5/15/30-minute contexts contain moderate favorable-excursion reach information.
- 30-minute contexts show some trend-termination information.
- Unconditional direction and continuation remain weak.
- Continuous expected policy-R regression remains generally negative.

## Full-history realized-R ruler

The downstream head used eligible history from 2019 onward and fixed outer tests inside
`[2024-07-01, 2025-07-01)`. Thresholds were selected by nested train-only isotonic calibration.
Execution used next-bar-open entry, no added delay, zero modeled round-trip slippage, declared
instrument-specific round-trip fees, adverse-first same-bar ambiguity, and one active trade per
policy/ticker/timeframe.

### Concatenated fusion

| Policy | Arm | Trades | Mean net R | PF | WR | Total R |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ATR-zigzag structural 3R/360m | causal | 350 | -0.0037 | 0.993 | 0.383 | -1.31 |
| ATR-zigzag structural 3R/360m | VICReg fusion | 348 | -0.0277 | 0.952 | 0.359 | -9.65 |
| fractal-k2 structural 3R/360m | causal | 954 | -0.0759 | 0.886 | 0.320 | -72.42 |
| fractal-k2 structural 3R/360m | VICReg fusion | 631 | -0.1126 | 0.828 | 0.317 | -71.05 |

Paired weekly-block delta-R-per-candidate intervals versus causal:

- ATR-zigzag: `-0.000734`, 95% interval `[-0.006621, +0.005028]`.
- Fractal-k2: `+0.000225`, 95% interval `[-0.012003, +0.013704]`.

### Residual fusion

| Policy | Arm | Trades | Mean net R | PF | Total R |
| --- | --- | ---: | ---: | ---: | ---: |
| ATR-zigzag structural 3R/360m | VICReg residual | 175 | -0.0974 | 0.852 | -17.05 |
| fractal-k2 structural 3R/360m | VICReg residual | 805 | -0.1193 | 0.825 | -96.00 |

Residual fusion also failed. Neither policy had positive economics, significant paired lift, or
FDR-adjusted evidence of improvement.

## Interpretation

The SSL methodology improved what MantisV2 represents, especially volatility, trend efficiency,
range expansion and forward move magnitude. That is a real representation-learning result.

It is not a trading result. Once causal features, calibration, fees and concurrency are applied,
the checkpoint adds no stable utility and often makes selection worse. The binding constraint in
this lane remains event/label/head economics rather than MantisV2 representation capacity.

## Artifacts

- Stage 0: `output/mantis_v2_ssl_pilot/stage0_parity.json`
- Training comparison: `output/mantis_v2_ssl_pilot_256_v1/comparison.json`
- Extended path score: `output/mantis_v2_ssl_pilot_256_v1/downstream_gate_scores/`
- Vanilla control: `output/mantis_v2_ssl_pilot_256_v1/downstream_gate_scores_vanilla/`
- Causal control: `output/mantis_v2_ssl_pilot_256_v1/downstream_gate_scores_causal/`
- Concatenated trading ruler: `output/mantis_v2_ssl_pilot_256_v1/trading_nested_isotonic_primary/`
- Residual trading ruler: `output/mantis_v2_ssl_pilot_256_v1/trading_residual_primary/`

## Next decision

Stop this SSL branch. Do not spend a second seed or build Stage 1/3 from this checkpoint. The next
highest-value work is to improve the causal event pools and conditional trading target/head, then
reuse this frozen VICReg checkpoint only as a control. Reopen SSL only if a corrected downstream
ruler shows representation quality is the binding constraint.
