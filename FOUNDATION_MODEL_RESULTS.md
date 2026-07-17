# Foundation Model Tournament
Generated from the sealed tournament artifacts on 2026-07-16.

## Executive verdict

The cross-family representation evaluation is complete for 12 trainable encoder arms across vanilla, Stage 1, Stage 2, and Stage 3 checkpoints. That produced 48 evaluated representations: 12 vanilla baselines and 36 trained stage checkpoints. Sundial and TabPFN-TS have explicit coverage exceptions described below.

The result is not a model ready for strategy promotion:

- Every trained checkpoint has negative forward absolute-move R².
- The best forward-direction AUC among trained checkpoints is only 0.5190.
- The best raw forward-direction result is the **vanilla Chronos V2** baseline at 0.5207; training reduced that score at every stage.
- Stage 3 does not provide a systematic transfer benefit and often damages forward metrics.
- The encoders learn descriptive market state—volatility, trend efficiency, range expansion, and in-window direction—far better than future tradeability or direction.

The tournament is useful because it rejects the assumption that completing three stages automatically improves a foundation model. It does not. The current objectives need redesign before another large training sweep.

## Status and scope

| Item | Value |
|---|---|
| Tournament status | `complete_with_declared_blockers` |
| Evaluation schema | `ffm_cross_family_representation_probe_v1` |
| Training interval | 2019-07-01 through 2024-07-01 |
| Validation interval | 2024-07-01 through 2025-07-01 |
| Reserved OOS boundary | 2025-07-01 and later |
| OOS read by this representation benchmark | **No** (`oos_read=false`) |
| Symbols | ES, NQ, RTY, YM, GC, SI, CL, ZB, ZN |
| Bar sizes | 1, 3, 5, 15, 30, and 60 minutes |
| Symbol/timeframe streams | 54 |
| Validation windows | 6,554 |
| Context | 256 bars |
| Forward horizon | 16 bars |
| Probe | Standardized Ridge (`alpha=1`, `lsqr`) for regression; Logistic Regression (`C=1`) for classification |
| Walk-forward | Five expanding calendar folds with a two-span embargo |
| Window artifact SHA-256 | `646b1dbeb0d45d074a1f7bd14f8f751488530630c36b59e47309206387158716` |
| Fold contract SHA-256 | `2348c5c9df335b931130af2c5278b8969754d80fa739858cc037715f68f9ab55` |

### What “apples to apples” means here

The evaluation is apples to apples: every representation uses the same cached windows, target definitions, chronological folds, embargo, and probe family.

Training is aligned, not mathematically identical. Each family sees the same date range, symbols, timeframes, OHLCV information, stage order, and bounded optimization budget, but model-native tokenization, context handling, trainable parameterization, memory-limited batch sizes, and objective heads differ. Claiming identical training exposure would be false. The controlled downstream evaluation is the reliable comparison.

## Stages

| Stage | Intended job | Main concern exposed by this tournament |
|---|---|---|
| Vanilla | Untuned pretrained representation control | Several vanilla models are already competitive; adaptation can erase useful pretrained structure. |
| Stage 1 | Masked/reconstruction-style futures domain adaptation | Often improves descriptive structure, but does not establish forward predictability. |
| Stage 2 | Contrastive regime/context learning, initialized from Stage 1 | Sometimes shifts representations toward volatility or forward magnitude, but gains are inconsistent and can trade off against trend/range information. |
| Stage 3 | Forecast-oriented learning, initialized from Stage 2 | No systematic downstream benefit; often regresses forward metrics. |

“Stage 2 diagnostic” and “Stage 3 diagnostic” for Mantis mean that the checkpoints were trained and measured, but their parent Stage 1 checkpoint failed the locked canonical promotion gate. They are evidence, not promotable lineage.

## Target definitions

All targets are derived causally from each context and its separate future horizon. The representation receives only the context.

| Target | Metric | Definition and interpretation |
|---|---|---|
| `vol` | R² | Standard deviation of log close-to-close returns inside the context; realized in-window volatility. |
| `trend_eff` | R² | Absolute net log return divided by the sum of absolute log-return steps; approaches one for an efficient directional trend and zero for chop. |
| `range_expand` | R² | Log ratio of the high-low range in the second half of the context to the range in the first half; compression versus expansion. |
| `fwd_absmove` | R² | Absolute log return from the final context close to the final future close; asks whether a material move follows. |
| `direction` | AUC | Sign of the net in-window return. Descriptive only; reported but not a forward gate. |
| `fwd_dir` | AUC | Sign of the return from the final context close to the final future close; forward direction. |

For R², zero means no improvement over predicting the fold mean and negative values are worse than that baseline. For AUC, 0.5 is random ranking.

## Full trained-stage results

These are mean scores across the five walk-forward folds.

| Model | Stage | vol R² | trend_eff R² | range_expand R² | fwd_absmove R² | direction AUC | fwd_dir AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Chronos Bolt | S1 | 0.1855 | 0.8171 | 0.7835 | -0.3119 | 0.9925 | **0.5190** |
| Chronos Bolt | S2 | 0.2790 | 0.7873 | **0.7877** | -0.3416 | 0.9929 | 0.5177 |
| Chronos Bolt | S3 | 0.1980 | 0.8079 | 0.7777 | -0.2936 | 0.9916 | 0.5132 |
| Chronos V1 | S1 | 0.9177 | 0.3634 | 0.5623 | -0.0996 | 0.9577 | 0.5130 |
| Chronos V1 | S2 | **0.9298** | 0.3042 | 0.4931 | -0.0968 | 0.9599 | 0.5156 |
| Chronos V1 | S3 | 0.9228 | 0.3660 | 0.5714 | -0.0965 | 0.9585 | 0.5082 |
| Chronos V2 | S1 | 0.1075 | 0.7822 | 0.6666 | -0.5554 | 0.9886 | 0.5176 |
| Chronos V2 | S2 | 0.3233 | 0.7235 | 0.6942 | -0.4722 | 0.9863 | 0.5023 |
| Chronos V2 | S3 | 0.1285 | 0.7688 | 0.6685 | -0.4982 | 0.9885 | 0.5146 |
| Kronos Mini | S1 | 0.2368 | 0.0975 | 0.5014 | -0.2281 | 0.9225 | 0.5047 |
| Kronos Mini | S2 | 0.2379 | 0.0572 | 0.4532 | -0.1796 | 0.9080 | 0.5078 |
| Kronos Mini | S3 | 0.1584 | -0.0178 | 0.4024 | -0.2792 | 0.9139 | 0.4986 |
| Kronos Small | S1 | 0.3370 | 0.6678 | 0.7231 | -0.1611 | 0.9751 | 0.4942 |
| Kronos Small | S2 | 0.3658 | 0.6408 | 0.7259 | -0.1395 | 0.9714 | 0.5040 |
| Kronos Small | S3 | 0.3783 | 0.6443 | 0.7370 | -0.1282 | 0.9723 | 0.4973 |
| Mantis V1 | S1 | 0.0969 | 0.6908 | 0.7192 | -0.4777 | 0.9836 | 0.5066 |
| Mantis V1 | S2 diagnostic | 0.1752 | 0.6584 | 0.7651 | -0.5253 | 0.9817 | 0.5032 |
| Mantis V1 | S3 diagnostic | 0.1623 | 0.7618 | 0.7473 | -0.6331 | 0.9919 | 0.4976 |
| Mantis V2 | S1 | 0.1447 | 0.7902 | 0.7767 | -0.3478 | 0.9864 | 0.5070 |
| Mantis V2 | S2 diagnostic | 0.2651 | 0.5920 | 0.7362 | -0.4669 | 0.9694 | 0.5125 |
| Mantis V2 | S3 diagnostic | 0.0863 | 0.7170 | 0.6953 | -0.6009 | 0.9870 | 0.4940 |
| Moirai-2 Small | S1 | 0.3145 | 0.6030 | 0.7022 | -0.1432 | 0.9813 | 0.4998 |
| Moirai-2 Small | S2 | 0.3548 | 0.5200 | 0.6727 | **-0.0766** | 0.9707 | 0.5076 |
| Moirai-2 Small | S3 | 0.3514 | 0.6319 | 0.7267 | -0.1095 | 0.9822 | 0.5044 |
| MOMENT Small | S1 | 0.0072 | 0.5753 | 0.6481 | -0.9407 | **0.9932** | 0.5017 |
| MOMENT Small | S2 | 0.0972 | 0.5426 | 0.6008 | -0.8478 | 0.9913 | 0.4890 |
| MOMENT Small | S3 | 0.0941 | 0.5602 | 0.5803 | -0.7792 | 0.9921 | 0.5059 |
| TimesFM 2.5 | S1 | -0.1658 | 0.4033 | 0.5527 | -0.9258 | 0.9831 | 0.5142 |
| TimesFM 2.5 | S2 | -0.0360 | 0.4487 | 0.5784 | -0.7391 | 0.9859 | 0.5075 |
| TimesFM 2.5 | S3 | -0.1653 | 0.3803 | 0.5730 | -0.9449 | 0.9842 | 0.5084 |
| Toto-2 22M | S1 | 0.2792 | 0.6313 | 0.6983 | -0.1002 | 0.9696 | 0.5073 |
| Toto-2 22M | S2 | 0.2878 | 0.6255 | 0.6938 | -0.0991 | 0.9696 | 0.5061 |
| Toto-2 22M | S3 | 0.2711 | 0.6301 | 0.7045 | -0.0945 | 0.9690 | 0.5052 |
| TTM-R2 | S1 | 0.1253 | 0.1062 | 0.5399 | -0.3919 | 0.9931 | 0.5005 |
| TTM-R2 | S2 | 0.1119 | 0.0407 | 0.5190 | -0.3708 | 0.9926 | 0.5018 |
| TTM-R2 | S3 | 0.1377 | 0.1901 | 0.5406 | -0.3728 | 0.9930 | 0.5119 |

Bold values are the best trained-stage mean for each column. They are not evidence of deployable edge.

## Vanilla pretrained baselines

Vanilla controls are essential because a trained checkpoint is only useful if adaptation beats the model before futures training.

| Model | vol R² | trend_eff R² | range_expand R² | fwd_absmove R² | direction AUC | fwd_dir AUC |
|---|---:|---:|---:|---:|---:|---:|
| Chronos Bolt | 0.1432 | 0.7926 | 0.7859 | -0.3300 | 0.9909 | 0.5129 |
| Chronos V1 | 0.9029 | 0.2512 | 0.4796 | -0.1599 | 0.9540 | 0.5135 |
| Chronos V2 | -0.0945 | 0.6730 | 0.5491 | -0.8190 | 0.9816 | **0.5207** |
| Kronos Mini | 0.2445 | 0.1531 | 0.4926 | -0.1975 | 0.9204 | 0.5058 |
| Kronos Small | 0.3034 | 0.6775 | 0.7199 | -0.1792 | 0.9749 | 0.4917 |
| Mantis V1 | 0.1540 | 0.5841 | 0.5899 | -0.5207 | 0.9342 | 0.5053 |
| Mantis V2 | 0.1494 | 0.5492 | 0.5117 | -0.4566 | 0.9156 | 0.5027 |
| Moirai-2 Small | 0.1285 | 0.7191 | 0.7352 | -0.1535 | 0.9845 | 0.4963 |
| MOMENT Small | -0.2944 | 0.5285 | 0.5450 | -1.1982 | 0.9906 | 0.5067 |
| TimesFM 2.5 | -0.2223 | 0.2330 | 0.4986 | -0.7928 | 0.9776 | 0.5050 |
| Toto-2 22M | 0.2826 | 0.6261 | 0.6956 | -0.0998 | 0.9709 | 0.5084 |
| TTM-R2 | 0.1030 | 0.3047 | 0.4945 | -0.4200 | 0.9934 | 0.5126 |

## Coverage and exceptions

| Model | Stage 1 | Stage 2 | Stage 3 | Note |
|---|---|---|---|---|
| Kronos Mini | Complete | Complete | Complete | Fully evaluated. |
| Kronos Small | Complete | Complete | Complete | Fully evaluated. |
| MOMENT Small | Complete | Complete | Complete | Fully evaluated. |
| Chronos V1 | Complete | Complete | Complete | Fully evaluated. |
| Chronos Bolt | Complete | Complete | Complete | Fully evaluated. |
| Chronos V2 | Complete | Complete | Complete | Fully evaluated. |
| TTM-R2 | Complete | Complete | Complete | Fully evaluated. |
| TimesFM 2.5 | Complete | Complete | Complete | Fully evaluated. |
| Moirai-2 Small | Complete | Complete | Complete | Fully evaluated. |
| Mantis V1 | Complete | Diagnostic | Diagnostic | Stage 1 failed the locked canonical gate; later checkpoints were not promoted. |
| Mantis V2 | Complete | Diagnostic | Diagnostic | Stage 1 failed the locked canonical gate; later checkpoints were not promoted. |
| Toto-2 22M | Complete | Complete | Complete | Fully evaluated. |
| Sundial Base | Blocked | Blocked | Blocked | Native hidden states became non-finite on real OHLCV at the first attention layer, including float32 and normalized smoke paths. Forecast samples were finite, but no valid frozen representation could be exported. |
| TabPFN-TS | Not applicable | Not applicable | Not applicable | In-context downstream model; it has no persistent staged encoder checkpoint matching this Stage 1→2→3 representation protocol. It belongs in downstream probe comparisons, not this encoder table. |

Sundial and TabPFN-TS must not be represented as zero-scoring models. One is blocked by an invalid embedding path; the other is outside the staged-encoder contract.

## Metric leaders and stability

| Question | Leader | Mean | Fold evidence | Interpretation |
|---|---|---:|---|---|
| Best realized-volatility representation | Chronos V1 S2 | 0.9298 R² | Fold standard deviation 0.0208 | Strong and stable descriptive volatility encoding. |
| Best trend-efficiency representation | Chronos Bolt S1 | 0.8171 R² | Fold standard deviation 0.0960 | Strong descriptive trend/chop separation. |
| Best range-expansion representation | Chronos Bolt S2 | 0.7877 R² | Fold standard deviation 0.1036 | Strong descriptive compression/expansion encoding. |
| Best future absolute-move score | Moirai-2 Small S2 | -0.0766 R² | Folds: -0.2268, -0.1874, -0.2260, 0.1344, 0.1229 | Still fails the mean baseline and is unstable across time. |
| Best in-window direction | MOMENT Small S1 | 0.9932 AUC | Descriptive target | Very high, but not forward evidence. |
| Best trained future direction | Chronos Bolt S1 | 0.5190 AUC | Folds: 0.5055, 0.5135, 0.5290, 0.5156, 0.5312; std 0.0109 | Modest ranking signal only; too weak for promotion by itself. |
| Best future direction including vanilla | Chronos V2 vanilla | 0.5207 AUC | Training reduced the mean at S1, S2, and S3 | Direct evidence that current adaptation can destroy useful pretrained signal. |

## Findings by model family

### Chronos

- **Chronos Bolt** is the best balanced descriptive encoder. Stage 1 leads trained forward-direction AUC and trend efficiency; Stage 2 leads range expansion. Its forward absolute-move R² remains deeply negative.
- **Chronos V1** dominates volatility representation at every trained stage. It does not transfer that strength to forward direction or forward move.
- **Chronos V2** improves descriptive R² substantially over vanilla, but vanilla has the best forward-direction AUC of the entire table. Stage 2 is especially destructive to that signal, falling from 0.5207 to 0.5023.
- The Chronos variants are not interchangeable. Architecture/pretraining differences produce materially different representation profiles.

### Kronos

- **Kronos Small** is consistently better than Mini on trend, range, and forward magnitude.
- Small improves forward absolute-move R² across stages, reaching -0.1282 at Stage 3, but forward-direction AUC remains effectively random.
- **Kronos Mini** regresses badly at Stage 3, including negative trend-efficiency R² and sub-random mean forward direction.
- Finance-native pretraining does not automatically translate into a superior frozen representation under this probe.

### Mantis

- **Mantis V2 Stage 1** is the best Mantis checkpoint in the valid lineage and is strong on trend/range description.
- Both Mantis Stage 1 checkpoints failed the locked canonical promotion gate; Stage 2 and Stage 3 results are diagnostic only.
- Stage 3 materially worsens `fwd_absmove`: V1 falls to -0.6331 and V2 falls to -0.6009.
- Mantis’s classification-oriented representation is useful for descriptive structure, but the current three-stage adaptation does not produce forward transfer.

### Moirai

- **Moirai-2 Small Stage 2** has the least-bad forward absolute-move R² and improves that target by 0.0769 versus vanilla.
- That gain is positive in four of five delta folds, but the absolute score is still negative and the raw fold scores change sign across time.
- Stage 2 pays for forward-magnitude improvement with large losses in trend and range representation versus vanilla.
- Moirai S2 is a redesign candidate, not a deployable winner.

### MOMENT

- Domain adaptation substantially repairs MOMENT’s very weak vanilla volatility and future-magnitude scores.
- It produces nearly perfect in-window direction encoding, which confirms descriptive sensitivity but says nothing about future direction.
- Stage 2 forward-direction AUC is 0.4890. Stage 3 recovers to 0.5059 but remains unconvincing.
- The issue is not simply lack of training. The representation/objective alignment is wrong for the desired forward task.

### TimesFM

- Stage 2 improves every R² target versus TimesFM vanilla, but absolute volatility and forward-move scores remain poor.
- Stage 3 gives those improvements back and is the weakest stage on future magnitude at -0.9449.
- A forecast teacher is not automatically a good frozen discriminative representation, especially after pooling and probing.

### Toto

- Toto is unusually stable across stages. Its scores barely move, suggesting either robust pretrained features or weak adaptation leverage under the current trainable path.
- Stage 3 has the best Toto forward-move score (-0.0945), a small +0.0053 improvement versus vanilla, while volatility and future-direction scores regress.
- The Stage 1/2/3 checkpoints are genuine trained artifacts, not frozen placeholders. The recorded Toto details are:

| Stage | Best validation loss | First gradient norm | Examples processed | Checkpoint SHA-256 |
|---|---:|---:|---:|---|
| S1 | 2.0046177283 | 0.0150786480 | 262,144 | `44ac2f3f72c162eb1c1cdb095d43f174ea2f5bace987191252c0c675211bede1` |
| S2 | 5.0829324722 | 0.0025103078 | 262,144 | `b61a15594d5313a4daa49289109ab1e971f92cff37abb5c4dfe73544e376671c` |
| S3 | 1.9461648092 | 0.0154044507 | 262,144 | `ae1c06819814f4b59fbfdc307c128c22b24b71035a5f20ce5df11d8ebf48c08f` |

### TTM

- TTM-R2 strongly captures in-window direction but is weak on trend efficiency and forward targets.
- Stage 3 improves trend efficiency and gives the family’s best future-direction result, 0.5119, while remaining below its vanilla forward-direction score of 0.5126.
- This is another case where more stages do not beat the pretrained control.

### Sundial and TabPFN-TS

- **Sundial** needs a separate forecast-output evaluation or a corrected, numerically valid hidden-state extraction contract. Fabricating representation scores from its finite forecast samples would not be comparable to the other encoders.
- **TabPFN-TS** should be evaluated as an in-context downstream head on fixed embeddings or causal tabular features. Forcing it into Stage 1→2→3 would create a fake checkpoint protocol.

## Cross-model conclusions

### 1. Descriptive representation is not the bottleneck

Multiple models achieve high R²/AUC on volatility, trend/chop, range expansion, and in-window direction. The pipeline can extract market-state summaries from OHLCV.

### 2. Forward transfer is the bottleneck

All 36 trained checkpoints have negative `fwd_absmove` R². Forward-direction AUC clusters around 0.5. The current stages mostly learn to summarize the observed window, not to identify an actionable future move.

### 3. Stage 3 is not justified as currently designed

Stage 3 sometimes improves one metric, but there is no consistent cross-family gain. It frequently erases Stage 1/2 strengths, and it is especially damaging for Mantis and TimesFM. Completing Stage 3 should not be treated as automatic progress.

### 4. Adaptation can hurt a strong pretrained prior

Chronos V2 vanilla at 0.5207 forward-direction AUC beats every trained checkpoint. This is the clearest counterexample to “training is always better than frozen.” Training is necessary only when the objective preserves or improves the target information.

### 5. No single scalar should pick the winner

The targets measure different properties and even use different metric families. Averaging R² and AUC would hide material regressions. Promotion must impose per-target constraints, fold stability, and repeated-seed evidence.

### 6. Lower noise at larger bars does not fix bar-based semantics

All timeframes are included, but a 16-bar horizon means 16 minutes at 1-minute bars and 16 hours at 60-minute bars. A pooled cross-timeframe mean therefore combines different economic horizons. Future work should sample and report by elapsed time as well as by bar count.

## What must change before another full sweep

1. **Redesign Stage 2 around elapsed time.** Use positives separated by economically meaningful wall-clock intervals, reduce trivial overlap, sample augmentations independently per observation, and ablate negative-free objectives against the current contrastive baseline.
2. **Replace the current Stage 3 objective.** Start with return quantiles, absolute move or realized-volatility targets, a direction auxiliary loss, and teacher-feature anchoring to the best Stage 2 representation.
3. **Protect pretrained signal.** Add a vanilla-feature anchoring or distillation term and require non-inferiority to vanilla on each primary target.
4. **Use per-target promotion gates.** Require declared gains on primary forward targets, explicit non-inferiority limits on descriptive targets, fold-dispersion limits, and no averaging of unlike metrics.
5. **Run multiple seeds only after the redesign passes a cheap falsification run.** At least two seeds plus shuffled-label and time-destroyed controls are required for finalists.
6. **Report elapsed-time slices.** Keep the identical global benchmark, but add per-timeframe and fixed-wall-clock target reports so 1-minute and 60-minute behavior cannot cancel each other.
7. **Run strategy benchmarks only on finalists.** The two requested gist-derived walk-forward scripts should come after a representation checkpoint clears the forward gate. Repeatedly running weak checkpoints on the reserved period increases selection contamination without fixing the representation.

## Current shortlist for controlled redesigns

This is a research shortlist, not a deployment ranking:

| Candidate | Why retain it | Why it is not promoted |
|---|---|---|
| Chronos Bolt S1 | Best trained future-direction AUC and best trend representation | Forward AUC is only 0.5190; forward-move R² is -0.3119. |
| Chronos V1 S2 | Best and most stable volatility representation | Weak trend/range balance and no compelling forward edge. |
| Moirai-2 Small S2 | Best future absolute-move R² and strongest improvement on that target versus vanilla | Absolute R² remains negative and varies sharply by fold. |
| Mantis V2 S1 | Best valid Mantis lineage; strong descriptive trend/range representation | Failed the locked canonical promotion gate and has weak forward metrics. |
| Chronos V2 vanilla | Best overall forward-direction AUC | Not improved by training; 0.5207 remains too weak to claim strategy value. |

## Reproducibility and verification

### Canonical artifacts

- Machine-readable results: `output/foundation_tournament/representation_apples/representation_results.json`
- Generated compact table: `output/foundation_tournament/representation_apples/representation_results.md`
- Shared window cache: `output/foundation_tournament/representation_apples/windows.npz`
- Per-model embeddings: `output/foundation_tournament/representation_apples/embeddings/`
- Stage checkpoints and training reports: `output/foundation_tournament/final_staged/`

### Verification completed

- All 36 trained checkpoint files and their 36 exported embedding artifacts were hash-audited with no audit errors.
- The shared benchmark contains exactly 6,554 rows and reports `oos_read=false`.
- The result file contains all 12 vanilla controls and all 36 completed trained-stage rows.
- Toto batched float32 embedding parity passed with maximum absolute difference `1.1920928955078125e-06` and `allclose(atol=1e-6)`.
- Full repository test run: **792 passed, 82 skipped, 0 failed**, with 6 warnings, in 39.86 seconds.
- `git diff --check` passed.

### Not verified or not available

- Sundial native representation extraction is not numerically valid on the real OHLCV smoke path.
- TabPFN-TS has no comparable persistent Stage 1→2→3 encoder.
- The current table is not a multi-seed uncertainty study.
- The table reports aggregate cross-stream means; it does not replace per-symbol, per-timeframe, or fixed-wall-clock analyses.
- It does not prove economic value, net profitability, capacity, or robustness to fees/slippage.
- This benchmark did not read the reserved OOS interval. However, dates from 2025 onward were inspected in earlier project research, so they are not globally pristine evidence for the project anymore.
- The worktree contained pre-existing uncommitted changes. The canonical JSON, hashes, and checkpoint-side source/report artifacts are the authoritative experiment records.

## Final decision

Do not scale the present Stage 1→2→3 recipe unchanged. Preserve the current checkpoints as controls, redesign Stage 2 and Stage 3 on a small bounded subset, demand per-target forward improvement across folds and seeds, and only then spend compute on another full 54-stream tournament and strategy evaluation.
