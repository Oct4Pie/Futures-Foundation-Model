# 🏛️ Futures Foundation Model (FFM)

![Python Unit Tests](https://github.com/johnamcruz/Futures-Foundation-Model/actions/workflows/main.yml/badge.svg)

**A futures-market foundation layer built on pretrained Chronos-Bolt — frozen embeddings of intraday market context, plus strategy-pluggable training, evaluation, and deployment pipelines.**

**Contents:** [Quick Start](#quick-start) · [Overview](#overview) · [Foundation Surface](#the-foundation-surface) · [Chronos Pipeline](#chronos-pipeline--training-evaluation-deployment) · [The Training Loop](#the-training-loop--overfit-driven) · [Add a Strategy](#add-a-strategy) · [Standalone Pipelines](#xgboost-pipeline-standalone) · [Data](#data) · [Project Structure](#project-structure) · [Releases](#releases)

---

## Quick Start

```bash
pip install -e .
pip install chronos-forecasting "xgboost>=2.0"      # foundation embed + heads
```

A strategy is a small **labeler** (event candidates + features) that rides the frozen foundation embedding. Two calls take it from idea to deployable bundle:

```python
from futures_foundation.chronos import evaluate as ev, produce

# 1) VALIDATE — the overfit-driven training loop (one call does it all):
#    default walk-forward → VAL→TEST generalization gate → Optuna ONLY if it
#    overfits → rerun → repeat until it passes → final FULL walk-forward.
verdict = ev.run(MyLabeler(), loop=True, return_verdict=True)

# 2) PRODUCE — only if it generalizes; trains on all data minus a holdout,
#    saves one joblib bundle the bot loads.
if verdict['final']['generalizes']:
    produce.train(MyLabeler(), holdout_months=1)
```

→ Labeler contract: [Add a Strategy](#add-a-strategy) · How validation catches & fixes overfitting: [The Training Loop](#the-training-loop--overfit-driven)

---

## Overview

FFM gives downstream trading models a shared **market-understanding layer**. The foundation is **`amazon/chronos-bolt-tiny`** — a time-series transformer pretrained by Amazon on large forecasting corpora — used **frozen**: a 128-bar log-close context ending at the decision bar becomes a 256-dim embedding that downstream heads (XGBoost classifiers/regressors) consume alongside strategy-specific features.

> **History.** FFM v0.x–v1.x trained a transformer backbone from scratch (68 hand-derived features, 4 self-supervised heads, ~2.3M bars). That stack is retired — building on a foundation model pretrained on vastly more data proved better than pretraining our own. The last version with the full from-scratch stack is preserved at git tag **`ffm-transformer-final`**. The *philosophy* survives unchanged; only who provides the embedding changed.

### Philosophy

> Separate **"understanding market context"** from **"making strategy-specific decisions."**

Just as BERT learns language structure before being fine-tuned for sentiment or Q&A, the foundation embedding captures market state before any strategy logic runs. The strategy adds only what the foundation cannot derive: setup geometry, zone age, entry distance, risk sizing. Market-context knowledge is never duplicated across strategies.

This architecture is **proven live**: a production selection model (frozen Bolt embedding + XGBoost head) runs in production, certified on the honest-ruler walk-forward with pre-registered controls.

### What the foundation actually knows (measured)

Every context target is probed with pre-registered gates, a shuffled-label control (leak detector), and a **trivial baseline** (8 trailing summary stats — "does the foundation know more than cheap features?"). Full pre-2023 corpus, 6 tickers × {3min, 5min}, ~236k decision bars. The decisive finding (FFM 2.1): the **enriched input recipe — `[Bolt embedding | 68-feature library]`** — beats both the embedding alone and the trivial adversary on every shipped head; close-only context was the binding constraint:

| Forward target | Emb only | Trivial | **Emb + 68 features** | Verdict |
|---|---|---|---|---|
| Realized-vol percentile (10-bar fwd) | r .52 | .41 | **r .64** | ✅ ships |
| Vol expansion >1.5× median (20-bar fwd) | AUC .78 | .70 | **AUC .82** | ✅ ships |
| Quiet persists (on quiet bars, 20-bar) | AUC .69 | .59 | **AUC .74** | ✅ ships |
| Structure: HH/HL vs LL/LH (20-bar fwd) | AUC .79 | .81 | **AUC .82** | ✅ ships (beats trivial only when enriched) |
| Range-bound (10-bar fwd) | AUC .59 | .67 | **AUC .70** | ✅ ships |
| Trendiness: fwd efficiency ratio | r .12 | .12 | **r .15** | ✅ ships (weak — a tilt, not a trigger) |

The division of labor this implies is the design: the foundation understands **conditions** (volatility regime, expansion risk, structure, persistence); the strategy supplies the **edge** (signal selection, direction, sizing). Probe harness: `scripts/probe_context_heads.py` (`--ff68` for the input-arm comparison).

### Context Heads — the per-candle market readout (FFM 2.1)

`futures_foundation.context.ContextHeads` packages that knowledge as **seven named, calibrated fields at any bar** — for entry confirmation, exit/management context, RL observations, or fusion into any downstream model:

```python
from futures_foundation.context import ContextHeads

heads = ContextHeads.load('heads_<date>.joblib')        # or $CONTEXT_HEADS_BUNDLE
ctx = heads.context_at(ohlcv_df, bar_indices, 'ES')     # DataFrame: 7 ctx_* columns
```

Certified out-of-sample on 80,786 bars (2023–2026) the heads never saw: calibration is monotone — `ctx_vol_expansion` deciles run 1%→90% realized, `ctx_structure` 10%→92%, `ctx_quiet_persist` 58%→99% — and a 7-feature regime classifier scores 52.2% vs a 30.2% majority baseline OOS (`scripts/demo_regime_model.py`). Heads are trained once on pre-2023 data (`scripts/train_context_heads.py`), frozen, and leak-guarded: downstream training that consumes `ctx_*` is restricted to ≥ 2023 by the fusion seam.

### Why this architecture

1. **Regime changes don't require retraining.** The frozen embedding maps any market state into the same representation space; downstream heads trained across regimes adjust automatically. Domain shift handling comes from Bolt's pretraining breadth, not from our retraining cadence.
2. **Adding new data is just a re-run.** The foundation is frozen; only the cheap XGBoost heads retrain (quarterly runbook: `docs/`).
3. **One foundation, unlimited strategies.** Each strategy is a thin labeler + features plug-in; many strategies ride the same embedding.
4. **Honest by construction.** Every result passes the honest ruler: walk-forward × {REAL, SHUFFLE, RANDOM, NAIVE} × seeds with a pre-registered auto-verdict. A number is believed only if REAL clearly beats every control.

---

## The Foundation Surface

`futures_foundation.foundation` is the canonical seam — the only way downstream code gets foundation embeddings:

```python
from futures_foundation import foundation

foundation.stamp_active_source(context='my run')   # loud backbone stamp — always call first
E = foundation.embed_bars(close, indices)          # [N, 256] float32, strictly causal
```

- `embed_bars(close, indices, ctx=128)` builds log-close windows of bars `<= t` for each decision index and embeds them. `embed(contexts)` is the lower-level batched form.
- **Process contract:** all torch/Chronos work runs in an **isolated subprocess** (`futures_foundation/_embed_worker.py`). The parent stays torch-free — torch and xgboost segfault in one process on macOS (libomp collision). `D_MODEL = 256` and `CTX = 128` are torch-free constants. Do not "optimize" the embed back in-process.
- **Backbone wiring guards** (post-incident, 2026-05-19): `$CHRONOS_FT_CKPT` selects a local fine-tuned checkpoint; unset = frozen vanilla. `stamp_active_source()` prints which backbone will load (`❄️ FROZEN` vs `🧪 FINE-TUNED`), scans `temp/` for fine-tune checkpoints sitting unused, and prints the exact `export` command if one is being silently ignored. The worker also stamps what it loaded to stderr (defense in depth).

Domain-adapting the foundation is supported but optional: `futures_foundation/chronos/bolt_finetune.py` (forecasting-loss fine-tune on our 9-instrument corpus) + `bolt_ab.py` (vanilla-vs-fine-tuned A/B on a real strategy). The measured verdict to date: domain fine-tuning improves forecasting loss but **not** selection edge — production runs vanilla frozen weights.

---

## Chronos Pipeline — training, evaluation, deployment

**`futures_foundation/chronos/` — the strategy-pluggable harness around the foundation: walk-forward evaluator with honest-ruler controls, production trainer, ONNX export.** This is the proven path every new strategy goes through.

**What it does:** a strategy labeler defines event candidates (e.g., trend-flip or channel-break events); for each event, the trailing 128-bar context → frozen foundation embedding → fused with hand-crafted features → XGBoost predicts `(P(take), R̂)`. Validation runs the **overfit-driven training loop** (below) on a **train / validate / test** walk-forward with **REAL/SHUFFLE/RANDOM/NAIVE controls** and a **pre-registered PASS/FAIL auto-verdict**. The production trainer then fits ONE signal head + ONE risk head on the full corpus minus an N-month holdout and saves a single joblib bundle the bot loads.

### The training loop — overfit-driven

`ev.run(labeler, loop=True)` runs the whole training process as one self-correcting loop. **Optuna is triggered only when overfitting is detected** — a strategy whose defaults already generalize keeps them untouched:

1. **Walk-forward** with the **default** XGBoost head.
2. **Generalizes?** (VAL→TEST meanR gap within tolerance) → **keep defaults, done.**
3. **Overfit?** → **Optuna scan** for params that generalize (objective rewards cross-fold stability; auto-falls-back to defaults unless the tuned params beat them on a held-out guard).
4. **Rerun** the walk-forward with the chosen params.
5. **Repeat** 2–4 until it passes (capped; if no params generalize, the model is **flagged**).
6. **One final FULL walk-forward** to confirm on unseen data.

Two guardrails make this honest: the **VAL→TEST generalization gate** (threshold is picked on *validation*, reported on *test*; an edge that decays from val to test is rejected as fake), and **auto-regularize** (when a head overfits train→val, it re-fits down a regularization ladder, keeping the rung with the best *validation* meanR). All tuning/selection sees train+validation only — **test is never consulted**.

### Add a strategy

```python
from futures_foundation.chronos.strategy import StrategyLabeler

class MyLabeler:
    n_classes = 2     # binary selection (take / skip)
    def calendar(self): ...                     # ticker × timestamp × target
    def build(self, lo, hi, test_start):
        # → (contexts: list of 128-bar log-close, labels: ndarray, keys)
        ...
    def features(self, keys): ...               # optional hand-craft to fuse
    def evaluate(self, keys, preds, risk_preds=None):
        # → per-trade realized R array
        ...
```

```python
from futures_foundation.chronos import evaluate as ev, produce

verdict = ev.run(MyLabeler(), loop=True, return_verdict=True)   # overfit-driven training loop
if verdict['final']['generalizes']:
    produce.train(MyLabeler(), holdout_months=1)                # production bundle
```

> `loop=True` runs the full [training loop](#the-training-loop--overfit-driven). Use `loop=False` (default) for a single walk-forward pass — e.g. inside A/B harnesses where each arm must see the *same* untuned pass.

```bash
# ONNX export + 3-layer verify (requires: pip install onnxmltools skl2onnx)
python3 -m futures_foundation.chronos.export_onnx <bundle.joblib>
```

### Pipeline components

| Component | Role |
|---|---|
| (backbone) | The seam is `futures_foundation.foundation`; modules here import it as `backbone` (removed `pipelines/chronos` entirely). |
| `evaluate.py` | Walk-forward harness (`run`) — batch-embed ONCE across all folds → per-fold thread-parallel XGBoost (5–10× speedup). **3-way train/validate/test** with VAL-selected threshold + **VAL→TEST generalization gate** + auto-regularize. `loop=True` runs the [overfit-driven training loop](#the-training-loop--overfit-driven); `loop=False` is the single-pass primitive. Pre-registered PASS/FAIL auto-verdict (constants at module top — goalpost-moving requires editing constants *before* the next run). |
| `train_loop.py` | The overfit-driven training loop: default WF → generalize check → Optuna only if overfit → rerun → repeat → final full WF. Returns chosen params + final verdict + history. |
| `tune_head.py` | Optuna head tuner with a **generalization-robust** objective + held-out guard and **auto-fallback to defaults**. `--walkforward` runs the scan then the full 3-way walk-forward with the tuned params. |
| `produce.py` | Production training: ONE fit on full corpus minus N-month holdout; saves joblib bundle (signal head + risk head + `feat_dim` + `ctx_window` + `chronos_ckpt` + labeler config + holdout threshold sweep). Production-scale defaults (`n_estimators=600, max_depth=5`). |
| `export_onnx.py` | joblib → 3 ONNX files (chronos + signal + risk) via subprocess-isolated phases. End-to-end `verify()`: per-stage drift < 1e-3, chained-pipeline equivalence, decision parity at the trading threshold. |
| `head_xgb.py` | `XGBHead` (signal classifier) + `XGBRiskHead` (log1p-transformed max-favorable-R regression for dynamic TP). |
| `bolt_finetune.py` / `bolt_ab.py` | Optional Bolt domain-adaptation fine-tune + vanilla-vs-fine-tuned A/B harness on a real strategy. |
| `_primitives.py` | Pure-numpy indicator/barrier primitives the **live** strategies certified against. Numerically divergent from `futures_foundation.primitives` — deliberately not consolidated (see module docstring). |
| `data.py` | Long-format assembly + leak-guarded rolling walk-forward folds. |
| `_ft/` | Vendored upstream Chronos T5 fine-tune path (historical). |

Extra deps (not in `requirements.txt`): `chronos-forecasting`, `xgboost>=2.0`, `onnxmltools` + `skl2onnx` (ONNX export only).

---

## Strategy Labeling & Evaluation Framework

**`futures_foundation.finetune` — the torch-free survivors of the v0.3–v1.3 fine-tuning framework: labeling, health monitoring, reporting, realized-R economics.** The torch walk-forward trainer was retired with the from-scratch backbone (training now happens in `futures_foundation.chronos`); the layers every pipeline still leans on remain:

| Component | Description |
|---|---|
| `StrategyLabeler` | ABC — implement `detect_events()` + `compute_features()`; the **final** `run()` applies a session-calibrated TP≥SL triple barrier (entry = next-bar open) and emits `signal_label` / `max_rr` / `sl_distance` / `direction`. The entry-after-signal / orientation bug class is centralized once, for every strategy. |
| `run_labeling()` | CSV I/O, timezone normalization, parquet caching per ticker. |
| `FoldHealthMonitor` | Stateful post-fold pathology detection (7 signals: EARLY_EPOCH, WEIGHT_LOCK, P80_DECLINE, VAL_TEST_GAP, N_COLLAPSE, CONFIDENCE_FLAT, ZERO_SIGNAL_FOLD) + consolidated `summary()`. Model-agnostic — feed it metrics from any trainer. |
| Reporting | `print_eval_summary` (confidence-threshold table), `print_fold_progression`, calibration block with monotonicity check. |
| Realized-R economics | PF / WR / mean-R / maxDD / no-top-1% from realized R under a trailing exit (not optimistic MFE), plus the CAGR·√Sortino *product* objective (can't be won by not-trading). |

`prepare_data()` (in `futures_foundation.prepare`) derives the 68 causal features from raw OHLCV CSVs to parquet — shared by the XGBoost pipeline and the quarterly retrain runbook.

---

## XGBoost Pipeline (Standalone)

**`pipelines/xgboost/` — a gradient-boosted direction classifier on the 68 causal features, fully independent of the foundation embedding.**

Every RTH bar is a candidate; a **V2 session-calibrated triple-barrier** labeler defines the target (long / no-trade / short); XGBoost predicts direction; a **hybrid Rogers-Satchell ATR/structure trailing stop** manages the exit; rolling 3-month-train / 1-month-test walk-forward with a per-window Optuna study. Objective: **CAGR·√Sortino with a −20% DD penalty** — a *product*, so the optimizer cannot win by not trading.

```python
from pipelines.xgboost.base import XGBStrategyLabeler, register

@register("my_strategy")
class MyLabeler(XGBStrategyLabeler):
    name = "my_strategy"
    def __init__(self, *, bar_minutes): self.bar_minutes = bar_minutes
    def label(self, df):        # df: datetime, OHLC, atr  →  Series of {-1,0,+1}
        return my_direction_logic(df)
```

```bash
python -m pipelines.xgboost.train --timeframe 5m --instrument ES --labeler my_strategy --trials 300
```

> **Verdict gate (non-negotiable):** a model is credible only if **every OOS month is profitable (PF > 1)** on the full multi-year rolling walk-forward. The leakage check carries a degenerate-shuffled guard (`_shuf_robust`): an economically-dead shuffled run is the desired no-leakage outcome and cannot raise a false flag.

Build spec: [`docs/xgboost-pipeline.md`](docs/xgboost-pipeline.md).

---

## RL Pipeline (Standalone)

**`pipelines/rl/` — a generic PPO walk-forward pipeline; the proprietary strategy is a private plug-in.**

Mechanical entry candidates (the strategy plug-in) → a PPO policy that learns, on a **frozen context embedding ⊕ position state**, an asymmetric *chop-veto* on entries (a veto must pay for itself) **and** the exit, jointly, under one **realized-R** reward. One frozen encoder = a stationary observation manifold. Episode = one trade.

| Component | Description |
|---|---|
| `RLStrategy` | ABC + registry — `detect_entries()` (mandatory causal-parity), `entry_filter` on/off, realized-R exit knobs |
| `shape_reward()` | The **single** extension point for account-aware reward (prop-firm balance, MLL / trailing drawdown). Default = identity — **FFM has zero account/prop-firm concept**; that IP lives only in the plug-in |
| `SingleTradeEnv` | obs = context ⊕ position-state; asymmetric veto + hold/exit; mechanical SL; terminal realized-R |
| `causal.py` | Generic causal-parity harness — streaming==batch; the look-ahead falsifier the shuffle audit *cannot* catch; mandatory gate for any detector before training |
| `run_walkforward()` | Windows (`common`) → injected trainer seam → OOS rollout → every-OOS-month-PF>1 + shuffle + multi-seed verdict |

> **IP boundary:** the public repo holds only generic machinery. Concrete strategies live in the private strategies repo.

---

## Data

### Supported instruments

9 instruments registered: **ES, NQ, RTY, YM** (equity indices), **GC, SI** (metals), **CL** (energy), **ZB, ZN** (rates).

### Input format

```
data/
├── ES_3min.csv      # datetime, open, high, low, close, volume
├── ES_5min.csv
└── ...
```

`databento/append_update.py` splices new DBN/CSV exports into `data/` continuously (see the quarterly retrain runbook in `docs/`).

### Feature derivation (68 causal features)

`derive_features` produces 68 instrument-agnostic (ATR-normalized), strictly causal features in 10 groups — bar anatomy, returns/momentum, volume dynamics, volatility, session context, market structure, CRT sweep state (1H/4H liquidity sweeps), candle psychology, HTF context (1H/4H/daily structure), and volume absorption/order flow. Used by the XGBoost pipeline and available as fusion features anywhere. Every feature is held to the no-look-ahead causal-parity rule (streaming == batch, per bar).

### Labels

Two label sets live in the library. `futures_foundation.labels` — the original 4-task self-supervised generators (regime / volatility / structure / range), still produced by `prepare_data` for the XGBoost pipeline's parquet cache. `futures_foundation.context.compute_context_labels` — the forward-looking, close-only context-head targets (volatility, vol-expansion, quiet-persist, structure, range-bound, trendiness, fwd-return) that train the FFM 2.1 context heads.

---

## Project Structure

```
Futures-Foundation-Model/
├── futures_foundation/           # The foundation package (torch-free to import)
│   ├── foundation.py             # ★ Chronos-Bolt seam: embed_bars/embed (subprocess), stamp_active_source, D_MODEL
│   ├── _embed_worker.py          # Subprocess worker (the only torch at runtime)
│   ├── features.py               # OHLCV → 68 causal features (10 groups)
│   ├── candle_psychology.py      # Candle psychology features
│   ├── labels.py                 # Legacy forward-looking label generation
│   ├── prepare.py                # prepare_data: raw CSVs → features+labels parquet
│   ├── primitives/               # Indicators, barriers, rolling, session, detection
│   ├── context.py                # ★ ContextHeads — 7 calibrated ctx_* fields per candle (FFM 2.1)
│   ├── chronos/                  # ★ Foundation training/eval/deploy harness (see above)
│   └── finetune/                 # Torch-free framework survivors
│       ├── base.py               # StrategyLabeler ABC (final run() = TP≥SL triple barrier)
│       ├── config.py             # TrainingConfig (labeling/eval params)
│       ├── health.py             # FoldHealthMonitor
│       └── trainer.py            # run_labeling + reporting + realized-R economics
├── pipelines/
│   ├── common/                   # Walk-forward windows, econ objective, robustness gates
│   ├── xgboost/                  # Standalone direction classifier on 68 features
│   └── rl/                       # Generic PPO walk-forward pipeline
├── scripts/
│   ├── probe_context_heads.py    # Capability probe (labels × input arms × gates/controls)
│   ├── train_context_heads.py    # Trains the production enriched heads bundle
│   └── demo_regime_model.py      # OOS certification: calibration + regime model
├── docs/                         # Build specs + runbooks
├── tests/                        # 487 unit tests (pre-commit gated; torch-free by contract)
└── data/                         # Raw OHLCV CSVs (gitignored)
```

---

## Releases

| Version | Description |
|---------|-------------|
| **v2.2** | **Overfit-driven training loop (train / validate / test + detect-and-fix).** The whole training process is now one self-correcting loop — `ev.run(labeler, loop=True)`: default walk-forward → **VAL→TEST generalization gate** → Optuna **only if overfit** → rerun → repeat until it passes → one final FULL walk-forward (`futures_foundation/chronos/train_loop.py`). Fixes a regression where the Chronos rebuild had collapsed the walk-forward to a 2-way train→test split (threshold-on-test bias): restored the **3-way split** (`walk_forward_folds` yields `train/val/test`, hard-asserted `train.max < val.min < test.min`) — head fit on **train**, threshold picked on **validation**, **test** reported untouched and never used for any selection. The **generalization gate** is a hard PASS/FAIL (an edge that decays val→test is rejected as fake). **Auto-regularize** re-fits down a regularization ladder when a head overfits train→val, keeping the best **validation** rung. The tuner (`tune_head`) optimizes a **generalization-robust** objective (cross-fold mean − penalty·std) and **auto-falls-back to defaults** unless tuned params beat them on a held-out guard. One entry point everywhere (`ev.run(loop=True)`); `loop=False` stays the single-pass primitive for A/B harnesses. 487 tests. |
| **v2.1** | **Enriched context heads — the foundation's second input pillar.** Measured recipe (5-arm probe, 236k bars): heads on **`[Bolt embedding \| 68-feature library]`** beat embedding-alone AND the trivial adversary on every shipped target — volume/orderflow/wicks/session/CRT/HTF were the missing inputs; close-only was the binding constraint. Ships 7 heads (`fwd_return`✱, `vol_expansion` .82, `volatility` r .64, `structure` .82 — beats trivial for the first time, `quiet_persist` .74, `trendiness` r .15✱, `range_bound` .69; ✱=weak/marginal, flagged). `context_at(df, idx, instrument)` consumes OHLCV; emb-only bundles back-compatible. OOS-certified on 80,786 unseen bars: monotone calibration, regime model +22.0pts over baseline. 462 tests. |
| **v2.0** | **Chronos-Bolt IS the foundation.** The from-scratch FFM transformer (model/dataset/pretrain/torch fine-tune trainer, ~6k lines) retired — preserved at tag `ffm-transformer-final`. The proven `pipelines/chronos` backbone seam promoted to **`futures_foundation.foundation`** (`embed_bars`, subprocess isolation, wiring-gap stamps); `import futures_foundation` is now torch-free by contract (tested). `finetune/` reduced to its torch-free, model-agnostic survivors (StrategyLabeler triple-barrier ABC, run_labeling, FoldHealthMonitor, reporting, realized-R economics); `prepare_data` rescued to `futures_foundation.prepare`. **Phase-0 capability probe** (`scripts/probe_context_heads.py`): frozen Bolt embeddings know future volatility regime beyond trivial features (vol percentile r=0.52 vs 0.41 trivial; expansion AUC 0.78 vs 0.70) — with shuffle controls clean. 436+ tests. |
| **v1.5** | `pipelines/chronos` — frozen Chronos backbone + XGBoost head pipeline: walk-forward batch-embed evaluator with REAL/SHUFFLE/RANDOM/NAIVE controls + 6-check pre-registered auto-verdict; `produce.py` production bundles; `export_onnx.py` 3-file export with 3-layer verify; `XGBRiskHead` log1p dynamic-TP; backbone wiring-gap guards (`stamp_active_source`). |
| **v1.4** | `pipelines/rl` — generic PPO walk-forward pipeline (SingleTradeEnv, causal-parity harness, shuffle + multi-seed verdicts, `shape_reward` IP seam). |
| **v1.3** | Finetune hardening: realized-R economic eval, shuffle-audit leakage gate, opt-in econ checkpoint selection, final-`run()` StrategyLabeler ABC; `pipelines/common` extracted. |
| **v1.2** | `pipelines/xgboost` standalone direction pipeline (V2 triple-barrier, RS hybrid trail, Optuna product objective, every-OOS-month gate). |
| **≤ v1.1** | From-scratch FFM era: backbone pretraining pipeline, walk-forward fine-tune framework, FoldHealthMonitor, 68-feature derivation, CRT/psychology/HTF features. Full history at tag `ffm-transformer-final`. |

---

## Roadmap

- [x] Chronos-Bolt as the foundation (seam promoted, torch stack retired, torch-free import contract)
- [x] Capability probes — measured what the foundation knows, per input recipe (5 arms, gates + shuffle + trivial adversary)
- [x] Bolt domain-adaptation fine-tune + A/B harness (verdict: vanilla wins for selection — stay frozen)
- [x] **Context heads (FFM 2.1)** — 7 calibrated `ctx_*` fields per candle on the enriched `[emb | 68-feature]` recipe, OOS-certified
- [ ] First ctx consumers, pre-registered A/Bs: RL exit observations; new strategies built with context from day one
- [ ] Enriched HTF readout (1h/4h ctx for enriched bundles — emb-only bundles cover it today)
- [ ] Single-file ONNX export (foundation + heads in one graph) for the bot
- [ ] Multivariate context / Chronos-2 (next information rung beyond bars+features)

---

## License

Apache 2.0 — See [LICENSE](LICENSE) for details.

---

## Disclaimer

This software is for **research and educational purposes only**. It does not constitute financial advice. Trading futures involves substantial risk of loss. Past performance of any model does not guarantee future results.
