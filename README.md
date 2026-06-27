# 🏛️ Futures Foundation Model (FFM)

![Python Unit Tests](https://github.com/johnamcruz/Futures-Foundation-Model/actions/workflows/main.yml/badge.svg)

**A futures-market foundation layer built on pretrained Chronos-Bolt — frozen embeddings of intraday market context, plus strategy-pluggable training, evaluation, and deployment pipelines.**

**Contents:** [Quick Start](#quick-start) · [Overview](#overview) · [Foundation Surface](#the-foundation-surface) · [Chronos Pipeline](#chronos-pipeline--training-evaluation-deployment) · [The Training Loop](#the-training-loop--overfit-driven) · [Add a Strategy](#add-a-strategy) · [Standalone Pipelines](#xgboost-pipeline-standalone) · [Data](#data) · [Project Structure](#project-structure)

---

## Quick Start

```bash
pip install -e .
pip install chronos-forecasting "xgboost>=2.0"      # foundation embed + heads
```

A strategy is a small **labeler** (event candidates + features) that rides the frozen foundation embedding. Two calls take it from idea to deployable bundle:

```python
from futures_foundation.pipeline import evaluate as ev, produce

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

The pipeline: **frozen Chronos-Bolt embedding (feature extractor) → market-regime HMM → XGBoost (classifier)**, concatenated with strategy-specific hand-crafted features. The embedding *is* the market-state representation; the regime HMM adds a compact **"what regime are we in now"** summary; the head learns whatever of it the task needs. No forward-prediction context layer — every input is either the frozen embedding, a data-discovered regime state, or hand-crafted geometry.

### Market-regime layer (HMM)

A Hidden Markov Model fit on **existing volatility features** discovers the market's regimes (trend / range / chop) as hidden states, and appends its **causal-filtered state posteriors** to the head's feature vector:

```
Chronos embed ─┐
               ├─► HMM (discovers regimes) ─► state posteriors [K] ─┐
vol features ──┘                                                    ▼
   [ 256 embed | strategy features | K regime posteriors ] ─► XGBoost
```

- **Data-discovered, not hand-tuned.** The HMM finds regimes unsupervised — no hand-set thresholds to overfit. Where the embedding describes *what the bar looks like*, the regime posteriors describe the *persistent context*.
- **Leak-safe + causal.** Fit on **train rows only** (per-fold in eval, train-span in produce); decoded with **causal forward-filtering** `P(state_t | obs_1..t)` (never smoothing) — warm-started across splits in one pass, no future peek.
- **Additive + opt-in.** Posteriors are *concatenated* (the embedding path is byte-identical); `use_regime` defaults **off** (Kalman and existing bundles unchanged) and **on for fractal**. The fitted HMM is baked into the bundle.
- **No new properties.** It observes volatility features the labeler already computes (selected by name) — it adds no hand-crafted inputs, only the regime summary.

Embeddings are **disk-cached** (content-hash verified, recipe-signature namespaced), so re-runs skip the multi-million-context embed and only the cheap heads/HMM recompute.

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

Select a backbone by name (HF-style): `CHRONOS_FT_CKPT=<name>` or `ChronosExtractor.from_pretrained('<name>')` (`vanilla`/unset = frozen base). Produced bundles stamp `chronos_ckpt`, so the consumer loads the **matching** backbone automatically (`from_pretrained(bundle['chronos_ckpt'])`).

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
from futures_foundation.pipeline.strategy import StrategyLabeler

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
from futures_foundation.pipeline import evaluate as ev, produce

verdict = ev.run(MyLabeler(), loop=True, return_verdict=True)   # overfit-driven training loop
if verdict['final']['generalizes']:
    produce.train(MyLabeler(), holdout_months=1)                # production bundle
```

> `loop=True` runs the full [training loop](#the-training-loop--overfit-driven). Use `loop=False` (default) for a single walk-forward pass — e.g. inside A/B harnesses where each arm must see the *same* untuned pass.

```bash
# ONNX export (XGBoost heads + Chronos encoder), each parity-checked vs the joblib.
# A) as part of produce (option):
produce.train(MyLabeler(), export_onnx=True)   # any produce/pipeline script: pass --onnx
# B) standalone, from an existing bundle:
python3 -m futures_foundation.extractors.chronos.onnx_export <bundle.joblib>
```

### Pipeline components

| Component | Role |
|---|---|
| (backbone) | The seam is `futures_foundation.foundation`; modules here import it as `backbone` (removed `pipelines/chronos` entirely). |
| `evaluate.py` | Walk-forward harness (`run`) — batch-embed ONCE across all folds → per-fold thread-parallel XGBoost (5–10× speedup). **3-way train/validate/test** with VAL-selected threshold + **VAL→TEST generalization gate** + auto-regularize. `loop=True` runs the [overfit-driven training loop](#the-training-loop--overfit-driven); `loop=False` is the single-pass primitive. Pre-registered PASS/FAIL auto-verdict (constants at module top — goalpost-moving requires editing constants *before* the next run). |
| `train_loop.py` | The overfit-driven training loop: default WF → generalize check → Optuna only if overfit → rerun → repeat → final full WF. Returns chosen params + final verdict + history. |
| `tune_head.py` | Optuna head tuner with a **generalization-robust** objective + held-out guard and **auto-fallback to defaults**. `--walkforward` runs the scan then the full 3-way walk-forward with the tuned params. |
| `produce.py` | Production training: ONE fit on full corpus minus N-month holdout; saves joblib bundle (signal head + risk head + `feat_dim` + `ctx_window` + `chronos_ckpt` + labeler config + holdout threshold sweep). Production-scale defaults (`n_estimators=600, max_depth=5`). |
| `extractors/chronos/onnx_export.py` | Bundle → ONNX (signal head + risk head + Chronos encoder), each **parity-checked vs the joblib** (heads ~1e-7/1e-6, encoder ~1e-5). Chronos-specific (encoder export), so it lives under `extractors/chronos`, not the generic pipeline. Used by `produce(export_onnx=True)` / `--onnx`, or standalone. Encoder export runs in a torch subprocess (`onnx_encoder.py`) to avoid the libomp collision. |
| `head_xgb.py` | `XGBHead` (signal classifier) + `XGBRiskHead` (log1p-transformed max-favorable-R regression for dynamic TP). |
| `bolt_finetune.py` / `bolt_ab.py` | Optional Bolt domain-adaptation fine-tune + vanilla-vs-fine-tuned A/B harness on a real strategy. |
| `_primitives.py` | Pure-numpy indicator/barrier primitives the **live** strategies certified against. Numerically divergent from `futures_foundation.primitives` — deliberately not consolidated (see module docstring). |
| `data.py` | Long-format assembly + leak-guarded rolling walk-forward folds. |
| `_ft/` | Vendored upstream Chronos T5 fine-tune path (historical). |

All deps are in `requirements.txt`, including the ONNX-export stack (`onnxmltools`, `onnxruntime`, `onnx`, `onnxscript`) used by `produce(export_onnx=True)` / `--onnx`.

---

## Strategy Labeling & Evaluation Framework

**`futures_foundation.finetune` — the torch-free survivors of the v0.3–v1.3 fine-tuning framework: labeling, health monitoring, reporting, realized-R economics.** The torch walk-forward trainer was retired with the from-scratch backbone (training now happens in `futures_foundation.pipeline`); the layers every pipeline still leans on remain:

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

`futures_foundation.labels` provides the 4-task self-supervised generators (regime / volatility / structure / range), produced by `prepare_data` for the XGBoost pipeline's parquet cache.

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
│   ├── regime.py                 # ★ Market-regime HMM (leak-safe, causal posteriors)
│   ├── pipeline/embed_cache.py   # Disk embed cache (content-hash + recipe-signature)
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
│   ├── build_tb_corpus.py        # Triple-barrier direction corpus builder
│   └── finetune_tb_direction.py  # TB-direction backbone fine-tune (research)
├── docs/                         # Build specs + runbooks
├── tests/                        # 487 unit tests (pre-commit gated; torch-free by contract)
└── data/                         # Raw OHLCV CSVs (gitignored)
```

---

## Roadmap

- [x] Chronos-Bolt as the foundation (seam promoted, torch stack retired, torch-free import contract)
- [x] Capability probes — measured what the foundation knows, per input recipe (5 arms, gates + shuffle + trivial adversary)
- [x] Bolt domain-adaptation fine-tune + A/B harness (verdict: vanilla wins for selection — stay frozen)
- [ ] Single-file ONNX export (frozen embedding + XGBoost head in one graph) for the bot
- [ ] Multivariate context / Chronos-2 (next information rung beyond bars+features)

---

## License

Apache 2.0 — See [LICENSE](LICENSE) for details.

---

## Disclaimer

This software is for **research and educational purposes only**. It does not constitute financial advice. Trading futures involves substantial risk of loss. Past performance of any model does not guarantee future results.
