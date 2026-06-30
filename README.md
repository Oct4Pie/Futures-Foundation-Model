# 🏛️ Futures Foundation Model (FFM)

![Python Unit Tests](https://github.com/johnamcruz/Futures-Foundation-Model/actions/workflows/main.yml/badge.svg)

**A model-agnostic classification foundation for futures markets — any pretrained time-series classification backbone learns market structure from raw OHLCV, then thin per-strategy heads finetune on top, all held to an honest-ruler walk-forward.**

**Contents:** [Quick Start](#quick-start) · [Philosophy](#philosophy--bert-for-futures) · [Overview](#overview) · [Self-Supervised Pretraining](#self-supervised-pretraining-the-bert-stage) · [The Classifier Seam](#the-classifier-seam--model-agnostic) · [Finetuning Pipeline](#finetuning-pipeline--walk-forward--produce) · [The Training Loop](#the-training-loop--overfit-driven) · [Add a Strategy](#add-a-strategy) · [Data](#data) · [Project Structure](#project-structure)

---

## Quick Start

```bash
pip install -e .
# + the package for your chosen classification backbone
```

FFM separates **learning the market** from **deciding a trade**. Stage 1 pretrains a backbone on raw OHLCV (self-supervised); stage 2 finetunes a thin per-strategy classifier on top, validated on the honest ruler.

```python
from futures_foundation.finetune import ssl, wf, produce

# 1) PRETRAIN (BERT stage) — self-supervised on raw OHLCV, runs on a GPU (Colab).
#    Output: an adapted backbone checkpoint that downstream classifiers start from.
ssl.loop_ssl(data_dir='…', out_path='ssl_ohlcv.pt')            # see colab/ runner

# 2) VALIDATE — walk-forward honest ruler with overfit→Optuna; classifier-agnostic.
verdict = wf.loop_streamed(make_labeler, streams,
                           clf_kwargs={'backbone_ckpt': 'ssl_ohlcv.pt'})

# 3) PRODUCE — only if it generalizes; trains on all data minus a holdout → ONNX bundle.
if verdict['generalizes']:
    produce.train_final_streamed(make_labeler, streams)
```

→ Labeler contract: [Add a Strategy](#add-a-strategy) · How validation catches & fixes overfitting: [The Training Loop](#the-training-loop--overfit-driven)

---

## Philosophy — BERT for futures

> Separate **"understanding market context"** from **"making strategy-specific decisions."**

Just as BERT learns language structure from unlabeled text before being finetuned for sentiment or Q&A, the FFM backbone learns **regime, structure, and volatility** from unlabeled futures OHLCV before any strategy logic runs. A strategy then adds only what the backbone cannot derive — setup geometry, entry distance, risk sizing — and finetunes a light classification head. Market-context knowledge is learned once and shared across every strategy.

Two principles shape everything below:

1. **The backbone is a pretrained foundation model designed for *classification*** — and ingests **multivariate raw OHLCV** (price + volume + range), so it can encode participation and volatility, the raw material of momentum compression → expansion.
2. **The backbone is swappable.** FFM commits to an *interface*, not a model. Any pretrained classification foundation model plugs in behind the same seam without touching a single strategy.

---

## Overview

The flow is two stages over one shared backbone:

```
raw OHLCV (9 tickers × 1/3/5/15min)
        │
        ▼  Stage 1 — SELF-SUPERVISED PRETRAIN  (finetune/ssl.py, GPU/Colab)
   masked modeling: mask bars → reconstruct  ──►  adapted backbone checkpoint
        │                                          (learns regime / structure / volatility)
        ▼  Stage 2 — FINETUNE  (finetune/wf.py → produce.py)
   strategy labeler + light classifier head  ──►  ONNX bundle the bot loads
```

- **Honest by construction.** Every result passes the honest ruler: walk-forward × {REAL, SHUFFLE, RANDOM} with an **overfit→Optuna** loop and a pre-registered PASS/FAIL auto-verdict. A number is believed only if REAL clearly beats every control, fold after fold.
- **2026 is a reserved out-of-sample year** — excluded from *both* pretraining and the rolling walk-forward, so the final OOS is never contaminated.
- **Causal by contract.** Every feature/window is strictly causal (streaming == batch, per bar); the leak audit is mandatory.

---

## Self-Supervised Pretraining (the BERT stage)

**`futures_foundation/finetune/ssl.py` — generic *masked modeling* on raw OHLCV (literally BERT's pretext), adapting the backbone to our markets with the same generalization discipline as the strategy pipeline.**

A random fraction of bars in each OHLCV window is **masked**, and the encoder must **reconstruct the masked bars from their surrounding context** (MSE on the masked positions only). To fill a gap, the model has to know what *normally* comes next given the local regime — so it learns volatility, structure, and compression→expansion dynamics. (Masked positions are filled with noise rather than zeroed, so the backbone's per-patch instance-norm never divides by zero.) It runs GPU-maximized (data resident on GPU, large batch, AMP).

We chose masked modeling over contrastive learning deliberately: contrastive proved **gameable** here — shuffled and noise windows scored as well as real ones, so the loss wasn't certifying that real temporal structure was learned. Masked reconstruction has a clean, shortcut-proof control (below).

It is **overfit-gated and Optuna-tuned**, mirroring the strategy pipeline:

| Gate | What it checks |
|---|---|
| Time-split val early-stop | generalizes forward in time (2026 excluded) |
| **REAL vs SHUFFLE vs RANDOM reconstruction** | the decisive pretext control: REAL must reach a far lower masked-reconstruction loss than time-shuffled and pure-noise windows. Observed REAL ≈ 0.35 vs SHUFFLE/RANDOM ≈ 0.97/0.98 → the encoder is reconstructing from genuine temporal structure, not a copy trick |
| NaN / variance guard | masked bars noise-filled so per-patch instance-norm stays finite |
| Optuna | if it doesn't generalize, tune lr / weight-decay / mask-ratio / channel-adapter width until it does |
| Linear probe (soft) | a linear probe reads regime / vol / structure off the frozen embedding — *informative but non-discriminating on its own* (generic domain-adaptation can pass it even for a noise-trained backbone). The **reconstruction control above** is the valid temporal test; the **downstream walk-forward A/B** is the decisive one |

---

## The Classifier Seam — model-agnostic

`futures_foundation.finetune.classifier` is the swap point: a `Classifier` ABC + a `get_classifier(name, **cfg)` registry. A strategy pipeline references a classifier **by name**; the backbone behind it can change with no strategy edits.

```python
from futures_foundation.finetune.classifier import get_classifier

clf = get_classifier(BACKBONE, backbone_ckpt='ssl_ohlcv.pt', ft_mode='partial')
```

Two ways to attach the backbone — both initialize from the SSL checkpoint via `backbone_ckpt`, both run torch in an **isolated subprocess** (the parent stays torch-free, so torch never collides with other native libraries in one process):

- **End-to-end fine-tune** — foundation model + per-strategy channel adapter + light head, all trained together. Maximum capacity; the backbone specializes to the task.
- **Frozen head-only** — embed each window **once** through the frozen encoder, then train a cheap **logistic or MLP head** per fold on the cached embedding (optionally concatenated with hand-crafted geometry features). This is the "embed once → head per fold" pattern: fast enough to iterate on local hardware, and a clean linear/​shallow probe of what the representation actually carries.
  - **Cross-run embedding cache** — the frozen embedding is deterministic in `(backbone_ckpt, bars, window spec)`, so it's cached to disk keyed on exactly those. The expensive embed cost is **paid once per backbone**: reruns, head swaps (logistic↔MLP), and interpretability checks reuse the cached vectors instead of re-embedding. `EMBED_CACHE=0` disables; `EMBED_CACHE_DIR` relocates it.

**Currently supported:** one pretrained classification backbone (installed via its own package), in both attach modes; **additional pretrained classification foundation models are planned behind the same interface.**

- **`logistic`** — a torch-free baseline / test vehicle for the whole pipeline.
- **Add your own backbone** by implementing `featurize()` + `fit_predict()` and registering it — the walk-forward, produce, and ONNX paths are all classifier-agnostic.

---

## Finetuning Pipeline — walk-forward → produce

**`futures_foundation/finetune/` — the strategy-pluggable harness: streamed walk-forward evaluator with honest-ruler controls, production trainer, ONNX export.** Every strategy goes through it; nothing about it is tied to a specific backbone.

**What it does:** a strategy labeler defines event candidates (e.g., a trend-pivot); for each event a multivariate context window → the classifier predicts `P(take)`, scored on **realized R** via the strategy's own evaluator. Validation runs the **overfit-driven training loop** on a rolling **train / validate / test** walk-forward with **REAL / SHUFFLE / RANDOM** controls and a pre-registered PASS/FAIL auto-verdict. The production trainer then fits one head on the full corpus minus the holdout and saves a single bundle + ONNX the bot loads.

| Component | Role |
|---|---|
| `wf.py` | Streamed walk-forward (`run_streamed`, `loop_streamed`) — featurize once across all streams (bounded RAM), rolling folds, VAL-selected operating point + **VAL→TEST generalization gate**, REAL/SHUFFLE/RANDOM, overfit→Optuna loop, PASS/FAIL verdict. 2026 excluded as OOS. |
| `produce.py` | Production training: one fit on the full corpus minus an N-month holdout; scores the 2026 OOS; emits the deployment bundle + signal contract + ONNX. |
| `tune.py` | Optuna search with a generalization-robust objective + held-out guard, auto-falling back to defaults unless the tuned config beats them. |
| `loop.py` | The overfit-driven loop: default WF → generalize check → Optuna only if it overfits → rerun → repeat → final full WF. |
| `_memmap.py` | Featurize-to-disk + streaming so full multi-timeframe, all-ticker runs fit in bounded RAM. |
| `classifier.py` / `classifiers/` | The model-agnostic seam (above) + backbone implementations. |

### The training loop — overfit-driven

`loop_streamed(...)` runs the whole process as one self-correcting loop. **Optuna fires only when overfitting is detected** — a config that already generalizes is left untouched:

1. Walk-forward with the **default** classifier config.
2. **Generalizes?** (VAL→TEST gap within tolerance, REAL beats controls fold-after-fold) → **keep defaults, done.**
3. **Overfit?** → **Optuna** for a config that generalizes (objective rewards cross-fold stability; auto-falls back to defaults unless the tuned config beats them on a held-out guard).
4. **Rerun**; repeat until it passes (capped — if nothing generalizes, the model is **flagged**).

Two guardrails keep it honest: the **VAL→TEST gate** (operating point chosen on *validation*, reported on *test*; an edge that decays is rejected) and tuning/selection that sees train+validation only — **test is never consulted**.

---

## Add a strategy

```python
class MyLabeler:
    n_classes = 2                               # binary selection (take / skip)
    def calendar(self): ...                     # ticker × timestamp
    def build(self, lo, hi, test_start):
        # → (contexts, labels, keys)  — keys carry realized-R per target
        ...
    def mv_contexts(self, keys):                # → [N, C, seq] multivariate windows
        ...
    def evaluate(self, keys, preds):            # → per-trade realized-R array
        ...
```

```python
from futures_foundation.finetune import wf, produce

verdict = wf.loop_streamed(make_labeler, streams,
                           clf_kwargs={'backbone_ckpt': 'ssl_ohlcv.pt'})
if verdict['generalizes']:
    produce.train_final_streamed(make_labeler, streams, export_onnx=True)
```

The labeler's `final run()` (in `finetune.base.StrategyLabeler`) applies a session-calibrated TP≥SL triple barrier (entry = next-bar open) and emits `signal_label` / `max_rr` / `sl_distance` / `direction`, centralizing the entry-after-signal / orientation bug class once for every strategy. `FoldHealthMonitor` flags per-fold pathologies (val/test gap, N-collapse, confidence-flat, zero-signal-fold); realized-R economics report PF / WR / mean-R / maxDD under a trailing exit (not optimistic MFE).

---

## Data

### Supported instruments

9 instruments: **ES, NQ, RTY, YM** (equity indices), **GC, SI** (metals), **CL** (energy), **ZB, ZN** (rates) — each at **1 / 3 / 5 / 15min**.

### Input format

```
data/
├── ES_3min.csv      # datetime, open, high, low, close, volume
├── ES_5min.csv
└── ...
```

`databento/build_continuous.py` resamples raw 1-min bars to any timeframe; `databento/append_update.py` splices new exports into `data/` continuously. A configurable `data_dir` (e.g. a Google-Drive mount on Colab) lets pretraining and finetuning read the same CSVs anywhere.

### Features

Raw OHLCV is the backbone's input. For strategies that fuse hand-crafted geometry, `futures_foundation.features.derive_features` produces instrument-agnostic (ATR-normalized), strictly causal features (bar anatomy, returns/momentum, volume dynamics, volatility, session context, market structure, HTF context) — every feature held to the no-look-ahead causal-parity rule (streaming == batch, per bar).

---

## Project Structure

```
Futures-Foundation-Model/
├── futures_foundation/                # Foundation package (torch-free to import)
│   ├── finetune/                      # ★ The model-agnostic classification pipeline
│   │   ├── ssl.py / ssl_data.py       #   SSL masked-modeling pretraining (BERT stage)
│   │   ├── _ssl_torch.py              #   GPU-max masked trainer + frozen embed_windows (subprocess/Colab)
│   │   ├── ssl_probe.py               #   linear probe: regime / vol / structure (soft signal)
│   │   ├── classifier.py              #   Classifier ABC + get_classifier registry (the seam)
│   │   ├── classifiers/               #   end-to-end FT + frozen head-only (cached embeddings) + logistic
│   │   ├── wf.py                      #   streamed walk-forward honest ruler + overfit→Optuna
│   │   ├── produce.py                 #   production trainer + 2026 OOS + ONNX + contract
│   │   ├── tune.py / loop.py          #   Optuna search + overfit-driven loop
│   │   ├── _memmap.py                 #   featurize-to-disk streaming (bounded RAM)
│   │   └── base.py / health.py        #   StrategyLabeler + FoldHealthMonitor
│   ├── extractors/                    #   Pluggable backbone interface (FeatureExtractor)
│   ├── features.py / primitives/      #   OHLCV → causal features; indicators/barriers
│   └── prepare.py                     #   raw CSVs → features parquet
├── colab/                             # ★ Colab runners: clone → install → Drive paths → run
├── databento/                         # Continuous-contract build + incremental update
├── docs/                              # Build specs + runbooks
├── tests/                             # Unit tests (pre-commit gated; torch-free by contract)
└── data/                              # Raw OHLCV CSVs (gitignored)
```
---

## License

Apache 2.0 — See [LICENSE](LICENSE) for details.

---

## Disclaimer

This software is for **research and educational purposes only**. It does not constitute financial advice. Trading futures involves substantial risk of loss. Past performance of any model does not guarantee future results.
