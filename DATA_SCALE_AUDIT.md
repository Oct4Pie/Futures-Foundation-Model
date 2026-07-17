# Data Scale Audit and Corpus v3 Decision

## Verdict

The earlier statement that available data was the binding limit was wrong. The experiments used a
deliberately narrow contract even though much more historical data exists locally.

The opposite conclusion—“there is automatically 50–100× usable data and confidence intervals
will shrink 5–7×”—is also unproven. Physical bytes, tick rows and contract directories are not
independent train/evaluation examples. Admission state, liquidity, continuity, duplicate exposure,
roll construction and cross-root dependence must be measured first.

What is proven today is narrower:

> The current SSL recipes failed to add trading value on the nine-root 2019–2024 training contract.
> They have not been tested on the full admitted historical tick universe.

## Verified inventory

### Existing FFM OHLCV corpus

`data/ssl_corpus_v2_6tf/MANIFEST.json` records:

- Nine roots: ES, NQ, RTY, YM, GC, SI, CL, ZB and ZN.
- 45,022,122 one-minute output rows before the higher-timeframe resamples.
- ES/NQ/YM/GC/SI/CL/ZB/ZN beginning on June 7, 2010.
- RTY beginning on July 9, 2017.
- Per-root endpoints between April 13 and July 10, 2026.
- Resampling performed within contract only, with no forward fill.

The canonical training protocol used `[2019-07-01, 2024-07-01)`. That five-year choice was an
experiment-budget decision, not the limit of the source history.

The claim that a literal `/home/m3hdi/projects/topstepx/data/bars/ES_1min.csv` proves the span is
incorrect: no such file exists in the current source directory. The span is nevertheless verified
by the hash-pinned FFM corpus manifest.

### Sealed Sierra Chart lake

The physical store `/mnt/nvme_work/topstepx/sc_clean` is read-only and reports `886G` through
`du`. Its pinned inventory contains:

| Item | Verified value |
|---|---:|
| All tick + depth bytes | 934,557,307,776 |
| All Parquet files | 4,111,644 |
| Tick files | 1,733,652 |
| Tick rows | 17,378,601,600 |
| Depth files | 2,377,992 |
| Depth rows | 99,387,919,830 |
| Tick contract directories | 6,322 |

Contract-directory count includes deliberate zero-row placeholders and is not a count of liquid
or usable contracts.

### Governed usable scope

AlphaForge's admission artifact authorizes `sc_historical_ticks_v2` as `admitted_limited` for 43
roots, with data strictly before January 1, 2026. Those roots contain 15,594,891,123 admitted tick
rows. M6J is blocked because it has no Sierra tick data.

The 43 admitted roots are:

```text
6A 6B 6C 6E 6J 6S BTC CL ES ETH GC HG HO M2K M6E MBT MCL MES MET MGC MHG
MNG MNQ MYM NG NKD NQ RB RTY SI SIL TN UB YM ZB ZC ZF ZL ZM ZN ZS ZT ZW
```

This corrects the claim of “60+ usable roots.” More products may physically appear in the store,
but 43 currently pass the declared admission contract.

### Depth and Databento restrictions

- Sierra depth is not generally admitted for model training. Only 302 exact root/date memberships
  across ES, MES, MNQ and NQ are admitted, and only for discovery. There is no continuous date
  envelope.
- Sierra depth is market-by-price data, not MBO; it has no order IDs and cannot support individual
  queue-position claims.
- Databento MBP-1 is blocked pending mutable-overlap audit.
- The central 2026 Databento MBP-10 and MBO slices are blocked because access history is unresolved.
- Four consumed NQ MBO dates are diagnostic-only and can never serve as validation evidence.

Corpus v3 must therefore start with admitted Sierra ticks. Depth and Databento are excluded unless
their governance status changes in a separately reviewed artifact.

## What the expanded data can and cannot fix

### It can plausibly improve

- Representation diversity across asset classes and historical regimes.
- The number of pullback, compression and other causal event examples.
- Exact tick-order labeling of favorable/adverse barrier paths wherever admitted ticks are valid.
- MFE, MAE and time-to-barrier resolution below a one-minute bar.
- Statistical power if the added events provide independent calendar/root information.

Tick ordering removes OHLC same-bar ordering ambiguity. It does not prove an executable fill at the
barrier price: spread, gaps, marketability and venue execution still require an explicit fill
contract.

### It does not automatically fix

- A misaligned SSL objective.
- Feature drift between 2011 and current microstructure.
- Cross-root correlation during macro events.
- Dependence among dense overlapping windows.
- Poor liquidity in old or minor contracts.
- Roll selection and overlapping-contract leakage.
- The absence of incremental information beyond a strong causal selector.

Confidence intervals scale with effective independent sample size, not nominal rows. The claimed
20–50× event multiplier and 5–7× interval contraction must not be used in planning until the new
event corpus measures calendar-block and cross-root dependence.

## Corpus v3: smallest safe program

### Gate A — coverage and admission matrix

Before training:

1. Pin the data-source registry, admission artifact, tick hash-of-hashes, loader revision and
   instrument economics.
2. For every admitted root, measure nonempty sessions, tick counts, active-contract coverage,
   gaps, crossed-BBO filtering, roll overlaps and usable years.
3. Apply a declared liquidity/continuity screen without looking at strategy outcomes.
4. Produce the final root list from that screen. Do not assume all 43 roots qualify.

### Gate B — tick-derived multi-resolution corpus

Build bars and labels from ordered admitted ticks with:

- Contract identity retained on every row.
- Causal most-active roll selection using data available through the roll anchor only.
- No window crossing a contract boundary.
- Candidate resolutions selected from 10s, 30s, 1m, 3m, 5m, 15m, 30m and 60m after measuring
  redundancy and storage cost.
- Exact ordered favorable/adverse barrier state, MFE, MAE and time-to-barrier.
- Hash-pinned shards, source row identity, prefix-invariance tests and deterministic rebuild tests.

Sub-minute resolutions are context/label sources first. They are not automatically authorized
trading timeframes.

### Gate C — count before funding

Run the existing causal detectors over a representative, hash-pinned subset and report:

- Candidate and executed-event counts by root, year, timeframe and event family.
- Calendar-block effective sample sizes and cross-root correlation.
- Barrier-class balance and neither-event rate.
- Coverage loss from rolls, gaps, purges and fixed wall-clock horizons.
- Compute/storage throughput and a full-build estimate.

Only measured yield can justify a full materialization. “Hundreds of thousands of events” remains
a hypothesis until this gate passes.

### Gate D — supervised Mantis barrier pilot first

The first training experiment should align directly with the OOS-surviving objective:

1. Frozen vanilla Mantis + identical barrier head.
2. Causal-only features + identical barrier head.
3. Frozen Mantis + causal features + identical barrier head.
4. End-to-end Mantis fine-tuning on actual barrier outcomes with a vanilla-feature anchor.
5. Optional causal-teacher distillation only as an auxiliary loss, never as the ground-truth
   replacement.

All arms must use identical rows, path labels, head capacity, calibration and execution rules. This
removes the head/objective confound in the completed OOS comparison. Promotion requires paired
incremental utility over the causal-only arm, not merely better classification AUC.

### Gate E — scaled SSL second

Only after Gate D establishes whether trainable Mantis can add information should one scaled SSL
branch be funded. It must:

- Use the deployment context contract and broad-root balanced sampling.
- Preserve the vanilla representation with an anchor.
- Optimize the predeclared forward path objective or a separately justified SSL objective.
- Compare direct-from-vanilla against any staged lineage.
- Pass representation, causal-incremental and economic gates before receiving a second seed.

This is a scaling test, not permission to restart a model-family tournament or Optuna sweep.

## Time splits and evaluation status

The existing split remains authoritative until Corpus v3 freezes a replacement:

```text
Foundation/development training: before 2024-07-01
Development evaluation:         2024-07-01 through 2025-06-30
Read legacy OOS:                2025-07-01 through 2026-04-13
Prospective confirmation:       data not used for training/validation after the new model freezes
```

The legacy OOS interval has already been read and cannot confirm a newly designed Corpus v3 model.
It may be reported as a spent diagnostic only. A new model needs confirmation data that remains
unused by its training, validation, objective choice, threshold choice and finalist selection.

Old history is suitable for pretraining and label diversity, but recent chronological folds must
control promotion. Breadth does not eliminate regime drift or justify random splits.

## Verified sources and hashes

| Source | SHA-256 / anchor |
|---|---|
| FFM six-timeframe corpus manifest | `d7cb5f4273e72e30642019f91a0581da7aafc27c342813c09be0fb5c4fc97865` |
| AlphaForge data-source registry | `f582c8f54f29077dc959e7ad3ab09befc980a5a00ef5347f68085d3112b4bf81` |
| AlphaForge admission roadmap | `aa9bf140b620d79b269fa33328d7fc5abf98d4e77a8c8193838fbf4831ee6c69` |
| Sierra tick admission artifact | `44ad9aa3bcdcd7aa14f7483d0763621078cfdc88cbf861266c77ccd5a8324c71` |
| Full lake hash-of-hashes | `632f859e138d6ad22b67801d8279ac42363a8ededabf79dbb767eb1d649baffe` |
| File-summary document | `9a846830c59213642cff5adac94264a62b25c80558f00ff600c46c613cd641ab` |
| Coverage audit document | `a58ea235da6c100c37c3cb8dff29b5a0783c40894df2f109ebcd929c2461f596` |

## Decision

Do not fund another small-corpus SSL variant. Do fund the read-only Corpus v3 coverage/event-yield
audit. If measured scale is materially larger and passes governance, run the identical-head
supervised barrier pilot before a single scaled SSL branch.
