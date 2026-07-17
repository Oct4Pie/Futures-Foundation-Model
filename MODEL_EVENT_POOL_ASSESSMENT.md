# Model Event-Pool Assessment

Date: 2026-07-16
Purpose: choose causal, broad, sufficiently large and non-redundant supervision for futures models
Legacy confirmatory interval read by this assessment: **false**. The interval is not globally
pristine because dates in it were inspected by earlier project research.

## Decision

Proceed with the existing event generators, but restructure them as three trigger families with
multi-policy path labels:

1. ATR zigzag.
2. Confirmed k=2 fractal.
3. SuperTrend flip.

Do not treat ATR-stop versus structural-stop variants as separate samples. They are alternative
outcome policies on the same trigger. Do not treat HTF-gated variants as separate event pools; store
HTF agreement as causal metadata. Fractal-zigzag should not be a separately weighted family because
90.6% of its events exactly coincide with k=2 fractal events.

## Existing pool measurements

These are from the one-year development artifact after roll, causality and tick corrections.
`Independent contexts` greedily requires 256 source bars between samples within each stream. It is a
conservative effective-volume diagnostic, not an IID claim.

| Trigger | Raw events | Streams | Stream entropy | 1m share | Approx. independent contexts | 3R positives | 3R entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| ATR zigzag | 997,830 | 54/54 | 0.832 | 60.1% | 19,333 | 23.9% | 0.793 |
| k=2 fractal | 705,181 | 54/54 | 0.833 | 58.1% | 19,287 | 24.4% | 0.802 |
| fractal-zigzag | 408,093 | 54/54 | 0.829 | 57.8% | 18,637 | 24.6% | 0.806 |
| SuperTrend | 151,649 | 54/54 | 0.814 | 64.5% | 17,804 | 23.7% | 0.790 |

All four have balanced long/short directions and usable target class entropy. Each has enough
positive examples at 2R/3R/4R. Six-R labels are rarer but still abundant. Quantity is not the
bottleneck.

The three recommended families produce 1,672,944 exact-deduplicated events but only about 19,591
non-overlapping 256-bar contexts. That is the important result: adding dense overlapping trigger rows
does not materially increase market-context coverage. Training should use block-aware sampling and
loss weights rather than pretending raw rows are independent.

## Redundancy

- Fractal-zigzag overlaps 90.6% of the smaller k=2 fractal pool exactly.
- ATR zigzag overlaps about 19% of the smaller k=2/fractal-zigzag pools.
- SuperTrend is the most distinct: exact overlap is 23.1% with ATR zigzag, 10.2% with k=2 fractal,
  and 4.9% with fractal-zigzag.
- Stop-policy variants share triggers by construction.
- HTF gating selects a subset and should be a feature/stratum, not duplicated supervision.

## AlphaForge audit

AlphaForge contributes useful supervision contracts, not another broad validated strategy pool.

### Worth porting

Its dense path outcome module labels every causal decision time with:

- terminal return;
- upside MFE and downside MAE;
- target-before-adverse outcomes;
- time to favorable/adverse barrier;
- large up/down-tail probabilities;
- executable long/short policy outcomes;
- 15, 60 and 240-minute horizons.

On NQ 5-minute discovery data, AlphaForge materialized roughly 82,434 rows for 15/60-minute arms and
37,470 for 240-minute arms. Seventeen distributional path arms survived repeated shuffled-label and
random-feature familywise controls. The strongest reported stable example was the 60-minute 90th
percentile upside-MFE target: median fold loss improvement 30.9%, positive in all four folds, worst
fold +11.0%. This is direct evidence that path magnitude/tail structure is more learnable than signed
return.

AlphaForge did **not** establish signed-return signal, barrier-probability signal, or incremental
economic value from its tested filter policy. Its TCN did not beat LightGBM on any stable arm. We
should therefore port the label contract, not its conclusions about model architecture or trading
policy.

The causal `price_structure.py` primitives are also worth a later matched screen:

- three-bar fair-value gaps;
- confirmed swings;
- liquidity sweeps/reclaims;
- impulse-confirmed order blocks.

They expose explicit confirmation indices and have planted causality tests. Their breadth, event
counts and redundancy across our nine-symbol/six-timeframe corpus have not been measured, so they
should not enter the training mix yet.

### Not suitable as broad foundation supervision

- NQ first-hour momentum: meaningful downstream benchmark, but only one root, one session decision,
  and roughly 458 discovery events in the governed v6 configuration.
- ORB variants: session/product-specific and better reserved for downstream heads.
- Order flow/L2: unavailable apples-to-apples in the OHLCV corpus.
- Cross-asset relative value: requires synchronized multi-instrument inputs and a separate model
  contract.

## Recommended dataset contract

Materialize one row per deduplicated context, with multi-hot causal event tags and multi-task future
labels:

- Trigger tags: `atr_zigzag`, `fractal_k2`, `supertrend_flip`.
- Context tags: HTF agreement, symbol, timeframe, session clock and contract-roll distance.
- Descriptive targets: realized volatility, trend efficiency and range expansion.
- Forward path targets: terminal return, MFE, MAE, absolute move, tail events, continuation/reversal,
  and time-to-barrier at fixed wall-clock horizons.
- Policy labels: ATR and structural stop outcomes attached to the same context, not duplicated rows.
- Training weights: balance symbols, timeframes, trigger families and outcome classes; cap highly
  overlapping short-timeframe contexts per batch.
- Evaluation: embargo by maximum context plus label span; report per-symbol/timeframe and paired
  downstream realized-R lift against a causal-feature baseline.

For cross-market training, AlphaForge's fixed 16/32/64-tick NQ barriers must **not** be copied
literally. Primary MFE, MAE, terminal-return and barrier distances should be expressed in causal
ATR/volatility or R units and rounded outward to executable ticks. Raw ticks and dollars remain
auxiliary execution labels. Horizons must remain fixed in wall-clock time rather than fixed bar
counts so that 1-minute and 60-minute samples describe the same future interval.

The trigger mix also should not replace full-corpus self-supervision:

- Stages 1 and 2 continue to sample all OHLCV contexts with balanced symbol/timeframe sampling.
- Stage 3 adds dense path-distribution supervision over all eligible contexts.
- Triggered contexts are tagged and deliberately oversampled for conditional continuation/reversal
  learning, without duplicating the same context for each detector or stop policy.
- Downstream strategy heads evaluate the tagged event subsets under the executable R ruler.

This gives Stage 3 a much better objective than point candle reconstruction. Stage 2 can still learn
market-state invariances, while Stage 3 learns the path distribution that determines whether a setup
is worth taking.

## Next action

Do not add AlphaForge's narrow deterministic strategies to the broad pool yet. First implement the
dense multi-horizon path labels above across the existing nine symbols and six timeframes, then attach
the three deduplicated trigger families as conditional tags. After that, compare foundation models on
both unconditional path prediction and event-conditioned realized-R selection.
