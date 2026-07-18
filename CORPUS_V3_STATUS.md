# Corpus v3 Status

## Decision

Corpus v3 is admitted for a deterministic, outcome-blind inventory audit only. It is not admitted
for tick materialization, labels, model training, validation, calibration, economic scoring or
deployment.

The canonical contract is
[`config/corpus_v3/contract.json`](config/corpus_v3/contract.json). The reproducible inventory is
[`config/corpus_v3/coverage_audit.json`](config/corpus_v3/coverage_audit.json).

## Verified inventory

- Source: `sc_historical_ticks_v2`, raw ticks only, strictly before `2026-01-01`.
- Candidate governed roots: 43. M6J remains blocked by the source admission artifact.
- Full tick inventory: 1,733,652 files and 17,378,601,600 rows.
- Admitted pre-cutoff inventory: 1,216,047 files and 15,594,891,123 rows.
- Admitted contract symbols: 4,343 across all manifest dates and 4,275 before the cutoff.
- Every registry, admission, QA, loader, economics, calendar, coverage and lake artifact hash passed.
- Eighteen semantic cross-bindings among those documents passed.
- The compressed 104 MiB leaf-hash manifest is pinned directly, in addition to its external
  hash-of-hashes.
- Duplicate admitted `(contract, UTC bucket)` rows and coerced/non-integer counters fail closed.
- The report rebuilds deterministically and contains no generation timestamp in its hashed payload.

Current internal hashes:

```text
coverage report payload: 772cb3791fd780b5980879ca08e0744650de387b61440f1d61dab583908d601b
lake hash-of-hashes:      632f859e138d6ad22b67801d8279ac42363a8ededabf79dbb767eb1d649baffe
leaf-manifest file:       367d5847ccfeae04e085d9fd55664bc6e5f52a2d56be686d1d25e43f46663928
```

## Why no roots are selected yet

The coverage manifest is organized by UTC file date, not by root-specific exchange session. A UTC
bucket is therefore not a valid denominator for session coverage, continuity or liquidity across
equity, energy, FX, rates, metals, crypto and agriculture products. Full-day top-contract activity
also cannot define a causal active contract for an earlier decision.

The report deliberately has:

```text
candidate_roots: 43
selected_roots:  0
selection_status: blocked_pending_sessionized_foundation_export
```

Its five diagnostic flags are report-only. They do not admit or reject MCL, MET, MHG, MNG or ZT.

After the streaming export exists, the root universe will have two frozen tiers:

- `core_comparable`: identical sessionized rows used for cross-family ranking and validation.
- `supplemental_pretraining`: training-only contract/year shards that cannot influence validation,
  thresholds, promotion or OOS composition.

## Materialization blocker

The current AlphaForge `session_store_v6` must not be called as a Corpus v3 export API. It has a
narrow purpose/date/window contract, does not retain complete UTC/source-row lineage, can synthesize
missing `event_seq`, and selects a session contract using activity through a fixed anchor.

AlphaForge must own a new `foundation_training` streaming export that emits unspliced contract-day
rows. Each row must retain:

```text
timestamp_utc_ns, session_day, time_us, event_seq,
price, bid, ask, volume, bid_volume, ask_volume,
contract_id, source_path, source_file_sha256, source_row_ordinal
```

The export receipt must bind the request, roots, dates, window, loader/config/governance hashes,
lake hash, selected leaf hashes, exclusions, row counts and output shard hashes. Missing or duplicate
`(timestamp_utc_ns, event_seq)` keys must be rejected. FFM must verify the receipt before deriving a
single bar or label.

FFM will then construct two separate views:

1. Contract-native pretraining streams with no roll selection or splicing.
2. A separately admitted downstream front-contract view whose selection cutoff never exceeds the
   decision time and whose windows and labels never cross a contract change.

## Labels and economics still blocked

The eventual tick labeler must separate:

- Observed trade-path MFE, MAE, barrier order and time-to-barrier.
- Executable marketable-quote outcomes using ask for long entry/bid for long exit, reversed for
  shorts.

Entry begins at the first strict event key after the decision; the signal-forming event cannot be
reused. A horizon without endpoint coverage is invalid rather than silently truncated.

Zero added slippage and zero added delay remain the primary ruler, with frozen one-tick sensitivity.
Spread and instrument fees are not zero. The existing fees are static approximations, so historical
economic promotion remains blocked until an effective-dated fee schedule is pinned. Gross path
labels must remain separate from costs.

## Verification required before training

1. AlphaForge export cold/warm/concurrent determinism.
2. Selected-leaf rehash against the pinned leaf manifest.
3. Strict event ordering, DST/session identity and source-row lineage.
4. Prefix-invariant causal contract selection for the downstream view.
5. Half-open bar construction and same-timestamp sequence tests.
6. Exact endpoint coverage, barrier order, gaps and marketable-quote outcomes.
7. Split-boundary purge by complete label end.
8. Proof that excluded holdout changes cannot alter training rows, scalers, rosters or hashes.
9. Deterministic representative-subset rebuild and byte comparison.
10. Separate MantisV2 training-surface, resume and export admission.

Until those pass, training remains prohibited.
