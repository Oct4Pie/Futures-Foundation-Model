# Corpus v3 Status

## Decision

Corpus v3 has passed one predeclared contract/session interoperability pilot. The admission is
restricted to `CL / CLK20 / 2020-04-20`; broad materialization, model training, validation,
calibration, economic scoring and deployment remain blocked.

The canonical contract is
[`config/corpus_v3/contract.json`](config/corpus_v3/contract.json). The reproducible inventory is
[`config/corpus_v3/coverage_audit.json`](config/corpus_v3/coverage_audit.json).
The pilot evidence is
[`config/corpus_v3/pilot_verification.json`](config/corpus_v3/pilot_verification.json).

## Representative shard result

AlphaForge commit `0a955b4` produced one deterministic Parquet-plus-receipt bundle. FFM then
independently reopened both sealed source leaves and reconstructed every eligible session row from
physical source ordinals before exposing a label index.

```text
request:                     CL / CLK20 / 2020-04-20 / foundation_pretraining
session:                     [2020-04-19 22:00Z, 2020-04-20 21:00Z)
verified rows:               61,150
negative trade rows:         4,255
zero-price trade rows:       42
invalid-quote rows retained: 0
nonpositive-quote rows retained: 4,314
minimum trade price:         -40.32
receipt SHA-256:             7595f40f263817f3b8d8112c59fc7fa04723a8344beb13210de970b5cf93c987
physical Parquet SHA-256:    9a6413635f7d937e3d703344e7c9091872fa18d188669491ccfcd16813a966dd
semantic shard SHA-256:      f5c5e7ab6f1aece30ebc5738af2b3274785f2d21530e89524bd51d943c3f8dce
```

This proves the bounded export/verifier seam. It does not prove full-lake session/calendar
coverage, live-arrival equivalence, or readiness to train.

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
coverage report payload: db0dd303d23f3d8adf5e35b45dcf592a3ca9a75d129b9a96a261523fb7886047
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

## Broad-materialization blocker

The current AlphaForge `session_store_v6` must not be called as a Corpus v3 export API. It has a
narrow purpose/date/window contract, does not retain complete UTC/source-row lineage, can synthesize
missing `event_seq`, and selects a session contract using activity through a fixed anchor.

AlphaForge now owns a `foundation_training` streaming export that emits unspliced contract-day
rows. Each row retains:

```text
timestamp_utc_ns, time_us, event_seq,
price, bid, ask, quote_valid,
volume, bid_volume, ask_volume,
source_file_index, source_row_ordinal
```

Each contract-day shard must separately bind scalar metadata:

```text
root, contract_id, session_day,
session_start_utc_ns, session_end_utc_ns,
coverage_start_utc_ns, coverage_end_utc_ns,
export_receipt_sha256, source_shard_sha256,
source_file_table_sha256, corpus_contract_sha256,
environment_receipt_sha256, instrument_spec_sha256,
tick_size, tick_value
```

The physical shard may replace the repeated per-row hash with `source_file_index`; the receipt must
bind that index to the exact source path and SHA-256.

The receipt-v2 contract binds the request, roots, dates, window, producer/governance hashes,
lake hash, selected leaf hashes, exclusions, row counts, output shard hashes, internal-gap/session
evidence, preservation of valid negative prices, and preservation of valid trade rows when the
attached quote is invalid. Missing or duplicate `(timestamp_utc_ns, event_seq)` keys must be
rejected. FFM verifies the receipt, output bytes, pinned raw leaves, normalization, economics,
calendar, all eligible source rows and physical lineage before deriving a bar or label.

FFM will then construct two separate views:

1. Contract-native pretraining streams with no roll selection or splicing.
2. A separately admitted downstream front-contract view whose selection cutoff never exceeds the
   decision time and whose windows and labels never cross a contract change.

## Strict label engine status

[`futures_foundation/tick_path_labels.py`](futures_foundation/tick_path_labels.py) implements the
strict `ffm_ordered_tick_path_labels_v2` semantics against synthetic governed rows. It now:

- Rejects missing or duplicate event keys and duplicate source-file/row lineage.
- Enters at the first lexicographic event strictly after the decision key.
- Requires a hash-bound decision/risk manifest and rejects supplied risk-known keys after the
  decision. This binds caller assertions; it does not prove the feature generator was causal.
- Requires supplied receipt, shard, source-table, environment and Corpus-contract hashes. The
  engine validates their form and binds them into the artifact; the production export verifier
  independently proves their content before exposing real rows to the label engine.
- Requires declared source coverage through the horizon and rejects stale entry/endpoints.
- Preserves valid negative futures prices and performs exact barrier comparisons in integer ticks.
- Converts fractional R targets to ticks with decimal multiplication and conservative ceiling.
- Separates observed trade-path targets from marketable-at-trade bid/ask proxies.
- Preserves valid trades when quotes are invalid and masks only the quote-derived path.
- Never reuses the entry event as an exit or barrier observation.
- Uses the actual observed quote on gap-through exits instead of clipping losses to the barrier.
- Emits entry, terminal and barrier-touch event/source lineage.
- Uses the declared wall-clock endpoint, never the last observed tick, as purge authority.
- Produces separate semantic and provenance-bound artifact fingerprints.
- Retains independent reference and indexed backends with randomized parity tests.
- Writes canonical array bundles that verify manifest, file, semantic and artifact hashes on load;
  tampered arrays fail closed.
- Keeps fees and added slippage outside the gross label artifact.

The engine consumed the verified CLK20 pilot through a production capability. Direct mapping
construction is now a private synthetic-test surface. Production label construction and bundle
write/load require the matching verified export identity. This is proven for the pilot, not yet for
a multi-root corpus materialization.

## Economics still blocked

The eventual tick labeler must separate:

- Observed trade-path MFE, MAE, barrier order and time-to-barrier.
- Marketable-at-trade quote proxies using ask for long entry/bid for long exit, reversed for shorts.
  These are not continuous-quote or fill evidence.

Entry begins at the first strict event key after the decision; the signal-forming event and entry
event cannot be reused as an exit observation. A horizon without endpoint coverage is invalid
rather than silently truncated. “Exact” means first touch among verified observed trade records;
it is not a claim of continuous market observation between records.

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

The representative seam, raw reconstruction and label-index boundary now pass. The remaining list
must pass at corpus scale before training is authorized.
