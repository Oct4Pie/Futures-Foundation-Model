# Corpus v3 Scale Admission Plan

## Decision

The verified `CL / CLK20 / 2020-04-20` bundle admits one producer/verifier/label-index seam. It
does not admit broad materialization or any model training. Scaling uses a new contract and never
widens or rewrites the successful pilot contract.

The next gate is an outcome-blind schema/calendar matrix. No strategy event, label, R, profit
factor, win rate, model prediction or holdout outcome may influence this matrix or root admission.

## Current evidence

- Pilot producer: AlphaForge `0a955b4`.
- Scale-safety branch: AlphaForge `corpus-v3-scale-safety` at local commit `27cfb0c`.
- Scale-safety exporter changes: full verification before atomic publication, destination-keyed
  staging, stale-stage scavenging under the exact destination lock, and post-rename recovery.
- Process tests: 14 exporter tests and 20 exporter-plus-coverage tests pass.
- Session-denominator tests at producer commit `6db33be`: 27 focused tests pass. The commit adds
  strict non-finite-JSON rejection and lexical/final symlink rejection to the earlier reviewed
  mechanism.
  Two independent adversarial reviews cleared the implementation after trust-boundary repairs.
- Evidence commit `27cfb0c` seals two independently reviewed, fail-closed snapshots. CME official
  v2 binds 25 inherited responses and 28 scoped notice/report facts; every fact is explicitly
  `production_admitted=false` (manifest SHA-256
  `15356ae97578112e0d6740556a009cf57bf538bf11c5f31efea8484cd411ea41`). CME holidays v2
  binds nine archived response entities and 196 ZIP members as a source-byte inventory only
  (manifest SHA-256 `3a9e8eeb668f7e760d860e47aedf2f94fd3b0f23ece8404b27d03860bc70f475`,
  inventory SHA-256 `71574f769aa4e52717d861f630d2f1cdf39bce1f8121e66ff042b03226dcb9f5`).
  The complete four-bundle evidence suite passes 110 tests.
- Full AlphaForge suite: 301 pass; the same three pre-existing burn-ledger/trap-oracle failures
  remain and are unrelated to the Corpus v3 scale branch.
- The independent FFM session-denominator verifier is accepted as mechanism evidence. The first
  contract-lifecycle implementation was adversarially rejected despite its focused and full-suite
  tests passing; it remains uncommitted and cannot authorize materialization. The first consumer-
  only plan and source-inventory prototypes were also rejected: syntactically valid but nonexistent
  leaves, relabelled OOS paths and scope widening were constructively accepted. They are not
  evidence and will not be committed. The authority chain must begin at the producer's verified
  governance/source-root boundary.
- Coverage report: rebuilt deterministically against the current pilot contract; still selects
  zero roots pending a sessionized liquidity matrix and scale admission.

The scale-safety AlphaForge commits are local-only. They are research evidence, not durable producer
provenance, until pushed to an authorized remote or stored in a hash-pinned source archive.

The denominator implementation is admitted as a mechanism, not as market-calendar evidence.
Production rules, source artifacts, all 43 effective-date histories, exceptional sessions, the
complete pre-OOS artifact and cross-repository consumption are still blocked.

FFM now has an independent outcome-blind consumer verifier. It does not import AlphaForge or a
calendar package, accepts the existing pretty consumer contract by exact physical hash, and
reopens/reverifies a denominator at the materialization boundary rather than retaining mutable
parsed capability state. This is mechanism evidence only because no production 43-root rules,
scope or denominator artifact exists.

Official effective-dated rules are authoritative for normal geometry. The pinned calendar library
is only an independent open/closed and early/late-exception diagnostic because it backfills modern
hours into older years and still reports the removed US-equity maintenance pause after 2021.

## Required scale artifacts and dependency order

The trust order is fixed:

```text
producer governance + frozen split/use contract
    → provider metadata-only contract candidate universe
    → official lifecycle/provider-identity capability
    → official session denominator
    → expected request denominator (before observing leaves)
    → lifecycle-authorized direct leaf probing
    → split-scoped observed inventory
    → exact materialization plan
    → use-specific request authorization
    → exporter receipt binding plan SHA + request ID
```

Neither an inventory nor a plan authorizes direct raw-filesystem access. A consumer-supplied hash
is byte consistency, not authority. Every downstream capability reopens and reverifies its parent,
and the producer rejects any export request that is not an exact admitted plan member. Lifecycle
never binds a later plan; the plan binds the earlier lifecycle and expected denominator.

The four final scale artifacts below are required, but they consume five earlier canonical
authority artifacts: `provider_candidate_universe_v1`, `contract_lifecycle_capability_v1`, the
already separate session denominator, `expected_request_denominator_v1`, and
`split_scoped_leaf_inventory_v1`. No final artifact can replace those parents with copied hashes
or caller-authored rows.

1. `scale_contract_v1.json`
   - References the immutable pilot contract and evidence.
   - Pins producer, verifier, calendar, timezone, registry, economics and leaf-manifest identities.
   - Authorizes only plan-bound materialization.
   - Explicitly denies training, validation, scoring and deployment.
2. `materialization_plan_v1.json`
   - Is generated only after inventory and lifecycle admission.
   - Contains the deterministically recomputed exact sorted contract/session requests and exact
     source-file set per request; callers cannot choose a subset.
   - Binds each request to one non-overlapping storage partition and permitted uses.
   - Contains no holdout requests and no strategy-derived values.
   - Supplies a use-specific exact-member authorization API, not a raw path.
3. `materialization_manifest_v1.json`
   - Contains exactly one result per planned request, no extras or duplicates.
   - Binds receipt, physical, semantic, source-table, environment and instrument hashes.
   - Records failures and quarantines explicitly; partial completion cannot advance.
4. `scale_admission_report_v1.json`
   - Binds the contract, plan, completed manifest and independent verification reports.
   - May authorize data materialization only.
   - Cannot authorize a model training surface.

The provider candidate universe contains metadata identities only—no lake paths, availability,
rows, prices, volume or strategy outcomes—and must have complete provider query/pagination proof.
The expected request denominator is the exact lifecycle × session × split intersection and is
frozen before probing. Missing, unsafe and zero-activity leaves remain explicit rows; they never
shrink lifecycle eligibility or the denominator.

The producer-side source capability and observed inventory additionally require canonical path-to-
root/contract/session/interval binding, contiguous source indexes, exact authorized-day closure,
real stat/hash verification on the same no-follow opened inode, generator/environment/governance
provenance, and proof that unrequested/OOS paths were never statted, listed or opened. Full UTC-day
leaves can straddle a split boundary. Such a leaf is `boundary_blocked` and must not be opened or
hashed unless a separately governed custodian creates a hash-pinned split-safe namespace. If the
current lake cannot provide that proof, scale inventory remains blocked rather than post-filtered.

All four formats reject unknown fields, unsafe paths, duplicate requests, noncanonical JSON and
unscoped hashes.

## Non-overlapping partitions

Physical sessions are exported once:

```text
pretrain_only:  [2011-01-01, 2019-07-01)
shared_train:   [2019-07-01, 2024-07-01)
development:    [2024-07-01, 2025-07-01)
legacy_holdout: excluded; never materialized by this program
```

`shared_train` may feed both foundation pretraining and supervised training. Contexts, labels,
normalizers and scalers cannot cross partitions. Purging uses the complete declared label-end
timestamp. A split-scoped eligible-leaf hash—not a full-lake hash containing excluded holdout
files—defines training identity. A separate full-lake hash remains provenance only.

## Contract lifecycle is a separate denominator

The root/session denominator answers only when the venue and product family are officially open.
It must not be used to infer when a dated contract first traded or ceased trading. Corpus v3 needs
a second canonical lifecycle artifact bound to the materialization plan and calendar-rules hash.

Each admitted dated contract must declare, from official exchange evidence:

```text
contract_id, root,
start_kind = official_exact | left_censored_at_scope_start,
official_eligibility_start_utc_ns,
end_kind = official_exact | right_censored_at_scope_end,
official_trading_end_exclusive_utc_ns,
official_contract_source_ids,
provider_symbol_identity_source_id,
disposition
```

Exact start is inclusive and exact end is explicitly half-open. Censoring is allowed only at a
verified selected-scope edge and requires official evidence that eligibility extends across that
edge. The eligible contract window is the exact intersection of those lifecycle bounds with every
root/session segment, and each clipped segment is hashed into the lifecycle artifact. The first or
final session may therefore be truncated; the exporter may not silently request a full venue
session. A request outside that intersection rejects before raw-file discovery. Missing or zero
activity inside an eligible interval is measured later by the liquidity matrix; observed files,
ticks, first/last activity, volume and liquidity may never define lifecycle eligibility.

Unknown provider identity is an explicit quarantine, not an inferred mapping. In particular,
pre-2017 Sierra `RTY` cannot be assigned CME RTY identity merely because the root string matches.
Contract lifecycle also does not choose a roll: liquidity measurement and any roll/continuation
policy consume admitted contract-day shards later and remain separate from the physical export.

The producer and FFM consumer must independently verify the lifecycle artifact, the provider
candidate-universe capability, an admitted claim-scoped evidence registry, split/OOS exclusion and
every root/session intersection. The later expected denominator and plan bind this lifecycle; the
lifecycle never consumes either. No month-code expiry heuristic or observed first/last file or tick
date is admissible. The lifecycle API exposes immutable admitted intersections only; quarantined
rows never produce an export capability.

## Outcome-blind schema matrix

### Tier A — all-root fall-DST canary

Session day `2023-11-06`, split `foundation_pretraining`, purpose
`foundation_schema_matrix`:

```text
6A=6AZ23   6B=6BZ23   6C=6CZ23   6E=6EZ23   6J=6JZ23   6S=6SZ23
BTC=BTCZ23 CL=CLZ23  ES=ESZ23   ETH=ETHZ23 GC=GCZ23   HG=HGZ23
HO=HOZ23  M2K=M2KZ23 M6E=M6EZ23 MBT=MBTZ23 MCL=MCLZ23 MES=MESZ23
MET=METZ23 MGC=MGCZ23 MHG=MHGZ23 MNG=MNGZ23 MNQ=MNQZ23 MYM=MYMZ23
NG=NGZ23  NKD=NKDZ23 NQ=NQZ23   RB=RBZ23   RTY=RTYZ23 SI=SIZ23
SIL=SILH24 TN=TNZ23  UB=UBZ23   YM=YMZ23   ZB=ZBZ23   ZC=ZCZ23
ZF=ZFZ23  ZL=ZLZ23   ZM=ZMZ23   ZN=ZNZ23   ZS=ZSX23   ZT=ZTZ23
ZW=ZWZ23
```

Two positive-size source containers were verified for every request. Expected non-agriculture
bounds are `[2023-11-05T23:00Z, 2023-11-06T22:00Z)`. Agriculture is not one continuous envelope:
its declared segments are `[2023-11-06T01:00Z, 13:45Z)` and
`[2023-11-06T14:30Z, 19:20Z)`.

### Tier B — spring-DST cross-family canaries

Session day `2024-03-11`:

```text
ES/ESH24  CL/CLJ24  GC/GCJ24  6E/6EH24  ZN/ZNH24  BTC/BTCH24  ZC/ZCH24
```

Expected non-agriculture bounds are `[2024-03-10T22:00Z, 2024-03-11T21:00Z)`. Agriculture uses
`[2024-03-11T00:00Z, 12:45Z)` and `[2024-03-11T13:30Z, 18:20Z)`. Required source containers
exist for all seven.

### Tier C — historical schema and multiplier canaries

```text
2012-03-12: ES/ESH12, CL/CLJ12, GC/GCJ12, 6E/6EH12, ZN/ZNH12, ZC/ZCH12
2018-03-12: BTC/BTCH18
2020-04-20: CL/CLK20 negative- and zero-price preservation control
```

These cover price multipliers `1`, `10`, `.01` and `100`, fractional-rate ticks, agriculture and
post-inception crypto.

### Tier D — split boundaries

```text
ES/ESU19/2019-07-01  supervised-training start
ES/ESU24/2024-06-28  final regular training week
ES/ESU24/2024-07-01  first development session; pipeline checks only
```

A training request for `2024-07-01`, every request on or after `2025-07-01`, and every request on
or after the source cutoff must reject before raw access.

### Tier E — fail-closed calendar controls

For representatives `ES, CL, GC, 6E, ZN, BTC, ZC`:

- `2023-11-24` rejects until an exact early-session override is pinned.
- `2023-12-25` rejects until an exact sourced closed-session override is pinned.
- `2023-11-05` is an explicit normal-weekly closed denominator row; raw export rejects it.
- Dates outside calendar coverage reject.
- Synthetic override tests must assert exact UTC nanosecond bounds.

## Per-shard promotion checks

- Two cold builds are byte-identical; a warm build performs no raw-row read.
- Independent-process cold contention performs exactly one build.
- Hard death during staging and after rename recovers without an invalid visible cache or litter.
- Receipt, raw leaves, schema, provenance, multiplier, tick grid and all output bytes rehash.
- FFM independently reconstructs every row from the pinned physical source ordinals.
- Event and lineage keys are strict and unique; session bounds are half-open.
- Valid trades survive invalid or nonpositive BBO-at-trade values.
- Only declared market-break segments are exempt from gap alarms.

## Remaining blockers

1. Create the production source-artifact table and exact root listing/effective dates; do not count
   pre-inception sessions. Resolve or quarantine the pre-2017 `RTY` provider identity.
2. Populate independently reviewed historical product rules and complete exceptional-session
   overrides. Domestic US equity and NKD require separate products. Grain/oilseed requires four
   regimes across 2012, 2013 and 2015 boundaries. Metals/energy require the 2015 close transition.
   Every weekday closure and library-flagged early/late session must have a source-backed override;
   fixture evidence is not production proof.
   The repaired listing/regime snapshot is suitable only as a hash-pinned evidence input. It uses
   bounded modality-aware excerpts and conservative notice/report language, but all 28 facts remain
   production-barred pending post-effective corroboration and exact root/product scope. The holiday
   byte inventory covers annual official CME archives for 2011-2017 and 2019 only; 2018 and
   2020-2025H1 retain explicit source gaps, including 2023 H2, all 2024 and 2025 H1.
3. Generate the contract-derived complete 43-root denominator through the end of development,
   excluding the reserved OOS interval by construction.
4. Measure full-range wall time, peak RSS, artifact bytes and reload verification, then run the
   complete schema/DST matrix.
5. Make AlphaForge export consume exact denominator segments and make FFM independently verify
   full root-date coverage and segmented raw membership.
6. Complete the provider candidate-universe capability, then rebuild and independently review the
   contract lifecycle with exact official eligibility and half-open trading-end semantics. Only
   afterward may the expected denominator, inventory and plan be built. The rejected first
   implementations are not evidence.
7. Push or archive AlphaForge `27cfb0c` durably.
8. Implement and independently review the four scale artifacts above.

Only after those gates and the complete schema matrix pass may the project build the sessionized
liquidity/continuity denominator and freeze `core_comparable` versus `supplemental_pretraining`
roots. Model-family training admission remains a later, separate gate.
