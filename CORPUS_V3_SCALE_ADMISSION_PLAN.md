# Corpus v3 Scale Admission Plan

## Decision

The verified `CL / CLK20 / 2020-04-20` bundle admits one producer/verifier/label-index seam. It
does not admit broad materialization or any model training. Scaling uses a new contract and never
widens or rewrites the successful pilot contract.

The next gate is an outcome-blind schema/calendar matrix. No strategy event, label, R, profit
factor, win rate, model prediction or holdout outcome may influence this matrix or root admission.

## Current evidence

- Pilot producer: AlphaForge `0a955b4`.
- Scale-safety branch: AlphaForge `corpus-v3-scale-safety` at local commit `9f4c4bd`.
- Scale-safety exporter changes: full verification before atomic publication, destination-keyed
  staging, stale-stage scavenging under the exact destination lock, and post-rename recovery.
- Process tests: 14 exporter tests and 20 exporter-plus-coverage tests pass.
- Full AlphaForge suite: 281 pass; three pre-existing burn-ledger/trap-oracle failures remain.
- FFM suite: 1,105 pass and 96 skip.
- Coverage report: rebuilt deterministically against the current pilot contract; still selects
  zero roots pending a sessionized liquidity matrix and scale admission.

The scale-safety AlphaForge commit is local-only. It is research evidence, not durable producer
provenance, until pushed to an authorized remote or stored in a hash-pinned source archive.

## Required scale artifacts

Four separate canonical artifacts are required:

1. `scale_contract_v1.json`
   - References the immutable pilot contract and evidence.
   - Pins producer, verifier, calendar, timezone, registry, economics and leaf-manifest identities.
   - Authorizes only plan-bound materialization.
   - Explicitly denies training, validation, scoring and deployment.
2. `materialization_plan_v1.json`
   - Contains exact sorted contract/session requests and expected source leaves.
   - Binds each request to one non-overlapping storage partition and permitted uses.
   - Contains no holdout requests and no strategy-derived values.
3. `materialization_manifest_v1.json`
   - Contains exactly one result per planned request, no extras or duplicates.
   - Binds receipt, physical, semantic, source-table, environment and instrument hashes.
   - Records failures and quarantines explicitly; partial completion cannot advance.
4. `scale_admission_report_v1.json`
   - Binds the contract, plan, completed manifest and independent verification reports.
   - May authorize data materialization only.
   - Cannot authorize a model training surface.

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
- `2023-12-25` rejects as closed.
- `2023-11-05` rejects as Sunday/closed.
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

1. Add root listing/effective dates; do not count pre-inception sessions.
2. Implement agriculture's official `19:00–07:45 CT` and `08:30–13:20 CT` segments and filter
   exported rows to those half-open intervals.
3. Add and pin effective-dated early/shortened-session overrides.
4. Pin timezone database and calendar dependency versions in producer and verifier evidence.
5. Run bounded-memory and throughput tests on realistic large contract-days.
6. Push or archive AlphaForge `9f4c4bd` durably.
7. Implement and independently review the four scale artifacts above.

Only after those gates and the complete schema matrix pass may the project build the sessionized
liquidity/continuity denominator and freeze `core_comparable` versus `supplemental_pretraining`
roots. Model-family training admission remains a later, separate gate.
