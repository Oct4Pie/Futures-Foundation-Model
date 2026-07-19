# Execution economics contract

`config/execution_costs.yaml` is the single research schedule for outright tick size, tick value,
and approximate all-in round-trip fees. Runtime consumers must load it through
`futures_foundation.execution_economics.load_execution_economics`; in-code tick tables and raw
schedule dictionaries are not admitted.

The loader fails closed when:

- the schedule or its pinned AlphaForge source is missing or hash-mismatched;
- a copied instrument value disagrees with that source;
- an instrument is missing;
- the caller omits UTC offsets or asks for dates outside the declared interval;
- the primary added-slippage assumption is not zero; or
- a requested sensitivity is not declared by the schedule.

The canonical source receipt is vendored under `config/economics_sources/`; runtime loading does
not depend on a sibling AlphaForge checkout. Consumers require a non-forgeable canonical
capability and reverify both schedule and source at use.

## Cost semantics

The primary executable downstream ruler subtracts the declared cash round-trip fee and **zero
added slippage ticks**, with next-open entry and zero additional delay. The standing sensitivity
adds one round-trip tick. Two- and three-tick scenarios remain declared stress diagnostics; they
are not the primary result.

Event-context shards store gross R only. They do not own or pre-apply a fee or slippage
assumption. Downstream policy materialization subtracts the capability's cash fee and selected
declared added-slippage scenario exactly once. The standalone trend ruler uses the same formula;
there is no in-code tick table or caller-authored source-cost override.

The fee values are constant research estimates copied from the pinned AlphaForge instrument
document. They are not historical broker-account statements. The current capability is explicitly
date-bounded; extending its end requires a reviewed schedule revision and creates a new schedule
hash.

## Artifact binding

Current event-context and downstream-policy artifacts include the complete economics capability
manifest: schedule/source paths and SHA-256 hashes, effective and requested UTC intervals,
slippage assumptions, and resolved instrument specifications. Schema revisions reject older
artifacts by default; legacy loading must be explicitly requested and is not ranking-admitted.
