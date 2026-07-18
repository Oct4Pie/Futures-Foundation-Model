# Native Real-Checkpoint Parity Runbook

This runbook produces technical evidence only. It does not train, read market data,
score trading performance, inspect OOS, or authorize an operational runtime.

The child worker is `scripts/native_parity_worker.py`. It must be launched through
`scripts/run_native_parity_evidence.py run` so the canonical synthetic fixture,
checkpoint, source, tokenizer/reference artifacts, worker source, logs, raw arrays,
and result JSON are hash-bound in one evidence bundle.

## Runtime profiles

| Profile | Arms | Required interpreter contract |
|---|---|---|
| `common` | Mantis V1/V2, MOMENT, Kronos Mini/Small, Chronos V1/Bolt/2, Toto 2 | Python 3.12, Torch 2.13; exact family package checked by the worker |
| `timesfm` | TimesFM 2.5 | Python 3.12, Torch 2.13, Transformers 5.13.1, official TimesFM source/reference |
| `ttm` | TTM R2 | Python 3.12, Torch 2.10, Transformers 4.57.6 |
| `moirai` | Moirai-2 Small | Python 3.11, Torch 2.10, Uni2TS 2.0.0; research-only |
| `sundial` | Sundial Base | Python 3.12, Torch 2.10, Transformers 4.40.1, HF Hub 0.36.2 |

The worker rejects the wrong profile, an unbound CLI path, a dirty/wrong Git source,
a wrong snapshot revision, non-finite output, pooled Chronos representations, and a
bundle/worker arm mismatch.

The matrix worker also installs a Python audit-hook/socket guard and forces the usual
HF/Transformers offline flags before family imports. The host currently denies
unprivileged network namespaces, so this is **application-layer Python network denial,
not kernel network isolation**. Native extensions or child processes are outside that
guard; canonical runs must use an externally network-isolated host if that stronger
property is required.

Numeric parity is fail-closed at the registry-bound `atol=1e-5`, `rtol=1e-5`.
Every persisted `official[_name]` output must have an `adapter[_name]` peer and pass
`numpy.allclose`. If a runner declares partition evidence, every public adapter output
must also have a passing `partitioned[_name]` peer. Sundial uses its deterministic
seeded repeat as the corresponding stochastic reproducibility check. Maximum absolute
errors are diagnostic only; they never substitute for the relative-and-absolute
tolerance decision.

## Required local artifacts

Use materialized or HF snapshot directories for `model` and `tokenizer`. The directory
basename must be the exact registry revision. Git sources must be clean, at the exact
HEAD and origin recorded by the registry.

The current durable local source root is:

```text
/home/m3hdi/.cache/ffm-native/sources/
```

Mantis is currently pinned at:

```text
/home/m3hdi/projects/mantis
```

The isolated environments are under:

```text
/home/m3hdi/.cache/ffm-native/envs/{ttm,moirai,sundial}
```

TimesFM additionally requires the exact official PyTorch reference checkpoint as the
`reference_model` artifact. Kronos Mini/Small require their exact, different tokenizer
artifacts. Bundled Chronos tokenizers are part of the model artifact and must not be
supplied separately.

## Sealed command template

For full coverage, use `ffm-native-parity-matrix` (or
`scripts/run_native_parity_matrix.py`) with an explicit runtime/source config. It derives
all admitted F/R pairs from the registry, validates exact local snapshots, runs them
sequentially, and refuses partial aggregation. The single-bundle command below remains
useful for diagnosis.

```bash
PY=/absolute/path/to/profile/python
WORKER=/absolute/path/to/Futures-Foundation-Model/scripts/native_parity_worker.py

$PY scripts/run_native_parity_evidence.py run \
  --arm ARM --track F_OR_R --output /empty/evidence/bundle \
  --artifact model=/exact/model/snapshot \
  --artifact source=/exact/source/artifact \
  --env HF_HUB_OFFLINE=1 -- \
  $PY $WORKER \
  --arm ARM --track F_OR_R --profile PROFILE \
  --model-snapshot /exact/model/snapshot \
  --source-repo /exact/source/artifact \
  --device cuda:0 --batch-size 4
```

Add both of these for Kronos:

```text
--artifact tokenizer=/exact/tokenizer/snapshot
--tokenizer-snapshot /exact/tokenizer/snapshot
```

Add both of these for TimesFM:

```text
--artifact reference_model=/exact/reference/snapshot
--reference-model-snapshot /exact/reference/snapshot
```

Every path passed to the worker must equal the corresponding
`FFM_NATIVE_PARITY_ARTIFACT_*` path injected by the sealing process.

## Canonical real-checkpoint evidence

The canonical and replay matrices use only the generated synthetic OHLCV fixture and exact cached
revisions. The canonical tracked archive is:

```text
output/native_parity_evidence_canonical/
```

The independent local replay is written separately and is not installed as canonical evidence.
Both runs must cover all 16 registry-admitted F/R pairs and produce byte-identical native arrays.
The canonical evidence JSON uses paths relative to the registry directory, so copying it out of the
aggregate does not change path meaning. Admission-report build and verification reopen the archive
and fail closed on missing or altered internal proof.

The family-level expectations are:

| Arm/track | Required evidence |
|---|---|
| Mantis V1/V2 `R` | Official transform/API parity, full/partition parity, unpooled `[B,C,D]` |
| MOMENT Small `R` | Official masked mean embedding, mask perturbation and full/partition parity |
| Kronos Mini/Small `F` | Exact tokenizer pair, joint OHLCVA, six UTC bar cadences, affine raw-scale check |
| Chronos V1/Bolt/2 `F/R` | Public forecast and unpooled `embed` surfaces; exact native scaling state |
| TimesFM 2.5 `F` | Transformers wrapper versus bound official PyTorch reference, raw point/quantiles |
| TTM R2 `F` | Exact selector, six resolution tokens, real 512-bar input, affine scale check |
| Moirai-2 Small `F` | Research-only packed multivariate quantiles, mask and affine scale checks |
| Toto 2.0 22M `F` | Grouped quantiles, masks, native scale and `decode_block_size=None` |
| Sundial Base `F` | Seeded finite samples and affine inverse scaling; hidden states forbidden |

The first Kronos smoke also caught an upstream-interface detail: the pinned predictor
documents a `DatetimeIndex` but calls the pandas `.dt` accessor internally. The worker
therefore supplies a `Series[datetime64, UTC]`, matching the executable public path. This
is retained as an explicit frequency/timestamp contract, not hidden coercion.

## Completion rule

A family is not technically ready until its real bundle:

1. verifies after a fresh process restart;
2. binds all model, source, tokenizer/reference and worker bytes;
3. preserves native output shape and probabilistic information;
4. has no failed mandatory check;
5. is independently rerun from the same pins;
6. is included in a complete aggregate covering every current admitted track.

Training remains blocked after technical parity. Training-specific loss, gradient,
resume, save/reload, deployment parity, and leakage evidence are separate gates.
