# Foundation Model Native Contract Dossiers

**Status:** technical native-contract repair completed for every locally verifiable family  
**Registry:** [`config/foundation_models/native_contracts.json`](config/foundation_models/native_contracts.json)  
**Technical evidence:** [`config/foundation_models/native_contract_evidence.json`](config/foundation_models/native_contract_evidence.json)  
**Methodology revision:** `8e2bd47d8fd6dc333dfa74ad0eea3d3613e63469`

The JSON registry owns identity pins, tokenizer pairings, licenses, tracks, and historical
dispositions. The evidence file owns the executable parity record and exact runtime surface covered
by that record. Markdown cannot promote a model.

Technical validity is not runtime authorization. Execution still requires a current
`ffm_native_admission_report_v3` bound to the registry, dossier, evidence record, measured runtime
environment, execution controls, exact model/tokenizer/source/execution-code trees, and two
independently signed Ed25519 approvals from the colocated trusted-key registry. Base inference evidence does not
admit training: all persistent training, tuning, and adapted-checkpoint routes remain blocked until
the training-specific checks pass.

New parity evidence also carries a complete runtime lock. Its portable-software surface binds the
exact interpreter, platform and every installed distribution. Its hardware surface binds the Torch
CUDA runtime, cuDNN, visible-device setting, device identity/capability/memory and driver probe when
measurable; unavailable hardware probes are explicit values. Both surfaces are remeasured and
compared exactly for operational admission. Existing evidence predates this lock and is therefore
inspectable but not operationally authorizing until regenerated.

Generated technical records point to the tracked canonical raw-bundle archive. Admission-report
build and verification reopen that archive and fail on missing or altered fixture, result, logs,
manifest, or native arrays. The archive binds the exact producing model/source trees; a current
runtime binds its own artifacts separately. The repository records selected package versions and
source revisions but does not claim a portable binary environment or kernel-level network sandbox.

Reviewer labels and public-key fingerprints must be distinct, the two-approval floor cannot be
lowered, and every approval signs the immutable request after request creation. The packaged trust
store is intentionally empty, so no operational authorization exists until reviewed public keys
are installed. Organizational independence remains a governance responsibility even though key
possession is now cryptographically authenticated.

Operational consumers are currently source-checkout-only so the exact executable Python surface
can be measured against parity evidence. Installed wheels may inspect the sealed archive but fail
closed if used to request operational admission.

## Track vocabulary

| Code | Meaning |
|---|---|
| `F` | Native probabilistic, quantile, or point forecasting through a public forecast interface |
| `R` | Official representation output through a documented public interface |
| `C` | Experimental hidden-state, token-pooling, channel-fusion, or forecast-feature transfer |
| `B` | Supervised barrier/path prediction using an admitted upstream interface |
| `D` | Downstream in-context control, isolated from persistent encoder pretraining |

`native_valid` means the named native track passed executable technical parity. `research_only`
adds a noncommercial restriction. `blocked` means the named task is unsupported, unverified, or
externally unavailable. Historical `configuration_specific` and `invalid_contract` labels do not
become valid merely because a different native track now passes.

## Current status

| Arm | Pins | License/use | F | R | C/B/D | Operational authorization |
|---|---:|---|---|---|---|---|
| `mantis_v1` | complete | Apache-2.0 | unsupported | **native valid** | blocked | none; report + 2 approvals required |
| `mantis_v2` | complete | Apache-2.0 code / MIT weights | unsupported | **native valid** | blocked | none; report + 2 approvals required |
| `moment_small` | complete | MIT | blocked: no pretrained forecast head | **native valid** | blocked | none; report + 2 approvals required |
| `kronos_mini` | complete | MIT | **native valid** | unsupported | blocked | none; report + 2 approvals required |
| `kronos_small` | complete | MIT | **native valid** | unsupported | blocked | none; report + 2 approvals required |
| `chronos_v1` | complete | Apache-2.0 | **native valid** | **native valid** | blocked | none; report + 2 approvals required |
| `chronos_bolt` | complete | Apache-2.0 | **native valid** | **native valid** | blocked | none; report + 2 approvals required |
| `chronos_v2` | complete | Apache-2.0 | **native valid** | **native valid** | blocked | none; report + 2 approvals required |
| `timesfm25` | complete | Apache-2.0 | **native valid** | unsupported | blocked | none; report + 2 approvals required |
| `ttm_r2` | complete | Apache-2.0 | **native valid** | unsupported | blocked | none; report + 2 approvals required |
| `moirai2_small` | complete | CC-BY-NC-4.0 | **research only** | unsupported | blocked | none; noncommercial report + approvals required |
| `toto2_22m` | complete | Apache-2.0 | **native valid** | unsupported | blocked | none; report + 2 approvals required |
| `sundial_base` | complete | Apache-2.0 | **native valid** | explicitly excluded | blocked | none; isolated-env report + approvals required |
| `tabpfn_ts` | incomplete artifact pin | separate model terms unresolved | unsupported | unsupported | **D blocked** | impossible until terms and checkpoint are available |

Totals: 12 unrestricted arms have at least one technically valid native track, Moirai has one
research-only native track, and TabPFN-TS remains externally blocked. No arm is training-admitted.
No arm is operationally authorized without a current independently approved report.

## Evidence-covered runtime surfaces

The current evidence is deliberately narrow. Runtime helpers reject shape, precision, sampling,
preprocessing, or task drift before model execution.

| Arm/track | Covered surface |
|---|---|
| Mantis V1 `R` | 512 samples, FP32, official final CLS, output `[B,C,D]`; channel fusion forbidden |
| Mantis V2 `R` | 512 samples, FP32, layer 2, `output_token=combined`, output `[B,C,D]` |
| MOMENT `R` | 512 samples, FP32, official masked `reduction="mean"`, output `[B,D]` |
| Kronos Mini/Small `F` | 512 context, 16 horizon, FP32, joint OHLCVA, exact tokenizer pairing; deterministic greedy only; UTC 1/3/5/15/30/60-minute timestamps |
| Chronos V1 `F` | 512×16, FP32, 20 samples, quantiles 0.1/0.5/0.9 |
| Chronos V1 `R` | 512 samples, FP32, unpooled per-channel tokens plus tokenizer state |
| Chronos-Bolt `F` | 512×16, FP32, quantiles 0.1/0.5/0.9 |
| Chronos-Bolt `R` | 512 samples, FP32, unpooled per-channel tokens plus location/scale state |
| Chronos-2 `F` | 512×16, FP32, quantiles 0.1/0.5/0.9, `cross_learning=false` |
| Chronos-2 `R` | 512 samples, FP32, unpooled tokens plus native scaling state |
| TimesFM 2.5 `F` | 512×16, FP32, flip invariance on, positive truncation off, raw quantiles before optional crossing repair |
| TTM-R2 `F` | 512 real bars, 16 filtered steps, FP32, native scaler, tokens 1/0/3/5/6/7 for 1/3/5/15/30/60 minutes, no channel mixer |
| Moirai-2 `F` | 512×16, FP32, noncommercial research only, masked values zero-filled before packing; native crossing quantiles retained |
| Toto 2.0 `F` | 512×16, FP32, semantic series IDs, masked values zero-filled, `decode_block_size=None` |
| Sundial `F` | 512×16, FP32, 20 seeded samples, isolated dependency lock; hidden states forbidden |

## Family dispositions

### Mantis V1 and V2

Both weight revisions, the source revision, and licenses are pinned. V1 passed the official final
CLS surface. V2 passed the enhanced layer-2 CLS-plus-mean surface. The native adapter preserves the
per-channel output. Lossless flattening/concatenation is only a layout operation; averaging,
learned fusion, external z-scoring, alternate layers, or final-CLS-only V2 extraction are Track C.
Historical staged runs remain configuration-specific.

### MOMENT Small

The pretrained checkpoint admits only the official masked embedding mean. That reduction averages
the native channel/patch representation and does not preserve OHLCV channel identity. Left-padding
is paired with the official input mask and internal RevIN. The checkpoint does not contain a trained
forecasting or classification head, so those tasks require separate task-specific evidence.
Historical per-channel pooled concatenation remains Track C.

### Kronos Mini and Small

Mini is bound to `NeoQuasar/Kronos-Tokenizer-2k`; Small is bound to
`NeoQuasar/Kronos-Tokenizer-base`. Both passed deterministic greedy joint-OHLCVA forecasting at
512×16 across UTC 1/3/5/15/30/60-minute timestamps, including inverse transformation and candle
ordering. Stochastic decoding is not admitted by this evidence. `decode_s1` pooling is not an
official representation contract. Historical Mini/base-tokenizer artifacts remain invalid.

### Chronos V1, Bolt, and Chronos-2

All three forecast families passed their public quantile APIs at 512×16 in FP32. The concrete
pinned V1 and Bolt pipelines also expose documented public `embed` APIs. Their per-channel token
axes and tokenizer or location/scale state are preserved unpooled; any pooling remains Track C.
Chronos-2 also passed its public `embed` interface with grouped tokens and scaling state preserved
unpooled. Official `fit` routes remain blocked until training-specific evidence passes.

### TimesFM 2.5

The pinned Transformers implementation matched the official TimesFM 2.5 PyTorch wrapper for point
and raw quantile outputs at 512×16. Parity requires flip invariance, no positive truncation, and no
quantile-crossing repair inside the wrapper comparison. The pinned public forward surface accepts
no external frequency argument; it is frequency-agnostic rather than frequency-aware.
Last-hidden-state pooling and LoRA training remain separate blocked tracks.

### Tiny Time Mixer R2

The repaired selector is `512-48-ft-r2.1`: 512 real causal bars, native horizon 48 filtered to 16,
resolution-prefix tuning enabled with tokens `1/0/3/5/6/7` for the six project timeframes
(3-minute is OOV `0`), forecast channel mixing disabled, and five independent OHLCV
channels. Inputs remain raw so the model's native scaler performs scale and inverse-scale. The old
256-real-plus-256-zero route and random channel mixer remain invalid.

### Moirai 2 Small

The packed native forecast is technically valid only for noncommercial research. Values excluded by
the observed mask must be zero-filled before packing because raw masked garbage affects outputs.
Native quantiles may cross; optional crossing repair is downstream post-processing. Token pooling is
not an official representation contract.

### Toto 2.0 22M

Only zero-shot forecast is admitted. The short-horizon path uses `decode_block_size=None`, explicit
semantic series IDs, masks, and native scaling. Persistent fine-tuning and internal-state pooling
remain unsupported; historical trained stages remain invalid.

### Sundial Base

Only seeded forecast sampling in the isolated pinned environment is technically valid. The current
contract covers 20 samples at 512×16 in FP32. Hidden-state extraction and persistent training remain
excluded because prior hidden outputs were non-finite.

### TabPFN-TS

The code role is repaired to downstream in-context control only, with an executable fold-containment
guard that forbids support rows after the training-fold boundary and query rows at or before it.
Technical admission cannot proceed because the TabPFN-TS-3 checkpoint is not locally available, its
exact artifact hash is unresolved, and the separate model terms have not been accepted. It is not an
SSL encoder and never joins universal staged pretraining.

## Admission invariants

Every runtime report is rejected when any of these drift:

1. methodology commit, registry hash, or evidence-registry hash;
2. family dossier, technical evidence record, resolved checks, or runtime contract;
3. model/source/tokenizer identity;
4. arm, track, route, or research-only use scope;
5. pinned environment;
6. required checkpoint or derived-artifact hash;
7. report check status or integrity hash;
8. either independent approval.

Training additionally requires explicit technical and report passes for gradient/freeze surface,
repeated-batch loss decrease, exact resume, and save/reload/export. Current base-track evidence marks
those checks not applicable, so no current native-valid arm is silently training-admitted.

Use `scripts/check_foundation_native_contracts.py list` to inspect technical status,
`scripts/check_foundation_native_contracts.py show ARM` to inspect resolved evidence, and
`scripts/check_foundation_native_contracts.py verify` to validate a report. Native forecast
evaluation is zero-shot and FP32 only; adapted historical checkpoints are rejected. Native
representations are extracted with `scripts/extract_native_representations.py`; the historical
common-vector pooling benchmark remains Track C.
