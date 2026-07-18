# Foundation Model Training Roster

## Decision

The training-readiness roster is frozen at 14 canonical arms across 10 conceptual families. A
frozen native forecast (`F`) or representation (`R`) parity result proves only that the official
inference surface was reproduced. It does not admit fine-tuning, SSL, barrier/path training,
custom pooling, export, or trading use.

Current status:

```text
registered arms:       14
training-admitted:      0
runtime-authorized:     0
unrestricted arms with at least one native-valid track: 12
research-only arms:     1 (Moirai-2 Small)
blocked arms:           1 (TabPFN-TS)
```

Authoritative inputs:

```text
registry: config/foundation_models/native_contracts.json
canonical registry-content SHA-256: e51b6a32a92ac96bf5d7b93cc0cca1fbb53dd3dc5db25e99e98aeca20a2fa5a1
evidence: config/foundation_models/native_contract_evidence.json
canonical evidence-content SHA-256: 1fd3115ae6bccf82136380bb24d540034899dc261f0f8dad852ba09825265edc
```

## Canonical arms and blockers

| Arm | Native-valid evidence | Training/pathway blocker |
| --- | --- | --- |
| `mantis_v1` | `R`: official final CLS per channel | Forecasting is not native. Channel fusion, preprocessing, barrier adaptation, resume and export require separate evidence. |
| `mantis_v2` | `R`: official layer-2 CLS+mean per channel | Final-CLS-only and alternate pooling are custom. Forecast/path training, resume and export are unadmitted. |
| `moment_small` | `R`: official masked embedding mean | The pretrained checkpoint has no trained forecast head. Forecast, classifier and path heads need task-specific admission. |
| `kronos_mini` | `F`: joint OHLCVA greedy forecast with Tokenizer-2k | No official representation contract. Hidden-state pooling and barrier use are custom. Historical wrong-tokenizer runs are invalid. |
| `kronos_small` | `F`: joint OHLCVA greedy forecast with Tokenizer-base | No official representation contract. Hidden-state pooling and barrier use are custom. |
| `chronos_v1` | `F` and `R`: seeded forecast plus public unpooled token state | Any pooling is custom. Barrier/path training needs separate admission. |
| `chronos_bolt` | `F` and `R`: quantile forecast plus unpooled tokens/location-scale | Mean pooling and location-scale fusion are custom. Fine-tuning and barrier paths are unadmitted. |
| `chronos_v2` | `F` and `R`: grouped multivariate forecast plus public tokens/scaling | Custom token pooling and path training require separate admission. Chronos-2 is not an alias of T5 or Bolt. |
| `timesfm25` | `F`: official-wrapper-equivalent point/raw quantile forecast | No official representation contract. Hidden-state pooling, LoRA and barrier use are unadmitted. |
| `ttm_r2` | `F`: selector, scaler and frequency-prefix parity | No official representation contract. Historical selector bypass, random mixer and padding runs are invalid. Separate representation/path evidence is absent. |
| `moirai2_small` | `F`: technically reproduced packed forecast | Research-only under CC-BY-NC-4.0. No official representation contract; commercial use and barrier work are unadmitted. |
| `toto2_22m` | `F`: native zero-shot joint forecast | No official representation contract. Internal-state pooling and historical Stage 1–3 adaptations are invalid. |
| `sundial_base` | `F`: isolated seeded forecast | Representation extraction is excluded because hidden states were non-finite. No training path is admitted. |
| `tabpfn_ts` | None | Downstream-only control, not a staged encoder. Terms, checkpoint identity and native parity are unresolved. |

## Alias rules

- Frozen and adapted Mantis states are not additional arms.
- MOMENT continued pretraining is a pathway, not another arm.
- Kronos Mini and Small remain distinct because their model/tokenizer pairs differ.
- Chronos V1, Chronos-Bolt and Chronos-2 are distinct architectures.
- Moirai/Salesforce names resolve to one registered arm.
- Historical Stage 1/2/3 artifacts never substitute for native or training-admission evidence.
- Forecast-only families cannot be converted into representation arms by an improvised hidden-state
  mean and then called apples-to-apples.

## Per-family training-admission contract

Every arm that proceeds must independently prove:

1. Official input layout, scaling, masking, channel and multivariate semantics.
2. The supported trainable surface and architecture-native loss.
3. Correct gradient flow and declared frozen/trainable parameter sets.
4. Repeated-batch loss reduction and shuffled/constant-input controls.
5. Exact full-state resume, including optimizer, scheduler, scaler, epoch and RNG.
6. Save/reload equality and frozen inference/export numerical parity.
7. A causal, split-purged path/barrier or downstream interface without reserved-OOS access.
8. Complete provenance: source, model, tokenizer, environment, corpus, command and artifact hashes.

Only after those checks may an arm enter the common-row dry-training matrix. Models that cannot
support a requested pathway are excluded with evidence instead of being forced through a foreign
three-stage recipe.
