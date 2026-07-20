#!/usr/bin/env python3
"""Build the current evidence-aware native-training readiness report."""
from __future__ import annotations

from pathlib import Path
import json
import os

from futures_foundation.finetune.native_contracts import canonical_json
from futures_foundation.finetune.native_training_readiness import (
    build_training_readiness_report,
    validate_training_readiness_report,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output/native_training_readiness_current_v2.json"

SMOKE = {
    "chronos_bolt:F:direct_native_quantile_pinball": "chronos_bolt_direct_native_quantile_pinball",
    "chronos_v1:F:native_64_t5_token_forecast_cross_entropy": "chronos_v1_native_64_t5_token_forecast_cross_entropy",
    "chronos_v2:F:official_fit_full": "chronos2_full",
    "chronos_v2:F:official_fit_lora": "chronos2_lora",
    "kronos_mini:F:hierarchical_autoregressive_tokens": "kronos_mini_hierarchical_autoregressive_tokens",
    "kronos_mini:F:tokenizer_reconstruction_bsq": "kronos_mini_tokenizer_reconstruction_bsq",
    "kronos_small:F:tokenizer_reconstruction_bsq": "kronos_small_tokenizer_reconstruction_bsq",
    "mantis_v1:C:supervised_classification_full": "mantis_v1_C_supervised_classification_full",
    "mantis_v1:C:supervised_classification_head": "mantis_v1_C_supervised_classification_head",
    "mantis_v1:R:official_crop_resize_contrastive": "mantis_v1_R_official_crop_resize_contrastive",
    "mantis_v2:C:supervised_classification_full": "mantis_v2_C_supervised_classification_full",
    "mantis_v2:C:supervised_classification_head": "mantis_v2_C_supervised_classification_head",
    "mantis_v2:R:official_crop_resize_contrastive": "mantis_v2_R_official_crop_resize_contrastive",
    "moirai2_small:F:custom_scaled_pinball_research": "moirai_research",
    "moment_small:C:classification_full": "moment_small_C_classification_full",
    "moment_small:C:classification_head_only": "moment_small_C_classification_head_only",
    "moment_small:F:forecast_full_raw_mse": "moment_small_F_forecast_full_raw_mse",
    "moment_small:F:forecast_head_only_raw_mse": "moment_small_F_forecast_head_only_raw_mse",
    "moment_small:R:masked_patch_reconstruction": "moment_small_masked_patch_reconstruction",
    "timesfm25:F:official_lora_forecast": "timesfm_lora",
    "ttm_r2:F:full_model_raw_hf_trainer_forecast": "ttm_full",
    "ttm_r2:F:head_prefix_raw_hf_trainer_forecast": "ttm_head",
}

PILOT = {
    "chronos_bolt:F:direct_native_quantile_pinball": "chronos_bolt_direct_native_quantile_pinball_1min",
    "chronos_v1:F:native_64_t5_token_forecast_cross_entropy": "chronos_v1_native_64_t5_token_forecast_cross_entropy_1min",
    "kronos_mini:F:hierarchical_autoregressive_tokens": "kronos_mini_hierarchical_autoregressive_tokens_1min_256",
    "kronos_mini:F:tokenizer_reconstruction_bsq": "kronos_mini_tokenizer_reconstruction_bsq_1min",
    "moment_small:R:masked_patch_reconstruction": "moment_small_masked_patch_reconstruction_1min",
    "mantis_v1:R:official_crop_resize_contrastive": "mantis_v1_official_crop_resize_contrastive_1min",
    "mantis_v2:R:official_crop_resize_contrastive": "mantis_v2_official_crop_resize_contrastive_1min",
    "moment_small:F:forecast_full_raw_mse": "moment_small_forecast_full_raw_mse_1min",
    "moment_small:F:forecast_head_only_raw_mse": "moment_small_forecast_head_only_raw_mse_1min",
    "ttm_r2:F:full_model_raw_hf_trainer_forecast": "ttm_r2_full_model_raw_forecast_1min",
    "ttm_r2:F:head_prefix_raw_hf_trainer_forecast": "ttm_r2_head_prefix_raw_forecast_1min",
    "timesfm25:F:official_lora_forecast": "timesfm25_official_lora_forecast_1min",
    "chronos_v2:F:official_fit_full": "chronos_v2_official_fit_full_1min",
    "chronos_v2:F:official_fit_lora": "chronos_v2_official_fit_lora_1min",
    "moirai2_small:F:custom_scaled_pinball_research": "moirai2_small_scaled_pinball_research_1min",
}

DOWNSTREAM = {
    "chronos_bolt:F:direct_native_quantile_pinball": "chronos_bolt",
    "chronos_v1:F:native_64_t5_token_forecast_cross_entropy": "chronos_v1",
    "kronos_mini:F:hierarchical_autoregressive_tokens": "kronos_mini_predictor",
    "moment_small:R:masked_patch_reconstruction": "moment",
    "mantis_v1:R:official_crop_resize_contrastive": "mantis_v1_contrastive",
    "mantis_v2:R:official_crop_resize_contrastive": "mantis_v2_contrastive",
    "moment_small:F:forecast_full_raw_mse": "moment_forecast_full",
    "moment_small:F:forecast_head_only_raw_mse": "moment_forecast_head",
}


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    temporary.write_bytes(canonical_json(value) + b"\n")
    os.replace(temporary, path)


def main() -> None:
    smoke_paths = {
        route: str((ROOT / "output/native_training_smoke" / directory / "smoke_evidence.json").resolve())
        for route, directory in SMOKE.items()
    }
    pilot_paths = {
        route: str((ROOT / "output/native_training_pilot" / directory / "pilot_evidence.json").resolve())
        for route, directory in PILOT.items()
    }
    downstream_paths = {
        route: str((
            ROOT
            / "output/foundation_tournament/downstream_current_1min_v4/incremental"
            / directory
            / "native_incremental_results.json"
        ).resolve())
        for route, directory in DOWNSTREAM.items()
    }
    report = build_training_readiness_report(
        smoke_evidence_paths=smoke_paths,
        pilot_evidence_paths=pilot_paths,
        downstream_screen_paths=downstream_paths,
    )
    report = validate_training_readiness_report(
        report,
        smoke_evidence_paths=smoke_paths,
        pilot_evidence_paths=pilot_paths,
        downstream_screen_paths=downstream_paths,
    )
    _write(OUTPUT, report)
    print(json.dumps({
        "schema_version": report["schema_version"],
        "readiness_sha256": report["readiness_sha256"],
        "counts": report["counts"],
        "all_exact_routes_pilot_dispositioned": report["all_exact_routes_pilot_dispositioned"],
        "all_surviving_pilots_downstream_dispositioned": report[
            "all_surviving_pilots_downstream_dispositioned"
        ],
        "pilot_admitted": report["pilot_admitted"],
        "training_admitted": report["training_admitted"],
        "live_trading_ready": report["live_trading_ready"],
        "output": str(OUTPUT),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
