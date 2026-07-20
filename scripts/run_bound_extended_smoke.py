#!/usr/bin/env python3
"""Invoke one extended exact-route smoke in its parity-bound Python environment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PARITY_ROOT = ROOT / "output/native_parity_evidence_current_config_v1"

ALIASES = {
    "z01": ("chronos_v2:F:official_fit_full", "chronos2_full", "chronos_v2", "F"),
    "z02": ("chronos_v2:F:official_fit_lora", "chronos2_lora", "chronos_v2", "F"),
    "z03": ("mantis_v1:C:supervised_classification_full", "mantis_v1_C_supervised_classification_full", "mantis_v1", "R"),
    "z04": ("mantis_v1:C:supervised_classification_head", "mantis_v1_C_supervised_classification_head", "mantis_v1", "R"),
    "z05": ("mantis_v1:R:official_crop_resize_contrastive", "mantis_v1_R_official_crop_resize_contrastive", "mantis_v1", "R"),
    "z06": ("mantis_v2:C:supervised_classification_full", "mantis_v2_C_supervised_classification_full", "mantis_v2", "R"),
    "z07": ("mantis_v2:C:supervised_classification_head", "mantis_v2_C_supervised_classification_head", "mantis_v2", "R"),
    "z08": ("mantis_v2:R:official_crop_resize_contrastive", "mantis_v2_R_official_crop_resize_contrastive", "mantis_v2", "R"),
    "z09": ("moirai2_small:F:custom_scaled_pinball_research", "moirai_research", "moirai2_small", "F"),
    "z10": ("moment_small:C:classification_full", "moment_small_C_classification_full", "moment_small", "R"),
    "z11": ("moment_small:C:classification_head_only", "moment_small_C_classification_head_only", "moment_small", "R"),
    "z12": ("moment_small:F:forecast_full_raw_mse", "moment_small_F_forecast_full_raw_mse", "moment_small", "R"),
    "z13": ("moment_small:F:forecast_head_only_raw_mse", "moment_small_F_forecast_head_only_raw_mse", "moment_small", "R"),
    "z14": ("timesfm25:F:official_lora_forecast", "timesfm_lora", "timesfm25", "F"),
    "z15": ("ttm_r2:F:full_model_raw_hf_trainer_forecast", "ttm_full", "ttm_r2", "F"),
    "z16": ("ttm_r2:F:head_prefix_raw_hf_trainer_forecast", "ttm_head", "ttm_r2", "F"),
}


def _python(arm: str, track: str) -> str:
    manifest = json.loads(
        (PARITY_ROOT / f"{arm}__{track}" / "bundle_manifest.json").read_text(encoding="utf-8")
    )
    return str(manifest["command"]["argv"][0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("alias", choices=sorted(ALIASES))
    parser.add_argument("--no-overwrite", action="store_true")
    args = parser.parse_args()
    route_key, directory, arm, track = ALIASES[args.alias]
    command = [
        _python(arm, track),
        str(ROOT / "scripts/smoke_extended_native_route.py"),
        "--route-key", route_key,
        "--parity-root", str(PARITY_ROOT),
        "--output", str(ROOT / "output/native_training_smoke" / directory),
        "--device", "cuda:0",
    ]
    if not args.no_overwrite:
        command.append("--overwrite")
    completed = subprocess.run(command, cwd=ROOT)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
