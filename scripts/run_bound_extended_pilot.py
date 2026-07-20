#!/usr/bin/env python3
"""Invoke an extended bounded pilot with the exact parity Python and canonical output path."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PARITY_ROOT = ROOT / "output/native_parity_evidence_current_config_v1"

ALIASES = {
    "x01": ("p01", "mantis_v1_official_crop_resize_contrastive_1min", "mantis_v1", "R"),
    "x02": ("p02", "mantis_v2_official_crop_resize_contrastive_1min", "mantis_v2", "R"),
    "x03": ("p03", "moment_small_forecast_full_raw_mse_1min", "moment_small", "R"),
    "x04": ("p04", "moment_small_forecast_head_only_raw_mse_1min", "moment_small", "R"),
    "x05": ("p05", "ttm_r2_full_model_raw_forecast_1min", "ttm_r2", "F"),
    "x06": ("p06", "ttm_r2_head_prefix_raw_forecast_1min", "ttm_r2", "F"),
    "x07": ("p07", "timesfm25_official_lora_forecast_1min", "timesfm25", "F"),
    "x08": ("p08", "chronos_v2_official_fit_full_1min", "chronos_v2", "F"),
    "x09": ("p09", "chronos_v2_official_fit_lora_1min", "chronos_v2", "F"),
    "x10": ("p10", "moirai2_small_scaled_pinball_research_1min", "moirai2_small", "F"),
}


def _python(arm: str, track: str) -> Path:
    manifest = json.loads(
        (PARITY_ROOT / f"{arm}__{track}" / "bundle_manifest.json").read_text(encoding="utf-8")
    )
    return Path(manifest["command"]["argv"][0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("alias", choices=sorted(ALIASES))
    parser.add_argument("--no-overwrite", action="store_true")
    args = parser.parse_args()
    route_alias, directory, arm, track = ALIASES[args.alias]
    output = ROOT / "output/native_training_pilot" / directory
    command = [
        str(_python(arm, track)),
        str(ROOT / "scripts/pilot_extended_native_route.py"),
        "--route-alias", route_alias,
        "--parity-root", str(PARITY_ROOT),
        "--output", str(output),
        "--device", "cuda:0",
        "--quiet",
    ]
    if not args.no_overwrite:
        command.append("--overwrite")
    completed = subprocess.run(command, cwd=ROOT)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
