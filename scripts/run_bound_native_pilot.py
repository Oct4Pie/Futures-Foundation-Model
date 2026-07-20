#!/usr/bin/env python3
"""Invoke an existing bounded native pilot with parity/cache/smoke-bound paths."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PARITY_ROOT = ROOT / "output/native_parity_evidence_current_config_v1"
CACHE_DIR = ROOT / "output/foundation_tournament/data_cache_v3"
CACHE_SHA256 = "41281860ff1ef3474e226d22a2df504e97f8d348839fce8d2438689d160b9e0a"


def _flag(argv: list[str], name: str) -> str:
    index = argv.index(name)
    return argv[index + 1]


def _bindings(arm: str, track: str = "F") -> dict[str, str]:
    manifest = json.loads(
        (PARITY_ROOT / f"{arm}__{track}" / "bundle_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    argv = list(manifest["command"]["argv"])
    result = {
        "model": _flag(argv, "--model-snapshot"),
        "source": _flag(argv, "--source-repo"),
    }
    if "--tokenizer-snapshot" in argv:
        result["tokenizer"] = _flag(argv, "--tokenizer-snapshot")
    return result


def _common(smoke: Path, output: Path, *, overwrite: bool) -> list[str]:
    values = [
        "--smoke-evidence", str(smoke),
        "--cache-dir", str(CACHE_DIR),
        "--cache-manifest-sha256", CACHE_SHA256,
        "--output", str(output),
        "--device", "cuda:0",
        "--timeframes", "1min",
        "--quiet",
    ]
    if overwrite:
        values.append("--overwrite")
    return values


def command(args: argparse.Namespace) -> list[str]:
    python = sys.executable
    alias = args.alias
    if alias == "b":
        bound = _bindings("chronos_bolt")
        return [
            python, str(ROOT / "scripts/pilot_chronos_bolt_route.py"),
            "--model-snapshot", bound["model"],
            *_common(
                ROOT / "output/native_training_smoke/chronos_bolt_direct_native_quantile_pinball/smoke_evidence.json",
                ROOT / "output/native_training_pilot/chronos_bolt_direct_native_quantile_pinball_1min",
                overwrite=args.overwrite,
            ),
        ]
    if alias == "v":
        bound = _bindings("chronos_v1")
        return [
            python, str(ROOT / "scripts/pilot_chronos_v1_route.py"),
            "--model-snapshot", bound["model"],
            *_common(
                ROOT / "output/native_training_smoke/chronos_v1_native_64_t5_token_forecast_cross_entropy/smoke_evidence.json",
                ROOT / "output/native_training_pilot/chronos_v1_native_64_t5_token_forecast_cross_entropy_1min",
                overwrite=args.overwrite,
            ),
        ]
    if alias == "m":
        bound = _bindings("moment_small", "R")
        return [
            python, str(ROOT / "scripts/pilot_moment_reconstruction_route.py"),
            "--model-snapshot", bound["model"],
            "--source-runtime", bound["source"],
            *_common(
                ROOT / "output/native_training_smoke/moment_small_masked_patch_reconstruction/smoke_evidence.json",
                ROOT / "output/native_training_pilot/moment_small_masked_patch_reconstruction_1min",
                overwrite=args.overwrite,
            ),
        ]
    if alias == "kt":
        bound = _bindings("kronos_mini")
        return [
            python, str(ROOT / "scripts/pilot_kronos_tokenizer_route.py"),
            "--model-snapshot", bound["model"],
            "--tokenizer-snapshot", bound["tokenizer"],
            "--source-runtime", bound["source"],
            *_common(
                ROOT / "output/native_training_smoke/kronos_mini_tokenizer_reconstruction_bsq/smoke_evidence.json",
                ROOT / "output/native_training_pilot/kronos_mini_tokenizer_reconstruction_bsq_1min",
                overwrite=args.overwrite,
            ),
        ]
    if alias in {"p1", "p2"}:
        bound = _bindings("kronos_mini")
        output = ROOT / (
            "output/native_training_pilot/kronos_mini_hierarchical_autoregressive_tokens_1min"
            if alias == "p1"
            else "output/native_training_pilot/kronos_mini_hierarchical_autoregressive_tokens_1min_256"
        )
        values = [
            python, str(ROOT / "scripts/pilot_kronos_predictor_route.py"),
            "--model-snapshot", bound["model"],
            "--tokenizer-snapshot", bound["tokenizer"],
            "--source-runtime", bound["source"],
            "--parent-pilot-evidence",
            str(ROOT / "output/native_training_pilot/kronos_mini_tokenizer_reconstruction_bsq_1min/pilot_evidence.json"),
            "--parent-tokenizer-bundle",
            str(ROOT / "output/native_training_pilot/kronos_mini_tokenizer_reconstruction_bsq_1min/kronos_mini_tokenizer_pilot.bundle.pt"),
            *_common(
                ROOT / "output/native_training_smoke/kronos_mini_hierarchical_autoregressive_tokens/smoke_evidence.json",
                output,
                overwrite=args.overwrite,
            ),
        ]
        if alias == "p2":
            values.extend(["--steps", "256", "--eval-every", "64"])
        return values
    raise ValueError(alias)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alias", required=True, choices=("b", "v", "m", "kt", "p1", "p2"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = subprocess.run(command(args), cwd=ROOT, check=False)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
