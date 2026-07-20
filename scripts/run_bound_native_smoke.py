#!/usr/bin/env python3
"""Invoke an existing exact-route smoke with paths bound by current parity evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PARITY_ROOT = ROOT / "output" / "native_parity_evidence_current_config_v1"
SHORT_ALIASES = {
    "b": "bolt",
    "v": "chronos_v1",
    "m": "moment_reconstruction",
    "km": "kronos_mini_tokenizer",
    "ks": "kronos_small_tokenizer",
    "kp": "kronos_mini_predictor",
}
DEFAULT_OUTPUTS = {
    "bolt": "output/native_training_smoke/chronos_bolt_direct_native_quantile_pinball",
    "chronos_v1": "output/native_training_smoke/chronos_v1_native_64_t5_token_forecast_cross_entropy",
    "moment_reconstruction": "output/native_training_smoke/moment_small_masked_patch_reconstruction",
    "kronos_mini_tokenizer": "output/native_training_smoke/kronos_mini_tokenizer_reconstruction_bsq",
    "kronos_small_tokenizer": "output/native_training_smoke/kronos_small_tokenizer_reconstruction_bsq",
    "kronos_mini_predictor": "output/native_training_smoke/kronos_mini_hierarchical_autoregressive_tokens",
}


def _flag(argv: list[str], name: str) -> str:
    index = argv.index(name)
    return argv[index + 1]


def _bindings(arm: str, track: str) -> dict[str, str]:
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


def command(args: argparse.Namespace) -> list[str]:
    python = sys.executable
    args.alias = SHORT_ALIASES.get(args.alias, args.alias)
    output_value = args.output or DEFAULT_OUTPUTS[args.alias]
    output = str(Path(output_value).expanduser().resolve())
    common = ["--output", output, "--device", args.device]
    if args.overwrite:
        common.append("--overwrite")
    if args.alias == "bolt":
        bound = _bindings("chronos_bolt", "F")
        return [
            python,
            str(ROOT / "scripts" / "smoke_chronos_bolt_route.py"),
            "--model-snapshot",
            bound["model"],
            *common,
        ]
    if args.alias == "chronos_v1":
        bound = _bindings("chronos_v1", "F")
        return [
            python,
            str(ROOT / "scripts" / "smoke_chronos_v1_route.py"),
            "--model-snapshot",
            bound["model"],
            *common,
        ]
    if args.alias == "moment_reconstruction":
        bound = _bindings("moment_small", "R")
        return [
            python,
            str(ROOT / "scripts" / "smoke_moment_reconstruction_route.py"),
            "--model-snapshot",
            bound["model"],
            "--source-runtime",
            bound["source"],
            *common,
        ]
    if args.alias in {"kronos_mini_tokenizer", "kronos_small_tokenizer"}:
        arm = "kronos_mini" if args.alias.startswith("kronos_mini") else "kronos_small"
        bound = _bindings(arm, "F")
        result = [
            python,
            str(ROOT / "scripts" / "smoke_kronos_tokenizer_route.py"),
            "--arm",
            arm,
            "--model-snapshot",
            bound["model"],
            "--tokenizer-snapshot",
            bound["tokenizer"],
            "--source-runtime",
            bound["source"],
            "--phase",
            args.phase,
            *common,
        ]
        return result
    if args.alias == "kronos_mini_predictor":
        bound = _bindings("kronos_mini", "F")
        pilot = (
            ROOT
            / "output/native_training_pilot/kronos_mini_tokenizer_reconstruction_bsq_1min/pilot_evidence.json"
        )
        bundle = (
            ROOT
            / "output/native_training_pilot/kronos_mini_tokenizer_reconstruction_bsq_1min/kronos_mini_tokenizer_pilot.bundle.pt"
        )
        return [
            python,
            str(ROOT / "scripts" / "smoke_kronos_predictor_route.py"),
            "--model-snapshot",
            bound["model"],
            "--tokenizer-snapshot",
            bound["tokenizer"],
            "--source-runtime",
            bound["source"],
            "--parent-pilot-evidence",
            str(pilot),
            "--parent-tokenizer-bundle",
            str(bundle),
            *common,
        ]
    raise ValueError(args.alias)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--alias",
        required=True,
        choices=(
            "bolt",
            "chronos_v1",
            "moment_reconstruction",
            "kronos_mini_tokenizer",
            "kronos_small_tokenizer",
            "kronos_mini_predictor",
            *SHORT_ALIASES,
        ),
    )
    parser.add_argument("--output")
    parser.add_argument(
        "--phase", choices=("real", "controls", "resume", "finalize", "all"), default="all"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = subprocess.run(command(args), cwd=ROOT, check=False)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
