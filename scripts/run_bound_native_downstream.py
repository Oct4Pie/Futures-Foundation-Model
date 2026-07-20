#!/usr/bin/env python3
"""Run current common-information feature extraction and incremental screens."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "output/foundation_tournament/downstream_current_1min_v4"
SAMPLE = BASE / "balanced_sample.npz"
ROWS = BASE / "representation_rows_300.npz"
CONTEXTS = BASE / "contexts.npz"
FEATURE_DIR = BASE / "representation_screen"
INCREMENTAL = BASE / "incremental"
FOLD_SHA = "d460f60d442407cb207a63d907bba5b85a84a8ee362f78c959d46de4cc52cdf1"

ROUTES = {
    "b": {
        "key": "chronos_bolt:F:direct_native_quantile_pinball",
        "pilot": ROOT / "output/native_training_pilot/chronos_bolt_direct_native_quantile_pinball_1min/pilot_evidence.json",
        "feature": FEATURE_DIR / "features_chronos_bolt.npz",
        "screen": INCREMENTAL / "chronos_bolt",
        "batch": 128,
    },
    "v": {
        "key": "chronos_v1:F:native_64_t5_token_forecast_cross_entropy",
        "pilot": ROOT / "output/native_training_pilot/chronos_v1_native_64_t5_token_forecast_cross_entropy_1min/pilot_evidence.json",
        "feature": FEATURE_DIR / "features_chronos_v1.npz",
        "screen": INCREMENTAL / "chronos_v1",
        "batch": 8,
    },
    "m": {
        "key": "moment_small:R:masked_patch_reconstruction",
        "pilot": ROOT / "output/native_training_pilot/moment_small_masked_patch_reconstruction_1min/pilot_evidence.json",
        "feature": FEATURE_DIR / "features_moment.npz",
        "screen": INCREMENTAL / "moment",
        "batch": 128,
    },
    "k": {
        "key": "kronos_mini:F:hierarchical_autoregressive_tokens",
        "pilot": ROOT / "output/native_training_pilot/kronos_mini_hierarchical_autoregressive_tokens_1min_256/pilot_evidence.json",
        "feature": FEATURE_DIR / "features_kronos_mini_predictor.npz",
        "screen": INCREMENTAL / "kronos_mini_predictor",
        "batch": 32,
    },
    "r": {
        "key": "mantis_v1:R:official_crop_resize_contrastive",
        "pilot": ROOT / "output/native_training_pilot/mantis_v1_official_crop_resize_contrastive_1min/pilot_evidence.json",
        "feature": FEATURE_DIR / "features_mantis_v1_contrastive.npz",
        "screen": INCREMENTAL / "mantis_v1_contrastive",
        "batch": 64,
    },
    "s": {
        "key": "mantis_v2:R:official_crop_resize_contrastive",
        "pilot": ROOT / "output/native_training_pilot/mantis_v2_official_crop_resize_contrastive_1min/pilot_evidence.json",
        "feature": FEATURE_DIR / "features_mantis_v2_contrastive.npz",
        "screen": INCREMENTAL / "mantis_v2_contrastive",
        "batch": 32,
    },
    "f": {
        "key": "moment_small:F:forecast_full_raw_mse",
        "pilot": ROOT / "output/native_training_pilot/moment_small_forecast_full_raw_mse_1min/pilot_evidence.json",
        "feature": FEATURE_DIR / "features_moment_forecast_full.npz",
        "screen": INCREMENTAL / "moment_forecast_full",
        "batch": 64,
    },
    "h": {
        "key": "moment_small:F:forecast_head_only_raw_mse",
        "pilot": ROOT / "output/native_training_pilot/moment_small_forecast_head_only_raw_mse_1min/pilot_evidence.json",
        "feature": FEATURE_DIR / "features_moment_forecast_head.npz",
        "screen": INCREMENTAL / "moment_forecast_head",
        "batch": 64,
    },
}


def _extract(spec: dict, *, overwrite: bool) -> list[str]:
    result = [
        sys.executable,
        str(ROOT / "scripts/extract_route_downstream_features.py"),
        "--route-key", str(spec["key"]),
        "--sample", str(SAMPLE),
        "--row-selection", str(ROWS),
        "--contexts", str(CONTEXTS),
        "--pilot-evidence", str(spec["pilot"]),
        "--output", str(spec["feature"]),
        "--device", "cuda:0",
        "--batch-size", str(spec["batch"]),
    ]
    if overwrite:
        result.append("--overwrite")
    return result


def _v1_chunk(index: int, *, overwrite: bool) -> list[str]:
    spec = ROUTES["v"]
    start = int(index) * 300
    stop = start + 300
    chunk = FEATURE_DIR / "v1_chunks" / f"chunk_{start:04d}_{stop:04d}.npz"
    result = _extract(spec, overwrite=overwrite)
    result.extend([
        "--chunk-output", str(chunk),
        "--chunk-start", str(start),
        "--chunk-stop", str(stop),
    ])
    return result


def _v1_finalize(*, overwrite: bool) -> list[str]:
    spec = ROUTES["v"]
    result = _extract(spec, overwrite=overwrite)
    for index in range(9):
        start = index * 300
        stop = start + 300
        result.extend([
            "--finalize-chunk",
            str(FEATURE_DIR / "v1_chunks" / f"chunk_{start:04d}_{stop:04d}.npz"),
        ])
    return result


def _screen(spec: dict) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts/benchmark_native_incremental.py"),
        "--sample", str(SAMPLE),
        "--row-selection", str(ROWS),
        "--features", str(spec["feature"]),
        "--output-dir", str(spec["screen"]),
        "--expected-fold-sha256", FOLD_SHA,
        "--block-weights",
    ]


def _aggregate() -> list[str]:
    result = [
        sys.executable,
        str(ROOT / "scripts/audit_native_downstream_screen.py"),
        "--output", str(INCREMENTAL / "SCREEN_COLLECTION.json"),
        "--compact",
    ]
    for alias in ("b", "v", "m", "k", "r", "s", "f", "h"):
        spec = ROUTES[alias]
        result.extend([
            "--report",
            f"{spec['key']}={spec['screen'] / 'native_incremental_results.json'}",
        ])
    return result


def command(args: argparse.Namespace) -> list[str]:
    if args.action == "extract":
        if args.route == "v":
            raise ValueError("Chronos V1 requires --action vchunk/vfinal")
        return _extract(ROUTES[args.route], overwrite=args.overwrite)
    if args.action == "vchunk":
        if args.chunk is None or not 0 <= args.chunk <= 8:
            raise ValueError("vchunk requires --chunk in [0,8]")
        return _v1_chunk(args.chunk, overwrite=args.overwrite)
    if args.action == "vfinal":
        return _v1_finalize(overwrite=args.overwrite)
    if args.action == "screen":
        return _screen(ROUTES[args.route])
    if args.action == "aggregate":
        return _aggregate()
    raise ValueError(args.action)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", required=True, choices=("extract", "vchunk", "vfinal", "screen", "aggregate"))
    parser.add_argument("--route", choices=tuple(ROUTES))
    parser.add_argument("--chunk", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.action in {"extract", "screen"} and args.route is None:
        parser.error("--route is required for extract/screen")
    result = subprocess.run(command(args), cwd=ROOT, check=False)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
