#!/usr/bin/env python3
"""Run one bounded non-OOS pilot for a newly exact self/forecast-supervised route.

The route must already have passing 20-check smoke evidence.  The pilot reads only the
externally SHA-bound tournament cache interval [2019-07-01, 2025-07-01), samples the
same nine one-minute roots uniformly, and compares the best adapted checkpoint with the
untouched vanilla checkpoint on one fixed validation schedule.  It cannot authorize
promotion, full training, OOS access, deployment, paper trading, or live trading.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The smoke runner owns the family adapters and exact parity-path recovery. Importing it
# does not execute smoke; this avoids a second, drifting set of family-specific mechanics.
from scripts.smoke_extended_native_route import (  # noqa: E402
    Adapter,
    PARITY_TRACK,
    _atomic_json,
    _clone_tree,
    _parity_paths,
    _sha256,
)

from futures_foundation.finetune.native_route_pilot import (  # noqa: E402
    build_route_pilot_evidence,
    validate_route_pilot_evidence,
)
from futures_foundation.finetune.native_route_smoke import load_route_smoke_evidence  # noqa: E402
from futures_foundation.finetune.native_training_readiness import _launcher_record  # noqa: E402
from futures_foundation.finetune.routes import (  # noqa: E402
    chronos2_native,
    mantis_native,
    moirai2_research,
    moment_tasks,
    timesfm_lora,
    ttm_native,
)
from futures_foundation.finetune.tournament_data import (  # noqa: E402
    CACHE_MANIFEST,
    balanced_schedule,
    gather_parent,
    load_adaptation_data,
    schedule_fingerprint,
)

RAW_SCHEMA = "ffm_extended_native_bounded_pilot_raw_v1"
CACHE_DIR = ROOT / "output/foundation_tournament/data_cache_v3"
CACHE_MANIFEST_SHA256 = "41281860ff1ef3474e226d22a2df504e97f8d348839fce8d2438689d160b9e0a"
TICKERS = ("ES", "NQ", "RTY", "YM", "GC", "SI", "CL", "ZB", "ZN")
TIMEFRAMES = ("1min",)
TOTAL_STEPS = 128
EVAL_EVERY = 32
VALIDATION_BATCHES = 16
REQUIRED_RELATIVE_IMPROVEMENT = 0.01
SEED = 20260718
VALIDATION_SEED = 20260719

ROUTE_ALIASES = {
    "p01": "mantis_v1:R:official_crop_resize_contrastive",
    "p02": "mantis_v2:R:official_crop_resize_contrastive",
    "p03": "moment_small:F:forecast_full_raw_mse",
    "p04": "moment_small:F:forecast_head_only_raw_mse",
    "p05": "ttm_r2:F:full_model_raw_hf_trainer_forecast",
    "p06": "ttm_r2:F:head_prefix_raw_hf_trainer_forecast",
    "p07": "timesfm25:F:official_lora_forecast",
    "p08": "chronos_v2:F:official_fit_full",
    "p09": "chronos_v2:F:official_fit_lora",
    "p10": "moirai2_small:F:custom_scaled_pinball_research",
}
ELIGIBLE_ROUTES = frozenset(ROUTE_ALIASES.values())
SMOKE_DIRECTORIES = {
    "mantis_v1:R:official_crop_resize_contrastive": "mantis_v1_R_official_crop_resize_contrastive",
    "mantis_v2:R:official_crop_resize_contrastive": "mantis_v2_R_official_crop_resize_contrastive",
    "moment_small:F:forecast_full_raw_mse": "moment_small_F_forecast_full_raw_mse",
    "moment_small:F:forecast_head_only_raw_mse": "moment_small_F_forecast_head_only_raw_mse",
    "ttm_r2:F:full_model_raw_hf_trainer_forecast": "ttm_full",
    "ttm_r2:F:head_prefix_raw_hf_trainer_forecast": "ttm_head",
    "timesfm25:F:official_lora_forecast": "timesfm_lora",
    "chronos_v2:F:official_fit_full": "chronos2_full",
    "chronos_v2:F:official_fit_lora": "chronos2_lora",
    "moirai2_small:F:custom_scaled_pinball_research": "moirai_research",
}


def _smoke_evidence(route_key: str) -> Path:
    path = (
        ROOT / "output/native_training_smoke" / SMOKE_DIRECTORIES[route_key]
        / "smoke_evidence.json"
    ).resolve()
    evidence = load_route_smoke_evidence(path)
    if evidence["route_key"] != route_key or evidence["smoke_admitted"] is not True:
        raise ValueError(f"extended pilot requires passing canonical smoke for {route_key}")
    return path


def _configure(adapter: Adapter) -> None:
    """Replace the 20-step smoke configuration with one bounded pilot configuration."""
    route_key = adapter.route_key
    batch = adapter.batch_size
    if adapter.family == "mantis":
        adapter.config = mantis_native.RouteConfig(
            route_key, total_steps=TOTAL_STEPS, batch_size=batch, n_classes=3, seed=SEED,
        )
    elif adapter.family == "moment":
        adapter.config = moment_tasks.RouteConfig(
            route_key, total_steps=TOTAL_STEPS, batch_size=batch, n_classes=3, seed=SEED,
        )
    elif adapter.family == "ttm":
        adapter.config = ttm_native.RouteConfig(
            route_key, total_steps=TOTAL_STEPS, batch_size=batch, seed=SEED,
        )
    elif adapter.family == "timesfm":
        adapter.config = timesfm_lora.RouteConfig(
            total_steps=TOTAL_STEPS, batch_size=batch, seed=SEED,
        )
    elif adapter.family == "chronos2":
        learning_rate = 1e-6 if chronos2_native.ROUTES[route_key]["surface"] == "full" else 1e-5
        adapter.config = chronos2_native.RouteConfig(
            route_key, total_steps=TOTAL_STEPS, batch_size=batch,
            learning_rate=learning_rate, seed=SEED,
        )
    elif adapter.family == "moirai":
        adapter.config = moirai2_research.RouteConfig(
            total_steps=TOTAL_STEPS, batch_size=batch, seed=SEED,
        )
    else:  # pragma: no cover - eligibility closes this branch
        raise ValueError(f"unsupported pilot family: {adapter.family}")
    if hasattr(adapter.config, "resolved"):
        adapter.config.resolved()
    else:
        adapter.config.validate()


def _module_state(adapter: Adapter) -> dict[str, Any]:
    return {
        name: _clone_tree(module.state_dict())
        for name, module in adapter.loaded.modules.items()
    }


def _restore_modules(adapter: Adapter, state: Mapping[str, Any]) -> None:
    if set(state) != set(adapter.loaded.modules):
        raise ValueError("best pilot state module closure changed")
    for name, module in adapter.loaded.modules.items():
        module.load_state_dict(_clone_tree(state[name]), strict=True)
    adapter.eval_mode()


def _evaluate(
    adapter: Adapter,
    big: np.ndarray,
    starts: np.ndarray,
    *,
    seed: int,
) -> float:
    losses: list[float] = []
    weights: list[int] = []
    batch = adapter.batch_size
    for offset in range(0, len(starts), batch):
        selected = starts[offset:offset + batch]
        if adapter.is_contrastive and len(selected) < 2:
            raise ValueError("contrastive validation batch cannot contain fewer than two windows")
        parent = gather_parent(big, selected, adapter.parent_length)
        adapter.seed_all(int(seed + offset))
        losses.append(adapter.loss(parent))
        weights.append(len(selected))
    value = float(np.average(np.asarray(losses, np.float64), weights=np.asarray(weights)))
    if not np.isfinite(value):
        raise FloatingPointError("pilot validation objective is non-finite")
    return value


def _stream_counts(groups: np.ndarray, stream_ids: list[str]) -> dict[str, int]:
    counts = Counter(int(value) for value in np.asarray(groups, np.int64))
    return {stream_id: int(counts.get(index, 0)) for index, stream_id in enumerate(stream_ids)}


def _capture_training_state(
    adapter: Adapter,
    optimizer: Any,
    scheduler: Any,
    history: list[Mapping[str, Any]],
    *,
    schedule_sha256: str,
) -> dict[str, Any]:
    return adapter.module.capture_training_state(
        adapter.loaded,
        optimizer,
        scheduler,
        adapter.config,
        global_step=TOTAL_STEPS,
        sampler_cursor=TOTAL_STEPS,
        history=list(history),
        sampler_state={
            "cursor": TOTAL_STEPS,
            "schedule_kind": "authority_bound_balanced_schedule_v1",
            "schedule_sha256": schedule_sha256,
        },
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    route_key = ROUTE_ALIASES[args.route_alias]
    if route_key not in ELIGIBLE_ROUTES:
        raise ValueError("route is not eligible for an unlabeled native-objective pilot")
    output = Path(args.output).expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"pilot output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    smoke_path = _smoke_evidence(route_key)
    smoke = load_route_smoke_evidence(smoke_path)
    parity_root = Path(args.parity_root).expanduser().resolve()
    paths = _parity_paths(route_key, parity_root)
    adapter = Adapter(route_key, paths, args.device, SEED)
    _configure(adapter)
    if adapter.is_classification:
        raise ValueError("classification routes require governed label authority")

    load_start = perf_counter()
    streams, big, train_starts, validation_starts, groups = load_adaptation_data(
        CACHE_DIR,
        TICKERS,
        TIMEFRAMES,
        parent_length=adapter.parent_length,
        cache_manifest_sha256=CACHE_MANIFEST_SHA256,
        session_gap_capabilities=None,
        verbose=not args.quiet,
    )
    data_load_seconds = perf_counter() - load_start
    stream_ids = [str(stream["sid"]) for stream in streams]
    train_examples = TOTAL_STEPS * adapter.batch_size
    validation_examples = VALIDATION_BATCHES * adapter.batch_size
    train_schedule, train_groups = balanced_schedule(
        train_starts, groups["train_bounds"], train_examples, SEED,
    )
    validation_schedule, validation_groups = balanced_schedule(
        validation_starts, groups["val_bounds"], validation_examples, VALIDATION_SEED,
    )
    train_schedule_sha = schedule_fingerprint(train_schedule, train_groups)
    validation_schedule_sha = schedule_fingerprint(validation_schedule, validation_groups)
    exposure_path = output / "exposure_schedule.npz"
    np.savez_compressed(
        exposure_path,
        train_starts=train_schedule,
        train_groups=train_groups,
        validation_starts=validation_schedule,
        validation_groups=validation_groups,
        stream_ids=np.asarray(stream_ids),
    )

    adapter.reset()
    adapter.seed_all(SEED)
    vanilla_validation = _evaluate(
        adapter, big, validation_schedule, seed=VALIDATION_SEED,
    )
    optimizer = adapter.make_optimizer()
    scheduler = adapter.make_scheduler(optimizer)
    best_validation = float("inf")
    best_step = 0
    best_modules = _module_state(adapter)
    history: list[dict[str, Any]] = []
    running_loss = 0.0
    running_steps = 0
    train_start = perf_counter()
    last_row: dict[str, float] | None = None
    for step in range(TOTAL_STEPS):
        lo = step * adapter.batch_size
        parent = gather_parent(
            big, train_schedule[lo:lo + adapter.batch_size], adapter.parent_length,
        )
        adapter.seed_all(SEED + 10_000 + step)
        last_row = adapter.step(optimizer, scheduler, parent)
        running_loss += float(last_row["loss"])
        running_steps += 1
        completed = step + 1
        if completed % EVAL_EVERY == 0:
            validation_loss = _evaluate(
                adapter, big, validation_schedule, seed=VALIDATION_SEED,
            )
            record = {
                "step": completed,
                "train_loss": float(running_loss / running_steps),
                "validation_loss": validation_loss,
                "learning_rate": float(scheduler.get_last_lr()[0]),
                "grad_norm": float(last_row["grad_norm"]),
            }
            history.append(record)
            if not args.quiet:
                print(
                    f"[extended pilot] {route_key} step={completed} "
                    f"train={record['train_loss']:.6f} val={validation_loss:.6f}",
                    flush=True,
                )
            if validation_loss < best_validation:
                best_validation = validation_loss
                best_step = completed
                best_modules = _module_state(adapter)
            running_loss = 0.0
            running_steps = 0
    training_seconds = perf_counter() - train_start
    if best_step < 1 or last_row is None:
        raise RuntimeError("extended pilot produced no validation checkpoint")

    state = _capture_training_state(
        adapter, optimizer, scheduler, history, schedule_sha256=train_schedule_sha,
    )
    training_state_path = output / "training_state.pt"
    training_state_identity = adapter.save(training_state_path, state)

    _restore_modules(adapter, best_modules)
    deployment_path = output / "deployment_bundle.pt"
    deployment_identity = adapter.save(deployment_path, adapter.build_export())
    adapted_validation = _evaluate(
        adapter, big, validation_schedule, seed=VALIDATION_SEED,
    )
    if not np.isclose(adapted_validation, best_validation, rtol=0.0, atol=1e-8):
        raise RuntimeError("best pilot validation objective is not reproducible")
    relative_improvement = float(
        (vanilla_validation - adapted_validation) / max(abs(vanilla_validation), 1e-12)
    )

    exposure = {
        "sampling_kind": "uniform_stream_then_uniform_window_v1",
        "train_schedule_sha256": train_schedule_sha,
        "validation_schedule_sha256": validation_schedule_sha,
        "train_examples": int(train_examples),
        "validation_examples": int(validation_examples),
        "train_stream_counts": _stream_counts(train_groups, stream_ids),
        "validation_stream_counts": _stream_counts(validation_groups, stream_ids),
        "seed": SEED,
        "validation_seed": VALIDATION_SEED,
    }
    metrics = {
        "vanilla_validation_loss": float(vanilla_validation),
        "adapted_validation_loss": float(adapted_validation),
        "relative_validation_improvement": relative_improvement,
        "required_relative_improvement": REQUIRED_RELATIVE_IMPROVEMENT,
        "best_step": int(best_step),
        "history": history,
    }
    raw = {
        "schema_version": RAW_SCHEMA,
        "route_key": route_key,
        "status": "complete",
        "oos_read": False,
        "window_gap_policy": "exact_cadence_hard_boundary_no_session_inference_v1",
        "session_gap_capability": None,
        "scope": (
            "research_noncommercial_only"
            if route_key == moirai2_research.ROUTE_KEY
            else "development_non_oos"
        ),
        "config": (
            adapter.config.resolved()
            if hasattr(adapter.config, "resolved")
            else vars(adapter.config)
        ),
        "data": {
            "cache_dir": str(CACHE_DIR.resolve()),
            "cache_manifest_sha256": CACHE_MANIFEST_SHA256,
            "streams": stream_ids,
            "train_windows": int(len(train_starts)),
            "validation_windows": int(len(validation_starts)),
            "data_load_seconds": float(data_load_seconds),
        },
        "exposure": exposure,
        "metrics": metrics,
        "runtime": {
            "python": str(paths["python"]),
            "device": args.device,
            "training_seconds": float(training_seconds),
            "peak_cuda_bytes": (
                int(__import__("torch").cuda.max_memory_allocated(args.device))
                if __import__("torch").cuda.is_available() and args.device.startswith("cuda")
                else 0
            ),
        },
        "artifacts": {
            "training_state": training_state_identity,
            "deployment_bundle": deployment_identity,
        },
        "promotion_admitted": False,
        "full_training_admitted": False,
        "live_trading_ready": False,
    }
    raw_path = _atomic_json(output / "raw_pilot_report.json", raw)
    launcher = _launcher_record(route_key)
    evidence = build_route_pilot_evidence(
        route_key=route_key,
        smoke_evidence_path=smoke_path,
        executor_path=Path(adapter.module.__file__).resolve(),
        executor_entrypoint=launcher["entrypoint"],
        cache_dir=CACHE_DIR,
        cache_manifest_sha256=CACHE_MANIFEST_SHA256,
        stream_ids=stream_ids,
        exposure=exposure,
        metrics=metrics,
        artifacts={
            "model_snapshot": paths["model_snapshot"],
            "smoke_evidence": smoke_path,
            "cache_manifest": CACHE_DIR / CACHE_MANIFEST,
            "exposure_schedule": exposure_path,
            "training_state": training_state_path,
            "deployment_bundle": deployment_path,
            "raw_report": raw_path,
            "pilot_runner": Path(__file__).resolve(),
        },
    )
    validate_route_pilot_evidence(evidence)
    evidence_path = _atomic_json(output / "pilot_evidence.json", evidence)
    return {
        "status": "pass" if evidence["native_objective_survived"] else "eliminated",
        "route_key": route_key,
        "pilot_completed": True,
        "native_objective_survived": evidence["native_objective_survived"],
        "promotion_admitted": False,
        "full_training_admitted": False,
        "evidence": {
            "path": str(evidence_path),
            "sha256": _sha256(evidence_path),
            "content_sha256": evidence["evidence_sha256"],
        },
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-alias", required=True, choices=sorted(ROUTE_ALIASES))
    parser.add_argument(
        "--parity-root",
        default=str(ROOT / "output/native_parity_evidence_current_config_v1"),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] == "eliminated":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
