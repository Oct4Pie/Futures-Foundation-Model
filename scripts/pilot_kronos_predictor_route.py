#!/usr/bin/env python3
"""Run the bounded, non-OOS Kronos Mini predictor pilot.

The route consumes a passing predictor smoke plus the same surviving tokenizer parent
bound by that smoke.  Training uses full-parent next-token cross-entropy and fixed
America/Chicago calendar stamps.  The result cannot authorize promotion, full
training, OOS access, deployment, or trading.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from futures_foundation.finetune.native_route_pilot import (
    build_route_pilot_evidence,
    validate_route_pilot_evidence,
)
from futures_foundation.finetune.native_route_smoke import load_route_smoke_evidence
from futures_foundation.finetune.routes import kronos_predictor as route
from futures_foundation.finetune.tournament_data import (
    CACHE_MANIFEST,
    balanced_schedule,
    gather_parent,
    gather_time_features,
    load_adaptation_data,
    schedule_fingerprint,
)


RAW_SCHEMA = "ffm_kronos_mini_predictor_bounded_pilot_raw_v1"


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: str | Path, value: object) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(target) + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)
    return target


def _evaluate(
    loaded: route.LoadedRoute,
    big: np.ndarray,
    schedule: np.ndarray,
    stamps: np.ndarray,
    *,
    batch_size: int,
    seed: int,
) -> float:
    torch = route._torch()
    if len(schedule) % batch_size or stamps.shape[0] != len(schedule):
        raise ValueError("validation exposure is not batch aligned")
    device_index = (
        torch.device(loaded.device).index
        if str(loaded.device).startswith("cuda")
        and torch.device(loaded.device).index is not None
        else torch.cuda.current_device()
        if str(loaded.device).startswith("cuda")
        else None
    )
    devices = [] if device_index is None else [device_index]
    loaded.predictor.eval()
    loaded.tokenizer.eval()
    total = 0.0
    with torch.no_grad(), torch.random.fork_rng(devices=devices, enabled=True):
        torch.manual_seed(int(seed))
        if torch.cuda.is_available() and str(loaded.device).startswith("cuda"):
            torch.cuda.manual_seed_all(int(seed))
        for start in range(0, len(schedule), batch_size):
            parent = gather_parent(
                big,
                schedule[start:start + batch_size],
                route.PARENT_LENGTH,
            )
            total += float(
                route.native_loss(
                    loaded,
                    parent,
                    stamps[start:start + batch_size],
                ).detach().cpu()
            )
    return total / (len(schedule) // batch_size)


def _stream_counts(group_ids: np.ndarray, stream_ids: list[str]) -> dict[str, int]:
    counts = np.bincount(
        np.asarray(group_ids, np.int64), minlength=len(stream_ids)
    )
    return {
        stream_id: int(counts[index])
        for index, stream_id in enumerate(stream_ids)
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch = route._torch()
    output = Path(args.output).expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"pilot output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    smoke = load_route_smoke_evidence(args.smoke_evidence)
    if smoke["route_key"] != route.ROUTE_KEY or smoke["smoke_admitted"] is not True:
        raise ValueError("Kronos Mini predictor pilot requires passing route smoke")
    if Path(smoke["executor"]["path"]).resolve() != Path(route.__file__).resolve():
        raise ValueError("Kronos Mini predictor smoke executor differs from current code")
    expected_artifacts = {
        "model_snapshot": args.model_snapshot,
        "tokenizer_snapshot": args.tokenizer_snapshot,
        "source_runtime": args.source_runtime,
        "parent_route_evidence": args.parent_pilot_evidence,
        "parent_route_bundle": args.parent_tokenizer_bundle,
    }
    for name, supplied in expected_artifacts.items():
        if Path(smoke["artifacts"][name]["path"]).resolve() != Path(supplied).resolve():
            raise ValueError(f"Kronos Mini predictor {name} differs from smoke")

    config = route.RouteConfig(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        max_gradient_norm=args.grad_clip,
        total_steps=args.steps,
        seed=args.seed,
    )
    config.validate()
    if not 128 <= config.total_steps <= 2048:
        raise ValueError("bounded predictor pilot steps must lie in [128,2048]")
    if args.eval_every < 1 or config.total_steps % args.eval_every:
        raise ValueError("eval_every must divide total pilot steps")
    if args.validation_batches < 1:
        raise ValueError("validation_batches must be positive")
    if not 0.0 <= args.required_relative_improvement < 1.0:
        raise ValueError("required_relative_improvement must lie in [0,1)")

    tickers = tuple(
        value.strip().upper() for value in args.tickers.split(",") if value.strip()
    )
    timeframes = tuple(
        value.strip() for value in args.timeframes.split(",") if value.strip()
    )
    load_started = perf_counter()
    streams, big, train_starts, validation_starts, groups = load_adaptation_data(
        args.cache_dir,
        tickers,
        timeframes,
        parent_length=route.PARENT_LENGTH,
        cache_manifest_sha256=args.cache_manifest_sha256,
        session_gap_capabilities=None,
        verbose=not args.quiet,
    )
    data_load_seconds = perf_counter() - load_started
    stream_ids = [str(stream["sid"]) for stream in streams]
    train_examples = config.total_steps * config.batch_size
    validation_examples = args.validation_batches * config.batch_size
    train_schedule, train_groups = balanced_schedule(
        train_starts,
        groups["train_bounds"],
        train_examples,
        config.seed,
    )
    validation_schedule, validation_groups = balanced_schedule(
        validation_starts,
        groups["val_bounds"],
        validation_examples,
        args.validation_seed,
    )
    train_stamps = gather_time_features(
        train_schedule,
        train_groups,
        streams,
        groups["row_bounds"],
        route.PARENT_LENGTH,
        timezone=route.VENUE_TIMEZONE,
    )
    validation_stamps = gather_time_features(
        validation_schedule,
        validation_groups,
        streams,
        groups["row_bounds"],
        route.PARENT_LENGTH,
        timezone=route.VENUE_TIMEZONE,
    )
    train_schedule_sha = schedule_fingerprint(train_schedule, train_groups)
    validation_schedule_sha = schedule_fingerprint(
        validation_schedule, validation_groups
    )
    exposure_path = output / "exposure_schedule.npz"
    temporary_exposure = Path(str(exposure_path) + f".{os.getpid()}.tmp.npz")
    np.savez_compressed(
        temporary_exposure,
        train_starts=train_schedule,
        train_groups=train_groups,
        train_stamps=train_stamps,
        validation_starts=validation_schedule,
        validation_groups=validation_groups,
        validation_stamps=validation_stamps,
        stream_ids=np.asarray(stream_ids),
    )
    os.replace(temporary_exposure, exposure_path)

    route.seed_everything(config.seed)
    loaded = route.load_route(
        model_snapshot=args.model_snapshot,
        tokenizer_snapshot=args.tokenizer_snapshot,
        source_runtime=args.source_runtime,
        parent_pilot_evidence=args.parent_pilot_evidence,
        parent_tokenizer_bundle=args.parent_tokenizer_bundle,
        device=args.device,
    )
    vanilla_validation = _evaluate(
        loaded,
        big,
        validation_schedule,
        validation_stamps,
        batch_size=config.batch_size,
        seed=args.validation_seed,
    )
    route.seed_everything(config.seed)
    optimizer = route.make_optimizer(loaded.predictor, config)
    scheduler = route.make_scheduler(optimizer, config)
    best_validation = float("inf")
    best_step = 0
    best_model = route.predictor_state_cpu(loaded.predictor)
    history: list[dict[str, Any]] = []
    running = 0.0
    running_steps = 0
    train_started = perf_counter()
    for step in range(config.total_steps):
        lo = step * config.batch_size
        parent = gather_parent(
            big,
            train_schedule[lo:lo + config.batch_size],
            route.PARENT_LENGTH,
        )
        row = route.optimizer_step(
            loaded,
            optimizer,
            scheduler,
            parent,
            train_stamps[lo:lo + config.batch_size],
            max_gradient_norm=config.max_gradient_norm,
        )
        running += row["loss"]
        running_steps += 1
        completed = step + 1
        if completed % args.eval_every == 0:
            validation_loss = _evaluate(
                loaded,
                big,
                validation_schedule,
                validation_stamps,
                batch_size=config.batch_size,
                seed=args.validation_seed,
            )
            record = {
                "step": completed,
                "train_loss": running / running_steps,
                "validation_loss": validation_loss,
                "learning_rate": float(scheduler.get_last_lr()[0]),
                "grad_norm": row["grad_norm"],
            }
            history.append(record)
            if not args.quiet:
                print(
                    f"[kronos-mini predictor pilot] step={completed} "
                    f"train={record['train_loss']:.6f} "
                    f"val={validation_loss:.6f}",
                    flush=True,
                )
            if validation_loss < best_validation:
                best_validation = validation_loss
                best_step = completed
                best_model = route.predictor_state_cpu(loaded.predictor)
            running = 0.0
            running_steps = 0
    training_seconds = perf_counter() - train_started
    if best_step < 1:
        raise RuntimeError("Kronos Mini predictor pilot produced no checkpoint")

    state = route.capture_training_state(
        loaded=loaded,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        global_step=config.total_steps,
        sampler_cursor=config.total_steps,
        history=history,
        sampler_state={
            "cursor": config.total_steps,
            "schedule_kind": "authority_bound_balanced_schedule_v1",
            "schedule_sha256": train_schedule_sha,
        },
    )
    training_state_path = output / "kronos_mini_predictor_pilot.train.pt"
    training_state_identity = route.save_training_state(training_state_path, state)

    loaded.predictor.load_state_dict(best_model, strict=True)
    loaded.predictor.eval()
    export = route.build_export_bundle(loaded)
    deployment_path = output / "kronos_mini_predictor_pilot.bundle.pt"
    deployment_identity = route.save_export_bundle(deployment_path, export)
    adapted_validation = _evaluate(
        loaded,
        big,
        validation_schedule,
        validation_stamps,
        batch_size=config.batch_size,
        seed=args.validation_seed,
    )
    if not np.isclose(adapted_validation, best_validation, rtol=0.0, atol=1e-8):
        raise RuntimeError("Kronos Mini best predictor loss is not reproducible")
    relative_improvement = (
        vanilla_validation - adapted_validation
    ) / max(abs(vanilla_validation), 1e-12)

    exposure = {
        "sampling_kind": "uniform_stream_then_uniform_window_v1",
        "train_schedule_sha256": train_schedule_sha,
        "validation_schedule_sha256": validation_schedule_sha,
        "train_examples": int(train_examples),
        "validation_examples": int(validation_examples),
        "train_stream_counts": _stream_counts(train_groups, stream_ids),
        "validation_stream_counts": _stream_counts(validation_groups, stream_ids),
        "seed": int(config.seed),
        "validation_seed": int(args.validation_seed),
    }
    metrics = {
        "vanilla_validation_loss": float(vanilla_validation),
        "adapted_validation_loss": float(adapted_validation),
        "relative_validation_improvement": float(relative_improvement),
        "required_relative_improvement": float(args.required_relative_improvement),
        "best_step": int(best_step),
        "history": history,
    }
    raw = {
        "schema_version": RAW_SCHEMA,
        "route_key": route.ROUTE_KEY,
        "status": "complete",
        "oos_read": False,
        "window_gap_policy": "exact_cadence_hard_boundary_no_session_inference_v1",
        "session_gap_capability": None,
        "venue_timezone": route.VENUE_TIMEZONE,
        "config": {
            **config.__dict__,
            "validation_batches": args.validation_batches,
            "eval_every": args.eval_every,
            "validation_seed": args.validation_seed,
        },
        "model_identity": loaded.identity,
        "data": {
            "cache_dir": str(Path(args.cache_dir).resolve()),
            "cache_manifest_sha256": args.cache_manifest_sha256,
            "streams": stream_ids,
            "train_windows": int(len(train_starts)),
            "validation_windows": int(len(validation_starts)),
            "data_load_seconds": float(data_load_seconds),
        },
        "exposure": exposure,
        "metrics": metrics,
        "runtime": {
            "training_seconds": float(training_seconds),
            "peak_cuda_bytes": (
                int(torch.cuda.max_memory_allocated(args.device))
                if torch.cuda.is_available() and str(args.device).startswith("cuda")
                else 0
            ),
        },
        "artifacts": {
            "training_state": training_state_identity,
            "deployment_bundle": deployment_identity,
            "exposure_schedule": {
                "path": str(exposure_path.resolve()),
                "sha256": _sha256(exposure_path),
                "bytes": exposure_path.stat().st_size,
            },
        },
        "promotion_admitted": False,
        "full_training_admitted": False,
        "live_trading_ready": False,
    }
    raw_path = _atomic_json(output / "raw_pilot_report.json", raw)
    evidence = build_route_pilot_evidence(
        route_key=route.ROUTE_KEY,
        smoke_evidence_path=args.smoke_evidence,
        executor_path=route.__file__,
        executor_entrypoint="native_loss/public_greedy_forecast",
        cache_dir=args.cache_dir,
        cache_manifest_sha256=args.cache_manifest_sha256,
        stream_ids=stream_ids,
        exposure=exposure,
        metrics=metrics,
        artifacts={
            "model_snapshot": Path(args.model_snapshot).resolve(),
            "smoke_evidence": Path(args.smoke_evidence).resolve(),
            "cache_manifest": Path(args.cache_dir).resolve() / CACHE_MANIFEST,
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
        "route_key": route.ROUTE_KEY,
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
    parser.add_argument("--model-snapshot", required=True)
    parser.add_argument("--tokenizer-snapshot", required=True)
    parser.add_argument("--source-runtime", required=True)
    parser.add_argument("--parent-pilot-evidence", required=True)
    parser.add_argument("--parent-tokenizer-bundle", required=True)
    parser.add_argument("--smoke-evidence", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--cache-manifest-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tickers", default="ES,NQ,RTY,YM,GC,SI,CL,ZB,ZN")
    parser.add_argument("--timeframes", default="1min")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=32)
    parser.add_argument("--validation-batches", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=4e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--validation-seed", type=int, default=20260719)
    parser.add_argument("--required-relative-improvement", type=float, default=0.01)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] == "eliminated":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
