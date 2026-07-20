#!/usr/bin/env python3
"""Run the exact Kronos Mini hierarchical autoregressive predictor smoke.

The frozen tokenizer is loaded only from a surviving tokenizer-pilot bundle.  The
predictor is initialized from the pinned vanilla Mini snapshot and trained with the
upstream full-parent next-token objective.  This command reads no market rows and
cannot grant pilot, full-training, OOS, deployment, or trading admission.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from typing import Any, Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from futures_foundation.finetune import ssl_data
from futures_foundation.finetune.native_contract_harness import (
    CheckResult,
    control_rejection_check,
    forward_backward_check,
    future_corruption_check,
    interruption_resume_parity_check,
    loss_decrease_check,
    negative_price_behavior_check,
    parity_check,
    performance_check,
    prefix_invariance_check,
    rejection_check,
)
from futures_foundation.finetune.native_route_smoke import (
    build_route_smoke_evidence,
    validate_route_smoke_evidence,
)
from futures_foundation.finetune.routes import kronos_predictor as route


RAW_SCHEMA = "ffm_kronos_mini_predictor_smoke_raw_v1"


def _sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: str | Path, value: object) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(target) + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)
    return target


def _fixture(batch: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    time = np.arange(route.PARENT_LENGTH, dtype=np.float64)
    level = rng.uniform(35, 90, batch)
    slope = rng.choice((-0.06, -0.035, 0.035, 0.06), size=batch)
    phase = rng.uniform(0, 2 * np.pi, batch)
    regime = np.where(time < 384, 1.0, 1.8)
    close = (
        level[:, None]
        + slope[:, None] * time
        + regime[None, :] * 0.45 * np.sin(time[None, :] / 11 + phase[:, None])
    )
    open_ = np.concatenate((close[:, :1], close[:, :-1]), axis=1)
    spread = 0.18 + 0.04 * np.abs(np.sin(time / 17))[None, :]
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = (
        900
        + rng.uniform(30, 90, batch)[:, None]
        + 0.8 * time
        + 35 * np.sin(time[None, :] / 29 + phase[:, None])
    )
    parent = route.parent_array(
        np.stack((open_, high, low, close, volume), axis=-1).astype(np.float32)
    )
    start = pd.Timestamp("2024-02-01 00:00:00", tz="UTC")
    timestamps = np.stack(
        [
            pd.date_range(
                start + pd.Timedelta(days=index),
                periods=route.PARENT_LENGTH,
                freq="1min",
                tz="UTC",
            ).asi8
            for index in range(batch)
        ],
        axis=0,
    )
    return parent, timestamps


def _negative(batch: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    parent, timestamps = _fixture(batch, seed)
    parent = parent.copy()
    shift = float(np.max(parent[:, :, :4]) + 10)
    parent[:, :, :4] -= shift
    return route.parent_array(parent), timestamps


def _fresh(base: route.LoadedRoute) -> route.LoadedRoute:
    loaded = route.LoadedRoute(
        tokenizer=copy.deepcopy(base.tokenizer).to(base.device),
        predictor=copy.deepcopy(base.predictor).to(base.device),
        public_predictor_class=base.public_predictor_class,
        identity=copy.deepcopy(base.identity),
        device=base.device,
    )
    loaded.tokenizer.eval()
    for parameter in loaded.tokenizer.parameters():
        parameter.requires_grad_(False)
    return loaded


def _loss(
    loaded: route.LoadedRoute,
    parent: np.ndarray,
    stamps: np.ndarray,
    *,
    seed: int,
) -> float:
    torch = route._torch()
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
    with torch.no_grad(), torch.random.fork_rng(devices=devices, enabled=True):
        torch.manual_seed(int(seed))
        if torch.cuda.is_available() and str(loaded.device).startswith("cuda"):
            torch.cuda.manual_seed_all(int(seed))
        return float(route.native_loss(loaded, parent, stamps).detach().cpu())


def _mismatched_loss(
    loaded: route.LoadedRoute,
    source_parent: np.ndarray,
    source_stamps: np.ndarray,
    target_parent: np.ndarray,
) -> Any:
    coarse_logits, fine_logits, _, _ = route.native_logits(
        loaded, source_parent, source_stamps
    )
    target_coarse, target_fine = route.native_tokens(loaded, target_parent)
    return loaded.predictor.head.compute_loss(
        coarse_logits,
        fine_logits,
        target_coarse[:, 1:],
        target_fine[:, 1:],
    )[0]


def _train_control(
    base: route.LoadedRoute,
    parent: np.ndarray,
    stamps: np.ndarray,
    validation_parent: np.ndarray,
    validation_stamps: np.ndarray,
    config: route.RouteConfig,
    *,
    target_parent: np.ndarray | None = None,
) -> tuple[list[float], float]:
    torch = route._torch()
    route.seed_everything(config.seed)
    loaded = _fresh(base)
    optimizer = route.make_optimizer(loaded.predictor, config)
    scheduler = route.make_scheduler(optimizer, config)
    losses: list[float] = []
    for _ in range(config.total_steps):
        if target_parent is None:
            row = route.optimizer_step(
                loaded,
                optimizer,
                scheduler,
                parent,
                stamps,
                max_gradient_norm=config.max_gradient_norm,
            )
            losses.append(row["loss"])
        else:
            loaded.predictor.train()
            optimizer.zero_grad(set_to_none=True)
            loss = _mismatched_loss(
                loaded, parent, stamps, target_parent
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                loaded.predictor.parameters(), config.max_gradient_norm
            )
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach().cpu()))
    return losses, _loss(
        loaded,
        validation_parent,
        validation_stamps,
        seed=config.seed + 50_000,
    )


def _result(ok: bool, metrics: Mapping[str, Any], reason: str) -> CheckResult:
    return CheckResult(
        status="pass" if ok else "fail",
        metrics=dict(metrics),
        reason=None if ok else reason,
    )


def _boundary(kind: str) -> CheckResult:
    delta = pd.Timedelta("1min")
    rows = 660
    timestamps = pd.date_range(
        "2024-01-01", periods=rows, freq=delta, tz="UTC"
    )

    def validate(case: Mapping[str, Any]) -> None:
        starts = ssl_data.window_starts(
            np.asarray(case["indices"], np.int64),
            route.PARENT_LENGTH,
            timestamps=case["timestamps"],
            expected_delta=delta,
            segment_ids=case.get("segments"),
        )
        if not len(starts):
            raise ValueError("invalid boundary rejected")

    if kind == "contract_roll":
        case = {
            "indices": np.arange(rows),
            "timestamps": timestamps,
            "segments": np.asarray(["A"] * 330 + ["B"] * 330),
        }
    elif kind == "session_gap":
        case = {
            "indices": np.arange(rows),
            "timestamps": timestamps[:330].append(
                timestamps[330:] + pd.Timedelta("1h")
            ),
            "segments": np.asarray(["A"] * rows),
        }
    elif kind == "split_boundary":
        case = {
            "indices": np.r_[np.arange(330), np.arange(430, 760)],
            "timestamps": pd.date_range(
                "2024-01-01", periods=760, freq=delta, tz="UTC"
            ),
            "segments": np.asarray(["A"] * 760),
        }
    else:
        case = {
            "indices": np.arange(330),
            "timestamps": timestamps,
            "segments": np.asarray(["A"] * rows),
        }
    return rejection_check(validate, {kind: case})


def _state_digest(value: Any) -> str:
    torch = route._torch()
    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor\0")
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
            digest.update(tensor.numpy().tobytes())
            return
        if isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            digest.update(b"ndarray\0")
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            digest.update(array.tobytes())
            return
        if isinstance(item, Mapping):
            digest.update(b"mapping\0")
            for key in sorted(item, key=lambda current: str(current)):
                update(str(key))
                update(item[key])
            return
        if isinstance(item, (list, tuple)):
            digest.update(b"sequence\0")
            digest.update(np.asarray([len(item)], dtype=np.int64).tobytes())
            for child in item:
                update(child)
            return
        digest.update(
            json.dumps(item, sort_keys=True, default=str, allow_nan=False).encode()
        )
        digest.update(b"\0")

    update(value)
    return digest.hexdigest()


def _compact_state(
    state: Mapping[str, Any],
    loaded: route.LoadedRoute,
    parent: np.ndarray,
    stamps: np.ndarray,
) -> dict[str, Any]:
    return {
        "state_sha256": _state_digest(state),
        "model_sha256": _state_digest(state["model"]),
        "optimizer_sha256": _state_digest(state["optimizer"]),
        "scheduler_sha256": _state_digest(state["scheduler"]),
        "rng_sha256": _state_digest(
            {
                "python_rng": state["python_rng"],
                "numpy_rng": state["numpy_rng"],
                "torch_cpu_rng": state["torch_cpu_rng"],
                "torch_cuda_rng": state["torch_cuda_rng"],
            }
        ),
        "history": state["history"],
        "global_step": state["global_step"],
        "sampler": state["sampler"],
        "next_loss": _loss(
            loaded,
            parent,
            stamps,
            seed=20269876,
        ),
    }


def _resume_check(
    base: route.LoadedRoute,
    parent: np.ndarray,
    stamps: np.ndarray,
    config: route.RouteConfig,
    path: Path,
) -> CheckResult:
    steps = 4

    def full() -> dict[str, Any]:
        route.seed_everything(config.seed + 31)
        loaded = _fresh(base)
        optimizer = route.make_optimizer(loaded.predictor, config)
        scheduler = route.make_scheduler(optimizer, config)
        history = []
        for index in range(steps):
            history.append(
                {
                    "step": index + 1,
                    **route.optimizer_step(
                        loaded,
                        optimizer,
                        scheduler,
                        parent,
                        stamps,
                        max_gradient_norm=config.max_gradient_norm,
                    ),
                }
            )
        state = route.capture_training_state(
            loaded=loaded,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            global_step=steps,
            sampler_cursor=steps,
            history=history,
        )
        return _compact_state(state, loaded, parent, stamps)

    def resumed() -> dict[str, Any]:
        route.seed_everything(config.seed + 31)
        loaded = _fresh(base)
        optimizer = route.make_optimizer(loaded.predictor, config)
        scheduler = route.make_scheduler(optimizer, config)
        history = []
        for index in range(2):
            history.append(
                {
                    "step": index + 1,
                    **route.optimizer_step(
                        loaded,
                        optimizer,
                        scheduler,
                        parent,
                        stamps,
                        max_gradient_norm=config.max_gradient_norm,
                    ),
                }
            )
        partial = route.capture_training_state(
            loaded=loaded,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            global_step=2,
            sampler_cursor=2,
            history=history,
        )
        route.save_training_state(path, partial)
        reopened = route.load_training_state(path)
        resumed_loaded = _fresh(base)
        resumed_optimizer = route.make_optimizer(resumed_loaded.predictor, config)
        resumed_scheduler = route.make_scheduler(resumed_optimizer, config)
        step, cursor, history = route.restore_training_state(
            reopened,
            loaded=resumed_loaded,
            optimizer=resumed_optimizer,
            scheduler=resumed_scheduler,
            config=config,
        )
        if (step, cursor) != (2, 2):
            raise RuntimeError("Kronos Mini predictor resume cursor drifted")
        for index in range(2, steps):
            history.append(
                {
                    "step": index + 1,
                    **route.optimizer_step(
                        resumed_loaded,
                        resumed_optimizer,
                        resumed_scheduler,
                        parent,
                        stamps,
                        max_gradient_norm=config.max_gradient_norm,
                    ),
                }
            )
        final = route.capture_training_state(
            loaded=resumed_loaded,
            optimizer=resumed_optimizer,
            scheduler=resumed_scheduler,
            config=config,
            global_step=steps,
            sampler_cursor=steps,
            history=history,
        )
        return _compact_state(final, resumed_loaded, parent, stamps)

    return interruption_resume_parity_check(full, resumed, atol=0, rtol=0)


def _forecast_times(timestamps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    context = timestamps[:, :route.CONTEXT_LENGTH]
    delta = context[:, -1] - context[:, -2]
    future = context[:, -1, None] + delta[:, None] * np.arange(
        1, route.HORIZON_LENGTH + 1, dtype=np.int64
    )[None, :]
    return context, future


def run(args: argparse.Namespace) -> dict[str, Any]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    output = Path(args.output).expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"smoke output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    config = route.RouteConfig(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        total_steps=args.steps,
        seed=args.seed,
    )
    config.validate()
    if config.total_steps != 20:
        raise ValueError("canonical Kronos Mini predictor smoke requires 20 steps")

    train, train_times = _fixture(config.batch_size, config.seed)
    validation, validation_times = _fixture(config.batch_size, config.seed + 1)
    train_stamps = route.calendar_stamps(train_times)
    validation_stamps = route.calendar_stamps(validation_times)
    fixture = output / "synthetic_fixture.npz"
    temporary_fixture = Path(str(fixture) + f".{os.getpid()}.tmp.npz")
    np.savez_compressed(
        temporary_fixture,
        train=train,
        train_times=train_times,
        validation=validation,
        validation_times=validation_times,
    )
    os.replace(temporary_fixture, fixture)
    fixture_manifest = _json(
        output / "synthetic_fixture.manifest.json",
        {
            "schema_version": "ffm_kronos_mini_predictor_smoke_fixture_v1",
            "market_data_read": False,
            "oos_read": False,
            "venue_timezone": route.VENUE_TIMEZONE,
            "artifact": {
                "path": str(fixture.resolve()),
                "sha256": _sha(fixture),
                "bytes": fixture.stat().st_size,
            },
        },
    )

    base = route.load_route(
        model_snapshot=args.model_snapshot,
        tokenizer_snapshot=args.tokenizer_snapshot,
        source_runtime=args.source_runtime,
        parent_pilot_evidence=args.parent_pilot_evidence,
        parent_tokenizer_bundle=args.parent_tokenizer_bundle,
        device=args.device,
    )

    checks: dict[str, CheckResult] = {}
    one = _fresh(base)
    one_optimizer = route.make_optimizer(one.predictor, config)
    one_scheduler = route.make_scheduler(one_optimizer, config)
    checks["one_batch_forward_backward"] = forward_backward_check(
        lambda: route.optimizer_step(
            one,
            one_optimizer,
            one_scheduler,
            train,
            train_stamps,
            max_gradient_norm=config.max_gradient_norm,
        )
    )

    route.seed_everything(config.seed)
    real = _fresh(base)
    optimizer = route.make_optimizer(real.predictor, config)
    scheduler = route.make_scheduler(optimizer, config)
    checks["controlled_learnable_loss_decrease"] = loss_decrease_check(
        lambda: _loss(
            real,
            train,
            train_stamps,
            seed=config.seed + 40_000,
        ),
        lambda: route.optimizer_step(
            real,
            optimizer,
            scheduler,
            train,
            train_stamps,
            max_gradient_norm=config.max_gradient_norm,
        ),
        steps=config.total_steps,
        min_relative_decrease=0.01,
        tail=2,
    )
    real_validation = _loss(
        real,
        validation,
        validation_stamps,
        seed=config.seed + 50_000,
    )

    target = np.roll(train, shift=1, axis=0)
    shuffled_losses, shuffled_validation = _train_control(
        base,
        train,
        train_stamps,
        validation,
        validation_stamps,
        config,
        target_parent=target,
    )
    rng = np.random.default_rng(config.seed + 91)
    destroyed = np.stack(
        [train[row, rng.permutation(route.PARENT_LENGTH)] for row in range(len(train))],
        axis=0,
    ).copy()
    destroyed_losses, destroyed_validation = _train_control(
        base,
        destroyed,
        train_stamps,
        validation,
        validation_stamps,
        config,
    )
    controls = control_rejection_check(
        real_validation,
        [shuffled_validation],
        [destroyed_validation],
        margin=0.0,
        higher_is_better=False,
    )
    checks["shuffle_control_rejection"] = _result(
        real_validation <= shuffled_validation,
        {
            **controls.metrics,
            "shuffle_final_train_loss": shuffled_losses[-1],
        },
        "real predictor did not beat mismatched next-token targets",
    )
    checks["time_destroyed_control_rejection"] = _result(
        real_validation <= destroyed_validation,
        {
            **controls.metrics,
            "time_destroyed_final_train_loss": destroyed_losses[-1],
        },
        "real predictor did not beat time-destroyed parents",
    )

    interrupted = output / "interrupted.train.pt"
    checks["exact_interruption_resume_trajectory"] = _resume_check(
        base, train, train_stamps, config, interrupted
    )

    history = [
        {"step": index + 1, "train_loss": float(value)}
        for index, value in enumerate(
            checks["controlled_learnable_loss_decrease"].metrics["losses"][1:]
        )
    ]
    state = route.capture_training_state(
        loaded=real,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        global_step=config.total_steps,
        sampler_cursor=config.total_steps,
        history=history,
    )
    state_path = output / "predictor.train.pt"
    route.save_training_state(state_path, state)
    bundle = route.build_export_bundle(real)
    bundle_path = output / "predictor.bundle.pt"
    route.save_export_bundle(bundle_path, bundle)
    exported, reopened_bundle = route.load_export_bundle(
        bundle_path,
        model_snapshot=args.model_snapshot,
        tokenizer_snapshot=args.tokenizer_snapshot,
        source_runtime=args.source_runtime,
        parent_pilot_evidence=args.parent_pilot_evidence,
        parent_tokenizer_bundle=args.parent_tokenizer_bundle,
        device=args.device,
    )

    context_times, future_times = _forecast_times(validation_times)
    route.seed_everything(config.seed + 60_000)
    reference_forecast = route.public_greedy_forecast(
        real,
        validation[:, :route.CONTEXT_LENGTH],
        context_times,
        future_times,
    )
    route.seed_everything(config.seed + 60_000)
    exported_forecast = route.public_greedy_forecast(
        exported,
        validation[:, :route.CONTEXT_LENGTH],
        context_times,
        future_times,
    )
    checks["training_exported_inference_parity"] = parity_check(
        reference_forecast,
        exported_forecast,
        atol=0,
        rtol=0,
        name="exported Kronos Mini greedy forecast",
    )

    changed = validation.copy()
    changed[:, route.CONTEXT_LENGTH:] += np.asarray(
        [10, 10, 10, 10, 100], dtype=np.float32
    )
    checks["prefix_invariance"] = prefix_invariance_check(
        lambda values: values[:, :route.CONTEXT_LENGTH],
        validation,
        changed,
        prefix_length=route.CONTEXT_LENGTH,
        atol=0,
        rtol=0,
    )
    checks["future_corruption_invariance"] = future_corruption_check(
        lambda values: route.public_greedy_forecast(
            exported,
            values[:, :route.CONTEXT_LENGTH],
            context_times,
            future_times,
        ),
        validation,
        changed,
        visible_length=route.CONTEXT_LENGTH,
        atol=0,
        rtol=0,
    )
    for name in (
        "contract_roll",
        "session_gap",
        "split_boundary",
        "oos_boundary",
    ):
        checks[f"{name}_rejection"] = _boundary(name)

    perturbed = validation[:, :route.CONTEXT_LENGTH].copy()
    perturbed[:, 176:240, 4] *= 2.0
    route.seed_everything(config.seed + 61_000)
    perturbed_forecast = route.public_greedy_forecast(
        exported, perturbed, context_times, future_times
    )
    changed_fraction = float(
        np.mean(np.abs(perturbed_forecast - exported_forecast) > 1e-7)
    )
    checks["multivariate_channel_grouping"] = _result(
        changed_fraction > 0,
        {
            "layout": "joint_multivariate",
            "changed_output_fraction": changed_fraction,
            "perturbed_channel": "volume_local_regime",
        },
        "joint predictor forecast did not react to volume/amount regime change",
    )

    missing = validation.copy()
    missing[:, :16] = np.nan
    checks["native_missing_data_mask"] = rejection_check(
        lambda values: route.parent_array(values),
        {"missing_parent": missing},
    )

    torch = route._torch()
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(args.device)
        memory_probe = lambda: int(torch.cuda.max_memory_allocated(args.device))
    else:
        memory_probe = None
    performance = performance_check(
        lambda: route.public_greedy_forecast(
            exported,
            validation[:, :route.CONTEXT_LENGTH],
            context_times,
            future_times,
        ),
        batch_size=config.batch_size,
        repeats=2,
        warmups=1,
        min_examples_per_second=0.01,
        memory_probe=memory_probe,
    )
    checks["memory_measurement"] = CheckResult(
        performance.status, dict(performance.metrics), performance.reason
    )
    checks["throughput_measurement"] = CheckResult(
        performance.status, dict(performance.metrics), performance.reason
    )

    negative, negative_times = _negative(config.batch_size, config.seed + 7)
    negative_context_times, negative_future_times = _forecast_times(negative_times)
    checks["negative_price_behavior"] = negative_price_behavior_check(
        lambda values: route.public_greedy_forecast(
            exported,
            values[:, :route.CONTEXT_LENGTH],
            negative_context_times,
            negative_future_times,
        ),
        negative,
        behavior="support",
    )
    route.seed_everything(config.seed + 60_000)
    repeated_forecast = route.public_greedy_forecast(
        exported,
        validation[:, :route.CONTEXT_LENGTH],
        context_times,
        future_times,
    )
    checks["native_output_parity"] = parity_check(
        exported_forecast,
        repeated_forecast,
        atol=0,
        rtol=0,
        name="Kronos Mini greedy forecast repeatability",
    )

    checks["checkpoint_lineage"] = _result(
        (
            state["route_key"] == route.ROUTE_KEY
            and state["model_identity"] == base.identity
            and reopened_bundle["route_key"] == route.ROUTE_KEY
            and reopened_bundle["model_identity"] == base.identity
            and base.identity["tokenizer_parent"]["pilot_evidence_sha256"]
            == json.loads(
                Path(args.parent_pilot_evidence).read_text(encoding="utf-8")
            )["evidence_sha256"]
        ),
        {
            "training_state_sha256": _sha(state_path),
            "deployment_bundle_sha256": _sha(bundle_path),
            "parent_pilot_evidence_sha256": base.identity["tokenizer_parent"][
                "pilot_evidence_sha256"
            ],
            "parent_tokenizer_bundle_sha256": base.identity["tokenizer_parent"][
                "tokenizer_bundle_sha256"
            ],
        },
        "Kronos Mini predictor checkpoint/parent lineage is incomplete",
    )
    fixture_document = json.loads(fixture_manifest.read_text(encoding="utf-8"))
    checks["data_lineage"] = _result(
        (
            _sha(fixture) == fixture_document["artifact"]["sha256"]
            and fixture_document["market_data_read"] is False
            and fixture_document["oos_read"] is False
            and fixture_document["venue_timezone"] == route.VENUE_TIMEZONE
        ),
        {
            "fixture_sha256": _sha(fixture),
            "market_data_read": False,
            "oos_read": False,
            "venue_timezone": route.VENUE_TIMEZONE,
        },
        "Kronos Mini predictor fixture lineage is invalid",
    )

    raw_checks = {name: check.manifest() for name, check in checks.items()}
    metrics = {
        "real_validation_loss": real_validation,
        "shuffle_validation_loss": shuffled_validation,
        "time_destroyed_validation_loss": destroyed_validation,
        "all_checks_pass": all(check.status == "pass" for check in checks.values()),
    }
    raw_path = _json(
        output / "raw_checks.json",
        {
            "schema_version": RAW_SCHEMA,
            "route_key": route.ROUTE_KEY,
            "checks": raw_checks,
            "metrics": metrics,
        },
    )
    evidence = build_route_smoke_evidence(
        route_key=route.ROUTE_KEY,
        executor_path=route.__file__,
        executor_entrypoint="native_loss/public_greedy_forecast",
        checks=raw_checks,
        artifacts={
            "model_snapshot": Path(args.model_snapshot).resolve(),
            "tokenizer_snapshot": Path(args.tokenizer_snapshot).resolve(),
            "source_runtime": Path(args.source_runtime).resolve(),
            "parent_route_evidence": Path(args.parent_pilot_evidence).resolve(),
            "parent_route_bundle": Path(args.parent_tokenizer_bundle).resolve(),
            "synthetic_fixture": fixture,
            "synthetic_fixture_manifest": fixture_manifest,
            "interrupted_state": interrupted,
            "training_state": state_path,
            "deployment_bundle": bundle_path,
            "raw_checks": raw_path,
            "smoke_runner": Path(__file__).resolve(),
        },
        metrics=metrics,
    )
    validate_route_smoke_evidence(evidence)
    evidence_path = _json(output / "smoke_evidence.json", evidence)
    failed = [name for name, check in checks.items() if check.status != "pass"]
    return {
        "status": "pass" if evidence["smoke_admitted"] else "fail",
        "route_key": route.ROUTE_KEY,
        "smoke_admitted": evidence["smoke_admitted"],
        "pilot_admitted": False,
        "training_admitted": False,
        "evidence": {
            "path": str(evidence_path),
            "sha256": _sha(evidence_path),
            "content_sha256": evidence["evidence_sha256"],
        },
        "metrics": metrics,
        "failed_checks": failed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-snapshot", required=True)
    parser.add_argument("--tokenizer-snapshot", required=True)
    parser.add_argument("--source-runtime", required=True)
    parser.add_argument("--parent-pilot-evidence", required=True)
    parser.add_argument("--parent-tokenizer-bundle", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=4e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] == "fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
