#!/usr/bin/env python3
"""Run the exact Chronos-Bolt native-quantile route on deterministic synthetic data.

This command is smoke-only.  It reads no market data and cannot grant pilot, full
training, deployment, OOS, or trading admission.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
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
from futures_foundation.finetune.routes import chronos_bolt as route


FIXTURE_SCHEMA = "ffm_chronos_bolt_route_smoke_fixture_v1"


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
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)
    return target


def _fixture(batch: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    time = np.arange(route.PARENT_LENGTH, dtype=np.float64)
    level = rng.uniform(25.0, 125.0, size=batch)
    slope = rng.uniform(-0.9, 0.9, size=batch)
    amplitude = rng.uniform(0.05, 0.55, size=batch)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=batch)
    close = (
        level[:, None]
        + slope[:, None] * time[None, :] / 50.0
        + amplitude[:, None] * np.sin(time[None, :] / 13.0 + phase[:, None])
    )
    open_ = np.concatenate((close[:, :1], close[:, :-1]), axis=1)
    spread = 0.03 + 0.01 * np.abs(np.sin(time / 17.0))[None, :]
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = (
        1_000.0
        + rng.uniform(20.0, 80.0, size=batch)[:, None]
        + 5.0 * time[None, :]
        + 25.0 * np.sin(time[None, :] / 19.0 + phase[:, None])
    )
    parent = np.stack((open_, high, low, close, volume), axis=-1).astype(np.float32)
    return route.parent_array(parent)


def _negative_fixture(batch: int, seed: int) -> np.ndarray:
    values = _fixture(batch, seed).copy()
    shift = float(np.max(values[:, :, :4]) + 20.0)
    values[:, :, :4] -= shift
    return route.parent_array(values)


def _write_fixture(directory: Path, train: np.ndarray, validation: np.ndarray) -> tuple[Path, Path]:
    artifact = directory / "synthetic_fixture.npz"
    np.savez_compressed(artifact, train=train, validation=validation)
    manifest = {
        "schema_version": FIXTURE_SCHEMA,
        "generator": "bounded_trend_sinusoid_ohlcv_v1",
        "market_data_read": False,
        "oos_read": False,
        "train_shape": list(train.shape),
        "validation_shape": list(validation.shape),
        "artifact": {
            "path": str(artifact.resolve()),
            "sha256": _sha256(artifact),
            "bytes": artifact.stat().st_size,
        },
    }
    manifest_path = _atomic_json(directory / "synthetic_fixture.manifest.json", manifest)
    return artifact, manifest_path


def _fresh_model(base_model: Any, initial: Mapping[str, Any], device: str) -> Any:
    model = copy.deepcopy(base_model).to(device)
    model.load_state_dict(initial, strict=True)
    return model


def _cpu_numpy(value: Any) -> np.ndarray:
    return value.detach().float().cpu().numpy()


def _loss(model: Any, parent: np.ndarray, device: str) -> float:
    torch = route._torch()
    model.eval()
    with torch.no_grad():
        return float(route.native_loss(model, parent, device=device).detach().cpu())


def _train_fixture(
    base_model: Any,
    initial: Mapping[str, Any],
    parent: np.ndarray,
    validation: np.ndarray,
    config: route.RouteConfig,
    *,
    device: str,
) -> tuple[Any, list[float], float]:
    route.seed_everything(config.seed)
    model = _fresh_model(base_model, initial, device)
    optimizer = route.make_optimizer(model, config)
    scheduler = route.make_scheduler(optimizer, config)
    losses = []
    for _ in range(config.total_steps):
        row = route.optimizer_step(
            model, optimizer, scheduler, parent,
            device=device, max_gradient_norm=config.max_gradient_norm,
        )
        losses.append(float(row["loss"]))
    return model, losses, _loss(model, validation, device)


def _simple_result(passed: bool, metrics: Mapping[str, Any], reason: str) -> CheckResult:
    return CheckResult(
        status="pass" if passed else "fail",
        metrics=dict(metrics),
        reason=None if passed else reason,
    )


def _boundary_result(kind: str) -> CheckResult:
    expected = pd.Timedelta("1min")
    length = 640
    timestamps = pd.date_range("2024-01-01", periods=length, freq=expected, tz="UTC")

    def validate(case: Mapping[str, Any]) -> None:
        indices = np.asarray(case["indices"], np.int64)
        starts = ssl_data.window_starts(
            indices,
            route.PARENT_LENGTH,
            timestamps=case["timestamps"],
            expected_delta=expected,
            segment_ids=case.get("segments"),
        )
        if not len(starts):
            raise ValueError(f"invalid {kind} case was rejected as required")

    if kind == "contract_roll":
        case = {
            "indices": np.arange(length),
            "timestamps": timestamps,
            "segments": np.asarray(["A"] * 320 + ["B"] * 320),
        }
    elif kind == "session_gap":
        shifted = timestamps[:320].append(timestamps[320:] + pd.Timedelta("1h"))
        case = {
            "indices": np.arange(length),
            "timestamps": shifted,
            "segments": np.asarray(["A"] * length),
        }
    elif kind == "split_boundary":
        case = {
            "indices": np.r_[np.arange(0, 320), np.arange(400, 720)],
            "timestamps": pd.date_range("2024-01-01", periods=720, freq=expected, tz="UTC"),
            "segments": np.asarray(["A"] * 720),
        }
    elif kind == "oos_boundary":
        cutoff = pd.Timestamp("2024-01-01 05:20", tz="UTC")
        eligible = np.flatnonzero(timestamps < cutoff)
        case = {
            "indices": eligible,
            "timestamps": timestamps,
            "segments": np.asarray(["A"] * length),
        }
    else:  # pragma: no cover
        raise ValueError(kind)
    return rejection_check(validate, {kind: case})


def _resume_result(
    base_model: Any,
    initial: Mapping[str, Any],
    parent: np.ndarray,
    config: route.RouteConfig,
    model_identity: Mapping[str, Any],
    *,
    device: str,
    temporary_state: Path,
) -> CheckResult:
    steps = 4

    def run_uninterrupted() -> dict[str, Any]:
        route.seed_everything(config.seed + 31)
        model = _fresh_model(base_model, initial, device)
        optimizer = route.make_optimizer(model, config)
        scheduler = route.make_scheduler(optimizer, config)
        history = []
        for step in range(steps):
            row = route.optimizer_step(
                model, optimizer, scheduler, parent,
                device=device, max_gradient_norm=config.max_gradient_norm,
            )
            history.append({"step": step + 1, **row})
        return route.capture_training_state(
            model=model, optimizer=optimizer, scheduler=scheduler, config=config,
            model_identity=model_identity, global_step=steps, sampler_cursor=steps,
            history=history,
        )

    def run_resumed() -> dict[str, Any]:
        route.seed_everything(config.seed + 31)
        model = _fresh_model(base_model, initial, device)
        optimizer = route.make_optimizer(model, config)
        scheduler = route.make_scheduler(optimizer, config)
        history = []
        for step in range(2):
            row = route.optimizer_step(
                model, optimizer, scheduler, parent,
                device=device, max_gradient_norm=config.max_gradient_norm,
            )
            history.append({"step": step + 1, **row})
        partial = route.capture_training_state(
            model=model, optimizer=optimizer, scheduler=scheduler, config=config,
            model_identity=model_identity, global_step=2, sampler_cursor=2,
            history=history,
        )
        route.save_training_state(temporary_state, partial)
        reopened = route.load_training_state(temporary_state)
        resumed = _fresh_model(base_model, initial, device)
        resumed_optimizer = route.make_optimizer(resumed, config)
        resumed_scheduler = route.make_scheduler(resumed_optimizer, config)
        global_step, cursor, history = route.restore_training_state(
            reopened, model=resumed, optimizer=resumed_optimizer,
            scheduler=resumed_scheduler, config=config, model_identity=model_identity,
        )
        if (global_step, cursor) != (2, 2):
            raise RuntimeError("resume cursor changed")
        for step in range(2, steps):
            row = route.optimizer_step(
                resumed, resumed_optimizer, resumed_scheduler, parent,
                device=device, max_gradient_norm=config.max_gradient_norm,
            )
            history.append({"step": step + 1, **row})
        return route.capture_training_state(
            model=resumed, optimizer=resumed_optimizer, scheduler=resumed_scheduler,
            config=config, model_identity=model_identity, global_step=steps,
            sampler_cursor=steps, history=history,
        )

    return interruption_resume_parity_check(
        run_uninterrupted, run_resumed, atol=0.0, rtol=0.0,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    torch = route._torch()
    output = Path(args.output).expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"smoke output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    config = route.RouteConfig(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        total_steps=args.steps,
        seed=args.seed,
    )
    config.validate()
    if config.total_steps != 20:
        raise ValueError("canonical Chronos-Bolt smoke requires exactly 20 steps")

    train = _fixture(config.batch_size, config.seed)
    validation = _fixture(config.batch_size, config.seed + 1)
    fixture_path, fixture_manifest_path = _write_fixture(output, train, validation)
    pipeline, base_model, model_identity = route.load_pipeline(
        args.model_snapshot, device=args.device,
    )
    base_model.eval()
    initial = route.model_state_cpu(base_model)

    checks: dict[str, CheckResult] = {}

    # One real update on a fresh copy.
    one_model = _fresh_model(base_model, initial, args.device)
    one_optimizer = route.make_optimizer(one_model, config)
    one_scheduler = route.make_scheduler(one_optimizer, config)
    checks["one_batch_forward_backward"] = forward_backward_check(
        lambda: route.optimizer_step(
            one_model, one_optimizer, one_scheduler, train,
            device=args.device, max_gradient_norm=config.max_gradient_norm,
        )
    )
    del one_model, one_optimizer, one_scheduler
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Controlled learnable fixture and retained final route state.
    route.seed_everything(config.seed)
    real_model = _fresh_model(base_model, initial, args.device)
    real_optimizer = route.make_optimizer(real_model, config)
    real_scheduler = route.make_scheduler(real_optimizer, config)
    checks["controlled_learnable_loss_decrease"] = loss_decrease_check(
        lambda: _loss(real_model, train, args.device),
        lambda: route.optimizer_step(
            real_model, real_optimizer, real_scheduler, train,
            device=args.device, max_gradient_norm=config.max_gradient_norm,
        ),
        steps=config.total_steps,
        min_relative_decrease=0.20,
        tail=3,
    )
    real_validation = _loss(real_model, validation, args.device)

    rng = np.random.default_rng(config.seed + 91)
    shuffled = train.copy()
    shuffled[:, route.CONTEXT_LENGTH:] = shuffled[
        rng.permutation(len(shuffled)), route.CONTEXT_LENGTH:
    ]
    destroyed = train.copy()
    destroyed[:, :route.CONTEXT_LENGTH] = destroyed[
        :, rng.permutation(route.CONTEXT_LENGTH)
    ]
    _, shuffle_losses, shuffle_validation = _train_fixture(
        base_model, initial, shuffled, validation, config,
        device=args.device,
    )
    _, destroyed_losses, destroyed_validation = _train_fixture(
        base_model, initial, destroyed, validation, config,
        device=args.device,
    )
    controls = control_rejection_check(
        real_validation, [shuffle_validation], [destroyed_validation],
        margin=0.05, higher_is_better=False,
    )
    checks["shuffle_control_rejection"] = _simple_result(
        controls.status == "pass" and real_validation + 0.05 <= shuffle_validation,
        {
            **controls.metrics,
            "shuffle_final_train_loss": float(shuffle_losses[-1]),
        },
        "real chronology did not beat the shuffled-target control",
    )
    checks["time_destroyed_control_rejection"] = _simple_result(
        controls.status == "pass" and real_validation + 0.05 <= destroyed_validation,
        {
            **controls.metrics,
            "time_destroyed_final_train_loss": float(destroyed_losses[-1]),
        },
        "real chronology did not beat the time-destroyed control",
    )

    # Exact serialization/resume trajectory.
    resume_transport = output / "interrupted.train.pt"
    checks["exact_interruption_resume_trajectory"] = _resume_result(
        base_model, initial, train, config, model_identity,
        device=args.device, temporary_state=resume_transport,
    )

    # Full checkpoint and deployment bundle from the controlled real route.
    history = [
        {"step": index + 1, "train_loss": float(value)}
        for index, value in enumerate(
            checks["controlled_learnable_loss_decrease"].metrics["losses"][1:]
        )
    ]
    state = route.capture_training_state(
        model=real_model, optimizer=real_optimizer, scheduler=real_scheduler,
        config=config, model_identity=model_identity,
        global_step=config.total_steps, sampler_cursor=config.total_steps,
        history=history,
    )
    checkpoint_path = output / "chronos_bolt_smoke.train.pt"
    checkpoint_identity = route.save_training_state(checkpoint_path, state)
    export_bundle = route.build_export_bundle(
        model=real_model, model_identity=model_identity,
    )
    export_path = output / "chronos_bolt_smoke.forecast.pt"
    export_identity = route.save_export_bundle(export_path, export_bundle)
    exported_pipeline, exported_model, reopened_bundle = route.load_export_bundle(
        export_path, snapshot=args.model_snapshot, device=args.device,
    )
    real_model.eval()
    exported_model.eval()
    reference_quantiles = _cpu_numpy(route.direct_quantiles(
        real_model, validation, device=args.device,
    ))
    exported_quantiles = _cpu_numpy(route.direct_quantiles(
        exported_model, validation, device=args.device,
    ))
    checks["training_exported_inference_parity"] = parity_check(
        reference_quantiles, exported_quantiles,
        atol=0.0, rtol=0.0, name="exported Chronos-Bolt quantiles",
    )

    # Context-only/no-future contracts.
    changed_future = validation.copy()
    changed_future[:, route.CONTEXT_LENGTH:, :4] += 50.0
    changed_future[:, route.CONTEXT_LENGTH:, 4] *= 2.0
    checks["prefix_invariance"] = prefix_invariance_check(
        lambda parent: route.parent_array(parent)[:, :route.CONTEXT_LENGTH],
        validation, changed_future,
        prefix_length=route.CONTEXT_LENGTH,
        atol=0.0, rtol=0.0,
    )
    checks["future_corruption_invariance"] = future_corruption_check(
        lambda parent: _cpu_numpy(route.direct_quantiles(
            exported_model, parent, device=args.device,
        )),
        validation, changed_future,
        visible_length=route.CONTEXT_LENGTH,
        atol=0.0, rtol=0.0,
    )

    checks["contract_roll_rejection"] = _boundary_result("contract_roll")
    checks["session_gap_rejection"] = _boundary_result("session_gap")
    checks["split_boundary_rejection"] = _boundary_result("split_boundary")
    checks["oos_boundary_rejection"] = _boundary_result("oos_boundary")

    # Five independent univariate channel routes.
    baseline = _cpu_numpy(route.direct_quantiles(
        exported_model, validation, device=args.device,
    ))
    perturbed = validation.copy()
    perturbed[:, :route.CONTEXT_LENGTH, 4] *= 1.10
    changed = _cpu_numpy(route.direct_quantiles(
        exported_model, perturbed, device=args.device,
    ))
    unaffected = [0, 1, 2, 3]
    unaffected_error = float(np.max(np.abs(baseline[:, unaffected] - changed[:, unaffected])))
    affected_change = float(np.max(np.abs(baseline[:, 4] - changed[:, 4])))
    checks["multivariate_channel_grouping"] = _simple_result(
        baseline.shape == (
            config.batch_size, len(route.CHANNELS), route.HORIZON_LENGTH,
            len(route.QUANTILES),
        )
        and unaffected_error == 0.0
        and affected_change > 0.0,
        {
            "layout": "independent_univariate_passes",
            "shape": list(baseline.shape),
            "unaffected_channel_max_abs": unaffected_error,
            "perturbed_channel": "volume",
            "affected_channel_max_change": affected_change,
        },
        "Chronos-Bolt channel-independent grouping was not preserved",
    )

    missing = validation.copy()
    missing[:, :16, :] = np.nan
    missing_public = _cpu_numpy(route.public_quantiles(
        exported_pipeline, missing, device=args.device,
    ))
    missing_direct = _cpu_numpy(route.direct_quantiles(
        exported_model, missing, device=args.device,
    ))
    missing_error = float(np.max(np.abs(missing_public - missing_direct)))
    checks["native_missing_data_mask"] = _simple_result(
        np.isfinite(missing_public).all()
        and np.isfinite(missing_direct).all()
        and np.allclose(missing_public, missing_direct, atol=1e-5, rtol=1e-5),
        {
            "missing_prefix_bars": 16,
            "public_direct_max_abs": missing_error,
            "finite": bool(np.isfinite(missing_direct).all()),
        },
        "Chronos-Bolt native missing mask produced non-finite or non-parity output",
    )

    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(args.device)
        memory_probe = lambda: int(torch.cuda.max_memory_allocated(args.device))
    else:
        memory_probe = None
    performance = performance_check(
        lambda: route.direct_quantiles(exported_model, validation, device=args.device),
        batch_size=config.batch_size,
        repeats=5,
        warmups=2,
        min_examples_per_second=0.1,
        memory_probe=memory_probe,
    )
    checks["memory_measurement"] = CheckResult(
        status=performance.status,
        metrics=dict(performance.metrics),
        reason=performance.reason,
    )
    checks["throughput_measurement"] = CheckResult(
        status=performance.status,
        metrics=dict(performance.metrics),
        reason=performance.reason,
    )

    checks["negative_price_behavior"] = negative_price_behavior_check(
        lambda parent: _cpu_numpy(route.direct_quantiles(
            exported_model, parent, device=args.device,
        )),
        _negative_fixture(config.batch_size, config.seed + 7),
        behavior="support",
    )

    public = _cpu_numpy(route.public_quantiles(
        exported_pipeline, validation, device=args.device,
    ))
    direct = _cpu_numpy(route.direct_quantiles(
        exported_model, validation, device=args.device,
    ))
    checks["native_output_parity"] = parity_check(
        public, direct, atol=1e-5, rtol=1e-5,
        name="Chronos-Bolt public/direct native quantiles",
    )

    reopened_state = route.load_training_state(checkpoint_path)
    checks["checkpoint_lineage"] = _simple_result(
        reopened_state.get("schema_version") == route.CHECKPOINT_SCHEMA
        and reopened_state.get("route_key") == route.ROUTE_KEY
        and reopened_state.get("model_identity") == model_identity
        and checkpoint_identity["sha256"] == _sha256(checkpoint_path)
        and reopened_bundle["route_key"] == route.ROUTE_KEY
        and export_identity["sha256"] == _sha256(export_path),
        {
            "checkpoint": checkpoint_identity,
            "export": export_identity,
            "required_state_fields": sorted(reopened_state),
        },
        "Chronos-Bolt checkpoint/export lineage is incomplete or stale",
    )
    fixture_document = json.loads(fixture_manifest_path.read_text(encoding="utf-8"))
    checks["data_lineage"] = _simple_result(
        fixture_document["artifact"]["sha256"] == _sha256(fixture_path)
        and fixture_document["market_data_read"] is False
        and fixture_document["oos_read"] is False,
        {
            "fixture_path": str(fixture_path),
            "fixture_sha256": _sha256(fixture_path),
            "market_data_read": False,
            "oos_read": False,
        },
        "Chronos-Bolt synthetic fixture lineage is invalid",
    )

    raw_checks = {name: result.manifest() for name, result in checks.items()}
    raw_report = {
        "schema_version": "ffm_chronos_bolt_route_smoke_raw_v1",
        "route_key": route.ROUTE_KEY,
        "config": {
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "batch_size": config.batch_size,
            "steps": config.total_steps,
            "seed": config.seed,
            "device": args.device,
        },
        "checks": raw_checks,
        "metrics": {
            "real_validation_loss": real_validation,
            "shuffle_validation_loss": shuffle_validation,
            "time_destroyed_validation_loss": destroyed_validation,
            "all_checks_pass": all(item.status == "pass" for item in checks.values()),
        },
    }
    raw_report_path = _atomic_json(output / "raw_checks.json", raw_report)

    route_source = Path(route.__file__).resolve()
    evidence = build_route_smoke_evidence(
        route_key=route.ROUTE_KEY,
        executor_path=route_source,
        executor_entrypoint="native_loss/direct_quantiles",
        checks=raw_checks,
        artifacts={
            "model_snapshot": Path(args.model_snapshot).resolve(),
            "source_runtime": Path(
                importlib.metadata.distribution("chronos-forecasting")._path
            ).resolve(),
            "synthetic_fixture": fixture_path,
            "synthetic_fixture_manifest": fixture_manifest_path,
            "interrupted_state": resume_transport,
            "training_state": checkpoint_path,
            "deployment_bundle": export_path,
            "raw_checks": raw_report_path,
            "smoke_runner": Path(__file__).resolve(),
        },
        metrics=raw_report["metrics"],
    )
    validate_route_smoke_evidence(evidence)
    evidence_path = _atomic_json(output / "smoke_evidence.json", evidence)
    return {
        "status": "pass" if evidence["smoke_admitted"] else "fail",
        "route_key": route.ROUTE_KEY,
        "smoke_admitted": evidence["smoke_admitted"],
        "pilot_admitted": False,
        "training_admitted": False,
        "evidence": {
            "path": str(evidence_path),
            "sha256": _sha256(evidence_path),
            "content_sha256": evidence["evidence_sha256"],
        },
        "metrics": raw_report["metrics"],
        "failed_checks": [
            name for name, result in checks.items() if result.status != "pass"
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-snapshot", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, sort_keys=True, indent=2))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
