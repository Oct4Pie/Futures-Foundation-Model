#!/usr/bin/env python3
"""Run exact Kronos Mini/Small tokenizer reconstruction smoke in bounded phases.

The route is expensive enough that one complete run can exceed an orchestration command
window.  Each phase is deterministic, hash-bound to the same route/config/runtime identity,
and writes a compact receipt.  Final evidence is emitted only after every phase and every
mandatory smoke check has been reloaded and verified.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import copy
import hashlib
import json
import os
from pathlib import Path
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
from futures_foundation.finetune.native_contracts import file_sha256
from futures_foundation.finetune.native_route_smoke import (
    build_route_smoke_evidence,
    validate_route_smoke_evidence,
)
from futures_foundation.finetune.native_training_schema_v2 import (
    canonical_route_profile_sha256,
)
from futures_foundation.finetune.routes import kronos_tokenizer as route


PHASE_SCHEMA = "ffm_kronos_tokenizer_smoke_phase_v3"
RAW_SCHEMA = "ffm_kronos_tokenizer_smoke_raw_v1"
CANONICAL_SMOKE_STEPS = 20
CONTROL_MARGIN = 0.0


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


def _read_json(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON document must be an object: {source}")
    return value


def _fixture(batch: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    time = np.arange(route.PARENT_LENGTH, dtype=np.float64)
    level = rng.uniform(35, 90, batch)
    slope = rng.uniform(-0.05, 0.05, batch)
    phase = rng.uniform(0, 2 * np.pi, batch)
    close = (
        level[:, None]
        + slope[:, None] * time
        + 0.35 * np.sin(time[None, :] / 13 + phase[:, None])
    )
    open_ = np.concatenate((close[:, :1], close[:, :-1]), axis=1)
    spread = 0.15 + 0.03 * np.abs(np.sin(time / 19))[None, :]
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = (
        800
        + rng.uniform(20, 70, batch)[:, None]
        + 0.5 * time
        + 25 * np.sin(time[None, :] / 23 + phase[:, None])
    )
    return route.parent_array(
        np.stack((open_, high, low, close, volume), axis=-1).astype(np.float32)
    )


def _negative(batch: int, seed: int) -> np.ndarray:
    value = _fixture(batch, seed).copy()
    value[:, :, :4] -= float(np.max(value[:, :, :4]) + 10)
    return route.parent_array(value)


def _fresh(base: Any, initial: Mapping[str, Any], device: str) -> Any:
    model = copy.deepcopy(base).to(device)
    model.load_state_dict(initial, strict=True)
    return model


def _np_codes(value: tuple[Any, Any]) -> np.ndarray:
    return np.concatenate(
        [part.detach().cpu().numpy().reshape(len(part), -1) for part in value],
        axis=1,
    )


def _loss(model: Any, parent: np.ndarray, device: str) -> float:
    torch = route._torch()
    model.eval()
    with torch.no_grad():
        return float(route.native_loss(model, parent, device=device).detach().cpu())


def _mismatch(
    model: Any,
    parent: np.ndarray,
    target: np.ndarray,
    device: str,
) -> Any:
    torch = route._torch()
    target_values = route.model_input(target, device=device)
    coarse, full, quantizer = route.native_components(model, parent, device=device)
    return (
        torch.nn.functional.mse_loss(coarse, target_values)
        + torch.nn.functional.mse_loss(full, target_values)
        + quantizer
    ) / 2.0


def _train_control(
    base: Any,
    initial: Mapping[str, Any],
    parent: np.ndarray,
    validation: np.ndarray,
    config: route.RouteConfig,
    device: str,
    *,
    target: np.ndarray | None = None,
) -> tuple[list[float], float]:
    torch = route._torch()
    route.seed_everything(config.seed)
    model = _fresh(base, initial, device)
    optimizer = route.make_optimizer(model, config)
    scheduler = route.make_scheduler(optimizer, config)
    losses: list[float] = []
    for _ in range(config.total_steps):
        if target is None:
            row = route.optimizer_step(
                model,
                optimizer,
                scheduler,
                parent,
                device=device,
                max_gradient_norm=config.max_gradient_norm,
            )
            losses.append(row["loss"])
        else:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = _mismatch(model, parent, target, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.max_gradient_norm
            )
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach().cpu()))
    return losses, _loss(model, validation, device)


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


def _compact_resume_record(
    state: Mapping[str, Any],
    model: Any,
    parent: np.ndarray,
    device: str,
) -> dict[str, Any]:
    torch = route._torch()
    devices = (
        [torch.device(device).index or 0]
        if str(device).startswith("cuda")
        else []
    )
    with torch.no_grad(), torch.random.fork_rng(devices=devices, enabled=True):
        next_loss = float(
            route.native_loss(model, parent, device=device).detach().cpu()
        )
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
        "next_loss": next_loss,
    }


def _resume_check(
    arm: str,
    base: Any,
    initial: Mapping[str, Any],
    parent: np.ndarray,
    config: route.RouteConfig,
    identity: Mapping[str, Any],
    device: str,
    path: Path,
) -> CheckResult:
    steps = 4

    def full() -> dict[str, Any]:
        route.seed_everything(config.seed + 31)
        model = _fresh(base, initial, device)
        optimizer = route.make_optimizer(model, config)
        scheduler = route.make_scheduler(optimizer, config)
        history = []
        for index in range(steps):
            history.append(
                {
                    "step": index + 1,
                    **route.optimizer_step(
                        model,
                        optimizer,
                        scheduler,
                        parent,
                        device=device,
                        max_gradient_norm=config.max_gradient_norm,
                    ),
                }
            )
        state = route.capture_training_state(
            arm_key=arm,
            tokenizer=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            model_identity=identity,
            global_step=steps,
            sampler_cursor=steps,
            history=history,
        )
        return _compact_resume_record(state, model, parent, device)

    def resumed() -> dict[str, Any]:
        route.seed_everything(config.seed + 31)
        model = _fresh(base, initial, device)
        optimizer = route.make_optimizer(model, config)
        scheduler = route.make_scheduler(optimizer, config)
        history = []
        for index in range(2):
            history.append(
                {
                    "step": index + 1,
                    **route.optimizer_step(
                        model,
                        optimizer,
                        scheduler,
                        parent,
                        device=device,
                        max_gradient_norm=config.max_gradient_norm,
                    ),
                }
            )
        partial = route.capture_training_state(
            arm_key=arm,
            tokenizer=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            model_identity=identity,
            global_step=2,
            sampler_cursor=2,
            history=history,
        )
        route.save_training_state(path, partial)
        saved = route.load_training_state(path)
        resumed_model = _fresh(base, initial, device)
        resumed_optimizer = route.make_optimizer(resumed_model, config)
        resumed_scheduler = route.make_scheduler(resumed_optimizer, config)
        step, cursor, history = route.restore_training_state(
            saved,
            arm_key=arm,
            tokenizer=resumed_model,
            optimizer=resumed_optimizer,
            scheduler=resumed_scheduler,
            config=config,
            model_identity=identity,
        )
        if (step, cursor) != (2, 2):
            raise RuntimeError("resume cursor drift")
        for index in range(2, steps):
            history.append(
                {
                    "step": index + 1,
                    **route.optimizer_step(
                        resumed_model,
                        resumed_optimizer,
                        resumed_scheduler,
                        parent,
                        device=device,
                        max_gradient_norm=config.max_gradient_norm,
                    ),
                }
            )
        final_state = route.capture_training_state(
            arm_key=arm,
            tokenizer=resumed_model,
            optimizer=resumed_optimizer,
            scheduler=resumed_scheduler,
            config=config,
            model_identity=identity,
            global_step=steps,
            sampler_cursor=steps,
            history=history,
        )
        return _compact_resume_record(
            final_state, resumed_model, parent, device
        )

    return interruption_resume_parity_check(full, resumed, atol=0, rtol=0)


def _check_from_manifest(value: Mapping[str, Any]) -> CheckResult:
    if set(value) != {"status", "metrics", "reason"}:
        raise ValueError("phase check manifest is malformed")
    return CheckResult(
        status=str(value["status"]),
        metrics=dict(value["metrics"]),
        reason=value["reason"],
    )


def _config(args: argparse.Namespace) -> route.RouteConfig:
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
    if config.total_steps != CANONICAL_SMOKE_STEPS:
        raise ValueError(
            f"canonical Kronos tokenizer smoke requires {CANONICAL_SMOKE_STEPS} steps"
        )
    return config


def _phase_signature(
    args: argparse.Namespace,
    config: route.RouteConfig,
) -> str:
    document = {
        "arm": args.arm,
        "route_key": route.route_key(args.arm),
        "route_profile_sha256": canonical_route_profile_sha256(
            args.arm, route.TRACK, route.ROUTE_ID
        ),
        "config": asdict(config),
        "model_snapshot": str(Path(args.model_snapshot).resolve()),
        "tokenizer_snapshot": str(Path(args.tokenizer_snapshot).resolve()),
        "source_runtime": str(Path(args.source_runtime).resolve()),
        "executor_sha256": file_sha256(Path(route.__file__).resolve()),
        "phase_contract": PHASE_SCHEMA,
    }
    return hashlib.sha256(
        json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _phase_path(output: Path, phase: str) -> Path:
    return output / f"phase_{phase}.json"


def _write_phase(
    output: Path,
    phase: str,
    signature: str,
    payload: Mapping[str, Any],
) -> Path:
    document = {
        "schema_version": PHASE_SCHEMA,
        "phase": phase,
        "phase_signature": signature,
        "payload": dict(payload),
    }
    document["phase_sha256"] = hashlib.sha256(
        json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return _json(_phase_path(output, phase), document)


def _load_phase(output: Path, phase: str, signature: str) -> dict[str, Any]:
    path = _phase_path(output, phase)
    document = _read_json(path)
    supplied = document.pop("phase_sha256", None)
    actual = hashlib.sha256(
        json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if supplied != actual:
        raise ValueError(f"Kronos tokenizer phase integrity mismatch: {phase}")
    if (
        document.get("schema_version") != PHASE_SCHEMA
        or document.get("phase") != phase
        or document.get("phase_signature") != signature
        or not isinstance(document.get("payload"), dict)
    ):
        raise ValueError(f"Kronos tokenizer phase is stale or malformed: {phase}")
    return dict(document["payload"])


def _ensure_fixture(
    output: Path,
    config: route.RouteConfig,
    *,
    overwrite: bool,
) -> tuple[np.ndarray, np.ndarray, Path, Path]:
    fixture = output / "synthetic_fixture.npz"
    manifest = output / "synthetic_fixture.manifest.json"
    if overwrite or not fixture.is_file() or not manifest.is_file():
        train = _fixture(config.batch_size, config.seed)
        validation = _fixture(config.batch_size, config.seed + 1)
        temporary = Path(str(fixture) + f".{os.getpid()}.tmp.npz")
        np.savez_compressed(temporary, train=train, validation=validation)
        os.replace(temporary, fixture)
        _json(
            manifest,
            {
                "schema_version": "ffm_kronos_tokenizer_smoke_fixture_v1",
                "market_data_read": False,
                "oos_read": False,
                "artifact": {
                    "path": str(fixture.resolve()),
                    "sha256": _sha(fixture),
                    "bytes": fixture.stat().st_size,
                },
            },
        )
    fixture_manifest = _read_json(manifest)
    if (
        fixture_manifest.get("schema_version")
        != "ffm_kronos_tokenizer_smoke_fixture_v1"
        or fixture_manifest.get("market_data_read") is not False
        or fixture_manifest.get("oos_read") is not False
        or fixture_manifest.get("artifact", {}).get("sha256") != _sha(fixture)
    ):
        raise ValueError("Kronos tokenizer synthetic fixture identity mismatch")
    with np.load(fixture, allow_pickle=False) as saved:
        train = route.parent_array(saved["train"])
        validation = route.parent_array(saved["validation"])
    if train.shape[0] != config.batch_size or validation.shape != train.shape:
        raise ValueError("Kronos tokenizer fixture/config batch mismatch")
    return train, validation, fixture, manifest


def _load_base(
    args: argparse.Namespace,
) -> tuple[Any, dict[str, Any]]:
    return route.load_tokenizer(
        args.arm,
        model_snapshot=args.model_snapshot,
        tokenizer_snapshot=args.tokenizer_snapshot,
        source_runtime=args.source_runtime,
        device=args.device,
    )


def run_real_phase(
    args: argparse.Namespace,
    output: Path,
    config: route.RouteConfig,
    signature: str,
) -> dict[str, Any]:
    train, validation, fixture, fixture_manifest = _ensure_fixture(
        output, config, overwrite=args.overwrite
    )
    base, identity = _load_base(args)
    initial = route.model_state_cpu(base)

    one = _fresh(base, initial, args.device)
    one_optimizer = route.make_optimizer(one, config)
    one_scheduler = route.make_scheduler(one_optimizer, config)
    one_batch = forward_backward_check(
        lambda: route.optimizer_step(
            one,
            one_optimizer,
            one_scheduler,
            train,
            device=args.device,
            max_gradient_norm=config.max_gradient_norm,
        )
    )
    del one, one_optimizer, one_scheduler

    route.seed_everything(config.seed)
    real = _fresh(base, initial, args.device)
    optimizer = route.make_optimizer(real, config)
    scheduler = route.make_scheduler(optimizer, config)
    decrease = loss_decrease_check(
        lambda: _loss(real, train, args.device),
        lambda: route.optimizer_step(
            real,
            optimizer,
            scheduler,
            train,
            device=args.device,
            max_gradient_norm=config.max_gradient_norm,
        ),
        steps=config.total_steps,
        min_relative_decrease=0.05,
        tail=min(2, config.total_steps),
    )
    real_validation_loss = _loss(real, validation, args.device)
    losses = list(decrease.metrics["losses"])[1:]
    history = [
        {"step": index + 1, "train_loss": float(value)}
        for index, value in enumerate(losses)
    ]
    state = route.capture_training_state(
        arm_key=args.arm,
        tokenizer=real,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        model_identity=identity,
        global_step=config.total_steps,
        sampler_cursor=config.total_steps,
        history=history,
    )
    state_path = output / "tokenizer.train.pt"
    state_identity = route.save_training_state(state_path, state)
    bundle = route.build_export_bundle(
        arm_key=args.arm, tokenizer=real, model_identity=identity
    )
    bundle_path = output / "tokenizer.bundle.pt"
    bundle_identity = route.save_export_bundle(bundle_path, bundle)

    phase_path = _write_phase(
        output,
        "real",
        signature,
        {
            "checks": {
                "one_batch_forward_backward": one_batch.manifest(),
                "controlled_learnable_loss_decrease": decrease.manifest(),
            },
            "metrics": {
                "real_validation_loss": real_validation_loss,
                "final_train_loss": losses[-1],
            },
            "identity": identity,
            "artifacts": {
                "fixture": {
                    "path": str(fixture.resolve()),
                    "sha256": _sha(fixture),
                },
                "fixture_manifest": {
                    "path": str(fixture_manifest.resolve()),
                    "sha256": _sha(fixture_manifest),
                },
                "training_state": state_identity,
                "deployment_bundle": bundle_identity,
            },
        },
    )
    return {
        "status": "complete",
        "phase": "real",
        "phase_path": str(phase_path),
        "real_validation_loss": real_validation_loss,
        "checks_pass": one_batch.status == "pass" and decrease.status == "pass",
    }


def run_controls_phase(
    args: argparse.Namespace,
    output: Path,
    config: route.RouteConfig,
    signature: str,
) -> dict[str, Any]:
    real_phase = _load_phase(output, "real", signature)
    train, validation, _, _ = _ensure_fixture(
        output, config, overwrite=False
    )
    base, _ = _load_base(args)
    initial = route.model_state_cpu(base)
    rng = np.random.default_rng(config.seed + 91)
    target = np.stack(
        [train[row, rng.permutation(route.PARENT_LENGTH)] for row in range(len(train))],
        axis=0,
    ).copy()
    shuffled_losses, shuffled_validation = _train_control(
        base,
        initial,
        train,
        validation,
        config,
        args.device,
        target=target,
    )
    destroyed = np.stack(
        [train[row, rng.permutation(route.PARENT_LENGTH)] for row in range(len(train))],
        axis=0,
    ).copy()
    destroyed_losses, destroyed_validation = _train_control(
        base,
        initial,
        destroyed,
        validation,
        config,
        args.device,
    )
    real_validation = float(real_phase["metrics"]["real_validation_loss"])
    controls = control_rejection_check(
        real_validation,
        [shuffled_validation],
        [destroyed_validation],
        margin=CONTROL_MARGIN,
        higher_is_better=False,
    )
    shuffle_check = _result(
        real_validation + CONTROL_MARGIN <= shuffled_validation,
        {
            **controls.metrics,
            "shuffle_final_train_loss": shuffled_losses[-1],
        },
        "real tokenizer did not beat mismatched reconstruction targets",
    )
    destroyed_check = _result(
        real_validation + CONTROL_MARGIN <= destroyed_validation,
        {
            **controls.metrics,
            "time_destroyed_final_train_loss": destroyed_losses[-1],
        },
        "real tokenizer did not beat time-destroyed input",
    )
    phase_path = _write_phase(
        output,
        "controls",
        signature,
        {
            "checks": {
                "shuffle_control_rejection": shuffle_check.manifest(),
                "time_destroyed_control_rejection": destroyed_check.manifest(),
            },
            "metrics": {
                "shuffle_validation_loss": shuffled_validation,
                "time_destroyed_validation_loss": destroyed_validation,
            },
        },
    )
    return {
        "status": "complete",
        "phase": "controls",
        "phase_path": str(phase_path),
        "shuffle_validation_loss": shuffled_validation,
        "time_destroyed_validation_loss": destroyed_validation,
        "checks_pass": (
            shuffle_check.status == "pass" and destroyed_check.status == "pass"
        ),
    }


def run_resume_phase(
    args: argparse.Namespace,
    output: Path,
    config: route.RouteConfig,
    signature: str,
) -> dict[str, Any]:
    train, _, _, _ = _ensure_fixture(output, config, overwrite=False)
    base, identity = _load_base(args)
    initial = route.model_state_cpu(base)
    interrupted = output / "interrupted.train.pt"
    check = _resume_check(
        args.arm,
        base,
        initial,
        train,
        config,
        identity,
        args.device,
        interrupted,
    )
    phase_path = _write_phase(
        output,
        "resume",
        signature,
        {
            "checks": {
                "exact_interruption_resume_trajectory": check.manifest(),
            },
            "artifact": {
                "path": str(interrupted.resolve()),
                "sha256": _sha(interrupted),
                "bytes": interrupted.stat().st_size,
            },
        },
    )
    return {
        "status": "complete",
        "phase": "resume",
        "phase_path": str(phase_path),
        "checks_pass": check.status == "pass",
    }


def run_finalize_phase(
    args: argparse.Namespace,
    output: Path,
    config: route.RouteConfig,
    signature: str,
) -> dict[str, Any]:
    real_phase = _load_phase(output, "real", signature)
    controls_phase = _load_phase(output, "controls", signature)
    resume_phase = _load_phase(output, "resume", signature)
    train, validation, fixture, fixture_manifest = _ensure_fixture(
        output, config, overwrite=False
    )
    state_path = output / "tokenizer.train.pt"
    bundle_path = output / "tokenizer.bundle.pt"
    interrupted = output / "interrupted.train.pt"
    for path in (state_path, bundle_path, interrupted):
        if not path.is_file():
            raise FileNotFoundError(f"required Kronos smoke artifact is missing: {path}")

    base, identity = _load_base(args)
    state = route.load_training_state(state_path)
    state_model = copy.deepcopy(base).to(args.device)
    state_model.load_state_dict(state["model"], strict=True)
    state_model.eval()
    exported, reopened_bundle = route.load_export_bundle(
        bundle_path,
        arm_key=args.arm,
        model_snapshot=args.model_snapshot,
        tokenizer_snapshot=args.tokenizer_snapshot,
        source_runtime=args.source_runtime,
        device=args.device,
    )

    checks: dict[str, CheckResult] = {}
    for phase in (real_phase, controls_phase, resume_phase):
        for name, manifest in phase["checks"].items():
            if name in checks:
                raise ValueError(f"duplicate Kronos smoke check across phases: {name}")
            checks[name] = _check_from_manifest(manifest)

    reference_codes = _np_codes(
        route.tokenizer_codes(state_model, validation, device=args.device)
    )
    exported_codes = _np_codes(
        route.tokenizer_codes(exported, validation, device=args.device)
    )
    checks["training_exported_inference_parity"] = parity_check(
        reference_codes,
        exported_codes,
        atol=0,
        rtol=0,
        name="exported Kronos tokenizer codes",
    )

    extra = np.concatenate(
        (validation, np.zeros((len(validation), 16, 5), np.float32)), axis=1
    )
    changed = extra.copy()
    changed[:, 528:] += 100
    checks["prefix_invariance"] = prefix_invariance_check(
        lambda values: route.parent_array(values[:, :528]),
        extra,
        changed,
        prefix_length=528,
        atol=0,
        rtol=0,
    )
    checks["future_corruption_invariance"] = future_corruption_check(
        lambda values: _np_codes(
            route.tokenizer_codes(exported, values[:, :528], device=args.device)
        ),
        extra,
        changed,
        visible_length=528,
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

    base_codes = _np_codes(
        route.tokenizer_codes(exported, validation, device=args.device)
    )
    perturbed = validation.copy()
    # A whole-channel scale factor is intentionally removed by Kronos's per-channel
    # normalization.  Change the temporal shape of volume instead so the joint
    # volume/amount code path is exercised without violating OHLCV geometry.
    perturb_start, perturb_stop = 176, 240
    perturbed[:, perturb_start:perturb_stop, 4] *= 2.0
    changed_codes = _np_codes(
        route.tokenizer_codes(exported, perturbed, device=args.device)
    )
    code_change = float(np.mean(base_codes != changed_codes))
    checks["multivariate_channel_grouping"] = _result(
        code_change > 0,
        {
            "layout": "joint_multivariate",
            "changed_code_fraction": code_change,
            "amount_source": "volume_times_mean_ohlc_v1",
            "perturbation": {
                "channel": "volume",
                "start": perturb_start,
                "stop": perturb_stop,
                "multiplier": 2.0,
            },
        },
        "joint tokenizer codes did not react to volume/amount change",
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
        lambda: route.tokenizer_codes(exported, validation, device=args.device),
        batch_size=config.batch_size,
        repeats=3,
        warmups=1,
        min_examples_per_second=0.05,
        memory_probe=memory_probe,
    )
    checks["memory_measurement"] = CheckResult(
        performance.status, dict(performance.metrics), performance.reason
    )
    checks["throughput_measurement"] = CheckResult(
        performance.status, dict(performance.metrics), performance.reason
    )
    checks["negative_price_behavior"] = negative_price_behavior_check(
        lambda values: _np_codes(
            route.tokenizer_codes(exported, values, device=args.device)
        ),
        _negative(config.batch_size, config.seed + 7),
        behavior="support",
    )
    repeated_codes = _np_codes(
        route.tokenizer_codes(exported, validation, device=args.device)
    )
    checks["native_output_parity"] = parity_check(
        exported_codes,
        repeated_codes,
        atol=0,
        rtol=0,
        name="Kronos tokenizer code repeatability",
    )

    state_sha = _sha(state_path)
    bundle_sha = _sha(bundle_path)
    checks["checkpoint_lineage"] = _result(
        (
            state.get("route_key") == route.route_key(args.arm)
            and state.get("model_identity") == identity
            and state_sha
            == real_phase["artifacts"]["training_state"]["sha256"]
            and reopened_bundle.get("route_key") == route.route_key(args.arm)
            and bundle_sha
            == real_phase["artifacts"]["deployment_bundle"]["sha256"]
            and _sha(interrupted) == resume_phase["artifact"]["sha256"]
        ),
        {
            "training_state_sha256": state_sha,
            "deployment_bundle_sha256": bundle_sha,
            "interrupted_state_sha256": _sha(interrupted),
        },
        "Kronos tokenizer checkpoint lineage is incomplete",
    )
    fixture_document = _read_json(fixture_manifest)
    checks["data_lineage"] = _result(
        (
            _sha(fixture) == fixture_document["artifact"]["sha256"]
            and fixture_document["market_data_read"] is False
            and fixture_document["oos_read"] is False
        ),
        {
            "fixture_sha256": _sha(fixture),
            "market_data_read": False,
            "oos_read": False,
        },
        "Kronos tokenizer fixture lineage is invalid",
    )

    raw_checks = {name: check.manifest() for name, check in checks.items()}
    raw = {
        "schema_version": RAW_SCHEMA,
        "route_key": route.route_key(args.arm),
        "checks": raw_checks,
        "metrics": {
            "real_validation_loss": real_phase["metrics"]["real_validation_loss"],
            "shuffle_validation_loss": controls_phase["metrics"][
                "shuffle_validation_loss"
            ],
            "time_destroyed_validation_loss": controls_phase["metrics"][
                "time_destroyed_validation_loss"
            ],
            "all_checks_pass": all(
                check.status == "pass" for check in checks.values()
            ),
        },
        "phase_receipts": {
            phase: {
                "path": str(_phase_path(output, phase).resolve()),
                "sha256": _sha(_phase_path(output, phase)),
            }
            for phase in ("real", "controls", "resume")
        },
    }
    raw_path = _json(output / "raw_checks.json", raw)
    evidence = build_route_smoke_evidence(
        route_key=route.route_key(args.arm),
        executor_path=route.__file__,
        executor_entrypoint="native_loss/tokenizer_codes",
        checks=raw_checks,
        artifacts={
            "model_snapshot": Path(args.model_snapshot).resolve(),
            "tokenizer_snapshot": Path(args.tokenizer_snapshot).resolve(),
            "source_runtime": Path(args.source_runtime).resolve(),
            "synthetic_fixture": fixture,
            "synthetic_fixture_manifest": fixture_manifest,
            "interrupted_state": interrupted,
            "training_state": state_path,
            "deployment_bundle": bundle_path,
            "raw_checks": raw_path,
            "smoke_runner": Path(__file__).resolve(),
        },
        metrics=raw["metrics"],
    )
    validate_route_smoke_evidence(evidence)
    evidence_path = _json(output / "smoke_evidence.json", evidence)
    failed = [name for name, check in checks.items() if check.status != "pass"]
    return {
        "status": "pass" if evidence["smoke_admitted"] else "fail",
        "phase": "finalize",
        "route_key": route.route_key(args.arm),
        "smoke_admitted": evidence["smoke_admitted"],
        "evidence": {
            "path": str(evidence_path),
            "sha256": _sha(evidence_path),
            "content_sha256": evidence["evidence_sha256"],
        },
        "metrics": raw["metrics"],
        "failed_checks": failed,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = _config(args)
    signature = _phase_signature(args, config)
    if args.phase == "real":
        return run_real_phase(args, output, config, signature)
    if args.phase == "controls":
        return run_controls_phase(args, output, config, signature)
    if args.phase == "resume":
        return run_resume_phase(args, output, config, signature)
    if args.phase == "finalize":
        return run_finalize_phase(args, output, config, signature)
    results = []
    for phase in ("real", "controls", "resume", "finalize"):
        args.phase = phase
        results.append(run(args))
    return {"status": results[-1]["status"], "phases": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=route.SUPPORTED_ARMS, required=True)
    parser.add_argument("--model-snapshot", required=True)
    parser.add_argument("--tokenizer-snapshot", required=True)
    parser.add_argument("--source-runtime", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--phase",
        choices=("real", "controls", "resume", "finalize", "all"),
        default="all",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=CANONICAL_SMOKE_STEPS)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result.get("status") == "fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
