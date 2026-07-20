"""Exact Chronos-Bolt direct native quantile route.

Route identity:
``chronos_bolt:F:direct_native_quantile_pinball``

The route is intentionally narrow:
- five OHLCV channels are folded into independent univariate series;
- 512 visible bars and 16 future bars are consumed from a 528-bar parent;
- the pinned Chronos-Bolt model's native nine-quantile pinball loss is used;
- the complete model is trainable in FP32;
- checkpoints include optimizer, scheduler, sampler cursor and RNG state;
- deployment bundles contain the full model plus the exact preprocessing/output contract.

This module implements computation only.  Governance and evidence modules decide whether
smoke, pilot, or training execution is authorized.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np

from futures_foundation.finetune.native_contracts import get_dossier
from futures_foundation.finetune.native_training_schema_v2 import (
    canonical_route_profile_sha256,
)


ROUTE_KEY = "chronos_bolt:F:direct_native_quantile_pinball"
ARM_KEY = "chronos_bolt"
TRACK = "F"
ROUTE_ID = "direct_native_quantile_pinball"
CHECKPOINT_SCHEMA = "ffm_chronos_bolt_native_training_state_v1"
EXPORT_SCHEMA = "ffm_chronos_bolt_native_forecast_bundle_v1"
CONTEXT_LENGTH = 512
HORIZON_LENGTH = 16
PARENT_LENGTH = CONTEXT_LENGTH + HORIZON_LENGTH
CHANNELS = ("open", "high", "low", "close", "volume")
QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


@dataclass(frozen=True)
class RouteConfig:
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 20
    gradient_accumulation_steps: int = 1
    max_gradient_norm: float = 1.0
    total_steps: int = 20
    seed: int = 20260718

    def validate(self) -> None:
        if not 1e-5 <= float(self.learning_rate) <= 3e-4:
            raise ValueError("Chronos-Bolt learning_rate is outside the catalog bound")
        if not 0.0 <= float(self.weight_decay) <= 0.05:
            raise ValueError("Chronos-Bolt weight_decay is outside the catalog bound")
        if not 2 <= int(self.batch_size) <= 32:
            raise ValueError("Chronos-Bolt batch_size is outside the catalog bound")
        if int(self.gradient_accumulation_steps) != 1:
            raise ValueError("Chronos-Bolt exact executor supports gradient_accumulation_steps=1 only")
        if not 0.5 <= float(self.max_gradient_norm) <= 3.0:
            raise ValueError("Chronos-Bolt gradient clipping is outside the catalog bound")
        if not 1 <= int(self.total_steps) <= 32768:
            raise ValueError("Chronos-Bolt total_steps is outside the catalog bound")


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_torch_save(path: str | Path, value: object) -> Path:
    import torch

    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(target) + f".{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, target)
    return target


def _torch() -> Any:
    import torch

    return torch


def seed_everything(seed: int) -> None:
    torch = _torch()
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def validate_snapshot(path: str | Path) -> tuple[Path, dict[str, Any]]:
    snapshot = Path(path).expanduser().resolve()
    if not snapshot.is_dir() or snapshot.is_symlink():
        raise FileNotFoundError(f"Chronos-Bolt snapshot is missing or unsafe: {snapshot}")
    config_path = snapshot / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Chronos-Bolt snapshot lacks a resolved config.json: {snapshot}")
    dossier = get_dossier(ARM_KEY)
    expected_revision = str(dossier["model_revision"])
    if snapshot.name != expected_revision:
        raise ValueError(
            f"Chronos-Bolt snapshot revision mismatch: expected {expected_revision}, got {snapshot.name}"
        )
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Chronos-Bolt snapshot config is invalid JSON") from exc
    chronos = config.get("chronos_config") if isinstance(config, dict) else None
    if not isinstance(chronos, dict):
        raise ValueError("Chronos-Bolt snapshot lacks chronos_config")
    if (
        int(chronos.get("input_patch_size", -1)) != 16
        or int(chronos.get("prediction_length", -1)) != 64
        or tuple(float(value) for value in chronos.get("quantiles", ())) != QUANTILES
    ):
        raise ValueError("Chronos-Bolt snapshot native geometry differs from the route contract")
    weight_files = [
        candidate for pattern in ("*.safetensors", "*.bin")
        for candidate in snapshot.glob(pattern) if candidate.is_file()
    ]
    if not weight_files:
        raise FileNotFoundError("Chronos-Bolt snapshot has no materialized weights")
    identity = {
        "path": str(snapshot),
        "model_id": dossier["model_id"],
        "model_revision": expected_revision,
        "config_sha256": _sha256(config_path),
        "weight_files": [
            {"path": item.name, "sha256": _sha256(item), "bytes": item.stat().st_size}
            for item in sorted(weight_files)
        ],
    }
    return snapshot, identity


def load_pipeline(snapshot: str | Path, *, device: str) -> tuple[Any, Any, dict[str, Any]]:
    torch = _torch()
    from chronos import BaseChronosPipeline

    resolved, identity = validate_snapshot(snapshot)
    pipeline = BaseChronosPipeline.from_pretrained(
        str(resolved), device_map="cpu", dtype=torch.float32,
    )
    model = pipeline.inner_model.to(device)
    loaded_revision = getattr(model.config, "_commit_hash", None)
    if loaded_revision != identity["model_revision"]:
        raise RuntimeError(
            f"Chronos-Bolt loaded revision mismatch: expected {identity['model_revision']}, got {loaded_revision}"
        )
    native = model.chronos_config
    if (
        int(native.input_patch_size) != 16
        or int(native.prediction_length) != 64
        or not np.allclose(
            np.asarray(model.quantiles.detach().cpu(), dtype=np.float64),
            np.asarray(QUANTILES, dtype=np.float64),
            rtol=0.0,
            atol=1e-7,
        )
    ):
        raise RuntimeError("loaded Chronos-Bolt native route geometry drifted")
    return pipeline, model, identity


def parent_array(value: Any) -> np.ndarray:
    parent = np.asarray(value, dtype=np.float32)
    if parent.ndim != 3 or parent.shape[1:] != (PARENT_LENGTH, len(CHANNELS)):
        raise ValueError(
            f"Chronos-Bolt parent must have shape [B,{PARENT_LENGTH},{len(CHANNELS)}], got {parent.shape}"
        )
    if np.isinf(parent).any():
        raise ValueError("Chronos-Bolt parent contains infinite values")
    context = parent[:, :CONTEXT_LENGTH]
    future = parent[:, CONTEXT_LENGTH:]
    if not np.isfinite(future).all():
        raise ValueError("Chronos-Bolt future targets must be finite")
    o, h, l, c, v = parent.transpose(2, 0, 1)
    finite_ohlc = np.isfinite(o) & np.isfinite(h) & np.isfinite(l) & np.isfinite(c)
    if np.any(
        finite_ohlc
        & ((h < np.maximum(o, c)) | (l > np.minimum(o, c)) | (h < l))
    ):
        raise ValueError("Chronos-Bolt parent violates finite-row OHLC geometry")
    if np.any(np.isfinite(v) & (v < 0)):
        raise ValueError("Chronos-Bolt parent contains negative finite volume")
    if not np.isfinite(context).any(axis=1).all():
        raise ValueError("Chronos-Bolt every channel requires at least one observed context value")
    return parent


def split_parent(parent: Any, *, device: str) -> tuple[Any, Any]:
    torch = _torch()
    values = parent_array(parent)
    context = values[:, :CONTEXT_LENGTH]
    future = values[:, CONTEXT_LENGTH:PARENT_LENGTH]
    flattened_context = np.transpose(context, (0, 2, 1)).reshape(-1, CONTEXT_LENGTH)
    flattened_future = np.transpose(future, (0, 2, 1)).reshape(-1, HORIZON_LENGTH)
    return (
        torch.as_tensor(flattened_context, dtype=torch.float32, device=device),
        torch.as_tensor(flattened_future, dtype=torch.float32, device=device),
    )


def contexts_only(parent: Any, *, device: str) -> Any:
    context, _ = split_parent(parent, device=device)
    return context


def native_loss(model: Any, parent: Any, *, device: str) -> Any:
    context, future = split_parent(parent, device=device)
    output = model(context=context, target=future)
    if output.loss is None or not _torch().isfinite(output.loss):
        raise FloatingPointError("Chronos-Bolt native loss is non-finite")
    return output.loss


def direct_quantiles(model: Any, parent: Any, *, device: str) -> Any:
    torch = _torch()
    values = parent_array(parent)
    batch = len(values)
    context, _ = split_parent(values, device=device)
    output = model(context=context).quantile_preds
    if output.shape != (batch * len(CHANNELS), len(QUANTILES), 64):
        raise RuntimeError(f"Chronos-Bolt native quantile shape drifted: {tuple(output.shape)}")
    result = output[:, :, :HORIZON_LENGTH].permute(0, 2, 1).reshape(
        batch, len(CHANNELS), HORIZON_LENGTH, len(QUANTILES),
    )
    if not torch.isfinite(result).all():
        raise FloatingPointError("Chronos-Bolt direct quantiles are non-finite")
    return result


def public_quantiles(pipeline: Any, parent: Any, *, device: str) -> Any:
    torch = _torch()
    values = parent_array(parent)
    batch = len(values)
    context, _ = split_parent(values, device=device)
    quantiles = pipeline.predict_quantiles(
        context,
        prediction_length=HORIZON_LENGTH,
        quantile_levels=list(QUANTILES),
    )[0]
    if quantiles.shape != (batch * len(CHANNELS), HORIZON_LENGTH, len(QUANTILES)):
        raise RuntimeError(f"Chronos-Bolt public quantile shape drifted: {tuple(quantiles.shape)}")
    result = quantiles.reshape(batch, len(CHANNELS), HORIZON_LENGTH, len(QUANTILES))
    if not torch.isfinite(result).all():
        raise FloatingPointError("Chronos-Bolt public quantiles are non-finite")
    return result


def make_optimizer(model: Any, config: RouteConfig) -> Any:
    torch = _torch()
    config.validate()
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
        betas=(0.9, 0.999),
        eps=1e-8,
    )


def make_scheduler(optimizer: Any, config: RouteConfig) -> Any:
    torch = _torch()
    config.validate()
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(config.total_steps),
    )


def model_state_cpu(model: Any) -> dict[str, Any]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def optimizer_step(
    model: Any,
    optimizer: Any,
    scheduler: Any,
    parent: Any,
    *,
    device: str,
    max_gradient_norm: float,
) -> dict[str, float]:
    torch = _torch()
    model.train()
    parameter = next(value for value in model.parameters() if value.requires_grad)
    before = parameter.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    loss = native_loss(model, parent, device=device)
    loss.backward()
    grad_norm = torch.sqrt(sum(
        value.grad.detach().float().square().sum()
        for value in model.parameters()
        if value.requires_grad and value.grad is not None
    ))
    torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_gradient_norm))
    optimizer.step()
    scheduler.step()
    parameter_delta = torch.max(torch.abs(parameter.detach() - before))
    return {
        "loss": float(loss.detach().cpu()),
        "grad_norm": float(grad_norm.detach().cpu()),
        "parameter_delta": float(parameter_delta.detach().cpu()),
    }


def _cpu_tree(value: Any) -> Any:
    torch = _torch()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    return value


def capture_training_state(
    *,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    config: RouteConfig,
    model_identity: Mapping[str, Any],
    global_step: int,
    sampler_cursor: int,
    history: list[dict[str, Any]],
    sampler_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    torch = _torch()
    config.validate()
    sampler = (
        {
            "cursor": int(sampler_cursor),
            "schedule_kind": "explicit_synthetic_batch_v1",
            "schedule_sha256": hashlib.sha256(
                b"explicit_synthetic_batch_v1"
            ).hexdigest(),
        }
        if sampler_state is None else dict(sampler_state)
    )
    if set(sampler) != {"cursor", "schedule_kind", "schedule_sha256"}:
        raise ValueError("Chronos-Bolt sampler state field closure is invalid")
    if int(sampler["cursor"]) != int(sampler_cursor):
        raise ValueError("Chronos-Bolt sampler cursor differs from global state")
    if (
        not isinstance(sampler["schedule_kind"], str)
        or not sampler["schedule_kind"]
        or not isinstance(sampler["schedule_sha256"], str)
        or len(sampler["schedule_sha256"]) != 64
    ):
        raise ValueError("Chronos-Bolt sampler identity is malformed")
    return {
        "schema_version": CHECKPOINT_SCHEMA,
        "route_key": ROUTE_KEY,
        "route_profile_sha256": canonical_route_profile_sha256(ARM_KEY, TRACK, ROUTE_ID),
        "model_identity": dict(model_identity),
        "config": asdict(config),
        "model": model_state_cpu(model),
        "optimizer": _cpu_tree(optimizer.state_dict()),
        "scheduler": _cpu_tree(scheduler.state_dict()),
        "scaler": None,
        "epoch": 0,
        "global_step": int(global_step),
        "sampler": sampler,
        "history": deepcopy_history(history),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_cpu_rng": torch.get_rng_state(),
        "torch_cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def deepcopy_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return json.loads(json.dumps(history, allow_nan=False))


def restore_training_state(
    state: Mapping[str, Any],
    *,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    config: RouteConfig,
    model_identity: Mapping[str, Any],
) -> tuple[int, int, list[dict[str, Any]]]:
    torch = _torch()
    required = {
        "schema_version", "route_key", "route_profile_sha256", "model_identity", "config",
        "model", "optimizer", "scheduler", "scaler", "epoch", "global_step", "sampler",
        "history", "python_rng", "numpy_rng", "torch_cpu_rng", "torch_cuda_rng",
    }
    if not isinstance(state, Mapping) or set(state) != required:
        raise ValueError("Chronos-Bolt training state field closure is invalid")
    if state["schema_version"] != CHECKPOINT_SCHEMA or state["route_key"] != ROUTE_KEY:
        raise ValueError("Chronos-Bolt training state route identity mismatch")
    if state["route_profile_sha256"] != canonical_route_profile_sha256(ARM_KEY, TRACK, ROUTE_ID):
        raise ValueError("Chronos-Bolt training state profile binding is stale")
    if dict(state["model_identity"]) != dict(model_identity) or dict(state["config"]) != asdict(config):
        raise ValueError("Chronos-Bolt training state model/config identity mismatch")
    if state["scaler"] is not None or state["epoch"] != 0:
        raise ValueError("Chronos-Bolt FP32 step-based state has invalid scaler/epoch fields")
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_cpu_rng"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda_rng"])
    sampler = state["sampler"]
    if not isinstance(sampler, Mapping) or set(sampler) != {
        "cursor", "schedule_kind", "schedule_sha256",
    }:
        raise ValueError("Chronos-Bolt sampler state is malformed")
    if (
        not isinstance(sampler["schedule_kind"], str)
        or not sampler["schedule_kind"]
        or not isinstance(sampler["schedule_sha256"], str)
        or len(sampler["schedule_sha256"]) != 64
    ):
        raise ValueError("Chronos-Bolt sampler identity is malformed")
    return int(state["global_step"]), int(sampler["cursor"]), deepcopy_history(state["history"])


def save_training_state(path: str | Path, state: Mapping[str, Any]) -> dict[str, Any]:
    target = _atomic_torch_save(path, dict(state))
    return {"path": str(target), "sha256": _sha256(target), "bytes": target.stat().st_size}


def load_training_state(path: str | Path) -> dict[str, Any]:
    torch = _torch()
    source = Path(path).expanduser().resolve()
    return torch.load(source, map_location="cpu", weights_only=False)


def build_export_bundle(
    *,
    model: Any,
    model_identity: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": EXPORT_SCHEMA,
        "route_key": ROUTE_KEY,
        "route_profile_sha256": canonical_route_profile_sha256(ARM_KEY, TRACK, ROUTE_ID),
        "model_identity": dict(model_identity),
        "model_state": model_state_cpu(model),
        "preprocessing": {
            "owner": "upstream_internal",
            "scaler_tag": "bolt_native",
            "missing_mask_tag": "native_patch_mask",
            "context_length": CONTEXT_LENGTH,
            "channel_layout": "independent_univariate_passes",
            "channel_order": list(CHANNELS),
        },
        "output": {
            "kind": "native_quantiles",
            "shape": ["batch", len(CHANNELS), HORIZON_LENGTH, len(QUANTILES)],
            "quantile_levels": list(QUANTILES),
            "native_prediction_length": 64,
            "deployment_filter": "first_16_native_positions",
            "hidden_state_contract": "none_forecast_route",
        },
    }


def save_export_bundle(path: str | Path, bundle: Mapping[str, Any]) -> dict[str, Any]:
    target = _atomic_torch_save(path, dict(bundle))
    return {"path": str(target), "sha256": _sha256(target), "bytes": target.stat().st_size}


def load_export_bundle(
    path: str | Path,
    *,
    snapshot: str | Path,
    device: str,
) -> tuple[Any, Any, dict[str, Any]]:
    torch = _torch()
    source = Path(path).expanduser().resolve()
    bundle = torch.load(source, map_location="cpu", weights_only=False)
    expected = {
        "schema_version", "route_key", "route_profile_sha256", "model_identity",
        "model_state", "preprocessing", "output",
    }
    if not isinstance(bundle, Mapping) or set(bundle) != expected:
        raise ValueError("Chronos-Bolt export bundle field closure is invalid")
    if bundle["schema_version"] != EXPORT_SCHEMA or bundle["route_key"] != ROUTE_KEY:
        raise ValueError("Chronos-Bolt export bundle route identity mismatch")
    if bundle["route_profile_sha256"] != canonical_route_profile_sha256(ARM_KEY, TRACK, ROUTE_ID):
        raise ValueError("Chronos-Bolt export bundle profile binding is stale")
    pipeline, model, identity = load_pipeline(snapshot, device=device)
    if dict(bundle["model_identity"]) != identity:
        raise ValueError("Chronos-Bolt export bundle model identity mismatch")
    model.load_state_dict(bundle["model_state"], strict=True)
    model.eval()
    return pipeline, model, dict(bundle)


__all__ = [
    "ARM_KEY", "CHANNELS", "CHECKPOINT_SCHEMA", "CONTEXT_LENGTH", "EXPORT_SCHEMA",
    "HORIZON_LENGTH", "PARENT_LENGTH", "QUANTILES", "ROUTE_ID", "ROUTE_KEY", "TRACK",
    "RouteConfig", "build_export_bundle", "capture_training_state", "contexts_only",
    "direct_quantiles", "load_export_bundle", "load_pipeline", "load_training_state",
    "make_optimizer", "make_scheduler", "model_state_cpu", "native_loss", "optimizer_step",
    "parent_array", "public_quantiles", "restore_training_state", "save_export_bundle",
    "save_training_state", "seed_everything", "split_parent", "validate_snapshot",
]
