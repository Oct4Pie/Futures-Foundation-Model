"""Exact noncommercial-research Moirai-2 Small scaled-pinball route.

The route uses the pinned Moirai2Module packing/scaling implementation and computes
pinball loss on every native predicted output patch.  It is permanently restricted to
noncommercial research by the checkpoint license and grants no production admission.
"""
from __future__ import annotations

from dataclasses import dataclass
import inspect
from pathlib import Path
import random
import subprocess
import sys
from typing import Any, Mapping

import numpy as np

from ..native_adapters import moirai2_native_forecast
from ..native_contracts import get_dossier
from ..native_family_route_catalog_v2 import load_family_route_catalog
from ..native_training_schema_v2 import canonical_route_profile_sha256
from ._exact_state import (
    atomic_torch_save,
    build_export_bundle as _build_export_bundle,
    capture_state as _capture_state,
    load_torch,
    restore_export_bundle as _restore_export_bundle,
    restore_state as _restore_state,
)

ARM_KEY = "moirai2_small"
TRACK = "F"
ROUTE_ID = "custom_scaled_pinball_research"
ROUTE_KEY = f"{ARM_KEY}:{TRACK}:{ROUTE_ID}"
PROFILE = "moirai_custom"
CONTEXT_LENGTH = 512
HORIZON_LENGTH = 16
CHANNELS = ("open", "high", "low", "close", "volume")
PATCH_SIZE = 16
QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
CHECKPOINT_SCHEMA = "ffm_moirai2_scaled_pinball_state_v1"
EXPORT_SCHEMA = "ffm_moirai2_scaled_pinball_export_v1"


def _torch():
    import torch
    return torch


def _route_hash() -> str:
    return canonical_route_profile_sha256(ARM_KEY, TRACK, ROUTE_ID)


def _normalize_origin(value: str) -> str:
    return value.strip().lower().removesuffix(".git").replace(
        "git@github.com:", "https://github.com/"
    )


def validate_source_runtime(path: str | Path) -> Path:
    source = Path(path).expanduser().resolve(); dossier = get_dossier(ARM_KEY)
    if not (source / ".git").is_dir():
        raise ValueError("Moirai source runtime must be a Git checkout")
    head = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True,
    ).strip()
    if head != dossier["source_revision"]:
        raise ValueError("Moirai source revision mismatch")
    dirty = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=all"],
        text=True,
    ).strip()
    if dirty:
        raise ValueError("Moirai source runtime must be clean")
    origin = subprocess.check_output(
        ["git", "-C", str(source), "remote", "get-url", "origin"], text=True,
    ).strip()
    if _normalize_origin(origin) != _normalize_origin(dossier["source_url"]):
        raise ValueError("Moirai source origin mismatch")
    return source


def validate_snapshot(path: str | Path) -> Path:
    snapshot = Path(path).expanduser().resolve(); dossier = get_dossier(ARM_KEY)
    if snapshot.name != dossier["model_revision"] or not (snapshot / "config.json").is_file():
        raise ValueError("Moirai snapshot identity mismatch")
    if not any(snapshot.glob("*.safetensors")):
        raise ValueError("Moirai snapshot weights are missing")
    return snapshot


@dataclass(frozen=True)
class RouteConfig:
    total_steps: int = 20
    batch_size: int = 64
    learning_rate: float = 5e-7
    weight_decay: float = 0.1
    max_gradient_norm: float = 1.0
    seed: int = 20260718

    def validate(self) -> None:
        policy = load_family_route_catalog()["constraint_profiles"][PROFILE][
            "optimization_hyperparameters"
        ]["value"]
        values = {
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "max_gradient_norm": self.max_gradient_norm,
        }
        for field, value in values.items():
            if not policy[field + "_min"] <= value <= policy[field + "_max"]:
                raise ValueError(f"Moirai {field} is outside the catalog bound")
        if not policy["smoke_steps"] <= self.total_steps <= policy["full_steps_max"]:
            raise ValueError("Moirai total_steps is outside the catalog bound")


@dataclass
class LoadedRoute:
    forecast: Any
    module: Any
    identity: dict[str, Any]
    device: str

    @property
    def modules(self) -> dict[str, Any]:
        return {"module": self.module}


def seed_everything(seed: int) -> None:
    torch = _torch(); random.seed(int(seed)); np.random.seed(int(seed)); torch.manual_seed(int(seed))
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def load_route(
    *, model_snapshot: str | Path, source_runtime: str | Path, device: str,
) -> LoadedRoute:
    source = validate_source_runtime(source_runtime); snapshot = validate_snapshot(model_snapshot)
    source_path = source / "src"
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))
    from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
    import uni2ts
    imported = Path(inspect.getfile(uni2ts)).resolve()
    if source not in imported.parents:
        raise RuntimeError("Moirai imported outside the pinned source")
    module = Moirai2Module.from_pretrained(str(snapshot)).to(device)
    if (
        int(module.patch_size) != PATCH_SIZE
        or int(module.num_predict_token) < 1
        or tuple(float(value) for value in module.quantile_levels) != QUANTILES
    ):
        raise RuntimeError("Moirai native patch/quantile configuration drifted")
    forecast = Moirai2Forecast(
        prediction_length=HORIZON_LENGTH,
        target_dim=len(CHANNELS),
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
        context_length=CONTEXT_LENGTH,
        module=module,
    ).to(device)
    identity = {
        "model_id": get_dossier(ARM_KEY)["model_id"],
        "model_revision": snapshot.name,
        "source_revision": get_dossier(ARM_KEY)["source_revision"],
        "source_runtime": str(source),
        "route_profile_sha256": _route_hash(),
        "license_scope": "research_noncommercial",
        "patch_size": PATCH_SIZE,
        "quantiles": list(QUANTILES),
    }
    return LoadedRoute(forecast, module, identity, str(device))


def split_parent(value: Any) -> tuple[Any, Any, Any, Any, Any]:
    torch = _torch(); parent = torch.as_tensor(value, dtype=torch.float32)
    if parent.ndim != 3 or tuple(parent.shape[1:]) != (
        CONTEXT_LENGTH + HORIZON_LENGTH, len(CHANNELS)
    ):
        raise ValueError("Moirai parent must have shape [B,528,5]")
    if torch.isinf(parent).any():
        raise ValueError("Moirai parent contains infinite values")
    context, future = parent[:, :CONTEXT_LENGTH], parent[:, CONTEXT_LENGTH:]
    context_observed = torch.isfinite(context)
    future_observed = torch.isfinite(future)
    context = torch.where(context_observed, context, torch.zeros_like(context))
    future = torch.where(future_observed, future, torch.zeros_like(future))
    past_is_pad = torch.zeros(parent.shape[0], CONTEXT_LENGTH, dtype=torch.bool)
    return context, context_observed, past_is_pad, future, future_observed


def packed_training_tensors(loaded: LoadedRoute, parent: Any) -> dict[str, Any]:
    context, context_observed, past_is_pad, future, future_observed = split_parent(parent)
    context = context.to(loaded.device); context_observed = context_observed.to(loaded.device)
    past_is_pad = past_is_pad.to(loaded.device); future = future.to(loaded.device)
    future_observed = future_observed.to(loaded.device)
    converted = loaded.forecast._convert(
        loaded.module.patch_size,
        context,
        context_observed,
        past_is_pad,
        future_target=future,
        future_observed_target=future_observed,
    )
    names = ("target", "observed_mask", "sample_id", "time_id", "variate_id", "prediction_mask")
    return {name: tensor for name, tensor in zip(names, converted)}


def native_predictions_and_target(loaded: LoadedRoute, parent: Any) -> tuple[Any, Any, Any]:
    torch = _torch(); packed = packed_training_tensors(loaded, parent)
    raw, scaled_target = loaded.module(
        packed["target"], packed["observed_mask"], packed["sample_id"],
        packed["time_id"], packed["variate_id"], packed["prediction_mask"],
        training_mode=True,
    )
    batch = raw.shape[0]
    raw = raw.reshape(
        batch, raw.shape[1], loaded.module.num_predict_token,
        loaded.module.num_quantiles, loaded.module.patch_size,
    )
    context_tokens = loaded.forecast.context_token_length(PATCH_SIZE)
    prediction_tokens = loaded.forecast.prediction_token_length(PATCH_SIZE)
    if prediction_tokens != 1:
        raise RuntimeError("Moirai route expects exactly one prediction token per variate")
    pred_index = torch.arange(
        context_tokens - 1, len(CHANNELS) * context_tokens, context_tokens,
        device=raw.device,
    )
    target_index = torch.arange(
        len(CHANNELS) * context_tokens,
        len(CHANNELS) * context_tokens + len(CHANNELS),
        device=raw.device,
    )
    predictions = raw[:, pred_index, 0]
    target = scaled_target[:, target_index]
    observed = packed["observed_mask"][:, target_index]
    if predictions.shape != (batch, len(CHANNELS), len(QUANTILES), PATCH_SIZE):
        raise RuntimeError("Moirai native prediction geometry drifted")
    return predictions, target, observed


def native_loss(loaded: LoadedRoute, parent: Any) -> Any:
    torch = _torch(); predictions, target, observed = native_predictions_and_target(loaded, parent)
    quantiles = torch.tensor(QUANTILES, device=predictions.device, dtype=predictions.dtype)
    error = target[:, :, None, :] - predictions
    pinball = torch.maximum((quantiles[None, None, :, None] - 1.0) * error,
                            quantiles[None, None, :, None] * error)
    weights = observed[:, :, None, :].to(pinball.dtype)
    denominator = weights.sum() * len(QUANTILES)
    if denominator <= 0:
        raise ValueError("Moirai future target has no observed values")
    loss = (pinball * weights).sum() / denominator
    if not torch.isfinite(loss):
        raise RuntimeError("Moirai native pinball loss is non-finite")
    return loss


def public_output(loaded: LoadedRoute, context: Any) -> Any:
    torch = _torch(); values = torch.as_tensor(context, dtype=torch.float32)
    if values.ndim != 3 or tuple(values.shape[1:]) != (CONTEXT_LENGTH, len(CHANNELS)):
        raise ValueError("Moirai context must have shape [B,512,5]")
    if torch.isinf(values).any():
        raise ValueError("Moirai context contains infinite values")
    observed = torch.isfinite(values)
    values = torch.where(observed, values, torch.zeros_like(values)).to(loaded.device)
    observed = observed.to(loaded.device)
    pad = torch.zeros(values.shape[:2], dtype=torch.bool, device=loaded.device)
    return moirai2_native_forecast(loaded.forecast, values, observed, pad)


def trainable_parameters(loaded: LoadedRoute) -> list[Any]:
    values = [parameter for parameter in loaded.module.parameters() if parameter.requires_grad]
    if not values:
        raise ValueError("Moirai route has no trainable parameters")
    return values


def make_optimizer_for_parameters(parameters: Any, config: RouteConfig) -> Any:
    config.validate(); params = list(parameters)
    if not params:
        raise ValueError("Moirai route has no optimizer parameters")
    return _torch().optim.AdamW(
        params, lr=config.learning_rate, betas=(0.9, 0.98), eps=1e-6,
        weight_decay=config.weight_decay,
    )


def make_optimizer(loaded: LoadedRoute, config: RouteConfig) -> Any:
    return make_optimizer_for_parameters(trainable_parameters(loaded), config)


def make_scheduler(optimizer: Any, config: RouteConfig) -> Any:
    config.validate()
    return _torch().optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)


def optimizer_step(
    loaded: LoadedRoute, optimizer: Any, scheduler: Any, parent: Any, *, config: RouteConfig,
) -> dict[str, float]:
    torch = _torch(); config.validate(); loaded.module.train(); optimizer.zero_grad(set_to_none=True)
    parameter = trainable_parameters(loaded)[0]; before = parameter.detach().clone()
    loss = native_loss(loaded, parent); loss.backward()
    grads = [p.grad.detach().float().square().sum() for p in trainable_parameters(loaded) if p.grad is not None]
    grad_norm = torch.sqrt(torch.stack(grads).sum()) if grads else torch.tensor(0.0, device=loaded.device)
    torch.nn.utils.clip_grad_norm_(trainable_parameters(loaded), config.max_gradient_norm)
    optimizer.step(); scheduler.step()
    return {
        "loss": float(loss.detach().cpu()),
        "grad_norm": float(grad_norm.detach().cpu()),
        "parameter_delta": float(torch.max(torch.abs(parameter.detach() - before)).cpu()),
    }


def _config_dict(config: RouteConfig) -> dict[str, Any]:
    config.validate()
    return {
        "total_steps": config.total_steps,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "max_gradient_norm": config.max_gradient_norm,
        "seed": config.seed,
    }


def capture_training_state(
    loaded: LoadedRoute, optimizer: Any, scheduler: Any, config: RouteConfig, *,
    global_step: int, sampler_cursor: int, history: list[Mapping[str, Any]],
    sampler_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return _capture_state(
        schema_version=CHECKPOINT_SCHEMA,
        route_key=ROUTE_KEY,
        route_profile_sha256=_route_hash(),
        model_identity=loaded.identity,
        config=_config_dict(config),
        modules=loaded.modules,
        optimizer=optimizer,
        scheduler=scheduler,
        global_step=global_step,
        sampler_cursor=sampler_cursor,
        history=list(history),
        sampler_state=sampler_state,
    )


def restore_training_state(
    state: Mapping[str, Any], loaded: LoadedRoute, optimizer: Any, scheduler: Any,
    config: RouteConfig,
) -> tuple[int, int, int, list[dict[str, Any]]]:
    return _restore_state(
        state,
        schema_version=CHECKPOINT_SCHEMA,
        route_key=ROUTE_KEY,
        route_profile_sha256=_route_hash(),
        model_identity=loaded.identity,
        config=_config_dict(config),
        modules=loaded.modules,
        optimizer=optimizer,
        scheduler=scheduler,
    )


def build_export_bundle(loaded: LoadedRoute) -> dict[str, Any]:
    return _build_export_bundle(
        schema_version=EXPORT_SCHEMA,
        route_key=ROUTE_KEY,
        route_profile_sha256=_route_hash(),
        model_identity=loaded.identity,
        modules=loaded.modules,
        preprocessing={
            "context_length": CONTEXT_LENGTH,
            "prediction_length": HORIZON_LENGTH,
            "patch_size": PATCH_SIZE,
            "channel_order": list(CHANNELS),
            "layout": "packed_joint_multivariate",
            "scaler": "packed_std",
            "missing_policy": "observed_mask_zero_fill",
        },
        output={
            "kind": "native_quantiles",
            "quantile_levels": list(QUANTILES),
            "shape": ["batch", len(QUANTILES), HORIZON_LENGTH, len(CHANNELS)],
            "crossing_repair": False,
        },
        extra={"license_scope": "research_noncommercial", "production_admitted": False},
    )


def restore_export_bundle(bundle: Mapping[str, Any], loaded: LoadedRoute) -> dict[str, Any]:
    return _restore_export_bundle(
        bundle,
        schema_version=EXPORT_SCHEMA,
        route_key=ROUTE_KEY,
        route_profile_sha256=_route_hash(),
        model_identity=loaded.identity,
        modules=loaded.modules,
    )


def load_export_bundle(
    path: str | Path,
    *,
    loaded: LoadedRoute,
) -> tuple[LoadedRoute, dict[str, Any]]:
    bundle = load_torch(path)
    restored = restore_export_bundle(bundle, loaded)
    return loaded, restored


def save_artifact(path: str | Path, value: Any) -> dict[str, Any]:
    return atomic_torch_save(path, value)


def load_artifact(path: str | Path) -> Any:
    return load_torch(path)


__all__ = [
    "ARM_KEY", "CHANNELS", "CHECKPOINT_SCHEMA", "CONTEXT_LENGTH", "EXPORT_SCHEMA",
    "HORIZON_LENGTH", "LoadedRoute", "PATCH_SIZE", "PROFILE", "QUANTILES", "ROUTE_ID",
    "ROUTE_KEY", "RouteConfig", "build_export_bundle", "capture_training_state",
    "load_artifact", "load_export_bundle", "load_route", "make_optimizer", "make_optimizer_for_parameters",
    "make_scheduler", "native_loss", "native_predictions_and_target", "optimizer_step",
    "packed_training_tensors", "public_output", "restore_export_bundle",
    "restore_training_state", "save_artifact", "seed_everything", "split_parent",
    "trainable_parameters", "validate_snapshot", "validate_source_runtime",
]
