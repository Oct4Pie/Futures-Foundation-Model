"""Exact MOMENT-small classification and forecast task routes.

The pinned MOMENT implementation owns RevIN, patching, input masks, encoder, and task heads.
This module owns exact stepwise optimizer/scheduler state, sampler/RNG resume, and export
closure because the tutorial loops do not provide those guarantees.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import inspect
import random
import sys
from typing import Any, Mapping

import numpy as np

from ..native_contracts import get_dossier
from ..native_family_route_catalog_v2 import load_family_route_catalog
from ..native_training_schema_v2 import canonical_route_profile_sha256
from . import moment_reconstruction
from ._exact_state import (
    atomic_torch_save,
    build_export_bundle as _build_export_bundle,
    capture_state as _capture_state,
    load_torch,
    restore_export_bundle as _restore_export_bundle,
    restore_state as _restore_state,
)

ARM_KEY = "moment_small"
CONTEXT_LENGTH = 512
HORIZON_LENGTH = 16
CHANNELS = ("open", "high", "low", "close", "volume")
CHECKPOINT_SCHEMA = "ffm_moment_task_training_state_v1"
EXPORT_SCHEMA = "ffm_moment_task_bundle_v1"

ROUTES = {
    "moment_small:C:classification_full": {
        "track": "C", "route_id": "classification_full",
        "profile": "moment_classification_full", "task": "classification", "surface": "full",
    },
    "moment_small:C:classification_head_only": {
        "track": "C", "route_id": "classification_head_only",
        "profile": "moment_classification_head", "task": "classification", "surface": "head",
    },
    "moment_small:F:forecast_full_raw_mse": {
        "track": "F", "route_id": "forecast_full_raw_mse",
        "profile": "moment_forecast_full", "task": "forecast", "surface": "full",
    },
    "moment_small:F:forecast_head_only_raw_mse": {
        "track": "F", "route_id": "forecast_head_only_raw_mse",
        "profile": "moment_forecast_head", "task": "forecast", "surface": "head",
    },
}


def _torch():
    import torch
    return torch


def _route_hash(route_key: str) -> str:
    spec = ROUTES[route_key]
    return canonical_route_profile_sha256(ARM_KEY, spec["track"], spec["route_id"])


@dataclass(frozen=True)
class RouteConfig:
    route_key: str
    total_steps: int = 20
    n_classes: int = 3
    batch_size: int | None = None
    learning_rate: float | None = None
    weight_decay: float | None = None
    max_gradient_norm: float | None = None
    seed: int = 20260718

    def resolved(self) -> dict[str, Any]:
        if self.route_key not in ROUTES:
            raise ValueError("unknown MOMENT task route")
        policy = load_family_route_catalog()["constraint_profiles"][
            ROUTES[self.route_key]["profile"]
        ]["optimization_hyperparameters"]["value"]
        result = {
            "route_key": self.route_key,
            "total_steps": int(self.total_steps), "n_classes": int(self.n_classes),
            "batch_size": int(self.batch_size or policy["batch_size_default"]),
            "learning_rate": float(
                policy["learning_rate_default"] if self.learning_rate is None else self.learning_rate
            ),
            "weight_decay": float(
                policy["weight_decay_default"] if self.weight_decay is None else self.weight_decay
            ),
            "max_gradient_norm": float(
                policy["max_gradient_norm_default"]
                if self.max_gradient_norm is None else self.max_gradient_norm
            ),
            "seed": int(self.seed),
        }
        if result["n_classes"] < 2:
            raise ValueError("MOMENT classification requires at least two classes")
        if not policy["batch_size_min"] <= result["batch_size"] <= policy["batch_size_max"]:
            raise ValueError("MOMENT task batch size is outside the catalog bound")
        for field in ("learning_rate", "weight_decay", "max_gradient_norm"):
            if not policy[field + "_min"] <= result[field] <= policy[field + "_max"]:
                raise ValueError(f"MOMENT task {field} is outside the catalog bound")
        if not policy["smoke_steps"] <= result["total_steps"] <= policy["full_steps_max"]:
            raise ValueError("MOMENT task total_steps is outside the catalog bound")
        return result


@dataclass
class LoadedRoute:
    route_key: str
    model: Any
    identity: dict[str, Any]
    device: str

    @property
    def spec(self) -> dict[str, Any]:
        return ROUTES[self.route_key]

    @property
    def modules(self) -> dict[str, Any]:
        return {"model": self.model}


def seed_everything(seed: int) -> None:
    torch = _torch(); random.seed(int(seed)); np.random.seed(int(seed)); torch.manual_seed(int(seed))
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def load_route(
    route_key: str, *, model_snapshot: str | Path, source_runtime: str | Path,
    device: str, n_classes: int = 3,
) -> LoadedRoute:
    if route_key not in ROUTES:
        raise ValueError("unknown MOMENT task route")
    spec = ROUTES[route_key]
    source, source_identity = moment_reconstruction.validate_source_runtime(source_runtime)
    snapshot, model_identity = moment_reconstruction.validate_snapshot(model_snapshot)
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    try:
        from momentfm import MOMENTPipeline
    finally:
        if sys.path[0] == str(source):
            sys.path.pop(0)
    imported = Path(inspect.getfile(MOMENTPipeline)).resolve()
    if source not in imported.parents:
        raise RuntimeError("MOMENT task route did not import from pinned source")
    head_only = spec["surface"] == "head"
    kwargs = {
        "task_name": "classification" if spec["task"] == "classification" else "forecasting",
        "freeze_encoder": head_only,
        "freeze_embedder": head_only,
        "freeze_head": False,
        "enable_gradient_checkpointing": False,
        "head_dropout": 0.1,
    }
    if spec["task"] == "classification":
        kwargs.update({"n_channels": len(CHANNELS), "num_class": int(n_classes), "reduction": "concat"})
    else:
        kwargs["forecast_horizon"] = HORIZON_LENGTH
    model = MOMENTPipeline.from_pretrained(str(snapshot), model_kwargs=kwargs)
    model.init(); model.to(device)
    if int(model.seq_len) != CONTEXT_LENGTH or int(model.patch_len) != 8:
        raise RuntimeError("MOMENT task geometry drifted")
    if head_only:
        if any(parameter.requires_grad for parameter in model.encoder.parameters()):
            raise RuntimeError("MOMENT head-only encoder is not frozen")
        if any(parameter.requires_grad for parameter in model.patch_embedding.parameters()):
            raise RuntimeError("MOMENT head-only embedder is not frozen")
    identity = {
        **model_identity, "source_runtime": source_identity,
        "route_profile_sha256": _route_hash(route_key),
        "task": spec["task"], "surface": spec["surface"],
    }
    return LoadedRoute(route_key=route_key, model=model, identity=identity, device=str(device))


def _bars_and_mask(value: Any, *, expected_length: int, label: str) -> tuple[np.ndarray, np.ndarray]:
    bars = np.asarray(value, dtype=np.float32)
    if bars.ndim != 3 or bars.shape[1:] != (expected_length, len(CHANNELS)):
        raise ValueError(f"{label} must have shape [B,{expected_length},5]")
    finite = np.isfinite(bars)
    valid = finite.all(axis=2)
    partial = finite.any(axis=2) & ~valid
    if partial.any() or np.isinf(bars).any():
        raise ValueError(f"{label} has partial or infinite missing bars")
    filled = np.where(valid[:, :, None], bars, 0.0).astype(np.float32)
    return filled, valid.astype(np.float32)


def model_input(value: Any, *, device: str) -> tuple[Any, Any]:
    torch = _torch(); bars, mask = _bars_and_mask(value, expected_length=CONTEXT_LENGTH, label="MOMENT context")
    x = torch.as_tensor(bars.transpose(0, 2, 1), dtype=torch.float32, device=device)
    input_mask = torch.as_tensor(mask, dtype=torch.float32, device=device)
    return x, input_mask


def split_forecast_parent(value: Any) -> tuple[np.ndarray, np.ndarray]:
    bars = np.asarray(value, dtype=np.float32)
    if bars.ndim != 3 or bars.shape[1:] != (CONTEXT_LENGTH + HORIZON_LENGTH, len(CHANNELS)):
        raise ValueError("MOMENT forecast parent must have shape [B,528,5]")
    context = bars[:, :CONTEXT_LENGTH]
    target = bars[:, CONTEXT_LENGTH:]
    if not np.isfinite(target).all():
        raise ValueError("MOMENT forecast target must be complete and finite")
    _bars_and_mask(context, expected_length=CONTEXT_LENGTH, label="MOMENT context")
    return context, target


def classification_logits(loaded: LoadedRoute, context: Any) -> Any:
    if loaded.spec["task"] != "classification":
        raise ValueError("MOMENT route is not classification")
    x, mask = model_input(context, device=loaded.device)
    return loaded.model(x_enc=x, input_mask=mask).logits


def forecast_values(loaded: LoadedRoute, context: Any) -> Any:
    if loaded.spec["task"] != "forecast":
        raise ValueError("MOMENT route is not forecasting")
    x, mask = model_input(context, device=loaded.device)
    output = loaded.model(x_enc=x, input_mask=mask).forecast
    expected = (x.shape[0], len(CHANNELS), HORIZON_LENGTH)
    if tuple(output.shape) != expected or not _torch().isfinite(output).all():
        raise RuntimeError("MOMENT native forecast geometry or finiteness drifted")
    return output


def native_loss(loaded: LoadedRoute, parent: Any, *, labels: Any | None = None) -> Any:
    torch = _torch()
    if loaded.spec["task"] == "classification":
        if labels is None:
            raise ValueError("MOMENT classification labels are required")
        logits = classification_logits(loaded, parent)
        target = torch.as_tensor(labels, dtype=torch.long, device=loaded.device)
        if target.shape != (logits.shape[0],):
            raise ValueError("MOMENT labels are not aligned")
        return torch.nn.functional.cross_entropy(logits, target)
    context, target = split_forecast_parent(parent)
    forecast = forecast_values(loaded, context)
    target_tensor = torch.as_tensor(target.transpose(0, 2, 1), dtype=torch.float32, device=loaded.device)
    return torch.nn.functional.mse_loss(forecast, target_tensor)


def trainable_parameters(loaded: LoadedRoute) -> list[Any]:
    values = [parameter for parameter in loaded.model.parameters() if parameter.requires_grad]
    if not values:
        raise ValueError("MOMENT task route has no trainable parameters")
    return values


def make_optimizer_for_parameters(parameters: Any, config: RouteConfig) -> Any:
    torch = _torch(); value = config.resolved(); parameters = list(parameters)
    if not parameters:
        raise ValueError("MOMENT task has no optimizer parameters")
    return torch.optim.Adam(parameters, lr=value["learning_rate"], weight_decay=value["weight_decay"])


def make_optimizer(loaded: LoadedRoute, config: RouteConfig) -> Any:
    return make_optimizer_for_parameters(trainable_parameters(loaded), config)


def make_scheduler(optimizer: Any, config: RouteConfig) -> Any:
    torch = _torch(); value = config.resolved(); spec = ROUTES[config.route_key]
    if spec["task"] == "classification":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=value["learning_rate"], total_steps=value["total_steps"],
        pct_start=0.3, anneal_strategy="cos",
    )


def optimizer_step(
    loaded: LoadedRoute, optimizer: Any, scheduler: Any, parent: Any, *,
    config: RouteConfig, labels: Any | None = None,
) -> dict[str, float]:
    torch = _torch(); value = config.resolved()
    if config.route_key != loaded.route_key:
        raise ValueError("MOMENT config route differs from loaded route")
    loaded.model.train(); optimizer.zero_grad(set_to_none=True)
    parameters = trainable_parameters(loaded)
    before = [parameter.detach().clone() for parameter in parameters]
    loss = native_loss(loaded, parent, labels=labels); loss.backward()
    grads = [p.grad.detach().float().square().sum() for p in parameters if p.grad is not None]
    grad_norm = torch.sqrt(torch.stack(grads).sum()) if grads else torch.tensor(0.0, device=loaded.device)
    if value["max_gradient_norm"] > 0:
        torch.nn.utils.clip_grad_norm_(parameters, value["max_gradient_norm"])
    optimizer.step(); scheduler.step()
    parameter_delta = max(
        float(torch.max(torch.abs(parameter.detach() - previous)).cpu())
        for parameter, previous in zip(parameters, before)
    )
    return {
        "loss": float(loss.detach().cpu()), "grad_norm": float(grad_norm.detach().cpu()),
        "parameter_delta": parameter_delta,
    }


def capture_training_state(
    loaded: LoadedRoute, optimizer: Any, scheduler: Any, config: RouteConfig, *,
    global_step: int, sampler_cursor: int, history: list[Mapping[str, Any]],
    sampler_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if config.route_key != loaded.route_key:
        raise ValueError("MOMENT config route differs from loaded route")
    return _capture_state(
        schema_version=CHECKPOINT_SCHEMA, route_key=loaded.route_key,
        route_profile_sha256=_route_hash(loaded.route_key), model_identity=loaded.identity,
        config=config.resolved(), modules=loaded.modules, optimizer=optimizer,
        scheduler=scheduler, global_step=global_step, sampler_cursor=sampler_cursor,
        history=list(history), sampler_state=sampler_state,
    )


def restore_training_state(
    state: Mapping[str, Any], loaded: LoadedRoute, optimizer: Any, scheduler: Any,
    config: RouteConfig,
) -> tuple[int, int, int, list[dict[str, Any]]]:
    return _restore_state(
        state, schema_version=CHECKPOINT_SCHEMA, route_key=loaded.route_key,
        route_profile_sha256=_route_hash(loaded.route_key), model_identity=loaded.identity,
        config=config.resolved(), modules=loaded.modules, optimizer=optimizer,
        scheduler=scheduler,
    )


def deployment_output(loaded: LoadedRoute, context: Any) -> Any:
    loaded.model.eval()
    with _torch().no_grad():
        return (
            classification_logits(loaded, context)
            if loaded.spec["task"] == "classification" else forecast_values(loaded, context)
        )


def build_export_bundle(loaded: LoadedRoute, config: RouteConfig) -> dict[str, Any]:
    output = (
        {"kind": "classification_logits", "classes": config.resolved()["n_classes"]}
        if loaded.spec["task"] == "classification" else
        {"kind": "raw_forecast", "shape": ["batch", len(CHANNELS), HORIZON_LENGTH]}
    )
    return _build_export_bundle(
        schema_version=EXPORT_SCHEMA, route_key=loaded.route_key,
        route_profile_sha256=_route_hash(loaded.route_key), model_identity=loaded.identity,
        modules=loaded.modules,
        preprocessing={
            "input_shape": ["batch", CONTEXT_LENGTH, len(CHANNELS)],
            "model_layout": ["batch", len(CHANNELS), CONTEXT_LENGTH],
            "input_mask": "complete_timestamp_mask", "scaler": "moment_revin_per_channel",
            "missing_fill": "zero_before_mask",
        },
        output=output, extra={"task": loaded.spec["task"], "surface": loaded.spec["surface"]},
    )


def restore_export_bundle(bundle: Mapping[str, Any], loaded: LoadedRoute) -> dict[str, Any]:
    return _restore_export_bundle(
        bundle, schema_version=EXPORT_SCHEMA, route_key=loaded.route_key,
        route_profile_sha256=_route_hash(loaded.route_key), model_identity=loaded.identity,
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
    "HORIZON_LENGTH", "LoadedRoute", "ROUTES", "RouteConfig", "build_export_bundle",
    "capture_training_state", "classification_logits", "deployment_output", "forecast_values",
    "load_artifact", "load_export_bundle", "load_route", "make_optimizer", "make_optimizer_for_parameters",
    "make_scheduler", "model_input", "native_loss", "optimizer_step", "restore_export_bundle",
    "restore_training_state", "save_artifact", "seed_everything", "split_forecast_parent",
    "trainable_parameters",
]
