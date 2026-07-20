"""Exact TTM-R2 full and head/decoder/frequency-prefix forecast routes."""
from __future__ import annotations

from dataclasses import dataclass
import inspect
from pathlib import Path
import random
import subprocess
import sys
from typing import Any, Mapping

import numpy as np

from ..native_adapters import ttm_native_forecast
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

ARM_KEY = "ttm_r2"
CONTEXT_LENGTH = 512
HORIZON_LENGTH = 16
NATIVE_HORIZON = 48
CHANNELS = ("open", "high", "low", "close", "volume")
SELECTOR = "512-48-ft-r2.1"
FREQUENCY_TOKENS = {"1min": 1, "3min": 0, "5min": 3, "15min": 5, "30min": 6, "60min": 7}
CHECKPOINT_SCHEMA = "ffm_ttm_r2_training_state_v1"
EXPORT_SCHEMA = "ffm_ttm_r2_export_v1"
ROUTES = {
    "ttm_r2:F:full_model_raw_hf_trainer_forecast": {
        "track": "F", "route_id": "full_model_raw_hf_trainer_forecast",
        "profile": "ttm_forecast_full", "surface": "full",
    },
    "ttm_r2:F:head_prefix_raw_hf_trainer_forecast": {
        "track": "F", "route_id": "head_prefix_raw_hf_trainer_forecast",
        "profile": "ttm_forecast_head_prefix", "surface": "head_prefix",
    },
}


def _torch():
    import torch
    return torch


def _route_hash(route_key: str) -> str:
    spec = ROUTES[route_key]
    return canonical_route_profile_sha256(ARM_KEY, spec["track"], spec["route_id"])


def _normalize_origin(value: str) -> str:
    return value.strip().lower().removesuffix(".git").replace("git@github.com:", "https://github.com/")


def validate_source_runtime(path: str | Path) -> Path:
    source = Path(path).expanduser().resolve(); dossier = get_dossier(ARM_KEY)
    if not (source / ".git").is_dir():
        raise ValueError("TTM source runtime must be a Git checkout")
    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if head != dossier["source_revision"]:
        raise ValueError("TTM source revision mismatch")
    dirty = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=all"], text=True,
    ).strip()
    if dirty:
        raise ValueError("TTM source runtime must be clean")
    origin = subprocess.check_output(["git", "-C", str(source), "remote", "get-url", "origin"], text=True).strip()
    if _normalize_origin(origin) != _normalize_origin(dossier["source_url"]):
        raise ValueError("TTM source origin mismatch")
    return source


def validate_snapshot(path: str | Path) -> Path:
    snapshot = Path(path).expanduser().resolve(); dossier = get_dossier(ARM_KEY)
    if snapshot.name != dossier["model_revision"] or not (snapshot / "config.json").is_file():
        raise ValueError("TTM snapshot identity mismatch")
    if not any(snapshot.glob("*.safetensors")):
        raise ValueError("TTM snapshot weights are missing")
    return snapshot


@dataclass(frozen=True)
class RouteConfig:
    route_key: str
    total_steps: int = 20
    batch_size: int | None = None
    learning_rate: float | None = None
    weight_decay: float | None = None
    max_gradient_norm: float | None = None
    seed: int = 20260718

    def resolved(self) -> dict[str, Any]:
        if self.route_key not in ROUTES:
            raise ValueError("unknown TTM route")
        policy = load_family_route_catalog()["constraint_profiles"][ROUTES[self.route_key]["profile"]][
            "optimization_hyperparameters"
        ]["value"]
        value = {
            "route_key": self.route_key, "total_steps": int(self.total_steps),
            "batch_size": int(self.batch_size or policy["batch_size_default"]),
            "learning_rate": float(policy["learning_rate_default"] if self.learning_rate is None else self.learning_rate),
            "weight_decay": float(policy["weight_decay_default"] if self.weight_decay is None else self.weight_decay),
            "max_gradient_norm": float(
                policy["max_gradient_norm_default"] if self.max_gradient_norm is None else self.max_gradient_norm
            ),
            "seed": int(self.seed),
        }
        if not policy["batch_size_min"] <= value["batch_size"] <= policy["batch_size_max"]:
            raise ValueError("TTM batch size is outside the catalog bound")
        for field in ("learning_rate", "weight_decay", "max_gradient_norm"):
            if not policy[field + "_min"] <= value[field] <= policy[field + "_max"]:
                raise ValueError(f"TTM {field} is outside the catalog bound")
        if not policy["smoke_steps"] <= value["total_steps"] <= policy["full_steps_max"]:
            raise ValueError("TTM total_steps is outside the catalog bound")
        return value


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
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(int(seed))


def load_route(
    route_key: str, *, model_snapshot: str | Path, source_runtime: str | Path, device: str,
) -> LoadedRoute:
    if route_key not in ROUTES:
        raise ValueError("unknown TTM route")
    source = validate_source_runtime(source_runtime); snapshot = validate_snapshot(model_snapshot)
    if str(source) not in sys.path: sys.path.insert(0, str(source))
    from tsfm_public.toolkit.get_model import get_model
    import tsfm_public
    imported = Path(inspect.getfile(tsfm_public)).resolve()
    if source not in imported.parents:
        raise RuntimeError("TTM imported outside the pinned source")
    selected = get_model(
        get_dossier(ARM_KEY)["model_id"], context_length=CONTEXT_LENGTH,
        prediction_length=HORIZON_LENGTH, freq_prefix_tuning=True,
        prefer_longer_context=True, force_return=None, return_model_key=True,
    )
    if selected != SELECTOR:
        raise RuntimeError("TTM selector drifted")
    model = get_model(
        str(snapshot), context_length=CONTEXT_LENGTH, prediction_length=HORIZON_LENGTH,
        model_revision=snapshot.name, num_input_channels=len(CHANNELS),
        enable_forecast_channel_mixing=False,
    ).to(device)
    expected = {
        "context_length": CONTEXT_LENGTH, "prediction_length": NATIVE_HORIZON,
        "prediction_filter_length": HORIZON_LENGTH, "resolution_prefix_tuning": True,
        "enable_forecast_channel_mixing": False, "num_input_channels": len(CHANNELS),
    }
    actual = {field: getattr(model.config, field, None) for field in expected}
    if actual != expected:
        raise RuntimeError(f"TTM loaded configuration drifted: {actual}")
    if ROUTES[route_key]["surface"] == "head_prefix":
        for name, parameter in model.named_parameters():
            parameter.requires_grad = (
                name.startswith("decoder.") or name.startswith("head.")
                or name.startswith("backbone.encoder.freq_mod.")
            )
    identity = {
        "model_id": get_dossier(ARM_KEY)["model_id"], "model_revision": snapshot.name,
        "source_revision": get_dossier(ARM_KEY)["source_revision"], "source_runtime": str(source),
        "selector": selected, "loaded_config": actual, "route_profile_sha256": _route_hash(route_key),
    }
    return LoadedRoute(route_key, model, identity, str(device))


def frequency_token(timeframe: str, *, batch: int, device: str) -> Any:
    if timeframe not in FREQUENCY_TOKENS:
        raise ValueError("unsupported TTM timeframe")
    return _torch().full((int(batch),), FREQUENCY_TOKENS[timeframe], dtype=_torch().long, device=device)


def split_parent(value: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    parent = np.asarray(value, dtype=np.float32)
    if parent.ndim != 3 or parent.shape[1:] != (CONTEXT_LENGTH + HORIZON_LENGTH, len(CHANNELS)):
        raise ValueError("TTM parent must have shape [B,528,5]")
    if np.isinf(parent).any():
        raise ValueError("TTM parent contains infinite values")
    context, target = parent[:, :CONTEXT_LENGTH], parent[:, CONTEXT_LENGTH:]
    context_mask = np.isfinite(context); target_mask = np.isfinite(target)
    return (
        np.where(context_mask, context, 0.0).astype(np.float32), context_mask.astype(np.float32),
        np.where(target_mask, target, 0.0).astype(np.float32), target_mask.astype(np.float32),
    )


def native_output(loaded: LoadedRoute, context: Any, *, timeframe: str) -> Any:
    torch = _torch(); values = np.asarray(context, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (CONTEXT_LENGTH, len(CHANNELS)):
        raise ValueError("TTM context must have shape [B,512,5]")
    if not np.isfinite(values).all():
        raise ValueError("TTM public forecast context must be finite")
    tensor = torch.as_tensor(values, dtype=torch.float32, device=loaded.device)
    return ttm_native_forecast(
        loaded.model, tensor, frequency_token(timeframe, batch=len(values), device=loaded.device)
    )


def native_loss(loaded: LoadedRoute, parent: Any, *, timeframe: str) -> Any:
    torch = _torch(); context, context_mask, target, target_mask = split_parent(parent)
    output = loaded.model(
        past_values=torch.as_tensor(context, dtype=torch.float32, device=loaded.device),
        future_values=torch.as_tensor(target, dtype=torch.float32, device=loaded.device),
        past_observed_mask=torch.as_tensor(context_mask, dtype=torch.float32, device=loaded.device),
        future_observed_mask=torch.as_tensor(target_mask, dtype=torch.float32, device=loaded.device),
        freq_token=frequency_token(timeframe, batch=len(context), device=loaded.device),
        return_loss=True,
    )
    if output.loss is None or not torch.isfinite(output.loss):
        raise RuntimeError("TTM native loss is missing or non-finite")
    return output.loss


def trainable_parameters(loaded: LoadedRoute) -> list[Any]:
    params = [parameter for parameter in loaded.model.parameters() if parameter.requires_grad]
    if not params: raise ValueError("TTM route has no trainable parameters")
    return params


def make_optimizer_for_parameters(parameters: Any, config: RouteConfig) -> Any:
    torch = _torch(); value = config.resolved(); params = list(parameters)
    if not params: raise ValueError("TTM route has no optimizer parameters")
    return torch.optim.AdamW(params, lr=value["learning_rate"], weight_decay=value["weight_decay"])


def make_optimizer(loaded: LoadedRoute, config: RouteConfig) -> Any:
    return make_optimizer_for_parameters(trainable_parameters(loaded), config)


def make_scheduler(optimizer: Any, config: RouteConfig) -> Any:
    from transformers import get_linear_schedule_with_warmup
    value = config.resolved()
    return get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=value["total_steps"])


def optimizer_step(
    loaded: LoadedRoute, optimizer: Any, scheduler: Any, parent: Any, *,
    timeframe: str, config: RouteConfig,
) -> dict[str, float]:
    torch = _torch(); value = config.resolved()
    if config.route_key != loaded.route_key: raise ValueError("TTM config route mismatch")
    loaded.model.train(); optimizer.zero_grad(set_to_none=True)
    parameter = trainable_parameters(loaded)[0]; before = parameter.detach().clone()
    loss = native_loss(loaded, parent, timeframe=timeframe); loss.backward()
    grads = [p.grad.detach().float().square().sum() for p in trainable_parameters(loaded) if p.grad is not None]
    grad_norm = torch.sqrt(torch.stack(grads).sum()) if grads else torch.tensor(0.0, device=loaded.device)
    torch.nn.utils.clip_grad_norm_(trainable_parameters(loaded), value["max_gradient_norm"])
    optimizer.step(); scheduler.step()
    return {
        "loss": float(loss.detach().cpu()), "grad_norm": float(grad_norm.detach().cpu()),
        "parameter_delta": float(torch.max(torch.abs(parameter.detach() - before)).cpu()),
    }


def capture_training_state(
    loaded: LoadedRoute, optimizer: Any, scheduler: Any, config: RouteConfig, *,
    global_step: int, sampler_cursor: int, history: list[Mapping[str, Any]],
    sampler_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return _capture_state(
        schema_version=CHECKPOINT_SCHEMA, route_key=loaded.route_key,
        route_profile_sha256=_route_hash(loaded.route_key), model_identity=loaded.identity,
        config=config.resolved(), modules=loaded.modules, optimizer=optimizer, scheduler=scheduler,
        global_step=global_step, sampler_cursor=sampler_cursor, history=list(history),
        sampler_state=sampler_state,
    )


def restore_training_state(
    state: Mapping[str, Any], loaded: LoadedRoute, optimizer: Any, scheduler: Any,
    config: RouteConfig,
) -> tuple[int, int, int, list[dict[str, Any]]]:
    return _restore_state(
        state, schema_version=CHECKPOINT_SCHEMA, route_key=loaded.route_key,
        route_profile_sha256=_route_hash(loaded.route_key), model_identity=loaded.identity,
        config=config.resolved(), modules=loaded.modules, optimizer=optimizer, scheduler=scheduler,
    )


def build_export_bundle(loaded: LoadedRoute) -> dict[str, Any]:
    return _build_export_bundle(
        schema_version=EXPORT_SCHEMA, route_key=loaded.route_key,
        route_profile_sha256=_route_hash(loaded.route_key), model_identity=loaded.identity,
        modules=loaded.modules,
        preprocessing={
            "context_length": CONTEXT_LENGTH, "channel_order": list(CHANNELS),
            "scaler": "native_std", "frequency_tokens": dict(FREQUENCY_TOKENS),
            "missing_policy": "native_observed_mask_training_complete_public_context",
        },
        output={
            "kind": "raw_forecast", "native_horizon": NATIVE_HORIZON,
            "deployment_filter": HORIZON_LENGTH, "shape": ["batch", HORIZON_LENGTH, len(CHANNELS)],
        },
        extra={"surface": loaded.spec["surface"], "selector": SELECTOR},
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


def save_artifact(path: str | Path, value: Any) -> dict[str, Any]: return atomic_torch_save(path, value)
def load_artifact(path: str | Path) -> Any: return load_torch(path)


__all__ = [
    "ARM_KEY", "CHANNELS", "CHECKPOINT_SCHEMA", "CONTEXT_LENGTH", "EXPORT_SCHEMA",
    "FREQUENCY_TOKENS", "HORIZON_LENGTH", "LoadedRoute", "NATIVE_HORIZON", "ROUTES",
    "RouteConfig", "SELECTOR", "build_export_bundle", "capture_training_state",
    "frequency_token", "load_artifact", "load_export_bundle", "load_route", "make_optimizer",
    "make_optimizer_for_parameters", "make_scheduler", "native_loss", "native_output",
    "optimizer_step", "restore_export_bundle", "restore_training_state", "save_artifact",
    "seed_everything", "split_parent", "trainable_parameters", "validate_snapshot",
    "validate_source_runtime",
]
