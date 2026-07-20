"""Exact Chronos-2 full and LoRA native quantile routes.

The public ``fit`` helper saves model weights only.  This module uses the same native
Chronos2Model loss and LoRA surface step-by-step so optimizer, scheduler, sampler, and
all RNG states are exactly resumable.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np

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

ARM_KEY = "chronos_v2"
CONTEXT_LENGTH = 512
HORIZON_LENGTH = 16
NATIVE_HORIZON = 64
CHANNELS = ("open", "high", "low", "close", "volume")
QUANTILES = (0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
CHECKPOINT_SCHEMA = "ffm_chronos2_native_training_state_v1"
EXPORT_SCHEMA = "ffm_chronos2_native_export_v1"
LORA_R = 8
LORA_ALPHA = 16
LORA_TARGET = ("q", "k", "v", "o")
LORA_MODULES_TO_SAVE = ("input_patch_embedding", "output_patch_embedding")
ROUTES = {
    "chronos_v2:F:official_fit_full": {
        "track": "F", "route_id": "official_fit_full", "profile": "chronos2_full",
        "surface": "full",
    },
    "chronos_v2:F:official_fit_lora": {
        "track": "F", "route_id": "official_fit_lora", "profile": "chronos2_lora",
        "surface": "lora",
    },
}


def _torch():
    import torch
    return torch


def _route_hash(route_key: str) -> str:
    spec = ROUTES[route_key]
    return canonical_route_profile_sha256(ARM_KEY, spec["track"], spec["route_id"])


def validate_snapshot(path: str | Path) -> Path:
    snapshot = Path(path).expanduser().resolve(); dossier = get_dossier(ARM_KEY)
    if snapshot.name != dossier["model_revision"] or not (snapshot / "config.json").is_file():
        raise ValueError("Chronos-2 snapshot identity mismatch")
    if not any(snapshot.glob("*.safetensors")):
        raise ValueError("Chronos-2 snapshot weights are missing")
    return snapshot


@dataclass(frozen=True)
class RouteConfig:
    route_key: str
    total_steps: int = 20
    batch_size: int = 256
    learning_rate: float | None = None
    weight_decay: float = 0.0
    max_gradient_norm: float = 1.0
    seed: int = 20260718

    def resolved(self) -> dict[str, Any]:
        if self.route_key not in ROUTES:
            raise ValueError("unknown Chronos-2 route")
        policy = load_family_route_catalog()["constraint_profiles"][ROUTES[self.route_key]["profile"]][
            "optimization_hyperparameters"
        ]["value"]
        value = {
            "route_key": self.route_key,
            "total_steps": int(self.total_steps),
            "batch_size": int(self.batch_size),
            "learning_rate": float(
                policy["learning_rate_default"] if self.learning_rate is None else self.learning_rate
            ),
            "weight_decay": float(self.weight_decay),
            "max_gradient_norm": float(self.max_gradient_norm),
            "seed": int(self.seed),
        }
        for field in ("batch_size", "learning_rate", "weight_decay", "max_gradient_norm"):
            if not policy[field + "_min"] <= value[field] <= policy[field + "_max"]:
                raise ValueError(f"Chronos-2 {field} is outside the catalog bound")
        if not policy["smoke_steps"] <= value["total_steps"] <= policy["full_steps_max"]:
            raise ValueError("Chronos-2 total_steps is outside the catalog bound")
        return value


class AdapterState:
    def __init__(self, model: Any): self.model = model
    def state_dict(self):
        from peft import get_peft_model_state_dict
        return get_peft_model_state_dict(self.model)
    def load_state_dict(self, state: Mapping[str, Any], strict: bool = True):
        from peft import set_peft_model_state_dict
        result = set_peft_model_state_dict(self.model, dict(state))
        if strict and getattr(result, "unexpected_keys", None):
            raise RuntimeError(f"unexpected Chronos-2 adapter keys: {result.unexpected_keys}")
        return result


@dataclass
class LoadedRoute:
    route_key: str
    pipeline: Any
    model: Any
    state_module: Any
    identity: dict[str, Any]
    device: str

    @property
    def spec(self) -> dict[str, Any]: return ROUTES[self.route_key]
    @property
    def modules(self) -> dict[str, Any]:
        return {"adapter" if self.spec["surface"] == "lora" else "model": self.state_module}


def seed_everything(seed: int) -> None:
    torch = _torch(); random.seed(int(seed)); np.random.seed(int(seed)); torch.manual_seed(int(seed))
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(int(seed))


def load_route(route_key: str, *, model_snapshot: str | Path, device: str) -> LoadedRoute:
    if route_key not in ROUTES: raise ValueError("unknown Chronos-2 route")
    torch = _torch(); snapshot = validate_snapshot(model_snapshot)
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("Chronos-2 exact training requires CUDA BF16/TF32")
    major, _ = torch.cuda.get_device_capability(torch.device(device))
    if major < 8:
        raise RuntimeError("Chronos-2 exact training requires compute capability >= 8")
    from chronos import Chronos2Pipeline
    pipeline = Chronos2Pipeline.from_pretrained(
        str(snapshot), device_map=device, dtype=torch.bfloat16,
    )
    if tuple(float(value) for value in pipeline.quantiles) != QUANTILES:
        raise RuntimeError("Chronos-2 native quantile grid drifted")
    model = pipeline.model
    if bool(getattr(model, "supports_gradient_checkpointing", False)):
        model.gradient_checkpointing_enable()
    surface = ROUTES[route_key]["surface"]
    state_module: Any = model
    if surface == "lora":
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(
            r=LORA_R, lora_alpha=LORA_ALPHA, target_modules=list(LORA_TARGET),
            modules_to_save=list(LORA_MODULES_TO_SAVE),
        ))
        pipeline.model = model
        trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        if not trainable or any(
            "lora_" not in name and "modules_to_save" not in name for name in trainable
        ):
            raise RuntimeError("Chronos-2 LoRA trainable surface drifted")
        state_module = AdapterState(model)
    else:
        if not all(parameter.requires_grad for parameter in model.parameters()):
            raise RuntimeError("Chronos-2 full route has frozen parameters")
    torch.backends.cuda.matmul.allow_tf32 = True
    identity = {
        "model_id": get_dossier(ARM_KEY)["model_id"],
        "model_revision": snapshot.name,
        "source_revision": get_dossier(ARM_KEY)["source_revision"],
        "package": get_dossier(ARM_KEY)["package"],
        "route_profile_sha256": _route_hash(route_key),
        "compute": "bf16_tf32_sm80",
        "gradient_checkpointing": bool(getattr(model, "supports_gradient_checkpointing", False)),
        "surface": surface,
        "lora": (
            None if surface == "full" else {
                "r": LORA_R, "alpha": LORA_ALPHA, "target": list(LORA_TARGET),
                "modules_to_save": list(LORA_MODULES_TO_SAVE),
            }
        ),
    }
    return LoadedRoute(route_key, pipeline, model, state_module, identity, str(device))


def split_parent(value: Any) -> tuple[Any, Any, Any, Any]:
    torch = _torch(); parent = torch.as_tensor(value, dtype=torch.float32)
    if parent.ndim != 3 or tuple(parent.shape[1:]) != (
        CONTEXT_LENGTH + HORIZON_LENGTH, len(CHANNELS)
    ):
        raise ValueError("Chronos-2 parent must have shape [B,528,5]")
    if torch.isinf(parent).any(): raise ValueError("Chronos-2 parent contains infinite values")
    batch = parent.shape[0]
    context = parent[:, :CONTEXT_LENGTH].permute(0, 2, 1).reshape(batch * len(CHANNELS), CONTEXT_LENGTH)
    future = parent[:, CONTEXT_LENGTH:].permute(0, 2, 1).reshape(batch * len(CHANNELS), HORIZON_LENGTH)
    group_ids = torch.arange(batch).repeat_interleave(len(CHANNELS))
    future_covariates = torch.full_like(future, torch.nan)
    return context, future, future_covariates, group_ids


def native_model_output(loaded: LoadedRoute, parent: Any, *, with_target: bool) -> Any:
    context, future, future_covariates, group_ids = split_parent(parent)
    return loaded.model(
        context=context.to(loaded.device),
        future_target=future.to(loaded.device) if with_target else None,
        future_covariates=future_covariates.to(loaded.device),
        group_ids=group_ids.to(loaded.device),
    )


def native_loss(loaded: LoadedRoute, parent: Any) -> Any:
    output = native_model_output(loaded, parent, with_target=True)
    if output.loss is None or not _torch().isfinite(output.loss):
        raise RuntimeError("Chronos-2 native loss is missing or non-finite")
    return output.loss


def native_quantiles(loaded: LoadedRoute, context: Any) -> Any:
    torch = _torch(); values = torch.as_tensor(context, dtype=torch.float32)
    if values.ndim != 3 or tuple(values.shape[1:]) != (CONTEXT_LENGTH, len(CHANNELS)):
        raise ValueError("Chronos-2 context must have shape [B,512,5]")
    if torch.isinf(values).any(): raise ValueError("Chronos-2 context contains infinite values")
    rows = []
    for parent in values:
        flat = parent.transpose(0, 1)
        group_ids = torch.zeros(len(CHANNELS), dtype=torch.long)
        covariates = torch.full(
            (len(CHANNELS), HORIZON_LENGTH), torch.nan, dtype=torch.float32,
        )
        output = loaded.model(
            context=flat.to(loaded.device), future_target=None,
            future_covariates=covariates.to(loaded.device),
            group_ids=group_ids.to(loaded.device),
        ).quantile_preds[..., :HORIZON_LENGTH]
        if output.shape[:2] != (len(CHANNELS), len(QUANTILES)):
            raise RuntimeError("Chronos-2 quantile output geometry drifted")
        rows.append(output)
    return torch.stack(rows, dim=0)


def trainable_parameters(loaded: LoadedRoute) -> list[Any]:
    values = [parameter for parameter in loaded.model.parameters() if parameter.requires_grad]
    if not values: raise ValueError("Chronos-2 route has no trainable parameters")
    return values


def make_optimizer_for_parameters(parameters: Any, config: RouteConfig) -> Any:
    torch = _torch(); value = config.resolved(); params = list(parameters)
    if not params: raise ValueError("Chronos-2 route has no optimizer parameters")
    fused = all(parameter.is_cuda for parameter in params)
    return torch.optim.AdamW(
        params, lr=value["learning_rate"], weight_decay=value["weight_decay"], fused=fused,
    )


def make_optimizer(loaded: LoadedRoute, config: RouteConfig) -> Any:
    if config.route_key != loaded.route_key: raise ValueError("Chronos-2 config route mismatch")
    return make_optimizer_for_parameters(trainable_parameters(loaded), config)


def make_scheduler(optimizer: Any, config: RouteConfig) -> Any:
    from transformers import get_linear_schedule_with_warmup
    value = config.resolved()
    return get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=value["total_steps"])


def optimizer_step(
    loaded: LoadedRoute, optimizer: Any, scheduler: Any, parent: Any, *, config: RouteConfig,
) -> dict[str, float]:
    torch = _torch(); value = config.resolved()
    if config.route_key != loaded.route_key: raise ValueError("Chronos-2 config route mismatch")
    loaded.model.train(); optimizer.zero_grad(set_to_none=True)
    parameter = trainable_parameters(loaded)[0]; before = parameter.detach().clone()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = native_loss(loaded, parent)
    loss.backward()
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
            "context_length": CONTEXT_LENGTH, "prediction_length": HORIZON_LENGTH,
            "channel_order": list(CHANNELS), "layout": "grouped_multivariate",
            "group_ids": "same_parent_five_variates", "scaler": "chronos2_native",
            "missing_policy": "native_nan_mask",
        },
        output={
            "kind": "native_quantiles", "quantile_levels": list(QUANTILES),
            "shape": ["batch", len(CHANNELS), len(QUANTILES), HORIZON_LENGTH],
        },
        extra={"surface": loaded.spec["surface"], "base_model_required": loaded.spec["surface"] == "lora"},
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
    "HORIZON_LENGTH", "LORA_ALPHA", "LORA_MODULES_TO_SAVE", "LORA_R", "LORA_TARGET", "LoadedRoute",
    "NATIVE_HORIZON", "QUANTILES", "ROUTES", "RouteConfig", "AdapterState",
    "build_export_bundle", "capture_training_state", "load_artifact", "load_export_bundle", "load_route",
    "make_optimizer", "make_optimizer_for_parameters", "make_scheduler", "native_loss",
    "native_model_output", "native_quantiles", "optimizer_step", "restore_export_bundle",
    "restore_training_state", "save_artifact", "seed_everything", "split_parent",
    "trainable_parameters", "validate_snapshot",
]
