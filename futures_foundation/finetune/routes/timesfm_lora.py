"""Exact TimesFM 2.5 official LoRA forecast route."""
from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata
import inspect
from pathlib import Path
import random
import subprocess
from typing import Any, Mapping

import numpy as np

from ..native_adapters import timesfm25_transformers_forecast
from ..native_contracts import get_dossier
from ..native_family_route_catalog_v2 import load_family_route_catalog
from ..native_parity_runtime import validate_distribution_record
from ..native_training_schema_v2 import canonical_route_profile_sha256
from ._exact_state import (
    atomic_torch_save,
    build_export_bundle as _build_export_bundle,
    capture_state as _capture_state,
    load_torch,
    restore_export_bundle as _restore_export_bundle,
    restore_state as _restore_state,
)

ARM_KEY = "timesfm25"
TRACK = "F"
ROUTE_ID = "official_lora_forecast"
ROUTE_KEY = f"{ARM_KEY}:{TRACK}:{ROUTE_ID}"
PROFILE = "timesfm_lora"
CONTEXT_LENGTH = 512
HORIZON_LENGTH = 16
CHANNELS = ("open", "high", "low", "close", "volume")
CHECKPOINT_SCHEMA = "ffm_timesfm25_lora_training_state_v1"
EXPORT_SCHEMA = "ffm_timesfm25_lora_export_v1"
LORA_R = 4
LORA_ALPHA = 8
LORA_DROPOUT = 0.05


def _torch():
    import torch
    return torch


def _route_hash() -> str:
    return canonical_route_profile_sha256(ARM_KEY, TRACK, ROUTE_ID)


def _normalize_origin(value: str) -> str:
    return value.strip().lower().removesuffix(".git").replace("git@github.com:", "https://github.com/")


def validate_source_runtime(path: str | Path) -> Path:
    source = Path(path).expanduser().resolve(); dossier = get_dossier(ARM_KEY)
    if not (source / ".git").is_dir(): raise ValueError("TimesFM source must be a Git checkout")
    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if head != dossier["source_revision"]: raise ValueError("TimesFM source revision mismatch")
    dirty = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=all"], text=True,
    ).strip()
    if dirty: raise ValueError("TimesFM source must be clean")
    origin = subprocess.check_output(["git", "-C", str(source), "remote", "get-url", "origin"], text=True).strip()
    if _normalize_origin(origin) != _normalize_origin(dossier["source_url"]):
        raise ValueError("TimesFM source origin mismatch")
    return source


def validate_execution_source(path: str | Path) -> Path:
    root = Path(path).expanduser().resolve(); dossier = get_dossier(ARM_KEY)
    declared = dossier["native_parity"]["execution_source_distribution"]
    distribution = importlib.metadata.distribution(declared["name"])
    if Path(distribution._path).resolve() != root or distribution.version != declared["version"]:
        raise ValueError("TimesFM Transformers execution source drifted")
    validate_distribution_record(root)
    return root


def validate_snapshot(path: str | Path) -> Path:
    snapshot = Path(path).expanduser().resolve(); dossier = get_dossier(ARM_KEY)
    if snapshot.name != dossier["model_revision"] or not (snapshot / "config.json").is_file():
        raise ValueError("TimesFM snapshot identity mismatch")
    if not (snapshot / "model.safetensors").is_file(): raise ValueError("TimesFM weights are missing")
    return snapshot


@dataclass(frozen=True)
class RouteConfig:
    total_steps: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_gradient_norm: float = 1.0
    seed: int = 20260718

    def validate(self) -> None:
        policy = load_family_route_catalog()["constraint_profiles"][PROFILE]["optimization_hyperparameters"]["value"]
        values = {
            "batch_size": self.batch_size, "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay, "max_gradient_norm": self.max_gradient_norm,
        }
        for field, value in values.items():
            if not policy[field + "_min"] <= value <= policy[field + "_max"]:
                raise ValueError(f"TimesFM {field} is outside the catalog bound")
        if not policy["smoke_steps"] <= self.total_steps <= policy["full_steps_max"]:
            raise ValueError("TimesFM total_steps is outside the catalog bound")


class AdapterState:
    def __init__(self, model: Any): self.model = model
    def state_dict(self):
        from peft import get_peft_model_state_dict
        return get_peft_model_state_dict(self.model)
    def load_state_dict(self, state: Mapping[str, Any], strict: bool = True):
        from peft import set_peft_model_state_dict
        result = set_peft_model_state_dict(self.model, dict(state))
        if strict and getattr(result, "unexpected_keys", None):
            raise RuntimeError(f"unexpected TimesFM adapter keys: {result.unexpected_keys}")
        return result


@dataclass
class LoadedRoute:
    model: Any
    adapter_state: AdapterState
    identity: dict[str, Any]
    device: str

    @property
    def modules(self) -> dict[str, Any]: return {"adapter": self.adapter_state}


def seed_everything(seed: int) -> None:
    torch = _torch(); random.seed(int(seed)); np.random.seed(int(seed)); torch.manual_seed(int(seed))
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(int(seed))


def load_route(
    *, model_snapshot: str | Path, source_runtime: str | Path,
    execution_source: str | Path, device: str,
) -> LoadedRoute:
    torch = _torch(); source = validate_source_runtime(source_runtime)
    execution = validate_execution_source(execution_source); snapshot = validate_snapshot(model_snapshot)
    import transformers
    from transformers import TimesFm2_5ModelForPrediction
    imported = Path(inspect.getfile(transformers)).resolve()
    if execution.parent not in imported.parents and execution.parent != imported.parent:
        # The RECORD root and package directory are siblings under site-packages.
        if execution.parent != imported.parents[1]:
            raise RuntimeError("TimesFM did not import from the bound Transformers distribution")
    model = TimesFm2_5ModelForPrediction.from_pretrained(
        str(snapshot), dtype=torch.bfloat16, local_files_only=True,
    ).to(device)
    from peft import LoraConfig, get_peft_model
    model = get_peft_model(model, LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, target_modules="all-linear",
        lora_dropout=LORA_DROPOUT, bias="none",
    ))
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable or any("lora_" not in name for name in trainable):
        raise RuntimeError("TimesFM trainable surface is not LoRA-only")
    identity = {
        "model_id": get_dossier(ARM_KEY)["model_id"], "model_revision": snapshot.name,
        "source_revision": get_dossier(ARM_KEY)["source_revision"], "source_runtime": str(source),
        "execution_source": str(execution), "execution_version": "5.13.1",
        "route_profile_sha256": _route_hash(),
        "lora": {"r": LORA_R, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT, "target": "all-linear"},
    }
    return LoadedRoute(model, AdapterState(model), identity, str(device))


def split_parent(value: Any) -> tuple[Any, Any]:
    torch = _torch(); parent = torch.as_tensor(value, dtype=torch.float32)
    if parent.ndim != 3 or tuple(parent.shape[1:]) != (CONTEXT_LENGTH + HORIZON_LENGTH, len(CHANNELS)):
        raise ValueError("TimesFM parent must have shape [B,528,5]")
    if not torch.isfinite(parent).all():
        raise ValueError("TimesFM parent must be finite; the pinned Transformers surface has no stable missing-value path")
    context = parent[:, :CONTEXT_LENGTH].permute(0, 2, 1).reshape(-1, CONTEXT_LENGTH)
    target = parent[:, CONTEXT_LENGTH:].permute(0, 2, 1).reshape(-1, HORIZON_LENGTH)
    return context, target


def native_loss(loaded: LoadedRoute, parent: Any) -> Any:
    context, target = split_parent(parent); context = context.to(loaded.device); target = target.to(loaded.device)
    output = loaded.model(
        past_values=context, future_values=target, forecast_context_len=CONTEXT_LENGTH,
        truncate_negative=False, force_flip_invariance=True,
    )
    if output.loss is None or not _torch().isfinite(output.loss):
        raise RuntimeError("TimesFM native loss is missing or non-finite")
    return output.loss


def public_output(loaded: LoadedRoute, context: Any) -> tuple[Any, Any]:
    torch = _torch(); values = torch.as_tensor(context, dtype=torch.float32)
    if values.ndim != 3 or tuple(values.shape[1:]) != (CONTEXT_LENGTH, len(CHANNELS)):
        raise ValueError("TimesFM context must have shape [B,512,5]")
    if not torch.isfinite(values).all():
        raise ValueError("TimesFM context must be finite; the pinned Transformers surface has no stable missing-value path")
    points = []
    quantiles = []
    for row in values:
        channels = row.transpose(0, 1).to(loaded.device)
        point, raw_quantiles = timesfm25_transformers_forecast(
            loaded.model,
            channels,
            prediction_length=HORIZON_LENGTH,
            context_length=CONTEXT_LENGTH,
        )
        points.append(point)
        quantiles.append(raw_quantiles)
    return torch.cat(points, dim=0), torch.cat(quantiles, dim=0)


def trainable_parameters(loaded: LoadedRoute) -> list[Any]:
    values = [parameter for parameter in loaded.model.parameters() if parameter.requires_grad]
    if not values: raise ValueError("TimesFM route has no trainable adapter parameters")
    return values


def make_optimizer_for_parameters(parameters: Any, config: RouteConfig) -> Any:
    config.validate(); params = list(parameters)
    if not params: raise ValueError("TimesFM route has no optimizer parameters")
    return _torch().optim.AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)


def make_optimizer(loaded: LoadedRoute, config: RouteConfig) -> Any:
    return make_optimizer_for_parameters(trainable_parameters(loaded), config)


def make_scheduler(optimizer: Any, config: RouteConfig) -> Any:
    config.validate()
    return _torch().optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.total_steps)


def optimizer_step(
    loaded: LoadedRoute, optimizer: Any, scheduler: Any, parent: Any, *, config: RouteConfig,
) -> dict[str, float]:
    torch = _torch(); config.validate(); loaded.model.train(); optimizer.zero_grad(set_to_none=True)
    parameter = trainable_parameters(loaded)[0]; before = parameter.detach().clone()
    with torch.autocast(device_type=Path(loaded.device).name if False else ("cuda" if str(loaded.device).startswith("cuda") else "cpu"), dtype=torch.bfloat16, enabled=str(loaded.device).startswith("cuda")):
        loss = native_loss(loaded, parent)
    loss.backward()
    grads = [p.grad.detach().float().square().sum() for p in trainable_parameters(loaded) if p.grad is not None]
    grad_norm = torch.sqrt(torch.stack(grads).sum()) if grads else torch.tensor(0.0, device=loaded.device)
    torch.nn.utils.clip_grad_norm_(trainable_parameters(loaded), config.max_gradient_norm)
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
    config.validate()
    return _capture_state(
        schema_version=CHECKPOINT_SCHEMA, route_key=ROUTE_KEY,
        route_profile_sha256=_route_hash(), model_identity=loaded.identity,
        config={
            "total_steps": config.total_steps, "batch_size": config.batch_size,
            "learning_rate": config.learning_rate, "weight_decay": config.weight_decay,
            "max_gradient_norm": config.max_gradient_norm, "seed": config.seed,
        },
        modules=loaded.modules, optimizer=optimizer, scheduler=scheduler,
        global_step=global_step, sampler_cursor=sampler_cursor, history=list(history),
        sampler_state=sampler_state,
    )


def restore_training_state(
    state: Mapping[str, Any], loaded: LoadedRoute, optimizer: Any, scheduler: Any,
    config: RouteConfig,
) -> tuple[int, int, int, list[dict[str, Any]]]:
    return _restore_state(
        state, schema_version=CHECKPOINT_SCHEMA, route_key=ROUTE_KEY,
        route_profile_sha256=_route_hash(), model_identity=loaded.identity,
        config={
            "total_steps": config.total_steps, "batch_size": config.batch_size,
            "learning_rate": config.learning_rate, "weight_decay": config.weight_decay,
            "max_gradient_norm": config.max_gradient_norm, "seed": config.seed,
        },
        modules=loaded.modules, optimizer=optimizer, scheduler=scheduler,
    )


def build_export_bundle(loaded: LoadedRoute) -> dict[str, Any]:
    return _build_export_bundle(
        schema_version=EXPORT_SCHEMA, route_key=ROUTE_KEY,
        route_profile_sha256=_route_hash(), model_identity=loaded.identity,
        modules=loaded.modules,
        preprocessing={
            "context_length": CONTEXT_LENGTH, "channel_order": list(CHANNELS),
            "layout": "independent_univariate_passes", "normalization": "timesfm_internal_revin",
            "missing_policy": "reject_nonfinite_context_and_target",
            "force_flip_invariance": True, "truncate_negative": False,
            "fix_quantile_crossing": False,
        },
        output={
            "kind": "point_and_raw_quantiles", "horizon": HORIZON_LENGTH,
            "point_shape": ["batch_times_channel", HORIZON_LENGTH],
        },
        extra={"base_model_required": True, "adapter_only": True},
    )


def restore_export_bundle(bundle: Mapping[str, Any], loaded: LoadedRoute) -> dict[str, Any]:
    return _restore_export_bundle(
        bundle, schema_version=EXPORT_SCHEMA, route_key=ROUTE_KEY,
        route_profile_sha256=_route_hash(), model_identity=loaded.identity,
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
    "HORIZON_LENGTH", "LORA_ALPHA", "LORA_DROPOUT", "LORA_R", "LoadedRoute",
    "PROFILE", "ROUTE_ID", "ROUTE_KEY", "RouteConfig", "AdapterState",
    "build_export_bundle", "capture_training_state", "load_artifact", "load_export_bundle", "load_route",
    "make_optimizer", "make_optimizer_for_parameters", "make_scheduler", "native_loss",
    "optimizer_step", "public_output", "restore_export_bundle", "restore_training_state",
    "save_artifact", "seed_everything", "split_parent", "trainable_parameters",
    "validate_execution_source", "validate_snapshot", "validate_source_runtime",
]
