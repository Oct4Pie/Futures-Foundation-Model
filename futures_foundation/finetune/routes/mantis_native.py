"""Exact architecture-native Mantis V1/V2 classification and contrastive routes.

The pinned upstream trainer owns model, augmentation, loss, and optimizer semantics.  This
module executes those semantics step-by-step so sampler/RNG/optimizer/scheduler state and
export parity are exact.  Importing this module grants no training or deployment admission.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import random
import subprocess
import sys
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

CONTEXT_LENGTH = 512
CHANNELS = ("open", "high", "low", "close", "volume")
CHECKPOINT_SCHEMA = "ffm_mantis_native_training_state_v1"
EXPORT_SCHEMA = "ffm_mantis_native_export_v1"
SOURCE_ORIGIN = "https://github.com/vcerqueira/mantis"

ROUTES = {
    "mantis_v1:C:supervised_classification_full": {
        "arm": "mantis_v1", "version": 1, "track": "C",
        "route_id": "supervised_classification_full", "profile": "mantis_v1_classifier_full",
        "task": "classification", "surface": "full",
    },
    "mantis_v1:C:supervised_classification_head": {
        "arm": "mantis_v1", "version": 1, "track": "C",
        "route_id": "supervised_classification_head", "profile": "mantis_v1_classifier_head",
        "task": "classification", "surface": "head",
    },
    "mantis_v1:R:official_crop_resize_contrastive": {
        "arm": "mantis_v1", "version": 1, "track": "R",
        "route_id": "official_crop_resize_contrastive", "profile": "mantis_v1_contrastive",
        "task": "contrastive", "surface": "full",
    },
    "mantis_v2:C:supervised_classification_full": {
        "arm": "mantis_v2", "version": 2, "track": "C",
        "route_id": "supervised_classification_full", "profile": "mantis_v2_classifier_full",
        "task": "classification", "surface": "full",
    },
    "mantis_v2:C:supervised_classification_head": {
        "arm": "mantis_v2", "version": 2, "track": "C",
        "route_id": "supervised_classification_head", "profile": "mantis_v2_classifier_head",
        "task": "classification", "surface": "head",
    },
    "mantis_v2:R:official_crop_resize_contrastive": {
        "arm": "mantis_v2", "version": 2, "track": "R",
        "route_id": "official_crop_resize_contrastive", "profile": "mantis_v2_contrastive",
        "task": "contrastive", "surface": "full",
    },
}


def _torch():
    import torch
    return torch


def _normalize_origin(value: str) -> str:
    return value.strip().lower().removesuffix(".git").replace(
        "git@github.com:", "https://github.com/"
    )


def validate_source_runtime(path: str | Path, *, arm_key: str) -> Path:
    source = Path(path).expanduser().resolve()
    dossier = get_dossier(arm_key)
    if not (source / ".git").is_dir():
        raise ValueError("Mantis source runtime must be a Git checkout")
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True,
    ).strip()
    if revision != dossier["source_revision"]:
        raise ValueError("Mantis source revision mismatch")
    dirty = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=all"],
        text=True,
    ).strip()
    if dirty:
        raise ValueError("Mantis source runtime must be clean")
    origin = subprocess.check_output(
        ["git", "-C", str(source), "remote", "get-url", "origin"], text=True,
    ).strip()
    if _normalize_origin(origin) != _normalize_origin(dossier["source_url"]):
        raise ValueError("Mantis source origin mismatch")
    return source


def validate_snapshot(path: str | Path, *, arm_key: str) -> Path:
    snapshot = Path(path).expanduser().resolve()
    revision = get_dossier(arm_key)["model_revision"]
    if snapshot.name != revision or not snapshot.is_dir() or not (snapshot / "config.json").is_file():
        raise ValueError("Mantis model snapshot identity mismatch")
    return snapshot


@dataclass(frozen=True)
class RouteConfig:
    route_key: str
    total_steps: int = 20
    n_classes: int = 3
    batch_size: int | None = None
    learning_rate: float | None = None
    weight_decay: float | None = None
    max_gradient_norm: float | None = None
    warmup_fraction: float | None = None
    seed: int = 20260718

    def resolved(self) -> dict[str, Any]:
        if self.route_key not in ROUTES:
            raise ValueError("unknown Mantis route key")
        profile = load_family_route_catalog()["constraint_profiles"][
            ROUTES[self.route_key]["profile"]
        ]["optimization_hyperparameters"]["value"]
        value = {
            "route_key": self.route_key,
            "total_steps": int(self.total_steps),
            "n_classes": int(self.n_classes),
            "batch_size": int(self.batch_size or profile["batch_size_default"]),
            "learning_rate": float(
                profile["learning_rate_default"] if self.learning_rate is None else self.learning_rate
            ),
            "weight_decay": float(
                profile["weight_decay_default"] if self.weight_decay is None else self.weight_decay
            ),
            "max_gradient_norm": float(
                profile["max_gradient_norm_default"]
                if self.max_gradient_norm is None else self.max_gradient_norm
            ),
            "warmup_fraction": float(
                profile["warmup_fraction_default"]
                if self.warmup_fraction is None else self.warmup_fraction
            ),
            "seed": int(self.seed),
        }
        if value["n_classes"] < 2 or not profile["batch_size_min"] <= value["batch_size"] <= profile["batch_size_max"]:
            raise ValueError("Mantis class count or batch size is outside the catalog bound")
        for name in ("learning_rate", "weight_decay", "max_gradient_norm", "warmup_fraction"):
            low = profile[name + "_min"]
            high = profile[name + "_max"]
            if not low <= value[name] <= high:
                raise ValueError(f"Mantis {name} is outside the catalog bound")
        if not profile["smoke_steps"] <= value["total_steps"] <= profile["full_steps_max"]:
            raise ValueError("Mantis total_steps is outside the catalog bound")
        return value


@dataclass
class LoadedRoute:
    route_key: str
    backbone: Any
    head: Any | None
    identity: dict[str, Any]

    @property
    def spec(self) -> dict[str, Any]:
        return ROUTES[self.route_key]

    @property
    def modules(self) -> dict[str, Any]:
        result = {"backbone": self.backbone}
        if self.head is not None:
            result["head"] = self.head
        return result


def seed_everything(seed: int) -> None:
    torch = _torch()
    random.seed(int(seed)); np.random.seed(int(seed)); torch.manual_seed(int(seed))
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _route_hash(route_key: str) -> str:
    spec = ROUTES[route_key]
    return canonical_route_profile_sha256(spec["arm"], spec["track"], spec["route_id"])


def load_route(
    route_key: str, *, model_snapshot: str | Path, source_runtime: str | Path,
    device: str, n_classes: int = 3,
) -> LoadedRoute:
    if route_key not in ROUTES:
        raise ValueError("unknown Mantis route key")
    torch = _torch(); spec = ROUTES[route_key]
    source = validate_source_runtime(source_runtime, arm_key=spec["arm"])
    snapshot = validate_snapshot(model_snapshot, arm_key=spec["arm"])
    if str(source / "src") not in sys.path:
        sys.path.insert(0, str(source / "src"))
    from mantis.architecture import MantisV1, MantisV2
    pretraining = spec["task"] == "contrastive"
    if spec["version"] == 1:
        backbone = MantisV1(device=device, pre_training=pretraining).from_pretrained(str(snapshot))
    else:
        backbone = MantisV2(
            device=device, pre_training=pretraining,
            return_transf_layer=-1, output_token="cls_token",
        ).from_pretrained(str(snapshot))
    backbone.to(device)
    head = None
    if spec["task"] == "classification":
        head = torch.nn.Linear(int(backbone.hidden_dim) * len(CHANNELS), int(n_classes)).to(device)
        freeze = spec["surface"] == "head"
        for parameter in backbone.parameters():
            parameter.requires_grad = not freeze
    identity = {
        "arm_key": spec["arm"],
        "model_id": get_dossier(spec["arm"])["model_id"],
        "model_revision": snapshot.name,
        "source_revision": get_dossier(spec["arm"])["source_revision"],
        "source_runtime": str(source),
        "route_profile_sha256": _route_hash(route_key),
    }
    return LoadedRoute(route_key=route_key, backbone=backbone, head=head, identity=identity)


def parent_tensor(parent: Any, *, device: str) -> Any:
    torch = _torch(); value = torch.as_tensor(parent, dtype=torch.float32, device=device)
    if value.ndim != 3 or tuple(value.shape[1:]) != (CONTEXT_LENGTH, len(CHANNELS)):
        raise ValueError("Mantis parent must have shape [batch,512,5]")
    if not torch.isfinite(value).all():
        raise ValueError("Mantis parent must be finite")
    return value


def channel_embeddings(loaded: LoadedRoute, parent: Any, *, device: str) -> Any:
    torch = _torch(); value = parent_tensor(parent, device=device)
    outputs = []
    for row in value:
        channels = row.transpose(0, 1).unsqueeze(1)
        embedded = loaded.backbone(channels)
        if embedded.ndim != 2 or embedded.shape[0] != len(CHANNELS):
            raise ValueError("Mantis backbone output geometry mismatch")
        outputs.append(embedded)
    return torch.stack(outputs, dim=0)


def classification_logits(loaded: LoadedRoute, parent: Any, *, device: str) -> Any:
    if loaded.spec["task"] != "classification" or loaded.head is None:
        raise ValueError("Mantis route is not a classifier")
    embedded = channel_embeddings(loaded, parent, device=device)
    return loaded.head(embedded.reshape(embedded.shape[0], -1))


def _augment_views(value: Any, *, source_runtime: str) -> tuple[Any, Any]:
    source = Path(source_runtime)
    if str(source / "src") not in sys.path:
        sys.path.insert(0, str(source / "src"))
    from mantis.trainer.trainer_utils.augmentation import RandomCropResize
    augmentation_1 = RandomCropResize(crop_rate_range=[0, 0.2], size=CONTEXT_LENGTH)
    augmentation_2 = RandomCropResize(crop_rate_range=[0, 0.2], size=CONTEXT_LENGTH)
    return augmentation_1(value), augmentation_2(value)


def contrastive_loss(loaded: LoadedRoute, parent: Any, *, device: str) -> Any:
    torch = _torch(); value = parent_tensor(parent, device=device)
    batch = value.shape[0]
    if batch < 2:
        raise ValueError("Mantis contrastive route requires at least two parent windows")
    flattened = value.permute(0, 2, 1).reshape(batch * len(CHANNELS), 1, CONTEXT_LENGTH)
    first, second = _augment_views(flattened, source_runtime=loaded.identity["source_runtime"])
    q = loaded.backbone(first).reshape(batch, len(CHANNELS), -1)
    k = loaded.backbone(second).reshape(batch, len(CHANNELS), -1)
    labels = torch.arange(batch, device=value.device)
    losses = []
    for channel in range(len(CHANNELS)):
        query = torch.nn.functional.normalize(q[:, channel], dim=1)
        key = torch.nn.functional.normalize(k[:, channel], dim=1)
        logits = query @ key.T / 0.1
        losses.append(torch.nn.functional.cross_entropy(logits, labels))
    return torch.stack(losses).mean()


def native_loss(
    loaded: LoadedRoute, parent: Any, *, device: str, labels: Any | None = None,
) -> Any:
    if loaded.spec["task"] == "contrastive":
        return contrastive_loss(loaded, parent, device=device)
    torch = _torch()
    if labels is None:
        raise ValueError("Mantis classification labels are required")
    target = torch.as_tensor(labels, dtype=torch.long, device=device)
    logits = classification_logits(loaded, parent, device=device)
    if target.shape != (logits.shape[0],):
        raise ValueError("Mantis labels are not aligned")
    return torch.nn.functional.cross_entropy(logits, target)


def trainable_parameters(loaded: LoadedRoute) -> list[Any]:
    parameters = [p for module in loaded.modules.values() for p in module.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("Mantis route has no trainable parameters")
    return parameters


def make_optimizer_for_parameters(parameters: Any, config: RouteConfig) -> Any:
    value = config.resolved(); torch = _torch()
    parameters = list(parameters)
    if not parameters:
        raise ValueError("Mantis route has no optimizer parameters")
    return torch.optim.AdamW(
        parameters, lr=value["learning_rate"],
        betas=(0.9, 0.999), weight_decay=value["weight_decay"],
    )


def make_optimizer(loaded: LoadedRoute, config: RouteConfig) -> Any:
    return make_optimizer_for_parameters(trainable_parameters(loaded), config)


def make_scheduler(optimizer: Any, config: RouteConfig) -> Any:
    value = config.resolved(); torch = _torch()
    total = value["total_steps"]
    warmup = max(1, int(round(total * value["warmup_fraction"])))
    def schedule(step: int) -> float:
        if step < warmup:
            return float(step + 1) / float(warmup)
        progress = min(max((step - warmup + 1) / max(total - warmup, 1), 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def optimizer_step(
    loaded: LoadedRoute, optimizer: Any, scheduler: Any, parent: Any, *, device: str,
    config: RouteConfig, labels: Any | None = None,
) -> dict[str, float]:
    torch = _torch(); resolved = config.resolved()
    for module in loaded.modules.values():
        module.train()
    optimizer.zero_grad(set_to_none=True)
    parameter = trainable_parameters(loaded)[0]; before = parameter.detach().clone()
    loss = native_loss(loaded, parent, device=device, labels=labels)
    loss.backward()
    grads = [p.grad.detach().float().square().sum() for p in trainable_parameters(loaded) if p.grad is not None]
    grad_norm = torch.sqrt(torch.stack(grads).sum()) if grads else torch.tensor(0.0, device=device)
    if resolved["max_gradient_norm"] > 0:
        torch.nn.utils.clip_grad_norm_(trainable_parameters(loaded), resolved["max_gradient_norm"])
    optimizer.step(); scheduler.step()
    return {
        "loss": float(loss.detach().cpu()),
        "grad_norm": float(grad_norm.detach().cpu()),
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


def deployment_output(loaded: LoadedRoute, parent: Any, *, device: str) -> Any:
    if loaded.spec["task"] == "classification":
        for module in loaded.modules.values():
            module.eval()
        return classification_logits(loaded, parent, device=device)
    backbone = loaded.backbone
    old = {
        "pre_training": getattr(backbone, "pre_training", None),
        "return_transf_layer": getattr(backbone, "return_transf_layer", None),
        "output_token": getattr(backbone, "output_token", None),
    }
    backbone.pre_training = False
    if loaded.spec["version"] == 2:
        backbone.return_transf_layer = 2; backbone.output_token = "combined"
    try:
        backbone.eval()
        return channel_embeddings(loaded, parent, device=device)
    finally:
        for name, value in old.items():
            if value is not None:
                setattr(backbone, name, value)


def build_export_bundle(loaded: LoadedRoute, config: RouteConfig) -> dict[str, Any]:
    spec = loaded.spec
    output = (
        {"kind": "classification_logits", "classes": config.resolved()["n_classes"]}
        if spec["task"] == "classification" else
        {
            "kind": "official_representation",
            "version": spec["version"],
            "reduction": "final_cls" if spec["version"] == 1 else "layer2_cls_mean",
            "per_channel": True,
        }
    )
    return _build_export_bundle(
        schema_version=EXPORT_SCHEMA, route_key=loaded.route_key,
        route_profile_sha256=_route_hash(loaded.route_key), model_identity=loaded.identity,
        modules=loaded.modules,
        preprocessing={
            "input_shape": ["batch", CONTEXT_LENGTH, len(CHANNELS)],
            "channel_order": list(CHANNELS), "scaler": "raw",
            "missing_policy": "reject",
        },
        output=output,
        extra={"task": spec["task"], "surface": spec["surface"]},
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
    "CHANNELS", "CHECKPOINT_SCHEMA", "CONTEXT_LENGTH", "EXPORT_SCHEMA", "LoadedRoute",
    "ROUTES", "RouteConfig", "build_export_bundle", "capture_training_state",
    "channel_embeddings", "classification_logits", "contrastive_loss", "deployment_output",
    "load_artifact", "load_export_bundle", "load_route", "make_optimizer", "make_optimizer_for_parameters",
    "make_scheduler", "native_loss",
    "optimizer_step", "parent_tensor", "restore_export_bundle", "restore_training_state",
    "save_artifact", "seed_everything", "trainable_parameters", "validate_snapshot",
    "validate_source_runtime",
]
