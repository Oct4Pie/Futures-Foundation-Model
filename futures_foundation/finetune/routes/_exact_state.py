"""Strict shared checkpoint and export transport for exact native routes.

This module owns transport only.  Family modules still own model loading, preprocessing,
objectives, trainable surfaces, optimizers, schedulers, and public output semantics.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np


def _torch():
    import torch
    return torch


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cpu_tree(value: Any) -> Any:
    torch = _torch()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {deepcopy(key): cpu_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    return deepcopy(value)


def module_states(modules: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    if not modules or any(not isinstance(name, str) or not name for name in modules):
        raise ValueError("exact route modules must have nonempty unique names")
    return {
        name: {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}
        for name, module in sorted(modules.items())
    }


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def capture_state(
    *,
    schema_version: str,
    route_key: str,
    route_profile_sha256: str,
    model_identity: Mapping[str, Any],
    config: Mapping[str, Any],
    modules: Mapping[str, Any],
    optimizer: Any,
    scheduler: Any | None,
    global_step: int,
    sampler_cursor: int,
    history: list[Mapping[str, Any]],
    sampler_state: Mapping[str, Any] | None = None,
    scaler: Any | None = None,
    epoch: int = 0,
) -> dict[str, Any]:
    torch = _torch()
    if not schema_version or not route_key or len(route_profile_sha256) != 64:
        raise ValueError("exact route checkpoint identity is malformed")
    sampler = dict(sampler_state or {
        "cursor": int(sampler_cursor),
        "schedule_kind": "explicit_step_batches_v1",
        "schedule_sha256": hashlib.sha256(b"explicit_step_batches_v1").hexdigest(),
    })
    if set(sampler) != {"cursor", "schedule_kind", "schedule_sha256"}:
        raise ValueError("exact route sampler state field closure is invalid")
    if int(sampler["cursor"]) != int(sampler_cursor):
        raise ValueError("exact route sampler cursor mismatch")
    if (
        not isinstance(sampler["schedule_kind"], str) or not sampler["schedule_kind"]
        or not isinstance(sampler["schedule_sha256"], str)
        or len(sampler["schedule_sha256"]) != 64
    ):
        raise ValueError("exact route sampler identity is malformed")
    return {
        "schema_version": schema_version,
        "route_key": route_key,
        "route_profile_sha256": route_profile_sha256,
        "model_identity": _json_copy(dict(model_identity)),
        "config": _json_copy(dict(config)),
        "modules": module_states(modules),
        "optimizer": cpu_tree(optimizer.state_dict()),
        "scheduler": None if scheduler is None else cpu_tree(scheduler.state_dict()),
        "scaler": None if scaler is None else cpu_tree(scaler.state_dict()),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "sampler": sampler,
        "history": _json_copy(list(history)),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_cpu_rng": torch.get_rng_state(),
        "torch_cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_state(
    state: Mapping[str, Any],
    *,
    schema_version: str,
    route_key: str,
    route_profile_sha256: str,
    model_identity: Mapping[str, Any],
    config: Mapping[str, Any],
    modules: Mapping[str, Any],
    optimizer: Any,
    scheduler: Any | None,
    scaler: Any | None = None,
) -> tuple[int, int, int, list[dict[str, Any]]]:
    torch = _torch()
    required = {
        "schema_version", "route_key", "route_profile_sha256", "model_identity", "config",
        "modules", "optimizer", "scheduler", "scaler", "epoch", "global_step", "sampler",
        "history", "python_rng", "numpy_rng", "torch_cpu_rng", "torch_cuda_rng",
    }
    if not isinstance(state, Mapping) or set(state) != required:
        raise ValueError("exact route checkpoint field closure is invalid")
    if (
        state["schema_version"] != schema_version
        or state["route_key"] != route_key
        or state["route_profile_sha256"] != route_profile_sha256
        or dict(state["model_identity"]) != _json_copy(dict(model_identity))
        or dict(state["config"]) != _json_copy(dict(config))
    ):
        raise ValueError("exact route checkpoint identity is stale or mismatched")
    expected_names = set(modules)
    if set(state["modules"]) != expected_names:
        raise ValueError("exact route checkpoint module closure mismatch")
    for name, module in modules.items():
        module.load_state_dict(state["modules"][name], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    if (scheduler is None) != (state["scheduler"] is None):
        raise ValueError("exact route scheduler presence mismatch")
    if scheduler is not None:
        scheduler.load_state_dict(state["scheduler"])
    if (scaler is None) != (state["scaler"] is None):
        raise ValueError("exact route scaler presence mismatch")
    if scaler is not None:
        scaler.load_state_dict(state["scaler"])
    sampler = state["sampler"]
    if not isinstance(sampler, Mapping) or set(sampler) != {
        "cursor", "schedule_kind", "schedule_sha256",
    }:
        raise ValueError("exact route sampler state is malformed")
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_cpu_rng"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda_rng"])
    return (
        int(state["epoch"]), int(state["global_step"]), int(sampler["cursor"]),
        _json_copy(state["history"]),
    )


def build_export_bundle(
    *,
    schema_version: str,
    route_key: str,
    route_profile_sha256: str,
    model_identity: Mapping[str, Any],
    modules: Mapping[str, Any],
    preprocessing: Mapping[str, Any],
    output: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    bundle = {
        "schema_version": schema_version,
        "route_key": route_key,
        "route_profile_sha256": route_profile_sha256,
        "model_identity": _json_copy(dict(model_identity)),
        "modules": module_states(modules),
        "preprocessing": _json_copy(dict(preprocessing)),
        "output": _json_copy(dict(output)),
        "extra": _json_copy(dict(extra or {})),
    }
    return bundle


def restore_export_bundle(
    bundle: Mapping[str, Any],
    *,
    schema_version: str,
    route_key: str,
    route_profile_sha256: str,
    model_identity: Mapping[str, Any],
    modules: Mapping[str, Any],
) -> dict[str, Any]:
    required = {
        "schema_version", "route_key", "route_profile_sha256", "model_identity",
        "modules", "preprocessing", "output", "extra",
    }
    if not isinstance(bundle, Mapping) or set(bundle) != required:
        raise ValueError("exact route export field closure is invalid")
    if (
        bundle["schema_version"] != schema_version
        or bundle["route_key"] != route_key
        or bundle["route_profile_sha256"] != route_profile_sha256
        or dict(bundle["model_identity"]) != _json_copy(dict(model_identity))
        or set(bundle["modules"]) != set(modules)
    ):
        raise ValueError("exact route export identity is stale or mismatched")
    for name, module in modules.items():
        module.load_state_dict(bundle["modules"][name], strict=True)
    return _json_copy({
        "preprocessing": bundle["preprocessing"],
        "output": bundle["output"],
        "extra": bundle["extra"],
    })


def atomic_torch_save(path: str | Path, value: Any) -> dict[str, Any]:
    torch = _torch()
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, target)
    return {"path": str(target), "sha256": sha256_file(target), "bytes": target.stat().st_size}


def load_torch(path: str | Path) -> Any:
    return _torch().load(Path(path).expanduser().resolve(), map_location="cpu", weights_only=False)


__all__ = [
    "atomic_torch_save", "build_export_bundle", "capture_state", "cpu_tree", "load_torch",
    "module_states", "restore_export_bundle", "restore_state", "sha256_file",
]
