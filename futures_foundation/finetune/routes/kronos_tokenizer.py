"""Exact Kronos Mini/Small native tokenizer reconstruction routes."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from typing import Any, Mapping

import numpy as np

from futures_foundation.finetune.native_contracts import get_dossier
from futures_foundation.finetune.native_training_schema_v2 import canonical_route_profile_sha256


TRACK = "F"
ROUTE_ID = "tokenizer_reconstruction_bsq"
CHECKPOINT_SCHEMA = "ffm_kronos_tokenizer_training_state_v1"
EXPORT_SCHEMA = "ffm_kronos_tokenizer_bundle_v1"
CONTEXT_LENGTH = 528
PARENT_LENGTH = 528
INPUT_CHANNELS = ("open", "high", "low", "close", "volume")
NATIVE_CHANNELS = ("open", "high", "low", "close", "volume", "amount")
SUPPORTED_ARMS = ("kronos_mini", "kronos_small")


@dataclass(frozen=True)
class RouteConfig:
    learning_rate: float = 2e-5
    weight_decay: float = 0.05
    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    max_gradient_norm: float = 3.0
    total_steps: int = 20
    seed: int = 20260718

    def validate(self) -> None:
        if not 1e-6 <= float(self.learning_rate) <= 1e-4:
            raise ValueError("Kronos tokenizer learning_rate is outside the catalog bound")
        if not 0.0 <= float(self.weight_decay) <= 0.1:
            raise ValueError("Kronos tokenizer weight_decay is outside the catalog bound")
        if not 2 <= int(self.batch_size) <= 32:
            raise ValueError("Kronos tokenizer batch_size is outside the catalog bound")
        if int(self.gradient_accumulation_steps) != 1:
            raise ValueError("Kronos tokenizer exact executor supports gradient_accumulation_steps=1 only")
        if not 0.5 <= float(self.max_gradient_norm) <= 5.0:
            raise ValueError("Kronos tokenizer gradient clipping is outside the catalog bound")
        if not 1 <= int(self.total_steps) <= 32768:
            raise ValueError("Kronos tokenizer total_steps is outside the catalog bound")


def route_key(arm_key: str) -> str:
    if arm_key not in SUPPORTED_ARMS:
        raise ValueError(f"unsupported Kronos tokenizer arm: {arm_key}")
    return f"{arm_key}:{TRACK}:{ROUTE_ID}"


def _torch() -> Any:
    import torch
    return torch


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_save(path: str | Path, value: object) -> Path:
    torch = _torch()
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(target) + f".{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, target)
    return target


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


def _history(value: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return json.loads(json.dumps(value, allow_nan=False))


def seed_everything(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch = _torch()
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def validate_source_runtime(path: str | Path, arm_key: str) -> tuple[Path, dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    if not source.is_dir() or source.is_symlink() or not (source / ".git").is_dir():
        raise FileNotFoundError(f"Kronos source runtime is missing or unsafe: {source}")
    dossier = get_dossier(arm_key)
    head = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True,
    ).strip()
    if head != dossier["source_revision"]:
        raise ValueError("Kronos source revision differs from the inference dossier")
    dirty = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=all"],
        text=True,
    ).strip()
    if dirty:
        raise ValueError("Kronos source runtime is dirty")
    source_file = source / "model" / "kronos.py"
    if not source_file.is_file() or source_file.is_symlink():
        raise FileNotFoundError("Kronos source runtime lacks model/kronos.py")
    return source, {
        "path": str(source),
        "head_revision": head,
        "model_source_sha256": _sha256(source_file),
    }


def _snapshot_identity(path: str | Path, expected_revision: str, label: str) -> tuple[Path, dict[str, Any]]:
    snapshot = Path(path).expanduser().resolve()
    if not snapshot.is_dir() or snapshot.is_symlink() or snapshot.name != expected_revision:
        raise FileNotFoundError(f"{label} snapshot is missing, unsafe, or has wrong revision")
    config_path = snapshot / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"{label} snapshot lacks config.json")
    weights = [
        item for pattern in ("*.safetensors", "*.bin", "*.pt")
        for item in snapshot.glob(pattern) if item.is_file()
    ]
    if not weights:
        raise FileNotFoundError(f"{label} snapshot has no materialized weights")
    return snapshot, {
        "path": str(snapshot),
        "revision": expected_revision,
        "config_sha256": _sha256(config_path),
        "weight_files": [
            {"path": item.name, "sha256": _sha256(item), "bytes": item.stat().st_size}
            for item in sorted(weights)
        ],
    }


def load_tokenizer(
    arm_key: str,
    *,
    model_snapshot: str | Path,
    tokenizer_snapshot: str | Path,
    source_runtime: str | Path,
    device: str,
) -> tuple[Any, dict[str, Any]]:
    if arm_key not in SUPPORTED_ARMS:
        raise ValueError(f"unsupported Kronos tokenizer arm: {arm_key}")
    dossier = get_dossier(arm_key)
    source, source_identity = validate_source_runtime(source_runtime, arm_key)
    model_path, model_identity = _snapshot_identity(
        model_snapshot, str(dossier["model_revision"]), f"{arm_key} predictor",
    )
    tokenizer = dossier["tokenizer"]
    tokenizer_path, tokenizer_identity = _snapshot_identity(
        tokenizer_snapshot, str(tokenizer["revision"]), f"{arm_key} tokenizer",
    )
    sys.path.insert(0, str(source))
    try:
        from model import KronosTokenizer
    finally:
        sys.path.pop(0)
    imported = Path(inspect.getfile(KronosTokenizer)).resolve()
    if source not in imported.parents:
        raise RuntimeError("KronosTokenizer did not import from the pinned source runtime")
    model = KronosTokenizer.from_pretrained(str(tokenizer_path)).to(device)
    identity = {
        "arm_key": arm_key,
        "model_id": dossier["model_id"],
        "model_snapshot": model_identity,
        "tokenizer_id": tokenizer["id"],
        "tokenizer_snapshot": tokenizer_identity,
        "source_runtime": source_identity,
    }
    del model_path
    return model, identity


def parent_array(value: Any) -> np.ndarray:
    parent = np.asarray(value, dtype=np.float32)
    if parent.ndim != 3 or parent.shape[1:] != (PARENT_LENGTH, len(INPUT_CHANNELS)):
        raise ValueError(f"Kronos parent must have shape [B,{PARENT_LENGTH},5]")
    if not np.isfinite(parent).all():
        raise ValueError("Kronos tokenizer route rejects missing or non-finite parents")
    o, h, l, c, v = parent.transpose(2, 0, 1)
    if np.any((h < np.maximum(o, c)) | (l > np.minimum(o, c)) | (h < l) | (v < 0)):
        raise ValueError("Kronos parent violates OHLCV geometry")
    return parent


def native_ohlcva(parent: Any) -> np.ndarray:
    values = parent_array(parent)
    amount = values[:, :, 4:5] * values[:, :, :4].mean(axis=2, keepdims=True)
    return np.concatenate((values, amount), axis=2).astype(np.float32)


def normalize(parent: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = native_ohlcva(parent)
    mean = values.mean(axis=1, keepdims=True)
    std = np.maximum(values.std(axis=1, keepdims=True), 1e-5)
    normalized = np.clip((values - mean) / std, -5.0, 5.0).astype(np.float32)
    return normalized, mean.astype(np.float32), std.astype(np.float32)


def model_input(parent: Any, *, device: str) -> Any:
    torch = _torch()
    normalized, _, _ = normalize(parent)
    return torch.as_tensor(normalized, dtype=torch.float32, device=device)


def native_components(tokenizer: Any, parent: Any, *, device: str) -> tuple[Any, Any, Any]:
    values = model_input(parent, device=device)
    (coarse, full), quantizer_loss, _, _ = tokenizer(values)
    if coarse.shape != values.shape or full.shape != values.shape:
        raise RuntimeError("Kronos tokenizer reconstruction shape drifted")
    if not _torch().isfinite(coarse).all() or not _torch().isfinite(full).all() or not _torch().isfinite(quantizer_loss):
        raise FloatingPointError("Kronos tokenizer output contains non-finite values")
    return coarse, full, quantizer_loss


def native_loss(tokenizer: Any, parent: Any, *, device: str) -> Any:
    torch = _torch()
    values = model_input(parent, device=device)
    (coarse, full), quantizer_loss, _, _ = tokenizer(values)
    reconstruction = torch.nn.functional.mse_loss(coarse, values) + torch.nn.functional.mse_loss(full, values)
    loss = (reconstruction + quantizer_loss) / 2.0
    if not torch.isfinite(loss):
        raise FloatingPointError("Kronos tokenizer native loss is non-finite")
    return loss


def tokenizer_codes(tokenizer: Any, parent: Any, *, device: str) -> tuple[Any, Any]:
    values = model_input(parent, device=device)
    coarse, fine = tokenizer.encode(values, half=True)
    expected = (values.shape[0], PARENT_LENGTH)
    if (
        coarse.ndim != 2
        or fine.ndim != 2
        or tuple(coarse.shape) != expected
        or tuple(fine.shape) != expected
    ):
        raise RuntimeError(
            "Kronos tokenizer codes must be coarse/fine integer matrices "
            f"with shape {expected}"
        )
    if coarse.dtype.is_floating_point or fine.dtype.is_floating_point:
        raise RuntimeError("Kronos tokenizer codes must use integer dtypes")
    return coarse, fine


def make_optimizer(tokenizer: Any, config: RouteConfig) -> Any:
    torch = _torch(); config.validate()
    return torch.optim.AdamW(
        tokenizer.parameters(), lr=config.learning_rate,
        weight_decay=config.weight_decay, betas=(0.9, 0.95), eps=1e-8,
    )


def make_scheduler(optimizer: Any, config: RouteConfig) -> Any:
    torch = _torch(); config.validate()
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.total_steps)


def model_state_cpu(tokenizer: Any) -> dict[str, Any]:
    return {key: value.detach().cpu().clone() for key, value in tokenizer.state_dict().items()}


def optimizer_step(
    tokenizer: Any, optimizer: Any, scheduler: Any, parent: Any,
    *, device: str, max_gradient_norm: float,
) -> dict[str, float]:
    torch = _torch(); tokenizer.train()
    parameter = next(value for value in tokenizer.parameters() if value.requires_grad)
    before = parameter.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    loss = native_loss(tokenizer, parent, device=device)
    loss.backward()
    grad = torch.sqrt(sum(
        value.grad.detach().float().square().sum()
        for value in tokenizer.parameters() if value.requires_grad and value.grad is not None
    ))
    torch.nn.utils.clip_grad_norm_(tokenizer.parameters(), float(max_gradient_norm))
    optimizer.step(); scheduler.step()
    delta = torch.max(torch.abs(parameter.detach() - before))
    return {
        "loss": float(loss.detach().cpu()),
        "grad_norm": float(grad.detach().cpu()),
        "parameter_delta": float(delta.detach().cpu()),
    }


def capture_training_state(
    *, arm_key: str, tokenizer: Any, optimizer: Any, scheduler: Any,
    config: RouteConfig, model_identity: Mapping[str, Any], global_step: int,
    sampler_cursor: int, history: list[dict[str, Any]],
    sampler_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    torch = _torch(); config.validate(); key = route_key(arm_key)
    sampler = dict(sampler_state or {
        "cursor": int(sampler_cursor),
        "schedule_kind": "explicit_synthetic_batch_v1",
        "schedule_sha256": hashlib.sha256(b"explicit_synthetic_batch_v1").hexdigest(),
    })
    if set(sampler) != {"cursor", "schedule_kind", "schedule_sha256"} or int(sampler["cursor"]) != int(sampler_cursor):
        raise ValueError("Kronos tokenizer sampler state is invalid")
    return {
        "schema_version": CHECKPOINT_SCHEMA,
        "route_key": key,
        "route_profile_sha256": canonical_route_profile_sha256(arm_key, TRACK, ROUTE_ID),
        "model_identity": dict(model_identity),
        "config": asdict(config),
        "model": model_state_cpu(tokenizer),
        "optimizer": _cpu_tree(optimizer.state_dict()),
        "scheduler": _cpu_tree(scheduler.state_dict()),
        "scaler": None,
        "epoch": 0,
        "global_step": int(global_step),
        "sampler": sampler,
        "history": _history(history),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_cpu_rng": torch.get_rng_state(),
        "torch_cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_training_state(
    state: Mapping[str, Any], *, arm_key: str, tokenizer: Any,
    optimizer: Any, scheduler: Any, config: RouteConfig,
    model_identity: Mapping[str, Any],
) -> tuple[int, int, list[dict[str, Any]]]:
    torch = _torch()
    required = {
        "schema_version", "route_key", "route_profile_sha256", "model_identity", "config",
        "model", "optimizer", "scheduler", "scaler", "epoch", "global_step", "sampler",
        "history", "python_rng", "numpy_rng", "torch_cpu_rng", "torch_cuda_rng",
    }
    if not isinstance(state, Mapping) or set(state) != required:
        raise ValueError("Kronos tokenizer training state closure is invalid")
    if (
        state["schema_version"] != CHECKPOINT_SCHEMA
        or state["route_key"] != route_key(arm_key)
        or state["route_profile_sha256"] != canonical_route_profile_sha256(arm_key, TRACK, ROUTE_ID)
        or dict(state["model_identity"]) != dict(model_identity)
        or dict(state["config"]) != asdict(config)
    ):
        raise ValueError("Kronos tokenizer training state identity is stale")
    tokenizer.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"])
    random.setstate(state["python_rng"]); np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_cpu_rng"])
    if torch.cuda.is_available(): torch.cuda.set_rng_state_all(state["torch_cuda_rng"])
    sampler = state["sampler"]
    return int(state["global_step"]), int(sampler["cursor"]), _history(state["history"])


def save_training_state(path: str | Path, state: Mapping[str, Any]) -> dict[str, Any]:
    target = _atomic_save(path, dict(state))
    return {"path": str(target), "sha256": _sha256(target), "bytes": target.stat().st_size}


def load_training_state(path: str | Path) -> dict[str, Any]:
    return _torch().load(Path(path).resolve(), map_location="cpu", weights_only=False)


def build_export_bundle(
    *, arm_key: str, tokenizer: Any, model_identity: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": EXPORT_SCHEMA,
        "route_key": route_key(arm_key),
        "route_profile_sha256": canonical_route_profile_sha256(arm_key, TRACK, ROUTE_ID),
        "model_identity": dict(model_identity),
        "model_state": model_state_cpu(tokenizer),
        "preprocessing": {
            "input_shape": ["batch", CONTEXT_LENGTH, len(NATIVE_CHANNELS)],
            "channel_order": list(NATIVE_CHANNELS),
            "amount_source": "volume_times_mean_ohlc_v1",
            "normalization": "full_528_context_mean_std_clip5",
            "missing_policy": "reject_parent",
        },
        "output": {
            "kind": "coarse_and_fine_codes",
            "hidden_state_contract": "tokenizer_codes_only",
        },
    }


def save_export_bundle(path: str | Path, bundle: Mapping[str, Any]) -> dict[str, Any]:
    target = _atomic_save(path, dict(bundle))
    return {"path": str(target), "sha256": _sha256(target), "bytes": target.stat().st_size}


def load_export_bundle(
    path: str | Path, *, arm_key: str, model_snapshot: str | Path,
    tokenizer_snapshot: str | Path, source_runtime: str | Path, device: str,
) -> tuple[Any, dict[str, Any]]:
    bundle = _torch().load(Path(path).resolve(), map_location="cpu", weights_only=False)
    expected = {
        "schema_version", "route_key", "route_profile_sha256", "model_identity",
        "model_state", "preprocessing", "output",
    }
    if not isinstance(bundle, Mapping) or set(bundle) != expected:
        raise ValueError("Kronos tokenizer export bundle closure is invalid")
    if (
        bundle["schema_version"] != EXPORT_SCHEMA
        or bundle["route_key"] != route_key(arm_key)
        or bundle["route_profile_sha256"] != canonical_route_profile_sha256(arm_key, TRACK, ROUTE_ID)
    ):
        raise ValueError("Kronos tokenizer export bundle identity is stale")
    tokenizer, identity = load_tokenizer(
        arm_key, model_snapshot=model_snapshot, tokenizer_snapshot=tokenizer_snapshot,
        source_runtime=source_runtime, device=device,
    )
    if dict(bundle["model_identity"]) != identity:
        raise ValueError("Kronos tokenizer export model identity mismatch")
    tokenizer.load_state_dict(bundle["model_state"], strict=True); tokenizer.eval()
    return tokenizer, dict(bundle)


__all__ = [
    "CHECKPOINT_SCHEMA", "CONTEXT_LENGTH", "EXPORT_SCHEMA", "INPUT_CHANNELS",
    "NATIVE_CHANNELS", "PARENT_LENGTH", "ROUTE_ID", "SUPPORTED_ARMS", "TRACK",
    "RouteConfig", "build_export_bundle", "capture_training_state", "load_export_bundle",
    "load_tokenizer", "load_training_state", "make_optimizer", "make_scheduler",
    "model_input", "model_state_cpu", "native_components", "native_loss", "native_ohlcva",
    "normalize", "optimizer_step", "parent_array", "restore_training_state", "route_key",
    "save_export_bundle", "save_training_state", "seed_everything", "tokenizer_codes",
    "validate_source_runtime",
]
