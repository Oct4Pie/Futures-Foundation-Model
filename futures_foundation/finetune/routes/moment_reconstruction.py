"""Exact MOMENT-small native masked-reconstruction route."""
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


ROUTE_KEY = "moment_small:R:masked_patch_reconstruction"
ARM_KEY = "moment_small"
TRACK = "R"
ROUTE_ID = "masked_patch_reconstruction"
CHECKPOINT_SCHEMA = "ffm_moment_reconstruction_training_state_v1"
EXPORT_SCHEMA = "ffm_moment_reconstruction_bundle_v1"
CONTEXT_LENGTH = 512
PARENT_LENGTH = 512
CHANNELS = ("open", "high", "low", "close", "volume")
MASK_RATIO = 0.4


@dataclass(frozen=True)
class RouteConfig:
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    max_gradient_norm: float = 1.0
    total_steps: int = 20
    seed: int = 20260718

    def validate(self) -> None:
        if not 1e-6 <= float(self.learning_rate) <= 1e-4:
            raise ValueError("MOMENT learning_rate is outside the catalog bound")
        if not 0.0 <= float(self.weight_decay) <= 0.05:
            raise ValueError("MOMENT weight_decay is outside the catalog bound")
        if not 2 <= int(self.batch_size) <= 16:
            raise ValueError("MOMENT batch_size is outside the catalog bound")
        if int(self.gradient_accumulation_steps) != 1:
            raise ValueError("MOMENT exact executor supports gradient_accumulation_steps=1 only")
        if not 0.5 <= float(self.max_gradient_norm) <= 3.0:
            raise ValueError("MOMENT gradient clipping is outside the catalog bound")
        if not 1 <= int(self.total_steps) <= 32768:
            raise ValueError("MOMENT total_steps is outside the catalog bound")


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


def validate_source_runtime(path: str | Path) -> tuple[Path, dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    if not source.is_dir() or source.is_symlink() or not (source / ".git").is_dir():
        raise FileNotFoundError(f"MOMENT source runtime is missing or unsafe: {source}")
    dossier = get_dossier(ARM_KEY)
    head = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True,
    ).strip()
    if head != dossier["source_revision"]:
        raise ValueError("MOMENT source revision differs from the inference dossier")
    dirty = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=all"],
        text=True,
    ).strip()
    if dirty:
        raise ValueError("MOMENT source runtime is dirty")
    model_source = source / "momentfm" / "models" / "moment.py"
    if not model_source.is_file() or model_source.is_symlink():
        raise FileNotFoundError("MOMENT source runtime lacks moment.py")
    return source, {
        "path": str(source),
        "head_revision": head,
        "model_source_sha256": _sha256(model_source),
    }


def validate_snapshot(path: str | Path) -> tuple[Path, dict[str, Any]]:
    snapshot = Path(path).expanduser().resolve()
    if not snapshot.is_dir() or snapshot.is_symlink():
        raise FileNotFoundError(f"MOMENT snapshot is missing or unsafe: {snapshot}")
    dossier = get_dossier(ARM_KEY)
    if snapshot.name != dossier["model_revision"]:
        raise ValueError("MOMENT snapshot revision differs from the inference dossier")
    config_path = snapshot / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError("MOMENT snapshot lacks config.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if int(config.get("seq_len", -1)) != CONTEXT_LENGTH or int(config.get("patch_len", -1)) != 8:
        raise ValueError("MOMENT snapshot sequence/patch geometry differs from the route")
    weights = [
        item for pattern in ("*.safetensors", "*.bin")
        for item in snapshot.glob(pattern) if item.is_file()
    ]
    if not weights:
        raise FileNotFoundError("MOMENT snapshot has no materialized weights")
    return snapshot, {
        "path": str(snapshot),
        "model_id": dossier["model_id"],
        "model_revision": dossier["model_revision"],
        "config_sha256": _sha256(config_path),
        "weight_files": [
            {"path": item.name, "sha256": _sha256(item), "bytes": item.stat().st_size}
            for item in sorted(weights)
        ],
    }


def load_model(
    snapshot: str | Path,
    *,
    source_runtime: str | Path,
    device: str,
) -> tuple[Any, dict[str, Any]]:
    source, source_identity = validate_source_runtime(source_runtime)
    model_path, model_identity = validate_snapshot(snapshot)
    sys.path.insert(0, str(source))
    try:
        from momentfm import MOMENTPipeline
    finally:
        sys.path.pop(0)
    imported = Path(inspect.getfile(MOMENTPipeline)).resolve()
    if source not in imported.parents:
        raise RuntimeError("MOMENTPipeline did not import from the pinned source runtime")
    model = MOMENTPipeline.from_pretrained(
        str(model_path),
        model_kwargs={
            "task_name": "reconstruction",
            "mask_ratio": MASK_RATIO,
            "freeze_encoder": False,
            "freeze_embedder": False,
            "freeze_head": False,
            "enable_gradient_checkpointing": False,
        },
    )
    model.init()
    if int(model.seq_len) != CONTEXT_LENGTH or int(model.patch_len) != 8:
        raise RuntimeError("loaded MOMENT sequence/patch geometry drifted")
    identity = {**model_identity, "source_runtime": source_identity}
    return model.to(device), identity


def parent_array(value: Any) -> np.ndarray:
    parent = np.asarray(value, dtype=np.float32)
    if parent.ndim != 3 or parent.shape[1:] != (PARENT_LENGTH, len(CHANNELS)):
        raise ValueError(f"MOMENT parent must have shape [B,{PARENT_LENGTH},{len(CHANNELS)}]")
    if np.isinf(parent).any():
        raise ValueError("MOMENT parent contains infinite values")
    finite = np.isfinite(parent)
    timestamp_finite = finite.all(axis=2)
    timestamp_missing = (~finite).all(axis=2)
    if not np.all(timestamp_finite | timestamp_missing):
        raise ValueError("MOMENT supports shared timestamp missingness, not partial-channel NaNs")
    if np.any(timestamp_finite.sum(axis=1) < 2):
        raise ValueError("MOMENT every parent requires at least two observed timestamps")
    safe = np.where(finite, parent, 0.0)
    o, h, l, c, v = safe.transpose(2, 0, 1)
    if np.any(timestamp_finite & ((h < np.maximum(o, c)) | (l > np.minimum(o, c)) | (h < l))):
        raise ValueError("MOMENT parent violates observed OHLC geometry")
    if np.any(timestamp_finite & (v < 0)):
        raise ValueError("MOMENT parent contains negative observed volume")
    return parent


def model_inputs(parent: Any, *, device: str) -> tuple[Any, Any]:
    torch = _torch()
    values = parent_array(parent)
    observed = np.isfinite(values).all(axis=2)
    filled = np.where(np.isfinite(values), values, 0.0)
    x = torch.as_tensor(np.transpose(filled, (0, 2, 1)), dtype=torch.float32, device=device)
    mask = torch.as_tensor(observed.astype(np.int64), device=device)
    return x, mask


def native_output(model: Any, parent: Any, *, device: str) -> tuple[Any, Any, Any]:
    x, input_mask = model_inputs(parent, device=device)
    output = model(x_enc=x, input_mask=input_mask)
    if (
        output.reconstruction is None
        or output.pretrain_mask is None
        or output.reconstruction.shape != x.shape
        or output.pretrain_mask.shape != input_mask.shape
    ):
        raise RuntimeError("MOMENT reconstruction/pretrain-mask output shape drifted")
    if not _torch().isfinite(output.reconstruction).all():
        raise FloatingPointError("MOMENT reconstruction contains non-finite values")
    return output, x, input_mask


def native_loss(model: Any, parent: Any, *, device: str) -> Any:
    output, original, input_mask = native_output(model, parent, device=device)
    hidden = input_mask * (1 - output.pretrain_mask)
    if int(hidden.sum()) < 1:
        raise RuntimeError("MOMENT native mask hid no valid timestamps")
    squared = (output.reconstruction - original).square()
    loss = (squared * hidden[:, None, :]).sum() / (
        hidden.sum() * original.shape[1]
    )
    if not _torch().isfinite(loss):
        raise FloatingPointError("MOMENT native masked raw MSE is non-finite")
    return loss


def reconstruction(model: Any, parent: Any, *, device: str, seed: int) -> Any:
    seed_everything(seed)
    output, _, _ = native_output(model, parent, device=device)
    return output.reconstruction


def mean_embedding(model: Any, parent: Any, *, device: str) -> Any:
    x, input_mask = model_inputs(parent, device=device)
    output = model.embed(x_enc=x, input_mask=input_mask, reduction="mean")
    embedding = output.embeddings
    if embedding is None or embedding.shape != (len(parent_array(parent)), 512):
        raise RuntimeError(f"MOMENT mean embedding shape drifted: {getattr(embedding, 'shape', None)}")
    if not _torch().isfinite(embedding).all():
        raise FloatingPointError("MOMENT mean embedding contains non-finite values")
    return embedding


def make_optimizer(model: Any, config: RouteConfig) -> Any:
    torch = _torch(); config.validate()
    return torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
        betas=(0.9, 0.95), eps=1e-8,
    )


def make_scheduler(optimizer: Any, config: RouteConfig) -> Any:
    torch = _torch(); config.validate()
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.total_steps)


def model_state_cpu(model: Any) -> dict[str, Any]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def optimizer_step(
    model: Any, optimizer: Any, scheduler: Any, parent: Any,
    *, device: str, max_gradient_norm: float,
) -> dict[str, float]:
    torch = _torch(); model.train()
    parameter = next(value for value in model.parameters() if value.requires_grad)
    before = parameter.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    loss = native_loss(model, parent, device=device)
    loss.backward()
    grad = torch.sqrt(sum(
        value.grad.detach().float().square().sum()
        for value in model.parameters() if value.requires_grad and value.grad is not None
    ))
    torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_gradient_norm))
    optimizer.step(); scheduler.step()
    delta = torch.max(torch.abs(parameter.detach() - before))
    return {
        "loss": float(loss.detach().cpu()),
        "grad_norm": float(grad.detach().cpu()),
        "parameter_delta": float(delta.detach().cpu()),
    }


def capture_training_state(
    *, model: Any, optimizer: Any, scheduler: Any, config: RouteConfig,
    model_identity: Mapping[str, Any], global_step: int, sampler_cursor: int,
    history: list[dict[str, Any]], sampler_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    torch = _torch(); config.validate()
    sampler = dict(sampler_state or {
        "cursor": int(sampler_cursor),
        "schedule_kind": "explicit_synthetic_batch_v1",
        "schedule_sha256": hashlib.sha256(b"explicit_synthetic_batch_v1").hexdigest(),
    })
    if set(sampler) != {"cursor", "schedule_kind", "schedule_sha256"}:
        raise ValueError("MOMENT sampler state closure is invalid")
    if int(sampler["cursor"]) != int(sampler_cursor):
        raise ValueError("MOMENT sampler cursor mismatch")
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
        "history": _history(history),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_cpu_rng": torch.get_rng_state(),
        "torch_cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_training_state(
    state: Mapping[str, Any], *, model: Any, optimizer: Any, scheduler: Any,
    config: RouteConfig, model_identity: Mapping[str, Any],
) -> tuple[int, int, list[dict[str, Any]]]:
    torch = _torch()
    required = {
        "schema_version", "route_key", "route_profile_sha256", "model_identity", "config",
        "model", "optimizer", "scheduler", "scaler", "epoch", "global_step", "sampler",
        "history", "python_rng", "numpy_rng", "torch_cpu_rng", "torch_cuda_rng",
    }
    if not isinstance(state, Mapping) or set(state) != required:
        raise ValueError("MOMENT training state field closure is invalid")
    if (
        state["schema_version"] != CHECKPOINT_SCHEMA
        or state["route_key"] != ROUTE_KEY
        or state["route_profile_sha256"] != canonical_route_profile_sha256(
            ARM_KEY, TRACK, ROUTE_ID
        )
        or dict(state["model_identity"]) != dict(model_identity)
        or dict(state["config"]) != asdict(config)
    ):
        raise ValueError("MOMENT training state identity is stale")
    sampler = state["sampler"]
    if not isinstance(sampler, Mapping) or set(sampler) != {
        "cursor", "schedule_kind", "schedule_sha256",
    }:
        raise ValueError("MOMENT sampler state is malformed")
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"])
    random.setstate(state["python_rng"]); np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_cpu_rng"])
    if torch.cuda.is_available(): torch.cuda.set_rng_state_all(state["torch_cuda_rng"])
    return int(state["global_step"]), int(sampler["cursor"]), _history(state["history"])


def save_training_state(path: str | Path, state: Mapping[str, Any]) -> dict[str, Any]:
    target = _atomic_save(path, dict(state))
    return {"path": str(target), "sha256": _sha256(target), "bytes": target.stat().st_size}


def load_training_state(path: str | Path) -> dict[str, Any]:
    return _torch().load(Path(path).resolve(), map_location="cpu", weights_only=False)


def build_export_bundle(*, model: Any, model_identity: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": EXPORT_SCHEMA,
        "route_key": ROUTE_KEY,
        "route_profile_sha256": canonical_route_profile_sha256(ARM_KEY, TRACK, ROUTE_ID),
        "model_identity": dict(model_identity),
        "model_state": model_state_cpu(model),
        "preprocessing": {
            "input_shape": ["batch", len(CHANNELS), CONTEXT_LENGTH],
            "input_mask": "shared_valid_timestamp_mask",
            "pretrain_mask": "native_shared_timestamp_mask_ratio_0_4",
            "scaler_tag": "moment_revin_per_channel",
            "missing_fill": "zero_before_native_mask",
            "channel_order": list(CHANNELS),
        },
        "output": {
            "kind": "mean_embedding",
            "shape": ["batch", 512],
            "reduction": "official_mean",
            "hidden_state_contract": "official_mean_embedding_only",
        },
    }


def save_export_bundle(path: str | Path, bundle: Mapping[str, Any]) -> dict[str, Any]:
    target = _atomic_save(path, dict(bundle))
    return {"path": str(target), "sha256": _sha256(target), "bytes": target.stat().st_size}


def load_export_bundle(
    path: str | Path, *, snapshot: str | Path, source_runtime: str | Path, device: str,
) -> tuple[Any, dict[str, Any]]:
    bundle = _torch().load(Path(path).resolve(), map_location="cpu", weights_only=False)
    expected = {
        "schema_version", "route_key", "route_profile_sha256", "model_identity",
        "model_state", "preprocessing", "output",
    }
    if not isinstance(bundle, Mapping) or set(bundle) != expected:
        raise ValueError("MOMENT export bundle field closure is invalid")
    if (
        bundle["schema_version"] != EXPORT_SCHEMA
        or bundle["route_key"] != ROUTE_KEY
        or bundle["route_profile_sha256"] != canonical_route_profile_sha256(
            ARM_KEY, TRACK, ROUTE_ID
        )
    ):
        raise ValueError("MOMENT export bundle identity is stale")
    model, identity = load_model(snapshot, source_runtime=source_runtime, device=device)
    if dict(bundle["model_identity"]) != identity:
        raise ValueError("MOMENT export model/source identity mismatch")
    model.load_state_dict(bundle["model_state"], strict=True); model.eval()
    return model, dict(bundle)


__all__ = [
    "ARM_KEY", "CHANNELS", "CHECKPOINT_SCHEMA", "CONTEXT_LENGTH", "EXPORT_SCHEMA",
    "MASK_RATIO", "PARENT_LENGTH", "ROUTE_ID", "ROUTE_KEY", "TRACK", "RouteConfig",
    "build_export_bundle", "capture_training_state", "load_export_bundle", "load_model",
    "load_training_state", "make_optimizer", "make_scheduler", "mean_embedding",
    "model_inputs", "model_state_cpu", "native_loss", "native_output", "optimizer_step",
    "parent_array", "reconstruction", "restore_training_state", "save_export_bundle",
    "save_training_state", "seed_everything", "validate_snapshot", "validate_source_runtime",
]
