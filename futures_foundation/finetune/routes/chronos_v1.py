"""Exact Chronos V1 64-token cross-entropy forecast route."""
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
from futures_foundation.finetune.native_training_schema_v2 import canonical_route_profile_sha256


ROUTE_KEY = "chronos_v1:F:native_64_t5_token_forecast_cross_entropy"
ARM_KEY = "chronos_v1"
TRACK = "F"
ROUTE_ID = "native_64_t5_token_forecast_cross_entropy"
CHECKPOINT_SCHEMA = "ffm_chronos_v1_native_training_state_v1"
EXPORT_SCHEMA = "ffm_chronos_v1_native_forecast_bundle_v1"
CONTEXT_LENGTH = 512
HORIZON_LENGTH = 64
EFFECTIVE_HORIZON = 16
PARENT_LENGTH = 576
CHANNELS = ("open", "high", "low", "close", "volume")
NUM_SAMPLES = 20


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
            raise ValueError("Chronos V1 learning_rate is outside the catalog bound")
        if not 0.0 <= float(self.weight_decay) <= 0.05:
            raise ValueError("Chronos V1 weight_decay is outside the catalog bound")
        if not 2 <= int(self.batch_size) <= 32:
            raise ValueError("Chronos V1 batch_size is outside the catalog bound")
        if int(self.gradient_accumulation_steps) != 1:
            raise ValueError("Chronos V1 exact executor supports gradient_accumulation_steps=1 only")
        if not 0.5 <= float(self.max_gradient_norm) <= 3.0:
            raise ValueError("Chronos V1 gradient clipping is outside the catalog bound")
        if not 1 <= int(self.total_steps) <= 32768:
            raise ValueError("Chronos V1 total_steps is outside the catalog bound")


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


def validate_snapshot(path: str | Path) -> tuple[Path, dict[str, Any]]:
    snapshot = Path(path).expanduser().resolve()
    if not snapshot.is_dir() or snapshot.is_symlink():
        raise FileNotFoundError(f"Chronos V1 snapshot is missing or unsafe: {snapshot}")
    config_path = snapshot / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError("Chronos V1 snapshot lacks config.json")
    dossier = get_dossier(ARM_KEY)
    if snapshot.name != dossier["model_revision"]:
        raise ValueError("Chronos V1 snapshot revision differs from the inference dossier")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    chronos = config.get("chronos_config")
    if not isinstance(chronos, dict):
        raise ValueError("Chronos V1 snapshot lacks chronos_config")
    if (
        int(chronos.get("context_length", -1)) != CONTEXT_LENGTH
        or int(chronos.get("prediction_length", -1)) != HORIZON_LENGTH
        or int(chronos.get("n_tokens", -1)) != 4096
        or int(chronos.get("eos_token_id", -1)) != 1
        or bool(chronos.get("use_eos_token")) is not True
    ):
        raise ValueError("Chronos V1 snapshot tokenizer geometry differs from the route")
    weights = [
        item for pattern in ("*.safetensors", "*.bin")
        for item in snapshot.glob(pattern) if item.is_file()
    ]
    if not weights:
        raise FileNotFoundError("Chronos V1 snapshot has no materialized weights")
    identity = {
        "path": str(snapshot),
        "model_id": dossier["model_id"],
        "model_revision": dossier["model_revision"],
        "config_sha256": _sha256(config_path),
        "weight_files": [
            {"path": item.name, "sha256": _sha256(item), "bytes": item.stat().st_size}
            for item in sorted(weights)
        ],
    }
    return snapshot, identity


def load_pipeline(snapshot: str | Path, *, device: str) -> tuple[Any, Any, dict[str, Any]]:
    torch = _torch()
    from chronos import BaseChronosPipeline
    source, identity = validate_snapshot(snapshot)
    pipeline = BaseChronosPipeline.from_pretrained(
        str(source), device_map="cpu", dtype=torch.float32,
    )
    model = pipeline.inner_model.to(device)
    if getattr(model.config, "_commit_hash", None) != identity["model_revision"]:
        raise RuntimeError("Chronos V1 loaded model revision drifted")
    config = pipeline.tokenizer.config
    if (
        int(config.context_length) != CONTEXT_LENGTH
        or int(config.prediction_length) != HORIZON_LENGTH
        or int(config.n_tokens) != 4096
        or int(config.eos_token_id) != 1
        or bool(config.use_eos_token) is not True
    ):
        raise RuntimeError("Chronos V1 loaded tokenizer contract drifted")
    return pipeline, model, identity


def parent_array(value: Any) -> np.ndarray:
    parent = np.asarray(value, dtype=np.float32)
    if parent.ndim != 3 or parent.shape[1:] != (PARENT_LENGTH, len(CHANNELS)):
        raise ValueError(
            f"Chronos V1 parent must have shape [B,{PARENT_LENGTH},{len(CHANNELS)}]"
        )
    if np.isinf(parent).any() or not np.isfinite(parent[:, CONTEXT_LENGTH:]).all():
        raise ValueError("Chronos V1 future must be finite and parent cannot contain infinities")
    o, h, l, c, v = parent.transpose(2, 0, 1)
    finite = np.isfinite(o) & np.isfinite(h) & np.isfinite(l) & np.isfinite(c)
    if np.any(finite & ((h < np.maximum(o, c)) | (l > np.minimum(o, c)) | (h < l))):
        raise ValueError("Chronos V1 parent violates finite-row OHLC geometry")
    if np.any(np.isfinite(v) & (v < 0)):
        raise ValueError("Chronos V1 parent contains negative volume")
    if not np.isfinite(parent[:, :CONTEXT_LENGTH]).any(axis=1).all():
        raise ValueError("Chronos V1 every channel needs an observed context value")
    return parent


def split_parent(parent: Any) -> tuple[Any, Any]:
    torch = _torch()
    values = parent_array(parent)
    context = values[:, :CONTEXT_LENGTH]
    future = values[:, CONTEXT_LENGTH:]
    context = np.transpose(context, (0, 2, 1)).reshape(-1, CONTEXT_LENGTH)
    future = np.transpose(future, (0, 2, 1)).reshape(-1, HORIZON_LENGTH)
    return torch.as_tensor(context), torch.as_tensor(future)


def tokenized_batch(pipeline: Any, parent: Any, *, device: str) -> tuple[Any, Any, Any]:
    context, future = split_parent(parent)
    input_ids, attention_mask, scale = pipeline.tokenizer.context_input_transform(context)
    label_ids, label_mask, _ = pipeline.tokenizer._input_transform(future, scale=scale)
    if pipeline.tokenizer.config.use_eos_token:
        label_ids, label_mask = pipeline.tokenizer._append_eos_token(label_ids, label_mask)
    labels = label_ids.masked_fill(~label_mask, -100)
    return input_ids.to(device), attention_mask.to(device), labels.to(device)


def native_loss(pipeline: Any, model: Any, parent: Any, *, device: str) -> Any:
    input_ids, attention_mask, labels = tokenized_batch(pipeline, parent, device=device)
    output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    if output.loss is None or not _torch().isfinite(output.loss):
        raise FloatingPointError("Chronos V1 native token loss is non-finite")
    return output.loss


def teacher_forced_logits(pipeline: Any, model: Any, parent: Any, *, device: str) -> Any:
    input_ids, attention_mask, labels = tokenized_batch(pipeline, parent, device=device)
    output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    logits = output.logits
    batch = len(parent_array(parent))
    if logits.shape[:2] != (batch * len(CHANNELS), HORIZON_LENGTH + 1):
        raise RuntimeError(f"Chronos V1 token-logit shape drifted: {tuple(logits.shape)}")
    if not _torch().isfinite(logits).all():
        raise FloatingPointError("Chronos V1 token logits are non-finite")
    return logits.reshape(batch, len(CHANNELS), HORIZON_LENGTH + 1, -1)


def forecast_samples(
    pipeline: Any,
    parent: Any,
    *,
    device: str,
    seed: int,
    num_samples: int = NUM_SAMPLES,
) -> Any:
    del device
    values = parent_array(parent)
    context, _ = split_parent(values)
    seed_everything(seed)
    samples = pipeline.predict(
        context,
        prediction_length=HORIZON_LENGTH,
        num_samples=int(num_samples),
        limit_prediction_length=True,
    )
    batch = len(values)
    if samples.shape != (batch * len(CHANNELS), int(num_samples), HORIZON_LENGTH):
        raise RuntimeError(f"Chronos V1 sample shape drifted: {tuple(samples.shape)}")
    result = samples.reshape(batch, len(CHANNELS), int(num_samples), HORIZON_LENGTH)
    if not _torch().isfinite(result).all():
        raise FloatingPointError("Chronos V1 forecast samples are non-finite")
    return result


def make_optimizer(model: Any, config: RouteConfig) -> Any:
    torch = _torch(); config.validate()
    return torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
        betas=(0.9, 0.999), eps=1e-8,
    )


def make_scheduler(optimizer: Any, config: RouteConfig) -> Any:
    torch = _torch(); config.validate()
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.total_steps)


def model_state_cpu(model: Any) -> dict[str, Any]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def optimizer_step(
    pipeline: Any, model: Any, optimizer: Any, scheduler: Any, parent: Any,
    *, device: str, max_gradient_norm: float,
) -> dict[str, float]:
    torch = _torch(); model.train()
    parameter = next(value for value in model.parameters() if value.requires_grad)
    before = parameter.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    loss = native_loss(pipeline, model, parent, device=device)
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
        raise ValueError("Chronos V1 sampler state closure is invalid")
    if int(sampler["cursor"]) != int(sampler_cursor):
        raise ValueError("Chronos V1 sampler cursor mismatch")
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
        raise ValueError("Chronos V1 training state field closure is invalid")
    if (
        state["schema_version"] != CHECKPOINT_SCHEMA
        or state["route_key"] != ROUTE_KEY
        or state["route_profile_sha256"] != canonical_route_profile_sha256(
            ARM_KEY, TRACK, ROUTE_ID
        )
        or dict(state["model_identity"]) != dict(model_identity)
        or dict(state["config"]) != asdict(config)
    ):
        raise ValueError("Chronos V1 training state identity is stale")
    sampler = state["sampler"]
    if not isinstance(sampler, Mapping) or set(sampler) != {
        "cursor", "schedule_kind", "schedule_sha256",
    }:
        raise ValueError("Chronos V1 sampler state is malformed")
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
            "tokenizer": "MeanScaleUniformBins",
            "scaler_tag": "chronos_mean_abs",
            "context_length": CONTEXT_LENGTH,
            "native_horizon": HORIZON_LENGTH,
            "missing_label_id": -100,
            "eos_token_id": 1,
            "channel_layout": "independent_univariate_passes",
            "channel_order": list(CHANNELS),
        },
        "output": {
            "kind": "forecast_samples",
            "shape": ["batch", len(CHANNELS), NUM_SAMPLES, HORIZON_LENGTH],
            "effective_deployment_horizon": EFFECTIVE_HORIZON,
            "deployment_filter": "first_16_of_native_64",
            "hidden_state_contract": "none_forecast_route",
        },
    }


def save_export_bundle(path: str | Path, bundle: Mapping[str, Any]) -> dict[str, Any]:
    target = _atomic_save(path, dict(bundle))
    return {"path": str(target), "sha256": _sha256(target), "bytes": target.stat().st_size}


def load_export_bundle(path: str | Path, *, snapshot: str | Path, device: str) -> tuple[Any, Any, dict[str, Any]]:
    bundle = _torch().load(Path(path).resolve(), map_location="cpu", weights_only=False)
    expected = {
        "schema_version", "route_key", "route_profile_sha256", "model_identity",
        "model_state", "preprocessing", "output",
    }
    if not isinstance(bundle, Mapping) or set(bundle) != expected:
        raise ValueError("Chronos V1 export bundle field closure is invalid")
    if (
        bundle["schema_version"] != EXPORT_SCHEMA
        or bundle["route_key"] != ROUTE_KEY
        or bundle["route_profile_sha256"] != canonical_route_profile_sha256(
            ARM_KEY, TRACK, ROUTE_ID
        )
    ):
        raise ValueError("Chronos V1 export bundle identity is stale")
    pipeline, model, identity = load_pipeline(snapshot, device=device)
    if dict(bundle["model_identity"]) != identity:
        raise ValueError("Chronos V1 export model identity mismatch")
    model.load_state_dict(bundle["model_state"], strict=True); model.eval()
    return pipeline, model, dict(bundle)


__all__ = [
    "ARM_KEY", "CHANNELS", "CHECKPOINT_SCHEMA", "CONTEXT_LENGTH", "EFFECTIVE_HORIZON",
    "EXPORT_SCHEMA", "HORIZON_LENGTH", "NUM_SAMPLES", "PARENT_LENGTH", "ROUTE_ID",
    "ROUTE_KEY", "TRACK", "RouteConfig", "build_export_bundle", "capture_training_state",
    "forecast_samples", "load_export_bundle", "load_pipeline", "load_training_state",
    "make_optimizer", "make_scheduler", "model_state_cpu", "native_loss", "optimizer_step",
    "parent_array", "restore_training_state", "save_export_bundle", "save_training_state",
    "seed_everything", "split_parent", "teacher_forced_logits", "tokenized_batch",
    "validate_snapshot",
]
