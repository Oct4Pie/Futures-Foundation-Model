"""Exact Kronos Mini/Small hierarchical autoregressive predictor routes.

The predictor is initialized from the matching pinned vanilla snapshot, but its
frozen tokenizer must come from a passing, authority-bound tokenizer pilot for the
same arm.  Native training follows upstream Kronos semantics exactly:

* OHLCVA amount is ``volume * mean(OHLC)``;
* statistics are fitted on the first 512 context bars and applied to all 528 bars;
* time stamps are minute/hour/weekday/day/month in ``America/Chicago``;
* targets are next-token codes for positions 1..527 across the full parent;
* the second codebook head conditions on a sampled, detached first-code token;
* only predictor parameters are trainable.

The deployment surface is the official public greedy forecast with ``top_k=1`` and
one sample.  Smoke, pilot, full-training, OOS, and trading admissions remain separate.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import sys
from typing import Any, Mapping

import numpy as np
import pandas as pd

from futures_foundation.finetune.native_contracts import get_dossier
from futures_foundation.finetune.native_route_pilot import load_route_pilot_evidence
from futures_foundation.finetune.native_training_schema_v2 import (
    canonical_route_profile_sha256,
)
from futures_foundation.finetune.routes import kronos_tokenizer as tokenizer_route


SUPPORTED_ARMS = ("kronos_mini", "kronos_small")
ARM_KEY = "kronos_mini"  # Backward-compatible default alias.
TRACK = "F"
ROUTE_ID = "hierarchical_autoregressive_tokens"


def route_key(arm_key: str) -> str:
    if arm_key not in SUPPORTED_ARMS:
        raise ValueError(f"unsupported Kronos predictor arm: {arm_key!r}")
    return f"{arm_key}:{TRACK}:{ROUTE_ID}"


def parent_route_key(arm_key: str) -> str:
    return tokenizer_route.route_key(arm_key)


ROUTE_KEY = route_key(ARM_KEY)
PARENT_ROUTE_KEY = parent_route_key(ARM_KEY)
CHECKPOINT_SCHEMA = "ffm_kronos_predictor_training_state_v2"
EXPORT_SCHEMA = "ffm_kronos_predictor_bundle_v2"
CONTEXT_LENGTH = 512
HORIZON_LENGTH = 16
PARENT_LENGTH = CONTEXT_LENGTH + HORIZON_LENGTH
INPUT_CHANNELS = ("open", "high", "low", "close", "volume")
NATIVE_CHANNELS = ("open", "high", "low", "close", "volume", "amount")
STAMP_CHANNELS = ("minute", "hour", "weekday", "day", "month")
VENUE_TIMEZONE = "America/Chicago"


@dataclass(frozen=True)
class RouteConfig:
    arm_key: str = ARM_KEY
    learning_rate: float = 4e-5
    weight_decay: float = 0.1
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    max_gradient_norm: float = 3.0
    total_steps: int = 20
    seed: int = 20260718

    def validate(self) -> None:
        route_key(self.arm_key)
        if not 1e-7 <= float(self.learning_rate) <= 4e-5:
            raise ValueError("Kronos Mini predictor learning_rate is outside the catalog bound")
        if not 0.0 <= float(self.weight_decay) <= 0.1:
            raise ValueError("Kronos Mini predictor weight_decay is outside the catalog bound")
        if not 2 <= int(self.batch_size) <= 32:
            raise ValueError("Kronos Mini predictor batch_size is outside the catalog bound")
        if int(self.gradient_accumulation_steps) != 1:
            raise ValueError("Kronos Mini predictor exact executor supports gradient_accumulation_steps=1 only")
        if not 0.5 <= float(self.max_gradient_norm) <= 5.0:
            raise ValueError("Kronos Mini predictor gradient clipping is outside the catalog bound")
        if not 1 <= int(self.total_steps) <= 32768:
            raise ValueError("Kronos Mini predictor total_steps is outside the catalog bound")


@dataclass
class LoadedRoute:
    tokenizer: Any
    predictor: Any
    public_predictor_class: type
    identity: dict[str, Any]
    device: str
    arm_key: str = ARM_KEY

    @property
    def route_key(self) -> str:
        return route_key(self.arm_key)

    @property
    def parent_route_key(self) -> str:
        return parent_route_key(self.arm_key)


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


def _snapshot_identity(
    path: str | Path,
    expected_revision: str,
    label: str,
) -> tuple[Path, dict[str, Any]]:
    snapshot = Path(path).expanduser().resolve()
    if not snapshot.is_dir() or snapshot.is_symlink() or snapshot.name != expected_revision:
        raise FileNotFoundError(
            f"{label} snapshot is missing, unsafe, or has the wrong revision"
        )
    config = snapshot / "config.json"
    if not config.is_file():
        raise FileNotFoundError(f"{label} snapshot lacks config.json")
    weights = [
        item
        for pattern in ("*.safetensors", "*.bin", "*.pt")
        for item in snapshot.glob(pattern)
        if item.is_file()
    ]
    if not weights:
        raise FileNotFoundError(f"{label} snapshot has no materialized weights")
    return snapshot, {
        "path": str(snapshot),
        "revision": expected_revision,
        "config_sha256": _sha256(config),
        "weight_files": [
            {
                "path": item.name,
                "sha256": _sha256(item),
                "bytes": int(item.stat().st_size),
            }
            for item in sorted(weights)
        ],
    }


def _parent_identity(
    *,
    arm_key: str,
    pilot_evidence_path: str | Path,
    tokenizer_bundle_path: str | Path,
) -> tuple[dict[str, Any], Path]:
    pilot_path = Path(pilot_evidence_path).expanduser().resolve()
    pilot = load_route_pilot_evidence(pilot_path)
    expected_parent_route = parent_route_key(arm_key)
    if (
        pilot["route_key"] != expected_parent_route
        or pilot["pilot_completed"] is not True
        or pilot["native_objective_survived"] is not True
    ):
        raise ValueError(
            "Kronos predictor requires a surviving tokenizer pilot for the same arm"
        )
    bundle_path = Path(tokenizer_bundle_path).expanduser().resolve()
    declared = pilot["artifacts"]["deployment_bundle"]
    if Path(str(declared["path"])).resolve() != bundle_path:
        raise ValueError("Kronos tokenizer bundle differs from pilot evidence")
    if not bundle_path.is_file() or bundle_path.is_symlink():
        raise FileNotFoundError("Kronos tokenizer parent bundle is missing or unsafe")
    bundle_sha = _sha256(bundle_path)
    if declared.get("sha256") != bundle_sha:
        raise ValueError("Kronos tokenizer parent bundle bytes changed")
    return {
        "route_key": expected_parent_route,
        "pilot_evidence_path": str(pilot_path),
        "pilot_evidence_sha256": pilot["evidence_sha256"],
        "tokenizer_bundle_path": str(bundle_path),
        "tokenizer_bundle_sha256": bundle_sha,
        "tokenizer_route_profile_sha256": pilot["route_profile_sha256"],
        "native_objective_survived": True,
    }, bundle_path


def load_route(
    *,
    arm_key: str = ARM_KEY,
    model_snapshot: str | Path,
    tokenizer_snapshot: str | Path,
    source_runtime: str | Path,
    parent_pilot_evidence: str | Path,
    parent_tokenizer_bundle: str | Path,
    device: str,
) -> LoadedRoute:
    route_key(arm_key)
    dossier = get_dossier(arm_key)
    source, source_identity = tokenizer_route.validate_source_runtime(
        source_runtime, arm_key
    )
    model_path, model_identity = _snapshot_identity(
        model_snapshot,
        str(dossier["model_revision"]),
        f"{arm_key} predictor",
    )
    parent_identity, bundle_path = _parent_identity(
        arm_key=arm_key,
        pilot_evidence_path=parent_pilot_evidence,
        tokenizer_bundle_path=parent_tokenizer_bundle,
    )
    tokenizer, tokenizer_bundle = tokenizer_route.load_export_bundle(
        bundle_path,
        arm_key=arm_key,
        model_snapshot=model_snapshot,
        tokenizer_snapshot=tokenizer_snapshot,
        source_runtime=source_runtime,
        device=device,
    )
    if tokenizer_bundle["route_key"] != parent_route_key(arm_key):
        raise ValueError("Kronos tokenizer parent route identity is wrong")
    tokenizer.eval()
    for parameter in tokenizer.parameters():
        parameter.requires_grad_(False)

    sys.path.insert(0, str(source))
    try:
        from model import Kronos, KronosPredictor
    finally:
        sys.path.pop(0)
    imported = Path(inspect.getfile(Kronos)).resolve()
    if source not in imported.parents:
        raise RuntimeError("Kronos predictor did not import from the pinned source runtime")
    predictor = Kronos.from_pretrained(str(model_path)).to(device)
    identity = {
        "arm_key": arm_key,
        "model_id": dossier["model_id"],
        "predictor_snapshot": model_identity,
        "source_runtime": source_identity,
        "tokenizer_parent": parent_identity,
        "tokenizer_identity": dict(tokenizer_bundle["model_identity"]),
    }
    return LoadedRoute(
        arm_key=arm_key,
        tokenizer=tokenizer,
        predictor=predictor,
        public_predictor_class=KronosPredictor,
        identity=identity,
        device=str(device),
    )


def _ohlcv_array(value: Any, *, length: int, label: str) -> np.ndarray:
    values = np.asarray(value, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (int(length), len(INPUT_CHANNELS)):
        raise ValueError(f"{label} must have shape [B,{int(length)},5]")
    if not np.isfinite(values).all():
        raise ValueError(f"{label} rejects missing or non-finite values")
    o, h, l, c, volume = values.transpose(2, 0, 1)
    if np.any(
        (h < np.maximum(o, c))
        | (l > np.minimum(o, c))
        | (h < l)
        | (volume < 0)
    ):
        raise ValueError(f"{label} violates OHLCV geometry")
    return values


def parent_array(value: Any) -> np.ndarray:
    return _ohlcv_array(value, length=PARENT_LENGTH, label="Kronos Mini parent")


def context_array(value: Any) -> np.ndarray:
    return _ohlcv_array(value, length=CONTEXT_LENGTH, label="Kronos Mini context")


def normalize_parent(parent: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = tokenizer_route.native_ohlcva(parent_array(parent))
    context = values[:, :CONTEXT_LENGTH]
    mean = context.mean(axis=1, keepdims=True)
    std = np.maximum(context.std(axis=1, keepdims=True), 1e-5)
    normalized = np.clip((values - mean) / std, -5.0, 5.0).astype(np.float32)
    return normalized, mean.astype(np.float32), std.astype(np.float32)


def model_input(parent: Any, *, device: str) -> Any:
    torch = _torch()
    normalized, _, _ = normalize_parent(parent)
    return torch.as_tensor(normalized, dtype=torch.float32, device=device)


def calendar_stamps(timestamps: Any) -> np.ndarray:
    raw = np.asarray(timestamps)
    if raw.ndim != 2 or raw.shape[1] != PARENT_LENGTH:
        raise ValueError(f"Kronos Mini timestamps must have shape [B,{PARENT_LENGTH}]")
    parsed = pd.DatetimeIndex(pd.to_datetime(raw.reshape(-1), utc=True))
    values_ns = parsed.asi8.reshape(raw.shape)
    if np.any(values_ns == np.iinfo(np.int64).min):
        raise ValueError("Kronos Mini timestamps contain NaT")
    if np.any(np.diff(values_ns, axis=1) <= 0):
        raise ValueError("Kronos Mini timestamps must be strictly increasing")
    venue = parsed.tz_convert(VENUE_TIMEZONE)
    stamps = np.stack(
        (venue.minute, venue.hour, venue.weekday, venue.day, venue.month),
        axis=1,
    ).astype(np.float32)
    return stamps.reshape(raw.shape[0], PARENT_LENGTH, len(STAMP_CHANNELS))


def stamps_array(value: Any) -> np.ndarray:
    stamps = np.asarray(value, dtype=np.float32)
    expected = (stamps.shape[0], PARENT_LENGTH, len(STAMP_CHANNELS)) if stamps.ndim else ()
    if stamps.ndim != 3 or stamps.shape[1:] != (PARENT_LENGTH, len(STAMP_CHANNELS)):
        raise ValueError(f"Kronos Mini stamps must have shape [B,{PARENT_LENGTH},5]")
    if not np.isfinite(stamps).all():
        raise ValueError("Kronos Mini stamps contain non-finite values")
    ranges = ((0, 59), (0, 23), (0, 6), (1, 31), (1, 12))
    for index, (lower, upper) in enumerate(ranges):
        if np.any((stamps[:, :, index] < lower) | (stamps[:, :, index] > upper)):
            raise ValueError(
                f"Kronos Mini stamp channel {STAMP_CHANNELS[index]} is out of range"
            )
    del expected
    return stamps


def native_tokens(loaded: LoadedRoute, parent: Any) -> tuple[Any, Any]:
    torch = _torch()
    values = model_input(parent, device=loaded.device)
    loaded.tokenizer.eval()
    with torch.no_grad():
        coarse, fine = loaded.tokenizer.encode(values, half=True)
    expected = (values.shape[0], PARENT_LENGTH)
    if tuple(coarse.shape) != expected or tuple(fine.shape) != expected:
        raise RuntimeError("Kronos Mini tokenizer parent code geometry drifted")
    return coarse, fine


def native_logits(
    loaded: LoadedRoute,
    parent: Any,
    stamps: Any,
) -> tuple[Any, Any, Any, Any]:
    torch = _torch()
    stamp_values = stamps_array(stamps)
    coarse, fine = native_tokens(loaded, parent)
    stamp_tensor = torch.as_tensor(
        stamp_values[:, :-1], dtype=torch.float32, device=loaded.device
    )
    coarse_in, fine_in = coarse[:, :-1], fine[:, :-1]
    coarse_target, fine_target = coarse[:, 1:], fine[:, 1:]
    coarse_logits, fine_logits = loaded.predictor(
        coarse_in,
        fine_in,
        stamp_tensor,
    )
    if (
        coarse_logits.shape[:2] != coarse_target.shape
        or fine_logits.shape[:2] != fine_target.shape
    ):
        raise RuntimeError("Kronos Mini predictor logit/target geometry drifted")
    if not torch.isfinite(coarse_logits).all() or not torch.isfinite(fine_logits).all():
        raise FloatingPointError("Kronos Mini predictor logits contain non-finite values")
    return coarse_logits, fine_logits, coarse_target, fine_target


def native_loss(loaded: LoadedRoute, parent: Any, stamps: Any) -> Any:
    torch = _torch()
    coarse_logits, fine_logits, coarse_target, fine_target = native_logits(
        loaded, parent, stamps
    )
    loss = loaded.predictor.head.compute_loss(
        coarse_logits,
        fine_logits,
        coarse_target,
        fine_target,
    )[0]
    if not torch.isfinite(loss):
        raise FloatingPointError("Kronos Mini predictor native loss is non-finite")
    return loss


def make_optimizer(predictor: Any, config: RouteConfig) -> Any:
    torch = _torch()
    config.validate()
    return torch.optim.AdamW(
        predictor.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=config.weight_decay,
    )


def make_scheduler(optimizer: Any, config: RouteConfig) -> Any:
    torch = _torch()
    config.validate()
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        total_steps=config.total_steps,
        pct_start=0.03,
        div_factor=10.0,
    )


def predictor_state_cpu(predictor: Any) -> dict[str, Any]:
    return {
        key: value.detach().cpu().clone()
        for key, value in predictor.state_dict().items()
    }


def optimizer_step(
    loaded: LoadedRoute,
    optimizer: Any,
    scheduler: Any,
    parent: Any,
    stamps: Any,
    *,
    max_gradient_norm: float,
) -> dict[str, float]:
    torch = _torch()
    loaded.tokenizer.eval()
    loaded.predictor.train()
    parameter = next(
        value for value in loaded.predictor.parameters() if value.requires_grad
    )
    before = parameter.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    loss = native_loss(loaded, parent, stamps)
    loss.backward()
    grad = torch.sqrt(
        sum(
            value.grad.detach().float().square().sum()
            for value in loaded.predictor.parameters()
            if value.requires_grad and value.grad is not None
        )
    )
    torch.nn.utils.clip_grad_norm_(
        loaded.predictor.parameters(), float(max_gradient_norm)
    )
    optimizer.step()
    scheduler.step()
    delta = torch.max(torch.abs(parameter.detach() - before))
    return {
        "loss": float(loss.detach().cpu()),
        "grad_norm": float(grad.detach().cpu()),
        "parameter_delta": float(delta.detach().cpu()),
    }


def public_greedy_forecast(
    loaded: LoadedRoute,
    context: Any,
    context_timestamps: Any,
    forecast_timestamps: Any,
) -> np.ndarray:
    values = context_array(context)
    context_raw = np.asarray(context_timestamps)
    forecast_raw = np.asarray(forecast_timestamps)
    if context_raw.shape != (len(values), CONTEXT_LENGTH):
        raise ValueError("Kronos Mini context timestamps have the wrong shape")
    if forecast_raw.shape != (len(values), HORIZON_LENGTH):
        raise ValueError("Kronos Mini forecast timestamps have the wrong shape")
    context_utc = pd.to_datetime(context_raw.reshape(-1), utc=True)
    forecast_utc = pd.to_datetime(forecast_raw.reshape(-1), utc=True)
    context_ns = pd.DatetimeIndex(context_utc).asi8.reshape(context_raw.shape)
    forecast_ns = pd.DatetimeIndex(forecast_utc).asi8.reshape(forecast_raw.shape)
    if (
        np.any(np.diff(context_ns, axis=1) <= 0)
        or np.any(np.diff(forecast_ns, axis=1) <= 0)
        or np.any(forecast_ns[:, 0] <= context_ns[:, -1])
    ):
        raise ValueError("Kronos Mini forecast timestamps are not strictly future ordered")

    frames = [
        pd.DataFrame(row, columns=list(INPUT_CHANNELS))
        for row in values
    ]
    context_local = pd.DatetimeIndex(context_utc).tz_convert(VENUE_TIMEZONE)
    forecast_local = pd.DatetimeIndex(forecast_utc).tz_convert(VENUE_TIMEZONE)
    context_series = [
        pd.Series(
            context_local[index * CONTEXT_LENGTH:(index + 1) * CONTEXT_LENGTH]
        )
        for index in range(len(values))
    ]
    forecast_series = [
        pd.Series(
            forecast_local[index * HORIZON_LENGTH:(index + 1) * HORIZON_LENGTH]
        )
        for index in range(len(values))
    ]
    loaded.tokenizer.eval()
    loaded.predictor.eval()
    public = loaded.public_predictor_class(
        loaded.predictor,
        loaded.tokenizer,
        device=loaded.device,
        max_context=CONTEXT_LENGTH,
        clip=5,
    )
    output = public.predict_batch(
        frames,
        context_series,
        forecast_series,
        pred_len=HORIZON_LENGTH,
        T=1.0,
        top_k=1,
        top_p=1.0,
        sample_count=1,
        verbose=False,
    )
    forecast = np.stack(
        [
            frame[list(NATIVE_CHANNELS)].to_numpy(dtype=np.float32)
            for frame in output
        ],
        axis=0,
    )
    if forecast.shape != (len(values), HORIZON_LENGTH, len(NATIVE_CHANNELS)):
        raise RuntimeError("Kronos Mini public forecast geometry drifted")
    if not np.isfinite(forecast).all():
        raise FloatingPointError("Kronos Mini public forecast contains non-finite values")
    return forecast


def capture_training_state(
    *,
    loaded: LoadedRoute,
    optimizer: Any,
    scheduler: Any,
    config: RouteConfig,
    global_step: int,
    sampler_cursor: int,
    history: list[dict[str, Any]],
    sampler_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    torch = _torch()
    config.validate()
    if config.arm_key != loaded.arm_key:
        raise ValueError("Kronos predictor config arm differs from loaded route")
    sampler = dict(
        sampler_state
        or {
            "cursor": int(sampler_cursor),
            "schedule_kind": "explicit_synthetic_batch_v1",
            "schedule_sha256": hashlib.sha256(
                b"explicit_synthetic_batch_v1"
            ).hexdigest(),
        }
    )
    if (
        set(sampler) != {"cursor", "schedule_kind", "schedule_sha256"}
        or int(sampler["cursor"]) != int(sampler_cursor)
    ):
        raise ValueError("Kronos Mini predictor sampler state is invalid")
    return {
        "schema_version": CHECKPOINT_SCHEMA,
        "route_key": loaded.route_key,
        "route_profile_sha256": canonical_route_profile_sha256(
            loaded.arm_key, TRACK, ROUTE_ID
        ),
        "model_identity": dict(loaded.identity),
        "config": asdict(config),
        "model": predictor_state_cpu(loaded.predictor),
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
        "torch_cuda_rng": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
    }


def restore_training_state(
    state: Mapping[str, Any],
    *,
    loaded: LoadedRoute,
    optimizer: Any,
    scheduler: Any,
    config: RouteConfig,
) -> tuple[int, int, list[dict[str, Any]]]:
    torch = _torch()
    config.validate()
    if config.arm_key != loaded.arm_key:
        raise ValueError("Kronos predictor config arm differs from loaded route")
    required = {
        "schema_version",
        "route_key",
        "route_profile_sha256",
        "model_identity",
        "config",
        "model",
        "optimizer",
        "scheduler",
        "scaler",
        "epoch",
        "global_step",
        "sampler",
        "history",
        "python_rng",
        "numpy_rng",
        "torch_cpu_rng",
        "torch_cuda_rng",
    }
    if not isinstance(state, Mapping) or set(state) != required:
        raise ValueError("Kronos Mini predictor training state closure is invalid")
    if (
        state["schema_version"] != CHECKPOINT_SCHEMA
        or state["route_key"] != loaded.route_key
        or state["route_profile_sha256"]
        != canonical_route_profile_sha256(loaded.arm_key, TRACK, ROUTE_ID)
        or dict(state["model_identity"]) != dict(loaded.identity)
        or dict(state["config"]) != asdict(config)
        or state["scaler"] is not None
        or state["epoch"] != 0
    ):
        raise ValueError("Kronos Mini predictor training state identity is stale")
    loaded.predictor.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_cpu_rng"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda_rng"])
    sampler = state["sampler"]
    return (
        int(state["global_step"]),
        int(sampler["cursor"]),
        _history(state["history"]),
    )


def save_training_state(path: str | Path, state: Mapping[str, Any]) -> dict[str, Any]:
    target = _atomic_save(path, dict(state))
    return {
        "path": str(target),
        "sha256": _sha256(target),
        "bytes": int(target.stat().st_size),
    }


def load_training_state(path: str | Path) -> dict[str, Any]:
    return _torch().load(
        Path(path).expanduser().resolve(),
        map_location="cpu",
        weights_only=False,
    )


def build_export_bundle(loaded: LoadedRoute) -> dict[str, Any]:
    return {
        "schema_version": EXPORT_SCHEMA,
        "route_key": loaded.route_key,
        "route_profile_sha256": canonical_route_profile_sha256(
            loaded.arm_key, TRACK, ROUTE_ID
        ),
        "model_identity": dict(loaded.identity),
        "predictor_state": predictor_state_cpu(loaded.predictor),
        "preprocessing": {
            "input_shape": ["batch", CONTEXT_LENGTH, len(INPUT_CHANNELS)],
            "channel_order": list(INPUT_CHANNELS),
            "native_channel_order": list(NATIVE_CHANNELS),
            "amount_source": "volume_times_mean_ohlc_v1",
            "normalization": "first_512_context_mean_std_clip5",
            "timestamp_features": list(STAMP_CHANNELS),
            "venue_timezone": VENUE_TIMEZONE,
            "missing_policy": "reject_parent",
        },
        "training_objective": {
            "target_interval": [1, PARENT_LENGTH],
            "loss": "hierarchical_autoregressive_cross_entropy_full_parent_next_token",
            "sibling_parent_policy": "sampled_s1_detached_native_training",
        },
        "output": {
            "kind": "joint_ohlcva_forecast",
            "shape": ["batch", HORIZON_LENGTH, len(NATIVE_CHANNELS)],
            "prediction_length": HORIZON_LENGTH,
            "temperature": 1.0,
            "top_k": 1,
            "top_p": 1.0,
            "sample_count": 1,
            "hidden_state_contract": "none_forecast_route",
        },
    }


def save_export_bundle(path: str | Path, bundle: Mapping[str, Any]) -> dict[str, Any]:
    target = _atomic_save(path, dict(bundle))
    return {
        "path": str(target),
        "sha256": _sha256(target),
        "bytes": int(target.stat().st_size),
    }


def load_export_bundle(
    path: str | Path,
    *,
    arm_key: str = ARM_KEY,
    model_snapshot: str | Path,
    tokenizer_snapshot: str | Path,
    source_runtime: str | Path,
    parent_pilot_evidence: str | Path,
    parent_tokenizer_bundle: str | Path,
    device: str,
) -> tuple[LoadedRoute, dict[str, Any]]:
    bundle = _torch().load(
        Path(path).expanduser().resolve(),
        map_location="cpu",
        weights_only=False,
    )
    expected = {
        "schema_version",
        "route_key",
        "route_profile_sha256",
        "model_identity",
        "predictor_state",
        "preprocessing",
        "training_objective",
        "output",
    }
    if not isinstance(bundle, Mapping) or set(bundle) != expected:
        raise ValueError("Kronos Mini predictor export bundle closure is invalid")
    if (
        bundle["schema_version"] != EXPORT_SCHEMA
        or bundle["route_key"] != route_key(arm_key)
        or bundle["route_profile_sha256"]
        != canonical_route_profile_sha256(arm_key, TRACK, ROUTE_ID)
    ):
        raise ValueError("Kronos Mini predictor export bundle identity is stale")
    loaded = load_route(
        arm_key=arm_key,
        model_snapshot=model_snapshot,
        tokenizer_snapshot=tokenizer_snapshot,
        source_runtime=source_runtime,
        parent_pilot_evidence=parent_pilot_evidence,
        parent_tokenizer_bundle=parent_tokenizer_bundle,
        device=device,
    )
    if dict(bundle["model_identity"]) != loaded.identity:
        raise ValueError("Kronos Mini predictor export model identity mismatch")
    loaded.predictor.load_state_dict(bundle["predictor_state"], strict=True)
    loaded.predictor.eval()
    return loaded, dict(bundle)


__all__ = [
    "ARM_KEY",
    "CHECKPOINT_SCHEMA",
    "CONTEXT_LENGTH",
    "EXPORT_SCHEMA",
    "HORIZON_LENGTH",
    "INPUT_CHANNELS",
    "LoadedRoute",
    "NATIVE_CHANNELS",
    "PARENT_LENGTH",
    "PARENT_ROUTE_KEY",
    "ROUTE_ID",
    "ROUTE_KEY",
    "SUPPORTED_ARMS",
    "RouteConfig",
    "STAMP_CHANNELS",
    "TRACK",
    "VENUE_TIMEZONE",
    "build_export_bundle",
    "calendar_stamps",
    "capture_training_state",
    "context_array",
    "load_export_bundle",
    "load_route",
    "load_training_state",
    "make_optimizer",
    "make_scheduler",
    "model_input",
    "native_logits",
    "native_loss",
    "native_tokens",
    "normalize_parent",
    "optimizer_step",
    "parent_array",
    "predictor_state_cpu",
    "public_greedy_forecast",
    "restore_training_state",
    "save_export_bundle",
    "save_training_state",
    "seed_everything",
    "stamps_array",
    "parent_route_key",
    "route_key",
]
