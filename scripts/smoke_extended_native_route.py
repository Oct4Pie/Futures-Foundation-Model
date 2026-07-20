#!/usr/bin/env python3
"""Run one newly exact native route on deterministic synthetic data.

This is a smoke-only, non-authorizing command. It reads no market or OOS data.
The snapshot/source paths are recovered from the current sealed native-parity matrix.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from futures_foundation.finetune import ssl_data
from futures_foundation.finetune.native_configuration_audit import load_current_parity_aggregate
from futures_foundation.finetune.native_contract_harness import (
    CheckResult,
    control_rejection_check,
    forward_backward_check,
    future_corruption_check,
    interruption_resume_parity_check,
    loss_decrease_check,
    negative_price_behavior_check,
    parity_check,
    performance_check,
    prefix_invariance_check,
    rejection_check,
)
from futures_foundation.finetune.native_route_smoke import (
    build_route_smoke_evidence,
    validate_route_smoke_evidence,
)
from futures_foundation.finetune.native_training_readiness import _launcher_record
from futures_foundation.finetune.native_smoke_contract import REQUIRED_SMOKE_CHECKS
from futures_foundation.finetune.routes import (
    chronos2_native,
    mantis_native,
    moirai2_research,
    moment_tasks,
    timesfm_lora,
    ttm_native,
)

FIXTURE_SCHEMA = "ffm_extended_exact_route_smoke_fixture_v1"
PARITY_TRACK = {
    "chronos_v2": "F",
    "mantis_v1": "R",
    "mantis_v2": "R",
    "moirai2_small": "F",
    "moment_small": "R",
    "timesfm25": "F",
    "ttm_r2": "F",
}
SUPPORTED = frozenset(
    set(chronos2_native.ROUTES)
    | set(mantis_native.ROUTES)
    | set(moment_tasks.ROUTES)
    | set(ttm_native.ROUTES)
    | {timesfm_lora.ROUTE_KEY, moirai2_research.ROUTE_KEY}
)


def _seed_initialization(seed: int) -> None:
    """Seed every RNG before constructors create task heads or adapter weights."""
    import random
    import torch

    value = int(seed)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
ROUTE_ALIASES = {
    "ttm_head": "ttm_r2:F:head_prefix_raw_hf_trainer_forecast",
    "ttm_full": "ttm_r2:F:full_model_raw_hf_trainer_forecast",
    "timesfm_lora": timesfm_lora.ROUTE_KEY,
    "moirai_research": moirai2_research.ROUTE_KEY,
    "chronos2_full": "chronos_v2:F:official_fit_full",
    "chronos2_lora": "chronos_v2:F:official_fit_lora",
}


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: str | Path, value: object) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(target) + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)
    return target


def _clone_tree(value: Any) -> Any:
    try:
        import torch
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
    except Exception:
        pass
    if isinstance(value, Mapping):
        return {key: _clone_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_clone_tree(item) for item in value)
    if isinstance(value, list):
        return [_clone_tree(item) for item in value]
    return deepcopy(value)


def _numpy(value: Any) -> np.ndarray:
    try:
        import torch
        if isinstance(value, torch.Tensor):
            return value.detach().float().cpu().numpy()
    except Exception:
        pass
    return np.asarray(value)


def _simple(passed: bool, metrics: Mapping[str, Any], reason: str) -> CheckResult:
    return CheckResult(
        status="pass" if passed else "fail",
        metrics=dict(metrics),
        reason=None if passed else reason,
    )


def _fixture(batch: int, length: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    labels = np.arange(batch, dtype=np.int64) % 3
    time = np.arange(length, dtype=np.float64)
    level = 30.0 + 24.0 * labels + rng.uniform(-0.5, 0.5, size=batch)
    slopes = np.asarray([-0.16, 0.06, 0.24], dtype=np.float64)[labels]
    phase = rng.uniform(0.0, 2.0 * np.pi, size=batch)
    close = (
        level[:, None]
        + slopes[:, None] * time[None, :]
        + 0.18 * np.sin(time[None, :] / 13.0 + phase[:, None])
    )
    open_ = np.concatenate((close[:, :1], close[:, :-1]), axis=1)
    spread = 0.04 + 0.01 * np.abs(np.sin(time / 17.0))[None, :]
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = (
        750.0 + 220.0 * labels[:, None] + 0.75 * time[None, :]
        + 8.0 * np.sin(time[None, :] / 19.0 + phase[:, None])
    )
    values = np.stack((open_, high, low, close, volume), axis=-1).astype(np.float32)
    return values, labels


def _negative_fixture(values: np.ndarray) -> np.ndarray:
    result = values.copy()
    result[:, :, :4] -= float(np.max(result[:, :, :4]) + 10.0)
    return result


def _write_fixture(
    directory: Path,
    train: np.ndarray,
    validation: np.ndarray,
    train_labels: np.ndarray,
    validation_labels: np.ndarray,
    *,
    generator_tag: str = "three_regime_trend_sinusoid_ohlcv_v1",
) -> tuple[Path, Path]:
    artifact = directory / "synthetic_fixture.npz"
    np.savez_compressed(
        artifact,
        train=train,
        validation=validation,
        train_labels=train_labels,
        validation_labels=validation_labels,
    )
    manifest = {
        "schema_version": FIXTURE_SCHEMA,
        "generator": str(generator_tag),
        "market_data_read": False,
        "oos_read": False,
        "train_shape": list(train.shape),
        "validation_shape": list(validation.shape),
        "artifact": {
            "path": str(artifact.resolve()),
            "sha256": _sha256(artifact),
            "bytes": int(artifact.stat().st_size),
        },
    }
    return artifact, _atomic_json(directory / "synthetic_fixture.manifest.json", manifest)


def _flag(argv: list[str], name: str) -> str | None:
    if name not in argv:
        return None
    index = argv.index(name)
    if index + 1 >= len(argv):
        raise ValueError(f"parity command flag {name} has no value")
    return argv[index + 1]


def _parity_paths(route_key: str, root: Path) -> dict[str, Path]:
    arm = route_key.split(":", 1)[0]
    track = PARITY_TRACK[arm]
    aggregate_path = root / "native_parity_aggregate.json"
    aggregate = load_current_parity_aggregate(aggregate_path)
    if (arm, track) not in {
        (row["arm_key"], row["track"]) for row in aggregate["bundles"]
    }:
        raise ValueError(f"current parity aggregate lacks {arm}:{track}")
    manifest_path = root / f"{arm}__{track}" / "bundle_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    argv = list(manifest["command"]["argv"])
    values = {
        "manifest": manifest_path,
        "python": Path(argv[0]).resolve(),
        "model_snapshot": Path(str(_flag(argv, "--model-snapshot"))).resolve(),
        "source_runtime": Path(str(_flag(argv, "--source-repo"))).resolve(),
    }
    execution = _flag(argv, "--execution-source")
    if execution is not None:
        values["execution_source"] = Path(execution).resolve()
    return values


class Adapter:
    def __init__(self, route_key: str, paths: Mapping[str, Path], device: str, seed: int):
        if route_key not in SUPPORTED:
            raise ValueError(f"unsupported extended smoke route: {route_key}")
        self.route_key = route_key
        self.arm = route_key.split(":", 1)[0]
        self.paths = dict(paths)
        self.device = str(device)
        self.seed = int(seed)
        self.labels: np.ndarray | None = None
        self.family: str
        _seed_initialization(self.seed)
        if route_key in mantis_native.ROUTES:
            self.family = "mantis"
            self.module = mantis_native
            self.task = mantis_native.ROUTES[route_key]["task"]
            self.batch_size = 6
            self.parent_length = mantis_native.CONTEXT_LENGTH
            self.config = mantis_native.RouteConfig(
                route_key, total_steps=20, batch_size=self.batch_size, n_classes=3, seed=seed,
            )
            self.loaded = mantis_native.load_route(
                route_key,
                model_snapshot=paths["model_snapshot"],
                source_runtime=paths["source_runtime"],
                device=device,
                n_classes=3,
            )
        elif route_key in moment_tasks.ROUTES:
            self.family = "moment"
            self.module = moment_tasks
            self.task = moment_tasks.ROUTES[route_key]["task"]
            self.batch_size = 6 if self.task == "classification" else 4
            self.parent_length = (
                moment_tasks.CONTEXT_LENGTH
                if self.task == "classification"
                else moment_tasks.CONTEXT_LENGTH + moment_tasks.HORIZON_LENGTH
            )
            self.config = moment_tasks.RouteConfig(
                route_key, total_steps=20, batch_size=self.batch_size, n_classes=3, seed=seed,
            )
            self.loaded = moment_tasks.load_route(
                route_key,
                model_snapshot=paths["model_snapshot"],
                source_runtime=paths["source_runtime"],
                device=device,
                n_classes=3,
            )
        elif route_key in ttm_native.ROUTES:
            self.family = "ttm"
            self.module = ttm_native
            self.task = "forecast"
            self.batch_size = 4
            self.parent_length = ttm_native.CONTEXT_LENGTH + ttm_native.HORIZON_LENGTH
            self.config = ttm_native.RouteConfig(
                route_key, total_steps=20, batch_size=self.batch_size, seed=seed,
            )
            self.loaded = ttm_native.load_route(
                route_key,
                model_snapshot=paths["model_snapshot"],
                source_runtime=paths["source_runtime"],
                device=device,
            )
        elif route_key == timesfm_lora.ROUTE_KEY:
            self.family = "timesfm"
            self.module = timesfm_lora
            self.task = "forecast"
            self.batch_size = 2
            self.parent_length = timesfm_lora.CONTEXT_LENGTH + timesfm_lora.HORIZON_LENGTH
            self.config = timesfm_lora.RouteConfig(total_steps=20, batch_size=2, seed=seed)
            self.config.validate()
            self.loaded = timesfm_lora.load_route(
                model_snapshot=paths["model_snapshot"],
                source_runtime=paths["source_runtime"],
                execution_source=paths["execution_source"],
                device=device,
            )
        elif route_key in chronos2_native.ROUTES:
            self.family = "chronos2"
            self.module = chronos2_native
            self.task = "forecast"
            self.batch_size = 2
            self.parent_length = chronos2_native.CONTEXT_LENGTH + chronos2_native.HORIZON_LENGTH
            smoke_learning_rate = (
                1e-6 if chronos2_native.ROUTES[route_key]["surface"] == "full" else 1e-5
            )
            self.config = chronos2_native.RouteConfig(
                route_key,
                total_steps=20,
                batch_size=2,
                learning_rate=smoke_learning_rate,
                seed=seed,
            )
            self.config.resolved()
            self.loaded = chronos2_native.load_route(
                route_key, model_snapshot=paths["model_snapshot"], device=device,
            )
        elif route_key == moirai2_research.ROUTE_KEY:
            self.family = "moirai"
            self.module = moirai2_research
            self.task = "forecast"
            self.batch_size = 2
            self.parent_length = moirai2_research.CONTEXT_LENGTH + moirai2_research.HORIZON_LENGTH
            self.config = moirai2_research.RouteConfig(total_steps=20, batch_size=2, seed=seed)
            self.config.validate()
            self.loaded = moirai2_research.load_route(
                model_snapshot=paths["model_snapshot"],
                source_runtime=paths["source_runtime"],
                device=device,
            )
        else:  # pragma: no cover
            raise AssertionError(route_key)
        self.initial_modules = {
            name: _clone_tree(module.state_dict())
            for name, module in self.loaded.modules.items()
        }

    @property
    def is_forecast(self) -> bool:
        return self.task == "forecast"

    @property
    def is_classification(self) -> bool:
        return self.task == "classification"

    @property
    def is_contrastive(self) -> bool:
        return self.task == "contrastive"

    def seed_all(self, seed: int) -> None:
        self.module.seed_everything(int(seed))

    def reset(self) -> None:
        for name, module in self.loaded.modules.items():
            module.load_state_dict(_clone_tree(self.initial_modules[name]), strict=True)
        self.eval_mode()

    def eval_mode(self) -> None:
        for name in ("model", "backbone", "head", "module", "forecast"):
            value = getattr(self.loaded, name, None)
            if value is not None and hasattr(value, "eval"):
                value.eval()

    def make_optimizer(self) -> Any:
        return self.module.make_optimizer(self.loaded, self.config)

    def make_scheduler(self, optimizer: Any) -> Any:
        return self.module.make_scheduler(optimizer, self.config)

    def loss(self, parent: np.ndarray, labels: np.ndarray | None = None) -> float:
        self.eval_mode()
        with __import__("torch").no_grad():
            if self.family == "mantis":
                value = self.module.native_loss(
                    self.loaded, parent, device=self.device, labels=labels,
                )
            elif self.family == "moment":
                value = self.module.native_loss(self.loaded, parent, labels=labels)
            elif self.family == "ttm":
                value = self.module.native_loss(self.loaded, parent, timeframe="1min")
            else:
                value = self.module.native_loss(self.loaded, parent)
        return float(value.detach().cpu())

    def step(
        self,
        optimizer: Any,
        scheduler: Any,
        parent: np.ndarray,
        labels: np.ndarray | None = None,
    ) -> dict[str, float]:
        if self.family == "mantis":
            return self.module.optimizer_step(
                self.loaded, optimizer, scheduler, parent,
                device=self.device, config=self.config, labels=labels,
            )
        if self.family == "moment":
            return self.module.optimizer_step(
                self.loaded, optimizer, scheduler, parent,
                config=self.config, labels=labels,
            )
        if self.family == "ttm":
            return self.module.optimizer_step(
                self.loaded, optimizer, scheduler, parent,
                timeframe="1min", config=self.config,
            )
        return self.module.optimizer_step(
            self.loaded, optimizer, scheduler, parent, config=self.config,
        )

    def output(self, parent: np.ndarray) -> np.ndarray:
        self.eval_mode()
        with __import__("torch").no_grad():
            if self.family == "mantis":
                result = self.module.deployment_output(
                    self.loaded, parent, device=self.device,
                )
            elif self.family == "moment":
                visible = parent if self.is_classification else parent[:, :moment_tasks.CONTEXT_LENGTH]
                result = self.module.deployment_output(self.loaded, visible)
            elif self.family == "ttm":
                result = self.module.native_output(
                    self.loaded, parent[:, :ttm_native.CONTEXT_LENGTH], timeframe="1min",
                )
            elif self.family == "timesfm":
                point, quantiles = self.module.public_output(
                    self.loaded, parent[:, :timesfm_lora.CONTEXT_LENGTH],
                )
                result = __import__("torch").cat((point[..., None], quantiles), dim=-1)
            elif self.family == "chronos2":
                result = self.module.native_quantiles(
                    self.loaded, parent[:, :chronos2_native.CONTEXT_LENGTH],
                )
            else:
                result = self.module.public_output(
                    self.loaded, parent[:, :moirai2_research.CONTEXT_LENGTH],
                )
        output = _numpy(result)
        if not np.isfinite(output).all():
            raise FloatingPointError("route output is non-finite")
        return output

    def capture(
        self,
        optimizer: Any,
        scheduler: Any,
        *,
        step: int,
        history: list[Mapping[str, Any]],
    ) -> dict[str, Any]:
        return self.module.capture_training_state(
            self.loaded, optimizer, scheduler, self.config,
            global_step=step, sampler_cursor=step, history=list(history),
        )

    def restore(self, state: Mapping[str, Any], optimizer: Any, scheduler: Any) -> tuple[Any, ...]:
        return tuple(
            self.module.restore_training_state(
                state, self.loaded, optimizer, scheduler, self.config,
            )
        )

    def build_export(self) -> dict[str, Any]:
        if self.family in {"mantis", "moment"}:
            return self.module.build_export_bundle(self.loaded, self.config)
        return self.module.build_export_bundle(self.loaded)

    def save(self, path: str | Path, value: Any) -> dict[str, Any]:
        return self.module.save_artifact(path, value)

    def load(self, path: str | Path) -> Any:
        return self.module.load_artifact(path)

    def load_export(self, path: str | Path) -> dict[str, Any]:
        _, bundle = self.module.load_export_bundle(path, loaded=self.loaded)
        return bundle

    def shuffled_contrastive_loss(self, parent: np.ndarray) -> float:
        if self.family != "mantis" or not self.is_contrastive:
            raise ValueError("shuffled contrastive control is Mantis-only")
        torch = __import__("torch")
        self.eval_mode()
        value = self.module.parent_tensor(parent, device=self.device)
        batch = value.shape[0]
        flattened = value.permute(0, 2, 1).reshape(
            batch * len(self.module.CHANNELS), 1, self.module.CONTEXT_LENGTH,
        )
        first, second = self.module._augment_views(
            flattened, source_runtime=self.loaded.identity["source_runtime"],
        )
        q = self.loaded.backbone(first).reshape(batch, len(self.module.CHANNELS), -1)
        k = self.loaded.backbone(second).reshape(batch, len(self.module.CHANNELS), -1)
        labels = torch.arange(batch, device=value.device)
        losses = []
        for channel in range(len(self.module.CHANNELS)):
            query = torch.nn.functional.normalize(q[:, channel], dim=1)
            key = torch.nn.functional.normalize(k[:, channel], dim=1).roll(1, dims=0)
            losses.append(torch.nn.functional.cross_entropy(query @ key.T / 0.1, labels))
        return float(torch.stack(losses).mean().detach().cpu())

    def missing_check(self, validation: np.ndarray) -> CheckResult:
        missing = validation.copy()
        missing[:, :16, :] = np.nan
        if self.family in {"mantis", "timesfm"}:
            try:
                self.output(missing)
            except Exception as exc:
                return _simple(
                    True,
                    {"behavior": "explicit_reject", "error_type": type(exc).__name__},
                    "",
                )
            return _simple(False, {"behavior": "explicit_reject"}, "route accepted unsupported missing data")
        if self.family == "ttm":
            try:
                loss = self.loss(missing)
                finite = np.isfinite(loss)
            except Exception as exc:
                return _simple(False, {"error_type": type(exc).__name__}, "TTM observed mask failed")
            return _simple(finite, {"behavior": "native_observed_mask", "loss": loss}, "TTM mask was non-finite")
        try:
            output = self.output(missing)
        except Exception as exc:
            return _simple(False, {"error_type": type(exc).__name__}, "native missing mask raised")
        return _simple(
            bool(np.isfinite(output).all()),
            {"behavior": "native_mask", "shape": list(output.shape), "finite": True},
            "native missing mask produced non-finite output",
        )

    def grouping_check(self, validation: np.ndarray) -> CheckResult:
        if self.family == "mantis":
            baseline = _numpy(self.module.channel_embeddings(
                self.loaded, validation, device=self.device,
            ))
            changed_input = validation.copy(); changed_input[:, :, 4] *= 1.1
            changed = _numpy(self.module.channel_embeddings(
                self.loaded, changed_input, device=self.device,
            ))
            unaffected = float(np.max(np.abs(baseline[:, :4] - changed[:, :4])))
            affected = float(np.max(np.abs(baseline[:, 4] - changed[:, 4])))
            passed = baseline.shape[1] == 5 and unaffected == 0.0 and affected > 0.0
            return _simple(passed, {
                "layout": "independent_univariate_passes",
                "embedding_shape": list(baseline.shape),
                "unaffected_max_abs": unaffected,
                "affected_max_change": affected,
            }, "Mantis channel grouping drifted")
        if self.family == "moment" and self.is_classification:
            x, mask = self.module.model_input(validation, device=self.device)
            passed = tuple(x.shape[1:]) == (5, 512) and tuple(mask.shape[1:]) == (512,)
            return _simple(passed, {"model_input_shape": list(x.shape), "mask_shape": list(mask.shape)}, "MOMENT classification grouping drifted")
        if self.family in {"moment", "ttm", "timesfm"}:
            baseline = self.output(validation)
            changed_input = validation.copy(); changed_input[:, :512, 4] *= 1.1
            changed = self.output(changed_input)
            if self.family == "timesfm":
                shape = (len(validation), 5, *baseline.shape[1:])
                baseline = baseline.reshape(shape); changed = changed.reshape(shape)
                channel_axis = 1
            elif self.family == "ttm":
                channel_axis = 2
            else:
                channel_axis = 1
            unaffected = [0, 1, 2, 3]
            left = np.take(baseline, unaffected, axis=channel_axis)
            right = np.take(changed, unaffected, axis=channel_axis)
            affected_left = np.take(baseline, [4], axis=channel_axis)
            affected_right = np.take(changed, [4], axis=channel_axis)
            unaffected_error = float(np.max(np.abs(left - right)))
            affected_change = float(np.max(np.abs(affected_left - affected_right)))
            return _simple(
                unaffected_error <= 1e-6 and affected_change > 0.0,
                {"unaffected_max_abs": unaffected_error, "affected_max_change": affected_change},
                "channel-independent output grouping drifted",
            )
        if self.family == "chronos2":
            context, _, _, groups = self.module.split_parent(validation)
            expected = np.repeat(np.arange(len(validation)), 5)
            actual = _numpy(groups)
            return _simple(
                context.shape == (len(validation) * 5, 512) and np.array_equal(actual, expected),
                {"flattened_context_shape": list(context.shape), "group_ids": actual.tolist()},
                "Chronos-2 grouped multivariate IDs drifted",
            )
        packed = self.module.packed_training_tensors(self.loaded, validation)
        variates = _numpy(packed["variate_id"])
        prediction = _numpy(packed["prediction_mask"])
        passed = variates.shape == prediction.shape and set(np.unique(variates).tolist()) == set(range(5))
        return _simple(
            passed,
            {"packed_shape": list(variates.shape), "variates": sorted(np.unique(variates).tolist())},
            "Moirai packed variate grouping drifted",
        )


def _boundary_result(kind: str, parent_length: int) -> CheckResult:
    expected = pd.Timedelta("1min")
    length = max(640, parent_length + 112)
    timestamps = pd.date_range("2024-01-01", periods=length, freq=expected, tz="UTC")
    split = length // 2

    def validate(case: Mapping[str, Any]) -> None:
        starts = ssl_data.window_starts(
            np.asarray(case["indices"], np.int64),
            parent_length,
            timestamps=case["timestamps"],
            expected_delta=expected,
            segment_ids=case.get("segments"),
        )
        if not len(starts):
            raise ValueError(f"invalid {kind} case was rejected as required")

    if kind == "contract_roll":
        case = {
            "indices": np.arange(length), "timestamps": timestamps,
            "segments": np.asarray(["A"] * split + ["B"] * (length - split)),
        }
    elif kind == "session_gap":
        shifted = timestamps[:split].append(timestamps[split:] + pd.Timedelta("1h"))
        case = {
            "indices": np.arange(length), "timestamps": shifted,
            "segments": np.asarray(["A"] * length),
        }
    elif kind == "split_boundary":
        raw_length = length + 80
        case = {
            "indices": np.r_[np.arange(0, split), np.arange(split + 80, raw_length)],
            "timestamps": pd.date_range("2024-01-01", periods=raw_length, freq=expected, tz="UTC"),
            "segments": np.asarray(["A"] * raw_length),
        }
    elif kind == "oos_boundary":
        eligible = np.arange(min(split, parent_length - 1))
        case = {
            "indices": eligible, "timestamps": timestamps,
            "segments": np.asarray(["A"] * length),
        }
    else:  # pragma: no cover
        raise ValueError(kind)
    return rejection_check(validate, {kind: case})


def _controls(
    adapter: Adapter,
    validation: np.ndarray,
    labels: np.ndarray,
) -> tuple[CheckResult, CheckResult, dict[str, float]]:
    rng = np.random.default_rng(adapter.seed + 91)
    adapter.seed_all(adapter.seed + 501)
    real = adapter.loss(validation, labels if adapter.is_classification else None)
    if adapter.is_classification:
        shuffled_labels = np.roll(labels, 1)
        shuffled = adapter.loss(validation, shuffled_labels)
    elif adapter.is_contrastive:
        adapter.seed_all(adapter.seed + 501)
        shuffled = adapter.shuffled_contrastive_loss(validation)
    else:
        shuffled_parent = validation.copy()
        shuffled_parent[:, 512:] = shuffled_parent[np.roll(np.arange(len(validation)), 1), 512:]
        shuffled = adapter.loss(shuffled_parent)
    destroyed = validation.copy()
    destroyed[:, :512] = destroyed[:, rng.permutation(512)]
    adapter.seed_all(adapter.seed + 501)
    time_destroyed = adapter.loss(
        destroyed, labels if adapter.is_classification else None,
    )
    result = control_rejection_check(
        real, [shuffled], [time_destroyed], margin=0.0, higher_is_better=False,
    )
    shuffle_result = _simple(
        result.status == "pass" and real <= shuffled,
        dict(result.metrics),
        "real route did not reject the shuffled control",
    )
    time_result = _simple(
        result.status == "pass" and real <= time_destroyed,
        dict(result.metrics),
        "real route did not reject the time-destroyed control",
    )
    return shuffle_result, time_result, {
        "real_validation_loss": real,
        "shuffle_validation_loss": shuffled,
        "time_destroyed_validation_loss": time_destroyed,
    }


def _resume_result(
    adapter: Adapter,
    train: np.ndarray,
    labels: np.ndarray,
    path: Path,
) -> CheckResult:
    steps = 4

    def uninterrupted() -> dict[str, Any]:
        adapter.reset(); adapter.seed_all(adapter.seed + 31)
        optimizer = adapter.make_optimizer(); scheduler = adapter.make_scheduler(optimizer)
        history = []
        for step in range(steps):
            row = adapter.step(
                optimizer, scheduler, train,
                labels if adapter.is_classification else None,
            )
            history.append({"step": step + 1, **row})
        return adapter.capture(optimizer, scheduler, step=steps, history=history)

    def resumed() -> dict[str, Any]:
        adapter.reset(); adapter.seed_all(adapter.seed + 31)
        optimizer = adapter.make_optimizer(); scheduler = adapter.make_scheduler(optimizer)
        history = []
        for step in range(2):
            row = adapter.step(
                optimizer, scheduler, train,
                labels if adapter.is_classification else None,
            )
            history.append({"step": step + 1, **row})
        partial = adapter.capture(optimizer, scheduler, step=2, history=history)
        adapter.save(path, partial)
        adapter.reset()
        resumed_optimizer = adapter.make_optimizer(); resumed_scheduler = adapter.make_scheduler(resumed_optimizer)
        restored = adapter.restore(adapter.load(path), resumed_optimizer, resumed_scheduler)
        if tuple(restored[1:3]) != (2, 2):
            raise RuntimeError("resume global step/cursor changed")
        resumed_history = list(restored[-1])
        for step in range(2, steps):
            row = adapter.step(
                resumed_optimizer, resumed_scheduler, train,
                labels if adapter.is_classification else None,
            )
            resumed_history.append({"step": step + 1, **row})
        return adapter.capture(
            resumed_optimizer, resumed_scheduler, step=steps, history=resumed_history,
        )

    return interruption_resume_parity_check(uninterrupted, resumed, atol=0.0, rtol=0.0)


def run(args: argparse.Namespace) -> dict[str, Any]:
    route_key = str(args.route_key or ROUTE_ALIASES[args.route_alias])
    output = Path(args.output).expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"smoke output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    paths = _parity_paths(route_key, Path(args.parity_root).expanduser().resolve())
    adapter = Adapter(route_key, paths, args.device, args.seed)
    train, train_labels = _fixture(adapter.batch_size, adapter.parent_length, adapter.seed)
    validation, validation_labels = _fixture(
        adapter.batch_size, adapter.parent_length, adapter.seed + 1,
    )
    fixture_generator = "three_regime_trend_sinusoid_ohlcv_v1"
    if adapter.family == "chronos2":
        for values in (train, validation):
            values[:, 512:, :4] += 20.0
            values[:, 512:, 4] += 500.0
        fixture_generator = "chronos2_future_level_volume_shift_stress_v1"
    fixture_path, fixture_manifest_path = _write_fixture(
        output,
        train,
        validation,
        train_labels,
        validation_labels,
        generator_tag=fixture_generator,
    )
    checks: dict[str, CheckResult] = {}

    adapter.reset(); adapter.seed_all(adapter.seed)
    one_optimizer = adapter.make_optimizer(); one_scheduler = adapter.make_scheduler(one_optimizer)
    checks["one_batch_forward_backward"] = forward_backward_check(
        lambda: adapter.step(
            one_optimizer, one_scheduler, train,
            train_labels if adapter.is_classification else None,
        )
    )

    adapter.reset(); adapter.seed_all(adapter.seed + 7)
    optimizer = adapter.make_optimizer(); scheduler = adapter.make_scheduler(optimizer)
    step_counter = {"value": 0}

    def evaluate_loss() -> float:
        adapter.seed_all(adapter.seed + 101)
        return adapter.loss(train, train_labels if adapter.is_classification else None)

    def train_step() -> dict[str, float]:
        step_counter["value"] += 1
        adapter.seed_all(adapter.seed + 1000 + step_counter["value"])
        return adapter.step(
            optimizer, scheduler, train,
            train_labels if adapter.is_classification else None,
        )

    checks["controlled_learnable_loss_decrease"] = loss_decrease_check(
        evaluate_loss, train_step, steps=20, min_relative_decrease=0.001, tail=3,
    )
    shuffle_result, time_result, control_metrics = _controls(
        adapter, validation, validation_labels,
    )
    checks["shuffle_control_rejection"] = shuffle_result
    checks["time_destroyed_control_rejection"] = time_result

    history = [
        {"step": index + 1, "train_loss": float(value)}
        for index, value in enumerate(
            checks["controlled_learnable_loss_decrease"].metrics.get("losses", [])[1:]
        )
    ]
    training_state = adapter.capture(optimizer, scheduler, step=20, history=history)
    checkpoint_path = output / "training_state.pt"
    checkpoint_identity = adapter.save(checkpoint_path, training_state)
    export_path = output / "deployment_bundle.pt"
    export_identity = adapter.save(export_path, adapter.build_export())
    reference_output = adapter.output(validation)

    interrupted_path = output / "interrupted_state.pt"
    checks["exact_interruption_resume_trajectory"] = _resume_result(
        adapter, train, train_labels, interrupted_path,
    )

    adapter.reset()
    raw_export_bundle = adapter.load(export_path)
    reopened_bundle = adapter.load_export(export_path)
    exported_output = adapter.output(validation)
    checks["training_exported_inference_parity"] = parity_check(
        reference_output, exported_output, atol=0.0, rtol=0.0,
        name="training/exported route output",
    )

    if adapter.is_forecast:
        changed_future = validation.copy()
        changed_future[:, 512:, :4] += 50.0
        changed_future[:, 512:, 4] *= 2.0
        checks["prefix_invariance"] = prefix_invariance_check(
            lambda parent: parent[:, :512],
            validation, changed_future, prefix_length=512, atol=0.0, rtol=0.0,
        )
        checks["future_corruption_invariance"] = future_corruption_check(
            adapter.output, validation, changed_future,
            visible_length=512, atol=0.0, rtol=0.0,
        )
    else:
        checks["prefix_invariance"] = _simple(
            True, {"not_applicable": True, "reason": "route consumes exactly one 512-bar context"}, "",
        )
        checks["future_corruption_invariance"] = _simple(
            True, {"not_applicable": True, "reason": "route has no hidden future input"}, "",
        )

    checks["contract_roll_rejection"] = _boundary_result("contract_roll", adapter.parent_length)
    checks["session_gap_rejection"] = _boundary_result("session_gap", adapter.parent_length)
    checks["split_boundary_rejection"] = _boundary_result("split_boundary", adapter.parent_length)
    checks["oos_boundary_rejection"] = _boundary_result("oos_boundary", adapter.parent_length)
    checks["multivariate_channel_grouping"] = adapter.grouping_check(validation)
    checks["native_missing_data_mask"] = adapter.missing_check(validation)

    torch = __import__("torch")
    if torch.cuda.is_available() and args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(args.device)
        memory_probe = lambda: int(torch.cuda.max_memory_allocated(args.device))
    else:
        memory_probe = None
    performance = performance_check(
        lambda: adapter.output(validation),
        batch_size=adapter.batch_size,
        repeats=2,
        warmups=1,
        min_examples_per_second=0.01,
        memory_probe=memory_probe,
    )
    checks["memory_measurement"] = performance
    checks["throughput_measurement"] = performance
    checks["negative_price_behavior"] = negative_price_behavior_check(
        adapter.output, _negative_fixture(validation), behavior="support",
    )

    full = adapter.output(validation)
    split = adapter.batch_size // 2
    partitioned = np.concatenate(
        [adapter.output(validation[:split]), adapter.output(validation[split:])], axis=0,
    )
    checks["native_output_parity"] = parity_check(
        full, partitioned, atol=1e-5, rtol=1e-5,
        name="full/partitioned native output",
    )

    reopened_state = adapter.load(checkpoint_path)
    checks["checkpoint_lineage"] = _simple(
        reopened_state.get("schema_version") == adapter.module.CHECKPOINT_SCHEMA
        and reopened_state.get("route_key") == route_key
        and checkpoint_identity["sha256"] == _sha256(checkpoint_path)
        and raw_export_bundle.get("schema_version") == adapter.module.EXPORT_SCHEMA
        and raw_export_bundle.get("route_key") == route_key
        and reopened_bundle.get("output") == raw_export_bundle.get("output")
        and export_identity["sha256"] == _sha256(export_path),
        {
            "checkpoint": checkpoint_identity,
            "export": export_identity,
            "state_fields": sorted(reopened_state),
        },
        "checkpoint/export lineage is incomplete or stale",
    )
    fixture_document = json.loads(fixture_manifest_path.read_text(encoding="utf-8"))
    checks["data_lineage"] = _simple(
        fixture_document["artifact"]["sha256"] == _sha256(fixture_path)
        and fixture_document["market_data_read"] is False
        and fixture_document["oos_read"] is False,
        {
            "fixture_path": str(fixture_path),
            "fixture_sha256": _sha256(fixture_path),
            "market_data_read": False,
            "oos_read": False,
        },
        "synthetic fixture lineage is invalid",
    )

    if set(checks) != set(REQUIRED_SMOKE_CHECKS):
        raise RuntimeError("extended smoke check closure drifted")
    raw_checks = {name: result.manifest() for name, result in checks.items()}
    metrics = {
        **control_metrics,
        "initial_loss": checks["controlled_learnable_loss_decrease"].metrics.get("initial_loss"),
        "final_loss": checks["controlled_learnable_loss_decrease"].metrics.get("final_tail_mean_loss"),
        "all_checks_pass": all(result.status == "pass" for result in checks.values()),
    }
    raw_path = _atomic_json(output / "raw_checks.json", {
        "schema_version": "ffm_extended_exact_route_smoke_raw_v1",
        "route_key": route_key,
        "config": (
            adapter.config.resolved()
            if hasattr(adapter.config, "resolved")
            else vars(adapter.config)
        ),
        "checks": raw_checks,
        "metrics": metrics,
    })
    evidence = build_route_smoke_evidence(
        route_key=route_key,
        executor_path=Path(adapter.module.__file__).resolve(),
        executor_entrypoint=_launcher_record(route_key)["entrypoint"],
        checks=raw_checks,
        artifacts={
            "model_snapshot": paths["model_snapshot"],
            "source_runtime": paths["source_runtime"],
            "synthetic_fixture": fixture_path,
            "synthetic_fixture_manifest": fixture_manifest_path,
            "interrupted_state": interrupted_path,
            "training_state": checkpoint_path,
            "deployment_bundle": export_path,
            "raw_checks": raw_path,
            "smoke_runner": Path(__file__).resolve(),
        },
        metrics=metrics,
    )
    validate_route_smoke_evidence(evidence)
    evidence_path = _atomic_json(output / "smoke_evidence.json", evidence)
    return {
        "status": "pass" if evidence["smoke_admitted"] else "fail",
        "route_key": route_key,
        "smoke_admitted": evidence["smoke_admitted"],
        "pilot_admitted": False,
        "training_admitted": False,
        "evidence": {
            "path": str(evidence_path),
            "sha256": _sha256(evidence_path),
            "content_sha256": evidence["evidence_sha256"],
        },
        "metrics": metrics,
        "failed_checks": [name for name, result in checks.items() if result.status != "pass"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    route_group = parser.add_mutually_exclusive_group(required=True)
    route_group.add_argument("--route-key", choices=sorted(SUPPORTED))
    route_group.add_argument("--route-alias", choices=sorted(ROUTE_ALIASES))
    parser.add_argument(
        "--parity-root",
        default=str(ROOT / "output" / "native_parity_evidence_current_config_v1"),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
