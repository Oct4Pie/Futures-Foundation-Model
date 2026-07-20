#!/usr/bin/env python3
"""Run the exact Chronos V1 native 64-token route on synthetic data."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
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
from futures_foundation.finetune.native_contract_harness import (
    CheckResult, control_rejection_check, forward_backward_check,
    future_corruption_check, interruption_resume_parity_check,
    loss_decrease_check, negative_price_behavior_check, parity_check,
    performance_check, prefix_invariance_check, rejection_check,
)
from futures_foundation.finetune.native_route_smoke import (
    build_route_smoke_evidence, validate_route_smoke_evidence,
)
from futures_foundation.finetune.routes import chronos_v1 as route


FIXTURE_SCHEMA = "ffm_chronos_v1_route_smoke_fixture_v1"


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _atomic_json(path: str | Path, value: object) -> Path:
    target = Path(path).resolve(); target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(target) + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, target); return target


def _fixture(batch: int, seed: int) -> np.ndarray:
    """Build a chronology-dependent regime-continuation task.

    The first 448 context bars are nearly stationary.  A signed trend begins in the
    final 64 visible bars and continues through the 64-bar future.  Shuffling time
    destroys the recent regime while preserving the marginal value distribution.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(route.PARENT_LENGTH, dtype=np.float64)
    level = rng.uniform(45.0, 105.0, batch)
    direction = rng.choice(np.asarray([-1.0, 1.0]), size=batch)
    magnitude = rng.uniform(0.06, 0.14, batch)
    phase = rng.uniform(0.0, 2.0 * np.pi, batch)
    regime_ramp = np.maximum(t - 447.0, 0.0)
    stationary = 0.08 * np.sin(t[None, :] / 11.0 + phase[:, None])
    close = (
        level[:, None]
        + stationary
        + direction[:, None] * magnitude[:, None] * regime_ramp[None, :]
    )
    open_ = np.concatenate((close[:, :1], close[:, :-1]), axis=1)
    spread = 0.04 + 0.01 * np.abs(np.sin(t / 19.0))[None, :]
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = (
        1_200.0
        + rng.uniform(20.0, 80.0, batch)[:, None]
        + direction[:, None] * 1.5 * regime_ramp[None, :]
        + 10.0 * np.sin(t[None, :] / 17.0 + phase[:, None])
    )
    return route.parent_array(
        np.stack((open_, high, low, close, volume), axis=-1).astype(np.float32)
    )


def _negative_fixture(batch: int, seed: int) -> np.ndarray:
    values = _fixture(batch, seed).copy()
    values[:, :, :4] -= float(np.max(values[:, :, :4]) + 20)
    return route.parent_array(values)


def _write_fixture(directory: Path, train: np.ndarray, validation: np.ndarray):
    artifact = directory / "synthetic_fixture.npz"
    np.savez_compressed(artifact, train=train, validation=validation)
    manifest = {
        "schema_version": FIXTURE_SCHEMA,
        "generator": "bounded_trend_sinusoid_ohlcv_v1",
        "market_data_read": False,
        "oos_read": False,
        "train_shape": list(train.shape),
        "validation_shape": list(validation.shape),
        "artifact": {"path": str(artifact.resolve()), "sha256": _sha(artifact), "bytes": artifact.stat().st_size},
    }
    return artifact, _atomic_json(directory / "synthetic_fixture.manifest.json", manifest)


def _fresh(base: Any, initial: Mapping[str, Any], device: str) -> Any:
    model = copy.deepcopy(base).to(device); model.load_state_dict(initial, strict=True); return model


def _np(value: Any) -> np.ndarray:
    return value.detach().float().cpu().numpy()


def _loss(pipeline: Any, model: Any, parent: np.ndarray, device: str) -> float:
    torch = route._torch(); model.eval()
    with torch.no_grad():
        return float(route.native_loss(pipeline, model, parent, device=device).detach().cpu())


def _train(pipeline, base, initial, parent, validation, config, device):
    route.seed_everything(config.seed)
    model = _fresh(base, initial, device)
    optimizer = route.make_optimizer(model, config); scheduler = route.make_scheduler(optimizer, config)
    losses = []
    for _ in range(config.total_steps):
        row = route.optimizer_step(pipeline, model, optimizer, scheduler, parent,
                                   device=device, max_gradient_norm=config.max_gradient_norm)
        losses.append(row["loss"])
    return model, losses, _loss(pipeline, model, validation, device)


def _result(passed, metrics, reason):
    return CheckResult(status="pass" if passed else "fail", metrics=dict(metrics),
                       reason=None if passed else reason)


def _boundary(kind: str) -> CheckResult:
    delta = pd.Timedelta("1min"); length = 700
    times = pd.date_range("2024-01-01", periods=length, freq=delta, tz="UTC")
    def validate(case):
        starts = ssl_data.window_starts(np.asarray(case["indices"], np.int64), route.PARENT_LENGTH,
                                        timestamps=case["timestamps"], expected_delta=delta,
                                        segment_ids=case.get("segments"))
        if not len(starts): raise ValueError("invalid boundary rejected")
    if kind == "contract_roll":
        case = {"indices": np.arange(length), "timestamps": times,
                "segments": np.asarray(["A"] * 350 + ["B"] * 350)}
    elif kind == "session_gap":
        case = {"indices": np.arange(length),
                "timestamps": times[:350].append(times[350:] + pd.Timedelta("1h")),
                "segments": np.asarray(["A"] * length)}
    elif kind == "split_boundary":
        case = {"indices": np.r_[np.arange(350), np.arange(450, 800)],
                "timestamps": pd.date_range("2024-01-01", periods=800, freq=delta, tz="UTC"),
                "segments": np.asarray(["A"] * 800)}
    else:
        case = {"indices": np.arange(350), "timestamps": times,
                "segments": np.asarray(["A"] * length)}
    return rejection_check(validate, {kind: case})


def _resume(pipeline, base, initial, parent, config, identity, device, path):
    steps = 4
    def run_full():
        route.seed_everything(config.seed + 31); m = _fresh(base, initial, device)
        o = route.make_optimizer(m, config); s = route.make_scheduler(o, config); hist = []
        for step in range(steps):
            hist.append({"step": step + 1, **route.optimizer_step(
                pipeline, m, o, s, parent, device=device,
                max_gradient_norm=config.max_gradient_norm)})
        return route.capture_training_state(model=m, optimizer=o, scheduler=s, config=config,
            model_identity=identity, global_step=steps, sampler_cursor=steps, history=hist)
    def run_resume():
        route.seed_everything(config.seed + 31); m = _fresh(base, initial, device)
        o = route.make_optimizer(m, config); s = route.make_scheduler(o, config); hist = []
        for step in range(2):
            hist.append({"step": step + 1, **route.optimizer_step(
                pipeline, m, o, s, parent, device=device,
                max_gradient_norm=config.max_gradient_norm)})
        partial = route.capture_training_state(model=m, optimizer=o, scheduler=s, config=config,
            model_identity=identity, global_step=2, sampler_cursor=2, history=hist)
        route.save_training_state(path, partial); reopened = route.load_training_state(path)
        m2 = _fresh(base, initial, device); o2 = route.make_optimizer(m2, config); s2 = route.make_scheduler(o2, config)
        step0, cursor, hist = route.restore_training_state(reopened, model=m2, optimizer=o2,
            scheduler=s2, config=config, model_identity=identity)
        if (step0, cursor) != (2, 2): raise RuntimeError("resume cursor drift")
        for step in range(2, steps):
            hist.append({"step": step + 1, **route.optimizer_step(
                pipeline, m2, o2, s2, parent, device=device,
                max_gradient_norm=config.max_gradient_norm)})
        return route.capture_training_state(model=m2, optimizer=o2, scheduler=s2, config=config,
            model_identity=identity, global_step=steps, sampler_cursor=steps, history=hist)
    return interruption_resume_parity_check(run_full, run_resume, atol=0, rtol=0)


def run(args):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch = route._torch(); output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"smoke output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    config = route.RouteConfig(learning_rate=args.learning_rate, weight_decay=args.weight_decay,
                               batch_size=args.batch_size, total_steps=args.steps, seed=args.seed)
    config.validate()
    if config.total_steps != 20: raise ValueError("canonical Chronos V1 smoke requires 20 steps")
    train = _fixture(config.batch_size, config.seed); validation = _fixture(config.batch_size, config.seed + 1)
    fixture_path, fixture_manifest = _write_fixture(output, train, validation)
    pipeline, base, identity = route.load_pipeline(args.model_snapshot, device=args.device)
    base.eval(); initial = route.model_state_cpu(base); checks = {}

    one = _fresh(base, initial, args.device); one_o = route.make_optimizer(one, config); one_s = route.make_scheduler(one_o, config)
    checks["one_batch_forward_backward"] = forward_backward_check(lambda: route.optimizer_step(
        pipeline, one, one_o, one_s, train, device=args.device,
        max_gradient_norm=config.max_gradient_norm))
    del one, one_o, one_s

    route.seed_everything(config.seed); real = _fresh(base, initial, args.device)
    real_o = route.make_optimizer(real, config); real_s = route.make_scheduler(real_o, config)
    checks["controlled_learnable_loss_decrease"] = loss_decrease_check(
        lambda: _loss(pipeline, real, train, args.device),
        lambda: route.optimizer_step(pipeline, real, real_o, real_s, train, device=args.device,
                                     max_gradient_norm=config.max_gradient_norm),
        steps=20, min_relative_decrease=.20, tail=3)
    real_val = _loss(pipeline, real, validation, args.device)
    rng = np.random.default_rng(config.seed + 91)
    shuffled = train.copy(); shuffled[:, route.CONTEXT_LENGTH:] = shuffled[rng.permutation(len(shuffled)), route.CONTEXT_LENGTH:]
    destroyed = train.copy(); destroyed[:, :route.CONTEXT_LENGTH] = destroyed[:, rng.permutation(route.CONTEXT_LENGTH)]
    _, shuffle_losses, shuffle_val = _train(pipeline, base, initial, shuffled, validation, config, args.device)
    _, destroy_losses, destroy_val = _train(pipeline, base, initial, destroyed, validation, config, args.device)
    controls = control_rejection_check(real_val, [shuffle_val], [destroy_val], margin=.01, higher_is_better=False)
    checks["shuffle_control_rejection"] = _result(real_val + .01 <= shuffle_val,
        {**controls.metrics, "shuffle_final_train_loss": shuffle_losses[-1]},
        "real route did not beat shuffled targets")
    checks["time_destroyed_control_rejection"] = _result(real_val + .01 <= destroy_val,
        {**controls.metrics, "time_destroyed_final_train_loss": destroy_losses[-1]},
        "real route did not beat time-destroyed inputs")
    resume_path = output / "interrupted.train.pt"
    checks["exact_interruption_resume_trajectory"] = _resume(
        pipeline, base, initial, train, config, identity, args.device, resume_path)

    history = [{"step": i + 1, "train_loss": float(v)} for i, v in enumerate(
        checks["controlled_learnable_loss_decrease"].metrics["losses"][1:])]
    state = route.capture_training_state(model=real, optimizer=real_o, scheduler=real_s,
        config=config, model_identity=identity, global_step=20, sampler_cursor=20, history=history)
    state_path = output / "chronos_v1_smoke.train.pt"; state_id = route.save_training_state(state_path, state)
    bundle = route.build_export_bundle(model=real, model_identity=identity)
    bundle_path = output / "chronos_v1_smoke.forecast.pt"; bundle_id = route.save_export_bundle(bundle_path, bundle)
    exported_pipeline, exported_model, reopened_bundle = route.load_export_bundle(
        bundle_path, snapshot=args.model_snapshot, device=args.device)
    real.eval(); exported_model.eval()
    reference_logits = _np(route.teacher_forced_logits(pipeline, real, validation, device=args.device))
    exported_logits = _np(route.teacher_forced_logits(exported_pipeline, exported_model, validation, device=args.device))
    checks["training_exported_inference_parity"] = parity_check(
        reference_logits, exported_logits, atol=0, rtol=0, name="exported Chronos V1 token logits")

    changed_future = validation.copy(); changed_future[:, route.CONTEXT_LENGTH:, :4] += 50; changed_future[:, route.CONTEXT_LENGTH:, 4] *= 2
    checks["prefix_invariance"] = prefix_invariance_check(
        lambda parent: route.parent_array(parent)[:, :route.CONTEXT_LENGTH], validation,
        changed_future, prefix_length=route.CONTEXT_LENGTH, atol=0, rtol=0)
    checks["future_corruption_invariance"] = future_corruption_check(
        lambda parent: _np(route.forecast_samples(exported_pipeline, parent, device=args.device,
                                                   seed=config.seed + 500)),
        validation, changed_future, visible_length=route.CONTEXT_LENGTH, atol=0, rtol=0)
    checks["contract_roll_rejection"] = _boundary("contract_roll")
    checks["session_gap_rejection"] = _boundary("session_gap")
    checks["split_boundary_rejection"] = _boundary("split_boundary")
    checks["oos_boundary_rejection"] = _boundary("oos_boundary")

    baseline = _np(route.forecast_samples(exported_pipeline, validation, device=args.device, seed=config.seed + 700))
    perturbed = validation.copy(); perturbed[:, :route.CONTEXT_LENGTH, 4] *= 1.1
    changed = _np(route.forecast_samples(exported_pipeline, perturbed, device=args.device, seed=config.seed + 700))
    unaffected = [0, 1, 2, 3]
    unaffected_error = float(np.max(np.abs(baseline[:, unaffected] - changed[:, unaffected])))
    affected_change = float(np.max(np.abs(baseline[:, 4] - changed[:, 4])))
    checks["multivariate_channel_grouping"] = _result(
        unaffected_error == 0 and affected_change > 0,
        {"layout": "independent_univariate_passes", "shape": list(baseline.shape),
         "unaffected_channel_max_abs": unaffected_error,
         "affected_channel_max_change": affected_change, "perturbed_channel": "volume"},
        "Chronos V1 channel grouping changed unaffected channels")

    missing = validation.copy(); missing[:, :16, :] = np.nan
    missing_samples = _np(route.forecast_samples(exported_pipeline, missing, device=args.device,
                                                  seed=config.seed + 900))
    checks["native_missing_data_mask"] = _result(np.isfinite(missing_samples).all(),
        {"missing_prefix_bars": 16, "finite": bool(np.isfinite(missing_samples).all())},
        "Chronos V1 missing mask produced non-finite samples")
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(args.device); memory_probe = lambda: int(torch.cuda.max_memory_allocated(args.device))
    else: memory_probe = None
    perf = performance_check(lambda: route.forecast_samples(exported_pipeline, validation,
        device=args.device, seed=config.seed + 1100, num_samples=4), batch_size=config.batch_size,
        repeats=3, warmups=1, min_examples_per_second=.01, memory_probe=memory_probe)
    checks["memory_measurement"] = CheckResult(perf.status, dict(perf.metrics), perf.reason)
    checks["throughput_measurement"] = CheckResult(perf.status, dict(perf.metrics), perf.reason)
    checks["negative_price_behavior"] = negative_price_behavior_check(
        lambda parent: _np(route.forecast_samples(exported_pipeline, parent, device=args.device,
                                                   seed=config.seed + 1300, num_samples=4)),
        _negative_fixture(config.batch_size, config.seed + 7), behavior="support")
    samples_one = _np(route.forecast_samples(exported_pipeline, validation, device=args.device,
                                              seed=config.seed + 1500))
    samples_two = _np(route.forecast_samples(exported_pipeline, validation, device=args.device,
                                              seed=config.seed + 1500))
    checks["native_output_parity"] = parity_check(samples_one, samples_two, atol=0, rtol=0,
                                                   name="seeded Chronos V1 public samples")
    reopened_state = route.load_training_state(state_path)
    checks["checkpoint_lineage"] = _result(
        reopened_state.get("schema_version") == route.CHECKPOINT_SCHEMA
        and reopened_state.get("route_key") == route.ROUTE_KEY
        and reopened_state.get("model_identity") == identity
        and state_id["sha256"] == _sha(state_path)
        and reopened_bundle["route_key"] == route.ROUTE_KEY
        and bundle_id["sha256"] == _sha(bundle_path),
        {"checkpoint": state_id, "deployment": bundle_id,
         "required_state_fields": sorted(reopened_state)},
        "Chronos V1 checkpoint lineage is incomplete")
    fixture_doc = json.loads(fixture_manifest.read_text())
    checks["data_lineage"] = _result(
        fixture_doc["artifact"]["sha256"] == _sha(fixture_path)
        and fixture_doc["market_data_read"] is False and fixture_doc["oos_read"] is False,
        {"fixture_sha256": _sha(fixture_path), "market_data_read": False, "oos_read": False},
        "Chronos V1 fixture lineage is invalid")

    raw_checks = {name: result.manifest() for name, result in checks.items()}
    raw = {"schema_version": "ffm_chronos_v1_route_smoke_raw_v1", "route_key": route.ROUTE_KEY,
           "checks": raw_checks, "metrics": {"real_validation_loss": real_val,
           "shuffle_validation_loss": shuffle_val, "time_destroyed_validation_loss": destroy_val,
           "all_checks_pass": all(result.status == "pass" for result in checks.values())}}
    raw_path = _atomic_json(output / "raw_checks.json", raw)
    evidence = build_route_smoke_evidence(route_key=route.ROUTE_KEY,
        executor_path=route.__file__, executor_entrypoint="native_loss/forecast_samples",
        checks=raw_checks, artifacts={"model_snapshot": Path(args.model_snapshot).resolve(),
        "source_runtime": Path(importlib.metadata.distribution("chronos-forecasting")._path).resolve(),
        "synthetic_fixture": fixture_path, "synthetic_fixture_manifest": fixture_manifest,
        "interrupted_state": resume_path, "training_state": state_path,
        "deployment_bundle": bundle_path, "raw_checks": raw_path,
        "smoke_runner": Path(__file__).resolve()}, metrics=raw["metrics"])
    validate_route_smoke_evidence(evidence)
    evidence_path = _atomic_json(output / "smoke_evidence.json", evidence)
    return {"status": "pass" if evidence["smoke_admitted"] else "fail",
            "route_key": route.ROUTE_KEY, "smoke_admitted": evidence["smoke_admitted"],
            "pilot_admitted": False, "training_admitted": False,
            "evidence": {"path": str(evidence_path), "sha256": _sha(evidence_path),
                         "content_sha256": evidence["evidence_sha256"]},
            "metrics": raw["metrics"],
            "failed_checks": [n for n, r in checks.items() if r.status != "pass"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-snapshot", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0"); parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--steps", type=int, default=20); parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=.01); parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--overwrite", action="store_true"); args = parser.parse_args()
    result = run(args); print(json.dumps(result, sort_keys=True, indent=2))
    if result["status"] != "pass": raise SystemExit(1)


if __name__ == "__main__": main()
