#!/usr/bin/env python3
"""Generate point forecasts for one trained arm on the locked shared validation rows."""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from futures_foundation.finetune.foundation_eval import (
    load_window_artifact, save_prediction_artifact,
)
from futures_foundation.finetune.foundation_roster import get_arm
from futures_foundation.finetune.tournament import MAX_CONTEXT

def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _causal_suffix(contexts, length, arm):
    contexts = np.asarray(contexts, np.float32)
    if contexts.ndim != 3:
        raise ValueError(f"shared context must be [row,time,channel], got {contexts.shape}")
    if contexts.shape[1] < length:
        raise ValueError(
            f"shared context has {contexts.shape[1]} bars; {arm} requires {length}"
        )
    return contexts[:, -length:]


def _windows_with_causal_suffix(windows, length, arm):
    native = dict(windows)
    native["context"] = _causal_suffix(windows["context"], length, arm)
    if "context_time_ns" in windows:
        native["context_time_ns"] = np.asarray(windows["context_time_ns"])[:, -length:]
    native["context_length"] = int(length)
    return native


def _load_bundle(path, schema):
    import torch
    path = Path(path).expanduser().resolve()
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    if bundle.get("schema_version") != schema:
        raise ValueError(f"unsupported checkpoint schema in {path}")
    return path, bundle


def _predict_ttm(args, windows):
    import torch
    from scripts.train_ttm_tournament import FREQUENCY_TOKEN, _load_model
    checkpoint, bundle = _load_bundle(args.checkpoint, "ffm_ttm_staged_bundle_v1")
    if bundle.get("stage") != "stage3_forecast":
        raise ValueError("TTM forecast evaluation requires a Stage 3 checkpoint")
    model_args = argparse.Namespace(
        model_id=bundle["arm"]["model_id"], model_revision=bundle["arm"]["model_revision"],
    )
    model = _load_model(model_args)
    model.load_state_dict(bundle["model_state"], strict=True)
    model.to(args.device).eval()
    contexts = np.asarray(windows["context"], np.float32)
    output = np.empty((len(contexts), windows["horizon"], 5), np.float32)
    for lo in range(0, len(contexts), args.batch_size):
        hi = min(len(contexts), lo + args.batch_size)
        raw = contexts[lo:hi]
        mean = raw.mean(axis=1, keepdims=True)
        std = np.maximum(raw.std(axis=1, keepdims=True), 1e-5)
        scaled = (raw - mean) / std
        freq = np.asarray([FREQUENCY_TOKEN[str(value)]
                           for value in windows["timeframe"][lo:hi]], np.int64)
        with torch.no_grad(), torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=str(args.device).startswith("cuda")):
            forecast = model(
                past_values=torch.as_tensor(scaled, device=args.device),
                freq_token=torch.as_tensor(freq, device=args.device), return_loss=False,
            ).prediction_outputs
        output[lo:hi] = forecast.float().cpu().numpy() * std + mean
        print(f"[ttm-r2] {lo}:{hi}", flush=True)
    return output, checkpoint, bundle


def _predict_timesfm(args, windows):
    import torch
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from transformers import TimesFm2_5ModelForPrediction
    checkpoint, bundle = _load_bundle(args.checkpoint, "ffm_timesfm_staged_bundle_v1")
    if bundle.get("stage") != "stage3_forecast":
        raise ValueError("TimesFM forecast evaluation requires a Stage 3 checkpoint")
    arm, lora = bundle["arm"], bundle["lora"]
    base = TimesFm2_5ModelForPrediction.from_pretrained(
        arm["model_id"], revision=arm["model_revision"], dtype=torch.bfloat16,
    )
    model = get_peft_model(base, LoraConfig(
        r=lora["rank"], lora_alpha=lora["alpha"], target_modules="all-linear",
        lora_dropout=lora["dropout"], bias="none",
    ))
    set_peft_model_state_dict(model, bundle["adapter_state"])
    model.to(args.device).eval()
    contexts = _causal_suffix(windows["context"], MAX_CONTEXT, "TimesFM 2.5")
    output = np.empty((len(contexts), windows["horizon"], 5), np.float32)
    for lo in range(0, len(contexts), args.batch_size):
        hi = min(len(contexts), lo + args.batch_size)
        flat = contexts[lo:hi].transpose(0, 2, 1).reshape(-1, contexts.shape[1])
        with torch.no_grad(), torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=str(args.device).startswith("cuda")):
            forecast = model(
                past_values=torch.as_tensor(flat, device=args.device),
                forecast_context_len=contexts.shape[1],
            ).mean_predictions
        forecast = forecast[:, :windows["horizon"]].float().cpu().numpy()
        output[lo:hi] = forecast.reshape(hi - lo, 5, windows["horizon"]).transpose(0, 2, 1)
        print(f"[timesfm25] {lo}:{hi}", flush=True)
    return output, checkpoint, bundle


def _predict_moirai(args, windows):
    import torch
    from scripts.train_moirai2_tournament import _load_model
    checkpoint, bundle = _load_bundle(args.checkpoint, "ffm_moirai2_staged_bundle_v1")
    if bundle.get("stage") != "stage3_forecast":
        raise ValueError("Moirai-2 forecast evaluation requires a Stage 3 checkpoint")
    model_args = argparse.Namespace(
        model_id=bundle["arm"]["model_id"], model_revision=bundle["arm"]["model_revision"],
    )
    model = _load_model(model_args)
    model.load_state_dict(bundle["model_state"], strict=True)
    model.to(args.device).eval()
    levels = np.asarray(model.module.quantile_levels, float)
    median_index = int(np.argmin(np.abs(levels - 0.5)))
    contexts = _causal_suffix(windows["context"], MAX_CONTEXT, "Moirai")
    # The shared artifact is deliberately long enough for every arm.  Moirai was
    # trained with its native 256-bar contract, so give it the most recent causal
    # suffix.  Passing all 512 bars makes Moirai's packed target and prediction
    # masks disagree because its Forecast wrapper is configured for 256 bars.
    output = np.empty((len(contexts), windows["horizon"], 5), np.float32)
    for lo in range(0, len(contexts), args.batch_size):
        hi = min(len(contexts), lo + args.batch_size)
        past = torch.as_tensor(contexts[lo:hi], device=args.device)
        with torch.no_grad(), torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=str(args.device).startswith("cuda")):
            quantiles = model(
                past_target=past, past_observed_target=torch.ones_like(past, dtype=torch.bool),
                past_is_pad=torch.zeros(past.shape[:2], dtype=torch.bool, device=args.device),
            )
        prediction = quantiles[:, median_index].float().cpu().numpy()
        if prediction.shape != output[lo:hi].shape:
            raise ValueError(f"unexpected Moirai forecast shape: {prediction.shape}")
        output[lo:hi] = prediction
        print(f"[moirai2-small] {lo}:{hi}", flush=True)
    return output, checkpoint, bundle


def _predict_kronos(args, windows):
    import pandas as pd
    import torch
    from scripts.benchmark_kronos import _context_frame, _load_predictor, _predict
    checkpoint, bundle = _load_bundle(args.checkpoint, "ffm_kronos_tournament_bundle_v1")
    predictor_meta, tokenizer_meta = bundle["predictor"], bundle["tokenizer"]
    expected = get_arm(args.arm)
    if (predictor_meta["id"], predictor_meta["revision"]) != (
            expected.model_id, expected.model_revision):
        raise ValueError(f"Kronos checkpoint does not belong to {args.arm}")
    native_windows = _windows_with_causal_suffix(windows, MAX_CONTEXT, args.arm)
    predictor = _load_predictor(
        Path(args.upstream_repo).resolve(), args.device,
        predictor_meta["id"], predictor_meta["revision"],
        tokenizer_meta["id"], tokenizer_meta["revision"], native_windows["context_length"],
    )
    predictor.tokenizer.load_state_dict(bundle["tokenizer_state"], strict=True)
    predictor.model.load_state_dict(bundle["predictor_state"], strict=True)
    predictor.tokenizer.eval(); predictor.model.eval()
    # Reuse the audited official-predictor adapter; its deterministic greedy path applies
    # exactly the context-only normalization used during tournament training.
    prediction = _predict(
        predictor, native_windows, cache_dir=args.cache_dir,
        signature=f"shared-{args.arm}-{_sha256(checkpoint)}-{args.seed}",
        batch_size=args.batch_size, horizon=windows["horizon"], temperature=1.0,
        top_k=1, top_p=1.0, sample_count=1, seed=args.seed, input_mode="ohlcv",
    )[:, :, :5]
    return prediction, checkpoint, bundle


def _predict_moment(args, windows):
    import torch
    from futures_foundation.finetune.moment_eval import left_pad_contexts
    from scripts.train_moment_tournament import BUNDLE_SCHEMA, _load_task_model
    checkpoint, bundle = _load_bundle(args.checkpoint, BUNDLE_SCHEMA)
    if bundle.get("stage") != "stage3_forecast":
        raise ValueError("MOMENT shared forecast requires a Stage-3 checkpoint")
    expected = get_arm(args.arm)
    if bundle.get("model") != {"id": expected.model_id, "revision": expected.model_revision}:
        raise ValueError("MOMENT checkpoint identity mismatch")
    model_args = argparse.Namespace(
        model_id=expected.model_id, model_revision=expected.model_revision,
        mask_ratio=0.0, gradient_checkpointing=False,
    )
    model = _load_task_model(
        Path(args.upstream_repo).resolve(), model_args, "forecasting",
        forecast_horizon=int(windows["horizon"]),
    )
    model.load_state_dict(bundle["model_state"], strict=True)
    model.to(args.device).eval()
    contexts = _causal_suffix(windows["context"], MAX_CONTEXT, args.arm)
    output = np.empty((len(contexts), windows["horizon"], 5), np.float32)
    with torch.no_grad():
        for lo in range(0, len(contexts), args.batch_size):
            hi = min(len(contexts), lo + args.batch_size)
            padded, mask = left_pad_contexts(contexts[lo:hi], 512)
            x = torch.as_tensor(padded, device=args.device)
            input_mask = torch.as_tensor(mask, device=args.device)
            with torch.autocast(
                    "cuda", dtype=torch.bfloat16,
                    enabled=str(args.device).startswith("cuda")):
                forecast = model(x_enc=x, input_mask=input_mask).forecast
            output[lo:hi] = forecast.float().cpu().numpy().transpose(0, 2, 1)
            print(f"[moment] generated {lo}:{hi}", flush=True)
    return output, checkpoint, bundle


def _predict_chronos(args, windows):
    import torch
    from futures_foundation.finetune.chronos_family import resolve_candidates
    from scripts.benchmark_chronos_family import _load_pipeline
    checkpoint, bundle = _load_bundle(args.checkpoint, "ffm_chronos_tournament_bundle_v1")
    candidate = resolve_candidates((args.arm,))[0]
    pipeline, _ = _load_pipeline(candidate, args.device, args.dtype, checkpoint)
    contexts = _causal_suffix(windows["context"], MAX_CONTEXT, args.arm).transpose(0, 2, 1)
    output = np.empty((len(contexts), windows["horizon"], 5), np.float32)
    for lo in range(0, len(contexts), args.batch_size):
        hi = min(len(contexts), lo + args.batch_size)
        torch.manual_seed(args.seed + lo)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed + lo)
        if candidate.family == "chronos_2":
            quantiles, _ = pipeline.predict_quantiles(
                torch.as_tensor(contexts[lo:hi]), prediction_length=windows["horizon"],
                quantile_levels=[0.5], batch_size=args.batch_size,
                context_length=contexts.shape[-1], cross_learning=False,
            )
            if not isinstance(quantiles, list) or len(quantiles) != hi - lo:
                raise ValueError("unexpected Chronos-2 forecast container")
            point = torch.stack([row[:, :, 0] for row in quantiles]).float().cpu().numpy()
            output[lo:hi] = point.transpose(0, 2, 1)
        else:
            flat = contexts[lo:hi].reshape(-1, contexts.shape[-1])
            kwargs = ({"num_samples": args.samples}
                      if candidate.family == "original_chronos_t5" else {})
            quantiles, _ = pipeline.predict_quantiles(
                torch.as_tensor(flat), prediction_length=windows["horizon"],
                quantile_levels=[0.5], **kwargs,
            )
            point = quantiles[:, :, 0].float().cpu().numpy()
            output[lo:hi] = point.reshape(hi - lo, 5, windows["horizon"]).transpose(0, 2, 1)
        print(f"[{args.arm}] {lo}:{hi}", flush=True)
    return output, checkpoint, bundle


def _predict_toto2(args, windows):
    import torch
    from toto2 import Toto2Model
    arm = get_arm("toto2_22m")
    model = Toto2Model.from_pretrained(
        arm.model_id, revision=arm.model_revision, map_location="cpu",
    ).to(args.device).eval()
    contexts = np.asarray(windows["context"], np.float32).transpose(0, 2, 1)
    output = np.empty((len(contexts), windows["horizon"], 5), np.float32)
    for lo in range(0, len(contexts), args.batch_size):
        hi = min(len(contexts), lo + args.batch_size)
        target = torch.as_tensor(contexts[lo:hi], device=args.device)
        quantiles = model.forecast(
            {
                "target": target, "target_mask": torch.ones_like(target, dtype=torch.bool),
                "series_ids": torch.zeros(target.shape[:2], dtype=torch.long, device=args.device),
            },
            horizon=windows["horizon"], decode_block_size=768, has_missing_values=False,
        )
        point = quantiles[4].float().cpu().numpy()
        output[lo:hi] = point.transpose(0, 2, 1)
        print(f"[toto2-22m-zero-shot] {lo}:{hi}", flush=True)
    return output, None, {
        "arm": arm.manifest(),
        "input": {"context": windows["context_length"], "horizon": windows["horizon"],
                  "ohlcv_mode": "joint_ohlcv", "adaptation": "zero_shot_only"},
    }


def _predict_sundial(args, windows):
    import torch
    from transformers import AutoModelForCausalLM
    arm = get_arm("sundial_base")
    model = AutoModelForCausalLM.from_pretrained(
        arm.model_id, revision=arm.model_revision, trust_remote_code=True,
    ).to(args.device).eval()
    contexts = np.asarray(windows["context"], np.float32)
    output = np.empty((len(contexts), windows["horizon"], 5), np.float32)
    for lo in range(0, len(contexts), args.batch_size):
        hi = min(len(contexts), lo + args.batch_size)
        flat = contexts[lo:hi].transpose(0, 2, 1).reshape(-1, contexts.shape[1])
        torch.manual_seed(args.seed + lo)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed + lo)
        samples = model.generate(
            torch.as_tensor(flat, device=args.device, dtype=torch.float32),
            max_new_tokens=windows["horizon"], num_samples=args.samples,
        )
        point = samples.float().mean(dim=1).cpu().numpy()
        output[lo:hi] = point.reshape(hi - lo, 5, windows["horizon"]).transpose(0, 2, 1)
        print(f"[sundial-base-zero-shot] {lo}:{hi}", flush=True)
    return output, None, {
        "arm": arm.manifest(),
        "input": {"context": windows["context_length"], "horizon": windows["horizon"],
                  "ohlcv_mode": "channel_independent_ohlcv", "adaptation": "zero_shot_only",
                  "samples": args.samples, "dtype": "float32_official_default"},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=(
        "ttm_r2", "timesfm25", "moirai2_small", "kronos_mini", "kronos_small",
        "moment_small",
        "chronos_v1", "chronos_bolt", "chronos_v2",
        "toto2_22m",
        "sundial_base",
    ), required=True)
    parser.add_argument("--windows", required=True); parser.add_argument("--checkpoint")
    parser.add_argument("--output", required=True); parser.add_argument("--upstream-repo")
    parser.add_argument("--cache-dir", default="output/foundation_tournament/shared_validation/cache")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0"); parser.add_argument("--seed", type=int, default=7400)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--samples", type=int, default=64)
    args = parser.parse_args()
    if (args.arm.startswith("kronos") or args.arm == "moment_small") and not args.upstream_repo:
        parser.error("Kronos/MOMENT prediction requires --upstream-repo")
    if args.arm not in {"toto2_22m", "sundial_base"} and not args.checkpoint:
        parser.error("trained arms require --checkpoint")
    windows, manifest = load_window_artifact(args.windows)
    dispatch = {
        "ttm_r2": _predict_ttm, "timesfm25": _predict_timesfm,
        "moirai2_small": _predict_moirai,
        "kronos_mini": _predict_kronos, "kronos_small": _predict_kronos,
        "moment_small": _predict_moment,
        "chronos_v1": _predict_chronos, "chronos_bolt": _predict_chronos,
        "chronos_v2": _predict_chronos,
        "toto2_22m": _predict_toto2,
        "sundial_base": _predict_sundial,
    }
    prediction, checkpoint, bundle = dispatch[args.arm](args, windows)
    stamped_arm = bundle.get("arm")
    if stamped_arm is not None and stamped_arm.get("key") != args.arm:
        raise ValueError(f"checkpoint arm mismatch: expected {args.arm}")
    checkpoint_metadata = ({
        "checkpoint": str(checkpoint), "checkpoint_sha256": _sha256(checkpoint),
    } if checkpoint is not None else {})
    save_prediction_artifact(
        args.output, prediction, windows_manifest=manifest, arm=args.arm,
        metadata={
            **checkpoint_metadata, "input_contract": bundle.get("input"), "seed": args.seed,
        },
    )


if __name__ == "__main__":
    main()
