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
from futures_foundation.finetune.native_adapters import (
    chronos_native_quantiles,
    moirai2_native_forecast,
    sundial_native_forecast,
    timesfm25_transformers_forecast,
    toto2_native_forecast,
    ttm_native_forecast,
)
from futures_foundation.finetune.native_contracts import (
    add_admission_argument,
    require_admission_from_args,
    technical_runtime_contract,
    validate_runtime_contract,
)
from futures_foundation.finetune.tournament import MAX_CONTEXT


QUANTILE_LEVELS = [0.1, 0.5, 0.9]
TIMEFRAMES_MINUTES = [1, 3, 5, 15, 30, 60]
TTM_FREQUENCY_TOKENS = {
    "1min": 1, "3min": 0, "5min": 3,
    "15min": 5, "30min": 6, "60min": 7,
}


def _forecast_runtime_facts(
    arm: str,
    *,
    context_length: int,
    prediction_length: int,
    dtype: str,
    samples: int,
    kronos_decoding: dict | None = None,
    ttm_frequency_tokens: dict[str, int] | None = None,
) -> dict:
    """Construct the complete runtime surface from the path the caller executes."""
    facts = {
        "context_length": int(context_length),
        "prediction_length": int(prediction_length),
        "dtype": str(dtype),
    }
    if arm.startswith("kronos"):
        if kronos_decoding is None:
            raise ValueError("Kronos runtime facts require the actual decoding configuration")
        facts.update({
            "decoding": [dict(kronos_decoding)],
            "timeframes_minutes": list(TIMEFRAMES_MINUTES),
            "timestamp_timezone": "UTC",
        })
    elif arm == "chronos_v1":
        facts.update({"num_samples": int(samples), "quantile_levels": list(QUANTILE_LEVELS)})
    elif arm == "chronos_bolt":
        facts["quantile_levels"] = list(QUANTILE_LEVELS)
    elif arm == "chronos_v2":
        facts.update({"quantile_levels": list(QUANTILE_LEVELS), "cross_learning": False})
    elif arm == "timesfm25":
        facts.update({
            "force_flip_invariance": True,
            "truncate_negative": False,
            "fix_quantile_crossing": False,
        })
    elif arm == "ttm_r2":
        mapping = dict(ttm_frequency_tokens or TTM_FREQUENCY_TOKENS)
        if mapping != TTM_FREQUENCY_TOKENS:
            raise ValueError(f"TTM runtime frequency mapping drifted: {mapping}")
        facts.update({
            "frequency_tokens_by_timeframe": mapping,
            "selector": "512-48-ft-r2.1",
        })
    elif arm == "moirai2_small":
        facts.update({
            "use_scope": "research_noncommercial",
            "masked_value_fill": 0.0,
            "quantile_crossing_repair": "forbidden_in_parity",
        })
    elif arm == "toto2_22m":
        facts.update({
            "decode_block_size": None,
            "masked_value_fill": 0.0,
            "series_ids": "one_semantic_group_per_item",
            "has_missing_values": False,
        })
    elif arm == "sundial_base":
        facts.update({
            "num_samples": int(samples),
            "isolated_environment": True,
            "hidden_states": "forbidden",
        })
    else:
        raise ValueError(f"no native forecast runtime fact builder for {arm!r}")
    return facts

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
    from scripts.train_ttm_tournament import FREQUENCY_TOKEN, _load_model, _validate_source
    arm = get_arm("ttm_r2")
    source = _validate_source(Path(args.upstream_repo).resolve(), arm.source_revision)
    model_args = argparse.Namespace(
        model_id=arm.model_id, model_revision=arm.model_revision,
    )
    model = _load_model(model_args, source=source).to(args.device).eval()
    contexts = _causal_suffix(windows["context"], 512, "TTM R2")
    output = np.empty((len(contexts), windows["horizon"], 5), np.float32)
    for lo in range(0, len(contexts), args.batch_size):
        hi = min(len(contexts), lo + args.batch_size)
        raw = contexts[lo:hi]
        freq = np.asarray([FREQUENCY_TOKEN[str(value)]
                           for value in windows["timeframe"][lo:hi]], np.int64)
        with torch.no_grad():
            forecast = ttm_native_forecast(
                model,
                torch.as_tensor(raw, device=args.device),
                torch.as_tensor(freq, device=args.device),
            )
        output[lo:hi] = forecast.float().cpu().numpy()
        print(f"[ttm-r2-zero-shot] {lo}:{hi}", flush=True)
    return output, None, {
        "arm": arm.manifest(),
        "input": {"context": 512, "horizon": windows["horizon"],
                  "ohlcv_mode": arm.ohlcv_mode, "adaptation": "zero_shot_only",
                  "dtype": "float32"},
    }


def _predict_timesfm(args, windows):
    import torch
    from transformers import TimesFm2_5ModelForPrediction
    arm = get_arm("timesfm25")
    model = TimesFm2_5ModelForPrediction.from_pretrained(
        arm.model_id, revision=arm.model_revision, dtype=torch.float32,
    ).to(args.device).eval()
    contexts = _causal_suffix(windows["context"], MAX_CONTEXT, "TimesFM 2.5")
    output = np.empty((len(contexts), windows["horizon"], 5), np.float32)
    for lo in range(0, len(contexts), args.batch_size):
        hi = min(len(contexts), lo + args.batch_size)
        flat = contexts[lo:hi].transpose(0, 2, 1).reshape(-1, contexts.shape[1])
        with torch.no_grad():
            forecast, _ = timesfm25_transformers_forecast(
                model,
                torch.as_tensor(flat, device=args.device),
                prediction_length=windows["horizon"],
                context_length=contexts.shape[1],
            )
        forecast = forecast.float().cpu().numpy()
        output[lo:hi] = forecast.reshape(hi - lo, 5, windows["horizon"]).transpose(0, 2, 1)
        print(f"[timesfm25-zero-shot] {lo}:{hi}", flush=True)
    return output, None, {
        "arm": arm.manifest(),
        "input": {"context": contexts.shape[1], "horizon": windows["horizon"],
                  "ohlcv_mode": arm.ohlcv_mode, "adaptation": "zero_shot_only",
                  "dtype": "float32", "force_flip_invariance": True,
                  "truncate_negative": False},
    }


def _predict_moirai(args, windows):
    import torch
    from scripts.train_moirai2_tournament import _load_model, _validate_source
    arm = get_arm("moirai2_small")
    _validate_source(Path(args.upstream_repo).resolve(), arm.source_revision)
    model_args = argparse.Namespace(
        model_id=arm.model_id, model_revision=arm.model_revision,
    )
    model = _load_model(model_args, context_length=MAX_CONTEXT).to(args.device).eval()
    levels = np.asarray(model.module.quantile_levels, float)
    median_index = int(np.argmin(np.abs(levels - 0.5)))
    contexts = _causal_suffix(windows["context"], MAX_CONTEXT, "Moirai")
    output = np.empty((len(contexts), windows["horizon"], 5), np.float32)
    for lo in range(0, len(contexts), args.batch_size):
        hi = min(len(contexts), lo + args.batch_size)
        past = torch.as_tensor(contexts[lo:hi], device=args.device)
        with torch.no_grad():
            quantiles = moirai2_native_forecast(
                model,
                past,
                torch.ones_like(past, dtype=torch.bool),
                torch.zeros(past.shape[:2], dtype=torch.bool, device=args.device),
            )
        prediction = quantiles[:, median_index].float().cpu().numpy()
        if prediction.shape != output[lo:hi].shape:
            raise ValueError(f"unexpected Moirai forecast shape: {prediction.shape}")
        output[lo:hi] = prediction
        print(f"[moirai2-small-research-zero-shot] {lo}:{hi}", flush=True)
    return output, None, {
        "arm": arm.manifest(),
        "input": {"context": MAX_CONTEXT, "horizon": windows["horizon"],
                  "ohlcv_mode": arm.ohlcv_mode, "adaptation": "zero_shot_only",
                  "dtype": "float32", "use_scope": "research_noncommercial"},
    }


def _predict_kronos(args, windows):
    from scripts.benchmark_kronos import _load_predictor, _predict
    arm = get_arm(args.arm)
    native_windows = _windows_with_causal_suffix(windows, MAX_CONTEXT, args.arm)
    predictor = _load_predictor(
        Path(args.upstream_repo).resolve(), args.device,
        arm.model_id, arm.model_revision,
        arm.tokenizer_id, arm.tokenizer_revision, native_windows["context_length"],
    )
    prediction = _predict(
        predictor, native_windows, cache_dir=args.cache_dir,
        signature=f"shared-zero-shot-{args.arm}-{args.seed}",
        batch_size=args.batch_size, horizon=windows["horizon"], temperature=1.0,
        top_k=1, top_p=1.0, sample_count=1, seed=args.seed, input_mode="ohlcv",
    )[:, :, :5]
    return prediction, None, {
        "arm": arm.manifest(),
        "input": {"context": native_windows["context_length"],
                  "horizon": windows["horizon"], "ohlcv_mode": arm.ohlcv_mode,
                  "adaptation": "zero_shot_only", "dtype": "float32",
                  "sample_count": 1, "top_k": 1},
    }


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
    arm = get_arm(args.arm)
    candidate = resolve_candidates((args.arm,))[0]
    pipeline, _ = _load_pipeline(candidate, args.device, "float32", None)
    contexts = _causal_suffix(windows["context"], MAX_CONTEXT, args.arm).transpose(0, 2, 1)
    output = np.empty((len(contexts), windows["horizon"], 5), np.float32)
    for lo in range(0, len(contexts), args.batch_size):
        hi = min(len(contexts), lo + args.batch_size)
        torch.manual_seed(args.seed + lo)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed + lo)
        if candidate.family == "chronos_2":
            quantiles, _ = chronos_native_quantiles(
                pipeline,
                torch.as_tensor(contexts[lo:hi]),
                family=candidate.family,
                prediction_length=windows["horizon"],
                quantile_levels=QUANTILE_LEVELS,
                batch_size=args.batch_size,
                context_length=contexts.shape[-1],
            )
            if not isinstance(quantiles, list) or len(quantiles) != hi - lo:
                raise ValueError("unexpected Chronos-2 forecast container")
            point = torch.stack([row[:, :, 1] for row in quantiles]).float().cpu().numpy()
            output[lo:hi] = point.transpose(0, 2, 1)
        else:
            flat = contexts[lo:hi].reshape(-1, contexts.shape[-1])
            quantiles, _ = chronos_native_quantiles(
                pipeline,
                torch.as_tensor(flat),
                family=candidate.family,
                prediction_length=windows["horizon"],
                quantile_levels=QUANTILE_LEVELS,
                num_samples=args.samples,
            )
            point = quantiles[:, :, 1].float().cpu().numpy()
            output[lo:hi] = point.reshape(hi - lo, 5, windows["horizon"]).transpose(0, 2, 1)
        print(f"[{args.arm}] {lo}:{hi}", flush=True)
    return output, None, {
        "arm": arm.manifest(),
        "input": {"context": contexts.shape[-1], "horizon": windows["horizon"],
                  "ohlcv_mode": arm.ohlcv_mode, "adaptation": "zero_shot_only",
                  "dtype": "float32", "samples": args.samples},
    }


def _predict_toto2(args, windows):
    import torch
    from toto2 import Toto2Model
    arm = get_arm("toto2_22m")
    model = Toto2Model.from_pretrained(
        arm.model_id, revision=arm.model_revision, map_location="cpu",
    ).to(args.device).eval()
    contexts = _causal_suffix(
        windows["context"], MAX_CONTEXT, "Toto 2.0"
    ).transpose(0, 2, 1)
    output = np.empty((len(contexts), windows["horizon"], 5), np.float32)
    for lo in range(0, len(contexts), args.batch_size):
        hi = min(len(contexts), lo + args.batch_size)
        target = torch.as_tensor(contexts[lo:hi], device=args.device)
        quantiles = toto2_native_forecast(
            model,
            target,
            torch.ones_like(target, dtype=torch.bool),
            torch.zeros(target.shape[:2], dtype=torch.long, device=args.device),
            prediction_length=windows["horizon"],
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
    contexts = _causal_suffix(windows["context"], MAX_CONTEXT, "Sundial")
    output = np.empty((len(contexts), windows["horizon"], 5), np.float32)
    for lo in range(0, len(contexts), args.batch_size):
        hi = min(len(contexts), lo + args.batch_size)
        flat = contexts[lo:hi].transpose(0, 2, 1).reshape(-1, contexts.shape[1])
        samples = sundial_native_forecast(
            model,
            torch.as_tensor(flat, device=args.device, dtype=torch.float32),
            prediction_length=windows["horizon"],
            num_samples=args.samples,
            seed=args.seed + lo,
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
        "chronos_v1", "chronos_bolt", "chronos_v2",
        "toto2_22m",
        "sundial_base",
    ), required=True)
    parser.add_argument("--windows", required=True); parser.add_argument("--checkpoint")
    parser.add_argument("--output", required=True); parser.add_argument("--upstream-repo")
    parser.add_argument("--cache-dir", default="output/foundation_tournament/shared_validation/cache")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0"); parser.add_argument("--seed", type=int, default=7400)
    parser.add_argument("--dtype", choices=("float32",), default="float32")
    parser.add_argument("--samples", type=int, default=20)
    add_admission_argument(parser)
    args = parser.parse_args()
    if (args.arm.startswith("kronos") or args.arm in {"moirai2_small", "ttm_r2"}) and not args.upstream_repo:
        parser.error("Kronos/Moirai/TTM prediction requires --upstream-repo")
    if args.checkpoint:
        parser.error(
            "native-track evaluation is zero-shot only; historical/adapted checkpoints are not admitted"
        )
    admission = require_admission_from_args(
        args, arm_key=args.arm, track="F", route=None, require_training=False,
    )
    windows, manifest = load_window_artifact(args.windows)
    decoding = None
    if args.arm.startswith("kronos"):
        decoding = {
            "temperature": 1.0, "top_k": 1, "top_p": 1.0, "sample_count": 1,
        }
    ttm_tokens = None
    if args.arm == "ttm_r2":
        from scripts.train_ttm_tournament import FREQUENCY_TOKEN
        ttm_tokens = dict(FREQUENCY_TOKEN)
    runtime_facts = _forecast_runtime_facts(
        args.arm,
        context_length=MAX_CONTEXT,
        prediction_length=int(windows["horizon"]),
        dtype=args.dtype,
        samples=args.samples,
        kronos_decoding=decoding,
        ttm_frequency_tokens=ttm_tokens,
    )
    runtime_contract = validate_runtime_contract(args.arm, "F", runtime_facts)
    if decoding is not None:
        if decoding not in runtime_contract["decoding"]:
            raise ValueError("shared Kronos decoding is not covered by technical evidence")
    dispatch = {
        "ttm_r2": _predict_ttm, "timesfm25": _predict_timesfm,
        "moirai2_small": _predict_moirai,
        "kronos_mini": _predict_kronos, "kronos_small": _predict_kronos,
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
            "admission": {
                "schema_version": admission["schema_version"],
                "integrity": admission["integrity"],
                "registry_sha256": admission["registry_sha256"],
                "dossier_sha256": admission["dossier_sha256"],
                "evidence_registry_sha256": admission["evidence_registry_sha256"],
                "technical_evidence_id": admission["technical_evidence_id"],
            },
            **checkpoint_metadata,
            "input_contract": bundle.get("input"),
            "technical_runtime_contract": technical_runtime_contract(args.arm, "F"),
            "seed": args.seed,
        },
    )


if __name__ == "__main__":
    main()
