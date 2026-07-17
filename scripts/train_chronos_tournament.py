#!/usr/bin/env python3
"""Leak-safe Chronos v1/Bolt/v2 adaptation for the equal-history tournament."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from futures_foundation.finetune import chronos_family, tournament, tournament_data
from futures_foundation.finetune.chronos_family import CANDIDATES
from futures_foundation.finetune.tournament import (
    FORECAST_HORIZON, MAX_CONTEXT, OOS_START, PARENT_LENGTH, TRAIN_START,
    VALIDATION_START,
)
from futures_foundation.finetune.tournament_data import (
    balanced_schedule, gather_parent, load_adaptation_data, schedule_fingerprint,
)


def _sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _atomic_save(path, value):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=float) + "\n")
    os.replace(tmp, path)


def _cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _config_signature(args):
    ignored = {"output", "resume", "stop_after"}
    value = {key: value for key, value in vars(args).items() if key not in ignored}
    raw = json.dumps(value, sort_keys=True, default=list, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _seed(seed):
    import torch
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _archive_sources(output, extra_sources=()):
    destination = Path(str(output) + ".source")
    destination.mkdir(parents=True, exist_ok=True)
    archived = {}
    for source in (
        Path(__file__).resolve(), Path(chronos_family.__file__).resolve(),
        Path(tournament.__file__).resolve(), Path(tournament_data.__file__).resolve(),
        *tuple(Path(value).resolve() for value in extra_sources),
    ):
        target = destination / source.name
        tmp = Path(str(target) + ".tmp")
        shutil.copyfile(source, tmp)
        os.replace(tmp, target)
        archived[source.name] = {"path": str(target.resolve()), "sha256": _sha256(target)}
    return archived


def _load_pipeline(candidate, device):
    import torch
    from chronos import BaseChronosPipeline, Chronos2Pipeline
    cls = Chronos2Pipeline if candidate.family == "chronos_2" else BaseChronosPipeline
    pipeline = cls.from_pretrained(
        candidate.model_id, revision=candidate.revision,
        device_map="cpu", dtype=torch.float32,
    )
    resolved = getattr(pipeline.inner_model.config, "_commit_hash", None)
    if resolved != candidate.revision:
        raise RuntimeError(
            f"model revision drift: expected {candidate.revision}, loaded {resolved}"
        )
    model = pipeline.inner_model.to(device)
    return pipeline, model


def _split_parent(parent, stage="stage3_forecast"):
    parent = np.asarray(parent, np.float32)
    if parent.ndim != 3 or parent.shape[1] < MAX_CONTEXT + FORECAST_HORIZON:
        raise ValueError("parent windows must have shape [B,>=272,5]")
    if stage == "stage1_reconstruction":
        context = parent[:, :MAX_CONTEXT - FORECAST_HORIZON]
        future = parent[:, MAX_CONTEXT - FORECAST_HORIZON:MAX_CONTEXT]
    elif stage == "stage3_forecast":
        context = parent[:, :MAX_CONTEXT]
        future = parent[:, MAX_CONTEXT:MAX_CONTEXT + FORECAST_HORIZON]
    else:
        raise ValueError(f"unsupported Chronos training stage: {stage}")
    return context, future


def _univariate_series(context, future, mode):
    if mode == "close_only":
        return context[:, :, 3], future[:, :, 3]
    if mode == "channel_independent_ohlcv":
        return (
            np.transpose(context, (0, 2, 1)).reshape(-1, context.shape[1]),
            np.transpose(future, (0, 2, 1)).reshape(-1, FORECAST_HORIZON),
        )
    raise ValueError(f"unsupported univariate input mode: {mode}")


def _chronos_v1_loss(pipeline, model, context, future, device, *, input_mode):
    """Original Chronos token cross-entropy on the common 16-bar horizon."""
    import torch
    tokenizer = pipeline.tokenizer
    context, future = _univariate_series(context, future, input_mode)
    context = torch.as_tensor(context, dtype=torch.float32)
    future = torch.as_tensor(future, dtype=torch.float32)
    input_ids, attention_mask, scale = tokenizer.context_input_transform(context)
    # The official tokenizer asserts its pretraining horizon (64). Tokenize the locked
    # tournament horizon with the same scale, then append EOS at the true horizon.
    label_ids, label_mask, _ = tokenizer._input_transform(future, scale=scale)
    if tokenizer.config.use_eos_token:
        label_ids, label_mask = tokenizer._append_eos_token(label_ids, label_mask)
    labels = label_ids.masked_fill(~label_mask, -100)
    output = model(
        input_ids=input_ids.to(device),
        attention_mask=attention_mask.to(device),
        labels=labels.to(device),
    )
    return output.loss


def _chronos_bolt_loss(_pipeline, model, context, future, device, *, input_mode):
    import torch
    context, future = _univariate_series(context, future, input_mode)
    context = torch.as_tensor(context, device=device)
    future = torch.as_tensor(future, device=device)
    return model(context=context, target=future).loss


def _chronos_v2_loss(_pipeline, model, context, future, device, *, mode):
    import torch
    if mode == "joint_ohlcv":
        # [anchor,time,channel] -> [anchor*channel,time], five variates per group.
        x = np.transpose(context, (0, 2, 1)).reshape(-1, context.shape[1])
        y = np.transpose(future, (0, 2, 1)).reshape(-1, FORECAST_HORIZON)
        groups = np.repeat(np.arange(len(context), dtype=np.int64), context.shape[2])
    elif mode == "close_only":
        x, y = context[:, :, 3], future[:, :, 3]
        groups = np.arange(len(context), dtype=np.int64)
    else:
        raise ValueError(f"unsupported Chronos-2 mode: {mode}")
    x = torch.as_tensor(x, device=device)
    y = torch.as_tensor(y, device=device)
    group_ids = torch.as_tensor(groups, device=device)
    future_covariates = torch.full_like(y, torch.nan)
    output_patch = int(model.chronos_config.output_patch_size)
    output = model(
        context=x, group_ids=group_ids,
        future_covariates=future_covariates,
        future_target=y,
        num_output_patches=math.ceil(FORECAST_HORIZON / output_patch),
    )
    return output.loss


def _native_loss(candidate, pipeline, model, parent, device, *, chronos2_mode,
                 univariate_input, stage="stage3_forecast"):
    context, future = _split_parent(parent, stage=stage)
    if candidate.family == "original_chronos_t5":
        return _chronos_v1_loss(
            pipeline, model, context, future, device, input_mode=univariate_input,
        )
    if candidate.family == "chronos_bolt":
        return _chronos_bolt_loss(
            pipeline, model, context, future, device, input_mode=univariate_input,
        )
    return _chronos_v2_loss(
        pipeline, model, context, future, device, mode=chronos2_mode,
    )


def train(args):
    import torch
    if args.family not in CANDIDATES:
        raise ValueError(f"unsupported Chronos family: {args.family}")
    candidate = CANDIDATES[args.family]
    if args.family != "chronos_v2" and args.chronos2_mode != "close_only":
        raise ValueError("--chronos2-mode only applies to chronos_v2")
    if min(args.max_steps, args.batch_size, args.eval_every, args.val_batches) < 1:
        raise ValueError("training and validation budgets must be positive")
    if args.stop_after is not None and not 1 <= args.stop_after <= args.max_steps:
        raise ValueError("stop_after must lie in [1,max_steps]")
    if args.stage == "stage1_reconstruction" and args.warm_checkpoint:
        raise ValueError("Chronos Stage 1 cannot have a warm checkpoint")
    if args.stage == "stage3_forecast" and not args.warm_checkpoint:
        raise ValueError("Chronos Stage 3 requires a Stage-2 warm checkpoint")

    output = Path(args.output).resolve()
    state_path = Path(str(output) + ".train.pt")
    if output.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite {output} without --resume")
    if args.resume and not state_path.is_file():
        raise FileNotFoundError(f"exact resume state missing: {state_path}")
    signature = _config_signature(args)
    _seed(args.seed)

    streams, big, train_starts, val_starts, groups = load_adaptation_data(
        args.data_dir, args.tickers, args.timeframes, parent_length=PARENT_LENGTH,
    )
    train_schedule, train_groups = balanced_schedule(
        train_starts, groups["train_bounds"], args.max_steps * args.batch_size,
        args.seed,
    )
    val_schedule, val_groups = balanced_schedule(
        val_starts, groups["val_bounds"], args.val_batches * args.batch_size,
        args.validation_seed,
    )
    pipeline, model = _load_pipeline(candidate, args.device)
    warm_parent = None
    if args.warm_checkpoint:
        warm = Path(args.warm_checkpoint).expanduser().resolve()
        if not warm.is_file():
            raise FileNotFoundError(f"Chronos warm checkpoint missing: {warm}")
        bundle = torch.load(warm, map_location="cpu", weights_only=False)
        if bundle.get("schema_version") != "ffm_chronos_tournament_bundle_v1":
            raise ValueError("unsupported Chronos warm checkpoint schema")
        if bundle.get("stage") != "stage2_contrastive":
            raise ValueError("Chronos Stage 3 requires a Stage-2 parent")
        if bundle.get("candidate") != candidate.manifest():
            raise ValueError("Chronos warm checkpoint candidate identity mismatch")
        model.load_state_dict(bundle["model_state"], strict=True)
        warm_parent = {"path": str(warm), "sha256": _sha256(warm),
                       "stage": bundle["stage"]}
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_steps,
    )
    history, best_val, best_model, start = [], float("inf"), None, 0

    if args.resume:
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state.get("config_signature") != signature:
            raise ValueError("exact resume refused: configuration changed")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        history = state["history"]
        best_val = float(state["best_val"])
        best_model = state["best_model"]
        start = int(state["step"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        if torch.cuda.is_available() and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])

    def parent_batch(schedule, step):
        lo = step * args.batch_size
        return gather_parent(big, schedule[lo:lo + args.batch_size], PARENT_LENGTH)

    def validate():
        model.eval()
        total = 0.0
        with torch.no_grad():
            for step in range(args.val_batches):
                parent = parent_batch(val_schedule, step)
                with torch.autocast(
                    "cuda", dtype=torch.bfloat16,
                    enabled=str(args.device).startswith("cuda"),
                ):
                    loss = _native_loss(
                        candidate, pipeline, model, parent, args.device,
                        chronos2_mode=args.chronos2_mode,
                        univariate_input=args.univariate_input,
                        stage=args.stage,
                    )
                total += float(loss)
        return total / args.val_batches

    def save_state(step):
        _atomic_save(state_path, {
            "schema_version": "ffm_chronos_training_state_v1",
            "config_signature": signature, "step": int(step),
            "model": _cpu_state(model), "best_model": best_model,
            "best_val": best_val, "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "history": history,
            "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        })

    running, running_steps = 0.0, 0
    model.train()
    for step in range(start, args.max_steps):
        parent_values = parent_batch(train_schedule, step)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            "cuda", dtype=torch.bfloat16,
            enabled=str(args.device).startswith("cuda"),
        ):
            loss = _native_loss(
                candidate, pipeline, model, parent_values, args.device,
                chronos2_mode=args.chronos2_mode,
                univariate_input=args.univariate_input,
                stage=args.stage,
            )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite training loss at step {step + 1}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        running += float(loss.detach())
        running_steps += 1
        completed = step + 1
        evaluate = completed % args.eval_every == 0 or completed == args.max_steps
        stop = args.stop_after is not None and completed == args.stop_after
        if evaluate or stop:
            val = validate()
            row = {
                "step": completed, "train_loss": running / running_steps,
                "val_loss": val, "learning_rate": scheduler.get_last_lr()[0],
            }
            history.append(row)
            print(
                f"[{args.family}-train] step={completed} "
                f"train={row['train_loss']:.6f} val={val:.6f}", flush=True,
            )
            running, running_steps = 0.0, 0
            if val < best_val:
                best_val = val
                best_model = _cpu_state(model)
                _atomic_save(output, {
                    "schema_version": "ffm_chronos_tournament_bundle_v1",
                    "stage": args.stage,
                    "candidate": candidate.manifest(),
                    "chronos2_mode": args.chronos2_mode,
                    "model_state": best_model,
                    "parent": warm_parent,
                    "input": {
                        "context": MAX_CONTEXT, "horizon": FORECAST_HORIZON,
                        "channels": (
                            "joint_ohlcv" if args.chronos2_mode == "joint_ohlcv"
                            else args.univariate_input
                        ),
                    },
                })
            save_state(completed)
            model.train()
        if stop and completed < args.max_steps:
            print(f"[{args.family}-train] intentional stop at {completed}", flush=True)
            return {"status": "interrupted", "step": completed}

    report = {
        "schema_version": "ffm_chronos_tournament_train_v1",
        "status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": args.stage, "parent": warm_parent,
        "split": {
            "train_start": TRAIN_START, "validation_start": VALIDATION_START,
            "oos_start": OOS_START, "oos_read": False,
        },
        "model": {
            **candidate.manifest(),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        },
        "data": {
            "streams": [stream["sid"] for stream in streams],
            "data_manifest_sha256": _sha256(next(path for path in (
                Path(args.data_dir) / "TOURNAMENT_CACHE.json",
                Path(args.data_dir) / "MANIFEST.json",
            ) if path.is_file())),
            "train_windows": int(len(train_starts)),
            "validation_windows": int(len(val_starts)),
            "train_anchors_seen": int(args.max_steps * args.batch_size),
            "target_series_per_anchor": (
                5 if (args.chronos2_mode == "joint_ohlcv" or
                      args.univariate_input == "channel_independent_ohlcv") else 1
            ),
        },
        "sampling": {
            "train_schedule_sha256": schedule_fingerprint(train_schedule, train_groups),
            "validation_schedule_sha256": schedule_fingerprint(val_schedule, val_groups),
            "training_seed": args.seed, "validation_seed": args.validation_seed,
        },
        "objective": {
            "chronos_v1": "native tokenized T5 cross_entropy",
            "chronos_bolt": "native multi_quantile_loss",
            "chronos_v2": "native grouped multi_quantile_loss",
        }[args.family],
        "config": vars(args), "best_val_loss": best_val, "history": history,
        "checkpoint": {"path": str(output), "sha256": _sha256(output)},
        "training_state": {"path": str(state_path), "sha256": _sha256(state_path)},
        "local_sources": _archive_sources(output),
    }
    _atomic_json(str(output) + ".report.json", report)
    return report


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=tuple(CANDIDATES), required=True)
    parser.add_argument("--stage", choices=("stage1_reconstruction", "stage3_forecast"),
                        required=True)
    parser.add_argument("--warm-checkpoint")
    parser.add_argument("--data-dir", default="output/foundation_tournament/data_cache")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tickers", default="ES,NQ,RTY,YM,GC,SI,CL,ZB,ZN")
    parser.add_argument("--timeframes", default="1min,3min,5min,15min,30min,60min")
    parser.add_argument("--max-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=256)
    parser.add_argument("--val-batches", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--chronos2-mode", choices=("close_only", "joint_ohlcv"),
                        default="close_only")
    parser.add_argument(
        "--univariate-input", choices=("close_only", "channel_independent_ohlcv"),
        default="channel_independent_ohlcv",
        help="v1/Bolt share one univariate model across either close or all OHLCV channels",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=4400)
    parser.add_argument("--validation-seed", type=int, default=5400)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int)
    return parser


def main():
    args = _parser().parse_args()
    args.tickers = tuple(x.strip() for x in args.tickers.split(",") if x.strip())
    args.timeframes = tuple(x.strip() for x in args.timeframes.split(",") if x.strip())
    train(args)


if __name__ == "__main__":
    main()
