#!/usr/bin/env python3
"""Leak-safe staged TTM-R2 adaptation on the locked 5y/1y split."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from futures_foundation.finetune import tournament, tournament_data
from futures_foundation.finetune.foundation_roster import get_arm
from futures_foundation.finetune.tournament import (
    FORECAST_HORIZON, OOS_START, TRAIN_START, VALIDATION_START,
)
from futures_foundation.finetune.tournament_data import (
    balanced_schedule, gather_parent, load_adaptation_data, schedule_fingerprint,
)


CONTEXT = 512
PARENT_LENGTH = CONTEXT + FORECAST_HORIZON
FREQUENCY_TOKEN = {
    "1min": 1, "3min": 0, "5min": 3, "15min": 5, "30min": 6, "60min": 7,
}


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
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=float) + "\n")
    os.replace(tmp, path)


def _snapshot(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _config_signature(args):
    value = {key: value for key, value in vars(args).items()
             if key not in {"output", "resume", "stop_after_step"}}
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, default=list, separators=(",", ":"),
    ).encode()).hexdigest()


def _validate_source(repo, expected):
    repo = Path(repo).expanduser().resolve()
    if not (repo / "tsfm_public" / "toolkit" / "get_model.py").is_file():
        raise FileNotFoundError(f"official granite-tsfm source missing: {repo}")
    actual = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if actual != expected:
        raise ValueError(f"granite-tsfm revision mismatch: expected {expected}, got {actual}")
    return repo


def _normalize_parent(raw, context_start, context_length=CONTEXT):
    """Official per-channel scaling, fitted strictly on the causal context."""
    raw = np.asarray(raw, np.float32)
    context = raw[:, context_start:context_start + context_length]
    mean = context.mean(axis=1, keepdims=True)
    std = context.std(axis=1, keepdims=True)
    return (raw - mean) / np.maximum(std, 1e-5), mean, std


def _load_parent(path, required_stage):
    import torch
    path = Path(path).expanduser().resolve()
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    if bundle.get("schema_version") != "ffm_ttm_staged_bundle_v1":
        raise ValueError(f"unsupported TTM parent schema in {path}")
    if bundle.get("stage") != required_stage:
        raise ValueError(
            f"TTM parent must be {required_stage}, got {bundle.get('stage')!r}"
        )
    return path, bundle, {"path": str(path), "sha256": _sha256(path),
                          "stage": required_stage}


def _frequency_tokens(group_ids, streams):
    return np.asarray([FREQUENCY_TOKEN[streams[int(group)]["tf"]] for group in group_ids], np.int64)


def _load_model(args):
    from tsfm_public.toolkit.get_model import get_model
    return get_model(
        args.model_id, context_length=CONTEXT, prediction_length=FORECAST_HORIZON,
        model_revision=args.model_revision, num_input_channels=5,
        enable_forecast_channel_mixing=True,
    )


def train(args):
    import torch
    arm = get_arm("ttm_r2")
    if (args.model_id, args.model_revision, args.source_revision) != (
            arm.model_id, arm.model_revision, arm.source_revision):
        raise ValueError("TTM model/source pins do not match the admitted arm")
    source = _validate_source(args.ttm_repo, args.source_revision)
    if min(args.max_steps, args.batch_size, args.eval_every, args.val_batches) < 1:
        raise ValueError("training and validation budgets must be positive")
    if args.stage == "stage1_reconstruction" and args.warm_checkpoint:
        raise ValueError("Stage 1 must start from the admitted pretrained TTM checkpoint")
    if args.stage == "stage3_forecast" and not args.warm_checkpoint:
        raise ValueError("Stage 3 requires a verified Stage 2 checkpoint")
    output = Path(args.output).resolve()
    state_path = Path(str(output) + ".train.pt")
    if output.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite {output} without --resume")
    if args.resume and not state_path.is_file():
        raise FileNotFoundError(f"exact resume state missing: {state_path}")

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    streams, big, train_starts, val_starts, groups = load_adaptation_data(
        args.data_dir, args.tickers, args.timeframes, parent_length=PARENT_LENGTH,
    )
    train_schedule, train_groups = balanced_schedule(
        train_starts, groups["train_bounds"], args.max_steps * args.batch_size, args.seed,
    )
    val_schedule, val_groups = balanced_schedule(
        val_starts, groups["val_bounds"], args.val_batches * args.batch_size,
        args.validation_seed,
    )
    model = _load_model(args).to(args.device)
    parent = None
    if args.stage == "stage3_forecast":
        _, parent_bundle, parent = _load_parent(
            args.warm_checkpoint, "stage2_contrastive",
        )
        model.load_state_dict(parent_bundle["model_state"], strict=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_steps)
    signature = _config_signature(args)
    history, start, best_val, best_model = [], 0, float("inf"), _snapshot(model)
    if args.resume:
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state.get("config_signature") != signature:
            raise ValueError("exact resume refused: configuration changed")
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start, history = int(state["step"]), state["history"]
        best_val, best_model = float(state["best_val"]), state["best_model"]
        random.setstate(state["python_rng"]); np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        if torch.cuda.is_available() and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])

    def batch(schedule, schedule_groups, step):
        lo = step * args.batch_size
        starts = schedule[lo:lo + args.batch_size]
        gids = schedule_groups[lo:lo + args.batch_size]
        raw = gather_parent(big, starts, PARENT_LENGTH)
        if args.stage == "stage1_reconstruction":
            visible = CONTEXT - FORECAST_HORIZON
            values, _, _ = _normalize_parent(raw, 0, visible)
            context = values[:, :CONTEXT].copy()
            context[:, visible:] = 0.0
            future = values[:, visible:CONTEXT]
        else:
            values, _, _ = _normalize_parent(raw, 0)
            context = values[:, :CONTEXT]
            future = values[:, CONTEXT:CONTEXT + FORECAST_HORIZON]
        return (
            torch.as_tensor(context, device=args.device),
            torch.as_tensor(future, device=args.device),
            torch.as_tensor(_frequency_tokens(gids, streams), device=args.device),
        )

    def validate():
        model.eval(); total = 0.0
        with torch.no_grad():
            for val_step in range(args.val_batches):
                context, future, freq = batch(val_schedule, val_groups, val_step)
                with torch.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=str(args.device).startswith("cuda")):
                    total += float(model(
                        past_values=context, future_values=future, freq_token=freq,
                    ).loss)
        return total / args.val_batches

    running, running_steps = 0.0, 0
    model.train()
    end_step = min(args.max_steps, args.stop_after_step or args.max_steps)
    for step in range(start, end_step):
        context, future, freq = batch(train_schedule, train_groups, step)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=str(args.device).startswith("cuda")):
            loss = model(past_values=context, future_values=future, freq_token=freq).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step(); scheduler.step()
        running += float(loss.detach()); running_steps += 1
        completed = step + 1
        if completed % args.eval_every == 0 or completed == end_step:
            val = validate()
            row = {"step": completed, "train_loss": running / running_steps,
                   "val_loss": val, "learning_rate": scheduler.get_last_lr()[0]}
            history.append(row)
            print(f"[ttm-r2:{args.stage}] step={completed} "
                  f"train={row['train_loss']:.6f} val={val:.6f}", flush=True)
            running, running_steps = 0.0, 0
            if val < best_val:
                best_val, best_model = val, _snapshot(model)
                _atomic_save(output, {
                    "schema_version": "ffm_ttm_staged_bundle_v1",
                    "stage": args.stage, "model_state": best_model,
                    "arm": arm.manifest(), "parent": parent,
                    "input": {"context": CONTEXT, "horizon": FORECAST_HORIZON,
                              "normalization": "context_only_per_channel_zscore",
                              "ohlcv_mode": "joint_ohlcv_random_init_channel_mix_head",
                              "stage1_mask": "zero_normalized_suffix_16"},
                })
            _atomic_save(state_path, {
                "schema_version": "ffm_ttm_staged_training_state_v1",
                "config_signature": signature, "step": completed,
                "model": _snapshot(model), "best_model": best_model, "best_val": best_val,
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "history": history, "python_rng": random.getstate(),
                "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            })
            model.train()

    source_dir = Path(str(output) + ".source"); source_dir.mkdir(parents=True, exist_ok=True)
    archived = {}
    for path in (Path(__file__).resolve(), Path(tournament.__file__).resolve(),
                 Path(tournament_data.__file__).resolve()):
        target = source_dir / path.name; shutil.copyfile(path, target)
        archived[path.name] = {"path": str(target), "sha256": _sha256(target)}
    manifest = next(path for path in (
        Path(args.data_dir) / "TOURNAMENT_CACHE.json", Path(args.data_dir) / "MANIFEST.json",
    ) if path.is_file())
    report = {
        "schema_version": "ffm_ttm_staged_train_v1",
        "status": "complete" if end_step == args.max_steps else "partial",
        "created_utc": datetime.now(timezone.utc).isoformat(), "arm": arm.manifest(),
        "stage": args.stage, "parent": parent,
        "split": {"train_start": TRAIN_START, "validation_start": VALIDATION_START,
                  "oos_start": OOS_START, "oos_read": False},
        "upstream": {"repo": str(source), "revision": args.source_revision},
        "local_sources": archived,
        "data": {"streams": [stream["sid"] for stream in streams],
                 "data_manifest_sha256": _sha256(manifest),
                 "train_windows": int(len(train_starts)),
                 "validation_windows": int(len(val_starts)),
                 "examples_seen": int(args.max_steps * args.batch_size)},
        "sampling": {"train_schedule_sha256": schedule_fingerprint(train_schedule, train_groups),
                     "validation_schedule_sha256": schedule_fingerprint(val_schedule, val_groups)},
        "config": vars(args), "parameters": sum(p.numel() for p in model.parameters()),
        "best_val_loss": best_val, "history": history,
        "checkpoint": {"path": str(output), "sha256": _sha256(output)},
        "training_state": {"path": str(state_path), "sha256": _sha256(state_path)},
    }
    _atomic_json(str(output) + ".report.json", report)
    return report


def _parser():
    arm = get_arm("ttm_r2")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="output/foundation_tournament/data_cache")
    parser.add_argument("--ttm-repo", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stage", required=True,
                        choices=("stage1_reconstruction", "stage3_forecast"))
    parser.add_argument("--warm-checkpoint")
    parser.add_argument("--tickers", default="ES,NQ,RTY,YM,GC,SI,CL,ZB,ZN")
    parser.add_argument("--timeframes", default="1min,3min,5min,15min,30min,60min")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-steps", type=int, default=256)
    parser.add_argument("--eval-every", type=int, default=512)
    parser.add_argument("--val-batches", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=4400)
    parser.add_argument("--validation-seed", type=int, default=5400)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-step", type=int)
    parser.add_argument("--source-revision", default=arm.source_revision)
    parser.add_argument("--model-id", default=arm.model_id)
    parser.add_argument("--model-revision", default=arm.model_revision)
    return parser


def main():
    args = _parser().parse_args()
    args.tickers = tuple(value for value in args.tickers.split(",") if value)
    args.timeframes = tuple(value for value in args.timeframes.split(",") if value)
    train(args)


if __name__ == "__main__":
    main()
