#!/usr/bin/env python3
"""Stage-3 causal MOMENT forecasting from a verified Stage-2 parent."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from futures_foundation.finetune import moment_eval, tournament, tournament_data
from futures_foundation.finetune.moment_eval import left_pad_contexts
from futures_foundation.finetune.tournament import (
    FORECAST_HORIZON, MAX_CONTEXT, OOS_START, PARENT_LENGTH, TRAIN_START,
    VALIDATION_START,
)
from futures_foundation.finetune.tournament_data import (
    balanced_schedule, gather_parent, load_adaptation_data, schedule_fingerprint,
)
from scripts.train_moment_tournament import (
    BUNDLE_SCHEMA, MODEL_ID, MODEL_REVISION, SOURCE_REVISION, _atomic_json,
    _atomic_torch_save, _load_parent_bundle, _load_task_model, _sha256,
    _validate_source,
)


def _seed(seed):
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _signature(args):
    values = {key: value for key, value in vars(args).items() if key not in {"output", "resume"}}
    return hashlib.sha256(json.dumps(values, sort_keys=True, default=list).encode()).hexdigest()


def _archive_sources(output):
    destination = Path(str(output) + ".source")
    destination.mkdir(parents=True, exist_ok=True)
    result = {}
    for source in (Path(__file__).resolve(), Path(moment_eval.__file__).resolve(),
                   Path(tournament.__file__).resolve(), Path(tournament_data.__file__).resolve()):
        target = destination / source.name
        shutil.copyfile(source, target)
        result[source.name] = {"path": str(target), "sha256": _sha256(target)}
    return result


def _batch(big, starts, device):
    import torch
    parent = gather_parent(big, starts, PARENT_LENGTH)
    context = parent[:, :MAX_CONTEXT]
    future = np.transpose(parent[:, MAX_CONTEXT:MAX_CONTEXT + FORECAST_HORIZON], (0, 2, 1))
    padded, mask = left_pad_contexts(context, 512)
    return (torch.as_tensor(padded, device=device), torch.as_tensor(mask, device=device),
            torch.as_tensor(future, device=device))


def _scale_normalized_mse(forecast, future, context, input_mask):
    valid = input_mask[:, None, :].to(context.dtype)
    count = valid.sum(dim=2, keepdim=True).clamp_min(2)
    mean = (context * valid).sum(dim=2, keepdim=True) / count
    variance = ((context - mean).square() * valid).sum(dim=2, keepdim=True) / (count - 1)
    scale = variance.sqrt().clamp_min(1e-6)
    return ((forecast - future) / scale).square().mean()


def train(args):
    import torch
    repo, source = _validate_source(args.moment_repo, args.source_revision)
    if min(args.batch_size, args.max_steps, args.eval_every, args.val_batches) < 1:
        raise ValueError("forecast budgets must be positive")
    output = Path(args.output).resolve()
    state_path = Path(str(output) + ".train.pt")
    if output.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite {output} without --resume")
    if args.resume and not state_path.is_file():
        raise FileNotFoundError(f"exact resume state missing: {state_path}")
    signature = _signature(args)
    _seed(args.seed)
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
    model = _load_task_model(
        repo, args, "forecasting", forecast_horizon=FORECAST_HORIZON,
    )
    parent = _load_parent_bundle(args.warm_checkpoint, model, args, "stage2_contrastive")
    model = model.to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_steps)
    start_step, best_val, history = 0, float("inf"), []
    if args.resume:
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state.get("config_signature") != signature:
            raise ValueError("exact resume refused: configuration changed")
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_step, best_val, history = state["step"], state["best_val"], state["history"]
        torch.set_rng_state(state["torch_rng"])
        if torch.cuda.is_available() and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])

    def losses(starts):
        context, mask, future = _batch(big, starts, args.device)
        forecast = model(x_enc=context, input_mask=mask).forecast
        loss = _scale_normalized_mse(forecast, future, context, mask)
        persistence = context[:, :, -1:].expand_as(future)
        baseline = _scale_normalized_mse(persistence, future, context, mask)
        return loss, baseline

    def validate():
        model.eval(); total = 0.0; baseline = 0.0
        with torch.no_grad():
            for index in range(args.val_batches):
                lo = index * args.batch_size
                with torch.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=str(args.device).startswith("cuda")):
                    loss, base = losses(val_schedule[lo:lo + args.batch_size])
                total += float(loss); baseline += float(base)
        return total / args.val_batches, baseline / args.val_batches

    running = 0.0; running_steps = 0; first_grad = None
    model.train()
    for step in range(start_step, args.max_steps):
        lo = step * args.batch_size
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=str(args.device).startswith("cuda")):
            loss, _ = losses(train_schedule[lo:lo + args.batch_size])
        loss.backward()
        backbone_grad = torch.sqrt(sum(
            parameter.grad.detach().float().square().sum()
            for parameter in model.encoder.parameters() if parameter.grad is not None
        ))
        if first_grad is None:
            first_grad = float(backbone_grad)
            if not np.isfinite(first_grad) or first_grad <= 0:
                raise RuntimeError("MOMENT Stage 3 did not reach the encoder backbone")
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step(); scheduler.step()
        running += float(loss.detach()); running_steps += 1
        completed = step + 1
        if completed % args.eval_every == 0 or completed == args.max_steps:
            val, baseline = validate()
            skill = 1.0 - val / baseline if baseline > 0 else float("-inf")
            row = {"step": completed, "train_loss": running / running_steps,
                   "val_loss": val, "persistence_val_loss": baseline,
                   "skill_vs_persistence": skill, "backbone_grad_norm": float(backbone_grad),
                   "learning_rate": scheduler.get_last_lr()[0]}
            history.append(row)
            print(f"[moment-forecast] step={completed} train={row['train_loss']:.6f} "
                  f"val={val:.6f} skill={skill:.4f} grad={row['backbone_grad_norm']:.4f}",
                  flush=True)
            if val < best_val:
                best_val = val
                _atomic_torch_save(output, {
                    "schema_version": BUNDLE_SCHEMA, "stage": "stage3_forecast",
                    "model": {"id": args.model_id, "revision": args.model_revision},
                    "model_state": model.state_dict(), "parent": parent,
                    "input": {"context": MAX_CONTEXT, "horizon": FORECAST_HORIZON,
                              "normalization": "native_moment_revin_context_only",
                              "ohlcv_mode": "channel_independent_ohlcv"},
                })
            _atomic_torch_save(state_path, {
                "schema_version": "ffm_moment_stage3_training_state_v1", "step": completed,
                "config_signature": signature, "best_val": best_val, "history": history,
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            })
            running = 0.0; running_steps = 0; model.train()

    report = {
        "schema_version": "ffm_moment_stage3_train_v1", "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(), "parent": parent,
        "split": {"train_start": TRAIN_START, "validation_start": VALIDATION_START,
                  "oos_start": OOS_START, "oos_read": False},
        "data": {"streams": [stream["sid"] for stream in streams],
                 "train_windows": int(len(train_starts)),
                 "validation_windows": int(len(val_starts)),
                 "train_examples_seen": int(args.max_steps * args.batch_size)},
        "sampling": {"train_schedule_sha256": schedule_fingerprint(train_schedule, train_groups),
                     "validation_schedule_sha256": schedule_fingerprint(val_schedule, val_groups)},
        "config": vars(args), "best_val_loss": best_val, "first_backbone_grad_norm": first_grad,
        "history": history, "upstream": {"repo": str(repo), "source_sha256": _sha256(source)},
        "local_sources": _archive_sources(output),
        "checkpoint": {"path": str(output), "sha256": _sha256(output)},
        "training_state": {"path": str(state_path), "sha256": _sha256(state_path)},
    }
    _atomic_json(str(output) + ".report.json", report)
    return report


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/ssl_corpus_v2_6tf")
    parser.add_argument("--moment-repo", required=True)
    parser.add_argument("--warm-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tickers", default="ES,NQ,RTY,YM,GC,SI,CL,ZB,ZN")
    parser.add_argument("--timeframes", default="1min,3min,5min,15min,30min,60min")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=16384)
    parser.add_argument("--eval-every", type=int, default=512)
    parser.add_argument("--val-batches", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=8400)
    parser.add_argument("--validation-seed", type=int, default=5400)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--source-revision", default=SOURCE_REVISION)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    return parser


def main():
    args = _parser().parse_args()
    args.tickers = tuple(value for value in args.tickers.split(",") if value)
    args.timeframes = tuple(value for value in args.timeframes.split(",") if value)
    train(args)


if __name__ == "__main__":
    main()
