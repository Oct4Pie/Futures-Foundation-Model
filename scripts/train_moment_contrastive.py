#!/usr/bin/env python3
"""Stage-2 contrastive MOMENT adaptation on the locked 5y/1y/1y futures corpus."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
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
from futures_foundation.finetune.native_contracts import (
    add_admission_argument, require_admission_from_args, validate_identity,
)
from futures_foundation.finetune.tournament import (
    MAX_CONTEXT, OOS_START, PARENT_LENGTH, TRAIN_START, VALIDATION_START,
)
from futures_foundation.finetune.tournament_data import (
    balanced_schedule, gather_contexts, load_adaptation_data, schedule_fingerprint,
)
from scripts.train_kronos_contrastive import _nt_xent
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
    raw = gather_contexts(big, starts, MAX_CONTEXT, parent_length=MAX_CONTEXT)
    views = []
    for half in (raw[:, :, : MAX_CONTEXT // 2], raw[:, :, MAX_CONTEXT // 2 :]):
        padded, mask = left_pad_contexts(np.transpose(half, (0, 2, 1)), 512)
        views.append((torch.as_tensor(padded, device=device),
                      torch.as_tensor(mask, device=device)))
    return views


def _pooled_embedding(model, x, mask):
    output = model.embed(x_enc=x, input_mask=mask, reduction="none").embeddings
    patch_mask = mask.to(output.dtype).unfold(
        1, int(model.patch_len), int(model.patch_len)
    ).mean(dim=-1)
    weight = patch_mask[:, None, :, None].to(output.dtype)
    pooled = (output * weight).sum(dim=2) / weight.sum(dim=2).clamp_min(1)
    return pooled.flatten(1)


def train(args):
    import torch
    validate_identity(
        "moment_small", model_id=args.model_id, model_revision=args.model_revision,
        source_revision=args.source_revision,
    )
    admission = require_admission_from_args(
        args, arm_key="moment_small", track="C", route="adjacent_half_contrastive",
        require_training=True,
    )
    repo, source = _validate_source(args.moment_repo, args.source_revision)
    if args.batch_size < 2 or args.max_steps < 1 or args.eval_every < 1 or args.val_batches < 1:
        raise ValueError("contrastive budgets require batch>=2 and positive steps")
    if not 0 < args.temperature <= 2:
        raise ValueError("temperature must lie in (0,2]")
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
    model = _load_task_model(repo, args, "embedding")
    parent = _load_parent_bundle(args.warm_checkpoint, model, args, "stage1_reconstruction")
    model = model.to(args.device)
    projection = torch.nn.Sequential(
        torch.nn.Linear(int(model.config.d_model) * 5, 512), torch.nn.GELU(),
        torch.nn.Linear(512, args.projection_dim),
    ).to(args.device)
    parameters = list(model.parameters()) + list(projection.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate,
                                  weight_decay=args.weight_decay, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_steps)
    start_step, best_val, history = 0, float("inf"), []
    if args.resume:
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state.get("config_signature") != signature:
            raise ValueError("exact resume refused: configuration changed")
        model.load_state_dict(state["model"]); projection.load_state_dict(state["projection"])
        optimizer.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"])
        start_step, best_val, history = state["step"], state["best_val"], state["history"]
        torch.set_rng_state(state["torch_rng"])
        if torch.cuda.is_available() and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])

    def loss_for(starts):
        views = _batch(big, starts, args.device)
        embeddings = []
        for x, mask in views:
            embeddings.append(projection(_pooled_embedding(model, x, mask)))
        return _nt_xent(embeddings[0], embeddings[1], args.temperature)

    def validate():
        model.eval(); projection.eval(); total = 0.0
        with torch.no_grad():
            for index in range(args.val_batches):
                lo = index * args.batch_size
                with torch.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=str(args.device).startswith("cuda")):
                    total += float(loss_for(val_schedule[lo:lo + args.batch_size]))
        return total / args.val_batches

    running = 0.0; running_steps = 0; first_grad = None
    model.train(); projection.train()
    for step in range(start_step, args.max_steps):
        lo = step * args.batch_size
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=str(args.device).startswith("cuda")):
            loss = loss_for(train_schedule[lo:lo + args.batch_size])
        loss.backward()
        backbone_grad = torch.sqrt(sum(
            parameter.grad.detach().float().square().sum()
            for parameter in model.encoder.parameters() if parameter.grad is not None
        ))
        if first_grad is None:
            first_grad = float(backbone_grad)
            if not np.isfinite(first_grad) or first_grad <= 0:
                raise RuntimeError("MOMENT Stage 2 did not reach the encoder backbone")
        torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
        optimizer.step(); scheduler.step()
        running += float(loss.detach()); running_steps += 1
        completed = step + 1
        if completed % args.eval_every == 0 or completed == args.max_steps:
            val = validate()
            row = {"step": completed, "train_loss": running / running_steps,
                   "val_loss": val, "backbone_grad_norm": float(backbone_grad),
                   "learning_rate": scheduler.get_last_lr()[0]}
            history.append(row)
            print(f"[moment-contrastive] step={completed} train={row['train_loss']:.6f} "
                  f"val={val:.6f} grad={row['backbone_grad_norm']:.4f}", flush=True)
            if val < best_val:
                best_val = val
                _atomic_torch_save(output, {
                    "schema_version": BUNDLE_SCHEMA, "stage": "stage2_contrastive",
                    "model": {"id": args.model_id, "revision": args.model_revision},
                    "model_state": model.state_dict(), "projection_state": projection.state_dict(),
                    "parent": parent,
                })
            _atomic_torch_save(state_path, {
                "schema_version": "ffm_moment_stage2_training_state_v1", "step": completed,
                "config_signature": signature, "best_val": best_val, "history": history,
                "model": model.state_dict(), "projection": projection.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            })
            running = 0.0; running_steps = 0; model.train(); projection.train()

    report = {
        "schema_version": "ffm_moment_stage2_train_v1", "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(), "parent": parent,
        "admission": admission,
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
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=8400)
    parser.add_argument("--validation-seed", type=int, default=5400)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--source-revision", default=SOURCE_REVISION)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    add_admission_argument(parser)
    return parser


def main():
    args = _parser().parse_args()
    args.tickers = tuple(value for value in args.tickers.split(",") if value)
    args.timeframes = tuple(value for value in args.timeframes.split(",") if value)
    train(args)


if __name__ == "__main__":
    main()
