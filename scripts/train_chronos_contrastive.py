#!/usr/bin/env python3
"""Stage-2 Chronos-family contrastive adaptation with verified encoder gradients."""
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

from futures_foundation.finetune import chronos_family, tournament, tournament_data
from futures_foundation.finetune.chronos_family import CANDIDATES
from futures_foundation.finetune.native_contracts import (
    add_admission_argument, require_admission_from_args,
)
from futures_foundation.finetune.tournament import (
    MAX_CONTEXT, OOS_START, PARENT_LENGTH, TRAIN_START, VALIDATION_START,
)
from futures_foundation.finetune.tournament_data import (
    balanced_schedule, gather_parent, load_adaptation_data, schedule_fingerprint,
)
from scripts.train_chronos_tournament import (
    _archive_sources, _atomic_json, _atomic_save, _cpu_state, _load_pipeline, _sha256,
)
from scripts.train_kronos_contrastive import _nt_xent


def _seed(seed):
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _signature(args):
    value = {key: item for key, item in vars(args).items() if key not in {"output", "resume"}}
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=list).encode()).hexdigest()


def _series(context):
    return np.transpose(context, (0, 2, 1)).reshape(-1, context.shape[1])


def _channel_concat(hidden, batch_size, channels=5):
    return hidden.reshape(batch_size, channels, -1).flatten(1)


def _encoder_dim(candidate, model):
    if candidate.family == "chronos_2":
        return int(model.model_dim)
    return int(model.encoder.config.d_model)


def _encode(candidate, pipeline, model, context, device):
    """Return one representation per five-channel anchor using native encoder states."""
    import torch
    batch = len(context)
    values = _series(context)
    if candidate.family == "original_chronos_t5":
        ids, mask, _ = pipeline.tokenizer.context_input_transform(
            torch.as_tensor(values, dtype=torch.float32)
        )
        output = model.encoder(
            input_ids=ids.to(device), attention_mask=mask.to(device), return_dict=True,
        ).last_hidden_state
        weight = mask.to(device=device, dtype=output.dtype).unsqueeze(-1)
        pooled = (output * weight).sum(1) / weight.sum(1).clamp_min(1)
    elif candidate.family == "chronos_bolt":
        output, _, _, mask = model.encode(torch.as_tensor(values, device=device))
        weight = mask.to(output.dtype).unsqueeze(-1)
        pooled = (output * weight).sum(1) / weight.sum(1).clamp_min(1)
    else:
        tensor = torch.as_tensor(values, device=device)
        group_ids = torch.arange(batch, device=device).repeat_interleave(5)
        encoded, _, _, context_patches = model.encode(
            context=tensor, group_ids=group_ids, num_output_patches=1,
        )
        pooled = encoded[0][:, :context_patches].mean(dim=1)
    return _channel_concat(pooled, batch)


def _load_parent(path, candidate, model):
    import torch
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Chronos Stage-1 checkpoint missing: {path}")
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    if bundle.get("schema_version") != "ffm_chronos_tournament_bundle_v1":
        raise ValueError("unsupported Chronos parent schema")
    if bundle.get("stage") != "stage1_reconstruction":
        raise ValueError("Chronos Stage 2 requires a Stage-1 parent")
    if bundle.get("candidate") != candidate.manifest():
        raise ValueError("Chronos parent candidate identity mismatch")
    model.load_state_dict(bundle["model_state"], strict=True)
    return {"path": str(path), "sha256": _sha256(path), "stage": bundle["stage"]}


def train(args):
    from futures_foundation.finetune.native_training_routes import block_unadmitted_optimizer
    block_unadmitted_optimizer("scripts.train_chronos_contrastive.train")
    import torch
    if args.family not in CANDIDATES:
        raise ValueError(f"unsupported Chronos family: {args.family}")
    require_admission_from_args(
        args, arm_key=args.family, track="C",
        route="historical_hidden_state_contrastive", require_training=True,
    )
    if args.batch_size < 2 or min(args.max_steps, args.eval_every, args.val_batches) < 1:
        raise ValueError("contrastive budgets require batch>=2 and positive steps")
    candidate = CANDIDATES[args.family]
    output = Path(args.output).resolve(); state_path = Path(str(output) + ".train.pt")
    if output.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite {output} without --resume")
    if args.resume and not state_path.is_file():
        raise FileNotFoundError(f"exact resume state missing: {state_path}")
    signature = _signature(args); _seed(args.seed)
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
    pipeline, model = _load_pipeline(candidate, args.device)
    parent = _load_parent(args.warm_checkpoint, candidate, model)
    projection = torch.nn.Sequential(
        torch.nn.Linear(_encoder_dim(candidate, model) * 5, 512), torch.nn.GELU(),
        torch.nn.Linear(512, args.projection_dim),
    ).to(args.device)
    parameters = list(model.parameters()) + list(projection.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate,
                                  weight_decay=args.weight_decay, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_steps)
    start, best_val, history = 0, float("inf"), []
    if args.resume:
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state.get("config_signature") != signature:
            raise ValueError("exact resume refused: configuration changed")
        model.load_state_dict(state["model"]); projection.load_state_dict(state["projection"])
        optimizer.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"])
        start, best_val, history = state["step"], state["best_val"], state["history"]
        torch.set_rng_state(state["torch_rng"])
        if torch.cuda.is_available() and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])

    def loss_for(starts):
        context = gather_parent(big, starts, PARENT_LENGTH)[:, :MAX_CONTEXT]
        first, second = context[:, :MAX_CONTEXT // 2], context[:, MAX_CONTEXT // 2:]
        return _nt_xent(
            projection(_encode(candidate, pipeline, model, first, args.device)),
            projection(_encode(candidate, pipeline, model, second, args.device)),
            args.temperature,
        )

    def validate():
        model.eval(); projection.eval(); total = 0.0
        with torch.no_grad():
            for step in range(args.val_batches):
                lo = step * args.batch_size
                with torch.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=str(args.device).startswith("cuda")):
                    total += float(loss_for(val_schedule[lo:lo + args.batch_size]))
        return total / args.val_batches

    running = 0.0; running_steps = 0; first_grad = None
    model.train(); projection.train()
    for step in range(start, args.max_steps):
        lo = step * args.batch_size; optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=str(args.device).startswith("cuda")):
            loss = loss_for(train_schedule[lo:lo + args.batch_size])
        loss.backward()
        grad = torch.sqrt(sum(
            value.grad.detach().float().square().sum()
            for value in model.encoder.parameters() if value.grad is not None
        ))
        if first_grad is None:
            first_grad = float(grad)
            if not np.isfinite(first_grad) or first_grad <= 0:
                raise RuntimeError("Chronos Stage 2 did not reach the encoder backbone")
        torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
        optimizer.step(); scheduler.step(); running += float(loss.detach()); running_steps += 1
        completed = step + 1
        if completed % args.eval_every == 0 or completed == args.max_steps:
            val = validate()
            row = {"step": completed, "train_loss": running / running_steps,
                   "val_loss": val, "backbone_grad_norm": float(grad),
                   "learning_rate": scheduler.get_last_lr()[0]}
            history.append(row)
            print(f"[{args.family}-contrastive] step={completed} train={row['train_loss']:.6f} "
                  f"val={val:.6f} grad={row['backbone_grad_norm']:.4f}", flush=True)
            if val < best_val:
                best_val = val
                _atomic_save(output, {
                    "schema_version": "ffm_chronos_tournament_bundle_v1",
                    "stage": "stage2_contrastive", "candidate": candidate.manifest(),
                    "model_state": _cpu_state(model), "projection_state": _cpu_state(projection),
                    "parent": parent,
                    "input": {"context": MAX_CONTEXT, "channels": "joint_ohlcv_representation"},
                })
            _atomic_save(state_path, {
                "schema_version": "ffm_chronos_stage2_training_state_v1", "step": completed,
                "config_signature": signature, "best_val": best_val, "history": history,
                "model": _cpu_state(model), "projection": _cpu_state(projection),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            })
            running = 0.0; running_steps = 0; model.train(); projection.train()
    report = {
        "schema_version": "ffm_chronos_stage2_train_v1", "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(), "parent": parent,
        "model": candidate.manifest(),
        "split": {"train_start": TRAIN_START, "validation_start": VALIDATION_START,
                  "oos_start": OOS_START, "oos_read": False},
        "data": {"streams": [stream["sid"] for stream in streams],
                 "train_windows": int(len(train_starts)), "validation_windows": int(len(val_starts)),
                 "train_examples_seen": int(args.max_steps * args.batch_size)},
        "sampling": {"train_schedule_sha256": schedule_fingerprint(train_schedule, train_groups),
                     "validation_schedule_sha256": schedule_fingerprint(val_schedule, val_groups)},
        "config": vars(args), "best_val_loss": best_val,
        "first_backbone_grad_norm": first_grad, "history": history,
        "local_sources": _archive_sources(output, extra_sources=(Path(__file__),)),
        "checkpoint": {"path": str(output), "sha256": _sha256(output)},
        "training_state": {"path": str(state_path), "sha256": _sha256(state_path)},
    }
    _atomic_json(str(output) + ".report.json", report)
    return report


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=tuple(CANDIDATES), required=True)
    parser.add_argument("--data-dir", default="output/foundation_tournament/data_cache")
    parser.add_argument("--warm-checkpoint", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--tickers", default="ES,NQ,RTY,YM,GC,SI,CL,ZB,ZN")
    parser.add_argument("--timeframes", default="1min,3min,5min,15min,30min,60min")
    parser.add_argument("--max-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=128)
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
    parser.add_argument("--resume", action="store_true")
    add_admission_argument(parser)
    return parser


def main():
    args = _parser().parse_args()
    args.tickers = tuple(value for value in args.tickers.split(",") if value)
    args.timeframes = tuple(value for value in args.timeframes.split(",") if value)
    train(args)


if __name__ == "__main__":
    main()
