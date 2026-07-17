#!/usr/bin/env python3
"""Leak-safe native masked-reconstruction adaptation of MOMENT for the 5y/1y/1y study."""
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

from futures_foundation.finetune.moment_eval import left_pad_contexts
from futures_foundation.finetune import moment_eval, tournament, tournament_data
from futures_foundation.finetune.tournament import (
    MAX_CONTEXT, OOS_START, PARENT_LENGTH, TRAIN_START, VALIDATION_START,
)
from futures_foundation.finetune.tournament_data import (
    balanced_schedule, gather_contexts, load_adaptation_data, schedule_fingerprint,
)


SOURCE_REVISION = "38f7310ad594100747ca2a8357e9c7ca7d323e0e"
MODEL_ID = "AutonLab/MOMENT-1-small"
MODEL_REVISION = "411e288267f82cce86296dbe4d6c8bc533cc162f"
BUNDLE_SCHEMA = "ffm_moment_staged_bundle_v1"


def _sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _atomic_torch_save(path, value):
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


def _archive_sources(output):
    destination = Path(str(output) + ".source")
    destination.mkdir(parents=True, exist_ok=True)
    archived = {}
    for source in (Path(__file__).resolve(), Path(moment_eval.__file__).resolve(),
                   Path(tournament.__file__).resolve(), Path(tournament_data.__file__).resolve()):
        target = destination / source.name
        shutil.copyfile(source, target)
        archived[source.name] = {"path": str(target), "sha256": _sha256(target)}
    return archived


def _config_signature(args):
    value = {key: value for key, value in vars(args).items()
             if key not in {"resume", "output"}}
    raw = json.dumps(value, sort_keys=True, default=list, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _validate_source(repo, revision):
    repo = Path(repo).expanduser().resolve()
    source = repo / "momentfm" / "models" / "moment.py"
    if not source.is_file():
        raise FileNotFoundError(f"official MOMENT source missing: {source}")
    actual = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True,
    ).strip()
    if actual != revision:
        raise ValueError(f"MOMENT revision mismatch: expected {revision}, got {actual}")
    return repo, source


def _seed(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_task_model(repo, args, task_name, **task_kwargs):
    sys.path.insert(0, str(repo))
    try:
        from momentfm import MOMENTPipeline
    finally:
        sys.path.pop(0)
    model = MOMENTPipeline.from_pretrained(
        args.model_id, revision=args.model_revision,
        model_kwargs={
            "task_name": task_name, "mask_ratio": getattr(args, "mask_ratio", 0.0),
            "freeze_encoder": False, "freeze_embedder": False, "freeze_head": False,
            "enable_gradient_checkpointing": getattr(args, "gradient_checkpointing", False),
            **task_kwargs,
        },
    )
    model.init()
    return model


def _load_model(repo, args):
    return _load_task_model(repo, args, "reconstruction")


def _load_parent_bundle(path, model, args, expected_stage):
    """Load transferable MOMENT tensors from a typed parent and attest its identity."""
    import torch
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"MOMENT parent checkpoint missing: {path}")
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    if bundle.get("schema_version") != BUNDLE_SCHEMA:
        raise ValueError("unsupported MOMENT staged checkpoint schema")
    if bundle.get("stage") != expected_stage:
        raise ValueError(
            f"MOMENT parent stage mismatch: expected {expected_stage}, got {bundle.get('stage')}"
        )
    if bundle.get("model") != {"id": args.model_id, "revision": args.model_revision}:
        raise ValueError("MOMENT parent model identity mismatch")
    current = model.state_dict()
    source = bundle["model_state"]
    transferable = {
        key: value for key, value in source.items()
        if key in current and current[key].shape == value.shape and not key.startswith("head.")
    }
    required = [key for key in current if key.startswith(("encoder.", "patch_embedding."))]
    missing = sorted(set(required) - set(transferable))
    if missing:
        raise ValueError(f"MOMENT parent misses {len(missing)} backbone tensors")
    model.load_state_dict(transferable, strict=False)
    return {
        "path": str(path), "sha256": _sha256(path), "stage": bundle["stage"],
        "transferred_tensors": len(transferable),
    }


def _batch(big, starts, context, device):
    import torch
    raw = gather_contexts(big, starts, context, parent_length=MAX_CONTEXT)
    padded, mask = left_pad_contexts(np.transpose(raw, (0, 2, 1)), 512)
    return (torch.as_tensor(padded, device=device),
            torch.as_tensor(mask, device=device))


def _masked_loss(output, original, input_mask):
    # Official MOMENT pretraining contract: MSE only on valid, randomly hidden positions.
    # The public model returns de-normalized values; raw futures volume is orders of magnitude
    # larger than price returns. Normalize the residual by PAST-CONTEXT channel scale so one
    # unit convention cannot own the objective. Statistics use valid input only (padding never
    # contributes), and raw inputs still flow through MOMENT's native RevIN path.
    hidden = input_mask * (1 - output.pretrain_mask)
    valid = input_mask[:, None, :].to(original.dtype)
    count = valid.sum(dim=2, keepdim=True).clamp_min(2)
    mean = (original * valid).sum(dim=2, keepdim=True) / count
    variance = ((original - mean).square() * valid).sum(dim=2, keepdim=True) / (count - 1)
    scale = variance.sqrt().clamp_min(1e-6)
    squared = ((output.reconstruction - original) / scale).square()
    return (squared * hidden[:, None, :]).sum() / (hidden.sum().clamp_min(1) * original.shape[1])


def _stage_bundle(model, args):
    return {
        "schema_version": BUNDLE_SCHEMA,
        "stage": "stage1_reconstruction",
        "model": {"id": args.model_id, "revision": args.model_revision},
        "model_state": model.state_dict(),
        "parent": None,
    }


def train(args):
    import torch
    repo, source = _validate_source(args.moment_repo, args.source_revision)
    if args.context not in (64, 128, 256) or args.context % 8:
        raise ValueError("MOMENT tournament context must be one of 64,128,256")
    if args.max_steps < 1 or args.eval_every < 1 or args.val_batches < 1:
        raise ValueError("step/evaluation budgets must be positive")
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
        train_starts, groups["train_bounds"], args.max_steps * args.batch_size, args.seed,
    )
    val_schedule, val_groups = balanced_schedule(
        val_starts, groups["val_bounds"], args.val_batches * args.batch_size,
        args.validation_seed,
    )
    model = _load_model(repo, args).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_steps)
    start_step, best_val, history = 0, float("inf"), []
    if args.resume:
        state = torch.load(state_path, map_location="cpu")
        if state.get("config_signature") != signature:
            raise ValueError("exact resume refused: configuration changed")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_step, best_val, history = state["step"], state["best_val"], state["history"]
        torch.set_rng_state(state["torch_rng"])
        if torch.cuda.is_available() and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])

    def validate(step):
        model.eval()
        total = 0.0
        torch.manual_seed(args.validation_seed)
        with torch.no_grad():
            for offset in range(0, len(val_schedule), args.batch_size):
                x, mask = _batch(
                    big, val_schedule[offset:offset + args.batch_size], args.context, args.device,
                )
                with torch.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=str(args.device).startswith("cuda")):
                    out = model(x_enc=x, input_mask=mask)
                    loss = _masked_loss(out, x, mask)
                total += float(loss)
        return total / args.val_batches

    model.train()
    running = 0.0
    running_steps = 0
    for step in range(start_step, args.max_steps):
        lo = step * args.batch_size
        x, mask = _batch(big, train_schedule[lo:lo + args.batch_size], args.context, args.device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=str(args.device).startswith("cuda")):
            out = model(x_enc=x, input_mask=mask)
            loss = _masked_loss(out, x, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        running += float(loss.detach())
        running_steps += 1
        completed = step + 1
        if completed % args.eval_every == 0 or completed == args.max_steps:
            val = validate(completed)
            row = {"step": completed, "train_loss": running / running_steps,
                   "val_loss": val, "learning_rate": scheduler.get_last_lr()[0]}
            history.append(row)
            print(f"[moment-train] step={completed} train={row['train_loss']:.6f} "
                  f"val={val:.6f}", flush=True)
            running = 0.0
            running_steps = 0
            if val < best_val:
                best_val = val
                _atomic_torch_save(output, _stage_bundle(model, args))
            _atomic_torch_save(state_path, {
                "schema_version": "ffm_moment_training_state_v1", "step": completed,
                "config_signature": signature,
                "best_val": best_val, "history": history, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            })
            model.train()

    report = {
        "schema_version": "ffm_moment_tournament_train_v1", "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": {"id": args.model_id, "revision": args.model_revision,
                  "parameters": sum(p.numel() for p in model.parameters())},
        "upstream": {"repo": str(repo), "revision": args.source_revision,
                     "source_sha256": _sha256(source)},
        "local_sources": _archive_sources(output),
        "split": {"train_start": TRAIN_START, "validation_start": VALIDATION_START,
                  "oos_start": OOS_START, "oos_read": False},
        "data": {"streams": [stream["sid"] for stream in streams],
                 "data_manifest_sha256": _sha256(
                     next(path for path in (
                         Path(args.data_dir) / "TOURNAMENT_CACHE.json",
                         Path(args.data_dir) / "MANIFEST.json",
                     ) if path.is_file())),
                 "train_windows": int(len(train_starts)),
                 "validation_windows": int(len(val_starts)),
                 "train_examples_seen": int(args.max_steps * args.batch_size)},
        "sampling": {"train_schedule_sha256": schedule_fingerprint(
                          train_schedule, train_groups),
                     "validation_schedule_sha256": schedule_fingerprint(
                          val_schedule, val_groups)},
        "config": vars(args), "best_val_loss": best_val, "history": history,
        "checkpoint": {"path": str(output), "sha256": _sha256(output)},
        "training_state": {"path": str(state_path), "sha256": _sha256(state_path)},
    }
    _atomic_json(str(output) + ".report.json", report)
    return report


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/ssl_corpus_v2_6tf")
    parser.add_argument("--moment-repo", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tickers", default="ES,NQ,RTY,YM,GC,SI,CL,ZB,ZN")
    parser.add_argument("--timeframes", default="1min,3min,5min,15min,30min,60min")
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--val-batches", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--mask-ratio", type=float, default=0.3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validation-seed", type=int, default=5400)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--source-revision", default=SOURCE_REVISION)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    return parser


def main():
    args = _parser().parse_args()
    args.tickers = tuple(x for x in args.tickers.split(",") if x)
    args.timeframes = tuple(x for x in args.timeframes.split(",") if x)
    train(args)


if __name__ == "__main__":
    main()
