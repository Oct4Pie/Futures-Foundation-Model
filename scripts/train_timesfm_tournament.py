#!/usr/bin/env python3
"""Leak-safe staged TimesFM 2.5 LoRA adaptation on channel-independent OHLCV."""
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
from futures_foundation.finetune.native_contracts import (
    add_admission_argument, require_admission_from_args,
)
from futures_foundation.finetune.tournament import (
    FORECAST_HORIZON, MAX_CONTEXT, OOS_START, TRAIN_START, VALIDATION_START,
)
from futures_foundation.finetune.tournament_data import (
    balanced_schedule, gather_parent, load_adaptation_data, schedule_fingerprint,
)


PARENT_LENGTH = MAX_CONTEXT + FORECAST_HORIZON


def _sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _atomic_save(path, value):
    import torch
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp"); torch.save(value, tmp); os.replace(tmp, path)


def _atomic_json(path, value):
    path = Path(path); tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=float) + "\n"); os.replace(tmp, path)


def _validate_source(repo, expected):
    repo = Path(repo).expanduser().resolve()
    if not (repo / "timesfm-forecasting" / "examples" / "finetuning" / "finetune_lora.py").is_file():
        raise FileNotFoundError(f"official TimesFM fine-tuning source missing: {repo}")
    actual = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if actual != expected:
        raise ValueError(f"TimesFM revision mismatch: expected {expected}, got {actual}")
    return repo


def _channel_independent(parent, stage="stage3_forecast"):
    """Convert ``[anchor,time,OHLCV]`` into five shared-weight univariate series."""
    value = np.asarray(parent, np.float32)
    if value.ndim != 3 or value.shape[1:] != (PARENT_LENGTH, 5):
        raise ValueError(f"invalid TimesFM parent shape: {value.shape}")
    if stage == "stage1_reconstruction":
        context_len = MAX_CONTEXT - 2 * FORECAST_HORIZON
        target_start = context_len
    else:
        context_len = MAX_CONTEXT
        target_start = MAX_CONTEXT
    context = value[:, :context_len].transpose(0, 2, 1).reshape(-1, context_len)
    future = value[:, target_start:target_start + FORECAST_HORIZON].transpose(0, 2, 1).reshape(
        -1, FORECAST_HORIZON
    )
    return context, future


def _load_parent(path, arm, required_stage, args):
    import torch
    from peft import set_peft_model_state_dict
    path = Path(path).expanduser().resolve()
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    if bundle.get("schema_version") != "ffm_timesfm_staged_bundle_v1":
        raise ValueError(f"unsupported TimesFM parent schema in {path}")
    if bundle.get("stage") != required_stage:
        raise ValueError(f"TimesFM parent must be {required_stage}, got {bundle.get('stage')!r}")
    if bundle.get("arm") != arm.manifest():
        raise ValueError("TimesFM parent arm identity mismatch")
    expected = {"rank": args.lora_rank, "alpha": args.lora_alpha,
                "dropout": args.lora_dropout}
    if bundle.get("lora") != expected:
        raise ValueError("TimesFM parent LoRA configuration mismatch")
    return path, bundle, {"path": str(path), "sha256": _sha256(path),
                          "stage": required_stage}


def _adapter_state(model):
    from peft import get_peft_model_state_dict
    return {key: value.detach().cpu().clone()
            for key, value in get_peft_model_state_dict(model).items()}


def _load_model(args):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import TimesFm2_5ModelForPrediction
    base = TimesFm2_5ModelForPrediction.from_pretrained(
        args.model_id, revision=args.model_revision, dtype=torch.bfloat16,
    )
    config = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_alpha,
        target_modules="all-linear", lora_dropout=args.lora_dropout, bias="none",
    )
    return get_peft_model(base, config).to(args.device)


def _signature(args):
    value = {key: value for key, value in vars(args).items()
             if key not in {"output", "resume", "stop_after_step"}}
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, default=list, separators=(",", ":"),
    ).encode()).hexdigest()


def train(args):
    import torch
    from peft import set_peft_model_state_dict
    arm = get_arm("timesfm25")
    require_admission_from_args(
        args, arm_key="timesfm25", track="F",
        route="historical_universal_stage_chain", require_training=True,
    )
    if (args.model_id, args.model_revision, args.source_revision) != (
            arm.model_id, arm.model_revision, arm.source_revision):
        raise ValueError("TimesFM model/source pins do not match the admitted arm")
    source = _validate_source(args.timesfm_repo, args.source_revision)
    if min(args.max_steps, args.batch_size, args.eval_every, args.val_batches) < 1:
        raise ValueError("training and validation budgets must be positive")
    if args.stage == "stage1_reconstruction" and args.warm_checkpoint:
        raise ValueError("Stage 1 must start from the admitted pretrained TimesFM checkpoint")
    if args.stage == "stage3_forecast" and not args.warm_checkpoint:
        raise ValueError("Stage 3 requires a verified Stage 2 checkpoint")
    output = Path(args.output).resolve(); state_path = Path(str(output) + ".train.pt")
    if output.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite {output} without --resume")
    if args.resume and not state_path.is_file():
        raise FileNotFoundError(f"exact resume state missing: {state_path}")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
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
    model = _load_model(args)
    parent = None
    if args.stage == "stage3_forecast":
        _, parent_bundle, parent = _load_parent(
            args.warm_checkpoint, arm, "stage2_contrastive", args,
        )
        set_peft_model_state_dict(model, parent_bundle["adapter_state"])
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_steps)
    signature = _signature(args)
    start, history, best_val, best_adapter = 0, [], float("inf"), _adapter_state(model)
    if args.resume:
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state.get("config_signature") != signature:
            raise ValueError("exact resume refused: configuration changed")
        set_peft_model_state_dict(model, state["adapter"])
        optimizer.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"])
        start, history = int(state["step"]), state["history"]
        best_val, best_adapter = float(state["best_val"]), state["best_adapter"]
        random.setstate(state["python_rng"]); np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        if torch.cuda.is_available() and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])

    def batch(schedule, step):
        lo = step * args.batch_size
        raw = gather_parent(big, schedule[lo:lo + args.batch_size], PARENT_LENGTH)
        context, future = _channel_independent(raw, args.stage)
        return (torch.as_tensor(context, device=args.device),
                torch.as_tensor(future, device=args.device))

    def loss_for(context, future):
        return model(
            past_values=context, future_values=future,
            forecast_context_len=context.shape[1],
        ).loss

    def validate():
        model.eval(); total = 0.0
        with torch.no_grad():
            for step in range(args.val_batches):
                context, future = batch(val_schedule, step)
                total += float(loss_for(context, future))
        return total / args.val_batches

    end_step = min(args.max_steps, args.stop_after_step or args.max_steps)
    running, running_steps = 0.0, 0; model.train()
    for step in range(start, end_step):
        context, future = batch(train_schedule, step)
        optimizer.zero_grad(set_to_none=True); loss = loss_for(context, future)
        loss.backward(); torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        optimizer.step(); scheduler.step()
        running += float(loss.detach()); running_steps += 1; completed = step + 1
        if completed % args.eval_every == 0 or completed == end_step:
            val = validate()
            row = {"step": completed, "train_loss": running / running_steps,
                   "val_loss": val, "learning_rate": scheduler.get_last_lr()[0]}
            history.append(row); running, running_steps = 0.0, 0
            print(f"[timesfm25:{args.stage}] step={completed} "
                  f"train={row['train_loss']:.6f} val={val:.6f}", flush=True)
            if val < best_val:
                best_val, best_adapter = val, _adapter_state(model)
                _atomic_save(output, {
                    "schema_version": "ffm_timesfm_staged_bundle_v1",
                    "stage": args.stage, "adapter_state": best_adapter,
                    "arm": arm.manifest(), "parent": parent,
                    "lora": {"rank": args.lora_rank, "alpha": args.lora_alpha,
                             "dropout": args.lora_dropout},
                    "input": {"context": int(context.shape[1]), "horizon": FORECAST_HORIZON,
                              "normalization": "native_revin_raw_values",
                              "ohlcv_mode": "channel_independent_ohlcv"},
                })
            _atomic_save(state_path, {
                "schema_version": "ffm_timesfm_staged_training_state_v1",
                "config_signature": signature, "step": completed,
                "adapter": _adapter_state(model), "best_adapter": best_adapter,
                "best_val": best_val, "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "history": history,
                "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state(),
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
        "schema_version": "ffm_timesfm_staged_train_v1",
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
                 "anchors_seen": int(args.max_steps * args.batch_size),
                 "univariate_series_seen": int(args.max_steps * args.batch_size * 5)},
        "sampling": {"train_schedule_sha256": schedule_fingerprint(train_schedule, train_groups),
                     "validation_schedule_sha256": schedule_fingerprint(val_schedule, val_groups)},
        "config": vars(args), "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in trainable),
        "best_val_loss": best_val, "history": history,
        "checkpoint": {"path": str(output), "sha256": _sha256(output)},
        "training_state": {"path": str(state_path), "sha256": _sha256(state_path)},
    }
    _atomic_json(str(output) + ".report.json", report)
    return report


def _parser():
    arm = get_arm("timesfm25"); parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="output/foundation_tournament/data_cache")
    parser.add_argument("--timesfm-repo", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--stage", required=True,
                        choices=("stage1_reconstruction", "stage3_forecast"))
    parser.add_argument("--warm-checkpoint")
    parser.add_argument("--tickers", default="ES,NQ,RTY,YM,GC,SI,CL,ZB,ZN")
    parser.add_argument("--timeframes", default="1min,3min,5min,15min,30min,60min")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=2048)
    parser.add_argument("--eval-every", type=int, default=512)
    parser.add_argument("--val-batches", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=int, default=8)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=4400)
    parser.add_argument("--validation-seed", type=int, default=5400)
    parser.add_argument("--resume", action="store_true"); parser.add_argument("--stop-after-step", type=int)
    parser.add_argument("--source-revision", default=arm.source_revision)
    parser.add_argument("--model-id", default=arm.model_id)
    parser.add_argument("--model-revision", default=arm.model_revision)
    add_admission_argument(parser)
    return parser


def main():
    args = _parser().parse_args()
    args.tickers = tuple(value for value in args.tickers.split(",") if value)
    args.timeframes = tuple(value for value in args.timeframes.split(",") if value)
    train(args)


if __name__ == "__main__":
    main()
