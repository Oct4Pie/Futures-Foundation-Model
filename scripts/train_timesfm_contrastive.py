#!/usr/bin/env python3
"""Stage-2 TimesFM 2.5 contrastive LoRA adaptation using native hidden states."""
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

from futures_foundation.finetune import tournament, tournament_data
from futures_foundation.finetune.foundation_roster import get_arm
from futures_foundation.finetune.native_contracts import (
    add_admission_argument, require_admission_from_args,
)
from futures_foundation.finetune.tournament import MAX_CONTEXT, OOS_START, TRAIN_START, VALIDATION_START
from futures_foundation.finetune.tournament_data import (
    balanced_schedule, gather_parent, load_adaptation_data, schedule_fingerprint,
)
from scripts.train_kronos_contrastive import _nt_xent
from scripts.train_timesfm_tournament import (
    _adapter_state, _atomic_json, _atomic_save, _load_model, _sha256, _validate_source,
)

VIEW_LENGTH = MAX_CONTEXT // 2
PAIR_LENGTH = MAX_CONTEXT


def _signature(args):
    value = {key: item for key, item in vars(args).items() if key not in {"output", "resume"}}
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=list).encode()).hexdigest()


def _seed(seed):
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def _series(view):
    return np.asarray(view, np.float32).transpose(0, 2, 1).reshape(-1, VIEW_LENGTH)


def _encode(model, view, batch_size):
    output = model(past_values=view, forecast_context_len=VIEW_LENGTH).last_hidden_state
    if output is None or output.ndim != 3:
        raise RuntimeError("TimesFM did not return [series,patch,feature] native hidden states")
    return output.mean(dim=1).reshape(batch_size, 5, -1).flatten(1).float()


def _load_stage1(path, arm, model, args):
    import torch
    from peft import set_peft_model_state_dict
    path = Path(path).expanduser().resolve()
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    if bundle.get("schema_version") != "ffm_timesfm_staged_bundle_v1":
        raise ValueError("unsupported TimesFM parent schema")
    if bundle.get("stage") != "stage1_reconstruction":
        raise ValueError("TimesFM Stage 2 requires a Stage-1 parent")
    if bundle.get("arm") != arm.manifest():
        raise ValueError("TimesFM parent arm identity mismatch")
    expected = {"rank": args.lora_rank, "alpha": args.lora_alpha, "dropout": args.lora_dropout}
    if bundle.get("lora") != expected:
        raise ValueError("TimesFM parent LoRA configuration mismatch")
    set_peft_model_state_dict(model, bundle["adapter_state"])
    return {"path": str(path), "sha256": _sha256(path), "stage": bundle["stage"]}


def _snapshot(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _archive(output):
    directory = Path(str(output) + ".source"); directory.mkdir(parents=True, exist_ok=True)
    result = {}
    for path in (Path(__file__).resolve(), ROOT / "scripts" / "train_timesfm_tournament.py",
                 Path(tournament.__file__).resolve(), Path(tournament_data.__file__).resolve()):
        target = directory / path.name; shutil.copyfile(path, target)
        result[path.name] = {"path": str(target), "sha256": _sha256(target)}
    return result


def train(args):
    from futures_foundation.finetune.native_training_routes import block_unadmitted_optimizer
    block_unadmitted_optimizer("scripts.train_timesfm_contrastive.train")
    import torch
    from peft import set_peft_model_state_dict
    arm = get_arm("timesfm25")
    require_admission_from_args(
        args, arm_key="timesfm25", track="C",
        route="historical_hidden_state_contrastive", require_training=True,
    )
    if (args.model_id, args.model_revision, args.source_revision) != (
            arm.model_id, arm.model_revision, arm.source_revision):
        raise ValueError("TimesFM model/source pins do not match the admitted arm")
    source = _validate_source(args.timesfm_repo, args.source_revision)
    if args.batch_size < 2 or min(args.max_steps, args.eval_every, args.val_batches) < 1:
        raise ValueError("contrastive budgets require batch>=2 and positive steps")
    output = Path(args.output).resolve(); state_path = Path(str(output) + ".train.pt")
    if output.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite {output} without --resume")
    if args.resume and not state_path.is_file():
        raise FileNotFoundError(f"exact resume state missing: {state_path}")
    _seed(args.seed); signature = _signature(args)
    streams, big, train_starts, val_starts, groups = load_adaptation_data(
        args.data_dir, args.tickers, args.timeframes, parent_length=PAIR_LENGTH,
    )
    train_schedule, train_groups = balanced_schedule(
        train_starts, groups["train_bounds"], args.max_steps * args.batch_size, args.seed,
    )
    val_schedule, val_groups = balanced_schedule(
        val_starts, groups["val_bounds"], args.val_batches * args.batch_size, args.validation_seed,
    )
    model = _load_model(args); parent = _load_stage1(args.warm_checkpoint, arm, model, args)
    hidden = int(model.config.hidden_size) * 5
    projection = torch.nn.Sequential(torch.nn.Linear(hidden, 512), torch.nn.GELU(),
                                     torch.nn.Linear(512, args.projection_dim)).to(args.device)
    trainable_model = [p for p in model.parameters() if p.requires_grad]
    parameters = trainable_model + list(projection.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate,
                                  weight_decay=args.weight_decay, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_steps)
    start, best_val, history = 0, float("inf"), []
    if args.resume:
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state.get("config_signature") != signature:
            raise ValueError("exact resume refused: configuration changed")
        set_peft_model_state_dict(model, state["adapter"])
        projection.load_state_dict(state["projection"]); optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start, best_val, history = state["step"], state["best_val"], state["history"]
        random.setstate(state["python_rng"]); np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        if torch.cuda.is_available() and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])

    def loss_for(schedule, step):
        lo = step * args.batch_size
        raw = gather_parent(big, schedule[lo:lo + args.batch_size], PAIR_LENGTH)
        first = torch.as_tensor(_series(raw[:, :VIEW_LENGTH]), device=args.device)
        second = torch.as_tensor(_series(raw[:, VIEW_LENGTH:]), device=args.device)
        return _nt_xent(projection(_encode(model, first, args.batch_size)),
                        projection(_encode(model, second, args.batch_size)), args.temperature)

    def validate():
        model.eval(); projection.eval(); total = 0.0
        with torch.no_grad():
            for step in range(args.val_batches): total += float(loss_for(val_schedule, step))
        return total / args.val_batches

    running = 0.0; count = 0; first_grad = None; model.train(); projection.train()
    for step in range(start, args.max_steps):
        optimizer.zero_grad(set_to_none=True); loss = loss_for(train_schedule, step); loss.backward()
        grad = torch.sqrt(sum(p.grad.detach().float().square().sum()
                              for p in trainable_model if p.grad is not None))
        if first_grad is None:
            first_grad = float(grad)
            if not np.isfinite(first_grad) or first_grad <= 0:
                raise RuntimeError("TimesFM Stage 2 did not reach LoRA backbone adapters")
        torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
        optimizer.step(); scheduler.step(); running += float(loss.detach()); count += 1
        completed = step + 1
        if completed % args.eval_every == 0 or completed == args.max_steps:
            val = validate()
            row = {"step": completed, "train_loss": running / count, "val_loss": val,
                   "adapter_grad_norm": float(grad), "learning_rate": scheduler.get_last_lr()[0]}
            history.append(row)
            print(f"[timesfm25:stage2] step={completed} train={row['train_loss']:.6f} "
                  f"val={val:.6f} grad={row['adapter_grad_norm']:.4f}", flush=True)
            if val < best_val:
                best_val = val
                _atomic_save(output, {"schema_version": "ffm_timesfm_staged_bundle_v1",
                    "stage": "stage2_contrastive", "adapter_state": _adapter_state(model),
                    "projection_state": _snapshot(projection), "arm": arm.manifest(),
                    "lora": {"rank": args.lora_rank, "alpha": args.lora_alpha,
                             "dropout": args.lora_dropout}, "parent": parent,
                    "input": {"context": VIEW_LENGTH, "channels": "channel_independent_ohlcv",
                              "positive_pair": "adjacent_nonoverlapping_128_bar_views"}})
            _atomic_save(state_path, {"schema_version": "ffm_timesfm_stage2_training_state_v1",
                "config_signature": signature, "step": completed, "best_val": best_val,
                "history": history, "adapter": _adapter_state(model),
                "projection": _snapshot(projection), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "python_rng": random.getstate(),
                "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None})
            running = 0.0; count = 0; model.train(); projection.train()

    manifest = next(p for p in (Path(args.data_dir) / "TOURNAMENT_CACHE.json",
                                Path(args.data_dir) / "MANIFEST.json") if p.is_file())
    report = {"schema_version": "ffm_timesfm_stage2_train_v1", "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(), "arm": arm.manifest(),
        "parent": parent, "split": {"train_start": TRAIN_START,
        "validation_start": VALIDATION_START, "oos_start": OOS_START, "oos_read": False},
        "upstream": {"repo": str(source), "revision": args.source_revision},
        "data": {"streams": [s["sid"] for s in streams], "data_manifest_sha256": _sha256(manifest),
        "train_windows": int(len(train_starts)), "validation_windows": int(len(val_starts)),
        "anchors_seen": int(args.max_steps * args.batch_size)},
        "sampling": {"train_schedule_sha256": schedule_fingerprint(train_schedule, train_groups),
        "validation_schedule_sha256": schedule_fingerprint(val_schedule, val_groups)},
        "config": vars(args), "best_val_loss": best_val, "first_adapter_grad_norm": first_grad,
        "history": history, "local_sources": _archive(output),
        "checkpoint": {"path": str(output), "sha256": _sha256(output)},
        "training_state": {"path": str(state_path), "sha256": _sha256(state_path)}}
    _atomic_json(str(output) + ".report.json", report); return report


def _parser():
    arm = get_arm("timesfm25"); parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="output/foundation_tournament/data_cache")
    parser.add_argument("--timesfm-repo", required=True); parser.add_argument("--warm-checkpoint", required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--tickers", default="ES,NQ,RTY,YM,GC,SI,CL,ZB,ZN")
    parser.add_argument("--timeframes", default="1min,3min,5min,15min,30min,60min")
    parser.add_argument("--max-steps", type=int, default=4096); parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=1024); parser.add_argument("--val-batches", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-5); parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.2); parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--grad-clip", type=float, default=1.0); parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16); parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--device", default="cuda:0"); parser.add_argument("--seed", type=int, default=8400)
    parser.add_argument("--validation-seed", type=int, default=5400); parser.add_argument("--resume", action="store_true")
    parser.add_argument("--source-revision", default=arm.source_revision); parser.add_argument("--model-id", default=arm.model_id)
    parser.add_argument("--model-revision", default=arm.model_revision)
    add_admission_argument(parser)
    return parser


def main():
    args = _parser().parse_args(); args.tickers = tuple(x for x in args.tickers.split(",") if x)
    args.timeframes = tuple(x for x in args.timeframes.split(",") if x); train(args)


if __name__ == "__main__": main()
