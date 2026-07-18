#!/usr/bin/env python3
"""Research-only, leak-safe staged Moirai-2 adaptation (CC-BY-NC weights)."""
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


def _snapshot(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _validate_source(repo, expected):
    repo = Path(repo).expanduser().resolve()
    if not (repo / "src" / "uni2ts" / "model" / "moirai2" / "module.py").is_file():
        raise FileNotFoundError(f"official Uni2TS source missing: {repo}")
    actual = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if actual != expected:
        raise ValueError(f"Uni2TS revision mismatch: expected {expected}, got {actual}")
    return repo


def _load_model(args, context_length=MAX_CONTEXT):
    from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
    module = Moirai2Module.from_pretrained(args.model_id, revision=args.model_revision)
    return Moirai2Forecast(
        prediction_length=FORECAST_HORIZON, target_dim=5,
        feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0,
        context_length=context_length, module=module,
    )


def _load_parent(path, arm, required_stage):
    import torch
    path = Path(path).expanduser().resolve()
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    if bundle.get("schema_version") != "ffm_moirai2_staged_bundle_v1":
        raise ValueError(f"unsupported Moirai-2 parent schema in {path}")
    if bundle.get("stage") != required_stage:
        raise ValueError(
            f"Moirai-2 parent must be {required_stage}, got {bundle.get('stage')!r}"
        )
    if bundle.get("arm") != arm.manifest():
        raise ValueError("Moirai-2 parent arm identity mismatch")
    return path, bundle, {"path": str(path), "sha256": _sha256(path),
                          "stage": required_stage}


def _native_loss(model, past, future):
    """Moirai-2 scaled pinball loss at its exact first forecast-token seam."""
    import torch
    patch = int(model.module.patch_size)
    observed_past = torch.ones_like(past, dtype=torch.bool)
    observed_future = torch.ones_like(future, dtype=torch.bool)
    is_pad_past = torch.zeros(past.shape[:2], dtype=torch.bool, device=past.device)
    is_pad_future = torch.zeros(future.shape[:2], dtype=torch.bool, device=future.device)
    target, observed, sample_id, time_id, variate_id, prediction_mask = model._convert(
        patch, past_target=past, past_observed_target=observed_past,
        past_is_pad=is_pad_past, future_target=future,
        future_observed_target=observed_future, future_is_pad=is_pad_future,
    )
    pred, scaled_target = model.module(
        target, observed, sample_id, time_id, variate_id, prediction_mask,
        training_mode=True,
    )
    context_tokens = model.context_token_length(patch)
    future_tokens = model.prediction_token_length(patch)
    if future_tokens != 1:
        raise ValueError("locked 16-bar Moirai horizon must occupy exactly one patch")
    pred_index = torch.arange(
        context_tokens - 1, model.hparams.target_dim * context_tokens,
        context_tokens, device=past.device,
    )
    target_index = torch.arange(
        model.hparams.target_dim * context_tokens,
        model.hparams.target_dim * (context_tokens + future_tokens),
        device=past.device,
    )
    pred = pred[:, pred_index].reshape(
        len(past), model.hparams.target_dim, model.module.num_predict_token,
        model.module.num_quantiles, patch,
    )[:, :, 0]
    actual = scaled_target[:, target_index].unsqueeze(2)
    error = actual - pred
    level = torch.as_tensor(
        model.module.quantile_levels, dtype=pred.dtype, device=pred.device,
    ).view(1, 1, -1, 1)
    return torch.maximum(level * error, (level - 1.0) * error).mean()


def _signature(args):
    value = {key: value for key, value in vars(args).items()
             if key not in {"output", "resume", "stop_after_step"}}
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, default=list, separators=(",", ":"),
    ).encode()).hexdigest()


def train(args):
    import torch
    arm = get_arm("moirai2_small")
    require_admission_from_args(
        args, arm_key="moirai2_small", track="F",
        route="historical_universal_stage_chain", require_training=True,
    )
    if (args.model_id, args.model_revision, args.source_revision) != (
            arm.model_id, arm.model_revision, arm.source_revision):
        raise ValueError("Moirai-2 model/source pins do not match the admitted arm")
    source = _validate_source(args.uni2ts_repo, args.source_revision)
    if min(args.max_steps, args.batch_size, args.eval_every, args.val_batches) < 1:
        raise ValueError("training and validation budgets must be positive")
    if args.stage == "stage1_reconstruction" and args.warm_checkpoint:
        raise ValueError("Stage 1 must start from the admitted pretrained Moirai-2 checkpoint")
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
    context_length = MAX_CONTEXT - FORECAST_HORIZON \
        if args.stage == "stage1_reconstruction" else MAX_CONTEXT
    model = _load_model(args, context_length=context_length).to(args.device)
    parent = None
    if args.stage == "stage3_forecast":
        _, parent_bundle, parent = _load_parent(
            args.warm_checkpoint, arm, "stage2_contrastive",
        )
        model.load_state_dict(parent_bundle["model_state"], strict=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_steps)
    signature = _signature(args)
    start, history, best_val, best_model = 0, [], float("inf"), _snapshot(model)
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

    def batch(schedule, step):
        lo = step * args.batch_size
        parent = gather_parent(big, schedule[lo:lo + args.batch_size], PARENT_LENGTH)
        if args.stage == "stage1_reconstruction":
            visible = MAX_CONTEXT - FORECAST_HORIZON
            past, future = parent[:, :visible], parent[:, visible:MAX_CONTEXT]
        else:
            past, future = parent[:, :MAX_CONTEXT], parent[:, MAX_CONTEXT:]
        return (torch.as_tensor(past, device=args.device),
                torch.as_tensor(future, device=args.device))

    def validate():
        model.eval(); total = 0.0
        with torch.no_grad():
            for step in range(args.val_batches):
                past, future = batch(val_schedule, step)
                with torch.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=str(args.device).startswith("cuda")):
                    total += float(_native_loss(model, past, future))
        return total / args.val_batches

    end_step = min(args.max_steps, args.stop_after_step or args.max_steps)
    running, running_steps = 0.0, 0; model.train()
    for step in range(start, end_step):
        past, future = batch(train_schedule, step); optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=str(args.device).startswith("cuda")):
            loss = _native_loss(model, past, future)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step(); scheduler.step()
        running += float(loss.detach()); running_steps += 1; completed = step + 1
        if completed % args.eval_every == 0 or completed == end_step:
            val = validate()
            row = {"step": completed, "train_loss": running / running_steps,
                   "val_loss": val, "learning_rate": scheduler.get_last_lr()[0]}
            history.append(row); running, running_steps = 0.0, 0
            print(f"[moirai2:{args.stage}] step={completed} "
                  f"train={row['train_loss']:.6f} val={val:.6f}", flush=True)
            if val < best_val:
                best_val, best_model = val, _snapshot(model)
                _atomic_save(output, {
                    "schema_version": "ffm_moirai2_staged_bundle_v1",
                    "stage": args.stage, "model_state": best_model,
                    "arm": arm.manifest(), "parent": parent,
                    "input": {"context": context_length, "horizon": FORECAST_HORIZON,
                              "normalization": "native_packed_std_scaler",
                              "ohlcv_mode": "joint_ohlcv"},
                })
            _atomic_save(state_path, {
                "schema_version": "ffm_moirai2_staged_training_state_v1",
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
        "schema_version": "ffm_moirai2_staged_train_v1",
        "status": "complete" if end_step == args.max_steps else "partial",
        "created_utc": datetime.now(timezone.utc).isoformat(), "arm": arm.manifest(),
        "stage": args.stage, "parent": parent,
        "license_gate": "research_only_noncommercial_weights",
        "split": {"train_start": TRAIN_START, "validation_start": VALIDATION_START,
                  "oos_start": OOS_START, "oos_read": False},
        "upstream": {"repo": str(source), "revision": args.source_revision},
        "local_sources": archived,
        "data": {"streams": [stream["sid"] for stream in streams],
                 "data_manifest_sha256": _sha256(manifest),
                 "train_windows": int(len(train_starts)),
                 "validation_windows": int(len(val_starts)),
                 "anchors_seen": int(args.max_steps * args.batch_size)},
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
    arm = get_arm("moirai2_small"); parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="output/foundation_tournament/data_cache")
    parser.add_argument("--uni2ts-repo", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--stage", required=True,
                        choices=("stage1_reconstruction", "stage3_forecast"))
    parser.add_argument("--warm-checkpoint")
    parser.add_argument("--tickers", default="ES,NQ,RTY,YM,GC,SI,CL,ZB,ZN")
    parser.add_argument("--timeframes", default="1min,3min,5min,15min,30min,60min")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=512)
    parser.add_argument("--eval-every", type=int, default=512)
    parser.add_argument("--val-batches", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0"); parser.add_argument("--seed", type=int, default=4400)
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
