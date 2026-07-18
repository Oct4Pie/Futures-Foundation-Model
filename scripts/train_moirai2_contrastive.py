#!/usr/bin/env python3
"""Stage-2 Moirai-2 contrastive adaptation with verified encoder gradients."""
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
from futures_foundation.finetune.tournament import (
    MAX_CONTEXT, OOS_START, TRAIN_START, VALIDATION_START,
)
from futures_foundation.finetune.tournament_data import (
    balanced_schedule, gather_parent, load_adaptation_data, schedule_fingerprint,
)
from scripts.train_kronos_contrastive import _nt_xent
from scripts.train_moirai2_tournament import (
    _atomic_json, _atomic_save, _load_model, _sha256, _snapshot, _validate_source,
)

VIEW_LENGTH = MAX_CONTEXT // 2
PAIR_LENGTH = MAX_CONTEXT


def _seed(seed):
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _signature(args):
    value = {key: item for key, item in vars(args).items()
             if key not in {"output", "resume"}}
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=list).encode()).hexdigest()


def _encode(model, past):
    """Pool actual native transformer states for a joint-OHLCV context."""
    import torch
    from uni2ts.common.torch_util import packed_causal_attention_mask

    observed = torch.ones_like(past, dtype=torch.bool)
    is_pad = torch.zeros(past.shape[:2], dtype=torch.bool, device=past.device)
    target, mask, sample_id, time_id, variate_id, prediction_mask = model._convert(
        model.module.patch_size, past_target=past, past_observed_target=observed,
        past_is_pad=is_pad,
    )
    loc, scale = model.module.scaler(
        target, mask * ~prediction_mask.unsqueeze(-1), sample_id, variate_id,
    )
    scaled = (target - loc) / scale
    tokens = model.module.in_proj(torch.cat([scaled, mask.to(torch.float32)], dim=-1))
    representations = model.module.encoder(
        tokens, packed_causal_attention_mask(sample_id, time_id),
        time_id=time_id, var_id=variate_id,
    )
    weight = (~prediction_mask).to(representations.dtype).unsqueeze(-1)
    return (representations * weight).sum(1) / weight.sum(1).clamp_min(1)


def _load_parent(path, arm, model):
    import torch
    path = Path(path).expanduser().resolve()
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    if bundle.get("schema_version") != "ffm_moirai2_staged_bundle_v1":
        raise ValueError("unsupported Moirai-2 parent schema")
    if bundle.get("stage") != "stage1_reconstruction":
        raise ValueError("Moirai-2 Stage 2 requires a Stage-1 parent")
    if bundle.get("arm") != arm.manifest():
        raise ValueError("Moirai-2 parent arm identity mismatch")
    model.load_state_dict(bundle["model_state"], strict=True)
    return {"path": str(path), "sha256": _sha256(path), "stage": bundle["stage"]}


def _archive_sources(output):
    source_dir = Path(str(output) + ".source"); source_dir.mkdir(parents=True, exist_ok=True)
    archived = {}
    for path in (Path(__file__).resolve(),
                 ROOT / "scripts" / "train_moirai2_tournament.py",
                 Path(tournament.__file__).resolve(), Path(tournament_data.__file__).resolve()):
        target = source_dir / path.name; shutil.copyfile(path, target)
        archived[path.name] = {"path": str(target), "sha256": _sha256(target)}
    return archived


def train(args):
    import torch
    arm = get_arm("moirai2_small")
    require_admission_from_args(
        args, arm_key="moirai2_small", track="C",
        route="historical_hidden_state_contrastive", require_training=True,
    )
    if (args.model_id, args.model_revision, args.source_revision) != (
            arm.model_id, arm.model_revision, arm.source_revision):
        raise ValueError("Moirai-2 model/source pins do not match the admitted arm")
    source = _validate_source(args.uni2ts_repo, args.source_revision)
    if args.batch_size < 2 or min(args.max_steps, args.eval_every, args.val_batches) < 1:
        raise ValueError("contrastive budgets require batch>=2 and positive steps")
    output = Path(args.output).resolve(); state_path = Path(str(output) + ".train.pt")
    if output.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite {output} without --resume")
    if args.resume and not state_path.is_file():
        raise FileNotFoundError(f"exact resume state missing: {state_path}")
    signature = _signature(args); _seed(args.seed)
    streams, big, train_starts, val_starts, groups = load_adaptation_data(
        args.data_dir, args.tickers, args.timeframes, parent_length=PAIR_LENGTH,
    )
    train_schedule, train_groups = balanced_schedule(
        train_starts, groups["train_bounds"], args.max_steps * args.batch_size, args.seed,
    )
    val_schedule, val_groups = balanced_schedule(
        val_starts, groups["val_bounds"], args.val_batches * args.batch_size,
        args.validation_seed,
    )
    model = _load_model(args, context_length=VIEW_LENGTH).to(args.device)
    parent = _load_parent(args.warm_checkpoint, arm, model)
    projection = torch.nn.Sequential(
        torch.nn.Linear(int(model.module.d_model), 512), torch.nn.GELU(),
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
        random.setstate(state["python_rng"]); np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        if torch.cuda.is_available() and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])

    def loss_for(schedule, step):
        lo = step * args.batch_size
        raw = gather_parent(big, schedule[lo:lo + args.batch_size], PAIR_LENGTH)
        first = torch.as_tensor(raw[:, :VIEW_LENGTH], device=args.device)
        second = torch.as_tensor(raw[:, VIEW_LENGTH:], device=args.device)
        return _nt_xent(projection(_encode(model, first)),
                        projection(_encode(model, second)), args.temperature)

    def validate():
        model.eval(); projection.eval(); total = 0.0
        with torch.no_grad():
            for step in range(args.val_batches):
                with torch.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=str(args.device).startswith("cuda")):
                    total += float(loss_for(val_schedule, step))
        return total / args.val_batches

    running = 0.0; running_steps = 0; first_grad = None
    model.train(); projection.train()
    for step in range(start, args.max_steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=str(args.device).startswith("cuda")):
            loss = loss_for(train_schedule, step)
        loss.backward()
        grad = torch.sqrt(sum(
            value.grad.detach().float().square().sum()
            for value in model.module.encoder.parameters() if value.grad is not None
        ))
        if first_grad is None:
            first_grad = float(grad)
            if not np.isfinite(first_grad) or first_grad <= 0:
                raise RuntimeError("Moirai-2 Stage 2 did not reach the encoder")
        torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
        optimizer.step(); scheduler.step(); running += float(loss.detach()); running_steps += 1
        completed = step + 1
        if completed % args.eval_every == 0 or completed == args.max_steps:
            val = validate()
            row = {"step": completed, "train_loss": running / running_steps,
                   "val_loss": val, "backbone_grad_norm": float(grad),
                   "learning_rate": scheduler.get_last_lr()[0]}
            history.append(row)
            print(f"[moirai2:stage2] step={completed} train={row['train_loss']:.6f} "
                  f"val={val:.6f} grad={row['backbone_grad_norm']:.4f}", flush=True)
            if val < best_val:
                best_val = val
                _atomic_save(output, {
                    "schema_version": "ffm_moirai2_staged_bundle_v1",
                    "stage": "stage2_contrastive", "model_state": _snapshot(model),
                    "projection_state": _snapshot(projection), "arm": arm.manifest(),
                    "parent": parent,
                    "input": {"context": VIEW_LENGTH, "channels": "joint_ohlcv_representation",
                              "positive_pair": "adjacent_nonoverlapping_128_bar_views"},
                })
            _atomic_save(state_path, {
                "schema_version": "ffm_moirai2_stage2_training_state_v1", "step": completed,
                "config_signature": signature, "best_val": best_val, "history": history,
                "model": _snapshot(model), "projection": _snapshot(projection),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            })
            running = 0.0; running_steps = 0; model.train(); projection.train()

    manifest = next(path for path in (
        Path(args.data_dir) / "TOURNAMENT_CACHE.json", Path(args.data_dir) / "MANIFEST.json",
    ) if path.is_file())
    report = {
        "schema_version": "ffm_moirai2_stage2_train_v1", "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(), "parent": parent,
        "arm": arm.manifest(), "license_gate": "research_only_noncommercial_weights",
        "split": {"train_start": TRAIN_START, "validation_start": VALIDATION_START,
                  "oos_start": OOS_START, "oos_read": False},
        "upstream": {"repo": str(source), "revision": args.source_revision},
        "data": {"streams": [stream["sid"] for stream in streams],
                 "data_manifest_sha256": _sha256(manifest),
                 "train_windows": int(len(train_starts)),
                 "validation_windows": int(len(val_starts)),
                 "train_examples_seen": int(args.max_steps * args.batch_size)},
        "sampling": {"train_schedule_sha256": schedule_fingerprint(train_schedule, train_groups),
                     "validation_schedule_sha256": schedule_fingerprint(val_schedule, val_groups)},
        "config": vars(args), "best_val_loss": best_val,
        "first_backbone_grad_norm": first_grad, "history": history,
        "local_sources": _archive_sources(output),
        "checkpoint": {"path": str(output), "sha256": _sha256(output)},
        "training_state": {"path": str(state_path), "sha256": _sha256(state_path)},
    }
    _atomic_json(str(output) + ".report.json", report)
    return report


def _parser():
    arm = get_arm("moirai2_small")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="output/foundation_tournament/data_cache")
    parser.add_argument("--uni2ts-repo", required=True)
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
