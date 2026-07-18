#!/usr/bin/env python3
"""Stage-2 contrastive adaptation of a Stage-1 Kronos bundle on the locked futures corpus."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from futures_foundation.finetune.foundation_roster import get_arm
from futures_foundation.finetune.native_contracts import (
    add_admission_argument, require_admission_from_args, validate_identity,
)
from futures_foundation.finetune.tournament import (
    OOS_START, PARENT_LENGTH, TRAIN_START, VALIDATION_START,
)
from futures_foundation.finetune.tournament_data import (
    balanced_schedule, load_adaptation_data, schedule_fingerprint,
)
from scripts.train_kronos_tournament import (
    SOURCE_REVISION, TOKENIZER_ID, TOKENIZER_REVISION, _load_models, _load_warm_bundle,
    _normalized_batch, _sha256, _validate_source,
)


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


def _seed(seed):
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _nt_xent(z1, z2, temperature):
    """Symmetric in-batch InfoNCE with paired views on the off-diagonal blocks."""
    import torch
    import torch.nn.functional as F
    if z1.ndim != 2 or z1.shape != z2.shape or len(z1) < 2:
        raise ValueError("NT-Xent requires two equal [batch,dim] tensors and batch >= 2")
    z = F.normalize(torch.cat((z1, z2), dim=0), dim=1)
    logits = z @ z.T / float(temperature)
    logits.fill_diagonal_(-torch.inf)
    batch = len(z1)
    labels = (torch.arange(2 * batch, device=z.device) + batch) % (2 * batch)
    return F.cross_entropy(logits, labels)


def _encode_half(predictor, tokenizer, values, stamps, projector):
    import torch
    with torch.no_grad():
        first, second = tokenizer.encode(values, half=True)
    _, hidden = predictor.decode_s1(first, second, stamps)
    return projector(hidden.mean(dim=1))


def train(args):
    import torch
    arm = validate_identity(
        args.arm,
        model_id=args.model_id,
        model_revision=args.model_revision,
        source_revision=args.source_revision,
        tokenizer_id=args.tokenizer_id,
        tokenizer_revision=args.tokenizer_revision,
    )
    if arm.family != "kronos" or not arm.supported_training:
        raise ValueError(f"{args.arm} does not declare a Kronos adaptation route")
    admission = require_admission_from_args(
        args, arm_key=args.arm, track="C", route="adjacent_half_contrastive",
        require_training=True,
    )
    repo, source = _validate_source(args.kronos_repo, args.source_revision)
    if args.max_steps < 1 or args.batch_size < 2 or args.val_batches < 1:
        raise ValueError("positive steps/validation and batch >= 2 are required")
    if not 0 < args.temperature <= 1:
        raise ValueError("temperature must lie in (0,1]")
    output = Path(args.output).resolve()
    state_path = Path(str(output) + ".train.pt")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")

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
    tokenizer, predictor = _load_models(repo, args)
    parent = _load_warm_bundle(args.warm_checkpoint, tokenizer, predictor, args)
    if parent.get("stage") != "stage1_reconstruction":
        raise ValueError("Kronos Stage 2 requires a Stage-1 reconstruction parent")
    tokenizer, predictor = tokenizer.to(args.device), predictor.to(args.device)
    tokenizer.eval()
    for parameter in tokenizer.parameters():
        parameter.requires_grad_(False)
    projector = torch.nn.Sequential(
        torch.nn.Linear(predictor.d_model, predictor.d_model), torch.nn.GELU(),
        torch.nn.Linear(predictor.d_model, args.projection_dim),
    ).to(args.device)
    optimizer = torch.optim.AdamW(
        list(predictor.parameters()) + list(projector.parameters()),
        lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.95),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_steps)

    def batch(schedule, schedule_groups, step):
        lo = step * args.batch_size
        return _normalized_batch(
            big, schedule[lo:lo + args.batch_size], schedule_groups[lo:lo + args.batch_size],
            streams, groups["row_bounds"], args.device, args.clip,
        )

    def loss_for(values, stamps):
        # Non-overlapping adjacent halves: no near-identical positive-window shortcut.
        first = _encode_half(
            predictor, tokenizer, values[:, :128], stamps[:, :128], projector,
        )
        second = _encode_half(
            predictor, tokenizer, values[:, 128:256], stamps[:, 128:256], projector,
        )
        return _nt_xent(first, second, args.temperature)

    @torch.no_grad()
    def validate():
        predictor.eval(); projector.eval()
        total = 0.0
        for step in range(args.val_batches):
            values, stamps = batch(val_schedule, val_groups, step)
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=str(args.device).startswith("cuda")):
                total += float(loss_for(values, stamps))
        return total / args.val_batches

    best_val = float("inf"); best_predictor = None; best_projector = None
    history = []; running = 0.0; running_steps = 0; first_grad_norm = None
    predictor.train(); projector.train()
    for step in range(args.max_steps):
        values, stamps = batch(train_schedule, train_groups, step)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=str(args.device).startswith("cuda")):
            loss = loss_for(values, stamps)
        loss.backward()
        grad_sq = sum(
            float(parameter.grad.detach().float().square().sum())
            for parameter in predictor.parameters() if parameter.grad is not None
        )
        grad_norm = grad_sq ** 0.5
        if first_grad_norm is None:
            first_grad_norm = grad_norm
            if not np.isfinite(grad_norm) or grad_norm <= 0:
                raise RuntimeError("Stage-2 loss produced no Kronos backbone gradient")
        torch.nn.utils.clip_grad_norm_(
            list(predictor.parameters()) + list(projector.parameters()), args.grad_clip,
        )
        optimizer.step(); scheduler.step()
        running += float(loss.detach()); running_steps += 1
        completed = step + 1
        if completed % args.eval_every == 0 or completed == args.max_steps:
            val = validate()
            row = {"step": completed, "train_loss": running / running_steps,
                   "val_loss": val, "learning_rate": scheduler.get_last_lr()[0],
                   "backbone_grad_norm": grad_norm}
            history.append(row)
            print(f"[kronos-contrastive] step={completed} train={row['train_loss']:.6f} "
                  f"val={val:.6f} grad={grad_norm:.4f}", flush=True)
            running = 0.0; running_steps = 0
            if val < best_val:
                best_val = val
                best_predictor = {
                    key: value.detach().cpu() for key, value in predictor.state_dict().items()
                }
                best_projector = {
                    key: value.detach().cpu() for key, value in projector.state_dict().items()
                }
                parent_bundle = torch.load(
                    Path(args.warm_checkpoint).resolve(), map_location="cpu", weights_only=False,
                )
                _atomic_save(output, {
                    "schema_version": "ffm_kronos_tournament_bundle_v1",
                    "stage": "stage2_contrastive", "parent": parent,
                    "tokenizer_state": parent_bundle["tokenizer_state"],
                    "predictor_state": best_predictor,
                    "projector_state": best_projector,
                    "tokenizer": parent_bundle["tokenizer"],
                    "predictor": parent_bundle["predictor"],
                    "input": parent_bundle["input"],
                })
            _atomic_save(state_path, {
                "schema_version": "ffm_kronos_contrastive_state_v1", "step": completed,
                "predictor": predictor.state_dict(), "projector": projector.state_dict(),
                "best_predictor": best_predictor, "best_projector": best_projector,
                "best_val": best_val, "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "history": history,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            })
            predictor.train(); projector.train()

    report = {
        "schema_version": "ffm_kronos_contrastive_train_v1", "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(), "arm": arm.manifest(),
        "admission": admission,
        "split": {"train_start": TRAIN_START, "validation_start": VALIDATION_START,
                  "oos_start": OOS_START, "oos_read": False},
        "parent": parent, "first_backbone_grad_norm": first_grad_norm,
        "data": {"streams": [stream["sid"] for stream in streams],
                 "train_windows": int(len(train_starts)),
                 "validation_windows": int(len(val_starts)),
                 "examples_seen": int(args.max_steps * args.batch_size)},
        "sampling": {"train_schedule_sha256": schedule_fingerprint(
                          train_schedule, train_groups),
                     "validation_schedule_sha256": schedule_fingerprint(
                          val_schedule, val_groups)},
        "best_val_loss": best_val, "history": history, "config": vars(args),
        "upstream": {"repo": str(repo), "revision": args.source_revision,
                     "source_sha256": _sha256(source)},
        "checkpoint": {"path": str(output), "sha256": _sha256(output)},
        "training_state": {"path": str(state_path), "sha256": _sha256(state_path)},
    }
    _atomic_json(str(output) + ".report.json", report)
    return report


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kronos-repo", required=True)
    parser.add_argument("--warm-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-dir", default="data/ssl_corpus_v2_6tf")
    parser.add_argument("--arm", choices=("kronos_mini", "kronos_small"), default="kronos_small")
    parser.add_argument("--tickers", default="ES,NQ,RTY,YM,GC,SI,CL,ZB,ZN")
    parser.add_argument("--timeframes", default="1min,3min,5min,15min,30min,60min")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=2048)
    parser.add_argument("--eval-every", type=int, default=512)
    parser.add_argument("--val-batches", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--grad-clip", type=float, default=3.0)
    parser.add_argument("--clip", type=float, default=3.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validation-seed", type=int, default=5400)
    parser.add_argument("--source-revision", default=SOURCE_REVISION)
    parser.add_argument("--tokenizer-id")
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--model-id")
    parser.add_argument("--model-revision")
    add_admission_argument(parser)
    return parser


def main():
    args = _parser().parse_args()
    arm = get_arm(args.arm)
    args.model_id = args.model_id or arm.model_id
    args.model_revision = args.model_revision or arm.model_revision
    args.tokenizer_id = args.tokenizer_id or arm.tokenizer_id
    args.tokenizer_revision = args.tokenizer_revision or arm.tokenizer_revision
    args.tickers = tuple(value for value in args.tickers.split(",") if value)
    args.timeframes = tuple(value for value in args.timeframes.split(",") if value)
    train(args)


if __name__ == "__main__":
    main()
