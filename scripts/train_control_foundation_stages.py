#!/usr/bin/env python3
"""Train Toto-2 or Sundial through the locked three-stage futures curriculum.

Toto uses its full native backbone. Sundial uses LoRA because upstream has not released
fine-tuning code and full 128M-state adaptation is not comparable to the other adapter arms.
Task heads/projectors are training-only; deployment checkpoints contain backbone adaptation.
"""
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
    FORECAST_HORIZON, MAX_CONTEXT, OOS_START, TRAIN_START, VALIDATION_START,
)
from futures_foundation.finetune.tournament_data import (
    balanced_schedule, gather_parent, load_adaptation_data, schedule_fingerprint,
)
from scripts.train_kronos_contrastive import _nt_xent

STAGES = ("stage1_reconstruction", "stage2_contrastive", "stage3_forecast")
PARENT_LENGTH = MAX_CONTEXT + FORECAST_HORIZON
VIEW_LENGTH = MAX_CONTEXT // 2


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_save(path, value):
    import torch
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    torch.save(value, temporary); os.replace(temporary, path)


def _atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=float) + "\n")
    os.replace(temporary, path)


def _seed(value):
    import torch
    random.seed(value); np.random.seed(value); torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def _signature(args):
    config = {key: value for key, value in vars(args).items()
              if key not in {"output", "resume", "stop_after_step"}}
    return hashlib.sha256(json.dumps(
        config, sort_keys=True, default=list, separators=(",", ":"),
    ).encode()).hexdigest()


def _normalize(raw, fit_length):
    """Causal per-channel normalization fitted only on the declared visible context."""
    raw = np.asarray(raw, np.float32)
    visible = raw[:, :fit_length]
    mean = visible.mean(axis=1, keepdims=True)
    std = np.maximum(visible.std(axis=1, keepdims=True), 1e-5)
    normalized = (raw - mean) / std
    # Thin/zero-volume windows can otherwise create extreme values that destabilize the
    # released Sundial float32 attention path. This is the same bounded-z-score guard used
    # by the project's Mantis trainers and is fitted without future information.
    return np.nan_to_num(np.clip(normalized, -10.0, 10.0), nan=0.0, posinf=10.0, neginf=-10.0)


def _validate_source(args, arm):
    repo = Path(args.upstream_repo).expanduser().resolve()
    marker = repo / ("toto2/toto2/model.py" if args.family == "toto2_22m" else "README.md")
    if not marker.is_file():
        raise FileNotFoundError(f"pinned upstream source missing: {marker}")
    revision = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True,
    ).strip()
    if revision != arm.source_revision:
        raise ValueError(f"source revision mismatch: expected {arm.source_revision}, got {revision}")
    return repo


def _load_model(args, arm):
    if args.family == "toto2_22m":
        from toto2 import Toto2Model
        model = Toto2Model.from_pretrained(
            arm.model_id, revision=arm.model_revision, map_location="cpu",
        )
        return model.to(args.device), {"mode": "full_native"}

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM
    base = AutoModelForCausalLM.from_pretrained(
        arm.model_id, revision=arm.model_revision, trust_remote_code=True,
        dtype="auto",
    )
    config = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules="all-linear", bias="none",
    )
    return get_peft_model(base, config).to(args.device), {
        "mode": "lora_native_hidden_states", "rank": args.lora_rank,
        "alpha": args.lora_alpha, "dropout": args.lora_dropout,
    }


def _backbone_state(model, family):
    if family == "toto2_22m":
        return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    from peft import get_peft_model_state_dict
    return {key: value.detach().cpu().clone()
            for key, value in get_peft_model_state_dict(model).items()}


def _restore_backbone(model, family, state):
    if family == "toto2_22m":
        model.load_state_dict(state, strict=True)
    else:
        from peft import set_peft_model_state_dict
        set_peft_model_state_dict(model, state)


def _toto_encode(model, values, observed=None):
    """Return one joint multivariate native state per observation."""
    import torch
    from einops import rearrange, reduce, repeat
    target = values.transpose(1, 2)
    target_mask = (torch.ones_like(target, dtype=torch.bool) if observed is None
                   else observed.transpose(1, 2))
    scaled, _, _ = model.scaler(target, target_mask)
    scaled = scaled.asinh()
    patch = model.config.patch_size
    x = model.patch_proj(torch.cat([
        rearrange(scaled, "... (seq patch) -> ... seq patch", patch=patch),
        rearrange((~target_mask).to(target.dtype),
                  "... (seq patch) -> ... seq patch", patch=patch),
    ], dim=-1))
    series_ids = torch.arange(target.shape[-2], device=target.device).expand(target.shape[0], -1)
    group_ids = repeat(series_ids, "b v -> b v seq", seq=x.shape[-2]).clone()
    empty = reduce(target_mask, "b v (seq patch) -> b v seq", "sum", patch=patch) == 0
    group_ids[empty] = -1
    states = model.transformer(x, group_ids=group_ids)
    return states.mean(dim=-2).mean(dim=-2).float()


def _sundial_encode(model, values, _observed=None):
    """Encode five channels independently and concatenate their native patch states."""
    batch, length, channels = values.shape
    flat = values.transpose(1, 2).reshape(batch * channels, length)
    states = model.base_model.model.model(
        input_ids=flat, use_cache=False, return_dict=True,
    ).last_hidden_state.mean(dim=1)
    return states.reshape(batch, channels, -1).flatten(1).float()


def encode(model, family, values, observed=None):
    return (_toto_encode(model, values, observed) if family == "toto2_22m"
            else _sundial_encode(model, values, observed))


def _load_parent(path, model, args, arm, required_stage, adaptation):
    import torch
    path = Path(path).expanduser().resolve()
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    if bundle.get("schema_version") != "ffm_control_staged_bundle_v1":
        raise ValueError("unsupported staged control checkpoint")
    if bundle.get("stage") != required_stage:
        raise ValueError(f"parent must be {required_stage}, got {bundle.get('stage')!r}")
    if bundle.get("arm") != arm.manifest() or bundle.get("adaptation") != adaptation:
        raise ValueError("parent identity/adaptation mismatch")
    _restore_backbone(model, args.family, bundle["backbone_state"])
    return {"path": str(path), "sha256": _sha256(path), "stage": required_stage}


def _archive_sources(output, args, source):
    directory = Path(str(output) + ".source"); directory.mkdir(parents=True, exist_ok=True)
    inputs = [Path(__file__).resolve(), Path(tournament.__file__).resolve(),
              Path(tournament_data.__file__).resolve()]
    if args.family == "toto2_22m":
        inputs.extend([source / "toto2/toto2/model.py", source / "toto2/toto2/configuration.py"])
    else:
        snapshot = next((Path.home() / ".cache/huggingface/hub/models--thuml--sundial-base-128m/snapshots").glob("*"))
        inputs.extend(snapshot / name for name in (
            "modeling_sundial.py", "configuration_sundial.py", "flow_loss.py",
            "ts_generation_mixin.py",
        ))
    result = {}
    for index, path in enumerate(inputs):
        target = directory / f"{index:02d}_{path.name}"
        shutil.copyfile(path, target)
        result[target.name] = {"path": str(target), "sha256": _sha256(target)}
    return result


def train(args):
    import torch
    arm = get_arm(args.family)
    if (args.model_id, args.model_revision, args.source_revision) != (
            arm.model_id, arm.model_revision, arm.source_revision):
        raise ValueError("model/source pins do not match admitted arm")
    if min(args.max_steps, args.batch_size, args.eval_every, args.val_batches) < 1:
        raise ValueError("training budgets must be positive")
    if args.stage == STAGES[0] and args.warm_checkpoint:
        raise ValueError("Stage 1 must start from the admitted pretrained checkpoint")
    if args.stage != STAGES[0] and not args.warm_checkpoint:
        raise ValueError(f"{args.stage} requires its staged predecessor")
    output = Path(args.output).resolve(); state_path = Path(str(output) + ".train.pt")
    if output.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite {output} without --resume")
    if args.resume and not state_path.is_file():
        raise FileNotFoundError(f"exact resume state missing: {state_path}")

    source = _validate_source(args, arm); _seed(args.seed)
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
    model, adaptation = _load_model(args, arm)
    parent = None
    if args.stage == STAGES[1]:
        parent = _load_parent(args.warm_checkpoint, model, args, arm, STAGES[0], adaptation)
    elif args.stage == STAGES[2]:
        parent = _load_parent(args.warm_checkpoint, model, args, arm, STAGES[1], adaptation)

    representation_dim = (int(model.config.d_model) if args.family == "toto2_22m"
                          else int(model.config.hidden_size) * 5)
    if args.stage == STAGES[1]:
        head = torch.nn.Sequential(
            torch.nn.Linear(representation_dim, 512), torch.nn.GELU(),
            torch.nn.Linear(512, args.projection_dim),
        ).to(args.device)
    else:
        head = torch.nn.Linear(representation_dim, 5 * FORECAST_HORIZON).to(args.device)
    backbone_parameters = [value for value in model.parameters() if value.requires_grad]
    parameters = backbone_parameters + list(head.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate,
                                  weight_decay=args.weight_decay, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_steps)
    signature = _signature(args); start, history, best_val = 0, [], float("inf")
    best_backbone = _backbone_state(model, args.family)
    if args.resume:
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state.get("config_signature") != signature:
            raise ValueError("exact resume refused: configuration changed")
        _restore_backbone(model, args.family, state["backbone_state"])
        head.load_state_dict(state["head_state"]); optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start, history, best_val = state["step"], state["history"], state["best_val"]
        best_backbone = state["best_backbone"]
        random.setstate(state["python_rng"]); np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        if torch.cuda.is_available() and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])

    def raw_batch(schedule, step):
        lo = step * args.batch_size
        return gather_parent(big, schedule[lo:lo + args.batch_size], PARENT_LENGTH)

    def loss_for(schedule, step):
        raw = raw_batch(schedule, step)
        if args.stage == STAGES[0]:
            values = _normalize(raw, MAX_CONTEXT - FORECAST_HORIZON)
            context = values[:, :MAX_CONTEXT].copy()
            target = values[:, MAX_CONTEXT - FORECAST_HORIZON:MAX_CONTEXT]
            context[:, MAX_CONTEXT - FORECAST_HORIZON:] = 0.0
            observed = np.ones(context.shape, dtype=bool)
            observed[:, MAX_CONTEXT - FORECAST_HORIZON:] = False
            representation = encode(
                model, args.family, torch.as_tensor(context, device=args.device),
                torch.as_tensor(observed, device=args.device),
            )
            prediction = head(representation).reshape(-1, FORECAST_HORIZON, 5)
            return torch.nn.functional.mse_loss(
                prediction, torch.as_tensor(target, device=args.device),
            )
        if args.stage == STAGES[2]:
            values = _normalize(raw, MAX_CONTEXT)
            context = torch.as_tensor(values[:, :MAX_CONTEXT], device=args.device)
            target = torch.as_tensor(
                values[:, MAX_CONTEXT:MAX_CONTEXT + FORECAST_HORIZON], device=args.device,
            )
            prediction = head(encode(model, args.family, context)).reshape(
                -1, FORECAST_HORIZON, 5,
            )
            return torch.nn.functional.mse_loss(prediction, target)

        first = _normalize(raw[:, :VIEW_LENGTH], VIEW_LENGTH)
        second = _normalize(raw[:, VIEW_LENGTH:MAX_CONTEXT], VIEW_LENGTH)
        first = torch.as_tensor(first, device=args.device)
        second = torch.as_tensor(second, device=args.device)
        return _nt_xent(
            head(encode(model, args.family, first)),
            head(encode(model, args.family, second)), args.temperature,
        )

    # Sundial's released weights and remote implementation use float32 as the official
    # default; bf16 produced non-finite adapter gradients on real OHLCV smoke batches.
    use_amp = str(args.device).startswith("cuda") and args.family != "sundial_base"
    def validate():
        model.eval(); head.eval(); total = 0.0
        with torch.no_grad():
            for step in range(args.val_batches):
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    total += float(loss_for(val_schedule, step))
        return total / args.val_batches

    running = 0.0; count = 0; first_grad = None
    end_step = min(args.max_steps, args.stop_after_step or args.max_steps)
    model.train(); head.train()
    for step in range(start, end_step):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            loss = loss_for(train_schedule, step)
        loss.backward()
        grad = torch.sqrt(sum(
            value.grad.detach().float().square().sum()
            for value in backbone_parameters if value.grad is not None
        ))
        if first_grad is None:
            first_grad = float(grad)
            if not np.isfinite(first_grad) or first_grad <= 0:
                raise RuntimeError(
                    f"objective gradient did not reach the backbone: norm={first_grad}"
                )
        torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
        optimizer.step(); scheduler.step(); running += float(loss.detach()); count += 1
        completed = step + 1
        if completed % args.eval_every == 0 or completed == end_step:
            val = validate()
            row = {"step": completed, "train_loss": running / count, "val_loss": val,
                   "backbone_grad_norm": float(grad),
                   "learning_rate": scheduler.get_last_lr()[0]}
            history.append(row)
            print(f"[{args.family}:{args.stage}] step={completed} "
                  f"train={row['train_loss']:.6f} val={val:.6f} grad={float(grad):.4f}",
                  flush=True)
            if val < best_val:
                best_val = val; best_backbone = _backbone_state(model, args.family)
                _atomic_save(output, {
                    "schema_version": "ffm_control_staged_bundle_v1", "stage": args.stage,
                    "arm": arm.manifest(), "adaptation": adaptation,
                    "backbone_state": best_backbone, "parent": parent,
                    "input": {"context": MAX_CONTEXT, "horizon": FORECAST_HORIZON,
                              "normalization": "causal_context_per_channel_zscore",
                              "ohlcv_mode": arm.ohlcv_mode},
                })
            _atomic_save(state_path, {
                "schema_version": "ffm_control_staged_training_state_v1",
                "config_signature": signature, "step": completed, "history": history,
                "best_val": best_val, "backbone_state": _backbone_state(model, args.family),
                "best_backbone": best_backbone, "head_state": head.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            })
            running = 0.0; count = 0; model.train(); head.train()

    manifest = next(path for path in (
        Path(args.data_dir) / "TOURNAMENT_CACHE.json", Path(args.data_dir) / "MANIFEST.json",
    ) if path.is_file())
    report = {
        "schema_version": "ffm_control_staged_train_v1",
        "status": "complete" if end_step == args.max_steps else "partial",
        "created_utc": datetime.now(timezone.utc).isoformat(), "arm": arm.manifest(),
        "stage": args.stage, "adaptation": adaptation, "parent": parent,
        "split": {"train_start": TRAIN_START, "validation_start": VALIDATION_START,
                  "oos_start": OOS_START, "oos_read": False},
        "upstream": {"repo": str(source), "revision": args.source_revision},
        "data": {"streams": [stream["sid"] for stream in streams],
                 "data_manifest_sha256": _sha256(manifest),
                 "train_windows": int(len(train_starts)),
                 "validation_windows": int(len(val_starts)),
                 "examples_seen": int(end_step * args.batch_size)},
        "sampling": {"train_schedule_sha256": schedule_fingerprint(train_schedule, train_groups),
                     "validation_schedule_sha256": schedule_fingerprint(val_schedule, val_groups)},
        "config": vars(args), "best_val_loss": best_val,
        "first_backbone_grad_norm": first_grad, "history": history,
        "local_sources": _archive_sources(output, args, source),
        "checkpoint": {"path": str(output), "sha256": _sha256(output)},
        "training_state": {"path": str(state_path), "sha256": _sha256(state_path)},
    }
    _atomic_json(str(output) + ".report.json", report)
    return report


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("toto2_22m", "sundial_base"), required=True)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--data-dir", default="output/foundation_tournament/data_cache")
    parser.add_argument("--upstream-repo", required=True)
    parser.add_argument("--warm-checkpoint"); parser.add_argument("--output", required=True)
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
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=8400)
    parser.add_argument("--validation-seed", type=int, default=5400)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-step", type=int)
    parser.add_argument("--model-id"); parser.add_argument("--model-revision")
    parser.add_argument("--source-revision")
    return parser


def main():
    args = _parser().parse_args(); arm = get_arm(args.family)
    args.model_id = args.model_id or arm.model_id
    args.model_revision = args.model_revision or arm.model_revision
    args.source_revision = args.source_revision or arm.source_revision
    args.tickers = tuple(value for value in args.tickers.split(",") if value)
    args.timeframes = tuple(value for value in args.timeframes.split(",") if value)
    train(args)


if __name__ == "__main__":
    main()
