#!/usr/bin/env python3
"""Leak-safe two-phase Kronos-small adaptation for the equal-history tournament."""
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

from futures_foundation.finetune.tournament import (
    FORECAST_HORIZON, MAX_CONTEXT, OOS_START, PARENT_LENGTH, TRAIN_START, VALIDATION_START,
)
from futures_foundation.finetune.tournament_data import (
    balanced_schedule, gather_parent, gather_time_features, load_adaptation_data,
    schedule_fingerprint,
)
from futures_foundation.finetune import tournament, tournament_data
from futures_foundation.finetune.foundation_roster import get_arm


SOURCE_REVISION = "67b630e67f6a18c9e9be918d9b4337c960db1e9a"
MODEL_ID = "NeoQuasar/Kronos-small"
MODEL_REVISION = "901c26c1332695a2a8f243eb2f37243a37bea320"
TOKENIZER_ID = "NeoQuasar/Kronos-Tokenizer-base"
TOKENIZER_REVISION = "0e0117387f39004a9016484a186a908917e22426"


def _sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


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


def _archive_sources(output):
    destination = Path(str(output) + ".source")
    destination.mkdir(parents=True, exist_ok=True)
    archived = {}
    for source in (Path(__file__).resolve(), Path(tournament.__file__).resolve(),
                   Path(tournament_data.__file__).resolve()):
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
    source = repo / "model" / "kronos.py"
    if not source.is_file():
        raise FileNotFoundError(f"official Kronos source missing: {source}")
    actual = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True,
    ).strip()
    if actual != revision:
        raise ValueError(f"Kronos revision mismatch: expected {revision}, got {actual}")
    return repo, source


def _seed(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_models(repo, args):
    sys.path.insert(0, str(repo))
    try:
        from model import Kronos, KronosTokenizer
    finally:
        sys.path.pop(0)
    tokenizer = KronosTokenizer.from_pretrained(
        args.tokenizer_id, revision=args.tokenizer_revision,
    )
    predictor = Kronos.from_pretrained(args.model_id, revision=args.model_revision)
    return tokenizer, predictor


def _load_warm_bundle(path, tokenizer, predictor, args):
    """Load a staged parent bundle and reject model/tokenizer identity drift."""
    import torch
    if not path:
        return None
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Kronos warm checkpoint missing: {path}")
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    if bundle.get("schema_version") != "ffm_kronos_tournament_bundle_v1":
        raise ValueError("unsupported Kronos warm checkpoint schema")
    if bundle.get("tokenizer") != {
            "id": args.tokenizer_id, "revision": args.tokenizer_revision}:
        raise ValueError("Kronos warm checkpoint tokenizer identity mismatch")
    if bundle.get("predictor") != {
            "id": args.model_id, "revision": args.model_revision}:
        raise ValueError("Kronos warm checkpoint predictor identity mismatch")
    tokenizer.load_state_dict(bundle["tokenizer_state"], strict=True)
    predictor.load_state_dict(bundle["predictor_state"], strict=True)
    return {"path": str(path), "sha256": _sha256(path),
            "stage": bundle.get("stage")}


def _context_normalized_values(raw, clip):
    """Add amount and normalize the whole parent with context-only statistics."""
    raw = np.asarray(raw, np.float32)
    amount = raw[:, :, 4:5] * raw[:, :, :4].mean(axis=2, keepdims=True)
    values = np.concatenate((raw, amount), axis=2).astype(np.float32)
    context = values[:, :MAX_CONTEXT]
    mean = context.mean(axis=1, keepdims=True)
    std = context.std(axis=1, keepdims=True)
    return np.clip((values - mean) / np.maximum(std, 1e-5), -clip, clip)


def _normalized_batch(big, starts, group_ids, streams, row_bounds, device, clip):
    import torch
    raw = gather_parent(big, starts, PARENT_LENGTH)
    # Strictly context-only statistics. The official CSV trainer standardizes the complete
    # context+future parent and therefore leaks future scale into predictor inputs.
    values = _context_normalized_values(raw, clip)
    stamps = gather_time_features(starts, group_ids, streams, row_bounds, PARENT_LENGTH)
    return (torch.as_tensor(values, device=device),
            torch.as_tensor(stamps, device=device))


def _tokenizer_loss(tokenizer, values):
    import torch.nn.functional as F
    (coarse, full), quantizer_loss, _, _ = tokenizer(values)
    reconstruction = F.mse_loss(coarse, values) + F.mse_loss(full, values)
    return (reconstruction + quantizer_loss) / 2


def _predictor_loss(predictor, tokenizer, values, stamps):
    import torch
    with torch.no_grad():
        first, second = tokenizer.encode(values, half=True)
    logits = predictor(first[:, :-1], second[:, :-1], stamps[:, :-1])
    return predictor.head.compute_loss(
        logits[0], logits[1], first[:, 1:], second[:, 1:]
    )[0]


def train(args):
    import torch
    arm = get_arm(args.arm)
    if arm.family != "kronos" or not arm.supported_training:
        raise ValueError(f"{args.arm} is not an admitted trainable Kronos arm")
    if (args.model_id, args.model_revision, args.source_revision) != (
            arm.model_id, arm.model_revision, arm.source_revision):
        raise ValueError("Kronos model/source pins do not match the admitted arm")
    repo, source = _validate_source(args.kronos_repo, args.source_revision)
    if min(args.tokenizer_steps, args.predictor_steps, args.eval_every, args.val_batches) < 0:
        raise ValueError("training budgets cannot be negative")
    if args.tokenizer_steps + args.predictor_steps < 1:
        raise ValueError("at least one Kronos training phase must be positive")
    if args.eval_every < 1 or args.val_batches < 1:
        raise ValueError("evaluation budgets must be positive")
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
    train_examples = max(args.tokenizer_steps, args.predictor_steps) * args.batch_size
    train_schedule, train_groups = balanced_schedule(
        train_starts, groups["train_bounds"], train_examples, args.seed,
    )
    val_schedule, val_groups = balanced_schedule(
        val_starts, groups["val_bounds"], args.val_batches * args.batch_size,
        args.validation_seed,
    )
    tokenizer, predictor = _load_models(repo, args)
    warm_parent = _load_warm_bundle(args.warm_checkpoint, tokenizer, predictor, args)
    tokenizer, predictor = tokenizer.to(args.device), predictor.to(args.device)
    history = {"tokenizer": [], "predictor": []}
    resume_state = torch.load(state_path, map_location="cpu") if args.resume else None
    if resume_state is not None:
        if resume_state.get("config_signature") != signature:
            raise ValueError("exact resume refused: configuration changed")
        history = resume_state["history"]

    def batch(schedule, schedule_groups, step):
        lo = step * args.batch_size
        return _normalized_batch(
            big, schedule[lo:lo + args.batch_size],
            schedule_groups[lo:lo + args.batch_size], streams, groups["row_bounds"],
            args.device, args.clip,
        )

    def validate_tokenizer():
        tokenizer.eval()
        total = 0.0
        with torch.no_grad():
            for step in range(args.val_batches):
                values, _ = batch(val_schedule, val_groups, step)
                with torch.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=str(args.device).startswith("cuda")):
                    total += float(_tokenizer_loss(tokenizer, values))
        return total / args.val_batches

    def validate_predictor():
        tokenizer.eval(); predictor.eval()
        total = 0.0
        with torch.no_grad():
            for step in range(args.val_batches):
                values, stamps = batch(val_schedule, val_groups, step)
                with torch.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=str(args.device).startswith("cuda")):
                    total += float(_predictor_loss(predictor, tokenizer, values, stamps))
        return total / args.val_batches

    best_tokenizer = {key: value.detach().cpu() for key, value in tokenizer.state_dict().items()}
    best_tokenizer_val = float("inf")
    tokenizer_start = 0
    resumed_phase = resume_state.get("phase") if resume_state is not None else None
    if resumed_phase == "tokenizer":
        tokenizer.load_state_dict(resume_state["tokenizer"])
        best_tokenizer = resume_state["best_tokenizer"]
        best_tokenizer_val = resume_state["best_tokenizer_val"]
        tokenizer_start = int(resume_state["step"])
    elif resumed_phase == "predictor":
        best_tokenizer = resume_state["tokenizer"]
        best_tokenizer_val = resume_state["best_tokenizer_val"]
        tokenizer.load_state_dict(best_tokenizer)

    if args.tokenizer_steps and resumed_phase != "predictor":
        optimizer = torch.optim.AdamW(
            tokenizer.parameters(), lr=args.tokenizer_learning_rate,
            weight_decay=args.weight_decay, betas=(0.9, 0.95),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.tokenizer_steps,
        )
        if resumed_phase == "tokenizer":
            optimizer.load_state_dict(resume_state["optimizer"])
            scheduler.load_state_dict(resume_state["scheduler"])
            torch.set_rng_state(resume_state["torch_rng"])
            if torch.cuda.is_available() and resume_state.get("cuda_rng") is not None:
                torch.cuda.set_rng_state_all(resume_state["cuda_rng"])
        running = 0.0; running_steps = 0
        tokenizer.train()
        for step in range(tokenizer_start, args.tokenizer_steps):
            values, _ = batch(train_schedule, train_groups, step)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=str(args.device).startswith("cuda")):
                loss = _tokenizer_loss(tokenizer, values)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tokenizer.parameters(), args.grad_clip)
            optimizer.step(); scheduler.step()
            running += float(loss.detach()); running_steps += 1
            completed = step + 1
            if completed % args.eval_every == 0 or completed == args.tokenizer_steps:
                val = validate_tokenizer()
                row = {"step": completed, "train_loss": running / running_steps,
                       "val_loss": val, "learning_rate": scheduler.get_last_lr()[0]}
                history["tokenizer"].append(row)
                print(f"[kronos-tokenizer] step={completed} train={row['train_loss']:.6f} "
                      f"val={val:.6f}", flush=True)
                running = 0.0; running_steps = 0
                if val < best_tokenizer_val:
                    best_tokenizer_val = val
                    best_tokenizer = {
                        key: value.detach().cpu() for key, value in tokenizer.state_dict().items()
                    }
                _atomic_save(state_path, {
                    "schema_version": "ffm_kronos_training_state_v1",
                    "config_signature": signature, "phase": "tokenizer", "step": completed,
                    "tokenizer": tokenizer.state_dict(), "best_tokenizer": best_tokenizer,
                    "best_tokenizer_val": best_tokenizer_val,
                    "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "history": history, "torch_rng": torch.get_rng_state(),
                    "cuda_rng": (torch.cuda.get_rng_state_all()
                                 if torch.cuda.is_available() else None),
                })
                tokenizer.train()
        tokenizer.load_state_dict(best_tokenizer)
    else:
        best_tokenizer_val = validate_tokenizer()

    for parameter in tokenizer.parameters():
        parameter.requires_grad_(False)
    tokenizer.eval()
    optimizer = torch.optim.AdamW(
        predictor.parameters(), lr=args.predictor_learning_rate,
        weight_decay=args.weight_decay, betas=(0.9, 0.95),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.predictor_steps),
    )
    best_predictor_val = float("inf")
    best_predictor = None
    predictor_start = 0
    if resumed_phase == "predictor":
        predictor.load_state_dict(resume_state["predictor"])
        optimizer.load_state_dict(resume_state["optimizer"])
        scheduler.load_state_dict(resume_state["scheduler"])
        best_predictor_val = resume_state["best_predictor_val"]
        best_predictor = resume_state["best_predictor"]
        predictor_start = int(resume_state["step"])
        torch.set_rng_state(resume_state["torch_rng"])
        if torch.cuda.is_available() and resume_state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(resume_state["cuda_rng"])
    running = 0.0; running_steps = 0
    predictor.train()
    for step in range(predictor_start, args.predictor_steps):
        values, stamps = batch(train_schedule, train_groups, step)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=str(args.device).startswith("cuda")):
            loss = _predictor_loss(predictor, tokenizer, values, stamps)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), args.grad_clip)
        optimizer.step(); scheduler.step()
        running += float(loss.detach()); running_steps += 1
        completed = step + 1
        if completed % args.eval_every == 0 or completed == args.predictor_steps:
            val = validate_predictor()
            row = {"step": completed, "train_loss": running / running_steps,
                   "val_loss": val, "learning_rate": scheduler.get_last_lr()[0]}
            history["predictor"].append(row)
            print(f"[kronos-predictor] step={completed} train={row['train_loss']:.6f} "
                  f"val={val:.6f}", flush=True)
            running = 0.0; running_steps = 0
            if val < best_predictor_val:
                best_predictor_val = val
                best_predictor = {
                    key: value.detach().cpu() for key, value in predictor.state_dict().items()
                }
                _atomic_save(output, {
                    "schema_version": "ffm_kronos_tournament_bundle_v1",
                    "stage": "stage3_forecast",
                    "parent": warm_parent,
                    "tokenizer_state": best_tokenizer,
                    "predictor_state": best_predictor,
                    "tokenizer": {"id": args.tokenizer_id, "revision": args.tokenizer_revision},
                    "predictor": {"id": args.model_id, "revision": args.model_revision},
                    "input": {"context": MAX_CONTEXT, "horizon": FORECAST_HORIZON,
                              "normalization": "context_only_per_channel_zscore_clip"},
                })
            _atomic_save(state_path, {
                "schema_version": "ffm_kronos_training_state_v1", "phase": "predictor",
                "config_signature": signature,
                "step": completed, "tokenizer": best_tokenizer,
                "best_tokenizer_val": best_tokenizer_val,
                "predictor": predictor.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "history": history,
                "best_predictor_val": best_predictor_val,
                "best_predictor": best_predictor,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            })
            predictor.train()

    if args.predictor_steps == 0:
        best_predictor_val = None
        best_predictor = {
            key: value.detach().cpu() for key, value in predictor.state_dict().items()
        }
        _atomic_save(output, {
            "schema_version": "ffm_kronos_tournament_bundle_v1",
            "stage": "stage1_reconstruction",
            "parent": warm_parent,
            "tokenizer_state": best_tokenizer,
            "predictor_state": best_predictor,
            "tokenizer": {"id": args.tokenizer_id, "revision": args.tokenizer_revision},
            "predictor": {"id": args.model_id, "revision": args.model_revision},
            "input": {"context": MAX_CONTEXT, "horizon": FORECAST_HORIZON,
                      "normalization": "context_only_per_channel_zscore_clip"},
        })

    report = {
        "schema_version": "ffm_kronos_tournament_train_v1", "status": "complete",
        "arm": arm.manifest(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "split": {"train_start": TRAIN_START, "validation_start": VALIDATION_START,
                  "oos_start": OOS_START, "oos_read": False},
        "models": {
            "tokenizer": {"id": args.tokenizer_id, "revision": args.tokenizer_revision,
                          "parameters": sum(p.numel() for p in tokenizer.parameters())},
            "predictor": {"id": args.model_id, "revision": args.model_revision,
                          "parameters": sum(p.numel() for p in predictor.parameters())},
        },
        "upstream": {"repo": str(repo), "revision": args.source_revision,
                     "source_sha256": _sha256(source)},
        "parent": warm_parent,
        "local_sources": _archive_sources(output),
        "data": {"streams": [stream["sid"] for stream in streams],
                 "data_manifest_sha256": _sha256(
                     next(path for path in (
                         Path(args.data_dir) / "TOURNAMENT_CACHE.json",
                         Path(args.data_dir) / "MANIFEST.json",
                     ) if path.is_file())),
                 "train_windows": int(len(train_starts)),
                 "validation_windows": int(len(val_starts)),
                 "tokenizer_examples_seen": int(args.tokenizer_steps * args.batch_size),
                 "predictor_examples_seen": int(args.predictor_steps * args.batch_size)},
        "sampling": {"train_schedule_sha256": schedule_fingerprint(
                          train_schedule, train_groups),
                     "validation_schedule_sha256": schedule_fingerprint(
                          val_schedule, val_groups)},
        "config": vars(args), "best_tokenizer_val_loss": best_tokenizer_val,
        "best_predictor_val_loss": best_predictor_val, "history": history,
        "checkpoint": {"path": str(output), "sha256": _sha256(output)},
        "training_state": {"path": str(state_path), "sha256": _sha256(state_path)},
    }
    _atomic_json(str(output) + ".report.json", report)
    return report


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/ssl_corpus_v2_6tf")
    parser.add_argument("--kronos-repo", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tickers", default="ES,NQ,RTY,YM,GC,SI,CL,ZB,ZN")
    parser.add_argument("--timeframes", default="1min,3min,5min,15min,30min,60min")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--tokenizer-steps", type=int, default=1000)
    parser.add_argument("--predictor-steps", type=int, default=3000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--val-batches", type=int, default=64)
    parser.add_argument("--tokenizer-learning-rate", type=float, default=2e-5)
    parser.add_argument("--predictor-learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=3.0)
    parser.add_argument("--clip", type=float, default=5.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validation-seed", type=int, default=5400)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--warm-checkpoint")
    parser.add_argument("--arm", choices=("kronos_mini", "kronos_small"),
                        default="kronos_small")
    parser.add_argument("--source-revision", default=SOURCE_REVISION)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--tokenizer-id", default=TOKENIZER_ID)
    parser.add_argument("--tokenizer-revision", default=TOKENIZER_REVISION)
    return parser


def main():
    args = _parser().parse_args()
    arm = get_arm(args.arm)
    # Arm selection owns immutable model/source pins.  Explicit conflicting values are
    # rejected later rather than silently loading a different checkpoint.
    if args.model_id == MODEL_ID and args.model_revision == MODEL_REVISION:
        args.model_id, args.model_revision = arm.model_id, arm.model_revision
    args.tickers = tuple(x for x in args.tickers.split(",") if x)
    args.timeframes = tuple(x for x in args.timeframes.split(",") if x)
    train(args)


if __name__ == "__main__":
    main()
