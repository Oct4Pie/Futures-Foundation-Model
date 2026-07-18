#!/usr/bin/env python3
"""Zero-shot, post-pretraining Kronos forecast benchmark on the sealed OHLCV corpus.

The official Kronos source is kept external and pinned.  Expensive forecast batches are
atomically cached, so an interrupted run resumes without regenerating completed batches.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import random
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from futures_foundation.finetune.kronos_eval import (
    build_forecast_windows, evaluate_predictions, window_fingerprint,
)
from futures_foundation.finetune import kronos_eval, ssl_data
from futures_foundation.finetune.foundation_roster import get_arm
from futures_foundation.finetune.native_contracts import (
    add_admission_argument,
    require_admission_from_args,
    technical_runtime_contract,
    validate_identity,
    validate_runtime_contract,
)
from scripts.predict_foundation_forecasts import _forecast_runtime_facts


KRONOS_SOURCE_REVISION = "67b630e67f6a18c9e9be918d9b4337c960db1e9a"
MODEL_ID = "NeoQuasar/Kronos-small"
MODEL_REVISION = "901c26c1332695a2a8f243eb2f37243a37bea320"
TOKENIZER_ID = "NeoQuasar/Kronos-Tokenizer-base"
TOKENIZER_REVISION = "0e0117387f39004a9016484a186a908917e22426"


def _sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def _atomic_npz(path, **values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(tmp, **values)
    os.replace(tmp, path)


def _archive_local_sources(output):
    """Preserve the exact local benchmark logic, even when the Git tree is dirty."""
    destination = Path(output).parent / "source"
    destination.mkdir(parents=True, exist_ok=True)
    sources = (Path(__file__).resolve(), Path(kronos_eval.__file__).resolve(),
               Path(ssl_data.__file__).resolve())
    archived = {}
    for source in sources:
        target = destination / source.name
        tmp = target.with_suffix(target.suffix + ".tmp")
        shutil.copyfile(source, tmp)
        os.replace(tmp, target)
        archived[source.name] = {"path": str(target.resolve()), "sha256": _sha256(target)}
    return archived


def _git_revision(path):
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"{path} is not a readable Git checkout") from exc


def _validate_source(path, expected_revision):
    path = Path(path).expanduser().resolve()
    source = path / "model" / "kronos.py"
    if not source.is_file():
        raise FileNotFoundError(f"{source} not found; clone the official Kronos repository")
    actual = _git_revision(path)
    if actual != expected_revision:
        raise ValueError(
            f"Kronos source revision mismatch: expected {expected_revision}, got {actual}. "
            "Use the pinned revision so benchmark code cannot drift."
        )
    return path, source


def _seed_everything(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_predictor(repo, device, model_id, model_revision, tokenizer_id,
                    tokenizer_revision, max_context):
    sys.path.insert(0, str(repo))
    try:
        from model import Kronos, KronosPredictor, KronosTokenizer
    finally:
        sys.path.pop(0)
    tokenizer = KronosTokenizer.from_pretrained(tokenizer_id, revision=tokenizer_revision)
    model = Kronos.from_pretrained(model_id, revision=model_revision)
    tokenizer.eval()
    model.eval()
    return KronosPredictor(model, tokenizer, device=device, max_context=max_context)


def _signature(config):
    raw = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _context_frame(context, input_mode):
    columns = ("open", "high", "low", "close", "volume")
    if input_mode == "ohlc":
        return pd.DataFrame(np.asarray(context)[:, :4], columns=columns[:4])
    if input_mode == "ohlcv":
        return pd.DataFrame(np.asarray(context)[:, :5], columns=columns)
    raise ValueError(f"unsupported Kronos input mode: {input_mode}")


def _predict(predictor, windows, *, cache_dir, signature, batch_size, horizon,
             temperature, top_k, top_p, sample_count, seed, input_mode):
    cache_dir = Path(cache_dir) / signature
    cache_dir.mkdir(parents=True, exist_ok=True)
    predictions = np.empty((len(windows["context"]), horizon, 6), np.float32)
    for lo in range(0, len(predictions), batch_size):
        hi = min(len(predictions), lo + batch_size)
        shard = cache_dir / f"batch_{lo:07d}_{hi:07d}.npz"
        if shard.is_file():
            with np.load(shard, allow_pickle=False) as saved:
                if str(saved["signature"].item()) != signature:
                    raise ValueError(f"cache signature mismatch: {shard}")
                pred = saved["prediction"]
            if pred.shape != (hi - lo, horizon, 6):
                raise ValueError(f"invalid cached prediction shape in {shard}: {pred.shape}")
            predictions[lo:hi] = pred
            print(f"[kronos] cache {lo}:{hi}", flush=True)
            continue
        _seed_everything(seed + lo)
        frames, x_times, y_times = [], [], []
        for row in range(lo, hi):
            frames.append(_context_frame(windows["context"][row], input_mode))
            x_times.append(pd.Series(pd.to_datetime(
                windows["context_time_ns"][row], unit="ns", utc=True)))
            y_times.append(pd.Series(pd.to_datetime(
                windows["future_time_ns"][row], unit="ns", utc=True)))
        outputs = predictor.predict_batch(
            frames, x_times, y_times, pred_len=horizon, T=temperature,
            top_k=top_k, top_p=top_p, sample_count=sample_count, verbose=False,
        )
        pred = np.stack([frame[["open", "high", "low", "close", "volume", "amount"]]
                         .to_numpy(np.float32) for frame in outputs])
        if not np.isfinite(pred).all():
            raise ValueError(f"Kronos returned non-finite predictions for rows {lo}:{hi}")
        predictions[lo:hi] = pred
        _atomic_npz(shard, signature=np.array(signature), prediction=pred)
        print(f"[kronos] generated {lo}:{hi}", flush=True)
    return predictions


def benchmark(args):
    validate_identity(
        args.arm,
        model_id=args.model_id,
        model_revision=args.model_revision,
        source_revision=args.source_revision,
        tokenizer_id=args.tokenizer_id,
        tokenizer_revision=args.tokenizer_revision,
    )
    admission = require_admission_from_args(
        args, arm_key=args.arm, track="F", route=None, require_training=False,
    )
    decoding = {
        "temperature": float(args.temperature),
        "top_k": int(args.top_k),
        "top_p": float(args.top_p),
        "sample_count": int(args.sample_count),
    }
    runtime_contract = validate_runtime_contract(
        args.arm, "F", _forecast_runtime_facts(
            args.arm,
            context_length=args.context,
            prediction_length=args.horizon,
            dtype="float32",
            samples=args.sample_count,
            kronos_decoding=decoding,
        ),
    )
    if decoding not in runtime_contract["decoding"]:
        raise ValueError(
            f"Kronos decoding is outside the evidence-covered configurations: {decoding}"
        )
    if args.input_mode != "ohlcv":
        raise ValueError("current Kronos native evidence covers joint OHLCVA input only")
    timeframe_minutes = sorted(int(value.removesuffix("min")) for value in args.timeframes)
    unsupported = sorted(set(timeframe_minutes) - set(runtime_contract["timeframes_minutes"]))
    if unsupported:
        raise ValueError(f"Kronos timeframe minutes are outside technical evidence: {unsupported}")
    repo, source = _validate_source(args.kronos_repo, args.source_revision)
    windows = build_forecast_windows(
        args.data_dir, args.tickers, args.timeframes, context=args.context,
        horizon=args.horizon, eval_start=args.eval_start, eval_end=args.eval_end,
        max_per_stream=args.max_per_stream, separation_bars=args.separation_bars,
        seed=args.seed, chunksize=args.csv_chunksize,
    )
    fingerprint = window_fingerprint(windows)
    config = {
        "schema_version": "ffm_kronos_zero_shot_v1",
        "admission": {
            "integrity": admission["integrity"],
            "registry_sha256": admission["registry_sha256"],
            "dossier_sha256": admission["dossier_sha256"],
            "evidence_registry_sha256": admission["evidence_registry_sha256"],
            "technical_evidence_id": admission["technical_evidence_id"],
        },
        "technical_runtime_contract": technical_runtime_contract(args.arm, "F"),
        "window_fingerprint": fingerprint,
        "source_revision": args.source_revision,
        "source_sha256": _sha256(source),
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "tokenizer_id": args.tokenizer_id,
        "tokenizer_revision": args.tokenizer_revision,
        "context": args.context,
        "horizon": args.horizon,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "sample_count": args.sample_count,
        "seed": args.seed,
        "input_mode": args.input_mode,
        "amount_policy": ("official predictor: volume and amount set to zero"
                          if args.input_mode == "ohlc" else
                          "official predictor: amount = volume * mean(OHLC)"),
    }
    signature = _signature(config)
    predictor = _load_predictor(
        repo, args.device, args.model_id, args.model_revision, args.tokenizer_id,
        args.tokenizer_revision, args.context,
    )
    prediction = _predict(
        predictor, windows, cache_dir=args.cache_dir, signature=signature,
        batch_size=args.batch_size, horizon=args.horizon,
        temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
        sample_count=args.sample_count, seed=args.seed, input_mode=args.input_mode,
    )
    metrics = evaluate_predictions(windows, prediction)
    manifest = Path(args.data_dir) / "MANIFEST.json"
    versions = {}
    for package in ("torch", "numpy", "pandas", "huggingface_hub", "einops", "safetensors"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    report = {
        "schema_version": "ffm_kronos_zero_shot_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "policy": "zero-shot only; evaluation strictly after published June-2024 pretraining cutoff",
        "config": config,
        "signature": signature,
        "data": {
            "data_dir": str(Path(args.data_dir).resolve()),
            "corpus_manifest_sha256": _sha256(manifest) if manifest.is_file() else None,
            "eval_start": windows["eval_start"], "eval_end": windows["eval_end"],
            "tickers": list(args.tickers), "timeframes": list(args.timeframes),
            "windows": int(len(windows["context"])), "per_stream": windows["counts"],
            "separation_bars": windows["separation_bars"],
            "target_overlap_policy": "future label windows do not overlap within a stream",
        },
        "upstream": {
            "repository": "https://github.com/shiyu-coder/Kronos",
            "checkout": str(repo), "source_revision": args.source_revision,
            "source_sha256": _sha256(source),
        },
        "local_sources": _archive_local_sources(args.output),
        "environment": versions,
        "metrics": metrics,
    }
    _atomic_json(args.output, report)
    prediction_path = Path(args.output).with_suffix(".predictions.npz")
    _atomic_npz(
        prediction_path, signature=np.array(signature), prediction=prediction,
        actual=windows["future"], ticker=windows["ticker"],
        timeframe=windows["timeframe"], target_time_ns=windows["future_time_ns"],
    )
    print(json.dumps(metrics["overall"], indent=2), flush=True)
    print(f"[kronos] report -> {args.output}", flush=True)
    print(f"[kronos] predictions -> {prediction_path}", flush=True)
    return report


def _parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="data/ssl_corpus_v2_6tf")
    p.add_argument("--kronos-repo", required=True)
    p.add_argument("--output", default="output/kronos/zero_shot_report.json")
    p.add_argument("--cache-dir", default="output/kronos/cache")
    p.add_argument("--tickers", default="ES,NQ,RTY,YM,GC,SI,CL,ZB,ZN")
    p.add_argument("--timeframes", default="1min,3min,5min,15min,30min,60min")
    p.add_argument("--eval-start", default="2024-07-01")
    p.add_argument("--eval-end", default="2025-07-01")
    p.add_argument("--context", type=int, default=512)
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--max-per-stream", type=int, default=200)
    p.add_argument("--separation-bars", type=int, default=16)
    p.add_argument("--csv-chunksize", type=int, default=250000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=1,
                   help="1 is deterministic greedy decoding; use 0 for paper-style sampling")
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--sample-count", type=int, default=1)
    p.add_argument("--input-mode", choices=("ohlcv", "ohlc"), default="ohlcv")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--arm", choices=("kronos_mini", "kronos_small"),
                   default="kronos_small")
    p.add_argument("--source-revision", default=KRONOS_SOURCE_REVISION)
    p.add_argument("--model-id")
    p.add_argument("--model-revision")
    p.add_argument("--tokenizer-id")
    p.add_argument("--tokenizer-revision")
    add_admission_argument(p)
    return p


def main():
    args = _parser().parse_args()
    arm = get_arm(args.arm)
    args.model_id = args.model_id or arm.model_id
    args.model_revision = args.model_revision or arm.model_revision
    args.tokenizer_id = args.tokenizer_id or arm.tokenizer_id
    args.tokenizer_revision = args.tokenizer_revision or arm.tokenizer_revision
    args.tickers = tuple(x.strip() for x in args.tickers.split(",") if x.strip())
    args.timeframes = tuple(x.strip() for x in args.timeframes.split(",") if x.strip())
    benchmark(args)


if __name__ == "__main__":
    main()
