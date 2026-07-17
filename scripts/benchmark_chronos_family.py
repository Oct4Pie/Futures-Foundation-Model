#!/usr/bin/env python3
"""Validation-only, pinned Chronos v1/Bolt/v2 zero-shot forecast bake-off.

All candidates forecast the same close series on the same non-overlapping windows.
Chronos-2 additionally gets a separately labeled joint-OHLCV capability track.
No adaptation is performed and no locked OOS rows are loaded.  Context for the
earliest validation target may correctly extend into the preceding train interval.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from futures_foundation.finetune.chronos_family import (
    QUANTILE_LEVELS, benchmark_signature, evaluate_close_forecasts,
    persistence_quantiles, resolve_candidates,
)
from futures_foundation.finetune.kronos_eval import build_forecast_windows, window_fingerprint
from futures_foundation.finetune.tournament import OOS_START, VALIDATION_START, protocol


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def _atomic_npz(path, **values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp.npz")
    np.savez_compressed(tmp, **values)
    os.replace(tmp, path)


def _seed_everything(seed):
    import torch
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _load_pipeline(candidate, device, dtype, checkpoint=None):
    import torch
    from chronos import BaseChronosPipeline, Chronos2Pipeline
    torch_dtype = getattr(torch, dtype)
    cls = Chronos2Pipeline if candidate.family == "chronos_2" else BaseChronosPipeline
    pipeline = cls.from_pretrained(
        candidate.model_id, revision=candidate.revision,
        device_map=device, dtype=torch_dtype,
    )
    resolved = getattr(pipeline.inner_model.config, "_commit_hash", None)
    if resolved != candidate.revision:
        raise RuntimeError(
            f"model revision drift for {candidate.key}: expected {candidate.revision}, "
            f"loaded {resolved}"
        )
    checkpoint_meta = None
    if checkpoint is not None:
        checkpoint = Path(checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"trained checkpoint missing: {checkpoint}")
        bundle = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if bundle.get("schema_version") != "ffm_chronos_tournament_bundle_v1":
            raise ValueError(f"unsupported trained checkpoint schema: {checkpoint}")
        stamped = bundle.get("candidate") or {}
        if (stamped.get("key"), stamped.get("revision")) != (
                candidate.key, candidate.revision):
            raise ValueError(f"checkpoint candidate mismatch: {checkpoint}")
        pipeline.inner_model.load_state_dict(bundle["model_state"], strict=True)
        checkpoint_meta = {
            "path": str(checkpoint), "sha256": _sha256(checkpoint),
            "chronos2_mode": bundle.get("chronos2_mode"),
            "input": bundle.get("input"),
        }
    pipeline.inner_model.eval()
    return pipeline, checkpoint_meta


def _parse_checkpoint_map(value):
    output = {}
    if not value:
        return output
    for item in str(value).split(","):
        if "=" not in item:
            raise ValueError("checkpoint map entries must use candidate=path")
        key, path = (part.strip() for part in item.split("=", 1))
        if key not in {"chronos_v1", "chronos_bolt", "chronos_v2"}:
            raise ValueError(f"unknown checkpoint candidate: {key}")
        if not path or key in output:
            raise ValueError("checkpoint map paths must be nonempty and keys unique")
        output[key] = path
    return output


def _normalize_quantiles(candidate, values, batch_rows, horizon, n_quantiles,
                         *, close_channel=None):
    """Normalize official pipeline outputs to [B,H,Q]."""
    import torch
    if candidate.family == "chronos_2":
        if not isinstance(values, list) or len(values) != batch_rows:
            raise ValueError("Chronos-2 must return one tensor per input series")
        channel = 0 if close_channel is None else int(close_channel)
        values = torch.stack([row[channel] for row in values], dim=0)
    values = values.float().cpu().numpy()
    expected = (batch_rows, horizon, n_quantiles)
    if values.shape != expected:
        raise ValueError(f"forecast shape mismatch: expected {expected}, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("model returned non-finite quantile forecasts")
    return values.astype(np.float32, copy=False)


def _predict_batch(pipeline, candidate, contexts, *, horizon, levels, samples,
                   batch_size, joint_ohlcv):
    import torch
    if candidate.family == "chronos_2":
        inputs = contexts if joint_ohlcv else contexts[:, 3:4, :]
        quantiles, _mean = pipeline.predict_quantiles(
            torch.as_tensor(inputs), prediction_length=horizon,
            quantile_levels=list(levels), batch_size=batch_size,
            context_length=contexts.shape[-1], cross_learning=False,
        )
        return _normalize_quantiles(
            candidate, quantiles, len(contexts), horizon, len(levels),
            close_channel=3 if joint_ohlcv else 0,
        )
    quantiles, _mean = pipeline.predict_quantiles(
        torch.as_tensor(contexts[:, 3, :]), prediction_length=horizon,
        quantile_levels=list(levels),
        **({"num_samples": int(samples)} if candidate.family == "original_chronos_t5" else {}),
    )
    return _normalize_quantiles(
        candidate, quantiles, len(contexts), horizon, len(levels),
    )


def _forecast(pipeline, candidate, windows, *, cache_dir, signature, batch_size,
              samples, seed, joint_ohlcv=False):
    cache_dir = Path(cache_dir) / signature
    cache_dir.mkdir(parents=True, exist_ok=True)
    contexts = np.transpose(np.asarray(windows["context"], np.float32), (0, 2, 1))
    horizon = int(windows["horizon"])
    output = np.empty((len(contexts), horizon, len(QUANTILE_LEVELS)), np.float32)
    for lo in range(0, len(contexts), int(batch_size)):
        hi = min(len(contexts), lo + int(batch_size))
        shard = cache_dir / f"batch_{lo:07d}_{hi:07d}.npz"
        if shard.is_file():
            with np.load(shard, allow_pickle=False) as saved:
                if str(saved["signature"].item()) != signature:
                    raise ValueError(f"cache signature mismatch: {shard}")
                quantiles = saved["quantiles"]
            if quantiles.shape != output[lo:hi].shape:
                raise ValueError(f"cached forecast shape mismatch: {shard}")
            output[lo:hi] = quantiles
            print(f"[{candidate.key}] cache {lo}:{hi}", flush=True)
            continue
        _seed_everything(seed + lo)
        quantiles = _predict_batch(
            pipeline, candidate, contexts[lo:hi], horizon=horizon,
            levels=QUANTILE_LEVELS, samples=samples, batch_size=batch_size,
            joint_ohlcv=joint_ohlcv,
        )
        _atomic_npz(shard, signature=np.array(signature), quantiles=quantiles)
        output[lo:hi] = quantiles
        print(f"[{candidate.key}] generated {lo}:{hi}", flush=True)
    return output


def _archive_sources(output):
    from futures_foundation.finetune import chronos_family, kronos_eval, tournament
    destination = Path(output).parent / "source"
    destination.mkdir(parents=True, exist_ok=True)
    archived = {}
    for source in (
        Path(__file__).resolve(), Path(chronos_family.__file__).resolve(),
        Path(kronos_eval.__file__).resolve(), Path(tournament.__file__).resolve(),
    ):
        target = destination / source.name
        tmp = Path(str(target) + ".tmp")
        shutil.copyfile(source, tmp)
        os.replace(tmp, target)
        archived[source.name] = {"path": str(target.resolve()), "sha256": _sha256(target)}
    return archived


def benchmark(args):
    import torch
    candidates = resolve_candidates(args.candidates)
    separation = args.context + args.horizon
    windows = build_forecast_windows(
        args.data_dir, args.tickers, args.timeframes,
        context=args.context, horizon=args.horizon,
        eval_start=VALIDATION_START, eval_end=OOS_START,
        max_per_stream=args.max_per_stream, separation_bars=separation,
        seed=args.seed, chunksize=args.csv_chunksize,
    )
    fingerprint = window_fingerprint(windows)
    results = {
        "persistence": evaluate_close_forecasts(
            windows, persistence_quantiles(windows), QUANTILE_LEVELS,
        )
    }
    configs = {}
    versions = {
        name: importlib.metadata.version(name)
        for name in ("chronos-forecasting", "torch", "transformers", "numpy")
    }
    for candidate in candidates:
        modes = ("close_only", "joint_ohlcv") if (
            candidate.native_multivariate and args.chronos2_joint
        ) else ("close_only",)
        pipeline, checkpoint_meta = _load_pipeline(
            candidate, args.device, args.dtype, args.checkpoints.get(candidate.key),
        )
        for mode in modes:
            key = candidate.key if mode == "close_only" else f"{candidate.key}_joint_ohlcv"
            config = {
                "schema_version": "ffm_chronos_family_candidate_v1",
                "candidate": candidate.manifest(), "mode": mode,
                "window_fingerprint": fingerprint,
                "context": args.context, "horizon": args.horizon,
                "quantile_levels": list(QUANTILE_LEVELS),
                "original_chronos_samples": args.samples,
                "seed": args.seed, "package_versions": versions,
                "trained_checkpoint": checkpoint_meta,
                "inference": {
                    "batch_size": args.batch_size, "dtype": args.dtype,
                    "device": args.device, "cross_learning": False,
                },
            }
            signature = benchmark_signature(config)
            quantiles = _forecast(
                pipeline, candidate, windows, cache_dir=args.cache_dir,
                signature=signature, batch_size=args.batch_size,
                samples=args.samples, seed=args.seed,
                joint_ohlcv=mode == "joint_ohlcv",
            )
            results[key] = evaluate_close_forecasts(windows, quantiles, QUANTILE_LEVELS)
            configs[key] = {**config, "cache_signature": signature}
        del pipeline
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    manifest = Path(args.data_dir) / "MANIFEST.json"
    report = {
        "schema_version": "ffm_chronos_family_benchmark_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "policy": (
            "development benchmark of configured frozen or train-adapted checkpoints; "
            "validation interval only; no benchmark-time adaptation and no locked OOS access; "
            "earliest validation contexts may use "
            "preceding train-period history; upstream pretraining overlap is unknown"
        ),
        "protocol": protocol(),
        "data": {
            "data_dir": str(Path(args.data_dir).resolve()),
            "corpus_manifest_sha256": _sha256(manifest) if manifest.is_file() else None,
            "window_fingerprint": fingerprint,
            "windows": int(len(windows["context"])),
            "per_stream": windows["counts"],
            "tickers": list(args.tickers), "timeframes": list(args.timeframes),
            "validation_start": windows["eval_start"],
            "validation_end_exclusive": windows["eval_end"],
            "separation_bars": separation,
            "overlap_policy": "complete context+future parents do not overlap per stream",
        },
        "config": {
            "candidates": configs, "batch_size": args.batch_size,
            "dtype": args.dtype, "device": args.device,
        },
        "results": results,
        "environment": versions,
        "local_sources": _archive_sources(args.output),
    }
    _atomic_json(args.output, report)
    summary = {
        key: {
            "macro_path_skill": value["macro_stream"]["path_skill_vs_persistence"],
            "macro_fwd_dir_auc": value["macro_stream"]["fwd_dir_auc"],
            "macro_fwd_absmove_r2": value["macro_stream"]["fwd_absmove_r2"],
            "gate": value["diagnostic_gate"],
        }
        for key, value in results.items()
    }
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[chronos-family] report -> {args.output}", flush=True)
    return report


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/ssl_corpus_v2_6tf")
    parser.add_argument("--output", default="output/chronos_family/validation/report.json")
    parser.add_argument("--cache-dir", default="output/chronos_family/cache")
    parser.add_argument(
        "--candidates", default="chronos_v1,chronos_bolt,chronos_v2",
        help="comma-separated pinned candidate keys",
    )
    parser.add_argument("--tickers", default="ES,NQ,RTY,YM,GC,SI,CL,ZB,ZN")
    parser.add_argument("--timeframes", default="1min,3min,5min,15min,30min,60min")
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--max-per-stream", type=int, default=200)
    parser.add_argument("--csv-chunksize", type=int, default=250000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument(
        "--checkpoint-map", default="",
        help="comma-separated candidate=/path/to/trained.pt entries",
    )
    parser.add_argument("--seed", type=int, default=5400)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument(
        "--no-chronos2-joint", action="store_false", dest="chronos2_joint",
        help="disable the separately labeled Chronos-2 joint-OHLCV capability track",
    )
    parser.set_defaults(chronos2_joint=True)
    return parser


def main():
    args = _parser().parse_args()
    args.candidates = tuple(x.strip() for x in args.candidates.split(",") if x.strip())
    args.tickers = tuple(x.strip() for x in args.tickers.split(",") if x.strip())
    args.timeframes = tuple(x.strip() for x in args.timeframes.split(",") if x.strip())
    args.checkpoints = _parse_checkpoint_map(args.checkpoint_map)
    if args.context < 8 or args.context > 256:
        raise ValueError("tournament context must lie in [8,256]")
    if args.horizon != 16:
        raise ValueError("foundation_5y1y1y_v1 forecast horizon is locked to 16")
    if args.samples < 1:
        raise ValueError("original Chronos sample count must be positive")
    benchmark(args)


if __name__ == "__main__":
    main()
