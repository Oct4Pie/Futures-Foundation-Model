#!/usr/bin/env python3
"""Frozen MOMENT-small versus vanilla and Stage-2 MantisV2 on shared causal windows.

This is a representation probe, not a trading backtest.  Every candidate receives the exact
same raw contexts, targets, expanding walk-forward folds, and linear probe family.  MOMENT's
unreduced per-channel patch outputs are valid-patch pooled and concatenated so OHLCV channel
identity is not destroyed by its default cross-channel mean.
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

from futures_foundation.finetune import kronos_eval, moment_eval, ssl_data, ssl_probe
from futures_foundation.finetune.kronos_eval import build_forecast_windows, window_fingerprint
from futures_foundation.finetune.moment_eval import (
    left_pad_contexts, pool_channel_patches, targets_from_context_future,
)
from futures_foundation.finetune.native_contracts import verify_admission_report


MOMENT_SOURCE_REVISION = "38f7310ad594100747ca2a8357e9c7ca7d323e0e"
MOMENT_MODEL_ID = "AutonLab/MOMENT-1-small"
MOMENT_MODEL_REVISION = "411e288267f82cce86296dbe4d6c8bc533cc162f"
MANTIS_MODEL_ID = "paris-noah/MantisV2"
MANTIS_MODEL_VERSION = "v2"


def _sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _signature(config):
    raw = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


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


def _git_revision(path):
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"{path} is not a readable Git checkout") from exc


def _validate_moment_source(path, expected_revision):
    path = Path(path).expanduser().resolve()
    source = path / "momentfm" / "models" / "moment.py"
    if not source.is_file():
        raise FileNotFoundError(f"{source} not found; clone the official MOMENT repository")
    actual = _git_revision(path)
    if actual != expected_revision:
        raise ValueError(
            f"MOMENT source revision mismatch: expected {expected_revision}, got {actual}"
        )
    return path, source


def _archive_local_sources(output):
    destination = Path(output).parent / "source"
    destination.mkdir(parents=True, exist_ok=True)
    sources = (
        Path(__file__).resolve(), Path(moment_eval.__file__).resolve(),
        Path(kronos_eval.__file__).resolve(), Path(ssl_probe.__file__).resolve(),
        Path(ssl_data.__file__).resolve(),
    )
    archived = {}
    for source in sources:
        target = destination / source.name
        tmp = target.with_suffix(target.suffix + ".tmp")
        shutil.copyfile(source, tmp)
        os.replace(tmp, target)
        archived[source.name] = {"path": str(target.resolve()), "sha256": _sha256(target)}
    return archived


def _seed_everything(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_moment(repo, model_id, model_revision, device):
    sys.path.insert(0, str(repo))
    try:
        from momentfm import MOMENTPipeline
    finally:
        sys.path.pop(0)
    model = MOMENTPipeline.from_pretrained(
        model_id, revision=model_revision, model_kwargs={"task_name": "embedding"},
    )
    model.init()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.to(device).eval()


def _load_moment_encoder_checkpoint(model, checkpoint):
    """Load every deployable MOMENT encoder/embedder tensor, excluding task-only heads."""
    import torch
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state.get("schema_version") == "ffm_moment_staged_bundle_v1":
        state = state["model_state"]
    current = model.state_dict()
    transferable = {
        key: value for key, value in state.items()
        if key in current and current[key].shape == value.shape and not key.startswith("head.")
    }
    required = [key for key in current if (
        key.startswith("encoder.") or key.startswith("patch_embedding.")
    )]
    missing = sorted(set(required) - set(transferable))
    if missing:
        raise ValueError(f"MOMENT checkpoint misses {len(missing)} encoder/embedder tensors")
    result = model.load_state_dict(transferable, strict=False)
    unexpected = [key for key in result.unexpected_keys if not key.startswith("head.")]
    if unexpected:
        raise ValueError(f"unexpected MOMENT checkpoint tensors: {unexpected[:5]}")
    return len(transferable)


def _moment_embeddings(model, contexts, *, cache_dir, signature, batch_size, device,
                       native_length=512):
    import torch
    cache_dir = Path(cache_dir) / "moment" / signature
    cache_dir.mkdir(parents=True, exist_ok=True)
    pieces = []
    patch_len = int(model.patch_len)
    with torch.no_grad():
        for lo in range(0, len(contexts), batch_size):
            hi = min(len(contexts), lo + batch_size)
            shard = cache_dir / f"batch_{lo:07d}_{hi:07d}.npz"
            if shard.is_file():
                with np.load(shard, allow_pickle=False) as saved:
                    if str(saved["signature"].item()) != signature:
                        raise ValueError(f"cache signature mismatch: {shard}")
                    pooled = saved["embedding"]
                print(f"[moment] cache {lo}:{hi}", flush=True)
            else:
                padded, mask = left_pad_contexts(contexts[lo:hi], native_length)
                x = torch.as_tensor(padded, device=device)
                input_mask = torch.as_tensor(mask, device=device)
                outputs = model.embed(x_enc=x, input_mask=input_mask, reduction="none")
                patches = outputs.embeddings.float().cpu().numpy()
                pooled = pool_channel_patches(patches, mask, patch_len=patch_len)
                if not np.isfinite(pooled).all():
                    raise ValueError(f"MOMENT returned non-finite embeddings for {lo}:{hi}")
                _atomic_npz(shard, signature=np.array(signature), embedding=pooled)
                print(f"[moment] generated {lo}:{hi}", flush=True)
            if pooled.ndim != 2 or len(pooled) != hi - lo:
                raise ValueError(f"invalid MOMENT cache shape in {shard}: {pooled.shape}")
            pieces.append(np.asarray(pooled, np.float32))
    return np.concatenate(pieces, axis=0)


def _mantis_embeddings(contexts, *, cache_dir, signature, checkpoint, model_id,
                       model_version, preprocessing, batch_size, device):
    cache = Path(cache_dir) / "mantis" / f"{signature}.npz"
    if cache.is_file():
        with np.load(cache, allow_pickle=False) as saved:
            if str(saved["signature"].item()) != signature:
                raise ValueError(f"cache signature mismatch: {cache}")
            embedding = saved["embedding"]
        if embedding.ndim != 2 or len(embedding) != len(contexts):
            raise ValueError(f"invalid Mantis cache shape: {embedding.shape}")
        print(f"[mantis] cache {checkpoint or 'vanilla'}", flush=True)
        return embedding
    from futures_foundation.finetune.pretext._torch.common import embed_windows
    windows = np.transpose(np.asarray(contexts, np.float32), (0, 2, 1))
    embedding = embed_windows(
        windows, ckpt=checkpoint, model_id=model_id, model_version=model_version,
        device=device, batch=batch_size, preprocessing=preprocessing,
    )
    if not np.isfinite(embedding).all():
        raise ValueError("Mantis returned non-finite embeddings")
    _atomic_npz(cache, signature=np.array(signature), embedding=embedding)
    print(f"[mantis] generated {checkpoint or 'vanilla'}", flush=True)
    return embedding


def _probe_splits(windows, folds):
    streams = np.char.add(np.char.add(windows["ticker"], "@"), windows["timeframe"])
    _, groups = np.unique(streams, return_inverse=True)
    # Source offsets are stream-local; calendar timestamps create common fold boundaries so a
    # synchronized market event cannot be train data at 1m and test data at 60m.
    times = windows["context_time_ns"][:, 0]
    durations = [
        (windows["context_length"] + windows["horizon"]) * pd.Timedelta(str(tf)).value
        for tf in np.unique(windows["timeframe"])
    ]
    span_ns = int(max(durations))
    splits = ssl_probe.walk_forward_splits(
        windows["source_start"], groups, folds=folds,
        span=windows["context_length"] + windows["horizon"],
        timestamps=times, span_ns=span_ns,
    )
    return groups, span_ns, splits


def benchmark(args):
    moment_admission = verify_admission_report(
        args.moment_admission_report, arm_key="moment_small", track="C",
        route="historical_custom_representation_extraction", require_training=False,
    )
    mantis_admission = verify_admission_report(
        args.mantis_admission_report, arm_key="mantis_v2", track="C",
        route="historical_custom_representation_extraction", require_training=False,
    )
    repo, moment_source = _validate_moment_source(args.moment_repo, args.source_revision)
    checkpoint = Path(args.stage2_checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Stage-2 checkpoint not found: {checkpoint}")
    separation = args.context + args.horizon
    windows = build_forecast_windows(
        args.data_dir, args.tickers, args.timeframes, context=args.context,
        horizon=args.horizon, eval_start=args.eval_start, eval_end=args.eval_end,
        max_per_stream=args.max_per_stream, separation_bars=separation,
        seed=args.seed, chunksize=args.csv_chunksize,
    )
    fingerprint = window_fingerprint(windows)
    groups, span_ns, splits = _probe_splits(windows, args.folds)
    targets = targets_from_context_future(windows["context"], windows["future"])
    common = {
        "schema_version": "ffm_frozen_representation_benchmark_v1",
        "window_fingerprint": fingerprint,
        "context": args.context, "horizon": args.horizon,
        "eval_start": windows["eval_start"], "eval_end": windows["eval_end"],
        "seed": args.seed,
    }
    moment_config = {
        **common, "candidate": "moment", "source_revision": args.source_revision,
        "source_sha256": _sha256(moment_source), "model_id": args.moment_model_id,
        "model_revision": args.moment_model_revision, "native_length": 512,
        "input": "raw_ohlcv", "padding": "left_with_input_mask",
        "pooling": "valid_patch_mean_per_channel_then_ohlcv_concat",
    }
    moment_signature = _signature(moment_config)
    _seed_everything(args.seed)
    model = _load_moment(
        repo, args.moment_model_id, args.moment_model_revision, args.device,
    )
    moment_embedding = _moment_embeddings(
        model, windows["context"], cache_dir=args.cache_dir,
        signature=moment_signature, batch_size=args.moment_batch_size,
        device=args.device,
    )
    del model
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    continued_embedding = None
    continued_config = None
    if args.moment_checkpoint:
        moment_checkpoint = Path(args.moment_checkpoint).expanduser().resolve()
        if not moment_checkpoint.is_file():
            raise FileNotFoundError(f"continued MOMENT checkpoint missing: {moment_checkpoint}")
        continued_config = {
            **common, "candidate": "moment_continued_futures", "source_revision": args.source_revision,
            "model_id": args.moment_model_id, "model_revision": args.moment_model_revision,
            "checkpoint_sha256": _sha256(moment_checkpoint), "native_length": 512,
            "input": "raw_ohlcv", "pooling": moment_config["pooling"],
        }
        _seed_everything(args.seed)
        model = _load_moment(
            repo, args.moment_model_id, args.moment_model_revision, args.device,
        )
        transferred = _load_moment_encoder_checkpoint(model, moment_checkpoint)
        continued_config["transferred_tensors"] = transferred
        continued_embedding = _moment_embeddings(
            model, windows["context"], cache_dir=args.cache_dir,
            signature=_signature(continued_config), batch_size=args.moment_batch_size,
            device=args.device,
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    vanilla_config = {
        **common, "candidate": "mantis_v2_vanilla", "model_id": args.mantis_model_id,
        "model_version": args.mantis_model_version, "checkpoint": None,
        "preprocessing": args.mantis_preprocessing,
    }
    vanilla_embedding = _mantis_embeddings(
        windows["context"], cache_dir=args.cache_dir,
        signature=_signature(vanilla_config), checkpoint=None,
        model_id=args.mantis_model_id, model_version=args.mantis_model_version,
        preprocessing=args.mantis_preprocessing, batch_size=args.mantis_batch_size,
        device=args.device,
    )
    stage2_config = {
        **common, "candidate": "mantis_v2_stage2", "model_id": args.mantis_model_id,
        "model_version": args.mantis_model_version,
        "checkpoint_sha256": _sha256(checkpoint),
        "preprocessing": args.mantis_preprocessing,
    }
    stage2_embedding = _mantis_embeddings(
        windows["context"], cache_dir=args.cache_dir,
        signature=_signature(stage2_config), checkpoint=checkpoint,
        model_id=args.mantis_model_id, model_version=args.mantis_model_version,
        preprocessing=args.mantis_preprocessing, batch_size=args.mantis_batch_size,
        device=args.device,
    )
    moment_comparison = ssl_probe.compare(
        moment_embedding, vanilla_embedding, targets, seed=args.seed, splits=splits,
    )
    continued_comparison = (ssl_probe.compare(
        continued_embedding, vanilla_embedding, targets, seed=args.seed, splits=splits,
    ) if continued_embedding is not None else None)
    stage2_comparison = ssl_probe.compare(
        stage2_embedding, vanilla_embedding, targets, seed=args.seed, splits=splits,
    )

    versions = {}
    for package in (
        "torch", "numpy", "pandas", "scikit-learn", "transformers",
        "huggingface_hub", "safetensors", "mantis-tsfm",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    manifest = Path(args.data_dir) / "MANIFEST.json"
    report = {
        "schema_version": "ffm_frozen_representation_benchmark_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "policy": (
            "development representation probe only; no finetuning and no 2025-07+ holdout; "
            "MOMENT pretraining overlap with this futures corpus is not independently proven"
        ),
        "admission": {
            "moment_small": {
                "integrity": moment_admission["integrity"],
                "registry_sha256": moment_admission["registry_sha256"],
                "dossier_sha256": moment_admission["dossier_sha256"],
            },
            "mantis_v2": {
                "integrity": mantis_admission["integrity"],
                "registry_sha256": mantis_admission["registry_sha256"],
                "dossier_sha256": mantis_admission["dossier_sha256"],
            },
        },
        "config": {
            "moment": moment_config, "mantis_vanilla": vanilla_config,
            "mantis_stage2": stage2_config, "moment_continued": continued_config,
            "folds": args.folds,
        },
        "data": {
            "data_dir": str(Path(args.data_dir).resolve()),
            "corpus_manifest_sha256": _sha256(manifest) if manifest.is_file() else None,
            "windows": int(len(windows["context"])), "per_stream": windows["counts"],
            "tickers": list(args.tickers), "timeframes": list(args.timeframes),
            "separation_bars": separation,
            "overlap_policy": "complete context+future parent windows do not overlap per stream",
            "calendar_span_ns": span_ns, "groups": int(len(np.unique(groups))),
        },
        "probe": {
            "protocol": "five-fold expanding calendar walk-forward with two-span embargo",
            "targets": list(targets), "head": "standardized linear Ridge/logistic",
            "fold_sizes": [
                {"train": int(len(train)), "test": int(len(test))}
                for train, test in splits
            ],
        },
        "embedding_dimensions": {
            "moment": int(moment_embedding.shape[1]),
            "moment_continued": (int(continued_embedding.shape[1])
                                 if continued_embedding is not None else None),
            "mantis_v2_vanilla": int(vanilla_embedding.shape[1]),
            "mantis_v2_stage2": int(stage2_embedding.shape[1]),
        },
        "results": {
            "moment_vs_mantis_v2_vanilla": moment_comparison,
            "moment_continued_vs_mantis_v2_vanilla": continued_comparison,
            "mantis_v2_stage2_vs_vanilla": stage2_comparison,
        },
        "upstream": {
            "moment_repository": "https://github.com/moment-timeseries-foundation-model/moment",
            "checkout": str(repo), "source_revision": args.source_revision,
            "source_sha256": _sha256(moment_source),
        },
        "local_sources": _archive_local_sources(args.output),
        "environment": versions,
    }
    _atomic_json(args.output, report)
    print(json.dumps(report["results"], indent=2), flush=True)
    print(f"[moment] report -> {args.output}", flush=True)
    return report


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/ssl_corpus_v2_6tf")
    parser.add_argument("--moment-repo", required=True)
    parser.add_argument("--moment-checkpoint")
    parser.add_argument("--stage2-checkpoint", default=(
        "output/checkpoints/"
        "mantis_v2_stage2_contrastive_train_pre2024_val_2024_2025h1_6tf.pt"
    ))
    parser.add_argument("--output", default="output/moment/shared_probe/report.json")
    parser.add_argument("--cache-dir", default="output/moment/cache")
    parser.add_argument("--tickers", default="ES,NQ,RTY,YM,GC,SI,CL,ZB,ZN")
    parser.add_argument("--timeframes", default="1min,3min,5min,15min,30min,60min")
    parser.add_argument("--eval-start", default="2024-07-01")
    parser.add_argument("--eval-end", default="2025-07-01")
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--max-per-stream", type=int, default=200)
    parser.add_argument("--csv-chunksize", type=int, default=250000)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--moment-batch-size", type=int, default=16)
    parser.add_argument("--mantis-batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--source-revision", default=MOMENT_SOURCE_REVISION)
    parser.add_argument("--moment-model-id", default=MOMENT_MODEL_ID)
    parser.add_argument("--moment-model-revision", default=MOMENT_MODEL_REVISION)
    parser.add_argument("--mantis-model-id", default=MANTIS_MODEL_ID)
    parser.add_argument("--mantis-model-version", default=MANTIS_MODEL_VERSION)
    parser.add_argument(
        "--mantis-preprocessing", default="per_window_per_channel_zscore_v1",
    )
    parser.add_argument("--moment-admission-report", required=True)
    parser.add_argument("--mantis-admission-report", required=True)
    return parser


def main():
    args = _parser().parse_args()
    args.tickers = tuple(x.strip() for x in args.tickers.split(",") if x.strip())
    args.timeframes = tuple(x.strip() for x in args.timeframes.split(",") if x.strip())
    benchmark(args)


if __name__ == "__main__":
    main()
