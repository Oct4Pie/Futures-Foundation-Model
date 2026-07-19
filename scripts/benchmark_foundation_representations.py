#!/usr/bin/env python3
"""Apples-to-apples frozen representation probes for staged foundation models.

Every checkpoint receives the same causal OHLCV contexts, targets, calendar walk-forward
folds, embargo, and linear probe.  Only backbone/native encoder states are exported: task
heads and contrastive projectors are deliberately excluded.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from futures_foundation.finetune import ssl_probe
from futures_foundation.finetune.foundation_roster import ARMS, get_arm
from futures_foundation.finetune.native_contracts import (
    add_admission_argument, require_admission_from_args,
)
from futures_foundation.finetune.kronos_eval import (
    build_forecast_windows, window_fingerprint,
)
from futures_foundation.finetune.moment_eval import targets_from_context_future
from futures_foundation.finetune.probe_targets import TARGET_SEMANTICS_VERSION


SCHEMA = "ffm_cross_family_representation_probe_v2"
CONTEXT = 256
HORIZON = 16
FOLDS = 5
EVAL_START = "2024-07-01"
EVAL_END = "2025-07-01"
WINDOW_SEED = 123
TICKERS = ("ES", "NQ", "RTY", "YM", "GC", "SI", "CL", "ZB", "ZN")
TIMEFRAMES = ("1min", "3min", "5min", "15min", "30min", "60min")
TRAINED_ARMS = (
    "kronos_mini", "kronos_small", "moment_small", "chronos_v1",
    "chronos_bolt", "chronos_v2", "ttm_r2", "moirai2_small", "timesfm25",
)
STAGE_NAMES = {
    "stage1": "stage1_reconstruction",
    "stage2": "stage2_contrastive",
    "stage3": "stage3_forecast",
}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _signature(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _atomic_npz(path, **values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp.npz")
    np.savez_compressed(temporary, **values)
    os.replace(temporary, path)


def _git_revision(path):
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True,
    ).strip()


def prepare_windows(args):
    """Seal one non-overlapping validation parent-window artifact."""
    output = Path(args.windows).resolve()
    manifest_path = Path(str(output) + ".manifest.json")
    if output.is_file() and manifest_path.is_file() and not args.overwrite:
        print(f"[windows] existing {output}", flush=True)
        return
    windows = build_forecast_windows(
        args.data_dir, TICKERS, TIMEFRAMES, context=CONTEXT, horizon=HORIZON,
        eval_start=EVAL_START, eval_end=EVAL_END, max_per_stream=args.max_per_stream,
        separation_bars=CONTEXT + HORIZON, seed=WINDOW_SEED,
        chunksize=args.csv_chunksize,
    )
    fingerprint = window_fingerprint(windows)
    values = {
        key: windows[key] for key in (
            "context", "future", "context_time_ns", "future_time_ns", "ticker",
            "timeframe", "source_start",
        )
    }
    _atomic_npz(output, **values)
    manifest = {
        "schema_version": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "artifact": {"path": str(output), "sha256": _sha256(output)},
        "window_fingerprint": fingerprint,
        "split": {"validation_start": EVAL_START, "oos_start": EVAL_END,
                  "oos_read": False},
        "shape": {"windows": int(len(windows["context"])), "context": CONTEXT,
                  "horizon": HORIZON, "channels": 5},
        "counts": windows["counts"],
        "config": {"data_dir": str(Path(args.data_dir).resolve()),
                   "tickers": list(TICKERS), "timeframes": list(TIMEFRAMES),
                   "max_per_stream": args.max_per_stream,
                   "separation_bars": CONTEXT + HORIZON, "seed": WINDOW_SEED},
        "overlap_policy": "complete context+future parent windows do not overlap per stream",
    }
    _atomic_json(manifest_path, manifest)
    print(f"[windows] {len(windows['context'])} -> {output}", flush=True)


def _load_windows(path):
    path = Path(path).resolve()
    manifest_path = Path(str(path) + ".manifest.json")
    if not path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"sealed representation windows missing: {path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != SCHEMA:
        raise ValueError("unsupported representation-window schema")
    if manifest["artifact"]["sha256"] != _sha256(path):
        raise ValueError("representation-window hash mismatch")
    if manifest["split"].get("oos_read") is not False:
        raise ValueError("representation windows are not validation-only")
    with np.load(path, allow_pickle=False) as saved:
        windows = {key: saved[key] for key in saved.files}
    if windows["context"].shape[1:] != (CONTEXT, 5):
        raise ValueError(f"invalid context shape: {windows['context'].shape}")
    if windows["future"].shape[1:] != (HORIZON, 5):
        raise ValueError(f"invalid future shape: {windows['future'].shape}")
    return windows, manifest


def _fold_contract(windows):
    streams = np.char.add(np.char.add(windows["ticker"], "@"), windows["timeframe"])
    stream_names, groups = np.unique(streams, return_inverse=True)
    times = windows["context_time_ns"][:, 0]
    span_ns = max(
        (CONTEXT + HORIZON) * pd.Timedelta(str(tf)).value
        for tf in np.unique(windows["timeframe"])
    )
    splits = ssl_probe.walk_forward_splits(
        windows["source_start"], groups, folds=FOLDS, span=CONTEXT + HORIZON,
        timestamps=times, span_ns=int(span_ns),
    )
    fold_rows = [{"train": tr.tolist(), "test": te.tolist()} for tr, te in splits]
    contract = {
        "stream_names": stream_names.tolist(), "groups": groups.tolist(),
        "time_ns": times.tolist(), "span_ns": int(span_ns), "folds": fold_rows,
    }
    return groups, splits, _signature(contract)


def _checkpoint_path(args, arm, stage):
    root = Path(args.checkpoint_root).resolve() / arm
    canonical = root / f"{stage}.pt"
    diagnostic = root / f"{stage}_diagnostic.pt"
    if arm in {"mantis_v1", "mantis_v2"} and not canonical.is_file() and diagnostic.is_file():
        return diagnostic
    return canonical


def _load_typed_bundle(path, schema, stage, identity_key=None, identity=None):
    import torch
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    if bundle.get("schema_version") != schema:
        raise ValueError(f"{path}: expected schema {schema}")
    if bundle.get("stage") != STAGE_NAMES[stage]:
        raise ValueError(f"{path}: expected {STAGE_NAMES[stage]}, got {bundle.get('stage')}")
    if identity_key and bundle.get(identity_key) != identity:
        raise ValueError(f"{path}: {identity_key} identity mismatch")
    return bundle


def _save_embedding(args, arm, stage, checkpoint, embedding, config, window_manifest):
    if embedding.ndim != 2 or not np.isfinite(embedding).all():
        raise ValueError(f"{arm}/{stage} returned invalid embeddings {embedding.shape}")
    output = Path(args.output_dir).resolve() / "embeddings" / arm / f"{stage}.npz"
    checkpoint_sha = _sha256(checkpoint) if checkpoint else None
    row_index = getattr(args, "row_index", None)
    selection_manifest = getattr(args, "row_selection_manifest", None)
    context_manifest = getattr(args, "context_manifest", None)
    if row_index is not None and (selection_manifest is None or context_manifest is None):
        raise ValueError("row-bound embeddings require selection and context manifests")
    admission = getattr(args, "native_admission", None)
    if not isinstance(admission, dict):
        raise ValueError("embedding extraction requires verified native admission metadata")
    metadata = {
        "schema_version": SCHEMA, "arm": arm, "stage": stage,
        "admission": {
            "integrity": admission["integrity"],
            "registry_sha256": admission["registry_sha256"],
            "dossier_sha256": admission["dossier_sha256"],
            "track": admission["track"],
            "route": admission["route"],
        },
        "checkpoint": str(Path(checkpoint).resolve()) if checkpoint else None,
        "checkpoint_sha256": checkpoint_sha,
        "window_fingerprint": window_manifest["window_fingerprint"],
        "windows_sha256": window_manifest["artifact"]["sha256"],
        "shape": list(embedding.shape), "config": config, "oos_read": False,
    }
    values = {"embedding": np.asarray(embedding, np.float32)}
    if row_index is not None:
        row_index = np.asarray(row_index, np.int32)
        if row_index.shape != (len(embedding),) or len(np.unique(row_index)) != len(row_index):
            raise ValueError("embedding row identity must be unique and aligned")
        metadata["row_selection"] = {
            "sha256": selection_manifest["artifact"]["sha256"],
            "content_fingerprint": selection_manifest["content_fingerprint"],
        }
        metadata["contexts"] = {
            "sha256": context_manifest["artifact"]["sha256"],
            "content_fingerprint": context_manifest["content_fingerprint"],
        }
        values["row_index"] = row_index
    signature = _signature(metadata)
    values.update(signature=np.array(signature), metadata=np.array(json.dumps(metadata)))
    _atomic_npz(output, **values)
    metadata["artifact"] = {"path": str(output), "sha256": _sha256(output)}
    _atomic_json(str(output) + ".manifest.json", metadata)
    print(f"[{arm}] {stage} {embedding.shape} -> {output}", flush=True)


def _batched(contexts, batch_size):
    for lo in range(0, len(contexts), batch_size):
        hi = min(len(contexts), lo + batch_size)
        yield lo, hi, contexts[lo:hi]


def _extract_kronos(args, arm, stages, windows, manifest):
    import torch
    from scripts.train_kronos_tournament import (
        _context_normalized_values, _load_models,
    )
    admitted = get_arm(arm)
    ns = SimpleNamespace(
        tokenizer_id=admitted.tokenizer_id,
        tokenizer_revision=admitted.tokenizer_revision,
        model_id=admitted.model_id,
        model_revision=admitted.model_revision,
    )
    repo = Path(args.kronos_repo).resolve()
    if _git_revision(repo) != admitted.source_revision:
        raise ValueError("Kronos source revision mismatch")
    tokenizer, predictor = _load_models(repo, ns)
    tokenizer, predictor = tokenizer.to(args.device).eval(), predictor.to(args.device).eval()

    def embed():
        pieces = []
        for lo, hi, context in _batched(windows["context"], args.kronos_batch):
            values = _context_normalized_values(context, args.kronos_clip)
            timestamp = pd.DatetimeIndex(
                windows["context_time_ns"][lo:hi].reshape(-1), tz="UTC",
            )
            stamps = np.stack((timestamp.minute, timestamp.hour, timestamp.weekday,
                               timestamp.day, timestamp.month), axis=1)
            stamps = stamps.astype(np.float32).reshape(hi - lo, CONTEXT, 5)
            with torch.inference_mode(), torch.autocast(
                    "cuda", dtype=torch.bfloat16,
                    enabled=args.device.startswith("cuda") and getattr(args, "amp", True)):
                first, second = tokenizer.encode(
                    torch.as_tensor(values, device=args.device), half=True,
                )
                _, hidden = predictor.decode_s1(
                    first, second, torch.as_tensor(stamps, device=args.device),
                )
                pieces.append(hidden.mean(dim=1).float().cpu().numpy())
            print(f"[{arm}] extract {lo}:{hi}", flush=True)
        return np.concatenate(pieces)

    for stage in stages:
        checkpoint = None if stage == "vanilla" else _checkpoint_path(args, arm, stage)
        if checkpoint:
            bundle = _load_typed_bundle(
                checkpoint, "ffm_kronos_tournament_bundle_v1", stage,
            )
            tokenizer.load_state_dict(bundle["tokenizer_state"], strict=True)
            predictor.load_state_dict(bundle["predictor_state"], strict=True)
        else:
            tokenizer, predictor = _load_models(repo, ns)
            tokenizer, predictor = tokenizer.to(args.device).eval(), predictor.to(args.device).eval()
        _save_embedding(args, arm, stage, checkpoint, embed(), {
            "pooling": "mean native predictor decode_s1 states",
            "input": "joint normalized OHLCVA plus calendar stamps",
            "projector": "excluded", "context": CONTEXT,
        }, manifest)


def _extract_chronos(args, arm, stages, windows, manifest):
    import torch
    from futures_foundation.finetune.chronos_family import CANDIDATES
    from scripts.train_chronos_contrastive import _encode
    from scripts.train_chronos_tournament import _load_pipeline
    candidate = CANDIDATES[arm]

    def fresh():
        return _load_pipeline(candidate, args.device)

    for stage in stages:
        pipeline, model = fresh()
        checkpoint = None if stage == "vanilla" else _checkpoint_path(args, arm, stage)
        if checkpoint:
            bundle = _load_typed_bundle(
                checkpoint, "ffm_chronos_tournament_bundle_v1", stage,
                "candidate", candidate.manifest(),
            )
            model.load_state_dict(bundle["model_state"], strict=True)
        model.eval(); pieces = []
        with torch.inference_mode():
            for lo, hi, context in _batched(windows["context"], args.chronos_batch):
                with torch.autocast(
                    "cuda", dtype=torch.bfloat16,
                    enabled=args.device.startswith("cuda") and getattr(args, "amp", True),
                ):
                    value = _encode(candidate, pipeline, model, context, args.device)
                pieces.append(value.float().cpu().numpy())
                print(f"[{arm}] {stage} {lo}:{hi}", flush=True)
        _save_embedding(args, arm, stage, checkpoint, np.concatenate(pieces), {
            "pooling": "valid native encoder token mean then OHLCV concatenate",
            "input": "channel-independent raw OHLCV", "projector": "excluded",
            "context": CONTEXT,
            "precision": ("bf16_autocast" if getattr(args, "amp", True) else "float32"),
        }, manifest)
        del pipeline, model; gc.collect(); torch.cuda.empty_cache()


def _extract_moment(args, arm, stages, windows, manifest):
    import torch
    from scripts.benchmark_moment import (
        _load_moment_encoder_checkpoint, _load_moment, _moment_embeddings,
    )
    admitted = get_arm(arm)
    repo = Path(args.moment_repo).resolve()
    if _git_revision(repo) != admitted.source_revision:
        raise ValueError("MOMENT source revision mismatch")
    for stage in stages:
        model = _load_moment(repo, admitted.model_id, admitted.model_revision, args.device)
        checkpoint = None if stage == "vanilla" else _checkpoint_path(args, "moment", stage)
        if checkpoint:
            bundle = _load_typed_bundle(
                checkpoint, "ffm_moment_staged_bundle_v1", stage,
            )
            _load_moment_encoder_checkpoint(model, checkpoint)
        config = {"pooling": "valid-patch mean per channel then OHLCV concatenate",
                  "input": "raw OHLCV left-padded to 512", "projector": "excluded",
                  "context": CONTEXT}
        sig = _signature({"arm": arm, "stage": stage,
                          "checkpoint": _sha256(checkpoint) if checkpoint else None,
                          "windows": manifest["window_fingerprint"], **config})
        embedding = _moment_embeddings(
            model, windows["context"], cache_dir=Path(args.output_dir) / "batch_cache",
            signature=sig, batch_size=args.moment_batch, device=args.device,
        )
        _save_embedding(args, arm, stage, checkpoint, embedding, config, manifest)
        del model; gc.collect(); torch.cuda.empty_cache()


def _extract_ttm(args, arm, stages, windows, manifest):
    import torch
    # Granite-TSFM's pinned source predates Transformers 5, which removed three utility
    # exports.  Training used the same narrow compatibility surface.  The URL downloader is
    # not exercised for pinned Hub IDs, but must exist for the upstream module to import.
    import tempfile
    import urllib.parse
    import urllib.request
    import transformers.utils as transformer_utils
    from transformers.utils.hub import is_offline_mode
    from transformers import PreTrainedModel
    if not hasattr(transformer_utils, "is_offline_mode"):
        transformer_utils.is_offline_mode = is_offline_mode
    if not hasattr(transformer_utils, "is_remote_url"):
        transformer_utils.is_remote_url = lambda value: (
            urllib.parse.urlparse(str(value)).scheme in {"http", "https"}
        )
    if not hasattr(transformer_utils, "download_url"):
        def _download_url(url, proxies=None):
            del proxies
            suffix = Path(urllib.parse.urlparse(str(url)).path).suffix
            handle, destination = tempfile.mkstemp(suffix=suffix)
            os.close(handle)
            urllib.request.urlretrieve(str(url), destination)
            return destination
        transformer_utils.download_url = _download_url
    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
        # Transformers 5 expects this attribute during load finalization; the pinned TTM
        # implementation predates it and does not declare tied parameters.
        PreTrainedModel.all_tied_weights_keys = {}
    repo = Path(args.ttm_repo).resolve()
    sys.path.insert(0, str(repo))
    try:
        from scripts.train_ttm_contrastive import _encode
        from scripts.train_ttm_tournament import (
            CONTEXT as TTM_CONTEXT,
            FREQUENCY_TOKEN,
            _load_model,
            _normalize_parent,
        )
        admitted = get_arm(arm)
        ns = SimpleNamespace(model_id=admitted.model_id, model_revision=admitted.model_revision)
        if windows["context"].shape[1] != TTM_CONTEXT:
            raise ValueError(
                "TTM native extraction requires 512 real causal bars; rebuild the shared "
                f"windows instead of padding {windows['context'].shape[1]} bars"
            )
        frequency = np.asarray([FREQUENCY_TOKEN[str(tf)] for tf in windows["timeframe"]])
        for stage in stages:
            model = _load_model(ns, source=repo).to(args.device).eval()
            checkpoint = None if stage == "vanilla" else _checkpoint_path(args, arm, stage)
            if checkpoint:
                bundle = _load_typed_bundle(
                    checkpoint, "ffm_ttm_staged_bundle_v1", stage,
                    "arm", admitted.manifest(),
                )
                model.load_state_dict(bundle["model_state"], strict=True)
            pieces = []
            with torch.inference_mode():
                for lo, hi, context in _batched(windows["context"], args.ttm_batch):
                    normalized = _normalize_parent(context, 0, TTM_CONTEXT)[0]
                    with torch.autocast(
                        "cuda", dtype=torch.bfloat16,
                        enabled=args.device.startswith("cuda") and getattr(args, "amp", True),
                    ):
                        value = _encode(
                            model, torch.as_tensor(normalized, device=args.device),
                            torch.as_tensor(frequency[lo:hi], device=args.device),
                        )
                    pieces.append(value.float().cpu().numpy())
                    print(f"[{arm}] {stage} {lo}:{hi}", flush=True)
            _save_embedding(args, arm, stage, checkpoint, np.concatenate(pieces), {
                "pooling": "custom backbone patch mean then OHLCV concatenate",
                "input": "512 real causal bars with official scaler and frequency prefix",
                "projector": "excluded", "context": TTM_CONTEXT,
                "track": "C", "native_forecast_required_first": True,
            }, manifest)
            del model; gc.collect(); torch.cuda.empty_cache()
    finally:
        sys.path.remove(str(repo))


def _extract_timesfm(args, arm, stages, windows, manifest):
    import torch
    from peft import set_peft_model_state_dict
    from scripts.train_timesfm_tournament import _load_model
    admitted = get_arm(arm)
    ns = SimpleNamespace(
        model_id=admitted.model_id, model_revision=admitted.model_revision,
        lora_rank=8, lora_alpha=16, lora_dropout=0.0, device=args.device,
    )
    for stage in stages:
        # Training uses bf16 for throughput, but TimesFM's bf16 hidden-state reductions
        # changed materially when the same rows were partitioned into different batches.
        # Frozen evaluation therefore promotes the loaded weights to float32.
        model = _load_model(ns).float().eval()
        checkpoint = None if stage == "vanilla" else _checkpoint_path(args, arm, stage)
        if checkpoint:
            bundle = _load_typed_bundle(
                checkpoint, "ffm_timesfm_staged_bundle_v1", stage,
                "arm", admitted.manifest(),
            )
            set_peft_model_state_dict(model, bundle["adapter_state"])
        pieces = []
        with torch.inference_mode():
            for lo, hi, context in _batched(windows["context"], args.timesfm_batch):
                series = context.transpose(0, 2, 1).reshape(-1, CONTEXT)
                output = model(
                    past_values=torch.as_tensor(series, device=args.device),
                    forecast_context_len=CONTEXT,
                ).last_hidden_state
                value = output.mean(dim=1).reshape(hi - lo, 5, -1).flatten(1)
                pieces.append(value.float().cpu().numpy())
                print(f"[{arm}] {stage} {lo}:{hi}", flush=True)
        _save_embedding(args, arm, stage, checkpoint, np.concatenate(pieces), {
            "pooling": "native last-hidden-state patch mean then OHLCV concatenate",
            "input": "channel-independent raw OHLCV", "projector": "excluded",
            "context": CONTEXT, "precision": "float32",
        }, manifest)
        del model; gc.collect(); torch.cuda.empty_cache()


def _extract_moirai(args, arm, stages, windows, manifest):
    import torch
    repo = Path(args.uni2ts_repo).resolve()
    source = str(repo / "src")
    sys.path.insert(0, source)
    try:
        from scripts.train_moirai2_contrastive import _encode
        from scripts.train_moirai2_tournament import _load_model
        admitted = get_arm(arm)
        ns = SimpleNamespace(model_id=admitted.model_id, model_revision=admitted.model_revision)
        for stage in stages:
            model = _load_model(ns, context_length=CONTEXT).to(args.device).eval()
            checkpoint = None if stage == "vanilla" else _checkpoint_path(args, arm, stage)
            if checkpoint:
                bundle = _load_typed_bundle(
                    checkpoint, "ffm_moirai2_staged_bundle_v1", stage,
                    "arm", admitted.manifest(),
                )
                model.load_state_dict(bundle["model_state"], strict=True)
            pieces = []
            with torch.inference_mode():
                for lo, hi, context in _batched(windows["context"], args.moirai_batch):
                    with torch.autocast(
                        "cuda", dtype=torch.bfloat16,
                        enabled=args.device.startswith("cuda") and getattr(args, "amp", True),
                    ):
                        value = _encode(model, torch.as_tensor(context, device=args.device))
                    pieces.append(value.float().cpu().numpy())
                    print(f"[{arm}] {stage} {lo}:{hi}", flush=True)
            _save_embedding(args, arm, stage, checkpoint, np.concatenate(pieces), {
                "pooling": "valid native packed transformer token mean",
                "input": "joint raw OHLCV", "projector": "excluded", "context": CONTEXT,
            }, manifest)
            del model; gc.collect(); torch.cuda.empty_cache()
    finally:
        sys.path.remove(source)


def _extract_mantis(args, arm, stages, windows, manifest):
    from futures_foundation.finetune.pretext._torch.common import embed_windows
    configs = {
        "mantis_v1": ("paris-noah/Mantis-8M", "v1"),
        "mantis_v2": ("paris-noah/MantisV2", "v2"),
    }
    model_id, version = configs[arm]
    for stage in stages:
        checkpoint = None if stage == "vanilla" else _checkpoint_path(args, arm, stage)
        if checkpoint and not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        embedding = embed_windows(
            windows["context"].transpose(0, 2, 1), ckpt=checkpoint,
            model_id=model_id, model_version=version, device=args.device,
            batch=args.mantis_batch, preprocessing="per_window_per_channel_zscore_v1",
        )
        _save_embedding(args, arm, stage, checkpoint, embedding, {
            "pooling": "canonical Mantis per-channel embedding concatenate",
            "input": "per-window per-channel z-scored OHLCV", "projector": "excluded",
            "context": CONTEXT,
        }, manifest)


def _extract_toto2(args, arm, stages, windows, manifest):
    import torch
    from toto2 import Toto2Model
    from scripts.train_control_foundation_stages import _normalize, _toto_encode
    admitted = get_arm(arm)
    for stage in stages:
        model = Toto2Model.from_pretrained(
            admitted.model_id, revision=admitted.model_revision, map_location="cpu",
        ).to(args.device).eval()
        checkpoint = None if stage == "vanilla" else _checkpoint_path(args, arm, stage)
        if checkpoint:
            bundle = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if bundle.get("schema_version") != "ffm_control_staged_bundle_v1":
                raise ValueError(f"unsupported Toto staged bundle: {checkpoint}")
            if bundle.get("stage") != STAGE_NAMES[stage] or bundle.get("arm") != admitted.manifest():
                raise ValueError(f"Toto checkpoint stage/identity mismatch: {checkpoint}")
            model.load_state_dict(bundle["backbone_state"], strict=True)
        pieces = []
        with torch.no_grad():
            for _, _, context in _batched(windows["context"], args.toto_batch):
                values = _normalize(context, CONTEXT)
                pieces.append(_toto_encode(
                    model, torch.as_tensor(values, device=args.device),
                ).cpu().numpy())
        _save_embedding(args, arm, stage, checkpoint, np.concatenate(pieces), {
            "pooling": "native joint variate/time transformer state mean",
            "input": "causal per-channel z-scored joint OHLCV", "projector": "excluded",
            "context": CONTEXT,
        }, manifest)
        del model; gc.collect(); torch.cuda.empty_cache()


EXTRACTORS = {
    "kronos_mini": _extract_kronos, "kronos_small": _extract_kronos,
    "moment_small": _extract_moment,
    "chronos_v1": _extract_chronos, "chronos_bolt": _extract_chronos,
    "chronos_v2": _extract_chronos, "ttm_r2": _extract_ttm,
    "timesfm25": _extract_timesfm, "moirai2_small": _extract_moirai,
    "mantis_v1": _extract_mantis, "mantis_v2": _extract_mantis,
    "toto2_22m": _extract_toto2,
}


def extract(args):
    args.native_admission = require_admission_from_args(
        args, arm_key=args.arm, track="C",
        route="historical_custom_representation_extraction",
        require_training=False,
    )
    windows, manifest = _load_windows(args.windows)
    stages = tuple(part.strip() for part in args.stages.split(",") if part.strip())
    unknown = set(stages) - {"vanilla", "stage1", "stage2", "stage3"}
    if unknown:
        raise ValueError(f"unknown stages: {sorted(unknown)}")
    EXTRACTORS[args.arm](args, args.arm, stages, windows, manifest)


def _embedding_artifacts(output_dir):
    root = Path(output_dir).resolve() / "embeddings"
    for path in sorted(root.glob("*/*.npz")):
        manifest_path = Path(str(path) + ".manifest.json")
        if manifest_path.is_file():
            yield path, json.loads(manifest_path.read_text())


def score(args):
    windows, window_manifest = _load_windows(args.windows)
    _, splits, fold_hash = _fold_contract(windows)
    targets = targets_from_context_future(windows["context"], windows["future"])
    rows = {}
    for path, artifact in _embedding_artifacts(args.output_dir):
        if artifact.get("window_fingerprint") != window_manifest["window_fingerprint"]:
            raise ValueError(f"window fingerprint mismatch: {path}")
        if artifact["artifact"]["sha256"] != _sha256(path):
            raise ValueError(f"embedding hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as saved:
            embedding = saved["embedding"]
        if len(embedding) != len(windows["context"]):
            raise ValueError(f"embedding row mismatch: {path}")
        metrics = {}
        for target, values in targets.items():
            folds = ssl_probe._probe_scores(
                embedding, values, ssl_probe._TARGET_KIND[target],
                seed=args.probe_seed, splits=splits,
            )
            metrics[target] = {
                "metric": "AUC" if ssl_probe._TARGET_KIND[target] == "bin" else "R2",
                "mean": float(np.mean(folds)), "folds": [float(value) for value in folds],
                "std": float(np.std(folds, ddof=1)),
            }
        key = f"{artifact['arm']}:{artifact['stage']}"
        rows[key] = {"arm": artifact["arm"], "stage": artifact["stage"],
                     "checkpoint": artifact.get("checkpoint"),
                     "checkpoint_sha256": artifact.get("checkpoint_sha256"),
                     "embedding": artifact["artifact"], "embedding_shape": list(embedding.shape),
                     "metrics": metrics}
        print(f"[score] {key}", flush=True)

    for row in rows.values():
        baseline = rows.get(f"{row['arm']}:vanilla")
        if baseline is None or row["stage"] == "vanilla":
            row["delta_vs_vanilla"] = None
            continue
        delta = {}
        for target in targets:
            current = row["metrics"][target]
            vanilla = baseline["metrics"][target]
            fold_delta = np.asarray(current["folds"]) - np.asarray(vanilla["folds"])
            delta[target] = {
                "mean": float(current["mean"] - vanilla["mean"]),
                "folds": fold_delta.tolist(),
                "consistent_positive_fraction": float(np.mean(fold_delta > 0)),
            }
        row["delta_vs_vanilla"] = delta

    expected = {}
    # Coverage follows the authoritative roster. Extractor availability is a separate
    # capability and must not silently remove a blocked or unsupported arm from reporting.
    for arm in ARMS:
        expected[arm] = {}
        for stage in ("stage1", "stage2", "stage3"):
            key = f"{arm}:{stage}"
            if key in rows:
                checkpoint = rows[key].get("checkpoint") or ""
                status = ("complete_diagnostic_non_promotable"
                          if "_diagnostic.pt" in checkpoint else "complete")
            elif arm == "sundial_base":
                status = "blocked_nonfinite_native_hidden_states"
            elif arm in {"tabpfn_ts3_forecast", "tabpfn_v3_downstream"}:
                status = "not_applicable_in_context_model"
            else:
                status = "missing_checkpoint_or_embedding"
            expected[arm][stage] = status

    report = {
        "schema_version": SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": _coverage_status(expected, rows),
        "oos_read": False,
        "windows": {"path": str(Path(args.windows).resolve()),
                    "sha256": window_manifest["artifact"]["sha256"],
                    "fingerprint": window_manifest["window_fingerprint"],
                    "rows": int(len(windows["context"])), "context": CONTEXT,
                    "horizon": HORIZON},
        "probe": {"protocol": "five-fold expanding calendar walk-forward with two-span embargo",
                  "fold_contract_sha256": fold_hash,
                  "head": "standardized Ridge(alpha=1,lsqr) / LogisticRegression(C=1)",
                  "targets": list(targets),
                  "target_semantics_version": TARGET_SEMANTICS_VERSION},
        "results": rows, "coverage": expected,
    }
    output, _ = _write_report_artifacts(args.output_dir, report)
    print(f"[score] report -> {output}", flush=True)


def _write_report_artifacts(output_dir, report):
    output = Path(output_dir).resolve() / "representation_results.json"
    _atomic_json(output, report)
    markdown = Path(output_dir).resolve() / "representation_results.md"
    temporary = Path(str(markdown) + ".tmp")
    temporary.write_text(_render_markdown(report))
    os.replace(temporary, markdown)
    return output, markdown


def render(args):
    """Render an already scored JSON artifact without recomputing linear probes."""
    _, window_manifest = _load_windows(args.windows)
    output = Path(args.output_dir).resolve() / "representation_results.json"
    if not output.is_file():
        raise FileNotFoundError(output)
    report = json.loads(output.read_text())
    if report.get("windows", {}).get("fingerprint") != window_manifest["window_fingerprint"]:
        raise ValueError("report/window fingerprint mismatch")
    if window_manifest["split"].get("oos_read") is not False:
        raise ValueError("cannot attest report as validation-only")
    report["oos_read"] = False
    report["status"] = _coverage_status(report["coverage"], report["results"])
    output, markdown = _write_report_artifacts(args.output_dir, report)
    print(f"[render] {output} and {markdown}", flush=True)


def _coverage_status(coverage, rows):
    if not rows:
        return "empty"
    if any(status == "missing_checkpoint_or_embedding"
           for arm in EXTRACTORS for status in coverage[arm].values()):
        return "incomplete"
    if any(status.startswith("blocked_")
           for stages in coverage.values() for status in stages.values()):
        return "complete_with_declared_blockers"
    if any(status.startswith("not_applicable_")
           for stages in coverage.values() for status in stages.values()):
        return "complete_with_declared_non_staged_controls"
    return "complete"


def _render_markdown(report):
    targets = ("vol", "trend_eff", "range_expand", "fwd_absmove", "direction", "fwd_dir")
    lines = [
        "# Foundation representation results", "",
        f"Status: `{report['status']}`. OOS read: `{str(report['oos_read']).lower()}`.", "",
        (f"Validation windows: {report['windows']['rows']:,}; context "
         f"{report['windows']['context']}; horizon {report['windows']['horizon']}."), "",
        "| Model | Stage | vol R² | trend_eff R² | range_expand R² | "
        "fwd_absmove R² | direction AUC | fwd_dir AUC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    stage_order = {"stage1": 1, "stage2": 2, "stage3": 3}
    rows = sorted(
        (row for row in report["results"].values() if row["stage"] != "vanilla"),
        key=lambda row: (row["arm"], stage_order[row["stage"]]),
    )
    for row in rows:
        values = [row["metrics"][target]["mean"] for target in targets]
        label = row["stage"].replace("stage", "S")
        if "_diagnostic.pt" in (row.get("checkpoint") or ""):
            label += " (diagnostic)"
        lines.append(
            f"| {row['arm']} | {label} | " + " | ".join(f"{value:.4f}" for value in values) + " |"
        )
    lines.extend(["", "## Coverage", "", "| Model | Stage 1 | Stage 2 | Stage 3 |",
                  "|---|---|---|---|"])
    for arm, stages in report["coverage"].items():
        lines.append(f"| {arm} | {stages['stage1']} | {stages['stage2']} | {stages['stage3']} |")
    lines.extend([
        "", "Diagnostic means training was continued solely to measure the stage; the failed "
        "parent was not promoted. Sundial is blocked because its pinned native hidden-state "
        "path produced non-finite values on real OHLCV smoke windows. TabPFN-TS is an in-context "
        "downstream model, not a staged trainable encoder.", "",
    ])
    return "\n".join(lines)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "extract", "score", "render"))
    parser.add_argument("--windows", default=(
        "output/foundation_tournament/representation_apples/windows.npz"
    ))
    parser.add_argument("--output-dir", default=(
        "output/foundation_tournament/representation_apples"
    ))
    parser.add_argument("--checkpoint-root", default=(
        "output/foundation_tournament/final_staged"
    ))
    parser.add_argument("--data-dir", default="data/ssl_corpus_v2_6tf")
    parser.add_argument("--arm", choices=tuple(EXTRACTORS))
    parser.add_argument("--stages", default="vanilla,stage1,stage2,stage3")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--probe-seed", type=int, default=5400)
    parser.add_argument("--max-per-stream", type=int, default=200)
    parser.add_argument("--csv-chunksize", type=int, default=250000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--kronos-repo", default="/tmp/ffm-kronos-inspect")
    parser.add_argument("--moment-repo", default="/tmp/ffm-moment-inspect")
    parser.add_argument("--ttm-repo", default="/tmp/ffm-granite-tsfm")
    parser.add_argument("--uni2ts-repo", default="/tmp/ffm-uni2ts")
    parser.add_argument("--kronos-batch", type=int, default=256)
    parser.add_argument("--chronos-batch", type=int, default=128)
    parser.add_argument("--moment-batch", type=int, default=16)
    parser.add_argument("--ttm-batch", type=int, default=256)
    parser.add_argument("--timesfm-batch", type=int, default=16)
    parser.add_argument("--moirai-batch", type=int, default=64)
    parser.add_argument("--mantis-batch", type=int, default=256)
    parser.add_argument("--toto-batch", type=int, default=256)
    parser.add_argument("--kronos-clip", type=float, default=3.0)
    add_admission_argument(parser)
    return parser


def main():
    args = _parser().parse_args()
    if args.command == "prepare":
        prepare_windows(args)
    elif args.command == "extract":
        if not args.arm:
            raise ValueError("extract requires --arm")
        extract(args)
    elif args.command == "score":
        score(args)
    else:
        render(args)


if __name__ == "__main__":
    main()
