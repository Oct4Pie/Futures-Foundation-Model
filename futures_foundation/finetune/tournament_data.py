"""Shared leak-safe sampling primitives for foundation-model adaptation workers."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import pandas as pd

from futures_foundation._authority_bundle_io import (
    AuthorityBundleIOError,
    canonical_absolute_path,
    canonical_json_bytes,
    read_canonical_json_file,
    require_sha256,
    sha256_regular_file,
)

from . import ssl, ssl_data
from .tournament_cache_authority import (
    array_content_sha256,
    load_tournament_source_authority,
    require_tournament_source_authority,
    source_stream_identity,
    transformation_receipt,
    verify_transformation_receipt,
)
from .tournament import OOS_START, TRAIN_START, VALIDATION_START


CACHE_MANIFEST = "TOURNAMENT_CACHE.json"
CACHE_SCHEMA_VERSION = "ffm_foundation_tournament_cache_v3"
_MAX_CACHE_MANIFEST_BYTES = 32 * 1024 * 1024


def _sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _atomic_npy(path, value):
    path = Path(path)
    tmp = Path(str(path) + ".tmp")
    with tmp.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
    os.replace(tmp, path)


def _absolute_path(value) -> Path:
    return Path(os.path.abspath(os.fspath(value)))


def _cache_manifest_path(cache_dir) -> Path:
    return _absolute_path(cache_dir) / CACHE_MANIFEST


def cache_manifest_sha256(cache_dir) -> str:
    """Return the physical receipt hash that must be recorded out-of-band."""
    path = _cache_manifest_path(cache_dir)
    try:
        _, document, physical = read_canonical_json_file(
            path,
            label="tournament cache manifest",
            max_bytes=_MAX_CACHE_MANIFEST_BYTES,
            max_nodes=500_000,
            max_depth=24,
        )
    except AuthorityBundleIOError as exc:
        raise ValueError(str(exc)) from exc
    if document.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError("unsupported tournament cache schema")
    return physical


def build_cache(
    source_dir,
    cache_dir,
    tickers,
    timeframes,
    *,
    source_authority_path,
    source_authority_sha256,
    verbose=True,
):
    """Materialize one authority-bound, OOS-free mmap cache.

    The returned dictionary is not itself an authority.  Call
    :func:`cache_manifest_sha256` and bind that physical hash into the next
    consumer's admitted input contract before loading the cache.
    """
    source_dir = _absolute_path(source_dir)
    cache_dir = _absolute_path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    authority = load_tournament_source_authority(
        _absolute_path(source_authority_path),
        expected_sha256=source_authority_sha256,
    )
    require_tournament_source_authority(authority)
    transform = transformation_receipt(
        tournament_data_path=Path(__file__).absolute(),
        ssl_data_path=Path(ssl_data.__file__).absolute(),
    )
    entries = {}
    for ticker in tuple(str(value).strip().upper() for value in tickers):
        for timeframe in tuple(str(value).strip() for value in timeframes):
            source_identity = source_stream_identity(
                authority,
                source_dir=source_dir,
                ticker=ticker,
                timeframe=timeframe,
            )
            streams = ssl_data.load_ohlcv(
                source_dir, (ticker,), (timeframe,), verbose=verbose,
                start=TRAIN_START, end=OOS_START,
            )
            if len(streams) != 1:
                raise ValueError(f"authorized source stream did not load exactly once: {ticker}@{timeframe}")
            # Detect replacement or mutation across the actual CSV parse.
            if source_stream_identity(
                authority,
                source_dir=source_dir,
                ticker=ticker,
                timeframe=timeframe,
            ) != source_identity:
                raise ValueError(f"authorized source stream changed while loading: {ticker}@{timeframe}")
            stream = streams[0]
            stem = f"{ticker}_{timeframe}"
            paths = {
                "ohlcv": cache_dir / f"{stem}.ohlcv.npy",
                "timestamps": cache_dir / f"{stem}.timestamps.npy",
                "contract_id": cache_dir / f"{stem}.contract_id.npy",
            }
            timestamp_ns = pd.DatetimeIndex(stream["ts"]).asi8.astype(np.int64)
            contract_text = np.asarray(stream["contract_id"], dtype=str)
            _atomic_npy(paths["ohlcv"], np.asarray(stream["ohlcv"], np.float32))
            _atomic_npy(paths["timestamps"], timestamp_ns)
            _atomic_npy(paths["contract_id"], contract_text)
            if not len(timestamp_ns):
                raise ValueError(f"authorized source stream is empty in the locked interval: {ticker}@{timeframe}")
            entries[f"{ticker}@{timeframe}"] = {
                "ticker": ticker,
                "timeframe": timeframe,
                "bar_size": timeframe,
                "rows": int(len(timestamp_ns)),
                "loaded_interval": {
                    "start": TRAIN_START,
                    "end_exclusive": OOS_START,
                    "first_timestamp_utc": pd.Timestamp(timestamp_ns[0], tz="UTC").isoformat(),
                    "last_timestamp_utc": pd.Timestamp(timestamp_ns[-1], tz="UTC").isoformat(),
                },
                "source": source_identity,
                "contract_ids": sorted(np.unique(contract_text).tolist()),
                "contract_id_sequence_sha256": array_content_sha256(contract_text),
                "files": {
                    key: {
                        "path": path.name,
                        "sha256": _sha256(path),
                        "bytes": int(path.stat().st_size),
                        "content_sha256": array_content_sha256(
                            np.load(path, mmap_mode="r", allow_pickle=False)
                        ),
                    }
                    for key, path in paths.items()
                },
            }
    if not entries:
        raise ValueError("source authority produced no tournament cache entries")
    report = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "status": "complete",
        "training_admitted": False,
        "interval": {
            "start": TRAIN_START,
            "end_exclusive": OOS_START,
            "contains_oos": False,
        },
        "source_dir": str(source_dir),
        "source_authority": {
            "path": str(authority.path),
            "sha256": authority.physical_sha256,
            "semantic_sha256": authority.semantic_sha256,
        },
        "source_manifest": {
            "path": str(authority.manifest_path),
            "sha256": authority.manifest_physical_sha256,
            "bytes": authority.manifest_bytes,
            "schema_version": authority.manifest["schema_version"],
        },
        "transformation": transform,
        "entries": entries,
    }
    target = cache_dir / CACHE_MANIFEST
    tmp = Path(str(target) + ".tmp")
    tmp.write_bytes(canonical_json_bytes(report))
    os.replace(tmp, target)
    return report


def load_cache_manifest(cache_dir, *, expected_manifest_sha256):
    """Load a cache only when its physical manifest hash is supplied externally."""
    cache_dir = _absolute_path(cache_dir)
    manifest_path = cache_dir / CACHE_MANIFEST
    expected_manifest_sha256 = require_sha256(
        expected_manifest_sha256, "expected tournament cache manifest SHA-256",
    )
    try:
        _, manifest, physical = read_canonical_json_file(
            manifest_path,
            label="tournament cache manifest",
            max_bytes=_MAX_CACHE_MANIFEST_BYTES,
            max_nodes=500_000,
            max_depth=24,
        )
    except AuthorityBundleIOError as exc:
        raise ValueError(str(exc)) from exc
    if physical != expected_manifest_sha256:
        raise ValueError("tournament cache manifest physical SHA-256 mismatch")
    if set(manifest) != {
        "schema_version", "status", "training_admitted", "interval", "source_dir",
        "source_authority", "source_manifest", "transformation", "entries",
    }:
        raise ValueError("tournament cache manifest keys mismatch")
    if (
        manifest["schema_version"] != CACHE_SCHEMA_VERSION
        or manifest["status"] != "complete"
        or manifest["training_admitted"] is not False
    ):
        raise ValueError("unsupported or falsely authorizing tournament cache schema")
    interval = manifest.get("interval") or {}
    if set(interval) != {"start", "end_exclusive", "contains_oos"} or (
        interval.get("start"), interval.get("end_exclusive"), interval.get("contains_oos")
    ) != (TRAIN_START, OOS_START, False):
        raise ValueError("tournament cache interval does not match the locked protocol")
    source_dir = canonical_absolute_path(manifest["source_dir"], "tournament source directory")
    authority_identity = manifest.get("source_authority") or {}
    if set(authority_identity) != {"path", "sha256", "semantic_sha256"}:
        raise ValueError("tournament cache source-authority identity is malformed")
    authority = load_tournament_source_authority(
        authority_identity["path"], expected_sha256=authority_identity["sha256"],
    )
    if authority.semantic_sha256 != authority_identity["semantic_sha256"]:
        raise ValueError("tournament cache source-authority semantic identity mismatch")
    source_manifest = manifest.get("source_manifest") or {}
    expected_source_manifest = {
        "path": str(authority.manifest_path),
        "sha256": authority.manifest_physical_sha256,
        "bytes": authority.manifest_bytes,
        "schema_version": authority.manifest["schema_version"],
    }
    if source_manifest != expected_source_manifest or authority.manifest_path != source_dir / "MANIFEST.json":
        raise ValueError("tournament cache source-manifest identity mismatch")
    verify_transformation_receipt(
        manifest["transformation"],
        tournament_data_path=Path(__file__).absolute(),
        ssl_data_path=Path(ssl_data.__file__).absolute(),
    )
    entries = manifest.get("entries")
    if not isinstance(entries, dict) or not entries:
        raise ValueError("tournament cache has no entries")
    required_entry_keys = {
        "ticker", "timeframe", "bar_size", "rows", "loaded_interval", "source",
        "contract_ids", "contract_id_sequence_sha256", "files",
    }
    for stream_id, entry in entries.items():
        if not isinstance(entry, dict) or set(entry) != required_entry_keys:
            raise ValueError(f"tournament cache entry keys mismatch: {stream_id}")
        if stream_id != f"{entry['ticker']}@{entry['timeframe']}" or entry["bar_size"] != entry["timeframe"]:
            raise ValueError(f"tournament cache stream/bar-size identity mismatch: {stream_id}")
        if type(entry["rows"]) is not int or entry["rows"] < 1:
            raise ValueError(f"tournament cache row count is invalid: {stream_id}")
        loaded_interval = entry["loaded_interval"]
        if not isinstance(loaded_interval, dict) or set(loaded_interval) != {
            "start", "end_exclusive", "first_timestamp_utc", "last_timestamp_utc",
        } or (loaded_interval["start"], loaded_interval["end_exclusive"]) != (
            TRAIN_START, OOS_START,
        ):
            raise ValueError(f"tournament cache loaded interval is invalid: {stream_id}")
        require_sha256(
            entry["contract_id_sequence_sha256"],
            f"{stream_id} contract sequence SHA-256",
        )
        if (
            not isinstance(entry["contract_ids"], list)
            or not entry["contract_ids"]
            or sorted(set(entry["contract_ids"])) != entry["contract_ids"]
            or any(not isinstance(value, str) or not value.strip() for value in entry["contract_ids"])
        ):
            raise ValueError(f"tournament cache contract IDs are invalid: {stream_id}")
    return manifest


def load_cache_entry(cache_dir, manifest, ticker, timeframe):
    """Return one source- and cache-identity-verified mmap stream."""
    cache_dir = _absolute_path(cache_dir)
    ticker = str(ticker).strip().upper()
    timeframe = str(timeframe).strip()
    stream_id = f"{ticker}@{timeframe}"
    entry = (manifest.get("entries") or {}).get(stream_id)
    if entry is None:
        raise KeyError(f"tournament cache lacks {stream_id}")

    authority_identity = manifest["source_authority"]
    authority = load_tournament_source_authority(
        authority_identity["path"], expected_sha256=authority_identity["sha256"],
    )
    current_source = source_stream_identity(
        authority,
        source_dir=manifest["source_dir"],
        ticker=ticker,
        timeframe=timeframe,
    )
    if current_source != entry["source"]:
        raise ValueError(f"tournament cache source-file receipt mismatch for {stream_id}")

    files = entry.get("files") or {}
    if set(files) != {"ohlcv", "timestamps", "contract_id"}:
        raise ValueError(f"cache files are incomplete for {stream_id}")
    resolved = {}
    verified = {"source": current_source}
    for name, identity in files.items():
        if not isinstance(identity, dict) or set(identity) != {
            "path", "sha256", "bytes", "content_sha256",
        }:
            raise ValueError(f"cache file receipt is malformed for {stream_id}:{name}")
        relative = Path(str(identity["path"]))
        if relative.is_absolute() or relative.name != str(identity["path"]) or ".." in relative.parts:
            raise ValueError(f"cache file path is not a safe leaf for {stream_id}:{name}")
        file_path = cache_dir / relative
        require_sha256(identity["sha256"], f"{stream_id}:{name} file SHA-256")
        require_sha256(identity["content_sha256"], f"{stream_id}:{name} content SHA-256")
        if type(identity["bytes"]) is not int or identity["bytes"] <= 0:
            raise ValueError(f"cache file byte count is invalid for {stream_id}:{name}")
        try:
            verified_path, physical = sha256_regular_file(
                file_path,
                label=f"tournament cache file {stream_id}:{name}",
                expected_size=identity["bytes"],
            )
        except AuthorityBundleIOError as exc:
            raise ValueError(str(exc)) from exc
        if verified_path != file_path or physical != identity["sha256"]:
            raise ValueError(f"cache file identity mismatch for {stream_id}:{name}")
        resolved[name] = file_path
        verified[name] = {
            "path": str(file_path),
            "sha256": identity["sha256"],
            "bytes": int(identity["bytes"]),
            "content_sha256": identity["content_sha256"],
        }

    values = np.load(resolved["ohlcv"], mmap_mode="r", allow_pickle=False)
    timestamp_ns = np.load(resolved["timestamps"], mmap_mode="r", allow_pickle=False)
    contract = np.load(resolved["contract_id"], mmap_mode="r", allow_pickle=False)
    timestamp_values = np.asarray(timestamp_ns)
    contract_text = np.asarray(contract, dtype=str)
    rows = int(entry["rows"])
    if (
        values.dtype != np.float32
        or values.shape != (rows, 5)
        or timestamp_values.dtype != np.int64
        or timestamp_values.shape != (rows,)
        or contract.shape != (rows,)
        or contract.dtype.kind != "U"
        or not np.isfinite(values).all()
        or not np.all(np.diff(timestamp_values) > 0)
        or np.any(np.char.str_len(np.char.strip(contract_text)) == 0)
    ):
        raise ValueError(f"invalid cached array shape or dtype for {stream_id}")
    o, h, l, c, v = np.asarray(values).T
    if np.any((h < np.maximum(o, c)) | (l > np.minimum(o, c)) | (h < l) | (v < 0)):
        raise ValueError(f"invalid cached OHLCV geometry for {stream_id}")
    for name, value in (
        ("ohlcv", values), ("timestamps", timestamp_values), ("contract_id", contract_text),
    ):
        if array_content_sha256(value) != files[name]["content_sha256"]:
            raise ValueError(f"cache array content receipt mismatch for {stream_id}:{name}")
        # Reopen after mmap creation so path replacement during the load is detected.
        _, physical = sha256_regular_file(
            resolved[name],
            label=f"tournament cache file {stream_id}:{name}",
            expected_size=files[name]["bytes"],
        )
        if physical != files[name]["sha256"]:
            raise ValueError(f"cache file changed while loading for {stream_id}:{name}")
    loaded_interval = entry["loaded_interval"]
    if (
        pd.Timestamp(timestamp_values[0], tz="UTC").isoformat()
        != loaded_interval["first_timestamp_utc"]
        or pd.Timestamp(timestamp_values[-1], tz="UTC").isoformat()
        != loaded_interval["last_timestamp_utc"]
        or sorted(np.unique(contract_text).tolist()) != entry["contract_ids"]
        or array_content_sha256(contract_text) != entry["contract_id_sequence_sha256"]
    ):
        raise ValueError(f"cache interval or contract receipt mismatch for {stream_id}")
    stream = {
        "sid": stream_id,
        "ticker": ticker,
        "tf": timeframe,
        "ohlcv": values,
        "ts": timestamp_values.astype("datetime64[ns]"),
        "contract_id": contract,
    }
    return stream, verified


def load_cache(
    cache_dir, tickers, timeframes, *, expected_manifest_sha256, verbose=True,
):
    cache_dir = _absolute_path(cache_dir)
    manifest = load_cache_manifest(
        cache_dir, expected_manifest_sha256=expected_manifest_sha256,
    )
    streams = []
    for ticker in tickers:
        for timeframe in timeframes:
            if f"{ticker}@{timeframe}" not in (manifest.get("entries") or {}):
                if verbose:
                    print(f"  [tournament-cache] skip missing {ticker}@{timeframe}", flush=True)
                continue
            stream, _ = load_cache_entry(cache_dir, manifest, ticker, timeframe)
            streams.append(stream)
            if verbose:
                print(f"  [tournament-cache] {ticker}@{timeframe} bars={len(stream['ohlcv'])}", flush=True)
    if not streams:
        raise FileNotFoundError("no requested streams in tournament cache")
    return streams


def resolve_cache_manifest_sha256(value=None) -> str:
    """Resolve one external cache receipt, never from the cache being opened."""
    supplied = value if value is not None else os.environ.get(
        "FFM_TOURNAMENT_CACHE_MANIFEST_SHA256"
    )
    if supplied is None:
        raise ValueError(
            "cache_manifest_sha256 is required; pass it explicitly or set "
            "FFM_TOURNAMENT_CACHE_MANIFEST_SHA256"
        )
    return require_sha256(
        supplied, "expected tournament cache manifest SHA-256",
    )


def load_adaptation_data(
    data_dir,
    tickers,
    timeframes,
    *,
    parent_length,
    cache_manifest_sha256=None,
    session_gap_capabilities=None,
    verbose=True,
):
    """Load only an externally hash-bound tournament cache for adaptation."""
    data_dir = _absolute_path(data_dir)
    if not (data_dir / CACHE_MANIFEST).exists():
        raise ValueError("model adaptation requires an authority-bound tournament cache")
    cache_manifest_sha256 = resolve_cache_manifest_sha256(cache_manifest_sha256)
    streams = load_cache(
        data_dir,
        tickers,
        timeframes,
        expected_manifest_sha256=cache_manifest_sha256,
        verbose=verbose,
    )
    big, train, validation, groups = ssl.assemble(
        streams, seq=int(parent_length), max_jitter=0, val_frac=0.0,
        train_start=TRAIN_START, val_start=VALIDATION_START, holdout_start=OOS_START,
        return_groups=True, verbose=verbose, allow_aligned_market_gaps=False,
        session_gap_capabilities=session_gap_capabilities,
    )
    if len(groups["train_bounds"]) != len(streams) or len(groups["val_bounds"]) != len(streams):
        raise ValueError("every requested stream must have train and validation windows")
    groups["cache_manifest_sha256"] = cache_manifest_sha256
    groups["cache_schema_version"] = CACHE_SCHEMA_VERSION
    groups["contains_oos"] = False
    return streams, big, train, validation, groups


def balanced_schedule(starts, bounds, examples, seed):
    """Sample exactly ``examples`` anchors with uniform stream probability."""
    starts = np.asarray(starts, np.int64)
    bounds = np.asarray(bounds, np.int64)
    examples = int(examples)
    if examples < 1 or bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError("examples must be positive and bounds must have shape [G,2]")
    if np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("every stream must contain eligible anchors")
    rng = np.random.default_rng(int(seed))
    group = rng.integers(0, len(bounds), size=examples)
    draw = rng.random(examples)
    lo, hi = bounds[group, 0], bounds[group, 1]
    row = lo + np.floor(draw * (hi - lo)).astype(np.int64)
    return starts[row], group.astype(np.int64)


def schedule_fingerprint(starts, groups):
    h = hashlib.sha256()
    for value in (np.asarray(starts, np.int64), np.asarray(groups, np.int64)):
        h.update(str(value.shape).encode())
        h.update(np.ascontiguousarray(value).view(np.uint8))
    return h.hexdigest()


def gather_contexts(big, starts, context, *, parent_length=None):
    """Gather a causal suffix from common parent anchors as ``[B,C,T]``."""
    big = np.asarray(big)
    starts = np.asarray(starts, np.int64)
    context = int(context)
    parent_length = int(parent_length or context)
    if context < 1 or context > parent_length:
        raise ValueError("context must lie in [1,parent_length]")
    shifted = starts + parent_length - context
    rows = shifted[:, None] + np.arange(context, dtype=np.int64)[None, :]
    return np.transpose(big[rows], (0, 2, 1)).astype(np.float32, copy=False)


def gather_parent(big, starts, length):
    """Gather exact common parent rows as ``[B,T,C]``."""
    starts = np.asarray(starts, np.int64)
    rows = starts[:, None] + np.arange(int(length), dtype=np.int64)[None, :]
    return np.asarray(big)[rows].astype(np.float32, copy=False)


def gather_time_features(
    starts, group_ids, streams, row_bounds, length, *, timezone="UTC",
):
    """Kronos minute/hour/weekday/day/month features in one explicit timezone.

    Cache timestamps are UTC.  Kronos calendar stamps are derived only after an
    explicit timezone conversion so venue-local routes cannot depend on the host
    timezone or silently use UTC features.
    """
    starts = np.asarray(starts, np.int64)
    group_ids = np.asarray(group_ids, np.int64)
    row_bounds = np.asarray(row_bounds, np.int64)
    if len(starts) != len(group_ids):
        raise ValueError("starts and group_ids must align")
    try:
        timezone_name = str(timezone)
        ZoneInfo(timezone_name)
    except (TypeError, ValueError, ZoneInfoNotFoundError) as exc:
        raise ValueError("time-feature timezone must be a valid IANA timezone") from exc
    output = np.empty((len(starts), int(length), 5), np.float32)
    for group in np.unique(group_ids):
        rows = np.flatnonzero(group_ids == group)
        local = starts[rows] - int(row_bounds[group, 0])
        offsets = local[:, None] + np.arange(int(length), dtype=np.int64)[None, :]
        times = pd.DatetimeIndex(streams[int(group)]["ts"])
        if offsets.min() < 0 or offsets.max() >= len(times):
            raise ValueError(f"time-feature rows escape stream {group}")
        selected = (
            pd.DatetimeIndex(times.asi8[offsets.reshape(-1)])
            .tz_localize("UTC")
            .tz_convert(timezone_name)
        )
        values = np.stack([
            selected.minute, selected.hour, selected.weekday, selected.day, selected.month,
        ], axis=1).astype(np.float32)
        output[rows] = values.reshape(len(rows), int(length), 5)
    return output
