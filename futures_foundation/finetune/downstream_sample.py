"""Sealed, balanced development sample and purged calendar folds.

The dense context collection is intentionally much denser at short timeframes.  Using every row
would make a pooled score mostly a one-minute score.  This module selects the same number of
chronologically distributed rows from every symbol/timeframe stream and binds every selected row
back to its verified source shard.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from futures_foundation.finetune.event_contexts import load_context_shard


SCHEMA_VERSION = "ffm_downstream_sample_v1"
FOLD_SCHEMA_VERSION = "ffm_purged_calendar_folds_v1"
SELECTION_SCHEMA_VERSION = "ffm_downstream_row_selection_v1"

_ROW_KEYS = (
    "ticker", "timeframe", "contract_id", "context_start_source_idx",
    "decision_source_idx", "decision_time_ns", "block_id", "features", "causal_scale",
    "context_direction", "cme_session_minute", "contract_segment_id",
    "bars_since_contract_start", "terminal_log_return", "terminal_move_r",
    "forward_abs_move_r", "forward_realized_vol", "upside_mfe_r", "downside_mae_r",
    "forward_trend_eff", "label_end_time_ns", "trend_path_class", "barrier_state",
    "time_to_favorable_minutes", "time_to_adverse_minutes", "policy_r_gross", "tags",
    "tag_direction", "tag_origin_source_idx", "tag_htf_agreement", "htf_direction",
)
_STATIC_KEYS = (
    "feature_names", "tag_names", "horizons_minutes", "targets_r", "directions",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _fingerprint(arrays: dict[str, np.ndarray], metadata: dict[str, object]) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode())
    for key in sorted(arrays):
        value = np.ascontiguousarray(arrays[key])
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def temporally_distributed_rows(row_count: int, sample_count: int) -> np.ndarray:
    """Return unique midpoint-quantile rows covering the complete ordered stream."""
    row_count, sample_count = int(row_count), int(sample_count)
    if row_count < 1 or sample_count < 1 or sample_count > row_count:
        raise ValueError("sample_count must be in [1, row_count]")
    selected = np.floor(
        (np.arange(sample_count, dtype=np.float64) + 0.5) * row_count / sample_count
    ).astype(np.int64)
    if len(np.unique(selected)) != sample_count or selected.min() < 0 or selected.max() >= row_count:
        raise RuntimeError("temporal sampling did not produce unique in-range rows")
    return selected


def _selected_block_weights(
    stream_index: np.ndarray,
    block_id: np.ndarray,
    *,
    equal_stream_mass: bool = False,
) -> np.ndarray:
    """Control dense blocks and optionally give every stream equal total fitting mass."""
    stream_index = np.asarray(stream_index, np.int32)
    block_id = np.asarray(block_id, np.int64)
    weights = np.empty(len(stream_index), dtype=np.float64)
    for stream in np.unique(stream_index):
        rows = np.flatnonzero(stream_index == stream)
        _, inverse, counts = np.unique(block_id[rows], return_inverse=True, return_counts=True)
        local = 1.0 / counts[inverse].astype(np.float64)
        weights[rows] = local / local.sum() if equal_stream_mass else local / local.mean()
    if equal_stream_mass:
        weights *= len(weights) / weights.sum()
    return weights.astype(np.float32)


def build_balanced_sample(
    collection_manifest: str | Path,
    *,
    rows_per_stream: int = 1200,
    event_tags: tuple[str, ...] = (),
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Load verified shards and build a deterministic dense or event-stratified sample.

    In event mode, ``rows_per_stream`` is a cap *per requested tag per stream*. Every rare event is
    retained while common tags are thinned by chronological midpoint quantiles. This avoids the
    old dense-row sampler silently discarding nearly all sparse pullback candidates.
    """
    collection_path = Path(collection_manifest).resolve()
    collection = json.loads(collection_path.read_text())
    if (
        collection.get("schema_version") != "ffm_event_context_collection_v1"
        or collection.get("status") != "complete"
        or collection.get("oos_read") is not False
    ):
        raise ValueError("balanced sample requires the completed development-only collection")
    rows_per_stream = int(rows_per_stream)
    if rows_per_stream < 1:
        raise ValueError("rows_per_stream must be positive")
    event_tags = tuple(dict.fromkeys(str(value) for value in event_tags))
    if any(not value for value in event_tags):
        raise ValueError("event tag names must be non-empty")

    chunks: dict[str, list[np.ndarray]] = {key: [] for key in _ROW_KEYS}
    stream_ids, stream_indices, shard_rows = [], [], []
    static: dict[str, np.ndarray] = {}
    source_shards: dict[str, dict[str, object]] = {}

    for stream_i, (stream_id, declared) in enumerate(sorted(collection["shards"].items())):
        path = Path(declared["path"])
        arrays, manifest = load_context_shard(path)
        if (
            manifest["artifact"]["sha256"] != declared["sha256"]
            or manifest["content_fingerprint"] != declared["content_fingerprint"]
        ):
            raise ValueError(f"collection/shard identity mismatch for {stream_id}")
        n = int(manifest["metadata"]["rows"])
        if not event_tags and n < rows_per_stream:
            raise ValueError(
                f"{stream_id} has {n} rows, fewer than rows_per_stream={rows_per_stream}"
            )
        if event_tags:
            names = [str(value) for value in arrays["tag_names"]]
            missing = sorted(set(event_tags) - set(names))
            if missing:
                raise ValueError(f"{stream_id} is missing requested event tags: {missing}")
            selected_parts = []
            selected_tag_counts = {}
            for name in event_tags:
                rows = np.flatnonzero(np.asarray(arrays["tags"])[:, names.index(name)])
                if not len(rows):
                    selected_tag_counts[name] = 0
                    continue
                if len(rows) > rows_per_stream:
                    rows = rows[temporally_distributed_rows(len(rows), rows_per_stream)]
                selected_tag_counts[name] = int(len(rows))
                selected_parts.append(rows)
            if not selected_parts:
                continue
            selected = np.unique(np.concatenate(selected_parts))
        else:
            selected_tag_counts = {}
            selected = temporally_distributed_rows(n, rows_per_stream)
        if not np.all(np.diff(np.asarray(arrays["decision_time_ns"])[selected]) > 0):
            raise ValueError(f"selected decision times are not increasing for {stream_id}")
        for key in _ROW_KEYS:
            value = np.asarray(arrays[key])
            if value.shape[0] != n:
                raise ValueError(f"{stream_id} row array {key} is misaligned")
            chunks[key].append(value[selected])
        for key in _STATIC_KEYS:
            value = np.asarray(arrays[key])
            if key in static and not np.array_equal(static[key], value):
                raise ValueError(f"static array {key} differs for {stream_id}")
            static[key] = value.copy()
        selected_count = len(selected)
        stream_ids.append(np.full(selected_count, stream_id))
        stream_indices.append(np.full(selected_count, stream_i, dtype=np.int16))
        shard_rows.append(selected)
        source_shards[stream_id] = {
            "path": str(path.resolve()),
            "sha256": declared["sha256"],
            "content_fingerprint": declared["content_fingerprint"],
            "source_rows": n,
            "selected_rows": selected_count,
            "selected_event_tag_counts": selected_tag_counts,
            "first_shard_row": int(selected[0]),
            "last_shard_row": int(selected[-1]),
        }

    if not source_shards:
        raise ValueError("no requested event appears in any verified stream")
    output = {key: np.concatenate(value, axis=0) for key, value in chunks.items()}
    output.update(static)
    output["stream_id"] = np.concatenate(stream_ids)
    output["stream_index"] = np.concatenate(stream_indices)
    output["shard_row"] = np.concatenate(shard_rows)
    output["sample_weight"] = _selected_block_weights(
        output["stream_index"], output["block_id"], equal_stream_mass=bool(event_tags),
    )
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "oos_read": False,
        "selection": {
            "method": ("event_tag_stratified_midpoint_quantiles" if event_tags
                       else "ordered_midpoint_quantiles"),
            "rows_per_stream": rows_per_stream,
            "event_tags": list(event_tags),
            "streams": len(source_shards),
            "rows": int(len(output["stream_id"])),
        },
        "source_collection": {
            "path": str(collection_path),
            "sha256": _sha256(collection_path),
            "schema_version": collection["schema_version"],
        },
        "source_shards": source_shards,
    }
    return output, metadata


def save_balanced_sample(
    path: str | Path,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, object],
) -> dict[str, object]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fingerprint = _fingerprint(arrays, metadata)
    temporary = Path(str(path) + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "oos_read": False,
        "content_fingerprint": fingerprint,
        "artifact": {
            "path": str(path.resolve()),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        },
        "metadata": metadata,
    }
    manifest_path = Path(str(path) + ".manifest.json")
    temporary_manifest = Path(str(manifest_path) + ".tmp")
    temporary_manifest.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    os.replace(temporary_manifest, manifest_path)
    return manifest


def load_balanced_sample(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    path = Path(path)
    manifest = json.loads(Path(str(path) + ".manifest.json").read_text())
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("status") != "complete"
        or manifest.get("oos_read") is not False
    ):
        raise ValueError("unsupported or incomplete downstream sample")
    if _sha256(path) != manifest.get("artifact", {}).get("sha256"):
        raise ValueError("downstream sample artifact hash mismatch")
    with np.load(path, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    if _fingerprint(arrays, manifest["metadata"]) != manifest["content_fingerprint"]:
        raise ValueError("downstream sample content fingerprint mismatch")
    return arrays, manifest


def build_balanced_row_selection(
    sample: dict[str, np.ndarray],
    sample_manifest: dict[str, object],
    *,
    rows_per_stream: int | None,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Select a balanced subset, or every row from an already event-focused sealed sample."""
    rows_per_stream = None if rows_per_stream is None else int(rows_per_stream)
    if rows_per_stream is not None and rows_per_stream < 1:
        raise ValueError("rows_per_stream must be positive or None")
    selected = []
    counts = {}
    for stream_id in sorted(str(value) for value in np.unique(sample["stream_id"])):
        stream_rows = np.flatnonzero(sample["stream_id"] == stream_id)
        rows = (stream_rows if rows_per_stream is None else
                stream_rows[temporally_distributed_rows(len(stream_rows), rows_per_stream)])
        if not np.all(np.diff(sample["decision_time_ns"][rows]) > 0):
            raise ValueError(f"row selection is not chronological for {stream_id}")
        selected.append(rows)
        counts[stream_id] = int(len(rows))
    row_index = np.sort(np.concatenate(selected)).astype(np.int32)
    arrays = {"row_index": row_index}
    metadata = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "status": "complete", "oos_read": False,
        "method": ("all_sample_rows" if rows_per_stream is None
                   else "nested_ordered_midpoint_quantiles"),
        "rows_per_stream": rows_per_stream,
        "rows": int(len(row_index)), "streams": len(counts),
        "stream_counts": counts,
        "sample": {
            "path": sample_manifest["artifact"]["path"],
            "sha256": sample_manifest["artifact"]["sha256"],
            "content_fingerprint": sample_manifest["content_fingerprint"],
        },
    }
    return arrays, metadata


def save_row_selection(
    path: str | Path,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, object],
) -> dict[str, object]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fingerprint = _fingerprint(arrays, metadata)
    temporary = Path(str(path) + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)
    manifest = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "status": "complete", "oos_read": False,
        "content_fingerprint": fingerprint,
        "artifact": {
            "path": str(path.resolve()), "sha256": _sha256(path),
            "bytes": int(path.stat().st_size),
        },
        "metadata": metadata,
    }
    manifest_path = Path(str(path) + ".manifest.json")
    temporary_manifest = Path(str(manifest_path) + ".tmp")
    temporary_manifest.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    os.replace(temporary_manifest, manifest_path)
    return manifest


def load_row_selection(
    path: str | Path,
    *,
    sample_manifest: dict[str, object] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    path = Path(path)
    manifest = json.loads(Path(str(path) + ".manifest.json").read_text())
    if (
        manifest.get("schema_version") != SELECTION_SCHEMA_VERSION
        or manifest.get("status") != "complete"
        or manifest.get("oos_read") is not False
    ):
        raise ValueError("unsupported or incomplete downstream row selection")
    if _sha256(path) != manifest.get("artifact", {}).get("sha256"):
        raise ValueError("downstream row-selection artifact hash mismatch")
    with np.load(path, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    if _fingerprint(arrays, manifest["metadata"]) != manifest["content_fingerprint"]:
        raise ValueError("downstream row-selection content fingerprint mismatch")
    rows = np.asarray(arrays.get("row_index"), np.int64)
    if rows.ndim != 1 or len(rows) == 0 or np.any(np.diff(rows) <= 0):
        raise ValueError("downstream row selection must be non-empty, unique, and increasing")
    if sample_manifest is not None:
        if manifest["metadata"]["sample"]["sha256"] != sample_manifest["artifact"]["sha256"]:
            raise ValueError("downstream row-selection/sample identity mismatch")
    return arrays, manifest


def purged_calendar_splits(
    decision_time_ns: np.ndarray,
    label_end_time_ns: np.ndarray,
    group_ids: np.ndarray,
    *,
    folds: int = 5,
    embargo_ns: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict[str, object]]:
    """Expanding calendar folds purged by each row's actual label end plus embargo.

    ``embargo_ns`` is cadence-specific.  For the canonical Gate-3 score it equals the timeframe's
    256-bar context duration.  This is deliberately conservative: a training label must finish one
    full deployment context before the next test block starts.
    """
    decision = np.asarray(decision_time_ns, np.int64)
    label_end = np.asarray(label_end_time_ns, np.int64)
    groups = np.asarray(group_ids)
    folds, embargo_ns = int(folds), int(embargo_ns)
    if label_end.ndim == 2:
        label_end = label_end.max(axis=1)
    if not (decision.ndim == label_end.ndim == groups.ndim == 1):
        raise ValueError("split arrays must be one-dimensional after label-end reduction")
    if not (len(decision) == len(label_end) == len(groups)) or len(decision) == 0:
        raise ValueError("split arrays must be non-empty and aligned")
    if folds < 1 or embargo_ns < 0 or np.any(label_end < decision):
        raise ValueError("invalid folds, embargo, or label-end ordering")

    unique_groups = np.unique(groups)
    # Score only the calendar support shared by every symbol.  Futures source coverage can begin
    # or end on different dates; global min/max edges can otherwise create an empty test fold for
    # one symbol while still looking well populated in aggregate.
    group_lows = [int(decision[groups == group].min()) for group in unique_groups]
    group_highs = [int(decision[groups == group].max()) + 1 for group in unique_groups]
    lo, hi = max(group_lows), min(group_highs)
    if hi <= lo:
        raise ValueError("groups have no shared calendar support")
    edges = np.linspace(lo, hi, folds + 2).astype(np.int64)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    fold_records = []
    for fold in range(1, folds + 1):
        test_lo, test_hi = int(edges[fold]), int(edges[fold + 1])
        train = np.flatnonzero(label_end + embargo_ns <= test_lo)
        test = np.flatnonzero((decision >= test_lo) & (decision < test_hi))
        for group in unique_groups:
            if not np.any(groups[train] == group) or not np.any(groups[test] == group):
                raise ValueError(f"group {group} fold {fold} is empty after purge")
        if np.intersect1d(train, test).size or np.any(label_end[train] + embargo_ns > test_lo):
            raise RuntimeError("purged calendar split contract was violated")
        splits.append((train, test))
        fold_records.append({
            "fold": fold,
            "test_start_ns": test_lo,
            "test_end_ns": test_hi,
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "train_last_label_end_ns": int(label_end[train].max()),
        })

    contract = {
        "schema_version": FOLD_SCHEMA_VERSION,
        "folds": folds,
        "embargo_ns": embargo_ns,
        "shared_start_ns": lo,
        "shared_end_ns": hi,
        "groups": int(len(unique_groups)),
        "rows": int(len(decision)),
        "records": fold_records,
    }
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    contract["contract_sha256"] = hashlib.sha256(encoded).hexdigest()
    return splits, contract


def purged_interval_splits(
    decision_time_ns: np.ndarray,
    label_end_time_ns: np.ndarray,
    group_ids: np.ndarray,
    *,
    eval_start_ns: int,
    eval_end_ns: int,
    folds: int = 5,
    embargo_ns: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict[str, object]]:
    """Expanding splits whose tests are confined to one declared evaluation interval."""
    decision = np.asarray(decision_time_ns, np.int64)
    label_end = np.asarray(label_end_time_ns, np.int64)
    groups = np.asarray(group_ids)
    if label_end.ndim == 2:
        label_end = label_end.max(axis=1)
    folds, embargo_ns = int(folds), int(embargo_ns)
    eval_start_ns, eval_end_ns = int(eval_start_ns), int(eval_end_ns)
    if (
        decision.ndim != 1 or label_end.ndim != 1 or groups.ndim != 1
        or not (len(decision) == len(label_end) == len(groups)) or len(decision) == 0
        or folds < 1 or embargo_ns < 0 or eval_end_ns <= eval_start_ns
        or np.any(label_end < decision)
    ):
        raise ValueError("invalid interval split contract")
    unique_groups = np.unique(groups)
    edges = np.linspace(eval_start_ns, eval_end_ns, folds + 1).astype(np.int64)
    splits, records = [], []
    for fold in range(folds):
        test_lo, test_hi = int(edges[fold]), int(edges[fold + 1])
        train = np.flatnonzero(label_end + embargo_ns <= test_lo)
        test = np.flatnonzero((decision >= test_lo) & (decision < test_hi))
        for group in unique_groups:
            if not np.any(groups[train] == group) or not np.any(groups[test] == group):
                raise ValueError(f"group {group} interval fold {fold + 1} is empty after purge")
        if np.intersect1d(train, test).size or np.any(label_end[train] + embargo_ns > test_lo):
            raise RuntimeError("purged interval split contract was violated")
        splits.append((train, test))
        records.append({
            "fold": fold + 1, "test_start_ns": test_lo, "test_end_ns": test_hi,
            "train_rows": int(len(train)), "test_rows": int(len(test)),
            "train_last_label_end_ns": int(label_end[train].max()),
        })
    contract = {
        "schema_version": "ffm_purged_interval_folds_v1", "folds": folds,
        "embargo_ns": embargo_ns, "eval_start_ns": eval_start_ns,
        "eval_end_ns": eval_end_ns, "groups": int(len(unique_groups)),
        "rows": int(len(decision)), "records": records,
    }
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    contract["contract_sha256"] = hashlib.sha256(encoded).hexdigest()
    return splits, contract


__all__ = [
    "SCHEMA_VERSION", "FOLD_SCHEMA_VERSION", "temporally_distributed_rows",
    "build_balanced_sample", "save_balanced_sample", "load_balanced_sample",
    "build_balanced_row_selection", "save_row_selection", "load_row_selection",
    "purged_calendar_splits", "purged_interval_splits",
]
