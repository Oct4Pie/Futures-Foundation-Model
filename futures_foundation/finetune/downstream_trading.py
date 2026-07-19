"""Sealed policy-event materialization for the downstream trading gate."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from futures_foundation.execution_economics import (
    ExecutionEconomics, require_execution_economics,
)
from futures_foundation.finetune.event_contexts import load_context_shard


POLICY_SCHEMA_VERSION = "ffm_downstream_policy_events_v4"
LEGACY_POLICY_SCHEMA_VERSIONS = {
    "ffm_downstream_policy_events_v1", "ffm_downstream_policy_events_v2",
    "ffm_downstream_policy_events_v3",
}


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _content_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _policy_fingerprint(arrays: dict[str, np.ndarray], metadata: dict) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(
        metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8"))
    for key in sorted(arrays):
        value = np.ascontiguousarray(arrays[key])
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(np.asarray(value.shape, np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _load_bound_source_shard(stream_id: str, source_info: dict) -> tuple[dict, dict]:
    """Load a context shard only through its hash-bound manifest and sample identity."""
    source_path = Path(str(source_info.get("path", ""))).resolve()
    manifest_path = Path(str(source_path) + ".manifest.json")
    declared_manifest_path = Path(str(source_info.get("manifest_path", ""))).resolve()
    if declared_manifest_path != manifest_path.resolve():
        raise ValueError(f"source manifest path mismatch for {stream_id}")
    if not manifest_path.is_file() or _sha256(manifest_path) != source_info.get("manifest_sha256"):
        raise ValueError(f"source manifest hash mismatch for {stream_id}")
    arrays, manifest = load_context_shard(source_path)
    if (
        manifest.get("artifact", {}).get("sha256") != source_info.get("sha256")
        or manifest.get("content_fingerprint") != source_info.get("content_fingerprint")
    ):
        raise ValueError(f"source shard/sample identity mismatch for {stream_id}")
    metadata = manifest.get("metadata")
    config = metadata.get("config") if isinstance(metadata, dict) else None
    if not isinstance(config, dict):
        raise ValueError(f"source manifest lacks a bound config for {stream_id}")
    identity = {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": source_info["manifest_sha256"],
        "source_config_sha256": _content_sha256(config),
        "source_execution_economics": metadata.get("execution_economics"),
    }
    return arrays, identity


def build_policy_events(
    sample: dict[str, np.ndarray],
    selected_rows: np.ndarray,
    source_shards: dict,
    execution_economics: ExecutionEconomics,
    *,
    slippage_ticks: float,
) -> tuple[dict[str, np.ndarray], dict]:
    """Apply one economics capability to gross source outcomes for selected contexts."""
    selected_rows = np.asarray(selected_rows, np.int64)
    if selected_rows.ndim != 1 or not len(selected_rows) or len(np.unique(selected_rows)) != len(selected_rows):
        raise ValueError("selected rows must be a non-empty unique vector")
    if np.any(selected_rows < 0) or np.any(selected_rows >= len(sample["stream_id"])):
        raise ValueError("selected row is outside the sample")

    columns: dict[str, list[np.ndarray]] = {key: [] for key in (
        "context_row", "source_event_row", "tag_index", "direction", "mode_index",
        "horizon_index", "target_index", "signal_time_ns", "exit_time_ns",
        "gross_r", "realized_r", "reached", "barrier_state", "risk_ticks", "slippage_r", "fee_r",
        "total_cost_r", "ticker", "timeframe",
        "tag", "mode", "policy_key",
    )}
    verified_shards = {}
    sample_tags = [str(value) for value in sample["tag_names"]]
    execution_economics = require_execution_economics(execution_economics)
    slippage_ticks = execution_economics.validate_added_slippage(slippage_ticks)
    selected_times = np.asarray(sample["decision_time_ns"], np.int64)[selected_rows]
    economics_start_ns = int(execution_economics.evaluation_start_utc.timestamp() * 1e9)
    economics_end_ns = int(execution_economics.evaluation_end_exclusive_utc.timestamp() * 1e9)
    if selected_times.min() < economics_start_ns or selected_times.max() >= economics_end_ns:
        raise ValueError("execution-economics capability does not cover selected policy rows")
    for stream_id in sorted(str(value) for value in np.unique(sample["stream_id"][selected_rows])):
        global_rows = selected_rows[sample["stream_id"][selected_rows] == stream_id]
        source_info = source_shards.get(stream_id)
        if source_info is None:
            raise ValueError(f"source shard is absent for {stream_id}")
        source_path = Path(source_info["path"]).resolve()
        loaded, source_identity = _load_bound_source_shard(stream_id, source_info)
        required_source_keys = {
            "tag_names", "horizons_minutes", "targets_r", "policy_mode_names",
            "policy_event_context_row", "policy_event_tag_index", "policy_event_direction",
            "policy_valid", "policy_risk_price", "policy_risk_ticks", "policy_gross_r",
            "policy_reached", "policy_barrier_state", "policy_exit_time_ns",
        }
        missing_source_keys = sorted(required_source_keys - set(loaded))
        if missing_source_keys:
            raise ValueError(f"source shard is missing policy arrays for {stream_id}: {missing_source_keys}")
        source = {key: loaded[key] for key in required_source_keys}
        ticker_name = str(sample["ticker"][global_rows[0]])
        spec = execution_economics.instrument(ticker_name)
        source_economics = source_identity.get("source_execution_economics")
        source_spec = (
            source_economics.get("instruments", {}).get(ticker_name)
            if isinstance(source_economics, dict) else None
        )
        if (
            not isinstance(source_spec, dict)
            or not np.isclose(
                float(source_spec.get("tick_size", np.nan)), spec.tick_size,
                rtol=0.0, atol=1e-12,
            )
        ):
            raise ValueError(f"source/current tick geometry mismatch for {stream_id}")
        tick_value = spec.tick_value_usd
        fee_usd = spec.fee_rt_usd
        if [str(value) for value in source["tag_names"]] != sample_tags:
            raise ValueError(f"tag contract mismatch for {stream_id}")
        shard_to_global = {
            int(local): int(global_row)
            for local, global_row in zip(sample["shard_row"][global_rows], global_rows)
        }
        source_context = np.asarray(source["policy_event_context_row"], np.int64)
        event_rows = np.flatnonzero(np.isin(source_context, np.fromiter(
            shard_to_global, dtype=np.int64, count=len(shard_to_global),
        )))
        modes = [str(value) for value in source["policy_mode_names"]]
        horizons = np.asarray(source["horizons_minutes"], np.int32)
        targets = np.asarray(source["targets_r"], np.float32)
        for mode_index, mode in enumerate(modes):
            valid_events = event_rows[np.asarray(source["policy_valid"])[event_rows, mode_index]]
            for horizon_index, horizon in enumerate(horizons):
                for target_index, target in enumerate(targets):
                    gross = np.asarray(source["policy_gross_r"])[
                        valid_events, mode_index, horizon_index, target_index
                    ]
                    exits = np.asarray(source["policy_exit_time_ns"])[
                        valid_events, mode_index, horizon_index, target_index
                    ]
                    risk_ticks_all = np.asarray(source["policy_risk_ticks"])[
                        valid_events, mode_index
                    ]
                    risk_price_all = np.asarray(source["policy_risk_price"])[
                        valid_events, mode_index
                    ]
                    expected_risk_ticks = risk_price_all / spec.tick_size
                    if not np.allclose(
                        risk_ticks_all, expected_risk_ticks, rtol=1e-6, atol=1e-6,
                    ):
                        raise ValueError(f"source risk/tick geometry mismatch for {stream_id}")
                    source_one_tick_r = np.divide(
                        1.0, risk_ticks_all,
                        out=np.full(risk_ticks_all.shape, np.nan, dtype=np.float64),
                        where=risk_ticks_all > 0,
                    )
                    slippage_r = source_one_tick_r * float(slippage_ticks)
                    fee_r = fee_usd / (risk_ticks_all * tick_value)
                    realized = gross - slippage_r - fee_r
                    keep = (
                        np.isfinite(realized) & np.isfinite(gross) & np.isfinite(fee_r)
                        & (risk_ticks_all > 0) & (exits >= 0)
                    )
                    events = valid_events[keep]
                    if not len(events):
                        continue
                    context_rows = np.asarray([
                        shard_to_global[int(value)] for value in source_context[events]
                    ], np.int32)
                    tag_index = np.asarray(source["policy_event_tag_index"])[events].astype(np.int8)
                    tag = np.asarray([sample_tags[int(value)] for value in tag_index])
                    policy_key = np.asarray([
                        f"{name}__{mode}__{int(horizon)}m__{float(target):g}R"
                        for name in tag
                    ])
                    count = len(events)
                    columns["context_row"].append(context_rows)
                    columns["source_event_row"].append(events.astype(np.int32))
                    columns["tag_index"].append(tag_index)
                    columns["direction"].append(
                        np.asarray(source["policy_event_direction"])[events].astype(np.int8)
                    )
                    columns["mode_index"].append(np.full(count, mode_index, np.int8))
                    columns["horizon_index"].append(np.full(count, horizon_index, np.int8))
                    columns["target_index"].append(np.full(count, target_index, np.int8))
                    columns["signal_time_ns"].append(
                        sample["decision_time_ns"][context_rows].astype(np.int64)
                    )
                    columns["exit_time_ns"].append(exits[keep].astype(np.int64))
                    columns["gross_r"].append(gross[keep].astype(np.float32))
                    columns["realized_r"].append(realized[keep].astype(np.float32))
                    columns["reached"].append(
                        np.asarray(source["policy_reached"])[
                            events, mode_index, horizon_index, target_index
                        ].astype(bool)
                    )
                    columns["barrier_state"].append(
                        np.asarray(source["policy_barrier_state"])[
                            events, mode_index, horizon_index, target_index
                        ].astype(np.int8)
                    )
                    columns["risk_ticks"].append(risk_ticks_all[keep].astype(np.float32))
                    columns["slippage_r"].append(slippage_r[keep].astype(np.float32))
                    columns["fee_r"].append(fee_r[keep].astype(np.float32))
                    columns["total_cost_r"].append(
                        (slippage_r[keep] + fee_r[keep]).astype(np.float32)
                    )
                    columns["ticker"].append(sample["ticker"][context_rows])
                    columns["timeframe"].append(sample["timeframe"][context_rows])
                    columns["tag"].append(tag)
                    columns["mode"].append(np.full(count, mode))
                    columns["policy_key"].append(policy_key)
        verified_shards[stream_id] = {
            "path": str(source_path), "sha256": source_info["sha256"],
            "content_fingerprint": source_info["content_fingerprint"],
            **source_identity,
            "selected_contexts": int(len(global_rows)), "source_events": int(len(event_rows)),
        }
    if not columns["context_row"]:
        raise ValueError("no policy events match the selected contexts")
    arrays = {key: np.concatenate(values) for key, values in columns.items()}
    order = np.lexsort((arrays["policy_key"], arrays["signal_time_ns"]))
    arrays = {key: value[order] for key, value in arrays.items()}
    if np.any(arrays["exit_time_ns"] < arrays["signal_time_ns"]):
        raise ValueError("policy exit precedes its decision")
    if (
        arrays["signal_time_ns"].min() < economics_start_ns
        or arrays["exit_time_ns"].max() >= economics_end_ns
    ):
        raise ValueError("execution-economics capability does not cover full policy exposure")
    metadata = {
        "schema_version": POLICY_SCHEMA_VERSION, "status": "complete", "oos_read": False,
        "rows": int(len(arrays["context_row"])),
        "contexts": int(len(np.unique(arrays["context_row"]))),
        "policies": int(len(np.unique(arrays["policy_key"]))),
        "source_shards": verified_shards,
        "cost_contract": "gross R minus declared round-trip slippage ticks and cash fees",
        "slippage_ticks_round_trip": float(slippage_ticks),
        "entry_contract": (
            "signal confirmed at decision-bar close; enter next bar open; zero additional delay"
        ),
        "execution_economics": execution_economics.manifest(),
        "same_bar_policy": "adverse_first",
        "outcome_contract": (
            "barrier_state preserves favorable/adverse/neither/ambiguous; executable realized R "
            "maps ambiguous to adverse-first"
        ),
    }
    return arrays, metadata


def save_policy_events(path: str | Path, arrays: dict, metadata: dict) -> dict:
    if metadata.get("schema_version") != POLICY_SCHEMA_VERSION:
        raise ValueError("policy metadata must use the current schema")
    if "artifact" in metadata or "content_fingerprint" in metadata:
        raise ValueError("policy metadata contains reserved sealing fields")
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)
    manifest = dict(metadata)
    manifest["content_fingerprint"] = _policy_fingerprint(arrays, metadata)
    manifest["artifact"] = {"path": str(path), "sha256": _sha256(path)}
    manifest_path = Path(str(path) + ".manifest.json")
    temporary_manifest = Path(str(manifest_path) + ".tmp")
    temporary_manifest.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    os.replace(temporary_manifest, manifest_path)
    return manifest


def load_policy_events(
    path: str | Path, *, allow_legacy: bool = False,
) -> tuple[dict[str, np.ndarray], dict]:
    path = Path(path).resolve()
    manifest = json.loads(Path(str(path) + ".manifest.json").read_text())
    admitted = {POLICY_SCHEMA_VERSION} | (
        LEGACY_POLICY_SCHEMA_VERSIONS if allow_legacy else set()
    )
    if (
        manifest.get("schema_version") not in admitted
        or manifest.get("status") != "complete"
    ):
        raise ValueError("unsupported or incomplete policy artifact")
    if manifest.get("oos_read") is not False or manifest.get("artifact", {}).get("sha256") != _sha256(path):
        raise ValueError("policy artifact hash/OOS guard failed")
    with np.load(path, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    if manifest.get("schema_version") == POLICY_SCHEMA_VERSION:
        metadata = {
            key: value for key, value in manifest.items()
            if key not in {"artifact", "content_fingerprint"}
        }
        if _policy_fingerprint(arrays, metadata) != manifest.get("content_fingerprint"):
            raise ValueError("policy artifact content fingerprint mismatch")
    if len(arrays.get("context_row", ())) != manifest["rows"]:
        raise ValueError("policy artifact row count mismatch")
    return arrays, manifest


__all__ = [
    "POLICY_SCHEMA_VERSION", "build_policy_events", "save_policy_events",
    "load_policy_events",
]
