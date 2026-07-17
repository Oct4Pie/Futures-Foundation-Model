"""Sealed policy-event materialization for the downstream trading gate."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import yaml

from futures_foundation.finetune.trend_strategy_eval import TICK_SIZES


POLICY_SCHEMA_VERSION = "ffm_downstream_policy_events_v2"
SUPPORTED_POLICY_SCHEMA_VERSIONS = {
    "ffm_downstream_policy_events_v1", POLICY_SCHEMA_VERSION,
}


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_policy_events(
    sample: dict[str, np.ndarray],
    selected_rows: np.ndarray,
    source_shards: dict,
    execution_costs: dict,
    *,
    source_cost_ticks: float = 1.0,
    slippage_ticks: float = 1.0,
) -> tuple[dict[str, np.ndarray], dict]:
    """Expand tick-costed source policy outcomes for the selected context rows."""
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
    if source_cost_ticks <= 0 or slippage_ticks < 0:
        raise ValueError("source cost ticks must be positive and slippage ticks nonnegative")
    for stream_id in sorted(str(value) for value in np.unique(sample["stream_id"][selected_rows])):
        global_rows = selected_rows[sample["stream_id"][selected_rows] == stream_id]
        source_info = source_shards.get(stream_id)
        if source_info is None:
            raise ValueError(f"source shard is absent for {stream_id}")
        source_path = Path(source_info["path"]).resolve()
        if _sha256(source_path) != source_info["sha256"]:
            raise ValueError(f"source shard hash mismatch for {stream_id}")
        with np.load(source_path, allow_pickle=False) as saved:
            source = {key: saved[key] for key in (
                "tag_names", "horizons_minutes", "targets_r", "policy_mode_names",
                "policy_event_context_row", "policy_event_tag_index",
                "policy_event_direction", "policy_valid", "policy_risk_ticks",
                "policy_cost_r", "policy_realized_r", "policy_reached",
                "policy_barrier_state", "policy_exit_time_ns",
            )}
        ticker_name = str(sample["ticker"][global_rows[0]])
        spec = execution_costs.get(ticker_name)
        if spec is None:
            raise ValueError(f"execution costs are absent for {ticker_name}")
        if not np.isclose(float(spec["tick_size"]), float(TICK_SIZES[ticker_name])):
            raise ValueError(f"tick-size mismatch for {ticker_name}")
        tick_value = float(spec["tick_value_usd"])
        fee_usd = float(spec["fee_rt_usd"])
        if tick_value <= 0 or fee_usd < 0:
            raise ValueError(f"invalid execution costs for {ticker_name}")
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
                    source_realized = np.asarray(source["policy_realized_r"])[
                        valid_events, mode_index, horizon_index, target_index
                    ]
                    exits = np.asarray(source["policy_exit_time_ns"])[
                        valid_events, mode_index, horizon_index, target_index
                    ]
                    source_one_tick_r = np.asarray(source["policy_cost_r"])[
                        valid_events, mode_index
                    ] / float(source_cost_ticks)
                    risk_ticks_all = np.asarray(source["policy_risk_ticks"])[
                        valid_events, mode_index
                    ]
                    gross = source_realized + source_one_tick_r * float(source_cost_ticks)
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
            "selected_contexts": int(len(global_rows)), "source_events": int(len(event_rows)),
        }
    if not columns["context_row"]:
        raise ValueError("no policy events match the selected contexts")
    arrays = {key: np.concatenate(values) for key, values in columns.items()}
    order = np.lexsort((arrays["policy_key"], arrays["signal_time_ns"]))
    arrays = {key: value[order] for key, value in arrays.items()}
    if np.any(arrays["exit_time_ns"] < arrays["signal_time_ns"]):
        raise ValueError("policy exit precedes its decision")
    metadata = {
        "schema_version": POLICY_SCHEMA_VERSION, "status": "complete", "oos_read": False,
        "rows": int(len(arrays["context_row"])),
        "contexts": int(len(np.unique(arrays["context_row"]))),
        "policies": int(len(np.unique(arrays["policy_key"]))),
        "source_shards": verified_shards,
        "cost_contract": "gross R minus declared round-trip slippage ticks and cash fees",
        "source_cost_ticks": float(source_cost_ticks),
        "slippage_ticks_round_trip": float(slippage_ticks),
        "entry_contract": (
            "signal confirmed at decision-bar close; enter next bar open; zero additional delay"
        ),
        "fee_schedule": execution_costs,
        "same_bar_policy": "adverse_first",
        "outcome_contract": (
            "barrier_state preserves favorable/adverse/neither/ambiguous; executable realized R "
            "maps ambiguous to adverse-first"
        ),
    }
    return arrays, metadata


def save_policy_events(path: str | Path, arrays: dict, metadata: dict) -> dict:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)
    manifest = dict(metadata)
    manifest["artifact"] = {"path": str(path), "sha256": _sha256(path)}
    manifest_path = Path(str(path) + ".manifest.json")
    temporary_manifest = Path(str(manifest_path) + ".tmp")
    temporary_manifest.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    os.replace(temporary_manifest, manifest_path)
    return manifest


def load_policy_events(path: str | Path) -> tuple[dict[str, np.ndarray], dict]:
    path = Path(path).resolve()
    manifest = json.loads(Path(str(path) + ".manifest.json").read_text())
    if (
        manifest.get("schema_version") not in SUPPORTED_POLICY_SCHEMA_VERSIONS
        or manifest.get("status") != "complete"
    ):
        raise ValueError("unsupported or incomplete policy artifact")
    if manifest.get("oos_read") is not False or manifest.get("artifact", {}).get("sha256") != _sha256(path):
        raise ValueError("policy artifact hash/OOS guard failed")
    with np.load(path, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    if len(arrays.get("context_row", ())) != manifest["rows"]:
        raise ValueError("policy artifact row count mismatch")
    return arrays, manifest


def load_execution_costs(path: str | Path, required_tickers=()) -> tuple[dict, dict]:
    path = Path(path).resolve()
    document = yaml.safe_load(path.read_text())
    if document.get("schema_version") != "ffm_execution_costs_v1":
        raise ValueError("unsupported execution-cost schema")
    instruments = document.get("instruments", {})
    missing = sorted(set(str(value) for value in required_tickers) - set(instruments))
    if missing:
        raise ValueError(f"execution costs missing instruments: {missing}")
    for ticker, spec in instruments.items():
        values = [float(spec[key]) for key in ("tick_size", "tick_value_usd", "fee_rt_usd")]
        if values[0] <= 0 or values[1] <= 0 or values[2] < 0:
            raise ValueError(f"invalid execution cost values for {ticker}")
    identity = {
        "path": str(path), "sha256": _sha256(path),
        "schema_version": document["schema_version"], "source": document.get("source"),
    }
    return instruments, {**identity, "document": document}


__all__ = [
    "POLICY_SCHEMA_VERSION", "build_policy_events", "save_policy_events",
    "load_policy_events", "load_execution_costs",
]
