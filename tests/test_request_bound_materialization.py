from __future__ import annotations

import hashlib
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from futures_foundation._authority_bundle_io import canonical_json_bytes, content_sha256
from futures_foundation.corpus_v3_expected_requests import POLICY, SCHEMA_VERSION
from futures_foundation.corpus_v3_materialization_plan import (
    _expected_requests,
    build_materialization_plan_v1,
    build_split_scoped_inventory_v1,
)
from futures_foundation.corpus_v3_request_authority import (
    load_and_verify_request_authority_v1,
)
from futures_foundation.finetune import ssl
from futures_foundation.finetune.event_contexts import (
    load_context_shard,
    materialize_context_stream,
    save_context_shard,
    validate_context_shard,
)


ROOT = Path(__file__).resolve().parents[1]


def _request(
    *, partition: str, root: str, contract: str, use: str,
    session_day: str, start: int, end: int,
) -> dict:
    return {
        "candidate_index": 0,
        "provider_instrument_id": f"{contract.lower()}-fixture",
        "provider_symbol": contract,
        "contract_id": contract,
        "venue": "GLBX",
        "session_day": session_day,
        "session_segment_index": 0,
        "request_start_utc_ns": int(start),
        "request_end_exclusive_utc_ns": int(end),
    }


def _authority(tmp_path: Path, specs: list[dict]):
    shards = []
    for spec in sorted(specs, key=lambda row: (row["partition"], row["root"])):
        request = _request(
            partition=spec["partition"], root=spec["root"], contract=spec["contract"],
            use=spec["use"], session_day=spec["session_day"],
            start=spec["start"], end=spec["end"],
        )
        shard = {
            "partition_id": spec["partition"],
            "root": spec["root"],
            "permitted_uses": [spec["use"]],
            "parent_session_shard_semantic_sha256": "a" * 64,
            "session_disposition_count": 1,
            "request_count": 1,
            "requests": [request],
        }
        shard["request_shard_semantic_sha256"] = content_sha256(
            shard, "request_shard_semantic_sha256"
        )
        shards.append(shard)
    expected = {
        "schema_version": SCHEMA_VERSION,
        "policy": POLICY,
        "purpose": "expected_contract_session_requests_no_market_observation",
        "parent_capabilities": {},
        "split_assignment_basis": "exchange_session_day_only",
        "interval_semantics": "half_open_start_inclusive_end_exclusive",
        "roots": sorted({row["root"] for row in specs}),
        "partitions": sorted({row["partition"] for row in specs}),
        "counts": {
            "candidate_dispositions": len(specs),
            "session_dispositions": len(specs),
            "request_shards": len(shards),
            "expected_requests": len(specs),
        },
        "candidate_dispositions": [],
        "request_shards": shards,
        "data_access": {
            "market_namespace_opened": False,
            "market_files_enumerated": False,
            "market_content_read": False,
            "availability_or_liquidity_read": False,
            "materialization_plan_read": False,
            "labels_or_outcomes_read": False,
        },
        "production_admitted": False,
        "materialization_admitted": False,
        "training_admitted": False,
        "oos_admitted": False,
    }
    expected["expected_request_denominator_sha256"] = content_sha256(
        expected, "expected_request_denominator_sha256"
    )
    requests = _expected_requests(expected)
    inventory_rows = []
    for index, request in enumerate(requests):
        inventory_rows.append({
            "request_semantic_sha256": request["request_semantic_sha256"],
            "status": "available_exact",
            "reason": None,
            "source_files": [{
                "relative_path": f"{request['contract_id']}/{request['session_day']}/{index}.parquet",
                "interval_start_utc_ns": request["request_start_utc_ns"],
                "interval_end_exclusive_utc_ns": request["request_end_exclusive_utc_ns"],
                "size": 1,
                "sha256": f"{index + 1:x}" * 64,
            }],
        })
    inventory = build_split_scoped_inventory_v1(
        expected_request_denominator=expected,
        inventory_rows=inventory_rows,
    )
    plan = build_materialization_plan_v1(
        expected_request_denominator=expected,
        inventory=inventory,
    )
    paths = {
        "expected": tmp_path / "expected.json",
        "inventory": tmp_path / "inventory.json",
        "plan": tmp_path / "plan.json",
    }
    documents = {"expected": expected, "inventory": inventory, "plan": plan}
    hashes = {}
    for name, path in paths.items():
        path.write_bytes(canonical_json_bytes(documents[name]))
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return load_and_verify_request_authority_v1(
        expected_path=paths["expected"], expected_physical_sha256=hashes["expected"],
        inventory_path=paths["inventory"], inventory_physical_sha256=hashes["inventory"],
        plan_path=paths["plan"], plan_physical_sha256=hashes["plan"],
    )


def test_ssl_windows_are_bound_to_exact_train_and_validation_requests(tmp_path):
    timestamps = pd.date_range("2024-01-01", periods=100, freq="1min", tz="UTC")
    split = 60
    authority = _authority(tmp_path, [
        {
            "partition": "development", "root": "ES", "contract": "ESH4",
            "use": "validation", "session_day": "2024-01-02",
            "start": timestamps[split].value,
            "end": (timestamps[-1] + pd.Timedelta(minutes=1)).value,
        },
        {
            "partition": "shared_train", "root": "ES", "contract": "ESH4",
            "use": "self_supervised_training", "session_day": "2024-01-01",
            "start": timestamps[0].value, "end": timestamps[split].value,
        },
    ])
    stream = {
        "sid": "ES@1min", "ticker": "ES", "tf": "1min",
        "ts": timestamps.to_numpy(),
        "ohlcv": np.ones((len(timestamps), 5), dtype=np.float32),
        "contract_id": np.full(len(timestamps), "ESH4"),
    }
    _, train, validation, groups = ssl.assemble(
        [stream], seq=5, max_jitter=0, val_frac=0.1,
        train_start=timestamps[0], val_start=timestamps[split],
        holdout_start=timestamps[-1] + pd.Timedelta(minutes=1),
        request_authorities={"ES@1min": authority},
        return_groups=True, verbose=False,
    )
    assert len(train) and len(validation)
    assert np.all(train + 5 <= split)
    assert np.all(validation >= split)
    assert groups["request_authorities"] == (authority.manifest(),)
    assert groups["train_request_use"] == "self_supervised_training"
    assert groups["validation_request_use"] == "validation"
    with pytest.raises(ValueError, match="exactly cover"):
        ssl.assemble(
            [stream], seq=5, max_jitter=0, val_frac=0.1,
            train_start=timestamps[0], val_start=timestamps[split],
            holdout_start=timestamps[-1] + pd.Timedelta(minutes=1),
            request_authorities={}, verbose=False,
        )


def test_event_context_and_label_endpoints_remain_in_one_planned_request(tmp_path):
    namespace = runpy.run_path(str(ROOT / "tests/test_event_contexts.py"))
    frame = namespace["_frame"](n=1100)
    config = namespace["_config"](frame)
    economics = namespace["_economics"]()
    start_row, end_row = 200, 1000
    authority = _authority(tmp_path, [{
        "partition": "development", "root": "ES", "contract": "ESH4",
        "use": "validation", "session_day": "2024-01-01",
        "start": int(frame["datetime"].iloc[start_row].value),
        "end": int(frame["datetime"].iloc[end_row].value),
    }])
    arrays, metadata = materialize_context_stream(
        frame, ticker="ES", timeframe="1min", config=config,
        execution_economics=economics,
        request_authority=authority, requested_use="validation",
    )
    assert metadata["request_authority"] == authority.manifest()
    assert metadata["requested_use"] == "validation"
    assert np.all(arrays["request_segment_id"] == 0)
    assert np.all(arrays["decision_time_ns"] >= frame["datetime"].iloc[start_row].value)
    assert np.all(arrays["label_end_time_ns"] < frame["datetime"].iloc[end_row].value)
    output = tmp_path / "authorized-events.npz"
    save_context_shard(output, arrays, metadata, source={"fixture": True})
    loaded, manifest = load_context_shard(output)
    np.testing.assert_array_equal(loaded["request_segment_id"], arrays["request_segment_id"])
    assert manifest["metadata"]["request_authority"] == authority.manifest()

    tampered = dict(arrays)
    tampered["request_segment_id"] = arrays["request_segment_id"].copy()
    tampered["request_segment_id"][0] = 99
    with pytest.raises(ValueError, match="request membership"):
        validate_context_shard(tampered, metadata)
