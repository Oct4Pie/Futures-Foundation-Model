from __future__ import annotations

from dataclasses import replace
import hashlib
import runpy
from pathlib import Path

import numpy as np
import pytest

from futures_foundation._authority_bundle_io import canonical_json_bytes, content_sha256
from futures_foundation.corpus_v3_materialization_plan import (
    build_materialization_plan_v1,
    build_split_scoped_inventory_v1,
)
from futures_foundation.corpus_v3_request_authority import (
    CorpusV3RequestAuthorityError,
    load_and_verify_request_authority_v1,
    load_request_authority_manifest_v1,
    request_segment_ids_v1,
    require_request_authority_v1,
)


ROOT = Path(__file__).resolve().parents[1]


def _artifacts(tmp_path: Path):
    namespace = runpy.run_path(str(ROOT / "tests/test_corpus_v3_materialization_plan.py"))
    expected = namespace["_expected"]()
    # The shared materialization fixture uses tiny synthetic UTC-ns values whose
    # pretrain/shared intervals overlap despite representing years-apart sessions.
    # Normalize that test-only geometry so request membership remains unambiguous.
    expected["request_shards"][0]["requests"][0]["request_end_exclusive_utc_ns"] = 170
    expected["request_shards"][0]["request_shard_semantic_sha256"] = content_sha256(
        expected["request_shards"][0], "request_shard_semantic_sha256"
    )
    expected["expected_request_denominator_sha256"] = content_sha256(
        expected, "expected_request_denominator_sha256"
    )
    rows = namespace["_rows"](expected)
    inventory = build_split_scoped_inventory_v1(
        expected_request_denominator=expected,
        inventory_rows=rows,
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
    capability = load_and_verify_request_authority_v1(
        expected_path=paths["expected"],
        expected_physical_sha256=hashes["expected"],
        inventory_path=paths["inventory"],
        inventory_physical_sha256=hashes["inventory"],
        plan_path=paths["plan"],
        plan_physical_sha256=hashes["plan"],
    )
    return capability, paths, hashes, documents


def test_request_authority_reopens_chain_and_manifest(tmp_path):
    capability, _, _, _ = _artifacts(tmp_path)
    assert len(capability.requests) == 2
    assert capability.production_admitted is False
    assert capability.materialization_admitted is False
    assert capability.training_admitted is False
    assert require_request_authority_v1(capability) == capability
    assert load_request_authority_manifest_v1(capability.manifest()) == capability


def test_request_segment_ids_require_exact_root_contract_use_and_interval(tmp_path):
    capability, _, _, _ = _artifacts(tmp_path)
    requests = capability.requests
    first, second = requests
    timestamps = np.asarray([
        first.request_start_utc_ns,
        first.request_end_exclusive_utc_ns - 1,
        first.request_end_exclusive_utc_ns,
        second.request_start_utc_ns,
        second.request_end_exclusive_utc_ns - 1,
    ], dtype=np.int64)
    contracts = np.asarray([
        first.contract_id, first.contract_id, first.contract_id,
        second.contract_id, "WRONG",
    ])
    self_supervised = request_segment_ids_v1(
        capability,
        root=first.root,
        requested_use="self_supervised_training",
        timestamps_ns=timestamps,
        contract_ids=contracts,
    )
    assert self_supervised.tolist() == [0, 0, -1, -1, -1]
    supervised = request_segment_ids_v1(
        capability,
        root=first.root,
        requested_use="supervised_training",
        timestamps_ns=timestamps,
        contract_ids=contracts,
    )
    assert supervised.tolist() == [-1, -1, -1, 1, -1]
    assert np.all(request_segment_ids_v1(
        capability,
        root=first.root,
        requested_use="validation",
        timestamps_ns=timestamps,
        contract_ids=contracts,
    ) == -1)


def test_request_authority_rejects_hash_substitution_and_parent_mutation(tmp_path):
    capability, paths, hashes, documents = _artifacts(tmp_path)
    with pytest.raises(CorpusV3RequestAuthorityError, match="physical SHA-256"):
        load_and_verify_request_authority_v1(
            expected_path=paths["expected"], expected_physical_sha256="f" * 64,
            inventory_path=paths["inventory"], inventory_physical_sha256=hashes["inventory"],
            plan_path=paths["plan"], plan_physical_sha256=hashes["plan"],
        )
    altered = dict(documents["plan"])
    altered["selected_requests"] = []
    paths["plan"].write_bytes(canonical_json_bytes(altered))
    with pytest.raises(CorpusV3RequestAuthorityError):
        require_request_authority_v1(capability)


def test_forged_dataclass_and_overlapping_selected_requests_reject(tmp_path):
    capability, paths, _, documents = _artifacts(tmp_path)
    forged = replace(capability, requests=())
    with pytest.raises(CorpusV3RequestAuthorityError, match="changed before use"):
        require_request_authority_v1(forged)

    plan = dict(documents["plan"])
    duplicate = dict(plan["selected_requests"][0])
    duplicate["request_semantic_sha256"] = "e" * 64
    plan["selected_requests"] = [plan["selected_requests"][0], duplicate, *plan["selected_requests"][1:]]
    plan["counts"] = dict(plan["counts"])
    plan["counts"]["selected_requests"] += 1
    plan["counts"]["excluded_requests"] -= 1
    plan["plan_sha256"] = content_sha256(plan, "plan_sha256")
    paths["plan"].write_bytes(canonical_json_bytes(plan))
    with pytest.raises(Exception):
        load_and_verify_request_authority_v1(
            expected_path=paths["expected"],
            expected_physical_sha256=hashlib.sha256(paths["expected"].read_bytes()).hexdigest(),
            inventory_path=paths["inventory"],
            inventory_physical_sha256=hashlib.sha256(paths["inventory"].read_bytes()).hexdigest(),
            plan_path=paths["plan"],
            plan_physical_sha256=hashlib.sha256(paths["plan"].read_bytes()).hexdigest(),
        )
