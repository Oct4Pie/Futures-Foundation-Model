from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest

from futures_foundation._authority_bundle_io import canonical_json_bytes, content_sha256
from futures_foundation.corpus_v3_producer_governance import (
    PRODUCER_COMPATIBILITY_COMMIT,
    CorpusV3ProducerGovernanceError,
    evaluate_boundary_leaf_v1,
    evaluate_session_request_v1,
    load_and_verify_frozen_split_use_contract_v1,
    load_and_verify_producer_governance_v1,
    reopen_and_verify_frozen_split_use_contract_v1,
    reopen_and_verify_producer_governance_v1,
)


ROOT = Path(__file__).resolve().parents[1]
ALPHA = ROOT.parent / "alphaforge-corpus-v3-scale"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(path: Path, value: object) -> None:
    path.write_bytes(canonical_json_bytes(value))


def _rehash(value: dict, field: str) -> None:
    value[field] = content_sha256(value, field)


def _producer_fixture(tmp_path: Path) -> dict[str, str]:
    python = ALPHA / ".venv/bin/python"
    tests = ALPHA / "tests/test_producer_governance.py"
    if not python.is_file() or not tests.is_file():
        pytest.skip("pinned AlphaForge producer-governance worktree is unavailable")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ALPHA, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    assert revision == PRODUCER_COMPATIBILITY_COMMIT
    output = tmp_path / "producer-sidecar.json"
    script = r'''
import hashlib, json, runpy, sys
from pathlib import Path
namespace = runpy.run_path(sys.argv[1])
root = Path(sys.argv[2]).resolve()
root.mkdir(parents=True)
producer_path, split_path, producer, split = namespace["_fixture"](root)
result = {
    "producer_path": str(producer_path),
    "producer_sha256": hashlib.sha256(producer_path.read_bytes()).hexdigest(),
    "producer_semantic_sha256": producer["governance_semantic_sha256"],
    "split_path": str(split_path),
    "split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
    "split_semantic_sha256": split["contract_semantic_sha256"],
}
Path(sys.argv[3]).write_text(json.dumps(result, sort_keys=True))
'''
    subprocess.run(
        [str(python), "-c", script, str(tests), str(tmp_path / "producer"), str(output)],
        cwd=ALPHA, check=True, capture_output=True, text=True,
    )
    return json.loads(output.read_text(encoding="utf-8"))


def _load(fixture: dict[str, str]):
    producer = load_and_verify_producer_governance_v1(
        fixture["producer_path"], expected_sha256=fixture["producer_sha256"],
    )
    split = load_and_verify_frozen_split_use_contract_v1(
        fixture["split_path"], expected_sha256=fixture["split_sha256"],
        producer_governance_path=fixture["producer_path"],
        producer_governance_sha256=fixture["producer_sha256"],
    )
    return producer, split


@pytest.fixture()
def producer_fixture(tmp_path):
    return _producer_fixture(tmp_path)


def test_ffm_independently_accepts_pinned_producer_and_split_artifacts(producer_fixture):
    producer, split = _load(producer_fixture)
    assert producer.physical_sha256 == producer_fixture["producer_sha256"]
    assert producer.semantic_sha256 == producer_fixture["producer_semantic_sha256"]
    assert producer.provider_id == "sierra_chart"
    assert producer.source_id == "sc_historical_ticks_v2"
    assert producer.data_mode == "raw_ticks"
    assert producer.production_admitted is False
    assert split.physical_sha256 == producer_fixture["split_sha256"]
    assert split.semantic_sha256 == producer_fixture["split_semantic_sha256"]
    assert split.production_admitted is False
    assert reopen_and_verify_producer_governance_v1(producer) == producer
    assert reopen_and_verify_frozen_split_use_contract_v1(split) == split


def test_split_use_boundary_evaluation_is_exchange_session_day_and_content_blocked(producer_fixture):
    _, split = _load(producer_fixture)
    assert evaluate_session_request_v1(
        split, partition_id="development", requested_use="validation",
        session_day="2025-06-30",
    ) == "eligible_by_split_use_contract_not_content_authorized"
    assert evaluate_session_request_v1(
        split, partition_id="development", requested_use="validation",
        session_day="2025-07-01",
    ) == "boundary_blocked"
    assert evaluate_session_request_v1(
        split, partition_id="legacy_holdout", requested_use="validation",
        session_day="2025-07-01",
    ) == "boundary_blocked"
    assert evaluate_boundary_leaf_v1(
        split, partition_id="shared_train", requested_use="supervised_training",
        session_day="2024-06-28", interval_start_utc_ns=1,
        interval_end_exclusive_utc_ns=2,
    ) == "requires_session_denominator"
    assert evaluate_boundary_leaf_v1(
        split, partition_id="shared_train", requested_use="supervised_training",
        session_day="2024-07-01", interval_start_utc_ns=1,
        interval_end_exclusive_utc_ns=2,
    ) == "boundary_blocked"


def test_expected_physical_hashes_prevent_path_or_byte_substitution(producer_fixture):
    with pytest.raises(CorpusV3ProducerGovernanceError, match="physical SHA-256"):
        load_and_verify_producer_governance_v1(
            producer_fixture["producer_path"], expected_sha256="f" * 64,
        )
    with pytest.raises(CorpusV3ProducerGovernanceError, match="physical SHA-256"):
        load_and_verify_frozen_split_use_contract_v1(
            producer_fixture["split_path"], expected_sha256="f" * 64,
            producer_governance_path=producer_fixture["producer_path"],
            producer_governance_sha256=producer_fixture["producer_sha256"],
        )


def test_arbitrary_detached_provider_cannot_escalate_production(producer_fixture):
    producer_path = Path(producer_fixture["producer_path"])
    split_path = Path(producer_fixture["split_path"])
    producer = json.loads(producer_path.read_text(encoding="ascii"))
    split = json.loads(split_path.read_text(encoding="ascii"))
    producer["source_namespace"].update(
        provider_id="arbitrary_provider", source_id="unproven_source",
        namespace_root="/unproven/provider/namespace",
    )
    _rehash(producer, "governance_semantic_sha256")
    _canonical(producer_path, producer)
    split["parent_producer_governance"].update(
        physical_sha256=_sha(producer_path),
        semantic_sha256=producer["governance_semantic_sha256"],
    )
    split["source_namespace"] = dict(producer["source_namespace"])
    _rehash(split, "contract_semantic_sha256")
    _canonical(split_path, split)
    fixture = {
        **producer_fixture,
        "producer_sha256": _sha(producer_path),
        "split_sha256": _sha(split_path),
    }
    loaded_producer, loaded_split = _load(fixture)
    assert loaded_producer.provider_id == "arbitrary_provider"
    assert loaded_producer.production_admitted is False
    assert loaded_split.production_admitted is False


def test_unknown_noncanonical_duplicate_and_bool_as_int_artifacts_reject(producer_fixture):
    producer_path = Path(producer_fixture["producer_path"])
    original = producer_path.read_bytes()
    producer_path.write_text('{"schema_version":"x","schema_version":"y"}', encoding="utf-8")
    with pytest.raises(CorpusV3ProducerGovernanceError, match="duplicate JSON key"):
        load_and_verify_producer_governance_v1(
            producer_path, expected_sha256=_sha(producer_path),
        )
    producer_path.write_bytes(original)
    value = json.loads(original)
    producer_path.write_text(json.dumps(value, indent=2), encoding="utf-8")
    with pytest.raises(CorpusV3ProducerGovernanceError, match="canonical JSON"):
        load_and_verify_producer_governance_v1(
            producer_path, expected_sha256=_sha(producer_path),
        )
    value["unknown"] = True
    _rehash(value, "governance_semantic_sha256")
    _canonical(producer_path, value)
    with pytest.raises(CorpusV3ProducerGovernanceError, match="unknown"):
        load_and_verify_producer_governance_v1(
            producer_path, expected_sha256=_sha(producer_path),
        )
    value.pop("unknown")
    value["production_admission"] = 0
    _rehash(value, "governance_semantic_sha256")
    _canonical(producer_path, value)
    with pytest.raises(CorpusV3ProducerGovernanceError, match="production_admission"):
        load_and_verify_producer_governance_v1(
            producer_path, expected_sha256=_sha(producer_path),
        )


def test_split_parent_rebinding_and_protocol_widening_reject(producer_fixture):
    split_path = Path(producer_fixture["split_path"])
    split = json.loads(split_path.read_text(encoding="ascii"))
    split["parent_producer_governance"]["semantic_sha256"] = "f" * 64
    _rehash(split, "contract_semantic_sha256")
    _canonical(split_path, split)
    with pytest.raises(CorpusV3ProducerGovernanceError, match="parent hash mismatch"):
        load_and_verify_frozen_split_use_contract_v1(
            split_path, expected_sha256=_sha(split_path),
            producer_governance_path=producer_fixture["producer_path"],
            producer_governance_sha256=producer_fixture["producer_sha256"],
        )
    split = json.loads(Path(producer_fixture["split_path"]).read_text(encoding="ascii"))
    split["parent_producer_governance"]["semantic_sha256"] = producer_fixture[
        "producer_semantic_sha256"
    ]
    split["protocol_scope"]["source_max_date_exclusive"] = "2027-07-01"
    _rehash(split, "contract_semantic_sha256")
    _canonical(split_path, split)
    with pytest.raises(CorpusV3ProducerGovernanceError, match="source_max_date_exclusive"):
        load_and_verify_frozen_split_use_contract_v1(
            split_path, expected_sha256=_sha(split_path),
            producer_governance_path=producer_fixture["producer_path"],
            producer_governance_sha256=producer_fixture["producer_sha256"],
        )


def test_dataclass_copy_and_post_load_replacement_cannot_act_as_capability(producer_fixture):
    producer, split = _load(producer_fixture)
    forged_producer = replace(producer, provider_id="forged_provider")
    with pytest.raises(CorpusV3ProducerGovernanceError, match="changed before use"):
        reopen_and_verify_producer_governance_v1(forged_producer)
    forged_split = replace(split, boundary_leaf_policy="allowed")
    with pytest.raises(CorpusV3ProducerGovernanceError, match="changed before use"):
        reopen_and_verify_frozen_split_use_contract_v1(forged_split)

    split_path = Path(producer_fixture["split_path"])
    value = json.loads(split_path.read_text(encoding="ascii"))
    value["permitted_use_matrix"]["legacy_holdout"] = ["validation"]
    _rehash(value, "contract_semantic_sha256")
    _canonical(split_path, value)
    with pytest.raises(CorpusV3ProducerGovernanceError, match="physical SHA-256"):
        reopen_and_verify_frozen_split_use_contract_v1(split)


def test_symlink_and_hardlink_authority_files_reject(producer_fixture, tmp_path):
    producer_path = Path(producer_fixture["producer_path"])
    original = tmp_path / "producer-original.json"
    original.write_bytes(producer_path.read_bytes())
    link = tmp_path / "producer-link.json"
    link.symlink_to(original)
    with pytest.raises(CorpusV3ProducerGovernanceError, match="symlink|directory component"):
        load_and_verify_producer_governance_v1(link, expected_sha256=_sha(original))
    hardlink = tmp_path / "producer-hardlink.json"
    os.link(original, hardlink)
    with pytest.raises(CorpusV3ProducerGovernanceError, match="bounded regular file"):
        load_and_verify_producer_governance_v1(hardlink, expected_sha256=_sha(original))
