from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from futures_foundation._authority_bundle_io import canonical_json_bytes, content_sha256
from futures_foundation.corpus_v3_contract_lifecycle import (
    CorpusV3ContractLifecycleError,
    PRODUCER_COMPATIBILITY_COMMIT,
    load_and_verify_contract_lifecycle_v2,
    reopen_and_verify_contract_lifecycle_v2,
)
from futures_foundation.corpus_v3_producer_governance import (
    load_and_verify_frozen_split_use_contract_v1,
)
from futures_foundation.corpus_v3_provider_candidates import (
    load_and_verify_provider_candidate_universe_v1,
)


ROOT = Path(__file__).resolve().parents[1]
ALPHA = ROOT.parent / "alphaforge-corpus-v3-scale"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def _write_frozen(path: Path, value: object) -> None:
    if path.exists():
        path.chmod(0o644)
    _write(path, value)
    path.chmod(0o444)


def _rehash(value: dict, field: str) -> None:
    value[field] = content_sha256(value, field)


def _joined_fixture(tmp_path: Path) -> dict[str, str]:
    python = ALPHA / ".venv/bin/python"
    producer_tests = ALPHA / "tests/test_producer_governance.py"
    provider_tests = ALPHA / "tests/test_provider_candidate_universe.py"
    lifecycle_tests = ALPHA / "tests/test_contract_lifecycle_capability.py"
    provider_source = ALPHA / "src/alphaforge/provider_candidate_universe.py"
    required = (python, producer_tests, provider_tests, lifecycle_tests, provider_source)
    if any(not path.is_file() for path in required):
        pytest.skip("pinned AlphaForge lifecycle compatibility worktree is unavailable")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ALPHA, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    assert revision == PRODUCER_COMPATIBILITY_COMMIT
    sidecar = tmp_path / "lifecycle-sidecar.json"
    script = r'''
import hashlib, json, runpy, sys
from pathlib import Path
producer_ns = runpy.run_path(sys.argv[1])
provider_ns = runpy.run_path(sys.argv[2])
lifecycle_ns = runpy.run_path(sys.argv[3])
root = Path(sys.argv[4]).resolve()
root.mkdir(parents=True)
producer_path, split_path, producer, split = producer_ns["_fixture"](root / "governance")
original_scope = provider_ns["_scope"]
def bound_scope(directory):
    path, scope = original_scope(directory)
    scope["claimed_producer_governance_sha256"] = producer["governance_semantic_sha256"]
    scope["claimed_frozen_split_use_contract_sha256"] = split["contract_semantic_sha256"]
    scope.pop("scope_semantic_sha256", None)
    scope["scope_semantic_sha256"] = provider_ns["content_sha256"](scope)
    provider_ns["_rewrite_frozen"](path, scope)
    return path, scope
provider_ns["_bundle"].__globals__["_scope"] = bound_scope
evidence, manifest, scope_path, scope, receipts, pages, prohibited = provider_ns["_bundle"](
    root / "provider"
)
# Keep the two CL/ES contracts used by the lifecycle fixture and preserve a
# complete two-page terminal pagination proof with an empty second page.
response_path = pages[1][1]
response = json.loads(response_path.read_text())
response["candidates"] = []
response["response_semantic_sha256"] = provider_ns["content_sha256"]({
    key: value for key, value in response.items() if key != "response_semantic_sha256"
})
provider_ns["_rewrite_frozen"](response_path, response)
receipt_path = receipts[1]
receipt = json.loads(receipt_path.read_text())
raw = response_path.read_bytes()
receipt["response_sha256"] = hashlib.sha256(raw).hexdigest()
receipt["response_size"] = len(raw)
receipt["receipt_semantic_sha256"] = provider_ns["content_sha256"]({
    key: value for key, value in receipt.items() if key != "receipt_semantic_sha256"
})
provider_ns["_rewrite_frozen"](receipt_path, receipt)
provider_ns["_refresh_manifest"](manifest)
universe = provider_ns["build_provider_candidate_universe"](
    evidence, manifest, prohibited_roots=prohibited,
)
universe_path = provider_ns["_write_frozen"](root / "provider-universe.json", universe)
paths = {
    "producer": producer_path,
    "split": split_path,
    "universe": universe_path,
    "registry": root / "registry.json",
    "lifecycle": root / "lifecycle.json",
}
registry = lifecycle_ns["_registry"](universe)
documents = {
    "producer": producer,
    "split": split,
    "universe": universe,
    "registry": registry,
}
lifecycle = lifecycle_ns["_lifecycle"](paths, documents)
lifecycle_ns["_write"](paths["registry"], registry)
lifecycle_ns["_write"](paths["lifecycle"], lifecycle)
result = {
    "producer_path": str(producer_path),
    "producer_sha256": hashlib.sha256(producer_path.read_bytes()).hexdigest(),
    "split_path": str(split_path),
    "split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
    "evidence_root": str(evidence),
    "manifest_path": str(manifest),
    "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    "provider_universe_path": str(universe_path),
    "provider_universe_sha256": hashlib.sha256(universe_path.read_bytes()).hexdigest(),
    "registry_path": str(paths["registry"]),
    "registry_sha256": hashlib.sha256(paths["registry"].read_bytes()).hexdigest(),
    "lifecycle_path": str(paths["lifecycle"]),
    "lifecycle_sha256": hashlib.sha256(paths["lifecycle"].read_bytes()).hexdigest(),
}
Path(sys.argv[5]).write_text(json.dumps(result, sort_keys=True))
'''
    subprocess.run(
        [
            str(python), "-c", script, str(producer_tests), str(provider_tests),
            str(lifecycle_tests), str(tmp_path / "joined"), str(sidecar),
        ],
        cwd=ALPHA, check=True, capture_output=True, text=True,
    )
    return json.loads(sidecar.read_text(encoding="utf-8"))


def _split(fixture: dict[str, str]):
    return load_and_verify_frozen_split_use_contract_v1(
        fixture["split_path"], expected_sha256=fixture["split_sha256"],
        producer_governance_path=fixture["producer_path"],
        producer_governance_sha256=fixture["producer_sha256"],
    )


def _candidates(fixture: dict[str, str]):
    return load_and_verify_provider_candidate_universe_v1(
        fixture["evidence_root"],
        manifest_path=fixture["manifest_path"],
        manifest_sha256=fixture["manifest_sha256"],
        split_capability=_split(fixture),
    )


def _load(fixture: dict[str, str]):
    return load_and_verify_contract_lifecycle_v2(
        fixture["lifecycle_path"],
        lifecycle_sha256=fixture["lifecycle_sha256"],
        registry_path=fixture["registry_path"],
        registry_sha256=fixture["registry_sha256"],
        provider_universe_path=fixture["provider_universe_path"],
        provider_universe_sha256=fixture["provider_universe_sha256"],
        split_capability=_split(fixture),
        provider_candidate_capability=_candidates(fixture),
    )


@pytest.fixture()
def lifecycle_fixture(tmp_path):
    return _joined_fixture(tmp_path)


def _refresh_lifecycle(fixture: dict[str, str], value: dict) -> None:
    _rehash(value, "lifecycle_semantic_sha256")
    path = Path(fixture["lifecycle_path"])
    _write(path, value)
    fixture["lifecycle_sha256"] = _sha(path)


def _refresh_registry(fixture: dict[str, str], value: dict) -> None:
    _rehash(value, "registry_semantic_sha256")
    path = Path(fixture["registry_path"])
    _write(path, value)
    fixture["registry_sha256"] = _sha(path)
    lifecycle_path = Path(fixture["lifecycle_path"])
    lifecycle = json.loads(lifecycle_path.read_text(encoding="ascii"))
    parent = lifecycle["parent_artifacts"]["official_lifecycle_evidence_registry"]
    parent["physical_sha256"] = fixture["registry_sha256"]
    parent["semantic_sha256"] = value["registry_semantic_sha256"]
    _refresh_lifecycle(fixture, lifecycle)


def test_ffm_reopens_complete_joined_lifecycle_chain(lifecycle_fixture):
    capability = _load(lifecycle_fixture)
    assert capability.production_admitted is False
    assert capability.evidence_status == (
        "synthetic_lifecycle_mechanism_with_reverified_candidate_chain"
    )
    assert [row.contract_id for row in capability.rows] == ["CLU24", "ESU24"]
    assert [row.disposition for row in capability.rows] == ["admit", "admit"]
    document = capability.document()
    assert document["market_data_read"] is False
    assert document["materialization_admitted"] is False
    assert document["training_admitted"] is False
    assert reopen_and_verify_contract_lifecycle_v2(capability) == capability


def test_provider_universe_artifact_must_reproduce_reverified_pagination(lifecycle_fixture):
    path = Path(lifecycle_fixture["provider_universe_path"])
    universe = json.loads(path.read_text(encoding="ascii"))
    universe["candidates"][0]["provider_symbol"] = "CLZ24"
    _rehash(universe, "universe_semantic_sha256")
    _write_frozen(path, universe)
    lifecycle_fixture["provider_universe_sha256"] = _sha(path)
    lifecycle_path = Path(lifecycle_fixture["lifecycle_path"])
    lifecycle = json.loads(lifecycle_path.read_text(encoding="ascii"))
    parent = lifecycle["parent_artifacts"]["provider_candidate_universe"]
    parent["physical_sha256"] = lifecycle_fixture["provider_universe_sha256"]
    parent["semantic_sha256"] = universe["universe_semantic_sha256"]
    _refresh_lifecycle(lifecycle_fixture, lifecycle)
    with pytest.raises(CorpusV3ContractLifecycleError, match="differs from reverified|candidate rows differ"):
        _load(lifecycle_fixture)


def test_lifecycle_parent_hashes_and_paths_are_exact(lifecycle_fixture):
    path = Path(lifecycle_fixture["lifecycle_path"])
    lifecycle = json.loads(path.read_text(encoding="ascii"))
    lifecycle["parent_artifacts"]["frozen_split_use_contract"]["semantic_sha256"] = "f" * 64
    _refresh_lifecycle(lifecycle_fixture, lifecycle)
    with pytest.raises(CorpusV3ContractLifecycleError, match="parent path/hash"):
        _load(lifecycle_fixture)


def test_plan_and_observed_market_fields_are_forbidden(lifecycle_fixture):
    path = Path(lifecycle_fixture["lifecycle_path"])
    lifecycle = json.loads(path.read_text(encoding="ascii"))
    lifecycle["materialization_plan"] = "f" * 64
    _rehash(lifecycle, "lifecycle_semantic_sha256")
    _write(path, lifecycle)
    lifecycle_fixture["lifecycle_sha256"] = _sha(path)
    with pytest.raises(CorpusV3ContractLifecycleError, match="forbidden field"):
        _load(lifecycle_fixture)


def test_missing_candidate_requires_honest_wholly_null_quarantine(lifecycle_fixture):
    registry_path = Path(lifecycle_fixture["registry_path"])
    registry = json.loads(registry_path.read_text(encoding="ascii"))
    registry["claims"] = [
        claim for claim in registry["claims"]
        if claim["provider_instrument_id"] != "es-202409"
    ]
    registry["official_sources"] = [registry["official_sources"][0]]
    _refresh_registry(lifecycle_fixture, registry)
    lifecycle_path = Path(lifecycle_fixture["lifecycle_path"])
    lifecycle = json.loads(lifecycle_path.read_text(encoding="ascii"))
    row = lifecycle["rows"][1]
    row.update(
        start_kind=None, eligibility_start_utc_ns=None, end_kind=None,
        trading_end_exclusive_utc_ns=None, official_source_ids=[], disposition="quarantine",
    )
    _refresh_lifecycle(lifecycle_fixture, lifecycle)
    capability = _load(lifecycle_fixture)
    assert capability.rows[1].disposition == "quarantine"
    assert capability.rows[1].eligibility_start_utc_ns is None


def test_partial_evidence_caller_disposition_and_free_text_mapping_reject(lifecycle_fixture):
    path = Path(lifecycle_fixture["lifecycle_path"])
    lifecycle = json.loads(path.read_text(encoding="ascii"))
    lifecycle["rows"][0]["eligibility_start_utc_ns"] = None
    _refresh_lifecycle(lifecycle_fixture, lifecycle)
    with pytest.raises(CorpusV3ContractLifecycleError, match="complete or wholly"):
        _load(lifecycle_fixture)

    lifecycle_fixture = _joined_fixture(path.parent / "second")
    path = Path(lifecycle_fixture["lifecycle_path"])
    lifecycle = json.loads(path.read_text(encoding="ascii"))
    lifecycle["rows"][0]["disposition"] = "quarantine"
    _refresh_lifecycle(lifecycle_fixture, lifecycle)
    with pytest.raises(CorpusV3ContractLifecycleError, match="not derived"):
        _load(lifecycle_fixture)

    lifecycle_fixture = _joined_fixture(path.parent / "third")
    path = Path(lifecycle_fixture["lifecycle_path"])
    lifecycle = json.loads(path.read_text(encoding="ascii"))
    lifecycle["rows"][0]["contract_id"] = "CUSTOM_MAPPING"
    _refresh_lifecycle(lifecycle_fixture, lifecycle)
    with pytest.raises(CorpusV3ContractLifecycleError, match="must equal provider_symbol"):
        _load(lifecycle_fixture)


def test_censoring_and_official_claim_values_are_exact(lifecycle_fixture):
    path = Path(lifecycle_fixture["lifecycle_path"])
    lifecycle = json.loads(path.read_text(encoding="ascii"))
    lifecycle["rows"][1]["eligibility_start_utc_ns"] += 1
    _refresh_lifecycle(lifecycle_fixture, lifecycle)
    with pytest.raises(CorpusV3ContractLifecycleError, match="left censoring"):
        _load(lifecycle_fixture)

    lifecycle_fixture = _joined_fixture(path.parent / "claim")
    registry_path = Path(lifecycle_fixture["registry_path"])
    registry = json.loads(registry_path.read_text(encoding="ascii"))
    registry["claims"][1]["claim_utc_ns"] += 1
    _refresh_registry(lifecycle_fixture, registry)
    with pytest.raises(CorpusV3ContractLifecycleError, match="start is not established"):
        _load(lifecycle_fixture)


def test_dataclass_copy_and_post_load_parent_replacement_reject(lifecycle_fixture):
    capability = _load(lifecycle_fixture)
    forged = replace(capability, evidence_status="forged")
    with pytest.raises(CorpusV3ContractLifecycleError, match="changed before use"):
        reopen_and_verify_contract_lifecycle_v2(forged)
    registry_path = Path(lifecycle_fixture["registry_path"])
    registry = json.loads(registry_path.read_text(encoding="ascii"))
    registry["claims"][0]["claim_utc_ns"] += 1
    _rehash(registry, "registry_semantic_sha256")
    _write(registry_path, registry)
    with pytest.raises(CorpusV3ContractLifecycleError, match="physical SHA-256"):
        reopen_and_verify_contract_lifecycle_v2(capability)
