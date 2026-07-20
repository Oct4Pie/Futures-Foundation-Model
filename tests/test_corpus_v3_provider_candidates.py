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
    load_and_verify_frozen_split_use_contract_v1,
)
from futures_foundation.corpus_v3_provider_candidates import (
    CorpusV3ProviderCandidateError,
    load_and_verify_provider_candidate_universe_v1,
    reopen_and_verify_provider_candidate_universe_v1,
)


ROOT = Path(__file__).resolve().parents[1]
ALPHA = ROOT.parent / "alphaforge-corpus-v3-scale"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_frozen(path: Path, value: object) -> None:
    if path.exists():
        path.chmod(0o644)
    path.write_bytes(canonical_json_bytes(value))
    path.chmod(0o444)


def _rehash(value: dict, field: str) -> None:
    value[field] = content_sha256(value, field)


def _refresh_manifest(manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    for entry in manifest["entries"]:
        path = manifest_path.parent / entry["path"]
        entry["sha256"] = _sha(path)
        entry["size"] = path.stat().st_size
    _rehash(manifest, "manifest_semantic_sha256")
    _write_frozen(manifest_path, manifest)


def _refresh_page(manifest: Path, response: Path, receipt: Path) -> None:
    response_value = json.loads(response.read_text(encoding="ascii"))
    _rehash(response_value, "response_semantic_sha256")
    _write_frozen(response, response_value)
    receipt_value = json.loads(receipt.read_text(encoding="ascii"))
    receipt_value["response_sha256"] = _sha(response)
    receipt_value["response_size"] = response.stat().st_size
    _rehash(receipt_value, "receipt_semantic_sha256")
    _write_frozen(receipt, receipt_value)
    _refresh_manifest(manifest)


def _joined_fixture(tmp_path: Path) -> dict[str, str]:
    python = ALPHA / ".venv/bin/python"
    producer_tests = ALPHA / "tests/test_producer_governance.py"
    provider_tests = ALPHA / "tests/test_provider_candidate_universe.py"
    if not python.is_file() or not producer_tests.is_file() or not provider_tests.is_file():
        pytest.skip("pinned AlphaForge provider-candidate worktree is unavailable")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ALPHA, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    assert revision == PRODUCER_COMPATIBILITY_COMMIT
    sidecar = tmp_path / "joined-sidecar.json"
    script = r'''
import hashlib, json, runpy, sys
from pathlib import Path
producer_ns = runpy.run_path(sys.argv[1])
provider_ns = runpy.run_path(sys.argv[2])
root = Path(sys.argv[3]).resolve()
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
result = {
    "producer_path": str(producer_path),
    "producer_sha256": hashlib.sha256(producer_path.read_bytes()).hexdigest(),
    "split_path": str(split_path),
    "split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
    "evidence_root": str(evidence),
    "manifest_path": str(manifest),
    "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    "scope_path": str(scope_path),
    "receipt_0": str(receipts[0]),
    "receipt_1": str(receipts[1]),
    "response_0": str(pages[0][1]),
    "response_1": str(pages[1][1]),
    "market_lake": str(prohibited[0]),
}
Path(sys.argv[4]).write_text(json.dumps(result, sort_keys=True))
'''
    subprocess.run(
        [
            str(python), "-c", script, str(producer_tests), str(provider_tests),
            str(tmp_path / "joined"), str(sidecar),
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


def _load(fixture: dict[str, str]):
    return load_and_verify_provider_candidate_universe_v1(
        fixture["evidence_root"],
        manifest_path=fixture["manifest_path"],
        manifest_sha256=fixture["manifest_sha256"],
        split_capability=_split(fixture),
    )


@pytest.fixture()
def joined_fixture(tmp_path):
    return _joined_fixture(tmp_path)


def test_ffm_derives_joined_metadata_only_candidate_universe(joined_fixture):
    capability = _load(joined_fixture)
    assert capability.production_admitted is False
    assert capability.evidence_status == "synthetic_provider_metadata_pagination_only"
    assert [item.provider_symbol for item in capability.candidates] == [
        "CLU24", "ESU24", "ESZ24",
    ]
    assert capability.split.producer.namespace_root == "/sealed/provider/ticks"
    document = capability.document()
    assert document["candidate_count"] == 3
    assert document["market_namespace_opened"] is False
    assert document["availability_claimed"] is False
    assert document["lifecycle_claimed"] is False
    assert document["materialization_admitted"] is False
    assert document["training_admitted"] is False
    assert reopen_and_verify_provider_candidate_universe_v1(capability) == capability


def test_scope_must_bind_reverified_producer_and_split_semantics(joined_fixture):
    scope_path = Path(joined_fixture["scope_path"])
    scope = json.loads(scope_path.read_text(encoding="ascii"))
    scope["claimed_producer_governance_sha256"] = "f" * 64
    _rehash(scope, "scope_semantic_sha256")
    _write_frozen(scope_path, scope)
    _refresh_manifest(Path(joined_fixture["manifest_path"]))
    joined_fixture["manifest_sha256"] = _sha(Path(joined_fixture["manifest_path"]))
    with pytest.raises(CorpusV3ProviderCandidateError, match="claims differ"):
        _load(joined_fixture)


def test_forbidden_observed_data_fields_reject_even_when_bundle_is_rehashed(joined_fixture):
    response = Path(joined_fixture["response_0"])
    value = json.loads(response.read_text(encoding="ascii"))
    value["price"] = 123
    _write_frozen(response, value)
    _refresh_page(
        Path(joined_fixture["manifest_path"]), response,
        Path(joined_fixture["receipt_0"]),
    )
    joined_fixture["manifest_sha256"] = _sha(Path(joined_fixture["manifest_path"]))
    with pytest.raises(CorpusV3ProviderCandidateError, match="forbidden observed-data"):
        _load(joined_fixture)


def test_pagination_cursor_chain_and_terminal_page_are_exact(joined_fixture):
    response = Path(joined_fixture["response_0"])
    value = json.loads(response.read_text(encoding="ascii"))
    value["next_cursor"] = "different-cursor"
    _write_frozen(response, value)
    _refresh_page(
        Path(joined_fixture["manifest_path"]), response,
        Path(joined_fixture["receipt_0"]),
    )
    joined_fixture["manifest_sha256"] = _sha(Path(joined_fixture["manifest_path"]))
    with pytest.raises(CorpusV3ProviderCandidateError, match="cursor chain"):
        _load(joined_fixture)


def test_candidate_ids_and_symbols_are_globally_unique(joined_fixture):
    response = Path(joined_fixture["response_1"])
    value = json.loads(response.read_text(encoding="ascii"))
    value["candidates"][0].update(
        provider_instrument_id="es-202409", provider_symbol="ESU24",
    )
    _write_frozen(response, value)
    _refresh_page(
        Path(joined_fixture["manifest_path"]), response,
        Path(joined_fixture["receipt_1"]),
    )
    joined_fixture["manifest_sha256"] = _sha(Path(joined_fixture["manifest_path"]))
    with pytest.raises(CorpusV3ProviderCandidateError, match="globally unique"):
        _load(joined_fixture)


def test_bundle_has_exact_file_and_manifest_hash_closure(joined_fixture):
    extra = Path(joined_fixture["evidence_root"]) / "unmanifested.json"
    _write_frozen(extra, {"unexpected": True})
    with pytest.raises(CorpusV3ProviderCandidateError, match="closure"):
        _load(joined_fixture)
    extra.unlink()
    with pytest.raises(CorpusV3ProviderCandidateError, match="physical SHA-256"):
        load_and_verify_provider_candidate_universe_v1(
            joined_fixture["evidence_root"],
            manifest_path=joined_fixture["manifest_path"],
            manifest_sha256="f" * 64,
            split_capability=_split(joined_fixture),
        )


def test_dataclass_copy_and_parent_replacement_cannot_act_as_capability(joined_fixture):
    capability = _load(joined_fixture)
    forged = replace(capability, provider_id="forged_provider")
    with pytest.raises(CorpusV3ProviderCandidateError, match="changed before use"):
        reopen_and_verify_provider_candidate_universe_v1(forged)

    receipt = Path(joined_fixture["receipt_0"])
    value = json.loads(receipt.read_text(encoding="ascii"))
    value["captured_utc"] = "2026-07-18T00:01:00Z"
    _rehash(value, "receipt_semantic_sha256")
    _write_frozen(receipt, value)
    _refresh_manifest(Path(joined_fixture["manifest_path"]))
    with pytest.raises(CorpusV3ProviderCandidateError, match="physical SHA-256"):
        reopen_and_verify_provider_candidate_universe_v1(capability)


def test_symlinked_evidence_root_rejects_and_market_namespace_is_never_required(joined_fixture, tmp_path):
    alias = tmp_path / "evidence-alias"
    alias.symlink_to(Path(joined_fixture["evidence_root"]), target_is_directory=True)
    with pytest.raises(CorpusV3ProviderCandidateError, match="symlink|unavailable"):
        load_and_verify_provider_candidate_universe_v1(
            alias,
            manifest_path=alias / "bundle-manifest.json",
            manifest_sha256=joined_fixture["manifest_sha256"],
            split_capability=_split(joined_fixture),
        )
    # The producer namespace is deliberately nonexistent in the synthetic fixture;
    # successful direct loading proves the consumer did not open or enumerate it.
    assert not Path("/sealed/provider/ticks").exists()
    assert _load(joined_fixture).production_admitted is False
