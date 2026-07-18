from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from futures_foundation.finetune.native_contracts import NativeContractError, content_sha256
from futures_foundation.finetune.native_training_data_authority import (
    AUTHORITY_ID,
    AUTHORITY_PATH,
    BINDING_BLOCKERS,
    CORPUS_CONTRACT_PATH,
    TRAINING_DATA_AUTHORITY,
    TRUSTED_AUTHORITY_CONTENT_SHA256,
    load_training_data_authority,
    resolve_training_data_authority,
    training_data_authority_sha256,
    validate_training_data_authority,
)


def _corpus_contract():
    return json.loads(Path(CORPUS_CONTRACT_PATH).read_text(encoding="utf-8"))


def test_authority_is_minimal_blocked_and_bound_to_actual_corpus_contract():
    authority = load_training_data_authority()
    contract = _corpus_contract()
    assert authority == TRAINING_DATA_AUTHORITY
    assert set(authority) == {
        "schema_version", "authority_id", "status", "non_authorizing",
        "corpus_contract_ref", "unresolved_bindings", "blocker_tags",
    }
    assert authority["status"] == "blocked"
    assert authority["non_authorizing"] is True
    assert training_data_authority_sha256(authority) == TRUSTED_AUTHORITY_CONTENT_SHA256
    assert authority["corpus_contract_ref"] == {
        "schema_version": contract["schema_version"],
        "contract_id": contract["contract_id"],
        "content_sha256": content_sha256(contract),
    }
    assert set(authority["unresolved_bindings"]) == set(BINDING_BLOCKERS)
    assert authority["blocker_tags"] == sorted(BINDING_BLOCKERS.values())
    for name, blocker in BINDING_BLOCKERS.items():
        assert authority["unresolved_bindings"][name] == {
            "state": "unresolved", "value": None, "blocker_tag": blocker,
        }
    serialized = json.dumps(authority, sort_keys=True)
    for forbidden_copy in (
        "admitted_roots", "splits", "execution_ruler", "timeframes_minutes",
        "physical_root", "artifacts",
    ):
        assert forbidden_copy not in serialized


def test_authority_resolver_accepts_only_the_canonical_id_and_hash():
    digest = training_data_authority_sha256()
    assert resolve_training_data_authority(AUTHORITY_ID, digest) == TRAINING_DATA_AUTHORITY
    with pytest.raises(NativeContractError, match="id is unknown"):
        resolve_training_data_authority("self_issued_authority", digest)
    with pytest.raises(NativeContractError, match="self-authored or stale"):
        resolve_training_data_authority(AUTHORITY_ID, "0" * 64)


def test_rejects_alternate_path_even_with_same_basename_and_bytes(tmp_path):
    alternate = tmp_path / AUTHORITY_PATH.name
    shutil.copyfile(AUTHORITY_PATH, alternate)
    with pytest.raises(NativeContractError, match="outside the packaged authority locations"):
        load_training_data_authority(alternate)
    alternate_contract = tmp_path / CORPUS_CONTRACT_PATH.name
    shutil.copyfile(CORPUS_CONTRACT_PATH, alternate_contract)
    with pytest.raises(NativeContractError, match="outside the packaged authority locations"):
        load_training_data_authority(corpus_contract_path=alternate_contract)


def test_rejects_status_escalation_hash_substitution_and_fake_resolution():
    authority = deepcopy(TRAINING_DATA_AUTHORITY)
    authority["status"] = "verified"
    authority["non_authorizing"] = False
    with pytest.raises(NativeContractError, match="must remain blocked"):
        validate_training_data_authority(authority)
    authority = deepcopy(TRAINING_DATA_AUTHORITY)
    authority["corpus_contract_ref"]["content_sha256"] = "0" * 64
    with pytest.raises(NativeContractError, match="stale or substituted"):
        validate_training_data_authority(authority)
    authority = deepcopy(TRAINING_DATA_AUTHORITY)
    authority["unresolved_bindings"]["sample_manifest"] = {
        "state": "resolved", "value": "1" * 64,
        "blocker_tag": "sample_manifest_unresolved",
    }
    with pytest.raises(NativeContractError, match="is not unresolved"):
        validate_training_data_authority(authority)


def test_rejects_modified_corpus_even_with_a_self_issued_matching_hash():
    contract = _corpus_contract()
    contract["purpose"] = "self-issued replacement contract"
    authority = deepcopy(TRAINING_DATA_AUTHORITY)
    authority["corpus_contract_ref"]["content_sha256"] = content_sha256(contract)
    with pytest.raises(NativeContractError, match="code-pinned trust anchor"):
        validate_training_data_authority(authority, corpus_contract=contract)


def test_fresh_import_rejects_coherent_packaged_pair_rewrite(tmp_path):
    """A wheel-local attacker cannot redefine both JSON files before first import."""
    root = Path(__file__).resolve().parents[1]
    shutil.copytree(
        root / "futures_foundation",
        tmp_path / "futures_foundation",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copytree(root / "config", tmp_path / "config")
    contract_path = tmp_path / "config" / "corpus_v3" / "contract.json"
    authority_path = (
        tmp_path / "config" / "foundation_models"
        / "native_training_data_authority_v1.json"
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["admitted_roots"] = sorted([*contract["admitted_roots"], "ZZ"])
    contract_path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    authority["corpus_contract_ref"]["content_sha256"] = content_sha256(contract)
    authority_path.write_text(json.dumps(authority, indent=2) + "\n", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "-c", (
            "from futures_foundation.finetune.native_training_data_authority "
            "import load_training_data_authority; load_training_data_authority()"
        )],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "code-pinned trust anchor" in result.stderr
