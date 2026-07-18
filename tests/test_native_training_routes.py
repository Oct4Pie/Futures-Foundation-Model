import copy
import json
import os
from pathlib import Path

import pytest

from futures_foundation.finetune import native_contracts, native_training_routes
from futures_foundation.finetune.native_contracts import NativeContractError, REGISTRY_PATH


def _clear_caches():
    native_contracts.load_registry.cache_clear()
    native_contracts.load_evidence.cache_clear()
    native_training_routes.load_route_registry.cache_clear()
    native_training_routes.load_route_evidence.cache_clear()


def _copy_contract_tree(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    registry_path = tmp_path / "native_contracts.json"
    for name in (
        "native_contracts.json",
        "native_contract_evidence.json",
        "native_training_routes.json",
        "native_training_route_evidence.json",
        "trusted_approvers.json",
    ):
        registry_path.with_name(name).write_bytes(REGISTRY_PATH.with_name(name).read_bytes())
    _clear_caches()
    return registry_path


def _passing_checks():
    return {
        name: {"status": "pass", "evidence": f"fixture:{name}"}
        for name in native_training_routes.TRAINING_CHECKS
    }


def _admit_route(
    registry_path: Path,
    key: str,
    *,
    status: str,
    evidence_status: str,
    clear_caches: bool = True,
):
    route_path = registry_path.with_name("native_training_routes.json")
    evidence_path = registry_path.with_name("native_training_route_evidence.json")
    routes = json.loads(route_path.read_text(encoding="utf-8"))
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    source = {"kind": "git_commit", "revision": "a" * 40}
    routes["training_methodology_source"] = source
    route = routes["routes"][key]
    route["status"] = status
    evidence_id = f"{key}:fixture"
    route["evidence_id"] = evidence_id
    if status == "research_only":
        route["allowed_use_scopes"] = ["research_noncommercial"]
    evidence["training_methodology_source"] = source
    evidence["records"] = {
        evidence_id: {
            "arm_key": route["arm_key"],
            "track": route["track"],
            "route_id": route["route_id"],
            "route_key": key,
            "route_sha256": native_contracts.content_sha256(route),
            "status": evidence_status,
            "checks": _passing_checks(),
            "environment": {"python": "fixture", "dtype": "float32"},
            "artifacts": {"trainer": {"sha256": "b" * 64}},
            "reason": "synthetic route-admission fixture",
        }
    }
    evidence["route_registry_sha256"] = native_contracts.content_sha256(routes)
    route_path.write_text(json.dumps(routes), encoding="utf-8")
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    if clear_caches:
        _clear_caches()
    return routes, evidence, evidence_id


def test_canonical_training_route_layer_is_complete_and_fully_blocked():
    registry = native_training_routes.load_route_registry()
    assert len(registry["routes"]) == 32
    assert {route["status"] for route in registry["routes"].values()} == {"blocked"}
    assert all(route["evidence_id"] is None for route in registry["routes"].values())
    assert native_training_routes.load_route_evidence()["records"] == {}
    assert "ttm_r2:F:direct_raw_hf_trainer_forecast" in registry["routes"]
    assert "toto2_22m:F:no_released_toto2_finetuning" in registry["routes"]
    assert not any(arm.supported_training for arm in native_contracts.all_arms().values())
    assert not any(arm.training_admitted for arm in native_contracts.all_arms().values())


def test_phase_a_rejects_every_self_authored_nonblocked_route(tmp_path):
    for status, evidence_status in (
        ("admitted", "pass"),
        ("research_only", "research_only_pass"),
    ):
        registry_path = _copy_contract_tree(tmp_path / status)
        key = "kronos_small:F:native_tokenizer_and_hierarchical_autoregressive"
        _admit_route(
            registry_path, key, status=status, evidence_status=evidence_status
        )
        with pytest.raises(NativeContractError, match="Phase A forbids nonblocked"):
            native_training_routes.authorize_route(
                arm_key="kronos_small", track="F",
                route_id="native_tokenizer_and_hierarchical_autoregressive",
                use_scope=("production" if status == "admitted"
                           else "research_noncommercial"),
                path=registry_path,
            )


def test_mandatory_training_check_cannot_be_not_applicable():
    checks = _passing_checks()
    checks["causal_prefix_invariance"] = {
        "status": "not_applicable", "evidence": "attempted bypass",
    }
    with pytest.raises(NativeContractError, match="invalid status"):
        native_training_routes._validate_checks(checks, "fixture")


def test_route_profile_matrix_is_frozen_independently_of_registry_hash(tmp_path):
    registry_path = _copy_contract_tree(tmp_path)
    route_path = registry_path.with_name("native_training_routes.json")
    evidence_path = registry_path.with_name("native_training_route_evidence.json")
    routes = json.loads(route_path.read_text(encoding="utf-8"))
    key = "kronos_small:F:native_tokenizer_and_hierarchical_autoregressive"
    routes["routes"][key]["contract_profile"] = "timesfm_lora"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["route_registry_sha256"] = native_contracts.content_sha256(routes)
    route_path.write_text(json.dumps(routes), encoding="utf-8")
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    _clear_caches()
    with pytest.raises(NativeContractError, match="matrix drifted from code"):
        native_training_routes.load_route_registry(registry_path)


def test_training_evidence_cannot_be_orphaned(tmp_path):
    registry_path = _copy_contract_tree(tmp_path)
    evidence_path = registry_path.with_name("native_training_route_evidence.json")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["records"]["orphan"] = {
        "arm_key": "kronos_small",
        "track": "F",
        "route_id": "native_tokenizer_and_hierarchical_autoregressive",
        "route_key": "kronos_small:F:native_tokenizer_and_hierarchical_autoregressive",
        "route_sha256": "0" * 64,
        "status": "pass",
        "checks": _passing_checks(),
        "environment": {"python": "fixture"},
        "artifacts": {"trainer": {"sha256": "b" * 64}},
        "reason": "orphan attempt",
    }
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    _clear_caches()
    with pytest.raises(NativeContractError):
        native_training_routes.load_route_registry(registry_path)


def test_authorization_reopens_files_after_cached_blocked_state_is_replaced(tmp_path):
    registry_path = _copy_contract_tree(tmp_path)
    key = "kronos_small:F:native_tokenizer_and_hierarchical_autoregressive"
    # Populate the non-authorizing display cache with the valid blocked snapshot, then
    # self-author a synthetic admitted state without clearing that cache.  Authorization
    # must reopen the files and reject the changed state under the Phase-A policy.
    native_training_routes.load_route_registry(registry_path)
    _admit_route(
        registry_path, key, status="admitted", evidence_status="pass",
        clear_caches=False,
    )
    with pytest.raises(NativeContractError, match="Phase A forbids nonblocked"):
        native_training_routes.authorize_route(
            arm_key="kronos_small", track="F",
            route_id="native_tokenizer_and_hierarchical_autoregressive",
            use_scope="production", path=registry_path,
        )


def test_authorization_revalidates_evidence_even_when_display_registry_is_cached(tmp_path):
    registry_path = _copy_contract_tree(tmp_path)
    evidence_path = registry_path.with_name("native_training_route_evidence.json")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    native_training_routes.load_route_registry(registry_path)
    evidence["records"] = []
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    with pytest.raises(NativeContractError, match="records must be an object"):
        native_training_routes.authorize_route(
            arm_key="kronos_small", track="F",
            route_id="native_tokenizer_and_hierarchical_autoregressive",
            use_scope="production", path=registry_path,
        )


def test_authorization_rejects_symlinked_route_contract(tmp_path):
    registry_path = _copy_contract_tree(tmp_path)
    route_path = registry_path.with_name("native_training_routes.json")
    target = tmp_path / "route-target.json"
    target.write_bytes(route_path.read_bytes())
    route_path.unlink()
    route_path.symlink_to(target)
    with pytest.raises(NativeContractError, match="single-link regular file"):
        native_training_routes.authorize_route(
            arm_key="kronos_small", track="F",
            route_id="native_tokenizer_and_hierarchical_autoregressive",
            use_scope="production", path=registry_path,
        )


def test_authorization_rejects_symlinked_ancestor_directory(tmp_path):
    real = tmp_path / "real"
    registry_path = _copy_contract_tree(real)
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    alias_registry = alias / registry_path.name
    with pytest.raises(NativeContractError, match="securely open training contract directory"):
        native_training_routes.authorize_route(
            arm_key="kronos_small", track="F",
            route_id="native_tokenizer_and_hierarchical_autoregressive",
            use_scope="production", path=alias_registry,
        )


def test_authorization_rejects_hardlinked_contract(tmp_path):
    registry_path = _copy_contract_tree(tmp_path)
    route_path = registry_path.with_name("native_training_routes.json")
    os.link(route_path, tmp_path / "second-route-link.json")
    with pytest.raises(NativeContractError, match="single-link regular file"):
        native_training_routes.authorize_route(
            arm_key="kronos_small", track="F",
            route_id="native_tokenizer_and_hierarchical_autoregressive",
            use_scope="production", path=registry_path,
        )


def test_authorization_rejects_duplicate_json_keys(tmp_path):
    registry_path = _copy_contract_tree(tmp_path)
    evidence_path = registry_path.with_name("native_training_route_evidence.json")
    evidence = evidence_path.read_text(encoding="utf-8")
    duplicate = evidence.replace('"records": {}', '"records": {}, "records": {}', 1)
    assert duplicate != evidence
    evidence_path.write_text(duplicate, encoding="utf-8")
    with pytest.raises(NativeContractError, match="duplicate key 'records'"):
        native_training_routes.authorize_route(
            arm_key="kronos_small", track="F",
            route_id="native_tokenizer_and_hierarchical_autoregressive",
            use_scope="production", path=registry_path,
        )


def test_authorization_rejects_oversized_contract(tmp_path):
    registry_path = _copy_contract_tree(tmp_path)
    evidence_path = registry_path.with_name("native_training_route_evidence.json")
    evidence_path.write_bytes(
        evidence_path.read_bytes() + b" " * native_training_routes.MAX_CONTRACT_BYTES
    )
    with pytest.raises(NativeContractError, match="exceeds"):
        native_training_routes.authorize_route(
            arm_key="kronos_small", track="F",
            route_id="native_tokenizer_and_hierarchical_autoregressive",
            use_scope="production", path=registry_path,
        )


@pytest.mark.parametrize("payload", [b'{"x":1e999}', b'{"x":' + b"9" * 5000 + b'}'])
def test_strict_json_rejects_numeric_resource_attacks(payload):
    with pytest.raises(NativeContractError, match="strict UTF-8 JSON|non-finite"):
        native_training_routes._strict_json_object(payload, "attack.json")
