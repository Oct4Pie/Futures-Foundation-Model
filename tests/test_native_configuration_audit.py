from copy import deepcopy

import pytest

from futures_foundation.finetune.native_configuration_audit import (
    build_native_configuration_audit,
    load_current_parity_aggregate,
    validate_native_configuration_audit,
)
from futures_foundation.finetune.native_contracts import NativeContractError, content_sha256
from futures_foundation.finetune.native_family_route_catalog_v2 import (
    load_family_route_catalog,
)


def test_configuration_audit_closes_every_model_and_route_without_admission():
    report = build_native_configuration_audit()
    assert report["counts"] == {
        "models": 15,
        "catalog_routes": 42,
        "constraint_profiles": 29,
        "admitted_native_inference_tracks": 16,
        "models_without_admitted_native_track": 2,
        "exact_training_executors": 23,
        "non_exact_or_blocked_routes": 19,
        "configuration_discrepancies": 0,
        "unresolved_constraints": 0,
        "closed_unsupported_routes": 17,
        "externally_blocked_routes": 2,
    }
    assert report["configuration_integrity_passed"] is True
    assert report["configuration_contracts_complete"] is True
    assert report["all_routes_dispositioned"] is True
    assert report["current_inference_parity_complete"] is False
    assert report["all_models_execution_ready"] is False
    assert report["all_training_routes_execution_ready"] is False
    assert report["training_admitted"] is False
    assert report["live_trading_ready"] is False
    assert report["discrepancies"] == []
    assert validate_native_configuration_audit(report) == report


def test_exact_routes_reject_unimplemented_gradient_accumulation():
    report = build_native_configuration_audit()
    assert len(report["exact_routes"]) == 23
    for route in report["exact_routes"]:
        assert route["configuration_consistent"] is True
        if "gradient_accumulation_steps" in route["config"]:
            assert route["config"]["gradient_accumulation_steps"] == 1
        assert route["optimizer_runtime"]["optimizer"] in {"Adam", "AdamW"}


def test_audit_rejects_cross_layer_catalog_drift(monkeypatch):
    catalog = deepcopy(load_family_route_catalog())
    catalog["constraint_profiles"]["chronos_bolt_forecast"]["export"]["value"][
        "output_tag"
    ] = "invented_hidden_state_surface"
    monkeypatch.setattr(
        "futures_foundation.finetune.native_configuration_audit.load_family_route_catalog",
        lambda: catalog,
    )
    with pytest.raises(NativeContractError, match="canonical route semantics"):
        build_native_configuration_audit()


def test_audit_rejects_integrity_forgery():
    report = build_native_configuration_audit()
    report["all_models_execution_ready"] = True
    payload = dict(report)
    payload.pop("audit_sha256")
    report["audit_sha256"] = content_sha256(payload)
    with pytest.raises(NativeContractError, match="stale|non-canonical"):
        validate_native_configuration_audit(report)


def test_configuration_audit_survives_json_round_trip():
    import json

    report = build_native_configuration_audit()
    reopened = json.loads(json.dumps(report, allow_nan=False))
    assert validate_native_configuration_audit(reopened) == reopened


def test_historical_parity_aggregate_is_not_current():
    with pytest.raises(NativeContractError, match="stale"):
        load_current_parity_aggregate(
            "output/native_parity_evidence_v8_a/native_parity_aggregate.json"
        )
