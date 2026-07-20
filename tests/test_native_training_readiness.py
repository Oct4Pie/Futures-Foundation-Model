from copy import deepcopy
import importlib.metadata
from pathlib import Path
import re

import pytest

from futures_foundation.finetune.native_contracts import (
    NativeContractError,
    content_sha256,
)
from futures_foundation.finetune.native_route_smoke import build_route_smoke_evidence
from futures_foundation.finetune.native_training_readiness import (
    REQUIRED_SMOKE_CHECKS,
    build_training_readiness_report,
    validate_training_readiness_report,
)
from futures_foundation.finetune.native_route_smoke import REQUIRED_SMOKE_ARTIFACTS
from futures_foundation.finetune.native_contracts import get_dossier
from futures_foundation.finetune.routes import chronos_bolt


def _rehash(report):
    value = deepcopy(report)
    value.pop("readiness_sha256", None)
    value["readiness_sha256"] = content_sha256(value)
    return value


def test_readiness_audits_every_route_without_granting_admission():
    report = validate_training_readiness_report(build_training_readiness_report())
    counts = report["counts"]
    assert counts == {
        "catalog_routes": 42,
        "exact_executable_routes": 23,
        "exact_route_executor": 23,
        "smoke_admitted": 0,
        "pilot_completed": 0,
        "native_objective_survived": 0,
        "pilot_terminal_dispositions": 8,
        "downstream_screen_completed": 0,
        "downstream_terminal_dispositions": 8,
        "downstream_screen_survived": 0,
        "nonlinear_sensitivity_funded": 0,
        "candidate_methodology_mismatch": 0,
        "not_implemented": 0,
        "unsupported_by_canonical_method": 17,
        "in_context_route_not_training_launcher": 2,
        "training_admitted": 0,
    }
    assert len(report["routes"]) == 42
    assert report["training_data_authority"]["status"] == "blocked"
    assert report["training_data_authority"]["non_authorizing"] is True
    assert report["all_exact_routes_pilot_dispositioned"] is False
    assert report["all_surviving_pilots_downstream_dispositioned"] is False
    assert report["pilot_admitted"] is False
    assert report["training_admitted"] is False
    assert report["live_trading_ready"] is False
    assert re.fullmatch(r"[0-9a-f]{64}", report["readiness_sha256"])
    assert re.fullmatch(r"[0-9a-f]{40}", report["methodology"]["head_revision"])

    for route in report["routes"].values():
        assert route["training_admitted"] is False
        assert route["smoke"] == {
            "status": "missing",
            "required_checks": list(REQUIRED_SMOKE_CHECKS),
            "passed_checks": [],
            "evidence_path": None,
            "evidence_sha256": None,
        }
        pilot_status = route["pilot"]["status"]
        launcher_status = route["launcher"]["status"]
        execution_scope = route["launcher"]["execution_scope"]
        if (
            route["identity"]["task_kind"] == "classification"
            and route["launcher"]["exact_catalog_route"]
        ):
            assert pilot_status == "blocked_missing_governed_labels"
        elif execution_scope == "exact_executor_smoke_control_failed":
            assert pilot_status == "blocked_smoke_control_failed"
        elif execution_scope == "exact_executor_parent_tokenizer_smoke_failed":
            assert pilot_status == "blocked_parent_route_failed"
        elif launcher_status == "unsupported_by_canonical_method":
            assert pilot_status == "not_applicable_unsupported"
        elif launcher_status == "in_context_route_not_training_launcher":
            assert pilot_status == "not_applicable_in_context"
        else:
            assert pilot_status == "not_started"
        assert route["pilot"]["evidence_path"] is None
        assert route["pilot"]["evidence_sha256"] is None
        assert route["pilot"]["native_objective_survived"] is False
        assert route["pilot"]["promotion_admitted"] is False
        expected_downstream = (
            "not_applicable_pilot_blocked"
            if pilot_status.startswith("blocked_")
            else "not_applicable_route"
            if pilot_status.startswith("not_applicable_")
            else "not_started"
        )
        assert route["downstream"]["status"] == expected_downstream
        assert route["downstream"]["evidence_path"] is None
        assert route["downstream"]["evidence_sha256"] is None
        assert route["downstream"]["downstream_screen_survived"] is False
        assert route["downstream"]["nonlinear_sensitivity_funded"] is False
        assert route["downstream"]["full_training_admitted"] is False
        assert route["blockers"]
    for route_key in (
        "chronos_bolt:F:direct_native_quantile_pinball",
        "chronos_v1:F:native_64_t5_token_forecast_cross_entropy",
        "moment_small:R:masked_patch_reconstruction",
        "kronos_mini:F:tokenizer_reconstruction_bsq",
        "kronos_mini:F:hierarchical_autoregressive_tokens",
        "kronos_small:F:tokenizer_reconstruction_bsq",
    ):
        exact = report["routes"][route_key]
        assert exact["launcher"]["exact_catalog_route"] is True
        assert exact["launcher"]["status"] == "exact_route_executor"
    assert sum(
        route["launcher"]["exact_catalog_route"]
        for route in report["routes"].values()
    ) == 23


def test_readiness_distinguishes_candidate_unsupported_in_context_and_missing():
    routes = build_training_readiness_report()["routes"]
    assert routes[
        "chronos_bolt:F:direct_native_quantile_pinball"
    ]["launcher"]["status"] == "exact_route_executor"
    assert routes[
        "moment_small:R:masked_patch_reconstruction"
    ]["launcher"]["status"] == "exact_route_executor"
    assert routes[
        "mantis_v1:R:official_crop_resize_contrastive"
    ]["launcher"]["status"] == "exact_route_executor"
    assert routes[
        "kronos_mini:F:tokenizer_reconstruction_bsq"
    ]["launcher"]["status"] == "exact_route_executor"
    assert routes[
        "kronos_small:F:tokenizer_reconstruction_bsq"
    ]["launcher"]["execution_scope"] == "exact_executor_smoke_control_failed"
    assert routes[
        "toto2_22m:F:no_released_toto2_finetuning"
    ]["launcher"]["status"] == "unsupported_by_canonical_method"
    assert routes[
        "tabpfn_ts3_forecast:F:official_ts3_support_query_forecast"
    ]["launcher"]["status"] == "in_context_route_not_training_launcher"
    assert routes[
        "moment_small:C:classification_head_only"
    ]["launcher"]["status"] == "exact_route_executor"


def _passing_smoke(tmp_path, *, failed_check=None):
    revision = get_dossier("chronos_bolt")["model_revision"]
    snapshot = tmp_path / revision
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}\n")
    artifacts = {
        "model_snapshot": snapshot,
        "source_runtime": Path(
            importlib.metadata.distribution("chronos-forecasting")._path
        ).resolve(),
    }
    for name in sorted(
        REQUIRED_SMOKE_ARTIFACTS - {"model_snapshot", "source_runtime"}
    ):
        path = tmp_path / f"{name}.bin"
        path.write_bytes(name.encode())
        artifacts[name] = path
    checks = {
        name: {"status": "pass", "metrics": {"measured": True}, "reason": None}
        for name in REQUIRED_SMOKE_CHECKS
    }
    if failed_check is not None:
        checks[failed_check] = {
            "status": "fail",
            "metrics": {"measured": True},
            "reason": "fixture control failed",
        }
    evidence = build_route_smoke_evidence(
        route_key="chronos_bolt:F:direct_native_quantile_pinball",
        executor_path=chronos_bolt.__file__,
        executor_entrypoint="native_loss/direct_quantiles",
        checks=checks,
        artifacts=artifacts,
        metrics={"fixture": True},
        created_utc="2026-07-18T00:00:00Z",
    )
    path = tmp_path / "smoke.json"
    path.write_text(__import__("json").dumps(evidence))
    return path


def test_readiness_reverifies_supplied_smoke_without_granting_training(tmp_path):
    path = _passing_smoke(tmp_path)
    mapping = {"chronos_bolt:F:direct_native_quantile_pinball": path}
    report = build_training_readiness_report(smoke_evidence_paths=mapping)
    report = validate_training_readiness_report(
        report, smoke_evidence_paths=mapping,
    )
    assert report["counts"]["smoke_admitted"] == 1
    route = report["routes"]["chronos_bolt:F:direct_native_quantile_pinball"]
    assert route["smoke"]["status"] == "pass"
    assert route["smoke"]["passed_checks"] == list(REQUIRED_SMOKE_CHECKS)
    assert route["training_admitted"] is False
    assert report["pilot_admitted"] is False
    assert report["training_admitted"] is False


def test_readiness_records_failed_smoke_without_admitting_it(tmp_path):
    path = _passing_smoke(tmp_path, failed_check="shuffle_control_rejection")
    mapping = {"chronos_bolt:F:direct_native_quantile_pinball": path}
    report = build_training_readiness_report(smoke_evidence_paths=mapping)
    report = validate_training_readiness_report(
        report, smoke_evidence_paths=mapping,
    )
    assert report["counts"]["smoke_admitted"] == 0
    route = report["routes"]["chronos_bolt:F:direct_native_quantile_pinball"]
    assert route["smoke"]["status"] == "fail"
    assert "shuffle_control_rejection" not in route["smoke"]["passed_checks"]
    assert route["pilot"]["status"] == "not_started"
    assert route["training_admitted"] is False


def test_readiness_rejects_admission_forgery_even_after_rehash():
    forged = build_training_readiness_report()
    forged["training_admitted"] = True
    forged["counts"]["training_admitted"] = 42
    forged = _rehash(forged)
    with pytest.raises(NativeContractError, match="stale|cannot grant"):
        validate_training_readiness_report(forged)


def test_readiness_rejects_launcher_or_smoke_forgery_after_rehash():
    forged = build_training_readiness_report()
    route = forged["routes"]["mantis_v1:R:official_crop_resize_contrastive"]
    route["launcher"]["exact_catalog_route"] = True
    route["launcher"]["status"] = "exact"
    route["smoke"]["status"] = "pass"
    route["smoke"]["passed_checks"] = list(REQUIRED_SMOKE_CHECKS)
    route["training_admitted"] = True
    forged = _rehash(forged)
    with pytest.raises(NativeContractError, match="stale"):
        validate_training_readiness_report(forged)
