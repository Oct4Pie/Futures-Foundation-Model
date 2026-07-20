"""Fail-closed execution-readiness audit for architecture-native training routes.

The family-route catalog describes methodology.  A launcher is executable code.  Smoke
artifacts are empirical evidence.  Training admission requires all three plus admitted data
and governance.  This module keeps those concepts separate and intentionally cannot grant
training admission.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import subprocess
from typing import Any, Mapping

from .native_contracts import NativeContractError, content_sha256, file_sha256, load_registry
from .native_downstream_screen import load_incremental_screen_report
from .native_family_route_catalog_v2 import catalog_sha256, load_family_route_catalog
from .native_route_pilot import load_route_pilot_evidence
from .native_route_smoke import load_route_smoke_evidence
from .native_smoke_contract import REQUIRED_SMOKE_CHECKS
from .native_training_data_authority import (
    load_training_data_authority,
    training_data_authority_sha256,
)
from .native_training_schema_v2 import canonical_route_profile_sha256


READINESS_SCHEMA = "ffm_native_training_readiness_v1"
READINESS_POLICY = "non_authorizing_fail_closed_preflight_v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

# Candidate code is listed only when it materially overlaps the catalog route.  None of these
# bindings is exact today: every candidate has at least one explicit methodological discrepancy.
# A route absent from this table has no concrete optimizer/in-context launcher in this checkout.
_EXACT_EXECUTORS: dict[str, dict[str, Any]] = {
    "chronos_bolt:F:direct_native_quantile_pinball": {
        "path": "futures_foundation/finetune/routes/chronos_bolt.py",
        "entrypoint": "native_loss/direct_quantiles",
        "execution_scope": "smoke_ready_pilot_data_unbound",
    },
    "chronos_v1:F:native_64_t5_token_forecast_cross_entropy": {
        "path": "futures_foundation/finetune/routes/chronos_v1.py",
        "entrypoint": "native_loss/forecast_samples",
        "execution_scope": "smoke_ready_pilot_data_unbound",
    },
    "chronos_v2:F:official_fit_full": {
        "path": "futures_foundation/finetune/routes/chronos2_native.py",
        "entrypoint": "native_loss/grouped_quantile/full",
        "execution_scope": "exact_executor_smoke_missing",
    },
    "chronos_v2:F:official_fit_lora": {
        "path": "futures_foundation/finetune/routes/chronos2_native.py",
        "entrypoint": "native_loss/grouped_quantile/lora",
        "execution_scope": "exact_executor_smoke_missing",
    },
    "moment_small:R:masked_patch_reconstruction": {
        "path": "futures_foundation/finetune/routes/moment_reconstruction.py",
        "entrypoint": "native_loss/mean_embedding",
        "execution_scope": "smoke_ready_pilot_data_unbound",
    },
    "moment_small:C:classification_full": {
        "path": "futures_foundation/finetune/routes/moment_tasks.py",
        "entrypoint": "native_loss/classification_logits/full",
        "execution_scope": "exact_executor_smoke_missing",
    },
    "moment_small:C:classification_head_only": {
        "path": "futures_foundation/finetune/routes/moment_tasks.py",
        "entrypoint": "native_loss/classification_logits/head",
        "execution_scope": "exact_executor_smoke_missing",
    },
    "moment_small:F:forecast_full_raw_mse": {
        "path": "futures_foundation/finetune/routes/moment_tasks.py",
        "entrypoint": "native_loss/raw_forecast/full",
        "execution_scope": "exact_executor_smoke_missing",
    },
    "moment_small:F:forecast_head_only_raw_mse": {
        "path": "futures_foundation/finetune/routes/moment_tasks.py",
        "entrypoint": "native_loss/raw_forecast/head",
        "execution_scope": "exact_executor_smoke_missing",
    },
    "kronos_mini:F:tokenizer_reconstruction_bsq": {
        "path": "futures_foundation/finetune/routes/kronos_tokenizer.py",
        "entrypoint": "native_loss/tokenizer_codes",
        "execution_scope": "smoke_ready_pilot_data_unbound",
    },
    "kronos_mini:F:hierarchical_autoregressive_tokens": {
        "path": "futures_foundation/finetune/routes/kronos_predictor.py",
        "entrypoint": "native_loss/public_greedy_forecast",
        "execution_scope": "smoke_ready_parent_bound_pilot_complete",
    },
    "kronos_small:F:tokenizer_reconstruction_bsq": {
        "path": "futures_foundation/finetune/routes/kronos_tokenizer.py",
        "entrypoint": "native_loss/tokenizer_codes",
        "execution_scope": "exact_executor_smoke_control_failed",
    },
    "kronos_small:F:hierarchical_autoregressive_tokens": {
        "path": "futures_foundation/finetune/routes/kronos_predictor.py",
        "entrypoint": "native_loss/public_greedy_forecast",
        "execution_scope": "exact_executor_parent_tokenizer_smoke_failed",
    },
    "mantis_v1:C:supervised_classification_full": {
        "path": "futures_foundation/finetune/routes/mantis_native.py",
        "entrypoint": "native_loss/classification_logits/full",
        "execution_scope": "exact_executor_smoke_missing",
    },
    "mantis_v1:C:supervised_classification_head": {
        "path": "futures_foundation/finetune/routes/mantis_native.py",
        "entrypoint": "native_loss/classification_logits/head",
        "execution_scope": "exact_executor_smoke_missing",
    },
    "mantis_v1:R:official_crop_resize_contrastive": {
        "path": "futures_foundation/finetune/routes/mantis_native.py",
        "entrypoint": "native_loss/official_crop_resize_contrastive",
        "execution_scope": "exact_executor_smoke_missing",
    },
    "mantis_v2:C:supervised_classification_full": {
        "path": "futures_foundation/finetune/routes/mantis_native.py",
        "entrypoint": "native_loss/classification_logits/full",
        "execution_scope": "exact_executor_smoke_missing",
    },
    "mantis_v2:C:supervised_classification_head": {
        "path": "futures_foundation/finetune/routes/mantis_native.py",
        "entrypoint": "native_loss/classification_logits/head",
        "execution_scope": "exact_executor_smoke_missing",
    },
    "mantis_v2:R:official_crop_resize_contrastive": {
        "path": "futures_foundation/finetune/routes/mantis_native.py",
        "entrypoint": "native_loss/official_crop_resize_contrastive",
        "execution_scope": "exact_executor_smoke_missing",
    },
    "ttm_r2:F:full_model_raw_hf_trainer_forecast": {
        "path": "futures_foundation/finetune/routes/ttm_native.py",
        "entrypoint": "native_loss/raw_forecast/full",
        "execution_scope": "exact_executor_smoke_missing",
    },
    "ttm_r2:F:head_prefix_raw_hf_trainer_forecast": {
        "path": "futures_foundation/finetune/routes/ttm_native.py",
        "entrypoint": "native_loss/raw_forecast/head_prefix",
        "execution_scope": "exact_executor_smoke_missing",
    },
    "timesfm25:F:official_lora_forecast": {
        "path": "futures_foundation/finetune/routes/timesfm_lora.py",
        "entrypoint": "native_loss/peft_lora_forecast",
        "execution_scope": "exact_executor_smoke_missing",
    },
    "moirai2_small:F:custom_scaled_pinball_research": {
        "path": "futures_foundation/finetune/routes/moirai2_research.py",
        "entrypoint": "native_loss/scaled_pinball_every_output_patch",
        "execution_scope": "exact_research_executor_smoke_missing",
    },
}


_CANDIDATE_LAUNCHERS: dict[str, dict[str, Any]] = {
    "mantis_v1:R:official_crop_resize_contrastive": {
        "path": "futures_foundation/finetune/ssl.py",
        "entrypoint": "loop_ssl",
        "discrepancies": [
            "legacy_universal_ssl_orchestrator_not_catalog_route",
            "official_one_way_mantis_v1_objective_not_isolated",
            "catalog_native_export_contract_not_emitted",
        ],
    },
    "mantis_v2:R:official_crop_resize_contrastive": {
        "path": "futures_foundation/finetune/ssl.py",
        "entrypoint": "loop_ssl",
        "discrepancies": [
            "legacy_universal_ssl_orchestrator_not_catalog_route",
            "official_mantis_v2_objective_not_isolated",
            "catalog_native_export_contract_not_emitted",
        ],
    },
    "mantis_v1:C:supervised_classification_full": {
        "path": "futures_foundation/finetune/classifiers/mantis/_torch.py",
        "entrypoint": "MantisFineTuneWorker",
        "discrepancies": [
            "no_standalone_catalog_route_launcher",
            "exact_resume_and_export_bundle_not_implemented",
        ],
    },
    "mantis_v1:C:supervised_classification_head": {
        "path": "futures_foundation/finetune/classifiers/mantis/_torch.py",
        "entrypoint": "MantisFineTuneWorker",
        "discrepancies": [
            "no_standalone_catalog_route_launcher",
            "head_only_surface_not_bound_to_route_instance",
            "exact_resume_and_export_bundle_not_implemented",
        ],
    },
    "mantis_v2:C:supervised_classification_full": {
        "path": "futures_foundation/finetune/classifiers/mantis/_torch.py",
        "entrypoint": "MantisFineTuneWorker",
        "discrepancies": [
            "no_standalone_catalog_route_launcher",
            "exact_resume_and_export_bundle_not_implemented",
        ],
    },
    "mantis_v2:C:supervised_classification_head": {
        "path": "futures_foundation/finetune/classifiers/mantis/_torch.py",
        "entrypoint": "MantisFineTuneWorker",
        "discrepancies": [
            "no_standalone_catalog_route_launcher",
            "head_only_surface_not_bound_to_route_instance",
            "exact_resume_and_export_bundle_not_implemented",
        ],
    },
    "moment_small:F:forecast_full_raw_mse": {
        "path": "scripts/train_moment_forecast.py",
        "entrypoint": "train",
        "discrepancies": [
            "requires_non_native_universal_stage2_parent",
            "uses_scale_normalized_mse_not_catalog_raw_mse",
            "launcher_route_id_differs_from_catalog_route_id",
        ],
    },
    "kronos_small:F:hierarchical_autoregressive_tokens": {
        "path": "scripts/train_kronos_tournament.py",
        "entrypoint": "train",
        "discrepancies": [
            "combined_tokenizer_predictor_launcher_not_single_catalog_route",
            "exact_native_scaler_and_source_fields_unresolved",
            "launcher_route_id_combines_multiple_catalog_routes",
        ],
    },
    "kronos_mini:R:adjacent_half_contrastive": {
        "path": "scripts/train_kronos_contrastive.py",
        "entrypoint": "train",
        "discrepancies": [
            "project_contrastive_profile_is_unresolved",
            "catalog_native_export_contract_not_emitted",
        ],
    },
    "kronos_small:R:adjacent_half_contrastive": {
        "path": "scripts/train_kronos_contrastive.py",
        "entrypoint": "train",
        "discrepancies": [
            "project_contrastive_profile_is_unresolved",
            "catalog_native_export_contract_not_emitted",
        ],
    },
    "chronos_v2:F:official_fit_full": {
        "path": "scripts/train_chronos_tournament.py",
        "entrypoint": "train",
        "discrepancies": [
            "requires_non_native_universal_stage_parent",
            "official_fit_api_not_used_as_catalog_declares",
            "launcher_admission_route_is_historical_universal_chain",
        ],
    },
    "timesfm25:F:official_lora_forecast": {
        "path": "scripts/train_timesfm_tournament.py",
        "entrypoint": "train",
        "discrepancies": [
            "requires_non_native_universal_stage_parent",
            "launcher_admission_route_is_historical_universal_chain",
            "official_fit_lora_contract_not_emitted",
        ],
    },
    "ttm_r2:F:full_model_raw_hf_trainer_forecast": {
        "path": "scripts/train_ttm_tournament.py",
        "entrypoint": "train",
        "discrepancies": [
            "requires_non_native_universal_stage_parent",
            "raw_hf_trainer_route_not_isolated",
            "launcher_admission_route_is_historical_universal_chain",
        ],
    },
    "moirai2_small:F:custom_scaled_pinball_research": {
        "path": "scripts/train_moirai2_tournament.py",
        "entrypoint": "train",
        "discrepancies": [
            "requires_non_native_universal_stage_parent",
            "launcher_admission_route_is_historical_universal_chain",
            "research_only_license_scope",
        ],
    },
    "sundial_base:F:custom_timeflow_research": {
        "path": "scripts/train_control_foundation_stages.py",
        "entrypoint": "train",
        "discrepancies": [
            "historical_three_stage_hidden_state_objective_is_not_timeflow",
            "upstream_finetuning_method_is_unresolved",
            "research_only_custom_method_not_catalog_complete",
        ],
    },
}

_CATALOG_ROUTE_DISPOSITIONS = load_family_route_catalog()["routes"]
_UNSUPPORTED_ROUTE_KEYS = frozenset(
    route_key
    for route_key, route in _CATALOG_ROUTE_DISPOSITIONS.items()
    if route["pathway_kind"] == "unsupported"
)
_IN_CONTEXT_ROUTE_KEYS = frozenset(
    route_key
    for route_key, route in _CATALOG_ROUTE_DISPOSITIONS.items()
    if route["pathway_kind"] == "in_context_fit"
)


def _git_state() -> dict[str, Any]:
    try:
        head = subprocess.check_output(
            ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"], text=True,
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", str(REPOSITORY_ROOT), "status", "--porcelain=v1"], text=True,
        ).splitlines()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise NativeContractError("cannot bind native-training readiness to Git state") from exc
    tracked_dirty = sorted(line for line in status if not line.startswith("??"))
    untracked = sorted(line[3:] for line in status if line.startswith("??"))
    untracked_python = sorted(path for path in untracked if path.endswith(".py"))
    return {
        "head_revision": head,
        "tracked_dirty": tracked_dirty,
        "untracked_paths": untracked,
        "untracked_python": untracked_python,
        "methodology_sealable": not tracked_dirty and not untracked_python,
    }


def _unresolved_constraints(profile: Mapping[str, Any]) -> list[dict[str, Any]]:
    unresolved: list[dict[str, Any]] = []
    for field, constraint in profile.items():
        if not isinstance(constraint, Mapping):
            continue
        state = constraint.get("state")
        if state == "unresolved":
            unresolved.append({
                "field": str(field),
                "state": "unresolved",
                "unresolved_fields": [],
            })
        elif state == "partial":
            value = constraint.get("value")
            fields = value.get("unresolved_fields", []) if isinstance(value, Mapping) else []
            unresolved.append({
                "field": str(field),
                "state": "partial",
                "unresolved_fields": list(fields) if isinstance(fields, list) else [],
            })
    return unresolved


def _launcher_record(route_key: str) -> dict[str, Any]:
    exact = _EXACT_EXECUTORS.get(route_key)
    if exact is not None:
        relative = Path(exact["path"])
        path = REPOSITORY_ROOT / relative
        if not path.is_file() or path.is_symlink():
            raise NativeContractError(f"exact route executor is missing or unsafe: {relative}")
        return {
            "status": "exact_route_executor",
            "path": relative.as_posix(),
            "entrypoint": exact["entrypoint"],
            "sha256": file_sha256(path),
            "exact_catalog_route": True,
            "execution_scope": exact["execution_scope"],
            "discrepancies": [],
        }
    if route_key in _UNSUPPORTED_ROUTE_KEYS:
        return {
            "status": "unsupported_by_canonical_method",
            "path": None,
            "entrypoint": None,
            "sha256": None,
            "exact_catalog_route": False,
            "execution_scope": "unsupported",
            "discrepancies": ["canonical_route_declares_no_supported_training_method"],
        }
    if route_key in _IN_CONTEXT_ROUTE_KEYS:
        return {
            "status": "in_context_route_not_training_launcher",
            "path": None,
            "entrypoint": None,
            "sha256": None,
            "exact_catalog_route": False,
            "execution_scope": "in_context_unbound",
            "discrepancies": ["exact_support_query_fit_executor_not_bound_to_route_instance"],
        }
    candidate = _CANDIDATE_LAUNCHERS.get(route_key)
    if candidate is None:
        return {
            "status": "not_implemented",
            "path": None,
            "entrypoint": None,
            "sha256": None,
            "exact_catalog_route": False,
            "execution_scope": "absent",
            "discrepancies": ["no_concrete_launcher_for_canonical_route"],
        }
    relative = Path(candidate["path"])
    path = REPOSITORY_ROOT / relative
    if not path.is_file() or path.is_symlink():
        raise NativeContractError(f"candidate launcher is missing or unsafe: {relative}")
    return {
        "status": "candidate_methodology_mismatch",
        "path": relative.as_posix(),
        "entrypoint": candidate["entrypoint"],
        "sha256": file_sha256(path),
        "exact_catalog_route": False,
        "execution_scope": "candidate_only",
        "discrepancies": list(candidate["discrepancies"]),
    }


def build_training_readiness_report(
    *,
    smoke_evidence_paths: Mapping[str, str | Path] | None = None,
    pilot_evidence_paths: Mapping[str, str | Path] | None = None,
    downstream_screen_paths: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Build a deterministic, non-authorizing report for every canonical route.

    Smoke evidence is opt-in and reverified from raw artifacts.  The default audit never
    discovers mutable output directories implicitly.
    """
    catalog = load_family_route_catalog()
    registry = load_registry()
    authority = load_training_data_authority()
    evidence_paths = dict(smoke_evidence_paths or {})
    pilot_paths = dict(pilot_evidence_paths or {})
    screen_paths = dict(downstream_screen_paths or {})
    unknown_evidence = sorted(set(evidence_paths) - set(catalog["routes"]))
    unknown_pilots = sorted(set(pilot_paths) - set(catalog["routes"]))
    unknown_screens = sorted(set(screen_paths) - set(catalog["routes"]))
    if unknown_evidence:
        raise NativeContractError(
            f"smoke evidence names unknown routes: {unknown_evidence}"
        )
    if unknown_pilots:
        raise NativeContractError(
            f"pilot evidence names unknown routes: {unknown_pilots}"
        )
    if unknown_screens:
        raise NativeContractError(
            f"downstream screen evidence names unknown routes: {unknown_screens}"
        )
    if set(pilot_paths) - set(evidence_paths):
        raise NativeContractError(
            "every supplied pilot evidence requires the same route's smoke evidence"
        )
    if set(screen_paths) - set(pilot_paths):
        raise NativeContractError(
            "every supplied downstream screen requires the same route's pilot evidence"
        )
    routes: dict[str, Any] = {}
    counts = {
        "catalog_routes": 0,
        "exact_executable_routes": 0,
        "exact_route_executor": 0,
        "smoke_admitted": 0,
        "pilot_completed": 0,
        "native_objective_survived": 0,
        "pilot_terminal_dispositions": 0,
        "downstream_screen_completed": 0,
        "downstream_terminal_dispositions": 0,
        "downstream_screen_survived": 0,
        "nonlinear_sensitivity_funded": 0,
        "candidate_methodology_mismatch": 0,
        "not_implemented": 0,
        "unsupported_by_canonical_method": 0,
        "in_context_route_not_training_launcher": 0,
        "training_admitted": 0,
    }
    for route_key, route in sorted(catalog["routes"].items()):
        arm_key = route["arm_key"]
        track = route["track"]
        route_id = route["route_id"]
        profile = catalog["constraint_profiles"][route["constraint_profile"]]
        launcher = _launcher_record(route_key)
        counts["catalog_routes"] += 1
        counts[launcher["status"]] += 1
        if launcher["exact_catalog_route"]:
            counts["exact_executable_routes"] += 1
        inference_track = registry["models"][arm_key]["tracks"][track]
        smoke = {
            "status": "missing",
            "required_checks": list(REQUIRED_SMOKE_CHECKS),
            "passed_checks": [],
            "evidence_path": None,
            "evidence_sha256": None,
        }
        resolved_route_blockers: set[str] = set()
        if launcher["exact_catalog_route"]:
            resolved_route_blockers.update({
                "optimization_hyperparameters_unresolved",
                "family_catalog_review_pending",
            })
        evidence_path = evidence_paths.get(route_key)
        if evidence_path is not None:
            evidence = load_route_smoke_evidence(evidence_path)
            if evidence["route_key"] != route_key:
                raise NativeContractError(
                    f"smoke evidence route mismatch for {route_key}"
                )
            if not launcher["exact_catalog_route"]:
                raise NativeContractError(
                    f"smoke evidence cannot bind a non-exact launcher: {route_key}"
                )
            executor = evidence["executor"]
            expected_executor = str((REPOSITORY_ROOT / launcher["path"]).resolve())
            if (
                executor["path"] != expected_executor
                or executor["sha256"] != launcher["sha256"]
                or executor["entrypoint"] != launcher["entrypoint"]
            ):
                raise NativeContractError(
                    f"smoke evidence executor differs from readiness for {route_key}"
                )
            passed_checks = [
                name for name in REQUIRED_SMOKE_CHECKS
                if evidence["checks"][name]["status"] == "pass"
            ]
            failed_checks = [
                name for name in REQUIRED_SMOKE_CHECKS
                if evidence["checks"][name]["status"] == "fail"
            ]
            if len(passed_checks) + len(failed_checks) != len(REQUIRED_SMOKE_CHECKS):
                raise NativeContractError(
                    f"smoke evidence check closure is incomplete for {route_key}"
                )
            if evidence["smoke_admitted"] is True:
                if passed_checks != list(REQUIRED_SMOKE_CHECKS) or failed_checks:
                    raise NativeContractError(
                        f"admitted smoke evidence contains failed checks for {route_key}"
                    )
                smoke_status = "pass"
                counts["smoke_admitted"] += 1
                resolved_route_blockers.update({
                    "raw_route_evidence_missing", "exact_resume_unverified", "export_unverified",
                })
            else:
                if not failed_checks:
                    raise NativeContractError(
                        f"failed smoke evidence contains no failed check for {route_key}"
                    )
                smoke_status = "fail"
            smoke = {
                "status": smoke_status,
                "required_checks": list(REQUIRED_SMOKE_CHECKS),
                "passed_checks": passed_checks,
                "evidence_path": str(Path(evidence_path).expanduser().resolve()),
                "evidence_sha256": evidence["evidence_sha256"],
            }
        pilot = {
            "status": "not_started",
            "evidence_path": None,
            "evidence_sha256": None,
            "native_objective_survived": False,
            "promotion_admitted": False,
        }
        pilot_path = pilot_paths.get(route_key)
        pilot_evidence = None
        pilot_blockers = ["pilot_evidence_missing"]
        if pilot_path is None:
            if route["task_kind"] == "classification" and launcher["exact_catalog_route"]:
                pilot["status"] = "blocked_missing_governed_labels"
                pilot_blockers = ["governed_classification_label_authority_missing"]
            elif launcher["execution_scope"] == "exact_executor_smoke_control_failed":
                pilot["status"] = "blocked_smoke_control_failed"
                pilot_blockers = ["route_smoke_control_failed"]
            elif launcher["execution_scope"] == "exact_executor_parent_tokenizer_smoke_failed":
                pilot["status"] = "blocked_parent_route_failed"
                pilot_blockers = ["parent_route_smoke_failed"]
            elif launcher["status"] == "unsupported_by_canonical_method":
                pilot["status"] = "not_applicable_unsupported"
                pilot_blockers = []
            elif launcher["status"] == "in_context_route_not_training_launcher":
                pilot["status"] = "not_applicable_in_context"
                pilot_blockers = []
        if pilot_path is not None:
            pilot_evidence = load_route_pilot_evidence(pilot_path)
            if pilot_evidence["route_key"] != route_key:
                raise NativeContractError(
                    f"pilot evidence route mismatch for {route_key}"
                )
            if smoke["status"] != "pass":
                raise NativeContractError(
                    f"pilot evidence cannot exist without passing smoke: {route_key}"
                )
            if (
                pilot_evidence["smoke_evidence"]["evidence_sha256"]
                != smoke["evidence_sha256"]
            ):
                raise NativeContractError(
                    f"pilot/smoke evidence lineage differs for {route_key}"
                )
            pilot = {
                "status": (
                    "native_objective_pass"
                    if pilot_evidence["native_objective_survived"]
                    else "native_objective_eliminated"
                ),
                "evidence_path": str(Path(pilot_path).expanduser().resolve()),
                "evidence_sha256": pilot_evidence["evidence_sha256"],
                "native_objective_survived": bool(
                    pilot_evidence["native_objective_survived"]
                ),
                "promotion_admitted": False,
            }
            counts["pilot_completed"] += 1
            if pilot["native_objective_survived"]:
                counts["native_objective_survived"] += 1
            pilot_blockers = list(pilot_evidence["promotion_blockers"])
        downstream = {
            "status": "not_started",
            "evidence_path": None,
            "evidence_sha256": None,
            "model_control_wins": 0,
            "incremental_point_wins": 0,
            "residual_bonferroni_ci_wins": 0,
            "primary_point_degradations": 0,
            "downstream_screen_survived": False,
            "nonlinear_sensitivity_funded": False,
            "full_training_admitted": False,
        }
        screen_blockers: list[str] = []
        screen_path = screen_paths.get(route_key)
        if screen_path is None:
            if pilot["status"] == "native_objective_eliminated":
                downstream["status"] = "not_applicable_pilot_eliminated"
                screen_blockers = ["native_objective_pilot_eliminated"]
            elif pilot["status"].startswith("blocked_"):
                downstream["status"] = "not_applicable_pilot_blocked"
                screen_blockers = ["pilot_gate_blocked"]
            elif pilot["status"].startswith("not_applicable_"):
                downstream["status"] = "not_applicable_route"
            elif (
                route_key == "kronos_mini:F:tokenizer_reconstruction_bsq"
                and pilot["native_objective_survived"]
            ):
                downstream["status"] = "not_applicable_common_view_mismatch"
                screen_blockers = ["common_information_512_view_mismatch"]
        if screen_path is not None:
            if pilot_evidence is None or not pilot["native_objective_survived"]:
                raise NativeContractError(
                    f"downstream screen requires a surviving native pilot: {route_key}"
                )
            screen_report = load_incremental_screen_report(screen_path)
            if screen_report["route_key"] != route_key:
                raise NativeContractError(
                    f"downstream screen route mismatch for {route_key}"
                )
            parent = screen_report["verified_feature_pilot_evidence"]
            if parent["content_sha256"] != pilot["evidence_sha256"]:
                raise NativeContractError(
                    f"downstream screen/pilot lineage differs for {route_key}"
                )
            verdict = screen_report["screen_verdict"]
            downstream = {
                "status": (
                    "screen_passed"
                    if verdict["downstream_screen_survived"]
                    else "screen_failed"
                ),
                "evidence_path": str(Path(screen_path).expanduser().resolve()),
                "evidence_sha256": screen_report["report_identity"]["sha256"],
                "model_control_wins": int(verdict["model_control_wins"]),
                "incremental_point_wins": int(verdict["incremental_point_wins"]),
                "residual_bonferroni_ci_wins": int(
                    verdict["residual_bonferroni_ci_wins"]
                ),
                "primary_point_degradations": int(
                    verdict["primary_point_degradations"]
                ),
                "downstream_screen_survived": bool(
                    verdict["downstream_screen_survived"]
                ),
                "nonlinear_sensitivity_funded": bool(
                    verdict["nonlinear_sensitivity_funded"]
                ),
                "full_training_admitted": False,
            }
            counts["downstream_screen_completed"] += 1
            if downstream["downstream_screen_survived"]:
                counts["downstream_screen_survived"] += 1
            if downstream["nonlinear_sensitivity_funded"]:
                counts["nonlinear_sensitivity_funded"] += 1
            pilot_blockers = [
                blocker for blocker in pilot_blockers
                if blocker not in {
                    "causal_feature_baseline_missing",
                    "residual_over_causal_gate_missing",
                }
            ]
            screen_blockers = [
                "nonlinear_sensitivity_evidence_missing"
                if downstream["downstream_screen_survived"]
                else "downstream_incremental_screen_failed"
            ]
        blockers = sorted(set(
            [
                blocker for blocker in route["blocker_tags"]
                if blocker not in resolved_route_blockers
            ]
            + list(authority["blocker_tags"])
            + launcher["discrepancies"]
            + ([] if smoke["status"] == "pass" else ["route_smoke_evidence_missing"])
            + pilot_blockers
            + screen_blockers
        ))
        if launcher["exact_catalog_route"] and pilot["status"] != "not_started":
            counts["pilot_terminal_dispositions"] += 1
        if launcher["exact_catalog_route"] and downstream["status"] != "not_started":
            counts["downstream_terminal_dispositions"] += 1
        routes[route_key] = {
            "identity": {
                "arm_key": arm_key,
                "track": track,
                "route_id": route_id,
                "task_kind": route["task_kind"],
                "pathway_kind": route["pathway_kind"],
                "method_provenance": route["method_provenance"],
            },
            "catalog": {
                "status": route["status"],
                "constraint_profile": route["constraint_profile"],
                "route_profile_sha256": canonical_route_profile_sha256(
                    arm_key, track, route_id,
                ),
                "unresolved_constraints": _unresolved_constraints(profile),
            },
            "inference_contract": {
                "status": inference_track["status"],
                "training_supported": bool(inference_track.get("training_supported", False)),
                "evidence_id": inference_track.get("evidence_id"),
            },
            "launcher": launcher,
            "smoke": smoke,
            "pilot": pilot,
            "downstream": downstream,
            "training_admitted": False,
            "blockers": blockers,
        }
    report = {
        "schema_version": READINESS_SCHEMA,
        "policy": READINESS_POLICY,
        "catalog_sha256": catalog_sha256(catalog),
        "training_data_authority": {
            "authority_id": authority["authority_id"],
            "status": authority["status"],
            "non_authorizing": authority["non_authorizing"],
            "sha256": training_data_authority_sha256(authority),
            "blocker_tags": list(authority["blocker_tags"]),
        },
        "methodology": _git_state(),
        "required_smoke_checks": list(REQUIRED_SMOKE_CHECKS),
        "counts": counts,
        "routes": routes,
        "all_exact_routes_pilot_dispositioned": all(
            row["pilot"]["status"] != "not_started"
            for row in routes.values()
            if row["launcher"]["exact_catalog_route"]
        ),
        "all_surviving_pilots_downstream_dispositioned": bool([
            row for row in routes.values()
            if row["pilot"]["native_objective_survived"]
        ]) and all(
            row["downstream"]["status"] != "not_started"
            for row in routes.values()
            if row["pilot"]["native_objective_survived"]
        ),
        "pilot_admitted": False,
        "training_admitted": False,
        "live_trading_ready": False,
    }
    report["readiness_sha256"] = content_sha256(report)
    return report


def validate_training_readiness_report(
    value: Any,
    *,
    smoke_evidence_paths: Mapping[str, str | Path] | None = None,
    pilot_evidence_paths: Mapping[str, str | Path] | None = None,
    downstream_screen_paths: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Accept only an exact report for the current catalog/code/data-authority state."""
    if not isinstance(value, Mapping):
        raise NativeContractError("native-training readiness report must be a mapping")
    candidate = deepcopy(dict(value))
    supplied = candidate.pop("readiness_sha256", None)
    if supplied != content_sha256(candidate):
        raise NativeContractError("native-training readiness report integrity mismatch")
    expected = build_training_readiness_report(
        smoke_evidence_paths=smoke_evidence_paths,
        pilot_evidence_paths=pilot_evidence_paths,
        downstream_screen_paths=downstream_screen_paths,
    )
    if dict(value) != expected:
        raise NativeContractError("native-training readiness report is stale or non-canonical")
    if any(
        bool(value.get(field))
        for field in ("pilot_admitted", "training_admitted", "live_trading_ready")
    ):
        raise NativeContractError("readiness audit cannot grant pilot, training, or trading admission")
    return deepcopy(expected)


def readiness_sha256(
    value: Any | None = None,
    *,
    smoke_evidence_paths: Mapping[str, str | Path] | None = None,
    pilot_evidence_paths: Mapping[str, str | Path] | None = None,
    downstream_screen_paths: Mapping[str, str | Path] | None = None,
) -> str:
    report = (
        build_training_readiness_report(
            smoke_evidence_paths=smoke_evidence_paths,
            pilot_evidence_paths=pilot_evidence_paths,
            downstream_screen_paths=downstream_screen_paths,
        )
        if value is None
        else validate_training_readiness_report(
            value,
            smoke_evidence_paths=smoke_evidence_paths,
            pilot_evidence_paths=pilot_evidence_paths,
            downstream_screen_paths=downstream_screen_paths,
        )
    )
    return str(report["readiness_sha256"])


__all__ = [
    "READINESS_POLICY",
    "READINESS_SCHEMA",
    "REQUIRED_SMOKE_CHECKS",
    "build_training_readiness_report",
    "readiness_sha256",
    "validate_training_readiness_report",
]
