from copy import deepcopy
import inspect

import pytest

from futures_foundation.finetune import native_training_schema_v2 as schema
from futures_foundation.finetune.native_contracts import (
    NativeContractError,
    content_sha256,
    load_registry,
)
from futures_foundation.finetune.native_family_route_catalog_v2 import (
    load_family_route_catalog,
)
from futures_foundation.finetune.native_training_data_authority import (
    AUTHORITY_ID,
    training_data_authority_sha256,
)


def _canonical_parts(
    arm="chronos_bolt", track="F", route_id="direct_native_quantile_pinball",
):
    catalog = load_family_route_catalog()
    registry = load_registry()
    key = f"{arm}:{track}:{route_id}"
    route = catalog["routes"][key]
    dossier = registry["models"][arm]
    return key, route, dossier


def _template():
    key, route, dossier = _canonical_parts()
    tokenizer = dossier.get("tokenizer") or {}
    return {
        "schema_version": schema.ROUTE_TEMPLATE_SCHEMA,
        "template_status": route["status"],
        "blocker_tags": list(route["blocker_tags"]),
        "family_contract_sha256": schema.canonical_route_profile_sha256(
            route["arm_key"], route["track"], route["route_id"],
        ),
        "governance_evidence_sha256": content_sha256(dossier["license"]),
        "identity": {
            "arm_key": route["arm_key"],
            "track": route["track"],
            "route_id": route["route_id"],
            "pathway_kind": route["pathway_kind"],
            "task_kind": "quantile_forecast",
            "method_provenance": route["method_provenance"],
        },
        "base_binding": {
            "inference_dossier_sha256": content_sha256(dossier),
            "inference_evidence_id": dossier["tracks"]["F"]["evidence_id"],
            "model_revision": dossier["model_revision"],
            "source_revision": dossier["source_revision"],
            "tokenizer_revision": tokenizer.get("revision", "not_applicable"),
            "processor_revision": "not_applicable",
        },
        "input": {
            "layout_tag": "independent_univariate_passes",
            "axes": ["batch_times_channel", "time"],
            "context_length": 512,
            "horizon_length": 16,
            "parent_length": 528,
            "channel_order": ["open", "high", "low", "close", "volume"],
            "grouping_tag": "no_cross_channel_interaction",
            "dtype": "fp32",
            "missing_policy_tag": "native_missing_mask",
            "guard_policy_tag": "parent_interval_contained",
        },
        "time": {
            "horizon_unit": "bars",
            "timestamp_tag": "utc_bar_close",
            "venue_timezone_tag": "utc",
            "session_policy_tag": "bound_at_route_instance",
            "roll_policy_tag": "bound_at_route_instance",
        },
        "preprocessing": {
            "owner_tag": "upstream_internal",
            "scaler_tag": "bolt_native",
            "statistics_interval_tag": "context_only",
            "external_scaler_tag": "none",
            "mask_tag": "native_patch_mask",
        },
        "target": {
            "target_tag": "future_quantiles",
            "interval": [512, 528],
            "axes": ["batch_times_channel", "horizon"],
            "normalization_tag": "native_internal",
        },
        "objective": {
            "loss_tag": "quantile_pinball",
            "reduction_tag": "upstream_exact",
            "parameters": {
                "quantiles": [0.1, 0.5, 0.9],
                "native_prediction_length": 64,
                "effective_horizon": 16,
            },
        },
        "optimization": {
            "trainable_surface_tag": "full_model",
            "frozen_surface_tag": "none",
            "parameter_selector_sha256": "b" * 64,
            "precision_tag": "fp32",
            "optimizer_tag": "adamw",
            "scheduler_tag": "linear_warmup",
            "learning_rate": 1e-5,
            "weight_decay": 0.01,
            "betas": [0.9, 0.999],
            "epsilon": 1e-8,
            "batch_size": 8,
            "gradient_accumulation_steps": 1,
            "max_gradient_norm": 1.0,
        },
        "lineage": {
            "initialization_tag": "vanilla_pinned_checkpoint",
            "parent_bindings": [],
            "forbidden_parent_route_keys": [],
        },
        "resume": {
            "required_state": [
                "model", "optimizer", "scheduler", "scaler", "epoch", "global_step",
                "sampler", "python_rng", "numpy_rng", "torch_cpu_rng", "torch_cuda_rng",
            ],
            "exact_next_batch_required": True,
            "exact_next_loss_required": True,
        },
        "export": {
            "bundle_tag": "full_model_and_preprocessor",
            "output_tag": "quantile_forecast",
            "deployment_route_tag": "bolt_public_forecast",
            "preprocessing_hash_required": True,
            "output_slots": ["trained_model"],
        },
        "governance": {
            "license_id": dossier["license"]["id"],
            "permitted_use_scopes": list(route["permitted_use_scopes"]),
            "terms_evidence_sha256": content_sha256(dossier["license"]),
        },
        "smoke_profile_tag": "quantile_forecast_fp32_v1",
    }


def _unsupported_template():
    arm, track, route_id = "toto2_22m", "F", "no_released_toto2_finetuning"
    _, route, dossier = _canonical_parts(arm, track, route_id)
    return {
        "schema_version": schema.ROUTE_TEMPLATE_SCHEMA,
        "template_status": "blocked",
        "blocker_tags": list(route["blocker_tags"]),
        "family_contract_sha256": schema.canonical_route_profile_sha256(
            arm, track, route_id,
        ),
        "governance_evidence_sha256": content_sha256(dossier["license"]),
        "identity": {
            "arm_key": arm, "track": track, "route_id": route_id,
            "pathway_kind": "unsupported", "task_kind": "unsupported",
            "method_provenance": "unsupported",
        },
        "base_binding": {
            "inference_dossier_sha256": content_sha256(dossier),
            "inference_evidence_id": dossier["tracks"][track].get("evidence_id"),
            "model_revision": dossier["model_revision"],
            "source_revision": dossier["source_revision"],
            "tokenizer_revision": "not_applicable",
            "processor_revision": "not_applicable",
        },
        "governance": {
            "license_id": dossier["license"]["id"],
            "permitted_use_scopes": [],
            "terms_evidence_sha256": content_sha256(dossier["license"]),
        },
    }


def _instance_payload(template_sha, *, authority_hash=None):
    return {
        "schema_version": schema.ROUTE_INSTANCE_SCHEMA,
        "template_sha256": template_sha,
        "data_binding": {
            "training_data_authority_id": AUTHORITY_ID,
            "training_data_authority_sha256": (
                authority_hash or training_data_authority_sha256()
            ),
        },
        "template_evidence_bundle_sha256": "e" * 64,
    }


def _manual_instance(template, *, authority_hash=None):
    template_sha = content_sha256(schema.validate_route_template_candidate(template))
    payload = _instance_payload(template_sha, authority_hash=authority_hash)
    payload["instance_sha256"] = content_sha256(payload)
    return payload


def test_blocked_template_is_strict_deterministic_and_non_authorizing():
    template = _template()
    assert schema.validate_route_template_candidate(template) == template
    candidate_sha = content_sha256(schema.validate_route_template_candidate(template))
    assert candidate_sha == content_sha256(
        schema.validate_route_template_candidate(deepcopy(template))
    )
    with pytest.raises(NativeContractError, match="cannot materialize"):
        schema.validate_route_template(template)
    with pytest.raises(NativeContractError, match="cannot materialize"):
        schema.route_template_sha256(template)
    verified = deepcopy(template)
    verified["template_status"] = "verified"
    verified["blocker_tags"] = []
    with pytest.raises(NativeContractError, match="status/blockers"):
        schema.validate_route_template_candidate(verified)
    with pytest.raises(NativeContractError, match="blocked and non-authorizing"):
        schema.build_route_instance(
            _instance_payload(candidate_sha),
            templates_by_sha256={candidate_sha: template},
        )


def test_public_api_has_no_caller_supplied_family_or_governance_maps():
    for function in (
        schema.validate_route_template_candidate, schema.validate_route_template,
        schema.route_template_sha256,
        schema.validate_route_instance, schema.build_route_instance,
        schema.validate_pipeline_dag, schema.build_pipeline_dag, schema.evidence_key,
    ):
        parameters = inspect.signature(function).parameters
        assert "family_contracts_by_sha256" not in parameters
        assert "governance_evidence_by_sha256" not in parameters
    with pytest.raises(TypeError):
        schema.validate_route_template(
            _template(), family_contracts_by_sha256={},
            governance_evidence_by_sha256={},
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda t: t.__setitem__("family_contract_sha256", "0" * 64), "route/profile"),
        (lambda t: t.__setitem__("governance_evidence_sha256", "0" * 64), "governance"),
        (
            lambda t: t["base_binding"].__setitem__("inference_dossier_sha256", "0" * 64),
            "base binding",
        ),
        (lambda t: t["identity"].__setitem__("route_id", "invented_route"), "not declared"),
        (
            lambda t: t["identity"].__setitem__("method_provenance", "upstream_native"),
            "differs from canonical",
        ),
    ],
)
def test_template_rejects_self_authored_or_swapped_semantics(mutation, message):
    template = _template()
    mutation(template)
    with pytest.raises(NativeContractError, match=message):
        schema.validate_route_template_candidate(template)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda t: t["identity"].__setitem__("task_kind", "continuous_forecast"),
        lambda t: t["objective"].__setitem__(
            "loss_tag", "point_plus_quantile"
        ),
        lambda t: t["input"].__setitem__("layout_tag", "joint_multivariate"),
        lambda t: t["optimization"].__setitem__("trainable_surface_tag", "head_only"),
        lambda t: t["export"].__setitem__("output_tag", "continuous_forecast"),
    ],
)
def test_authoritative_template_rejects_task_objective_input_surface_and_export_swaps(
    mutation,
):
    template = _template()
    mutation(template)
    with pytest.raises(NativeContractError):
        schema.validate_route_template(template)


def test_structural_validation_still_rejects_shape_objective_resume_and_unknown_fields():
    template = _template()
    template["input"]["parent_length"] = 527
    with pytest.raises(NativeContractError, match=r"context_length \+ horizon_length"):
        schema.validate_route_template_candidate(template)
    template = _template()
    template["objective"]["parameters"]["effective_horizon"] = 15
    with pytest.raises(NativeContractError, match="must equal input horizon_length"):
        schema.validate_route_template_candidate(template)
    template = _template()
    template["resume"]["required_state"].remove("sampler")
    with pytest.raises(NativeContractError, match="complete exact-resume"):
        schema.validate_route_template_candidate(template)
    template = _template()
    template["surprise"] = True
    with pytest.raises(NativeContractError, match="unknown=.*surprise"):
        schema.validate_route_template_candidate(template)


def test_unsupported_route_is_exact_and_cannot_claim_an_alternate_contract():
    template = _unsupported_template()
    assert schema.validate_route_template_candidate(template) == template
    template["identity"]["route_id"] = "no_released_finetuning"
    with pytest.raises(NativeContractError, match="not declared"):
        schema.validate_route_template_candidate(template)


def test_route_instance_rejects_digest_bags_and_blocked_authority_before_admission():
    template = _template()
    template_sha = content_sha256(schema.validate_route_template_candidate(template))
    payload = _instance_payload(template_sha)
    payload["data_binding"]["sample_manifest_sha256"] = "2" * 64
    payload["instance_sha256"] = content_sha256(payload)
    with pytest.raises(NativeContractError, match="data_binding fields mismatch"):
        schema.validate_route_instance(
            payload, templates_by_sha256={template_sha: template},
        )
    payload = _instance_payload(template_sha, authority_hash="f" * 64)
    payload["instance_sha256"] = content_sha256(payload)
    with pytest.raises(NativeContractError, match="self-authored or stale"):
        schema.validate_route_instance(
            payload, templates_by_sha256={template_sha: template},
        )
    payload = _instance_payload(template_sha)
    payload["instance_sha256"] = content_sha256(payload)
    with pytest.raises(NativeContractError, match="blocked and non-authorizing"):
        schema.validate_route_instance(
            payload, templates_by_sha256={template_sha: template},
        )


def _pipeline_fixtures(monkeypatch, *, child_authority_hash=None):
    parent_template = _template()
    parent_sha = content_sha256(schema.validate_route_template_candidate(parent_template))
    child_template = deepcopy(parent_template)
    child_template["lineage"] = {
        "initialization_tag": "parent_route_artifact",
        "parent_bindings": [{
            "route_key": "chronos_bolt:F:direct_native_quantile_pinball",
            "template_sha256": parent_sha,
            "artifact_tag": "forecast_bundle",
            "child_input_slot": "parent_model",
        }],
        "forbidden_parent_route_keys": [],
    }
    child_sha = content_sha256(schema.validate_route_template_candidate(child_template))
    parent = _manual_instance(parent_template)
    child = _manual_instance(child_template, authority_hash=child_authority_hash)
    instances = {
        parent["instance_sha256"]: parent,
        child["instance_sha256"]: child,
    }
    templates = {parent_sha: parent_template, child_sha: child_template}

    def validate_fixture_instance(value, *, templates_by_sha256):
        assert value["template_sha256"] in templates_by_sha256
        assert value["instance_sha256"] == content_sha256({
            key: item for key, item in value.items() if key != "instance_sha256"
        })
        return deepcopy(value)

    monkeypatch.setattr(schema, "validate_route_instance", validate_fixture_instance)
    monkeypatch.setattr(
        schema, "validate_route_template", schema.validate_route_template_candidate,
    )
    manifest = {
        "schema_version": schema.PIPELINE_ARTIFACT_MANIFEST_SCHEMA,
        "artifact_sha256": "0" * 64,
        "artifact_tag": "forecast_bundle",
        "producer_route_instance_sha256": parent["instance_sha256"],
        "output_slot": "trained_model",
    }
    manifest_sha = content_sha256(manifest)
    payload = {
        "schema_version": schema.PIPELINE_SCHEMA,
        "pipeline_id": "two_stage_native_pipeline",
        "nodes": [
            {"node_id": "stage_one", "route_instance_sha256": parent["instance_sha256"]},
            {"node_id": "stage_two", "route_instance_sha256": child["instance_sha256"]},
        ],
        "edges": [{
            "parent_node_id": "stage_one", "child_node_id": "stage_two",
            "artifact_tag": "forecast_bundle", "artifact_sha256": "0" * 64,
            "manifest_sha256": manifest_sha, "parent_output_slot": "trained_model",
            "child_input_slot": "parent_model",
        }],
    }
    return payload, instances, templates, {manifest_sha: manifest}


def test_pipeline_binds_full_parent_route_template_and_exact_cardinality(monkeypatch):
    payload, instances, templates, manifests = _pipeline_fixtures(monkeypatch)
    pipeline = schema.build_pipeline_dag(
        payload, instances_by_sha256=instances, templates_by_sha256=templates,
        artifact_manifests_by_sha256=manifests,
    )
    assert schema.validate_pipeline_dag(
        pipeline, instances_by_sha256=instances, templates_by_sha256=templates,
        artifact_manifests_by_sha256=manifests,
    ) == pipeline

    wrong_route_templates = deepcopy(templates)
    child_sha = next(sha for sha, item in templates.items() if item["lineage"]["parent_bindings"])
    wrong_route_templates[child_sha]["lineage"]["parent_bindings"][0]["route_key"] = (
        "chronos_v1:F:direct_t5_token_forecast_cross_entropy"
    )
    with pytest.raises(NativeContractError, match="child lineage"):
        schema.build_pipeline_dag(
            payload, instances_by_sha256=instances,
            templates_by_sha256=wrong_route_templates,
            artifact_manifests_by_sha256=manifests,
        )

    wrong_sha_templates = deepcopy(templates)
    wrong_sha_templates[child_sha]["lineage"]["parent_bindings"][0][
        "template_sha256"
    ] = "f" * 64
    with pytest.raises(NativeContractError, match="child lineage"):
        schema.build_pipeline_dag(
            payload, instances_by_sha256=instances,
            templates_by_sha256=wrong_sha_templates,
            artifact_manifests_by_sha256=manifests,
        )

    duplicated = deepcopy(payload)
    duplicated["edges"].append(deepcopy(duplicated["edges"][0]))
    with pytest.raises(NativeContractError, match="duplicate edge"):
        schema.build_pipeline_dag(
            duplicated, instances_by_sha256=instances, templates_by_sha256=templates,
            artifact_manifests_by_sha256=manifests,
        )


def test_pipeline_requires_one_exact_data_binding_across_every_node(monkeypatch):
    payload, instances, templates, manifests = _pipeline_fixtures(
        monkeypatch, child_authority_hash="f" * 64,
    )
    with pytest.raises(NativeContractError, match="one exact data_binding"):
        schema.build_pipeline_dag(
            payload, instances_by_sha256=instances, templates_by_sha256=templates,
            artifact_manifests_by_sha256=manifests,
        )


def test_evidence_key_cannot_exist_for_a_blocked_catalog_route():
    template = _template()
    instance = _manual_instance(template)
    with pytest.raises(NativeContractError, match="cannot materialize"):
        schema.evidence_key(template, instance)
