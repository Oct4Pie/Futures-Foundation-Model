from copy import deepcopy

import pytest

from futures_foundation.finetune.native_contracts import NativeContractError, content_sha256
from futures_foundation.finetune.native_training_schema_v2 import (
    PIPELINE_ARTIFACT_MANIFEST_SCHEMA,
    PIPELINE_SCHEMA,
    ROUTE_INSTANCE_SCHEMA,
    ROUTE_TEMPLATE_SCHEMA,
    build_pipeline_dag as _build_pipeline_dag,
    build_route_instance as _build_route_instance,
    evidence_key as _evidence_key,
    route_template_sha256 as _route_template_sha256,
    validate_pipeline_dag as _validate_pipeline_dag,
    validate_route_instance as _validate_route_instance,
    validate_route_template as _validate_route_template,
)


H = "a" * 64
FAMILY_CONTRACT = {"schema_version": "fixture_family_contract_v1", "family": "chronos_bolt"}
GOVERNANCE_EVIDENCE = {"schema_version": "fixture_governance_v1", "license": "Apache-2.0"}
FAMILY_SHA = content_sha256(FAMILY_CONTRACT)
GOVERNANCE_SHA = content_sha256(GOVERNANCE_EVIDENCE)
FAMILY_MAP = {FAMILY_SHA: FAMILY_CONTRACT}
GOVERNANCE_MAP = {GOVERNANCE_SHA: GOVERNANCE_EVIDENCE}


def _template():
    return {
        "schema_version": ROUTE_TEMPLATE_SCHEMA,
        "template_status": "verified",
        "blocker_tags": [],
        "family_contract_sha256": FAMILY_SHA,
        "governance_evidence_sha256": GOVERNANCE_SHA,
        "identity": {
            "arm_key": "chronos_bolt",
            "track": "F",
            "route_id": "native_quantile_forecast",
            "pathway_kind": "optimizer_training",
            "task_kind": "quantile_forecast",
            "method_provenance": "native_derived",
        },
        "base_binding": {
            "inference_dossier_sha256": H,
            "inference_evidence_id": "chronos_bolt:F:fixture",
            "model_revision": "model-revision",
            "source_revision": "source-revision",
            "tokenizer_revision": "bundled",
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
            "learning_rate": 0.00001,
            "weight_decay": 0.01,
            "betas": [0.9, 0.999],
            "epsilon": 1e-8,
            "batch_size": 8,
            "gradient_accumulation_steps": 1,
            "max_gradient_norm": 1.0,
        },
        "lineage": {
            "initialization_tag": "vanilla_pinned_checkpoint",
            "allowed_parent_route_tags": [],
            "forbidden_parent_route_tags": ["generic_stage_1", "generic_stage_2"],
            "parent_artifact_requirements": [],
            "parent_input_slots": [],
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
            "license_id": "Apache-2.0",
            "permitted_use_scopes": ["production", "research_noncommercial"],
            "terms_evidence_sha256": "c" * 64,
        },
        "smoke_profile_tag": "quantile_forecast_fp32_v1",
    }


def validate_route_template(value):
    return _validate_route_template(
        value, family_contracts_by_sha256=FAMILY_MAP,
        governance_evidence_by_sha256=GOVERNANCE_MAP,
    )


def route_template_sha256(value):
    return _route_template_sha256(
        value, family_contracts_by_sha256=FAMILY_MAP,
        governance_evidence_by_sha256=GOVERNANCE_MAP,
    )


def build_route_instance(value, *, templates_by_sha256=None):
    templates = templates_by_sha256 or {route_template_sha256(_template()): _template()}
    return _build_route_instance(
        value, templates_by_sha256=templates,
        family_contracts_by_sha256=FAMILY_MAP,
        governance_evidence_by_sha256=GOVERNANCE_MAP,
    )


def validate_route_instance(value, *, templates_by_sha256):
    return _validate_route_instance(
        value, templates_by_sha256=templates_by_sha256,
        family_contracts_by_sha256=FAMILY_MAP,
        governance_evidence_by_sha256=GOVERNANCE_MAP,
    )


def _instance_payload(template_sha=None, *, sample_hash="2" * 64):
    return {
        "schema_version": ROUTE_INSTANCE_SCHEMA,
        "template_sha256": template_sha or route_template_sha256(_template()),
        "data_binding": {
            "corpus_contract_sha256": "1" * 64,
            "sample_manifest_sha256": sample_hash,
            "split_contract_sha256": "3" * 64,
            "session_denominator_sha256": "4" * 64,
            "expected_request_denominator_sha256": "5" * 64,
            "lifecycle_registry_sha256": "6" * 64,
            "roll_registry_sha256": "7" * 64,
            "label_contract_sha256": "8" * 64,
            "allowed_training_splits": ["pretrain", "shared_train"],
            "allowed_validation_splits": ["development"],
            "forbidden_splits": ["legacy_holdout", "prospective_holdout"],
            "timeframes_minutes": [1, 3, 5, 15, 30, 60],
        },
        "exact_choices": {
            "amount_source_tag": "not_applicable",
            "venue_timezone_registry_sha256": "9" * 64,
            "session_calendar_registry_sha256": "d" * 64,
        },
        "template_evidence_bundle_sha256": "e" * 64,
    }


def test_strict_route_template_and_hash_are_deterministic():
    template = _template()
    assert validate_route_template(template) == template
    assert route_template_sha256(template) == route_template_sha256(deepcopy(template))


@pytest.mark.parametrize("location", ["top", "nested"])
def test_route_template_rejects_unknown_fields(location):
    template = _template()
    target = template if location == "top" else template["input"]
    target["surprise"] = True
    with pytest.raises(NativeContractError, match="unknown=.*surprise"):
        validate_route_template(template)


def test_route_template_rejects_free_text_instead_of_a_tag():
    template = _template()
    template["input"]["grouping_tag"] = "the channels should probably stay separate"
    with pytest.raises(NativeContractError, match="invalid tag"):
        validate_route_template(template)


def test_route_template_rejects_future_in_parent_shape_and_scaling():
    template = _template()
    template["input"]["parent_length"] = 527
    with pytest.raises(NativeContractError, match=r"context_length \+ horizon_length"):
        validate_route_template(template)
    template = _template()
    template["preprocessing"]["statistics_interval_tag"] = "full_parent"
    with pytest.raises(NativeContractError, match="invalid tag"):
        validate_route_template(template)
    template = _template()
    template["input"]["grouping_tag"] = "same_parent_joint_channels"
    with pytest.raises(NativeContractError, match="requires grouping"):
        validate_route_template(template)
    template = _template()
    template["objective"]["parameters"]["effective_horizon"] = 15
    with pytest.raises(NativeContractError, match="must equal input horizon_length"):
        validate_route_template(template)


def test_optimizer_route_requires_complete_exact_resume():
    template = _template()
    template["resume"]["required_state"].remove("sampler")
    with pytest.raises(NativeContractError, match="complete exact-resume"):
        validate_route_template(template)


@pytest.mark.parametrize(("field", "value"), [
    ("learning_rate", 0.0), ("epsilon", 0.0),
])
def test_optimizer_route_rejects_zero_numerics(field, value):
    template = _template()
    template["optimization"][field] = value
    with pytest.raises(NativeContractError, match="positive learning_rate and epsilon"):
        validate_route_template(template)


def test_objective_weights_missing_id_and_surface_pairs_fail_closed():
    template = _template()
    template["optimization"]["frozen_surface_tag"] = "encoder"
    with pytest.raises(NativeContractError, match="invalid trainable/frozen surface pair"):
        validate_route_template(template)
    template = _template()
    template["objective"] = {
        "loss_tag": "point_plus_quantile", "reduction_tag": "upstream_exact",
        "parameters": {
            "quantiles": [0.1, 0.5, 0.9], "point_weight": 0.0,
            "quantile_weight": 1.0, "effective_horizon": 16,
        },
    }
    with pytest.raises(NativeContractError, match="weights must be positive"):
        validate_route_template(template)
    template = _template()
    template["objective"] = {
        "loss_tag": "token_cross_entropy", "reduction_tag": "upstream_exact",
        "parameters": {
            "eos_policy_tag": "immediately_after_horizon", "missing_label_id": -1,
            "effective_horizon": 16,
        },
    }
    with pytest.raises(NativeContractError, match="missing_label_id=-100"):
        validate_route_template(template)
    template = _template()
    template["objective"] = {
        "loss_tag": "path_composite", "reduction_tag": "mean_valid",
        "parameters": {
            "component_tags": ["continuation"],
            "component_weights": {"continuation": 0.0},
        },
    }
    with pytest.raises(NativeContractError, match="component weights must be positive"):
        validate_route_template(template)


def test_template_status_controls_pins_and_blockers_without_authorizing():
    blocked = _template()
    blocked["template_status"] = "blocked"
    blocked["blocker_tags"] = ["route_evidence_missing"]
    for field in (
        "inference_evidence_id", "model_revision", "source_revision",
        "tokenizer_revision", "processor_revision",
    ):
        blocked["base_binding"][field] = None
    assert validate_route_template(blocked) == blocked
    verified = _template()
    verified["blocker_tags"] = ["route_evidence_missing"]
    with pytest.raises(NativeContractError, match="cannot retain blockers"):
        validate_route_template(verified)
    with pytest.raises(NativeContractError, match="resolved evidence mapping"):
        _validate_route_template(_template())
    with pytest.raises(NativeContractError, match="not present"):
        _validate_route_template(
            _template(), family_contracts_by_sha256={},
            governance_evidence_by_sha256=GOVERNANCE_MAP,
        )


def test_task_loss_target_track_and_temporal_intervals_are_tagged_unions():
    template = _template()
    template["target"]["interval"] = [511, 527]
    with pytest.raises(NativeContractError, match=r"\[context, parent\]"):
        validate_route_template(template)
    template = _template()
    template["objective"] = {
        "loss_tag": "raw_mse", "reduction_tag": "mean_valid", "parameters": {},
    }
    with pytest.raises(NativeContractError, match="incompatible with loss"):
        validate_route_template(template)
    classifier = _template()
    classifier["identity"]["track"] = "C"
    classifier["identity"]["task_kind"] = "classification"
    classifier["input"]["horizon_length"] = 0
    classifier["input"]["parent_length"] = 512
    classifier["time"]["horizon_unit"] = "none"
    classifier["target"] = {
        "target_tag": "class_label", "interval": None, "axes": ["batch"],
        "normalization_tag": "train_split_only",
    }
    classifier["objective"] = {
        "loss_tag": "classification_cross_entropy", "reduction_tag": "mean_valid",
        "parameters": {"class_count": 3, "label_smoothing": 0.0},
    }
    classifier["export"]["output_tag"] = "classification_logits"
    classifier["smoke_profile_tag"] = "classification_fp32_v1"
    assert validate_route_template(classifier) == classifier
    classifier["target"]["interval"] = [0, 512]
    with pytest.raises(NativeContractError, match="interval=null"):
        validate_route_template(classifier)


def test_unsupported_template_is_a_strict_separate_union():
    base = _template()["base_binding"]
    governance = _template()["governance"]
    template = {
        "schema_version": ROUTE_TEMPLATE_SCHEMA,
        "template_status": "blocked",
        "blocker_tags": ["no_upstream_training_api"],
        "family_contract_sha256": FAMILY_SHA,
        "governance_evidence_sha256": GOVERNANCE_SHA,
        "identity": {
            "arm_key": "toto2_22m", "track": "F",
            "route_id": "no_released_finetuning", "pathway_kind": "unsupported",
            "task_kind": "unsupported", "method_provenance": "unsupported",
        },
        "base_binding": base,
        "governance": {
            **governance,
            "permitted_use_scopes": [],
        },
    }
    assert validate_route_template(template) == template
    template["objective"] = {}
    with pytest.raises(NativeContractError, match="unknown=.*objective"):
        validate_route_template(template)
    del template["objective"]
    template = deepcopy(template)
    template["governance"]["permitted_use_scopes"] = ["research_noncommercial"]
    with pytest.raises(NativeContractError, match="no permitted use scope"):
        validate_route_template(template)


def test_route_instance_binds_all_data_authorities_and_template():
    template = _template()
    template_sha = route_template_sha256(template)
    instance = build_route_instance(_instance_payload(template_sha))
    assert validate_route_instance(
        instance, templates_by_sha256={template_sha: template}
    ) == instance


def test_route_instance_rejects_oos_or_missing_holdout_prohibition():
    payload = _instance_payload()
    payload["data_binding"]["allowed_validation_splits"] = ["legacy_holdout"]
    with pytest.raises(NativeContractError):
        build_route_instance(payload)
    payload = _instance_payload()
    payload["data_binding"]["forbidden_splits"] = ["prospective_holdout"]
    with pytest.raises(NativeContractError, match="explicitly forbid legacy_holdout"):
        build_route_instance(payload)
    payload = _instance_payload()
    payload["data_binding"]["forbidden_splits"] = ["legacy_holdout"]
    with pytest.raises(NativeContractError, match="prospective_holdout"):
        build_route_instance(payload)
    payload = _instance_payload()
    payload["data_binding"]["forbidden_splits"].append("pretrain")
    with pytest.raises(NativeContractError, match="splits overlap"):
        build_route_instance(payload)


def test_route_instance_rejects_missing_authority_and_post_hash_mutation():
    payload = _instance_payload()
    del payload["data_binding"]["session_denominator_sha256"]
    with pytest.raises(NativeContractError, match="session_denominator_sha256"):
        build_route_instance(payload)
    instance = build_route_instance(_instance_payload())
    instance["data_binding"]["timeframes_minutes"] = [1, 5]
    with pytest.raises(NativeContractError, match="hash does not bind"):
        validate_route_instance(instance, templates_by_sha256=_template_map())


def test_route_instance_rejects_stale_or_swapped_template_mapping():
    template = _template()
    template_sha = route_template_sha256(template)
    instance = build_route_instance(_instance_payload(template_sha))
    swapped = _template()
    swapped["identity"]["route_id"] = "another_route"
    with pytest.raises(NativeContractError, match="stale or swapped"):
        validate_route_instance(instance, templates_by_sha256={template_sha: swapped})
    with pytest.raises(TypeError):
        _build_route_instance(_instance_payload(template_sha))
    blocked = _template()
    blocked["template_status"] = "blocked"
    blocked["blocker_tags"] = ["route_evidence_missing"]
    blocked_sha = route_template_sha256(blocked)
    payload = _instance_payload(blocked_sha)
    payload["instance_sha256"] = content_sha256(payload)
    with pytest.raises(NativeContractError, match="only verified"):
        _validate_route_instance(
            payload, templates_by_sha256={blocked_sha: blocked},
            family_contracts_by_sha256=FAMILY_MAP,
            governance_evidence_by_sha256=GOVERNANCE_MAP,
        )


def _template_map():
    template = _template()
    return {route_template_sha256(template): template}


def _child_template(parent=None):
    parent = parent or _template()
    child = deepcopy(_template())
    child["identity"]["route_id"] = "child_quantile_forecast"
    child["lineage"] = {
        "initialization_tag": "parent_route_artifact",
        "allowed_parent_route_tags": [parent["identity"]["route_id"]],
        "forbidden_parent_route_tags": [],
        "parent_artifact_requirements": ["forecast_bundle"],
        "parent_input_slots": ["parent_model"],
    }
    return child


def _two_instances(*, child_template=None):
    parent_template = _template()
    child_template = child_template or _child_template(parent_template)
    templates = {
        route_template_sha256(parent_template): parent_template,
        route_template_sha256(child_template): child_template,
    }
    first = build_route_instance(
        _instance_payload(route_template_sha256(parent_template), sample_hash="2" * 64),
        templates_by_sha256=templates,
    )
    second = build_route_instance(
        _instance_payload(route_template_sha256(child_template), sample_hash="f" * 64),
        templates_by_sha256=templates,
    )
    return first, second, {
        first["instance_sha256"]: first,
        second["instance_sha256"]: second,
    }, templates


def _artifact_manifest(parent, *, output_slot="trained_model", tag="forecast_bundle",
                       artifact_sha="0" * 64):
    manifest = {
        "schema_version": PIPELINE_ARTIFACT_MANIFEST_SCHEMA,
        "artifact_sha256": artifact_sha,
        "artifact_tag": tag,
        "producer_route_instance_sha256": parent["instance_sha256"],
        "output_slot": output_slot,
    }
    return manifest, {content_sha256(manifest): manifest}


def _pipeline_payload(first, second, manifest):
    return {
        "schema_version": PIPELINE_SCHEMA,
        "pipeline_id": "two_stage_native_pipeline",
        "nodes": [
            {"node_id": "stage_one", "route_instance_sha256": first["instance_sha256"]},
            {"node_id": "stage_two", "route_instance_sha256": second["instance_sha256"]},
        ],
        "edges": [{
            "parent_node_id": "stage_one", "child_node_id": "stage_two",
            "artifact_tag": manifest["artifact_tag"],
            "artifact_sha256": manifest["artifact_sha256"],
            "manifest_sha256": content_sha256(manifest),
            "parent_output_slot": manifest["output_slot"],
            "child_input_slot": "parent_model",
        }],
    }


def test_pipeline_requires_exact_instance_closure_and_hash():
    first, second, instances, templates = _two_instances()
    manifest, manifests = _artifact_manifest(first)
    pipeline = _build_pipeline_dag(
        _pipeline_payload(first, second, manifest), instances_by_sha256=instances,
        templates_by_sha256=templates, family_contracts_by_sha256=FAMILY_MAP,
        governance_evidence_by_sha256=GOVERNANCE_MAP,
        artifact_manifests_by_sha256=manifests,
    )
    assert _validate_pipeline_dag(
        pipeline, instances_by_sha256=instances, templates_by_sha256=templates,
        family_contracts_by_sha256=FAMILY_MAP,
        governance_evidence_by_sha256=GOVERNANCE_MAP,
        artifact_manifests_by_sha256=manifests,
    ) == pipeline
    missing = dict(instances)
    missing.pop(second["instance_sha256"])
    with pytest.raises(NativeContractError, match="unknown route instance"):
        _validate_pipeline_dag(
            pipeline, instances_by_sha256=missing, templates_by_sha256=templates,
            family_contracts_by_sha256=FAMILY_MAP,
            governance_evidence_by_sha256=GOVERNANCE_MAP,
            artifact_manifests_by_sha256=manifests,
        )
    with pytest.raises(NativeContractError, match="unknown template"):
        _validate_pipeline_dag(
            pipeline, instances_by_sha256=instances, templates_by_sha256={},
            family_contracts_by_sha256=FAMILY_MAP,
            governance_evidence_by_sha256=GOVERNANCE_MAP,
            artifact_manifests_by_sha256=manifests,
        )
    extra_template = _template()
    extra_template["identity"]["route_id"] = "unused_route"
    extra_templates = dict(templates)
    extra_templates[route_template_sha256(extra_template)] = extra_template
    with pytest.raises(NativeContractError, match="template mapping must have exact"):
        _validate_pipeline_dag(
            pipeline, instances_by_sha256=instances, templates_by_sha256=extra_templates,
            family_contracts_by_sha256=FAMILY_MAP,
            governance_evidence_by_sha256=GOVERNANCE_MAP,
            artifact_manifests_by_sha256=manifests,
        )
    pipeline["pipeline_id"] = "mutated_pipeline"
    with pytest.raises(NativeContractError, match="hash does not bind"):
        _validate_pipeline_dag(
            pipeline, instances_by_sha256=instances, templates_by_sha256=templates,
            family_contracts_by_sha256=FAMILY_MAP,
            governance_evidence_by_sha256=GOVERNANCE_MAP,
            artifact_manifests_by_sha256=manifests,
        )


def test_pipeline_rejects_dangling_edges_and_cycles():
    first, second, instances, templates = _two_instances()
    manifest, manifests = _artifact_manifest(first)
    payload = _pipeline_payload(first, second, manifest)
    payload["edges"][0]["child_node_id"] = "missing"
    with pytest.raises(NativeContractError, match="node closure"):
        _build_pipeline_dag(
            payload, instances_by_sha256=instances, templates_by_sha256=templates,
            family_contracts_by_sha256=FAMILY_MAP,
            governance_evidence_by_sha256=GOVERNANCE_MAP,
            artifact_manifests_by_sha256=manifests,
        )


def test_pipeline_rejects_incomplete_child_lineage_and_actual_cycles():
    parent = _template()
    child = _child_template(parent)
    child["lineage"]["allowed_parent_route_tags"].append("missing_parent_route")
    first, second, instances, templates = _two_instances(child_template=child)
    manifest, manifests = _artifact_manifest(first)
    with pytest.raises(NativeContractError, match="closure is not exact"):
        _build_pipeline_dag(
            _pipeline_payload(first, second, manifest),
            instances_by_sha256=instances, templates_by_sha256=templates,
            family_contracts_by_sha256=FAMILY_MAP,
            governance_evidence_by_sha256=GOVERNANCE_MAP,
            artifact_manifests_by_sha256=manifests,
        )

    route_a = _template()
    route_a["identity"]["route_id"] = "cycle_a"
    route_b = _template()
    route_b["identity"]["route_id"] = "cycle_b"
    for template, parent_id in ((route_a, "cycle_b"), (route_b, "cycle_a")):
        template["lineage"] = {
            "initialization_tag": "parent_route_artifact",
            "allowed_parent_route_tags": [parent_id],
            "forbidden_parent_route_tags": [],
            "parent_artifact_requirements": ["forecast_bundle"],
            "parent_input_slots": ["parent_model"],
        }
    cycle_templates = {
        route_template_sha256(route_a): route_a,
        route_template_sha256(route_b): route_b,
    }
    instance_a = build_route_instance(
        _instance_payload(route_template_sha256(route_a), sample_hash="a" * 64),
        templates_by_sha256=cycle_templates,
    )
    instance_b = build_route_instance(
        _instance_payload(route_template_sha256(route_b), sample_hash="b" * 64),
        templates_by_sha256=cycle_templates,
    )
    cycle_instances = {
        instance_a["instance_sha256"]: instance_a,
        instance_b["instance_sha256"]: instance_b,
    }
    manifest_a, map_a = _artifact_manifest(instance_a, artifact_sha="c" * 64)
    manifest_b, map_b = _artifact_manifest(instance_b, artifact_sha="d" * 64)
    cycle_payload = {
        "schema_version": PIPELINE_SCHEMA,
        "pipeline_id": "actual_cycle",
        "nodes": [
            {"node_id": "node_a", "route_instance_sha256": instance_a["instance_sha256"]},
            {"node_id": "node_b", "route_instance_sha256": instance_b["instance_sha256"]},
        ],
        "edges": [
            {
                "parent_node_id": "node_a", "child_node_id": "node_b",
                "artifact_tag": "forecast_bundle",
                "artifact_sha256": manifest_a["artifact_sha256"],
                "manifest_sha256": content_sha256(manifest_a),
                "parent_output_slot": "trained_model", "child_input_slot": "parent_model",
            },
            {
                "parent_node_id": "node_b", "child_node_id": "node_a",
                "artifact_tag": "forecast_bundle",
                "artifact_sha256": manifest_b["artifact_sha256"],
                "manifest_sha256": content_sha256(manifest_b),
                "parent_output_slot": "trained_model", "child_input_slot": "parent_model",
            },
        ],
    }
    with pytest.raises(NativeContractError, match="acyclic"):
        _build_pipeline_dag(
            cycle_payload, instances_by_sha256=cycle_instances,
            templates_by_sha256=cycle_templates,
            family_contracts_by_sha256=FAMILY_MAP,
            governance_evidence_by_sha256=GOVERNANCE_MAP,
            artifact_manifests_by_sha256={**map_a, **map_b},
        )
    payload = _pipeline_payload(first, second, manifest)
    payload["edges"][0]["child_input_slot"] = "undeclared_slot"
    with pytest.raises(NativeContractError, match="child lineage"):
        _build_pipeline_dag(
            payload, instances_by_sha256=instances, templates_by_sha256=templates,
            family_contracts_by_sha256=FAMILY_MAP,
            governance_evidence_by_sha256=GOVERNANCE_MAP,
            artifact_manifests_by_sha256=manifests,
        )


def test_pipeline_rejects_manifest_swap_and_vanilla_inbound_edge():
    first, second, instances, templates = _two_instances()
    manifest, manifests = _artifact_manifest(first)
    payload = _pipeline_payload(first, second, manifest)
    payload["edges"][0]["artifact_sha256"] = "1" * 64
    with pytest.raises(NativeContractError, match="does not match"):
        _build_pipeline_dag(
            payload, instances_by_sha256=instances, templates_by_sha256=templates,
            family_contracts_by_sha256=FAMILY_MAP,
            governance_evidence_by_sha256=GOVERNANCE_MAP,
            artifact_manifests_by_sha256=manifests,
        )
    reverse_manifest, reverse_manifests = _artifact_manifest(second)
    reverse_payload = _pipeline_payload(first, second, reverse_manifest)
    reverse_payload["edges"][0]["parent_node_id"] = "stage_two"
    reverse_payload["edges"][0]["child_node_id"] = "stage_one"
    with pytest.raises(NativeContractError, match="parent-route initialization|indegree zero"):
        _build_pipeline_dag(
            reverse_payload, instances_by_sha256=instances, templates_by_sha256=templates,
            family_contracts_by_sha256=FAMILY_MAP,
            governance_evidence_by_sha256=GOVERNANCE_MAP,
            artifact_manifests_by_sha256=reverse_manifests,
        )


def test_evidence_key_binds_arm_track_route_template_and_instance():
    template = _template()
    instance = build_route_instance(_instance_payload(route_template_sha256(template)))
    key = _evidence_key(
        template, instance, family_contracts_by_sha256=FAMILY_MAP,
        governance_evidence_by_sha256=GOVERNANCE_MAP,
    )
    assert key == ":".join((
        "chronos_bolt", "F", "native_quantile_forecast",
        route_template_sha256(template), instance["instance_sha256"],
    ))
