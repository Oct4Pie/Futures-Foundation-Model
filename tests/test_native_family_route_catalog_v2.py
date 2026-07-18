from copy import deepcopy
from collections import Counter

import pytest

from futures_foundation.finetune.native_contracts import (
    NativeContractError,
    content_sha256,
    load_registry,
)
from futures_foundation.finetune.native_family_route_catalog_v2 import (
    CATALOG,
    CATALOG_PATH,
    EXPECTED_ARMS,
    TASK_KINDS,
    catalog_sha256,
    load_family_route_catalog,
    validate_family_route_catalog,
)


EXPECTED_ROUTES = {
    "mantis_v1:R:official_crop_resize_contrastive",
    "mantis_v1:C:supervised_classification_head",
    "mantis_v1:C:supervised_classification_full",
    "mantis_v1:B:supervised_barrier_experimental_task",
    "mantis_v2:R:official_crop_resize_contrastive",
    "mantis_v2:C:supervised_classification_head",
    "mantis_v2:C:supervised_classification_full",
    "mantis_v2:B:supervised_barrier_experimental_task",
    "moment_small:R:masked_patch_reconstruction",
    "moment_small:C:classification_head_only",
    "moment_small:C:classification_full",
    "moment_small:F:forecast_head_only_raw_mse",
    "moment_small:F:forecast_full_raw_mse",
    "moment_small:R:adjacent_half_contrastive",
    "moment_small:B:supervised_barrier_experimental_task",
    "kronos_mini:F:tokenizer_reconstruction_bsq",
    "kronos_mini:F:hierarchical_autoregressive_tokens",
    "kronos_mini:R:adjacent_half_contrastive",
    "kronos_mini:B:supervised_barrier_experimental_task",
    "kronos_small:F:tokenizer_reconstruction_bsq",
    "kronos_small:F:hierarchical_autoregressive_tokens",
    "kronos_small:R:adjacent_half_contrastive",
    "kronos_small:B:supervised_barrier_experimental_task",
    "chronos_v1:F:native_64_t5_token_forecast_cross_entropy",
    "chronos_v1:B:supervised_barrier_experimental_task",
    "chronos_bolt:F:direct_native_quantile_pinball",
    "chronos_bolt:B:supervised_barrier_experimental_task",
    "chronos_v2:F:official_fit_full",
    "chronos_v2:F:official_fit_lora",
    "chronos_v2:B:supervised_barrier_experimental_task",
    "timesfm25:F:official_lora_forecast",
    "timesfm25:B:supervised_barrier_experimental_task",
    "ttm_r2:F:head_prefix_raw_hf_trainer_forecast",
    "ttm_r2:F:full_model_raw_hf_trainer_forecast",
    "moirai2_small:F:custom_scaled_pinball_research",
    "moirai2_small:B:custom_supervised_barrier_research",
    "toto2_22m:F:no_released_toto2_finetuning",
    "sundial_base:F:no_released_sundial_finetuning",
    "sundial_base:F:custom_timeflow_research",
    "sundial_base:R:nonfinite_hidden_state_representation",
    "tabpfn_ts3_forecast:F:official_ts3_support_query_forecast",
    "tabpfn_v3_downstream:D:nested_downstream_support_query",
}


def _value(profile, field):
    item = CATALOG["constraint_profiles"][profile][field]
    assert item["state"] in {"resolved", "partial"}
    return item["value"] if item["state"] == "resolved" else item["value"]["known"]


def test_catalog_has_exact_corrected_inventory_and_is_non_authorizing():
    catalog = validate_family_route_catalog()
    assert set(catalog["arms"]) == set(EXPECTED_ARMS)
    assert set(catalog["routes"]) == EXPECTED_ROUTES
    assert len(catalog["arms"]) == 15
    assert len(catalog["routes"]) == 42
    assert catalog["non_authorizing"] is True
    assert {arm["status"] for arm in catalog["arms"].values()} == {"blocked"}
    assert {route["status"] for route in catalog["routes"].values()} == {"blocked"}
    assert all(route["evidence_id"] is None for route in catalog["routes"].values())
    assert all(route["task_kind"] in TASK_KINDS for route in catalog["routes"].values())
    assert catalog_sha256() == catalog_sha256(deepcopy(CATALOG))


def test_every_route_has_the_exact_audited_task_kind():
    assert Counter(route["task_kind"] for route in CATALOG["routes"].values()) == {
        "classification": 6,
        "continuous_forecast": 4,
        "contrastive_representation": 5,
        "generative_forecast": 1,
        "masked_reconstruction": 1,
        "path_supervision": 10,
        "quantile_forecast": 5,
        "support_query_downstream": 1,
        "support_query_forecast": 1,
        "token_forecast": 3,
        "tokenizer_reconstruction": 2,
        "unsupported": 3,
    }
    expected_examples = {
        "mantis_v1:R:official_crop_resize_contrastive": "contrastive_representation",
        "moment_small:R:masked_patch_reconstruction": "masked_reconstruction",
        "mantis_v2:C:supervised_classification_full": "classification",
        "moment_small:F:forecast_full_raw_mse": "continuous_forecast",
        "chronos_bolt:F:direct_native_quantile_pinball": "quantile_forecast",
        "chronos_v1:F:native_64_t5_token_forecast_cross_entropy": "token_forecast",
        "kronos_small:F:tokenizer_reconstruction_bsq": "tokenizer_reconstruction",
        "timesfm25:B:supervised_barrier_experimental_task": "path_supervision",
        "sundial_base:F:custom_timeflow_research": "generative_forecast",
        "tabpfn_ts3_forecast:F:official_ts3_support_query_forecast": (
            "support_query_forecast"
        ),
        "tabpfn_v3_downstream:D:nested_downstream_support_query": (
            "support_query_downstream"
        ),
        "toto2_22m:F:no_released_toto2_finetuning": "unsupported",
    }
    for route_key, task_kind in expected_examples.items():
        assert CATALOG["routes"][route_key]["task_kind"] == task_kind


def test_every_arm_binds_the_authoritative_inference_dossier_by_content_hash():
    inference = load_registry()["models"]
    for arm_key in EXPECTED_ARMS:
        catalog_arm = CATALOG["arms"][arm_key]
        dossier = inference[arm_key]
        assert set(catalog_arm) == {"status", "dossier_ref"}
        assert catalog_arm["dossier_ref"] == {
            "registry_schema": "ffm_native_contract_registry_v1",
            "arm_key": arm_key,
            "content_sha256": content_sha256(dossier),
        }


def test_packaged_json_is_the_loaded_training_semantics_ssot():
    assert CATALOG_PATH.name == "native_family_route_catalog_v2.json"
    assert load_family_route_catalog() == CATALOG


def test_tabpfn_is_split_without_fake_identity_or_checkpoint_pins():
    inference = load_registry()["models"]
    forecast = inference["tabpfn_ts3_forecast"]
    downstream = inference["tabpfn_v3_downstream"]
    assert forecast["model_id"] == (
        "tabpfn-v3-regressor-v3_20260506_timeseries.ckpt"
    )
    assert forecast["model_revision"] == "license_gated_unresolved_sha256"
    assert downstream["model_id"] == "unresolved_tabpfn_v3_downstream_checkpoint"
    assert downstream["model_revision"] == "unresolved_checkpoint_revision"
    assert forecast["license"]["deployment"] == "blocked_until_terms_and_version_verified"
    assert downstream["license"]["deployment"] == "blocked_until_terms_and_version_verified"


def test_mantis_profiles_preserve_version_specific_training_and_deployment_contracts():
    v1 = CATALOG["constraint_profiles"]["mantis_v1_contrastive"]
    v2 = CATALOG["constraint_profiles"]["mantis_v2_contrastive"]
    assert v1["objective"]["value"] == {
        "loss_tag": "mantis_v1_one_way_info_nce", "temperature": 0.1,
        "direction_tag": "query_to_key", "resize_length": 512,
        "negative_eligibility_tag": "other_parent_windows_only",
        "sibling_parent_policy_tag": "exclude_ohlcv_siblings",
    }
    assert v1["export"]["value"]["output_tag"] == "final_cls_per_channel"
    assert _value("mantis_v2_contrastive", "objective")["training_projector_dim"] == 256
    assert _value("mantis_v2_contrastive", "objective")["deployment_embedding_dim"] == 512
    assert set(v2["objective"]["value"]["unresolved_fields"]) == {
        "temperature", "direction_tag", "crop_sampling_scope_tag",
    }
    assert v2["export"]["value"]["output_tag"] == "layer2_cls_mean_per_channel"
    assert CATALOG["routes"]["mantis_v1:C:supervised_classification_head"][
        "constraint_profile"
    ] == "mantis_v1_classifier_head"
    assert CATALOG["routes"]["mantis_v2:C:supervised_classification_head"][
        "constraint_profile"
    ] == "mantis_v2_classifier_head"
    assert _value("mantis_v1_classifier_head", "optimization_surface")["trainable_tag"] == "head_only"
    assert _value("mantis_v1_classifier_full", "optimization_surface")["trainable_tag"] == "encoder_and_head"
    assert _value("mantis_v2_classifier_head", "objective")["encoder_embedding_tag"] == "final_layer_cls_per_channel"
    assert _value("mantis_v2_classifier_full", "optimization_surface")["trainable_tag"] == "encoder_and_head"
    for profile in ("mantis_v1_contrastive", "mantis_v2_contrastive"):
        objective = _value(profile, "objective")
        assert objective["negative_eligibility_tag"] == "other_parent_windows_only"
        assert objective["sibling_parent_policy_tag"] == "exclude_ohlcv_siblings"


def test_moment_head_and_full_routes_are_distinct_and_native_masks_are_structured():
    head = _value("moment_classification_head", "optimization_surface")
    full = _value("moment_classification_full", "optimization_surface")
    assert head["trainable_tag"] == "head_only"
    assert head["frozen_tag"] == "encoder"
    assert full["trainable_tag"] == "encoder_and_head"
    assert _value("moment_reconstruction", "preprocessing")["mask_tag"] == (
        "valid_plus_shared_timestamp_pretrain_mask"
    )
    assert _value("moment_forecast_head", "target")["interval"] == [512, 528]


@pytest.mark.parametrize("arm", ["kronos_mini", "kronos_small"])
def test_kronos_is_two_routes_with_causal_normalization_and_exact_lineage(arm):
    tokenizer_key = f"{arm}:F:tokenizer_reconstruction_bsq"
    predictor_key = f"{arm}:F:hierarchical_autoregressive_tokens"
    assert CATALOG["routes"][tokenizer_key]["method_provenance"] == "native_derived"
    assert CATALOG["routes"][predictor_key]["method_provenance"] == "native_derived"
    tokenizer_input = _value(f"{arm}_tokenizer", "input")
    predictor_input = _value(f"{arm}_predictor", "input")
    assert tokenizer_input["parent_length"] == 528
    assert predictor_input["context_length"] == 512
    assert predictor_input["horizon_length"] == 16
    assert predictor_input["stamp_shape"] == [528, 5]
    assert _value(f"{arm}_predictor", "preprocessing")["scaler_tag"] == (
        "kronos_context_only_normalize_clip5"
    )
    expected_bundle = (
        "kronos_mini_tokenizer_2k_bundle" if arm == "kronos_mini"
        else "kronos_small_tokenizer_base_bundle"
    )
    assert _value(f"{arm}_predictor", "lineage")["parent_artifacts"] == [expected_bundle]


def test_forecast_family_constraints_retain_audited_shapes_and_objectives():
    chronos_v1 = _value("chronos_v1_forecast", "objective")
    assert chronos_v1["missing_label_id"] == -100
    assert chronos_v1["native_horizon"] == 64
    assert chronos_v1["effective_horizon"] == 16
    assert _value("chronos_v1_forecast", "input")["parent_length"] == 576
    assert _value("chronos_bolt_forecast", "objective")["effective_horizon"] == 16
    assert _value("chronos2_full", "input")["grouping_tag"] == "same_parent_group_ids"
    assert _value("chronos2_lora", "optimization_surface")["trainable_tag"] == (
        "lora_only_fail_closed_without_peft"
    )
    assert _value("timesfm_lora", "preprocessing")["mask_tag"] == "native_padding_missing_mask"
    timesfm = _value("timesfm_lora", "objective")
    assert timesfm["force_flip_invariance"] is True
    assert timesfm["truncate_negative"] is False
    assert timesfm["fix_quantile_crossing"] is False
    assert _value("timesfm_lora", "optimization_surface")["adapter_target_tag"] == "all_linear"
    ttm = _value("ttm_forecast_head_prefix", "objective")
    assert ttm["native_horizon"] == 48
    assert ttm["prediction_filter_length"] == 16


def test_sundial_and_toto_dispositions_do_not_claim_released_training():
    assert CATALOG["routes"]["toto2_22m:F:no_released_toto2_finetuning"][
        "pathway_kind"
    ] == "unsupported"
    assert CATALOG["routes"]["sundial_base:F:no_released_sundial_finetuning"][
        "pathway_kind"
    ] == "unsupported"
    timeflow = CATALOG["routes"]["sundial_base:F:custom_timeflow_research"]
    assert timeflow["method_provenance"] == "project_extension"
    assert timeflow["constraint_profile"] == "sundial_timeflow"
    assert "nonfinite_hidden_states" in CATALOG["routes"][
        "sundial_base:R:nonfinite_hidden_state_representation"
    ]["blocker_tags"]


def test_unfilled_methodology_is_explicit_and_never_hidden_by_prose():
    for profile_id in ("project_contrastive_unresolved", "project_path_unresolved"):
        profile = CATALOG["constraint_profiles"][profile_id]
        assert profile["target"] == {"state": "unresolved", "value": None}
        assert profile["objective"] == {"state": "unresolved", "value": None}
        assert profile["optimization_hyperparameters"] == {
            "state": "unresolved", "value": None,
        }
    for route in CATALOG["routes"].values():
        assert route["blocker_tags"]
        assert all(isinstance(tag, str) and " " not in tag for tag in route["blocker_tags"])
    for profile_id in (
        "kronos_mini_tokenizer", "kronos_mini_predictor",
        "kronos_small_tokenizer", "kronos_small_predictor",
    ):
        preprocessing = CATALOG["constraint_profiles"][profile_id]["preprocessing"]
        assert preprocessing["state"] == "partial"
        assert preprocessing["value"]["unresolved_fields"] == [
            "amount_source_route_instance_choice"
        ]
    ttm_pre = _value("ttm_forecast_head_prefix", "preprocessing")
    assert ttm_pre["three_minute_prefix_id"] == 0
    assert ttm_pre["selector_id"] == "512-48-ft-r2.1"
    assert ttm_pre["frequency_tokens_by_timeframe"] == {
        "1": 1, "3": 0, "5": 3, "15": 5, "30": 6, "60": 7,
    }


def test_catalog_rejects_unknown_fields_authorization_and_license_escalation():
    catalog = deepcopy(CATALOG)
    catalog["surprise"] = True
    with pytest.raises(NativeContractError, match="unknown=.*surprise"):
        validate_family_route_catalog(catalog)
    catalog = deepcopy(CATALOG)
    catalog["non_authorizing"] = False
    with pytest.raises(NativeContractError, match="non-authorizing"):
        validate_family_route_catalog(catalog)
    catalog = deepcopy(CATALOG)
    key = "moirai2_small:F:custom_scaled_pinball_research"
    catalog["routes"][key]["permitted_use_scopes"] = ["production"]
    with pytest.raises(NativeContractError, match="license scopes"):
        validate_family_route_catalog(catalog)


def test_catalog_rejects_fake_evidence_missing_blockers_and_orphans():
    catalog = deepcopy(CATALOG)
    key = "chronos_bolt:F:direct_native_quantile_pinball"
    catalog["routes"][key]["evidence_id"] = "self-authored"
    with pytest.raises(NativeContractError, match="without evidence"):
        validate_family_route_catalog(catalog)
    catalog = deepcopy(CATALOG)
    catalog["routes"][key]["blocker_tags"] = []
    with pytest.raises(NativeContractError, match="blockers are invalid"):
        validate_family_route_catalog(catalog)
    catalog = deepcopy(CATALOG)
    catalog["routes"][key]["constraint_profile"] = "missing"
    with pytest.raises(NativeContractError, match="unknown profile"):
        validate_family_route_catalog(catalog)
    catalog = deepcopy(CATALOG)
    del catalog["routes"]["ttm_r2:F:head_prefix_raw_hf_trainer_forecast"]
    with pytest.raises(NativeContractError, match="constraint profile closure|canonical route semantics"):
        validate_family_route_catalog(catalog)


def test_catalog_rejects_free_text_nested_fields_and_malformed_partial_constraints():
    catalog = deepcopy(CATALOG)
    catalog["constraint_profiles"]["chronos_bolt_forecast"]["objective"]["value"][
        "description"
    ] = "this should train a forecast"
    with pytest.raises(NativeContractError, match="nested fields mismatch"):
        validate_family_route_catalog(catalog)
    catalog = deepcopy(CATALOG)
    catalog["constraint_profiles"]["chronos_bolt_forecast"]["objective"]["value"][
        "loss_tag"
    ] = "use whatever loss works"
    with pytest.raises(NativeContractError, match="structured tag"):
        validate_family_route_catalog(catalog)
    catalog = deepcopy(CATALOG)
    catalog["constraint_profiles"]["mantis_v2_contrastive"]["objective"]["value"][
        "unresolved_fields"
    ] = []
    with pytest.raises(NativeContractError, match="unresolved_fields is malformed"):
        validate_family_route_catalog(catalog)
    catalog = deepcopy(CATALOG)
    catalog["arms"]["tabpfn_v3_downstream"]["dossier_ref"]["content_sha256"] = None
    with pytest.raises(NativeContractError, match="content hash is malformed"):
        validate_family_route_catalog(catalog)


def test_family_specific_audit_corrections_remain_explicit():
    assert CATALOG["routes"]["chronos_v2:F:official_fit_full"]["method_provenance"] == (
        "upstream_native"
    )
    assert CATALOG["routes"]["chronos_v2:F:official_fit_lora"]["method_provenance"] == (
        "upstream_native"
    )
    inference = load_registry()["models"]
    for arm in ("chronos_v1", "chronos_bolt", "chronos_v2"):
        assert inference[arm]["tokenizer"]["revision"] == "model_revision"
    moirai = _value("moirai_custom", "preprocessing")
    assert moirai["mask_tag"] == "observed_mask_zero_fill_before_pack"
    moirai_objective = _value("moirai_custom", "objective")
    assert moirai_objective["quantile_crossing_tag"] == "native_crossing_quantiles"
    assert moirai_objective["output_patch_audit_tag"] == "every_output_patch"
    sundial = CATALOG["constraint_profiles"]["sundial_timeflow"]
    assert sundial["input"]["state"] == "resolved"
    assert sundial["objective"]["value"]["loss_tag"] == "differentiable_timeflow"
    for field in (
        "time", "preprocessing", "target", "optimization_surface",
        "optimization_hyperparameters", "lineage", "export",
    ):
        assert sundial[field] == {"state": "unresolved", "value": None}
    assert CATALOG["routes"]["sundial_base:F:custom_timeflow_research"][
        "permitted_use_scopes"
    ] == ["research_noncommercial"]
    assert "checkpoint_unavailable" in CATALOG["routes"][
        "tabpfn_ts3_forecast:F:official_ts3_support_query_forecast"
    ]["blocker_tags"]


@pytest.mark.parametrize(
    ("mutation",),
    [
        ("swap_mantis_profiles",),
        ("swap_model_identities",),
        ("change_revision",),
        ("change_provenance",),
        ("invent_route",),
        ("promote_toto",),
    ],
)
def test_catalog_rejects_coherent_but_false_semantic_mutations(mutation):
    catalog = deepcopy(CATALOG)
    if mutation == "swap_mantis_profiles":
        one = "mantis_v1:C:supervised_classification_head"
        two = "mantis_v2:C:supervised_classification_head"
        catalog["routes"][one]["constraint_profile"], catalog["routes"][two][
            "constraint_profile"
        ] = catalog["routes"][two]["constraint_profile"], catalog["routes"][one][
            "constraint_profile"
        ]
    elif mutation == "swap_model_identities":
        catalog["arms"]["chronos_v1"]["dossier_ref"], catalog["arms"]["chronos_bolt"][
            "dossier_ref"
        ] = catalog["arms"]["chronos_bolt"]["dossier_ref"], catalog["arms"]["chronos_v1"][
            "dossier_ref"
        ]
    elif mutation == "change_revision":
        catalog["arms"]["timesfm25"]["dossier_ref"]["content_sha256"] = "0" * 64
    elif mutation == "change_provenance":
        catalog["routes"]["chronos_v2:F:official_fit_full"][
            "method_provenance"
        ] = "native_derived"
    elif mutation == "invent_route":
        source = deepcopy(catalog["routes"]["chronos_bolt:F:direct_native_quantile_pinball"])
        source["route_id"] = "invented_optimizer"
        catalog["routes"]["chronos_bolt:F:invented_optimizer"] = source
    elif mutation == "promote_toto":
        route = catalog["routes"]["toto2_22m:F:no_released_toto2_finetuning"]
        route["method_provenance"] = "upstream_native"
        route["pathway_kind"] = "optimizer_training"
        route["permitted_use_scopes"] = ["production"]
    with pytest.raises(
        NativeContractError,
        match=(
            "canonical route semantics|cannot claim use|dossier reference identity|"
            "dossier hash mismatch|unsupported task/pathway mismatch"
        ),
    ):
        validate_family_route_catalog(catalog)
