"""Strict, non-authorizing schemas for native training methodology version 2.

This module deliberately has no connection to the current admission gate.  It validates
candidate route templates, concrete data-bound route instances, and pipeline DAGs while
the v1 registry remains fail-closed.  Human prose is not part of any operational contract.
"""
from __future__ import annotations

from copy import deepcopy
import math
import re
from typing import Any, Mapping

from .native_contracts import NativeContractError, content_sha256


ROUTE_TEMPLATE_SCHEMA = "ffm_native_route_template_v2"
ROUTE_INSTANCE_SCHEMA = "ffm_native_route_instance_v1"
PIPELINE_SCHEMA = "ffm_native_pipeline_dag_v1"
PIPELINE_ARTIFACT_MANIFEST_SCHEMA = "ffm_native_pipeline_artifact_manifest_v1"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]*$")

TRACKS = frozenset({"F", "R", "C", "B", "D"})
PATHWAY_KINDS = frozenset({"optimizer_training", "in_context_fit", "unsupported"})
TASK_KINDS = frozenset({
    "contrastive_representation", "masked_reconstruction", "classification",
    "continuous_forecast", "quantile_forecast", "token_forecast",
    "tokenizer_reconstruction", "path_supervision", "generative_forecast",
    "support_query_forecast", "support_query_downstream", "unsupported",
})
TEMPLATE_STATUSES = frozenset({"blocked", "verified"})
METHOD_PROVENANCE = frozenset({
    "upstream_native", "native_derived", "project_extension", "unsupported",
})
LAYOUT_TAGS = frozenset({
    "independent_univariate_passes", "joint_multivariate", "packed_multivariate",
    "grouped_multivariate", "support_query_rows", "not_applicable",
})
AXIS_TAGS = frozenset({
    "batch", "batch_times_channel", "channel", "time", "horizon", "feature",
    "sample", "variate", "candidate",
})
GROUPING_TAGS = frozenset({
    "no_cross_channel_interaction", "same_parent_joint_channels",
    "same_parent_group_ids", "packed_sample_time_variate_ids", "support_query_fold",
    "not_applicable",
})
MISSING_POLICY_TAGS = frozenset({
    "reject_parent", "native_missing_mask", "masked_zero_fill", "support_only_imputation",
    "not_applicable",
})
GUARD_POLICY_TAGS = frozenset({
    "parent_interval_contained", "support_query_purged", "not_applicable",
})
HORIZON_UNITS = frozenset({"bars", "wall_clock", "none"})
TIMESTAMP_TAGS = frozenset({
    "utc_bar_close", "venue_local_calendar_stamps", "support_query_timestamp",
    "not_used",
})
TIMEZONE_TAGS = frozenset({"venue_registry", "utc", "not_used"})
SESSION_POLICY_TAGS = frozenset({"bound_at_route_instance", "not_applicable"})
ROLL_POLICY_TAGS = frozenset({"bound_at_route_instance", "not_applicable"})
PREPROCESSING_OWNERS = frozenset({
    "upstream_internal", "route_wrapper", "support_only", "not_applicable",
})
SCALER_TAGS = frozenset({
    "none", "mantis_raw", "moment_revin", "kronos_context_normalize_clip5",
    "chronos_mean_abs", "bolt_native", "chronos2_native", "timesfm_internal_revin",
    "ttm_native", "moirai_native", "sundial_native", "support_only",
    "route_defined_causal", "not_applicable",
})
STATISTICS_INTERVAL_TAGS = frozenset({"context_only", "support_only", "none"})
MASK_TAGS = frozenset({
    "none", "mantis_native_views", "moment_valid_and_pretrain",
    "chronos_eos_and_missing_labels", "native_patch_mask", "packed_native_mask",
    "support_query_purge_mask", "route_defined_causal", "not_applicable",
})
TARGET_TAGS = frozenset({
    "same_context_views", "masked_context_values", "class_label", "future_raw_channels",
    "future_tokens", "future_quantiles", "tokenizer_reconstruction",
    "forward_path_labels", "flow_noise_target", "support_query_labels", "not_applicable",
})
NORMALIZATION_TAGS = frozenset({
    "none", "native_internal", "context_only", "train_split_only", "support_only",
    "not_applicable",
})
LOSS_TAGS = frozenset({
    "mantis_v1_one_way_info_nce", "mantis_v2_info_nce", "masked_raw_mse",
    "classification_cross_entropy", "raw_mse", "quantile_pinball",
    "point_plus_quantile", "token_cross_entropy", "tokenizer_reconstruction_bsq",
    "hierarchical_token_cross_entropy", "path_composite", "timeflow",
    "support_query_native",
})
REDUCTION_TAGS = frozenset({"upstream_exact", "mean_valid", "sum_then_mean_batch"})
TRAINABLE_SURFACE_TAGS = frozenset({
    "full_model", "encoder_and_projector", "head_only", "encoder_and_head",
    "tokenizer_only", "predictor_only", "lora_only", "route_defined_adapter",
    "no_persistent_weights",
})
FROZEN_SURFACE_TAGS = frozenset({
    "none", "full_base_model", "encoder", "tokenizer", "predictor", "backbone",
    "not_applicable",
})
PRECISION_TAGS = frozenset({"fp32"})
OPTIMIZER_TAGS = frozenset({"adam", "adamw", "adafactor", "native_declared", "none"})
SCHEDULER_TAGS = frozenset({
    "constant", "linear_warmup", "cosine_warmup", "native_declared", "none",
})
INITIALIZATION_TAGS = frozenset({
    "vanilla_pinned_checkpoint", "parent_route_artifact", "fresh_head_on_pinned_base",
    "support_query_only",
})
BUNDLE_TAGS = frozenset({
    "full_model_and_preprocessor", "base_plus_adapter_and_preprocessor",
    "tokenizer_bundle", "predictor_plus_tokenizer_bundle", "classifier_bundle",
    "representation_bundle", "support_query_contract", "path_model_bundle",
})
OUTPUT_TAGS = frozenset({
    "mantis_v1_final_cls_per_channel", "mantis_v2_layer2_cls_mean_per_channel",
    "moment_masked_embedding_mean",
    "classification_logits", "continuous_forecast", "quantile_forecast",
    "token_forecast_distribution", "joint_ohlcva_forecast", "path_distribution",
    "support_query_predictions", "tokenizer_codes",
})
USE_SCOPES = frozenset({"production", "research_noncommercial"})
SMOKE_PROFILE_TAGS = frozenset({
    "contrastive_fp32_v1", "reconstruction_fp32_v1", "classification_fp32_v1",
    "continuous_forecast_fp32_v1", "quantile_forecast_fp32_v1",
    "token_forecast_fp32_v1", "tokenizer_fp32_v1", "path_fp32_v1",
    "generative_forecast_fp32_v1", "support_query_v1",
})
AMOUNT_SOURCE_TAGS = frozenset({
    "provider_turnover", "volume_times_mean_ohlc", "not_applicable",
})
PIPELINE_ARTIFACT_TAGS = frozenset({
    "vanilla_checkpoint_bundle", "frozen_encoder_bundle", "frozen_tokenizer_bundle",
    "full_training_state_bundle", "representation_bundle", "forecast_bundle",
    "classifier_bundle", "path_bundle",
})
UNSUPPORTED_BLOCKER_TAGS = frozenset({
    "no_upstream_training_api", "nonfinite_hidden_states", "terms_unaccepted",
    "checkpoint_unavailable", "checkpoint_hash_unavailable", "unsupported_task",
    "route_evidence_missing", "exact_resume_missing", "base_inference_evidence_missing",
    "license_evidence_missing", "native_output_parity_missing", "data_authority_missing",
    "implementation_missing", "methodology_review_missing",
})
PATH_COMPONENT_TAGS = frozenset({
    "forward_realized_volatility", "favorable_mfe_quantiles", "adverse_mae_quantiles",
    "continuation", "termination", "reversal", "vanilla_anchor",
})

_FULL_RESUME_STATE = frozenset({
    "model", "optimizer", "scheduler", "scaler", "epoch", "global_step", "sampler",
    "python_rng", "numpy_rng", "torch_cpu_rng", "torch_cuda_rng",
})
_SURFACE_PAIRS = {
    "full_model": {"none"},
    "encoder_and_projector": {"none", "not_applicable"},
    "head_only": {"encoder", "backbone", "full_base_model"},
    "encoder_and_head": {"none"},
    "tokenizer_only": {"predictor", "not_applicable"},
    "predictor_only": {"tokenizer"},
    "lora_only": {"full_base_model", "backbone"},
    "route_defined_adapter": {"full_base_model", "backbone", "encoder"},
    "no_persistent_weights": {"full_base_model", "not_applicable"},
}

_TASK_LOSSES = {
    "contrastive_representation": {"mantis_v1_one_way_info_nce", "mantis_v2_info_nce"},
    "masked_reconstruction": {"masked_raw_mse"},
    "classification": {"classification_cross_entropy"},
    "continuous_forecast": {"raw_mse"},
    "quantile_forecast": {"quantile_pinball", "point_plus_quantile"},
    "token_forecast": {"token_cross_entropy", "hierarchical_token_cross_entropy"},
    "tokenizer_reconstruction": {"tokenizer_reconstruction_bsq"},
    "path_supervision": {"path_composite"},
    "generative_forecast": {"timeflow"},
    "support_query_forecast": {"support_query_native"},
    "support_query_downstream": {"support_query_native"},
}
_TASK_TARGETS = {
    "contrastive_representation": {"same_context_views"},
    "masked_reconstruction": {"masked_context_values"},
    "classification": {"class_label"},
    "continuous_forecast": {"future_raw_channels"},
    "quantile_forecast": {"future_quantiles"},
    "token_forecast": {"future_tokens"},
    "tokenizer_reconstruction": {"tokenizer_reconstruction"},
    "path_supervision": {"forward_path_labels"},
    "generative_forecast": {"flow_noise_target"},
    "support_query_forecast": {"support_query_labels"},
    "support_query_downstream": {"support_query_labels"},
}
_TASK_TRACKS = {
    "contrastive_representation": {"R"}, "masked_reconstruction": {"R"},
    "classification": {"C"}, "continuous_forecast": {"F"},
    "quantile_forecast": {"F"}, "token_forecast": {"F"},
    "tokenizer_reconstruction": {"F"}, "path_supervision": {"B"},
    "generative_forecast": {"F"}, "support_query_forecast": {"F"},
    "support_query_downstream": {"D"},
}
_TASK_OUTPUTS = {
    "contrastive_representation": {
        "mantis_v1_final_cls_per_channel", "mantis_v2_layer2_cls_mean_per_channel",
    },
    "masked_reconstruction": {
        "mantis_v1_final_cls_per_channel", "mantis_v2_layer2_cls_mean_per_channel",
        "moment_masked_embedding_mean",
    },
    "classification": {"classification_logits"},
    "continuous_forecast": {"continuous_forecast", "joint_ohlcva_forecast"},
    "quantile_forecast": {"quantile_forecast"},
    "token_forecast": {"token_forecast_distribution", "joint_ohlcva_forecast"},
    "tokenizer_reconstruction": {"tokenizer_codes"},
    "path_supervision": {"path_distribution"},
    "generative_forecast": {"quantile_forecast", "continuous_forecast"},
    "support_query_forecast": {"support_query_predictions"},
    "support_query_downstream": {"support_query_predictions"},
}
_TASK_SMOKE_PROFILES = {
    "contrastive_representation": {"contrastive_fp32_v1"},
    "masked_reconstruction": {"reconstruction_fp32_v1"},
    "classification": {"classification_fp32_v1"},
    "continuous_forecast": {"continuous_forecast_fp32_v1"},
    "quantile_forecast": {"quantile_forecast_fp32_v1"},
    "token_forecast": {"token_forecast_fp32_v1"},
    "tokenizer_reconstruction": {"tokenizer_fp32_v1"},
    "path_supervision": {"path_fp32_v1"},
    "generative_forecast": {"generative_forecast_fp32_v1"},
    "support_query_forecast": {"support_query_v1"},
    "support_query_downstream": {"support_query_v1"},
}


def _object(value: Any, fields: set[str], field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise NativeContractError(f"{field} must be an object")
    unknown = set(value) - fields
    missing = fields - set(value)
    if unknown or missing:
        raise NativeContractError(
            f"{field} fields mismatch: missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    return value


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise NativeContractError(f"{field} must be a nonempty trimmed string")
    return value


def _identifier_value(value: Any, field: str) -> str:
    text = _string(value, field)
    if not _IDENTIFIER.fullmatch(text):
        raise NativeContractError(f"{field} must be lowercase snake-case")
    return text


def _enum(value: Any, allowed: frozenset[str], field: str) -> str:
    if value not in allowed:
        raise NativeContractError(f"{field} has invalid tag {value!r}")
    return str(value)


def _sha(value: Any, field: str, *, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise NativeContractError(f"{field} must be a lowercase SHA-256")
    return value


def _integer(value: Any, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise NativeContractError(f"{field} must be an integer >= {minimum}")
    return value


def _number(value: Any, field: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NativeContractError(f"{field} must be numeric")
    output = float(value)
    if not math.isfinite(output) or (minimum is not None and output < minimum):
        raise NativeContractError(f"{field} is outside its finite range")
    return output


def _bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise NativeContractError(f"{field} must be boolean")
    return value


def _string_list(
    value: Any, field: str, *, allowed: frozenset[str] | None = None,
    nonempty: bool = False, identifiers: bool = False,
) -> list[str]:
    if not isinstance(value, list) or (nonempty and not value):
        raise NativeContractError(f"{field} must be a{' nonempty' if nonempty else ''} list")
    output: list[str] = []
    for index, item in enumerate(value):
        text = (
            _identifier_value(item, f"{field}[{index}]")
            if identifiers else _string(item, f"{field}[{index}]")
        )
        if allowed is not None and text not in allowed:
            raise NativeContractError(f"{field}[{index}] has invalid tag {text!r}")
        output.append(text)
    if len(output) != len(set(output)):
        raise NativeContractError(f"{field} must not contain duplicates")
    return output


def _validate_identity(value: Any) -> Mapping[str, Any]:
    item = _object(value, {
        "arm_key", "track", "route_id", "pathway_kind", "task_kind",
        "method_provenance",
    }, "identity")
    _identifier_value(item["arm_key"], "identity.arm_key")
    _enum(item["track"], TRACKS, "identity.track")
    _identifier_value(item["route_id"], "identity.route_id")
    pathway = _enum(item["pathway_kind"], PATHWAY_KINDS, "identity.pathway_kind")
    task = _enum(item["task_kind"], TASK_KINDS, "identity.task_kind")
    provenance = _enum(
        item["method_provenance"], METHOD_PROVENANCE, "identity.method_provenance"
    )
    if (pathway == "unsupported") != (task == "unsupported") or (
        pathway == "unsupported"
    ) != (provenance == "unsupported"):
        raise NativeContractError("unsupported pathway, task, and provenance must agree")
    if pathway == "in_context_fit" and task not in {
        "support_query_forecast", "support_query_downstream",
    }:
        raise NativeContractError("in-context fit requires a support-query task")
    return item


def _resolve_hash_mapping(
    sha256: Any, mapping: Mapping[str, Any] | None, field: str,
) -> None:
    digest = _sha(sha256, field)
    if mapping is None:
        raise NativeContractError(f"{field} requires an explicit resolved evidence mapping")
    record = mapping.get(str(digest))
    if record is None:
        raise NativeContractError(f"{field} is not present in its resolved evidence mapping")
    if content_sha256(record) != digest:
        raise NativeContractError(f"{field} mapping is stale or content-swapped")


def _validate_base_binding(value: Any, *, status: str) -> None:
    item = _object(value, {
        "inference_dossier_sha256", "inference_evidence_id", "model_revision",
        "source_revision", "tokenizer_revision", "processor_revision",
    }, "base_binding")
    _sha(item["inference_dossier_sha256"], "base_binding.inference_dossier_sha256")
    for name in (
        "inference_evidence_id", "model_revision", "source_revision",
        "tokenizer_revision", "processor_revision",
    ):
        if item[name] is not None:
            _string(item[name], f"base_binding.{name}")
    if status == "verified" and any(item[name] is None for name in (
        "inference_evidence_id", "model_revision", "source_revision",
        "tokenizer_revision", "processor_revision",
    )):
        raise NativeContractError(
            "verified route templates require exact base evidence and revision pins; "
            "use 'not_applicable' rather than null for an absent tokenizer or processor"
        )


def _validate_input(value: Any) -> tuple[int, int, int, str]:
    item = _object(value, {
        "layout_tag", "axes", "context_length", "horizon_length", "parent_length",
        "channel_order", "grouping_tag", "dtype", "missing_policy_tag",
        "guard_policy_tag",
    }, "input")
    layout = _enum(item["layout_tag"], LAYOUT_TAGS, "input.layout_tag")
    _string_list(item["axes"], "input.axes", allowed=AXIS_TAGS, nonempty=True)
    context = _integer(item["context_length"], "input.context_length", minimum=1)
    horizon = _integer(item["horizon_length"], "input.horizon_length")
    parent = _integer(item["parent_length"], "input.parent_length", minimum=1)
    if parent != context + horizon:
        raise NativeContractError("input.parent_length must equal context_length + horizon_length")
    _string_list(item["channel_order"], "input.channel_order", nonempty=True,
                 identifiers=True)
    grouping = _enum(item["grouping_tag"], GROUPING_TAGS, "input.grouping_tag")
    expected_grouping = {
        "independent_univariate_passes": "no_cross_channel_interaction",
        "joint_multivariate": "same_parent_joint_channels",
        "grouped_multivariate": "same_parent_group_ids",
        "packed_multivariate": "packed_sample_time_variate_ids",
        "support_query_rows": "support_query_fold",
        "not_applicable": "not_applicable",
    }[layout]
    if grouping != expected_grouping:
        raise NativeContractError(
            f"input layout {layout!r} requires grouping {expected_grouping!r}"
        )
    _enum(item["dtype"], PRECISION_TAGS, "input.dtype")
    _enum(item["missing_policy_tag"], MISSING_POLICY_TAGS, "input.missing_policy_tag")
    _enum(item["guard_policy_tag"], GUARD_POLICY_TAGS, "input.guard_policy_tag")
    return context, horizon, parent, layout


def _validate_time(value: Any, *, horizon: int) -> None:
    item = _object(value, {
        "horizon_unit", "timestamp_tag", "venue_timezone_tag", "session_policy_tag",
        "roll_policy_tag",
    }, "time")
    unit = _enum(item["horizon_unit"], HORIZON_UNITS, "time.horizon_unit")
    if (horizon == 0) != (unit == "none"):
        raise NativeContractError("zero horizon requires horizon_unit='none' and vice versa")
    _enum(item["timestamp_tag"], TIMESTAMP_TAGS, "time.timestamp_tag")
    _enum(item["venue_timezone_tag"], TIMEZONE_TAGS, "time.venue_timezone_tag")
    _enum(item["session_policy_tag"], SESSION_POLICY_TAGS, "time.session_policy_tag")
    _enum(item["roll_policy_tag"], ROLL_POLICY_TAGS, "time.roll_policy_tag")


def _validate_preprocessing(value: Any) -> None:
    item = _object(value, {
        "owner_tag", "scaler_tag", "statistics_interval_tag", "external_scaler_tag",
        "mask_tag",
    }, "preprocessing")
    _enum(item["owner_tag"], PREPROCESSING_OWNERS, "preprocessing.owner_tag")
    _enum(item["scaler_tag"], SCALER_TAGS, "preprocessing.scaler_tag")
    _enum(
        item["statistics_interval_tag"], STATISTICS_INTERVAL_TAGS,
        "preprocessing.statistics_interval_tag",
    )
    if item["external_scaler_tag"] != "none":
        raise NativeContractError("preprocessing.external_scaler_tag must be 'none'")
    _enum(item["mask_tag"], MASK_TAGS, "preprocessing.mask_tag")


def _validate_target(value: Any, *, context: int, parent: int) -> str:
    item = _object(value, {
        "target_tag", "interval", "axes", "normalization_tag",
    }, "target")
    target = _enum(item["target_tag"], TARGET_TAGS, "target.target_tag")
    interval = item["interval"]
    non_temporal = {"class_label", "support_query_labels", "not_applicable"}
    if target in non_temporal:
        if interval is not None:
            raise NativeContractError(f"target {target!r} requires interval=null")
    else:
        if not isinstance(interval, list) or len(interval) != 2:
            raise NativeContractError("temporal target.interval must contain [start, end]")
        start = _integer(interval[0], "target.interval[0]")
        end = _integer(interval[1], "target.interval[1]")
        if not start < end <= parent:
            raise NativeContractError("target.interval must be nonempty and inside the parent")
        future = {"future_raw_channels", "future_tokens", "future_quantiles",
                  "forward_path_labels", "flow_noise_target"}
        context_targets = {
            "same_context_views", "masked_context_values", "tokenizer_reconstruction",
        }
        if target in future and [start, end] != [context, parent]:
            raise NativeContractError("future/path target interval must equal [context, parent]")
        if target in context_targets and [start, end] != [0, context]:
            raise NativeContractError("context target interval must equal [0, context]")
    _string_list(item["axes"], "target.axes", allowed=AXIS_TAGS, nonempty=True)
    _enum(item["normalization_tag"], NORMALIZATION_TAGS, "target.normalization_tag")
    return target


def _validate_objective(value: Any) -> str:
    item = _object(value, {"loss_tag", "reduction_tag", "parameters"}, "objective")
    loss = _enum(item["loss_tag"], LOSS_TAGS, "objective.loss_tag")
    _enum(item["reduction_tag"], REDUCTION_TAGS, "objective.reduction_tag")
    parameters = item["parameters"]
    schemas: dict[str, set[str]] = {
        "mantis_v1_one_way_info_nce": {
            "temperature", "direction_tag", "crop_sampling_scope_tag", "resize_length",
            "projector_dim",
        },
        "mantis_v2_info_nce": {
            "temperature", "direction_tag", "crop_sampling_scope_tag", "resize_length",
            "projector_dim",
        },
        "masked_raw_mse": {"mask_alignment_tag"},
        "classification_cross_entropy": {"class_count", "label_smoothing"},
        "raw_mse": set(),
        "quantile_pinball": {"quantiles", "native_prediction_length", "effective_horizon"},
        "point_plus_quantile": {
            "quantiles", "point_weight", "quantile_weight", "effective_horizon",
        },
        "token_cross_entropy": {"eos_policy_tag", "missing_label_id", "effective_horizon"},
        "tokenizer_reconstruction_bsq": {"reconstruction_loss_tag", "bsq_loss_tag"},
        "hierarchical_token_cross_entropy": {
            "coarse_loss_weight", "fine_loss_weight", "effective_horizon",
        },
        "path_composite": {"component_tags", "component_weights"},
        "timeflow": {"flow_loss_tag"},
        "support_query_native": {"fit_tag"},
    }
    params = _object(parameters, schemas[loss], f"objective.parameters[{loss}]")
    if loss.startswith("mantis_"):
        _number(params["temperature"], "objective.parameters.temperature", minimum=1e-12)
        _enum(params["direction_tag"], frozenset({"query_to_key", "symmetric"}),
              "objective.parameters.direction_tag")
        _enum(params["crop_sampling_scope_tag"],
              frozenset({"batch_global", "per_observation"}),
              "objective.parameters.crop_sampling_scope_tag")
        _integer(params["resize_length"], "objective.parameters.resize_length", minimum=1)
        _integer(params["projector_dim"], "objective.parameters.projector_dim", minimum=1)
    elif loss == "masked_raw_mse":
        _enum(params["mask_alignment_tag"],
              frozenset({"same_timestamp_across_channels"}),
              "objective.parameters.mask_alignment_tag")
    elif loss == "classification_cross_entropy":
        _integer(params["class_count"], "objective.parameters.class_count", minimum=2)
        smoothing = _number(params["label_smoothing"],
                            "objective.parameters.label_smoothing", minimum=0)
        if smoothing >= 1:
            raise NativeContractError("classification label smoothing must be < 1")
    elif loss in {"quantile_pinball", "point_plus_quantile"}:
        quantiles = params["quantiles"]
        if not isinstance(quantiles, list) or not quantiles:
            raise NativeContractError("objective quantiles must be a nonempty list")
        numbers = [_number(q, "objective.parameters.quantiles", minimum=0) for q in quantiles]
        if any(q <= 0 or q >= 1 for q in numbers) or numbers != sorted(set(numbers)):
            raise NativeContractError("objective quantiles must be unique, increasing, and in (0,1)")
        _integer(params["effective_horizon"], "objective.parameters.effective_horizon",
                 minimum=1)
        if loss == "quantile_pinball":
            _integer(params["native_prediction_length"],
                     "objective.parameters.native_prediction_length", minimum=1)
        else:
            point_weight = _number(
                params["point_weight"], "objective.parameters.point_weight", minimum=0,
            )
            quantile_weight = _number(
                params["quantile_weight"], "objective.parameters.quantile_weight",
                minimum=0,
            )
            if point_weight <= 0 or quantile_weight <= 0:
                raise NativeContractError("point and quantile objective weights must be positive")
    elif loss == "token_cross_entropy":
        _enum(params["eos_policy_tag"], frozenset({"immediately_after_horizon"}),
              "objective.parameters.eos_policy_tag")
        if params["missing_label_id"] != -100:
            raise NativeContractError(
                "Chronos token cross-entropy requires missing_label_id=-100"
            )
        _integer(params["effective_horizon"], "objective.parameters.effective_horizon",
                 minimum=1)
    elif loss == "tokenizer_reconstruction_bsq":
        _identifier_value(params["reconstruction_loss_tag"],
                          "objective.parameters.reconstruction_loss_tag")
        _identifier_value(params["bsq_loss_tag"], "objective.parameters.bsq_loss_tag")
    elif loss == "hierarchical_token_cross_entropy":
        coarse = _number(params["coarse_loss_weight"],
                         "objective.parameters.coarse_loss_weight", minimum=0)
        fine = _number(params["fine_loss_weight"],
                       "objective.parameters.fine_loss_weight", minimum=0)
        if coarse <= 0 or fine <= 0:
            raise NativeContractError("hierarchical objective weights must be positive")
        _integer(params["effective_horizon"], "objective.parameters.effective_horizon",
                 minimum=1)
    elif loss == "path_composite":
        components = _string_list(
            params["component_tags"], "objective.parameters.component_tags",
            allowed=PATH_COMPONENT_TAGS, nonempty=True,
        )
        weights = _object(
            params["component_weights"], set(components),
            "objective.parameters.component_weights",
        )
        for component in components:
            weight = _number(
                weights[component], f"objective.parameters.component_weights.{component}",
                minimum=0,
            )
            if weight <= 0:
                raise NativeContractError("path objective component weights must be positive")
    elif loss in {"timeflow", "support_query_native"}:
        key = "flow_loss_tag" if loss == "timeflow" else "fit_tag"
        _identifier_value(params[key], f"objective.parameters.{key}")
    return loss


def _validate_optimization(value: Any, *, pathway: str) -> None:
    item = _object(value, {
        "trainable_surface_tag", "frozen_surface_tag", "parameter_selector_sha256",
        "precision_tag", "optimizer_tag", "scheduler_tag", "learning_rate",
        "weight_decay", "betas", "epsilon", "batch_size",
        "gradient_accumulation_steps", "max_gradient_norm",
    }, "optimization")
    trainable = _enum(item["trainable_surface_tag"], TRAINABLE_SURFACE_TAGS,
                      "optimization.trainable_surface_tag")
    frozen = _enum(item["frozen_surface_tag"], FROZEN_SURFACE_TAGS,
                   "optimization.frozen_surface_tag")
    if frozen not in _SURFACE_PAIRS[trainable]:
        raise NativeContractError(
            f"invalid trainable/frozen surface pair: {trainable!r}/{frozen!r}"
        )
    _sha(item["parameter_selector_sha256"], "optimization.parameter_selector_sha256")
    _enum(item["precision_tag"], PRECISION_TAGS, "optimization.precision_tag")
    optimizer = _enum(item["optimizer_tag"], OPTIMIZER_TAGS, "optimization.optimizer_tag")
    scheduler = _enum(item["scheduler_tag"], SCHEDULER_TAGS,
                      "optimization.scheduler_tag")
    learning_rate = _number(
        item["learning_rate"], "optimization.learning_rate", minimum=0,
    )
    _number(item["weight_decay"], "optimization.weight_decay", minimum=0)
    betas = item["betas"]
    if not isinstance(betas, list) or len(betas) != 2:
        raise NativeContractError("optimization.betas must contain two values")
    beta_values = [_number(x, "optimization.betas", minimum=0) for x in betas]
    if any(x >= 1 for x in beta_values):
        raise NativeContractError("optimization betas must be < 1")
    epsilon = _number(item["epsilon"], "optimization.epsilon", minimum=0)
    _integer(item["batch_size"], "optimization.batch_size", minimum=1)
    _integer(item["gradient_accumulation_steps"],
             "optimization.gradient_accumulation_steps", minimum=1)
    _number(item["max_gradient_norm"], "optimization.max_gradient_norm", minimum=0)
    if pathway == "in_context_fit":
        if trainable != "no_persistent_weights" or optimizer != "none" or scheduler != "none":
            raise NativeContractError("in-context fit cannot declare persistent optimization")
    elif trainable == "no_persistent_weights" or optimizer == "none":
        raise NativeContractError("optimizer training requires a persistent trainable surface")
    elif learning_rate <= 0 or epsilon <= 0:
        raise NativeContractError("optimizer training requires positive learning_rate and epsilon")


def _validate_lineage(value: Any, *, pathway: str) -> None:
    item = _object(value, {
        "initialization_tag", "allowed_parent_route_tags", "forbidden_parent_route_tags",
        "parent_artifact_requirements", "parent_input_slots",
    }, "lineage")
    initialization = _enum(item["initialization_tag"], INITIALIZATION_TAGS,
                           "lineage.initialization_tag")
    allowed = _string_list(item["allowed_parent_route_tags"],
                           "lineage.allowed_parent_route_tags", identifiers=True)
    forbidden = _string_list(item["forbidden_parent_route_tags"],
                             "lineage.forbidden_parent_route_tags", identifiers=True)
    if set(allowed) & set(forbidden):
        raise NativeContractError("lineage allowed and forbidden parents overlap")
    requirements = _string_list(item["parent_artifact_requirements"],
                                "lineage.parent_artifact_requirements",
                                allowed=PIPELINE_ARTIFACT_TAGS)
    slots = _string_list(
        item["parent_input_slots"], "lineage.parent_input_slots", identifiers=True,
    )
    has_parent_contract = bool(allowed or requirements or slots)
    if (initialization == "parent_route_artifact") != has_parent_contract:
        raise NativeContractError(
            "parent-route initialization and artifact requirements must agree"
        )
    if initialization == "parent_route_artifact" and not (
        allowed and requirements and slots
    ):
        raise NativeContractError(
            "parent-route initialization requires parent routes, artifacts, and input slots"
        )
    if pathway == "in_context_fit" and initialization != "support_query_only":
        raise NativeContractError("in-context fit requires support_query_only initialization")


def _validate_resume(value: Any, *, pathway: str) -> None:
    item = _object(value, {
        "required_state", "exact_next_batch_required", "exact_next_loss_required",
    }, "resume")
    state = _string_list(item["required_state"], "resume.required_state", identifiers=True)
    exact_batch = _bool(item["exact_next_batch_required"],
                        "resume.exact_next_batch_required")
    exact_loss = _bool(item["exact_next_loss_required"],
                       "resume.exact_next_loss_required")
    if pathway == "optimizer_training":
        if set(state) != set(_FULL_RESUME_STATE) or not exact_batch or not exact_loss:
            raise NativeContractError("optimizer training requires the complete exact-resume state")
    elif state or exact_batch or exact_loss:
        raise NativeContractError("in-context fit cannot claim persistent resume state")


def _validate_export(value: Any) -> str:
    item = _object(value, {
        "bundle_tag", "output_tag", "deployment_route_tag",
        "preprocessing_hash_required", "output_slots",
    }, "export")
    _enum(item["bundle_tag"], BUNDLE_TAGS, "export.bundle_tag")
    output = _enum(item["output_tag"], OUTPUT_TAGS, "export.output_tag")
    _identifier_value(item["deployment_route_tag"], "export.deployment_route_tag")
    _string_list(item["output_slots"], "export.output_slots", identifiers=True,
                 nonempty=True)
    if not _bool(item["preprocessing_hash_required"],
                 "export.preprocessing_hash_required"):
        raise NativeContractError("deployment export must bind preprocessing")
    return output


def _validate_governance(value: Any, *, unsupported: bool = False) -> None:
    item = _object(value, {
        "license_id", "permitted_use_scopes", "terms_evidence_sha256",
    }, "governance")
    _string(item["license_id"], "governance.license_id")
    scopes = _string_list(
        item["permitted_use_scopes"], "governance.permitted_use_scopes",
        allowed=USE_SCOPES, nonempty=not unsupported,
    )
    if unsupported and scopes:
        raise NativeContractError("unsupported routes must have no permitted use scope")
    _sha(item["terms_evidence_sha256"], "governance.terms_evidence_sha256", nullable=True)


def validate_route_template(
    value: Any, *, family_contracts_by_sha256: Mapping[str, Any] | None = None,
    governance_evidence_by_sha256: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate and return an isolated copy of one strict v2 route template."""
    if not isinstance(value, Mapping):
        raise NativeContractError("route template must be an object")
    identity = _validate_identity(value.get("identity"))
    pathway = str(identity["pathway_kind"])
    status = _enum(value.get("template_status"), TEMPLATE_STATUSES, "template_status")
    blocker_tags = _string_list(
        value.get("blocker_tags"), "blocker_tags", allowed=UNSUPPORTED_BLOCKER_TAGS,
        nonempty=status == "blocked",
    )
    if status == "verified" and blocker_tags:
        raise NativeContractError("verified route templates cannot retain blockers")
    family_contract_sha = value.get("family_contract_sha256")
    governance_evidence_sha = value.get("governance_evidence_sha256")
    if status == "verified":
        # Resolution happens below after exact top-level field validation.
        _sha(family_contract_sha, "family_contract_sha256")
        _sha(governance_evidence_sha, "governance_evidence_sha256")
    else:
        _sha(family_contract_sha, "family_contract_sha256", nullable=True)
        _sha(governance_evidence_sha, "governance_evidence_sha256", nullable=True)
    if pathway == "unsupported":
        item = _object(value, {
            "schema_version", "template_status", "blocker_tags", "identity",
            "family_contract_sha256", "governance_evidence_sha256", "base_binding",
            "governance",
        }, "route template")
        if item["schema_version"] != ROUTE_TEMPLATE_SCHEMA:
            raise NativeContractError(f"route template schema must be {ROUTE_TEMPLATE_SCHEMA!r}")
        if status != "blocked":
            raise NativeContractError("unsupported routes must remain blocked")
        _validate_base_binding(item["base_binding"], status=status)
        _validate_governance(item["governance"], unsupported=True)
        return deepcopy(dict(item))

    item = _object(value, {
        "schema_version", "template_status", "blocker_tags", "identity",
        "family_contract_sha256", "governance_evidence_sha256", "base_binding", "input", "time",
        "preprocessing", "target", "objective", "optimization", "lineage", "resume",
        "export", "governance", "smoke_profile_tag",
    }, "route template")
    if item["schema_version"] != ROUTE_TEMPLATE_SCHEMA:
        raise NativeContractError(f"route template schema must be {ROUTE_TEMPLATE_SCHEMA!r}")
    if status == "verified":
        _resolve_hash_mapping(
            item["family_contract_sha256"], family_contracts_by_sha256,
            "family_contract_sha256",
        )
        _resolve_hash_mapping(
            item["governance_evidence_sha256"], governance_evidence_by_sha256,
            "governance_evidence_sha256",
        )
    _validate_base_binding(item["base_binding"], status=status)
    context, horizon, parent, layout = _validate_input(item["input"])
    _validate_time(item["time"], horizon=horizon)
    _validate_preprocessing(item["preprocessing"])
    target = _validate_target(item["target"], context=context, parent=parent)
    loss = _validate_objective(item["objective"])
    _validate_optimization(item["optimization"], pathway=pathway)
    _validate_lineage(item["lineage"], pathway=pathway)
    _validate_resume(item["resume"], pathway=pathway)
    output = _validate_export(item["export"])
    _validate_governance(item["governance"])
    smoke = _enum(item["smoke_profile_tag"], SMOKE_PROFILE_TAGS, "smoke_profile_tag")
    task = str(identity["task_kind"])
    if loss not in _TASK_LOSSES[task]:
        raise NativeContractError(f"task {task!r} is incompatible with loss {loss!r}")
    if target not in _TASK_TARGETS[task]:
        raise NativeContractError(f"task {task!r} is incompatible with target {target!r}")
    if identity["track"] not in _TASK_TRACKS[task]:
        raise NativeContractError(
            f"task {task!r} is incompatible with track {identity['track']!r}"
        )
    if output not in _TASK_OUTPUTS[task]:
        raise NativeContractError(f"task {task!r} is incompatible with output {output!r}")
    if smoke not in _TASK_SMOKE_PROFILES[task]:
        raise NativeContractError(
            f"task {task!r} is incompatible with smoke profile {smoke!r}"
        )
    context_only_tasks = {
        "contrastive_representation", "masked_reconstruction", "classification",
        "tokenizer_reconstruction", "support_query_downstream",
    }
    if (task in context_only_tasks) != (horizon == 0):
        raise NativeContractError(
            f"task {task!r} has incompatible context/horizon geometry"
        )
    if task.startswith("support_query_"):
        if layout != "support_query_rows" or item["input"]["guard_policy_tag"] != "support_query_purged":
            raise NativeContractError(
                "support-query tasks require support_query_rows and support_query_purged"
            )
    elif layout == "support_query_rows":
        raise NativeContractError("support_query_rows layout requires a support-query task")
    parameters = item["objective"]["parameters"]
    if "effective_horizon" in parameters and parameters["effective_horizon"] != horizon:
        raise NativeContractError("objective effective_horizon must equal input horizon_length")
    if "native_prediction_length" in parameters and (
        parameters["native_prediction_length"] < horizon
    ):
        raise NativeContractError("native prediction length cannot be shorter than the horizon")
    if pathway == "in_context_fit" and loss != "support_query_native":
        raise NativeContractError("in-context fit requires support_query_native objective")
    if target == "support_query_labels" and pathway != "in_context_fit":
        raise NativeContractError("support-query labels require an in-context route")
    return deepcopy(dict(item))


def route_template_sha256(
    value: Any, *, family_contracts_by_sha256: Mapping[str, Any] | None = None,
    governance_evidence_by_sha256: Mapping[str, Any] | None = None,
) -> str:
    return content_sha256(validate_route_template(
        value, family_contracts_by_sha256=family_contracts_by_sha256,
        governance_evidence_by_sha256=governance_evidence_by_sha256,
    ))


def _instance_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: deepcopy(item) for key, item in value.items() if key != "instance_sha256"}


def validate_route_instance(
    value: Any, *, templates_by_sha256: Mapping[str, Any],
    family_contracts_by_sha256: Mapping[str, Any],
    governance_evidence_by_sha256: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate an exact template plus governed data-authority binding."""
    item = _object(value, {
        "schema_version", "template_sha256", "data_binding", "exact_choices",
        "template_evidence_bundle_sha256", "instance_sha256",
    }, "route instance")
    if item["schema_version"] != ROUTE_INSTANCE_SCHEMA:
        raise NativeContractError(f"route instance schema must be {ROUTE_INSTANCE_SCHEMA!r}")
    template_sha = _sha(item["template_sha256"], "route instance.template_sha256")
    _sha(item["template_evidence_bundle_sha256"],
         "route instance.template_evidence_bundle_sha256")
    data = _object(item["data_binding"], {
        "corpus_contract_sha256", "sample_manifest_sha256", "split_contract_sha256",
        "session_denominator_sha256", "expected_request_denominator_sha256",
        "lifecycle_registry_sha256", "roll_registry_sha256", "label_contract_sha256",
        "allowed_training_splits", "allowed_validation_splits", "forbidden_splits",
        "timeframes_minutes",
    }, "route instance.data_binding")
    for name in (
        "corpus_contract_sha256", "sample_manifest_sha256", "split_contract_sha256",
        "session_denominator_sha256", "expected_request_denominator_sha256",
        "lifecycle_registry_sha256", "roll_registry_sha256",
    ):
        _sha(data[name], f"route instance.data_binding.{name}")
    _sha(data["label_contract_sha256"],
         "route instance.data_binding.label_contract_sha256", nullable=True)
    training = _string_list(data["allowed_training_splits"],
                            "route instance.data_binding.allowed_training_splits",
                            allowed=frozenset({"pretrain", "shared_train"}), nonempty=True)
    validation = _string_list(data["allowed_validation_splits"],
                              "route instance.data_binding.allowed_validation_splits",
                              allowed=frozenset({"development"}), nonempty=True)
    forbidden = _string_list(data["forbidden_splits"],
                             "route instance.data_binding.forbidden_splits", nonempty=True)
    required_holdouts = {"legacy_holdout", "prospective_holdout"}
    if not required_holdouts.issubset(forbidden):
        raise NativeContractError(
            "route instance must explicitly forbid legacy_holdout and prospective_holdout"
        )
    if set(training + validation) & set(forbidden):
        raise NativeContractError("allowed and forbidden route-instance splits overlap")
    for name in training + validation:
        lowered = name.lower()
        if "holdout" in lowered or "oos" in lowered:
            raise NativeContractError("OOS/holdout split cannot be authorized")
    frames = data["timeframes_minutes"]
    if not isinstance(frames, list) or not frames:
        raise NativeContractError("route instance timeframes must be nonempty")
    parsed_frames = [
        _integer(frame, "route instance.data_binding.timeframes_minutes", minimum=1)
        for frame in frames
    ]
    if parsed_frames != sorted(set(parsed_frames)):
        raise NativeContractError("route instance timeframes must be unique and increasing")
    choices = _object(item["exact_choices"], {
        "amount_source_tag", "venue_timezone_registry_sha256",
        "session_calendar_registry_sha256",
    }, "route instance.exact_choices")
    _enum(choices["amount_source_tag"], AMOUNT_SOURCE_TAGS,
          "route instance.exact_choices.amount_source_tag")
    _sha(choices["venue_timezone_registry_sha256"],
         "route instance.exact_choices.venue_timezone_registry_sha256")
    _sha(choices["session_calendar_registry_sha256"],
         "route instance.exact_choices.session_calendar_registry_sha256")
    instance_sha = _sha(item["instance_sha256"], "route instance.instance_sha256")
    expected_sha = content_sha256(_instance_payload(item))
    if instance_sha != expected_sha:
        raise NativeContractError("route instance hash does not bind its complete payload")
    template = templates_by_sha256.get(str(template_sha))
    if template is None:
        raise NativeContractError("route instance references an unknown template")
    validated_template = validate_route_template(
        template, family_contracts_by_sha256=family_contracts_by_sha256,
        governance_evidence_by_sha256=governance_evidence_by_sha256,
    )
    if route_template_sha256(
        validated_template, family_contracts_by_sha256=family_contracts_by_sha256,
        governance_evidence_by_sha256=governance_evidence_by_sha256,
    ) != template_sha:
        raise NativeContractError("route instance template mapping is stale or swapped")
    if validated_template["template_status"] != "verified":
        raise NativeContractError("only verified route templates can have route instances")
    if validated_template["identity"]["pathway_kind"] == "unsupported":
        raise NativeContractError("unsupported route templates cannot have route instances")
    return deepcopy(dict(item))


def build_route_instance(
    value_without_instance_sha256: Mapping[str, Any], *,
    templates_by_sha256: Mapping[str, Any],
    family_contracts_by_sha256: Mapping[str, Any],
    governance_evidence_by_sha256: Mapping[str, Any],
) -> dict[str, Any]:
    """Add the deterministic self-hash, then apply full validation."""
    if "instance_sha256" in value_without_instance_sha256:
        raise NativeContractError("route instance builder input must omit instance_sha256")
    value = deepcopy(dict(value_without_instance_sha256))
    value["instance_sha256"] = content_sha256(value)
    return validate_route_instance(
        value, templates_by_sha256=templates_by_sha256,
        family_contracts_by_sha256=family_contracts_by_sha256,
        governance_evidence_by_sha256=governance_evidence_by_sha256,
    )


def validate_pipeline_dag(
    value: Any, *, instances_by_sha256: Mapping[str, Any],
    templates_by_sha256: Mapping[str, Any],
    family_contracts_by_sha256: Mapping[str, Any],
    governance_evidence_by_sha256: Mapping[str, Any],
    artifact_manifests_by_sha256: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate exact node closure, edge closure, acyclicity, and the pipeline self-hash."""
    item = _object(value, {
        "schema_version", "pipeline_id", "nodes", "edges", "pipeline_sha256",
    }, "pipeline")
    if item["schema_version"] != PIPELINE_SCHEMA:
        raise NativeContractError(f"pipeline schema must be {PIPELINE_SCHEMA!r}")
    _identifier_value(item["pipeline_id"], "pipeline.pipeline_id")
    nodes = item["nodes"]
    if not isinstance(nodes, list) or not nodes:
        raise NativeContractError("pipeline.nodes must be a nonempty list")
    node_map: dict[str, str] = {}
    node_templates: dict[str, dict[str, Any]] = {}
    referenced_template_shas: set[str] = set()
    for index, raw in enumerate(nodes):
        node = _object(raw, {"node_id", "route_instance_sha256"},
                       f"pipeline.nodes[{index}]")
        node_id = _identifier_value(node["node_id"], f"pipeline.nodes[{index}].node_id")
        instance_sha = _sha(node["route_instance_sha256"],
                            f"pipeline.nodes[{index}].route_instance_sha256")
        if node_id in node_map or instance_sha in node_map.values():
            raise NativeContractError("pipeline nodes and route instances must be unique")
        instance = instances_by_sha256.get(str(instance_sha))
        if instance is None:
            raise NativeContractError("pipeline references an unknown route instance")
        validated = validate_route_instance(
            instance, templates_by_sha256=templates_by_sha256,
            family_contracts_by_sha256=family_contracts_by_sha256,
            governance_evidence_by_sha256=governance_evidence_by_sha256,
        )
        if validated["instance_sha256"] != instance_sha:
            raise NativeContractError("pipeline route-instance mapping is stale or swapped")
        node_map[node_id] = str(instance_sha)
        template_sha = str(validated["template_sha256"])
        referenced_template_shas.add(template_sha)
        node_templates[node_id] = validate_route_template(
            templates_by_sha256[template_sha],
            family_contracts_by_sha256=family_contracts_by_sha256,
            governance_evidence_by_sha256=governance_evidence_by_sha256,
        )
    if set(instances_by_sha256) != set(node_map.values()):
        raise NativeContractError("pipeline instance mapping must have exact node closure")
    if set(templates_by_sha256) != referenced_template_shas:
        raise NativeContractError("pipeline template mapping must have exact node closure")
    edges = item["edges"]
    if not isinstance(edges, list):
        raise NativeContractError("pipeline.edges must be a list")
    adjacency = {node_id: set() for node_id in node_map}
    edge_keys: set[tuple[str, str, str, str]] = set()
    incoming: dict[str, list[dict[str, str]]] = {node: [] for node in node_map}
    referenced_manifest_shas: set[str] = set()
    for index, raw in enumerate(edges):
        edge = _object(raw, {
            "parent_node_id", "child_node_id", "artifact_tag", "artifact_sha256",
            "manifest_sha256", "parent_output_slot", "child_input_slot",
        }, f"pipeline.edges[{index}]")
        parent = _identifier_value(edge["parent_node_id"],
                                   f"pipeline.edges[{index}].parent_node_id")
        child = _identifier_value(edge["child_node_id"],
                                  f"pipeline.edges[{index}].child_node_id")
        artifact = _enum(edge["artifact_tag"], PIPELINE_ARTIFACT_TAGS,
                         f"pipeline.edges[{index}].artifact_tag")
        artifact_sha = _sha(
            edge["artifact_sha256"], f"pipeline.edges[{index}].artifact_sha256",
        )
        manifest_sha = _sha(
            edge["manifest_sha256"], f"pipeline.edges[{index}].manifest_sha256",
        )
        parent_output_slot = _identifier_value(
            edge["parent_output_slot"], f"pipeline.edges[{index}].parent_output_slot",
        )
        child_input_slot = _identifier_value(
            edge["child_input_slot"], f"pipeline.edges[{index}].child_input_slot",
        )
        if parent not in node_map or child not in node_map or parent == child:
            raise NativeContractError("pipeline edge has invalid node closure")
        manifest = artifact_manifests_by_sha256.get(str(manifest_sha))
        if manifest is None or content_sha256(manifest) != manifest_sha:
            raise NativeContractError("pipeline artifact manifest is missing, stale, or swapped")
        manifest_item = _object(manifest, {
            "schema_version", "artifact_sha256", "artifact_tag",
            "producer_route_instance_sha256", "output_slot",
        }, f"pipeline artifact manifest {manifest_sha}")
        referenced_manifest_shas.add(str(manifest_sha))
        if manifest_item["schema_version"] != PIPELINE_ARTIFACT_MANIFEST_SCHEMA:
            raise NativeContractError("pipeline artifact manifest schema is invalid")
        if (
            manifest_item["artifact_sha256"] != artifact_sha
            or manifest_item["artifact_tag"] != artifact
            or manifest_item["producer_route_instance_sha256"] != node_map[parent]
            or manifest_item["output_slot"] != parent_output_slot
        ):
            raise NativeContractError("pipeline edge does not match its artifact manifest")
        parent_template = node_templates[parent]
        child_template = node_templates[child]
        if parent_output_slot not in parent_template["export"]["output_slots"]:
            raise NativeContractError("pipeline parent output slot is undeclared")
        lineage = child_template["lineage"]
        parent_route_id = str(parent_template["identity"]["route_id"])
        if lineage["initialization_tag"] != "parent_route_artifact":
            raise NativeContractError("pipeline child does not declare parent-route initialization")
        if (
            parent_route_id not in lineage["allowed_parent_route_tags"]
            or parent_route_id in lineage["forbidden_parent_route_tags"]
            or artifact not in lineage["parent_artifact_requirements"]
            or child_input_slot not in lineage["parent_input_slots"]
        ):
            raise NativeContractError("pipeline edge violates the child lineage contract")
        key = (parent, child, artifact, child_input_slot)
        if key in edge_keys:
            raise NativeContractError("pipeline contains a duplicate edge")
        edge_keys.add(key)
        adjacency[parent].add(child)
        incoming[child].append({
            "parent_route_id": parent_route_id, "artifact_tag": artifact,
            "child_input_slot": child_input_slot,
        })
    if set(artifact_manifests_by_sha256) != referenced_manifest_shas:
        raise NativeContractError("pipeline artifact-manifest mapping must have exact edge closure")
    indegree = {node: 0 for node in node_map}
    for children in adjacency.values():
        for child in children:
            indegree[child] += 1
    for node_id, degree in indegree.items():
        lineage = node_templates[node_id]["lineage"]
        initialization = lineage["initialization_tag"]
        if initialization in {
            "vanilla_pinned_checkpoint", "fresh_head_on_pinned_base", "support_query_only",
        } and degree != 0:
            raise NativeContractError("vanilla/fresh/support-query pipeline nodes require indegree zero")
        if initialization == "parent_route_artifact":
            if degree == 0:
                raise NativeContractError("parent-initialized pipeline node has no parent edge")
            incoming_items = incoming[node_id]
            if (
                {item["parent_route_id"] for item in incoming_items}
                != set(lineage["allowed_parent_route_tags"])
                or {item["artifact_tag"] for item in incoming_items}
                != set(lineage["parent_artifact_requirements"])
                or {item["child_input_slot"] for item in incoming_items}
                != set(lineage["parent_input_slots"])
            ):
                raise NativeContractError("pipeline child parent/artifact/slot closure is not exact")
    queue = [node for node, degree in indegree.items() if degree == 0]
    visited = 0
    while queue:
        node = queue.pop()
        visited += 1
        for child in adjacency[node]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    if visited != len(node_map):
        raise NativeContractError("pipeline graph must be acyclic")
    pipeline_sha = _sha(item["pipeline_sha256"], "pipeline.pipeline_sha256")
    payload = {key: deepcopy(raw) for key, raw in item.items() if key != "pipeline_sha256"}
    if pipeline_sha != content_sha256(payload):
        raise NativeContractError("pipeline hash does not bind its complete payload")
    return deepcopy(dict(item))


def build_pipeline_dag(
    value_without_pipeline_sha256: Mapping[str, Any], *,
    instances_by_sha256: Mapping[str, Any], templates_by_sha256: Mapping[str, Any],
    family_contracts_by_sha256: Mapping[str, Any],
    governance_evidence_by_sha256: Mapping[str, Any],
    artifact_manifests_by_sha256: Mapping[str, Any],
) -> dict[str, Any]:
    if "pipeline_sha256" in value_without_pipeline_sha256:
        raise NativeContractError("pipeline builder input must omit pipeline_sha256")
    value = deepcopy(dict(value_without_pipeline_sha256))
    value["pipeline_sha256"] = content_sha256(value)
    return validate_pipeline_dag(
        value, instances_by_sha256=instances_by_sha256,
        templates_by_sha256=templates_by_sha256,
        family_contracts_by_sha256=family_contracts_by_sha256,
        governance_evidence_by_sha256=governance_evidence_by_sha256,
        artifact_manifests_by_sha256=artifact_manifests_by_sha256,
    )


def evidence_key(
    route_template: Any, route_instance: Any, *,
    family_contracts_by_sha256: Mapping[str, Any],
    governance_evidence_by_sha256: Mapping[str, Any],
) -> str:
    """Return the canonical evidence key for one exact method/data combination."""
    template = validate_route_template(
        route_template, family_contracts_by_sha256=family_contracts_by_sha256,
        governance_evidence_by_sha256=governance_evidence_by_sha256,
    )
    template_sha = route_template_sha256(
        template, family_contracts_by_sha256=family_contracts_by_sha256,
        governance_evidence_by_sha256=governance_evidence_by_sha256,
    )
    instance = validate_route_instance(
        route_instance, templates_by_sha256={template_sha: template},
        family_contracts_by_sha256=family_contracts_by_sha256,
        governance_evidence_by_sha256=governance_evidence_by_sha256,
    )
    identity = template["identity"]
    return ":".join((
        str(identity["arm_key"]), str(identity["track"]), str(identity["route_id"]),
        template_sha, str(instance["instance_sha256"]),
    ))
