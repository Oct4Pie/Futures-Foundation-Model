"""Cross-layer configuration audit for every foundation-model identity and route.

This module answers two separate questions:

* Is every declared configuration internally consistent or explicitly fail-closed?
* Is every model and training route execution-ready?

The first may pass while the second remains false.  Blocked checkpoints, unresolved optimizer
contracts, unsupported upstream training methods and failed empirical gates are valid fail-closed
states; silently guessing their settings is not.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import re
from typing import Any, Mapping

from .native_contracts import (
    ADMITTED_STATUSES,
    NativeContractError,
    _validate_evidence,
    content_sha256,
    load_registry,
    registry_sha256,
)
from .native_evidence_bundle import verify_parity_bundle
from .native_family_route_catalog_v2 import catalog_sha256, load_family_route_catalog
from .routes import (
    chronos_bolt,
    chronos_v1,
    chronos2_native,
    kronos_predictor,
    kronos_tokenizer,
    mantis_native,
    moirai2_research,
    moment_reconstruction,
    moment_tasks,
    timesfm_lora,
    ttm_native,
)


CONFIGURATION_AUDIT_SCHEMA = "ffm_native_configuration_audit_v1"
CONFIGURATION_AUDIT_POLICY = "cross_layer_fail_closed_model_configuration_v1"


_EXACT_ROUTE_SPECS: dict[str, dict[str, Any]] = {
    chronos_bolt.ROUTE_KEY: {
        "module": chronos_bolt,
        "profile": "chronos_bolt_forecast",
        "input": {
            "layout_tag": "independent_univariate_passes",
            "context_length": chronos_bolt.CONTEXT_LENGTH,
            "horizon_length": chronos_bolt.HORIZON_LENGTH,
            "parent_length": chronos_bolt.PARENT_LENGTH,
            "channel_order": list(chronos_bolt.CHANNELS),
            "grouping_tag": "no_cross_channel_interaction",
            "dtype": "fp32",
        },
        "time": {
            "horizon_unit": "bars", "timestamp_tag": "utc_bar_close",
            "timezone_tag": "utc",
        },
        "export": {
            "bundle_tag": "forecast_bundle",
            "output_tag": "native_quantiles_first16_no_hidden_state",
        },
        "optimizer": {"name": "AdamW", "betas": [0.9, 0.999], "eps": 1e-8},
        "scheduler": "CosineAnnealingLR",
        "scheduler_tag": "cosine_no_warmup",
    },
    chronos_v1.ROUTE_KEY: {
        "module": chronos_v1,
        "profile": "chronos_v1_forecast",
        "input": {
            "layout_tag": "independent_univariate_passes",
            "context_length": chronos_v1.CONTEXT_LENGTH,
            "horizon_length": chronos_v1.HORIZON_LENGTH,
            "parent_length": chronos_v1.PARENT_LENGTH,
            "channel_order": list(chronos_v1.CHANNELS),
            "grouping_tag": "no_cross_channel_interaction",
            "dtype": "fp32",
        },
        "time": {
            "horizon_unit": "bars", "timestamp_tag": "utc_bar_close",
            "timezone_tag": "utc",
        },
        "export": {
            "bundle_tag": "forecast_bundle",
            "deployment_filter_tag": "first_16_of_native_64",
            "output_tag": "native_samples_64_first16_deployment_no_hidden_state",
        },
        "optimizer": {"name": "AdamW", "betas": [0.9, 0.999], "eps": 1e-8},
        "scheduler": "CosineAnnealingLR",
        "scheduler_tag": "cosine_no_warmup",
    },
    moment_reconstruction.ROUTE_KEY: {
        "module": moment_reconstruction,
        "profile": "moment_reconstruction",
        "input": {
            "layout_tag": "B_C_512_internal_fold_only",
            "context_length": moment_reconstruction.CONTEXT_LENGTH,
            "horizon_length": 0,
            "parent_length": moment_reconstruction.PARENT_LENGTH,
            "channel_order": list(moment_reconstruction.CHANNELS),
            "grouping_tag": "no_cross_channel_interaction",
            "dtype": "fp32",
        },
        "time": {
            "horizon_unit": "none", "timestamp_tag": "utc_bar_close",
            "timezone_tag": "utc",
        },
        "export": {
            "bundle_tag": "representation_bundle",
            "output_tag": "masked_embedding_mean",
        },
        "optimizer": {"name": "AdamW", "betas": [0.9, 0.95], "eps": 1e-8},
        "scheduler": "CosineAnnealingLR",
        "scheduler_tag": "cosine_no_warmup",
    },
    kronos_tokenizer.route_key("kronos_mini"): {
        "module": kronos_tokenizer,
        "arm_key": "kronos_mini",
        "profile": "kronos_mini_tokenizer",
        "input": {
            "layout_tag": "joint_multivariate",
            "context_length": kronos_tokenizer.CONTEXT_LENGTH,
            "horizon_length": 0,
            "parent_length": kronos_tokenizer.PARENT_LENGTH,
            "channel_order": list(kronos_tokenizer.NATIVE_CHANNELS),
            "grouping_tag": "same_parent_joint_channels",
            "dtype": "fp32",
        },
        "time": {
            "horizon_unit": "none", "timestamp_tag": "not_consumed_by_tokenizer",
            "timezone_tag": "not_applicable",
        },
        "export": {
            "bundle_tag": "kronos_mini_tokenizer_2k_bundle",
            "output_tag": "coarse_and_fine_codes",
        },
        "optimizer": {"name": "AdamW", "betas": [0.9, 0.95], "eps": 1e-8},
        "scheduler": "CosineAnnealingLR",
        "scheduler_tag": "cosine_no_warmup",
    },
    kronos_tokenizer.route_key("kronos_small"): {
        "module": kronos_tokenizer,
        "arm_key": "kronos_small",
        "profile": "kronos_small_tokenizer",
        "input": {
            "layout_tag": "joint_multivariate",
            "context_length": kronos_tokenizer.CONTEXT_LENGTH,
            "horizon_length": 0,
            "parent_length": kronos_tokenizer.PARENT_LENGTH,
            "channel_order": list(kronos_tokenizer.NATIVE_CHANNELS),
            "grouping_tag": "same_parent_joint_channels",
            "dtype": "fp32",
        },
        "time": {
            "horizon_unit": "none", "timestamp_tag": "not_consumed_by_tokenizer",
            "timezone_tag": "not_applicable",
        },
        "export": {
            "bundle_tag": "kronos_small_tokenizer_base_bundle",
            "output_tag": "coarse_and_fine_codes",
        },
        "optimizer": {"name": "AdamW", "betas": [0.9, 0.95], "eps": 1e-8},
        "scheduler": "CosineAnnealingLR",
        "scheduler_tag": "cosine_no_warmup",
    },
    kronos_predictor.ROUTE_KEY: {
        "module": kronos_predictor,
        "profile": "kronos_mini_predictor",
        "input": {
            "layout_tag": "joint_multivariate",
            "context_length": kronos_predictor.CONTEXT_LENGTH,
            "horizon_length": kronos_predictor.HORIZON_LENGTH,
            "parent_length": kronos_predictor.PARENT_LENGTH,
            "channel_order": list(kronos_predictor.NATIVE_CHANNELS),
            "grouping_tag": "same_parent_joint_channels",
            "dtype": "fp32",
            "stamp_shape": [kronos_predictor.PARENT_LENGTH, len(kronos_predictor.STAMP_CHANNELS)],
        },
        "time": {
            "horizon_unit": "bars",
            "timestamp_tag": "venue_local_minute_hour_weekday_day_month",
            "timezone_tag": "america_chicago_cme_venue_v1",
        },
        "export": {
            "bundle_tag": "kronos_mini_predictor_plus_tokenizer_bundle",
            "output_tag": "joint_ohlcva_forecast",
        },
        "optimizer": {"name": "AdamW", "betas": [0.9, 0.95], "eps": 1e-8},
        "scheduler": "OneCycleLR",
        "scheduler_tag": "one_cycle_pct03_div10_final1e4_cycle_beta1_085_095_cos_native",
        "scheduler_runtime": {
            "cycle_momentum": True,
            "use_beta1": True,
            "base_momentum": 0.85,
            "max_momentum": 0.95,
            "anneal_strategy": "cos",
        },
    },
}

for _route_key, _route_spec in chronos2_native.ROUTES.items():
    _EXACT_ROUTE_SPECS[_route_key] = {
        "module": chronos2_native,
        "profile": _route_spec["profile"],
        "config_factory": lambda key=_route_key: chronos2_native.RouteConfig(key),
        "optimizer_factory": "parameters_cuda",
        "config_has_accumulation_field": False,
        "input": {
            "layout_tag": "grouped_multivariate",
            "context_length": chronos2_native.CONTEXT_LENGTH,
            "horizon_length": chronos2_native.HORIZON_LENGTH,
            "parent_length": chronos2_native.CONTEXT_LENGTH + chronos2_native.HORIZON_LENGTH,
            "channel_order": list(chronos2_native.CHANNELS),
            "grouping_tag": "same_parent_group_ids", "dtype": "fp32",
        },
        "time": {
            "horizon_unit": "bars", "timestamp_tag": "utc_bar_close", "timezone_tag": "utc",
        },
        "export": (
            {"bundle_tag": "forecast_bundle", "output_tag": "grouped_quantiles_public_tokens"}
            if _route_spec["surface"] == "full" else
            {"bundle_tag": "base_plus_adapter", "output_tag": "grouped_quantiles_public_tokens"}
        ),
        "optimizer": {"name": "AdamW", "betas": [0.9, 0.999], "eps": 1e-8},
        "scheduler": "LambdaLR", "scheduler_tag": "linear_no_warmup",
    }

_small_predictor_spec = dict(_EXACT_ROUTE_SPECS[kronos_predictor.ROUTE_KEY])
_small_predictor_spec.update({
    "profile": "kronos_small_predictor",
    "config_factory": lambda: kronos_predictor.RouteConfig(arm_key="kronos_small"),
    "export": {
        "bundle_tag": "kronos_small_predictor_plus_tokenizer_bundle",
        "output_tag": "joint_ohlcva_forecast",
    },
})
_EXACT_ROUTE_SPECS[kronos_predictor.route_key("kronos_small")] = _small_predictor_spec

_EXACT_ROUTE_SPECS[moirai2_research.ROUTE_KEY] = {
    "module": moirai2_research,
    "profile": moirai2_research.PROFILE,
    "optimizer_factory": "parameters",
    "config_has_accumulation_field": False,
    "input": {
        "layout_tag": "packed_multivariate",
        "context_length": moirai2_research.CONTEXT_LENGTH,
        "horizon_length": moirai2_research.HORIZON_LENGTH,
        "parent_length": moirai2_research.CONTEXT_LENGTH + moirai2_research.HORIZON_LENGTH,
        "channel_order": list(moirai2_research.CHANNELS),
        "grouping_tag": "sample_time_variate_ids", "dtype": "fp32",
    },
    "time": {
        "horizon_unit": "bars", "timestamp_tag": "utc_bar_close", "timezone_tag": "utc",
    },
    "export": {"bundle_tag": "research_forecast_bundle", "output_tag": "probabilistic_quantiles"},
    "optimizer": {"name": "AdamW", "betas": [0.9, 0.98], "eps": 1e-6},
    "scheduler": "LambdaLR", "scheduler_tag": "constant_no_warmup",
}

_EXACT_ROUTE_SPECS[timesfm_lora.ROUTE_KEY] = {
    "module": timesfm_lora,
    "profile": timesfm_lora.PROFILE,
    "optimizer_factory": "parameters",
    "config_has_accumulation_field": False,
    "input": {
        "layout_tag": "independent_univariate_passes",
        "context_length": timesfm_lora.CONTEXT_LENGTH,
        "horizon_length": timesfm_lora.HORIZON_LENGTH,
        "parent_length": timesfm_lora.CONTEXT_LENGTH + timesfm_lora.HORIZON_LENGTH,
        "channel_order": list(timesfm_lora.CHANNELS),
        "grouping_tag": "no_cross_channel_interaction", "dtype": "fp32",
    },
    "time": {
        "horizon_unit": "bars", "timestamp_tag": "utc_bar_close", "timezone_tag": "utc",
    },
    "export": {"bundle_tag": "base_plus_adapter", "output_tag": "point_and_raw_quantiles"},
    "optimizer": {"name": "AdamW", "betas": [0.9, 0.999], "eps": 1e-8},
    "scheduler": "CosineAnnealingLR", "scheduler_tag": "cosine_no_warmup",
}

for _route_key, _route_spec in ttm_native.ROUTES.items():
    _EXACT_ROUTE_SPECS[_route_key] = {
        "module": ttm_native,
        "profile": _route_spec["profile"],
        "config_factory": lambda key=_route_key: ttm_native.RouteConfig(key),
        "optimizer_factory": "parameters",
        "config_has_accumulation_field": False,
        "input": {
            "layout_tag": "B_512_C", "context_length": ttm_native.CONTEXT_LENGTH,
            "horizon_length": ttm_native.HORIZON_LENGTH,
            "parent_length": ttm_native.CONTEXT_LENGTH + ttm_native.HORIZON_LENGTH,
            "channel_order": list(ttm_native.CHANNELS),
            "grouping_tag": "channel_independent_mixer_off", "dtype": "fp32",
        },
        "time": {
            "horizon_unit": "bars", "timestamp_tag": "utc_bar_close",
            "timezone_tag": "utc",
        },
        "export": {"bundle_tag": "forecast_bundle", "output_tag": "B_16_C"},
        "optimizer": {"name": "AdamW", "betas": [0.9, 0.999], "eps": 1e-8},
        "scheduler": "LambdaLR", "scheduler_tag": "linear_no_warmup",
    }

for _route_key, _route_spec in moment_tasks.ROUTES.items():
    _classification = _route_spec["task"] == "classification"
    _EXACT_ROUTE_SPECS[_route_key] = {
        "module": moment_tasks,
        "profile": _route_spec["profile"],
        "config_factory": lambda key=_route_key: moment_tasks.RouteConfig(key),
        "optimizer_factory": "parameters",
        "config_has_accumulation_field": False,
        "input": (
            {
                "layout_tag": "B_C_512_internal_fold_only",
                "context_length": moment_tasks.CONTEXT_LENGTH,
                "horizon_length": 0,
                "parent_length": moment_tasks.CONTEXT_LENGTH,
                "channel_order": list(moment_tasks.CHANNELS),
                "grouping_tag": "no_cross_channel_interaction", "dtype": "fp32",
            }
            if _classification else
            {
                "layout_tag": "independent_univariate_passes",
                "context_length": moment_tasks.CONTEXT_LENGTH,
                "horizon_length": moment_tasks.HORIZON_LENGTH,
                "parent_length": moment_tasks.CONTEXT_LENGTH + moment_tasks.HORIZON_LENGTH,
                "channel_order": list(moment_tasks.CHANNELS),
                "grouping_tag": "no_cross_channel_interaction", "dtype": "fp32",
            }
        ),
        "time": (
            {"horizon_unit": "none", "timestamp_tag": "utc_bar_close", "timezone_tag": "utc"}
            if _classification else
            {"horizon_unit": "bars", "timestamp_tag": "utc_bar_close", "timezone_tag": "utc"}
        ),
        "export": (
            {"bundle_tag": "classifier_bundle", "output_tag": "classification_logits"}
            if _classification else {"bundle_tag": "forecast_bundle", "output_tag": "B_C_16"}
        ),
        "optimizer": {"name": "Adam", "betas": [0.9, 0.999], "eps": 1e-8},
        "scheduler": "LambdaLR" if _classification else "OneCycleLR",
        "scheduler_tag": "constant_no_warmup" if _classification else "one_cycle_pct30",
        **({
            "scheduler_runtime": {
                "cycle_momentum": True, "use_beta1": True,
                "base_momentum": 0.85, "max_momentum": 0.95,
                "anneal_strategy": "cos",
            }
        } if not _classification else {}),
    }

for _route_key, _route_spec in mantis_native.ROUTES.items():
    _classification = _route_spec["task"] == "classification"
    _EXACT_ROUTE_SPECS[_route_key] = {
        "module": mantis_native,
        "profile": _route_spec["profile"],
        "config_factory": lambda key=_route_key: mantis_native.RouteConfig(key),
        "optimizer_factory": "parameters",
        "config_has_accumulation_field": False,
        "input": {
            "layout_tag": "independent_univariate_passes",
            "context_length": mantis_native.CONTEXT_LENGTH,
            "horizon_length": 0,
            "parent_length": mantis_native.CONTEXT_LENGTH,
            "channel_order": list(mantis_native.CHANNELS),
            "grouping_tag": "no_cross_channel_interaction",
            "dtype": "fp32",
        },
        "time": {
            "horizon_unit": "none", "timestamp_tag": "utc_bar_close",
            "timezone_tag": "utc",
        },
        "export": (
            {"bundle_tag": "classifier_bundle", "output_tag": "classification_logits"}
            if _classification else
            {
                "bundle_tag": "representation_bundle",
                "output_tag": (
                    "final_cls_per_channel" if _route_spec["version"] == 1
                    else "layer2_cls_mean_per_channel"
                ),
            }
        ),
        "optimizer": {"name": "AdamW", "betas": [0.9, 0.999], "eps": 1e-8},
        "scheduler": "LambdaLR",
        "scheduler_tag": "warmup_10_epochs_then_cosine_to_zero",
    }


def _resolved(profile: Mapping[str, Any], field: str) -> dict[str, Any]:
    item = profile[field]
    if item.get("state") != "resolved" or not isinstance(item.get("value"), Mapping):
        raise NativeContractError(f"exact route profile field is unresolved: {field}")
    return dict(item["value"])


def _compare(
    discrepancies: list[dict[str, Any]],
    *,
    route_key: str,
    field: str,
    actual: Any,
    expected: Any,
) -> None:
    if actual != expected:
        discrepancies.append({
            "route_key": route_key,
            "field": field,
            "actual": actual,
            "expected": expected,
        })


def _optimizer_record(
    module: Any, config: Any, *, optimizer_factory: str = "model",
) -> dict[str, Any]:
    import torch

    model = torch.nn.Linear(2, 2)
    if optimizer_factory == "parameters_cuda":
        if not torch.cuda.is_available():
            raise NativeContractError("exact CUDA optimizer audit requires a visible GPU")
        model = model.to("cuda:0")
    optimizer = (
        module.make_optimizer_for_parameters(model.parameters(), config)
        if optimizer_factory in {"parameters", "parameters_cuda"}
        else module.make_optimizer(model, config)
    )
    group = optimizer.param_groups[0]
    before = {
        "learning_rate": float(group["lr"]),
        "weight_decay": float(group["weight_decay"]),
        "betas": [float(value) for value in group["betas"]],
        "eps": float(group["eps"]),
    }
    scheduler = module.make_scheduler(optimizer, config)
    scheduler_state = scheduler.state_dict()
    return {
        "optimizer": type(optimizer).__name__,
        "learning_rate_before_scheduler": before["learning_rate"],
        "weight_decay": before["weight_decay"],
        "betas": before["betas"],
        "eps": before["eps"],
        "scheduler": type(scheduler).__name__,
        "scheduler_total_steps": (
            int(getattr(scheduler, "total_steps"))
            if hasattr(scheduler, "total_steps") else None
        ),
        "scheduler_t_max": (
            int(getattr(scheduler, "T_max")) if hasattr(scheduler, "T_max") else None
        ),
        "cycle_momentum": bool(scheduler_state.get("cycle_momentum", False)),
        "use_beta1": bool(scheduler_state.get("use_beta1", False)),
        "base_momentum": (
            float(optimizer.param_groups[0]["base_momentum"])
            if "base_momentum" in optimizer.param_groups[0] else None
        ),
        "max_momentum": (
            float(optimizer.param_groups[0]["max_momentum"])
            if "max_momentum" in optimizer.param_groups[0] else None
        ),
        "anneal_strategy": scheduler_state.get("_anneal_func_type"),
    }


def _audit_exact_route(
    route_key: str,
    spec: Mapping[str, Any],
    catalog: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    discrepancies: list[dict[str, Any]] = []
    route = catalog["routes"][route_key]
    _compare(
        discrepancies, route_key=route_key, field="constraint_profile",
        actual=route["constraint_profile"], expected=spec["profile"],
    )
    profile = catalog["constraint_profiles"][spec["profile"]]
    for field in ("input", "time", "export"):
        _compare(
            discrepancies, route_key=route_key, field=field,
            actual=_resolved(profile, field), expected=dict(spec[field]),
        )

    module = spec["module"]
    config = (
        spec["config_factory"]() if "config_factory" in spec else module.RouteConfig()
    )
    if hasattr(config, "resolved"):
        config_values = dict(config.resolved())
    else:
        config.validate()
        config_values = asdict(config)
    policy = _resolved(profile, "optimization_hyperparameters")
    default_map = {
        "learning_rate": "learning_rate_default",
        "weight_decay": "weight_decay_default",
        "batch_size": "batch_size_default",
        "max_gradient_norm": "max_gradient_norm_default",
        "total_steps": "smoke_steps",
    }
    if spec.get("config_has_accumulation_field", True):
        default_map["gradient_accumulation_steps"] = "gradient_accumulation_default"
    else:
        _compare(
            discrepancies, route_key=route_key,
            field="optimization_hyperparameters.gradient_accumulation_default",
            actual=policy["gradient_accumulation_default"], expected=1,
        )
    for config_field, policy_field in default_map.items():
        _compare(
            discrepancies, route_key=route_key,
            field=f"optimization_hyperparameters.{policy_field}",
            actual=config_values[config_field], expected=policy[policy_field],
        )
    _compare(
        discrepancies, route_key=route_key,
        field="optimization_hyperparameters.gradient_accumulation_max",
        actual=policy["gradient_accumulation_max"], expected=1,
    )
    _compare(
        discrepancies, route_key=route_key,
        field="optimization_hyperparameters.scheduler_tag",
        actual=policy["scheduler_tag"], expected=spec["scheduler_tag"],
    )
    if policy["scheduler_tag"] == "cosine_no_warmup":
        _compare(
            discrepancies, route_key=route_key,
            field="optimization_hyperparameters.warmup_fraction",
            actual=[
                policy["warmup_fraction_min"], policy["warmup_fraction_default"],
                policy["warmup_fraction_max"],
            ], expected=[0.0, 0.0, 0.0],
        )
    if spec.get("config_has_accumulation_field", True):
        invalid = dict(config_values)
        invalid["gradient_accumulation_steps"] = 2
        rejected = False
        try:
            module.RouteConfig(**invalid).validate()
        except ValueError:
            rejected = True
        _compare(
            discrepancies, route_key=route_key,
            field="RouteConfig.rejects_unimplemented_accumulation",
            actual=rejected, expected=True,
        )

    optimizer = _optimizer_record(
        module, config, optimizer_factory=spec.get("optimizer_factory", "model"),
    )
    expected_optimizer = spec["optimizer"]
    _compare(
        discrepancies, route_key=route_key, field="optimizer.name",
        actual=optimizer["optimizer"], expected=expected_optimizer["name"],
    )
    _compare(
        discrepancies, route_key=route_key, field="optimizer.learning_rate",
        actual=optimizer["learning_rate_before_scheduler"],
        expected=float(config_values["learning_rate"]),
    )
    _compare(
        discrepancies, route_key=route_key, field="optimizer.weight_decay",
        actual=optimizer["weight_decay"], expected=float(config_values["weight_decay"]),
    )
    _compare(
        discrepancies, route_key=route_key, field="optimizer.betas",
        actual=optimizer["betas"], expected=expected_optimizer["betas"],
    )
    _compare(
        discrepancies, route_key=route_key, field="optimizer.eps",
        actual=optimizer["eps"], expected=expected_optimizer["eps"],
    )
    _compare(
        discrepancies, route_key=route_key, field="scheduler.name",
        actual=optimizer["scheduler"], expected=spec["scheduler"],
    )
    if spec["scheduler"] == "CosineAnnealingLR":
        _compare(
            discrepancies, route_key=route_key, field="scheduler.T_max",
            actual=optimizer["scheduler_t_max"], expected=int(config_values["total_steps"]),
        )
    elif spec["scheduler"] == "OneCycleLR":
        _compare(
            discrepancies, route_key=route_key, field="scheduler.total_steps",
            actual=optimizer["scheduler_total_steps"],
            expected=int(config_values["total_steps"]),
        )
        for field, expected in spec.get("scheduler_runtime", {}).items():
            _compare(
                discrepancies, route_key=route_key,
                field=f"scheduler.{field}", actual=optimizer[field], expected=expected,
            )

    return {
        "route_key": route_key,
        "profile": spec["profile"],
        "config": config_values,
        "optimizer_runtime": optimizer,
        "configuration_consistent": not discrepancies,
    }, discrepancies


def load_current_parity_aggregate(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NativeContractError(f"cannot read native parity aggregate: {source}") from exc
    if not isinstance(value, Mapping):
        raise NativeContractError("native parity aggregate must be an object")
    payload = deepcopy(dict(value))
    supplied = payload.pop("aggregate_sha256", None)
    if supplied != content_sha256(payload):
        raise NativeContractError("native parity aggregate integrity mismatch")
    registry = load_registry()
    if (
        value.get("registry_sha256") != registry_sha256()
        or value.get("methodology_commit") != registry["methodology_commit"]
        or value.get("require_all_current") is not True
    ):
        raise NativeContractError("native parity aggregate is stale for the current registry")
    candidate = value.get("candidate_evidence")
    if not isinstance(candidate, Mapping):
        raise NativeContractError("native parity aggregate lacks candidate evidence")
    if value.get("candidate_evidence_sha256") != content_sha256(candidate):
        raise NativeContractError("native parity candidate evidence integrity mismatch")
    _validate_evidence(candidate, registry)
    required = {
        (arm_key, track)
        for arm_key, dossier in registry["models"].items()
        for track, capability in dossier["tracks"].items()
        if capability["status"] in ADMITTED_STATUSES
    }
    records = value.get("bundles")
    if not isinstance(records, list):
        raise NativeContractError("native parity aggregate bundle index is malformed")
    actual = {(row["arm_key"], row["track"]) for row in records}
    if actual != required:
        raise NativeContractError("native parity aggregate does not cover current admitted tracks")
    for row in records:
        bundle = source.parent / f"{row['arm_key']}__{row['track']}"
        manifest, _ = verify_parity_bundle(bundle)
        if manifest["bundle_sha256"] != row["bundle_sha256"]:
            raise NativeContractError("native parity aggregate bundle hash mismatch")
    return deepcopy(dict(value))


def build_native_configuration_audit(
    *, parity_aggregate_path: str | Path | None = None,
) -> dict[str, Any]:
    registry = load_registry()
    catalog = load_family_route_catalog()
    discrepancies: list[dict[str, Any]] = []
    model_rows = []
    admitted_pairs = 0
    blocked_models = 0
    for arm_key, dossier in sorted(registry["models"].items()):
        statuses = {track: capability["status"] for track, capability in dossier["tracks"].items()}
        admitted = sorted(
            track for track, status in statuses.items() if status in ADMITTED_STATUSES
        )
        admitted_pairs += len(admitted)
        blocked_models += int(not admitted)
        pin_complete = bool(dossier["pin_complete"])
        if pin_complete:
            for field in ("model_revision", "source_revision"):
                if not re.fullmatch(r"[0-9a-f]{40}|[^=]+==[^=]+", str(dossier[field])):
                    discrepancies.append({
                        "arm_key": arm_key, "field": field,
                        "actual": dossier[field], "expected": "exact revision pin",
                    })
        model_rows.append({
            "arm_key": arm_key,
            "model_id": dossier["model_id"],
            "model_revision": dossier["model_revision"],
            "source_revision": dossier["source_revision"],
            "tokenizer": deepcopy(dossier.get("tokenizer")),
            "license": deepcopy(dossier["license"]),
            "pin_complete": pin_complete,
            "overall_status": dossier["overall_status"],
            "track_statuses": statuses,
            "admitted_native_tracks": admitted,
        })

    unresolved_constraints = []
    for profile_id, profile in sorted(catalog["constraint_profiles"].items()):
        for field, constraint in sorted(profile.items()):
            if constraint["state"] != "resolved":
                unresolved_constraints.append({
                    "profile": profile_id, "field": field,
                    "state": constraint["state"],
                })
                discrepancies.append({
                    "profile": profile_id,
                    "field": field,
                    "actual": constraint["state"],
                    "expected": "resolved",
                })

    exact_rows = []
    for route_key, spec in sorted(_EXACT_ROUTE_SPECS.items()):
        row, route_discrepancies = _audit_exact_route(route_key, spec, catalog)
        exact_rows.append(row)
        discrepancies.extend(route_discrepancies)

    non_exact = sorted(set(catalog["routes"]) - set(_EXACT_ROUTE_SPECS))
    closed_exclusions = 0
    externally_blocked_routes = 0
    external_tags = {
        "terms_unaccepted", "checkpoint_unavailable", "checkpoint_hash_unavailable",
        "model_identity_unresolved", "native_output_parity_missing",
    }
    for route_key in non_exact:
        route = catalog["routes"][route_key]
        profile = catalog["constraint_profiles"][route["constraint_profile"]]
        if not route["blocker_tags"] or route["status"] != "blocked":
            discrepancies.append({
                "route_key": route_key,
                "field": "fail_closed_status",
                "actual": {"status": route["status"], "blocker_tags": route["blocker_tags"]},
                "expected": "blocked route with explicit blocker tags",
            })
        if route["pathway_kind"] == "unsupported":
            closed_exclusions += 1
            expected = {
                "method_provenance": "unsupported",
                "task_kind": "unsupported",
                "permitted_use_scopes": [],
            }
            actual = {field: route[field] for field in expected}
            if actual != expected or any(
                constraint["state"] != "resolved" for constraint in profile.values()
            ):
                discrepancies.append({
                    "route_key": route_key,
                    "field": "closed_exclusion",
                    "actual": actual,
                    "expected": expected,
                })
        if set(route["blocker_tags"]) & external_tags:
            externally_blocked_routes += 1

    parity = None
    if parity_aggregate_path is not None:
        aggregate = load_current_parity_aggregate(parity_aggregate_path)
        parity = {
            "path": str(Path(parity_aggregate_path).expanduser().resolve()),
            "aggregate_sha256": aggregate["aggregate_sha256"],
            "tracks_verified": len(aggregate["bundles"]),
            "current_registry_complete": True,
        }

    document = {
        "schema_version": CONFIGURATION_AUDIT_SCHEMA,
        "policy": CONFIGURATION_AUDIT_POLICY,
        "registry_sha256": registry_sha256(),
        "catalog_sha256": catalog_sha256(catalog),
        "counts": {
            "models": len(registry["models"]),
            "catalog_routes": len(catalog["routes"]),
            "constraint_profiles": len(catalog["constraint_profiles"]),
            "admitted_native_inference_tracks": admitted_pairs,
            "models_without_admitted_native_track": blocked_models,
            "exact_training_executors": len(_EXACT_ROUTE_SPECS),
            "non_exact_or_blocked_routes": len(non_exact),
            "configuration_discrepancies": len(discrepancies),
            "unresolved_constraints": len(unresolved_constraints),
            "closed_unsupported_routes": closed_exclusions,
            "externally_blocked_routes": externally_blocked_routes,
        },
        "models": model_rows,
        "exact_routes": exact_rows,
        "parity_evidence": parity,
        "configuration_integrity_passed": not discrepancies,
        "configuration_contracts_complete": not unresolved_constraints,
        "all_routes_dispositioned": (
            not discrepancies
            and not unresolved_constraints
            and closed_exclusions + len(_EXACT_ROUTE_SPECS) <= len(catalog["routes"])
        ),
        "current_inference_parity_complete": parity is not None,
        "all_models_execution_ready": False,
        "all_training_routes_execution_ready": False,
        "training_admitted": False,
        "live_trading_ready": False,
        "discrepancies": discrepancies,
    }
    document["audit_sha256"] = content_sha256(document)
    return document


def validate_native_configuration_audit(
    value: Any,
    *, parity_aggregate_path: str | Path | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise NativeContractError("native configuration audit must be an object")
    candidate = deepcopy(dict(value))
    supplied = candidate.pop("audit_sha256", None)
    if supplied != content_sha256(candidate):
        raise NativeContractError("native configuration audit integrity mismatch")
    expected = build_native_configuration_audit(
        parity_aggregate_path=parity_aggregate_path,
    )
    if dict(value) != expected:
        raise NativeContractError("native configuration audit is stale or non-canonical")
    return deepcopy(expected)


__all__ = [
    "CONFIGURATION_AUDIT_POLICY", "CONFIGURATION_AUDIT_SCHEMA",
    "build_native_configuration_audit", "load_current_parity_aggregate",
    "validate_native_configuration_audit",
]
