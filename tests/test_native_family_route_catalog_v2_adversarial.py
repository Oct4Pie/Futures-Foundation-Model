"""Adversarial semantic-integrity tests for the native family route catalog.

These tests deliberately preserve the catalog's generic shape while corrupting one
audited fact.  A schema-only validator is expected to reject every mutation: a
coherently re-keyed or re-hashed lie is still a lie.

The cases below are representative invariants, not a duplicate catalog inventory.
The canonical route specification remains the single source of truth.
"""
from __future__ import annotations

from copy import deepcopy

import pytest

from futures_foundation.finetune.native_contracts import NativeContractError, load_registry
from futures_foundation.finetune.native_family_route_catalog_v2 import (
    CATALOG,
    validate_family_route_catalog,
)


def _must_reject(catalog: dict) -> None:
    with pytest.raises(NativeContractError):
        validate_family_route_catalog(catalog)


def test_rejects_coherent_arm_identity_swap() -> None:
    catalog = deepcopy(CATALOG)
    catalog["arms"]["mantis_v1"], catalog["arms"]["mantis_v2"] = (
        catalog["arms"]["mantis_v2"],
        catalog["arms"]["mantis_v1"],
    )
    _must_reject(catalog)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("content_sha256", "0" * 64),
        ("arm_key", "chronos_v1"),
        ("registry_schema", "invented_registry_v1"),
    ],
)
def test_rejects_rewritten_dossier_reference(field: str, replacement: str) -> None:
    catalog = deepcopy(CATALOG)
    catalog["arms"]["chronos_bolt"]["dossier_ref"][field] = replacement
    _must_reject(catalog)


def test_rejects_authoritative_dossier_tampering_without_catalog_rebinding() -> None:
    registry = deepcopy(load_registry())
    registry["models"]["chronos_bolt"]["model_revision"] = "0" * 40
    with pytest.raises(NativeContractError, match="dossier hash mismatch"):
        validate_family_route_catalog(CATALOG, registry=registry)


def test_rejects_cross_version_profile_swap_even_when_profile_closure_is_preserved() -> None:
    catalog = deepcopy(CATALOG)
    v1_key = "mantis_v1:R:official_crop_resize_contrastive"
    v2_key = "mantis_v2:R:official_crop_resize_contrastive"
    catalog["routes"][v1_key]["constraint_profile"], catalog["routes"][v2_key][
        "constraint_profile"
    ] = (
        catalog["routes"][v2_key]["constraint_profile"],
        catalog["routes"][v1_key]["constraint_profile"],
    )
    _must_reject(catalog)


def test_rejects_cross_family_profile_swap_even_when_profile_closure_is_preserved() -> None:
    catalog = deepcopy(CATALOG)
    t5_key = "chronos_v1:F:native_64_t5_token_forecast_cross_entropy"
    bolt_key = "chronos_bolt:F:direct_native_quantile_pinball"
    catalog["routes"][t5_key]["constraint_profile"], catalog["routes"][bolt_key][
        "constraint_profile"
    ] = (
        catalog["routes"][bolt_key]["constraint_profile"],
        catalog["routes"][t5_key]["constraint_profile"],
    )
    _must_reject(catalog)


def test_rejects_valid_cross_route_task_kind_swap() -> None:
    """Both values are valid Track-F kinds; exact route identity must still reject them."""
    catalog = deepcopy(CATALOG)
    token_key = "chronos_v1:F:native_64_t5_token_forecast_cross_entropy"
    quantile_key = "chronos_bolt:F:direct_native_quantile_pinball"
    catalog["routes"][token_key]["task_kind"], catalog["routes"][quantile_key][
        "task_kind"
    ] = (
        catalog["routes"][quantile_key]["task_kind"],
        catalog["routes"][token_key]["task_kind"],
    )
    _must_reject(catalog)


def test_rejects_invented_task_kind() -> None:
    catalog = deepcopy(CATALOG)
    key = "chronos_bolt:F:direct_native_quantile_pinball"
    catalog["routes"][key]["task_kind"] = "generic_forecast"
    _must_reject(catalog)


def test_rejects_supported_and_unsupported_route_disposition_swap() -> None:
    catalog = deepcopy(CATALOG)
    supported_key = "chronos_bolt:F:direct_native_quantile_pinball"
    unsupported_key = "toto2_22m:F:no_released_toto2_finetuning"
    semantic_fields = (
        "method_provenance",
        "pathway_kind",
        "permitted_use_scopes",
        "constraint_profile",
        "blocker_tags",
    )
    for field in semantic_fields:
        catalog["routes"][supported_key][field], catalog["routes"][unsupported_key][field] = (
            catalog["routes"][unsupported_key][field],
            catalog["routes"][supported_key][field],
        )
    _must_reject(catalog)


def test_rejects_extra_well_formed_fake_route() -> None:
    catalog = deepcopy(CATALOG)
    source_key = "chronos_bolt:F:direct_native_quantile_pinball"
    route = deepcopy(catalog["routes"][source_key])
    route["route_id"] = "plausible_but_unaudited_training_route"
    fake_key = f"{route['arm_key']}:{route['track']}:{route['route_id']}"
    catalog["routes"][fake_key] = route
    _must_reject(catalog)


def test_rejects_coherently_rekeyed_track_change() -> None:
    catalog = deepcopy(CATALOG)
    old_key = "chronos_bolt:F:direct_native_quantile_pinball"
    route = catalog["routes"].pop(old_key)
    route["track"] = "C"
    new_key = f"{route['arm_key']}:{route['track']}:{route['route_id']}"
    catalog["routes"][new_key] = route
    _must_reject(catalog)


def test_rejects_structured_but_fake_objective() -> None:
    catalog = deepcopy(CATALOG)
    catalog["constraint_profiles"]["chronos_bolt_forecast"]["objective"]["value"][
        "loss_tag"
    ] = "invented_native_loss"
    _must_reject(catalog)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("method_provenance", "upstream_native"),
        ("pathway_kind", "in_context_fit"),
    ],
)
def test_rejects_plausible_but_false_method_or_pathway(
    field: str, replacement: str
) -> None:
    catalog = deepcopy(CATALOG)
    key = "chronos_bolt:F:direct_native_quantile_pinball"
    catalog["routes"][key][field] = replacement
    _must_reject(catalog)


def test_rejects_arbitrary_unresolved_field_name() -> None:
    catalog = deepcopy(CATALOG)
    unresolved = catalog["constraint_profiles"]["mantis_v2_contrastive"]["objective"][
        "value"
    ]["unresolved_fields"]
    unresolved.append("trust_me_later")
    _must_reject(catalog)


def test_rejects_known_but_semantically_irrelevant_blocker() -> None:
    catalog = deepcopy(CATALOG)
    key = "chronos_bolt:F:direct_native_quantile_pinball"
    catalog["routes"][key]["blocker_tags"] = ["checkpoint_unavailable"]
    _must_reject(catalog)


def test_rejects_kronos_tokenizer_identity_swap_between_model_arms() -> None:
    registry = deepcopy(load_registry())
    mini = registry["models"]["kronos_mini"]
    small = registry["models"]["kronos_small"]
    mini["tokenizer"], small["tokenizer"] = small["tokenizer"], mini["tokenizer"]
    with pytest.raises(NativeContractError, match="dossier hash mismatch"):
        validate_family_route_catalog(CATALOG, registry=registry)


def test_rejects_kronos_predictor_cross_arm_parent_lineage() -> None:
    catalog = deepcopy(CATALOG)
    lineage = catalog["constraint_profiles"]["kronos_mini_predictor"]["lineage"]["value"]
    lineage["parent_artifacts"] = ["kronos_small_tokenizer_base_bundle"]
    _must_reject(catalog)
