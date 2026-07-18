from copy import deepcopy
from pathlib import Path

import pytest

from futures_foundation.finetune import native_contracts, native_training_routes
from futures_foundation.finetune.native_contracts import NativeContractError
from futures_foundation.finetune.native_family_route_catalog_v2 import (
    CATALOG,
    catalog_sha256,
)


def test_compatibility_registry_is_catalog_derived_and_non_authorizing():
    registry = native_training_routes.load_route_registry()
    assert registry == CATALOG
    assert registry is not CATALOG
    assert registry["non_authorizing"] is True
    assert len(registry["arms"]) == 15
    assert {route["status"] for route in registry["routes"].values()} == {"blocked"}
    assert native_training_routes.registry_sha256() == catalog_sha256()


def test_display_registry_is_a_defensive_copy():
    display = native_training_routes.load_route_registry()
    display["routes"].clear()
    assert native_training_routes.load_route_registry() == CATALOG


def test_display_route_uses_catalog_identity_and_cannot_claim_admission():
    route = native_training_routes.get_route(
        "kronos_small", "F", "hierarchical_autoregressive_tokens"
    )
    assert route == CATALOG["routes"][
        "kronos_small:F:hierarchical_autoregressive_tokens"
    ]
    assert route["status"] == "blocked"
    assert route["evidence_id"] is None
    assert native_training_routes.admitted_routes_for_arm("kronos_small") == ()
    assert native_training_routes.admitted_routes_for_arm(
        "moirai2_small", include_research=True
    ) == ()


def test_unknown_display_route_is_rejected():
    with pytest.raises(NativeContractError, match="undeclared training route"):
        native_training_routes.get_route("kronos_small", "F", "invented")


@pytest.mark.parametrize(
    "call",
    [
        lambda path: native_training_routes.authorize_route(
            arm_key="kronos_small",
            track="F",
            route_id="hierarchical_autoregressive_tokens",
            use_scope="production",
            path=path,
        ),
        lambda path: native_training_routes.authorize_route(
            arm_key="invented",
            track="invented",
            route_id=None,
            use_scope=None,
            path=path,
        ),
        lambda path: native_training_routes.evidence_for_route(
            "kronos_small", "F", "hierarchical_autoregressive_tokens", path
        ),
    ],
)
def test_authorization_and_evidence_fail_before_any_path_access(tmp_path, call):
    hostile = tmp_path / "must-not-be-read"
    hostile.mkdir()
    (hostile / "native_training_routes.json").write_text("not json", encoding="utf-8")
    with pytest.raises(NativeContractError, match="v1 route/evidence authority was retired"):
        call(hostile / "native_contracts.json")


def test_optimizer_kill_switch_remains_unconditional():
    with pytest.raises(NativeContractError, match="optimizer entrypoint 'fixture' is disabled"):
        native_training_routes.block_unadmitted_optimizer("fixture")


def test_caller_path_cannot_select_an_alternate_route_authority(tmp_path):
    fake_catalog = deepcopy(CATALOG)
    fake_catalog["non_authorizing"] = False
    (tmp_path / "native_training_routes.json").write_text(
        repr(fake_catalog), encoding="utf-8"
    )
    assert native_training_routes.load_route_registry(
        tmp_path / "native_contracts.json"
    ) == CATALOG
    assert native_training_routes.route_registry_path(
        tmp_path / "native_contracts.json"
    ).name == "native_family_route_catalog_v2.json"


def test_native_contract_arms_cannot_inherit_catalog_inventory_as_training_authority():
    assert not any(arm.supported_training for arm in native_contracts.all_arms().values())
    assert not any(arm.training_admitted for arm in native_contracts.all_arms().values())


def test_retired_evidence_path_has_no_compatibility_file():
    for call in (
        lambda: native_training_routes.route_evidence_path(Path("unused")),
        lambda: native_training_routes.load_route_evidence(Path("unused")),
        lambda: native_training_routes.evidence_sha256(Path("unused")),
    ):
        with pytest.raises(
            NativeContractError, match="v1 route/evidence authority was retired"
        ):
            call()
