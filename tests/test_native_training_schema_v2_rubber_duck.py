"""Independent adversarial checks for route-lineage ambiguity."""

import pytest

from futures_foundation.finetune import native_training_schema_v2 as schema
from futures_foundation.finetune.native_contracts import NativeContractError


def test_lineage_rejects_multiple_artifacts_for_one_child_input_slot():
    """One input slot must have exactly one producer/artifact binding."""
    lineage = {
        "initialization_tag": "parent_route_artifact",
        "parent_bindings": [
            {
                "route_key": "mantis_v1:R:official_crop_resize_contrastive",
                "template_sha256": "1" * 64,
                "artifact_tag": "representation_bundle",
                "child_input_slot": "parent_model",
            },
            {
                "route_key": "mantis_v1:R:official_crop_resize_contrastive",
                "template_sha256": "1" * 64,
                "artifact_tag": "full_training_state_bundle",
                "child_input_slot": "parent_model",
            },
        ],
        "forbidden_parent_route_keys": [],
    }

    with pytest.raises(NativeContractError, match="child input slot"):
        schema._validate_lineage(lineage, pathway="optimizer_training")
