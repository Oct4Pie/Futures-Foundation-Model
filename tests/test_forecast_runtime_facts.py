import copy

import pytest

from futures_foundation.finetune.native_contracts import (
    NativeContractError,
    validate_runtime_contract,
)
from scripts.predict_foundation_forecasts import (
    QUANTILE_LEVELS,
    TIMEFRAMES_MINUTES,
    TTM_FREQUENCY_TOKENS,
    _forecast_runtime_facts,
)


BASE = dict(context_length=512, prediction_length=16, dtype="float32", samples=20)
GREEDY = {"temperature": 1.0, "top_k": 1, "top_p": 1.0, "sample_count": 1}


@pytest.mark.parametrize("arm", [
    "kronos_mini", "kronos_small", "chronos_v1", "chronos_bolt", "chronos_v2",
    "timesfm25", "ttm_r2", "moirai2_small", "toto2_22m", "sundial_base",
])
def test_forecast_runtime_facts_exactly_validate_current_contract(arm):
    facts = _forecast_runtime_facts(
        arm, **BASE,
        kronos_decoding=GREEDY if arm.startswith("kronos") else None,
        ttm_frequency_tokens=TTM_FREQUENCY_TOKENS if arm == "ttm_r2" else None,
    )
    assert validate_runtime_contract(arm, "F", facts) == facts


def test_family_specific_runtime_facts_cover_every_operational_switch():
    kronos = _forecast_runtime_facts("kronos_mini", **BASE, kronos_decoding=GREEDY)
    assert kronos["decoding"] == [GREEDY]
    assert kronos["timeframes_minutes"] == TIMEFRAMES_MINUTES
    assert kronos["timestamp_timezone"] == "UTC"
    assert _forecast_runtime_facts("chronos_v1", **BASE)["quantile_levels"] == QUANTILE_LEVELS
    assert _forecast_runtime_facts("chronos_v2", **BASE)["cross_learning"] is False
    timesfm = _forecast_runtime_facts("timesfm25", **BASE)
    assert timesfm["force_flip_invariance"] is True
    assert timesfm["truncate_negative"] is False
    assert timesfm["fix_quantile_crossing"] is False
    ttm = _forecast_runtime_facts(
        "ttm_r2", **BASE, ttm_frequency_tokens=TTM_FREQUENCY_TOKENS
    )
    assert ttm["frequency_tokens_by_timeframe"] == TTM_FREQUENCY_TOKENS
    assert ttm["selector"] == "512-48-ft-r2.1"
    toto = _forecast_runtime_facts("toto2_22m", **BASE)
    assert toto["decode_block_size"] is None
    assert toto["masked_value_fill"] == 0.0
    assert toto["series_ids"] == "one_semantic_group_per_item"
    assert toto["has_missing_values"] is False
    sundial = _forecast_runtime_facts("sundial_base", **BASE)
    assert sundial["num_samples"] == 20
    assert sundial["isolated_environment"] is True
    assert sundial["hidden_states"] == "forbidden"


@pytest.mark.parametrize("arm,field,bad", [
    ("timesfm25", "force_flip_invariance", False),
    ("chronos_v1", "quantile_levels", [0.5]),
    ("chronos_v2", "cross_learning", True),
    ("ttm_r2", "selector", "wrong"),
    ("toto2_22m", "decode_block_size", 16),
    ("sundial_base", "hidden_states", "enabled"),
    ("kronos_mini", "timestamp_timezone", "naive"),
])
def test_runtime_validation_rejects_one_field_drift(arm, field, bad):
    facts = _forecast_runtime_facts(
        arm, **BASE,
        kronos_decoding=GREEDY if arm.startswith("kronos") else None,
        ttm_frequency_tokens=TTM_FREQUENCY_TOKENS if arm == "ttm_r2" else None,
    )
    changed = copy.deepcopy(facts)
    changed[field] = bad
    with pytest.raises(NativeContractError, match="runtime contract mismatch"):
        validate_runtime_contract(arm, "F", changed)


def test_ttm_fact_builder_rejects_partial_or_wrong_timeframe_mapping():
    with pytest.raises(ValueError, match="frequency mapping drifted"):
        _forecast_runtime_facts(
            "ttm_r2", **BASE,
            ttm_frequency_tokens={"1min": 1, "60min": 7},
        )
