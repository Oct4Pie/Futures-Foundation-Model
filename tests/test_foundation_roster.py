import pytest

from futures_foundation.finetune.foundation_roster import ARMS, get_arm


EXPECTED_ARMS = {
    "mantis_v1", "mantis_v2", "moment_small", "kronos_mini", "kronos_small",
    "chronos_v1", "chronos_bolt", "chronos_v2", "timesfm25", "ttm_r2",
    "moirai2_small", "toto2_22m", "sundial_base", "tabpfn_ts3_forecast",
    "tabpfn_v3_downstream",
}


def test_roster_is_registry_backed_and_separates_native_from_training_admission():
    assert set(ARMS) == EXPECTED_ARMS
    assert sum(arm.overall_status == "native_valid" for arm in ARMS.values()) == 12
    assert get_arm("moirai2_small").overall_status == "research_only"
    assert get_arm("tabpfn_ts3_forecast").overall_status == "blocked"
    assert get_arm("tabpfn_v3_downstream").overall_status == "blocked"
    assert not any(arm.training_admitted for arm in ARMS.values())
    assert get_arm("moment_small").ohlcv_mode == "channel_independent_ohlcv"
    assert not get_arm("ttm_r2").supported_training
    assert get_arm("timesfm25").adaptation == "none"
    assert not get_arm("timesfm25").supported_training
    assert get_arm("moirai2_small").license == "CC-BY-NC-4.0"
    assert not get_arm("toto2_22m").supported_training
    assert not get_arm("sundial_base").supported_training
    assert get_arm("tabpfn_ts3_forecast").ohlcv_mode == "support_query_timeseries_rows"
    assert get_arm("tabpfn_v3_downstream").ohlcv_mode == "tabular_support_query_rows"
    with pytest.raises(ValueError, match="unknown foundation arm"):
        get_arm("tabpfn_ts")


def test_kronos_tokenizer_pairing_is_owned_by_each_arm():
    mini = get_arm("kronos_mini")
    small = get_arm("kronos_small")
    assert mini.tokenizer_id == "NeoQuasar/Kronos-Tokenizer-2k"
    assert mini.tokenizer_revision == "26966d0035065a0cae0ebad7af8ece35bc1fb51c"
    assert small.tokenizer_id == "NeoQuasar/Kronos-Tokenizer-base"
    assert small.tokenizer_revision == "0e0117387f39004a9016484a186a908917e22426"


def test_unknown_roster_arm_fails_closed():
    with pytest.raises(ValueError, match="unknown foundation arm"):
        get_arm("made_up")
