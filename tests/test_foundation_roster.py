import pytest

from futures_foundation.finetune.foundation_roster import ARMS, get_arm


def test_roster_separates_supported_training_from_controls():
    assert get_arm("moment_small").ohlcv_mode == "channel_independent_ohlcv"
    assert get_arm("ttm_r2").supported_training
    assert get_arm("timesfm25").adaptation == "lora_native"
    assert get_arm("moirai2_small").license == "CC-BY-NC-4.0"
    assert not get_arm("toto2_22m").supported_training
    assert not get_arm("sundial_base").supported_training
    assert get_arm("tabpfn_ts").ohlcv_mode == "decomposed_univariate"
    assert len(ARMS) == len(set(ARMS))


def test_unknown_roster_arm_fails_closed():
    with pytest.raises(ValueError, match="unknown foundation arm"):
        get_arm("made_up")
