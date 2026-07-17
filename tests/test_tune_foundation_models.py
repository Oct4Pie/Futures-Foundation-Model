from scripts.tune_foundation_models import (
    _guard_validation_fingerprint,
    _mantis_validation_values,
    _selected_trial,
    _validation_fingerprint,
    build_command,
)


def test_mantis_objective_uses_paired_deltas_not_raw_scores():
    report = {"probe": {"per_target": {
        "fwd_absmove": {"ssl": 0.9, "vanilla": 0.8, "delta": 0.1},
        "fwd_dir": {"ssl": 0.4, "vanilla": 0.6, "delta": -0.2},
    }}}
    assert _mantis_validation_values(report) == (0.1, -0.2)


class _Study:
    def __init__(self):
        self.user_attrs = {}

    def set_user_attr(self, key, value):
        self.user_attrs[key] = value


class _Trial(_Study):
    pass


def test_tournament_commands_never_receive_oos_end_or_evaluation_flag(tmp_path):
    mantis = {
        "learning_rate": 1e-4, "weight_decay": .05, "context": 128,
        "preprocessing": "per_window_per_channel_zscore_v1", "temperature": .1,
        "crop_max": .2, "aug_noise": .1, "aug_scale": .2, "aug_tmask": .15,
    }
    moment = {"learning_rate": 1e-5, "weight_decay": .05, "context": 256,
              "mask_ratio": .3}
    kronos = {"tokenizer_steps": 32, "tokenizer_learning_rate": 2e-5,
              "predictor_learning_rate": 1e-6, "weight_decay": .05, "clip": 5.0}
    chronos = {"learning_rate": 1e-5, "weight_decay": .05, "grad_clip": 1.0}
    ttm = {"learning_rate": 1e-4, "weight_decay": .05, "grad_clip": 1.0}
    timesfm = {**ttm, "lora_rank": 4, "lora_alpha": 8, "lora_dropout": .05}
    moirai = {"learning_rate": 1e-4, "weight_decay": .05, "grad_clip": 1.0}
    commands = [
        build_command("mantis_v2", mantis, data_dir=tmp_path, output=tmp_path / "m.pt",
                      trial_steps=128, seed=1),
        build_command("moment", moment, data_dir=tmp_path, output=tmp_path / "o.pt",
                      trial_steps=128, seed=1, moment_repo="/moment"),
        build_command("kronos", kronos, data_dir=tmp_path, output=tmp_path / "k.pt",
                      trial_steps=128, seed=1, kronos_repo="/kronos"),
        build_command("kronos_mini", kronos, data_dir=tmp_path,
                      output=tmp_path / "km.pt", trial_steps=128, seed=1,
                      kronos_repo="/kronos"),
        build_command("chronos_v1", chronos, data_dir=tmp_path,
                      output=tmp_path / "c1.pt", trial_steps=128, seed=1),
        build_command("chronos_bolt", chronos, data_dir=tmp_path,
                      output=tmp_path / "cb.pt", trial_steps=128, seed=1),
        build_command("chronos_v2", chronos, data_dir=tmp_path,
                      output=tmp_path / "c2.pt", trial_steps=128, seed=1),
        build_command("ttm_r2", ttm, data_dir=tmp_path,
                      output=tmp_path / "t.pt", trial_steps=128, seed=1,
                      ttm_repo="/ttm", ttm_python="/ttm/python"),
        build_command("timesfm25", timesfm, data_dir=tmp_path,
                      output=tmp_path / "tf.pt", trial_steps=128, seed=1,
                      timesfm_repo="/timesfm"),
        build_command("moirai2_small", moirai, data_dir=tmp_path,
                      output=tmp_path / "mo.pt", trial_steps=128, seed=1,
                      uni2ts_repo="/uni2ts", moirai_python="/moirai/python"),
    ]
    for command in commands:
        rendered = " ".join(command)
        assert "2026-07-01" not in rendered
        assert "--oos" not in rendered and "--test" not in rendered
    assert "--train-start 2019-07-01" in " ".join(commands[0])
    assert "--holdout-start 2025-07-01" in " ".join(commands[0])
    assert "--chronos2-mode joint_ohlcv" in " ".join(commands[-4])
    assert "--univariate-input channel_independent_ohlcv" in " ".join(commands[-5])
    assert "--arm kronos_mini" in " ".join(commands[3])
    assert commands[-3][0] == "/ttm/python"
    assert "--timesfm-repo /timesfm" in " ".join(commands[-2])
    assert commands[-1][0] == "/moirai/python"


def test_mantis_stage1_tuning_command_uses_span_masking(tmp_path):
    params = {
        "learning_rate": 5e-5, "weight_decay": .05, "context": 256,
        "preprocessing": "per_window_shared_ohlc_zscore_v1", "mask_ratio": .4,
        "span_mean": 16.0, "span_max": 64, "feature_anchor_weight": .05,
    }
    command = build_command(
        "mantis_v2", params, data_dir=tmp_path, output=tmp_path / "stage1.pt",
        trial_steps=2048, seed=1, mantis_stage="mask",
    )
    rendered = " ".join(command)
    assert "--stage mask" in rendered
    assert "--lineage vanilla" in rendered
    assert "--seq 256" in rendered
    assert "--span-mean 16.0" in rendered
    assert "--span-max 64" in rendered
    assert "--feature-anchor-weight 0.05" in rendered
    assert "--temperature" not in rendered


def test_mantis_stage3_tuning_requires_and_uses_promoted_parent(tmp_path):
    import pytest
    params = {
        "learning_rate": 2e-5, "weight_decay": .05, "context": 128,
        "preprocessing": "per_window_per_channel_zscore_v1",
        "context_lengths": "64,128", "objective": "candle_direction",
        "dir_weight": .2, "freeze_encoder_layers": 3,
    }
    kwargs = dict(data_dir=tmp_path, output=tmp_path / "stage3.pt",
                  trial_steps=2048, seed=1, mantis_stage="forecast")
    with pytest.raises(ValueError, match="promoted contrastive"):
        build_command("mantis_v2", params, **kwargs)
    command = build_command(
        "mantis_v2", params, warm_checkpoint=tmp_path / "stage2.pt", **kwargs)
    rendered = " ".join(command)
    assert "--stage forecast --lineage canonical" in rendered
    assert f"--warm-checkpoint {tmp_path / 'stage2.pt'}" in rendered
    assert "--objective candle_direction" in rendered
    assert "--dir-weight 0.2" in rendered
    assert "--freeze-encoder-layers 3" in rendered


def test_mantis_stage3_suggestion_keeps_context_universe_fixed():
    from scripts.tune_foundation_models import _suggest

    class Trial:
        def suggest_categorical(self, name, choices):
            assert name != "context_lengths"
            return choices[0]

        def suggest_float(self, name, low, high, **kwargs):
            return low

    params = _suggest(Trial(), "mantis_v2", 2048, "forecast")
    assert params["context_lengths"] == "64,128,192"


def test_validation_fingerprint_is_model_specific_and_guarded():
    mantis = {"sampling": {"probe_sample_sha256": "same"}}
    native = {"sampling": {"validation_schedule_sha256": "same"}}
    assert _validation_fingerprint("mantis_v2", mantis) == "same"
    assert _validation_fingerprint("moment", native) == "same"

    study, trial = _Study(), _Trial()
    _guard_validation_fingerprint(study, trial, "moment", native)
    assert study.user_attrs["validation_schedule_sha256"] == "same"
    assert trial.user_attrs["validation_schedule_sha256"] == "same"

    import pytest
    with pytest.raises(RuntimeError, match="validation sample drift"):
        _guard_validation_fingerprint(
            study, _Trial(), "moment",
            {"sampling": {"validation_schedule_sha256": "different"}},
        )


def test_validation_fingerprint_is_required():
    import pytest
    with pytest.raises(RuntimeError, match="missing"):
        _guard_validation_fingerprint(_Study(), _Trial(), "kronos", {})


def test_mantis_selection_uses_primary_delta_then_auc_delta():
    class Trial:
        def __init__(self, number, values, promoted=False):
            self.number, self.values = number, values
            self.state = type("State", (), {"name": "COMPLETE"})()
            self.user_attrs = {"promotion_passed": promoted}

    study = type("Study", (), {"trials": [
        Trial(0, (-0.2, 0.9)), Trial(1, (-0.1, 0.5)), Trial(2, (-0.1, 0.6)),
    ]})()
    assert _selected_trial(study, "mantis_v2").number == 2


def test_mantis_selection_restricts_to_promoted_trials():
    class Trial:
        def __init__(self, number, values, promoted):
            self.number, self.values = number, values
            self.user_attrs = {"promotion_passed": promoted}
            self.state = type("State", (), {"name": "COMPLETE"})()

    study = type("Study", (), {"trials": [
        Trial(0, (0.2, 0.2), False), Trial(1, (0.1, 0.1), True),
    ]})()
    assert _selected_trial(study, "mantis_v2").number == 1
