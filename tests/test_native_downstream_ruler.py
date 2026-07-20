from __future__ import annotations

import numpy as np

from futures_foundation.finetune.native_downstream_ruler import (
    PRIMARY_TARGETS,
    SCREEN_POLICY,
    fit_fold_arms,
    fit_model_bottleneck,
    root_calendar_block_bootstrap,
    screen_verdict,
)


def test_fold_local_bottleneck_is_finite_and_control_specific():
    rng = np.random.default_rng(7)
    features = rng.normal(size=(120, 40)).astype(np.float32)
    groups = np.repeat(np.asarray(["A", "B", "C"]), 40)
    train = np.arange(0, 80)
    test = np.arange(80, 120)
    real_train, real_test, real_meta = fit_model_bottleneck(
        features, groups, train, test,
        control="real", components=12, seed=11,
    )
    destroyed_train, destroyed_test, destroyed_meta = fit_model_bottleneck(
        features, groups, train, test,
        control="time_destroyed", components=12, seed=11,
    )
    assert real_train.shape == (80, 12)
    assert real_test.shape == (40, 12)
    assert destroyed_train.shape == real_train.shape
    assert destroyed_test.shape == real_test.shape
    assert np.isfinite(real_train).all()
    assert real_meta["components"] == 12
    assert destroyed_meta["control"] == "time_destroyed"
    assert not np.array_equal(real_train, destroyed_train)


def test_residual_over_causal_recovers_incremental_regression_signal():
    rng = np.random.default_rng(19)
    rows = 240
    causal = rng.normal(size=(rows, 2)).astype(np.float32)
    model = rng.normal(size=(rows, 3)).astype(np.float32)
    y = (2.0 * causal[:, 0] + 1.5 * model[:, 0] + 0.05 * rng.normal(size=rows)).astype(np.float32)
    groups = np.repeat(np.asarray(["A", "B", "C", "D"]), rows // 4)
    train = np.arange(0, 180)
    test = np.arange(180, rows)
    predictions = fit_fold_arms(
        causal,
        model[train],
        model[test],
        y,
        groups,
        train,
        test,
        kind="reg",
        sample_weight=None,
        seed=23,
        random_train=rng.normal(size=(len(train), 3)).astype(np.float32),
        random_test=rng.normal(size=(len(test), 3)).astype(np.float32),
        destroyed_train=model[train][::-1].copy(),
        destroyed_test=model[test][::-1].copy(),
    )
    causal_mse = np.mean((y[test] - predictions["causal"]) ** 2)
    residual_mse = np.mean((y[test] - predictions["residual_over_causal"]) ** 2)
    combined_mse = np.mean((y[test] - predictions["causal_plus_model"]) ** 2)
    assert residual_mse < causal_mse * 0.1
    assert combined_mse < causal_mse * 0.1


def test_root_calendar_bootstrap_reports_positive_paired_delta():
    rng = np.random.default_rng(29)
    rows = 360
    y = rng.normal(size=rows)
    better = y + 0.1 * rng.normal(size=rows)
    worse = y + 0.8 * rng.normal(size=rows)
    time = (
        np.arange(rows, dtype=np.int64) * 24 * 60 * 60 * 1_000_000_000
    )
    roots = np.resize(np.asarray(["ES", "NQ", "CL"]), rows)
    result = root_calendar_block_bootstrap(
        y, better, worse, time, roots,
        kind="reg", block_days=7, repetitions=200,
        confidence=0.99, seed=31,
    )
    assert result["delta"] > 0
    assert result["ci_low_99"] > 0
    assert result["positive_probability"] > 0.99


def _primary_row(target: str, *, pass_row: bool) -> dict:
    if pass_row:
        model, controls, combined, residual, ci = 0.2, 0.1, 0.25, 0.27, 0.01
    else:
        model, controls, combined, residual, ci = 0.05, 0.1, 0.08, 0.07, -0.02
    return {
        "target": target,
        "kind": "reg",
        "metrics": {
            "causal": {"r2": 0.1},
            "model": {"r2": model},
            "causal_plus_model": {"r2": combined},
            "residual_over_causal": {"r2": residual},
            "model_shuffled_label": {"r2": controls},
            "model_random_feature": {"r2": controls - 0.01},
            "model_time_destroyed": {"r2": controls - 0.02},
        },
        "bootstrap": {
            "causal_plus_model": {"delta": combined - 0.1},
            "residual_over_causal": {"delta": residual - 0.1, "ci_low_99": ci},
        },
    }


def test_screen_verdict_uses_exact_primary_closure_and_never_promotes():
    rows = [
        _primary_row(target, pass_row=index < 4)
        for index, target in enumerate(PRIMARY_TARGETS)
    ]
    verdict = screen_verdict(rows)
    assert verdict["policy"]["policy_id"] == SCREEN_POLICY["policy_id"]
    assert verdict["downstream_screen_survived"] is True
    assert verdict["nonlinear_sensitivity_funded"] is True
    assert verdict["promotion_admitted"] is False
    assert verdict["full_training_admitted"] is False
