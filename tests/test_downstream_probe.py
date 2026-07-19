import numpy as np
import pytest

from futures_foundation.finetune.downstream_probe import (
    causal_feature_matrix,
    fit_predict_fold,
    fold_target_issue,
    paired_calendar_block_bootstrap,
    prediction_metrics,
    target_specs,
    target_values,
)


def _arrays(n=120):
    rng = np.random.default_rng(7)
    signal = rng.normal(size=n).astype(np.float32)
    terminal = np.column_stack((signal, signal, signal)).astype(np.float32)
    trend = np.where(signal[:, None] > 0, 1, 0).astype(np.int8)
    directions = np.asarray([1, -1], np.int8)
    context_direction = np.where(signal > 0, 1, -1).astype(np.int8)
    upside = np.abs(terminal) + 1.0
    downside = np.abs(terminal) + 0.5
    barrier = np.zeros((n, 3, 2, 3), np.int8)
    barrier[:, :, :, 0] = 1
    time_fav = np.full_like(barrier, 10, dtype=np.int32)
    time_adv = np.full_like(barrier, 20, dtype=np.int32)
    policy_r = np.ones_like(barrier, dtype=np.float32)
    return {
        "features": np.column_stack((signal, rng.normal(size=n))).astype(np.float32),
        "feature_names": np.array(["signal", "noise"]),
        "ticker": np.repeat(["ES", "NQ"], n // 2),
        "timeframe": np.full(n, "1min"),
        "horizons_minutes": np.array([60, 180, 360]),
        "terminal_move_r": terminal,
        "forward_abs_move_r": np.abs(terminal),
        "forward_realized_vol": np.abs(terminal) + 0.1,
        "forward_trend_eff": np.abs(terminal) / 2,
        "trend_path_class": trend,
        "targets_r": np.asarray([1.0, 2.0, 3.0], np.float32),
        "directions": directions,
        "context_direction": context_direction,
        "upside_mfe_r": upside,
        "downside_mae_r": downside,
        "barrier_state": barrier,
        "time_to_favorable_minutes": time_fav,
        "time_to_adverse_minutes": time_adv,
        "policy_r_gross": policy_r,
    }


def test_target_contract_and_causal_feature_identity():
    arrays = _arrays()
    specs = target_specs(arrays)
    assert len(specs) == 81
    continuation = next(spec for spec in specs if spec.name == "trend_continuation_60m")
    y, valid = target_values(arrays, continuation)
    assert valid.all() and set(np.unique(y)) == {0, 1}
    X, names = causal_feature_matrix(arrays, np.arange(120))
    assert X.shape == (120, 4)
    assert names[-2:] == ("ticker_ES", "ticker_NQ")


def test_linear_fold_is_predictive_and_controls_break_alignment():
    arrays = _arrays()
    X, _ = causal_feature_matrix(arrays, np.arange(120))
    y = (arrays["terminal_move_r"][:, 0] > 0).astype(np.int8)
    groups = arrays["ticker"]
    train, test = np.arange(0, 80), np.arange(80, 120)
    real = fit_predict_fold(
        X, y, groups, train, test, kind="bin", head="linear", seed=2,
    )
    shuffled = fit_predict_fold(
        X, y, groups, train, test, kind="bin", head="linear",
        control="shuffled_label", seed=2,
    )
    assert prediction_metrics(y[test], real, "bin")["auc"] > 0.95
    assert prediction_metrics(y[test], shuffled, "bin")["auc"] < 0.8


def test_regression_metrics_are_exact_for_perfect_predictions():
    y = np.array([1.0, 2.0, 3.0])
    metrics = prediction_metrics(y, y.copy(), "reg")
    assert metrics["r2"] == 1.0 and metrics["mae"] == 0.0
    assert metrics["spearman"] == pytest.approx(1.0)
    constant = prediction_metrics(y, np.ones_like(y), "reg")
    assert constant["spearman"] == 0.0


def test_direction_relative_path_and_barrier_targets():
    arrays = _arrays()
    specs = {spec.name: spec for spec in target_specs(arrays)}
    favorable, valid = target_values(arrays, specs["favorable_mfe_r_60m"])
    expected = np.where(
        arrays["context_direction"] > 0,
        arrays["upside_mfe_r"][:, 0], arrays["downside_mae_r"][:, 0],
    )
    assert valid.all()
    assert np.array_equal(favorable, expected)
    reached, reached_valid = target_values(
        arrays, specs["target_before_adverse_1r_60m"],
    )
    assert reached_valid.all() and reached.all()
    time, time_valid = target_values(arrays, specs["time_to_favorable_1r_60m"])
    assert time_valid.all() and np.allclose(time, 10 / 60)


def test_future_label_perturbations_cannot_change_causal_features():
    arrays = _arrays()
    rows = np.arange(len(arrays["ticker"]))
    before, names = causal_feature_matrix(arrays, rows)
    rng = np.random.default_rng(91)
    for key in (
        "terminal_move_r", "forward_abs_move_r", "forward_realized_vol",
        "forward_trend_eff", "upside_mfe_r", "downside_mae_r", "trend_path_class",
        "barrier_state", "time_to_favorable_minutes", "time_to_adverse_minutes",
        "policy_r_gross",
    ):
        arrays[key] = rng.permutation(arrays[key], axis=0)
    after, after_names = causal_feature_matrix(arrays, rows)

    assert names == after_names
    np.testing.assert_array_equal(before, after)


def test_one_return_volatility_and_efficiency_are_masked():
    arrays = _arrays()
    arrays["timeframe"] = np.full(120, "60min")
    specs = {spec.name: spec for spec in target_specs(arrays)}
    for name in ("forward_realized_vol_60m", "forward_trend_eff_60m"):
        _, valid = target_values(arrays, specs[name])
        assert not valid.any()
    _, direction_valid = target_values(arrays, specs["forward_direction_60m"])
    assert direction_valid.all()


def test_paired_calendar_bootstrap_detects_better_predictions():
    rng = np.random.default_rng(8)
    y = rng.normal(size=400)
    better = y + rng.normal(0, 0.1, len(y))
    worse = rng.normal(size=len(y))
    time = np.arange(len(y), dtype=np.int64) * 24 * 60 * 60 * 1_000_000_000
    result = paired_calendar_block_bootstrap(
        y, better, worse, time, kind="reg", repetitions=100, seed=3,
    )
    assert result["delta"] > 0.8
    assert result["ci_low_95"] > 0
    assert result["positive_probability"] == 1.0


def test_sparse_or_single_class_fold_is_explicitly_rejected():
    y = np.asarray([0] * 30 + [1] * 30)
    assert fold_target_issue(y, np.arange(10), np.arange(30, 35), "bin") == "insufficient_fold_rows"
    assert fold_target_issue(y, np.arange(25), np.arange(30, 40), "bin") == "single_class_train"
    assert fold_target_issue(y, np.arange(40), np.arange(40, 50), "bin") == "single_class_test"
    assert fold_target_issue(y, np.r_[0:20, 30:50], np.r_[20:30, 50:60], "bin") is None
