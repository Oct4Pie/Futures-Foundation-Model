"""Tests for the shared, metric-agnostic overfit/generalization library."""
import pytest

from futures_foundation import overfit as O


# ---- overfit_trigger -------------------------------------------------------
def test_overfit_trigger_fires_and_quiets():
    assert O.overfit_trigger(1.0, 0.0, 0.30) is True       # gap 1.0 > 0.30
    assert O.overfit_trigger(0.50, 0.40, 0.30) is False    # gap 0.10 < 0.30
    assert O.overfit_trigger(0.40, 0.50, 0.30) is False    # val > train
    # metric-agnostic: works for AUC tolerances too
    assert O.overfit_trigger(0.90, 0.80, 0.05) is True     # 0.10 AUC gap > 0.05
    assert O.overfit_trigger(0.82, 0.80, 0.05) is False    # 0.02 AUC gap < 0.05


# ---- generalizes -----------------------------------------------------------
def test_generalizes_gate():
    assert O.generalizes(0.50, 0.55, 0.30) is True         # test > val
    assert O.generalizes(0.50, 0.45, 0.30) is True         # small decay
    assert O.generalizes(1.00, 0.50, 0.30) is False        # 0.50 decay > 0.30
    assert O.generalizes(0.62, 0.55, 0.05) is False        # 0.07 AUC decay > 0.05
    assert O.generalizes(0.62, 0.59, 0.05) is True         # 0.03 AUC decay <= 0.05


def test_generalizes_missing_is_false():
    assert O.generalizes(None, 0.5, 0.3) is False
    assert O.generalizes(0.5, None, 0.3) is False


# ---- gen_score -------------------------------------------------------------
def test_gen_score_prefers_stable_over_peaky():
    stable = O.gen_score([1.0, 1.0, 1.0])
    peaky = O.gen_score([2.0, 0.0, 1.0])                   # same mean, unstable
    assert stable > peaky
    assert stable == pytest.approx(1.0)


def test_gen_score_matches_formula_and_penalty():
    import numpy as np
    pf = [1.5, 0.5]
    assert O.gen_score(pf, 0.5) == pytest.approx(
        float(np.mean(pf)) - 0.5 * float(np.std(pf)))
    # bigger penalty punishes instability harder
    assert O.gen_score(pf, 1.0) < O.gen_score(pf, 0.5)


def test_gen_score_empty_is_floor():
    assert O.gen_score([]) == float('-inf')
    assert O.gen_score(None) == float('-inf')


# ---- best_config (selection + auto-fallback) -------------------------------
def test_best_config_picks_highest_above_default():
    a, b, c = {'i': 1}, {'i': 2}, {'i': 3}
    assert O.best_config(0.10, [(a, 0.20), (b, 0.50), (c, 0.30)]) is b


def test_best_config_falls_back_when_none_beat_default():
    a, b = {'i': 1}, {'i': 2}
    assert O.best_config(0.60, [(a, 0.20), (b, 0.55)]) is None


def test_best_config_accept_margin():
    a = {'i': 1}
    # tuned 0.33 vs default 0.32: clears margin 0 but not margin 0.05
    assert O.best_config(0.32, [(a, 0.33)], accept_margin=0.0) is a
    assert O.best_config(0.32, [(a, 0.33)], accept_margin=0.05) is None


def test_best_config_empty_and_none_metrics():
    assert O.best_config(0.30, []) is None
    assert O.best_config(0.30, [({'i': 1}, None)]) is None
