"""Process tests for the head tuner: generalization-robust objective + the
accept/auto-fallback decision. The Optuna study and the heavy fold build/score
are mocked so the DECISION LOGIC is verified deterministically.
"""
import sys
import types

import pytest

from futures_foundation.pipeline import tune_head as TH


# ---- generalization-robust objective --------------------------------------
def test_gen_score_prefers_stable_over_peaky():
    # equal mean (1.0), but peaky [2,0] is penalized for instability;
    # the stable config must score strictly higher.
    stable = TH._gen_score([1.0, 1.0, 1.0])
    peaky = TH._gen_score([2.0, 0.0, 1.0])
    assert stable > peaky
    assert stable == pytest.approx(1.0)


def test_gen_score_penalty_is_mean_minus_penalty_times_std():
    import numpy as np
    pf = [1.5, 0.5]
    expected = float(np.mean(pf)) - TH.GEN_STD_PENALTY * float(np.std(pf))
    assert TH._gen_score(pf) == pytest.approx(expected)


def test_gen_score_empty_is_floor():
    # delegates to the shared library, whose empty floor is -inf
    assert TH._gen_score([]) == float('-inf')


# ---- accept / auto-fallback decision --------------------------------------
class _FakeStudy:
    """Stands in for an optuna study: no-op optimize, fixed best_params."""
    def __init__(self, best_params):
        self.best_params = best_params

    def optimize(self, objective, n_trials):    # objective never actually run
        return None


def _patch_tuner(monkeypatch, best_params, guard_means):
    """Mock _build_folds, optuna, and _pooled_meanR so only the decision runs.

    guard_means = (default_guard, tuned_guard) — the held-out meanR the loop
    compares to decide accept vs fall-back.
    """
    # 10 marker folds → split 7 tune / 3 guard (holdout_frac 0.3)
    folds = [{'i': k} for k in range(10)]
    monkeypatch.setattr(TH, '_build_folds', lambda *a, **k: folds)

    def fake_pooled(labeler, fold_subset, params, seed):
        is_guard = fold_subset[0]['i'] >= 7
        is_default = (params == {})
        if is_guard:
            mean = guard_means[0] if is_default else guard_means[1]
        else:                                    # tune folds (not the decider)
            mean = 0.30 if is_default else 0.80
        return mean, 100
    monkeypatch.setattr(TH, '_pooled_meanR', fake_pooled)

    fake_optuna = types.ModuleType('optuna')
    fake_optuna.logging = types.SimpleNamespace(
        set_verbosity=lambda *a, **k: None, WARNING=0)
    fake_optuna.samplers = types.SimpleNamespace(TPESampler=lambda **k: None)
    fake_optuna.pruners = types.SimpleNamespace(MedianPruner=lambda **k: None)
    fake_optuna.create_study = lambda **k: _FakeStudy(best_params)
    fake_optuna.TrialPruned = Exception
    monkeypatch.setitem(sys.modules, 'optuna', fake_optuna)


class _Lab:
    n_classes = 2


def test_accept_tuned_when_it_generalizes(monkeypatch):
    # tuned beats default on the GUARD by >= margin -> ACCEPT tuned params.
    best = {'max_depth': 3, 'min_child_weight': 30}
    _patch_tuner(monkeypatch, best_params=best, guard_means=(0.32, 0.45))
    res = TH.tune_head(_Lab(), n_trials=5, max_folds=10)
    assert res['generalizes'] is True
    assert res['chosen'] == 'tuned'
    assert res['params'] == best
    assert res['guard_lift'] == pytest.approx(0.45 - 0.32)


def test_fallback_to_defaults_when_no_held_out_lift(monkeypatch):
    # tuned does NOT beat default on the GUARD -> KEEP DEFAULTS (params={}).
    best = {'max_depth': 3}
    _patch_tuner(monkeypatch, best_params=best, guard_means=(0.32, 0.33))
    res = TH.tune_head(_Lab(), n_trials=5, max_folds=10)
    assert res['generalizes'] is False
    assert res['chosen'] == 'default'
    assert res['params'] == {}                  # auto-fallback
    assert res['best_params'] == best           # raw winner still reported


def test_fallback_when_tuned_overfits_guard_collapses(monkeypatch):
    # classic overfit: tuned aces TUNE but collapses on GUARD -> fall back.
    best = {'max_depth': 6, 'n_estimators': 800}
    _patch_tuner(monkeypatch, best_params=best, guard_means=(0.32, 0.00))
    res = TH.tune_head(_Lab(), n_trials=5, max_folds=10)
    assert res['generalizes'] is False
    assert res['params'] == {}
    assert res['guard_lift'] < 0
