"""Process tests for evaluate.py's new overfit/generalization logic:
  - _overfit_trigger : the auto-regularize trigger (train≫val)
  - _best_rung       : regularization-rung selection on VALIDATION only
  - _operating_verdict: the pre-registered verdict incl. the HARD VAL→TEST gate

These are the pure decision units the production walk-forward (`run`) now calls,
so testing them tests the real training path.
"""
from futures_foundation.pipeline import evaluate as ev


# ---- _overfit_trigger ------------------------------------------------------
def test_overfit_trigger_fires_when_train_far_above_val():
    assert ev._overfit_trigger(1.0, 0.0) is True            # gap 1.0 > tol
    assert ev._overfit_trigger(0.5, 0.5 - ev.OVERFIT_GAP_R - 0.01) is True


def test_overfit_trigger_quiet_when_generalizing():
    assert ev._overfit_trigger(0.5, 0.5) is False           # no gap
    assert ev._overfit_trigger(0.5, 0.4) is False           # gap 0.1 < tol 0.30
    assert ev._overfit_trigger(0.2, 0.5) is False           # val>train


# ---- _best_rung ------------------------------------------------------------
def test_best_rung_picks_highest_val_above_default():
    a, b, c = {'d': 2}, {'d': 3}, {'d': 4}
    best = ev._best_rung(0.10, [(a, 0.20), (b, 0.50), (c, 0.30)])
    assert best is b                                         # 0.50 is best


def test_best_rung_keeps_default_when_none_beat_it():
    a, b = {'d': 2}, {'d': 3}
    # default val 0.60; no rung beats it -> None (keep default head)
    assert ev._best_rung(0.60, [(a, 0.20), (b, 0.55)]) is None


def test_best_rung_empty_keeps_default():
    assert ev._best_rung(0.30, []) is None


# ---- _operating_verdict (incl. the generalization gate) --------------------
def _at(meanR=0.5, wr=0.60, n=200):
    return dict(meanR=meanR, wr=wr, n=n, pf=2.0)


def _strong_edges():
    # REAL clearly beats controls
    return dict(real_m=0.50, shuf_m=0.05, rand_m=0.05, naive_m=0.10)


def test_verdict_passes_when_generalizes_and_edge_strong():
    e = _strong_edges()
    checks, all_pass, v = ev._operating_verdict(
        _at(), _at(), gap=-0.10, real_m=e['real_m'], shuf_m=e['shuf_m'],
        rand_m=e['rand_m'], naive_m=e['naive_m'], thr=0.6)
    assert all_pass is True
    assert v['generalizes'] is True
    assert v['gap'] == -0.10 and v['thr'] == 0.6


def test_verdict_fails_when_does_not_generalize_even_if_edge_strong():
    # FAKE EDGE: strong controls margin but val→test gap blows the gate.
    e = _strong_edges()
    bad_gap = ev.GEN_GAP_TOL + 0.20
    checks, all_pass, v = ev._operating_verdict(
        _at(), _at(), gap=bad_gap, real_m=e['real_m'], shuf_m=e['shuf_m'],
        rand_m=e['rand_m'], naive_m=e['naive_m'], thr=0.6)
    assert v['generalizes'] is False
    assert all_pass is False                                 # gen-gate hard-fails
    # and the failing check is specifically the generalization one
    gen_check = [ok for ok, msg in checks if 'GENERALIZES' in msg][0]
    assert gen_check is False


def test_verdict_fails_when_edge_too_weak():
    # generalizes fine, but REAL barely beats SHUFFLE -> edge checks fail.
    checks, all_pass, v = ev._operating_verdict(
        _at(), _at(), gap=-0.05, real_m=0.10, shuf_m=0.09, rand_m=0.09,
        naive_m=0.09, thr=0.6)
    assert v['generalizes'] is True
    assert all_pass is False


# ---- _pooled_auc -----------------------------------------------------------
def _rec(proba, y):
    import numpy as np
    return (None, np.asarray(proba, np.float32), np.asarray(y, np.int8), None)


def test_pooled_auc_perfect_ranking():
    # winners all scored above losers → AUC 1.0
    r = _rec([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0])
    assert ev._pooled_auc([r]) == 1.0


def test_pooled_auc_random_is_half():
    r = _rec([0.5, 0.5, 0.5, 0.5], [1, 0, 1, 0])
    assert ev._pooled_auc([r]) == 0.5


def test_pooled_auc_pools_across_records():
    a = _rec([0.9, 0.1], [1, 0])
    b = _rec([0.8, 0.2], [1, 0])
    assert ev._pooled_auc([a, b]) == 1.0


def test_pooled_auc_degenerate_single_class_none():
    assert ev._pooled_auc([_rec([0.7, 0.3], [1, 1])]) is None
    assert ev._pooled_auc([]) is None


def test_verdict_carries_auc():
    e = _strong_edges()
    _, _, v = ev._operating_verdict(
        _at(), _at(), gap=-0.05, real_m=e['real_m'], shuf_m=e['shuf_m'],
        rand_m=e['rand_m'], naive_m=e['naive_m'], thr=0.6, auc=0.61)
    assert v['auc'] == 0.61


def test_verdict_gap_none_does_not_generalize():
    e = _strong_edges()
    checks, all_pass, v = ev._operating_verdict(
        _at(), None, gap=None, real_m=e['real_m'], shuf_m=e['shuf_m'],
        rand_m=e['rand_m'], naive_m=e['naive_m'], thr=0.6)
    assert v['generalizes'] is False
    assert all_pass is False
    assert v['test_meanR'] is None and v['test_n'] == 0
