"""Phase-1 distributional forward-R risk head: label extraction, survival-curve math, head.

The head predicts a CALIBRATED survival curve P(reach >= Xr) from the embedding (the accurate
replacement for the snapshot peak-R regressor). These tests lock: labels come from the existing
keys (no relabel), the curve is a valid survival function (monotone, [0,1]), the stats (expected
reach / dynamic TP / big-win prob) are correct, and calibration is leak-free + preserves ranking.

Ladder = TARGETS (2,3,4,6,8) — the pivot's FIXED_TARGETS; 8R is the max-trend rung. WR@3R stays the
ship metric downstream; here we lock the ladder math on all 5 rungs.
"""
import numpy as np

from futures_foundation.finetune.risk_head import (
    reach_labels, monotone_survival, survival_to_stats, RiskHead, TARGETS)

NT = len(TARGETS)                                        # 5 rungs: 2,3,4,6,8


def _key(realized):                                      # (name,i,d,peak, R@2,R@3,R@4,R@6,R@8)
    assert len(realized) == NT
    return ('ES@3min', 0, 1, 0.0, *realized)


# ---------------------------------------------------------------- labels from keys
def test_reach_labels_from_keys():
    keys = [_key([1.97, 2.97, -0.5, -0.5, -0.5]),         # reached 3R (not 4R/6R/8R)
            _key([-1.03, -1.03, -1.03, -1.03, -1.03]),    # stopped immediately
            _key([1.97, 2.97, 3.97, 5.97, 7.97])]         # ran all the way to 8R
    lab = reach_labels(keys)
    assert lab.shape == (3, NT)
    assert lab[0].tolist() == [1, 1, 0, 0, 0]
    assert lab[1].tolist() == [0, 0, 0, 0, 0]
    assert lab[2].tolist() == [1, 1, 1, 1, 1]


# ---------------------------------------------------------------- survival curve is valid
def test_monotone_survival_enforced_and_clipped():
    s = monotone_survival([[0.3, 0.5, 0.2, 0.1, 0.05], [1.2, 0.9, 0.9, -0.1, 0.5]])
    assert np.all(np.diff(s, axis=1) <= 1e-12)           # non-increasing
    assert s.min() >= 0 and s.max() <= 1                 # clipped to [0,1]
    assert np.allclose(s[0], [0.3, 0.3, 0.2, 0.1, 0.05]) # 0.5 pulled down to 0.3


def test_survival_to_stats_always_reaches():
    st = survival_to_stats([[1, 1, 1, 1, 1]])            # certain to reach 8R
    assert np.isclose(st['exp_reach'][0], TARGETS[-1])   # E[peak] == 8
    assert st['tp'][0] == TARGETS[-1] and st['p_bigwin'][0] == 1.0


def test_survival_to_stats_reaches_2R_only():
    st = survival_to_stats([[1, 0, 0, 0, 0]])
    assert np.isclose(st['exp_reach'][0], 2.5)           # 1*2 + 0.5*(1+0)*1
    assert st['tp'][0] == 2.0 and st['p_bigwin'][0] == 0.0


def test_survival_to_stats_dynamic_tp_threshold():
    # q_tp=0.33: reach probs [0.5,0.4,0.3,0.2,0.1] -> only 2R(0.5) & 3R(0.4) clear 0.33 -> TP=3R
    st = survival_to_stats([[0.5, 0.4, 0.3, 0.2, 0.1]], q_tp=0.33)
    assert st['tp'][0] == 3.0


# ---------------------------------------------------------------- the head
def _toy(n, D=8, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, D)).astype(np.float32)
    w = rng.normal(size=D)
    s = X @ w
    offs = [0.0, 0.8, 1.6, 2.8, 4.0]                      # higher target = harder (5 rungs)
    keys = []
    for r in range(n):
        realized, reached_prev = [], True
        for off in offs:
            p = 1.0 / (1.0 + np.exp(-(s[r] - off)))
            hit = reached_prev and (rng.uniform() < p)    # can't reach 4R without 2R (monotone)
            reached_prev = hit
            realized.append(1.0 if hit else -1.0)
        keys.append(_key(realized))
    return X, keys


def test_risk_head_survival_is_valid():
    Xtr, ktr = _toy(2000, seed=1)
    Xval, kval = _toy(1000, seed=2)
    Xev, _ = _toy(1000, seed=3)
    h = RiskHead(head='logistic', calibrate=True).fit(Xtr, ktr, Xval, kval)
    surv = h.predict_survival(Xev)
    assert surv.shape == (1000, NT)
    assert surv.min() >= 0 and surv.max() <= 1
    assert np.all(np.diff(surv, axis=1) <= 1e-9)          # valid survival curve out of the head


def test_risk_head_calibrates_and_ranks():
    from sklearn.metrics import roc_auc_score
    Xtr, ktr = _toy(2500, seed=4)
    Xval, kval = _toy(1500, seed=5)
    h = RiskHead(head='logistic', calibrate=True).fit(Xtr, ktr, Xval, kval)
    # calibrator fit for the non-degenerate 3R threshold, and ranking preserved (Platt monotone)
    clf, platt = h._heads[1]
    assert isinstance(platt, tuple) and len(platt) == 2
    y3 = reach_labels(kval)[:, 1]
    raw = clf.predict_proba(Xval)[:, 1]
    cal = h.predict_survival(Xval)[:, 1]
    if len(np.unique(y3)) == 2:
        assert abs(roc_auc_score(y3, raw) - roc_auc_score(y3, cal)) < 1e-6


def test_risk_head_mlp_runs_and_stats():
    Xtr, ktr = _toy(1500, seed=6)
    Xev, _ = _toy(400, seed=7)
    h = RiskHead(head='mlp', calibrate=False, mlp_batch=256, max_iter=100).fit(Xtr, ktr)
    st = h.predict_stats(Xev)
    assert st['exp_reach'].shape == (400,)
    assert np.all(st['exp_reach'] >= 0) and np.all(st['tp'] >= TARGETS[0])
    assert st['surv'].min() >= 0 and st['surv'].max() <= 1


def test_expected_reach_ranks_bigger_runners_higher():
    # a pivot that reaches every rung must out-rank one that dies at 2R (the whole point of
    # ranking by expected-R: big-runner detection surfaces the trends that run)
    runner = survival_to_stats([[1, 1, 1, 1, 1]])['exp_reach'][0]
    stall = survival_to_stats([[1, 0, 0, 0, 0]])['exp_reach'][0]
    assert runner > stall


def test_parallel_rungs_match_sequential():
    # the rungs are independent + per-rung seeded, so fitting them concurrently must give the SAME
    # survival curve as sequential (rung_jobs=1). Locks that the parallel speedup is result-neutral.
    Xtr, ktr = _toy(1200, seed=20)
    Xev, _ = _toy(300, seed=21)
    seq = RiskHead(head='logistic', calibrate=False, rung_jobs=1).fit(Xtr, ktr)
    par = RiskHead(head='logistic', calibrate=False, rung_jobs=4).fit(Xtr, ktr)
    assert np.allclose(seq.predict_survival(Xev), par.predict_survival(Xev))


def test_degenerate_threshold_is_constant_head():
    # a threshold never reached -> constant head, no crash, survival still valid
    X = np.random.default_rng(0).normal(size=(300, 8)).astype(np.float32)
    keys = [_key([1.0, -1.0, -1.0, -1.0, -1.0]) for _ in range(300)]  # 3R+ never hit
    h = RiskHead(head='logistic', calibrate=True).fit(X, keys, X, keys)
    surv = h.predict_survival(X)
    assert surv.shape == (300, NT)
    assert np.allclose(surv[:, 1:], 0.0)                 # unreached thresholds -> ~0
