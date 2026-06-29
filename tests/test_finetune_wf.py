"""Walk-forward honest ruler (finetune/wf.py) — full flow, torch-free.

Uses a synthetic StrategyLabeler + the torch-free 'logistic' classifier so the
ENTIRE pipeline is exercised — walk_forward_folds, mv_contexts, the isolated
subprocess worker, realized-R eval, REAL/SHUFFLE/RANDOM, and the verdict — with
no torch. (Mantis-specific behavior is covered in test_mantis_classifier.py.)
"""
import numpy as np
import pandas as pd
import pytest

from futures_foundation.finetune import wf


class SyntheticLabeler:
    """Daily bars over ~5y; class-1 windows carry a separable signal (channel 0
    elevated in the last bars). evaluate: a taken good pivot = +2R, bad = -1R."""
    n_classes = 2

    def __init__(self, n_bars=1600, seq=32, C=4, seed=0):
        rng = np.random.default_rng(seed)
        self.ts = pd.date_range('2020-01-01', periods=n_bars, freq='1D', tz='UTC')
        self.y = rng.integers(0, 2, n_bars)
        self.W = rng.standard_normal((n_bars, C, seq)).astype(np.float32)
        self.W[self.y == 1, 0, -6:] += 2.5
        self.seq, self.C = seq, C

    def calendar(self):
        return pd.DataFrame({'item_id': 'SYN', 'timestamp': self.ts, 'target': self.y})

    def build(self, lo, hi, test_start):
        idx = np.flatnonzero((self.ts >= lo) & (self.ts < hi))
        K = [(int(i),) for i in idx]
        Y = self.y[idx]
        C = [self.W[i, 0] for i in idx]      # 1-D context (protocol filler)
        return C, Y, K

    def mv_contexts(self, keys):
        return np.stack([self.W[k[0]] for k in keys]) if keys else np.zeros((0, self.C, self.seq))

    def evaluate(self, keys, preds):
        return np.array([(2.0 if self.y[k[0]] == 1 else -1.0)
                         for k, p in zip(keys, preds) if p == 1])


# ---- pure helper unit tests ------------------------------------------------
def test_pct_threshold():
    p = np.array([0.1, 0.2, 0.3, 0.4])
    assert wf._pct_threshold(p, 0.5) == pytest.approx(0.25, abs=1e-9)
    assert wf._pct_threshold(np.array([]), 0.5) == 1.0


def test_meanR():
    assert wf._meanR([1.0, -1.0, 2.0]) == pytest.approx(2 / 3)
    assert wf._meanR([]) == 0.0


def test_arm_R_selects_top_and_evaluates():
    lab = SyntheticLabeler(n_bars=50)
    keys = [(i,) for i in range(20)]
    proba = np.linspace(0, 1, 20)
    R = wf._arm_R(lab, keys, proba, thr=0.5)       # take top ~half
    assert R.ndim == 1 and len(R) > 0
    # all-skip -> empty
    assert wf._arm_R(lab, keys, np.zeros(20), thr=0.5).size == 0


# ---- full flow (subprocess worker + logistic, no torch) --------------------
def test_wf_full_flow_real_beats_controls():
    lab = SyntheticLabeler(n_bars=1600, seed=0)
    v = wf.run(lab, classifier='logistic', clf_kwargs=dict(),
               seeds=(0,), train_m=12, val_m=3, test_m=3, max_folds=1, verbose=False)
    # verdict shape
    for k in ('all_pass', 'auc', 'real_meanR', 'shuffle_meanR', 'random_meanR',
              'gap', 'n_folds', 'real_trades'):
        assert k in v
    assert v['n_folds'] >= 1
    assert v['real_trades'] > 0
    # the signal is learnable -> REAL must beat both controls and discriminate
    assert v['auc'] is not None and v['auc'] > 0.8
    assert v['real_meanR'] - v['shuffle_meanR'] >= wf.PASS_LIFT_MARGIN_R
    assert v['real_meanR'] - v['random_meanR'] >= wf.PASS_LIFT_MARGIN_R


def test_wf_no_productive_folds_returns_zero():
    lab = SyntheticLabeler(n_bars=1600, seed=0)
    # holdout in the far future -> no test data -> zero folds, graceful
    v = wf.run(lab, classifier='logistic', seeds=(0,), train_m=12, val_m=3,
               test_m=3, max_folds=1, holdout_start='2099-01-01', verbose=False)
    assert v['n_folds'] == 0 and v['real_trades'] == 0
