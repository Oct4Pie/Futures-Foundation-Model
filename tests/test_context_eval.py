"""Tests for the context-head overfit-driven evaluation (context_eval).

head_verdict is the pure decision core; run_context_eval is exercised on small
synthetic embeddings + labels (real tiny XGBoost, no foundation embedding).
"""
import numpy as np
import pandas as pd

from futures_foundation import context_eval as CE


# ---- head_verdict (pure) ---------------------------------------------------
def test_reg_head_accurate():
    v = CE.head_verdict('reg', train_m=0.60, val_m=0.55, test_m=0.52,
                        trivial_m=0.20, shuffle_m=0.00)
    assert v['generalizes'] and v['has_skill']
    assert v['beats_trivial'] and v['beats_shuffle']
    assert v['accurate'] is True


def test_reg_head_fails_when_not_generalizing():
    # val 0.55 → test 0.40 = 0.15 decay > REG_GEN_GAP 0.10
    v = CE.head_verdict('reg', 0.60, 0.55, 0.40, trivial_m=0.20, shuffle_m=0.0)
    assert v['generalizes'] is False
    assert v['accurate'] is False


def test_reg_head_fails_below_skill_floor():
    v = CE.head_verdict('reg', 0.06, 0.05, 0.03, trivial_m=0.0, shuffle_m=0.0)
    assert v['has_skill'] is False        # 0.03 < GATE_REG_PEARSON 0.05
    assert v['accurate'] is False


def test_reg_head_fails_when_not_beating_trivial():
    v = CE.head_verdict('reg', 0.60, 0.55, 0.52, trivial_m=0.60, shuffle_m=0.0)
    assert v['beats_trivial'] is False
    assert v['accurate'] is False


def test_clf_head_accurate_and_overfit_flag():
    v = CE.head_verdict('clf', train_m=0.95, val_m=0.78, test_m=0.76,
                        trivial_m=0.60, shuffle_m=0.50)
    assert v['generalizes'] and v['has_skill'] and v['accurate']
    assert v['overfit'] is True           # 0.95−0.78 = 0.17 > CLF_OVERFIT_GAP


def test_clf_head_not_generalizing():
    v = CE.head_verdict('clf', 0.80, 0.78, 0.70, trivial_m=0.6, shuffle_m=0.5)
    assert v['generalizes'] is False      # 0.08 AUC decay > 0.05
    assert v['accurate'] is False


def test_verdict_controls_optional():
    # no controls given → not penalized for them
    v = CE.head_verdict('reg', 0.60, 0.55, 0.52)
    assert v['beats_trivial'] and v['beats_shuffle'] and v['accurate']


# ---- run_context_eval (synthetic end-to-end) -------------------------------
def _synth(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    E = rng.normal(size=(n, 8)).astype(np.float32)
    ts = pd.Series(pd.date_range('2019-01-01', '2024-06-01', periods=n, tz='UTC'))
    # learnable: volatility (reg) from E[:,0]; vol_expansion (clf) from E[:,1]
    y_reg = (3.0 * E[:, 0] + 0.2 * rng.normal(size=n)).astype(np.float32)
    y_clf = (E[:, 1] + 0.2 * rng.normal(size=n) > 0).astype(np.float32)
    labels = pd.DataFrame({'volatility': y_reg, 'vol_expansion': y_clf})
    T = E[:, 5:7].copy()                  # trivial baseline = unrelated columns
    return E, labels, ts, T


def test_run_context_eval_learnable_heads_generalize():
    E, labels, ts, T = _synth()
    res = CE.run_context_eval(
        E, labels, ts,
        val_start=pd.Timestamp('2022-11-01', tz='UTC'),
        cutoff=pd.Timestamp('2023-01-01', tz='UTC'),
        T=T, specs=[('volatility', 'reg'), ('vol_expansion', 'clf')],
        verbose=False)
    assert set(res) == {'volatility', 'vol_expansion'}
    for name, v in res.items():
        assert v['test'] is not None and v['n_test'] >= 30
        assert v['has_skill'] is True          # learnable → clears the floor
        assert v['generalizes'] is True        # signal holds val→test
        assert v['beats_shuffle'] is True      # real signal beats permuted labels


def test_run_context_eval_skips_thin_heads():
    E, labels, ts, T = _synth(n=2000)
    labels['empty_head'] = np.nan              # no finite rows → skipped
    res = CE.run_context_eval(
        E, labels, ts,
        val_start=pd.Timestamp('2022-11-01', tz='UTC'),
        cutoff=pd.Timestamp('2023-01-01', tz='UTC'),
        T=T, specs=[('empty_head', 'reg'), ('volatility', 'reg')],
        verbose=False)
    assert 'empty_head' not in res
    assert 'volatility' in res
