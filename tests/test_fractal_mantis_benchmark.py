import numpy as np
import pandas as pd

from scripts.benchmark_fractal_mantis import (
    _fit_score_thresholds, _query_codes, _reach_grades, _session_days, _split_cache,
    _split_cache_calibrated, _take, _take_causal)


def test_split_purges_train_and_eval_forward_labels():
    ts = pd.DatetimeIndex([
        '2025-06-20', '2025-06-29', '2025-07-01', '2025-12-20',
        '2025-12-30', '2025-08-01',
    ], tz='UTC')
    label_end = pd.DatetimeIndex([
        '2025-06-25', '2025-07-03', '2025-07-03', '2025-12-28',
        '2026-01-03', '2025-08-05',
    ], tz='UTC')
    cache = {
        'ts': ts, 'label_end': label_end,
        'tk': np.array(['ES'] * len(ts)),
        'tf': np.array(['1min', '1min', '3min', '3min', '3min', '15min']),
    }
    tr, ev = _split_cache(cache, holdout_start='2025-07-01', eval_end='2026-01-01',
                          n_train=100, seed=0)
    assert tr.tolist() == [0]                 # row 1's label crosses into evaluation
    assert ev.tolist() == [2, 3]              # row 4's label crosses into locked 2026
    assert (label_end[tr] < pd.Timestamp('2025-07-01', tz='UTC')).all()
    assert (label_end[ev] < pd.Timestamp('2026-01-01', tz='UTC')).all()


def test_split_subsample_is_deterministic():
    ts = pd.date_range('2024-01-01', periods=100, freq='1D', tz='UTC')
    cache = {
        'ts': ts, 'label_end': ts + pd.Timedelta(hours=1),
        'tk': np.array(['ES'] * len(ts)),
        'tf': np.array(['3min'] * len(ts)),
    }
    a, _ = _split_cache(cache, holdout_start='2024-03-01', eval_end=None,
                        n_train=10, seed=7)
    b, _ = _split_cache(cache, holdout_start='2024-03-01', eval_end=None,
                        n_train=10, seed=7)
    assert np.array_equal(a, b) and len(a) == 10


def test_split_subsample_never_fragments_a_query_group():
    ts = pd.date_range('2024-01-01 23:00', periods=60, freq='4h', tz='UTC')
    cache = {
        'ts': ts, 'label_end': ts + pd.Timedelta(minutes=30),
        'tk': np.array(['ES'] * 30 + ['NQ'] * 30),
        'tf': np.array(['3min'] * len(ts)),
    }
    train, _ = _split_cache(cache, holdout_start='2024-01-06', eval_end=None,
                            n_train=20, seed=3)
    all_q = _query_codes(cache['tk'], cache['tf'], ts)
    selected = set(all_q[train])
    eligible = np.flatnonzero((ts < pd.Timestamp('2024-01-06', tz='UTC')) &
                              (cache['label_end'] < pd.Timestamp('2024-01-06', tz='UTC')))
    for q in selected:
        group_eligible = set(eligible[all_q[eligible] == q])
        assert group_eligible.issubset(set(train))
    assert len(train) <= 20


def test_take_is_exact_per_ticker_session_not_period_quota():
    ts = pd.DatetimeIndex([
        '2025-01-02 23:00Z', '2025-01-03 01:00Z', '2025-01-03 15:00Z',
        '2025-01-03 23:00Z', '2025-01-04 01:00Z',
    ])
    session = _session_days(ts)
    score = np.array([.1, .9, .2, .3, .8])
    got = _take(score, np.array(['ES'] * 5), session, rate=1)
    assert got.tolist() == [1, 4]                         # best one inside each session


def test_reach_grades_require_nested_first_touch_labels():
    reached = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=bool)
    assert _reach_grades(reached).tolist() == [0, 1, 3]
    with np.testing.assert_raises(ValueError):
        _reach_grades(np.array([[0, 1, 0]], dtype=bool))


def test_calibrated_split_purges_train_calibration_and_eval_boundaries():
    ts = pd.date_range('2024-01-01', '2025-12-20', freq='10D', tz='UTC')
    cache = {
        'ts': ts, 'label_end': ts + pd.Timedelta(days=3),
        'tk': np.array(['ES'] * len(ts)), 'tf': np.array(['3min'] * len(ts)),
    }
    tr, ca, ev, cs = _split_cache_calibrated(
        cache, holdout_start='2025-07-01', eval_end='2026-01-01',
        calibration_months=6, n_train=1000, seed=0)
    le = cache['label_end']
    hs, ee = pd.Timestamp('2025-07-01', tz='UTC'), pd.Timestamp('2026-01-01', tz='UTC')
    assert (ts[tr] < cs).all() and (le[tr] < cs).all()
    assert (ts[ca] >= cs).all() and (ts[ca] < hs).all() and (le[ca] < hs).all()
    assert (ts[ev] >= hs).all() and (ts[ev] < ee).all() and (le[ev] < ee).all()


def test_causal_take_cannot_replace_earlier_trade_with_later_higher_score():
    ts = pd.DatetimeIndex(['2025-01-02 23:00Z', '2025-01-03 01:00Z'])
    session = _session_days(ts)
    tickers = np.array(['ES', 'ES'])
    score = np.array([.6, .99])
    thresholds = {'ES': {1: .5}}
    assert _take_causal(score, tickers, session, ts, thresholds, 1).tolist() == [0]
    assert _take(score, tickers, session, 1).tolist() == [1]  # diagnostic uses hindsight


def test_thresholds_use_calibration_scores_only():
    score = np.array([.9, .8, .7, .6, .5, .4])
    tickers = np.array(['ES'] * 6)
    sessions = np.array(['d1', 'd1', 'd1', 'd2', 'd2', 'd2'])
    thresholds = _fit_score_thresholds(score, tickers, sessions, rates=(1, 2))
    assert thresholds['ES'][1] == .8                    # two sessions -> two raw qualifiers
    assert thresholds['ES'][2] == .6                    # four raw qualifiers
