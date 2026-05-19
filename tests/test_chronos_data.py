"""Born-tested: the Chronos data contract + LEAK-FREENESS.

Causal/leak-free splitting is the non-negotiable that undid prior work, so it
is pinned here before any fine-tune is built on it. Pure pandas; no torch.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from pipelines.chronos.data import load_long, walk_forward_folds, TICKERS

_HAS = (Path(__file__).parent.parent / 'data' / 'ES_3min.csv').exists()


def _toy(n_per_tk=24 * 30 * 6):              # ~6 months of hourly bars
    rows = []
    ts = pd.date_range('2022-01-01', periods=n_per_tk, freq='h', tz='UTC')
    rng = np.random.default_rng(0)
    for tk in ('AA', 'BB'):
        close = 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, n_per_tk)))
        rows.append(pd.DataFrame({'item_id': tk, 'timestamp': ts,
                                  'target': np.log(pd.Series(close)).diff()})
                    .dropna())
    return pd.concat(rows, ignore_index=True).sort_values(
        ['item_id', 'timestamp']).reset_index(drop=True)


def test_logret_is_causal_and_correct():
    c = pd.Series([100.0, 101.0, 99.0, 99.0])
    expect = np.log(c).diff()                      # uses only t and t-1
    assert np.allclose(expect.iloc[1], np.log(101.0 / 100.0))
    assert pd.isna(expect.iloc[0])                 # first bar has no return


def test_target_validation():
    with pytest.raises(ValueError):
        load_long(target='bogus')


def test_walk_forward_is_strictly_leak_free():
    df = _toy()
    folds = list(walk_forward_folds(df, train_months=3, test_months=1))
    assert len(folds) >= 1
    for i, tr, te in folds:
        # the core invariant: training never sees a bar at/after test start
        assert tr['timestamp'].max() < te['timestamp'].min()
        assert len(tr) and len(te)
        assert set(tr.columns) == {'item_id', 'timestamp', 'target'}


def test_folds_roll_and_are_deterministic():
    df = _toy()
    f1 = [(i, len(tr), len(te)) for i, tr, te in walk_forward_folds(df)]
    f2 = [(i, len(tr), len(te)) for i, tr, te in walk_forward_folds(df)]
    assert f1 == f2 and [x[0] for x in f1] == list(range(len(f1)))


@pytest.mark.skipif(not _HAS, reason='data/ES_3min.csv absent')
def test_real_data_loads_long_format():
    df = load_long('3min', tickers=['ES', 'NQ'], target='logret')
    assert set(df['item_id'].unique()) == {'ES', 'NQ'}
    assert list(df.columns) == ['item_id', 'timestamp', 'target']
    assert df['target'].notna().all()
    assert df['target'].abs().median() < 0.01        # 3-min logrets are tiny
    g = df.groupby('item_id')['timestamp']
    assert (g.is_monotonic_increasing).all()
