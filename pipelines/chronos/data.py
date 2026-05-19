"""Step 1 — give Chronos all our futures data, leak-free.

Assembles the 6-ticker bars into the long format Chronos/AutoGluon fine-tune
expects (item_id, timestamp, target) and yields rolling, strictly-causal
walk-forward folds. Pure pandas/numpy (no torch/chronos import) so the data
contract + leak-freeness are testable before any model exists.

target options:
  'logret' (default) — log return of close. Stationary + instrument-
      comparable, so all 6 tickers pool cleanly and a trend-following
      decision is just the sign of the cumulative forecast.
  'close'            — raw close (Chronos scales internally). Kept for
      comparison; non-stationary, not pooled-comparable.
"""
import os
import numpy as np
import pandas as pd

_DATA = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
TICKERS = ['ES', 'NQ', 'RTY', 'YM', 'GC', 'SI']


def load_long(timeframe: str = '3min', tickers=None, target: str = 'logret',
              data_dir: str | None = None) -> pd.DataFrame:
    """Return a long DataFrame [item_id, timestamp, target], sorted by
    (item_id, timestamp), no NaN target, strictly causal (logret uses only
    close[t]/close[t-1])."""
    tickers = tickers or TICKERS
    ddir = data_dir or _DATA
    if target not in ('logret', 'close'):
        raise ValueError(f"target must be 'logret' or 'close', got {target!r}")
    frames = []
    for tk in tickers:
        path = os.path.join(ddir, f'{tk}_{timeframe}.csv')
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        df = pd.read_csv(path, usecols=['datetime', 'close'])
        df['timestamp'] = pd.to_datetime(df['datetime'], utc=True)
        df = df.sort_values('timestamp').reset_index(drop=True)
        if target == 'logret':
            val = np.log(df['close']).diff()          # causal: t vs t-1
        else:
            val = df['close'].astype(float)
        out = pd.DataFrame({'item_id': tk, 'timestamp': df['timestamp'],
                            'target': val})
        out = out.dropna(subset=['target']).reset_index(drop=True)
        frames.append(out)
    return (pd.concat(frames, ignore_index=True)
              .sort_values(['item_id', 'timestamp'])
              .reset_index(drop=True))


def walk_forward_folds(long_df: pd.DataFrame, train_months: int = 3,
                       test_months: int = 1):
    """Yield (fold_idx, train_df, test_df) rolling month-aligned folds.
    HARD leak guard: train.timestamp.max() < test.timestamp.min() always.
    Unanchored (drop oldest month each step), stride = test_months."""
    ts = pd.DatetimeIndex(long_df['timestamp'])
    per = ts.tz_localize(None).to_period('M') if ts.tz is not None \
        else ts.to_period('M')
    months = per.unique().sort_values()
    fold = 0
    s = 0
    while s + train_months + test_months <= len(months):
        tr = months[s:s + train_months]
        te = months[s + train_months:s + train_months + test_months]
        trm = np.asarray(per.isin(tr))
        tem = np.asarray(per.isin(te))
        if trm.any() and tem.any():
            train_df = long_df[trm].reset_index(drop=True)
            test_df = long_df[tem].reset_index(drop=True)
            # non-negotiable: the fine-tune NEVER sees forward bars
            assert train_df['timestamp'].max() < test_df['timestamp'].min(), \
                f'LEAK: fold {fold} train overlaps/precedes test'
            fold += 1
            yield fold - 1, train_df, test_df
        s += test_months
