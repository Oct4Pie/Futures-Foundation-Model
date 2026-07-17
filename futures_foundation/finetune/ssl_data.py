"""SSL data assembly — torch-free, raw OHLCV from the data/ CSVs.

Loads raw OHLCV bars for the 9 futures tickers x {1,3,5,15,30,60}min from a configurable
directory (local `data/` or a Colab Google-Drive mount), and yields the leak-safe
time split the contrastive pretraining trains on:

  * per-stream OHLCV arrays (float32 [N, 5] = open/high/low/close/volume) + timestamps
  * a strictly causal TRAIN / VAL split (VAL = the last `val_frac` of each stream's
    PRE-HOLDOUT timeline) so the val NT-Xent early-stop measures generalization
    forward in time — the SSL analogue of the WF VAL->TEST gap
  * the 2026 HOLDOUT is EXCLUDED entirely (never seen by SSL) so the downstream
    classifier's 2026 OOS stays uncontaminated by backbone adaptation

No torch here (testable without the GPU stack). The torch trainer (_ssl_torch)
consumes these arrays + the window-start indices.
"""
import os

import numpy as np
import pandas as pd

TICKERS_9 = ['ES', 'NQ', 'RTY', 'YM', 'GC', 'SI', 'CL', 'ZB', 'ZN']
TFS_ALL = ['1min', '3min', '5min', '15min', '30min', '60min']
OHLCV_COLS = ['open', 'high', 'low', 'close', 'volume']

_DATA = os.path.join(os.path.dirname(__file__), '..', '..', 'data')


def load_ohlcv(data_dir=None, tickers=None, tfs=None, verbose=True, *, start=None, end=None,
               chunksize=250_000):
    """Return a list of stream dicts {sid, ticker, tf, ohlcv[N,5], ts[N]} for every
    (ticker, tf) CSV found under `data_dir` (default repo data/). Missing files are
    skipped with a note (not all tickers have every TF historically)."""
    ddir = data_dir or _DATA
    tickers = tickers or TICKERS_9
    tfs = tfs or TFS_ALL
    streams = []
    for tk in tickers:
        for tf in tfs:
            path = os.path.join(ddir, f'{tk}_{tf}.csv')
            if not os.path.exists(path):
                if verbose:
                    print(f"  [ssl-data] skip (missing) {tk}_{tf}", flush=True)
                continue
            available = set(pd.read_csv(path, nrows=0).columns)
            required = {'datetime', *OHLCV_COLS}
            missing = required - available
            if missing:
                raise ValueError(f"{path}: missing required columns {sorted(missing)}")
            optional = ['contract_id'] if 'contract_id' in available else []
            usecols = ['datetime'] + OHLCV_COLS + optional
            if start is None and end is None:
                df = pd.read_csv(path, usecols=usecols)
                df['ts'] = pd.to_datetime(df['datetime'], utc=True)
            else:
                lower = _utc_timestamp(start) if start is not None else None
                upper = _utc_timestamp(end) if end is not None else None
                if lower is not None and upper is not None and lower >= upper:
                    raise ValueError(f"load start must precede end: {lower} >= {upper}")
                pieces = []
                for chunk in pd.read_csv(path, usecols=usecols, chunksize=int(chunksize)):
                    timestamps = pd.to_datetime(chunk['datetime'], utc=True, errors='coerce')
                    if timestamps.isna().any():
                        raise ValueError(f"{path}: invalid timestamps")
                    keep = np.ones(len(chunk), dtype=bool)
                    if lower is not None:
                        keep &= timestamps >= lower
                    if upper is not None:
                        keep &= timestamps < upper
                    if keep.any():
                        selected = chunk.loc[keep].copy()
                        selected['ts'] = timestamps.loc[keep]
                        pieces.append(selected)
                    # Sealed corpus files are timestamp-sorted. Stop once this chunk reaches the
                    # exclusive upper bound instead of scanning unused later years.
                    if upper is not None and len(timestamps) and timestamps.iloc[-1] >= upper:
                        break
                if not pieces:
                    if verbose:
                        print(f"  [ssl-data] skip (empty bounded interval) {tk}_{tf}", flush=True)
                    continue
                df = pd.concat(pieces, ignore_index=True)
            if df['ts'].isna().any():
                raise ValueError(f"{path}: invalid timestamps")
            df = df.sort_values('ts').reset_index(drop=True)
            if df['ts'].duplicated().any():
                raise ValueError(f"{path}: duplicate timestamps")
            ohlcv = df[OHLCV_COLS].to_numpy(np.float32)
            if not np.isfinite(ohlcv).all():
                raise ValueError(f"{path}: non-finite OHLCV values")
            o, h, l, c, v = ohlcv.T
            invalid = (h < np.maximum(o, c)) | (l > np.minimum(o, c)) | (h < l) | (v < 0)
            if invalid.any():
                raise ValueError(f"{path}: {int(invalid.sum())} invalid OHLCV rows")
            ts = df['ts'].to_numpy()
            streams.append({'sid': f'{tk}@{tf}', 'ticker': tk, 'tf': tf,
                            'ohlcv': ohlcv, 'ts': ts,
                            'contract_id': (df['contract_id'].to_numpy()
                                            if 'contract_id' in df else None)})
            if verbose:
                interval = ("" if start is None and end is None else
                            f" interval=[{start or '-inf'},{end or '+inf'})")
                print(f"  [ssl-data] {tk}_{tf} bars={len(df)}{interval}", flush=True)
    if not streams:
        raise FileNotFoundError(f"no OHLCV CSVs found under {ddir} for "
                                f"tickers={tickers} tfs={tfs}")
    return streams


def _utc_timestamp(value):
    out = pd.Timestamp(value)
    return out.tz_localize('UTC') if out.tzinfo is None else out.tz_convert('UTC')


def time_split(ts, val_frac=0.1, holdout_start='2026-01-01', embargo=0, val_start=None,
               train_start=None):
    """Strictly causal split of one stream's timestamps into (train_idx, val_idx).

    HOLDOUT (>= holdout_start) is excluded from BOTH so the backbone never sees it.
    With ``val_start``, TRAIN is [train_start, val_start) when ``train_start`` is supplied
    (otherwise every earlier row), and VAL is [val_start, holdout_start). Otherwise VAL is the
    last ``val_frac`` of pre-holdout bars.  A declared lower bound is required for equal-history
    model tournaments; silently giving one backbone ten extra years is not a fair comparison.
    Returns (train_idx, val_idx) int arrays into ts.
    """
    ts = pd.DatetimeIndex(pd.to_datetime(ts, utc=True))
    embargo = max(0, int(embargo))
    if val_start is not None:
        val_cut = _utc_timestamp(val_start)
        train_cut = _utc_timestamp(train_start) if train_start is not None else None
        hold_cut = _utc_timestamp(holdout_start) if holdout_start is not None else None
        if train_cut is not None and train_cut >= val_cut:
            raise ValueError(f"train_start must precede val_start: {train_cut} >= {val_cut}")
        if hold_cut is not None and val_cut >= hold_cut:
            raise ValueError(f"val_start must precede holdout_start: {val_cut} >= {hold_cut}")
        tr_mask = ts < val_cut
        if train_cut is not None:
            tr_mask &= ts >= train_cut
        tr = np.flatnonzero(tr_mask)
        va_mask = ts >= val_cut
        if hold_cut is not None:
            va_mask &= ts < hold_cut
        va = np.flatnonzero(va_mask)
        if embargo:
            tr = tr[:-embargo] if len(tr) > embargo else np.array([], dtype=int)
            # The validation tail is purged whenever a declared OOS boundary follows, even if
            # the current source snapshot does not yet contain rows beyond that future boundary.
            if hold_cut is not None:
                va = va[:-embargo] if len(va) > embargo else np.array([], dtype=int)
        return tr.astype(int), va.astype(int)

    n = len(ts)
    usable = np.arange(n)
    if holdout_start is not None:
        cut = _utc_timestamp(holdout_start)
        usable = usable[ts[usable] < cut]
    if len(usable) == 0:
        return np.array([], int), np.array([], int)
    n_val = int(len(usable) * val_frac)
    if n_val == 0:
        return usable, np.array([], int)
    split = len(usable) - n_val
    tr_end = max(0, split - embargo)
    # When a real holdout follows, also purge the validation tail. The windows already cannot
    # cross the boundary; this additional gap prevents near-boundary dependence from becoming
    # an optimistic checkpoint-selection signal.
    has_holdout = bool(holdout_start is not None and len(usable) < n)
    va_end = max(split, len(usable) - embargo) if has_holdout else len(usable)
    return usable[:tr_end], usable[split:va_end]


def window_starts(idx, seq_total, contiguous=True, *, timestamps=None,
                  expected_delta=None, max_gap=None, segment_ids=None):
    """Valid window-start positions within an index range such that
    [start, start+seq_total) stays inside `idx`. With contiguous=True (default) the
    full window must be a run of consecutive bar indices (no split/holdout gap inside
    the window). Returns an int array of start positions (absolute bar indices)."""
    idx = np.asarray(idx, int)
    if len(idx) < seq_total:
        return np.array([], int)
    if not contiguous:
        return idx[:len(idx) - seq_total + 1]
    # keep starts whose next seq_total-1 indices are consecutive (idx[i+k] == idx[i]+k)
    starts = idx[:len(idx) - seq_total + 1]
    ahead = idx[seq_total - 1:]                       # idx shifted by seq_total-1
    valid = (ahead - starts) == (seq_total - 1)

    # Array-index contiguity alone is insufficient: adjacent rows may straddle a weekend,
    # maintenance break, missing-data hole, or front-contract roll. Mark every invalid edge in
    # the full stream and reject a candidate whose [start, end] interval contains one.
    if timestamps is not None or segment_ids is not None:
        n = len(timestamps) if timestamps is not None else len(segment_ids)
        bad_edge = np.zeros(max(0, n - 1), dtype=bool)
        if timestamps is not None and expected_delta is not None and n > 1:
            t = np.asarray(pd.to_datetime(timestamps, utc=True), dtype='datetime64[ns]')
            delta_ns = int(pd.Timedelta(expected_delta).value)
            observed = np.diff(t).astype('timedelta64[ns]').astype(np.int64)
            if max_gap is None:
                bad_edge |= observed != delta_ns
            else:
                max_gap_ns = int(pd.Timedelta(max_gap).value)
                if max_gap_ns < delta_ns:
                    raise ValueError("max_gap must be >= expected_delta")
                # Coarser futures contexts need to cross the known one-hour daily maintenance
                # closure. Keep only aligned intervals under that explicit ceiling; weekends,
                # holidays, off-grid timestamps, and larger missing-data holes remain boundaries.
                bad_edge |= ((observed < delta_ns) | (observed > max_gap_ns)
                             | ((observed % delta_ns) != 0))
        if segment_ids is not None and n > 1:
            seg = np.asarray(segment_ids)
            bad_edge |= seg[1:] != seg[:-1]
        prefix = np.zeros(n, dtype=np.int64)
        if n > 1:
            prefix[1:] = np.cumsum(bad_edge)
        valid &= (prefix[ahead] - prefix[starts]) == 0
    return starts[valid]
