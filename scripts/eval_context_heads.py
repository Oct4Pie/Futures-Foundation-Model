"""Walk-forward, overfit-driven evaluation of the foundation CONTEXT heads.

The SAME process the strategy selection heads use — reusing walk_forward_folds
(rolling train/val/test across ALL data) + the shared overfit library — applied
to the context heads on the production enriched input [emb | 68 features]. Each
head is judged in EVERY regime, not just a single 2023+ slice.

  python3 scripts/eval_context_heads.py --smoke     # ES 3min, minutes
  python3 scripts/eval_context_heads.py             # full 6 tickers x 2 TFs
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from futures_foundation.extractors.chronos import backbone           # noqa: E402
from futures_foundation import context_eval as CE               # noqa: E402
from futures_foundation.context import (                        # noqa: E402
    compute_context_labels, HEAD_SPECS, MAX_LABEL_HORIZON)
from futures_foundation.features import (                       # noqa: E402
    derive_features, get_model_feature_columns)

CTX = 128
EMBED_CHUNK = 20000
TFS = ['3min', '5min']


def trivial_features(close: pd.Series) -> pd.DataFrame:
    """8 trailing summary stats — the 'does Bolt beat trivial?' baseline."""
    lc = np.log(close)
    r1 = lc.diff()
    rh = close.rolling(20).max()
    rl = close.rolling(20).min()
    width = (rh - rl).replace(0, np.nan)
    f = pd.DataFrame(index=close.index)
    f['ret_1'] = r1
    f['ret_5'] = lc.diff(5)
    f['ret_20'] = lc.diff(20)
    f['ret_60'] = lc.diff(60)
    f['vol_10'] = r1.rolling(10).std()
    f['vol_20'] = r1.rolling(20).std()
    f['range_pos_now'] = ((close - rl) / width).clip(0, 1)
    f['close_z_100'] = ((close - close.rolling(100).mean())
                        / close.rolling(100).std().replace(0, np.nan))
    return f


def build_dataset(tickers, tfs, stride):
    """Decision bars across the FULL span (all regimes). Returns contexts,
    labels, trivial features, ff68 features, ts, label-close ts, item ids."""
    ctxs, labs, trivs, ffs, tss, tends, items = [], [], [], [], [], [], []
    for tk in tickers:
        for tf in tfs:
            path = ROOT / 'data' / f'{tk}_{tf}.csv'
            if not path.exists():
                print(f"  [skip] {path.name} not found")
                continue
            df = pd.read_csv(path)
            df['ts'] = pd.to_datetime(df['datetime'], utc=True)
            df = df.sort_values('ts').reset_index(drop=True)
            close = df['close'].astype(float)
            ts = df['ts']
            lab = compute_context_labels(close)
            triv = trivial_features(close)
            lp = np.log(close.to_numpy(np.float64))
            t0 = time.time()
            fdf = derive_features(df, tk)
            cols = [c for c in get_model_feature_columns() if c in fdf.columns]
            ts_end = ts.shift(-MAX_LABEL_HORIZON)
            ok = (np.arange(len(df)) >= max(CTX, 200)) & ts_end.notna().to_numpy()
            idx = np.flatnonzero(ok)[::stride]
            if not len(idx):
                continue
            ctxs.append(np.stack([lp[i - CTX + 1:i + 1] for i in idx]).astype(np.float32))
            labs.append(lab.iloc[idx].reset_index(drop=True))
            trivs.append(triv.iloc[idx].reset_index(drop=True))
            ffs.append(fdf[cols].iloc[idx].reset_index(drop=True).to_numpy(np.float32))
            tss.append(ts.iloc[idx].reset_index(drop=True))
            tends.append(ts_end.iloc[idx].reset_index(drop=True))
            items += [f'{tk}_{tf}'] * len(idx)
            print(f"  [data] {tk}_{tf}: {len(idx):,} bars ({cols and len(cols)} ff68, "
                  f"{time.time() - t0:.0f}s)")
    if not ctxs:
        raise SystemExit("no data — nothing to evaluate")
    return (np.concatenate(ctxs),
            pd.concat(labs, ignore_index=True),
            pd.concat(trivs, ignore_index=True).to_numpy(np.float32),
            np.concatenate(ffs),
            pd.concat(tss, ignore_index=True),
            pd.concat(tends, ignore_index=True),
            np.asarray(items))


def embed_chunked(contexts):
    parts = []
    for s in range(0, len(contexts), EMBED_CHUNK):
        chunk = contexts[s:s + EMBED_CHUNK]
        t0 = time.time()
        parts.append(backbone.embed(chunk))
        print(f"  [embed] {s + len(chunk):,}/{len(contexts):,} "
              f"({time.time() - t0:.0f}s)")
    return np.concatenate(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true', help='ES 3min only, big stride')
    ap.add_argument('--tfs', nargs='*', default=None,
                    help='timeframes (default: 3min 5min; e.g. --tfs 3min)')
    ap.add_argument('--stride', type=int, default=4)
    ap.add_argument('--train-m', type=int, default=6)
    ap.add_argument('--val-m', type=int, default=2)
    ap.add_argument('--test-m', type=int, default=2)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    backbone.stamp_active_source(context='context-head eval')
    tickers = ['ES'] if a.smoke else ['ES', 'NQ', 'RTY', 'YM', 'GC', 'SI']
    tfs = a.tfs or (['3min'] if a.smoke else TFS)
    stride = 20 if a.smoke else a.stride

    print(f"[eval] tickers={tickers} tfs={tfs} stride={stride}")
    C, labels, T, F, ts, ts_end, items = build_dataset(tickers, tfs, stride)
    print(f"[eval] {len(C):,} decision bars; ff68={F.shape[1]}; embedding...")
    E = embed_chunked(C)
    X = np.hstack([E, F])                          # enriched [emb | ff68], production recipe

    print(f"\n{'=' * 70}\n  CONTEXT-HEAD WALK-FORWARD OVERFIT-DRIVEN EVAL "
          f"(enriched [emb|ff68])\n{'=' * 70}")
    res = CE.run_context_eval(X, labels, ts, ts_end, items, train_m=a.train_m,
                              val_m=a.val_m, test_m=a.test_m, seed=a.seed,
                              T=T, specs=HEAD_SPECS)

    acc = [n for n, v in res.items() if v.get('accurate')]
    print(f"\n{'=' * 70}\n  ACCURATE + GENERALIZING (usable as model context): "
          f"{len(acc)}/{len(res)}\n   {acc or 'NONE'}\n{'=' * 70}")
    for n, v in res.items():
        if not v.get('accurate') and v.get('n_folds'):
            why = []
            if not v['has_skill']:
                why.append(f"below floor (mean_test {v['mean_test']:+.3f})")
            if v['gen_frac'] < CE.MIN_GENERALIZE_FRAC:
                why.append(f"generalizes only {v['gen_frac']:.0%} of folds")
            if not v['beats_trivial']:
                why.append('<= trivial')
            if not v['beats_shuffle']:
                why.append('<= shuffle')
            print(f"   ⚠️  {n}: {', '.join(why)}")


if __name__ == '__main__':
    main()
