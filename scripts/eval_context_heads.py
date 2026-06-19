"""Run the overfit-driven context-head evaluation (futures_foundation.context_eval).

The SAME process the strategy selection heads use, applied to the foundation
context heads — and crucially the first eval that checks accuracy on the
>= HEADS_CUTOFF period where the heads actually feed downstream strategies:

  3-way train / validate / TEST  →  fit default  →  overfit-detect + auto-
  regularize  →  VAL→TEST generalization gate  →  beat shuffle + trivial.

Unlike scripts/probe_context_heads.py (pre-cutoff validation only), this builds
the post-cutoff TEST split too, so a head is judged where it's used.

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

from futures_foundation import foundation as backbone           # noqa: E402
from futures_foundation import context_eval as CE               # noqa: E402
from futures_foundation.context import (                        # noqa: E402
    compute_context_labels, HEAD_SPECS, HEADS_CUTOFF)

CTX = 128
EMBED_CHUNK = 20000
VAL_START = pd.Timestamp('2022-11-01', tz='UTC')     # mirror the probe split


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


def build_3way(tickers, tfs, stride):
    """Decision bars across the FULL span (pre- AND post-cutoff) so we get a
    real TEST split. Per-head label finiteness is filtered in run_context_eval."""
    ctxs, labs, trivs, tss = [], [], [], []
    for tk in tickers:
        for tf in tfs:
            path = ROOT / 'data' / f'{tk}_{tf}.csv'
            if not path.exists():
                print(f"  [skip] {path.name} not found")
                continue
            df = pd.read_csv(path, usecols=['datetime', 'close'])
            df['ts'] = pd.to_datetime(df['datetime'], utc=True)
            df = df.sort_values('ts').reset_index(drop=True)
            close = df['close'].astype(float)
            ts = df['ts']
            lab = compute_context_labels(close)
            triv = trivial_features(close)
            lp = np.log(close.to_numpy(np.float64))
            ts20 = ts.shift(-20)
            ok = (np.arange(len(df)) >= max(CTX, 200)) & ts20.notna().to_numpy()
            idx = np.flatnonzero(ok)[::stride]
            if not len(idx):
                continue
            ctxs.append(np.stack([lp[i - CTX + 1:i + 1] for i in idx])
                        .astype(np.float32))
            labs.append(lab.iloc[idx].reset_index(drop=True))
            trivs.append(triv.iloc[idx].reset_index(drop=True))
            tss.append(ts.iloc[idx].reset_index(drop=True))
            print(f"  [data] {tk}_{tf}: {len(idx):,} bars "
                  f"({ts.iloc[idx[0]].date()} -> {ts.iloc[idx[-1]].date()})")
    if not ctxs:
        raise SystemExit("no data — nothing to evaluate")
    return (np.concatenate(ctxs),
            pd.concat(labs, ignore_index=True),
            pd.concat(trivs, ignore_index=True).to_numpy(np.float32),
            pd.concat(tss, ignore_index=True))


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
    ap.add_argument('--stride', type=int, default=3)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    backbone.stamp_active_source(context='context-head eval')
    tickers = ['ES'] if a.smoke else ['ES', 'NQ', 'RTY', 'YM', 'GC', 'SI']
    tfs = ['3min'] if a.smoke else ['3min', '5min']
    stride = 20 if a.smoke else a.stride

    print(f"[eval] tickers={tickers} tfs={tfs} stride={stride} "
          f"cutoff={HEADS_CUTOFF.date()}")
    C, labels, T, ts = build_3way(tickers, tfs, stride)
    print(f"[eval] {len(C):,} decision bars; embedding...")
    E = embed_chunked(C)

    print(f"\n{'=' * 70}\n  CONTEXT-HEAD OVERFIT-DRIVEN EVAL "
          f"(3-way; TEST = >= {HEADS_CUTOFF.date()}, the used-period)\n{'=' * 70}")
    res = CE.run_context_eval(E, labels, ts, VAL_START, HEADS_CUTOFF,
                              seed=a.seed, T=T, specs=HEAD_SPECS)

    acc = [n for n, v in res.items() if v['accurate']]
    print(f"\n{'=' * 70}\n  ACCURATE + GENERALIZING (usable as model context): "
          f"{len(acc)}/{len(res)}\n   {acc or 'NONE'}\n{'=' * 70}")
    for n, v in res.items():
        if not v['accurate']:
            why = []
            if not v['has_skill']:
                why.append('below floor')
            if not v['generalizes']:
                why.append('does not generalize (val→test)')
            if not v['beats_trivial']:
                why.append('<= trivial')
            if not v['beats_shuffle']:
                why.append('<= shuffle')
            print(f"   ⚠️  {n}: {', '.join(why)}")


if __name__ == '__main__':
    main()
