"""Demo — does the trained ContextHeads bundle KNOW the market?

Sample regime model built from JUST the ctx_* context features (no raw
embedding, no strategy features), evaluated strictly OUT-OF-SAMPLE: all
data here is >= 2023-01-01, which the heads (trained pre-cutoff) have
never seen. Works with both enriched (emb+ff68, FFM 2.1) and legacy
emb-only bundles — context_at handles the input build.

Two proofs:
1. CALIBRATION — per head, bucket the head's per-candle prediction into
   deciles and show the REALIZED outcome per decile. Monotonic = the head
   genuinely tracks the market, OOS, years past its training window.
2. SAMPLE REGIME MODEL — a tiny XGBoost on the ctx vector predicts
   a 4-class intraday regime (volatile-expansion / trending-up /
   trending-down / rotational) defined from REALIZED forward outcomes.
   Compared against the majority-class baseline.

Usage:
  export CONTEXT_HEADS_BUNDLE=temp/context_heads/heads_<...>.joblib
  python3 scripts/demo_regime_model.py [--tickers ES NQ] [--stride 16]
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from futures_foundation import foundation                      # noqa: E402
from futures_foundation.context import (                       # noqa: E402
    ContextHeads, compute_context_labels, HEADS_CUTOFF)

CTX = 128
REGIMES = {0: 'volatile_expansion', 1: 'trending_up',
           2: 'trending_down', 3: 'rotational'}


def realized_regime(lab: pd.DataFrame) -> np.ndarray:
    """4-class regime from REALIZED forward outcomes: expansion dominates;
    else direction from realized structure; mixed structure = rotational."""
    r = np.full(len(lab), 3, float)                     # rotational default
    st = lab['structure'].to_numpy()
    r[st == 1.0] = 1                                    # trending up
    r[st == 0.0] = 2                                    # trending down
    r[lab['vol_expansion'].to_numpy() == 1.0] = 0       # volatile (override)
    r[lab['vol_expansion'].isna().to_numpy()] = np.nan
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tickers', nargs='*', default=['ES', 'NQ'])
    ap.add_argument('--tfs', nargs='*', default=['3min', '5min'])
    ap.add_argument('--stride', type=int, default=16)
    ap.add_argument('--bundle', default=None)
    a = ap.parse_args()

    bundle = a.bundle or os.environ.get('CONTEXT_HEADS_BUNDLE')
    if not bundle:
        cands = sorted((ROOT / 'temp' / 'context_heads').glob('heads_*.joblib'))
        cands = [c for c in cands if 'smoke' not in c.name]
        if not cands:
            raise SystemExit("no bundle — run scripts/train_context_heads.py")
        bundle = str(cands[-1])
    heads = ContextHeads.load(bundle)
    foundation.stamp_active_source(context='regime-model demo')
    print(f"[demo] heads bundle : {bundle}")
    print(f"[demo] active heads : {heads.active_names}")
    print(f"[demo] OOS window   : >= {HEADS_CUTOFF.date()} "
          f"(heads never saw these bars)\n")

    # ---- build OOS dataset: ctx features + realized labels ----------------
    X_parts, lab_parts, ts_parts = [], [], []
    for tk in a.tickers:
        for tf in a.tfs:
            path = ROOT / 'data' / f'{tk}_{tf}.csv'
            if not path.exists():
                continue
            df = pd.read_csv(path)          # full OHLCV — enriched heads
            df['ts'] = pd.to_datetime(df['datetime'], utc=True)
            df = df.sort_values('ts').reset_index(drop=True)
            close = df['close'].astype(float)
            lab = compute_context_labels(close)
            ok = (np.arange(len(df)) >= max(CTX, 200)) \
                & (df['ts'] >= HEADS_CUTOFF).to_numpy() \
                & lab['vol_expansion'].notna().to_numpy() \
                & lab['volatility'].notna().to_numpy()
            idx = np.flatnonzero(ok)[::a.stride]
            if not len(idx):
                continue
            ctx_df = heads.context_at(df, idx, tk)
            X_parts.append(ctx_df.to_numpy())
            lab_parts.append(lab.iloc[idx].reset_index(drop=True))
            ts_parts.append(df['ts'].iloc[idx].reset_index(drop=True))
            print(f"  [data] {tk}_{tf}: {len(idx):,} OOS bars "
                  f"({df['ts'].iloc[idx[0]].date()} -> "
                  f"{df['ts'].iloc[idx[-1]].date()})")
    X = np.concatenate(X_parts)
    lab = pd.concat(lab_parts, ignore_index=True)
    ts = pd.concat(ts_parts, ignore_index=True)
    names = heads.active_names
    print(f"\n[demo] {len(X):,} OOS bars, features: {names}")

    # ---- Proof 1: per-head OOS calibration --------------------------------
    checks = [('ctx_volatility', lab['volatility'], 'realized fwd-vol pctile'),
              ('ctx_vol_expansion', lab['vol_expansion'],
               'realized expansion rate'),
              ('ctx_structure', lab['structure'], 'realized bull-structure rate'),
              ('ctx_trendiness', lab['trendiness'],
               'realized fwd efficiency ratio'),
              ('ctx_range_bound', lab['range_bound'],
               'realized range-bound rate'),
              ('ctx_quiet_persist', lab['quiet_persist'],
               'realized quiet-persists rate (quiet bars only)')]
    print(f"\n{'=' * 64}\nPROOF 1 — OOS CALIBRATION (prediction decile -> "
          f"realized outcome)\n{'=' * 64}")
    for feat, realized, what in checks:
        if feat not in names:
            continue
        p = X[:, names.index(feat)]
        y = realized.to_numpy(float)
        m = ~np.isnan(y)
        dec = pd.qcut(p[m], 10, labels=False, duplicates='drop')
        tab = pd.Series(y[m]).groupby(dec).mean()
        mono = float(np.corrcoef(tab.index, tab.to_numpy())[0, 1])
        line = '  '.join(f"{v:.2f}" for v in tab.to_numpy())
        print(f"\n  {feat}  ->  {what}")
        print(f"    deciles 1..10: {line}")
        print(f"    monotonicity (decile vs realized corr): {mono:+.3f}  "
              f"{'✅' if mono > 0.9 else ('🟡' if mono > 0.6 else '❌')}")

    # ---- Proof 2: 4-feature regime model, train 2023-24, test 2025+ -------
    y = realized_regime(lab)
    m = ~np.isnan(y)
    split = pd.Timestamp('2025-01-01', tz='UTC')
    tr = m & (ts < split).to_numpy()
    te = m & (ts >= split).to_numpy()
    import xgboost as xgb
    clf = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                            tree_method='hist', random_state=0, n_jobs=0)
    clf.fit(X[tr], y[tr].astype(int))
    pred = clf.predict(X[te])
    yte = y[te].astype(int)
    acc = float((pred == yte).mean())
    base = float(np.bincount(yte).max() / len(yte))
    print(f"\n{'=' * 64}\nPROOF 2 — SAMPLE REGIME MODEL ({len(names)} ctx "
          f"features ONLY)\n{'=' * 64}")
    print(f"  train: 2023-2024 ({tr.sum():,} bars)   "
          f"test: 2025+ ({te.sum():,} bars)")
    print(f"  4-class accuracy : {acc:.1%}   vs majority baseline "
          f"{base:.1%}   (lift {acc - base:+.1%})")
    print(f"\n  per-class recall (test):")
    for k, name in REGIMES.items():
        mk = yte == k
        if mk.sum():
            rec = float((pred[mk] == k).mean())
            print(f"    {name:<20} n={mk.sum():>6,}  recall={rec:.1%}")
    verdict = (f'✅ the {len(names)} ctx features carry real regime '
               f'knowledge OOS'
               if acc - base > 0.05 else '❌ no material lift over baseline')
    print(f"\n  {verdict}  (pre-registered bar: lift > +5pts)")


if __name__ == '__main__':
    main()
