"""Phase-0 probe — is regime/vol/structure/range knowledge present in
frozen Chronos-Bolt embeddings at all?

Throwaway experiment that gates whether `futures_foundation.context`
(XGBoost context heads on Bolt embeddings) deserves to be built. No
refactor required: runs on the `futures_foundation.foundation.embed`
subprocess seam, so the parent stays torch-free (macOS libomp constraint).

Five fixed forward-looking labels (the Step-2 redefinitions — all
computable from close-only series, matching Bolt's close-only context):

  fwd_return     reg  20-bar fwd log-return, z-scored by trailing 200-bar
                      std of 20-bar returns, clipped +/-4
  vol_expansion  clf  fwd 20-bar realized vol > 1.5x trailing 200-bar
                      median of 20-bar realized vol
  volatility     reg  fwd 10-bar realized-vol percentile vs the trailing
                      100 bars' 10-bar vols, continuous [0,1]
  structure      clf  fwd 20-bar close max/min vs trailing 12-bar close
                      max/min: both higher = bull(1), both lower = bear(0),
                      mixed = NaN (dropped)
  range_pos      reg  close at t+10 position within the trailing 20-bar
                      close range, clipped [0,1]

Pre-registered gates (probe passes if it clears the gate on the
PRE-CUTOFF validation slice): regression Pearson r > 0.05, classification
AUC > 0.55. Controls, fit identically per head:
  SHUFFLE — labels permuted; must NOT clear the gate (else leak/bug).
  TRIVIAL — 8 trailing summary stats instead of the embedding; tells us
            whether Bolt knows more than trivial features.

Leak discipline: everything here uses bars < HEADS_CUTOFF (2023-01-01);
rows whose forward window crosses the cutoff are dropped. Train < VAL_START,
val = [VAL_START, cutoff). Downstream signal-pipeline folds (2023+) are
never touched.

Usage:
  python3 scripts/probe_context_heads.py --smoke     # ES 3min, ~minutes
  python3 scripts/probe_context_heads.py             # 6 tickers x 2 TFs
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from futures_foundation import foundation as backbone  # noqa: E402  (torch-free parent)
from futures_foundation.context import (  # noqa: E402
    compute_context_labels as compute_labels,
    HEAD_SPECS, CANDIDATE_HEAD_SPECS,
    GATE_REG_PEARSON, GATE_CLF_AUC, HEADS_CUTOFF,
)

# probe evaluates established heads AND candidates; candidates are promoted
# into HEAD_SPECS only if they beat their gate AND the trivial baseline.
HEADS = HEAD_SPECS + CANDIDATE_HEAD_SPECS

# unconditional labels every decision bar must have (conditional labels
# like quiet_persist are NaN by design on most bars — never row-filtered).
REQUIRED_LABELS = ['vol_expansion', 'volatility']

CTX = 128                    # bars per Bolt context (matches *_chronos labelers)
VAL_START = pd.Timestamp('2022-11-01', tz='UTC')
TICKERS = ['ES', 'NQ', 'RTY', 'YM', 'GC', 'SI']
TFS = ['3min', '5min']
EMBED_CHUNK = 50_000


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


def build_dataset(tickers, tfs, stride, ff68=False):
    """Per (ticker, tf): sample decision bars pre-cutoff, return contexts,
    labels, trivial features, timestamps."""
    ctxs, labs, trivs, tss, ffs = [], [], [], [], []
    for tk in tickers:
        for tf in tfs:
            path = ROOT / 'data' / f'{tk}_{tf}.csv'
            if not path.exists():
                print(f"  [skip] {path.name} not found")
                continue
            usecols = None if ff68 else ['datetime', 'close']
            df = pd.read_csv(path, usecols=usecols)
            df['ts'] = pd.to_datetime(df['datetime'], utc=True)
            df = df.sort_values('ts').reset_index(drop=True)
            close = df['close'].astype(float)
            ts = df['ts']

            lab = compute_labels(close)
            triv = trivial_features(close)
            lp = np.log(close.to_numpy(np.float64))

            # decision bars: full context + trailing windows available,
            # strictly pre-cutoff INCLUDING the 20-bar forward window
            ts20 = ts.shift(-20)
            ok = (np.arange(len(df)) >= max(CTX, 200)) \
                & ts20.notna().to_numpy() \
                & (ts20 < HEADS_CUTOFF).to_numpy() \
                & lab[REQUIRED_LABELS].notna().all(axis=1).to_numpy()
            idx = np.flatnonzero(ok)[::stride]
            if not len(idx):
                print(f"  [skip] {tk}_{tf}: no pre-cutoff rows")
                continue
            ctxs.append(np.stack([lp[i - CTX + 1:i + 1] for i in idx])
                        .astype(np.float32))
            labs.append(lab.iloc[idx].reset_index(drop=True))
            trivs.append(triv.iloc[idx].reset_index(drop=True))
            tss.append(ts.iloc[idx].reset_index(drop=True))
            if ff68:
                from futures_foundation.features import (
                    derive_features, get_model_feature_columns)
                t0 = time.time()
                fdf = derive_features(df, tk)
                cols = [c for c in get_model_feature_columns()
                        if c in fdf.columns]
                ffs.append(fdf[cols].iloc[idx].reset_index(drop=True)
                           .to_numpy(np.float32))
                print(f"    [ff68] {len(cols)} features "
                      f"({time.time() - t0:.0f}s)")
            print(f"  [data] {tk}_{tf}: {len(idx):,} decision bars "
                  f"({ts.iloc[idx[0]].date()} -> {ts.iloc[idx[-1]].date()})")
    if not ctxs:
        raise SystemExit("no data — nothing to probe")
    return (np.concatenate(ctxs),
            pd.concat(labs, ignore_index=True),
            pd.concat(trivs, ignore_index=True).to_numpy(np.float32),
            pd.concat(tss, ignore_index=True),
            np.concatenate(ffs) if ffs else None)


def embed_chunked(contexts):
    parts = []
    for s in range(0, len(contexts), EMBED_CHUNK):
        chunk = contexts[s:s + EMBED_CHUNK]
        t0 = time.time()
        parts.append(backbone.embed(chunk))
        print(f"  [embed] {s + len(chunk):,}/{len(contexts):,} "
              f"({time.time() - t0:.0f}s)")
    return np.concatenate(parts)


def _fit_probe(kind, X, y, seed, n_estimators):
    import xgboost as xgb
    common = dict(n_estimators=n_estimators, max_depth=5, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, tree_method='hist',
                  random_state=seed, n_jobs=0)
    if kind == 'reg':
        return xgb.XGBRegressor(objective='reg:squarederror', **common).fit(X, y)
    return xgb.XGBClassifier(objective='binary:logistic',
                             eval_metric='logloss', **common).fit(X, y)


def _score(kind, model, X, y):
    if kind == 'reg':
        p = model.predict(X)
        if p.std() == 0 or y.std() == 0:
            return 0.0
        return float(np.corrcoef(p, y)[0, 1])
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y)) < 2:
        return 0.5
    return float(roc_auc_score(y, model.predict_proba(X)[:, 1]))


def run_probes(E, T, labels, ts, seed, n_estimators, F=None):
    tr = (ts < VAL_START).to_numpy()
    va = ((ts >= VAL_START) & (ts < HEADS_CUTOFF)).to_numpy()
    print(f"\n[split] train={tr.sum():,}  val={va.sum():,}  "
          f"(val = {VAL_START.date()} .. {HEADS_CUTOFF.date()})")
    rng = np.random.default_rng(seed)
    results = {}
    for name, kind in HEADS:
        y = labels[name].to_numpy(np.float32)
        m_tr = tr & ~np.isnan(y)
        m_va = va & ~np.isnan(y)
        if m_tr.sum() < 500 or m_va.sum() < 100:
            print(f"  [{name}] skipped — too few rows "
                  f"(tr={m_tr.sum()}, va={m_va.sum()})")
            continue
        ytr, yva = y[m_tr], y[m_va]
        gate = GATE_REG_PEARSON if kind == 'reg' else GATE_CLF_AUC
        metric = 'pearson_r' if kind == 'reg' else 'auc'

        t0 = time.time()
        emb = _score(kind, _fit_probe(kind, E[m_tr], ytr, seed,
                                      n_estimators), E[m_va], yva)
        # COMBINED arm: embedding + trailing stats — the enriched-head
        # candidate. Ships only if it beats BOTH single-source arms.
        ET_tr = np.hstack([E[m_tr], T[m_tr]])
        ET_va = np.hstack([E[m_va], T[m_va]])
        comb = _score(kind, _fit_probe(kind, ET_tr, ytr, seed,
                                       n_estimators), ET_va, yva)
        ff = embff = None
        if F is not None:
            Fc = np.nan_to_num(F, copy=True)
            ff = _score(kind, _fit_probe(kind, Fc[m_tr], ytr, seed,
                                         n_estimators), Fc[m_va], yva)
            EF_tr = np.hstack([E[m_tr], Fc[m_tr]])
            EF_va = np.hstack([E[m_va], Fc[m_va]])
            embff = _score(kind, _fit_probe(kind, EF_tr, ytr, seed,
                                            n_estimators), EF_va, yva)
        ysh = rng.permutation(ytr)
        shuf = _score(kind, _fit_probe(kind, E[m_tr], ysh, seed,
                                       n_estimators), E[m_va], yva)
        triv = _score(kind, _fit_probe(kind, T[m_tr], ytr, seed,
                                       n_estimators), T[m_va], yva)
        passed = emb > gate
        results[name] = dict(kind=kind, metric=metric, emb=emb, shuffle=shuf,
                             trivial=triv, combined=comb, ff68=ff,
                             emb_ff68=embff, gate=gate,
                             passed=bool(passed),
                             n_train=int(m_tr.sum()), n_val=int(m_va.sum()))
        flag = '✅ PASS' if passed else '❌ FAIL'
        beats_triv = '> trivial' if emb > triv else '<= trivial'
        both = '✚ COMBINED WINS' if comb > max(emb, triv) else ''
        if ff is not None:
            both += f"  FF68={ff:+.3f}  EMB+FF68={embff:+.3f}" 
        print(f"  [{name:<13}] {metric}: EMB={emb:+.3f}  SHUFFLE={shuf:+.3f} "
              f" TRIVIAL={triv:+.3f}  COMB={comb:+.3f}  gate>{gate}  {flag}  "
              f"({beats_triv} {both}, "
              f"ntr={m_tr.sum():,} nva={m_va.sum():,}, "
              f"{time.time() - t0:.0f}s)")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true',
                    help='ES 3min only, stride 64, 50 trees')
    ap.add_argument('--tickers', nargs='*', default=None)
    ap.add_argument('--tfs', nargs='*', default=None)
    ap.add_argument('--stride', type=int, default=8)
    ap.add_argument('--trees', type=int, default=400)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--ff68', action='store_true',
                    help='add the 68-feature library as head-input arms')
    ap.add_argument('--out', default=None,
                    help='JSON results path (default temp/probe_context_heads_<mode>.json)')
    a = ap.parse_args()

    tickers = a.tickers or (['ES'] if a.smoke else TICKERS)
    tfs = a.tfs or (['3min'] if a.smoke else TFS)
    stride = 64 if a.smoke and a.stride == 8 else a.stride
    trees = 50 if a.smoke and a.trees == 400 else a.trees

    backbone.stamp_active_source(context='context-heads probe (phase 0)')
    print(f"[probe] tickers={tickers} tfs={tfs} stride={stride} "
          f"trees={trees} cutoff={HEADS_CUTOFF.date()}")

    C, labels, T, ts, F = build_dataset(tickers, tfs, stride, ff68=a.ff68)
    print(f"[probe] total decision bars: {len(C):,}")
    E = embed_chunked(C)
    assert E.shape[0] == len(C), E.shape   # d_model varies by backbone
    np.nan_to_num(T, copy=False)   # XGBoost handles NaN, but keep parity

    results = run_probes(E, T, labels, ts, a.seed, trees, F=F)

    n_pass = sum(r['passed'] for r in results.values())
    print(f"\n{'=' * 64}\n🚦 PROBE VERDICT: {n_pass}/{len(results)} heads "
          f"clear their pre-registered gate\n{'=' * 64}")
    if n_pass == 0:
        print("   -> Knowledge NOT recoverable from frozen Bolt embeddings "
              "at these gates.\n   -> Per plan: context heads do not get "
              "built; refactor proceeds without futures_foundation.context.")
    else:
        for name, r in results.items():
            if r['passed']:
                print(f"   {name}: {r['metric']}={r['emb']:+.3f} "
                      f"(shuffle {r['shuffle']:+.3f}, "
                      f"trivial {r['trivial']:+.3f})")
        print("   -> Heads with signal exist. Next gate: do they ADD to the "
              "signal pipeline (pre-registered A/B, >= +0.10R)?")

    out = a.out or str(ROOT / 'temp' /
                       f"probe_context_heads_{'smoke' if a.smoke else 'full'}.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    meta = dict(tickers=tickers, tfs=tfs, stride=stride, trees=trees,
                seed=a.seed, ctx=CTX, cutoff=str(HEADS_CUTOFF),
                val_start=str(VAL_START), backbone=backbone.active_source(),
                gates=dict(reg_pearson=GATE_REG_PEARSON, clf_auc=GATE_CLF_AUC),
                n_rows=int(len(C)), results=results)
    Path(out).write_text(json.dumps(meta, indent=2))
    print(f"\n[probe] results -> {out}")


if __name__ == '__main__':
    main()
