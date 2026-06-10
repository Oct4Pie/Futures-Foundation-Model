"""Train the production ContextHeads bundle (FFM 2.1 enriched inputs).

Heads consume [Bolt embedding | 68-feature library] per the FF68 probe
(temp/probe_ff68_full.json — emb+ff68 beat emb-only on every surviving
head). Reuses the Phase-0 probe's dataset machinery (same labels, same
decision-bar sampling, same split) but fits
`futures_foundation.context.ContextHeads` on the hstacked matrix and
saves a frozen joblib bundle with full metadata + the exact env line to
activate it downstream.

Leak discipline: trains ONLY on bars whose forward label window ends before
HEADS_CUTOFF (2023-01-01 UTC); validation = last 2 pre-cutoff months,
sanity/gate only. Downstream signal training that fuses these heads is
restricted to >= HEADS_CUTOFF by futures_foundation/chronos/context_fusion.

Usage:
  python3 scripts/train_context_heads.py --smoke    # ES 3min, minutes
  python3 scripts/train_context_heads.py            # full 6 tickers x 2 TFs
"""
import argparse
import datetime as _dt
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from futures_foundation.context import ContextHeads, HEADS_CUTOFF  # noqa: E402
from futures_foundation.features import get_model_feature_columns  # noqa: E402
from futures_foundation import foundation  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    'probe_context_heads', Path(__file__).parent / 'probe_context_heads.py')
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--tickers', nargs='*', default=None)
    ap.add_argument('--tfs', nargs='*', default=None)
    ap.add_argument('--stride', type=int, default=8)
    ap.add_argument('--trees', type=int, default=400)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    tickers = a.tickers or (['ES'] if a.smoke else probe.TICKERS)
    tfs = a.tfs or (['3min'] if a.smoke else probe.TFS)
    stride = 64 if a.smoke and a.stride == 8 else a.stride
    trees = 50 if a.smoke and a.trees == 400 else a.trees

    foundation.stamp_active_source(context='context-heads training')
    print(f"[train] tickers={tickers} tfs={tfs} stride={stride} "
          f"trees={trees} cutoff={HEADS_CUTOFF.date()}")

    C, labels, _T, ts, F = probe.build_dataset(tickers, tfs, stride,
                                               ff68=True)
    print(f"[train] decision bars: {len(C):,}  "
          f"ff68 features: {F.shape[1]}")
    E = probe.embed_chunked(C)
    X = np.hstack([E, F])                      # enriched: [emb | ff68]

    tr = (ts < probe.VAL_START).to_numpy()
    va = ((ts >= probe.VAL_START) & (ts < HEADS_CUTOFF)).to_numpy()
    print(f"[split] train={tr.sum():,}  val={va.sum():,}  "
          f"(val = {probe.VAL_START.date()} .. {HEADS_CUTOFF.date()})\n")

    heads = ContextHeads(seed=a.seed, n_estimators=trees)
    heads.fit(X[tr], labels[tr].reset_index(drop=True),
              X[va], labels[va].reset_index(drop=True))

    try:
        sha = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                             cwd=ROOT, capture_output=True,
                             text=True).stdout.strip()
    except Exception:
        sha = 'unknown'
    heads.meta = dict(
        inputs='emb+ff68', feature_cols=get_model_feature_columns(),
        cutoff=str(HEADS_CUTOFF), val_start=str(probe.VAL_START),
        train_span=(str(ts[tr].min()), str(ts[tr].max())),
        tickers=tickers, tfs=tfs, stride=stride, ctx=probe.CTX,
        d_model=foundation.D_MODEL, backbone=foundation.active_source(),
        n_rows=int(len(C)), git_sha=sha,
        train_date=_dt.date.today().isoformat(), seed=a.seed,
        n_estimators=trees)

    date = _dt.date.today().strftime('%Y%m%d')
    out = a.out or str(ROOT / 'temp' / 'context_heads'
                       / f"heads_{date}_{sha}{'_smoke' if a.smoke else ''}.joblib")
    path = heads.save(out)

    print(f"\n[train] active heads: {heads.active_names or 'NONE'}")
    print(f"[train] input_dim: {heads.input_dim} "
          f"(= {foundation.D_MODEL} emb + {F.shape[1]} ff68)")
    print(f"[train] bundle -> {path}")
    print("\nEnriched (emb+ff68) bundle — consume via "
          "ContextHeads.context_at(df, idx, instrument).")
    print("NOTE: the chronos event-fusion seam (fuse()) only has "
          "embeddings and will REFUSE this bundle.")


if __name__ == '__main__':
    main()
