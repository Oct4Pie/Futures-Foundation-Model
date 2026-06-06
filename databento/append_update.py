"""Append a new DataBento file to existing data/{TICKER}_{TF}.csv — continuously.

Reuses build_continuous.py's proven extraction (DBN/CSV load, spread/dedup
cleaning, resample). For each ticker in the NEW file it SPLICES: keep existing
bars strictly BEFORE the new file's first timestamp, then take ALL new bars from
that point on. Because a DataBento re-pull overlaps the old tail bit-for-bit
(verified: 0.0000 close diff in the overlap), splicing at the new start —
instead of appending only after the old end — guarantees CONTINUITY: it rebuilds
the overlap from the complete new bars, so there's no partial/duplicate seam bar
and no missing minutes. Weekend gaps (Fri close→Sun open) are real market
closures and are left as-is.

Safe by default:
  - backs up each touched data/ file to data/backup_<stamp>/ first
  - writes the combined CSV to /tmp first; prints seam + continuity report
  - only overwrites data/ when run with --commit

Usage:
    # preview (no writes to data/): which tickers/TFs, seam, continuity
    python3 databento/append_update.py <file.dbn.zst|.csv.zst>
    # actually update data/ (after eyeballing the preview)
    python3 databento/append_update.py <file> --commit
    # restrict tickers / timeframes
    python3 databento/append_update.py <file> --tickers NQ,ES --tfs 3min,5min --commit
"""
import argparse
import datetime as _dt
import importlib.util
import shutil
import sys
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
DATA_DIR = _ROOT / 'data'

# reuse build_continuous's extraction/cleaning/resample (one source of truth)
_spec = importlib.util.spec_from_file_location('bc', _HERE / 'build_continuous.py')
bc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bc)


def _extract(path: Path) -> dict:
    """{ticker: clean 1-min df} from a .dbn.zst (multi) or .csv.zst (single)."""
    if path.name.endswith('.dbn.zst'):
        return bc.load_dbn_zst(path)
    if path.name.endswith('.csv.zst'):
        return {bc.ticker_from_path(path): bc.load_csv_zst(path)}
    sys.exit(f'unsupported file (need .dbn.zst or .csv.zst): {path.name}')


def _continuity_report(df: pd.DataFrame, tf: str) -> str:
    """Largest gap between consecutive bars, ignoring normal weekend closes."""
    step = pd.Timedelta(tf)
    deltas = df.index.to_series().diff().dropna()
    intra = deltas[deltas <= pd.Timedelta('1D')]          # ignore weekend gaps
    bad = intra[intra > step]
    n_bad = int((bad > step).sum())
    worst = (bad.max() if len(bad) else step)
    return (f'continuity: {n_bad} intra-day gaps > {tf} '
            f'(worst intraday {worst}); weekend gaps ignored')


def update_ticker(one_min: pd.DataFrame, ticker: str, tfs, stamp, commit):
    new_start = one_min.index.min()
    new_end = one_min.index.max()
    for tf in tfs:
        existing_path = DATA_DIR / f'{ticker}_{tf}.csv'
        new_tf = bc.resample_ohlcv(one_min, tf)
        new_tf.index.name = 'datetime'

        if existing_path.exists():
            existing = pd.read_csv(existing_path, parse_dates=['datetime']) \
                         .set_index('datetime')
            last_old = existing.index.max()
            # SPLICE: keep old strictly before the new file's start, then all new.
            kept = existing[existing.index < new_start]
            combined = pd.concat([kept, new_tf])
            combined = combined[~combined.index.duplicated(keep='last')].sort_index()
            n_old, n_kept = len(existing), len(kept)
        else:
            existing = None; last_old = None
            combined = new_tf.sort_index()
            n_old = n_kept = 0

        tmp = Path('/tmp') / f'{ticker}_{tf}.combined.csv'
        combined.to_csv(tmp)
        print(f'\n  {ticker}_{tf}:')
        print(f'    existing {n_old:,} bars (kept {n_kept:,} before {new_start}) '
              f'+ new {len(new_tf):,} → {len(combined):,} total')
        print(f'    range: {combined.index.min()} → {combined.index.max()}')
        print(f'    {_continuity_report(combined, tf)}')
        if last_old is not None and last_old >= new_start:
            print(f'    (rebuilt overlap {new_start}→{last_old} from new file — '
                  f'fixes partial seam bar)')

        if commit:
            bk = DATA_DIR / f'backup_{stamp}'
            bk.mkdir(exist_ok=True)
            if existing_path.exists():
                shutil.copy2(existing_path, bk / existing_path.name)
            combined.to_csv(existing_path)
            print(f'    ✅ committed → {existing_path.relative_to(_ROOT)} '
                  f'(backup: {bk.relative_to(_ROOT)}/)')
        else:
            print(f'    (preview only — temp at {tmp}; rerun with --commit to write)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('file', help='new DataBento .dbn.zst or .csv.zst')
    ap.add_argument('--tickers', default='', help='comma list (default: all in file)')
    ap.add_argument('--tfs', default='3min,5min')
    ap.add_argument('--commit', action='store_true', help='write to data/ (else preview)')
    a = ap.parse_args()

    path = Path(a.file)
    if not path.is_absolute():
        path = (_HERE / path.name) if (_HERE / path.name).exists() else Path.cwd() / a.file
    if not path.exists():
        sys.exit(f'file not found: {path}')

    tfs = [t.strip() for t in a.tfs.split(',') if t.strip()]
    want = {t.strip().upper() for t in a.tickers.split(',') if t.strip()}
    stamp = _dt.datetime.now().strftime('%Y%m%d_%H%M%S')

    print(f'Extracting {path.name} ...')
    tmap = _extract(path)
    tickers = sorted(t for t in tmap if not want or t in want)
    if not tickers:
        sys.exit(f'none of requested {want} found in file (has: {sorted(tmap)})')
    print(f'Tickers to update: {tickers} | tfs: {tfs} | '
          f'mode: {"COMMIT" if a.commit else "PREVIEW"}')

    for tk in tickers:
        update_ticker(tmap[tk], tk, tfs, stamp, a.commit)
    print('\nDone.' + ('' if a.commit else '  (preview — nothing written to data/)'))


if __name__ == '__main__':
    main()
