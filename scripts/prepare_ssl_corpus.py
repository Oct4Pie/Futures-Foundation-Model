#!/usr/bin/env python3
"""Build a sealed, leak-auditable OHLCV corpus from daily root-contract parquet bars.

The source layout is expected to be::

    <source>/F.US.ES/1m/YYYY-MM-DD.parquet

Only explicitly requested roots are read. Archive/quarantine siblings are never scanned. Bars
are resampled independently inside each contract with left-closed/left-labeled UTC buckets and
no forward fill. The emitted ``contract_id`` lets SSL window assembly reject rollover-crossing
windows. A content-hash manifest seals the mutable upstream snapshot used for training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOTS = ('ES', 'NQ', 'RTY', 'YM', 'GC', 'SI', 'CL', 'ZB', 'ZN')
TIMEFRAMES = (1, 3, 5, 15, 30, 60)
REQUIRED = ('timestamp', 'open', 'high', 'low', 'close', 'volume', 'contract_id')
OUT_COLS = ('datetime', 'open', 'high', 'low', 'close', 'volume', 'contract_id')


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def _validate(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    missing = set(REQUIRED) - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    out = frame[list(REQUIRED)].copy()
    out['timestamp'] = pd.to_datetime(out['timestamp'], utc=True, errors='coerce')
    if out['timestamp'].isna().any():
        raise ValueError(f"{path}: invalid timestamps")
    if not out['timestamp'].is_monotonic_increasing:
        raise ValueError(f"{path}: timestamps are not monotonic")
    if out['timestamp'].duplicated().any():
        raise ValueError(f"{path}: duplicate timestamps")
    values = out[['open', 'high', 'low', 'close', 'volume']].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: non-finite OHLCV values")
    o, h, l, c, v = values.T
    bad = (h < np.maximum(o, c)) | (l > np.minimum(o, c)) | (h < l) | (v < 0)
    if bad.any():
        raise ValueError(f"{path}: {int(bad.sum())} invalid OHLCV rows")
    if out['contract_id'].isna().any() or (out['contract_id'].astype(str).str.len() == 0).any():
        raise ValueError(f"{path}: missing contract_id")
    return out


def _resample(frame: pd.DataFrame, minutes: int) -> pd.DataFrame:
    if minutes == 1:
        out = frame.copy()
    else:
        pieces = []
        rule = f'{minutes}min'
        for contract, part in frame.groupby('contract_id', sort=False):
            x = part.set_index('timestamp')
            agg = x.resample(rule, origin='epoch', closed='left', label='left').agg({
                'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum',
            })
            agg = agg.dropna(subset=['open', 'high', 'low', 'close']).reset_index()
            agg['contract_id'] = contract
            pieces.append(agg)
        out = (pd.concat(pieces, ignore_index=True).sort_values('timestamp')
               if pieces else pd.DataFrame(columns=REQUIRED))
    if out['timestamp'].duplicated().any():
        raise ValueError(f"resampling produced duplicate {minutes}-minute timestamps")
    out = out.rename(columns={'timestamp': 'datetime'})
    return out[list(OUT_COLS)]


def build_corpus(source: str | Path, output: str | Path, *, roots=ROOTS,
                 timeframes=TIMEFRAMES, overwrite=False, verbose=True) -> dict:
    source, output = Path(source).resolve(), Path(output).resolve()
    roots = tuple(str(x).upper() for x in roots)
    timeframes = tuple(sorted(set(int(x) for x in timeframes)))
    if not source.is_dir():
        raise FileNotFoundError(f"source directory does not exist: {source}")
    if any(x <= 0 for x in timeframes):
        raise ValueError("timeframes must be positive minute counts")
    output.mkdir(parents=True, exist_ok=True)

    final_paths = {(root, tf): output / f'{root}_{tf}min.csv'
                   for root in roots for tf in timeframes}
    existing = [str(p) for p in final_paths.values() if p.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"refusing to overwrite {len(existing)} corpus files; use --overwrite")
    tmp_paths = {key: path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
                 for key, path in final_paths.items()}
    for path in tmp_paths.values():
        path.unlink(missing_ok=True)

    source_digest = hashlib.sha256()
    root_reports = {}
    try:
        for root in roots:
            src_dir = source / f'F.US.{root}' / '1m'
            files = sorted(src_dir.glob('*.parquet'))
            if not files:
                raise FileNotFoundError(f"no daily parquet files found for {root}: {src_dir}")
            rows_in = 0
            rows_out = {tf: 0 for tf in timeframes}
            first_ts = last_ts = prev_ts = prev_contract = None
            time_gaps = contract_changes = 0
            for i, path in enumerate(files):
                rel = path.relative_to(source).as_posix()
                digest = _sha256(path)
                source_digest.update(rel.encode() + b'\0' + digest.encode() + b'\n')
                frame = _validate(pd.read_parquet(path), path)
                if len(frame) == 0:
                    continue
                cur_first = frame['timestamp'].iloc[0]
                cur_last = frame['timestamp'].iloc[-1]
                if prev_ts is not None and cur_first <= prev_ts:
                    raise ValueError(f"{path}: overlaps or reverses previous file ending {prev_ts}")
                ts_ns = frame['timestamp'].astype('int64').to_numpy()
                contracts = frame['contract_id'].astype(str).to_numpy()
                if prev_ts is not None:
                    time_gaps += int(cur_first - prev_ts != pd.Timedelta('1min'))
                    contract_changes += int(contracts[0] != prev_contract)
                if len(frame) > 1:
                    time_gaps += int((np.diff(ts_ns) != pd.Timedelta('1min').value).sum())
                    contract_changes += int((contracts[1:] != contracts[:-1]).sum())
                rows_in += len(frame)
                first_ts = cur_first if first_ts is None else first_ts
                last_ts = prev_ts = cur_last
                prev_contract = contracts[-1]
                for tf in timeframes:
                    bars = _resample(frame, tf)
                    bars.to_csv(tmp_paths[(root, tf)], mode='a', header=(rows_out[tf] == 0),
                                index=False)
                    rows_out[tf] += len(bars)
                if verbose and ((i + 1) % 500 == 0 or i + 1 == len(files)):
                    print(f"[{root}] {i + 1:,}/{len(files):,} files, {rows_in:,} source rows",
                          flush=True)
            root_reports[root] = {
                'files': len(files), 'source_rows': int(rows_in),
                'first_timestamp': first_ts.isoformat(), 'last_timestamp': last_ts.isoformat(),
                'one_minute_gap_edges': int(time_gaps),
                'contract_change_edges': int(contract_changes),
                'output_rows': {f'{tf}min': int(rows_out[tf]) for tf in timeframes},
            }

        outputs = {}
        for key, tmp in tmp_paths.items():
            root, tf = key
            final = final_paths[key]
            os.replace(tmp, final)
            outputs[f'{root}_{tf}min'] = {
                'path': str(final), 'bytes': final.stat().st_size, 'sha256': _sha256(final),
                'rows': root_reports[root]['output_rows'][f'{tf}min'],
            }
        manifest = {
            'schema_version': 'ffm_ssl_corpus_v1',
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'purpose': 'self-supervised OHLCV only; no labels or outcomes read',
            'source_root': str(source),
            'source_snapshot_sha256': source_digest.hexdigest(),
            'roots': list(roots), 'timeframes_minutes': list(timeframes),
            'resample': {'closed': 'left', 'label': 'left', 'origin': 'epoch',
                         'forward_fill': False, 'within_contract_only': True},
            'roots_report': root_reports, 'outputs': outputs,
        }
        manifest_path = output / 'MANIFEST.json'
        tmp_manifest = output / f'MANIFEST.json.{os.getpid()}.tmp'
        tmp_manifest.write_text(json.dumps(manifest, indent=2) + '\n')
        os.replace(tmp_manifest, manifest_path)
        if verbose:
            print(f"sealed corpus manifest -> {manifest_path}", flush=True)
        return manifest
    except Exception:
        for path in tmp_paths.values():
            path.unlink(missing_ok=True)
        raise


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True, help='root containing F.US.<ROOT>/1m daily parquet')
    p.add_argument('--output', required=True)
    p.add_argument('--roots', default=','.join(ROOTS))
    p.add_argument('--timeframes', default=','.join(map(str, TIMEFRAMES)))
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()
    build_corpus(args.source, args.output, roots=args.roots.split(','),
                 timeframes=tuple(int(x) for x in args.timeframes.split(',')),
                 overwrite=args.overwrite)


if __name__ == '__main__':
    main()
