#!/usr/bin/env python3
"""Leak-audited SuperTrend checkpoint benchmark derived from the author's gist.

Development mode runs fixed checkpoint x head configurations through purged rolling
walk-forward and excludes the declared holdout. OOS mode is deliberately gated: it
trains on pre-holdout data and evaluates the holdout once, after finalists are frozen.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from futures_foundation import indicators as I


ROOTS = ('ES', 'NQ', 'RTY', 'YM', 'GC', 'SI', 'CL', 'ZB', 'ZN')
AUTHOR_TFS = ('1min', '3min', '5min', '15min')
FIXED_TARGETS = (2.0, 3.0, 4.0, 6.0)
OOS_CONFIRMATION = 'RUN-2026-ONE-SHOT'


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=float) + '\n')
    os.replace(tmp, path)


@dataclass(frozen=True)
class Arm:
    name: str
    version: str
    checkpoint: Path

    @property
    def model_id(self) -> str:
        return 'paris-noah/MantisV2' if self.version == 'v2' else 'paris-noah/Mantis-8M'


def _parse_arm(text: str) -> Arm:
    try:
        name, version, raw_path = text.split(',', 2)
    except ValueError as exc:
        raise argparse.ArgumentTypeError('arm must be NAME,v1|v2,CHECKPOINT') from exc
    if version not in {'v1', 'v2'}:
        raise argparse.ArgumentTypeError(f'arm version must be v1 or v2, got {version!r}')
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file() or path.stat().st_size < 1024:
        raise argparse.ArgumentTypeError(f'checkpoint missing or only an LFS pointer: {path}')
    return Arm(name=name, version=version, checkpoint=path)


def _parse_admission(text: str):
    try:
        name, raw_path = text.split('=', 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError('admission must be ARM_NAME=/path/report.json') from exc
    path = Path(raw_path).expanduser().resolve()
    if not name or not path.is_file():
        raise argparse.ArgumentTypeError(f'invalid admission mapping: {text}')
    return name, path


def _verify_arm_admissions(arms, mappings):
    from futures_foundation.finetune.native_contracts import verify_admission_report
    supplied = dict(mappings)
    names = [arm.name for arm in arms]
    if len(supplied) != len(mappings):
        raise ValueError('admission arm names must be unique')
    if set(supplied) != set(names):
        raise ValueError(
            f'admission mappings must match arms exactly: missing={sorted(set(names) - set(supplied))}, '
            f'unknown={sorted(set(supplied) - set(names))}'
        )
    verified = {}
    for arm in arms:
        arm_key = 'mantis_v2' if arm.version == 'v2' else 'mantis_v1'
        verified[arm.name] = verify_admission_report(
            supplied[arm.name], arm_key=arm_key, track='B',
            route='supervised_barrier_experimental_task', require_training=False,
            required_artifacts={'checkpoint': arm.checkpoint},
        )
    return verified


class SuperTrendMantis:
    """Embedding-only SuperTrend(10,3) flip selector matching the linked gist."""

    n_classes = 2
    MV_MODE = 'ohlcv'
    MV_SEQ = 64
    ATR_P = 20
    CTX = 128
    VERT = 120
    STOP_ATR = 0.5
    COST_R = 0.03
    MIN_GAP = 20
    PRIMARY_R = 3.0

    def __init__(self, data_dir: Path, tickers, tfs, data_months=72):
        self.data_dir = Path(data_dir)
        self.data_months = int(data_months)
        self._b = {(tk, tf): self._bars(tk, tf) for tf in tfs for tk in tickers}
        self.TRAIN_SCOPE = {'tickers': list(tickers), 'timeframes': list(tfs)}

    def _bars(self, tk, tf):
        path = self.data_dir / f'{tk}_{tf}.csv'
        names = set(pd.read_csv(path, nrows=0).columns)
        contract_col = 'contract_id' if 'contract_id' in names else 'instrument_id'
        required = ['datetime', 'open', 'high', 'low', 'close', 'volume', contract_col]
        if contract_col not in names:
            raise ValueError(f'{path} lacks contract_id/instrument_id; strict roll-safe benchmark '
                             'refuses continuous futures without contract boundaries')
        df = pd.read_csv(path, usecols=required)
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
        if not df['datetime'].is_monotonic_increasing:
            df = df.sort_values('datetime')
        cutoff = df['datetime'].max() - pd.DateOffset(months=self.data_months)
        df = df[df['datetime'] >= cutoff].drop_duplicates('datetime', keep='last').reset_index(drop=True)
        ts = df['datetime'].to_numpy()
        o, h, l, c = (df[k].to_numpy(float) for k in ('open', 'high', 'low', 'close'))
        v = df['volume'].to_numpy(float)
        contract = df[contract_col].astype(str).to_numpy()
        segment = np.r_[0, np.cumsum(contract[1:] != contract[:-1])].astype(np.int32)
        atr = I.compute_atr(h, l, c, self.ATR_P)
        st, st_line, _ = I.compute_supertrend(h, l, c, 10, 3.0)
        return dict(ts=ts, o=o, h=h, l=l, c=c, v=v, atr=atr, st=st, st_line=st_line,
                    contract=contract, segment=segment)

    def calendar(self):
        return pd.concat([
            pd.DataFrame({'item_id': f'{tk}@{tf}', 'timestamp': b['ts'], 'target': b['c']})
            for (tk, tf), b in self._b.items()
        ], ignore_index=True)

    @staticmethod
    def _utc(value):
        ts = pd.Timestamp(value)
        return ts.tz_localize('UTC') if ts.tzinfo is None else ts.tz_convert('UTC')

    def _signals(self, b, lo, hi):
        lo, hi = self._utc(lo), self._utc(hi)
        ts, st = b['ts'], b['st']
        last_long = last_short = -self.MIN_GAP
        for i in range(1, len(st) - 1):
            if not (lo <= ts[i] < hi) or st[i] == st[i - 1]:
                continue
            direction = int(st[i])
            if direction == 1 and i - last_long >= self.MIN_GAP:
                last_long = i
                yield i, 1
            elif direction == -1 and i - last_short >= self.MIN_GAP:
                last_short = i
                yield i, -1

    def _fixed_outcomes(self, b, i, direction):
        atr = b['atr'][i]
        if not (np.isfinite(atr) and atr > 0) or i + 1 >= len(b['c']):
            return None
        entry = b['o'][i + 1]
        risk = self.STOP_ATR * atr
        peak, last, stop_at = 0.0, i, None
        hit_at = [None] * len(FIXED_TARGETS)
        for j in range(i + 1, min(i + 1 + self.VERT, len(b['c']))):
            last = j
            favorable = ((b['h'][j] - entry) / risk if direction == 1
                         else (entry - b['l'][j]) / risk)
            adverse = ((b['l'][j] - entry) / risk if direction == 1
                       else (entry - b['h'][j]) / risk)
            peak = max(peak, favorable)
            for ti, target in enumerate(FIXED_TARGETS):
                if hit_at[ti] is None and favorable >= target:
                    hit_at[ti] = j
            if stop_at is None and adverse <= -1.0:
                stop_at = j
            if stop_at is not None and all(t is not None for t in hit_at):
                break
        midpoint = (b['h'][last] + b['l'][last]) / 2.0
        close_r = ((midpoint - entry) / risk if direction == 1 else (entry - midpoint) / risk)
        realized = np.empty(len(FIXED_TARGETS), np.float32)
        for ti, target in enumerate(FIXED_TARGETS):
            if hit_at[ti] is not None and (stop_at is None or hit_at[ti] <= stop_at):
                realized[ti] = target - self.COST_R
            elif stop_at is not None and (hit_at[ti] is None or stop_at < hit_at[ti]):
                realized[ti] = -1.0 - self.COST_R
            else:
                realized[ti] = close_r - self.COST_R
        return realized, peak

    def build(self, lo, hi, test_start):
        test_start = None if test_start is None else self._utc(test_start)
        contexts, labels, keys = [], [], []
        for (tk, tf), b in self._b.items():
            ts, n = b['ts'], len(b['c'])
            for i, direction in self._signals(b, lo, hi):
                if i < self.CTX or i + 1 + self.VERT >= n:
                    continue
                # Never let a continuous-contract roll create an input pattern or
                # a synthetic TP/SL outcome. Endpoint equality is sufficient because
                # segment ids increase monotonically at every contract change.
                if (b['segment'][i - self.MV_SEQ + 1] != b['segment'][i] or
                        b['segment'][i] != b['segment'][i + 1 + self.VERT]):
                    continue
                label_end = ts[i + 1 + self.VERT]
                if test_start is not None and label_end >= test_start:
                    continue
                result = self._fixed_outcomes(b, i, direction)
                if result is None:
                    continue
                realized, peak = result
                primary = realized[FIXED_TARGETS.index(self.PRIMARY_R)]
                contexts.append(i)
                labels.append(int(primary > 0))
                keys.append((f'{tk}@{tf}', int(i), direction, float(peak),
                             *realized.tolist()))
        return contexts, np.asarray(labels, np.int8), keys

    def label_end_times(self, keys):
        return [self._b[tuple(k[0].split('@'))]['ts'][int(k[1]) + 1 + self.VERT]
                for k in keys]

    def sample_time_bounds_ns(self, keys):
        """Exact inclusive input/decision/outcome bounds for purged inner validation."""
        bounds = []
        for key in keys:
            bars = self._b[tuple(key[0].split('@'))]
            decision = int(key[1])
            bounds.append((
                pd.Timestamp(bars['ts'][decision - self.MV_SEQ + 1]).value,
                pd.Timestamp(bars['ts'][decision]).value,
                pd.Timestamp(bars['ts'][decision + 1 + self.VERT]).value,
            ))
        return np.asarray(bounds, dtype=np.int64)

    def evaluate(self, keys, preds, risk_preds=None):
        col = 4 + FIXED_TARGETS.index(self.PRIMARY_R)
        return np.asarray([k[col] for k, pred in zip(keys, preds) if pred == 1])

    def mv_feature_names(self):
        return ['open', 'high', 'low', 'close', 'volume']

    def mv_contexts(self, keys):
        out = []
        for key in keys:
            b = self._b[tuple(key[0].split('@'))]
            i = int(key[1])
            sl = slice(i - self.MV_SEQ + 1, i + 1)
            out.append(np.nan_to_num(np.stack([
                b['o'][sl], b['h'][sl], b['l'][sl], b['c'][sl], b['v'][sl]
            ]).astype(np.float32)))
        return np.stack(out)

    def features(self, keys):
        return np.zeros((len(keys), 0), np.float32)

    def feature_names(self):
        return []


def _data_attestation(data_dir: Path, tickers, tfs):
    def endpoints(path):
        with path.open('rb') as f:
            header = f.readline().decode().rstrip('\r\n')
            first = f.readline().decode().rstrip('\r\n')
            f.seek(0, os.SEEK_END)
            end = f.tell()
            pos = max(0, end - 2)
            while pos > 0:
                f.seek(pos)
                if f.read(1) == b'\n' and pos < end - 1:
                    break
                pos -= 1
            f.seek(pos + (1 if pos else 0))
            last = f.readline().decode().rstrip('\r\n')
        names = next(csv.reader([header]))
        first_row = dict(zip(names, next(csv.reader([first]))))
        last_row = dict(zip(names, next(csv.reader([last]))))
        if 'contract_id' not in names and 'instrument_id' not in names:
            raise ValueError(f'{path} lacks contract_id/instrument_id')
        return first_row['datetime'], last_row['datetime']

    rows = []
    for tk in tickers:
        for tf in tfs:
            path = data_dir / f'{tk}_{tf}.csv'
            if not path.is_file():
                raise FileNotFoundError(path)
            start, end = endpoints(path)
            rows.append({'stream': f'{tk}@{tf}', 'bytes': path.stat().st_size,
                         'start': pd.Timestamp(start).isoformat(),
                         'end': pd.Timestamp(end).isoformat()})
    return rows


def _clf_kwargs(arm: Arm, head: str, args):
    return dict(backbone_ckpt=str(arm.checkpoint), model_id=arm.model_id,
                model_version=arm.version, device=args.device, head=head,
                batch=args.embed_batch, raw_C=5, raw_seq=SuperTrendMantis.MV_SEQ,
                with_features=False, max_fit_rows=args.max_fit_rows)


def run(args):
    admissions = _verify_arm_admissions(args.arms, args.admissions)
    data_dir = args.data_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    streams = [(tk, tf) for tk in args.tickers for tf in args.tfs]
    attestation = _data_attestation(data_dir, args.tickers, args.tfs)
    latest = max(pd.Timestamp(r['end']) for r in attestation)
    holdout = pd.Timestamp(args.holdout_start, tz='UTC')
    if latest < holdout:
        raise ValueError(f'no OOS data: latest={latest}, holdout={holdout}')
    if args.mode == 'oos' and args.confirm_oos != OOS_CONFIRMATION:
        raise ValueError(f'OOS is one-shot; pass --confirm-oos {OOS_CONFIRMATION} only after '
                         'the finalists and head settings are frozen')

    os.environ.setdefault('EMBED_CACHE_DIR', str(args.output_dir / 'embed_cache'))
    if args.fp16_cache:
        os.environ.setdefault('EMBED_CACHE_FP16', '1')
        os.environ.setdefault('FEATURIZE_FP16', '1')

    base = {
        'schema_version': 'supertrend_mantis_benchmark_v1',
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'mode': args.mode, 'holdout_start': args.holdout_start,
        'tickers': list(args.tickers), 'timeframes': list(args.tfs),
        'data_months': args.data_months, 'data': attestation,
        'label': {'trigger': 'SuperTrend(10,3) flip', 'entry': 'next_bar_open',
                  'stop': '0.5*ATR20', 'target': '3R', 'vertical_bars': 120,
                  'cost_R': 0.03, 'same_bar_policy': 'author_tp_first',
                  'roll_policy': 'context_and_outcome_must_remain_in_one_contract'},
        'arms': [{
            'name': a.name, 'version': a.version, 'model_id': a.model_id,
            'checkpoint': str(a.checkpoint), 'sha256': _sha256(a.checkpoint),
            'admission': {
                'integrity': admissions[a.name]['integrity'],
                'registry_sha256': admissions[a.name]['registry_sha256'],
                'dossier_sha256': admissions[a.name]['dossier_sha256'],
            },
        } for a in args.arms],
        'heads': list(args.heads), 'results': [],
    }
    report = args.output_dir / f'{args.mode}_report.json'
    _atomic_json(report, dict(base, status='running'))

    def factory(tk, tf):
        return SuperTrendMantis(data_dir, [tk], [tf], data_months=args.data_months)

    for arm in args.arms:
        for head in args.heads:
            name = f'{arm.name}_{head}'
            rundir = args.output_dir / args.mode / name
            rundir.mkdir(parents=True, exist_ok=True)
            kwargs = _clf_kwargs(arm, head, args)
            if args.mode == 'wf':
                from futures_foundation.finetune import wf
                result = wf.run_streamed(
                    factory, streams, classifier='mantis_frozen', clf_kwargs=kwargs,
                    train_m=args.train_months, val_m=args.val_months,
                    test_m=args.test_months, max_folds=args.max_folds,
                    holdout_start=args.holdout_start, output_path=str(rundir / 'run'),
                    chunk=args.chunk, seed=args.seed, verbose=True,
                    fold_ckpt=str(rundir / 'folds.npz'))
            else:
                from futures_foundation.finetune import produce
                result = produce.train_final_streamed(
                    factory, streams, classifier='mantis_frozen', clf_kwargs=kwargs,
                    holdout_start=args.holdout_start, seed=args.seed, chunk=args.chunk,
                    export_onnx=False, output_path=str(rundir / 'model'), verbose=True)
            base['results'].append({'arm': arm.name, 'head': head, 'metrics': result})
            _atomic_json(report, dict(base, status='running'))
    base.update(status='complete', finished_utc=datetime.now(timezone.utc).isoformat())
    _atomic_json(report, base)
    return base


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('wf', 'oos'), default='wf')
    p.add_argument('--arm', dest='arms', action='append', type=_parse_arm, required=True,
                   help='repeat NAME,v1|v2,CHECKPOINT')
    p.add_argument('--admission', dest='admissions', action='append', type=_parse_admission,
                   required=True, help='repeat ARM_NAME=/path/current-admission.json')
    p.add_argument('--heads', default='logistic,mlp')
    p.add_argument('--data-dir', type=Path, default=Path('data/ssl_corpus_v2_6tf'))
    p.add_argument('--output-dir', type=Path, default=Path('output/supertrend_benchmark'))
    p.add_argument('--tickers', default=','.join(ROOTS))
    p.add_argument('--tfs', default=','.join(AUTHOR_TFS))
    p.add_argument('--data-months', type=int, default=72)
    p.add_argument('--holdout-start', default='2026-01-01')
    p.add_argument('--train-months', type=int, default=3)
    p.add_argument('--val-months', type=int, default=1)
    p.add_argument('--test-months', type=int, default=1)
    p.add_argument('--max-folds', type=int)
    p.add_argument('--embed-batch', type=int, default=256)
    p.add_argument('--max-fit-rows', type=int, default=0)
    p.add_argument('--chunk', type=int, default=2000)
    p.add_argument('--device', default='cuda')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--fp16-cache', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--confirm-oos')
    args = p.parse_args()
    args.data_dir = args.data_dir.expanduser()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.tickers = tuple(x for x in args.tickers.split(',') if x)
    args.tfs = tuple(x for x in args.tfs.split(',') if x)
    args.heads = tuple(x for x in args.heads.split(',') if x)
    if not args.heads or any(x not in {'logistic', 'mlp'} for x in args.heads):
        p.error('--heads must contain logistic and/or mlp')
    run(args)


if __name__ == '__main__':
    main()
