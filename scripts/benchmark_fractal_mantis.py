#!/usr/bin/env python3
"""Version-aware, purged fractal-pivot checkpoint benchmark.

This is the local audited counterpart of the author's one-shot gist. The default
development window is 2025-07-01 through 2025-12-31; 2026 requires an explicit
confirmation token and is never used for configuration selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOTS = ('ES', 'NQ', 'RTY', 'YM', 'GC', 'SI', 'CL', 'ZB', 'ZN')
TFS = ('1min', '3min', '5min', '15min')
REACH_TARGETS = (2.0, 3.0, 4.0, 6.0, 8.0)
OOS_CONFIRMATION = 'RUN-2026-ONE-SHOT'
RATES = (5, 4, 3, 2, 1)


@dataclass(frozen=True)
class Arm:
    name: str
    version: str
    checkpoint: Path

    @property
    def model_id(self):
        return 'paris-noah/MantisV2' if self.version == 'v2' else 'paris-noah/Mantis-8M'


def _arm(text):
    try:
        name, version, raw = text.split(',', 2)
    except ValueError as exc:
        raise argparse.ArgumentTypeError('arm must be NAME,v1|v2,CHECKPOINT') from exc
    if version not in {'v1', 'v2'}:
        raise argparse.ArgumentTypeError('arm version must be v1 or v2')
    path = Path(raw).expanduser().resolve()
    if not path.is_file() or path.stat().st_size < 1024:
        raise argparse.ArgumentTypeError(f'missing checkpoint or LFS pointer: {path}')
    return Arm(name, version, path)


def _sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=float) + '\n')
    os.replace(tmp, path)


def _dataset_fingerprint(data_dir, tickers, tfs):
    manifest = Path(data_dir) / 'MANIFEST.json'
    if manifest.is_file():
        return _sha256(manifest)
    h = hashlib.sha256()
    for tk in tickers:
        for tf in tfs:
            path = Path(data_dir) / f'{tk}_{tf}.csv'
            st = path.stat()
            h.update(f'{path.name}:{st.st_size}:{st.st_mtime_ns}'.encode())
    return h.hexdigest()


def _cache_expected(args):
    return {
        'version': 4, 'tickers': list(args.tickers), 'tfs': list(args.tfs),
        'vert': args.vert, 'data_months': args.data_months, 'roll_guard': True,
        'htf_gate': True, 'atr_period': 20, 'min_history': 128, 'trend_n': 480,
        'rev_atr': 1.25, 'fractal_k': 2, 'fractal_leg_atr': 1.25,
        'stop_atr': 0.5, 'stop_buffer_atr': args.stop_buffer_atr,
        'stop_cap_atr': args.stop_cap_atr, 'cost_r': 0.03,
        'same_bar_policy': args.same_bar_policy,
    }


def _cache_matches(cache, args):
    if (cache.get('label_end') is None or cache.get('realized') is None or
            cache.get('reached') is None or cache.get('targets') is None):
        return False
    if not np.allclose(cache['targets'], REACH_TARGETS):
        return False
    if cache['seq'] != args.seq or cache['detector'] != 'fractal_zigzag':
        return False
    if cache['stop'] != 'structural':
        return False
    meta, expected = cache.get('meta') or {}, _cache_expected(args)
    for key, value in expected.items():
        got = meta.get(key)
        if isinstance(value, float):
            if got is None or not np.isclose(got, value):
                return False
        elif got != value:
            return False
    return set(np.unique(cache['tk'])) == set(args.tickers) and set(np.unique(cache['tf'])) == set(args.tfs)


def _session_days(timestamps, timezone='America/Chicago', roll_hour=17):
    """Exchange-session labels: CME session [17:00 CT, 17:00 CT next day)."""
    local = pd.DatetimeIndex(timestamps).tz_convert(timezone)
    return (local - pd.Timedelta(hours=roll_hour)).normalize().tz_localize(None).to_numpy()


def _query_codes(tickers, tfs, timestamps):
    """Contiguous integer ids for a ticker/timeframe/session candidate set."""
    keys = pd.MultiIndex.from_arrays([
        np.asarray(tickers).astype(str), np.asarray(tfs).astype(str), _session_days(timestamps)])
    return pd.factorize(keys, sort=False)[0].astype(np.int64)


def _complete_group_sample(cache, rows, cap, seed):
    """Seeded row cap that either keeps every candidate in a query group or none of them."""
    rows = np.asarray(rows, np.int64)
    if not cap or len(rows) <= cap:
        return rows
    qid = _query_codes(np.asarray(cache['tk'])[rows], np.asarray(cache['tf'])[rows],
                       pd.DatetimeIndex(cache['ts'])[rows])
    sizes = np.bincount(qid)
    chosen, total = [], 0
    for q in np.random.default_rng(seed).permutation(len(sizes)):
        n = int(sizes[q])
        if total + n <= cap:
            chosen.append(q); total += n
    if not chosen:
        raise ValueError(f'n_train={cap} is smaller than every complete query group')
    out = rows[np.isin(qid, np.asarray(chosen))]
    return np.sort(out)


def _split_cache(cache, *, holdout_start, eval_end, eval_tf='3min', n_train=200000, seed=0):
    """Return fixed train/eval rows with full forward-label purge and whole-group sampling."""
    ts = pd.DatetimeIndex(cache['ts'])
    label_end = cache.get('label_end')
    if label_end is None:
        raise ValueError('cache has no label_end; refusing an unpurged benchmark')
    label_end = pd.DatetimeIndex(label_end)
    hs = pd.Timestamp(holdout_start)
    hs = hs.tz_localize('UTC') if hs.tzinfo is None else hs.tz_convert('UTC')
    ee = None if eval_end is None else pd.Timestamp(eval_end)
    if ee is not None:
        ee = ee.tz_localize('UTC') if ee.tzinfo is None else ee.tz_convert('UTC')
        if ee <= hs:
            raise ValueError(f'eval_end {ee} must follow holdout_start {hs}')
    train = (ts < hs) & (label_end < hs)
    evaluate = (ts >= hs) & (np.asarray(cache['tf']) == eval_tf)
    if ee is not None:
        evaluate &= (ts < ee) & (label_end < ee)
    tr_idx, ev_idx = np.flatnonzero(train), np.flatnonzero(evaluate)
    tr_idx = _complete_group_sample(cache, tr_idx, n_train, seed)
    if not len(tr_idx) or not len(ev_idx):
        raise ValueError(f'empty split: train={len(tr_idx)} eval={len(ev_idx)}')
    return tr_idx, ev_idx


def _split_cache_calibrated(cache, *, holdout_start, eval_end, calibration_months=6,
                            eval_tf='3min', n_train=200000, seed=0):
    """Model-train / threshold-calibration / eval split, purged at both forward boundaries."""
    if calibration_months <= 0:
        raise ValueError('calibration_months must be positive')
    ts, label_end = pd.DatetimeIndex(cache['ts']), pd.DatetimeIndex(cache['label_end'])
    hs = pd.Timestamp(holdout_start)
    hs = hs.tz_localize('UTC') if hs.tzinfo is None else hs.tz_convert('UTC')
    cs = hs - pd.DateOffset(months=calibration_months)
    ee = pd.Timestamp(eval_end)
    ee = ee.tz_localize('UTC') if ee.tzinfo is None else ee.tz_convert('UTC')
    if not cs < hs < ee:
        raise ValueError(f'invalid calibrated split: calibration={cs}, holdout={hs}, end={ee}')
    tf = np.asarray(cache['tf'])
    train = (ts < cs) & (label_end < cs)
    calibrate = (ts >= cs) & (ts < hs) & (label_end < hs) & (tf == eval_tf)
    evaluate = (ts >= hs) & (ts < ee) & (label_end < ee) & (tf == eval_tf)
    tr_idx = _complete_group_sample(cache, np.flatnonzero(train), n_train, seed)
    ca_idx, ev_idx = np.flatnonzero(calibrate), np.flatnonzero(evaluate)
    if not len(tr_idx) or not len(ca_idx) or not len(ev_idx):
        raise ValueError(f'empty calibrated split: train={len(tr_idx)} calibration={len(ca_idx)} '
                         f'eval={len(ev_idx)}')
    return tr_idx, ca_idx, ev_idx, cs


def _days_by_ticker(tickers, session_days):
    out = {}
    for ticker in dict.fromkeys(tickers.tolist()):
        rows = tickers == ticker
        out[str(ticker)] = max(1, np.unique(session_days[rows]).size)
    return out


def _take(score, tickers, session_days, rate):
    """Hindsight ranking diagnostic only; never use this selector for operating metrics."""
    selected = []
    qid = pd.MultiIndex.from_arrays([np.asarray(tickers).astype(str), session_days])
    for key in qid.unique():
        rows = np.flatnonzero(qid == key)
        selected.extend(rows[np.argsort(-score[rows], kind='stable')[:rate]])
    return np.asarray(selected, np.int64)


def _fit_score_thresholds(score, tickers, session_days, rates=RATES):
    """Train-calibration-only score cutoffs targeting ``rate`` raw qualifiers per session."""
    score, tickers = np.asarray(score), np.asarray(tickers).astype(str)
    thresholds = {}
    for ticker in np.unique(tickers):
        rows = tickers == ticker
        values = np.sort(score[rows])[::-1]
        n_days = max(1, np.unique(session_days[rows]).size)
        thresholds[ticker] = {}
        for rate in rates:
            wanted = min(len(values), max(1, int(rate * n_days)))
            thresholds[ticker][int(rate)] = float(values[wanted - 1])
    return thresholds


def _take_causal(score, tickers, session_days, timestamps, thresholds, rate):
    """Sequential gate: fixed pre-eval threshold, then first qualifiers up to the session cap."""
    score = np.asarray(score)
    tickers = np.asarray(tickers).astype(str)
    timestamps = pd.DatetimeIndex(timestamps)
    selected, counts = [], {}
    for i in np.argsort(timestamps.asi8, kind='stable'):
        ticker = tickers[i]
        threshold = thresholds.get(ticker, {}).get(int(rate), float('inf'))
        key = (ticker, session_days[i])
        if score[i] >= threshold and counts.get(key, 0) < rate:
            selected.append(int(i)); counts[key] = counts.get(key, 0) + 1
    return np.asarray(selected, np.int64)


def _operating_metrics(score, r3, reached3, peak, tickers, session_days, timestamps,
                       days, thresholds):
    out = []
    avg_days = float(np.mean(list(days.values())))
    for rate in RATES:
        idx = _take_causal(score, tickers, session_days, timestamps, thresholds, rate)
        rr, reached = r3[idx], reached3[idx]
        gains, losses = rr[rr > 0].sum(), -rr[rr < 0].sum()
        out.append({
            'per_ticker_day': rate, 'signals': len(idx),
            'total_per_day': len(idx) / avg_days,
            'wr3r': float(reached.mean()),
            'pf': float(gains / losses) if losses > 0 else float('inf'),
            'avg_win_peak_r': float(peak[idx][reached].mean()) if reached.any() else 0.0,
            'mean_r': float(rr.mean()),
        })
    return out


def _counter_metrics(score, reached3, direction, trend):
    from sklearn.metrics import roc_auc_score
    counter = direction * trend < 0
    strong = np.abs(trend) >= np.median(np.abs(trend))
    rows_out = []
    for name, mask in (('counter', counter), ('strong_counter', counter & strong),
                       ('aligned', ~counter)):
        rows = np.flatnonzero(mask)
        if len(rows) < 50:
            rows_out.append({'group': name, 'n': len(rows)})
            continue
        y = reached3[rows].astype(int)
        auc = roc_auc_score(y, score[rows]) if len(np.unique(y)) == 2 else None
        cuts = {}
        for frac in (1.0, .5, .25, .1):
            n = max(1, round(frac * len(rows)))
            chosen = rows[np.argsort(-score[rows])[:n]]
            cuts[str(frac)] = float(reached3[chosen].mean())
        rows_out.append({'group': name, 'n': len(rows), 'auc': auc, 'wr_by_top_fraction': cuts})
    return rows_out


def _reach_grades(reached):
    """Ordinal relevance 0..T: the highest forward-R rung touched before the stop."""
    reached = np.asarray(reached, bool)
    if reached.ndim != 2:
        raise ValueError('reached must be [N, targets]')
    # First-touch reach must be nested: reaching X implies every lower rung was reached.
    if reached.shape[1] > 1 and np.any(reached[:, 1:] & ~reached[:, :-1]):
        raise ValueError('non-monotone reach labels in cache')
    return reached.sum(axis=1).astype(np.int8)


def _ranking_metrics(score, grades, qid, ks=(1, 3, 5)):
    """Mean per-query NDCG, excluding groups with no relevant candidate."""
    from sklearn.metrics import ndcg_score
    out = {f'ndcg@{k}': [] for k in ks}
    for q in np.unique(qid):
        rows = qid == q
        if rows.sum() < 2 or grades[rows].max() == 0:
            continue
        for k in ks:
            out[f'ndcg@{k}'].append(ndcg_score(
                grades[rows][None, :], score[rows][None, :], k=min(k, rows.sum())))
    result = {k: (float(np.mean(v)) if v else None) for k, v in out.items()}
    result['effective_groups'] = max((len(v) for v in out.values()), default=0)
    return result


def _print_result(label, result):
    print(f'\n=== {label} ===', flush=True)
    print(f"  {'/tkr/day':>9}{'signals':>9}{'WR@3R':>9}{'PF':>8}{'avgWinR':>10}{'meanR':>9}",
          flush=True)
    for row in result['operating_points']:
        print(f"  {row['per_ticker_day']:>7}/d{row['signals']:>9}{row['wr3r']:>8.1%}"
              f"{row['pf']:>8.2f}{row['avg_win_peak_r']:>+10.2f}{row['mean_r']:>+9.2f}",
              flush=True)


def run(args):
    import torch
    from futures_foundation.finetune.pretext._torch.common import embed_windows
    from futures_foundation.finetune.wr_windows import build_wr_cache, load_wr_cache
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    hs = pd.Timestamp(args.holdout_start)
    if hs.year >= 2026 and args.confirm_oos != OOS_CONFIRMATION:
        raise ValueError(f'2026 is one-shot; pass --confirm-oos {OOS_CONFIRMATION} only after '
                         'the finalists are frozen')
    if args.eval_end is None and hs.year < 2026:
        args.eval_end = '2026-01-01'
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = _dataset_fingerprint(args.data_dir, args.tickers, args.tfs)
    cache_path = (args.cache or args.output_dir /
                  f'wr_fractal_seq{args.seq}_struct_rollsafe_{args.same_bar_policy}_'
                  f'{fingerprint[:12]}.npz')
    cache = load_wr_cache(cache_path) if cache_path.is_file() else None
    if cache is None or not _cache_matches(cache, args):
        if cache is not None:
            print(f'[cache] stale/incompatible cache {cache_path}; rebuilding', flush=True)
        build_wr_cache(
            cache_path, data_dir=str(args.data_dir), tickers=args.tickers, tfs=args.tfs,
            seq=args.seq, detector='fractal_zigzag', stop_mode='structural',
            stop_buffer_atr=args.stop_buffer_atr, stop_cap_atr=args.stop_cap_atr,
            data_months=args.data_months, vert=args.vert, roll_guard=True,
            targets=REACH_TARGETS, same_bar_policy=args.same_bar_policy, verbose=True)
        cache = load_wr_cache(cache_path)
    tr_idx, ca_idx, ev_idx, calibration_start = _split_cache_calibrated(
        cache, holdout_start=args.holdout_start, eval_end=args.eval_end,
        calibration_months=args.calibration_months, n_train=args.n_train, seed=args.seed)
    split_hash = hashlib.sha256(
        tr_idx.tobytes() + ca_idx.tobytes() + ev_idx.tobytes()).hexdigest()
    targets = tuple(float(x) for x in cache['targets'])
    t3i = targets.index(3.0)
    reached_train = np.asarray(cache['reached'][tr_idx], bool)
    reached_eval = np.asarray(cache['reached'][ev_idx], bool)
    y_train = reached_train[:, t3i].astype(np.int8)
    reached3 = reached_eval[:, t3i]
    grades_train, grades_eval = _reach_grades(reached_train), _reach_grades(reached_eval)
    r3, peak = cache['r3'][ev_idx], cache['peak'][ev_idx]
    tickers = np.asarray(cache['tk'])[ev_idx]
    timestamps = pd.DatetimeIndex(cache['ts'])[ev_idx]
    session_days = _session_days(timestamps)
    days = _days_by_ticker(tickers, session_days)
    cal_tickers = np.asarray(cache['tk'])[ca_idx]
    cal_timestamps = pd.DatetimeIndex(cache['ts'])[ca_idx]
    cal_session_days = _session_days(cal_timestamps)
    train_qid = _query_codes(np.asarray(cache['tk'])[tr_idx], np.asarray(cache['tf'])[tr_idx],
                             pd.DatetimeIndex(cache['ts'])[tr_idx])
    eval_qid = _query_codes(tickers, np.asarray(cache['tf'])[ev_idx], timestamps)
    report = {
        'schema_version': 'fractal_mantis_benchmark_v2', 'status': 'running',
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'data_dir': str(args.data_dir), 'dataset_fingerprint': fingerprint,
        'cache': str(cache_path), 'cache_meta': cache['meta'],
        'holdout_start': args.holdout_start, 'eval_end': args.eval_end,
        'oos_spent': hs.year >= 2026, 'train_rows': len(tr_idx),
        'calibration_rows': len(ca_idx), 'eval_rows': len(ev_idx),
        'calibration_start': calibration_start.isoformat(),
        'split_sha256': split_hash, 'eval_days_by_ticker': days,
        'selection': 'causal_calibrated_threshold_then_per_session_cap',
        'same_bar_policy': args.same_bar_policy,
        'label': 'explicit first-touch 3R before stop', 'reach_targets': list(targets),
        'heads': list(args.heads), 'results': [], 'controls': {},
        'arms': [{'name': a.name, 'version': a.version, 'model_id': a.model_id,
                  'checkpoint': str(a.checkpoint), 'sha256': _sha256(a.checkpoint)} for a in args.arms],
    }
    report_path = args.output_dir / 'report.json'
    rng = np.random.default_rng(args.seed)
    random_cal_score, random_score = rng.random(len(ca_idx)), rng.random(len(ev_idx))
    random_thresholds = _fit_score_thresholds(
        random_cal_score, cal_tickers, cal_session_days)
    report['controls']['random'] = {
        'operating_points': _operating_metrics(
            random_score, r3, reached3, peak, tickers, session_days, timestamps, days,
            random_thresholds),
        'ranking': _ranking_metrics(random_score, grades_eval, eval_qid),
    }
    _atomic_json(report_path, report)
    print(f"[bench] train={len(tr_idx):,} calibration={len(ca_idx):,} eval={len(ev_idx):,} window="
          f"[{args.holdout_start}, {args.eval_end or 'end'}) split={split_hash[:12]}", flush=True)

    for arm in args.arms:
        e_train = embed_windows(cache['win'][tr_idx], ckpt=str(arm.checkpoint),
                                model_id=arm.model_id, model_version=arm.version,
                                device=args.device, batch=args.embed_batch)
        scaler = StandardScaler().fit(e_train)
        x_train = scaler.transform(e_train)
        del e_train
        e_cal = embed_windows(cache['win'][ca_idx], ckpt=str(arm.checkpoint),
                              model_id=arm.model_id, model_version=arm.version,
                              device=args.device, batch=args.embed_batch)
        x_cal = scaler.transform(e_cal)
        del e_cal
        e_eval = embed_windows(cache['win'][ev_idx], ckpt=str(arm.checkpoint),
                               model_id=arm.model_id, model_version=arm.version,
                               device=args.device, batch=args.embed_batch)
        x_eval = scaler.transform(e_eval)
        del e_eval
        for head in args.heads:
            score_meaning = 'P(reach 3R before stop)'
            if head == 'mlp':
                model = MLPClassifier(hidden_layer_sizes=(128,), max_iter=300,
                                      batch_size=4096, alpha=1e-4,
                                      early_stopping=True, random_state=args.seed)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', ConvergenceWarning)
                    model.fit(x_train, y_train)
                score_cal = model.predict_proba(x_cal)[:, 1]
                score = model.predict_proba(x_eval)[:, 1]
            elif head == 'logistic':
                model = LogisticRegression(max_iter=2000, C=1.0, random_state=args.seed)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', ConvergenceWarning)
                    model.fit(x_train, y_train)
                score_cal = model.predict_proba(x_cal)[:, 1]
                score = model.predict_proba(x_eval)[:, 1]
            elif head == 'lambdamart':
                import xgboost as xgb
                order = np.argsort(train_qid, kind='stable')
                model = xgb.XGBRanker(
                    objective='rank:ndcg', eval_metric=['ndcg@1', 'ndcg@3', 'ndcg@5'],
                    tree_method='hist', device=args.xgb_device, n_estimators=250,
                    max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.5,
                    min_child_weight=10, reg_lambda=5.0, random_state=args.seed, n_jobs=1,
                    lambdarank_pair_method='topk', lambdarank_num_pair_per_sample=5)
                model.fit(x_train[order], grades_train[order], qid=train_qid[order], verbose=False)
                score_cal = model.predict(x_cal)
                score = model.predict(x_eval)
                score_meaning = 'LambdaMART relevance for forward reach rung'
            elif head == 'xgb_ladder':
                import xgboost as xgb
                from futures_foundation.finetune.risk_head import expected_reach_weights
                raw_cal, raw = [], []
                model = []
                for ti, target in enumerate(targets):
                    y = reached_train[:, ti].astype(np.int8)
                    if len(np.unique(y)) < 2:
                        raw_cal.append(np.full(len(x_cal), float(y.mean()), np.float32))
                        raw.append(np.full(len(x_eval), float(y.mean()), np.float32))
                        model.append(None)
                        continue
                    rung = xgb.XGBClassifier(
                        objective='binary:logistic', eval_metric='logloss', tree_method='hist',
                        device=args.xgb_device, n_estimators=200, max_depth=4,
                        learning_rate=0.05, subsample=0.8, colsample_bytree=0.5,
                        min_child_weight=10, reg_lambda=5.0, random_state=args.seed + ti,
                        n_jobs=1)
                    rung.fit(x_train, y, verbose=False)
                    raw_cal.append(rung.predict_proba(x_cal)[:, 1])
                    raw.append(rung.predict_proba(x_eval)[:, 1])
                    model.append(rung)
                survival_cal = np.minimum.accumulate(np.column_stack(raw_cal), axis=1)
                survival = np.minimum.accumulate(np.column_stack(raw), axis=1)
                score_cal = survival_cal @ expected_reach_weights(targets)
                score = survival @ expected_reach_weights(targets)
                score_meaning = 'expected forward reach from monotone XGBoost survival ladder'
            else:                                               # guarded by argparse; fail closed
                raise ValueError(f'unknown head {head}')
            thresholds = _fit_score_thresholds(
                score_cal, cal_tickers, cal_session_days)
            result = {
                'arm': arm.name, 'head': head, 'score_meaning': score_meaning,
                'thresholds': thresholds,
                'operating_points': _operating_metrics(
                    score, r3, reached3, peak, tickers, session_days, timestamps, days,
                    thresholds),
                'ranking': _ranking_metrics(score, grades_eval, eval_qid),
                'counter_trend': _counter_metrics(
                    score, reached3, cache['dir'][ev_idx], cache['trend'][ev_idx]),
            }
            report['results'].append(result)
            _print_result(f'{arm.name} [{head}]', result)
            _atomic_json(report_path, report)
        del x_train, x_cal, x_eval
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    report.update(status='complete', finished_utc=datetime.now(timezone.utc).isoformat())
    _atomic_json(report_path, report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm', dest='arms', action='append', type=_arm, required=True,
                   help='repeat NAME,v1|v2,CHECKPOINT')
    p.add_argument('--heads', default='logistic,mlp')
    p.add_argument('--data-dir', type=Path, default=Path('data/ssl_corpus_v2_6tf'))
    p.add_argument('--output-dir', type=Path, default=Path('output/fractal_benchmark'))
    p.add_argument('--cache', type=Path)
    p.add_argument('--tickers', default=','.join(ROOTS))
    p.add_argument('--tfs', default=','.join(TFS))
    p.add_argument('--seq', type=int, default=256)
    p.add_argument('--vert', type=int, default=150)
    p.add_argument('--data-months', type=int, default=72)
    p.add_argument('--stop-buffer-atr', type=float, default=.05)
    p.add_argument('--stop-cap-atr', type=float, default=0.0)
    p.add_argument('--same-bar-policy', choices=('stop_first', 'tp_first'), default='stop_first')
    p.add_argument('--holdout-start', default='2025-07-01')
    p.add_argument('--eval-end')
    p.add_argument('--n-train', type=int, default=200000)
    p.add_argument('--calibration-months', type=int, default=6)
    p.add_argument('--embed-batch', type=int, default=512)
    p.add_argument('--device', default='cuda')
    p.add_argument('--xgb-device', default='cuda')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--confirm-oos')
    args = p.parse_args()
    args.data_dir = args.data_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.cache = None if args.cache is None else args.cache.expanduser().resolve()
    args.tickers = tuple(x for x in args.tickers.split(',') if x)
    args.tfs = tuple(x for x in args.tfs.split(',') if x)
    args.heads = tuple(x for x in args.heads.split(',') if x)
    allowed_heads = {'logistic', 'mlp', 'lambdamart', 'xgb_ladder'}
    if any(h not in allowed_heads for h in args.heads):
        p.error(f'--heads must contain only {sorted(allowed_heads)}')
    run(args)


if __name__ == '__main__':
    main()
