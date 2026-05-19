"""End-to-end training CLI (spec section 9).

  python -m pipelines.xgboost.train --timeframe 5m --instrument ES --trials 300

features (FFM 68) -> V2 labels -> rolling walk-forward [Optuna(combined obj,
hybrid-trail backtest) -> refit -> OOS backtest] -> aggregate -> save joblib
-> full stat block (every OOS month printed). xgboost/joblib imported lazily.

Optional --rf-gate / --hmm (spec 10/11) are NOT implemented (out of the
primary 1-9 scope); the flags are accepted and ignored with a notice so a
caller is never silently misled.
"""
import argparse
import datetime as _dt
import os
import sys

import numpy as np
import pandas as pd

from futures_foundation.features import derive_features
from .base import get_labeler, XGBStrategyLabeler
from . import labeler as _v2          # noqa: F401 — registers v2_triple_barrier
from pipelines.common.walkforward import walk_forward_windows, optuna_holdout
from .tuner import (tune, _fit_xgb, _signals_from_proba, CONF_THRESHOLD,
                    _TO_XGB)
from .backtest import run_backtest, _stats
from pipelines.common.objective import PERIODS_PER_YEAR

_TF = {'5m': ('5min', 14, 5), '3m': ('3min', 20, 3)}   # file, atr_period, bar_min
_DATA = os.path.join(os.path.dirname(__file__), '..', '..', 'data')


def _print_stats(tag: str, st: dict):
    pf = st['profit_factor']
    print(f"  [{tag}] trades={st['trades']} WR={st['win_rate']:.1%} "
          f"PnL={st['pnl']:+.4f} PF={pf:.2f} "
          f"avgW={st['avg_win']:+.4f} avgL={st['avg_loss']:+.4f} "
          f"maxDD={st['max_dd']:+.2%} "
          f"maxWcons={st.get('max_consec_win',0)} "
          f"maxLcons={st.get('max_consec_loss',0)}")


def _load_ticker(labeler, instrument, period, atr_p, prepared_dir, FCOLS):
    """Load ONE ticker -> dict(X[all bars float ndarray], y[all bars],
    ev[event_mask all bars], ohlcv[all bars], periods[month PeriodIndex]).
    Seam guards identical to the legacy single-ticker path; just factored
    per ticker so run_pipeline can pool 1-or-many."""
    csv = os.path.join(_DATA, f'{instrument}_{period}.csv')
    if not os.path.exists(csv):
        sys.exit(f'data file not found: {csv} '
                 f'(run databento/build_continuous.py {period})')
    df = pd.read_csv(csv)
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    if prepared_dir is None:
        feat = derive_features(df, instrument=instrument, atr_period=atr_p)
    else:
        pq = os.path.join(prepared_dir, f'{instrument}_features.parquet')
        if not os.path.exists(pq):
            sys.exit(f'prepared parquet not found: {pq} '
                     f'(build it with colabs/prepare_data_3min.py)')
        feat = pd.read_parquet(pq).reset_index(drop=True)
        # parquet is row-1:1 with the raw CSV by construction (prepare_data
        # drops no rows); PROVE on datetime before positional alignment.
        if len(feat) != len(df):
            sys.exit(f'seam abort: parquet rows {len(feat)} != CSV rows '
                     f'{len(df)} ({instrument}) — rebuild parquet')
        if not np.array_equal(pd.to_datetime(feat['_datetime'], utc=True)
                              .to_numpy(), df['datetime'].to_numpy()):
            sys.exit(f'seam abort: parquet _datetime != CSV datetime '
                     f'({instrument}) — cannot positionally align')
        miss = [c for c in (*FCOLS, 'vty_atr_raw') if c not in feat.columns]
        if miss:
            sys.exit(f'seam abort: parquet missing {miss[:8]} '
                     f'(stale/mismatched prepare_data build)')
    lab_df = pd.DataFrame({'datetime': df['datetime'].values,
                           'open': df['open'].values, 'high': df['high'].values,
                           'low': df['low'].values, 'close': df['close'].values,
                           'atr': feat['vty_atr_raw'].values})
    y = np.asarray(labeler.label(lab_df))
    ev = np.asarray(labeler.event_mask(lab_df), dtype=bool)
    if ev.shape[0] != len(df):
        sys.exit(f'event_mask len {ev.shape[0]} != rows {len(df)} '
                 f'({instrument})')
    di = pd.DatetimeIndex(df['datetime'])
    di = di.tz_localize(None) if di.tz is not None else di
    return {'inst': instrument,
            'X': feat[FCOLS].to_numpy(),               # NO nan-fill (xgb native)
            'y': y, 'ev': ev,
            'ohlcv': df[['datetime', 'open', 'high', 'low',
                         'close']].reset_index(drop=True),
            'periods': di.to_period('M'), 'csv': csv,
            'span': (di[0].date(), di[-1].date())}


def run_pipeline(labeler: XGBStrategyLabeler, timeframe: str,
                 instrument: str = 'ES', trials: int = 300,
                 max_windows: int | None = None,
                 train_months: int = 3, test_months: int = 1,
                 val_frac: float = 0.15,
                 shuffle_train_labels: bool = False,
                 save_artifact: bool = True, seed: int = 42,
                 prepared_dir: str | None = None,
                 device: str = 'cpu',
                 instruments: str | list | None = None,
                 window_select: set | None = None,
                 optuna_jobs: int = 1,
                 backtest_config: dict | None = None) -> dict:
    """End-to-end run for ANY strategy labeler (finetune-parity API):

        run_pipeline(MyLabeler(bar_minutes=5), '5m', 'ES', trials=300)

    The harness owns features/walk-forward/Optuna/trail/gate/artifact; the
    labeler owns only the {-1,0,+1} target + event_mask + (opt) feature_cols.

    train_months/test_months: walk-forward window (spec-validated = 3/1).
    val_frac: Optuna validation = last `val_frac` of each train window.
    shuffle_train_labels: ROBUSTNESS CONTROL — permute TRAIN labels per
        window only (OOS untouched); a PASS then = leakage, not edge.

    prepared_dir: ADDITIVE INPUT-SOURCE SEAM (flagged vs literal spec §3).
        Load the 68 features + vty_atr_raw from
        {prepared_dir}/{INSTRUMENT}_features.parquet — the validated
        derive_features output prepare_data caches; identical matrix, no
        recompute. OHLCV/open still from the raw CSV (parquet has no _open);
        hard datetime-aligned per ticker or abort.
    device: 'cpu' (default, unchanged) or 'cuda' (Colab GPU). Additive,
        spec-CONSISTENT (XGBoost-accel-is-CUDA-only).

    instruments: 1 OR MANY (ADDITIVE, flagged vs spec's single-instrument
        default). None -> [instrument] (legacy path, byte-equivalent for an
        all-True event_mask). A list -> ONE POOLED model (cisd-style: FFM
        features are ATR-normalised/instrument-agnostic). Pooling rules,
        each a correctness requirement:
          * model FIT/PREDICT only on labeler.event_mask rows (the actual
            decision points; default all-True = every bar = legacy);
          * walk-forward on the SHARED monthly calendar (union of tickers);
          * a backtest is ONE sequential book — it CANNOT span tickers, so
            each ticker is backtested on its own contiguous slice (full bar
            series; signals only at event rows) and per-trade returns are
            concatenated for the Optuna objective and the OOS month.

    window_select / optuna_jobs: ADDITIVE, flagged, back-compat (defaults
        None / 1 = unchanged). window_select = the set of 1-based walk-forward
        window indices to run (indices are globally deterministic from the
        fixed month calendar) so independent PROCESSES can each own a disjoint
        subset for GIL-free parallelism; the return dict carries
        month_returns + per_ticker_returns for an exact merge. optuna_jobs =
        Optuna's own (thread) trial parallelism (spec-sanctioned; real
        speedup is the process/window route)."""
    period, atr_p, bar_min = _TF[timeframe]
    if instruments is None:
        tickers = [instrument]
    elif isinstance(instruments, str):
        tickers = [instruments]
    else:
        tickers = list(instruments)
    FCOLS = labeler.feature_cols()
    src = f'parquet:{prepared_dir}' if prepared_dir else 'csv+derive_features'
    print(f'== XGBoost pipeline | {",".join(tickers)} {timeframe} | '
          f'labeler={labeler.name} | trials={trials} | feat={src} | '
          f'device={device} | pooled={len(tickers) > 1} ==', flush=True)
    data = [_load_ticker(labeler, t, period, atr_p, prepared_dir, FCOLS)
            for t in tickers]

    bar_tot = sum(len(d['y']) for d in data)
    ev_tot = sum(int(d['ev'].sum()) for d in data)
    pos_tot = sum(int((d['y'][d['ev']] != 0).sum()) for d in data)
    print(f'  pooled: {bar_tot:,} bars | {ev_tot:,} event rows '
          f'({ev_tot / max(bar_tot, 1):.1%}) | {pos_tot:,} directional '
          f'({pos_tot / max(ev_tot, 1):.1%} of events)', flush=True)
    if shuffle_train_labels:
        print('  ⚠ SHUFFLE CONTROL: train labels permuted per window '
              '(OOS untouched) — a PASS here means leakage/overfit, not edge',
              flush=True)
    rng = np.random.default_rng(seed)

    months = pd.PeriodIndex(sorted(set().union(
        *[set(d['periods'].unique()) for d in data])), freq='M')

    oos_returns, month_rows, last_model = [], [], None
    per_tk = {}                                  # inst -> [returns] (OOS)
    w, s = 0, 0
    while s + train_months + test_months <= len(months):
        trP = months[s:s + train_months]
        teP = months[s + train_months:s + train_months + test_months]
        s += test_months
        w += 1
        if max_windows is not None and w > max_windows:
            print(f'  (stopping at --max-windows={max_windows})', flush=True)
            break
        # window index w is globally deterministic (fixed month calendar),
        # so independent processes can each own a disjoint window subset and
        # the merge is exact. Default None = all windows (unchanged).
        if window_select is not None and w not in window_select:
            continue

        Xf_p, yf_p, Xtr_p, ytr_p, Xvl_p, yvl_p = [], [], [], [], [], []
        val_blocks, oos_blocks = [], []
        pooled_val_ev = pooled_te_ev = 0

        for d in data:
            per, ev = d['periods'], d['ev']
            tr_m = np.asarray(per.isin(trP))
            te_m = np.asarray(per.isin(teP))
            if not tr_m.any() or not te_m.any():
                continue
            fit_m, val_m = optuna_holdout(tr_m, val_frac)
            yw = d['y'].copy()
            if shuffle_train_labels:
                ti = np.flatnonzero(tr_m)        # permute TRAIN labels only
                yw[ti] = rng.permutation(yw[ti])

            fe, ve, tre = fit_m & ev, val_m & ev, tr_m & ev
            if fe.any():
                Xf_p.append(d['X'][fe]); yf_p.append(yw[fe])
            if tre.any():
                Xtr_p.append(d['X'][tre])
                ytr_p.append(np.array([_TO_XGB[v] for v in yw[tre]]))
            if ve.any():
                Xvl_p.append(d['X'][ve])
                yvl_p.append(np.array([_TO_XGB[v] for v in yw[ve]]))

            vpos = np.flatnonzero(val_m)         # contiguous train tail
            if vpos.size:
                vs = slice(vpos[0], vpos[-1] + 1)
                vev = ev[vs]
                pooled_val_ev += int(vev.sum())
                if vev.any():
                    val_blocks.append({
                        'Xv': d['X'][vs][vev],
                        'ohlcv': d['ohlcv'].iloc[vs].reset_index(drop=True),
                        'ev_pos': np.flatnonzero(vev)})
            tpos = np.flatnonzero(te_m)          # contiguous test month(s)
            if tpos.size:
                ts = slice(tpos[0], tpos[-1] + 1)
                tev = ev[ts]
                pooled_te_ev += int(tev.sum())
                if tev.any():
                    oos_blocks.append({
                        'inst': d['inst'],
                        'Xv': d['X'][ts][tev],
                        'ohlcv': d['ohlcv'].iloc[ts].reset_index(drop=True),
                        'ev_pos': np.flatnonzero(tev)})

        if pooled_val_ev < 20 or pooled_te_ev < 10 or not Xf_p:
            continue
        Xf, yf = np.concatenate(Xf_p), np.concatenate(yf_p)
        print(f'  window {w} [{trP[0]}..{teP[-1]}]: fit_ev={len(yf)} '
              f'val_ev={pooled_val_ev} test_ev={pooled_te_ev} '
              f'({len(data)}tk) — tuning...', flush=True)

        best = tune(Xf, yf, val_blocks, timeframe, n_trials=trials,
                    device=device, n_jobs=optuna_jobs,
                    backtest_config=backtest_config)

        import xgboost as xgb
        model = xgb.XGBClassifier(
            objective='multi:softprob', num_class=3, eval_metric='mlogloss',
            tree_method='hist', device=device, n_jobs=-1,
            early_stopping_rounds=50, **best)
        Xtr, ytr = np.concatenate(Xtr_p), np.concatenate(ytr_p)
        if Xvl_p:
            model.fit(Xtr, ytr,
                      eval_set=[(np.concatenate(Xvl_p),
                                 np.concatenate(yvl_p))], verbose=False)
        else:
            model.fit(Xtr, ytr, verbose=False)
        last_model = model

        mret = []                                # per-ticker OOS, pooled month
        for b in oos_blocks:
            esig = _signals_from_proba(model.predict_proba(b['Xv']),
                                       CONF_THRESHOLD)
            sig = np.zeros(len(b['ohlcv']), dtype=np.int8)
            sig[b['ev_pos']] = esig
            r = run_backtest(b['ohlcv'], sig, backtest_config)['returns']
            mret.append(r)
            per_tk.setdefault(b['inst'], []).append(r)   # pooling-harm check
        if not mret:
            continue
        mr = pd.concat(mret, ignore_index=True)
        st = _stats(mr, [])
        mlabel = str(teP[0])
        month_rows.append((mlabel, st))
        oos_returns.append(mr)
        _print_stats(f'OOS {mlabel}', st)

    if not oos_returns:
        if window_select is not None:            # a parallel worker's subset
            return {'gate_pass': True, 'months': [], 'aggregate': None,
                    'artifact': None, 'n_months': 0, 'month_returns': [],
                    'per_ticker_returns': {}}     # nothing to merge from here
        sys.exit('No completed walk-forward windows (need >=4 months data).')

    agg = pd.concat(oos_returns, ignore_index=True)
    print('\n=== AGGREGATE OOS ===')
    _print_stats('AGG', _stats(agg, []))
    print('\n=== PER-OOS-MONTH ===')
    pf_floor_ok = True
    for m, st in month_rows:
        _print_stats(m, st)
        if not (st['profit_factor'] > 1.0):
            pf_floor_ok = False
    print(f"\nGATE (every OOS month PF>1): "
          f"{'PASS' if pf_floor_ok else 'FAIL'}")
    if len(data) > 1:                            # pooling-harm visibility
        print('\n=== PER-TICKER OOS (pooled model; spot a ticker pooling '
              'hurts) ===')
        for inst in tickers:
            rs = per_tk.get(inst)
            if rs:
                _print_stats(inst, _stats(pd.concat(rs, ignore_index=True),
                                          []))

    out = None
    if save_artifact and not shuffle_train_labels:
        date_str = _dt.date.today().strftime('%Y%m%d')
        tag = (tickers[0].lower() if len(tickers) == 1
               else f'pool{len(tickers)}')
        out = f'xgb_{tag}_{timeframe}_{labeler.name}_{date_str}.joblib'
        import joblib
        joblib.dump({'model': last_model, 'feature_names': FCOLS,
                     'classes': [-1, 0, 1],
                     'confidence_threshold': CONF_THRESHOLD,
                     'timeframe': timeframe, 'instruments': tickers,
                     'atr_period': atr_p, 'labeler': labeler.name,
                     'labeler_config': labeler.config_dict()}, out)
        print(f'\nsaved model -> {out}', flush=True)
    for d in data:
        print(f'  data: {d["csv"]} | span {d["span"][0]}..{d["span"][1]}',
              flush=True)
    return {'gate_pass': pf_floor_ok, 'months': month_rows,
            'aggregate': _stats(agg, []), 'artifact': out,
            'n_months': len(month_rows),
            # raw returns for a multiprocess driver to MERGE exactly
            # (additive; sequential callers can ignore):
            'month_returns': list(zip([m for m, _ in month_rows],
                                      oos_returns)),
            'per_ticker_returns': per_tk}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--timeframe', choices=['5m', '3m'], default='5m')
    ap.add_argument('--instrument', default='ES',
                    help='single ticker (legacy path)')
    ap.add_argument('--instruments', default=None,
                    help='ADDITIVE (flagged vs spec single-instrument): '
                         'comma list -> ONE POOLED model (cisd-style), e.g. '
                         'ES,NQ,RTY,YM,GC,SI. Overrides --instrument.')
    ap.add_argument('--labeler', default='v2_triple_barrier',
                    help='registered strategy labeler name (see base.LABELERS)')
    ap.add_argument('--trials', type=int, default=300)
    ap.add_argument('--max-windows', type=int, default=None,
                    help='cap walk-forward windows (smoke: e.g. 3). '
                         'trials only bounds Optuna; this bounds the run.')
    ap.add_argument('--parquet-dir', default=None,
                    help='ADDITIVE seam (flagged vs spec §3): load 68 features '
                         '+ vty_atr_raw from {dir}/{INSTRUMENT}_features.parquet '
                         '(validated prepare_data output) instead of deriving; '
                         'OHLCV/open still from the raw CSV.')
    ap.add_argument('--device', choices=['cpu', 'cuda'], default='cpu',
                    help="XGBoost device. 'cuda' for Colab GPU. Default cpu = "
                         "unchanged behaviour.")
    ap.add_argument('--rf-gate', action='store_true')
    ap.add_argument('--hmm', action='store_true')
    a = ap.parse_args(argv)
    if a.rf_gate or a.hmm:
        print('NOTE: --rf-gate/--hmm are optional spec sections 10/11 and are '
              'NOT implemented (primary path = sections 1-9). Ignoring.')
    bar_min = _TF[a.timeframe][2]
    labeler = get_labeler(a.labeler, bar_minutes=bar_min)
    insts = ([t.strip() for t in a.instruments.split(',') if t.strip()]
             if a.instruments else None)
    run_pipeline(labeler, a.timeframe, a.instrument, a.trials, a.max_windows,
                 prepared_dir=a.parquet_dir, device=a.device,
                 instruments=insts)


if __name__ == '__main__':
    main()
