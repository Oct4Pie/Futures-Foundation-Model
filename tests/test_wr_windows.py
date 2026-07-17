"""Shared backbone-eval window cache (build_wr_cache / load_wr_cache).

Contract: raw OHLCV windows [N,5,seq] ending at each gated pivot's confirm bar; seq + detector are
honored and self-described; the window is strictly the PAST (ends at confirm, no future bars); the
npz round-trips.
"""
import numpy as np
import pandas as pd
import pytest

from futures_foundation.finetune.wr_windows import (
    FIXED_TARGETS, _triple_barrier, build_wr_cache, load_wr_cache)


def _write_csv(dirpath, tk, tf, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    c = 100 + rng.normal(0, 1, n).cumsum()
    o = np.roll(c, 1); o[0] = c[0]
    h = np.maximum(o, c) + np.abs(rng.normal(0, 0.3, n))
    l = np.minimum(o, c) - np.abs(rng.normal(0, 0.3, n))
    v = rng.integers(100, 1000, n).astype(float)
    ts = pd.date_range('2020-01-01', periods=n, freq='3min', tz='UTC')
    contract = np.where(np.arange(n) < n // 2, 'H24', 'M24')
    df = pd.DataFrame({'datetime': ts, 'open': o, 'high': h, 'low': l, 'close': c,
                       'volume': v, 'contract_id': contract})
    p = dirpath / f'{tk}_{tf}.csv'
    df.to_csv(p, index=False)
    return p


def test_shape_seq_and_roundtrip(tmp_path):
    _write_csv(tmp_path, 'ES', '3min', seed=1)
    out = tmp_path / 'wr_seq128.npz'
    build_wr_cache(out, data_dir=str(tmp_path), tickers=['ES'], tfs=['3min'],
                   seq=128, detector='fractal_zigzag', verbose=False)
    d = load_wr_cache(out)
    assert d['win'].shape[1] == 5 and d['win'].shape[2] == 128     # [N,5,seq]
    assert d['seq'] == 128 and d['detector'] == 'fractal_zigzag'   # self-describing
    assert d['dir'] is not None and d['trend'] is not None         # v2 counter-trend fields
    assert len(d['peak']) == len(d['r3']) == d['win'].shape[0]
    assert d['realized'].shape == d['reached'].shape
    assert d['realized'].shape == (d['win'].shape[0], len(FIXED_TARGETS))
    assert np.allclose(d['targets'], FIXED_TARGETS)
    assert d['meta']['version'] == 4


def test_positive_vertical_exit_is_not_mislabeled_as_target_reached():
    # Entry=100, price closes around +0.5R but never touches the 2R target.
    o = np.array([100., 100., 105.])
    h = np.array([100., 105., 115.])
    l = np.array([100., 95., 95.])
    realized, peak, reached = _triple_barrier(
        o, h, l, 0, 1, len(o), risk=10., cost_r=.03, vert=2, targets=(2.,),
        same_bar_policy='stop_first')
    assert realized[0] > 0 and peak < 2
    assert reached.tolist() == [False]


def test_same_bar_stop_first_is_conservative_and_stops_the_path():
    o = np.array([100., 100., 200.])
    h = np.array([100., 121., 300.])
    l = np.array([100., 89., 50.])
    stop_r, stop_peak, stop_hit = _triple_barrier(
        o, h, l, 0, 1, len(o), risk=10., cost_r=.03, vert=2, targets=(2.,),
        same_bar_policy='stop_first')
    tp_r, _, tp_hit = _triple_barrier(
        o, h, l, 0, 1, len(o), risk=10., cost_r=.03, vert=2, targets=(2.,),
        same_bar_policy='tp_first')
    assert stop_r[0] < 0 and not stop_hit[0]
    assert tp_r[0] > 0 and tp_hit[0]
    assert stop_peak == pytest.approx(0.0)                # stop bar high + later bars are unobservable


def test_seq_controls_window_width(tmp_path):
    _write_csv(tmp_path, 'ES', '3min', seed=2)
    short = tmp_path / 's64.npz'; long = tmp_path / 's256.npz'
    build_wr_cache(short, data_dir=str(tmp_path), tickers=['ES'], tfs=['3min'],
                   seq=64, detector='fractal_zigzag', verbose=False)
    build_wr_cache(long, data_dir=str(tmp_path), tickers=['ES'], tfs=['3min'],
                   seq=256, detector='fractal_zigzag', verbose=False)
    assert load_wr_cache(short)['win'].shape[2] == 64
    assert load_wr_cache(long)['win'].shape[2] == 256


def test_unknown_detector_raises(tmp_path):
    # atr_zigzag DELETED 2026-07-16 (lookahead in R/is_trend/leg_end) — only the certified
    # production trigger remains; asking for anything else must fail loud, never fall back.
    _write_csv(tmp_path, 'ES', '3min', seed=3)
    with pytest.raises(ValueError, match='fractal_zigzag'):
        build_wr_cache(tmp_path / 'zz.npz', data_dir=str(tmp_path), tickers=['ES'],
                       tfs=['3min'], seq=64, detector='zigzag', verbose=False)


def test_window_is_causal_past_only(tmp_path):
    """The window must end AT the confirm bar — reconstruct the pivot confirm from ts and assert the
    last window bar's close equals the CSV close at that timestamp (no future bar leaked in)."""
    _write_csv(tmp_path, 'ES', '3min', seed=4)
    out = tmp_path / 'c.npz'
    build_wr_cache(out, data_dir=str(tmp_path), tickers=['ES'], tfs=['3min'],
                   seq=64, detector='fractal_zigzag', verbose=False)
    d = load_wr_cache(out)
    df = pd.read_csv(tmp_path / 'ES_3min.csv')
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    close_at = dict(zip(df['datetime'], df['close']))
    for k in range(0, d['win'].shape[0], max(1, d['win'].shape[0] // 20)):
        last_close = float(d['win'][k, 3, -1])              # channel 3 = close, last bar
        assert last_close == pytest.approx(close_at[d['ts'][k]], abs=1e-4)


def test_structural_stop_differs_and_self_describes(tmp_path):
    _write_csv(tmp_path, 'ES', '3min', seed=5)
    atr_c = tmp_path / 'atr.npz'; struct_c = tmp_path / 'struct.npz'
    build_wr_cache(atr_c, data_dir=str(tmp_path), tickers=['ES'], tfs=['3min'], seq=64,
                   detector='fractal_zigzag', stop_mode='atr', verbose=False)
    build_wr_cache(struct_c, data_dir=str(tmp_path), tickers=['ES'], tfs=['3min'], seq=64,
                   detector='fractal_zigzag', stop_mode='structural', stop_buffer_atr=0.05, verbose=False)
    a, s = load_wr_cache(atr_c), load_wr_cache(struct_c)
    assert a['stop'] == 'atr' and s['stop'] == 'structural'       # self-describing
    # structural risk differs from 0.5*ATR -> different r3 label distribution (not identical)
    assert not np.array_equal(np.sort(a['r3']), np.sort(s['r3']))
    # windows (raw OHLCV) are the SAME schema regardless of stop model
    assert a['win'].shape[1:] == s['win'].shape[1:]


def test_atr_stop_is_backward_compatible(tmp_path):
    """stop_mode='atr' (default) must be byte-identical to the pre-stop-mode builder: r3 depends only
    on stop_atr*ATR. Guards the backward-compat promise."""
    _write_csv(tmp_path, 'ES', '3min', seed=6)
    a1 = tmp_path / 'a1.npz'; a2 = tmp_path / 'a2.npz'
    build_wr_cache(a1, data_dir=str(tmp_path), tickers=['ES'], tfs=['3min'], seq=64,
                   detector='fractal_zigzag', verbose=False)                       # default stop
    build_wr_cache(a2, data_dir=str(tmp_path), tickers=['ES'], tfs=['3min'], seq=64,
                   detector='fractal_zigzag', stop_mode='atr', stop_atr=0.5, verbose=False)
    assert np.array_equal(load_wr_cache(a1)['r3'], load_wr_cache(a2)['r3'])


def test_missing_data_raises(tmp_path):
    with pytest.raises(RuntimeError):
        build_wr_cache(tmp_path / 'x.npz', data_dir=str(tmp_path), tickers=['ZZ'],
                       tfs=['3min'], seq=64, verbose=False)


def test_cache_stores_forward_label_end_for_exact_holdout_purge(tmp_path):
    _write_csv(tmp_path, 'ES', '3min', seed=7)
    out = tmp_path / 'ends.npz'
    build_wr_cache(out, data_dir=str(tmp_path), tickers=['ES'], tfs=['3min'],
                   seq=64, detector='fractal_zigzag', verbose=False)
    d = load_wr_cache(out)
    assert d['label_end'] is not None and len(d['label_end']) == len(d['ts'])
    assert (d['label_end'] > d['ts']).all()


def test_roll_guard_keeps_context_and_outcome_inside_one_contract(tmp_path):
    _write_csv(tmp_path, 'ES', '3min', seed=8)
    out = tmp_path / 'rollsafe.npz'
    seq, vert = 64, 150
    build_wr_cache(out, data_dir=str(tmp_path), tickers=['ES'], tfs=['3min'],
                   seq=seq, vert=vert, detector='fractal_zigzag', roll_guard=True,
                   verbose=False)
    d = load_wr_cache(out)
    df = pd.read_csv(tmp_path / 'ES_3min.csv')
    pos = {pd.Timestamp(t): i for i, t in enumerate(pd.to_datetime(df['datetime'], utc=True))}
    contract = df['contract_id'].to_numpy()
    for t in d['ts']:
        i = pos[pd.Timestamp(t)]
        assert len(set(contract[i - seq + 1:i + vert + 2])) == 1
