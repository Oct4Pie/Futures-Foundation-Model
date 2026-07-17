"""SSL masked-modeling pretraining — torch-free data/assembly/probe/gate (non-gated) +
torch masked trainer (gated behind CHRONOS_TORCH_TESTS=1, libomp isolation).

Run torch parts: CHRONOS_TORCH_TESTS=1 pytest tests/test_finetune_ssl.py
"""
import os

import numpy as np
import pandas as pd
import pytest

from futures_foundation.finetune import ssl, ssl_data, ssl_probe
from futures_foundation.finetune.pretext.nextleg import NextLegTask

torch_test = pytest.mark.skipif(
    os.environ.get('CHRONOS_TORCH_TESTS') != '1',
    reason='torch test — set CHRONOS_TORCH_TESTS=1 (libomp isolation)')


def _write_csv(path, n, start='2024-01-01', freq='3min', base=4000.0):
    ts = pd.date_range(start, periods=n, freq=freq, tz='UTC')
    rng = np.random.default_rng(abs(hash(path)) % 1000)
    close = base + np.cumsum(rng.standard_normal(n))
    pd.DataFrame({'datetime': ts.astype(str), 'open': close, 'high': close + 1,
                  'low': close - 1, 'close': close,
                  'volume': rng.integers(100, 1000, n).astype(float)}).to_csv(path, index=False)


# ---------------------------------------------------------------- torch-free data tests
def test_load_ohlcv(tmp_path):
    _write_csv(tmp_path / 'ES_3min.csv', 500)
    _write_csv(tmp_path / 'NQ_5min.csv', 300)
    streams = ssl_data.load_ohlcv(str(tmp_path), tickers=['ES', 'NQ'],
                                  tfs=['3min', '5min'], verbose=False)
    sids = {s['sid'] for s in streams}
    assert sids == {'ES@3min', 'NQ@5min'}                # only existing CSVs
    es = next(s for s in streams if s['sid'] == 'ES@3min')
    assert es['ohlcv'].shape == (500, 5) and es['ohlcv'].dtype == np.float32
    assert len(es['ts']) == 500


def test_load_ohlcv_physically_reads_only_requested_dates(tmp_path):
    _write_csv(tmp_path / 'ES_1D.csv', 1200, start='2018-01-01', freq='1D')
    streams = ssl_data.load_ohlcv(
        str(tmp_path), tickers=['ES'], tfs=['1D'], verbose=False,
        start='2019-07-01', end='2020-07-01', chunksize=73,
    )
    ts = pd.DatetimeIndex(streams[0]['ts'])
    assert ts.min() == pd.Timestamp('2019-07-01', tz='UTC')
    assert ts.max() == pd.Timestamp('2020-06-30', tz='UTC')
    assert len(ts) == 366


def test_time_split_excludes_holdout_and_is_causal():
    ts = pd.date_range('2024-01-01', periods=1000, freq='1D', tz='UTC')   # into 2026
    tr, va = ssl_data.time_split(ts, val_frac=0.2, holdout_start='2026-01-01')
    cut = pd.Timestamp('2026-01-01', tz='UTC')
    tsi = pd.DatetimeIndex(ts)
    assert (tsi[tr] < cut).all() and (tsi[va] < cut).all()    # 2026 never present
    assert tsi[tr].max() < tsi[va].min()                      # train strictly before val
    n_usable = int((tsi < cut).sum())
    assert len(va) == int(n_usable * 0.2)


def test_calendar_viability_filter_drops_only_sparse_groups_without_targets():
    from futures_foundation.finetune.ssl_probe import (
        calendar_viable_group_rows, walk_forward_splits)
    day = 24 * 60 * 60 * 10**9
    # Group 0 covers the calendar. Group 1 exists only at the end and cannot provide an earlier
    # training block. The filter receives no labels or model values.
    dense = np.arange(36, dtype=np.int64) * 10 * day
    sparse = np.arange(30, 36, dtype=np.int64) * 10 * day
    times = np.concatenate((dense, sparse))
    groups = np.concatenate((np.zeros(len(dense), np.int64),
                             np.ones(len(sparse), np.int64)))
    keep, excluded = calendar_viable_group_rows(times, groups, folds=3, span_ns=day)
    assert excluded == [1] and np.unique(groups[keep]).tolist() == [0]
    splits = walk_forward_splits(
        np.arange(len(keep)), groups[keep], folds=3, span=1,
        timestamps=times[keep], span_ns=day)
    assert len(splits) == 3 and all(len(train) and len(test) for train, test in splits)


def test_time_split_embargo_purges_train_and_holdout_boundaries():
    ts = pd.date_range('2024-01-01', periods=900, freq='1D', tz='UTC')
    tr, va = ssl_data.time_split(ts, val_frac=0.2, holdout_start='2026-01-01', embargo=32)
    usable = int((ts < pd.Timestamp('2026-01-01', tz='UTC')).sum())
    split = usable - int(usable * 0.2)
    assert tr[-1] == split - 33                       # 32 purged rows before val
    assert va[0] == split and va[-1] == usable - 33  # 32 purged rows before holdout
    assert va[-1] < usable <= 900


def test_time_split_explicit_v2_development_and_oos_dates():
    ts = pd.date_range('2023-01-01', '2025-08-01', freq='1D', tz='UTC')
    tr, va = ssl_data.time_split(ts, val_start='2024-01-01',
                                 holdout_start='2025-07-01', embargo=8)
    tsi = pd.DatetimeIndex(ts)
    assert (tsi[tr] < pd.Timestamp('2024-01-01', tz='UTC')).all()
    assert (tsi[va] >= pd.Timestamp('2024-01-01', tz='UTC')).all()
    assert (tsi[va] < pd.Timestamp('2025-07-01', tz='UTC')).all()
    assert tsi[tr[-1]] == pd.Timestamp('2023-12-23', tz='UTC')  # eight-row train/val purge
    assert tsi[va[-1]] == pd.Timestamp('2025-06-22', tz='UTC')  # eight-row val/OOS purge
    with pytest.raises(ValueError):
        ssl_data.time_split(ts, val_start='2025-07-01', holdout_start='2025-07-01')


def test_time_split_enforces_equal_history_lower_bound():
    ts = pd.date_range('2018-01-01', '2026-07-01', freq='1D', tz='UTC')
    tr, va = ssl_data.time_split(
        ts, train_start='2019-07-01', val_start='2024-07-01',
        holdout_start='2025-07-01', embargo=8,
    )
    tsi = pd.DatetimeIndex(ts)
    assert tsi[tr].min() == pd.Timestamp('2019-07-01', tz='UTC')
    assert tsi[tr].max() == pd.Timestamp('2024-06-22', tz='UTC')
    assert tsi[va].min() == pd.Timestamp('2024-07-01', tz='UTC')
    assert tsi[va].max() == pd.Timestamp('2025-06-22', tz='UTC')
    with pytest.raises(ValueError, match='train_start must precede'):
        ssl_data.time_split(ts, train_start='2024-07-01', val_start='2024-07-01')


def test_window_starts_contiguous():
    idx = np.arange(100)
    s = ssl_data.window_starts(idx, seq_total=10)
    assert len(s) == 91 and s[0] == 0 and s[-1] == 90
    gapped = np.concatenate([np.arange(0, 20), np.arange(50, 70)])   # a hole at 20..50
    sg = ssl_data.window_starts(gapped, seq_total=10)
    assert (((sg + 9 < 20) | (sg >= 50))).all()                      # no window spans the hole
    assert 11 not in sg and 60 in sg                                  # 11..19 can't fit; 50..60 can


def test_window_starts_rejects_time_gaps_and_contract_rolls():
    idx = np.arange(30)
    base_ts = pd.date_range('2024-01-01', periods=30, freq='3min', tz='UTC')
    ts = base_ts[:15].append(base_ts[15:] + pd.Timedelta('1h'))  # hole between rows 14/15
    contracts = np.array(['H24'] * 24 + ['M24'] * 6)  # roll between rows 23/24
    starts = ssl_data.window_starts(idx, 8, timestamps=ts, expected_delta='3min',
                                    segment_ids=contracts)
    assert 8 not in starts and 14 not in starts        # windows crossing the time gap rejected
    assert 17 not in starts and 23 not in starts       # windows touching the roll rejected
    assert 7 in starts and 15 in starts and 16 in starts  # windows ending before breaks stay usable


def test_window_starts_can_allow_only_bounded_aligned_session_gap():
    # Bridge a normal one-hour maintenance closure for 60m bars (observed delta=120m), but not
    # a weekend-sized hole. This exception is opt-in; exact cadence remains the default.
    ts = pd.DatetimeIndex([
        '2024-01-01 20:00Z', '2024-01-01 22:00Z', '2024-01-01 23:00Z',
        '2024-01-02 00:00Z', '2024-01-05 00:00Z', '2024-01-05 01:00Z',
    ])
    exact = ssl_data.window_starts(np.arange(6), 3, timestamps=ts, expected_delta='60min')
    allowed = ssl_data.window_starts(np.arange(6), 3, timestamps=ts, expected_delta='60min',
                                     max_gap='120min')
    assert 0 not in exact and 0 in allowed
    assert 2 not in allowed and 3 not in allowed


def test_assemble_windows_stay_within_stream(tmp_path):
    _write_csv(tmp_path / 'ES_3min.csv', 400)
    _write_csv(tmp_path / 'NQ_3min.csv', 400)
    streams = ssl_data.load_ohlcv(str(tmp_path), ['ES', 'NQ'], ['3min'], verbose=False)
    big, tr, va = ssl.assemble(streams, seq=32, max_jitter=8, val_frac=0.1,
                               holdout_start=None, verbose=False)
    assert big.shape == (800, 5)
    parent = 32 + 8
    bounds = [0, 400, 800]                                # stream boundaries
    for s in np.concatenate([tr, va]):
        # window [s, s+parent) must lie inside exactly one stream segment
        seg = 0 if s < 400 else 1
        assert bounds[seg] <= s and s + parent <= bounds[seg + 1]


def test_assemble_returns_contiguous_stream_group_bounds(tmp_path):
    _write_csv(tmp_path / 'ES_1D.csv', 700, start='2023-01-01', freq='1D')
    _write_csv(tmp_path / 'NQ_1D.csv', 700, start='2023-01-01', freq='1D')
    streams = ssl_data.load_ohlcv(str(tmp_path), ['ES', 'NQ'], ['1D'], verbose=False)
    _, tr, va, groups = ssl.assemble(
        streams, seq=16, max_jitter=4, val_frac=0.1, val_start='2024-01-01',
        holdout_start='2024-10-01', return_groups=True, verbose=False)
    for name, starts in [('train', tr), ('val', va)]:
        bounds = groups[f'{name}_bounds']
        assert bounds[0, 0] == 0 and bounds[-1, 1] == len(starts)
        assert np.array_equal(bounds[1:, 0], bounds[:-1, 1])
        assert tuple(groups[f'{name}_labels']) == ('ES@1D', 'NQ@1D')


def test_assemble_exposes_stream_and_contract_objective_segments():
    ts = pd.date_range('2023-01-01', periods=20, freq='1h', tz='UTC')
    streams = [
        {
            'sid': 'ES@1h', 'ticker': 'ES', 'tf': '1h', 'ts': ts,
            'ohlcv': np.ones((20, 5), np.float32),
            'contract_id': np.asarray(['ESH3'] * 12 + ['ESM3'] * 8),
        },
        {
            'sid': 'NQ@1h', 'ticker': 'NQ', 'tf': '1h', 'ts': ts,
            'ohlcv': np.ones((20, 5), np.float32), 'contract_id': None,
        },
    ]
    _, _, _, groups = ssl.assemble(
        streams, seq=3, max_jitter=0, val_frac=0.2, holdout_start=None,
        return_groups=True, verbose=False,
    )
    assert groups['objective_row_bounds'].tolist() == [[0, 12], [12, 20], [20, 40]]


def test_nextleg_reserve_covers_both_bounded_future_legs():
    reserve = NextLegTask().reserve({'context_lengths': (64, 200), 'leg_cap': 256})
    assert reserve == 200 + 2 * 256


@torch_test
def test_nextleg_targets_are_computed_independently_per_objective_segment(monkeypatch):
    from futures_foundation.finetune.pretext._torch import nextleg

    monkeypatch.setattr(
        nextleg, '_alternating_fractals',
        lambda high, low, k: [(1, 3, 1), (5, 7, -1), (8, 9, 1)],
    )
    big = np.ones((20, 5), np.float32)
    confirms, targets, valid = nextleg._leg_targets(
        big, 2, 10, np.asarray([[0, 10], [10, 20]]),
    )
    assert confirms.tolist() == [3, 13]
    np.testing.assert_allclose(np.expm1(targets), [[2, 3], [2, 3]])
    assert valid.tolist() == [True, True]


def test_assemble_returns_start_aligned_elapsed_time_metadata(tmp_path):
    _write_csv(tmp_path / 'ES_3min.csv', 900, start='2023-01-01', freq='3min')
    _write_csv(tmp_path / 'NQ_60min.csv', 900, start='2023-01-01', freq='60min')
    streams = ssl_data.load_ohlcv(str(tmp_path), ['ES', 'NQ'], ['3min', '60min'], verbose=False)
    _, tr, va, groups = ssl.assemble(
        streams, seq=16, max_jitter=4, val_frac=0.2, holdout_start=None,
        return_groups=True, verbose=False)
    assert len(groups['train_start_times_ns']) == len(tr)
    assert len(groups['val_start_times_ns']) == len(va)
    assert groups['stream_bar_ns'].tolist() == [3 * 60 * 10**9, 60 * 60 * 10**9]
    for split in ('train', 'val'):
        times = groups[f'{split}_start_times_ns']
        for lo, hi in groups[f'{split}_bounds']:
            assert (np.diff(times[lo:hi]) > 0).all()


def test_assemble_physically_excludes_oos_rows(tmp_path):
    _write_csv(tmp_path / 'ES_1D.csv', 1000, start='2023-01-01', freq='1D')
    streams = ssl_data.load_ohlcv(str(tmp_path), ['ES'], ['1D'], verbose=False)
    big, tr, va = ssl.assemble(
        streams, seq=8, max_jitter=2, val_frac=0.1, val_start='2024-01-01',
        holdout_start='2025-01-01', verbose=False)
    expected = int((pd.DatetimeIndex(streams[0]['ts']) <
                    pd.Timestamp('2025-01-01', tz='UTC')).sum())
    assert len(big) == expected
    assert (tr + 10 <= len(big)).all() and (va + 10 <= len(big)).all()


def test_balanced_probe_sample_caps_each_stream_equally():
    starts = np.arange(1010)
    bounds = np.array([[0, 1000], [1000, 1010]])
    out, group_ids = ssl._balanced_group_sample(
        starts, bounds, max_windows=100, seed=0, return_group_ids=True)
    assert (out < 1000).sum() == 50
    assert (out >= 1000).sum() == 10  # never duplicates a short stream merely to hit a quota
    assert (group_ids == 0).sum() == 50 and (group_ids == 1).sum() == 10


# ---------------------------------------------------------------- probe + gate (torch-free)
def test_targets_from_windows():
    seq = 8
    ramp = np.linspace(100, 107, seq)                    # pure uptrend
    chop = np.array([100, 101, 100, 101, 100, 101, 100, 101.0])
    def stk(close):
        return np.stack([close, close + 0.5, close - 0.5, close,
                         np.full(seq, 500.0)], 1).astype(np.float32)
    big = np.concatenate([stk(ramp), stk(chop)], 0)      # T=16, 5 cols
    t = ssl_probe.targets_from_windows(big, [0, 8], seq, fwd_k=4)
    assert t['trend_eff'][0] > 0.9 and t['trend_eff'][1] < 0.3   # trend vs chop
    assert t['direction'][0] == 1                                # net up
    assert set(t) == {'vol', 'trend_eff', 'range_expand', 'fwd_absmove',
                      'direction', 'fwd_dir'}                     # + forward buy/sell targets
    assert (t['fwd_absmove'] >= 0).all()


def test_probe_rows_never_share_context_or_forward_bars():
    starts = np.array([0, 1, 79, 80, 81, 160, 1000, 1001])
    rows = ssl_probe.non_overlapping_rows(starts, span=80)  # seq64 + fwd16
    used = starts[rows]
    assert np.array_equal(used, [0, 80, 160, 1000])
    assert (np.diff(used) >= 80).all()


def test_probe_walk_forward_is_stream_local_past_only_with_full_embargo():
    span = 80
    starts = np.concatenate([np.arange(24) * span,
                             100_000 + np.arange(24) * span]).astype(np.int64)
    groups = np.repeat([0, 1], 24)
    splits = ssl_probe.walk_forward_splits(starts, groups, folds=3, span=span)
    assert len(splits) == 3
    for tr, te in splits:
        assert not np.intersect1d(tr, te).size
        for group in (0, 1):
            gtr = tr[groups[tr] == group]
            gte = te[groups[te] == group]
            assert len(gtr) and len(gte)
            # Complete train span plus one full-span embargo precedes every test span.
            assert starts[gtr].max() + 2 * span <= starts[gte].min()


def test_probe_walk_forward_uses_shared_calendar_cutoffs_across_timeframes():
    hour = 3_600_000_000_000
    starts = np.arange(80, dtype=np.int64)
    groups = np.repeat([0, 1], 40)
    # Both streams cover the same wall-clock interval but have unrelated tensor offsets.
    times = np.concatenate([np.arange(40) * 24 * hour,
                            np.arange(40) * 24 * hour]).astype(np.int64)
    span_ns = 2 * hour
    splits = ssl_probe.walk_forward_splits(
        starts, groups, folds=3, span=1, timestamps=times, span_ns=span_ns)
    for tr, te in splits:
        first_test = times[te].min()
        assert (times[tr] + 2 * span_ns <= first_test).all()
        assert set(groups[tr]) == {0, 1} and set(groups[te]) == {0, 1}


def test_probe_embedding_recovers_signal():
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((400, 6)).astype(np.float32)
    y_reg = emb[:, 0] * 2.0 + rng.standard_normal(400) * 0.1       # linearly encoded
    assert ssl_probe.probe_embedding(emb, y_reg, 'reg', seed=0) > 0.8
    y_bin = (emb[:, 1] > 0).astype(int)
    assert ssl_probe.probe_embedding(emb, y_bin, 'bin', seed=0) > 0.85
    y_noise = rng.standard_normal(400)                            # unrelated
    assert ssl_probe.probe_embedding(emb, y_noise, 'reg', seed=0) < 0.2
    # k-fold CV path (folds>1) returns a valid averaged score in the same range
    assert ssl_probe.probe_embedding(emb, y_reg, 'reg', seed=0, folds=5) > 0.8
    assert ssl_probe.probe_embedding(emb, y_bin, 'bin', seed=0, folds=5) > 0.85


def test_probe_compare_flags_ssl_better():
    rng = np.random.default_rng(1)
    core = ['vol', 'trend_eff', 'range_expand', 'fwd_absmove']     # the gate's core targets
    tgt = {k: rng.standard_normal(300).astype(np.float32) for k in core}
    tgt['direction'] = rng.integers(0, 2, 300)
    tgt['fwd_dir'] = rng.integers(0, 2, 300)
    emb_ssl = (np.stack([tgt[k] for k in core], 1)
               + rng.standard_normal((300, 4)) * 0.05).astype(np.float32)   # encodes targets
    emb_van = rng.standard_normal((300, 4)).astype(np.float32)              # encodes nothing
    out = ssl_probe.compare(emb_ssl, emb_van, tgt, seed=0)
    assert out['learns_regime_vol_structure'] and out['mean_core_delta'] > 0


def _gate_probe(**overrides):
    deltas = {'vol': 0.02, 'trend_eff': 0.03, 'range_expand': 0.04,
              'fwd_absmove': 0.01, 'direction': 0.0, 'fwd_dir': 0.005}
    deltas.update(overrides)
    per = {name: {'ssl': 0.5 + delta, 'vanilla': 0.5, 'delta': delta,
                  'kind': ('bin' if name in ('direction', 'fwd_dir') else 'reg'),
                  'fold_delta': [delta] * 5}
           for name, delta in deltas.items()}
    core = np.mean([deltas[x] for x in ('vol', 'trend_eff', 'range_expand', 'fwd_absmove')])
    return {'per_target': per, 'mean_core_delta': float(core),
            'descriptive_delta': float(np.mean([deltas[x] for x in
                                                ('vol', 'trend_eff', 'range_expand')])),
            'fwd_absmove_delta': deltas['fwd_absmove'], 'fwd_dir_delta': deltas['fwd_dir'],
            'forward_score': deltas['fwd_absmove'] + deltas['fwd_dir'],
            'learns_regime_vol_structure': bool(core > 0)}


def test_passes_gate_on_probe_not_loss():
    # GATE = per-target probe content vs vanilla, not pretext loss or a mixed-metric average.
    good = _gate_probe()
    ok, d = ssl._passes(good, std=0.5)
    assert ok and d['learns_regime_vol_structure']
    # One material target regression fails even when the aggregate remains positive.
    bad = _gate_probe(vol=-0.01)
    assert not ssl._passes(bad, std=0.5)[0]
    # collapse -> fail regardless of probe
    assert not ssl._passes(good, std=0.001)[0]


def test_passes_forecast_gate_is_forward_move_centric_anti_shortcut():
    """Forecast gate requires forward move-size improvement; direction remains diagnostic."""
    # shortcut: big descriptive lift, but forward targets flat/negative -> FAIL on forecast...
    shortcut = _gate_probe(vol=0.10, trend_eff=0.10, range_expand=0.10,
                           fwd_absmove=0.0, fwd_dir=-0.02)
    assert not ssl._passes(shortcut, std=0.5, pretext='forecast')[0]
    # Mask can pass its structural gate; the forward-direction regression remains diagnostic.
    mask_ok, mask_detail = ssl._passes(shortcut, std=0.5, pretext='mask')
    assert mask_ok and not mask_detail['fwd_dir_ok']
    # genuine forward learning: move size up, direction not worse -> PASS on forecast
    genuine = _gate_probe(fwd_absmove=0.03, fwd_dir=0.01)
    ok, d = ssl._passes(genuine, std=0.5, pretext='forecast')
    assert ok and d['fwd_size_ok'] and d['fwd_dir_ok'] and d['descriptive_ok']
    # Direction is noisy at this sample size: report the regression, but do not gate on it.
    dir_reg = _gate_probe(fwd_absmove=0.03, fwd_dir=-0.01)
    ok, d = ssl._passes(dir_reg, std=0.5, pretext='forecast')
    assert ok and not d['fwd_dir_ok']
    assert d['diagnostic_target_checks']['fwd_dir']['gated'] is False
    # descriptive regresses -> FAIL
    desc_reg = _gate_probe(vol=-0.01, fwd_absmove=0.03, fwd_dir=0.01)
    assert not ssl._passes(desc_reg, std=0.5, pretext='forecast')[0]


def test_compare_exposes_forward_and_descriptive_deltas():
    """compare() splits descriptive vs forward content so the stage-2 gate can be anti-shortcut."""
    rng = np.random.default_rng(2)
    core = ['vol', 'trend_eff', 'range_expand', 'fwd_absmove']
    tgt = {k: rng.standard_normal(300).astype(np.float32) for k in core}
    tgt['direction'] = rng.integers(0, 2, 300)
    tgt['fwd_dir'] = rng.integers(0, 2, 300)
    emb = rng.standard_normal((300, 4)).astype(np.float32)
    out = ssl_probe.compare(emb, emb, tgt, seed=0)        # identical embeddings -> ~0 deltas
    for k in ('descriptive_delta', 'fwd_absmove_delta', 'fwd_dir_delta', 'forward_score'):
        assert k in out
    assert abs(out['forward_score'] - (out['fwd_absmove_delta'] + out['fwd_dir_delta'])) < 1e-9


def test_mantis_frozen_head_fit_predict():
    # the head-only path: a cheap head trains on (already-embedded) features; backbone frozen
    from futures_foundation.finetune.classifier import get_classifier
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    X = rng.standard_normal((400, 32)).astype(np.float32)
    X[y == 1, 0] += 3.0                                              # separable
    clf = get_classifier('mantis_frozen', head='logistic')
    pv, pe, auc = clf.fit_predict(X[:300], y[:300], X[300:], y[300:], X[300:], seed=0)
    assert auc > 0.9 and len(pv) == 100 and len(pe) == 100


# ---------------------------------------------- seq2seq forecast pretext: orchestration (torch-free)
def test_assemble_reserves_forecast_parent(tmp_path):
    """Stage-2 forecast needs (max context + max horizon) in-stream: assemble reserves
    max(seq+max_jitter, forecast_parent), and every window stays inside one stream."""
    _write_csv(tmp_path / 'ES_3min.csv', 400)
    _write_csv(tmp_path / 'NQ_3min.csv', 400)
    streams = ssl_data.load_ohlcv(str(tmp_path), ['ES', 'NQ'], ['3min'], verbose=False)
    seq, max_jitter, forecast_parent = 32, 8, 90            # 90 (=max_ctx+max_h) dominates 40
    big, tr, va = ssl.assemble(streams, seq=seq, max_jitter=max_jitter,
                               forecast_parent=forecast_parent,
                               val_frac=0.1, holdout_start=None, verbose=False)
    parent = max(seq + max_jitter, forecast_parent)        # 90
    assert parent == 90
    bounds = [0, 400, 800]
    for s in np.concatenate([tr, va]):
        seg = 0 if s < 400 else 1
        assert bounds[seg] <= s and s + parent <= bounds[seg + 1]   # context+horizon in-stream


def test_assemble_forecast_parent_zero_matches_mask(tmp_path):
    """forecast_parent=0 (mask/stage-1) reserves only seq+max_jitter — backward-compatible."""
    _write_csv(tmp_path / 'ES_3min.csv', 400)
    streams = ssl_data.load_ohlcv(str(tmp_path), ['ES'], ['3min'], verbose=False)
    _, tr0, _ = ssl.assemble(streams, seq=32, max_jitter=8, forecast_parent=0,
                             val_frac=0.1, holdout_start=None, verbose=False)
    _, tr_default, _ = ssl.assemble(streams, seq=32, max_jitter=8,
                                    val_frac=0.1, holdout_start=None, verbose=False)
    assert np.array_equal(tr0, tr_default)


def test_base_cfg_has_multihorizon_keys():
    cfg = ssl._base_cfg()
    assert cfg['pretext'] == 'mask' and cfg['backbone_ckpt'] is None
    assert cfg['horizons'] == (5, 10, 20, 25) and cfg['context_lengths'] == (64, 100, 150, 200)
    over = ssl._base_cfg(pretext='forecast', horizons=(5, 10), context_lengths=(64,),
                         backbone_ckpt='/x/enc.pt')
    assert over['pretext'] == 'forecast' and over['horizons'] == (5, 10)
    assert over['context_lengths'] == (64,) and over['backbone_ckpt'] == '/x/enc.pt'


def test_train_dispatches_on_pretext(monkeypatch):
    """_train routes to the forecast trainer iff pretext='forecast', else the masked trainer,
    and strips the 'pretext' key (not a trainer kwarg). Uses a fake _ssl_torch — no torch."""
    import sys, types
    calls = {}
    fake = types.ModuleType('futures_foundation.finetune._ssl_torch')

    def _mask(big, tr, va, control='real', **kw):
        calls['fn'] = 'mask'; calls['kw'] = kw; return ('mask_state', [])

    def _forecast(big, tr, va, control='real', **kw):
        calls['fn'] = 'forecast'; calls['kw'] = kw; return ('fc_state', [])
    fake.train_ssl_mask = _mask
    fake.train_ssl_forecast = _forecast
    monkeypatch.setitem(sys.modules, 'futures_foundation.finetune._ssl_torch', fake)

    cfg = ssl._base_cfg(pretext='forecast', horizons=(5, 10))
    st, _ = ssl._train(None, None, None, cfg, control='real')
    assert st == 'fc_state' and calls['fn'] == 'forecast'
    assert 'pretext' not in calls['kw'] and calls['kw']['horizons'] == (5, 10)

    cfg2 = ssl._base_cfg(pretext='mask')
    st2, _ = ssl._train(None, None, None, cfg2, control='real')
    assert st2 == 'mask_state' and calls['fn'] == 'mask' and 'pretext' not in calls['kw']


# ------------------------------------------ pretext-task registry (pluggable, no if-chains) — torch-free
def test_pretext_registry_resolves_all_tasks():
    """Every pretext resolves to its task; unknown fails fast; None -> mask (default)."""
    assert ssl.get_pretext('mask').__class__.__name__ == 'MaskTask'
    assert ssl.get_pretext('forecast').__class__.__name__ == 'ForecastTask'
    assert ssl.get_pretext('contrastive').__class__.__name__ == 'ContrastiveTask'
    assert ssl.get_pretext(None).name == 'mask'
    with pytest.raises(KeyError):
        ssl.get_pretext('does_not_exist')


def test_pretext_reserve_per_task():
    """Each task declares its own window reserve — no pretext if-chain in the orchestrator."""
    cfg = ssl._base_cfg(context_lengths=(64, 200), horizons=(5, 25))
    assert ssl.get_pretext('mask').reserve(cfg) == 0                 # stage-1: none
    assert ssl.get_pretext('forecast').reserve(cfg) == 200 + 25      # stage-2: ctx + horizon
    # Stage 2 v2 reserves its furthest elapsed-time context plus the positive context itself.
    assert ssl.get_pretext('contrastive').reserve(cfg) == 3 * cfg['seq']
    legacy = {**cfg, 'contrastive_objective': 'bar_offset_v1'}
    assert ssl.get_pretext('contrastive').reserve(legacy) == cfg['seq'] + max(cfg['pos_deltas'])


def test_contrastive_shifted_windows_stay_inside_split_and_stream(tmp_path):
    """Every anchor and its furthest positive must remain in the same temporal partition and
    stream. This locks the v2 context-relative reservation that prevents train->val and
    val->holdout contamination."""
    _write_csv(tmp_path / 'ES_3min.csv', 1000)
    _write_csv(tmp_path / 'NQ_3min.csv', 1000)
    streams = ssl_data.load_ohlcv(str(tmp_path), ['ES', 'NQ'], ['3min'], verbose=False)
    cfg = ssl._base_cfg(pretext='contrastive', seq=64, max_jitter=16,
                        pos_deltas=(2, 16, 64))
    parent = ssl.get_pretext('contrastive').reserve(cfg)
    assert parent == 192
    _, tr, va = ssl.assemble(streams, seq=cfg['seq'], max_jitter=cfg['max_jitter'],
                             forecast_parent=parent, val_frac=0.2,
                             holdout_start=None, verbose=False)
    # Per 1000-row stream: train=[0,800), val=[800,1000). Global second-stream offsets +1000.
    for starts, lo0, hi0, lo1, hi1 in (
        (tr, 0, 800, 1000, 1800),
        (va, 800, 1000, 1800, 2000),
    ):
        for s in starts:
            lo, hi = (lo0, hi0) if s < 1000 else (lo1, hi1)
            assert lo <= s and s + parent <= hi


def test_base_cfg_has_contrastive_keys():
    cfg = ssl._base_cfg()
    assert cfg['temperature'] == 0.1 and cfg['crop_max'] == 0.2 and cfg['proj_dim'] == 128
    assert cfg['pos_deltas'] == (2, 16, 64) and cfg['far_min'] == 512   # temporal knobs
    assert cfg['contrastive_objective'] == 'elapsed_time_v2'
    assert cfg['positive_gap_fractions'] == (0.6, 1.0, 2.0)
    assert cfg['max_positive_overlap'] == 0.5 and cfg['vol_weight'] == 0.0
    assert cfg['vicreg_invariance_weight'] == 25.0
    assert cfg['vicreg_variance_weight'] == 25.0
    assert cfg['vicreg_covariance_weight'] == 1.0
    assert cfg['vicreg_variance_target'] == 1.0
    over = ssl._base_cfg(pretext='contrastive', temperature=0.07, crop_max=0.1,
                         pos_deltas=(1, 8, 32), far_min=256, vol_weight=0.5)
    assert over['pretext'] == 'contrastive' and over['temperature'] == 0.07 and over['crop_max'] == 0.1
    assert over['pos_deltas'] == (1, 8, 32) and over['far_min'] == 256 and over['vol_weight'] == 0.5


def test_passes_contrastive_gate_report_only():
    """Contrastive uses the same per-target promotion schema as every other stage."""
    good = _gate_probe()
    ok, d = ssl._passes(good, std=0.5, pretext='contrastive')
    assert ok and d['descriptive_ok'] and d['no_collapse']
    assert not ssl._passes(good, std=0.001, pretext='contrastive')[0]        # collapse -> fail
    desc_reg = _gate_probe(vol=-0.01)
    assert not ssl._passes(desc_reg, std=0.5, pretext='contrastive')[0]      # regress -> fail


def test_train_dispatches_contrastive(monkeypatch):
    """_train routes pretext='contrastive' to train_ssl_contrastive via the task (no if-chain),
    stripping 'pretext'. Fake _ssl_torch — no torch."""
    import sys, types
    calls = {}
    fake = types.ModuleType('futures_foundation.finetune._ssl_torch')
    fake.train_ssl_contrastive = lambda big, tr, va, control='real', **kw: (
        calls.setdefault('fn', 'contrastive'), calls.setdefault('kw', kw), ('c_state', []))[-1]
    monkeypatch.setitem(sys.modules, 'futures_foundation.finetune._ssl_torch', fake)
    cfg = ssl._base_cfg(pretext='contrastive', temperature=0.05)
    st, _ = ssl._train(None, None, None, cfg, control='real')
    assert st == 'c_state' and calls['fn'] == 'contrastive'
    assert 'pretext' not in calls['kw'] and calls['kw']['temperature'] == 0.05


# ------------------- stage-2.5 forecast_dist (distributional refine on stage-2) — torch-free
def test_pretext_registry_resolves_forecast_dist():
    """forecast_dist is its OWN pretext (stage-2 untouched): same reserve as stage-2 (same
    targets), routed to its own trainer."""
    t = ssl.get_pretext('forecast_dist')
    assert t.__class__.__name__ == 'ForecastDistTask'
    assert t.trainer == 'train_ssl_forecast_dist'
    cfg = ssl._base_cfg(context_lengths=(64, 200), horizons=(5, 25))
    assert t.reserve(cfg) == 200 + 25


def test_train_dispatches_forecast_dist(monkeypatch):
    """_train routes pretext='forecast_dist' to train_ssl_forecast_dist (no if-chain), passing
    the distributional objective + weight through. Fake _ssl_torch — no torch."""
    import sys, types
    calls = {}
    fake = types.ModuleType('futures_foundation.finetune._ssl_torch')
    fake.train_ssl_forecast_dist = lambda big, tr, va, control='real', **kw: (
        calls.setdefault('fn', 'dist'), calls.setdefault('kw', kw), ('d_state', []))[-1]
    monkeypatch.setitem(sys.modules, 'futures_foundation.finetune._ssl_torch', fake)
    cfg = ssl._base_cfg(pretext='forecast_dist', objective='candle_bins', dir_weight=0.7)
    st, _ = ssl._train(None, None, None, cfg, control='real')
    assert st == 'd_state' and calls['fn'] == 'dist'
    assert 'pretext' not in calls['kw']
    assert calls['kw']['objective'] == 'candle_bins' and calls['kw']['dir_weight'] == 0.7


# ------------------------------------------------------------- masked-modeling trainer (gated)
@torch_test
def test_embed_windows_frozen():
    import numpy as _np
    from futures_foundation.finetune import _ssl_torch as S
    W = _np.random.default_rng(0).standard_normal((6, 5, 64)).astype(_np.float32)
    emb = S.embed_windows(W, ckpt=None, device='cpu')               # vanilla, encoder-only
    assert emb.shape[0] == 6 and emb.shape[1] > 0 and _np.isfinite(emb).all()


@pytest.mark.parametrize('seq', [64, 256])
@pytest.mark.parametrize('preprocessing', [
    'per_window_per_channel_zscore_v1', 'per_window_shared_ohlc_zscore_v1',
    'per_window_log_price_rel_volume_zscore_v1'])
@torch_test
def test_ssl_clean_embedding_matches_legacy_and_bundle_deployment(tmp_path, seq, preprocessing):
    import torch
    from futures_foundation.finetune import _ssl_torch as S
    from futures_foundation.finetune.pretext._torch.common import (
        preprocess_windows, make_deployment_bundle)
    rng = np.random.default_rng(seq)
    windows = rng.standard_normal((3, 5, seq)).astype(np.float32)
    net = S.MaskNetwork(C=5, new_channels=8, seq=seq).eval()
    with torch.no_grad():
        training_clean = net.embed(preprocess_windows(
            torch.from_numpy(windows), preprocessing)).numpy()
    state_path = tmp_path / f'legacy-{seq}.pt'
    bundle_path = tmp_path / f'bundle-{seq}.pt'
    torch.save(net.encoder.state_dict(), state_path)
    torch.save(make_deployment_bundle(
        net.encoder.state_dict(), model_id='paris-noah/Mantis-8M', model_version=None,
        channels=5, train_context_lengths=[seq], preprocessing=preprocessing), bundle_path)
    legacy = S.embed_windows(windows, ckpt=state_path, device='cpu',
                             preprocessing=preprocessing)
    bundled = S.embed_windows(windows, ckpt=bundle_path, device='cpu')
    assert np.allclose(training_clean, legacy, atol=1e-6, rtol=1e-6)
    assert np.allclose(training_clean, bundled, atol=1e-6, rtol=1e-6)


@torch_test
def test_shared_ohlc_preprocessing_preserves_candle_geometry():
    import torch
    from futures_foundation.finetune.pretext._torch.common import preprocess_windows
    raw = torch.tensor([[[10., 11., 12.], [13., 14., 15.], [8., 9., 10.],
                         [11., 12., 13.], [100., 150., 200.]]])
    z = preprocess_windows(raw, 'per_window_shared_ohlc_zscore_v1')
    # A shared affine transform preserves every within-candle price difference ratio.
    raw_ratio = (raw[:, 1] - raw[:, 0]) / (raw[:, 3] - raw[:, 0])
    z_ratio = (z[:, 1] - z[:, 0]) / (z[:, 3] - z[:, 0])
    assert torch.allclose(raw_ratio, z_ratio)
    assert torch.allclose(z[:, 4].mean(1), torch.zeros(1), atol=1e-6)


@torch_test
def test_log_price_preprocessing_preserves_return_amplitude_and_candle_geometry():
    import torch
    from futures_foundation.finetune.pretext._torch.common import preprocess_windows
    contract = 'per_window_log_price_rel_volume_zscore_v1'
    low_close = torch.tensor([100., 100.1, 100.2, 100.3])
    high_close = torch.tensor([100., 101., 102., 103.])
    windows = []
    for close in (low_close, high_close):
        windows.append(torch.stack((close - .05, close + .15, close - .15, close,
                                    torch.tensor([100., 120., 90., 150.]))))
    raw = torch.stack(windows)
    out = preprocess_windows(raw, contract)
    assert out[1, 3].diff().std() > 5 * out[0, 3].diff().std()
    # A common log reference retains exact log candle spreads and absolute price-level invariance.
    assert torch.allclose(out[:, 1] - out[:, 0],
                          torch.log(raw[:, 1] / raw[:, 0]), atol=1e-6)
    scaled = raw.clone(); scaled[:, :4] *= 7
    assert torch.allclose(out, preprocess_windows(scaled, contract), atol=1e-6)
    assert torch.allclose(out[:, 4].mean(1), torch.zeros(2), atol=1e-6)


@torch_test
def test_log_price_context_future_uses_context_only_volume_stats():
    import torch
    from futures_foundation.finetune.pretext._torch.common import preprocess_context_and_future
    contract = 'per_window_log_price_rel_volume_zscore_v1'
    ctx = torch.tensor([[[100., 101.], [101., 102.], [99., 100.], [100., 101.], [10., 20.]]])
    fut = torch.tensor([[[102.], [103.], [101.], [102.], [40.]]])
    cs, fs = preprocess_context_and_future(ctx, fut, contract)
    expected_close = torch.log(torch.tensor(102. / 100.))
    assert torch.allclose(fs[0, 3, 0], expected_close)
    cv = torch.log1p(ctx[:, 4:])
    expected_volume = ((torch.log1p(fut[:, 4:]) - cv.mean(2, keepdim=True)) /
                       cv.std(2, keepdim=True))
    assert torch.allclose(fs[:, 4:], expected_volume)


@torch_test
def test_bundle_preprocessing_cannot_be_overridden(tmp_path):
    import torch
    from futures_foundation.finetune import _ssl_torch as S
    from futures_foundation.finetune.pretext._torch.common import make_deployment_bundle
    net = S.MaskNetwork(C=5, seq=64)
    path = tmp_path / 'shared.bundle.pt'
    torch.save(make_deployment_bundle(
        net.encoder.state_dict(), model_id='paris-noah/Mantis-8M', model_version=None,
        channels=5, preprocessing='per_window_shared_ohlc_zscore_v1'), path)
    windows = np.random.default_rng(1).normal(size=(2, 5, 64)).astype(np.float32)
    with pytest.raises(ValueError, match='conflicts'):
        S.embed_windows(windows, ckpt=path, device='cpu',
                        preprocessing='per_window_per_channel_zscore_v1')


@torch_test
def test_deployment_loader_rejects_full_training_state(tmp_path):
    import torch
    from futures_foundation.finetune.pretext._torch.common import (
        TRAINING_STATE_SCHEMA, encoder_state_from_checkpoint)
    path = tmp_path / 'run.train.pt'
    torch.save({'schema_version': TRAINING_STATE_SCHEMA, 'model_state': {}}, path)
    with pytest.raises(ValueError, match='cannot be used for deployment'):
        encoder_state_from_checkpoint(path)


@torch_test
def test_mask_network_and_trainer(tmp_path):
    import torch
    from futures_foundation.finetune import _ssl_torch as S
    from futures_foundation.finetune.classifiers.mantis._torch import build_model
    net = S.MaskNetwork(C=5, new_channels=4, seq=64)
    x = torch.randn(8, 5, 64)
    assert net(x).shape == (8, 5, 64)                               # reconstruct full window
    assert net.embed(x).shape[0] == 8
    rng = np.random.default_rng(0)
    big = rng.standard_normal((2000, 5)).astype(np.float32)
    starts = np.arange(0, 1900, 4)
    state, hist = S.train_ssl_mask(big, starts, starts[-50:], seq=32, new_channels=4,
                                   mask_ratio=0.4, epochs=2, steps_per_epoch=3, batch=16,
                                   device='cpu', control='real', verbose=False)
    assert len(hist) >= 1 and np.isfinite(hist[-1]['val_loss']) and 'std' in hist[-1]
    anchored_state, anchored_hist = S.train_ssl_mask(
        big, starts, starts[-50:], seq=32, new_channels=4, mask_ratio=0.4,
        feature_anchor_weight=0.05, epochs=1, steps_per_epoch=1, batch=8,
        device='cpu', control='real', verbose=False)
    assert anchored_state and np.isfinite(anchored_hist[-1]['val_loss'])
    ckpt = str(tmp_path / 'enc.pt'); torch.save(state, ckpt)        # encoder ckpt round-trips
    _, new_c = build_model(5, new_channels=4, device='cpu', backbone_ckpt=ckpt)
    assert new_c == 4


@torch_test
def test_mantis_v2_mask_network_loads_matching_architecture():
    import torch
    from futures_foundation.finetune import _ssl_torch as S
    net = S.MaskNetwork(C=5, new_channels=3, seq=64,
                        model_id='paris-noah/MantisV2', model_version='v2')
    assert type(net.encoder).__name__ == 'MantisV2'
    assert sum(p.numel() for p in net.encoder.parameters()) == 4_188_672
    assert net(torch.randn(2, 5, 64)).shape == (2, 5, 64)


@torch_test
def test_mantis_v2_mask_teacher_anchor_builds_and_is_frozen():
    from futures_foundation.finetune.pretext._torch.mask import _MaskTrainer
    rng = np.random.default_rng(17)
    big = (100 + rng.standard_normal((160, 5)).cumsum(0) * .01).astype(np.float32)
    starts = np.arange(80)
    trainer = _MaskTrainer(
        big, starts, starts, seq=32, batch=2, epochs=1, steps_per_epoch=1,
        feature_anchor_weight=.05, model_id='paris-noah/MantisV2', model_version='v2',
        device='cpu', verbose=False)
    trainer.build_net()
    assert trainer.teacher is not None
    assert not any(parameter.requires_grad for parameter in trainer.teacher.parameters())
    assert np.isfinite(float(trainer.compute_loss(trainer.make_batch(trainer.tr)).detach()))


@torch_test
def test_group_sampler_balances_dense_and_sparse_streams():
    from futures_foundation.finetune.pretext._torch.mask import _MaskTrainer
    rng = np.random.default_rng(3)
    big = rng.standard_normal((200, 5)).astype(np.float32)
    starts = np.arange(110)
    # Stream 0 has 100 candidate windows, stream 1 only 10. Sampling must be ~50/50, not 91/9.
    trainer = _MaskTrainer(big, starts, starts, seq=16, new_channels=2, batch=20000,
                           epochs=1, steps_per_epoch=1, device='cpu', verbose=False,
                           train_group_bounds=np.array([[0, 100], [100, 110]]),
                           val_group_bounds=np.array([[0, 100], [100, 110]]))
    _, group = trainer.sample_start_indices(trainer.tr, return_groups=True)
    frac = float((group == 0).float().mean())
    assert 0.48 < frac < 0.52


@torch_test
def test_mask_validation_reuses_identical_windows_and_corruption():
    """Checkpoint selection must compare one fixed validation experiment across epochs."""
    import torch
    from futures_foundation.finetune.pretext._torch.mask import _MaskTrainer
    rng = np.random.default_rng(7)
    big = rng.standard_normal((1200, 5)).astype(np.float32)
    starts = np.arange(0, 1100)
    trainer = _MaskTrainer(big, starts[:800], starts[800:], seq=32, new_channels=3,
                           mask_ratio=0.4, epochs=1, steps_per_epoch=1, batch=16,
                           device='cpu', seed=11, verbose=False)
    trainer.build_net()
    gen_before = trainer.gen.get_state().clone()
    torch_before = torch.random.get_rng_state().clone()
    first_loss, first_extra = trainer.val_eval()
    second_loss, second_extra = trainer.val_eval()
    assert first_loss == second_loss and first_extra == second_extra
    assert torch.equal(trainer.gen.get_state(), gen_before)         # validation does not move train RNG
    assert torch.equal(torch.random.get_rng_state(), torch_before)


# --------------------------- multi-horizon / variable-context candle seq2seq trainer (gated)
@torch_test
def test_multihorizon_net_shape():
    import torch
    from futures_foundation.finetune import _ssl_torch as S
    net = S.MultiHorizonForecastNet(C=5, new_channels=4, horizons=(5, 10, 20))   # out=4 (OHLC)
    for L in (48, 96):                                              # variable context length works
        candles, aux = net(ctx := torch.randn(6, 5, L))            # net ALWAYS returns (candles, aux)
        assert candles.shape == (6, 5, 3)                          # [B, OHLCV=5, n_horizons]
        assert aux is None                                         # candle-only objective -> no aux head
        assert net.embed(ctx).shape[0] == 6 and net.embed(ctx).shape[1] > 0


def test_forecast_objective_registry_is_pluggable():
    """Torch-free: the forecast OBJECTIVE registry resolves by name (no if-chains), each declares its
    own aux_dim, and unknown names fail fast. candle_mse = candle-only; candle_direction = +nH logits."""
    from futures_foundation.finetune.pretext._torch.forecast_objectives import get_forecast_objective
    assert get_forecast_objective(None).name == 'candle_mse'       # default / backward-compat
    assert get_forecast_objective('candle_mse').aux_dim(4) == 0    # candle-only -> no aux head
    assert get_forecast_objective('candle_direction').aux_dim(4) == 4   # one direction logit / horizon
    try:
        get_forecast_objective('nope'); assert False, 'unknown objective must raise'
    except KeyError:
        pass


@torch_test
def test_forecast_direction_objective_aux_head_and_loss():
    """candle_direction adds a LINEAR aux head (nH logits) that shapes the encoder via BCE on sign(fwd
    close move); candle_mse keeps aux=None. Loss is finite for both -> the objective is wired end-to-end."""
    import torch
    from futures_foundation.finetune import _ssl_torch as S
    from futures_foundation.finetune.pretext._torch.forecast_objectives import get_forecast_objective
    net = S.MultiHorizonForecastNet(C=5, new_channels=4, horizons=(5, 10, 20), aux_dim=3)
    candles, aux = net(torch.randn(6, 5, 48))
    assert candles.shape == (6, 5, 3) and aux.shape == (6, 3)      # aux = per-horizon direction logits
    target = torch.randn(6, 5, 3)
    obj = get_forecast_objective('candle_direction')
    loss = obj.loss(candles, aux, target, close_ch=3, weight=0.5)
    assert torch.isfinite(loss) and loss.item() > 0


@torch_test
def test_train_multihorizon_runs_variable_context_and_warmstart(tmp_path):
    """Multi-horizon / variable-context candle trainer runs, returns finite val loss + skill,
    saves an encoder ckpt that loads downstream, and accepts a warm-start ckpt (from stage-1)."""
    import torch
    from futures_foundation.finetune import _ssl_torch as S
    from futures_foundation.finetune.classifiers.mantis._torch import build_model
    rng = np.random.default_rng(0)
    big = (100 + np.cumsum(rng.standard_normal((3000, 5)) * 0.1, 0)).astype(np.float32)
    big[:, 4] = np.abs(big[:, 4]) * 100 + 500                       # positive-ish volume
    hz, cl = (5, 10, 20), (32, 48)                                  # parent = 48 + 20 = 68
    starts = np.arange(0, 3000 - 68 - 1, 4)
    state, hist = S.train_ssl_forecast(big, starts, starts[-50:], horizons=hz, context_lengths=cl,
                                       new_channels=4, epochs=2, steps_per_epoch=3,
                                       batch=16, device='cpu', control='real', verbose=False)
    assert len(hist) >= 1 and np.isfinite(hist[-1]['val_loss']) and hist[-1]['std'] > 0
    assert 'persist_loss' in hist[-1] and hist[-1]['persist_loss'] > 0    # anti-shortcut baseline
    assert 'skill' in hist[-1] and np.isfinite(hist[-1]['skill'])
    ckpt = str(tmp_path / 'enc1.pt'); torch.save(state, ckpt)
    _, new_c = build_model(5, new_channels=4, device='cpu', backbone_ckpt=ckpt)
    assert new_c == 4
    state2, hist2 = S.train_ssl_forecast(big, starts, starts[-50:], horizons=hz, context_lengths=cl,
                                         new_channels=4, epochs=1, steps_per_epoch=2, batch=16,
                                         device='cpu', control='real', backbone_ckpt=ckpt, verbose=False)
    assert set(state2.keys()) == set(state.keys()) and np.isfinite(hist2[-1]['val_loss'])


@torch_test
def test_forecast_validation_reuses_identical_windows_and_context_lengths():
    """Forecast skill and checkpoint selection use fixed windows/contexts at every epoch."""
    import torch
    from futures_foundation.finetune.pretext._torch.forecast import _ForecastTrainer
    rng = np.random.default_rng(9)
    big = (100 + np.cumsum(rng.standard_normal((1800, 5)) * 0.1, axis=0)).astype(np.float32)
    big[:, 4] = np.abs(big[:, 4]) * 100 + 500
    starts = np.arange(0, 1700)
    trainer = _ForecastTrainer(big, starts[:1200], starts[1200:], horizons=(5, 10),
                               context_lengths=(32, 48), new_channels=3, epochs=1,
                               steps_per_epoch=1, batch=16, device='cpu', seed=13,
                               verbose=False)
    trainer.build_net()
    gen_before = trainer.gen.get_state().clone()
    torch_before = torch.random.get_rng_state().clone()
    first_loss, first_extra = trainer.val_eval()
    second_loss, second_extra = trainer.val_eval()
    assert first_loss == second_loss and first_extra == second_extra
    assert torch.equal(trainer.gen.get_state(), gen_before)
    assert torch.equal(torch.random.get_rng_state(), torch_before)






# --------------------- stage-2.5 distributional forecast objectives (torch, gated)
@torch_test
def test_dist_objectives_registry_and_aux_dims():
    """The DIST registry holds ONLY the distributional objectives (candle_mse stays in the stage-2
    registry — no cross-contamination); aux dims size the head per objective; the faithfulness
    knobs (bolt9 quantiles / finer bins / pure mse_weight=0) configure per instance with defaults
    matching the original refine-study behavior."""
    from futures_foundation.finetune import _ssl_torch as S
    q, b = S.get_dist_objective('candle_quantile'), S.get_dist_objective('candle_bins')
    assert q.aux_dim(4) == 4 * 2 and b.aux_dim(4) == 4 * 41          # original defaults
    assert S.get_dist_objective('candle_quantile', quantile_taus='bolt9').aux_dim(4) == 4 * 8
    assert S.get_dist_objective('candle_bins', bins_k=257).aux_dim(4) == 4 * 257
    assert S.get_dist_objective(None).name == 'candle_quantile'
    with pytest.raises(KeyError):
        S.get_dist_objective('candle_mse')


@torch_test
def test_dist_pure_mode_no_mse_anchor():
    """mse_weight=0 = PURE Chronos loss: the candle-MSE term contributes NOTHING — moving the
    candle prediction (with aux fixed) must not change the loss (only the median pinball reads
    the candle head in quantile mode; bins mode ignores candles entirely)."""
    import torch
    from futures_foundation.finetune import _ssl_torch as S
    torch.manual_seed(0)
    B, nH = 16, 4
    target = torch.randn(B, 5, nH)
    aux = torch.randn(B, nH * 41)
    pure = S.get_dist_objective('candle_bins', mse_weight=0.0)
    mixed = S.get_dist_objective('candle_bins', mse_weight=1.0)
    good_c, bad_c = target.clone(), target + 3.0
    # pure bins: candle head irrelevant -> identical loss; mixed: worse candles = higher loss
    assert float(pure.loss(good_c, aux, target, 3, 1.0)) == float(pure.loss(bad_c, aux, target, 3, 1.0))
    assert float(mixed.loss(bad_c, aux, target, 3, 1.0)) > float(mixed.loss(good_c, aux, target, 3, 1.0))


@torch_test
def test_candle_quantile_pinball_orders_quantiles():
    """Bolt-style pinball prefers correctly-bracketing quantiles (lo below truth, hi above) over
    inverted ones — the loss teaches the distribution's SPREAD, which plain MSE cannot."""
    import torch
    from futures_foundation.finetune import _ssl_torch as S
    obj = S.get_dist_objective('candle_quantile')
    torch.manual_seed(0)
    B, nH = 64, 4
    target = torch.randn(B, 5, nH)
    candles = target.clone()                              # perfect median isolates the aux term
    t = target[:, 3, :]
    good = torch.stack([t - 1.0, t + 1.0], -1).reshape(B, -1)   # lo < truth < hi
    bad = torch.stack([t + 1.0, t - 1.0], -1).reshape(B, -1)    # inverted bracket
    assert float(obj.loss(candles, good, target, 3, 1.0)) < float(obj.loss(candles, bad, target, 3, 1.0))
    # weight=0 defaults to 1.0 — no silent fall-through to plain MSE
    assert float(obj.loss(candles, bad, target, 3, 0.0)) == float(obj.loss(candles, bad, target, 3, 1.0))


@torch_test
def test_candle_bins_ce_rewards_true_bin():
    """Chronos-classic-style bin classification: logits peaked on the TRUE move bin lose less than
    logits peaked away from it — the head learns a per-horizon move DISTRIBUTION."""
    import torch
    from futures_foundation.finetune import _ssl_torch as S
    obj = S.get_dist_objective('candle_bins')
    torch.manual_seed(0)
    B, nH, K = 32, 4, obj.K
    target = torch.randn(B, 5, nH)
    candles = target.clone()
    edges = torch.linspace(-obj.BIN_RANGE, obj.BIN_RANGE, K + 1)[1:-1]
    idx = torch.bucketize(target[:, 3, :].contiguous(), edges)
    good = torch.full((B, nH, K), -5.0)
    good.scatter_(2, idx.unsqueeze(-1), 5.0)              # peaked ON the true bin
    bad = -good                                            # peaked everywhere BUT the true bin
    assert (float(obj.loss(candles, good.reshape(B, -1), target, 3, 1.0))
            < float(obj.loss(candles, bad.reshape(B, -1), target, 3, 1.0)))


@torch_test
def test_candle_mixture_nll_rewards_calibrated_density():
    """Moirai-style mixture NLL: parameters that place a tight, correctly-centered density on
    the true move lose less than mispeaked ones; pure mode (mse_weight=0) ignores the candle
    head; gradients are finite (softplus/clamp stability guards)."""
    import torch
    from futures_foundation.finetune import _ssl_torch as S
    obj = S.get_dist_objective('candle_mixture')
    assert obj.aux_dim(4) == 4 * 9
    torch.manual_seed(0)
    B, nH = 32, 4
    target = torch.randn(B, 5, nH)
    candles = target.clone()
    t = target[:, 3, :]

    def params(center):
        p = torch.zeros(B, nH, 9)
        p[..., 0:3] = torch.tensor([0.0, 0.0, 4.0])       # weight the low-variance component
        p[..., 3] = center; p[..., 6] = center; p[..., 8] = center
        return p.reshape(B, -1)

    good = params(t)                                       # density centered ON the true move
    bad = params(t + 3.0)                                  # mispeaked by 3 sigma
    assert float(obj.loss(candles, good, target, 3, 1.0)) < float(obj.loss(candles, bad, target, 3, 1.0))
    # pure mode: candle head contributes nothing
    pure = S.get_dist_objective('candle_mixture', mse_weight=0.0)
    assert float(pure.loss(candles, bad, target, 3, 1.0)) == float(pure.loss(candles + 5, bad, target, 3, 1.0))
    # finite gradients through softplus/df/logsumexp
    aux = torch.randn(B, nH * 9, requires_grad=True)
    obj.loss(candles, aux, target, 3, 1.0).backward()
    assert torch.isfinite(aux.grad).all()


@torch_test
def test_candle_mixture_collapse_guards():
    """Anti-collapse guards: (1) the load-balance penalty PENALIZES a mixture that puts all weight
    on one component vs a balanced one; (2) diagnostics EXPOSE collapse — mix_entropy ~0 for a
    one-component mixture, ~1 for uniform; mix_mean_df is finite/read."""
    import torch
    from futures_foundation.finetune import _ssl_torch as S
    obj = S.get_dist_objective('candle_mixture', mse_weight=0.0, balance_w=0.1)
    B, nH = 64, 4
    target = torch.zeros(B, 5, nH)
    candles = target.clone()

    def aux_with_weights(logits3):
        p = torch.zeros(B, nH, 9)
        p[..., 0:3] = torch.tensor(logits3, dtype=torch.float)
        return p.reshape(B, -1)

    collapsed = aux_with_weights([9.0, -9.0, -9.0])       # all weight on component 0
    balanced = aux_with_weights([0.0, 0.0, 0.0])          # uniform
    # the balance penalty makes the collapsed mixture's loss strictly higher (same densities)
    assert float(obj.loss(candles, collapsed, target, 3, 1.0)) > float(obj.loss(candles, balanced, target, 3, 1.0))
    # diagnostics see it: entropy ~0 collapsed, ~1 uniform
    dc = obj.diagnostics(collapsed, target, 3)
    db = obj.diagnostics(balanced, target, 3)
    assert dc['mix_entropy'] < 0.1 and db['mix_entropy'] > 0.95
    assert np.isfinite(dc['mix_mean_df']) and np.isfinite(db['mix_mean_df'])
    # BATCH-MEAN entropy (not per-sample): CONFIDENT PER-SAMPLE ROUTING (different samples pick
    # different components) is HEALTHY, not collapse — must read HIGH entropy. Half the batch
    # hard-routes to comp 0, half to comp 1 -> balanced batch-mean despite one-hot per sample.
    routed = torch.zeros(B, nH, 9)
    routed[:B // 2, :, 0:3] = torch.tensor([9.0, -9.0, -9.0])
    routed[B // 2:, :, 0:3] = torch.tensor([-9.0, 9.0, -9.0])
    assert obj.diagnostics(routed.reshape(B, -1), target, 3)['mix_entropy'] > 0.5


@torch_test
def test_forecast_dist_trainer_smoke():
    """train_ssl_forecast_dist runs end-to-end via the subclassed trainer (aux head sized by the
    swapped objective; forecast.py untouched) -> encoder state + the same comparable metrics as
    stage-2 (skill / dir_acc / std)."""
    import numpy as np
    from futures_foundation.finetune import _ssl_torch as S
    rng = np.random.default_rng(0)
    big = (100 + np.cumsum(rng.standard_normal((3000, 5)) * 0.1, 0)).astype(np.float32)
    starts = np.arange(0, 3000 - (48 + 8) - 1, 4)
    state, hist = S.train_ssl_forecast_dist(big, starts, starts[-50:], horizons=(4, 8),
                                            context_lengths=(32, 48), new_channels=4,
                                            objective='candle_quantile', epochs=2,
                                            steps_per_epoch=3, batch=16, device='cpu',
                                            control='real', verbose=False)
    assert len(hist) >= 1 and np.isfinite(hist[-1]['val_loss'])
    assert 'skill' in hist[-1] and 'dir_acc' in hist[-1] and hist[-1]['std'] > 0


# --------------------------- stage-3 temporal-neighborhood contrastive (torch, gated)
@torch_test
def test_contrastive_snap_and_sigma():
    """_snap_to_starts finds the nearest valid start (boundary-safe); _vol_sigma orders a calm
    window below a chaotic one (the data-driven down-weighting signal)."""
    import torch
    from futures_foundation.finetune.pretext._torch.contrastive import _snap_to_starts, _vol_sigma
    starts = torch.tensor([0, 10, 20, 30, 100, 110], dtype=torch.long)
    s, d = _snap_to_starts(starts, torch.tensor([12, 95, 40], dtype=torch.long))
    assert s.tolist() == [10, 100, 30] and d.tolist() == [2, 5, 10]
    calm = torch.full((1, 5, 64), 100.0); calm[0, 3, :] += torch.linspace(0, 0.5, 64)
    wild = torch.full((1, 5, 64), 100.0); wild[0, 3, ::2] += 3.0
    assert float(_vol_sigma(calm)) < float(_vol_sigma(wild))


@torch_test
def test_elapsed_time_pairs_scale_by_stream_clock_and_enforce_overlap():
    from futures_foundation.finetune.pretext._torch.contrastive import _ContrastiveTrainer
    minute = 60 * 10**9
    rng = np.random.default_rng(22)
    big = rng.standard_normal((1300, 5)).astype(np.float32)
    starts = np.concatenate([np.arange(0, 220), np.arange(1000, 1220)])
    bounds = np.array([[0, 220], [220, 440]])
    times = np.concatenate([np.arange(220) * minute, np.arange(220) * 60 * minute])
    trainer = _ContrastiveTrainer(
        big, starts, starts, seq=16, positive_gap_fractions=(0.6, 1.0, 2.0),
        max_positive_overlap=0.5, new_channels=2, proj_dim=8, batch=512,
        epochs=1, steps_per_epoch=1, device='cpu', verbose=False,
        train_group_bounds=bounds, val_group_bounds=bounds,
        train_start_times_ns=times, val_start_times_ns=times,
        stream_bar_ns=np.array([minute, 60 * minute]))
    batch = trainer.make_batch(trainer.tr)
    ok, anchor_times, positive_times, context_ns = batch[2], batch[7], batch[8], batch[9]
    actual = (positive_times - anchor_times[None, :]).abs().float()
    overlap = (1.0 - actual / context_ns[None, :].float()).clamp(0, 1)
    assert ok.any() and float(overlap[ok].max()) <= 0.5 + 1e-6
    # The same relative scale maps to different real elapsed gaps for 1m and 60m streams.
    groups = batch[6]
    assert float(actual[1, groups == 1].median()) > 50 * float(actual[1, groups == 0].median())


@torch_test
def test_contrastive_augmentation_is_per_observation_in_v2():
    import torch
    from futures_foundation.finetune.pretext._torch.contrastive import _augment
    x = torch.ones(32, 5, 64)
    gen = torch.Generator().manual_seed(3)
    _, params = _augment(x, gen, noise=0, scale=0, tmask=0.15, crop_max=0.2,
                         independent=True, return_params=True)
    assert params['crop_start'].unique().numel() > 1
    assert params['mask_start'].unique().numel() > 1
    _, legacy = _augment(x, gen, noise=0, scale=0, tmask=0.15, crop_max=0.2,
                         independent=False, return_params=True)
    assert legacy['crop_start'].unique().numel() == 1
    assert legacy['mask_start'].unique().numel() == 1


@torch_test
def test_elapsed_objective_never_downweights_high_volatility():
    import torch
    from futures_foundation.finetune.pretext._torch.contrastive import _ContrastiveTrainer
    big = np.random.default_rng(23).standard_normal((300, 5)).astype(np.float32)
    starts = np.arange(200)
    v2 = _ContrastiveTrainer(big, starts, starts, seq=16, batch=8, epochs=1,
                             steps_per_epoch=1, device='cpu', verbose=False, vol_weight=1.0)
    legacy = _ContrastiveTrainer(
        big, starts, starts, seq=16, batch=8, epochs=1, steps_per_epoch=1,
        device='cpu', verbose=False, contrastive_objective='bar_offset_v1', vol_weight=1.0)
    sigma = torch.tensor([0.1, 1.0, 10.0])
    assert torch.equal(v2._sigma_weights(sigma), torch.ones(3))
    assert legacy._sigma_weights(sigma)[0] > legacy._sigma_weights(sigma)[-1]


@torch_test
def test_vicreg_loss_is_finite_noncollapsed_and_differentiable():
    import torch
    from futures_foundation.finetune.pretext._torch.contrastive import _vicreg_loss
    z1 = torch.randn(16, 12, requires_grad=True)
    z2 = (z1.detach() + .05 * torch.randn(16, 12)).requires_grad_()
    loss, diag = _vicreg_loss(z1, z2)
    assert torch.isfinite(loss) and loss > 0
    assert set(diag) == {'vicreg_invariance', 'vicreg_variance', 'vicreg_covariance'}
    loss.backward()
    assert z1.grad is not None and z2.grad is not None
    collapsed = torch.zeros(16, 12)
    collapsed_loss, collapsed_diag = _vicreg_loss(collapsed, collapsed)
    assert torch.isfinite(collapsed_loss)
    assert collapsed_diag['vicreg_variance'] > diag['vicreg_variance']


@torch_test
def test_vicreg_trainer_uses_same_context_two_views_and_frozen_teacher():
    from futures_foundation.finetune.pretext._torch.contrastive import _ContrastiveTrainer
    rng = np.random.default_rng(27)
    big = (100 + rng.standard_normal((500, 5)).cumsum(0) * .01).astype(np.float32)
    starts = np.arange(300)
    trainer = _ContrastiveTrainer(
        big, starts, starts, seq=32, batch=8, epochs=1, steps_per_epoch=1,
        contrastive_objective='vicreg_v1', feature_anchor_weight=.1,
        model_id='paris-noah/MantisV2', model_version='v2',
        device='cpu', verbose=False)
    trainer.build_net()
    batch = trainer.make_batch(trainer.tr)
    assert batch[1] == [] and batch[2].shape == (0, 8) and batch[3].shape == (0, 8)
    assert trainer.teacher is not None
    assert not any(parameter.requires_grad for parameter in trainer.teacher.parameters())
    loss = trainer.compute_loss(batch)
    assert np.isfinite(float(loss.detach()))
    assert trainer._last_loss_diagnostics['valid_negatives_mean'] == 0.0


@torch_test
def test_synchronized_cross_stream_rows_are_not_false_negatives():
    import torch
    import torch.nn.functional as F
    from futures_foundation.finetune.pretext._torch.contrastive import _weighted_supcon
    minute = 60 * 10**9
    z = F.normalize(torch.tensor([[1., 0], [1., 0], [0., 1], [0., 1]]), dim=1)
    group = torch.tensor([0, 0, 1, 1]); ok = torch.ones(4, dtype=torch.bool)
    positions = torch.arange(4); weights = torch.ones(4)
    streams = torch.tensor([0, 0, 1, 1]); contexts = torch.full((4,), 10 * minute)
    synced = torch.tensor([0, 0, 30 * minute, 30 * minute])
    loss, diag = _weighted_supcon(
        z, group, ok, positions, weights, 0.1, 64, stream_ids=streams,
        timestamps_ns=synced, context_ns=contexts, negative_min_contexts=4,
        sync_exclusion_ns=60 * minute, return_diagnostics=True)
    assert float(loss) == 0.0 and diag['valid_rows_fraction'] == 0.0
    assert diag['sync_excluded_fraction'] == 1.0
    far = torch.tensor([0, 0, 120 * minute, 120 * minute])
    loss, diag = _weighted_supcon(
        z, group, ok, positions, weights, 0.1, 64, stream_ids=streams,
        timestamps_ns=far, context_ns=contexts, negative_min_contexts=4,
        sync_exclusion_ns=60 * minute, return_diagnostics=True)
    assert torch.isfinite(loss) and diag['valid_rows_fraction'] == 1.0


@torch_test
def test_contrastive_positives_never_snap_across_stream_groups():
    from futures_foundation.finetune.pretext._torch.contrastive import _ContrastiveTrainer
    rng = np.random.default_rng(21)
    big = rng.standard_normal((1200, 5)).astype(np.float32)
    starts = np.concatenate([np.arange(0, 100), np.arange(1000, 1100)])
    bounds = np.array([[0, 100], [100, 200]])
    trainer = _ContrastiveTrainer(
        big, starts, starts, seq=16, pos_deltas=(4, 16, 64), new_channels=2,
        proj_dim=8, batch=512, epochs=1, steps_per_epoch=1, device='cpu', verbose=False,
        train_group_bounds=bounds, val_group_bounds=bounds)
    batch = trainer.make_batch(trainer.tr)
    pos_s, stream_groups = batch[3], batch[6]
    for positives in pos_s:
        assert bool(((stream_groups == 0) == (positives < 1000)).all())


@torch_test
def test_weighted_supcon_prefers_temporal_grouping():
    """The sigma-weighted SupCon gives LOWER loss when same-group (anchor views + temporal
    positives) embeddings are aligned than anti-aligned, and EXCLUDES near-but-not-positive
    pairs from the denominator (they are neither pulled nor pushed)."""
    import torch
    import torch.nn.functional as F
    from futures_foundation.finetune.pretext._torch.contrastive import _weighted_supcon
    group = torch.tensor([0, 0, 1, 1])
    ok = torch.ones(4, dtype=torch.bool)
    positions = torch.tensor([0, 2, 5000, 5002])
    w = torch.ones(4)
    aligned = F.normalize(torch.tensor([[1., 0], [1, 0], [0, 1.], [0, 1]]), dim=1)
    anti = F.normalize(torch.tensor([[1., 0], [-1, 0], [0, 1.], [0, -1]]), dim=1)
    la = _weighted_supcon(aligned, group, ok, positions, w, 0.1, far_min=64)
    lb = _weighted_supcon(anti, group, ok, positions, w, 0.1, far_min=64)
    assert torch.isfinite(la) and float(la) < float(lb)


@torch_test
def test_contrastive_net_shape_and_trainer_smoke(tmp_path):
    """ContrastiveTrendNet -> L2-normalized [B, proj_dim]; the temporal trainer runs end-to-end,
    reports the spec's A-E regime metrics, returns an encoder state loadable downstream, and
    accepts a warm-start ckpt; regime_gate evaluates the metrics dict."""
    import torch
    from futures_foundation.finetune import _ssl_torch as S
    from futures_foundation.finetune.classifiers.mantis._torch import build_model
    net = S.ContrastiveTrendNet(C=5, new_channels=4, proj_dim=64).to('cpu')
    z = net(torch.randn(6, 5, 64))
    assert z.shape == (6, 64) and torch.allclose(z.norm(dim=1), torch.ones(6), atol=1e-4)
    rng = np.random.default_rng(0)
    big = (100 + np.cumsum(rng.standard_normal((3000, 5)) * 0.1, 0)).astype(np.float32)
    big[:, 4] = np.abs(big[:, 4]) * 100 + 500                      # positive-ish volume
    starts = np.arange(0, 3000 - 64 - 32 - 1, 2)                   # reserve = max delta (32)
    state, hist = S.train_ssl_contrastive(big, starts, starts[-120:], seq=64,
                                          pos_deltas=(2, 8, 32), far_min=128, metrics_n=48,
                                          new_channels=4, proj_dim=64, epochs=1,
                                          steps_per_epoch=2, batch=8, device='cpu',
                                          control='real', verbose=False)
    assert len(hist) >= 1 and np.isfinite(hist[-1]['val_loss']) and hist[-1]['std'] > 0
    for k in ('smooth', 'sil', 'scale_span', 'scale_mono', 'vol_ratio', 'drift'):
        assert k in hist[-1]                                       # the spec's A-E metrics
    ok, checks = S.regime_gate(hist[-1])
    assert set(checks) == {'A_temporal_consistency', 'B_emergent_structure', 'C_multi_scale',
                           'D_noise_robustness', 'E_temporal_stability'}
    ckpt = str(tmp_path / 'enc.pt'); torch.save(state, ckpt)
    _, new_c = build_model(5, new_channels=4, device='cpu', backbone_ckpt=ckpt)
    assert new_c == 4                                              # encoder ckpt loads downstream
    state2, _ = S.train_ssl_contrastive(big, starts, starts[-120:], seq=64,
                                        pos_deltas=(2, 8, 32), far_min=128, metrics_n=48,
                                        new_channels=4, proj_dim=64, epochs=1, steps_per_epoch=1,
                                        batch=8, device='cpu', control='real',
                                        backbone_ckpt=ckpt, verbose=False)
    assert set(state2.keys()) == set(state.keys())                # warm-start same encoder keys


# --------------------------------------------- save/resume + anti-forgetting freeze (all pretexts)
def test_base_cfg_has_ckpt_resume_freeze_keys():
    cfg = ssl._base_cfg()
    assert cfg['ckpt_path'] is None and cfg['resume'] is False and cfg['freeze_encoder_layers'] == 0
    over = ssl._base_cfg(resume=True, freeze_encoder_layers=4)
    assert over['resume'] is True and over['freeze_encoder_layers'] == 4


@torch_test
def test_freeze_encoder_layers_anchors_early_leaves_objective_head_trainable():
    from futures_foundation.finetune.pretext._torch.common import _freeze_encoder
    from futures_foundation.finetune.pretext._torch.contrastive import ContrastiveTrendNet
    net = ContrastiveTrendNet(C=5, new_channels=4, proj_dim=32).to('cpu')
    before = sum(p.requires_grad for p in net.parameters())
    n = _freeze_encoder(net.encoder, 4)
    after = sum(p.requires_grad for p in net.parameters())
    assert n == 4 and after < before                          # froze tokenizer + first 4 blocks
    assert not hasattr(net, 'adapter')                              # no train/deploy skew
    assert any(p.requires_grad for p in net.prj.parameters())
    assert _freeze_encoder(net.encoder, 0) == 0               # n<=0 -> no-op


@torch_test
def test_contrastive_save_resume_and_control_guard(tmp_path):
    import os
    import torch
    from futures_foundation.finetune import _ssl_torch as S
    rng = np.random.default_rng(0)
    big = (100 + np.cumsum(rng.standard_normal((3000, 5)) * 0.1, 0)).astype(np.float32)
    big[:, 4] = np.abs(big[:, 4]) * 100 + 500
    # The furthest positive starts at anchor+64 and consumes seq=64 more bars. Keep train/val
    # disjoint and leave enough tail for every shifted positive in both partitions.
    train_starts = np.arange(0, 2200, 4)
    val_starts = np.arange(2400, 3000 - (64 + 64) + 1, 4)
    ck = str(tmp_path / 'enc.pt')
    st, _ = S.train_ssl_contrastive(big, train_starts, val_starts,
                                    new_channels=4, proj_dim=32, epochs=2, steps_per_epoch=3,
                                    batch=16, device='cpu', control='real', ckpt_path=ck, verbose=False)
    assert os.path.exists(ck) and os.path.exists(ck + '.meta.json')
    assert os.path.exists(ck + '.train.pt')                             # full exact-resume state
    st2, _ = S.train_ssl_contrastive(big, train_starts, val_starts,
                                     new_channels=4, proj_dim=32, epochs=2, steps_per_epoch=3,
                                     batch=16, device='cpu', control='real', ckpt_path=ck,
                                     resume=True, verbose=False)
    assert all(torch.equal(st2[k], st[k]) for k in st)                  # exact completed restore
    with pytest.raises(ValueError, match='resume configuration differs'):
        S.train_ssl_contrastive(
            big, train_starts, val_starts, new_channels=4, proj_dim=32, epochs=2,
            steps_per_epoch=3, batch=16, device='cpu', control='real', ckpt_path=ck,
            resume=True, contrastive_objective='bar_offset_v1', verbose=False)
    before = os.path.getmtime(ck)                                       # controls must NOT touch ckpt
    S.train_ssl_contrastive(big, train_starts, val_starts,
                            new_channels=4, proj_dim=32, epochs=1, steps_per_epoch=2, batch=16,
                            device='cpu', control='shuffle', ckpt_path=ck, verbose=False)
    assert os.path.getmtime(ck) == before                              # shuffle control didn't save


@torch_test
def test_exact_resume_matches_uninterrupted_trajectory(tmp_path):
    import torch
    import torch.nn as nn
    from futures_foundation.finetune.pretext._torch.common import BaseTrainer

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(3, 4)
            self.head = nn.Linear(4, 1)

        def forward(self, x):
            return self.head(torch.tanh(self.encoder(x)))

    class Trainer(BaseTrainer):
        def build_net(self):
            self.net = Net().to(self.dev)

        def make_batch(self, starts):
            idx = self.sample_start_indices(starts)
            return self.big_t[starts[idx]]

        def compute_loss(self, batch):
            return self.net(batch).square().mean()

        @torch.no_grad()
        def val_eval(self):
            self.net.eval()
            with self.fixed_validation_rng():
                loss = float(self.compute_loss(self.make_batch(self.va)))
            self.net.train()
            return loss, {'std': 1.0}

    big = np.random.default_rng(9).standard_normal((128, 3)).astype(np.float32)
    starts = np.arange(len(big))
    common = dict(epochs=4, steps_per_epoch=3, batch=16, lr=2e-3, patience=10,
                  device='cpu', seed=17, verbose=False)
    full_path = str(tmp_path / 'full.pt')
    resumed_path = str(tmp_path / 'resumed.pt')
    full_best, full_hist = Trainer(big, starts[:96], starts[96:], ckpt_path=full_path,
                                   **common).fit()
    Trainer(big, starts[:96], starts[96:], ckpt_path=resumed_path,
            stop_after_epoch=1, **common).fit()
    resumed_best, resumed_hist = Trainer(big, starts[:96], starts[96:], ckpt_path=resumed_path,
                                         resume=True, **common).fit()
    full_state = torch.load(full_path + '.train.pt', weights_only=False)
    resumed_state = torch.load(resumed_path + '.train.pt', weights_only=False)
    assert full_hist == resumed_hist
    assert all(torch.equal(full_best[k], resumed_best[k]) for k in full_best)
    assert all(torch.equal(full_state['model_state'][k], resumed_state['model_state'][k])
               for k in full_state['model_state'])
    assert full_state['scheduler_state'] == resumed_state['scheduler_state']


# ---------------------------------------------- stage-2 forecast: optional DIRECTION-head squeeze
def test_base_cfg_has_direction_keys():
    cfg = ssl._base_cfg()
    assert cfg['dir_weight'] == 0.0 and cfg['dir_close_ch'] == 3      # off by default (backward-compat)
    assert ssl._base_cfg(dir_weight=0.5)['dir_weight'] == 0.5


@torch_test
def test_forecast_direction_head_optional_and_backcompat():
    """Net ALWAYS returns (candles, aux) (no forward if-chain): candle-only objective -> aux=None
    (backward-compat); candle_direction (aux_dim=nH) -> aux = per-horizon dir logits; the trainer with
    objective='candle_direction' + dir_weight>0 reports a val 'dir_acc'."""
    import torch
    from futures_foundation.finetune import _ssl_torch as S
    # backward-compat: default objective is candle-only -> aux is None
    candles0, aux0 = S.MultiHorizonForecastNet(C=5, new_channels=4, horizons=(5, 10, 20))(torch.randn(6, 5, 64))
    assert candles0.shape == (6, 5, 3) and aux0 is None
    # direction objective: aux head sized to nH -> (candles, dir_logits)
    netd = S.MultiHorizonForecastNet(C=5, new_channels=4, horizons=(5, 10, 20), aux_dim=3).to('cpu')
    candles, dir_logits = netd(torch.randn(6, 5, 64))
    assert candles.shape == (6, 5, 3) and dir_logits.shape == (6, 3)
    # trainer with the direction objective runs + reports dir_acc in history
    rng = np.random.default_rng(0)
    big = (100 + np.cumsum(rng.standard_normal((3000, 5)) * 0.1, 0)).astype(np.float32)
    big[:, 4] = np.abs(big[:, 4]) * 100 + 500
    hz, cl = (5, 10, 20), (32, 48); starts = np.arange(0, 3000 - 68 - 1, 4)
    _, hist = S.train_ssl_forecast(big, starts, starts[-50:], horizons=hz, context_lengths=cl,
                                   new_channels=4, epochs=2, steps_per_epoch=3, batch=16, device='cpu',
                                   control='real', objective='candle_direction', dir_weight=0.5, verbose=False)
    assert 'dir_acc' in hist[-1] and 0.0 <= hist[-1]['dir_acc'] <= 1.0
    assert np.isfinite(hist[-1]['val_loss']) and 'skill' in hist[-1]    # candle metrics still there
