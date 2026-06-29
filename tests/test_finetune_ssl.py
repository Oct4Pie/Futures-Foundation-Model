"""SSL contrastive pretraining — torch-free data/assembly/verdict (non-gated) +
torch trainer (gated behind CHRONOS_TORCH_TESTS=1, libomp isolation).

Run torch parts: CHRONOS_TORCH_TESTS=1 pytest tests/test_finetune_ssl.py
"""
import os

import numpy as np
import pandas as pd
import pytest

from futures_foundation.finetune import ssl, ssl_data, ssl_probe

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


def test_time_split_excludes_holdout_and_is_causal():
    ts = pd.date_range('2024-01-01', periods=1000, freq='1D', tz='UTC')   # into 2026
    tr, va = ssl_data.time_split(ts, val_frac=0.2, holdout_start='2026-01-01')
    cut = pd.Timestamp('2026-01-01', tz='UTC')
    tsi = pd.DatetimeIndex(ts)
    assert (tsi[tr] < cut).all() and (tsi[va] < cut).all()    # 2026 never present
    assert tsi[tr].max() < tsi[va].min()                      # train strictly before val
    n_usable = int((tsi < cut).sum())
    assert len(va) == int(n_usable * 0.2)


def test_window_starts_contiguous():
    idx = np.arange(100)
    s = ssl_data.window_starts(idx, seq_total=10)
    assert len(s) == 91 and s[0] == 0 and s[-1] == 90
    gapped = np.concatenate([np.arange(0, 20), np.arange(50, 70)])   # a hole at 20..50
    sg = ssl_data.window_starts(gapped, seq_total=10)
    assert (((sg + 9 < 20) | (sg >= 50))).all()                      # no window spans the hole
    assert 11 not in sg and 60 in sg                                  # 11..19 can't fit; 50..60 can


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


# ---------------------------------------------------------------- probe + gate (torch-free)
def test_targets_from_windows():
    seq = 8
    ramp = np.linspace(100, 107, seq)                    # pure uptrend
    chop = np.array([100, 101, 100, 101, 100, 101, 100, 101.0])
    def stk(close):
        return np.stack([close, close + 0.5, close - 0.5, close,
                         np.full(seq, 500.0)], 1).astype(np.float32)
    big = np.concatenate([stk(ramp), stk(chop)], 0)      # T=16, 5 cols
    t = ssl_probe.targets_from_windows(big, [0, 8], seq)
    assert t['trend_eff'][0] > 0.9 and t['trend_eff'][1] < 0.3   # trend vs chop
    assert t['direction'][0] == 1                                # net up
    assert set(t) == {'vol', 'trend_eff', 'range_expand', 'direction'}


def test_probe_embedding_recovers_signal():
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((400, 6)).astype(np.float32)
    y_reg = emb[:, 0] * 2.0 + rng.standard_normal(400) * 0.1       # linearly encoded
    assert ssl_probe.probe_embedding(emb, y_reg, 'reg', seed=0) > 0.8
    y_bin = (emb[:, 1] > 0).astype(int)
    assert ssl_probe.probe_embedding(emb, y_bin, 'bin', seed=0) > 0.85
    y_noise = rng.standard_normal(400)                            # unrelated
    assert ssl_probe.probe_embedding(emb, y_noise, 'reg', seed=0) < 0.2


def test_probe_compare_flags_ssl_better():
    rng = np.random.default_rng(1)
    tgt = {'vol': rng.standard_normal(300).astype(np.float32),
           'trend_eff': rng.standard_normal(300).astype(np.float32),
           'range_expand': rng.standard_normal(300).astype(np.float32),
           'direction': rng.integers(0, 2, 300)}
    emb_ssl = np.stack([tgt['vol'], tgt['trend_eff'], tgt['range_expand']], 1) \
        + rng.standard_normal((300, 3)) * 0.05            # encodes the targets
    emb_van = rng.standard_normal((300, 3)).astype(np.float32)     # encodes nothing
    out = ssl_probe.compare(emb_ssl.astype(np.float32), emb_van, tgt, seed=0)
    assert out['learns_regime_vol_structure'] and out['mean_core_delta'] > 0


def test_generalizes_gate():
    good = {'real': {'best_val': 2.0, 'final_std': 0.5, 'val_gap': 0.1},
            'shuffle': {'best_val': 4.0}, 'random': {'best_val': 5.0}}
    ok, d = ssl._generalizes(good)
    assert ok and d['real_beats_controls'] and d['no_collapse'] and d['val_stable']
    coll = {'real': {'best_val': 2.0, 'final_std': 0.001, 'val_gap': 0.1},
            'shuffle': {'best_val': 4.0}}
    assert not ssl._generalizes(coll)[0]
    notbeat = {'real': {'best_val': 4.5, 'final_std': 0.5, 'val_gap': 0.1},
               'shuffle': {'best_val': 4.0}}
    assert not ssl._generalizes(notbeat)[0]


# ------------------------------------------------------------------- torch trainer (gated)
@torch_test
def test_ssl_network_and_augment_shapes():
    import torch
    from futures_foundation.finetune import _ssl_torch as S
    net = S.build_ssl_net(C=5, new_channels=4, proj_dim=64, device='cpu')
    x = torch.randn(8, 5, 64)
    z = net(x)
    assert z.shape == (8, 64)
    assert torch.allclose(z.norm(dim=1), torch.ones(8), atol=1e-4)   # L2-normalized
    parent = torch.randn(8, 5, 72)
    v1, v2 = S._two_views(parent, seq=64, max_jitter=8)
    assert v1.shape == (8, 5, 64) and v2.shape == (8, 5, 64)
    loss = S.nt_xent(z, net(x), temp=0.2)
    assert torch.isfinite(loss)
    cm = S.collapse_metrics(z, net(x))
    assert set(cm) == {'std', 'align', 'uniformity'}


@torch_test
def test_train_ssl_runs_and_ckpt_loads_into_build_model(tmp_path):
    import torch
    from futures_foundation.finetune import _ssl_torch as S
    from futures_foundation.finetune.classifiers._mantis_torch import build_model
    rng = np.random.default_rng(0)
    big = rng.standard_normal((2000, 5)).astype(np.float32)
    starts = np.arange(0, 1800, 4)
    state, hist = S.train_ssl(big, starts, starts[-50:], seq=32, max_jitter=8,
                              new_channels=4, proj_dim=32, epochs=2, steps_per_epoch=3,
                              batch=16, device='cpu', control='real', verbose=False)
    assert len(hist) >= 1 and np.isfinite(hist[-1]['val_loss'])
    ckpt = str(tmp_path / 'enc.pt'); torch.save(state, ckpt)
    # the SSL encoder ckpt initializes the downstream classifier backbone
    model, new_c = build_model(5, new_channels=4, device='cpu', backbone_ckpt=ckpt)
    assert new_c == 4
