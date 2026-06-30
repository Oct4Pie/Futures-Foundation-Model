"""SSL masked-modeling pretraining — torch-free data/assembly/probe/gate (non-gated) +
torch masked trainer (gated behind CHRONOS_TORCH_TESTS=1, libomp isolation).

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
    t = ssl_probe.targets_from_windows(big, [0, 8], seq, fwd_k=4)
    assert t['trend_eff'][0] > 0.9 and t['trend_eff'][1] < 0.3   # trend vs chop
    assert t['direction'][0] == 1                                # net up
    assert set(t) == {'vol', 'trend_eff', 'range_expand', 'fwd_absmove',
                      'direction', 'fwd_dir'}                     # + forward buy/sell targets
    assert (t['fwd_absmove'] >= 0).all()


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
    core = ['vol', 'trend_eff', 'range_expand', 'fwd_absmove']     # the gate's core targets
    tgt = {k: rng.standard_normal(300).astype(np.float32) for k in core}
    tgt['direction'] = rng.integers(0, 2, 300)
    tgt['fwd_dir'] = rng.integers(0, 2, 300)
    emb_ssl = (np.stack([tgt[k] for k in core], 1)
               + rng.standard_normal((300, 4)) * 0.05).astype(np.float32)   # encodes targets
    emb_van = rng.standard_normal((300, 4)).astype(np.float32)              # encodes nothing
    out = ssl_probe.compare(emb_ssl, emb_van, tgt, seed=0)
    assert out['learns_regime_vol_structure'] and out['mean_core_delta'] > 0


def test_passes_gate_on_probe_not_loss():
    # GATE = probe (representation content) vs vanilla, NOT contrastive loss. ORIGINAL mask gate.
    good = {'mean_core_delta': 0.05, 'learns_regime_vol_structure': True}
    ok, d = ssl._passes(good, std=0.5)                    # default pretext='mask' (stage 1, unchanged)
    assert ok and d['learns_regime_vol_structure']
    # probe ties/loses vanilla -> fail even though training "looked fine"
    bad = {'mean_core_delta': -0.01, 'learns_regime_vol_structure': False}
    assert not ssl._passes(bad, std=0.5)[0]
    # collapse -> fail regardless of probe
    assert not ssl._passes(good, std=0.001)[0]


def test_passes_forecast_gate_is_forward_centric_anti_shortcut():
    """Stage-2 (forecast) gate: descriptive gains alone CANNOT pass — forward MOVE SIZE must
    improve and forward DIRECTION must not regress. Stage-1 (mask) gate is unaffected."""
    base = dict(mean_core_delta=0.05, learns_regime_vol_structure=True)
    # shortcut: big descriptive lift, but forward targets flat/negative -> FAIL on forecast...
    shortcut = dict(base, descriptive_delta=0.10, fwd_absmove_delta=0.0, fwd_dir_delta=-0.02,
                    forward_score=-0.02)
    assert not ssl._passes(shortcut, std=0.5, pretext='forecast')[0]
    # ...yet the SAME probe passes the ORIGINAL mask gate (mean_core_delta>0) -> stage 1 intact
    assert ssl._passes(shortcut, std=0.5, pretext='mask')[0]
    # genuine forward learning: move size up, direction not worse -> PASS on forecast
    genuine = dict(base, descriptive_delta=0.02, fwd_absmove_delta=0.03, fwd_dir_delta=0.01,
                   forward_score=0.04)
    ok, d = ssl._passes(genuine, std=0.5, pretext='forecast')
    assert ok and d['fwd_size_ok'] and d['fwd_dir_ok'] and d['descriptive_ok']
    # direction regresses -> FAIL even with forward move-size gain
    dir_reg = dict(genuine, fwd_dir_delta=-0.01)
    assert not ssl._passes(dir_reg, std=0.5, pretext='forecast')[0]
    # descriptive regresses -> FAIL
    desc_reg = dict(genuine, descriptive_delta=-0.01)
    assert not ssl._passes(desc_reg, std=0.5, pretext='forecast')[0]


def test_suggest_ssl_forecast_searches_channel_weights():
    """Stage-2 scan searches channel weights (close up / volume down) + forecast knobs; the
    assembled weights are O,H,L fixed at 1 with close_weight, vol_weight from the trial."""
    fc = _FakeTrial(); d = ssl._suggest_ssl(fc, 'forecast')
    assert 'channel_weights' in d and 'horizon' in d
    assert {'close_weight', 'vol_weight', 'lr', 'weight_decay', 'new_channels'} <= set(fc.asked)
    cw = d['channel_weights']
    assert len(cw) == 5 and cw[0] == cw[1] == cw[2] == 1.0     # O,H,L fixed
    # mask pretext must NOT search channel weights (stage-1 untouched)
    mk = _FakeTrial(); d2 = ssl._suggest_ssl(mk, 'mask')
    assert 'channel_weights' not in d2 and 'close_weight' not in mk.asked


def test_rebuild_channel_weights_from_best_params():
    """study.best_params records close_weight/vol_weight -> reassembled into the trainer vector."""
    p = {'lr': 1e-4, 'horizon': 16, 'close_weight': 2.5, 'vol_weight': 0.0, 'new_channels': 8}
    out = ssl._rebuild_channel_weights(p)
    assert out['channel_weights'] == [1.0, 1.0, 1.0, 2.5, 0.0]
    assert 'close_weight' not in out and 'vol_weight' not in out and out['horizon'] == 16


def test_base_cfg_has_channel_weights_key():
    assert 'channel_weights' in ssl._base_cfg() and ssl._base_cfg()['channel_weights'] is None
    assert ssl._base_cfg(channel_weights=[1, 1, 1, 2, 0])['channel_weights'] == [1, 1, 1, 2, 0]


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
def test_assemble_reserves_forecast_horizon(tmp_path):
    """The forecast pretext needs context+horizon in-stream: assemble reserves
    seq + max(max_jitter, horizon), and every window stays inside one stream."""
    _write_csv(tmp_path / 'ES_3min.csv', 400)
    _write_csv(tmp_path / 'NQ_3min.csv', 400)
    streams = ssl_data.load_ohlcv(str(tmp_path), ['ES', 'NQ'], ['3min'], verbose=False)
    seq, max_jitter, horizon = 32, 8, 24
    big, tr, va = ssl.assemble(streams, seq=seq, max_jitter=max_jitter, horizon=horizon,
                               val_frac=0.1, holdout_start=None, verbose=False)
    parent = seq + max(max_jitter, horizon)               # 32 + 24 = 56 (horizon dominates)
    assert parent == 56
    bounds = [0, 400, 800]
    for s in np.concatenate([tr, va]):
        seg = 0 if s < 400 else 1
        assert bounds[seg] <= s and s + parent <= bounds[seg + 1]   # context+horizon in-stream


def test_assemble_horizon_zero_matches_mask(tmp_path):
    """horizon=0 (mask pretext) reserves only max_jitter — backward-compatible with stage 1."""
    _write_csv(tmp_path / 'ES_3min.csv', 400)
    streams = ssl_data.load_ohlcv(str(tmp_path), ['ES'], ['3min'], verbose=False)
    _, tr0, _ = ssl.assemble(streams, seq=32, max_jitter=8, horizon=0,
                             val_frac=0.1, holdout_start=None, verbose=False)
    _, tr_default, _ = ssl.assemble(streams, seq=32, max_jitter=8,
                                    val_frac=0.1, holdout_start=None, verbose=False)
    assert np.array_equal(tr0, tr_default)                # default horizon=0


def test_base_cfg_has_seq2seq_keys():
    cfg = ssl._base_cfg()
    assert cfg['pretext'] == 'mask' and cfg['horizon'] == 16 and cfg['backbone_ckpt'] is None
    over = ssl._base_cfg(pretext='forecast', horizon=24, backbone_ckpt='/x/enc.pt')
    assert over['pretext'] == 'forecast' and over['horizon'] == 24
    assert over['backbone_ckpt'] == '/x/enc.pt'


class _FakeTrial:
    """Records which hyperparameters were requested; returns the low bound deterministically."""
    def __init__(self):
        self.asked = []

    def suggest_float(self, name, lo, hi, log=False):
        self.asked.append(name); return lo

    def suggest_int(self, name, lo, hi):
        self.asked.append(name); return lo


def test_suggest_ssl_is_pretext_aware():
    fc = _FakeTrial(); d = ssl._suggest_ssl(fc, 'forecast')
    assert 'horizon' in d and 'mask_ratio' not in d and 'horizon' in fc.asked
    mk = _FakeTrial(); d2 = ssl._suggest_ssl(mk, 'mask')
    assert 'mask_ratio' in d2 and 'horizon' not in d2 and 'mask_ratio' in mk.asked
    # shared knobs present in both
    assert {'lr', 'weight_decay', 'new_channels'} <= set(d) and {'lr', 'weight_decay'} <= set(d2)


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

    cfg = ssl._base_cfg(pretext='forecast', horizon=20)
    st, _ = ssl._train(None, None, None, cfg, control='real')
    assert st == 'fc_state' and calls['fn'] == 'forecast'
    assert 'pretext' not in calls['kw'] and calls['kw']['horizon'] == 20

    cfg2 = ssl._base_cfg(pretext='mask')
    st2, _ = ssl._train(None, None, None, cfg2, control='real')
    assert st2 == 'mask_state' and calls['fn'] == 'mask' and 'pretext' not in calls['kw']


# ------------------------------------------------------------- masked-modeling trainer (gated)
@torch_test
def test_embed_windows_frozen():
    import numpy as _np
    from futures_foundation.finetune import _ssl_torch as S
    W = _np.random.default_rng(0).standard_normal((6, 5, 64)).astype(_np.float32)
    emb = S.embed_windows(W, ckpt=None, device='cpu')               # vanilla, encoder-only
    assert emb.shape[0] == 6 and emb.shape[1] > 0 and _np.isfinite(emb).all()
@torch_test
def test_mask_network_and_trainer(tmp_path):
    import torch
    from futures_foundation.finetune import _ssl_torch as S
    from futures_foundation.finetune.classifiers._mantis_torch import build_model
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
    ckpt = str(tmp_path / 'enc.pt'); torch.save(state, ckpt)        # encoder ckpt round-trips
    _, new_c = build_model(5, new_channels=4, device='cpu', backbone_ckpt=ckpt)
    assert new_c == 4


# ------------------------------------------------------ seq2seq forecast trainer (gated)
@torch_test
def test_standardize_ctx_is_leak_safe():
    """Future bars are standardized by the CONTEXT's mean/std (not their own) — so the target
    carries no future-level/scale leak. Context standardizes to ~0 mean / ~1 std per channel."""
    import torch
    from futures_foundation.finetune import _ssl_torch as S
    rng = np.random.default_rng(0)
    ctx = torch.as_tensor(rng.standard_normal((4, 5, 32)).astype(np.float32) * 3 + 7)
    fut = ctx[:, :, :8] * 10.0 + 100.0                             # very different level/scale
    cs, fs = S._standardize_ctx(ctx, fut, clamp=100.0)       # high clamp: no saturation here
    assert torch.allclose(cs.mean(2), torch.zeros(4, 5), atol=1e-4)
    assert torch.allclose(cs.std(2, unbiased=True), torch.ones(4, 5), atol=1e-2)  # unbiased: matches code
    # fut standardized with context stats: recompute and compare (NOT standardized by its own)
    m = ctx.mean(2, keepdim=True); s = ctx.std(2, keepdim=True) + 1e-6
    assert torch.allclose(fs, ((fut - m) / s).clamp(-100, 100), atol=1e-5)
    assert not torch.allclose(fs.mean(2), torch.zeros(4, 5), atol=1e-2)   # fut not self-normalized


@torch_test
def test_standardize_ctx_clamps_flat_context_blowup():
    """The bug that detonated stage-2 training: a FLAT context window (near-zero std) divides a
    real future move by ~0 -> astronomical standardized target. Clamp must bound it."""
    import torch
    from futures_foundation.finetune import _ssl_torch as S
    ctx = torch.full((2, 5, 32), 100.0)                      # perfectly flat -> std ~ 0
    ctx += torch.randn(2, 5, 32) * 1e-7                       # microscopic jitter (tiny std)
    fut = torch.full((2, 5, 8), 130.0)                       # a real 30-pt move after compression
    cs, fs = S._standardize_ctx(ctx, fut, clamp=10.0)
    assert torch.isfinite(cs).all() and torch.isfinite(fs).all()
    assert cs.abs().max() <= 10.0 + 1e-4 and fs.abs().max() <= 10.0 + 1e-4   # bounded, no blowup


@torch_test
def test_forecast_network_shape():
    import torch
    from futures_foundation.finetune import _ssl_torch as S
    net = S.ForecastNetwork(C=5, new_channels=4, seq=64, horizon=16)
    ctx = torch.randn(8, 5, 64)
    assert net(ctx).shape == (8, 5, 16)                            # predict next horizon bars
    assert net.embed(ctx).shape[0] == 8 and net.embed(ctx).shape[1] > 0


@torch_test
def test_train_ssl_forecast_runs_and_warmstarts(tmp_path):
    """Forecast trainer runs, returns finite val MSE + emb std, and its encoder ckpt loads
    downstream via build_model — AND a warm-start ckpt is accepted (FT on stage-1 encoder)."""
    import torch
    from futures_foundation.finetune import _ssl_torch as S
    from futures_foundation.finetune.classifiers._mantis_torch import build_model
    rng = np.random.default_rng(0)
    big = rng.standard_normal((2000, 5)).astype(np.float32)
    starts = np.arange(0, 1880, 4)                                 # room for seq+horizon
    state, hist = S.train_ssl_forecast(big, starts, starts[-50:], seq=32, horizon=16,
                                       new_channels=4, epochs=2, steps_per_epoch=3, batch=16,
                                       device='cpu', control='real', verbose=False)
    assert len(hist) >= 1 and np.isfinite(hist[-1]['val_loss']) and hist[-1]['std'] > 0
    # anti-shortcut readout present: persistence (copy-last-bar) baseline + skill vs it
    assert 'persist_loss' in hist[-1] and hist[-1]['persist_loss'] > 0
    assert 'skill' in hist[-1] and np.isfinite(hist[-1]['skill'])
    ckpt = str(tmp_path / 'enc1.pt'); torch.save(state, ckpt)
    _, new_c = build_model(5, new_channels=4, device='cpu', backbone_ckpt=ckpt)
    assert new_c == 4
    # warm-start: a second forecast run initialized from the first encoder ckpt
    state2, hist2 = S.train_ssl_forecast(big, starts, starts[-50:], seq=32, horizon=16,
                                         new_channels=4, epochs=1, steps_per_epoch=2, batch=16,
                                         device='cpu', control='real', backbone_ckpt=ckpt,
                                         verbose=False)
    assert set(state2.keys()) == set(state.keys()) and np.isfinite(hist2[-1]['val_loss'])
    # NB: the REAL-vs-SHUFFLE/RANDOM control is a research-time PROBE diagnostic measured at
    # scale on Colab (see ssl._finalize), NOT a unit-test invariant — an undertrained 8M model
    # on 16-bar-ahead forecasting does not reliably separate the controls in a CPU smoke run.


@torch_test
def test_train_ssl_forecast_channel_weighted(tmp_path):
    """Channel-weighted loss (price-path) runs; skill stays finite. With volume zeroed, the
    loss/skill exclude volume -> pure price skill, and the encoder ckpt still loads downstream."""
    import torch
    from futures_foundation.finetune import _ssl_torch as S
    from futures_foundation.finetune.classifiers._mantis_torch import build_model
    rng = np.random.default_rng(1)
    big = rng.standard_normal((2000, 5)).astype(np.float32)
    big[:, 4] = rng.standard_normal(2000) * 50 + 500           # volume: different scale + noisy
    starts = np.arange(0, 1880, 4)
    state, hist = S.train_ssl_forecast(big, starts, starts[-50:], seq=32, horizon=16,
                                       new_channels=4, epochs=2, steps_per_epoch=3, batch=16,
                                       device='cpu', control='real',
                                       channel_weights=[1.0, 1.0, 1.0, 2.0, 0.0],  # price-path
                                       verbose=False)
    assert np.isfinite(hist[-1]['val_loss']) and np.isfinite(hist[-1]['skill'])
    ckpt = str(tmp_path / 'enc_pw.pt'); torch.save(state, ckpt)
    _, new_c = build_model(5, new_channels=4, device='cpu', backbone_ckpt=ckpt)
    assert new_c == 4


