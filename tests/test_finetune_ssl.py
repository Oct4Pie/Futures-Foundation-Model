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
    # stage-3 v2: ctx + the FUTURE key horizon (the key reads the next contrast_horizon bars)
    assert ssl.get_pretext('contrastive').reserve(cfg) == 200 + cfg['contrast_horizon']


def test_base_cfg_has_contrastive_keys():
    cfg = ssl._base_cfg()
    assert cfg['temperature'] == 0.1 and cfg['crop_max'] == 0.2 and cfg['proj_dim'] == 128
    assert cfg['contrast_horizon'] == 25 and cfg['pos_cap'] == 64    # v2 forward-key knobs
    over = ssl._base_cfg(pretext='contrastive', temperature=0.07, crop_max=0.1, contrast_horizon=10)
    assert over['pretext'] == 'contrastive' and over['temperature'] == 0.07 and over['crop_max'] == 0.1
    assert over['contrast_horizon'] == 10


def test_passes_contrastive_gate_report_only():
    """Stage-3 gate is report-only: no-collapse + descriptive content doesn't regress (the REAL
    gate = trend-AUC + WR@3R offline). Stage-1/2 gates are unaffected."""
    good = dict(mean_core_delta=0.03, descriptive_delta=0.02, learns_regime_vol_structure=True)
    ok, d = ssl._passes(good, std=0.5, pretext='contrastive')
    assert ok and d['descriptive_ok'] and d['no_collapse']
    assert not ssl._passes(good, std=0.001, pretext='contrastive')[0]        # collapse -> fail
    desc_reg = dict(good, descriptive_delta=-0.01)
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
    from futures_foundation.finetune.classifiers._mantis_torch import build_model
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


# --------------------------- stage-3 v3 forward barrier-excursion contrastive (torch, gated)
@torch_test
def test_future_key_separates_trend_vs_chop():
    """The FORWARD key buckets windows by their FUTURE's character: future up/down runners land in
    different direction buckets (key//3 = 0 down/1 flat/2 up) and a whipsaw future — price never
    RUNS a unit before retracing — lands in the CHOP excursion bucket (key%3 == 0)."""
    import torch
    from futures_foundation.finetune import _ssl_torch as S
    torch.manual_seed(0)
    L, h = 64, 16
    cs = torch.randn(24, 5, L) * 0.1                               # IDENTICAL-character context (noise)
    cs[:, 3, :] = 0.0                                              # entry ('now' close) at exactly 0
    t = torch.linspace(0, 1, h)
    up = (t * 4).expand(8, h)                                      # future: clean up runner (8 units)
    dn = (-t * 4).expand(8, h)                                     # future: clean down runner
    alt = torch.tensor([0.3, -0.3]).repeat(h // 2)                 # future: whipsaw around entry
    chop = alt.expand(8, h)                                        # (never runs a retrace-unit)
    fs = torch.zeros(24, 5, h)
    fs[:, 3, :] = torch.cat([up, dn, chop], 0) + torch.randn(24, h) * 0.02   # close channel = 3
    key = S._future_key(cs, fs, dz=0.5, e1=1.0, e2=3.0)            # FIXED edges (retrace-unit=dz)
    dir_b, run_b = key // 3, key % 3
    assert (dir_b[:8] == 2).float().mean() > 0.7                   # future up runner -> up bucket
    assert (dir_b[8:16] == 0).float().mean() > 0.7                 # future down runner -> down bucket
    assert (run_b[16:] == 0).float().mean() > 0.7                  # whipsaw -> CHOP (no clean run)
    assert (run_b[:16] == 2).float().mean() > 0.7                  # clean drifts -> RUNNER bucket


@torch_test
def test_future_key_barrier_ordering():
    """The v3 stat is an ORDERING statistic (the whole point vs v2's efficiency): a path that first
    runs +2 units and THEN crashes -6 scores a 2-unit run (the crash doesn't count — a short from
    'now' was stopped by the +2 first), while the same crash without the fakeout scores ~6 units.
    Net move alone can't tell these apart; the barrier excursion can."""
    import torch
    from futures_foundation.finetune import _ssl_torch as S
    h = 20
    cs = torch.zeros(2, 5, 8)                                      # entry close = 0
    fs = torch.zeros(2, 5, h)
    fs[0, 3, :] = torch.cat([torch.linspace(0.1, 1.0, 5), torch.linspace(0.6, -3.0, 15)])  # fakeout
    fs[1, 3, :] = torch.linspace(-0.2, -3.0, h)                    # clean down runner
    net, run = S._future_barrier_stats(cs, fs, unit=0.5)
    assert abs(float(run[0]) - 2.0) < 0.3                          # fakeout: only the +1.0 pre-crash run
    assert float(run[1]) > 5.0                                     # clean crash: ~6-unit down run
    assert float(net[0]) < 0 and float(net[1]) < 0                 # same net sign — net can't separate


@torch_test
def test_future_key_groups_by_future_not_past():
    """THE anti-shortcut property v1 lacked: two windows with the SAME context (past) but different
    FUTURES get DIFFERENT keys — so the key cannot be computed from the input, and the encoder must
    learn causal precursors of the future character (trend detection), not restate the past."""
    import torch
    from futures_foundation.finetune import _ssl_torch as S
    torch.manual_seed(1)
    L, h = 64, 16
    ctx = torch.randn(1, 5, L).repeat(2, 1, 1)                     # EXACT same past for both
    ctx[:, 3, -1] = 0.0                                            # shared entry close at 0
    t = torch.linspace(0, 1, h)
    fs = torch.zeros(2, 5, h)
    fs[0, 3, :] = t * 4                                            # window A future: runs up 8 units
    fs[1, 3, :] = torch.tensor([0.3, -0.3]).repeat(h // 2)         # window B future: whipsaws
    key = S._future_key(ctx, fs, dz=0.5, e1=1.0, e2=3.0)
    assert int(key[0]) != int(key[1])                              # same past, different key


@torch_test
def test_multi_positive_infonce_rewards_trend_grouping():
    """Multi-positive InfoNCE gives LOWER loss when same-key (same-trend) embeddings are aligned
    than anti-aligned -> the loss PULLS same-trend windows together (fixes false-negatives)."""
    import torch
    import torch.nn.functional as F
    from futures_foundation.finetune import _ssl_torch as S
    key = torch.tensor([0, 0, 1, 1]); inst = torch.tensor([0, 1, 2, 3])
    aligned = F.normalize(torch.tensor([[1., 0], [1, 0], [0, 1.], [0, 1]]), dim=1)
    anti = F.normalize(torch.tensor([[1., 0], [-1, 0], [0, 1.], [0, -1]]), dim=1)
    la = S._multi_positive_infonce(aligned, key, inst, 0.1)
    lb = S._multi_positive_infonce(anti, key, inst, 0.1)
    assert torch.isfinite(la) and float(la) < float(lb)


@torch_test
def test_contrastive_net_shape_and_trainer_smoke(tmp_path):
    """ContrastiveTrendNet -> L2-normalized [B, proj_dim]; the v2 trainer runs (fixed key edges
    calibrated, key_gap reported), returns an encoder state loadable downstream + finite val loss,
    and accepts a warm-start ckpt (from the stage-2 winner)."""
    import torch
    from futures_foundation.finetune import _ssl_torch as S
    from futures_foundation.finetune.classifiers._mantis_torch import build_model
    net = S.ContrastiveTrendNet(C=5, new_channels=4, proj_dim=64).to('cpu')
    z = net(torch.randn(6, 5, 64))
    assert z.shape == (6, 64) and torch.allclose(z.norm(dim=1), torch.ones(6), atol=1e-4)
    rng = np.random.default_rng(0)
    big = (100 + np.cumsum(rng.standard_normal((3000, 5)) * 0.1, 0)).astype(np.float32)
    big[:, 4] = np.abs(big[:, 4]) * 100 + 500                      # positive-ish volume
    cl, h = (32, 48), 8; starts = np.arange(0, 3000 - (48 + 8) - 1, 4)   # parent = ctx + horizon
    state, hist = S.train_ssl_contrastive(big, starts, starts[-50:], context_lengths=cl,
                                          contrast_horizon=h, new_channels=4, proj_dim=64,
                                          epochs=2, steps_per_epoch=3, batch=16, device='cpu',
                                          control='real', verbose=False)
    assert len(hist) >= 1 and np.isfinite(hist[-1]['val_loss']) and hist[-1]['std'] > 0
    assert 'key_gap' in hist[-1] and np.isfinite(hist[-1]['key_gap'])   # separation diagnostic
    ckpt = str(tmp_path / 'enc.pt'); torch.save(state, ckpt)
    _, new_c = build_model(5, new_channels=4, device='cpu', backbone_ckpt=ckpt)
    assert new_c == 4                                              # encoder ckpt loads downstream
    state2, _ = S.train_ssl_contrastive(big, starts, starts[-50:], context_lengths=cl,
                                        contrast_horizon=h, new_channels=4, proj_dim=64, epochs=1,
                                        steps_per_epoch=2, batch=16, device='cpu', control='real',
                                        backbone_ckpt=ckpt, verbose=False)
    assert set(state2.keys()) == set(state.keys())                # warm-start same encoder keys


# --------------------------------------------- save/resume + anti-forgetting freeze (all pretexts)
def test_base_cfg_has_ckpt_resume_freeze_keys():
    cfg = ssl._base_cfg()
    assert cfg['ckpt_path'] is None and cfg['resume'] is False and cfg['freeze_encoder_layers'] == 0
    over = ssl._base_cfg(resume=True, freeze_encoder_layers=4)
    assert over['resume'] is True and over['freeze_encoder_layers'] == 4


@torch_test
def test_freeze_encoder_layers_anchors_early_leaves_adapter_trainable():
    from futures_foundation.finetune.pretext._torch.common import _freeze_encoder
    from futures_foundation.finetune.pretext._torch.contrastive import ContrastiveTrendNet
    net = ContrastiveTrendNet(C=5, new_channels=4, proj_dim=32).to('cpu')
    before = sum(p.requires_grad for p in net.parameters())
    n = _freeze_encoder(net.encoder, 4)
    after = sum(p.requires_grad for p in net.parameters())
    assert n == 4 and after < before                          # froze tokenizer + first 4 blocks
    assert any(p.requires_grad for p in net.adapter.parameters())   # embedding can still adapt
    assert any(p.requires_grad for p in net.prj.parameters())
    assert _freeze_encoder(net.encoder, 0) == 0               # n<=0 -> no-op


@torch_test
def test_contrastive_save_resume_and_control_guard(tmp_path):
    import os
    from futures_foundation.finetune import _ssl_torch as S
    rng = np.random.default_rng(0)
    big = (100 + np.cumsum(rng.standard_normal((3000, 5)) * 0.1, 0)).astype(np.float32)
    big[:, 4] = np.abs(big[:, 4]) * 100 + 500
    cl, h = (32, 48), 8                                                # parent = ctx + FUTURE horizon
    starts = np.arange(0, 3000 - (48 + 8) - 1, 4); ck = str(tmp_path / 'enc.pt')
    st, _ = S.train_ssl_contrastive(big, starts, starts[-50:], context_lengths=cl, contrast_horizon=h,
                                    new_channels=4, proj_dim=32, epochs=2, steps_per_epoch=3,
                                    batch=16, device='cpu', control='real', ckpt_path=ck, verbose=False)
    assert os.path.exists(ck) and os.path.exists(ck + '.meta.json')     # progressively saved
    st2, _ = S.train_ssl_contrastive(big, starts, starts[-50:], context_lengths=cl, contrast_horizon=h,
                                     new_channels=4, proj_dim=32, epochs=1, steps_per_epoch=2,
                                     batch=16, device='cpu', control='real', ckpt_path=ck,
                                     resume=True, verbose=False)
    assert set(st2.keys()) == set(st.keys())                            # resumed + returned encoder
    before = os.path.getmtime(ck)                                       # controls must NOT touch ckpt
    S.train_ssl_contrastive(big, starts, starts[-50:], context_lengths=cl, contrast_horizon=h,
                            new_channels=4, proj_dim=32, epochs=1, steps_per_epoch=2, batch=16,
                            device='cpu', control='shuffle', ckpt_path=ck, verbose=False)
    assert os.path.getmtime(ck) == before                              # shuffle control didn't save


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
