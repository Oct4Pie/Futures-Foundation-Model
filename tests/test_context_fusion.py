"""Tests — futures_foundation/chronos/context_fusion seam + per-candle readout.

backbone.embed is monkeypatched (deterministic, torch-free) so the full
evaluate.run() path runs in-process with XGBoost only — the default suite
is torch-free, so no libomp collision.
"""
import numpy as np
import pandas as pd
import pytest

from futures_foundation.context import ContextHeads, HEADS_CUTOFF
from futures_foundation.chronos import context_fusion as cf
from futures_foundation.chronos import evaluate as ev
from futures_foundation.extractors.chronos import backbone

RNG = np.random.default_rng(3)
D = backbone.D_MODEL


def _fake_embed(contexts, batch=64, pool='mean', return_loc_scale=False):
    """Deterministic embedding stub: first dims carry the window's tail
    stats so heads/labels are learnable; rest zeros. Accepts the Tier-1
    pool/return_loc_scale kwargs (real foundation.embed signature)."""
    X = np.asarray(contexts, np.float32)
    dim = 2 * D if pool == 'meanreg' else D
    out = np.zeros((len(X), dim), np.float32)
    if len(X):
        out[:, 0] = X[:, -1] - X[:, 0]
        out[:, 1] = X.std(axis=1)
        out[:, 2] = X[:, -1] - X[:, -20:].mean(axis=1)
    if return_loc_scale:
        ls = np.zeros((len(X), 2), np.float32)
        if len(X):
            ls[:, 0] = X.mean(axis=1)
            ls[:, 1] = np.clip(X.std(axis=1), 1e-6, None)
        return out, ls
    return out


def _heads(n=2000):
    E = RNG.normal(0, 1, (n, D)).astype(np.float32)
    lab = pd.DataFrame({
        'vol_expansion': (E[:, 1] > 0).astype(float),
        'volatility': 1 / (1 + np.exp(-E[:, 2])),
        'structure': (E[:, 0] > 0).astype(float),
        'range_bound': (E[:, 0] < 0).astype(float)})
    cut = int(n * .8)
    return ContextHeads(seed=0, n_estimators=40).fit(
        E[:cut], lab.iloc[:cut], E[cut:], lab.iloc[cut:], verbose=False)


@pytest.fixture(scope='module')
def heads():
    return _heads()


# ---------------------------------------------------------------------------
# fuse() — the A/B arms
# ---------------------------------------------------------------------------

def test_fuse_no_heads_is_legacy_identical(heads):
    E = RNG.normal(0, 1, (10, D)).astype(np.float32)
    F = RNG.normal(0, 1, (10, 3)).astype(np.float32)
    np.testing.assert_array_equal(cf.fuse(E, F, None, 'both'),
                                  np.hstack([E, F]).astype(np.float32))
    np.testing.assert_array_equal(cf.fuse(E, None, None, 'heads'), E)


def _enriched_heads(n=1500, k_ff=8):
    """A heads bundle trained on [emb | ff68] (meta inputs='emb+ff68')."""
    emb = RNG.normal(0, 1, (n, D)).astype(np.float32)
    ff = RNG.normal(0, 1, (n, k_ff)).astype(np.float32)
    X = np.hstack([emb, ff])
    lab = pd.DataFrame({
        'vol_expansion': (X[:, 1] > 0).astype(float),
        'volatility': 1 / (1 + np.exp(-X[:, 2])),
        'structure': np.sign(X[:, 3]),
        'range_bound': (X[:, 0] < 0).astype(float)})
    cut = int(n * .8)
    h = ContextHeads(seed=0, n_estimators=40).fit(
        X[:cut], lab.iloc[:cut], X[cut:], lab.iloc[cut:], verbose=False)
    h.meta = {'inputs': 'emb+ff68'}
    return h, emb, ff


def test_fuse_enriched_needs_ff68_and_runs_with_it():
    h, emb, ff = _enriched_heads()
    nc = len(h.active_names)
    # with ff68 → [emb | ctx], heads run on [emb|ff68]
    out = cf.fuse(emb[:6], None, h, 'both', ff68=ff[:6])
    assert out.shape == (6, D + nc)
    # ctx-only arm
    assert cf.fuse(emb[:6], None, h, 'heads', ff68=ff[:6]).shape == (6, nc)
    # emb arm needs no ff68 (no ctx emitted)
    assert cf.fuse(emb[:6], None, h, 'emb').shape == (6, D)
    # emitting ctx without ff68 → raises
    with pytest.raises(ValueError, match='ff68'):
        cf.fuse(emb[:6], None, h, 'both')


def test_fuse_arm_dims(heads):
    E = RNG.normal(0, 1, (10, D)).astype(np.float32)
    F = RNG.normal(0, 1, (10, 3)).astype(np.float32)
    n_ctx = len(heads.active_names)
    assert cf.fuse(E, F, heads, 'both').shape == (10, D + n_ctx + 3)   # arm B
    assert cf.fuse(E, F, heads, 'emb').shape == (10, D + 3)            # arm A
    assert cf.fuse(E, F, heads, 'heads').shape == (10, n_ctx + 3)      # arm C
    assert cf.fuse(E, None, heads, 'heads').shape == (10, n_ctx)       # arm D
    with pytest.raises(ValueError):
        cf.fuse(E, F, heads, 'bogus')


def test_fuse_raises_on_enriched_bundle(heads):
    """The event-fusion seam only has embeddings — an emb+ff68 bundle
    must be refused loudly, never silently mis-fed."""
    E = RNG.normal(0, 1, (5, D)).astype(np.float32)
    enriched = ContextHeads(seed=0, n_estimators=40)
    enriched.models, enriched.metrics = heads.models, heads.metrics
    enriched.meta = {'inputs': 'emb+ff68'}
    with pytest.raises(ValueError, match='emb\\+ff68'):
        cf.fuse(E, None, enriched, 'both')


# ---------------------------------------------------------------------------
# resolve_heads() — env-var wiring guard
# ---------------------------------------------------------------------------

def test_resolve_heads_default_off(monkeypatch):
    monkeypatch.delenv(cf.ENV_VAR, raising=False)
    assert cf.resolve_heads(None) is None


def test_resolve_heads_env_and_stamp(monkeypatch, tmp_path, heads, capsys):
    p = heads.save(tmp_path / 'heads.joblib')
    monkeypatch.setenv(cf.ENV_VAR, p)
    loaded = cf.resolve_heads(None)
    out = capsys.readouterr().out
    assert loaded.active_names == heads.active_names
    assert 'CONTEXT HEADS' in out and p in out


# ---------------------------------------------------------------------------
# enforce_cutoff() — leak guard
# ---------------------------------------------------------------------------

def test_enforce_cutoff(heads):
    cf.enforce_cutoff(None, '2020-01-01')                  # no heads: pass
    cf.enforce_cutoff(heads, HEADS_CUTOFF)                 # at cutoff: pass
    with pytest.raises(ValueError, match='leak guard'):
        cf.enforce_cutoff(heads, '2022-12-31')


# ---------------------------------------------------------------------------
# evaluate.run() — fold guard + widened feat_dim end-to-end
# ---------------------------------------------------------------------------

class _SynthLabeler:
    """Minimal StrategyLabeler: synthetic hourly signal calendar
    2022-07..2023-12 (intraday cadence, like the real 3m/5m strategies),
    learnable binary labels from the context tail."""
    n_classes = 2

    def __init__(self):
        ts = pd.date_range('2022-07-01', '2023-12-31', freq='1h', tz='UTC')
        self._ts = ts
        rng = np.random.default_rng(5)
        self._drift = rng.normal(0, 1, len(ts))

    def calendar(self):
        return pd.DataFrame({'item_id': 'ES', 'timestamp': self._ts,
                             'target': 0.0})

    def build(self, lo, hi, test_start):
        m = (self._ts >= lo) & (self._ts <= hi)
        idx = np.flatnonzero(m)
        C, Y, K = [], [], []
        for i in idx:
            d = self._drift[i]
            w = np.linspace(0, d, 128) + 0.01 * RNG.normal(size=128)
            C.append(w.astype(np.float32))
            Y.append(int(d > 0))
            K.append(('ES', i, int(d > 0), 2.0))
        return C, np.asarray(Y), K

    def evaluate(self, keys, preds, risk_preds=None):
        R = [(2.0 if k[2] == 1 else -1.0)
             for k, p in zip(keys, np.asarray(preds)) if p != 0]
        return np.asarray(R, float)


def test_evaluate_run_fold_guard_and_feat_dim(monkeypatch, tmp_path, heads,
                                              capsys):
    monkeypatch.setattr(backbone, 'embed', _fake_embed)
    monkeypatch.setattr(ev.backbone, 'embed', _fake_embed)
    monkeypatch.delenv(cf.ENV_VAR, raising=False)
    p = heads.save(tmp_path / 'heads.joblib')

    ev.run(_SynthLabeler(), seeds=(0,), max_folds=2, context_heads_path=p,
           emb_mode='both')
    out = capsys.readouterr().out
    n_ctx = len(heads.active_names)
    # leak guard auto-applied: pre-2023 folds excluded, message printed
    assert 'leak guard' in out
    assert f'feat_dim={D + n_ctx}' in out          # widened by ctx heads
    assert 'CONTEXT HEADS' in out


def test_evaluate_run_default_off_unchanged(monkeypatch, capsys):
    monkeypatch.setattr(backbone, 'embed', _fake_embed)
    monkeypatch.setattr(ev.backbone, 'embed', _fake_embed)
    monkeypatch.delenv(cf.ENV_VAR, raising=False)
    ev.run(_SynthLabeler(), seeds=(0,), max_folds=1)
    out = capsys.readouterr().out
    assert f'feat_dim={D}' in out                  # legacy width
    assert 'CONTEXT HEADS' not in out


# ---------------------------------------------------------------------------
# per-candle readout — context_at()
# ---------------------------------------------------------------------------

def test_context_at_per_candle_readout(monkeypatch, heads):
    from futures_foundation.extractors.chronos import backbone as foundation
    monkeypatch.setattr(foundation, 'embed', _fake_embed)
    n = 500
    close = 100 * np.exp(np.cumsum(RNG.normal(0, .002, n)))
    ohlcv = pd.DataFrame({
        'datetime': pd.date_range('2023-06-01', periods=n, freq='5min',
                                  tz='UTC'),
        'open': close, 'high': close * 1.001, 'low': close * 0.999,
        'close': close, 'volume': np.full(n, 1000.0)})
    idx = [200, 300, 499]
    df = heads.context_at(ohlcv, idx, 'ES')   # emb-only bundle: emb alone
    assert list(df.index) == idx
    assert list(df.columns) == heads.active_names
    assert df.notna().all().all()


# ---------------------------------------------------------------------------
# HTF readout — htf_context_at() (causal cross-TF context for 1m/3m/5m)
# ---------------------------------------------------------------------------

def _base_series(n_days=12, freq='5min'):
    ts = pd.date_range('2023-03-01', periods=n_days * 288,
                       freq=freq, tz='UTC')
    close = 100 * np.exp(np.cumsum(RNG.normal(0, .0005, len(ts))))
    return ts, close


def test_htf_context_columns_and_nan_warmup(monkeypatch, heads):
    from futures_foundation.extractors.chronos import backbone as foundation
    monkeypatch.setattr(foundation, 'embed', _fake_embed)
    ts, close = _base_series()
    idx = [10, 1500, 3000]                      # 10: far too little HTF history
    df = heads.htf_context_at(ts, close, idx, htf='1h', ctx=8)
    assert list(df.columns) == [f'{n}_1h' for n in heads.active_names]
    assert df.loc[10].isna().all()              # warm-up rows NaN, not filled
    assert df.loc[3000].notna().all()


def test_htf_context_is_strictly_causal(monkeypatch, heads):
    """Changing base bars at/after the decision bar — INCLUDING the
    still-forming HTF bucket — must not change the HTF readout."""
    from futures_foundation.extractors.chronos import backbone as foundation
    monkeypatch.setattr(foundation, 'embed', _fake_embed)
    ts, close = _base_series()
    i = 3000                                    # mid-series decision bar
    a = heads.htf_context_at(ts, close, [i], htf='1h', ctx=8)
    mod = close.copy()
    # shock everything from the start of the decision bar's CURRENT hour
    hour_start = np.flatnonzero(ts.floor('1h') == ts[i].floor('1h'))[0]
    mod[hour_start:] *= 1.10
    b = heads.htf_context_at(ts, mod, [i], htf='1h', ctx=8)
    pd.testing.assert_frame_equal(a, b)         # forming bucket invisible


def test_htf_context_sees_completed_buckets(monkeypatch, heads):
    """Changing a COMPLETED prior HTF bucket must change the readout."""
    from futures_foundation.extractors.chronos import backbone as foundation
    monkeypatch.setattr(foundation, 'embed', _fake_embed)
    ts, close = _base_series()
    i = 3000
    a = heads.htf_context_at(ts, close, [i], htf='1h', ctx=8)
    mod = close.copy()
    prev_hour = np.flatnonzero(ts.floor('1h') == (ts[i].floor('1h')
                                                  - pd.Timedelta('1h')))
    mod[prev_hour] *= 1.10                      # completed bucket changes
    b = heads.htf_context_at(ts, mod, [i], htf='1h', ctx=8)
    assert not a.equals(b)
