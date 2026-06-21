"""Tests — futures_foundation.context (ContextHeads + promoted labels).

xgboost runs in-process here (the default suite is torch-free since the
Bolt-foundation refactor, so no libomp collision). No torch imports.
"""
import numpy as np
import pandas as pd
import pytest

from futures_foundation import context as ctx
from futures_foundation.context import ContextHeads, compute_context_labels

RNG = np.random.default_rng(11)


def test_probe_script_uses_library_generators():
    """The probe script must alias the library functions — no drift."""
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        'probe_context_heads',
        Path(__file__).resolve().parents[1] / 'scripts'
        / 'probe_context_heads.py')
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    assert probe.compute_labels is compute_context_labels
    assert probe.HEADS == ctx.HEAD_SPECS + ctx.CANDIDATE_HEAD_SPECS


def _synthetic_dataset(n=3000, d=24, informative=True):
    """Embeddings where the labels ARE (noisily) recoverable from the
    first dims — so every head must clear its gate; or pure noise so none
    may. Returns (E_tr, lab_tr, E_va, lab_va)."""
    E = RNG.normal(0, 1, (n, d)).astype(np.float32)
    lab = pd.DataFrame()
    if informative:
        lab['vol_expansion'] = (E[:, 1] + RNG.normal(0, .3, n) > 0).astype(float)
        lab['volatility'] = 1 / (1 + np.exp(-(E[:, 2] + RNG.normal(0, .3, n))))
        s = E[:, 3] + RNG.normal(0, .3, n)
        lab['structure'] = np.where(s > .5, 1.0, np.where(s < -.5, 0.0, np.nan))
        lab['range_bound'] = (E[:, 6] + RNG.normal(0, .3, n) > 0).astype(float)
    else:
        lab['vol_expansion'] = (RNG.random(n) > .5).astype(float)
        lab['volatility'] = RNG.random(n)
        lab['structure'] = (RNG.random(n) > .5).astype(float)
        lab['range_bound'] = (RNG.random(n) > .5).astype(float)
    cut = int(n * .8)
    return E[:cut], lab.iloc[:cut], E[cut:], lab.iloc[cut:]


@pytest.fixture(scope='module')
def fitted_heads():
    E_tr, lab_tr, E_va, lab_va = _synthetic_dataset(informative=True)
    return (ContextHeads(seed=0, n_estimators=60)
            .fit(E_tr, lab_tr, E_va, lab_va, verbose=False)), E_va


def test_fit_gates_pass_on_recoverable_labels(fitted_heads):
    heads, _ = fitted_heads
    for name, kind in ctx.HEAD_SPECS:
        m = heads.metrics[name]
        assert m['passed'], f"{name} should clear its gate: {m}"
        assert m['metric'] == ('pearson_r' if kind == 'reg' else 'auc')
    assert heads.active_names == [f'ctx_{n}' for n, _ in ctx.HEAD_SPECS]


def test_transform_shape_order_dtype(fitted_heads):
    heads, E_va = fitted_heads
    X = heads.transform(E_va)
    assert X.shape == (len(E_va), len(heads.active_names))
    assert X.dtype == np.float32
    # clf columns are probabilities
    i_ve = heads.active_names.index('ctx_vol_expansion')
    assert 0.0 <= X[:, i_ve].min() and X[:, i_ve].max() <= 1.0


def test_transform_include_override_for_ablation(fitted_heads):
    heads, E_va = fitted_heads
    X = heads.transform(E_va, include=['volatility', 'vol_expansion'])
    assert X.shape == (len(E_va), 2)


def test_noise_labels_fail_gates_and_transform_is_empty():
    # large n: with 3 clf heads, chance AUC > 0.55 must be ~impossible
    E_tr, lab_tr, E_va, lab_va = _synthetic_dataset(n=8000, informative=False)
    heads = ContextHeads(seed=0, n_estimators=40).fit(
        E_tr, lab_tr, E_va, lab_va, verbose=False)
    assert heads.active_names == []          # nothing may pass on noise
    assert heads.transform(E_va).shape == (len(E_va), 0)


def test_save_load_roundtrip(fitted_heads, tmp_path):
    heads, E_va = fitted_heads
    heads.meta = {'cutoff': str(ctx.HEADS_CUTOFF), 'note': 'unit'}
    p = heads.save(tmp_path / 'heads.joblib')
    loaded = ContextHeads.load(p)
    assert loaded.active_names == heads.active_names
    assert loaded.meta['note'] == 'unit'
    assert loaded.input_dim == heads.input_dim == E_va.shape[1]
    np.testing.assert_allclose(loaded.transform(E_va), heads.transform(E_va),
                               rtol=1e-6)


def test_transform_input_dim_mismatch_raises(fitted_heads):
    heads, E_va = fitted_heads
    bad = np.hstack([E_va, np.zeros((len(E_va), 3), np.float32)])
    with pytest.raises(ValueError, match='input_dim'):
        heads.transform(bad)


# ---------------------------------------------------------------------------
# context_at — enriched (emb+ff68) vs emb-only input build
# ---------------------------------------------------------------------------

D_EMB, K_FF = 16, 8


def _ohlcv_df(n=400):
    ts = pd.date_range('2024-01-01', periods=n, freq='3min', tz='UTC')
    close = 100 * np.exp(np.cumsum(RNG.normal(0, .001, n)))
    return pd.DataFrame({'datetime': ts, 'open': close,
                         'high': close * 1.001, 'low': close * 0.999,
                         'close': close, 'volume': np.full(n, 1000.0)})


def _fake_embed_bars(d):
    def f(close, indices, ctx=128, batch=64):
        out = np.zeros((len(indices), d), np.float32)
        out[:, 0] = np.asarray(close, float)[np.asarray(indices)]
        return out
    return f


def test_context_at_enriched_hstacks_embedding_and_features(monkeypatch):
    """Enriched bundle: X = [embed_bars | context_features[indices]] —
    monkeypatched embed + features (no subprocess, no heavy
    derive_features)."""
    from futures_foundation.extractors.chronos import backbone as foundation
    E_tr, lab_tr, E_va, lab_va = _synthetic_dataset(d=D_EMB + K_FF)
    heads = ContextHeads(seed=0, n_estimators=40).fit(
        E_tr, lab_tr, E_va, lab_va, verbose=False)
    heads.meta = {'inputs': 'emb+ff68'}
    monkeypatch.setattr(foundation, 'embed_bars', _fake_embed_bars(D_EMB))
    seen = {}

    def fake_features(df, instrument):
        seen['instrument'] = instrument
        return RNG.normal(0, 1, (len(df), K_FF)).astype(np.float32)

    monkeypatch.setattr(ctx, 'context_features', fake_features)
    df = _ohlcv_df()
    idx = [200, 300, 399]
    out = heads.context_at(df, idx, 'ES')
    assert seen['instrument'] == 'ES'
    assert list(out.index) == idx
    assert list(out.columns) == heads.active_names
    assert out.shape == (3, len(heads.active_names))
    assert out.notna().all().all()


def test_context_at_emb_only_backcompat_skips_features(monkeypatch,
                                                       fitted_heads):
    """Old emb-only bundle (no meta inputs): embedding alone, the feature
    library must never be touched."""
    from futures_foundation.extractors.chronos import backbone as foundation
    heads, _ = fitted_heads
    monkeypatch.setattr(foundation, 'embed_bars', _fake_embed_bars(24))

    def boom(df, instrument):
        raise AssertionError('context_features must not be called for '
                             'emb-only bundles')

    monkeypatch.setattr(ctx, 'context_features', boom)
    out = heads.context_at(_ohlcv_df(), [100, 200], 'ES')
    assert out.shape == (2, len(heads.active_names))


def test_htf_context_at_enriched_raises():
    heads = ContextHeads()
    heads.meta = {'inputs': 'emb+ff68'}
    ts = pd.date_range('2023-01-01', periods=10, freq='5min', tz='UTC')
    with pytest.raises(NotImplementedError, match='emb-only'):
        heads.htf_context_at(ts, np.ones(10), [5], htf='1h')


def test_structure_nan_rows_dropped_not_filled(fitted_heads):
    heads, _ = fitted_heads
    m = heads.metrics['structure']
    # synthetic structure has NaN (mixed) rows — they must be excluded
    assert m['n_train'] < 2400 and m['n_val'] < 600


def test_too_few_rows_marks_head_skipped():
    E_tr, lab_tr, E_va, lab_va = _synthetic_dataset(n=400)
    heads = ContextHeads(n_estimators=10).fit(
        E_tr, lab_tr, E_va, lab_va, verbose=False)   # 320 train < 500 min
    assert all(not m['passed'] for m in heads.metrics.values())
    assert all('skipped' in m for m in heads.metrics.values())


def test_heads_cutoff_constant_is_utc_2023():
    assert ctx.HEADS_CUTOFF == pd.Timestamp('2023-01-01', tz='UTC')
    assert ctx.MAX_LABEL_HORIZON == 20
