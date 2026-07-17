"""Memmap streaming (full-data path) — torch-free.

Verifies featurize_to_memmap writes a correct disk-backed array in chunks, the
streaming standardize stats are sane, and produce.train_final(stream=True) runs the
ENTIRE memmap path end-to-end (featurize-to-disk -> per-sample load -> metrics ->
contract) via the torch-free 'logistic' classifier (which loads memmap paths).
"""
import json

import numpy as np
import pandas as pd

from futures_foundation.finetune import produce
from futures_foundation.finetune._memmap import (featurize_to_memmap,
                                                 memmap_standardize_stats)
from futures_foundation.finetune.classifier import get_classifier


class _StubLabeler:
    def __init__(self, X):
        self._X = X
    def mv_contexts(self, keys):
        return np.stack([self._X[k] for k in keys])


def test_featurize_to_memmap_chunked_matches(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 4, 16)).astype(np.float32)
    lab = _StubLabeler(X)
    clf = get_classifier('logistic')
    path = str(tmp_path / 'X.npy')
    p, shape = featurize_to_memmap(clf, lab, list(range(50)), path, chunk=7)
    assert shape == (50, 4, 16)
    mm = np.load(path, mmap_mode='r')
    assert mm.shape == (50, 4, 16)
    assert np.allclose(np.asarray(mm), X, atol=1e-5)   # chunked write == direct featurize


def test_memmap_standardize_stats(tmp_path):
    rng = np.random.default_rng(1)
    X = (rng.standard_normal((200, 3, 8)).astype(np.float32) * 5 + 2)
    path = str(tmp_path / 'X.npy')
    featurize_to_memmap(get_classifier('logistic'), _StubLabeler(X),
                        list(range(200)), path, chunk=32)
    mu, sd = memmap_standardize_stats(path, sample=200)
    assert mu.shape == (3,) and sd.shape == (3,)
    # per-channel mean ~2, std ~5 (the planted distribution)
    assert np.allclose(mu, 2.0, atol=0.6) and np.allclose(sd, 5.0, atol=1.0)


class _StreamLabeler:
    """Daily-bar synthetic with mv_contexts (separable signal) for the full produce
    stream path via logistic."""
    n_classes = 2

    def __init__(self, n_bars=1600, seq=32, C=4, seed=0):
        rng = np.random.default_rng(seed)
        self.ts = pd.date_range('2020-01-01', periods=n_bars, freq='1D', tz='UTC')
        self.y = rng.integers(0, 2, n_bars)
        self.W = rng.standard_normal((n_bars, C, seq)).astype(np.float32)
        self.W[self.y == 1, 0, -6:] += 2.5
        self.seq, self.C = seq, C

    def calendar(self):
        return pd.DataFrame({'item_id': 'SYN', 'timestamp': self.ts, 'target': self.y})

    def build(self, lo, hi, test_start):
        i = np.flatnonzero((self.ts >= lo) & (self.ts < hi))
        return [None] * len(i), self.y[i], [(int(j),) for j in i]

    def mv_contexts(self, keys):
        return (np.stack([self.W[k[0]] for k in keys]) if keys
                else np.zeros((0, self.C, self.seq)))

    def mv_feature_names(self):
        return [f'ch{i}' for i in range(self.C)]

    def evaluate(self, keys, preds):
        return np.array([(2.0 if self.y[k[0]] == 1 else -1.0)
                         for k, p in zip(keys, preds) if p == 1])


class _RKeyLabeler:
    """Per-stream labeler with R baked into the key tuple (like the fractal labeler) so
    any instance's evaluate works on accumulated cross-stream keys."""
    n_classes = 2

    def __init__(self, n_bars=1000, seq=32, C=4, seed=0):
        rng = np.random.default_rng(seed)
        self.ts = pd.date_range('2020-01-01', periods=n_bars, freq='1D', tz='UTC')
        self.y = rng.integers(0, 2, n_bars)
        self.W = rng.standard_normal((n_bars, C, seq)).astype(np.float32)
        self.W[self.y == 1, 0, -6:] += 2.5
        self.seq, self.C = seq, C

    def calendar(self):
        return pd.DataFrame({'item_id': 'S', 'timestamp': self.ts, 'target': self.y})

    def build(self, lo, hi, test_start):
        i = np.flatnonzero((self.ts >= lo) & (self.ts < hi))
        K = [(int(j), 2.0 if self.y[j] == 1 else -1.0) for j in i]   # R baked in key
        return [None] * len(i), self.y[i], K

    def mv_contexts(self, keys):
        return (np.stack([self.W[k[0]] for k in keys]) if keys
                else np.zeros((0, self.C, self.seq)))

    def mv_feature_names(self):
        return [f'ch{i}' for i in range(self.C)]

    def evaluate(self, keys, preds):
        return np.array([k[1] for k, p in zip(keys, preds) if p == 1])


def test_produce_streamed_per_stream(tmp_path):
    def make_labeler(tk, tf):
        return _RKeyLabeler(n_bars=1000, seed=abs(hash((tk, tf))) % 1000)
    streams = [('A', '3min'), ('B', '3min'), ('C', '3min')]
    out = produce.train_final_streamed(
        make_labeler, streams, classifier='logistic', holdout_start='2021-06-01',
        seed=0, chunk=200, export_onnx=True, output_path=str(tmp_path / 'm'), verbose=False)
    assert out['n_oos'] > 0 and out['oos_trades'] > 0
    assert out['oos_auc'] is not None and out['oos_auc'] > 0.75   # learnable across streams
    c = json.loads(open(out['artifacts']['contract']).read())
    assert c['train_scope']['tickers'] == ['A', 'B', 'C']
    assert c['train_scope']['timeframes'] == ['3min']
    assert (tmp_path / '_Xtr.npy').exists()      # concatenated full memmap written


def test_produce_stream_end_to_end(tmp_path):
    lab = _StreamLabeler(n_bars=1600, seed=0)
    out = produce.train_final(lab, classifier='logistic', holdout_start='2023-06-01',
                              seed=0, stream=True, chunk=500, export_onnx=True,
                              output_path=str(tmp_path / 'm'), verbose=False)
    # the stream path (featurize-to-memmap -> per-sample load) produced metrics + contract
    assert out['n_oos'] > 0 and out['oos_trades'] > 0
    assert out['oos_auc'] is not None and out['oos_auc'] > 0.8   # learnable signal holds
    c = json.loads(open(out['artifacts']['contract']).read())
    assert c['input']['channels'] == lab.C and c['channel_names'] == ['ch0', 'ch1', 'ch2', 'ch3']
    # the memmaps were written to the run dir
    assert (tmp_path / '_Xtr.npy').exists()


class _WFStreamLabeler:
    """Per-stream labeler for the streamed walk-forward test: fractal-like keys
    (sid, bar, R), _b['ts'] for fold-window timestamps, separable signal."""
    n_classes = 2

    def __init__(self, tk, tf, n_bars=400, seq=32, C=4, seed=0):
        rng = np.random.default_rng(seed)
        ts = pd.date_range('2020-01-01', periods=n_bars, freq='1D', tz='UTC')
        y = rng.integers(0, 2, n_bars)
        W = rng.standard_normal((n_bars, C, seq)).astype(np.float32)
        W[y == 1, 0, -6:] += 2.5
        self.sid = f'{tk}@{tf}'; self.ts = ts; self.y = y; self.W = W
        self.seq, self.C = seq, C
        self._b = {(tk, tf): {'ts': ts}}

    def calendar(self):
        return pd.DataFrame({'item_id': self.sid, 'timestamp': self.ts, 'target': self.y})

    def build(self, lo, hi, test_start):
        i = np.flatnonzero((self.ts >= lo) & (self.ts < hi))
        K = [(self.sid, int(j), 2.0 if self.y[j] == 1 else -1.0) for j in i]
        return [None] * len(i), self.y[i], K

    def label_end_times(self, keys):
        # Synthetic labels are contemporaneous; real strategies return the final
        # bar consumed by their forward triple-barrier outcome.
        return [self.ts[k[1]] for k in keys]

    def mv_contexts(self, keys):
        return (np.stack([self.W[k[1]] for k in keys]) if keys
                else np.zeros((0, self.C, self.seq)))

    def mv_feature_names(self):
        return [f'ch{i}' for i in range(self.C)]

    def evaluate(self, keys, preds):
        return np.array([k[2] for k, p in zip(keys, preds) if p == 1])


def _mk(tk, tf):
    return _WFStreamLabeler(tk, tf, n_bars=400, seed=abs(hash((tk, tf))) % 1000)


def test_wf_run_streamed(tmp_path):
    from futures_foundation.finetune import wf
    v = wf.run_streamed(_mk, [('A', '3min'), ('B', '3min')], classifier='logistic',
                        train_m=3, val_m=1, test_m=1, max_folds=4,
                        output_path=str(tmp_path / 'm'), chunk=200, verbose=False)
    for k in ('all_pass', 'generalizes', 'auc', 'real_meanR', 'shuffle_meanR',
              'random_meanR', 'gap', 'n_folds'):
        assert k in v
    assert v['n_folds'] >= 1
    assert v['auc'] is not None and v['auc'] > 0.7      # rolling folds learn the signal


def test_wf_loop_streamed_overfit_guard(tmp_path):
    from futures_foundation.finetune import wf
    v = wf.loop_streamed(_mk, [('A', '3min'), ('B', '3min')], classifier='logistic',
                         train_m=3, val_m=1, test_m=1, max_folds=4, max_iters=1, n_trials=2,
                         output_path=str(tmp_path / 'm'), chunk=200, verbose=False)
    # the overfit->Optuna loop ran and returned a config + history
    assert 'history' in v and 'final_config' in v and v['n_folds'] >= 1
    assert v['history'][0]['source'] == 'default'


def test_rolling_folds_excludes_holdout():
    from futures_foundation.finetune.wf import _rolling_folds
    ts = pd.DatetimeIndex(pd.date_range('2024-01-01', periods=900, freq='1D', tz='UTC'))
    folds = _rolling_folds(ts, 3, 1, 1, holdout_start='2026-01-01')
    assert len(folds) >= 1
    cutoff = pd.Timestamp('2026-01-01', tz='UTC')
    for tr, va, te in folds:                        # 2026 NEVER in any fold (reserved OOS)
        for rows in (tr, va, te):
            assert (ts[rows] < cutoff).all()


def test_rolling_folds_purges_forward_labels_at_every_boundary():
    """A streamed fold is keyed by signal time, but its label may finish later.

    Train labels must finish before validation, validation labels before test, and
    the last pre-holdout test labels before the untouched holdout.
    """
    from futures_foundation.finetune.wf import _rolling_folds
    ts = pd.DatetimeIndex(pd.date_range('2024-01-01', '2025-06-30', freq='1D', tz='UTC'))
    label_end = ts + pd.Timedelta(days=10)
    folds = _rolling_folds(ts, 3, 1, 1, holdout_start='2025-07-01',
                           label_end_ts=label_end)
    assert folds
    cutoff = pd.Timestamp('2025-07-01', tz='UTC')
    for tr, va, te in folds:
        val_start = ts[va].min()
        test_start = ts[te].min()
        assert (label_end[tr] < val_start).all()
        assert (label_end[va] < test_start).all()
        assert (label_end[te] < cutoff).all()


def test_run_folds_standardizes_from_each_fold_train_only(tmp_path):
    """Future/holdout rows must not contribute normalization statistics."""
    from futures_foundation.finetune import wf
    from futures_foundation.finetune.classifier import register_classifier, Classifier

    seen = []

    @register_classifier('_capture_fold_std')
    class _CaptureFoldStd(Classifier):
        needs_standardize = True

        def __init__(self, **kw):
            self.kw = kw

        def featurize(self, labeler, keys):
            raise NotImplementedError

        def fit_predict(self, Xtr, ytr, Xval, yval, Xeval, seed=0,
                        keys_tr=None, keys_val=None):
            seen.append(np.asarray(self.kw['standardize_mu'], float))
            nv = len(np.load(Xval, mmap_mode='r'))
            ne = len(np.load(Xeval, mmap_mode='r'))
            return np.linspace(0, 1, nv), np.linspace(0, 1, ne), 0.5

    class _Eval:
        def evaluate(self, keys, preds):
            return np.array([k[2] for k, p in zip(keys, preds) if p == 1])

    # Fold 0 train mean=0; fold 1 train mean=10. All later rows are 1000 and
    # would visibly contaminate either statistic if the full memmap were used.
    X = np.full((60, 1, 2), 1000.0, np.float32)
    X[:10] = 0.0
    X[10:20] = 10.0
    path = str(tmp_path / 'X.npy')
    np.save(path, X)
    Y = (np.arange(60) % 2).astype(int)
    keys = [('S@1', int(i), 2.0 if Y[i] else -1.0) for i in range(60)]
    folds = [(np.arange(0, 10), np.arange(20, 30), np.arange(30, 40)),
             (np.arange(10, 20), np.arange(30, 40), np.arange(40, 50))]
    wf._run_folds('_capture_fold_std', {}, path, Y, keys, _Eval(), tmp_path,
                  folds, seed=0, verbose=False, monitor=None)
    assert len(seen) == 4                       # REAL + SHUFFLE for two folds
    assert np.allclose(seen[0], 0.0) and np.allclose(seen[1], 0.0)
    assert np.allclose(seen[2], 10.0) and np.allclose(seen[3], 10.0)


def test_featurize_fp16_halves_disk_and_flows_end_to_end(tmp_path, monkeypatch):
    """FEATURIZE_FP16=1: memmaps store fp16 (half disk — the Colab local/Drive fix), dtype is
    PRESERVED through concat + slice, stats/heads read fp32-upcast values (contract: ~O(1) embeds
    keep 3 sig digits; every consumer upcasts before standardize/fit)."""
    import os as _os
    from futures_foundation.finetune._memmap import concat_memmaps, slice_memmap
    rng = np.random.default_rng(2)
    X = rng.standard_normal((60, 4, 16)).astype(np.float32)
    lab = _StubLabeler(X)
    clf = get_classifier('logistic')
    p32 = str(tmp_path / 'x32.npy')
    featurize_to_memmap(clf, lab, list(range(60)), p32, chunk=13)
    monkeypatch.setenv('FEATURIZE_FP16', '1')
    p16a = str(tmp_path / 'a16.npy'); p16b = str(tmp_path / 'b16.npy')
    featurize_to_memmap(clf, lab, list(range(30)), p16a, chunk=13)
    featurize_to_memmap(clf, lab, list(range(30, 60)), p16b, chunk=13)
    assert _os.path.getsize(p16a) < 0.6 * _os.path.getsize(p32) // 2 + 200   # ~half per part
    # concat + slice preserve fp16; values ~= fp32 originals
    full, shape = concat_memmaps([(p16a, 30), (p16b, 30)], str(tmp_path / 'full16.npy'))
    mm = np.load(full, mmap_mode='r')
    assert mm.dtype == np.float16 and shape == (60, 4, 16)
    assert np.abs(np.asarray(mm, np.float32) - X).max() < 0.01
    sl, _ = slice_memmap(full, np.arange(10, 40), str(tmp_path / 'sl16.npy'))
    assert np.load(sl, mmap_mode='r').dtype == np.float16
    # stats read fine (upcast inside)
    mu, sd = memmap_standardize_stats(full)
    assert mu.shape == (4,) and np.isfinite(mu).all() and (sd > 0).all()
