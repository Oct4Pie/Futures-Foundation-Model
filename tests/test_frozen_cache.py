"""Embedding-cache behavior for MantisFrozenClassifier (torch-free; _embed is mocked).

Covers: legacy per-keyset cache is preferred (WF caching preserved), bar-indexed cache lets
SUBSETS reuse one embed (produce train/val/oos), and partial requests only embed missing bars.
"""
import numpy as np

from futures_foundation.finetune.classifiers.mantis.frozen import (
    MantisFrozenClassifier, _embed_cache_path, _save_bar_cache, _load_bar_cache, _bar_cache_path)


class FakeLab:
    MV_MODE = 'ohlcv'
    MV_SEQ = 64

    def __init__(self, nbars=500):
        self._b = {('NQ', '3min'): {'c': np.zeros(nbars, np.float32)}}

    def mv_contexts(self, keys):
        return np.zeros((len(keys), 5, self.MV_SEQ), np.float32)

    def features(self, keys):
        # 3 named handcraft cols, distinct values per col so selection is verifiable
        return np.tile(np.array([10., 20., 30.], np.float32), (len(keys), 1))

    def feature_names(self):
        return ['a', 'b', 'c']


def _keys(bars):
    return [("NQ@3min", int(i), 1, 0, 0.0, 0.0, 0.0, 0.0) for i in bars]


def _emb_for(keys):
    """deterministic fake embedding: each row = its bar index repeated (so we can verify slices)."""
    return np.array([[float(k[1])] * 4 for k in keys], np.float32)


def test_bar_cache_roundtrip(tmp_path):
    p = tmp_path / 'b.npz'
    idx = np.array([3, 7, 9], np.int64); emb = np.random.rand(3, 4).astype(np.float32)
    _save_bar_cache(p, idx, emb)
    ci, ce = _load_bar_cache(p)
    assert list(ci) == [3, 7, 9] and np.allclose(ce, emb)


def test_bar_cache_subset_reuse_and_partial(tmp_path, monkeypatch):
    monkeypatch.setenv('EMBED_CACHE_DIR', str(tmp_path))
    monkeypatch.setenv('EMBED_CACHE', '1')
    clf = MantisFrozenClassifier(backbone_ckpt=None, with_features=False)
    calls = []

    def fake_embed(labeler, keys):
        calls.append(sorted(int(k[1]) for k in keys))
        return _emb_for(keys)
    clf._embed = fake_embed
    lab = FakeLab()

    # seed bars 0..19
    X = clf.featurize(lab, _keys(range(20)))
    assert X.shape[0] == 20 and calls == [list(range(20))]

    # SUBSET 5..9 -> full HIT, no new embed, correct slice
    calls.clear()
    Xs = clf.featurize(lab, _keys(range(5, 10)))
    assert calls == [] and np.allclose(Xs[:, 0, 0], [5, 6, 7, 8, 9])

    # SUPERSET 18..24 -> only 20..24 embedded (missing), values still correct
    calls.clear()
    Xs2 = clf.featurize(lab, _keys(range(18, 25)))
    assert calls == [[20, 21, 22, 23, 24]]
    assert np.allclose(Xs2[:, 0, 0], list(range(18, 25)))


def test_legacy_cache_preferred(tmp_path, monkeypatch):
    """An existing legacy per-keyset cache (WF) is used as-is — no bar lookup, no embed."""
    monkeypatch.setenv('EMBED_CACHE_DIR', str(tmp_path))
    monkeypatch.setenv('EMBED_CACHE', '1')
    clf = MantisFrozenClassifier(backbone_ckpt=None, with_features=False)
    lab = FakeLab()
    keys = _keys(range(10))
    legacy = _embed_cache_path({'backbone_ckpt': None}, lab, keys)
    legacy.parent.mkdir(parents=True, exist_ok=True)
    np.save(legacy, np.full((10, 7), 9.0, np.float32))      # pretend WF wrote this

    def boom(*a, **k):
        raise AssertionError("should not embed when legacy cache hits")
    clf._embed = boom
    X = clf.featurize(lab, keys)
    assert X.shape[0] == 10 and np.allclose(X[:, 0, 0], 9.0)


def test_bar_cache_multi_tf_mode_distinct_and_reusable(tmp_path, monkeypatch):
    """Multi-TF modes (MV_MODE='ohlcv_agg1-5') get their OWN bar-cache file — never colliding with
    the single-TF cache — and the plain 'ohlcv' filename is UNCHANGED (existing caches keep
    hitting). Non-ohlcv modes stay uncached."""
    monkeypatch.setenv('EMBED_CACHE_DIR', str(tmp_path))
    monkeypatch.setenv('EMBED_CACHE', '1')
    keys = _keys(range(5))
    single = FakeLab()
    multi = FakeLab(); multi.MV_MODE = 'ohlcv_agg1-5'
    other = FakeLab(); other.MV_MODE = 'zscore'
    p_single = _bar_cache_path({'backbone_ckpt': None}, single, keys)
    p_multi = _bar_cache_path({'backbone_ckpt': None}, multi, keys)
    assert p_single is not None and p_multi is not None
    assert p_single != p_multi                                 # distinct files, no collision
    assert p_single.name.endswith('_64.npz')                   # plain-ohlcv filename unchanged
    assert 'ohlcv_agg1-5' in p_multi.name                      # mode-tagged
    assert _bar_cache_path({'backbone_ckpt': None}, other, keys) is None   # non-ohlcv -> uncached

    # the multi-TF cache round-trips through featurize (seed then subset-hit, no re-embed)
    clf = MantisFrozenClassifier(backbone_ckpt=None, with_features=False)
    calls = []
    clf._embed = lambda l, k: (calls.append(sorted(int(x[1]) for x in k)) or _emb_for(k))
    clf.featurize(multi, _keys(range(10)))
    assert calls == [list(range(10))]
    calls.clear()
    Xs = clf.featurize(multi, _keys(range(3, 7)))
    assert calls == [] and np.allclose(Xs[:, 0, 0], [3, 4, 5, 6])


def test_cache_off(tmp_path, monkeypatch):
    monkeypatch.setenv('EMBED_CACHE', '0')
    clf = MantisFrozenClassifier(backbone_ckpt=None, with_features=False)
    lab = FakeLab()
    assert _bar_cache_path({'backbone_ckpt': None}, lab, _keys(range(3))) is None
    n = []
    clf._embed = lambda l, k: (n.append(1) or _emb_for(k))
    clf.featurize(lab, _keys(range(3)))
    assert n == [1]                                          # embedded directly, no cache
