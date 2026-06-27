"""Unit tests for the disk-backed embedding cache (torch-free — backbone.embed
is mocked). Verifies: cache HIT correctness, no re-embed on hit, content-hash
invalidation when a context changes, and recipe-signature namespacing."""
import numpy as np
import pytest

from futures_foundation.pipeline import embed_cache as EC


@pytest.fixture
def stub_backbone(monkeypatch):
    """Mock backbone so the cache logic is tested without torch. Returns a dict
    tracking how many contexts were actually embedded."""
    from futures_foundation.extractors.chronos import backbone
    state = {'embedded': 0, 'rs': False}

    def fake_embed(contexts, pool='mean'):
        state['embedded'] += len(contexts)
        # deterministic 2-d "embedding": [mean, std] of each window
        return np.array([[float(np.mean(c)), float(np.std(c))] for c in contexts],
                        np.float32)

    monkeypatch.setattr(backbone, 'embed', fake_embed)
    monkeypatch.setattr(backbone, 'active_source', lambda: 'test-model')
    monkeypatch.setattr(backbone, '_return_shape_on', lambda: state['rs'])
    monkeypatch.setattr(backbone, 'CTX', 8, raising=False)
    monkeypatch.setattr(backbone, 'D_MODEL', 2, raising=False)
    return state


def _data(n=10, seed=0):
    rng = np.random.default_rng(seed)
    ctx = [rng.standard_normal(8).astype(np.float32) for _ in range(n)]
    keys = [('NQ@3min', i) for i in range(n)]
    return ctx, keys


def _ref(ctx):
    return np.array([[float(np.mean(c)), float(np.std(c))] for c in ctx], np.float32)


def test_first_call_embeds_all_and_is_correct(stub_backbone, tmp_path):
    ctx, keys = _data()
    e = EC.embed_with_cache(ctx, keys, tmp_path, verbose=False)
    assert stub_backbone['embedded'] == 10
    np.testing.assert_allclose(e, _ref(ctx), atol=1e-5)


def test_second_call_all_hits_no_reembed(stub_backbone, tmp_path):
    ctx, keys = _data()
    e1 = EC.embed_with_cache(ctx, keys, tmp_path, verbose=False)
    n_after_first = stub_backbone['embedded']
    e2 = EC.embed_with_cache(ctx, keys, tmp_path, verbose=False)
    assert stub_backbone['embedded'] == n_after_first      # nothing re-embedded
    np.testing.assert_array_equal(e1, e2)


def test_content_change_reembeds_only_that_bar(stub_backbone, tmp_path):
    ctx, keys = _data()
    EC.embed_with_cache(ctx, keys, tmp_path, verbose=False)
    base = stub_backbone['embedded']
    ctx2 = [c.copy() for c in ctx]
    ctx2[3] = ctx2[3] + 99.0                                # same key, new content
    e = EC.embed_with_cache(ctx2, keys, tmp_path, verbose=False)
    assert stub_backbone['embedded'] == base + 1           # only bar 3 re-embedded
    np.testing.assert_allclose(e, _ref(ctx2), atol=1e-5)   # correct new values


def test_signature_namespacing_invalidates(stub_backbone, tmp_path):
    ctx, keys = _data()
    EC.embed_with_cache(ctx, keys, tmp_path, verbose=False)
    base = stub_backbone['embedded']
    stub_backbone['rs'] = True                              # recipe changed
    EC.embed_with_cache(ctx, keys, tmp_path, verbose=False)
    assert stub_backbone['embedded'] == base + len(ctx)     # all re-embedded


def test_partial_overlap_only_new_keys_embed(stub_backbone, tmp_path):
    ctx, keys = _data(n=10)
    EC.embed_with_cache(ctx, keys, tmp_path, verbose=False)
    base = stub_backbone['embedded']
    # add 5 new bars to the same stream; first 10 should hit
    rng = np.random.default_rng(1)
    new_ctx = [rng.standard_normal(8).astype(np.float32) for _ in range(5)]
    new_keys = [('NQ@3min', 10 + i) for i in range(5)]
    EC.embed_with_cache(ctx + new_ctx, keys + new_keys, tmp_path, verbose=False)
    assert stub_backbone['embedded'] == base + 5
