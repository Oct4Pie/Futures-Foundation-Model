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


# ---- channel namespacing (volume embed) -----------------------------------
def test_signature_channel_namespaces_but_price_unchanged(stub_backbone):
    """'volume' channel → distinct namespace; None/'price' → ORIGINAL signature
    (so existing price caches stay valid)."""
    h_default, _ = EC.signature('mean')
    h_price, _ = EC.signature('mean', channel='price')
    h_none, _ = EC.signature('mean', channel=None)
    h_vol, _ = EC.signature('mean', channel='volume')
    assert h_default == h_price == h_none          # price path unchanged
    assert h_vol != h_default                       # volume is its own namespace


def test_volume_and_price_dont_collide_same_key(stub_backbone, tmp_path):
    """Price and volume embeds for the SAME (item, bar) must not overwrite each
    other — they live in separate channel namespaces."""
    ctx_p, keys = _data(n=6, seed=0)
    ctx_v = [c + 50.0 for c in ctx_p]              # different content, same keys
    ep = EC.embed_with_cache(ctx_p, keys, tmp_path, verbose=False)
    ev = EC.embed_with_cache(ctx_v, keys, tmp_path, verbose=False, channel='volume')
    # both correct (no collision), and re-reading each hits its own cache
    np.testing.assert_allclose(ep, _ref(ctx_p), atol=1e-5)
    np.testing.assert_allclose(ev, _ref(ctx_v), atol=1e-5)
    base = stub_backbone['embedded']
    ep2 = EC.embed_with_cache(ctx_p, keys, tmp_path, verbose=False)
    ev2 = EC.embed_with_cache(ctx_v, keys, tmp_path, verbose=False, channel='volume')
    assert stub_backbone['embedded'] == base       # both full cache hits, no re-embed
    np.testing.assert_array_equal(ep, ep2)
    np.testing.assert_array_equal(ev, ev2)
