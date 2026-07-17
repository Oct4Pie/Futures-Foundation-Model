import numpy as np
import pytest

from futures_foundation.finetune.moment_eval import (
    left_pad_contexts, pool_channel_patches, targets_from_context_future,
)


def test_left_pad_and_channel_preserving_patch_pool():
    context = np.arange(2 * 16 * 5, dtype=np.float32).reshape(2, 16, 5)
    padded, mask = left_pad_contexts(context, native_length=32)
    assert padded.shape == (2, 5, 32)
    assert mask.shape == (2, 32)
    np.testing.assert_array_equal(padded[:, :, -16:], context.transpose(0, 2, 1))
    assert not padded[:, :, :16].any()
    assert not mask[:, :16].any() and mask[:, -16:].all()

    # Four patches: the two left-padding patches are deliberately extreme and must be ignored.
    embeddings = np.zeros((2, 5, 4, 3), np.float32)
    embeddings[:, :, :2] = 999
    embeddings[:, :, 2] = np.arange(5)[None, :, None]
    embeddings[:, :, 3] = (np.arange(5) + 2)[None, :, None]
    pooled = pool_channel_patches(embeddings, mask, patch_len=8)
    expected = (np.arange(5) + 1).repeat(3)
    np.testing.assert_allclose(pooled[0], expected)
    assert pooled.shape == (2, 15)


def test_moment_contract_rejects_invalid_shapes():
    with pytest.raises(ValueError, match="exceeds"):
        left_pad_contexts(np.zeros((1, 33, 5), np.float32), native_length=32)
    with pytest.raises(ValueError, match="divisible"):
        pool_channel_patches(np.zeros((1, 1, 2, 3)), np.ones((1, 15)), patch_len=8)


def test_targets_match_plain_price_action_definitions():
    n, context_len, horizon = 8, 16, 4
    base = np.arange(context_len + horizon, dtype=np.float64)
    context, future = [], []
    for row in range(n):
        close = 100 * np.exp((row - 3) * 0.0002 * base + 0.001 * np.sin(base))
        values = np.stack(
            [close, close + .5, close - .5, close, 100 + base], axis=1
        )
        context.append(values[:context_len])
        future.append(values[context_len:])
    targets = targets_from_context_future(context, future)
    assert set(targets) == {
        "vol", "trend_eff", "range_expand", "fwd_absmove", "direction", "fwd_dir"
    }
    assert all(value.shape == (n,) for value in targets.values())
    assert np.isfinite(np.concatenate([x.astype(float) for x in targets.values()])).all()
