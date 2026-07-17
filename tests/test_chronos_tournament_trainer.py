from argparse import Namespace

import numpy as np
import pytest

from scripts.train_chronos_tournament import (
    _config_signature, _split_parent, _univariate_series,
)


def test_parent_split_is_strictly_causal_and_leaves_protocol_guard_row():
    parent = np.arange(2 * 273 * 5, dtype=np.float32).reshape(2, 273, 5)
    context, future = _split_parent(parent)
    assert context.shape == (2, 256, 5)
    assert future.shape == (2, 16, 5)
    np.testing.assert_array_equal(context[:, -1], parent[:, 255])
    np.testing.assert_array_equal(future[:, 0], parent[:, 256])
    np.testing.assert_array_equal(future[:, -1], parent[:, 271])
    assert not np.shares_memory(future[:, -1], parent[:, 272])


def test_stage1_reconstruction_uses_only_the_historical_context():
    parent = np.arange(2 * 273 * 5, dtype=np.float32).reshape(2, 273, 5)
    context, target = _split_parent(parent, stage="stage1_reconstruction")
    assert context.shape == (2, 240, 5)
    assert target.shape == (2, 16, 5)
    np.testing.assert_array_equal(context[:, -1], parent[:, 239])
    np.testing.assert_array_equal(target[:, 0], parent[:, 240])
    np.testing.assert_array_equal(target[:, -1], parent[:, 255])


def test_resume_signature_allows_only_resume_transport_fields_to_change():
    base = Namespace(family="chronos_v1", max_steps=8, output="a.pt",
                     resume=False, stop_after=2, learning_rate=1e-5)
    changed = Namespace(family="chronos_v1", max_steps=8, output="b.pt",
                        resume=True, stop_after=None, learning_rate=1e-5)
    assert _config_signature(base) == _config_signature(changed)
    changed.learning_rate = 2e-5
    assert _config_signature(base) != _config_signature(changed)


def test_parent_split_rejects_short_or_malformed_inputs():
    with pytest.raises(ValueError, match="parent windows"):
        _split_parent(np.zeros((2, 271, 5), np.float32))
    with pytest.raises(ValueError, match="parent windows"):
        _split_parent(np.zeros((2, 273), np.float32))


def test_univariate_families_receive_every_ohlcv_channel_without_mixing_anchors():
    context = np.arange(2 * 256 * 5, dtype=np.float32).reshape(2, 256, 5)
    future = np.arange(2 * 16 * 5, dtype=np.float32).reshape(2, 16, 5)
    x, y = _univariate_series(context, future, "channel_independent_ohlcv")
    assert x.shape == (10, 256) and y.shape == (10, 16)
    np.testing.assert_array_equal(x[0], context[0, :, 0])
    np.testing.assert_array_equal(x[4], context[0, :, 4])
    np.testing.assert_array_equal(x[5], context[1, :, 0])
    close_x, close_y = _univariate_series(context, future, "close_only")
    np.testing.assert_array_equal(close_x, context[:, :, 3])
    np.testing.assert_array_equal(close_y, future[:, :, 3])
