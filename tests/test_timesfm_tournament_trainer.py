import numpy as np

from futures_foundation.finetune.tournament import FORECAST_HORIZON, MAX_CONTEXT
from scripts.train_timesfm_tournament import PARENT_LENGTH, _channel_independent


def test_timesfm_uses_all_five_channels_without_cross_anchor_mixing():
    parent = np.arange(2 * PARENT_LENGTH * 5, dtype=np.float32).reshape(2, PARENT_LENGTH, 5)
    context, future = _channel_independent(parent)
    assert context.shape == (10, MAX_CONTEXT)
    assert future.shape == (10, FORECAST_HORIZON)
    np.testing.assert_array_equal(context[0], parent[0, :MAX_CONTEXT, 0])
    np.testing.assert_array_equal(context[4], parent[0, :MAX_CONTEXT, 4])
    np.testing.assert_array_equal(context[5], parent[1, :MAX_CONTEXT, 0])
    np.testing.assert_array_equal(future[9], parent[1, MAX_CONTEXT:, 4])


def test_timesfm_stage1_cannot_read_later_bars():
    parent = np.arange(PARENT_LENGTH * 5, dtype=np.float32).reshape(1, PARENT_LENGTH, 5)
    changed = parent.copy()
    changed[:, MAX_CONTEXT:] += 1_000_000
    context, target = _channel_independent(parent, "stage1_reconstruction")
    other_context, other_target = _channel_independent(changed, "stage1_reconstruction")
    np.testing.assert_array_equal(context, other_context)
    np.testing.assert_array_equal(target, other_target)
