import numpy as np
import pytest

from futures_foundation.finetune.probe_targets import causal_probe_targets


def test_probe_targets_support_negative_prices_and_are_family_shared():
    close = np.array([[5.0, 2.0, -3.0, -8.0]])
    context = np.stack((close, close + 1.0, close - 1.0, close, np.ones_like(close)), 2)
    future_close = np.array([[-4.0, 1.0]])
    future = np.stack((future_close, future_close + 1.0, future_close - 1.0,
                       future_close, np.ones_like(future_close)), 2)
    result = causal_probe_targets(context, future)
    assert set(result) == {
        "vol", "trend_eff", "range_expand", "fwd_absmove", "direction", "fwd_dir",
    }
    assert all(np.isfinite(value).all() for value in result.values())
    assert result["fwd_dir"][0] == 1


def test_probe_targets_reject_invalid_geometry():
    context = np.ones((1, 4, 5), np.float64)
    future = np.ones((1, 2, 5), np.float64)
    context[:, :, 1] = 0.0
    with pytest.raises(ValueError, match="geometry"):
        causal_probe_targets(context, future)
