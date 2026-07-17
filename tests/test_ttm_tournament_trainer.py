import numpy as np

from scripts.train_ttm_tournament import CONTEXT, PARENT_LENGTH, _normalize_parent


def test_ttm_scaler_cannot_read_future_values():
    rng = np.random.default_rng(7)
    parent = rng.normal(size=(3, PARENT_LENGTH, 5)).astype(np.float32)
    changed = parent.copy()
    changed[:, CONTEXT:] *= 1000
    first, first_mean, first_std = _normalize_parent(parent, 0)
    second, second_mean, second_std = _normalize_parent(changed, 0)
    np.testing.assert_allclose(first[:, :CONTEXT], second[:, :CONTEXT])
    np.testing.assert_allclose(first_mean, second_mean)
    np.testing.assert_allclose(first_std, second_std)
    assert not np.allclose(first[:, CONTEXT:], second[:, CONTEXT:])
