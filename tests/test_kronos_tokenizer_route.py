import numpy as np
import pytest

from futures_foundation.finetune.routes import kronos_tokenizer as route


def _parent(batch: int = 2) -> np.ndarray:
    time = np.arange(route.PARENT_LENGTH, dtype=np.float32)
    close = 50.0 + 0.01 * time[None, :] + np.arange(batch, dtype=np.float32)[:, None]
    open_ = np.concatenate((close[:, :1], close[:, :-1]), axis=1)
    high = np.maximum(open_, close) + 0.25
    low = np.minimum(open_, close) - 0.25
    volume = np.broadcast_to(1000.0 + time[None, :], close.shape)
    return np.stack((open_, high, low, close, volume), axis=-1).astype(np.float32)


class _MatrixCodeTokenizer:
    def encode(self, values, half=False):
        import torch

        assert half is True
        batch = values.shape[0]
        return (
            torch.zeros((batch, route.PARENT_LENGTH), dtype=torch.int64, device=values.device),
            torch.ones((batch, route.PARENT_LENGTH), dtype=torch.int32, device=values.device),
        )


class _BadCodeTokenizer:
    def encode(self, values, half=False):
        import torch

        batch = values.shape[0]
        return (
            torch.zeros((batch, route.PARENT_LENGTH, 1), dtype=torch.int64, device=values.device),
            torch.ones((batch, route.PARENT_LENGTH), dtype=torch.int64, device=values.device),
        )


def test_kronos_tokenizer_codes_are_two_integer_matrices():
    coarse, fine = route.tokenizer_codes(
        _MatrixCodeTokenizer(), _parent(), device="cpu"
    )
    assert tuple(coarse.shape) == (2, route.PARENT_LENGTH)
    assert tuple(fine.shape) == (2, route.PARENT_LENGTH)
    assert not coarse.dtype.is_floating_point
    assert not fine.dtype.is_floating_point


def test_kronos_tokenizer_rejects_noncanonical_code_geometry():
    with pytest.raises(RuntimeError, match="integer matrices"):
        route.tokenizer_codes(_BadCodeTokenizer(), _parent(), device="cpu")


def test_kronos_amount_and_normalization_contract_are_exact():
    parent = _parent(1)
    native = route.native_ohlcva(parent)
    expected_amount = parent[:, :, 4] * parent[:, :, :4].mean(axis=2)
    np.testing.assert_allclose(native[:, :, 5], expected_amount, rtol=0.0, atol=0.0)

    normalized, mean, std = route.normalize(parent)
    assert normalized.shape == (1, route.PARENT_LENGTH, len(route.NATIVE_CHANNELS))
    np.testing.assert_allclose(mean, native.mean(axis=1, keepdims=True), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        std,
        np.maximum(native.std(axis=1, keepdims=True), 1e-5),
        rtol=0.0,
        atol=0.0,
    )
    assert np.isfinite(normalized).all()
    assert np.max(np.abs(normalized)) <= 5.0
