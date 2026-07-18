import os
from types import SimpleNamespace

import numpy as np
import pytest

from futures_foundation.finetune.native_adapters import (
    NativeAdapterError,
    left_pad_channel_first,
    moirai2_native_forecast,
    moment_native_embedding,
    sundial_native_forecast,
    timesfm25_transformers_forecast,
    toto2_native_forecast,
    ttm_native_forecast,
    validate_tabpfn_fold_containment,
)


TORCH_TEST = pytest.mark.skipif(
    os.environ.get("CHRONOS_TORCH_TESTS") != "1",
    reason="set CHRONOS_TORCH_TESTS=1 for isolated Torch adapter tests",
)


def test_left_pad_channel_first_preserves_suffix_and_mask():
    values = np.arange(2 * 3 * 2, dtype=np.float32).reshape(2, 3, 2)
    padded, mask = left_pad_channel_first(values, target_length=5)
    assert padded.shape == (2, 2, 5)
    assert mask.shape == (2, 5)
    np.testing.assert_array_equal(padded[:, :, -3:], values.transpose(0, 2, 1))
    np.testing.assert_array_equal(padded[:, :, :2], 0)
    np.testing.assert_array_equal(mask[:, :2], 0)
    np.testing.assert_array_equal(mask[:, -3:], 1)


class _MomentModel:
    def embed(self, *, x_enc, input_mask, reduction):
        assert reduction == "mean"
        assert x_enc.shape[-1] == input_mask.shape[-1]
        return SimpleNamespace(embeddings=x_enc.mean(dim=(1, 2))[:, None])


@TORCH_TEST
def test_moment_adapter_uses_official_mean_surface():
    import torch

    values = torch.arange(2 * 3 * 8, dtype=torch.float32).reshape(2, 3, 8)
    mask = torch.ones((2, 8), dtype=torch.float32)
    output = moment_native_embedding(_MomentModel(), values, mask)
    assert output.shape == (2, 1)
    assert torch.isfinite(output).all()


class _TimesFmModel:
    def __init__(self):
        self.kwargs = None

    def __call__(self, **kwargs):
        import torch

        self.kwargs = kwargs
        batch = kwargs["past_values"].shape[0]
        return SimpleNamespace(
            mean_predictions=torch.arange(batch * 12, dtype=torch.float32).reshape(batch, 12),
            full_predictions=torch.ones((batch, 12, 10), dtype=torch.float32),
        )


@TORCH_TEST
def test_timesfm_adapter_locks_flip_and_raw_quantile_contract():
    import torch

    model = _TimesFmModel()
    point, quantiles = timesfm25_transformers_forecast(
        model, torch.ones((3, 64)), prediction_length=8, context_length=64
    )
    assert point.shape == (3, 8)
    assert quantiles.shape == (3, 8, 10)
    assert model.kwargs["forecast_context_len"] == 64
    assert model.kwargs["force_flip_invariance"] is True
    assert model.kwargs["truncate_negative"] is False


class _TTMModel:
    def __init__(self):
        self.kwargs = None

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        values = kwargs["past_values"]
        return SimpleNamespace(prediction_outputs=values[:, -16:, :])


@TORCH_TEST
def test_ttm_adapter_uses_raw_512_by_5_values_and_frequency_prefix():
    import torch

    model = _TTMModel()
    values = torch.randn(2, 512, 5)
    token = torch.tensor([7, 7])
    output = ttm_native_forecast(model, values, token)
    assert output.shape == (2, 16, 5)
    assert model.kwargs["past_values"] is values
    assert model.kwargs["freq_token"] is token
    assert model.kwargs["return_loss"] is False
    with pytest.raises(NativeAdapterError, match=r"\[B,512,5\]"):
        ttm_native_forecast(model, values[:, :256], token)


class _MoiraiModel:
    def __init__(self):
        self.past_target = None

    def __call__(self, **kwargs):
        import torch

        self.past_target = kwargs["past_target"].clone()
        batch, _, channels = self.past_target.shape
        return torch.zeros((batch, 9, 16, channels), dtype=self.past_target.dtype)


@TORCH_TEST
def test_moirai_adapter_zero_fills_masked_values_before_packing():
    import torch

    model = _MoiraiModel()
    values = torch.ones((2, 32, 5))
    values[:, 7, :] = 99999
    observed = torch.ones_like(values, dtype=torch.bool)
    observed[:, 7, :] = False
    output = moirai2_native_forecast(
        model, values, observed, torch.zeros((2, 32), dtype=torch.bool)
    )
    assert output.shape == (2, 9, 16, 5)
    assert torch.equal(model.past_target[:, 7, :], torch.zeros((2, 5)))


class _TotoModel:
    def __init__(self):
        self.inputs = None
        self.kwargs = None

    def forecast(self, inputs, horizon, **kwargs):
        import torch

        self.inputs = inputs
        self.kwargs = {"horizon": horizon, **kwargs}
        batch, channels, _ = inputs["target"].shape
        return torch.zeros((9, batch, channels, horizon), dtype=inputs["target"].dtype)


@TORCH_TEST
def test_toto_adapter_zero_fills_masks_and_disables_decode_blocking():
    import torch

    model = _TotoModel()
    values = torch.ones((2, 5, 32))
    values[:, :, 9] = 99999
    mask = torch.ones_like(values, dtype=torch.bool)
    mask[:, :, 9] = False
    groups = torch.zeros((2, 5), dtype=torch.long)
    output = toto2_native_forecast(
        model, values, mask, groups, prediction_length=16
    )
    assert output.shape == (9, 2, 5, 16)
    assert torch.equal(model.inputs["target"][:, :, 9], torch.zeros((2, 5)))
    assert model.kwargs == {
        "horizon": 16,
        "decode_block_size": None,
        "has_missing_values": True,
    }


class _SundialModel:
    def generate(self, values, *, max_new_tokens, num_samples):
        import torch

        return torch.rand((values.shape[0], num_samples, max_new_tokens))


@TORCH_TEST
def test_sundial_adapter_is_seeded_and_restores_global_rng_state():
    import torch

    model = _SundialModel()
    values = torch.ones((2, 32))
    torch.manual_seed(90)
    before = torch.random.get_rng_state().clone()
    first = sundial_native_forecast(
        model, values, prediction_length=8, num_samples=4, seed=123
    )
    after = torch.random.get_rng_state()
    second = sundial_native_forecast(
        model, values, prediction_length=8, num_samples=4, seed=123
    )
    assert torch.equal(before, after)
    assert torch.equal(first, second)
    assert first.shape == (2, 4, 8)


def test_tabpfn_fold_containment_rejects_support_or_query_leakage():
    validate_tabpfn_fold_containment(
        [10, 20, 30], [41, 42], fold_train_end_ns=40
    )
    with pytest.raises(NativeAdapterError, match="after the training-fold boundary"):
        validate_tabpfn_fold_containment(
            [10, 41], [42], fold_train_end_ns=40
        )
    with pytest.raises(NativeAdapterError, match="strictly after"):
        validate_tabpfn_fold_containment(
            [10, 20], [40, 41], fold_train_end_ns=40
        )
