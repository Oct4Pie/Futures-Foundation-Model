from types import SimpleNamespace

import numpy as np
import pytest

from futures_foundation.finetune.native_contracts import get_dossier
from scripts.train_ttm_contrastive import PAIR_LENGTH, _native_contrastive_views
from scripts.train_ttm_tournament import (
    CONTEXT, PARENT_LENGTH, _normalize_parent, _validate_model_contract,
)


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


def test_ttm_contrastive_views_use_two_real_full_native_contexts_without_padding():
    raw = np.arange(2 * PAIR_LENGTH * 5, dtype=np.float32).reshape(2, PAIR_LENGTH, 5)
    first, second = _native_contrastive_views(raw)
    expected_first = _normalize_parent(raw[:, :CONTEXT], 0, CONTEXT)[0]
    expected_second = _normalize_parent(raw[:, CONTEXT:], 0, CONTEXT)[0]
    assert first.shape == second.shape == (2, CONTEXT, 5)
    np.testing.assert_allclose(first, expected_first)
    np.testing.assert_allclose(second, expected_second)
    assert not np.all(first[:, CONTEXT // 2:] == 0)
    assert not np.all(second[:, CONTEXT // 2:] == 0)


def test_ttm_native_selector_and_loaded_configuration_are_exact():
    selector = get_dossier("ttm_r2")["selector"]
    config = SimpleNamespace(
        context_length=512,
        prediction_length=48,
        prediction_filter_length=16,
        resolution_prefix_tuning=True,
        enable_forecast_channel_mixing=False,
        num_input_channels=5,
    )
    model = SimpleNamespace(config=config)
    manifest = _validate_model_contract(model, "512-48-ft-r2.1", get_dossier("ttm_r2"))
    assert manifest["official_model_key"] == "512-48-ft-r2.1"
    assert manifest["enable_forecast_channel_mixing"] is False

    config.enable_forecast_channel_mixing = True
    with pytest.raises(ValueError, match="loaded configuration drift"):
        _validate_model_contract(model, selector["official_model_key"], get_dossier("ttm_r2"))
