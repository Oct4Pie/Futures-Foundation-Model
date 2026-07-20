from types import SimpleNamespace

import numpy as np
import pytest

from futures_foundation.finetune.routes import ttm_native as route


class DummyTTM:
    def __init__(self, torch):
        self.backbone = torch.nn.Module()
        self.backbone.encoder = torch.nn.Module()
        self.backbone.encoder.core = torch.nn.Linear(route.CONTEXT_LENGTH, route.HORIZON_LENGTH)
        self.backbone.encoder.freq_mod = torch.nn.Sequential(torch.nn.Embedding(10, 4), torch.nn.Linear(4, 4))
        self.decoder = torch.nn.Linear(len(route.CHANNELS), len(route.CHANNELS))
        self.head = torch.nn.Linear(len(route.CHANNELS), len(route.CHANNELS))

    def parameters(self):
        for module in (self.backbone, self.decoder, self.head): yield from module.parameters()

    def named_parameters(self):
        for prefix, module in (("backbone", self.backbone), ("decoder", self.decoder), ("head", self.head)):
            for name, parameter in module.named_parameters(): yield f"{prefix}.{name}", parameter

    def state_dict(self):
        result = {}
        for name, parameter in self.named_parameters(): result[name] = parameter.detach().clone()
        return result

    def load_state_dict(self, state, strict=True):
        own = dict(self.named_parameters())
        if strict and set(own) != set(state): raise RuntimeError("state mismatch")
        for name, value in state.items(): own[name].data.copy_(value)

    def train(self):
        self.backbone.train(); self.decoder.train(); self.head.train(); return self

    def eval(self):
        self.backbone.eval(); self.decoder.eval(); self.head.eval(); return self

    def __call__(self, *, past_values, freq_token, future_values=None, return_loss=True, **kwargs):
        import torch
        base = self.backbone.encoder.core(past_values.transpose(1, 2)).transpose(1, 2)
        forecast = self.head(self.decoder(base))
        if future_values is not None:
            loss = torch.nn.functional.mse_loss(forecast, future_values)
        else:
            loss = None
        return SimpleNamespace(prediction_outputs=forecast, loss=loss)


def _loaded(key):
    import torch
    return route.LoadedRoute(key, DummyTTM(torch), {"fixture": key}, "cpu")


def _parent(batch=3):
    return np.random.default_rng(0).normal(size=(batch, 528, 5)).astype("float32")


def test_ttm_inventory_defaults_and_frequency_tokens():
    assert len(route.ROUTES) == 2
    for key in route.ROUTES:
        config = route.RouteConfig(key).resolved()
        assert config["learning_rate"] == 1e-4
        assert config["batch_size"] == 8
    assert route.FREQUENCY_TOKENS == {
        "1min": 1, "3min": 0, "5min": 3, "15min": 5, "30min": 6, "60min": 7,
    }
    with pytest.raises(ValueError, match="timeframe"):
        route.frequency_token("2min", batch=1, device="cpu")


def test_ttm_native_loss_geometry_and_missing_masks():
    import torch
    loaded = _loaded("ttm_r2:F:full_model_raw_hf_trainer_forecast")
    parent = _parent(); parent[0, 10, 2] = np.nan; parent[0, 520, 1] = np.nan
    loss = route.native_loss(loaded, parent, timeframe="1min")
    assert torch.isfinite(loss)
    context, context_mask, target, target_mask = route.split_parent(parent)
    assert context_mask[0, 10, 2] == 0 and context[0, 10, 2] == 0
    assert target_mask[0, 8, 1] == 0 and target[0, 8, 1] == 0


def test_ttm_head_prefix_surface_freezes_only_declared_modules():
    loaded = _loaded("ttm_r2:F:head_prefix_raw_hf_trainer_forecast")
    for name, parameter in loaded.model.named_parameters():
        parameter.requires_grad = (
            name.startswith("decoder.") or name.startswith("head.")
            or name.startswith("backbone.encoder.freq_mod.")
        )
    trainable = {name for name, parameter in loaded.model.named_parameters() if parameter.requires_grad}
    assert trainable
    assert all(
        name.startswith("decoder.") or name.startswith("head.")
        or name.startswith("backbone.encoder.freq_mod.")
        for name in trainable
    )
    assert not any(name.startswith("backbone.encoder.core.") for name in trainable)


def test_ttm_step_resume_and_export_roundtrip():
    key = "ttm_r2:F:full_model_raw_hf_trainer_forecast"
    loaded = _loaded(key); config = route.RouteConfig(key)
    optimizer = route.make_optimizer(loaded, config); scheduler = route.make_scheduler(optimizer, config)
    metrics = route.optimizer_step(
        loaded, optimizer, scheduler, _parent(), timeframe="5min", config=config,
    )
    assert metrics["parameter_delta"] > 0
    state = route.capture_training_state(
        loaded, optimizer, scheduler, config, global_step=1, sampler_cursor=1,
        history=[metrics],
    )
    assert route.restore_training_state(state, loaded, optimizer, scheduler, config)[:3] == (0, 1, 1)
    bundle = route.build_export_bundle(loaded)
    metadata = route.restore_export_bundle(bundle, loaded)
    assert metadata["output"]["deployment_filter"] == 16
    assert metadata["preprocessing"]["frequency_tokens"]["3min"] == 0
