from types import SimpleNamespace

import numpy as np
import pytest

from futures_foundation.finetune.routes import chronos2_native as route


class DummyModel:
    def __init__(self, torch):
        self.layer = torch.nn.Linear(route.CONTEXT_LENGTH, route.NATIVE_HORIZON)

    def parameters(self): return self.layer.parameters()
    def named_parameters(self): return self.layer.named_parameters()
    def state_dict(self): return self.layer.state_dict()
    def load_state_dict(self, state, strict=True): return self.layer.load_state_dict(state, strict=strict)
    def train(self): self.layer.train(); return self
    def eval(self): self.layer.eval(); return self

    def __call__(self, *, context, future_target=None, **kwargs):
        import torch
        point = self.layer(torch.nan_to_num(context.float()))
        quantiles = point[:, None, :].repeat(1, len(route.QUANTILES), 1)
        loss = None
        if future_target is not None:
            valid = torch.isfinite(future_target)
            target = torch.nan_to_num(future_target)
            loss = ((point[:, :route.HORIZON_LENGTH] - target).square() * valid).sum() / valid.sum()
        return SimpleNamespace(quantile_preds=quantiles, loss=loss)


class DummyPipeline:
    def __init__(self, model): self.model = model


def _loaded(key):
    import torch
    model = DummyModel(torch)
    return route.LoadedRoute(key, DummyPipeline(model), model, model, {"fixture": key}, "cpu")


def _parent(batch=2):
    return np.random.default_rng(0).normal(size=(batch, 528, 5)).astype("float32")


def test_chronos2_inventory_and_route_defaults():
    assert len(route.ROUTES) == 2
    full = route.RouteConfig("chronos_v2:F:official_fit_full", batch_size=2).resolved()
    lora = route.RouteConfig("chronos_v2:F:official_fit_lora", batch_size=2).resolved()
    assert full["learning_rate"] == 1e-6
    assert lora["learning_rate"] == 1e-5
    assert (route.LORA_R, route.LORA_ALPHA) == (8, 16)
    assert route.LORA_TARGET == ("q", "k", "v", "o")
    assert route.LORA_MODULES_TO_SAVE == (
        "input_patch_embedding", "output_patch_embedding",
    )
    with pytest.raises(ValueError, match="batch_size"):
        route.RouteConfig("chronos_v2:F:official_fit_full", batch_size=1).resolved()


def test_chronos2_grouped_parent_and_native_quantiles():
    import torch
    loaded = _loaded("chronos_v2:F:official_fit_full")
    context, future, covariates, groups = route.split_parent(_parent(3))
    assert context.shape == (15, 512) and future.shape == covariates.shape == (15, 16)
    assert groups.tolist() == [0] * 5 + [1] * 5 + [2] * 5
    loss = route.native_loss(loaded, _parent(3))
    assert torch.isfinite(loss)
    quantiles = route.native_quantiles(loaded, _parent(3)[:, :512])
    assert quantiles.shape == (3, 5, 13, 16)
    missing = _parent(1); missing[0, 0, 0] = np.nan
    assert torch.isfinite(route.native_loss(loaded, missing))


def test_chronos2_full_step_resume_and_export():
    key = "chronos_v2:F:official_fit_full"
    loaded = _loaded(key); config = route.RouteConfig(key, batch_size=2)
    optimizer = route.make_optimizer(loaded, config); scheduler = route.make_scheduler(optimizer, config)
    assert optimizer.param_groups[0]["fused"] is False
    metrics = route.optimizer_step(loaded, optimizer, scheduler, _parent(), config=config)
    assert metrics["parameter_delta"] > 0
    state = route.capture_training_state(
        loaded, optimizer, scheduler, config,
        global_step=1, sampler_cursor=1, history=[metrics],
    )
    assert route.restore_training_state(state, loaded, optimizer, scheduler, config)[:3] == (0, 1, 1)
    bundle = route.build_export_bundle(loaded)
    metadata = route.restore_export_bundle(bundle, loaded)
    assert metadata["output"]["shape"] == ["batch", 5, 13, 16]
    assert metadata["preprocessing"]["group_ids"] == "same_parent_five_variates"


def test_chronos2_config_rejects_cross_route_substitution():
    loaded = _loaded("chronos_v2:F:official_fit_full")
    config = route.RouteConfig("chronos_v2:F:official_fit_lora", batch_size=2)
    with pytest.raises(ValueError, match="config route"):
        route.make_optimizer(loaded, config)
