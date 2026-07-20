from types import SimpleNamespace

import numpy as np
import pytest

from futures_foundation.finetune.routes import timesfm_lora as route


class DummyModel:
    def __init__(self, torch):
        self.adapter = torch.nn.Linear(route.CONTEXT_LENGTH, route.HORIZON_LENGTH, bias=False)

    def parameters(self): return self.adapter.parameters()
    def named_parameters(self):
        for name, parameter in self.adapter.named_parameters():
            yield f"base_model.model.adapter.lora_A.{name}", parameter
    def train(self): self.adapter.train(); return self
    def eval(self): self.adapter.eval(); return self
    def __call__(self, *, past_values, future_values=None, **kwargs):
        import torch
        point = self.adapter(past_values.float())
        full = point[..., None].repeat(1, 1, 3)
        loss = None if future_values is None else torch.nn.functional.mse_loss(point, future_values.float())
        return SimpleNamespace(mean_predictions=point, full_predictions=full, loss=loss)


class DummyAdapterState:
    def __init__(self, model): self.model = model
    def state_dict(self): return self.model.adapter.state_dict()
    def load_state_dict(self, state, strict=True): return self.model.adapter.load_state_dict(state, strict=strict)


def _loaded():
    import torch
    model = DummyModel(torch)
    return route.LoadedRoute(model, DummyAdapterState(model), {"fixture": True}, "cpu")


def _parent(batch=2):
    return np.random.default_rng(0).normal(size=(batch, 528, 5)).astype("float32")


def test_timesfm_lora_config_is_exact_and_bounded():
    config = route.RouteConfig(); config.validate()
    assert (route.LORA_R, route.LORA_ALPHA, route.LORA_DROPOUT) == (4, 8, 0.05)
    assert config.batch_size == 32 and config.learning_rate == 1e-4
    with pytest.raises(ValueError, match="batch_size"):
        route.RouteConfig(batch_size=33).validate()


def test_timesfm_parent_flattening_and_native_loss():
    import torch
    context, target = route.split_parent(_parent(3))
    assert context.shape == (15, 512) and target.shape == (15, 16)
    loaded = _loaded(); loss = route.native_loss(loaded, _parent(3))
    assert torch.isfinite(loss)
    point, quantiles = route.public_output(loaded, _parent(3)[:, :512])
    assert point.shape == (15, 16) and quantiles.shape == (15, 16, 3)
    missing_context = _parent(1); missing_context[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="no stable missing-value path"):
        route.split_parent(missing_context)
    invalid_target = _parent(1); invalid_target[0, route.CONTEXT_LENGTH, 0] = np.nan
    with pytest.raises(ValueError, match="no stable missing-value path"):
        route.split_parent(invalid_target)
    infinite = _parent(1); infinite[0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="no stable missing-value path"):
        route.split_parent(infinite)


def test_timesfm_adapter_only_state_resume_and_export():
    loaded = _loaded(); config = route.RouteConfig()
    optimizer = route.make_optimizer(loaded, config); scheduler = route.make_scheduler(optimizer, config)
    metrics = route.optimizer_step(loaded, optimizer, scheduler, _parent(), config=config)
    assert metrics["parameter_delta"] > 0
    state = route.capture_training_state(
        loaded, optimizer, scheduler, config, global_step=1, sampler_cursor=1,
        history=[metrics],
    )
    assert set(state["modules"]) == {"adapter"}
    assert route.restore_training_state(state, loaded, optimizer, scheduler, config)[:3] == (0, 1, 1)
    bundle = route.build_export_bundle(loaded)
    assert set(bundle["modules"]) == {"adapter"}
    metadata = route.restore_export_bundle(bundle, loaded)
    assert metadata["extra"] == {"adapter_only": True, "base_model_required": True}


def test_timesfm_optimizer_scheduler_match_official_recipe():
    loaded = _loaded(); config = route.RouteConfig()
    optimizer = route.make_optimizer(loaded, config); scheduler = route.make_scheduler(optimizer, config)
    group = optimizer.param_groups[0]
    assert type(optimizer).__name__ == "AdamW"
    assert group["lr"] == 1e-4 and group["weight_decay"] == 0.01
    assert type(scheduler).__name__ == "CosineAnnealingLR"
    assert scheduler.T_max == 20
