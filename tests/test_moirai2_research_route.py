import numpy as np
import pytest

from futures_foundation.finetune.routes import moirai2_research as route


class DummyForecast:
    context_length = route.CONTEXT_LENGTH

    def context_token_length(self, patch_size): return route.CONTEXT_LENGTH // patch_size
    def prediction_token_length(self, patch_size): return route.HORIZON_LENGTH // patch_size

    def _convert(
        self, patch_size, past_target, past_observed_target, past_is_pad,
        *, future_target, future_observed_target,
    ):
        import torch
        b = past_target.shape[0]; c = len(route.CHANNELS); context_tokens = 32
        past = past_target.permute(0, 2, 1).reshape(b, c * context_tokens, patch_size)
        future = future_target.permute(0, 2, 1).reshape(b, c, patch_size)
        target = torch.cat([past, future], dim=1)
        past_mask = past_observed_target.permute(0, 2, 1).reshape(b, c * context_tokens, patch_size)
        future_mask = future_observed_target.permute(0, 2, 1).reshape(b, c, patch_size)
        observed = torch.cat([past_mask, future_mask], dim=1)
        sample = torch.ones(b, target.shape[1], dtype=torch.long, device=target.device)
        time = torch.arange(target.shape[1], device=target.device).repeat(b, 1)
        variate = torch.cat([
            torch.arange(c, device=target.device).repeat_interleave(context_tokens),
            torch.arange(c, device=target.device),
        ]).repeat(b, 1)
        prediction = torch.cat([
            torch.zeros(b, c * context_tokens, dtype=torch.bool, device=target.device),
            torch.ones(b, c, dtype=torch.bool, device=target.device),
        ], dim=1)
        return target, observed, sample, time, variate, prediction


class DummyModule:
    def __init__(self, torch):
        self.patch_size = route.PATCH_SIZE
        self.num_predict_token = 4
        self.num_quantiles = len(route.QUANTILES)
        self.quantile_levels = route.QUANTILES
        self.weight = torch.nn.Parameter(torch.tensor(0.0))

    def parameters(self): yield self.weight
    def state_dict(self): return {"weight": self.weight.detach().clone()}
    def load_state_dict(self, state, strict=True): self.weight.data.copy_(state["weight"])
    def train(self): return self
    def eval(self): return self

    def __call__(self, target, observed_mask, sample_id, time_id, variate_id, prediction_mask, training_mode=True):
        import torch
        b, seq, patch = target.shape
        raw = torch.zeros(
            b, seq, self.num_predict_token * self.num_quantiles * patch,
            dtype=target.dtype, device=target.device,
        )
        raw = raw + self.weight
        return raw, target


def _loaded():
    import torch
    module = DummyModule(torch)
    return route.LoadedRoute(
        forecast=DummyForecast(), module=module,
        identity={"fixture": True, "license_scope": "research_noncommercial"},
        device="cpu",
    )


def _parent(batch=2):
    return np.random.default_rng(0).normal(size=(batch, 528, 5)).astype("float32")


def test_moirai_config_and_research_scope_are_fixed():
    config = route.RouteConfig(); config.validate()
    assert config.learning_rate == 5e-7 and config.weight_decay == 0.1
    assert route.PATCH_SIZE == route.HORIZON_LENGTH == 16
    assert route.QUANTILES == tuple(value / 10 for value in range(1, 10))
    with pytest.raises(ValueError, match="batch_size"):
        route.RouteConfig(batch_size=513).validate()


def test_moirai_native_patch_geometry_and_pinball_mask():
    import torch
    loaded = _loaded(); parent = _parent()
    predictions, target, observed = route.native_predictions_and_target(loaded, parent)
    assert predictions.shape == (2, 5, 9, 16)
    assert target.shape == observed.shape == (2, 5, 16)
    loss = route.native_loss(loaded, parent)
    assert torch.isfinite(loss) and float(loss.detach()) > 0
    parent[:, 512:, :] = np.nan
    with pytest.raises(ValueError, match="no observed"):
        route.native_loss(loaded, parent)


def test_moirai_quantile_direction_is_native_scaled_pinball():
    loaded = _loaded(); parent = np.zeros((1, 528, 5), dtype=np.float32)
    parent[:, 512:, :] = 1.0
    loaded.module.weight.data.fill_(0.0)
    below = float(route.native_loss(loaded, parent).detach())
    loaded.module.weight.data.fill_(1.0)
    exact = float(route.native_loss(loaded, parent).detach())
    assert exact == pytest.approx(0.0, abs=1e-7)
    assert below > exact


def test_moirai_step_resume_and_research_export():
    loaded = _loaded(); config = route.RouteConfig()
    optimizer = route.make_optimizer(loaded, config); scheduler = route.make_scheduler(optimizer, config)
    group = optimizer.param_groups[0]
    assert group["betas"] == (0.9, 0.98) and group["eps"] == 1e-6
    metrics = route.optimizer_step(loaded, optimizer, scheduler, _parent(), config=config)
    assert metrics["parameter_delta"] > 0
    state = route.capture_training_state(
        loaded, optimizer, scheduler, config,
        global_step=1, sampler_cursor=1, history=[metrics],
    )
    assert route.restore_training_state(state, loaded, optimizer, scheduler, config)[:3] == (0, 1, 1)
    bundle = route.build_export_bundle(loaded)
    metadata = route.restore_export_bundle(bundle, loaded)
    assert metadata["extra"] == {
        "license_scope": "research_noncommercial", "production_admitted": False,
    }
    assert metadata["output"]["crossing_repair"] is False
