from types import SimpleNamespace

import numpy as np
import pytest

from futures_foundation.finetune.routes import moment_tasks as route


class DummyMoment:
    def __init__(self, torch, task: str, classes: int = 3):
        self.task = task
        self.encoder = torch.nn.Linear(1, 1)
        self.patch_embedding = torch.nn.Linear(1, 1)
        self.head = (
            torch.nn.Linear(len(route.CHANNELS), classes)
            if task == "classification" else torch.nn.Linear(route.CONTEXT_LENGTH, route.HORIZON_LENGTH)
        )

    def parameters(self):
        for module in (self.encoder, self.patch_embedding, self.head):
            yield from module.parameters()

    def state_dict(self):
        result = {}
        for prefix, module in (("encoder", self.encoder), ("patch_embedding", self.patch_embedding), ("head", self.head)):
            for key, value in module.state_dict().items():
                result[f"{prefix}.{key}"] = value
        return result

    def load_state_dict(self, state, strict=True):
        for prefix, module in (("encoder", self.encoder), ("patch_embedding", self.patch_embedding), ("head", self.head)):
            module.load_state_dict({key.split(".", 1)[1]: value for key, value in state.items() if key.startswith(prefix + ".")})

    def train(self):
        for module in (self.encoder, self.patch_embedding, self.head): module.train()
        return self

    def eval(self):
        for module in (self.encoder, self.patch_embedding, self.head): module.eval()
        return self

    def __call__(self, *, x_enc, input_mask):
        import torch
        masked = x_enc * input_mask[:, None, :]
        if self.task == "classification":
            features = masked.mean(dim=2)
            return SimpleNamespace(logits=self.head(features))
        forecast = self.head(masked)
        return SimpleNamespace(forecast=forecast)


def _loaded(key):
    import torch
    spec = route.ROUTES[key]
    model = DummyMoment(torch, spec["task"])
    if spec["surface"] == "head":
        for module in (model.encoder, model.patch_embedding):
            for parameter in module.parameters(): parameter.requires_grad = False
    return route.LoadedRoute(key, model, {"fixture": True, "route": key}, "cpu")


def _context(batch=3):
    return np.random.default_rng(0).normal(size=(batch, 512, 5)).astype("float32")


def _parent(batch=3):
    return np.random.default_rng(1).normal(size=(batch, 528, 5)).astype("float32")


def test_moment_task_inventory_and_defaults_are_resolved():
    assert len(route.ROUTES) == 4
    for key, spec in route.ROUTES.items():
        config = route.RouteConfig(key).resolved()
        assert config["total_steps"] == 20
        assert config["learning_rate"] == 1e-4
        assert config["batch_size"] in {8, 64}
        assert spec["surface"] in {"full", "head"}


def test_moment_classification_and_forecast_native_geometry():
    import torch
    classifier = _loaded("moment_small:C:classification_full")
    logits = route.classification_logits(classifier, _context())
    assert logits.shape == (3, 3)
    loss = route.native_loss(classifier, _context(), labels=np.arange(3))
    assert torch.isfinite(loss)
    forecast_route = _loaded("moment_small:F:forecast_full_raw_mse")
    context, target = route.split_forecast_parent(_parent())
    forecast = route.forecast_values(forecast_route, context)
    assert forecast.shape == (3, 5, 16)
    assert target.shape == (3, 16, 5)
    assert torch.isfinite(route.native_loss(forecast_route, _parent()))


def test_moment_mask_rejects_partial_missing_and_accepts_complete_missing_bar():
    values = _context(1)
    values[:, 10, :] = np.nan
    x, mask = route.model_input(values, device="cpu")
    assert mask[0, 10].item() == 0
    assert x[:, :, 10].abs().sum().item() == 0
    values = _context(1); values[:, 10, 0] = np.nan
    with pytest.raises(ValueError, match="partial"):
        route.model_input(values, device="cpu")


def test_moment_head_only_updates_head_and_exact_state_roundtrips():
    import torch
    key = "moment_small:C:classification_head_only"
    loaded = _loaded(key); config = route.RouteConfig(key)
    optimizer = route.make_optimizer(loaded, config); scheduler = route.make_scheduler(optimizer, config)
    before_encoder = {k: v.clone() for k, v in loaded.model.encoder.state_dict().items()}
    metrics = route.optimizer_step(
        loaded, optimizer, scheduler, _context(4), labels=np.arange(4) % 3, config=config,
    )
    assert metrics["parameter_delta"] > 0
    for key2, value in before_encoder.items():
        assert torch.equal(loaded.model.encoder.state_dict()[key2], value)
    state = route.capture_training_state(
        loaded, optimizer, scheduler, config, global_step=1, sampler_cursor=1,
        history=[metrics],
    )
    assert route.restore_training_state(state, loaded, optimizer, scheduler, config)[:3] == (0, 1, 1)
    bundle = route.build_export_bundle(loaded, config)
    metadata = route.restore_export_bundle(bundle, loaded)
    assert metadata["output"] == {"kind": "classification_logits", "classes": 3}


def test_moment_forecast_uses_onecycle_pct_start_point_three():
    key = "moment_small:F:forecast_head_only_raw_mse"
    loaded = _loaded(key); config = route.RouteConfig(key)
    optimizer = route.make_optimizer(loaded, config); scheduler = route.make_scheduler(optimizer, config)
    assert type(scheduler).__name__ == "OneCycleLR"
    assert scheduler.state_dict()["total_steps"] == 20
    assert scheduler.state_dict()["_schedule_phases"][0]["end_step"] == pytest.approx(5.0)
