import numpy as np
import pytest

from futures_foundation.finetune.routes import mantis_native as route


class DummyBackbone:
    def __init__(self, torch, *, output_dim=4):
        self.module = torch.nn.Linear(route.CONTEXT_LENGTH, output_dim)
        self.hidden_dim = output_dim
        self.pre_training = True
        self.return_transf_layer = -1
        self.output_token = "cls_token"

    def __call__(self, x):
        return self.module(x[:, 0])

    def __getattr__(self, name):
        if name == "module":
            raise AttributeError(name)
        return getattr(self.module, name)

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, *args, **kwargs):
        return self.module.load_state_dict(*args, **kwargs)

    def parameters(self):
        return self.module.parameters()

    def train(self):
        return self.module.train()

    def eval(self):
        return self.module.eval()


def _identity(key):
    return {
        "arm_key": route.ROUTES[key]["arm"], "model_id": "dummy",
        "model_revision": "a" * 40, "source_revision": "b" * 40,
        "source_runtime": "/tmp", "route_profile_sha256": "c" * 64,
    }


def test_mantis_route_inventory_and_configs_are_exactly_closed():
    assert len(route.ROUTES) == 6
    for key, spec in route.ROUTES.items():
        cfg = route.RouteConfig(key).resolved()
        assert cfg["total_steps"] == 20
        assert cfg["batch_size"] in {256, 512}
        assert cfg["learning_rate"] in {2e-4, 2e-3}
        assert spec["task"] in {"classification", "contrastive"}
    with pytest.raises(ValueError, match="unknown"):
        route.RouteConfig("mantis:bad:route").resolved()


def test_classification_geometry_and_head_only_freeze_surface():
    import torch
    key = "mantis_v1:C:supervised_classification_head"
    backbone = DummyBackbone(torch)
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    loaded = route.LoadedRoute(
        route_key=key, backbone=backbone,
        head=torch.nn.Linear(4 * len(route.CHANNELS), 3), identity=_identity(key),
    )
    parent = np.random.default_rng(0).normal(size=(5, 512, 5)).astype("float32")
    logits = route.classification_logits(loaded, parent, device="cpu")
    assert logits.shape == (5, 3)
    loss = route.native_loss(loaded, parent, labels=np.arange(5) % 3, device="cpu")
    loss.backward()
    assert all(parameter.grad is None for parameter in backbone.parameters())
    assert any(parameter.grad is not None for parameter in loaded.head.parameters())


def test_contrastive_loss_excludes_channel_siblings(monkeypatch):
    import torch
    key = "mantis_v2:R:official_crop_resize_contrastive"
    loaded = route.LoadedRoute(
        route_key=key, backbone=DummyBackbone(torch), head=None, identity=_identity(key),
    )
    monkeypatch.setattr(route, "_augment_views", lambda value, **_: (value, value.clone()))
    parent = np.random.default_rng(1).normal(size=(4, 512, 5)).astype("float32")
    loss = route.contrastive_loss(loaded, parent, device="cpu")
    assert loss.ndim == 0 and torch.isfinite(loss)
    # The target matrix is B×B for each channel, not (B*5)×(B*5).
    assert float(loss.detach()) < np.log(4) + 1e-5


def test_mantis_exact_resume_and_export_roundtrip(monkeypatch):
    import torch
    key = "mantis_v1:C:supervised_classification_full"
    loaded = route.LoadedRoute(
        route_key=key, backbone=DummyBackbone(torch),
        head=torch.nn.Linear(4 * len(route.CHANNELS), 3), identity=_identity(key),
    )
    config = route.RouteConfig(key)
    optimizer = route.make_optimizer(loaded, config)
    scheduler = route.make_scheduler(optimizer, config)
    parent = np.random.default_rng(2).normal(size=(4, 512, 5)).astype("float32")
    labels = np.arange(4) % 3
    metrics = route.optimizer_step(
        loaded, optimizer, scheduler, parent, labels=labels, device="cpu", config=config,
    )
    state = route.capture_training_state(
        loaded, optimizer, scheduler, config, global_step=1, sampler_cursor=1,
        history=[metrics],
    )
    epoch, step, cursor, history = route.restore_training_state(
        state, loaded, optimizer, scheduler, config,
    )
    assert (epoch, step, cursor) == (0, 1, 1)
    assert history[0]["loss"] == pytest.approx(metrics["loss"])
    bundle = route.build_export_bundle(loaded, config)
    metadata = route.restore_export_bundle(bundle, loaded)
    assert metadata["output"] == {"kind": "classification_logits", "classes": 3}
