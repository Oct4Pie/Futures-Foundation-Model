from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from futures_foundation.finetune.routes import kronos_predictor as route


def _parent(batch: int = 2) -> np.ndarray:
    time = np.arange(route.PARENT_LENGTH, dtype=np.float32)
    close = 50.0 + 0.02 * time[None, :] + np.arange(batch, dtype=np.float32)[:, None]
    open_ = np.concatenate((close[:, :1], close[:, :-1]), axis=1)
    high = np.maximum(open_, close) + 0.25
    low = np.minimum(open_, close) - 0.25
    volume = np.broadcast_to(1000.0 + 0.5 * time[None, :], close.shape)
    return np.stack((open_, high, low, close, volume), axis=-1).astype(np.float32)


class _FakeTokenizer:
    def __init__(self):
        import torch

        self.training = False
        self._parameter = torch.nn.Parameter(torch.tensor(1.0), requires_grad=False)

    def eval(self):
        self.training = False
        return self

    def parameters(self):
        return iter((self._parameter,))

    def encode(self, values, half=False):
        import torch

        assert half is True
        batch = values.shape[0]
        base = torch.arange(route.PARENT_LENGTH, device=values.device)
        coarse = base[None, :].expand(batch, -1) % 4
        fine = (base[None, :].expand(batch, -1) + 1) % 5
        return coarse.long(), fine.long()


class _FakeHead:
    def __init__(self):
        self.targets = None

    def compute_loss(self, first_logits, second_logits, first_target, second_target):
        import torch.nn.functional as functional

        self.targets = (first_target.detach().clone(), second_target.detach().clone())
        first = functional.cross_entropy(
            first_logits.reshape(-1, first_logits.shape[-1]),
            first_target.reshape(-1),
        )
        second = functional.cross_entropy(
            second_logits.reshape(-1, second_logits.shape[-1]),
            second_target.reshape(-1),
        )
        loss = (first + second) / 2
        return loss, first, second


class _FakePredictor:
    def __init__(self):
        import torch

        self.weight = torch.nn.Parameter(torch.tensor(0.1))
        self.head = _FakeHead()
        self.training = False
        self.calls = []

    def parameters(self):
        return iter((self.weight,))

    def train(self):
        self.training = True
        return self

    def eval(self):
        self.training = False
        return self

    def state_dict(self):
        return {"weight": self.weight.detach().clone()}

    def load_state_dict(self, state, strict=True):
        self.weight.data.copy_(state["weight"])
        return None

    def __call__(self, first, second, stamps):
        import torch

        self.calls.append((first.detach().clone(), second.detach().clone(), stamps.detach().clone()))
        first_basis = torch.arange(4, device=first.device, dtype=torch.float32)
        second_basis = torch.arange(5, device=first.device, dtype=torch.float32)
        first_logits = -(
            first_basis[None, None, :] - first[..., None].float()
        ).square() * (1.0 + self.weight)
        second_logits = -(
            second_basis[None, None, :] - second[..., None].float()
        ).square() * (1.0 + self.weight)
        return first_logits, second_logits


def _loaded(arm_key: str = route.ARM_KEY) -> route.LoadedRoute:
    return route.LoadedRoute(
        tokenizer=_FakeTokenizer(),
        predictor=_FakePredictor(),
        public_predictor_class=object,
        identity={"fixture": True, "arm_key": arm_key},
        device="cpu",
        arm_key=arm_key,
    )


def test_predictor_normalization_uses_only_first_512_context_bars():
    parent = _parent(1)
    changed = parent.copy()
    changed[:, route.CONTEXT_LENGTH:, 3] += 1000.0
    changed[:, route.CONTEXT_LENGTH:, 1] += 1000.0

    normalized, mean, std = route.normalize_parent(parent)
    changed_normalized, changed_mean, changed_std = route.normalize_parent(changed)
    np.testing.assert_array_equal(mean, changed_mean)
    np.testing.assert_array_equal(std, changed_std)
    np.testing.assert_array_equal(
        normalized[:, :route.CONTEXT_LENGTH],
        changed_normalized[:, :route.CONTEXT_LENGTH],
    )
    assert not np.array_equal(
        normalized[:, route.CONTEXT_LENGTH:],
        changed_normalized[:, route.CONTEXT_LENGTH:],
    )


def test_predictor_calendar_stamps_use_cme_local_time_across_dst():
    prefix = pd.date_range(
        "2024-03-09 23:12:00+00:00",
        periods=route.PARENT_LENGTH - 2,
        freq="1min",
    )
    tail = pd.to_datetime(
        ["2024-03-10 07:59:00+00:00", "2024-03-10 08:00:00+00:00"],
        utc=True,
    )
    timestamps = np.asarray(prefix.append(tail))[None, :]
    stamps = route.calendar_stamps(timestamps)
    assert stamps[0, -2:, 0].tolist() == [59.0, 0.0]
    assert stamps[0, -2:, 1].tolist() == [1.0, 3.0]
    assert stamps[0, -2:, 2].tolist() == [6.0, 6.0]


def test_predictor_native_loss_targets_every_next_token_position():
    loaded = _loaded()
    stamps = np.zeros((2, route.PARENT_LENGTH, 5), dtype=np.float32)
    stamps[:, :, 3] = 1
    stamps[:, :, 4] = 1
    loss = route.native_loss(loaded, _parent(), stamps)
    assert np.isfinite(float(loss.detach()))
    first_input, second_input, observed_stamps = loaded.predictor.calls[-1]
    assert tuple(first_input.shape) == (2, route.PARENT_LENGTH - 1)
    assert tuple(second_input.shape) == (2, route.PARENT_LENGTH - 1)
    assert tuple(observed_stamps.shape) == (2, route.PARENT_LENGTH - 1, 5)
    first_target, second_target = loaded.predictor.head.targets
    assert tuple(first_target.shape) == (2, route.PARENT_LENGTH - 1)
    assert tuple(second_target.shape) == (2, route.PARENT_LENGTH - 1)
    assert first_target[0, 0].item() == 1
    assert first_target[0, -1].item() == (route.PARENT_LENGTH - 1) % 4
    assert second_target[0, 0].item() == 2


def test_predictor_optimizer_updates_only_predictor_surface():
    loaded = _loaded()
    config = route.RouteConfig(total_steps=20, batch_size=2)
    optimizer = route.make_optimizer(loaded.predictor, config)
    scheduler = route.make_scheduler(optimizer, config)
    stamps = np.zeros((2, route.PARENT_LENGTH, 5), dtype=np.float32)
    stamps[:, :, 3] = 1
    stamps[:, :, 4] = 1
    before_predictor = loaded.predictor.weight.detach().clone()
    before_tokenizer = next(loaded.tokenizer.parameters()).detach().clone()
    row = route.optimizer_step(
        loaded,
        optimizer,
        scheduler,
        _parent(),
        stamps,
        max_gradient_norm=config.max_gradient_norm,
    )
    assert row["grad_norm"] > 0
    assert row["parameter_delta"] > 0
    assert not np.array_equal(
        loaded.predictor.weight.detach().numpy(), before_predictor.numpy()
    )
    np.testing.assert_array_equal(
        next(loaded.tokenizer.parameters()).detach().numpy(),
        before_tokenizer.numpy(),
    )


def test_predictor_rejects_invalid_stamp_domains():
    stamps = np.zeros((1, route.PARENT_LENGTH, 5), dtype=np.float32)
    stamps[:, :, 3] = 1
    stamps[:, :, 4] = 1
    stamps[0, 0, 1] = 24
    with pytest.raises(ValueError, match="hour"):
        route.stamps_array(stamps)


def test_predictor_arm_identity_prevents_mini_small_state_substitution():
    small = _loaded("kronos_small")
    config = route.RouteConfig(arm_key="kronos_small", total_steps=20, batch_size=2)
    optimizer = route.make_optimizer(small.predictor, config)
    scheduler = route.make_scheduler(optimizer, config)
    state = route.capture_training_state(
        loaded=small, optimizer=optimizer, scheduler=scheduler, config=config,
        global_step=0, sampler_cursor=0, history=[],
    )
    assert state["route_key"] == route.route_key("kronos_small")
    assert state["route_profile_sha256"] != route.canonical_route_profile_sha256(
        "kronos_mini", route.TRACK, route.ROUTE_ID,
    )
    mini = _loaded("kronos_mini")
    mini_config = route.RouteConfig(total_steps=20, batch_size=2)
    mini_optimizer = route.make_optimizer(mini.predictor, mini_config)
    mini_scheduler = route.make_scheduler(mini_optimizer, mini_config)
    with pytest.raises(ValueError, match="identity"):
        route.restore_training_state(
            state, loaded=mini, optimizer=mini_optimizer,
            scheduler=mini_scheduler, config=mini_config,
        )
    with pytest.raises(ValueError, match="config arm"):
        route.capture_training_state(
            loaded=small, optimizer=optimizer, scheduler=scheduler,
            config=mini_config, global_step=0, sampler_cursor=0, history=[],
        )
