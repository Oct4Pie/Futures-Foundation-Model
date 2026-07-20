from copy import deepcopy

import numpy as np
import pytest

from futures_foundation.finetune.routes._exact_state import (
    build_export_bundle,
    capture_state,
    restore_export_bundle,
    restore_state,
)


def _objects():
    import torch
    model = torch.nn.Linear(3, 2)
    head = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()), lr=1e-3,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4)
    return torch, model, head, optimizer, scheduler


def test_exact_state_restores_modules_optimizer_scheduler_and_rng():
    torch, model, head, optimizer, scheduler = _objects()
    torch.manual_seed(7); np.random.seed(7)
    x = torch.ones(4, 3)
    loss = head(model(x)).square().mean(); loss.backward(); optimizer.step(); scheduler.step()
    expected_model = deepcopy(model.state_dict())
    expected_head = deepcopy(head.state_dict())
    state = capture_state(
        schema_version="test_v1", route_key="arm:F:route",
        route_profile_sha256="a" * 64, model_identity={"model": "x"},
        config={"steps": 4}, modules={"model": model, "head": head},
        optimizer=optimizer, scheduler=scheduler, global_step=1, sampler_cursor=1,
        history=[{"loss": float(loss.detach())}],
    )
    random_after = torch.rand(3)
    with torch.no_grad():
        for parameter in list(model.parameters()) + list(head.parameters()):
            parameter.add_(10)
    epoch, step, cursor, history = restore_state(
        state, schema_version="test_v1", route_key="arm:F:route",
        route_profile_sha256="a" * 64, model_identity={"model": "x"},
        config={"steps": 4}, modules={"model": model, "head": head},
        optimizer=optimizer, scheduler=scheduler,
    )
    assert (epoch, step, cursor) == (0, 1, 1)
    assert history[0]["loss"] == pytest.approx(float(loss.detach()))
    for key, value in expected_model.items():
        assert torch.equal(model.state_dict()[key], value)
    for key, value in expected_head.items():
        assert torch.equal(head.state_dict()[key], value)
    assert torch.equal(torch.rand(3), random_after)
    assert state["optimizer"]["state"]
    assert all(isinstance(key, int) for key in state["optimizer"]["state"])


def test_exact_state_preserves_adam_trajectory_across_interruption():
    import torch

    torch.manual_seed(11)
    template = torch.nn.Linear(3, 1)
    initial = deepcopy(template.state_dict())
    x = torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0]])
    y = torch.tensor([[1.5], [-0.25]])

    def fresh():
        model = torch.nn.Linear(3, 1)
        model.load_state_dict(initial)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4)
        return model, optimizer, scheduler

    def step(model, optimizer, scheduler):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        optimizer.step()
        scheduler.step()
        return float(loss.detach())

    model_a, optimizer_a, scheduler_a = fresh()
    history_a = []
    for index in range(4):
        history_a.append({"step": index + 1, "loss": step(model_a, optimizer_a, scheduler_a)})
    final_a = capture_state(
        schema_version="test_v1", route_key="arm:F:route",
        route_profile_sha256="a" * 64, model_identity={"model": "x"},
        config={"steps": 4}, modules={"model": model_a},
        optimizer=optimizer_a, scheduler=scheduler_a, global_step=4, sampler_cursor=4,
        history=history_a,
    )

    model_b, optimizer_b, scheduler_b = fresh()
    history_b = []
    for index in range(2):
        history_b.append({"step": index + 1, "loss": step(model_b, optimizer_b, scheduler_b)})
    interrupted = capture_state(
        schema_version="test_v1", route_key="arm:F:route",
        route_profile_sha256="a" * 64, model_identity={"model": "x"},
        config={"steps": 4}, modules={"model": model_b},
        optimizer=optimizer_b, scheduler=scheduler_b, global_step=2, sampler_cursor=2,
        history=history_b,
    )
    model_c, optimizer_c, scheduler_c = fresh()
    _, global_step, cursor, history_c = restore_state(
        interrupted, schema_version="test_v1", route_key="arm:F:route",
        route_profile_sha256="a" * 64, model_identity={"model": "x"},
        config={"steps": 4}, modules={"model": model_c},
        optimizer=optimizer_c, scheduler=scheduler_c,
    )
    assert (global_step, cursor) == (2, 2)
    for index in range(2, 4):
        history_c.append({"step": index + 1, "loss": step(model_c, optimizer_c, scheduler_c)})
    final_c = capture_state(
        schema_version="test_v1", route_key="arm:F:route",
        route_profile_sha256="a" * 64, model_identity={"model": "x"},
        config={"steps": 4}, modules={"model": model_c},
        optimizer=optimizer_c, scheduler=scheduler_c, global_step=4, sampler_cursor=4,
        history=history_c,
    )
    assert final_c["history"] == final_a["history"]
    assert final_c["optimizer"]["state"].keys() == final_a["optimizer"]["state"].keys()
    for key in final_a["modules"]["model"]:
        assert torch.equal(final_c["modules"]["model"][key], final_a["modules"]["model"][key])
    for parameter_id, reference in final_a["optimizer"]["state"].items():
        for field, value in reference.items():
            candidate = final_c["optimizer"]["state"][parameter_id][field]
            if isinstance(value, torch.Tensor):
                assert torch.equal(candidate, value)
            else:
                assert candidate == value


def test_exact_state_rejects_module_or_profile_substitution():
    _, model, head, optimizer, scheduler = _objects()
    state = capture_state(
        schema_version="test_v1", route_key="arm:F:route",
        route_profile_sha256="a" * 64, model_identity={"model": "x"}, config={},
        modules={"model": model, "head": head}, optimizer=optimizer,
        scheduler=scheduler, global_step=0, sampler_cursor=0, history=[],
    )
    with pytest.raises(ValueError, match="module closure"):
        restore_state(
            state, schema_version="test_v1", route_key="arm:F:route",
            route_profile_sha256="a" * 64, model_identity={"model": "x"}, config={},
            modules={"model": model}, optimizer=optimizer, scheduler=scheduler,
        )
    with pytest.raises(ValueError, match="identity"):
        restore_state(
            state, schema_version="test_v1", route_key="arm:F:route",
            route_profile_sha256="b" * 64, model_identity={"model": "x"}, config={},
            modules={"model": model, "head": head}, optimizer=optimizer,
            scheduler=scheduler,
        )


def test_exact_export_reopens_only_matching_named_modules():
    torch, model, head, _, _ = _objects()
    bundle = build_export_bundle(
        schema_version="export_v1", route_key="arm:C:route",
        route_profile_sha256="c" * 64, model_identity={"model": "x"},
        modules={"model": model, "head": head},
        preprocessing={"layout": "BLC"}, output={"kind": "logits"},
        extra={"classes": 3},
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    metadata = restore_export_bundle(
        bundle, schema_version="export_v1", route_key="arm:C:route",
        route_profile_sha256="c" * 64, model_identity={"model": "x"},
        modules={"model": model, "head": head},
    )
    assert metadata["output"] == {"kind": "logits"}
    forged = deepcopy(bundle); forged["modules"].pop("head")
    with pytest.raises(ValueError, match="identity"):
        restore_export_bundle(
            forged, schema_version="export_v1", route_key="arm:C:route",
            route_profile_sha256="c" * 64, model_identity={"model": "x"},
            modules={"model": model, "head": head},
        )
