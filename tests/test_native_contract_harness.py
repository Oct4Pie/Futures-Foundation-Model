import numpy as np

from futures_foundation.finetune.native_contract_harness import (
    batch_partition_check,
    channel_independence_check,
    context_boundary_check,
    control_rejection_check,
    finite_check,
    forward_backward_check,
    future_corruption_check,
    interruption_resume_parity_check,
    loss_decrease_check,
    multivariate_layout_check,
    negative_price_behavior_check,
    parity_check,
    performance_check,
    prefix_invariance_check,
    rejection_check,
    scaling_roundtrip_check,
)


def test_finite_and_parity_checks_report_metrics():
    assert finite_check(np.ones((2, 3))).status == "pass"
    failed = finite_check(np.asarray([1.0, np.nan]))
    assert failed.status == "fail"
    assert failed.metrics["finite_fraction"] == 0.5
    close = parity_check(np.ones(3), np.ones(3) + 1e-7, atol=1e-6, rtol=0)
    assert close.status == "pass"
    assert close.metrics["max_abs_error"] > 0


def test_batch_partition_detects_batch_dependent_adapter():
    values = np.arange(24, dtype=np.float32).reshape(6, 4)
    stable = batch_partition_check(lambda batch: batch * 2, values, (1, 2, 3), atol=0, rtol=0)
    assert stable.status == "pass"
    unstable = batch_partition_check(
        lambda batch: batch - batch.mean(axis=0, keepdims=True),
        values, (1, 2, 3), atol=0, rtol=0,
    )
    assert unstable.status == "fail"


def test_scaling_roundtrip_and_prefix_invariance_are_causal():
    rng = np.random.default_rng(4)
    values = rng.normal(size=(3, 8, 2)).astype(np.float32)

    def scale(batch):
        mean = batch[:, :4].mean(axis=1, keepdims=True)
        std = batch[:, :4].std(axis=1, keepdims=True) + 1e-6
        return (batch - mean) / std, (mean, std)

    def inverse(batch, state):
        mean, std = state
        return batch * std + mean

    result = scaling_roundtrip_check(scale, inverse, values, atol=1e-5, rtol=1e-5)
    assert result.status == "pass"

    changed = values.copy()
    changed[:, 4:] += 1000
    causal = prefix_invariance_check(
        lambda batch: scale(batch)[0], values, changed,
        prefix_length=4, atol=0, rtol=0,
    )
    assert causal.status == "pass"
    leaking = prefix_invariance_check(
        lambda batch: (batch - batch.mean(axis=1, keepdims=True)), values, changed,
        prefix_length=4, atol=0, rtol=0,
    )
    assert leaking.status == "fail"


def test_forward_backward_and_control_checks_require_real_signal():
    passed = forward_backward_check(lambda: {
        "loss": 1.0, "grad_norm": 0.5, "parameter_delta": 0.01,
    })
    assert passed.status == "pass"
    assert forward_backward_check(lambda: {
        "loss": 1.0, "grad_norm": 0.0, "parameter_delta": 0.0,
    }).status == "fail"

    assert control_rejection_check(
        0.8, (0.4, 0.5), (0.3, 0.45), margin=0.2,
    ).status == "pass"
    assert control_rejection_check(
        0.5, (0.49,), (0.48,), margin=0.05,
    ).status == "fail"
    assert control_rejection_check(
        0.1, (0.4,), (0.3,), margin=0.1, higher_is_better=False,
    ).status == "pass"


def test_controlled_loss_decrease_and_resume_parity():
    state = {"weight": 0.0}

    def evaluate():
        return (state["weight"] - 2.0) ** 2

    def step():
        state["weight"] += 0.25 * (2.0 - state["weight"])

    result = loss_decrease_check(
        evaluate, step, steps=8, min_relative_decrease=0.8, tail=2,
    )
    assert result.status == "pass"
    assert result.metrics["relative_decrease"] > 0.8

    def uninterrupted():
        return {
            "trajectory": [4.0, 2.0, 1.0],
            "model": np.asarray([1.0, 2.0]),
            "optimizer": {"step": 3, "momentum": np.asarray([0.4])},
        }

    assert interruption_resume_parity_check(
        uninterrupted, uninterrupted, atol=0, rtol=0,
    ).status == "pass"
    assert interruption_resume_parity_check(
        uninterrupted,
        lambda: {**uninterrupted(), "model": np.asarray([1.0, 2.1])},
        atol=0,
        rtol=0,
    ).status == "fail"

    import torch
    bf16_state = lambda: {"model": torch.tensor([1.0, 2.0], dtype=torch.bfloat16)}
    assert interruption_resume_parity_check(
        bf16_state, bf16_state, atol=0, rtol=0,
    ).status == "pass"
    assert interruption_resume_parity_check(
        bf16_state,
        lambda: {"model": torch.tensor([1.0, 2.5], dtype=torch.bfloat16)},
        atol=0,
        rtol=0,
    ).status == "fail"


def test_future_corruption_and_boundary_rejection():
    left = np.arange(16, dtype=np.float32).reshape(2, 8)
    right = left.copy()
    right[:, 4:] += 1000
    causal = future_corruption_check(
        lambda batch: batch[:, :4].sum(axis=1),
        left,
        right,
        visible_length=4,
        atol=0,
        rtol=0,
    )
    assert causal.status == "pass"
    leaking = future_corruption_check(
        lambda batch: batch.sum(axis=1),
        left,
        right,
        visible_length=4,
        atol=0,
        rtol=0,
    )
    assert leaking.status == "fail"

    def reject_negative(value):
        if value < 0:
            raise ValueError("negative")

    assert rejection_check(
        reject_negative, {"roll": -1, "gap": -2, "split": -3},
    ).status == "pass"
    assert rejection_check(
        reject_negative, {"roll": -1, "accepted": 1},
    ).status == "fail"


def test_layout_performance_and_negative_price_contracts():
    values = np.ones((2, 8, 5), np.float32)
    assert multivariate_layout_check(
        lambda batch: batch.mean(axis=1),
        values,
        expected_output_shape=(2, 5),
    ).status == "pass"
    assert multivariate_layout_check(
        lambda batch: batch.mean(axis=(1, 2)),
        values,
        expected_output_shape=(2, 5),
    ).status == "fail"

    performance = performance_check(
        lambda: np.add(values, 1.0),
        batch_size=2,
        repeats=2,
        warmups=0,
        min_examples_per_second=0.0,
        max_peak_bytes=10_000_000,
    )
    assert performance.status == "pass"
    assert performance.metrics["examples_per_second"] > 0

    negative = np.asarray([[-2.0, 1.0]], np.float32)
    assert negative_price_behavior_check(
        lambda batch: batch * 2, negative, behavior="support",
    ).status == "pass"

    def reject(batch):
        if np.any(batch < 0):
            raise ValueError("negative prices unsupported")
        return batch

    assert negative_price_behavior_check(
        reject, negative, behavior="reject",
    ).status == "pass"
    assert negative_price_behavior_check(
        lambda batch: batch, negative, behavior="reject",
    ).status == "fail"


def test_context_boundaries_and_channel_independence():
    values = np.arange(2 * 8 * 3, dtype=np.float32).reshape(2, 8, 3)
    boundaries = context_boundary_check(
        lambda batch, length: batch[:, -1:] + length,
        values, (1, 4, 8),
    )
    assert boundaries.status == "pass"
    assert [row["context"] for row in boundaries.metrics["boundaries"]] == [1, 4, 8]

    independent = channel_independence_check(
        lambda batch: batch * 3,
        values,
        perturb_channel=1,
        perturbation=5,
        atol=0,
        rtol=0,
    )
    assert independent.status == "pass"
    mixed = channel_independence_check(
        lambda batch: batch + batch.mean(axis=-1, keepdims=True),
        values,
        perturb_channel=1,
        perturbation=5,
        atol=0,
        rtol=0,
    )
    assert mixed.status == "fail"
