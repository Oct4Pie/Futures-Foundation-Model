import numpy as np

from futures_foundation.finetune.native_contract_harness import (
    batch_partition_check,
    channel_independence_check,
    context_boundary_check,
    finite_check,
    parity_check,
    prefix_invariance_check,
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
