"""Shared, model-agnostic parity checks for foundation-model admission.

Adapters remain family-specific.  These checks are deliberately small and deterministic so every
family proves the same invariants without pretending that preprocessing, outputs, or adaptation
surfaces are universal.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np


ArrayLike = Any


@dataclass(frozen=True)
class CheckResult:
    status: str
    metrics: dict[str, Any]
    reason: str | None = None

    def manifest(self) -> dict[str, Any]:
        return asdict(self)


def _array(value: ArrayLike, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype == object:
        raise ValueError(f"{name} produced an object array")
    return array


def _difference(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, float]:
    absolute = np.abs(candidate.astype(np.float64) - reference.astype(np.float64))
    max_abs = float(absolute.max(initial=0.0))
    scale = np.maximum(np.abs(reference.astype(np.float64)), 1e-12)
    max_rel = float((absolute / scale).max(initial=0.0))
    return max_abs, max_rel


def finite_check(value: ArrayLike) -> CheckResult:
    array = _array(value, "adapter")
    finite = np.isfinite(array)
    return CheckResult(
        status="pass" if bool(finite.all()) else "fail",
        metrics={
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "finite_fraction": float(finite.mean()) if array.size else 1.0,
        },
        reason=None if bool(finite.all()) else "output contains non-finite values",
    )


def parity_check(
    reference: ArrayLike,
    candidate: ArrayLike,
    *,
    atol: float,
    rtol: float,
    name: str = "output",
) -> CheckResult:
    if atol < 0 or rtol < 0:
        raise ValueError("parity tolerances must be nonnegative")
    expected = _array(reference, "reference")
    actual = _array(candidate, name)
    if actual.shape != expected.shape:
        return CheckResult(
            status="fail",
            metrics={"reference_shape": list(expected.shape), "candidate_shape": list(actual.shape)},
            reason=f"{name} shape differs from reference",
        )
    max_abs, max_rel = _difference(expected, actual)
    passed = bool(np.allclose(expected, actual, atol=atol, rtol=rtol, equal_nan=False))
    return CheckResult(
        status="pass" if passed else "fail",
        metrics={
            "shape": list(expected.shape),
            "atol": float(atol),
            "rtol": float(rtol),
            "max_abs_error": max_abs,
            "max_relative_error": max_rel,
        },
        reason=None if passed else f"{name} exceeds parity tolerance",
    )


def batch_partition_check(
    adapter: Callable[[np.ndarray], ArrayLike],
    inputs: ArrayLike,
    partitions: Sequence[int],
    *,
    atol: float,
    rtol: float,
) -> CheckResult:
    """Compare one full batch with concatenated outputs from declared partitions."""
    values = _array(inputs, "inputs")
    if values.ndim < 1 or len(values) < 1:
        raise ValueError("batch partition check requires at least one row")
    sizes = tuple(int(size) for size in partitions)
    if not sizes or any(size < 1 for size in sizes) or sum(sizes) != len(values):
        raise ValueError("partitions must be positive and sum to the batch size")
    reference = _array(adapter(values), "full batch output")
    pieces = []
    start = 0
    for size in sizes:
        piece = _array(adapter(values[start:start + size]), "partition output")
        pieces.append(piece)
        start += size
    try:
        candidate = np.concatenate(pieces, axis=0)
    except ValueError as exc:
        return CheckResult(
            status="fail",
            metrics={"partition_shapes": [list(piece.shape) for piece in pieces]},
            reason=f"partition outputs cannot be concatenated: {exc}",
        )
    result = parity_check(reference, candidate, atol=atol, rtol=rtol, name="partitioned output")
    metrics = dict(result.metrics)
    metrics["partitions"] = list(sizes)
    return CheckResult(status=result.status, metrics=metrics, reason=result.reason)


def scaling_roundtrip_check(
    fit_transform: Callable[[np.ndarray], tuple[ArrayLike, Any]],
    inverse_transform: Callable[[np.ndarray, Any], ArrayLike],
    values: ArrayLike,
    *,
    atol: float,
    rtol: float,
) -> CheckResult:
    """Verify causal scaling and inverse scaling are a numerical round trip."""
    original = _array(values, "values")
    scaled, state = fit_transform(original.copy())
    scaled = _array(scaled, "scaled values")
    if scaled.shape != original.shape:
        return CheckResult(
            status="fail",
            metrics={"input_shape": list(original.shape), "scaled_shape": list(scaled.shape)},
            reason="scaler changed the value shape",
        )
    restored = inverse_transform(scaled.copy(), state)
    result = parity_check(original, restored, atol=atol, rtol=rtol, name="inverse-scaled values")
    metrics = dict(result.metrics)
    metrics["scaled_finite"] = bool(np.isfinite(scaled).all())
    status = result.status if metrics["scaled_finite"] else "fail"
    reason = result.reason if metrics["scaled_finite"] else "scaled values are non-finite"
    return CheckResult(status=status, metrics=metrics, reason=reason)


def prefix_invariance_check(
    transform: Callable[[np.ndarray], ArrayLike],
    first: ArrayLike,
    second: ArrayLike,
    *,
    prefix_length: int,
    atol: float,
    rtol: float,
) -> CheckResult:
    """Changing data after a causal prefix must not alter transformed prefix values."""
    left = _array(first, "first input")
    right = _array(second, "second input")
    if left.shape != right.shape:
        raise ValueError("prefix-invariance inputs must share a shape")
    if left.ndim < 2 or not 1 <= prefix_length <= left.shape[1]:
        raise ValueError("prefix_length must address axis 1")
    if not np.array_equal(left[:, :prefix_length], right[:, :prefix_length]):
        raise ValueError("prefix-invariance inputs must have identical causal prefixes")
    transformed_left = _array(transform(left.copy()), "first transformed input")
    transformed_right = _array(transform(right.copy()), "second transformed input")
    if transformed_left.ndim < 2 or transformed_right.ndim < 2:
        return CheckResult(
            status="fail", metrics={}, reason="transform output has no causal time axis"
        )
    result = parity_check(
        transformed_left[:, :prefix_length], transformed_right[:, :prefix_length],
        atol=atol, rtol=rtol, name="transformed causal prefix",
    )
    metrics = dict(result.metrics)
    metrics["prefix_length"] = int(prefix_length)
    return CheckResult(status=result.status, metrics=metrics, reason=result.reason)


def context_boundary_check(
    adapter: Callable[[np.ndarray, int], ArrayLike],
    values: ArrayLike,
    lengths: Sequence[int],
) -> CheckResult:
    """Exercise every declared context boundary and record exact output shapes."""
    array = _array(values, "values")
    checked = []
    failures = []
    for length in tuple(int(value) for value in lengths):
        if length < 1 or length > array.shape[1]:
            raise ValueError(f"invalid context boundary {length}")
        try:
            output = _array(adapter(array[:, -length:].copy(), length), f"context={length}")
            finite = bool(np.isfinite(output).all())
            checked.append({"context": length, "shape": list(output.shape), "finite": finite})
            if not finite:
                failures.append(f"context {length} produced non-finite output")
        except Exception as exc:  # the harness records adapter failures rather than hiding them
            checked.append({"context": length, "error": f"{type(exc).__name__}: {exc}"})
            failures.append(f"context {length} raised {type(exc).__name__}")
    return CheckResult(
        status="pass" if not failures else "fail",
        metrics={"boundaries": checked},
        reason=None if not failures else "; ".join(failures),
    )


def channel_independence_check(
    adapter: Callable[[np.ndarray], ArrayLike],
    values: ArrayLike,
    *,
    channel_axis: int = -1,
    perturb_channel: int = 0,
    perturbation: float = 1.0,
    atol: float,
    rtol: float,
) -> CheckResult:
    """For channel-independent adapters, perturbing one input channel may change only it."""
    inputs = _array(values, "values").astype(np.float64, copy=True)
    axis = channel_axis if channel_axis >= 0 else inputs.ndim + channel_axis
    if axis <= 0 or axis >= inputs.ndim:
        raise ValueError("channel_axis must not be the batch axis")
    channel_count = inputs.shape[axis]
    if not 0 <= perturb_channel < channel_count:
        raise ValueError("perturb_channel is out of range")
    baseline = _array(adapter(inputs.copy()), "baseline output")
    changed_inputs = inputs.copy()
    selector = [slice(None)] * inputs.ndim
    selector[axis] = perturb_channel
    changed_inputs[tuple(selector)] += float(perturbation)
    changed = _array(adapter(changed_inputs), "perturbed output")
    if baseline.shape != changed.shape or baseline.ndim != inputs.ndim:
        return CheckResult(
            status="fail",
            metrics={"baseline_shape": list(baseline.shape), "changed_shape": list(changed.shape)},
            reason="channel-independence check requires output shape aligned to input channels",
        )
    output_axis = axis
    unaffected = [index for index in range(channel_count) if index != perturb_channel]
    if unaffected:
        baseline_unaffected = np.take(baseline, unaffected, axis=output_axis)
        changed_unaffected = np.take(changed, unaffected, axis=output_axis)
        result = parity_check(
            baseline_unaffected, changed_unaffected,
            atol=atol, rtol=rtol, name="unperturbed output channels",
        )
    else:
        result = CheckResult(status="pass", metrics={"shape": list(baseline.shape)})
    affected_before = np.take(baseline, [perturb_channel], axis=output_axis)
    affected_after = np.take(changed, [perturb_channel], axis=output_axis)
    changed_effect = float(np.max(np.abs(affected_after - affected_before), initial=0.0))
    metrics = dict(result.metrics)
    metrics.update({
        "perturb_channel": int(perturb_channel),
        "perturbation": float(perturbation),
        "affected_channel_max_change": changed_effect,
    })
    return CheckResult(status=result.status, metrics=metrics, reason=result.reason)


def check_manifest(results: Mapping[str, CheckResult | Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Convert named harness results to the admission-report check representation."""
    output: dict[str, dict[str, Any]] = {}
    for name, result in results.items():
        if isinstance(result, CheckResult):
            output[name] = result.manifest()
        else:
            output[name] = dict(result)
    return output
