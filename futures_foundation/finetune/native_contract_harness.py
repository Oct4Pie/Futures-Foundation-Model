"""Shared, model-agnostic parity checks for foundation-model admission.

Adapters remain family-specific.  These checks are deliberately small and deterministic so every
family proves the same invariants without pretending that preprocessing, outputs, or adaptation
surfaces are universal.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from time import perf_counter
import tracemalloc
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


def forward_backward_check(
    run_batch: Callable[[], Mapping[str, Any]],
) -> CheckResult:
    """Require one batch to produce finite loss, gradient flow, and a parameter update."""
    try:
        result = run_batch()
    except Exception as exc:
        return CheckResult(
            status="fail",
            metrics={"error_type": type(exc).__name__},
            reason=f"forward/backward batch raised {type(exc).__name__}: {exc}",
        )
    if not isinstance(result, Mapping):
        raise ValueError("forward/backward runner must return a mapping")
    required = {"loss", "grad_norm", "parameter_delta"}
    missing = sorted(required - set(result))
    if missing:
        raise ValueError(f"forward/backward result is missing fields: {missing}")
    metrics = {
        "loss": float(result["loss"]),
        "grad_norm": float(result["grad_norm"]),
        "parameter_delta": float(result["parameter_delta"]),
    }
    finite = all(math.isfinite(value) for value in metrics.values())
    passed = finite and metrics["grad_norm"] > 0.0 and metrics["parameter_delta"] > 0.0
    return CheckResult(
        status="pass" if passed else "fail",
        metrics=metrics,
        reason=(
            None if passed else
            "one-batch training did not produce finite loss, positive gradient flow, and a parameter update"
        ),
    )


def loss_decrease_check(
    evaluate_loss: Callable[[], float],
    train_step: Callable[[], Any],
    *,
    steps: int,
    min_relative_decrease: float = 0.05,
    tail: int = 1,
) -> CheckResult:
    """Prove a controlled learnable fixture decreases loss, not merely that code runs."""
    if steps < 1 or tail < 1 or tail > steps:
        raise ValueError("loss-decrease steps/tail are invalid")
    if not 0.0 <= min_relative_decrease < 1.0:
        raise ValueError("min_relative_decrease must lie in [0,1)")
    losses = [float(evaluate_loss())]
    try:
        for _ in range(steps):
            train_step()
            losses.append(float(evaluate_loss()))
    except Exception as exc:
        return CheckResult(
            status="fail",
            metrics={"losses": losses, "error_type": type(exc).__name__},
            reason=f"controlled training raised {type(exc).__name__}: {exc}",
        )
    finite = bool(np.isfinite(np.asarray(losses, np.float64)).all())
    initial = losses[0]
    final = float(np.mean(losses[-tail:]))
    denominator = max(abs(initial), 1e-12)
    relative = float((initial - final) / denominator)
    passed = finite and final < initial and relative >= min_relative_decrease
    return CheckResult(
        status="pass" if passed else "fail",
        metrics={
            "initial_loss": initial,
            "final_tail_mean_loss": final,
            "relative_decrease": relative,
            "required_relative_decrease": float(min_relative_decrease),
            "steps": int(steps),
            "tail": int(tail),
            "losses": losses,
        },
        reason=None if passed else "controlled learnable loss did not decrease enough",
    )


def control_rejection_check(
    real_score: float,
    shuffle_scores: Sequence[float],
    time_destroyed_scores: Sequence[float],
    *,
    margin: float = 0.0,
    higher_is_better: bool = True,
) -> CheckResult:
    """Require real chronology to beat both shuffled and time-destroyed controls."""
    if margin < 0:
        raise ValueError("control margin must be nonnegative")
    shuffle = np.asarray(tuple(shuffle_scores), np.float64)
    destroyed = np.asarray(tuple(time_destroyed_scores), np.float64)
    if not len(shuffle) or not len(destroyed):
        raise ValueError("both control families require at least one score")
    all_values = np.r_[float(real_score), shuffle, destroyed]
    finite = bool(np.isfinite(all_values).all())
    if higher_is_better:
        hardest = float(max(shuffle.max(), destroyed.max()))
        separation = float(real_score - hardest)
    else:
        hardest = float(min(shuffle.min(), destroyed.min()))
        separation = float(hardest - real_score)
    passed = finite and separation >= margin
    return CheckResult(
        status="pass" if passed else "fail",
        metrics={
            "real_score": float(real_score),
            "shuffle_scores": shuffle.tolist(),
            "time_destroyed_scores": destroyed.tolist(),
            "hardest_control_score": hardest,
            "separation": separation,
            "required_margin": float(margin),
            "higher_is_better": bool(higher_is_better),
        },
        reason=None if passed else "real chronology did not reject both control families",
    )


def _state_differences(
    reference: Any,
    candidate: Any,
    *,
    path: str,
    atol: float,
    rtol: float,
    differences: list[dict[str, Any]],
) -> None:
    if isinstance(reference, Mapping) or isinstance(candidate, Mapping):
        if not isinstance(reference, Mapping) or not isinstance(candidate, Mapping):
            differences.append({"path": path, "reason": "type mismatch"})
            return
        if set(reference) != set(candidate):
            differences.append({
                "path": path,
                "reason": "mapping keys differ",
                "reference_keys": sorted(map(str, reference)),
                "candidate_keys": sorted(map(str, candidate)),
            })
            return
        for key in sorted(reference, key=str):
            _state_differences(
                reference[key], candidate[key], path=f"{path}.{key}",
                atol=atol, rtol=rtol, differences=differences,
            )
        return
    if isinstance(reference, (tuple, list)) or isinstance(candidate, (tuple, list)):
        if not isinstance(reference, (tuple, list)) or not isinstance(candidate, (tuple, list)):
            differences.append({"path": path, "reason": "type mismatch"})
            return
        if len(reference) != len(candidate):
            differences.append({
                "path": path, "reason": "sequence length differs",
                "reference_length": len(reference), "candidate_length": len(candidate),
            })
            return
        for index, (left, right) in enumerate(zip(reference, candidate)):
            _state_differences(
                left, right, path=f"{path}[{index}]", atol=atol, rtol=rtol,
                differences=differences,
            )
        return
    if isinstance(reference, (str, bytes, bool, type(None))) or isinstance(
        candidate, (str, bytes, bool, type(None))
    ):
        if reference != candidate:
            differences.append({"path": path, "reason": "value differs"})
        return
    try:
        import torch
    except ImportError:  # pragma: no cover - torch-free environments use NumPy below
        torch = None
    if torch is not None and (
        isinstance(reference, torch.Tensor) or isinstance(candidate, torch.Tensor)
    ):
        if not isinstance(reference, torch.Tensor) or not isinstance(candidate, torch.Tensor):
            differences.append({"path": path, "reason": "tensor type mismatch"})
            return
        if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
            differences.append({
                "path": path,
                "reason": "tensor shape or dtype differs",
                "reference_shape": list(reference.shape),
                "candidate_shape": list(candidate.shape),
                "reference_dtype": str(reference.dtype),
                "candidate_dtype": str(candidate.dtype),
            })
            return
        left_tensor = reference.detach().cpu()
        right_tensor = candidate.detach().cpu()
        if atol == 0.0 and rtol == 0.0:
            passed = bool(torch.equal(left_tensor, right_tensor))
        elif left_tensor.is_floating_point() or left_tensor.is_complex():
            passed = bool(torch.allclose(
                left_tensor.float(), right_tensor.float(),
                atol=atol, rtol=rtol, equal_nan=False,
            ))
        else:
            passed = bool(torch.equal(left_tensor, right_tensor))
        if not passed:
            metrics: dict[str, Any] = {"path": path, "reason": "tensor differs"}
            if left_tensor.is_floating_point() or left_tensor.is_complex():
                absolute = torch.abs(left_tensor.float() - right_tensor.float())
                scale = torch.clamp(torch.abs(left_tensor.float()), min=1e-12)
                metrics.update({
                    "max_abs_error": float(absolute.max().item()) if absolute.numel() else 0.0,
                    "max_relative_error": float((absolute / scale).max().item()) if absolute.numel() else 0.0,
                })
            differences.append(metrics)
        return
    try:
        left = _array(reference, f"{path} reference")
        right = _array(candidate, f"{path} candidate")
    except (TypeError, ValueError):
        if reference != candidate:
            differences.append({"path": path, "reason": "unsupported value differs"})
        return
    if left.shape != right.shape:
        differences.append({
            "path": path, "reason": "shape differs",
            "reference_shape": list(left.shape), "candidate_shape": list(right.shape),
        })
        return
    if left.dtype.kind in "iub" and right.dtype.kind in "iub":
        passed = bool(np.array_equal(left, right))
    elif left.dtype.kind in "fc" and right.dtype.kind in "fc":
        passed = bool(np.allclose(left, right, atol=atol, rtol=rtol, equal_nan=False))
    else:
        passed = bool(np.array_equal(left, right))
    if not passed:
        metrics: dict[str, Any] = {"path": path, "reason": "array differs"}
        if left.dtype.kind in "ifuc" and right.dtype.kind in "ifuc":
            max_abs, max_rel = _difference(left, right)
            metrics.update({"max_abs_error": max_abs, "max_relative_error": max_rel})
        differences.append(metrics)


def interruption_resume_parity_check(
    uninterrupted: Callable[[], Any],
    interrupted_resume: Callable[[], Any],
    *,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> CheckResult:
    """Compare exact final state and trajectory from uninterrupted versus resumed runs."""
    if atol < 0 or rtol < 0:
        raise ValueError("resume parity tolerances must be nonnegative")
    try:
        reference = uninterrupted()
        candidate = interrupted_resume()
    except Exception as exc:
        return CheckResult(
            status="fail",
            metrics={"error_type": type(exc).__name__},
            reason=f"resume parity execution raised {type(exc).__name__}: {exc}",
        )
    differences: list[dict[str, Any]] = []
    _state_differences(
        reference, candidate, path="state", atol=atol, rtol=rtol,
        differences=differences,
    )
    return CheckResult(
        status="pass" if not differences else "fail",
        metrics={
            "atol": float(atol), "rtol": float(rtol),
            "difference_count": len(differences),
            "differences": differences[:32],
        },
        reason=None if not differences else "interrupted/resumed trajectory differs",
    )


def future_corruption_check(
    predict: Callable[[np.ndarray], ArrayLike],
    first: ArrayLike,
    second: ArrayLike,
    *,
    visible_length: int,
    atol: float,
    rtol: float,
) -> CheckResult:
    """Corrupt unavailable future values and require the decision output to remain invariant."""
    left = _array(first, "first input")
    right = _array(second, "second input")
    if left.shape != right.shape or left.ndim < 2:
        raise ValueError("future-corruption inputs must share a batch/time shape")
    if not 1 <= visible_length < left.shape[1]:
        raise ValueError("visible_length must leave a non-empty hidden future")
    if not np.array_equal(left[:, :visible_length], right[:, :visible_length]):
        raise ValueError("future-corruption inputs must share the visible prefix")
    if np.array_equal(left[:, visible_length:], right[:, visible_length:]):
        raise ValueError("future-corruption inputs must differ after the visible prefix")
    reference = predict(left.copy())
    candidate = predict(right.copy())
    result = parity_check(
        reference, candidate, atol=atol, rtol=rtol,
        name="future-corrupted decision output",
    )
    metrics = dict(result.metrics)
    metrics["visible_length"] = int(visible_length)
    return CheckResult(status=result.status, metrics=metrics, reason=result.reason)


def rejection_check(
    validator: Callable[[Any], Any],
    cases: Mapping[str, Any],
) -> CheckResult:
    """Require every declared invalid boundary/control case to fail closed."""
    if not cases:
        raise ValueError("rejection check requires at least one named case")
    rows = []
    failures = []
    for name, value in cases.items():
        try:
            validator(value)
        except Exception as exc:
            rows.append({"case": str(name), "rejected": True, "error_type": type(exc).__name__})
        else:
            rows.append({"case": str(name), "rejected": False})
            failures.append(str(name))
    return CheckResult(
        status="pass" if not failures else "fail",
        metrics={"cases": rows, "rejected": len(rows) - len(failures), "total": len(rows)},
        reason=None if not failures else f"invalid cases were accepted: {failures}",
    )


def multivariate_layout_check(
    adapter: Callable[[np.ndarray], ArrayLike],
    values: ArrayLike,
    *,
    expected_output_shape: Sequence[int],
) -> CheckResult:
    """Exercise one declared multivariate grouping/layout contract exactly."""
    inputs = _array(values, "multivariate values")
    try:
        output = _array(adapter(inputs.copy()), "multivariate output")
    except Exception as exc:
        return CheckResult(
            status="fail",
            metrics={"input_shape": list(inputs.shape), "error_type": type(exc).__name__},
            reason=f"multivariate adapter raised {type(exc).__name__}: {exc}",
        )
    expected = tuple(int(value) for value in expected_output_shape)
    finite = bool(np.isfinite(output).all())
    passed = output.shape == expected and finite
    return CheckResult(
        status="pass" if passed else "fail",
        metrics={
            "input_shape": list(inputs.shape), "output_shape": list(output.shape),
            "expected_output_shape": list(expected), "finite": finite,
        },
        reason=None if passed else "multivariate output shape or finiteness differs from contract",
    )


def performance_check(
    run_batch: Callable[[], Any],
    *,
    batch_size: int,
    repeats: int = 3,
    warmups: int = 1,
    min_examples_per_second: float = 0.0,
    max_peak_bytes: int | None = None,
    memory_probe: Callable[[], int] | None = None,
) -> CheckResult:
    """Measure bounded throughput and peak memory without defining family-specific budgets."""
    if batch_size < 1 or repeats < 1 or warmups < 0 or min_examples_per_second < 0:
        raise ValueError("performance-check budgets must be nonnegative and non-empty")
    if max_peak_bytes is not None and max_peak_bytes < 1:
        raise ValueError("max_peak_bytes must be positive when supplied")
    try:
        for _ in range(warmups):
            run_batch()
        use_tracemalloc = memory_probe is None
        if use_tracemalloc:
            tracemalloc.start()
        start = perf_counter()
        observed_memory = 0
        for _ in range(repeats):
            run_batch()
            if memory_probe is not None:
                observed_memory = max(observed_memory, int(memory_probe()))
        elapsed = perf_counter() - start
        if use_tracemalloc:
            _, observed_memory = tracemalloc.get_traced_memory()
            tracemalloc.stop()
    except Exception as exc:
        if tracemalloc.is_tracing():
            tracemalloc.stop()
        return CheckResult(
            status="fail", metrics={"error_type": type(exc).__name__},
            reason=f"performance measurement raised {type(exc).__name__}: {exc}",
        )
    examples = int(batch_size * repeats)
    throughput = float(examples / max(elapsed, 1e-12))
    passed = throughput >= min_examples_per_second and (
        max_peak_bytes is None or observed_memory <= max_peak_bytes
    )
    return CheckResult(
        status="pass" if passed else "fail",
        metrics={
            "batch_size": int(batch_size), "repeats": int(repeats),
            "warmups": int(warmups), "elapsed_seconds": float(elapsed),
            "examples_per_second": throughput, "min_examples_per_second": float(min_examples_per_second),
            "peak_bytes": int(observed_memory), "max_peak_bytes": max_peak_bytes,
        },
        reason=None if passed else "throughput or peak memory exceeded the declared bound",
    )


def negative_price_behavior_check(
    transform: Callable[[np.ndarray], ArrayLike],
    values: ArrayLike,
    *,
    behavior: str,
) -> CheckResult:
    """Verify a route either supports finite negative prices or rejects them explicitly."""
    if behavior not in {"support", "reject"}:
        raise ValueError("negative-price behavior must be 'support' or 'reject'")
    inputs = _array(values, "negative-price inputs")
    if not np.any(inputs < 0):
        raise ValueError("negative-price check requires at least one negative value")
    try:
        output = _array(transform(inputs.copy()), "negative-price output")
    except Exception as exc:
        passed = behavior == "reject"
        return CheckResult(
            status="pass" if passed else "fail",
            metrics={"behavior": behavior, "error_type": type(exc).__name__},
            reason=None if passed else f"negative-price support raised {type(exc).__name__}: {exc}",
        )
    finite = bool(np.isfinite(output).all())
    passed = behavior == "support" and finite
    return CheckResult(
        status="pass" if passed else "fail",
        metrics={"behavior": behavior, "output_shape": list(output.shape), "finite": finite},
        reason=None if passed else "negative prices were accepted under reject policy or produced non-finite output",
    )


def check_manifest(results: Mapping[str, CheckResult | Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Convert named harness results to the admission-report check representation."""
    output: dict[str, dict[str, Any]] = {}
    for name, result in results.items():
        if isinstance(result, CheckResult):
            output[name] = result.manifest()
        else:
            output[name] = dict(result)
    return output
