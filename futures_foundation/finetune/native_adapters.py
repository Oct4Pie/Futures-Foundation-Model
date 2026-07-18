"""Narrow native-output adapters for admitted foundation-model tracks.

These helpers intentionally expose only public upstream inference/representation
interfaces.  They do not pool hidden states, invent forecast heads, append synthetic
context, or silently alter channel grouping.  Heavy third-party imports stay inside the
call sites so the core package remains importable without every model environment.
"""
from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np


class NativeAdapterError(ValueError):
    """Raised when an input would violate a frozen native model contract."""


def _as_finite_float32(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if not np.isfinite(array).all():
        raise NativeAdapterError(f"{name} must contain only finite values")
    return array


def left_pad_channel_first(
    values: Any,
    *,
    target_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Left-pad ``[batch,time,channel]`` values and return MOMENT's sequence mask."""
    array = _as_finite_float32(values, name="values")
    if array.ndim != 3:
        raise NativeAdapterError(f"values must be [batch,time,channel], got {array.shape}")
    if target_length < 1 or array.shape[1] > target_length:
        raise NativeAdapterError(
            f"target_length must be >= input length; got {target_length} for {array.shape[1]}"
        )
    padded = np.zeros((array.shape[0], array.shape[2], target_length), dtype=np.float32)
    mask = np.zeros((array.shape[0], target_length), dtype=np.float32)
    padded[:, :, -array.shape[1] :] = array.transpose(0, 2, 1)
    mask[:, -array.shape[1] :] = 1.0
    return padded, mask


def mantis_native_representation(
    model: Any,
    values: Any,
    *,
    batch_size: int = 256,
    target_length: int = 512,
) -> np.ndarray:
    """Return official per-channel Mantis representations as ``[B,C,D]``.

    Mantis is natively univariate.  Channel fusion is deliberately not performed here;
    flattening, averaging, or concatenating the channel axis is a separate Track-C choice.
    """
    if batch_size < 1 or target_length < 1:
        raise NativeAdapterError("batch_size and target_length must be positive")
    array = _as_finite_float32(values, name="values")
    if array.ndim != 3:
        raise NativeAdapterError(f"values must be [batch,time,channel], got {array.shape}")
    import torch
    import torch.nn.functional as functional
    from mantis.trainer import MantisTrainer

    tensor = torch.as_tensor(array.transpose(0, 2, 1).reshape(-1, 1, array.shape[1]))
    if tensor.shape[-1] != target_length:
        tensor = functional.interpolate(
            tensor,
            size=target_length,
            mode="linear",
            align_corners=False,
        )
    try:
        device = str(next(model.parameters()).device)
    except (AttributeError, StopIteration):
        device = "cpu"
    trainer = MantisTrainer(device=device, network=model)
    representation = np.asarray(
        trainer.transform(tensor.numpy(), batch_size=batch_size), dtype=np.float32
    )
    if representation.ndim != 2 or representation.shape[0] != tensor.shape[0]:
        raise NativeAdapterError(
            f"unexpected Mantis representation shape: {representation.shape}"
        )
    if not np.isfinite(representation).all():
        raise NativeAdapterError("Mantis representation contains non-finite values")
    return representation.reshape(array.shape[0], array.shape[2], -1)


def moment_native_embedding(
    model: Any,
    values: Any,
    input_mask: Any,
) -> Any:
    """Call MOMENT's official masked ``embed(..., reduction='mean')`` contract."""
    import torch

    tensor = values if isinstance(values, torch.Tensor) else torch.as_tensor(values)
    mask = input_mask if isinstance(input_mask, torch.Tensor) else torch.as_tensor(input_mask)
    if tensor.ndim != 3 or mask.ndim != 2 or tensor.shape[0] != mask.shape[0]:
        raise NativeAdapterError(
            f"MOMENT expects values [B,C,T] and mask [B,T], got {tensor.shape}, {mask.shape}"
        )
    if tensor.shape[-1] != mask.shape[-1]:
        raise NativeAdapterError("MOMENT mask length must match the sequence length")
    output = model.embed(x_enc=tensor, input_mask=mask, reduction="mean").embeddings
    if output.ndim != 2 or output.shape[0] != tensor.shape[0]:
        raise NativeAdapterError(f"unexpected MOMENT embedding shape: {tuple(output.shape)}")
    if not torch.isfinite(output).all():
        raise NativeAdapterError("MOMENT embedding contains non-finite values")
    return output


def kronos_native_forecast(
    predictor: Any,
    frames: Sequence[Any],
    context_timestamps: Sequence[Any],
    forecast_timestamps: Sequence[Any],
    *,
    prediction_length: int,
    temperature: float = 1.0,
    top_k: int = 1,
    top_p: float = 1.0,
    sample_count: int = 1,
    verbose: bool = False,
) -> list[Any]:
    """Call the official Kronos batch predictor without custom hidden-state access."""
    if not frames or len(frames) != len(context_timestamps) or len(frames) != len(forecast_timestamps):
        raise NativeAdapterError("Kronos frames and timestamp lists must have equal nonzero length")
    if prediction_length < 1 or sample_count < 1:
        raise NativeAdapterError("prediction_length and sample_count must be positive")
    output = predictor.predict_batch(
        list(frames),
        list(context_timestamps),
        list(forecast_timestamps),
        pred_len=prediction_length,
        T=temperature,
        top_k=top_k,
        top_p=top_p,
        sample_count=sample_count,
        verbose=verbose,
    )
    if len(output) != len(frames):
        raise NativeAdapterError("Kronos returned the wrong batch cardinality")
    return list(output)


def chronos_native_quantiles(
    pipeline: Any,
    inputs: Any,
    *,
    family: str,
    prediction_length: int,
    quantile_levels: Sequence[float],
    batch_size: int = 256,
    context_length: int | None = None,
    num_samples: int = 64,
) -> tuple[Any, Any]:
    """Dispatch to the public Chronos family quantile API with native shapes."""
    if prediction_length < 1 or not quantile_levels:
        raise NativeAdapterError("prediction_length and quantile_levels are required")
    if family == "chronos_2":
        return pipeline.predict_quantiles(
            inputs,
            prediction_length=prediction_length,
            quantile_levels=list(quantile_levels),
            batch_size=batch_size,
            context_length=context_length,
            cross_learning=False,
        )
    kwargs = {"num_samples": num_samples} if family == "original_chronos_t5" else {}
    return pipeline.predict_quantiles(
        inputs,
        prediction_length=prediction_length,
        quantile_levels=list(quantile_levels),
        **kwargs,
    )


def chronos2_native_embedding(
    pipeline: Any,
    inputs: Any,
    *,
    batch_size: int = 256,
    context_length: int | None = None,
) -> tuple[Any, Any]:
    """Return Chronos-2's public token embeddings and scaling state unchanged."""
    embeddings, state = pipeline.embed(
        inputs, batch_size=batch_size, context_length=context_length
    )
    if len(embeddings) != len(state):
        raise NativeAdapterError("Chronos-2 embedding/state cardinality mismatch")
    return embeddings, state


def chronos_native_embedding(pipeline: Any, inputs: Any) -> tuple[Any, Any]:
    """Return a concrete Chronos/Chronos-Bolt public embedding without pooling.

    ``BaseChronosPipeline`` is only the loader/abstract surface.  The concrete
    original and Bolt pipelines returned by ``from_pretrained`` both expose a
    documented ``embed`` API.  This adapter deliberately preserves its token axis and
    tokenizer or location/scale state; reducing those tensors is a Track-C decision.
    """
    import torch

    embed = getattr(pipeline, "embed", None)
    if not callable(embed):
        raise NativeAdapterError(
            f"concrete Chronos pipeline {type(pipeline).__name__} has no public embed API"
        )
    embeddings, state = embed(inputs)
    if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 3:
        raise NativeAdapterError(
            f"unexpected Chronos embedding shape: {getattr(embeddings, 'shape', None)}"
        )
    if not torch.isfinite(embeddings).all():
        raise NativeAdapterError("Chronos embedding contains non-finite values")
    batch = int(embeddings.shape[0])
    state_values = state if isinstance(state, tuple) else (state,)
    for index, value in enumerate(state_values):
        if not isinstance(value, torch.Tensor) or value.shape[0] != batch:
            raise NativeAdapterError(
                f"Chronos embedding state[{index}] does not preserve batch cardinality"
            )
        if not torch.isfinite(value).all():
            raise NativeAdapterError(f"Chronos embedding state[{index}] is non-finite")
    return embeddings, state


def timesfm25_transformers_forecast(
    model: Any,
    past_values: Any,
    *,
    prediction_length: int,
    context_length: int | None = None,
) -> tuple[Any, Any]:
    """Match the official TimesFM 2.5 wrapper before optional quantile repair.

    The official parity surface uses native normalization, flip invariance, no positive
    truncation, and raw (possibly crossing) quantiles.  Quantile-crossing repair is an
    explicit downstream post-process and must not be mixed into wrapper parity.
    """
    import torch

    if prediction_length < 1:
        raise NativeAdapterError("prediction_length must be positive")
    tensor = past_values if isinstance(past_values, torch.Tensor) else torch.as_tensor(past_values)
    if tensor.ndim != 2:
        raise NativeAdapterError(f"TimesFM expects [series,time], got {tuple(tensor.shape)}")
    output = model(
        past_values=tensor,
        forecast_context_len=context_length or tensor.shape[-1],
        truncate_negative=False,
        force_flip_invariance=True,
    )
    point = output.mean_predictions[:, :prediction_length]
    quantiles = output.full_predictions[:, :prediction_length]
    if point.shape[:2] != (tensor.shape[0], prediction_length):
        raise NativeAdapterError(f"unexpected TimesFM point shape: {tuple(point.shape)}")
    if not torch.isfinite(point).all() or not torch.isfinite(quantiles).all():
        raise NativeAdapterError("TimesFM forecast contains non-finite values")
    return point, quantiles


def ttm_native_forecast(
    model: Any,
    past_values: Any,
    frequency_token: Any,
) -> Any:
    """Run TTM-R2 on raw values with its native scaler and frequency prefix."""
    import torch

    values = past_values if isinstance(past_values, torch.Tensor) else torch.as_tensor(past_values)
    token = frequency_token if isinstance(frequency_token, torch.Tensor) else torch.as_tensor(frequency_token)
    if values.ndim != 3 or values.shape[1:] != (512, 5):
        raise NativeAdapterError(f"TTM expects [B,512,5], got {tuple(values.shape)}")
    if token.ndim != 1 or token.shape[0] != values.shape[0]:
        raise NativeAdapterError("TTM frequency token must be [B]")
    output = model(past_values=values, freq_token=token, return_loss=False).prediction_outputs
    if output.ndim != 3 or output.shape[0] != values.shape[0] or output.shape[2] != 5:
        raise NativeAdapterError(f"unexpected TTM forecast shape: {tuple(output.shape)}")
    if not torch.isfinite(output).all():
        raise NativeAdapterError("TTM forecast contains non-finite values")
    return output


def moirai2_native_forecast(
    model: Any,
    past_target: Any,
    past_observed_target: Any,
    past_is_pad: Any,
) -> Any:
    """Run Moirai-2 after zero-filling values excluded by the observed mask."""
    import torch

    target = past_target if isinstance(past_target, torch.Tensor) else torch.as_tensor(past_target)
    observed = (
        past_observed_target
        if isinstance(past_observed_target, torch.Tensor)
        else torch.as_tensor(past_observed_target)
    ).bool()
    is_pad = past_is_pad if isinstance(past_is_pad, torch.Tensor) else torch.as_tensor(past_is_pad)
    if target.ndim != 3 or observed.shape != target.shape or is_pad.shape != target.shape[:2]:
        raise NativeAdapterError(
            f"Moirai expects target/observed [B,T,C] and pad [B,T], got "
            f"{tuple(target.shape)}, {tuple(observed.shape)}, {tuple(is_pad.shape)}"
        )
    clean = torch.where(observed, target, torch.zeros((), dtype=target.dtype, device=target.device))
    output = model(
        past_target=clean,
        past_observed_target=observed,
        past_is_pad=is_pad.bool(),
    )
    if not torch.isfinite(output).all():
        raise NativeAdapterError("Moirai-2 forecast contains non-finite values")
    return output


def toto2_native_forecast(
    model: Any,
    target: Any,
    target_mask: Any,
    series_ids: Any,
    *,
    prediction_length: int,
) -> Any:
    """Run Toto 2.0's leaderboard short-horizon path with explicit masks/groups."""
    import torch

    values = target if isinstance(target, torch.Tensor) else torch.as_tensor(target)
    mask = target_mask if isinstance(target_mask, torch.Tensor) else torch.as_tensor(target_mask)
    groups = series_ids if isinstance(series_ids, torch.Tensor) else torch.as_tensor(series_ids)
    mask = mask.bool()
    if values.ndim != 3 or mask.shape != values.shape or groups.shape != values.shape[:2]:
        raise NativeAdapterError(
            f"Toto expects target/mask [B,C,T] and series_ids [B,C], got "
            f"{tuple(values.shape)}, {tuple(mask.shape)}, {tuple(groups.shape)}"
        )
    clean = torch.where(mask, values, torch.zeros((), dtype=values.dtype, device=values.device))
    output = model.forecast(
        {"target": clean, "target_mask": mask, "series_ids": groups.long()},
        horizon=prediction_length,
        decode_block_size=None,
        has_missing_values=not bool(mask.all().item()),
    )
    if output.ndim != 4 or output.shape[1:3] != values.shape[:2]:
        raise NativeAdapterError(f"unexpected Toto forecast shape: {tuple(output.shape)}")
    if not torch.isfinite(output).all():
        raise NativeAdapterError("Toto forecast contains non-finite values")
    return output


def sundial_native_forecast(
    model: Any,
    inputs: Any,
    *,
    prediction_length: int,
    num_samples: int,
    seed: int,
) -> Any:
    """Run Sundial's isolated public generator without requesting hidden states."""
    import torch

    values = inputs if isinstance(inputs, torch.Tensor) else torch.as_tensor(inputs)
    if values.ndim != 2:
        raise NativeAdapterError(f"Sundial expects [series,time], got {tuple(values.shape)}")
    if prediction_length < 1 or num_samples < 1:
        raise NativeAdapterError("prediction_length and num_samples must be positive")
    devices: Iterable[int] = ()
    if values.is_cuda:
        devices = (values.device.index or 0,)
    with torch.random.fork_rng(devices=list(devices)):
        torch.manual_seed(seed)
        if values.is_cuda:
            torch.cuda.manual_seed_all(seed)
        output = model.generate(
            values.float(),
            max_new_tokens=prediction_length,
            num_samples=num_samples,
        )
    if output.shape != (values.shape[0], num_samples, prediction_length):
        raise NativeAdapterError(f"unexpected Sundial forecast shape: {tuple(output.shape)}")
    if not torch.isfinite(output).all():
        raise NativeAdapterError("Sundial forecast contains non-finite values")
    return output


def validate_tabpfn_fold_containment(
    support_time_ns: Any,
    query_time_ns: Any,
    *,
    fold_train_end_ns: int,
) -> None:
    """Fail if TabPFN support rows escape the current training fold or query rows leak back."""
    support = np.asarray(support_time_ns, dtype=np.int64).reshape(-1)
    query = np.asarray(query_time_ns, dtype=np.int64).reshape(-1)
    if support.size == 0 or query.size == 0:
        raise NativeAdapterError("TabPFN support and query sets must be nonempty")
    if int(support.max()) > int(fold_train_end_ns):
        raise NativeAdapterError("TabPFN support set contains rows after the training-fold boundary")
    if int(query.min()) <= int(fold_train_end_ns):
        raise NativeAdapterError("TabPFN query set must begin strictly after the training-fold boundary")
    if int(support.max()) >= int(query.min()):
        raise NativeAdapterError("TabPFN support/query chronology overlaps")
