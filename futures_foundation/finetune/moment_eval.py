"""Torch-free preprocessing and targets for the frozen MOMENT benchmark."""
from __future__ import annotations

import numpy as np


def left_pad_contexts(context, native_length=512):
    """Convert ``[N,T,C]`` causal contexts to MOMENT's ``[N,C,512]`` contract.

    MOMENT's official classification datasets left-pad short histories.  The mask ensures
    padding is excluded from both RevIN statistics and the final patch pooling operation.
    """
    context = np.asarray(context, np.float32)
    if context.ndim != 3:
        raise ValueError("context must have shape [N,T,C]")
    native_length = int(native_length)
    if context.shape[1] > native_length:
        raise ValueError(
            f"context length {context.shape[1]} exceeds MOMENT native length {native_length}"
        )
    if not np.isfinite(context).all():
        raise ValueError("context contains non-finite values")
    out = np.zeros((len(context), context.shape[2], native_length), np.float32)
    mask = np.zeros((len(context), native_length), np.int64)
    length = context.shape[1]
    out[:, :, -length:] = np.transpose(context, (0, 2, 1))
    mask[:, -length:] = 1
    return out, mask


def pool_channel_patches(embeddings, input_mask, patch_len=8):
    """Mean-pool valid patches per channel, then concatenate channels in source order."""
    embeddings = np.asarray(embeddings, np.float32)
    input_mask = np.asarray(input_mask)
    if embeddings.ndim != 4:
        raise ValueError("embeddings must have shape [N,C,P,D]")
    if input_mask.ndim != 2 or len(input_mask) != len(embeddings):
        raise ValueError("input_mask must have shape [N,T]")
    patch_len = int(patch_len)
    if patch_len < 1 or input_mask.shape[1] % patch_len:
        raise ValueError("mask length must be divisible by patch_len")
    patch_mask = input_mask.reshape(len(input_mask), -1, patch_len).all(axis=2)
    if patch_mask.shape[1] != embeddings.shape[2]:
        raise ValueError("patch count does not match embedding output")
    denom = patch_mask.sum(axis=1, keepdims=True)
    if np.any(denom == 0):
        raise ValueError("each row must contain at least one complete valid patch")
    pooled = np.sum(embeddings * patch_mask[:, None, :, None], axis=2)
    pooled /= denom[:, :, None]
    return pooled.reshape(len(pooled), -1).astype(np.float32, copy=False)


def targets_from_context_future(context, future):
    """Compute the same six causal probe targets used by the Mantis SSL validator."""
    context = np.asarray(context, np.float64)
    future = np.asarray(future, np.float64)
    if context.ndim != 3 or future.ndim != 3 or context.shape[2] < 4 or future.shape[2] < 4:
        raise ValueError("context/future must have shape [N,T,C>=4]")
    if len(context) != len(future) or context.shape[1] < 4 or future.shape[1] < 1:
        raise ValueError("context/future rows must align and contain sufficient history")
    high, low, close = context[:, :, 1], context[:, :, 2], context[:, :, 3]
    logret = np.diff(np.log(np.clip(close, 1e-9, None)), axis=1)
    net = logret.sum(axis=1)
    half = context.shape[1] // 2
    first_range = high[:, :half].max(1) - low[:, :half].min(1)
    second_range = high[:, half:].max(1) - low[:, half:].min(1)
    fwd_ret = (
        np.log(np.clip(future[:, -1, 3], 1e-9, None))
        - np.log(np.clip(context[:, -1, 3], 1e-9, None))
    )
    return {
        "vol": logret.std(1).astype(np.float32),
        "trend_eff": (np.abs(net) / (np.abs(logret).sum(1) + 1e-9)).astype(np.float32),
        "range_expand": np.log(
            (second_range + 1e-9) / (first_range + 1e-9)
        ).astype(np.float32),
        "fwd_absmove": np.abs(fwd_ret).astype(np.float32),
        "direction": (net > 0).astype(np.int32),
        "fwd_dir": (fwd_ret > 0).astype(np.int32),
    }
