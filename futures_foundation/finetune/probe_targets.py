"""One causal, negative-price-safe definition of shared representation-probe targets."""
from __future__ import annotations

import numpy as np


TARGET_SEMANTICS_VERSION = "ffm_causal_probe_targets_v2"


def causal_probe_targets(context, future):
    """Compute six probe targets from context and strictly subsequent future bars.

    Price changes are normalized by a robust range scale computed only from the context.  This
    supports contracts trading through zero and prevents model-family scorers from redefining the
    shared ruler.
    """
    context = np.asarray(context, np.float64)
    future = np.asarray(future, np.float64)
    if context.ndim != 3 or future.ndim != 3 or context.shape[2] < 4 or future.shape[2] < 4:
        raise ValueError("context/future must have shape [N,T,C>=4]")
    if len(context) != len(future) or context.shape[1] < 4 or future.shape[1] < 1:
        raise ValueError("context/future rows must align and contain sufficient history")
    if not np.isfinite(context[:, :, :4]).all() or not np.isfinite(future[:, :, :4]).all():
        raise ValueError("probe target OHLC values must be finite")
    o, high, low, close = (context[:, :, index] for index in range(4))
    if np.any((high < np.maximum(o, close)) | (low > np.minimum(o, close)) | (high < low)):
        raise ValueError("context contains invalid OHLC geometry")
    prev = np.concatenate((close[:, :1], close[:, :-1]), axis=1)
    true_range = np.maximum(high - low, np.maximum(np.abs(high - prev), np.abs(low - prev)))
    scale = np.median(true_range, axis=1)
    if np.any(~np.isfinite(scale) | (scale <= 0)):
        raise ValueError("context has no positive causal range scale")
    price_change = np.diff(close, axis=1)
    normalized_change = price_change / scale[:, None]
    net = (close[:, -1] - close[:, 0]) / scale
    half = context.shape[1] // 2
    first_range = high[:, :half].max(1) - low[:, :half].min(1)
    second_range = high[:, half:].max(1) - low[:, half:].min(1)
    fwd_move = (future[:, -1, 3] - close[:, -1]) / scale
    return {
        "vol": normalized_change.std(1).astype(np.float32),
        "trend_eff": (
            np.abs(net) / (np.abs(normalized_change).sum(1) + 1e-9)
        ).astype(np.float32),
        "range_expand": np.log(
            (second_range + 1e-9) / (first_range + 1e-9)
        ).astype(np.float32),
        "fwd_absmove": np.abs(fwd_move).astype(np.float32),
        "direction": (net > 0).astype(np.int32),
        "fwd_dir": (fwd_move > 0).astype(np.int32),
    }


__all__ = ["TARGET_SEMANTICS_VERSION", "causal_probe_targets"]
