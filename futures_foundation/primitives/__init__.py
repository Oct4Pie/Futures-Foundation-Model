"""
futures_foundation.primitives
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Shared numpy signal-detection building blocks for both Colab training scripts
and algoTraderAI inference modules.

    from futures_foundation.primitives import compute_supertrend, apply_rr_barriers
"""

from .indicators import (
    compute_atr,
    compute_supertrend,
    compute_supertrend_htf,
    compute_ema,
    compute_rsi,
)
from .rolling import (
    rolling_mean,
    rolling_atr_percentile,
)
from .detection import (
    detect_pivots,
    detect_cisd_signals,
    compute_ote_zones,
)
from .barriers import (
    apply_rr_barriers,
    best_rr_hit,
    realized_r_trailing,
    r_bucket,
)
from .session import (
    session_mask,
    session_end_mask,
    compute_vwap,
)
from .windows import (
    aggregate_ohlcv_window,
    multi_scale_ohlcv_window,
)

__all__ = [
    "compute_atr", "compute_supertrend", "compute_supertrend_htf",
    "compute_ema", "compute_rsi",
    "rolling_mean", "rolling_atr_percentile",
    "detect_pivots", "detect_cisd_signals", "compute_ote_zones",
    "apply_rr_barriers", "best_rr_hit", "realized_r_trailing", "r_bucket",
    "session_mask", "session_end_mask", "compute_vwap",
    "aggregate_ohlcv_window", "multi_scale_ohlcv_window",
]
