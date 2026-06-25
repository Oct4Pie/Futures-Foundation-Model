"""Window-level feature transforms — companions to the frozen embedding.

These operate on the SAME log-price context window that is fed to the foundation
backbone (e.g. Chronos), and produce a small vector of scalars to CONCATENATE with
the embedding before the downstream head (XGBoost).

Motivation (UniShape lesson): foundation backbones that instance-normalize their
input (Chronos-Bolt) strip absolute level/scale and are direction-poor — their
embedding captures volatility/shape but not the sign/asymmetry of returns. The
differenced series (log-returns) carries that direction information. Empirically a
handful of cheap return-shape scalars beat a second (return) embedding pass through
the same instance-normalizing backbone — so we add the scalars, not another embed.

Pure numpy, instrument-agnostic, no torch / no pandas — safe to import anywhere.
Input convention: W is [n, L] of log-prices (n windows, L bars each, causal:
bars <= decision time). All outputs are float32 and finite (constant windows ->
zeros, never NaN/inf).
"""
from typing import List

import numpy as np

__all__ = [
    "RETURN_SHAPE_DIM",
    "log_return_window",
    "return_shape_features",
    "return_shape_feature_names",
    "subwindow_musig",
    "subwindow_musig_names",
]

RETURN_SHAPE_DIM = 7   # width of return_shape_features (mean,std,skew,signrun,acf1-3)


def log_return_window(W: np.ndarray) -> np.ndarray:
    """[n, L] log-prices -> [n, L] log-returns, zero-prepended (first bar = 0)."""
    W = np.asarray(W, np.float32)
    R = np.zeros_like(W)
    R[:, 1:] = np.diff(W, axis=1)
    return R


def return_shape_features(W: np.ndarray) -> np.ndarray:
    """[n, L] log-price windows -> [n, 7] return-shape scalars.

    Columns (see return_shape_feature_names): mean return, return std, return skew,
    sign-run (|mean sign| = directional persistence), and lag-1/2/3 autocorrelation.
    These encode drift, dispersion, asymmetry and momentum-vs-mean-reversion shape —
    the direction-bearing structure an instance-normalizing backbone discards.
    """
    W = np.asarray(W, np.float32)
    R = np.diff(W, axis=1)                              # [n, L-1] log-returns
    if R.shape[1] < 4:                                  # too short for lag-3 autocorr
        raise ValueError(f"need >=5 bars per window, got L={W.shape[1]}")
    mean = R.mean(1)
    std = R.std(1)
    safe = std + 1e-9
    skew = (((R - R.mean(1, keepdims=True)) / safe[:, None]) ** 3).mean(1)
    signrun = np.abs(np.sign(R).mean(1))               # directional persistence

    def autocorr(k: int) -> np.ndarray:
        a, b = R[:, :-k], R[:, k:]
        am, bm = a.mean(1, keepdims=True), b.mean(1, keepdims=True)
        num = ((a - am) * (b - bm)).mean(1)
        den = a.std(1) * b.std(1) + 1e-9
        return num / den

    feats = np.column_stack([mean, std, skew, signrun,
                             autocorr(1), autocorr(2), autocorr(3)])
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def return_shape_feature_names() -> List[str]:
    return ["ret_mean", "ret_std", "ret_skew", "ret_signrun",
            "ret_acf1", "ret_acf2", "ret_acf3"]


def subwindow_musig(W: np.ndarray, sub: int = 16) -> np.ndarray:
    """[n, L] log-prices -> [n, 2*ceil(L/sub)] per-sub-window mean & std.

    Re-injects the local level/scale that instance-norm strips (UniShape's μ/σ
    re-embedding, in the cheap scalar form a tree head can use directly).
    """
    W = np.asarray(W, np.float32)
    n, L = W.shape
    cols = []
    for s in range(0, L, sub):
        w = W[:, s:s + sub]
        cols.append(w.mean(1)); cols.append(w.std(1))
    return np.nan_to_num(np.column_stack(cols), nan=0.0, posinf=0.0,
                         neginf=0.0).astype(np.float32)


def subwindow_musig_names(L: int, sub: int = 16) -> List[str]:
    names = []
    for s in range(0, L, sub):
        names += [f"mu_{s}_{s + sub}", f"sg_{s}_{s + sub}"]
    return names
