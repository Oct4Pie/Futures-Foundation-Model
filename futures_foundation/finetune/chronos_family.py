"""Pinned Chronos-family admission candidates and comparable forecast metrics.

This module is intentionally torch-free.  Model loading and inference live in the
benchmark script so the data/metric contract can be tested without downloading a
foundation model.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np
import pandas as pd

from futures_foundation.finetune.native_contracts import get_arm, get_dossier


QUANTILE_LEVELS = (0.1, 0.5, 0.9)


@dataclass(frozen=True)
class ChronosCandidate:
    key: str
    family: str
    model_id: str
    revision: str
    parameters: int
    public_embedding_api: bool
    native_multivariate: bool

    def manifest(self) -> dict:
        return asdict(self)


# Identity pins come from the single native-contract registry.  The booleans below are
# API-shape descriptors only; they do not imply admission.  Track R remains blocked until
# exact public-API parity is recorded in a current admission report.
def _candidate(key, *, family, public_embedding_api, native_multivariate):
    arm = get_arm(key)
    dossier = get_dossier(key)
    return ChronosCandidate(
        key=key,
        family=family,
        model_id=arm.model_id,
        revision=arm.model_revision,
        parameters=int(dossier["parameters"]),
        public_embedding_api=public_embedding_api,
        native_multivariate=native_multivariate,
    )


CANDIDATES = {
    "chronos_v1": _candidate(
        "chronos_v1", family="original_chronos_t5",
        public_embedding_api=False, native_multivariate=False,
    ),
    "chronos_bolt": _candidate(
        "chronos_bolt", family="chronos_bolt",
        public_embedding_api=True, native_multivariate=False,
    ),
    "chronos_v2": _candidate(
        "chronos_v2", family="chronos_2",
        public_embedding_api=True, native_multivariate=True,
    ),
}


def resolve_candidates(keys) -> tuple[ChronosCandidate, ...]:
    keys = tuple(str(key).strip() for key in keys if str(key).strip())
    unknown = sorted(set(keys) - set(CANDIDATES))
    if unknown:
        raise ValueError(f"unknown Chronos candidates: {unknown}")
    if len(set(keys)) != len(keys):
        raise ValueError("Chronos candidates must be unique")
    return tuple(CANDIDATES[key] for key in keys)


def benchmark_signature(config: dict) -> str:
    raw = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _r2(actual, predicted):
    from sklearn.metrics import r2_score
    return float(r2_score(actual, predicted)) if len(actual) >= 2 else None


def _auc(actual_positive, score):
    from sklearn.metrics import roc_auc_score
    if len(np.unique(actual_positive)) != 2:
        return None
    return float(roc_auc_score(actual_positive, score))


def _corr(actual, predicted, *, rank=False):
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    if rank:
        actual = pd.Series(actual).rank(method="average").to_numpy()
        predicted = pd.Series(predicted).rank(method="average").to_numpy()
    if len(actual) < 2 or np.std(actual) == 0 or np.std(predicted) == 0:
        return None
    return float(np.corrcoef(actual, predicted)[0, 1])


def _trend_eff(path):
    step = np.diff(np.log(np.clip(path, 1e-12, None)), axis=1)
    return np.abs(step.sum(1)) / (np.abs(step).sum(1) + 1e-12)


def _pinball(actual, predicted, level):
    error = actual - predicted
    return float(np.mean(np.maximum(level * error, (level - 1.0) * error)))


def _close_metrics(context_close, future_close, quantiles, levels):
    context_close = np.asarray(context_close, np.float64)
    future_close = np.asarray(future_close, np.float64)
    quantiles = np.asarray(quantiles, np.float64)
    levels = tuple(float(level) for level in levels)
    if context_close.ndim != 2 or future_close.ndim != 2:
        raise ValueError("close contexts and futures must have shape [N,T]")
    if quantiles.shape != (*future_close.shape, len(levels)):
        raise ValueError(
            f"quantiles must have shape {(*future_close.shape, len(levels))}, "
            f"got {quantiles.shape}"
        )
    if 0.5 not in levels:
        raise ValueError("the common point forecast requires the 0.5 quantile")
    if not np.isfinite(quantiles).all():
        raise ValueError("forecast contains non-finite values")

    positive = quantiles > 0
    safe_quantiles = np.clip(quantiles, 1e-12, None)
    base = np.clip(context_close[:, -1], 1e-12, None)
    actual = np.clip(future_close, 1e-12, None)
    median = safe_quantiles[:, :, levels.index(0.5)]
    actual_path = np.log(actual / base[:, None])
    pred_path = np.log(median / base[:, None])
    quantile_paths = np.log(safe_quantiles / base[:, None, None])

    path_mse = float(np.mean((pred_path - actual_path) ** 2))
    persistence_mse = float(np.mean(actual_path ** 2))
    terminal, pred_terminal = actual_path[:, -1], pred_path[:, -1]
    actual_series = np.concatenate([base[:, None], actual], axis=1)
    predicted_series = np.concatenate([base[:, None], median], axis=1)
    actual_rv = np.sum(np.diff(np.log(actual_series), axis=1) ** 2, axis=1)
    predicted_rv = np.sum(np.diff(np.log(predicted_series), axis=1) ** 2, axis=1)
    calibration = {
        str(level): {
            "pinball_log_return": _pinball(actual_path, quantile_paths[:, :, idx], level),
            "empirical_coverage": float(np.mean(actual_path <= quantile_paths[:, :, idx])),
        }
        for idx, level in enumerate(levels)
    }
    return {
        "n": int(len(future_close)),
        "path_log_return_mse": path_mse,
        "persistence_path_mse": persistence_mse,
        "path_skill_vs_persistence": (
            1.0 - path_mse / persistence_mse if persistence_mse > 0 else None
        ),
        "terminal_return_ic": _corr(terminal, pred_terminal),
        "terminal_return_rank_ic": _corr(terminal, pred_terminal, rank=True),
        "fwd_dir_auc": _auc(terminal > 0, pred_terminal),
        "fwd_dir_accuracy": float(np.mean((terminal > 0) == (pred_terminal > 0))),
        "fwd_absmove_r2": _r2(np.abs(terminal), np.abs(pred_terminal)),
        "vol_r2": _r2(actual_rv, predicted_rv),
        "vol_mae": float(np.mean(np.abs(actual_rv - predicted_rv))),
        "trend_eff_r2": _r2(_trend_eff(actual_series), _trend_eff(predicted_series)),
        "positive_forecast_fraction": float(np.mean(positive)),
        "calibration": calibration,
    }


def evaluate_close_forecasts(windows, quantiles, levels=QUANTILE_LEVELS):
    """Evaluate one common close-only forecast track overall and by stream."""
    context_close = np.asarray(windows["context"])[:, :, 3]
    future_close = np.asarray(windows["future"])[:, :, 3]
    quantiles = np.asarray(quantiles, np.float32)
    overall = _close_metrics(context_close, future_close, quantiles, levels)
    labels = np.char.add(
        np.char.add(np.asarray(windows["ticker"]).astype(str), "@"),
        np.asarray(windows["timeframe"]).astype(str),
    )
    per_stream = {}
    for label in sorted(np.unique(labels)):
        rows = labels == label
        per_stream[str(label)] = _close_metrics(
            context_close[rows], future_close[rows], quantiles[rows], levels,
        )
    macro = {}
    for key in overall:
        if key in {"n", "calibration"}:
            continue
        values = [metrics[key] for metrics in per_stream.values() if metrics[key] is not None]
        macro[key] = float(np.mean(values)) if values else None
    macro["stream_count"] = len(per_stream)
    macro["calibration"] = {
        str(level): {
            name: float(np.mean([
                metrics["calibration"][str(level)][name] for metrics in per_stream.values()
            ]))
            for name in ("pinball_log_return", "empirical_coverage")
        }
        for level in levels
    }
    gate = {
        "finite_positive_forecasts": overall["positive_forecast_fraction"] == 1.0,
        "macro_path_skill_positive": macro["path_skill_vs_persistence"] > 0.0,
        "macro_direction_noninferior": (
            macro["fwd_dir_auc"] is not None and macro["fwd_dir_auc"] >= 0.5
        ),
        "macro_move_r2_nonnegative": (
            macro["fwd_absmove_r2"] is not None and macro["fwd_absmove_r2"] >= 0.0
        ),
    }
    gate["all_pass"] = all(gate.values())
    return {
        "overall": overall,
        "macro_stream": macro,
        "per_stream": per_stream,
        "diagnostic_gate": gate,
    }


def persistence_quantiles(windows, levels=QUANTILE_LEVELS):
    base = np.asarray(windows["context"], np.float32)[:, -1, 3]
    horizon = np.asarray(windows["future"]).shape[1]
    return np.broadcast_to(base[:, None, None], (len(base), horizon, len(levels))).copy()
