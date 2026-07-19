"""Leakage-safe downstream probes for dense futures path targets."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from futures_foundation.finetune.path_labels import (
    BARRIER_ADVERSE_FIRST,
    BARRIER_AMBIGUOUS,
    BARRIER_FAVORABLE_FIRST,
    TREND_CONTINUATION,
    TREND_REVERSAL,
    TREND_TERMINATION,
    TREND_UNDEFINED,
)


@dataclass(frozen=True)
class TargetSpec:
    name: str
    kind: str
    horizon_index: int
    horizon_minutes: int
    target_index: int = -1
    target_r: float = 0.0


def target_specs(arrays: dict[str, np.ndarray]) -> tuple[TargetSpec, ...]:
    horizons = np.asarray(arrays["horizons_minutes"], np.int32)
    output = []
    for horizon_i, horizon in enumerate(horizons):
        suffix = f"{int(horizon)}m"
        output.extend((
            TargetSpec(f"forward_abs_move_r_{suffix}", "reg", horizon_i, int(horizon)),
            TargetSpec(f"forward_realized_vol_{suffix}", "reg", horizon_i, int(horizon)),
            TargetSpec(f"forward_trend_eff_{suffix}", "reg", horizon_i, int(horizon)),
            TargetSpec(f"forward_direction_{suffix}", "bin", horizon_i, int(horizon)),
            TargetSpec(f"trend_continuation_{suffix}", "bin", horizon_i, int(horizon)),
            TargetSpec(f"trend_termination_{suffix}", "bin", horizon_i, int(horizon)),
            TargetSpec(f"trend_reversal_{suffix}", "bin", horizon_i, int(horizon)),
            TargetSpec(f"favorable_mfe_r_{suffix}", "reg", horizon_i, int(horizon)),
            TargetSpec(f"adverse_mae_r_{suffix}", "reg", horizon_i, int(horizon)),
        ))
        for target_i, target_r in enumerate(np.asarray(arrays["targets_r"], np.float32)):
            token = f"{float(target_r):g}r"
            output.extend((
                TargetSpec(
                    f"favorable_mfe_ge_{token}_{suffix}", "bin", horizon_i, int(horizon),
                    target_i, float(target_r),
                ),
                TargetSpec(
                    f"target_before_adverse_{token}_{suffix}", "bin", horizon_i, int(horizon),
                    target_i, float(target_r),
                ),
                TargetSpec(
                    f"adverse_or_ambiguous_{token}_{suffix}", "bin", horizon_i, int(horizon),
                    target_i, float(target_r),
                ),
                TargetSpec(
                    f"time_to_favorable_{token}_{suffix}", "reg", horizon_i, int(horizon),
                    target_i, float(target_r),
                ),
                TargetSpec(
                    f"time_to_adverse_{token}_{suffix}", "reg", horizon_i, int(horizon),
                    target_i, float(target_r),
                ),
                TargetSpec(
                    f"policy_gross_r_{token}_{suffix}", "reg", horizon_i, int(horizon),
                    target_i, float(target_r),
                ),
            ))
    return tuple(output)


def target_values(
    arrays: dict[str, np.ndarray], spec: TargetSpec,
) -> tuple[np.ndarray, np.ndarray]:
    horizon = int(spec.horizon_index)
    context_direction = np.asarray(arrays.get("context_direction", ()), np.int8)
    direction_valid = context_direction != 0 if len(context_direction) else None
    if spec.name.startswith("forward_abs_move_r_"):
        y = np.asarray(arrays["forward_abs_move_r"][:, horizon], np.float32)
        valid = np.isfinite(y)
    elif spec.name.startswith("forward_realized_vol_"):
        y = np.asarray(arrays["forward_realized_vol"][:, horizon], np.float32)
        valid = np.isfinite(y) & (_forward_steps(arrays, spec.horizon_minutes) >= 2)
    elif spec.name.startswith("forward_trend_eff_"):
        y = np.asarray(arrays["forward_trend_eff"][:, horizon], np.float32)
        valid = np.isfinite(y) & (_forward_steps(arrays, spec.horizon_minutes) >= 2)
    elif spec.name.startswith("forward_direction_"):
        raw = np.asarray(arrays["terminal_move_r"][:, horizon], np.float32)
        valid = np.isfinite(raw)
        y = (raw > 0).astype(np.int8)
    elif spec.name.startswith("trend_continuation_"):
        raw = np.asarray(arrays["trend_path_class"][:, horizon], np.int8)
        valid = raw != TREND_UNDEFINED
        y = (raw == TREND_CONTINUATION).astype(np.int8)
    elif spec.name.startswith("trend_termination_"):
        raw = np.asarray(arrays["trend_path_class"][:, horizon], np.int8)
        valid = raw != TREND_UNDEFINED
        y = (raw == TREND_TERMINATION).astype(np.int8)
    elif spec.name.startswith("trend_reversal_"):
        raw = np.asarray(arrays["trend_path_class"][:, horizon], np.int8)
        valid = raw != TREND_UNDEFINED
        y = (raw == TREND_REVERSAL).astype(np.int8)
    elif spec.name.startswith(("favorable_mfe_r_", "adverse_mae_r_", "favorable_mfe_ge_")):
        favorable, adverse, valid = _direction_relative_excursions(arrays, horizon)
        if spec.name.startswith("favorable_mfe_r_"):
            y = favorable
        elif spec.name.startswith("adverse_mae_r_"):
            y = adverse
        else:
            y = (favorable >= float(spec.target_r)).astype(np.int8)
    elif spec.name.startswith((
        "target_before_adverse_", "adverse_or_ambiguous_", "time_to_favorable_",
        "time_to_adverse_", "policy_gross_r_",
    )):
        if direction_valid is None:
            raise ValueError("direction-relative target requires context_direction")
        direction_index = _context_direction_indices(arrays)
        rows = np.arange(len(context_direction), dtype=np.int64)
        target = int(spec.target_index)
        state = np.asarray(arrays["barrier_state"])[rows, horizon, direction_index, target]
        valid = direction_valid & (state >= 0)
        if spec.name.startswith("target_before_adverse_"):
            y = (state == BARRIER_FAVORABLE_FIRST).astype(np.int8)
        elif spec.name.startswith("adverse_or_ambiguous_"):
            y = np.isin(state, (BARRIER_ADVERSE_FIRST, BARRIER_AMBIGUOUS)).astype(np.int8)
        elif spec.name.startswith("time_to_favorable_"):
            raw = np.asarray(arrays["time_to_favorable_minutes"])[
                rows, horizon, direction_index, target
            ]
            valid &= raw >= 0
            y = (raw / float(spec.horizon_minutes)).astype(np.float32)
        elif spec.name.startswith("time_to_adverse_"):
            raw = np.asarray(arrays["time_to_adverse_minutes"])[
                rows, horizon, direction_index, target
            ]
            valid &= raw >= 0
            y = (raw / float(spec.horizon_minutes)).astype(np.float32)
        else:
            y = np.asarray(arrays["policy_r_gross"])[
                rows, horizon, direction_index, target
            ].astype(np.float32)
            valid &= np.isfinite(y)
    else:
        raise KeyError(spec.name)
    return y, valid


def _context_direction_indices(arrays: dict[str, np.ndarray]) -> np.ndarray:
    context = np.asarray(arrays["context_direction"], np.int8)
    directions = np.asarray(arrays["directions"], np.int8)
    if directions.shape != (2,) or set(directions.tolist()) != {-1, 1}:
        raise ValueError("direction axis must contain exactly long and short")
    output = np.zeros(len(context), np.int8)
    for index, direction in enumerate(directions):
        output[context == direction] = index
    return output


def _direction_relative_excursions(
    arrays: dict[str, np.ndarray], horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    direction = np.asarray(arrays["context_direction"], np.int8)
    upside = np.asarray(arrays["upside_mfe_r"][:, horizon], np.float32)
    downside = np.asarray(arrays["downside_mae_r"][:, horizon], np.float32)
    favorable = np.where(direction > 0, upside, downside).astype(np.float32)
    adverse = np.where(direction > 0, downside, upside).astype(np.float32)
    valid = (direction != 0) & np.isfinite(favorable) & np.isfinite(adverse)
    return favorable, adverse, valid


def _forward_steps(arrays: dict[str, np.ndarray], horizon_minutes: int) -> np.ndarray:
    """Elapsed-horizon bar count; dispersion/path-efficiency need at least two returns."""
    timeframe = np.asarray(arrays["timeframe"])
    minutes = np.asarray([int(str(value)[:-3]) for value in timeframe], np.int32)
    if np.any(int(horizon_minutes) % minutes):
        raise ValueError("horizon is not divisible by every row timeframe")
    return int(horizon_minutes) // minutes


def causal_feature_matrix(
    arrays: dict[str, np.ndarray], rows: np.ndarray,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Context features plus known-at-decision ticker identity."""
    rows = np.asarray(rows, np.int64)
    base = np.asarray(arrays["features"][rows], np.float32)
    tickers = np.asarray(arrays["ticker"])
    ticker_names = tuple(str(value) for value in np.unique(tickers))
    one_hot = np.column_stack([(tickers[rows] == ticker).astype(np.float32) for ticker in ticker_names])
    names = tuple(str(value) for value in arrays["feature_names"])
    names += tuple(f"ticker_{ticker}" for ticker in ticker_names)
    output = np.column_stack((base, one_hot)).astype(np.float32, copy=False)
    if not np.isfinite(output).all():
        raise ValueError("causal feature matrix contains non-finite values")
    return output, names


def fold_target_issue(
    y: np.ndarray,
    train_rows: np.ndarray,
    test_rows: np.ndarray,
    kind: str,
    *,
    min_train: int = 20,
    min_test: int = 5,
) -> str | None:
    """Return an explicit reason when a target cannot be scored honestly in one fold."""
    train_rows, test_rows = np.asarray(train_rows, np.int64), np.asarray(test_rows, np.int64)
    if len(train_rows) < min_train or len(test_rows) < min_test:
        return "insufficient_fold_rows"
    if kind == "bin":
        if len(np.unique(np.asarray(y)[train_rows])) < 2:
            return "single_class_train"
        if len(np.unique(np.asarray(y)[test_rows])) < 2:
            return "single_class_test"
    return None


def _permuted_within_groups(
    values: np.ndarray,
    groups: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    output = np.asarray(values).copy()
    groups = np.asarray(groups)
    for group in np.unique(groups):
        rows = np.flatnonzero(groups == group)
        output[rows] = output[rows[rng.permutation(len(rows))]]
    return output


def _fit_predict_linear(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    kind: str,
    sample_weight: np.ndarray | None,
) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X_train)
    train = scaler.transform(X_train)
    test = scaler.transform(X_test)
    if kind == "reg":
        model = Ridge(alpha=1.0, solver="lsqr", tol=1e-6)
        model.fit(train, y_train, sample_weight=sample_weight)
        return np.asarray(model.predict(test), np.float32)
    if len(np.unique(y_train)) < 2:
        raise ValueError("binary training fold has only one class")
    model = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
    model.fit(train, y_train, sample_weight=sample_weight)
    return np.asarray(model.predict_proba(test)[:, 1], np.float32)


def _fit_predict_xgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    kind: str,
    sample_weight: np.ndarray | None,
    seed: int,
) -> np.ndarray:
    import xgboost as xgb

    common = dict(
        n_estimators=120, max_depth=3, learning_rate=0.04, subsample=0.8,
        colsample_bytree=0.8, reg_lambda=10.0, min_child_weight=20.0,
        tree_method="hist", random_state=int(seed), n_jobs=1, verbosity=0,
    )
    if kind == "reg":
        model = xgb.XGBRegressor(objective="reg:squarederror", **common)
        model.fit(X_train, y_train, sample_weight=sample_weight)
        return np.asarray(model.predict(X_test), np.float32)
    if len(np.unique(y_train)) < 2:
        raise ValueError("binary training fold has only one class")
    model = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", **common)
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return np.asarray(model.predict_proba(X_test)[:, 1], np.float32)


def fit_predict_fold(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    train_rows: np.ndarray,
    test_rows: np.ndarray,
    *,
    kind: str,
    head: str,
    control: str = "real",
    sample_weight: np.ndarray | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Fit on earlier rows and predict one later fold under a declared negative control."""
    X = np.asarray(X, np.float32)
    y = np.asarray(y)
    groups = np.asarray(groups)
    train_rows, test_rows = np.asarray(train_rows, np.int64), np.asarray(test_rows, np.int64)
    rng = np.random.default_rng(int(seed))
    X_train, X_test = X[train_rows], X[test_rows]
    y_train = y[train_rows].copy()
    if control == "shuffled_label":
        y_train = _permuted_within_groups(y_train, groups[train_rows], rng)
    elif control == "random_feature":
        X_train = rng.standard_normal(X_train.shape, dtype=np.float32)
        X_test = rng.standard_normal(X_test.shape, dtype=np.float32)
    elif control == "time_destroyed":
        X_train = _permuted_within_groups(X_train, groups[train_rows], rng)
        X_test = _permuted_within_groups(X_test, groups[test_rows], rng)
    elif control != "real":
        raise ValueError(f"unknown control: {control}")

    weights = None if sample_weight is None else np.asarray(sample_weight, np.float32)[train_rows]
    if head == "linear":
        return _fit_predict_linear(
            X_train, y_train, X_test, kind=kind, sample_weight=weights,
        )
    if head == "xgb":
        return _fit_predict_xgb(
            X_train, y_train, X_test, kind=kind, sample_weight=weights, seed=seed,
        )
    raise ValueError(f"unknown head: {head}")


def prediction_metrics(y_true: np.ndarray, prediction: np.ndarray, kind: str) -> dict[str, float]:
    y_true = np.asarray(y_true)
    prediction = np.asarray(prediction)
    if len(y_true) == 0 or len(y_true) != len(prediction) or not np.isfinite(prediction).all():
        raise ValueError("predictions must be finite, non-empty, and aligned")
    if kind == "reg":
        from scipy.stats import spearmanr
        from sklearn.metrics import mean_absolute_error, r2_score
        if np.std(y_true) == 0 or np.std(prediction) == 0:
            spearman = 0.0
        else:
            statistic = float(spearmanr(y_true, prediction).statistic)
            spearman = statistic if np.isfinite(statistic) else 0.0
        return {
            "r2": float(r2_score(y_true, prediction)),
            "mae": float(mean_absolute_error(y_true, prediction)),
            "spearman": spearman,
        }
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    if len(np.unique(y_true)) < 2:
        raise ValueError("binary test fold has only one class")
    return {
        "auc": float(roc_auc_score(y_true, prediction)),
        "pr_auc": float(average_precision_score(y_true, prediction)),
        "brier": float(brier_score_loss(y_true, prediction)),
        "prevalence": float(np.mean(y_true)),
    }


def paired_calendar_block_bootstrap(
    y_true: np.ndarray,
    prediction_a: np.ndarray,
    prediction_b: np.ndarray,
    decision_time_ns: np.ndarray,
    *,
    kind: str,
    block_days: int = 7,
    repetitions: int = 500,
    seed: int = 0,
) -> dict[str, float | int]:
    """Paired metric delta with calendar blocks as the resampling unit."""
    y = np.asarray(y_true)
    a = np.asarray(prediction_a)
    b = np.asarray(prediction_b)
    time = np.asarray(decision_time_ns, np.int64)
    if not (len(y) == len(a) == len(b) == len(time)) or len(y) == 0:
        raise ValueError("paired bootstrap arrays must be non-empty and aligned")
    block_days, repetitions = int(block_days), int(repetitions)
    if block_days < 1 or repetitions < 1:
        raise ValueError("block_days and repetitions must be positive")
    if kind == "reg":
        from sklearn.metrics import r2_score
        score = r2_score
        metric = "r2"
    else:
        from sklearn.metrics import roc_auc_score
        score = roc_auc_score
        metric = "auc"
    point = float(score(y, a) - score(y, b))
    block_ns = block_days * 24 * 60 * 60 * 1_000_000_000
    block_id = time // block_ns
    unique_blocks = np.unique(block_id)
    if len(unique_blocks) < 2:
        raise ValueError("paired bootstrap requires at least two calendar blocks")
    members = [np.flatnonzero(block_id == block) for block in unique_blocks]
    rng = np.random.default_rng(int(seed))
    deltas = []
    for _ in range(repetitions):
        sampled = rng.integers(0, len(members), size=len(members))
        rows = np.concatenate([members[index] for index in sampled])
        if kind == "bin" and len(np.unique(y[rows])) < 2:
            continue
        delta = float(score(y[rows], a[rows]) - score(y[rows], b[rows]))
        deltas.append(delta)
    if len(deltas) < max(20, repetitions // 2):
        raise ValueError("too few valid paired bootstrap repetitions")
    values = np.asarray(deltas, np.float64)
    return {
        "metric": metric,
        "delta": float(point),
        "ci_low_95": float(np.quantile(values, 0.025)),
        "ci_high_95": float(np.quantile(values, 0.975)),
        "positive_probability": float(np.mean(values > 0)),
        "calendar_blocks": int(len(unique_blocks)),
        "block_days": block_days,
        "valid_repetitions": int(len(values)),
    }


__all__ = [
    "TargetSpec", "target_specs", "target_values", "causal_feature_matrix",
    "fold_target_issue", "fit_predict_fold", "prediction_metrics",
    "paired_calendar_block_bootstrap",
]
