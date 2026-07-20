"""Low-variance incremental-information ruler for native route outputs.

The ruler uses one frozen common-information row selection and purged fold contract.
Model tensors are compressed by a train-fold-fitted, target-independent 32-component
PCA bottleneck.  The primary arms are causal-only, model-only, causal-plus-model, and
residual-over-causal.  Model-only negative controls use shuffled labels, random
features, and time-destroyed features while causal inputs remain unchanged elsewhere.

This screen can fund a capacity-controlled nonlinear sensitivity.  It cannot grant
route promotion, full training, OOS access, deployment, or trading.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


PRIMARY_TARGETS = (
    "forward_realized_vol_180m",
    "favorable_mfe_r_180m",
    "adverse_mae_r_180m",
    "target_before_adverse_2r_180m",
    "policy_gross_r_2r_180m",
)
SCREEN_POLICY = {
    "policy_id": "ffm_native_incremental_screen_v1",
    "primary_targets": list(PRIMARY_TARGETS),
    "pca_components": 32,
    "min_model_control_wins": 3,
    "min_incremental_point_wins": 3,
    "min_residual_bonferroni_ci_wins": 1,
    "max_primary_point_degradations": 1,
    "bootstrap_confidence": 0.99,
    "bootstrap_repetitions": 500,
    "bootstrap_block_days": 7,
    "promotion_admitted": False,
}


@dataclass(frozen=True)
class LinearHead:
    scaler: Any
    model: Any
    kind: str

    def predict(self, values: np.ndarray) -> np.ndarray:
        transformed = self.scaler.transform(np.asarray(values, np.float32))
        if self.kind == "reg":
            return np.asarray(self.model.predict(transformed), np.float32)
        return np.asarray(self.model.predict_proba(transformed)[:, 1], np.float32)


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


def fit_linear_head(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    kind: str,
    sample_weight: np.ndarray | None,
) -> LinearHead:
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.preprocessing import StandardScaler

    X_train = np.asarray(X_train, np.float32)
    y_train = np.asarray(y_train)
    scaler = StandardScaler().fit(X_train)
    transformed = scaler.transform(X_train)
    if kind == "reg":
        model = Ridge(alpha=1.0, solver="lsqr", tol=1e-6)
        model.fit(transformed, y_train, sample_weight=sample_weight)
    else:
        if len(np.unique(y_train)) < 2:
            raise ValueError("binary training fold has only one class")
        model = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        model.fit(transformed, y_train, sample_weight=sample_weight)
    return LinearHead(scaler=scaler, model=model, kind=kind)


def fit_model_bottleneck(
    features: np.ndarray,
    groups: np.ndarray,
    train_rows: np.ndarray,
    test_rows: np.ndarray,
    *,
    control: str,
    components: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit one unsupervised train-only bottleneck under a model-feature control."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    values = np.asarray(features, np.float32)
    groups = np.asarray(groups)
    train_rows = np.asarray(train_rows, np.int64)
    test_rows = np.asarray(test_rows, np.int64)
    train = values[train_rows].copy()
    test = values[test_rows].copy()
    rng = np.random.default_rng(int(seed))
    if control == "random_feature":
        train = rng.standard_normal(train.shape, dtype=np.float32)
        test = rng.standard_normal(test.shape, dtype=np.float32)
    elif control == "time_destroyed":
        train = _permuted_within_groups(train, groups[train_rows], rng)
        test = _permuted_within_groups(test, groups[test_rows], rng)
    elif control != "real":
        raise ValueError(f"unsupported model-feature bottleneck control: {control}")
    scaler = StandardScaler().fit(train)
    train_scaled = scaler.transform(train)
    test_scaled = scaler.transform(test)
    count = min(int(components), train_scaled.shape[1], len(train_scaled) - 1)
    if count < 1:
        raise ValueError("model bottleneck has no admissible components")
    pca = PCA(n_components=count, svd_solver="randomized", random_state=int(seed))
    train_output = np.asarray(pca.fit_transform(train_scaled), np.float32)
    test_output = np.asarray(pca.transform(test_scaled), np.float32)
    if not np.isfinite(train_output).all() or not np.isfinite(test_output).all():
        raise ValueError("model bottleneck produced non-finite features")
    return train_output, test_output, {
        "control": control,
        "components": int(count),
        "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        "train_rows": int(len(train_rows)),
        "test_rows": int(len(test_rows)),
        "seed": int(seed),
    }


def fit_fold_arms(
    causal_features: np.ndarray,
    model_train: np.ndarray,
    model_test: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    train_rows: np.ndarray,
    test_rows: np.ndarray,
    *,
    kind: str,
    sample_weight: np.ndarray | None,
    seed: int,
    random_train: np.ndarray,
    random_test: np.ndarray,
    destroyed_train: np.ndarray,
    destroyed_test: np.ndarray,
) -> dict[str, np.ndarray]:
    """Fit all linear primary/control arms for one target/fold."""
    causal = np.asarray(causal_features, np.float32)
    y = np.asarray(y)
    groups = np.asarray(groups)
    train_rows = np.asarray(train_rows, np.int64)
    test_rows = np.asarray(test_rows, np.int64)
    y_train = y[train_rows]
    weights = (
        None
        if sample_weight is None
        else np.asarray(sample_weight, np.float32)[train_rows]
    )
    causal_train, causal_test = causal[train_rows], causal[test_rows]

    causal_head = fit_linear_head(
        causal_train, y_train, kind=kind, sample_weight=weights,
    )
    causal_train_prediction = causal_head.predict(causal_train)
    causal_test_prediction = causal_head.predict(causal_test)

    model_head = fit_linear_head(
        model_train, y_train, kind=kind, sample_weight=weights,
    )
    model_prediction = model_head.predict(model_test)

    combined_head = fit_linear_head(
        np.column_stack((causal_train, model_train)),
        y_train,
        kind=kind,
        sample_weight=weights,
    )
    combined_prediction = combined_head.predict(
        np.column_stack((causal_test, model_test))
    )

    if kind == "reg":
        residual_target = y_train - causal_train_prediction
    else:
        residual_target = y_train.astype(np.float64) - causal_train_prediction
    residual_head = fit_linear_head(
        model_train,
        residual_target,
        kind="reg",
        sample_weight=weights,
    )
    residual_correction = residual_head.predict(model_test)
    residual_prediction = causal_test_prediction + residual_correction
    if kind == "bin":
        residual_prediction = np.clip(residual_prediction, 1e-6, 1.0 - 1e-6)
    residual_prediction = np.asarray(residual_prediction, np.float32)

    rng = np.random.default_rng(int(seed))
    shuffled_y = _permuted_within_groups(y_train, groups[train_rows], rng)
    shuffled_head = fit_linear_head(
        model_train, shuffled_y, kind=kind, sample_weight=weights,
    )
    random_head = fit_linear_head(
        random_train, y_train, kind=kind, sample_weight=weights,
    )
    destroyed_head = fit_linear_head(
        destroyed_train, y_train, kind=kind, sample_weight=weights,
    )
    return {
        "causal": causal_test_prediction,
        "model": model_prediction,
        "causal_plus_model": combined_prediction,
        "residual_over_causal": residual_prediction,
        "model_shuffled_label": shuffled_head.predict(model_test),
        "model_random_feature": random_head.predict(random_test),
        "model_time_destroyed": destroyed_head.predict(destroyed_test),
    }


def root_calendar_block_bootstrap(
    y_true: np.ndarray,
    prediction_a: np.ndarray,
    prediction_b: np.ndarray,
    decision_time_ns: np.ndarray,
    roots: np.ndarray,
    *,
    kind: str,
    block_days: int,
    repetitions: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    """Paired metric delta resampled by root × calendar block."""
    from sklearn.metrics import r2_score, roc_auc_score

    y = np.asarray(y_true)
    a = np.asarray(prediction_a)
    b = np.asarray(prediction_b)
    time = np.asarray(decision_time_ns, np.int64)
    root = np.asarray(roots).astype(str)
    if not (len(y) == len(a) == len(b) == len(time) == len(root)) or len(y) == 0:
        raise ValueError("root-calendar bootstrap arrays must be non-empty and aligned")
    if not 0.5 < confidence < 1.0:
        raise ValueError("bootstrap confidence must lie in (0.5,1)")
    score = r2_score if kind == "reg" else roc_auc_score
    metric = "r2" if kind == "reg" else "auc"
    point = float(score(y, a) - score(y, b))
    block_ns = int(block_days) * 24 * 60 * 60 * 1_000_000_000
    block_id = time // block_ns
    keys = np.asarray([f"{value}:{block}" for value, block in zip(root, block_id)])
    unique = np.unique(keys)
    members = [np.flatnonzero(keys == value) for value in unique]
    if len(members) < 2:
        raise ValueError("root-calendar bootstrap requires at least two blocks")
    rng = np.random.default_rng(int(seed))
    deltas = []
    for _ in range(int(repetitions)):
        sampled = rng.integers(0, len(members), size=len(members))
        rows = np.concatenate([members[index] for index in sampled])
        if kind == "bin" and len(np.unique(y[rows])) < 2:
            continue
        deltas.append(float(score(y[rows], a[rows]) - score(y[rows], b[rows])))
    if len(deltas) < max(50, int(repetitions) // 2):
        raise ValueError("too few valid root-calendar bootstrap repetitions")
    values = np.asarray(deltas, np.float64)
    alpha = 1.0 - float(confidence)
    suffix = str(int(round(confidence * 100)))
    return {
        "metric": metric,
        "delta": point,
        f"ci_low_{suffix}": float(np.quantile(values, alpha / 2)),
        f"ci_high_{suffix}": float(np.quantile(values, 1.0 - alpha / 2)),
        "positive_probability": float(np.mean(values > 0)),
        "root_calendar_blocks": int(len(members)),
        "block_days": int(block_days),
        "valid_repetitions": int(len(values)),
        "confidence": float(confidence),
    }


def screen_verdict(primary_rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Apply the predeclared funding rule to primary target summaries."""
    by_target = {str(row["target"]): row for row in primary_rows}
    if set(by_target) != set(PRIMARY_TARGETS):
        raise ValueError("screen verdict requires the exact primary target closure")
    model_control_wins = 0
    incremental_wins = 0
    residual_ci_wins = 0
    degradations = 0
    target_rows = []
    for target in PRIMARY_TARGETS:
        row = by_target[target]
        score_name = "r2" if row["kind"] == "reg" else "auc"
        scores = row["metrics"]
        model_score = float(scores["model"][score_name])
        controls = max(
            float(scores[name][score_name])
            for name in (
                "model_shuffled_label", "model_random_feature", "model_time_destroyed",
            )
        )
        control_win = model_score > controls
        combined_delta = float(row["bootstrap"]["causal_plus_model"]["delta"])
        residual_delta = float(row["bootstrap"]["residual_over_causal"]["delta"])
        incremental = max(combined_delta, residual_delta) > 0
        degradation = max(combined_delta, residual_delta) < 0
        ci_key = "ci_low_99"
        raw_residual_ci = row["bootstrap"]["residual_over_causal"].get(ci_key)
        residual_ci = None if raw_residual_ci is None else float(raw_residual_ci)
        ci_win = residual_ci is not None and residual_ci > 0
        model_control_wins += int(control_win)
        incremental_wins += int(incremental)
        residual_ci_wins += int(ci_win)
        degradations += int(degradation)
        target_rows.append({
            "target": target,
            "model_control_win": bool(control_win),
            "combined_delta": combined_delta,
            "residual_delta": residual_delta,
            "residual_ci_low_99": residual_ci,
            "incremental_point_win": bool(incremental),
            "residual_bonferroni_ci_win": bool(ci_win),
            "point_degradation": bool(degradation),
        })
    survived = (
        model_control_wins >= SCREEN_POLICY["min_model_control_wins"]
        and incremental_wins >= SCREEN_POLICY["min_incremental_point_wins"]
        and residual_ci_wins >= SCREEN_POLICY["min_residual_bonferroni_ci_wins"]
        and degradations <= SCREEN_POLICY["max_primary_point_degradations"]
    )
    return {
        "policy": dict(SCREEN_POLICY),
        "model_control_wins": model_control_wins,
        "incremental_point_wins": incremental_wins,
        "residual_bonferroni_ci_wins": residual_ci_wins,
        "primary_point_degradations": degradations,
        "target_rows": target_rows,
        "downstream_screen_survived": bool(survived),
        "nonlinear_sensitivity_funded": bool(survived),
        "promotion_admitted": False,
        "full_training_admitted": False,
    }


__all__ = [
    "PRIMARY_TARGETS", "SCREEN_POLICY", "fit_fold_arms", "fit_linear_head",
    "fit_model_bottleneck", "root_calendar_block_bootstrap", "screen_verdict",
]
