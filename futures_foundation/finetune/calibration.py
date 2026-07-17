"""Generic probability calibration — backbone-agnostic (no Mantis/foundation coupling).

Platt scaling used by any head (signal, risk, or a future MOMENT head): rescale a raw
out-of-sample proba so P is a trustworthy confidence, WITHOUT changing ranking/AUC.
"""
import numpy as np


def fit_platt(p1, y):
    """Platt scaling: fit (A,B) so cal = sigmoid(A·logit(p1)+B) tracks the empirical hit rate.
    p1 = OUT-OF-SAMPLE proba (a val set the head never trained on -> leak-free), y = its labels.
    MONOTONIC — rescales the proba to ≈P(win) WITHOUT changing ranking/AUC, so P=0.5 means a true
    ~50% signal and the proba is a trustworthy confidence ACROSS tiers. Returns (A, B); needs both
    classes present (else None)."""
    from sklearn.linear_model import LogisticRegression
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return None
    eps = 1e-6
    p = np.clip(np.asarray(p1, np.float64), eps, 1 - eps)
    z = np.log(p / (1 - p)).reshape(-1, 1)              # logit of OOS proba
    lr = LogisticRegression(C=1e6, solver='lbfgs').fit(z, y)   # ~unregularized Platt
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def apply_platt(p1, platt):
    """cal = sigmoid(A·logit(p1)+B), vectorized + clip-safe. platt=None -> raw (identity no-op)."""
    if platt is None:
        return np.asarray(p1, np.float64)
    A, B = platt
    eps = 1e-6
    p = np.clip(np.asarray(p1, np.float64), eps, 1 - eps)
    z = np.log(p / (1 - p))
    return 1.0 / (1.0 + np.exp(-(A * z + B)))


def fit_isotonic_expected_value(score, realized):
    """Fit a portable monotonic raw-score -> expected-realized-value mapping.

    ``score`` must be genuinely out-of-fold.  This helper deliberately accepts no feature matrix
    or estimator, which keeps the calibration boundary explicit and testable.  The returned knots
    are plain arrays suitable for a checkpoint/report rather than a pickled sklearn object.
    """
    from sklearn.isotonic import IsotonicRegression

    score = np.asarray(score, np.float64)
    realized = np.asarray(realized, np.float64)
    if (
        score.ndim != 1 or realized.ndim != 1 or len(score) != len(realized)
        or len(score) < 2 or not np.isfinite(score).all() or not np.isfinite(realized).all()
    ):
        raise ValueError("isotonic calibration inputs must be finite aligned vectors")
    if len(np.unique(score)) < 2:
        raise ValueError("isotonic calibration requires at least two distinct scores")
    model = IsotonicRegression(increasing=True, out_of_bounds="clip")
    model.fit(score, realized)
    x = np.asarray(model.X_thresholds_, np.float64)
    y = np.asarray(model.y_thresholds_, np.float64)
    if len(x) < 2 or len(x) != len(y) or not np.all(np.diff(x) > 0):
        raise RuntimeError("invalid isotonic calibration knots")
    return {"method": "isotonic_expected_net_r_v1", "x": x, "y": y}


def apply_isotonic_expected_value(score, calibration):
    """Apply the portable clipped isotonic expected-value mapping."""
    score = np.asarray(score, np.float64)
    if not np.isfinite(score).all() or calibration.get("method") != "isotonic_expected_net_r_v1":
        raise ValueError("invalid score or isotonic calibration contract")
    x = np.asarray(calibration["x"], np.float64)
    y = np.asarray(calibration["y"], np.float64)
    if len(x) < 2 or len(x) != len(y) or not np.all(np.diff(x) > 0):
        raise ValueError("invalid isotonic calibration knots")
    return np.interp(score, x, y, left=y[0], right=y[-1])
