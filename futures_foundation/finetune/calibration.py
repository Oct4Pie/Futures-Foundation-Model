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
