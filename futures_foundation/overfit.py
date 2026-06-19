"""Reusable overfit-detection & generalization primitives — metric-agnostic.

ONE home for the "default → check generalization → tune only if overfit →
auto-fall-back" decision math, so every training path reuses it instead of
reinventing the wheel:

  - strategy SELECTION heads  (chronos.evaluate / tune_head / train_loop) —
    metric = realized meanR
  - foundation CONTEXT heads  (context training) —
    metric = Pearson r (regression) / ROC-AUC (classification)

Every supported metric is "higher = better", so the same comparisons apply;
callers pass metric-appropriate tolerances (a 0.30R meanR gap and a 0.05 AUC
gap are both "this overfit"). These functions are pure (numpy only), so they
are unit-tested directly and behave identically wherever they are used.
"""
import numpy as np


def overfit_trigger(train_metric, val_metric, gap_tol):
    """True if the model memorized TRAIN relative to VALIDATION — train exceeds
    val by more than `gap_tol`. The signal to remediate (regularize / tune)."""
    return (train_metric - val_metric) > gap_tol


def generalizes(val_metric, test_metric, gap_tol):
    """True if the VAL→TEST decay is within tolerance (`val - test <= gap_tol`),
    i.e. the result holds on data never used for fitting OR selection. A wide
    positive gap = an edge that looked good on validation but decayed on test
    = fake/fragile. Missing val or test → False (can't confirm generalization)."""
    if val_metric is None or test_metric is None:
        return False
    return (val_metric - test_metric) <= gap_tol


def gen_score(per_fold, std_penalty=0.5):
    """Generalization-robust score for tuning objectives: cross-fold
    `mean - std_penalty * std`. A config that spikes on a few folds and
    collapses on others (the overfit signature) is penalized; a consistent
    config wins. Empty input → -inf (worst possible)."""
    if per_fold is None or len(per_fold) == 0:
        return float('-inf')
    a = np.asarray(per_fold, dtype=float)
    mean = float(a.mean())
    std = float(a.std()) if len(a) > 1 else 0.0
    return mean - std_penalty * std


def best_config(default_val_metric, candidates, accept_margin=0.0):
    """Pick the candidate config with the best VALIDATION metric — but only if it
    beats the default by at least `accept_margin`. Otherwise return None =
    AUTO-FALL-BACK to the default (no candidate generalizes better). Selection
    sees validation only; test is never consulted.

    candidates: iterable of (config, val_metric).
    Returns the winning config object, or None to keep the default.
    """
    floor = default_val_metric + accept_margin
    best_m, best_cfg = floor, None
    for cfg, m in candidates:
        if m is not None and m > best_m:
            best_m, best_cfg = m, cfg
    return best_cfg
