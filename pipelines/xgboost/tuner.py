"""Optuna study (spec sections 6 + 8). TPE sampler, MedianPruner, 300 trials.

Each trial: sample XGB params within the section-6 anti-overfit bounds, fit on
the train-fit slice, build signals on the Optuna validation fold, run the
hybrid-trail backtest -> per-trade returns -> combined objective. Optuna
maximizes the combined score.

xgboost/optuna are imported lazily so the rest of the pipeline (and its
tests) import fine before the owner installs the deps.
"""
import numpy as np
import pandas as pd

from pipelines.common.objective import combined_objective, PERIODS_PER_YEAR
from .backtest import run_backtest

# spec section 6 — tight anti-overfit bounds (do NOT widen)
_BOUNDS = dict(
    max_depth=(3, 6), learning_rate=(0.01, 0.1),
    subsample=(0.6, 0.85), colsample_bytree=(0.7, 1.0),
    reg_lambda=(1, 10), min_child_weight=(5, 50),
    n_estimators=(200, 800),
)
CONF_THRESHOLD = 0.4
_CLASSES = [-1, 0, 1]
_TO_XGB = {-1: 0, 0: 1, 1: 2}            # contiguous for XGBoost
_FROM_XGB = {0: -1, 1: 0, 2: 1}


def _signals_from_proba(proba: np.ndarray, thr: float) -> np.ndarray:
    idx = proba.argmax(axis=1)
    conf = proba[np.arange(len(proba)), idx]
    sig = np.array([_FROM_XGB[i] for i in idx])
    sig[(sig == 0) | (conf < thr)] = 0
    return sig


def _fit_xgb(params, Xf, yf, device: str = 'cpu'):
    import xgboost as xgb
    # device: ADDITIVE (flagged) vs literal spec text — spec-CONSISTENT, since
    # the spec/owner rule is "XGBoost accel is CUDA-only (no Metal)"; 'cuda' on
    # Colab is the intended GPU path. Default 'cpu' => behaviour unchanged.
    m = xgb.XGBClassifier(
        objective="multi:softprob", num_class=3, eval_metric="mlogloss",
        tree_method="hist", device=device, n_jobs=-1, **params)
    m.fit(Xf, np.array([_TO_XGB[v] for v in yf]))
    return m


def _score_val_blocks(model, val_blocks: list, ppy: int) -> float:
    """Score the Optuna objective on PER-TICKER val backtests.

    Pooled training mixes tickers, but a backtest simulates ONE sequential
    position book — it cannot span tickers. So each ticker is backtested on
    its own contiguous val slice (full bar series; model fires only on the
    event rows) and the per-trade returns are concatenated before scoring.
    Single-ticker is just the 1-block case (== prior behaviour)."""
    rets = []
    for b in val_blocks:
        proba = model.predict_proba(b['Xv'])
        ev_sig = _signals_from_proba(proba, CONF_THRESHOLD)
        sig = np.zeros(len(b['ohlcv']), dtype=np.int8)
        sig[b['ev_pos']] = ev_sig                  # signals only at event rows
        rets.append(run_backtest(b['ohlcv'], sig)['returns'])
    if not rets:
        return 0.0
    return combined_objective(pd.concat(rets, ignore_index=True), ppy)


def tune(Xf, yf, val_blocks: list, timeframe: str, n_trials: int = 300,
         seed: int = 42, device: str = 'cpu', n_jobs: int = 1) -> dict:
    """Xf/yf: pooled train-fit EVENT-row features/labels. val_blocks: list of
    {'Xv': event-row features, 'ohlcv': full val bar series, 'ev_pos': int
    positions of the event rows within ohlcv} — one per ticker. Returns the
    best XGB param dict."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    ppy = PERIODS_PER_YEAR[timeframe]

    def objective(trial):
        params = dict(
            max_depth=trial.suggest_int("max_depth", *_BOUNDS["max_depth"]),
            learning_rate=trial.suggest_float("learning_rate",
                                              *_BOUNDS["learning_rate"], log=True),
            subsample=trial.suggest_float("subsample", *_BOUNDS["subsample"]),
            colsample_bytree=trial.suggest_float("colsample_bytree",
                                                 *_BOUNDS["colsample_bytree"]),
            reg_lambda=trial.suggest_float("reg_lambda", *_BOUNDS["reg_lambda"]),
            min_child_weight=trial.suggest_int("min_child_weight",
                                               *_BOUNDS["min_child_weight"]),
            n_estimators=trial.suggest_int("n_estimators",
                                           *_BOUNDS["n_estimators"]),
        )
        model = _fit_xgb(params, Xf, yf, device)
        return _score_val_blocks(model, val_blocks, ppy)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner())
    # n_jobs>1 = Optuna's own (thread) trial parallelism (spec-sanctioned).
    # NOTE: real GIL-free speedup comes from PROCESS-level window
    # parallelism (run_pipeline window_select + a multiprocess driver) —
    # threaded n_jobs mostly overlaps the XGBoost fit, not the Python
    # backtest. Default 1 = unchanged behaviour.
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs,
                   show_progress_bar=False)
    return study.best_params
