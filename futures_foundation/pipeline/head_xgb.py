"""Generic XGBoost classifier head — the trained component.

Strategy-agnostic: consumes a feature matrix X (frozen foundation embedding,
optionally fused with the strategy's own features) and integer labels;
emits a class per row plus class probabilities (the conviction signal a
downstream gate / account layer consumes). Deterministic given `seed`
(single-threaded, fixed random_state). ONNX-convertible (skl2onnx) — see
the deployment memory; nothing here blocks the single-file export.
"""
import numpy as np


class XGBHead:
    """Head with fit(X,y,seed) / predict(X) / predict_proba(X).
    Conforms to the head-agnostic contract evaluate.py speaks."""

    def __init__(self, n_classes, n_estimators=200, max_depth=4,
                 learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                 **extra):
        # **extra forwards any additional XGBClassifier args (e.g. reg_lambda,
        # min_child_weight) — used by the head-tuner. Default {} = unchanged.
        self.n_classes = int(n_classes)
        self._p = dict(n_estimators=n_estimators, max_depth=max_depth,
                       learning_rate=learning_rate, subsample=subsample,
                       colsample_bytree=colsample_bytree, **extra)
        self._clf = None
        # Platt calibration of P(take): (A, B) s.t. cal = sigmoid(A·logit(p)+B).
        # None = uncalibrated (raw XGB proba). Binary heads only. Set by
        # fit_calibration(); applied transparently in predict_proba() and
        # baked into the ONNX export so the bot reads calibrated proba.
        self._platt = None

    def fit(self, X, y, seed=0, sample_weight=None):
        # Let XGBClassifier infer the objective from y: binary:logistic for
        # 2 classes, multi:softprob for >2. Forcing multi+num_class breaks
        # the binary case (predict returns 2-D).
        # sample_weight (optional): per-row training weight. Used for
        # R-WEIGHTED ranking — weigh each trade by the R it would realize so
        # the loss cares about profit magnitude, not just hit/miss. None ->
        # uniform (the win-rate objective).
        import xgboost as xgb
        self._clf = xgb.XGBClassifier(
            tree_method='hist', random_state=seed, n_jobs=1,
            verbosity=0, **self._p)
        sw = None if sample_weight is None else np.asarray(sample_weight, np.float32)
        self._clf.fit(np.asarray(X, np.float32), np.asarray(y), sample_weight=sw)
        return self

    def fit_calibration(self, X, y, n_splits=3, seed=0):
        """Fit Platt scaling on P(take) using OUT-OF-FOLD predictions so the
        calibrator never sees a row the predicting model trained on (no leak).
        Monotonic — it rescales the proba so it tracks the empirical hit rate;
        it does NOT change ranking/AUC. Binary heads only; no-op otherwise.

        NOTE for deployment: this changes the proba SCALE (calibrated ≈ P(win)).
        Any downstream sizing/threshold tuned on the raw proba must be
        re-tuned to the calibrated scale."""
        if self.n_classes != 2:
            return self
        import xgboost as xgb
        from sklearn.model_selection import cross_val_predict, StratifiedKFold
        from sklearn.linear_model import LogisticRegression
        X = np.asarray(X, np.float32); y = np.asarray(y).astype(int)
        base = xgb.XGBClassifier(tree_method='hist', random_state=seed,
                                 n_jobs=1, verbosity=0, **self._p)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                              random_state=seed)
        oof = cross_val_predict(base, X, y, cv=skf,
                                method='predict_proba', n_jobs=1)[:, 1]
        eps = 1e-6
        p = np.clip(oof, eps, 1 - eps)
        z = np.log(p / (1 - p)).reshape(-1, 1)        # logit of OOF P(take)
        lr = LogisticRegression(C=1e6, solver='lbfgs')  # ~unregularized Platt
        lr.fit(z, y)
        self._platt = (float(lr.coef_[0, 0]), float(lr.intercept_[0]))
        return self

    @staticmethod
    def _apply_platt(p1, platt):
        """cal = sigmoid(A·logit(p1)+B), vectorized, clip-safe."""
        A, B = platt
        eps = 1e-6
        p = np.clip(np.asarray(p1, np.float64), eps, 1 - eps)
        z = np.log(p / (1 - p))
        return 1.0 / (1.0 + np.exp(-(A * z + B)))

    def feature_importances(self):
        """Gain-weighted importances [feat_dim] of the fitted head (sums to ~1).
        Pair with `group_importance` for the chronos/return-shape/handcraft split."""
        return np.asarray(self._clf.feature_importances_, np.float32)

    def predict(self, X):
        return self._clf.predict(np.asarray(X, np.float32)).astype(int)

    def predict_proba(self, X):
        p = self._clf.predict_proba(np.asarray(X, np.float32))
        # getattr: bundles pickled before calibration existed have no _platt;
        # they must keep returning RAW proba (no break, no silent calibration).
        platt = getattr(self, '_platt', None)
        if platt is not None and self.n_classes == 2:
            cal1 = self._apply_platt(p[:, 1], platt)
            p = np.column_stack([1.0 - cal1, cal1]).astype(np.float32)
        return p


class XGBRiskHead:
    """Regression head — predicts max_rr_realized per signal (the peak R
    reached before stop / vertical barrier). Used at inference to set
    dynamic TP per trade (TP = clip(0.8 * R_hat, 1.5, 8.0)).

    Calibration design choices (learned from a broken first version):
      * Target is **log1p-transformed** before training. max_rr_realized is
        heavily right-tailed (most ~1-3R with a long tail to 10R+). Trees
        with Huber loss on raw R shrank predictions toward the median and
        under-predicted by ~3R systematically. log1p tames the tail,
        expm1 inverts at inference.
      * Larger n_estimators + deeper trees vs the signal head — capturing
        the right tail needs more capacity than binary classification.
      * Still pseudohubererror (twice-differentiable Huber) — robust to the
        few extreme outliers that remain after log-transform.
    """

    def __init__(self, n_estimators=500, max_depth=5, learning_rate=0.05,
                 subsample=0.8, colsample_bytree=0.8):
        self._p = dict(n_estimators=n_estimators, max_depth=max_depth,
                       learning_rate=learning_rate, subsample=subsample,
                       colsample_bytree=colsample_bytree)
        self._reg = None

    def fit(self, X, y_max_rr, seed=0):
        import xgboost as xgb
        y = np.asarray(y_max_rr, np.float32)
        y_log = np.log1p(np.clip(y, 0.0, None))           # tame right-tail
        self._reg = xgb.XGBRegressor(
            objective='reg:pseudohubererror', huber_slope=1.0,
            tree_method='hist', random_state=seed, n_jobs=1,
            verbosity=0, **self._p)
        self._reg.fit(np.asarray(X, np.float32), y_log)
        return self

    def predict(self, X):
        # Invert log1p, clip to plausible R range.
        log_pred = self._reg.predict(np.asarray(X, np.float32))
        return np.clip(np.expm1(log_pred).astype(np.float32), 0.0, 15.0)
