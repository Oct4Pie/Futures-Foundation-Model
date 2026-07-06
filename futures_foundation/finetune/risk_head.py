"""Distributional forward-R risk head — backbone-AGNOSTIC (works on any frozen embedding).

The accurate replacement for a snapshot peak-R point regressor (which shrinks to the median and
under-predicts the tail). Instead of a point estimate, this predicts the forward-R SURVIVAL CURVE
P(reach >= Xr before the -1R stop) for X in TARGETS, from a frozen-encoder EMBEDDING — whichever
backbone produced it (Mantis, MOMENT, ...). No snapshot features, no custom indicators: the
foundation's learned representation IS the input, and the head deduces "how far will this run"
from it. Each threshold is Platt-calibrated (P is trustworthy, not a magnitude guess); the curve
is enforced monotone. From it we read a data-driven TP (ride/exit) and big-win prob (sizing).

Labels reuse the strategy keys' per-target realized R (already computed in the labeler's build):
a trade reached >= Xr before stop  <=>  realized-at-target-X  > 0. So no relabeling.
"""
import numpy as np

from .calibration import fit_platt, apply_platt

TARGETS = (2.0, 3.0, 4.0, 6.0)


def reach_labels(keys, targets=TARGETS):
    """Per-threshold binary reach label from strategy keys: 1 if the trade reached >= Xr
    before the -1R stop (realized-at-target > 0), else 0. Shape [N, len(targets)]."""
    n_t = len(targets)
    out = np.zeros((len(keys), n_t), np.int8)
    for r, k in enumerate(keys):
        for ti in range(n_t):
            out[r, ti] = 1 if float(k[4 + ti]) > 0.0 else 0
    return out


def monotone_survival(surv):
    """Enforce a valid survival curve: non-increasing across thresholds and in [0,1].
    Independent per-threshold heads can violate this; a real survival function cannot."""
    surv = np.clip(np.asarray(surv, np.float64), 0.0, 1.0)
    return np.minimum.accumulate(surv, axis=-1)


def survival_to_stats(surv, targets=TARGETS, q_tp=0.33):
    """Turn the per-threshold survival curve into decisions.
    Returns dict with:
      surv     : the monotone survival curve [N, T]  (P(reach >= X))
      exp_reach: approx E[peak favorable R] = area under survival (Riemann, base at targets[0])
      p_bigwin : P(reach >= the largest target)  -> the sizing 'press' signal
      tp       : dynamic take-profit = largest X with P(reach>=X) >= q_tp (>= targets[0])
    The TP is the calibrated-probability version of a static peak-R dynamic TP."""
    surv = monotone_survival(surv)
    t = np.asarray(targets, np.float64)
    n, T = surv.shape
    exp_reach = surv[:, 0] * t[0]
    for i in range(T - 1):
        exp_reach = exp_reach + 0.5 * (surv[:, i] + surv[:, i + 1]) * (t[i + 1] - t[i])
    reach_q = surv >= q_tp
    tp = np.full(n, t[0], np.float64)
    for i in range(T):
        tp = np.where(reach_q[:, i], t[i], tp)
    return {'surv': surv, 'exp_reach': exp_reach, 'p_bigwin': surv[:, -1], 'tp': tp}


class RiskHead:
    """Per-threshold Platt-calibrated survival head on a frozen embedding (backbone-agnostic).
    fit(X, keys) reads labels from the keys; predict_survival(X) returns the monotone calibrated
    survival curve. Head type mirrors the signal head (mlp | logistic). Independent heads per
    threshold (simple, each cleanly calibrated); can become a shared-trunk multi-output later."""

    def __init__(self, targets=TARGETS, head='mlp', calibrate=True, **cfg):
        self.targets = tuple(targets)
        self.head = head
        self.calibrate = calibrate
        self.cfg = cfg
        self._heads = []                                # (clf, platt) per threshold

    def _make(self, seed):
        if self.head == 'mlp':
            from sklearn.neural_network import MLPClassifier
            return MLPClassifier(hidden_layer_sizes=tuple(self.cfg.get('hidden', (128,))),
                                 max_iter=int(self.cfg.get('max_iter', 300)),
                                 batch_size=int(self.cfg.get('mlp_batch', 4096)),
                                 alpha=float(self.cfg.get('mlp_alpha', 1e-4)),
                                 early_stopping=True, random_state=seed)
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=int(self.cfg.get('max_iter', 1000)),
                                  C=float(self.cfg.get('C', 1.0)))

    def fit(self, Xtr, keys_tr, Xval=None, keys_val=None, seed=0):
        """Fit one calibrated binary head per threshold. Platt is fit on val (leak-free) when
        given, else raw (no-op-safe)."""
        Ytr = reach_labels(keys_tr, self.targets)
        Yval = reach_labels(keys_val, self.targets) if keys_val is not None else None
        self._heads = []
        for ti in range(len(self.targets)):
            clf = self._make(seed)
            ytr = Ytr[:, ti]
            if len(np.unique(ytr)) < 2:                 # degenerate threshold -> constant head
                self._heads.append((None, float(ytr.mean())))
                continue
            clf.fit(Xtr, ytr)
            platt = None
            if self.calibrate and Xval is not None and Yval is not None:
                raw = clf.predict_proba(Xval)[:, 1]
                platt = fit_platt(raw, Yval[:, ti])
            self._heads.append((clf, platt))
        return self

    def predict_survival(self, X):
        cols = []
        for clf, platt in self._heads:
            if clf is None:                             # constant head (degenerate threshold)
                cols.append(np.full(len(X), float(platt)))
            else:
                raw = clf.predict_proba(X)[:, 1]
                cols.append(apply_platt(raw, platt) if isinstance(platt, tuple) else raw)
        return monotone_survival(np.stack(cols, axis=1))

    def predict_stats(self, X, q_tp=0.33):
        return survival_to_stats(self.predict_survival(X), self.targets, q_tp=q_tp)
