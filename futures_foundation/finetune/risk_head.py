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

TARGETS = (2.0, 3.0, 4.0, 6.0, 8.0)      # reach ladder (8R = max trend); matches the pivot FIXED_TARGETS


def _start_heartbeat(msg, every=60):
    """Start a daemon liveness heartbeat -> returns a stop() callable. The ladder's rung fits are
    silent sklearn solvers on millions of rows; without this the REAL fit reads as HUNG. Prints
    elapsed every `every`s; a fit that finishes under `every`s prints nothing (no WF-fold spam)."""
    import threading
    import time
    t0 = time.time()
    ev = threading.Event()

    def _beat():
        while not ev.wait(every):
            print(f"    [risk-head] {msg} ... {time.time() - t0:,.0f}s elapsed (alive)", flush=True)

    th = threading.Thread(target=_beat, daemon=True)
    th.start()

    def stop():
        ev.set()
        th.join(timeout=1)
    return stop


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

    def _rung_jobs(self, n_active):
        """How many rungs to fit CONCURRENTLY. The rungs are independent -> embarrassingly parallel;
        default = fit all of them at once, capped by cores. rung_jobs=1 forces the old sequential
        path. Thread-parallel (shared read-only Xtr, no copy) — process-parallel would duplicate the
        multi-GB embedding per rung and OOM at produce scale."""
        import os as _os
        want = self.cfg.get('rung_jobs')
        if want is not None:
            return max(1, min(int(want), n_active))
        return max(1, min(n_active, (_os.cpu_count() or 2)))

    def fit(self, Xtr, keys_tr, Xval=None, keys_val=None, seed=0):
        """Fit one calibrated binary head per threshold, the rungs IN PARALLEL (independent fits).
        Platt is fit on val (leak-free) when given, else raw (no-op-safe)."""
        import os as _os
        from concurrent.futures import ThreadPoolExecutor
        Ytr = reach_labels(keys_tr, self.targets)
        Yval = reach_labels(keys_val, self.targets) if keys_val is not None else None
        T = len(self.targets)
        active = [ti for ti in range(T) if len(np.unique(Ytr[:, ti])) >= 2]
        jobs = self._rung_jobs(len(active)) if active else 1
        # cap BLAS threads PER rung so `jobs` concurrent fits don't oversubscribe the cores (each
        # sklearn fit is itself BLAS-threaded; jobs * per_rung ~= cpu_count). threadpoolctl if present.
        per_rung = max(1, (_os.cpu_count() or 2) // max(1, jobs))
        try:
            from threadpoolctl import threadpool_limits
        except Exception:                                # pragma: no cover - optional dep
            from contextlib import nullcontext
            def threadpool_limits(limits=None):          # no-op fallback
                return nullcontext()

        def _fit_one(ti):
            ytr = Ytr[:, ti]
            clf = self._make(seed)
            with threadpool_limits(limits=(per_rung if jobs > 1 else None)):
                clf.fit(Xtr, ytr)
            platt = None
            if self.calibrate and Xval is not None and Yval is not None:
                raw = clf.predict_proba(Xval)[:, 1]
                platt = fit_platt(raw, Yval[:, ti])
            return ti, (clf, platt)

        heads = [None] * T
        for ti in range(T):                              # degenerate thresholds -> constant heads
            if ti not in active:
                heads[ti] = (None, float(Ytr[:, ti].mean()))
        if active:
            stop = _start_heartbeat(f"fitting {len(active)} rungs (x{jobs} parallel, {per_rung} "
                                    f"thr/rung) on {len(Xtr):,}x{Xtr.shape[1]}")
            try:
                if jobs == 1:
                    done = [_fit_one(ti) for ti in active]
                else:
                    with ThreadPoolExecutor(max_workers=jobs) as ex:
                        done = list(ex.map(_fit_one, active))
            finally:
                stop()
            for ti, hd in done:
                heads[ti] = hd
        self._heads = heads
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
