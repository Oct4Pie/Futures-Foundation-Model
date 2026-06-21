"""Walk-forward, overfit-driven evaluation for the foundation CONTEXT heads.

Validated the SAME WAY as the strategy selection heads, reusing the shared
machinery instead of reinventing it:

  - futures_foundation.pipeline.data.walk_forward_folds  → rolling train/val/test
    windows across ALL data (not a single fixed 2023+ split) — so each head is
    judged in EVERY regime, the analog of the strategies' per-OOS-month gate.
  - futures_foundation.overfit.regularized_fit          → the auto-regularize
    wheel (fit default → re-fit ladder if it overfit train→val → keep best-on-val).
  - futures_foundation.overfit.generalizes              → the VAL→TEST gate.

Metrics are the context heads' own: Pearson r (regression) / ROC-AUC
(classification). Deployment stays frozen-at-cutoff; this is the validation.

`head_verdict` (one fold) and `aggregate_verdict` (across folds) are the pure,
unit-tested decision cores; `run_context_eval` orchestrates the walk-forward.
"""
import numpy as np
import pandas as pd

from futures_foundation import overfit as _of
from futures_foundation.context import HEAD_SPECS, GATE_REG_PEARSON, GATE_CLF_AUC
from futures_foundation.pipeline.data import walk_forward_folds

# Per-metric tolerances (Pearson r and ROC-AUC live on different scales).
REG_OVERFIT_GAP = 0.10        # train→val Pearson-r gap that means "memorized"
CLF_OVERFIT_GAP = 0.05        # train→val AUC gap
REG_GEN_GAP = 0.10            # val→test Pearson-r decay still trusted
CLF_GEN_GAP = 0.05            # val→test AUC decay still trusted
MIN_GENERALIZE_FRAC = 0.60    # head must generalize in >= this fraction of folds

# Regularization ladder (more regularized rungs), analogous to evaluate.REG_LADDER.
CONTEXT_REG_LADDER = [
    dict(max_depth=4, min_child_weight=10, reg_lambda=3.0, subsample=0.8),
    dict(max_depth=3, min_child_weight=30, reg_lambda=8.0, subsample=0.7,
         n_estimators=300),
    dict(max_depth=2, min_child_weight=60, reg_lambda=15.0, subsample=0.6,
         n_estimators=250),
]


def _tols(kind):
    """(overfit_gap, gen_gap, skill_floor) for a head kind."""
    if kind == 'reg':
        return REG_OVERFIT_GAP, REG_GEN_GAP, GATE_REG_PEARSON
    return CLF_OVERFIT_GAP, CLF_GEN_GAP, GATE_CLF_AUC


def head_verdict(kind, train_m, val_m, test_m, trivial_m=None, shuffle_m=None):
    """One-fold verdict via the shared overfit library. ACCURATE = clears the
    skill floor on TEST, GENERALIZES (val→test within tolerance), and beats both
    the SHUFFLE control and the TRIVIAL baseline on TEST."""
    of_gap, gen_gap, floor = _tols(kind)
    generalizes = _of.generalizes(val_m, test_m, gen_gap)
    overfit = _of.overfit_trigger(train_m, val_m, of_gap)
    has_skill = test_m is not None and test_m > floor
    beats_trivial = (trivial_m is None) or (test_m is not None
                                            and test_m > trivial_m)
    beats_shuffle = (shuffle_m is None) or (test_m is not None
                                            and test_m > shuffle_m)
    accurate = bool(has_skill and generalizes and beats_trivial and beats_shuffle)
    return dict(kind=kind, train=train_m, val=val_m, test=test_m,
                trivial=trivial_m, shuffle=shuffle_m, floor=floor,
                generalizes=generalizes, overfit=overfit, has_skill=has_skill,
                beats_trivial=beats_trivial, beats_shuffle=beats_shuffle,
                accurate=accurate)


def aggregate_verdict(kind, fold_verdicts, min_generalize_frac=MIN_GENERALIZE_FRAC):
    """Roll per-fold verdicts into a head verdict. ACCURATE across the walk-
    forward = mean TEST clears the floor, the head GENERALIZES in >=
    min_generalize_frac of folds, and mean TEST beats the trivial + shuffle
    controls. (Pure — unit-tested.)"""
    floor = _tols(kind)[2]
    n = len(fold_verdicts)
    if n == 0:
        return dict(kind=kind, n_folds=0, accurate=False)
    mt = float(np.mean([v['test'] for v in fold_verdicts]))
    mv = float(np.mean([v['val'] for v in fold_verdicts]))
    gen_frac = float(np.mean([1.0 if v['generalizes'] else 0.0
                              for v in fold_verdicts]))
    trivs = [v['trivial'] for v in fold_verdicts if v['trivial'] is not None]
    shufs = [v['shuffle'] for v in fold_verdicts if v['shuffle'] is not None]
    m_triv = float(np.mean(trivs)) if trivs else None
    m_shuf = float(np.mean(shufs)) if shufs else None
    beats_trivial = (m_triv is None) or (mt > m_triv)
    beats_shuffle = (m_shuf is None) or (mt > m_shuf)
    has_skill = mt > floor
    accurate = bool(has_skill and gen_frac >= min_generalize_frac
                    and beats_trivial and beats_shuffle)
    return dict(kind=kind, n_folds=n, mean_test=mt, mean_val=mv,
                gen_frac=gen_frac, mean_trivial=m_triv, mean_shuffle=m_shuf,
                beats_trivial=beats_trivial, beats_shuffle=beats_shuffle,
                has_skill=has_skill, accurate=accurate, folds=fold_verdicts)


def _fit(kind, X, y, seed, **params):
    import xgboost as xgb
    common = dict(n_estimators=400, max_depth=5, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, tree_method='hist',
                  random_state=seed, n_jobs=0)
    common.update(params)
    if kind == 'reg':
        return xgb.XGBRegressor(objective='reg:squarederror', **common).fit(X, y)
    return xgb.XGBClassifier(objective='binary:logistic',
                             eval_metric='logloss', **common).fit(X, y)


def _score(kind, model, X, y):
    if kind == 'reg':
        p = model.predict(X)
        if p.std() == 0 or y.std() == 0:
            return 0.0
        return float(np.corrcoef(p, y)[0, 1])
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y)) < 2:
        return 0.5
    return float(roc_auc_score(y, model.predict_proba(X)[:, 1]))


class WalkForwardCtxProvider:
    """Per-fold context heads for NESTED walk-forward ctx fusion (used by
    evaluate.run(ctx_provider=…)). Holds the dense ctx dataset; `fit` retrains
    the heads on one fold's TRAIN window (label-purged before val), `transform`
    emits ctx_* for query bars. Same overfit-safe discipline as the strategy
    folds — the heads never see the fold's val/test.

    X_ctx: [N, 324] dense [emb|ff68]. labels: DataFrame (head columns).
    ts / ts_end: decision-bar timestamp / its forward-label close time.
    """

    def __init__(self, X_ctx, labels, ts, ts_end, seed=0, specs=HEAD_SPECS):
        self.X = np.asarray(X_ctx, np.float32)
        self.labels = labels
        self.ts = pd.DatetimeIndex(ts)
        self.ts_end = pd.DatetimeIndex(ts_end)
        self.seed = seed
        self.specs = specs
        self._models = {}

    def fit(self, tr_lo, tr_hi, val0):
        mask = (np.asarray(self.ts >= tr_lo) & np.asarray(self.ts <= tr_hi)
                & np.asarray(self.ts_end < val0))       # leak purge before val
        idx = np.flatnonzero(mask)
        self._models = {}
        for name, kind in self.specs:
            if name not in self.labels.columns:
                continue
            y = self.labels[name].to_numpy(np.float32)
            ok = idx[np.isfinite(y[idx])]
            if len(ok) < 100:
                continue
            self._models[name] = (kind, _fit(kind, self.X[ok], y[ok], self.seed))

    def transform(self, E, ff68):
        """ctx_* for query bars, from [E | ff68] (one column per head, in
        specs order; 0.0 for any head that did not fit this fold)."""
        Xq = np.hstack([np.asarray(E, np.float32),
                        np.asarray(ff68, np.float32)])
        cols = []
        for name, kind in self.specs:
            m = self._models.get(name)
            if m is None:
                cols.append(np.zeros(len(Xq), np.float32))
            elif m[0] == 'reg':
                cols.append(m[1].predict(Xq).astype(np.float32))
            else:
                cols.append(m[1].predict_proba(Xq)[:, 1].astype(np.float32))
        return np.column_stack(cols)


def run_context_eval(X, labels, ts, ts_end, item_ids, *, train_m=6, val_m=2,
                     test_m=2, seed=0, T=None, specs=HEAD_SPECS,
                     min_generalize_frac=MIN_GENERALIZE_FRAC, verbose=True):
    """Walk-forward overfit-driven eval over already-built inputs.

    X: [N, D] head inputs (enriched [emb | ff68] to match production).
    labels: DataFrame, one column per head. ts: decision-bar timestamps.
    ts_end: timestamp at which each bar's forward LABEL window closes (for the
            leak purge: a train bar is used only if its label closed before the
            val window starts). T: optional trivial-baseline features.
    Returns {head_name: aggregate_verdict_dict}.
    """
    cal = pd.DataFrame({'item_id': np.asarray(item_ids),
                        'timestamp': pd.DatetimeIndex(ts),
                        'target': 0.0,
                        'pos': np.arange(len(ts)),
                        'tsend': pd.DatetimeIndex(ts_end)})
    folds = list(walk_forward_folds(cal, train_m, val_m, test_m))
    if verbose:
        print(f"[walk-forward] {len(folds)} folds "
              f"(train={train_m}mo val={val_m}mo test={test_m}mo)")
    rng = np.random.default_rng(seed)
    results = {}
    for name, kind in specs:
        if name not in labels.columns:
            continue
        y_all = labels[name].to_numpy(np.float32)
        of_gap, gen_gap, _ = _tols(kind)
        fold_vs = []
        for fold, tr_df, va_df, te_df in folds:
            va_lo = va_df['timestamp'].min()
            te_lo = te_df['timestamp'].min()
            # leak purge: a train/val bar counts only if its label window
            # closed before the NEXT split begins.
            trp = tr_df.loc[tr_df['tsend'] < va_lo, 'pos'].to_numpy()
            vap = va_df.loc[va_df['tsend'] < te_lo, 'pos'].to_numpy()
            tep = te_df['pos'].to_numpy()

            def _fin(pos):
                return pos[np.isfinite(y_all[pos])]
            trp, vap, tep = _fin(trp), _fin(vap), _fin(tep)
            if len(trp) < 50 or len(vap) < 30 or len(tep) < 30:
                continue

            model, remediated, va = _of.regularized_fit(
                fit=lambda cfg: _fit(kind, X[trp], y_all[trp], seed, **cfg),
                score_train=lambda m: _score(kind, m, X[trp], y_all[trp]),
                score_val=lambda m: _score(kind, m, X[vap], y_all[vap]),
                reg_candidates=CONTEXT_REG_LADDER, overfit_gap=of_gap)
            te = _score(kind, model, X[tep], y_all[tep])
            ysh = y_all[trp].copy(); rng.shuffle(ysh)
            shuf = _score(kind, _fit(kind, X[trp], ysh, seed), X[tep], y_all[tep])
            triv = None
            if T is not None:
                triv = _score(kind, _fit(kind, T[trp], y_all[trp], seed),
                              T[tep], y_all[tep])
            v = head_verdict(kind, _score(kind, model, X[trp], y_all[trp]),
                             va, te, trivial_m=triv, shuffle_m=shuf)
            v['fold'] = fold
            v['remediated'] = remediated
            v['n_test'] = int(len(tep))
            fold_vs.append(v)

        agg = aggregate_verdict(kind, fold_vs, min_generalize_frac)
        results[name] = agg
        if verbose and agg['n_folds']:
            flag = '✅' if agg['accurate'] else '❌'
            print(f"  {flag} {name:<14} {kind}  folds={agg['n_folds']}  "
                  f"mean_test={agg['mean_test']:+.3f}  gen_frac={agg['gen_frac']:.0%}"
                  f"  triv={None if agg['mean_trivial'] is None else round(agg['mean_trivial'],3)}"
                  f"  shuf={agg['mean_shuffle']:+.3f}")
    return results
