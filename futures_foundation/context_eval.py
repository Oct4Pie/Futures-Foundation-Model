"""Overfit-driven evaluation for the foundation CONTEXT heads.

Same process the strategy selection heads use (futures_foundation.chronos), via
the shared futures_foundation.overfit library — adapted to the context heads'
metrics (Pearson r for regression, ROC-AUC for classification):

  1. 3-way split — train / validate / TEST, where TEST is the >= HEADS_CUTOFF
     period the heads ACTUALLY feed downstream models. (The probe only ever
     checked pre-cutoff validation; this confirms accuracy where it's used.)
  2. Fit the default head on train; score train / val / test.
  3. Overfit detect (train→val) → conditional re-fit down a regularization
     ladder, keeping the rung with the best VALIDATION metric (auto-fallback to
     default if none beats it). Selection sees train+val only.
  4. GENERALIZATION gate — VAL→TEST metric decay within tolerance: a head whose
     accuracy decays on the used-period is NOT trustworthy as a model input.
  5. Honest controls — must beat the SHUFFLE control and the TRIVIAL baseline.

`head_verdict` is the pure decision core (unit-tested); `run_context_eval`
orchestrates fit/score/controls over already-embedded inputs.
"""
import numpy as np

from futures_foundation import overfit as _of
from futures_foundation.context import HEAD_SPECS, GATE_REG_PEARSON, GATE_CLF_AUC

# Per-metric tolerances (Pearson r and ROC-AUC live on different scales).
REG_OVERFIT_GAP = 0.10        # train→val Pearson-r gap that means "memorized"
CLF_OVERFIT_GAP = 0.05        # train→val AUC gap
REG_GEN_GAP = 0.10            # val→test Pearson-r decay still trusted
CLF_GEN_GAP = 0.05            # val→test AUC decay still trusted

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
    """Pure verdict for one context head, using the shared overfit library.

    A head is ACCURATE (usable as a model input) only if it: clears the skill
    floor on TEST, GENERALIZES (val→test decay within tolerance), and beats both
    the SHUFFLE control and the TRIVIAL baseline on TEST. `overfit` flags a
    train→val gap (remediation should have fired). Returns a dict.
    """
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


def run_context_eval(E, labels, ts, val_start, cutoff, seed=0,
                     T=None, specs=HEAD_SPECS, verbose=True):
    """3-way overfit-driven evaluation over already-embedded inputs.

    E: [N, D] foundation embeddings (+ optional fused features baked in).
    labels: DataFrame with a column per head name. ts: pd.Series of timestamps.
    T: optional [N, t] trivial-baseline features (the "beat trivial?" control).
    Returns {head_name: verdict_dict}. Selection/remediation never see TEST.
    """
    ts = ts.reset_index(drop=True)
    tr = (ts < val_start).to_numpy()
    va = ((ts >= val_start) & (ts < cutoff)).to_numpy()
    te = (ts >= cutoff).to_numpy()
    if verbose:
        print(f"[split] train={tr.sum():,}  val={va.sum():,}  "
              f"test={te.sum():,}  (test = >= {cutoff.date()}, where heads "
              f"feed models)")
    rng = np.random.default_rng(seed)
    results = {}
    for name, kind in specs:
        if name not in labels.columns:
            continue
        y = labels[name].to_numpy(np.float32)
        ok = np.isfinite(y)
        m_tr, m_va, m_te = tr & ok, va & ok, te & ok
        if m_tr.sum() < 50 or m_va.sum() < 30 or m_te.sum() < 30:
            if verbose:
                print(f"  [skip] {name}: too few rows "
                      f"(tr={m_tr.sum()} va={m_va.sum()} te={m_te.sum()})")
            continue

        # default head + train/val metrics
        head = _fit(kind, E[m_tr], y[m_tr], seed)
        tr_m = _score(kind, head, E[m_tr], y[m_tr])
        va_m = _score(kind, head, E[m_va], y[m_va])

        # conditional remediation: only if overfit train→val
        of_gap = _tols(kind)[0]
        remediated = None
        if _of.overfit_trigger(tr_m, va_m, of_gap):
            cands = []
            for cfg in CONTEXT_REG_LADDER:
                h = _fit(kind, E[m_tr], y[m_tr], seed, **cfg)
                cands.append((cfg, _score(kind, h, E[m_va], y[m_va]), h))
            best_cfg = _of.best_config(va_m, [(c, m) for c, m, _ in cands])
            if best_cfg is not None:
                head = next(h for c, m, h in cands if c is best_cfg)
                remediated = best_cfg
                va_m = _score(kind, head, E[m_va], y[m_va])

        # TEST metric (the used-period) + controls
        te_m = _score(kind, head, E[m_te], y[m_te])
        ysh = y[m_tr].copy(); rng.shuffle(ysh)
        shuf = _score(kind, _fit(kind, E[m_tr], ysh, seed), E[m_te], y[m_te])
        triv = None
        if T is not None:
            triv = _score(kind, _fit(kind, T[m_tr], y[m_tr], seed),
                          T[m_te], y[m_te])

        v = head_verdict(kind, tr_m, va_m, te_m, trivial_m=triv, shuffle_m=shuf)
        v['remediated'] = remediated
        v['n_test'] = int(m_te.sum())
        results[name] = v
        if verbose:
            flag = '✅' if v['accurate'] else '❌'
            print(f"  {flag} {name:<14} {kind}  train={tr_m:+.3f} val={va_m:+.3f} "
                  f"TEST={te_m:+.3f}  triv={triv if triv is None else round(triv,3)}"
                  f" shuf={shuf:+.3f}  gen={v['generalizes']} "
                  f"{'(remediated)' if remediated else ''}")
    return results
