"""XGBoost selection-head hyperparameter tuner (Optuna, anti-overfit bounds).

Strategy- and backbone-agnostic. Tunes ONLY the XGBHead params on top of the
frozen Chronos embedding + the labeler's features — it does NOT touch the
architecture, the signal generator, or any labeler. Use it to fill the
"XGB head tuning" lever: the pipeline otherwise ships fixed params.

Goal: the best setting that GENERALIZES — not the best in-sample fit.

Method (overfit-aware):
  1. Build the walk-forward folds for a labeler, embed ONCE (batched), fuse the
     labeler features — same construction the evaluator uses.
  2. Split folds in time: the first (1-holdout_frac) are the TUNE set, the last
     holdout_frac are an untouched GUARD set.
  3. Optuna (TPE + MedianPruner) searches the tight anti-overfit bounds, scoring
     each trial by a GENERALIZATION-ROBUST objective: cross-fold mean − penalty·
     std of per-fold meanR on the TUNE folds' OOS slices. A config that spikes on
     a few folds and collapses on others (overfit signature) cannot win; only a
     config that is consistent across folds scores high.
  4. The winning params are scored on the untouched GUARD folds vs the shipped
     defaults. ACCEPT the tuned params ONLY if they beat defaults on the GUARD by
     >= GEN_ACCEPT_MARGIN_R (they generalize); otherwise AUTO-FALL-BACK to
     defaults. The function returns what should ACTUALLY be used — params={} means
     keep defaults.

`tune_head(...)` returns dict(params=<{} if keep-defaults else tuned>,
generalizes=bool, chosen='tuned'|'default', best_params=...). With --walkforward
the chosen params are run straight through the full 3-way walk-forward, whose
VAL->TEST gate is the final independent generalization check.

CLI:
    python -m futures_foundation.chronos.tune_head \
        --labeler colabs.kalman_nw_chronos2:KalmanNWChronos2 \
        --trials 80 --max-folds 14
"""
import argparse
import importlib

import numpy as np
import pandas as pd

from .data import walk_forward_folds
from futures_foundation import foundation as backbone
from futures_foundation import overfit as _of
from . import context_fusion
from .head_xgb import XGBHead

# Tight anti-overfit search space (ported from the original XGBoost pipeline's
# spec-section-6 bounds — deliberately narrow; do NOT widen without re-checking
# the overfit guard).
BOUNDS = dict(
    max_depth=(3, 6), learning_rate=(0.01, 0.1),
    subsample=(0.6, 0.85), colsample_bytree=(0.7, 1.0),
    reg_lambda=(1.0, 10.0), min_child_weight=(5, 50),
    n_estimators=(200, 800),
)

# Generalization knobs. STD_PENALTY: how hard the objective punishes cross-fold
# instability (overfit signature) — higher = more conservative. ACCEPT_MARGIN:
# the held-out (GUARD) lift the tuned params must clear to be accepted over the
# shipped defaults; below it we auto-fall-back to defaults.
GEN_STD_PENALTY = 0.5
GEN_ACCEPT_MARGIN_R = 0.05


def _gen_score(per_fold):
    """Generalization-robust score (shared overfit library, meanR penalty).
    A config consistent across folds beats one that spikes on a few and
    collapses on others (the overfit signature), even at equal mean."""
    return _of.gen_score(per_fold, GEN_STD_PENALTY)


def _suggest(trial):
    return dict(
        max_depth=trial.suggest_int('max_depth', *BOUNDS['max_depth']),
        learning_rate=trial.suggest_float('learning_rate',
                                          *BOUNDS['learning_rate'], log=True),
        subsample=trial.suggest_float('subsample', *BOUNDS['subsample']),
        colsample_bytree=trial.suggest_float('colsample_bytree',
                                             *BOUNDS['colsample_bytree']),
        reg_lambda=trial.suggest_float('reg_lambda', *BOUNDS['reg_lambda'],
                                       log=True),
        min_child_weight=trial.suggest_int('min_child_weight',
                                           *BOUNDS['min_child_weight']),
        n_estimators=trial.suggest_int('n_estimators', *BOUNDS['n_estimators'],
                                       step=50),
    )


def _build_folds(labeler, train_m, test_m, max_folds):
    """Build + batch-embed + fuse folds (mirrors evaluate.run phases 1-2)."""
    fold_data = []
    for fold, tr, _val, te in walk_forward_folds(labeler.calendar(), train_m,
                                                 test_months=test_m):
        if max_folds is not None and len(fold_data) >= max_folds:
            break
        ts0 = te['timestamp'].min()
        Ctr, Ytr, Ktr = labeler.build(tr['timestamp'].min(),
                                      tr['timestamp'].max(), ts0)
        Cte, Yte, Kte = labeler.build(
            te['timestamp'].min(),
            te['timestamp'].max() + np.timedelta64(1, 'ns'), None)
        if len(Ytr) < 50 or len(Cte) < 50:
            continue
        fold_data.append(dict(fold=fold, Ctr=Ctr, Ytr=np.asarray(Ytr),
                              Ktr=Ktr, Cte=Cte, Kte=Kte))
    if not fold_data:
        raise SystemExit("No productive folds to tune on.")
    flat = []
    for d in fold_data:
        flat.extend(d['Ctr']); flat.extend(d['Cte'])
    print(f"[tune] batch-embed {len(flat):,} contexts across "
          f"{len(fold_data)} folds...")
    emb = backbone.embed(flat)
    feats_fn = getattr(labeler, 'features', None)
    o = 0
    for d in fold_data:
        ntr, nte = len(d['Ctr']), len(d['Cte'])
        Etr = emb[o:o + ntr]; o += ntr
        Ete = emb[o:o + nte]; o += nte
        xtr = (np.asarray(feats_fn(d['Ktr']), np.float32)
               if feats_fn is not None else None)
        xte = (np.asarray(feats_fn(d['Kte']), np.float32)
               if feats_fn is not None else None)
        # fuse (heads=None) — embed + labeler features, same as evaluate
        d['Xtr'] = context_fusion.fuse(Etr, xtr, None, 'both')
        d['Xte'] = context_fusion.fuse(Ete, xte, None, 'both')
        del d['Ctr'], d['Cte']                       # free raw contexts
    print(f"[tune] feat_dim={fold_data[0]['Xtr'].shape[1]}")
    return fold_data


def _pooled_meanR(labeler, folds, params, seed):
    """Fit XGBHead(**params) per fold, predict its OOS slice, pool realized R."""
    nc = labeler.n_classes
    Rs = []
    for d in folds:
        head = XGBHead(nc, **params).fit(d['Xtr'], d['Ytr'], seed)
        preds = head.predict(d['Xte'])
        R = labeler.evaluate(d['Kte'], preds)
        if len(R):
            Rs.append(R)
    if not Rs:
        return 0.0, 0
    R = np.concatenate(Rs)
    return float(R.mean()), int(len(R))


def tune_head(labeler, *, n_trials=80, seed=42, max_folds=14,
              holdout_frac=0.3, train_m=3, test_m=1):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    folds = _build_folds(labeler, train_m, test_m, max_folds)
    n_guard = max(1, int(round(len(folds) * holdout_frac)))
    tune_folds, guard_folds = folds[:-n_guard], folds[-n_guard:]
    print(f"[tune] {len(tune_folds)} tune folds, {len(guard_folds)} guard "
          f"(held-out) folds\n")

    def objective(trial):
        # Score = "best that GENERALIZES", not best in-sample. We optimize the
        # cross-fold-robust meanR (mean − GEN_STD_PENALTY·std of per-fold meanR),
        # so a config that spikes on a few tune folds and collapses on others —
        # the signature of overfitting — cannot win. A config that generalizes
        # is consistent across folds and scores high here. The untouched GUARD
        # set then independently verifies (below).
        p = _suggest(trial)
        per_fold, pooled = [], []
        for k, d in enumerate(tune_folds):
            head = XGBHead(labeler.n_classes, **p).fit(d['Xtr'], d['Ytr'], seed)
            R = labeler.evaluate(d['Kte'], head.predict(d['Xte']))
            if len(R):
                per_fold.append(float(R.mean()))
                pooled.append(R)
            running = float(np.concatenate(pooled).mean()) if pooled else 0.0
            trial.report(running, k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        score = _gen_score(per_fold)                # generalization-robust score
        return score if score != float('-inf') else -1.0   # finite floor for TPE

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2))
    print(f"[tune] Optuna {n_trials} trials over anti-overfit bounds...\n")
    study.optimize(objective, n_trials=n_trials)

    best = study.best_params
    # default-params baseline vs tuned, on BOTH tune and guard folds
    base = {}
    b_tune, n_bt = _pooled_meanR(labeler, tune_folds, base, seed)
    b_guard, n_bg = _pooled_meanR(labeler, guard_folds, base, seed)
    t_tune, n_tt = _pooled_meanR(labeler, tune_folds, best, seed)
    t_guard, n_tg = _pooled_meanR(labeler, guard_folds, best, seed)

    bar = "=" * 64
    print(f"\n{bar}\n🔧 HEAD-TUNER RESULT\n{bar}")
    print(f"  {'arm':<16}{'TUNE meanR':>12}{'GUARD meanR':>13}")
    print(f"  {'default':<16}{b_tune:>+12.3f}{b_guard:>+13.3f}")
    print(f"  {'tuned':<16}{t_tune:>+12.3f}{t_guard:>+13.3f}")
    lift_guard = t_guard - b_guard
    print(f"\n  GUARD lift (tuned − default): {lift_guard:+.3f}R  "
          f"(this is the honest, held-out lift)")

    # ---- ACCEPT / FALL-BACK decision (auto) -----------------------------
    # Accept the tuned params ONLY if they GENERALIZE: beat the shipped defaults
    # on the untouched GUARD set by a real margin. Otherwise the search overfit
    # the tune folds (won TUNE, lost GUARD) → auto-fall-back to defaults. We
    # return what should ACTUALLY be used, not the raw Optuna winner.
    # accept tuned only if it beats default on the held-out GUARD by the margin
    # (shared overfit library — same selection logic as the reg-ladder rungs).
    generalizes = _of.best_config(
        b_guard, [(best, t_guard)], accept_margin=GEN_ACCEPT_MARGIN_R) is not None
    if generalizes:
        chosen, chosen_name = best, 'tuned'
        print(f"\n  ✅ ACCEPT tuned — generalizes (GUARD lift "
              f"{lift_guard:+.3f}R ≥ {GEN_ACCEPT_MARGIN_R}R).")
    else:
        chosen, chosen_name = {}, 'default'
        why = ("overfit the tune folds" if t_tune > t_guard + 0.15
               else "no held-out lift")
        print(f"\n  ↩️  KEEP DEFAULTS — tuned did not generalize "
              f"({why}; GUARD lift {lift_guard:+.3f}R < {GEN_ACCEPT_MARGIN_R}R).")
    print(f"\n  chosen params ({chosen_name}): {chosen or 'shipped defaults'}")
    if generalizes:
        print(f"\n  → certify before use:")
        print(f"    ev.run(lab, head_factory=lambda nc: XGBHead(nc, **{best}))")
    return dict(params=chosen, generalizes=generalizes, chosen=chosen_name,
                best_params=best, tune_meanR=t_tune, guard_meanR=t_guard,
                default_tune_meanR=b_tune, default_guard_meanR=b_guard,
                guard_lift=lift_guard)


def _load_labeler(spec):
    mod_name, cls_name = spec.split(':')
    mod = importlib.import_module(mod_name)
    return getattr(mod, cls_name)()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--labeler', required=True,
                    help='module:Class, e.g. colabs.kalman_nw_chronos2:'
                         'KalmanNWChronos2')
    ap.add_argument('--trials', type=int, default=80)
    ap.add_argument('--max-folds', type=int, default=14)
    ap.add_argument('--holdout-frac', type=float, default=0.3)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--walkforward', action='store_true',
                    help='after the Optuna scan, run the FULL train/validate/test '
                         'walk-forward with the tuned params (honest OOS + VAL→TEST '
                         'generalization gate). This is the "scan → walk-forward" '
                         'process.')
    args = ap.parse_args()
    backbone.stamp_active_source(context='head-tuner')
    lab = _load_labeler(args.labeler)
    res = tune_head(lab, n_trials=args.trials, seed=args.seed,
                    max_folds=args.max_folds, holdout_frac=args.holdout_frac)
    if args.walkforward:
        from . import evaluate as ev
        from .head_xgb import XGBHead
        p = res['params']                       # {} → shipped defaults (auto-fallback)
        bar = "█" * 64
        print(f"\n{bar}\n  WALK-FORWARD with {res['chosen']} params:"
              f"\n  {p or 'shipped defaults'}\n{bar}")
        # auto_regularize OFF — the scan already chose the params (tuned if they
        # generalized on the guard, else defaults); the walk-forward reports the
        # honest 3-way OOS and the VAL→TEST gate is the final generalization check.
        ev.run(lab, head_factory=lambda nc: XGBHead(nc, **p),
               auto_regularize=False)


if __name__ == '__main__':
    main()
