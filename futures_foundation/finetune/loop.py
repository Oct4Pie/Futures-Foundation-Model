"""Overfit-driven training loop — Classifier-agnostic (concept from pipeline.train_loop).

The self-correcting fit process:
  1. Walk-forward with DEFAULT params.
  2. Generalization gate (VAL->TEST gap).  Generalizes -> keep defaults.
  3. Overfit -> Optuna search for params that generalize (on a held-out val guard).
  4. Rerun the walk-forward with the chosen params.
  5. Repeat 2-4 until it generalizes OR max_iters / Optuna can't help.
  6. Final full walk-forward with the stable params.

Optuna is triggered ONLY on overfit, so a config that already generalizes is kept
untouched. Any backbone (Mantis/logistic/...) via the Classifier seam.
"""
import numpy as np

from . import wf, tune
from .classifier import get_classifier
from futures_foundation.pipeline.data import walk_forward_folds


def _first_fold_arrays(labeler, classifier, clf_kwargs, train_m, val_m, test_m, holdout_start):
    """Featurize the first productive fold's (train, val) for Optuna tuning."""
    clf = get_classifier(classifier, **dict(clf_kwargs or {}))
    for fold, tr, val, te in walk_forward_folds(labeler.calendar(), train_m, val_m,
                                                test_m, holdout_start=holdout_start):
        val0, te0 = val['timestamp'].min(), te['timestamp'].min()
        _, Ytr, Ktr = labeler.build(tr['timestamp'].min(), tr['timestamp'].max(), val0)
        _, Yval, Kval = labeler.build(val0, val['timestamp'].max(), te0)
        Ytr, Yval = np.asarray(Ytr).astype(int), np.asarray(Yval).astype(int)
        if len(Ytr) < 50 or len(Yval) < 10:
            continue
        Xtr = clf.featurize(labeler, Ktr)
        Xval = clf.featurize(labeler, Kval)
        if clf.needs_standardize:
            Xtr, Xval, _, _, _ = wf._standardize_on_train(Xtr, Xval, Xval)
        return Xtr, Ytr, Xval, Yval
    return None


def train_loop(labeler, classifier='mantis', base_kwargs=None, *, max_iters=2,
               loop_max_folds=1, final_max_folds=None, seeds=(0,), train_m=12, val_m=2,
               test_m=2, holdout_start='2026-01-01', n_trials=8, seed=42, verbose=True):
    base = dict(base_kwargs or {})

    def _wf(params, mf, tag):
        if verbose:
            print(f"\n{'='*60}\n  WALK-FORWARD [{tag}] params={params or 'defaults'}\n{'='*60}")
        return wf.run(labeler, classifier, params, seeds=seeds, max_folds=mf, train_m=train_m,
                      val_m=val_m, test_m=test_m, holdout_start=holdout_start, verbose=verbose)

    v = _wf(base, loop_max_folds, "iter 0 · defaults")
    params, source = base, 'default'
    history = [dict(it=0, source=source, generalizes=v['generalizes'], gap=v['gap'])]

    if not v['generalizes']:
        if verbose:
            print(f"\n!! OVERFIT (VAL->TEST gap {v['gap']}) -> Optuna search")
        arrays = _first_fold_arrays(labeler, classifier, base, train_m, val_m, test_m, holdout_start)
        if arrays is not None:
            Xtr, ytr, Xval, yval = arrays
            for it in range(1, max_iters + 1):
                scan = tune.tune(labeler, classifier, Xtr, ytr, Xval, yval,
                                 n_trials=n_trials, base_kwargs=base, seed=seed + it,
                                 verbose=verbose)
                if not scan['generalizes']:
                    if verbose:
                        print(f"  Optuna found nothing better (iter {it}); stop.")
                    break
                params, source = scan['params'], f'tuned(iter{it})'
                v = _wf(params, loop_max_folds, f"iter {it} · tuned")
                history.append(dict(it=it, source=source, generalizes=v['generalizes'],
                                    gap=v['gap']))
                if v['generalizes']:
                    if verbose:
                        print(f"\n** TUNED params generalize (iter {it}).")
                    break

    final = _wf(params, final_max_folds, "FINAL · all folds")
    if verbose:
        print(f"\n# TRAIN-LOOP RESULT: chosen={source} params={params or 'defaults'}")
        print(f"#  final: generalizes={final['generalizes']} gap={final['gap']} "
              f"all_pass={final['all_pass']} AUC={final['auc']} "
              f"edge(REAL-SHUF)={final['edge_shuffle']:+.3f}")
    return dict(params=params, source=source, final=final, history=history)
