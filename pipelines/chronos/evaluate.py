"""Honest-ruler evaluation — strategy-agnostic.

Leak-free walk-forward (data.walk_forward_folds) x {REAL, SHUFFLE, RANDOM}
x seeds. A number is believed ONLY if REAL clearly beats SHUFFLE and RANDOM
on realized R (cost is inside the strategy's evaluate()). No strategy
specifics live here — it speaks only the StrategyLabeler protocol.
"""
import numpy as np

from .data import walk_forward_folds
from . import finetune as ft


def _stats(R):
    R = np.asarray(R, float)
    if not len(R):
        return "trades=0"
    return (f"trades={len(R)} win={np.mean(R > 0):.1%} "
            f"sumR={R.sum():+.1f} meanR={R.mean():+.3f}")


def run(labeler, cfg=None, seeds=(0, 1, 2), train_m=3, test_m=1,
        max_folds=1):
    """labeler: a StrategyLabeler. Prints REAL/SHUFFLE/RANDOM per fold-seed.
    Returns the per-(fold,seed) R arrays for downstream inspection."""
    cfg = cfg or ft.FTConfig(n_classes=labeler.n_classes)
    out, done = [], 0
    for fold, tr, te in walk_forward_folds(labeler.calendar(), train_m,
                                           test_m):
        if done >= max_folds:
            break
        ts0 = te['timestamp'].min()
        Ctr, Ytr, _ = labeler.build(tr['timestamp'].min(),
                                     tr['timestamp'].max(), ts0)
        Cte, _, Kte = labeler.build(
            te['timestamp'].min(),
            te['timestamp'].max() + np.timedelta64(1, 'ns'), None)
        if len(Ytr) < 50 or len(Cte) < 50:
            continue                       # unproductive fold: don't count
        done += 1
        Ytr = np.asarray(Ytr)
        print(f"\n== fold {fold} | ntr={len(Ytr)} nte={len(Cte)} | "
              f"FT {cfg.steps}st b{cfg.batch} | classes={labeler.n_classes} ==")
        for seed in seeds:
            m, h = ft.train(Ctr, Ytr, cfg, seed=seed)
            R = labeler.evaluate(Kte, ft.predict(m, h, Cte))
            ysh = Ytr.copy()
            np.random.default_rng(seed + 1).shuffle(ysh)
            ms, hs = ft.train(Ctr, ysh, cfg, seed=seed)
            Rs = labeler.evaluate(Kte, ft.predict(ms, hs, Cte))
            rnd = np.random.default_rng(seed + 2).integers(
                0, labeler.n_classes, len(Kte))
            Rr = labeler.evaluate(Kte, rnd)
            print(f"  seed {seed}  [REAL   ] {_stats(R)}")
            print(f"          [SHUFFLE] {_stats(Rs)}")
            print(f"          [RANDOM ] {_stats(Rr)}")
            out.append({'fold': fold, 'seed': seed,
                        'REAL': R, 'SHUFFLE': Rs, 'RANDOM': Rr})
    print("\n-> Believe a result only if REAL clearly beats SHUFFLE AND "
          "RANDOM on sumR/meanR across seeds (cost already in evaluate()).")
    return out
