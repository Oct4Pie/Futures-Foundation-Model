"""Overfit-driven training loop: default → check → (Optuna only if overfit) → repeat → final.

The robust, self-correcting process for fitting a strategy's XGBoost selection
head. Optuna is NOT run by default — it is triggered ONLY when the walk-forward
detects overfitting, so a model whose defaults already generalize keeps them
untouched, and a model that overfits gets searched for params that generalize.

Loop (exactly):
  1. Run the walk-forward with the DEFAULT XGBoost params.
  2. Check for overfitting (VAL->TEST generalization gate).
       passes (generalizes) -> KEEP DEFAULTS as final.
  3. Overfit detected -> trigger an Optuna scan for better params (the scan only
     RETURNS params that beat defaults on its own held-out guard; otherwise it
     falls back to defaults).
  4. Rerun the walk-forward with the chosen params.
  5. Repeat 2-4 until the walk-forward passes without overfitting, OR a
     termination guard fires (max_iters, or Optuna can't find generalizing
     params -> stop and flag the model).
  6. One FINAL full walk-forward (all folds) with the stable params, to confirm
     performance on unseen data.

The loop iterations use a subsampled walk-forward (loop_max_folds) for speed; the
final confirmation (step 6) uses every fold.

CLI:
    python -m futures_foundation.pipeline.train_loop \
        --labeler colabs.supertrend_chronos:SuperTrendChronos
"""
import argparse
import importlib

from . import evaluate as ev
from . import tune_head as TH
from .head_xgb import XGBHead
from futures_foundation.extractors.chronos import backbone


def _walk_forward(labeler, params, *, seeds, max_folds, tag,
                  use_regime=False, regime_states=4, holdout_start=None,
                  embed_cache_dir=None):
    bar = "█" * 64
    print(f"\n{bar}\n  WALK-FORWARD [{tag}] — params: "
          f"{params or 'shipped defaults'}\n{bar}", flush=True)
    return ev.run(labeler, head_factory=lambda nc, _p=params: XGBHead(nc, **_p),
                  seeds=seeds, max_folds=max_folds,
                  auto_regularize=False, return_verdict=True,
                  use_regime=use_regime, regime_states=regime_states,
                  holdout_start=holdout_start, embed_cache_dir=embed_cache_dir)


def _overfit(v):
    """True if the walk-forward verdict shows overfitting (does NOT generalize)."""
    return v is not None and not v.get('generalizes', True)


def train_loop(labeler, *, max_iters=3, loop_max_folds=12, final_max_folds=None,
               seeds=(0, 1, 2), scan_trials=40, seed=42,
               use_regime=False, regime_states=4, holdout_start=None,
               embed_cache_dir=None):
    _rk = dict(use_regime=use_regime, regime_states=regime_states,
               holdout_start=holdout_start, embed_cache_dir=embed_cache_dir)
    backbone.stamp_active_source(context='train-loop')
    name = type(labeler).__name__
    print(f"\n{'#'*64}\n# TRAIN LOOP — {name}\n{'#'*64}")

    # ---- Step 1: walk-forward with DEFAULT params -----------------------
    params, source = {}, 'default'
    v = _walk_forward(labeler, params, seeds=seeds,
                      max_folds=loop_max_folds, tag="iter 0 · defaults", **_rk)

    history = [dict(iter=0, source='default', params={},
                    generalizes=(v or {}).get('generalizes'),
                    gap=(v or {}).get('gap'),
                    all_pass=(v or {}).get('all_pass'))]

    # ---- Step 2: if it generalizes, keep defaults -----------------------
    if not _overfit(v):
        print(f"\n✅ DEFAULTS GENERALIZE — no overfitting "
              f"(gap {(v or {}).get('gap')}). Keeping default params.")
    else:
        # ---- Steps 3-5: overfit -> Optuna loop, repeat until clean ------
        print(f"\n⚠️  OVERFIT DETECTED on defaults "
              f"(VAL→TEST gap {(v or {}).get('gap'):+.3f}R > {ev.GEN_GAP_TOL}R) "
              f"→ triggering Optuna scan.")
        stuck = False
        for it in range(1, max_iters + 1):
            print(f"\n— Optuna scan (iter {it}/{max_iters}) —")
            scan = TH.tune_head(labeler, n_trials=scan_trials, seed=seed + it,
                                max_folds=loop_max_folds)
            if not scan.get('generalizes'):
                print(f"\n↩️  Optuna found NO generalizing params this round "
                      f"(guard lift {scan.get('guard_lift'):+.3f}R). ")
                stuck = True
                break                      # tuning can't fix it → stop, flag
            params, source = scan['params'], f'tuned(iter{it})'
            v = _walk_forward(labeler, params, seeds=seeds,
                              max_folds=loop_max_folds, tag=f"iter {it} · tuned",
                              **_rk)
            history.append(dict(iter=it, source=source, params=params,
                                generalizes=(v or {}).get('generalizes'),
                                gap=(v or {}).get('gap'),
                                all_pass=(v or {}).get('all_pass')))
            if not _overfit(v):
                print(f"\n✅ TUNED PARAMS GENERALIZE (iter {it}) — overfitting "
                      f"resolved.")
                break
            print(f"\n⚠️  Still overfit after iter {it}; continuing.")
        else:
            stuck = True
            print(f"\n🚩 Loop exhausted {max_iters} iters still overfit.")
        if stuck and _overfit(v):
            print(f"\n🚩 FLAG: could not reach a generalizing config — "
                  f"defaults are the best available but still overfit. "
                  f"This model needs attention (not an XGB-head problem).")
            params, source = {}, 'default(flagged)'

    # ---- Step 6: final FULL walk-forward to confirm on unseen data ------
    print(f"\n{'═'*64}\n  STEP 6 — FINAL FULL walk-forward confirmation\n{'═'*64}")
    final = _walk_forward(labeler, params, seeds=seeds,
                          max_folds=final_max_folds, tag="FINAL · all folds", **_rk)

    print(f"\n{'#'*64}\n# TRAIN-LOOP RESULT — {name}")
    print(f"#  chosen params : {source} → {params or 'shipped defaults'}")
    if final:
        print(f"#  final OOS     : generalizes={final.get('generalizes')}  "
              f"gap={final.get('gap')}  all_pass={final.get('all_pass')}  "
              f"AUC={final.get('auc')}")
        print(f"#  final TEST    : meanR={final.get('test_meanR')}  "
              f"n={final.get('test_n')}  edge(REAL−SHUF)={final.get('edge_shuffle')}")
    print(f"{'#'*64}", flush=True)
    return dict(params=params, source=source, final=final, history=history)


def _load_labeler(spec):
    mod_name, cls_name = spec.split(':')
    return getattr(importlib.import_module(mod_name), cls_name)()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--labeler', required=True, help='module:Class')
    ap.add_argument('--max-iters', type=int, default=3)
    ap.add_argument('--loop-max-folds', type=int, default=12)
    ap.add_argument('--final-max-folds', type=int, default=None)
    ap.add_argument('--scan-trials', type=int, default=40)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    lab = _load_labeler(args.labeler)
    train_loop(lab, max_iters=args.max_iters, loop_max_folds=args.loop_max_folds,
               final_max_folds=args.final_max_folds, scan_trials=args.scan_trials,
               seed=args.seed)


if __name__ == '__main__':
    main()
