"""Selection pipeline — backbone-agnostic, strategy-pluggable honest-ruler eval.

Lives inside `futures_foundation` (renamed from `chronos`; formerly
`pipelines/chronos`). A pluggable strategy supplies labels; a frozen foundation
embedding (from the configured extractor — `futures_foundation.extractors`,
default Chronos-Bolt) + the strategy's features feed an XGBoost selection head;
results are judged only under a leak-free, shuffle/random/cost honest ruler.
Strategy specifics live OUTSIDE this public package (private colabs/); this
package stays strategy- AND backbone-agnostic (swap the extractor, not this).

Files:
  (backbone)  — the foundation embedding comes from the configured extractor
                (`futures_foundation.extractors`; default Chronos-Bolt at
                `extractors/chronos/backbone.py`). Modules import the default
                as `backbone`; swap via `extractors.evaluate_with_extractor`.
  strategy.py — StrategyLabeler protocol (calendar/build/evaluate);
                contract: causal contexts, leak-free purge, cost inside
                evaluate(). The single pluggable seam.
  evaluate.py — honest ruler: leak-free walk-forward x seeds x
                REAL/SHUFFLE/RANDOM; believe a result only if REAL clearly
                beats SHUFFLE and RANDOM.
  data.py     — 6-ticker bars -> long format; leak-free walk-forward splits
                (born-tested: tests/test_chronos_data.py).
  head_xgb.py — XGBoost selection head; produce/tune_head/train_loop — production
                fit, Optuna head-tuning, overfit-driven loop.

Non-negotiable: strictly causal (bars <= t), splits never leak forward, no
performance number believed until it clears shuffle + random + cost.
"""
import sys as _sys

# LEGACY-PICKLE + RENAME COMPAT: production joblib bundles reference pipeline
# classes under OLD module paths — 'pipelines.chronos.*' (pre-consolidation
# bundles) and 'futures_foundation.pipeline.*' (post-consolidation, pre-rename
# bundles, e.g. kalman_nw_chronos.joblib / supertrend_chronos.joblib). This
# package was renamed chronos -> pipeline; aliasing BOTH old names to it lets
# Python resolve those submodules through OUR __path__, so existing bundles keep
# unpickling unmodified. The 'futures_foundation.chronos' alias also bridges any
# code still importing the old package path during the colabs migration.
_sys.modules.setdefault('pipelines.chronos', _sys.modules[__name__])
_sys.modules.setdefault('futures_foundation.chronos', _sys.modules[__name__])
