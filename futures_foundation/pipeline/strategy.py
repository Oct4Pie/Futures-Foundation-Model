"""StrategyLabeler protocol — the single pluggable seam.

A concrete strategy lives OUTSIDE this public package (private colabs/): it
owns its bars, turns causal context into (context, label) training pairs,
and scores predicted decisions into realized per-trade R. The generic
harness and the honest ruler never import a concrete strategy — they only
speak this protocol.

Contract (all implementations MUST honour):
  * causal — every context is bars <= the decision bar; the label is a
    TRAINING TARGET ONLY and may read the realized future.
  * leak-free — in build(), purge any decision whose realized-future label
    window reaches >= test_start (None on the test split = no purge).
  * evaluate() walks the realized future from each decision and returns
    per-trade R INCLUDING cost (cost is the strategy's, not the harness's).
  * contexts are EQUAL-LENGTH 1-D windows (the harness batches them into a
    tensor; ragged lengths break embedding).

Optional: a labeler MAY also expose `features(decision_keys) -> 2-D float
array` (one row per key, aligned to build()'s order). The harness fuses it
with the frozen foundation embedding (hstack) before the head. Omit it for an
embedding-only model. Keep features causal and reproducible as plain tensor
ops (the deployed single-ONNX export folds them in).
"""
from typing import Protocol, Tuple, List, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class StrategyLabeler(Protocol):
    n_classes: int                     # e.g. 3 for Buy/Sell/Hold

    def calendar(self) -> pd.DataFrame:
        """Long frame [item_id, timestamp, target] spanning all bars — fed
        to the leak-free walk-forward splitter. No labels here."""
        ...

    def build(self, lo, hi, test_start
              ) -> Tuple[List[np.ndarray], np.ndarray, List]:
        """Causal decisions with timestamp in [lo, hi).
        Returns (contexts, labels, decision_keys):
          contexts      — list of 1-D causal context windows (bars <= t)
          labels        — int array in [0, n_classes)
          decision_keys — opaque per-decision handles evaluate() understands
        Purge any decision whose label/eval future reaches >= test_start."""
        ...

    def evaluate(self, decision_keys: List, preds: np.ndarray) -> np.ndarray:
        """Predicted class per decision -> realized per-trade R (cost
        included). Skipped/Hold predictions contribute no trade."""
        ...


class BaseChronosStrategy:
    """Optional base giving every strategy a SELF-DESCRIBING signal contract for
    the `<base>_signal.json` sidecar (produce writes it; the bot reads it to
    validate it can run the triplet). Inherit it and declare the class attrs —
    no per-strategy boilerplate:

        class FooChronos(BaseChronosStrategy):
            n_classes = 2
            FLIP_SCHEME    = 'foo'
            FLIP_PARAMS    = {'fast': 9, 'slow': 20}      # or override flip_params()
            DIRECTION_RULE = 'cross_side'
            MIN_GAP        = MIN_GAP
            HANDCRAFT_NAMES = ('ema_spread', 'adx')       # cols AFTER the FFM lib
            STOP_ATR, RR, VERT = STOP_ATR, RR, VERT
            VERSION     = 'foo-1.0'
            TRAIN_SCOPE = {'tickers': TICKERS, 'timeframes': ['3min']}

    Relies on the labeler convention `self._b[tk]['feat_cols']` (the ordered FFM
    library columns). A subclass that declares no attrs still yields a valid
    MINIMAL contract; declaring HANDCRAFT_NAMES is what makes the feature
    contract exact (produce flags a width mismatch loudly)."""
    # contract class-attrs — override in the subclass
    FLIP_SCHEME = None
    FLIP_PARAMS: dict = {}
    DIRECTION_RULE = None
    ENTRY_TIMING = 'next_bar_open'
    MIN_GAP = None
    HANDCRAFT_NAMES: tuple = ()        # ordered handcraft cols AFTER the FFM lib
    STOP_ATR = None
    RR = None
    VERT = None
    FEATURE_LIB = 'futures_foundation'
    PROBA_MEANING = 'P(trade reaches TP before SL)'
    VERSION = None
    TRAIN_SCOPE = None

    def feature_names(self):
        """EXACT ordered handcraft columns features() emits: the FFM library
        cols (self._b[*]['feat_cols']) then this strategy's HANDCRAFT_NAMES."""
        feat_cols = list(next(iter(self._b.values()))['feat_cols'])
        return feat_cols + list(self.HANDCRAFT_NAMES)

    def flip_params(self) -> dict:
        return dict(self.FLIP_PARAMS)

    def signal_contract(self) -> dict:
        label = (f'triple_barrier SL={self.STOP_ATR}ATR TP={self.RR}R '
                 f'VERT={self.VERT}') if self.STOP_ATR is not None else None
        return {
            'flip_scheme': self.FLIP_SCHEME,
            'flip_params': self.flip_params(),
            'direction_rule': self.DIRECTION_RULE,
            'entry_timing': self.ENTRY_TIMING,
            'min_gap_bars': self.MIN_GAP,
            'label_def': label,
            'proba_meaning': self.PROBA_MEANING,
            'feature_lib': self.FEATURE_LIB,
            'version': self.VERSION,
            'train_scope': self.TRAIN_SCOPE,
        }
