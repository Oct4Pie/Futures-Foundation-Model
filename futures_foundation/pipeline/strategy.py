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
