"""MOMENT frozen-encoder classifier (STUB). Mirror classifiers/mantis/frozen.py to activate:
featurize returns raw windows [N, C, seq]; fit_predict embeds them with the frozen MOMENT
encoder (isolated subprocess, like Mantis) + fits the head. Once these return embeddings/probas,
the generic walk-forward, calibration, and the distributional risk_head all work with no changes.
"""
import numpy as np

from ...classifier import Classifier, register_classifier


@register_classifier('moment_frozen')
class MomentFrozenClassifier(Classifier):
    needs_standardize = True            # harness standardizes the embeddings on train stats

    def __init__(self, **cfg):
        self.cfg = cfg

    def featurize(self, labeler, keys):
        # Same labeler API as Mantis: raw OHLCV windows [N, C, seq] -> the MOMENT encoder.
        return np.asarray(labeler.mv_contexts(keys), np.float32)

    def fit_predict(self, Xtr, ytr, Xval, yval, Xeval, seed=0):
        raise NotImplementedError(
            "MOMENT backbone is a stub. Implement the frozen MOMENT-encoder embedding + head here "
            "(mirror classifiers/mantis/frozen.py); the generic harness, calibration, and "
            "risk_head consume it unchanged once this returns (p_val, p_eval, best_val_auc).")
