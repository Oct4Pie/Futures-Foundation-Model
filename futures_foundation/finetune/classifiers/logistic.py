"""LogisticClassifier — torch-free baseline Classifier.

A logistic regression over flattened (standardized) multivariate windows. Two
jobs: (1) a cheap, fast baseline to A/B against fine-tuned backbones, and (2) the
torch-free reference impl that lets the walk-forward harness + subprocess worker
be unit-tested end-to-end WITHOUT loading torch. Registered as 'logistic'.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ..classifier import Classifier, register_classifier


@register_classifier('logistic')
class LogisticClassifier(Classifier):
    def __init__(self, n_channels, C=1.0, max_iter=1000, **_ignored):
        self.n_channels = n_channels
        self.C = C
        self.max_iter = max_iter
        self.scaler = None
        self.clf = None
        self.best_val_auc = float('nan')

    def fit(self, X, y, X_val=None, y_val=None, seed=0):
        Xf = np.asarray(X, np.float32).reshape(len(X), -1)
        y = np.asarray(y).astype(int)
        self.scaler = StandardScaler().fit(Xf)
        self.clf = LogisticRegression(C=self.C, max_iter=self.max_iter, random_state=seed)
        self.clf.fit(self.scaler.transform(Xf), y)
        if X_val is not None and y_val is not None and len(np.unique(y_val)) == 2:
            from sklearn.metrics import roc_auc_score
            self.best_val_auc = float(roc_auc_score(y_val, self.predict_proba(X_val)))
        return self

    def predict_proba(self, X):
        Xf = np.asarray(X, np.float32).reshape(len(X), -1)
        return self.clf.predict_proba(self.scaler.transform(Xf))[:, 1]
