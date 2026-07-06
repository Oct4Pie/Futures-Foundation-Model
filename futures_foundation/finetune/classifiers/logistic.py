"""LogisticClassifier — torch-free baseline + the reference impl that lets the
generic harness/loop/tuner be unit-tested end-to-end without torch.

Logistic regression over flattened windows. Registered as 'logistic'.
"""
import numpy as np

from ..classifier import Classifier, register_classifier


@register_classifier('logistic')
class LogisticClassifier(Classifier):
    needs_standardize = False           # scales internally

    @staticmethod
    def suggest_params(trial):
        return dict(C=trial.suggest_float('C', 0.01, 10.0, log=True))

    def __init__(self, C=1.0, max_iter=1000, **_ignored):
        self.C = C
        self.max_iter = max_iter

    def featurize(self, labeler, keys):
        return np.asarray(labeler.mv_contexts(keys), np.float32)

    def fit_predict(self, Xtr, ytr, Xval, yval, Xeval, seed=0):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import roc_auc_score

        def flat(A):
            if isinstance(A, str):                  # memmap path (stream mode) -> load
                A = np.load(A, mmap_mode='r')
            A = np.asarray(A, np.float32)
            return A.reshape(len(A), -1)

        sc = StandardScaler().fit(flat(Xtr))
        clf = LogisticRegression(C=self.C, max_iter=self.max_iter, random_state=seed)
        clf.fit(sc.transform(flat(Xtr)), np.asarray(ytr).astype(int))
        p_val = clf.predict_proba(sc.transform(flat(Xval)))[:, 1]
        p_eval = clf.predict_proba(sc.transform(flat(Xeval)))[:, 1]
        yv = np.asarray(yval).astype(int)
        ba = float(roc_auc_score(yv, p_val)) if len(np.unique(yv)) == 2 else float('nan')
        return p_val, p_eval, ba
