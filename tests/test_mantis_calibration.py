"""Platt calibration for the Mantis frozen head (fit_platt / apply_platt + fit_predict `calibrate`).

The pivot MLP head is typically OVER-CONFIDENT, so raw P=0.5 isn't a true 50% signal. Calibration
rescales the proba to track the empirical hit rate -> P=0.5 means ~50% accurate and the proba is a
trustworthy confidence ACROSS tiers (what proba-weighted sizing needs). It MUST be monotonic (no
ranking/AUC change) and a NO-OP when off (backward compat with existing bundles)."""
import numpy as np

from futures_foundation.finetune.classifiers.mantis.frozen import (
    fit_platt, apply_platt, MantisFrozenClassifier)


def _ece(p, y, bins=10):
    """Expected calibration error: |confidence - accuracy| averaged over proba bins (lower=better)."""
    p, y = np.asarray(p), np.asarray(y)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i in range(bins):
        hi = p <= edges[i + 1] if i == bins - 1 else p < edges[i + 1]
        m = (p >= edges[i]) & hi
        if m.sum():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return e


# ---------------------------------------------------------------- calibration math
def test_apply_platt_none_is_identity():
    p = np.array([0.1, 0.5, 0.9])
    assert np.allclose(apply_platt(p, None), p)              # off -> raw, no silent change


def test_apply_platt_identity_params():
    p = np.array([0.01, 0.3, 0.5, 0.7, 0.99])
    assert np.allclose(apply_platt(p, (1.0, 0.0)), p, atol=1e-9)   # A=1,B=0 -> sigmoid(logit(p))=p


def test_apply_platt_monotonic():
    p = np.linspace(0.01, 0.99, 50)
    assert np.all(np.diff(apply_platt(p, (2.5, -0.4))) > 0)  # ranking preserved


def test_fit_platt_one_class_returns_none():
    assert fit_platt(np.array([0.2, 0.8, 0.6]), np.array([1, 1, 1])) is None


def test_fit_platt_fixes_overconfidence():
    rng = np.random.default_rng(0)
    n = 4000
    p_true = rng.uniform(0.15, 0.85, n)
    y = (rng.uniform(size=n) < p_true).astype(int)
    z = np.log(p_true / (1 - p_true))
    p_over = 1 / (1 + np.exp(-3.0 * z))                      # sharpened = over-confident (MLP-like)
    p_cal = apply_platt(p_over, fit_platt(p_over, y))
    assert _ece(p_cal, y) < _ece(p_over, y)                  # calibration IMPROVED
    assert _ece(p_cal, y) < 0.05                             # and it's actually well-calibrated


# ---------------------------------------------------------------- fit_predict integration
def _toy_fit(calibrate, head='logistic', seed=0, **extra):
    rng = np.random.default_rng(seed)
    D = 8
    w = rng.normal(size=D)

    def make(n):
        X = rng.normal(size=(n, D)).astype(np.float32)
        p = 1 / (1 + np.exp(-(X @ w)))
        return X, (rng.uniform(size=n) < p).astype(int)

    Xtr, ytr = make(2000); Xval, yval = make(1500); Xev, yev = make(1500)
    clf = MantisFrozenClassifier(head=head, calibrate=calibrate, max_iter=300, **extra)
    p_val, p_eval, auc = clf.fit_predict(Xtr, ytr, Xval, yval, Xev, seed=seed)
    return clf, p_val, p_eval, auc, yval


def test_calibrate_off_is_no_op():
    clf, p_val, p_eval, auc, yval = _toy_fit(calibrate=False)
    assert clf._platt is None                                # no calibrator fit (backward compat)
    assert p_val.min() >= 0 and p_val.max() <= 1


def test_calibrate_on_sets_platt_and_preserves_ranking():
    from sklearn.metrics import roc_auc_score
    _, pv_off, _, auc_off, yv = _toy_fit(calibrate=False, seed=1)
    clf_on, pv_on, pe_on, auc_on, _ = _toy_fit(calibrate=True, seed=1)
    assert isinstance(clf_on._platt, tuple) and len(clf_on._platt) == 2   # calibrator fit
    assert abs(auc_on - auc_off) < 1e-9                      # Platt monotonic -> AUC unchanged
    assert abs(roc_auc_score(yv, pv_on) - roc_auc_score(yv, pv_off)) < 1e-9  # ranking preserved
    assert pe_on.min() >= 0 and pe_on.max() <= 1


def test_calibrate_mlp_head_runs_and_calibrates():
    clf, p_val, p_eval, auc, yval = _toy_fit(calibrate=True, head='mlp', seed=2)
    assert clf._platt is not None
    assert p_eval.min() >= 0 and p_eval.max() <= 1


# ---------------------------------------------------------------- deploy contract carries the Platt
def _emit_contract(tmp_path, platt):
    import json
    from futures_foundation.finetune.produce import _emit
    out = {'n_train': 100, 'n_oos': 50, 'oos_auc': 0.6, 'oos_meanR': 0.5, 'shuffle_meanR': 0.3,
           'edge_shuffle': 0.2, 'oos_trades': 40, 'platt': platt,
           'val_percentiles': {'ES@3min': {'p50': 0.15, 'p90': 0.31, 'p99': 0.44}}}
    _emit(dict(out), 'mantis_frozen', {}, None, None, None, 5, 64,
          None, ['ES'], ['3min'], '2026-01-01', True, str(tmp_path / 'model'), False)
    return json.loads((tmp_path / 'model_signal.json').read_text())


def test_contract_carries_platt(tmp_path):
    c = _emit_contract(tmp_path, (2.0, -0.5))
    assert c['calibration']['method'] == 'platt'
    assert c['calibration']['A'] == 2.0 and c['calibration']['B'] == -0.5
    assert 'CALIBRATED' in c['proba_meaning']                 # bot knows to apply it


def test_single_head_carries_standardized_score_scale(tmp_path):
    # STANDARDIZED 0-100 SCORE: the single (P(x)) head emits the SAME per_stream_val_percentile
    # block as the ladder, ranking calibrated_proba — so the bot floors on one 0-100 axis and a
    # head swap (proba <-> ladder) needs no change. Mirror of the ladder test's score_scale guard.
    c = _emit_contract(tmp_path, (2.0, -0.5))
    ss = c['score_scale']
    assert ss['kind'] == 'per_stream_val_percentile' and ss['signal'] == 'calibrated_proba'
    assert ss['percentiles'] == {'ES@3min': {'p50': 0.15, 'p90': 0.31, 'p99': 0.44}}
    assert ss['p_min'] == 0.15                                # VAL-median proba backstop
    assert 'calibrated_proba' in ss['rule']


def test_contract_uncalibrated_when_no_platt(tmp_path):
    c = _emit_contract(tmp_path, None)
    assert c['calibration'] is None                           # serve raw, no silent calibration
    assert 'CALIBRATED' not in c['proba_meaning']


# ---------------------------------------------------------------- MLP speed knobs + SKIP_SHUFFLE
def test_mlp_batch_and_alpha_pass_through_and_still_calibrate():
    # custom batch_size + alpha reach the MLP and calibration still runs (no quality path broken)
    clf, p_val, p_eval, auc, yval = _toy_fit(calibrate=True, head='mlp', seed=6,
                                             mlp_batch=64, mlp_alpha=1e-2)
    assert clf._platt is not None
    assert p_eval.min() >= 0 and p_eval.max() <= 1


def test_skip_shuffle_keeps_real_and_calibration(monkeypatch):
    # SKIP_SHUFFLE=1 -> no shuffle control, but REAL metrics + calibration (Platt) are preserved
    from futures_foundation.finetune import produce
    monkeypatch.setenv('SKIP_SHUFFLE', '1')
    rng = np.random.default_rng(0)
    D = 8; w = rng.normal(size=D)

    def make(n):
        X = rng.normal(size=(n, D)).astype(np.float32)
        return X, (rng.uniform(size=n) < 1 / (1 + np.exp(-(X @ w)))).astype(int)

    Xtr, ytr = make(1500); Xval, yval = make(600); Xte, yte = make(600)
    keys = list(range(len(Xte)))

    class Lab:                                                # _arm_R needs only .evaluate(keys, preds)
        def evaluate(self, ks, preds):
            return np.array([1.0 if yte[k] else -1.0 for k, p in zip(ks, preds) if p == 1])

    out = produce._fit_score('mantis_frozen', dict(head='logistic', calibrate=True),
                             Lab(), Xtr, ytr, Xval, yval, Xte, keys, yte, 0, False)
    assert out['shuffle_meanR'] is None                       # shuffle skipped
    assert out['edge_shuffle'] is None
    assert out['oos_meanR'] is not None                       # REAL still computed
    assert out['platt'] is not None                           # calibration ran on the REAL fit
