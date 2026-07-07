"""Distributional reach-ladder head wired into MantisFrozenClassifier.fit_predict (torch-free).

The head-only fit_predict path is pure sklearn (the frozen encoder runs elsewhere), so we drive it
directly with synthetic embeddings + strategy keys carrying per-target realized R. Locks:
  - rank='expected_reach' fits the RiskHead ladder and returns expected-R as the ranking score
  - keys back-compat: without rank= the single-head path is byte-unchanged (ignores keys)
  - the ladder ranks bigger-runners above stall-outs (the whole point — entry selection)
  - the SHUFFLE control (permuted keys) genuinely degrades ranking (control isn't a no-op)
"""
import numpy as np
import pytest

from futures_foundation.finetune.classifier import get_classifier
from futures_foundation.finetune.risk_head import TARGETS

NT = len(TARGETS)


def _data(n, D=16, seed=0):
    """Embeddings + keys where a latent score s drives BOTH the 3R metric label and how far the
    trade runs up the ladder. Reach = (s + light noise) clears the rung's threshold, so the ladder
    is monotone (can't reach 4R without 3R) and s-ranked (an informative-but-imperfect signal)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, D, 1)).astype(np.float32)     # [N, D, 1] (frozen-embed memmap shape)
    w = np.random.default_rng(12345).normal(size=D)       # FIXED latent direction (shared across splits)
    s = X[:, :, 0] @ w
    s = (s - s.mean()) / (s.std() + 1e-9)
    offs = [-1.0, -0.3, 0.3, 1.1, 1.9]                    # rung difficulty (2,3,4,6,8), increasing
    keys, y3 = [], []
    for r in range(n):
        z = s[r] + rng.normal(0, 0.4)                     # per-trade noise -> imperfect but real
        realized = [1.0 if z > off else -1.0 for off in offs]   # monotone (offs increasing)
        keys.append(('ES@3min', r, 1, 0.0, *realized))
        y3.append(1 if realized[1] > 0 else 0)            # metric label = 3R rung
    return X, np.array(y3), keys


def _clf(**cfg):
    # head-only sklearn path; no backbone_ckpt -> featurize never called (we pass X directly)
    return get_classifier('mantis_frozen', head='mlp', calibrate=True,
                          mlp_batch=256, max_iter=120, **cfg)


def test_distributional_returns_expected_reach_ranking():
    Xtr, ytr, ktr = _data(1600, seed=1)
    Xval, yval, kval = _data(600, seed=2)
    Xev, yev, kev = _data(600, seed=3)
    clf = _clf(rank='expected_reach', reach_targets=list(TARGETS))
    p_val, p_ev, auc = clf.fit_predict(Xtr, ytr, Xval, yval, Xev, seed=0,
                                       keys_tr=ktr, keys_val=kval)
    assert p_val.shape == (600,) and p_ev.shape == (600,)
    # expected-R is an area over the ladder -> spans a real range (not a constant), >= 0
    assert p_ev.min() >= 0 and p_ev.max() > p_ev.min()
    assert clf._risk_head is not None and len(clf._risk_head._heads) == NT
    # ranking is informative: expected-R separates the 3R-reachers from the rest
    from sklearn.metrics import roc_auc_score
    if len(np.unique(yev)) == 2:
        assert roc_auc_score(yev, p_ev) > 0.6


def test_distributional_ranks_runners_above_stalls():
    Xtr, ytr, ktr = _data(1800, seed=4)
    Xev, yev, kev = _data(800, seed=5)
    clf = _clf(rank='expected_reach', reach_targets=list(TARGETS))
    _, p_ev, _ = clf.fit_predict(Xtr, ytr, Xtr, ytr, Xev, seed=0, keys_tr=ktr, keys_val=ktr)
    reach8 = np.array([k[4 + (NT - 1)] > 0 for k in kev])  # actually ran to 8R
    stalled = np.array([k[4] < 0 for k in kev])            # never even reached 2R
    if reach8.any() and stalled.any():
        assert p_ev[reach8].mean() > p_ev[stalled].mean()


def test_shuffle_keys_degrades_ranking():
    # the harness permutes keys in lockstep for the SHUFFLE control; a permuted ladder must destroy
    # the signal (else the control collapses onto REAL). Same X, shuffled key->X association.
    from sklearn.metrics import roc_auc_score
    Xtr, ytr, ktr = _data(1800, seed=6)
    Xev, yev, kev = _data(700, seed=7)
    clf = _clf(rank='expected_reach', reach_targets=list(TARGETS))
    _, p_real, _ = clf.fit_predict(Xtr, ytr, Xtr, ytr, Xev, seed=0, keys_tr=ktr, keys_val=ktr)
    perm = np.random.default_rng(0).permutation(len(ytr))
    ksh = [ktr[i] for i in perm]
    clf2 = _clf(rank='expected_reach', reach_targets=list(TARGETS))
    _, p_shuf, _ = clf2.fit_predict(Xtr, ytr[perm], Xtr, ytr, Xev, seed=0,
                                    keys_tr=ksh, keys_val=ktr)
    if len(np.unique(yev)) == 2:
        assert roc_auc_score(yev, p_real) > roc_auc_score(yev, p_shuf) + 0.05


def test_keys_backcompat_single_head_unchanged():
    # without rank=, passing keys_tr must NOT change the single-head result at all (ignored)
    Xtr, ytr, ktr = _data(900, seed=8)
    Xval, yval, kval = _data(300, seed=9)
    Xev, yev, kev = _data(300, seed=10)
    base = _clf().fit_predict(Xtr, ytr, Xval, yval, Xev, seed=0)
    withk = _clf().fit_predict(Xtr, ytr, Xval, yval, Xev, seed=0, keys_tr=ktr, keys_val=kval)
    assert np.allclose(base[0], withk[0]) and np.allclose(base[1], withk[1])
    assert base[2] == pytest.approx(withk[2])


def test_ladder_head_onnx_parity(tmp_path):
    # the exported ONNX head must match the sklearn RiskHead: p_3r == the calibrated 3R rung,
    # expected_reach == the area-under-survival ranking. This is the deploy-correctness gate.
    import onnxruntime as ort
    from futures_foundation.finetune.risk_head import (
        RiskHead, export_ladder_head_onnx, reach_labels)
    Xtr, ytr, ktr = _data(2500, seed=14)
    Xev, yev, kev = _data(600, seed=15)
    Xtr2 = Xtr[:, :, 0]; Xev2 = Xev[:, :, 0]              # [N, D] (risk head takes flat features)
    rh = RiskHead(head='mlp', calibrate=True, mlp_batch=256, max_iter=200).fit(Xtr2, ktr, Xtr2, ktr)
    path = str(tmp_path / 'ladder_head.onnx')
    ti = list(TARGETS).index(3.0)
    export_ladder_head_onnx(rh._heads, TARGETS, Xtr2.shape[1], path, primary_ti=ti)
    sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    res = dict(zip([o.name for o in sess.get_outputs()],
                   sess.run(None, {'input': Xev2.astype(np.float32)})))
    assert np.abs(res['p_3r'][:, 0] - rh.predict_rung(Xev2, ti)).max() < 1e-4
    assert np.abs(res['expected_reach'][:, 0] - rh.predict_stats(Xev2)['exp_reach']).max() < 1e-4


def test_expected_reach_weights_area():
    # the linear reduction weights must reproduce the Riemann area: all-survival=1 -> E[R]=max target
    from futures_foundation.finetune.risk_head import expected_reach_weights
    w = expected_reach_weights(TARGETS)
    assert np.isclose(w.sum(), TARGETS[-1])               # surv all-ones -> expected_reach == 8R
    assert np.isclose(w[0], TARGETS[0] + 0.5 * (TARGETS[1] - TARGETS[0]))


def test_distributional_falls_back_without_keys():
    # rank= set but keys not threaded -> single-head path (never crashes on a missing ladder)
    Xtr, ytr, _ = _data(700, seed=11)
    Xval, yval, _ = _data(200, seed=12)
    Xev, yev, _ = _data(200, seed=13)
    clf = _clf(rank='expected_reach', reach_targets=list(TARGETS))
    p_val, p_ev, auc = clf.fit_predict(Xtr, ytr, Xval, yval, Xev, seed=0)   # no keys
    assert p_val.shape == (200,) and p_ev.shape == (200,)
    assert getattr(clf, '_risk_head', None) is None       # ladder not fit
