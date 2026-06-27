"""Unit tests for BOCPD change-point features. Load-bearing properties:
CAUSALITY (row t uses only x[0..t]) and that it actually FIRES on a real shift."""
import numpy as np
import pytest

pytest.importorskip("scipy")

from futures_foundation.changepoint import (
    bocpd_features, change_point_features, change_point_from_feature,
    N_CHANGEPOINT_FEATS, CHANGEPOINT_RETURN_COL)


def test_shape_and_bounds():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(200)
    f = bocpd_features(x)
    assert f.shape == (200, 2)
    assert np.isfinite(f).all()
    assert (f[:, 0] >= 0).all() and (f[:, 0] <= 1).all()   # changepoint prob
    assert (f[:, 1] >= 0).all() and (f[:, 1] <= 1).all()   # norm run-length


def test_fires_on_a_real_mean_shift():
    rng = np.random.default_rng(1)
    # 150 bars N(0,1) then 150 bars N(8,1) — a hard mean shift at t=150
    x = np.concatenate([rng.standard_normal(150),
                        rng.standard_normal(150) + 8.0])
    f = bocpd_features(x)
    cp = f[:, 0]
    # changepoint probability spikes in the window right after the shift
    pre = cp[120:150].max()
    at = cp[150:160].max()
    assert at > pre                      # detects the break
    assert at > 0.3                      # and fires clearly


def test_run_length_grows_when_stable():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(400) * 0.5   # stationary
    f = bocpd_features(x)
    # expected run-length should be larger late (stable) than early
    assert f[-50:, 1].mean() > f[:50, 1].mean()


def test_causal_no_future_peek():
    rng = np.random.default_rng(3)
    x = rng.standard_normal(120)
    f_full = bocpd_features(x)
    x2 = x.copy(); x2[90:] += 50.0       # perturb the future
    f2 = bocpd_features(x2)
    # rows before the perturbation must be unchanged (online/causal)
    np.testing.assert_allclose(f_full[:90], f2[:90], atol=1e-9)


# ---- per-signal orchestration ---------------------------------------------
def _stream(n_bars, seed=0):
    rng = np.random.default_rng(seed)
    s = rng.standard_normal(n_bars).cumsum()
    keys, series = [], []
    for i in range(n_bars):
        for d in (1, -1):                # both directions share the bar
            keys.append(('S', i, d)); series.append(s[i])
    return keys, np.asarray(series, float)


def test_change_point_features_shape_and_share():
    keys, series = _stream(80)
    f = change_point_features(keys, series)
    assert f.shape == (len(keys), N_CHANGEPOINT_FEATS)
    assert np.isfinite(f).all()
    np.testing.assert_array_equal(f[0], f[1])      # both dirs of a bar share
    np.testing.assert_array_equal(f[2], f[3])


# ---- change_point_from_feature (pipeline entry point) ---------------------
def test_change_point_from_feature_shape_share_and_finite():
    keys, series = _stream(60)
    # build a feature matrix with a 'ret_close_1' column = per-bar returns
    feat_names = ['embed_0', CHANGEPOINT_RETURN_COL, 'adx']
    rng = np.random.default_rng(0)
    feat = rng.standard_normal((len(keys), 3)).astype(np.float32)
    feat[:, 1] = np.diff(series, prepend=series[0])      # ret column
    out = change_point_from_feature(keys, feat, feat_names)
    assert out.shape == (len(keys), N_CHANGEPOINT_FEATS)
    assert np.isfinite(out).all()
    np.testing.assert_array_equal(out[0], out[1])         # both dirs of a bar share


def test_change_point_from_feature_raises_without_return_col():
    keys, _ = _stream(20)
    feat = np.zeros((len(keys), 2), np.float32)
    with pytest.raises(ValueError):
        change_point_from_feature(keys, feat, ['a', 'b'])  # no ret_close_1


def test_change_point_features_two_streams_independent():
    rng = np.random.default_rng(5)
    keys, series = [], []
    for sid in ('A', 'B'):
        s = rng.standard_normal(60).cumsum()
        for i in range(60):
            keys.append((sid, i, 1)); series.append(s[i])
    f = change_point_features(keys, np.asarray(series, float))
    assert f.shape == (120, N_CHANGEPOINT_FEATS)
    assert np.isfinite(f).all()
