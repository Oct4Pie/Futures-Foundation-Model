"""Torch-free unit tests for pipeline.onnx_export — the XGBoost head export +
parity path. The Chronos encoder export is torch (a subprocess) and is mocked
here; its real parity is covered by the produce(export_onnx=True) integration."""
import numpy as np
import pytest

pytest.importorskip("onnxmltools")
pytest.importorskip("onnxruntime")

from futures_foundation.pipeline.head_xgb import XGBHead, XGBRiskHead
from futures_foundation.pipeline import onnx_export


def _tiny_bundle(n=12, with_risk=False):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((250, n)).astype(np.float32)
    y = (X[:, 0] + 0.5 * rng.standard_normal(250) > 0).astype(int)
    b = {'signal_head': XGBHead(n_classes=2, n_estimators=25, max_depth=3).fit(X, y, seed=0),
         'risk_head': None, 'feat_dim': n, 'chronos_ckpt': 'vanilla', 'ctx_window': 128}
    if with_risk:
        b['risk_head'] = XGBRiskHead(n_estimators=25, max_depth=3).fit(X, np.abs(y) + 1.0, seed=0)
    return b


class _FakeProc:
    returncode = 0
    stdout = "[onnx-encoder] parity max|Δ|=1e-06 ✓"
    stderr = ""


@pytest.fixture
def _mock_encoder(monkeypatch):
    import subprocess
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: _FakeProc())


def test_signal_head_onnx_parity(tmp_path, _mock_encoder):
    b = _tiny_bundle()
    res = onnx_export.export_bundle_onnx(b, tmp_path / 'm.joblib', verbose=False)
    path, delta, ok = res['signal_head']
    assert ok is True and delta < onnx_export.PARITY_TOL      # real parity passes
    assert (tmp_path / 'm_signal_head.onnx').exists()
    assert 'risk_head' not in res                              # none in bundle
    assert res['encoder'][2] is True                           # mocked subprocess


def test_risk_head_onnx_parity(tmp_path, _mock_encoder):
    b = _tiny_bundle(with_risk=True)
    res = onnx_export.export_bundle_onnx(b, tmp_path / 'm.joblib', verbose=False)
    assert res['signal_head'][2] is True
    assert res['risk_head'][2] is True and res['risk_head'][1] < onnx_export.PARITY_TOL
    assert (tmp_path / 'm_risk_head.onnx').exists()


def test_encoder_failure_surfaces(tmp_path, monkeypatch):
    import subprocess

    class _Bad:
        returncode = 1
        stdout = ""
        stderr = "boom"
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: _Bad())
    res = onnx_export.export_bundle_onnx(_tiny_bundle(), tmp_path / 'm.joblib', verbose=False)
    assert res['signal_head'][2] is True       # head still fine
    assert res['encoder'][2] is False          # encoder failure surfaced, not swallowed
