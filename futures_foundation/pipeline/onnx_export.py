"""Export a produce() bundle to ONNX for in-process bot inference (no torch /
xgboost daemons; sidesteps the macOS libomp collision). Two parts:

  - signal / risk XGBoost heads -> ONNX (onnxmltools convert_xgboost, in-process)
  - Chronos encoder -> ONNX (torch) via the extractors.chronos.onnx_encoder
    SUBPROCESS — torch must not share a process with xgboost.

Every export is PARITY-CHECKED against the joblib bundle before it's accepted.
Files are written next to the joblib: <stem>_signal_head.onnx,
<stem>_risk_head.onnx (if a risk head exists), <stem>_encoder.onnx.
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

PARITY_TOL = 1e-5          # XGBoost head proba/raw (ULP-level drift floor)


def _extract_proba(outs, ncols):
    """onnxmltools classifier output is [label, proba] or a ZipMap list."""
    for o in outs:
        if getattr(o, 'ndim', 0) == 2 and o.shape[1] == ncols:
            return o
        if isinstance(o, list) and o and isinstance(o[0], dict):
            ks = sorted(o[0])
            return np.array([[d[k] for k in ks] for d in o], np.float32)
    return None


def export_bundle_onnx(bundle, output_path, *, verbose=True, samples=50, seed=42):
    """Export the bundle's heads + encoder to ONNX next to output_path,
    each parity-checked vs the joblib. Returns {name: (path, delta_or_msg, ok)}."""
    from onnxmltools.convert import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType
    import onnxruntime as ort

    stem = Path(output_path).with_suffix('')
    n = int(bundle['feat_dim'])
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((samples, n)).astype(np.float32)
    results = {}

    # ── signal head (XGBClassifier) ──────────────────────────────────────
    sig = Path(f"{stem}_signal_head.onnx")
    onx = convert_xgboost(bundle['signal_head']._clf,
                          initial_types=[('input', FloatTensorType([None, n]))],
                          target_opset=15)
    sig.write_bytes(onx.SerializeToString())
    ref = bundle['signal_head'].predict_proba(X)
    proba = _extract_proba(
        ort.InferenceSession(str(sig), providers=['CPUExecutionProvider'])
        .run(None, {'input': X}), ref.shape[1])
    d = float(np.abs(proba[:, 1] - ref[:, 1]).max()) if proba is not None else float('inf')
    results['signal_head'] = (str(sig), d, d < PARITY_TOL)

    # ── risk head (XGBRegressor, binary labelers only) ───────────────────
    if bundle.get('risk_head') is not None:
        rsk = Path(f"{stem}_risk_head.onnx")
        onx = convert_xgboost(bundle['risk_head']._reg,
                              initial_types=[('input', FloatTensorType([None, n]))],
                              target_opset=15)
        rsk.write_bytes(onx.SerializeToString())
        ref_r = bundle['risk_head']._reg.predict(X)
        raw = (ort.InferenceSession(str(rsk), providers=['CPUExecutionProvider'])
               .run(None, {'input': X})[0].squeeze())
        d = float(np.abs(raw - ref_r).max())
        results['risk_head'] = (str(rsk), d, d < PARITY_TOL)

    # ── Chronos encoder (torch SUBPROCESS — no xgboost in that process) ──
    enc = Path(f"{stem}_encoder.onnx")
    ck = str(bundle['chronos_ckpt'])
    ctx = int(bundle['ctx_window'])
    from ..extractors.chronos import backbone as _bb
    root = str(_bb._ROOT)
    env = dict(os.environ, PYTHONPATH=root + os.pathsep + os.environ.get('PYTHONPATH', ''))
    r = subprocess.run(
        [sys.executable, '-m', 'futures_foundation.extractors.chronos.onnx_encoder',
         ck, str(ctx), str(enc)],
        cwd=root, env=env, capture_output=True, text=True)
    enc_ok = r.returncode == 0
    msg = (r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-400:])
    results['encoder'] = (str(enc), msg, enc_ok)

    if verbose:
        print("\n[onnx] export + parity (vs joblib):")
        for k, (p, d, ok) in results.items():
            print(f"  {('✓' if ok else '✗')} {k:12s} {p}  {d}")
        if not all(ok for _, _, ok in results.values()):
            print("  ⚠ one or more ONNX exports FAILED parity — do NOT ship.")
    return results
