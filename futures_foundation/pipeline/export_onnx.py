"""Convert a Chronos+XGBoost joblib bundle (from produce.py) to ONNX.

First-cut deliverable: THREE ONNX files the bot loads via onnxruntime.
A single-file merge (via onnx.compose) is a follow-up; chained inference
is fine for live in the meantime.

  <stem>_chronos.onnx    Chronos backbone:  [B, ctx_window] log-close → [B, 256]
  <stem>_signal.onnx     XGB signal head:   [B, feat_dim] → (label, proba)
  <stem>_risk.onnx       XGB risk head:     [B, feat_dim] → R̂   (binary labelers only)

Bot pseudocode at inference time:
    emb     = onnx_chronos.run(None, {'context': context_128})    # [1, 256]
    fused   = np.concatenate([emb, hand_craft_features], axis=1)  # [1, 334]
    proba   = onnx_signal.run(None, {'features': fused})[1]       # [1, 2]
    r_hat   = onnx_risk.run(None, {'features': fused})[0]         # [1]
    if proba[0, 1] >= THR:
        tp_r = float(np.clip(0.8 * r_hat[0], 1.5, 8.0))
        place_trade(direction, sl=entry-0.5*atr, tp=entry+tp_r*risk)

Required pip installs (not in default env):
    pip install onnxmltools skl2onnx

torch+xgboost OpenMP collision (macOS): the orchestrator parent imports
neither — each export stage runs in its own subprocess.

Usage:
    python3 -m futures_foundation.pipeline.export_onnx chronos_supertrendchronos_production_20260519.joblib
    python3 -m futures_foundation.pipeline.export_onnx <bundle.joblib> --out-dir models/
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# ---- subprocess phase scripts (run as `python -c "..."` so the parent
#      keeps importing neither torch nor xgboost) ---------------------------

_CHRONOS_EXPORT = '''
import joblib, sys
import numpy as np
import torch
from chronos import BaseChronosPipeline

bundle_path, out_path = sys.argv[1], sys.argv[2]
bundle = joblib.load(bundle_path)
ckpt = bundle.get("chronos_ckpt") or "amazon/chronos-bolt-tiny"
ctx_window = int(bundle.get("ctx_window") or 128)

print(f"  [chronos] loading {ckpt}  (ctx_window={ctx_window})", flush=True)
pipe = BaseChronosPipeline.from_pretrained(
    ckpt, device_map="cpu", dtype=torch.float32)

class EmbedWrap(torch.nn.Module):
    """Wraps pipe.embed() + mean-pool so torch.onnx.export can trace it."""
    def __init__(self, pipe):
        super().__init__()
        self.pipe = pipe
    def forward(self, context):
        emb, _ = self.pipe.embed(context)        # [B, T, D]
        return emb.mean(1)                       # [B, D]

m = EmbedWrap(pipe).eval()
dummy = torch.zeros((1, ctx_window), dtype=torch.float32)
print(f"  [chronos] tracing → {out_path}", flush=True)
with torch.no_grad():
    torch.onnx.export(
        m, dummy, out_path,
        input_names=["context"],
        output_names=["embedding"],
        dynamic_axes={"context": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )

# sanity: re-run via onnxruntime and compare
import onnxruntime as ort
sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
np.random.seed(0)
x = np.random.randn(3, ctx_window).astype(np.float32)
with torch.no_grad():
    ref = m(torch.tensor(x)).cpu().numpy()
out = sess.run(None, {"context": x})[0]
err = float(np.max(np.abs(ref - out)))
print(f"  [chronos] OK  shape={out.shape}  max|onnx-torch|={err:.2e}", flush=True)
if err > 1e-3:
    print(f"  [chronos] WARN: drift larger than 1e-3", flush=True)
'''

_HEADS_EXPORT = '''
import joblib, sys
import numpy as np

try:
    from onnxmltools import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType
except ImportError:
    sys.exit("ERROR: pip install onnxmltools skl2onnx")

bundle_path, sig_path, risk_path = sys.argv[1], sys.argv[2], sys.argv[3]
bundle = joblib.load(bundle_path)
feat_dim = int(bundle["feat_dim"])
sig_xgb  = bundle["signal_head"]._clf
risk_xgb = bundle["risk_head"]._reg if bundle.get("risk_head") else None
print(f"  [heads] feat_dim={feat_dim}  sig={type(sig_xgb).__name__}  "
      f"risk={type(risk_xgb).__name__ if risk_xgb else None}", flush=True)

initial = [("features", FloatTensorType([None, feat_dim]))]
sig_onnx = convert_xgboost(sig_xgb, initial_types=initial, target_opset=17)
with open(sig_path, "wb") as f: f.write(sig_onnx.SerializeToString())

if risk_xgb is not None:
    risk_onnx = convert_xgboost(risk_xgb, initial_types=initial, target_opset=17)
    with open(risk_path, "wb") as f: f.write(risk_onnx.SerializeToString())

# sanity: re-run via onnxruntime and compare
import onnxruntime as ort
np.random.seed(0)
x = np.random.randn(3, feat_dim).astype(np.float32)
ref_proba = sig_xgb.predict_proba(x)
sig_sess = ort.InferenceSession(sig_path, providers=["CPUExecutionProvider"])
out = sig_sess.run(None, {"features": x})
# convention from onnxmltools: outputs are (labels, [{class: proba}, ...])
proba_out = out[1]
if isinstance(proba_out, list):                  # list of dicts -> matrix
    proba_out = np.asarray(
        [[d[k] for k in sorted(d)] for d in proba_out], np.float32)
err = float(np.max(np.abs(ref_proba - proba_out)))
print(f"  [heads/signal] OK  max|onnx-xgb|={err:.2e}", flush=True)
if err > 1e-3:
    print(f"  [heads/signal] WARN: drift larger than 1e-3", flush=True)

if risk_xgb is not None:
    ref_r = risk_xgb.predict(x)
    risk_sess = ort.InferenceSession(risk_path, providers=["CPUExecutionProvider"])
    r_out = risk_sess.run(None, {"features": x})[0].squeeze()
    err = float(np.max(np.abs(ref_r - r_out)))
    print(f"  [heads/risk]   OK  max|onnx-xgb|={err:.2e}", flush=True)
    if err > 1e-3:
        print(f"  [heads/risk]   WARN: drift larger than 1e-3", flush=True)
'''


def _run_phase(label: str, code: str, args: list) -> bool:
    """Run a self-contained phase in a child python; True on success."""
    print(f"\n[{label}] launching subprocess...", flush=True)
    proc = subprocess.run(
        [sys.executable, "-c", code, *args],
        capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH":
             str(Path(__file__).resolve().parents[2])
             + os.pathsep + os.environ.get("PYTHONPATH", "")},
    )
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.returncode != 0:
        print(f"[{label}] FAILED (exit {proc.returncode})", flush=True)
        if proc.stderr:
            print("--- stderr ---")
            print(proc.stderr)
        return False
    return True


def verify(bundle_path: Path, chronos_onnx: Path, signal_onnx: Path,
           risk_onnx: Optional[Path] = None, *, sample_n: int = 200,
           thr: float = 0.65, drift_tol: float = 1e-3) -> dict:
    """End-to-end + decision-parity verification of the exported ONNX
    against the original joblib bundle.

    Three layers — each must pass cleanly before shipping:

      Layer 1: Chronos embed equivalence — chronos.onnx vs backbone.embed
      Layer 2: chained pipeline equivalence — proba(joblib) == proba(onnx)
               and R̂(joblib) == R̂(onnx) within `drift_tol`
      Layer 3: decision parity — for the same N inputs, the set of trades
               taken at `thr` must match, and dynamic-TP values must
               agree to the cent.

    Synthetic inputs (random contexts + random hand-craft) — we're
    testing math correctness, not model accuracy. Real-data behaviour
    is governed by the holdout eval that produce.py already prints.
    """
    import joblib
    import onnxruntime as ort

    bundle_path = Path(bundle_path)
    chronos_onnx = Path(chronos_onnx)
    signal_onnx = Path(signal_onnx)
    risk_onnx = Path(risk_onnx) if risk_onnx else None

    print(f"\n=== END-TO-END VERIFICATION ===")
    print(f"  bundle  : {bundle_path}")
    print(f"  chronos : {chronos_onnx}")
    print(f"  signal  : {signal_onnx}")
    print(f"  risk    : {risk_onnx}")

    b = joblib.load(bundle_path)
    sig_head = b['signal_head']
    risk_head = b.get('risk_head')
    feat_dim = int(b['feat_dim'])
    embed_dim = int(b['embed_dim'])
    ctx_window = int(b['ctx_window'])
    hand_dim = feat_dim - embed_dim

    # --- Generate synthetic inputs --------------------------------------
    rng = np.random.default_rng(0)
    contexts = [rng.standard_normal(ctx_window).astype(np.float32)
                for _ in range(sample_n)]
    hand = rng.standard_normal((sample_n, hand_dim)).astype(np.float32)

    # --- Path A (joblib): backbone.embed (subprocess-isolated torch)
    #     + XGB heads in parent (xgboost; no torch). No collision.
    from futures_foundation.extractors.chronos import backbone
    print(f"\n  Path A (joblib): backbone.embed + signal/risk heads")
    emb_jl = backbone.embed(contexts)
    X_jl = np.hstack([emb_jl, hand]).astype(np.float32)
    proba_jl = sig_head.predict_proba(X_jl)[:, 1]
    risk_jl = (risk_head.predict(X_jl)
               if risk_head is not None else None)

    # --- Path B (ONNX): onnxruntime end-to-end (no torch, no xgboost)
    print(f"  Path B (ONNX):   chronos.onnx + signal.onnx + risk.onnx")
    sess_chronos = ort.InferenceSession(
        str(chronos_onnx), providers=["CPUExecutionProvider"])
    sess_signal = ort.InferenceSession(
        str(signal_onnx), providers=["CPUExecutionProvider"])
    sess_risk = (ort.InferenceSession(
        str(risk_onnx), providers=["CPUExecutionProvider"])
        if risk_onnx and risk_onnx.exists() else None)

    ctx_arr = np.asarray(contexts, np.float32)
    emb_ox = sess_chronos.run(None, {"context": ctx_arr})[0]
    X_ox = np.hstack([emb_ox, hand]).astype(np.float32)
    sig_out = sess_signal.run(None, {"features": X_ox})
    # onnxmltools convention: outputs are (labels, proba). proba can be
    # either a 2-D array or a list of dicts depending on op version.
    proba_raw = sig_out[1]
    if isinstance(proba_raw, list):
        proba_raw = np.asarray(
            [[d[k] for k in sorted(d)] for d in proba_raw], np.float32)
    proba_ox = np.asarray(proba_raw, np.float32)[:, 1]
    risk_ox = (sess_risk.run(None, {"features": X_ox})[0].squeeze()
               if sess_risk is not None else None)

    # --- Compute drifts -------------------------------------------------
    embed_drift = float(np.max(np.abs(emb_jl - emb_ox)))
    proba_drift = float(np.max(np.abs(proba_jl - proba_ox)))
    risk_drift = (float(np.max(np.abs(risk_jl - risk_ox)))
                  if risk_jl is not None and risk_ox is not None else None)

    # --- Decision parity @ thr -----------------------------------------
    take_jl = (proba_jl >= thr)
    take_ox = (proba_ox >= thr)
    decision_match_pct = float((take_jl == take_ox).mean())
    decision_ok = bool((take_jl == take_ox).all())

    tp_drift = None
    if risk_jl is not None and risk_ox is not None:
        tp_jl = np.clip(0.8 * risk_jl, 1.5, 8.0)
        tp_ox = np.clip(0.8 * risk_ox, 1.5, 8.0)
        tp_drift = float(np.max(np.abs(tp_jl - tp_ox)))

    # --- Report ---------------------------------------------------------
    def _mark(x, tol=drift_tol):
        return "✓" if (x is not None and x < tol) else "✗"

    print(f"\n  Sample: {sample_n} synthetic signals  (drift_tol={drift_tol:.0e})")
    print(f"\n  Layer 1 — Chronos embed equivalence:")
    print(f"    max|emb(joblib) - emb(onnx)|     = {embed_drift:.2e}  "
          f"{_mark(embed_drift)}")
    print(f"\n  Layer 2 — chained pipeline equivalence:")
    print(f"    max|proba(joblib) - proba(onnx)| = {proba_drift:.2e}  "
          f"{_mark(proba_drift)}")
    if risk_drift is not None:
        print(f"    max|R̂(joblib)     - R̂(onnx)|    = {risk_drift:.2e}  "
              f"{_mark(risk_drift)}")
    print(f"\n  Layer 3 — decision parity @ thr={thr}:")
    print(f"    joblib_takes={int(take_jl.sum())}  "
          f"onnx_takes={int(take_ox.sum())}  "
          f"match={100*decision_match_pct:.1f}%  "
          f"{'✓' if decision_ok else '✗'}")
    if tp_drift is not None:
        print(f"    dynamic-TP (R-units) agreement: max diff = "
              f"{tp_drift:.2e}R  {_mark(tp_drift)}")

    all_pass = (
        embed_drift < drift_tol
        and proba_drift < drift_tol
        and decision_ok
        and (risk_drift is None or risk_drift < drift_tol)
        and (tp_drift is None or tp_drift < drift_tol)
    )
    if all_pass:
        print(f"\n  ✅ ONNX bundle is equivalent to joblib. Safe to ship.")
    else:
        print(f"\n  ⚠ Drift exceeds {drift_tol:.0e} OR decisions disagree — "
              f"investigate before live deploy.")

    return {
        'embed_drift': embed_drift,
        'proba_drift': proba_drift,
        'risk_drift': risk_drift,
        'decision_match_pct': decision_match_pct,
        'decision_ok': decision_ok,
        'tp_drift': tp_drift,
        'all_pass': all_pass,
    }


def export(bundle_path: Path, out_dir: Path, *,
           run_verify: bool = True) -> dict:
    bundle_path = Path(bundle_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = bundle_path.stem
    chronos_path = out_dir / f"{stem}_chronos.onnx"
    signal_path  = out_dir / f"{stem}_signal.onnx"
    risk_path    = out_dir / f"{stem}_risk.onnx"

    print(f"=== ONNX export | bundle={bundle_path} | out_dir={out_dir} ===")

    ok_c = _run_phase("CHRONOS", _CHRONOS_EXPORT,
                      [str(bundle_path), str(chronos_path)])
    ok_h = _run_phase("HEADS",  _HEADS_EXPORT,
                      [str(bundle_path), str(signal_path), str(risk_path)])

    artifacts = {}
    if ok_c and chronos_path.exists():
        artifacts["chronos"] = str(chronos_path)
    if ok_h and signal_path.exists():
        artifacts["signal"] = str(signal_path)
    if ok_h and risk_path.exists():
        artifacts["risk"] = str(risk_path)

    print(f"\n=== ARTIFACTS ===")
    for k, v in artifacts.items():
        sz = Path(v).stat().st_size / 1e6
        print(f"  {k:8s}  {v}  ({sz:.1f} MB)")
    if len(artifacts) < 2:
        print("\n⚠ Some phases failed — re-run after fixing the errors above.")
        return artifacts

    # End-to-end verification — runs only if both phases produced files
    if run_verify and ok_c and ok_h:
        verify(
            bundle_path, chronos_path, signal_path,
            risk_path if risk_path.exists() else None,
        )
    return artifacts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle", help="joblib bundle from produce.py")
    ap.add_argument("--out-dir", default=".",
                    help="where to write the .onnx files (default: CWD)")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the end-to-end verification step after export")
    args = ap.parse_args()
    export(Path(args.bundle), Path(args.out_dir),
           run_verify=not args.no_verify)


if __name__ == "__main__":
    main()
