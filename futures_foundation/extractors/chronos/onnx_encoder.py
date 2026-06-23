"""Export the Chronos-Bolt encoder (+ mean-pool) to ONNX in a torch-only
subprocess — NO xgboost in this process, so it sidesteps the macOS libomp
collision (same isolation discipline as the embed worker). Ported from the
proven algoTraderAI export: nanmean->mean patch (real inputs never NaN),
the dynamo exporter, and the Gather-int64 patch. Resolves the checkpoint via
the FFM resolver so 'chronos_bolt_ft' (or any checkpoints/<name>) works.

CLI: python3 -m futures_foundation.extractors.chronos.onnx_encoder \
        <chronos_ckpt> <ctx_window> <out.onnx>
Exits 0 iff parity passes.
"""
import sys
from pathlib import Path

import numpy as np

PARITY_TOL_EMBED = 1e-4


def _wrapper(src):
    """torch.nn.Module: Chronos encode() + mean-pool over tokens -> [B, d]."""
    import torch.nn as nn
    import torch
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained(src, device_map='cpu', dtype=torch.float32)

    class _W(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, context):
            e, _ls, *_ = self.m.encode(context=context)   # (B, n_patches+1, d)
            return e.mean(1)                               # mean-pool -> (B, d)

    return _W(pipe.model).eval()


def _patch_gather_int64(onnx_path):
    """Cast(int64) before any Gather whose indices are a float tensor (a
    dynamo-trace quirk ORT rejects). No-op when indices are already ints."""
    import onnx
    from onnx import helper, TensorProto
    m = onnx.load(onnx_path)
    init_dtypes = {i.name: i.data_type for i in m.graph.initializer}
    new, cid, patched = [], 0, 0
    for n in m.graph.node:
        if (n.op_type == 'Gather' and len(n.input) >= 2
                and init_dtypes.get(n.input[1]) not in (TensorProto.INT32, TensorProto.INT64)):
            cid += 1
            co = f'__gc_{cid}'
            new.append(helper.make_node('Cast', [n.input[1]], [co],
                                        to=TensorProto.INT64, name=f'__pc_{cid}'))
            pn = onnx.NodeProto(); pn.CopyFrom(n); pn.input[1] = co
            new.append(pn); patched += 1
            continue
        new.append(n)
    del m.graph.node[:]
    m.graph.node.extend(new)
    onnx.save(m, onnx_path)
    return patched


def export(src, ctx_window, out_path):
    import torch
    _ofn, _omt = torch.nanmean, torch.Tensor.nanmean
    torch.nanmean = lambda x, *a, **k: torch.mean(x, *a, **k)
    torch.Tensor.nanmean = lambda self, *a, **k: torch.mean(self, *a, **k)
    try:
        w = _wrapper(src)
        dummy = torch.zeros((2, ctx_window), dtype=torch.float32)
        batch = torch.export.Dim('batch')
        out = Path(out_path)
        if out.exists():
            out.unlink()
        torch.onnx.export(w, (dummy,), str(out), opset_version=18, dynamo=True,
                          dynamic_shapes={'context': {0: batch}})
    finally:
        torch.nanmean = _ofn
        torch.Tensor.nanmean = _omt
    return _patch_gather_int64(str(out))


def parity(src, ctx_window, out_path, n=20, seed=42):
    import onnxruntime as ort
    import torch
    sess = ort.InferenceSession(str(out_path), providers=['CPUExecutionProvider'])
    name = sess.get_inputs()[0].name
    rng = np.random.default_rng(seed)
    prices = 100 * np.exp(np.cumsum(rng.standard_normal((n, ctx_window)) * 0.01, axis=1))
    lc = np.log(prices).astype(np.float32)
    w = _wrapper(src)
    with torch.no_grad():
        ref = w(torch.tensor(lc)).numpy()
    got = sess.run(None, {name: lc})[0]
    return float(np.abs(ref - got).max())


def main():
    from . import backbone
    ck, ctx, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    src = backbone.resolve_ckpt(ck) or ck
    print(f"[onnx-encoder] export {ck!r} (resolved {src}) ctx={ctx} -> {out}", flush=True)
    patched = export(src, ctx, out)
    d = parity(src, ctx, out)
    ok = d < PARITY_TOL_EMBED
    print(f"[onnx-encoder] parity max|Δ|={d:.2e} (tol {PARITY_TOL_EMBED:.0e}) "
          f"{'✓' if ok else '✗'}  (patched {patched} Gather node(s))", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
