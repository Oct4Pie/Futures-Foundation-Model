"""Chronos-Bolt backbone seam — load, pristine reset, causal pooling.

The ONLY module that touches the Chronos library. `embed()` runs the
torch/Chronos work in an ISOLATED SUBPROCESS: on macOS torch's libomp and
xgboost's libomp segfault in one process, so the parent (which also runs
XGBoost) must stay torch-free. The pretrained-load / pool / fresh_model
helpers below are torch and used ONLY inside the subprocess worker or the
legacy in-process NN fine-tune path — never by the XGBoost eval parent.
"""
import os
from pathlib import Path

import numpy as np

MODEL = 'amazon/chronos-bolt-tiny'
D_MODEL = 256                          # bolt-tiny (torch-free constant)
_PIPE = None
_PRISTINE = None                       # cloned pretrained state_dict


def active_source() -> str:
    """Resolve which Chronos checkpoint embed() will load. Parent-safe
    (no torch import). Returns the explicit CHRONOS_FT_CKPT path if set,
    else the frozen HF model name."""
    return os.environ.get('CHRONOS_FT_CKPT') or MODEL


def _find_unused_finetunes(root: Path) -> list:
    """Scan temp/ for unused fine-tune checkpoints on disk. Returns paths
    of any directory containing model.safetensors under the canonical
    fine-tune output tree. Used by stamp_active_source() to warn when a
    fine-tune exists but is silently ignored (the 2026-05-19 wiring gap)."""
    found = []
    for pat in ('temp/chronos_*_ft/out/run-*/checkpoint-final',
                'temp/chronos_*_ft/out/run-*/checkpoint-[0-9]*'):
        for p in root.glob(pat):
            if (p / 'model.safetensors').exists():
                found.append(p)
    return sorted(set(found))


def stamp_active_source(context: str = '') -> str:
    """Loud one-line stamp of which backbone embed() will load. Call at
    the start of every training/eval entry-point so a wiring gap (env
    var unset, fine-tuned ckpt sitting unused) is impossible to miss.

    Also scans temp/ for unused fine-tune checkpoints — if one exists
    but CHRONOS_FT_CKPT is unset, prints the exact export command. This
    is the assumption-vs-reality gap that bit us on 2026-05-19."""
    src = active_source()
    # HF model ids contain '/' too ('amazon/chronos-bolt-tiny'), so '/' alone
    # is not a "local" signal. Use absolute-path OR filesystem-exists.
    is_local = os.path.isabs(src) or os.path.exists(src)
    tag = '🧪 FINE-TUNED (local)' if is_local else '❄️  FROZEN (vanilla HF)'
    ctx = f" [{context}]" if context else ''
    print(f"\n{'='*72}")
    print(f"  CHRONOS BACKBONE{ctx}: {tag}")
    print(f"  source: {src}")
    if not is_local:
        root = Path(__file__).resolve().parents[2]
        candidates = _find_unused_finetunes(root)
        if candidates:
            print(f"\n  ⚠ Found {len(candidates)} unused fine-tune "
                  f"checkpoint(s) on disk:")
            for p in candidates[:3]:
                try:
                    rel = p.relative_to(root)
                except ValueError:
                    rel = p
                print(f"    - {rel}")
            print(f"  ⚠ CHRONOS_FT_CKPT is UNSET — vanilla backbone will "
                  f"be used.")
            print(f"  ⚠ To use the fine-tuned backbone instead, abort and "
                  f"export first:")
            print(f"    export CHRONOS_FT_CKPT={candidates[-1]}")
    print(f"{'='*72}\n", flush=True)
    return src


def pipeline():
    global _PIPE
    if _PIPE is None:
        import torch
        from chronos import BaseChronosPipeline
        _PIPE = BaseChronosPipeline.from_pretrained(
            MODEL, device_map='cpu', dtype=torch.float32)
    return _PIPE


def _model():
    p = pipeline()
    return getattr(p, 'inner_model', None) or p.model


def d_model():
    return _model().config.d_model


def fresh_model():
    """The backbone reset to pretrained weights (independent fine-tunes)."""
    global _PRISTINE
    m = _model()
    if _PRISTINE is None:
        _PRISTINE = {k: v.detach().clone()
                     for k, v in m.state_dict().items()}
    else:
        m.load_state_dict(_PRISTINE)
    return m


def pool(m, ctx):
    """Masked-mean pool of the encoder hidden states. ctx: [B,L] tensor of
    causal log-price context (bars <= decision t — the caller's contract)."""
    h, _ls, _emb, mask = m.encode(context=ctx)
    w = mask.unsqueeze(-1).to(h.dtype)
    return (h * w).sum(1) / w.sum(1).clamp(min=1.0)


def embed(contexts, batch=64):
    """FROZEN batched embeddings, computed in an isolated subprocess so the
    torch-free parent can run XGBoost safely (macOS OpenMP segfault if torch
    + xgboost share a process). Deterministic; pretrained weights, no grad.
    contexts: iterable of equal-length 1-D causal windows. -> float32
    [N, D_MODEL]. This function imports NO torch in the parent."""
    import os
    import sys
    import subprocess
    import tempfile

    X = np.asarray(contexts, dtype=np.float32)
    if len(X) == 0:
        return np.zeros((0, D_MODEL), np.float32)
    root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory() as d:
        ip, op = os.path.join(d, 'in.npy'), os.path.join(d, 'out.npy')
        np.save(ip, X)
        env = dict(os.environ,
                   PYTHONPATH=str(root) + os.pathsep
                   + os.environ.get('PYTHONPATH', ''))
        r = subprocess.run(
            [sys.executable, '-m', 'pipelines.chronos._embed_worker',
             ip, op, str(batch)],
            cwd=str(root), env=env, capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(op):
            raise RuntimeError(
                "chronos embed worker failed:\n" + r.stderr[-2000:])
        return np.load(op)
