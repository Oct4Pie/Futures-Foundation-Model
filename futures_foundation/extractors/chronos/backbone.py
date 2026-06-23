"""The FFM foundation surface — Chronos-Bolt, frozen, subprocess-isolated.

FFM's foundation model is **pretrained Chronos-Bolt** (`amazon/chronos-bolt-tiny`),
not a from-scratch transformer. This module is the canonical seam every
downstream consumer uses to get foundation embeddings; it was promoted from
`pipelines/chronos/backbone.py` (the proven, live-production seam) when the
from-scratch FFM backbone was retired.

Process contract: the parent process stays **torch-free** (on macOS torch's
libomp and xgboost's libomp segfault in one process). `embed()` /
`embed_bars()` run all torch/Chronos work in an ISOLATED SUBPROCESS
(`futures_foundation/_embed_worker.py`). The torch helpers at the bottom
(`pipeline`, `fresh_model`, `pool`) are used ONLY inside the worker or
explicit torch-side fine-tune scripts — never import them in a parent that
also runs XGBoost.

Backbone resolution (the 2026-05-19 wiring-gap lesson): `$CHRONOS_FT_CKPT`
points at a local fine-tuned checkpoint; unset means frozen vanilla HF
weights. Call `stamp_active_source()` at the start of every entry-point so
the active backbone is impossible to miss.
"""
import os
from pathlib import Path

import numpy as np

MODEL = 'amazon/chronos-bolt-tiny'
D_MODEL = 256                          # bolt-tiny (torch-free constant)
CTX = 128                              # canonical context bars (log-close)
_PIPE = None
_PRISTINE = None                       # cloned pretrained state_dict

# Repo root when running from a checkout (temp/ scan, worker cwd).
_ROOT = Path(__file__).resolve().parents[3]   # extractors/chronos/backbone.py -> repo root


def pooled_dim(pool: str = 'mean') -> int:
    """Width of embed()'s output for a pool mode (torch-free). 'meanreg'
    concatenates mean + [REG] → 2*D_MODEL; 'mean'/'reg' → D_MODEL."""
    base = 2 * D_MODEL if pool == 'meanreg' else D_MODEL
    return base + (2 if os.environ.get('CHRONOS_POOL_LOCSCALE') == '1' else 0)


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
    but CHRONOS_FT_CKPT is unset, prints the exact export command."""
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
        candidates = _find_unused_finetunes(_ROOT)
        if candidates:
            print(f"\n  ⚠ Found {len(candidates)} unused fine-tune "
                  f"checkpoint(s) on disk:")
            for p in candidates[:3]:
                try:
                    rel = p.relative_to(_ROOT)
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


def embed(contexts, batch=64, pool='mean', return_loc_scale=False):
    """FROZEN batched embeddings, computed in an isolated subprocess so the
    torch-free parent can run XGBoost safely. Deterministic; no grad.
    contexts: iterable of equal-length 1-D causal windows. Imports NO torch in
    the parent.

    pool (Tier-1 lever): 'mean' (legacy, byte-identical) | 'reg' ([REG] token)
      | 'meanreg' (concat → 2*D_MODEL).
    return_loc_scale: also return [N,2] (loc, scale) = the window mean/std that
      instance_norm strips from the embedding (Chronos's magnitude blind spot;
      feed log(scale) to XGBoost as the volatility feature).
    -> [N, D] (or ([N,D], [N,2]) if return_loc_scale).
    """
    import sys
    import subprocess
    import tempfile

    if pool not in ('mean', 'reg', 'meanreg'):
        raise ValueError(f"pool must be 'mean'|'reg'|'meanreg', got {pool!r}")
    dim = 2 * D_MODEL if pool == 'meanreg' else D_MODEL
    X = np.asarray(contexts, dtype=np.float32)
    if len(X) == 0:
        empty = np.zeros((0, dim), np.float32)
        return (empty, np.zeros((0, 2), np.float32)) if return_loc_scale else empty
    with tempfile.TemporaryDirectory() as d:
        ip, op = os.path.join(d, 'in.npy'), os.path.join(d, 'out.npy')
        ls = os.path.join(d, 'ls.npy') if return_loc_scale else None
        np.save(ip, X)
        env = dict(os.environ,
                   PYTHONPATH=str(_ROOT) + os.pathsep
                   + os.environ.get('PYTHONPATH', ''))
        cmd = [sys.executable, '-m',
               'futures_foundation.extractors.chronos._worker',
               ip, op, str(batch), pool]
        if ls:
            cmd.append(ls)
        r = subprocess.run(cmd, cwd=str(_ROOT), env=env,
                           capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(op):
            raise RuntimeError(
                "chronos embed worker failed:\n" + r.stderr[-2000:])
        E = np.load(op)
        return (E, np.load(ls)) if return_loc_scale else E


def embed_bars(close, indices, ctx: int = CTX, batch: int = 64,
               pool='mean', return_loc_scale=False):
    """Foundation embeddings for decision bars in a close series.

    The canonical downstream entry-point: builds the log-close context
    window ending at each decision index (bars <= t — strictly causal) and
    embeds them in one subprocess call. `pool` / `return_loc_scale` are the
    Tier-1 levers (see embed()). `ctx` can be raised (e.g. 512) for more history.

    close:   1-D array-like of close prices (chronological).
    indices: decision-bar integer positions; each must be >= ctx-1.
    -> float32 [len(indices), D] (or (E, loc_scale) if return_loc_scale).
    """
    c = np.asarray(close, dtype=np.float64)
    idx = np.asarray(indices, dtype=np.int64)
    if len(idx) == 0:
        dim = pooled_dim(pool)
        empty = np.zeros((0, dim), np.float32)
        return (empty, np.zeros((0, 2), np.float32)) if return_loc_scale else empty
    if idx.min() < ctx - 1 or idx.max() >= len(c):
        raise ValueError(
            f"indices must lie in [{ctx - 1}, {len(c) - 1}] "
            f"(got [{idx.min()}, {idx.max()}])")
    lp = np.log(c)
    windows = np.stack([lp[i - ctx + 1:i + 1] for i in idx]).astype(np.float32)
    return embed(windows, batch=batch, pool=pool, return_loc_scale=return_loc_scale)


# ---------------------------------------------------------------------------
# Torch-side helpers — ONLY for the subprocess worker / explicit fine-tune
# scripts. Never import these from a parent that also runs XGBoost.
# ---------------------------------------------------------------------------

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
    causal log-price context (bars <= decision t — the caller's contract).

    With CHRONOS_POOL_LOCSCALE=1, appends bolt's own loc+scale (instance-norm
    de-norm terms it otherwise discards) → dim 256→258. Default off =
    byte-identical 256-d (live/vanilla untouched). Must match the FT pool."""
    h, ls, _emb, mask = m.encode(context=ctx)
    w = mask.unsqueeze(-1).to(h.dtype)
    pooled = (h * w).sum(1) / w.sum(1).clamp(min=1.0)
    if os.environ.get('CHRONOS_POOL_LOCSCALE') == '1':
        import torch
        loc, scale = ls
        pooled = torch.cat([pooled,
                            loc.reshape(loc.shape[0], -1).to(pooled.dtype),
                            scale.reshape(scale.shape[0], -1).to(pooled.dtype)], dim=1)
    return pooled
