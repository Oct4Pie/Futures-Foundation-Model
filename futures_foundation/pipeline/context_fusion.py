"""Context-heads fusion seam — additive, default-off.

Wires `futures_foundation.context.ContextHeads` outputs (named `ctx_*`
market-understanding features) into the signal pipeline's feature matrix.
With no bundle configured, behavior is byte-identical to the pre-seam
pipeline (embedding + strategy features only).

Activation mirrors the CHRONOS_FT_CKPT wiring-guard pattern (the
2026-05-19 lesson — config by env var MUST be loudly stamped):
  - explicit `context_heads_path=` argument, or
  - `export CONTEXT_HEADS_BUNDLE=<path/to/heads.joblib>`
`resolve_heads()` prints a loud stamp of what loaded (active heads +
training metadata) or, when unset, stays silent.

emb_mode (the pre-registered A/B arms):
  'both'  — embedding + ctx_* heads            (arm B)
  'emb'   — embedding only (heads ignored)     (arm A / baseline)
  'heads' — ctx_* heads replace the embedding  (arm C)

LEAK GUARD: heads are trained on pre-HEADS_CUTOFF bars and frozen. Any
signal training that consumes ctx_* features must use rows at/after
HEADS_CUTOFF — `enforce_cutoff()` is the single chokepoint both
evaluate.run() and produce.train() call.
"""
import os

import numpy as np
import pandas as pd

from futures_foundation.context import HEADS_CUTOFF

ENV_VAR = 'CONTEXT_HEADS_BUNDLE'
EMB_MODES = ('both', 'emb', 'heads')


def resolve_heads(path=None, verbose=True):
    """Load a ContextHeads bundle from `path` or $CONTEXT_HEADS_BUNDLE.
    Returns None (silently) when neither is set — the default-off path."""
    p = path or os.environ.get(ENV_VAR)
    if not p:
        return None
    from futures_foundation.context import ContextHeads
    heads = ContextHeads.load(p)
    if verbose:
        print(f"\n{'=' * 72}")
        print(f"  CONTEXT HEADS: 🧠 ACTIVE")
        print(f"  source: {p}")
        print(f"  emits : {heads.active_names or '(none passed gate!)'}")
        if heads.meta.get('inputs') == 'emb+ff68':
            print(f"  ENRICHED bundle (inputs=emb+ff68): needs the 68-feature "
                  f"library at decision bars")
            print(f"  via the labeler's ff68_at(keys) hook (fuse ff68=...). "
                  f"emb-only bundles need no hook.")
        if heads.meta:
            for k in ('cutoff', 'train_span', 'tickers', 'tfs', 'git_sha'):
                if k in heads.meta:
                    print(f"  {k:<6}: {heads.meta[k]}")
        print(f"{'=' * 72}\n", flush=True)
    return heads


def fuse(E, extra=None, heads=None, emb_mode='both', ff68=None):
    """Build the feature matrix: [embedding?] + [ctx_* heads?] + [strategy
    features?]. heads=None -> byte-identical legacy behavior regardless of
    emb_mode. -> float32 [N, feat_dim].

    ENRICHED bundles (inputs='emb+ff68') need the 68-feature library AT the
    decision bars: pass `ff68` ([N, 68], canonical order). The heads then run on
    [embedding | ff68]. ff68 is only required when ctx_* are actually emitted
    (emb_mode in {'both','heads'}); for 'emb' (baseline) it is ignored.
    """
    if emb_mode not in EMB_MODES:
        raise ValueError(f"emb_mode must be one of {EMB_MODES}, "
                         f"got {emb_mode!r}")
    E = np.asarray(E, np.float32)
    enriched = heads is not None and heads.meta.get('inputs') == 'emb+ff68'
    emit_ctx = heads is not None and emb_mode in ('heads', 'both')
    if enriched and emit_ctx and ff68 is None:
        raise ValueError(
            "enriched context bundle (inputs='emb+ff68') needs the 68-feature "
            "library at the decision bars — pass ff68=[N,68] (e.g. via the "
            "labeler's ff68_at(keys) hook), or use an emb-only bundle.")
    parts = []
    if heads is None or emb_mode in ('emb', 'both'):
        parts.append(E)
    if emit_ctx:
        head_in = (np.hstack([E, np.asarray(ff68, np.float32)])
                   if enriched else E)
        parts.append(heads.transform(head_in))
    if extra is not None:
        extra = np.asarray(extra, np.float32)
        if extra.size:
            parts.append(extra.reshape(len(E), -1))
    return (np.hstack(parts) if len(parts) > 1 else parts[0]).astype(
        np.float32)


def enforce_cutoff(heads, train_start, what='training window'):
    """LEAK GUARD: with heads active, signal training may not start before
    HEADS_CUTOFF (the heads saw those bars). Raises on violation."""
    if heads is None:
        return
    ts = pd.Timestamp(train_start)
    if ts.tzinfo is None:
        ts = ts.tz_localize('UTC')
    if ts < HEADS_CUTOFF:
        raise ValueError(
            f"context-heads leak guard: {what} starts {ts} — before "
            f"HEADS_CUTOFF {HEADS_CUTOFF}. The heads trained on those "
            f"bars; signal training on them double-dips. Restrict the "
            f"run to >= {HEADS_CUTOFF.date()}.")
