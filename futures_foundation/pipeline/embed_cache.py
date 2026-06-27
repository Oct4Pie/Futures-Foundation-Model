"""Disk-backed embedding cache — embed each context ONCE, reuse across runs.

The walk-forward eval re-embeds millions of contexts every run. This caches the
frozen embeddings to disk and reloads them, re-embedding ONLY what is new or
changed.

CORRECTNESS (why this is safe to reuse):
  - The cache directory is keyed by a RECIPE SIGNATURE = (backbone checkpoint,
    pool mode, return-shape on/off, context length, embed dim). Change the recipe
    → different signature → the old cache is never read (auto-invalidation).
  - Within a signature, each entry is keyed by (item_id, bar_index) AND stored
    with a CONTENT HASH of the context window. On load, the hash is re-checked
    against the live context; if the underlying data shifted (e.g. new bars
    appended remap the indices) the hash mismatches and that bar is re-embedded.
  So a cache HIT is only ever used when the exact recipe AND the exact context
  bytes match — otherwise it falls back to a fresh embed. No stale embeddings.

Scope: mean-pool, no loc_scale (matches the eval's cache path). The full embed
(including the return-shape block appended by backbone.embed) is what's cached,
since that is what the head consumes.
"""
from pathlib import Path
import hashlib

import numpy as np


def signature(pool='mean', channel=None):
    """(short_hash, human_string) of the embedding recipe — the cache namespace.

    `channel` namespaces a non-price embed (e.g. 'volume') into its own cache so
    it can't collide with the price embed for the same (item, bar). channel=None
    or 'price' keeps the ORIGINAL signature (existing price caches stay valid)."""
    from futures_foundation.extractors.chronos import backbone
    s = (f"src={backbone.active_source()}|pool={pool}"
         f"|rs={backbone._return_shape_on()}|ctx={backbone.CTX}"
         f"|d={backbone.D_MODEL}")
    if channel and channel != 'price':
        s += f"|ch={channel}"
    return hashlib.sha1(s.encode()).hexdigest()[:12], s


def _hash_ctx(ctx):
    return int.from_bytes(
        hashlib.blake2b(np.ascontiguousarray(ctx, np.float32).tobytes(),
                        digest_size=8).digest(), 'little')


def _fname(cdir, item):
    return cdir / (str(item).replace('/', '_').replace('@', '_') + '.npz')


def embed_with_cache(contexts, keys, cache_dir, pool='mean', verbose=True,
                     channel=None):
    """Return flat embeddings [N, D] for `contexts` (aligned to `keys`), using a
    disk cache. keys[i] = (item_id, bar_index, ...). Cache HITs are content-hash
    verified; everything else is embedded fresh via backbone.embed and stored.
    `channel` ('volume', ...) namespaces a non-price embed into its own cache."""
    from futures_foundation.extractors.chronos import backbone
    sigh, _ = signature(pool, channel)
    cdir = Path(cache_dir) / sigh
    cdir.mkdir(parents=True, exist_ok=True)
    N = len(keys)

    hashes = np.fromiter((_hash_ctx(c) for c in contexts), np.uint64, N)

    # load per-item caches: item -> {bar_index: (hash, emb_row)}
    cache = {}
    items = {k[0] for k in keys}
    for item in items:
        f = _fname(cdir, item)
        cmap = {}
        if f.exists():
            z = np.load(f, allow_pickle=False)
            zi, zh, ze = z['idx'], z['hash'], z['emb']
            for t in range(len(zi)):
                cmap[int(zi[t])] = (np.uint64(zh[t]), ze[t])
        cache[item] = cmap

    hit = np.zeros(N, bool)
    for n, k in enumerate(keys):
        ent = cache[k[0]].get(int(k[1]))
        if ent is not None and ent[0] == hashes[n]:
            hit[n] = True
    miss = np.flatnonzero(~hit)
    if verbose:
        print(f"[embed-cache] sig={sigh}  hits={int(hit.sum()):,}/{N:,}  "
              f"misses={len(miss):,}", flush=True)

    if len(miss):
        Em = backbone.embed([contexts[n] for n in miss], pool=pool)
        D = Em.shape[1]
    else:
        D = next(iter(next(iter(cache.values())).values()))[1].shape[0]

    flat = np.empty((N, D), np.float32)
    hit_rows = np.flatnonzero(hit)
    if len(hit_rows):
        flat[hit_rows] = np.stack(
            [cache[keys[n][0]][int(keys[n][1])][1] for n in hit_rows])
    for j, n in enumerate(miss):
        flat[n] = Em[j]
        cache[keys[n][0]][int(keys[n][1])] = (hashes[n], Em[j])

    # persist only the streams that gained entries
    for item in {keys[n][0] for n in miss}:
        cmap = cache[item]
        idx = np.array(sorted(cmap), np.int64)
        hsh = np.array([int(cmap[i][0]) for i in idx], np.uint64)
        emb = np.stack([cmap[i][1] for i in idx]).astype(np.float32)
        np.savez(_fname(cdir, item), idx=idx, hash=hsh, emb=emb)
    return flat
