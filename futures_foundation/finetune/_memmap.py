"""Memmap featurization — train on FULL data without holding the feature array in RAM.

featurize_to_memmap writes the [N, C, seq] windows to a disk-backed .npy in CHUNKS
(parent RAM = one chunk, not all N), so a 1M-window set (~5.6GB) never materializes
in memory. The worker then mmaps it and trains per-batch (rows paged on demand), so
worker RAM = one batch. Standardize stats are computed in one streaming pass (or on a
sample) and applied per-batch — no standardized full copy.

torch-free. Used by produce/loop for large/full-data runs; small runs can still use
the in-RAM array path.
"""
import os

import numpy as np


def concat_memmaps(parts, out_path):
    """Concatenate part memmaps [(path, n), ...] into one [sum n, C, seq] memmap,
    one part at a time (bounded RAM), deleting each part after. Returns (out_path,
    shape). Enables per-(ticker,tf) featurize: each stream writes a part, then we
    stitch them without ever holding all bars/features at once."""
    parts = [(p, n) for p, n in parts if n > 0]
    if not parts:
        raise ValueError("no parts to concatenate")
    m0 = np.load(parts[0][0], mmap_mode='r')
    C, seq = int(m0.shape[1]), int(m0.shape[2])
    del m0
    total = sum(n for _, n in parts)
    out = np.lib.format.open_memmap(out_path, mode='w+', dtype=np.float32,
                                    shape=(total, C, seq))
    off = 0
    for p, n in parts:
        src = np.load(p, mmap_mode='r')
        out[off:off + n] = src[:]
        off += n
        del src
        try:
            os.remove(p)
        except OSError:
            pass
    out.flush()
    del out
    return out_path, (total, C, seq)


def featurize_to_memmap(clf, labeler, keys, path, chunk=2000):
    """Featurize `keys` in chunks straight to a disk memmap at `path`. Returns
    (path, shape). RAM cost = one chunk of windows + the labeler's bars."""
    n = len(keys)
    if n == 0:
        raise ValueError("no keys to featurize")
    x0 = np.asarray(clf.featurize(labeler, keys[:1]), np.float32)   # learn (C, seq)
    C, seq = int(x0.shape[1]), int(x0.shape[2])
    mm = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=(n, C, seq))
    mm[0] = x0[0]
    for s in range(1, n, chunk):
        e = min(s + chunk, n)
        mm[s:e] = np.asarray(clf.featurize(labeler, keys[s:e]), np.float32)
    mm.flush()
    del mm
    return path, (n, C, seq)


def memmap_standardize_stats(path, sample=30000, seed=0):
    """Per-channel (mu, sd) over a row sample of the memmap (one light pass).
    The serve path / worker standardizes per-batch with these — no full copy."""
    mm = np.load(path, mmap_mode='r')
    n, C, seq = mm.shape
    if n > sample:
        idx = np.sort(np.random.default_rng(seed).choice(n, sample, replace=False))
        block = np.asarray(mm[idx], np.float32)
    else:
        block = np.asarray(mm, np.float32)
    flat = block.transpose(0, 2, 1).reshape(-1, C)
    mu = flat.mean(0).astype(np.float32)
    sd = (flat.std(0) + 1e-6).astype(np.float32)
    del mm
    return mu, sd
