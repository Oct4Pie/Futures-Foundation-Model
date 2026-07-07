"""Multi-scale OHLCV window primitives — form higher-timeframe candles from a base stream.

The generic building block for MULTI-TF EMBEDDING: a strategy on one timeframe (e.g. 3min) hands
the frozen encoder the SAME pivot at coarser scales (e.g. 15min = factor 5) by aggregating its own
bars — no second data stream, no cross-TF alignment, no closed-candle lookahead trap. Candles are
ANCHORED at the query bar (the last aggregated candle ENDS exactly at bar i), so every scale is
strictly causal and maximally fresh (a clock-aligned 15min candle can be up to 14 minutes stale at
the query moment; an anchored one never is). Statistically an anchored candle is a phase-shifted
clock candle — in-distribution for an encoder SSL-trained on native multi-TF bars.

Aggregation semantics per factor-bar block (identical to a native resample):
    O = first open, H = max high, L = min low, C = last close, V = sum volume
"""
import numpy as np


def aggregate_ohlcv_window(o, h, l, c, v, i, seq, factor=1):
    """Causal [5, seq] OHLCV window ENDING at bar i, at `factor`x the base timeframe.

    Takes the last seq*factor base bars up to and including i and groups each consecutive
    factor-bar block into one candle (O=first, H=max, L=min, C=last, V=sum), anchored so the
    final candle closes at bar i. factor=1 = the raw window (identity). float32, NaN-safe.
    Raises ValueError if bar i has fewer than seq*factor bars of history (fail fast — the
    caller's CTX guard must cover seq*max(factor))."""
    n = int(seq) * int(factor)
    if i + 1 < n:
        raise ValueError(f'insufficient history: bar {i} has {i + 1} bars, needs {n} '
                         f'(seq={seq} x factor={factor})')
    sl = slice(i - n + 1, i + 1)
    if factor == 1:
        w = np.stack([o[sl], h[sl], l[sl], c[sl], v[sl]])
        return np.nan_to_num(np.asarray(w, np.float32))
    O = np.asarray(o[sl], np.float64).reshape(seq, factor)
    H = np.asarray(h[sl], np.float64).reshape(seq, factor)
    L = np.asarray(l[sl], np.float64).reshape(seq, factor)
    C = np.asarray(c[sl], np.float64).reshape(seq, factor)
    V = np.asarray(v[sl], np.float64).reshape(seq, factor)
    w = np.stack([O[:, 0], H.max(axis=1), L.min(axis=1), C[:, -1], V.sum(axis=1)])
    return np.nan_to_num(w.astype(np.float32))


def multi_scale_ohlcv_window(o, h, l, c, v, i, seq, factors=(1,)):
    """Multi-scale stack: one [5, seq] window per aggregation factor, concatenated along the
    CHANNEL axis -> [5*len(factors), seq]. With a channel-independent frozen encoder (each channel
    embedded separately, concat) this yields [emb_scale1 | emb_scale2 | ...] downstream — the
    multi-TF embedding — with zero encoder/harness changes. factors=(1,) = plain window."""
    return np.concatenate([aggregate_ohlcv_window(o, h, l, c, v, i, seq, f) for f in factors],
                          axis=0)
