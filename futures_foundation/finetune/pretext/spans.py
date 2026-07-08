"""SpanBERT-style SPAN masking — contiguous multi-bar corruption (torch-free, unit-testable).

Random single-bar masking lets the model interpolate a hole from its neighbors — a LOCAL skill.
Masking a CONTIGUOUS RUN of bars forces it to model how a MOVE DEVELOPS across bars — the span
phenomenon trend formation actually is (SpanBERT's win on span reasoning, mapped to candles).
Used by span-ELECTRA: the generator must fake a plausible multi-bar sequence (much harder) and
the encoder must detect the fake span (much richer signal than single-bar tells).

Span lengths ~ Geometric(1/mean_span) clipped to max_span (the SpanBERT recipe); spans are
sampled until ~ratio of the window is covered. Adjacent/overlapping spans may merge into longer
runs — that's fine (SpanBERT's do too); max_span clips the SAMPLED span, not the merged run.
"""
import numpy as np


def sample_span_mask(rng, batch, seq, ratio, mean_span=4.0, max_span=10):
    """Per-sample contiguous-span mask [batch, seq] bool covering ~ratio of each row (>=1 bar
    always). rng = np.random.Generator (deterministic). mean_span = geometric mean span length;
    max_span clips each sampled span."""
    target = max(1, int(round(ratio * seq)))
    p = 1.0 / max(mean_span, 1.0)
    m = np.zeros((batch, seq), bool)
    for b in range(batch):
        covered = 0
        for _ in range(64):                                # guard: never spin forever
            L = int(min(max_span, max(1, rng.geometric(p))))
            s = int(rng.integers(0, seq))
            e = min(seq, s + L)
            covered += int(np.count_nonzero(~m[b, s:e]))
            m[b, s:e] = True
            if covered >= target:
                break
        if not m[b].any():                                 # unreachable, but keep the invariant
            m[b, 0] = True
    return m


def local_turns(h, l, w=3):
    """Turning bars of one window: indices whose HIGH is the max (a peak) or whose LOW is the min
    (a trough) of the ±w-bar neighborhood — the swing/reversal points where trends turn, i.e. the
    same structural event a pivot-entry strategy trades. h, l: [seq]; returns sorted unique indices.
    Placement-only helper (the mask may use full-window context — BERT masking is bidirectional;
    the DOWNSTREAM consumer still reads strictly causal windows)."""
    h, l = np.asarray(h, float), np.asarray(l, float)
    seq = len(h)
    out = []
    for t in range(seq):
        lo, hi = max(0, t - int(w)), min(seq, t + int(w) + 1)
        if h[t] >= h[lo:hi].max() or l[t] <= l[lo:hi].min():
            out.append(t)
    return np.asarray(out, int)


def sample_turn_span_mask(rng, H, L, ratio, mean_span=4.0, max_span=10, turn_w=3, turn_bias=0.85):
    """TURN-BIASED span mask (the salient-span move, mapped to market structure): contiguous spans
    ~Geometric(1/mean_span) clipped to max_span, covering ~ratio of each row — but each span is
    CENTERED ON A DETECTED TURN (local_turns of that window's H/L) with prob turn_bias, else placed
    uniformly (keeps some generic coverage). Masking the turning regions forces the model to learn
    HOW REAL TURNS DEVELOP — the fakeout-vs-reversal skill — while the objective stays pure SSL
    (reconstruct / discriminate the data itself; no labels).

    H, L: [batch, seq]. Returns (mask [batch, seq] bool, turn_cov float) where turn_cov = fraction
    of masked bars within ±turn_w of a turn (diagnostic: how turn-focused the corruption really is).
    rng = np.random.Generator (deterministic)."""
    H, L = np.asarray(H, float), np.asarray(L, float)
    batch, seq = H.shape
    target = max(1, int(round(ratio * seq)))
    p = 1.0 / max(mean_span, 1.0)
    m = np.zeros((batch, seq), bool)
    near_turn = np.zeros((batch, seq), bool)
    for b in range(batch):
        turns = local_turns(H[b], L[b], w=turn_w)
        for t in turns:
            near_turn[b, max(0, t - turn_w):t + turn_w + 1] = True
        covered = 0
        for _ in range(64):                                # guard: never spin forever
            Ln = int(min(max_span, max(1, rng.geometric(p))))
            if len(turns) and rng.random() < turn_bias:    # center the span on a random turn
                c = int(turns[rng.integers(0, len(turns))])
                s = max(0, min(seq - Ln, c - Ln // 2))
            else:                                          # uniform placement (generic coverage)
                s = int(rng.integers(0, seq))
            e = min(seq, s + Ln)
            covered += int(np.count_nonzero(~m[b, s:e]))
            m[b, s:e] = True
            if covered >= target:
                break
        if not m[b].any():                                 # unreachable, but keep the invariant
            m[b, 0] = True
    tot = int(m.sum())
    return m, (float((m & near_turn).sum()) / tot if tot else 0.0)
