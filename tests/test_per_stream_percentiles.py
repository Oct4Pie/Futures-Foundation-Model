"""PER-STREAM PERCENTILE SCALE — the 0-100 deploy score's contract.

WHY IT EXISTS: a calibrated proba is anchored to its LABEL'S BASE RATE, so its absolute scale
moves whenever the label moves (4R head base 23% -> centres ~0.24; strict-6R base 14.9% ->
centres ~0.15). A floor tuned for one takes LITERALLY NOTHING from the other — that is the live
starvation, reproducible on demand. The scale also differs BY STREAM at a fixed rate (measured:
15 takes/day needs 0.145 on ES@3min but 0.186 on NQ@1min).

These tests pin the properties a bot depends on: per-stream, monotone, base-rate invariant,
generic in the stream key, and small-stream safety.
"""
import numpy as np
import pytest

from futures_foundation.finetune.produce import per_stream_percentiles


def _keys(spec):
    """spec: {stream: n} -> a key list shaped like the harness's ((sid, bar, ...) tuples)."""
    out = []
    for s, n in spec.items():
        out += [(s, i) for i in range(n)]
    return out


def test_keyed_per_stream_not_pooled():
    rng = np.random.default_rng(0)
    keys = _keys({'NQ@1min': 1000, 'ES@3min': 1000})
    p = np.concatenate([rng.normal(0.30, 0.02, 1000), rng.normal(0.10, 0.02, 1000)])
    t = per_stream_percentiles(p, keys)
    assert set(t) == {'NQ@1min', 'ES@3min'}
    # each stream's p50 tracks ITS OWN centre — a pooled table would land both near 0.20
    assert t['NQ@1min']['p50'] == pytest.approx(0.30, abs=0.02)
    assert t['ES@3min']['p50'] == pytest.approx(0.10, abs=0.02)


def test_percentiles_are_monotone():
    rng = np.random.default_rng(1)
    keys = _keys({'NQ@1min': 2000})
    t = per_stream_percentiles(rng.random(2000), keys)['NQ@1min']
    vals = [t[f'p{q:g}'] for q in (10, 25, 50, 75, 90, 95, 97, 99, 99.5)]
    assert all(a <= b for a, b in zip(vals, vals[1:])), vals


def test_invariant_to_label_base_rate():
    """THE POINT. Two heads whose probas differ ONLY by scale (a 4R-like head centred at 0.24 vs a
    6R-like head at 0.15) must produce the SAME percentile RANKING. A fixed proba floor cannot do
    this — 0.44 is meaningful for one and takes nothing from the other."""
    rng = np.random.default_rng(2)
    keys = _keys({'NQ@1min': 4000})
    base = rng.random(4000)
    p_4r = 0.10 + 0.30 * base                          # centred ~0.25
    p_6r = 0.05 + 0.20 * base                          # centred ~0.15, SAME ordering
    t4 = per_stream_percentiles(p_4r, keys)['NQ@1min']
    t6 = per_stream_percentiles(p_6r, keys)['NQ@1min']
    assert t4['p90'] > t6['p90']                       # absolute cutoffs differ...
    # ...but the SAME pivots sit above p90 in both -> one score_floor serves both models
    top4 = set(np.where(p_4r >= t4['p90'])[0])
    top6 = set(np.where(p_6r >= t6['p90'])[0])
    assert len(top4 ^ top6) == 0


def test_p90_selects_about_a_tenth():
    rng = np.random.default_rng(3)
    keys = _keys({'NQ@1min': 5000})
    p = rng.random(5000)
    t = per_stream_percentiles(p, keys)['NQ@1min']
    assert (p >= t['p90']).mean() == pytest.approx(0.10, abs=0.01)
    assert (p >= t['p50']).mean() == pytest.approx(0.50, abs=0.02)


def test_small_streams_skipped():
    """A quantile off a handful of rows is noise a bot would deploy on. <200 val rows -> no entry
    (the caller sees a MISSING stream and must decide, rather than trusting a fabricated cutoff)."""
    keys = _keys({'NQ@1min': 500, 'ZB@15min': 40})
    t = per_stream_percentiles(np.random.default_rng(4).random(540), keys)
    assert 'NQ@1min' in t and 'ZB@15min' not in t


def test_generic_in_the_stream_key():
    """FFM must not know what a stream IS. Groups by the opaque key the labeler emits — no '@'
    parsing, no ticker/timeframe assumptions."""
    keys = [('anything', i) for i in range(300)] + [('42', i) for i in range(300)]
    t = per_stream_percentiles(np.random.default_rng(5).random(600), keys)
    assert set(t) == {'anything', '42'}


def test_degenerate_inputs_return_empty_not_garbage():
    assert per_stream_percentiles(np.array([]), []) == {}
    assert per_stream_percentiles(np.array([0.1, 0.2]), None) == {}
    # length mismatch must NOT silently mis-align probas to keys
    assert per_stream_percentiles(np.array([0.1, 0.2, 0.3]), _keys({'NQ@1min': 2})) == {}
