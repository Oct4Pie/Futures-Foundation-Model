"""THE PRODUCTION PIVOT TRIGGER on the public indicators surface (2026-07-16).

Why this exists: a consumer (the bot) re-implemented the detector and fed the 4R model ~2x the
trained pivot rate — out-of-distribution setups whose probas are extrapolations (flat deciles,
bulk expectancy ~0). The fix is ONE importable certified implementation; these tests pin that
the re-exports ARE the certified functions and the convenience wrapper reproduces the training
labeler's exact trigger + stop math.
"""
import numpy as np

from futures_foundation import indicators as I


def _bars(n=400, seed=0):
    rng = np.random.default_rng(seed)
    c = 100 + np.cumsum(rng.standard_normal(n) * 0.5)
    h = c + rng.random(n) * 0.6
    l = c - rng.random(n) * 0.6
    o = np.concatenate([[c[0]], c[:-1]])
    return o, h, l, c


def test_reexports_are_the_certified_implementations_not_copies():
    """Identity, not equality: indicators must expose the SAME objects as the certified modules
    (the primitives-divergence trap: a fork drifts silently)."""
    from futures_foundation.primitives import detection as D
    from futures_foundation.pipeline import _primitives as P
    assert I.detect_fractal_pivots is D.detect_fractal_pivots
    assert I.detect_fractal_zigzag_pivots is D.detect_fractal_zigzag_pivots
    assert I.detect_atr_zigzag_pivots is D.detect_atr_zigzag_pivots
    assert I.compute_atr is P.compute_atr


def test_production_params_are_frozen():
    assert I.PIVOT_TREND_TRIGGER == {'k': 2, 'min_leg_atr': 1.25, 'atr_period': 20}
    assert I.PIVOT_TREND_STOP_BUFFER_ATR == 0.05


def test_candidates_match_the_raw_detector_with_production_params():
    """pivot_trend_candidates = detect_fractal_zigzag_pivots(k=2, leg=1.25, atr=20) exactly —
    same pivots, same confirms, same directions (no extra filtering, no missed ones except
    NaN-ATR confirms)."""
    o, h, l, c = _bars()
    ref = I.detect_fractal_zigzag_pivots(o, h, l, c, live_edge=True, **I.PIVOT_TREND_TRIGGER)
    atr = I.compute_atr(h, l, c, 20)
    ref = [p for p in ref if p['confirm'] < len(atr)
           and np.isfinite(atr[p['confirm']]) and atr[p['confirm']] > 0]
    got = I.pivot_trend_candidates(o, h, l, c, live_edge=True)
    assert [(g['confirm'], g['direction'], g['origin']) for g in got] \
        == [(int(p['confirm']), int(p['direction']), int(p['origin'])) for p in ref]
    assert len(got) > 5                                   # non-degenerate on random walk bars


def test_stop_math_matches_the_training_labeler():
    """stop = extreme -/+ 0.05*ATR(confirm) — CONFIRM-bar ATR (the labeler's _fixed_outcomes
    convention), origin-bar extreme. Long stop below the low; short stop above the high."""
    o, h, l, c = _bars(seed=3)
    atr = I.compute_atr(h, l, c, 20)
    for g in I.pivot_trend_candidates(o, h, l, c):
        ext = l[g['origin']] if g['direction'] == 1 else h[g['origin']]
        assert g['extreme_px'] == float(ext)
        buf = 0.05 * atr[g['confirm']]
        want = ext - buf if g['direction'] == 1 else ext + buf
        assert abs(g['stop_px'] - want) < 1e-12
        if g['direction'] == 1:
            assert g['stop_px'] < ext                     # long stop strictly below the pivot low
        else:
            assert g['stop_px'] > ext


def test_kept_stream_alternates_directions():
    """The zigzag keep-FIRST invariant: no two consecutive kept pivots share a direction
    (the bot-side acceptance check #6)."""
    o, h, l, c = _bars(seed=7)
    dirs = [g['direction'] for g in I.pivot_trend_candidates(o, h, l, c)]
    assert all(a != b for a, b in zip(dirs, dirs[1:]))
    assert set(dirs) <= {1, -1}


def test_live_edge_emits_the_newest_confirm_batch_mode_does_not():
    """live_edge=True (streaming) may emit a pivot confirming on the LAST closed bar; batch
    labeling mode (live_edge=False) requires an entry bar to exist. live-mode candidates are
    a superset whose extras confirm only at the array edge."""
    o, h, l, c = _bars(seed=11)
    live = I.pivot_trend_candidates(o, h, l, c, live_edge=True)
    batch = I.pivot_trend_candidates(o, h, l, c, live_edge=False)
    keys = lambda rows: {(g['confirm'], g['direction']) for g in rows}
    assert keys(batch) <= keys(live)
    for cf, _ in keys(live) - keys(batch):
        assert cf >= len(c) - 1                           # extras only at the live edge


def test_contract_carries_the_trigger_spec_from_the_labeler():
    """The deploy contract must DECLARE its candidate universe: produce reads TRIGGER_SPEC off
    the labeler into contract['trigger'], so a bundle self-describes which pivots to feed it
    (the misconfig this whole module exists to prevent becomes machine-checkable)."""
    import inspect
    from futures_foundation.finetune import produce
    src = inspect.getsource(produce)
    assert src.count("'trigger': getattr(") == 2          # streamed + legacy contract builders
    assert "TRIGGER_SPEC" in src
