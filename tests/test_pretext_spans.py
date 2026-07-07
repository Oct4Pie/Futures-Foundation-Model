"""SpanBERT-style span masking (pretext/spans.py) + its wiring into the electra pretext.

Span mode must be a pure OPTION: span_mean=0 keeps the original bar-ELECTRA byte-identical;
span_mean>0 switches only the corruption PATTERN (contiguous multi-bar spans) — same task, same
network, same loss. These tests lock the sampler's statistics (coverage, CONTIGUITY, min-one,
determinism) and the cfg plumbing (the _base_cfg silent-drop trap).
"""
import numpy as np

from futures_foundation.finetune.pretext.spans import sample_span_mask


def _run_lengths(row):
    """Lengths of contiguous True runs in a 1-D bool array."""
    out, n = [], 0
    for v in row:
        if v:
            n += 1
        elif n:
            out.append(n); n = 0
    if n:
        out.append(n)
    return out


# ---------------------------------------------------------------- sampler statistics
def test_shape_dtype_and_min_one():
    m = sample_span_mask(np.random.default_rng(0), 256, 64, 0.2, mean_span=4)
    assert m.shape == (256, 64) and m.dtype == bool
    assert m.any(axis=1).all()                             # >=1 masked bar per row, always


def test_coverage_tracks_ratio():
    m = sample_span_mask(np.random.default_rng(1), 512, 64, 0.2, mean_span=4)
    assert 0.15 < m.mean() < 0.32                          # ~ratio (spans overshoot a little)


def test_spans_are_contiguous_not_scattered():
    # THE SpanBERT property: multi-bar runs, not single-bar salt-and-pepper. Mean run length
    # must be well above 1 (bar-mode masks average ~1.15); sampled spans respect max_span.
    rng = np.random.default_rng(2)
    m = sample_span_mask(rng, 512, 64, 0.2, mean_span=4, max_span=10)
    runs = [r for row in m for r in _run_lengths(row)]
    assert np.mean(runs) > 2.0                             # genuinely contiguous
    assert max(runs) <= 20                                 # merged runs bounded (2x max_span slack)


def test_mean_span_knob_changes_contiguity():
    rng = np.random.default_rng(3)
    short = [r for row in sample_span_mask(rng, 256, 64, 0.2, mean_span=2) for r in _run_lengths(row)]
    rng = np.random.default_rng(3)
    long_ = [r for row in sample_span_mask(rng, 256, 64, 0.2, mean_span=8) for r in _run_lengths(row)]
    assert np.mean(long_) > np.mean(short)                 # the knob does what it says


def test_deterministic_with_seed():
    a = sample_span_mask(np.random.default_rng(7), 64, 64, 0.2)
    b = sample_span_mask(np.random.default_rng(7), 64, 64, 0.2)
    assert np.array_equal(a, b)


def test_tiny_ratio_still_masks():
    m = sample_span_mask(np.random.default_rng(4), 32, 64, 0.01, mean_span=4)
    assert m.any(axis=1).all()
    assert m.mean() < 0.2                                  # small target -> small coverage


# ---------------------------------------------------------------- cfg plumbing (silent-drop trap)
def test_base_cfg_keeps_span_knobs():
    from futures_foundation.finetune.ssl import _base_cfg
    cfg = _base_cfg(pretext='electra', span_mean=4.0, span_max=8)
    assert cfg['span_mean'] == 4.0
    assert cfg['span_max'] == 8


def test_span_default_off_is_bar_mode():
    # span_mean defaults to 0 -> the original bar-ELECTRA path (backward compat, byte-identical)
    from futures_foundation.finetune.ssl import _base_cfg
    assert _base_cfg(pretext='electra')['span_mean'] == 0.0


# ---------------------------------------------------------------- mask-recon span wiring (SpanBERT)
def test_mask_trainer_accepts_span_knobs():
    # SpanBERT = the mask/reconstruction pretext with span_mean>0. Signature must accept the knobs
    # (they flow cfg -> loop_ssl -> train_ssl_mask); a plain mask run (span_mean=0) is unaffected.
    import inspect
    from futures_foundation.finetune.pretext._torch.mask import train_ssl_mask
    sig = inspect.signature(train_ssl_mask).parameters
    assert 'span_mean' in sig and 'span_max' in sig
    assert sig['span_mean'].default == 0.0                 # default off = original BERT masking


def test_base_cfg_span_knobs_reach_mask_pretext():
    from futures_foundation.finetune.ssl import _base_cfg
    cfg = _base_cfg(pretext='mask', span_mean=5.0, span_max=12, mask_ratio=0.25)
    assert cfg['span_mean'] == 5.0 and cfg['span_max'] == 12 and cfg['mask_ratio'] == 0.25
