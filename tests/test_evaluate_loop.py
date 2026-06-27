"""Tests for ev.run's loop wiring: the `loop=True` path delegates to the
overfit-driven training loop, and the single-pass primitive stays the default
(so A/B harnesses and train_loop's own single passes are unaffected).
"""
from futures_foundation.pipeline import evaluate as ev
from futures_foundation.pipeline import train_loop as TLmod


# ---- _should_loop gate -----------------------------------------------------
def test_should_loop_requires_all_three():
    assert ev._should_loop(True, True, True) is True
    assert ev._should_loop(False, True, True) is False     # not requested
    assert ev._should_loop(True, False, True) is False     # 3-class labeler
    assert ev._should_loop(True, True, False) is False      # custom head


class _Lab:
    n_classes = 2


def _stub_env(monkeypatch):
    # keep ev.run out of any heavy backbone work
    monkeypatch.setattr(ev.backbone, 'stamp_active_source', lambda **k: None)


def test_loop_true_delegates_and_returns_records(monkeypatch):
    _stub_env(monkeypatch)
    res = dict(params={}, source='default',
               final=dict(records=[1, 2, 3], generalizes=True, all_pass=True),
               history=[])
    seen = {}

    def fake_tl(labeler, **kw):
        seen.update(kw)
        return res
    monkeypatch.setattr(TLmod, 'train_loop', fake_tl)

    out = ev.run(_Lab(), loop=True, seeds=(0,), max_folds=5)
    assert out == [1, 2, 3]                  # records when not return_verdict
    assert seen['seeds'] == (0,)
    assert seen['loop_max_folds'] == 5 and seen['final_max_folds'] == 5


class _LabVol:
    n_classes = 2
    def volume_contexts(self, keys): return []      # hook present (passes guard)


def test_loop_threads_volume_embed_params(monkeypatch):
    """use_volume_embed / volume_pool must reach train_loop (so loop-tuned
    validation actually uses the volume embed)."""
    _stub_env(monkeypatch)
    seen = {}

    def fake_tl(labeler, **kw):
        seen.update(kw)
        return dict(params={}, source='default',
                    final=dict(records=[], generalizes=True, all_pass=True),
                    history=[])
    monkeypatch.setattr(TLmod, 'train_loop', fake_tl)

    ev.run(_LabVol(), loop=True, seeds=(0,), use_volume_embed=True, volume_pool='reg')
    assert seen['use_volume_embed'] is True
    assert seen['volume_pool'] == 'reg'


def test_loop_volume_embed_defaults_off(monkeypatch):
    """Default is OFF (existing models unchanged)."""
    _stub_env(monkeypatch)
    seen = {}
    monkeypatch.setattr(TLmod, 'train_loop',
                        lambda labeler, **kw: (seen.update(kw) or
                        dict(final=dict(records=[]))))
    ev.run(_Lab(), loop=True, seeds=(0,))
    assert seen['use_volume_embed'] is False


def test_loop_true_returns_full_verdict_when_requested(monkeypatch):
    _stub_env(monkeypatch)
    res = dict(params={'max_depth': 3}, source='tuned',
               final=dict(records=[], generalizes=True, all_pass=True), history=[])
    monkeypatch.setattr(TLmod, 'train_loop', lambda labeler, **kw: res)
    out = ev.run(_Lab(), loop=True, return_verdict=True)
    assert out is res                        # full train_loop result dict


def test_loop_does_not_delegate_for_custom_head(monkeypatch):
    # custom head_factory => single pass; train_loop must NOT be called.
    _stub_env(monkeypatch)
    monkeypatch.setattr(TLmod, 'train_loop',
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("train_loop should not be called")))
    # _should_loop is the gate; a custom head means default_head=False
    assert ev._should_loop(True, True, False) is False
