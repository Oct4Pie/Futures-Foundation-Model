"""Basic unit tests for the generic Chronos finetune framework.

Pure-logic tests run always; tests that touch the Chronos-Bolt backbone
skip cleanly if the library/weights are unavailable. The framework's
non-negotiables pinned here: deterministic fine-tune, pristine backbone
reset (independent runs), correct pooling shape, and the honest-ruler
orchestration over the StrategyLabeler protocol.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from pipelines.chronos import finetune as ft
from pipelines.chronos import evaluate, strategy

try:
    import chronos                                    # noqa: F401
    import torch                                      # noqa: F401
    _CHRONOS = True
except Exception:                                     # pragma: no cover
    _CHRONOS = False

chronos_only = pytest.mark.skipif(
    not _CHRONOS, reason='chronos-forecasting / torch unavailable')


# ---- pure logic (no backbone) -------------------------------------------

def test_ftconfig_defaults():
    c = ft.FTConfig()
    assert (c.steps, c.batch, c.n_classes) == (150, 16, 3)
    assert c.lr_head > c.lr_back > 0


def test_stats_formatting():
    assert evaluate._stats([]) == 'trades=0'
    s = evaluate._stats(np.array([1.0, -1.0, 2.0]))
    assert 'trades=3' in s and 'sumR=+2.0' in s


def test_strategy_protocol_is_duck_typed():
    class Ok:
        n_classes = 3
        def calendar(self): ...
        def build(self, lo, hi, ts): ...
        def evaluate(self, k, p): ...
    assert isinstance(Ok(), strategy.StrategyLabeler)
    assert not isinstance(object(), strategy.StrategyLabeler)


# ---- backbone-dependent --------------------------------------------------

@chronos_only
def test_backbone_loads_pools_and_resets():
    import torch
    from pipelines.chronos import backbone
    assert backbone.d_model() == 256
    m = backbone.fresh_model()
    ctx = torch.tensor(np.random.default_rng(0).standard_normal(
        (2, 48)).astype('float32'))
    out = backbone.pool(m, ctx)
    assert tuple(out.shape) == (2, 256)
    # pristine reset: perturb a param, fresh_model() must restore it
    p = next(m.parameters())
    orig = p.detach().clone()
    with torch.no_grad():
        p.add_(1.0)
    backbone.fresh_model()
    assert torch.allclose(next(m.parameters()).detach(), orig)


@chronos_only
def test_finetune_deterministic_and_bounded():
    rng = np.random.default_rng(1)
    C = [rng.standard_normal(48).astype('float32') for _ in range(16)]
    Y = rng.integers(0, 3, 16)
    cfg = ft.FTConfig(steps=2, batch=4)
    m1, h1 = ft.train(C, Y, cfg, seed=0)
    m2, h2 = ft.train(C, Y, cfg, seed=0)
    p1, p2 = ft.predict(m1, h1, C), ft.predict(m2, h2, C)
    assert np.array_equal(p1, p2)                 # same seed -> identical
    assert len(p1) == 16 and set(np.unique(p1)) <= {0, 1, 2}


class _DummyLabeler:
    """Synthetic StrategyLabeler — no real strategy, exercises the seam."""
    n_classes = 3

    def __init__(self):
        ts = pd.date_range('2021-01-01', periods=24 * 150, freq='h',
                            tz='UTC')
        rng = np.random.default_rng(0)
        self._cal = pd.concat(
            [pd.DataFrame({'item_id': tk, 'timestamp': ts,
                           'target': np.cumsum(rng.standard_normal(len(ts)))})
             for tk in ('A', 'B')], ignore_index=True)

    def calendar(self):
        return self._cal

    def build(self, lo, hi, test_start):
        rng = np.random.default_rng(int(pd.Timestamp(lo).value) % 2**31)
        n = 80
        C = [rng.standard_normal(48).astype('float32') for _ in range(n)]
        return C, rng.integers(0, 3, n), list(range(n))

    def evaluate(self, keys, preds):
        rng = np.random.default_rng(len(keys))
        return np.array([rng.standard_normal()
                         for p in preds if p != 0])


@chronos_only
def test_evaluate_run_orchestrates_honest_ruler():
    res = evaluate.run(_DummyLabeler(),
                       cfg=ft.FTConfig(steps=2, batch=4, n_classes=3),
                       seeds=(0,), max_folds=1)
    assert len(res) == 1
    r = res[0]
    assert {'fold', 'seed', 'REAL', 'SHUFFLE', 'RANDOM'} <= set(r)
    assert all(isinstance(r[k], np.ndarray)
               for k in ('REAL', 'SHUFFLE', 'RANDOM'))
