"""Basic unit tests for the generic Chronos finetune framework.

This process must stay TORCH-FREE: torch's libomp and xgboost's libomp
segfault in one process on macOS. So we never `import torch`/`chronos`
here — availability is probed via find_spec (no import), embedding runs in
a subprocess (backbone.embed), and the in-process torch tests (legacy NN
fine-tune, raw pool/reset) are gated behind CHRONOS_TORCH_TESTS=1 to be run
in their own isolated process.
"""
import importlib.util
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from pipelines.chronos import finetune as ft     # torch is lazy inside it
from pipelines.chronos import evaluate, strategy
from pipelines.chronos.head_xgb import XGBHead

# find_spec does NOT import the module -> pytest process stays torch-free.
_CHRONOS = (importlib.util.find_spec('chronos') is not None
            and importlib.util.find_spec('torch') is not None)

chronos_only = pytest.mark.skipif(
    not _CHRONOS, reason='chronos-forecasting / torch unavailable')

# The repo's FFM tests import torch into the shared `pytest tests/`
# process. xgboost in that same process segfaults (macOS OpenMP). So
# in-process-xgboost tests are gated and run in their OWN process:
#   CHRONOS_ISOLATED=1 pytest tests/test_chronos_framework.py
iso_only = pytest.mark.skipif(
    os.environ.get('CHRONOS_ISOLATED') != '1',
    reason='in-process xgboost clashes with FFM torch in the combined '
           'suite; run: CHRONOS_ISOLATED=1 pytest '
           'tests/test_chronos_framework.py')

# torch in-process conflicts with xgboost's OpenMP; run these isolated:
#   CHRONOS_TORCH_TESTS=1 pytest -k <name>
torch_inproc = pytest.mark.skipif(
    os.environ.get('CHRONOS_TORCH_TESTS') != '1',
    reason='torch in-process (xgboost OpenMP clash); set '
           'CHRONOS_TORCH_TESTS=1 to run isolated')


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

@torch_inproc
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


@torch_inproc
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


class _DummyLabelerFeat(_DummyLabeler):
    """Adds the optional features() hook — exercises embedding+feature
    fusion (one extra column per decision)."""

    def features(self, keys):
        return np.asarray(keys, np.float32).reshape(-1, 1)


# ---- XGBoost head (in-process xgboost — isolated from FFM torch) ---------

@iso_only
@pytest.mark.parametrize('nc', [2, 3])
def test_xgbhead_deterministic_and_bounded(nc):
    """Binary (nc=2) AND multiclass (nc=3). The binary case caught a real
    regression where forcing objective='multi:softprob' + num_class=2 made
    predict() return a 2-D array — must stay covered."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((60, 16)).astype('float32')
    y = rng.integers(0, nc, 60)
    p1 = XGBHead(nc).fit(X, y, seed=0).predict(X)
    p2 = XGBHead(nc).fit(X, y, seed=0).predict(X)
    assert np.array_equal(p1, p2)                 # same seed -> identical
    assert p1.ndim == 1 and len(p1) == 60         # not 2-D (binary regress)
    assert set(np.unique(p1)) <= set(range(nc))
    proba = XGBHead(nc).fit(X, y, seed=0).predict_proba(X)
    assert proba.shape == (60, nc)
    assert np.allclose(proba.sum(1), 1.0, atol=1e-4)


# ---- frozen embedding + head-agnostic honest ruler -----------------------

@chronos_only
def test_backbone_embed_is_frozen_deterministic():
    from pipelines.chronos import backbone
    C = [np.random.default_rng(k).standard_normal(48).astype('float32')
         for k in range(10)]
    e1 = backbone.embed(C)
    e2 = backbone.embed(C)
    assert e1.shape == (10, 256)
    assert np.array_equal(e1, e2)                 # frozen -> deterministic


@iso_only
@chronos_only
def test_evaluate_run_orchestrates_honest_ruler():
    res = evaluate.run(_DummyLabeler(), seeds=(0,), max_folds=1)
    assert len(res) == 1
    r = res[0]
    assert {'fold', 'seed', 'REAL', 'SHUFFLE', 'RANDOM'} <= set(r)
    assert all(isinstance(r[k], np.ndarray)
               for k in ('REAL', 'SHUFFLE', 'RANDOM'))


@iso_only
@chronos_only
def test_evaluate_run_fuses_optional_features():
    res = evaluate.run(_DummyLabelerFeat(), seeds=(0,), max_folds=1)
    assert len(res) == 1                          # runs with embed+feature
    assert isinstance(res[0]['REAL'], np.ndarray)
