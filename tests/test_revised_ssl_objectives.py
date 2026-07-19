import os

import numpy as np
import pytest


torch_test = pytest.mark.skipif(
    os.environ.get('CHRONOS_TORCH_TESTS') != '1',
    reason='torch test — set CHRONOS_TORCH_TESTS=1 (libomp isolation)',
)


@torch_test
def test_structural_targets_preserve_price_scale_and_geometry():
    import torch
    from futures_foundation.finetune.pretext._torch.structure_mask import structural_targets

    close = torch.tensor([[100.0, 101.0, 100.5, 102.0]])
    raw = torch.stack((close - .2, close + .5, close - .7, close,
                       torch.tensor([[10., 20., 15., 30.]])), 1)
    scaled = raw.clone(); scaled[:, :4] *= 10
    one, two = structural_targets(raw), structural_targets(scaled)
    torch.testing.assert_close(one, two, atol=2e-5, rtol=2e-5)
    assert one.shape == (1, 6, 4) and torch.isfinite(one).all()


@torch_test
def test_wall_clock_steps_equalize_elapsed_horizons():
    import torch
    from futures_foundation.finetune.pretext._torch.path import wall_clock_steps

    bar_ns = torch.tensor([60, 3 * 60, 60 * 60], dtype=torch.long) * 1_000_000_000
    steps = wall_clock_steps(bar_ns, (60, 180, 360))
    assert steps.tolist() == [[60, 180, 360], [20, 60, 120], [1, 3, 6]]
    with pytest.raises(ValueError, match='divisible'):
        wall_clock_steps(bar_ns, (61,))


@torch_test
def test_path_targets_do_not_read_beyond_each_horizon():
    import torch
    from futures_foundation.finetune.pretext._torch.path import path_targets

    context_close = torch.linspace(100, 102, 32)[None, :]
    context = torch.stack((context_close, context_close + .3, context_close - .3,
                           context_close, torch.full_like(context_close, 1000)), 1)
    future_close = torch.linspace(102.1, 104, 12)[None, :]
    future = torch.stack((future_close, future_close + .2, future_close - .2,
                          future_close, torch.full_like(future_close, 1100)), 1)
    steps = torch.tensor([[3, 6, 12]])
    first = path_targets(context, future, steps, context_minutes=6,
                         bar_ns=torch.tensor([60_000_000_000]))
    changed = future.clone()
    changed[:, :4, 3:] += 1000
    second = path_targets(context, changed, steps, context_minutes=6,
                          bar_ns=torch.tensor([60_000_000_000]))
    for name in ('log_vol', 'favorable_r', 'adverse_r', 'path_class'):
        torch.testing.assert_close(first[name][:, 0], second[name][:, 0])


@torch_test
def test_path_and_structural_targets_support_negative_prices():
    import torch
    from futures_foundation.finetune.pretext._torch.path import path_targets
    from futures_foundation.finetune.pretext._torch.structure_mask import structural_targets

    close = torch.tensor([[5., 2., -3., -8., -4., 1.]])
    context = torch.stack((close, close + 1., close - 1., close, torch.ones_like(close)), 1)
    future_close = torch.tensor([[-2., -6., 3.]])
    future = torch.stack((future_close, future_close + 1., future_close - 1.,
                          future_close, torch.ones_like(future_close)), 1)
    target = path_targets(
        context, future, torch.tensor([[3]]), context_minutes=3,
        bar_ns=torch.tensor([60_000_000_000]),
    )
    assert all(torch.isfinite(value).all() for value in target.values())
    assert torch.isfinite(structural_targets(context)).all()


@torch_test
def test_path_loss_is_finite_and_quantiles_are_monotone():
    import torch
    from futures_foundation.finetune.pretext._torch.path import (
        _monotone_quantiles, path_loss,
    )
    output = torch.randn(8, 3, 10, requires_grad=True)
    target = {
        'log_vol': torch.rand(8, 3), 'favorable_r': torch.rand(8, 3) * 3,
        'adverse_r': torch.rand(8, 3) * 2,
        'path_class': torch.randint(0, 3, (8, 3)),
    }
    loss, parts = path_loss(output, target)
    loss.backward()
    q = _monotone_quantiles(output.detach()[..., 1:4])
    assert torch.all(q[..., 1:] >= q[..., :-1])
    assert torch.isfinite(loss) and output.grad is not None
    assert set(parts) == {'vol_loss', 'excursion_loss', 'class_loss'}


@torch_test
def test_revised_trainers_smoke_and_emit_encoder_states():
    from futures_foundation.finetune import _ssl_torch as ssl_torch

    rng = np.random.default_rng(4)
    close = 100 + np.cumsum(rng.normal(0, .05, 1800)).astype(np.float32)
    big = np.column_stack((close, close + .2, close - .2, close,
                           rng.integers(100, 1000, len(close)))).astype(np.float32)
    starts = np.arange(0, 1500, dtype=np.int64)
    bounds = np.array([[0, 750], [750, 1500]], np.int64)
    structure, structure_history = ssl_torch.train_ssl_structure_mask(
        big, starts, starts, seq=32, mask_ratio=.25, span_mean=4, span_max=8,
        epochs=1, steps_per_epoch=1, batch=4, device='cpu', verbose=False,
        train_group_bounds=bounds, val_group_bounds=bounds, val_batches=1,
    )
    path, path_history = ssl_torch.train_ssl_path(
        big, starts, starts, seq=32, path_horizons_minutes=(3, 6),
        path_context_minutes=3, path_max_future_bars=6, epochs=1, steps_per_epoch=1,
        batch=4, device='cpu', verbose=False, train_group_bounds=bounds,
        val_group_bounds=bounds, stream_bar_ns=np.array([60, 3 * 60]) * 1_000_000_000,
        objective_row_bounds=np.array([[0, len(big)]], np.int64), val_batches=1,
    )
    assert set(structure) == set(path)
    assert np.isfinite(structure_history[-1]['val_loss'])
    assert np.isfinite(path_history[-1]['val_loss'])
    assert 'class_acc' in path_history[-1]
