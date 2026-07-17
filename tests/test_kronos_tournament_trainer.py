import os
import numpy as np
import pytest
from types import SimpleNamespace

from futures_foundation.finetune.tournament import MAX_CONTEXT, PARENT_LENGTH
from scripts.train_kronos_tournament import (
    _context_normalized_values, _load_warm_bundle, _parser,
)
from scripts.train_kronos_contrastive import _nt_xent


torch_test = pytest.mark.skipif(
    os.environ.get("CHRONOS_TORCH_TESTS") != "1",
    reason="torch test — set CHRONOS_TORCH_TESTS=1 (libomp isolation)",
)


def test_kronos_normalization_cannot_read_future_scale():
    rng = np.random.default_rng(4)
    raw = rng.normal(size=(2, PARENT_LENGTH, 5)).astype(np.float32)
    raw[:, :, :4] += 100
    raw[:, :, 4] = np.abs(raw[:, :, 4] * 1000 + 5000)
    changed = raw.copy()
    changed[:, MAX_CONTEXT:] *= 100
    first = _context_normalized_values(raw, 5.0)
    second = _context_normalized_values(changed, 5.0)
    np.testing.assert_allclose(first[:, :MAX_CONTEXT], second[:, :MAX_CONTEXT])
    assert not np.allclose(first[:, MAX_CONTEXT:], second[:, MAX_CONTEXT:])
    assert first.shape == (2, PARENT_LENGTH, 6)


def test_kronos_arm_is_explicit_and_defaults_to_small():
    args = _parser().parse_args(["--kronos-repo", "/tmp/k", "--output", "/tmp/x"])
    assert args.arm == "kronos_small"


@torch_test
def test_kronos_staged_warm_bundle_loads_both_backbones(tmp_path):
    import torch
    tokenizer = torch.nn.Linear(2, 2)
    predictor = torch.nn.Linear(2, 2)
    expected_tokenizer = {key: torch.full_like(value, 2) for key, value in tokenizer.state_dict().items()}
    expected_predictor = {key: torch.full_like(value, 3) for key, value in predictor.state_dict().items()}
    args = SimpleNamespace(
        tokenizer_id="tok", tokenizer_revision="tok-rev",
        model_id="pred", model_revision="pred-rev",
    )
    path = tmp_path / "stage1.pt"
    torch.save({
        "schema_version": "ffm_kronos_tournament_bundle_v1",
        "stage": "stage1_reconstruction",
        "tokenizer": {"id": "tok", "revision": "tok-rev"},
        "predictor": {"id": "pred", "revision": "pred-rev"},
        "tokenizer_state": expected_tokenizer,
        "predictor_state": expected_predictor,
    }, path)
    parent = _load_warm_bundle(path, tokenizer, predictor, args)
    assert parent["stage"] == "stage1_reconstruction"
    assert len(parent["sha256"]) == 64
    for actual, expected in zip(tokenizer.state_dict().values(), expected_tokenizer.values()):
        torch.testing.assert_close(actual, expected)
    for actual, expected in zip(predictor.state_dict().values(), expected_predictor.values()):
        torch.testing.assert_close(actual, expected)


@torch_test
def test_kronos_staged_warm_bundle_rejects_identity_drift(tmp_path):
    import torch
    tokenizer = torch.nn.Linear(1, 1)
    predictor = torch.nn.Linear(1, 1)
    path = tmp_path / "bad.pt"
    torch.save({
        "schema_version": "ffm_kronos_tournament_bundle_v1",
        "tokenizer": {"id": "wrong", "revision": "tok-rev"},
        "predictor": {"id": "pred", "revision": "pred-rev"},
        "tokenizer_state": tokenizer.state_dict(),
        "predictor_state": predictor.state_dict(),
    }, path)
    args = SimpleNamespace(
        tokenizer_id="tok", tokenizer_revision="tok-rev",
        model_id="pred", model_revision="pred-rev",
    )
    with pytest.raises(ValueError, match="tokenizer identity mismatch"):
        _load_warm_bundle(path, tokenizer, predictor, args)


@torch_test
def test_kronos_nt_xent_rewards_correct_pairs_and_backpropagates():
    import torch
    first = torch.eye(4, requires_grad=True)
    paired = first + 0.01
    wrong = paired.roll(1, dims=0)
    good_loss = _nt_xent(first, paired, 0.1)
    bad_loss = _nt_xent(first, wrong, 0.1)
    assert good_loss < bad_loss
    good_loss.backward()
    assert first.grad is not None
    assert torch.isfinite(first.grad).all()
