import os
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.train_moment_tournament import (
    BUNDLE_SCHEMA, _load_parent_bundle, _masked_loss, _stage_bundle,
)
from scripts.train_moment_contrastive import _batch as _contrastive_batch, _pooled_embedding
from scripts.train_moment_forecast import _batch as _forecast_batch, _scale_normalized_mse


torch_test = pytest.mark.skipif(
    os.environ.get("CHRONOS_TORCH_TESTS") != "1",
    reason="torch test — set CHRONOS_TORCH_TESTS=1 (libomp isolation)",
)


@torch_test
def test_moment_masked_loss_is_padding_safe_and_channel_scale_invariant():
    import torch
    original = torch.zeros(2, 2, 16)
    valid = torch.arange(8, dtype=torch.float32)
    original[:, 0, -8:] = valid
    original[:, 1, -8:] = 10_000 * valid
    input_mask = torch.zeros(2, 16)
    input_mask[:, -8:] = 1
    pretrain_mask = input_mask.clone()
    pretrain_mask[:, -8:-4] = 0
    # Add one standard deviation on hidden positions in both differently scaled channels.
    reconstruction = original.clone()
    reconstruction[:, 0, -8:-4] += valid.std()
    reconstruction[:, 1, -8:-4] += 10_000 * valid.std()
    output = SimpleNamespace(reconstruction=reconstruction, pretrain_mask=pretrain_mask)
    loss = _masked_loss(output, original, input_mask)
    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(1.0, rel=1e-5)


@torch_test
def test_stage1_bundle_records_identity_and_state():
    import torch
    model = torch.nn.Linear(2, 3)
    args = SimpleNamespace(model_id="moment", model_revision="revision")
    bundle = _stage_bundle(model, args)
    assert bundle["schema_version"] == BUNDLE_SCHEMA
    assert bundle["stage"] == "stage1_reconstruction"
    assert bundle["model"] == {"id": "moment", "revision": "revision"}
    assert bundle["parent"] is None
    assert set(bundle["model_state"]) == set(model.state_dict())


@torch_test
def test_parent_bundle_rejects_stage_or_identity_drift(tmp_path):
    import torch
    model = torch.nn.Module()
    model.encoder = torch.nn.Linear(2, 2)
    model.patch_embedding = torch.nn.Linear(2, 2)
    args = SimpleNamespace(model_id="moment", model_revision="revision")
    bundle = {
        "schema_version": BUNDLE_SCHEMA, "stage": "stage1_reconstruction",
        "model": {"id": "moment", "revision": "revision"},
        "model_state": model.state_dict(), "parent": None,
    }
    path = tmp_path / "parent.pt"
    torch.save(bundle, path)
    loaded = _load_parent_bundle(path, model, args, "stage1_reconstruction")
    assert loaded["transferred_tensors"] == len(model.state_dict())
    with pytest.raises(ValueError, match="stage mismatch"):
        _load_parent_bundle(path, model, args, "stage2_contrastive")
    bundle["model"] = {"id": "wrong", "revision": "revision"}
    torch.save(bundle, path)
    with pytest.raises(ValueError, match="identity mismatch"):
        _load_parent_bundle(path, model, args, "stage1_reconstruction")


@torch_test
def test_contrastive_pool_excludes_left_padding_and_keeps_channels():
    import torch
    class FakeModel:
        patch_len = 2
        def embed(self, *, x_enc, input_mask, reduction):
            assert reduction == "none"
            values = torch.tensor([[[[100.0], [100.0], [2.0], [4.0]],
                                    [[200.0], [200.0], [6.0], [10.0]]]])
            return SimpleNamespace(embeddings=values)
    mask = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]], dtype=torch.long)
    pooled = _pooled_embedding(FakeModel(), torch.zeros(1, 2, 8), mask)
    assert pooled.tolist() == [[3.0, 8.0]]


@torch_test
def test_forecast_loss_is_channel_scale_invariant():
    import torch
    context = torch.arange(8, dtype=torch.float32).reshape(1, 1, 8).repeat(1, 2, 1)
    context[:, 1] *= 10_000
    mask = torch.ones(1, 8)
    future = context[:, :, -1:].repeat(1, 1, 2)
    forecast = future.clone()
    forecast[:, 0] += context[:, 0].std()
    forecast[:, 1] += context[:, 1].std()
    loss = _scale_normalized_mse(forecast, future, context, mask)
    assert loss.item() == pytest.approx(1.0, rel=1e-5)


@torch_test
def test_moment_stage_batches_keep_five_channels_and_causal_boundaries():
    import torch
    from futures_foundation.finetune.tournament import MAX_CONTEXT, PARENT_LENGTH
    big = np.arange(PARENT_LENGTH * 5, dtype=np.float32).reshape(PARENT_LENGTH, 5)
    views = _contrastive_batch(big, np.array([0]), "cpu")
    assert [tuple(x.shape) for x, _ in views] == [(1, 5, 512), (1, 5, 512)]
    assert torch.equal(views[0][0][0, :, -1], torch.as_tensor(big[127]))
    assert torch.equal(views[1][0][0, :, -1], torch.as_tensor(big[MAX_CONTEXT - 1]))
    context, mask, future = _forecast_batch(big, np.array([0]), "cpu")
    assert tuple(context.shape) == (1, 5, 512)
    assert tuple(future.shape) == (1, 5, 16)
    assert torch.equal(context[0, :, -1], torch.as_tensor(big[MAX_CONTEXT - 1]))
    assert torch.equal(future[0, :, 0], torch.as_tensor(big[MAX_CONTEXT]))
