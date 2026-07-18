from types import SimpleNamespace

import numpy as np
import pytest

from scripts import extract_native_representations as extractor


def _contexts(rows: int = 3) -> np.ndarray:
    return np.arange(rows * 512 * 5, dtype=np.float32).reshape(rows, 512, 5)


def test_chronos_v1_restores_channel_and_token_axes(monkeypatch):
    torch = pytest.importorskip("torch")
    seen = []

    monkeypatch.setattr(extractor, "_load_chronos_pipeline", lambda args: object())

    def fake_embedding(pipeline, flattened):
        assert flattened.ndim == 2 and flattened.shape[1] == 512
        seen.append(int(flattened.shape[0]))
        rows = int(flattened.shape[0])
        embedding = torch.arange(rows * 7 * 4, dtype=torch.float32).reshape(rows, 7, 4)
        state = torch.arange(rows, dtype=torch.float32)
        return embedding, state

    monkeypatch.setattr(extractor, "chronos_native_embedding", fake_embedding)
    output = extractor._extract_chronos_v1_bolt(
        SimpleNamespace(arm="chronos_v1", batch_size=2), _contexts()
    )

    assert seen == [10, 5]
    assert set(output) == {"representation", "tokenizer_state"}
    assert output["representation"].shape == (3, 5, 7, 4)
    assert output["tokenizer_state"].shape == (3, 5)
    np.testing.assert_array_equal(output["tokenizer_state"][0], np.arange(5))


def test_chronos_bolt_restores_channel_tokens_location_and_scale(monkeypatch):
    torch = pytest.importorskip("torch")

    monkeypatch.setattr(extractor, "_load_chronos_pipeline", lambda args: object())

    def fake_embedding(pipeline, flattened):
        rows = int(flattened.shape[0])
        embedding = torch.ones((rows, 3, 6), dtype=torch.float32)
        location = torch.arange(rows, dtype=torch.float32)
        scale = location + 100
        return embedding, (location, scale)

    monkeypatch.setattr(extractor, "chronos_native_embedding", fake_embedding)
    output = extractor._extract_chronos_v1_bolt(
        SimpleNamespace(arm="chronos_bolt", batch_size=2), _contexts()
    )

    assert set(output) == {"representation", "scaling_location", "scaling_scale"}
    assert output["representation"].shape == (3, 5, 3, 6)
    assert output["scaling_location"].shape == (3, 5)
    assert output["scaling_scale"].shape == (3, 5)
    np.testing.assert_array_equal(
        output["scaling_scale"][0] - output["scaling_location"][0],
        np.full(5, 100, dtype=np.float32),
    )


def test_chronos_state_contract_is_family_specific(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(extractor, "_load_chronos_pipeline", lambda args: object())
    monkeypatch.setattr(
        extractor,
        "chronos_native_embedding",
        lambda pipeline, flattened: (
            torch.ones((len(flattened), 2, 3)),
            (torch.ones(len(flattened)), torch.ones(len(flattened))),
        ),
    )

    with pytest.raises(ValueError, match="V1.*unexpected tuple"):
        extractor._extract_chronos_v1_bolt(
            SimpleNamespace(arm="chronos_v1", batch_size=2), _contexts(1)
        )


def test_representation_runtime_facts_are_complete_and_family_specific():
    assert extractor.RUNTIME_FACTS["chronos_v1"] == {
        "context_length": 512,
        "dtype": "float32",
        "output": "unpooled_embeddings_and_tokenizer_state",
    }
    assert extractor.RUNTIME_FACTS["chronos_bolt"] == {
        "context_length": 512,
        "dtype": "float32",
        "output": "unpooled_embeddings_and_location_scale",
    }
    assert extractor.EXTRACTORS["chronos_v1"] is extractor._extract_chronos_v1_bolt
    assert extractor.EXTRACTORS["chronos_bolt"] is extractor._extract_chronos_v1_bolt
