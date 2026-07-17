import json
from pathlib import Path

import numpy as np
import pytest

from scripts.benchmark_downstream_embedding import load_bound_embedding, reduce_embedding_fold


def _sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_bound_embedding_loader_checks_rows_hash_and_metadata(tmp_path):
    path = tmp_path / "embedding.npz"
    embedded = {"arm": "m", "stage": "vanilla", "window_fingerprint": "screen"}
    np.savez_compressed(
        path, embedding=np.ones((2, 3), np.float32), row_index=np.array([2, 5]),
        metadata=np.array(json.dumps(embedded)), signature=np.array("signature"),
    )
    manifest = {
        **embedded, "oos_read": False,
        "artifact": {"sha256": _sha(path)},
        "row_selection": {"sha256": "rows"},
    }
    Path(str(path) + ".manifest.json").write_text(json.dumps(manifest))
    selection = {"artifact": {"sha256": "rows"}}
    value, loaded = load_bound_embedding(
        path, selection_manifest=selection, expected_rows=np.array([2, 5]),
    )
    assert value.shape == (2, 3) and loaded["arm"] == "m"
    with pytest.raises(ValueError, match="differ from the sealed selection"):
        load_bound_embedding(
            path, selection_manifest=selection, expected_rows=np.array([2, 6]),
        )


def test_fold_reduction_is_train_only_and_bounded():
    rng = np.random.default_rng(4)
    value = rng.normal(size=(40, 20)).astype(np.float32)
    train, test = np.arange(25), np.arange(25, 40)
    first, metadata = reduce_embedding_fold(
        value, train, test, max_components=8, seed=2,
    )
    changed = value.copy()
    changed[test] *= 1000
    second, _ = reduce_embedding_fold(
        changed, train, test, max_components=8, seed=2,
    )
    np.testing.assert_allclose(first[train], second[train])
    assert metadata["components"] == 8
    assert 0 < metadata["explained_variance"] <= 1
