import json
from pathlib import Path

import numpy as np

from scripts import benchmark_foundation_representations as benchmark
from scripts.extract_downstream_representations import _screen_fingerprint, _verified_existing


def test_row_bound_embedding_saves_selection_and_context_identity(tmp_path):
    args = type("Args", (), {
        "output_dir": str(tmp_path),
        "row_index": np.array([3, 7], np.int32),
        "row_selection_manifest": {
            "artifact": {"sha256": "selection-sha"}, "content_fingerprint": "selection-fp",
        },
        "context_manifest": {
            "artifact": {"sha256": "context-sha"}, "content_fingerprint": "context-fp",
        },
    })()
    windows = {"window_fingerprint": "screen", "artifact": {"sha256": "context-sha"}}
    benchmark._save_embedding(
        args, "arm", "vanilla", None, np.ones((2, 4), np.float32), {}, windows,
    )
    path = tmp_path / "embeddings" / "arm" / "vanilla.npz"
    manifest = json.loads(Path(str(path) + ".manifest.json").read_text())
    assert manifest["row_selection"]["sha256"] == "selection-sha"
    assert manifest["contexts"]["sha256"] == "context-sha"
    with np.load(path, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["row_index"], [3, 7])
    assert _verified_existing(path, arm="arm", stage="vanilla", screen="screen")


def test_screen_fingerprint_changes_with_selection():
    context = {"content_fingerprint": "context"}
    first = {"content_fingerprint": "rows-a"}
    second = {"content_fingerprint": "rows-b"}
    assert _screen_fingerprint(context, first) != _screen_fingerprint(context, second)
