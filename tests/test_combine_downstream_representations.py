import hashlib
import json

import numpy as np
import pytest

from scripts.combine_downstream_representations import combine


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_bound(path, values, rows, *, selection="same"):
    metadata = {"arm": path.stem, "stage": "vanilla", "window_fingerprint": "window"}
    np.savez_compressed(
        path, embedding=np.asarray(values, np.float32), row_index=np.asarray(rows, np.int32),
        signature=np.array("signature"), metadata=np.array(json.dumps(metadata)),
    )
    manifest = {
        "arm": path.stem, "stage": "vanilla", "window_fingerprint": "window",
        "windows_sha256": "contexts", "row_selection": {"sha256": selection},
        "contexts": {"sha256": "contexts"}, "oos_read": False,
        "artifact": {"path": str(path), "sha256": _sha(path)},
    }
    (path.parent / f"{path.name}.manifest.json").write_text(json.dumps(manifest))


def test_combine_preserves_rows_and_concatenates(tmp_path):
    one, two, output = tmp_path / "one.npz", tmp_path / "two.npz", tmp_path / "out.npz"
    _write_bound(one, [[1, 2], [3, 4]], [7, 9])
    _write_bound(two, [[5], [6]], [7, 9])
    manifest = combine([one, two], output, arm="one_two", stage="vanilla")
    with np.load(output, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["row_index"], [7, 9])
        np.testing.assert_allclose(saved["embedding"], [[1, 2, 5], [3, 4, 6]])
    assert manifest["oos_read"] is False
    assert manifest["artifact"]["sha256"] == _sha(output)


def test_combine_rejects_different_row_contract(tmp_path):
    one, two = tmp_path / "one.npz", tmp_path / "two.npz"
    _write_bound(one, [[1], [2]], [7, 9])
    _write_bound(two, [[3], [4]], [7, 10])
    with pytest.raises(ValueError, match="sealed row contract"):
        combine([one, two], tmp_path / "out.npz", arm="one_two", stage="vanilla")
