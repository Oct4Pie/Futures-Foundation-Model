from pathlib import Path

import numpy as np
import pytest

from scripts.check_downstream_representation_parity import compare


def _save(path: Path, embedding, rows=(3, 7)):
    np.savez_compressed(
        path,
        embedding=np.asarray(embedding, np.float32),
        row_index=np.asarray(rows, np.int32),
    )


def test_compare_accepts_close_row_aligned_embeddings(tmp_path):
    left = tmp_path / "left.npz"
    right = tmp_path / "right.npz"
    _save(left, [[1.0, 2.0], [3.0, 4.0]])
    _save(right, [[1.0 + 1e-7, 2.0], [3.0, 4.0]])

    report = compare(left, right, atol=1e-6, rtol=1e-6)

    assert report["status"] == "passed"
    assert report["rows"] == 2
    assert report["dimensions"] == 2


def test_compare_rejects_row_mismatch(tmp_path):
    left = tmp_path / "left.npz"
    right = tmp_path / "right.npz"
    _save(left, [[1.0], [2.0]])
    _save(right, [[1.0], [2.0]], rows=(3, 8))

    with pytest.raises(ValueError, match="row_index mismatch"):
        compare(left, right, atol=1e-6, rtol=1e-6)


def test_compare_reports_failed_tolerance(tmp_path):
    left = tmp_path / "left.npz"
    right = tmp_path / "right.npz"
    _save(left, [[1.0], [2.0]])
    _save(right, [[1.1], [2.0]])

    report = compare(left, right, atol=1e-6, rtol=1e-6)

    assert report["status"] == "failed"
    assert report["max_absolute_error"] == pytest.approx(0.1)
