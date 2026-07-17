import numpy as np
import pytest

from scripts.compare_downstream_predictions import aligned_rows


def _prediction(rows=(1, 3), prediction=(0.2, 0.8)):
    return {
        "row_index": np.asarray(rows, np.int32),
        "target_index": np.asarray((0, 0), np.int16),
        "fold": np.asarray((1, 2), np.int8),
        "y_true": np.asarray((0.0, 1.0), np.float32),
        "prediction": np.asarray(prediction, np.float32),
    }


def test_aligned_rows_accepts_different_storage_order():
    left = _prediction()
    right = _prediction(rows=(3, 1), prediction=(0.7, 0.3))
    right["fold"] = right["fold"][::-1]
    right["y_true"] = right["y_true"][::-1]
    mask = np.ones(4, bool)

    left_rows, right_rows = aligned_rows(left, right, 0, mask)

    assert left["row_index"][left_rows].tolist() == [1, 3]
    assert right["row_index"][right_rows].tolist() == [1, 3]


def test_aligned_rows_rejects_truth_mismatch():
    left, right = _prediction(), _prediction()
    right["y_true"][1] = 0.0

    with pytest.raises(ValueError, match="y_true"):
        aligned_rows(left, right, 0, np.ones(4, bool))


def test_aligned_rows_skips_when_both_sides_are_empty():
    left, right = _prediction(), _prediction()

    assert aligned_rows(left, right, 1, np.ones(4, bool)) is None
