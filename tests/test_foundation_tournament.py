import json

import pytest

from futures_foundation.finetune.tournament import (
    coverage_from_manifest, require_complete_oos, validate_boundaries,
)


def _manifest(path, ends):
    path.write_text(json.dumps({
        "roots_report": {
            symbol: {"last_timestamp": timestamp} for symbol, timestamp in ends.items()
        }
    }))


def test_tournament_dates_are_immutable():
    value = validate_boundaries("2019-07-01", "2024-07-01", "2025-07-01")
    assert value["train"]["start"] == "2019-07-01"
    with pytest.raises(ValueError, match="immutable"):
        validate_boundaries("2018-07-01", "2024-07-01", "2025-07-01")


def test_oos_coverage_gate_refuses_incomplete_symbols(tmp_path):
    manifest = tmp_path / "MANIFEST.json"
    _manifest(manifest, {"ES": "2026-05-04T23:59:00+00:00",
                         "NQ": "2026-07-02T00:00:00+00:00"})
    coverage = coverage_from_manifest(manifest)
    assert not coverage["common_oos_complete"]
    assert coverage["incomplete_roots"] == ["ES"]
    with pytest.raises(ValueError, match="locked OOS evaluation refused"):
        require_complete_oos(coverage)
    assert require_complete_oos(coverage, allow_partial=True)
