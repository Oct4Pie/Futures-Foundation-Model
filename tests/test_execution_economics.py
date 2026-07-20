import hashlib
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from futures_foundation.execution_economics import (
    load_execution_economics, require_execution_economics,
)


ROOT = Path(__file__).resolve().parents[1]


def test_repository_schedule_is_source_verified_scoped_and_immutable():
    economics = load_execution_economics(
        ROOT / "config/execution_costs.yaml",
        evaluation_start="2024-07-01T00:00:00Z",
        evaluation_end="2025-07-01T00:00:00Z",
        required_roots=("ES", "ZN"),
    )
    assert economics.instrument("es").tick_size == 0.25
    assert economics.instrument("ZN").tick_value_usd == 15.625
    assert economics.primary_added_slippage_ticks_round_trip == 0.0
    assert 1.0 in economics.sensitivity_added_slippage_ticks_round_trip
    assert economics.validate_added_slippage(0.0) == 0.0
    assert economics.validate_added_slippage(1.0) == 1.0
    with pytest.raises(TypeError):
        economics.instruments["ES"] = economics.instrument("ES")
    assert require_execution_economics(economics) is economics
    with pytest.raises(TypeError, match="canonical verified"):
        require_execution_economics(replace(economics, canonical_admitted=False))


def test_schedule_rejects_missing_root_and_effective_date_ambiguity():
    path = ROOT / "config/execution_costs.yaml"
    with pytest.raises(ValueError, match="missing instruments"):
        load_execution_economics(
            path,
            evaluation_start="2024-01-01T00:00:00Z",
            evaluation_end="2025-01-01T00:00:00Z",
            required_roots=("DOES_NOT_EXIST",),
        )
    with pytest.raises(ValueError, match="outside the declared"):
        load_execution_economics(
            path,
            evaluation_start="2026-07-18T00:00:00Z",
            evaluation_end="2026-07-20T00:00:00Z",
        )
    with pytest.raises(ValueError, match="UTC offset"):
        load_execution_economics(
            path, evaluation_start="2024-01-01", evaluation_end="2025-01-01",
        )


def _temporary_schedule(tmp_path: Path, *, copied_fee: float = 4.36) -> Path:
    repository = tmp_path / "repo"
    config = repository / "config"
    source_dir = repository / "source"
    config.mkdir(parents=True)
    source_dir.mkdir(parents=True)
    source_path = source_dir / "instruments.yaml"
    source_path.write_text(yaml.safe_dump({
        "schema_version": "source_v1",
        "instruments": {
            "ES": {
                "tick_size": 0.25, "tick_value_usd": 12.5, "fee_rt_usd": 4.36,
            },
        },
    }))
    schedule = {
        "schema_version": "ffm_execution_economics_v2",
        "description": "test",
        "source": {
            "project_path": "source", "document_path": "instruments.yaml",
            "schema_version": "source_v1",
            "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(), "note": "test",
        },
        "effective": {
            "start_utc": "2024-01-01T00:00:00Z",
            "end_exclusive_utc": "2025-01-01T00:00:00Z",
            "basis": "test_constant_assumption",
        },
        "primary_added_slippage_ticks_round_trip": 0.0,
        "sensitivity_added_slippage_ticks_round_trip": [0.0, 1.0],
        "instruments": {
            "ES": {
                "tick_size": 0.25, "tick_value_usd": 12.5, "fee_rt_usd": copied_fee,
            },
        },
    }
    path = config / "economics.yaml"
    path.write_text(yaml.safe_dump(schedule, sort_keys=False))
    return path


def test_schedule_rejects_values_that_disagree_with_pinned_source(tmp_path):
    path = _temporary_schedule(tmp_path, copied_fee=0.0)
    with pytest.raises(ValueError, match="mismatch pinned source"):
        load_execution_economics(
            path,
            evaluation_start="2024-01-01T00:00:00Z",
            evaluation_end="2024-12-31T00:00:00Z",
        )


def test_schedule_and_source_reject_symlink_and_hardlink_transport(tmp_path):
    path = _temporary_schedule(tmp_path)
    schedule_link = tmp_path / "economics-link.yaml"
    schedule_link.symlink_to(path)
    with pytest.raises(ValueError, match="symlink"):
        load_execution_economics(
            schedule_link,
            evaluation_start="2024-01-01T00:00:00Z",
            evaluation_end="2024-12-31T00:00:00Z",
        )

    schedule_hardlink = tmp_path / "economics-hardlink.yaml"
    schedule_hardlink.hardlink_to(path)
    with pytest.raises(ValueError, match="bounded regular file"):
        load_execution_economics(
            path,
            evaluation_start="2024-01-01T00:00:00Z",
            evaluation_end="2024-12-31T00:00:00Z",
        )

    fresh = _temporary_schedule(tmp_path / "source-link")
    source = tmp_path / "source-link/repo/source/instruments.yaml"
    target = source.with_name("instruments-real.yaml")
    source.rename(target)
    source.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        load_execution_economics(
            fresh,
            evaluation_start="2024-01-01T00:00:00Z",
            evaluation_end="2024-12-31T00:00:00Z",
        )


def test_schedule_rejects_yaml_aliases_and_oversized_inputs(tmp_path):
    path = _temporary_schedule(tmp_path)
    path.write_text(path.read_text() + "alias_probe: &probe [1]\nalias_copy: *probe\n")
    with pytest.raises(ValueError, match="anchors or aliases"):
        load_execution_economics(
            path,
            evaluation_start="2024-01-01T00:00:00Z",
            evaluation_end="2024-12-31T00:00:00Z",
        )

    oversized = _temporary_schedule(tmp_path / "oversized")
    with oversized.open("a") as stream:
        stream.write("#" + "x" * (4 * 1024 * 1024) + "\n")
    with pytest.raises(ValueError, match="bounded regular file"):
        load_execution_economics(
            oversized,
            evaluation_start="2024-01-01T00:00:00Z",
            evaluation_end="2024-12-31T00:00:00Z",
        )


def test_schedule_rejects_source_hash_drift_and_undeclared_slippage(tmp_path):
    path = _temporary_schedule(tmp_path)
    source_path = tmp_path / "repo/source/instruments.yaml"
    source_path.write_text(source_path.read_text() + "# drift\n")
    with pytest.raises(ValueError, match="source hash mismatch"):
        load_execution_economics(
            path,
            evaluation_start="2024-01-01T00:00:00Z",
            evaluation_end="2024-12-31T00:00:00Z",
        )

    path = _temporary_schedule(tmp_path / "fresh")
    economics = load_execution_economics(
        path,
        evaluation_start="2024-01-01T00:00:00Z",
        evaluation_end="2024-12-31T00:00:00Z",
    )
    with pytest.raises(ValueError, match="not declared"):
        economics.validate_added_slippage(0.5)
