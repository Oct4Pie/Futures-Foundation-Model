#!/usr/bin/env python3
"""Create an immutable native-contract disposition snapshot of historical artifacts.

The source index and artifacts are never edited.  This script records their hashes and overlays
methodology statuses in a separate JSON document so historical evidence remains reproducible but
cannot be mistaken for a native-performance ranking.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from futures_foundation.finetune.native_contracts import (
    all_arms,
    file_sha256,
    evidence_sha256,
    historical_disposition,
    registry_sha256,
)


SCHEMA = "ffm_historical_native_contract_snapshot_v1"
HISTORICAL_ALIASES = {
    "moment": "moment_small",
    # The historical combined arm was used as the downstream in-context control. The
    # new TS3 forecast arm is a distinct identity and must not inherit those artifacts.
    "tabpfn_ts": "tabpfn_v3_downstream",
}
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _portable_path(path: Path) -> str:
    """Use repository-relative paths for tracked artifacts and absolute paths elsewhere."""
    path = Path(path).resolve()
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _atomic_json(path: str | Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    temporary.write_bytes(snapshot_json_bytes(value))
    os.replace(temporary, path)


def snapshot_json_bytes(value: dict) -> bytes:
    """Serialize snapshots canonically enough for byte-for-byte reconstruction."""
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def build_snapshot(index_path: str | Path, artifact_root: str | Path) -> dict:
    index_path = Path(index_path).resolve()
    artifact_root = Path(artifact_root).resolve()
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    if not artifact_root.is_dir():
        raise FileNotFoundError(artifact_root)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    indexed = index.get("models") or {}
    observed_directories = {
        path.name: path for path in artifact_root.iterdir() if path.is_dir()
    }
    arms = all_arms()
    rows = {}
    for arm_key, arm in arms.items():
        historical_key = next(
            (key for key, alias in HISTORICAL_ALIASES.items() if alias == arm_key),
            arm_key,
        )
        indexed_value = indexed.get(historical_key)
        directory = observed_directories.get(historical_key)
        disposition = historical_disposition(arm_key)
        artifact_files = sorted(path for path in directory.rglob("*") if path.is_file()) if directory else []
        rows[arm_key] = {
            "historical_key": historical_key,
            "status": disposition["default_status"],
            "reason": disposition["reason"],
            "artifact_directory": _portable_path(directory) if directory else None,
            "artifact_file_count": len(artifact_files),
            "index_present": indexed_value is not None,
            "complete_chain_claim": (
                indexed_value.get("complete_chain") if isinstance(indexed_value, dict) else None
            ),
            "shared_validation_present": bool(
                isinstance(indexed_value, dict) and indexed_value.get("shared_validation")
            ),
            "native_ranking_eligible": False,
        }
    unregistered = sorted(
        set(observed_directories)
        - {row["historical_key"] for row in rows.values()}
    )
    return {
        "schema_version": SCHEMA,
        "source_index": {
            "path": _portable_path(index_path),
            "sha256": file_sha256(index_path),
            "schema_version": index.get("schema_version"),
            "created_utc": index.get("created_utc"),
        },
        "artifact_root": _portable_path(artifact_root),
        "registry_sha256": registry_sha256(),
        "evidence_sha256": evidence_sha256(),
        "methodology": (
            "Historical scores remain evidence under their exact adapter contracts. "
            "No row in this snapshot is eligible for a native cross-family ranking."
        ),
        "models": rows,
        "unregistered_artifact_directories": unregistered,
        "coverage": {
            "registered_model_count": len(rows),
            "artifact_directory_count": sum(row["artifact_directory"] is not None for row in rows.values()),
            "indexed_model_count": sum(row["index_present"] for row in rows.values()),
            "native_ranking_eligible_count": 0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index",
        default="output/foundation_tournament/final_staged/STAGE_RESULTS_INDEX.json",
    )
    parser.add_argument(
        "--artifact-root",
        default="output/foundation_tournament/final_staged",
    )
    parser.add_argument(
        "--output",
        default=(
            "output/foundation_tournament/final_staged/"
            "HISTORICAL_NATIVE_CONTRACT_SNAPSHOT.json"
        ),
    )
    args = parser.parse_args()
    _atomic_json(args.output, build_snapshot(args.index, args.artifact_root))


if __name__ == "__main__":
    main()
