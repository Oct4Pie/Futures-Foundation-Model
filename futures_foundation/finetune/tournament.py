"""Immutable split and coverage contract for the equal-history foundation tournament."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


PROTOCOL_NAME = "foundation_5y1y1y_v1"
TRAIN_START = "2019-07-01"
VALIDATION_START = "2024-07-01"
OOS_START = "2025-07-01"
OOS_END = "2026-07-01"
MAX_CONTEXT = 256
FORECAST_HORIZON = 16
PARENT_LENGTH = MAX_CONTEXT + FORECAST_HORIZON + 1


def protocol():
    return {
        "name": PROTOCOL_NAME,
        "train": {"start": TRAIN_START, "end_exclusive": VALIDATION_START},
        "validation": {"start": VALIDATION_START, "end_exclusive": OOS_START},
        "oos": {"start": OOS_START, "end_exclusive": OOS_END},
        "sampling": {"max_context": MAX_CONTEXT, "forecast_horizon": FORECAST_HORIZON,
                     "parent_length": PARENT_LENGTH},
        "oos_status": "chronological_test_previously_inspected_not_pristine_holdout",
    }


def validate_boundaries(train_start, val_start, oos_start):
    expected = (TRAIN_START, VALIDATION_START, OOS_START)
    actual = tuple(str(pd.Timestamp(value).date()) for value in (train_start, val_start, oos_start))
    if actual != expected:
        raise ValueError(f"{PROTOCOL_NAME} boundaries are immutable: expected {expected}, got {actual}")
    return protocol()


def coverage_from_manifest(manifest_path):
    path = Path(manifest_path)
    raw = path.read_bytes()
    manifest = json.loads(raw)
    required_end = pd.Timestamp(OOS_END, tz="UTC")
    roots = {}
    for symbol, report in sorted((manifest.get("roots_report") or {}).items()):
        last = pd.Timestamp(report["last_timestamp"])
        last = last.tz_localize("UTC") if last.tzinfo is None else last.tz_convert("UTC")
        roots[symbol] = {
            "last_timestamp": last.isoformat(),
            "complete_oos": bool(last >= required_end),
            "missing_after": None if last >= required_end else last.isoformat(),
        }
    incomplete = [symbol for symbol, value in roots.items() if not value["complete_oos"]]
    return {
        "schema_version": "ffm_foundation_tournament_coverage_v1",
        "protocol": protocol(),
        "corpus_manifest": str(path.resolve()),
        "corpus_manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "roots": roots,
        "common_oos_complete": not incomplete,
        "incomplete_roots": incomplete,
    }


def require_complete_oos(coverage, *, allow_partial=False):
    if not coverage.get("common_oos_complete") and not allow_partial:
        missing = ",".join(coverage.get("incomplete_roots") or ())
        raise ValueError(
            f"locked OOS evaluation refused: missing bars through {OOS_END} for {missing}"
        )
    return True
