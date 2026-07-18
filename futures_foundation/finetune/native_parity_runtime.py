"""Shared immutable runtime/source policy for real native-parity workers."""
from __future__ import annotations

import base64
import csv
import hashlib
from pathlib import Path

PROFILE_ARMS = {
    "common": {
        "mantis_v1", "mantis_v2", "moment_small", "kronos_mini",
        "kronos_small", "chronos_v1", "chronos_bolt", "chronos_v2",
        "toto2_22m",
    },
    "timesfm": {"timesfm25"},
    "ttm": {"ttm_r2"},
    "moirai": {"moirai2_small"},
    "sundial": {"sundial_base"},
}
PACKAGE_PROFILES = {
    "common": {"torch": "2.13.0"},
    "timesfm": {"torch": "2.13.0", "transformers": "5.13.1"},
    "ttm": {"torch": "2.10.0", "transformers": "4.57.6"},
    "moirai": {"torch": "2.10.0", "uni2ts": "2.0.0"},
    "sundial": {
        "torch": "2.10.0", "transformers": "4.40.1",
        "huggingface-hub": "0.36.2",
    },
}
PROFILE_PYTHON = {
    "common": (3, 12), "timesfm": (3, 12), "ttm": (3, 12),
    "moirai": (3, 11), "sundial": (3, 12),
}
ARM_PACKAGES = {
    "mantis_v1": {"mantis-tsfm": "1.0.0"},
    "mantis_v2": {"mantis-tsfm": "1.0.0"},
    "moment_small": {"momentfm": "0.1.5"},
    "chronos_v1": {"chronos-forecasting": "2.3.1"},
    "chronos_bolt": {"chronos-forecasting": "2.3.1"},
    "chronos_v2": {"chronos-forecasting": "2.3.1"},
    "toto2_22m": {"toto-2": "2.0.0"},
}

GIT_SOURCE_ARMS = {
    "mantis_v1", "mantis_v2", "moment_small", "kronos_mini", "kronos_small",
    "timesfm25", "ttm_r2", "moirai2_small", "toto2_22m", "sundial_base",
}
PACKAGE_SOURCE_ARMS = {"chronos_v1", "chronos_bolt", "chronos_v2"}


def validate_distribution_record(path: str | Path) -> Path:
    """Verify every hashed file in an installed wheel's RECORD manifest."""
    root = Path(path).resolve()
    record = root / "RECORD"
    if not root.name.endswith(".dist-info") or not record.is_file():
        raise RuntimeError(f"installed distribution RECORD root required: {root}")
    checked = 0
    with record.open(newline="", encoding="utf-8") as stream:
        for relative, encoded_hash, size in csv.reader(stream):
            target = (root.parent / relative).resolve()
            if not target.is_file():
                raise RuntimeError(f"distribution RECORD file is missing: {target}")
            if not encoded_hash:
                if target != record:
                    raise RuntimeError(f"distribution RECORD lacks a hash: {relative}")
                continue
            algorithm, encoded = encoded_hash.split("=", 1)
            if algorithm != "sha256":
                raise RuntimeError(f"unsupported RECORD hash algorithm: {algorithm}")
            actual = hashlib.sha256(target.read_bytes()).digest()
            expected = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
            if actual != expected:
                raise RuntimeError(f"distribution RECORD hash mismatch: {target}")
            if size and target.stat().st_size != int(size):
                raise RuntimeError(f"distribution RECORD size mismatch: {target}")
            checked += 1
    if not checked:
        raise RuntimeError(f"distribution RECORD contains no hashed files: {record}")
    return root
