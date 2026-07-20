#!/usr/bin/env python3
"""Publish an externally hashable authority for an existing sealed OHLCV corpus."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from futures_foundation._authority_bundle_io import (
    canonical_absolute_path,
    read_regular_file,
)
from futures_foundation.finetune.tournament_cache_authority import (
    SOURCE_AUTHORITY_SCHEMA_VERSION,
    SOURCE_MANIFEST_SCHEMA_VERSION,
    canonical_authority_document,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _stream_ids(document: dict, roots: tuple[str, ...], timeframes: tuple[str, ...]) -> list[str]:
    outputs = document.get("outputs")
    if not isinstance(outputs, dict) or not outputs:
        raise ValueError("source manifest has no outputs")
    streams = []
    for root in roots:
        for timeframe in timeframes:
            key = f"{root}_{timeframe}"
            if key not in outputs:
                raise ValueError(f"source manifest lacks requested output: {key}")
            streams.append(f"{root}@{timeframe}")
    return sorted(streams)


def run(args: argparse.Namespace) -> dict[str, object]:
    source_dir = canonical_absolute_path(
        Path(args.source_dir).expanduser().resolve(), "source corpus directory"
    )
    manifest_path = source_dir / "MANIFEST.json"
    reopened, raw, manifest_sha = read_regular_file(
        manifest_path,
        label="source corpus manifest",
        max_bytes=32 * 1024 * 1024,
    )
    if reopened != manifest_path:
        raise ValueError("source manifest path changed during open")
    try:
        manifest = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("source corpus manifest is not valid JSON") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != SOURCE_MANIFEST_SCHEMA_VERSION:
        raise ValueError("source corpus manifest schema is unsupported")
    roots = tuple(dict.fromkeys(
        value.strip().upper() for value in args.roots.split(",") if value.strip()
    ))
    timeframes = tuple(dict.fromkeys(
        value.strip() for value in args.timeframes.split(",") if value.strip()
    ))
    if not roots or not timeframes:
        raise ValueError("roots and timeframes are required")
    admitted_streams = _stream_ids(manifest, roots, timeframes)
    document = {
        "schema_version": SOURCE_AUTHORITY_SCHEMA_VERSION,
        "authority_id": args.authority_id,
        "purpose": "tournament_cache_source_admission",
        "source_manifest": {
            "path": str(manifest_path),
            "sha256": manifest_sha,
            "bytes": len(raw),
            "schema_version": SOURCE_MANIFEST_SCHEMA_VERSION,
        },
        "admitted_streams": admitted_streams,
        "cache_construction_admitted": True,
        "training_admitted": False,
    }
    output = Path(args.output).expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite source authority: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(output) + f".{os.getpid()}.tmp")
    temporary.write_bytes(canonical_authority_document(document))
    os.replace(temporary, output)
    return {
        "status": "complete",
        "path": str(output),
        "sha256": _sha256(output),
        "bytes": output.stat().st_size,
        "streams": len(admitted_streams),
        "cache_construction_admitted": True,
        "training_admitted": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--roots", required=True)
    parser.add_argument("--timeframes", required=True)
    parser.add_argument("--authority-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
