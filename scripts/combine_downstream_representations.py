#!/usr/bin/env python3
"""Create a hash-bound late-fusion representation from aligned frozen embeddings."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _atomic_npz(path: Path, **values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp.npz")
    np.savez_compressed(temporary, **values)
    os.replace(temporary, path)


def combine(inputs: list[Path], output: Path, *, arm: str, stage: str) -> dict:
    if len(inputs) < 2:
        raise ValueError("late fusion requires at least two representations")
    arrays: list[np.ndarray] = []
    rows = None
    reference = None
    sources = []
    for source in inputs:
        source = source.resolve()
        sidecar = Path(str(source) + ".manifest.json")
        if not source.is_file() or not sidecar.is_file():
            raise FileNotFoundError(source)
        manifest = json.loads(sidecar.read_text())
        if manifest.get("oos_read") is not False:
            raise ValueError(f"{source}: OOS guard failed")
        if manifest.get("artifact", {}).get("sha256") != _sha256(source):
            raise ValueError(f"{source}: artifact hash mismatch")
        with np.load(source, allow_pickle=False) as saved:
            embedding = np.asarray(saved["embedding"], np.float32)
            source_rows = np.asarray(saved["row_index"], np.int32)
        if embedding.ndim != 2 or len(embedding) != len(source_rows):
            raise ValueError(f"{source}: invalid embedding shape")
        if not np.isfinite(embedding).all() or len(np.unique(source_rows)) != len(source_rows):
            raise ValueError(f"{source}: invalid embedding values or row identity")
        identity = {
            key: manifest.get(key)
            for key in ("window_fingerprint", "windows_sha256", "row_selection", "contexts")
        }
        if reference is None:
            reference, rows = identity, source_rows
        elif identity != reference or not np.array_equal(rows, source_rows):
            raise ValueError("source representations do not share one sealed row contract")
        arrays.append(embedding)
        sources.append({
            "path": str(source), "sha256": _sha256(source),
            "arm": manifest.get("arm"), "stage": manifest.get("stage"),
            "shape": list(embedding.shape),
        })

    combined = np.concatenate(arrays, axis=1).astype(np.float32, copy=False)
    metadata = {
        "schema_version": "ffm_cross_family_representation_probe_v1",
        "arm": arm, "stage": stage,
        "checkpoint": None, "checkpoint_sha256": None,
        **reference,
        "shape": list(combined.shape),
        "config": {
            "fusion": "frozen_feature_concatenation_v1",
            "projector": "excluded",
            "sources": sources,
        },
        "oos_read": False,
    }
    signature = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    _atomic_npz(
        output,
        embedding=combined,
        row_index=rows,
        signature=np.array(signature),
        metadata=np.array(json.dumps(metadata)),
    )
    metadata["artifact"] = {"path": str(output.resolve()), "sha256": _sha256(output)}
    _atomic_json(Path(str(output) + ".manifest.json"), metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--stage", default="vanilla")
    args = parser.parse_args()
    result = combine(
        [Path(value) for value in args.input], Path(args.output),
        arm=args.arm, stage=args.stage,
    )
    print(json.dumps({
        "status": "complete", "arm": result["arm"], "stage": result["stage"],
        "shape": result["shape"], "artifact": result["artifact"],
    }, indent=2))


if __name__ == "__main__":
    main()
