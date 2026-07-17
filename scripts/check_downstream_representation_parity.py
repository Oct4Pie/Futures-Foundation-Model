#!/usr/bin/env python3
"""Verify repeat-extraction parity for row-bound downstream embeddings."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def compare(left_path: Path, right_path: Path, *, atol: float, rtol: float) -> dict:
    with np.load(left_path, allow_pickle=False) as left, np.load(
        right_path, allow_pickle=False,
    ) as right:
        required = {"embedding", "row_index"}
        if not required.issubset(left.files) or not required.issubset(right.files):
            raise ValueError("both artifacts must contain embedding and row_index")
        left_rows = np.asarray(left["row_index"])
        right_rows = np.asarray(right["row_index"])
        if not np.array_equal(left_rows, right_rows):
            raise ValueError("row_index mismatch; artifacts are not comparable")
        left_embedding = np.asarray(left["embedding"], np.float64)
        right_embedding = np.asarray(right["embedding"], np.float64)

    if left_embedding.shape != right_embedding.shape:
        raise ValueError(
            f"embedding shape mismatch: {left_embedding.shape} != {right_embedding.shape}"
        )
    if not np.isfinite(left_embedding).all() or not np.isfinite(right_embedding).all():
        raise ValueError("non-finite embedding values")

    difference = np.abs(left_embedding - right_embedding)
    denominator = np.linalg.norm(left_embedding, axis=1) * np.linalg.norm(
        right_embedding, axis=1,
    )
    cosine = np.divide(
        np.sum(left_embedding * right_embedding, axis=1),
        denominator,
        out=np.ones_like(denominator),
        where=denominator > 0,
    )
    passed = bool(np.allclose(left_embedding, right_embedding, atol=atol, rtol=rtol))
    return {
        "schema_version": "ffm_downstream_representation_parity_v1",
        "status": "passed" if passed else "failed",
        "left": {"path": str(left_path.resolve()), "sha256": _sha256(left_path)},
        "right": {"path": str(right_path.resolve()), "sha256": _sha256(right_path)},
        "rows": int(left_embedding.shape[0]),
        "dimensions": int(left_embedding.shape[1]),
        "atol": float(atol),
        "rtol": float(rtol),
        "mean_absolute_error": float(difference.mean()),
        "max_absolute_error": float(difference.max()),
        "relative_l2_error": float(
            np.linalg.norm(left_embedding - right_embedding)
            / max(np.linalg.norm(left_embedding), np.finfo(np.float64).eps)
        ),
        "cosine_similarity": {
            "mean": float(cosine.mean()),
            "minimum": float(cosine.min()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = compare(args.left, args.right, atol=args.atol, rtol=args.rtol)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(payload)
        temporary.replace(args.output)
    print(payload, end="")
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
