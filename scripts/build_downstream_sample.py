#!/usr/bin/env python3
"""Build the sealed, stream-balanced Gate-3 development sample."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from futures_foundation.finetune.downstream_sample import (
    build_balanced_sample,
    load_balanced_sample,
    save_balanced_sample,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collection",
        default="output/foundation_tournament/event_contexts_v1/MANIFEST.json",
    )
    parser.add_argument(
        "--output",
        default="output/foundation_tournament/downstream_gate_v1/balanced_sample.npz",
    )
    parser.add_argument("--rows-per-stream", type=int, default=1200)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        arrays, manifest = load_balanced_sample(output)
        expected = int(args.rows_per_stream) * len(manifest["metadata"]["source_shards"])
        if manifest["metadata"]["selection"]["rows_per_stream"] != int(args.rows_per_stream):
            raise ValueError("existing sample uses a different rows_per_stream; use --overwrite")
        print(json.dumps({
            "status": "verified_existing",
            "rows": int(len(arrays["stream_id"])),
            "expected_rows": expected,
            "sha256": manifest["artifact"]["sha256"],
        }, indent=2))
        return
    arrays, metadata = build_balanced_sample(
        args.collection, rows_per_stream=args.rows_per_stream,
    )
    manifest = save_balanced_sample(output, arrays, metadata)
    print(json.dumps({
        "status": "complete",
        "rows": int(len(arrays["stream_id"])),
        "streams": int(len(metadata["source_shards"])),
        "sha256": manifest["artifact"]["sha256"],
        "content_fingerprint": manifest["content_fingerprint"],
        "bytes": manifest["artifact"]["bytes"],
    }, indent=2))


if __name__ == "__main__":
    main()
