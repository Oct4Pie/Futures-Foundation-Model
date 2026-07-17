#!/usr/bin/env python3
"""Seal the balanced row subset used for the frozen representation screen."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from futures_foundation.finetune.downstream_sample import (
    build_balanced_row_selection,
    load_balanced_sample,
    load_row_selection,
    save_row_selection,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample",
        default="output/foundation_tournament/downstream_gate_v1/balanced_sample.npz",
    )
    parser.add_argument(
        "--output",
        default="output/foundation_tournament/downstream_gate_v1/representation_rows.npz",
    )
    parser.add_argument("--rows-per-stream", type=int, default=400)
    parser.add_argument(
        "--all-rows", action="store_true",
        help="select every row in an already event-focused sample",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    rows_per_stream = None if args.all_rows else int(args.rows_per_stream)
    sample, sample_manifest = load_balanced_sample(args.sample)
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        arrays, manifest = load_row_selection(output, sample_manifest=sample_manifest)
        if manifest["metadata"]["rows_per_stream"] != rows_per_stream:
            raise ValueError("existing row selection uses another count; use --overwrite")
        print(json.dumps({
            "status": "verified_existing", "rows": int(len(arrays["row_index"])),
            "sha256": manifest["artifact"]["sha256"],
        }, indent=2))
        return
    arrays, metadata = build_balanced_row_selection(
        sample, sample_manifest, rows_per_stream=rows_per_stream,
    )
    manifest = save_row_selection(output, arrays, metadata)
    print(json.dumps({
        "status": "complete", "rows": int(len(arrays["row_index"])),
        "streams": metadata["streams"], "sha256": manifest["artifact"]["sha256"],
        "content_fingerprint": manifest["content_fingerprint"],
    }, indent=2))


if __name__ == "__main__":
    main()
