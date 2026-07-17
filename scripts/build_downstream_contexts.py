#!/usr/bin/env python3
"""Build the shared raw-context artifact for frozen foundation representations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from futures_foundation.finetune.downstream_contexts import (
    build_downstream_contexts,
    load_downstream_contexts,
    save_downstream_contexts,
)
from futures_foundation.finetune.downstream_sample import load_balanced_sample


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample",
        default="output/foundation_tournament/downstream_gate_v1/balanced_sample.npz",
    )
    parser.add_argument(
        "--cache-manifest",
        default="output/foundation_tournament/data_cache/TOURNAMENT_CACHE.json",
    )
    parser.add_argument(
        "--output",
        default="output/foundation_tournament/downstream_gate_v1/contexts.npz",
    )
    parser.add_argument("--context-bars", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    sample, sample_manifest = load_balanced_sample(args.sample)
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        arrays, manifest = load_downstream_contexts(
            output, sample_manifest=sample_manifest,
        )
        print(json.dumps({
            "status": "verified_existing", "rows": int(len(arrays["row_index"])),
            "sha256": manifest["artifact"]["sha256"],
        }, indent=2))
        return
    arrays, metadata = build_downstream_contexts(
        sample, sample_manifest, args.cache_manifest, context_bars=args.context_bars,
    )
    manifest = save_downstream_contexts(output, arrays, metadata)
    print(json.dumps({
        "status": "complete", "rows": int(len(arrays["row_index"])),
        "shape": list(arrays["context"].shape),
        "sha256": manifest["artifact"]["sha256"],
        "content_fingerprint": manifest["content_fingerprint"],
        "bytes": manifest["artifact"]["bytes"],
    }, indent=2))


if __name__ == "__main__":
    main()
