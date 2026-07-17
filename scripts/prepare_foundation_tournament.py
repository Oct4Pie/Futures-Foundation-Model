#!/usr/bin/env python3
"""Write the immutable split/coverage manifest for the equal-history model tournament."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from futures_foundation.finetune.tournament import coverage_from_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-manifest", default="data/ssl_corpus_v2_6tf/MANIFEST.json")
    parser.add_argument("--output", default="output/foundation_tournament/protocol.json")
    args = parser.parse_args()
    report = coverage_from_manifest(args.corpus_manifest)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(output) + ".tmp")
    tmp.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(tmp, output)
    print(json.dumps(report, indent=2))
    print(f"[tournament] protocol -> {output}")


if __name__ == "__main__":
    main()
