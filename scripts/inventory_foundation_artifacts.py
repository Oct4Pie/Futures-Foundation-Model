#!/usr/bin/env python3
"""Verify and inventory frozen inputs for the downstream foundation-model gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from futures_foundation.finetune.artifact_inventory import build_frozen_inventory, write_inventory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, help="canonical representation_results.json")
    parser.add_argument("--output", required=True, help="inventory JSON to write atomically")
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()
    inventory = build_frozen_inventory(args.results, repo=args.repo)
    write_inventory(args.output, inventory)
    print(json.dumps({
        "passed": inventory["passed"], "counts": inventory["counts"],
        "errors": inventory["errors"], "output": str(Path(args.output).resolve()),
    }, indent=2))
    if not inventory["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
