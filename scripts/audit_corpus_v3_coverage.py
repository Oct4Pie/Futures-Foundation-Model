#!/usr/bin/env python3
"""Verify the Corpus v3 pins and produce an outcome-blind root/year coverage audit."""
from __future__ import annotations

import argparse
import json

from futures_foundation.corpus_v3 import build_coverage_audit, load_contract, write_coverage_audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", default="config/corpus_v3/contract.json")
    parser.add_argument("--output", default="config/corpus_v3/coverage_audit.json")
    args = parser.parse_args()
    report = build_coverage_audit(load_contract(args.contract))
    path = write_coverage_audit(report, args.output)
    print(json.dumps({
        "output": str(path.resolve()),
        "report_sha256": report["report_sha256"],
        "candidate_roots": report["candidate_roots"],
        "selected_roots": report["selected_roots"],
        "diagnostic_flagged_roots": report["diagnostic_flagged_roots"],
        "selection_status": report["selection_status"],
    }, indent=2))


if __name__ == "__main__":
    main()
