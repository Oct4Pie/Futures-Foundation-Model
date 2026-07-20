#!/usr/bin/env python3
"""Emit the current non-authorizing architecture-native training readiness report."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from futures_foundation.finetune.native_contracts import canonical_json
from futures_foundation.finetune.native_training_readiness import (
    build_training_readiness_report,
    validate_training_readiness_report,
)


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit every canonical native-training route without granting admission",
    )
    parser.add_argument("--output", help="optional canonical JSON output path")
    parser.add_argument(
        "--smoke-evidence", action="append", default=[], metavar="ROUTE=PATH",
        help="repeat for each raw route smoke bundle to reverify",
    )
    parser.add_argument(
        "--pilot-evidence", action="append", default=[], metavar="ROUTE=PATH",
        help="repeat for each bounded route pilot bundle to reverify",
    )
    parser.add_argument(
        "--downstream-screen", action="append", default=[], metavar="ROUTE=PATH",
        help="repeat for each verified native incremental-screen report",
    )
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    smoke_evidence = {}
    for item in args.smoke_evidence:
        if "=" not in item:
            parser.error("--smoke-evidence must use ROUTE=PATH")
        route, path = item.split("=", 1)
        if not route or not path or route in smoke_evidence:
            parser.error(f"invalid or duplicate smoke evidence route: {route!r}")
        smoke_evidence[route] = str(Path(path).expanduser().resolve())

    pilot_evidence = {}
    for item in args.pilot_evidence:
        if "=" not in item:
            parser.error("--pilot-evidence must use ROUTE=PATH")
        route, path = item.split("=", 1)
        if not route or not path or route in pilot_evidence:
            parser.error(f"invalid or duplicate pilot evidence route: {route!r}")
        pilot_evidence[route] = str(Path(path).expanduser().resolve())

    downstream_screen = {}
    for item in args.downstream_screen:
        if "=" not in item:
            parser.error("--downstream-screen must use ROUTE=PATH")
        route, path = item.split("=", 1)
        if not route or not path or route in downstream_screen:
            parser.error(f"invalid or duplicate downstream screen route: {route!r}")
        downstream_screen[route] = str(Path(path).expanduser().resolve())

    report = build_training_readiness_report(
        smoke_evidence_paths=smoke_evidence,
        pilot_evidence_paths=pilot_evidence,
        downstream_screen_paths=downstream_screen,
    )
    report = validate_training_readiness_report(
        report,
        smoke_evidence_paths=smoke_evidence,
        pilot_evidence_paths=pilot_evidence,
        downstream_screen_paths=downstream_screen,
    )
    if args.output:
        _atomic_write(Path(args.output).resolve(), canonical_json(report) + b"\n")
    if args.compact:
        summary = {
            "schema_version": report["schema_version"],
            "readiness_sha256": report["readiness_sha256"],
            "training_data_authority": report["training_data_authority"],
            "methodology": report["methodology"],
            "counts": report["counts"],
            "pilot_admitted": report["pilot_admitted"],
            "training_admitted": report["training_admitted"],
            "live_trading_ready": report["live_trading_ready"],
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
