#!/usr/bin/env python3
"""Audit cross-layer model configuration without granting execution or training admission."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from futures_foundation.finetune.native_contracts import canonical_json
from futures_foundation.finetune.native_configuration_audit import (
    build_native_configuration_audit,
    validate_native_configuration_audit,
)


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parity-aggregate",
        help="optional current-registry complete native parity aggregate",
    )
    parser.add_argument("--output", help="optional canonical JSON output path")
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    aggregate = (
        None if not args.parity_aggregate
        else str(Path(args.parity_aggregate).expanduser().resolve())
    )
    report = build_native_configuration_audit(parity_aggregate_path=aggregate)
    report = validate_native_configuration_audit(
        report, parity_aggregate_path=aggregate,
    )
    if args.output:
        _atomic_write(Path(args.output).resolve(), canonical_json(report) + b"\n")
    if args.compact:
        value = {
            "schema_version": report["schema_version"],
            "audit_sha256": report["audit_sha256"],
            "registry_sha256": report["registry_sha256"],
            "catalog_sha256": report["catalog_sha256"],
            "counts": report["counts"],
            "configuration_integrity_passed": report["configuration_integrity_passed"],
            "current_inference_parity_complete": report["current_inference_parity_complete"],
            "all_models_execution_ready": report["all_models_execution_ready"],
            "all_training_routes_execution_ready": report[
                "all_training_routes_execution_ready"
            ],
            "training_admitted": report["training_admitted"],
            "live_trading_ready": report["live_trading_ready"],
            "discrepancies": report["discrepancies"],
        }
    else:
        value = report
    print(json.dumps(value, indent=2, sort_keys=True))
    if not report["configuration_integrity_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
