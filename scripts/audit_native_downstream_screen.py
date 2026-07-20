#!/usr/bin/env python3
"""Reverify and aggregate the exact common-information incremental screens."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from futures_foundation.finetune.native_contracts import canonical_json
from futures_foundation.finetune.native_downstream_screen import (
    COMMON_INFORMATION_ROUTES,
    build_screen_collection,
    validate_screen_collection,
)


def _mapping(values: list[str]) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for value in values:
        route, separator, path = value.partition("=")
        if not separator or route not in COMMON_INFORMATION_ROUTES or not path:
            raise argparse.ArgumentTypeError(
                "--report values must be a canonical ROUTE=PATH pair"
            )
        if route in output:
            raise argparse.ArgumentTypeError(f"duplicate report route: {route}")
        output[route] = Path(path).expanduser().resolve()
    if set(output) != COMMON_INFORMATION_ROUTES:
        missing = sorted(COMMON_INFORMATION_ROUTES - set(output))
        extra = sorted(set(output) - COMMON_INFORMATION_ROUTES)
        raise argparse.ArgumentTypeError(
            f"report route closure mismatch: missing={missing}, extra={extra}"
        )
    return output


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + f".{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report", action="append", default=[], metavar="ROUTE=PATH",
        help="repeat exactly once for each common-information route",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    try:
        paths = _mapping(args.report)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    collection = validate_screen_collection(
        build_screen_collection(paths), report_paths=paths,
    )
    output = Path(args.output).expanduser().resolve()
    _atomic_write(output, canonical_json(collection) + b"\n")
    if args.compact:
        value = {
            "schema_version": collection["schema_version"],
            "collection_sha256": collection["collection_sha256"],
            "counts": collection["counts"],
            "surviving_routes": collection["surviving_routes"],
            "nonlinear_sensitivity_funded_routes": (
                collection["nonlinear_sensitivity_funded_routes"]
            ),
            "full_training_admitted": collection["full_training_admitted"],
            "oos_admitted": collection["oos_admitted"],
            "live_trading_ready": collection["live_trading_ready"],
        }
        print(json.dumps(value, indent=2, sort_keys=True))
    else:
        print(json.dumps(collection, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
