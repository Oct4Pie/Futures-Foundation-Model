#!/usr/bin/env python3
"""Build an externally hashable set of verified per-stream session capabilities."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from futures_foundation._authority_bundle_io import canonical_json_bytes
from futures_foundation.session_denominator import (
    load_and_verify_session_denominator,
    load_calendar_rules,
    load_consumer_contract,
    load_denominator_scope,
)
from futures_foundation.session_gap import (
    SESSION_GAP_SET_SCHEMA_VERSION,
    build_session_gap_capability,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> dict[str, object]:
    output = Path(args.output).expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite capability set: {output}")
    roots = tuple(dict.fromkeys(
        value.strip().upper() for value in args.roots.split(",") if value.strip()
    ))
    timeframes = tuple(dict.fromkeys(
        int(value.strip()) for value in args.timeframes.split(",") if value.strip()
    ))
    if not roots or not timeframes or any(value <= 0 for value in timeframes):
        raise ValueError("roots and positive integer-minute timeframes are required")

    rules = load_calendar_rules(
        args.calendar_rules,
        expected_sha256=args.calendar_rules_sha256,
    )
    consumer = load_consumer_contract(
        args.consumer_contract,
        expected_sha256=args.consumer_contract_sha256,
    )
    scope = load_denominator_scope(
        args.denominator_scope,
        expected_sha256=args.denominator_scope_sha256,
        rules=rules,
        consumer=consumer,
    )
    denominator = load_and_verify_session_denominator(
        args.session_denominator,
        expected_sha256=args.session_denominator_sha256,
        rules=rules,
        scope=scope,
        consumer=consumer,
    )

    capabilities = {}
    for root in roots:
        for minutes in timeframes:
            capability = build_session_gap_capability(
                denominator,
                rules=rules,
                scope=scope,
                consumer=consumer,
                root=root,
                expected_delta=f"{minutes}min",
            )
            capabilities[f"{root}@{minutes}min"] = capability.manifest()
    document = {
        "schema_version": SESSION_GAP_SET_SCHEMA_VERSION,
        "purpose": "verified_session_continuity_for_governed_windows",
        "capabilities": dict(sorted(capabilities.items())),
        "training_admitted": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(output) + f".{os.getpid()}.tmp")
    temporary.write_bytes(canonical_json_bytes(document))
    os.replace(temporary, output)
    return {
        "status": "complete",
        "path": str(output),
        "sha256": _sha256(output),
        "bytes": output.stat().st_size,
        "streams": len(capabilities),
        "training_admitted": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calendar-rules", required=True)
    parser.add_argument("--calendar-rules-sha256", required=True)
    parser.add_argument("--consumer-contract", required=True)
    parser.add_argument("--consumer-contract-sha256", required=True)
    parser.add_argument("--denominator-scope", required=True)
    parser.add_argument("--denominator-scope-sha256", required=True)
    parser.add_argument("--session-denominator", required=True)
    parser.add_argument("--session-denominator-sha256", required=True)
    parser.add_argument("--roots", required=True, help="comma-separated roots")
    parser.add_argument(
        "--timeframes", required=True,
        help="comma-separated integer-minute bar sizes",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
