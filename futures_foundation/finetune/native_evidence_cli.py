"""Installed command-line interface for raw native-parity evidence generation."""
from __future__ import annotations

import argparse
import json
import sys

from .native_evidence_bundle import (
    NativeEvidenceError,
    aggregate_parity_bundles,
    create_shared_fixture,
    run_parity_bundle,
    verify_parity_bundle,
)


def _pairs(values: list[str], label: str) -> dict[str, str]:
    output = {}
    for value in values:
        if "=" not in value:
            raise NativeEvidenceError(f"{label} must use NAME=VALUE: {value!r}")
        name, item = value.split("=", 1)
        if not name or name in output:
            raise NativeEvidenceError(f"invalid or duplicate {label} name: {name!r}")
        output[name] = item
    return output


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Generate and verify raw native-parity evidence without authorizing training."
    )
    sub = value.add_subparsers(dest="action", required=True)
    fixture = sub.add_parser("fixture", help="write the shared deterministic OHLCV fixture")
    fixture.add_argument("output")
    fixture.add_argument("--seed", type=int, default=20260717)
    fixture.add_argument("--batch-size", type=int, default=4)
    fixture.add_argument("--context-length", type=int, default=512)

    run = sub.add_parser("run", help="execute one real parity command and seal its bundle")
    run.add_argument("--arm", required=True)
    run.add_argument("--track", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--registry")
    run.add_argument("--created-utc")
    run.add_argument("--artifact", action="append", default=[], metavar="NAME=PATH")
    run.add_argument("--env", action="append", default=[], metavar="NAME=VALUE")
    run.add_argument("command", nargs=argparse.REMAINDER)

    verify = sub.add_parser(
        "verify", help="rehash and verify one evidence bundle (strict by default)"
    )
    verify.add_argument("bundle")
    verify.add_argument("--registry")
    verify.add_argument(
        "--archive-only", action="store_true",
        help=(
            "verify only immutable files stored inside the archive; do not claim that "
            "the producing model/source/runtime trees are present or unchanged"
        ),
    )

    aggregate = sub.add_parser("aggregate", help="generate a reviewed evidence candidate")
    aggregate.add_argument("--output", required=True)
    aggregate.add_argument("--registry")
    aggregate.add_argument("--generated-utc")
    aggregate.add_argument("--allow-partial", action="store_true")
    aggregate.add_argument("bundles", nargs="+")
    return value


def main() -> int:
    args = parser().parse_args()
    try:
        if args.action == "fixture":
            result = create_shared_fixture(
                args.output,
                seed=args.seed,
                batch_size=args.batch_size,
                context_length=args.context_length,
            )
        elif args.action == "run":
            command = list(args.command)
            if command and command[0] == "--":
                command.pop(0)
            result = run_parity_bundle(
                arm_key=args.arm,
                track=args.track,
                command=command,
                output_directory=args.output,
                artifacts=_pairs(args.artifact, "artifact"),
                environment=_pairs(args.env, "environment"),
                created_utc=args.created_utc,
                registry_path=args.registry,
            )
        elif args.action == "verify":
            manifest = verify_parity_bundle(
                args.bundle, registry_path=args.registry,
                verify_external_artifacts=not args.archive_only,
            )[0]
            result = (
                {
                    "verification_scope": "archive_only",
                    "external_artifacts_verified": False,
                    "warning": (
                        "archive integrity only; this does not authorize or attest the "
                        "current model, source, runtime, training, or deployment artifacts"
                    ),
                    "manifest": manifest,
                }
                if args.archive_only else manifest
            )
        else:
            result = aggregate_parity_bundles(
                args.bundles,
                output_path=args.output,
                generated_utc=args.generated_utc,
                require_all_current=not args.allow_partial,
                registry_path=args.registry,
            )
    except (NativeEvidenceError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
