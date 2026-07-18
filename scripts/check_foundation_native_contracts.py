#!/usr/bin/env python3
"""Inspect and verify executable foundation-model native contracts.

This command distinguishes technical native-track validity from runtime authorization.
Execution still requires a current evidence-bound report and two independent approvals.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from futures_foundation.finetune.native_contracts import (
    REGISTRY_PATH,
    all_arms,
    dossier_sha256,
    evidence_sha256,
    get_dossier,
    historical_disposition,
    load_registry,
    registry_sha256,
    technical_evidence,
    verify_admission_report,
)


def _print_json(value) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


def _keyed_paths(values, *, label: str) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for value in values or ():
        if "=" not in value:
            raise ValueError(f"{label} must use name=/path")
        name, raw_path = (part.strip() for part in value.split("=", 1))
        if not name or not raw_path or name in output:
            raise ValueError(f"{label} names and paths must be nonempty and unique")
        output[name] = Path(raw_path).expanduser().resolve()
    return output


def list_contracts(args) -> None:
    registry = load_registry(args.registry)
    rows = {}
    for key, arm in all_arms(args.registry).items():
        rows[key] = {
            "family": arm.family,
            "model_id": arm.model_id,
            "model_revision": arm.model_revision,
            "tokenizer": ({
                "id": arm.tokenizer_id, "revision": arm.tokenizer_revision,
            } if arm.tokenizer_id else None),
            "pin_complete": arm.pin_complete,
            "overall_status": arm.overall_status,
            "training_admitted": arm.training_admitted,
            "tracks": {
                capability.track: {
                    "status": capability.status,
                    "reason": capability.reason,
                    "technical_evidence_id": capability.evidence_id,
                    "training_admitted": capability.training_admitted,
                    "runtime_authorized": False,
                }
                for capability in arm.tracks
            },
            "historical": historical_disposition(key, args.registry),
            "dossier_sha256": dossier_sha256(key, args.registry),
        }
    _print_json({
        "schema_version": registry["schema_version"],
        "methodology_commit": registry["methodology_commit"],
        "registry_path": str(Path(args.registry).resolve()),
        "registry_sha256": registry_sha256(args.registry),
        "evidence_sha256": evidence_sha256(args.registry),
        "runtime_authorization": "requires_current_report_and_two_independent_approvals",
        "models": rows,
    })


def show_dossier(args) -> None:
    dossier = get_dossier(args.arm, args.registry)
    evidence = {}
    for track, capability in dossier["tracks"].items():
        if capability.get("evidence_id"):
            evidence_id, record, checks = technical_evidence(
                args.arm, track, args.registry
            )
            evidence[track] = {
                "evidence_id": evidence_id,
                "record": record,
                "resolved_checks": checks,
            }
    _print_json({
        "arm_key": args.arm,
        "registry_sha256": registry_sha256(args.registry),
        "evidence_sha256": evidence_sha256(args.registry),
        "dossier_sha256": dossier_sha256(args.arm, args.registry),
        "dossier": dossier,
        "technical_evidence": evidence,
        "runtime_authorized": False,
        "historical": historical_disposition(args.arm, args.registry),
    })


def verify_report(args) -> None:
    artifacts = _keyed_paths(args.artifact, label="artifact")
    report = verify_admission_report(
        args.report,
        arm_key=args.arm,
        track=args.track,
        route=args.route,
        require_training=args.training,
        required_artifacts=artifacts,
        path=args.registry,
    )
    _print_json({
        "verified": True,
        "arm_key": args.arm,
        "track": args.track,
        "route": args.route,
        "training": bool(args.training),
        "verified_artifacts": {
            name: str(path) for name, path in artifacts.items()
        },
        "report_integrity": report["integrity"],
        "registry_sha256": report["registry_sha256"],
        "dossier_sha256": report["dossier_sha256"],
        "evidence_registry_sha256": report["evidence_registry_sha256"],
        "technical_evidence_id": report["technical_evidence_id"],
        "approvals": report["approvals"],
    })


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default=str(REGISTRY_PATH))
    commands = parser.add_subparsers(dest="command", required=True)

    listing = commands.add_parser("list", help="show every model and track status")
    listing.set_defaults(func=list_contracts)

    dossier = commands.add_parser("show", help="show one authoritative dossier")
    dossier.add_argument("--arm", required=True)
    dossier.set_defaults(func=show_dossier)

    verify = commands.add_parser("verify", help="verify a hash-bound admission report")
    verify.add_argument("--arm", required=True)
    verify.add_argument("--track", choices=("F", "R", "C", "B", "D"), required=True)
    verify.add_argument("--route")
    verify.add_argument("--report", required=True)
    verify.add_argument(
        "--artifact", action="append", default=[],
        help="repeat name=/path to require an exact report-bound artifact SHA-256",
    )
    verify.add_argument("--training", action="store_true")
    verify.set_defaults(func=verify_report)
    return parser


def main() -> None:
    args = _parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
