#!/usr/bin/env python3
"""Build the governed Corpus-v3 expected-request denominator from exact parents."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from futures_foundation.corpus_v3_contract_lifecycle import (
    load_and_verify_contract_lifecycle_v2,
)
from futures_foundation.corpus_v3_expected_requests import (
    build_expected_request_denominator_v1,
    validate_expected_request_denominator_v1,
    write_expected_request_denominator_v1,
)
from futures_foundation.corpus_v3_producer_governance import (
    load_and_verify_frozen_split_use_contract_v1,
)
from futures_foundation.corpus_v3_provider_candidates import (
    load_and_verify_provider_candidate_universe_v1,
)
from futures_foundation.session_denominator_bundle import (
    load_and_verify_session_denominator_bundle_v2,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--producer-governance", required=True)
    parser.add_argument("--producer-governance-sha256", required=True)
    parser.add_argument("--split-contract", required=True)
    parser.add_argument("--split-contract-sha256", required=True)
    parser.add_argument("--provider-evidence-root", required=True)
    parser.add_argument("--provider-manifest", required=True)
    parser.add_argument("--provider-manifest-sha256", required=True)
    parser.add_argument("--provider-universe", required=True)
    parser.add_argument("--provider-universe-sha256", required=True)
    parser.add_argument("--lifecycle-registry", required=True)
    parser.add_argument("--lifecycle-registry-sha256", required=True)
    parser.add_argument("--lifecycle", required=True)
    parser.add_argument("--lifecycle-sha256", required=True)
    parser.add_argument("--session-denominator-bundle", required=True)
    parser.add_argument("--session-denominator-manifest-sha256", required=True)
    parser.add_argument("--calendar-rules", required=True)
    parser.add_argument("--calendar-rules-sha256", required=True)
    parser.add_argument("--denominator-scope", required=True)
    parser.add_argument("--denominator-scope-sha256", required=True)
    parser.add_argument("--consumer-scope", required=True)
    parser.add_argument("--consumer-scope-sha256", required=True)
    parser.add_argument(
        "--output",
        default=str(ROOT / "output/corpus_v3/expected_request_denominator_v1.json"),
    )
    args = parser.parse_args()

    split = load_and_verify_frozen_split_use_contract_v1(
        args.split_contract,
        expected_sha256=args.split_contract_sha256,
        producer_governance_path=args.producer_governance,
        producer_governance_sha256=args.producer_governance_sha256,
    )
    candidates = load_and_verify_provider_candidate_universe_v1(
        args.provider_evidence_root,
        manifest_path=args.provider_manifest,
        manifest_sha256=args.provider_manifest_sha256,
        split_capability=split,
    )
    lifecycle = load_and_verify_contract_lifecycle_v2(
        args.lifecycle,
        lifecycle_sha256=args.lifecycle_sha256,
        registry_path=args.lifecycle_registry,
        registry_sha256=args.lifecycle_registry_sha256,
        provider_universe_path=args.provider_universe,
        provider_universe_sha256=args.provider_universe_sha256,
        split_capability=split,
        provider_candidate_capability=candidates,
    )
    session = load_and_verify_session_denominator_bundle_v2(
        args.session_denominator_bundle,
        expected_manifest_sha256=args.session_denominator_manifest_sha256,
        calendar_rules_path=args.calendar_rules,
        calendar_rules_sha256=args.calendar_rules_sha256,
        scope_v2_path=args.denominator_scope,
        scope_v2_sha256=args.denominator_scope_sha256,
        consumer_scope_path=args.consumer_scope,
        consumer_scope_sha256=args.consumer_scope_sha256,
    )
    report = build_expected_request_denominator_v1(
        split_capability=split,
        session_denominator_capability=session,
        lifecycle_capability=lifecycle,
    )
    validate_expected_request_denominator_v1(
        report,
        split_capability=split,
        session_denominator_capability=session,
        lifecycle_capability=lifecycle,
    )
    output = write_expected_request_denominator_v1(report, args.output)
    print(json.dumps({
        "schema_version": report["schema_version"],
        "expected_request_denominator_sha256": report[
            "expected_request_denominator_sha256"
        ],
        "counts": report["counts"],
        "production_admitted": report["production_admitted"],
        "materialization_admitted": report["materialization_admitted"],
        "training_admitted": report["training_admitted"],
        "output": str(output),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
