#!/usr/bin/env python3
"""Build the non-authorizing Corpus-v3 sessionized coverage/yield audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from futures_foundation.corpus_v3_session_audit import (
    build_sessionized_coverage_audit,
    load_verified_exports_from_index,
    validate_sessionized_coverage_audit,
    write_sessionized_coverage_audit,
)
from futures_foundation.session_denominator_bundle import (
    load_and_verify_session_denominator_bundle_v2,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        default=str(ROOT / "config/corpus_v3/contract.json"),
    )
    parser.add_argument("--denominator-bundle", required=True)
    parser.add_argument("--denominator-manifest-sha256", required=True)
    parser.add_argument("--calendar-rules", required=True)
    parser.add_argument("--calendar-rules-sha256", required=True)
    parser.add_argument("--scope-v2", required=True)
    parser.add_argument("--scope-v2-sha256", required=True)
    parser.add_argument("--consumer-scope", required=True)
    parser.add_argument("--consumer-scope-sha256", required=True)
    parser.add_argument("--export-index", required=True)
    parser.add_argument(
        "--output",
        default=str(ROOT / "output/corpus_v3/sessionized_coverage_audit.json"),
    )
    parser.add_argument(
        "--allow-test-contract",
        action="store_true",
        help="test-fixture seam only; production verification requires the canonical contract",
    )
    args = parser.parse_args()

    denominator = load_and_verify_session_denominator_bundle_v2(
        args.denominator_bundle,
        expected_manifest_sha256=args.denominator_manifest_sha256,
        calendar_rules_path=args.calendar_rules,
        calendar_rules_sha256=args.calendar_rules_sha256,
        scope_v2_path=args.scope_v2,
        scope_v2_sha256=args.scope_v2_sha256,
        consumer_scope_path=args.consumer_scope,
        consumer_scope_sha256=args.consumer_scope_sha256,
    )
    exports, index_identity = load_verified_exports_from_index(
        args.export_index,
        contract_path=args.contract,
        allow_test_contract=args.allow_test_contract,
    )
    report = build_sessionized_coverage_audit(
        contract_path=args.contract,
        denominator=denominator,
        exports=exports,
        export_index_identity=index_identity,
        allow_test_contract=args.allow_test_contract,
    )
    validate_sessionized_coverage_audit(
        report,
        contract_path=args.contract,
        denominator=denominator,
        exports=exports,
        export_index_identity=index_identity,
        allow_test_contract=args.allow_test_contract,
    )
    output = write_sessionized_coverage_audit(report, args.output)
    print(json.dumps({
        "schema_version": report["schema_version"],
        "audit_sha256": report["audit_sha256"],
        "counts": report["counts"],
        "complete_against_denominator": report["complete_against_denominator"],
        "selected_roots": report["selected_roots"],
        "materialization_admitted": report["materialization_admitted"],
        "training_admitted": report["training_admitted"],
        "output": str(output),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
