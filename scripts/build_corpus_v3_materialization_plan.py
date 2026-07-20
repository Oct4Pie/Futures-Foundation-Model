#!/usr/bin/env python3
"""Build split-scoped inventory and a non-executable Corpus-v3 plan."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from futures_foundation._authority_bundle_io import (
    AuthorityBundleIOError,
    read_canonical_json_file,
    require_sha256,
)
from futures_foundation.corpus_v3_materialization_plan import (
    build_materialization_plan_v1,
    build_split_scoped_inventory_v1,
    load_inventory_observations_v1,
    validate_materialization_plan_v1,
    validate_split_scoped_inventory_v1,
    write_materialization_artifact,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-requests", required=True)
    parser.add_argument("--expected-requests-sha256", required=True)
    parser.add_argument("--inventory-observations", required=True)
    parser.add_argument("--inventory-observations-sha256", required=True)
    parser.add_argument(
        "--inventory-output",
        default=str(ROOT / "output/corpus_v3/split_scoped_inventory_v1.json"),
    )
    parser.add_argument(
        "--plan-output",
        default=str(ROOT / "output/corpus_v3/materialization_plan_v1.json"),
    )
    args = parser.parse_args()

    try:
        _, expected, physical = read_canonical_json_file(
            Path(args.expected_requests).expanduser().resolve(),
            label="Corpus-v3 expected-request denominator",
            max_bytes=512 * 1024 * 1024,
            max_nodes=8_000_000,
            max_depth=30,
        )
    except AuthorityBundleIOError as exc:
        parser.error(str(exc))
    if physical != require_sha256(
        args.expected_requests_sha256,
        "expected-request denominator physical SHA-256",
    ):
        parser.error("expected-request denominator physical SHA-256 differs")
    rows, observations = load_inventory_observations_v1(
        args.inventory_observations,
        expected_physical_sha256=args.inventory_observations_sha256,
        expected_request_denominator=expected,
    )
    inventory = build_split_scoped_inventory_v1(
        expected_request_denominator=expected,
        inventory_rows=rows,
    )
    validate_split_scoped_inventory_v1(
        inventory, expected_request_denominator=expected,
    )
    plan = build_materialization_plan_v1(
        expected_request_denominator=expected,
        inventory=inventory,
    )
    validate_materialization_plan_v1(
        plan, expected_request_denominator=expected, inventory=inventory,
    )
    inventory_path = write_materialization_artifact(
        inventory, args.inventory_output,
    )
    plan_path = write_materialization_artifact(plan, args.plan_output)
    print(json.dumps({
        "inventory_sha256": inventory["inventory_sha256"],
        "plan_sha256": plan["plan_sha256"],
        "observations": observations,
        "counts": plan["counts"],
        "execution_status": plan["execution_status"],
        "materialization_admitted": plan["materialization_admitted"],
        "training_admitted": plan["training_admitted"],
        "inventory_output": str(inventory_path),
        "plan_output": str(plan_path),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
