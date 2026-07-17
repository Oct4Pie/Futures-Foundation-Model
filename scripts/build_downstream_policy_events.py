#!/usr/bin/env python3
"""Materialize tick-costed policy events for the sealed representation screen."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from futures_foundation.finetune.downstream_sample import load_balanced_sample, load_row_selection
from futures_foundation.finetune.downstream_trading import (
    build_policy_events,
    load_execution_costs,
    save_policy_events,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample", default="output/foundation_tournament/downstream_gate_v1/balanced_sample.npz",
    )
    parser.add_argument(
        "--row-selection", default="output/foundation_tournament/downstream_gate_v1/representation_rows.npz",
    )
    parser.add_argument(
        "--output", default="output/foundation_tournament/downstream_gate_v1/screen/policy_events.npz",
    )
    parser.add_argument("--execution-costs", default="config/execution_costs.yaml")
    parser.add_argument("--slippage-ticks", type=float)
    args = parser.parse_args()
    sample, sample_manifest = load_balanced_sample(args.sample)
    selection, selection_manifest = load_row_selection(
        args.row_selection, sample_manifest=sample_manifest,
    )
    instruments, cost_manifest = load_execution_costs(
        args.execution_costs, required_tickers=np.unique(sample["ticker"]),
    )
    slippage_ticks = (
        float(args.slippage_ticks) if args.slippage_ticks is not None
        else float(cost_manifest["document"]["primary_slippage_ticks_round_trip"])
    )
    collection_path = Path(sample_manifest["metadata"]["source_collection"]["path"])
    collection = json.loads(collection_path.read_text())
    source_cost_ticks = float(collection["config"]["round_trip_cost_ticks"])
    arrays, metadata = build_policy_events(
        sample, selection["row_index"], sample_manifest["metadata"]["source_shards"],
        instruments, source_cost_ticks=source_cost_ticks, slippage_ticks=slippage_ticks,
    )
    metadata["sample_sha256"] = sample_manifest["artifact"]["sha256"]
    metadata["row_selection_sha256"] = selection_manifest["artifact"]["sha256"]
    metadata["execution_costs"] = {
        key: value for key, value in cost_manifest.items() if key != "document"
    }
    manifest = save_policy_events(Path(args.output), arrays, metadata)
    print(
        f"[complete] rows={manifest['rows']:,} contexts={manifest['contexts']:,} "
        f"policies={manifest['policies']} sha256={manifest['artifact']['sha256']}"
    )


if __name__ == "__main__":
    main()
