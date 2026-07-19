#!/usr/bin/env python3
"""Materialize tick-costed policy events for the sealed representation screen."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from futures_foundation.execution_economics import load_execution_economics
from futures_foundation.finetune.downstream_sample import load_balanced_sample, load_row_selection
from futures_foundation.finetune.downstream_trading import (
    build_policy_events,
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
    selected_times = np.asarray(sample["decision_time_ns"])[selection["row_index"]]
    economics = load_execution_economics(
        args.execution_costs,
        evaluation_start=pd.Timestamp(int(selected_times.min()), unit="ns", tz="UTC").isoformat(),
        evaluation_end=pd.Timestamp(
            int(selected_times.max()) + 1, unit="ns", tz="UTC",
        ).isoformat(),
        required_roots=np.unique(sample["ticker"]),
    )
    slippage_ticks = (
        float(args.slippage_ticks) if args.slippage_ticks is not None
        else economics.primary_added_slippage_ticks_round_trip
    )
    arrays, metadata = build_policy_events(
        sample, selection["row_index"], sample_manifest["metadata"]["source_shards"],
        economics, slippage_ticks=slippage_ticks,
    )
    metadata["sample_sha256"] = sample_manifest["artifact"]["sha256"]
    metadata["row_selection_sha256"] = selection_manifest["artifact"]["sha256"]
    manifest = save_policy_events(Path(args.output), arrays, metadata)
    print(
        f"[complete] rows={manifest['rows']:,} contexts={manifest['contexts']:,} "
        f"policies={manifest['policies']} sha256={manifest['artifact']['sha256']}"
    )


if __name__ == "__main__":
    main()
