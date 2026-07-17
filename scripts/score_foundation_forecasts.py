#!/usr/bin/env python3
"""Score point forecasts against the one immutable validation-window artifact."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from futures_foundation.finetune.foundation_eval import (
    load_window_artifact, persistence_prediction, score_prediction_artifact,
)
from futures_foundation.finetune.kronos_eval import evaluate_predictions


def _atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", required=True)
    parser.add_argument("--prediction", action="append", default=[], help="arm=artifact.npz")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    windows, window_manifest = load_window_artifact(args.windows)
    results = {"persistence": evaluate_predictions(windows, persistence_prediction(windows))}
    artifacts = {}
    for item in args.prediction:
        if "=" not in item:
            raise ValueError("prediction entries must use arm=artifact.npz")
        arm, path = item.split("=", 1)
        if arm in results:
            raise ValueError(f"duplicate forecast arm: {arm}")
        prediction_manifest, metrics = score_prediction_artifact(
            windows, window_manifest, path,
        )
        if prediction_manifest["arm"] != arm:
            raise ValueError(f"prediction arm mismatch for {path}")
        results[arm], artifacts[arm] = metrics, prediction_manifest
    _atomic_json(args.output, {
        "schema_version": "ffm_shared_forecast_scorecard_v1",
        "window_manifest": window_manifest, "prediction_artifacts": artifacts,
        "results": results, "oos_read": False,
    })


if __name__ == "__main__":
    main()
