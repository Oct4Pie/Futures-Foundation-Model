#!/usr/bin/env python3
"""Paired economic analysis of a sealed downstream trading benchmark."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from futures_foundation.finetune.downstream_trading import load_policy_events


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def paired_utility_interval(
    delta: np.ndarray,
    time_ns: np.ndarray,
    *,
    block_days: int = 7,
    repetitions: int = 2000,
    seed: int = 20260716,
) -> dict:
    """Cluster-bootstrap mean R/opportunity using fixed UTC calendar blocks."""
    delta = np.asarray(delta, np.float64)
    time_ns = np.asarray(time_ns, np.int64)
    if delta.ndim != 1 or time_ns.shape != delta.shape or not len(delta):
        raise ValueError("delta/time must be equal non-empty vectors")
    if not np.isfinite(delta).all() or block_days <= 0 or repetitions <= 0:
        raise ValueError("invalid paired utility inputs")
    block_ns = int(block_days) * 86_400 * 1_000_000_000
    block = time_ns // block_ns
    unique, inverse = np.unique(block, return_inverse=True)
    sums = np.bincount(inverse, weights=delta, minlength=len(unique))
    counts = np.bincount(inverse, minlength=len(unique)).astype(np.float64)
    rng = np.random.default_rng(seed)
    draws = np.empty(repetitions, np.float64)
    for index in range(repetitions):
        sampled = rng.integers(0, len(unique), size=len(unique))
        draws[index] = sums[sampled].sum() / counts[sampled].sum()
    low, high = np.quantile(draws, (0.025, 0.975))
    point = float(delta.mean())
    return {
        "delta_r_per_candidate": point,
        "delta_total_r": float(delta.sum()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "bootstrap_positive_probability": float(np.mean(draws > 0)),
        "bootstrap_two_sided_p": float(
            min(1.0, 2.0 * min(np.mean(draws > 0), np.mean(draws <= 0)))
        ),
        "calendar_blocks": int(len(unique)),
        "candidates": int(len(delta)),
    }


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Return monotone Benjamini-Hochberg adjusted p-values."""
    values = np.asarray(p_values, np.float64)
    if values.ndim != 1 or not len(values) or np.any((values < 0) | (values > 1)):
        raise ValueError("p-values must be a non-empty vector in [0, 1]")
    order = np.argsort(values, kind="stable")
    ranked = values[order] * len(values) / np.arange(1, len(values) + 1)
    adjusted_ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty_like(values)
    adjusted[order] = np.minimum(adjusted_ranked, 1.0)
    return adjusted


def fixed_cost_metrics(
    gross_r: np.ndarray,
    fee_r: np.ndarray,
    slippage_r_per_tick: np.ndarray,
    slippage_ticks: float,
) -> dict:
    """Reprice fixed executed trades without refitting or changing selections."""
    realized = (
        np.asarray(gross_r, np.float64)
        - np.asarray(fee_r, np.float64)
        - np.asarray(slippage_r_per_tick, np.float64) * float(slippage_ticks)
    )
    if realized.ndim != 1 or not len(realized) or not np.isfinite(realized).all():
        raise ValueError("fixed cost inputs must produce a finite non-empty vector")
    wins, losses = realized[realized > 0], realized[realized < 0]
    gross_loss = -float(losses.sum())
    return {
        "executed": int(len(realized)),
        "mean_r": float(realized.mean()),
        "total_r": float(realized.sum()),
        "win_rate": float(np.mean(realized > 0)),
        "profit_factor": float(wins.sum() / gross_loss) if gross_loss > 0 else None,
    }


def slippage_r_per_round_trip_tick(events: dict) -> np.ndarray:
    """Convert one round-trip tick to R directly from the sealed risk geometry."""
    risk_ticks = np.asarray(events["risk_ticks"], np.float64)
    if risk_ticks.ndim != 1 or not len(risk_ticks) or np.any(risk_ticks <= 0):
        raise ValueError("risk ticks must be a positive non-empty vector")
    return 1.0 / risk_ticks


def _load_results(path: str | Path) -> tuple[dict, dict, dict]:
    results_path = Path(path).resolve()
    report = json.loads(results_path.read_text())
    if report.get("status") != "complete" or report.get("oos_read") is not False:
        raise ValueError("trading report is incomplete or not development-only")
    prediction_path = Path(report["predictions"]["path"])
    if _sha256(prediction_path) != report["predictions"]["sha256"]:
        raise ValueError("trading prediction hash mismatch")
    with np.load(prediction_path, allow_pickle=False) as saved:
        predictions = {key: saved[key] for key in saved.files}
    if len(predictions.get("event_row", ())) != report["predictions"]["rows"]:
        raise ValueError("trading prediction row-count mismatch")
    return report, predictions, {
        "results": str(results_path),
        "results_sha256": _sha256(results_path),
        "predictions": str(prediction_path),
        "predictions_sha256": _sha256(prediction_path),
    }


def _aligned_arm_rows(
    predictions: dict, policy_index: int, left_index: int, right_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    def rows(arm_index: int) -> np.ndarray:
        selected = np.flatnonzero(
            (predictions["policy_index"] == policy_index)
            & (predictions["arm_index"] == arm_index)
        )
        return selected[np.lexsort((
            predictions["fold"][selected], predictions["event_row"][selected],
        ))]

    left, right = rows(left_index), rows(right_index)
    if not len(left) or not len(right):
        raise ValueError("paired arm contains no predictions")
    for key in ("event_row", "fold"):
        if not np.array_equal(predictions[key][left], predictions[key][right]):
            raise ValueError(f"paired trading mismatch for {key}")
    return left, right


def run(args) -> dict:
    benchmark, predictions, identity = _load_results(args.results)
    events, event_manifest = load_policy_events(args.policy_events)
    if benchmark["policy_events_sha256"] != event_manifest["artifact"]["sha256"]:
        raise ValueError("trading report/policy event identity mismatch")
    policy_names = [str(value) for value in predictions["policy_names"]]
    arm_names = [str(value) for value in predictions["arm_names"]]
    if args.baseline_arm not in arm_names:
        raise ValueError(f"baseline arm {args.baseline_arm!r} is absent")
    baseline_index = arm_names.index(args.baseline_arm)
    scenarios = tuple(float(item.strip()) for item in args.slippage_scenarios.split(",") if item.strip())
    if not scenarios or any(value < 0 for value in scenarios):
        raise ValueError("slippage scenarios must be nonnegative")
    primary_ticks = float(event_manifest["slippage_ticks_round_trip"])
    if primary_ticks < 0:
        raise ValueError("primary policy artifact must contain nonnegative slippage ticks")
    # Derive the sensitivity unit from risk geometry.  Dividing the stored slippage R by
    # primary_ticks is undefined for the deliberate zero-slippage primary configuration.
    slippage_per_tick = slippage_r_per_round_trip_tick(events)

    summary_by_key = {
        (str(row["policy"]), str(row["arm"])): row for row in benchmark["summary"]
    }
    comparisons, skipped, cost_sensitivity = [], [], []
    for policy_index, policy in enumerate(policy_names):
        baseline_summary = summary_by_key.get((policy, args.baseline_arm))
        if baseline_summary is None:
            skipped.append({"policy": policy, "reason": "baseline_summary_absent"})
            continue
        for arm_index, arm in enumerate(arm_names):
            arm_summary = summary_by_key.get((policy, arm))
            if arm_summary is None:
                continue
            arm_rows = np.flatnonzero(
                (predictions["policy_index"] == policy_index)
                & (predictions["arm_index"] == arm_index)
                & np.asarray(predictions["executed"], bool)
            )
            executed_events = predictions["event_row"][arm_rows]
            if len(executed_events):
                for scenario in scenarios:
                    metrics = fixed_cost_metrics(
                        events["gross_r"][executed_events], events["fee_r"][executed_events],
                        slippage_per_tick[executed_events], scenario,
                    )
                    if np.isclose(scenario, primary_ticks) and not np.isclose(
                        metrics["mean_r"], arm_summary["mean_r"], atol=1e-7,
                    ):
                        raise ValueError("primary cost repricing does not reproduce benchmark")
                    cost_sensitivity.append({
                        "policy": policy, "arm": arm,
                        "slippage_ticks_round_trip": scenario, **metrics,
                    })
            if arm_index == baseline_index:
                continue
            left, right = _aligned_arm_rows(
                predictions, policy_index, arm_index, baseline_index,
            )
            event_rows = predictions["event_row"][left]
            realized = np.asarray(events["realized_r"][event_rows], np.float64)
            left_utility = realized * np.asarray(predictions["executed"][left], bool)
            right_utility = realized * np.asarray(predictions["executed"][right], bool)
            delta = left_utility - right_utility
            folds = np.asarray(predictions["fold"][left], np.int8)
            fold_delta = [
                float(delta[folds == fold].mean())
                for fold in sorted(int(value) for value in np.unique(folds))
            ]
            interval = paired_utility_interval(
                delta, events["signal_time_ns"][event_rows],
                block_days=args.block_days, repetitions=args.repetitions,
                seed=args.seed + policy_index * 101 + arm_index,
            )
            comparisons.append({
                "policy": policy,
                "arm": arm,
                "baseline_arm": args.baseline_arm,
                **interval,
                "positive_fold_fraction": float(np.mean(np.asarray(fold_delta) > 0)),
                "worst_fold_delta_r_per_candidate": float(min(fold_delta)),
                "fold_delta_r_per_candidate": fold_delta,
                "arm_executed": int(arm_summary["executed"]),
                "arm_mean_r": arm_summary["mean_r"],
                "arm_profit_factor": arm_summary["profit_factor"],
                "baseline_executed": int(baseline_summary["executed"]),
                "baseline_mean_r": baseline_summary["mean_r"],
                "baseline_profit_factor": baseline_summary["profit_factor"],
                "economically_positive": bool(
                    arm_summary["mean_r"] is not None
                    and arm_summary["mean_r"] > 0
                    and arm_summary["profit_factor"] is not None
                    and arm_summary["profit_factor"] > 1
                ),
                "significant_positive_lift": bool(interval["ci95_low"] > 0),
            })

    if comparisons:
        adjusted = benjamini_hochberg(np.asarray([
            row["bootstrap_two_sided_p"] for row in comparisons
        ]))
        for row, q_value in zip(comparisons, adjusted):
            row["fdr_q_value"] = float(q_value)
            row["significant_positive_lift_fdr_05"] = bool(
                row["delta_r_per_candidate"] > 0 and q_value < 0.05
            )
    ranked = sorted(
        comparisons,
        key=lambda row: (
            bool(row["economically_positive"]), row["delta_r_per_candidate"],
        ),
        reverse=True,
    )
    report = {
        "schema_version": "ffm_downstream_trading_analysis_v1",
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "oos_read": False,
        "source": identity,
        "policy_events_sha256": event_manifest["artifact"]["sha256"],
        "configuration": {
            "baseline_arm": args.baseline_arm,
            "block_days": args.block_days,
            "repetitions": args.repetitions,
            "seed": args.seed,
            "utility_definition": "realized R when executed, otherwise zero R",
            "delta_definition": "arm utility minus baseline utility on matched candidates",
            "resampling_unit": "UTC calendar block",
            "slippage_scenarios_round_trip_ticks": list(scenarios),
            "cost_sensitivity_contract": (
                "freeze primary-cost out-of-fold selections and reprice exact executions; no refit"
            ),
        },
        "counts": {
            "policies_in_prediction_artifact": len(policy_names),
            "arms": len(arm_names),
            "comparisons": len(comparisons),
            "economically_positive": int(sum(row["economically_positive"] for row in comparisons)),
            "significant_positive_lift": int(
                sum(row["significant_positive_lift"] for row in comparisons)
            ),
            "significant_positive_lift_fdr_05": int(
                sum(row["significant_positive_lift_fdr_05"] for row in comparisons)
            ),
        },
        "comparisons": ranked,
        "cost_sensitivity": cost_sensitivity,
        "skipped": skipped,
    }
    _atomic_json(Path(args.output).resolve(), report)
    print(json.dumps({
        "status": "complete", **report["counts"],
        "output": str(Path(args.output).resolve()),
        "sha256": _sha256(args.output),
    }, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", required=True,
        help="Path to trading_results.json from benchmark_downstream_trading.py",
    )
    parser.add_argument(
        "--policy-events",
        default="output/foundation_tournament/downstream_gate_v1/screen/policy_events.npz",
    )
    parser.add_argument("--baseline-arm", default="causal_xgb")
    parser.add_argument("--block-days", type=int, default=7)
    parser.add_argument("--repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--slippage-scenarios", default="0,1,2,3")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
