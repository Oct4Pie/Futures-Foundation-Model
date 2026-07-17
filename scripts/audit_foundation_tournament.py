#!/usr/bin/env python3
"""Fail-closed audit of selected equal-history foundation-model training runs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from futures_foundation.finetune.tournament import (
    OOS_START, TRAIN_START, VALIDATION_START,
)


EXPECTED_EXAMPLES = 262_144
EXPECTED_STREAMS = {
    f"{ticker}@{timeframe}"
    for ticker in ("ES", "NQ", "RTY", "YM", "GC", "SI", "CL", "ZB", "ZN")
    for timeframe in ("1min", "3min", "5min", "15min", "30min", "60min")
}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _eligible_trials(study, trial_budget):
    complete = [trial for trial in study["trials"] if trial["state"] == "COMPLETE"]
    if trial_budget is not None:
        complete = [trial for trial in complete if int(trial["number"]) < trial_budget]
    return complete


def _select(study, trial_budget=None):
    complete = _eligible_trials(study, trial_budget)
    if not complete:
        raise ValueError(f"{study.get('model')}: no complete trials")
    if study["model"] in {"mantis_v1", "mantis_v2"}:
        return max(complete, key=lambda trial: (trial["values"][0], trial["values"][1]))
    return min(complete, key=lambda trial: trial["values"][0])


def _read_run(arm, checkpoint):
    report_path = Path(str(checkpoint) + ".report.json")
    if not report_path.is_file():
        raise FileNotFoundError(f"training report missing: {report_path}")
    report = json.loads(report_path.read_text())
    if arm.startswith("mantis_"):
        run_path = Path(str(checkpoint) + ".run.json")
        run = json.loads(run_path.read_text())
        config = report["config"]
        examples = int(config["batch"] * config["epochs"] * config["steps_per_epoch"])
        streams = {f"{ticker}@{timeframe}" for ticker in run["tickers"]
                   for timeframe in run["timeframes"]}
        split = {
            "train_start": run["train_start"], "validation_start": run["val_start"],
            "oos_start": run["holdout_start"],
            "oos_read": "excluded" not in run["holdout_policy"],
        }
        manifest = run.get("corpus_manifest_sha256")
        source_attestation = run.get("source_archive")
    else:
        data = report["data"]
        examples = data.get("train_examples_seen", data.get("train_anchors_seen",
                   data.get("anchors_seen", data.get("examples_seen"))))
        if examples is None and "predictor_examples_seen" in data:
            examples = int(data.get("tokenizer_examples_seen", 0)) + int(
                data["predictor_examples_seen"])
        examples = int(examples)
        streams = set(data["streams"])
        split = report["split"]
        manifest = data.get("data_manifest_sha256")
        source_attestation = report.get("local_sources")
    return report, {
        "examples": examples, "streams": streams, "split": split,
        "data_manifest_sha256": manifest, "source_attestation": source_attestation,
    }


def audit(studies, output, trial_budget=None):
    rows, errors, warnings = {}, [], []
    bound_manifests = set()
    for arm, study_path in studies.items():
        study = json.loads(Path(study_path).read_text())
        if study.get("model") not in {arm, "kronos"}:
            errors.append(f"{arm}: study model mismatch ({study.get('model')})")
        complete = len(_eligible_trials(study, trial_budget))
        required = trial_budget if trial_budget is not None else 8
        if complete < required:
            errors.append(f"{arm}: only {complete} eligible completed Optuna trials")
        selected = _select(study, trial_budget)
        checkpoint = Path(selected["user_attrs"]["checkpoint"])
        report, facts = _read_run(arm, checkpoint)
        expected_split = {
            "train_start": TRAIN_START, "validation_start": VALIDATION_START,
            "oos_start": OOS_START, "oos_read": False,
        }
        if facts["split"] != expected_split:
            errors.append(f"{arm}: invalid split/OOS attestation {facts['split']}")
        if facts["examples"] != EXPECTED_EXAMPLES:
            errors.append(f"{arm}: {facts['examples']} examples, expected {EXPECTED_EXAMPLES}")
        if facts["streams"] != EXPECTED_STREAMS:
            errors.append(f"{arm}: stream roster differs from the locked 54 streams")
        if not checkpoint.is_file():
            errors.append(f"{arm}: selected checkpoint missing")
        stamped = report.get("checkpoint", {}).get("sha256")
        actual = _sha256(checkpoint) if checkpoint.is_file() else None
        if stamped and stamped != actual:
            errors.append(f"{arm}: checkpoint hash mismatch")
        if facts["data_manifest_sha256"]:
            bound_manifests.add(facts["data_manifest_sha256"])
        else:
            warnings.append(f"{arm}: run did not cryptographically bind its data manifest")
        if not facts["source_attestation"]:
            errors.append(f"{arm}: exact source snapshot/attestation missing")
        rows[arm] = {
            "study": str(Path(study_path).resolve()), "selected_trial": selected["number"],
            "selection_values": selected["values"], "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": actual, "examples": facts["examples"],
            "streams": len(facts["streams"]), "split": facts["split"],
            "data_manifest_sha256": facts["data_manifest_sha256"],
        }
    if len(bound_manifests) > 1:
        errors.append(f"data manifest disagreement: {sorted(bound_manifests)}")
    result = {
        "schema_version": "ffm_foundation_tournament_audit_v1",
        "passed": not errors, "expected_examples_per_trial": EXPECTED_EXAMPLES,
        "optuna_trial_budget": trial_budget,
        "expected_streams": len(EXPECTED_STREAMS), "runs": rows,
        "errors": errors, "warnings": warnings,
    }
    _atomic_json(output, result)
    if errors:
        raise RuntimeError("foundation tournament audit failed: " + "; ".join(errors))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", action="append", required=True, help="arm=study.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--trial-budget", type=int)
    args = parser.parse_args()
    studies = {}
    for item in args.study:
        if "=" not in item:
            parser.error("study entries must use arm=study.json")
        arm, path = item.split("=", 1)
        if arm in studies:
            parser.error(f"duplicate arm: {arm}")
        studies[arm] = path
    if args.trial_budget is not None and args.trial_budget < 1:
        parser.error("--trial-budget must be positive")
    result = audit(studies, args.output, trial_budget=args.trial_budget)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
