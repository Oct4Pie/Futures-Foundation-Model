#!/usr/bin/env python3
"""Build one machine-readable index of staged foundation-model artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


MODELS = (
    "kronos_mini", "kronos_small", "moment", "mantis_v1", "mantis_v2",
    "chronos_v1", "chronos_bolt", "chronos_v2", "ttm_r2", "moirai2_small",
    "timesfm25",
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path):
    return json.loads(Path(path).read_text())


def _atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _stage(path):
    import torch
    result = {"checkpoint": str(path), "checkpoint_sha256": _sha256(path)}
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    result["schema_version"] = bundle.get("schema_version")
    result["declared_stage"] = bundle.get("stage")
    result["parent"] = bundle.get("parent")
    for suffix, key in ((".report.json", "report"), (".train.pt", "resume_state")):
        sibling = Path(str(path) + suffix)
        if sibling.is_file():
            result[key] = {"path": str(sibling), "sha256": _sha256(sibling)}
    report_path = Path(str(path) + ".report.json")
    if report_path.is_file():
        report = _json(report_path)
        result["status"] = report.get("status")
        result["best_val_loss"] = report.get("best_val_loss")
        result["examples_seen"] = (report.get("data") or {}).get(
            "train_examples_seen",
            (report.get("data") or {}).get("examples_seen",
             (report.get("data") or {}).get("anchors_seen")),
        )
        result["oos_read"] = (report.get("split") or {}).get("oos_read")
    return result


def _scorecard(path):
    card = _json(path); arm = next(key for key in card["results"] if key != "persistence")
    result = card["results"][arm]
    keys = ("path_skill_vs_persistence", "fwd_dir_auc", "fwd_absmove_r2", "vol_r2",
            "trend_eff_r2", "range_expand_r2", "valid_candle_fraction",
            "nonnegative_volume_fraction")
    return {"path": str(path), "sha256": _sha256(path), "arm": arm,
            "oos_read": card.get("oos_read"),
            "overall": {key: result["overall"][key] for key in keys},
            "macro_stream": {key: result["macro_stream"][key] for key in keys}}


def build(root):
    root = Path(root).resolve(); models = {}; observed_oos = []
    for model in MODELS:
        directory = root / model; stages = {}
        for number in (1, 2, 3):
            path = directory / f"stage{number}.pt"
            if path.is_file():
                stages[f"stage{number}"] = _stage(path)
                observed_oos.append(stages[f"stage{number}"].get("oos_read"))
        value = {"directory": str(directory), "stages": stages,
                 "complete_chain": all(f"stage{number}" in stages for number in (1, 2, 3))}
        score = directory / "validation_scorecard.json"
        if score.is_file():
            value["shared_validation"] = _scorecard(score)
            observed_oos.append(value["shared_validation"].get("oos_read"))
        models[model] = value
    present_oos = [flag for flag in observed_oos if flag is not None]
    return {"schema_version": "ffm_stage_results_index_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(), "root": str(root),
            "models": models,
            "coverage": {"model_count": len(MODELS),
                         "complete_chain_count": sum(v["complete_chain"] for v in models.values()),
                         "any_recorded_oos_read_true": any(flag is True for flag in present_oos),
                         "all_present_oos_flags_false": all(flag is False for flag in present_oos),
                         "present_oos_flag_count": len(present_oos),
                         "missing_oos_flag_count": len(observed_oos) - len(present_oos)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="output/foundation_tournament/final_staged")
    parser.add_argument("--output", default="output/foundation_tournament/final_staged/STAGE_RESULTS_INDEX.json")
    args = parser.parse_args(); _atomic_json(args.output, build(args.root))


if __name__ == "__main__":
    main()
