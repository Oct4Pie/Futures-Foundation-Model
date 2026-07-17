#!/usr/bin/env python3
"""Extract row-bound frozen representations on the sealed Gate-3 screen."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from futures_foundation.finetune.downstream_contexts import load_sample_and_contexts
from futures_foundation.finetune.downstream_sample import load_row_selection
from scripts import benchmark_foundation_representations as source


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _screen_fingerprint(context_manifest: dict, selection_manifest: dict) -> str:
    value = {
        "schema_version": "ffm_downstream_representation_screen_v1",
        "contexts": context_manifest["content_fingerprint"],
        "selection": selection_manifest["content_fingerprint"],
    }
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _verified_existing(path: Path, *, arm: str, stage: str, screen: str) -> bool:
    manifest_path = Path(str(path) + ".manifest.json")
    if not path.is_file() or not manifest_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("arm") != arm or manifest.get("stage") != stage
        or manifest.get("window_fingerprint") != screen
        or manifest.get("oos_read") is not False
        or manifest.get("artifact", {}).get("sha256") != _sha256(path)
    ):
        return False
    with np.load(path, allow_pickle=False) as saved:
        if "row_index" not in saved.files or "embedding" not in saved.files:
            return False
        if saved["embedding"].shape[0] != len(saved["row_index"]):
            return False
        if not np.isfinite(saved["embedding"]).all():
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=tuple(source.EXTRACTORS))
    parser.add_argument("--stages", default="vanilla,stage1,stage2,stage3")
    parser.add_argument(
        "--sample", default="output/foundation_tournament/downstream_gate_v1/balanced_sample.npz",
    )
    parser.add_argument(
        "--contexts", default="output/foundation_tournament/downstream_gate_v1/contexts.npz",
    )
    parser.add_argument(
        "--row-selection",
        default="output/foundation_tournament/downstream_gate_v1/representation_rows.npz",
    )
    parser.add_argument(
        "--output-dir",
        default="output/foundation_tournament/downstream_gate_v1/screen/representations",
    )
    parser.add_argument(
        "--checkpoint-root", default="output/foundation_tournament/final_staged",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--amp", action="store_true",
        help="Use bf16 autocast where supported; float32 is the deterministic default.",
    )
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--kronos-repo", default="/tmp/ffm-kronos-inspect")
    parser.add_argument("--moment-repo", default="/tmp/ffm-moment-inspect")
    parser.add_argument("--ttm-repo", default="/tmp/ffm-granite-tsfm")
    parser.add_argument("--uni2ts-repo", default="/tmp/ffm-uni2ts")
    parser.add_argument("--kronos-batch", type=int, default=256)
    parser.add_argument("--chronos-batch", type=int, default=128)
    parser.add_argument("--moment-batch", type=int, default=16)
    parser.add_argument("--ttm-batch", type=int, default=256)
    parser.add_argument("--timesfm-batch", type=int, default=16)
    parser.add_argument("--moirai-batch", type=int, default=64)
    parser.add_argument("--mantis-batch", type=int, default=256)
    parser.add_argument("--toto-batch", type=int, default=256)
    parser.add_argument("--kronos-clip", type=float, default=3.0)
    args = parser.parse_args()

    sample, sample_manifest, contexts, context_manifest = load_sample_and_contexts(
        args.sample, args.contexts,
    )
    selection, selection_manifest = load_row_selection(
        args.row_selection, sample_manifest=sample_manifest,
    )
    row_index = np.asarray(selection["row_index"], np.int32)
    if args.max_rows is not None:
        if args.max_rows < 1 or args.max_rows > len(row_index):
            raise ValueError("max_rows must be within the sealed row selection")
        row_index = row_index[:args.max_rows]
    screen = _screen_fingerprint(context_manifest, selection_manifest)
    if args.max_rows is not None:
        screen = hashlib.sha256(
            (screen + f":prefix:{args.max_rows}").encode()
        ).hexdigest()
    stages = tuple(value.strip() for value in args.stages.split(",") if value.strip())
    if not stages or set(stages) - {"vanilla", "stage1", "stage2", "stage3"}:
        raise ValueError("stages must be drawn from vanilla,stage1,stage2,stage3")
    pending = []
    for stage in stages:
        path = Path(args.output_dir).resolve() / "embeddings" / args.arm / f"{stage}.npz"
        if _verified_existing(path, arm=args.arm, stage=stage, screen=screen):
            print(f"[{args.arm}] {stage}: verified existing {path}", flush=True)
        else:
            pending.append(stage)
    if not pending:
        return

    windows = {
        "context": contexts["context"][row_index],
        "context_time_ns": contexts["context_time_ns"][row_index],
        "timeframe": sample["timeframe"][row_index],
    }
    window_manifest = {
        "schema_version": "ffm_downstream_representation_screen_v1",
        "window_fingerprint": screen,
        "artifact": {"sha256": context_manifest["artifact"]["sha256"]},
        "oos_read": False,
    }
    args.stages = ",".join(pending)
    args.row_index = row_index
    args.row_selection_manifest = selection_manifest
    args.context_manifest = context_manifest
    source.EXTRACTORS[args.arm](args, args.arm, pending, windows, window_manifest)


if __name__ == "__main__":
    main()
