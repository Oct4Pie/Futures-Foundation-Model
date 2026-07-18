"""Installed CLI for the fail-closed native F/R parity matrix."""
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys

from .native_contracts import load_registry
from .native_parity_matrix import (
    MATRIX_CONFIG_SCHEMA,
    NativeParityMatrixError,
    _pairs,
    build_matrix_plan,
    execute_matrix,
    load_matrix_config,
    plan_record,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Execute every admitted native F/R parity bundle with Python-network denial."
    )
    value.add_argument("--config", help=f"JSON config with schema {MATRIX_CONFIG_SCHEMA}")
    value.add_argument("--runtime-profile", action="append", default=[], metavar="NAME=PYTHON")
    value.add_argument("--source-root", action="append", default=[], metavar="ARM=PATH")
    value.add_argument("--hf-cache-root", action="append", default=[], metavar="PATH")
    value.add_argument("--output", required=True, help="durable matrix bundle directory")
    value.add_argument("--aggregate", help="aggregate output; defaults below --output")
    value.add_argument("--runner", help="native parity worker path")
    value.add_argument("--runner-source", help="clean project source or installed RECORD root")
    value.add_argument("--env", action="append", default=[], metavar="NAME=VALUE")
    value.add_argument("--device")
    value.add_argument("--batch-size", type=int)
    value.add_argument("--samples", type=int)
    value.add_argument("--seed", type=int)
    value.add_argument("--generated-utc")
    value.add_argument("--dry-run", action="store_true")
    return value


def _installed_worker() -> Path:
    checkout_worker = PROJECT_ROOT / "scripts/native_parity_worker.py"
    if (PROJECT_ROOT / ".git").exists() and checkout_worker.is_file():
        return checkout_worker.resolve()
    spec = importlib.util.find_spec("scripts.native_parity_worker")
    if spec is None or spec.origin is None:
        raise NativeParityMatrixError(
            "packaged native_parity_worker is unavailable; supply --runner explicitly"
        )
    return Path(spec.origin).resolve()


def _installed_distribution_root() -> Path:
    if (PROJECT_ROOT / ".git").exists():
        return PROJECT_ROOT
    distribution = importlib.metadata.distribution("futures-foundation-model")
    return Path(distribution._path).resolve()


def main() -> int:
    args = parser().parse_args()
    try:
        config: dict = {}
        base = Path.cwd()
        if args.config:
            config, base = load_matrix_config(args.config)
        runtime = dict(config.get("runtime_profiles") or {})
        runtime.update(_pairs(args.runtime_profile, "runtime profile"))
        sources = dict(config.get("source_roots") or {})
        sources.update(_pairs(args.source_root, "source root"))
        cache_roots = list(args.hf_cache_root or config.get("hf_cache_roots") or [])
        runner_raw = args.runner or config.get("runner") or str(_installed_worker())
        runner = Path(os.path.expandvars(os.path.expanduser(str(runner_raw))))
        if not runner.is_absolute():
            runner = base / runner
        runner = runner.resolve()
        runner_source_raw = (
            args.runner_source or config.get("runner_source")
            or str(_installed_distribution_root())
        )
        runner_source = Path(os.path.expandvars(os.path.expanduser(str(runner_source_raw))))
        if not runner_source.is_absolute():
            runner_source = base / runner_source
        runner_source = runner_source.resolve()
        output = Path(args.output).expanduser().resolve()
        entries = build_matrix_plan(
            registry=load_registry(), runtime_pythons=runtime, source_roots=sources,
            hf_cache_roots=cache_roots, output_directory=output,
            runner=runner, runner_source=runner_source, path_base=base,
        )
        if args.dry_run:
            result = plan_record(entries)
        else:
            environment = dict(config.get("environment") or {})
            environment.update(_pairs(args.env, "environment"))
            aggregate = args.aggregate or str(output / "native_parity_aggregate.json")
            result = execute_matrix(
                entries, runner=runner, aggregate_output=aggregate,
                environment=environment,
                device=args.device or config.get("device", "cuda:0"),
                batch_size=(args.batch_size if args.batch_size is not None
                            else int(config.get("batch_size", 4))),
                samples=(args.samples if args.samples is not None
                         else int(config.get("samples", 20))),
                seed=(args.seed if args.seed is not None
                      else int(config.get("seed", 20260717))),
                generated_utc=args.generated_utc,
            )
    except (NativeParityMatrixError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0
