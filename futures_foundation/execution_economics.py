"""Hash-bound execution economics for research rulers.

This module is the sole runtime owner of futures tick sizes, tick values, and the
declared research fee schedule.  It deliberately does not pretend that a static
schedule is a historical broker statement.  Callers must bind it to an explicit
UTC evaluation interval before using an instrument specification.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import math
import os
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping

import yaml

from futures_foundation._authority_bundle_io import (
    AuthorityBundleIOError,
    read_regular_file,
)


EXECUTION_ECONOMICS_SCHEMA_VERSION = "ffm_execution_economics_v2"
CANONICAL_SCHEDULE_SHA256 = "0a644bb0de81b9a2119d6df94a7585bdb8e19e19651d78473462d556501d8ea4"
_CAPABILITY_TOKEN = object()
_MAX_ECONOMICS_BYTES = 4 * 1024 * 1024
_MAX_YAML_NODES = 100_000
_MAX_YAML_DEPTH = 24


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(loader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f"duplicate YAML key: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping,
)


def _absolute_lexical_path(path: str | Path) -> Path:
    """Make a path absolute without resolving away symlink evidence."""
    return Path(os.path.abspath(os.fspath(path)))


def _safe_relative_path(value: object, *, field: str) -> PurePosixPath:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a relative path string")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or relative.as_posix() != value
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError(f"{field} must be a canonical relative path")
    return relative


def _read_authority_bytes(path: str | Path, *, label: str) -> tuple[Path, bytes, str]:
    try:
        return read_regular_file(
            _absolute_lexical_path(path),
            label=label,
            max_bytes=_MAX_ECONOMICS_BYTES,
        )
    except AuthorityBundleIOError as exc:
        raise ValueError(str(exc)) from exc


def _sha256(path: Path) -> str:
    return _read_authority_bytes(path, label="execution-economics artifact")[2]


def _strict_yaml(path: str | Path, *, label: str) -> tuple[Path, dict, str]:
    source, raw, physical = _read_authority_bytes(path, label=label)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} must be UTF-8 YAML") from exc
    try:
        tokens = list(yaml.scan(text))
    except yaml.YAMLError as exc:
        raise ValueError(f"cannot scan {label}") from exc
    if len(tokens) > _MAX_YAML_NODES:
        raise ValueError(f"{label} exceeds YAML token limits")
    if any(isinstance(token, (yaml.tokens.AnchorToken, yaml.tokens.AliasToken)) for token in tokens):
        raise ValueError(f"{label} may not contain YAML anchors or aliases")
    try:
        document = yaml.load(text, Loader=_UniqueKeyLoader)
    except (yaml.YAMLError, RecursionError) as exc:
        raise ValueError(f"cannot parse {label}") from exc
    if not isinstance(document, dict):
        raise ValueError(f"expected a YAML mapping: {source}")

    nodes = 0

    def walk(value: Any, depth: int) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > _MAX_YAML_NODES or depth > _MAX_YAML_DEPTH:
            raise ValueError(f"{label} exceeds YAML structure limits")
        if isinstance(value, dict):
            for key, child in value.items():
                if not isinstance(key, str):
                    raise ValueError(f"{label} mapping keys must be strings")
                walk(child, depth + 1)
        elif isinstance(value, list):
            for child in value:
                walk(child, depth + 1)
        elif value is not None and not isinstance(value, (str, bool, int, float)):
            raise ValueError(f"{label} contains an unsupported scalar")

    walk(document, 0)
    return source, document, physical


def _utc(value: str, *, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must include a UTC offset")
    parsed = parsed.astimezone(timezone.utc)
    return parsed


def _exact_keys(value: dict, expected: set[str], *, field: str) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        raise ValueError(f"{field} keys mismatch; missing={missing}, unknown={unknown}")


@dataclass(frozen=True)
class InstrumentEconomics:
    root: str
    tick_size: float
    tick_value_usd: float
    fee_rt_usd: float

    def __post_init__(self) -> None:
        if self.root != self.root.strip().upper() or not self.root:
            raise ValueError("instrument root must be non-empty uppercase text")
        values = (self.tick_size, self.tick_value_usd, self.fee_rt_usd)
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"non-finite execution economics for {self.root}")
        if self.tick_size <= 0 or self.tick_value_usd <= 0 or self.fee_rt_usd < 0:
            raise ValueError(f"invalid execution economics for {self.root}")

    def as_dict(self) -> dict[str, float]:
        return {
            "tick_size": float(self.tick_size),
            "tick_value_usd": float(self.tick_value_usd),
            "fee_rt_usd": float(self.fee_rt_usd),
        }


@dataclass(frozen=True)
class ExecutionEconomics:
    schema_version: str
    schedule_path: str
    schedule_sha256: str
    source_path: str
    source_sha256: str
    schedule_basis: str
    effective_start_utc: datetime
    effective_end_exclusive_utc: datetime
    evaluation_start_utc: datetime
    evaluation_end_exclusive_utc: datetime
    primary_added_slippage_ticks_round_trip: float
    sensitivity_added_slippage_ticks_round_trip: tuple[float, ...]
    instruments: Mapping[str, InstrumentEconomics]
    canonical_admitted: bool
    _token: object

    def instrument(self, root: str) -> InstrumentEconomics:
        normalized = str(root).strip().upper()
        try:
            return self.instruments[normalized]
        except KeyError as exc:
            raise KeyError(f"execution economics missing instrument: {normalized}") from exc

    def assert_covers(self, start: str, end: str) -> None:
        requested_start = _utc(start, field="requested evaluation start")
        requested_end = _utc(end, field="requested evaluation end")
        if requested_end <= requested_start:
            raise ValueError("requested execution-economics interval is empty")
        if (
            requested_start < self.evaluation_start_utc
            or requested_end > self.evaluation_end_exclusive_utc
        ):
            raise ValueError("execution-economics capability does not cover the requested interval")

    def validate_added_slippage(self, ticks: float) -> float:
        value = float(ticks)
        admitted = (
            self.primary_added_slippage_ticks_round_trip,
            *self.sensitivity_added_slippage_ticks_round_trip,
        )
        if not math.isfinite(value) or value < 0 or not any(
            math.isclose(value, allowed, rel_tol=0.0, abs_tol=1e-12)
            for allowed in admitted
        ):
            raise ValueError(
                f"added slippage {value!r} is not declared by the execution-economics schedule"
            )
        return value

    def manifest(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "schedule_path": self.schedule_path,
            "schedule_sha256": self.schedule_sha256,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "schedule_basis": self.schedule_basis,
            "effective_start_utc": self.effective_start_utc.isoformat(),
            "effective_end_exclusive_utc": self.effective_end_exclusive_utc.isoformat(),
            "evaluation_start_utc": self.evaluation_start_utc.isoformat(),
            "evaluation_end_exclusive_utc": self.evaluation_end_exclusive_utc.isoformat(),
            "primary_added_slippage_ticks_round_trip": (
                self.primary_added_slippage_ticks_round_trip
            ),
            "sensitivity_added_slippage_ticks_round_trip": list(
                self.sensitivity_added_slippage_ticks_round_trip
            ),
            "instruments": {
                root: spec.as_dict() for root, spec in sorted(self.instruments.items())
            },
            "canonical_admitted": self.canonical_admitted,
        }


def require_execution_economics(value: object) -> ExecutionEconomics:
    """Authenticate and re-hash the canonical economics capability at its use boundary."""
    if (
        type(value) is not ExecutionEconomics
        or value._token is not _CAPABILITY_TOKEN
        or value.canonical_admitted is not True
        or value.schedule_sha256 != CANONICAL_SCHEDULE_SHA256
    ):
        raise TypeError("a canonical verified ExecutionEconomics capability is required")
    if _sha256(Path(value.schedule_path)) != value.schedule_sha256:
        raise ValueError("execution-economics schedule changed after verification")
    if _sha256(Path(value.source_path)) != value.source_sha256:
        raise ValueError("execution-economics source changed after verification")
    reopened = load_execution_economics(
        value.schedule_path,
        evaluation_start=value.evaluation_start_utc.isoformat(),
        evaluation_end=value.evaluation_end_exclusive_utc.isoformat(),
        required_roots=value.instruments,
    )
    if reopened != value:
        raise TypeError("execution-economics capability differs from canonical re-verification")
    return value


def load_execution_economics(
    path: str | Path,
    *,
    evaluation_start: str,
    evaluation_end: str,
    required_roots=(),
) -> ExecutionEconomics:
    """Load, source-verify, date-scope, and freeze an economics capability."""
    schedule_path, document, schedule_sha = _strict_yaml(
        path, label="execution-economics schedule",
    )
    _exact_keys(
        document,
        {
            "schema_version", "description", "source", "effective",
            "primary_added_slippage_ticks_round_trip",
            "sensitivity_added_slippage_ticks_round_trip", "instruments",
        },
        field="execution-economics document",
    )
    if document["schema_version"] != EXECUTION_ECONOMICS_SCHEMA_VERSION:
        raise ValueError("unsupported execution-economics schema")

    source = document["source"]
    if not isinstance(source, dict):
        raise ValueError("source must be a mapping")
    _exact_keys(
        source,
        {"project_path", "document_path", "schema_version", "sha256", "note"},
        field="source",
    )
    repository_root = schedule_path.parent.parent
    project_path = (
        PurePosixPath()
        if source["project_path"] == "."
        else _safe_relative_path(source["project_path"], field="source.project_path")
    )
    document_path = _safe_relative_path(source["document_path"], field="source.document_path")
    source_path, source_document, actual_source_sha = _strict_yaml(
        repository_root / Path(project_path) / Path(document_path),
        label="pinned execution-economics source",
    )
    if actual_source_sha != source["sha256"]:
        raise ValueError("pinned economics source hash mismatch")
    if source_document.get("schema_version") != source["schema_version"]:
        raise ValueError("pinned economics source schema mismatch")
    source_instruments = source_document.get("instruments")
    if not isinstance(source_instruments, dict):
        raise ValueError("pinned economics source has no instrument mapping")

    effective = document["effective"]
    if not isinstance(effective, dict):
        raise ValueError("effective must be a mapping")
    _exact_keys(
        effective, {"start_utc", "end_exclusive_utc", "basis"}, field="effective",
    )
    effective_start = _utc(effective["start_utc"], field="effective.start_utc")
    effective_end = _utc(
        effective["end_exclusive_utc"], field="effective.end_exclusive_utc",
    )
    evaluation_start_utc = _utc(evaluation_start, field="evaluation_start")
    evaluation_end_utc = _utc(evaluation_end, field="evaluation_end")
    if effective_end <= effective_start:
        raise ValueError("effective interval is empty")
    if evaluation_end_utc <= evaluation_start_utc:
        raise ValueError("evaluation interval is empty")
    if evaluation_start_utc < effective_start or evaluation_end_utc > effective_end:
        raise ValueError(
            "evaluation interval is outside the declared execution-economics interval"
        )

    primary = float(document["primary_added_slippage_ticks_round_trip"])
    if primary != 0.0:
        raise ValueError("primary execution ruler must use zero added slippage ticks")
    sensitivity = tuple(
        float(value) for value in document["sensitivity_added_slippage_ticks_round_trip"]
    )
    if (
        not sensitivity
        or any(not math.isfinite(value) or value < 0 for value in sensitivity)
        or tuple(sorted(set(sensitivity))) != sensitivity
        or 1.0 not in sensitivity
    ):
        raise ValueError(
            "slippage sensitivities must be sorted, unique, nonnegative, and include one tick"
        )

    raw_instruments = document["instruments"]
    if not isinstance(raw_instruments, dict) or not raw_instruments:
        raise ValueError("instruments must be a non-empty mapping")
    instruments: dict[str, InstrumentEconomics] = {}
    for root, raw in raw_instruments.items():
        root = str(root)
        if not isinstance(raw, dict):
            raise ValueError(f"instrument {root} must be a mapping")
        _exact_keys(raw, {"tick_size", "tick_value_usd", "fee_rt_usd"}, field=root)
        spec = InstrumentEconomics(
            root=root,
            tick_size=float(raw["tick_size"]),
            tick_value_usd=float(raw["tick_value_usd"]),
            fee_rt_usd=float(raw["fee_rt_usd"]),
        )
        upstream = source_instruments.get(root)
        if not isinstance(upstream, dict) or any(
            not math.isclose(
                float(upstream[field]), getattr(spec, field), rel_tol=0.0, abs_tol=1e-12,
            )
            for field in ("tick_size", "tick_value_usd", "fee_rt_usd")
        ):
            raise ValueError(f"execution economics mismatch pinned source for {root}")
        instruments[root] = spec

    missing = sorted(
        {str(value).strip().upper() for value in required_roots} - set(instruments)
    )
    if missing:
        raise ValueError(f"execution economics missing instruments: {missing}")
    return ExecutionEconomics(
        schema_version=document["schema_version"],
        schedule_path=str(schedule_path),
        schedule_sha256=schedule_sha,
        source_path=str(source_path),
        source_sha256=actual_source_sha,
        schedule_basis=str(effective["basis"]),
        effective_start_utc=effective_start,
        effective_end_exclusive_utc=effective_end,
        evaluation_start_utc=evaluation_start_utc,
        evaluation_end_exclusive_utc=evaluation_end_utc,
        primary_added_slippage_ticks_round_trip=primary,
        sensitivity_added_slippage_ticks_round_trip=sensitivity,
        instruments=MappingProxyType(instruments),
        canonical_admitted=(schedule_sha == CANONICAL_SCHEDULE_SHA256),
        _token=_CAPABILITY_TOKEN,
    )


__all__ = [
    "EXECUTION_ECONOMICS_SCHEMA_VERSION", "ExecutionEconomics", "InstrumentEconomics",
    "CANONICAL_SCHEDULE_SHA256", "load_execution_economics", "require_execution_economics",
]
