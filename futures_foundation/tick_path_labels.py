"""Strict, integer-tick path labels for one unspliced futures contract-day.

The indexed backend is the production candidate; the independent reference backend is retained as
its correctness oracle. Observed trades and BBO-at-trade proxies are separate. Neither is passive
fill evidence, and fees/added slippage are deliberately outside this gross label artifact.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping as MappingABC
from dataclasses import asdict, dataclass
from datetime import date
from decimal import Decimal, ROUND_CEILING
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np


SCHEMA_VERSION = "ffm_ordered_tick_path_labels_v2"
DECISION_SCHEMA_VERSION = "ffm_tick_decision_manifest_v1"
BUNDLE_SCHEMA_VERSION = "ffm_tick_path_label_bundle_v1"

PATH_NEITHER = np.int8(0)
PATH_FAVORABLE_FIRST = np.int8(1)
PATH_ADVERSE_FIRST = np.int8(2)

INVALID_NONE = np.uint8(0)
INVALID_NO_STRICT_ENTRY = np.uint8(1)
INVALID_HORIZON_OUTSIDE_SESSION = np.uint8(2)
INVALID_NO_TERMINAL_TICK = np.uint8(3)
INVALID_STALE_ENDPOINT = np.uint8(4)
INVALID_RISK = np.uint8(5)
INVALID_DECISION_OUTSIDE_COVERAGE = np.uint8(6)
INVALID_STALE_ENTRY = np.uint8(7)

_SHA_RE = re.compile(r"[0-9a-f]{64}")
_SESSION_DAY_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
_MONTH_CODES = "FGHJKMNQUVXZ"
_INT64_MAX = np.iinfo(np.int64).max
_INT64_MIN = np.iinfo(np.int64).min
_VERIFIED_LABEL_TOKEN = object()


class VerifiedTickPathLabels(MappingABC[str, Any]):
    """Label mapping produced by the verified export path, not by caller metadata strings."""

    def __init__(self, values: Mapping[str, Any], *, _token: object) -> None:
        if _token is not _VERIFIED_LABEL_TOKEN:
            raise TypeError("VerifiedTickPathLabels can only be created by the verified builder")
        frozen: dict[str, Any] = {}
        for key, value in values.items():
            if isinstance(value, np.ndarray):
                value = value.copy()
                value.flags.writeable = False
            frozen[key] = value
        self._values = frozen
        self._verified_label_token = _token

    def __getitem__(self, key: str) -> Any:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def is_authentic(self) -> bool:
        return self._verified_label_token is _VERIFIED_LABEL_TOKEN


def _labels_match_verified_export(labels: Mapping[str, Any], verified: Any) -> bool:
    expected = {
        "root": verified.root,
        "contract_id": verified.contract_id,
        "session_day": verified.session_day,
        "split_use": verified.split_use,
        "export_receipt_sha256": verified.receipt_sha256,
        "source_shard_sha256": verified.output_shard_sha256,
        "source_file_table_sha256": verified.source_file_table_sha256,
        "corpus_contract_sha256": verified.contract_sha256,
        "environment_receipt_sha256": verified.environment_receipt_sha256,
        "instrument_spec_sha256": verified.instrument_spec_sha256,
        "tick_size": verified.tick_size,
        "tick_value": verified.tick_value,
    }
    return all(labels.get(key) == value for key, value in expected.items())


@dataclass(frozen=True)
class TickPathLabelConfig:
    horizons_seconds: tuple[int, ...] = (3600, 10800, 21600)
    targets_r: tuple[float, ...] = (1.0, 2.0, 3.0)
    entry_tolerance_seconds: int = 60
    endpoint_tolerance_seconds: int = 60
    price_alignment_atol_ticks: float = 1e-6

    def validate(self) -> None:
        if (
            not self.horizons_seconds
            or any(type(value) is not int for value in self.horizons_seconds)
            or any(value <= 0 or value > 7 * 24 * 3600 for value in self.horizons_seconds)
        ):
            raise ValueError("horizons_seconds must contain bounded positive integers")
        if tuple(sorted(set(self.horizons_seconds))) != tuple(self.horizons_seconds):
            raise ValueError("horizons_seconds must be unique and increasing")
        if (
            not self.targets_r
            or any(type(value) not in (int, float) for value in self.targets_r)
            or any(not math.isfinite(float(value)) or float(value) <= 0 for value in self.targets_r)
        ):
            raise ValueError("targets_r must contain finite positive numbers")
        if tuple(sorted(set(map(float, self.targets_r)))) != tuple(map(float, self.targets_r)):
            raise ValueError("targets_r must be unique and increasing")
        if (
            type(self.entry_tolerance_seconds) is not int
            or type(self.endpoint_tolerance_seconds) is not int
            or not 0 <= self.entry_tolerance_seconds <= 3600
            or not 0 <= self.endpoint_tolerance_seconds <= 3600
        ):
            raise ValueError("entry and endpoint tolerances must be bounded nonnegative integers")
        if (
            type(self.price_alignment_atol_ticks) not in (int, float)
            or not math.isfinite(float(self.price_alignment_atol_ticks))
            or not 0 <= float(self.price_alignment_atol_ticks) <= 0.01
        ):
            raise ValueError("price_alignment_atol_ticks must be finite and between 0 and 0.01")


def _sha(value: Any, name: str) -> str:
    text = str(value)
    if not _SHA_RE.fullmatch(text):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return text


def _scalar(value: Any, name: str) -> Any:
    array = np.asarray(value)
    if array.ndim != 0:
        raise ValueError(f"{name} must be shard-level scalar metadata")
    return array.item()


def _string_scalar(value: Any, name: str) -> str:
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind not in ("U", "S", "O"):
        raise ValueError(f"{name} must be shard-level string metadata")
    item = array.item()
    if not isinstance(item, (str, bytes)):
        raise ValueError(f"{name} must be shard-level string metadata")
    text = item.decode("utf-8") if isinstance(item, bytes) else item
    if (
        not text
        or text != text.strip()
        or text.lower() in {"none", "null", "nan"}
        or any(character.isspace() for character in text)
    ):
        raise ValueError(f"{name} must be a nonempty string")
    return text


def _integer_scalar(value: Any, name: str) -> int:
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind not in ("i", "u"):
        raise ValueError(f"{name} must be shard-level integer metadata")
    result = int(array.item())
    if result < _INT64_MIN or result > _INT64_MAX:
        raise ValueError(f"{name} exceeds signed int64")
    return result


def _finite_positive_scalar(value: Any, name: str) -> float:
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind not in ("i", "u", "f"):
        raise ValueError(f"{name} must be shard-level numeric metadata")
    result = float(array.item())
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _integer_array(value: Any, name: str, *, unsigned: bool = False) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or array.dtype.kind not in ("i", "u"):
        raise ValueError(f"{name} must be a one-dimensional integer array")
    if unsigned and array.dtype.kind == "i" and np.any(array < 0):
        raise ValueError(f"{name} must be nonnegative")
    if not unsigned and array.dtype.kind == "u" and np.any(array > _INT64_MAX):
        raise ValueError(f"{name} exceeds signed int64")
    return array.astype(np.uint64 if unsigned else np.int64, copy=True)


def _price_ticks(
    values: Any,
    name: str,
    *,
    tick_size: float,
    tolerance_ticks: float,
    valid: np.ndarray | None = None,
) -> np.ndarray:
    raw = np.asarray(values, dtype=np.float64)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    mask = np.ones(len(raw), dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    if mask.shape != raw.shape:
        raise ValueError(f"{name} validity mask has the wrong shape")
    if not np.isfinite(raw[mask]).all():
        raise ValueError(f"valid {name} values must be finite")
    scaled = raw[mask] / float(tick_size)
    rounded = np.rint(scaled)
    if np.any(np.abs(scaled - rounded) > float(tolerance_ticks)):
        raise ValueError(f"{name} is not aligned to tick_size")
    if np.any(rounded < -(2 ** 63)) or np.any(rounded >= 2 ** 63):
        raise ValueError(f"{name} tick indices exceed int64")
    result = np.zeros(len(raw), dtype=np.int64)
    result[mask] = rounded.astype(np.int64)
    return result


class _ExtremaIndex:
    """Deterministic range extrema and leftmost threshold queries."""

    def __init__(self, values: np.ndarray, valid: np.ndarray | None = None) -> None:
        values = np.asarray(values, dtype=np.int64)
        valid = np.ones(len(values), dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
        self.length = len(values)
        self.size = 1 << max(0, (self.length - 1).bit_length())
        self.minimum = np.full(2 * self.size, _INT64_MAX, dtype=np.int64)
        self.maximum = np.full(2 * self.size, _INT64_MIN, dtype=np.int64)
        positions = np.arange(self.length) + self.size
        self.minimum[positions[valid]] = values[valid]
        self.maximum[positions[valid]] = values[valid]
        for node in range(self.size - 1, 0, -1):
            self.minimum[node] = min(self.minimum[2 * node], self.minimum[2 * node + 1])
            self.maximum[node] = max(self.maximum[2 * node], self.maximum[2 * node + 1])
        self.minimum.flags.writeable = False
        self.maximum.flags.writeable = False

    def range_minmax(self, left: int, right: int) -> tuple[int, int]:
        """Return min/max over inclusive bounds."""
        lo, hi = left + self.size, right + self.size + 1
        minimum, maximum = _INT64_MAX, _INT64_MIN
        while lo < hi:
            if lo & 1:
                minimum, maximum = min(minimum, int(self.minimum[lo])), max(maximum, int(self.maximum[lo]))
                lo += 1
            if hi & 1:
                hi -= 1
                minimum, maximum = min(minimum, int(self.minimum[hi])), max(maximum, int(self.maximum[hi]))
            lo //= 2
            hi //= 2
        return minimum, maximum

    def first_ge(self, left: int, right: int, threshold: int) -> int:
        return self._first(left, right, int(threshold), greater=True, node=1, lo=0, hi=self.size)

    def first_le(self, left: int, right: int, threshold: int) -> int:
        return self._first(left, right, int(threshold), greater=False, node=1, lo=0, hi=self.size)

    def _first(
        self, left: int, right: int, threshold: int, *, greater: bool, node: int, lo: int, hi: int
    ) -> int:
        if hi <= left or right < lo:
            return -1
        bound = int(self.maximum[node] if greater else self.minimum[node])
        if (greater and bound < threshold) or (not greater and bound > threshold):
            return -1
        if hi - lo == 1:
            return lo if lo < self.length else -1
        mid = (lo + hi) // 2
        found = self._first(left, right, threshold, greater=greater, node=2 * node, lo=lo, hi=mid)
        return found if found >= 0 else self._first(
            left, right, threshold, greater=greater, node=2 * node + 1, lo=mid, hi=hi
        )


class OrderedTickPathIndex:
    """Validated integer-tick index reusable across timeframe-specific decision sets.

    Direct construction is retained as a low-level/synthetic-test surface. Production
    Corpus-v3 materialization must enter through ``VerifiedContractDayExport.build_path_index``
    so receipt hashes cannot be invented by a caller-provided mapping.
    """

    def __init__(
        self, ticks: Mapping[str, Any], *, tick_size: float, config: TickPathLabelConfig
    ) -> None:
        config.validate()
        if not math.isfinite(float(tick_size)) or float(tick_size) <= 0:
            raise ValueError("tick_size must be finite and positive")
        required_rows = {
            "timestamp_utc_ns", "event_seq", "price", "bid", "ask", "quote_valid",
            "source_file_index", "source_row_ordinal",
        }
        required_meta = {
            "root", "contract_id", "session_day", "split_use", "session_start_utc_ns",
            "session_end_utc_ns", "coverage_start_utc_ns", "coverage_end_utc_ns",
            "export_receipt_sha256", "source_shard_sha256", "source_file_table_sha256",
            "corpus_contract_sha256", "environment_receipt_sha256",
            "instrument_spec_sha256", "tick_size", "tick_value",
        }
        missing = sorted((required_rows | required_meta) - set(ticks))
        if missing:
            raise ValueError(f"tick contract-day is missing fields: {missing}")
        self.timestamp_utc_ns = _integer_array(ticks["timestamp_utc_ns"], "timestamp_utc_ns")
        self.event_seq = _integer_array(ticks["event_seq"], "event_seq", unsigned=True)
        self.source_file_index = _integer_array(ticks["source_file_index"], "source_file_index")
        self.source_row_ordinal = _integer_array(ticks["source_row_ordinal"], "source_row_ordinal")
        quote_valid_raw = np.asarray(ticks["quote_valid"])
        if quote_valid_raw.ndim != 1 or quote_valid_raw.dtype.kind != "b":
            raise ValueError("quote_valid must be a one-dimensional boolean array")
        self.quote_valid = quote_valid_raw.astype(bool, copy=True)
        lengths = {
            len(value) for value in (
                self.timestamp_utc_ns, self.event_seq, self.source_file_index,
                self.source_row_ordinal, self.quote_valid,
            )
        }
        if lengths != {len(self.timestamp_utc_ns)} or not len(self.timestamp_utc_ns):
            raise ValueError("tick row arrays must have one equal nonzero length")
        self.tick_size = float(tick_size)
        self.price_alignment_atol_ticks = float(config.price_alignment_atol_ticks)
        shard_tick_size = _finite_positive_scalar(ticks["tick_size"], "tick_size")
        self.tick_value = _finite_positive_scalar(ticks["tick_value"], "tick_value")
        if shard_tick_size != self.tick_size:
            raise ValueError("tick_size differs from the hash-bound shard instrument specification")
        tolerance = float(config.price_alignment_atol_ticks)
        self.price_ticks = _price_ticks(
            ticks["price"], "price", tick_size=self.tick_size, tolerance_ticks=tolerance
        )
        self.bid_ticks = _price_ticks(
            ticks["bid"], "bid", tick_size=self.tick_size, tolerance_ticks=tolerance,
            valid=self.quote_valid,
        )
        self.ask_ticks = _price_ticks(
            ticks["ask"], "ask", tick_size=self.tick_size, tolerance_ticks=tolerance,
            valid=self.quote_valid,
        )
        if not (
            len(self.price_ticks) == len(self.timestamp_utc_ns)
            and len(self.bid_ticks) == len(self.timestamp_utc_ns)
            and len(self.ask_ticks) == len(self.timestamp_utc_ns)
        ):
            raise ValueError("price and quote row arrays must match timestamp row count")
        if np.any(self.bid_ticks[self.quote_valid] > self.ask_ticks[self.quote_valid]):
            raise ValueError("valid BBO-at-trade quotes must not cross")
        if np.any(self.source_file_index < 0) or np.any(self.source_row_ordinal < 0):
            raise ValueError("source lineage indices must be nonnegative")
        lineage = np.rec.fromarrays(
            (self.source_file_index, self.source_row_ordinal), names=("file", "row")
        )
        if len(np.unique(lineage)) != len(lineage):
            raise ValueError("source file/row lineage keys must be unique")
        if np.any(self.timestamp_utc_ns[1:] < self.timestamp_utc_ns[:-1]):
            raise ValueError("timestamp_utc_ns must be nondecreasing")
        same = self.timestamp_utc_ns[1:] == self.timestamp_utc_ns[:-1]
        if np.any(self.event_seq[1:][same] <= self.event_seq[:-1][same]):
            raise ValueError("(timestamp_utc_ns, event_seq) must be strictly increasing and unique")

        self.root = _string_scalar(ticks["root"], "root")
        self.contract_id = _string_scalar(ticks["contract_id"], "contract_id")
        self.session_day = _string_scalar(ticks["session_day"], "session_day")
        self.split_use = _string_scalar(ticks["split_use"], "split_use")
        if self.split_use not in {"foundation_pretraining", "supervised_training", "development"}:
            raise ValueError("split_use is not an admitted materialization split")
        if not _SESSION_DAY_RE.fullmatch(self.session_day):
            raise ValueError("session_day must be an ISO YYYY-MM-DD date")
        try:
            date.fromisoformat(self.session_day)
        except ValueError as exc:
            raise ValueError("session_day must be a valid ISO date") from exc
        contract_pattern = rf"{re.escape(self.root)}[{_MONTH_CODES}]\d{{2}}"
        if not re.fullmatch(contract_pattern, self.contract_id):
            raise ValueError("contract_id must be one unspliced dated contract matching root")
        self.session_start_utc_ns = _integer_scalar(
            ticks["session_start_utc_ns"], "session_start_utc_ns"
        )
        self.session_end_utc_ns = _integer_scalar(
            ticks["session_end_utc_ns"], "session_end_utc_ns"
        )
        self.coverage_start_utc_ns = _integer_scalar(
            ticks["coverage_start_utc_ns"], "coverage_start_utc_ns"
        )
        self.coverage_end_utc_ns = _integer_scalar(
            ticks["coverage_end_utc_ns"], "coverage_end_utc_ns"
        )
        if not (
            self.session_start_utc_ns <= self.coverage_start_utc_ns <= int(self.timestamp_utc_ns[0])
            <= int(self.timestamp_utc_ns[-1]) <= self.coverage_end_utc_ns <= self.session_end_utc_ns
        ):
            raise ValueError("session and verified coverage bounds are inconsistent")
        self.export_receipt_sha256 = _sha(ticks["export_receipt_sha256"], "export_receipt_sha256")
        self.source_shard_sha256 = _sha(ticks["source_shard_sha256"], "source_shard_sha256")
        self.source_file_table_sha256 = _sha(
            ticks["source_file_table_sha256"], "source_file_table_sha256"
        )
        self.corpus_contract_sha256 = _sha(
            ticks["corpus_contract_sha256"], "corpus_contract_sha256"
        )
        self.environment_receipt_sha256 = _sha(
            ticks["environment_receipt_sha256"], "environment_receipt_sha256"
        )
        self.instrument_spec_sha256 = _sha(
            ticks["instrument_spec_sha256"], "instrument_spec_sha256"
        )
        self.trade_index = _ExtremaIndex(self.price_ticks)
        self.bid_index = _ExtremaIndex(self.bid_ticks, self.quote_valid)
        self.ask_index = _ExtremaIndex(self.ask_ticks, self.quote_valid)
        self.invalid_quote_prefix = np.r_[0, np.cumsum(~self.quote_valid, dtype=np.int64)]
        for array in (
            self.timestamp_utc_ns,
            self.event_seq,
            self.source_file_index,
            self.source_row_ordinal,
            self.quote_valid,
            self.price_ticks,
            self.bid_ticks,
            self.ask_ticks,
            self.invalid_quote_prefix,
        ):
            array.flags.writeable = False

    def entry_index(self, decision_ts: int, decision_seq: int) -> int:
        left = int(np.searchsorted(self.timestamp_utc_ns, decision_ts, side="left"))
        right = int(np.searchsorted(self.timestamp_utc_ns, decision_ts, side="right"))
        if left == right:
            return left
        return left + int(np.searchsorted(
            self.event_seq[left:right], np.uint64(decision_seq), side="right"
        ))

    def quotes_complete(self, left: int, right: int) -> bool:
        return bool(self.invalid_quote_prefix[right + 1] == self.invalid_quote_prefix[left])


def decision_manifest_sha256(
    index: OrderedTickPathIndex,
    decision_time_utc_ns: Any,
    decision_event_seq: Any,
    risk_ticks: Any,
    risk_known_time_utc_ns: Any,
    risk_known_event_seq: Any,
) -> str:
    arrays = {
        "decision_time_utc_ns": _integer_array(decision_time_utc_ns, "decision_time_utc_ns"),
        "decision_event_seq": _integer_array(decision_event_seq, "decision_event_seq", unsigned=True),
        "risk_ticks": _integer_array(risk_ticks, "risk_ticks"),
        "risk_known_time_utc_ns": _integer_array(risk_known_time_utc_ns, "risk_known_time_utc_ns"),
        "risk_known_event_seq": _integer_array(
            risk_known_event_seq, "risk_known_event_seq", unsigned=True
        ),
    }
    payload: dict[str, Any] = {
        "schema_version": DECISION_SCHEMA_VERSION,
        "root": index.root,
        "contract_id": index.contract_id,
        "session_day": index.session_day,
        "export_receipt_sha256": index.export_receipt_sha256,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    )
    for name, array in sorted(arrays.items()):
        canonical = np.ascontiguousarray(array.astype(array.dtype.newbyteorder("<"), copy=False))
        digest.update(name.encode() + b"\0" + canonical.dtype.str.encode() + b"\0")
        digest.update(json.dumps(canonical.shape).encode() + b"\0" + canonical.tobytes())
    return digest.hexdigest()


def _first_reference(
    values: np.ndarray,
    left: int,
    right: int,
    threshold: int,
    ge: bool,
    valid: np.ndarray | None,
) -> int:
    mask = values[left:right + 1] >= threshold if ge else values[left:right + 1] <= threshold
    if valid is not None:
        mask &= valid[left:right + 1]
    found = np.flatnonzero(mask)
    return int(left + found[0]) if len(found) else -1


def _first(
    tree: _ExtremaIndex,
    values: np.ndarray,
    left: int,
    right: int,
    threshold: int,
    *,
    ge: bool,
    backend: str,
    valid: np.ndarray | None = None,
) -> int:
    if left > right:
        return -1
    if backend == "reference":
        return _first_reference(values, left, right, threshold, ge, valid)
    return tree.first_ge(left, right, threshold) if ge else tree.first_le(left, right, threshold)


def _state(favorable: int, adverse: int, terminal: int) -> np.int8:
    favorable = favorable if 0 <= favorable <= terminal else -1
    adverse = adverse if 0 <= adverse <= terminal else -1
    if favorable < 0 and adverse < 0:
        return PATH_NEITHER
    if favorable >= 0 and (adverse < 0 or favorable < adverse):
        return PATH_FAVORABLE_FIRST
    return PATH_ADVERSE_FIRST


def _event_lineage(
    index: OrderedTickPathIndex, positions: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resolve source lineage for touch indices while preserving invalid sentinels."""
    positions = np.asarray(positions, dtype=np.int64)
    times = np.full(positions.shape, -1, dtype=np.int64)
    sequences = np.zeros(positions.shape, dtype=np.uint64)
    files = np.full(positions.shape, -1, dtype=np.int64)
    rows = np.full(positions.shape, -1, dtype=np.int64)
    good = positions >= 0
    selected = positions[good]
    times[good] = index.timestamp_utc_ns[selected]
    sequences[good] = index.event_seq[selected]
    files[good] = index.source_file_index[selected]
    rows[good] = index.source_row_ordinal[selected]
    return times, sequences, files, rows


def touches_by_horizon(
    positions: np.ndarray, terminals: np.ndarray
) -> np.ndarray:
    """Expand max-horizon touch indices and mask touches after each horizon terminal."""
    positions = np.asarray(positions, dtype=np.int64)
    terminals = np.asarray(terminals, dtype=np.int64)
    expanded = np.broadcast_to(
        positions[:, None, ...], (len(positions), terminals.shape[1], *positions.shape[1:])
    ).copy()
    terminal_shape = terminals.shape + (1,) * (positions.ndim - 1)
    horizon_terminals = terminals.reshape(terminal_shape)
    expanded[(expanded < 0) | (horizon_terminals < 0) | (expanded > horizon_terminals)] = -1
    return expanded


def _build_tick_path_labels_impl(
    ticks: Mapping[str, Any] | OrderedTickPathIndex | Any,
    *,
    decision_time_utc_ns: Any,
    decision_event_seq: Any,
    risk_ticks: Any,
    risk_known_time_utc_ns: Any,
    risk_known_event_seq: Any,
    decision_manifest_sha256_value: str,
    tick_size: float | None = None,
    config: TickPathLabelConfig | None = None,
    backend: str = "indexed",
    _allow_unverified: bool = False,
) -> dict[str, Any]:
    """Build normalized path labels with declared endpoints as the sole purge authority."""
    config = config or TickPathLabelConfig()
    config.validate()
    if backend not in {"indexed", "reference"}:
        raise ValueError("backend must be 'indexed' or 'reference'")
    source_verification = "verified_contract_day_export"
    from .corpus_v3_export import VerifiedContractDayExport
    if type(ticks) is VerifiedContractDayExport and ticks.is_authentic():
        index = ticks.build_path_index(config=config)
        if tick_size is not None and float(tick_size) != index.tick_size:
            raise ValueError("tick_size differs from the verified export")
    elif isinstance(ticks, OrderedTickPathIndex):
        if not _allow_unverified:
            raise ValueError("production labels require a VerifiedContractDayExport capability")
        source_verification = "synthetic_test_only"
        index = ticks
        if tick_size is not None and float(tick_size) != index.tick_size:
            raise ValueError("tick_size differs from the reusable index")
        if float(config.price_alignment_atol_ticks) != index.price_alignment_atol_ticks:
            raise ValueError("price-alignment policy differs from the reusable index")
    else:
        if not _allow_unverified:
            raise ValueError("production labels require a VerifiedContractDayExport capability")
        source_verification = "synthetic_test_only"
        if tick_size is None:
            raise ValueError("tick_size is required when constructing an index")
        index = OrderedTickPathIndex(ticks, tick_size=float(tick_size), config=config)

    decision_ts = _integer_array(decision_time_utc_ns, "decision_time_utc_ns")
    decision_seq = _integer_array(decision_event_seq, "decision_event_seq", unsigned=True)
    risks = _integer_array(risk_ticks, "risk_ticks")
    known_ts = _integer_array(risk_known_time_utc_ns, "risk_known_time_utc_ns")
    known_seq = _integer_array(risk_known_event_seq, "risk_known_event_seq", unsigned=True)
    if len({len(value) for value in (decision_ts, decision_seq, risks, known_ts, known_seq)}) != 1:
        raise ValueError("decision/risk arrays must have equal lengths")
    if len(decision_ts) and (
        np.any(decision_ts[1:] < decision_ts[:-1])
        or np.any(
            (decision_ts[1:] == decision_ts[:-1])
            & (decision_seq[1:] <= decision_seq[:-1])
        )
    ):
        raise ValueError("decision keys must be strictly increasing")
    future_known = (known_ts > decision_ts) | ((known_ts == decision_ts) & (known_seq > decision_seq))
    if np.any(future_known):
        raise ValueError("risk provenance must be known no later than the decision key")
    expected_manifest = decision_manifest_sha256(
        index, decision_ts, decision_seq, risks, known_ts, known_seq
    )
    if _sha(decision_manifest_sha256_value, "decision_manifest_sha256") != expected_manifest:
        raise ValueError("decision manifest hash does not bind the supplied decisions and risks")

    n, h_count, t_count = len(decision_ts), len(config.horizons_seconds), len(config.targets_r)
    valid = np.zeros((n, h_count), dtype=bool)
    quote_valid = np.zeros((n, h_count), dtype=bool)
    invalid_reason = np.full((n, h_count), INVALID_NO_STRICT_ENTRY, dtype=np.uint8)
    entry_index = np.full(n, -1, dtype=np.int64)
    entry_time_ns = np.full(n, -1, dtype=np.int64)
    entry_event_seq = np.zeros(n, dtype=np.uint64)
    entry_wait_ns = np.full(n, -1, dtype=np.int64)
    terminal_index = np.full((n, h_count), -1, dtype=np.int64)
    terminal_time_ns = np.full((n, h_count), -1, dtype=np.int64)
    declared_end_ns = np.full((n, h_count), -1, dtype=np.int64)
    observed_mfe_r = np.full((n, h_count, 2), np.nan, dtype=np.float64)
    observed_mae_r = np.full((n, h_count, 2), np.nan, dtype=np.float64)
    observed_terminal_r = np.full((n, h_count, 2), np.nan, dtype=np.float64)
    state_shape = (n, h_count, 2, t_count)
    observed_state = np.full(state_shape, -1, dtype=np.int8)
    market_state = np.full(state_shape, -1, dtype=np.int8)
    market_gross_r = np.full(state_shape, np.nan, dtype=np.float64)
    observed_adverse_index = np.full((n, 2), -1, dtype=np.int64)
    observed_favorable_index = np.full((n, 2, t_count), -1, dtype=np.int64)
    market_adverse_index = np.full((n, 2), -1, dtype=np.int64)
    market_favorable_index = np.full((n, 2, t_count), -1, dtype=np.int64)
    target_distance_ticks = np.full((n, t_count), -1, dtype=np.int64)

    entry_tolerance_ns = config.entry_tolerance_seconds * 1_000_000_000
    endpoint_tolerance_ns = config.endpoint_tolerance_seconds * 1_000_000_000
    ts = index.timestamp_utc_ns
    for row in range(n):
        decision = int(decision_ts[row])
        if not index.session_start_utc_ns <= decision < index.session_end_utc_ns:
            invalid_reason[row, :] = INVALID_HORIZON_OUTSIDE_SESSION
            continue
        if not index.coverage_start_utc_ns <= decision <= index.coverage_end_utc_ns:
            invalid_reason[row, :] = INVALID_DECISION_OUTSIDE_COVERAGE
            continue
        risk = int(risks[row])
        if risk <= 0:
            invalid_reason[row, :] = INVALID_RISK
            continue
        target_distances = np.asarray([
            int(
                (Decimal(str(target)) * Decimal(risk)).to_integral_value(
                    rounding=ROUND_CEILING
                )
            )
            for target in config.targets_r
        ], dtype=object)
        if any(distance <= 0 or distance > _INT64_MAX for distance in target_distances):
            raise ValueError("target distance in ticks exceeds signed int64")
        target_distances = target_distances.astype(np.int64)
        target_distance_ticks[row] = target_distances
        entry = index.entry_index(decision, int(decision_seq[row]))
        if entry >= len(ts):
            invalid_reason[row, :] = INVALID_NO_STRICT_ENTRY
            continue
        wait = int(ts[entry]) - decision
        entry_index[row], entry_time_ns[row], entry_event_seq[row], entry_wait_ns[row] = (
            entry, int(ts[entry]), index.event_seq[entry], wait
        )
        if wait < 0 or wait > entry_tolerance_ns:
            invalid_reason[row, :] = INVALID_STALE_ENTRY
            continue
        for horizon_i, horizon_seconds in enumerate(config.horizons_seconds):
            horizon_ns = horizon_seconds * 1_000_000_000
            if decision > _INT64_MAX - horizon_ns:
                raise ValueError("decision timestamp plus horizon exceeds int64")
            endpoint = decision + horizon_ns
            declared_end_ns[row, horizon_i] = endpoint
            if endpoint >= index.session_end_utc_ns:
                invalid_reason[row, horizon_i] = INVALID_HORIZON_OUTSIDE_SESSION
                continue
            if endpoint > index.coverage_end_utc_ns:
                invalid_reason[row, horizon_i] = INVALID_NO_TERMINAL_TICK
                continue
            terminal = int(np.searchsorted(ts, endpoint, side="right") - 1)
            if terminal <= entry:
                invalid_reason[row, horizon_i] = INVALID_NO_TERMINAL_TICK
                continue
            if endpoint - int(ts[terminal]) > endpoint_tolerance_ns:
                invalid_reason[row, horizon_i] = INVALID_STALE_ENDPOINT
                continue
            valid[row, horizon_i] = True
            invalid_reason[row, horizon_i] = INVALID_NONE
            terminal_index[row, horizon_i] = terminal
            terminal_time_ns[row, horizon_i] = int(ts[terminal])
            quote_valid[row, horizon_i] = index.quotes_complete(entry, terminal)

        valid_terminals = terminal_index[row][valid[row]]
        if not len(valid_terminals):
            continue
        max_terminal = int(np.max(valid_terminals))
        trade_entry = int(index.price_ticks[entry])
        for direction_i, direction in enumerate((1, -1)):
            observed_adverse_index[row, direction_i] = _first(
                index.trade_index, index.price_ticks, entry + 1, max_terminal,
                trade_entry - risk if direction > 0 else trade_entry + risk,
                ge=direction < 0, backend=backend,
            )
            for target_i, distance in enumerate(target_distances):
                observed_favorable_index[row, direction_i, target_i] = _first(
                    index.trade_index, index.price_ticks, entry + 1, max_terminal,
                    trade_entry + int(distance) if direction > 0 else trade_entry - int(distance),
                    ge=direction > 0, backend=backend,
                )
            if index.quote_valid[entry]:
                quote_values = index.bid_ticks if direction > 0 else index.ask_ticks
                quote_tree = index.bid_index if direction > 0 else index.ask_index
                quote_entry = int(index.ask_ticks[entry] if direction > 0 else index.bid_ticks[entry])
                market_adverse_index[row, direction_i] = _first(
                    quote_tree, quote_values, entry + 1, max_terminal,
                    quote_entry - risk if direction > 0 else quote_entry + risk,
                    ge=direction < 0, backend=backend, valid=index.quote_valid,
                )
                for target_i, distance in enumerate(target_distances):
                    market_favorable_index[row, direction_i, target_i] = _first(
                        quote_tree, quote_values, entry + 1, max_terminal,
                        quote_entry + int(distance) if direction > 0 else quote_entry - int(distance),
                        ge=direction > 0, backend=backend, valid=index.quote_valid,
                    )

            for horizon_i in np.flatnonzero(valid[row]):
                terminal = int(terminal_index[row, horizon_i])
                minimum, maximum = (
                    index.trade_index.range_minmax(entry, terminal)
                    if backend == "indexed"
                    else (
                        int(np.min(index.price_ticks[entry:terminal + 1])),
                        int(np.max(index.price_ticks[entry:terminal + 1])),
                    )
                )
                if direction > 0:
                    observed_mfe_r[row, horizon_i, direction_i] = (maximum - trade_entry) / risk
                    observed_mae_r[row, horizon_i, direction_i] = (minimum - trade_entry) / risk
                    observed_terminal_r[row, horizon_i, direction_i] = (
                        int(index.price_ticks[terminal]) - trade_entry
                    ) / risk
                else:
                    observed_mfe_r[row, horizon_i, direction_i] = (trade_entry - minimum) / risk
                    observed_mae_r[row, horizon_i, direction_i] = (trade_entry - maximum) / risk
                    observed_terminal_r[row, horizon_i, direction_i] = (
                        trade_entry - int(index.price_ticks[terminal])
                    ) / risk
                for target_i in range(t_count):
                    observed_state[row, horizon_i, direction_i, target_i] = _state(
                        int(observed_favorable_index[row, direction_i, target_i]),
                        int(observed_adverse_index[row, direction_i]), terminal,
                    )
                    if not quote_valid[row, horizon_i]:
                        continue
                    favorable = int(market_favorable_index[row, direction_i, target_i])
                    adverse = int(market_adverse_index[row, direction_i])
                    state = _state(favorable, adverse, terminal)
                    market_state[row, horizon_i, direction_i, target_i] = state
                    exit_index = (
                        favorable if state == PATH_FAVORABLE_FIRST
                        else adverse if state == PATH_ADVERSE_FIRST else terminal
                    )
                    quote_entry = int(
                        index.ask_ticks[entry] if direction > 0 else index.bid_ticks[entry]
                    )
                    exit_ticks = int(
                        index.bid_ticks[exit_index] if direction > 0 else index.ask_ticks[exit_index]
                    )
                    market_gross_r[row, horizon_i, direction_i, target_i] = (
                        direction * (exit_ticks - quote_entry) / risk
                    )

    entry_source_file = np.full(n, -1, dtype=np.int64)
    entry_source_row = np.full(n, -1, dtype=np.int64)
    good_entry = entry_index >= 0
    entry_source_file[good_entry] = index.source_file_index[entry_index[good_entry]]
    entry_source_row[good_entry] = index.source_row_ordinal[entry_index[good_entry]]
    terminal_source_file = np.full((n, h_count), -1, dtype=np.int64)
    terminal_source_row = np.full((n, h_count), -1, dtype=np.int64)
    terminal_event_seq = np.zeros((n, h_count), dtype=np.uint64)
    good_terminal = terminal_index >= 0
    terminal_source_file[good_terminal] = index.source_file_index[terminal_index[good_terminal]]
    terminal_source_row[good_terminal] = index.source_row_ordinal[terminal_index[good_terminal]]
    terminal_event_seq[good_terminal] = index.event_seq[terminal_index[good_terminal]]
    observed_adverse_lineage = _event_lineage(index, observed_adverse_index)
    observed_favorable_lineage = _event_lineage(index, observed_favorable_index)
    market_adverse_lineage = _event_lineage(index, market_adverse_index)
    market_favorable_lineage = _event_lineage(index, market_favorable_index)

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "backend": backend,
        "config": asdict(config),
        "root": index.root,
        "contract_id": index.contract_id,
        "session_day": index.session_day,
        "split_use": index.split_use,
        "tick_size": index.tick_size,
        "tick_value": index.tick_value,
        "export_receipt_sha256": index.export_receipt_sha256,
        "source_shard_sha256": index.source_shard_sha256,
        "source_file_table_sha256": index.source_file_table_sha256,
        "corpus_contract_sha256": index.corpus_contract_sha256,
        "environment_receipt_sha256": index.environment_receipt_sha256,
        "instrument_spec_sha256": index.instrument_spec_sha256,
        "decision_manifest_sha256": expected_manifest,
        "algorithm_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "directions": np.asarray((1, -1), dtype=np.int8),
        "horizons_seconds": np.asarray(config.horizons_seconds, dtype=np.int64),
        "targets_r": np.asarray(config.targets_r, dtype=np.float64),
        "target_distance_ticks": target_distance_ticks,
        "decision_time_utc_ns": decision_ts.copy(),
        "decision_event_seq": decision_seq.copy(),
        "risk_ticks": risks.copy(),
        "risk_known_time_utc_ns": known_ts.copy(),
        "risk_known_event_seq": known_seq.copy(),
        "valid": valid,
        "invalid_reason": invalid_reason,
        "marketable_at_trade_valid": quote_valid,
        "entry_index": entry_index,
        "entry_time_utc_ns": entry_time_ns,
        "entry_event_seq": entry_event_seq,
        "entry_wait_ns": entry_wait_ns,
        "entry_source_file_index": entry_source_file,
        "entry_source_row_ordinal": entry_source_row,
        "terminal_index": terminal_index,
        "terminal_time_utc_ns": terminal_time_ns,
        "terminal_event_seq": terminal_event_seq,
        "terminal_source_file_index": terminal_source_file,
        "terminal_source_row_ordinal": terminal_source_row,
        "declared_label_end_utc_ns": declared_end_ns,
        "purge_time_utc_ns": declared_end_ns.copy(),
        "observed_mfe_r": observed_mfe_r,
        "observed_mae_r": observed_mae_r,
        "observed_terminal_r": observed_terminal_r,
        "observed_adverse_first_index_max_horizon": observed_adverse_index,
        "observed_adverse_first_time_utc_ns_max_horizon": observed_adverse_lineage[0],
        "observed_adverse_first_event_seq_max_horizon": observed_adverse_lineage[1],
        "observed_adverse_first_source_file_index_max_horizon": observed_adverse_lineage[2],
        "observed_adverse_first_source_row_ordinal_max_horizon": observed_adverse_lineage[3],
        "observed_favorable_first_index_max_horizon": observed_favorable_index,
        "observed_favorable_first_time_utc_ns_max_horizon": observed_favorable_lineage[0],
        "observed_favorable_first_event_seq_max_horizon": observed_favorable_lineage[1],
        "observed_favorable_first_source_file_index_max_horizon": observed_favorable_lineage[2],
        "observed_favorable_first_source_row_ordinal_max_horizon": observed_favorable_lineage[3],
        "observed_barrier_state": observed_state,
        "marketable_at_trade_adverse_first_index_max_horizon": market_adverse_index,
        "marketable_at_trade_adverse_first_time_utc_ns_max_horizon": market_adverse_lineage[0],
        "marketable_at_trade_adverse_first_event_seq_max_horizon": market_adverse_lineage[1],
        "marketable_at_trade_adverse_first_source_file_index_max_horizon": market_adverse_lineage[2],
        "marketable_at_trade_adverse_first_source_row_ordinal_max_horizon": market_adverse_lineage[3],
        "marketable_at_trade_favorable_first_index_max_horizon": market_favorable_index,
        "marketable_at_trade_favorable_first_time_utc_ns_max_horizon": market_favorable_lineage[0],
        "marketable_at_trade_favorable_first_event_seq_max_horizon": market_favorable_lineage[1],
        "marketable_at_trade_favorable_first_source_file_index_max_horizon": market_favorable_lineage[2],
        "marketable_at_trade_favorable_first_source_row_ordinal_max_horizon": market_favorable_lineage[3],
        "marketable_at_trade_barrier_state": market_state,
        "marketable_at_trade_gross_r": market_gross_r,
        "observed_trade_path_semantics": "first_touch_among_verified_observed_trade_records",
        "quote_semantics": "bbo_at_trade_proxy_not_continuous_quote_stream",
        "touch_index_semantics": (
            "compact_first_touch_through_largest_valid_horizon_use_touches_by_horizon_to_mask"
        ),
        "marketable_at_trade_is_fill_proof": False,
        "fees_included": False,
        "added_slippage_included": False,
        "source_verification": source_verification,
    }
    semantic = tick_label_fingerprint(result)
    result["semantic_fingerprint_sha256"] = semantic
    result["artifact_fingerprint_sha256"] = tick_label_artifact_fingerprint(result)
    if source_verification == "verified_contract_day_export":
        return VerifiedTickPathLabels(result, _token=_VERIFIED_LABEL_TOKEN)
    return result


def build_tick_path_labels(ticks: Any, **kwargs: Any) -> VerifiedTickPathLabels:
    """Build production labels only from an authentic, raw-verified export capability."""
    result = _build_tick_path_labels_impl(ticks, _allow_unverified=False, **kwargs)
    if type(result) is not VerifiedTickPathLabels or not result.is_authentic():
        raise ValueError("production label construction did not retain verified provenance")
    return result


def _build_tick_path_labels_for_test(ticks: Any, **kwargs: Any) -> dict[str, Any]:
    """Synthetic oracle surface; deliberately private and never accepted by production writers."""
    return _build_tick_path_labels_impl(ticks, _allow_unverified=True, **kwargs)


def tick_label_artifact_fingerprint(labels: Mapping[str, Any]) -> str:
    semantic = tick_label_fingerprint(labels)
    provenance = {
        "semantic_fingerprint_sha256": semantic,
        "export_receipt_sha256": labels["export_receipt_sha256"],
        "source_shard_sha256": labels["source_shard_sha256"],
        "source_file_table_sha256": labels["source_file_table_sha256"],
        "decision_manifest_sha256": labels["decision_manifest_sha256"],
        "algorithm_source_sha256": labels["algorithm_source_sha256"],
        "corpus_contract_sha256": labels["corpus_contract_sha256"],
        "environment_receipt_sha256": labels["environment_receipt_sha256"],
        "instrument_spec_sha256": labels["instrument_spec_sha256"],
    }
    return hashlib.sha256(
        json.dumps(provenance, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _canonical_array(array: np.ndarray) -> np.ndarray:
    value = np.asarray(array)
    target_dtype = value.dtype.newbyteorder("<")
    canonical = np.ascontiguousarray(value.astype(target_dtype, copy=False))
    if canonical.dtype.kind == "f" and np.isnan(canonical).any():
        canonical = canonical.copy()
        canonical[np.isnan(canonical)] = np.nan
    return canonical


def tick_label_fingerprint(labels: Mapping[str, Any]) -> str:
    excluded = {
        "semantic_fingerprint_sha256", "artifact_fingerprint_sha256", "backend",
        "export_receipt_sha256", "source_shard_sha256", "source_file_table_sha256",
        "decision_manifest_sha256", "algorithm_source_sha256", "corpus_contract_sha256",
        "environment_receipt_sha256",
        "instrument_spec_sha256",
    }
    digest = hashlib.sha256()
    for name in sorted(key for key in labels if key not in excluded):
        value = labels[name]
        digest.update(name.encode() + b"\0")
        if isinstance(value, np.ndarray):
            array = _canonical_array(value)
            digest.update(array.dtype.str.encode() + b"\0")
            digest.update(json.dumps(array.shape).encode() + b"\0")
            digest.update(array.tobytes())
        else:
            digest.update(
                json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
                + b"\0"
            )
    return digest.hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_tick_label_bundle_impl(
    labels: Mapping[str, Any], destination: str | Path, *,
    verified_export: Any = None, _allow_unverified: bool = False,
) -> Path:
    """Write a canonical, hash-verified directory bundle; refuse overwrite or object arrays."""
    if not _allow_unverified:
        from .corpus_v3_export import VerifiedContractDayExport
        if not (
            type(labels) is VerifiedTickPathLabels
            and labels.is_authentic()
            and type(verified_export) is VerifiedContractDayExport
            and verified_export.is_authentic()
            and _labels_match_verified_export(labels, verified_export)
        ):
            raise ValueError("production label bundles require the matching verified export capability")
    semantic = tick_label_fingerprint(labels)
    if labels.get("semantic_fingerprint_sha256") != semantic:
        raise ValueError("label semantic fingerprint is missing or stale")
    artifact = tick_label_artifact_fingerprint(labels)
    if labels.get("artifact_fingerprint_sha256") != artifact:
        raise ValueError("label artifact fingerprint is missing or stale")
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=False)
    arrays: dict[str, Any] = {}
    scalars: dict[str, Any] = {}
    for name in sorted(labels):
        value = labels[name]
        if isinstance(value, np.ndarray):
            if value.dtype.kind == "O":
                raise ValueError(f"object array {name!r} is forbidden in label bundles")
            filename = f"{name}.npy"
            path = root / filename
            with path.open("xb") as handle:
                np.save(handle, _canonical_array(value), allow_pickle=False)
            arrays[name] = {
                "file": filename,
                "dtype": _canonical_array(value).dtype.str,
                "shape": list(value.shape),
                "sha256": _sha256_path(path),
            }
        else:
            scalars[name] = value
    manifest: dict[str, Any] = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "label_schema_version": labels.get("schema_version"),
        "semantic_fingerprint_sha256": semantic,
        "artifact_fingerprint_sha256": artifact,
        "arrays": arrays,
        "scalars": scalars,
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    (root / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False),
        encoding="utf-8",
    )
    return root


def write_tick_label_bundle(
    labels: VerifiedTickPathLabels, destination: str | Path, *, verified_export: Any,
) -> Path:
    """Persist labels only when the matching verified export capability is supplied."""
    return _write_tick_label_bundle_impl(
        labels, destination, verified_export=verified_export, _allow_unverified=False,
    )


def _write_tick_label_bundle_for_test(labels: Mapping[str, Any], destination: str | Path) -> Path:
    return _write_tick_label_bundle_impl(labels, destination, _allow_unverified=True)


def _load_tick_label_bundle_impl(
    source: str | Path, *, verified_export: Any = None, _allow_unverified: bool = False,
) -> dict[str, Any]:
    """Load and fully verify a canonical label bundle before returning immutable arrays."""
    root = Path(source)
    manifest_path = root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("cannot read tick-label bundle manifest") from exc
    claimed_manifest = _sha(manifest.get("manifest_sha256"), "manifest_sha256")
    unhashed = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    actual_manifest = hashlib.sha256(
        json.dumps(unhashed, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    if actual_manifest != claimed_manifest:
        raise ValueError("tick-label bundle manifest hash mismatch")
    if manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise ValueError("unsupported tick-label bundle schema")
    arrays = manifest.get("arrays")
    scalars = manifest.get("scalars")
    if not isinstance(arrays, dict) or not isinstance(scalars, dict):
        raise ValueError("tick-label bundle arrays/scalars must be mappings")
    expected_files = {"manifest.json"} | {str(record.get("file")) for record in arrays.values()}
    actual_files = {path.name for path in root.iterdir() if path.is_file()}
    if actual_files != expected_files:
        raise ValueError("tick-label bundle file set mismatch")
    labels: dict[str, Any] = dict(scalars)
    for name, record in sorted(arrays.items()):
        if not re.fullmatch(r"[A-Za-z0-9_]+", str(name)) or not isinstance(record, dict):
            raise ValueError("invalid label array manifest entry")
        filename = record.get("file")
        if filename != f"{name}.npy":
            raise ValueError("label array filename is noncanonical")
        path = root / filename
        if _sha256_path(path) != _sha(record.get("sha256"), f"arrays.{name}.sha256"):
            raise ValueError(f"label array hash mismatch: {name}")
        array = np.load(path, allow_pickle=False)
        if array.dtype.kind == "O" or array.dtype.str != record.get("dtype"):
            raise ValueError(f"label array dtype mismatch: {name}")
        if list(array.shape) != record.get("shape"):
            raise ValueError(f"label array shape mismatch: {name}")
        array.flags.writeable = False
        labels[name] = array
    if labels.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("label payload schema mismatch")
    if tick_label_fingerprint(labels) != manifest.get("semantic_fingerprint_sha256"):
        raise ValueError("label semantic fingerprint mismatch after load")
    if tick_label_artifact_fingerprint(labels) != manifest.get("artifact_fingerprint_sha256"):
        raise ValueError("label artifact fingerprint mismatch after load")
    if _allow_unverified:
        return labels
    from .corpus_v3_export import VerifiedContractDayExport
    if not (
        type(verified_export) is VerifiedContractDayExport
        and verified_export.is_authentic()
        and labels.get("source_verification") == "verified_contract_day_export"
        and _labels_match_verified_export(labels, verified_export)
    ):
        raise ValueError("production label bundle load requires the matching verified export capability")
    return VerifiedTickPathLabels(labels, _token=_VERIFIED_LABEL_TOKEN)


def load_tick_label_bundle(
    source: str | Path, *, verified_export: Any,
) -> VerifiedTickPathLabels:
    """Load a production bundle only against its matching verified export capability."""
    result = _load_tick_label_bundle_impl(
        source, verified_export=verified_export, _allow_unverified=False,
    )
    if type(result) is not VerifiedTickPathLabels or not result.is_authentic():
        raise ValueError("production label load did not retain verified provenance")
    return result


def _load_tick_label_bundle_for_test(source: str | Path) -> dict[str, Any]:
    return _load_tick_label_bundle_impl(source, _allow_unverified=True)
