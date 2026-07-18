"""Independent verifier for AlphaForge Corpus-v3 contract/session exports.

The producer receipt is evidence, not authority.  This module re-hashes the final
receipt, output Parquet, producer source binding, pinned lake manifest, selected raw
leaves, instrument economics, row semantics, and source lineage before exposing rows
to the path-label engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping
from zoneinfo import ZoneInfo

import numpy as np

from .corpus_v3 import CorpusV3Error, load_contract, sha256_file


RECEIPT_SCHEMA_VERSION = "alphaforge_foundation_export_receipt_v2"
EXPORT_SCHEMA_VERSION = "alphaforge_foundation_contract_day_v2"
OUTPUT_FILENAME = "ticks.parquet"
ARRAY_FIELDS = (
    "timestamp_utc_ns", "time_us", "event_seq", "price", "bid", "ask",
    "quote_valid", "volume", "bid_volume", "ask_volume", "source_file_index",
    "source_row_ordinal",
)
_SHA_RE = re.compile(r"[0-9a-f]{64}")
_VERIFIED_CAPABILITY_TOKEN = object()
_SOURCE_FILE_RE = re.compile(
    r"^(?P<contract>.+)_(?P<start>\d{8}T\d{6}Z)_(?P<end>\d{8}T\d{6}Z)_ticks\.parquet$"
)


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False,
    ).encode("utf-8")


def _content_sha(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _load_json_strict(path: Path) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise CorpusV3Error(f"duplicate JSON key in {path.name}: {key}")
            result[key] = value
        return result
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs)
    except (OSError, json.JSONDecodeError) as exc:
        raise CorpusV3Error(f"cannot read canonical JSON: {path}") from exc
    if not isinstance(value, dict):
        raise CorpusV3Error(f"{path.name} must contain a JSON object")
    if path.read_bytes() != _json_bytes(value):
        raise CorpusV3Error(f"{path.name} is not canonical JSON")
    return value


def _require_sha(value: Any, name: str) -> str:
    text = str(value)
    if not _SHA_RE.fullmatch(text):
        raise CorpusV3Error(f"{name} must be a lowercase SHA-256")
    return text


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _canonical_array(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind == "O":
        raise CorpusV3Error("object arrays are forbidden in verified exports")
    array = np.ascontiguousarray(array.astype(array.dtype.newbyteorder("<"), copy=False))
    if array.dtype.kind == "f" and np.isnan(array).any():
        array = array.copy()
        array[np.isnan(array)] = np.nan
    return array


def _output_schema():
    try:
        import pyarrow as pa
    except ImportError as exc:  # pragma: no cover - optional data environment
        raise CorpusV3Error("Corpus-v3 export verification requires the data extra") from exc
    return pa.schema([
        pa.field("timestamp_utc_ns", pa.int64(), nullable=False),
        pa.field("time_us", pa.int64(), nullable=False),
        pa.field("event_seq", pa.uint64(), nullable=False),
        pa.field("price", pa.float64(), nullable=False),
        pa.field("bid", pa.float64(), nullable=False),
        pa.field("ask", pa.float64(), nullable=False),
        pa.field("quote_valid", pa.bool_(), nullable=False),
        pa.field("volume", pa.float64(), nullable=False),
        pa.field("bid_volume", pa.float64(), nullable=False),
        pa.field("ask_volume", pa.float64(), nullable=False),
        pa.field("source_file_index", pa.uint32(), nullable=False),
        pa.field("source_row_ordinal", pa.uint64(), nullable=False),
    ])


def _schema_sha256() -> str:
    schema = _output_schema()
    fields = [
        {"name": field.name, "type": str(field.type), "nullable": field.nullable}
        for field in schema
    ]
    return _content_sha({"schema_version": EXPORT_SCHEMA_VERSION, "fields": fields})


def _semantic_sha256(arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(b"alphaforge-foundation-semantic-shard-v1\0")
    digest.update(_json_bytes(metadata))
    for name in ARRAY_FIELDS:
        array = _canonical_array(arrays[name])
        digest.update(name.encode("ascii") + b"\0")
        digest.update(array.dtype.str.encode("ascii") + b"\0")
        digest.update(np.asarray(array.size, dtype="<u8").tobytes())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _tick_aligned(values: np.ndarray, tick_size: float) -> np.ndarray:
    finite = np.isfinite(values)
    result = np.zeros(values.shape, dtype=bool)
    scaled = values[finite] / tick_size
    result[finite] = np.abs(scaled - np.rint(scaled)) <= 1e-6
    return result


def _safe_source_path(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts or not candidate.parts:
        raise CorpusV3Error(f"unsafe source path in receipt: {relative!r}")
    resolved_root = root.resolve()
    cursor = resolved_root
    for part in candidate.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise CorpusV3Error(f"source path uses a symlink: {relative!r}")
    resolved = cursor.resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise CorpusV3Error(f"source path escapes the pinned physical root: {relative!r}") from exc
    return resolved


def _expected_session_bounds(
    contract: Mapping[str, Any], *, root: str, session_day: str,
) -> tuple[int, int, str]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise CorpusV3Error("calendar verification requires the data extra") from exc
    artifact = (contract.get("artifacts") or {}).get("market_calendar") or {}
    path = Path(str(artifact.get("path")))
    expected_sha = _require_sha(artifact.get("sha256"), "market calendar sha256")
    if not path.is_file() or sha256_file(path) != expected_sha:
        raise CorpusV3Error("pinned market calendar is unavailable or changed")
    calendar = yaml.safe_load(path.read_text()) or {}
    products = [
        product for product in (calendar.get("products") or {}).values()
        if root in (product.get("roots") or [])
    ]
    if len(products) != 1:
        raise CorpusV3Error("pinned calendar does not map root to exactly one product")
    normal = products[0].get("normal_day") or {}
    try:
        open_s, end_s = int(normal["session_open_s"]), int(normal["maintenance_start_s"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CorpusV3Error("pinned calendar lacks normal session bounds") from exc
    day = date.fromisoformat(session_day)
    start_day = day - timedelta(days=1) if open_s >= end_s else day
    tz = ZoneInfo(str(calendar.get("timezone")))
    start = datetime.combine(start_day, datetime.min.time(), tzinfo=tz) + timedelta(seconds=open_s)
    end = datetime.combine(day, datetime.min.time(), tzinfo=tz) + timedelta(seconds=end_s)
    return (
        int(start.astimezone(timezone.utc).timestamp() * 1_000_000_000),
        int(end.astimezone(timezone.utc).timestamp() * 1_000_000_000),
        expected_sha,
    )


def _verify_selected_sources(
    contract: Mapping[str, Any], receipt: Mapping[str, Any],
) -> dict[int, Path]:
    source_table = (receipt.get("source_file_table") or {}).get("files") or []
    leaf_record = (contract.get("artifacts") or {}).get("lake_leaf_manifest") or {}
    manifest_path = Path(str(leaf_record.get("path")))
    expected_manifest_sha = _require_sha(leaf_record.get("sha256"), "lake leaf manifest")
    if not manifest_path.is_file() or sha256_file(manifest_path) != expected_manifest_sha:
        raise CorpusV3Error("pinned lake leaf manifest is unavailable or has changed")
    if receipt.get("leaf_manifest_sha256") != expected_manifest_sha:
        raise CorpusV3Error("producer receipt does not bind the pinned lake leaf manifest")
    wanted = {str(row.get("path")): row for row in source_table}
    if len(wanted) != len(source_table) or any(not path for path in wanted):
        raise CorpusV3Error("source file table contains duplicate or empty logical paths")
    observed: dict[str, dict[str, Any]] = {}
    with gzip.open(manifest_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            logical = str(row.get("path"))
            if logical in wanted:
                observed[logical] = row
    if set(observed) != set(wanted):
        raise CorpusV3Error("selected source is absent from the pinned lake leaf manifest")
    physical_root = Path(str((contract.get("source") or {}).get("physical_root")))
    paths: dict[int, Path] = {}
    request = receipt.get("request") or {}
    expected_contract = str(request.get("contract_id"))
    intervals: list[tuple[int, int]] = []
    for logical, record in wanted.items():
        if type(record.get("source_file_index")) is not int or int(record["source_file_index"]) < 0:
            raise CorpusV3Error("source file index must be a nonnegative integer")
        if type(record.get("rows")) is not int or int(record["rows"]) < 0:
            raise CorpusV3Error("source row count must be a nonnegative integer")
        physical_relative = str(record.get("physical_relative_path"))
        physical_parts = Path(physical_relative).parts
        if len(physical_parts) != 3 or physical_parts[0] != expected_contract:
            raise CorpusV3Error("source physical path does not belong to the requested contract")
        if logical != f"sc_v2_ticks/raw/{physical_relative}":
            raise CorpusV3Error("logical and physical source paths are not cross-derived")
        match = _SOURCE_FILE_RE.fullmatch(physical_parts[-1])
        if not match or match.group("contract") != expected_contract:
            raise CorpusV3Error("source filename does not bind the requested contract")
        parse = lambda text: int(
            datetime.strptime(text, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc).timestamp()
            * 1_000_000_000
        )
        interval = (parse(match.group("start")), parse(match.group("end")))
        if interval != (
            int(record.get("filename_interval_start_utc_ns", -1)),
            int(record.get("filename_interval_end_utc_ns", -1)),
        ):
            raise CorpusV3Error("source filename interval differs from the receipt")
        intervals.append(interval)
        leaf = observed[logical]
        expected_sha = _require_sha(record.get("sha256"), "selected source sha256")
        if expected_sha != leaf.get("sha256") or int(record.get("size", -1)) != int(leaf.get("size", -2)):
            raise CorpusV3Error("selected source table differs from the pinned lake leaf")
        path = _safe_source_path(physical_root, str(record.get("physical_relative_path")))
        if not path.is_file() or path.stat().st_size != int(record["size"]) or sha256_file(path) != expected_sha:
            raise CorpusV3Error(f"selected source bytes are unavailable or changed: {logical}")
        paths[int(record["source_file_index"])] = path
    evidence = receipt.get("session_bounds_and_internal_gap_evidence") or {}
    session_start, session_end = int(evidence["session_start_utc_ns"]), int(evidence["session_end_utc_ns"])
    ordered = sorted(intervals)
    for previous, current in zip(ordered, ordered[1:]):
        if current[0] < previous[1]:
            raise CorpusV3Error("source filename intervals overlap")
    cursor = session_start
    for interval_start, interval_end in ordered:
        if interval_start > cursor:
            raise CorpusV3Error("source filename intervals leave a session gap")
        cursor = max(cursor, interval_end)
    if cursor < session_end:
        raise CorpusV3Error("source filename intervals do not cover the session")
    return paths


def _verify_instrument(
    contract: Mapping[str, Any], receipt: Mapping[str, Any],
) -> tuple[float, float, float]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise CorpusV3Error("instrument verification requires the data extra") from exc
    artifact = (contract.get("artifacts") or {}).get("instrument_economics") or {}
    path = Path(str(artifact.get("path")))
    expected_sha = _require_sha(artifact.get("sha256"), "instrument artifact sha256")
    if not path.is_file() or sha256_file(path) != expected_sha:
        raise CorpusV3Error("pinned instrument-economics artifact is unavailable or changed")
    instrument = receipt.get("instrument_spec") or {}
    request = receipt.get("request") or {}
    if instrument.get("source_artifact_sha256") != expected_sha or instrument.get("root") != request.get("root"):
        raise CorpusV3Error("producer instrument specification does not bind the pinned artifact")
    spec = (yaml.safe_load(path.read_text()) or {}).get("instruments", {}).get(request.get("root"))
    if not isinstance(spec, dict):
        raise CorpusV3Error("requested root is missing from pinned instrument economics")
    tick_size = float(instrument.get("tick_size"))
    tick_value = float(instrument.get("tick_value_usd"))
    if tick_size != float(spec.get("tick_size")) or tick_value != float(spec.get("tick_value_usd")):
        raise CorpusV3Error("receipt tick economics differ from the pinned instrument artifact")
    registry_artifact = (contract.get("artifacts") or {}).get("data_source_registry") or {}
    registry_path = Path(str(registry_artifact.get("path")))
    registry_sha = _require_sha(registry_artifact.get("sha256"), "source registry sha256")
    if not registry_path.is_file() or sha256_file(registry_path) != registry_sha:
        raise CorpusV3Error("pinned source registry is unavailable or changed")
    if instrument.get("price_normalization_artifact_sha256") != registry_sha:
        raise CorpusV3Error("producer price normalization does not bind the pinned source registry")
    registry = yaml.safe_load(registry_path.read_text()) or {}
    try:
        multiplier = float(
            registry["sources"]["sierra_chart"]["dataset_governance"]["ticks"]
            ["price_normalization"]["multiplier_by_root"][request.get("root")]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CorpusV3Error("pinned source registry lacks the requested price multiplier") from exc
    if float(instrument.get("source_price_multiplier")) != multiplier or multiplier <= 0:
        raise CorpusV3Error("producer source-price multiplier differs from the pinned registry")
    return tick_size, tick_value, multiplier


def _source_timestamp_ns(column: Any, field_type: Any) -> np.ndarray:
    import pyarrow as pa
    if field_type != pa.timestamp("us", tz="UTC"):
        raise CorpusV3Error(f"raw source timestamp type is not timestamp[us, tz=UTC]: {field_type}")
    values = column.cast(pa.int64()).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    limit = np.iinfo(np.int64).max // 1_000
    if np.any(values > limit) or np.any(values < -limit):
        raise CorpusV3Error("raw source timestamp overflows nanosecond representation")
    return values * np.int64(1_000)


def _verify_output_against_raw_sources(
    receipt: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    source_paths: Mapping[int, Path],
    *,
    price_multiplier: float,
    tick_size: float,
) -> None:
    """Reconstruct all in-session rows so genuine leaves cannot cover invented output."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    evidence = receipt["session_bounds_and_internal_gap_evidence"]
    start = int(evidence["session_start_utc_ns"])
    end = int(evidence["session_end_utc_ns"])
    source_table = (receipt.get("source_file_table") or {}).get("files") or []
    expected: dict[str, list[np.ndarray]] = {name: [] for name in ARRAY_FIELDS}
    required = {"timestamp", "event_seq", "price", "bid", "ask", "volume"}
    for record in source_table:
        index = int(record["source_file_index"])
        path = source_paths[index]
        parquet = pq.ParquetFile(path)
        names = set(parquet.schema.names)
        if required - names:
            raise CorpusV3Error("raw source is missing a required reconstruction field")
        timestamp_field = parquet.schema_arrow.field("timestamp")
        sequence_field = parquet.schema_arrow.field("event_seq")
        metadata = parquet.schema_arrow.metadata or {}
        numeric_fields_exact = all(
            parquet.schema_arrow.field(name).type == pa.float64()
            and parquet.schema_arrow.field(name).nullable is False
            for name in ("price", "bid", "ask", "volume", "bid_volume", "ask_volume")
            if name in names
        )
        if not (
            timestamp_field.type == pa.timestamp("us", tz="UTC")
            and timestamp_field.nullable is False
            and sequence_field.type == pa.uint64()
            and sequence_field.nullable is False
            and metadata.get(b"event_sequence_scope") == b"file_order"
            and metadata.get(b"source_system") == b"sierra_chart_dtc"
            and metadata.get(b"dtc_endpoint") == b"historical_price_data"
            and metadata.get(b"dtc_record_interval") == b"0"
            and numeric_fields_exact
        ):
            raise CorpusV3Error("raw source timestamp/event_seq schema violates the governed contract")
        columns = list(required) + [name for name in ("bid_volume", "ask_volume") if name in names]
        row_base = 0
        for batch in parquet.iter_batches(batch_size=262_144, columns=columns):
            length = batch.num_rows
            timestamps = _source_timestamp_ns(
                batch.column(batch.schema.get_field_index("timestamp")),
                batch.schema.field("timestamp").type,
            )
            sequences = batch.column(batch.schema.get_field_index("event_seq")).to_numpy(
                zero_copy_only=False,
            ).astype(np.uint64, copy=False)
            if not np.array_equal(sequences, np.arange(row_base, row_base + length, dtype=np.uint64)):
                raise CorpusV3Error("raw source event_seq is not its physical row ordinal")
            in_session = (timestamps >= start) & (timestamps < end)

            def values(name: str, default: float = np.nan) -> np.ndarray:
                position = batch.schema.get_field_index(name)
                if position < 0:
                    return np.full(length, default, dtype=np.float64)
                return batch.column(position).to_numpy(zero_copy_only=False).astype(np.float64, copy=False)

            price = values("price") * price_multiplier
            if np.any(in_session & ~np.isfinite(price)):
                raise CorpusV3Error("raw source has a nonfinite in-session trade")
            if np.any(in_session & ~_tick_aligned(price, tick_size)):
                raise CorpusV3Error("raw source has an off-grid in-session trade")
            volume = values("volume")
            if np.any(in_session & (~np.isfinite(volume) | (volume < 0))):
                raise CorpusV3Error("raw source has invalid in-session volume")
            bid, ask = values("bid") * price_multiplier, values("ask") * price_multiplier
            quote_valid = _tick_aligned(bid, tick_size) & _tick_aligned(ask, tick_size) & (bid <= ask)
            count = int(np.count_nonzero(in_session))
            if count:
                selected_ts = timestamps[in_session]
                expected["timestamp_utc_ns"].append(selected_ts)
                expected["time_us"].append((selected_ts - start) // 1_000)
                expected["event_seq"].append(sequences[in_session])
                expected["price"].append(price[in_session])
                expected["bid"].append(bid[in_session])
                expected["ask"].append(ask[in_session])
                expected["quote_valid"].append(quote_valid[in_session])
                expected["volume"].append(volume[in_session])
                expected["bid_volume"].append(values("bid_volume")[in_session])
                expected["ask_volume"].append(values("ask_volume")[in_session])
                expected["source_file_index"].append(np.full(count, index, dtype=np.uint32))
                expected["source_row_ordinal"].append(
                    np.arange(row_base, row_base + length, dtype=np.uint64)[in_session]
                )
            row_base += length
        if row_base != int(record["rows"]):
            raise CorpusV3Error("raw source footer row count differs from the receipt")
    if not expected["timestamp_utc_ns"]:
        raise CorpusV3Error("verified raw sources contain no in-session trades")
    rebuilt = {name: _canonical_array(np.concatenate(parts)) for name, parts in expected.items()}
    order = np.lexsort((rebuilt["event_seq"], rebuilt["timestamp_utc_ns"]))
    rebuilt = {name: value[order] for name, value in rebuilt.items()}
    for name in ARRAY_FIELDS:
        observed, wanted = arrays[name], rebuilt[name]
        if observed.dtype.kind == "f":
            equal = np.array_equal(observed, wanted, equal_nan=True)
        else:
            equal = np.array_equal(observed, wanted)
        if not equal:
            raise CorpusV3Error(f"producer output does not reconstruct from raw source lineage: {name}")


@dataclass(frozen=True)
class VerifiedContractDayExport:
    """Immutable capability returned only after independent byte and semantic verification."""

    export_path: Path
    receipt_sha256: str
    contract_sha256: str
    receipt: Mapping[str, Any]
    arrays: Mapping[str, np.ndarray]
    tick_size: float
    tick_value: float
    root: str
    contract_id: str
    session_day: str
    split_use: str
    session_start_utc_ns: int
    session_end_utc_ns: int
    output_shard_sha256: str
    source_file_table_sha256: str
    environment_receipt_sha256: str
    instrument_spec_sha256: str
    _capability_token: object = field(repr=False, compare=False)

    def is_authentic(self) -> bool:
        return self._capability_token is _VERIFIED_CAPABILITY_TOKEN

    def label_rows(self) -> Mapping[str, Any]:
        timestamps = self.arrays["timestamp_utc_ns"]
        if _semantic_sha256(self.arrays, dict(self.receipt["semantic_metadata"])) != self.receipt[
            "semantic_shard_sha256"
        ]:
            raise CorpusV3Error("verified export arrays changed after admission")
        result: dict[str, Any] = {
            name: np.asarray(value).copy() for name, value in self.arrays.items()
        }
        result.update({
            "root": self.root,
            "contract_id": self.contract_id,
            "session_day": self.session_day,
            "split_use": self.split_use,
            "session_start_utc_ns": self.session_start_utc_ns,
            "session_end_utc_ns": self.session_end_utc_ns,
            "coverage_start_utc_ns": int(timestamps[0]),
            "coverage_end_utc_ns": int(timestamps[-1]),
            "export_receipt_sha256": self.receipt_sha256,
            "source_shard_sha256": self.output_shard_sha256,
            "source_file_table_sha256": self.source_file_table_sha256,
            "corpus_contract_sha256": self.contract_sha256,
            "environment_receipt_sha256": self.environment_receipt_sha256,
            "instrument_spec_sha256": self.instrument_spec_sha256,
            "tick_size": self.tick_size,
            "tick_value": self.tick_value,
        })
        return MappingProxyType(result)

    def build_path_index(self, *, config: Any = None):
        from .tick_path_labels import OrderedTickPathIndex, TickPathLabelConfig
        selected = config or TickPathLabelConfig()
        return OrderedTickPathIndex(self.label_rows(), tick_size=self.tick_size, config=selected)


def verify_contract_day_export(
    export_path: str | Path,
    *,
    contract_path: str | Path,
    expected_request: Mapping[str, Any],
    allow_test_contract: bool = False,
) -> VerifiedContractDayExport:
    """Fail closed unless the producer bundle, raw leaves, and consumer contract agree."""
    root = Path(export_path)
    if not root.is_dir() or {path.name for path in root.iterdir() if path.is_file()} != {
        OUTPUT_FILENAME, "receipt.json",
    }:
        raise CorpusV3Error("export must contain exactly ticks.parquet and receipt.json")
    contract_file = Path(contract_path)
    canonical_contract = Path(__file__).resolve().parents[1] / "config" / "corpus_v3" / "contract.json"
    if not allow_test_contract and contract_file.resolve() != canonical_contract.resolve():
        raise CorpusV3Error("production verification requires the canonical packaged Corpus-v3 contract")
    contract = load_contract(contract_file)
    contract_sha = sha256_file(contract_file)
    receipt_path = root / "receipt.json"
    receipt = _load_json_strict(receipt_path)
    declared_receipt_fields = set(
        (contract.get("required_export_seam") or {}).get("required_receipt_bindings") or []
    )
    if set(receipt) != declared_receipt_fields:
        raise CorpusV3Error("producer receipt fields do not exactly match the sealed receipt-v2 contract")
    if receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION or receipt.get("status") != "complete":
        raise CorpusV3Error("unsupported or incomplete producer receipt")
    if receipt.get("purpose") != "foundation_training":
        raise CorpusV3Error("producer receipt has the wrong governed purpose")
    payload = dict(receipt)
    supplied_payload_sha = payload.pop("receipt_payload_sha256", None)
    if supplied_payload_sha != _content_sha(payload):
        raise CorpusV3Error("producer receipt payload hash mismatch")
    if receipt.get("consumer_contract_sha256") != contract_sha:
        raise CorpusV3Error("producer export was not built against the current Corpus-v3 contract")
    if receipt.get("lake_hash_of_hashes_sha256") != contract["source"]["hash_of_hashes_sha256"]:
        raise CorpusV3Error("producer lake hash-of-hashes differs from the Corpus-v3 contract")
    request = receipt.get("request") or {}
    expected_request = dict(expected_request)
    if request != expected_request:
        raise CorpusV3Error("producer request does not match the caller-declared bundle identity")
    if request.get("purpose") != "foundation_training":
        raise CorpusV3Error("request purpose is not foundation_training")
    root_name, contract_id = str(request.get("root")), str(request.get("contract_id"))
    if not re.fullmatch(rf"{re.escape(root_name)}[FGHJKMNQUVXZ]\d{{2}}", contract_id):
        raise CorpusV3Error("request contract_id is not a dated contract matching root")
    try:
        from datetime import date
        date.fromisoformat(str(request.get("session_day")))
    except ValueError as exc:
        raise CorpusV3Error("request session_day is not a valid ISO date") from exc
    if receipt.get("request_sha256") != _content_sha(request):
        raise CorpusV3Error("producer request hash mismatch")
    if request.get("root") not in contract["admitted_roots"]:
        raise CorpusV3Error("producer request root is not admitted")
    split_name = str(request.get("split_use"))
    split = (contract.get("splits") or {}).get(split_name)
    day = str(request.get("session_day"))
    if not isinstance(split, dict) or not (str(split.get("start")) <= day < str(split.get("end_exclusive"))):
        raise CorpusV3Error("producer request is outside its declared split")
    allowed_split_uses = {
        "foundation_pretraining": "training_only",
        "supervised_training": "training_only",
        "development": "validation_model_selection",
    }
    if split_name not in allowed_split_uses or split.get("use") != allowed_split_uses[split_name]:
        raise CorpusV3Error("training verifier refuses excluded holdout/coverage-only rows")
    seam = contract.get("required_export_seam") or {}
    admission = contract.get("current_admission") or {}
    if admission.get("materialization") not in {"representative_shard_only", "enabled"}:
        raise CorpusV3Error("Corpus-v3 materialization remains blocked by the canonical contract")
    if admission.get("materialization") == "representative_shard_only" and seam.get("pilot_request") != expected_request:
        raise CorpusV3Error("request is not the one predeclared representative pilot shard")
    expected_exporter = _require_sha(seam.get("producer_exporter_sha256"), "producer exporter sha256")
    producer_files = (receipt.get("producer_source_manifest") or {}).get("files") or []
    matches = [row for row in producer_files if row.get("path") == "src/alphaforge/foundation_export.py"]
    if len(matches) != 1 or matches[0].get("sha256") != expected_exporter:
        raise CorpusV3Error("producer source manifest does not bind the admitted exporter")
    repository = (receipt.get("producer_source_manifest") or {}).get("repository") or {}
    if repository.get("git_clean") is not True or not re.fullmatch(
        r"[0-9a-f]{40}", str(repository.get("git_commit"))
    ) or repository.get("git_commit") != seam.get("producer_git_commit"):
        raise CorpusV3Error("producer repository was not a clean reproducible Git revision")
    for object_field, hash_field in (
        ("producer_source_manifest", "producer_source_manifest_sha256"),
        ("environment", "environment_receipt_sha256"),
        ("instrument_spec", "instrument_spec_sha256"),
        ("source_file_table", "source_file_table_sha256"),
    ):
        if _content_sha(receipt.get(object_field)) != receipt.get(hash_field):
            raise CorpusV3Error(f"producer object binding mismatch: {hash_field}")
    artifacts = contract.get("artifacts") or {}
    expected_governance = {
        "data_admission_sha256": (artifacts.get("data_admission") or {}).get("sha256"),
        "qa_artifact_sha256": (artifacts.get("tick_admission") or {}).get("sha256"),
        "loader_smoke_sha256": (artifacts.get("loader_smoke") or {}).get("sha256"),
        "registry_sha256": (artifacts.get("data_source_registry") or {}).get("sha256"),
    }
    if receipt.get("governance") != expected_governance or receipt.get(
        "governance_sha256"
    ) != _content_sha(expected_governance):
        raise CorpusV3Error("producer governance evidence differs from pinned artifacts")
    if receipt.get("output_schema_sha256") != _schema_sha256():
        raise CorpusV3Error("producer output schema hash mismatch")
    shard = root / OUTPUT_FILENAME
    if sha256_file(shard) != receipt.get("output_shard_sha256"):
        raise CorpusV3Error("producer Parquet byte hash mismatch")
    try:
        import pyarrow.parquet as pq
        parquet = pq.ParquetFile(shard)
    except Exception as exc:
        raise CorpusV3Error("cannot open producer Parquet shard") from exc
    if not parquet.schema_arrow.equals(_output_schema(), check_metadata=True):
        raise CorpusV3Error("producer Parquet schema mismatch")
    table = parquet.read(columns=list(ARRAY_FIELDS))
    arrays = {
        name: _canonical_array(table.column(name).combine_chunks().to_numpy(zero_copy_only=False))
        for name in ARRAY_FIELDS
    }
    row_count = len(arrays["timestamp_utc_ns"])
    if row_count <= 0 or row_count != int((receipt.get("output_row_counts") or {}).get("trade_rows", -1)):
        raise CorpusV3Error("producer row count mismatch")
    if _semantic_sha256(arrays, receipt.get("semantic_metadata") or {}) != receipt.get("semantic_shard_sha256"):
        raise CorpusV3Error("producer semantic shard hash mismatch")
    timestamps, sequences = arrays["timestamp_utc_ns"], arrays["event_seq"]
    if np.any(timestamps[1:] < timestamps[:-1]) or np.any(
        (timestamps[1:] == timestamps[:-1]) & (sequences[1:] <= sequences[:-1])
    ):
        raise CorpusV3Error("producer event keys are not strict and unique")
    evidence = receipt.get("session_bounds_and_internal_gap_evidence") or {}
    start, end = int(evidence.get("session_start_utc_ns", 0)), int(evidence.get("session_end_utc_ns", 0))
    expected_start, expected_end, calendar_sha = _expected_session_bounds(
        contract, root=root_name, session_day=day,
    )
    if (start, end) != (expected_start, expected_end) or evidence.get(
        "market_calendar_sha256"
    ) != calendar_sha:
        raise CorpusV3Error("producer session bounds differ from the pinned market calendar")
    expected_semantic_metadata = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "root": root_name,
        "contract_id": contract_id,
        "session_day": day,
        "session_start_utc_ns": start,
        "session_end_utc_ns": end,
        "source_file_table_sha256": receipt.get("source_file_table_sha256"),
        "consumer_contract_sha256": contract_sha,
    }
    if receipt.get("semantic_metadata") != expected_semantic_metadata:
        raise CorpusV3Error("producer semantic metadata is not independently cross-bound")
    if not (start <= int(timestamps[0]) <= int(timestamps[-1]) < end):
        raise CorpusV3Error("producer rows fall outside the exchange session")
    if not np.array_equal(arrays["time_us"], (timestamps - start) // 1_000):
        raise CorpusV3Error("time_us is not elapsed microseconds from session start")
    if evidence.get("internal_source_interval_gaps") != []:
        raise CorpusV3Error("producer reports an internal source-container interval gap")
    if receipt.get("continuous_market_completeness") is not False or receipt.get(
        "market_path_completeness_claim"
    ) != "complete_among_verified_observed_source_records_only":
        raise CorpusV3Error("producer overclaims market-path completeness")
    ordering = receipt.get("source_ordering_evidence") or {}
    if (
        ordering.get("output_order") != ["timestamp_utc_ns", "event_seq"]
        or ordering.get("ordering_semantics")
        != "historical_event_time_not_live_arrival_or_zero_delay_proof"
        or type(ordering.get("timestamp_inversion_count")) is not int
        or type(ordering.get("max_source_timestamp_regression_ns")) is not int
    ):
        raise CorpusV3Error("producer ordering evidence is unsupported or overclaims live equivalence")
    tick_size, tick_value, price_multiplier = _verify_instrument(contract, receipt)
    if np.any(~_tick_aligned(arrays["price"], tick_size)):
        raise CorpusV3Error("producer trade prices are off the pinned tick grid")
    expected_quotes = (
        _tick_aligned(arrays["bid"], tick_size)
        & _tick_aligned(arrays["ask"], tick_size)
        & (arrays["bid"] <= arrays["ask"])
    )
    if not np.array_equal(arrays["quote_valid"], expected_quotes):
        raise CorpusV3Error("producer quote-validity mask is inconsistent")
    negative = receipt.get("negative_price_preservation") or {}
    if (
        negative.get("policy")
        != "finite_tick_aligned_trade_prices_are_preserved_without_positive_filter"
        or int(negative.get("negative_trade_rows", -1)) != int(np.count_nonzero(arrays["price"] < 0))
        or int(negative.get("zero_trade_rows", -1)) != int(np.count_nonzero(arrays["price"] == 0))
    ):
        raise CorpusV3Error("producer negative-price preservation claim is inconsistent")
    quote_claim = receipt.get("trade_row_preservation_independent_of_quote_validity") or {}
    if (
        quote_claim.get("policy")
        != "invalid_bbo_at_trade_sets_quote_valid_false_without_dropping_trade"
        or int(quote_claim.get("invalid_quote_rows_preserved", -1))
        != int(np.count_nonzero(~arrays["quote_valid"]))
    ):
        raise CorpusV3Error("producer invalid-quote preservation claim is inconsistent")
    files = (receipt.get("source_file_table") or {}).get("files") or []
    indexes = [int(row.get("source_file_index", -1)) for row in files]
    if indexes != list(range(len(files))):
        raise CorpusV3Error("producer source-file indexes are not contiguous")
    file_index = arrays["source_file_index"].astype(np.int64)
    if np.any(file_index < 0) or np.any(file_index >= len(files)):
        raise CorpusV3Error("row source_file_index is invalid")
    if any(type(row.get("rows")) is not int or int(row["rows"]) < 0 for row in files):
        raise CorpusV3Error("producer source row counts must be nonnegative integers")
    limits = np.asarray([int(row["rows"]) for row in files], dtype=np.int64)
    if np.any(arrays["source_row_ordinal"] >= limits[file_index].astype(np.uint64)):
        raise CorpusV3Error("row source ordinal exceeds the physical source file")
    lineage = np.rec.fromarrays([arrays["source_file_index"], arrays["source_row_ordinal"]], names="file,row")
    if len(np.unique(lineage)) != row_count:
        raise CorpusV3Error("producer source lineage is duplicated")
    source_paths = _verify_selected_sources(contract, receipt)
    _verify_output_against_raw_sources(
        receipt, arrays, source_paths,
        price_multiplier=price_multiplier, tick_size=tick_size,
    )
    frozen_arrays: dict[str, np.ndarray] = {}
    for name, array in arrays.items():
        array.flags.writeable = False
        frozen_arrays[name] = array
    return VerifiedContractDayExport(
        export_path=root.resolve(),
        receipt_sha256=sha256_file(receipt_path),
        contract_sha256=contract_sha,
        receipt=_deep_freeze(receipt),
        arrays=MappingProxyType(frozen_arrays),
        tick_size=tick_size,
        tick_value=tick_value,
        root=root_name,
        contract_id=contract_id,
        session_day=day,
        split_use=split_name,
        session_start_utc_ns=start,
        session_end_utc_ns=end,
        output_shard_sha256=_require_sha(receipt.get("output_shard_sha256"), "output shard sha256"),
        source_file_table_sha256=_require_sha(
            receipt.get("source_file_table_sha256"), "source file table sha256",
        ),
        environment_receipt_sha256=_require_sha(
            receipt.get("environment_receipt_sha256"), "environment receipt sha256",
        ),
        instrument_spec_sha256=_require_sha(
            receipt.get("instrument_spec_sha256"), "instrument spec sha256",
        ),
        _capability_token=_VERIFIED_CAPABILITY_TOKEN,
    )


__all__ = ["VerifiedContractDayExport", "verify_contract_day_export"]
