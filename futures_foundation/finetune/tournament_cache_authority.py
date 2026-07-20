"""External authority and transformation receipts for tournament caches.

The source corpus manifest is useful only after an out-of-band expected hash admits
its bytes.  This module turns that expected hash into a non-forgeable capability,
reopens every authority input through no-follow bounded transport, and exposes only
source streams enumerated by the admitted corpus manifest.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from futures_foundation._authority_bundle_io import (
    AuthorityBundleIOError,
    canonical_absolute_path,
    canonical_json_bytes,
    content_sha256,
    read_canonical_json_file,
    read_regular_file,
    require_sha256,
    sha256_regular_file,
)


SOURCE_AUTHORITY_SCHEMA_VERSION = "ffm_tournament_source_authority_v1"
SOURCE_MANIFEST_SCHEMA_VERSION = "ffm_ssl_corpus_v1"
CACHE_TRANSFORMATION_ID = "ffm_tournament_cache_materialization_v1"
_MAX_AUTHORITY_BYTES = 2 * 1024 * 1024
_MAX_MANIFEST_BYTES = 32 * 1024 * 1024
_MAX_JSON_NODES = 500_000
_MAX_JSON_DEPTH = 24
_CAPABILITY_TOKEN = object()


@dataclass(frozen=True)
class VerifiedTournamentSourceAuthority:
    path: Path
    physical_sha256: str
    semantic_sha256: str
    manifest_path: Path
    manifest_physical_sha256: str
    manifest_bytes: int
    admitted_streams: tuple[str, ...]
    document: Mapping[str, Any]
    manifest: Mapping[str, Any]
    _token: object


def _exact_keys(value: Any, expected: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        raise ValueError(f"{label} keys mismatch; missing={missing}, unknown={unknown}")
    return value


def _positive_int(value: Any, label: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _strict_json_bytes(raw: bytes, *, label: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in items:
            if key in output:
                raise ValueError(f"duplicate JSON key in {label}: {key}")
            output[key] = value
        return output

    try:
        document = json.loads(
            raw,
            object_pairs_hook=pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant in {label}: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise ValueError(f"cannot parse {label}") from exc
    nodes = 0

    def walk(value: Any, depth: int) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > _MAX_JSON_NODES or depth > _MAX_JSON_DEPTH:
            raise ValueError(f"{label} exceeds JSON limits")
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
    if not isinstance(document, dict):
        raise ValueError(f"{label} must be a JSON object")
    return document


def _read_json_file(
    path: str | Path, *, label: str, max_bytes: int,
) -> tuple[Path, dict[str, Any], str, int]:
    try:
        source, raw, physical = read_regular_file(
            canonical_absolute_path(path, label), label=label, max_bytes=max_bytes,
        )
    except AuthorityBundleIOError as exc:
        raise ValueError(str(exc)) from exc
    return source, _strict_json_bytes(raw, label=label), physical, len(raw)


def _stream_id(value: Any) -> str:
    if not isinstance(value, str) or "@" not in value or value != value.strip():
        raise ValueError("admitted stream IDs must use TICKER@TIMEFRAME")
    ticker, timeframe = value.split("@", 1)
    if (
        not ticker
        or ticker != ticker.upper()
        or not timeframe
        or any(character.isspace() for character in value)
    ):
        raise ValueError("admitted stream IDs must use uppercase TICKER@TIMEFRAME")
    return value


def _validate_source_manifest(document: Mapping[str, Any]) -> None:
    required = {
        "schema_version", "created_utc", "purpose", "source_root",
        "source_snapshot_sha256", "roots", "timeframes_minutes", "resample",
        "roots_report", "outputs",
    }
    _exact_keys(document, required, "source corpus manifest")
    if document["schema_version"] != SOURCE_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported source corpus manifest schema")
    require_sha256(document["source_snapshot_sha256"], "source snapshot SHA-256")
    if not isinstance(document["outputs"], Mapping) or not document["outputs"]:
        raise ValueError("source corpus manifest has no output files")
    for key, raw in document["outputs"].items():
        if not isinstance(key, str) or not key:
            raise ValueError("source corpus output IDs must be non-empty strings")
        entry = _exact_keys(raw, {"path", "bytes", "sha256", "rows"}, f"source output {key}")
        canonical_absolute_path(entry["path"], f"source output {key} path")
        _positive_int(entry["bytes"], f"source output {key} bytes")
        _positive_int(entry["rows"], f"source output {key} rows")
        require_sha256(entry["sha256"], f"source output {key} SHA-256")


def load_tournament_source_authority(
    path: str | Path, *, expected_sha256: str,
) -> VerifiedTournamentSourceAuthority:
    """Load a canonical authority whose physical hash was supplied out-of-band."""
    expected_sha256 = require_sha256(expected_sha256, "source authority SHA-256")
    try:
        source, document, physical = read_canonical_json_file(
            canonical_absolute_path(path, "tournament source authority"),
            label="tournament source authority",
            max_bytes=_MAX_AUTHORITY_BYTES,
            max_nodes=50_000,
            max_depth=16,
        )
    except AuthorityBundleIOError as exc:
        raise ValueError(str(exc)) from exc
    if physical != expected_sha256:
        raise ValueError("tournament source authority physical SHA-256 mismatch")
    _exact_keys(
        document,
        {
            "schema_version", "authority_id", "purpose", "source_manifest",
            "admitted_streams", "cache_construction_admitted", "training_admitted",
            "authority_semantic_sha256",
        },
        "tournament source authority",
    )
    if document["schema_version"] != SOURCE_AUTHORITY_SCHEMA_VERSION:
        raise ValueError("unsupported tournament source authority schema")
    if (
        not isinstance(document["authority_id"], str)
        or not document["authority_id"].strip()
        or document["purpose"] != "tournament_cache_source_admission"
        or document["cache_construction_admitted"] is not True
        or document["training_admitted"] is not False
    ):
        raise ValueError("tournament source authority admission semantics are invalid")
    supplied_semantic = require_sha256(
        document["authority_semantic_sha256"], "source authority semantic SHA-256",
    )
    if content_sha256(document, "authority_semantic_sha256") != supplied_semantic:
        raise ValueError("tournament source authority semantic SHA-256 mismatch")
    raw_streams = document["admitted_streams"]
    if not isinstance(raw_streams, list) or not raw_streams:
        raise ValueError("tournament source authority must admit at least one stream")
    admitted = tuple(_stream_id(value) for value in raw_streams)
    if tuple(sorted(set(admitted))) != admitted:
        raise ValueError("tournament source authority streams must be sorted and unique")

    source_manifest = _exact_keys(
        document["source_manifest"],
        {"path", "sha256", "bytes", "schema_version"},
        "source manifest identity",
    )
    manifest_path = canonical_absolute_path(
        source_manifest["path"], "source manifest path",
    )
    manifest_sha = require_sha256(
        source_manifest["sha256"], "source manifest SHA-256",
    )
    manifest_bytes = _positive_int(source_manifest["bytes"], "source manifest bytes")
    if source_manifest["schema_version"] != SOURCE_MANIFEST_SCHEMA_VERSION:
        raise ValueError("source authority names an unsupported corpus manifest schema")
    reopened_path, manifest, actual_sha, actual_bytes = _read_json_file(
        manifest_path, label="admitted source corpus manifest", max_bytes=_MAX_MANIFEST_BYTES,
    )
    if (
        reopened_path != manifest_path
        or actual_sha != manifest_sha
        or actual_bytes != manifest_bytes
    ):
        raise ValueError("admitted source corpus manifest identity mismatch")
    _validate_source_manifest(manifest)
    for stream in admitted:
        ticker, timeframe = stream.split("@", 1)
        if f"{ticker}_{timeframe}" not in manifest["outputs"]:
            raise ValueError(f"source authority admits missing corpus output: {stream}")

    return VerifiedTournamentSourceAuthority(
        path=source,
        physical_sha256=physical,
        semantic_sha256=supplied_semantic,
        manifest_path=manifest_path,
        manifest_physical_sha256=manifest_sha,
        manifest_bytes=manifest_bytes,
        admitted_streams=admitted,
        document=MappingProxyType(document),
        manifest=MappingProxyType(manifest),
        _token=_CAPABILITY_TOKEN,
    )


def require_tournament_source_authority(
    value: object,
) -> VerifiedTournamentSourceAuthority:
    if (
        type(value) is not VerifiedTournamentSourceAuthority
        or value._token is not _CAPABILITY_TOKEN
    ):
        raise TypeError("a verified tournament source authority is required")
    reopened = load_tournament_source_authority(
        value.path, expected_sha256=value.physical_sha256,
    )
    if reopened != value:
        raise ValueError("tournament source authority changed before use")
    return value


def source_stream_identity(
    authority: VerifiedTournamentSourceAuthority,
    *,
    source_dir: str | Path,
    ticker: str,
    timeframe: str,
) -> dict[str, Any]:
    authority = require_tournament_source_authority(authority)
    source_dir = canonical_absolute_path(source_dir, "tournament source directory")
    expected_manifest = source_dir / "MANIFEST.json"
    if authority.manifest_path != expected_manifest:
        raise ValueError("source authority is not bound to the requested source directory")
    stream = _stream_id(f"{ticker}@{timeframe}")
    if stream not in authority.admitted_streams:
        raise ValueError(f"source authority does not admit stream: {stream}")
    key = f"{ticker}_{timeframe}"
    entry = authority.manifest["outputs"][key]
    expected_path = source_dir / f"{ticker}_{timeframe}.csv"
    declared_path = canonical_absolute_path(entry["path"], f"source output {stream}")
    if declared_path != expected_path:
        raise ValueError(f"source manifest path differs from canonical stream path: {stream}")
    try:
        verified_path, physical = sha256_regular_file(
            declared_path,
            label=f"authorized source stream {stream}",
            expected_size=int(entry["bytes"]),
        )
    except AuthorityBundleIOError as exc:
        raise ValueError(str(exc)) from exc
    if verified_path != declared_path or physical != entry["sha256"]:
        raise ValueError(f"authorized source stream identity mismatch: {stream}")
    return {
        "path": str(declared_path),
        "sha256": str(entry["sha256"]),
        "bytes": int(entry["bytes"]),
        "rows": int(entry["rows"]),
        "source_manifest_output_id": key,
    }


def code_identity(path: str | Path, *, label: str) -> dict[str, Any]:
    source = canonical_absolute_path(path, label)
    try:
        metadata = source.stat(follow_symlinks=False)
        verified, physical = sha256_regular_file(
            source, label=label, expected_size=metadata.st_size,
        )
    except (OSError, AuthorityBundleIOError) as exc:
        raise ValueError(f"cannot verify {label}") from exc
    return {"path": str(verified), "sha256": physical, "bytes": int(metadata.st_size)}


def transformation_receipt(*, tournament_data_path: Path, ssl_data_path: Path) -> dict[str, Any]:
    return {
        "transformation_id": CACHE_TRANSFORMATION_ID,
        "source_columns": [
            "datetime", "open", "high", "low", "close", "volume", "contract_id",
        ],
        "output_arrays": {
            "ohlcv": {"dtype": "float32", "layout": ["open", "high", "low", "close", "volume"]},
            "timestamps": {"dtype": "int64", "unit": "nanoseconds_utc"},
            "contract_id": {"dtype": "unicode", "segmentation": "exact_source_contract_id"},
        },
        "row_order": "strict_timestamp_ascending",
        "duplicates": "rejected",
        "missing_values": "rejected",
        "date_filter": {"lower_inclusive": True, "upper_exclusive": True},
        "code_revision": {
            "tournament_data": code_identity(
                tournament_data_path, label="tournament cache transformation code",
            ),
            "ssl_data": code_identity(
                ssl_data_path, label="tournament source loader code",
            ),
        },
    }


def verify_transformation_receipt(
    receipt: Mapping[str, Any], *, tournament_data_path: Path, ssl_data_path: Path,
) -> None:
    expected = transformation_receipt(
        tournament_data_path=tournament_data_path, ssl_data_path=ssl_data_path,
    )
    if dict(receipt) != expected:
        raise ValueError("tournament cache transformation/code revision mismatch")


def array_content_sha256(value: Any) -> str:
    import numpy as np

    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def canonical_authority_document(value: Mapping[str, Any]) -> bytes:
    """Return canonical bytes for independently publishing an authority document."""
    document = dict(value)
    document["authority_semantic_sha256"] = content_sha256(
        document, "authority_semantic_sha256",
    )
    return canonical_json_bytes(document)


__all__ = [
    "CACHE_TRANSFORMATION_ID",
    "SOURCE_AUTHORITY_SCHEMA_VERSION",
    "SOURCE_MANIFEST_SCHEMA_VERSION",
    "VerifiedTournamentSourceAuthority",
    "array_content_sha256",
    "canonical_authority_document",
    "code_identity",
    "load_tournament_source_authority",
    "require_tournament_source_authority",
    "source_stream_identity",
    "transformation_receipt",
    "verify_transformation_receipt",
]
