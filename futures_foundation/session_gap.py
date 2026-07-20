"""Verified exchange-session continuity for model window construction.

A session gap is continuity only when an independently verified denominator says
that the left bar is the final bar of one admitted segment and the right bar is the
first bar of the immediately following admitted segment.  Every other non-cadence
edge remains a hard boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from futures_foundation._authority_bundle_io import (
    read_canonical_json_file,
    require_sha256,
)
from futures_foundation.session_denominator import (
    VerifiedCalendarRules,
    VerifiedConsumerContract,
    VerifiedDenominatorScope,
    VerifiedSessionDenominator,
    load_and_verify_session_denominator,
    load_calendar_rules,
    load_consumer_contract,
    load_denominator_scope,
    session_denominator_document,
)


SESSION_GAP_SCHEMA_VERSION = "ffm_verified_session_gap_capability_v1"
SESSION_GAP_SET_SCHEMA_VERSION = "ffm_verified_session_gap_capability_set_v1"
_MAX_CAPABILITY_SET_BYTES = 32 * 1024 * 1024
_CAPABILITY_TOKEN = object()


@dataclass(frozen=True)
class VerifiedSessionGapCapability:
    root: str
    expected_delta_ns: int
    segment_starts_ns: tuple[int, ...]
    segment_ends_ns: tuple[int, ...]
    denominator: VerifiedSessionDenominator
    rules: VerifiedCalendarRules
    scope: VerifiedDenominatorScope
    consumer: VerifiedConsumerContract
    denominator_semantic_sha256: str
    segments_sha256: str
    _token: object

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": SESSION_GAP_SCHEMA_VERSION,
            "root": self.root,
            "expected_delta_ns": self.expected_delta_ns,
            "segment_count": len(self.segment_starts_ns),
            "segments_sha256": self.segments_sha256,
            "denominator": {
                "path": str(self.denominator.path),
                "physical_sha256": self.denominator.physical_sha256,
                "semantic_sha256": self.denominator_semantic_sha256,
            },
            "calendar_rules": {
                "path": str(self.rules.path),
                "physical_sha256": self.rules.physical_sha256,
            },
            "denominator_scope": {
                "path": str(self.scope.path),
                "physical_sha256": self.scope.physical_sha256,
            },
            "consumer_contract": {
                "path": str(self.consumer.path),
                "physical_sha256": self.consumer.physical_sha256,
                "semantic_sha256": self.consumer.semantic_sha256,
            },
            "session_continuity_admitted": True,
            "training_admitted": False,
        }


def _delta_ns(value: Any) -> int:
    try:
        delta = int(pd.Timedelta(value).value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("expected_delta must be a positive pandas-compatible duration") from exc
    if delta <= 0:
        raise ValueError("expected_delta must be positive")
    return delta


def _segments_from_document(document: dict[str, Any], root: str, delta_ns: int) -> tuple[tuple[int, ...], tuple[int, ...], str]:
    rows = document.get("rows")
    if not isinstance(rows, list):
        raise ValueError("session denominator rows are unavailable")
    segments: list[tuple[int, int]] = []
    found_root = False
    for row in rows:
        if not isinstance(row, dict) or row.get("root") != root:
            continue
        found_root = True
        raw_segments = row.get("segments_utc_ns")
        if not isinstance(raw_segments, list):
            raise ValueError("session denominator segment rows are malformed")
        for raw in raw_segments:
            if (
                not isinstance(raw, list)
                or len(raw) != 2
                or type(raw[0]) is not int
                or type(raw[1]) is not int
            ):
                raise ValueError("session denominator segment geometry is malformed")
            start, end = int(raw[0]), int(raw[1])
            if start >= end or end - start < delta_ns or (end - start) % delta_ns:
                raise ValueError("session denominator segment is not aligned to the requested bar size")
            segments.append((start, end))
    if not found_root:
        raise ValueError(f"session denominator does not contain root {root}")
    if not segments:
        raise ValueError(f"session denominator contains no open segments for root {root}")
    segments.sort()
    for previous, current in zip(segments, segments[1:]):
        if current[0] < previous[1]:
            raise ValueError("session denominator segments overlap or regress")
    starts = tuple(start for start, _ in segments)
    ends = tuple(end for _, end in segments)
    packed = np.asarray(segments, dtype=np.int64)
    digest = hashlib.sha256()
    digest.update(str(packed.dtype).encode("ascii"))
    digest.update(np.asarray(packed.shape, np.int64).tobytes())
    digest.update(packed.tobytes())
    return starts, ends, digest.hexdigest()


def build_session_gap_capability(
    denominator: VerifiedSessionDenominator,
    *,
    rules: VerifiedCalendarRules,
    scope: VerifiedDenominatorScope,
    consumer: VerifiedConsumerContract,
    root: str,
    expected_delta: Any,
) -> VerifiedSessionGapCapability:
    root = str(root).strip().upper()
    if not root:
        raise ValueError("session-gap root must be non-empty uppercase text")
    delta_ns = _delta_ns(expected_delta)
    document = session_denominator_document(
        denominator, rules=rules, scope=scope, consumer=consumer,
    )
    starts, ends, semantic = _segments_from_document(document, root, delta_ns)
    return VerifiedSessionGapCapability(
        root=root,
        expected_delta_ns=delta_ns,
        segment_starts_ns=starts,
        segment_ends_ns=ends,
        denominator=denominator,
        rules=rules,
        scope=scope,
        consumer=consumer,
        denominator_semantic_sha256=denominator.semantic_sha256,
        segments_sha256=semantic,
        _token=_CAPABILITY_TOKEN,
    )


def require_session_gap_capability(
    value: object,
) -> VerifiedSessionGapCapability:
    if (
        type(value) is not VerifiedSessionGapCapability
        or value._token is not _CAPABILITY_TOKEN
    ):
        raise TypeError("a verified session-gap capability is required")
    reopened = build_session_gap_capability(
        value.denominator,
        rules=value.rules,
        scope=value.scope,
        consumer=value.consumer,
        root=value.root,
        expected_delta=pd.Timedelta(value.expected_delta_ns, unit="ns"),
    )
    if reopened != value:
        raise ValueError("verified session-gap capability changed before use")
    return value


def load_session_gap_capability(
    manifest: Mapping[str, Any],
) -> VerifiedSessionGapCapability:
    """Reopen and reverify a serialized session-gap capability.

    The manifest is not a bearer token.  Every parent artifact is reopened through
    the denominator verifier and the complete reconstructed capability must match
    the serialized document byte-for-semantics.
    """
    if not isinstance(manifest, Mapping):
        raise TypeError("session-gap capability manifest must be a mapping")
    expected_fields = {
        "schema_version", "root", "expected_delta_ns", "segment_count",
        "segments_sha256", "denominator", "calendar_rules", "denominator_scope",
        "consumer_contract", "session_continuity_admitted", "training_admitted",
    }
    if set(manifest) != expected_fields:
        raise ValueError("session-gap capability manifest fields mismatch")
    if (
        manifest.get("schema_version") != SESSION_GAP_SCHEMA_VERSION
        or manifest.get("session_continuity_admitted") is not True
        or manifest.get("training_admitted") is not False
    ):
        raise ValueError("session-gap capability admission semantics are invalid")

    def parent(name: str, fields: set[str]) -> Mapping[str, Any]:
        value = manifest.get(name)
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError(f"session-gap capability {name} identity is malformed")
        return value

    rules_ref = parent("calendar_rules", {"path", "physical_sha256"})
    consumer_ref = parent(
        "consumer_contract", {"path", "physical_sha256", "semantic_sha256"},
    )
    scope_ref = parent("denominator_scope", {"path", "physical_sha256"})
    denominator_ref = parent(
        "denominator", {"path", "physical_sha256", "semantic_sha256"},
    )
    rules = load_calendar_rules(
        str(rules_ref["path"]), expected_sha256=str(rules_ref["physical_sha256"]),
    )
    consumer = load_consumer_contract(
        str(consumer_ref["path"]),
        expected_sha256=str(consumer_ref["physical_sha256"]),
    )
    if consumer.semantic_sha256 != consumer_ref["semantic_sha256"]:
        raise ValueError("session-gap consumer semantic identity changed")
    scope = load_denominator_scope(
        str(scope_ref["path"]),
        expected_sha256=str(scope_ref["physical_sha256"]),
        rules=rules,
        consumer=consumer,
    )
    denominator = load_and_verify_session_denominator(
        str(denominator_ref["path"]),
        expected_sha256=str(denominator_ref["physical_sha256"]),
        rules=rules,
        scope=scope,
        consumer=consumer,
    )
    if denominator.semantic_sha256 != denominator_ref["semantic_sha256"]:
        raise ValueError("session-gap denominator semantic identity changed")
    capability = build_session_gap_capability(
        denominator,
        rules=rules,
        scope=scope,
        consumer=consumer,
        root=str(manifest["root"]),
        expected_delta=pd.Timedelta(int(manifest["expected_delta_ns"]), unit="ns"),
    )
    if capability.manifest() != dict(manifest):
        raise ValueError("session-gap capability manifest differs from re-verification")
    return capability


def load_session_gap_capability_set(
    path: str | Path,
    *,
    expected_sha256: str,
) -> dict[str, VerifiedSessionGapCapability]:
    """Load an externally hash-bound set of per-stream capabilities."""
    expected_sha256 = require_sha256(
        expected_sha256, "session-gap capability-set SHA-256",
    )
    document, _, _ = read_canonical_json_file(
        path,
        label="session-gap capability set",
        max_bytes=_MAX_CAPABILITY_SET_BYTES,
        expected_sha256=expected_sha256,
    )
    if not isinstance(document, Mapping) or set(document) != {
        "schema_version", "purpose", "capabilities", "training_admitted",
    }:
        raise ValueError("session-gap capability-set fields mismatch")
    if (
        document["schema_version"] != SESSION_GAP_SET_SCHEMA_VERSION
        or document["purpose"] != "verified_session_continuity_for_governed_windows"
        or document["training_admitted"] is not False
    ):
        raise ValueError("session-gap capability-set semantics are invalid")
    raw = document["capabilities"]
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("session-gap capability set must contain capabilities")
    capabilities: dict[str, VerifiedSessionGapCapability] = {}
    minute_ns = 60 * 1_000_000_000
    for stream_id, manifest in sorted(raw.items()):
        if not isinstance(stream_id, str) or stream_id.count("@") != 1:
            raise ValueError("session-gap capability-set stream identity is malformed")
        root, timeframe = stream_id.split("@", 1)
        capability = load_session_gap_capability(manifest)
        if capability.root != root:
            raise ValueError("session-gap capability-set root identity mismatch")
        if capability.expected_delta_ns % minute_ns:
            raise ValueError("session-gap capability set supports integer-minute bars only")
        expected_timeframe = f"{capability.expected_delta_ns // minute_ns}min"
        if timeframe != expected_timeframe:
            raise ValueError("session-gap capability-set timeframe identity mismatch")
        capabilities[stream_id] = capability
    return capabilities


def advance_admitted_bars(
    timestamps_ns: Any,
    bars: Any,
    *,
    capability: VerifiedSessionGapCapability,
) -> np.ndarray:
    """Advance legal bar-open timestamps across verified session boundaries.

    ``bars`` counts admitted market bars, not elapsed wall-clock minutes.  The
    function is vectorized and fails if a starting timestamp is outside the
    verified session geometry or the target lies beyond the denominator scope.
    """
    capability = require_session_gap_capability(capability)
    timestamps = np.asarray(timestamps_ns)
    offsets = np.asarray(bars)
    try:
        timestamps, offsets = np.broadcast_arrays(timestamps, offsets)
    except ValueError as exc:
        raise ValueError("timestamps and bar offsets are not broadcast-compatible") from exc
    if timestamps.dtype.kind not in "iu" or offsets.dtype.kind not in "iu":
        raise TypeError("timestamps and bar offsets must be integer arrays")
    timestamps = timestamps.astype(np.int64, copy=False)
    offsets = offsets.astype(np.int64, copy=False)
    if np.any(offsets < 0):
        raise ValueError("bar offsets must be nonnegative")

    starts = np.asarray(capability.segment_starts_ns, dtype=np.int64)
    ends = np.asarray(capability.segment_ends_ns, dtype=np.int64)
    delta = int(capability.expected_delta_ns)
    counts = (ends - starts) // delta
    cumulative_ends = np.cumsum(counts, dtype=np.int64)
    cumulative_starts = np.r_[np.int64(0), cumulative_ends[:-1]]

    segment = np.searchsorted(starts, timestamps, side="right") - 1
    safe_segment = np.clip(segment, 0, len(starts) - 1)
    relative = timestamps - starts[safe_segment]
    legal = (
        (segment >= 0)
        & (segment < len(starts))
        & (relative >= 0)
        & (relative % delta == 0)
        & (timestamps + delta <= ends[safe_segment])
    )
    if not bool(np.all(legal)):
        raise ValueError("timestamp is not an admitted bar open in the verified denominator")
    ordinal = cumulative_starts[safe_segment] + relative // delta
    target = ordinal + offsets
    if np.any(target >= cumulative_ends[-1]):
        raise ValueError("bar advance escapes the verified denominator scope")
    target_segment = np.searchsorted(cumulative_ends, target, side="right")
    result = starts[target_segment] + (
        target - cumulative_starts[target_segment]
    ) * delta
    return result.astype(np.int64, copy=False)


def verified_session_edge_mask(
    timestamps: Any,
    *,
    expected_delta: Any,
    capability: VerifiedSessionGapCapability,
) -> np.ndarray:
    """Return validity for every adjacent timestamp edge.

    Same-segment edges require exact bar cadence.  Cross-segment edges require the
    final possible bar open of one official segment followed by the exact start of
    the immediately following official segment.  Skipping any admitted open segment
    or any bar inside a segment therefore fails closed.
    """
    capability = require_session_gap_capability(capability)
    delta_ns = _delta_ns(expected_delta)
    if delta_ns != capability.expected_delta_ns:
        raise ValueError("session-gap capability bar size differs from expected_delta")
    parsed = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True))
    values = parsed.asi8
    if np.any(values == np.iinfo(np.int64).min):
        raise ValueError("session-gap timestamps contain NaT")
    if len(values) < 2:
        return np.zeros(0, dtype=bool)
    if np.any(np.diff(values) <= 0):
        raise ValueError("session-gap timestamps must be strictly increasing")

    starts = np.asarray(capability.segment_starts_ns, np.int64)
    ends = np.asarray(capability.segment_ends_ns, np.int64)
    segment = np.searchsorted(starts, values, side="right") - 1
    inside = (
        (segment >= 0)
        & (segment < len(starts))
    )
    bounded_segment = np.clip(segment, 0, len(starts) - 1)
    inside &= values >= starts[bounded_segment]
    inside &= values + delta_ns <= ends[bounded_segment]

    left_segment = segment[:-1]
    right_segment = segment[1:]
    left_inside = inside[:-1]
    right_inside = inside[1:]
    same = (
        left_inside
        & right_inside
        & (left_segment == right_segment)
        & (values[1:] - values[:-1] == delta_ns)
    )
    adjacent = right_segment == left_segment + 1
    safe_left = np.clip(left_segment, 0, len(starts) - 1)
    safe_right = np.clip(right_segment, 0, len(starts) - 1)
    cross = (
        left_inside
        & right_inside
        & adjacent
        & (values[:-1] + delta_ns == ends[safe_left])
        & (values[1:] == starts[safe_right])
    )
    return same | cross


__all__ = [
    "SESSION_GAP_SCHEMA_VERSION",
    "SESSION_GAP_SET_SCHEMA_VERSION",
    "VerifiedSessionGapCapability",
    "build_session_gap_capability",
    "load_session_gap_capability",
    "load_session_gap_capability_set",
    "require_session_gap_capability",
    "advance_admitted_bars",
    "verified_session_edge_mask",
]
