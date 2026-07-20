from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime, time, timedelta, timezone
import json
from pathlib import Path
from zoneinfo import TZPATH, ZoneInfo

import numpy as np
import pandas as pd
import pytest

from futures_foundation.execution_economics import load_execution_economics
from futures_foundation.finetune import ssl_data
from futures_foundation.finetune.event_contexts import (
    EventContextConfig,
    load_context_shard,
    materialize_context_stream,
    save_context_shard,
)
from futures_foundation.finetune.path_labels import PathLabelConfig, build_dense_path_labels
from futures_foundation.session_gap import (
    advance_admitted_bars,
    build_session_gap_capability,
    load_session_gap_capability,
    verified_session_edge_mask,
)
from futures_foundation.session_denominator import (
    DENOMINATOR_SCHEMA,
    RULES_SCHEMA,
    SCOPE_PURPOSE,
    SCOPE_SCHEMA,
    SessionDenominatorVerificationError,
    content_sha256,
    load_and_verify_session_denominator,
    load_calendar_rules,
    load_consumer_contract,
    load_denominator_scope,
    session_denominator_document,
    sha256_file,
    verify_session_denominator,
)


SOURCE_ID = "cme_official_fixture"


def _canonical(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    )
    return path


def _tzif_path() -> Path:
    for directory in TZPATH:
        candidate = Path(directory) / "America/Chicago"
        if candidate.is_file() and not candidate.is_symlink():
            return candidate
    raise AssertionError("America/Chicago TZif unavailable")


def _dependencies() -> dict[str, str]:
    tzif = _tzif_path()
    tzdata = tzif.parents[1] / "tzdata.zi"
    return {
        "pandas_market_calendars_version": "fixture",
        "pandas_market_calendars_distribution_sha256": "1" * 64,
        "exchange_calendars_version": "fixture",
        "exchange_calendars_distribution_sha256": "2" * 64,
        "calendar_dependency_closure_sha256": "3" * 64,
        "environment_lock_sha256": "4" * 64,
        "python_implementation": "fixture",
        "python_version": "fixture",
        "python_executable_sha256": "5" * 64,
        "timezone_key": "America/Chicago",
        "tzif_sha256": sha256_file(tzif),
        "tzdata_zi_sha256": sha256_file(tzdata),
    }


def _segment(start_offset: int, start_seconds: int, end_offset: int, end_seconds: int):
    return {
        "start_day_offset": start_offset,
        "start_time_s": start_seconds,
        "end_day_offset": end_offset,
        "end_time_s": end_seconds,
    }


def _rules_document(tmp_path: Path) -> dict:
    source = tmp_path / "sources" / "cme.txt"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"official exchange schedule fixture\n")
    return {
        "schema_version": RULES_SCHEMA,
        "coverage": {"start": "2019-01-01", "end_exclusive": "2021-01-01"},
        "dependencies": _dependencies(),
        "source_artifacts": [
            {
                "source_id": SOURCE_ID,
                "path": "sources/cme.txt",
                "artifact_type": "official_exchange_schedule",
                "stable_identifier": "fixture:cme:energy:v1",
                "size": source.stat().st_size,
                "sha256": sha256_file(source),
            }
        ],
        "products": {
            "energy": {
                "exchange_calendar": "CMEGlobex_Energy",
                "open_weekdays": [0, 1, 2, 3, 4],
                "weekday_source_id": SOURCE_ID,
                "rules": [
                    {
                        "rule_id": "energy_normal_v1",
                        "effective_start": "2019-01-01",
                        "effective_end_exclusive": "2021-01-01",
                        "segments": [_segment(-1, 61_200, 0, 57_600)],
                        "source_id": SOURCE_ID,
                    }
                ],
            }
        },
        "roots": {
            "CL": {
                "product": "energy",
                "effective_start": "2020-01-03",
                "effective_end_exclusive": "2020-01-11",
                "source_id": SOURCE_ID,
            }
        },
        "overrides": [
            {
                "override_id": "fixture_closure_20200109",
                "session_day": "2020-01-09",
                "products": ["energy"],
                "roots": [],
                "status": "closed",
                "segments": [],
                "source_id": SOURCE_ID,
            }
        ],
    }


def _consumer_document() -> dict:
    document = json.loads(Path("config/corpus_v3/contract.json").read_text(encoding="utf-8"))
    document["admitted_roots"] = ["CL"]
    document["blocked_roots"] = {}
    document["splits"] = {
        "foundation_pretraining": {
            "start": "2019-01-01",
            "end_exclusive": "2020-01-01",
            "use": "training_only",
        },
        "supervised_training": {
            "start": "2020-01-01",
            "end_exclusive": "2020-01-13",
            "use": "training_only",
        },
        "development": {
            "start": "2020-01-13",
            "end_exclusive": "2020-01-20",
            "use": "validation_model_selection",
        },
        "legacy_holdout_excluded": {
            "start": "2020-01-20",
            "end_exclusive": "2020-02-01",
            "use": "coverage_report_only_never_training_validation_calibration_or_selection",
            "caveat": "test-only reserved interval",
        },
    }
    return document


def _scope_document(rules_sha: str, consumer_sha: str) -> dict:
    return {
        "schema_version": SCOPE_SCHEMA,
        "consumer_contract_sha256": consumer_sha,
        "purpose": SCOPE_PURPOSE,
        "calendar_rules_sha256": rules_sha,
        "split_uses": ["supervised_training"],
        "roots": ["CL"],
        "start": "2020-01-01",
        "end_exclusive": "2020-01-13",
        "reserved_oos_excluded": True,
    }


def _resolved_regular(day_text: str) -> list[list[int]]:
    day = date.fromisoformat(day_text)
    tz = ZoneInfo("America/Chicago")
    start = datetime.combine(day - timedelta(days=1), time(17), tzinfo=tz)
    end = datetime.combine(day, time(16), tzinfo=tz)
    return [
        [
            int(start.astimezone(timezone.utc).timestamp() * 1_000_000_000),
            int(end.astimezone(timezone.utc).timestamp() * 1_000_000_000),
        ]
    ]


def _row(day_text: str) -> dict:
    day = date.fromisoformat(day_text)
    base = {
        "root": "CL",
        "product": "energy",
        "session_day": day_text,
        "root_source_id": SOURCE_ID,
        "weekday_source_id": SOURCE_ID,
    }
    if day < date(2020, 1, 3) or day >= date(2020, 1, 11):
        segments = []
        return {
            **base,
            "status": "prelisting" if day < date(2020, 1, 3) else "delisted",
            "segments_utc_ns": segments,
            "rule_id": None,
            "rule_source_id": None,
            "override_id": None,
            "override_source_id": None,
            "calendar_observation": None,
            "calendar_exception_types": [],
            "segment_semantic_sha256": content_sha256(segments),
        }
    if day == date(2020, 1, 9):
        segments = []
        return {
            **base,
            "status": "closed",
            "segments_utc_ns": segments,
            "rule_id": "energy_normal_v1",
            "rule_source_id": SOURCE_ID,
            "override_id": "fixture_closure_20200109",
            "override_source_id": SOURCE_ID,
            "calendar_observation": "closed",
            "calendar_exception_types": [],
            "segment_semantic_sha256": content_sha256(segments),
        }
    if day.weekday() >= 5:
        segments = []
        status = "closed"
        observation = "closed"
    else:
        segments = _resolved_regular(day_text)
        status = "regular"
        observation = "open"
    return {
        **base,
        "status": status,
        "segments_utc_ns": segments,
        "rule_id": "energy_normal_v1",
        "rule_source_id": SOURCE_ID,
        "override_id": None,
        "override_source_id": None,
        "calendar_observation": observation,
        "calendar_exception_types": [],
        "segment_semantic_sha256": content_sha256(segments),
    }


def _denominator_document(rules_sha: str, scope_sha: str, dependencies: dict) -> dict:
    rows = [
        _row((date(2020, 1, 1) + timedelta(days=offset)).isoformat())
        for offset in range(12)
    ]
    document = {
        "schema_version": DENOMINATOR_SCHEMA,
        "calendar_rules_sha256": rules_sha,
        "denominator_scope_sha256": scope_sha,
        "dependencies": dependencies,
        "row_count": len(rows),
        "rows": rows,
    }
    document["denominator_semantic_sha256"] = content_sha256(document)
    return document


def _rehash(document: dict) -> None:
    document.pop("denominator_semantic_sha256", None)
    document["denominator_semantic_sha256"] = content_sha256(document)


def _fixture(tmp_path: Path):
    rules_path = _canonical(tmp_path / "rules.json", _rules_document(tmp_path))
    rules = load_calendar_rules(rules_path, expected_sha256=sha256_file(rules_path))
    consumer_path = tmp_path / "consumer.json"
    consumer_path.write_text(
        json.dumps(_consumer_document(), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    consumer = load_consumer_contract(
        consumer_path, expected_sha256=sha256_file(consumer_path)
    )
    scope_path = _canonical(
        tmp_path / "scope.json",
        _scope_document(rules.physical_sha256, consumer.physical_sha256),
    )
    scope = load_denominator_scope(
        scope_path,
        expected_sha256=sha256_file(scope_path),
        rules=rules,
        consumer=consumer,
    )
    denominator = _denominator_document(
        rules.physical_sha256, scope.physical_sha256, rules.document["dependencies"]
    )
    return rules, consumer, scope, denominator


def _session_gap_fixture(tmp_path: Path):
    rules, consumer, scope, document = _fixture(tmp_path)
    denominator_path = _canonical(tmp_path / "denominator.json", document)
    denominator = load_and_verify_session_denominator(
        denominator_path,
        expected_sha256=sha256_file(denominator_path),
        rules=rules,
        scope=scope,
        consumer=consumer,
    )
    capability = build_session_gap_capability(
        denominator,
        rules=rules,
        scope=scope,
        consumer=consumer,
        root="CL",
        expected_delta="1min",
    )
    return capability, denominator_path


def test_verified_session_gap_accepts_only_adjacent_official_segments(tmp_path):
    capability, _ = _session_gap_fixture(tmp_path)
    first_end = capability.segment_ends_ns[0]
    second_start = capability.segment_starts_ns[1]
    times = pd.to_datetime([
        first_end - 2 * 60_000_000_000,
        first_end - 60_000_000_000,
        second_start,
        second_start + 60_000_000_000,
    ], utc=True)
    assert verified_session_edge_mask(
        times, expected_delta="1min", capability=capability,
    ).tolist() == [True, True, True]
    starts = ssl_data.window_starts(
        np.arange(4), 4, timestamps=times, expected_delta="1min",
        session_gap_capability=capability,
    )
    assert starts.tolist() == [0]


def test_session_gap_manifest_reopens_and_bar_clock_advances_across_closure(tmp_path):
    capability, _ = _session_gap_fixture(tmp_path)
    reopened = load_session_gap_capability(capability.manifest())
    assert reopened == capability
    delta = capability.expected_delta_ns
    first = capability.segment_ends_ns[0] - 2 * delta
    advanced = advance_admitted_bars(
        np.asarray([first, first, first]),
        np.asarray([0, 1, 2]),
        capability=capability,
    )
    assert advanced.tolist() == [
        first,
        capability.segment_ends_ns[0] - delta,
        capability.segment_starts_ns[1],
    ]
    with pytest.raises(ValueError, match="not an admitted bar open"):
        advance_admitted_bars(
            np.asarray([capability.segment_ends_ns[0]]),
            np.asarray([1]),
            capability=capability,
        )


def test_dense_path_labels_use_verified_bar_clock_across_official_closure(tmp_path):
    capability, _ = _session_gap_fixture(tmp_path)
    delta = capability.expected_delta_ns
    timestamps = pd.to_datetime([
        capability.segment_ends_ns[0] - 2 * delta,
        capability.segment_ends_ns[0] - delta,
        capability.segment_starts_ns[1],
        capability.segment_starts_ns[1] + delta,
        capability.segment_starts_ns[1] + 2 * delta,
    ], utc=True)
    close = np.asarray([70.0, 70.2, 70.4, 70.6, 70.8])
    frame = pd.DataFrame({
        "datetime": timestamps,
        "open": close,
        "high": close + 0.1,
        "low": close - 0.1,
        "close": close,
        "volume": np.ones(len(close)),
        "contract_id": np.full(len(close), "CLG20"),
    })
    config = PathLabelConfig(
        horizons_minutes=(2,), targets_r=(1.0,), atr_period=1,
        context_minutes=1,
    )
    exact = build_dense_path_labels(
        frame, timeframe_minutes=1, config=config,
    )
    verified = build_dense_path_labels(
        frame, timeframe_minutes=1, config=config,
        session_gap_capability=capability,
    )
    assert exact["valid"][0, 0] is np.False_
    assert bool(verified["valid"][0, 0])
    assert verified["label_end_time_ns"][0, 0] == capability.segment_starts_ns[1]
    assert verified["target_semantics"]["horizon_basis"] == (
        "admitted_bar_minutes_across_verified_sessions_v1"
    )
    assert verified["session_gap_capability"] == capability.manifest()


def test_event_shard_materialization_uses_and_reverifies_session_capability(tmp_path):
    capability, _ = _session_gap_fixture(tmp_path)
    delta = capability.expected_delta_ns
    first = np.arange(
        capability.segment_ends_ns[0] - 220 * delta,
        capability.segment_ends_ns[0],
        delta,
        dtype=np.int64,
    )
    second = np.arange(
        capability.segment_starts_ns[1],
        capability.segment_starts_ns[1] + 400 * delta,
        delta,
        dtype=np.int64,
    )
    timestamps = np.r_[first, second]
    close = 70.0 + np.arange(len(timestamps), dtype=np.float64) * 0.01
    frame = pd.DataFrame({
        "datetime": pd.to_datetime(timestamps, utc=True),
        "open": close,
        "high": close + 0.05,
        "low": close - 0.05,
        "close": close,
        "volume": np.full(len(close), 100.0),
        "contract_id": np.full(len(close), "CLG20"),
        "source_row_idx": np.arange(len(close), dtype=np.int64),
    })
    start = pd.Timestamp(capability.segment_starts_ns[1], unit="ns", tz="UTC")
    end = start + pd.Timedelta(minutes=400)
    config = EventContextConfig(
        eval_start=start.isoformat(),
        eval_end=end.isoformat(),
        context_bars=256,
        atr_period=20,
        path=PathLabelConfig(
            horizons_minutes=(60,), targets_r=(1.0,), atr_period=20,
            context_minutes=60,
        ),
    )
    economics = load_execution_economics(
        Path(__file__).resolve().parents[1] / "config/execution_costs.yaml",
        evaluation_start=start.isoformat(),
        evaluation_end=end.isoformat(),
        required_roots=("CL",),
    )
    exact_arrays, _ = materialize_context_stream(
        frame,
        ticker="CL",
        timeframe="1min",
        execution_economics=economics,
        config=config,
    )
    arrays, metadata = materialize_context_stream(
        frame,
        ticker="CL",
        timeframe="1min",
        execution_economics=economics,
        config=config,
        session_gap_capability=capability,
    )
    assert arrays["decision_time_ns"].min() < exact_arrays["decision_time_ns"].min()
    assert metadata["session_gap_capability"] == capability.manifest()
    assert metadata["context_gap_policy"] == "verified_session_edges_and_contract_segments_v1"
    path = tmp_path / "CL_1min.npz"
    manifest = save_context_shard(path, arrays, metadata)
    loaded, reopened = load_context_shard(path)
    assert reopened["content_fingerprint"] == manifest["content_fingerprint"]
    np.testing.assert_array_equal(loaded["label_end_time_ns"], arrays["label_end_time_ns"])


def test_verified_session_gap_rejects_missing_bars_and_skipped_open_session(tmp_path):
    capability, _ = _session_gap_fixture(tmp_path)
    first_start = capability.segment_starts_ns[0]
    missing_inside = pd.to_datetime([
        first_start, first_start + 2 * 60_000_000_000,
    ], utc=True)
    assert not verified_session_edge_mask(
        missing_inside, expected_delta="1min", capability=capability,
    )[0]

    skipped_open = pd.to_datetime([
        capability.segment_ends_ns[0] - 60_000_000_000,
        capability.segment_starts_ns[2],
    ], utc=True)
    assert not verified_session_edge_mask(
        skipped_open, expected_delta="1min", capability=capability,
    )[0]


def test_verified_session_gap_rejects_wrong_bar_size_and_post_load_mutation(tmp_path):
    capability, denominator_path = _session_gap_fixture(tmp_path)
    times = pd.to_datetime([
        capability.segment_starts_ns[0],
        capability.segment_starts_ns[0] + 60_000_000_000,
    ], utc=True)
    with pytest.raises(ValueError, match="bar size"):
        verified_session_edge_mask(
            times, expected_delta="5min", capability=capability,
        )
    denominator_path.write_bytes(denominator_path.read_bytes() + b"\n")
    with pytest.raises(SessionDenominatorVerificationError, match="bytes changed"):
        verified_session_edge_mask(
            times, expected_delta="1min", capability=capability,
        )


def test_independent_verifier_accepts_complete_non_oos_denominator(tmp_path):
    rules, consumer, scope, document = _fixture(tmp_path)
    denominator_path = _canonical(tmp_path / "denominator.json", document)
    verified = load_and_verify_session_denominator(
        denominator_path,
        expected_sha256=sha256_file(denominator_path),
        rules=rules,
        scope=scope,
        consumer=consumer,
    )
    assert verified.semantic_sha256 == document["denominator_semantic_sha256"]
    reopened = session_denominator_document(
        verified, rules=rules, scope=scope, consumer=consumer
    )
    assert len(reopened["rows"]) == 12
    assert reopened["rows"][0]["status"] == "prelisting"
    assert reopened["rows"][-1]["status"] == "delisted"


def test_verified_denominator_capability_rejects_post_load_tamper(tmp_path):
    rules, consumer, scope, document = _fixture(tmp_path)
    denominator_path = _canonical(tmp_path / "denominator.json", document)
    verified = load_and_verify_session_denominator(
        denominator_path,
        expected_sha256=sha256_file(denominator_path),
        rules=rules,
        scope=scope,
        consumer=consumer,
    )
    denominator_path.write_bytes(denominator_path.read_bytes() + b"\n")
    with pytest.raises(SessionDenominatorVerificationError, match="bytes changed"):
        session_denominator_document(
            verified, rules=rules, scope=scope, consumer=consumer
        )


def test_omitted_date_is_rejected_even_after_self_hash_repair(tmp_path):
    rules, consumer, scope, document = _fixture(tmp_path)
    document["rows"].pop(5)
    document["row_count"] -= 1
    _rehash(document)
    with pytest.raises(SessionDenominatorVerificationError, match="exactly cover"):
        verify_session_denominator(document, rules=rules, scope=scope, consumer=consumer)


def test_reserved_oos_split_cannot_enter_scope(tmp_path):
    rules, consumer, _, _ = _fixture(tmp_path)
    scope_document = _scope_document(rules.physical_sha256, consumer.physical_sha256)
    scope_document.update(
        {
            "split_uses": ["legacy_holdout_excluded"],
            "start": "2020-01-20",
            "end_exclusive": "2020-02-01",
        }
    )
    scope_path = _canonical(tmp_path / "oos_scope.json", scope_document)
    with pytest.raises(SessionDenominatorVerificationError, match="split uses"):
        load_denominator_scope(
            scope_path,
            expected_sha256=sha256_file(scope_path),
            rules=rules,
            consumer=consumer,
        )


def test_official_source_byte_tamper_invalidates_loaded_capability(tmp_path):
    rules, consumer, scope, document = _fixture(tmp_path)
    (tmp_path / "sources" / "cme.txt").write_bytes(b"tampered schedule\n")
    with pytest.raises(SessionDenominatorVerificationError, match="source artifact bytes differ"):
        verify_session_denominator(document, rules=rules, scope=scope, consumer=consumer)


def test_wrong_root_effective_boundary_is_rejected(tmp_path):
    rules, consumer, scope, document = _fixture(tmp_path)
    row = document["rows"][1]
    row.update(
        {
            "status": "regular",
            "segments_utc_ns": _resolved_regular("2020-01-02"),
            "rule_id": "energy_normal_v1",
            "rule_source_id": SOURCE_ID,
            "calendar_observation": "open",
        }
    )
    row["segment_semantic_sha256"] = content_sha256(row["segments_utc_ns"])
    _rehash(document)
    with pytest.raises(SessionDenominatorVerificationError, match="inactive-root|official rules"):
        verify_session_denominator(document, rules=rules, scope=scope, consumer=consumer)


def test_missing_holiday_override_is_rejected(tmp_path):
    rules, consumer, scope, document = _fixture(tmp_path)
    row = document["rows"][8]
    row.update(
        {
            "status": "regular",
            "segments_utc_ns": _resolved_regular("2020-01-09"),
            "override_id": None,
            "override_source_id": None,
        }
    )
    row["segment_semantic_sha256"] = content_sha256(row["segments_utc_ns"])
    _rehash(document)
    with pytest.raises(SessionDenominatorVerificationError, match="official rules"):
        verify_session_denominator(document, rules=rules, scope=scope, consumer=consumer)


def test_altered_utc_segments_are_rejected_after_rehash(tmp_path):
    rules, consumer, scope, document = _fixture(tmp_path)
    row = document["rows"][7]
    row["segments_utc_ns"][0][1] -= 60_000_000_000
    row["segment_semantic_sha256"] = content_sha256(row["segments_utc_ns"])
    _rehash(document)
    with pytest.raises(SessionDenominatorVerificationError, match="official rules"):
        verify_session_denominator(document, rules=rules, scope=scope, consumer=consumer)


@pytest.mark.parametrize(
    ("row_index", "observation", "segments", "status", "message"),
    [
        (9, "closed", None, "regular", "weekday closure"),
        (3, "open", "regular", "regular", "closed-weekday opening"),
    ],
)
def test_unsourced_weekday_open_or_closed_exception_is_rejected(
    tmp_path, row_index, observation, segments, status, message
):
    rules, consumer, scope, document = _fixture(tmp_path)
    row = document["rows"][row_index]
    row["calendar_observation"] = observation
    row["status"] = status
    if segments == "regular":
        row["segments_utc_ns"] = _resolved_regular(row["session_day"])
        row["segment_semantic_sha256"] = content_sha256(row["segments_utc_ns"])
    _rehash(document)
    with pytest.raises(SessionDenominatorVerificationError, match=message):
        verify_session_denominator(document, rules=rules, scope=scope, consumer=consumer)


def test_calendar_observation_does_not_override_sourced_geometry(tmp_path):
    rules, consumer, scope, document = _fixture(tmp_path)
    # The diagnostic calendar may disagree, but a source-backed closed override still
    # determines an empty session.  It cannot be used to synthesize open geometry.
    document["rows"][8]["calendar_observation"] = "open"
    _rehash(document)
    verify_session_denominator(document, rules=rules, scope=scope, consumer=consumer)


def test_stale_open_override_matching_normal_session_is_rejected(tmp_path):
    rules_document = _rules_document(tmp_path)
    rules_document["overrides"].append(
        {
            "override_id": "stale_open_20200107",
            "session_day": "2020-01-07",
            "products": ["energy"],
            "roots": [],
            "status": "open",
            "segments": [_segment(-1, 61_200, 0, 57_600)],
            "source_id": SOURCE_ID,
        }
    )
    rules_path = _canonical(tmp_path / "rules-stale.json", rules_document)
    rules = load_calendar_rules(rules_path, expected_sha256=sha256_file(rules_path))
    consumer_path = tmp_path / "consumer-stale.json"
    consumer_path.write_text(
        json.dumps(_consumer_document(), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    consumer = load_consumer_contract(
        consumer_path, expected_sha256=sha256_file(consumer_path)
    )
    scope_path = _canonical(
        tmp_path / "scope-stale.json",
        _scope_document(rules.physical_sha256, consumer.physical_sha256),
    )
    scope = load_denominator_scope(
        scope_path,
        expected_sha256=sha256_file(scope_path),
        rules=rules,
        consumer=consumer,
    )
    document = _denominator_document(
        rules.physical_sha256, scope.physical_sha256, rules.document["dependencies"]
    )
    row = document["rows"][6]
    assert row["session_day"] == "2020-01-07"
    row["override_id"] = "stale_open_20200107"
    row["override_source_id"] = SOURCE_ID
    _rehash(document)
    with pytest.raises(SessionDenominatorVerificationError, match="stale open override"):
        verify_session_denominator(document, rules=rules, scope=scope, consumer=consumer)


def test_empty_dependency_provenance_string_is_rejected(tmp_path):
    rules_document = _rules_document(tmp_path)
    rules_document["dependencies"]["python_version"] = ""
    path = _canonical(tmp_path / "empty-provenance.json", rules_document)
    with pytest.raises(SessionDenominatorVerificationError, match="nonempty string"):
        load_calendar_rules(path, expected_sha256=sha256_file(path))


def test_physical_hash_and_canonical_json_are_both_required(tmp_path):
    rules_document = _rules_document(tmp_path)
    rules_path = _canonical(tmp_path / "rules.json", rules_document)
    with pytest.raises(SessionDenominatorVerificationError, match="physical SHA"):
        load_calendar_rules(rules_path, expected_sha256="f" * 64)

    pretty_path = tmp_path / "pretty.json"
    pretty_path.write_text(json.dumps(rules_document, indent=2), encoding="utf-8")
    with pytest.raises(SessionDenominatorVerificationError, match="canonical JSON"):
        load_calendar_rules(pretty_path, expected_sha256=sha256_file(pretty_path))


def test_actual_pretty_checked_in_consumer_contract_is_accepted():
    path = Path(__file__).resolve().parents[1] / "config" / "corpus_v3" / "contract.json"
    verified = load_consumer_contract(path, expected_sha256=sha256_file(path))
    assert verified.document["schema_version"] == "ffm_corpus_v3_contract_v1"
    assert verified.physical_sha256 != verified.semantic_sha256


def test_consumer_contract_rejects_duplicate_keys_and_nonfinite_values(tmp_path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"schema_version":"ffm_corpus_v3_contract_v1","schema_version":"duplicate"}',
        encoding="utf-8",
    )
    with pytest.raises(SessionDenominatorVerificationError, match="duplicate JSON key"):
        load_consumer_contract(duplicate, expected_sha256=sha256_file(duplicate))

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"schema_version":NaN}', encoding="utf-8")
    with pytest.raises(SessionDenominatorVerificationError, match="non-finite JSON"):
        load_consumer_contract(nonfinite, expected_sha256=sha256_file(nonfinite))


def test_consumer_physical_byte_tamper_after_load_is_rejected(tmp_path):
    rules, consumer, scope, document = _fixture(tmp_path)
    consumer.path.write_bytes(consumer.path.read_bytes() + b"\n")
    with pytest.raises(SessionDenominatorVerificationError, match="consumer contract changed"):
        verify_session_denominator(document, rules=rules, scope=scope, consumer=consumer)


def test_in_memory_or_on_disk_contract_mutation_is_rejected(tmp_path):
    rules, consumer, scope, document = _fixture(tmp_path)
    mutable = consumer.document
    assert isinstance(mutable, dict)
    mutable["admitted_roots"] = []
    with pytest.raises(SessionDenominatorVerificationError, match="consumer contract changed"):
        verify_session_denominator(document, rules=rules, scope=scope, consumer=consumer)


def test_final_file_symlinks_are_rejected_for_every_trust_artifact(tmp_path):
    rules, consumer, scope, document = _fixture(tmp_path)
    denominator_path = _canonical(tmp_path / "denominator.json", document)

    rules_link = tmp_path / "rules-link.json"
    rules_link.symlink_to(rules.path)
    with pytest.raises(SessionDenominatorVerificationError, match="symlink"):
        load_calendar_rules(rules_link, expected_sha256=rules.physical_sha256)

    consumer_link = tmp_path / "consumer-link.json"
    consumer_link.symlink_to(consumer.path)
    with pytest.raises(SessionDenominatorVerificationError, match="symlink"):
        load_consumer_contract(consumer_link, expected_sha256=consumer.physical_sha256)

    scope_link = tmp_path / "scope-link.json"
    scope_link.symlink_to(scope.path)
    with pytest.raises(SessionDenominatorVerificationError, match="symlink"):
        load_denominator_scope(
            scope_link,
            expected_sha256=scope.physical_sha256,
            rules=rules,
            consumer=consumer,
        )

    denominator_link = tmp_path / "denominator-link.json"
    denominator_link.symlink_to(denominator_path)
    with pytest.raises(SessionDenominatorVerificationError, match="symlink"):
        load_and_verify_session_denominator(
            denominator_link,
            expected_sha256=sha256_file(denominator_path),
            rules=rules,
            scope=scope,
            consumer=consumer,
        )


def test_parent_directory_symlink_is_rejected(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    rules_path = _canonical(real / "rules.json", _rules_document(real))
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(SessionDenominatorVerificationError, match="symlink"):
        load_calendar_rules(alias / "rules.json", expected_sha256=sha256_file(rules_path))
