"""Fail-closed Corpus v3 contract and outcome-blind coverage audit.

This module intentionally reads only the pinned coverage inventory.  It does not open market
Parquet files, construct strategy events, calculate labels, or inspect holdout outcomes.  Its job
is to decide whether a root has enough admitted source coverage to enter later materialization.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date
import gzip
import hashlib
import json
from pathlib import Path
import re
from statistics import median
from typing import Any, Mapping


SCHEMA_VERSION = "ffm_corpus_v3_contract_v1"
AUDIT_SCHEMA_VERSION = "ffm_corpus_v3_coverage_audit_v1"
CONTRACT_RE = re.compile(r"^(.+)([FGHJKMNQUVXZ])(\d{2})$")
REQUIRED_EXPORT_ROWS = {
    "timestamp_utc_ns", "time_us", "event_seq", "price", "bid", "ask", "quote_valid",
    "volume", "bid_volume", "ask_volume", "source_file_index", "source_row_ordinal",
}
REQUIRED_SHARD_METADATA = {
    "root", "contract_id", "session_day", "session_start_utc_ns", "session_end_utc_ns",
    "coverage_start_utc_ns", "coverage_end_utc_ns", "export_receipt_sha256",
    "source_shard_sha256", "source_file_table_sha256", "corpus_contract_sha256",
    "environment_receipt_sha256", "instrument_spec_sha256", "tick_size", "tick_value",
}
REQUIRED_RECEIPT_BINDINGS = {
    "schema_version", "status", "purpose", "request", "request_sha256", "roots",
    "date_range", "window_contract", "output_format", "output_file",
    "output_shard_sha256", "semantic_shard_sha256", "output_schema_sha256",
    "semantic_metadata", "producer_source_manifest", "producer_source_manifest_sha256",
    "environment", "environment_receipt_sha256", "instrument_spec",
    "instrument_spec_sha256", "governance", "governance_sha256",
    "lake_hash_of_hashes_sha256", "leaf_manifest_sha256", "consumer_contract_sha256",
    "source_file_table", "source_file_table_sha256", "selected_source_file_sha256",
    "session_bounds_and_internal_gap_evidence", "excluded_row_counts", "output_row_counts",
    "source_rows_read", "negative_price_preservation",
    "trade_row_preservation_independent_of_quote_validity", "source_ordering_evidence",
    "market_path_completeness_claim", "continuous_market_completeness",
    "receipt_payload_sha256",
}
REQUIRED_PROHIBITIONS = {
    "strategy_outcome_based_universe_selection", "random_train_validation_splits",
    "windows_crossing_contract_rolls", "labels_using_ticks_at_or_before_decision_or_entry",
    "training_validation_or_calibration_on_legacy_holdout_excluded",
    "depth_or_databento_without_separate_admission",
    "passive_fill_or_queue_claims_from_trade_ticks",
}


class CorpusV3Error(ValueError):
    """Raised when the sealed data contract cannot be verified."""


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def content_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(raw).hexdigest()


def load_contract(path: str | Path) -> dict[str, Any]:
    contract_path = Path(path).resolve()
    try:
        value = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CorpusV3Error(f"cannot read Corpus v3 contract: {contract_path}") from exc
    verify_contract(value)
    return value


def _parse_day(value: Any, field: str) -> date:
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise CorpusV3Error(f"{field} must be an ISO date") from exc


def _require_sha(value: Any, field: str) -> str:
    text = str(value)
    if not re.fullmatch(r"[0-9a-f]{64}", text):
        raise CorpusV3Error(f"{field} must be a lowercase SHA-256")
    return text


def verify_contract(contract: Mapping[str, Any], *, verify_artifacts: bool = False) -> None:
    if contract.get("schema_version") != SCHEMA_VERSION:
        raise CorpusV3Error(f"contract schema must be {SCHEMA_VERSION!r}")
    roots = contract.get("admitted_roots")
    if not isinstance(roots, list) or not roots or roots != sorted(set(map(str, roots))):
        raise CorpusV3Error("admitted_roots must be a nonempty sorted unique list")
    if set(roots) & set((contract.get("blocked_roots") or {}).keys()):
        raise CorpusV3Error("a root cannot be both admitted and blocked")
    admission = contract.get("current_admission") or {}
    admission_pair = (admission.get("status"), admission.get("materialization"))
    if admission_pair not in {
        ("coverage_audit_only", "blocked"),
        ("representative_shard_pilot", "representative_shard_only"),
    }:
        raise CorpusV3Error("Corpus v3 admission state is unsupported or too broad")
    source = contract.get("source") or {}
    source_end = _parse_day(source.get("max_date_exclusive"), "source.max_date_exclusive")
    _require_sha(source.get("hash_of_hashes_sha256"), "source.hash_of_hashes_sha256")
    if source.get("status_required") != "admitted_limited" or source.get("data_mode") != "raw_ticks":
        raise CorpusV3Error("Corpus v3 source must remain admitted-limited raw ticks")
    if source.get("quote_semantics") != "bid_and_ask_are_bbo_at_trade_not_quote_stream":
        raise CorpusV3Error("source quote semantics have drifted")
    loader = contract.get("loader") or {}
    _require_sha(loader.get("sha256"), "loader.sha256")
    if loader.get("disposition") != "reference_only_not_authorized_as_corpus_v3_export":
        raise CorpusV3Error("session_store_v6 cannot be treated as a Corpus v3 export API")
    export = contract.get("required_export_seam") or {}
    if export.get("owner") != "alphaforge" or export.get("purpose_token") != "foundation_training":
        raise CorpusV3Error("the required AlphaForge foundation export seam is not declared")
    if export.get("missing_event_seq_policy") != "reject":
        raise CorpusV3Error("Corpus v3 must reject missing event_seq rather than synthesize it")
    if export.get("must_reuse_admitted_loader_internals") is not True:
        raise CorpusV3Error("the export must reuse admitted loader internals")
    if export.get("must_not_use_purpose_tokens") != [
        "qa", "validation", "historical_validation"
    ]:
        raise CorpusV3Error("the export purpose-token prohibition has drifted")
    if export.get("receipt_verification") != (
        "fail_closed_before_any_FFM_bar_or_label_materialization"
    ):
        raise CorpusV3Error("export receipt verification must remain fail-closed")
    if export.get("mode") != "streaming_contract_day_shards_without_roll_splicing":
        raise CorpusV3Error("Corpus v3 export must emit unspliced contract-day shards")
    if export.get("receipt_schema_version") != "alphaforge_foundation_export_receipt_v2":
        raise CorpusV3Error("Corpus v3 requires the reconciled AlphaForge receipt-v2 schema")
    if export.get("output_bundle_files") != ["ticks.parquet", "receipt.json"]:
        raise CorpusV3Error("Corpus v3 export bundle must contain exactly Parquet plus receipt")
    if export.get("physical_hash_meaning") != "sha256_of_exact_ticks_parquet_bytes":
        raise CorpusV3Error("physical output hash meaning has drifted")
    if export.get("semantic_hash_meaning") != "canonical_ordered_rows_plus_bound_semantic_metadata":
        raise CorpusV3Error("semantic output hash meaning has drifted")
    if admission_pair == ("representative_shard_pilot", "representative_shard_only"):
        _require_sha(export.get("producer_exporter_sha256"), "producer_exporter_sha256")
        if not re.fullmatch(r"[0-9a-f]{40}", str(export.get("producer_git_commit"))):
            raise CorpusV3Error("pilot producer_git_commit must be a full lowercase Git SHA")
        pilot = export.get("pilot_request") or {}
        if (
            set(pilot) != {"root", "contract_id", "session_day", "split_use", "purpose"}
            or pilot.get("root") not in roots
            or contract_root(str(pilot.get("contract_id")), roots) != pilot.get("root")
            or pilot.get("purpose") != "foundation_training"
            or pilot.get("split_use") not in {
                "foundation_pretraining", "supervised_training", "development",
            }
        ):
            raise CorpusV3Error("representative pilot request is invalid")
    if set(export.get("required_row_fields") or []) != REQUIRED_EXPORT_ROWS:
        raise CorpusV3Error("Corpus v3 export row schema does not match the sealed contract")
    if set(export.get("required_shard_metadata") or []) != REQUIRED_SHARD_METADATA:
        raise CorpusV3Error("Corpus v3 shard metadata does not match the sealed contract")
    if set(export.get("required_receipt_bindings") or []) != REQUIRED_RECEIPT_BINDINGS:
        raise CorpusV3Error("Corpus v3 receipt bindings do not match the sealed contract")
    if export.get("ordering_must_be_strict_and_unique") != [
        "timestamp_utc_ns", "event_seq"
    ]:
        raise CorpusV3Error("Corpus v3 rows require a strict unique ordered event key")
    label = contract.get("label_contract") or {}
    if label.get("schema_version") != "ffm_ordered_tick_path_labels_v2":
        raise CorpusV3Error("Corpus v3 tick-label schema must be v2 integer-tick semantics")
    if label.get("purge_authority") != "declared_wall_clock_horizon_only":
        raise CorpusV3Error("declared label endpoints must be the sole purge authority")
    if label.get("quote_track_is_fill_proof") is not False:
        raise CorpusV3Error("BBO-at-trade labels must explicitly deny fill proof")
    if label.get("negative_prices") != "allowed_when_tick_aligned_and_finite":
        raise CorpusV3Error("the export and label contract must preserve valid negative prices")
    expected_label = {
        "decision_time": "regular_bar_close",
        "decision_manifest": (
            "hash_binds_caller_supplied_decision_risk_and_known_by_keys_"
            "receipt_verification_still_required"
        ),
        "entry_time": "first_ordered_trade_tick_strictly_after_decision_with_bounded_wait",
        "horizons_minutes": [60, 180, 360],
        "tick_order": ["timestamp_microseconds", "event_seq"],
        "barrier_order": "integer_tick_first_ordered_trade_touch_after_entry_record",
        "fractional_target_tick_rounding": "decimal_string_multiply_then_ceiling",
        "mfe_mae": "ordered_post_entry_trade_prices_in_tick_units",
        "label_end_observation": "last_ordered_tick_at_or_before_declared_wall_clock_horizon",
        "quote_track": (
            "marketable_at_trade_proxy_masked_when_any_required_bbo_at_trade_is_invalid"
        ),
        "fill_claim": "none_without_separate_marketability_contract",
    }
    for field, expected in expected_label.items():
        if label.get(field) != expected:
            raise CorpusV3Error(f"label_contract.{field} has drifted from sealed semantics")
    if label.get("no_contract_crossing") is not True:
        raise CorpusV3Error("labels must never cross a contract")
    if label.get("no_session_truncation") is not True:
        raise CorpusV3Error("labels must never silently truncate at a session boundary")
    splits = contract.get("splits") or {}
    required_splits = (
        "foundation_pretraining", "supervised_training", "development",
        "legacy_holdout_excluded",
    )
    if set(required_splits) - set(splits):
        raise CorpusV3Error("contract is missing required chronological splits")
    parsed: dict[str, tuple[date, date]] = {}
    for name in required_splits:
        start = _parse_day(splits[name].get("start"), f"splits.{name}.start")
        end = _parse_day(splits[name].get("end_exclusive"), f"splits.{name}.end_exclusive")
        if not start < end <= source_end:
            raise CorpusV3Error(f"invalid or out-of-source split {name}: {start}..{end}")
        parsed[name] = (start, end)
    supervised = parsed["supervised_training"]
    development = parsed["development"]
    holdout = parsed["legacy_holdout_excluded"]
    if not (supervised[1] == development[0] and development[1] == holdout[0]):
        raise CorpusV3Error("supervised/development/holdout boundaries must be contiguous")
    if parsed["foundation_pretraining"][1] > development[0]:
        raise CorpusV3Error("foundation pretraining must end before development begins")
    if splits["legacy_holdout_excluded"].get("use") != (
        "coverage_report_only_never_training_validation_calibration_or_selection"
    ):
        raise CorpusV3Error("legacy holdout must explicitly forbid training and validation")
    screen = contract.get("universe_screen") or {}
    if screen.get("uses_strategy_outcomes") is not False:
        raise CorpusV3Error("universe screen must explicitly forbid strategy outcomes")
    if screen.get("authorizes_universe_selection") is not False:
        raise CorpusV3Error("manifest-only audit cannot authorize universe selection")
    if "legacy_holdout_excluded" in set(screen.get("eligible_periods") or []):
        raise CorpusV3Error("legacy holdout cannot influence universe eligibility")
    ruler = contract.get("execution_ruler") or {}
    if type(ruler.get("primary_added_slippage_ticks_round_trip")) is not int or ruler.get(
        "primary_added_slippage_ticks_round_trip"
    ) != 0:
        raise CorpusV3Error("primary execution contract must preserve declared zero added slippage")
    expected_ruler = {
        "fees": "instrument_specific_round_trip_fee_from_pinned_economics_artifact",
        "fee_schedule_status": "provisional_static_approximation_not_effective_dated",
        "historical_economic_promotion": "blocked_pending_effective_dated_fee_schedule",
        "primary_added_delay": "none",
        "frozen_sensitivity_added_slippage_ticks_round_trip": 1,
    }
    for field, expected in expected_ruler.items():
        if ruler.get(field) != expected:
            raise CorpusV3Error(f"execution_ruler.{field} has drifted from sealed semantics")
    if set(contract.get("prohibited") or []) != REQUIRED_PROHIBITIONS:
        raise CorpusV3Error("the Corpus v3 prohibited-operation list has drifted")
    views = contract.get("derived_views") or {}
    native = views.get("contract_native_pretraining") or {}
    if not (
        native.get("status") == "required_primary_foundation_view"
        and native.get("stream_identity") == ["root", "contract_id"]
        and native.get("roll_selection") == "none"
        and native.get("window_crosses_contract") is False
    ):
        raise CorpusV3Error("contract-native pretraining view has drifted")
    downstream = views.get("causal_front_contract_downstream") or {}
    if not (
        downstream.get("status") == "blocked_pending_separate_admission"
        and downstream.get("selection_cutoff") == "not_later_than_each_decision_time"
        and downstream.get("future_or_full_day_activity") == "forbidden"
        and downstream.get("window_or_label_crosses_contract") is False
    ):
        raise CorpusV3Error("causal downstream view has drifted")
    artifacts = contract.get("artifacts") or {}
    required_artifacts = {
        "data_source_registry", "data_admission", "tick_admission", "loader_smoke",
        "instrument_economics", "market_calendar", "coverage_manifest", "lake_hash_summary",
        "lake_leaf_manifest",
    }
    if required_artifacts - set(artifacts):
        raise CorpusV3Error("contract is missing required governance artifacts")
    for name, record in artifacts.items():
        if not isinstance(record, Mapping) or not record.get("path"):
            raise CorpusV3Error(f"artifact {name!r} must have a path and SHA-256")
        expected = _require_sha(record.get("sha256"), f"artifacts.{name}.sha256")
        if verify_artifacts:
            path = Path(str(record["path"]))
            if not path.is_file():
                raise CorpusV3Error(f"required artifact is unavailable: {name} -> {path}")
            actual = sha256_file(path)
            if actual != expected:
                raise CorpusV3Error(
                    f"artifact hash drift for {name}: expected {expected}, observed {actual}"
                )
    if verify_artifacts:
        loader_path = Path(str(loader.get("path")))
        if not loader_path.is_file() or sha256_file(loader_path) != loader["sha256"]:
            raise CorpusV3Error("pinned loader is unavailable or has changed")
        if not Path(str(source.get("physical_root"))).is_dir():
            raise CorpusV3Error("admitted physical tick root is unavailable")


def contract_root(symbol: str, admitted_roots: list[str] | tuple[str, ...]) -> str | None:
    """Return a unique admitted root for a canonical futures contract symbol."""
    match = CONTRACT_RE.fullmatch(str(symbol))
    if not match:
        return None
    prefix = match.group(1)
    matches = [root for root in admitted_roots if prefix == root]
    if len(matches) > 1:
        raise CorpusV3Error(f"ambiguous contract root for {symbol!r}: {matches}")
    return matches[0] if matches else None


def _verify_governance_semantics(contract: Mapping[str, Any]) -> dict[str, bool]:
    """Cross-bind the hashed governance documents instead of trusting filenames alone."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - exercised by the optional data environment
        raise CorpusV3Error("Corpus v3 audit requires PyYAML; install the data extra") from exc

    artifacts = contract["artifacts"]
    roots = list(contract["admitted_roots"])
    source_id = contract["source"]["source_id"]
    source_end = contract["source"]["max_date_exclusive"]
    admission = yaml.safe_load(Path(artifacts["data_admission"]["path"]).read_text()) or {}
    matches = [
        row for row in admission.get("admissions") or []
        if row.get("source_id") == source_id
    ]
    if len(matches) != 1:
        raise CorpusV3Error("data admission must contain exactly one source record")
    admitted = matches[0]
    tick_qa = yaml.safe_load(Path(artifacts["tick_admission"]["path"]).read_text()) or {}
    smoke = yaml.safe_load(Path(artifacts["loader_smoke"]["path"]).read_text()) or {}
    instruments = yaml.safe_load(
        Path(artifacts["instrument_economics"]["path"]).read_text()
    ) or {}
    calendar = yaml.safe_load(Path(artifacts["market_calendar"]["path"]).read_text()) or {}
    summary = json.loads(Path(artifacts["lake_hash_summary"]["path"]).read_text())
    calendar_roots = {
        str(root)
        for product in (calendar.get("products") or {}).values()
        for root in product.get("roots") or []
    }
    checks = {
        "admission_source": admitted.get("status") == "admitted_limited",
        "admission_roots": sorted(map(str, admitted.get("roots") or [])) == roots,
        "admission_mode": admitted.get("admitted_data_modes") == ["raw_ticks"],
        "admission_cutoff": admitted.get("max_date_exclusive") == source_end,
        "qa_source": tick_qa.get("source_id") == source_id
        and tick_qa.get("decision") == "admitted_limited",
        "qa_roots": sorted(map(str, (tick_qa.get("summary") or {}).get("admitted") or []))
        == roots,
        "qa_cutoff": (tick_qa.get("scope") or {}).get("max_date_exclusive") == source_end,
        "qa_registry": (tick_qa.get("registry") or {}).get("sha256")
        == artifacts["data_source_registry"]["sha256"],
        "qa_lake": (tick_qa.get("corpus") or {}).get("hash_of_hashes_sha256")
        == contract["source"]["hash_of_hashes_sha256"],
        "qa_loader": (tick_qa.get("loader_policy") or {}).get("schema_version")
        == contract["loader"]["schema_version"],
        "smoke_source": smoke.get("source_id") == source_id,
        "smoke_admission": smoke.get("admission_artifact_sha256")
        == artifacts["tick_admission"]["sha256"],
        "smoke_roots": (smoke.get("summary") or {}).get("roots") == len(roots)
        and (smoke.get("summary") or {}).get("passed") == len(roots)
        and (smoke.get("summary") or {}).get("failed") == 0,
        "instrument_roots": set(roots).issubset(set((instruments.get("instruments") or {}).keys())),
        "calendar_roots": set(roots).issubset(calendar_roots),
        "calendar_cutoff": (calendar.get("coverage") or {}).get("end")
        == (date.fromisoformat(source_end).replace(day=1) - date.resolution).isoformat(),
        "leaf_summary_name": summary.get("manifest_jsonl")
        == Path(artifacts["lake_leaf_manifest"]["path"]).name,
        "leaf_summary_hash": summary.get("hash_of_hashes_sha256")
        == contract["source"]["hash_of_hashes_sha256"],
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise CorpusV3Error(f"governance semantic cross-binding failed: {failed}")
    return checks


def _period_metrics(days: Mapping[date, Mapping[str, Any]], start: date, end: date) -> dict[str, Any]:
    selected = [(day, value) for day, value in sorted(days.items()) if start <= day < end]
    expected = (end - start).days
    if not selected:
        return {
            "active_days": 0, "first_day": None, "last_day": None, "total_ticks": 0,
            "utc_bucket_density": 0.0, "median_top_contract_ticks": 0.0,
            "p10_top_contract_ticks": 0.0, "overlap_day_fraction": 0.0,
            "max_utc_bucket_gap_days": expected, "years_with_150_utc_buckets": 0,
        }
    day_keys = [item[0] for item in selected]
    top = sorted(int(item[1]["top_rows"]) for item in selected)
    year_counts: dict[int, int] = defaultdict(int)
    for day in day_keys:
        year_counts[day.year] += 1
    gaps = [max(0, (right - left).days - 1) for left, right in zip(day_keys, day_keys[1:])]
    p10_index = max(0, int(0.10 * (len(top) - 1)))
    return {
        "active_days": len(selected),
        "first_day": day_keys[0].isoformat(),
        "last_day": day_keys[-1].isoformat(),
        "total_ticks": sum(int(item[1]["rows"]) for item in selected),
        "utc_bucket_density": round(len(selected) / expected, 8) if expected else 0.0,
        "median_top_contract_ticks": float(median(top)),
        "p10_top_contract_ticks": float(top[p10_index]),
        "overlap_day_fraction": round(
            sum(int(item[1]["contracts"]) > 1 for item in selected) / len(selected), 8
        ),
        "max_utc_bucket_gap_days": max(gaps, default=0),
        "years_with_150_utc_buckets": sum(value >= 150 for value in year_counts.values()),
    }


def _diagnostic_flags(
    metrics: Mapping[str, Mapping[str, Any]], screen: Mapping[str, Any]
) -> dict[str, Any]:
    """Report rough UTC-bucket flags without admitting or rejecting a root."""
    train = metrics["supervised_training"]
    development = metrics["development"]
    thresholds = screen["provisional_diagnostic_thresholds"]
    checks = {
        "training_utc_buckets": train["active_days"]
        >= int(thresholds["min_training_utc_buckets"]),
        "training_years_with_150_utc_buckets": train["years_with_150_utc_buckets"]
        >= int(thresholds["min_training_years_with_150_utc_buckets"]),
        "development_utc_buckets": development["active_days"]
        >= int(thresholds["min_development_utc_buckets"]),
        "training_median_top_contract_ticks_per_utc_bucket": train["median_top_contract_ticks"]
        >= int(thresholds["min_median_top_contract_ticks_per_utc_bucket"]),
        "training_utc_bucket_gap_days": train["max_utc_bucket_gap_days"]
        <= int(thresholds["max_training_utc_bucket_gap_days"]),
    }
    return {
        "report_only": True,
        "passes_all_provisional_diagnostics": all(checks.values()),
        "checks": checks,
        "diagnostic_flags": sorted(name for name, passed in checks.items() if not passed),
        "warning": "UTC inventory buckets are not exchange sessions and cannot select roots",
    }


def build_coverage_audit(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Build a deterministic root/year audit from the pinned inventory only."""
    verify_contract(contract, verify_artifacts=True)
    semantic_checks = _verify_governance_semantics(contract)
    roots = list(contract["admitted_roots"])
    manifest = Path(contract["artifacts"]["coverage_manifest"]["path"])
    source_end = _parse_day(contract["source"]["max_date_exclusive"], "source end")
    root_days: dict[str, dict[date, dict[str, Any]]] = {root: {} for root in roots}
    ignored = defaultdict(int)
    inventory_lines = 0
    all_tick_files = all_tick_rows = 0
    admitted_files = admitted_rows = 0
    admitted_symbols: set[str] = set()
    admitted_symbols_all_dates: set[str] = set()
    admitted_keys: set[tuple[str, str]] = set()
    with gzip.open(manifest, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CorpusV3Error(f"invalid coverage JSON on line {line_number}") from exc
            inventory_lines += 1
            if set(row) != {"sym", "kind", "day", "files", "rows"}:
                raise CorpusV3Error(f"unexpected coverage fields on line {line_number}")
            if not all(type(row[name]) is str for name in ("sym", "kind", "day")):
                raise CorpusV3Error(f"coverage text fields have invalid types on line {line_number}")
            if not all(type(row[name]) is int for name in ("files", "rows")):
                raise CorpusV3Error(f"coverage counters must be strict integers on line {line_number}")
            if row.get("kind") != "ticks":
                ignored["non_tick"] += 1
                continue
            root = contract_root(str(row.get("sym", "")), roots)
            all_tick_files += row["files"]
            all_tick_rows += row["rows"]
            if root is None:
                ignored["unadmitted_symbol"] += 1
                continue
            admitted_symbols_all_dates.add(row["sym"])
            try:
                day = date.fromisoformat(str(row["day"]))
            except (KeyError, ValueError) as exc:
                raise CorpusV3Error(f"invalid coverage day on line {line_number}") from exc
            rows = int(row.get("rows", 0))
            files = int(row.get("files", 0))
            if rows < 0 or files < 0:
                raise CorpusV3Error(f"negative coverage count on line {line_number}")
            if day >= source_end:
                ignored["at_or_after_source_end"] += 1
                continue
            key = (row["sym"], row["day"])
            if key in admitted_keys:
                raise CorpusV3Error(f"duplicate admitted symbol/day coverage row: {key}")
            admitted_keys.add(key)
            admitted_files += row["files"]
            admitted_rows += row["rows"]
            admitted_symbols.add(row["sym"])
            if rows == 0:
                ignored["zero_row_placeholder"] += 1
                continue
            value = root_days[root].setdefault(
                day, {"rows": 0, "top_rows": 0, "files": 0, "contracts": 0}
            )
            value["rows"] += rows
            value["top_rows"] = max(value["top_rows"], rows)
            value["files"] += files
            value["contracts"] += 1

    expected_totals = contract["source"]["inventory_totals"]
    observed_totals = {
        "all_tick_files": all_tick_files,
        "all_tick_rows": all_tick_rows,
        "admitted_pre_cutoff_files": admitted_files,
        "admitted_pre_cutoff_rows": admitted_rows,
        "admitted_contract_symbols_all_dates": len(admitted_symbols_all_dates),
        "admitted_contract_symbols_pre_cutoff": len(admitted_symbols),
    }
    if observed_totals != expected_totals:
        raise CorpusV3Error(
            f"coverage inventory conservation failed: expected={expected_totals}, "
            f"observed={observed_totals}"
        )

    parsed_splits = {
        name: (
            _parse_day(value["start"], f"splits.{name}.start"),
            _parse_day(value["end_exclusive"], f"splits.{name}.end_exclusive"),
        )
        for name, value in contract["splits"].items()
    }
    reports: dict[str, Any] = {}
    diagnostic_flagged_roots: list[str] = []
    for root in roots:
        days = root_days[root]
        periods = {
            name: _period_metrics(days, start, end)
            for name, (start, end) in parsed_splits.items()
        }
        years: dict[str, Any] = {}
        if days:
            for year in range(min(days).year, max(days).year + 1):
                years[str(year)] = _period_metrics(
                    days, date(year, 1, 1), date(year + 1, 1, 1)
                )
        diagnostics = _diagnostic_flags(periods, contract["universe_screen"])
        if diagnostics["diagnostic_flags"]:
            diagnostic_flagged_roots.append(root)
        reports[root] = {"periods": periods, "years": years, "screen": diagnostics}

    artifact_hashes = {
        name: sha256_file(record["path"])
        for name, record in sorted(contract["artifacts"].items())
    }
    report: dict[str, Any] = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "contract_id": contract["contract_id"],
        "contract_sha256": content_sha256(contract),
        "purpose": "coverage_and_liquidity_only_no_strategy_events_labels_or_outcomes_read",
        "source_scope": {
            "source_id": contract["source"]["source_id"],
            "max_date_exclusive": contract["source"]["max_date_exclusive"],
            "coverage_manifest_sha256": artifact_hashes["coverage_manifest"],
            "lake_hash_of_hashes_sha256": contract["source"]["hash_of_hashes_sha256"],
            "inventory_lines": inventory_lines,
            "inventory_totals": observed_totals,
            "ignored": dict(sorted(ignored.items())),
        },
        "screen": dict(contract["universe_screen"]),
        "candidate_roots": roots,
        "selected_roots": [],
        "selection_status": "blocked_pending_sessionized_foundation_export",
        "diagnostic_flagged_roots": diagnostic_flagged_roots,
        "roots": reports,
        "artifact_sha256": artifact_hashes,
        "governance_semantic_checks": semantic_checks,
        "holdout_use": contract["splits"]["legacy_holdout_excluded"]["use"],
        "materialization_status": contract["current_admission"],
    }
    report["report_sha256"] = content_sha256(report)
    return report


def write_coverage_audit(report: Mapping[str, Any], output: str | Path) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, sort_keys=True, indent=2) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path)
    return path
