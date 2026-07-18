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
    if admission.get("status") != "coverage_audit_only":
        raise CorpusV3Error("current Corpus v3 contract is admitted only for coverage audit")
    if admission.get("materialization") != "blocked":
        raise CorpusV3Error("materialization must remain blocked until the export seam passes")
    source = contract.get("source") or {}
    source_end = _parse_day(source.get("max_date_exclusive"), "source.max_date_exclusive")
    _require_sha(source.get("hash_of_hashes_sha256"), "source.hash_of_hashes_sha256")
    loader = contract.get("loader") or {}
    _require_sha(loader.get("sha256"), "loader.sha256")
    if loader.get("disposition") != "reference_only_not_authorized_as_corpus_v3_export":
        raise CorpusV3Error("session_store_v6 cannot be treated as a Corpus v3 export API")
    export = contract.get("required_export_seam") or {}
    if export.get("owner") != "alphaforge" or export.get("purpose_token") != "foundation_training":
        raise CorpusV3Error("the required AlphaForge foundation export seam is not declared")
    if export.get("missing_event_seq_policy") != "reject":
        raise CorpusV3Error("Corpus v3 must reject missing event_seq rather than synthesize it")
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
    if "never_training_validation" not in str(splits["legacy_holdout_excluded"].get("use")):
        raise CorpusV3Error("legacy holdout must explicitly forbid training and validation")
    screen = contract.get("universe_screen") or {}
    if screen.get("uses_strategy_outcomes") is not False:
        raise CorpusV3Error("universe screen must explicitly forbid strategy outcomes")
    if screen.get("authorizes_universe_selection") is not False:
        raise CorpusV3Error("manifest-only audit cannot authorize universe selection")
    if "legacy_holdout_excluded" in set(screen.get("eligible_periods") or []):
        raise CorpusV3Error("legacy holdout cannot influence universe eligibility")
    if int((contract.get("execution_ruler") or {}).get(
        "primary_added_slippage_ticks_round_trip", -1
    )) != 0:
        raise CorpusV3Error("primary execution contract must preserve declared zero added slippage")
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
