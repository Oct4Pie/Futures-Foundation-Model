"""Verified downstream incremental-screen evidence for native routes.

The incremental screen can fund a nonlinear sensitivity only when the frozen
policy passes.  It never grants route promotion, full training, OOS access,
deployment, or trading.  Reports are reopened against their sample, row
selection, native feature table, predictions, fold contract, implementation
bytes, and recomputed screen verdict.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .downstream_sample import load_balanced_sample, load_row_selection
from .native_contracts import NativeContractError, content_sha256
from .native_downstream_features import load_feature_table
from .native_downstream_ruler import PRIMARY_TARGETS, SCREEN_POLICY, screen_verdict


INCREMENTAL_REPORT_SCHEMA = "ffm_native_incremental_benchmark_v1"
SCREEN_COLLECTION_SCHEMA = "ffm_native_incremental_screen_collection_v1"
COMMON_INFORMATION_ROUTES = frozenset({
    "chronos_bolt:F:direct_native_quantile_pinball",
    "chronos_v1:F:native_64_t5_token_forecast_cross_entropy",
    "moment_small:R:masked_patch_reconstruction",
    "kronos_mini:F:hierarchical_autoregressive_tokens",
    "mantis_v1:R:official_crop_resize_contrastive",
    "mantis_v2:R:official_crop_resize_contrastive",
    "moment_small:F:forecast_full_raw_mse",
    "moment_small:F:forecast_head_only_raw_mse",
})
_REPORT_FIELDS = {
    "schema_version", "status", "created_utc", "oos_read", "route_key",
    "sample", "row_selection", "feature_table", "configuration",
    "fold_contract", "causal_feature_names", "bottlenecks", "skipped_targets",
    "fold_scores", "summary", "screen_verdict", "predictions", "source",
    "nonlinear_sensitivity_funded", "promotion_admitted",
    "full_training_admitted", "live_trading_ready",
}


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    if not path.is_file() or path.is_symlink():
        raise NativeContractError(f"incremental screen report is missing or unsafe: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NativeContractError(f"cannot read incremental screen report: {path}") from exc


def _identity(value: Any, label: str) -> dict[str, Any]:
    required = {"path", "sha256", "content_fingerprint"}
    if not isinstance(value, Mapping) or set(value) != required:
        raise NativeContractError(f"{label} identity fields mismatch")
    path = Path(str(value["path"])).expanduser().resolve()
    if not path.is_file() or path.is_symlink() or _sha256(path) != value["sha256"]:
        raise NativeContractError(f"{label} artifact bytes changed")
    return {"path": path, **dict(value)}


def _validate_predictions(
    value: Any,
    *,
    feature_table_sha256: str,
    fold_contract_sha256: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256", "rows"}:
        raise NativeContractError("incremental predictions identity is malformed")
    path = Path(str(value["path"])).expanduser().resolve()
    if not path.is_file() or path.is_symlink() or _sha256(path) != value["sha256"]:
        raise NativeContractError("incremental prediction artifact bytes changed")
    with np.load(path, allow_pickle=False) as saved:
        required = {
            "row_index", "target_index", "fold", "y_true", "causal", "model",
            "causal_plus_model", "residual_over_causal", "model_shuffled_label",
            "model_random_feature", "model_time_destroyed", "target_names",
            "arm_names", "feature_table_sha256", "fold_contract_sha256",
        }
        if set(saved.files) != required:
            raise NativeContractError("incremental prediction array closure is invalid")
        lengths = [len(saved[name]) for name in (
            "row_index", "target_index", "fold", "y_true", "causal", "model",
            "causal_plus_model", "residual_over_causal", "model_shuffled_label",
            "model_random_feature", "model_time_destroyed",
        )]
        if len(set(lengths)) != 1 or lengths[0] != int(value["rows"]):
            raise NativeContractError("incremental prediction rows are inconsistent")
        if (
            str(saved["feature_table_sha256"].item()) != feature_table_sha256
            or str(saved["fold_contract_sha256"].item()) != fold_contract_sha256
        ):
            raise NativeContractError("incremental prediction lineage changed")
        for name in (
            "y_true", "causal", "model", "causal_plus_model",
            "residual_over_causal", "model_shuffled_label",
            "model_random_feature", "model_time_destroyed",
        ):
            if not np.isfinite(saved[name]).all():
                raise NativeContractError("incremental predictions contain non-finite values")
    return {"path": str(path), "sha256": value["sha256"], "rows": int(value["rows"])}


def _validate_source(value: Any) -> dict[str, Any]:
    required = {"git_revision", "working_tree_dirty", "implementation_sha256"}
    if not isinstance(value, Mapping) or set(value) != required:
        raise NativeContractError("incremental report source fields mismatch")
    implementation = value["implementation_sha256"]
    if not isinstance(implementation, Mapping) or set(implementation) != {
        "probe", "ruler", "runner",
    }:
        raise NativeContractError("incremental implementation closure is invalid")
    root = Path(__file__).resolve().parents[2]
    paths = {
        "probe": root / "futures_foundation/finetune/downstream_probe.py",
        "ruler": root / "futures_foundation/finetune/native_downstream_ruler.py",
        "runner": root / "scripts/benchmark_native_incremental.py",
    }
    for name, path in paths.items():
        if not path.is_file() or path.is_symlink() or _sha256(path) != implementation[name]:
            raise NativeContractError(f"incremental {name} implementation bytes changed")
    return deepcopy(dict(value))


def load_incremental_screen_report(path: str | Path) -> dict[str, Any]:
    """Reopen one complete incremental report and recompute its fail/pass verdict."""
    report_path = Path(path).expanduser().resolve()
    value = _read_json(report_path)
    if not isinstance(value, Mapping) or set(value) != _REPORT_FIELDS:
        raise NativeContractError("incremental report field closure is invalid")
    report = deepcopy(dict(value))
    if (
        report["schema_version"] != INCREMENTAL_REPORT_SCHEMA
        or report["status"] != "complete"
        or report["oos_read"] is not False
    ):
        raise NativeContractError("incremental report schema/status/OOS scope is invalid")
    route_key = report["route_key"]
    if route_key not in COMMON_INFORMATION_ROUTES:
        raise NativeContractError("incremental report route is outside the common-information screen")

    sample_identity = _identity(report["sample"], "incremental sample")
    sample, sample_manifest = load_balanced_sample(sample_identity["path"])
    if (
        sample_manifest["artifact"]["sha256"] != report["sample"]["sha256"]
        or sample_manifest["content_fingerprint"] != report["sample"]["content_fingerprint"]
    ):
        raise NativeContractError("incremental sample identity is stale")
    selection_identity = _identity(report["row_selection"], "incremental row selection")
    selection, selection_manifest = load_row_selection(
        selection_identity["path"], sample_manifest=sample_manifest,
    )
    if (
        selection_manifest["artifact"]["sha256"] != report["row_selection"]["sha256"]
        or selection_manifest["content_fingerprint"]
        != report["row_selection"]["content_fingerprint"]
    ):
        raise NativeContractError("incremental row-selection identity is stale")

    feature = report["feature_table"]
    required_feature = {
        "path", "sha256", "content_fingerprint", "feature_count",
        "feature_kind", "information_view",
    }
    if not isinstance(feature, Mapping) or set(feature) != required_feature:
        raise NativeContractError("incremental feature-table identity is malformed")
    feature_path = Path(str(feature["path"])).expanduser().resolve()
    arrays, feature_manifest = load_feature_table(feature_path)
    metadata = feature_manifest["metadata"]
    if (
        feature_manifest["artifact"]["sha256"] != feature["sha256"]
        or feature_manifest["content_fingerprint"] != feature["content_fingerprint"]
        or metadata["route_key"] != route_key
        or metadata["feature_count"] != feature["feature_count"]
        or metadata["feature_kind"] != feature["feature_kind"]
        or metadata["information_view"] != feature["information_view"]
        or metadata["sample"]["sha256"] != report["sample"]["sha256"]
        or metadata["row_selection"]["sha256"] != report["row_selection"]["sha256"]
        or not np.array_equal(arrays["row_index"], selection["row_index"])
    ):
        raise NativeContractError("incremental feature table is stale or substituted")

    configuration = report["configuration"]
    required_configuration = {
        "targets", "folds", "context_bars", "expected_fold_sha256",
        "pca_components", "bootstrap_repetitions", "seed", "block_weights",
        "linear_heads",
    }
    if not isinstance(configuration, Mapping) or set(configuration) != required_configuration:
        raise NativeContractError("incremental configuration fields mismatch")
    if (
        configuration["targets"] != list(PRIMARY_TARGETS)
        or int(configuration["pca_components"]) != SCREEN_POLICY["pca_components"]
        or int(configuration["bootstrap_repetitions"])
        != SCREEN_POLICY["bootstrap_repetitions"]
        or int(configuration["folds"]) != 2
        or int(configuration["context_bars"]) != 512
        or configuration["block_weights"] is not True
    ):
        raise NativeContractError("incremental configuration differs from the frozen policy")
    fold = report["fold_contract"]
    if (
        not isinstance(fold, Mapping)
        or fold.get("contract_sha256") != configuration["expected_fold_sha256"]
        or fold.get("folds") != 2
        or fold.get("groups") != 9
        or fold.get("rows") != len(selection["row_index"])
    ):
        raise NativeContractError("incremental fold contract is stale")

    summary = report["summary"]
    if not isinstance(summary, list) or len(summary) != len(PRIMARY_TARGETS):
        raise NativeContractError("incremental target summary closure is invalid")
    recomputed = screen_verdict(summary)
    if recomputed != report["screen_verdict"]:
        raise NativeContractError("incremental screen verdict differs from measured summaries")
    survived = recomputed["downstream_screen_survived"]
    if (
        report["nonlinear_sensitivity_funded"] is not survived
        or report["promotion_admitted"] is not False
        or report["full_training_admitted"] is not False
        or report["live_trading_ready"] is not False
    ):
        raise NativeContractError("incremental report authorization flags are inconsistent")
    predictions = _validate_predictions(
        report["predictions"],
        feature_table_sha256=feature["sha256"],
        fold_contract_sha256=fold["contract_sha256"],
    )
    _validate_source(report["source"])
    report["report_identity"] = {
        "path": str(report_path),
        "sha256": _sha256(report_path),
        "bytes": report_path.stat().st_size,
    }
    report["verified_feature_pilot_evidence"] = deepcopy(
        metadata["pilot_evidence"]
    )
    report["verified_predictions"] = predictions
    return report


def build_screen_collection(report_paths: Mapping[str, str | Path]) -> dict[str, Any]:
    """Build one non-authorizing collection for the exact common-information route set."""
    if not isinstance(report_paths, Mapping) or set(report_paths) != COMMON_INFORMATION_ROUTES:
        raise NativeContractError("screen collection requires the exact common-information route set")
    rows: dict[str, Any] = {}
    for route_key in sorted(COMMON_INFORMATION_ROUTES):
        report = load_incremental_screen_report(report_paths[route_key])
        if report["route_key"] != route_key:
            raise NativeContractError("screen collection route/report identity mismatch")
        verdict = report["screen_verdict"]
        rows[route_key] = {
            "report": report["report_identity"],
            "feature_table": deepcopy(report["feature_table"]),
            "pilot_evidence": deepcopy(report["verified_feature_pilot_evidence"]),
            "model_control_wins": int(verdict["model_control_wins"]),
            "incremental_point_wins": int(verdict["incremental_point_wins"]),
            "residual_bonferroni_ci_wins": int(
                verdict["residual_bonferroni_ci_wins"]
            ),
            "primary_point_degradations": int(verdict["primary_point_degradations"]),
            "downstream_screen_survived": bool(
                verdict["downstream_screen_survived"]
            ),
            "nonlinear_sensitivity_funded": bool(
                verdict["nonlinear_sensitivity_funded"]
            ),
            "full_training_admitted": False,
        }
    survived = sorted(
        route for route, row in rows.items() if row["downstream_screen_survived"]
    )
    document = {
        "schema_version": SCREEN_COLLECTION_SCHEMA,
        "policy": deepcopy(SCREEN_POLICY),
        "routes": rows,
        "counts": {
            "reports_completed": len(rows),
            "downstream_screen_survived": len(survived),
            "nonlinear_sensitivity_funded": len(survived),
            "full_training_admitted": 0,
        },
        "surviving_routes": survived,
        "nonlinear_sensitivity_funded_routes": survived,
        "promotion_admitted": False,
        "full_training_admitted": False,
        "oos_admitted": False,
        "live_trading_ready": False,
    }
    document["collection_sha256"] = content_sha256(document)
    return document


def validate_screen_collection(
    value: Any,
    *,
    report_paths: Mapping[str, str | Path],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise NativeContractError("screen collection must be a mapping")
    expected = build_screen_collection(report_paths)
    if dict(value) != expected:
        raise NativeContractError("screen collection is stale or non-canonical")
    return deepcopy(expected)


__all__ = [
    "COMMON_INFORMATION_ROUTES", "INCREMENTAL_REPORT_SCHEMA",
    "SCREEN_COLLECTION_SCHEMA", "build_screen_collection",
    "load_incremental_screen_report", "validate_screen_collection",
]
