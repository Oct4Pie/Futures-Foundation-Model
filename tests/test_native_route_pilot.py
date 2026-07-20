import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from futures_foundation.finetune.native_contracts import NativeContractError, get_dossier
from futures_foundation.finetune.native_route_pilot import (
    REQUIRED_PILOT_ARTIFACTS,
    build_route_pilot_evidence,
    load_route_pilot_evidence,
    validate_route_pilot_evidence,
)
from futures_foundation.finetune.native_route_smoke import (
    REQUIRED_SMOKE_ARTIFACTS,
    build_route_smoke_evidence,
)
from futures_foundation.finetune.native_smoke_contract import REQUIRED_SMOKE_CHECKS
from futures_foundation.finetune.routes import chronos_bolt
from futures_foundation.finetune.tournament_cache_authority import (
    SOURCE_AUTHORITY_SCHEMA_VERSION,
    canonical_authority_document,
)
from futures_foundation.finetune.tournament_data import (
    CACHE_MANIFEST,
    build_cache,
    cache_manifest_sha256,
)


ROUTE_KEY = "chronos_bolt:F:direct_native_quantile_pinball"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cache(tmp_path: Path):
    source = tmp_path / "source"
    cache = tmp_path / "cache"
    source.mkdir()
    ts = pd.date_range("2019-06-28", "2025-07-03", freq="1D", tz="UTC")
    close = 100 + np.arange(len(ts), dtype=float)
    csv = source / "ES_1D.csv"
    pd.DataFrame({
        "datetime": ts,
        "open": close,
        "high": close + 1,
        "low": close - 1,
        "close": close,
        "volume": 10.0,
        "contract_id": "ESZ4",
    }).to_csv(csv, index=False)
    manifest = {
        "schema_version": "ffm_ssl_corpus_v1",
        "created_utc": "2026-07-18T00:00:00+00:00",
        "purpose": "self-supervised OHLCV only; no labels or outcomes read",
        "source_root": str(source.resolve()),
        "source_snapshot_sha256": _sha(csv),
        "roots": ["ES"],
        "timeframes_minutes": [1440],
        "resample": {
            "closed": "left", "label": "left", "origin": "epoch",
            "forward_fill": False, "within_contract_only": True,
        },
        "roots_report": {},
        "outputs": {
            "ES_1D": {
                "path": str(csv.resolve()),
                "bytes": csv.stat().st_size,
                "sha256": _sha(csv),
                "rows": len(ts),
            },
        },
    }
    manifest_path = source / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest) + "\n")
    authority = tmp_path / "authority.json"
    authority.write_bytes(canonical_authority_document({
        "schema_version": SOURCE_AUTHORITY_SCHEMA_VERSION,
        "authority_id": "pilot-test-source",
        "purpose": "tournament_cache_source_admission",
        "source_manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": _sha(manifest_path),
            "bytes": manifest_path.stat().st_size,
            "schema_version": "ffm_ssl_corpus_v1",
        },
        "admitted_streams": ["ES@1D"],
        "cache_construction_admitted": True,
        "training_admitted": False,
    }))
    build_cache(
        source, cache, ("ES",), ("1D",),
        source_authority_path=authority,
        source_authority_sha256=_sha(authority),
        verbose=False,
    )
    return cache, cache_manifest_sha256(cache)


def _snapshot(tmp_path: Path) -> Path:
    revision = get_dossier("chronos_bolt")["model_revision"]
    path = tmp_path / revision
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}\n")
    return path


def _smoke(tmp_path: Path, snapshot: Path) -> Path:
    artifacts = {
        "model_snapshot": snapshot,
        "source_runtime": Path(
            importlib.metadata.distribution("chronos-forecasting")._path
        ).resolve(),
    }
    for name in sorted(
        REQUIRED_SMOKE_ARTIFACTS - {"model_snapshot", "source_runtime"}
    ):
        path = tmp_path / f"smoke-{name}.bin"
        path.write_bytes(name.encode())
        artifacts[name] = path
    checks = {
        name: {"status": "pass", "metrics": {"measured": True}, "reason": None}
        for name in REQUIRED_SMOKE_CHECKS
    }
    evidence = build_route_smoke_evidence(
        route_key=ROUTE_KEY,
        executor_path=chronos_bolt.__file__,
        executor_entrypoint="native_loss/direct_quantiles",
        checks=checks,
        artifacts=artifacts,
        metrics={"fixture": True},
        created_utc="2026-07-18T00:00:00Z",
    )
    path = tmp_path / "smoke.json"
    path.write_text(json.dumps(evidence))
    return path


def _pilot(tmp_path: Path):
    cache, cache_sha = _cache(tmp_path)
    snapshot = _snapshot(tmp_path / "model")
    smoke = _smoke(tmp_path, snapshot)
    artifacts = {
        "model_snapshot": snapshot,
        "smoke_evidence": smoke,
        "cache_manifest": cache / CACHE_MANIFEST,
    }
    for name in sorted(REQUIRED_PILOT_ARTIFACTS - set(artifacts)):
        path = tmp_path / f"pilot-{name}.bin"
        path.write_bytes(name.encode())
        artifacts[name] = path
    exposure = {
        "sampling_kind": "uniform_stream_then_uniform_window_v1",
        "train_schedule_sha256": "1" * 64,
        "validation_schedule_sha256": "2" * 64,
        "train_examples": 10,
        "validation_examples": 4,
        "train_stream_counts": {"ES@1D": 10},
        "validation_stream_counts": {"ES@1D": 4},
        "seed": 1,
        "validation_seed": 2,
    }
    metrics = {
        "vanilla_validation_loss": 10.0,
        "adapted_validation_loss": 9.0,
        "relative_validation_improvement": 0.1,
        "required_relative_improvement": 0.01,
        "best_step": 1,
        "history": [{"step": 1, "validation_loss": 9.0}],
    }
    evidence = build_route_pilot_evidence(
        route_key=ROUTE_KEY,
        smoke_evidence_path=smoke,
        executor_path=chronos_bolt.__file__,
        executor_entrypoint="native_loss/direct_quantiles",
        cache_dir=cache,
        cache_manifest_sha256=cache_sha,
        stream_ids=["ES@1D"],
        exposure=exposure,
        metrics=metrics,
        artifacts=artifacts,
        created_utc="2026-07-18T00:00:00Z",
    )
    return evidence, artifacts


def test_pilot_evidence_reopens_complete_chain_and_never_promotes(tmp_path):
    evidence, _ = _pilot(tmp_path)
    assert evidence["pilot_completed"] is True
    assert evidence["native_objective_survived"] is True
    assert evidence["promotion_admitted"] is False
    assert evidence["full_training_admitted"] is False
    assert validate_route_pilot_evidence(evidence) == evidence
    path = tmp_path / "pilot.json"
    path.write_text(json.dumps(evidence))
    assert load_route_pilot_evidence(path) == evidence


def test_pilot_rejects_artifact_and_cache_tamper(tmp_path):
    evidence, artifacts = _pilot(tmp_path)
    Path(artifacts["raw_report"]).write_bytes(b"tampered")
    with pytest.raises(NativeContractError, match="drifted"):
        validate_route_pilot_evidence(evidence)


def test_pilot_rejects_inconsistent_improvement(tmp_path):
    evidence, _ = _pilot(tmp_path)
    evidence["metrics"]["relative_validation_improvement"] = 0.5
    payload = dict(evidence)
    payload.pop("evidence_sha256")
    from futures_foundation.finetune.native_contracts import content_sha256
    evidence["evidence_sha256"] = content_sha256(payload)
    with pytest.raises(NativeContractError, match="inconsistent"):
        validate_route_pilot_evidence(evidence)


def test_pilot_accepts_finite_negative_native_losses(tmp_path):
    evidence, _ = _pilot(tmp_path)
    evidence["metrics"].update({
        "vanilla_validation_loss": -0.1,
        "adapted_validation_loss": -0.12,
        "relative_validation_improvement": 0.2,
    })
    evidence["native_objective_survived"] = True
    payload = dict(evidence)
    payload.pop("evidence_sha256")
    from futures_foundation.finetune.native_contracts import content_sha256
    evidence["evidence_sha256"] = content_sha256(payload)
    assert validate_route_pilot_evidence(evidence)["native_objective_survived"] is True


def test_pilot_rejects_promotion_forgery(tmp_path):
    evidence, _ = _pilot(tmp_path)
    evidence["promotion_admitted"] = True
    payload = dict(evidence)
    payload.pop("evidence_sha256")
    from futures_foundation.finetune.native_contracts import content_sha256
    evidence["evidence_sha256"] = content_sha256(payload)
    with pytest.raises(NativeContractError, match="cannot grant"):
        validate_route_pilot_evidence(evidence)
