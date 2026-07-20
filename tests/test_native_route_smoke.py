from copy import deepcopy
import importlib.metadata
import json
from pathlib import Path

import pytest

from futures_foundation.finetune.native_contracts import NativeContractError, get_dossier
from futures_foundation.finetune.native_route_smoke import (
    REQUIRED_SMOKE_ARTIFACTS,
    build_route_smoke_evidence,
    load_route_smoke_evidence,
    validate_route_smoke_evidence,
)
from futures_foundation.finetune.native_smoke_contract import REQUIRED_SMOKE_CHECKS
from futures_foundation.finetune.routes import chronos_bolt


ROUTE_KEY = "chronos_bolt:F:direct_native_quantile_pinball"


def _checks(status="pass"):
    return {
        name: {
            "status": status,
            "metrics": {"check": name, "measured": True},
            "reason": None if status == "pass" else "fixture failure",
        }
        for name in REQUIRED_SMOKE_CHECKS
    }


def _artifacts(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    revision = get_dossier("chronos_bolt")["model_revision"]
    snapshot = tmp_path / revision
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}\n")
    values = {
        "model_snapshot": snapshot,
        "source_runtime": Path(
            importlib.metadata.distribution("chronos-forecasting")._path
        ).resolve(),
    }
    for name in sorted(
        REQUIRED_SMOKE_ARTIFACTS - {"model_snapshot", "source_runtime"}
    ):
        path = tmp_path / f"{name}.bin"
        path.write_bytes((name + "\n").encode())
        values[name] = path
    return values


def _evidence(tmp_path: Path, *, status="pass"):
    return build_route_smoke_evidence(
        route_key=ROUTE_KEY,
        executor_path=chronos_bolt.__file__,
        executor_entrypoint="native_loss/direct_quantiles",
        checks=_checks(status),
        artifacts=_artifacts(tmp_path),
        metrics={"fixture": True},
        created_utc="2026-07-18T00:00:00Z",
    )


def test_route_smoke_binds_exact_check_and_artifact_closure(tmp_path):
    evidence = _evidence(tmp_path)
    assert evidence["smoke_admitted"] is True
    assert set(evidence["checks"]) == set(REQUIRED_SMOKE_CHECKS)
    assert set(evidence["artifacts"]) == REQUIRED_SMOKE_ARTIFACTS
    assert validate_route_smoke_evidence(evidence) == evidence
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(evidence))
    assert load_route_smoke_evidence(path) == evidence


def test_route_smoke_rejects_missing_check_and_artifact(tmp_path):
    checks = _checks()
    checks.pop("data_lineage")
    with pytest.raises(NativeContractError, match="check closure"):
        build_route_smoke_evidence(
            route_key=ROUTE_KEY,
            executor_path=chronos_bolt.__file__,
            executor_entrypoint="native_loss/direct_quantiles",
            checks=checks,
            artifacts=_artifacts(tmp_path),
            metrics={},
        )
    artifacts = _artifacts(tmp_path / "second")
    artifacts.pop("raw_checks")
    with pytest.raises(NativeContractError, match="artifact closure"):
        build_route_smoke_evidence(
            route_key=ROUTE_KEY,
            executor_path=chronos_bolt.__file__,
            executor_entrypoint="native_loss/direct_quantiles",
            checks=_checks(),
            artifacts=artifacts,
            metrics={},
        )


def test_route_smoke_reopens_artifacts_and_rejects_tamper(tmp_path):
    evidence = _evidence(tmp_path)
    Path(evidence["artifacts"]["raw_checks"]["path"]).write_bytes(b"tampered\n")
    with pytest.raises(NativeContractError, match="drifted"):
        validate_route_smoke_evidence(evidence)


def test_failed_smoke_is_valid_evidence_but_never_admitted(tmp_path):
    evidence = _evidence(tmp_path, status="fail")
    assert evidence["smoke_admitted"] is False
    assert evidence["pilot_admitted"] is False
    assert evidence["training_admitted"] is False
    assert validate_route_smoke_evidence(evidence) == evidence


def test_route_smoke_rejects_model_snapshot_revision_substitution(tmp_path):
    artifacts = _artifacts(tmp_path)
    wrong = tmp_path / "wrong-revision"
    wrong.mkdir()
    (wrong / "weights.bin").write_bytes(b"wrong")
    artifacts["model_snapshot"] = wrong
    with pytest.raises(NativeContractError, match="snapshot path"):
        build_route_smoke_evidence(
            route_key=ROUTE_KEY,
            executor_path=chronos_bolt.__file__,
            executor_entrypoint="native_loss/direct_quantiles",
            checks=_checks(),
            artifacts=artifacts,
            metrics={},
        )


def test_route_smoke_rejects_integrity_forgery(tmp_path):
    evidence = _evidence(tmp_path)
    forged = deepcopy(evidence)
    forged["smoke_admitted"] = False
    with pytest.raises(NativeContractError, match="integrity"):
        validate_route_smoke_evidence(forged)
