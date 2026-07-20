from copy import deepcopy

import pytest

from futures_foundation.finetune.native_contracts import NativeContractError, content_sha256
from futures_foundation.finetune import native_downstream_screen as screen


def _paths():
    return {route: f"/{index}/report.json" for index, route in enumerate(sorted(screen.COMMON_INFORMATION_ROUTES))}


def _report(route: str, *, survived: bool = False):
    return {
        "route_key": route,
        "report_identity": {
            "path": f"/{route}/report.json", "sha256": "1" * 64, "bytes": 10,
        },
        "feature_table": {
            "path": f"/{route}/features.npz", "sha256": "2" * 64,
            "content_fingerprint": "3" * 64, "feature_count": 32,
            "feature_kind": "fixture", "information_view": "common_information_512_v1",
        },
        "verified_feature_pilot_evidence": {
            "path": f"/{route}/pilot.json", "evidence_sha256": "4" * 64,
        },
        "screen_verdict": {
            "model_control_wins": 3,
            "incremental_point_wins": 3 if survived else 1,
            "residual_bonferroni_ci_wins": 1 if survived else 0,
            "primary_point_degradations": 0 if survived else 4,
            "downstream_screen_survived": survived,
            "nonlinear_sensitivity_funded": survived,
        },
    }


def test_screen_collection_requires_exact_route_closure(monkeypatch):
    monkeypatch.setattr(
        screen, "load_incremental_screen_report",
        lambda path: _report(next(route for route, value in _paths().items() if value == str(path))),
    )
    collection = screen.build_screen_collection(_paths())
    assert collection["counts"] == {
        "reports_completed": 8,
        "downstream_screen_survived": 0,
        "nonlinear_sensitivity_funded": 0,
        "full_training_admitted": 0,
    }
    assert collection["surviving_routes"] == []
    assert collection["full_training_admitted"] is False
    assert collection["oos_admitted"] is False
    assert collection["live_trading_ready"] is False
    assert collection["collection_sha256"] == content_sha256({
        key: value for key, value in collection.items() if key != "collection_sha256"
    })


def test_screen_collection_records_only_measured_survivors(monkeypatch):
    paths = _paths()
    selected = sorted(screen.COMMON_INFORMATION_ROUTES)[0]
    monkeypatch.setattr(
        screen, "load_incremental_screen_report",
        lambda path: _report(
            next(route for route, value in paths.items() if value == str(path)),
            survived=(str(path) == paths[selected]),
        ),
    )
    collection = screen.build_screen_collection(paths)
    assert collection["surviving_routes"] == [selected]
    assert collection["nonlinear_sensitivity_funded_routes"] == [selected]
    assert collection["counts"]["downstream_screen_survived"] == 1
    assert collection["full_training_admitted"] is False


def test_screen_collection_rejects_missing_route_and_forgery(monkeypatch):
    paths = _paths()
    monkeypatch.setattr(
        screen, "load_incremental_screen_report",
        lambda path: _report(next(route for route, value in paths.items() if value == str(path))),
    )
    incomplete = dict(paths)
    incomplete.pop(next(iter(incomplete)))
    with pytest.raises(NativeContractError, match="exact common-information route set"):
        screen.build_screen_collection(incomplete)

    collection = screen.build_screen_collection(paths)
    forged = deepcopy(collection)
    forged["full_training_admitted"] = True
    payload = dict(forged)
    payload.pop("collection_sha256")
    forged["collection_sha256"] = content_sha256(payload)
    with pytest.raises(NativeContractError, match="stale|non-canonical"):
        screen.validate_screen_collection(forged, report_paths=paths)
