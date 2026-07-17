from scripts.audit_foundation_tournament import _select


def test_audit_selection_respects_equal_optuna_trial_budget():
    study = {
        "model": "ttm_r2",
        "trials": [
            {"number": number, "state": "COMPLETE", "values": [float(8 - number)]}
            for number in range(12)
        ],
    }
    assert _select(study, trial_budget=8)["number"] == 7
    assert _select(study)["number"] == 11


def test_mantis_selection_respects_equal_optuna_trial_budget():
    study = {
        "model": "mantis_v2",
        "trials": [
            {"number": 0, "state": "COMPLETE", "values": [0.1, 0.8]},
            {"number": 1, "state": "COMPLETE", "values": [0.2, 0.7]},
            {"number": 8, "state": "COMPLETE", "values": [0.9, 0.9]},
        ],
    }
    assert _select(study, trial_budget=2)["number"] == 1
