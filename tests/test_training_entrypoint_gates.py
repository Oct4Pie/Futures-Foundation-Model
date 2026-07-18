"""Phase-A regressions for every known optimizer-capable public training surface.

These tests deliberately do not authorize a route.  Until raw route-bundle evidence is
implemented, the parent API must fail before data materialization, subprocess creation,
model construction, or optimizer use.  Lower-level training mechanics remain testable only
when their tests explicitly replace the central blocker; no runtime bypass exists here.
"""
from pathlib import Path

import pytest

from futures_foundation.finetune.native_contracts import NativeContractError


BLOCKED = "optimizer entrypoint"


def test_ssl_orchestrator_fails_before_configuration_or_data(monkeypatch):
    from futures_foundation.finetune import ssl

    monkeypatch.setattr(
        ssl, "_load_assemble", lambda: (_ for _ in ()).throw(AssertionError("data loaded")),
    )
    with pytest.raises(NativeContractError, match=BLOCKED):
        ssl.loop_ssl()


def test_chronos_bolt_parent_fails_before_window_build(monkeypatch):
    from futures_foundation.extractors.chronos import bolt_finetune

    monkeypatch.setattr(
        bolt_finetune, "_build_windows",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("windows built")),
    )
    with pytest.raises(NativeContractError, match=BLOCKED):
        bolt_finetune.run(smoke=True)


def test_chronos_supervised_and_ssl_entrypoints_fail_before_inputs_are_used():
    from futures_foundation.extractors.chronos import finetune

    with pytest.raises(NativeContractError, match=BLOCKED):
        finetune.train(None, None)
    with pytest.raises(NativeContractError, match=BLOCKED):
        finetune._train_ssl(None, None, 0)


def test_mantis_classifier_parent_fails_before_serialization_or_subprocess(monkeypatch):
    from futures_foundation.finetune.classifiers.mantis.classifier import MantisClassifier

    monkeypatch.setattr(
        "subprocess.run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("worker spawned")),
    )
    with pytest.raises(NativeContractError, match=BLOCKED):
        MantisClassifier().fit_predict(None, None, None, None, None)


def test_torch_optimizer_surfaces_fail_before_object_state_or_arrays_are_used():
    pytest.importorskip("torch")
    from futures_foundation.extractors.chronos import shape_adapter
    from futures_foundation.finetune.classifiers.mantis import _torch as mantis_torch
    from futures_foundation.finetune.pretext._torch.common import BaseTrainer

    with pytest.raises(NativeContractError, match=BLOCKED):
        shape_adapter.fit_and_infer(None, None, None)
    with pytest.raises(NativeContractError, match=BLOCKED):
        mantis_torch.fit_predict_torch(None, None, None, None, None)
    with pytest.raises(NativeContractError, match=BLOCKED):
        object.__new__(BaseTrainer).fit()


def test_all_legacy_mantis_ssl_scripts_reach_the_gated_orchestrator():
    repo = Path(__file__).resolve().parents[1]
    scripts = sorted((repo / "scripts").glob("mantis_ssl_*.py"))
    assert [path.name for path in scripts] == [
        "mantis_ssl_contrastive.py",
        "mantis_ssl_electra.py",
        "mantis_ssl_mixture.py",
        "mantis_ssl_nextleg.py",
        "mantis_ssl_pretrain.py",
        "mantis_ssl_seq2seq.py",
        "mantis_ssl_spanrecon.py",
    ]
    for path in scripts:
        source = path.read_text(encoding="utf-8")
        assert "ssl.loop_ssl(" in source, f"{path.name} bypasses the gated orchestrator"


def test_optimizer_call_sites_remain_covered_by_gated_entrypoints():
    """Force conscious review when a new direct backward/step surface is added."""
    repo = Path(__file__).resolve().parents[1]
    expected = {
        "futures_foundation/extractors/chronos/finetune.py",
        "futures_foundation/extractors/chronos/shape_adapter.py",
        "futures_foundation/finetune/classifiers/mantis/_torch.py",
        "futures_foundation/finetune/pretext/_torch/common.py",
    }
    found = {
        path.relative_to(repo).as_posix()
        for path in (repo / "futures_foundation").rglob("*.py")
        if ".backward(" in path.read_text(encoding="utf-8")
        or ".step(" in path.read_text(encoding="utf-8")
    }
    assert found == expected
    for relative in expected:
        assert "block_unadmitted_optimizer(" in (repo / relative).read_text(
            encoding="utf-8"
        )
