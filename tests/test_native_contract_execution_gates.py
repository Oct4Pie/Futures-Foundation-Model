from pathlib import Path
from types import SimpleNamespace

import pytest

from futures_foundation.finetune.native_contracts import NativeContractError, get_arm
from scripts import benchmark_foundation_representations
from scripts import benchmark_kronos
from scripts import train_control_foundation_stages
from scripts import train_ssl_local


MODEL_EXECUTION_MARKERS = (
    "from_pretrained", "embed_windows", "MOMENTPipeline", "Toto2Model",
    "AutoModel", "predict_quantiles", ".fit(", "train(",
)
UNGATED_PRE_ADMISSION_EVIDENCE_TOOLS = {
    "check_mantis_stage0_parity.py",
    "smoke_chronos_bolt_route.py",
    "smoke_chronos_v1_route.py",
    "smoke_moment_reconstruction_route.py",
    "smoke_kronos_tokenizer_route.py",
    "smoke_kronos_predictor_route.py",
}


TRAINING_ENTRY_POINTS = (
    "scripts/train_kronos_tournament.py",
    "scripts/train_kronos_contrastive.py",
    "scripts/train_moment_tournament.py",
    "scripts/train_moment_contrastive.py",
    "scripts/train_moment_forecast.py",
    "scripts/train_chronos_tournament.py",
    "scripts/train_chronos_contrastive.py",
    "scripts/train_timesfm_tournament.py",
    "scripts/train_timesfm_contrastive.py",
    "scripts/train_ttm_tournament.py",
    "scripts/train_ttm_contrastive.py",
    "scripts/train_moirai2_tournament.py",
    "scripts/train_moirai2_contrastive.py",
    "scripts/train_control_foundation_stages.py",
    "scripts/train_ssl_local.py",
)


def test_only_declared_pre_admission_evidence_tools_may_load_models_without_admission():
    ungated = set()
    for path in Path("scripts").glob("*.py"):
        source = path.read_text(encoding="utf-8")
        if any(marker in source for marker in MODEL_EXECUTION_MARKERS) and "admission" not in source:
            ungated.add(path.name)
    assert ungated <= UNGATED_PRE_ADMISSION_EVIDENCE_TOOLS
    for name in UNGATED_PRE_ADMISSION_EVIDENCE_TOOLS - {"check_mantis_stage0_parity.py"}:
        source = (Path("scripts") / name).read_text(encoding="utf-8")
        assert "synthetic" in source.lower(), f"{name} is not a synthetic evidence tool"
        assert "market_data_read" in source and "False" in source
        assert "build_route_smoke_evidence" in source
        assert "load_adaptation_data" not in source


@pytest.mark.parametrize("path", TRAINING_ENTRY_POINTS)
def test_every_foundation_training_entry_point_has_executable_admission_gate(path):
    source = Path(path).read_text(encoding="utf-8")
    assert "admission" in source
    assert "require_admission_from_args" in source


def test_control_trainer_fails_before_source_or_data(monkeypatch):
    monkeypatch.setattr(
        train_control_foundation_stages,
        "_validate_source",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("source touched")),
    )
    monkeypatch.setattr(
        train_control_foundation_stages,
        "load_adaptation_data",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("data touched")),
    )
    with pytest.raises(NativeContractError, match="optimizer entrypoint"):
        train_control_foundation_stages.train(SimpleNamespace(family="toto2_22m"))


def test_mantis_legacy_trainer_fails_before_model_or_data():
    with pytest.raises(NativeContractError, match="blocked without --admission-report"):
        train_ssl_local.run(SimpleNamespace(model_version="v2"))


def test_historical_representation_extractor_fails_before_windows():
    with pytest.raises(NativeContractError, match="blocked without --admission-report"):
        benchmark_foundation_representations.extract(SimpleNamespace(arm="ttm_r2"))


def test_kronos_direct_benchmark_fails_before_source_or_data(monkeypatch):
    arm = get_arm("kronos_mini")
    monkeypatch.setattr(
        benchmark_kronos,
        "_validate_source",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("source touched")),
    )
    args = SimpleNamespace(
        arm=arm.key,
        model_id=arm.model_id,
        model_revision=arm.model_revision,
        source_revision=arm.source_revision,
        tokenizer_id=arm.tokenizer_id,
        tokenizer_revision=arm.tokenizer_revision,
    )
    with pytest.raises(NativeContractError, match="blocked without --admission-report"):
        benchmark_kronos.benchmark(args)
