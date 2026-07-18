from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from futures_foundation.finetune import native_runtime_binding as binding
from scripts import extract_native_representations as representations
from scripts import predict_foundation_forecasts as forecasts


def test_representation_loader_paths_are_the_admission_paths(tmp_path):
    model = tmp_path / "model"
    source = tmp_path / "source"
    args = SimpleNamespace(model_snapshot=str(model), source_repo=str(source))

    assert representations._representation_artifact_paths(args) == {
        "model": model.resolve(),
        "source": source.resolve(),
    }


@pytest.mark.parametrize("arm", ["kronos_mini", "kronos_small"])
def test_kronos_forecast_couples_model_source_and_tokenizer(tmp_path, arm):
    args = SimpleNamespace(
        arm=arm,
        model_snapshot=tmp_path / "model",
        upstream_repo=tmp_path / "source",
        tokenizer_snapshot=tmp_path / "tokenizer",
        reference_model_snapshot=None,
    )

    assert forecasts._forecast_artifact_paths(args) == {
        "model": (tmp_path / "model").resolve(),
        "source": (tmp_path / "source").resolve(),
        "tokenizer": (tmp_path / "tokenizer").resolve(),
    }


def test_different_legacy_loader_alias_is_rejected(tmp_path):
    admitted = tmp_path / "admitted-source"
    different = tmp_path / "parallel-source"

    with pytest.raises(binding.NativeRuntimeBindingError, match="differs from"):
        binding.require_same_path(different, admitted.resolve(), "loader source")

    assert binding.require_same_path(admitted, admitted.resolve(), "loader source") == admitted.resolve()


def test_chronos_loader_uses_admitted_local_snapshot(monkeypatch, tmp_path):
    seen = {}

    class FakePipeline:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            seen["path"] = path
            seen["kwargs"] = kwargs
            return object()

    import chronos
    monkeypatch.setattr(chronos, "BaseChronosPipeline", FakePipeline)
    snapshot = (tmp_path / "admitted-model").resolve()
    args = SimpleNamespace(
        arm="chronos_bolt", device="cpu", native_artifacts={"model": snapshot}
    )

    representations._load_chronos_pipeline(args)

    assert seen["path"] == str(snapshot)
    assert seen["kwargs"]["local_files_only"] is True


def test_shadow_module_must_be_within_record_verified_package(tmp_path):
    verified_package = tmp_path / "site-packages" / "chronos"
    shadow = tmp_path / "shadow" / "chronos" / "__init__.py"
    shadow.parent.mkdir(parents=True)
    shadow.write_text("# shadow\n")

    with pytest.raises(binding.NativeRuntimeBindingError, match="outside admitted source"):
        binding.require_module_within(
            str(shadow), verified_package, "Chronos"
        )


def test_shadow_import_is_rejected_before_import(monkeypatch, tmp_path):
    shadow = tmp_path / "shadow" / "chronos" / "__init__.py"
    shadow.parent.mkdir(parents=True)
    shadow.write_text("raise RuntimeError('must not execute')\n")
    monkeypatch.syspath_prepend(str(tmp_path / "shadow"))
    monkeypatch.delitem(sys.modules, "chronos", raising=False)

    with pytest.raises(binding.NativeRuntimeBindingError, match="outside admitted source"):
        binding.require_import_origin(
            "chronos", tmp_path / "admitted" / "chronos", "Chronos"
        )


def test_timesfm_artifact_map_binds_reference_and_executed_source(tmp_path):
    args = SimpleNamespace(
        arm="timesfm25",
        model_snapshot=tmp_path / "model",
        upstream_repo=tmp_path / "reference-source",
        tokenizer_snapshot=None,
        reference_model_snapshot=tmp_path / "reference-model",
        execution_source=tmp_path / "transformers.dist-info",
    )
    assert forecasts._forecast_artifact_paths(args) == {
        "model": (tmp_path / "model").resolve(),
        "source": (tmp_path / "reference-source").resolve(),
        "reference_model": (tmp_path / "reference-model").resolve(),
        "execution_source": (tmp_path / "transformers.dist-info").resolve(),
    }


def test_moirai_artifact_map_has_no_unexecuted_aliases(tmp_path):
    args = SimpleNamespace(
        arm="moirai2_small",
        model_snapshot=tmp_path / "model",
        upstream_repo=tmp_path / "source",
        tokenizer_snapshot=None,
        reference_model_snapshot=None,
        execution_source=None,
    )
    assert forecasts._forecast_artifact_paths(args) == {
        "model": (tmp_path / "model").resolve(),
        "source": (tmp_path / "source").resolve(),
    }


def test_native_consumers_do_not_load_hub_ids_after_admission():
    representation_source = Path(representations.__file__).read_text()
    forecast_source = Path(forecasts.__file__).read_text()

    assert ".from_pretrained(arm.model_id" not in representation_source
    assert ".from_pretrained(arm.model_id" not in forecast_source
    assert 'str(args.native_artifacts["model"])' in representation_source
    assert 'str(args.native_artifacts["model"])' in forecast_source
