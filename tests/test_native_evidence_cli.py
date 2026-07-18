import json
import sys

from futures_foundation.finetune import native_evidence_cli as cli


def test_verify_defaults_to_strict_external_artifact_verification(monkeypatch, capsys):
    seen = {}

    def fake_verify(bundle, *, registry_path, verify_external_artifacts):
        seen.update(
            bundle=bundle,
            registry_path=registry_path,
            verify_external_artifacts=verify_external_artifacts,
        )
        return ({"arm_key": "test"}, {"status": "pass"})

    monkeypatch.setattr(cli, "verify_parity_bundle", fake_verify)
    monkeypatch.setattr(sys, "argv", ["ffm-native-parity-evidence", "verify", "bundle"])
    assert cli.main() == 0
    assert seen["verify_external_artifacts"] is True
    assert json.loads(capsys.readouterr().out) == {"arm_key": "test"}


def test_archive_only_verify_is_explicit_and_scope_labeled(monkeypatch, capsys):
    seen = {}

    def fake_verify(bundle, *, registry_path, verify_external_artifacts):
        seen.update(
            bundle=bundle,
            registry_path=registry_path,
            verify_external_artifacts=verify_external_artifacts,
        )
        return ({"arm_key": "test"}, {"status": "pass"})

    monkeypatch.setattr(cli, "verify_parity_bundle", fake_verify)
    monkeypatch.setattr(
        sys, "argv",
        ["ffm-native-parity-evidence", "verify", "bundle", "--archive-only"],
    )
    assert cli.main() == 0
    assert seen["verify_external_artifacts"] is False
    output = json.loads(capsys.readouterr().out)
    assert output["verification_scope"] == "archive_only"
    assert output["external_artifacts_verified"] is False
    assert output["manifest"] == {"arm_key": "test"}
    assert "does not authorize" in output["warning"]
