import hashlib
import json
from pathlib import Path
import subprocess

from futures_foundation.finetune.artifact_inventory import (
    SCHEMA_VERSION, build_frozen_inventory, sha256_file,
)


def _git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def test_inventory_verifies_frozen_artifacts_and_fails_on_mutation(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "requirements.txt").write_text("numpy\n")
    (repo / "setup.py").write_text("# test\n")
    (repo / "source.py").write_text("VALUE = 1\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "test")

    windows = tmp_path / "windows.npz"
    embedding = tmp_path / "embedding.npz"
    checkpoint = tmp_path / "stage1.pt"
    windows.write_bytes(b"windows")
    embedding.write_bytes(b"embedding")
    checkpoint.write_bytes(b"checkpoint")
    results = {
        "schema_version": "ffm_cross_family_representation_probe_v1",
        "created_utc": "2026-07-16T00:00:00+00:00",
        "oos_read": False,
        "windows": {
            "path": str(windows), "sha256": sha256_file(windows), "fingerprint": "abc",
        },
        "probe": {"fold_contract_sha256": "fold"},
        "results": {
            "arm:vanilla": {
                "stage": "vanilla", "checkpoint": None, "checkpoint_sha256": None,
                "embedding": {"path": str(embedding), "sha256": sha256_file(embedding)},
            },
            "arm:stage1": {
                "stage": "stage1", "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "embedding": {"path": str(tmp_path / "embedding2.npz"), "sha256": None},
            },
        },
    }
    (tmp_path / "embedding2.npz").write_bytes(b"embedding2")
    results["results"]["arm:stage1"]["embedding"]["sha256"] = sha256_file(
        tmp_path / "embedding2.npz"
    )
    results_path = tmp_path / "results.json"
    results_path.write_text(json.dumps(results))

    inventory = build_frozen_inventory(results_path, repo=repo)
    assert inventory["schema_version"] == SCHEMA_VERSION
    assert inventory["passed"]
    assert inventory["counts"] == {"embeddings": 2, "checkpoints": 1, "artifacts": 4}
    assert inventory["source"]["dirty"] is False

    checkpoint.write_bytes(b"mutated")
    failed = build_frozen_inventory(results_path, repo=repo)
    assert not failed["passed"]
    assert any("checkpoint hash mismatch" in error for error in failed["errors"])
