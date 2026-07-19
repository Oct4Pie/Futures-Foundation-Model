"""Fail-closed inventory for frozen downstream-comparison inputs."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess


SCHEMA_VERSION = "ffm_frozen_downstream_inventory_v1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True,
    )
    return result.stdout


def source_tree_fingerprint(repo: str | Path) -> dict[str, object]:
    """Hash tracked and non-ignored source files while excluding data/result environments."""
    repo = Path(repo).resolve()
    commit = _git(repo, "rev-parse", "HEAD").strip()
    status = _git(repo, "status", "--porcelain=v1", "--untracked-files=all")
    candidates = _git(
        repo, "ls-files", "--cached", "--others", "--exclude-standard",
    ).splitlines()
    excluded_roots = (".venv/", "data/", "output/")
    files = sorted(path for path in candidates if not path.startswith(excluded_roots))
    digest = hashlib.sha256()
    included = 0
    for relative in files:
        path = repo / relative
        if not path.is_file():
            continue
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
        included += 1
    return {
        "git_commit": commit,
        "dirty": bool(status),
        "status_sha256": hashlib.sha256(status.encode()).hexdigest(),
        "source_tree_sha256": digest.hexdigest(),
        "source_files": included,
    }


def _verify_file(
    *, path: str | Path, expected_sha256: str | None, kind: str, key: str,
    errors: list[str],
) -> dict[str, object]:
    path = Path(path)
    row: dict[str, object] = {
        "kind": kind, "key": key, "path": str(path.resolve()), "exists": path.is_file(),
        "expected_sha256": expected_sha256,
    }
    if not path.is_file():
        errors.append(f"{kind} missing for {key}: {path}")
        row["actual_sha256"] = None
        return row
    actual = sha256_file(path)
    row.update({"actual_sha256": actual, "bytes": path.stat().st_size})
    if expected_sha256 is not None and actual != expected_sha256:
        errors.append(f"{kind} hash mismatch for {key}: expected {expected_sha256}, got {actual}")
    return row


def build_frozen_inventory(
    results_path: str | Path,
    *,
    repo: str | Path,
) -> dict[str, object]:
    """Recompute every canonical window, embedding and checkpoint hash."""
    repo = Path(repo).resolve()
    results_path = Path(results_path).resolve()
    results = json.loads(results_path.read_text())
    errors: list[str] = []
    if results.get("schema_version") != "ffm_cross_family_representation_probe_v2":
        errors.append("unsupported representation-results schema")
    if results.get("probe", {}).get("target_semantics_version") != "ffm_causal_probe_targets_v2":
        errors.append("representation results lack the current target-semantics binding")
    if results.get("oos_read") is not False:
        errors.append("representation results do not attest oos_read=false")

    window = results.get("windows", {})
    artifacts = [
        _verify_file(
            path=window.get("path", ""), expected_sha256=window.get("sha256"),
            kind="windows", key="sealed_validation", errors=errors,
        )
    ]
    seen_paths: set[str] = set()
    checkpoint_count = embedding_count = 0
    for key, row in sorted(results.get("results", {}).items()):
        embedding = row.get("embedding") or {}
        embedding_path = str(Path(embedding.get("path", "")).resolve())
        if embedding_path in seen_paths:
            errors.append(f"duplicate embedding path: {embedding_path}")
        seen_paths.add(embedding_path)
        artifacts.append(_verify_file(
            path=embedding.get("path", ""), expected_sha256=embedding.get("sha256"),
            kind="embedding", key=key, errors=errors,
        ))
        embedding_count += 1
        checkpoint = row.get("checkpoint")
        checkpoint_sha = row.get("checkpoint_sha256")
        if checkpoint is None:
            if row.get("stage") != "vanilla" or checkpoint_sha is not None:
                errors.append(f"invalid checkpoint declaration for {key}")
            continue
        artifacts.append(_verify_file(
            path=checkpoint, expected_sha256=checkpoint_sha,
            kind="checkpoint", key=key, errors=errors,
        ))
        checkpoint_count += 1

    dependency_files = {}
    for relative in ("requirements.txt", "setup.py"):
        path = repo / relative
        dependency_files[relative] = sha256_file(path) if path.is_file() else None

    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": not errors,
        "errors": errors,
        "evidence_boundary": {
            "representation_oos_read": results.get("oos_read"),
            "legacy_confirmatory_globally_pristine": False,
        },
        "source": source_tree_fingerprint(repo),
        "dependencies": dependency_files,
        "representation_results": {
            "path": str(results_path),
            "sha256": sha256_file(results_path),
            "schema_version": results.get("schema_version"),
            "created_utc": results.get("created_utc"),
        },
        "window_fingerprint": window.get("fingerprint"),
        "fold_contract_sha256": results.get("probe", {}).get("fold_contract_sha256"),
        "counts": {
            "embeddings": embedding_count,
            "checkpoints": checkpoint_count,
            "artifacts": len(artifacts),
        },
        "artifacts": artifacts,
    }


def write_inventory(path: str | Path, inventory: dict[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(inventory, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


__all__ = [
    "SCHEMA_VERSION", "sha256_file", "source_tree_fingerprint",
    "build_frozen_inventory", "write_inventory",
]
