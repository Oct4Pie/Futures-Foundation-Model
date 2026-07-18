"""Reproducible, fail-closed native-parity evidence bundles.

This module does not decide whether a model is useful and does not authorize training.
It binds a real parity command to a deterministic input fixture, pinned registry identity,
model/source/tokenizer artifacts, raw outputs, logs, and the exact check results.  Verified
bundles can be aggregated into a *candidate* native-contract evidence document.  Installing
that candidate remains a separately reviewed repository change.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit

import numpy as np

from .native_contracts import (
    ADMITTED_STATUSES,
    EVIDENCE_SCHEMA,
    MANDATORY_TECHNICAL_CHECKS,
    NativeContractError,
    REGISTRY_PATH,
    _validate_evidence,
    canonical_json,
    content_sha256,
    dossier_sha256,
    file_sha256,
    get_dossier,
    load_registry,
    registry_sha256,
)


FIXTURE_SCHEMA = "ffm_native_parity_fixture_v1"
RESULT_SCHEMA = "ffm_native_parity_result_v2"
LEGACY_RESULT_SCHEMA = "ffm_native_parity_result_v1"
BUNDLE_SCHEMA = "ffm_native_parity_bundle_v1"
AGGREGATE_SCHEMA = "ffm_native_parity_aggregate_v1"
DEFAULT_SEED = 20260717
DEFAULT_BATCH_SIZE = 4
DEFAULT_CONTEXT_LENGTH = 512
DEFAULT_CHANNELS = ("open", "high", "low", "close", "volume")


class NativeEvidenceError(NativeContractError):
    """Raised when raw parity evidence is missing, stale, or tampered with."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(value) + b"\n")


def _read_json(path: Path, field: str) -> dict[str, Any]:
    if not path.is_file():
        raise NativeEvidenceError(f"{field} not found: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NativeEvidenceError(f"{field} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise NativeEvidenceError(f"{field} must be a JSON object: {path}")
    return value


def _safe_relative(path: str, field: str) -> Path:
    value = Path(path)
    if value.is_absolute() or ".." in value.parts or value == Path("."):
        raise NativeEvidenceError(f"{field} must be a safe relative path: {path!r}")
    return value


def _file_description(path: Path, *, display_path: str) -> dict[str, Any]:
    if path.is_symlink():
        if not path.exists():
            raise NativeEvidenceError(f"artifact contains a broken symlink: {path}")
        resolved = path.resolve(strict=True)
        if not resolved.is_file():
            raise NativeEvidenceError(f"artifact symlink must resolve to a file: {path}")
        return {
            "path": display_path,
            "kind": "file_symlink",
            "link_target": os.readlink(path),
            "resolved_sha256": file_sha256(resolved),
            "sha256": file_sha256(resolved),
            "size_bytes": resolved.stat().st_size,
        }
    if not path.is_file():
        raise NativeEvidenceError(f"artifact is not a regular file: {path}")
    return {
        "path": display_path,
        "kind": "file",
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _git(
    root: Path, *args: str, text: bool = False, check: bool = True
) -> subprocess.CompletedProcess[Any]:
    try:
        return subprocess.run(
            ["git", "-C", str(root), *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=check,
            text=text,
        )
    except FileNotFoundError as exc:
        raise NativeEvidenceError("git executable is required to bind a source checkout") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() if text and isinstance(exc.stderr, str) else exc.stderr
        raise NativeEvidenceError(f"git {' '.join(args)} failed for {root}: {detail}") from exc


def _normalize_git_origin(value: str) -> str:
    """Normalize common HTTPS/SSH Git remotes without embedding credentials."""
    remote = value.strip()
    if not remote:
        return ""
    match = re.fullmatch(r"(?:[^@/]+@)?([^:/]+):(.+)", remote)
    if match and "://" not in remote:
        remote = f"https://{match.group(1)}/{match.group(2)}"
    parsed = urlsplit(remote)
    if parsed.scheme in {"ssh", "git"}:
        scheme = "https"
    else:
        scheme = parsed.scheme.lower()
    host = (parsed.hostname or "").lower()
    port = f":{parsed.port}" if parsed.port else ""
    if host:
        path = parsed.path.rstrip("/")
        if path.endswith(".git"):
            path = path[:-4]
        return urlunsplit((scheme or "https", host + port, path, "", ""))
    # Local/file origins are useful for tests but remain explicitly machine-local.
    path = remote.rstrip("/")
    return path[:-4] if path.endswith(".git") else path


def _git_checkout_description(
    root: Path,
    *,
    display_path: str,
    untracked_policy: str,
) -> dict[str, Any]:
    if untracked_policy not in {"ignore", "reject"}:
        raise NativeEvidenceError(
            f"git untracked policy must be 'ignore' or 'reject', got {untracked_policy!r}"
        )
    top = Path(_git(root, "rev-parse", "--show-toplevel", text=True).stdout.strip()).resolve()
    if top != root.resolve():
        raise NativeEvidenceError(
            f"Git checkout artifact must be its repository root: supplied={root}, root={top}"
        )
    head = _git(root, "rev-parse", "HEAD", text=True).stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", head):
        raise NativeEvidenceError(f"Git checkout has an invalid HEAD revision: {head!r}")
    tracked_status = _git(
        root, "status", "--porcelain=v1", "--untracked-files=no", text=True
    ).stdout
    if tracked_status.strip():
        raise NativeEvidenceError(
            f"Git checkout has tracked/index drift and cannot be sealed: {root}"
        )
    untracked_raw = _git(
        root, "ls-files", "-z", "--others", "--exclude-standard"
    ).stdout
    untracked = [item.decode("utf-8", "surrogateescape") for item in untracked_raw.split(b"\0") if item]
    if untracked_policy == "reject" and untracked:
        raise NativeEvidenceError(
            f"Git checkout has untracked files under reject policy: {untracked[:8]}"
        )
    origin_process = _git(root, "remote", "get-url", "origin", text=True, check=False)
    origin = (
        _normalize_git_origin(origin_process.stdout)
        if origin_process.returncode == 0 else ""
    )
    stage_raw = _git(root, "ls-files", "--stage", "-z").stdout
    entries: list[dict[str, Any]] = []
    for raw in stage_raw.split(b"\0"):
        if not raw:
            continue
        try:
            metadata, raw_path = raw.split(b"\t", 1)
            mode, blob, stage = metadata.decode("ascii").split()
        except ValueError as exc:
            raise NativeEvidenceError(f"cannot parse git index entry in {root}") from exc
        path = raw_path.decode("utf-8", "surrogateescape")
        if stage != "0":
            raise NativeEvidenceError(f"Git checkout has an unmerged index entry: {path}")
        if mode == "160000":
            entries.append({"path": path, "mode": mode, "gitlink_revision": blob})
            continue
        content = _git(root, "cat-file", "blob", f":{path}").stdout
        entries.append({
            "path": path,
            "mode": mode,
            "git_blob": blob,
            "sha256": hashlib.sha256(content).hexdigest(),
            "size_bytes": len(content),
        })
    if not entries:
        raise NativeEvidenceError(f"Git checkout contains no tracked files: {root}")
    identity = {
        "head_revision": head,
        "origin": origin,
        "untracked_policy": untracked_policy,
        "entries": entries,
    }
    return {
        "path": display_path,
        "kind": "git_checkout",
        **identity,
        "sha256": content_sha256(identity),
        "file_count": len(entries),
        "size_bytes": sum(item.get("size_bytes", 0) for item in entries),
    }


def _tree_description(
    path: str | Path,
    *,
    relative_to: Path | None = None,
    git_untracked_policy: str = "ignore",
) -> dict[str, Any]:
    """Hash a file or tree, including HF-style file symlinks by resolved bytes.

    Directory symlinks are rejected: following them can escape or duplicate a snapshot tree.
    File symlinks record both their literal link target and the resolved content digest.
    """
    supplied = Path(path).expanduser().absolute()
    display = (
        supplied.relative_to(relative_to.resolve()).as_posix()
        if relative_to is not None and supplied.is_relative_to(relative_to.resolve())
        else str(supplied)
    )
    if supplied.is_symlink():
        return _file_description(supplied, display_path=display)
    root = supplied.resolve()
    if not root.exists():
        raise NativeEvidenceError(f"bound artifact does not exist: {root}")
    if root.is_file():
        return _file_description(root, display_path=display)
    if not root.is_dir():
        raise NativeEvidenceError(f"unsupported bound artifact type: {root}")
    if root.name.endswith(".dist-info"):
        from .native_parity_runtime import validate_distribution_record

        try:
            validate_distribution_record(root)
        except RuntimeError as exc:
            raise NativeEvidenceError(
                f"installed distribution source failed RECORD validation: {exc}"
            ) from exc
    if (root / ".git").exists():
        return _git_checkout_description(
            root, display_path=display, untracked_policy=git_untracked_policy
        )
    entries: list[dict[str, Any]] = []
    for candidate in sorted(root.rglob("*")):
        if candidate.is_symlink():
            if not candidate.exists():
                raise NativeEvidenceError(f"artifact tree contains a broken symlink: {candidate}")
            if candidate.resolve(strict=True).is_dir():
                raise NativeEvidenceError(
                    f"artifact tree contains a directory symlink: {candidate}"
                )
            item = _file_description(
                candidate, display_path=candidate.relative_to(root).as_posix()
            )
            entries.append(item)
            continue
        if not candidate.is_file():
            continue
        entries.append({
            "path": candidate.relative_to(root).as_posix(),
            "sha256": file_sha256(candidate),
            "size_bytes": candidate.stat().st_size,
        })
    if not entries:
        raise NativeEvidenceError(f"artifact directory is empty: {root}")
    return {
        "path": display,
        "kind": "directory",
        "sha256": content_sha256(entries),
        "file_count": len(entries),
        "size_bytes": sum(item["size_bytes"] for item in entries),
        "entries": entries,
    }


def execution_code_description(path: str | Path) -> dict[str, Any]:
    """Hash the stable executable Python surface without registry/output self-reference."""
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise NativeEvidenceError(f"execution-code root is not a directory: {root}")
    if (root / ".git").exists():
        untracked_raw = _git(
            root, "ls-files", "-z", "--others", "--exclude-standard"
        ).stdout
        untracked_code = sorted(
            item.decode("utf-8", "surrogateescape")
            for item in untracked_raw.split(b"\0")
            if item
            and item.decode("utf-8", "surrogateescape").endswith(".py")
            and item.decode("utf-8", "surrogateescape").startswith(
                ("futures_foundation/", "scripts/")
            )
        )
        if untracked_code:
            raise NativeEvidenceError(
                "execution-code roots contain untracked Python sources: "
                f"{untracked_code[:8]}"
            )
        raw_entries = _git_checkout_description(
            root, display_path=str(root), untracked_policy="ignore"
        )["entries"]
        entries = [
            {
                "path": item["path"],
                "sha256": item.get("sha256"),
                "size_bytes": item.get("size_bytes"),
            }
            for item in raw_entries
            if str(item.get("path", "")).endswith(".py")
            and str(item.get("path", "")).startswith(("futures_foundation/", "scripts/"))
        ]
    else:
        entries = []
        for relative_root in (Path("futures_foundation"), Path("scripts")):
            code_root = root / relative_root
            if not code_root.is_dir() and root.name == "futures_foundation" and relative_root.name == root.name:
                code_root = root
            if not code_root.is_dir():
                continue
            for candidate in sorted(code_root.rglob("*.py")):
                if not candidate.is_file() or candidate.is_symlink():
                    raise NativeEvidenceError(
                        f"execution-code manifest contains an unsafe path: {candidate}"
                    )
                entries.append({
                    "path": candidate.relative_to(root).as_posix(),
                    "sha256": file_sha256(candidate),
                    "size_bytes": candidate.stat().st_size,
                })
    entries.sort(key=lambda item: item["path"])
    if not entries:
        raise NativeEvidenceError("execution-code manifest contains no Python sources")
    return {
        "path": str(root),
        "kind": "execution_code",
        "sha256": content_sha256(entries),
        "file_count": len(entries),
        "size_bytes": sum(int(item["size_bytes"]) for item in entries),
        "entries": entries,
    }


def _verify_tree(
    description: Mapping[str, Any], field: str, *, relative_to: Path | None = None
) -> None:
    path = description.get("path")
    if not isinstance(path, str):
        raise NativeEvidenceError(f"{field}.path is missing")
    supplied = Path(path)
    if not supplied.is_absolute():
        if relative_to is None:
            raise NativeEvidenceError(f"{field}.path is relative without a bundle root")
        supplied = relative_to / supplied
    actual = (
        execution_code_description(supplied)
        if description.get("kind") == "execution_code"
        else _tree_description(
            supplied,
            relative_to=relative_to,
            git_untracked_policy=str(description.get("untracked_policy", "ignore")),
        )
    )
    for key in ("kind", "sha256", "size_bytes"):
        if actual.get(key) != description.get(key):
            raise NativeEvidenceError(
                f"{field} drifted for {key}: expected {description.get(key)!r}, "
                f"got {actual.get(key)!r}"
            )
    if actual.get("link_target") != description.get("link_target"):
        raise NativeEvidenceError(f"{field} symlink target drifted")
    if actual.get("resolved_sha256") != description.get("resolved_sha256"):
        raise NativeEvidenceError(f"{field} symlink content drifted")
    if actual["kind"] == "git_checkout":
        for key in ("head_revision", "origin", "untracked_policy", "entries"):
            if actual.get(key) != description.get(key):
                raise NativeEvidenceError(f"{field} Git checkout {key} drifted")
    if actual["kind"] == "directory" and actual.get("entries") != description.get("entries"):
        raise NativeEvidenceError(f"{field} directory members drifted")
    if actual["kind"] == "execution_code" and actual.get("entries") != description.get("entries"):
        raise NativeEvidenceError(f"{field} execution-code members drifted")


def create_shared_fixture(
    directory: str | Path,
    *,
    seed: int = DEFAULT_SEED,
    batch_size: int = DEFAULT_BATCH_SIZE,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
) -> dict[str, Any]:
    """Write the deterministic finite OHLCV fixture used by every native parity run."""
    if batch_size < 2 or context_length < 16:
        raise NativeEvidenceError("fixture requires batch_size>=2 and context_length>=16")
    target = Path(directory).resolve()
    target.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(seed))
    innovations = rng.normal(0.0, 0.0018, (batch_size, context_length)).astype(np.float64)
    close = 100.0 * np.exp(np.cumsum(innovations, axis=1))
    open_ = np.concatenate((close[:, :1], close[:, :-1]), axis=1)
    spread = np.abs(rng.normal(0.0008, 0.00025, close.shape)) * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = rng.lognormal(mean=8.0, sigma=0.45, size=close.shape)
    values = np.stack((open_, high, low, close, volume), axis=-1).astype("<f4")
    if not (
        np.isfinite(values).all()
        and np.all(values[:, :, 1] >= np.maximum(values[:, :, 0], values[:, :, 3]))
        and np.all(values[:, :, 2] <= np.minimum(values[:, :, 0], values[:, :, 3]))
        and np.all(values[:, :, 4] > 0)
    ):
        raise NativeEvidenceError("internal fixture construction invariant failed")

    values_path = target / "ohlcv_f32.npy"
    np.save(values_path, values, allow_pickle=False)
    start_ns = np.datetime64("2024-01-02T14:30:00", "ns").astype(np.int64)
    timestamps = (
        start_ns
        + np.arange(context_length, dtype=np.int64)[None, :] * 60_000_000_000
        + np.arange(batch_size, dtype=np.int64)[:, None]
        * context_length
        * 60_000_000_000
    )
    timestamps_path = target / "timestamps_ns_i64.npy"
    np.save(timestamps_path, timestamps.astype("<i8"), allow_pickle=False)
    manifest = {
        "schema_version": FIXTURE_SCHEMA,
        "generator": "numpy_pcg64_ohlcv_v1",
        "seed": int(seed),
        "batch_size": int(batch_size),
        "context_length": int(context_length),
        "channels": list(DEFAULT_CHANNELS),
        "values": {
            "path": values_path.name,
            "shape": list(values.shape),
            "dtype": str(values.dtype),
            "sha256": file_sha256(values_path),
        },
        "timestamps": {
            "path": timestamps_path.name,
            "shape": list(timestamps.shape),
            "dtype": str(timestamps.dtype),
            "sha256": file_sha256(timestamps_path),
        },
    }
    manifest["fixture_sha256"] = content_sha256(manifest)
    _write_json(target / "fixture_manifest.json", manifest)
    return manifest


def verify_shared_fixture(directory: str | Path) -> dict[str, Any]:
    target = Path(directory).resolve()
    manifest = _read_json(target / "fixture_manifest.json", "fixture manifest")
    if manifest.get("schema_version") != FIXTURE_SCHEMA:
        raise NativeEvidenceError("unsupported fixture schema")
    expected = dict(manifest)
    digest = expected.pop("fixture_sha256", None)
    if digest != content_sha256(expected):
        raise NativeEvidenceError("fixture manifest integrity mismatch")
    for name in ("values", "timestamps"):
        item = manifest.get(name)
        if not isinstance(item, Mapping):
            raise NativeEvidenceError(f"fixture {name} record is missing")
        path = target / _safe_relative(str(item.get("path", "")), f"fixture.{name}.path")
        if not path.is_file() or file_sha256(path) != item.get("sha256"):
            raise NativeEvidenceError(f"fixture {name} bytes do not match the manifest")
        array = np.load(path, allow_pickle=False)
        if list(array.shape) != item.get("shape") or str(array.dtype) != item.get("dtype"):
            raise NativeEvidenceError(f"fixture {name} shape/dtype drifted")
        if not np.isfinite(array).all():
            raise NativeEvidenceError(f"fixture {name} contains non-finite values")
    return manifest


def _identity(dossier: Mapping[str, Any]) -> dict[str, Any]:
    tokenizer = dossier.get("tokenizer") or {}
    value = {
        "model_id": dossier["model_id"],
        "model_revision": dossier["model_revision"],
        "source_url": dossier["source_url"],
        "source_revision": dossier["source_revision"],
        "tokenizer_id": tokenizer.get("id"),
        "tokenizer_revision": tokenizer.get("revision"),
    }
    return {
        **value,
        "model_identity_sha256": content_sha256({
            "id": value["model_id"], "revision": value["model_revision"]
        }),
        "source_identity_sha256": content_sha256({
            "url": value["source_url"], "revision": value["source_revision"]
        }),
        "tokenizer_identity_sha256": (
            content_sha256({"id": value["tokenizer_id"], "revision": value["tokenizer_revision"]})
            if value["tokenizer_id"] else None
        ),
    }


def _command_description(command: Sequence[str]) -> dict[str, Any]:
    values = [str(item) for item in command]
    if not values:
        raise NativeEvidenceError("parity command is empty")
    executable = Path(values[0]).expanduser()
    if not executable.is_absolute():
        resolved = next(
            (Path(base) / executable for base in os.environ.get("PATH", "").split(os.pathsep)
             if (Path(base) / executable).is_file()),
            None,
        )
        if resolved is None:
            raise NativeEvidenceError(f"parity executable not found: {values[0]}")
        executable = resolved
    executable = executable.resolve()
    if not executable.is_file():
        raise NativeEvidenceError(f"parity executable not found: {executable}")
    value = {
        "argv": values,
        "shell_rendering": shlex.join(values),
        "argv_sha256": content_sha256(values),
        "executable": _tree_description(executable),
        "file_arguments": [],
    }
    for index, argument in enumerate(values[1:], start=1):
        candidate = Path(argument).expanduser()
        if not candidate.exists() and not candidate.is_symlink():
            continue
        if not candidate.is_absolute():
            raise NativeEvidenceError(
                f"file-valued command argument must be absolute for replay: {argument!r}"
            )
        if candidate.is_dir() and not candidate.is_symlink():
            continue
        value["file_arguments"].append({
            "argv_index": index,
            "artifact": _tree_description(candidate),
        })
    value["command_sha256"] = content_sha256(value)
    return value


def _required_artifacts(
    dossier: Mapping[str, Any], command_record: Mapping[str, Any]
) -> tuple[set[str], set[str]]:
    required = {"model", "source"}
    tokenizer = dossier.get("tokenizer") or {}
    if tokenizer and tokenizer.get("revision") != "model_revision":
        required.add("tokenizer")
    parity = dossier.get("native_parity") or {}
    extras = parity.get("required_artifacts") or []
    if not isinstance(extras, list) or any(
        not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", name)
        for name in extras
    ):
        raise NativeEvidenceError(
            "dossier.native_parity.required_artifacts must be unique snake-case names"
        )
    if len(extras) != len(set(extras)):
        raise NativeEvidenceError("dossier native parity artifacts must be unique")
    required.update(extras)
    allowed = set(required)
    allowed.add("runner")
    if not command_record.get("file_arguments"):
        required.add("runner")
    return required, allowed


def _git_policy(dossier: Mapping[str, Any], artifact_name: str) -> str:
    parity = dossier.get("native_parity") or {}
    policies = parity.get("git_untracked_policy") or {}
    if not isinstance(policies, Mapping):
        raise NativeEvidenceError(
            "dossier.native_parity.git_untracked_policy must be an artifact-policy object"
        )
    policy = policies.get(artifact_name, "ignore")
    if policy not in {"ignore", "reject"}:
        raise NativeEvidenceError(
            f"invalid Git untracked policy for artifact {artifact_name!r}: {policy!r}"
        )
    return str(policy)


def _validate_source_checkout(dossier: Mapping[str, Any], description: Mapping[str, Any]) -> None:
    if description.get("kind") != "git_checkout":
        return
    if description.get("head_revision") != dossier.get("source_revision"):
        raise NativeEvidenceError(
            "source checkout HEAD does not match dossier source_revision: "
            f"expected={dossier.get('source_revision')}, got={description.get('head_revision')}"
        )
    expected_origin = _normalize_git_origin(str(dossier.get("source_url", "")))
    if description.get("origin") != expected_origin:
        raise NativeEvidenceError(
            "source checkout origin does not match dossier source_url: "
            f"expected={expected_origin!r}, got={description.get('origin')!r}"
        )


def _validate_result(
    result: Mapping[str, Any], *, arm_key: str, track: str, required_checks: Sequence[str]
) -> None:
    result_schema = result.get("schema_version")
    if result_schema not in {RESULT_SCHEMA, LEGACY_RESULT_SCHEMA}:
        raise NativeEvidenceError(
            f"result schema must be {RESULT_SCHEMA!r} or legacy {LEGACY_RESULT_SCHEMA!r}"
        )
    if result.get("arm_key") != arm_key or result.get("track") != track:
        raise NativeEvidenceError("parity result arm/track mismatch")
    if result.get("status") not in {"pass", "research_only_pass", "fail"}:
        raise NativeEvidenceError("parity result has an invalid status")
    for field in ("environment", "admitted_runtime", "metrics"):
        if not isinstance(result.get(field), Mapping) or not result[field]:
            raise NativeEvidenceError(f"parity result requires nonempty {field}")
    if result_schema == RESULT_SCHEMA:
        from .native_parity_runtime import NativeParityRuntimeError, validate_runtime_lock

        try:
            validate_runtime_lock(result.get("runtime_lock") or {})
        except NativeParityRuntimeError as exc:
            raise NativeEvidenceError(f"parity result runtime lock is invalid: {exc}") from exc
    checks = result.get("checks")
    if not isinstance(checks, Mapping) or set(checks) != set(required_checks):
        raise NativeEvidenceError(
            "parity result checks must exactly match the current registry: "
            f"missing={sorted(set(required_checks) - set(checks or {}))}, "
            f"unknown={sorted(set(checks or {}) - set(required_checks))}"
        )
    for name, item in checks.items():
        if not isinstance(item, Mapping) or item.get("status") not in {
            "pass", "fail", "not_applicable"
        }:
            raise NativeEvidenceError(f"check {name} has an invalid result")
        if item["status"] == "pass" and not str(item.get("evidence", "")).strip():
            raise NativeEvidenceError(f"check {name} pass lacks concrete evidence")
        if item["status"] == "fail" and not str(item.get("evidence", "")).strip():
            raise NativeEvidenceError(f"check {name} failure lacks concrete evidence")
        if item["status"] == "not_applicable" and not str(item.get("reason", "")).strip():
            raise NativeEvidenceError(f"check {name} not-applicable lacks a reason")
    output_files = result.get("output_files")
    if not isinstance(output_files, list) or not output_files:
        raise NativeEvidenceError("parity result requires at least one raw output file")
    for index, value in enumerate(output_files):
        if not isinstance(value, str):
            raise NativeEvidenceError(f"output_files[{index}] must be a string")
        _safe_relative(value, f"output_files[{index}]")


def run_parity_bundle(
    *,
    arm_key: str,
    track: str,
    command: Sequence[str],
    output_directory: str | Path,
    artifacts: Mapping[str, str | Path],
    environment: Mapping[str, str] | None = None,
    created_utc: str | None = None,
    registry_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run one real parity command and bind all raw evidence into a sealed bundle."""
    registry_args = {} if registry_path is None else {"path": registry_path}
    registry = load_registry(**registry_args)
    dossier = get_dossier(arm_key, **registry_args)
    capability = dossier["tracks"].get(track)
    if not isinstance(capability, Mapping) or capability.get("status") not in ADMITTED_STATUSES:
        raise NativeEvidenceError(f"{arm_key}.{track} is not a technical-admission candidate")
    command_record = _command_description(command)
    required_artifacts, allowed_artifacts = _required_artifacts(dossier, command_record)
    missing_artifacts = sorted(required_artifacts - set(artifacts))
    unknown_artifacts = sorted(set(artifacts) - allowed_artifacts)
    if missing_artifacts or unknown_artifacts:
        raise NativeEvidenceError(
            f"bound artifacts mismatch: missing={missing_artifacts}, unknown={unknown_artifacts}"
        )

    bundle = Path(output_directory).resolve()
    if bundle.exists() and any(bundle.iterdir()):
        raise NativeEvidenceError(f"bundle directory must be empty: {bundle}")
    fixture_dir = bundle / "fixture"
    raw_dir = bundle / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    fixture = create_shared_fixture(fixture_dir)
    artifact_records = {
        name: (
            execution_code_description(path)
            if name == "runner"
            else _tree_description(path, git_untracked_policy=_git_policy(dossier, name))
        )
        for name, path in sorted(artifacts.items())
    }
    _validate_source_checkout(dossier, artifact_records["source"])
    declared_environment = {str(key): str(value) for key, value in sorted((environment or {}).items())}
    child_environment = os.environ.copy()
    child_environment.update(declared_environment)
    child_environment.update({
        "FFM_NATIVE_PARITY_FIXTURE": str(fixture_dir / "fixture_manifest.json"),
        "FFM_NATIVE_PARITY_VALUES": str(fixture_dir / fixture["values"]["path"]),
        "FFM_NATIVE_PARITY_TIMESTAMPS": str(fixture_dir / fixture["timestamps"]["path"]),
        "FFM_NATIVE_PARITY_RESULT_DIR": str(raw_dir),
        "FFM_NATIVE_PARITY_ARM": arm_key,
        "FFM_NATIVE_PARITY_TRACK": track,
    })
    for name, description in artifact_records.items():
        child_environment[f"FFM_NATIVE_PARITY_ARTIFACT_{name.upper()}"] = description["path"]
    completed = subprocess.run(
        list(command),
        cwd=str(bundle),
        env=child_environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    stdout_path = bundle / "stdout.log"
    stderr_path = bundle / "stderr.log"
    stdout_path.write_bytes(completed.stdout)
    stderr_path.write_bytes(completed.stderr)
    if completed.returncode != 0:
        raise NativeEvidenceError(
            f"parity command failed with exit {completed.returncode}; "
            f"see {stdout_path} and {stderr_path}"
        )
    result_path = raw_dir / "result.json"
    result = _read_json(result_path, "parity result")
    _validate_result(
        result, arm_key=arm_key, track=track, required_checks=registry["required_checks"]
    )
    if result["status"] == "fail":
        raise NativeEvidenceError("parity command reported failure")
    expected_status = "research_only_pass" if capability["status"] == "research_only" else "pass"
    if result["status"] != expected_status:
        raise NativeEvidenceError(
            f"parity result status must be {expected_status!r} for {capability['status']!r}"
        )
    failures = [name for name, item in result["checks"].items() if item["status"] == "fail"]
    mandatory_missing = [
        name for name in sorted(MANDATORY_TECHNICAL_CHECKS)
        if result["checks"][name]["status"] != "pass"
    ]
    if failures or mandatory_missing:
        raise NativeEvidenceError(
            f"parity result is not admissible: failed={failures}, "
            f"mandatory_not_passed={mandatory_missing}"
        )
    raw_outputs = {}
    for relative in result["output_files"]:
        path = raw_dir / _safe_relative(relative, "output file")
        try:
            path.resolve().relative_to(raw_dir.resolve())
        except ValueError as exc:
            raise NativeEvidenceError(f"output escapes raw directory: {relative}") from exc
        if path.name == "result.json":
            raise NativeEvidenceError("result.json may not be its own raw output")
        raw_outputs[relative] = _tree_description(path, relative_to=bundle)

    manifest = {
        "schema_version": BUNDLE_SCHEMA,
        "created_utc": created_utc or _utc_now(),
        "arm_key": arm_key,
        "track": track,
        "evidence_id": capability["evidence_id"],
        "methodology_commit": registry["methodology_commit"],
        "registry_sha256": registry_sha256(**registry_args),
        "dossier_sha256": dossier_sha256(arm_key, **registry_args),
        "identity": _identity(dossier),
        "fixture": {
            "path": "fixture/fixture_manifest.json",
            "sha256": file_sha256(fixture_dir / "fixture_manifest.json"),
            "fixture_sha256": fixture["fixture_sha256"],
        },
        "bound_artifacts": artifact_records,
        "command": command_record,
        "declared_environment": declared_environment,
        "orchestrator_environment": {
            "python_executable": str(Path(sys.executable).resolve()),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "returncode": completed.returncode,
        "logs": {
            "stdout": {"path": "stdout.log", "sha256": file_sha256(stdout_path)},
            "stderr": {"path": "stderr.log", "sha256": file_sha256(stderr_path)},
        },
        "result": {"path": "raw/result.json", "sha256": file_sha256(result_path)},
        "raw_outputs": raw_outputs,
    }
    manifest["bundle_sha256"] = content_sha256(manifest)
    _write_json(bundle / "bundle_manifest.json", manifest)
    verify_parity_bundle(bundle, registry_path=registry_path)
    return manifest


def verify_parity_bundle(
    directory: str | Path,
    *,
    registry_path: str | Path | None = None,
    verify_external_artifacts: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    bundle = Path(directory).resolve()
    manifest = _read_json(bundle / "bundle_manifest.json", "bundle manifest")
    if manifest.get("schema_version") != BUNDLE_SCHEMA:
        raise NativeEvidenceError("unsupported evidence bundle schema")
    integrity = dict(manifest)
    digest = integrity.pop("bundle_sha256", None)
    if digest != content_sha256(integrity):
        raise NativeEvidenceError("bundle manifest integrity mismatch")
    registry_args = {} if registry_path is None else {"path": registry_path}
    registry = load_registry(**registry_args)
    arm_key, track = manifest.get("arm_key"), manifest.get("track")
    dossier = get_dossier(str(arm_key), **registry_args)
    capability = dossier["tracks"].get(track)
    if not isinstance(capability, Mapping) or capability.get("status") not in ADMITTED_STATUSES:
        raise NativeEvidenceError(f"bundle arm/track is no longer admissible: {arm_key}.{track}")
    expected = {
        "methodology_commit": registry["methodology_commit"],
        "registry_sha256": registry_sha256(**registry_args),
        "dossier_sha256": dossier_sha256(str(arm_key), **registry_args),
        "evidence_id": capability["evidence_id"],
        "identity": _identity(dossier),
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise NativeEvidenceError(f"bundle {field} is stale or mismatched")
    fixture_record = manifest.get("fixture") or {}
    fixture_manifest_path = bundle / _safe_relative(
        str(fixture_record.get("path", "")), "bundle.fixture.path"
    )
    if file_sha256(fixture_manifest_path) != fixture_record.get("sha256"):
        raise NativeEvidenceError("bound fixture manifest hash mismatch")
    fixture = verify_shared_fixture(fixture_manifest_path.parent)
    if fixture.get("fixture_sha256") != fixture_record.get("fixture_sha256"):
        raise NativeEvidenceError("bound fixture identity mismatch")
    command = manifest.get("command")
    if not isinstance(command, Mapping):
        raise NativeEvidenceError("bundle command record is missing")
    required_artifacts, allowed_artifacts = _required_artifacts(dossier, command)
    artifact_names = set(manifest.get("bound_artifacts") or {})
    if not required_artifacts.issubset(artifact_names) or not artifact_names.issubset(allowed_artifacts):
        raise NativeEvidenceError(
            "bundle bound artifacts mismatch: "
            f"required={sorted(required_artifacts)}, allowed={sorted(allowed_artifacts)}, "
            f"got={sorted(artifact_names)}"
        )
    if verify_external_artifacts:
        for name, description in (manifest.get("bound_artifacts") or {}).items():
            _verify_tree(description, f"bound_artifacts.{name}")
        _validate_source_checkout(dossier, manifest["bound_artifacts"]["source"])
    command_integrity = dict(command)
    command_digest = command_integrity.pop("command_sha256", None)
    if command_digest != content_sha256(command_integrity):
        raise NativeEvidenceError("command record integrity mismatch")
    if verify_external_artifacts:
        _verify_tree(command["executable"], "command.executable")
    for index, item in enumerate(command.get("file_arguments") or []):
        if not isinstance(item, Mapping) or not isinstance(item.get("artifact"), Mapping):
            raise NativeEvidenceError(f"command.file_arguments[{index}] is invalid")
        argv_index = item.get("argv_index")
        if not isinstance(argv_index, int) or not 0 < argv_index < len(command.get("argv") or []):
            raise NativeEvidenceError(f"command.file_arguments[{index}] index is invalid")
        if command["argv"][argv_index] != item["artifact"].get("path"):
            raise NativeEvidenceError(f"command.file_arguments[{index}] argv binding mismatch")
        if verify_external_artifacts:
            _verify_tree(item["artifact"], f"command.file_arguments[{index}]")
    for name, item in (manifest.get("logs") or {}).items():
        path = bundle / _safe_relative(str(item.get("path", "")), f"logs.{name}.path")
        if not path.is_file() or file_sha256(path) != item.get("sha256"):
            raise NativeEvidenceError(f"{name} log hash mismatch")
    result_record = manifest.get("result") or {}
    result_path = bundle / _safe_relative(str(result_record.get("path", "")), "result.path")
    if file_sha256(result_path) != result_record.get("sha256"):
        raise NativeEvidenceError("result hash mismatch")
    result = _read_json(result_path, "parity result")
    _validate_result(
        result, arm_key=str(arm_key), track=str(track),
        required_checks=registry["required_checks"],
    )
    expected_status = (
        "research_only_pass" if capability["status"] == "research_only" else "pass"
    )
    if result["status"] != expected_status:
        raise NativeEvidenceError(
            f"bundle result status must be {expected_status!r}, got {result['status']!r}"
        )
    for relative, description in (manifest.get("raw_outputs") or {}).items():
        safe = _safe_relative(relative, "raw output")
        expected_path = (Path("raw") / safe).as_posix()
        if str(description.get("path")) != expected_path:
            raise NativeEvidenceError(f"raw output path binding mismatch: {relative}")
        _verify_tree(description, f"raw_outputs.{relative}", relative_to=bundle)
    if set(manifest.get("raw_outputs") or {}) != set(result["output_files"]):
        raise NativeEvidenceError("result output list differs from sealed raw outputs")
    if result["status"] == "fail" or any(
        item["status"] == "fail" for item in result["checks"].values()
    ):
        raise NativeEvidenceError("bundle contains failed parity evidence")
    mandatory_missing = [
        name for name in sorted(MANDATORY_TECHNICAL_CHECKS)
        if result["checks"][name]["status"] != "pass"
    ]
    if mandatory_missing:
        raise NativeEvidenceError(
            f"bundle lacks mandatory technical passes: {mandatory_missing}"
        )
    return manifest, result


def _admitted_pairs(registry: Mapping[str, Any]) -> set[tuple[str, str]]:
    return {
        (arm_key, track)
        for arm_key, dossier in registry["models"].items()
        for track, capability in dossier["tracks"].items()
        if capability["status"] in ADMITTED_STATUSES
    }


def aggregate_parity_bundles(
    bundle_directories: Iterable[str | Path],
    *,
    output_path: str | Path,
    generated_utc: str | None = None,
    require_all_current: bool = True,
    registry_path: str | Path | None = None,
) -> dict[str, Any]:
    """Verify bundles and generate a canonical evidence candidate, never install it."""
    registry_args = {} if registry_path is None else {"path": registry_path}
    registry = load_registry(**registry_args)
    destination = Path(output_path).resolve()
    effective_registry_path = Path(registry_path or REGISTRY_PATH).resolve()
    evidence_parent = effective_registry_path.parent
    verified: dict[tuple[str, str], tuple[dict[str, Any], dict[str, Any], Path]] = {}
    for raw_path in bundle_directories:
        path = Path(raw_path).resolve()
        manifest, result = verify_parity_bundle(path, registry_path=registry_path)
        key = (manifest["arm_key"], manifest["track"])
        if key in verified:
            raise NativeEvidenceError(f"duplicate parity bundle for {key[0]}.{key[1]}")
        verified[key] = (manifest, result, path)
    required = _admitted_pairs(registry)
    if require_all_current and set(verified) != required:
        raise NativeEvidenceError(
            "bundle coverage differs from current technical tracks: "
            f"missing={sorted(required - set(verified))}, "
            f"unexpected={sorted(set(verified) - required)}"
        )
    if not verified:
        raise NativeEvidenceError("cannot aggregate zero evidence bundles")

    default_profile = {
        name: {"status": "not_applicable", "reason": "overridden by generated bundle"}
        for name in registry["required_checks"]
    }
    records: dict[str, Any] = {}
    bundle_index = []
    for key in sorted(verified):
        manifest, result, path = verified[key]
        evidence_id = manifest["evidence_id"]
        if evidence_id in records:
            raise NativeEvidenceError(f"duplicate generated evidence id: {evidence_id}")
        identity = manifest["identity"]
        records[evidence_id] = {
            "arm_key": key[0],
            "track": key[1],
            "status": result["status"],
            "profile": "generated_bundle",
            "identity": {
                field: identity[field]
                for field in (
                    "model_id", "model_revision", "source_revision",
                    "tokenizer_id", "tokenizer_revision",
                )
                if identity.get(field) is not None
            },
            "environment": dict(result["environment"]),
            **({"runtime_lock": dict(result["runtime_lock"])} if result.get("runtime_lock") else {}),
            "checks": {name: dict(item) for name, item in result["checks"].items()},
            "admitted_runtime": dict(result["admitted_runtime"]),
            "metrics": dict(result["metrics"]),
            "bundle": {
                # The candidate is reviewed and then copied out of the aggregate.  Bind
                # paths to the canonical evidence/registry directory so they remain valid
                # after installation instead of silently changing meaning.
                "path": os.path.relpath(path, evidence_parent),
                "path_base": "evidence_registry_parent",
                "bundle_sha256": manifest["bundle_sha256"],
                "fixture_sha256": manifest["fixture"]["fixture_sha256"],
                "command_sha256": manifest["command"]["command_sha256"],
                "result_sha256": manifest["result"]["sha256"],
                "stdout_sha256": manifest["logs"]["stdout"]["sha256"],
                "stderr_sha256": manifest["logs"]["stderr"]["sha256"],
            },
        }
        bundle_index.append({
            "arm_key": key[0], "track": key[1], "evidence_id": evidence_id,
            "bundle_sha256": manifest["bundle_sha256"],
        })
    created = generated_utc or _utc_now()
    candidate = {
        "schema_version": EVIDENCE_SCHEMA,
        "methodology_commit": registry["methodology_commit"],
        "generated_utc": created,
        "policy": {
            "source": "verified_raw_native_parity_bundles",
            "aggregate_schema": AGGREGATE_SCHEMA,
            "require_all_current": bool(require_all_current),
            "complete_for_installation": bool(require_all_current),
            "registry_sha256": registry_sha256(**registry_args),
        },
        "check_profiles": {"generated_bundle": default_profile},
        "records": records,
    }
    aggregate = {
        "schema_version": AGGREGATE_SCHEMA,
        "generated_utc": created,
        "registry_sha256": registry_sha256(**registry_args),
        "methodology_commit": registry["methodology_commit"],
        "require_all_current": bool(require_all_current),
        "bundles": bundle_index,
        "candidate_evidence_sha256": content_sha256(candidate),
        "candidate_evidence": candidate,
    }
    aggregate["aggregate_sha256"] = content_sha256(aggregate)
    canonical_path = effective_registry_path.with_name("native_contract_evidence.json")
    if destination == canonical_path:
        raise NativeEvidenceError(
            "aggregation may not overwrite canonical evidence; review and install separately"
        )
    if require_all_current:
        _validate_evidence(candidate, registry)
    _write_json(destination, aggregate)
    return aggregate
