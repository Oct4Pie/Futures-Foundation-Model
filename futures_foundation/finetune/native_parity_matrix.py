"""Fail-closed orchestration for the complete native F/R parity matrix.

The matrix runner is deliberately boring: it resolves only exact local artifacts,
executes one sealed parity bundle at a time, verifies any bundle before resuming it,
and asks :mod:`native_evidence_bundle` to enforce complete registry coverage.  It
does not download models, train, read market data, or install generated evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Mapping, Sequence

from .native_contracts import ADMITTED_STATUSES, REGISTRY_PATH, load_registry
from .native_evidence_bundle import (
    NativeEvidenceError,
    aggregate_parity_bundles,
    run_parity_bundle,
    verify_parity_bundle,
)
from .native_parity_runtime import (
    ARM_PACKAGES,
    GIT_SOURCE_ARMS,
    PACKAGE_PROFILES,
    PACKAGE_SOURCE_ARMS,
    PROFILE_ARMS,
    PROFILE_PYTHON,
    validate_distribution_record,
)


MATRIX_CONFIG_SCHEMA = "ffm_native_parity_matrix_config_v1"
MATRIX_PLAN_SCHEMA = "ffm_native_parity_matrix_plan_v1"
NATIVE_TRACKS = frozenset({"F", "R"})
OFFLINE_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "TOKENIZERS_PARALLELISM": "false",
}


class NativeParityMatrixError(NativeEvidenceError):
    """Raised when the matrix cannot be executed without guessing."""


@dataclass(frozen=True)
class MatrixEntry:
    arm_key: str
    track: str
    profile: str
    python: Path
    source: Path
    model: Path
    tokenizer: Path | None
    extra_artifacts: tuple[tuple[str, Path], ...]
    runner_source: Path
    bundle: Path

    @property
    def key(self) -> tuple[str, str]:
        return self.arm_key, self.track

    @property
    def artifacts(self) -> dict[str, Path]:
        result = {"model": self.model, "source": self.source}
        if self.tokenizer is not None:
            result["tokenizer"] = self.tokenizer
        result.update(self.extra_artifacts)
        result["runner"] = self.runner_source
        return result

    def as_record(self) -> dict[str, Any]:
        return {
            "arm_key": self.arm_key,
            "track": self.track,
            "runtime_profile": self.profile,
            "python": str(self.python),
            "source": str(self.source),
            "model": str(self.model),
            "tokenizer": None if self.tokenizer is None else str(self.tokenizer),
            "extra_artifacts": {
                name: str(path) for name, path in self.extra_artifacts
            },
            "runner_source": str(self.runner_source),
            "bundle": str(self.bundle),
        }


def _expanded_path(value: str | Path, *, base: Path | None = None) -> Path:
    raw = os.path.expandvars(os.path.expanduser(str(value)))
    candidate = Path(raw)
    if not candidate.is_absolute() and base is not None:
        candidate = base / candidate
    return candidate.resolve()


def _expanded_executable(value: str | Path, *, base: Path | None = None) -> Path:
    """Preserve a venv launcher symlink; resolving it discards pyvenv.cfg semantics."""
    raw = os.path.expandvars(os.path.expanduser(str(value)))
    candidate = Path(raw)
    if not candidate.is_absolute() and base is not None:
        candidate = base / candidate
    return Path(os.path.abspath(candidate))


def _pairs(values: Sequence[str], field: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise NativeParityMatrixError(f"{field} must use NAME=VALUE: {value!r}")
        name, raw = value.split("=", 1)
        if not name or not raw or name in result:
            raise NativeParityMatrixError(f"invalid or duplicate {field}: {name!r}")
        result[name] = raw
    return result


def load_matrix_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = Path(path).expanduser().resolve()
    try:
        value = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NativeParityMatrixError(f"cannot read matrix config: {config_path}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != MATRIX_CONFIG_SCHEMA:
        raise NativeParityMatrixError(
            f"matrix config schema must be {MATRIX_CONFIG_SCHEMA!r}"
        )
    allowed = {
        "schema_version", "runtime_profiles", "source_roots", "hf_cache_roots",
        "environment", "device", "batch_size", "samples", "seed", "runner",
        "runner_source", "execution_sources",
    }
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise NativeParityMatrixError(f"unknown matrix config fields: {unknown}")
    return value, config_path.parent


def admitted_native_pairs(registry: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    pairs = []
    for arm_key, dossier in registry["models"].items():
        for track, capability in dossier["tracks"].items():
            if capability["status"] not in ADMITTED_STATUSES:
                continue
            if track not in NATIVE_TRACKS:
                raise NativeParityMatrixError(
                    f"technical track {arm_key}.{track} is outside the native F/R matrix"
                )
            pairs.append((arm_key, track))
    if not pairs:
        raise NativeParityMatrixError("registry has no admitted native F/R tracks")
    return tuple(sorted(pairs))


def _repo_cache_name(model_id: str) -> str:
    parts = model_id.split("/")
    if len(parts) != 2 or any(not part or part in {".", ".."} for part in parts):
        raise NativeParityMatrixError(f"unsupported Hugging Face model id: {model_id!r}")
    return "models--" + "--".join(parts)


def _snapshot_root(root: Path) -> Path:
    return root / "hub" if (root / "hub").is_dir() else root


def resolve_offline_snapshot(
    cache_roots: Sequence[Path], *, model_id: str, revision: str
) -> Path:
    """Resolve one exact materialized HF snapshot without refs or network fallback."""
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise NativeParityMatrixError(
            f"snapshot revision for {model_id!r} is not an exact 40-hex commit: {revision!r}"
        )
    matches = []
    for raw_root in cache_roots:
        root = _snapshot_root(raw_root)
        candidate = root / _repo_cache_name(model_id) / "snapshots" / revision
        if candidate.is_dir():
            matches.append(candidate.resolve())
    unique = sorted(set(matches))
    if len(unique) != 1:
        raise NativeParityMatrixError(
            f"exact offline snapshot resolution for {model_id}@{revision} found "
            f"{len(unique)} matches: {[str(item) for item in unique]}"
        )
    snapshot = unique[0]
    if snapshot.name != revision or not (snapshot / "config.json").is_file():
        raise NativeParityMatrixError(
            f"snapshot is not fully materialized with config.json: {snapshot}"
        )
    broken = [path for path in snapshot.rglob("*") if path.is_symlink() and not path.exists()]
    if broken:
        raise NativeParityMatrixError(
            f"snapshot contains broken artifact links: {[str(path) for path in broken[:8]]}"
        )
    try:
        config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NativeParityMatrixError(f"snapshot config is invalid JSON: {snapshot}") from exc
    auto_map = config.get("auto_map") or {}
    if not isinstance(auto_map, dict):
        raise NativeParityMatrixError(f"snapshot auto_map must be an object: {snapshot}")
    remote_modules = set()
    for reference in auto_map.values():
        values = reference if isinstance(reference, list) else [reference]
        for value in values:
            if not isinstance(value, str) or "." not in value:
                raise NativeParityMatrixError(
                    f"snapshot auto_map reference is invalid: {value!r}"
                )
            module = value.split("--", 1)[-1].rsplit(".", 1)[0]
            remote_modules.add(module.replace(".", "/") + ".py")
    missing_modules = sorted(name for name in remote_modules if not (snapshot / name).is_file())
    if missing_modules:
        raise NativeParityMatrixError(
            f"snapshot auto_map references missing Python modules: {missing_modules}"
        )
    index_files = sorted(snapshot.glob("*.index.json"))
    for index_path in index_files:
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise NativeParityMatrixError(f"invalid snapshot weight index: {index_path}") from exc
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise NativeParityMatrixError(f"snapshot weight index has no weight_map: {index_path}")
        missing = sorted({name for name in weight_map.values() if not (snapshot / name).is_file()})
        if missing:
            raise NativeParityMatrixError(
                f"snapshot weight index references missing shards: {missing[:8]}"
            )
    weight_files = [
        path for pattern in ("*.safetensors", "*.bin", "*.pt", "*.ckpt")
        for path in snapshot.glob(pattern) if path.is_file()
    ]
    if not weight_files:
        raise NativeParityMatrixError(f"snapshot has no materialized model weights: {snapshot}")
    return snapshot


def _profile_maps() -> tuple[dict[str, set[str]], dict[str, tuple[int, int]], dict[str, dict[str, str]]]:
    packages = {
        profile: dict(required)
        for profile, required in PACKAGE_PROFILES.items()
    }
    for profile, arms in PROFILE_ARMS.items():
        for arm in arms:
            packages[profile].update(ARM_PACKAGES.get(arm, {}))
    return PROFILE_ARMS, PROFILE_PYTHON, packages


def _profile_for_arm(arm_key: str, profiles: Mapping[str, set[str]]) -> str:
    matches = [name for name, arms in profiles.items() if arm_key in arms]
    if len(matches) != 1:
        raise NativeParityMatrixError(
            f"arm must have exactly one worker runtime profile: {arm_key} -> {matches}"
        )
    return matches[0]


def _validate_python(
    profile: str,
    executable: Path,
    *,
    expected_python: tuple[int, int],
    expected_packages: Mapping[str, str],
) -> None:
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise NativeParityMatrixError(
            f"runtime profile {profile!r} Python is not executable: {executable}"
        )
    program = (
        "import importlib.metadata as m,json,sys;"
        "print(json.dumps({'python':list(sys.version_info[:2]),'executable':sys.executable,"
        "'packages':{n:m.version(n) for n in " + repr(sorted(expected_packages)) + "}}))"
    )
    try:
        completed = subprocess.run(
            [str(executable), "-I", "-c", program],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        actual = json.loads(completed.stdout)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        detail = getattr(exc, "stderr", "")
        raise NativeParityMatrixError(
            f"cannot validate runtime profile {profile!r}: {detail}"
        ) from exc
    if tuple(actual.get("python", ())) != expected_python:
        raise NativeParityMatrixError(
            f"runtime profile {profile!r} Python drift: expected {expected_python}, "
            f"got {actual.get('python')}"
        )
    if actual.get("packages") != dict(expected_packages):
        raise NativeParityMatrixError(
            f"runtime profile {profile!r} package drift: expected "
            f"{dict(expected_packages)}, got {actual.get('packages')}"
        )


def _validate_source_root(
    arm_key: str,
    source: Path,
    *,
    dossier: Mapping[str, Any],
    python: Path,
) -> None:
    if arm_key in GIT_SOURCE_ARMS:
        try:
            if not (source / ".git").is_dir():
                raise NativeParityMatrixError(f"source is not a Git checkout: {source}")
            revision = subprocess.check_output(
                ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
            ).strip()
            if revision != dossier["source_revision"]:
                raise NativeParityMatrixError(
                    f"source revision mismatch: expected {dossier['source_revision']}, "
                    f"got {revision}"
                )
            dirty = subprocess.check_output(
                ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=all"],
                text=True,
            ).strip()
            if dirty:
                raise NativeParityMatrixError(f"source checkout is dirty: {source}")
            origin = subprocess.check_output(
                ["git", "-C", str(source), "remote", "get-url", "origin"], text=True
            ).strip()
            normalized = origin.lower().removesuffix(".git").replace(
                "git@github.com:", "https://github.com/"
            )
            expected = str(dossier["source_url"]).lower().removesuffix(".git")
            if normalized != expected:
                raise NativeParityMatrixError(
                    f"source origin mismatch: expected {expected!r}, got {normalized!r}"
                )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise NativeParityMatrixError(
                f"source root validation failed for {arm_key}: {exc}"
            ) from exc
        return
    if arm_key not in PACKAGE_SOURCE_ARMS:
        raise NativeParityMatrixError(f"worker has no source policy for {arm_key}")
    program = (
        "from pathlib import Path;import importlib.metadata as m;"
        "print(Path(m.distribution('chronos-forecasting')._path).resolve())"
    )
    try:
        completed = subprocess.run(
            [str(python), "-I", "-c", program], check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise NativeParityMatrixError(
            f"cannot resolve Chronos package source for {arm_key}: "
            f"{getattr(exc, 'stderr', '')}"
        ) from exc
    imported = Path(completed.stdout.strip()).resolve()
    if imported != source:
        raise NativeParityMatrixError(
            f"source root for {arm_key} differs from imported Chronos package: "
            f"configured={source}, imported={imported}"
        )
    try:
        validate_distribution_record(source)
    except RuntimeError as exc:
        raise NativeParityMatrixError(
            f"Chronos distribution RECORD validation failed: {exc}"
        ) from exc


def _validate_execution_source(
    arm_key: str,
    source: Path,
    *,
    dossier: Mapping[str, Any],
    python: Path,
) -> None:
    declared = (dossier.get("native_parity") or {}).get(
        "execution_source_distribution"
    ) or {}
    name, version = declared.get("name"), declared.get("version")
    if not isinstance(name, str) or not isinstance(version, str):
        raise NativeParityMatrixError(
            f"{arm_key} has no pinned execution-source distribution"
        )
    program = (
        "from pathlib import Path;import importlib.metadata as m;"
        f"d=m.distribution({name!r});"
        "print(str(Path(d._path).resolve())+'\\n'+d.version)"
    )
    try:
        completed = subprocess.run(
            [str(python), "-I", "-c", program], check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise NativeParityMatrixError(
            f"cannot resolve execution source for {arm_key}: "
            f"{getattr(exc, 'stderr', '')}"
        ) from exc
    actual_path, actual_version = completed.stdout.strip().splitlines()
    if Path(actual_path).resolve() != source or actual_version != version:
        raise NativeParityMatrixError(
            f"{arm_key} execution source drift: expected {source} version {version}, "
            f"got {actual_path} version {actual_version}"
        )
    try:
        validate_distribution_record(source)
    except RuntimeError as exc:
        raise NativeParityMatrixError(
            f"{arm_key} execution-source RECORD validation failed: {exc}"
        ) from exc


def build_matrix_plan(
    *,
    registry: Mapping[str, Any],
    runtime_pythons: Mapping[str, str | Path],
    source_roots: Mapping[str, str | Path],
    hf_cache_roots: Sequence[str | Path],
    output_directory: str | Path,
    runner: str | Path,
    runner_source: str | Path,
    execution_sources: Mapping[str, str | Path] | None = None,
    path_base: Path | None = None,
    validate_environments: bool = True,
) -> tuple[MatrixEntry, ...]:
    profiles, python_versions, profile_packages = _profile_maps()
    pairs = admitted_native_pairs(registry)
    arms = {arm for arm, _ in pairs}
    required_profiles = {_profile_for_arm(arm, profiles) for arm in arms}
    if set(runtime_pythons) != required_profiles:
        raise NativeParityMatrixError(
            "runtime profile mapping must exactly cover the matrix: "
            f"missing={sorted(required_profiles - set(runtime_pythons))}, "
            f"unknown={sorted(set(runtime_pythons) - required_profiles)}"
        )
    if set(source_roots) != arms:
        raise NativeParityMatrixError(
            "source-root mapping must exactly cover admitted arms: "
            f"missing={sorted(arms - set(source_roots))}, "
            f"unknown={sorted(set(source_roots) - arms)}"
        )
    cache_roots = tuple(_expanded_path(path, base=path_base) for path in hf_cache_roots)
    if not cache_roots or any(not path.is_dir() for path in cache_roots):
        raise NativeParityMatrixError("every HF cache root must be an existing directory")
    python_paths = {
        name: _expanded_executable(path, base=path_base)
        for name, path in runtime_pythons.items()
    }
    if validate_environments:
        for profile in sorted(required_profiles):
            _validate_python(
                profile, python_paths[profile],
                expected_python=python_versions[profile],
                expected_packages=profile_packages[profile],
            )
    sources = {
        arm: _expanded_path(path, base=path_base) for arm, path in source_roots.items()
    }
    missing_sources = [arm for arm, path in sources.items() if not path.is_dir()]
    if missing_sources:
        raise NativeParityMatrixError(f"source roots do not exist: {missing_sources}")
    if validate_environments:
        for arm in sorted(arms):
            profile = _profile_for_arm(arm, profiles)
            _validate_source_root(
                arm, sources[arm], dossier=registry["models"][arm],
                python=python_paths[profile],
            )
    execution_source_arms = {
        arm for arm in arms
        if "execution_source" in (
            registry["models"][arm].get("native_parity") or {}
        ).get("required_artifacts", ())
    }
    supplied_execution_sources = dict(execution_sources or {})
    if set(supplied_execution_sources) != execution_source_arms:
        raise NativeParityMatrixError(
            "execution-source mapping must exactly cover declared arms: "
            f"missing={sorted(execution_source_arms - set(supplied_execution_sources))}, "
            f"unknown={sorted(set(supplied_execution_sources) - execution_source_arms)}"
        )
    resolved_execution_sources = {
        arm: _expanded_path(path, base=path_base)
        for arm, path in supplied_execution_sources.items()
    }
    if validate_environments:
        for arm, source in sorted(resolved_execution_sources.items()):
            profile = _profile_for_arm(arm, profiles)
            _validate_execution_source(
                arm, source, dossier=registry["models"][arm],
                python=python_paths[profile],
            )
    runner_path = _expanded_path(runner, base=path_base)
    if not runner_path.is_file():
        raise NativeParityMatrixError(f"native parity worker not found: {runner_path}")
    runner_source_path = _expanded_path(runner_source, base=path_base)
    if not runner_source_path.exists():
        raise NativeParityMatrixError(
            f"runner/transitive source artifact not found: {runner_source_path}"
        )
    if runner_source_path.name.endswith(".dist-info"):
        try:
            validate_distribution_record(runner_source_path)
        except RuntimeError as exc:
            raise NativeParityMatrixError(
                f"runner distribution RECORD validation failed: {exc}"
            ) from exc
    destination = _expanded_path(output_directory, base=path_base)
    entries = []
    for arm_key, track in pairs:
        dossier = registry["models"][arm_key]
        profile = _profile_for_arm(arm_key, profiles)
        model = resolve_offline_snapshot(
            cache_roots,
            model_id=dossier["model_id"], revision=dossier["model_revision"],
        )
        tokenizer = dossier.get("tokenizer") or {}
        tokenizer_path = None
        if tokenizer and tokenizer.get("revision") != "model_revision":
            tokenizer_path = resolve_offline_snapshot(
                cache_roots, model_id=tokenizer["id"], revision=tokenizer["revision"]
            )
        extras = []
        native_parity = dossier.get("native_parity") or {}
        for name in native_parity.get("required_artifacts") or []:
            if name == "execution_source":
                extras.append((name, resolved_execution_sources[arm_key]))
                continue
            model_id = native_parity.get(f"{name}_id")
            revision = native_parity.get(f"{name}_revision")
            if not isinstance(model_id, str) or not isinstance(revision, str):
                raise NativeParityMatrixError(
                    f"{arm_key} cannot resolve required artifact {name!r} from dossier"
                )
            extras.append((name, resolve_offline_snapshot(
                cache_roots, model_id=model_id, revision=revision
            )))
        entries.append(MatrixEntry(
            arm_key=arm_key, track=track, profile=profile,
            python=python_paths[profile], source=sources[arm_key], model=model,
            tokenizer=tokenizer_path, extra_artifacts=tuple(extras),
            runner_source=runner_source_path,
            bundle=destination / f"{arm_key}__{track}",
        ))
    if len({entry.key for entry in entries}) != len(entries):
        raise NativeParityMatrixError("matrix contains duplicate arm/track entries")
    return tuple(entries)


def worker_command(
    entry: MatrixEntry,
    *,
    runner: str | Path,
    device: str,
    batch_size: int,
    samples: int,
    seed: int,
) -> list[str]:
    command = [
        str(entry.python), "-I", str(Path(runner).resolve()),
        "--arm", entry.arm_key, "--track", entry.track,
        "--profile", entry.profile,
        "--model-snapshot", str(entry.model),
        "--source-repo", str(entry.source),
        "--device", device,
        "--batch-size", str(batch_size),
        "--samples", str(samples), "--seed", str(seed),
        "--network-policy", "python_socket_deny",
    ]
    if entry.tokenizer is not None:
        command.extend(["--tokenizer-snapshot", str(entry.tokenizer)])
    for name, path in entry.extra_artifacts:
        if name == "reference_model":
            command.extend(["--reference-model-snapshot", str(path)])
        elif name == "execution_source":
            command.extend(["--execution-source", str(path)])
        else:
            raise NativeParityMatrixError(
                f"worker has no CLI binding for extra artifact {name!r}"
            )
    return command


def execute_matrix(
    entries: Sequence[MatrixEntry],
    *,
    runner: str | Path,
    aggregate_output: str | Path,
    registry_path: str | Path | None = None,
    environment: Mapping[str, str] | None = None,
    device: str = "cuda:0",
    batch_size: int = 4,
    samples: int = 20,
    seed: int = 20260717,
    generated_utc: str | None = None,
) -> dict[str, Any]:
    if batch_size < 1 or samples < 1:
        raise NativeParityMatrixError("batch size and sample count must be positive")
    registry_args = {} if registry_path is None else {"path": registry_path}
    registry = load_registry(**registry_args)
    required_keys = set(admitted_native_pairs(registry))
    entry_keys = [entry.key for entry in entries]
    duplicates = sorted({key for key in entry_keys if entry_keys.count(key) > 1})
    supplied_keys = set(entry_keys)
    if duplicates or supplied_keys != required_keys:
        raise NativeParityMatrixError(
            "matrix must exactly and uniquely cover current admitted F/R tracks before execution: "
            f"duplicates={duplicates}, missing={sorted(required_keys - supplied_keys)}, "
            f"unexpected={sorted(supplied_keys - required_keys)}"
        )
    declared_environment = {str(key): str(value) for key, value in (environment or {}).items()}
    for name, required in OFFLINE_ENVIRONMENT.items():
        supplied = declared_environment.get(name)
        if supplied is not None and supplied != required:
            raise NativeParityMatrixError(
                f"offline environment {name} may not be overridden with {supplied!r}"
            )
        declared_environment[name] = required
    verified = []
    verified_keys = set()
    expected_keys = required_keys
    for entry in entries:
        bundle = entry.bundle.resolve()
        command = worker_command(
            entry, runner=runner, device=device, batch_size=batch_size,
            samples=samples, seed=seed,
        )
        if bundle.exists() and any(bundle.iterdir()) and not (
            bundle / "bundle_manifest.json"
        ).is_file():
            suffix = 1
            while True:
                quarantine = bundle.with_name(f"{bundle.name}.incomplete-{suffix:03d}")
                if not quarantine.exists():
                    break
                suffix += 1
            bundle.rename(quarantine)
        if bundle.exists() and any(bundle.iterdir()):
            manifest, _ = verify_parity_bundle(bundle, registry_path=registry_path)
            if (manifest["arm_key"], manifest["track"]) != entry.key:
                raise NativeParityMatrixError(
                    f"resume bundle pair mismatch at {bundle}: "
                    f"{manifest['arm_key']}.{manifest['track']} != "
                    f"{entry.arm_key}.{entry.track}"
                )
            expected_artifacts = {
                name: str(path.resolve()) for name, path in entry.artifacts.items()
            }
            actual_artifacts = {
                name: str(Path(item["path"]).resolve())
                for name, item in manifest["bound_artifacts"].items()
            }
            if manifest["command"]["argv"] != command:
                raise NativeParityMatrixError(
                    f"resume command/settings drift for {entry.arm_key}.{entry.track}"
                )
            if manifest.get("declared_environment") != declared_environment:
                raise NativeParityMatrixError(
                    f"resume environment drift for {entry.arm_key}.{entry.track}"
                )
            if actual_artifacts != expected_artifacts:
                raise NativeParityMatrixError(
                    f"resume artifact mapping drift for {entry.arm_key}.{entry.track}"
                )
        else:
            run_parity_bundle(
                arm_key=entry.arm_key,
                track=entry.track,
                command=command,
                output_directory=bundle,
                artifacts=entry.artifacts,
                environment=declared_environment,
                registry_path=registry_path,
            )
            manifest, _ = verify_parity_bundle(bundle, registry_path=registry_path)
        verified.append(bundle)
        verified_keys.add((manifest["arm_key"], manifest["track"]))
    if verified_keys != expected_keys:
        raise NativeParityMatrixError(
            f"verified matrix coverage drift: missing={sorted(expected_keys - verified_keys)}, "
            f"unexpected={sorted(verified_keys - expected_keys)}"
        )
    return aggregate_parity_bundles(
        verified,
        output_path=aggregate_output,
        generated_utc=generated_utc,
        require_all_current=True,
        registry_path=registry_path,
    )


def plan_record(entries: Sequence[MatrixEntry]) -> dict[str, Any]:
    return {
        "schema_version": MATRIX_PLAN_SCHEMA,
        "mode": "dry_run_no_models_loaded_no_bundles_written",
        "coverage_count": len(entries),
        "coverage": [entry.as_record() for entry in entries],
    }
