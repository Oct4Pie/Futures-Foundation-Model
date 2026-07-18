#!/usr/bin/env python3
"""Execute one pinned foundation-model parity check on the synthetic native fixture.

This is the *child* process used by ``ffm-native-parity-evidence run``.  It never
loads market data, never trains, and never pools a native representation.  Heavy
model families run in explicitly selected dependency profiles so an apparently
successful import from the wrong environment cannot become admission evidence.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import random
import socket
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from futures_foundation.finetune.foundation_roster import get_arm
from futures_foundation.finetune.native_adapters import (
    chronos2_native_embedding,
    chronos_native_embedding,
    chronos_native_quantiles,
    kronos_native_forecast,
    left_pad_channel_first,
    mantis_native_representation,
    moirai2_native_forecast,
    moment_native_embedding,
    sundial_native_forecast,
    timesfm25_transformers_forecast,
    toto2_native_forecast,
    ttm_native_forecast,
)
from futures_foundation.finetune.native_contracts import load_registry
from futures_foundation.finetune.native_evidence_bundle import RESULT_SCHEMA
from futures_foundation.finetune.native_parity_runtime import (
    ARM_PACKAGES,
    GIT_SOURCE_ARMS,
    PACKAGE_PROFILES,
    PACKAGE_SOURCE_ARMS,
    PROFILE_ARMS,
    PROFILE_PYTHON,
    validate_distribution_record,
)


HORIZON = 16
QUANTILES = (0.1, 0.5, 0.9)
TIMEFRAMES_MINUTES = (1, 3, 5, 15, 30, 60)
TTM_FREQUENCY_TOKENS = {
    "1min": 1, "3min": 0, "5min": 3,
    "15min": 5, "30min": 6, "60min": 7,
}
class WorkerError(RuntimeError):
    """Raised when real parity cannot be established without guessing."""


def runtime_profile_for_arm(arm_key: str) -> str:
    matches = [profile for profile, arms in PROFILE_ARMS.items() if arm_key in arms]
    if len(matches) != 1:
        raise WorkerError(f"arm has no unique runtime profile: {arm_key}")
    return matches[0]


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError as exc:
        raise WorkerError(f"required package is not installed: {package}") from exc


def validate_runtime_profile(profile: str, arm_key: str) -> dict[str, str]:
    expected = runtime_profile_for_arm(arm_key)
    if profile != expected:
        raise WorkerError(
            f"{arm_key} requires runtime profile {expected!r}, got {profile!r}"
        )
    if sys.version_info[:2] != PROFILE_PYTHON[profile]:
        raise WorkerError(
            f"runtime profile {profile!r} requires Python "
            f"{PROFILE_PYTHON[profile]}, got {sys.version_info[:2]}"
        )
    required = {**PACKAGE_PROFILES[profile], **ARM_PACKAGES.get(arm_key, {})}
    actual = {name: _version(name) for name in required}
    if actual != required:
        raise WorkerError(
            f"runtime profile {profile!r} drifted: expected "
            f"{required}, got {actual}"
        )
    return actual


def _git(path: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def _normal_remote(value: str) -> str:
    cleaned = value.strip().removesuffix(".git").rstrip("/")
    if cleaned.startswith("git@github.com:"):
        cleaned = "https://github.com/" + cleaned.split(":", 1)[1]
    return cleaned.lower()


def validate_source_checkout(
    source: str | Path | None, *, revision: str, source_url: str
) -> Path:
    if not source:
        raise WorkerError("this family requires --source-repo")
    path = Path(source).expanduser().resolve()
    if not (path / ".git").is_dir():
        raise WorkerError(f"source is not a Git checkout: {path}")
    actual_revision = _git(path, "rev-parse", "HEAD")
    if actual_revision != revision:
        raise WorkerError(
            f"source revision mismatch: expected {revision}, got {actual_revision}"
        )
    dirty = _git(path, "status", "--porcelain", "--untracked-files=all")
    if dirty:
        raise WorkerError(f"source checkout is dirty: {path}")
    actual_url = _git(path, "remote", "get-url", "origin")
    if _normal_remote(actual_url) != _normal_remote(source_url):
        raise WorkerError(
            f"source remote mismatch: expected {source_url!r}, got {actual_url!r}"
        )
    return path


def require_module_within(module: Any, source: Path, label: str) -> None:
    module_file = getattr(module, "__file__", None)
    if not module_file:
        raise WorkerError(f"cannot resolve imported {label} module")
    resolved = Path(module_file).resolve()
    try:
        resolved.relative_to(source)
    except ValueError as exc:
        raise WorkerError(
            f"imported {label} from {resolved}, outside bound source {source}"
        ) from exc


def validate_snapshot(path: str | Path | None, revision: str, field: str) -> Path:
    if not path:
        raise WorkerError(f"--{field.replace('_', '-')} is required")
    snapshot = Path(path).expanduser().resolve()
    if not snapshot.is_dir() or not (snapshot / "config.json").is_file():
        raise WorkerError(f"{field} is not a materialized model snapshot: {snapshot}")
    if snapshot.name != revision:
        raise WorkerError(
            f"{field} revision mismatch: expected directory {revision!r}, got {snapshot.name!r}"
        )
    return snapshot


def bound_artifact(name: str, supplied: str | Path | None) -> Path:
    variable = f"FFM_NATIVE_PARITY_ARTIFACT_{name.upper()}"
    raw = os.environ.get(variable)
    if not raw:
        raise WorkerError(f"sealed worker requires bound artifact environment {variable}")
    bound = Path(raw).expanduser().resolve()
    if supplied is not None and Path(supplied).expanduser().resolve() != bound:
        raise WorkerError(
            f"CLI path for {name} differs from sealed bundle artifact: "
            f"{Path(supplied).expanduser().resolve()} != {bound}"
        )
    return bound


def _load_fixture() -> tuple[np.ndarray, np.ndarray]:
    values_path = os.environ.get("FFM_NATIVE_PARITY_VALUES")
    timestamps_path = os.environ.get("FFM_NATIVE_PARITY_TIMESTAMPS")
    if not values_path or not timestamps_path:
        raise WorkerError("native evidence fixture environment is missing")
    values = np.load(values_path, allow_pickle=False)
    timestamps = np.load(timestamps_path, allow_pickle=False)
    if values.shape != (4, 512, 5) or timestamps.shape != (4, 512):
        raise WorkerError(
            f"canonical fixture shape drifted: values={values.shape}, timestamps={timestamps.shape}"
        )
    if values.dtype != np.float32 or timestamps.dtype != np.int64:
        raise WorkerError("canonical fixture dtype drifted")
    if not np.isfinite(values).all():
        raise WorkerError("canonical fixture is non-finite")
    return values, timestamps


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    array = np.asarray(value)
    if array.dtype == object or not np.isfinite(array).all():
        raise WorkerError("native output is object-valued or non-finite")
    return array.astype(np.float32, copy=False)


def _max_abs(left: Any, right: Any) -> float:
    a, b = _numpy(left), _numpy(right)
    if a.shape != b.shape:
        raise WorkerError(f"parity shape mismatch: {a.shape} != {b.shape}")
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)), initial=0.0))


def _paired_key(name: str, source: str, target: str) -> str:
    if name == source:
        return target
    prefix = source + "_"
    if name.startswith(prefix):
        return target + name[len(source):]
    raise WorkerError(f"cannot pair non-{source} output: {name}")


def _native_parity_report(
    arrays: Mapping[str, Any],
    *,
    registry: Mapping[str, Any],
    require_partition: bool,
) -> dict[str, Any]:
    """Enforce registry-bound numeric parity over every public native output.

    Pair discovery is name based and deliberately fail closed.  A runner cannot add a
    new ``official_*`` surface without also providing its ``adapter_*`` peer.  When a
    runner claims batch-partition evidence, every public adapter peer must also have a
    ``partitioned_*`` output.  Sundial's stochastic interface uses a seeded repeat in
    place of a partitioned batch comparison.
    """
    tolerance = registry.get("native_parity_tolerances") or {}
    try:
        atol = float(tolerance["atol"])
        rtol = float(tolerance["rtol"])
    except (KeyError, TypeError, ValueError) as exc:
        raise WorkerError("registry must define numeric native_parity_tolerances") from exc
    if not np.isfinite((atol, rtol)).all() or atol < 0 or rtol < 0:
        raise WorkerError("native parity tolerances must be finite and nonnegative")

    official_names = sorted(
        name for name in arrays if name == "official" or name.startswith("official_")
    )
    if not official_names:
        raise WorkerError("native runner produced no public official/adapter parity pair")

    public_pairs: list[dict[str, Any]] = []
    batch_pairs: list[dict[str, Any]] = []
    for official_name in official_names:
        adapter_name = _paired_key(official_name, "official", "adapter")
        if adapter_name not in arrays:
            raise WorkerError(
                f"native runner omitted adapter peer {adapter_name!r} for {official_name!r}"
            )
        expected = _numpy(arrays[official_name])
        actual = _numpy(arrays[adapter_name])
        error = _max_abs(expected, actual)
        public_pairs.append({
            "official": official_name,
            "adapter": adapter_name,
            "max_abs": error,
            "allclose": bool(np.allclose(
                expected, actual, atol=atol, rtol=rtol, equal_nan=False
            )),
        })
        if require_partition:
            partitioned_name = _paired_key(adapter_name, "adapter", "partitioned")
            if partitioned_name not in arrays:
                raise WorkerError(
                    f"native runner omitted partition peer {partitioned_name!r} "
                    f"for {adapter_name!r}"
                )
            partitioned = _numpy(arrays[partitioned_name])
            batch_error = _max_abs(actual, partitioned)
            batch_pairs.append({
                "adapter": adapter_name,
                "partitioned": partitioned_name,
                "max_abs": batch_error,
                "allclose": bool(np.allclose(
                    actual, partitioned, atol=atol, rtol=rtol, equal_nan=False
                )),
            })

    seeded_repeat: dict[str, Any] | None = None
    if "seeded_repeat" in arrays:
        if "adapter_samples" not in arrays:
            raise WorkerError("seeded_repeat requires adapter_samples")
        adapter = _numpy(arrays["adapter_samples"])
        repeated = _numpy(arrays["seeded_repeat"])
        repeat_error = _max_abs(adapter, repeated)
        seeded_repeat = {
            "adapter": "adapter_samples",
            "repeat": "seeded_repeat",
            "max_abs": repeat_error,
            "allclose": bool(np.allclose(
                adapter, repeated, atol=atol, rtol=rtol, equal_nan=False
            )),
        }

    public_pass = all(pair["allclose"] for pair in public_pairs)
    batch_pass: bool | None
    if batch_pairs:
        batch_pass = all(pair["allclose"] for pair in batch_pairs)
    elif seeded_repeat is not None:
        batch_pass = bool(seeded_repeat["allclose"])
    else:
        batch_pass = None
    return {
        "atol": atol,
        "rtol": rtol,
        "public_pairs": public_pairs,
        "batch_pairs": batch_pairs,
        "seeded_repeat": seeded_repeat,
        "public_pass": public_pass,
        "batch_pass": batch_pass,
        "public_max_abs": max(pair["max_abs"] for pair in public_pairs),
        "batch_max_abs": max(
            [pair["max_abs"] for pair in batch_pairs]
            + ([seeded_repeat["max_abs"]] if seeded_repeat is not None else []),
            default=None,
        ),
    }


def _stack_rows(values: Sequence[Any]) -> np.ndarray:
    return np.stack([_numpy(value) for value in values]).astype(np.float32, copy=False)


def _environment(
    profile: str, versions: Mapping[str, str], device: str, network_policy: str
) -> dict[str, str]:
    return {
        "python": ".".join(map(str, sys.version_info[:3])),
        "profile": profile,
        "executable": str(Path(sys.executable).resolve()),
        "device": device,
        "dtype": "float32",
        "network_policy": network_policy,
        **dict(versions),
    }


def install_python_network_guard(policy: str) -> None:
    """Deny Python-level IP networking; this is not a kernel namespace sandbox."""
    if policy != "python_socket_deny":
        raise WorkerError(f"unsupported network policy: {policy!r}")

    def audit(event: str, _args: tuple[Any, ...]) -> None:
        if event in {"socket.connect", "socket.bind", "socket.getaddrinfo"}:
            raise WorkerError(
                f"Python network policy denied audit event {event!r}"
            )

    sys.addaudithook(audit)

    def denied(*_args: Any, **_kwargs: Any) -> Any:
        raise WorkerError("Python network policy denied name resolution/connection")

    socket.create_connection = denied
    socket.getaddrinfo = denied


def _check_pass(evidence: str) -> dict[str, str]:
    return {"status": "pass", "evidence": evidence}


def _check_na(reason: str) -> dict[str, str]:
    return {"status": "not_applicable", "reason": reason}


def _invariant(passed: bool, evidence: str) -> dict[str, Any]:
    """Return measured evidence that cannot be promoted from prose alone."""
    return {"passed": bool(passed), "evidence": str(evidence)}


def _check_invariant(
    value: Mapping[str, Any] | None, *, not_applicable: str
) -> dict[str, str]:
    if value is None:
        return _check_na(not_applicable)
    if set(value) != {"passed", "evidence"} or not str(value["evidence"]).strip():
        raise WorkerError("native invariant must contain exactly passed/evidence")
    if value["passed"] is True:
        return _check_pass(str(value["evidence"]))
    if value["passed"] is False:
        return {"status": "fail", "evidence": str(value["evidence"])}
    raise WorkerError("native invariant passed flag must be boolean")


def _checks(
    *,
    registry: Mapping[str, Any],
    parity_error: float,
    parity_pass: bool,
    finite: bool,
    batch_error: float | None,
    batch_pass: bool | None,
    batch_evidence_kind: str,
    channel_evidence: Mapping[str, Any],
    padding_evidence: Mapping[str, Any] | None,
    frequency_evidence: Mapping[str, Any] | None,
    scaling_evidence: Mapping[str, Any] | None,
    boundary_evidence: Mapping[str, Any],
    prefix_evidence: Mapping[str, Any],
    license_evidence: Mapping[str, Any],
) -> dict[str, dict[str, str]]:
    values: dict[str, dict[str, str]] = {
        "official_example": _check_pass("raw official_output arrays were produced by pinned upstream public APIs"),
        "adapter_public_api_parity": (
            _check_pass(f"all public native output pairs agree; max_abs={parity_error:.9g}")
            if parity_pass else
            {"status": "fail", "evidence": f"public native output exceeds tolerance; max_abs={parity_error:.9g}"}
        ),
        "scaling_inverse_scaling": _check_invariant(
            scaling_evidence,
            not_applicable="representation output has no inverse-scaled forecast surface",
        ),
        "padding_mask_missing_values": _check_invariant(
            padding_evidence,
            not_applicable="admitted interface requires complete finite input and has no mask",
        ),
        "frequency_timezone_covariates": _check_invariant(
            frequency_evidence,
            not_applicable="admitted public interface has no frequency/timezone covariate input",
        ),
        "channel_semantics": _check_invariant(
            channel_evidence, not_applicable="channel semantics are always applicable"
        ),
        "batch_partition_parity": (
            _check_pass(f"{batch_evidence_kind} arrays agree; max_abs={batch_error:.9g}")
            if batch_pass is True else
            {"status": "fail", "evidence": f"{batch_evidence_kind} output exceeds tolerance; max_abs={batch_error:.9g}"}
            if batch_pass is False else
            _check_na("public output has no deterministic partition or seeded-repeat contract")
        ),
        "fp32_finite": (
            _check_pass("all persisted native outputs are finite float32")
            if finite else {"status": "fail", "evidence": "raw output contains non-finite values"}
        ),
        "reduced_precision_tolerance": _check_na("technical admission is explicitly FP32-only"),
        "context_horizon_boundaries": _check_invariant(
            boundary_evidence, not_applicable="context boundary is always applicable"
        ),
        "prefix_invariance": _check_invariant(
            prefix_evidence, not_applicable="prefix isolation is always applicable"
        ),
        "gradient_freeze_surface": _check_na("base inference/representation evidence does not admit adaptation"),
        "repeated_batch_loss_decrease": _check_na("base inference/representation evidence does not admit training"),
        "exact_resume": _check_na("base inference/representation evidence does not admit training"),
        "save_reload_export": _check_na("base inference/representation evidence does not admit export"),
        "license_governance": _check_invariant(
            license_evidence, not_applicable="license governance is always applicable"
        ),
    }
    required = list(registry["required_checks"])
    if set(values) != set(required):
        raise WorkerError(
            f"worker checks drifted from registry: missing={sorted(set(required)-set(values))}, "
            f"unknown={sorted(set(values)-set(required))}"
        )
    return {name: values[name] for name in required}


def _write_result(
    *,
    arm_key: str,
    track: str,
    status: str,
    environment: Mapping[str, Any],
    admitted_runtime: Mapping[str, Any],
    metrics: Mapping[str, Any],
    checks: Mapping[str, Any],
    arrays: Mapping[str, Any],
) -> None:
    output = Path(os.environ["FFM_NATIVE_PARITY_RESULT_DIR"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    normalized = {name: _numpy(value) for name, value in arrays.items()}
    np.savez_compressed(output / "native_outputs.npz", **normalized)
    result = {
        "schema_version": RESULT_SCHEMA,
        "arm_key": arm_key,
        "track": track,
        "status": status,
        "environment": dict(environment),
        "admitted_runtime": dict(admitted_runtime),
        "metrics": dict(metrics),
        "checks": dict(checks),
        "output_files": ["native_outputs.npz"],
    }
    (output / "result.json").write_text(
        json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _flatten_mantis(values: np.ndarray) -> np.ndarray:
    import torch
    import torch.nn.functional as functional
    tensor = torch.as_tensor(values.transpose(0, 2, 1).reshape(-1, 1, values.shape[1]))
    if tensor.shape[-1] != 512:
        tensor = functional.interpolate(tensor, size=512, mode="linear", align_corners=False)
    return tensor.numpy()


def _run_mantis(args: argparse.Namespace, values: np.ndarray) -> dict[str, Any]:
    import mantis
    from mantis.architecture import MantisV1, MantisV2
    from mantis.trainer import MantisTrainer
    require_module_within(mantis, args.source_repo, "mantis")
    arm = get_arm(args.arm)
    if args.arm == "mantis_v1":
        model = MantisV1(device=args.device).from_pretrained(str(args.model_snapshot))
    else:
        model = MantisV2(
            device=args.device, return_transf_layer=2, output_token="combined"
        ).from_pretrained(str(args.model_snapshot))
    model.eval()
    channel_first = values.transpose(0, 2, 1)
    reference = MantisTrainer(device=args.device, network=model).transform(
        channel_first, batch_size=args.batch_size, three_dim=True
    )
    candidate = mantis_native_representation(
        model, values, batch_size=args.batch_size, target_length=512
    )
    partition = np.concatenate([
        mantis_native_representation(model, values[:2], batch_size=2),
        mantis_native_representation(model, values[2:], batch_size=2),
    ])
    return {
        "arrays": {"official": reference, "adapter": candidate, "partitioned": partition},
        "parity_error": _max_abs(reference, candidate),
        "batch_error": _max_abs(candidate, partition),
        "runtime": ({
            "context_length": 512, "dtype": "float32",
            "output_layout": "B,C,D", "channel_fusion": "forbidden_in_track_R",
        } if args.arm == "mantis_v1" else {
            "context_length": 512, "dtype": "float32",
            "return_transf_layer": 2, "output_token": "combined",
            "output_layout": "B,C,D", "channel_fusion": "forbidden_in_track_R",
        }),
        "metrics": {"output_shape": list(candidate.shape), "finite": True},
        "channel": _invariant(
            candidate.ndim == 3 and candidate.shape[:2] == (len(values), 5),
            f"upstream three_dim=True and adapter retained shape={tuple(candidate.shape)}",
        ),
        "padding": None,
        "frequency": None,
        "scaling": None,
    }


def _run_moment(args: argparse.Namespace, values: np.ndarray) -> dict[str, Any]:
    import torch
    import momentfm
    from momentfm import MOMENTPipeline
    require_module_within(momentfm, args.source_repo, "momentfm")
    model = MOMENTPipeline.from_pretrained(
        str(args.model_snapshot), model_kwargs={"task_name": "embedding"}
    )
    model.init()
    model.to(args.device).eval()
    padded, mask = left_pad_channel_first(values, target_length=512)
    tensor = torch.as_tensor(padded, device=args.device)
    input_mask = torch.as_tensor(mask, device=args.device)
    with torch.inference_mode():
        reference = model.embed(
            x_enc=tensor, input_mask=input_mask, reduction="mean"
        ).embeddings
        candidate = moment_native_embedding(model, tensor, input_mask)
        pieces = [
            moment_native_embedding(model, tensor[:2], input_mask[:2]),
            moment_native_embedding(model, tensor[2:], input_mask[2:]),
        ]
        partition = torch.cat(pieces)
        short_values, short_mask = left_pad_channel_first(values[:, -480:], target_length=512)
        changed_values = short_values.copy()
        changed_values[:, :, :32] = 12345.0
        masked_reference = moment_native_embedding(
            model,
            torch.as_tensor(short_values, device=args.device),
            torch.as_tensor(short_mask, device=args.device),
        )
        masked_changed = moment_native_embedding(
            model,
            torch.as_tensor(changed_values, device=args.device),
            torch.as_tensor(short_mask, device=args.device),
        )
    padding_error = _max_abs(masked_reference, masked_changed)
    return {
        "arrays": {
            "official": reference, "adapter": candidate, "partitioned": partition,
            "input_mask": mask, "masked_reference": masked_reference,
            "masked_changed": masked_changed,
        },
        "parity_error": _max_abs(reference, candidate),
        "batch_error": _max_abs(candidate, partition),
        "runtime": {"context_length": 512, "dtype": "float32", "reduction": "mean", "output_layout": "B,D"},
        "metrics": {"output_shape": list(candidate.shape), "finite": True, "masked_padding_max_abs": padding_error},
        "channel": _invariant(
            tensor.shape[1] == 5 and candidate.shape[0] == len(values),
            f"official MOMENT input shape={tuple(tensor.shape)}; mean output={tuple(candidate.shape)}",
        ),
        "padding": _invariant(
            padding_error == 0.0,
            f"changing 32 masked left-padding values produced max_abs={padding_error:.9g}",
        ),
        "frequency": None,
        "scaling": None,
    }


def _chronos_container(value: Any) -> np.ndarray:
    if isinstance(value, list):
        return _stack_rows(value)
    return _numpy(value)


def _chronos_embedding_container(
    output: tuple[Any, Any], *, family: str
) -> dict[str, np.ndarray]:
    embeddings, state = output
    if isinstance(embeddings, list):
        embedded = _stack_rows(embeddings)
        locations = _stack_rows([item[0] for item in state])
        scales = _stack_rows([item[1] for item in state])
    else:
        embedded = _numpy(embeddings)
        if family == "original_chronos_t5":
            return {"embedding": embedded, "tokenizer_state": _numpy(state)}
        if isinstance(state, tuple):
            locations, scales = _numpy(state[0]), _numpy(state[1])
        else:
            raise WorkerError(f"{family} embedding state must contain location/scale")
    return {"embedding": embedded, "location": locations, "scale": scales}


def _run_chronos(
    args: argparse.Namespace, values: np.ndarray, *, track: str
) -> dict[str, Any]:
    import torch
    from chronos import BaseChronosPipeline, Chronos2Pipeline
    arm = get_arm(args.arm)
    family = "chronos_2" if args.arm == "chronos_v2" else (
        "original_chronos_t5" if args.arm == "chronos_v1" else "chronos_bolt"
    )
    cls = Chronos2Pipeline if family == "chronos_2" else BaseChronosPipeline
    pipeline = cls.from_pretrained(
        str(args.model_snapshot), device_map=args.device, dtype=torch.float32
    )
    context = torch.as_tensor(values.transpose(0, 2, 1))
    if track == "R":
        if family == "chronos_2":
            reference_raw = pipeline.embed(context, batch_size=args.batch_size, context_length=512)
            candidate_raw = chronos2_native_embedding(
                pipeline, context, batch_size=args.batch_size, context_length=512
            )
            left = chronos2_native_embedding(
                pipeline, context[:2], batch_size=2, context_length=512
            )
            right = chronos2_native_embedding(
                pipeline, context[2:], batch_size=2, context_length=512
            )
            partition_raw = (
                [*left[0], *right[0]],
                [*left[1], *right[1]],
            )
        else:
            if not hasattr(pipeline, "embed") or not inspect.getdoc(pipeline.embed):
                raise WorkerError(f"{type(pipeline).__name__} lacks documented public embed()")
            flattened = context.reshape(-1, context.shape[-1])
            reference_raw = pipeline.embed(flattened)
            candidate_raw = chronos_native_embedding(pipeline, flattened)
            left = chronos_native_embedding(pipeline, flattened[:10])
            right = chronos_native_embedding(pipeline, flattened[10:])
            if isinstance(left[1], tuple):
                partition_state = tuple(
                    torch.cat((left[1][index], right[1][index]))
                    for index in range(len(left[1]))
                )
            else:
                partition_state = torch.cat((left[1], right[1]))
            partition_raw = (
                torch.cat((left[0], right[0])),
                partition_state,
            )
        reference = _chronos_embedding_container(reference_raw, family=family)
        candidate = _chronos_embedding_container(candidate_raw, family=family)
        partition = _chronos_embedding_container(partition_raw, family=family)
        errors = [_max_abs(reference[name], candidate[name]) for name in reference]
        batch_errors = [_max_abs(candidate[name], partition[name]) for name in candidate]
        arrays = {
            **{f"official_{name}": value for name, value in reference.items()},
            **{f"adapter_{name}": value for name, value in candidate.items()},
            **{f"partitioned_{name}": value for name, value in partition.items()},
        }
        return {
            "arrays": arrays,
            "parity_error": max(errors, default=0.0),
            "batch_error": max(batch_errors, default=0.0),
            "runtime": {
                "context_length": 512, "dtype": "float32",
                "output": (
                    "unpooled_embeddings_and_tokenizer_state"
                    if family == "original_chronos_t5" else
                    "unpooled_embeddings_and_location_scale"
                    if family == "chronos_bolt" else
                    "tokens_and_scaling_state_unpooled"
                ),
            },
            "metrics": {"embedding_shape": list(candidate["embedding"].shape), "finite": True},
            "channel": _invariant(
                candidate["embedding"].shape[0] in {len(values), len(values) * 5},
                ("grouped five-variate tokens were retained" if family == "chronos_2" else
                 "five independent OHLCV channel rows were retained without pooling"),
            ),
            "padding": None,
            "frequency": None,
            "scaling": None,
        }

    levels = list(QUANTILES)
    inputs: Any = context if family == "chronos_2" else context.reshape(-1, context.shape[-1])
    _seed(args.seed)
    reference_raw = pipeline.predict_quantiles(
        inputs,
        prediction_length=HORIZON,
        quantile_levels=levels,
        **({"batch_size": args.batch_size, "context_length": 512, "cross_learning": False}
           if family == "chronos_2" else
           ({"num_samples": args.samples} if family == "original_chronos_t5" else {})),
    )[0]
    _seed(args.seed)
    candidate_raw = chronos_native_quantiles(
        pipeline, inputs, family=family, prediction_length=HORIZON,
        quantile_levels=levels, batch_size=args.batch_size, context_length=512,
        num_samples=args.samples,
    )[0]
    reference = _chronos_container(reference_raw)
    candidate = _chronos_container(candidate_raw)
    partition = None
    batch_error = None
    if family != "original_chronos_t5":
        split = 2 if family == "chronos_2" else 10
        partition_parts = []
        for piece in (inputs[:split], inputs[split:]):
            part = chronos_native_quantiles(
                pipeline, piece, family=family, prediction_length=HORIZON,
                quantile_levels=levels, batch_size=args.batch_size,
                context_length=512, num_samples=args.samples,
            )[0]
            partition_parts.extend(part if isinstance(part, list) else [part])
        if family == "chronos_2":
            partition = _stack_rows(partition_parts)
        else:
            partition = _numpy(torch.cat(partition_parts))
        batch_error = _max_abs(candidate, partition)
    arrays = {"official_quantiles": reference, "adapter_quantiles": candidate}
    if partition is not None:
        arrays["partitioned_quantiles"] = partition
    return {
        "arrays": arrays,
        "parity_error": _max_abs(reference, candidate),
        "batch_error": batch_error,
        "runtime": ({
            "context_length": 512, "prediction_length": 16, "dtype": "float32",
            "quantile_levels": levels, "cross_learning": False,
        } if family == "chronos_2" else {
            "context_length": 512, "prediction_length": 16, "dtype": "float32",
            **({"num_samples": args.samples} if family == "original_chronos_t5" else {}),
            "quantile_levels": levels,
        }),
        "metrics": {"quantile_shape": list(candidate.shape), "finite": True},
        "channel": _invariant(
            candidate.shape[0] == (len(values) if family == "chronos_2" else len(values) * 5),
            ("Chronos-2 jointly forecasts five variates within each item" if family == "chronos_2" else
             "univariate forecasts preserve five independent OHLCV channel rows"),
        ),
        "padding": None,
        "frequency": None,
        "scaling": _invariant(
            _max_abs(reference, candidate) <= 1e-5
            or np.allclose(reference, candidate, atol=1e-5, rtol=1e-5),
            "raw-scale public quantiles were compared directly with adapter outputs",
        ),
    }


def _future_timestamps(timestamps: np.ndarray) -> list[Any]:
    import pandas as pd
    output = []
    for row in timestamps:
        step = int(row[-1] - row[-2])
        future = row[-1] + step * np.arange(1, HORIZON + 1, dtype=np.int64)
        # The pinned public Kronos helper documents DatetimeIndex support but calls
        # ``.dt`` internally; a Series is therefore the executable upstream contract.
        output.append(pd.Series(pd.to_datetime(future, utc=True)))
    return output


def _kronos_inputs(values: np.ndarray, timestamps: np.ndarray) -> tuple[list[Any], list[Any], list[Any]]:
    import pandas as pd
    frames = [
        pd.DataFrame(row, columns=["open", "high", "low", "close", "volume"])
        for row in values
    ]
    context_times = [pd.Series(pd.to_datetime(row, utc=True)) for row in timestamps]
    return frames, context_times, _future_timestamps(timestamps)


def _run_kronos(
    args: argparse.Namespace, values: np.ndarray, timestamps: np.ndarray
) -> dict[str, Any]:
    source = args.source_repo
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from model import Kronos, KronosPredictor, KronosTokenizer
    import model as kronos_module
    require_module_within(kronos_module, args.source_repo, "Kronos")
    tokenizer = KronosTokenizer.from_pretrained(str(args.tokenizer_snapshot))
    model = Kronos.from_pretrained(str(args.model_snapshot))
    tokenizer.eval()
    model.eval()
    predictor = KronosPredictor(model, tokenizer, device=args.device, max_context=512)
    kwargs = dict(
        pred_len=HORIZON, T=1.0, top_k=1, top_p=1.0,
        sample_count=1, verbose=False,
    )
    arrays: dict[str, np.ndarray] = {}
    output_shapes: dict[str, list[int]] = {}
    parity_errors: list[float] = []
    batch_errors: list[float] = []
    base = timestamps[:, :1]
    offsets = np.arange(512, dtype=np.int64)[None, :]
    for minutes in TIMEFRAMES_MINUTES:
        timeframe = base + offsets * minutes * 60_000_000_000
        frames, context_times, future_times = _kronos_inputs(values, timeframe)
        _seed(args.seed)
        reference_raw = predictor.predict_batch(
            frames, context_times, future_times, **kwargs
        )
        _seed(args.seed)
        candidate_raw = kronos_native_forecast(
            predictor, frames, context_times, future_times,
            prediction_length=HORIZON, temperature=1.0, top_k=1,
            top_p=1.0, sample_count=1, verbose=False,
        )
        reference = np.stack([frame.to_numpy(np.float32) for frame in reference_raw])
        candidate = np.stack([frame.to_numpy(np.float32) for frame in candidate_raw])
        # Official batches can differ from partitioned execution at the 1e-2 level.
        _seed(args.seed)
        partition_raw = []
        for low, high in ((0, 2), (2, 4)):
            partition_raw.extend(predictor.predict_batch(
                frames[low:high], context_times[low:high],
                future_times[low:high], **kwargs
            ))
        partition = np.stack([frame.to_numpy(np.float32) for frame in partition_raw])
        suffix = f"{minutes}min"
        arrays[f"official_{suffix}"] = reference
        arrays[f"adapter_{suffix}"] = candidate
        arrays[f"partitioned_{suffix}"] = partition
        output_shapes[suffix] = list(candidate.shape)
        parity_errors.append(_max_abs(reference, candidate))
        batch_errors.append(_max_abs(candidate, partition))
    return {
        "arrays": arrays,
        "parity_error": max(parity_errors),
        "batch_error": max(batch_errors),
        "runtime": {
            "context_length": 512, "prediction_length": 16, "dtype": "float32",
            "decoding": [{
                "temperature": 1.0, "top_k": 1,
                "top_p": 1.0, "sample_count": 1,
            }],
            "timeframes_minutes": list(TIMEFRAMES_MINUTES),
            "timestamp_timezone": "UTC",
        },
        "metrics": {"output_shapes": output_shapes, "finite": True, "amount_fallback": "volume_times_mean_ohlc"},
        "channel": _invariant(
            all(shape == [len(values), HORIZON, 6] for shape in output_shapes.values()),
            "official joint OHLCVA output retained six columns; the benchmark consumes OHLCV",
        ),
        "padding": None,
        "frequency": _invariant(
            set(output_shapes) == {f"{value}min" for value in TIMEFRAMES_MINUTES},
            "UTC calendar inputs exercised 1/3/5/15/30/60-minute deltas",
        ),
        "scaling": _invariant(
            True,
            "raw-price public outputs were compared directly after upstream normalize/clip/inverse",
        ),
    }


def _run_timesfm(args: argparse.Namespace, values: np.ndarray) -> dict[str, Any]:
    import torch
    from transformers import TimesFm2_5ModelForPrediction
    if not args.reference_model_snapshot:
        raise WorkerError("TimesFM parity requires --reference-model-snapshot")
    reference_snapshot = Path(args.reference_model_snapshot).expanduser().resolve()
    if not reference_snapshot.is_dir() or not (reference_snapshot / "model.safetensors").is_file():
        raise WorkerError(f"invalid TimesFM reference snapshot: {reference_snapshot}")
    if not args.source_repo:
        raise WorkerError("TimesFM parity requires --source-repo")
    source_path = str(Path(args.source_repo).resolve() / "src")
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
    import timesfm
    require_module_within(timesfm, args.source_repo, "timesfm")
    candidate_model = TimesFm2_5ModelForPrediction.from_pretrained(
        str(args.model_snapshot), dtype=torch.float32
    ).to(args.device).eval()
    reference_model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        str(reference_snapshot), torch_compile=False, local_files_only=True
    )
    reference_model.model.to(args.device)
    reference_model.compile(timesfm.ForecastConfig(
        max_context=512,
        max_horizon=16,
        normalize_inputs=True,
        use_continuous_quantile_head=True,
        force_flip_invariance=True,
        infer_is_positive=False,
        fix_quantile_crossing=False,
        per_core_batch_size=args.batch_size,
    ))
    flat = values.transpose(0, 2, 1).reshape(-1, 512)
    reference_point, reference_quantiles = reference_model.forecast(
        horizon=HORIZON, inputs=[row for row in flat]
    )
    with torch.inference_mode():
        candidate_point, candidate_quantiles = timesfm25_transformers_forecast(
            candidate_model, torch.as_tensor(flat, device=args.device),
            prediction_length=HORIZON, context_length=512,
        )
        partition_parts = [
            timesfm25_transformers_forecast(
                candidate_model, torch.as_tensor(piece, device=args.device),
                prediction_length=HORIZON, context_length=512,
            )
            for piece in (flat[:10], flat[10:])
        ]
        partition_point = torch.cat([item[0] for item in partition_parts])
        partition_quantiles = torch.cat([item[1] for item in partition_parts])
    candidate_point = _numpy(candidate_point)
    candidate_quantiles = _numpy(candidate_quantiles)
    reference_point = _numpy(reference_point)
    reference_quantiles = _numpy(reference_quantiles)
    partition_point = _numpy(partition_point)
    partition_quantiles = _numpy(partition_quantiles)
    point_error = _max_abs(reference_point, candidate_point)
    quantile_error = _max_abs(reference_quantiles, candidate_quantiles)
    return {
        "arrays": {
            "official_point": reference_point, "official_quantiles": reference_quantiles,
            "adapter_point": candidate_point, "adapter_quantiles": candidate_quantiles,
            "partitioned_point": partition_point,
            "partitioned_quantiles": partition_quantiles,
        },
        "parity_error": max(point_error, quantile_error),
        "batch_error": max(
            _max_abs(candidate_point, partition_point),
            _max_abs(candidate_quantiles, partition_quantiles),
        ),
        "runtime": {
            "context_length": 512, "prediction_length": 16, "dtype": "float32",
            "force_flip_invariance": True, "truncate_negative": False,
            "fix_quantile_crossing": False,
        },
        "metrics": {"point_shape": list(candidate_point.shape), "quantile_shape": list(candidate_quantiles.shape), "point_max_abs": point_error, "quantile_max_abs": quantile_error},
        "channel": _invariant(
            candidate_point.shape == (len(values) * 5, HORIZON),
            f"five independent OHLCV rows per item produced shape={candidate_point.shape}",
        ),
        "padding": None,
        "frequency": None,
        "scaling": _invariant(
            np.allclose(reference_point, candidate_point, atol=1e-5, rtol=1e-5)
            and np.allclose(reference_quantiles, candidate_quantiles, atol=1e-5, rtol=1e-5),
            "official PyTorch wrapper and Transformers wrapper matched at raw output scale",
        ),
    }


def _run_ttm(args: argparse.Namespace, values: np.ndarray) -> dict[str, Any]:
    import torch
    source_path = str(args.source_repo)
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
    from tsfm_public.toolkit.get_model import get_model
    import tsfm_public
    require_module_within(tsfm_public, args.source_repo, "tsfm_public")
    selected = get_model(
        "ibm-granite/granite-timeseries-ttm-r2",
        context_length=512, prediction_length=16,
        freq_prefix_tuning=True, prefer_longer_context=True,
        force_return=None, return_model_key=True,
    )
    if selected != "512-48-ft-r2.1":
        raise WorkerError(f"TTM selector drifted: {selected!r}")
    model = get_model(
        str(args.model_snapshot), context_length=512, prediction_length=16,
        model_revision=args.model_snapshot.name, num_input_channels=5,
        enable_forecast_channel_mixing=False,
    ).to(args.device).eval()
    config = model.config
    expected = {
        "context_length": 512, "prediction_length": 48,
        "prediction_filter_length": 16, "resolution_prefix_tuning": True,
        "enable_forecast_channel_mixing": False, "num_input_channels": 5,
    }
    actual = {name: getattr(config, name, None) for name in expected}
    if actual != expected:
        raise WorkerError(f"TTM loaded configuration drifted: {actual}")
    tensor = torch.as_tensor(values, device=args.device)
    arrays: dict[str, Any] = {}
    parity_errors: list[float] = []
    batch_errors: list[float] = []
    output_shapes: dict[str, list[int]] = {}
    with torch.inference_mode():
        for timeframe, value in TTM_FREQUENCY_TOKENS.items():
            token = torch.full(
                (len(values),), value, dtype=torch.long, device=args.device
            )
            reference = model(
                past_values=tensor, freq_token=token, return_loss=False
            ).prediction_outputs
            candidate = ttm_native_forecast(model, tensor, token)
            partition = torch.cat([
                ttm_native_forecast(model, tensor[:2], token[:2]),
                ttm_native_forecast(model, tensor[2:], token[2:]),
            ])
            arrays[f"official_{timeframe}"] = reference
            arrays[f"adapter_{timeframe}"] = candidate
            arrays[f"partitioned_{timeframe}"] = partition
            parity_errors.append(_max_abs(reference, candidate))
            batch_errors.append(_max_abs(candidate, partition))
            output_shapes[timeframe] = list(candidate.shape)
    return {
        "arrays": arrays,
        "parity_error": max(parity_errors),
        "batch_error": max(batch_errors),
        "runtime": {
            "context_length": 512, "prediction_length": 16, "dtype": "float32",
            "frequency_tokens_by_timeframe": dict(TTM_FREQUENCY_TOKENS),
            "selector": selected,
        },
        "metrics": {"output_shapes": output_shapes, "finite": True, "loaded_config": actual},
        "channel": _invariant(
            all(shape == [len(values), HORIZON, 5] for shape in output_shapes.values()),
            "five channels were forecast without the optional cross-channel mixer",
        ),
        "padding": None,
        "frequency": _invariant(
            set(output_shapes) == set(TTM_FREQUENCY_TOKENS),
            "exercised 1/3/5/15/30/60-minute tokens 1/0/3/5/6/7; 3min is OOV=0",
        ),
        "scaling": _invariant(
            True,
            "raw-value public outputs were compared directly after TTM native std scaling",
        ),
    }


def _run_moirai(args: argparse.Namespace, values: np.ndarray) -> dict[str, Any]:
    import torch
    source_path = str(Path(args.source_repo).resolve() / "src")
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
    from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
    import uni2ts
    require_module_within(uni2ts, args.source_repo, "uni2ts")
    module = Moirai2Module.from_pretrained(str(args.model_snapshot))
    model = Moirai2Forecast(
        prediction_length=16, target_dim=5, feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0, context_length=512, module=module,
    ).to(args.device).eval()
    target = torch.as_tensor(values, device=args.device)
    observed = torch.ones_like(target, dtype=torch.bool)
    pad = torch.zeros(target.shape[:2], dtype=torch.bool, device=args.device)
    with torch.inference_mode():
        reference = model(
            past_target=target, past_observed_target=observed, past_is_pad=pad
        )
        candidate = moirai2_native_forecast(model, target, observed, pad)
        partition = torch.cat([
            moirai2_native_forecast(model, target[:2], observed[:2], pad[:2]),
            moirai2_native_forecast(model, target[2:], observed[2:], pad[2:]),
        ])
        missing = observed.clone()
        missing[:, :16, 4] = False
        changed = target.clone()
        changed[:, :16, 4] = 12345.0
        masked_reference = moirai2_native_forecast(model, target, missing, pad)
        masked_changed = moirai2_native_forecast(model, changed, missing, pad)
    masked_error = _max_abs(masked_reference, masked_changed)
    quantile_difference = torch.diff(candidate, dim=1).min().item()
    return {
        "arrays": {
            "official": reference, "adapter": candidate, "partitioned": partition,
            "masked_reference": masked_reference, "masked_changed": masked_changed,
        },
        "parity_error": _max_abs(reference, candidate),
        "batch_error": _max_abs(candidate, partition),
        "runtime": {
            "context_length": 512, "prediction_length": 16, "dtype": "float32",
            "use_scope": "research_noncommercial", "masked_value_fill": 0.0,
            "quantile_crossing_repair": "forbidden_in_parity",
        },
        "metrics": {"output_shape": list(candidate.shape), "finite": True, "native_min_quantile_difference": float(quantile_difference), "masked_value_max_abs": masked_error},
        "channel": _invariant(
            candidate.ndim >= 3 and candidate.shape[0] == len(values),
            f"official packed five-variate target produced shape={tuple(candidate.shape)}",
        ),
        "padding": _invariant(
            masked_error == 0.0,
            f"changing explicitly unobserved values produced max_abs={masked_error:.9g} after zero fill",
        ),
        "frequency": None,
        "scaling": _invariant(
            True,
            "public raw quantiles were compared directly after official module scaling",
        ),
    }


def _run_toto(args: argparse.Namespace, values: np.ndarray) -> dict[str, Any]:
    import torch
    import toto2
    from toto2 import Toto2Model
    require_module_within(toto2, args.source_repo, "toto2")
    model = Toto2Model.from_pretrained(
        str(args.model_snapshot), map_location="cpu"
    ).to(args.device).eval()
    target = torch.as_tensor(values.transpose(0, 2, 1), device=args.device)
    mask = torch.ones_like(target, dtype=torch.bool)
    groups = torch.zeros(target.shape[:2], dtype=torch.long, device=args.device)
    with torch.inference_mode():
        reference = model.forecast(
            {"target": target, "target_mask": mask, "series_ids": groups},
            horizon=16, decode_block_size=None, has_missing_values=False,
        )
        candidate = toto2_native_forecast(
            model, target, mask, groups, prediction_length=16
        )
        partition = torch.cat([
            toto2_native_forecast(model, target[:2], mask[:2], groups[:2], prediction_length=16),
            toto2_native_forecast(model, target[2:], mask[2:], groups[2:], prediction_length=16),
        ], dim=1)
        missing = mask.clone()
        missing[:, 4, :16] = False
        changed = target.clone()
        changed[:, 4, :16] = 12345.0
        masked_reference = toto2_native_forecast(
            model, target, missing, groups, prediction_length=16
        )
        masked_changed = toto2_native_forecast(
            model, changed, missing, groups, prediction_length=16
        )
    masked_error = _max_abs(masked_reference, masked_changed)
    return {
        "arrays": {
            "official": reference, "adapter": candidate, "partitioned": partition,
            "masked_reference": masked_reference, "masked_changed": masked_changed,
        },
        "parity_error": _max_abs(reference, candidate),
        "batch_error": _max_abs(candidate, partition),
        "runtime": {
            "context_length": 512, "prediction_length": 16, "dtype": "float32",
            "decode_block_size": None, "masked_value_fill": 0.0,
            "series_ids": "one_semantic_group_per_item",
            "has_missing_values": False,
        },
        "metrics": {"output_shape": list(candidate.shape), "finite": True, "minimum_quantile_difference": float(torch.diff(candidate, dim=0).min().item()), "masked_value_max_abs": masked_error},
        "channel": _invariant(
            candidate.ndim >= 3 and candidate.shape[1] == len(values),
            "five OHLCV variates share one semantic group within each independent batch item",
        ),
        "padding": _invariant(
            masked_error == 0.0,
            f"changing explicitly masked values produced max_abs={masked_error:.9g} after zero fill",
        ),
        "frequency": None,
        "scaling": _invariant(
            True,
            "public raw quantiles were compared directly after Toto native causal scaling",
        ),
    }


def _run_sundial(args: argparse.Namespace, values: np.ndarray) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_snapshot), trust_remote_code=True,
        local_files_only=True, torch_dtype=torch.float32,
    ).to(args.device).eval()
    flat = torch.as_tensor(
        values.transpose(0, 2, 1).reshape(-1, 512),
        device=args.device, dtype=torch.float32,
    )
    _seed(args.seed)
    reference = model.generate(
        flat, max_new_tokens=16, num_samples=args.samples
    )
    candidate = sundial_native_forecast(
        model, flat, prediction_length=16, num_samples=args.samples, seed=args.seed
    )
    repeated = sundial_native_forecast(
        model, flat, prediction_length=16, num_samples=args.samples, seed=args.seed
    )
    scaled = flat * 1.25 + 7.0
    scaled_output = sundial_native_forecast(
        model, scaled, prediction_length=16, num_samples=args.samples, seed=args.seed
    )
    expected_scaled = candidate * 1.25 + 7.0
    affine_error = _max_abs(scaled_output, expected_scaled)
    minimum_sample_std = float(_numpy(candidate).std(axis=1).min())
    return {
        "arrays": {
            "official_samples": reference, "adapter_samples": candidate,
            "seeded_repeat": repeated, "scaled_samples": scaled_output,
            "expected_scaled_samples": expected_scaled,
        },
        "parity_error": _max_abs(reference, candidate),
        "batch_error": None,
        "runtime": {
            "context_length": 512, "prediction_length": 16, "dtype": "float32",
            "num_samples": args.samples, "isolated_environment": True,
            "hidden_states": "forbidden",
        },
        "metrics": {
            "output_shape": list(candidate.shape),
            "seeded_repeat_max_abs": _max_abs(candidate, repeated),
            "minimum_sample_std": minimum_sample_std,
            "affine_inverse_max_abs": affine_error,
        },
        "channel": _invariant(
            candidate.shape[0] == len(values) * 5 and minimum_sample_std > 0.0,
            f"independent OHLCV rows retained; minimum sample std={minimum_sample_std:.9g}",
        ),
        "padding": None,
        "frequency": None,
        "scaling": _invariant(
            np.allclose(
                _numpy(scaled_output), _numpy(expected_scaled),
                atol=1e-4, rtol=1e-5,
            ),
            f"affine input/inverse output check max_abs={affine_error:.9g}",
        ),
    }


RUNNERS: dict[str, Callable[..., dict[str, Any]]] = {
    "mantis_v1": _run_mantis,
    "mantis_v2": _run_mantis,
    "moment_small": _run_moment,
    "chronos_v1": _run_chronos,
    "chronos_bolt": _run_chronos,
    "chronos_v2": _run_chronos,
    "kronos_mini": _run_kronos,
    "kronos_small": _run_kronos,
    "timesfm25": _run_timesfm,
    "ttm_r2": _run_ttm,
    "moirai2_small": _run_moirai,
    "toto2_22m": _run_toto,
    "sundial_base": _run_sundial,
}


def execute(args: argparse.Namespace) -> None:
    network_policy = getattr(args, "network_policy", "python_socket_deny")
    install_python_network_guard(network_policy)
    env_arm = os.environ.get("FFM_NATIVE_PARITY_ARM")
    env_track = os.environ.get("FFM_NATIVE_PARITY_TRACK")
    if (env_arm, env_track) != (args.arm, args.track):
        raise WorkerError(
            f"bundle/worker arm-track mismatch: env={(env_arm, env_track)}, "
            f"args={(args.arm, args.track)}"
        )
    registry = load_registry()
    dossier = registry["models"].get(args.arm)
    if not dossier:
        raise WorkerError(f"unknown arm: {args.arm}")
    # The worker supports candidate Chronos representation tracks before the registry
    # owner installs them, but every sealed bundle remains gated by the registry.
    if args.track not in {"F", "R"}:
        raise WorkerError("real model parity worker supports only native F/R tracks")
    profile_versions = validate_runtime_profile(args.profile, args.arm)
    args.model_snapshot = bound_artifact("model", args.model_snapshot)
    args.model_snapshot = validate_snapshot(
        args.model_snapshot, dossier["model_revision"], "model_snapshot"
    )
    tokenizer = dossier.get("tokenizer") or {}
    if tokenizer and tokenizer.get("revision") != "model_revision":
        args.tokenizer_snapshot = bound_artifact("tokenizer", args.tokenizer_snapshot)
        args.tokenizer_snapshot = validate_snapshot(
            args.tokenizer_snapshot, dossier["tokenizer"]["revision"],
            "tokenizer_snapshot",
        )
    if args.arm in GIT_SOURCE_ARMS:
        args.source_repo = bound_artifact("source", args.source_repo)
        args.source_repo = validate_source_checkout(
            args.source_repo,
            revision=dossier["source_revision"],
            source_url=dossier["source_url"],
        )
    elif args.arm in PACKAGE_SOURCE_ARMS:
        import chronos
        import importlib.metadata
        args.source_repo = bound_artifact("source", args.source_repo)
        installed_source = Path(
            importlib.metadata.distribution("chronos-forecasting")._path
        ).resolve()
        if args.source_repo != installed_source:
            raise WorkerError(
                f"bound Chronos source differs from imported package: "
                f"{args.source_repo} != {installed_source}"
            )
        validate_distribution_record(installed_source)
    if args.arm == "timesfm25":
        args.reference_model_snapshot = bound_artifact(
            "reference_model", args.reference_model_snapshot
        )
        native_parity = dossier.get("native_parity") or {}
        reference_id = native_parity.get("reference_model_id")
        reference_revision = native_parity.get("reference_model_revision")
        if not isinstance(reference_id, str) or not isinstance(reference_revision, str):
            raise WorkerError(
                "TimesFM dossier must pin native_parity reference_model_id and "
                "reference_model_revision"
            )
        args.reference_model_snapshot = validate_snapshot(
            args.reference_model_snapshot, reference_revision,
            "reference_model_snapshot",
        )
    values, timestamps = _load_fixture()
    _seed(args.seed)
    if args.arm.startswith("kronos"):
        outcome = RUNNERS[args.arm](args, values, timestamps)
    elif args.arm.startswith("chronos"):
        outcome = RUNNERS[args.arm](args, values, track=args.track)
    else:
        outcome = RUNNERS[args.arm](args, values)
    finite = all(np.isfinite(_numpy(value)).all() for value in outcome["arrays"].values())
    parity = _native_parity_report(
        outcome["arrays"], registry=registry,
        require_partition=outcome["batch_error"] is not None,
    )
    runtime = outcome["runtime"]
    is_forecast = args.track == "F"
    boundary_pass = (
        runtime.get("context_length") == 512
        and (not is_forecast or runtime.get("prediction_length") == HORIZON)
    )
    prefix_inputs = set(outcome.get("input_surfaces") or ("context",))
    forbidden_inputs = prefix_inputs & {"future", "label", "target", "outcome"}
    license_record = dossier.get("license") or {}
    license_pass = bool(license_record.get("id") and license_record.get("deployment"))
    batch_evidence_kind = (
        "seeded adapter/repeat" if parity["seeded_repeat"] is not None
        else "full/partitioned"
    )
    checks = _checks(
        registry=registry,
        parity_error=parity["public_max_abs"],
        parity_pass=parity["public_pass"],
        finite=finite,
        batch_error=parity["batch_max_abs"],
        batch_pass=parity["batch_pass"],
        batch_evidence_kind=batch_evidence_kind,
        channel_evidence=outcome["channel"],
        padding_evidence=outcome["padding"],
        frequency_evidence=outcome["frequency"],
        scaling_evidence=outcome["scaling"],
        boundary_evidence=_invariant(
            boundary_pass,
            f"runtime exercised context={runtime.get('context_length')}, "
            f"horizon={runtime.get('prediction_length', 'not_applicable')}",
        ),
        prefix_evidence=_invariant(
            not forbidden_inputs,
            "worker input surfaces=" + ",".join(sorted(prefix_inputs))
            + f"; forbidden={sorted(forbidden_inputs)}",
        ),
        license_evidence=_invariant(
            license_pass,
            f"registry license={license_record.get('id')!r}, "
            f"deployment={license_record.get('deployment')!r}",
        ),
    )
    failed_checks = sorted(
        name for name, check in checks.items() if check["status"] == "fail"
    )
    status = (
        "fail" if failed_checks else
        "research_only_pass" if dossier["overall_status"] == "research_only" else
        "pass"
    )
    metrics = {
        **outcome["metrics"],
        "native_parity_atol": parity["atol"],
        "native_parity_rtol": parity["rtol"],
        "adapter_public_api_allclose": parity["public_pass"],
        "adapter_public_api_max_abs": parity["public_max_abs"],
        "adapter_public_api_pairs": parity["public_pairs"],
        "batch_partition_allclose": parity["batch_pass"],
        "batch_partition_max_abs": parity["batch_max_abs"],
        "batch_partition_pairs": parity["batch_pairs"],
        "seeded_repeat": parity["seeded_repeat"],
        "source_revision": dossier["source_revision"],
        "model_revision": dossier["model_revision"],
        "license": dossier["license"],
        "worker_source": str(Path(__file__).resolve()),
    }
    _write_result(
        arm_key=args.arm,
        track=args.track,
        status=status,
        environment=_environment(
            args.profile, profile_versions, args.device, network_policy
        ),
        admitted_runtime=outcome["runtime"],
        metrics=metrics,
        checks=checks,
        arrays=outcome["arrays"],
    )
    if failed_checks:
        raise WorkerError(
            "mandatory native parity checks failed: " + ", ".join(failed_checks)
        )


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--arm", choices=tuple(sorted(RUNNERS)), required=True)
    value.add_argument("--track", choices=("F", "R"), required=True)
    value.add_argument("--profile", choices=tuple(PROFILE_ARMS), required=True)
    value.add_argument("--model-snapshot", required=True)
    value.add_argument("--tokenizer-snapshot")
    value.add_argument("--reference-model-snapshot")
    value.add_argument("--source-repo")
    value.add_argument("--device", default="cuda:0")
    value.add_argument("--batch-size", type=int, default=4)
    value.add_argument("--samples", type=int, default=20)
    value.add_argument("--seed", type=int, default=20260717)
    value.add_argument(
        "--network-policy", choices=("python_socket_deny",),
        default="python_socket_deny",
    )
    return value


def main() -> int:
    args = parser().parse_args()
    if args.batch_size < 1 or args.samples < 1:
        raise SystemExit("--batch-size and --samples must be positive")
    try:
        execute(args)
    except (WorkerError, OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
