"""Shared immutable runtime/source policy for real native-parity workers."""
from __future__ import annotations

import base64
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import socket
import subprocess
import sys
from typing import Any, Mapping

PROFILE_ARMS = {
    "common": {
        "mantis_v1", "mantis_v2", "moment_small", "kronos_mini",
        "kronos_small", "chronos_v1", "chronos_bolt", "chronos_v2",
        "toto2_22m",
    },
    "timesfm": {"timesfm25"},
    "ttm": {"ttm_r2"},
    "moirai": {"moirai2_small"},
    "sundial": {"sundial_base"},
}
PACKAGE_PROFILES = {
    "common": {"torch": "2.13.0"},
    "timesfm": {"torch": "2.13.0", "transformers": "5.13.1"},
    "ttm": {"torch": "2.10.0", "transformers": "4.57.6"},
    "moirai": {"torch": "2.10.0", "uni2ts": "2.0.0"},
    "sundial": {
        "torch": "2.10.0", "transformers": "4.40.1",
        "huggingface-hub": "0.36.2",
    },
}
PROFILE_PYTHON = {
    "common": (3, 12), "timesfm": (3, 12), "ttm": (3, 12),
    "moirai": (3, 11), "sundial": (3, 12),
}
ARM_PACKAGES = {
    "mantis_v1": {"mantis-tsfm": "1.0.0"},
    "mantis_v2": {"mantis-tsfm": "1.0.0"},
    "moment_small": {"momentfm": "0.1.5"},
    "chronos_v1": {"chronos-forecasting": "2.3.1"},
    "chronos_bolt": {"chronos-forecasting": "2.3.1"},
    "chronos_v2": {"chronos-forecasting": "2.3.1"},
    "toto2_22m": {"toto-2": "2.0.0"},
}

GIT_SOURCE_ARMS = {
    "mantis_v1", "mantis_v2", "moment_small", "kronos_mini", "kronos_small",
    "timesfm25", "ttm_r2", "moirai2_small", "toto2_22m", "sundial_base",
}
PACKAGE_SOURCE_ARMS = {"chronos_v1", "chronos_bolt", "chronos_v2"}
RUNTIME_LOCK_SCHEMA = "ffm_native_runtime_lock_v1"


def _normalized_distribution_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _installed_distributions() -> list[dict[str, str]]:
    versions: dict[str, str] = {}
    for item in importlib.metadata.distributions():
        name = item.metadata.get("Name") or getattr(item, "name", None)
        if name:
            normalized = _normalized_distribution_name(str(name))
            item_version = str(item.version)
            previous = versions.get(normalized)
            if previous is not None and previous != item_version:
                raise NativeParityRuntimeError(
                    f"multiple installed versions found for distribution {normalized!r}: "
                    f"{previous!r}, {item_version!r}"
                )
            versions[normalized] = item_version
    return [
        {"name": name, "version": item_version}
        for name, item_version in sorted(versions.items())
    ]


def _hardware_runtime() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {
            "torch_importable": False, "cuda_available": False,
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_cuda_runtime": None, "cudnn_version": None, "devices": [],
            "driver_probe": {"status": "unavailable", "rows": []},
        }
    available = bool(torch.cuda.is_available())
    devices = []
    if available:
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            devices.append({
                "index": index,
                "name": str(props.name),
                "capability": [int(props.major), int(props.minor)],
                "total_memory": int(props.total_memory),
                "uuid": (str(props.uuid) if getattr(props, "uuid", None) else None),
            })
    try:
        probe = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,driver_version",
             "--format=csv,noheader,nounits"],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        driver = {
            "status": "measured",
            "rows": sorted(line.strip() for line in probe.stdout.splitlines() if line.strip()),
        }
    except (FileNotFoundError, subprocess.CalledProcessError):
        driver = {"status": "unavailable", "rows": []}
    cudnn = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
    return {
        "torch_importable": True,
        "cuda_available": available,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_cuda_runtime": getattr(torch.version, "cuda", None),
        "cudnn_version": int(cudnn) if cudnn is not None else None,
        "devices": devices,
        "driver_probe": driver,
    }


def measure_runtime_lock() -> dict[str, Any]:
    """Measure exact software plus exact measurable hardware/runtime identity."""
    return {
        "schema_version": RUNTIME_LOCK_SCHEMA,
        "comparison_policy": {
            "portable_software": "exact",
            "hardware_runtime": "exact_when_measurable_explicit_when_unavailable",
        },
        "portable_software": {
            "python_executable": str(Path(sys.executable).resolve()),
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "distributions": _installed_distributions(),
        },
        "hardware_runtime": _hardware_runtime(),
    }


def validate_runtime_lock(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or value.get("schema_version") != RUNTIME_LOCK_SCHEMA:
        raise NativeParityRuntimeError(
            f"runtime lock schema must be {RUNTIME_LOCK_SCHEMA!r}"
        )
    required = {
        "schema_version", "comparison_policy", "portable_software", "hardware_runtime",
    }
    if set(value) != required:
        raise NativeParityRuntimeError("runtime lock fields are incomplete or unknown")
    expected_policy = {
        "portable_software": "exact",
        "hardware_runtime": "exact_when_measurable_explicit_when_unavailable",
    }
    if value.get("comparison_policy") != expected_policy:
        raise NativeParityRuntimeError("runtime lock comparison policy drifted")
    portable = value.get("portable_software")
    hardware = value.get("hardware_runtime")
    if not isinstance(portable, Mapping) or not isinstance(hardware, Mapping):
        raise NativeParityRuntimeError("runtime lock surfaces must be objects")
    portable_fields = {
        "python_executable", "python_version", "python_implementation", "platform",
        "distributions",
    }
    if set(portable) != portable_fields:
        raise NativeParityRuntimeError("runtime lock portable-software fields drifted")
    for field in portable_fields - {"distributions"}:
        if not isinstance(portable.get(field), str) or not portable[field]:
            raise NativeParityRuntimeError(
                f"runtime lock portable field {field!r} must be a nonempty string"
            )
    rows = portable.get("distributions")
    if not isinstance(rows, list) or not rows:
        raise NativeParityRuntimeError("runtime lock must contain installed distributions")
    pairs: list[tuple[str, str]] = []
    for item in rows:
        if not isinstance(item, Mapping) or set(item) != {"name", "version"}:
            raise NativeParityRuntimeError("runtime lock distribution rows have invalid fields")
        name, item_version = item.get("name"), item.get("version")
        if (
            not isinstance(name, str) or not name
            or name != _normalized_distribution_name(name)
            or not isinstance(item_version, str) or not item_version
        ):
            raise NativeParityRuntimeError("runtime lock distribution identity is invalid")
        pairs.append((name, item_version))
    if pairs != sorted(pairs) or len({name for name, _ in pairs}) != len(pairs):
        raise NativeParityRuntimeError(
            "runtime lock distribution names must be normalized, unique and sorted"
        )
    hardware_fields = {
        "torch_importable", "cuda_available", "visible_devices", "torch_cuda_runtime",
        "cudnn_version", "devices", "driver_probe",
    }
    if set(hardware) != hardware_fields:
        raise NativeParityRuntimeError("runtime lock hardware fields drifted")
    if not isinstance(hardware.get("torch_importable"), bool) or not isinstance(
        hardware.get("cuda_available"), bool
    ):
        raise NativeParityRuntimeError("runtime lock hardware availability flags must be boolean")
    if hardware.get("visible_devices") is not None and not isinstance(
        hardware.get("visible_devices"), str
    ):
        raise NativeParityRuntimeError("runtime lock visible_devices must be string or null")
    if hardware.get("torch_cuda_runtime") is not None and not isinstance(
        hardware.get("torch_cuda_runtime"), str
    ):
        raise NativeParityRuntimeError("runtime lock Torch CUDA version must be string or null")
    if hardware.get("cudnn_version") is not None and not isinstance(
        hardware.get("cudnn_version"), int
    ):
        raise NativeParityRuntimeError("runtime lock cuDNN version must be integer or null")
    devices = hardware.get("devices")
    if not isinstance(devices, list):
        raise NativeParityRuntimeError("runtime lock devices must be a list")
    indices = []
    for device in devices:
        if not isinstance(device, Mapping) or set(device) != {
            "index", "name", "capability", "total_memory", "uuid",
        }:
            raise NativeParityRuntimeError("runtime lock device fields drifted")
        capability = device.get("capability")
        if (
            not isinstance(device.get("index"), int)
            or not isinstance(device.get("name"), str) or not device["name"]
            or not isinstance(capability, list) or len(capability) != 2
            or any(not isinstance(item, int) for item in capability)
            or not isinstance(device.get("total_memory"), int)
            or device["total_memory"] <= 0
            or (device.get("uuid") is not None and not isinstance(device["uuid"], str))
        ):
            raise NativeParityRuntimeError("runtime lock device identity is invalid")
        indices.append(device["index"])
    if indices != list(range(len(indices))):
        raise NativeParityRuntimeError("runtime lock device indices must be contiguous and ordered")
    if bool(devices) != bool(hardware["cuda_available"]):
        raise NativeParityRuntimeError("runtime lock CUDA availability disagrees with devices")
    driver = hardware.get("driver_probe")
    if not isinstance(driver, Mapping) or set(driver) != {"status", "rows"}:
        raise NativeParityRuntimeError("runtime lock driver probe fields drifted")
    if driver.get("status") not in {"measured", "unavailable"}:
        raise NativeParityRuntimeError("runtime lock driver probe status is invalid")
    driver_rows = driver.get("rows")
    if (
        not isinstance(driver_rows, list)
        or any(not isinstance(item, str) or not item for item in driver_rows)
        or driver_rows != sorted(set(driver_rows))
        or (driver["status"] == "unavailable" and driver_rows)
    ):
        raise NativeParityRuntimeError("runtime lock driver rows are invalid")
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


class NativeParityRuntimeError(RuntimeError):
    """Raised when an execution process differs from its declared native profile."""


def runtime_profile_for_arm(arm_key: str) -> str:
    matches = [profile for profile, arms in PROFILE_ARMS.items() if arm_key in arms]
    if len(matches) != 1:
        raise NativeParityRuntimeError(f"arm has no unique runtime profile: {arm_key}")
    return matches[0]


def validate_runtime_profile(profile: str, arm_key: str) -> dict[str, str]:
    expected = runtime_profile_for_arm(arm_key)
    if profile != expected:
        raise NativeParityRuntimeError(
            f"{arm_key} requires runtime profile {expected!r}, got {profile!r}"
        )
    if sys.version_info[:2] != PROFILE_PYTHON[profile]:
        raise NativeParityRuntimeError(
            f"runtime profile {profile!r} requires Python "
            f"{PROFILE_PYTHON[profile]}, got {sys.version_info[:2]}"
        )
    required = {**PACKAGE_PROFILES[profile], **ARM_PACKAGES.get(arm_key, {})}
    try:
        actual = {
            name: importlib.metadata.version(name)
            for name in required
        }
    except importlib.metadata.PackageNotFoundError as exc:
        raise NativeParityRuntimeError(
            f"required package is not installed: {exc.name}"
        ) from exc
    if actual != required:
        raise NativeParityRuntimeError(
            f"runtime profile {profile!r} drifted: expected {required}, got {actual}"
        )
    return actual


def install_python_network_guard(policy: str = "python_socket_deny") -> None:
    """Install the same Python-level offline guard used by native parity workers."""
    if policy != "python_socket_deny":
        raise NativeParityRuntimeError(f"unsupported network policy: {policy!r}")

    def audit(event: str, _args: tuple[object, ...]) -> None:
        if event in {"socket.connect", "socket.bind", "socket.getaddrinfo"}:
            raise NativeParityRuntimeError(
                f"Python network policy denied audit event {event!r}"
            )

    sys.addaudithook(audit)

    def denied(*_args: object, **_kwargs: object) -> object:
        raise NativeParityRuntimeError(
            "Python network policy denied name resolution/connection"
        )

    socket.create_connection = denied
    socket.getaddrinfo = denied


def validate_distribution_record(path: str | Path) -> Path:
    """Verify every hashed file in an installed wheel's RECORD manifest."""
    root = Path(path).resolve()
    record = root / "RECORD"
    if not root.name.endswith(".dist-info") or not record.is_file():
        raise RuntimeError(f"installed distribution RECORD root required: {root}")
    checked = 0
    with record.open(newline="", encoding="utf-8") as stream:
        for relative, encoded_hash, size in csv.reader(stream):
            target = (root.parent / relative).resolve()
            if not target.is_file():
                raise RuntimeError(f"distribution RECORD file is missing: {target}")
            if not encoded_hash:
                if target != record:
                    raise RuntimeError(f"distribution RECORD lacks a hash: {relative}")
                continue
            algorithm, encoded = encoded_hash.split("=", 1)
            if algorithm != "sha256":
                raise RuntimeError(f"unsupported RECORD hash algorithm: {algorithm}")
            actual = hashlib.sha256(target.read_bytes()).digest()
            expected = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
            if actual != expected:
                raise RuntimeError(f"distribution RECORD hash mismatch: {target}")
            if size and target.stat().st_size != int(size):
                raise RuntimeError(f"distribution RECORD size mismatch: {target}")
            checked += 1
    if not checked:
        raise RuntimeError(f"distribution RECORD contains no hashed files: {record}")
    return root
