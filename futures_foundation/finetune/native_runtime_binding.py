"""Fail-closed coupling between native admission artifacts and model execution.

Admission proves artifact identities.  Consumers must still execute *those same paths*;
checking an arbitrary ``--runtime-artifact`` beside a loader that uses a Hub ID is not a
security boundary.  This module deliberately contains no admission policy.  It only
checks loader/import provenance and validates installed-distribution code against its
admitted ``RECORD``.  Runner identity is measured separately by the admission gate.
"""
from __future__ import annotations

import base64
import hashlib
import importlib.metadata
import importlib.util
from pathlib import Path


class NativeRuntimeBindingError(ValueError):
    """Raised when execution cannot be coupled to admitted artifact paths."""


def require_same_path(alias: str | Path | None, admitted: Path, label: str) -> Path:
    """Reject a legacy loader alias that points anywhere except the admitted artifact."""
    if alias is not None and Path(alias).expanduser().resolve() != admitted:
        raise NativeRuntimeBindingError(
            f"{label} differs from the admitted execution artifact: "
            f"{Path(alias).expanduser().resolve()} != {admitted}"
        )
    return admitted


def require_module_within(module_file: str | None, source: Path, label: str) -> Path:
    if not module_file:
        raise NativeRuntimeBindingError(f"cannot resolve imported {label} module file")
    resolved = Path(module_file).resolve()
    try:
        resolved.relative_to(source.resolve())
    except ValueError as exc:
        raise NativeRuntimeBindingError(
            f"imported {label} from {resolved}, outside admitted source {source}"
        ) from exc
    return resolved


def require_import_origin(module_name: str, source: Path, label: str) -> Path:
    """Check module provenance before importing and executing its top-level code."""
    spec = importlib.util.find_spec(module_name)
    origin = None if spec is None else spec.origin
    if not origin:
        raise NativeRuntimeBindingError(f"cannot resolve {label} import origin")
    return require_module_within(origin, source, label)


def require_distribution_record(
    admitted_dist_info: Path,
    *,
    distribution_name: str,
    package_prefix: str,
) -> Path:
    """Verify installed package code using the admitted dist-info ``RECORD``.

    A dist-info directory alone does not prevent installed Python files from being edited.
    Every hashed file below ``package_prefix`` is therefore rehashed before use.
    """
    distribution = importlib.metadata.distribution(distribution_name)
    actual_dist_info = Path(distribution._path).resolve()  # type: ignore[attr-defined]
    admitted_dist_info = admitted_dist_info.resolve()
    if actual_dist_info != admitted_dist_info:
        raise NativeRuntimeBindingError(
            f"loaded {distribution_name} metadata from {actual_dist_info}, not admitted "
            f"{admitted_dist_info}"
        )
    verified = 0
    prefix = package_prefix.rstrip("/") + "/"
    for item in distribution.files or ():
        relative = str(item).replace("\\", "/")
        if not relative.startswith(prefix) or item.hash is None:
            continue
        if item.hash.mode != "sha256":
            raise NativeRuntimeBindingError(
                f"unsupported RECORD hash for {relative}: {item.hash.mode}"
            )
        path = Path(distribution.locate_file(item)).resolve()
        if not path.is_file():
            raise NativeRuntimeBindingError(f"installed package file is missing: {path}")
        encoded = base64.urlsafe_b64encode(hashlib.sha256(path.read_bytes()).digest())
        actual = encoded.rstrip(b"=").decode("ascii")
        if actual != item.hash.value:
            raise NativeRuntimeBindingError(
                f"installed package file differs from admitted RECORD: {path}"
            )
        verified += 1
    if verified == 0:
        raise NativeRuntimeBindingError(
            f"admitted RECORD contains no hashed files below {package_prefix!r}"
        )
    return Path(distribution.locate_file(package_prefix)).resolve()
