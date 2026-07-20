"""Fail-closed filesystem and canonical-JSON primitives for authority bundles.

This module contains transport security only.  Authority schemas and business
invariants remain in their owning consumer modules.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Any


class AuthorityBundleIOError(ValueError):
    """Raised when immutable bundle transport invariants do not hold."""


class AuthorityBundleFileSizeError(AuthorityBundleIOError):
    """Raised when a regular authority artifact does not have its declared size."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise AuthorityBundleIOError("value is not canonical-JSON encodable") from exc


def content_sha256(value: dict[str, Any], semantic_field: str) -> str:
    payload = dict(value)
    payload.pop(semantic_field, None)
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise AuthorityBundleIOError(f"{label} must be a lowercase SHA-256")
    return value


def canonical_absolute_path(value: str | Path, label: str) -> Path:
    raw = os.fspath(value)
    if (
        not isinstance(raw, str)
        or not PurePosixPath(raw).is_absolute()
        or PurePosixPath(raw).as_posix() != raw
        or raw == "/"
        or any(part in {"", ".", ".."} for part in PurePosixPath(raw).parts)
    ):
        raise AuthorityBundleIOError(f"{label} must be a canonical absolute path")
    return Path(raw)


def _safe_leaf(value: str, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or PurePosixPath(value).name != value
        or value in {".", ".."}
        or "/" in value
    ):
        raise AuthorityBundleIOError(f"{label} is not a safe bundle leaf")
    return value


def _identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _open_directory_from_root(path: Path, label: str) -> int:
    """Open an absolute directory without ever resolving a pathname component."""
    descriptor = -1
    try:
        descriptor = os.open(
            os.path.sep,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        for component in path.parts[1:]:
            child = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = child
        return descriptor
    except OSError as exc:
        if descriptor >= 0:
            os.close(descriptor)
        raise AuthorityBundleIOError(
            f"{label} crosses a symlink or unavailable directory component"
        ) from exc


def _read_json_at(
    directory_fd: int,
    leaf: str,
    *,
    label: str,
    max_bytes: int,
    max_nodes: int,
    max_depth: int,
) -> tuple[dict[str, Any], str]:
    leaf = _safe_leaf(leaf, label)
    if type(max_bytes) is not int or max_bytes <= 0:
        raise AuthorityBundleIOError("max_bytes must be a positive integer")
    if type(max_nodes) is not int or max_nodes <= 0:
        raise AuthorityBundleIOError("max_nodes must be a positive integer")
    if type(max_depth) is not int or max_depth < 0:
        raise AuthorityBundleIOError("max_depth must be a nonnegative integer")
    descriptor = -1
    try:
        named_before = os.stat(leaf, dir_fd=directory_fd, follow_symlinks=False)
        if stat.S_ISLNK(named_before.st_mode):
            raise AuthorityBundleIOError(f"{label} is a symlink")
        if (
            not stat.S_ISREG(named_before.st_mode)
            or named_before.st_nlink != 1
            or named_before.st_size <= 0
            or named_before.st_size > max_bytes
        ):
            raise AuthorityBundleIOError(f"{label} is not a bounded regular file")
        descriptor = os.open(
            leaf,
            os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=directory_fd,
        )
        opened = os.fstat(descriptor)
        if _identity(named_before) != _identity(opened):
            raise AuthorityBundleIOError(f"{label} changed while being opened")
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining:
            block = os.read(descriptor, min(1 << 20, remaining))
            if not block:
                break
            chunks.append(block)
            remaining -= len(block)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        named_after = os.stat(leaf, dir_fd=directory_fd, follow_symlinks=False)
        if (
            len(raw) > max_bytes
            or _identity(opened) != _identity(after)
            or _identity(after) != _identity(named_after)
        ):
            raise AuthorityBundleIOError(f"{label} changed while being read")
    except AuthorityBundleIOError:
        raise
    except OSError as exc:
        raise AuthorityBundleIOError(f"cannot safely read {label}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise AuthorityBundleIOError(f"duplicate JSON key in {label}: {key}")
            result[key] = value
        return result

    try:
        document = json.loads(
            raw,
            object_pairs_hook=pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                AuthorityBundleIOError(
                    f"non-finite JSON constant in {label}: {value}"
                )
            ),
        )
    except AuthorityBundleIOError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise AuthorityBundleIOError(f"cannot parse {label}") from exc
    nodes = 0

    def walk(value: Any, depth: int) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > max_nodes or depth > max_depth:
            raise AuthorityBundleIOError(f"{label} exceeds JSON limits")
        if isinstance(value, dict):
            for child in value.values():
                walk(child, depth + 1)
        elif isinstance(value, list):
            for child in value:
                walk(child, depth + 1)
        elif value is not None and not isinstance(value, (str, bool, int)):
            raise AuthorityBundleIOError(f"unsupported scalar in {label}")

    try:
        walk(document, 0)
    except RecursionError as exc:
        raise AuthorityBundleIOError(f"{label} exceeds JSON limits") from exc
    if not isinstance(document, dict) or raw != canonical_json_bytes(document):
        raise AuthorityBundleIOError(f"{label} is not canonical JSON")
    return document, hashlib.sha256(raw).hexdigest()


def read_canonical_json_file(
    path: str | Path,
    *,
    label: str,
    max_bytes: int,
    max_nodes: int,
    max_depth: int = 20,
) -> tuple[Path, dict[str, Any], str]:
    """Read one canonical JSON file through an identity-bound parent dirfd."""
    source = canonical_absolute_path(path, label)
    parent_fd = _open_directory_from_root(source.parent, f"{label} parent")
    parent_identity = _identity(os.fstat(parent_fd))
    try:
        document, physical = _read_json_at(
            parent_fd,
            source.name,
            label=label,
            max_bytes=max_bytes,
            max_nodes=max_nodes,
            max_depth=max_depth,
        )
        reopened_parent_fd = _open_directory_from_root(
            source.parent, f"{label} parent"
        )
        try:
            if _identity(os.fstat(reopened_parent_fd)) != parent_identity:
                raise AuthorityBundleIOError(f"{label} parent path changed")
        finally:
            os.close(reopened_parent_fd)
        return source, document, physical
    finally:
        os.close(parent_fd)


def read_regular_file(
    path: str | Path,
    *,
    label: str,
    max_bytes: int,
) -> tuple[Path, bytes, str]:
    """Read one bounded identity-stable, single-link regular file.

    Unlike :func:`read_canonical_json_file`, this primitive deliberately owns no
    serialization semantics.  It is the transport boundary for authority inputs
    such as YAML whose owning module must validate the returned bytes.
    """
    if type(max_bytes) is not int or max_bytes <= 0:
        raise AuthorityBundleIOError("max_bytes must be a positive integer")
    source = canonical_absolute_path(path, label)
    parent_fd = _open_directory_from_root(source.parent, f"{label} parent")
    parent_identity = _identity(os.fstat(parent_fd))
    descriptor = -1
    try:
        named_before = os.stat(source.name, dir_fd=parent_fd, follow_symlinks=False)
        if stat.S_ISLNK(named_before.st_mode):
            raise AuthorityBundleIOError(f"{label} is a symlink")
        if (
            not stat.S_ISREG(named_before.st_mode)
            or named_before.st_nlink != 1
            or named_before.st_size <= 0
            or named_before.st_size > max_bytes
        ):
            raise AuthorityBundleIOError(f"{label} is not a bounded regular file")
        descriptor = os.open(
            source.name,
            os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=parent_fd,
        )
        opened = os.fstat(descriptor)
        if _identity(named_before) != _identity(opened):
            raise AuthorityBundleIOError(f"{label} changed while being opened")
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining:
            block = os.read(descriptor, min(1 << 20, remaining))
            if not block:
                break
            chunks.append(block)
            remaining -= len(block)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        named_after = os.stat(source.name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            len(raw) > max_bytes
            or _identity(opened) != _identity(after)
            or _identity(after) != _identity(named_after)
        ):
            raise AuthorityBundleIOError(f"{label} changed while being read")
        reopened_parent_fd = _open_directory_from_root(
            source.parent, f"{label} parent"
        )
        try:
            if _identity(os.fstat(reopened_parent_fd)) != parent_identity:
                raise AuthorityBundleIOError(f"{label} parent path changed")
        finally:
            os.close(reopened_parent_fd)
        return source, raw, hashlib.sha256(raw).hexdigest()
    except AuthorityBundleIOError:
        raise
    except OSError as exc:
        raise AuthorityBundleIOError(f"cannot safely read {label}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_fd)


def sha256_regular_file(
    path: str | Path, *, label: str, expected_size: int,
) -> tuple[Path, str]:
    """Hash one identity-stable, single-link regular file through a parent dirfd."""
    if type(expected_size) is not int or expected_size < 0:
        raise AuthorityBundleIOError("expected_size must be a nonnegative integer")
    source = canonical_absolute_path(path, label)
    parent_fd = _open_directory_from_root(source.parent, f"{label} parent")
    parent_identity = _identity(os.fstat(parent_fd))
    descriptor = -1
    try:
        named_before = os.stat(source.name, dir_fd=parent_fd, follow_symlinks=False)
        if stat.S_ISLNK(named_before.st_mode):
            raise AuthorityBundleIOError(f"{label} is a symlink")
        if not stat.S_ISREG(named_before.st_mode) or named_before.st_nlink != 1:
            raise AuthorityBundleIOError(f"{label} is not the declared regular file")
        if named_before.st_size != expected_size:
            raise AuthorityBundleFileSizeError(f"{label} differs from its declared size")
        descriptor = os.open(
            source.name,
            os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=parent_fd,
        )
        opened = os.fstat(descriptor)
        if _identity(named_before) != _identity(opened):
            raise AuthorityBundleIOError(f"{label} changed while being opened")
        digest = hashlib.sha256()
        total = 0
        while True:
            block = os.read(descriptor, 1 << 20)
            if not block:
                break
            total += len(block)
            if total > expected_size:
                raise AuthorityBundleIOError(f"{label} exceeds its declared size")
            digest.update(block)
        after = os.fstat(descriptor)
        named_after = os.stat(source.name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            total != expected_size
            or _identity(opened) != _identity(after)
            or _identity(after) != _identity(named_after)
        ):
            raise AuthorityBundleIOError(f"{label} changed while being hashed")
        reopened_parent_fd = _open_directory_from_root(source.parent, f"{label} parent")
        try:
            if _identity(os.fstat(reopened_parent_fd)) != parent_identity:
                raise AuthorityBundleIOError(f"{label} parent path changed")
        finally:
            os.close(reopened_parent_fd)
        return source, digest.hexdigest()
    except AuthorityBundleIOError:
        raise
    except OSError as exc:
        raise AuthorityBundleIOError(f"cannot safely hash {label}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_fd)


class VerifiedDirectoryReader:
    """Directory-descriptor reader that detects replacement and mutation."""

    def __init__(self, path: str | Path, *, label: str) -> None:
        self.path = canonical_absolute_path(path, label)
        self._label = label
        self._parent_fd = -1
        self._directory_fd = -1
        self._leaf = self.path.name
        try:
            self._parent_fd = _open_directory_from_root(
                self.path.parent, f"{label} parent"
            )
            try:
                self._directory_fd = os.open(
                    self._leaf,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                    dir_fd=self._parent_fd,
                )
            except OSError as exc:
                raise AuthorityBundleIOError(
                    f"{label} is a symlink or unavailable directory"
                ) from exc
            directory = os.fstat(self._directory_fd)
            named = os.stat(self._leaf, dir_fd=self._parent_fd, follow_symlinks=False)
            if not stat.S_ISDIR(directory.st_mode) or _identity(directory) != _identity(named):
                raise AuthorityBundleIOError(f"{label} directory identity mismatch")
            self._initial_identity = _identity(directory)
            self._initial_names = self.names()
        except BaseException:
            self.close()
            raise

    def names(self) -> list[str]:
        if self._directory_fd < 0:
            raise AuthorityBundleIOError(f"{self._label} reader is closed")
        return sorted(os.listdir(self._directory_fd))

    def read_json(
        self,
        leaf: str,
        *,
        label: str,
        max_bytes: int,
        max_nodes: int,
        max_depth: int = 20,
    ) -> tuple[dict[str, Any], str]:
        return _read_json_at(
            self._directory_fd,
            leaf,
            label=label,
            max_bytes=max_bytes,
            max_nodes=max_nodes,
            max_depth=max_depth,
        )

    def assert_unchanged(self) -> None:
        if self.names() != self._initial_names:
            raise AuthorityBundleIOError(f"{self._label} directory closure changed")
        directory = os.fstat(self._directory_fd)
        named = os.stat(self._leaf, dir_fd=self._parent_fd, follow_symlinks=False)
        if _identity(directory) != self._initial_identity or _identity(named) != self._initial_identity:
            raise AuthorityBundleIOError(f"{self._label} directory identity changed")
        reopened_fd = _open_directory_from_root(self.path, self._label)
        try:
            if _identity(os.fstat(reopened_fd)) != self._initial_identity:
                raise AuthorityBundleIOError(
                    f"{self._label} absolute path identity changed"
                )
        finally:
            os.close(reopened_fd)

    def close(self) -> None:
        if self._directory_fd >= 0:
            os.close(self._directory_fd)
            self._directory_fd = -1
        if self._parent_fd >= 0:
            os.close(self._parent_fd)
            self._parent_fd = -1

    def __enter__(self) -> "VerifiedDirectoryReader":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


__all__ = [
    "AuthorityBundleIOError",
    "VerifiedDirectoryReader",
    "canonical_absolute_path",
    "canonical_json_bytes",
    "content_sha256",
    "read_canonical_json_file",
    "read_regular_file",
    "require_sha256",
    "sha256_regular_file",
]
