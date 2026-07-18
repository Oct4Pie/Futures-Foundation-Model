"""Fail-closed SSOT for native-training data authority.

This module reads only two packaged JSON contracts.  It never opens market data, sample
shards, labels, or OOS artifacts.  Corpus semantics remain owned by Corpus v3; this
authority binds that contract and records which materialized prerequisites do not exist.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from futures_foundation.corpus_v3 import CorpusV3Error, verify_contract

from .native_contracts import REGISTRY_PATH, NativeContractError, content_sha256


AUTHORITY_SCHEMA = "ffm_native_training_data_authority_v1"
AUTHORITY_ID = "corpus_v3_native_training_data_v1"
CORPUS_SCHEMA = "ffm_corpus_v3_contract_v1"
# Independent trust anchor for the packaged authority.  The authority JSON binds the Corpus v3
# contract; this code-pinned digest prevents a coherent pre-import rewrite of both mutable JSON
# files from redefining that pair as canonical.  It is an integrity anchor, not a second copy of
# any corpus semantics.  Changing the authority requires an explicit reviewed code change too.
TRUSTED_AUTHORITY_CONTENT_SHA256 = (
    "5bc2ecbb714d61be0e02df766128699f506d32d8de86ccb787d91bce81b6a528"
)
SOURCE_AUTHORITY_PATH = (
    Path(__file__).resolve().parents[2]
    / "config"
    / "foundation_models"
    / "native_training_data_authority_v1.json"
)
SOURCE_CORPUS_CONTRACT_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "corpus_v3" / "contract.json"
)
BINDING_BLOCKERS = {
    "sample_manifest": "sample_manifest_unresolved",
    "session_denominator": "session_denominator_unresolved",
    "expected_request_denominator": "expected_request_denominator_unresolved",
    "lifecycle_registry": "lifecycle_registry_unresolved",
    "roll_registry": "roll_registry_unresolved",
    "label_bundle": "label_bundle_unresolved",
}


def _allowed_authority_paths() -> tuple[Path, ...]:
    return tuple(dict.fromkeys((
        SOURCE_AUTHORITY_PATH.resolve(),
        REGISTRY_PATH.with_name("native_training_data_authority_v1.json").resolve(),
    )))


def _allowed_corpus_contract_paths() -> tuple[Path, ...]:
    installed = REGISTRY_PATH.parent.parent / "corpus_v3" / "contract.json"
    return tuple(dict.fromkeys((
        SOURCE_CORPUS_CONTRACT_PATH.resolve(), installed.resolve(),
    )))


def _resolve_allowed_path(
    candidates: Iterable[str | Path], allowed: tuple[Path, ...], field: str,
) -> Path:
    allowed_set = set(allowed)
    for value in candidates:
        candidate = Path(value).resolve()
        if candidate not in allowed_set:
            raise NativeContractError(f"{field} is outside the packaged authority locations")
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"{field} not found; checked " + ", ".join(str(path) for path in candidates)
    )


def resolve_training_data_authority_path(
    candidates: Iterable[str | Path] | None = None,
) -> Path:
    """Resolve only the source or installed authority artifact."""
    allowed = _allowed_authority_paths()
    return _resolve_allowed_path(candidates or allowed, allowed, "training data authority")


def resolve_corpus_contract_path(
    candidates: Iterable[str | Path] | None = None,
) -> Path:
    """Resolve only the source or installed Corpus v3 contract."""
    allowed = _allowed_corpus_contract_paths()
    return _resolve_allowed_path(candidates or allowed, allowed, "Corpus v3 contract")


AUTHORITY_PATH = resolve_training_data_authority_path()
CORPUS_CONTRACT_PATH = resolve_corpus_contract_path()


def _read_json(path: Path, field: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NativeContractError(f"{field} is not valid readable JSON: {path}") from exc


def _exact(value: Any, fields: set[str], field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise NativeContractError(f"{field} must be an object")
    if set(value) != fields:
        raise NativeContractError(
            f"{field} fields mismatch: missing={sorted(fields-set(value))}, "
            f"unknown={sorted(set(value)-fields)}"
        )
    return value


def _validate_authority(value: Any, corpus_contract: Mapping[str, Any]) -> dict[str, Any]:
    item = _exact(value, {
        "schema_version", "authority_id", "status", "non_authorizing",
        "corpus_contract_ref", "unresolved_bindings", "blocker_tags",
    }, "training data authority")
    if item["schema_version"] != AUTHORITY_SCHEMA or item["authority_id"] != AUTHORITY_ID:
        raise NativeContractError("training data authority identity is invalid")
    if item["status"] != "blocked" or item["non_authorizing"] is not True:
        raise NativeContractError("training data authority must remain blocked and non-authorizing")
    try:
        verify_contract(corpus_contract)
    except CorpusV3Error as exc:
        raise NativeContractError("bound Corpus v3 contract is invalid") from exc
    ref = _exact(
        item["corpus_contract_ref"],
        {"schema_version", "contract_id", "content_sha256"},
        "training data authority.corpus_contract_ref",
    )
    expected_ref = {
        "schema_version": corpus_contract.get("schema_version"),
        "contract_id": corpus_contract.get("contract_id"),
        "content_sha256": content_sha256(corpus_contract),
    }
    if ref != expected_ref or ref["schema_version"] != CORPUS_SCHEMA:
        raise NativeContractError("training data authority Corpus v3 binding is stale or substituted")
    bindings = item["unresolved_bindings"]
    if not isinstance(bindings, Mapping) or set(bindings) != set(BINDING_BLOCKERS):
        raise NativeContractError("training data authority unresolved binding closure is invalid")
    for name, blocker in BINDING_BLOCKERS.items():
        binding = _exact(
            bindings[name], {"state", "value", "blocker_tag"},
            f"training data authority.unresolved_bindings.{name}",
        )
        if binding != {"state": "unresolved", "value": None, "blocker_tag": blocker}:
            raise NativeContractError(f"training data authority binding {name} is not unresolved")
    blockers = item["blocker_tags"]
    if not isinstance(blockers, list) or blockers != sorted(BINDING_BLOCKERS.values()):
        raise NativeContractError("training data authority blockers do not match unresolved bindings")
    return deepcopy(dict(item))


def _verify_authority_trust_anchor(value: Mapping[str, Any]) -> None:
    if content_sha256(value) != TRUSTED_AUTHORITY_CONTENT_SHA256:
        raise NativeContractError(
            "training data authority differs from the code-pinned trust anchor"
        )


_CANONICAL_CORPUS_CONTRACT = _read_json(CORPUS_CONTRACT_PATH, "Corpus v3 contract")
_CANONICAL_AUTHORITY = _validate_authority(
    _read_json(AUTHORITY_PATH, "training data authority"), _CANONICAL_CORPUS_CONTRACT,
)
_verify_authority_trust_anchor(_CANONICAL_AUTHORITY)
TRAINING_DATA_AUTHORITY = deepcopy(_CANONICAL_AUTHORITY)


def load_training_data_authority(
    path: str | Path = AUTHORITY_PATH,
    *,
    corpus_contract_path: str | Path = CORPUS_CONTRACT_PATH,
) -> dict[str, Any]:
    """Load the exact packaged authority and its exact packaged Corpus v3 parent."""
    authority_path = resolve_training_data_authority_path((path,))
    contract_path = resolve_corpus_contract_path((corpus_contract_path,))
    value = _validate_authority(
        _read_json(authority_path, "training data authority"),
        _read_json(contract_path, "Corpus v3 contract"),
    )
    _verify_authority_trust_anchor(value)
    if value != _CANONICAL_AUTHORITY:
        raise NativeContractError("training data authority differs from canonical semantics")
    return value


def validate_training_data_authority(
    value: Any | None = None,
    *,
    corpus_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a value and reject self-issued status, hashes, or bindings."""
    candidate = _CANONICAL_AUTHORITY if value is None else value
    validated = _validate_authority(
        candidate, corpus_contract or _CANONICAL_CORPUS_CONTRACT,
    )
    _verify_authority_trust_anchor(validated)
    if validated != _CANONICAL_AUTHORITY:
        raise NativeContractError("training data authority differs from canonical semantics")
    return validated


def training_data_authority_sha256(value: Any | None = None) -> str:
    return content_sha256(validate_training_data_authority(value))


def resolve_training_data_authority(
    authority_id: Any, authority_sha256: Any,
) -> dict[str, Any]:
    """Resolve an instance reference to the canonical authority; convey no admission."""
    authority = load_training_data_authority()
    if authority_id != authority["authority_id"]:
        raise NativeContractError("route instance training data authority id is unknown")
    if authority_sha256 != training_data_authority_sha256(authority):
        raise NativeContractError("route instance training data authority hash is self-authored or stale")
    return authority
