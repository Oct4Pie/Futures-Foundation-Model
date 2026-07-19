from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

import pytest

from futures_foundation._authority_bundle_io import (
    AuthorityBundleIOError,
    VerifiedDirectoryReader,
    read_canonical_json_file,
)
from futures_foundation.session_denominator_bundle import (
    BLOCKERS,
    PRODUCER_COMPATIBILITY_COMMIT,
    SessionDenominatorBundleVerificationError,
    VerifiedSessionDenominatorBundleV2,
    iter_verified_session_shards,
    load_and_verify_session_denominator_bundle_v2,
)


ALPHAFORGE_COMMIT = PRODUCER_COMPATIBILITY_COMMIT


def _canonical(path: Path, value: object) -> None:
    path.write_bytes(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _producer_fixture(tmp_path: Path) -> dict[str, object]:
    alpha = Path(__file__).resolve().parents[2] / "alphaforge-corpus-v3-scale"
    python = alpha / ".venv" / "bin" / "python"
    tests = alpha / "tests" / "test_session_denominator.py"
    if not python.is_file() or not tests.is_file():
        pytest.skip("pinned AlphaForge compatibility worktree is unavailable")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=alpha,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert revision == ALPHAFORGE_COMMIT
    output = tmp_path / "producer-sidecar.json"
    script = r'''
import hashlib, json, runpy, sys
from pathlib import Path
namespace = runpy.run_path(sys.argv[1])
root = Path(sys.argv[2]).resolve()
root.mkdir(parents=True)
rules, consumer, scope, bundle = namespace["_built_session_bundle"](root)
verified = namespace["_load_session_bundle"](rules, consumer, scope, bundle)
manifest = bundle / "manifest.json"
result = {
    "rules_path": str(rules.source_path),
    "rules_sha256": rules.sha256,
    "consumer_path": str(consumer),
    "consumer_sha256": hashlib.sha256(consumer.read_bytes()).hexdigest(),
    "scope_path": str(scope),
    "scope_sha256": hashlib.sha256(scope.read_bytes()).hexdigest(),
    "bundle_path": str(bundle),
    "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    "shard_count": verified.shard_count,
    "row_count": verified.row_count,
}
Path(sys.argv[3]).write_text(json.dumps(result, sort_keys=True))
'''
    subprocess.run(
        [str(python), "-c", script, str(tests), str(tmp_path / "producer"), str(output)],
        cwd=alpha,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(output.read_text(encoding="utf-8"))


def _load(fixture: dict[str, object]):
    return load_and_verify_session_denominator_bundle_v2(
        str(fixture["bundle_path"]),
        expected_manifest_sha256=str(fixture["manifest_sha256"]),
        calendar_rules_path=str(fixture["rules_path"]),
        calendar_rules_sha256=str(fixture["rules_sha256"]),
        scope_v2_path=str(fixture["scope_path"]),
        scope_v2_sha256=str(fixture["scope_sha256"]),
        consumer_scope_path=str(fixture["consumer_path"]),
        consumer_scope_sha256=str(fixture["consumer_sha256"]),
    )


@pytest.fixture()
def producer_fixture(tmp_path):
    return _producer_fixture(tmp_path)


def test_ffm_independently_accepts_pinned_alphaforge_session_bundle(producer_fixture):
    verified = _load(producer_fixture)
    assert verified.production_admitted is False
    assert verified.shard_count == producer_fixture["shard_count"] == 6
    assert verified.row_count == producer_fixture["row_count"] == 6
    assert [
        (shard["partition_id"], shard["root"])
        for shard in iter_verified_session_shards(verified)
    ] == [
        (partition, root)
        for partition in ("shared_train", "development")
        for root in ("ES", "MCL", "ZC")
    ]


def test_bundle_cannot_escalate_production_status(producer_fixture):
    manifest_path = Path(str(producer_fixture["bundle_path"])) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    manifest["production_admission"] = True
    manifest.pop("manifest_semantic_sha256")
    manifest["manifest_semantic_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    _canonical(manifest_path, manifest)
    producer_fixture["manifest_sha256"] = _sha(manifest_path)
    with pytest.raises(SessionDenominatorBundleVerificationError, match="production-blocked"):
        _load(producer_fixture)


def test_admission_remains_blocked_without_optional_producer_fixture():
    assert PRODUCER_COMPATIBILITY_COMMIT == (
        "b84925763459c2f1a7f4300d11e9760867083629"
    )
    assert "production_admission_unavailable" in BLOCKERS
    forged = VerifiedSessionDenominatorBundleV2(
        Path("/untrusted/session-bundle"),
        "0" * 64,
        "0" * 64,
        Path("/untrusted/rules.json"),
        "0" * 64,
        Path("/untrusted/scope.json"),
        "0" * 64,
        Path("/untrusted/consumer.json"),
        "0" * 64,
        0,
        0,
        True,
        object(),
    )
    with pytest.raises(SessionDenominatorBundleVerificationError, match="capability"):
        list(iter_verified_session_shards(forged))


def test_authority_reader_rejects_fifo_socket_and_device_without_blocking(tmp_path):
    fifo = tmp_path / "fifo.json"
    os.mkfifo(fifo)
    unix_socket_path = tmp_path / "socket.json"
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(unix_socket_path))
    try:
        with VerifiedDirectoryReader(tmp_path, label="special-file fixture") as reader:
            for leaf in (fifo.name, unix_socket_path.name):
                with pytest.raises(AuthorityBundleIOError, match="regular file"):
                    reader.read_json(
                        leaf, label=leaf, max_bytes=1024, max_nodes=100,
                    )
        with pytest.raises(AuthorityBundleIOError, match="regular file"):
            read_canonical_json_file(
                Path("/dev/null"),
                label="character device",
                max_bytes=1024,
                max_nodes=100,
            )
    finally:
        listener.close()


def test_authority_reader_rejects_intermediate_ancestor_symlink(tmp_path):
    real = tmp_path / "real"
    bundle = real / "bundle"
    bundle.mkdir(parents=True)
    _canonical(bundle / "document.json", {"value": 1})
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(AuthorityBundleIOError, match="symlink"):
        read_canonical_json_file(
            alias / "bundle" / "document.json",
            label="ancestor-symlink document",
            max_bytes=1024,
            max_nodes=100,
        )


def test_bundle_rejects_orphan_and_noncanonical_path(producer_fixture, tmp_path):
    bundle = Path(str(producer_fixture["bundle_path"]))
    (bundle / "orphan.json").write_text("{}", encoding="ascii")
    with pytest.raises(SessionDenominatorBundleVerificationError, match="closure"):
        _load(producer_fixture)
    (bundle / "orphan.json").unlink()
    shim = bundle.parent / "shim"
    shim.mkdir()
    traversing = shim / ".." / bundle.name
    altered = dict(producer_fixture)
    altered["bundle_path"] = str(traversing)
    with pytest.raises(SessionDenominatorBundleVerificationError, match="canonical absolute"):
        _load(altered)


def test_bundle_and_shard_symlink_or_hardlink_are_rejected(producer_fixture):
    bundle = Path(str(producer_fixture["bundle_path"]))
    original_bundle = bundle.with_name("session-bundle-original")
    bundle.rename(original_bundle)
    bundle.symlink_to(original_bundle, target_is_directory=True)
    with pytest.raises(SessionDenominatorBundleVerificationError, match="symlink"):
        _load(producer_fixture)
    bundle.unlink()
    original_bundle.rename(bundle)

    manifest = json.loads((bundle / "manifest.json").read_text(encoding="ascii"))
    shard = bundle / manifest["shards"][0]["leaf"]
    original_shard = bundle.parent / f"{shard.name}.original"
    shard.rename(original_shard)
    os.link(original_shard, shard)
    with pytest.raises(SessionDenominatorBundleVerificationError, match="regular file"):
        _load(producer_fixture)


def test_locally_rehashed_semantic_row_tamper_is_rejected(producer_fixture):
    bundle = Path(str(producer_fixture["bundle_path"]))
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    entry = manifest["shards"][0]
    shard_path = bundle / entry["leaf"]
    shard = json.loads(shard_path.read_text(encoding="ascii"))
    shard["rows"][0]["status"] = "closed"
    shard["rows"][0]["segments_utc_ns"] = []
    shard["rows"][0]["segment_semantic_sha256"] = hashlib.sha256(b"[]").hexdigest()
    shard.pop("shard_semantic_sha256")
    shard["shard_semantic_sha256"] = hashlib.sha256(
        json.dumps(shard, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    _canonical(shard_path, shard)
    entry["physical_sha256"] = _sha(shard_path)
    entry["semantic_sha256"] = shard["shard_semantic_sha256"]
    manifest.pop("manifest_semantic_sha256")
    manifest["manifest_semantic_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    _canonical(manifest_path, manifest)
    producer_fixture["manifest_sha256"] = _sha(manifest_path)
    with pytest.raises(
        SessionDenominatorBundleVerificationError,
        match="calendar resolution|calendar observation|override",
    ):
        _load(producer_fixture)


def test_manifest_entry_rejects_boolean_row_count(producer_fixture):
    bundle = Path(str(producer_fixture["bundle_path"]))
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    assert manifest["shards"][0]["row_count"] == 1
    manifest["shards"][0]["row_count"] = True
    manifest.pop("manifest_semantic_sha256")
    manifest["manifest_semantic_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    _canonical(manifest_path, manifest)
    producer_fixture["manifest_sha256"] = _sha(manifest_path)
    with pytest.raises(SessionDenominatorBundleVerificationError, match="integer"):
        _load(producer_fixture)


def test_forged_capability_is_not_accepted(producer_fixture):
    verified = _load(producer_fixture)
    forged = VerifiedSessionDenominatorBundleV2(
        verified.bundle_path,
        verified.manifest_physical_sha256,
        verified.manifest_semantic_sha256,
        verified.calendar_rules_path,
        verified.calendar_rules_sha256,
        verified.scope_v2_path,
        verified.scope_v2_sha256,
        verified.consumer_scope_path,
        verified.consumer_scope_sha256,
        verified.shard_count,
        verified.row_count,
        False,
        object(),
    )
    with pytest.raises(SessionDenominatorBundleVerificationError, match="capability"):
        list(iter_verified_session_shards(forged))


def test_early_iterator_close_reopens_complete_bundle(producer_fixture):
    verified = _load(producer_fixture)
    stream = iter_verified_session_shards(verified)
    next(stream)
    bundle = Path(str(producer_fixture["bundle_path"]))
    (bundle / "orphan.json").write_text("{}", encoding="ascii")
    with pytest.raises(SessionDenominatorBundleVerificationError, match="closure"):
        stream.close()
