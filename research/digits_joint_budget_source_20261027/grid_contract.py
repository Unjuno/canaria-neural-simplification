"""Portable grid fresh-replication contract. Standard library only; no model imports.

No release is supplied by this source package. A release is an operator's
documented authorization, not a signature or an automatic approval mechanism.
The historical attempt is closed and must not be reopened using this contract.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import pathlib
import re
import time

CONTRACT = "canaria.grid-replication.v1"
EVIDENCE_CLASS = "new-grid-replication-on-seen-digits"
HISTORICAL_ATTEMPT = "CLOSED_NOT_REOPENED"
CUSTODY_ATTESTATION = "SINGLE_DURABLE_ROOT_NO_PRIOR_ATTEMPT_NO_RETRY"
REQUIRED_FILES = (
    "runner.py", "grid_contract.py", "parent_contract.py", "DEPENDENCY.json",
    "SEEDS.json", "PROTOCOL.md", "PARENT_SCHEMA.json", "RELEASE_SCHEMA.json",
    "ARTIFACTS.md", "ENVIRONMENT.md", "test_runner.py", "test_parent_contract.py",
)


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def manifest_hashes(package_root):
    root = pathlib.Path(package_root)
    return {name: sha(root / name) for name in REQUIRED_FILES}


def package_identity(package_root):
    value = json.dumps(manifest_hashes(package_root), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(value.encode()).hexdigest()


def replication_id(value):
    if not isinstance(value, str) or re.fullmatch(r"replication-[a-z0-9][a-z0-9-]{0,63}", value) is None:
        raise ValueError("a new, nonhistorical replication ID is required")
    return value


def deadline_timestamp(value):
    if not isinstance(value, str):
        raise ValueError("deadline_utc must be a UTC ISO-8601 string")
    try:
        parsed = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("invalid deadline_utc") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != datetime.timedelta(0):
        raise ValueError("deadline_utc must explicitly use UTC")
    return parsed.timestamp()


def validate_release(release, package_root, now_timestamp=None):
    required = {
        "contract", "replication_id", "evidence_class", "historical_attempt",
        "custody_attestation", "parent_approved", "independent_review", "reviewer",
        "approval_evidence", "package_sha256", "files", "deadline_utc",
        "parent_manifest_sha256", "parent_replication_id", "matched_package_sha256",
    }
    if not isinstance(release, dict) or set(release) != required:
        raise ValueError("release fields missing or unknown")
    for key, expected in (
        ("contract", CONTRACT), ("evidence_class", EVIDENCE_CLASS),
        ("historical_attempt", HISTORICAL_ATTEMPT),
        ("custody_attestation", CUSTODY_ATTESTATION),
        ("independent_review", "GO"),
    ):
        if release[key] != expected:
            raise ValueError("release contract mismatch: " + key)
    if release["parent_approved"] is not True:
        raise ValueError("operator approval and independent GO required")
    replication_id(release["replication_id"])
    replication_id(release["parent_replication_id"])
    for key in ("parent_manifest_sha256", "matched_package_sha256"):
        if not isinstance(release[key], str) or re.fullmatch(r"[0-9a-f]{64}", release[key]) is None:
            raise ValueError("invalid parent binding: " + key)
    for key in ("reviewer", "approval_evidence"):
        if not isinstance(release[key], str) or not release[key].strip():
            raise ValueError("private review provenance missing")
    if release["files"] != manifest_hashes(package_root):
        raise ValueError("release file mismatch")
    if release["package_sha256"] != package_identity(package_root):
        raise ValueError("release package mismatch")
    now_value = time.time() if now_timestamp is None else now_timestamp
    if now_value >= deadline_timestamp(release["deadline_utc"]):
        raise ValueError("replication deadline expired")
    return release


def context(workspace, identity, package_root):
    """Resolve fixed relative layout, without creating directories or attempts."""
    identity = replication_id(identity)
    raw = pathlib.Path(workspace)
    if not raw.is_dir() or raw.is_symlink():
        raise ValueError("existing nonsymlink durable custody workspace required")
    root = raw.resolve()
    package = pathlib.Path(package_root).resolve()
    registry = root / "custody" / "grid" / identity
    output = root / "artifacts" / identity
    for path in (registry, output):
        if path.is_relative_to(package):
            raise ValueError("custody and artifacts must be outside the source package")
        for part in (path, *path.parents):
            if part == root:
                break
            if part.is_symlink():
                raise ValueError("symlink in custody/artifact path")
        if not path.resolve().is_relative_to(root):
            raise ValueError("path escapes custody workspace")
    return registry, output


def public_release(release):
    """Scientific identity only; approval identities remain in private custody."""
    keys = (
        "contract", "replication_id", "evidence_class", "historical_attempt",
        "package_sha256", "files", "deadline_utc",
        "parent_manifest_sha256", "parent_replication_id", "matched_package_sha256",
    )
    return {key: release[key] for key in keys}
