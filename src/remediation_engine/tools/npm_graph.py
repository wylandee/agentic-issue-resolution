"""Neutral, deterministic npm manifest and lockfile graph primitives.

This module deliberately has no dependency on orchestration or worker code.  It
reads npm metadata from a repository and exposes immutable records that can be
consumed by synthetic-task materialisation and the portfolio solver.  Lockfile
package keys are physical paths (including nested ``node_modules`` copies), so
logical package names never erase occurrence identity.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import logging
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from semantic_version import NpmSpec, Version

from remediation_engine.runtime.path_policy import (
    WorkspacePathError,
    normalize_workspace_path,
    resolve_repository_path,
)

logger = logging.getLogger(__name__)

_MANIFEST_NAME = "package.json"
_LOCKFILE_NAMES = frozenset({"package-lock.json", "npm-shrinkwrap.json"})
_MANIFEST_SECTIONS = (
    "dependencies",
    "devDependencies",
    "optionalDependencies",
    "peerDependencies",
)
_LOCKFILE_DEPENDENCY_SECTIONS = (
    "dependencies",
    "optionalDependencies",
    "peerDependencies",
)
_EXACT_VERSION_RE = re.compile(r"^[=vV]?\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$")
_NPM_OPERATOR_SPACE_RE = re.compile(r"([<>=~^])\s+(?=[vV]?\d)")
_NODE_MODULES = "node_modules/"


def _freeze(value: Any) -> Any:
    """Recursively freeze JSON values for use in frozen records."""
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): _freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, tuple):
        return tuple(_freeze(v) for v in value)
    return value


def _thaw(value: Any) -> Any:
    """Convert frozen JSON values back to ordinary JSON-compatible values."""
    if isinstance(value, Mapping):
        return {str(k): _thaw(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_thaw(v) for v in value]
    return value


def _canonical_json(value: Any) -> str:
    """Serialize JSON-compatible data with stable ordering and separators."""
    return json.dumps(_thaw(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(value: Any) -> str:
    """Return a stable SHA-256 digest for JSON-compatible data."""
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def safe_repository_path(repo_root: str | Path, value: str | Path) -> str | None:
    """Return a safe repository-relative path, or ``None`` for unsafe input.

    The helper is intentionally non-throwing at ingestion boundaries.  It also
    rejects symlink escapes by resolving the candidate through the shared path
    policy.
    """
    try:
        root = Path(repo_root).resolve()
        raw_value = str(value).strip().replace("\\", "/")
        if not raw_value:
            return None
        raw_parts = Path(raw_value).parts
        if ".." in raw_parts:
            return None
        candidate = Path(raw_value)
        if candidate.is_absolute():
            relative = candidate.resolve().relative_to(root).as_posix()
        else:
            relative = raw_value
        return resolve_repository_path(root, relative).relative_to(root).as_posix()
    except (OSError, ValueError, WorkspacePathError):
        return None


def _package_name_from_lockfile_key(package_key: str) -> str | None:
    """Extract an npm package name from a physical lockfile package key."""
    normalized = package_key.replace("\\", "/").strip("/")
    if not normalized or _NODE_MODULES not in normalized:
        return None
    package_path = normalized.rsplit(_NODE_MODULES, 1)[-1].strip("/")
    if not package_path:
        return None
    parts = package_path.split("/")
    if parts[0].startswith("@"):
        return "/".join(parts[:2]) if len(parts) >= 2 and parts[1] else None
    return parts[0]


lockfile_package_name = _package_name_from_lockfile_key


def lockfile_key_matches_package(package_key: str, package_name: str) -> bool:
    """Return whether a physical lockfile key identifies the named package."""
    normalized = str(package_key or "").replace("\\", "/").strip("/")
    if not normalized or any(part in {"", ".", ".."} for part in normalized.split("/")):
        return False
    return normalized == package_name or _package_name_from_lockfile_key(normalized) == package_name


def _package_key_depth(package_key: str) -> int:
    """Return the physical nesting depth of a lockfile package key."""
    return package_key.replace("\\", "/").count(_NODE_MODULES)


def _ancestry_from_package_key(package_key: str) -> tuple[str, ...]:
    """Return ordered package names represented by nested ``node_modules`` keys."""
    normalized = package_key.replace("\\", "/")
    chunks = [chunk for chunk in normalized.split(_NODE_MODULES) if chunk]
    names: list[str] = []
    for chunk in chunks:
        parts = chunk.split("/")
        if parts[0].startswith("@") and len(parts) >= 2:
            names.append("/".join(parts[:2]))
        else:
            names.append(parts[0])
    return tuple(names)


@dataclass(frozen=True)
class NpmRangeCheck:
    """Result of checking one npm version against a range.

    ``matches`` is ``None`` when either input cannot be parsed, allowing
    callers to distinguish an invalid range from a normal mismatch.
    """

    matches: bool | None
    diagnostic: str | None = None


def check_npm_range(version_range: str | None, version: str | None) -> NpmRangeCheck:
    """Check an npm range and return an explicit diagnostic for invalid input."""
    requested = str(version_range or "").strip()
    requested = _NPM_OPERATOR_SPACE_RE.sub(r"\1", requested)
    candidate = str(version or "").strip().lstrip("vV")
    if not requested or requested in {"*", "latest"}:
        return NpmRangeCheck(True)
    if not candidate:
        return NpmRangeCheck(None, "npm version is empty")
    try:
        return NpmRangeCheck(NpmSpec(requested).match(Version.coerce(candidate)))
    except (TypeError, ValueError) as exc:
        return NpmRangeCheck(None, f"invalid npm range/version {requested!r}/{candidate!r}: {exc}")


def npm_range_contains(version_range: str | None, version: str | None) -> bool | None:
    """Return whether ``version`` satisfies an npm range, or ``None`` if invalid."""
    return check_npm_range(version_range, version).matches


# Kept as a local compatibility spelling for graph consumers, not orchestration.
_npm_range_contains = npm_range_contains


def normalize_dependency_ancestry(ancestry: str | Iterable[str] | None) -> tuple[str, ...]:
    """Normalize scanner or lockfile ancestry into ordered npm package names.

    Strings may use slash-separated scanner notation with optional ``:version``
    suffixes.  Scoped names are kept intact; blank and malformed tokens are
    discarded deterministically.
    """
    if ancestry is None:
        return ()
    if isinstance(ancestry, str):
        raw_tokens = _split_ancestry_tokens(ancestry)
    else:
        raw_tokens = [str(item).strip() for item in ancestry if str(item).strip()]
    result: list[str] = []
    for token in raw_tokens:
        token = token.strip().strip("/")
        if not token:
            continue
        if token.startswith("@"):
            colon = token.rfind(":")
            slash = token.find("/")
            if colon > slash:
                token = token[:colon]
        elif ":" in token:
            token = token.split(":", 1)[0]
        token = token.strip()
        if token and (not token.startswith("@") or "/" in token):
            result.append(token)
    return tuple(result)


def _split_ancestry_tokens(value: str) -> list[str]:
    """Split ancestry without breaking scoped package names."""
    remaining = value.strip().lstrip("/")
    result: list[str] = []
    while remaining:
        if remaining.startswith("@"):
            slash = remaining.find("/")
            if slash < 0:
                result.append(remaining)
                break
            next_slash = remaining.find("/", slash + 1)
            if next_slash < 0:
                result.append(remaining)
                break
            result.append(remaining[:next_slash])
            remaining = remaining[next_slash + 1 :]
            continue
        slash = remaining.find("/")
        if slash < 0:
            result.append(remaining)
            break
        result.append(remaining[:slash])
        remaining = remaining[slash + 1 :]
    return result


@dataclass(frozen=True)
class NpmManifest:
    """One validated repository-relative npm manifest."""

    path: str
    data: Mapping[str, Any]
    package_name: str | None = None
    workspace_patterns: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "data", _freeze(dict(self.data)))
        if self.package_name is None:
            name = self.data.get("name")
            object.__setattr__(self, "package_name", str(name).strip() if name else None)
        object.__setattr__(self, "workspace_patterns", tuple(sorted(set(self.workspace_patterns))))


@dataclass(frozen=True)
class NpmLockfilePackage:
    """One exact physical package entry, including nested lockfile identity."""

    lockfile_path: str
    manifest_path: str
    package_key: str
    package_name: str
    metadata: Mapping[str, Any]
    version: str | None = None
    depth: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _freeze(dict(self.metadata)))
        if self.version is None:
            value = self.metadata.get("version")
            object.__setattr__(self, "version", str(value).strip() if value else None)
        object.__setattr__(self, "depth", _package_key_depth(self.package_key))

    @property
    def ancestry(self) -> tuple[str, ...]:
        """Return package ancestry encoded in the physical key."""
        return _ancestry_from_package_key(self.package_key)


@dataclass(frozen=True)
class NpmLockfile:
    """One decoded npm lockfile and its physical package entries."""

    path: str
    version: int
    packages: tuple[NpmLockfilePackage, ...]
    raw_digest: str


@dataclass(frozen=True)
class NpmDependencyRecord:
    """One direct dependency declaration and shortest matching lock metadata."""

    manifest_path: str
    package_name: str
    declaration_type: str
    requested_spec: str
    resolved_version: str | None
    lockfile_package_key: str | None = None
    lockfile_path: str | None = None

    @property
    def key(self) -> tuple[str, str]:
        """Return logical manifest occurrence identity."""
        return self.manifest_path, self.package_name

    @property
    def occurrence_id(self) -> str:
        """Return stable direct occurrence identity."""
        return make_occurrence_id(self.manifest_path, self.package_name)


# Private spelling used by the old helper during migration.
_DependencyRecord = NpmDependencyRecord
DependencyRecord = NpmDependencyRecord


@dataclass(frozen=True)
class NpmWorkspaceMembership:
    """Membership of a manifest in a declaring workspace root."""

    workspace_root: str
    member_manifest: str

    @property
    def key(self) -> tuple[str, str]:
        """Return deterministic membership identity."""
        return self.workspace_root, self.member_manifest


def make_occurrence_id(
    manifest_path: str,
    package_name: str,
    lockfile_package_key: str | None = None,
) -> str:
    """Build a stable occurrence ID without collapsing separate manifests."""
    normalized_manifest = manifest_path.replace("\\", "/")
    normalized_package = package_name.strip()
    base = f"{normalized_manifest}::{normalized_package}"
    if lockfile_package_key:
        normalized_key = lockfile_package_key.replace("\\", "/")
        return f"{base}::{normalized_key}"
    return base


@dataclass(frozen=True)
class NpmOccurrence:
    """Occurrence-aware package identity used by graph extraction."""

    occurrence_id: str
    manifest_path: str
    package_name: str
    lockfile_package_key: str | None = None
    installed_version: str | None = None
    dependency_type: str | None = None
    is_direct: bool = True
    ancestry: tuple[str, ...] = ()


@dataclass(frozen=True)
class NpmGraphSnapshot:
    """Immutable npm graph input and deterministic repository fingerprint."""

    manifests: tuple[NpmManifest, ...] = ()
    lockfiles: tuple[NpmLockfile, ...] = ()
    dependency_records: tuple[NpmDependencyRecord, ...] = ()
    occurrences: tuple[NpmOccurrence, ...] = ()
    workspace_memberships: tuple[NpmWorkspaceMembership, ...] = ()
    fingerprint_inputs: tuple[tuple[str, str], ...] = ()
    repository_fingerprint: str = ""
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "manifests", tuple(sorted(self.manifests, key=lambda x: x.path)))
        object.__setattr__(self, "lockfiles", tuple(sorted(self.lockfiles, key=lambda x: x.path)))
        object.__setattr__(
            self,
            "dependency_records",
            tuple(sorted(self.dependency_records, key=lambda x: (x.manifest_path, x.package_name))),
        )
        object.__setattr__(
            self, "occurrences", tuple(sorted(self.occurrences, key=lambda x: x.occurrence_id))
        )
        object.__setattr__(
            self,
            "workspace_memberships",
            tuple(sorted(self.workspace_memberships, key=lambda x: x.key)),
        )
        object.__setattr__(self, "fingerprint_inputs", tuple(sorted(self.fingerprint_inputs)))
        object.__setattr__(self, "diagnostics", tuple(sorted(set(self.diagnostics))))
        if not self.repository_fingerprint:
            object.__setattr__(self, "repository_fingerprint", _digest(self.fingerprint_inputs))

    @property
    def manifests_by_path(self) -> dict[str, NpmManifest]:
        """Return a fresh path index of manifests."""
        return {manifest.path: manifest for manifest in self.manifests}

    @property
    def lockfile_packages(self) -> tuple[NpmLockfilePackage, ...]:
        """Return all physical lockfile package entries in stable order."""
        return tuple(package for lockfile in self.lockfiles for package in lockfile.packages)

    @property
    def workspace_membership_map(self) -> dict[str, set[str]]:
        """Return workspace roots keyed by member manifest path."""
        memberships: dict[str, set[str]] = {}
        for item in self.workspace_memberships:
            memberships.setdefault(item.member_manifest, set()).add(item.workspace_root)
        return memberships

    @property
    def digest(self) -> str:
        """Alias for the repository fingerprint used by solver inputs."""
        return self.repository_fingerprint

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible representation."""
        return {
            "manifests": [
                {
                    "path": m.path,
                    "data": _thaw(m.data),
                    "package_name": m.package_name,
                    "workspace_patterns": list(m.workspace_patterns),
                }
                for m in self.manifests
            ],
            "lockfiles": [
                {
                    "path": lock.path,
                    "version": lock.version,
                    "raw_digest": lock.raw_digest,
                    "packages": [
                        {
                            "lockfile_path": p.lockfile_path,
                            "manifest_path": p.manifest_path,
                            "package_key": p.package_key,
                            "package_name": p.package_name,
                            "metadata": _thaw(p.metadata),
                            "version": p.version,
                            "depth": p.depth,
                        }
                        for p in lock.packages
                    ],
                }
                for lock in self.lockfiles
            ],
            "dependency_records": [record.__dict__ for record in self.dependency_records],
            "occurrences": [occurrence.__dict__ for occurrence in self.occurrences],
            "workspace_memberships": [item.__dict__ for item in self.workspace_memberships],
            "fingerprint_inputs": [list(item) for item in self.fingerprint_inputs],
            "repository_fingerprint": self.repository_fingerprint,
            "diagnostics": list(self.diagnostics),
        }

    def serialize(self) -> str:
        """Serialize the snapshot deterministically as JSON."""
        return _canonical_json(self.to_dict())


# ---------------------------------------------------------------------------
# Loading and extraction
# ---------------------------------------------------------------------------


def _workspace_patterns(data: Mapping[str, Any]) -> tuple[str, ...]:
    """Read npm workspace patterns from list or object syntax."""
    value = data.get("workspaces")
    if isinstance(value, list):
        return tuple(sorted({str(item).strip() for item in value if str(item).strip()}))
    if isinstance(value, Mapping) and isinstance(value.get("packages"), list):
        return tuple(sorted({str(item).strip() for item in value["packages"] if str(item).strip()}))
    return ()


def _discover_files(root: Path, names: set[str], diagnostics: list[str]) -> list[tuple[str, Path]]:
    """Discover safe metadata files, skipping ignored directories and escapes."""
    found: list[tuple[str, Path]] = []
    try:
        candidates = root.rglob("*")
    except OSError as exc:
        diagnostics.append(f"cannot scan repository: {exc}")
        return found
    for path in candidates:
        if path.name not in names or not path.is_file():
            continue
        try:
            relative = path.relative_to(root).as_posix()
            if any(
                part in {".git", "node_modules", ".remedy-attempt-snapshots"}
                for part in path.relative_to(root).parts
            ):
                continue
            safe = safe_repository_path(root, relative)
            if safe is None:
                diagnostics.append(f"unsafe metadata path skipped: {relative!r}")
                continue
            found.append((safe, path))
        except (OSError, ValueError):
            diagnostics.append(f"unsafe metadata path skipped: {path}")
    return sorted(found, key=lambda item: item[0])


def _read_json(path: Path, relative: str, diagnostics: list[str]) -> tuple[Any, str] | None:
    """Read JSON and return payload plus raw fingerprint material."""
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        diagnostics.append(f"could not read {relative}: {exc}")
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        diagnostics.append(f"invalid JSON in {relative}: {exc.msg}")
        return None
    return payload, raw


def _parse_v1_packages(
    lockfile_path: str,
    manifest_path: str,
    dependencies: Mapping[str, Any],
    *,
    parent_key: str = "",
) -> list[NpmLockfilePackage]:
    """Flatten package-lock v1 nested dependency objects into physical keys."""
    packages: list[NpmLockfilePackage] = []
    for raw_name in sorted(dependencies, key=lambda item: str(item)):
        name = str(raw_name).strip()
        metadata = dependencies[raw_name]
        if not isinstance(metadata, Mapping) or not name:
            continue
        package_key = (
            f"{parent_key}{_NODE_MODULES}{name}" if parent_key else f"{_NODE_MODULES}{name}"
        )
        packages.append(
            NpmLockfilePackage(
                lockfile_path=lockfile_path,
                manifest_path=manifest_path,
                package_key=package_key,
                package_name=name.strip(),
                metadata=dict(metadata),
            )
        )
        nested = metadata.get("dependencies")
        if isinstance(nested, Mapping):
            packages.extend(
                _parse_v1_packages(lockfile_path, manifest_path, nested, parent_key=package_key)
            )
    return packages


def _parse_lockfile(
    relative: str,
    path: Path,
    payload: Mapping[str, Any],
    raw: str,
    diagnostics: list[str],
) -> NpmLockfile:
    """Parse one package-lock v1/v2/v3 payload retaining physical keys."""
    version_raw = payload.get("lockfileVersion", 1)
    try:
        version = int(version_raw)
    except (TypeError, ValueError):
        version = 0
    if version not in {1, 2, 3}:
        diagnostics.append(f"unsupported npm lockfile version {version!r}: {relative}")
    manifest_path = f"{Path(relative).parent.as_posix()}/{_MANIFEST_NAME}"
    if manifest_path == "./package.json":
        manifest_path = _MANIFEST_NAME
    packages: list[NpmLockfilePackage] = []
    package_map = payload.get("packages")
    if isinstance(package_map, Mapping):
        for raw_key in sorted(str(key) for key in package_map):
            metadata = package_map.get(raw_key)
            if not isinstance(metadata, Mapping):
                diagnostics.append(f"invalid package metadata skipped: {relative}#{raw_key}")
                continue
            if not raw_key:
                continue
            package_key = raw_key.replace("\\", "/")
            package_name = _package_name_from_lockfile_key(package_key)
            if package_name is None:
                diagnostics.append(
                    f"non-physical lockfile package key retained as diagnostic: {relative}#{raw_key}"
                )
                continue
            packages.append(
                NpmLockfilePackage(
                    lockfile_path=relative,
                    manifest_path=manifest_path,
                    package_key=package_key,
                    package_name=package_name,
                    metadata=dict(metadata),
                )
            )
    elif isinstance(payload.get("dependencies"), Mapping):
        packages.extend(
            _parse_v1_packages(
                relative,
                manifest_path,
                payload["dependencies"],
            )
        )
    else:
        diagnostics.append(f"lockfile has no packages/dependencies object: {relative}")
    return NpmLockfile(
        path=relative,
        version=version,
        packages=tuple(sorted(packages, key=lambda item: item.package_key)),
        raw_digest=_digest(raw),
    )


def _direct_dependency_records(
    manifests: Sequence[NpmManifest],
    lockfiles: Sequence[NpmLockfile],
) -> tuple[NpmDependencyRecord, ...]:
    """Extract declarations and choose shortest matching lock metadata.

    Ties are resolved by physical key and lockfile path, ensuring nested copies
    can never accidentally replace a direct package entry.
    """
    candidates: dict[tuple[str, str], list[NpmLockfilePackage]] = {}
    for lockfile in lockfiles:
        for package in lockfile.packages:
            candidates.setdefault((package.manifest_path, package.package_name), []).append(package)
    for values in candidates.values():
        values.sort(key=lambda item: (item.depth, item.package_key, item.lockfile_path))
    records: list[NpmDependencyRecord] = []
    for manifest in manifests:
        for section in _MANIFEST_SECTIONS:
            values = manifest.data.get(section)
            if not isinstance(values, Mapping):
                continue
            for raw_name in sorted(values, key=lambda item: str(item)):
                package_name = str(raw_name).strip()
                if not package_name or any(
                    item.package_name == package_name
                    for item in records
                    if item.manifest_path == manifest.path
                ):
                    continue
                requested = str(values[raw_name]).strip() if values[raw_name] is not None else ""
                metadata = candidates.get((manifest.path, package_name), [])
                selected = metadata[0] if metadata else None
                resolved = selected.version if selected and selected.version else None
                if not resolved and _EXACT_VERSION_RE.fullmatch(requested):
                    resolved = requested.lstrip("=vV")
                records.append(
                    NpmDependencyRecord(
                        manifest_path=manifest.path,
                        package_name=package_name,
                        declaration_type=section,
                        requested_spec=requested,
                        resolved_version=resolved,
                        lockfile_package_key=selected.package_key if selected else None,
                        lockfile_path=selected.lockfile_path if selected else None,
                    )
                )
    return tuple(sorted(records, key=lambda item: (item.manifest_path, item.package_name)))


def _workspace_memberships_for_manifests(
    manifests: Sequence[NpmManifest],
) -> tuple[NpmWorkspaceMembership, ...]:
    """Expand workspace declarations to explicit root/member pairs."""
    by_path = {manifest.path: manifest for manifest in manifests}
    result: set[tuple[str, str]] = set()
    for root_manifest in manifests:
        if not root_manifest.workspace_patterns:
            continue
        root_dir = Path(root_manifest.path).parent.as_posix()
        result.add((root_manifest.path, root_manifest.path))
        for member_path in by_path:
            if member_path == root_manifest.path:
                continue
            member_dir = Path(member_path).parent.as_posix()
            try:
                relative_dir = Path(member_dir).relative_to(Path(root_dir)).as_posix()
            except ValueError:
                continue
            for pattern in root_manifest.workspace_patterns:
                clean_pattern = pattern.rstrip("/")
                if fnmatch.fnmatchcase(relative_dir, clean_pattern) or fnmatch.fnmatchcase(
                    relative_dir + "/package.json", clean_pattern
                ):
                    result.add((root_manifest.path, member_path))
                    break
    return tuple(NpmWorkspaceMembership(root, member) for root, member in sorted(result))


def _build_occurrences(
    records: Sequence[NpmDependencyRecord],
    manifests: Sequence[NpmManifest] = (),
    lockfiles: Sequence[NpmLockfile] = (),
) -> tuple[NpmOccurrence, ...]:
    """Build direct manifest and exact lockfile occurrence identities."""
    direct_lockfile_keys = {
        (record.manifest_path, record.lockfile_package_key)
        for record in records
        if record.lockfile_package_key
    }
    by_id: dict[str, NpmOccurrence] = {
        record.occurrence_id: NpmOccurrence(
            occurrence_id=record.occurrence_id,
            manifest_path=record.manifest_path,
            package_name=record.package_name,
            lockfile_package_key=record.lockfile_package_key,
            installed_version=record.resolved_version,
            dependency_type=record.declaration_type,
            is_direct=True,
            ancestry=normalize_dependency_ancestry(
                _ancestry_from_package_key(record.lockfile_package_key or "")
            ),
        )
        for record in records
    }
    for manifest in manifests:
        package_name = (manifest.package_name or "").strip()
        if not package_name:
            continue
        occurrence_id = make_occurrence_id(manifest.path, package_name)
        if occurrence_id in by_id:
            continue
        raw_version = manifest.data.get("version")
        version = str(raw_version).strip() if raw_version else None
        by_id[occurrence_id] = NpmOccurrence(
            occurrence_id=occurrence_id,
            manifest_path=manifest.path,
            package_name=package_name,
            lockfile_package_key=f"node_modules/{package_name}",
            installed_version=version,
            dependency_type="workspace",
            is_direct=False,
        )
    for lockfile in lockfiles:
        for package in lockfile.packages:
            if (package.manifest_path, package.package_key) in direct_lockfile_keys:
                continue
            occurrence_id = make_occurrence_id(
                package.manifest_path,
                package.package_name,
                package.package_key,
            )
            if occurrence_id in by_id:
                continue
            by_id[occurrence_id] = NpmOccurrence(
                occurrence_id=occurrence_id,
                manifest_path=package.manifest_path,
                package_name=package.package_name,
                lockfile_package_key=package.package_key,
                installed_version=package.version,
                dependency_type=None,
                is_direct=False,
                ancestry=normalize_dependency_ancestry(package.ancestry),
            )
    return tuple(sorted(by_id.values(), key=lambda item: item.occurrence_id))


def repository_fingerprint_inputs(repo_root: str | Path) -> tuple[tuple[str, str], ...]:
    """Return canonical metadata inputs used for repository fingerprinting."""
    diagnostics: list[str] = []
    inputs: list[tuple[str, str]] = []
    for relative, path in _discover_files(
        Path(repo_root).resolve(), {_MANIFEST_NAME, *_LOCKFILE_NAMES}, diagnostics
    ):
        loaded = _read_json(path, relative, diagnostics)
        if loaded is None:
            try:
                raw = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            inputs.append((relative, raw))
            continue
        payload, _raw = loaded
        inputs.append((relative, _canonical_json(payload)))
    return tuple(sorted(inputs))


def _load_npm_graph_snapshot_from_documents(
    documents: Mapping[str, str],
    initial_diagnostics: Sequence[str] = (),
) -> NpmGraphSnapshot:
    """Parse safe npm metadata documents through the shared graph builder."""
    diagnostics = list(initial_diagnostics)
    normalized_documents: dict[str, str] = {}
    conflicting_paths: set[str] = set()
    for raw_path, raw_content in documents.items():
        if not isinstance(raw_path, str):
            diagnostics.append("npm metadata document path must be a string")
            continue
        try:
            relative = normalize_workspace_path(raw_path, allow_workspace_prefix=False)
        except WorkspacePathError as exc:
            diagnostics.append(f"unsafe npm metadata document path {raw_path!r}: {exc}")
            continue
        if any(
            part in {".git", "node_modules", ".remedy-attempt-snapshots", ".remedy-plan-cert"}
            for part in Path(relative).parts
        ):
            diagnostics.append(f"ignored npm metadata document path: {relative}")
            continue
        if Path(relative).name not in {_MANIFEST_NAME, *_LOCKFILE_NAMES}:
            diagnostics.append(f"unsupported npm metadata document path: {relative}")
            continue
        if not isinstance(raw_content, str):
            diagnostics.append(f"npm metadata document must contain text: {relative}")
            continue
        previous = normalized_documents.get(relative)
        if previous is not None and previous != raw_content:
            conflicting_paths.add(relative)
            diagnostics.append(f"conflicting npm metadata documents normalize to {relative!r}")
            continue
        normalized_documents[relative] = raw_content
    for relative in conflicting_paths:
        normalized_documents.pop(relative, None)

    manifests: list[NpmManifest] = []
    lockfiles: list[NpmLockfile] = []
    fingerprint_inputs: list[tuple[str, str]] = []
    for relative, raw in sorted(normalized_documents.items()):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            diagnostics.append(f"invalid JSON in {relative}: {exc.msg}")
            continue
        if Path(relative).name == _MANIFEST_NAME:
            if not isinstance(payload, Mapping):
                diagnostics.append(f"manifest is not an object: {relative}")
                continue
            fingerprint_inputs.append((relative, _canonical_json(payload)))
            manifests.append(
                NpmManifest(
                    path=relative,
                    data=dict(payload),
                    workspace_patterns=_workspace_patterns(payload),
                )
            )
        else:
            if not isinstance(payload, Mapping):
                diagnostics.append(f"lockfile is not an object: {relative}")
                continue
            fingerprint_inputs.append((relative, _canonical_json(payload)))
            lockfiles.append(_parse_lockfile(relative, Path(relative), payload, raw, diagnostics))

    manifests_tuple = tuple(sorted(manifests, key=lambda item: item.path))
    lockfiles_tuple = tuple(sorted(lockfiles, key=lambda item: item.path))
    records = _direct_dependency_records(manifests_tuple, lockfiles_tuple)
    memberships = _workspace_memberships_for_manifests(manifests_tuple)
    occurrences = _build_occurrences(records, manifests_tuple, lockfiles_tuple)
    canonical_inputs = tuple(sorted(fingerprint_inputs))
    return NpmGraphSnapshot(
        manifests=manifests_tuple,
        lockfiles=lockfiles_tuple,
        dependency_records=records,
        occurrences=occurrences,
        workspace_memberships=memberships,
        fingerprint_inputs=canonical_inputs,
        repository_fingerprint=_digest(canonical_inputs),
        diagnostics=tuple(diagnostics),
    )


def load_npm_graph_snapshot_from_documents(
    documents: Mapping[str, str],
) -> NpmGraphSnapshot:
    """Build an npm graph from safe relative manifest and lockfile documents."""
    if not isinstance(documents, Mapping):
        raise TypeError("documents must be a mapping of relative paths to text")
    return _load_npm_graph_snapshot_from_documents(documents)


def load_npm_graph_snapshot(repo_root: str | Path) -> NpmGraphSnapshot:
    """Load npm graph files and delegate parsing to the document-based loader.

    No network, subprocess, or repository mutation occurs. Repeated loads over
    unchanged files produce byte-identical :meth:`NpmGraphSnapshot.serialize`
    output.
    """
    root = Path(repo_root).resolve()
    diagnostics: list[str] = []
    if not root.exists() or not root.is_dir():
        diagnostics.append(f"repository root is not a directory: {repo_root!r}")
        return _load_npm_graph_snapshot_from_documents({}, diagnostics)
    documents: dict[str, str] = {}
    for relative, path in _discover_files(root, {_MANIFEST_NAME, *_LOCKFILE_NAMES}, diagnostics):
        try:
            documents[relative] = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            diagnostics.append(f"could not read {relative}: {exc}")
    return _load_npm_graph_snapshot_from_documents(documents, diagnostics)


def resolve_lockfile_dependency_package(
    snapshot: NpmGraphSnapshot,
    source_occurrence_id: str,
    package_name: str,
) -> NpmLockfilePackage | None:
    """Resolve the nearest physical npm dependency visible from one occurrence."""
    source = next(
        (item for item in snapshot.occurrences if item.occurrence_id == source_occurrence_id),
        None,
    )
    if source is None or not source.lockfile_package_key or not package_name.strip():
        return None
    source_key = source.lockfile_package_key.replace("\\", "/").strip("/")
    prefixes: list[str] = []
    current = source_key
    while current:
        prefixes.append(current)
        marker = current.rfind("/node_modules/")
        current = current[:marker] if marker >= 0 else ""
    prefixes.append("")
    packages_by_key: dict[tuple[str, str], list[NpmLockfilePackage]] = {}
    for package in snapshot.lockfile_packages:
        packages_by_key.setdefault(
            (package.manifest_path, package.package_key.replace("\\", "/")), []
        ).append(package)
    for prefix in prefixes:
        child_key = (
            f"{prefix}/node_modules/{package_name}" if prefix else f"node_modules/{package_name}"
        )
        matches = [
            item
            for item in packages_by_key.get((source.manifest_path, child_key), ())
            if item.package_name == package_name
        ]
        if matches:
            return min(matches, key=lambda item: (item.lockfile_path, item.package_key))
    return None


# Practical aliases for callers migrating from private portfolio helpers.
load_graph_snapshot = load_npm_graph_snapshot


def load_npm_manifests(repo_root: str | Path) -> tuple[NpmManifest, ...]:
    """Return the deterministic npm manifest projection for a repository."""
    return load_npm_graph_snapshot(repo_root).manifests


def load_lockfile_packages(repo_root: str | Path) -> tuple[NpmLockfilePackage, ...]:
    """Return the deterministic lockfile package projection for a repository."""
    return load_npm_graph_snapshot(repo_root).lockfile_packages


__all__ = [
    "DependencyRecord",
    "NpmDependencyRecord",
    "NpmGraphSnapshot",
    "NpmLockfile",
    "NpmLockfilePackage",
    "NpmManifest",
    "NpmOccurrence",
    "NpmRangeCheck",
    "NpmWorkspaceMembership",
    "check_npm_range",
    "load_graph_snapshot",
    "load_lockfile_packages",
    "load_npm_graph_snapshot",
    "load_npm_graph_snapshot_from_documents",
    "load_npm_manifests",
    "resolve_lockfile_dependency_package",
    "lockfile_package_name",
    "make_occurrence_id",
    "normalize_dependency_ancestry",
    "npm_range_contains",
    "repository_fingerprint_inputs",
    "safe_repository_path",
]
