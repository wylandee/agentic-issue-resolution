"""Supervisor registry access, release evidence, and bounded packument caching."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
import time
from collections.abc import Callable, Mapping
from functools import cmp_to_key
from pathlib import Path
from typing import Any, Final, TypeAlias
from urllib.parse import quote
from xml.etree import ElementTree as ET

import requests
from langchain_core.tools import tool
from langsmith import traceable
from packaging.requirements import InvalidRequirement, Requirement
from packaging.version import InvalidVersion, Version
from requests import RequestException
from semantic_version import NpmSpec
from semantic_version import Version as NpmVersion

from remediation_engine.contracts.schemas import SCARemediationStage
from remediation_engine.contracts.version_policy import (
    MavenRegistryCandidate,
    RegistryCandidate,
    compare_maven_versions,
    is_stable_maven_version,
    select_version,
)
from remediation_engine.tools.identity import normalize_python_package_name
from remediation_engine.tools.npm import _normalize_npm_operator_whitespace

Packument: TypeAlias = dict[str, Any]
PackumentFetcher: TypeAlias = Callable[[str], Mapping[str, Any]]

CACHE_SCHEMA_VERSION: Final[int] = 1
DEFAULT_MAX_PAYLOAD_BYTES: Final[int] = 5 * 1024 * 1024
DEFAULT_DIAGNOSTICS_LIMIT: Final[int] = 32
_MAX_DIAGNOSTIC_LENGTH: Final[int] = 512
_PACKAGE_NAME_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^(?:@[^/@\\\s]+/[^/@\\\s]+|[^@/\\\s]+)$"
)


class RegistryCacheError(ValueError):
    """Raised when a fetched packument cannot be validated for caching."""


def validate_package_name(package_name: str) -> str:
    """Validate and return an npm package name.

    The validator intentionally checks the identity and path-safety properties
    needed by the cache.  It accepts ordinary and scoped npm names, but rejects
    path separators, control/whitespace characters, empty segments, and dot
    traversal names.  The input is not lower-cased: the package identity stored
    by npm is compared exactly on cache reads.

    Args:
        package_name: Package name supplied by a registry caller.

    Returns:
        The stripped package name.

    Raises:
        ValueError: If ``package_name`` is not a valid cache package identity.
    """
    if not isinstance(package_name, str):
        raise ValueError("package_name must be a string")
    value = package_name.strip()
    if not value or value in {".", ".."} or not _PACKAGE_NAME_PATTERN.fullmatch(value):
        raise ValueError(f"Invalid npm package name: {package_name!r}")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError(f"Invalid npm package name: {package_name!r}")
    segments = value.split("/")
    if any(segment in {".", ".."} for segment in segments):
        raise ValueError(f"Invalid npm package name: {package_name!r}")
    return value


def validate_packument(package_name: str, packument: Mapping[str, Any]) -> Packument:
    """Validate and copy a raw npm packument.

    A packument must be a mapping with a mapping-valued ``versions`` member.
    When npm supplies a ``name`` member it must match the requested package;
    accepting a missing name keeps injected test/offline fetchers compatible
    with minimal packument fixtures while still enforcing stored identity in the
    cache envelope.

    Args:
        package_name: Expected npm package name.
        packument: Raw decoded registry JSON.

    Returns:
        A detached, mutable ``dict`` suitable for deterministic serialization.

    Raises:
        ValueError: If the package identity or packument shape is invalid.
    """
    expected_name = validate_package_name(package_name)
    if not isinstance(packument, Mapping):
        raise RegistryCacheError("registry packument must be a JSON object")
    supplied_name = packument.get("name")
    if supplied_name is not None and supplied_name != expected_name:
        raise RegistryCacheError(
            f"registry packument name {supplied_name!r} does not match {expected_name!r}"
        )
    versions = packument.get("versions")
    if not isinstance(versions, Mapping):
        raise RegistryCacheError("registry packument must contain an object-valued versions field")
    # JSON round-tripping is avoided here: callers may pass harmless mapping
    # subclasses, and cache serialization below performs the actual deep copy.
    return dict(packument)


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    """Encode a cache envelope using stable, compact UTF-8 JSON."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


class RegistryPackumentCache:
    """Atomic on-disk cache for validated raw npm packuments.

    The cache is disabled when ``cache_dir`` is ``None`` or ``enabled`` is
    false.  Cache failures are non-fatal: reads return misses and writes return
    false while a bounded diagnostic is retained for observability.  Network
    fetch failures are not swallowed by :func:`load_or_fetch_packument`.

    Args:
        cache_dir: Directory containing cache entries, or ``None`` to disable.
        enabled: Explicitly disable cache reads and writes when false.
        max_payload_bytes: Maximum encoded envelope size for one entry.
        diagnostics_limit: Maximum number of retained diagnostic messages.
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        *,
        enabled: bool = True,
        max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES,
        max_bytes: int | None = None,
        diagnostics_limit: int = DEFAULT_DIAGNOSTICS_LIMIT,
    ) -> None:
        """Initialize a bounded cache without performing filesystem I/O."""
        if max_bytes is not None:
            max_payload_bytes = max_bytes
        if max_payload_bytes <= 0:
            raise ValueError("max_payload_bytes must be greater than zero")
        if diagnostics_limit < 0:
            raise ValueError("diagnostics_limit must not be negative")
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.enabled = bool(enabled) and self.cache_dir is not None
        self.max_payload_bytes = int(max_payload_bytes)
        self.diagnostics_limit = int(diagnostics_limit)
        self._diagnostics: list[str] = []

    @property
    def diagnostics(self) -> tuple[str, ...]:
        """Return bounded diagnostics collected by cache reads and writes."""
        return tuple(self._diagnostics)

    def clear_diagnostics(self) -> None:
        """Discard retained cache diagnostics for a new planning operation."""
        self._diagnostics.clear()

    @staticmethod
    def cache_key(package_name: str) -> str:
        """Return a collision-resistant filename for a package identity."""
        name = validate_package_name(package_name)
        digest = hashlib.sha256(name.encode("utf-8")).hexdigest()
        return f"packument-v{CACHE_SCHEMA_VERSION}-{digest}.json"

    def path_for(self, package_name: str) -> Path:
        """Return the cache path for ``package_name`` without touching disk."""
        if self.cache_dir is None:
            raise ValueError("cache is disabled")
        return self.cache_dir / self.cache_key(package_name)

    def get(self, package_name: str) -> Packument | None:
        """Load a validated packument, returning ``None`` on a cache miss.

        Malformed, oversized, identity-mismatched, or partially written files
        are treated as corrupt entries.  They are removed on a best-effort basis
        and never escape as exceptions to the caller.
        """
        name = validate_package_name(package_name)
        if not self.enabled or self.cache_dir is None:
            return None
        path = self.path_for(name)
        try:
            size = path.stat().st_size
            if size > self.max_payload_bytes:
                raise RegistryCacheError("cache entry exceeds max_payload_bytes")
            with path.open("rb") as handle:
                raw = handle.read(self.max_payload_bytes + 1)
            if len(raw) > self.max_payload_bytes:
                raise RegistryCacheError("cache entry exceeds max_payload_bytes")
            envelope = json.loads(raw.decode("utf-8"))
            if not isinstance(envelope, Mapping):
                raise RegistryCacheError("cache entry is not a JSON object")
            if envelope.get("schema_version") != CACHE_SCHEMA_VERSION:
                raise RegistryCacheError("cache entry has an unsupported schema version")
            if envelope.get("package_name") != name:
                raise RegistryCacheError("cache entry package identity mismatch")
            cached = validate_packument(name, envelope.get("packument"))
            return cached
        except FileNotFoundError:
            return None
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            self._diagnose(f"read failed for {name!r}: {exc}")
            self._remove_corrupt_entry(path, name)
            return None

    def put(self, package_name: str, packument: Mapping[str, Any]) -> bool:
        """Atomically store a validated packument and report write success.

        Temporary files are created in the destination directory and are always
        cleaned up.  ``os.replace`` makes a completed write visible in one step;
        interrupted writes cannot leave a partial target entry.
        """
        name = validate_package_name(package_name)
        if not self.enabled or self.cache_dir is None:
            return False
        try:
            validated = validate_packument(name, packument)
            envelope: dict[str, Any] = {
                "package_name": name,
                "packument": validated,
                "schema_version": CACHE_SCHEMA_VERSION,
            }
            payload = _json_bytes(envelope)
            if len(payload) > self.max_payload_bytes:
                raise RegistryCacheError("packument exceeds max_payload_bytes")
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            destination = self.path_for(name)
            temporary_name: str | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    prefix=f".{destination.name}.",
                    suffix=".tmp",
                    dir=self.cache_dir,
                    delete=False,
                ) as handle:
                    temporary_name = handle.name
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary_name, destination)
                temporary_name = None
                return True
            finally:
                if temporary_name is not None:
                    try:
                        os.unlink(temporary_name)
                    except FileNotFoundError:
                        pass
                    except OSError as exc:
                        self._diagnose(f"cleanup failed for {name!r}: {exc}")
        except (OSError, TypeError, ValueError) as exc:
            self._diagnose(f"write failed for {name!r}: {exc}")
            return False

    def load_or_fetch(self, package_name: str, fetcher: PackumentFetcher) -> Packument:
        """Return a cached packument or fetch, validate, and cache it."""
        return load_or_fetch_packument(package_name, fetcher, cache=self)

    def _diagnose(self, message: str) -> None:
        """Append one bounded diagnostic without allowing diagnostics to fail work."""
        if self.diagnostics_limit <= 0:
            return
        self._diagnostics.append(str(message)[:_MAX_DIAGNOSTIC_LENGTH])
        del self._diagnostics[self.diagnostics_limit :]

    def _remove_corrupt_entry(self, path: Path, package_name: str) -> None:
        """Remove one corrupt entry without masking the original read failure."""
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            self._diagnose(f"cleanup failed for corrupt {package_name!r}: {exc}")


def load_or_fetch_packument(
    package_name: str,
    fetcher: PackumentFetcher,
    *,
    cache: RegistryPackumentCache | None = None,
) -> Packument:
    """Load a packument from cache or an injected fetcher.

    Args:
        package_name: npm package identity to fetch.
        fetcher: Callable receiving the package name and returning raw JSON.
        cache: Optional cache.  ``None`` and disabled caches perform no disk I/O.

    Returns:
        A validated detached packument mapping.

    Raises:
        ValueError: If the package name or fetched packument is invalid.
        Exception: Any exception raised by the injected fetcher is propagated.
    """
    name = validate_package_name(package_name)
    if cache is not None:
        cached = cache.get(name)
        if cached is not None:
            return cached
    fetched = validate_packument(name, fetcher(name))
    if cache is not None:
        cache.put(name, fetched)
    return fetched


logger = logging.getLogger(__name__)

_NPM_REGISTRY_URL = "https://registry.npmjs.org"
_NPM_REQUEST_TIMEOUT_SECONDS = 15


def _stable_version_key(version: str) -> tuple[int, int, int] | None:
    """Return a comparable key for a stable semver string."""
    match = re.match(r"^v?(\d+)\.(\d+)\.(\d+)$", version.strip())
    if not match:
        return None
    return tuple(int(part) for part in match.groups())


def _stable_semantic_version(raw_version: str) -> NpmVersion | None:
    """Return a stable semantic-version object, or ``None`` for prereleases."""
    value = str(raw_version or "").strip().lstrip("vV")
    try:
        version = NpmVersion(value)
    except ValueError:
        return None
    if version.prerelease:
        return None
    return version


def _dependency_requirement(metadata: dict[str, Any], package_name: str) -> str | None:
    """Return the published requirement for a child package, if declared."""
    for field in ("dependencies", "optionalDependencies", "peerDependencies"):
        requirements = metadata.get(field)
        if isinstance(requirements, dict) and package_name in requirements:
            requirement = requirements[package_name]
            if isinstance(requirement, str):
                return requirement
    return None


def _requirement_matches(requirement: str | None, version: NpmVersion) -> bool:
    """Return whether an npm dependency range accepts *version*."""
    if not requirement:
        return False
    try:
        normalized = _normalize_npm_operator_whitespace(requirement)
        return bool(normalized) and NpmSpec(normalized).match(version)
    except (TypeError, ValueError):
        return False


def _normalise_dependency_ancestry(
    dependency_ancestry: list[str] | tuple[str, ...] | None,
    *,
    parent_package_name: str,
    child_package_name: str,
) -> list[str]:
    """Validate and normalize the outermost-to-leaf dependency chain."""
    if not dependency_ancestry:
        return [parent_package_name, child_package_name]

    ancestry = [str(package).strip() for package in dependency_ancestry if str(package).strip()]
    if len(ancestry) < 2:
        raise ValueError("dependency_ancestry must contain the parent and vulnerable child")
    if ancestry[0] != parent_package_name or ancestry[-1] != child_package_name:
        raise ValueError(
            "dependency_ancestry must begin with parent_package_name and end with "
            "child_package_name"
        )
    return ancestry


def _parse_dependency_ancestry(value: str) -> list[str]:
    """Parse comma- or arrow-separated dependency ancestry from tool input."""
    return [
        package.strip() for package in re.split(r"\s*(?:,|->|→)\s*", value or "") if package.strip()
    ]


def _chain_accepts_fixed_child(
    ancestry: list[str],
    package_index: int,
    package_metadata: dict[str, Any],
    child_version: NpmVersion,
    registry_data_by_package: dict[str, dict[str, Any]],
    memo: dict[tuple[int, str], bool],
) -> bool:
    """Return whether one published package version can resolve the rest of a chain.

    The metadata at ``package_index`` is already fixed by the candidate being
    evaluated. For each intermediate package, this walks its published stable
    versions and keeps only versions accepted by the preceding package's
    dependency range. The final edge is checked directly against the OSV-fixed
    child version.
    """
    next_package = ancestry[package_index + 1]
    requirement = _dependency_requirement(package_metadata, next_package)
    if package_index == len(ancestry) - 2:
        return _requirement_matches(requirement, child_version)

    next_registry_data = registry_data_by_package.get(next_package)
    if not next_registry_data:
        return False

    for raw_version, next_metadata in (next_registry_data.get("versions") or {}).items():
        if not isinstance(next_metadata, dict):
            continue
        next_version = _stable_semantic_version(str(raw_version))
        if next_version is None or not _requirement_matches(requirement, next_version):
            continue
        memo_key = (package_index + 1, str(next_version))
        compatible = memo.get(memo_key)
        if compatible is None:
            compatible = _chain_accepts_fixed_child(
                ancestry,
                package_index + 1,
                next_metadata,
                child_version,
                registry_data_by_package,
                memo,
            )
            memo[memo_key] = compatible
        if compatible:
            return True
    return False


def select_npm_parent_version(
    data: dict[str, Any],
    *,
    parent_package_name: str | None = None,
    child_package_name: str,
    child_fixed_version: str,
    installed_parent_version: str,
    selection: str,
    attempted_versions: set[str] | None = None,
    dependency_ancestry: list[str] | tuple[str, ...] | None = None,
    registry_data_by_package: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Select a stable parent release compatible with the fixed child.

    Args:
        data: Raw npm registry metadata for the parent package.
        parent_package_name: Name of the directly declared parent. Required
            when ``dependency_ancestry`` is provided.
        child_package_name: Vulnerable transitive package name.
        child_fixed_version: The OSV-fixed child version to satisfy.
        installed_parent_version: Currently resolved parent version.
        selection: ``minimum``, ``same_major``, or ``latest``.
        attempted_versions: Parent versions already tried for this task.
        dependency_ancestry: Package names ordered from the directly declared
            parent to the vulnerable child. When omitted, the existing direct
            parent-to-child check is used.
        registry_data_by_package: Registry metadata for intermediate packages in
            ``dependency_ancestry``. The public planning tool fetches this data;
            callers of this pure selector can inject it for deterministic tests.

    Returns:
        A mapping containing compatible candidates, selected version, and
        latest compatible releases. Prereleases and releases at or below the
        installed parent are excluded.
    """
    selection = (selection or "").strip().lower()
    if selection not in {"minimum", "same_major", "latest"}:
        raise ValueError("selection must be minimum, same_major, or latest")
    child_version = _stable_semantic_version(child_fixed_version)
    installed_version = _stable_semantic_version(installed_parent_version)
    if child_version is None:
        raise ValueError("child_fixed_version must be a stable semver")
    if installed_version is None:
        raise ValueError("installed_parent_version must be a stable semver")

    ancestry: list[str] | None = None
    registry_data = dict(registry_data_by_package or {})
    if dependency_ancestry:
        effective_parent_name = (parent_package_name or str(dependency_ancestry[0])).strip()
        ancestry = _normalise_dependency_ancestry(
            dependency_ancestry,
            parent_package_name=effective_parent_name,
            child_package_name=child_package_name,
        )
        registry_data[effective_parent_name] = data

    attempted = {
        str(version).strip().lstrip("vV")
        for version in (attempted_versions or set())
        if str(version).strip()
    }
    compatible: list[NpmVersion] = []
    for raw_version, metadata in (data.get("versions") or {}).items():
        if not isinstance(metadata, dict):
            continue
        version = _stable_semantic_version(str(raw_version))
        if version is None or version <= installed_version:
            continue
        if ancestry:
            if not _chain_accepts_fixed_child(
                ancestry,
                0,
                metadata,
                child_version,
                registry_data,
                {},
            ):
                continue
        elif not _requirement_matches(
            _dependency_requirement(metadata, child_package_name), child_version
        ):
            continue
        compatible.append(version)

    compatible = sorted(set(compatible))
    same_major = [version for version in compatible if version.major == installed_version.major]
    unattempted_compatible = [version for version in compatible if str(version) not in attempted]
    unattempted_same_major = [version for version in same_major if str(version) not in attempted]
    osv_minimum = min(unattempted_compatible) if unattempted_compatible else None
    same_major_candidate = max(unattempted_same_major) if unattempted_same_major else None
    npm_latest_candidate: NpmVersion | None = None
    dist_tags = data.get("dist-tags") or {}
    if isinstance(dist_tags, dict):
        tagged = _stable_semantic_version(str(dist_tags.get("latest", "")))
        if tagged is not None and tagged in unattempted_compatible:
            npm_latest_candidate = tagged
    # Fixtures and private registries sometimes omit dist-tags.  In that
    # case, npm's effective latest stable release is the highest eligible
    # release.  It still occupies the single ``npm_latest`` policy slot.
    if npm_latest_candidate is None and unattempted_compatible:
        npm_latest_candidate = max(unattempted_compatible)

    # Keep the registry surface intentionally small.  These are the only
    # versions that the Supervisor may expose to a model or worker plan;
    # compatibility analysis above may still inspect the complete registry
    # internally for transitive range validation.
    strategic_versions = {
        version
        for version in (
            osv_minimum,
            same_major_candidate,
            npm_latest_candidate,
        )
        if version is not None
    }
    if selection == "minimum":
        selected = osv_minimum
    elif selection == "same_major":
        selected = same_major_candidate
    else:
        selected = npm_latest_candidate
    return {
        "selected": str(selected) if selected is not None else None,
        "compatible": [str(version) for version in compatible],
        "same_major": [str(version) for version in sorted(same_major, reverse=True)],
        "latest": str(max(compatible)) if compatible else None,
        "same_major_latest": str(max(same_major)) if same_major else None,
        "attempted": sorted(attempted),
        "osv_minimum": str(osv_minimum) if osv_minimum is not None else None,
        "npm_latest": str(npm_latest_candidate) if npm_latest_candidate is not None else None,
        "strategic_candidates": [str(version) for version in sorted(strategic_versions)],
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fetch_package_data(
    package_name: str,
    *,
    fetcher: PackumentFetcher | None = None,
    cache: RegistryPackumentCache | None = None,
) -> dict[str, Any]:
    """Fetch raw JSON, optionally using an injected fetcher and cache.

    The keyword-only seams are intentionally not part of any LangChain tool
    schema.  With no injected arguments this retains the direct network
    behavior used by existing callers.
    """
    if fetcher is None:

        def fetch_from_registry(name: str) -> dict[str, Any]:
            encoded = quote(name, safe="")
            url = f"{_NPM_REGISTRY_URL}/{encoded}"
            response = requests.get(url, timeout=_NPM_REQUEST_TIMEOUT_SECONDS)
            if response.status_code == 404:
                raise ValueError("404")
            response.raise_for_status()
            return response.json()

        fetcher = fetch_from_registry
    return load_or_fetch_packument(package_name, fetcher, cache=cache)


@traceable(run_type="tool", name="supervisor.fetch_registry_candidates")
def fetch_registry_candidates(
    package_name: str,
    security_floor: str,
    attempted_versions: set[str] | None = None,
    *,
    fetcher: PackumentFetcher | None = None,
    cache: RegistryPackumentCache | None = None,
) -> list[RegistryCandidate]:
    """Fetch a small stable npm candidate set as typed policy inputs.

    Args:
        package_name: Package name accepted by the npm registry.
        security_floor: Lowest stable version that addresses the finding.
        attempted_versions: Versions already tried by prior worker attempts.

    Returns:
        At most three candidates sorted by ascending semantic-version key.  The
        candidates represent the lowest OSV-safe version, the latest same-major
        version, and npm's effective ``latest`` release when those values are
        distinct and eligible.

    Raises:
        ValueError: If ``security_floor`` is not stable semver or the package
            is not present in the registry.
        requests.RequestException: If the registry request fails.
    """
    package_name = (package_name or "").strip()
    floor = (security_floor or "").strip().lstrip("vV")
    if not package_name:
        raise ValueError("package_name must not be empty")
    floor_key = _stable_version_key(floor)
    if floor_key is None:
        raise ValueError(f"Invalid security floor: {security_floor}")

    attempted = {
        str(version).strip().lstrip("vV")
        for version in (attempted_versions or set())
        if str(version).strip()
    }
    try:
        if fetcher is None and cache is None:
            # Keep the one-argument private seam intact for existing mocks.
            data = _fetch_package_data(package_name)
        else:
            data = _fetch_package_data(package_name, fetcher=fetcher, cache=cache)
    except requests.RequestException as exc:
        raise ValueError(f"Could not fetch registry data for {package_name}: {exc}") from exc
    all_candidates: list[RegistryCandidate] = []
    for raw_version in data.get("versions") or {}:
        version = str(raw_version).strip().lstrip("vV")
        key = _stable_version_key(version)
        if key is None:
            continue
        all_candidates.append(
            RegistryCandidate(
                version=version,
                semver_key=key,
                security_floor_met=key >= floor_key,
                is_stable=True,
                same_major=key[0] == floor_key[0],
                already_attempted=version in attempted,
            )
        )

    eligible = [
        candidate
        for candidate in all_candidates
        if candidate.is_stable and candidate.security_floor_met and not candidate.already_attempted
    ]
    eligible.sort(key=lambda candidate: (candidate.semver_key, candidate.version))
    by_version = {candidate.version: candidate for candidate in eligible}
    roles_by_version: dict[str, set[str]] = {}

    if eligible:
        roles_by_version.setdefault(eligible[0].version, set()).add("osv_minimum")
        same_major = [candidate for candidate in eligible if candidate.same_major]
        if same_major:
            roles_by_version.setdefault(same_major[-1].version, set()).add("same_major")
    dist_tags = data.get("dist-tags") or {}
    tagged_value = dist_tags.get("latest") if isinstance(dist_tags, dict) else None
    tagged_version = _stable_semantic_version(str(tagged_value or ""))
    npm_latest_version = (
        str(tagged_version)
        if tagged_version is not None and str(tagged_version) in by_version
        else eligible[-1].version
        if eligible
        else None
    )
    if npm_latest_version is not None:
        roles_by_version.setdefault(npm_latest_version, set()).add("npm_latest")

    candidates = [
        by_version[version].model_copy(
            update={"selection_roles": tuple(sorted(roles_by_version.get(version, set())))}
        )
        for version in sorted(roles_by_version, key=lambda value: _stable_version_key(value))
    ]
    return candidates


@tool
@traceable(run_type="tool", name="supervisor.plan_npm_parent_version")
def plan_npm_parent_version(
    parent_package_name: str,
    child_package_name: str,
    child_fixed_version: str,
    installed_parent_version: str,
    selection: str,
    attempted_versions: str = "",
    dependency_ancestry: str = "",
) -> str:
    """Select a parent release whose published range accepts a fixed child.

    ``minimum`` returns the lowest compatible stable parent release newer than
    the installed parent. ``same_major`` and ``latest`` preserve the normal
    retry ordering while remaining constrained by the child's fixed version.
    ``dependency_ancestry`` is a comma- or arrow-separated chain ordered from
    the directly declared parent to the vulnerable child. Intermediate package
    metadata is fetched and evaluated for multi-hop findings.
    This is a read-only supervisor planning tool; workers never receive it.
    """
    parent_package_name = (parent_package_name or "").strip()
    child_package_name = (child_package_name or "").strip()
    child_fixed_version = (child_fixed_version or "").strip().lstrip("vV")
    installed_parent_version = (installed_parent_version or "").strip().lstrip("vV")
    selection = (selection or "").strip().lower()
    attempted = {
        version.strip().lstrip("vV")
        for version in (attempted_versions or "").split(",")
        if version.strip()
    }
    if not parent_package_name or not child_package_name:
        return "ERROR: parent_package_name and child_package_name must not be empty."
    if selection not in {"minimum", "same_major", "latest"}:
        return "ERROR: selection must be minimum, same_major, or latest."

    ancestry = _parse_dependency_ancestry(dependency_ancestry)
    if ancestry:
        try:
            _normalise_dependency_ancestry(
                ancestry,
                parent_package_name=parent_package_name,
                child_package_name=child_package_name,
            )
        except ValueError as exc:
            return f"ERROR: Could not plan parent '{parent_package_name}': {exc}"

    fetched_package_name = parent_package_name
    try:
        data = _fetch_package_data(parent_package_name)
        registry_data_by_package = {parent_package_name: data}
        for intermediate_package in ancestry[1:-1]:
            fetched_package_name = intermediate_package
            registry_data_by_package[intermediate_package] = _fetch_package_data(
                intermediate_package
            )
        result = select_npm_parent_version(
            data,
            parent_package_name=parent_package_name,
            child_package_name=child_package_name,
            child_fixed_version=child_fixed_version,
            installed_parent_version=installed_parent_version,
            selection=selection,
            attempted_versions=attempted,
            dependency_ancestry=ancestry or None,
            registry_data_by_package=registry_data_by_package,
        )
    except ValueError as exc:
        if str(exc) == "404":
            return (
                f"PACKAGE NOT FOUND: '{fetched_package_name}' does not exist on the npm registry."
            )
        return f"ERROR: Could not plan parent '{parent_package_name}': {exc}"
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: Could not plan parent '{parent_package_name}': {exc}"

    same_major_latest = result["same_major_latest"]
    latest = result["latest"]
    strategic_candidates = result.get("strategic_candidates") or []
    lines = [
        f"# NPM Parent Version Plan: {parent_package_name}",
        f"- Selection: {selection}",
        f"- Child Package: {child_package_name}",
        f"- Dependency Ancestry: {' -> '.join(ancestry) if ancestry else f'{parent_package_name} -> {child_package_name}'}",
        f"- Child Security Floor: {child_fixed_version}",
        f"- Installed Parent: {installed_parent_version}",
        f"- Selected Version: {result['selected'] or 'NONE'}",
        f"- Same-Major Latest: {same_major_latest or 'NONE'}",
        f"- Latest Compatible: {latest or 'NONE'}",
        f"- Npm Latest: {result.get('npm_latest') or 'NONE'}",
        f"- Attempted Versions: {', '.join(result['attempted']) or 'none'}",
        f"- Compatible Parent Versions: {', '.join(strategic_candidates) or 'none'}",
        f"- Eligible Candidates: {', '.join(strategic_candidates) or 'none'}",
    ]
    return "\n".join(lines)


_MAVEN_SEARCH_URL = "https://search.maven.org/solrsearch/select"
_MAVEN_REPOSITORY_BASE_URL = "https://repo.maven.apache.org/maven2"
_MAVEN_REQUEST_TIMEOUT_SECONDS = 15
_REQUEST_MAX_ATTEMPTS = 3
_REQUEST_RETRY_BACKOFF_SECONDS = 0.5
_RETRYABLE_HTTP_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})
_ROWS_PER_PAGE = 200
_COORDINATE_PART_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*\Z")


def _canonical_coordinate(package_name: str) -> tuple[str, str]:
    """Validate a canonical group:artifact coordinate before querying Maven Central."""
    value = str(package_name or "").strip()
    parts = value.split(":")
    if (
        len(parts) != 2
        or any(not _COORDINATE_PART_RE.fullmatch(part) for part in parts)
        or any(part.endswith(".") or ".." in part for part in parts)
    ):
        raise ValueError("package_name must be a canonical Maven group:artifact coordinate")
    return parts[0], parts[1]


def _request_with_retries(
    url: str,
    *,
    package_name: str,
    source: str,
    params: dict[str, Any] | None = None,
) -> requests.Response:
    """Issue one bounded Maven Central request using the shared retry policy."""
    for attempt in range(_REQUEST_MAX_ATTEMPTS):
        try:
            if params is None:
                response = requests.get(url, timeout=_MAVEN_REQUEST_TIMEOUT_SECONDS)
            else:
                response = requests.get(
                    url,
                    params=params,
                    timeout=_MAVEN_REQUEST_TIMEOUT_SECONDS,
                )
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            retryable = isinstance(exc, (requests.Timeout, requests.ConnectionError)) or (
                status_code in _RETRYABLE_HTTP_STATUS_CODES
            )
            if not retryable or attempt + 1 == _REQUEST_MAX_ATTEMPTS:
                raise ValueError(
                    f"Could not query Maven Central {source} for {package_name} "
                    f"after {attempt + 1} attempt(s): {exc}"
                ) from exc
            time.sleep(_REQUEST_RETRY_BACKOFF_SECONDS * (2**attempt))
    raise ValueError(f"Could not query Maven Central {source} for {package_name}")


def _parse_metadata_versions(response: requests.Response, package_name: str) -> list[str]:
    """Parse and validate the canonical repository's version list."""
    content = getattr(response, "content", None)
    if content is None:
        content = getattr(response, "text", None)
    if not content:
        raise ValueError(f"Maven Central returned empty repository metadata for {package_name}")
    try:
        root = ET.fromstring(content)
    except (ET.ParseError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Maven Central returned invalid repository metadata for {package_name}: {exc}"
        ) from exc

    versioning_nodes = [element for element in root if element.tag == "versioning"]
    if root.tag != "metadata" or len(versioning_nodes) != 1:
        raise ValueError(f"Maven Central returned malformed repository metadata for {package_name}")
    versions_nodes = [element for element in versioning_nodes[0] if element.tag == "versions"]
    if len(versions_nodes) != 1:
        raise ValueError(f"Maven Central returned malformed repository metadata for {package_name}")

    versions_node = versions_nodes[0]
    version_nodes = list(versions_node)
    if not version_nodes:
        raise ValueError(f"Maven Central returned empty repository metadata for {package_name}")
    if versions_node.attrib or (versions_node.text and versions_node.text.strip()):
        raise ValueError(f"Maven Central returned malformed repository metadata for {package_name}")

    versions: list[str] = []
    seen_versions: set[str] = set()
    for version_node in version_nodes:
        if (
            version_node.tag != "version"
            or version_node.attrib
            or len(version_node)
            or (version_node.tail and version_node.tail.strip())
        ):
            raise ValueError(
                f"Maven Central returned malformed repository metadata for {package_name}"
            )
        version = (version_node.text or "").strip()
        if not version:
            raise ValueError(f"Maven Central returned empty repository metadata for {package_name}")
        try:
            compare_maven_versions(version, version)
        except ValueError as exc:
            raise ValueError(
                f"Maven Central returned an invalid Maven version for {package_name}: {version!r}"
            ) from exc
        if version not in seen_versions:
            seen_versions.add(version)
            versions.append(version)
    return versions


def _fetch_solr_versions(
    group_id: str,
    artifact_id: str,
    package_name: str,
) -> list[str]:
    """Fetch and deduplicate all Solr versions, preserving its pagination rules."""
    versions: list[str] = []
    seen_versions: set[str] = set()
    start = 0
    num_found: int | None = None
    while num_found is None or start < num_found:
        params = {
            "q": f'g:"{group_id}" AND a:"{artifact_id}"',
            "core": "gav",
            "rows": _ROWS_PER_PAGE,
            "start": start,
            "wt": "json",
        }
        response = _request_with_retries(
            _MAVEN_SEARCH_URL,
            package_name=package_name,
            source="Solr",
            params=params,
        )
        try:
            payload: Any = response.json()
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Maven Central returned invalid JSON for {package_name}: {exc}"
            ) from exc

        if not isinstance(payload, dict) or not isinstance(payload.get("response"), dict):
            raise ValueError(f"Maven Central returned an invalid response for {package_name}")
        registry_response = payload["response"]
        raw_num_found = registry_response.get("numFound")
        docs = registry_response.get("docs")
        if (
            not isinstance(raw_num_found, int)
            or isinstance(raw_num_found, bool)
            or raw_num_found < 0
            or not isinstance(docs, list)
        ):
            raise ValueError(f"Maven Central returned invalid pagination data for {package_name}")
        if num_found is None or raw_num_found != num_found:
            num_found = raw_num_found
        if not docs and start < num_found:
            raise ValueError(f"Maven Central pagination stopped early for {package_name}")
        for document in docs:
            if isinstance(document, dict) and isinstance(document.get("v"), str):
                version = document["v"].strip()
                if version and version not in seen_versions:
                    seen_versions.add(version)
                    versions.append(version)
        start += len(docs)
        if not docs:
            break
    return versions


def _metadata_url(group_id: str, artifact_id: str) -> str:
    group_path = group_id.replace(".", "/")
    return f"{_MAVEN_REPOSITORY_BASE_URL}/{group_path}/{artifact_id}/maven-metadata.xml"


def fetch_maven_registry_candidates(
    package_name: str,
    security_floor: str,
    attempted_versions: set[str] | None = None,
) -> list[MavenRegistryCandidate]:
    """Fetch only the lowest and highest stable Maven releases meeting a floor.

    Canonical repository metadata is preferred, with the Solr API as a bounded
    fallback. Registry timestamps, response ordering, and declared latest/release
    values never participate in version selection.
    """
    group_id, artifact_id = _canonical_coordinate(package_name)
    floor = str(security_floor or "").strip()
    if not is_stable_maven_version(floor):
        raise ValueError(f"Invalid Maven security floor: {security_floor!r}")

    attempted = tuple(str(value).strip() for value in (attempted_versions or set()))

    def already_attempted(version: str) -> bool:
        for attempted_version in attempted:
            try:
                if compare_maven_versions(version, attempted_version) == 0:
                    return True
            except ValueError:
                if version == attempted_version:
                    return True
        return False

    metadata_url = _metadata_url(group_id, artifact_id)
    try:
        metadata_response = _request_with_retries(
            metadata_url,
            package_name=package_name,
            source="repository metadata",
        )
        versions = _parse_metadata_versions(metadata_response, package_name)
    except ValueError as metadata_error:
        try:
            versions = _fetch_solr_versions(group_id, artifact_id, package_name)
        except ValueError as solr_error:
            raise ValueError(
                f"Could not fetch Maven Central versions for {package_name}; "
                f"repository metadata failed: {metadata_error}; "
                f"Solr fallback failed: {solr_error}"
            ) from solr_error
    eligible: list[str] = []
    for version in versions:
        if not is_stable_maven_version(version):
            continue
        try:
            if compare_maven_versions(version, floor) >= 0:
                eligible.append(version)
        except ValueError:
            continue
    if not eligible:
        return []
    eligible.sort(key=cmp_to_key(compare_maven_versions))
    minimum = eligible[0]
    latest = eligible[-1]
    if compare_maven_versions(minimum, latest) == 0:
        return [
            MavenRegistryCandidate(
                version=minimum,
                security_floor_met=True,
                is_stable=True,
                already_attempted=already_attempted(minimum),
                selection_roles=("maven_minimum", "maven_latest"),
            )
        ]
    return [
        MavenRegistryCandidate(
            version=minimum,
            security_floor_met=True,
            is_stable=True,
            already_attempted=already_attempted(minimum),
            selection_roles=("maven_minimum",),
        ),
        MavenRegistryCandidate(
            version=latest,
            security_floor_met=True,
            is_stable=True,
            already_attempted=already_attempted(latest),
            selection_roles=("maven_latest",),
        ),
    ]


_PYPI_JSON_URL = "https://pypi.org/pypi"
_PYPI_REQUEST_TIMEOUT_SECONDS = 15
_MAX_CANDIDATE_RELEASES = 3


def _normalized_name(package_name: str) -> str:
    normalized = normalize_python_package_name(str(package_name or "").strip())
    if not normalized:
        raise ValueError("package_name must not be empty")
    return normalized


def _fetch_json(package_name: str, version: str | None = None) -> dict[str, Any]:
    normalized = _normalized_name(package_name)
    suffix = f"/{quote(version, safe='')}" if version is not None else ""
    url = f"{_PYPI_JSON_URL}/{quote(normalized, safe='')}{suffix}/json"
    try:
        response = requests.get(url, timeout=_PYPI_REQUEST_TIMEOUT_SECONDS)
    except RequestException as exc:
        raise ValueError(
            f"NETWORK ERROR: Could not fetch PyPI data for {normalized}: {exc}"
        ) from exc
    if response.status_code == 404:
        raise ValueError(f"PACKAGE NOT FOUND: '{normalized}' does not exist on PyPI.")
    try:
        response.raise_for_status()
    except RequestException as exc:
        raise ValueError(
            f"NETWORK ERROR: Could not fetch PyPI data for {normalized}: {exc}"
        ) from exc
    try:
        data = response.json()
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"MALFORMED PYPI METADATA: response for '{normalized}' is not JSON."
        ) from exc
    if not isinstance(data, dict):
        raise ValueError(f"MALFORMED PYPI METADATA: response for '{normalized}' is not an object.")
    return data


def _stable_version(raw_version: Any) -> Version | None:
    try:
        version = Version(str(raw_version).strip())
    except InvalidVersion:
        return None
    if version.is_prerelease or version.is_devrelease:
        return None
    return version


def _release_file_evidence(data: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    releases = data.get("releases")
    if not isinstance(releases, dict):
        raise ValueError("MALFORMED PYPI METADATA: 'releases' must be an object.")
    valid: dict[str, list[dict[str, Any]]] = {}
    for raw_version, raw_files in releases.items():
        if not isinstance(raw_files, list):
            raise ValueError(
                f"MALFORMED PYPI METADATA: files for release '{raw_version}' are not a list."
            )
        if any(not isinstance(file_record, dict) for file_record in raw_files):
            raise ValueError(
                f"MALFORMED PYPI METADATA: invalid file record for release '{raw_version}'."
            )
        version = _stable_version(raw_version)
        # An empty release or a release with only yanked uploads is not an
        # installable registry candidate. Keep each file record until deciding.
        if (
            version is None
            or not raw_files
            or all(file_record.get("yanked", False) for file_record in raw_files)
        ):
            continue
        valid[str(version)] = raw_files
    return valid


def _canonical_attempts(attempted_versions: set[str] | None) -> set[Version]:
    normalized: set[Version] = set()
    for raw_version in attempted_versions or set():
        version = _stable_version(raw_version)
        if version is not None:
            normalized.add(version)
    return normalized


def _validated_floor(security_floor: str) -> Version:
    try:
        floor = Version(str(security_floor).strip())
    except InvalidVersion as exc:
        raise ValueError(f"INVALID SECURITY FLOOR: {security_floor}") from exc
    if floor.is_prerelease or floor.is_devrelease:
        raise ValueError(f"INVALID SECURITY FLOOR: {security_floor}")
    return floor


def _candidates_from_data(
    data: dict[str, Any],
    security_floor: str,
    attempted_versions: set[str] | None = None,
) -> list[RegistryCandidate]:
    floor = _validated_floor(security_floor)
    release_files = _release_file_evidence(data)
    attempted = _canonical_attempts(attempted_versions)
    eligible = sorted(
        version
        for version in (Version(value) for value in release_files)
        if version >= floor and version not in attempted
    )
    if not eligible:
        return []

    role_versions: dict[str, set[str]] = {str(eligible[0]): {"osv_minimum"}}
    same_major = [version for version in eligible if version.release[0] == floor.release[0]]
    if same_major:
        role_versions.setdefault(str(same_major[-1]), set()).add("same_major")
    role_versions.setdefault(str(eligible[-1]), set()).add("pypi_latest")

    candidates: list[RegistryCandidate] = []
    for version in sorted(Version(value) for value in role_versions):
        canonical = str(version)
        candidates.append(
            RegistryCandidate(
                version=canonical,
                ecosystem="pypi",
                semver_key=None,
                security_floor_met=version >= floor,
                is_stable=True,
                same_major=version.release[0] == floor.release[0],
                already_attempted=False,
                selection_roles=tuple(sorted(role_versions[canonical])),
            )
        )
    return candidates[:_MAX_CANDIDATE_RELEASES]


def get_pypi_versions(package_name: str) -> list[str]:
    """Return raw release keys advertised by the normalized PyPI project."""
    data = _fetch_json(package_name)
    releases = data.get("releases")
    if not isinstance(releases, dict):
        raise ValueError("MALFORMED PYPI METADATA: 'releases' must be an object.")
    return [str(version) for version in releases]


def get_pypi_latest_version(package_name: str) -> str | None:
    """Return the highest stable release with at least one non-yanked file."""
    data = _fetch_json(package_name)
    release_files = _release_file_evidence(data)
    if not release_files:
        return None
    return str(max(Version(version) for version in release_files))


def select_python_safe_version(
    package_name: str,
    security_floor: str,
    attempted_versions: set[str] | None = None,
) -> str | None:
    """Return the lowest unattempted stable PyPI version meeting the floor."""
    candidates = fetch_pypi_registry_candidates(package_name, security_floor, attempted_versions)
    return select_version(candidates, SCARemediationStage.OSV_MINIMUM, set())


def fetch_pypi_registry_candidates(
    package_name: str,
    security_floor: str,
    attempted_versions: set[str] | None = None,
) -> list[RegistryCandidate]:
    """Fetch at most three stable, file-backed PEP 440 candidates.

    Roles identify the lowest version at the security floor, the highest
    eligible version in the floor's major release, and the latest eligible
    stable release. A version shared by roles carries each applicable role.
    """
    normalized_name = _normalized_name(package_name)
    floor = _validated_floor(security_floor)
    data = _fetch_json(normalized_name)
    return _candidates_from_data(data, str(floor), attempted_versions)


def get_pypi_release_requires_dist(package_name: str, version: str) -> list[str]:
    """Read ``Requires-Dist`` metadata from one exact PyPI release endpoint."""
    normalized_name = _normalized_name(package_name)
    try:
        canonical_version = str(Version(str(version).strip()))
    except InvalidVersion as exc:
        raise ValueError(f"INVALID VERSION: {version}") from exc
    data = _fetch_json(normalized_name, canonical_version)
    info = data.get("info")
    if not isinstance(info, dict):
        raise ValueError(
            f"MALFORMED PYPI METADATA: release '{canonical_version}' has no info object."
        )
    requires_dist = info.get("requires_dist")
    if requires_dist is None:
        return []
    if not isinstance(requires_dist, list) or any(
        not isinstance(item, str) for item in requires_dist
    ):
        raise ValueError(
            f"MALFORMED PYPI METADATA: Requires-Dist for '{canonical_version}' is not a string list."
        )
    return list(requires_dist)


def _parent_requires_child(
    parent_name: str,
    parent_version: str,
    child_name: str,
    child_version: Version,
) -> bool:
    for raw_requirement in get_pypi_release_requires_dist(parent_name, parent_version):
        try:
            requirement = Requirement(raw_requirement)
        except InvalidRequirement:
            continue
        if (
            normalize_python_package_name(requirement.name) == child_name
            and not requirement.extras
            and requirement.marker is None
            and requirement.specifier.contains(child_version, prereleases=False)
        ):
            return True
    return False


@tool
def plan_python_parent_version(
    parent_package_name: str,
    child_package_name: str,
    child_fixed_version: str,
    installed_parent_version: str,
    selection: str,
    attempted_versions: str = "",
    dependency_ancestry: str = "",
) -> str:
    """Read-only report of PyPI parent releases accepting a fixed child."""
    try:
        parent_name = _normalized_name(parent_package_name)
        child_name = _normalized_name(child_package_name)
    except ValueError as exc:
        return f"ERROR: {exc}"
    try:
        child_version = Version(str(child_fixed_version).strip())
        installed_version = Version(str(installed_parent_version).strip())
    except InvalidVersion as exc:
        return f"ERROR: Invalid PEP 440 version: {exc}"
    if (
        child_version.is_prerelease
        or child_version.is_devrelease
        or installed_version.is_prerelease
        or installed_version.is_devrelease
    ):
        return "ERROR: Child and installed parent versions must be stable PEP 440 versions."
    selection = str(selection or "").strip().casefold()
    if selection not in {"minimum", "same_major", "latest"}:
        return "ERROR: selection must be minimum, same_major, or latest."
    ancestry = [
        part.strip()
        for part in re.split(r"\s*(?:,|->|→)\s*", dependency_ancestry or "")
        if part.strip()
    ]
    if ancestry and (
        len(ancestry) != 2
        or _normalized_name(ancestry[0]) != parent_name
        or _normalized_name(ancestry[-1]) != child_name
    ):
        return "ERROR: dependency_ancestry must be the one-hop parent-to-child path."

    attempted: set[str] = set()
    for raw_version in (attempted_versions or "").split(","):
        try:
            attempted.add(str(Version(raw_version.strip())))
        except InvalidVersion:
            continue
    attempted.add(str(installed_version))
    try:
        data = _fetch_json(parent_name)
        candidates = _candidates_from_data(data, str(installed_version), attempted)
        compatible = [
            candidate
            for candidate in candidates
            if _parent_requires_child(parent_name, candidate.version, child_name, child_version)
        ]
        release_files = _release_file_evidence(data)
        latest_stable = (
            str(max(Version(version) for version in release_files)) if release_files else None
        )
    except ValueError as exc:
        return f"ERROR: Could not plan PyPI parent '{parent_name}': {exc}"
    except Exception as exc:  # noqa: BLE001 - bounded tool-boundary diagnostics
        return f"ERROR: Could not plan PyPI parent '{parent_name}': {exc}"

    compatible.sort(key=lambda candidate: Version(candidate.version))
    role_versions: dict[str, set[str]] = {}
    if compatible:
        role_versions.setdefault(compatible[0].version, set()).add("osv_minimum")
        same_major = [
            candidate
            for candidate in compatible
            if Version(candidate.version).release[0] == installed_version.release[0]
        ]
        if same_major:
            role_versions.setdefault(same_major[-1].version, set()).add("same_major")
        role_versions.setdefault(compatible[-1].version, set()).add("pypi_latest")
    compatible = [
        candidate.model_copy(
            update={"selection_roles": tuple(sorted(role_versions[candidate.version]))}
        )
        for candidate in compatible
    ]
    role = {"minimum": "osv_minimum", "same_major": "same_major", "latest": "pypi_latest"}[
        selection
    ]
    selected_pool = [candidate for candidate in compatible if role in candidate.selection_roles]
    selected = selected_pool[0] if selected_pool else None
    pypi_latest = next(
        (
            candidate.version
            for candidate in compatible
            if "pypi_latest" in candidate.selection_roles
        ),
        None,
    )
    compatible_versions = ", ".join(candidate.version for candidate in compatible) or "NONE"
    eligible_versions = compatible_versions
    return "\n".join(
        [
            f"# PyPI Parent Version Plan: {parent_name}",
            f"- Selection: {selection}",
            f"- Child Package: {child_name}",
            f"- Dependency Ancestry: {' -> '.join(ancestry) if ancestry else f'{parent_name} -> {child_name}'}",
            f"- Child Security Floor: {child_version}",
            f"- Installed Parent: {installed_version}",
            f"- Selected Version: {selected.version if selected else 'NONE'}",
            f"- Selected: {selected.version if selected else 'NONE'}",
            f"- Eligible Candidates: {eligible_versions}",
            f"- Compatible Parent Versions: {compatible_versions}",
            f"- Latest Stable: {latest_stable or 'NONE'}",
            f"- PyPI Latest: {pypi_latest or 'NONE'}",
            f"- Attempted Versions: {', '.join(sorted(attempted, key=Version)) or 'none'}",
        ]
    )


__all__ = [
    "CACHE_SCHEMA_VERSION",
    "DEFAULT_DIAGNOSTICS_LIMIT",
    "DEFAULT_MAX_PAYLOAD_BYTES",
    "Packument",
    "PackumentFetcher",
    "RegistryCacheError",
    "RegistryPackumentCache",
    "load_or_fetch_packument",
    "validate_package_name",
    "validate_packument",
    "fetch_registry_candidates",
    "select_npm_parent_version",
    "plan_npm_parent_version",
    "fetch_maven_registry_candidates",
    "fetch_pypi_registry_candidates",
    "get_pypi_latest_version",
    "get_pypi_release_requires_dist",
    "get_pypi_versions",
    "plan_python_parent_version",
    "select_python_safe_version",
]
