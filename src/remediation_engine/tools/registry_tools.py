"""Read-only npm registry tools used by deterministic Supervisor planning.

The module exposes two focused boundaries:

* ``fetch_registry_candidates`` returns stable, typed registry candidates for
  direct dependency selection.
* ``plan_npm_parent_version`` returns a bounded report for transitive parent
  compatibility, including dependency-range evidence.

Both tools are read-only and URL-encode scoped package names. Network and
registry failures are represented at the tool boundary as typed errors for
candidate fetching or bounded text diagnostics for parent planning. Neither
tool modifies the workspace, registry state, or dependency manifests.
"""

from __future__ import annotations

import logging
import re
from typing import Any
from urllib.parse import quote

import requests
from langchain_core.tools import tool
from semantic_version import NpmSpec, Version

from remediation_engine.contracts.version_policy import RegistryCandidate

logger = logging.getLogger(__name__)

_NPM_REGISTRY_URL = "https://registry.npmjs.org"
_REQUEST_TIMEOUT_SECONDS = 15


def _stable_version_key(version: str) -> tuple[int, int, int] | None:
    """Return a comparable key for a stable semver string."""
    match = re.match(r"^v?(\d+)\.(\d+)\.(\d+)$", version.strip())
    if not match:
        return None
    return tuple(int(part) for part in match.groups())


def _stable_semantic_version(raw_version: str) -> Version | None:
    """Return a stable semantic-version object, or ``None`` for prereleases."""
    value = str(raw_version or "").strip().lstrip("vV")
    try:
        version = Version(value)
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


def _requirement_matches(requirement: str | None, version: Version) -> bool:
    """Return whether an npm dependency range accepts *version*."""
    if not requirement:
        return False
    try:
        return NpmSpec(requirement).match(version)
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
    child_version: Version,
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
    compatible: list[Version] = []
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
    eligible = same_major if selection == "same_major" else compatible
    ordered = sorted(eligible, reverse=selection != "minimum")
    selected = next((version for version in ordered if str(version) not in attempted), None)
    return {
        "selected": str(selected) if selected is not None else None,
        "compatible": [str(version) for version in compatible],
        "same_major": [str(version) for version in sorted(same_major, reverse=True)],
        "latest": str(max(compatible)) if compatible else None,
        "same_major_latest": str(max(same_major)) if same_major else None,
        "attempted": sorted(attempted),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fetch_package_data(package_name: str) -> dict[str, Any]:
    """Fetch raw JSON from the npm registry for *package_name*.

    Raises ``requests.RequestException`` on network errors.
    Raises ``json.JSONDecodeError`` on malformed JSON.
    Raises ``ValueError`` with message "404" if the package is not found.
    """
    encoded = quote(package_name, safe="")
    url = f"{_NPM_REGISTRY_URL}/{encoded}"
    response = requests.get(url, timeout=_REQUEST_TIMEOUT_SECONDS)
    if response.status_code == 404:
        raise ValueError("404")
    response.raise_for_status()
    return response.json()


def fetch_registry_candidates(
    package_name: str,
    security_floor: str,
    attempted_versions: set[str] | None = None,
) -> list[RegistryCandidate]:
    """Fetch stable npm versions as typed deterministic policy inputs.

    Args:
        package_name: Package name accepted by the npm registry.
        security_floor: Lowest stable version that addresses the finding.
        attempted_versions: Versions already tried by prior worker attempts.

    Returns:
        Candidates sorted by ascending semantic-version key.

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
        data = _fetch_package_data(package_name)
    except requests.RequestException as exc:
        raise ValueError(f"Could not fetch registry data for {package_name}: {exc}") from exc
    candidates: list[RegistryCandidate] = []
    for raw_version in data.get("versions") or {}:
        version = str(raw_version).strip().lstrip("vV")
        key = _stable_version_key(version)
        if key is None:
            continue
        candidates.append(
            RegistryCandidate(
                version=version,
                semver_key=key,
                security_floor_met=key >= floor_key,
                is_stable=True,
                same_major=key[0] == floor_key[0],
                already_attempted=version in attempted,
            )
        )
    candidates.sort(key=lambda candidate: (candidate.semver_key, candidate.version))
    return candidates


@tool
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

    compatible = result["compatible"]
    same_major_latest = result["same_major_latest"]
    latest = result["latest"]
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
        f"- Attempted Versions: {', '.join(result['attempted']) or 'none'}",
        f"- Compatible Parent Versions: {', '.join(compatible) or 'none'}",
        f"- Eligible Candidates: {', '.join(result['same_major'] if selection == 'same_major' else compatible) or 'none'}",
    ]
    return "\n".join(lines)
