"""Read-only PyPI registry tools used by deterministic Supervisor planning.

PyPI releases are ordered and normalized with PEP 440. Candidate eligibility
uses uploaded-file evidence rather than treating release keys as publishable
artifacts; no worker-facing operation in this module changes package state.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import quote

import requests
from langchain_core.tools import tool
from packaging.requirements import InvalidRequirement, Requirement
from packaging.version import InvalidVersion, Version
from requests import RequestException

from remediation_engine.contracts.schemas import SCARemediationStage
from remediation_engine.contracts.version_policy import RegistryCandidate, select_version
from remediation_engine.tools.package_identity import normalize_python_package_name

_PYPI_JSON_URL = "https://pypi.org/pypi"
_REQUEST_TIMEOUT_SECONDS = 15
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
        response = requests.get(url, timeout=_REQUEST_TIMEOUT_SECONDS)
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
    "fetch_pypi_registry_candidates",
    "get_pypi_latest_version",
    "get_pypi_release_requires_dist",
    "get_pypi_versions",
    "plan_python_parent_version",
    "select_python_safe_version",
]
