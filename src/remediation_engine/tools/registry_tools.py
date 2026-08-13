"""
registry_tools.py - LangChain tools for querying public package registries.

Provides:
    view_npm_package_versions(package_name: str) -> str
    plan_npm_version(package_name, security_floor, selection, attempted_versions) -> str
    plan_npm_parent_version(
        parent_package_name, child_package_name, child_fixed_version,
        installed_parent_version, selection, attempted_versions,
        dependency_ancestry
    ) -> str

The tool queries the npm registry REST API and returns a bounded, LLM-readable
report covering:
- Package dist-tags (latest, next, beta, etc.)
- Package created/modified timestamps
- Last 30 published versions (newest first) with publish time, engines.node,
  peerDependencies, and deprecated notices
- Latest stable version per major series (last 6 majors), surfacing backported patches

Design principles:
- Read-only; never modifies any file or network state.
- Never raises network errors to the caller â€” all failures produce a plain-text
  error report safe for an LLM to read.
- Scoped packages (e.g. @scope/name) are URL-encoded correctly.
- Output is capped to keep LLM token usage bounded.
- ``plan_npm_version`` is the supervisor-owned selector for stable releases at
  or above an OSV security floor; update workers do not receive these tools.
"""

from __future__ import annotations

import json
import logging
import re
from contextlib import suppress
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote

import requests
from langchain_core.tools import tool
from semantic_version import NpmSpec, Version

logger = logging.getLogger(__name__)

_NPM_REGISTRY_URL = "https://registry.npmjs.org"
_REQUEST_TIMEOUT_SECONDS = 15
_MAX_RECENT_VERSIONS = 30
_MAX_MAJOR_SERIES = 6


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


def _format_timestamp(iso_str: str | None) -> str:
    """Format an ISO-8601 timestamp as a short human-readable string."""
    if not iso_str:
        return "unknown"
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d")
    except Exception:  # noqa: BLE001
        return iso_str[:10] if len(iso_str) >= 10 else iso_str


def _parse_major(version: str) -> int | None:
    """Return the major version number, or None if not parseable."""
    try:
        return int(version.split(".")[0])
    except (ValueError, IndexError):
        return None


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


def _build_version_entries(
    data: dict[str, Any],
    time_map: dict[str, str],
) -> list[dict[str, Any]]:
    """Return a list of version dicts sorted newest-first with metadata."""
    versions_data: dict[str, Any] = data.get("versions", {})
    entries = []

    for version, meta in versions_data.items():
        if not isinstance(meta, dict):
            continue
        publish_time = time_map.get(version)
        ts_key: datetime | None = None
        if publish_time:
            with suppress(Exception):
                ts_key = datetime.fromisoformat(publish_time.replace("Z", "+00:00"))

        entries.append(
            {
                "version": version,
                "publish_time": publish_time,
                "_ts": ts_key,
                "engines_node": meta.get("engines", {}).get("node"),
                "peer_deps": meta.get("peerDependencies"),
                "deprecated": meta.get("deprecated"),
            }
        )

    entries.sort(key=lambda e: e["_ts"] or datetime.min.replace(tzinfo=UTC), reverse=True)
    return entries


def _build_report(package_name: str, data: dict[str, Any]) -> str:
    """Build a bounded, LLM-readable report from the npm registry response."""
    time_map: dict[str, str] = data.get("time", {})
    created = _format_timestamp(time_map.get("created"))
    modified = _format_timestamp(time_map.get("modified"))
    dist_tags: dict[str, str] = data.get("dist-tags", {})

    lines = [
        f"# NPM Registry Report: {package_name}",
        f"- Created  : {created}",
        f"- Modified  : {modified}",
        "",
        "## dist-tags",
    ]
    if dist_tags:
        for tag, ver in sorted(dist_tags.items()):
            lines.append(f"  {tag}: {ver}")
    else:
        lines.append("  (none)")

    all_entries = _build_version_entries(data, time_map)
    recent_entries = all_entries[:_MAX_RECENT_VERSIONS]

    lines += ["", f"## Last {len(recent_entries)} Published Versions (newest first)"]
    for e in recent_entries:
        ts = _format_timestamp(e["publish_time"])
        row = f"  {e['version']}  ({ts})"
        if e.get("engines_node"):
            row += f"  [node: {e['engines_node']}]"
        if e.get("peer_deps"):
            peer_str = ", ".join(f"{k}@{v}" for k, v in list(e["peer_deps"].items())[:4])
            row += f"  [peers: {peer_str}]"
        if e.get("deprecated"):
            dep_msg = str(e["deprecated"])[:80]
            row += f"  [DEPRECATED: {dep_msg}]"
        lines.append(row)

    # Latest stable per major (last _MAX_MAJOR_SERIES majors)
    best_per_major: dict[int, str] = {}
    for e in all_entries:
        major = _parse_major(e["version"])
        if major is None:
            continue
        # Only non-prerelease versions (no hyphen in the version string)
        if "-" in e["version"]:
            continue
        if major not in best_per_major:
            best_per_major[major] = e["version"]  # already newest-first

    sorted_majors = sorted(best_per_major.keys(), reverse=True)[:_MAX_MAJOR_SERIES]

    if sorted_majors:
        lines += ["", "## Latest Stable Version per Major Series"]
        for major in sorted_majors:
            ver = best_per_major[major]
            entry = next((e for e in all_entries if e["version"] == ver), None)
            ts = _format_timestamp(entry["publish_time"]) if entry else "unknown"
            row = f"  v{major}.x â†’ {ver}  ({ts})"
            if entry and entry.get("deprecated"):
                row += "  [DEPRECATED]"
            lines.append(row)

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public LangChain tool
# ---------------------------------------------------------------------------


@tool
def view_npm_package_versions(package_name: str) -> str:
    """Query the npm registry and return a bounded summary of published versions.

    Use this to investigate available versions for a package before writing a
    remediation instruction. The report includes dist-tags (latest, next, etc.),
    the most recent 30 published versions with publish date, Node.js engine
    requirements, peer dependencies, and deprecation notices, plus the latest
    stable release per major series.

    Handles scoped packages (e.g. ``@scope/name``) correctly.

    Args:
        package_name: The npm package name, e.g. ``lodash`` or ``@types/node``.

    Returns:
        A plain-text report suitable for LLM reasoning.
    """
    package_name = (package_name or "").strip()
    if not package_name:
        return "ERROR: package_name must not be empty."

    try:
        data = _fetch_package_data(package_name)
    except ValueError as exc:
        if str(exc) == "404":
            return (
                f"PACKAGE NOT FOUND: '{package_name}' does not exist on the npm registry "
                f"(HTTP 404). The package may have been unpublished, or the name is misspelled."
            )
        return f"ERROR: Unexpected error fetching '{package_name}': {exc}"
    except requests.Timeout:
        return (
            f"NETWORK TIMEOUT: Could not reach the npm registry within "
            f"{_REQUEST_TIMEOUT_SECONDS}s for package '{package_name}'."
        )
    except requests.RequestException as exc:
        return f"NETWORK ERROR: Failed to fetch '{package_name}' from npm registry: {exc}"
    except json.JSONDecodeError as exc:
        return (
            f"MALFORMED RESPONSE: npm registry returned non-JSON data for '{package_name}': {exc}"
        )
    except Exception as exc:  # noqa: BLE001
        return f"UNEXPECTED ERROR: {exc}"

    try:
        return _build_report(package_name, data)
    except Exception as exc:  # noqa: BLE001
        return f"REPORT BUILD ERROR: Successfully fetched data but could not format it: {exc}"


@tool
def plan_npm_version(
    package_name: str,
    security_floor: str,
    selection: str,
    attempted_versions: str = "",
) -> str:
    """Select the next stable NPM version for a supervisor retry.

    ``selection`` must be ``same_major`` or ``latest``. Both selections are
    constrained to stable versions at or above ``security_floor``. ``same_major``
    limits candidates to the security floor's major series; ``latest`` selects
    the highest stable version published in the registry metadata.
    """
    package_name = (package_name or "").strip()
    security_floor = (security_floor or "").strip().lstrip("vV")
    selection = (selection or "").strip().lower()
    attempted = {
        version.strip().lstrip("vV")
        for version in (attempted_versions or "").split(",")
        if version.strip()
    }

    if not package_name:
        return "ERROR: package_name must not be empty."
    if selection not in {"same_major", "latest"}:
        return "ERROR: selection must be either 'same_major' or 'latest'."
    floor_key = _stable_version_key(security_floor)
    if floor_key is None:
        return f"ERROR: security_floor '{security_floor}' is not a stable semver."

    try:
        data = _fetch_package_data(package_name)
    except ValueError as exc:
        if str(exc) == "404":
            return f"PACKAGE NOT FOUND: '{package_name}' does not exist on the npm registry."
        return f"ERROR: Could not query '{package_name}': {exc}"
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: Could not query '{package_name}': {exc}"

    stable_versions: list[tuple[tuple[int, int, int], str]] = []
    for raw_version in data.get("versions") or {}:
        version = str(raw_version).strip().lstrip("vV")
        key = _stable_version_key(version)
        if key is not None and key >= floor_key:
            stable_versions.append((key, version))

    same_major = [item for item in stable_versions if item[0][0] == floor_key[0]]
    eligible = same_major if selection == "same_major" else stable_versions
    eligible.sort(key=lambda item: item[0], reverse=True)
    unattempted = [item for item in eligible if item[1] not in attempted]
    selected = unattempted[0][1] if unattempted else None
    latest_stable = max(stable_versions, default=(None, None), key=lambda item: item[0])[1]
    same_major_latest = max(same_major, default=(None, None), key=lambda item: item[0])[1]
    same_major_stage_skipped = bool(
        same_major_latest and latest_stable and same_major_latest == latest_stable
    )

    lines = [
        f"# NPM Version Plan: {package_name}",
        f"- Selection: {selection}",
        f"- Security Floor: {security_floor}",
        f"- Selected Version: {selected or 'NONE'}",
        f"- Same-Major Latest: {same_major_latest or 'NONE'}",
        f"- Latest Stable: {latest_stable or 'NONE'}",
        f"- Same-Major Stage: {'SKIPPED (same-major latest equals latest stable)' if same_major_stage_skipped else 'APPLICABLE'}",
        f"- Attempted Versions: {', '.join(sorted(attempted)) or 'none'}",
        f"- Eligible Candidates: {', '.join(version for _, version in eligible[:30]) or 'none'}",
    ]
    return "\n".join(lines)


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
