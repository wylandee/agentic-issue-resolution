"""
registry_tools.py - LangChain tools for querying public package registries.

Provides:
    view_npm_package_versions(package_name: str) -> str
    plan_npm_version(package_name, security_floor, selection, attempted_versions) -> str

The tool queries the npm registry REST API and returns a bounded, LLM-readable
report covering:
- Package dist-tags (latest, next, beta, etc.)
- Package created/modified timestamps
- Last 30 published versions (newest first) with publish time, engines.node,
  peerDependencies, and deprecated notices
- Latest stable version per major series (last 6 majors), surfacing backported patches

Design principles:
- Read-only; never modifies any file or network state.
- Never raises network errors to the caller — all failures produce a plain-text
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
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import requests
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

_NPM_REGISTRY_URL = "https://registry.npmjs.org"
_REQUEST_TIMEOUT_SECONDS = 15
_MAX_RECENT_VERSIONS = 30
_MAX_MAJOR_SERIES = 6


def _stable_version_key(version: str) -> Optional[tuple[int, int, int]]:
    """Return a comparable key for a stable semver string."""
    match = re.match(r"^v?(\d+)\.(\d+)\.(\d+)$", version.strip())
    if not match:
        return None
    return tuple(int(part) for part in match.groups())


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _format_timestamp(iso_str: Optional[str]) -> str:
    """Format an ISO-8601 timestamp as a short human-readable string."""
    if not iso_str:
        return "unknown"
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d")
    except Exception:  # noqa: BLE001
        return iso_str[:10] if len(iso_str) >= 10 else iso_str


def _parse_major(version: str) -> Optional[int]:
    """Return the major version number, or None if not parseable."""
    try:
        return int(version.split(".")[0])
    except (ValueError, IndexError):
        return None


def _fetch_package_data(package_name: str) -> Dict[str, Any]:
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
    data: Dict[str, Any],
    time_map: Dict[str, str],
) -> List[Dict[str, Any]]:
    """Return a list of version dicts sorted newest-first with metadata."""
    versions_data: Dict[str, Any] = data.get("versions", {})
    entries = []

    for version, meta in versions_data.items():
        if not isinstance(meta, dict):
            continue
        publish_time = time_map.get(version)
        ts_key: Optional[datetime] = None
        if publish_time:
            try:
                ts_key = datetime.fromisoformat(publish_time.replace("Z", "+00:00"))
            except Exception:  # noqa: BLE001
                pass

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

    entries.sort(key=lambda e: (e["_ts"] or datetime.min.replace(tzinfo=timezone.utc)), reverse=True)
    return entries


def _build_report(package_name: str, data: Dict[str, Any]) -> str:
    """Build a bounded, LLM-readable report from the npm registry response."""
    time_map: Dict[str, str] = data.get("time", {})
    created = _format_timestamp(time_map.get("created"))
    modified = _format_timestamp(time_map.get("modified"))
    dist_tags: Dict[str, str] = data.get("dist-tags", {})

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
            peer_str = ", ".join(
                f"{k}@{v}" for k, v in list(e["peer_deps"].items())[:4]
            )
            row += f"  [peers: {peer_str}]"
        if e.get("deprecated"):
            dep_msg = str(e["deprecated"])[:80]
            row += f"  [DEPRECATED: {dep_msg}]"
        lines.append(row)

    # Latest stable per major (last _MAX_MAJOR_SERIES majors)
    best_per_major: Dict[int, str] = {}
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
            row = f"  v{major}.x → {ver}  ({ts})"
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
            f"MALFORMED RESPONSE: npm registry returned non-JSON data for "
            f"'{package_name}': {exc}"
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

    stable_versions: List[tuple[tuple[int, int, int], str]] = []
    for raw_version in (data.get("versions") or {}).keys():
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
