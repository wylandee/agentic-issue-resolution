"""
Fix Planner â€” pure planning tool for SCA vulnerability remediation.

Separation-of-Concerns role
----------------------------
This module sits *between* the SCA Manifest Locator and the Remedy agent.
It accepts a ``LocalizedIssue`` (which already knows *where* the package lives)
and answers: **what is the safest version to pin / what workaround exists?**

It does **not**:
  * Read or write files.
  * Modify Pydantic objects in place.
  * Apply any edits.

Public API
----------
``plan_fix(localized_issue) -> dict``
    Queries OSV for this finding's advisory and returns a plain ``dict`` whose
    keys mirror the ``FixPlan`` Pydantic model. It never queries NPM or Serper;
    those later strategy stages belong to the supervisor.
Current triage planner steps
----------------------------
1. **osv_api** â€” resolve this finding's CVE/GHSA advisory through OSV.
2. **none** â€” OSV supplied neither a fixed version nor mitigation guidance.

   Only attempted for npm/javascript ecosystem packages.
4. **serper** â€” Google Search via Serper.dev for workaround snippets.
   Silently skipped if ``SERPER_API_KEY`` is not set.
5. **none** â€” all strategies exhausted; return a ``no_fix`` plan.

All network failures are caught, logged, and trigger continuation to the next
planner step.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

import requests

from remediation_engine.contracts import FixPlan, FixPlanStatus, LocalizedIssue

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OSV_QUERYBATCH_URL = "https://api.osv.dev/v1/querybatch"
OSV_VULN_URL = "https://api.osv.dev/v1/vulns/{vuln_id}"
NPM_REGISTRY_URL = "https://registry.npmjs.org/{package}/latest"
SERPER_SEARCH_URL = "https://google.serper.dev/search"

_REQUEST_TIMEOUT = 10  # seconds

# Matches "update to version X.Y.Z", "upgrade to 3.0.0 or later", etc.
_VERSION_RE = re.compile(
    r"(?:update|upgrade|fix(?:ed)?\s+in|patched\s+in|use|requires?)\s+(?:to\s+)?(?:version\s+)?"
    r"v?(\d+\.\d+(?:\.\d+)?)(?:[-+][A-Za-z0-9._-]+)?",
    re.IGNORECASE,
)
_WORKAROUND_KEYWORDS = ("workaround", "mitigate", "mitigation")

# Instruction templates
_DIRECT_TMPL = 'Update "{package}" in {manifest} to version "{version}".'
_OVERRIDE_NPM_TMPL = (
    'Add or update "overrides": {{"{package}": "{version}"}} in {manifest} '
    "to pin the transitive dependency via npm overrides."
)
_OVERRIDE_YARN_TMPL = (
    'Add or update "resolutions": {{"{package}": "{version}"}} in {manifest} '
    "to pin the transitive dependency via Yarn resolutions."
)
_OVERRIDE_PNPM_TMPL = (
    'Add or update "pnpm": {{"overrides": {{"{package}": "{version}"}}}} in {manifest} '
    "to pin the transitive dependency via pnpm overrides."
)
_WORKAROUND_INSTRUCTION = (
    "Analyze the provided workaround_snippets to determine if a code edit "
    "can safely mitigate this vulnerability."
)
_NO_FIX_INSTRUCTION = "No upstream patch or workaround was found. Inform the user."


# ---------------------------------------------------------------------------
# Internal helpers â€” issue introspection
# ---------------------------------------------------------------------------


def _is_npm_issue(issue: Any) -> bool:
    """Return True if the issue's ecosystem is npm or javascript."""
    eco = (issue.ecosystem or "").lower()
    if eco in ("npm", "javascript"):
        return True
    purl = issue.purl or ""
    return purl.startswith("pkg:npm/") or purl.startswith("pkg:javascript/")


def _package_name_from_issue(issue: Any) -> str:
    """
    Return the canonical package name for network queries.

    Prefers ``issue.package_name``; falls back to a manual PURL parse for
    npm scoped packages where the library might have percent-encoded the name.
    """
    name = (issue.package_name or "").strip()
    if name:
        return name
    purl = issue.purl or ""
    if purl.startswith("pkg:npm/"):
        raw = purl[len("pkg:npm/"):]
        raw = raw.split("@")[0]  # drop version
        return raw.replace("%40", "@").replace("%2F", "/")
    return ""


def _extract_local_version(message: Optional[str]) -> Optional[str]:
    """
    Scan the ODC/Semgrep ``message`` for an embedded fixed-version hint.

    Examples that match:
      "Update to version 3.0.0 or later."
      "Fixed in v2.1.4."
      "Upgrade to 1.0.1."
    Returns the first version string found, or ``None``.
    """
    if not message:
        return None
    m = _VERSION_RE.search(message)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Internal helpers â€” instruction builder
# ---------------------------------------------------------------------------


def _build_instruction(
    package_name: str,
    fixed_version: str,
    package_manager: Optional[str],
    is_direct: Optional[bool],
    manifest_file: Optional[str],
) -> str:
    """
    Generate a terse, actionable instruction for the Remedy agent.

    Direct dependencies: pin in place.
    Transitive dependencies: use the package-manager override mechanism.
    """
    manifest_name = os.path.basename(manifest_file) if manifest_file else "package.json"
    pm = (package_manager or "npm").lower()

    if is_direct:
        return _DIRECT_TMPL.format(
            package=package_name,
            manifest=manifest_name,
            version=fixed_version,
        )

    if pm == "yarn":
        return _OVERRIDE_YARN_TMPL.format(
            package=package_name,
            manifest=manifest_name,
            version=fixed_version,
        )
    if pm == "pnpm":
        return _OVERRIDE_PNPM_TMPL.format(
            package=package_name,
            manifest=manifest_name,
            version=fixed_version,
        )
    return _OVERRIDE_NPM_TMPL.format(
        package=package_name,
        manifest=manifest_name,
        version=fixed_version,
    )


# ---------------------------------------------------------------------------
# Waterfall step 2 â€” OSV API
# ---------------------------------------------------------------------------


def _contains_workaround_keyword(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in _WORKAROUND_KEYWORDS)


def _extract_osv_workaround_snippets(vuln: Dict[str, Any]) -> Optional[List[str]]:
    """Extract workaround or mitigation text from OSV details/references."""
    snippets: List[str] = []

    def _add_snippet(candidate: Optional[str]) -> None:
        if not candidate:
            return
        snippet = candidate.strip()
        if not snippet or not _contains_workaround_keyword(snippet):
            return
        if snippet not in snippets:
            snippets.append(snippet)

    for field in ("details", "summary"):
        value = vuln.get(field)
        if isinstance(value, str):
            _add_snippet(value)

    for reference in (vuln.get("references") or []):
        if not isinstance(reference, dict):
            continue
        parts = [
            str(value).strip()
            for value in reference.values()
            if isinstance(value, str) and value.strip()
        ]
        if not parts:
            continue
        combined = " | ".join(parts)
        _add_snippet(combined)

    return snippets or None


def _extract_fixed_from_osv_vuln(
    vuln: Dict[str, Any],
    package_name: str,
) -> Tuple[Optional[str], Optional[List[str]]]:
    """
    Walk an OSV vuln object's ``affected[].ranges[].events[]`` to find a
    non-Git ``fixed`` version.

    Prefers SEMVER ranges; falls back to ECOSYSTEM; skips GIT commit ranges.
    """
    preferred: List[str] = []
    fallback: List[str] = []

    for affected in (vuln.get("affected") or []):
        # Try to match by package name (case-insensitive); skip mismatches
        pkg_info = affected.get("package") or {}
        affected_name = pkg_info.get("name", "")
        if affected_name and affected_name.lower() != package_name.lower():
            continue

        for rng in (affected.get("ranges") or []):
            rng_type = (rng.get("type") or "").upper()
            if rng_type == "GIT":
                continue  # commit hashes are not useful for manifest pins
            for event in (rng.get("events") or []):
                fixed = event.get("fixed")
                if fixed:
                    if rng_type == "SEMVER":
                        preferred.append(str(fixed))
                    else:
                        fallback.append(str(fixed))

    fixed = _minimum_fixed_version(preferred or fallback)
    if fixed:
        return fixed, None

    return None, _extract_osv_workaround_snippets(vuln)


def _minimum_fixed_version(versions: List[str]) -> Optional[str]:
    """Return the lowest semver-like version from a collection of fixes."""
    parsed: List[Tuple[Tuple[int, int, int, int, str], str]] = []
    for raw in versions:
        match = re.search(
            r"(?i)v?(\d+)(?:\.(\d+))?(?:\.(\d+))?(?:-([0-9A-Za-z.-]+))?",
            str(raw).strip(),
        )
        if not match:
            continue
        major = int(match.group(1))
        minor = int(match.group(2) or 0)
        patch = int(match.group(3) or 0)
        prerelease = match.group(4) or ""
        key = (major, minor, patch, 0 if prerelease else 1, prerelease)
        parsed.append((key, str(raw).strip().lstrip("vV")))
    if not parsed:
        return None
    return min(parsed, key=lambda item: item[0])[1]


def _fetch_osv_vuln_detail(vuln_id: str) -> Optional[Dict[str, Any]]:
    """GET /v1/vulns/{id} â€” fetch a single OSV advisory in full detail."""
    url = OSV_VULN_URL.format(vuln_id=vuln_id)
    try:
        resp = requests.get(url, timeout=_REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        log.warning("OSV vuln detail fetch failed for %s: %s", vuln_id, exc)
        return None


def _query_osv_fixed_version(issue: Any) -> Tuple[Optional[str], Optional[List[str]]]:
    """
    Query the OSV querybatch API for the fixed version.

    Strategy:
    1. Build a querybatch query using package+version.
    2. If the result already embeds ``affected`` ranges, extract the fix.
    3. If it only returns vuln IDs, fetch each via GET /v1/vulns/{id}.
    """
    package_name = _package_name_from_issue(issue)
    if not package_name:
        return None, None

    eco = (issue.ecosystem or "npm").lower()
    mapping = {
        "npm": "npm",
        "maven": "Maven",
        "pypi": "PyPI",
        "javascript": "npm",
    }
    eco = mapping.get(eco, eco.capitalize())

    query: Dict[str, Any] = {
        "package": {"name": package_name, "ecosystem": eco}
    }
    
    if issue.package_version:
        query["version"] = issue.package_version

    try:
        resp = requests.post(
            OSV_QUERYBATCH_URL,
            json={"queries": [query]},
            timeout=_REQUEST_TIMEOUT,
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
        results: List[Dict[str, Any]] = resp.json().get("results") or []
    except Exception as exc:
        log.warning("OSV querybatch failed: %s", exc)
        return None, None

    if not results:
        return None, None

    workaround_snippets: List[str] = []
    fixed_versions: List[str] = []

    def _merge_snippets(snippets: Optional[List[str]]) -> None:
        if not snippets:
            return
        for snippet in snippets:
            if snippet not in workaround_snippets:
                workaround_snippets.append(snippet)

    wanted_ids = {
        str(identifier).strip().lower()
        for identifier in (getattr(issue, "cve_id", None), getattr(issue, "ghsa_id", None))
        if identifier
    }

    all_vulns = [
        vuln
        for result in results
        for vuln in (result.get("vulns") or [])
        if isinstance(vuln, dict)
    ]

    def _matches_issue(vuln: Dict[str, Any]) -> bool:
        if not wanted_ids:
            return True
        returned_ids = {
            str(identifier).strip().lower()
            for identifier in [vuln.get("id"), *(vuln.get("aliases") or [])]
            if identifier
        }
        return bool(wanted_ids & returned_ids)

    matching_vulns = [vuln for vuln in all_vulns if _matches_issue(vuln)]

    def _consume_vuln(vuln: Dict[str, Any]) -> None:
        if "affected" in vuln:
            fixed, snippets = _extract_fixed_from_osv_vuln(vuln, package_name)
            if fixed:
                fixed_versions.append(fixed)
            _merge_snippets(snippets)
            return

        vuln_id = vuln.get("id")
        if not vuln_id:
            return
        detail = _fetch_osv_vuln_detail(str(vuln_id))
        if detail:
            fixed, snippets = _extract_fixed_from_osv_vuln(detail, package_name)
            if fixed:
                fixed_versions.append(fixed)
            _merge_snippets(snippets)

    for vuln in matching_vulns:
        _consume_vuln(vuln)

    # If querybatch returned no matching advisory, resolve the finding's own
    # CVE/GHSA directly instead of assigning another advisory's fix.
    if not matching_vulns and all_vulns and wanted_ids:
        for vuln_id in (
            getattr(issue, "cve_id", None),
            getattr(issue, "ghsa_id", None),
        ):
            if vuln_id:
                detail = _fetch_osv_vuln_detail(str(vuln_id))
                if detail:
                    fixed, snippets = _extract_fixed_from_osv_vuln(detail, package_name)
                    if fixed:
                        fixed_versions.append(fixed)
                    _merge_snippets(snippets)

    if fixed_versions:
        return _minimum_fixed_version(fixed_versions), None
    return None, (workaround_snippets or None)


# ---------------------------------------------------------------------------
# Waterfall step 3 â€” npm registry
# ---------------------------------------------------------------------------


def _fetch_npm_latest(package_name: str) -> Optional[str]:
    """
    Fetch the ``latest`` dist-tag version from the npm registry.

    Handles scoped packages like ``@scope/pkg`` (percent-encode the leading @).
    """
    encoded = quote(package_name, safe="")
    url = NPM_REGISTRY_URL.format(package=encoded)
    try:
        resp = requests.get(url, timeout=_REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        version = data.get("version")
        return str(version) if version else None
    except Exception as exc:
        log.warning("npm registry fetch failed for %s: %s", package_name, exc)
        return None


# ---------------------------------------------------------------------------
# Waterfall step 4 â€” Serper web search
# ---------------------------------------------------------------------------


def _search_serper_workarounds(issue: Any, package_name: str) -> Optional[List[str]]:
    """
    Search Google via Serper.dev for workaround snippets.

    Returns the top-3 organic ``snippet`` strings, or ``None`` if:
    - ``SERPER_API_KEY`` is not set.
    - The API call fails.
    - No organic results are returned.
    """
    api_key = os.environ.get("SERPER_API_KEY", "").strip()
    if not api_key:
        log.info("SERPER_API_KEY not set â€” skipping Serper workaround search.")
        return None

    vuln_id = issue.cve_id or issue.rule_id or ""
    
    query = f'{package_name} {vuln_id} vulnerability workaround github'
    
    log.info(f"Executing Serper search with query: {query}")

    try:
        resp = requests.post(
            SERPER_SEARCH_URL,
            json={"q": query},
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            timeout=_REQUEST_TIMEOUT,
        )

        resp.raise_for_status()
        data = resp.json()
        organic = data.get("organic") or []
        snippets = [
            item["snippet"]
            for item in organic
            if isinstance(item, dict) and item.get("snippet")
        ][:3]
        return snippets if snippets else None
    except Exception as exc:
        log.warning("Serper search failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plan_fix(localized_issue: LocalizedIssue) -> dict:
    """
    Plan one SCA finding from OSV advisory data only.

    NPM registry candidate selection intentionally does not happen here.  The
    supervisor owns the later retry stages so each advisory is classified
    independently before strategy-aware grouping.

    Returns a plain ``dict`` that mirrors the ``FixPlan`` Pydantic model and
    is always constructable as one:

        fix_plan = FixPlan(**plan_fix(localized_issue))

    The dict always has keys:
        status, fixed_version, workaround_snippets, instruction, strategy_used.

    No file I/O or mutations are performed.
    """
    issue = localized_issue.issue
    package_name = _package_name_from_issue(issue)
    package_manager = localized_issue.package_manager
    is_direct = localized_issue.is_direct_dependency
    manifest_file = localized_issue.manifest_file

    # Triage is deliberately OSV-first and OSV-only. NPM candidate selection is
    # owned by the Phase 5 supervisor after QA rejects this initial attempt.
    osv_result = _query_osv_fixed_version(issue)
    if isinstance(osv_result, tuple) and len(osv_result) == 2:
        fixed, snippets = osv_result
    else:  # Defensive fallback for integrations that return no OSV result.
        fixed, snippets = None, None
    if fixed:
        return _version_plan(
            package_name, fixed, package_manager, is_direct, manifest_file,
            strategy="osv_api",
        )
    if snippets:
        return {
            "status": FixPlanStatus.WORKAROUND_FOUND.value,
            "fixed_version": None,
            "workaround_snippets": snippets,
            "instruction": _WORKAROUND_INSTRUCTION,
            "strategy_used": "osv_api",
        }
    return {
        "status": FixPlanStatus.NO_FIX.value,
        "fixed_version": None,
        "workaround_snippets": None,
        "instruction": _NO_FIX_INSTRUCTION,
        "strategy_used": "none",
    }




# ---------------------------------------------------------------------------
# Internal plan builder
# ---------------------------------------------------------------------------


def _version_plan(
    package_name: str,
    fixed_version: str,
    package_manager: Optional[str],
    is_direct: Optional[bool],
    manifest_file: Optional[str],
    strategy: str,
) -> dict:
    """Return a ``version_found`` plan dict."""
    instruction = _build_instruction(
        package_name=package_name,
        fixed_version=fixed_version,
        package_manager=package_manager,
        is_direct=is_direct,
        manifest_file=manifest_file,
    )
    return {
        "status": FixPlanStatus.VERSION_FOUND.value,
        "fixed_version": fixed_version,
        "workaround_snippets": None,
        "instruction": instruction,
        "strategy_used": strategy,
    }


