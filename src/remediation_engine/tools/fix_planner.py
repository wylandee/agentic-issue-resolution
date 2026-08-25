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
4. **serper** â€” Google Search via Serper.dev for workaround snippets.
   Silently skipped if ``SERPER_API_KEY`` is not set.
5. **none** â€” all strategies exhausted; return a ``no_fix`` plan.

All network failures are caught, logged, and trigger continuation to the next
planner step.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Literal
from urllib.parse import quote, urlparse

import requests
from pydantic import BaseModel

from remediation_engine.contracts import FixPlanStatus, LocalizedIssue
from remediation_engine.orchestration.runtime_context import get_runtime_settings
from remediation_engine.tools.package_identity import package_name_from_purl

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OSV_QUERYBATCH_URL = "https://api.osv.dev/v1/querybatch"
OSV_VULN_URL = "https://api.osv.dev/v1/vulns/{vuln_id}"
NPM_REGISTRY_URL = "https://registry.npmjs.org/{package}/latest"
SERPER_SEARCH_URL = "https://google.serper.dev/search"

_REQUEST_TIMEOUT = 10  # seconds
_READ_WEB_PAGE_TIMEOUT = 15
_READ_WEB_PAGE_MAX_CHARS = 16_000
_JINA_READER_URL_PREFIX = "https://r.jina.ai/"
_GITHUB_API_URL_PREFIX = "https://api.github.com/"

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
# Structured Output Model for LLM Remediation Parsing
# ---------------------------------------------------------------------------


class SerperLLMResult(BaseModel):
    """Structured output schema for LLM extraction from Serper result web pages."""

    strategy: Literal["VERSION_BUMP", "CODE_WORKAROUND", "NO_FIX"]
    fixed_version: str | None = None
    workaround_snippets: list[str] | None = None
    reasoning: str = ""


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
    return package_name_from_purl(purl) or ""


def _extract_local_version(message: str | None) -> str | None:
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
    package_manager: str | None,
    is_direct: bool | None,
    manifest_file: str | None,
    parent_package_name: str | None = None,
    parent_declaration_type: str | None = None,
) -> str:
    """
    Generate a terse, actionable instruction for the Remedy agent.

    Direct dependencies: pin in place. Transitive dependencies with a known
    directly declared parent are explicitly parent-first; the Supervisor will
    choose the parent version before an override is considered.
    """
    manifest_name = os.path.basename(manifest_file) if manifest_file else "package.json"
    pm = (package_manager or "npm").lower()

    if is_direct:
        return _DIRECT_TMPL.format(
            package=package_name,
            manifest=manifest_name,
            version=fixed_version,
        )

    if parent_package_name:
        declaration = parent_declaration_type or "dependencies"
        return (
            f'Update the directly declared parent "{parent_package_name}" in {manifest_name} '
            f"({declaration}) to the minimum compatible released version that resolves "
            f'transitive package "{package_name}" to at least "{fixed_version}". '
            "Do not use a package override unless the parent update stages are exhausted."
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


def _extract_osv_workaround_snippets(vuln: dict[str, Any]) -> list[str] | None:
    """Extract workaround or mitigation text from OSV details/references."""
    snippets: list[str] = []

    def _add_snippet(candidate: str | None) -> None:
        if not candidate:
            return
        snippet = candidate.strip()
        if not snippet or not _contains_workaround_snippet(snippet):
            return
        if snippet not in snippets:
            snippets.append(snippet)

    for field in ("details", "summary"):
        value = vuln.get(field)
        if isinstance(value, str):
            _add_snippet(value)

    for reference in vuln.get("references") or []:
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


def _contains_workaround_snippet(text: str) -> bool:
    return any(keyword in text.lower() for keyword in _WORKAROUND_KEYWORDS)


def _extract_fixed_from_osv_vuln(
    vuln: dict[str, Any],
    package_name: str,
    current_version: str | None = None,
) -> tuple[str | None, list[str] | None]:
    """
    Walk an OSV vuln object's ``affected[].ranges[].events[]`` to find a
    non-Git ``fixed`` version.

    Prefers SEMVER ranges; falls back to ECOSYSTEM; checks database_specific
    extracted_events for GIT ranges before skipping raw commit hashes.
    """
    preferred: list[str] = []
    fallback: list[str] = []

    for affected in vuln.get("affected") or []:
        # Try to match by package name (case-insensitive); skip mismatches
        pkg_info = affected.get("package") or {}
        affected_name = pkg_info.get("name", "")
        if affected_name and affected_name.lower() != package_name.lower():
            continue

        for rng in affected.get("ranges") or []:
            rng_type = (rng.get("type") or "").upper()
            if rng_type == "GIT":
                db_specific = rng.get("database_specific") or {}
                extracted = db_specific.get("extracted_events") or []
                for event in extracted:
                    fixed = event.get("fixed")
                    if fixed:
                        fallback.append(str(fixed))
                continue  # commit hashes are not useful for manifest pins

            for event in rng.get("events") or []:
                fixed = event.get("fixed")
                if fixed:
                    if rng_type == "SEMVER":
                        preferred.append(str(fixed))
                    else:
                        fallback.append(str(fixed))

    fixed = _minimum_fixed_version(preferred or fallback, current_version=current_version)
    if fixed:
        return fixed, None

    return None, _extract_osv_workaround_snippets(vuln)


def _minimum_fixed_version(
    versions: list[str],
    current_version: str | None = None,
) -> str | None:
    """Return the lowest appropriate semver-like version from a collection of fixes.

    If current_version is provided, prioritizes fixes in the same major series
    (or the lowest fix >= current_version) over lower major backports.
    """
    parsed: list[tuple[tuple[int, int, int, int, str], str]] = []
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

    if current_version:
        cur_match = re.search(
            r"(?i)v?(\d+)(?:\.(\d+))?(?:\.(\d+))?(?:-([0-9A-Za-z.-]+))?",
            str(current_version).strip(),
        )
        if cur_match:
            cur_major = int(cur_match.group(1))
            cur_minor = int(cur_match.group(2) or 0)
            cur_patch = int(cur_match.group(3) or 0)
            cur_prerelease = cur_match.group(4) or ""
            cur_key = (cur_major, cur_minor, cur_patch, 0 if cur_prerelease else 1, cur_prerelease)

            # 1. Look for same-major fixes that are >= current_version
            same_major = [item for item in parsed if item[0][0] == cur_major and item[0] >= cur_key]
            if same_major:
                return min(same_major, key=lambda item: item[0])[1]

            # 2. Look for higher major fixes that are >= current_version
            higher_fixes = [item for item in parsed if item[0] >= cur_key]
            if higher_fixes:
                return min(higher_fixes, key=lambda item: item[0])[1]

    return min(parsed, key=lambda item: item[0])[1]


def _fetch_osv_vuln_detail(vuln_id: str) -> dict[str, Any] | None:
    """GET /v1/vulns/{id} — fetch a single OSV advisory in full detail."""
    url = OSV_VULN_URL.format(vuln_id=vuln_id)
    try:
        resp = requests.get(url, timeout=_REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        log.warning("OSV vuln detail fetch failed for %s: %s", vuln_id, exc)
        return None


def _query_osv_fixed_version(issue: Any) -> tuple[str | None, list[str] | None]:
    """
    Query the OSV querybatch API for the fixed version.

    Strategy:
    1. Build a querybatch query using package+version.
    2. If the result already embeds ``affected`` ranges, extract the fix.
    3. If it only returns vuln IDs, fetch each via GET /v1/vulns/{id}.
    4. Follow aliases (e.g. CVE -> GHSA) to retrieve structured ecosystem fix versions.
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

    query: dict[str, Any] = {"package": {"name": package_name, "ecosystem": eco}}

    current_version = getattr(issue, "package_version", None)
    if current_version:
        query["version"] = current_version

    try:
        resp = requests.post(
            OSV_QUERYBATCH_URL,
            json={"queries": [query]},
            timeout=_REQUEST_TIMEOUT,
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
        results: list[dict[str, Any]] = resp.json().get("results") or []
    except Exception as exc:
        log.warning("OSV querybatch failed: %s", exc)
        return None, None

    if not results:
        return None, None

    workaround_snippets: list[str] = []
    fixed_versions: list[str] = []
    fetched_details: dict[str, dict[str, Any]] = {}

    def _merge_snippets(snippets: list[str] | None) -> None:
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
        vuln for result in results for vuln in (result.get("vulns") or []) if isinstance(vuln, dict)
    ]

    def _get_vuln_detail(vuln_id: str) -> dict[str, Any] | None:
        if vuln_id in fetched_details:
            return fetched_details[vuln_id]
        detail = _fetch_osv_vuln_detail(vuln_id)
        if detail:
            fetched_details[vuln_id] = detail
        return detail

    def _matches_issue(vuln: dict[str, Any]) -> bool:
        if not wanted_ids:
            return True
        returned_ids = {
            str(identifier).strip().lower()
            for identifier in [vuln.get("id"), *(vuln.get("aliases") or [])]
            if identifier
        }
        if bool(wanted_ids & returned_ids):
            return True
        vuln_id = vuln.get("id")
        if vuln_id and not vuln.get("aliases") and not vuln.get("affected"):
            detail = _get_vuln_detail(str(vuln_id))
            if detail:
                detail_ids = {
                    str(identifier).strip().lower()
                    for identifier in [detail.get("id"), *(detail.get("aliases") or [])]
                    if identifier
                }
                return bool(wanted_ids & detail_ids)
        return False

    matching_vulns = [vuln for vuln in all_vulns if _matches_issue(vuln)]

    def _consume_vuln(vuln: dict[str, Any], depth: int = 0) -> None:
        vuln_to_process = vuln
        vuln_id = vuln.get("id")
        if "affected" not in vuln and vuln_id:
            detail = _get_vuln_detail(str(vuln_id))
            if detail:
                vuln_to_process = detail

        fixed, snippets = _extract_fixed_from_osv_vuln(
            vuln_to_process, package_name, current_version=current_version
        )
        if fixed:
            fixed_versions.append(fixed)
        elif snippets:
            _merge_snippets(snippets)

        # If no fixed version found directly, traverse ecosystem aliases (e.g. CVE -> GHSA)
        if not fixed and depth == 0 and isinstance(vuln_to_process, dict):
            aliases = vuln_to_process.get("aliases") or []
            for alias in aliases:
                alias_id = str(alias).strip()
                if (
                    alias_id
                    and alias_id.upper() != str(vuln_id).upper()
                    and alias_id.upper().startswith("GHSA-")
                ):
                    alias_detail = _get_vuln_detail(alias_id)
                    if alias_detail:
                        _consume_vuln(alias_detail, depth=depth + 1)

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
                detail = _get_vuln_detail(str(vuln_id))
                if detail:
                    _consume_vuln(detail)

    if fixed_versions:
        return _minimum_fixed_version(fixed_versions, current_version=current_version), None
    return None, (workaround_snippets or None)


# ---------------------------------------------------------------------------
# Waterfall step 3 â€” npm registry
# ---------------------------------------------------------------------------


def _fetch_npm_latest(package_name: str) -> str | None:
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
# Waterfall step 4 â€” Serper web search + LLM page content extraction
# ---------------------------------------------------------------------------


def _github_api_url(url: str) -> str | None:
    """Convert a github.com HTML URL to a GitHub REST API URL if applicable."""
    try:
        parsed = urlparse(url)
        if parsed.netloc.lower() not in {"github.com", "www.github.com"}:
            return None
        parts = [p for p in parsed.path.split("/") if p]
        if len(parts) < 2:
            return None
        owner, repository = parts[0], parts[1]
        if repository.endswith(".git"):
            repository = repository[:-4]
        base = f"{_GITHUB_API_URL_PREFIX}repos/{quote(owner, safe='')}/{quote(repository, safe='')}"
        if len(parts) == 2:
            return base
        resource = parts[2].lower()
        if resource in {"issues", "pulls", "commits"} and len(parts) >= 4:
            return f"{base}/{resource}/{quote(parts[3], safe='')}"
        if resource == "releases" and len(parts) >= 5 and parts[3].lower() == "tag":
            return f"{base}/releases/tags/{quote('/'.join(parts[4:]), safe='')}"
        if resource in {"blob", "raw", "tree"} and len(parts) >= 4:
            ref = quote(parts[3], safe="")
            content_path = quote("/".join(parts[4:]), safe="/")
            endpoint = f"{base}/contents/{content_path}" if content_path else f"{base}/contents"
            return f"{endpoint}?ref={ref}"
        return base
    except Exception:
        return None


def _github_headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github.raw+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = get_runtime_settings().github_token
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _decode_github_response(resp: Any, target_url: str) -> str:
    text = getattr(resp, "text", "") or ""
    if text.strip():
        try:
            payload = resp.json()
        except Exception:
            return text
        if isinstance(payload, dict):
            if payload.get("encoding") == "base64" and payload.get("content"):
                try:
                    from base64 import b64decode

                    return b64decode(str(payload["content"]).replace("\n", "")).decode("utf-8")
                except Exception:
                    return text
            title = (
                payload.get("title")
                or payload.get("name")
                or payload.get("login")
                or "GitHub Content"
            )
            body = payload.get("body") or payload.get("description") or payload.get("content") or ""
            if title or body:
                return f"Title: {title}\nURL: {target_url}\n\n{body}".strip()
            import json

            return json.dumps(payload, indent=2, ensure_ascii=False)
        if isinstance(payload, list):
            import json

            return json.dumps(payload, indent=2, ensure_ascii=False)
        return text
    return ""


def _fetch_page_content(url: str) -> str | None:
    """
    Fetch web page text using a 3-tier waterfall: GitHub API/Raw -> Jina Reader -> Direct HTTP.

    Truncates result to 16,000 characters. Catches all exceptions at each tier
    and falls through; returns None if all tiers fail.
    """
    target_url = (url or "").strip()
    if not target_url:
        return None

    # 1. GitHub API & Raw GitHub fallback
    if "github.com" in target_url and not target_url.startswith(
        "https://raw.githubusercontent.com"
    ):
        github_api_url = _github_api_url(target_url)
        if github_api_url:
            try:
                resp = requests.get(
                    github_api_url,
                    headers=_github_headers(),
                    timeout=_READ_WEB_PAGE_TIMEOUT,
                )
                resp.raise_for_status()
                text = _decode_github_response(resp, target_url)
                if text and text.strip():
                    return text[:_READ_WEB_PAGE_MAX_CHARS]
            except Exception as exc:
                log.debug("GitHub API fetch failed for %s: %s", target_url, exc)

        raw_url = (
            target_url.replace("github.com", "raw.githubusercontent.com")
            .replace("/blob/", "/")
            .replace("/tree/", "/")
        )
        try:
            resp = requests.get(raw_url, timeout=_READ_WEB_PAGE_TIMEOUT)
            resp.raise_for_status()
            text = resp.text or ""
            if text and text.strip():
                return text[:_READ_WEB_PAGE_MAX_CHARS]
        except Exception as exc:
            log.debug("Raw GitHub fetch failed for %s: %s", raw_url, exc)

    # 2. Jina Reader fallback
    jina_url = f"{_JINA_READER_URL_PREFIX}{target_url}"
    try:
        resp = requests.get(
            jina_url,
            headers={"Accept": "text/plain"},
            timeout=_READ_WEB_PAGE_TIMEOUT,
        )
        resp.raise_for_status()
        text = resp.text or ""
        if text and text.strip():
            return text[:_READ_WEB_PAGE_MAX_CHARS]
    except Exception as exc:
        log.debug("Jina Reader fetch failed for %s: %s", target_url, exc)

    # 3. Direct page fetch fallback
    try:
        resp = requests.get(target_url, timeout=_READ_WEB_PAGE_TIMEOUT)
        resp.raise_for_status()
        text = resp.text or ""
        if text and text.strip():
            return text[:_READ_WEB_PAGE_MAX_CHARS]
    except Exception as exc:
        log.debug("Direct page fetch failed for %s: %s", target_url, exc)

    return None


def _llm_extract_remediation(
    page_contents: list[dict[str, str]],
    package_name: str,
    vuln_id: str,
) -> dict[str, Any] | None:
    """
    Use LLM structured output to extract version bump or workaround steps from web page text.

    Returns a dict matching SerperLLMResult schema, or None if extraction fails.
    """
    if not page_contents:
        return None

    try:
        from langchain_openai import ChatOpenAI  # type: ignore[import]
    except ImportError:
        log.warning("langchain-openai not installed; skipping LLM page extraction.")
        return None

    settings = get_runtime_settings()
    api_key = settings.openai_api_key
    if not api_key:
        log.info("OPENAI_API_KEY not set â€” skipping LLM page extraction.")
        return None

    model_name = settings.triage_llm_model

    pages_text = []
    for idx, page in enumerate(page_contents, 1):
        url = page.get("url", "")
        content = page.get("content", "")
        pages_text.append(f"--- Page {idx}: {url} ---\n{content}")
    combined_pages = "\n\n".join(pages_text)

    prompt_text = (
        f"You are a security engineer analyzing web page findings for vulnerable package '{package_name}' "
        f"(vulnerability ID: '{vuln_id}').\n\n"
        f"Below are fetched web page contents from security advisories, release notes, or issue threads:\n\n"
        f"{combined_pages}\n\n"
        "Analyze the content carefully and extract actionable remediation info:\n"
        "1. If the page provides a clear patched version to bump to, set strategy='VERSION_BUMP' and fixed_version='X.Y.Z'.\n"
        "2. If the page provides a code workaround or mitigation steps, set strategy='CODE_WORKAROUND' and workaround_snippets=[...].\n"
        "3. If no actionable fixed version or code workaround is found, set strategy='NO_FIX'.\n"
    )

    try:
        llm = ChatOpenAI(model=model_name, temperature=0, api_key=api_key)
        structured_llm = llm.with_structured_output(SerperLLMResult)
        result: SerperLLMResult = structured_llm.invoke(prompt_text)
        return result.model_dump()
    except Exception as exc:
        log.warning("LLM extraction failed for %s (%s): %s", package_name, vuln_id, exc)
        return None


def _query_serper(issue: Any, package_name: str) -> list[dict[str, str]] | None:
    """
    Search Google via Serper.dev for vulnerability information.

    Returns up to 3 organic result items with keys 'url', 'snippet', 'title',
    or None if SERPER_API_KEY is not set, API fails, or organic results are empty.
    """
    api_key = get_runtime_settings().serper_api_key
    if not api_key:
        log.info("SERPER_API_KEY not set â€” skipping Serper search.")
        return None

    vuln_id = (
        getattr(issue, "cve_id", None)
        or getattr(issue, "ghsa_id", None)
        or getattr(issue, "rule_id", None)
        or ""
    )
    query = f"{package_name} {vuln_id} vulnerability workaround github".strip()

    log.info("Executing Serper search with query: %s", query)

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
        results: list[dict[str, str]] = []
        for item in organic:
            if isinstance(item, dict):
                link = item.get("link") or item.get("url") or ""
                if link:
                    results.append(
                        {
                            "url": str(link),
                            "snippet": str(item.get("snippet") or ""),
                            "title": str(item.get("title") or ""),
                        }
                    )
                    if len(results) >= 3:
                        break
        return results if results else None
    except Exception as exc:
        log.warning("Serper search failed: %s", exc)
        return None


def _serper_search_and_extract(issue: Any, package_name: str) -> dict[str, Any] | None:
    """
    Orchestrate Serper search, page fetch, and LLM remediation extraction.

    1. Query Serper for organic search results.
    2. Fetch page content for top result URLs (up to 3).
    3. Pass fetched page contents to LLM for structured extraction.
    4. Return LLM extraction dict, or None if any step fails/returns no content.
    """
    search_results = _query_serper(issue, package_name)
    if not search_results:
        return None

    pages: list[dict[str, str]] = []
    for item in search_results:
        url = item.get("url")
        if not url:
            continue
        content = _fetch_page_content(url)
        if content and content.strip():
            pages.append({"url": url, "content": content})

    if not pages:
        return None

    vuln_id = (
        getattr(issue, "cve_id", None)
        or getattr(issue, "ghsa_id", None)
        or getattr(issue, "rule_id", None)
        or ""
    )
    return _llm_extract_remediation(pages, package_name, str(vuln_id))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plan_fix(localized_issue: LocalizedIssue) -> dict:
    """
    Plan one SCA finding from OSV advisory data and Serper fallback.

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

    # Step 1: OSV query
    osv_result = _query_osv_fixed_version(issue)
    if isinstance(osv_result, tuple) and len(osv_result) == 2:
        fixed, snippets = osv_result
    else:  # Defensive fallback for integrations that return no OSV result.
        fixed, snippets = None, None

    if fixed:
        return _version_plan(
            package_name,
            fixed,
            package_manager,
            is_direct,
            manifest_file,
            strategy="osv_api",
            parent_package_name=localized_issue.parent_package_name,
            parent_declaration_type=localized_issue.parent_declaration_type,
        )
    if snippets:
        return {
            "status": FixPlanStatus.WORKAROUND_FOUND.value,
            "fixed_version": None,
            "workaround_snippets": snippets,
            "instruction": _WORKAROUND_INSTRUCTION,
            "strategy_used": "osv_api",
        }

    # Step 2: Serper web search fallback with LLM page parsing
    serper_llm_result = _serper_search_and_extract(issue, package_name)
    if serper_llm_result and isinstance(serper_llm_result, dict):
        strategy = serper_llm_result.get("strategy")
        if strategy == "VERSION_BUMP" and serper_llm_result.get("fixed_version"):
            return _version_plan(
                package_name,
                serper_llm_result["fixed_version"],
                package_manager,
                is_direct,
                manifest_file,
                strategy="serper_llm",
                parent_package_name=localized_issue.parent_package_name,
                parent_declaration_type=localized_issue.parent_declaration_type,
            )
        if strategy == "CODE_WORKAROUND" and serper_llm_result.get("workaround_snippets"):
            return {
                "status": FixPlanStatus.WORKAROUND_FOUND.value,
                "fixed_version": None,
                "workaround_snippets": serper_llm_result["workaround_snippets"],
                "instruction": _WORKAROUND_INSTRUCTION,
                "strategy_used": "serper_llm",
            }

    # Step 3: No fix found across all strategies
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
    package_manager: str | None,
    is_direct: bool | None,
    manifest_file: str | None,
    strategy: str,
    parent_package_name: str | None = None,
    parent_declaration_type: str | None = None,
) -> dict:
    """Return a ``version_found`` plan dict."""
    instruction = _build_instruction(
        package_name=package_name,
        fixed_version=fixed_version,
        package_manager=package_manager,
        is_direct=is_direct,
        manifest_file=manifest_file,
        parent_package_name=parent_package_name,
        parent_declaration_type=parent_declaration_type,
    )
    return {
        "status": FixPlanStatus.VERSION_FOUND.value,
        "fixed_version": fixed_version,
        "workaround_snippets": None,
        "instruction": instruction,
        "strategy_used": strategy,
    }
