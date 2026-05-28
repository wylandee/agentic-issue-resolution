"""
edit_request_builder.py — Deterministic ``EditRequest`` construction for SCA bumps.

Phase 4.1 scope
---------------
This module handles **direct-dependency SCA version bumps** only.  The package
version string is updated on the exact line where it appears in the manifest
snippet.  All other cases (transitive overrides, lockfile-only packages, missing
snippets) return ``None`` so the graph can exit gracefully.

Public API
----------
``build_edit_request(localized_issue, fix_plan, repo_root, dry_run) -> Optional[EditRequest]``

    Returns a ready-to-apply ``EditRequest`` when a deterministic bump is possible,
    or ``None`` with a reason string when it is not.

The function is pure — no file I/O, no network calls, no mutations.
"""

from __future__ import annotations

import logging
import re
from typing import Optional, Tuple

from src.contracts.schemas import EditRequest, FixPlan, LocalizedIssue

log = logging.getLogger(__name__)

# Regex patterns for version strings in package.json and lockfiles.
# We match both:
#   "lodash": "^4.17.20"
#   "lodash": "4.17.20"
#   "lodash": "~4.17.20"
#   "lodash": ">=4.0.0 <5"
_PKG_JSON_VERSION_RE = re.compile(
    r"""
    (                         # group 1: key + prefix
        "[^"]+"\s*:\s*"       # "package-name": "
        [~^>=<|* ]*           # optional semver range prefix chars
    )
    ([\w.\-+]+)               # group 2: the version token to replace
    (                         # group 3: closing quote + optional trailing
        "
    )
    """,
    re.VERBOSE,
)

# requirements.txt style: package==1.2.3 or package>=1.2.3
_REQUIREMENTS_VERSION_RE = re.compile(
    r"""
    (                         # group 1: package name + operator
        [\w\-.\[\]]+          # package name (with extras)
        \s*[=><!~^]+\s*       # version operator
    )
    ([\w.\-+]+)               # group 2: the version token to replace
    """,
    re.VERBOSE,
)


def _replace_version_in_line(line: str, new_version: str, package_name: str) -> Optional[str]:
    """Replace the version string in *line* with *new_version*.

    Returns the updated line, or ``None`` if no version was found.

    Tries JSON-style first, then requirements.txt style.
    """
    # JSON style (package.json, yarn.lock lines)
    m = _PKG_JSON_VERSION_RE.search(line)
    if m:
        updated = _PKG_JSON_VERSION_RE.sub(
            lambda _m: _m.group(1) + new_version + _m.group(3),
            line,
            count=1,
        )
        if updated != line:
            return updated

    # requirements.txt style
    m2 = _REQUIREMENTS_VERSION_RE.search(line)
    if m2:
        updated = _REQUIREMENTS_VERSION_RE.sub(
            lambda _m: _m.group(1) + new_version,
            line,
            count=1,
        )
        if updated != line:
            return updated

    return None


def _find_version_line(
    snippet: str,
    package_name: str,
    current_version: str,
    fixed_version: str,
) -> Tuple[Optional[str], Optional[str]]:
    """Scan *snippet* for the line containing *current_version* for *package_name*.

    Returns ``(old_line, new_line)`` where ``new_line`` has the version replaced
    with *fixed_version*, or ``(None, None)`` if no suitable line was found.

    Preference: lines that reference both the package name (or its basename)
    and the current version, to avoid accidentally touching unrelated deps.
    """
    pkg_lower = package_name.lower().lstrip("@").split("/")[-1]  # strip scope for fuzzy match

    for raw_line in snippet.splitlines():
        # Must contain the current version somewhere
        if current_version not in raw_line:
            continue

        # Prefer lines that also reference the package name (or its basename)
        if pkg_lower not in raw_line.lower() and package_name not in raw_line:
            continue

        # Try JSON-style replacement first
        m = _PKG_JSON_VERSION_RE.search(raw_line)
        if m and current_version in m.group(2):
            new_line = _PKG_JSON_VERSION_RE.sub(
                lambda _m: _m.group(1) + fixed_version + _m.group(3),
                raw_line,
                count=1,
            )
            if new_line != raw_line:
                return raw_line, new_line

        # Try requirements.txt-style replacement
        m2 = _REQUIREMENTS_VERSION_RE.search(raw_line)
        if m2 and current_version in m2.group(2):
            new_line = _REQUIREMENTS_VERSION_RE.sub(
                lambda _m: _m.group(1) + fixed_version,
                raw_line,
                count=1,
            )
            if new_line != raw_line:
                return raw_line, new_line

    return None, None


def build_edit_request(
    localized_issue: LocalizedIssue,
    fix_plan: FixPlan,
    repo_root: str,
    *,
    dry_run: bool = True,
) -> Tuple[Optional[EditRequest], Optional[str]]:
    """Construct a deterministic ``EditRequest`` for a direct SCA version bump.

    Args:
        localized_issue: Output of the locator node; must have ``manifest_file``
                         and ``manifest_snippet`` populated.
        fix_plan:        Output of the planner node; must have
                         ``status=VERSION_FOUND`` and a non-empty ``fixed_version``.
        repo_root:       Absolute path to the repo (used for ``EditRequest.repo_root``).
        dry_run:         Forwarded to ``EditRequest.dry_run``.

    Returns:
        A ``(EditRequest, None)`` tuple on success, or ``(None, reason_str)`` when
        the build cannot proceed deterministically.
    """
    issue = localized_issue.issue

    # ---- Guard: need a manifest file ----
    manifest_file = localized_issue.manifest_file
    if not manifest_file:
        reason = "Cannot build EditRequest: LocalizedIssue.manifest_file is not set."
        log.warning(reason)
        return None, reason

    # ---- Guard: need a version to fix to ----
    fixed_version = fix_plan.fixed_version
    if not fixed_version:
        reason = "Cannot build EditRequest: FixPlan.fixed_version is not set."
        log.warning(reason)
        return None, reason

    # ---- Guard: need the current version ----
    current_version = issue.package_version
    if not current_version:
        reason = (
            f"Cannot build EditRequest for {manifest_file}: "
            "VulnerabilityIssue.package_version is not set."
        )
        log.warning(reason)
        return None, reason

    # ---- Guard: need a snippet to use as old_text anchor ----
    manifest_snippet = localized_issue.manifest_snippet
    if not manifest_snippet:
        reason = (
            f"Cannot build EditRequest for {manifest_file}: "
            "LocalizedIssue.manifest_snippet is not set — cannot anchor old_text."
        )
        log.warning(reason)
        return None, reason

    # ---- Find the specific version line inside the snippet ----
    package_name = issue.package_name or ""
    old_line, new_line = _find_version_line(manifest_snippet, package_name, current_version, fixed_version)

    if old_line is None or new_line is None:
        reason = (
            f"Cannot build EditRequest for {manifest_file}: "
            f"could not locate version '{current_version}' for package '{package_name}' "
            "in manifest_snippet."
        )
        log.warning(reason)
        return None, reason

    final_new_line = new_line

    log.info(
        "EditRequest built: %s  %s → %s  (dry_run=%s)",
        manifest_file,
        current_version,
        fixed_version,
        dry_run,
    )

    edit_request = EditRequest(
        repo_root=repo_root,
        file_path=manifest_file,
        old_text=old_line,
        new_text=final_new_line,
        dry_run=dry_run,
        issue_id=issue.id,
        rationale=(
            f"Bump {package_name} from {current_version} to {fixed_version} "
            f"to address {issue.cve_id or issue.rule_id or str(issue.id)}. "
            f"Strategy: {fix_plan.strategy_used}."
        ),
    )
    return edit_request, None
