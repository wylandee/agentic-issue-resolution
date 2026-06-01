"""
grouper.py — Deterministic issue grouping for the triage layer.

Public API
----------
group_issues(issues)          → List[VulnerabilityGroup]
    Primary entry point.  Groups both SAST and SCA issues.

group_sca_issues(issues)      → List[VulnerabilityGroup]
    Backwards-compatible alias — filters to SCA only then delegates.

Design
------
SCA grouping key: "sca:{normalised_file_path}:{package_name or purl}"
    * Multiple CVEs affecting the same component → one group, multiple cve_ids.
    * Duplicate CVE on the same component → deduplicated.
    * Cross-tool duplicates (Semgrep SCA + ODC) with the same CVE + package +
      file → merged into one group; sources list is unioned.
    * Representative issue: prefers one with fixed_version, then the one with
      the most non-None fields, then first-encountered.

SAST grouping key: "sast:{file_path}:{rule_id}:{line_start}-{line_end}"
    * Each distinct SAST location becomes a singleton group.
    * No enrichment is expected for pure SAST groups (no CVE IDs).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional

from src.contracts.schemas import (
    IssueSource,
    IssueType,
    VulnerabilityGroup,
    VulnerabilityIssue,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Key builders
# ---------------------------------------------------------------------------


def _sca_key(issue: VulnerabilityIssue) -> str:
    """Return the normalised grouping key for an SCA issue."""
    # Prefer package_name; fall back to purl component; last resort "unknown"
    component = (
        issue.package_name
        or (issue.purl.split("/")[-1].split("@")[0] if issue.purl else None)
        or "unknown"
    )
    file_part = issue.file_path or ""
    return f"sca:{file_part}:{component}"


def _sast_key(issue: VulnerabilityIssue) -> str:
    """Return the normalised grouping key for a SAST issue."""
    file_part = issue.file_path or ""
    rule_part = issue.rule_id or "unknown_rule"
    if issue.line_range:
        line_part = f"{issue.line_range.start}-{issue.line_range.end}"
    else:
        line_part = "0-0"
    return f"sast:{file_part}:{rule_part}:{line_part}"


# ---------------------------------------------------------------------------
# Representative-issue selection
# ---------------------------------------------------------------------------


def _field_richness(issue: VulnerabilityIssue) -> int:
    """Count non-None fields as a proxy for data richness."""
    return sum(
        1
        for v in issue.model_dump().values()
        if v is not None and v != [] and v != {}
    )


def _choose_representative(issues: List[VulnerabilityIssue]) -> VulnerabilityIssue:
    """
    Choose the most informative issue from a group.

    Priority:
    1. Has fixed_version set (most actionable for SCA).
    2. Highest field richness (most complete record).
    3. First in insertion order (stable tie-break).
    """
    with_fix = [i for i in issues if i.fixed_version]
    pool = with_fix if with_fix else issues
    return max(pool, key=_field_richness)


# ---------------------------------------------------------------------------
# SCA grouping
# ---------------------------------------------------------------------------


def _group_sca(issues: List[VulnerabilityIssue]) -> Dict[str, VulnerabilityGroup]:
    """Return a dict of group_id → VulnerabilityGroup for SCA issues."""
    buckets: Dict[str, List[VulnerabilityIssue]] = defaultdict(list)
    for issue in issues:
        key = _sca_key(issue)
        buckets[key].append(issue)

    groups: Dict[str, VulnerabilityGroup] = {}
    for key, members in buckets.items():
        # Deduplicate CVE IDs and versions
        seen_cves: list[str] = []
        seen_versions: list[str] = []
        seen_cve_set: set[str] = set()
        seen_ver_set: set[str] = set()
        sources_set: set[IssueSource] = set()

        for m in members:
            if m.cve_id and m.cve_id not in seen_cve_set:
                seen_cves.append(m.cve_id)
                seen_cve_set.add(m.cve_id)
            if m.package_version and m.package_version not in seen_ver_set:
                seen_versions.append(m.package_version)
                seen_ver_set.add(m.package_version)
            sources_set.add(m.source)

        rep = _choose_representative(members)
        component = rep.package_name or (
            rep.purl.split("/")[-1].split("@")[0] if rep.purl else None
        )

        groups[key] = VulnerabilityGroup(
            group_id=key,
            issue_type=IssueType.SCA,
            vulnerable_component=component,
            file_path=rep.file_path,
            cve_ids=seen_cves,
            versions=seen_versions,
            sources=list(sources_set),
            representative_issue_id=rep.id,
            issues=members,
        )

    return groups


# ---------------------------------------------------------------------------
# SAST grouping
# ---------------------------------------------------------------------------


def _group_sast(issues: List[VulnerabilityIssue]) -> Dict[str, VulnerabilityGroup]:
    """Return a dict of group_id → VulnerabilityGroup for SAST issues."""
    groups: Dict[str, VulnerabilityGroup] = {}
    for issue in issues:
        key = _sast_key(issue)
        if key in groups:
            # Same SAST location seen twice (e.g. duplicate rule run) — merge sources
            existing = groups[key]
            if issue.source not in existing.sources:
                existing.sources.append(issue.source)
            if issue not in existing.issues:
                existing.issues.append(issue)
            # Re-evaluate representative
            rep = _choose_representative(existing.issues)
            existing.representative_issue_id = rep.id
        else:
            groups[key] = VulnerabilityGroup(
                group_id=key,
                issue_type=IssueType.SAST,
                vulnerable_component=issue.rule_id,
                file_path=issue.file_path,
                cve_ids=[],         # SAST findings rarely carry a CVE
                versions=[],
                sources=[issue.source],
                representative_issue_id=issue.id,
                issues=[issue],
            )

    return groups


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def group_issues(issues: List[VulnerabilityIssue]) -> List[VulnerabilityGroup]:
    """
    Group a mixed list of SAST and SCA ``VulnerabilityIssue`` records.

    Returns one ``VulnerabilityGroup`` per distinct vulnerable
    component/location.  Cross-tool duplicates sharing the same CVE +
    package + file are merged into a single group.

    Parameters
    ----------
    issues:
        Flat list of issues from any scanner (Semgrep, ODC, etc.).

    Returns
    -------
    List[VulnerabilityGroup]
        Ordered by group_id (deterministic, testable).
    """
    if not issues:
        return []

    sca_issues = [i for i in issues if i.issue_type == IssueType.SCA]
    sast_issues = [i for i in issues if i.issue_type == IssueType.SAST]

    logger.debug(
        "Grouping %d issues (%d SCA, %d SAST).",
        len(issues),
        len(sca_issues),
        len(sast_issues),
    )

    all_groups: Dict[str, VulnerabilityGroup] = {}
    all_groups.update(_group_sca(sca_issues))
    all_groups.update(_group_sast(sast_issues))

    result = sorted(all_groups.values(), key=lambda g: g.group_id)
    logger.debug("Produced %d groups.", len(result))
    return result


def group_sca_issues(issues: List[VulnerabilityIssue]) -> List[VulnerabilityGroup]:
    """
    Backwards-compatible alias that groups only SCA issues.

    Equivalent to calling ``group_issues`` on a pre-filtered SCA-only list.
    """
    sca_only = [i for i in issues if i.issue_type == IssueType.SCA]
    return group_issues(sca_only)
