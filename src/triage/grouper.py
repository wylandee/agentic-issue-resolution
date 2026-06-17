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
SCA grouping key: "sca:{manifest_file}:{package_name or purl}:{fix_strategy}"
    * Multiple CVEs affecting the same component are grouped only when the
      remediation strategy also matches (UPDATE_VERSION / WORKAROUND / NO_FIX).
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
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

try:
    from semantic_version import Version as SemanticVersion
except ImportError:  # pragma: no cover - optional runtime fallback
    SemanticVersion = None

from src.contracts.schemas import (
    FixPlan,
    FixPlanStatus,
    IssueSource,
    IssueType,
    LocalizedIssue,
    VulnerabilityGroup,
    VulnerabilityIssue,
)

logger = logging.getLogger(__name__)


def _dedupe_paths(paths: List[Optional[str]]) -> List[str]:
    """Return stable, deduplicated non-empty repo-relative paths."""
    result: List[str] = []
    seen: set[str] = set()
    for path in paths:
        if not path:
            continue
        if path in seen:
            continue
        result.append(path)
        seen.add(path)
    return result


# ---------------------------------------------------------------------------
# Key builders
# ---------------------------------------------------------------------------


def _component_from_issue(issue: VulnerabilityIssue) -> str:
    """Return the canonical SCA component identifier for an issue."""
    # Prefer package_name; fall back to purl component; last resort "unknown"
    return (
        issue.package_name
        or (issue.purl.split("/")[-1].split("@")[0] if issue.purl else None)
        or "unknown"
    )


def _fix_strategy_bucket(fix_plan: FixPlan) -> str:
    """Collapse planner outcomes into the grouping strategies used by SCA buckets."""
    if fix_plan.status == FixPlanStatus.VERSION_FOUND:
        return "UPDATE_VERSION"
    if fix_plan.status == FixPlanStatus.WORKAROUND_FOUND:
        return "WORKAROUND"
    return "NO_FIX"


def _sca_key(localized_issue: LocalizedIssue, fix_plan: FixPlan) -> str:
    """Return the normalised grouping key for an SCA issue + plan pair."""
    issue = localized_issue.issue
    component = _component_from_issue(issue)
    file_part = localized_issue.manifest_file or issue.file_path or ""
    strategy_part = _fix_strategy_bucket(fix_plan)
    return f"sca:{file_part}:{component}:{strategy_part}"


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


def _choose_representative_from_pairs(
    pairs: List[Tuple[LocalizedIssue, FixPlan]],
) -> VulnerabilityIssue:
    """Choose the most informative issue from a grouped SCA pair list."""
    with_fix = [localized.issue for localized, plan in pairs if plan.fixed_version]
    pool = with_fix if with_fix else [localized.issue for localized, _ in pairs]
    return max(pool, key=_field_richness)


def _normalise_semver_text(version: str) -> Optional[str]:
    """Extract a semver-like token and coerce partial versions for comparison."""
    match = re.search(
        r"(?i)v?\d+(?:\.\d+){0,2}(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?",
        version.strip(),
    )
    if not match:
        return None

    cleaned = match.group(0).lstrip("vV")
    core, sep, suffix = cleaned.partition("-")
    parts = core.split(".")
    while len(parts) < 3:
        parts.append("0")
    normalized = ".".join(parts[:3])
    if sep:
        normalized = f"{normalized}-{suffix}"
    return normalized


def _semver_sort_key(version: str) -> tuple[int, object]:
    """
    Return a deterministic comparison key for semver-like strings.

    Prefers ``semantic_version`` when available, while falling back to a manual
    integer tuple so grouping stays functional even if the dependency is absent.
    """
    normalized = _normalise_semver_text(version)
    if not normalized:
        return (0, version)

    if SemanticVersion is not None:
        try:
            parsed = SemanticVersion.coerce(normalized)
            return (2, parsed)
        except ValueError:
            logger.warning("Failed to coerce semantic version %s", version)

    core, _, prerelease = normalized.partition("-")
    parts = tuple(int(part) for part in core.split("."))
    return (1, (*parts, 0 if prerelease else 1, prerelease))


def _highest_fixed_version(
    pairs: List[Tuple[LocalizedIssue, FixPlan]],
) -> Tuple[Optional[str], Optional[FixPlan]]:
    """Return the highest fixed version and its originating plan from a bucket."""
    candidates = [
        (plan.fixed_version, plan)
        for _, plan in pairs
        if plan.fixed_version
    ]
    if not candidates:
        return None, None

    best_version, best_plan = max(
        candidates,
        key=lambda item: _semver_sort_key(item[0] or ""),
    )
    return best_version, best_plan


def _merge_workaround_snippets(
    pairs: List[Tuple[LocalizedIssue, FixPlan]],
) -> Optional[List[str]]:
    snippets: List[str] = []
    for _, plan in pairs:
        for snippet in plan.workaround_snippets or []:
            if snippet not in snippets:
                snippets.append(snippet)
    return snippets or None


def _build_group_fix_plan(
    pairs: List[Tuple[LocalizedIssue, FixPlan]],
) -> Optional[FixPlan]:
    """Create the unified group-level FixPlan for one SCA bucket."""
    if not pairs:
        return None

    strategy_bucket = _fix_strategy_bucket(pairs[0][1])
    exemplar_plan = pairs[0][1]

    if strategy_bucket == "UPDATE_VERSION":
        best_version, best_plan = _highest_fixed_version(pairs)
        plan_source = best_plan or exemplar_plan
        return plan_source.model_copy(
            update={
                "fixed_version": best_version,
                "workaround_snippets": None,
                "strategy_used": strategy_bucket,
            }
        )

    if strategy_bucket == "WORKAROUND":
        return exemplar_plan.model_copy(
            update={
                "fixed_version": None,
                "workaround_snippets": _merge_workaround_snippets(pairs),
                "strategy_used": strategy_bucket,
            }
        )

    return exemplar_plan.model_copy(
        update={
            "fixed_version": None,
            "workaround_snippets": None,
            "strategy_used": strategy_bucket,
        }
    )


# ---------------------------------------------------------------------------
# SCA grouping
# ---------------------------------------------------------------------------


def _group_sca(
    issue_plans: List[Tuple[LocalizedIssue, FixPlan]],
) -> Dict[str, VulnerabilityGroup]:
    """Return a dict of group_id → VulnerabilityGroup for SCA issue + plan pairs."""
    buckets: Dict[str, List[Tuple[LocalizedIssue, FixPlan]]] = defaultdict(list)
    for pair in issue_plans:
        localized_issue, fix_plan = pair
        key = _sca_key(localized_issue, fix_plan)
        buckets[key].append(pair)

    groups: Dict[str, VulnerabilityGroup] = {}
    for key, members in buckets.items():
        # Deduplicate CVE IDs, GHSA IDs, and versions
        seen_cves: list[str] = []
        seen_ghsas: list[str] = []
        seen_versions: list[str] = []
        seen_cve_set: set[str] = set()
        seen_ghsa_set: set[str] = set()
        seen_ver_set: set[str] = set()
        sources_set: set[IssueSource] = set()

        localized_members: List[LocalizedIssue] = []
        member_issues: List[VulnerabilityIssue] = []

        for localized_issue, _ in members:
            issue = localized_issue.issue
            localized_members.append(localized_issue)
            member_issues.append(issue)
            if issue.cve_id and issue.cve_id not in seen_cve_set:
                seen_cves.append(issue.cve_id)
                seen_cve_set.add(issue.cve_id)
            if issue.ghsa_id and issue.ghsa_id not in seen_ghsa_set:
                seen_ghsas.append(issue.ghsa_id)
                seen_ghsa_set.add(issue.ghsa_id)
            if issue.package_version and issue.package_version not in seen_ver_set:
                seen_versions.append(issue.package_version)
                seen_ver_set.add(issue.package_version)
            sources_set.add(issue.source)

        rep = _choose_representative_from_pairs(members)
        component = _component_from_issue(rep)
        group_fix_plan = _build_group_fix_plan(members)
        group_file_paths = _dedupe_paths(
            [
                localized_issue.manifest_file or localized_issue.issue.file_path
                for localized_issue in localized_members
            ]
        )
        group_file_path = group_file_paths[0] if group_file_paths else (localized_members[0].manifest_file or rep.file_path)

        groups[key] = VulnerabilityGroup(
            group_id=key,
            issue_type=IssueType.SCA,
            vulnerable_component=component,
            file_path=group_file_path,
            file_paths=group_file_paths,
            cve_ids=seen_cves,
            ghsa_ids=seen_ghsas,
            versions=seen_versions,
            sources=list(sources_set),
            representative_issue_id=rep.id,
            issues=member_issues,
            localized_issues=localized_members,
            fix_plan=group_fix_plan,
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
                file_paths=[issue.file_path] if issue.file_path else [],
                cve_ids=[],         # SAST findings rarely carry a CVE
                ghsa_ids=[],
                versions=[],
                sources=[issue.source],
                representative_issue_id=issue.id,
                issues=[issue],
            )

    return groups


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def group_issues(
    issues: List[VulnerabilityIssue],
    sca_issue_plans: Optional[List[Tuple[LocalizedIssue, FixPlan]]] = None,
) -> List[VulnerabilityGroup]:
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
    effective_sca_pairs = list(sca_issue_plans) if sca_issue_plans is not None else None
    if effective_sca_pairs is None:
        effective_sca_pairs = [
            (
                LocalizedIssue(
                    issue=issue,
                    manifest_file=issue.file_path,
                    localization_confidence=0.0,
                ),
                FixPlan(
                    status=FixPlanStatus.NO_FIX,
                    fixed_version=None,
                    workaround_snippets=None,
                    instruction="Compatibility fallback plan for pre-planned grouping.",
                    strategy_used="NO_FIX",
                ),
            )
            for issue in sca_issues
        ]
    else:
        planned_issue_ids = {localized.issue.id for localized, _ in effective_sca_pairs}
        for issue in sca_issues:
            if issue.id in planned_issue_ids:
                continue
            effective_sca_pairs.append(
                (
                    LocalizedIssue(
                        issue=issue,
                        manifest_file=issue.file_path,
                        localization_confidence=0.0,
                    ),
                    FixPlan(
                        status=FixPlanStatus.NO_FIX,
                        fixed_version=None,
                        workaround_snippets=None,
                        instruction="Compatibility fallback plan for missing pre-group plan.",
                        strategy_used="NO_FIX",
                    ),
                )
            )

    logger.debug(
        "Grouping %d issues (%d SCA, %d SAST).",
        len(issues),
        len(sca_issues),
        len(sast_issues),
    )

    all_groups: Dict[str, VulnerabilityGroup] = {}
    all_groups.update(_group_sca(effective_sca_pairs))
    all_groups.update(_group_sast(sast_issues))

    result = sorted(all_groups.values(), key=lambda g: g.group_id)
    logger.debug("Produced %d groups.", len(result))
    return result


def group_sca_issues(
    issues: List[VulnerabilityIssue],
    sca_issue_plans: Optional[List[Tuple[LocalizedIssue, FixPlan]]] = None,
) -> List[VulnerabilityGroup]:
    """
    Backwards-compatible alias that groups only SCA issues.

    Equivalent to calling ``group_issues`` on a pre-filtered SCA-only list.
    """
    sca_only = [i for i in issues if i.issue_type == IssueType.SCA]
    filtered_pairs = sca_issue_plans
    if filtered_pairs is not None:
        sca_issue_ids = {issue.id for issue in sca_only}
        filtered_pairs = [
            pair for pair in filtered_pairs
            if pair[0].issue.id in sca_issue_ids
        ]
    return group_issues(sca_only, sca_issue_plans=filtered_pairs)
