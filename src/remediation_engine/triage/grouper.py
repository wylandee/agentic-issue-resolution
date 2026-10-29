"""
grouper.py â€” Deterministic issue grouping for the triage layer.

Public API
----------
group_issues(issues)          â†’ List[VulnerabilityGroup]
    Primary entry point.  Groups both SAST and SCA issues.


Design
------
SCA grouping key: "sca:{manifest_file}:{package_name or purl}"
    * Multiple CVEs affecting the same package occurrence are grouped together;
      fix-plan evidence is populated later by Supervisor orchestration and does
      not affect group identity.
    * Duplicate CVE on the same component â†’ deduplicated.
    * Cross-tool duplicates (Semgrep SCA + ODC) with the same CVE + package +
      file â†’ merged into one group; sources list is unioned.
    * Representative issue: prefers one with fixed_version, then the one with
      the most non-None fields, then first-encountered.

SAST grouping key: "sast:{file_path}:{rule_id}:{line_start}-{line_end}"
    * Each distinct SAST location becomes a singleton group.
    * No enrichment is expected for pure SAST groups (no CVE IDs).
"""

from __future__ import annotations

import logging
from collections import defaultdict

from remediation_engine.contracts.schemas import (
    DependencyParentContext,
    IssueSource,
    IssueType,
    LocalizedIssue,
    VulnerabilityGroup,
    VulnerabilityIssue,
)
from remediation_engine.tools.package_identity import (
    normalize_python_package_name,
    package_name_from_purl,
)

logger = logging.getLogger(__name__)


def _dedupe_paths(paths: list[str | None]) -> list[str]:
    """Return stable, deduplicated non-empty repo-relative paths."""
    result: list[str] = []
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
    purl = issue.purl or ""
    is_pypi = (issue.ecosystem or "").strip().casefold() in {"python", "pypi"} or (
        purl.strip().casefold().startswith("pkg:pypi/")
    )
    component = issue.package_name or package_name_from_purl(purl) or "unknown"
    return normalize_python_package_name(component) if is_pypi else component


def _sca_key(localized_issue: LocalizedIssue) -> str:
    """Return the normalised package-occurrence key for an SCA issue."""
    issue = localized_issue.issue
    component = _component_from_issue(issue)
    file_part = localized_issue.manifest_file or issue.file_path or ""
    return f"sca:{file_part}:{component}"


def _parent_contexts(
    localized_issues: list[LocalizedIssue],
) -> list[DependencyParentContext]:
    """Return stable, deduplicated parent context for grouped SCA evidence."""
    contexts: dict[tuple[str, str, str], DependencyParentContext] = {}
    for localized in localized_issues:
        name = (localized.parent_package_name or "").strip()
        if not name:
            continue
        context = DependencyParentContext(
            package_name=name,
            package_version=localized.parent_package_version
            or localized.dependency_versions.get(name),
            declaration_type=localized.parent_declaration_type,
            manifest_file=localized.manifest_file,
            manifest_line=localized.parent_manifest_line,
            manifest_snippet=localized.parent_manifest_snippet,
            dependency_ancestry=list(localized.dependency_ancestry),
            dependency_versions=dict(localized.dependency_versions),
        )
        key = (
            context.package_name,
            context.declaration_type or "",
            context.manifest_file or "",
        )
        existing = contexts.get(key)
        if existing is None or (
            (context.package_version and not existing.package_version)
            or (context.manifest_line is not None and existing.manifest_line is None)
            or (context.manifest_snippet and not existing.manifest_snippet)
        ):
            contexts[key] = context
    return sorted(
        contexts.values(),
        key=lambda context: (
            context.package_name,
            context.declaration_type or "",
            context.package_version or "",
            context.manifest_file or "",
        ),
    )


def _merged_dependency_versions(localized_issues: list[LocalizedIssue]) -> dict[str, str]:
    """Merge resolved dependency versions without discarding populated evidence."""
    versions: dict[str, str] = {}
    for localized in localized_issues:
        for package_name, version in localized.dependency_versions.items():
            if package_name not in versions or not versions[package_name]:
                versions[package_name] = version
    return versions


def _sast_key(issue: VulnerabilityIssue) -> str:
    """Return the normalised grouping key for a SAST issue."""
    file_part = issue.file_path or ""
    rule_part = issue.rule_id or "unknown_rule"
    line_part = f"{issue.line_range.start}-{issue.line_range.end}" if issue.line_range else "0-0"
    return f"sast:{file_part}:{rule_part}:{line_part}"


# ---------------------------------------------------------------------------
# Representative-issue selection
# ---------------------------------------------------------------------------


def _field_richness(issue: VulnerabilityIssue) -> int:
    """Count non-None fields as a proxy for data richness."""
    return sum(1 for v in issue.model_dump().values() if v is not None and v != [] and v != {})


def _choose_representative(issues: list[VulnerabilityIssue]) -> VulnerabilityIssue:
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


def _group_sca(localized_issues: list[LocalizedIssue]) -> dict[str, VulnerabilityGroup]:
    """Return triage groups from localized SCA findings without fix plans."""
    buckets: dict[str, list[LocalizedIssue]] = defaultdict(list)
    for localized_issue in localized_issues:
        buckets[_sca_key(localized_issue)].append(localized_issue)

    groups: dict[str, VulnerabilityGroup] = {}
    for key, members in buckets.items():
        seen_cves: list[str] = []
        seen_ghsas: list[str] = []
        seen_versions: list[str] = []
        seen_cve_set: set[str] = set()
        seen_ghsa_set: set[str] = set()
        seen_ver_set: set[str] = set()
        sources_set: set[IssueSource] = set()
        member_issues: list[VulnerabilityIssue] = []

        for localized_issue in members:
            issue = localized_issue.issue
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

        representative = _choose_representative(member_issues)
        group_file_paths = _dedupe_paths(
            [localized.manifest_file or localized.issue.file_path for localized in members]
        )
        group_file_path = (
            group_file_paths[0]
            if group_file_paths
            else (members[0].manifest_file or representative.file_path)
        )
        parent_contexts = _parent_contexts(members)
        exemplar_localized = members[0] if members else None
        parent_localized = next(
            (
                localized
                for context in parent_contexts
                for localized in members
                if localized.parent_package_name == context.package_name
                and (
                    localized.parent_package_version
                    or localized.dependency_versions.get(localized.parent_package_name)
                    or ""
                )
                == (context.package_version or "")
                and (localized.parent_declaration_type or "") == (context.declaration_type or "")
                and (localized.manifest_file or "") == (context.manifest_file or "")
            ),
            None,
        )
        context_localized = parent_localized or exemplar_localized
        context_parent_name = context_localized.parent_package_name if context_localized else None
        context_parent_version = context_localized.parent_package_version or (
            context_localized.dependency_versions.get(context_parent_name)
            if context_localized and context_parent_name
            else None
        )

        groups[key] = VulnerabilityGroup(
            group_id=key,
            issue_type=IssueType.SCA,
            vulnerable_component=_component_from_issue(representative),
            file_path=group_file_path,
            file_paths=group_file_paths,
            cve_ids=seen_cves,
            ghsa_ids=seen_ghsas,
            versions=seen_versions,
            dependency_ancestry=(
                list(context_localized.dependency_ancestry) if context_localized else []
            ),
            dependency_versions=_merged_dependency_versions(members),
            parent_package_name=(
                context_localized.parent_package_name if context_localized else None
            ),
            parent_package_version=context_parent_version,
            parent_declaration_type=(
                context_localized.parent_declaration_type if context_localized else None
            ),
            parent_contexts=parent_contexts,
            sources=sorted(sources_set, key=lambda source: source.value),
            representative_issue_id=representative.id,
            issues=member_issues,
            localized_issues=list(members),
            fix_plan_candidates=[],
            fix_plan=None,
        )
    return groups


# ---------------------------------------------------------------------------
# SAST grouping
# ---------------------------------------------------------------------------


def _group_sast(issues: list[VulnerabilityIssue]) -> dict[str, VulnerabilityGroup]:
    """Return a dict of group_id â†’ VulnerabilityGroup for SAST issues."""
    groups: dict[str, VulnerabilityGroup] = {}
    for issue in issues:
        key = _sast_key(issue)
        if key in groups:
            # Same SAST location seen twice (e.g. duplicate rule run) â€” merge sources
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
                cve_ids=[],  # SAST findings rarely carry a CVE
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
    issues: list[VulnerabilityIssue],
    localized_sca_issues: list[LocalizedIssue] | None = None,
) -> list[VulnerabilityGroup]:
    """Group findings, retaining SCA localization and deferring fix planning.

    Args:
        issues: Scanner findings from any supported source.
        localized_sca_issues: Optional localization results keyed by issue ID.
            Missing results receive a conservative issue-path fallback.

    Returns:
        Groups ordered by deterministic group ID. SCA plan fields remain empty.
    """
    if not issues:
        return []

    sca_issues = [issue for issue in issues if issue.issue_type == IssueType.SCA]
    sast_issues = [issue for issue in issues if issue.issue_type == IssueType.SAST]
    localized_by_id = {localized.issue.id: localized for localized in (localized_sca_issues or [])}
    effective_localized = [
        localized_by_id.get(issue.id)
        or LocalizedIssue(issue=issue, manifest_file=issue.file_path, localization_confidence=0.0)
        for issue in sca_issues
    ]

    logger.debug(
        "Grouping %d issues (%d SCA, %d SAST).",
        len(issues),
        len(sca_issues),
        len(sast_issues),
    )
    all_groups: dict[str, VulnerabilityGroup] = {}
    all_groups.update(_group_sca(effective_localized))
    all_groups.update(_group_sast(sast_issues))
    result = sorted(all_groups.values(), key=lambda group: group.group_id)
    logger.debug("Produced %d groups.", len(result))
    return result
