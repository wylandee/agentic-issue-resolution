"""
pipeline.py â€” End-to-end triage pipeline for scanner findings.

Public API
----------
run_triage_pipeline(issues, system_context, repo_root=None)
    â†’ List[Tuple[VulnerabilityGroup, TriageResult]]

    Full pipeline:
    1. locate + plan each SCA issue  â†’ List[Tuple[LocalizedIssue, FixPlan]]
    2. group_issues(...)             â†’ List[VulnerabilityGroup]
    3. enrich_cves(all_cve_ids)      â†’ Dict[str, CVEEnrichment]
    4. Attach enrichment to groups   (primary CVE enrichment per group)
    5. Optional reachability analysis for SCA groups
    6. run_triage(group, context)    â†’ TriageResult per group
    7. Return all (group, result) pairs

select_issues_for_remediation(results)
    â†’ List[VulnerabilityIssue]

    Filter valid groups, pick the recommended issue per group, sort by
    revised priority (CRITICAL â†’ HIGH â†’ MEDIUM â†’ LOW â†’ INFO â†’ UNKNOWN).
    Safe to pass each returned issue directly into run_remediation().
"""

from __future__ import annotations

import logging
from pathlib import Path

from langsmith import traceable

from remediation_engine.contracts.schemas import (
    CVEEnrichment,
    FixPlan,
    FixPlanStatus,
    IssueType,
    LocalizedIssue,
    Severity,
    SystemContext,
    TriageResult,
    VulnerabilityGroup,
    VulnerabilityIssue,
)
from remediation_engine.settings import AppSettings
from remediation_engine.triage.agent import run_triage
from remediation_engine.triage.enrichment import enrich_cves
from remediation_engine.triage.grouper import group_issues
from remediation_engine.triage.reachability import analyze_reachability

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Priority sort order
# ---------------------------------------------------------------------------

_PRIORITY_ORDER: dict[Severity, int] = {
    Severity.CRITICAL: 0,
    Severity.HIGH: 1,
    Severity.MEDIUM: 2,
    Severity.LOW: 3,
    Severity.INFO: 4,
    Severity.UNKNOWN: 5,
}


class TriagePipelineError(RuntimeError):
    """Raised when one or more groups cannot receive a triage outcome."""

    def __init__(self, failed_group_ids: list[str], cause: BaseException) -> None:
        """Store failed group IDs and the first underlying triage exception."""
        self.failed_group_ids = tuple(failed_group_ids)
        self.cause = cause
        groups = ", ".join(self.failed_group_ids)
        super().__init__(f"triage failed for groups [{groups}]: {cause}")


class TriageSelectionError(RuntimeError):
    """Raised when a valid triage result cannot resolve an issue to execute."""

    def __init__(self, group_id: str) -> None:
        """Store the group whose valid result was internally inconsistent."""
        self.group_id = group_id
        super().__init__(
            f"triage result for group {group_id!r} has no resolvable recommended issue"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _attach_enrichment(
    groups: list[VulnerabilityGroup],
    enrichment_map: dict[str, CVEEnrichment],
) -> None:
    """
    Mutate groups in-place: attach the enrichment for the primary CVE.

    If a group has multiple CVEs, the one with the highest EPSS score (or the
    first KEV entry) is chosen as the primary enrichment.
    """
    for group in groups:
        if not group.cve_ids:
            continue

        candidates = [enrichment_map[c] for c in group.cve_ids if c in enrichment_map]
        if not candidates:
            continue

        # Prefer KEV; then highest EPSS; then first
        kev_hits = [c for c in candidates if c.in_kev]
        if kev_hits:
            group.enrichment = kev_hits[0]
        else:
            group.enrichment = max(candidates, key=lambda c: c.epss)


def _find_issue_by_id(
    groups: list[VulnerabilityGroup],
    issue_id,
) -> VulnerabilityIssue | None:
    """Locate a VulnerabilityIssue by UUID across all group members."""
    for group in groups:
        for issue in group.issues:
            if issue.id == issue_id:
                return issue
    return None


def _fallback_localized_issue(issue: VulnerabilityIssue) -> LocalizedIssue:
    """Build a minimal LocalizedIssue when repository localization is unavailable."""
    return LocalizedIssue(
        issue=issue,
        manifest_file=issue.file_path,
        localization_confidence=0.0,
    )


def _fallback_no_fix_plan() -> FixPlan:
    """Return a safe no-fix plan when planning fails unexpectedly."""
    return FixPlan(
        status=FixPlanStatus.NO_FIX,
        fixed_version=None,
        workaround_snippets=None,
        instruction="No upstream patch or workaround was found. Inform the user.",
        strategy_used="NO_FIX",
    )


@traceable(name="triage.grouping", run_type="chain")
def _run_grouping(
    issues: list[VulnerabilityIssue],
    sca_issue_plans: list[tuple[LocalizedIssue, FixPlan]],
) -> list[VulnerabilityGroup]:
    """Group normalized findings into remediation units."""
    return group_issues(issues, sca_issue_plans=sca_issue_plans)


@traceable(name="triage.cve_enrichment", run_type="chain")
def _run_enrichment(
    cve_ids: list[str],
    settings: AppSettings | None = None,
) -> dict[str, CVEEnrichment]:
    """Fetch threat-intelligence enrichment for the pipeline's CVE set."""
    if settings is None:
        return enrich_cves(cve_ids)
    return enrich_cves(cve_ids, settings=settings)


@traceable(name="triage.reachability", run_type="chain")
def _run_reachability(groups: list[VulnerabilityGroup], repo_root: str) -> None:
    """Annotate SCA groups with repository reachability evidence."""
    analyze_reachability(groups, repo_root)


@traceable(name="triage.sca_localization_and_planning", run_type="chain")
def _prepare_sca_issue_plans(
    sca_issues: list[VulnerabilityIssue],
    repo_root: str | None,
) -> list[tuple[LocalizedIssue, FixPlan]]:
    """Locate and plan SCA issues before grouping."""
    if not sca_issues:
        return []

    from remediation_engine.tools.fix_planner import plan_fix
    from remediation_engine.tools.manifest_locator import locate_from_issue

    repo_path = Path(repo_root) if repo_root and Path(repo_root).exists() else None
    issue_plans: list[tuple[LocalizedIssue, FixPlan]] = []

    for issue in sca_issues:
        localized_issue = _fallback_localized_issue(issue)
        if repo_path is not None:
            try:
                localized_issue = locate_from_issue(issue, repo_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "SCA localization failed for %s (%s); using fallback localization.",
                    issue.id,
                    exc,
                )

        try:
            fix_plan = FixPlan(**plan_fix(localized_issue))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Fix planning failed for %s (%s); using no-fix fallback.",
                issue.id,
                exc,
            )
            fix_plan = _fallback_no_fix_plan()

        issue_plans.append((localized_issue, fix_plan))

    return issue_plans


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@traceable(name="triage.pipeline", run_type="chain")
def run_triage_pipeline(
    issues: list[VulnerabilityIssue],
    system_context: SystemContext,
    repo_root: str | None = None,
    settings: AppSettings | None = None,
) -> list[tuple[VulnerabilityGroup, TriageResult]]:
    """
    Run the full triage pipeline on a flat list of ``VulnerabilityIssue`` records.

    Parameters
    ----------
    issues:
        All issues from any combination of scanners.
    system_context:
        Caller-supplied scan session metadata.
    repo_root:
        Optional absolute path to the repository. When present and it exists,
        SCA reachability analysis is run before triage.

    Returns
    -------
    List[Tuple[VulnerabilityGroup, TriageResult]]
        One pair per group.  Includes *all* groups â€” valid and invalid.
        Use ``select_issues_for_remediation`` to filter to actionable ones.
    """
    if not issues:
        logger.info("Triage pipeline: no issues provided, returning empty.")
        return []

    logger.info("Triage pipeline: processing %d issues.", len(issues))

    sca_issues = [issue for issue in issues if issue.issue_type == IssueType.SCA]
    sca_issue_plans = _prepare_sca_issue_plans(sca_issues, repo_root)

    # Step 1: Group after SCA locate + plan
    groups = _run_grouping(issues, sca_issue_plans)
    logger.info("Triage pipeline: produced %d groups.", len(groups))

    # Step 2: Collect all unique CVE IDs for bulk enrichment
    all_cve_ids: list[str] = []
    seen: set = set()
    for group in groups:
        for cve in group.cve_ids:
            if cve not in seen:
                all_cve_ids.append(cve)
                seen.add(cve)

    # Step 3: Enrich CVEs (failure-safe)
    enrichment_map: dict[str, CVEEnrichment] = {}
    if all_cve_ids:
        enrichment_map = _run_enrichment(all_cve_ids, settings=settings)
        logger.info(
            "Triage pipeline: enriched %d/%d CVEs.",
            len(enrichment_map),
            len(all_cve_ids),
        )

    # Step 4: Attach enrichment to groups
    _attach_enrichment(groups, enrichment_map)

    # Step 5: Reachability analysis for SCA groups (failure-safe)
    if repo_root and Path(repo_root).exists():
        _run_reachability(groups, repo_root)

    # Step 6: Triage each group
    results: list[tuple[VulnerabilityGroup, TriageResult]] = []
    failed_group_ids: list[str] = []
    first_error: BaseException | None = None
    for group in groups:
        try:
            triage_result = (
                run_triage(group, system_context)
                if settings is None
                else run_triage(group, system_context, settings=settings)
            )
            results.append((group, triage_result))
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Triage failed for group %s (%s); aborting result construction.",
                group.group_id,
                exc,
            )
            failed_group_ids.append(group.group_id)
            first_error = first_error or exc

    if failed_group_ids and first_error is not None:
        raise TriagePipelineError(failed_group_ids, first_error) from first_error

    valid_count = sum(1 for _, r in results if r.is_valid)
    logger.info("Triage pipeline: %d/%d groups are valid.", valid_count, len(groups))
    return results


def select_issues_for_remediation(
    results: list[tuple[VulnerabilityGroup, TriageResult]],
) -> list[VulnerabilityIssue]:
    """
    Select one representative ``VulnerabilityIssue`` per valid group.

    Invalid (false-positive) groups are excluded.  The remaining issues are
    sorted by revised priority (CRITICAL first).

    Parameters
    ----------
    results:
        Output of ``run_triage_pipeline``.

    Returns
    -------
    List[VulnerabilityIssue]
        Ready to pass one-by-one to ``run_remediation`` in the existing graph.
    """
    selected: list[tuple[Severity, VulnerabilityIssue]] = []

    for group, triage in results:
        if not triage.is_valid:
            logger.debug(
                "Skipping group %s (false positive: %s).",
                group.group_id,
                triage.false_positive_reason,
            )
            continue

        # Find the recommended issue among group members
        issue = _find_issue_by_id([group], triage.recommended_issue_id)
        if issue is None:
            # Fallback: use the representative issue
            issue = _find_issue_by_id([group], group.representative_issue_id)
        if issue is None and group.issues:
            issue = group.issues[0]

        if issue is None:
            raise TriageSelectionError(group.group_id)
        selected.append((triage.revised_priority, issue))

    # Sort by priority (lower _PRIORITY_ORDER value = higher urgency)
    selected.sort(key=lambda t: _PRIORITY_ORDER.get(t[0], 99))
    return [issue for _, issue in selected]
