"""
pipeline.py — End-to-end triage pipeline for the Phase 4.0 triage layer.

Public API
----------
run_triage_pipeline(issues, system_context, repo_root=None)
    → List[Tuple[VulnerabilityGroup, TriageResult]]

    Full pipeline:
    1. group_issues(issues)         → List[VulnerabilityGroup]
    2. enrich_cves(all_cve_ids)     → Dict[str, CVEEnrichment]
    3. Attach enrichment to groups  (primary CVE enrichment per group)
    4. run_triage(group, context)   → TriageResult per group
    5. Return all (group, result) pairs

select_issues_for_remediation(results)
    → List[VulnerabilityIssue]

    Filter valid groups, pick the recommended issue per group, sort by
    revised priority (CRITICAL → HIGH → MEDIUM → LOW → INFO → UNKNOWN).
    Safe to pass each returned issue directly into run_remediation().
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.contracts.schemas import (
    CVEEnrichment,
    Severity,
    SystemContext,
    TriageResult,
    VulnerabilityGroup,
    VulnerabilityIssue,
)
from src.triage.agent import run_triage
from src.triage.enrichment import enrich_cves
from src.triage.grouper import group_issues
from src.triage.reachability import analyze_reachability

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Priority sort order
# ---------------------------------------------------------------------------

_PRIORITY_ORDER: Dict[Severity, int] = {
    Severity.CRITICAL: 0,
    Severity.HIGH: 1,
    Severity.MEDIUM: 2,
    Severity.LOW: 3,
    Severity.INFO: 4,
    Severity.UNKNOWN: 5,
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _attach_enrichment(
    groups: List[VulnerabilityGroup],
    enrichment_map: Dict[str, CVEEnrichment],
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
    groups: List[VulnerabilityGroup],
    issue_id,
) -> Optional[VulnerabilityIssue]:
    """Locate a VulnerabilityIssue by UUID across all group members."""
    for group in groups:
        for issue in group.issues:
            if issue.id == issue_id:
                return issue
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_triage_pipeline(
    issues: List[VulnerabilityIssue],
    system_context: SystemContext,
    repo_root: Optional[str] = None,
) -> List[Tuple[VulnerabilityGroup, TriageResult]]:
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
        One pair per group.  Includes *all* groups — valid and invalid.
        Use ``select_issues_for_remediation`` to filter to actionable ones.
    """
    if not issues:
        logger.info("Triage pipeline: no issues provided, returning empty.")
        return []

    logger.info("Triage pipeline: processing %d issues.", len(issues))

    # Step 1: Group
    groups = group_issues(issues)
    logger.info("Triage pipeline: produced %d groups.", len(groups))

    # Step 2: Collect all unique CVE IDs for bulk enrichment
    all_cve_ids: List[str] = []
    seen: set = set()
    for group in groups:
        for cve in group.cve_ids:
            if cve not in seen:
                all_cve_ids.append(cve)
                seen.add(cve)

    # Step 3: Enrich CVEs (failure-safe)
    enrichment_map: Dict[str, CVEEnrichment] = {}
    if all_cve_ids:
        enrichment_map = enrich_cves(all_cve_ids)
        logger.info(
            "Triage pipeline: enriched %d/%d CVEs.",
            len(enrichment_map),
            len(all_cve_ids),
        )

    # Step 4: Attach enrichment to groups
    _attach_enrichment(groups, enrichment_map)

    # Step 5: Reachability analysis for SCA groups (failure-safe)
    if repo_root and Path(repo_root).exists():
        analyze_reachability(groups, repo_root)

    # Step 6: Triage each group
    results: List[Tuple[VulnerabilityGroup, TriageResult]] = []
    for group in groups:
        try:
            triage_result = run_triage(group, system_context)
            results.append((group, triage_result))
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Triage failed for group %s (%s); skipping.",
                group.group_id,
                exc,
            )

    valid_count = sum(1 for _, r in results if r.is_valid)
    logger.info(
        "Triage pipeline: %d/%d groups are valid.", valid_count, len(results)
    )
    return results


def select_issues_for_remediation(
    results: List[Tuple[VulnerabilityGroup, TriageResult]],
) -> List[VulnerabilityIssue]:
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
    selected: List[Tuple[Severity, VulnerabilityIssue]] = []

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

        if issue is not None:
            selected.append((triage.revised_priority, issue))

    # Sort by priority (lower _PRIORITY_ORDER value = higher urgency)
    selected.sort(key=lambda t: _PRIORITY_ORDER.get(t[0], 99))
    return [issue for _, issue in selected]
