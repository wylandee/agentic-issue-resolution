"""Pure helpers for plan-bearing group fixtures in orchestration tests."""

from __future__ import annotations

from collections.abc import Iterable

from remediation_engine.contracts.schemas import FixPlan, LocalizedIssue, VulnerabilityIssue
from remediation_engine.orchestration.supervisor_fix_plans import apply_issue_fix_plans_to_group
from remediation_engine.triage.grouper import group_issues


def group_with_fix_plans(
    issues: list[VulnerabilityIssue],
    *,
    plans_by_finding: Iterable[tuple[LocalizedIssue, FixPlan]],
):
    """Build plan-bearing groups without invoking advisory or registry services."""
    pairs = list(plans_by_finding)
    groups = group_issues(
        issues,
        localized_sca_issues=[localized for localized, _ in pairs],
    )
    plans_by_issue = {localized.issue.id: (localized, plan) for localized, plan in pairs}
    result = []
    for group in groups:
        group_pairs = [
            plans_by_issue[issue.id] for issue in group.issues if issue.id in plans_by_issue
        ]
        result.append(apply_issue_fix_plans_to_group(group, group_pairs) if group_pairs else group)
    return result
