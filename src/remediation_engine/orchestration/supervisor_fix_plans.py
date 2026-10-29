"""Supervisor-owned advisory fix planning for actionable vulnerability groups."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Iterable
from typing import Any

try:
    from semantic_version import Version as SemanticVersion
except ImportError:  # pragma: no cover - optional runtime fallback
    SemanticVersion = None

from remediation_engine.contracts.schemas import (
    FixPlan,
    FixPlanStatus,
    IssueType,
    LocalizedIssue,
    PackageFixPlanCandidate,
    VulnerabilityGroup,
)
from remediation_engine.orchestration.trajectory_exporter import invoke_with_trajectory
from remediation_engine.tools.fix_planner import plan_fix

logger = logging.getLogger(__name__)


def _normalise_semver_text(version: str) -> str | None:
    """Extract a semver-like token and coerce partial versions for comparison."""
    match = re.search(
        r"(?i)v?\d+(?:\.\d+){0,2}(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?",
        version.strip(),
    )
    if not match:
        return None
    cleaned = match.group(0).lstrip("vV").split("+", 1)[0]
    core, separator, suffix = cleaned.partition("-")
    parts = core.split(".")
    while len(parts) < 3:
        parts.append("0")
    normalized = ".".join(parts[:3])
    return f"{normalized}-{suffix}" if separator else normalized


def _semver_sort_key(version: str) -> tuple[int, object]:
    """Return the existing deterministic comparison key for fixed versions."""
    normalized = _normalise_semver_text(version)
    if not normalized:
        return (0, version)
    if SemanticVersion is not None:
        try:
            return (2, SemanticVersion.coerce(normalized))
        except ValueError:
            logger.warning("Failed to coerce semantic version %s", version)
    core, _, prerelease = normalized.partition("-")
    parts = tuple(int(part) for part in core.split("."))
    return (1, (*parts, 0 if prerelease else 1, prerelease))


def _highest_fixed_version(pairs: list[tuple[LocalizedIssue, FixPlan]]) -> FixPlan | None:
    """Return the plan carrying the highest fixed version in the group."""
    candidates = [plan for _, plan in pairs if plan.fixed_version]
    return max(
        candidates, key=lambda plan: _semver_sort_key(plan.fixed_version or ""), default=None
    )


def _merge_workaround_snippets(pairs: list[tuple[LocalizedIssue, FixPlan]]) -> list[str] | None:
    """Merge workaround snippets in stable first-seen order."""
    snippets: list[str] = []
    for _, plan in pairs:
        for snippet in plan.workaround_snippets or []:
            if snippet not in snippets:
                snippets.append(snippet)
    return snippets or None


def _build_group_fix_plan(pairs: list[tuple[LocalizedIssue, FixPlan]]) -> FixPlan | None:
    """Build the legacy group summary while retaining all member candidates."""
    if not pairs:
        return None
    version_pairs = [pair for pair in pairs if pair[1].status == FixPlanStatus.VERSION_FOUND]
    workaround_pairs = [pair for pair in pairs if pair[1].status == FixPlanStatus.WORKAROUND_FOUND]
    if version_pairs:
        source = _highest_fixed_version(version_pairs) or version_pairs[0][1]
        return source.model_copy(
            update={"workaround_snippets": None, "strategy_used": "UPDATE_VERSION"}
        )
    if workaround_pairs:
        source = workaround_pairs[0][1]
        return source.model_copy(
            update={
                "fixed_version": None,
                "workaround_snippets": _merge_workaround_snippets(workaround_pairs),
                "strategy_used": "WORKAROUND",
            }
        )
    return pairs[0][1].model_copy(
        update={"fixed_version": None, "workaround_snippets": None, "strategy_used": "NO_FIX"}
    )


def _build_fix_plan_candidates(
    pairs: list[tuple[LocalizedIssue, FixPlan]],
) -> list[PackageFixPlanCandidate]:
    """Retain unique issue-level plans in deterministic order."""
    candidates: dict[tuple[str, str], PackageFixPlanCandidate] = {}
    for localized, plan in pairs:
        issue_id = localized.issue.id
        key = (str(issue_id), plan.model_dump_json())
        candidates.setdefault(key, PackageFixPlanCandidate(issue_id=issue_id, plan=plan))
    return [candidates[key] for key in sorted(candidates)]


def apply_issue_fix_plans_to_group(
    group: VulnerabilityGroup,
    pairs: list[tuple[LocalizedIssue, FixPlan]],
) -> VulnerabilityGroup:
    """Return a group copy with aggregated per-finding plan evidence.

    Args:
        group: Original package group. It is never mutated.
        pairs: Localized findings and their typed advisory results.

    Returns:
        A copied group containing one candidate per unique finding/result and
        the compatibility summary plan, or empty plan fields when ``pairs`` is
        empty.
    """
    return group.model_copy(
        update={
            "fix_plan_candidates": _build_fix_plan_candidates(pairs),
            "fix_plan": _build_group_fix_plan(pairs),
        }
    )


def has_workaround_fix_plan(group: VulnerabilityGroup) -> bool:
    """Return whether a group has workaround evidence in candidates or summary.

    Args:
        group: SCA group to inspect.

    Returns:
        Whether at least one candidate is a workaround, or the legacy summary
        plan is a workaround when no candidates were serialized.
    """
    candidates = group.fix_plan_candidates or []
    if candidates:
        return any(
            candidate.plan.status == FixPlanStatus.WORKAROUND_FOUND for candidate in candidates
        )
    return bool(
        group.fix_plan is not None and group.fix_plan.status == FixPlanStatus.WORKAROUND_FOUND
    )


def _fallback_localized_issue(group: VulnerabilityGroup, issue: Any) -> LocalizedIssue:
    """Build a conservative localization when older serialized groups lack one."""
    ecosystem = str(getattr(issue, "ecosystem", "") or "").strip().casefold()
    purl = str(getattr(issue, "purl", "") or "").strip().casefold()
    is_maven = ecosystem == "maven" or purl.startswith("pkg:maven/")
    return LocalizedIssue(
        issue=issue,
        manifest_file=group.file_path or issue.file_path,
        package_manager="maven" if is_maven else None,
        localization_confidence=0.0,
    )


def _lookup_key(localized: LocalizedIssue, *, skip_osv: bool) -> str:
    """Return a key for every input that can change the planner result."""
    issue = localized.issue
    payload = {
        "mode": "workaround" if skip_osv else "default",
        "package_manager": localized.package_manager,
        "package_name": issue.package_name,
        "purl": issue.purl,
        "ecosystem": issue.ecosystem,
        "package_version": issue.package_version,
        "cve_id": issue.cve_id,
        "ghsa_id": issue.ghsa_id,
        "rule_id": issue.rule_id,
        "manifest_file": localized.manifest_file,
        "manifest_line": localized.manifest_line,
        "is_direct_dependency": localized.is_direct_dependency,
        "declaration_type": localized.declaration_type,
        "version_property_name": localized.version_property_name,
        "version_property_file": localized.version_property_file,
        "dependency_ancestry": list(localized.dependency_ancestry),
        "dependency_versions": dict(sorted(localized.dependency_versions.items())),
        "parent_package_name": localized.parent_package_name,
        "parent_package_version": localized.parent_package_version,
        "parent_declaration_type": localized.parent_declaration_type,
        "parent_manifest_line": localized.parent_manifest_line,
        "parent_manifest_snippet": localized.parent_manifest_snippet,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _run_plan(
    localized: LocalizedIssue,
    *,
    skip_osv: bool,
    cache: dict[str, FixPlan],
    lookup: Any,
) -> FixPlan:
    """Run or reuse one typed advisory lookup, converting failures to NO_FIX."""
    key = _lookup_key(localized, skip_osv=skip_osv)
    if key in cache:
        return cache[key]
    try:
        invoke = (
            (lambda: lookup(localized, skip_osv=True)) if skip_osv else (lambda: lookup(localized))
        )
        raw = invoke_with_trajectory("supervisor.plan_fix", invoke, localized)
        plan = raw if isinstance(raw, FixPlan) else FixPlan.model_validate(raw)
    except Exception as exc:  # noqa: BLE001 - external planner failures fail closed per finding.
        logger.warning("Supervisor fix planning failed for %s: %s", localized.issue.id, exc)
        plan = FixPlan(
            status=FixPlanStatus.NO_FIX,
            fixed_version=None,
            workaround_snippets=None,
            instruction="No upstream patch or workaround was found. Inform the user.",
            strategy_used="planner_error",
        )
    cache[key] = plan
    return plan


def ensure_group_fix_plans(
    group: VulnerabilityGroup,
    *,
    lookup_cache: dict[str, FixPlan] | None = None,
    lookup: Any | None = None,
) -> VulnerabilityGroup:
    """Materialize missing advisory plans for one actionable SCA group.

    Args:
        group: SCA group selected for Supervisor planning.
        lookup_cache: Optional operation-scoped cache shared across groups.
        lookup: Optional planner callable used by deterministic tests.

    Returns:
        The original group when it is out of scope or already planned; otherwise
        a copy containing member candidates and an aggregate compatibility plan.
    """
    if group.issue_type != IssueType.SCA or group.is_synthetic:
        return group
    if group.fix_plan is not None or group.fix_plan_candidates:
        return group
    localized_by_id = {item.issue.id: item for item in group.localized_issues or []}
    localized_issues = [
        localized_by_id.get(issue.id) or _fallback_localized_issue(group, issue)
        for issue in group.issues
    ]
    cache = lookup_cache if lookup_cache is not None else {}
    planner = lookup or plan_fix
    pairs = [
        (localized, _run_plan(localized, skip_osv=False, cache=cache, lookup=planner))
        for localized in localized_issues
    ]
    return apply_issue_fix_plans_to_group(group, pairs)


def plan_fixes_for_groups(
    groups: Iterable[VulnerabilityGroup],
    *,
    lookup: Any | None = None,
) -> list[VulnerabilityGroup]:
    """Materialize missing plans with one deterministic cache per batch.

    Args:
        groups: Actionable SCA and non-SCA groups selected by triage.
        lookup: Optional deterministic planner override for tests.

    Returns:
        A list of groups, preserving already-planned and out-of-scope group
        objects and copying groups that receive new plans.
    """
    cache: dict[str, FixPlan] = {}
    return [ensure_group_fix_plans(group, lookup_cache=cache, lookup=lookup) for group in groups]


def ensure_workaround_fix_plan(
    group: VulnerabilityGroup,
    *,
    lookup_cache: dict[str, FixPlan] | None = None,
    lookup: Any | None = None,
) -> VulnerabilityGroup:
    """Append targeted workaround evidence while retaining version plans.

    Args:
        group: SCA group for an accepted workaround transition.
        lookup_cache: Optional operation-scoped cache shared across groups.
        lookup: Optional planner callable used by deterministic tests.

    Returns:
        The original group when out of scope or when no workaround is found;
        otherwise a copied group with the new workaround candidates appended.
    """
    if group.issue_type != IssueType.SCA or group.is_synthetic:
        return group
    localized_by_id = {item.issue.id: item for item in group.localized_issues or []}
    candidates = list(group.fix_plan_candidates or [])
    if not candidates and has_workaround_fix_plan(group):
        return group
    workaround_issue_ids = {
        item.issue_id for item in candidates if item.plan.status == FixPlanStatus.WORKAROUND_FOUND
    }
    cache = lookup_cache if lookup_cache is not None else {}
    planner = lookup or plan_fix
    additions: list[PackageFixPlanCandidate] = []
    for issue in group.issues:
        if issue.id in workaround_issue_ids:
            continue
        localized = localized_by_id.get(issue.id) or _fallback_localized_issue(group, issue)
        plan = _run_plan(localized, skip_osv=True, cache=cache, lookup=planner)
        if plan.status == FixPlanStatus.WORKAROUND_FOUND:
            additions.append(PackageFixPlanCandidate(issue_id=issue.id, plan=plan))
    if not additions:
        return group
    candidates.extend(additions)
    candidate_pairs = [
        (
            localized_by_id.get(issue.id) or _fallback_localized_issue(group, issue),
            candidate.plan,
        )
        for issue in group.issues
        for candidate in candidates
        if candidate.issue_id == issue.id
    ]
    return group.model_copy(
        update={"fix_plan_candidates": _build_fix_plan_candidates(candidate_pairs)}
    )


def workaround_lookup_fingerprint(group: VulnerabilityGroup) -> str:
    """Return a stable fingerprint for targeted workaround lookup inputs.

    Args:
        group: SCA group whose issue-level planner inputs should be identified.

    Returns:
        A SHA-256 digest over sorted, normalized issue lookup keys.
    """
    localized_by_id = {item.issue.id: item for item in group.localized_issues or []}
    inputs = [
        _lookup_key(
            localized_by_id.get(issue.id) or _fallback_localized_issue(group, issue),
            skip_osv=True,
        )
        for issue in group.issues
    ]
    serialized = json.dumps(sorted(inputs), separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def workaround_lookup_ledger_key(group: VulnerabilityGroup, fingerprint: str) -> str:
    """Return an operation-ledger key for one group's workaround inputs.

    Args:
        group: Group identity owning the targeted lookup.
        fingerprint: Digest for the exact group lookup inputs.

    Returns:
        A stable key that remembers repeated or reverted input fingerprints.
    """
    return f"{group.group_id}:{fingerprint}"


def prepare_supervisor_fix_plan_state(
    groups: Iterable[VulnerabilityGroup],
    task_queue: dict[str, Any],
    task_ids_requiring_refresh: Iterable[str],
    project_language: Any,
    *,
    lookup: Any | None = None,
) -> tuple[list[VulnerabilityGroup], dict[str, Any], list[str]]:
    """Plan groups and refresh reopened tasks before portfolio derivation.

    Planning remains outside task construction. Reopened task identity,
    ancestry, attempts, and revision are kept while strategy-derived inputs
    are rebuilt against the newly planned group.

    Args:
        groups: Current triage groups.
        task_queue: Existing Supervisor task projection.
        task_ids_requiring_refresh: Reopened tasks whose planning fields need
            to be rebuilt against current group evidence.
        project_language: Resolved repository language used for task policy.
        lookup: Optional deterministic planner override for tests.

    Returns:
        Planned group copies, a copied/refreshed task queue, and refresh IDs
        that could not be processed because their task or group was absent.
    """
    from remediation_engine.orchestration.task_utils import build_initial_remediation_task

    planned_groups = plan_fixes_for_groups(groups, lookup=lookup)
    groups_by_id = {group.group_id: group for group in planned_groups}
    refreshed_queue = {task_id: task.model_copy(deep=True) for task_id, task in task_queue.items()}
    pending_refresh_ids: list[str] = []
    for task_id in sorted(set(task_ids_requiring_refresh)):
        task = refreshed_queue.get(task_id)
        group = groups_by_id.get(task.parent_group_id) if task is not None else None
        if task is None or group is None:
            pending_refresh_ids.append(task_id)
            continue
        fresh = build_initial_remediation_task(
            group,
            task.task_id,
            project_language=project_language,
        )
        refreshed_queue[task_id] = task.model_copy(
            update={
                "strategy": fresh.strategy,
                "strategy_stage": fresh.strategy_stage,
                "qa_policy": fresh.qa_policy,
                "no_fix_stage": fresh.no_fix_stage,
                "instruction": fresh.instruction,
                "selected_plan_issue_ids": list(fresh.selected_plan_issue_ids),
                "selected_version": None,
                "target_package_name": fresh.target_package_name,
                "target_dependency_type": fresh.target_dependency_type,
                "parent_package_name": fresh.parent_package_name,
                "parent_package_version": fresh.parent_package_version,
                "parent_minimum_version": None,
                "exhausted_update_path": False,
            }
        )
    return planned_groups, refreshed_queue, pending_refresh_ids
