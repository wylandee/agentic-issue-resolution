"""
task_utils.py - Utility helpers for Phase 5 task queue management.

Provides pure functions for creating and managing ``RemediationTask`` objects
from ``VulnerabilityGroup`` records.

Public API
----------
derive_initial_strategy(group) -> RoutingStrategy
    Pure function: decides VERSION_BUMP vs CODE_WORKAROUND from the fix plan.
build_initial_remediation_task(group, task_id) -> RemediationTask
    Factory: creates a Depth-0 task from a vulnerability group.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

try:
    from semantic_version import Version as SemanticVersion
except ImportError:  # pragma: no cover - optional runtime fallback
    SemanticVersion = None

from remediation_engine.contracts.accessors import model_or_dict_value
from remediation_engine.contracts.schemas import (
    FixPlan,
    FixPlanStatus,
    NoFixMitigationStage,
    QAEvaluation,
    QAPolicy,
    RemediationTask,
    RoutingStrategy,
    SCARemediationStage,
    TaskStatus,
    VulnerabilityGroup,
)

TERMINAL_TASK_STATUSES = frozenset(
    {
        TaskStatus.QA_PASSED,
        TaskStatus.UNFIXABLE,
        TaskStatus.INCONCLUSIVE,
        TaskStatus.PIVOTED,
    }
)

_UNRESOLVED_GROUP_STATUSES = frozenset(
    {
        TaskStatus.UNFIXABLE.value,
        TaskStatus.INCONCLUSIVE.value,
        TaskStatus.NEEDS_RETRY.value,
        TaskStatus.PENDING.value,
        TaskStatus.OPTIMISTICALLY_FIXED.value,
    }
)


_task_value = model_or_dict_value


@dataclass(frozen=True)
class PackageFixPlanSelection:
    """Supervisor-owned selection from plans retained by one package group."""

    plan: FixPlan | None
    issue_ids: tuple[str, ...] = ()


def _candidate_pairs(group: VulnerabilityGroup) -> list[tuple[str, FixPlan]]:
    """Return retained package plans, with a legacy summary fallback."""
    candidates = getattr(group, "fix_plan_candidates", None) or []
    pairs: list[tuple[str, FixPlan]] = []
    if not isinstance(candidates, (list, tuple)):
        candidates = []
    for candidate in candidates:
        issue_id = str(_task_value(candidate, "issue_id", ""))
        plan = _task_value(candidate, "plan")
        if issue_id and plan is not None:
            pairs.append((issue_id, plan))
    if pairs:
        return sorted(pairs, key=lambda pair: pair[0])
    legacy_plan = getattr(group, "fix_plan", None)
    return (
        [(str(getattr(group, "representative_issue_id", "")), legacy_plan)] if legacy_plan else []
    )


def _has_plan_status(plan: Any, status: FixPlanStatus) -> bool:
    """Compare typed or compatibility fix-plan statuses safely."""
    raw_status = _task_value(plan, "status")
    value = getattr(raw_status, "value", raw_status)
    return value == status or str(value).casefold() == status.value.casefold()


def _copy_selected_plan(plan: Any, **updates: Any) -> Any:
    """Copy a typed plan while preserving compatibility projections."""
    if isinstance(plan, FixPlan):
        return plan.model_copy(update=updates)
    return plan


def _plan_version_key(version: str) -> tuple[int, object]:
    """Return a stable comparison key for a fixed-version candidate."""
    if SemanticVersion is not None:
        try:
            return (2, SemanticVersion.coerce(version))
        except ValueError:
            pass
    parts: list[int] = []
    for token in version.lstrip("vV").split(".")[:3]:
        digits = "".join(character for character in token if character.isdigit())
        parts.append(int(digits or "0"))
    return (1, tuple(parts + [0] * (3 - len(parts))))


def select_package_fix_plan(
    group: VulnerabilityGroup,
    current_strategy: RoutingStrategy | None = None,
) -> PackageFixPlanSelection:
    """Select one active plan while retaining every package candidate.

    Version candidates are preferred for initial package remediation and the
    highest fixed version is selected. Workaround candidates are merged only
    with other workaround candidates. ``current_strategy`` allows retry and
    pivot routing to select a retained alternative without creating a second
    triage group.
    """
    pairs = _candidate_pairs(group)
    if not pairs:
        return PackageFixPlanSelection(plan=None)

    if current_strategy == RoutingStrategy.CODE_WORKAROUND:
        selected_pairs = [
            pair for pair in pairs if _has_plan_status(pair[1], FixPlanStatus.WORKAROUND_FOUND)
        ]
        if not selected_pairs:
            selected_pairs = [
                pair for pair in pairs if _has_plan_status(pair[1], FixPlanStatus.NO_FIX)
            ]
    else:
        selected_pairs = [
            pair for pair in pairs if _has_plan_status(pair[1], FixPlanStatus.VERSION_FOUND)
        ]
        if not selected_pairs:
            selected_pairs = [
                pair for pair in pairs if _has_plan_status(pair[1], FixPlanStatus.WORKAROUND_FOUND)
            ]
        if not selected_pairs:
            selected_pairs = [
                pair for pair in pairs if _has_plan_status(pair[1], FixPlanStatus.NO_FIX)
            ]

    if not selected_pairs:
        return PackageFixPlanSelection(plan=None)

    selected_ids = tuple(issue_id for issue_id, _ in selected_pairs)
    if _has_plan_status(selected_pairs[0][1], FixPlanStatus.VERSION_FOUND):
        version, source_plan = max(
            ((_task_value(plan, "fixed_version") or "", plan) for _, plan in selected_pairs),
            key=lambda item: _plan_version_key(item[0]),
        )
        return PackageFixPlanSelection(
            plan=_copy_selected_plan(
                source_plan,
                fixed_version=version,
                workaround_snippets=None,
                strategy_used="UPDATE_VERSION",
            ),
            issue_ids=selected_ids,
        )

    if _has_plan_status(selected_pairs[0][1], FixPlanStatus.WORKAROUND_FOUND):
        snippets: list[str] = []
        for _, plan in selected_pairs:
            for snippet in _task_value(plan, "workaround_snippets", []) or []:
                if snippet not in snippets:
                    snippets.append(snippet)
        source_plan = selected_pairs[0][1]
        return PackageFixPlanSelection(
            plan=_copy_selected_plan(
                source_plan,
                fixed_version=None,
                workaround_snippets=snippets or None,
                strategy_used="WORKAROUND",
            ),
            issue_ids=selected_ids,
        )

    return PackageFixPlanSelection(
        plan=_copy_selected_plan(
            selected_pairs[0][1],
            fixed_version=None,
            workaround_snippets=None,
            strategy_used="NO_FIX",
        ),
        issue_ids=selected_ids,
    )


def _task_status_name(task: Any) -> str:
    """Return a task status as its stable string value."""
    status = _task_value(task, "status", TaskStatus.PENDING)
    return str(getattr(status, "value", status))


def task_group_lineage(
    task_queue: Mapping[str, Any],
    group_id: str,
) -> list[Any]:
    """Return the root task and all pivot descendants for one group.

    Pivot children retain the package group's ``parent_group_id``. The task's
    ``parent_task_id`` is the authoritative lineage relationship, so this
    helper follows descendants while portfolio planning collapses the chain
    to one active leaf.

    Args:
        task_queue: Task ID to task mapping.
        group_id: Package-group identifier shared by initial and pivot tasks.

    Returns:
        Tasks in deterministic breadth-first lineage order. An empty list is
        returned when the group has no task.
    """
    tasks = {str(task_id): task for task_id, task in task_queue.items()}
    group_task_ids = {
        task_id
        for task_id, task in tasks.items()
        if _task_value(task, "parent_group_id") == group_id
    }
    if not group_task_ids:
        return []

    roots = sorted(
        task_id
        for task_id in group_task_ids
        if _task_value(tasks[task_id], "parent_task_id") not in group_task_ids
    )
    children_by_parent: dict[str, list[str]] = {}
    for task_id, task in tasks.items():
        parent_task_id = _task_value(task, "parent_task_id")
        if parent_task_id:
            children_by_parent.setdefault(str(parent_task_id), []).append(task_id)
    for child_ids in children_by_parent.values():
        child_ids.sort()

    ordered: list[Any] = []
    queue = list(roots)
    visited: set[str] = set()
    while queue:
        task_id = queue.pop(0)
        if task_id in visited or task_id not in tasks:
            continue
        visited.add(task_id)
        ordered.append(tasks[task_id])
        queue.extend(children_by_parent.get(task_id, []))
    return ordered


def effective_group_status(
    task_queue: Mapping[str, Any],
    group_id: str,
) -> str:
    """Compute a reader-facing outcome from one group's task lineage.

    Pivot markers are audit state rather than remediation outcomes. Once a
    child exists, failed or incomplete descendants take precedence over a
    parent that previously passed QA.

    Args:
        task_queue: Authoritative task-ID to task mapping.
        group_id: Initial vulnerability-group identifier.

    Returns:
        The effective lifecycle status as a string value.
    """
    statuses = [_task_status_name(task) for task in task_group_lineage(task_queue, group_id)]
    if not statuses:
        return TaskStatus.PENDING.value

    active_statuses = [status for status in statuses if status != TaskStatus.PIVOTED.value]
    if not active_statuses:
        return TaskStatus.PIVOTED.value
    if TaskStatus.UNFIXABLE.value in active_statuses:
        return TaskStatus.UNFIXABLE.value
    if TaskStatus.INCONCLUSIVE.value in active_statuses:
        return TaskStatus.INCONCLUSIVE.value
    if TaskStatus.NEEDS_RETRY.value in active_statuses:
        return TaskStatus.NEEDS_RETRY.value
    if TaskStatus.PENDING.value in active_statuses:
        return TaskStatus.PENDING.value
    if TaskStatus.OPTIMISTICALLY_FIXED.value in active_statuses:
        return TaskStatus.OPTIMISTICALLY_FIXED.value
    if TaskStatus.QA_PASSED.value in active_statuses:
        return TaskStatus.QA_PASSED.value
    return active_statuses[0]


def terminal_outcome_issues(state: Mapping[str, Any]) -> list[str]:
    """Return deterministic reasons the final run cannot be successful.

    This pure projection is shared by teardown, report generation, and the
    public API. A worker success cannot override an unfixable or incomplete
    task, and the authoritative final scan must not contain blocking findings.

    Args:
        state: Current or final orchestration state.

    Returns:
        Deduplicated human-readable failure reasons; an empty list means no
        terminal contradiction was found.
    """
    task_queue = {
        str(task_id): task for task_id, task in (state.get("task_queue", {}) or {}).items()
    }
    reasons: list[str] = []
    for task_id, task in sorted(task_queue.items()):
        status = _task_status_name(task)
        if status in _UNRESOLVED_GROUP_STATUSES:
            reasons.append(f"task {task_id} ended in {status}")
        elif status == TaskStatus.PIVOTED.value:
            descendants = [
                candidate
                for candidate in task_queue.values()
                if _task_value(candidate, "parent_task_id") == task_id
            ]
            if not descendants:
                reasons.append(f"task {task_id} is pivoted without a child task")

    final_scan = state.get("final_full_scan_result")
    if final_scan is not None:
        completed = _task_value(final_scan, "completed")
        scan_status = str(_task_value(final_scan, "status", "")).lower()
        if completed is False or scan_status in {"scan_failed", "failed", "error", "timeout"}:
            reasons.append("authoritative final scan failed")
        remaining = list(_task_value(final_scan, "remaining_target_identifiers", []) or [])
        if remaining:
            reasons.append(
                f"authoritative final scan still contains {len(set(remaining))} target identifier(s)"
            )
        new_identifiers = list(_task_value(final_scan, "new_identifiers", []) or [])
        if new_identifiers:
            reasons.append(
                f"authoritative final scan detected {len(set(new_identifiers))} new identifier(s)"
            )
    elif (
        task_queue
        and state.get("workspace_volume")
        and not state.get("final_full_scan_completed", False)
    ):
        reasons.append("authoritative final scan was not completed")

    # Keep the result stable when a malformed task projection repeats the same
    # reason through multiple compatibility paths.
    return list(dict.fromkeys(reasons))


def create_skinny_subagent_group(
    group: VulnerabilityGroup,
    *,
    keep_identifiers: int = 0,
) -> VulnerabilityGroup:
    """Create the bounded group projection supplied to an execution agent."""
    return group.model_copy(
        update={
            "cve_ids": group.cve_ids[:keep_identifiers],
            "ghsa_ids": group.ghsa_ids[:keep_identifiers],
            "versions": group.versions[:keep_identifiers],
            "issues": [],
        }
    )


def filter_constraints_ledger(
    constraints_ledger: Sequence[str],
    target_groups: VulnerabilityGroup | Sequence[VulnerabilityGroup],
) -> list[str]:
    """Keep only constraints relevant to the requested group components."""
    groups = [target_groups] if isinstance(target_groups, VulnerabilityGroup) else target_groups
    components = [group.vulnerable_component for group in groups if group.vulnerable_component]
    if not components:
        return list(constraints_ledger)
    return [
        constraint for constraint in constraints_ledger if any(c in constraint for c in components)
    ]


def is_no_fix_package_removal_task(task: RemediationTask) -> bool:
    """Return whether ``task`` is the deterministic NO_FIX removal stage."""
    return task.no_fix_stage == NoFixMitigationStage.PACKAGE_REMOVAL


def is_no_fix_group(group: VulnerabilityGroup) -> bool:
    """Return whether no usable version or workaround plan remains."""
    candidates = getattr(group, "fix_plan_candidates", None) or []
    if isinstance(candidates, (list, tuple)) and candidates:
        return all(
            _has_plan_status(candidate.plan, FixPlanStatus.NO_FIX) for candidate in candidates
        )
    plan = select_package_fix_plan(group).plan
    return plan is not None and _has_plan_status(plan, FixPlanStatus.NO_FIX)


def is_transitive_group(group: VulnerabilityGroup) -> bool:
    """Return whether an SCA group represents a transitive dependency."""
    if group.parent_package_name or group.parent_contexts:
        return True
    return any(localized.is_direct_dependency is False for localized in group.localized_issues)


def group_parent_context(group: VulnerabilityGroup) -> tuple[str | None, str | None, str | None]:
    """Return the direct parent name, version, and declaration type for a group."""
    parent_name = group.parent_package_name
    parent_version = group.parent_package_version
    parent_type = group.parent_declaration_type
    if parent_name:
        if not parent_version:
            parent_version = group.dependency_versions.get(parent_name)
        return parent_name, parent_version, parent_type
    for context in group.parent_contexts:
        return (
            context.package_name,
            context.package_version or context.dependency_versions.get(context.package_name),
            context.declaration_type,
        )
    for localized in group.localized_issues:
        if localized.parent_package_name:
            parent_version = localized.parent_package_version or localized.dependency_versions.get(
                localized.parent_package_name
            )
            return (
                localized.parent_package_name,
                parent_version,
                localized.parent_declaration_type,
            )
    return None, None, None


def _group_manifest_paths(group: VulnerabilityGroup) -> list[str]:
    """Return stable, deduplicated manifest paths declared by a group."""
    paths: list[str] = []
    seen: set[str] = set()
    for value in [
        *(localized.manifest_file for localized in group.localized_issues),
        *group.file_paths,
        group.file_path,
    ]:
        if not value:
            continue
        normalized = value.replace("\\", "/").lstrip("/")
        if normalized and normalized not in seen:
            seen.add(normalized)
            paths.append(normalized)
    return paths


def _group_package_managers(group: VulnerabilityGroup) -> list[str]:
    """Return package managers represented by the group's localized issues."""
    managers: list[str] = []
    seen: set[str] = set()
    for localized in group.localized_issues:
        manager = (localized.package_manager or "").strip().lower()
        if manager and manager not in seen:
            seen.add(manager)
            managers.append(manager)
    return managers


def _group_override_dependency_type(group: VulnerabilityGroup) -> str:
    """Return the native override field for the group's package manager."""
    managers = set(_group_package_managers(group))
    if "yarn" in managers:
        return "resolutions"
    if "pnpm" in managers:
        return "pnpm_overrides"
    return "overrides"


def build_no_fix_package_removal_instruction(group: VulnerabilityGroup) -> str:
    """Build the deterministic initial instruction for a NO_FIX task.

    Args:
        group: Vulnerability group whose package should be removed.

    Returns:
        A supervisor-owned instruction for the package-removal stage.
    """
    component = (group.vulnerable_component or "the vulnerable package").strip()
    manifests = ", ".join(_group_manifest_paths(group)) or "the group-authorized manifest paths"
    managers = ", ".join(_group_package_managers(group)) or "the detected package manager"
    return (
        "NO_FIX MITIGATION — PACKAGE REMOVAL: Attempt to completely remove "
        f"'{component}' from the application. Inspect and use only these "
        f"group-authorized manifest paths: {manifests}. Use the dedicated "
        f"package-removal tool for the detected package manager ({managers}); "
        "do not manually edit or delete lockfile nodes. Regenerate applicable "
        "lockfiles through package-manager synchronization with lifecycle scripts "
        "disabled. Remove imports, require calls, and source code that depends "
        "on the package, then validate that the package is absent from the "
        "resolved dependency graph and that the application still passes its "
        "validation gates. If the package is transitive and has no safe removable "
        "declaration, report that failure so the supervisor can advance to the "
        "vulnerable-code-removal stage. Hint: Prioritize searching the CVE and "
        "local import/call sites without quotation marks."
    )


def build_no_fix_retry_instruction(
    task: RemediationTask,
    group: VulnerabilityGroup | None,
    evaluation: QAEvaluation | None = None,
    failure_feedback: str | None = None,
) -> str:
    """Build the deterministic vulnerable-code-removal retry instruction.

    Args:
        task: NO_FIX task being retried.
        group: Vulnerability group, when available.
        evaluation: QA evaluation from the failed package-removal attempt.
        failure_feedback: Worker failure text when no QA evaluation exists.

    Returns:
        A supervisor-owned instruction for the vulnerable-code-removal stage.

    Raises:
        ValueError: If the task is not in the vulnerable-code-removal stage.
    """
    if task.no_fix_stage != NoFixMitigationStage.VULNERABLE_CODE_REMOVAL:
        raise ValueError("NO_FIX retry instructions require the VULNERABLE_CODE_REMOVAL stage.")

    component = (group.vulnerable_component if group else None) or task.parent_group_id
    identifiers = []
    if group:
        identifiers.extend(group.cve_ids)
        identifiers.extend(group.ghsa_ids)
    cve_ids = ", ".join(dict.fromkeys(identifier for identifier in identifiers if identifier))
    cve_label = cve_ids or "the reported vulnerabilities"
    feedback = (
        evaluation.retry_feedback
        if evaluation and evaluation.retry_feedback
        else failure_feedback
        or "The package-removal attempt failed before a complete QA evaluation was available."
    )
    evidence: list[str] = []
    if evaluation and evaluation.failure_evidence:
        evidence.extend(evaluation.failure_evidence.exact_diagnostics)
        evidence.extend(evaluation.failure_evidence.failed_tests)
        if evaluation.failure_evidence.raw_excerpt:
            evidence.append(evaluation.failure_evidence.raw_excerpt)
    evidence_text = "\n".join(dict.fromkeys(item.strip() for item in evidence if item.strip()))
    evidence_block = f"\nStructured evidence:\n{evidence_text}" if evidence_text else ""

    return (
        "NO_FIX MITIGATION — VULNERABLE CODE REMOVAL: The previous attempt "
        f"to remove '{component}' failed validation. QA/worker feedback: {feedback}"
        f"{evidence_block}\n\n"
        f"Keep '{component}' installed. For {cve_label}, identify the vulnerable "
        "functions, classes, exports, and call patterns using the advisory, "
        "scanner evidence, and installed package source. Remove direct usage of "
        "the vulnerable APIs, trace and remove indirect callers at whatever depth "
        "the codebase requires, and clean up dead code left behind. Do not modify "
        "package manifests, lockfiles, dependency versions, or tests. Hint: "
        "Prioritize searching the exact error, CVE, and vulnerable API names "
        "without quotation marks."
    )


def advance_no_fix_stage(task: RemediationTask) -> dict[str, Any]:
    """Return the next deterministic task projection after a failed NO_FIX attempt.

    Args:
        task: The failed NO_FIX task.

    Returns:
        A partial ``RemediationTask`` update.  The terminal stage returns an
        empty mapping because it cannot advance again.

    Raises:
        ValueError: If ``task.no_fix_stage`` is not set.
    """
    stage = task.no_fix_stage
    if stage is None:
        raise ValueError("Cannot advance a task without a NO_FIX mitigation stage.")
    if stage == NoFixMitigationStage.PACKAGE_REMOVAL:
        return {
            "status": TaskStatus.NEEDS_RETRY,
            "retry_count": task.retry_count + 1,
            "no_fix_stage": NoFixMitigationStage.VULNERABLE_CODE_REMOVAL,
        }
    if stage == NoFixMitigationStage.VULNERABLE_CODE_REMOVAL:
        return {
            "status": TaskStatus.UNFIXABLE,
            "retry_count": task.retry_count + 1,
            "no_fix_stage": NoFixMitigationStage.UNFIXABLE,
        }
    return {}


def derive_initial_strategy(group: VulnerabilityGroup) -> RoutingStrategy:
    """
    Derive the initial routing strategy from a group's fix plan.

    Returns ``VERSION_BUMP`` only when the fix plan has ``status=VERSION_FOUND``
    (i.e. a safe pinned version is available).  All other plans â€” workaround,
    no-fix, or absent â€” map to ``CODE_WORKAROUND``.
    """
    fix_plan = select_package_fix_plan(group).plan
    if fix_plan is not None and fix_plan.status == FixPlanStatus.VERSION_FOUND:
        return RoutingStrategy.VERSION_BUMP
    return RoutingStrategy.CODE_WORKAROUND


def derive_initial_qa_policy(group: VulnerabilityGroup) -> QAPolicy:
    """Derive the immutable QA policy for a newly created task.

    Args:
        group: Vulnerability group being scheduled.

    Returns:
        The supervisor-owned QA policy for the task's initial attempt.
    """
    if is_no_fix_group(group):
        return QAPolicy.NO_FIX_PACKAGE_REMOVAL
    if (
        select_package_fix_plan(group).plan is not None
        and derive_initial_strategy(group) == RoutingStrategy.VERSION_BUMP
    ):
        return QAPolicy.VERSION_BUMP
    return QAPolicy.INITIAL_CODE_WORKAROUND


def derive_missing_task_qa_policy(
    task: RemediationTask,
    group: VulnerabilityGroup | None,
) -> QAPolicy | None:
    """Derive a safe QA policy when a task lacks committed policy provenance.

    Only evidence that is deterministic from the task and its current
    vulnerability group is accepted. A pivoted child whose policy cannot be
    reconstructed returns ``None`` so the caller can fail closed.

    Args:
        task: Existing task whose policy may be absent.
        group: Current vulnerability group, when available.

    Returns:
        A deterministically recoverable policy, or ``None`` when provenance is
        ambiguous.
    """
    if task.no_fix_stage == NoFixMitigationStage.PACKAGE_REMOVAL:
        return QAPolicy.NO_FIX_PACKAGE_REMOVAL
    if task.no_fix_stage == NoFixMitigationStage.VULNERABLE_CODE_REMOVAL:
        return QAPolicy.NO_FIX_CODE_REMOVAL
    if group is not None and is_no_fix_group(group):
        return QAPolicy.NO_FIX_PACKAGE_REMOVAL
    if task.strategy == RoutingStrategy.VERSION_BUMP:
        return QAPolicy.VERSION_BUMP
    if (
        task.parent_task_id is None
        and task.strategy == RoutingStrategy.CODE_WORKAROUND
        and (group is None or not is_no_fix_group(group))
    ):
        return QAPolicy.INITIAL_CODE_WORKAROUND
    return None


def build_initial_remediation_task(
    group: VulnerabilityGroup,
    task_id: str,
) -> RemediationTask:
    """
    Create an initial Depth-0 ``RemediationTask`` from a vulnerability group.

    The task inherits the strategy derived from the group's fix plan and
    starts in the ``PENDING`` status.  The instruction is seeded from the
    fix plan's instruction field if available.

    Parameters
    ----------
    group:
        The ``VulnerabilityGroup`` to remediate.
    task_id:
        Unique identifier for this task (e.g. ``'task-1'``).

    Returns
    -------
    RemediationTask
        A freshly created task ready to be added to ``task_queue``.
    """
    selection = select_package_fix_plan(group)
    fix_plan = selection.plan
    strategy = derive_initial_strategy(group)
    qa_policy = derive_initial_qa_policy(group)
    no_fix_stage: NoFixMitigationStage | None = None
    if is_no_fix_group(group):
        no_fix_stage = NoFixMitigationStage.PACKAGE_REMOVAL
        instruction = build_no_fix_package_removal_instruction(group)
    else:
        instruction = ""
        if fix_plan is not None and fix_plan.instruction:
            instruction = fix_plan.instruction

    transitive = is_transitive_group(group)
    parent_name, parent_version, parent_type = group_parent_context(group)
    has_parent_target = (
        strategy == RoutingStrategy.VERSION_BUMP and transitive and bool(parent_name)
    )
    target_package_name = (
        parent_name
        if has_parent_target
        else (group.vulnerable_component if strategy == RoutingStrategy.VERSION_BUMP else None)
    )
    direct_dependency_type = next(
        (
            localized.declaration_type
            for localized in group.localized_issues
            if localized.declaration_type
        ),
        None if transitive else "dependencies",
    )
    target_dependency_type = (
        parent_type
        if has_parent_target
        else (
            _group_override_dependency_type(group)
            if transitive and strategy == RoutingStrategy.VERSION_BUMP
            else direct_dependency_type
        )
    )
    if has_parent_target:
        child_version = fix_plan.fixed_version if fix_plan else None
        declaration = parent_type or "dependencies"
        instruction = (
            f'Update directly declared parent "{parent_name}" in {declaration} '
            f"to the minimum compatible released version that resolves transitive "
            f'package "{group.vulnerable_component}" to at least '
            f'"{child_version or "the OSV-fixed version"}". '
            "Do not use a package override unless the parent update stages are exhausted."
        )
    initial_stage = (
        SCARemediationStage.OSV_MINIMUM
        if strategy == RoutingStrategy.VERSION_BUMP and has_parent_target
        else SCARemediationStage.PACKAGE_OVERRIDE
        if strategy == RoutingStrategy.VERSION_BUMP and transitive
        else SCARemediationStage.CODE_WORKAROUND
        if strategy == RoutingStrategy.CODE_WORKAROUND
        else SCARemediationStage.OSV_MINIMUM
    )

    return RemediationTask(
        task_id=task_id,
        parent_group_id=group.group_id,
        is_synthetic=bool(getattr(group, "is_synthetic", False)),
        qa_policy=qa_policy,
        strategy=strategy,
        strategy_stage=initial_stage,
        target_package_name=target_package_name,
        target_dependency_type=target_dependency_type,
        parent_package_name=parent_name,
        parent_package_version=parent_version,
        parent_minimum_version=None,
        no_fix_stage=no_fix_stage,
        selected_version=(
            None
            if no_fix_stage is not None or has_parent_target
            else (fix_plan.fixed_version if fix_plan is not None else None)
        ),
        selected_plan_issue_ids=list(selection.issue_ids),
        instruction=instruction,
        status=TaskStatus.PENDING,
        retry_count=0,
        ancestry_depth=0,
    )
