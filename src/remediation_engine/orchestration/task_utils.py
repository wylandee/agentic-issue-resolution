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

from typing import Any

from remediation_engine.contracts.schemas import (
    FixPlanStatus,
    NoFixMitigationStage,
    QAEvaluation,
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
        TaskStatus.MITIGATED,
        TaskStatus.PIVOTED,
    }
)


def is_no_fix_package_removal_task(task: RemediationTask) -> bool:
    """Return whether ``task`` is the deterministic NO_FIX removal stage."""
    return task.no_fix_stage == NoFixMitigationStage.PACKAGE_REMOVAL


def is_no_fix_group(group: VulnerabilityGroup) -> bool:
    """Return whether ``group`` has an explicit ``NO_FIX`` plan."""
    return group.fix_plan is not None and group.fix_plan.status == FixPlanStatus.NO_FIX


def is_transitive_group(group: VulnerabilityGroup) -> bool:
    """Return whether an SCA group represents a transitive dependency."""
    if group.parent_package_name:
        return True
    return any(localized.is_direct_dependency is False for localized in group.localized_issues)


def group_parent_context(group: VulnerabilityGroup) -> tuple[str | None, str | None, str | None]:
    """Return the direct parent name, version, and declaration type for a group."""
    parent_name = group.parent_package_name
    parent_version = group.parent_package_version
    parent_type = group.parent_declaration_type
    if parent_name:
        return parent_name, parent_version, parent_type
    for localized in group.localized_issues:
        if localized.parent_package_name:
            return (
                localized.parent_package_name,
                localized.parent_package_version,
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
    fix_plan = group.fix_plan
    if fix_plan is not None and fix_plan.status == FixPlanStatus.VERSION_FOUND:
        return RoutingStrategy.VERSION_BUMP
    return RoutingStrategy.CODE_WORKAROUND


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
    strategy = derive_initial_strategy(group)
    no_fix_stage: NoFixMitigationStage | None = None
    if is_no_fix_group(group):
        no_fix_stage = NoFixMitigationStage.PACKAGE_REMOVAL
        instruction = build_no_fix_package_removal_instruction(group)
    else:
        instruction = ""
        if group.fix_plan is not None and group.fix_plan.instruction:
            instruction = group.fix_plan.instruction

    transitive = is_transitive_group(group)
    parent_name, parent_version, parent_type = group_parent_context(group)
    has_parent_target = (
        strategy == RoutingStrategy.VERSION_BUMP and transitive and bool(parent_name)
    )
    target_package_name = (
        parent_name
        if has_parent_target
        else (
            group.vulnerable_component
            if transitive and strategy == RoutingStrategy.VERSION_BUMP
            else None
        )
    )
    target_dependency_type = (
        parent_type
        if has_parent_target
        else (
            _group_override_dependency_type(group)
            if transitive and strategy == RoutingStrategy.VERSION_BUMP
            else None
        )
    )
    if has_parent_target:
        child_version = group.fix_plan.fixed_version if group.fix_plan else None
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
            else (group.fix_plan.fixed_version if group.fix_plan is not None else None)
        ),
        instruction=instruction,
        status=TaskStatus.PENDING,
        retry_count=0,
        ancestry_depth=0,
    )
