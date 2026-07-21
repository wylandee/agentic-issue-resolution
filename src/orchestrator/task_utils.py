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

from src.contracts.schemas import (
    FixPlanStatus,
    RemediationTask,
    RoutingStrategy,
    SCARemediationStage,
    TaskStatus,
    VulnerabilityGroup,
)


def derive_initial_strategy(group: VulnerabilityGroup) -> RoutingStrategy:
    """
    Derive the initial routing strategy from a group's fix plan.

    Returns ``VERSION_BUMP`` only when the fix plan has ``status=VERSION_FOUND``
    (i.e. a safe pinned version is available).  All other plans — workaround,
    no-fix, or absent — map to ``CODE_WORKAROUND``.
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
    instruction = ""
    if group.fix_plan is not None and group.fix_plan.instruction:
        instruction = group.fix_plan.instruction

    return RemediationTask(
        task_id=task_id,
        parent_group_id=group.group_id,
        strategy=strategy,
        strategy_stage=(
            SCARemediationStage.OSV_MINIMUM
            if strategy == RoutingStrategy.VERSION_BUMP
            else SCARemediationStage.CODE_WORKAROUND
        ),
        selected_version=(
            group.fix_plan.fixed_version
            if group.fix_plan is not None
            else None
        ),
        instruction=instruction,
        status=TaskStatus.PENDING,
        retry_count=0,
        ancestry_depth=0,
    )
