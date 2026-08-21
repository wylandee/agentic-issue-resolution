"""Pure Supervisor routing and transition policy helpers.

The Supervisor node remains responsible for applying these policies to graph
state.  Keeping deterministic selection, retry, stage, and pivot rules here
gives the orchestration façade a stable seam for characterization tests and
future policy changes.
"""

from __future__ import annotations

from typing import Any

from remediation_engine.contracts.schemas import (
    FailureCategory,
    QAEvaluation,
    QAPolicy,
    RemediationTask,
    RoutingStrategy,
    SCARemediationStage,
    TaskStatus,
    UpdateRetryDiagnostics,
    VulnerabilityGroup,
)
from remediation_engine.orchestration.task_utils import TERMINAL_TASK_STATUSES

MAX_RETRIES: int = 3
"""Maximum number of QA-fail-retry cycles before a task is unfixable."""

_TERMINAL_STATUSES = TERMINAL_TASK_STATUSES
_WORKABLE_STATUSES = frozenset({TaskStatus.PENDING, TaskStatus.NEEDS_RETRY})
_SEVERITY_RANK: dict[str, int] = {
    "critical": 0,
    "high": 1,
    "medium": 2,
    "low": 3,
}


def _task_sort_key(
    task: RemediationTask,
    group_by_id: dict[str, VulnerabilityGroup],
) -> tuple[int, int, int, int, str, str]:
    """Return the stable priority key used for deterministic task selection."""
    group = group_by_id.get(task.parent_group_id)
    raw_severities: list[Any] = []
    if group is not None:
        raw_severities.append(getattr(group, "severity", None))
        raw_severities.extend(getattr(issue, "severity", None) for issue in group.issues or [])
    ranks = [
        _SEVERITY_RANK.get(str(getattr(value, "value", value)).lower(), 4)
        for value in raw_severities
        if value is not None
    ]
    severity_rank = min(ranks, default=4)
    strategy_rank = 0 if task.strategy == RoutingStrategy.VERSION_BUMP else 1
    status_rank = {TaskStatus.NEEDS_RETRY: 0, TaskStatus.PENDING: 1}.get(task.status, 2)
    return (
        severity_rank,
        strategy_rank,
        status_rank,
        task.retry_count,
        task.parent_group_id,
        task.task_id,
    )


def _dispatchable_task_ids_for_status(
    task_queue: dict[str, RemediationTask],
    statuses: set[TaskStatus],
    preferred_ids: list[str] | None = None,
    strategy: RoutingStrategy | None = None,
    limit: int | None = None,
    group_by_id: dict[str, VulnerabilityGroup] | None = None,
) -> list[str]:
    """Return non-terminal task IDs matching status and strategy filters."""
    candidate_ids = list(preferred_ids) if preferred_ids is not None else list(task_queue)
    tasks: list[RemediationTask] = []
    seen: set[str] = set()
    for task_id in candidate_ids:
        if task_id in seen:
            continue
        seen.add(task_id)
        task = task_queue.get(task_id)
        if task is None or task.status in _TERMINAL_STATUSES:
            continue
        if task.status not in statuses:
            continue
        if strategy is not None and task.strategy != strategy:
            continue
        tasks.append(task)

    if group_by_id is not None:
        tasks.sort(key=lambda task: _task_sort_key(task, group_by_id))
    dispatchable = [task.task_id for task in tasks]
    return dispatchable[:limit] if limit is not None else dispatchable


def _qa_ready_task_ids(
    task_queue: dict[str, RemediationTask],
    preferred_ids: list[str] | None = None,
    group_by_id: dict[str, VulnerabilityGroup] | None = None,
    limit: int | None = None,
) -> list[str]:
    """Return tasks whose worker result is ready for QA evaluation."""
    return _dispatchable_task_ids_for_status(
        task_queue,
        {TaskStatus.OPTIMISTICALLY_FIXED},
        preferred_ids=preferred_ids,
        group_by_id=group_by_id,
        limit=limit,
    )


def _is_exhausted_update_pivot_candidate(
    task: RemediationTask,
    diagnostics: UpdateRetryDiagnostics | None,
) -> bool:
    """Return whether a retry update task must pivot instead of retrying."""
    if task.parent_package_name and task.strategy_stage != SCARemediationStage.CODE_WORKAROUND:
        # A transitive task may only pivot after its explicit child-override
        # stage has also failed. Parent registry exhaustion is not terminal.
        return False
    return (
        task.strategy == RoutingStrategy.VERSION_BUMP
        and task.status == TaskStatus.NEEDS_RETRY
        and (
            task.strategy_stage == SCARemediationStage.CODE_WORKAROUND
            or (
                diagnostics is not None
                and (diagnostics.package_abandoned or diagnostics.exhausted_update_path)
            )
        )
    )


def _has_existing_workaround_child(
    task: RemediationTask,
    task_queue: dict[str, RemediationTask],
) -> bool:
    """Return whether a task already owns a code-workaround child."""
    return any(
        candidate.parent_task_id == task.task_id
        and candidate.strategy == RoutingStrategy.CODE_WORKAROUND
        for candidate in task_queue.values()
    )


def _next_sca_stage(
    stage: SCARemediationStage,
    transitive: bool = False,
) -> SCARemediationStage:
    """Advance one ordered SCA version strategy stage."""
    if stage == SCARemediationStage.OSV_MINIMUM:
        return SCARemediationStage.NPM_SAME_MAJOR
    if stage == SCARemediationStage.NPM_SAME_MAJOR:
        return SCARemediationStage.NPM_LATEST
    if stage == SCARemediationStage.NPM_LATEST and transitive:
        return SCARemediationStage.PACKAGE_OVERRIDE
    if stage == SCARemediationStage.PACKAGE_OVERRIDE:
        return SCARemediationStage.CODE_WORKAROUND
    return SCARemediationStage.CODE_WORKAROUND


def _selection_for_stage(stage: SCARemediationStage) -> str | None:
    """Return the registry selection mode for a strategy stage."""
    if stage == SCARemediationStage.NPM_SAME_MAJOR:
        return "same_major"
    if stage == SCARemediationStage.NPM_LATEST:
        return "latest"
    return None


def _worker_node_for_strategy(strategy: RoutingStrategy) -> str:
    """Return the worker node that handles a routing strategy."""
    if strategy == RoutingStrategy.VERSION_BUMP:
        return "update_subagent"
    return "workaround_subagent"


def _parent_status_for_strategy_pivot(
    parent_task: RemediationTask,
    new_strategy: RoutingStrategy,
    qa_evaluations: dict[str, QAEvaluation],
) -> TaskStatus:
    """Choose the terminal parent status when a pivot creates a child task."""
    evaluation = qa_evaluations.get(parent_task.task_id) or qa_evaluations.get(
        parent_task.parent_group_id
    )
    if (
        parent_task.strategy == RoutingStrategy.VERSION_BUMP
        and new_strategy == RoutingStrategy.CODE_WORKAROUND
        and evaluation is not None
    ):
        gates = evaluation.deterministic_gates
        if (
            parent_task.qa_policy == QAPolicy.VERSION_BUMP
            and gates is not None
            and gates.target_scanner_cleared is True
            and gates.tests_passed is False
        ):
            return TaskStatus.QA_PASSED
        if evaluation.failure_category == FailureCategory.BREAKING_CHANGE:
            return TaskStatus.PIVOTED
    return TaskStatus.UNFIXABLE


__all__ = [
    "MAX_RETRIES",
    "_TERMINAL_STATUSES",
    "_WORKABLE_STATUSES",
    "_dispatchable_task_ids_for_status",
    "_has_existing_workaround_child",
    "_is_exhausted_update_pivot_candidate",
    "_next_sca_stage",
    "_parent_status_for_strategy_pivot",
    "_qa_ready_task_ids",
    "_selection_for_stage",
    "_task_sort_key",
    "_worker_node_for_strategy",
]
