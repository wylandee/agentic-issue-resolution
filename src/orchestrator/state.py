"""
state.py - LangGraph state schemas for the AppSec Remediation Engine.

Phase 4.1 (Legacy):
    ``RemediationState`` - single-issue, kept for the current Phase 4.1 graph.

Phase 5:
    ``OrchestratorState`` - supervisor master state for the hub-and-spoke
    remedy architecture.
    ``SubagentState`` - ephemeral private state for specialist subagents.

Reducer notes
-------------
* ``errors`` uses ``operator.add`` in both states so each node can return
  only its new error strings and LangGraph will append them.
* ``messages`` exists only in ``SubagentState`` so subagent ReAct transcripts
  stay isolated from the long-lived supervisor state.
* ``changed_files`` uses ``operator.add`` so each node can report only the
  files it newly observed as changed.
* All other fields use the default "last writer wins" semantics.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Mapping, Optional, TypeVar

from langgraph.graph.message import AnyMessage, add_messages
from typing_extensions import TypedDict

from src.contracts.schemas import (
    AgentActionSummary,
    EditRequest,
    EditResult,
    FixPlan,
    FixPlanStatus,
    GroupRemediationStatus,
    LocalizedIssue,
    QAEvaluation,
    QAAttemptResult,
    RemediationTask,
    RoutingStrategy,
    SCARemediationStage,
    SupervisorDecision,
    SupervisorRetryPlan,
    StateConsistencyEvent,
    SystemContext,
    TaskStatus,
    UpdateRetryDiagnostics,
    TaskAttemptSnapshot,
    WorkerAttemptResult,
    VulnerabilityGroup,
    VulnerabilityIssue,
)

K = TypeVar("K")
V = TypeVar("V")


def merge_dict_reducer(
    left: Mapping[K, V] | None,
    right: Mapping[K, V] | None,
) -> Dict[K, V]:
    """Merge dict-like values without mutating either input."""
    merged: Dict[K, V] = dict(left or {})
    if right:
        merged.update(right)
    return merged


def replace_dict_reducer(
    _left: Mapping[K, V] | None,
    right: Mapping[K, V] | None,
) -> Dict[K, V]:
    """Replace an authoritative dict projection with the newest snapshot.

    Phase 5 supervisor projections are complete snapshots, not patches.  A
    merge reducer cannot represent deletion: returning ``{}`` would leave old
    retry plans or task records in the graph state.  Keep ``merge_dict_reducer``
    for additive/compatibility maps and use this reducer for state owned by
    the supervisor.
    """
    return dict(right or {})


def _derive_legacy_task_from_group(group: VulnerabilityGroup) -> RemediationTask:
    """Build a synthetic task for legacy group-based subagent callers."""
    fix_plan = group.fix_plan
    strategy = (
        RoutingStrategy.VERSION_BUMP
        if fix_plan is not None and fix_plan.status == FixPlanStatus.VERSION_FOUND
        else RoutingStrategy.CODE_WORKAROUND
    )
    instruction = fix_plan.instruction if fix_plan is not None else ""
    return RemediationTask(
        task_id=group.group_id,
        parent_group_id=group.group_id,
        strategy=strategy,
        strategy_stage=(
            SCARemediationStage.OSV_MINIMUM
            if strategy == RoutingStrategy.VERSION_BUMP
            else SCARemediationStage.CODE_WORKAROUND
        ),
        selected_version=(fix_plan.fixed_version if fix_plan is not None else None),
        instruction=instruction,
        status=TaskStatus.PENDING,
        retry_count=0,
        ancestry_depth=0,
    )


def _derive_feedback_by_group(
    target_tasks: List[RemediationTask],
    target_groups: List[VulnerabilityGroup],
    feedback_by_task: Optional[Mapping[str, str]],
) -> Dict[str, str]:
    """Translate task-keyed feedback into group-keyed feedback when possible."""
    feedback_by_group: Dict[str, str] = {}
    task_map = {task.task_id: task for task in target_tasks}
    group_map = {group.group_id: group for group in target_groups}

    for task_id, feedback in dict(feedback_by_task or {}).items():
        task = task_map.get(task_id)
        if task is not None:
            feedback_by_group[task.parent_group_id] = feedback
            continue
        if task_id in group_map:
            feedback_by_group[task_id] = feedback

    return feedback_by_group


class RemediationState(TypedDict, total=False):
    """
    Full state schema for the Phase 4.1 remediation graph.

    Fields with ``total=False`` are optional. Only ``issue`` and ``repo_root``
    are required in the initial invocation dict.
    """

    issue: VulnerabilityIssue
    repo_root: str

    localized_issue: Optional[LocalizedIssue]
    fix_plan: Optional[FixPlan]
    edit_request: Optional[EditRequest]
    edit_result: Optional[EditResult]

    status: str
    dry_run: bool
    errors: Annotated[list[str], operator.add]


class OrchestratorState(TypedDict, total=False):
    """
    Full state schema for the Phase 5 supervisor master state.

    Required inputs
    ---------------
    repo_root:
        Absolute path to the cloned repository on disk.
    valid_groups:
        Non-empty list of triaged ``VulnerabilityGroup`` records.

    Orchestration fields
    --------------------
    workspace_volume:
        Docker named volume shared across builder, remedy agent, and teardown.

    Supervisor memory / outputs
    ---------------------------
    changed_files:
        Repo-relative files successfully modified across subagent runs.
    task_queue:
        Dict mapping task_id → RemediationTask; the primary unit of Phase 5 work.
    """

    repo_root: str
    valid_groups: List[VulnerabilityGroup]

    issues: List[VulnerabilityIssue]
    system_context: SystemContext

    constraints_ledger: Annotated[List[str], operator.add]
    retry_counts: Annotated[Dict[str, int], merge_dict_reducer]
    group_strategies: Annotated[Dict[str, RoutingStrategy], merge_dict_reducer]
    group_statuses: Annotated[Dict[str, GroupRemediationStatus], merge_dict_reducer]
    qa_evaluations: Annotated[Dict[str, QAEvaluation], replace_dict_reducer]
    action_summaries: Annotated[List[AgentActionSummary], operator.add]
    retry_diagnostics_by_task: Annotated[
        Dict[str, UpdateRetryDiagnostics], replace_dict_reducer
    ]
    retry_plans_by_task: Annotated[
        Dict[str, SupervisorRetryPlan], replace_dict_reducer
    ]
    attempt_snapshots_by_id: Annotated[
        Dict[str, TaskAttemptSnapshot], merge_dict_reducer
    ]
    worker_results_by_attempt: Annotated[
        Dict[str, WorkerAttemptResult], merge_dict_reducer
    ]
    qa_results_by_attempt: Annotated[
        Dict[str, QAAttemptResult], merge_dict_reducer
    ]
    processed_worker_attempt_ids: Annotated[List[str], operator.add]
    processed_qa_attempt_ids: Annotated[List[str], operator.add]
    consistency_events: Annotated[List[StateConsistencyEvent], operator.add]
    state_revision: int
    changed_files: Annotated[List[str], operator.add]

    # Phase 5 Task Queue (primary orchestration unit)
    task_queue: Annotated[Dict[str, RemediationTask], replace_dict_reducer]
    active_target_task_ids: List[str]

    workspace_volume: Optional[str]

    # Supervisor routing fields
    next_routing_step: str
    active_target_group_ids: List[str]
    feedback_by_group: Annotated[Dict[str, str], replace_dict_reducer]
    feedback_by_task: Annotated[Dict[str, str], replace_dict_reducer]
    supervisor_instructions: str
    eval_status: str
    qa_investigation_report: str

    status: str
    diff: str
    langsmith_run_id: str
    langsmith_trace_url: str
    trajectory_path: str
    errors: Annotated[List[str], operator.add]


class SubagentState(TypedDict, total=False):
    """
    Ephemeral private state for one specialist subagent run.

    This is the only Phase 5 state that carries localized ReAct messages.
    """

    repo_root: str
    workspace_volume: str

    target_tasks: List[RemediationTask]
    target_groups: List[VulnerabilityGroup]
    feedback_by_group: Dict[str, str]
    feedback_by_task: Dict[str, str]
    previous_action_summaries_by_task: Dict[str, str]
    retry_diagnostics_by_task: Dict[str, UpdateRetryDiagnostics]
    target_attempt_snapshots: Dict[str, TaskAttemptSnapshot]

    target_task: RemediationTask
    target_group: VulnerabilityGroup
    attempt_snapshot: Optional[TaskAttemptSnapshot]
    constraints_ledger: List[str]
    previous_feedback: Optional[str]

    messages: Annotated[List[AnyMessage], add_messages]

    action_summaries: List[AgentActionSummary]
    changed_files: Annotated[List[str], operator.add]
    errors: Annotated[List[str], operator.add]


def initial_orchestrator_state(
    repo_root: str,
    valid_groups: List[VulnerabilityGroup],
    issues: Optional[List[VulnerabilityIssue]] = None,
    system_context: Optional[SystemContext] = None,
) -> Dict[str, Any]:
    """Build a well-formed initial ``OrchestratorState`` dict."""
    state: Dict[str, Any] = {
        "repo_root": repo_root,
        "valid_groups": valid_groups,
        "constraints_ledger": [],
        "retry_counts": {},
        "group_strategies": {},
        "group_statuses": {},
        "qa_evaluations": {},
        "action_summaries": [],
        "retry_diagnostics_by_task": {},
        "retry_plans_by_task": {},
        "attempt_snapshots_by_id": {},
        "worker_results_by_attempt": {},
        "qa_results_by_attempt": {},
        "processed_worker_attempt_ids": [],
        "processed_qa_attempt_ids": [],
        "consistency_events": [],
        "state_revision": 0,
        "changed_files": [],
        "task_queue": {},
        "active_target_task_ids": [],
        "workspace_volume": None,
        "status": "pending",
        "next_routing_step": "",
        "active_target_group_ids": [],
        "feedback_by_group": {},
        "feedback_by_task": {},
        "supervisor_instructions": "",
        "eval_status": "",
        "qa_investigation_report": "",
        "diff": "",
        "trajectory_path": "",
        "errors": [],
    }
    if issues is not None:
        state["issues"] = issues
    if system_context is not None:
        state["system_context"] = system_context
    return state


def initial_update_subagent_state(
    repo_root: str,
    workspace_volume: str,
    target_tasks: List[RemediationTask] | List[VulnerabilityGroup],
    target_groups: List[VulnerabilityGroup] | List[str],
    constraints_ledger: Optional[List[str]] = None,
    feedback_by_task: Optional[Dict[str, str]] = None,
    feedback_by_group: Optional[Dict[str, str]] = None,
    previous_action_summaries_by_task: Optional[Dict[str, str]] = None,
    retry_diagnostics_by_task: Optional[Dict[str, UpdateRetryDiagnostics]] = None,
    target_attempt_snapshots: Optional[Dict[str, TaskAttemptSnapshot]] = None,
) -> Dict[str, Any]:
    """
    Build a well-formed initial batch ``SubagentState`` dict.

    Supports both the current task-based signature and the legacy
    group-based signature used by older tests and callers:

    - new: ``(repo_root, workspace_volume, target_tasks, target_groups, constraints_ledger, ...)``
    - old: ``(repo_root, workspace_volume, target_groups, constraints_ledger, ...)``
    """
    legacy_mode = bool(target_tasks) and isinstance(list(target_tasks)[0], VulnerabilityGroup)
    if legacy_mode:
        legacy_groups = list(target_tasks)
        target_groups_list = legacy_groups
        target_tasks_list = [
            _derive_legacy_task_from_group(group)
            for group in legacy_groups
        ]
        constraints_list = list(target_groups)
        legacy_feedback = (
            dict(constraints_ledger)
            if isinstance(constraints_ledger, Mapping)
            else {}
        )
        feedback_by_group_dict = dict(feedback_by_group or legacy_feedback)
        feedback_by_task_dict = dict(
            feedback_by_task
            or feedback_by_group_dict
        )
        previous_summaries_dict = dict(previous_action_summaries_by_task or {})
        retry_diagnostics_dict = dict(retry_diagnostics_by_task or {})
        target_attempt_snapshots_dict = dict(target_attempt_snapshots or {})
    else:
        target_tasks_list = list(target_tasks)
        target_groups_list = list(target_groups)
        constraints_list = list(constraints_ledger)
        feedback_by_task_dict = dict(feedback_by_task or {})
        previous_summaries_dict = dict(previous_action_summaries_by_task or {})
        retry_diagnostics_dict = dict(retry_diagnostics_by_task or {})
        target_attempt_snapshots_dict = dict(target_attempt_snapshots or {})
        feedback_by_group_dict = dict(
            feedback_by_group
            or _derive_feedback_by_group(
                target_tasks_list,
                target_groups_list,
                feedback_by_task_dict,
            )
        )

    return {
        "repo_root": repo_root,
        "workspace_volume": workspace_volume,
        "target_tasks": target_tasks_list,
        "target_groups": target_groups_list,
        "feedback_by_group": feedback_by_group_dict,
        "feedback_by_task": feedback_by_task_dict,
        "previous_action_summaries_by_task": previous_summaries_dict,
        "retry_diagnostics_by_task": retry_diagnostics_dict,
        "target_attempt_snapshots": target_attempt_snapshots_dict,
        "constraints_ledger": constraints_list,
        "messages": [],
        "changed_files": [],
        "errors": [],
    }


def initial_workaround_subagent_state(
    repo_root: str,
    workspace_volume: str,
    target_task: RemediationTask | VulnerabilityGroup,
    target_group: VulnerabilityGroup | List[str],
    constraints_ledger: Optional[List[str]] = None,
    previous_feedback: Optional[str] = None,
    attempt_snapshot: Optional[TaskAttemptSnapshot] = None,
) -> Dict[str, Any]:
    """
    Build a well-formed single-task workaround ``SubagentState`` dict.

    Supports both the current task-based signature and the legacy
    group-based signature:

    - new: ``(repo_root, workspace_volume, target_task, target_group, constraints_ledger, ...)``
    - old: ``(repo_root, workspace_volume, target_group, constraints_ledger, ...)``
    """
    if constraints_ledger is None:
        target_group_obj = target_task
        target_task_obj = _derive_legacy_task_from_group(target_group_obj)
        constraints_list = list(target_group)
    else:
        target_task_obj = target_task
        target_group_obj = target_group
        constraints_list = list(constraints_ledger)

    return {
        "repo_root": repo_root,
        "workspace_volume": workspace_volume,
        "target_task": target_task_obj,
        "target_group": target_group_obj,
        "constraints_ledger": constraints_list,
        "previous_feedback": previous_feedback,
        "attempt_snapshot": attempt_snapshot,
        "messages": [],
        "changed_files": [],
        "errors": [],
    }
