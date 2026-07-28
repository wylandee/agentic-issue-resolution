"""
state.py - LangGraph state schemas for the AppSec Remediation Engine.

``OrchestratorState`` - supervisor master state for the hub-and-spoke remedy
architecture.
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
import ntpath
import os
from pathlib import Path
from typing import Annotated, Any, Dict, List, Mapping, Optional, TypeVar

from langgraph.graph.message import AnyMessage, add_messages
from typing_extensions import TypedDict

from remediation_engine.contracts.schemas import (
    AgentActionSummary,
    FixPlanStatus,
    GroupRemediationStatus,
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
    WorkaroundReplayPlan,
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


def _normalise_scan_identifiers(values: List[Optional[str]]) -> List[str]:
    """Return sorted, de-duplicated CVE/GHSA identifiers."""
    return sorted({value.strip().upper() for value in values if value and value.strip()})


def _scan_identifiers_from_issues(
    issues: List[VulnerabilityIssue],
) -> List[str]:
    """Collect scanner identifiers from the complete initial issue set."""
    values: List[Optional[str]] = []
    for issue in issues:
        values.extend([issue.cve_id, issue.ghsa_id])
    return _normalise_scan_identifiers(values)


def _scan_identifiers_from_groups(
    groups: List[VulnerabilityGroup],
) -> List[str]:
    """Collect scanner identifiers from groups for skip-triage compatibility."""
    values: List[Optional[str]] = []
    for group in groups:
        values.extend(group.cve_ids or [])
        values.extend(group.ghsa_ids or [])
        for issue in group.issues or []:
            values.extend([issue.cve_id, issue.ghsa_id])
    return _normalise_scan_identifiers(values)


def _repo_relative_path(value: str | None, repo_root: str) -> str | None:
    """Convert a manifest path to a POSIX path relative to ``repo_root``.

    Scanner reports and cached triage files may contain either POSIX or
    Windows absolute paths.  The worker tools intentionally reject absolute
    paths, so normalize them at the graph boundary before any task is built.
    Paths outside the repository are discarded rather than passed to a tool.
    """
    if not value:
        return None
    raw = str(value).strip().replace("\\", "/")
    if not raw:
        return None

    root = Path(repo_root).resolve()
    root_text = str(root).replace("\\", "/").rstrip("/")
    raw_casefold = raw.casefold()
    root_casefold = root_text.casefold()
    if raw_casefold == root_casefold:
        return None
    if raw_casefold.startswith(root_casefold + "/"):
        return raw[len(root_text) + 1 :].lstrip("/") or None

    # ``Path`` handles native absolute paths; ``ntpath`` also recognizes a
    # Windows drive path when a state file was produced on another platform.
    if os.path.isabs(raw) or ntpath.isabs(raw):
        try:
            return Path(raw).resolve().relative_to(root).as_posix()
        except (OSError, ValueError):
            return None

    while raw.startswith("./"):
        raw = raw[2:]
    return raw or None


def normalize_group_paths(
    groups: List[VulnerabilityGroup],
    repo_root: str,
) -> List[VulnerabilityGroup]:
    """Return groups whose manifest paths are safe repository-relative paths.

    This boundary normalization also updates nested ``LocalizedIssue``
    records and replaces an absolute path embedded in a legacy group ID.
    Newly generated groups are already relative; the helper is primarily for
    callers that load preprocessed JSON produced by older versions.
    """
    normalized: List[VulnerabilityGroup] = []
    for group in groups:
        replacements: dict[str, str] = {}
        localized_issues = []
        for localized in group.localized_issues or []:
            old = localized.manifest_file
            new = _repo_relative_path(old, repo_root)
            if old and old != new:
                replacements[str(old)] = new or "unknown-manifest"
            localized_issues.append(localized.model_copy(update={"manifest_file": new}))

        file_paths: List[str] = []
        for path in group.file_paths or []:
            new = _repo_relative_path(path, repo_root)
            if new and new not in file_paths:
                if str(path) != new:
                    replacements[str(path)] = new or "unknown-manifest"
                file_paths.append(new)
        file_path = _repo_relative_path(group.file_path, repo_root)
        if group.file_path and group.file_path != file_path:
            replacements[str(group.file_path)] = file_path or "unknown-manifest"
        if file_path and file_path not in file_paths:
            file_paths.insert(0, file_path)

        group_id = group.group_id.replace("\\", "/")
        for old, new in replacements.items():
            group_id = group_id.replace(str(old).replace("\\", "/"), new)
        fix_plan = group.fix_plan
        if fix_plan is not None and replacements:
            instruction = fix_plan.instruction or ""
            for old, new in replacements.items():
                instruction = instruction.replace(str(old), new).replace(
                    str(old).replace("\\", "/"), new
                )
            fix_plan = fix_plan.model_copy(update={"instruction": instruction})
        if (
            group_id == group.group_id
            and file_path == group.file_path
            and file_paths == list(group.file_paths or [])
            and localized_issues == list(group.localized_issues or [])
            and fix_plan is group.fix_plan
        ):
            # Keep object identity for already-canonical groups.  Some graph
            # reconciliation paths deliberately reuse unchanged group objects.
            normalized.append(group)
            continue
        normalized.append(
            group.model_copy(
                update={
                    "group_id": group_id,
                    "file_path": file_path,
                    "file_paths": file_paths,
                    "localized_issues": localized_issues,
                    "fix_plan": fix_plan,
                }
            )
        )
    return normalized


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
        Dict mapping task_id â†’ RemediationTask; the primary unit of Phase 5 work.
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
    workaround_replay_plans_by_task: Annotated[
        Dict[str, WorkaroundReplayPlan], replace_dict_reducer
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
    baseline_scan_identifiers: List[str]
    post_remediation_scan_identifiers: List[str]
    post_remediation_scan_issues: List[VulnerabilityIssue]
    new_vulnerability_identifiers: List[str]
    new_vulnerability_status: str
    triage_required: bool
    initial_triage_status: str
    initial_triage_executed: bool
    triage_reconciliation: Dict[str, List[str]]

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
    workaround_replay_plans_by_task: Dict[str, WorkaroundReplayPlan]
    target_attempt_snapshots: Dict[str, TaskAttemptSnapshot]

    target_task: RemediationTask
    target_group: VulnerabilityGroup
    attempt_snapshot: Optional[TaskAttemptSnapshot]
    constraints_ledger: List[str]
    previous_feedback: Optional[str]
    current_replay_plan: Optional[WorkaroundReplayPlan]

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
    valid_groups = normalize_group_paths(valid_groups, repo_root)
    baseline_scan_identifiers = (
        _scan_identifiers_from_issues(issues)
        if issues is not None
        else _scan_identifiers_from_groups(valid_groups)
    )
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
        "baseline_scan_identifiers": baseline_scan_identifiers,
        "post_remediation_scan_identifiers": [],
        "post_remediation_scan_issues": [],
        "new_vulnerability_identifiers": [],
        "new_vulnerability_status": "not_scanned",
        "triage_required": False,
        "initial_triage_status": "pending",
        "initial_triage_executed": False,
        "triage_reconciliation": {},
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
    current_replay_plan: Optional[WorkaroundReplayPlan] = None,
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
        "current_replay_plan": current_replay_plan,
        "messages": [],
        "changed_files": [],
        "errors": [],
    }


