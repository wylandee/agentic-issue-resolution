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
* ``changed_files`` uses an order-preserving set-like reducer so retries and
  bridge nodes cannot duplicate the same path in the final patch projection.
* All other fields use the default "last writer wins" semantics.
"""

from __future__ import annotations

import operator
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, TypeVar

from langgraph.graph.message import AnyMessage, add_messages
from typing_extensions import TypedDict

from remediation_engine.contracts.decision_codes import DecisionCode
from remediation_engine.contracts.schemas import (
    AgentActionSummary,
    FinalFullScanResult,
    FixPlanStatus,
    GroupRemediationStatus,
    NoFixMitigationStage,
    ODCScanEvidence,
    QAAttemptResult,
    QAEvaluation,
    QAPolicy,
    RemediationTask,
    RoutingStrategy,
    SCARemediationStage,
    StateConsistencyEvent,
    SupervisorRetryPlan,
    SystemContext,
    TaskAttemptSnapshot,
    TaskStatus,
    UpdateRetryDiagnostics,
    VulnerabilityGroup,
    VulnerabilityIssue,
    WorkaroundReplayPlan,
    WorkerAttemptResult,
)
from remediation_engine.contracts.supervisor_phases import AuditRecord
from remediation_engine.tools.manifest_locator import expand_dependency_ancestry_from_repository

K = TypeVar("K")
V = TypeVar("V")


def merge_dict_reducer(
    left: Mapping[K, V] | None,
    right: Mapping[K, V] | None,
) -> dict[K, V]:
    """Merge dict-like values without mutating either input."""
    merged: dict[K, V] = dict(left or {})
    if right:
        merged.update(right)
    return merged


def replace_dict_reducer(
    _left: Mapping[K, V] | None,
    right: Mapping[K, V] | None,
) -> dict[K, V]:
    """Replace an authoritative dict projection with the newest snapshot.

    Phase 5 supervisor projections are complete snapshots, not patches.  A
    merge reducer cannot represent deletion: returning ``{}`` would leave old
    retry plans or task records in the graph state.  Keep ``merge_dict_reducer``
    for additive/compatibility maps and use this reducer for state owned by
    the supervisor.
    """
    return dict(right or {})


def merge_changed_files_reducer(
    left: list[str] | None,
    right: list[str] | None,
) -> list[str]:
    """Merge changed-file projections while normalizing and de-duplicating paths."""
    merged: list[str] = []
    seen: set[str] = set()
    for value in [*(left or []), *(right or [])]:
        if not isinstance(value, str):
            continue
        path = value.replace("\\", "/").lstrip("/").strip()
        if path and path not in seen:
            seen.add(path)
            merged.append(path)
    return merged


def _normalise_scan_identifiers(values: list[str | None]) -> list[str]:
    """Return sorted, de-duplicated CVE/GHSA identifiers."""
    return sorted({value.strip().upper() for value in values if value and value.strip()})


def _scan_identifiers_from_issues(
    issues: list[VulnerabilityIssue],
) -> list[str]:
    """Collect scanner identifiers from the complete initial issue set."""
    values: list[str | None] = []
    for issue in issues:
        values.extend([issue.cve_id, issue.ghsa_id])
    return _normalise_scan_identifiers(values)


def _scan_identifiers_from_groups(
    groups: list[VulnerabilityGroup],
) -> list[str]:
    """Collect scanner identifiers from groups for skip-triage compatibility."""
    values: list[str | None] = []
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
    if value is None:
        return None
    raw = str(value).replace("\\", "/").strip()
    if not raw:
        return None

    try:
        return Path(raw).relative_to(Path(repo_root)).as_posix()
    except ValueError:
        pass
    try:
        return Path(raw).resolve().relative_to(Path(repo_root).resolve()).as_posix()
    except (OSError, ValueError):
        pass

    raw_clean = raw.lstrip("/")
    root_clean = str(repo_root).replace("\\", "/").strip().lstrip("/").rstrip("/")
    if raw_clean.casefold() == root_clean.casefold():
        return None
    if raw_clean.casefold().startswith(root_clean.casefold() + "/"):
        return raw_clean[len(root_clean) + 1 :].lstrip("/") or None

    while raw.startswith("./"):
        raw = raw[2:]
    return raw or None


def normalize_group_paths(
    groups: list[VulnerabilityGroup],
    repo_root: str,
) -> list[VulnerabilityGroup]:
    """Return groups whose manifest paths are safe repository-relative paths.

    This boundary normalization also updates nested ``LocalizedIssue``
    records and replaces an absolute path embedded in a legacy group ID.
    Newly generated groups are already relative; the helper is primarily for
    callers that load preprocessed JSON produced by older versions.
    """
    normalized: list[VulnerabilityGroup] = []
    for group in groups:
        replacements: dict[str, str] = {}
        localized_issues = []
        expanded_group_ancestry = list(group.dependency_ancestry)
        expanded_group_versions = dict(group.dependency_versions)
        for localized in group.localized_issues or []:
            old = localized.manifest_file
            new = _repo_relative_path(old, repo_root)
            if old and old != new:
                replacements[str(old)] = new or "unknown-manifest"
            odc_file_path = localized.issue.file_path or (localized.issue.raw_payload or {}).get(
                "filePath", ""
            )
            ancestry, versions = expand_dependency_ancestry_from_repository(
                Path(repo_root),
                new or old,
                odc_file_path,
                localized.dependency_ancestry,
                localized.dependency_versions,
            )
            localized_updates: dict[str, Any] = {"manifest_file": new}
            if ancestry != localized.dependency_ancestry:
                localized_updates.update(
                    {
                        "dependency_ancestry": ancestry,
                        "dependency_versions": versions,
                    }
                )
                if expanded_group_ancestry == list(group.dependency_ancestry):
                    expanded_group_ancestry = ancestry
                    expanded_group_versions = versions
            localized_issues.append(localized.model_copy(update=localized_updates))

        file_paths: list[str] = []
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
            and expanded_group_ancestry == list(group.dependency_ancestry)
            and expanded_group_versions == dict(group.dependency_versions)
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
                    "dependency_ancestry": expanded_group_ancestry,
                    "dependency_versions": expanded_group_versions,
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
        qa_policy=(
            QAPolicy.NO_FIX_PACKAGE_REMOVAL
            if fix_plan is not None and fix_plan.status == FixPlanStatus.NO_FIX
            else (
                QAPolicy.VERSION_BUMP
                if strategy == RoutingStrategy.VERSION_BUMP
                else QAPolicy.INITIAL_CODE_WORKAROUND
            )
        ),
        strategy=strategy,
        strategy_stage=(
            SCARemediationStage.OSV_MINIMUM
            if strategy == RoutingStrategy.VERSION_BUMP
            else SCARemediationStage.CODE_WORKAROUND
        ),
        no_fix_stage=(
            NoFixMitigationStage.PACKAGE_REMOVAL
            if fix_plan is not None and fix_plan.status == FixPlanStatus.NO_FIX
            else None
        ),
        selected_version=(
            None
            if fix_plan is not None and fix_plan.status == FixPlanStatus.NO_FIX
            else (fix_plan.fixed_version if fix_plan is not None else None)
        ),
        instruction=instruction,
        status=TaskStatus.PENDING,
        retry_count=0,
        ancestry_depth=0,
    )


def _derive_feedback_by_group(
    target_tasks: list[RemediationTask],
    target_groups: list[VulnerabilityGroup],
    feedback_by_task: Mapping[str, str] | None,
) -> dict[str, str]:
    """Translate task-keyed feedback into group-keyed feedback when possible."""
    feedback_by_group: dict[str, str] = {}
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
    run_id: str
    valid_groups: list[VulnerabilityGroup]
    initial_valid_groups: list[VulnerabilityGroup]
    run_started_at: str

    issues: list[VulnerabilityIssue]
    system_context: SystemContext

    constraints_ledger: Annotated[list[str], operator.add]
    retry_counts: Annotated[dict[str, int], merge_dict_reducer]
    group_strategies: Annotated[dict[str, RoutingStrategy], merge_dict_reducer]
    group_statuses: Annotated[dict[str, GroupRemediationStatus], merge_dict_reducer]
    qa_evaluations: Annotated[dict[str, QAEvaluation], replace_dict_reducer]
    action_summaries: Annotated[list[AgentActionSummary], operator.add]
    retry_diagnostics_by_task: Annotated[dict[str, UpdateRetryDiagnostics], replace_dict_reducer]
    retry_plans_by_task: Annotated[dict[str, SupervisorRetryPlan], replace_dict_reducer]
    workaround_replay_plans_by_task: Annotated[
        dict[str, WorkaroundReplayPlan], replace_dict_reducer
    ]
    attempt_snapshots_by_id: Annotated[dict[str, TaskAttemptSnapshot], merge_dict_reducer]
    workspace_rollback_anchors_by_task: Annotated[dict[str, str], merge_dict_reducer]
    worker_results_by_attempt: Annotated[dict[str, WorkerAttemptResult], merge_dict_reducer]
    qa_results_by_attempt: Annotated[dict[str, QAAttemptResult], merge_dict_reducer]
    scan_evidence_by_task: Annotated[dict[str, ODCScanEvidence], merge_dict_reducer]
    processed_worker_attempt_ids: Annotated[list[str], operator.add]
    processed_qa_attempt_ids: Annotated[list[str], operator.add]
    consistency_events: Annotated[list[StateConsistencyEvent], operator.add]
    state_revision: int
    changed_files: Annotated[list[str], merge_changed_files_reducer]

    # Phase 5 Task Queue (primary orchestration unit)
    task_queue: Annotated[dict[str, RemediationTask], replace_dict_reducer]
    active_target_task_ids: list[str]

    workspace_volume: str | None

    # Supervisor routing fields
    next_routing_step: str
    decision_code: DecisionCode | None
    supervisor_audit: AuditRecord | None
    active_target_group_ids: list[str]
    feedback_by_group: Annotated[dict[str, str], replace_dict_reducer]
    feedback_by_task: Annotated[dict[str, str], replace_dict_reducer]
    supervisor_instructions: str
    eval_status: str
    qa_investigation_report: str
    baseline_scan_identifiers: list[str]
    post_remediation_scan_identifiers: list[str]
    post_remediation_scan_issues: list[VulnerabilityIssue]
    new_vulnerability_identifiers: list[str]
    new_vulnerability_status: str
    final_full_scan_result: FinalFullScanResult | None
    final_full_scan_completed: bool
    triage_required: bool
    initial_triage_status: str
    initial_triage_executed: bool
    triage_reconciliation: dict[str, list[str]]

    status: str
    diff: str
    langsmith_run_id: str
    langsmith_trace_url: str
    trajectory_path: str
    report_markdown: str
    errors: Annotated[list[str], operator.add]


class SubagentState(TypedDict, total=False):
    """
    Ephemeral private state for one specialist subagent run.

    This is the only Phase 5 state that carries localized ReAct messages.
    """

    repo_root: str
    workspace_volume: str

    target_tasks: list[RemediationTask]
    target_groups: list[VulnerabilityGroup]
    feedback_by_group: dict[str, str]
    feedback_by_task: dict[str, str]
    previous_action_summaries_by_task: dict[str, str]
    retry_diagnostics_by_task: dict[str, UpdateRetryDiagnostics]
    workaround_replay_plans_by_task: dict[str, WorkaroundReplayPlan]
    target_attempt_snapshots: dict[str, TaskAttemptSnapshot]

    target_task: RemediationTask
    target_group: VulnerabilityGroup
    attempt_snapshot: TaskAttemptSnapshot | None
    constraints_ledger: list[str]
    previous_feedback: str | None
    current_replay_plan: WorkaroundReplayPlan | None

    messages: Annotated[list[AnyMessage], add_messages]

    action_summaries: list[AgentActionSummary]
    changed_files: Annotated[list[str], operator.add]
    errors: Annotated[list[str], operator.add]


def initial_orchestrator_state(
    repo_root: str,
    valid_groups: list[VulnerabilityGroup],
    issues: list[VulnerabilityIssue] | None = None,
    system_context: SystemContext | None = None,
) -> dict[str, Any]:
    """Build a well-formed initial ``OrchestratorState`` dict."""
    valid_groups = normalize_group_paths(valid_groups, repo_root)
    baseline_scan_identifiers = (
        _scan_identifiers_from_issues(issues)
        if issues is not None
        else _scan_identifiers_from_groups(valid_groups)
    )
    state: dict[str, Any] = {
        "repo_root": repo_root,
        "run_id": "",
        "valid_groups": valid_groups,
        "initial_valid_groups": list(valid_groups),
        "run_started_at": datetime.now(UTC).isoformat(),
        "constraints_ledger": [],
        "retry_counts": {},
        "group_strategies": {},
        "group_statuses": {},
        "qa_evaluations": {},
        "action_summaries": [],
        "retry_diagnostics_by_task": {},
        "retry_plans_by_task": {},
        "attempt_snapshots_by_id": {},
        "workspace_rollback_anchors_by_task": {},
        "worker_results_by_attempt": {},
        "qa_results_by_attempt": {},
        "scan_evidence_by_task": {},
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
        "final_full_scan_result": None,
        "final_full_scan_completed": False,
        "triage_required": False,
        "initial_triage_status": "pending",
        "initial_triage_executed": False,
        "triage_reconciliation": {},
        "diff": "",
        "trajectory_path": "",
        "report_markdown": "",
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
    target_tasks: list[RemediationTask] | list[VulnerabilityGroup],
    target_groups: list[VulnerabilityGroup] | list[str],
    constraints_ledger: list[str] | None = None,
    feedback_by_task: dict[str, str] | None = None,
    feedback_by_group: dict[str, str] | None = None,
    previous_action_summaries_by_task: dict[str, str] | None = None,
    retry_diagnostics_by_task: dict[str, UpdateRetryDiagnostics] | None = None,
    target_attempt_snapshots: dict[str, TaskAttemptSnapshot] | None = None,
) -> dict[str, Any]:
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
        target_tasks_list = [_derive_legacy_task_from_group(group) for group in legacy_groups]
        constraints_list = list(target_groups)
        legacy_feedback = (
            dict(constraints_ledger) if isinstance(constraints_ledger, Mapping) else {}
        )
        feedback_by_group_dict = dict(feedback_by_group or legacy_feedback)
        feedback_by_task_dict = dict(feedback_by_task or feedback_by_group_dict)
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
    target_group: VulnerabilityGroup | list[str],
    constraints_ledger: list[str] | None = None,
    previous_feedback: str | None = None,
    attempt_snapshot: TaskAttemptSnapshot | None = None,
    current_replay_plan: WorkaroundReplayPlan | None = None,
) -> dict[str, Any]:
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
