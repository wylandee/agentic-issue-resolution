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
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, TypeVar

from langgraph.graph.message import AnyMessage, add_messages
from typing_extensions import TypedDict

from remediation_engine.contracts.decision_codes import DecisionCode
from remediation_engine.contracts.schemas import (
    AgentActionSummary,
    FinalFullScanResult,
    IssueType,
    MultiPackageAction,
    ODCScanEvidence,
    PortfolioPlan,
    QAAttemptResult,
    QAEvaluation,
    RemediationTask,
    RoutingStrategy,
    SupervisorRetryPlan,
    SystemContext,
    TaskAttemptSnapshot,
    UpdateRetryDiagnostics,
    VulnerabilityGroup,
    VulnerabilityIssue,
    WorkaroundReplayPlan,
    WorkerAttemptResult,
)
from remediation_engine.contracts.solver_models import (
    PortfolioReplanRequest,
    SolverRemediationPlan,
)
from remediation_engine.contracts.supervisor_phases import AuditRecord
from remediation_engine.runtime.path_policy import (
    WorkspacePathError,
    normalize_workspace_path,
    repository_relative_path,
)
from remediation_engine.tools.manifest_locator import expand_dependency_ancestry_from_repository

K = TypeVar("K")
V = TypeVar("V")

DEVELOPMENT_ENVIRONMENTS = frozenset({"dev", "development"})


def normalize_target_packages(values: Sequence[str] | None) -> list[str]:
    """Return a deterministic, validated package allowlist.

    Args:
        values: Package names supplied by a development-only scoped run.

    Returns:
        Sorted, de-duplicated package names. ``None`` becomes an empty list,
        which means that the engine should process the full repository.

    Raises:
        ValueError: If a package name is not a non-empty string.
    """
    if isinstance(values, str):
        raise ValueError("target_packages must be a sequence of package names")
    normalized: set[str] = set()
    for value in values or ():
        if not isinstance(value, str) or not value.strip():
            raise ValueError("target_packages must contain non-empty package names")
        normalized.add(value.strip())
    return sorted(normalized)


def validate_target_package_scope(
    target_packages: Sequence[str] | None,
    system_context: SystemContext | None,
) -> list[str]:
    """Validate that package scoping is explicitly limited to development.

    Args:
        target_packages: Requested package allowlist.
        system_context: Caller-supplied environment metadata.

    Returns:
        Normalized package names.

    Raises:
        ValueError: If a non-empty package scope is requested without a
            development environment label.
    """
    normalized = normalize_target_packages(target_packages)
    environment = (system_context.environment if system_context else "") or ""
    if normalized and environment.strip().lower() not in DEVELOPMENT_ENVIRONMENTS:
        raise ValueError(
            "target_packages is development-only; set system_context.environment to "
            "'development' or 'dev'"
        )
    return normalized


def _group_package_identities(group: VulnerabilityGroup) -> set[str]:
    """Collect package names that identify one vulnerability group."""
    identities = {
        value.strip()
        for value in (
            group.vulnerable_component,
            group.parent_package_name,
            *(issue.package_name for issue in group.issues or []),
            *(localized.issue.package_name for localized in group.localized_issues or []),
        )
        if isinstance(value, str) and value.strip()
    }
    return identities


def filter_groups_to_target_packages(
    groups: Sequence[VulnerabilityGroup],
    target_packages: Sequence[str] | None,
) -> list[VulnerabilityGroup]:
    """Keep only SCA groups that belong to a requested package scope.

    Non-SCA groups are retained because package scoping controls dependency
    portfolio discovery, not source-code remediation.
    """
    scope = set(normalize_target_packages(target_packages))
    if not scope:
        return list(groups)
    return [
        group
        for group in groups
        if group.issue_type != IssueType.SCA or bool(_group_package_identities(group) & scope)
    ]


def filter_issues_to_target_packages(
    issues: Sequence[VulnerabilityIssue],
    target_packages: Sequence[str] | None,
) -> list[VulnerabilityIssue]:
    """Keep dependency findings inside a requested package scope.

    SAST findings remain available to the normal source-remediation path.
    """
    scope = set(normalize_target_packages(target_packages))
    if not scope:
        return list(issues)
    return [
        issue
        for issue in issues
        if issue.issue_type != IssueType.SCA or (issue.package_name or "").strip() in scope
    ]


class ChangedFilesProjection(list[str]):
    """Authoritative final changed-file projection for the state reducer.

    Worker nodes emit ordinary lists, which are accumulated across retries.
    Teardown emits this marker after comparing candidate paths with the
    workspace and host baseline. The reducer then replaces the historical
    candidate ledger with the files that actually appear in the final diff.
    """

    def __init__(self, values: list[str] | tuple[str, ...] = ()) -> None:
        """Initialize a projection from repository-relative paths."""
        super().__init__(values)


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

    Supervisor-owned projections are complete snapshots, not patches. A merge
    reducer cannot represent deletion, so this reducer preserves the exact
    task, attempt, retry, and result maps emitted for the next graph node.
    """
    return dict(right or {})


def merge_changed_files_reducer(
    left: list[str] | None,
    right: list[str] | ChangedFilesProjection | None,
) -> list[str]:
    """Merge changed-file projections while normalizing and de-duplicating paths."""
    values = (
        list(right or [])
        if isinstance(right, ChangedFilesProjection)
        else [
            *(left or []),
            *(right or []),
        ]
    )
    merged: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            raise WorkspacePathError(f"changed_files reducer rejected non-string path {value!r}")
        try:
            path = normalize_workspace_path(value)
        except WorkspacePathError as exc:
            raise WorkspacePathError(
                f"changed_files reducer rejected path {value!r}: {exc}"
            ) from exc
        if path not in seen:
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
    """Collect scanner identifiers from the groups used for graph seeding."""
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
    return repository_relative_path(value, repo_root)


def normalize_group_paths(
    groups: list[VulnerabilityGroup],
    repo_root: str,
) -> list[VulnerabilityGroup]:
    """Return groups with safe repository-relative manifest and issue paths.

    Nested localized issues are normalized at the graph boundary before tasks
    or attempt snapshots are built. Paths outside ``repo_root`` are discarded
    rather than passed to worker tools.
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
            odc_file_path = (
                repository_relative_path(
                    localized.issue.file_path
                    or (localized.issue.raw_payload or {}).get("filePath", ""),
                    repo_root,
                )
                or ""
            )
            if new is None:
                ancestry, versions = (
                    list(localized.dependency_ancestry),
                    dict(localized.dependency_versions),
                )
            else:
                ancestry, versions = expand_dependency_ancestry_from_repository(
                    Path(repo_root),
                    new,
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
        fix_plan_candidates = list(group.fix_plan_candidates or [])
        if replacements:
            replacement_items = tuple(replacements.items())

            def _rewrite_instruction(
                instruction: str,
                replacement_items: tuple[tuple[str, str], ...] = replacement_items,
            ) -> str:
                rewritten = instruction or ""
                for old, new in replacement_items:
                    rewritten = rewritten.replace(str(old), new).replace(
                        str(old).replace("\\", "/"), new
                    )
                return rewritten

            if fix_plan is not None:
                fix_plan = fix_plan.model_copy(
                    update={"instruction": _rewrite_instruction(fix_plan.instruction)}
                )
            fix_plan_candidates = [
                candidate.model_copy(
                    update={
                        "plan": candidate.plan.model_copy(
                            update={"instruction": _rewrite_instruction(candidate.plan.instruction)}
                        )
                    }
                )
                for candidate in fix_plan_candidates
            ]
        if (
            group_id == group.group_id
            and file_path == group.file_path
            and file_paths == list(group.file_paths or [])
            and localized_issues == list(group.localized_issues or [])
            and expanded_group_ancestry == list(group.dependency_ancestry)
            and expanded_group_versions == dict(group.dependency_versions)
            and fix_plan is group.fix_plan
            and fix_plan_candidates == list(group.fix_plan_candidates or [])
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
                    "fix_plan_candidates": fix_plan_candidates,
                }
            )
        )
    return normalized


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
    # Empty means the normal full-repository portfolio. Non-empty values are
    # accepted only for development runs and limit synthetic dependency
    # discovery to the requested package closure.
    target_packages: list[str]

    issues: list[VulnerabilityIssue]
    system_context: SystemContext

    constraints_ledger: Annotated[list[str], operator.add]
    retry_counts: Annotated[dict[str, int], merge_dict_reducer]
    group_strategies: Annotated[dict[str, RoutingStrategy], merge_dict_reducer]
    qa_evaluations: Annotated[dict[str, QAEvaluation], replace_dict_reducer]
    action_summaries: Annotated[list[AgentActionSummary], operator.add]
    retry_diagnostics_by_task: Annotated[dict[str, UpdateRetryDiagnostics], replace_dict_reducer]
    retry_plans_by_task: Annotated[dict[str, SupervisorRetryPlan], replace_dict_reducer]
    workaround_replay_plans_by_task: Annotated[
        dict[str, WorkaroundReplayPlan], replace_dict_reducer
    ]
    attempt_snapshots_by_id: Annotated[dict[str, TaskAttemptSnapshot], merge_dict_reducer]
    # This is an authoritative projection of immutable task baselines. A
    # merge reducer would leave removed snapshot IDs stale in state.
    workspace_rollback_anchors_by_task: Annotated[dict[str, str], replace_dict_reducer]
    worker_results_by_attempt: Annotated[dict[str, WorkerAttemptResult], merge_dict_reducer]
    qa_results_by_attempt: Annotated[dict[str, QAAttemptResult], merge_dict_reducer]
    scan_evidence_by_task: Annotated[dict[str, ODCScanEvidence], merge_dict_reducer]
    processed_worker_attempt_ids: Annotated[list[str], operator.add]
    processed_qa_attempt_ids: Annotated[list[str], operator.add]
    # Phase 5 Task Queue (primary orchestration unit)
    task_queue: Annotated[dict[str, RemediationTask], replace_dict_reducer]
    active_target_task_ids: list[str]
    portfolio_plan: PortfolioPlan | None
    portfolio_solver_plan: SolverRemediationPlan | None
    portfolio_iteration: int
    portfolio_replan_request: PortfolioReplanRequest | None
    # Counts replan requests by a stable diagnostic-reason key.  The outer
    # portfolio node uses this bounded ledger to fail closed on no-progress
    # replan loops without making ordinary one-off replans terminal.
    portfolio_replan_history: Annotated[dict[str, int], replace_dict_reducer]
    portfolio_dirty: bool
    # Legacy dictionary retained at the migration boundary. New callers should
    # use ``portfolio_replan_request``.
    portfolio_escalation: dict[str, Any] | None
    active_cluster_id: str | None
    active_dispatch_batch_id: str | None
    active_multi_package_action: MultiPackageAction | None
    delta_isolation_by_cluster: Annotated[dict[str, Any], merge_dict_reducer]

    workspace_volume: str | None

    # Supervisor routing fields
    next_routing_step: str
    decision_code: DecisionCode | None
    supervisor_audit: AuditRecord | None
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
    final_scan_workspace_fingerprint: str | None
    previous_final_scan_workspace_fingerprint: str | None
    triage_required: bool
    post_qa_retriage_count: int
    post_qa_retriage_limit_reached: bool
    initial_triage_status: str
    initial_triage_executed: bool
    triage_reconciliation: dict[str, list[str]]

    status: str
    diff: str
    langsmith_run_id: str
    langsmith_trace_url: str
    trajectory_path: str
    report_markdown: str
    report_path: str | None
    report_status: str
    report_error: str | None
    errors: Annotated[list[str], operator.add]


class SubagentState(TypedDict, total=False):
    """
    Ephemeral private state for one specialist subagent run.

    This is the only Phase 5 state that carries localized ReAct messages.
    """

    repo_root: str
    workspace_volume: str

    target_tasks: list[RemediationTask]
    active_cluster_id: str | None
    dispatch_batch_id: str | None
    multi_package_action: MultiPackageAction | None
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
    target_packages: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build a well-formed initial ``OrchestratorState`` dict.

    ``target_packages`` is an opt-in development allowlist.  An empty list
    preserves the production/default behavior of processing the entire
    repository.
    """
    normalized_target_packages = validate_target_package_scope(
        target_packages,
        system_context,
    )
    valid_groups = normalize_group_paths(valid_groups, repo_root)
    valid_groups = filter_groups_to_target_packages(valid_groups, normalized_target_packages)
    if issues is not None:
        issues = filter_issues_to_target_packages(issues, normalized_target_packages)
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
        "target_packages": normalized_target_packages,
        "constraints_ledger": [],
        "retry_counts": {},
        "group_strategies": {},
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
        "portfolio_plan": None,
        "portfolio_solver_plan": None,
        "portfolio_iteration": 0,
        "portfolio_replan_request": None,
        "portfolio_replan_history": {},
        "portfolio_dirty": True,
        "portfolio_escalation": None,
        "active_cluster_id": None,
        "active_dispatch_batch_id": None,
        "active_multi_package_action": None,
        "delta_isolation_by_cluster": {},
        "workspace_volume": None,
        "status": "pending",
        "next_routing_step": "",
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
        "final_scan_workspace_fingerprint": None,
        "previous_final_scan_workspace_fingerprint": None,
        "triage_required": False,
        "post_qa_retriage_count": 0,
        "post_qa_retriage_limit_reached": False,
        "initial_triage_status": "pending",
        "initial_triage_executed": False,
        "triage_reconciliation": {},
        "diff": "",
        "trajectory_path": "",
        "report_markdown": "",
        "report_path": None,
        "report_status": "pending",
        "report_error": None,
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
    target_tasks: Sequence[RemediationTask],
    target_groups: Sequence[VulnerabilityGroup],
    constraints_ledger: Sequence[str] = (),
    feedback_by_task: Mapping[str, str] | None = None,
    feedback_by_group: Mapping[str, str] | None = None,
    previous_action_summaries_by_task: Mapping[str, str] | None = None,
    retry_diagnostics_by_task: Mapping[str, UpdateRetryDiagnostics] | None = None,
    target_attempt_snapshots: Mapping[str, TaskAttemptSnapshot] | None = None,
    active_cluster_id: str | None = None,
    dispatch_batch_id: str | None = None,
    multi_package_action: MultiPackageAction | None = None,
    messages: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Build the initial update-worker state from committed task inputs."""
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
        "active_cluster_id": active_cluster_id,
        "dispatch_batch_id": dispatch_batch_id,
        "multi_package_action": multi_package_action,
        "constraints_ledger": constraints_list,
        "messages": list(messages or []),
        "changed_files": [],
        "errors": [],
    }


def initial_workaround_subagent_state(
    repo_root: str,
    workspace_volume: str,
    target_task: RemediationTask,
    target_group: VulnerabilityGroup,
    constraints_ledger: list[str] | None = None,
    previous_feedback: str | None = None,
    attempt_snapshot: TaskAttemptSnapshot | None = None,
    current_replay_plan: WorkaroundReplayPlan | None = None,
) -> dict[str, Any]:
    """Build a well-formed single-task workaround ``SubagentState`` dict."""
    return {
        "repo_root": repo_root,
        "workspace_volume": workspace_volume,
        "target_task": target_task,
        "target_group": target_group,
        "constraints_ledger": list(constraints_ledger or []),
        "previous_feedback": previous_feedback,
        "attempt_snapshot": attempt_snapshot,
        "current_replay_plan": current_replay_plan,
        "messages": [],
        "changed_files": [],
        "errors": [],
    }
