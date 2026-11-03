"""
state.py - LangGraph state schemas for the AppSec Remediation Engine.

``OrchestratorState`` — supervisor master state for the hub-and-spoke remedy
architecture.
``SubagentState`` — ephemeral private state for specialist subagents.

Reducer notes
-------------
* ``errors`` uses ``operator.add`` in both states so each node can return
  only its new error strings and LangGraph will append them.
* ``messages`` exists only in ``SubagentState`` so subagent ReAct transcripts
  stay isolated from the long-lived supervisor state.
* ``changed_files`` uses an order-preserving set-like reducer so retries and
  bridge nodes cannot duplicate the same path in the final patch projection.
* ``task_queue`` and committed attempt snapshots are the authoritative
  orchestration state and bind worker/QA results to Supervisor authorization.
* All other fields use the default "last writer wins" semantics.
"""

from __future__ import annotations

import hashlib
import logging
import operator
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, TypeVar

from langgraph.graph.message import AnyMessage, add_messages
from semantic_version import Version as SemanticVersion
from typing_extensions import TypedDict

from remediation_engine.contracts.decision_codes import (
    DecisionCode,
)
from remediation_engine.contracts.decision_codes import (
    validate_transition as _validate_transition,
)
from remediation_engine.contracts.schemas import (
    AgentActionSummary,
    AuditRecord,
    FixPlan,
    FixPlanStatus,
    IncrementalScanRecord,
    IssueType,
    MultiPackageAction,
    NoFixMitigationStage,
    ODCScanEvidence,
    PackageScanOccurrence,
    PortfolioPlan,
    QAAttemptResult,
    QAEvaluation,
    QAPolicy,
    RemediationTask,
    RoutingStrategy,
    ScanCoverageResult,
    SCARemediationStage,
    StateConsistencyEvent,
    SupervisorRetryPlan,
    SystemContext,
    TaskAttemptSnapshot,
    TaskScanRecord,
    TaskStatus,
    UpdateRetryDiagnostics,
    VulnerabilityGroup,
    VulnerabilityIssue,
    WorkaroundReplayPlan,
    WorkerAttemptResult,
    model_or_dict_value,
)
from remediation_engine.contracts.solver_models import (
    PortfolioReplanRequest,
    QAPassedWorkspacePrefix,
    SolverRemediationPlan,
)
from remediation_engine.language import ProjectLanguage, resolve_project_language
from remediation_engine.runtime.path_policy import (
    WorkspacePathError,
    normalize_workspace_path,
    repository_relative_path,
)
from remediation_engine.tools.manifest_locator import expand_dependency_ancestry_from_repository

_logger = logging.getLogger(__name__)

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
            old_property_file = localized.version_property_file
            new_property_file = _repo_relative_path(old_property_file, repo_root)
            if old and old != new:
                replacements[str(old)] = new or "unknown-manifest"
            if old_property_file and old_property_file != new_property_file:
                replacements[str(old_property_file)] = new_property_file or "unknown-property-owner"
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
            localized_updates: dict[str, Any] = {
                "manifest_file": new,
                "version_property_file": new_property_file,
            }
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
    maven_cache_volume:
        Run-owned Maven repository volume, separate from the workspace.

    Supervisor memory / outputs
    ---------------------------
    changed_files:
        Repo-relative files successfully modified across subagent runs.
    task_queue:
        Dict mapping task IDs to ``RemediationTask`` records; the primary unit
        of Phase 5 orchestration.
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
    project_language: ProjectLanguage

    issues: list[VulnerabilityIssue]
    system_context: SystemContext

    constraints_ledger: Annotated[list[str], operator.add]
    retry_counts: Annotated[dict[str, int], merge_dict_reducer]
    group_strategies: Annotated[dict[str, RoutingStrategy], merge_dict_reducer]
    qa_evaluations: Annotated[dict[str, QAEvaluation], replace_dict_reducer]
    action_summaries: Annotated[list[AgentActionSummary], operator.add]
    # Candidate files accumulate across worker/QA bridges. Teardown emits a
    # ``ChangedFilesProjection`` marker to replace that candidate ledger with
    # the files that actually produced the final diff.
    changed_files: Annotated[list[str], merge_changed_files_reducer]
    retry_diagnostics_by_task: Annotated[dict[str, UpdateRetryDiagnostics], replace_dict_reducer]
    retry_plans_by_task: Annotated[dict[str, SupervisorRetryPlan], replace_dict_reducer]
    workaround_replay_plans_by_task: Annotated[
        dict[str, WorkaroundReplayPlan], replace_dict_reducer
    ]
    attempt_snapshots_by_id: Annotated[dict[str, TaskAttemptSnapshot], merge_dict_reducer]
    # This is an authoritative projection of immutable task baselines. A
    # merge reducer would leave removed snapshot IDs stale in state.
    workspace_rollback_anchors_by_task: Annotated[dict[str, str], replace_dict_reducer]
    # Latest cumulative workspace state accepted by scoped QA. Its snapshot is
    # retained until teardown so a replan can restore the exact fingerprint.
    qa_passed_workspace_prefix: QAPassedWorkspacePrefix | None
    worker_results_by_attempt: Annotated[dict[str, WorkerAttemptResult], merge_dict_reducer]
    qa_results_by_attempt: Annotated[dict[str, QAAttemptResult], merge_dict_reducer]
    # Counts same-attempt QA reruns caused by inconclusive evidence or an
    # invalid evaluator contract. The Supervisor owns this bounded ledger.
    qa_rechecks_by_attempt: Annotated[dict[str, int], replace_dict_reducer]
    # Counts how many times QA has actually been dispatched for each committed
    # worker attempt. This also bounds retries when QA returns no usable result.
    qa_dispatch_counts_by_attempt: Annotated[dict[str, int], replace_dict_reducer]
    accepted_package_inventory_by_lockfile: Annotated[dict[str, str], replace_dict_reducer]
    current_scan_observations_by_occurrence: Annotated[
        dict[str, PackageScanOccurrence], replace_dict_reducer
    ]
    latest_successful_scan_by_task: Annotated[dict[str, TaskScanRecord], replace_dict_reducer]
    latest_incremental_scan_by_input: Annotated[
        dict[str, IncrementalScanRecord], replace_dict_reducer
    ]
    setup_scan_evidence: ODCScanEvidence | None
    setup_scan_committed: bool
    package_inventory_baseline_complete: bool
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
    active_cluster_id: str | None
    active_dispatch_batch_id: str | None
    active_multi_package_action: MultiPackageAction | None
    delta_isolation_by_cluster: Annotated[dict[str, Any], merge_dict_reducer]

    workspace_volume: str | None
    maven_cache_volume: str | None

    # Supervisor routing fields
    next_routing_step: str
    decision_code: DecisionCode | None
    supervisor_audit: AuditRecord | None
    feedback_by_group: Annotated[dict[str, str], replace_dict_reducer]
    feedback_by_task: Annotated[dict[str, str], replace_dict_reducer]
    supervisor_instructions: str
    eval_status: str
    qa_investigation_report: str
    baseline_scan_complete: bool
    baseline_scan_identifiers: list[str]
    post_remediation_scan_identifiers: list[str]
    post_remediation_scan_issues: list[VulnerabilityIssue]
    new_vulnerability_identifiers: list[str]
    new_vulnerability_status: str
    final_full_scan_result: ScanCoverageResult | None
    final_full_scan_enabled: bool
    triage_required: bool
    post_qa_triage_enabled: bool
    post_qa_retriage_count: int
    post_qa_retriage_limit_reached: bool
    initial_triage_status: str
    initial_triage_executed: bool
    triage_reconciliation: dict[str, list[str]]
    # Post-QA triage records reopened tasks here; the Supervisor replans their
    # groups and refreshes planning-derived task inputs before portfolio work.
    task_ids_requiring_fix_plan_refresh: list[str]
    # Group input fingerprints whose targeted workaround lookup already ran.
    workaround_lookup_fingerprints: Annotated[dict[str, str], replace_dict_reducer]

    status: str
    diff: str
    langsmith_run_id: str
    langsmith_trace_url: str
    trajectory_path: str
    report_markdown: str
    report_path: str | None
    report_status: str
    report_error: str | None
    report_summary_metadata: dict[str, Any] | None
    errors: Annotated[list[str], operator.add]


class SubagentState(TypedDict, total=False):
    """
    Ephemeral private state for one specialist subagent run.

    This is the only Phase 5 state that carries localized ReAct messages.
    """

    repo_root: str
    workspace_volume: str
    project_language: ProjectLanguage
    maven_cache_volume: str | None

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
    project_language: ProjectLanguage | str | None = None,
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
    resolved_project_language = (
        project_language
        if isinstance(project_language, ProjectLanguage)
        else resolve_project_language(
            Path(repo_root),
            project_language
            or (system_context.primary_language if system_context is not None else None),
        )
    )
    valid_groups = normalize_group_paths(valid_groups, repo_root)
    baseline_scan_complete = issues is not None
    baseline_scan_identifiers = (
        _scan_identifiers_from_issues(issues)
        if issues is not None
        else _scan_identifiers_from_groups(valid_groups)
    )
    valid_groups = filter_groups_to_target_packages(valid_groups, normalized_target_packages)
    if issues is not None:
        issues = filter_issues_to_target_packages(issues, normalized_target_packages)
    state: dict[str, Any] = {
        "repo_root": repo_root,
        "run_id": "",
        "valid_groups": valid_groups,
        "initial_valid_groups": list(valid_groups),
        "run_started_at": datetime.now(UTC).isoformat(),
        "target_packages": normalized_target_packages,
        "project_language": resolved_project_language,
        "constraints_ledger": [],
        "retry_counts": {},
        "group_strategies": {},
        "qa_evaluations": {},
        "action_summaries": [],
        "retry_diagnostics_by_task": {},
        "retry_plans_by_task": {},
        "attempt_snapshots_by_id": {},
        "workspace_rollback_anchors_by_task": {},
        "qa_passed_workspace_prefix": None,
        "worker_results_by_attempt": {},
        "qa_results_by_attempt": {},
        "qa_rechecks_by_attempt": {},
        "qa_dispatch_counts_by_attempt": {},
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
        "active_cluster_id": None,
        "active_dispatch_batch_id": None,
        "active_multi_package_action": None,
        "delta_isolation_by_cluster": {},
        "workspace_volume": None,
        "maven_cache_volume": None,
        "status": "pending",
        "next_routing_step": "",
        "feedback_by_group": {},
        "feedback_by_task": {},
        "supervisor_instructions": "",
        "eval_status": "",
        "qa_investigation_report": "",
        "baseline_scan_identifiers": baseline_scan_identifiers,
        "baseline_scan_complete": baseline_scan_complete,
        "package_inventory_baseline_complete": False,
        "accepted_package_inventory_by_lockfile": {},
        "current_scan_observations_by_occurrence": {},
        "latest_successful_scan_by_task": {},
        "latest_incremental_scan_by_input": {},
        "setup_scan_evidence": None,
        "setup_scan_committed": False,
        "post_remediation_scan_identifiers": [],
        "post_remediation_scan_issues": [],
        "new_vulnerability_identifiers": [],
        "new_vulnerability_status": "not_scanned",
        "final_full_scan_result": None,
        "final_full_scan_enabled": False,
        "triage_required": False,
        "post_qa_triage_enabled": True,
        "post_qa_retriage_count": 0,
        "post_qa_retriage_limit_reached": False,
        "initial_triage_status": "pending",
        "initial_triage_executed": False,
        "triage_reconciliation": {},
        "task_ids_requiring_fix_plan_refresh": [],
        "workaround_lookup_fingerprints": {},
        "diff": "",
        "trajectory_path": "",
        "report_markdown": "",
        "report_path": None,
        "report_status": "pending",
        "report_error": None,
        "report_summary_metadata": None,
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
    project_language: ProjectLanguage = ProjectLanguage.NODEJS,
    maven_cache_volume: str | None = None,
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
        "project_language": project_language,
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
        "maven_cache_volume": maven_cache_volume,
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
    project_language: ProjectLanguage = ProjectLanguage.NODEJS,
    maven_cache_volume: str | None = None,
) -> dict[str, Any]:
    """Build a well-formed single-task workaround ``SubagentState`` dict."""
    return {
        "repo_root": repo_root,
        "workspace_volume": workspace_volume,
        "project_language": project_language,
        "target_task": target_task,
        "target_group": target_group,
        "constraints_ledger": list(constraints_ledger or []),
        "previous_feedback": previous_feedback,
        "attempt_snapshot": attempt_snapshot,
        "current_replay_plan": current_replay_plan,
        "maven_cache_volume": maven_cache_volume,
        "messages": [],
        "changed_files": [],
        "errors": [],
    }


# Supervisor-owned task selection and attempt transitions.

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
    """Return retained package plans in the group's stable issue order."""
    candidates = getattr(group, "fix_plan_candidates", None) or []
    pairs: list[tuple[str, FixPlan]] = []
    if not isinstance(candidates, (list, tuple)):
        candidates = []
    for candidate in candidates:
        issue_id = str(_task_value(candidate, "issue_id", ""))
        plan = _task_value(candidate, "plan")
        if issue_id and plan is not None:
            pairs.append((issue_id, plan))
    return pairs


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
    task, and incomplete incremental coverage cannot be reported as a clean run.
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
        coverage_scope = str(
            _task_value(final_scan, "coverage_scope", "full_workspace") or "full_workspace"
        )
        is_incremental = coverage_scope == "incremental_dependency_inputs"
        completed = _task_value(final_scan, "completed")
        coverage_complete = _task_value(final_scan, "coverage_complete")
        scan_status = str(_task_value(final_scan, "status", "")).lower()
        if is_incremental:
            if coverage_complete is not True:
                reasons.append("incremental dependency scan coverage is incomplete")
        elif completed is False or scan_status in {
            "scan_failed",
            "failed",
            "error",
            "timeout",
        }:
            reasons.append("authoritative final scan failed")
        remaining = list(_task_value(final_scan, "remaining_target_identifiers", []) or [])
        if remaining:
            scope_label = "incremental scan" if is_incremental else "authoritative final scan"
            reasons.append(
                f"{scope_label} still contains {len(set(remaining))} target identifier(s)"
            )
        new_identifiers = list(
            _task_value(final_scan, "new_identifiers")
            or state.get("new_vulnerability_identifiers", [])
            or []
        )
        if new_identifiers:
            scope_label = "incremental scans" if is_incremental else "authoritative final scan"
            reasons.append(f"{scope_label} detected {len(set(new_identifiers))} new identifier(s)")

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
    manifest_paths = _group_manifest_paths(group)
    if set(_group_package_managers(group)) == {"maven"}:
        pom_paths = (
            ", ".join(
                path for path in manifest_paths if path.rsplit("/", 1)[-1].casefold() == "pom.xml"
            )
            or "the group-authorized pom.xml paths"
        )
        return (
            "NO_FIX MITIGATION — PACKAGE REMOVAL: Attempt to completely remove "
            f"the exact Maven dependency '{component}' from the application. Inspect only "
            f"these group-authorized pom.xml paths: {pom_paths}. Use "
            "remove_no_fix_dependency to remove its direct dependency declaration; do not "
            "manually edit other POM declarations. Remove imports, source calls, and code "
            "that depends on the package, then validate that it is absent from the resolved "
            "Maven dependency tree and that the application still passes its validation "
            "gates. If the package is transitive and has no safe removable declaration, "
            "report that failure so the supervisor can advance to the vulnerable-code-"
            "removal stage."
        )

    manifests = ", ".join(manifest_paths) or "the group-authorized manifest paths"
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


def is_maven_group(
    group: VulnerabilityGroup | None,
    project_language: ProjectLanguage = ProjectLanguage.NODEJS,
) -> bool:
    """Return whether this run-localized vulnerability group is a Maven target."""
    if group is None or project_language != ProjectLanguage.JAVA:
        return False
    return any(
        (getattr(localized, "package_manager", None) or "").strip().lower() == "maven"
        or (str(getattr(localized.issue, "ecosystem", "") or "").strip().lower() == "maven")
        or str(getattr(localized.issue, "purl", "") or "").strip().lower().startswith("pkg:maven/")
        for localized in group.localized_issues
    ) or any(
        str(getattr(issue, "ecosystem", "") or "").strip().lower() == "maven"
        or str(getattr(issue, "purl", "") or "").strip().lower().startswith("pkg:maven/")
        for issue in group.issues
    )


def _maven_group_dependency_type(group: VulnerabilityGroup) -> str:
    """Use only localized Maven edit targets, defaulting safely to management."""
    for localized in group.localized_issues:
        if localized.declaration_type in {"dependencies", "dependencyManagement"}:
            return localized.declaration_type
    return "dependencyManagement"


def build_initial_remediation_task(
    group: VulnerabilityGroup,
    task_id: str,
    *,
    project_language: ProjectLanguage = ProjectLanguage.NODEJS,
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
    maven_target = is_maven_group(group, project_language)
    has_parent_target = (
        not maven_target
        and strategy == RoutingStrategy.VERSION_BUMP
        and transitive
        and bool(parent_name)
    )
    direct_dependency_type = next(
        (
            localized.declaration_type
            for localized in group.localized_issues
            if localized.declaration_type
        ),
        None if transitive else "dependencies",
    )
    if maven_target and strategy == RoutingStrategy.VERSION_BUMP:
        target_package_name = group.vulnerable_component
        target_dependency_type = _maven_group_dependency_type(group)
    else:
        target_package_name = (
            parent_name
            if has_parent_target
            else (group.vulnerable_component if strategy == RoutingStrategy.VERSION_BUMP else None)
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
        SCARemediationStage.CODE_WORKAROUND
        if strategy == RoutingStrategy.CODE_WORKAROUND
        else SCARemediationStage.OSV_MINIMUM
        if maven_target or has_parent_target or not transitive
        else SCARemediationStage.PACKAGE_OVERRIDE
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
        parent_package_name=None if maven_target else parent_name,
        parent_package_version=None if maven_target else parent_version,
        parent_minimum_version=None,
        no_fix_stage=no_fix_stage,
        # A fix-plan version is a security floor, not a registry-verified
        # worker target. The Supervisor fills this field only after the active
        # task's candidate lookup has completed and the exact dispatch input
        # has been committed.
        selected_version=None,
        selected_plan_issue_ids=list(selection.issue_ids),
        instruction=instruction,
        status=TaskStatus.PENDING,
        retry_count=0,
        ancestry_depth=0,
    )


UPDATE_DISPATCH_LIMIT: int = 1

QA_DISPATCH_LIMIT: int = 30

_TERMINAL_STATUSES = TERMINAL_TASK_STATUSES

_WORKABLE_STATUSES = frozenset({TaskStatus.PENDING, TaskStatus.NEEDS_RETRY})

_SEVERITY_RANK: dict[str, int] = {
    "critical": 0,
    "high": 1,
    "medium": 2,
    "low": 3,
}


def instruction_digest(instruction: str) -> str:
    """Return the stable SHA-256 digest for a committed worker instruction."""
    normalized = " ".join((instruction or "").split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


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
    if task.parent_package_name and task.strategy_stage not in {
        SCARemediationStage.CODE_WORKAROUND,
        SCARemediationStage.PYPI_LATEST,
    }:
        # npm transitive updates retain the parent-first override stage. PyPI
        # has no override stage, so exhaustion at PYPI_LATEST may pivot.
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


def _build_consistency_event(
    *,
    error_code: str,
    task_id: str | None,
    expected_attempt_id: str | None,
    received_attempt_id: str | None,
    action: str,
    details: str,
) -> StateConsistencyEvent:
    # ``deferred`` was used by the registry-pending branch before it was
    # added to the typed event contract.  Keep this compatibility boundary
    # fail-closed: a deferred proposal is a rejected dispatch, and must never
    # crash the Supervisor while constructing its diagnostic event.
    if action == "deferred":
        action = "rejected"
    return StateConsistencyEvent(
        error_code=error_code,
        task_id=task_id,
        expected_attempt_id=expected_attempt_id,
        received_attempt_id=received_attempt_id,
        action=action,
        details=details,
    )


def _dedupe_consistency_events(events: list[StateConsistencyEvent]) -> list[StateConsistencyEvent]:
    seen: set[tuple[str | None, str | None, str]] = set()
    result: list[StateConsistencyEvent] = []
    for event in events:
        key = (event.task_id, event.received_attempt_id, event.error_code)
        if key not in seen:
            seen.add(key)
            result.append(event)
    return result


_ATTEMPT_INPUT_FIELDS = frozenset(
    {
        "task_revision",
        "strategy_stage",
        "selected_version",
        "exhausted_update_path",
        "instruction",
        "strategy",
        "no_fix_stage",
        "qa_policy",
        "target_package_name",
        "target_dependency_type",
        "parent_minimum_version",
    }
)


def _commit_task_transition(
    task_queue: dict[str, RemediationTask],
    task_id: str,
    *,
    updates: dict[str, Any],
    close_attempt: bool = False,
    clear_selected_version: bool = False,
    allow_breaking_change_pivot: bool = False,
    allow_verified_absent_target_pass: bool = False,
    consistency_events: list[StateConsistencyEvent] | None = None,
) -> RemediationTask | None:
    """Commit one coherent supervisor transition for a task.

    ``task_queue`` is the authoritative projection. This helper makes the
    transition explicit and ensures that any change to worker-input fields is
    either paired with a new task revision or closes the old attempt first.
    Worker successes that are waiting for QA intentionally do not use this
    helper for a status-only update: their current snapshot remains valid QA
    input. Every replan, surrender, terminalization, and pivot does use it.
    ``allow_breaking_change_pivot`` is retained for the Supervisor-owned
    breaking-change pivot path. ``allow_verified_absent_target_pass`` is
    reserved for workerless SCA completion after accepted absence proof.
    """
    task = task_queue.get(task_id)
    if task is None:
        return None

    if "status" in updates:
        current_status = task.status
        try:
            new_status = (
                updates["status"]
                if isinstance(updates["status"], TaskStatus)
                else TaskStatus(updates["status"])
            )
        except (TypeError, ValueError):
            new_status = None
        verified_absent_transition_allowed = (
            allow_verified_absent_target_pass
            and task.current_attempt_id is None
            and current_status in {TaskStatus.PENDING, TaskStatus.NEEDS_RETRY}
            and new_status == TaskStatus.QA_PASSED
        )
        pivot_transition_allowed = (
            allow_breaking_change_pivot
            and current_status == TaskStatus.NEEDS_RETRY
            and new_status == TaskStatus.QA_PASSED
        )
        if new_status is None or (
            new_status != current_status
            and not _validate_transition(current_status, new_status)
            and not pivot_transition_allowed
            and not verified_absent_transition_allowed
        ):
            event = _build_consistency_event(
                error_code="INVALID_TRANSITION",
                task_id=task_id,
                expected_attempt_id=task.current_attempt_id,
                received_attempt_id=None,
                action="rejected",
                details=(
                    f"Rejected transition {current_status.value} -> "
                    f"{updates['status']!s}; it is not in VALID_TRANSITIONS."
                ),
            )
            if consistency_events is not None:
                consistency_events.append(event)
            _logger.error(
                "supervisor: rejected invalid transition %s -> %s for task '%s'.",
                current_status.value,
                updates["status"],
                task_id,
            )
            return task
        updates = {**updates, "status": new_status}

    committed_updates = dict(updates)
    # Terminal status is a complete transition, not just a status projection.
    # Clear all future worker input here so callers cannot leave a selected
    # version or live attempt for the validator to repair later.
    committed_status = committed_updates.get("status")
    if isinstance(committed_status, TaskStatus) and committed_status in _TERMINAL_STATUSES:
        close_attempt = True
        clear_selected_version = True
    input_changed = any(
        field in committed_updates and committed_updates[field] != getattr(task, field)
        for field in _ATTEMPT_INPUT_FIELDS
    )
    if close_attempt:
        committed_updates["current_attempt_id"] = None
        if task.current_attempt_id is not None:
            input_changed = True
    if clear_selected_version:
        committed_updates["selected_version"] = None
        if task.selected_version is not None:
            input_changed = True

    if not committed_updates:
        return task

    if input_changed:
        committed_updates["task_revision"] = task.task_revision + 1

    committed_task = task.model_copy(update=committed_updates)
    task_queue[task_id] = committed_task
    return committed_task


def _validate_committed_state(
    task_queue: dict[str, RemediationTask],
    snapshots_by_id: dict[str, TaskAttemptSnapshot],
    retry_plans_by_task: dict[str, SupervisorRetryPlan],
    retry_diagnostics_by_task: dict[str, UpdateRetryDiagnostics],
    active_target_task_ids: list[str],
    next_node: str,
) -> tuple[list[StateConsistencyEvent], list[str]]:
    """Validate the state projection that will be handed to the next node."""
    events: list[StateConsistencyEvent] = []
    errors: list[str] = []

    # Reconcile every task before validating routing. Worker and QA bridges
    # cannot repair planner-owned fields. A matching task/attempt/revision
    # snapshot may repair corrupted input fields; foreign identity or revision
    # evidence is rejected without mutating the task. A terminal task has no
    # authorized future worker input, so its dangling attempt is detached.
    for task_id, task in list(task_queue.items()):
        if task.status in _TERMINAL_STATUSES and (
            task.current_attempt_id is not None or task.selected_version is not None
        ):
            expected_attempt_id = task.current_attempt_id
            task_queue[task_id] = task.model_copy(
                update={
                    "current_attempt_id": None,
                    "selected_version": None,
                    "task_revision": task.task_revision + 1,
                }
            )
            events.append(
                _build_consistency_event(
                    error_code="TERMINAL_TASK_FIELDS_NORMALIZED",
                    task_id=task_id,
                    expected_attempt_id=expected_attempt_id,
                    received_attempt_id=expected_attempt_id,
                    action="repaired",
                    details=(
                        "Terminal task cannot retain a current worker attempt or "
                        "a dispatchable selected version."
                    ),
                )
            )
            continue
        if not task.current_attempt_id:
            continue
        snapshot = snapshots_by_id.get(task.current_attempt_id)
        if snapshot is None:
            errors.append(f"supervisor: task {task_id} references a missing attempt snapshot.")
            continue

        if (
            snapshot.attempt_id != task.current_attempt_id
            or snapshot.task_id != task_id
            or snapshot.task_id != task.task_id
        ):
            detail = "Active task references a snapshot with foreign task or attempt identity."
            errors.append(f"supervisor: {detail}")
            events.append(
                _build_consistency_event(
                    error_code="TASK_SNAPSHOT_IDENTITY_REJECTED",
                    task_id=task_id,
                    expected_attempt_id=task.current_attempt_id,
                    received_attempt_id=snapshot.attempt_id,
                    action="rejected",
                    details=detail,
                )
            )
            continue
        if snapshot.task_revision != task.task_revision:
            detail = (
                "Active task revision differs from its committed attempt snapshot; "
                "snapshot repair was refused."
            )
            errors.append(f"supervisor: {detail}")
            events.append(
                _build_consistency_event(
                    error_code="TASK_SNAPSHOT_REVISION_REJECTED",
                    task_id=task_id,
                    expected_attempt_id=task.current_attempt_id,
                    received_attempt_id=snapshot.attempt_id,
                    action="rejected",
                    details=detail,
                )
            )
            continue

        snapshot_matches = (
            snapshot.strategy_stage == task.strategy_stage
            and snapshot.no_fix_stage == task.no_fix_stage
            and snapshot.qa_policy == task.qa_policy
            and snapshot.selected_version == task.selected_version
            and snapshot.target_package_name == task.target_package_name
            and snapshot.target_dependency_type == task.target_dependency_type
            and snapshot.parent_minimum_version == task.parent_minimum_version
            and snapshot.selected_plan_issue_ids == task.selected_plan_issue_ids
            and snapshot.instruction == task.instruction
            and snapshot.instruction_digest == instruction_digest(task.instruction)
            and (
                (
                    snapshot.dispatch_node == "update_subagent"
                    and task.strategy == RoutingStrategy.VERSION_BUMP
                )
                or (
                    snapshot.dispatch_node == "workaround_subagent"
                    and task.strategy == RoutingStrategy.CODE_WORKAROUND
                )
                or snapshot.dispatch_node == "qa_critic"
            )
        )
        if snapshot_matches:
            continue

        task_queue[task_id] = task.model_copy(
            update={
                "current_attempt_id": snapshot.attempt_id,
                "strategy_stage": snapshot.strategy_stage,
                "no_fix_stage": snapshot.no_fix_stage,
                "qa_policy": snapshot.qa_policy,
                "selected_version": snapshot.selected_version,
                "target_package_name": snapshot.target_package_name,
                "target_dependency_type": snapshot.target_dependency_type,
                "parent_minimum_version": snapshot.parent_minimum_version,
                "selected_plan_issue_ids": list(snapshot.selected_plan_issue_ids),
                "instruction": snapshot.instruction,
            }
        )
        events.append(
            _build_consistency_event(
                error_code="TASK_SNAPSHOT_REPAIRED",
                task_id=task_id,
                expected_attempt_id=task.current_attempt_id,
                received_attempt_id=snapshot.attempt_id,
                action="repaired",
                details="Active task fields were restored from its committed attempt snapshot.",
            )
        )

    for task_id in list(retry_plans_by_task):
        task = task_queue.get(task_id)
        plan = retry_plans_by_task[task_id]
        if task is None:
            retry_plans_by_task.pop(task_id, None)
            continue
        if task.status in _TERMINAL_STATUSES:
            retry_plans_by_task.pop(task_id, None)
            events.append(
                _build_consistency_event(
                    error_code="TERMINAL_TASK_PLAN_CLEARED",
                    task_id=task_id,
                    expected_attempt_id=task.current_attempt_id,
                    received_attempt_id=None,
                    action="repaired",
                    details="Removed a retry plan from a terminal task.",
                )
            )
            continue
        if plan.action == "retry_update" and (
            plan.selected_version is None or task.exhausted_update_path
        ):
            retry_plans_by_task.pop(task_id, None)
            events.append(
                _build_consistency_event(
                    error_code="INVALID_RETRY_PLAN_CLEARED",
                    task_id=task_id,
                    expected_attempt_id=task.current_attempt_id,
                    received_attempt_id=None,
                    action="replanned",
                    details="Cleared a retry plan that cannot be dispatched safely.",
                )
            )
            continue
        if plan.action == "retry_update" and (
            plan.source_task_revision > task.task_revision
            or plan.source_task_revision < max(0, task.task_revision - 1)
            or plan.strategy_stage != task.strategy_stage
            or plan.selected_version != task.selected_version
            or plan.exact_instruction != task.instruction
            or plan.exhausted_update_path != task.exhausted_update_path
        ):
            retry_plans_by_task.pop(task_id, None)
            events.append(
                _build_consistency_event(
                    error_code="RETRY_PLAN_TASK_CONTRADICTION",
                    task_id=task_id,
                    expected_attempt_id=task.current_attempt_id,
                    received_attempt_id=None,
                    action="replanned",
                    details="Cleared a retry plan that disagreed with the committed task queue.",
                )
            )
            continue

    for task_id in active_target_task_ids:
        task = task_queue.get(task_id)
        if task is None:
            continue
        if task.status in _TERMINAL_STATUSES:
            errors.append(f"supervisor: terminal task {task_id} remained active.")
            events.append(
                _build_consistency_event(
                    error_code="TERMINAL_TASK_ACTIVE",
                    task_id=task_id,
                    expected_attempt_id=task.current_attempt_id,
                    received_attempt_id=None,
                    action="ignored",
                    details="Terminal task removed from dispatch projection.",
                )
            )
            continue
        if task.current_attempt_id is None:
            errors.append(f"supervisor: active task {task_id} has no attempt snapshot.")
            events.append(
                _build_consistency_event(
                    error_code="ACTIVE_TASK_WITHOUT_ATTEMPT",
                    task_id=task_id,
                    expected_attempt_id=None,
                    received_attempt_id=None,
                    action="replanned",
                    details="Active target cannot be dispatched without a committed snapshot.",
                )
            )
            continue
        snapshot = snapshots_by_id.get(task.current_attempt_id)
        if snapshot is None:
            errors.append(f"supervisor: active task {task_id} references missing attempt.")
            continue
        if next_node == "update_subagent" and snapshot.dispatch_node == "update_subagent":
            maven_mode = snapshot.maven_target_operation is not None
            normalize_version = (
                (lambda value: str(value).strip())
                if maven_mode
                else (lambda value: str(value).strip().lstrip("vV"))
            )
            allowed_versions = {
                normalize_version(value)
                for value in snapshot.allowed_target_versions
                if str(value).strip()
            }
            selected_version = (
                normalize_version(task.selected_version) if task.selected_version else None
            )
            if not allowed_versions or (
                selected_version is None or selected_version not in allowed_versions
            ):
                detail = (
                    "Active update attempt lacks a non-empty candidate authorization "
                    "or its selected version is outside that authorization."
                )
                errors.append(f"supervisor: {detail}")
                events.append(
                    _build_consistency_event(
                        error_code="UPDATE_ATTEMPT_WITHOUT_CANDIDATE",
                        task_id=task_id,
                        expected_attempt_id=task.current_attempt_id,
                        received_attempt_id=task.current_attempt_id,
                        action="rejected",
                        details=detail,
                    )
                )
                continue
        if task.qa_policy is None or snapshot.qa_policy is None:
            errors.append(f"supervisor: active task {task_id} has missing QA policy provenance.")
            events.append(
                _build_consistency_event(
                    error_code="MISSING_QA_POLICY_PROVENANCE",
                    task_id=task_id,
                    expected_attempt_id=task.current_attempt_id,
                    received_attempt_id=task.current_attempt_id,
                    action="replanned",
                    details=(
                        "Active task and committed attempt must both carry a "
                        "supervisor-owned QA policy."
                    ),
                )
            )
            continue
        if (
            snapshot.task_revision != task.task_revision
            or snapshot.strategy_stage != task.strategy_stage
            or snapshot.no_fix_stage != task.no_fix_stage
            or snapshot.qa_policy != task.qa_policy
            or snapshot.selected_version != task.selected_version
            or snapshot.selected_plan_issue_ids != task.selected_plan_issue_ids
            or snapshot.instruction != task.instruction
            or snapshot.instruction_digest != instruction_digest(task.instruction)
        ):
            errors.append(f"supervisor: task {task_id} disagrees with its attempt snapshot.")
            events.append(
                _build_consistency_event(
                    error_code="TASK_SNAPSHOT_CONTRADICTION",
                    task_id=task_id,
                    expected_attempt_id=task.current_attempt_id,
                    received_attempt_id=task.current_attempt_id,
                    action="replanned",
                    details="Task projection and committed attempt snapshot differ.",
                )
            )
        if next_node == "update_subagent" and task.exhausted_update_path:
            errors.append(f"supervisor: exhausted task {task_id} cannot route to update.")

    for task_id, diagnostics in retry_diagnostics_by_task.items():
        task = task_queue.get(task_id)
        if task is None:
            continue
        current_snapshot = snapshots_by_id.get(task.current_attempt_id or "")
        maven_mode = current_snapshot is not None and (
            current_snapshot.maven_target_operation is not None
        )
        normalize_version = (
            (lambda value: str(value).strip().lower())
            if maven_mode
            else (lambda value: str(value).strip().lstrip("vV").lower())
        )
        if diagnostics.selected_version != task.selected_version:
            diagnostics = diagnostics.model_copy(update={"selected_version": task.selected_version})
            retry_diagnostics_by_task[task_id] = diagnostics
            events.append(
                _build_consistency_event(
                    error_code="DIAGNOSTICS_PROJECTION_REPAIRED",
                    task_id=task_id,
                    expected_attempt_id=task.current_attempt_id,
                    received_attempt_id=diagnostics.committed_attempt_id,
                    action="repaired",
                    details="Planner-owned selected version restored from task state.",
                )
            )
        if (
            next_node == "update_subagent"
            and task.selected_version
            and normalize_version(task.selected_version)
            in {normalize_version(version) for version in diagnostics.attempted_versions if version}
        ):
            errors.append(f"supervisor: selected version for {task_id} was already attempted.")
    return events, errors


def reconcile_phase5_state_before_teardown(
    state: OrchestratorState,
) -> dict[str, Any]:
    """Apply the final supervisor state barrier before teardown.

    Teardown is a cleanup operation, not a routing decision.  It must receive
    a terminal task projection with no retry plans, active targets, or current
    worker inputs.  This function deliberately performs no LLM calls and uses
    the same validator as the supervisor return path, so direct teardown
    callers and graph executions share the same invariant.
    """
    task_queue: dict[str, RemediationTask] = {
        task_id: task.model_copy() for task_id, task in dict(state.get("task_queue", {})).items()
    }
    snapshots_by_id: dict[str, TaskAttemptSnapshot] = dict(state.get("attempt_snapshots_by_id", {}))
    retry_plans_by_task: dict[str, SupervisorRetryPlan] = dict(state.get("retry_plans_by_task", {}))
    retry_diagnostics_by_task: dict[str, UpdateRetryDiagnostics] = dict(
        state.get("retry_diagnostics_by_task", {})
    )
    prior_events = list(state.get("consistency_events", []) or [])
    prior_event_keys = {
        (event.task_id, event.received_attempt_id, event.error_code) for event in prior_events
    }

    # A pivot child owns the remediation outcome. Repair stale parent status
    # projections before teardown so a historical QA_PASSED parent cannot
    # conceal an unresolved child in the final task queue.
    pivot_repair_events: list[StateConsistencyEvent] = []
    children_by_parent: dict[str, list[RemediationTask]] = {}
    for task in task_queue.values():
        if task.parent_task_id:
            children_by_parent.setdefault(task.parent_task_id, []).append(task)
    for parent_id, children in sorted(children_by_parent.items()):
        parent = task_queue.get(parent_id)
        if parent is None or parent.status == TaskStatus.PIVOTED:
            continue
        task_queue[parent_id] = parent.model_copy(
            update={
                "status": TaskStatus.PIVOTED,
                "current_attempt_id": None,
                "selected_version": None,
                "task_revision": parent.task_revision + 1,
            }
        )
        pivot_repair_events.append(
            _build_consistency_event(
                error_code="PIVOT_PARENT_STATUS_REPAIRED",
                task_id=parent_id,
                expected_attempt_id=parent.current_attempt_id,
                received_attempt_id=children[0].current_attempt_id,
                action="repaired",
                details=(
                    "Parent task status was normalized to PIVOTED because a child task "
                    "owns the current remediation outcome."
                ),
            )
        )

    events, errors = _validate_committed_state(
        task_queue,
        snapshots_by_id,
        retry_plans_by_task,
        retry_diagnostics_by_task,
        [],
        "teardown",
    )
    new_events = [
        event
        for event in _dedupe_consistency_events([*pivot_repair_events, *events])
        if (event.task_id, event.received_attempt_id, event.error_code) not in prior_event_keys
    ]
    prior_errors = set(state.get("errors", []) or [])
    new_errors = list(dict.fromkeys(error for error in errors if error not in prior_errors))

    return {
        "task_queue": task_queue,
        "retry_plans_by_task": retry_plans_by_task,
        "retry_diagnostics_by_task": retry_diagnostics_by_task,
        "workspace_rollback_anchors_by_task": {},
        "active_target_task_ids": [],
        "next_routing_step": "teardown",
        "state_revision": int(state.get("state_revision", 0)) + 1,
        "consistency_events": new_events,
        "errors": new_errors,
    }


def _update_worker_task_ids(
    task_queue: dict[str, RemediationTask],
    retry_diagnostics_by_task: dict[str, UpdateRetryDiagnostics],
    preferred_ids: list[str] | None = None,
    limit: int | None = UPDATE_DISPATCH_LIMIT,
    group_by_id: dict[str, VulnerabilityGroup] | None = None,
) -> list[str]:
    task_ids = _dispatchable_task_ids_for_status(
        task_queue,
        set(_WORKABLE_STATUSES),
        preferred_ids=preferred_ids,
        strategy=RoutingStrategy.VERSION_BUMP,
        group_by_id=group_by_id,
    )
    dispatchable = [
        task_id
        for task_id in task_ids
        if task_queue[task_id].current_attempt_id is None
        if not _is_exhausted_update_pivot_candidate(
            task_queue[task_id],
            retry_diagnostics_by_task.get(task_id),
        )
    ]
    if limit is not None:
        return dispatchable[:limit]
    return dispatchable


def _normalize_target_task_ids_for_node(
    next_node: str,
    target_task_ids: list[str],
    task_queue: dict[str, RemediationTask],
    retry_diagnostics_by_task: dict[str, UpdateRetryDiagnostics] | None = None,
    group_by_id: dict[str, VulnerabilityGroup] | None = None,
    allow_cluster: bool = False,
) -> list[str]:
    """Clamp returned active targets to the lifecycle state accepted by next_node."""
    retry_diagnostics_by_task = retry_diagnostics_by_task or {}
    if next_node == "qa_critic":
        return _qa_ready_task_ids(
            task_queue,
            preferred_ids=target_task_ids,
            group_by_id=group_by_id,
            limit=(None if allow_cluster else QA_DISPATCH_LIMIT),
        )
    if next_node == "update_subagent":
        return _update_worker_task_ids(
            task_queue,
            retry_diagnostics_by_task,
            preferred_ids=target_task_ids,
            limit=(None if allow_cluster else UPDATE_DISPATCH_LIMIT),
            group_by_id=group_by_id,
        )
    if next_node == "workaround_subagent":
        return [
            task_id
            for task_id in _dispatchable_task_ids_for_status(
                task_queue,
                set(_WORKABLE_STATUSES),
                preferred_ids=target_task_ids,
                strategy=RoutingStrategy.CODE_WORKAROUND,
                group_by_id=group_by_id,
            )
            if task_queue[task_id].current_attempt_id is None
        ][:1]
    return []


def _normalize_qa_evaluations_for_tasks(
    qa_evaluations: dict[str, QAEvaluation],
    task_queue: dict[str, RemediationTask],
    active_target_task_ids: list[str],
) -> dict[str, QAEvaluation]:
    """Keep only evaluations explicitly keyed by an active task ID."""
    active_ids = set(active_target_task_ids)
    normalized: dict[str, QAEvaluation] = {}
    for task_id, evaluation in qa_evaluations.items():
        if (
            task_id not in active_ids
            or task_id not in task_queue
            or evaluation.task_id != task_id
            or task_id in normalized
        ):
            continue
        normalized[task_id] = evaluation
    return normalized


def _constraint_entry_for_task(
    task: RemediationTask,
    group: VulnerabilityGroup,
) -> str:
    """Build a deterministic constraints-ledger entry for a QA-passed task."""
    component = (group.vulnerable_component or task.parent_group_id).strip()
    fix_plan = select_package_fix_plan(group, task.strategy).plan

    if task.strategy == RoutingStrategy.VERSION_BUMP:
        fixed_version = (fix_plan.fixed_version if fix_plan else None) or "unknown"
        return f"{component}: keep resolved version at {fixed_version}"

    return f"{component}: preserve validated security workaround"


def _missing_retry_revised_instructions(
    next_node: str,
    target_task_ids: list[str],
    revised_instructions: dict[str, str],
    task_queue: dict[str, RemediationTask],
) -> list[str]:
    """Return retry-bound update targets that are missing exact revised instructions."""
    if next_node != "update_subagent":
        return []

    missing: list[str] = []
    for task_id in target_task_ids:
        task = task_queue.get(task_id)
        if task is None or task.status != TaskStatus.NEEDS_RETRY:
            continue
        instruction = revised_instructions.get(task_id, "").strip()
        if not instruction:
            missing.append(task_id)
            continue
        # Retry workers are execution-only: a retry instruction is invalid
        # unless it carries both the strategy stage and a concrete semver.
        stage = getattr(task.strategy_stage, "value", task.strategy_stage)
        stage_marker = f"strategy stage {stage}".lower()
        if stage_marker not in instruction.lower() or not re.search(
            r"(?<!\d)v?\d+\.\d+\.\d+(?!\d)", instruction
        ):
            missing.append(task_id)
    return missing


def _no_fix_failure_transition(
    task: RemediationTask,
    group: VulnerabilityGroup | None,
    *,
    evaluation: QAEvaluation | None = None,
    failure_feedback: str | None = None,
) -> tuple[dict[str, Any], bool]:
    """Build the supervisor-owned transition after a failed NO_FIX attempt.

    Returns a task update mapping and whether the next attempt must reset the
    task-local workspace to the package-removal stage baseline.
    """
    updates = advance_no_fix_stage(task)
    next_stage = updates.get("no_fix_stage")
    if next_stage == NoFixMitigationStage.VULNERABLE_CODE_REMOVAL:
        updates["qa_policy"] = QAPolicy.NO_FIX_CODE_REMOVAL
    reset_workspace = next_stage == NoFixMitigationStage.VULNERABLE_CODE_REMOVAL
    if reset_workspace:
        retry_task = task.model_copy(update=updates)
        updates["instruction"] = build_no_fix_retry_instruction(
            retry_task,
            group,
            evaluation=evaluation,
            failure_feedback=failure_feedback,
        )
    return updates, reset_workspace


def _reset_no_fix_replay_plan(
    replay_plan: WorkaroundReplayPlan | None,
) -> WorkaroundReplayPlan | None:
    """Return a replay plan that restores the stage baseline without replaying edits."""
    if replay_plan is None:
        return None
    return replay_plan.model_copy(
        update={
            "successful_edit_sets": [],
            "validated_files": [],
            "validation_calls": 0,
            "per_gate_results": {},
            "final_selected_targeted_test": None,
            "original_to_alternative_test_mapping": {},
            "alternative_test_mapping_evidence": {},
            "alternative_test_mapping_details": {},
        }
    )
