"""
graph.py - LangGraph remediation orchestrator for the current Phase 5 runtime.

Phase 5 graph topology (hub-and-spoke)
---------------------------------------
::

    START
      |
    initial_triage (one preprocessing pass)
      | triage_completed / skipped -> workspace_builder
      | failed | no_work -> teardown
    workspace_builder
      | workspace_ready / failed -> teardown
    supervisor  <-----------------------------------+
      |                                            |
      +-> update_subagent ----------------------->-+
      |                                            |
      +-> workaround_subagent ------------------->-+
      |                                            |
      +-> qa_critic ------------------------------>+
      |                                            |
      +-> triage (post-QA reconciliation) -------->+
      |                                            |
      +-> final_full_scan ------------------------>+
      |
      +-> teardown -> report -> END

Public API:
``build_orchestrator_graph()``
``orchestrator_engine``
``run_orchestrator(...)``

Typed request/result models are exposed from ``remediation_engine.api`` so
callers do not need to construct LangGraph state directly.
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph

from remediation_engine.contracts.accessors import model_or_dict_value
from remediation_engine.contracts.schemas import (
    IssueSource,
    SystemContext,
    TaskStatus,
    VulnerabilityGroup,
    VulnerabilityIssue,
)
from remediation_engine.orchestration._qa_runtime import group_target_identifiers
from remediation_engine.orchestration.graph_wrappers import (
    _create_workspace_attempt_snapshot,
    _dispatch_boundary_rejection,
    _finalize_qa_workspace_snapshot,
    _finalize_worker_workspace_snapshot,
    _finish_all_workspace_rollback_anchors,
    _finish_parent_workspace_rollback_anchors,
    _finish_workspace_attempt_snapshot,
    _finish_workspace_rollback_anchors,
    _has_partial_update_success,
    _parent_workspace_rollback_anchors,
    _qa_workspace_rollback_anchor_updates,
    _restore_retained_workspace_anchors,
    _restore_workspace_snapshot,
    _worker_attempts_succeeded,
    _workspace_rollback_anchor_ids,
    _workspace_snapshot_id,
    run_qa_critic_from_orchestrator,
    run_update_subagent_from_orchestrator,
    run_workaround_subagent_from_orchestrator,
)
from remediation_engine.orchestration.langsmith_config import (
    build_phase5_runnable_config,
    resolve_phase5_trace_url,
)
from remediation_engine.orchestration.qa_critic import (
    run_final_full_scan_node,
    run_qa_critic_node,
)
from remediation_engine.orchestration.report_node import finalize_report, run_report_node
from remediation_engine.orchestration.runtime_context import (
    get_bound_runtime_settings,
    get_runtime_settings,
    use_runtime_settings,
)
from remediation_engine.orchestration.state import (
    OrchestratorState,
    initial_orchestrator_state,
    normalize_group_paths,
)
from remediation_engine.orchestration.supervisor_node import (
    instruction_digest,
    run_supervisor_node,
    supervisor_router,
)
from remediation_engine.orchestration.task_utils import (
    TERMINAL_TASK_STATUSES,
    build_initial_remediation_task,
    task_group_lineage,
)
from remediation_engine.orchestration.teardown_node import run_teardown_node
from remediation_engine.orchestration.trajectory_exporter import (
    TrajectoryRecorder,
    build_phase5_trajectory_path,
    export_phase5_trajectory,
    invoke_with_trajectory,
    use_trajectory_recorder,
)
from remediation_engine.orchestration.update_subagent import run_update_subagent_node
from remediation_engine.orchestration.workaround_subagent import run_workaround_subagent_node
from remediation_engine.orchestration.workspace_builder import run_workspace_builder_node
from remediation_engine.runtime.sandbox_mgr import DockerSandbox
from remediation_engine.settings import AppSettings
from remediation_engine.triage.pipeline import run_triage_pipeline

log = logging.getLogger(__name__)

__all__ = [
    "build_orchestrator_graph",
    "orchestrator_engine",
    "post_qa_triage_node",
    "route_after_triage",
    "route_after_workspace_builder",
    "run_orchestrator",
    "triage_node",
    # Compatibility exports for the extracted wrapper implementation.
    "_create_workspace_attempt_snapshot",
    "_dispatch_boundary_rejection",
    "_finalize_qa_workspace_snapshot",
    "_finalize_worker_workspace_snapshot",
    "_finish_all_workspace_rollback_anchors",
    "_finish_parent_workspace_rollback_anchors",
    "_finish_workspace_attempt_snapshot",
    "_finish_workspace_rollback_anchors",
    "_has_partial_update_success",
    "_parent_workspace_rollback_anchors",
    "_qa_workspace_rollback_anchor_updates",
    "_restore_retained_workspace_anchors",
    "_restore_workspace_snapshot",
    "_worker_attempts_succeeded",
    "_workspace_rollback_anchor_ids",
    "_workspace_snapshot_id",
    "run_qa_critic_from_orchestrator",
    "run_update_subagent_from_orchestrator",
    "run_workaround_subagent_from_orchestrator",
    "DockerSandbox",
    "instruction_digest",
    "run_qa_critic_node",
    "run_update_subagent_node",
    "run_workaround_subagent_node",
]


# ---------------------------------------------------------------------------
# Phase 5 triage node and routing
# ---------------------------------------------------------------------------


def triage_node(state: OrchestratorState) -> dict[str, Any]:
    """Run the one-time preprocessing triage pass."""
    # ``valid_groups`` is the explicit compatibility contract for callers
    # that already performed preprocessing (for example, the cached-group
    # Juice Shop driver).  Do not silently replace that caller-selected
    # scope with a second full triage pass.
    if state.get("valid_groups"):
        log.info("triage_node: pre-triaged groups supplied; skipping preprocessing triage.")
        return {
            "status": "triage_skipped",
            "initial_triage_status": "skipped_preprocessed_groups",
            "initial_triage_executed": False,
        }

    issues = state.get("issues")
    system_context = state.get("system_context")
    repo_root = state.get("repo_root")

    if not issues or not system_context:
        log.info("triage_node: issues or system_context not found, skipping triage.")
        return {
            "status": "triage_skipped",
            "initial_triage_status": "skipped_missing_input",
            "initial_triage_executed": False,
        }

    log.info("triage_node: running triage on %d issues.", len(issues))
    settings = get_bound_runtime_settings()

    try:

        def triage_call() -> Any:
            if settings is None:
                return run_triage_pipeline(issues, system_context, repo_root)
            return run_triage_pipeline(
                issues,
                system_context,
                repo_root,
                settings=settings,
            )

        results = invoke_with_trajectory(
            "triage.pipeline",
            triage_call,
            {
                "issue_count": len(issues),
                "repo_root": repo_root,
            },
            run_type="chain",
        )
        valid_groups = [group for group, result in results if result.is_valid]
        valid_groups = normalize_group_paths(valid_groups, repo_root)
        log.info("triage_node: produced %d valid groups.", len(valid_groups))

        if not valid_groups:
            return {
                "valid_groups": [],
                "initial_valid_groups": [],
                "status": "triage_completed_no_work",
                "initial_triage_status": "completed_no_work",
                "initial_triage_executed": True,
            }

        return {
            "valid_groups": valid_groups,
            "initial_valid_groups": valid_groups,
            "status": "triage_completed",
            "initial_triage_status": "completed",
            "initial_triage_executed": True,
        }
    except Exception as exc:
        log.exception("triage_node: triage pipeline raised")
        return {
            "valid_groups": [],
            "status": "failed",
            "initial_triage_status": "failed",
            "initial_triage_executed": True,
            "errors": [f"triage_node raised: {exc}"],
        }


def _stable_issue_fingerprint(issue: VulnerabilityIssue) -> str:
    """Return an issue fingerprint that ignores generated ingestion metadata."""
    payload = issue.model_dump(mode="json")
    payload.pop("id", None)
    payload.pop("ingested_at", None)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _stable_group_fingerprint(group: VulnerabilityGroup) -> str:
    """Compare meaningful group content while ignoring volatile triage metadata."""
    payload = {
        "group_id": group.group_id,
        "issue_type": group.issue_type.value,
        "vulnerable_component": group.vulnerable_component,
        "file_path": group.file_path,
        "file_paths": sorted(group.file_paths or []),
        "cve_ids": sorted(group.cve_ids or []),
        "ghsa_ids": sorted(group.ghsa_ids or []),
        "versions": sorted(group.versions or []),
        "dependency_ancestry": list(group.dependency_ancestry or []),
        "dependency_versions": dict(sorted((group.dependency_versions or {}).items())),
        "parent_package_name": group.parent_package_name,
        "parent_package_version": group.parent_package_version,
        "parent_declaration_type": group.parent_declaration_type,
        "parent_contexts": [
            context.model_dump(mode="json") for context in group.parent_contexts or []
        ],
        "sources": sorted(source.value for source in (group.sources or [])),
        "issues": sorted(_stable_issue_fingerprint(issue) for issue in (group.issues or [])),
        "fix_plan": group.fix_plan.model_dump(mode="json") if group.fix_plan else None,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _post_triage_issue_input(
    state: OrchestratorState,
) -> list[VulnerabilityIssue] | None:
    """Build the current issue universe from the parseable QA scan snapshot."""
    if "post_remediation_scan_issues" not in state:
        return None

    post_scan_issues = list(state.get("post_remediation_scan_issues") or [])
    baseline_issues = list(state.get("issues") or [])
    retained_non_odc = [issue for issue in baseline_issues if issue.source != IssueSource.ODC]

    # Skip-triage callers may omit the full initial issue set.  Preserve any
    # non-ODC findings carried by the supplied groups in that compatibility
    # mode, while the post-remediation ODC snapshot remains authoritative for
    # dependency findings.
    if not baseline_issues:
        seen_issue_fingerprints = {_stable_issue_fingerprint(issue) for issue in retained_non_odc}
        for group in state.get("valid_groups", []) or []:
            for issue in group.issues or []:
                if issue.source == IssueSource.ODC:
                    continue
                fingerprint = _stable_issue_fingerprint(issue)
                if fingerprint not in seen_issue_fingerprints:
                    retained_non_odc.append(issue)
                    seen_issue_fingerprints.add(fingerprint)

    return retained_non_odc + post_scan_issues


_worker_result_value = model_or_dict_value


def _accepted_remediation_task_ids(
    state: OrchestratorState,
    task_ids: set[str] | None = None,
) -> set[str]:
    """Return tasks whose changed workspace was accepted by QA.

    Worker status is provisional. The final scan accepts only the latest
    worker result associated with the task's committed attempt, together with
    a passing task-keyed QA evaluation and matching attempt provenance. A
    stale changed-file projection or failed attempt cannot prove remediation.
    """
    worker_results = state.get("worker_results_by_attempt", {}) or {}
    qa_results = state.get("qa_results_by_attempt", {}) or {}
    qa_evaluations = state.get("qa_evaluations", {}) or {}
    attempt_snapshots = state.get("attempt_snapshots_by_id", {}) or {}
    accepted: set[str] = set()

    # Results are retained for auditability, so do not let an old accepted
    # attempt make a later no-op retry look like fresh material work. Select
    # only the latest worker result for each task; attempt number and revision
    # are committed Supervisor fields and provide a deterministic ordering.
    latest_by_task: dict[str, tuple[tuple[int, int, int, str], str, Any]] = {}
    for attempt_key, result in worker_results.items():
        raw_task_id = _worker_result_value(result, "task_id", "")
        result_task_id = str(raw_task_id) if raw_task_id else ""
        if task_ids is not None and result_task_id not in task_ids:
            continue
        if not result_task_id:
            continue
        attempt_id = str(_worker_result_value(result, "attempt_id", "") or attempt_key)
        snapshot = attempt_snapshots.get(attempt_id)

        def _as_int(value: Any) -> int:
            try:
                return int(value or 0)
            except (TypeError, ValueError):
                return 0

        order = (
            _as_int(_worker_result_value(result, "task_revision")),
            _as_int(_worker_result_value(snapshot, "attempt_number")),
            _as_int(_worker_result_value(snapshot, "state_revision")),
            attempt_id,
        )
        previous = latest_by_task.get(result_task_id)
        if previous is None or order >= previous[0]:
            latest_by_task[result_task_id] = (order, attempt_id, result)

    for _order, attempt_id, result in latest_by_task.values():
        raw_task_id = _worker_result_value(result, "task_id", "")
        result_task_id = str(raw_task_id) if raw_task_id else ""
        changed_files = _worker_result_value(result, "changed_files", []) or []
        if not any(isinstance(path, str) and path.strip() for path in changed_files):
            continue

        qa_result = qa_results.get(attempt_id) if attempt_id else None
        evaluation = _worker_result_value(qa_result, "evaluation")
        if evaluation is None and result_task_id:
            evaluation = qa_evaluations.get(result_task_id)
        if evaluation is not None:
            if bool(_worker_result_value(evaluation, "passed", False)):
                accepted.add(result_task_id)
            continue

        # Compatibility callers may not emit a QA envelope. Only accept the
        # worker's own validated result in that legacy case; a non-success
        # result can never make a final scan reopen a group.
        status = _worker_result_value(result, "status")
        status_value = str(getattr(status, "value", status)).casefold()
        diagnostics = _worker_result_value(result, "execution_diagnostics")
        if status_value in {"success", "qa_passed"} and bool(
            _worker_result_value(diagnostics, "validation_passed", False)
        ):
            accepted.add(result_task_id)

    return accepted


def _final_scan_has_material_remediation_change(
    state: OrchestratorState,
    group_id: str | None = None,
) -> bool:
    """Return whether a final-scan target has an accepted material change.

    Final-scan findings are not enough to reopen work. A group must have an
    accepted worker change associated with its task lineage. Fingerprint
    comparison remains a compatibility fallback for old manually constructed
    states that have no attempt envelopes.

    Args:
        state: Current orchestrator state after the final full scan.
        group_id: Optional group whose task lineage is being evaluated. When
            omitted, the helper checks whether any accepted remediation changed
            the workspace.

    Returns:
        ``True`` when the supplied group (or any group) has material accepted
        remediation since the prior final scan; otherwise ``False``.
    """
    task_ids: set[str] | None = None
    task_queue = state.get("task_queue", {}) or {}
    if group_id is not None:
        task_ids = {
            str(task_id)
            for task in task_group_lineage(task_queue, group_id)
            if (task_id := _worker_result_value(task, "task_id"))
        }

    accepted_task_ids = _accepted_remediation_task_ids(state, task_ids)
    if accepted_task_ids:
        return True

    # A state with worker/QA envelopes is current-format state. Fail closed
    # for that state: a global fingerprint change from another group must not
    # reopen this group.
    has_attempt_evidence = bool(
        state.get("worker_results_by_attempt") or state.get("qa_results_by_attempt")
    )
    if has_attempt_evidence:
        return False

    if "final_scan_workspace_fingerprint" not in state:
        return False
    previous = state.get("previous_final_scan_workspace_fingerprint")
    if previous is None:
        # A first final scan has no prior fingerprint to compare. Without an
        # accepted attempt envelope, there is no authoritative proof that
        # this group's workspace changed.
        return False
    current = state.get("final_scan_workspace_fingerprint")
    return current is not None and current != previous


def _reconcile_triaged_groups(
    state: OrchestratorState,
    candidate_groups: list[VulnerabilityGroup],
) -> tuple[list[VulnerabilityGroup], dict[str, list[str]]]:
    """Reuse unchanged groups and retain active removed groups for QA handoff."""
    previous_groups = list(state.get("valid_groups", []) or [])
    previous_by_id = {group.group_id: group for group in previous_groups}
    task_queue = state.get("task_queue", {}) or {}
    active_task_ids = set(state.get("active_target_task_ids", []) or [])

    reused: list[str] = []
    changed: list[str] = []
    added: list[str] = []
    reappeared: list[str] = []
    result: list[VulnerabilityGroup] = []
    candidate_ids = {group.group_id for group in candidate_groups}

    for candidate in candidate_groups:
        previous = previous_by_id.get(candidate.group_id)
        if previous is not None:
            if _stable_group_fingerprint(previous) == _stable_group_fingerprint(candidate):
                result.append(previous)
                reused.append(candidate.group_id)
            else:
                result.append(candidate)
                changed.append(candidate.group_id)
            continue

        existing_task = next(
            (task for task in task_queue.values() if task.parent_group_id == candidate.group_id),
            None,
        )
        result.append(candidate)
        if existing_task is not None:
            reappeared.append(candidate.group_id)
        else:
            added.append(candidate.group_id)

    retained: list[str] = []
    for previous in previous_groups:
        if previous.group_id in candidate_ids:
            continue
        matching_tasks = [
            (task_id, task)
            for task_id, task in task_queue.items()
            if task.parent_group_id == previous.group_id
        ]
        if any(
            task_id in active_task_ids or task.status not in TERMINAL_TASK_STATUSES
            for task_id, task in matching_tasks
        ):
            result.append(previous)
            retained.append(previous.group_id)

    reconciliation = {
        "reused_group_ids": sorted(reused),
        "changed_group_ids": sorted(changed),
        "new_group_ids": sorted(added),
        "reappeared_group_ids": sorted(reappeared),
        "retained_removed_group_ids": sorted(retained),
        "removed_group_ids": sorted(set(previous_by_id) - candidate_ids - set(retained)),
    }
    return sorted(result, key=lambda group: group.group_id), reconciliation


def post_qa_triage_node(state: OrchestratorState) -> dict[str, Any]:
    """Re-triage the complete parseable post-remediation scan snapshot."""
    settings = get_runtime_settings()
    bound_settings = get_bound_runtime_settings()
    disable_retriage = settings.remedy_disable_post_qa_triage
    if disable_retriage or not state.get("triage_required"):
        return {
            "status": "triage_skipped",
            "triage_required": False,
            "triage_reconciliation": {},
            "active_target_task_ids": [],
        }

    if state.get("new_vulnerability_status") == "scan_failed":
        log.info("post_qa_triage_node: scan failed; preserving current groups.")
        return {
            "status": "triage_skipped",
            "triage_required": False,
            "triage_reconciliation": {},
            "active_target_task_ids": [],
        }

    issues = _post_triage_issue_input(state)
    if issues is None:
        return {
            "status": "triage_skipped",
            "triage_required": False,
            "triage_reconciliation": {},
            "active_target_task_ids": [],
        }

    retriage_count = int(state.get("post_qa_retriage_count", 0) or 0)
    retriage_limit = settings.remedy_retriage_limit
    if settings.remedy_retriage_limit_enabled and retriage_count >= retriage_limit:
        message = (
            "Development post-QA re-triage limit reached "
            f"({retriage_limit}); stopping further re-triage."
        )
        log.warning("post_qa_triage_node: %s", message)
        return {
            "status": "retriage_limit_reached",
            "triage_required": False,
            "post_qa_retriage_count": retriage_count,
            "post_qa_retriage_limit_reached": True,
            "triage_reconciliation": {},
            "active_target_task_ids": [],
            "errors": [message],
        }

    retriage_count += 1

    system_context = state.get("system_context") or SystemContext()
    repo_root = state.get("repo_root")
    log.info("post_qa_triage_node: re-triaging %d current issues.", len(issues))

    try:
        if bound_settings is None:
            results = run_triage_pipeline(issues, system_context, repo_root)
        else:
            results = run_triage_pipeline(
                issues,
                system_context,
                repo_root,
                settings=bound_settings,
            )
        candidate_groups = [group for group, triage_result in results if triage_result.is_valid]
        valid_groups, reconciliation = _reconcile_triaged_groups(
            state,
            candidate_groups,
        )
        task_queue = dict(state.get("task_queue", {}) or {})
        changed_group_ids = set(reconciliation["changed_group_ids"])
        changed_group_ids.update(reconciliation["reappeared_group_ids"])
        changed_group_ids.update(reconciliation["new_group_ids"])
        final_scan = state.get("final_full_scan_result")
        final_scan_identifiers = set(
            (
                final_scan.get("remaining_target_identifiers", [])
                if isinstance(final_scan, dict)
                else getattr(final_scan, "remaining_target_identifiers", [])
            )
            or []
        )
        final_scan_reopened_group_ids = sorted(
            group.group_id
            for group in state.get("valid_groups", []) or []
            if group_target_identifiers(group) & final_scan_identifiers
        )
        changed_group_ids.update(final_scan_reopened_group_ids)
        if final_scan_reopened_group_ids:
            reconciliation["final_scan_reopened_group_ids"] = final_scan_reopened_group_ids
        if final_scan_reopened_group_ids:
            skipped_reopens = {
                group_id
                for group_id in final_scan_reopened_group_ids
                if not _final_scan_has_material_remediation_change(state, group_id)
            }
            changed_group_ids.difference_update(skipped_reopens)
            if skipped_reopens:
                reconciliation["final_scan_reopen_skipped_no_material_change_group_ids"] = sorted(
                    skipped_reopens
                )
        groups_by_id = {group.group_id: group for group in valid_groups}
        reopened_task_ids: set[str] = set()
        preserved_unfixable_task_ids: set[str] = set()
        preserved_pivoted_task_ids: set[str] = set()
        prior_qa_evaluations = dict(state.get("qa_evaluations", {}) or {})
        for task_id, task in list(task_queue.items()):
            if task.parent_group_id not in changed_group_ids:
                continue
            if task.status == TaskStatus.UNFIXABLE:
                preserved_unfixable_task_ids.add(task_id)
                continue
            if task.status == TaskStatus.PIVOTED:
                # This parent is an audit record for work delegated to its
                # child. Re-triage must not resurrect it as a fresh task.
                preserved_pivoted_task_ids.add(task_id)
                continue
            group = groups_by_id.get(task.parent_group_id)
            if group is None:
                continue
            fresh_task = build_initial_remediation_task(group, task_id)
            task_queue[task_id] = task.model_copy(
                update={
                    "task_revision": task.task_revision + 1,
                    "current_attempt_id": None,
                    "strategy": fresh_task.strategy,
                    "qa_policy": fresh_task.qa_policy,
                    "strategy_stage": fresh_task.strategy_stage,
                    "selected_version": fresh_task.selected_version,
                    "exhausted_update_path": False,
                    "instruction": fresh_task.instruction,
                    "status": TaskStatus.PENDING,
                    "retry_count": 0,
                }
            )
            reopened_task_ids.add(task_id)

        # A previous QA result belongs to the pre-retriage task revision. It
        # remains available in the attempt history, but must not close the
        # newly reopened task.
        qa_evaluations = {
            key: value
            for key, value in prior_qa_evaluations.items()
            if key not in reopened_task_ids
        }
        log.info(
            "post_qa_triage_node: produced %d valid groups (%d reused, %d new, %d changed).",
            len(valid_groups),
            len(reconciliation["reused_group_ids"]),
            len(reconciliation["new_group_ids"]),
            len(reconciliation["changed_group_ids"]),
        )
        work_reopened = bool(
            reopened_task_ids
            or reconciliation["new_group_ids"]
            or reconciliation["reappeared_group_ids"]
        )
        if preserved_unfixable_task_ids:
            reconciliation["preserved_unfixable_task_ids"] = sorted(preserved_unfixable_task_ids)
        if preserved_pivoted_task_ids:
            reconciliation["preserved_pivoted_task_ids"] = sorted(preserved_pivoted_task_ids)
        return {
            "valid_groups": valid_groups,
            "status": "triage_completed" if valid_groups else "triage_completed_no_work",
            "triage_required": False,
            "post_qa_retriage_count": retriage_count,
            "triage_reconciliation": reconciliation,
            "task_queue": task_queue,
            "qa_evaluations": qa_evaluations,
            "active_target_task_ids": [],
            "final_full_scan_completed": False
            if work_reopened
            else state.get("final_full_scan_completed", False),
            "final_full_scan_result": None
            if work_reopened
            else state.get("final_full_scan_result"),
        }
    except Exception as exc:  # noqa: BLE001
        log.exception("post_qa_triage_node: triage pipeline raised")
        return {
            "status": "triage_failed",
            "triage_required": False,
            "post_qa_retriage_count": retriage_count,
            "triage_reconciliation": {},
            "errors": [f"post_qa_triage_node raised: {exc}"],
        }


def route_after_triage(state: OrchestratorState) -> str:
    """Route Phase 5 flow after the triage node."""
    status = state.get("status")
    if status in ("triage_completed_no_work", "failed"):
        return "teardown"
    return "workspace_builder"


# ---------------------------------------------------------------------------
# Phase 5 routing
# ---------------------------------------------------------------------------


def route_after_workspace_builder(state: OrchestratorState) -> str:
    """Route Phase 5 flow after the workspace builder node."""
    if state.get("status") == "workspace_ready":
        return "supervisor"
    return "teardown"


# ---------------------------------------------------------------------------
# Phase 5 graph construction
# ---------------------------------------------------------------------------


def build_orchestrator_graph():
    """Compile and return the Phase 5 orchestrator StateGraph."""
    workflow = StateGraph(OrchestratorState)

    # ``initial_triage`` is the one preprocessing pass.  The node named
    # ``triage`` is reserved for Supervisor-dispatched post-QA re-triage.
    workflow.add_node("initial_triage", triage_node)
    workflow.add_node("triage", post_qa_triage_node)
    workflow.add_node("workspace_builder", run_workspace_builder_node)
    workflow.add_node("supervisor", run_supervisor_node)
    workflow.add_node("update_subagent", run_update_subagent_from_orchestrator)
    workflow.add_node("workaround_subagent", run_workaround_subagent_from_orchestrator)
    workflow.add_node("qa_critic", run_qa_critic_from_orchestrator)
    workflow.add_node("final_full_scan", run_final_full_scan_node)
    workflow.add_node("teardown", run_teardown_node)
    workflow.add_node("report", run_report_node)

    workflow.add_edge(START, "initial_triage")
    workflow.add_conditional_edges("initial_triage", route_after_triage)
    workflow.add_conditional_edges("workspace_builder", route_after_workspace_builder)
    workflow.add_conditional_edges("supervisor", supervisor_router)
    workflow.add_edge("update_subagent", "supervisor")
    workflow.add_edge("workaround_subagent", "supervisor")
    workflow.add_edge("qa_critic", "supervisor")
    workflow.add_edge("triage", "supervisor")
    workflow.add_edge("final_full_scan", "supervisor")
    workflow.add_edge("teardown", "report")
    workflow.add_edge("report", END)

    return workflow.compile()


orchestrator_engine = build_orchestrator_graph()


def run_orchestrator(
    repo_root: str,
    valid_groups: list[VulnerabilityGroup],
    issues: list[VulnerabilityIssue] | None = None,
    system_context: SystemContext | None = None,
    settings: AppSettings | None = None,
) -> OrchestratorState:
    """Run the Phase 5 graph and return its final state.

    Args:
        repo_root: Absolute path to the repository workspace.
        valid_groups: Initial caller-selected actionable groups.
        issues: Optional canonical scanner findings used by initial triage.
        system_context: Optional deployment context for triage and QA.
        settings: Optional validated application settings used by report
            finalization; environment settings are used when omitted.

    Returns:
        The final graph state, including trajectory and report metadata when
        those artifacts were successfully produced.

    Raises:
        BaseException: Re-raises an orchestration failure after best-effort
            trajectory export, preserving the existing public behavior.
    """
    settings = settings or AppSettings.from_env()
    initial_state = initial_orchestrator_state(
        repo_root=repo_root,
        valid_groups=valid_groups,
        issues=issues,
        system_context=system_context,
    )
    config, run_id = build_phase5_runnable_config(repo_root, valid_groups, settings=settings)
    recorder = TrajectoryRecorder()
    langsmith_enabled = config is not None and run_id is not None
    trace_id = run_id if run_id is not None else uuid.uuid4()
    initial_state["run_id"] = str(trace_id)
    if config is None:
        runnable_config: dict[str, Any] = {
            "run_id": trace_id,
            "run_name": "phase5_orchestrator_local",
            "tags": ["phase-5", "orchestrator", "langgraph", "local-trajectory"],
            "metadata": {
                "repo_name": Path(repo_root).name,
                "repo_root": repo_root,
                "vulnerability_group_count": len(valid_groups),
            },
            "callbacks": [recorder],
        }
    else:
        runnable_config = config
        runnable_config["callbacks"] = list(config.get("callbacks") or []) + [recorder]

    recorder.record_manual(
        name="phase5.root_input",
        run_type="state",
        inputs=initial_state,
    )
    result: OrchestratorState | None = None
    run_error: BaseException | None = None
    trace_url: str | None = None
    try:
        with use_runtime_settings(settings), use_trajectory_recorder(recorder):
            result = orchestrator_engine.invoke(initial_state, runnable_config)
        if langsmith_enabled and run_id is not None:
            result["langsmith_run_id"] = str(run_id)
            trace_url = resolve_phase5_trace_url(run_id)
            if trace_url:
                result["langsmith_trace_url"] = trace_url
    except BaseException as exc:
        run_error = exc
        raise
    finally:
        if result is None:
            fallback_errors = list(initial_state.get("errors", []) or [])
            if run_error is not None:
                fallback_errors.append(f"orchestrator failed before report phase: {run_error}")
            else:
                fallback_errors.append("orchestrator produced no final state before report phase")
            result = {
                **initial_state,
                "status": "completed_with_errors",
                "errors": fallback_errors,
            }
            try:
                with use_runtime_settings(settings):
                    with use_trajectory_recorder(recorder):
                        report_update = run_report_node(result)
                    report_errors = list(report_update.pop("errors", []) or [])
                    result.update(report_update)
                    if report_errors:
                        result["errors"] = fallback_errors + report_errors
            except Exception as report_node_error:  # noqa: BLE001
                log.exception("run_orchestrator: fallback report node failed")
                result.setdefault("errors", []).append(
                    f"report_node fallback failed: {report_node_error}"
                )
        planned_trajectory_path = build_phase5_trajectory_path(trace_id)
        if result is not None:
            result.setdefault("report_markdown", "")
            result.setdefault("report_path", None)
            result.setdefault("report_status", "pending")
            result.setdefault("report_error", None)
            result["trajectory_path"] = str(planned_trajectory_path)
            try:
                report_markdown, report_path = finalize_report(
                    result,
                    recorder=recorder,
                    trajectory_path=str(planned_trajectory_path),
                    trace_url=trace_url or result.get("langsmith_trace_url"),
                    settings=settings,
                )
                result["report_markdown"] = report_markdown
                if report_path is not None:
                    result["report_path"] = str(report_path)
                    result["report_status"] = "persisted"
                    result["report_error"] = None
                else:
                    result["report_path"] = None
                    result["report_status"] = "rendered" if report_markdown else "failed"
                    result["report_error"] = "report persistence failed"
                    result.setdefault("errors", []).append(
                        "report persistence failed: no report path was returned"
                    )
            except Exception as report_error:  # noqa: BLE001 - report must not mask remediation
                log.exception("run_orchestrator: report finalization failed")
                result["report_path"] = None
                result["report_status"] = "failed"
                result["report_error"] = str(report_error)
                result.setdefault("errors", []).append(
                    f"report finalization failed: {report_error}"
                )

        recorder.record_manual(
            name="phase5.root_output",
            run_type="state",
            inputs={"error": str(run_error)} if run_error else None,
            outputs=result
            if result is not None
            else {"error": str(run_error) if run_error else "no result"},
            error=run_error,
        )
        try:
            trajectory_path = export_phase5_trajectory(
                trace_id=trace_id,
                repo_root=repo_root,
                initial_state=initial_state,
                final_state=result
                if result is not None
                else {"error": str(run_error) if run_error else "no result"},
                recorder=recorder,
                langsmith_enabled=langsmith_enabled,
                langsmith_url=trace_url,
                run_error=run_error,
                output_path=planned_trajectory_path,
            )
            if result is not None:
                result["trajectory_path"] = str(trajectory_path)
        except Exception as export_error:  # noqa: BLE001 - never mask remediation
            log.warning("run_orchestrator: trajectory export failed: %s", export_error)
            if result is not None:
                result.setdefault("errors", []).append(f"trajectory export failed: {export_error}")

    log.info(
        "run_orchestrator: repo_root=%s groups=%d final_status=%s",
        repo_root,
        len(result.get("valid_groups", [])) if result is not None else 0,
        result.get("status") if result is not None else "failed",
    )
    if result is None:
        raise RuntimeError("orchestrator produced no final state")
    return result
