"""
supervisor_node.py - Agentic Supervisor Node for Phase 5 hub-and-spoke orchestration.

Phase 5 architecture: deterministic Supervisor
-----------------------------------------------
The Supervisor owns the state machine and produces every routing and retry
decision in Python. Registry facts are inputs to the deterministic retry
planner; they are never selected by a model or delegated to a worker.

  Guardrails (Python):
    Validate and apply the decision: reject unknown task IDs, clamp cardinality,
    apply copy-on-write task updates, materialize spawn requests, enforce depth
    and queue-size caps.

Public API
----------
MAX_RETRIES : int
    Maximum number of QA-fail-retry cycles before a task is marked unfixable.
run_supervisor_node(state) -> Dict[str, Any]
    LangGraph node callable.
supervisor_router(state) -> str
    Conditional-edge callable: reads ``next_routing_step`` from state.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

from remediation_engine.contracts.decision_codes import (
    DecisionCode,
    validate_transition,
)
from remediation_engine.contracts.schemas import (
    AgentActionStatus,
    AgentActionSummary,
    FailureCategory,
    IssueType,
    MultiPackageAction,
    NoFixMitigationStage,
    PackageMutation,
    QAAttemptResult,
    QAEvaluation,
    QAFailureEvidence,
    RemediationTask,
    RoutingStrategy,
    SCARemediationStage,
    StateConsistencyEvent,
    SupervisorDecision,
    SupervisorRetryPlan,
    TacticalStrategy,
    TaskAttemptSnapshot,
    TaskSpawnRequest,
    TaskStatus,
    UpdateRetryDiagnostics,
    VulnerabilityGroup,
    WorkaroundContext,
    WorkaroundPhase,
    WorkaroundReplayPlan,
    WorkerAttemptResult,
)
from remediation_engine.contracts.solver_models import PortfolioReplanRequest
from remediation_engine.orchestration import _supervisor_execution as _supervisor_execution_helpers
from remediation_engine.orchestration.state import OrchestratorState
from remediation_engine.orchestration.supervisor_planner import (
    _OVERRIDE_DEPENDENCY_TYPES,
    _SCA_STAGE_ORDER,
    QA_DISPATCH_LIMIT,
    UPDATE_DISPATCH_LIMIT,
    _build_deterministic_retry_plan,
    _build_high_level_retry_instruction,
    _commit_retry_plans,
    _needs_planner,
    _override_dependency_type,
    _planner_plan_violations,
    _repair_invalid_planner_plans,
    _run_deterministic_retry_planner,
    instruction_digest,
)
from remediation_engine.orchestration.supervisor_policy import (
    _TERMINAL_STATUSES,
    _WORKABLE_STATUSES,
    MAX_RETRIES,
    _dispatchable_task_ids_for_status,
    _is_exhausted_update_pivot_candidate,
    _next_sca_stage,
    _parent_status_for_strategy_pivot,
    _qa_ready_task_ids,
    _task_sort_key,
    _worker_node_for_strategy,
)
from remediation_engine.orchestration.supervisor_routing import (
    _apply_transition,
    _build_consistency_event,
    _build_workaround_retry_instruction,
    _calculate_eligible_actions,
    _dedupe_consistency_events,
    _deterministic_routing,
    _emit_audit,
    _no_fix_decision_requires_fallback,
    _select_deterministic_action,
    _validate_invariants,
)
from remediation_engine.orchestration.supervisor_spawn import (
    _materialize_spawn_requests,
    _plan_initial_transitive_task,
    _reconcile_terminal_pivot_parents,
    _terminalize_pivot_parents,
)
from remediation_engine.orchestration.task_utils import (
    advance_no_fix_stage,
    build_no_fix_package_removal_instruction,
    build_no_fix_retry_instruction,
    group_parent_context,
    is_transitive_group,
    select_package_fix_plan,
)
from remediation_engine.tools.npm_graph import load_npm_graph_snapshot, make_occurrence_id

logger = logging.getLogger(__name__)

# Supervisor dispatch commits one task attempt at a time. Worker and QA
# boundaries consume only the typed task/attempt envelopes.

_VALID_NEXT_NODES: set[str] = {
    "portfolio",
    "update_subagent",
    "workaround_subagent",
    "qa_critic",
    "triage",
    "final_full_scan",
    "teardown",
}
_WORKER_NODES = frozenset({"update_subagent", "workaround_subagent", "qa_critic"})


__all__ = [
    "MAX_RETRIES",
    "UPDATE_DISPATCH_LIMIT",
    "QA_DISPATCH_LIMIT",
    "_SCA_STAGE_ORDER",
    "_OVERRIDE_DEPENDENCY_TYPES",
    "_task_sort_key",
    "instruction_digest",
    "run_supervisor_node",
    "supervisor_router",
    "_build_consistency_event",
    "_build_deterministic_retry_plan",
    "_build_high_level_retry_instruction",
    "_build_workaround_retry_instruction",
    "_calculate_eligible_actions",
    "_apply_transition",
    "_commit_retry_plans",
    "_commit_task_transition",
    "_create_attempt_snapshot",
    "_dedupe_consistency_events",
    "_deterministic_routing",
    "_emit_audit",
    "_materialize_spawn_requests",
    "_needs_planner",
    "_no_fix_decision_requires_fallback",
    "_plan_initial_transitive_task",
    "_planner_plan_violations",
    "_repair_invalid_planner_plans",
    "_run_deterministic_retry_planner",
    "_select_deterministic_action",
    "_terminalize_pivot_parents",
    "_validate_invariants",
    "_next_sca_stage",
    "_validate_committed_state",
]


def _attempts_for_task(
    snapshots_by_id: dict[str, TaskAttemptSnapshot],
    task_id: str,
) -> list[TaskAttemptSnapshot]:
    return sorted(
        (snapshot for snapshot in snapshots_by_id.values() if snapshot.task_id == task_id),
        key=lambda snapshot: (snapshot.attempt_number, snapshot.created_at),
    )


def _ordered_update_candidates(
    task: RemediationTask,
    *,
    plan: SupervisorRetryPlan | None = None,
    diagnostics: UpdateRetryDiagnostics | None = None,
) -> tuple[list[str], list[str]]:
    """Return only candidates committed by the outer Portfolio Orchestrator."""
    approved_versions = list(
        dict.fromkeys(
            str(value).strip().lstrip("vV")
            for value in task.allowed_target_versions
            if str(value).strip()
        )
    )
    selected_version = task.selected_version
    attempted_versions = {
        str(value).strip().lstrip("vV").lower()
        for value in (list(diagnostics.attempted_versions) if diagnostics else [])
        + (list(plan.attempted_versions) if plan else [])
        if str(value).strip()
    }
    versions: list[str] = []
    for value in [selected_version, *approved_versions]:
        if not value:
            continue
        normalized = str(value).strip().lstrip("vV")
        if (
            normalized
            and normalized.lower() not in attempted_versions
            and normalized not in versions
        ):
            versions.append(normalized)

    approved_types = list(
        dict.fromkeys(
            str(value).strip() for value in task.allowed_dependency_types if str(value).strip()
        )
    )
    selected_type = task.target_dependency_type
    types: list[str] = []
    for value in [selected_type, *approved_types]:
        if value and str(value).strip() not in types:
            types.append(str(value).strip())
    return versions, types


def _current_action_summaries(
    action_summaries: list[AgentActionSummary],
    task_queue: dict[str, RemediationTask],
    limit: int,
) -> list[AgentActionSummary]:
    """Return only summaries belonging to each task's committed attempt."""
    relevant: list[AgentActionSummary] = []
    for summary in action_summaries:
        task = task_queue.get(summary.task_id)
        if task is None:
            continue
        if task.current_attempt_id:
            if summary.attempt_id != task.current_attempt_id:
                continue
        elif summary.attempt_id is not None:
            continue
        relevant.append(summary)
    return relevant[-limit:]


def _extract_workaround_vulnerability_mechanism(group: VulnerabilityGroup) -> str:
    """Return the scanner-described security mechanism for workaround prompts."""
    for issue in getattr(group, "issues", []) or []:
        message = getattr(issue, "message", None)
        if not isinstance(message, str) or not message.strip():
            continue
        mechanism = message
        for marker in ("### Am I affected?", "### How to fix that?"):
            mechanism = mechanism.split(marker, 1)[0]
        mechanism = re.sub(r"\s+", " ", mechanism).strip()
        if mechanism:
            return mechanism[:1200]

    fix_plan = select_package_fix_plan(group).plan
    instruction = getattr(fix_plan, "instruction", None)
    if isinstance(instruction, str) and instruction.strip():
        return re.sub(r"\s+", " ", instruction).strip()[:1200]
    return ""


def _create_attempt_snapshot(
    task: RemediationTask,
    *,
    dispatch_node: str,
    snapshots_by_id: dict[str, TaskAttemptSnapshot],
    state_revision: int,
    plan_id: str | None = None,
    portfolio_plan_id: str | None = None,
    workaround_context: WorkaroundContext | None = None,
    allowed_target_versions: Iterable[str] = (),
    allowed_dependency_types: Iterable[str] = (),
    cluster_id: str | None = None,
    dispatch_batch_id: str | None = None,
    action_digest: str | None = None,
    manifest_path: str | None = None,
) -> tuple[RemediationTask, TaskAttemptSnapshot]:
    """Commit the exact worker input and return the revised task projection."""
    task_revision = task.task_revision + 1
    # Attempt identity is derived from committed state rather than wall-clock
    # randomness.  This makes replaying an identical OrchestratorState produce
    # the same dispatch projection while still changing identity whenever the
    # task revision or supervisor state revision changes.
    attempt_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"remediation-attempt:{task.task_id}:{task_revision}:{state_revision}:{dispatch_node}",
        )
    )
    snapshot = TaskAttemptSnapshot(
        attempt_id=attempt_id,
        task_id=task.task_id,
        is_synthetic=task.is_synthetic,
        state_revision=state_revision,
        task_revision=task_revision,
        attempt_number=len(_attempts_for_task(snapshots_by_id, task.task_id)) + 1,
        cluster_id=cluster_id,
        dispatch_batch_id=dispatch_batch_id,
        action_digest=action_digest,
        manifest_path=manifest_path,
        selected_plan_issue_ids=list(task.selected_plan_issue_ids),
        qa_policy=task.qa_policy,
        strategy_stage=task.strategy_stage,
        no_fix_stage=task.no_fix_stage,
        selected_version=task.selected_version,
        allowed_target_versions=list(
            dict.fromkeys(
                str(value).strip().lstrip("vV")
                for value in allowed_target_versions
                if str(value).strip()
            )
        ),
        target_package_name=task.target_package_name,
        target_dependency_type=task.target_dependency_type,
        allowed_dependency_types=list(
            dict.fromkeys(
                str(value).strip() for value in allowed_dependency_types if str(value).strip()
            )
        ),
        parent_minimum_version=task.parent_minimum_version,
        instruction=task.instruction,
        instruction_digest=instruction_digest(task.instruction),
        dispatch_node=dispatch_node,  # type: ignore[arg-type]
        plan_id=plan_id,
        portfolio_plan_id=portfolio_plan_id or task.portfolio_plan_id,
        created_at=datetime.fromtimestamp(state_revision, tz=UTC),
        workaround_context=workaround_context,
    )
    snapshots_by_id[attempt_id] = snapshot
    updated_task = task.model_copy(
        update={
            "task_revision": task_revision,
            "current_attempt_id": attempt_id,
        }
    )
    return updated_task, snapshot


def _cluster_dispatch_batch_id(
    cluster_id: str,
    task_ids: Iterable[str],
    state_revision: int,
) -> str:
    """Return a deterministic identifier shared by one cluster dispatch."""
    payload = f"{cluster_id}:{state_revision}:{','.join(sorted(task_ids))}"
    return f"batch-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:24]}"


def _is_hard_overflow_cluster(cluster: Any) -> bool:
    """Return whether a retained cluster exceeds the atomic action limit."""
    return str(getattr(cluster, "reason", "")).lower().startswith("hard atomic component exceeds ")


def _recover_active_target_task_ids(
    active_target_task_ids: Iterable[str],
    task_queue: dict[str, RemediationTask],
    worker_results_by_attempt: dict[str, WorkerAttemptResult],
    qa_results_by_attempt: dict[str, QAAttemptResult],
    processed_worker_attempt_ids: set[str],
    processed_qa_attempt_ids: set[str],
) -> list[str]:
    """Restore active task handles for unprocessed attempt-tagged results.

    The active-target list is a dispatch projection, not the source of truth
    for attempt results.  A portfolio replan can legitimately rebuild that
    projection while a worker or QA envelope is still buffered in state.  In
    that case the current task attempt is the only safe join key: recovering
    by task ID alone could attach a stale result to a newer attempt.

    Args:
        active_target_task_ids: Task IDs carried by the prior graph node.
        task_queue: Detached authoritative task projection for this pass.
        worker_results_by_attempt: Attempt-correlated worker results.
        qa_results_by_attempt: Attempt-correlated QA results.
        processed_worker_attempt_ids: Worker envelopes already consumed.
        processed_qa_attempt_ids: QA envelopes already consumed.

    Returns:
        The original active IDs followed by task IDs whose current attempt has
        an unprocessed worker or QA envelope.  Ordering is stable and IDs are
        unique.
    """
    recovered = list(dict.fromkeys(active_target_task_ids))
    known_ids = set(recovered)
    for task_id, task in task_queue.items():
        attempt_id = task.current_attempt_id
        if not attempt_id or task_id in known_ids:
            continue
        worker_pending = (
            attempt_id in worker_results_by_attempt
            and attempt_id not in processed_worker_attempt_ids
        )
        qa_pending = (
            attempt_id in qa_results_by_attempt and attempt_id not in processed_qa_attempt_ids
        )
        if worker_pending or qa_pending:
            recovered.append(task_id)
            known_ids.add(task_id)
    return recovered


def _portfolio_plan_violations(
    plan: Any,
    task_queue: dict[str, RemediationTask],
    valid_groups: list[VulnerabilityGroup],
    *,
    repo_root: str | None = None,
) -> list[str]:
    """Validate the immutable outer-plan contract before any dispatch."""
    if plan is None:
        return ["missing committed portfolio plan"]
    violations: list[str] = []
    for cluster in getattr(plan, "clusters", ()) or ():
        if not getattr(cluster, "dispatchable", True) and _is_hard_overflow_cluster(cluster):
            violations.append(
                f"portfolio cluster {cluster.cluster_id!r} is marked non-dispatchable"
            )
    plan_id = getattr(plan, "portfolio_plan_id", None) or getattr(plan, "plan_id", None)
    selected = getattr(getattr(plan, "solver_plan", None), "selected_plan", None)
    decisions = {
        decision.task_id: decision for decision in (getattr(selected, "task_decisions", None) or [])
    }
    sca_group_ids = {group.group_id for group in valid_groups if group.issue_type == IssueType.SCA}
    groups_by_id = {group.group_id: group for group in valid_groups}
    tasks_by_group: dict[str, list[RemediationTask]] = {}
    for task in task_queue.values():
        if task.parent_group_id in sca_group_ids:
            tasks_by_group.setdefault(task.parent_group_id, []).append(task)
    terminal_statuses = {
        TaskStatus.QA_PASSED,
        TaskStatus.UNFIXABLE,
        TaskStatus.INCONCLUSIVE,
        TaskStatus.PIVOTED,
    }
    active_sca_tasks = []
    for tasks in tasks_by_group.values():
        nonterminal = [task for task in tasks if task.status not in terminal_statuses]
        active_sca_tasks.append(
            max(nonterminal or tasks, key=lambda task: (task.task_revision, task.task_id))
        )
    sca_task_ids = {task.task_id for task in active_sca_tasks}
    planned_ids = set(getattr(plan, "task_ids", ()) or ())
    if sca_task_ids != planned_ids:
        violations.append(
            f"portfolio membership mismatch: active SCA tasks={sorted(sca_task_ids)!r}, "
            f"planned={sorted(planned_ids)!r}"
        )
    planned_revisions = getattr(plan, "planned_task_revisions", None) or {}
    strategies = getattr(plan, "task_strategies", None) or {}
    for task_id in sorted(planned_ids):
        task = task_queue.get(task_id)
        if task is None:
            violations.append(f"portfolio plan references missing task {task_id!r}")
            continue
        if task.portfolio_plan_id != plan_id:
            violations.append(
                f"task {task_id} portfolio plan {task.portfolio_plan_id!r} "
                f"differs from committed {plan_id!r}"
            )
        baseline = planned_revisions.get(task_id)
        if baseline is None:
            violations.append(f"task {task_id} has no committed planned task revision")
        elif task.task_revision < baseline:
            violations.append(
                f"task {task_id} is older than committed planned revision "
                f"{baseline!r} (current={task.task_revision})"
            )
        expected_strategy = strategies.get(task_id)
        if expected_strategy is not None and task.strategy != expected_strategy:
            violations.append(
                f"task {task_id} strategy {task.strategy.value!r} differs from "
                f"committed {expected_strategy.value!r}"
            )
        decision = decisions.get(task_id)
        if decision is None:
            violations.append(f"task {task_id} has no committed solver decision")
            continue
        decision_strategy = str(decision.selected_strategy).replace("-", "_").lower()
        expected = (
            RoutingStrategy.CODE_WORKAROUND
            if decision_strategy in {"code_workaround", "workaround", "no_fix"}
            else RoutingStrategy.VERSION_BUMP
        )
        if task.strategy != expected:
            violations.append(f"task {task_id} strategy is outside solver decision")
        group = groups_by_id.get(task.parent_group_id)
        expected_package = (
            (task.target_package_name or group.vulnerable_component or "").strip()
            if group is not None
            else ""
        )
        expected_manifest = _group_manifest_path(group) if group is not None else None
        expected_identity = {
            "target_occurrence_id": (
                make_occurrence_id(expected_manifest, expected_package)
                if expected_manifest and expected_package
                else None
            ),
            "target_group_id": task.parent_group_id,
            "target_package_name": expected_package or None,
            "manifest_path": expected_manifest,
            "lockfile_package_key": (
                f"node_modules/{expected_package}" if expected_package else None
            ),
        }
        for field_name, expected_value in expected_identity.items():
            if getattr(decision, field_name, None) != expected_value:
                violations.append(f"task {task_id} solver identity field {field_name} is invalid")
        try:
            if task.strategy_stage != SCARemediationStage(decision.strategy_stage):
                violations.append(f"task {task_id} strategy stage differs from solver decision")
        except ValueError:
            violations.append(f"task {task_id} has unknown committed strategy stage")
        approved_versions = [
            str(value).strip().lstrip("vV").lower()
            for value in ([decision.selected_version] + list(decision.allowed_alternative_versions))
            if value
        ]
        task_versions = [
            str(value).strip().lstrip("vV").lower()
            for value in task.allowed_target_versions
            if value
        ]
        if task_versions != approved_versions:
            violations.append(f"task {task_id} allowed_target_versions differ from solver")
        if task.selected_version and str(task.selected_version).strip().lstrip(
            "vV"
        ).lower() not in set(approved_versions):
            violations.append(f"task {task_id} selected_version is not solver-approved")
        approved_types = [str(value).strip() for value in decision.allowed_dependency_types]
        task_types = [str(value).strip() for value in task.allowed_dependency_types]
        if task_types != approved_types:
            violations.append(f"task {task_id} allowed_dependency_types differ from solver")
        if task.target_dependency_type and task.target_dependency_type not in set(task_types):
            violations.append(f"task {task_id} dependency type is not solver-approved")
    return violations


def _portfolio_plan_is_stale(
    plan: Any,
    task_queue: dict[str, RemediationTask],
    valid_groups: list[VulnerabilityGroup],
    repo_root: str | None = None,
) -> bool:
    """Compatibility predicate backed by committed plan fields and graph fingerprint."""
    violations = _portfolio_plan_violations(plan, task_queue, valid_groups)
    if repo_root and plan is not None:
        expected_fingerprint = getattr(plan, "repository_fingerprint", None)
        try:
            actual_fingerprint = load_npm_graph_snapshot(repo_root).repository_fingerprint
        except (OSError, ValueError):
            actual_fingerprint = None
        if not expected_fingerprint or actual_fingerprint != expected_fingerprint:
            violations.append("repository fingerprint differs from committed portfolio plan")
    return bool(violations)


def _build_multi_package_action(
    cluster_id: str,
    target_task_ids: Iterable[str],
    task_queue: dict[str, RemediationTask],
    group_by_id: dict[str, VulnerabilityGroup],
) -> MultiPackageAction | None:
    """Build an exact atomic action from Supervisor-committed task inputs."""
    mutations: list[PackageMutation] = []
    for task_id in target_task_ids:
        task = task_queue.get(task_id)
        group = group_by_id.get(task.parent_group_id) if task else None
        if task is None or group is None or task.strategy != RoutingStrategy.VERSION_BUMP:
            return None
        package_name = (task.target_package_name or group.vulnerable_component or "").strip()
        target_version = (task.selected_version or "").strip()
        manifest_path = _group_manifest_path(group) or ""
        dependency_type = task.target_dependency_type or "dependencies"
        if (
            not package_name
            or not target_version
            or not manifest_path
            or manifest_path.split("/")[-1] != "package.json"
        ):
            return None
        mutations.append(
            PackageMutation(
                task_id=task_id,
                package_name=package_name,
                manifest_path=manifest_path,
                target_version=target_version,
                dependency_type=dependency_type,
            )
        )
    if not mutations:
        return None
    selected_strategy = (
        TacticalStrategy.PACKAGE_OVERRIDE
        if all(
            mutation.dependency_type in {"overrides", "resolutions", "pnpm_overrides"}
            for mutation in mutations
        )
        else TacticalStrategy.VERSION_BUMP
    )
    try:
        return MultiPackageAction(
            cluster_id=cluster_id,
            selected_strategy=selected_strategy,
            package_mutations=mutations,
            rationale="Supervisor-committed atomic package-group update.",
        )
    except ValueError:
        return None


def _group_manifest_path(group: VulnerabilityGroup) -> str | None:
    """Return the one canonical package.json path committed for a group."""
    manifest_paths = sorted(
        {
            (localized.manifest_file or "").replace("\\", "/").lstrip("/")
            for localized in group.localized_issues
            if localized.manifest_file
        }
        | {path.replace("\\", "/").lstrip("/") for path in group.file_paths if path}
        | ({group.file_path.replace("\\", "/").lstrip("/")} if group.file_path else set())
    )
    if len(manifest_paths) != 1 or manifest_paths[0].split("/")[-1] != "package.json":
        return None
    return manifest_paths[0]


def _peer_conflict_escalation(
    task_id: str,
    evaluation: QAEvaluation,
    task_queue: dict[str, RemediationTask],
    group_by_id: dict[str, VulnerabilityGroup],
) -> tuple[dict[str, Any] | None, list[str]]:
    """Resolve deterministic peer evidence to active package-task pairs.

    Only active, non-terminal npm version-bump tasks may be added to a
    portfolio escalation. Missing or unsupported peer tasks are reported and
    left on the normal compatible-version retry path.
    """
    gates = evaluation.deterministic_gates
    if evaluation.failure_category != FailureCategory.PEER_CONFLICT or gates is None:
        return None, []
    leaf_ids = set(task_queue)
    package_to_task: dict[str, str] = {}
    for candidate_id in sorted(leaf_ids):
        candidate = task_queue.get(candidate_id)
        if candidate is None or candidate.status in _TERMINAL_STATUSES:
            continue
        if candidate.strategy != RoutingStrategy.VERSION_BUMP:
            continue
        group = group_by_id.get(candidate.parent_group_id)
        if group is None:
            continue
        manager_values = {
            (localized.package_manager or "").strip().lower()
            for localized in group.localized_issues
            if localized.package_manager
        }
        if manager_values and manager_values != {"npm"}:
            continue
        manifest_paths = {
            (localized.manifest_file or "").replace("\\", "/").lstrip("/")
            for localized in group.localized_issues
            if localized.manifest_file
        }
        manifest_paths.update(
            path.replace("\\", "/").lstrip("/") for path in group.file_paths if path
        )
        if len(manifest_paths) != 1 or next(iter(manifest_paths)).split("/")[-1] != "package.json":
            continue
        package_name = (candidate.target_package_name or group.vulnerable_component or "").strip()
        if package_name:
            package_to_task.setdefault(package_name, candidate_id)

    pairs: set[tuple[str, str]] = set()
    unresolved: list[str] = []
    for evidence in gates.peer_conflicts:
        requester = evidence.requester_package.strip()
        peer = evidence.peer_package.strip()
        requester_id = package_to_task.get(requester)
        if requester_id is None and task_id in leaf_ids:
            current = task_queue.get(task_id)
            current_group = group_by_id.get(current.parent_group_id) if current else None
            current_package = (
                (current.target_package_name if current else None)
                or (current_group.vulnerable_component if current_group else None)
                or ""
            ).strip()
            if current_package == requester:
                requester_id = task_id
        peer_id = package_to_task.get(peer)
        if requester_id and peer_id and requester_id != peer_id:
            pairs.add(tuple(sorted((requester_id, peer_id))))
        else:
            unresolved.append(
                f"peer escalation for {requester or '<unknown>'} -> {peer or '<unknown>'} "
                "has no existing eligible active package task"
            )
    if not pairs:
        return None, unresolved
    return {
        "reason": "PEER_CONFLICT_ESCALATION",
        "peer_conflict_pairs": sorted(pairs),
        "evidence": [evidence.model_dump(mode="json") for evidence in gates.peer_conflicts],
    }, unresolved


def _qa_failure_evidence_for_workaround_retry(
    task_id: str,
    qa_evaluations: dict[str, QAEvaluation],
    qa_results_by_attempt: dict[str, QAAttemptResult],
    *,
    related_task_ids: Iterable[str] = (),
) -> QAFailureEvidence | None:
    """Resolve QA evidence for a workaround task and its task ancestry.

    QA closes the failed worker attempt before the Supervisor creates the
    retry snapshot. Evidence is therefore correlated through the immutable
    attempt envelope and explicit task IDs only.
    """
    task_ids = list(dict.fromkeys([task_id, *related_task_ids]))

    for evaluation_key in task_ids:
        evaluation = qa_evaluations.get(evaluation_key)
        evaluation_evidence = evaluation.failure_evidence if evaluation else None

        if evaluation_evidence and evaluation_evidence.attempt_id:
            envelope = qa_results_by_attempt.get(evaluation_evidence.attempt_id)
            if (
                envelope is not None
                and envelope.task_id in task_ids
                and envelope.evaluation.failure_evidence is not None
            ):
                return envelope.evaluation.failure_evidence

        if evaluation_evidence is not None:
            return evaluation_evidence

    candidates = [
        result.evaluation.failure_evidence
        for result in qa_results_by_attempt.values()
        if (
            result.task_id in task_ids
            and not result.evaluation.passed
            and result.evaluation.failure_evidence is not None
        )
    ]
    return candidates[-1] if candidates else None


def _qa_evidence_indicates_test_regression(
    evidence: QAFailureEvidence | None,
) -> bool:
    """Return whether inherited QA evidence represents a test/build regression."""
    if evidence is None:
        return False
    if evidence.failed_tests:
        return True

    text = " ".join(
        [
            *(evidence.exact_diagnostics or []),
            evidence.raw_excerpt or "",
            *(evidence.source_locations or []),
        ]
    ).lower()
    return any(
        marker in text
        for marker in (
            "npm test",
            "test failed",
            "tests failed",
            "typecheck",
            "compile failed",
            "build failed",
            "is not a function",
            "typeerror",
            "/test/",
            "\\test\\",
        )
    )


def _workaround_task_ancestry(
    task: RemediationTask,
    task_queue: dict[str, RemediationTask],
) -> list[str]:
    """Return parent task IDs whose QA evidence may seed a child attempt.

    The task queue is authoritative for parent links. Missing or cyclic links
    are ignored defensively so malformed state cannot block workaround
    dispatch.
    """
    parent_task_ids: list[str] = []
    seen: set[str] = set()
    parent_task_id = task.parent_task_id

    while parent_task_id and parent_task_id not in seen:
        seen.add(parent_task_id)
        parent_task = task_queue.get(parent_task_id)
        if parent_task is None:
            break
        parent_task_ids.append(parent_task.task_id)
        parent_task_id = parent_task.parent_task_id

    return parent_task_ids


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
    consistency_events: list[StateConsistencyEvent] | None = None,
) -> RemediationTask | None:
    with _supervisor_execution_helpers._bind_dependencies(
        _build_consistency_event=_build_consistency_event,
        validate_transition=validate_transition,
        _TERMINAL_STATUSES=_TERMINAL_STATUSES,
        logger=logger,
    ):
        return _supervisor_execution_helpers._impl__commit_task_transition(
            task_queue,
            task_id,
            updates=updates,
            close_attempt=close_attempt,
            clear_selected_version=clear_selected_version,
            allow_breaking_change_pivot=allow_breaking_change_pivot,
            consistency_events=consistency_events,
        )


def _validate_committed_state(
    task_queue: dict[str, RemediationTask],
    snapshots_by_id: dict[str, TaskAttemptSnapshot],
    retry_plans_by_task: dict[str, SupervisorRetryPlan],
    retry_diagnostics_by_task: dict[str, UpdateRetryDiagnostics],
    active_target_task_ids: list[str],
    next_node: str,
) -> tuple[list[StateConsistencyEvent], list[str]]:
    with _supervisor_execution_helpers._bind_dependencies(
        _build_consistency_event=_build_consistency_event,
        instruction_digest=instruction_digest,
        _TERMINAL_STATUSES=_TERMINAL_STATUSES,
    ):
        return _supervisor_execution_helpers._impl__validate_committed_state(
            task_queue,
            snapshots_by_id,
            retry_plans_by_task,
            retry_diagnostics_by_task,
            active_target_task_ids,
            next_node,
        )


def reconcile_phase5_state_before_teardown(
    state: OrchestratorState,
) -> dict[str, Any]:
    with _supervisor_execution_helpers._bind_dependencies(
        _build_consistency_event=_build_consistency_event,
        _validate_committed_state=_validate_committed_state,
        _dedupe_consistency_events=_dedupe_consistency_events,
        _TERMINAL_STATUSES=_TERMINAL_STATUSES,
        instruction_digest=instruction_digest,
    ):
        return _supervisor_execution_helpers._impl_reconcile_phase5_state_before_teardown(state)


def _update_worker_task_ids(
    task_queue: dict[str, RemediationTask],
    retry_diagnostics_by_task: dict[str, UpdateRetryDiagnostics],
    preferred_ids: list[str] | None = None,
    limit: int | None = UPDATE_DISPATCH_LIMIT,
    group_by_id: dict[str, VulnerabilityGroup] | None = None,
) -> list[str]:
    with _supervisor_execution_helpers._bind_dependencies(
        _dispatchable_task_ids_for_status=_dispatchable_task_ids_for_status,
        _is_exhausted_update_pivot_candidate=_is_exhausted_update_pivot_candidate,
        _WORKABLE_STATUSES=_WORKABLE_STATUSES,
    ):
        return _supervisor_execution_helpers._impl__update_worker_task_ids(
            task_queue,
            retry_diagnostics_by_task,
            preferred_ids=preferred_ids,
            limit=limit,
            group_by_id=group_by_id,
        )


def _normalize_target_task_ids_for_node(
    next_node: str,
    target_task_ids: list[str],
    task_queue: dict[str, RemediationTask],
    retry_diagnostics_by_task: dict[str, UpdateRetryDiagnostics] | None = None,
    group_by_id: dict[str, VulnerabilityGroup] | None = None,
    allow_cluster: bool = False,
) -> list[str]:
    with _supervisor_execution_helpers._bind_dependencies(
        _qa_ready_task_ids=_qa_ready_task_ids,
        _update_worker_task_ids=_update_worker_task_ids,
        _dispatchable_task_ids_for_status=_dispatchable_task_ids_for_status,
        _WORKABLE_STATUSES=_WORKABLE_STATUSES,
        QA_DISPATCH_LIMIT=QA_DISPATCH_LIMIT,
        UPDATE_DISPATCH_LIMIT=UPDATE_DISPATCH_LIMIT,
    ):
        return _supervisor_execution_helpers._impl__normalize_target_task_ids_for_node(
            next_node,
            target_task_ids,
            task_queue,
            retry_diagnostics_by_task,
            group_by_id,
            allow_cluster,
        )


def _resolve_task_id_from_identifier(
    identifier: str,
    task_queue: dict[str, RemediationTask],
    active_target_task_ids: list[str],
) -> str | None:
    with _supervisor_execution_helpers._bind_dependencies(
        _TERMINAL_STATUSES=_TERMINAL_STATUSES,
    ):
        return _supervisor_execution_helpers._impl__resolve_task_id_from_identifier(
            identifier,
            task_queue,
            active_target_task_ids,
        )


def _normalize_qa_evaluations_for_tasks(
    qa_evaluations: dict[str, QAEvaluation],
    task_queue: dict[str, RemediationTask],
    active_target_task_ids: list[str],
) -> dict[str, QAEvaluation]:
    with _supervisor_execution_helpers._bind_dependencies(
        _resolve_task_id_from_identifier=_resolve_task_id_from_identifier,
        _TERMINAL_STATUSES=_TERMINAL_STATUSES,
    ):
        return _supervisor_execution_helpers._impl__normalize_qa_evaluations_for_tasks(
            qa_evaluations,
            task_queue,
            active_target_task_ids,
        )


def _constraint_entry_for_task(
    task: RemediationTask,
    group: VulnerabilityGroup,
) -> str:
    return _supervisor_execution_helpers._impl__constraint_entry_for_task(task, group)


def _missing_retry_revised_instructions(
    next_node: str,
    target_task_ids: list[str],
    revised_instructions: dict[str, str],
    task_queue: dict[str, RemediationTask],
) -> list[str]:
    return _supervisor_execution_helpers._impl__missing_retry_revised_instructions(
        next_node,
        target_task_ids,
        revised_instructions,
        task_queue,
    )


def _latest_action_summary_by_task(
    action_summaries: list[AgentActionSummary],
    task_queue: dict[str, RemediationTask],
    active_target_task_ids: list[str],
) -> dict[str, AgentActionSummary]:
    with _supervisor_execution_helpers._bind_dependencies(
        _resolve_task_id_from_identifier=_resolve_task_id_from_identifier,
        _TERMINAL_STATUSES=_TERMINAL_STATUSES,
    ):
        return _supervisor_execution_helpers._impl__latest_action_summary_by_task(
            action_summaries,
            task_queue,
            active_target_task_ids,
        )


def _no_fix_failure_transition(
    task: RemediationTask,
    group: VulnerabilityGroup | None,
    *,
    evaluation: QAEvaluation | None = None,
    failure_feedback: str | None = None,
) -> tuple[dict[str, Any], bool]:
    with _supervisor_execution_helpers._bind_dependencies(
        advance_no_fix_stage=advance_no_fix_stage,
        build_no_fix_retry_instruction=build_no_fix_retry_instruction,
    ):
        return _supervisor_execution_helpers._impl__no_fix_failure_transition(
            task,
            group,
            evaluation=evaluation,
            failure_feedback=failure_feedback,
        )


def _reset_no_fix_replay_plan(
    replay_plan: WorkaroundReplayPlan | None,
) -> WorkaroundReplayPlan | None:
    return _supervisor_execution_helpers._impl__reset_no_fix_replay_plan(replay_plan)


# ---------------------------------------------------------------------------
# Deterministic fallback router
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Planner phase helpers
# ---------------------------------------------------------------------------


# Spawn request materializer
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Supervisor node
# ---------------------------------------------------------------------------


def run_supervisor_node(state: OrchestratorState) -> dict[str, Any]:
    """
    LangGraph node â€” Supervisor commander for Phase 5 orchestration.

    Execution stages
    ----------------
    1. Normalize task_queue: create initial RemediationTask entries for any
       valid_groups not yet represented (copy-on-write via model_copy).
    2. Ingest subagent action summaries for current active_target_task_ids only.
    3. Ingest QA results for active task IDs only (when status == "qa_completed").
    4. Mark UNFIXABLE any task whose retry_count has reached MAX_RETRIES.
    5. Short-circuit: if an active task is optimistically_fixed â†’ qa_critic.
    6. If QA produced a parseable scan and set ``triage_required``, route to
       the post-QA triage node before any worker or teardown decision.
    7. Plan retry inputs from deterministic registry facts and committed state.
    8. Route through the deterministic state machine and validate its
       dispatch projection.
    9. Apply guarded: revised_instructions, strategy updates, status overrides,
       unfixable marks, new constraints, and materialized spawn requests.
    10. Return state patch.
    """
    if state.get("post_qa_retriage_limit_reached"):
        decision = SupervisorDecision(
            decision_code=DecisionCode.NO_ACTIONABLE_TASKS,
            next_node="teardown",
            target_task_ids=[],
            instructions="Development post-QA re-triage limit reached; proceed to teardown.",
            decision_reason="The configured development post-QA re-triage limit was reached.",
        )
        return {
            "status": "supervisor_routed",
            "next_routing_step": "teardown",
            "active_target_task_ids": [],
            "decision_code": decision.decision_code,
            "supervisor_audit": _emit_audit(decision, [], int(state.get("state_revision", 0)) + 1),
            "supervisor_instructions": decision.instructions,
        }
    valid_groups: list[VulnerabilityGroup] = list(state.get("valid_groups", []))
    if not valid_groups:
        if state.get("triage_required") and state.get("status") in {
            "qa_completed",
            "qa_failed",
            "final_scan_completed",
        }:
            decision = SupervisorDecision(
                decision_code=DecisionCode.TRIAGE_REQUIRED,
                next_node="triage",
                target_task_ids=[],
                instructions="Route the parseable post-remediation scan to triage.",
                decision_reason="Post-QA triage is required.",
            )
            return {
                "status": "supervisor_routed",
                "next_routing_step": "triage",
                "active_target_task_ids": [],
                "decision_code": decision.decision_code,
                "supervisor_audit": _emit_audit(
                    decision, [], int(state.get("state_revision", 0)) + 1
                ),
                "supervisor_instructions": "Route the parseable post-remediation scan to triage.",
            }
        logger.info("supervisor: no valid groups â€” routing to teardown.")
        decision = SupervisorDecision(
            decision_code=DecisionCode.NO_VALID_GROUPS,
            next_node="teardown",
            target_task_ids=[],
            instructions="No groups to process.",
            decision_reason="No valid vulnerability groups are available.",
        )
        return {
            "status": "supervisor_routed",
            "next_routing_step": "teardown",
            "active_target_task_ids": [],
            "decision_code": decision.decision_code,
            "supervisor_audit": _emit_audit(decision, [], int(state.get("state_revision", 0)) + 1),
            "supervisor_instructions": "No groups to process.",
        }

    group_by_id: dict[str, VulnerabilityGroup] = {g.group_id: g for g in valid_groups}
    existing_constraints: list[str] = list(state.get("constraints_ledger", []))
    retry_diagnostics_by_task: dict[str, UpdateRetryDiagnostics] = dict(
        state.get("retry_diagnostics_by_task", {})
    )
    retry_plans_by_task: dict[str, SupervisorRetryPlan] = dict(state.get("retry_plans_by_task", {}))
    workaround_replay_plans_by_task: dict[str, WorkaroundReplayPlan] = dict(
        state.get("workaround_replay_plans_by_task", {})
    )
    attempt_snapshots_by_id: dict[str, TaskAttemptSnapshot] = dict(
        state.get("attempt_snapshots_by_id", {})
    )
    worker_results_by_attempt: dict[str, WorkerAttemptResult] = dict(
        state.get("worker_results_by_attempt", {})
    )
    qa_results_by_attempt: dict[str, QAAttemptResult] = dict(state.get("qa_results_by_attempt", {}))
    processed_worker_attempt_ids: set[str] = set(state.get("processed_worker_attempt_ids", []))
    processed_qa_attempt_ids: set[str] = set(state.get("processed_qa_attempt_ids", []))
    prior_consistency_events: list[StateConsistencyEvent] = list(
        state.get("consistency_events", [])
    )
    consistency_events: list[StateConsistencyEvent] = []
    state_revision = int(state.get("state_revision", 0)) + 1
    # ``errors`` is an additive LangGraph reducer. A node must return only
    # errors discovered during this invocation; replaying the prior list here
    # is what caused identical planner errors to multiply across supervisor
    # loops.
    errors: list[str] = []
    prior_error_messages = set(state.get("errors", []) or [])

    # The outer Portfolio Orchestrator is the only owner of task creation,
    # synthetic dependency materialization, and version selection.
    raw_task_queue: dict[str, RemediationTask] = dict(state.get("task_queue", {}))
    task_queue: dict[str, RemediationTask] = {
        task_id: task.model_copy() for task_id, task in raw_task_queue.items()
    }
    active_target_task_ids = _recover_active_target_task_ids(
        state.get("active_target_task_ids") or [],
        task_queue,
        worker_results_by_attempt,
        qa_results_by_attempt,
        processed_worker_attempt_ids,
        processed_qa_attempt_ids,
    )
    committed_plan = state.get("portfolio_plan")
    has_sca_groups = any(group.issue_type == IssueType.SCA for group in valid_groups)
    plan_violations = (
        _portfolio_plan_violations(
            committed_plan,
            task_queue,
            valid_groups,
            repo_root=state.get("repo_root"),
        )
        if has_sca_groups or committed_plan is not None
        else []
    )
    # Attempt-tagged worker/QA envelopes must be consumed before a stale outer
    # plan can trigger another portfolio iteration.  The previous ordering
    # returned here with an empty active-target projection, so a result from a
    # task whose attempt survived a portfolio replan was never ingested.
    has_open_attempt = any(
        task.current_attempt_id is not None and task.status not in _TERMINAL_STATUSES
        for task in task_queue.values()
    )
    if plan_violations and not has_open_attempt:
        source_plan_id = (
            getattr(committed_plan, "portfolio_plan_id", None)
            if committed_plan is not None
            else None
        )
        request = PortfolioReplanRequest(
            reason="; ".join(plan_violations)[:2000],
            triggering_attempt_id=next(
                (
                    task.current_attempt_id
                    for task in task_queue.values()
                    if task.current_attempt_id
                ),
                None,
            ),
            source_portfolio_plan_id=source_plan_id,
        )
        decision = SupervisorDecision(
            decision_code=DecisionCode.PORTFOLIO_PLAN_REQUIRED,
            next_node="portfolio",
            target_task_ids=[],
            instructions="Rebuild the outer portfolio plan before Supervisor dispatch.",
            decision_reason=request.reason,
        )
        return {
            "status": "supervisor_routed",
            "next_routing_step": "portfolio",
            "active_target_task_ids": [],
            "decision_code": decision.decision_code,
            "supervisor_audit": _emit_audit(decision, [], state_revision),
            "supervisor_instructions": decision.instructions,
            "task_queue": task_queue,
            "valid_groups": valid_groups,
            "portfolio_replan_request": request,
            "portfolio_dirty": True,
            "state_revision": state_revision,
        }

    # ------------------------------------------------------------------
    # 2. Ingest attempt-tagged worker results (active targets only)
    # ------------------------------------------------------------------
    action_summaries: list[AgentActionSummary] = state.get("action_summaries") or []
    new_worker_attempt_ids: list[str] = []
    atomic_cluster_task_ids = [
        task_id
        for task_id in active_target_task_ids
        if (
            (task := task_queue.get(task_id)) is not None
            and task.current_attempt_id
            and (snapshot := attempt_snapshots_by_id.get(task.current_attempt_id)) is not None
            and snapshot.cluster_id
            and snapshot.dispatch_node == "update_subagent"
        )
    ]
    atomic_cluster_task_ids = atomic_cluster_task_ids if len(atomic_cluster_task_ids) > 1 else []

    for task_id in active_target_task_ids:
        task = task_queue.get(task_id)
        if task is None:
            continue
        current_attempt_id = task.current_attempt_id
        snapshot = attempt_snapshots_by_id.get(current_attempt_id) if current_attempt_id else None
        result = worker_results_by_attempt.get(current_attempt_id) if current_attempt_id else None
        snapshot_batch_id = (
            snapshot.dispatch_batch_id
            if snapshot is not None and isinstance(snapshot.dispatch_batch_id, str)
            else None
        )
        snapshot_cluster_id = (
            snapshot.cluster_id
            if snapshot is not None and isinstance(snapshot.cluster_id, str)
            else None
        )
        snapshot_action_digest = (
            snapshot.action_digest
            if snapshot is not None and isinstance(snapshot.action_digest, str)
            else None
        )
        for stale_result in worker_results_by_attempt.values():
            if (
                stale_result.task_id == task_id
                and stale_result.attempt_id != current_attempt_id
                and stale_result.attempt_id not in processed_worker_attempt_ids
            ):
                consistency_events.append(
                    _build_consistency_event(
                        error_code="STALE_WORKER_RESULT",
                        task_id=task_id,
                        expected_attempt_id=current_attempt_id,
                        received_attempt_id=stale_result.attempt_id,
                        action="ignored",
                        details="Worker result belongs to an older task attempt.",
                    )
                )
                processed_worker_attempt_ids.add(stale_result.attempt_id)
                new_worker_attempt_ids.append(stale_result.attempt_id)
        if result is not None and task.status in _TERMINAL_STATUSES:
            if current_attempt_id not in processed_worker_attempt_ids:
                consistency_events.append(
                    _build_consistency_event(
                        error_code="TERMINAL_TASK_RESULT_IGNORED",
                        task_id=task_id,
                        expected_attempt_id=current_attempt_id,
                        received_attempt_id=result.attempt_id,
                        action="ignored",
                        details="A late worker result cannot reopen a terminal task.",
                    )
                )
                processed_worker_attempt_ids.add(current_attempt_id)
                new_worker_attempt_ids.append(current_attempt_id)
            continue
        if result is not None and current_attempt_id not in processed_worker_attempt_ids:
            if (
                result.task_id != task_id
                or result.task_revision != task.task_revision
                or snapshot is None
                or (snapshot is not None and snapshot.task_revision != result.task_revision)
                or (snapshot is not None and snapshot.portfolio_plan_id != task.portfolio_plan_id)
                or result.instruction_digest != snapshot.instruction_digest
                or (snapshot_cluster_id is not None and result.cluster_id != snapshot_cluster_id)
                or (snapshot_batch_id is not None and result.dispatch_batch_id != snapshot_batch_id)
                or (
                    snapshot_action_digest is not None
                    and result.action_digest != snapshot_action_digest
                )
            ):
                consistency_events.append(
                    _build_consistency_event(
                        error_code="WORKER_ATTEMPT_MISMATCH",
                        task_id=task_id,
                        expected_attempt_id=current_attempt_id,
                        received_attempt_id=result.attempt_id,
                        action="ignored",
                        details=(
                            f"Expected revision {task.task_revision} and digest "
                            f"{snapshot.instruction_digest if snapshot else 'missing'}, "
                            f"received revision {result.task_revision} and digest "
                            f"{result.instruction_digest}."
                        ),
                    )
                )
                errors.append(
                    f"supervisor: ignored mismatched worker result for {task_id} "
                    f"attempt {result.attempt_id}."
                )
                processed_worker_attempt_ids.add(current_attempt_id)
                new_worker_attempt_ids.append(current_attempt_id)
                continue

            execution = result.execution_diagnostics
            result_status = result.status
            raw_executed_versions = list(
                dict.fromkeys(
                    [
                        *result.executed_versions,
                        *execution.executed_versions,
                    ]
                )
            )
            reported_effective_version = execution.effective_target_version or (
                raw_executed_versions[-1] if raw_executed_versions else None
            )
            effective_version = (
                reported_effective_version or snapshot.selected_version
                if result_status == AgentActionStatus.SUCCESS
                else None
            )
            reported_effective_dependency_type = execution.effective_dependency_type
            effective_dependency_type = (
                reported_effective_dependency_type or snapshot.target_dependency_type
                if result_status == AgentActionStatus.SUCCESS
                else None
            )
            attempted_versions = list(
                dict.fromkeys(
                    [
                        *execution.attempted_versions,
                        *raw_executed_versions,
                        *([reported_effective_version] if reported_effective_version else []),
                    ]
                )
            )
            allowed_versions = list(snapshot.allowed_target_versions)
            if not allowed_versions and snapshot.selected_version:
                allowed_versions = [snapshot.selected_version]
            normalized_executed = {
                version.strip().lstrip("vV").lower()
                for version in [*raw_executed_versions, reported_effective_version]
                if version
            }
            normalized_allowed = {
                version.strip().lstrip("vV").lower() for version in allowed_versions if version
            }
            unexpected_executed = normalized_executed - normalized_allowed
            if unexpected_executed:
                result_status = AgentActionStatus.SURRENDER
                consistency_events.append(
                    _build_consistency_event(
                        error_code="EXECUTED_VERSION_MISMATCH",
                        task_id=task_id,
                        expected_attempt_id=current_attempt_id,
                        received_attempt_id=result.attempt_id,
                        action="replanned",
                        details=(
                            f"Supervisor-approved versions were {', '.join(sorted(normalized_allowed))}; "
                            f"worker executed {', '.join(raw_executed_versions or [reported_effective_version or 'unknown'])}."
                        ),
                    )
                )
                errors.append(
                    f"supervisor: rejected worker result for {task_id} because the "
                    "executed version was outside the committed candidate allowlist."
                )
            allowed_dependency_types = list(snapshot.allowed_dependency_types)
            if not allowed_dependency_types and snapshot.target_dependency_type:
                allowed_dependency_types = [snapshot.target_dependency_type]
            observed_dependency_type = (
                str(reported_effective_dependency_type).strip()
                if reported_effective_dependency_type
                else None
            )
            normalized_effective_type = (
                effective_dependency_type.strip() if effective_dependency_type else None
            )
            normalized_allowed_dependency_types = {
                value.strip().lower() for value in allowed_dependency_types if value
            }
            if (
                observed_dependency_type
                and observed_dependency_type.lower() not in normalized_allowed_dependency_types
            ):
                result_status = AgentActionStatus.SURRENDER
                consistency_events.append(
                    _build_consistency_event(
                        error_code="EXECUTED_DEPENDENCY_TYPE_MISMATCH",
                        task_id=task_id,
                        expected_attempt_id=current_attempt_id,
                        received_attempt_id=result.attempt_id,
                        action="replanned",
                        details=(
                            f"Supervisor-approved dependency types were {', '.join(allowed_dependency_types)}; "
                            f"worker executed {observed_dependency_type}."
                        ),
                    )
                )
                errors.append(
                    f"supervisor: rejected worker result for {task_id} because the "
                    "executed dependency type was outside the committed candidate allowlist."
                )
            if result_status != AgentActionStatus.SUCCESS:
                # Effective values describe a successful committed
                # transaction only. Attempted but rejected candidates remain
                # in the attempted lists and must never be presented as the
                # effective Supervisor-approved result.
                effective_version = None
                effective_dependency_type = None
            else:
                effective_dependency_type = normalized_effective_type
            if (
                result_status == AgentActionStatus.SUCCESS
                and task.strategy == RoutingStrategy.CODE_WORKAROUND
            ):
                diag = result.execution_diagnostics
                val_files_match = set(diag.validated_files) == set(result.changed_files)
                overall_status = diag.per_gate_results.get("overall_status")
                structured_validation_passed = str(overall_status) in {
                    "PASS",
                    "WorkaroundValidationStatus.PASS",
                }
                is_valid = (
                    diag.validation_passed
                    and diag.validation_calls > 0
                    and val_files_match
                    and structured_validation_passed
                )
                if not is_valid:
                    result_status = AgentActionStatus.SURRENDER
                    errors.append(
                        f"supervisor: rejected workaround worker result for {task_id} due to invalid validation state."
                    )
            prior = retry_diagnostics_by_task.get(task_id)
            if prior is None:
                prior = UpdateRetryDiagnostics(task_id=task_id)
            group = group_by_id.get(task.parent_group_id)
            parent_name, _, parent_type = (
                group_parent_context(group) if group is not None else (None, None, None)
            )
            if task.strategy_stage == SCARemediationStage.PACKAGE_OVERRIDE:
                target_package_name = (
                    snapshot.target_package_name
                    or task.target_package_name
                    or (group.vulnerable_component if group is not None else None)
                )
                target_dependency_type = (
                    snapshot.target_dependency_type
                    or task.target_dependency_type
                    or (_override_dependency_type(group) if group is not None else None)
                )
            else:
                target_package_name = (
                    snapshot.target_package_name
                    or task.target_package_name
                    or (parent_name if group is not None and is_transitive_group(group) else None)
                    or (group.vulnerable_component if group is not None else None)
                )
                target_dependency_type = (
                    snapshot.target_dependency_type
                    or task.target_dependency_type
                    or (parent_type if group is not None and is_transitive_group(group) else None)
                )
            attempted_versions_by_target = dict(prior.attempted_versions_by_target)
            if target_package_name and attempted_versions:
                attempted_versions_by_target[target_package_name] = list(
                    dict.fromkeys(
                        [
                            *attempted_versions_by_target.get(target_package_name, []),
                            *attempted_versions,
                        ]
                    )
                )
            attempted_dependency_types = list(
                dict.fromkeys(
                    prior.attempted_dependency_types
                    + ([observed_dependency_type] if observed_dependency_type else [])
                )
            )
            used_overrides = (
                prior.used_overrides
                or task.strategy_stage == SCARemediationStage.PACKAGE_OVERRIDE
                or target_dependency_type in _OVERRIDE_DEPENDENCY_TYPES
            )
            retry_diagnostics_by_task[task_id] = prior.model_copy(
                update={
                    "committed_attempt_id": current_attempt_id,
                    "attempted_versions": list(
                        dict.fromkeys(prior.attempted_versions + attempted_versions)
                    ),
                    "executed_versions": list(
                        dict.fromkeys(
                            prior.executed_versions
                            + list(result.executed_versions or execution.executed_versions)
                        )
                    ),
                    "effective_target_version": effective_version or prior.effective_target_version,
                    "effective_dependency_type": effective_dependency_type
                    or prior.effective_dependency_type,
                    "attempted_dependency_types": attempted_dependency_types,
                    "candidate_versions_considered": list(
                        dict.fromkeys(prior.candidate_versions_considered + allowed_versions)
                    ),
                    "candidate_dependency_types": list(
                        dict.fromkeys(prior.candidate_dependency_types + allowed_dependency_types)
                    ),
                    "selected_version": task.selected_version,
                    "strategy_stage": task.strategy_stage,
                    "exhausted_update_path": task.exhausted_update_path,
                    "target_package_name": target_package_name or prior.target_package_name,
                    "target_dependency_type": target_dependency_type
                    or prior.target_dependency_type,
                    "attempted_versions_by_target": attempted_versions_by_target,
                    "used_overrides": used_overrides,
                    "instruction_digest": snapshot.instruction_digest,
                    "failure_reason": (
                        " | ".join(result.errors)
                        if result_status == AgentActionStatus.SURRENDER
                        else prior.failure_reason
                    ),
                    "reasoning_summary": (
                        result.action_summary.summary
                        if result.action_summary is not None
                        else prior.reasoning_summary
                    ),
                }
            )
            if result.replay_plan is not None:
                workaround_replay_plans_by_task[task_id] = result.replay_plan
            if result_status == AgentActionStatus.SUCCESS:
                # Package removal still needs the normal QA install/test
                # checks. QA deliberately skips only ODC for this stage;
                # keep every successful attempt open so the snapshot is
                # consumed by QA before the task becomes terminal.
                _commit_task_transition(
                    task_queue,
                    task_id,
                    updates={"status": TaskStatus.OPTIMISTICALLY_FIXED},
                )
            elif task.strategy == RoutingStrategy.CODE_WORKAROUND:
                if task.no_fix_stage is not None:
                    transition, reset_workspace = _no_fix_failure_transition(
                        task,
                        group_by_id.get(task.parent_group_id),
                        failure_feedback=(
                            " | ".join(result.errors)
                            or (
                                result.action_summary.summary
                                if result.action_summary is not None
                                else None
                            )
                        ),
                    )
                    _commit_task_transition(
                        task_queue,
                        task_id,
                        updates=transition,
                        close_attempt=True,
                        clear_selected_version=True,
                    )
                    if reset_workspace:
                        reset_plan = _reset_no_fix_replay_plan(
                            workaround_replay_plans_by_task.get(task_id)
                        )
                        if reset_plan is None:
                            workaround_replay_plans_by_task.pop(task_id, None)
                        else:
                            workaround_replay_plans_by_task[task_id] = reset_plan
                else:
                    # A surrender is a completed worker outcome, not an active
                    # worker input. Close it before the next routing decision so
                    # a terminal workaround task cannot reach teardown with a
                    # live current attempt.
                    _commit_task_transition(
                        task_queue,
                        task_id,
                        updates={"status": TaskStatus.UNFIXABLE},
                        close_attempt=True,
                        clear_selected_version=True,
                    )
            else:
                # Failed update attempts are replanned in this same
                # supervisor pass. Detach the consumed attempt first so the
                # planner cannot observe a new stage paired with an old
                # immutable snapshot.
                # A failed attempt consumes a task-local retry only.  The
                # outer plan owns stage/version transitions; Supervisor must
                # never infer a new stage or fixed version from group data.
                failure_updates: dict[str, Any] = {
                    "status": TaskStatus.NEEDS_RETRY,
                    "retry_count": task.retry_count + 1,
                }
                _commit_task_transition(
                    task_queue,
                    task_id,
                    updates=failure_updates,
                    close_attempt=True,
                    clear_selected_version=True,
                )
            processed_worker_attempt_ids.add(current_attempt_id)
            new_worker_attempt_ids.append(current_attempt_id)
            continue

    if atomic_cluster_task_ids:
        cluster_worker_results = [
            worker_results_by_attempt.get(task_queue[task_id].current_attempt_id)
            for task_id in atomic_cluster_task_ids
        ]
        worker_batch_succeeded = bool(cluster_worker_results) and all(
            result is not None
            and result.status == AgentActionStatus.SUCCESS
            and result.execution_diagnostics.validation_passed
            for result in cluster_worker_results
        )
        if worker_batch_succeeded:
            for task_id in atomic_cluster_task_ids:
                _commit_task_transition(
                    task_queue,
                    task_id,
                    updates={"status": TaskStatus.OPTIMISTICALLY_FIXED},
                )
        else:
            for task_id in atomic_cluster_task_ids:
                task = task_queue[task_id]
                baseline = raw_task_queue.get(task_id)
                _commit_task_transition(
                    task_queue,
                    task_id,
                    updates={
                        "status": TaskStatus.NEEDS_RETRY,
                        "retry_count": max(
                            task.retry_count,
                            (baseline.retry_count if baseline is not None else task.retry_count)
                            + 1,
                        ),
                    },
                    close_attempt=True,
                )
            errors.append(
                "supervisor: atomic package-cluster worker failure restored the complete batch."
            )

    # ------------------------------------------------------------------
    # 3. Ingest QA results (active targets only, when qa_completed)
    # ------------------------------------------------------------------
    qa_evaluations: dict[str, QAEvaluation] = _normalize_qa_evaluations_for_tasks(
        dict(state.get("qa_evaluations", {})),
        task_queue,
        active_target_task_ids,
    )
    qa_result_task_ids: set[str] = set()
    new_qa_attempt_ids: list[str] = []
    for task_id in active_target_task_ids:
        task = task_queue.get(task_id)
        if task is None or not task.current_attempt_id:
            continue
        qa_result = qa_results_by_attempt.get(task.current_attempt_id)
        for stale_result in qa_results_by_attempt.values():
            if (
                stale_result.task_id == task_id
                and stale_result.attempt_id != task.current_attempt_id
                and stale_result.attempt_id not in processed_qa_attempt_ids
            ):
                consistency_events.append(
                    _build_consistency_event(
                        error_code="STALE_QA_RESULT",
                        task_id=task_id,
                        expected_attempt_id=task.current_attempt_id,
                        received_attempt_id=stale_result.attempt_id,
                        action="ignored",
                        details="QA result belongs to an older task attempt.",
                    )
                )
                processed_qa_attempt_ids.add(stale_result.attempt_id)
                new_qa_attempt_ids.append(stale_result.attempt_id)
        if qa_result is None or task.current_attempt_id in processed_qa_attempt_ids:
            continue
        if task.status in _TERMINAL_STATUSES:
            consistency_events.append(
                _build_consistency_event(
                    error_code="TERMINAL_QA_RESULT_IGNORED",
                    task_id=task_id,
                    expected_attempt_id=task.current_attempt_id,
                    received_attempt_id=qa_result.attempt_id,
                    action="ignored",
                    details="A late QA result cannot reopen a terminal task.",
                )
            )
            processed_qa_attempt_ids.add(task.current_attempt_id)
            new_qa_attempt_ids.append(task.current_attempt_id)
            continue
        snapshot = attempt_snapshots_by_id.get(task.current_attempt_id)
        qa_requires_rerun = False
        snapshot_batch_id = (
            snapshot.dispatch_batch_id
            if snapshot is not None and isinstance(snapshot.dispatch_batch_id, str)
            else None
        )
        snapshot_cluster_id = (
            snapshot.cluster_id
            if snapshot is not None and isinstance(snapshot.cluster_id, str)
            else None
        )
        snapshot_action_digest = (
            snapshot.action_digest
            if snapshot is not None and isinstance(snapshot.action_digest, str)
            else None
        )
        if (
            qa_result.task_id != task_id
            or qa_result.task_revision != task.task_revision
            or snapshot is None
            or (snapshot is not None and snapshot.task_revision != qa_result.task_revision)
            or (snapshot is not None and snapshot.portfolio_plan_id != task.portfolio_plan_id)
            or (snapshot_cluster_id is not None and qa_result.cluster_id != snapshot_cluster_id)
            or (snapshot_batch_id is not None and qa_result.dispatch_batch_id != snapshot_batch_id)
            or (
                snapshot_action_digest is not None
                and qa_result.action_digest != snapshot_action_digest
            )
        ):
            consistency_events.append(
                _build_consistency_event(
                    error_code="QA_ATTEMPT_MISMATCH",
                    task_id=task_id,
                    expected_attempt_id=task.current_attempt_id,
                    received_attempt_id=qa_result.attempt_id,
                    action="ignored",
                    details=(
                        f"Expected revision {task.task_revision}; "
                        f"received revision {qa_result.task_revision}."
                    ),
                )
            )
            errors.append(
                f"supervisor: ignored mismatched QA result for {task_id} "
                f"attempt {qa_result.attempt_id}."
            )
        else:
            qa_evaluations[task_id] = qa_result.evaluation
            qa_result_task_ids.add(task_id)
            qa_requires_rerun = (
                qa_result.evaluation.evidence_inconclusive or qa_result.evaluation.contract_error
            )
            if not qa_requires_rerun and not atomic_cluster_task_ids:
                # QA closes the worker attempt before any status or stage
                # change. The next planner proposal must observe a task with
                # no active worker input; otherwise it can see the new retry
                # stage paired with the old attempt snapshot.
                _commit_task_transition(
                    task_queue,
                    task_id,
                    updates={},
                    close_attempt=True,
                    clear_selected_version=True,
                )
            else:
                # Keep the committed attempt open so QA can be rerun without
                # creating a worker retry or losing its provenance.
                logger.info(
                    "supervisor: preserving attempt for non-remediation QA rerun on %s.",
                    task_id,
                )
        if not qa_requires_rerun:
            processed_qa_attempt_ids.add(task.current_attempt_id)
            new_qa_attempt_ids.append(task.current_attempt_id)
    portfolio_escalation = state.get("portfolio_escalation")
    if atomic_cluster_task_ids and state.get("status") in {"qa_completed", "qa_failed"}:
        cluster_evaluations = {
            task_id: qa_evaluations.get(task_id) for task_id in atomic_cluster_task_ids
        }
        all_evaluations_present = all(
            evaluation is not None for evaluation in cluster_evaluations.values()
        )
        has_inconclusive = any(
            evaluation is not None
            and (evaluation.contract_error or evaluation.evidence_inconclusive)
            for evaluation in cluster_evaluations.values()
        )
        has_real_failure = not all_evaluations_present or any(
            evaluation is not None
            and not evaluation.passed
            and not evaluation.contract_error
            and not evaluation.evidence_inconclusive
            for evaluation in cluster_evaluations.values()
        )
        if not has_real_failure and has_inconclusive:
            for task_id in atomic_cluster_task_ids:
                _commit_task_transition(
                    task_queue,
                    task_id,
                    updates={"status": TaskStatus.OPTIMISTICALLY_FIXED},
                )
        elif not has_real_failure:
            for task_id in atomic_cluster_task_ids:
                _commit_task_transition(
                    task_queue,
                    task_id,
                    updates={},
                    close_attempt=True,
                )
                _commit_task_transition(
                    task_queue,
                    task_id,
                    updates={"status": TaskStatus.QA_PASSED},
                )
        else:
            peer_pairs: set[tuple[str, str]] = set()
            peer_evidence: dict[str, dict[str, Any]] = {}
            unresolved_peer_diagnostics: list[str] = []
            for task_id, evaluation in cluster_evaluations.items():
                if evaluation is None:
                    continue
                escalation, unresolved = _peer_conflict_escalation(
                    task_id,
                    evaluation,
                    task_queue,
                    group_by_id,
                )
                unresolved_peer_diagnostics.extend(unresolved)
                if escalation is None:
                    continue
                peer_pairs.update(tuple(pair) for pair in escalation["peer_conflict_pairs"])
                for evidence in escalation.get("evidence", []):
                    peer_evidence[json.dumps(evidence, sort_keys=True)] = evidence
            errors.extend(unresolved_peer_diagnostics)
            if peer_pairs:
                portfolio_escalation = {
                    "reason": "PEER_CONFLICT_ESCALATION",
                    "peer_conflict_pairs": sorted(peer_pairs),
                    "evidence": [peer_evidence[key] for key in sorted(peer_evidence)],
                }
            for task_id in atomic_cluster_task_ids:
                task = task_queue[task_id]
                _commit_task_transition(
                    task_queue,
                    task_id,
                    updates={
                        "status": TaskStatus.NEEDS_RETRY,
                        "retry_count": task.retry_count + 1,
                    },
                    close_attempt=True,
                )
            errors.append(
                "supervisor: atomic package-cluster QA failure kept every member non-passed."
            )
    portfolio_replan_request = state.get("portfolio_replan_request")
    if not isinstance(portfolio_replan_request, PortfolioReplanRequest):
        escalation = portfolio_escalation
        if isinstance(escalation, dict) and escalation.get("reason"):
            try:
                portfolio_replan_request = PortfolioReplanRequest(
                    reason=str(escalation["reason"]),
                    peer_conflict_pairs=escalation.get("peer_conflict_pairs", []),
                    forced_singleton_task_ids=escalation.get("forced_singleton_task_ids", []),
                    triggering_attempt_id=escalation.get("triggering_attempt_id"),
                    triggering_scan_id=escalation.get("triggering_scan_id"),
                    source_portfolio_plan_id=(
                        committed_plan.portfolio_plan_id
                        if committed_plan is not None
                        else escalation.get("source_portfolio_plan_id")
                    ),
                )
            except Exception:  # noqa: BLE001
                portfolio_replan_request = None

    auto_new_constraints: list[str] = []

    if not atomic_cluster_task_ids and state.get("status") in {"qa_completed", "qa_failed"}:
        for resolved_t_id, evaluation in qa_evaluations.items():
            task_for_result = task_queue.get(resolved_t_id)
            if (
                task_for_result is not None
                and task_for_result.current_attempt_id is not None
                and resolved_t_id not in qa_result_task_ids
            ):
                # A compatibility QA projection without the current attempt
                # identity cannot mutate an attempted task.
                continue
            task = task_queue[resolved_t_id]
            if task.status in (TaskStatus.UNFIXABLE, TaskStatus.QA_PASSED):
                continue
            if evaluation.contract_error:
                # A malformed judge response is a QA-contract failure, not a
                # remediation failure. Requeue QA without advancing the
                # strategy stage or consuming the worker retry budget.
                _commit_task_transition(
                    task_queue,
                    resolved_t_id,
                    updates={"status": TaskStatus.OPTIMISTICALLY_FIXED},
                )
                errors.append(
                    f"supervisor: QA contract error for {resolved_t_id}; "
                    "requeued QA without consuming a remediation retry."
                )
                continue
            if evaluation.evidence_inconclusive:
                _commit_task_transition(
                    task_queue,
                    resolved_t_id,
                    updates={"status": TaskStatus.OPTIMISTICALLY_FIXED},
                )
                errors.append(
                    f"supervisor: dependency evidence inconclusive for {resolved_t_id}; "
                    "requeued QA without consuming a remediation retry."
                )
                continue
            if evaluation.passed:
                _commit_task_transition(
                    task_queue,
                    resolved_t_id,
                    updates={"status": TaskStatus.QA_PASSED},
                )
                group = group_by_id.get(task.parent_group_id)
                if group:
                    constraint = _constraint_entry_for_task(task, group)
                    if (
                        constraint
                        and constraint not in existing_constraints
                        and constraint not in auto_new_constraints
                    ):
                        auto_new_constraints.append(constraint)
            else:
                peer_escalation, unresolved_peer_diagnostics = _peer_conflict_escalation(
                    resolved_t_id,
                    evaluation,
                    task_queue,
                    group_by_id,
                )
                if unresolved_peer_diagnostics:
                    errors.extend(unresolved_peer_diagnostics)
                if peer_escalation is not None:
                    portfolio_escalation = peer_escalation
                    _commit_task_transition(
                        task_queue,
                        resolved_t_id,
                        updates={
                            "status": TaskStatus.NEEDS_RETRY,
                            "retry_count": task.retry_count + 1,
                        },
                        close_attempt=True,
                    )
                    errors.append(
                        f"supervisor: peer conflict expanded the package portfolio for {resolved_t_id}."
                    )
                    continue
                if task.no_fix_stage is not None:
                    no_fix_updates, reset_workspace = _no_fix_failure_transition(
                        task,
                        group_by_id.get(task.parent_group_id),
                        evaluation=evaluation,
                    )
                    _commit_task_transition(
                        task_queue,
                        resolved_t_id,
                        updates=no_fix_updates,
                        clear_selected_version=True,
                    )
                    if reset_workspace:
                        reset_plan = _reset_no_fix_replay_plan(
                            workaround_replay_plans_by_task.get(resolved_t_id)
                        )
                        if reset_plan is None:
                            workaround_replay_plans_by_task.pop(resolved_t_id, None)
                        else:
                            workaround_replay_plans_by_task[resolved_t_id] = reset_plan
                    continue

                # QA failure stays within the committed task decision.  Do
                # not advance strategy stages or derive a package version in
                # the inner loop; retry planning can only consume the
                # task's solver-approved alternatives.
                task_updates = {
                    "status": TaskStatus.NEEDS_RETRY,
                    "retry_count": task.retry_count + 1,
                }
                _commit_task_transition(
                    task_queue,
                    resolved_t_id,
                    updates=task_updates,
                    close_attempt=True,
                    clear_selected_version=True,
                )

    # ------------------------------------------------------------------
    # Normalize any escalation emitted while reconciling the final QA result.
    if not isinstance(portfolio_replan_request, PortfolioReplanRequest):
        escalation = portfolio_escalation
        if isinstance(escalation, dict) and escalation.get("reason"):
            try:
                portfolio_replan_request = PortfolioReplanRequest(
                    reason=str(escalation["reason"]),
                    peer_conflict_pairs=escalation.get("peer_conflict_pairs", []),
                    forced_singleton_task_ids=escalation.get("forced_singleton_task_ids", []),
                    triggering_attempt_id=escalation.get("triggering_attempt_id"),
                    triggering_scan_id=escalation.get("triggering_scan_id"),
                    source_portfolio_plan_id=(
                        committed_plan.portfolio_plan_id
                        if committed_plan is not None
                        else escalation.get("source_portfolio_plan_id")
                    ),
                )
            except Exception:  # noqa: BLE001
                portfolio_replan_request = None
    # 4. Mark UNFIXABLE tasks that hit the retry cap
    # ------------------------------------------------------------------
    for task_id, task in task_queue.items():
        if (
            task.no_fix_stage == NoFixMitigationStage.UNFIXABLE
            and task.status not in _TERMINAL_STATUSES
        ):
            _commit_task_transition(
                task_queue,
                task_id,
                updates={"status": TaskStatus.UNFIXABLE},
                close_attempt=task.current_attempt_id is not None,
                clear_selected_version=task.selected_version is not None,
            )
        if task.no_fix_stage == NoFixMitigationStage.UNFIXABLE or (
            task.no_fix_stage is not None and task.status in _TERMINAL_STATUSES
        ):
            retry_plans_by_task.pop(task_id, None)
            workaround_replay_plans_by_task.pop(task_id, None)

    for task_id, task in task_queue.items():
        if (
            task.status == TaskStatus.NEEDS_RETRY
            and task.retry_count >= MAX_RETRIES
            and task.no_fix_stage is None
            and not _is_exhausted_update_pivot_candidate(
                task,
                retry_diagnostics_by_task.get(task_id),
            )
        ):
            _commit_task_transition(
                task_queue,
                task_id,
                updates={"status": TaskStatus.UNFIXABLE},
                close_attempt=task.current_attempt_id is not None,
                clear_selected_version=task.selected_version is not None,
            )
            logger.info(
                "supervisor: task '%s' marked UNFIXABLE after %d retries.",
                task_id,
                task.retry_count,
            )

    # Keep diagnostics aligned with terminal task state without emitting a
    # projection-repair event. The selected version is no longer dispatchable,
    # while target and attempt evidence remains useful in the final report.
    for task_id, task in task_queue.items():
        if task.status not in _TERMINAL_STATUSES:
            continue
        diagnostics = retry_diagnostics_by_task.get(task_id)
        if diagnostics is not None and diagnostics.selected_version is not None:
            retry_diagnostics_by_task[task_id] = diagnostics.model_copy(
                update={"selected_version": None}
            )

    # ------------------------------------------------------------------
    # 4.5. Reconcile terminal workaround children before routing.
    # ------------------------------------------------------------------
    stale_pivot_parent_ids = _reconcile_terminal_pivot_parents(
        task_queue,
        qa_evaluations,
        retry_diagnostics_by_task,
        retry_plans_by_task,
        group_by_id,
    )
    if stale_pivot_parent_ids:
        errors.append(
            "supervisor: terminalized stale pivot parent task(s) after their "
            "workaround child reached a terminal state: "
            f"{stale_pivot_parent_ids}."
        )
    # 5. Short-circuit: if an active task is optimistically_fixed â†’ qa_critic
    # ------------------------------------------------------------------
    decision: SupervisorDecision | None = None
    if state.get("status") != "qa_completed" and active_target_task_ids:
        active_qa_ready = _qa_ready_task_ids(
            task_queue,
            preferred_ids=active_target_task_ids,
            limit=QA_DISPATCH_LIMIT,
        )
        if active_qa_ready:
            decision = SupervisorDecision(
                next_node="qa_critic",
                target_task_ids=active_qa_ready,
                instructions="Run QA on the current remediated task before starting more remediation.",
                decision_reason=(
                    f"Routing task '{active_qa_ready[0]}' to QA after a successful worker attempt."
                ),
            )

    # The Supervisor is not permitted to bypass the post-QA triage handoff. This is
    # deliberately applied after the optimistic QA guard so Supervisor stays
    # the sole routing authority while preserving the required order:
    # worker -> supervisor -> QA -> supervisor -> triage -> supervisor.
    if state.get("triage_required") and state.get("status") in {
        "qa_completed",
        "qa_failed",
        "final_scan_completed",
    }:
        decision = SupervisorDecision(
            next_node="triage",
            target_task_ids=[],
            instructions="Route the completed QA scan to post-remediation triage before dispatching more work.",
            decision_reason="Supervisor guardrail: triage_required is set after a parseable QA scan.",
        )

    # Short-circuit: all remaining non-terminal tasks are optimistically_fixed
    if decision is None:
        tasks = list(task_queue.values())
        non_terminal = [t for t in tasks if t.status not in _TERMINAL_STATUSES]
        if not non_terminal:
            decision = SupervisorDecision(
                next_node="teardown",
                target_task_ids=[],
                instructions="All tasks are terminal. Proceeding to teardown.",
                decision_reason="No actionable tasks remain.",
            )
        else:
            qa_ready = _qa_ready_task_ids(task_queue, limit=QA_DISPATCH_LIMIT)
            if qa_ready:
                decision = SupervisorDecision(
                    next_node="qa_critic",
                    target_task_ids=qa_ready,
                    instructions="Run QA on the next remaining optimistically fixed task.",
                    decision_reason=f"Routing task '{qa_ready[0]}' to QA.",
                )

    # ------------------------------------------------------------------
    # 6. Deterministic retry planning
    # ------------------------------------------------------------------
    if decision is None and _needs_planner(
        task_queue,
        qa_evaluations,
        retry_diagnostics_by_task,
        str(state.get("status") or ""),
    ):
        parsed_diagnostics, parsed_plans = _run_deterministic_retry_planner(
            task_queue,
            group_by_id,
            retry_diagnostics_by_task,
        )
        planner_violations = _planner_plan_violations(
            parsed_plans,
            task_queue,
            parsed_diagnostics,
        )
        if planner_violations:
            errors.extend(
                f"supervisor: deterministic retry planning: {violation}"
                for violation in planner_violations
            )
            parsed_diagnostics, parsed_plans = _repair_invalid_planner_plans(
                parsed_plans,
                parsed_diagnostics,
                task_queue,
                group_by_id,
                violations=planner_violations,
            )
            repair_violations = _planner_plan_violations(
                parsed_plans,
                task_queue,
                parsed_diagnostics,
            )
            if repair_violations:
                errors.extend(
                    f"supervisor: deterministic retry repair: {violation}"
                    for violation in repair_violations
                )
        retry_diagnostics_by_task = parsed_diagnostics
        if not _planner_plan_violations(parsed_plans, task_queue, parsed_diagnostics):
            _commit_retry_plans(
                task_queue,
                retry_diagnostics_by_task,
                retry_plans_by_task,
                parsed_plans,
            )
            if any(plan.action == "pivot_workaround" for plan in parsed_plans.values()):
                decision = _deterministic_routing(
                    task_queue,
                    group_by_id,
                    qa_evaluations,
                    retry_diagnostics_by_task,
                    action_summaries=action_summaries,
                    active_target_task_ids=active_target_task_ids,
                    current_status=str(state.get("status") or ""),
                    portfolio_plan=state.get("portfolio_plan"),
                )

    # Deterministic retry planning above is the only Supervisor planning path.

    # Re-apply optimistic short-circuit after retry planning.
    if state.get("status") != "qa_completed" and active_target_task_ids:
        active_qa_ready = _qa_ready_task_ids(
            task_queue,
            preferred_ids=active_target_task_ids,
            limit=QA_DISPATCH_LIMIT,
        )
        if active_qa_ready:
            decision = SupervisorDecision(
                next_node="qa_critic",
                target_task_ids=active_qa_ready,
                instructions="Run QA on the current remediated task before starting more remediation.",
                decision_reason=(
                    f"Routing task '{active_qa_ready[0]}' to QA after a successful worker attempt."
                ),
            )

    # ------------------------------------------------------------------
    # 6b. Deterministic authority boundary
    # ------------------------------------------------------------------
    # A live attempt anywhere in the queue blocks an outer replan.  Looking
    # only at active_target_task_ids is unsafe because portfolio nodes may
    # clear that projection while leaving the immutable task attempt open.
    active_attempt_is_open = any(
        task.current_attempt_id is not None and task.status not in _TERMINAL_STATUSES
        for task in task_queue.values()
    )
    portfolio_replan_required = bool(
        state.get("portfolio_plan") is not None
        and not active_attempt_is_open
        and _portfolio_plan_is_stale(
            state.get("portfolio_plan"),
            task_queue,
            valid_groups,
            state.get("repo_root"),
        )
    )
    if portfolio_replan_required:
        deterministic_decision = SupervisorDecision(
            decision_code=DecisionCode.PORTFOLIO_PLAN_REQUIRED,
            next_node="portfolio",
            target_task_ids=[],
            instructions="Rebuild the package-group portfolio plan after task reconciliation.",
            decision_reason="The committed portfolio plan is stale after a completed attempt.",
        )
    else:
        deterministic_decision = _deterministic_routing(
            task_queue,
            group_by_id,
            qa_evaluations,
            retry_diagnostics_by_task,
            action_summaries=action_summaries,
            active_target_task_ids=active_target_task_ids,
            current_status=str(state.get("status") or ""),
            triage_required=bool(state.get("triage_required")),
            portfolio_plan=state.get("portfolio_plan"),
        )
    # The deterministic decision is authoritative. QA feedback below is
    # carried only when Python produced it for the selected task; no model
    # can add worker feedback or constraints to this projection.
    decision = deterministic_decision
    if portfolio_escalation and portfolio_escalation.get("reason") in {
        "PEER_CONFLICT_ESCALATION",
        "DELTA_ISOLATION_ATTRIBUTION",
    }:
        decision = SupervisorDecision(
            decision_code=(
                DecisionCode.PEER_CONFLICT_ESCALATION
                if portfolio_escalation.get("reason") == "PEER_CONFLICT_ESCALATION"
                else DecisionCode.PORTFOLIO_PLAN_REQUIRED
            ),
            next_node="portfolio",
            target_task_ids=[],
            instructions="Recompute package portfolio membership from the validated QA delta.",
            decision_reason="QA produced a portfolio-affecting cluster attribution.",
        )

    # ------------------------------------------------------------------
    # 7. Guardrails: validate and clamp (or deterministic fallback)
    # ------------------------------------------------------------------
    pivot_parent_status_by_parent: dict[str, TaskStatus] = {}
    pivot_target_parent_ids: set[str] = set()

    if decision is not None and _no_fix_decision_requires_fallback(decision, task_queue):
        errors.append(
            "supervisor: rejected router decision that attempted to bypass the "
            "deterministic NO_FIX mitigation lifecycle."
        )
        decision = None

    if decision is None:
        decision = _deterministic_routing(
            task_queue,
            group_by_id,
            qa_evaluations,
            retry_diagnostics_by_task,
            action_summaries=action_summaries,
            active_target_task_ids=active_target_task_ids,
            current_status=str(state.get("status") or ""),
            portfolio_plan=state.get("portfolio_plan"),
        )
        logger.info("supervisor: deterministic fallback â†’ next_node=%s", decision.next_node)
        fallback_pivot_strategy_by_parent = {
            req.parent_task_id: req.strategy
            for req in decision.spawn_requests
            if req.parent_task_id in task_queue
            and task_queue[req.parent_task_id].strategy != req.strategy
        }
        for parent_id, new_strategy in fallback_pivot_strategy_by_parent.items():
            pivot_parent_status_by_parent[parent_id] = _parent_status_for_strategy_pivot(
                task_queue[parent_id],
                new_strategy,
                qa_evaluations,
            )
        pivot_target_parent_ids = {
            task_id
            for task_id in decision.target_task_ids
            if task_id in fallback_pivot_strategy_by_parent
        }
    else:
        known_task_ids = set(task_queue.keys())

        raw_pivot_parent_ids = {
            req.parent_task_id
            for req in decision.spawn_requests
            if req.parent_task_id in known_task_ids
            and task_queue[req.parent_task_id].status not in _TERMINAL_STATUSES
            and task_queue[req.parent_task_id].status in _WORKABLE_STATUSES
            and task_queue[req.parent_task_id].strategy != req.strategy
            and decision.next_node == _worker_node_for_strategy(req.strategy)
        }
        raw_pivot_parent_ids.update(
            task_id
            for task_id, new_strategy in decision.updated_task_strategies.items()
            if task_id in known_task_ids
            and task_queue[task_id].status not in _TERMINAL_STATUSES
            and task_queue[task_id].status in _WORKABLE_STATUSES
            and task_queue[task_id].strategy != new_strategy
            and decision.next_node == _worker_node_for_strategy(new_strategy)
        )

        if decision.next_node == "qa_critic":
            valid_target_ids = _qa_ready_task_ids(
                task_queue,
                preferred_ids=list(decision.target_task_ids),
                limit=None if decision.cluster_id else QA_DISPATCH_LIMIT,
            )
        elif decision.next_node == "update_subagent":
            valid_target_ids = _update_worker_task_ids(
                task_queue,
                retry_diagnostics_by_task,
                preferred_ids=list(decision.target_task_ids),
                limit=None if decision.cluster_id else UPDATE_DISPATCH_LIMIT,
            )
        elif decision.next_node == "workaround_subagent":
            valid_target_ids = []
            for t_id in decision.target_task_ids:
                if t_id not in known_task_ids:
                    continue
                task = task_queue[t_id]
                if (
                    task.status in _TERMINAL_STATUSES
                    or task.status not in _WORKABLE_STATUSES
                    or task.current_attempt_id is not None
                ):
                    continue
                if task.strategy == RoutingStrategy.CODE_WORKAROUND or t_id in raw_pivot_parent_ids:
                    valid_target_ids.append(t_id)
            if not valid_target_ids and len(raw_pivot_parent_ids) == 1:
                # A pivot decision may contain the child spawn request but omit
                # the parent target. Recover the single unambiguous parent
                # before applying workaround cardinality validation.
                valid_target_ids = [next(iter(raw_pivot_parent_ids))]
        else:
            valid_target_ids = []

        valid_unfixable_ids = [
            t_id for t_id in decision.unfixable_task_ids if t_id in known_task_ids
        ]

        # Enforce cardinality constraints
        needs_fallback = False
        requested_target_count = len(decision.target_task_ids)
        if decision.next_node == "workaround_subagent" and len(valid_target_ids) != 1:
            logger.warning(
                "supervisor: workaround_subagent needs 1 target, got %d â€” falling back.",
                len(valid_target_ids),
            )
            needs_fallback = True
        elif decision.next_node == "update_subagent" and not valid_target_ids:
            logger.warning("supervisor: update_subagent needs â‰¥1 target, got 0 â€” falling back.")
            needs_fallback = True

        if (
            not needs_fallback
            and not decision.cluster_id
            and decision.next_node == "update_subagent"
            and requested_target_count > UPDATE_DISPATCH_LIMIT
        ):
            logger.warning(
                "supervisor: update_subagent current policy allows exactly 1 target, got %d â€” falling back.",
                requested_target_count,
            )
            needs_fallback = True
        if (
            not needs_fallback
            and not decision.cluster_id
            and decision.next_node == "qa_critic"
            and requested_target_count > QA_DISPATCH_LIMIT
        ):
            logger.warning(
                "supervisor: qa_critic current policy allows exactly 1 target, got %d â€” falling back.",
                requested_target_count,
            )
            needs_fallback = True
        if (
            not needs_fallback
            and not decision.cluster_id
            and decision.next_node == "update_subagent"
            and valid_target_ids
        ):
            has_retry_targets = any(
                task_queue[t_id].status == TaskStatus.NEEDS_RETRY
                or task_queue[t_id].retry_count > 0
                for t_id in valid_target_ids
            )
            has_first_pass_targets = any(
                task_queue[t_id].status != TaskStatus.NEEDS_RETRY
                and task_queue[t_id].retry_count == 0
                for t_id in valid_target_ids
            )
            if has_retry_targets and has_first_pass_targets:
                logger.warning(
                    "supervisor: update_subagent request mixed first-pass and retry tasks â€” falling back."
                )
                needs_fallback = True
        if not needs_fallback and decision.next_node == "qa_critic" and not valid_target_ids:
            logger.warning("supervisor: qa_critic needs at least 1 target, got 0 â€” falling back.")
            needs_fallback = True

        if needs_fallback:
            decision = _deterministic_routing(
                task_queue,
                group_by_id,
                qa_evaluations,
                retry_diagnostics_by_task,
                action_summaries=action_summaries,
                active_target_task_ids=active_target_task_ids,
                current_status=str(state.get("status") or ""),
                portfolio_plan=state.get("portfolio_plan"),
            )
            fallback_pivot_strategy_by_parent = {
                req.parent_task_id: req.strategy
                for req in decision.spawn_requests
                if req.parent_task_id in task_queue
                and task_queue[req.parent_task_id].strategy != req.strategy
            }
            for parent_id, new_strategy in fallback_pivot_strategy_by_parent.items():
                pivot_parent_status_by_parent[parent_id] = _parent_status_for_strategy_pivot(
                    task_queue[parent_id],
                    new_strategy,
                    qa_evaluations,
                )
            pivot_target_parent_ids = {
                task_id
                for task_id in decision.target_task_ids
                if task_id in fallback_pivot_strategy_by_parent
            }
        else:
            # Filter revised_instructions and feedback_by_task to known task IDs
            clean_revised_instructions = {
                k: v
                for k, v in decision.revised_instructions.items()
                if k in known_task_ids and v.strip()
            }
            for task_id in valid_target_ids:
                task = task_queue.get(task_id)
                if task is None or task.no_fix_stage is None:
                    continue
                group = group_by_id.get(task.parent_group_id)
                if task.no_fix_stage == NoFixMitigationStage.VULNERABLE_CODE_REMOVAL:
                    clean_revised_instructions[task_id] = build_no_fix_retry_instruction(
                        task,
                        group,
                        evaluation=qa_evaluations.get(task_id),
                    )
                elif task.no_fix_stage == NoFixMitigationStage.PACKAGE_REMOVAL and group:
                    # The package-removal instruction is supervisor-owned too;
                    # an untrusted decision cannot replace the scoped manifest capability with
                    # arbitrary prose or a generic source-only workaround.
                    clean_revised_instructions[task_id] = build_no_fix_package_removal_instruction(
                        group
                    )
            # The reconciled task queue is authoritative. Preserve the public
            # revised_instructions field while filling it from committed plans
            # when the router omits the field or repeats stale text.
            for task_id in valid_target_ids:
                task = task_queue.get(task_id)
                plan = retry_plans_by_task.get(task_id)
                if (
                    task is not None
                    and task.status == TaskStatus.NEEDS_RETRY
                    and plan is not None
                    and plan.action == "retry_update"
                ):
                    clean_revised_instructions[task_id] = task.instruction
            clean_feedback = {
                k: v for k, v in decision.feedback_by_task.items() if k in known_task_ids
            }
            clean_updated_task_strategies = {
                k: v
                for k, v in decision.updated_task_strategies.items()
                if k in known_task_ids and task_queue[k].status not in _TERMINAL_STATUSES
            }
            missing_retry_revisions = _missing_retry_revised_instructions(
                decision.next_node,
                valid_target_ids,
                clean_revised_instructions,
                task_queue,
            )
            if missing_retry_revisions:
                errors.append(
                    "supervisor: rejected update_subagent retry dispatch without task-specific "
                    f"revised_instructions for {missing_retry_revisions}; replanning deterministically."
                )
                recovery_tasks = {
                    task_id: task_queue[task_id]
                    for task_id in missing_retry_revisions
                    if task_id in task_queue
                    and task_queue[task_id].strategy == RoutingStrategy.VERSION_BUMP
                }
                recovery_inputs = {
                    task_id: retry_diagnostics_by_task.get(
                        task_id,
                        UpdateRetryDiagnostics(
                            task_id=task_id,
                            strategy_stage=task_queue[task_id].strategy_stage,
                        ),
                    )
                    for task_id in recovery_tasks
                }
                if recovery_tasks:
                    recovered_diagnostics, recovered_plans = _run_deterministic_retry_planner(
                        recovery_tasks,
                        group_by_id,
                        recovery_inputs,
                    )
                    retry_diagnostics_by_task.update(recovered_diagnostics)
                    _commit_retry_plans(
                        task_queue,
                        retry_diagnostics_by_task,
                        retry_plans_by_task,
                        recovered_plans,
                    )
                decision = _deterministic_routing(
                    task_queue,
                    group_by_id,
                    qa_evaluations,
                    retry_diagnostics_by_task,
                    action_summaries=action_summaries,
                    active_target_task_ids=active_target_task_ids,
                    current_status=str(state.get("status") or ""),
                    triage_required=bool(state.get("triage_required")),
                    portfolio_plan=state.get("portfolio_plan"),
                )
                valid_target_ids = list(decision.target_task_ids)
                valid_unfixable_ids = list(decision.unfixable_task_ids)
                clean_revised_instructions = dict(decision.revised_instructions)
                clean_feedback = dict(decision.feedback_by_task)
                clean_updated_task_strategies = dict(decision.updated_task_strategies)

            # Validate task_status_updates â€” only known tasks, only terminal statuses
            clean_status_updates: dict[str, TaskStatus] = {}
            _allowed_statuses = {TaskStatus.QA_PASSED, TaskStatus.UNFIXABLE}
            for t_id, new_status in decision.task_status_updates.items():
                if t_id not in known_task_ids:
                    errors.append(
                        f"supervisor: task_status_updates rejected unknown task_id '{t_id}'."
                    )
                    continue
                if new_status not in _allowed_statuses:
                    errors.append(
                        f"supervisor: task_status_updates rejected disallowed status "
                        f"'{new_status}' for task '{t_id}'."
                    )
                    continue
                clean_status_updates[t_id] = new_status

            clean_spawn_requests = [
                req
                for req in decision.spawn_requests
                if req.parent_task_id in known_task_ids
                and task_queue[req.parent_task_id].status not in _TERMINAL_STATUSES
                and task_queue[req.parent_task_id].no_fix_stage is None
            ]
            pivot_strategy_by_parent: dict[str, RoutingStrategy] = {
                req.parent_task_id: req.strategy
                for req in clean_spawn_requests
                if task_queue[req.parent_task_id].strategy != req.strategy
            }
            legacy_pivot_strategy_updates = {
                task_id: new_strategy
                for task_id, new_strategy in clean_updated_task_strategies.items()
                if task_queue[task_id].strategy != new_strategy
                and task_id not in pivot_strategy_by_parent
            }
            malformed_pivot_parent_ids: list[str] = []
            for task_id, new_strategy in legacy_pivot_strategy_updates.items():
                child_instruction = clean_revised_instructions.pop(task_id, "").strip()
                if not child_instruction:
                    malformed_pivot_parent_ids.append(task_id)
                    continue
                clean_spawn_requests.append(
                    TaskSpawnRequest(
                        parent_task_id=task_id,
                        strategy=new_strategy,
                        instruction=child_instruction,
                        reason=(
                            "Auto-converted legacy strategy pivot from "
                            f"{task_queue[task_id].strategy.value} to {new_strategy.value}. "
                            f"{decision.decision_reason}"
                        ),
                    )
                )
                pivot_strategy_by_parent[task_id] = new_strategy

            targeted_pivot_ids = [
                task_id for task_id in valid_target_ids if task_id in pivot_strategy_by_parent
            ]
            incompatible_targeted_pivots = [
                task_id
                for task_id in targeted_pivot_ids
                if decision.next_node
                != _worker_node_for_strategy(pivot_strategy_by_parent[task_id])
            ]
            pivot_validation_failed = bool(
                malformed_pivot_parent_ids or incompatible_targeted_pivots
            )
            if pivot_validation_failed:
                if malformed_pivot_parent_ids:
                    errors.append(
                        "supervisor: rejected strategy pivot without task-specific child "
                        f"instructions for {malformed_pivot_parent_ids}."
                    )
                if incompatible_targeted_pivots:
                    errors.append(
                        "supervisor: rejected strategy pivot because next_node does not match "
                        f"the child strategy for {incompatible_targeted_pivots}."
                    )
                failed_parent_ids = list(
                    {
                        *malformed_pivot_parent_ids,
                        *incompatible_targeted_pivots,
                    }
                )
                _terminalize_pivot_parents(
                    task_queue,
                    failed_parent_ids,
                    pivot_strategy_by_parent | legacy_pivot_strategy_updates,
                    qa_evaluations,
                    retry_diagnostics_by_task=retry_diagnostics_by_task,
                    retry_plans_by_task=retry_plans_by_task,
                    group_by_id=group_by_id,
                )
                decision = _deterministic_routing(
                    task_queue,
                    group_by_id,
                    qa_evaluations,
                    retry_diagnostics_by_task,
                    action_summaries=action_summaries,
                    active_target_task_ids=active_target_task_ids,
                    current_status=str(state.get("status") or ""),
                    portfolio_plan=state.get("portfolio_plan"),
                )
                valid_target_ids = list(decision.target_task_ids)
                valid_unfixable_ids = list(decision.unfixable_task_ids)
                clean_revised_instructions = dict(decision.revised_instructions)
                clean_feedback = dict(decision.feedback_by_task)
                clean_updated_task_strategies = dict(decision.updated_task_strategies)
                clean_status_updates = dict(decision.task_status_updates)
                clean_spawn_requests = list(decision.spawn_requests)
                pivot_strategy_by_parent = {
                    req.parent_task_id: req.strategy
                    for req in clean_spawn_requests
                    if req.parent_task_id in task_queue
                    and task_queue[req.parent_task_id].strategy != req.strategy
                }
                targeted_pivot_ids = [
                    task_id for task_id in valid_target_ids if task_id in pivot_strategy_by_parent
                ]

            for parent_id, new_strategy in pivot_strategy_by_parent.items():
                pivot_parent_status_by_parent[parent_id] = _parent_status_for_strategy_pivot(
                    task_queue[parent_id],
                    new_strategy,
                    qa_evaluations,
                )
            pivot_target_parent_ids = set(targeted_pivot_ids)

            try:
                decision = SupervisorDecision(
                    decision_code=decision.decision_code,
                    next_node=decision.next_node,
                    updated_task_strategies={},
                    target_task_ids=valid_target_ids,
                    unfixable_task_ids=valid_unfixable_ids,
                    new_constraints=decision.new_constraints,
                    feedback_by_task=clean_feedback,
                    revised_instructions=clean_revised_instructions,
                    spawn_requests=clean_spawn_requests,
                    task_status_updates=clean_status_updates,
                    cluster_id=decision.cluster_id,
                    multi_package_action=decision.multi_package_action,
                    instructions=decision.instructions,
                    decision_reason=decision.decision_reason,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("supervisor: decision rebuild failed (%s) â€” falling back.", exc)
                decision = _deterministic_routing(
                    task_queue,
                    group_by_id,
                    qa_evaluations,
                    retry_diagnostics_by_task,
                    action_summaries=action_summaries,
                    active_target_task_ids=active_target_task_ids,
                    current_status=str(state.get("status") or ""),
                    portfolio_plan=state.get("portfolio_plan"),
                )

    # ------------------------------------------------------------------
    # 8. Apply guarded updates to task_queue
    # ------------------------------------------------------------------

    # 8a. Apply revised_instructions (copy-on-write per task)
    for t_id, new_instr in decision.revised_instructions.items():
        if t_id in task_queue and new_instr.strip():
            task = task_queue[t_id]
            if task.current_attempt_id is not None:
                errors.append(
                    f"supervisor: ignored revised instruction for {t_id} because its "
                    "current attempt is still active."
                )
                continue
            _commit_task_transition(
                task_queue,
                t_id,
                updates={"instruction": new_instr},
            )

    # 8b. Apply direct strategy pivots (currently reserved for no-op / legacy cases)
    for t_id, new_strategy in decision.updated_task_strategies.items():
        if t_id in task_queue:
            _commit_task_transition(
                task_queue,
                t_id,
                updates={"strategy": new_strategy},
                close_attempt=task_queue[t_id].current_attempt_id is not None,
            )

    # 8c. Apply guarded task status overrides (only QA_PASSED and UNFIXABLE)
    _allowed_statuses = {TaskStatus.QA_PASSED, TaskStatus.UNFIXABLE}
    for t_id, new_status in decision.task_status_updates.items():
        if (
            t_id in task_queue
            and new_status in _allowed_statuses
            and task_queue[t_id].status not in _TERMINAL_STATUSES
        ):
            _commit_task_transition(
                task_queue,
                t_id,
                updates={"status": new_status},
                close_attempt=(
                    new_status in _TERMINAL_STATUSES
                    and task_queue[t_id].current_attempt_id is not None
                ),
                clear_selected_version=(
                    new_status in _TERMINAL_STATUSES
                    and task_queue[t_id].selected_version is not None
                ),
            )
            logger.info(
                "supervisor: task '%s' manually set to %s via task_status_updates.",
                t_id,
                new_status.value,
            )

    # 8d. Apply unfixable marks from decision
    for t_id in decision.unfixable_task_ids:
        if t_id in task_queue:
            _commit_task_transition(
                task_queue,
                t_id,
                updates={"status": TaskStatus.UNFIXABLE},
                close_attempt=task_queue[t_id].current_attempt_id is not None,
                clear_selected_version=task_queue[t_id].selected_version is not None,
            )

    # 8e. Materialize spawn requests
    child_ids_by_parent: dict[str, list[str]] = {}
    if decision.spawn_requests:
        new_tasks, child_ids_by_parent = _materialize_spawn_requests(
            spawn_requests=list(decision.spawn_requests),
            task_queue=task_queue,
            group_by_id=group_by_id,
            errors=errors,
            valid_groups=valid_groups,
            qa_evaluations=qa_evaluations,
            retry_diagnostics_by_task=retry_diagnostics_by_task,
            consistency_events=consistency_events,
        )
        task_queue.update(new_tasks)
        # Apply the complete parent transition after children are materialized.
        # This must detach the closed update attempt as well as terminalize the
        # parent; otherwise the parent remains paired with an update snapshot
        # while routing has already moved to the workaround child.
        _terminalize_pivot_parents(
            task_queue,
            list(pivot_parent_status_by_parent),
            {
                parent_id: next(
                    (
                        request.strategy
                        for request in decision.spawn_requests
                        if request.parent_task_id == parent_id
                    ),
                    RoutingStrategy.CODE_WORKAROUND,
                )
                for parent_id in pivot_parent_status_by_parent
            },
            qa_evaluations,
            retry_diagnostics_by_task=retry_diagnostics_by_task,
            retry_plans_by_task=retry_plans_by_task,
            group_by_id=group_by_id,
        )

    resolved_target_task_ids: list[str] = []
    remapped_feedback_by_task: dict[str, str] = {}
    for task_id in decision.target_task_ids:
        if task_id in pivot_target_parent_ids:
            child_ids = child_ids_by_parent.get(task_id, [])
            if child_ids:
                child_task_id = child_ids[0]
                child_task = task_queue.get(child_task_id)
                if (
                    child_task is not None
                    and child_task.strategy == RoutingStrategy.CODE_WORKAROUND
                    and child_task.status in _WORKABLE_STATUSES
                ):
                    resolved_target_task_ids.append(child_task_id)
                    if task_id in decision.feedback_by_task:
                        remapped_feedback_by_task[child_task_id] = decision.feedback_by_task[
                            task_id
                        ]
                else:
                    # A replayed pivot may find a child that is already
                    # optimistic/terminal.  It is valid evidence that the
                    # pivot exists, but it is not a worker target.  Let the
                    # deterministic reroute below choose QA, final scanning,
                    # teardown, or another actionable task.
                    logger.info(
                        "supervisor: pivot child '%s' is not dispatchable "
                        "(status=%s); recomputing the next route.",
                        child_task_id,
                        child_task.status.value if child_task is not None else "missing",
                    )
            else:
                errors.append(
                    "supervisor: dropped strategy-pivot target because child task could not "
                    f"be spawned for parent '{task_id}'."
                )
            continue
        resolved_target_task_ids.append(task_id)
        if task_id in decision.feedback_by_task:
            remapped_feedback_by_task[task_id] = decision.feedback_by_task[task_id]

    resolved_next_node = decision.next_node
    all_tasks_terminal = bool(task_queue) and all(
        task.status in _TERMINAL_STATUSES for task in task_queue.values()
    )
    if (
        all_tasks_terminal
        and state.get("workspace_volume")
        and not state.get("final_full_scan_completed", False)
    ):
        decision = decision.model_copy(
            update={
                "decision_code": DecisionCode.FINAL_FULL_SCAN_REQUIRED,
                "next_node": "final_full_scan",
                "target_task_ids": [],
                "instructions": "Run the authoritative full Dependency-Check scan before teardown.",
                "decision_reason": (
                    "Supervisor routing barrier required the final full scan before teardown."
                ),
            }
        )
        resolved_next_node = "final_full_scan"
        resolved_target_task_ids = []
    elif resolved_next_node == "final_full_scan":
        errors.append(
            "supervisor: rejected final_full_scan because the terminal workspace gate is not satisfied."
        )
        decision = _deterministic_routing(
            task_queue,
            group_by_id,
            qa_evaluations,
            retry_diagnostics_by_task,
            action_summaries=action_summaries,
            active_target_task_ids=active_target_task_ids,
            current_status=str(state.get("status") or ""),
            triage_required=bool(state.get("triage_required")),
            workspace_volume=state.get("workspace_volume"),
            final_full_scan_completed=bool(state.get("final_full_scan_completed")),
            portfolio_plan=state.get("portfolio_plan"),
        )
        resolved_next_node = decision.next_node
        resolved_target_task_ids = list(decision.target_task_ids)
    resolved_target_task_ids = _normalize_target_task_ids_for_node(
        resolved_next_node,
        resolved_target_task_ids,
        task_queue,
        retry_diagnostics_by_task,
        group_by_id,
        allow_cluster=bool(decision.cluster_id),
    )
    # Status overrides and parent terminalization above are also untrusted
    # router requests. Re-clamp after those mutations so a task that became
    # terminal in this transition cannot remain in the dispatch projection.
    resolved_target_task_ids = _normalize_target_task_ids_for_node(
        resolved_next_node,
        resolved_target_task_ids,
        task_queue,
        retry_diagnostics_by_task,
        group_by_id,
        allow_cluster=bool(decision.cluster_id),
    )
    remapped_feedback_by_task = {
        task_id: feedback
        for task_id, feedback in remapped_feedback_by_task.items()
        if task_id in set(resolved_target_task_ids)
    }

    # Programmatic override: forcefully inject the latest QA feedback into
    # workaround-subagent retries so the worker receives the evidence.
    if resolved_next_node == "workaround_subagent":
        for task_id in resolved_target_task_ids:
            task = task_queue.get(task_id)
            if task:
                eval_ = qa_evaluations.get(task_id)
                if eval_ and eval_.retry_feedback:
                    remapped_feedback_by_task[task_id] = eval_.retry_feedback
    if (
        resolved_next_node in {"update_subagent", "workaround_subagent", "qa_critic"}
        and not resolved_target_task_ids
    ):
        errors.append(
            "supervisor: recomputing routing because no dispatchable target tasks remained."
        )
        # This can happen when a replayed pivot reuses a child that has
        # already reached a terminal state.  Teardown would incorrectly skip
        # other non-terminal tasks, so ask the deterministic router for the
        # next eligible action after the parent/child transition is committed.
        decision = _deterministic_routing(
            task_queue,
            group_by_id,
            qa_evaluations,
            retry_diagnostics_by_task,
            action_summaries=action_summaries,
            active_target_task_ids=active_target_task_ids,
            current_status=str(state.get("status") or ""),
            triage_required=bool(state.get("triage_required")),
            workspace_volume=state.get("workspace_volume"),
            final_full_scan_completed=bool(state.get("final_full_scan_completed")),
            portfolio_plan=state.get("portfolio_plan"),
        )
        resolved_next_node = decision.next_node
        resolved_target_task_ids = _normalize_target_task_ids_for_node(
            resolved_next_node,
            list(decision.target_task_ids),
            task_queue,
            retry_diagnostics_by_task,
            group_by_id,
            allow_cluster=bool(decision.cluster_id),
        )
        remapped_feedback_by_task = {
            task_id: feedback
            for task_id, feedback in decision.feedback_by_task.items()
            if task_id in set(resolved_target_task_ids)
        }

    if resolved_next_node in _WORKER_NODES and not resolved_target_task_ids:
        stalled_task_ids = [
            task_id for task_id, task in task_queue.items() if task.status not in _TERMINAL_STATUSES
        ]
        for task_id in stalled_task_ids:
            task = task_queue[task_id]
            _commit_task_transition(
                task_queue,
                task_id,
                updates={"status": TaskStatus.UNFIXABLE},
                close_attempt=task.current_attempt_id is not None,
                clear_selected_version=task.selected_version is not None,
            )
        errors.append(
            "supervisor: fail-closed an empty worker route; no dispatchable "
            f"target remained. Marked unresolved tasks unfixable: {stalled_task_ids}."
        )

        terminal_workspace = (
            bool(task_queue)
            and bool(state.get("workspace_volume"))
            and all(task.status in _TERMINAL_STATUSES for task in task_queue.values())
            and not state.get("final_full_scan_completed", False)
        )
        if terminal_workspace:
            resolved_next_node = "final_full_scan"
            decision = SupervisorDecision(
                decision_code=DecisionCode.FINAL_FULL_SCAN_REQUIRED,
                next_node=resolved_next_node,
                target_task_ids=[],
                instructions="Run the authoritative full Dependency-Check scan before teardown.",
                decision_reason=(
                    "All tasks were terminalized after the Supervisor found no "
                    "dispatchable worker target."
                ),
            )
        else:
            resolved_next_node = "teardown"
            decision = SupervisorDecision(
                decision_code=DecisionCode.NO_ACTIONABLE_TASKS,
                next_node=resolved_next_node,
                target_task_ids=[],
                instructions="No dispatchable tasks remain; proceed to teardown.",
                decision_reason=(
                    "The Supervisor fail-closed after a worker route had no dispatchable target."
                ),
            )
        resolved_target_task_ids = []
        remapped_feedback_by_task = {}

    # Commit the exact input snapshot before exposing worker targets to the
    # graph. QA reuses the current worker attempt; update/workaround dispatches
    # always receive a new attempt identity.
    active_cluster_id: str | None = None
    active_dispatch_batch_id: str | None = None
    active_multi_package_action: MultiPackageAction | None = None
    portfolio_dispatch_rejected = False
    if resolved_next_node == "update_subagent" and decision.cluster_id:
        action = _build_multi_package_action(
            decision.cluster_id,
            resolved_target_task_ids,
            task_queue,
            group_by_id,
        )
        if action is None or len(resolved_target_task_ids) < 2:
            reason = (
                f"supervisor: rejected atomic cluster '{decision.cluster_id}' because "
                "its committed action could not be represented exactly; requesting "
                "a portfolio replan."
            )
            errors.append(reason)
            portfolio_dispatch_rejected = True
            portfolio_replan_request = PortfolioReplanRequest(
                reason=reason,
                source_portfolio_plan_id=(
                    committed_plan.portfolio_plan_id if committed_plan is not None else None
                ),
            )
            decision = SupervisorDecision(
                decision_code=DecisionCode.PORTFOLIO_PLAN_REQUIRED,
                next_node="portfolio",
                target_task_ids=[],
                instructions="Rebuild the portfolio plan before dispatching the package cluster.",
                decision_reason=reason,
            )
            resolved_next_node = "portfolio"
            resolved_target_task_ids = []
            remapped_feedback_by_task = {}
        else:
            active_cluster_id = decision.cluster_id
            active_dispatch_batch_id = _cluster_dispatch_batch_id(
                active_cluster_id,
                resolved_target_task_ids,
                state_revision,
            )
            action = action.model_copy(update={"dispatch_batch_id": active_dispatch_batch_id})
            active_multi_package_action = action
            decision = decision.model_copy(update={"multi_package_action": action})
    elif resolved_next_node == "qa_critic" and decision.cluster_id:
        active_cluster_id = decision.cluster_id
        active_dispatch_batch_id = state.get("active_dispatch_batch_id")
        active_multi_package_action = state.get("active_multi_package_action")

    if resolved_next_node in {"update_subagent", "workaround_subagent", "qa_critic"}:
        for task_id in list(resolved_target_task_ids):
            task = task_queue.get(task_id)
            if task is None:
                continue
            # QA can only consume the worker attempt that produced the current
            # workspace. Never synthesize an attempt for a task without one.
            if resolved_next_node == "qa_critic":
                if not task.current_attempt_id:
                    errors.append(
                        f"supervisor: rejected QA target '{task_id}' because no "
                        "committed worker attempt exists."
                    )
                continue
            plan = retry_plans_by_task.get(task_id)
            workaround_ctx = None
            if resolved_next_node == "workaround_subagent":
                attempts = _attempts_for_task(attempt_snapshots_by_id, task_id)
                parent_task_ids = _workaround_task_ancestry(task, task_queue)
                evidence = _qa_failure_evidence_for_workaround_retry(
                    task_id,
                    qa_evaluations,
                    qa_results_by_attempt,
                    related_task_ids=parent_task_ids,
                )
                phase = (
                    WorkaroundPhase.QA_REGRESSION_REPAIR
                    if attempts or _qa_evidence_indicates_test_regression(evidence)
                    else WorkaroundPhase.INITIAL_MITIGATION
                )
                group = next(
                    (
                        candidate
                        for candidate in state.get("valid_groups", []) or []
                        if candidate.group_id == task.parent_group_id
                    ),
                    None,
                )
                vulnerability_mechanism = (
                    _extract_workaround_vulnerability_mechanism(group) if group is not None else ""
                )
                workaround_ctx = WorkaroundContext(
                    phase=phase,
                    vulnerability_mechanism=vulnerability_mechanism,
                    qa_evidence=evidence,
                    no_fix_stage=task.no_fix_stage,
                    reset_prior_stage_workspace=(
                        task.no_fix_stage == NoFixMitigationStage.VULNERABLE_CODE_REMOVAL
                        and task.parent_group_id in group_by_id
                    ),
                )
            task, snapshot = _create_attempt_snapshot(
                task,
                dispatch_node=resolved_next_node,
                snapshots_by_id=attempt_snapshots_by_id,
                state_revision=state_revision,
                plan_id=plan.plan_id if plan is not None else None,
                portfolio_plan_id=(
                    committed_plan.portfolio_plan_id if committed_plan is not None else None
                ),
                workaround_context=workaround_ctx,
                allowed_target_versions=(
                    _ordered_update_candidates(
                        task,
                        plan=plan,
                        diagnostics=retry_diagnostics_by_task.get(task_id),
                    )[0]
                    if resolved_next_node == "update_subagent"
                    else []
                ),
                allowed_dependency_types=(
                    _ordered_update_candidates(
                        task,
                        plan=plan,
                        diagnostics=retry_diagnostics_by_task.get(task_id),
                    )[1]
                    if resolved_next_node == "update_subagent"
                    else []
                ),
                cluster_id=active_cluster_id,
                dispatch_batch_id=active_dispatch_batch_id,
                action_digest=(
                    instruction_digest(active_multi_package_action.model_dump_json())
                    if active_multi_package_action is not None
                    else None
                ),
                manifest_path=_group_manifest_path(group_by_id[task.parent_group_id])
                if task.parent_group_id in group_by_id
                else None,
            )
            task_queue[task_id] = task
            prior = retry_diagnostics_by_task.get(task_id)
            if prior is not None:
                retry_diagnostics_by_task[task_id] = prior.model_copy(
                    update={
                        "committed_attempt_id": snapshot.attempt_id,
                        "selected_version": task.selected_version,
                        "strategy_stage": task.strategy_stage,
                        "exhausted_update_path": task.exhausted_update_path,
                        "instruction_digest": snapshot.instruction_digest,
                    }
                )
            if plan is not None:
                # The plan was created against the pre-dispatch task revision.
                # Once its exact input is committed, keep the compatibility
                # plan projection correlated to that same revision.
                retry_plans_by_task[task_id] = plan.model_copy(
                    update={"source_task_revision": task.task_revision}
                )

    logger.info(
        "supervisor: routing to '%s' with targets=%s",
        resolved_next_node,
        resolved_target_task_ids,
    )

    # ------------------------------------------------------------------
    # 8f. Collect new constraints
    # ------------------------------------------------------------------
    returned_constraints: list[str] = list(auto_new_constraints)
    for constraint in decision.new_constraints:
        if (
            constraint
            and constraint not in existing_constraints
            and constraint not in returned_constraints
        ):
            returned_constraints.append(constraint)

    # Build feedback_by_group for bridge node backward compat
    feedback_by_task = remapped_feedback_by_task
    feedback_by_group: dict[str, str] = {}
    for t_id, fb in feedback_by_task.items():
        if t_id in task_queue:
            gid = task_queue[t_id].parent_group_id
            feedback_by_group[gid] = fb

    consistency_new_events, consistency_errors = _validate_committed_state(
        task_queue,
        attempt_snapshots_by_id,
        retry_plans_by_task,
        retry_diagnostics_by_task,
        resolved_target_task_ids,
        resolved_next_node,
    )
    consistency_events.extend(consistency_new_events)
    errors.extend(consistency_errors)
    existing_event_keys = {
        (event.task_id, event.received_attempt_id, event.error_code)
        for event in prior_consistency_events
    }
    consistency_events = [
        event
        for event in _dedupe_consistency_events(consistency_events)
        if (event.task_id, event.received_attempt_id, event.error_code) not in existing_event_keys
    ]
    # ``errors`` uses an additive reducer, so suppress both duplicate messages
    # from this invocation and exact messages already committed by an earlier
    # supervisor pass. Structured consistency events remain the detailed,
    # attempt-correlated audit record.
    errors = list(dict.fromkeys(error for error in errors if error not in prior_error_messages))

    portfolio_dirty = bool(state.get("portfolio_dirty", False)) or portfolio_dispatch_rejected
    if state.get("portfolio_plan") is not None:
        portfolio_dirty = portfolio_dirty or _portfolio_plan_is_stale(
            state.get("portfolio_plan"),
            task_queue,
            valid_groups,
            state.get("repo_root"),
        )
    if portfolio_dirty and not isinstance(portfolio_replan_request, PortfolioReplanRequest):
        stale_reasons = _portfolio_plan_violations(
            committed_plan,
            task_queue,
            valid_groups,
        )
        portfolio_replan_request = PortfolioReplanRequest(
            reason=(
                "; ".join(stale_reasons)
                if stale_reasons
                else "Committed portfolio plan was marked dirty by Supervisor reconciliation."
            )[:2000],
            triggering_attempt_id=next(
                (
                    task.current_attempt_id
                    for task in task_queue.values()
                    if task.current_attempt_id
                ),
                None,
            ),
            source_portfolio_plan_id=(
                committed_plan.portfolio_plan_id if committed_plan is not None else None
            ),
        )

    # ------------------------------------------------------------------
    # 9. Return state patch
    # ------------------------------------------------------------------
    return {
        "status": "supervisor_routed",
        "next_routing_step": resolved_next_node,
        "decision_code": decision.decision_code,
        "supervisor_audit": _emit_audit(decision, consistency_events, state_revision),
        "active_target_task_ids": resolved_target_task_ids,
        "active_cluster_id": active_cluster_id,
        "active_dispatch_batch_id": active_dispatch_batch_id,
        "active_multi_package_action": active_multi_package_action,
        "portfolio_dirty": portfolio_dirty,
        "portfolio_replan_request": portfolio_replan_request,
        "portfolio_escalation": portfolio_escalation,
        "feedback_by_group": feedback_by_group,
        "supervisor_instructions": decision.instructions,
        # Compatibility projection: the attempt-tagged QA envelope remains
        # authoritative, while this task-keyed view is retained for existing
        # callers and prompt builders.
        "qa_evaluations": qa_evaluations,
        "task_queue": task_queue,
        "valid_groups": valid_groups,
        "retry_diagnostics_by_task": retry_diagnostics_by_task,
        "retry_plans_by_task": retry_plans_by_task,
        "workaround_replay_plans_by_task": workaround_replay_plans_by_task,
        "attempt_snapshots_by_id": attempt_snapshots_by_id,
        "worker_results_by_attempt": worker_results_by_attempt,
        "qa_results_by_attempt": qa_results_by_attempt,
        "processed_worker_attempt_ids": list(new_worker_attempt_ids),
        "processed_qa_attempt_ids": list(new_qa_attempt_ids),
        "consistency_events": consistency_events,
        "state_revision": state_revision,
        # constraints_ledger uses operator.add â€” return only NEW entries
        "constraints_ledger": returned_constraints,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# Conditional-edge router
# ---------------------------------------------------------------------------


def supervisor_router(state: OrchestratorState) -> str:
    """Return the committed route, recomputing it when state is invalid."""
    if state.get("post_qa_retriage_limit_reached"):
        logger.warning("supervisor_router: post-QA re-triage limit reached; routing to teardown.")
        return "teardown"
    step = state.get("next_routing_step", "")
    task_queue = dict(state.get("task_queue", {}) or {})
    if (
        step == "teardown"
        and state.get("workspace_volume")
        and task_queue
        and not state.get("final_full_scan_completed", False)
        and all(task.status in _TERMINAL_STATUSES for task in task_queue.values())
    ):
        logger.warning("supervisor_router: enforcing final_full_scan before teardown.")
        return "final_full_scan"
    if step == "final_full_scan":
        terminal_workspace = (
            bool(task_queue)
            and bool(state.get("workspace_volume"))
            and all(task.status in _TERMINAL_STATUSES for task in task_queue.values())
            and not state.get("final_full_scan_completed", False)
        )
        if not terminal_workspace:
            logger.warning("supervisor_router: rejecting premature final_full_scan route.")
            step = ""
    if step in _WORKER_NODES and task_queue and not state.get("active_target_task_ids"):
        logger.error(
            "supervisor_router: refusing worker route '%s' without active targets; "
            "failing closed to teardown.",
            step,
        )
        terminal_workspace = (
            bool(task_queue)
            and bool(state.get("workspace_volume"))
            and all(task.status in _TERMINAL_STATUSES for task in task_queue.values())
            and not state.get("final_full_scan_completed", False)
        )
        return "final_full_scan" if terminal_workspace else "teardown"
    if step in _VALID_NEXT_NODES:
        return step
    logger.error("supervisor_router: invalid next_routing_step '%s' - recomputing.", step)

    valid_groups = list(state.get("valid_groups", []) or [])
    group_by_id = {group.group_id: group for group in valid_groups}
    decision = _deterministic_routing(
        dict(state.get("task_queue", {}) or {}),
        group_by_id,
        dict(state.get("qa_evaluations", {}) or {}),
        dict(state.get("retry_diagnostics_by_task", {}) or {}),
        active_target_task_ids=list(state.get("active_target_task_ids", []) or []),
        current_status=str(state.get("status") or ""),
        triage_required=bool(state.get("triage_required")),
        workspace_volume=state.get("workspace_volume"),
        final_full_scan_completed=bool(state.get("final_full_scan_completed")),
        portfolio_plan=state.get("portfolio_plan"),
    )
    if decision.next_node not in _VALID_NEXT_NODES:
        logger.critical(
            "supervisor_router: deterministic recomputation produced invalid node '%s'.",
            decision.next_node,
        )
        return "teardown"
    return decision.next_node
