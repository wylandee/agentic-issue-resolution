"""
supervisor_node.py - Agentic Supervisor Node for Phase 5 hub-and-spoke orchestration.

Phase 5 architecture: guarded tactical Supervisor
-----------------------------------------------
The Supervisor owns the state machine and produces every committed routing and
retry decision in Python. When an API key is configured, a structured tactical
reasoner may propose a strategy and instruction; Python verifies and commits it
or falls back to the deterministic retry planner. Registry facts are never
selected by a worker.

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

import logging
import re
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from remediation_engine.contracts.decision_codes import (
    DecisionCode,
    validate_transition,
)
from remediation_engine.contracts.schemas import (
    AgentActionStatus,
    AgentActionSummary,
    NoFixMitigationStage,
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
from remediation_engine.orchestration import _supervisor_execution as _supervisor_execution_helpers
from remediation_engine.orchestration.runtime_context import get_runtime_settings
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
    _supervisor_dependency_type_candidates,
    instruction_digest,
)
from remediation_engine.orchestration.supervisor_policy import (
    _TERMINAL_STATUSES,
    _WORKABLE_STATUSES,
    MAX_RETRIES,
    _canonical_security_floor,
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
from remediation_engine.orchestration.tactical_supervisor import (
    TacticalDiagnosticKind,
    _approved_candidate_pool,
    build_tactical_context,
    classify_diagnostics,
    propose_and_verify_tactical_action,
    registry_candidate_sets_for_context,
)
from remediation_engine.orchestration.task_utils import (
    advance_no_fix_stage,
    build_initial_remediation_task,
    build_no_fix_package_removal_instruction,
    build_no_fix_retry_instruction,
    derive_missing_task_qa_policy,
    group_parent_context,
    is_no_fix_group,
    is_transitive_group,
)

logger = logging.getLogger(__name__)

# Supervisor dispatch commits one task attempt at a time. Worker and QA
# boundaries consume only the typed task/attempt envelopes.

_VALID_NEXT_NODES: set[str] = {
    "update_subagent",
    "workaround_subagent",
    "qa_critic",
    "triage",
    "final_full_scan",
    "teardown",
}
_WORKER_NODES = frozenset({"update_subagent", "workaround_subagent", "qa_critic"})


@dataclass(frozen=True)
class _StagedTacticalResolution:
    """Uncommitted tactical task data held until routing guardrails pass."""

    task_id: str
    expected_task_revision: int
    decision_code: DecisionCode
    next_node: str
    instruction: str
    updates: dict[str, Any]
    retry_diagnostics: UpdateRetryDiagnostics
    retry_plan: SupervisorRetryPlan


__all__ = [
    "MAX_RETRIES",
    "UPDATE_DISPATCH_LIMIT",
    "QA_DISPATCH_LIMIT",
    "_SCA_STAGE_ORDER",
    "_OVERRIDE_DEPENDENCY_TYPES",
    "_task_sort_key",
    "_next_sca_stage",
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
    "_authorize_update_dispatch",
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
    """Build immutable version and dependency-type candidates for an update attempt."""
    selected_version = (
        plan.selected_version
        if plan is not None and plan.selected_version
        else task.selected_version
    )
    attempted_version_values = [
        *(diagnostics.attempted_versions if diagnostics else []),
        *(plan.attempted_versions if plan else []),
    ]
    attempted_versions = {item.strip().lstrip("vV") for item in attempted_version_values if item}
    if plan is not None:
        candidate_versions = list(plan.candidate_versions_considered)
    elif diagnostics is not None:
        candidate_versions = list(diagnostics.candidate_versions_considered)
    else:
        candidate_versions = []

    # A selected version is only dispatchable when it belongs to the
    # Supervisor-approved candidate pool.  Never let a stale task field widen
    # that pool after a planner or registry result has been committed.
    normalized_candidates = [
        version.strip().lstrip("vV")
        for version in candidate_versions
        if isinstance(version, str) and version.strip()
    ]
    candidate_pool = set(normalized_candidates)
    selected_candidate = (
        selected_version.strip().lstrip("vV")
        if isinstance(selected_version, str) and selected_version.strip()
        else None
    )
    ordered_versions = (
        [selected_candidate, *normalized_candidates]
        if selected_candidate in candidate_pool
        else normalized_candidates
    )

    allowed_versions: list[str] = []
    seen_versions: set[str] = set()
    for version in ordered_versions:
        if not version:
            continue
        normalized = version.strip().lstrip("vV")
        if not normalized or normalized in seen_versions:
            continue
        if normalized in attempted_versions:
            continue
        seen_versions.add(normalized)
        allowed_versions.append(normalized)
    selected_type = (
        (plan.target_dependency_type if plan is not None else None)
        or task.target_dependency_type
        or (diagnostics.target_dependency_type if diagnostics else None)
    )
    strategy_stage = plan.strategy_stage if plan is not None else task.strategy_stage
    policy_types = _supervisor_dependency_type_candidates(strategy_stage, selected_type)
    if plan is not None and plan.candidate_dependency_types:
        committed_plan_types = set(plan.candidate_dependency_types)
        candidate_types = [
            dependency_type
            for dependency_type in policy_types
            if dependency_type in committed_plan_types
        ]
    else:
        candidate_types = policy_types
    attempted_types = {
        item.strip()
        for item in (diagnostics.attempted_dependency_types if diagnostics else ())
        if item
    }
    allowed_types: list[str] = []
    seen_types: set[str] = set()
    for dependency_type in candidate_types:
        if not dependency_type:
            continue
        normalized = str(dependency_type).strip()
        if normalized in attempted_types:
            continue
        if normalized and normalized not in seen_types:
            seen_types.add(normalized)
            allowed_types.append(normalized)
    return allowed_versions, allowed_types


@dataclass(frozen=True)
class _UpdateDispatchAuthorization:
    """Immutable candidate authorization required for one update dispatch."""

    selected_version: str
    allowed_target_versions: tuple[str, ...]
    allowed_dependency_types: tuple[str, ...]


def _authorize_update_dispatch(
    task: RemediationTask,
    *,
    plan: SupervisorRetryPlan | None = None,
    diagnostics: UpdateRetryDiagnostics | None = None,
) -> _UpdateDispatchAuthorization | None:
    """Return a dispatch authorization or reject incomplete update state.

    Update workers may only receive a candidate pool that came from the
    committed retry plan or retry diagnostics.  A selected task version by
    itself is not sufficient registry provenance.
    """
    allowed_versions, allowed_dependency_types = _ordered_update_candidates(
        task,
        plan=plan,
        diagnostics=diagnostics,
    )
    if not task.instruction.strip():
        return None
    if not allowed_versions:
        return None
    expected_package = (
        plan.target_package_name
        if plan is not None and plan.target_package_name
        else diagnostics.target_package_name
        if diagnostics is not None
        else None
    )
    expected_type = (
        plan.target_dependency_type
        if plan is not None and plan.target_dependency_type
        else diagnostics.target_dependency_type
        if diagnostics is not None
        else None
    )
    if expected_package and task.target_package_name != expected_package:
        return None
    if expected_type and task.target_dependency_type != expected_type:
        return None
    plan_version = (
        plan.selected_version.strip().lstrip("vV")
        if plan is not None and plan.selected_version
        else None
    )
    task_version = task.selected_version.strip().lstrip("vV") if task.selected_version else None
    if plan_version and task_version != plan_version:
        return None
    selected_version = plan_version or task_version
    if selected_version is None or selected_version not in set(allowed_versions):
        return None
    return _UpdateDispatchAuthorization(
        selected_version=selected_version,
        allowed_target_versions=tuple(allowed_versions),
        allowed_dependency_types=tuple(allowed_dependency_types),
    )


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

    fix_plan = getattr(group, "fix_plan", None)
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
    workaround_context: WorkaroundContext | None = None,
    allowed_target_versions: Iterable[str] = (),
    allowed_dependency_types: Iterable[str] = (),
) -> tuple[RemediationTask, TaskAttemptSnapshot]:
    normalized_allowed_target_versions = list(
        dict.fromkeys(
            str(value).strip().lstrip("vV")
            for value in allowed_target_versions
            if str(value).strip()
        )
    )
    if dispatch_node == "update_subagent":
        if not normalized_allowed_target_versions:
            raise ValueError(
                "update_subagent snapshots require a non-empty candidate authorization."
            )
        if not task.selected_version:
            raise ValueError("update_subagent snapshots require a committed selected_version.")
        if task.selected_version.strip().lstrip("vV") not in normalized_allowed_target_versions:
            raise ValueError(
                "update_subagent snapshot selected_version must belong to its candidate authorization."
            )
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
        state_revision=state_revision,
        task_revision=task_revision,
        attempt_number=len(_attempts_for_task(snapshots_by_id, task.task_id)) + 1,
        qa_policy=task.qa_policy,
        strategy_stage=task.strategy_stage,
        no_fix_stage=task.no_fix_stage,
        selected_version=task.selected_version,
        allowed_target_versions=normalized_allowed_target_versions,
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


def _commit_registry_resolution_fallback(
    task_queue: dict[str, RemediationTask],
    task: RemediationTask,
    group: VulnerabilityGroup,
    evaluation: QAEvaluation | None,
    retry_diagnostics_by_task: dict[str, UpdateRetryDiagnostics],
    retry_plans_by_task: dict[str, SupervisorRetryPlan],
    candidate_sets: Iterable[Any],
    *,
    consistency_events: list[StateConsistencyEvent],
    errors: list[str],
    staged_resolutions: list[_StagedTacticalResolution] | None = None,
) -> SupervisorDecision | None:
    """Commit a verified deterministic update when tactical reasoning is absent.

    Registry facts are still Supervisor-owned even when the optional model is
    disabled or fails to produce a valid action.  The helper consumes only the
    candidate set already fetched for the active task and emits no worker
    request until the selected version is verified.
    """
    candidate_sets = tuple(candidate_sets)
    fallback_strategy = (
        TacticalStrategy.PACKAGE_OVERRIDE
        if task.strategy_stage == SCARemediationStage.PACKAGE_OVERRIDE
        else TacticalStrategy.VERSION_BUMP
    )
    version_set = next(
        (candidate for candidate in candidate_sets if candidate.strategy == fallback_strategy),
        None,
    )
    selected = task.selected_version.strip().lstrip("vV") if task.selected_version else None
    effective_stage = task.strategy_stage
    if version_set is None or not version_set.versions:
        # The inventory may prove that the transitive parent has no usable
        # candidate while the vulnerable child still has an authorized native
        # override.  Select that action directly rather than pretending the
        # task is still a parent VERSION_BUMP.  This is the deterministic
        # no-model fallback for the same action-inventory contract supplied to
        # the tactical Supervisor.
        override_set = next(
            (
                candidate
                for candidate in candidate_sets
                if candidate.strategy == TacticalStrategy.PACKAGE_OVERRIDE and candidate.versions
            ),
            None,
        )
        if (
            task.strategy_stage != SCARemediationStage.PACKAGE_OVERRIDE
            and task.strategy == RoutingStrategy.VERSION_BUMP
            and is_transitive_group(group)
            and override_set is not None
        ):
            fallback_strategy = TacticalStrategy.PACKAGE_OVERRIDE
            version_set = override_set
            effective_stage = SCARemediationStage.PACKAGE_OVERRIDE
        else:
            # An empty verified set is normal exhaustion, not a registry
            # outage. Return control to the deterministic planner so it can
            # choose another available action or pivot to a workaround. A
            # registry exception is handled earlier by ``_apply_tactical_supervisor``
            # and remains an INCONCLUSIVE, no-dispatch outcome.
            return None

    if selected not in set(version_set.versions):
        selected = version_set.canonical_version
    if not selected:
        return None
    target_type = version_set.dependency_type or task.target_dependency_type
    prior_diagnostics = retry_diagnostics_by_task.get(task.task_id)
    diagnostics_for_commit = prior_diagnostics or UpdateRetryDiagnostics(task_id=task.task_id)
    prior_pool = (
        list(diagnostics_for_commit.candidate_versions_considered)
        if (
            diagnostics_for_commit.candidate_versions_considered
            and (
                not diagnostics_for_commit.target_package_name
                or diagnostics_for_commit.target_package_name == version_set.target_package_name
            )
        )
        else []
    )
    approved_versions = list(dict.fromkeys(prior_pool or list(version_set.versions)))
    diagnostics_for_commit = diagnostics_for_commit.model_copy(
        update={
            "strategy_stage": effective_stage,
            "security_floor": version_set.security_floor,
            "selected_version": selected,
            "target_package_name": version_set.target_package_name,
            "target_dependency_type": target_type,
            "candidate_versions_considered": approved_versions,
            "candidate_dependency_types": _supervisor_dependency_type_candidates(
                effective_stage,
                target_type,
            ),
            "latest_version_seen": version_set.versions[-1],
            "registry_query_performed": True,
            "reasoning_summary": (
                "policy=deterministic_verified_candidate; "
                f"recommended_candidate={version_set.canonical_version or 'none'}; "
                f"action={fallback_strategy.value}"
            ),
        }
    )
    resolved_task = task.model_copy(
        update={
            "strategy_stage": effective_stage,
            "selected_version": selected,
            "target_package_name": version_set.target_package_name,
            "target_dependency_type": target_type,
        }
    )
    resolved_task = resolved_task.model_copy(
        update={
            "instruction": _build_high_level_retry_instruction(
                resolved_task,
                group,
                evaluation,
                diagnostics_for_commit,
            )
        }
    )
    updates = {
        "strategy_stage": effective_stage,
        "selected_version": resolved_task.selected_version,
        "target_package_name": resolved_task.target_package_name,
        "target_dependency_type": target_type,
        "instruction": resolved_task.instruction,
    }
    if staged_resolutions is not None:
        committed = _project_tactical_task_transition(task, updates)
    else:
        committed = _commit_task_transition(
            task_queue,
            task.task_id,
            updates=updates,
            consistency_events=consistency_events,
        )
    if committed is None:
        return None
    retry_plan = SupervisorRetryPlan(
        task_id=task.task_id,
        source_task_revision=committed.task_revision,
        strategy_stage=committed.strategy_stage,
        selected_version=selected,
        target_package_name=committed.target_package_name,
        target_dependency_type=committed.target_dependency_type,
        parent_minimum_version=committed.parent_minimum_version,
        candidate_versions_considered=approved_versions,
        candidate_dependency_types=list(diagnostics_for_commit.candidate_dependency_types),
        action="retry_update",
        exact_instruction=committed.instruction,
    )
    if staged_resolutions is not None:
        staged_resolutions.append(
            _StagedTacticalResolution(
                task_id=task.task_id,
                expected_task_revision=task.task_revision,
                decision_code=(
                    DecisionCode.RETRY_VERSION_BUMP
                    if task.status == TaskStatus.NEEDS_RETRY or task.retry_count > 0
                    else DecisionCode.NEW_VERSION_BUMP
                ),
                next_node="update_subagent",
                instruction=committed.instruction,
                updates=updates,
                retry_diagnostics=diagnostics_for_commit,
                retry_plan=retry_plan,
            )
        )
    else:
        retry_diagnostics_by_task[task.task_id] = diagnostics_for_commit
        retry_plans_by_task[task.task_id] = retry_plan
    return SupervisorDecision(
        decision_code=(
            DecisionCode.RETRY_VERSION_BUMP
            if task.status == TaskStatus.NEEDS_RETRY or task.retry_count > 0
            else DecisionCode.NEW_VERSION_BUMP
        ),
        next_node="update_subagent",
        target_task_ids=[task.task_id],
        revised_instructions={task.task_id: committed.instruction},
        instructions=committed.instruction,
        decision_reason=(
            f"Deterministic Supervisor selected the lowest verified candidate "
            f"{selected} for task '{task.task_id}'."
        ),
    )


def _apply_tactical_supervisor(
    task_queue: dict[str, RemediationTask],
    group_by_id: dict[str, VulnerabilityGroup],
    qa_evaluations: dict[str, QAEvaluation],
    retry_diagnostics_by_task: dict[str, UpdateRetryDiagnostics],
    retry_plans_by_task: dict[str, SupervisorRetryPlan],
    worker_results_by_attempt: dict[str, WorkerAttemptResult],
    attempt_snapshots_by_id: dict[str, TaskAttemptSnapshot] | None = None,
    *,
    target_task_id: str | None = None,
    consistency_events: list[StateConsistencyEvent],
    errors: list[str],
    staged_resolutions: list[_StagedTacticalResolution] | None = None,
) -> SupervisorDecision | None:
    """Apply one optional tactical proposal before deterministic routing.

    The tactical layer is deliberately limited to one task per Supervisor
    turn.  It may alter only the same task-input fields that the deterministic
    planner owns, or create the existing single workaround child pivot.  A
    disabled/unavailable/rejected proposal returns ``None`` so the existing
    deterministic router remains authoritative.
    """
    settings = get_runtime_settings()

    candidates = sorted(
        (
            task
            for task in task_queue.values()
            if (target_task_id is None or task.task_id == target_task_id)
            if task.current_attempt_id is None
            and task.status in _WORKABLE_STATUSES
            and task.no_fix_stage is None
        ),
        key=lambda task: _task_sort_key(task, group_by_id),
    )
    for task in candidates:
        group = group_by_id.get(task.parent_group_id)
        if group is None:
            continue
        evaluation = qa_evaluations.get(task.task_id)
        worker_result = next(
            (
                result
                for result in sorted(
                    worker_results_by_attempt.values(),
                    key=lambda item: item.attempt_id,
                    reverse=True,
                )
                if result.task_id == task.task_id
            ),
            None,
        )
        retry_diagnostics = retry_diagnostics_by_task.get(task.task_id)
        prior_attempts = tuple(
            snapshot
            for snapshot in (attempt_snapshots_by_id or {}).values()
            if snapshot.task_id == task.task_id
        )
        base_context = build_tactical_context(
            task,
            group,
            evaluation=evaluation,
            worker_result=worker_result,
            retry_diagnostics=retry_diagnostics,
            candidate_dependency_types=(
                task.target_dependency_type,
                *(
                    _supervisor_dependency_type_candidates(
                        task.strategy_stage,
                        task.target_dependency_type,
                    )
                    if task.strategy == RoutingStrategy.VERSION_BUMP
                    else []
                ),
            ),
            prior_attempts=prior_attempts,
        )
        if classify_diagnostics(base_context).value == "inconclusive":
            # QA contract/infrastructure/attribution gaps are rerun evidence,
            # not a remediation signal. Do not spend a model call or retry
            # budget on tactical replanning.
            return None
        candidate_sets, registry_error = registry_candidate_sets_for_context(base_context)
        if registry_error:
            detail = f"Registry verification pending for task {task.task_id}: {registry_error}"
            consistency_events.append(
                _build_consistency_event(
                    error_code="REGISTRY_VERIFICATION_PENDING",
                    task_id=task.task_id,
                    expected_attempt_id=task.current_attempt_id,
                    received_attempt_id=None,
                    action="rejected",
                    details=detail,
                )
            )
            errors.append(f"supervisor: {detail}")
            return SupervisorDecision(
                decision_code=DecisionCode.REGISTRY_VERIFICATION_PENDING,
                next_node="final_full_scan",
                target_task_ids=[],
                task_status_updates={task.task_id: TaskStatus.INCONCLUSIVE},
                instructions=(
                    "Registry verification is unavailable; defer remediation and preserve retry budget."
                ),
                decision_reason=detail,
            )
        if (
            not settings.openai_api_key
            and task.strategy == RoutingStrategy.VERSION_BUMP
            and task.status != TaskStatus.NEEDS_RETRY
            and task.retry_count == 0
        ):
            return _commit_registry_resolution_fallback(
                task_queue,
                task,
                group,
                evaluation,
                retry_diagnostics_by_task,
                retry_plans_by_task,
                candidate_sets,
                consistency_events=consistency_events,
                errors=errors,
                staged_resolutions=staged_resolutions,
            )
        candidate_versions = next(
            (
                candidate.versions
                for candidate in candidate_sets
                if candidate.strategy == TacticalStrategy.VERSION_BUMP
            ),
            (),
        )
        context = build_tactical_context(
            task,
            group,
            evaluation=evaluation,
            worker_result=worker_result,
            retry_diagnostics=retry_diagnostics,
            candidate_versions=candidate_versions,
            candidate_dependency_types=base_context.candidate_dependency_types,
            candidate_sets=candidate_sets,
            remaining_scanner_identifiers=(
                evaluation.deterministic_gates.target_remaining_identifiers
                if evaluation and evaluation.deterministic_gates
                else ()
            ),
            dependency_evidence=(
                evaluation.deterministic_gates.dependency_evidence
                if evaluation and evaluation.deterministic_gates
                else None
            ),
            prior_attempts=base_context.prior_attempts,
        )
        action, verification = propose_and_verify_tactical_action(
            context,
            settings=settings,
        )
        if action is None or verification is None:
            if task.strategy != RoutingStrategy.VERSION_BUMP:
                return None
            # A retry must choose a different authorized action or version;
            # the current-stage registry set was fetched for tactical
            # verification and must not silently replay the failed version.
            if task.status == TaskStatus.NEEDS_RETRY or task.retry_count > 0:
                return None
            return _commit_registry_resolution_fallback(
                task_queue,
                task,
                group,
                evaluation,
                retry_diagnostics_by_task,
                retry_plans_by_task,
                candidate_sets,
                consistency_events=consistency_events,
                errors=errors,
                staged_resolutions=staged_resolutions,
            )
        if not verification.accepted:
            detail = verification.reason
            if registry_error and "candidate" in detail.lower():
                detail = f"{detail} {registry_error}"
            consistency_events.append(
                _build_consistency_event(
                    error_code="TACTICAL_ACTION_REJECTED",
                    task_id=task.task_id,
                    expected_attempt_id=task.current_attempt_id,
                    received_attempt_id=None,
                    action="rejected",
                    details=detail,
                )
            )
            errors.append(f"supervisor: tactical action rejected for {task.task_id}: {detail}")
            return None

        diagnostic_kind = classify_diagnostics(context)
        instruction = verification.instruction or task.instruction
        verified_versions = list(verification.allowed_target_versions)
        canonical_version = (
            verified_versions[0] if verified_versions else verification.selected_version
        )
        decision_reason = (
            f"Tactical Supervisor selected {action.selected_strategy.value} for task "
            f"'{task.task_id}' after {diagnostic_kind.value} evidence: {action.rationale}"
        )
        if action.selected_strategy == TacticalStrategy.ESCALATE_TO_PORTFOLIO:
            referral = (
                f"{task.task_id}: peer-conflict referral requires a coordinated multi-package "
                "resolution; Phase 2 cannot safely select a compatible single-task candidate."
            )
            return SupervisorDecision(
                decision_code=DecisionCode.PEER_CONFLICT_ESCALATION,
                next_node="teardown",
                target_task_ids=[],
                unfixable_task_ids=[task.task_id],
                task_status_updates={task.task_id: TaskStatus.UNFIXABLE},
                new_constraints=[referral],
                decision_reason=decision_reason,
                instructions=(
                    "Peer conflict requires a multi-package portfolio resolution; "
                    "Phase 2 records the referral and ends single-task remediation."
                ),
            )
        if action.selected_strategy == TacticalStrategy.CODE_WORKAROUND:
            if task.strategy == RoutingStrategy.CODE_WORKAROUND:
                return SupervisorDecision(
                    decision_code=DecisionCode.WORKAROUND_DISPATCH,
                    next_node="workaround_subagent",
                    target_task_ids=[task.task_id],
                    revised_instructions={task.task_id: instruction},
                    instructions=instruction,
                    decision_reason=decision_reason,
                )
            return SupervisorDecision(
                decision_code=DecisionCode.PIVOT_TO_WORKAROUND,
                next_node="workaround_subagent",
                target_task_ids=[task.task_id],
                spawn_requests=[
                    TaskSpawnRequest(
                        parent_task_id=task.task_id,
                        strategy=RoutingStrategy.CODE_WORKAROUND,
                        instruction=instruction,
                        reason=decision_reason,
                    )
                ],
                instructions=instruction,
                decision_reason=decision_reason,
            )

        updates = {
            "strategy_stage": verification.strategy_stage or task.strategy_stage,
            "selected_version": verification.selected_version,
            "target_package_name": verification.target_package_name,
            "target_dependency_type": verification.target_dependency_type,
            "instruction": instruction,
            "exhausted_update_path": False,
        }
        if staged_resolutions is not None:
            committed = _project_tactical_task_transition(task, updates)
        else:
            committed = _commit_task_transition(
                task_queue,
                task.task_id,
                updates=updates,
                consistency_events=consistency_events,
            )
        if committed is None:
            return None
        approved_versions = list(
            _approved_candidate_pool(
                context,
                verification.target_package_name
                or task.target_package_name
                or group.vulnerable_component
                or "",
                verification.target_dependency_type or task.target_dependency_type,
            )
            or verified_versions
        )
        diagnostics_for_commit = (
            retry_diagnostics or UpdateRetryDiagnostics(task_id=task.task_id)
        ).model_copy(
            update={
                "strategy_stage": committed.strategy_stage,
                "selected_version": committed.selected_version,
                "target_package_name": committed.target_package_name,
                "target_dependency_type": committed.target_dependency_type,
                "security_floor": _canonical_security_floor(group)[0],
                "candidate_versions_considered": list(dict.fromkeys(approved_versions)),
                "candidate_dependency_types": list(
                    dict.fromkeys(
                        [
                            *verification.allowed_dependency_types,
                        ]
                    )
                ),
                "registry_query_performed": bool(candidate_sets),
                "reasoning_summary": (
                    "policy=verified_candidate_inventory; "
                    f"recommended_candidate={canonical_version or 'none'}; "
                    f"selected_candidate={verification.selected_version or 'none'}; "
                    f"diagnostic_basis={action.diagnostic_basis}; rationale={action.rationale}"
                ),
            }
        )
        retry_plan = SupervisorRetryPlan(
            task_id=task.task_id,
            source_task_revision=committed.task_revision,
            strategy_stage=committed.strategy_stage,
            selected_version=committed.selected_version,
            target_package_name=committed.target_package_name,
            target_dependency_type=committed.target_dependency_type,
            parent_minimum_version=committed.parent_minimum_version,
            attempted_versions=list(diagnostics_for_commit.attempted_versions),
            candidate_versions_considered=approved_versions,
            candidate_dependency_types=list(verification.allowed_dependency_types),
            action="retry_update",
            exact_instruction=instruction,
        )
        decision_code = (
            DecisionCode.RETRY_VERSION_BUMP
            if task.status == TaskStatus.NEEDS_RETRY or task.retry_count > 0
            else DecisionCode.NEW_VERSION_BUMP
        )
        if staged_resolutions is not None:
            staged_resolutions.append(
                _StagedTacticalResolution(
                    task_id=task.task_id,
                    expected_task_revision=task.task_revision,
                    decision_code=decision_code,
                    next_node="update_subagent",
                    instruction=instruction,
                    updates=updates,
                    retry_diagnostics=diagnostics_for_commit,
                    retry_plan=retry_plan,
                )
            )
        else:
            retry_diagnostics_by_task[task.task_id] = diagnostics_for_commit
            retry_plans_by_task[task.task_id] = retry_plan
        return SupervisorDecision(
            decision_code=decision_code,
            next_node="update_subagent",
            target_task_ids=[task.task_id],
            revised_instructions={task.task_id: instruction},
            instructions=instruction,
            decision_reason=decision_reason,
        )


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


def _project_tactical_task_transition(
    task: RemediationTask,
    updates: dict[str, Any],
) -> RemediationTask:
    """Project a tactical input transition without mutating orchestration state.

    Tactical verification happens before the Supervisor guardrails.  This
    helper mirrors the revision behavior of ``_commit_task_transition`` so a
    staged retry plan can be correlated to the revision it will receive if
    the final guarded decision still selects it.
    """
    committed_updates = dict(updates)
    input_changed = any(
        field in committed_updates and committed_updates[field] != getattr(task, field)
        for field in _ATTEMPT_INPUT_FIELDS
    )
    if input_changed:
        committed_updates["task_revision"] = task.task_revision + 1
    return task.model_copy(update=committed_updates)


def _commit_staged_tactical_resolutions(
    staged_resolutions: list[_StagedTacticalResolution],
    decision: SupervisorDecision,
    task_queue: dict[str, RemediationTask],
    retry_diagnostics_by_task: dict[str, UpdateRetryDiagnostics],
    retry_plans_by_task: dict[str, SupervisorRetryPlan],
    *,
    consistency_events: list[StateConsistencyEvent],
) -> None:
    """Commit only tactical data that survived the post-decision guardrails.

    A model proposal is not orchestration state.  The final decision may be
    replaced by deterministic routing after cardinality, NO_FIX, or stale
    target checks.  Comparing the rendered instruction and route before this
    commit prevents a tactical version or retry plan from leaking into a
    deterministic fallback.
    """
    for staged in staged_resolutions:
        if (
            decision.decision_code != staged.decision_code
            or decision.next_node != staged.next_node
            or decision.target_task_ids != [staged.task_id]
            or decision.revised_instructions.get(staged.task_id) != staged.instruction
        ):
            continue
        task = task_queue.get(staged.task_id)
        if (
            task is None
            or task.task_revision != staged.expected_task_revision
            or task.current_attempt_id is not None
        ):
            continue
        committed = _commit_task_transition(
            task_queue,
            staged.task_id,
            updates=staged.updates,
            consistency_events=consistency_events,
        )
        if committed is None:
            continue
        retry_diagnostics_by_task[staged.task_id] = staged.retry_diagnostics
        retry_plans_by_task[staged.task_id] = staged.retry_plan.model_copy(
            update={"source_task_revision": committed.task_revision}
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
    4. Resolve no-fix terminal states; defer the retry-cap terminalization
       until tactical and deterministic replanning has had a chance to pivot.
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

    # ------------------------------------------------------------------
    # 1. Normalize task_queue (copy-on-write)
    # ------------------------------------------------------------------
    raw_task_queue: dict[str, RemediationTask] = dict(state.get("task_queue", {}))
    # Copy-on-write: work with model copies so we never mutate state-owned objects
    task_queue: dict[str, RemediationTask] = {
        tid: t.model_copy() for tid, t in raw_task_queue.items()
    }
    existing_group_ids = {t.parent_group_id for t in task_queue.values()}
    next_task_index = len(task_queue) + 1
    for group in valid_groups:
        if group.group_id not in existing_group_ids:
            task_id = f"task-{next_task_index}"
            task_queue[task_id] = build_initial_remediation_task(group, task_id)
            next_task_index += 1

    # Keep task-owned planner fields synchronized with the initial OSV plan.
    # Later planner commits are the only source allowed to change these fields.
    for task_id, task in list(task_queue.items()):
        group = group_by_id.get(task.parent_group_id)
        task_updates: dict[str, Any] = {}
        if task.qa_policy is None and task.current_attempt_id is None:
            recovered_policy = derive_missing_task_qa_policy(task, group)
            if recovered_policy is not None:
                task_updates["qa_policy"] = recovered_policy
            elif task.status not in _TERMINAL_STATUSES:
                details = (
                    "Task has no recoverable QA policy provenance before dispatch; "
                    "the Supervisor will fail closed."
                )
                errors.append(f"supervisor: task {task_id} has missing QA policy provenance.")
                consistency_events.append(
                    _build_consistency_event(
                        error_code="MISSING_QA_POLICY_PROVENANCE",
                        task_id=task_id,
                        expected_attempt_id=None,
                        received_attempt_id=None,
                        action="rejected",
                        details=details,
                    )
                )
        if (
            task.status not in _TERMINAL_STATUSES
            and task.current_attempt_id is None
            and task.no_fix_stage is None
            and group is not None
            and is_no_fix_group(group)
        ):
            task_updates["no_fix_stage"] = NoFixMitigationStage.PACKAGE_REMOVAL
            if task.selected_version is not None:
                task_updates["selected_version"] = None
            if not task.instruction or task.instruction.strip().casefold() == (
                "no upstream patch or workaround was found. inform the user."
            ):
                task_updates["instruction"] = build_no_fix_package_removal_instruction(group)
        if (
            task.task_revision == 0
            and task.status not in _TERMINAL_STATUSES
            and task.current_attempt_id is None
            and not task.instruction
            and group is not None
            and group.fix_plan is not None
            and group.fix_plan.instruction
        ):
            task_updates["instruction"] = group.fix_plan.instruction
        if (
            task.task_revision == 0
            and task.status not in _TERMINAL_STATUSES
            and task.current_attempt_id is None
            and task.strategy == RoutingStrategy.VERSION_BUMP
            and task.selected_version is not None
        ):
            # Compatibility state may contain a fix-plan version projected
            # before registry verification existed. Clear that uncommitted
            # target so the first active-task routing pass must verify it.
            task_updates["selected_version"] = None
        if (
            task.task_revision == 0
            and task.current_attempt_id is None
            and group is not None
            and is_transitive_group(group)
        ):
            parent_name, parent_version, parent_type = group_parent_context(group)
            if parent_name and task.parent_package_name != parent_name:
                task_updates["parent_package_name"] = parent_name
            if parent_version and task.parent_package_version != parent_version:
                task_updates["parent_package_version"] = parent_version
            if parent_name and task.target_package_name is None:
                task_updates["target_package_name"] = parent_name
                task_updates["target_dependency_type"] = task.target_dependency_type or parent_type
        if task_updates:
            if "qa_policy" in task_updates or "selected_version" in task_updates:
                task_updates["task_revision"] = task.task_revision + 1
            task_queue[task_id] = task.model_copy(update=task_updates)

    # Registry planning is intentionally deferred until deterministic routing
    # has selected one active task below.  The former implementation planned
    # every transitive task while normalizing the queue, which made registry
    # spans and candidate evidence appear to belong to the wrong task and
    # could consume network work for tasks that were not dispatchable yet.

    # ------------------------------------------------------------------
    # 2. Ingest attempt-tagged worker results (active targets only)
    # ------------------------------------------------------------------
    active_target_task_ids = list(state.get("active_target_task_ids") or [])
    action_summaries: list[AgentActionSummary] = state.get("action_summaries") or []
    new_worker_attempt_ids: list[str] = []

    for task_id in active_target_task_ids:
        task = task_queue.get(task_id)
        if task is None:
            continue
        current_attempt_id = task.current_attempt_id
        snapshot = attempt_snapshots_by_id.get(current_attempt_id) if current_attempt_id else None
        result = worker_results_by_attempt.get(current_attempt_id) if current_attempt_id else None
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
                or result.instruction_digest != snapshot.instruction_digest
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
                failed_group = group_by_id.get(task.parent_group_id)
                transitive_failure = bool(failed_group and is_transitive_group(failed_group))
                failure_updates: dict[str, Any] = {
                    "status": TaskStatus.NEEDS_RETRY,
                    "retry_count": task.retry_count + 1,
                }
                _commit_task_transition(
                    task_queue,
                    task_id,
                    updates=failure_updates,
                    close_attempt=True,
                    clear_selected_version=False,
                )
                if transitive_failure:
                    committed_failure_task = task_queue[task_id]
                    retry_diagnostics_by_task[task_id] = prior.model_copy(
                        update={
                            "strategy_stage": committed_failure_task.strategy_stage,
                            "selected_version": committed_failure_task.selected_version,
                            "target_package_name": committed_failure_task.target_package_name,
                            "target_dependency_type": committed_failure_task.target_dependency_type,
                            "exhausted_update_path": committed_failure_task.exhausted_update_path,
                        }
                    )
            processed_worker_attempt_ids.add(current_attempt_id)
            new_worker_attempt_ids.append(current_attempt_id)
            continue

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
        if (
            qa_result.task_id != task_id
            or qa_result.task_revision != task.task_revision
            or snapshot is None
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
            if not qa_requires_rerun:
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
    auto_new_constraints: list[str] = []

    if state.get("status") in {"qa_completed", "qa_failed"}:
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
                        close_attempt=True,
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

                group = group_by_id.get(task.parent_group_id)
                task_updates = {
                    "status": TaskStatus.NEEDS_RETRY,
                    "retry_count": task.retry_count + 1,
                }
                _commit_task_transition(
                    task_queue,
                    resolved_t_id,
                    updates=task_updates,
                    close_attempt=True,
                )
                task = task_queue[resolved_t_id]
                if task.strategy == RoutingStrategy.VERSION_BUMP:
                    # Keep the failed stage intact until tactical reasoning
                    # has had the first opportunity to choose an immediate
                    # pivot. The deterministic planner advances stages only
                    # when tactical reasoning is unavailable or rejected.
                    next_stage = task.strategy_stage
                    prior_diag = retry_diagnostics_by_task.get(resolved_t_id)
                    parent_name, _, parent_type = (
                        group_parent_context(group) if group is not None else (None, None, None)
                    )
                    next_target = task.target_package_name or parent_name
                    next_target_type = task.target_dependency_type or parent_type
                    if prior_diag is None:
                        retry_diagnostics_by_task[resolved_t_id] = UpdateRetryDiagnostics(
                            task_id=resolved_t_id,
                            strategy_stage=next_stage,
                            security_floor=(
                                _canonical_security_floor(group)[0] if group is not None else None
                            ),
                            exhausted_update_path=(
                                task.strategy_stage == SCARemediationStage.CODE_WORKAROUND
                            ),
                            target_package_name=next_target,
                            target_dependency_type=next_target_type,
                            parent_package_name=parent_name,
                            parent_minimum_version=task.parent_minimum_version,
                            selected_version=task.selected_version,
                        )
                    else:
                        retry_diagnostics_by_task[resolved_t_id] = prior_diag.model_copy(
                            update={
                                "strategy_stage": next_stage,
                                "security_floor": (
                                    _canonical_security_floor(group)[0]
                                    if group is not None
                                    else prior_diag.security_floor
                                ),
                                "exhausted_update_path": task.strategy_stage
                                == SCARemediationStage.CODE_WORKAROUND,
                                "target_package_name": next_target,
                                "target_dependency_type": next_target_type,
                                "parent_package_name": parent_name,
                                "parent_minimum_version": task.parent_minimum_version,
                                "selected_version": task.selected_version,
                            }
                        )

    # ------------------------------------------------------------------
    # 4. Resolve no-fix terminal states
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
    # 6. Optional tactical planning, then deterministic retry planning
    # ------------------------------------------------------------------
    tactical_fallback_requires_stage_advance = False
    staged_tactical_resolutions: list[_StagedTacticalResolution] = []
    if decision is None:
        deterministic_target_decision = _deterministic_routing(
            task_queue,
            group_by_id,
            qa_evaluations,
            retry_diagnostics_by_task,
            action_summaries=action_summaries,
            active_target_task_ids=active_target_task_ids,
            current_status=str(state.get("status") or ""),
            triage_required=bool(state.get("triage_required")),
        )
        target_task_id = (
            deterministic_target_decision.target_task_ids[0]
            if deterministic_target_decision.next_node in {"update_subagent", "workaround_subagent"}
            and len(deterministic_target_decision.target_task_ids) == 1
            else None
        )
        target_task = task_queue.get(target_task_id) if target_task_id else None
        if target_task is None or target_task.no_fix_stage is not None:
            # NO_FIX and unsupported/multi-task decisions are deterministic
            # state-machine transitions.  They are never handed to the
            # tactical classifier, and their target is the only task that may
            # be considered this turn.
            decision = deterministic_target_decision
        else:
            target_group = group_by_id.get(target_task.parent_group_id)
            target_evaluation = qa_evaluations.get(target_task.task_id)
            target_worker_result = next(
                (
                    result
                    for result in sorted(
                        worker_results_by_attempt.values(),
                        key=lambda item: item.attempt_id,
                        reverse=True,
                    )
                    if result.task_id == target_task.task_id
                ),
                None,
            )
            if target_group is not None and target_task.status == TaskStatus.NEEDS_RETRY:
                tactical_fallback_context = build_tactical_context(
                    target_task,
                    target_group,
                    evaluation=target_evaluation,
                    worker_result=target_worker_result,
                    retry_diagnostics=retry_diagnostics_by_task.get(target_task.task_id),
                )
                tactical_fallback_requires_stage_advance = (
                    classify_diagnostics(tactical_fallback_context)
                    != TacticalDiagnosticKind.INCONCLUSIVE
                )
            # The tactical layer receives exactly the target selected by the
            # deterministic router.  It cannot scan the queue and drift to a
            # different task because another task happens to be workable.
            tactical_decision = _apply_tactical_supervisor(
                task_queue,
                group_by_id,
                qa_evaluations,
                retry_diagnostics_by_task,
                retry_plans_by_task,
                worker_results_by_attempt,
                attempt_snapshots_by_id,
                target_task_id=target_task_id,
                consistency_events=consistency_events,
                errors=errors,
                staged_resolutions=staged_tactical_resolutions,
            )
            if tactical_decision is not None:
                decision = tactical_decision

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
            advance_failed_stage=str(state.get("status") or "")
            in {
                "qa_completed",
                "qa_failed",
            }
            or tactical_fallback_requires_stage_advance,
            target_task_ids=[target_task_id] if target_task_id else None,
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
                )

    # Retry exhaustion is evaluated only after tactical reasoning and the
    # deterministic planner have had a chance to mark an update path as
    # exhausted or emit a workaround pivot.  The old ordering terminalized a
    # task immediately after QA incremented retry_count, so the planner never
    # saw the failed final update and express-jwt could not pivot.
    pivot_task_ids: set[str] = set()
    if decision is not None and decision.decision_code in {
        DecisionCode.PIVOT_TO_WORKAROUND,
        DecisionCode.EXHAUSTED_UPDATE_PIVOT,
    }:
        pivot_task_ids.update(decision.target_task_ids)
        pivot_task_ids.update(
            request.parent_task_id for request in decision.spawn_requests if request.parent_task_id
        )
    pending_override_task_ids = {
        resolution.task_id
        for resolution in staged_tactical_resolutions
        if resolution.updates.get("strategy_stage") == SCARemediationStage.PACKAGE_OVERRIDE
        and resolution.updates.get("selected_version")
    }
    retry_cap_terminalized = False
    for task_id, task in task_queue.items():
        if (
            task.status == TaskStatus.NEEDS_RETRY
            and task.retry_count >= MAX_RETRIES
            and task.no_fix_stage is None
            and task_id not in pivot_task_ids | pending_override_task_ids
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
            retry_cap_terminalized = True
            logger.info(
                "supervisor: task '%s' marked UNFIXABLE after %d retries.",
                task_id,
                task.retry_count,
            )
    if retry_cap_terminalized:
        # Any earlier routing proposal may have been computed against a task
        # that was just terminalized. Recompute it at the deterministic
        # authority boundary below.
        decision = None

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
    deterministic_decision = _deterministic_routing(
        task_queue,
        group_by_id,
        qa_evaluations,
        retry_diagnostics_by_task,
        action_summaries=action_summaries,
        active_target_task_ids=active_target_task_ids,
        current_status=str(state.get("status") or ""),
        triage_required=bool(state.get("triage_required")),
    )
    # The deterministic decision is authoritative. QA feedback below is
    # carried only when Python produced it for the selected task; no model
    # can add worker feedback or constraints to this projection.
    if decision is None:
        decision = deterministic_decision

    # ------------------------------------------------------------------
    # 7. Guardrails: validate and clamp (or deterministic fallback)
    # ------------------------------------------------------------------
    pivot_parent_status_by_parent: dict[str, TaskStatus] = {}
    pivot_target_parent_ids: set[str] = set()

    if decision is not None and _no_fix_decision_requires_fallback(
        decision, task_queue, group_by_id
    ):
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
                limit=QA_DISPATCH_LIMIT,
            )
        elif decision.next_node == "update_subagent":
            valid_target_ids = _update_worker_task_ids(
                task_queue,
                retry_diagnostics_by_task,
                preferred_ids=list(decision.target_task_ids),
                limit=UPDATE_DISPATCH_LIMIT,
            )
        elif decision.next_node == "workaround_subagent":
            valid_target_ids = []
            for t_id in decision.target_task_ids:
                if t_id not in known_task_ids:
                    continue
                task = task_queue[t_id]
                if task.status in _TERMINAL_STATUSES or task.status not in _WORKABLE_STATUSES:
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
            and decision.next_node == "qa_critic"
            and requested_target_count > QA_DISPATCH_LIMIT
        ):
            logger.warning(
                "supervisor: qa_critic current policy allows exactly 1 target, got %d â€” falling back.",
                requested_target_count,
            )
            needs_fallback = True
        if not needs_fallback and decision.next_node == "update_subagent" and valid_target_ids:
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
                )
                valid_target_ids = list(decision.target_task_ids)
                valid_unfixable_ids = list(decision.unfixable_task_ids)
                clean_revised_instructions = dict(decision.revised_instructions)
                clean_feedback = dict(decision.feedback_by_task)
                clean_updated_task_strategies = dict(decision.updated_task_strategies)

            # Validate task_status_updates — only known tasks and fail-closed statuses
            clean_status_updates: dict[str, TaskStatus] = {}
            _allowed_statuses = {
                TaskStatus.QA_PASSED,
                TaskStatus.UNFIXABLE,
                TaskStatus.INCONCLUSIVE,
            }
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
                    instructions=decision.instructions,
                    decision_reason=decision.decision_reason,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("supervisor: decision rebuild failed (%s) â€” falling back.", exc)
                pivot_parent_status_by_parent = {}
                pivot_target_parent_ids = set()
                decision = _deterministic_routing(
                    task_queue,
                    group_by_id,
                    qa_evaluations,
                    retry_diagnostics_by_task,
                    action_summaries=action_summaries,
                    active_target_task_ids=active_target_task_ids,
                    current_status=str(state.get("status") or ""),
                )

    # ------------------------------------------------------------------
    # 8. Apply guarded updates to task_queue
    # ------------------------------------------------------------------

    # Tactical registry resolutions are staged while the routing decision is
    # being checked.  Commit them only after all guardrails have preserved the
    # same route, target, and exact rendered instruction.
    _commit_staged_tactical_resolutions(
        staged_tactical_resolutions,
        decision,
        task_queue,
        retry_diagnostics_by_task,
        retry_plans_by_task,
        consistency_events=consistency_events,
    )

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

    # 8c. Apply guarded task status overrides.
    _allowed_statuses = {
        TaskStatus.QA_PASSED,
        TaskStatus.UNFIXABLE,
        TaskStatus.INCONCLUSIVE,
    }
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
        )
        resolved_next_node = decision.next_node
        resolved_target_task_ids = list(decision.target_task_ids)
    resolved_target_task_ids = _normalize_target_task_ids_for_node(
        resolved_next_node,
        resolved_target_task_ids,
        task_queue,
        retry_diagnostics_by_task,
        group_by_id,
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
    update_dispatch_authorizations: dict[str, _UpdateDispatchAuthorization] = {}
    if resolved_next_node == "update_subagent":
        dispatchable_update_ids: list[str] = []
        for task_id in resolved_target_task_ids:
            task = task_queue.get(task_id)
            if task is None:
                continue
            plan = retry_plans_by_task.get(task_id)
            diagnostics = retry_diagnostics_by_task.get(task_id)
            authorization = _authorize_update_dispatch(
                task,
                plan=plan,
                diagnostics=diagnostics,
            )
            if authorization is None and task.strategy == RoutingStrategy.VERSION_BUMP:
                group = group_by_id.get(task.parent_group_id)
                if group is not None:
                    recovery_input = diagnostics or UpdateRetryDiagnostics(
                        task_id=task_id,
                        strategy_stage=task.strategy_stage,
                    )
                    recovered_diagnostics, recovered_plans = _run_deterministic_retry_planner(
                        {task_id: task},
                        {task.parent_group_id: group},
                        {task_id: recovery_input},
                    )
                    if not _planner_plan_violations(
                        recovered_plans,
                        {task_id: task},
                        recovered_diagnostics,
                    ):
                        retry_diagnostics_by_task.update(recovered_diagnostics)
                        _commit_retry_plans(
                            task_queue,
                            retry_diagnostics_by_task,
                            retry_plans_by_task,
                            recovered_plans,
                        )
                        task = task_queue.get(task_id)
                        plan = retry_plans_by_task.get(task_id)
                        diagnostics = retry_diagnostics_by_task.get(task_id)
                        authorization = _authorize_update_dispatch(
                            task,
                            plan=plan,
                            diagnostics=diagnostics,
                        )
            if authorization is not None:
                update_dispatch_authorizations[task_id] = authorization
                dispatchable_update_ids.append(task_id)
                continue
            detail = (
                f"Supervisor rejected update dispatch for task {task_id}: "
                "no unattempted registry-approved target version is committed."
            )
            errors.append(f"supervisor: {detail}")
            consistency_events.append(
                _build_consistency_event(
                    error_code="UPDATE_DISPATCH_WITHOUT_CANDIDATE",
                    task_id=task_id,
                    expected_attempt_id=task.current_attempt_id,
                    received_attempt_id=None,
                    action="rejected",
                    details=detail,
                )
            )
            retry_plans_by_task.pop(task_id, None)
            if task.exhausted_update_path:
                continue
            if task.status not in _TERMINAL_STATUSES:
                _commit_task_transition(
                    task_queue,
                    task_id,
                    updates={"status": TaskStatus.INCONCLUSIVE},
                    close_attempt=task.current_attempt_id is not None,
                    clear_selected_version=task.selected_version is not None,
                    consistency_events=consistency_events,
                )
        resolved_target_task_ids = dispatchable_update_ids

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
        )
        resolved_next_node = decision.next_node
        resolved_target_task_ids = _normalize_target_task_ids_for_node(
            resolved_next_node,
            list(decision.target_task_ids),
            task_queue,
            retry_diagnostics_by_task,
            group_by_id,
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
            update_authorization = (
                update_dispatch_authorizations.get(task_id)
                if resolved_next_node == "update_subagent"
                else None
            )
            if resolved_next_node == "update_subagent" and update_authorization is None:
                update_authorization = _authorize_update_dispatch(
                    task,
                    plan=plan,
                    diagnostics=retry_diagnostics_by_task.get(task_id),
                )
                if update_authorization is None:
                    errors.append(
                        f"supervisor: skipped update snapshot for task {task_id}; "
                        "candidate authorization was not committed."
                    )
                    continue
                update_dispatch_authorizations[task_id] = update_authorization
            task, snapshot = _create_attempt_snapshot(
                task,
                dispatch_node=resolved_next_node,
                snapshots_by_id=attempt_snapshots_by_id,
                state_revision=state_revision,
                plan_id=plan.plan_id if plan is not None else None,
                workaround_context=workaround_ctx,
                allowed_target_versions=(
                    list(update_authorization.allowed_target_versions)
                    if update_authorization is not None
                    else []
                ),
                allowed_dependency_types=(
                    list(update_authorization.allowed_dependency_types)
                    if update_authorization is not None
                    else []
                ),
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

    # ------------------------------------------------------------------
    # 9. Return state patch
    # ------------------------------------------------------------------
    return {
        "status": "supervisor_routed",
        "next_routing_step": resolved_next_node,
        "decision_code": decision.decision_code,
        "supervisor_audit": _emit_audit(decision, consistency_events, state_revision),
        "active_target_task_ids": resolved_target_task_ids,
        "feedback_by_task": feedback_by_task,
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
    )
    if decision.next_node not in _VALID_NEXT_NODES:
        logger.critical(
            "supervisor_router: deterministic recomputation produced invalid node '%s'.",
            decision.next_node,
        )
        return "teardown"
    return decision.next_node
