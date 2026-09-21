"""Deterministic retry planning from an outer committed portfolio plan."""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any

from remediation_engine.contracts.schemas import (
    FailureCategory,
    QAEvaluation,
    RemediationTask,
    RoutingStrategy,
    SCARemediationStage,
    SupervisorRetryPlan,
    TaskStatus,
    UpdateRetryDiagnostics,
    VulnerabilityGroup,
)
from remediation_engine.orchestration.supervisor_policy import (
    _TERMINAL_STATUSES,
    _is_exhausted_update_pivot_candidate,
    _task_sort_key,
)
from remediation_engine.orchestration.task_utils import group_parent_context

logger = logging.getLogger(__name__)

UPDATE_DISPATCH_LIMIT: int = 1
QA_DISPATCH_LIMIT: int = 1
_SCA_STAGE_ORDER: dict[SCARemediationStage, int] = {
    SCARemediationStage.OSV_MINIMUM: 0,
    SCARemediationStage.NPM_SAME_MAJOR: 1,
    SCARemediationStage.NPM_LATEST: 2,
    SCARemediationStage.PACKAGE_OVERRIDE: 3,
    SCARemediationStage.CODE_WORKAROUND: 4,
}
_OVERRIDE_DEPENDENCY_TYPES = frozenset({"overrides", "resolutions", "pnpm_overrides"})


def _commit_task_transition(*args: Any, **kwargs: Any) -> Any:
    from remediation_engine.orchestration import supervisor_node

    return supervisor_node._commit_task_transition(*args, **kwargs)


def instruction_digest(instruction: str) -> str:
    """Return the stable digest used to correlate worker input and output."""
    normalized = " ".join((instruction or "").split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _supervisor_dependency_type_candidates(
    strategy_stage: SCARemediationStage,
    target_dependency_type: str | None,
) -> list[str]:
    """Return dependency types already authorized by the outer plan."""
    if target_dependency_type:
        return [target_dependency_type]
    if strategy_stage == SCARemediationStage.PACKAGE_OVERRIDE:
        return sorted(_OVERRIDE_DEPENDENCY_TYPES)
    return []


def _build_high_level_retry_instruction(
    task: RemediationTask,
    group: VulnerabilityGroup | None,
    evaluation: QAEvaluation | None,
    diagnostics: UpdateRetryDiagnostics | None,
) -> str:
    """Synthesize a retry instruction without consulting package metadata."""
    component = group.vulnerable_component if group else task.parent_group_id
    parent_name, _, parent_type = (
        group_parent_context(group) if group is not None else (None, None, None)
    )
    target = task.target_package_name or (
        parent_name if task.strategy_stage != SCARemediationStage.PACKAGE_OVERRIDE else component
    )
    dependency_type = task.target_dependency_type or parent_type
    if diagnostics and diagnostics.selected_version:
        manifest = group.file_paths[0] if group and group.file_paths else "package.json"
        return (
            f"Apply the outer-solver-approved dependency decision for {component}: "
            f"update only {target or component} in {manifest}"
            f"{f' ({dependency_type})' if dependency_type else ''} "
            f"to exact version {diagnostics.selected_version}; "
            "do not query a registry or choose another version; "
            "use modify_and_validate_npm_dependency so synchronization runs immediately after the edit."
        )
    if diagnostics and diagnostics.package_abandoned:
        return (
            f"The outer portfolio plan has no remaining approved update candidate for {component}. "
            "Do not select another version; report the bounded update failure so the Supervisor can pivot safely."
        )
    if evaluation and evaluation.failure_category == FailureCategory.PEER_CONFLICT:
        return (
            f"Use only the outer-solver-approved peer-compatible candidate for {component}; "
            "preserve the committed package and manifest targets."
        )
    return (
        f"Execute the next outer-solver-approved manifest candidate for {component}. "
        "Do not query a registry or invent a candidate."
    )


def _override_dependency_type(group: VulnerabilityGroup | None) -> str:
    """Return the package-manager-native override field for an SCA group."""
    managers = {
        (issue.package_manager or "").strip().lower()
        for issue in (group.localized_issues if group else [])
    }
    if "yarn" in managers:
        return "resolutions"
    if "pnpm" in managers:
        return "pnpm_overrides"
    return "overrides"


def _planner_plan_violations(
    plans: dict[str, SupervisorRetryPlan],
    task_queue: dict[str, RemediationTask],
    diagnostics_by_task: dict[str, UpdateRetryDiagnostics],
) -> list[str]:
    """Validate retry-plan semantics before a plan can mutate routing state.

    Free-form plan evidence is intentionally not treated as an authority. These
    checks enforce the small set of invariants that must hold for an exact
    worker instruction to be safe.  Returning human-readable violations also
    makes the corrective replan visible in LangSmith through the supervisor's
    accumulated ``errors`` field.
    """
    violations: list[str] = []
    for task_id, plan in plans.items():
        task = task_queue.get(task_id)
        if task is None:
            violations.append(f"task {task_id}: planner returned an unknown task")
            continue
        if task.status in _TERMINAL_STATUSES:
            violations.append(f"task {task_id}: planner returned a plan for terminal task")
        if plan.source_task_revision != task.task_revision:
            violations.append(
                f"task {task_id}: planner snapshot revision {plan.source_task_revision} "
                f"does not match current revision {task.task_revision}"
            )

        attempted = {
            version.strip().lstrip("vV").lower() for version in plan.attempted_versions if version
        }
        diagnostics = diagnostics_by_task.get(task_id)
        if diagnostics is not None:
            attempted.update(
                version.strip().lstrip("vV").lower()
                for version in diagnostics.attempted_versions
                if version
            )

        selected = (
            plan.selected_version.strip().lstrip("vV").lower() if plan.selected_version else None
        )
        if selected and selected in attempted:
            violations.append(
                f"task {task_id}: selected version {plan.selected_version} was already attempted"
            )
        if (
            plan.strategy_stage == SCARemediationStage.NPM_LATEST
            and selected
            and plan.latest_version_seen
            and selected != plan.latest_version_seen.strip().lstrip("vV").lower()
        ):
            violations.append(
                f"task {task_id}: npm_latest selected {plan.selected_version}, "
                f"but registry latest is {plan.latest_version_seen}"
            )
        if plan.action == "retry_update" and selected is None:
            violations.append(
                f"task {task_id}: retry_update requires an unattempted exact selected_version"
            )
        if plan.action == "retry_update" and plan.exhausted_update_path:
            violations.append(f"task {task_id}: exhausted update path cannot retry update")
        if (
            plan.action == "retry_update"
            and plan.strategy_stage == SCARemediationStage.CODE_WORKAROUND
        ):
            violations.append(f"task {task_id}: retry_update cannot use code_workaround stage")
        if (
            plan.action == "retry_update"
            and task.strategy == RoutingStrategy.VERSION_BUMP
            and _SCA_STAGE_ORDER[plan.strategy_stage] < _SCA_STAGE_ORDER[task.strategy_stage]
        ):
            violations.append(
                f"task {task_id}: retry plan stage {plan.strategy_stage.value} regresses "
                f"from committed stage {task.strategy_stage.value}"
            )
        if (
            plan.action == "pivot_workaround"
            and plan.strategy_stage != SCARemediationStage.NPM_LATEST
        ):
            violations.append(f"task {task_id}: workaround pivot must be committed at npm_latest")
        if plan.action == "pivot_workaround" and selected is not None:
            violations.append(
                f"task {task_id}: workaround pivot cannot retain selected version {plan.selected_version}"
            )
    return violations


def _repair_invalid_planner_plans(
    plans: dict[str, SupervisorRetryPlan],
    diagnostics_by_task: dict[str, UpdateRetryDiagnostics],
    task_queue: dict[str, RemediationTask],
    group_by_id: dict[str, VulnerabilityGroup],
    violations: list[str] | None = None,
) -> tuple[dict[str, UpdateRetryDiagnostics], dict[str, SupervisorRetryPlan]]:
    """Apply a deterministic, fail-closed repair after corrective replanning.

    A valid unattempted candidate already present in planner evidence is safe to
    commit.  If no such candidate exists at the latest stage, the only safe
    action is the existing workaround pivot.  In particular, this function
    never preserves an invalid selected version merely to keep the graph
    moving.
    """
    repaired_diagnostics = dict(diagnostics_by_task)
    repaired_plans: dict[str, SupervisorRetryPlan] = {}
    invalid_task_ids = {
        match.group(1)
        for violation in (violations or [])
        if (match := re.match(r"task\s+(task-[\w-]+):", violation))
    }
    for task_id, plan in plans.items():
        if invalid_task_ids and task_id not in invalid_task_ids:
            repaired_plans[task_id] = plan
            continue
        diagnostics = repaired_diagnostics.get(task_id)
        attempted = {
            version.strip().lstrip("vV").lower() for version in plan.attempted_versions if version
        }
        if diagnostics is not None:
            attempted.update(
                version.strip().lstrip("vV").lower()
                for version in diagnostics.attempted_versions
                if version
            )

        candidate = None
        plan_regresses = (
            task_queue[task_id].strategy == RoutingStrategy.VERSION_BUMP
            and _SCA_STAGE_ORDER[plan.strategy_stage]
            < _SCA_STAGE_ORDER[task_queue[task_id].strategy_stage]
        )
        # A correction cannot reopen an earlier stage.  In particular, a
        # stale same-major proposal must not revive a version-bump parent that
        # has already reached code_workaround.  Let the fail-closed branch
        # below create the deterministic latest-stage pivot instead.
        if not plan_regresses and plan.strategy_stage != SCARemediationStage.CODE_WORKAROUND:
            # At npm_latest, only the registry's declared latest version is
            # safe to repair into a retry. Older candidates are already
            # covered by the bounded update stages and selecting one here can
            # produce a null/stale dispatch state after the guardrail clears
            # the invalid plan. If latest was attempted, fail closed into the
            # deterministic workaround pivot.
            repair_candidates = (
                [plan.latest_version_seen]
                if plan.strategy_stage == SCARemediationStage.NPM_LATEST
                else [plan.latest_version_seen, *plan.candidate_versions_considered]
            )
            for version in repair_candidates:
                if version and version.strip().lstrip("vV").lower() not in attempted:
                    candidate = version.strip().lstrip("vV")
                    break

        if candidate:
            effective_stage = plan.strategy_stage
            if (
                effective_stage == SCARemediationStage.NPM_SAME_MAJOR
                and plan.latest_version_seen
                and candidate == plan.latest_version_seen.strip().lstrip("vV")
            ):
                effective_stage = SCARemediationStage.NPM_LATEST
            if diagnostics is None:
                diagnostics = UpdateRetryDiagnostics(task_id=task_id)
            group = group_by_id.get(task_queue[task_id].parent_group_id)
            target_package = task_queue[task_id].target_package_name or (
                group_parent_context(group)[0] if group is not None else None
            )
            target_type = task_queue[task_id].target_dependency_type
            diagnostics = diagnostics.model_copy(
                update={
                    "strategy_stage": effective_stage,
                    "selected_version": candidate,
                    "candidate_versions_considered": list(
                        dict.fromkeys([*diagnostics.candidate_versions_considered, candidate])
                    ),
                    "registry_query_performed": False,
                    "exhausted_update_path": False,
                    "target_package_name": target_package,
                    "target_dependency_type": target_type,
                }
            )
            repaired_diagnostics[task_id] = diagnostics
            instruction = _build_high_level_retry_instruction(
                task_queue[task_id].model_copy(update={"strategy_stage": effective_stage}),
                group,
                None,
                diagnostics,
            )
            repaired_plans[task_id] = plan.model_copy(
                update={
                    "strategy_stage": effective_stage,
                    "selected_version": candidate,
                    "candidate_versions_considered": diagnostics.candidate_versions_considered,
                    "action": "retry_update",
                    "exact_instruction": instruction,
                    "exhausted_update_path": False,
                    "target_package_name": target_package,
                    "target_dependency_type": target_type,
                }
            )
            continue

        # No direct unattempted candidate can be proven. Clear stale selection
        # and pivot at the terminal update stage so no guessed/old version is
        # sent to the dumb update worker.
        effective_stage = SCARemediationStage.NPM_LATEST
        if diagnostics is None:
            diagnostics = UpdateRetryDiagnostics(task_id=task_id)
        diagnostics = diagnostics.model_copy(
            update={
                "strategy_stage": effective_stage,
                "selected_version": None,
                "exhausted_update_path": True,
                "target_package_name": task_queue[task_id].target_package_name,
                "target_dependency_type": task_queue[task_id].target_dependency_type,
            }
        )
        repaired_diagnostics[task_id] = diagnostics
        group = group_by_id.get(task_queue[task_id].parent_group_id)
        component = group.vulnerable_component if group else task_queue[task_id].parent_group_id
        instruction = (
            f"Implement a code workaround or isolation strategy for {component} "
            "because the manifest-based update path is exhausted."
        )
        repaired_plans[task_id] = plan.model_copy(
            update={
                "strategy_stage": effective_stage,
                "selected_version": None,
                "exhausted_update_path": True,
                "action": "pivot_workaround",
                "exact_instruction": instruction,
                "target_package_name": task_queue[task_id].target_package_name,
                "target_dependency_type": task_queue[task_id].target_dependency_type,
            }
        )
    return repaired_diagnostics, repaired_plans


def _build_deterministic_retry_plan(
    task: RemediationTask,
    diagnostics: UpdateRetryDiagnostics,
    group: VulnerabilityGroup | None,
    *,
    requested_stage: SCARemediationStage | None = None,
) -> SupervisorRetryPlan:
    """Build a retry plan solely from the outer solver's committed allowlist."""
    requested = requested_stage or task.strategy_stage
    effective_stage = (
        requested
        if _SCA_STAGE_ORDER.get(requested, 99) >= _SCA_STAGE_ORDER.get(task.strategy_stage, 0)
        else task.strategy_stage
    )
    attempted = {
        str(version).strip().lstrip("vV").lower()
        for version in (*diagnostics.attempted_versions, *diagnostics.executed_versions)
        if str(version).strip()
    }
    approved_versions = list(
        dict.fromkeys(
            str(version).strip().lstrip("vV")
            for version in task.allowed_target_versions
            if str(version).strip()
        )
    )
    selected_version = next(
        (
            version
            for version in [task.selected_version, *approved_versions]
            if version and version.lower() not in attempted
        ),
        None,
    )
    exhausted = task.strategy == RoutingStrategy.VERSION_BUMP and selected_version is None
    if task.strategy == RoutingStrategy.CODE_WORKAROUND:
        exhausted = False
    target_package_name = task.target_package_name
    target_dependency_type = task.target_dependency_type
    candidate_dependency_types = list(
        dict.fromkeys(
            str(value).strip() for value in task.allowed_dependency_types if str(value).strip()
        )
    )
    failure_reason = (
        "The outer portfolio plan has no remaining approved version." if exhausted else ""
    )
    effective_task = task.model_copy(update={"selected_version": selected_version})
    effective_diagnostics = diagnostics.model_copy(
        update={
            "strategy_stage": effective_stage,
            "selected_version": selected_version,
            "candidate_versions_considered": approved_versions,
            "latest_version_seen": approved_versions[-1] if approved_versions else None,
            "registry_query_performed": False,
            "exhausted_update_path": exhausted,
            "target_package_name": target_package_name,
            "target_dependency_type": target_dependency_type,
            "candidate_dependency_types": candidate_dependency_types,
            "parent_package_name": (
                group.parent_package_name if group is not None else diagnostics.parent_package_name
            ),
            "parent_minimum_version": task.parent_minimum_version,
            "failure_reason": failure_reason,
        }
    )
    if exhausted:
        component = group.vulnerable_component if group else task.parent_group_id
        instruction = (
            f"Implement the committed workaround path for {component}; "
            "the outer portfolio plan has no remaining approved update candidate."
        )
    else:
        instruction = _build_high_level_retry_instruction(
            effective_task,
            group,
            None,
            effective_diagnostics,
        )
    return SupervisorRetryPlan(
        task_id=task.task_id,
        source_task_revision=task.task_revision,
        strategy_stage=effective_stage,
        selected_version=selected_version,
        attempted_versions=list(diagnostics.attempted_versions),
        candidate_versions_considered=approved_versions,
        candidate_dependency_types=candidate_dependency_types,
        latest_version_seen=approved_versions[-1] if approved_versions else None,
        exhausted_update_path=exhausted,
        package_abandoned=diagnostics.package_abandoned,
        target_package_name=target_package_name,
        target_dependency_type=target_dependency_type,
        parent_minimum_version=task.parent_minimum_version,
        action="pivot_workaround" if exhausted else "retry_update",
        exact_instruction=instruction,
    )


def _needs_planner(
    task_queue: dict[str, RemediationTask],
    qa_evaluations: dict[str, QAEvaluation],
    retry_diagnostics_by_task: dict[str, UpdateRetryDiagnostics],
    current_status: str,
) -> bool:
    """Return True when deterministic retry planning should run.

    Retry planning is reserved for VERSION_BUMP retry analysis and playbook selection.
    CODE_WORKAROUND tasks do not use the npm version planner.
    """
    version_bump_retries = [
        t
        for t in task_queue.values()
        if (
            t.status == TaskStatus.NEEDS_RETRY
            and t.strategy == RoutingStrategy.VERSION_BUMP
            and (
                t.strategy_stage
                in {
                    SCARemediationStage.OSV_MINIMUM,
                    SCARemediationStage.NPM_SAME_MAJOR,
                    SCARemediationStage.NPM_LATEST,
                }
                or (
                    t.strategy_stage == SCARemediationStage.CODE_WORKAROUND
                    and not t.parent_package_name
                    and t.target_dependency_type
                    not in {
                        "overrides",
                        "resolutions",
                        "pnpm_overrides",
                    }
                )
            )
        )
    ]
    if not version_bump_retries:
        return False
    return (
        current_status == "qa_completed" or bool(qa_evaluations) or bool(retry_diagnostics_by_task)
    )


def _run_deterministic_retry_planner(
    task_queue: dict[str, RemediationTask],
    group_by_id: dict[str, VulnerabilityGroup],
    retry_diagnostics_by_task: dict[str, UpdateRetryDiagnostics],
) -> tuple[dict[str, UpdateRetryDiagnostics], dict[str, SupervisorRetryPlan]]:
    """Plan every actionable retry from task state and deterministic registry facts."""
    updated_diagnostics = dict(retry_diagnostics_by_task)
    plans: dict[str, SupervisorRetryPlan] = {}
    retry_tasks = sorted(
        (
            task
            for task in task_queue.values()
            if task.status == TaskStatus.NEEDS_RETRY
            and task.strategy == RoutingStrategy.VERSION_BUMP
            and task.strategy_stage
            in {
                SCARemediationStage.OSV_MINIMUM,
                SCARemediationStage.NPM_SAME_MAJOR,
                SCARemediationStage.NPM_LATEST,
                SCARemediationStage.CODE_WORKAROUND,
            }
        ),
        key=lambda task: _task_sort_key(task, group_by_id),
    )
    for task in retry_tasks:
        diagnostics = updated_diagnostics.get(
            task.task_id,
            UpdateRetryDiagnostics(task_id=task.task_id, strategy_stage=task.strategy_stage),
        )
        # The deterministic router owns the exhausted-update pivot. Never
        # reopen a path that its guardrail has already proven exhausted.
        if _is_exhausted_update_pivot_candidate(task, diagnostics):
            continue
        group = group_by_id.get(task.parent_group_id)
        plan = _build_deterministic_retry_plan(task, diagnostics, group)
        plans[task.task_id] = plan
        updated_diagnostics[task.task_id] = diagnostics.model_copy(
            update={
                "strategy_stage": plan.strategy_stage,
                "security_floor": diagnostics.security_floor,
                "selected_version": plan.selected_version,
                "candidate_versions_considered": plan.candidate_versions_considered,
                "latest_version_seen": plan.latest_version_seen,
                "registry_query_performed": False,
                "exhausted_update_path": plan.exhausted_update_path,
                "target_package_name": plan.target_package_name,
                "target_dependency_type": plan.target_dependency_type,
                "candidate_dependency_types": plan.candidate_dependency_types,
                "parent_minimum_version": plan.parent_minimum_version,
            }
        )
    return updated_diagnostics, plans


def _commit_retry_plans(
    task_queue: dict[str, RemediationTask],
    retry_diagnostics_by_task: dict[str, UpdateRetryDiagnostics],
    retry_plans_by_task: dict[str, SupervisorRetryPlan],
    plans: dict[str, SupervisorRetryPlan],
) -> None:
    """Commit deterministic retry inputs before the next worker dispatch."""
    for task_id, plan in sorted(plans.items()):
        task = task_queue.get(task_id)
        if task is None:
            continue
        committed_task = _commit_task_transition(
            task_queue,
            task_id,
            updates={
                "strategy_stage": plan.strategy_stage,
                "instruction": plan.exact_instruction,
                "selected_version": plan.selected_version,
                "exhausted_update_path": plan.exhausted_update_path,
                "target_package_name": plan.target_package_name or task.target_package_name,
                "target_dependency_type": plan.target_dependency_type
                or task.target_dependency_type,
                "parent_minimum_version": plan.parent_minimum_version
                or task.parent_minimum_version,
            },
            close_attempt=True,
            clear_selected_version=plan.selected_version is None,
        )
        if committed_task is None:
            continue
        diagnostics = retry_diagnostics_by_task.get(task_id)
        if diagnostics is not None:
            retry_diagnostics_by_task[task_id] = diagnostics.model_copy(
                update={
                    "committed_attempt_id": committed_task.current_attempt_id,
                    "strategy_stage": committed_task.strategy_stage,
                    "selected_version": committed_task.selected_version,
                    "candidate_dependency_types": plan.candidate_dependency_types,
                    "exhausted_update_path": committed_task.exhausted_update_path,
                    "target_package_name": committed_task.target_package_name,
                    "target_dependency_type": committed_task.target_dependency_type,
                    "parent_minimum_version": committed_task.parent_minimum_version,
                    "instruction_digest": instruction_digest(committed_task.instruction),
                }
            )
        retry_plans_by_task[task_id] = plan.model_copy(
            update={"source_task_revision": committed_task.task_revision}
        )
