"""
supervisor_node.py - Agentic Supervisor Node for Phase 5 hub-and-spoke orchestration.

Phase 2 architecture: High-Level Commander
------------------------------------------
The Supervisor now acts as a high-level commander:

  Router:
    A zero-shot ``ChatOpenAI.with_structured_output(SupervisorDecision)`` call
    that decides which worker should run next, whether retry instructions must
    be refreshed, and whether a child task must be spawned for a strategy pivot.

  Guardrails (Python):
    Validate and apply the decision: reject unknown task IDs, clamp cardinality,
    apply copy-on-write task updates, materialize spawn requests, enforce depth
    and queue-size caps.

Public API
----------
MAX_RETRIES : int
    Maximum number of QA-fail-retry cycles before a task is marked unfixable.
build_supervisor_prompt(state) -> str
    Builds the Router prompt text.
run_supervisor_node(state) -> Dict[str, Any]
    LangGraph node callable.
supervisor_router(state) -> str
    Conditional-edge callable: reads ``next_routing_step`` from state.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Set, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from src.contracts.schemas import (
    MAX_ANCESTRY_DEPTH,
    MAX_TASK_QUEUE_SIZE,
    AgentActionStatus,
    AgentActionSummary,
    FailureCategory,
    QAEvaluation,
    RemediationTask,
    RoutingStrategy,
    SupervisorDecision,
    TaskSpawnRequest,
    TaskStatus,
    UpdateRetryDiagnostics,
    VulnerabilityGroup,
)
from src.orchestrator.state import OrchestratorState
from src.orchestrator.subagent_runtime import run_bounded_subagent_loop
from src.orchestrator.task_utils import build_initial_remediation_task
from src.tools.registry_tools import view_npm_package_versions

logger = logging.getLogger(__name__)

MAX_RETRIES: int = 2
UPDATE_BATCH_SIZE: int = 10

_VALID_NEXT_NODES: Set[str] = {
    "update_subagent",
    "workaround_subagent",
    "qa_critic",
    "teardown",
}
_DEFAULT_MODEL = "gpt-4o-mini"

# ---------------------------------------------------------------------------
# Task status helpers
# ---------------------------------------------------------------------------

_TERMINAL_STATUSES = frozenset({
    TaskStatus.QA_PASSED,
    TaskStatus.UNFIXABLE,
})
_WORKABLE_STATUSES = frozenset({
    TaskStatus.PENDING,
    TaskStatus.NEEDS_RETRY,
})


def _resolve_task_id_from_identifier(
    identifier: str,
    task_queue: Dict[str, RemediationTask],
    active_target_task_ids: List[str],
) -> Optional[str]:
    """
    Resolve a task or legacy group identifier to the most relevant task_id.

    Preference order:
    1. Exact task_id match.
    2. Active target task(s) whose parent_group_id matches the identifier.
    3. Non-terminal task(s) whose parent_group_id matches the identifier.
    4. Any task(s) whose parent_group_id matches the identifier.
    """
    if identifier in task_queue:
        return identifier

    active_matches = [
        task_id
        for task_id in active_target_task_ids
        if task_id in task_queue and task_queue[task_id].parent_group_id == identifier
    ]
    if len(active_matches) == 1:
        return active_matches[0]

    non_terminal_matches = [
        task.task_id
        for task in task_queue.values()
        if task.parent_group_id == identifier and task.status not in _TERMINAL_STATUSES
    ]
    if len(non_terminal_matches) == 1:
        return non_terminal_matches[0]

    all_matches = [
        task.task_id
        for task in task_queue.values()
        if task.parent_group_id == identifier
    ]
    if len(all_matches) == 1:
        return all_matches[0]

    return None


def _normalize_qa_evaluations_for_tasks(
    qa_evaluations: Dict[str, QAEvaluation],
    task_queue: Dict[str, RemediationTask],
    active_target_task_ids: List[str],
) -> Dict[str, QAEvaluation]:
    """Re-key QA evaluations to concrete task_ids when possible."""
    normalized: Dict[str, QAEvaluation] = {}
    for identifier, evaluation in qa_evaluations.items():
        resolved_t_id = _resolve_task_id_from_identifier(
            identifier,
            task_queue,
            active_target_task_ids,
        )
        if resolved_t_id is None:
            resolved_t_id = _resolve_task_id_from_identifier(
                evaluation.task_id,
                task_queue,
                active_target_task_ids,
            )
        if resolved_t_id is None:
            continue
        normalized[resolved_t_id] = QAEvaluation(
            task_id=resolved_t_id,
            passed=evaluation.passed,
            failure_category=evaluation.failure_category,
            retry_feedback=evaluation.retry_feedback,
        )
    return normalized


def _constraint_entry_for_task(
    task: RemediationTask,
    group: VulnerabilityGroup,
) -> str:
    """Build a deterministic constraints-ledger entry for a QA-passed task."""
    component = (group.vulnerable_component or task.parent_group_id).strip()
    fix_plan = group.fix_plan

    if task.strategy == RoutingStrategy.VERSION_BUMP:
        fixed_version = (fix_plan.fixed_version if fix_plan else None) or "unknown"
        return f"{component}: keep resolved version at {fixed_version}"

    return f"{component}: preserve validated security workaround"


def _missing_retry_revised_instructions(
    next_node: str,
    target_task_ids: List[str],
    revised_instructions: Dict[str, str],
    task_queue: Dict[str, RemediationTask],
) -> List[str]:
    """Return retry-bound update targets that are missing exact revised instructions."""
    if next_node != "update_subagent":
        return []

    missing: List[str] = []
    for task_id in target_task_ids:
        task = task_queue.get(task_id)
        if task is None or task.status != TaskStatus.NEEDS_RETRY:
            continue
        if not revised_instructions.get(task_id, "").strip():
            missing.append(task_id)
    return missing


def _latest_action_summary_by_task(
    action_summaries: List[AgentActionSummary],
    task_queue: Dict[str, RemediationTask],
    active_target_task_ids: List[str],
) -> Dict[str, AgentActionSummary]:
    """Return the most recent action summary keyed by resolved task_id."""
    latest: Dict[str, AgentActionSummary] = {}
    for summary in action_summaries:
        resolved_task_id = _resolve_task_id_from_identifier(
            summary.task_id,
            task_queue,
            active_target_task_ids,
        )
        if resolved_task_id is None:
            continue
        latest[resolved_task_id] = summary
    return latest


def _build_high_level_retry_instruction(
    task: RemediationTask,
    group: Optional[VulnerabilityGroup],
    evaluation: Optional[QAEvaluation],
    diagnostics: Optional[UpdateRetryDiagnostics],
) -> str:
    """Synthesize a high-level retry instruction for the update worker."""
    component = group.vulnerable_component if group else task.parent_group_id
    category = evaluation.failure_category if evaluation else None
    if diagnostics and diagnostics.package_abandoned:
        return (
            f"Investigate whether {component} still has a supported manifest-based update path. "
            "Use registry evidence to confirm whether the package is unpublished, abandoned, or "
            "otherwise exhausted, and report that clearly if no valid update path remains."
        )
    if category == FailureCategory.PEER_CONFLICT:
        return (
            f"Investigate compatible patched releases or override paths for {component}. "
            "Prioritize peer-compatible backports or npm overrides that preserve validation."
        )
    if category == FailureCategory.SECURITY_FLAG:
        return (
            f"Investigate patched manifest remediation paths for {component}. "
            "Use registry evidence to compare newer releases, backported patches, and override strategies."
        )
    if category == FailureCategory.BREAKING_CHANGE:
        return (
            f"Document the validated version outcome for {component} and the specific API or test regressions. "
            "Do not search for another version unless new evidence shows a safer compatible patch exists."
        )
    return (
        f"Investigate remaining safe manifest remediation paths for {component}. "
        "Use registry evidence and prior validation failures to choose the next bounded retry."
    )


def _worker_node_for_strategy(strategy: RoutingStrategy) -> str:
    """Return the worker node that handles a given routing strategy."""
    if strategy == RoutingStrategy.VERSION_BUMP:
        return "update_subagent"
    return "workaround_subagent"


def _parent_status_for_strategy_pivot(
    parent_task: RemediationTask,
    new_strategy: RoutingStrategy,
    qa_evaluations: Dict[str, QAEvaluation],
) -> TaskStatus:
    """
    Choose the terminal parent status when a strategy pivot spawns a child task.

    BREAKING_CHANGE means the version bump itself succeeded but caused regressions,
    so the parent attempt should become QA_PASSED and the child handles follow-on
    workaround work. Other pivots represent an exhausted parent strategy attempt.
    """
    evaluation = qa_evaluations.get(parent_task.task_id)
    if (
        parent_task.strategy == RoutingStrategy.VERSION_BUMP
        and new_strategy == RoutingStrategy.CODE_WORKAROUND
        and evaluation is not None
        and evaluation.failure_category == FailureCategory.BREAKING_CHANGE
    ):
        return TaskStatus.QA_PASSED
    return TaskStatus.UNFIXABLE


# ---------------------------------------------------------------------------
# Deterministic fallback router
# ---------------------------------------------------------------------------


def _deterministic_routing(
    task_queue: Dict[str, RemediationTask],
    group_by_id: Dict[str, VulnerabilityGroup],
    qa_evaluations: Dict[str, QAEvaluation],
    retry_diagnostics_by_task: Dict[str, UpdateRetryDiagnostics],
    action_summaries: Optional[List[AgentActionSummary]] = None,
    active_target_task_ids: Optional[List[str]] = None,
    current_status: str = "",
) -> SupervisorDecision:
    """
    Pure-Python fallback routing used when the LLM call fails.

    Implements the same priority rules described in the supervisor prompt.
    """
    tasks = list(task_queue.values())
    non_terminal = [t for t in tasks if t.status not in _TERMINAL_STATUSES]
    latest_summaries = _latest_action_summary_by_task(
        action_summaries or [],
        task_queue,
        list(active_target_task_ids or []),
    )

    # All tasks are terminal → teardown
    if not non_terminal:
        return SupervisorDecision(
            next_node="teardown",
            target_task_ids=[],
            instructions="All tasks are terminal. Proceeding to teardown.",
            decision_reason="No actionable tasks remain.",
        )

    # If active batch is all optimistically_fixed → route to qa_critic
    active_target_ids = set(active_target_task_ids or [])
    current_batch = [t for t in tasks if t.task_id in active_target_ids]
    if current_status != "qa_completed" and current_batch and all(
        t.status == TaskStatus.OPTIMISTICALLY_FIXED for t in current_batch
    ):
        return SupervisorDecision(
            next_node="qa_critic",
            target_task_ids=[t.task_id for t in current_batch],
            instructions="Run QA on the current remediated batch before starting more remediation.",
            decision_reason=(
                f"Routing the current batch of {len(current_batch)} optimistically fixed task(s) to QA."
            ),
        )

    all_optimistic = all(
        t.status == TaskStatus.OPTIMISTICALLY_FIXED for t in non_terminal
    )
    if all_optimistic:
        return SupervisorDecision(
            next_node="qa_critic",
            target_task_ids=[t.task_id for t in non_terminal],
            instructions="Run QA on the remaining optimistically fixed tasks.",
            decision_reason="Routing all remaining optimistically fixed tasks to QA.",
        )

    # Collect tasks that still need work
    workable = [t for t in non_terminal if t.status in _WORKABLE_STATUSES]

    exhausted_retry = next(
        (
            task
            for task in workable
            if task.strategy == RoutingStrategy.VERSION_BUMP
            and task.status == TaskStatus.NEEDS_RETRY
            and (
                retry_diagnostics_by_task.get(task.task_id) is not None
                and (
                    retry_diagnostics_by_task[task.task_id].package_abandoned
                    or retry_diagnostics_by_task[task.task_id].exhausted_update_path
                )
            )
        ),
        None,
    )
    if exhausted_retry is not None:
        component = (
            group_by_id.get(exhausted_retry.parent_group_id).vulnerable_component
            if exhausted_retry.parent_group_id in group_by_id
            else exhausted_retry.parent_group_id
        )
        diagnostics = retry_diagnostics_by_task.get(exhausted_retry.task_id)
        return SupervisorDecision(
            next_node="workaround_subagent",
            target_task_ids=[exhausted_retry.task_id],
            updated_task_strategies={
                exhausted_retry.task_id: RoutingStrategy.CODE_WORKAROUND
            },
            revised_instructions={
                exhausted_retry.task_id: (
                    f"Implement a code workaround or isolation strategy for {component} because "
                    "manifest-based update remediation appears exhausted after bounded registry-guided retries."
                )
            },
            instructions="Pivot exhausted update remediation to the workaround worker.",
            decision_reason=(
                f"Retry diagnostics show that '{component}' no longer has a remaining manifest-based update path."
            ),
        )

    retry_version_bump = [
        t
        for t in workable
        if t.strategy == RoutingStrategy.VERSION_BUMP and t.status == TaskStatus.NEEDS_RETRY
    ]
    if retry_version_bump:
        batch = retry_version_bump[:UPDATE_BATCH_SIZE]
        feedback_by_task: Dict[str, str] = {}
        revised_instructions: Dict[str, str] = {}
        for task in batch:
            evaluation = qa_evaluations.get(task.task_id)
            if evaluation and evaluation.retry_feedback:
                feedback_by_task[task.task_id] = evaluation.retry_feedback
            revised_instructions[task.task_id] = _build_high_level_retry_instruction(
                task,
                group_by_id.get(task.parent_group_id),
                evaluation,
                retry_diagnostics_by_task.get(task.task_id),
            )
        return SupervisorDecision(
            next_node="update_subagent",
            target_task_ids=[t.task_id for t in batch],
            feedback_by_task=feedback_by_task,
            revised_instructions=revised_instructions,
            instructions="Route retry-bound dependency tasks back to the update worker with high-level retry goals.",
            decision_reason=(
                f"Routing {len(batch)} retry VERSION_BUMP task(s) to update_subagent for registry-guided evidence gathering."
            ),
        )

    # VERSION_BUMP tasks batch to update_subagent only for non-retry work.
    version_bump = [
        t
        for t in workable
        if t.strategy == RoutingStrategy.VERSION_BUMP and t.status != TaskStatus.NEEDS_RETRY
    ]
    if version_bump:
        batch = version_bump[:UPDATE_BATCH_SIZE]
        feedback_by_task: Dict[str, str] = {}
        for t in batch:
            eval_ = qa_evaluations.get(t.task_id)
            if eval_ and eval_.retry_feedback:
                feedback_by_task[t.task_id] = eval_.retry_feedback
        return SupervisorDecision(
            next_node="update_subagent",
            target_task_ids=[t.task_id for t in batch],
            feedback_by_task=feedback_by_task,
            instructions="Apply the required version bump(s) in the package manifest(s) for this batch only.",
            decision_reason=(
                f"Routing {len(batch)} VERSION_BUMP task(s) to update_subagent (batch size cap {UPDATE_BATCH_SIZE})."
            ),
        )

    # CODE_WORKAROUND tasks: send exactly one at a time to workaround_subagent
    workaround = [t for t in workable if t.strategy == RoutingStrategy.CODE_WORKAROUND]
    if workaround:
        target = workaround[0]
        eval_ = qa_evaluations.get(target.task_id)
        feedback: Dict[str, str] = {}
        if eval_ and eval_.retry_feedback:
            feedback[target.task_id] = eval_.retry_feedback
        return SupervisorDecision(
            next_node="workaround_subagent",
            target_task_ids=[target.task_id],
            feedback_by_task=feedback,
            instructions="Apply the minimal safe code workaround for this vulnerability.",
            decision_reason=(
                f"Routing task '{target.task_id}' to workaround_subagent."
            ),
        )

    # Unexpected: no workable tasks found → teardown as safe default
    return SupervisorDecision(
        next_node="teardown",
        target_task_ids=[],
        instructions="No actionable tasks remain.",
        decision_reason=(
            "Deterministic fallback: no workable tasks found, routing to teardown."
        ),
    )


# ---------------------------------------------------------------------------
# Planner phase helpers
# ---------------------------------------------------------------------------


def _needs_planner(
    task_queue: Dict[str, RemediationTask],
    qa_evaluations: Dict[str, QAEvaluation],
    retry_diagnostics_by_task: Dict[str, UpdateRetryDiagnostics],
    current_status: str,
) -> bool:
    """Return True when the planner should be invoked.

    The planner is reserved for retry analysis and playbook selection.
    """
    if not any(task.status == TaskStatus.NEEDS_RETRY for task in task_queue.values()):
        return False
    return current_status == "qa_completed" or bool(qa_evaluations) or bool(
        retry_diagnostics_by_task
    )


def _build_planner_prompt(
    task_queue: Dict[str, RemediationTask],
    group_by_id: Dict[str, VulnerabilityGroup],
    qa_evaluations: Dict[str, QAEvaluation],
    retry_diagnostics_by_task: Dict[str, UpdateRetryDiagnostics],
    action_summaries: List[AgentActionSummary],
    constraints_ledger: List[str],
) -> str:
    """Build the planner system + user messages."""
    lines = [
        "You are the Planner phase of an AppSec remediation Supervisor.",
        "You own playbook reasoning for retry tasks.",
        "Your job is to investigate failures, assess whether manifest-based update remediation is exhausted,",
        "and write high-level retry guidance for the Router to enforce.",
        "",
        "The Update Subagent gathers retry evidence by querying npm and attempting bounded manifest changes.",
        "Do not decide exact dependency versions for the worker unless you are only verifying ambiguous evidence.",
        "Use view_npm_package_versions only when retry diagnostics are missing, inconsistent, or ambiguous.",
        "The default source of truth is the existing retry diagnostics, not a second full registry search.",
        "",
        "## Task Queue",
    ]

    for task in task_queue.values():
        group = group_by_id.get(task.parent_group_id)
        cves = ", ".join(group.cve_ids) if group and group.cve_ids else "none"
        ghsas = ", ".join(group.ghsa_ids) if group and group.ghsa_ids else "none"
        component = group.vulnerable_component if group else task.parent_group_id
        eval_ = qa_evaluations.get(task.task_id)
        diagnostics = retry_diagnostics_by_task.get(task.task_id)

        lines += [
            "",
            f"### Task: {task.task_id}",
            f"- Component     : {component}",
            f"- CVEs          : {cves}",
            f"- GHSAs         : {ghsas}",
            f"- Strategy      : {task.strategy.value}",
            f"- Status        : {task.status.value}",
            f"- Retries Used  : {task.retry_count}/{MAX_RETRIES}",
            f"- Ancestry Depth: {task.ancestry_depth}/{MAX_ANCESTRY_DEPTH}",
            f"- Current Instr : {task.instruction or '(none)'}",
        ]
        if eval_ and task.status == TaskStatus.NEEDS_RETRY:
            cat = eval_.failure_category.value if eval_.failure_category else "none"
            lines += [
                f"- Last QA Failed: category={cat}",
                f"  Feedback: {eval_.retry_feedback}",
            ]
        diagnostics = retry_diagnostics_by_task.get(task.task_id)
        if diagnostics is not None:
            attempted = ", ".join(diagnostics.attempted_versions) or "none"
            candidates = ", ".join(diagnostics.candidate_versions_considered[:10]) or "none"
            lines += [
                f"- Retry Diags  : registry={diagnostics.registry_query_performed}, exhausted={diagnostics.exhausted_update_path}, abandoned={diagnostics.package_abandoned}",
                f"  Attempted: {attempted}",
                f"  Candidates: {candidates}",
                f"  Latest Seen: {diagnostics.latest_version_seen or 'unknown'}",
                f"  Failure Reason: {diagnostics.failure_reason or 'none'}",
            ]

    lines += [
        "",
        "## Constraints Ledger (must not violate these)",
    ]
    if constraints_ledger:
        lines.extend(f"- {c}" for c in constraints_ledger)
    else:
        lines.append("- (none)")

    lines += [
        "",
        "## Recent Action Summaries",
    ]
    for summary in action_summaries[-6:]:
        lines.append(f"- [{summary.task_id}] {summary.status.value}: {summary.summary}")
    if not action_summaries:
        lines.append("- (none)")

    lines += [
        "",
        "## Planner Playbooks",
        "- SECURITY_FLAG: keep the task on update remediation unless retry diagnostics show the update path is exhausted.",
        "- PEER_CONFLICT: keep the task on update remediation unless retry diagnostics show no compatible patch or override path remains.",
        "- BREAKING_CHANGE: pivot from VERSION_BUMP to a CODE_WORKAROUND child task.",
        "- package_abandoned=True: pivot from VERSION_BUMP to a CODE_WORKAROUND child task.",
        "- exhausted_update_path=True: pivot from VERSION_BUMP to a CODE_WORKAROUND child task.",
        "- A strategy pivot must be expressed as a child-task recommendation, not as an in-place worker retry.",
        "",
        "## Queue Caps",
        f"- MAX_TASK_QUEUE_SIZE = {MAX_TASK_QUEUE_SIZE} (current: {len(task_queue)})",
        f"- MAX_ANCESTRY_DEPTH = {MAX_ANCESTRY_DEPTH}",
        "",
        "## Output Format",
        "Write a 'Strategy Scratchpad' with these sections for each task needing attention:",
        "  1. Observations",
        "  2. Playbook selected",
        "  3. Update-path assessment",
        "  4. Revised high-level instruction text",
        "  5. Strategy pivot recommendation (same-task retry or workaround child)",
        "  6. Routing notes",
        "",
        "Use high-level retry instructions, not exact version pins.",
        "Good example: Investigate compatible patched releases or override paths for ws; if the update path is exhausted, recommend a workaround child task.",
    ]

    return "\n".join(lines)


def _run_planner_phase(
    task_queue: Dict[str, RemediationTask],
    group_by_id: Dict[str, VulnerabilityGroup],
    qa_evaluations: Dict[str, QAEvaluation],
    retry_diagnostics_by_task: Dict[str, UpdateRetryDiagnostics],
    action_summaries: List[AgentActionSummary],
    constraints_ledger: List[str],
    llm: Any,
) -> str:
    """Run the bounded planner ReAct loop. Returns the scratchpad text."""
    planner_prompt = _build_planner_prompt(
        task_queue,
        group_by_id,
        qa_evaluations,
        retry_diagnostics_by_task,
        action_summaries,
        constraints_ledger,
    )
    tools = [view_npm_package_versions]
    initial_messages = [
        SystemMessage(content=planner_prompt),
        HumanMessage(content="Please analyse the task queue and write your Strategy Scratchpad."),
    ]
    try:
        result = run_bounded_subagent_loop(
            llm=llm,
            tools=tools,
            initial_messages=initial_messages,
            touched_files=set(),
        )
        scratchpad = result.final_text.strip()
        if scratchpad.startswith("<MagicMock"):
            scratchpad = ""
        if result.errors:
            logger.warning("supervisor planner: %d error(s): %s", len(result.errors), result.errors)
        return scratchpad or "(Planner produced no output.)"
    except Exception as exc:  # noqa: BLE001
        logger.warning("supervisor planner: loop failed (%s) — skipping planner.", exc)
        return f"(Planner failed: {exc})"


# ---------------------------------------------------------------------------
# Router prompt builder
# ---------------------------------------------------------------------------


def build_supervisor_prompt(state: OrchestratorState, scratchpad: str = "") -> str:
    """Build the structured Router LLM prompt for the Supervisor decision."""
    valid_groups: List[VulnerabilityGroup] = state.get("valid_groups", [])
    task_queue: Dict[str, RemediationTask] = state.get("task_queue", {})
    constraints_ledger: List[str] = state.get("constraints_ledger", [])
    action_summaries: List[AgentActionSummary] = state.get("action_summaries", [])
    qa_evaluations: Dict[str, QAEvaluation] = state.get("qa_evaluations", {})
    retry_diagnostics_by_task: Dict[str, UpdateRetryDiagnostics] = state.get(
        "retry_diagnostics_by_task", {}
    )
    eval_status: str = state.get("eval_status", "")

    group_by_id = {g.group_id: g for g in valid_groups}

    lines = [
        "You are the Router phase of an AppSec remediation Supervisor.",
        "Produce exactly one SupervisorDecision to route the next graph step.",
        "The Planner owns playbook and strategy reasoning.",
        "You only translate planner intent and current task state into routing.",
        "Do not invent dependency-version strategy beyond what the planner scratchpad and retry diagnostics support.",
        "",
    ]

    lines.append("## Remediation Tasks")

    for task in task_queue.values():
        group = group_by_id.get(task.parent_group_id)
        fix_plan = group.fix_plan if group else None
        cves = ", ".join(group.cve_ids) if group and group.cve_ids else "none"
        ghsas = ", ".join(group.ghsa_ids) if group and group.ghsa_ids else "none"
        eval_ = qa_evaluations.get(task.task_id)
        diagnostics = retry_diagnostics_by_task.get(task.task_id)

        lines += [
            "",
            f"### Task: {task.task_id}",
            f"- Parent Group  : {task.parent_group_id}",
            f"- Component     : {group.vulnerable_component if group else 'unknown'}",
            f"- Issue Type    : {group.issue_type.value if group else 'unknown'}",
            f"- CVEs          : {cves}",
            f"- GHSAs         : {ghsas}",
            f"- Fix Plan      : {fix_plan.status.value if fix_plan else 'none'}",
            f"- Strategy      : {task.strategy.value}",
            f"- Status        : {task.status.value}",
            f"- Retries Used  : {task.retry_count}/{MAX_RETRIES}",
            f"- Ancestry Depth: {task.ancestry_depth}/{MAX_ANCESTRY_DEPTH}",
            f"- Instruction   : {task.instruction or '(none)'}",
        ]
        if task.parent_task_id:
            lines.append(f"- Parent Task   : {task.parent_task_id}")
        if eval_ and task.status != TaskStatus.OPTIMISTICALLY_FIXED:
            cat = eval_.failure_category.value if eval_.failure_category else "none"
            lines.append(
                f"- Last QA       : passed={eval_.passed}, category={cat}, "
                f"feedback={eval_.retry_feedback}"
            )
        if diagnostics is not None:
            lines.append(
                f"- Retry Diags   : registry={diagnostics.registry_query_performed}, "
                f"latest={diagnostics.latest_version_seen or 'unknown'}, "
                f"exhausted={diagnostics.exhausted_update_path}, "
                f"abandoned={diagnostics.package_abandoned}"
            )
            if diagnostics.planning_answers:
                lines.append(
                    "- Planning Answers: "
                    + "; ".join(
                        f"{key}={value}" for key, value in diagnostics.planning_answers.items()
                    )
                )

    lines += [
        "",
        "## Constraints Ledger",
    ]
    if constraints_ledger:
        lines.extend(f"- {c}" for c in constraints_ledger)
    else:
        lines.append("- (none)")

    lines += [
        "",
        "## Recent Action Summaries",
    ]
    for summary in action_summaries[-10:]:
        lines.append(
            f"- [{summary.task_id}] {summary.status.value}: {summary.summary}"
        )
    if not action_summaries:
        lines.append("- (none)")

    lines += [
        "",
        f"## QA Evaluation Status: {eval_status or 'none'}",
        "",
        "## Planner Scratchpad",
        scratchpad or "(none)",
        "",
        f"## Queue Caps: {len(task_queue)}/{MAX_TASK_QUEUE_SIZE} tasks used, depth cap = {MAX_ANCESTRY_DEPTH}",
        "",
        "## Router Rules (follow strictly)",
        f"1. Send pending VERSION_BUMP tasks to update_subagent in batches of at most {UPDATE_BATCH_SIZE}.",
        f"2. Send retry VERSION_BUMP tasks to update_subagent in retry-only batches of at most {UPDATE_BATCH_SIZE}.",
        "3. Every retry task routed to update_subagent MUST have a non-empty revised_instructions entry.",
        "4. Retry revised_instructions are high-level directives, not exact version pins.",
        "5. Same-strategy retries reuse the same task.",
        "6. Any strategy pivot must spawn a child task; do not mutate the parent task's strategy in place.",
        "7. SECURITY_FLAG and PEER_CONFLICT remain update remediation first unless the planner indicates the update path is exhausted.",
        "8. BREAKING_CHANGE follow-on remediation must use CODE_WORKAROUND strategy.",
        "9. Send exactly one pending or retry CODE_WORKAROUND task to workaround_subagent.",
        "10. After a worker succeeds for the current active batch, route that batch to qa_critic.",
        "11. When no actionable non-terminal tasks remain, route to teardown.",
        f"12. Any task with {MAX_RETRIES}+ retries may be marked unfixable.",
        "13. unfixable, qa_passed, and optimistically_fixed tasks must not appear in worker target_task_ids.",
        "14. task_status_updates may only set QA_PASSED or UNFIXABLE.",
        f"15. update_subagent MUST have between 1 and {UPDATE_BATCH_SIZE} target_task_ids.",
        "16. workaround_subagent MUST have exactly one target_task_id.",
        "17. instructions is audit/routing rationale only; do not use it as a substitute for revised_instructions.",
        f"18. spawn_requests must respect parent depth < {MAX_ANCESTRY_DEPTH} and queue size <= {MAX_TASK_QUEUE_SIZE}.",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Spawn request materializer
# ---------------------------------------------------------------------------


def _materialize_spawn_requests(
    spawn_requests: List[TaskSpawnRequest],
    task_queue: Dict[str, RemediationTask],
    group_by_id: Dict[str, VulnerabilityGroup],
    errors: List[str],
) -> Tuple[Dict[str, RemediationTask], Dict[str, List[str]]]:
    """Validate and materialize spawn requests into new RemediationTask objects.

    Returns a dict of new task_id → RemediationTask to be merged into task_queue.
    Rejected requests are logged to errors.
    """
    next_index = len(task_queue) + 1
    new_tasks: Dict[str, RemediationTask] = {}
    child_ids_by_parent: Dict[str, List[str]] = {}
    current_queue_size = len(task_queue)

    for req in spawn_requests:
        # Guard: unknown parent task
        if req.parent_task_id not in task_queue:
            errors.append(
                f"supervisor: spawn rejected — parent task '{req.parent_task_id}' not in queue."
            )
            continue

        parent_task = task_queue[req.parent_task_id]

        # Guard: depth cap
        child_depth = parent_task.ancestry_depth + 1
        if child_depth > MAX_ANCESTRY_DEPTH:
            errors.append(
                f"supervisor: spawn rejected — parent '{req.parent_task_id}' at depth "
                f"{parent_task.ancestry_depth}, child would be depth {child_depth} "
                f"which exceeds MAX_ANCESTRY_DEPTH={MAX_ANCESTRY_DEPTH}."
            )
            continue

        # Guard: queue size cap
        if current_queue_size + len(new_tasks) + 1 > MAX_TASK_QUEUE_SIZE:
            errors.append(
                f"supervisor: spawn rejected — queue would exceed MAX_TASK_QUEUE_SIZE="
                f"{MAX_TASK_QUEUE_SIZE}. Rejected spawn for parent '{req.parent_task_id}'."
            )
            continue

        # Materialize child task
        child_task_id = f"task-{next_index}"
        next_index += 1

        new_task = RemediationTask(
            task_id=child_task_id,
            parent_group_id=parent_task.parent_group_id,
            parent_task_id=req.parent_task_id,
            strategy=req.strategy,
            instruction=req.instruction,
            status=TaskStatus.PENDING,
            retry_count=0,
            ancestry_depth=child_depth,
        )
        new_tasks[child_task_id] = new_task
        child_ids_by_parent.setdefault(req.parent_task_id, []).append(child_task_id)
        logger.info(
            "supervisor: spawned child task '%s' (parent='%s', depth=%d, strategy=%s) — %s",
            child_task_id,
            req.parent_task_id,
            child_depth,
            req.strategy.value,
            req.reason,
        )

    return new_tasks, child_ids_by_parent


# ---------------------------------------------------------------------------
# Supervisor node
# ---------------------------------------------------------------------------


def run_supervisor_node(state: OrchestratorState) -> Dict[str, Any]:
    """
    LangGraph node — Supervisor commander for Phase 5 orchestration.

    Execution stages
    ----------------
    1. Normalize task_queue: create initial RemediationTask entries for any
       valid_groups not yet represented (copy-on-write via model_copy).
    2. Ingest subagent action summaries for current active_target_task_ids only.
    3. Ingest QA results for active task IDs only (when status == "qa_completed").
    4. Mark UNFIXABLE any task whose retry_count has reached MAX_RETRIES.
    5. Short-circuit: if active batch is all optimistically_fixed → qa_critic.
    6. Router phase: ChatOpenAI.with_structured_output(SupervisorDecision).
    7. Guardrails: reject unknown IDs, enforce cardinality, fall back to
       deterministic routing if invalid.
    8. Apply guarded: revised_instructions, strategy updates, status overrides,
       unfixable marks, new constraints, and materialized spawn requests.
    9. Return state patch.
    """
    valid_groups: List[VulnerabilityGroup] = list(state.get("valid_groups", []))
    if not valid_groups:
        logger.info("supervisor: no valid groups — routing to teardown.")
        return {
            "status": "supervisor_routed",
            "next_routing_step": "teardown",
            "active_target_task_ids": [],
            "supervisor_instructions": "No groups to process.",
        }

    group_by_id: Dict[str, VulnerabilityGroup] = {g.group_id: g for g in valid_groups}
    existing_constraints: List[str] = list(state.get("constraints_ledger", []))
    retry_diagnostics_by_task: Dict[str, UpdateRetryDiagnostics] = dict(
        state.get("retry_diagnostics_by_task", {})
    )
    errors: List[str] = list(state.get("errors") or [])

    # ------------------------------------------------------------------
    # 1. Normalize task_queue (copy-on-write)
    # ------------------------------------------------------------------
    raw_task_queue: Dict[str, RemediationTask] = dict(state.get("task_queue", {}))
    # Copy-on-write: work with model copies so we never mutate state-owned objects
    task_queue: Dict[str, RemediationTask] = {
        tid: t.model_copy() for tid, t in raw_task_queue.items()
    }
    existing_group_ids = {t.parent_group_id for t in task_queue.values()}
    next_task_index = len(task_queue) + 1
    for group in valid_groups:
        if group.group_id not in existing_group_ids:
            task_id = f"task-{next_task_index}"
            task_queue[task_id] = build_initial_remediation_task(group, task_id)
            next_task_index += 1

    # ------------------------------------------------------------------
    # 2. Ingest subagent action summaries (active targets only)
    # ------------------------------------------------------------------
    active_target_task_ids = list(state.get("active_target_task_ids") or [])
    active_targets = set(active_target_task_ids)
    action_summaries: List[AgentActionSummary] = state.get("action_summaries") or []

    if active_targets and action_summaries:
        recent_summaries = {}
        for summary in action_summaries:
            resolved_t_id = _resolve_task_id_from_identifier(
                summary.task_id,
                task_queue,
                active_target_task_ids,
            )
            if resolved_t_id is None:
                continue
            if resolved_t_id in active_targets:
                recent_summaries[resolved_t_id] = summary

        for resolved_t_id, summary in recent_summaries.items():
            task = task_queue[resolved_t_id]
            if task.status in (TaskStatus.QA_PASSED, TaskStatus.UNFIXABLE):
                continue
            if summary.status == AgentActionStatus.SUCCESS:
                task.status = TaskStatus.OPTIMISTICALLY_FIXED
            else:
                task.status = TaskStatus.NEEDS_RETRY
                task.retry_count += 1

    # ------------------------------------------------------------------
    # 3. Ingest QA results (active targets only, when qa_completed)
    # ------------------------------------------------------------------
    qa_evaluations: Dict[str, QAEvaluation] = _normalize_qa_evaluations_for_tasks(
        dict(state.get("qa_evaluations", {})),
        task_queue,
        active_target_task_ids,
    )
    auto_new_constraints: List[str] = []

    if state.get("status") == "qa_completed":
        for resolved_t_id, evaluation in qa_evaluations.items():
            task = task_queue[resolved_t_id]
            if task.status in (TaskStatus.UNFIXABLE, TaskStatus.QA_PASSED):
                continue
            if evaluation.passed:
                task.status = TaskStatus.QA_PASSED
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
                task.status = TaskStatus.NEEDS_RETRY
                task.retry_count += 1

    # ------------------------------------------------------------------
    # 4. Mark UNFIXABLE tasks that hit the retry cap
    # ------------------------------------------------------------------
    for task_id, task in task_queue.items():
        if (
            task.status == TaskStatus.NEEDS_RETRY
            and task.retry_count >= MAX_RETRIES
        ):
            task.status = TaskStatus.UNFIXABLE
            logger.info(
                "supervisor: task '%s' marked UNFIXABLE after %d retries.",
                task_id,
                task.retry_count,
            )

    # ------------------------------------------------------------------
    # 5. Short-circuit: if active batch is all optimistically_fixed → qa_critic
    # ------------------------------------------------------------------
    decision: Optional[SupervisorDecision] = None
    if state.get("status") != "qa_completed" and active_target_task_ids:
        active_batch = [task_queue[t] for t in active_target_task_ids if t in task_queue]
        if active_batch and all(
            t.status == TaskStatus.OPTIMISTICALLY_FIXED for t in active_batch
        ):
            decision = SupervisorDecision(
                next_node="qa_critic",
                target_task_ids=[t.task_id for t in active_batch],
                instructions="Run QA on the current remediated batch before starting more remediation.",
                decision_reason=(
                    f"Routing the current batch of {len(active_batch)} optimistically fixed task(s) to QA."
                ),
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
        elif all(t.status == TaskStatus.OPTIMISTICALLY_FIXED for t in non_terminal):
            decision = SupervisorDecision(
                next_node="qa_critic",
                target_task_ids=[t.task_id for t in non_terminal],
                instructions="Run QA on the remaining optimistically fixed tasks.",
                decision_reason="Routing all remaining optimistically fixed tasks to QA.",
            )

    # ------------------------------------------------------------------
    # 6. Router phase (structured LLM call)
    # ------------------------------------------------------------------
    planner_scratchpad = ""
    if decision is None:
        try:
            from langchain_openai import ChatOpenAI  # type: ignore[import]

            model_name = os.environ.get("REMEDY_LLM_MODEL", _DEFAULT_MODEL)
            router_llm = ChatOpenAI(model=model_name, temperature=0)
            if _needs_planner(
                task_queue,
                qa_evaluations,
                retry_diagnostics_by_task,
                str(state.get("status") or ""),
            ):
                logger.info("supervisor: invoking planner phase for retry analysis.")
                planner_scratchpad = _run_planner_phase(
                    task_queue,
                    group_by_id,
                    qa_evaluations,
                    retry_diagnostics_by_task,
                    action_summaries,
                    existing_constraints,
                    router_llm,
                )
            structured_llm = router_llm.with_structured_output(
                SupervisorDecision, method="function_calling"
            )
            prompt_state: OrchestratorState = {  # type: ignore[typeddict-item]
                **state,
                "task_queue": task_queue,
                "qa_evaluations": qa_evaluations,
                "retry_diagnostics_by_task": retry_diagnostics_by_task,
            }
            prompt_text = build_supervisor_prompt(
                prompt_state,
                scratchpad=planner_scratchpad,
            )
            logger.info("supervisor: invoking structured router LLM.")
            decision = structured_llm.invoke(prompt_text)
            logger.info(
                "supervisor: router decision → next_node=%s targets=%s reason=%s",
                decision.next_node,
                decision.target_task_ids,
                decision.decision_reason,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "supervisor: LLM call failed (%s) — using deterministic fallback.", exc
            )

    # Re-apply optimistic short-circuit post-LLM (guard against LLM overriding)
    if state.get("status") != "qa_completed" and active_target_task_ids:
        active_batch = [task_queue[t] for t in active_target_task_ids if t in task_queue]
        if active_batch and all(
            t.status == TaskStatus.OPTIMISTICALLY_FIXED for t in active_batch
        ):
            decision = SupervisorDecision(
                next_node="qa_critic",
                target_task_ids=[t.task_id for t in active_batch],
                instructions="Run QA on the current remediated batch before starting more remediation.",
                decision_reason=(
                    f"Routing the current batch of {len(active_batch)} optimistically fixed task(s) to QA."
                ),
            )

    # ------------------------------------------------------------------
    # 7. Guardrails: validate and clamp (or deterministic fallback)
    # ------------------------------------------------------------------
    pivot_spawn_requests: List[TaskSpawnRequest] = []
    pivot_parent_status_by_parent: Dict[str, TaskStatus] = {}
    pivot_target_parent_ids: Set[str] = set()

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
        logger.info(
            "supervisor: deterministic fallback → next_node=%s", decision.next_node
        )
    else:
        known_task_ids = set(task_queue.keys())

        # Remove unknown, terminal, or optimistically_fixed task IDs from worker targets
        valid_target_ids = []
        for t_id in decision.target_task_ids:
            if t_id not in known_task_ids:
                continue
            t_status = task_queue[t_id].status
            if t_status in _TERMINAL_STATUSES:
                continue
            if decision.next_node in ("update_subagent", "workaround_subagent") and t_status == TaskStatus.OPTIMISTICALLY_FIXED:
                continue
            valid_target_ids.append(t_id)

        valid_unfixable_ids = [
            t_id for t_id in decision.unfixable_task_ids if t_id in known_task_ids
        ]

        # Enforce cardinality constraints
        needs_fallback = False
        if decision.next_node == "workaround_subagent" and len(valid_target_ids) != 1:
            logger.warning(
                "supervisor: workaround_subagent needs 1 target, got %d — falling back.",
                len(valid_target_ids),
            )
            needs_fallback = True
        elif decision.next_node == "update_subagent" and not valid_target_ids:
            logger.warning(
                "supervisor: update_subagent needs ≥1 target, got 0 — falling back."
            )
            needs_fallback = True

        if not needs_fallback and decision.next_node == "update_subagent" and len(valid_target_ids) > UPDATE_BATCH_SIZE:
            logger.warning(
                "supervisor: update_subagent supports at most %d targets, got %d — falling back.",
                UPDATE_BATCH_SIZE,
                len(valid_target_ids),
            )
            needs_fallback = True
        if (
            not needs_fallback
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
                    "supervisor: update_subagent batch mixed first-pass and retry tasks — falling back."
                )
                needs_fallback = True
        if not needs_fallback and decision.next_node == "qa_critic" and not valid_target_ids:
            logger.warning(
                "supervisor: qa_critic needs at least 1 target, got 0 — falling back."
            )
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
        else:
            # Filter revised_instructions and feedback_by_task to known task IDs
            clean_revised_instructions = {
                k: v
                for k, v in decision.revised_instructions.items()
                if k in known_task_ids and v.strip()
            }
            clean_feedback = {
                k: v
                for k, v in decision.feedback_by_task.items()
                if k in known_task_ids
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
                    f"revised_instructions for {missing_retry_revisions}."
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

            # Validate task_status_updates — only known tasks, only terminal statuses
            clean_status_updates: Dict[str, TaskStatus] = {}
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

            pivot_strategy_updates = {
                task_id: new_strategy
                for task_id, new_strategy in clean_updated_task_strategies.items()
                if task_queue[task_id].strategy != new_strategy
            }
            targeted_pivot_ids = [
                task_id for task_id in valid_target_ids if task_id in pivot_strategy_updates
            ]
            incompatible_targeted_pivots = [
                task_id
                for task_id in targeted_pivot_ids
                if decision.next_node != _worker_node_for_strategy(
                    pivot_strategy_updates[task_id]
                )
            ]
            missing_pivot_instructions = [
                task_id
                for task_id in targeted_pivot_ids
                if not clean_revised_instructions.get(task_id, "").strip()
            ]
            if incompatible_targeted_pivots:
                errors.append(
                    "supervisor: rejected strategy pivot because next_node does not match "
                    f"the child strategy for {incompatible_targeted_pivots}."
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
                pivot_strategy_updates = {}
                targeted_pivot_ids = []
            if not incompatible_targeted_pivots and missing_pivot_instructions:
                errors.append(
                    "supervisor: rejected strategy pivot without task-specific child "
                    f"instructions for {missing_pivot_instructions}."
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
                pivot_strategy_updates = {}
                targeted_pivot_ids = []

            for task_id, new_strategy in pivot_strategy_updates.items():
                child_instruction = clean_revised_instructions.pop(task_id, "").strip()
                if not child_instruction:
                    continue
                pivot_spawn_requests.append(
                    TaskSpawnRequest(
                        parent_task_id=task_id,
                        strategy=new_strategy,
                        instruction=child_instruction,
                        reason=(
                            "Auto-spawned due to strategy pivot from "
                            f"{task_queue[task_id].strategy.value} to {new_strategy.value}. "
                            f"{decision.decision_reason}"
                        ),
                    )
                )
                pivot_parent_status_by_parent[task_id] = _parent_status_for_strategy_pivot(
                    task_queue[task_id],
                    new_strategy,
                    qa_evaluations,
                )
            pivot_target_parent_ids = set(targeted_pivot_ids)

            try:
                decision = SupervisorDecision(
                    next_node=decision.next_node,
                    updated_task_strategies={},
                    target_task_ids=valid_target_ids,
                    unfixable_task_ids=valid_unfixable_ids,
                    new_constraints=decision.new_constraints,
                    feedback_by_task=clean_feedback,
                    revised_instructions=clean_revised_instructions,
                    spawn_requests=[*pivot_spawn_requests, *decision.spawn_requests],
                    task_status_updates=clean_status_updates,
                    instructions=decision.instructions,
                    decision_reason=decision.decision_reason,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "supervisor: decision rebuild failed (%s) — falling back.", exc
                )
                pivot_spawn_requests = []
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

    # 8a. Apply revised_instructions (copy-on-write per task)
    for t_id, new_instr in decision.revised_instructions.items():
        if t_id in task_queue and new_instr.strip():
            task_queue[t_id] = task_queue[t_id].model_copy(update={"instruction": new_instr})

    # 8b. Apply direct strategy pivots (currently reserved for no-op / legacy cases)
    for t_id, new_strategy in decision.updated_task_strategies.items():
        if t_id in task_queue:
            task_queue[t_id] = task_queue[t_id].model_copy(update={"strategy": new_strategy})

    # 8c. Apply guarded task status overrides (only QA_PASSED and UNFIXABLE)
    _allowed_statuses = {TaskStatus.QA_PASSED, TaskStatus.UNFIXABLE}
    for t_id, new_status in decision.task_status_updates.items():
        if t_id in task_queue and new_status in _allowed_statuses:
            if task_queue[t_id].status not in _TERMINAL_STATUSES:
                task_queue[t_id] = task_queue[t_id].model_copy(update={"status": new_status})
                logger.info(
                    "supervisor: task '%s' manually set to %s via task_status_updates.",
                    t_id,
                    new_status.value,
                )

    # 8d. Apply unfixable marks from decision
    for t_id in decision.unfixable_task_ids:
        if t_id in task_queue:
            task_queue[t_id] = task_queue[t_id].model_copy(update={"status": TaskStatus.UNFIXABLE})

    # 8e. Materialize spawn requests
    child_ids_by_parent: Dict[str, List[str]] = {}
    if decision.spawn_requests:
        new_tasks, child_ids_by_parent = _materialize_spawn_requests(
            spawn_requests=list(decision.spawn_requests),
            task_queue=task_queue,
            group_by_id=group_by_id,
            errors=errors,
        )
        task_queue.update(new_tasks)
        for parent_task_id, child_ids in child_ids_by_parent.items():
            if (
                child_ids
                and parent_task_id in pivot_parent_status_by_parent
                and parent_task_id in task_queue
                and task_queue[parent_task_id].status not in _TERMINAL_STATUSES
            ):
                task_queue[parent_task_id] = task_queue[parent_task_id].model_copy(
                    update={"status": pivot_parent_status_by_parent[parent_task_id]}
                )

    resolved_target_task_ids: List[str] = []
    remapped_feedback_by_task: Dict[str, str] = {}
    for task_id in decision.target_task_ids:
        if task_id in pivot_target_parent_ids:
            child_ids = child_ids_by_parent.get(task_id, [])
            if child_ids:
                child_task_id = child_ids[0]
                resolved_target_task_ids.append(child_task_id)
                if task_id in decision.feedback_by_task:
                    remapped_feedback_by_task[child_task_id] = decision.feedback_by_task[task_id]
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
    if resolved_next_node in {"update_subagent", "workaround_subagent", "qa_critic"} and not resolved_target_task_ids:
        errors.append(
            "supervisor: routing fell back to teardown because no dispatchable target tasks remained."
        )
        resolved_next_node = "teardown"

    logger.info(
        "supervisor: routing to '%s' with targets=%s",
        resolved_next_node,
        resolved_target_task_ids,
    )

    # ------------------------------------------------------------------
    # 8f. Collect new constraints
    # ------------------------------------------------------------------
    returned_constraints: List[str] = list(auto_new_constraints)
    for constraint in decision.new_constraints:
        if (
            constraint
            and constraint not in existing_constraints
            and constraint not in returned_constraints
        ):
            returned_constraints.append(constraint)

    # Build feedback_by_group for bridge node backward compat
    feedback_by_task = remapped_feedback_by_task
    feedback_by_group: Dict[str, str] = {}
    for t_id, fb in feedback_by_task.items():
        if t_id in task_queue:
            gid = task_queue[t_id].parent_group_id
            feedback_by_group[gid] = fb

    # ------------------------------------------------------------------
    # 9. Return state patch
    # ------------------------------------------------------------------
    return {
        "status": "supervisor_routed",
        "next_routing_step": resolved_next_node,
        "active_target_task_ids": resolved_target_task_ids,
        # Keep active_target_group_ids populated for bridge nodes that still use it
        "active_target_group_ids": [
            task_queue[t].parent_group_id
            for t in resolved_target_task_ids
            if t in task_queue
        ],
        "feedback_by_task": feedback_by_task,
        "feedback_by_group": feedback_by_group,
        "supervisor_instructions": decision.instructions,
        "task_queue": task_queue,
        # constraints_ledger uses operator.add — return only NEW entries
        "constraints_ledger": returned_constraints,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# Conditional-edge router
# ---------------------------------------------------------------------------


def supervisor_router(state: OrchestratorState) -> str:
    """
    Conditional-edge callable for the supervisor node.

    Reads ``state["next_routing_step"]`` and returns the target node name.
    Defaults to ``"teardown"`` for any unknown or missing value.
    """
    step = state.get("next_routing_step", "")
    if step in _VALID_NEXT_NODES:
        return step
    logger.warning(
        "supervisor_router: unknown next_routing_step '%s' — defaulting to teardown.",
        step,
    )
    return "teardown"
