"""
supervisor_node.py - Agentic Supervisor Node for Phase 5 hub-and-spoke orchestration.

The Supervisor runs exactly once per visit: no tools, no ReAct loop.
It performs Python-first state normalization, then makes a single structured
LLM call to obtain a ``SupervisorDecision``, validates and clamps that
decision, and returns a state patch that wires the next hop.

Public API
----------
MAX_RETRIES : int
    Maximum number of QA-fail-retry cycles before a task is marked unfixable.
build_supervisor_prompt(state) -> str
    Builds the structured prompt text for the LLM decision.
run_supervisor_node(state) -> Dict[str, Any]
    LangGraph node callable.
supervisor_router(state) -> str
    Conditional-edge callable: reads ``next_routing_step`` from state.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Set

from src.contracts.schemas import (
    AgentActionStatus,
    AgentActionSummary,
    FailureCategory,
    QAEvaluation,
    RemediationTask,
    RoutingStrategy,
    SupervisorDecision,
    TaskStatus,
    VulnerabilityGroup,
)
from src.orchestrator.state import OrchestratorState
from src.orchestrator.task_utils import build_initial_remediation_task

logger = logging.getLogger(__name__)

MAX_RETRIES: int = 1 # Keep at 1 for testing
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


# ---------------------------------------------------------------------------
# Deterministic fallback router
# ---------------------------------------------------------------------------


def _deterministic_routing(
    task_queue: Dict[str, RemediationTask],
    group_by_id: Dict[str, VulnerabilityGroup],
    qa_evaluations: Dict[str, QAEvaluation],
    active_target_task_ids: Optional[List[str]] = None,
    current_status: str = "",
) -> SupervisorDecision:
    """
    Pure-Python fallback routing used when the LLM call fails.

    Implements the same priority rules described in the supervisor prompt.
    """
    tasks = list(task_queue.values())
    non_terminal = [t for t in tasks if t.status not in _TERMINAL_STATUSES]

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

    # VERSION_BUMP tasks batch to update_subagent
    version_bump = [t for t in workable if t.strategy == RoutingStrategy.VERSION_BUMP]
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
# Prompt builder
# ---------------------------------------------------------------------------


def build_supervisor_prompt(state: OrchestratorState) -> str:
    """Build the structured LLM prompt for the Supervisor decision."""
    valid_groups: List[VulnerabilityGroup] = state.get("valid_groups", [])
    task_queue: Dict[str, RemediationTask] = state.get("task_queue", {})
    constraints_ledger: List[str] = state.get("constraints_ledger", [])
    action_summaries: List[AgentActionSummary] = state.get("action_summaries", [])
    qa_evaluations: Dict[str, QAEvaluation] = state.get("qa_evaluations", {})
    eval_status: str = state.get("eval_status", "")

    group_by_id = {g.group_id: g for g in valid_groups}

    lines = [
        "You are the Supervisor Agent of an AppSec remediation pipeline.",
        "Produce a single SupervisorDecision to route the next graph step.",
        "",
        "## Remediation Tasks",
    ]

    for task in task_queue.values():
        group = group_by_id.get(task.parent_group_id)
        fix_plan = group.fix_plan if group else None
        cves = ", ".join(group.cve_ids) if group and group.cve_ids else "none"
        ghsas = ", ".join(group.ghsa_ids) if group and group.ghsa_ids else "none"
        eval_ = qa_evaluations.get(task.task_id)

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
        ]
        if eval_ and task.status != TaskStatus.OPTIMISTICALLY_FIXED:
            cat = eval_.failure_category.value if eval_.failure_category else "none"
            lines.append(
                f"- Last QA       : passed={eval_.passed}, category={cat}, "
                f"feedback={eval_.retry_feedback}"
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
        "## Routing Rules (follow strictly)",
        f"1. Send pending/needs_retry VERSION_BUMP tasks to update_subagent in batches of at most {UPDATE_BATCH_SIZE}. Never send more than {UPDATE_BATCH_SIZE} target_task_ids to update_subagent in one decision.",
        "2. Send EXACTLY ONE pending/needs_retry CODE_WORKAROUND task → workaround_subagent.",
        "3. After a subagent succeeds for the current active batch, route that exact optimistically_fixed batch to qa_critic before starting another remediation batch. Do NOT route optimistically_fixed tasks back to subagents.",
        "4. When ALL non-terminal tasks are qa_passed OR all are unfixable → teardown.",
        "5. If a needs_retry task has PEER_CONFLICT: pivot the affected task strategy to CODE_WORKAROUND.",
        "6. If a needs_retry task has BREAKING_CHANGE: add a version constraint + pivot to CODE_WORKAROUND with refactor feedback.",
        "7. If a needs_retry task has SECURITY_FLAG: retry with current strategy (unless MAX_RETRIES reached).",
        f"8. Any task with {MAX_RETRIES}+ retries should appear in unfixable_task_ids, not targets.",
        "9. unfixable, qa_passed, and optimistically_fixed tasks MUST NOT appear in update_subagent or workaround_subagent target_task_ids.",
        "10. qa_critic target_task_ids MUST contain exactly the batch being evaluated.",
        "11. workaround_subagent MUST have exactly one target_task_id.",
        f"12. update_subagent MUST have between 1 and {UPDATE_BATCH_SIZE} target_task_ids.",
        "13. If a task is qa_passed, append its successful version bump or workaround to the constraints ledger via new_constraints.",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Supervisor node
# ---------------------------------------------------------------------------


def run_supervisor_node(state: OrchestratorState) -> Dict[str, Any]:
    """
    LangGraph node — Supervisor.

    Execution order
    ---------------
    1. Normalize task_queue: create initial RemediationTask entries for any
       valid_groups not yet represented.
    2. Update task statuses from subagent action summaries.
    3. Update task statuses from QA evaluations (only when status == "qa_completed").
    4. Increment retry_count for tasks newly entering NEEDS_RETRY.
    5. Mark UNFIXABLE any task whose retry_count has reached MAX_RETRIES.
    6. Build prompt from normalized state snapshot.
    7. Call ChatOpenAI.with_structured_output(SupervisorDecision); on any
       exception fall back to _deterministic_routing.
    8. Validate and clamp the decision (reject unknown IDs, enforce per-node
       target cardinality). Fall back to deterministic routing if clamping
       leaves the decision invalid.
    9. Apply strategy updates and unfixable marks from the decision.
    10. Return state patch.
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

    # ------------------------------------------------------------------
    # 1. Normalize task_queue: ensure every valid_group has a task
    # ------------------------------------------------------------------
    task_queue: Dict[str, RemediationTask] = dict(state.get("task_queue", {}))
    existing_group_ids = {t.parent_group_id for t in task_queue.values()}
    next_task_index = len(task_queue) + 1
    for group in valid_groups:
        if group.group_id not in existing_group_ids:
            task_id = f"task-{next_task_index}"
            task_queue[task_id] = build_initial_remediation_task(group, task_id)
            next_task_index += 1

    # Build a reverse lookup: parent_group_id → task
    task_by_group_id: Dict[str, RemediationTask] = {
        t.parent_group_id: t for t in task_queue.values()
    }

    # ------------------------------------------------------------------
    # 2. Update task statuses from subagent action summaries
    # ------------------------------------------------------------------
    active_target_task_ids = list(state.get("active_target_task_ids") or [])
    active_targets = set(active_target_task_ids)
    action_summaries: List[AgentActionSummary] = state.get("action_summaries") or []

    if active_targets and action_summaries:
        summary = action_summaries[-1]
        tid = summary.task_id
        # Support batch: prefix (task_id is "batch:<tid1>,<tid2>")
        if tid.startswith("batch:"):
            content = tid[len("batch:"):]
            tids = [t.strip() for t in content.split(",") if t.strip()]
        else:
            tids = [tid.strip()]

        for t_id in tids:
            resolved_t_id = t_id
            if t_id not in task_queue and t_id in task_by_group_id:
                resolved_t_id = task_by_group_id[t_id].task_id

            if resolved_t_id not in task_queue or resolved_t_id not in active_targets:
                continue
            task = task_queue[resolved_t_id]
            if task.status in (TaskStatus.QA_PASSED, TaskStatus.UNFIXABLE):
                continue
            if summary.status == AgentActionStatus.SUCCESS:
                task.status = TaskStatus.OPTIMISTICALLY_FIXED
            else:
                task.status = TaskStatus.NEEDS_RETRY
                task.retry_count += 1

    # ------------------------------------------------------------------
    # 3. Update task statuses from QA evaluations
    # ------------------------------------------------------------------
    qa_evaluations: Dict[str, QAEvaluation] = dict(state.get("qa_evaluations", {}))
    auto_new_constraints: List[str] = []

    if state.get("status") == "qa_completed":
        for eval_task_id, evaluation in qa_evaluations.items():
            resolved_t_id = eval_task_id
            if eval_task_id not in task_queue and eval_task_id in task_by_group_id:
                resolved_t_id = task_by_group_id[eval_task_id].task_id
                
            if resolved_t_id not in task_queue:
                continue
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
                # 4. Increment retry_count when newly entering NEEDS_RETRY
                task.status = TaskStatus.NEEDS_RETRY
                task.retry_count += 1

    # ------------------------------------------------------------------
    # 5. Mark UNFIXABLE tasks
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
    # 6. Build prompt state snapshot
    # ------------------------------------------------------------------
    prompt_state: OrchestratorState = {  # type: ignore[typeddict-item]
        **state,
        "task_queue": task_queue,
        "qa_evaluations": qa_evaluations,
    }

    # ------------------------------------------------------------------
    # 7. Short-circuit: if active batch is all optimistically_fixed → qa_critic
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

    # ------------------------------------------------------------------
    # 7b. Short-circuit: all remaining tasks are optimistically_fixed
    # ------------------------------------------------------------------
    if decision is None:
        tasks = list(task_queue.values())
        non_terminal = [t for t in tasks if t.status not in _TERMINAL_STATUSES]
        if non_terminal and all(t.status == TaskStatus.OPTIMISTICALLY_FIXED for t in non_terminal):
            decision = SupervisorDecision(
                next_node="qa_critic",
                target_task_ids=[t.task_id for t in non_terminal],
                instructions="Run QA on the remaining optimistically fixed tasks.",
                decision_reason="Routing all remaining optimistically fixed tasks to QA.",
            )

    # ------------------------------------------------------------------
    # 8. LLM structured call (only if short-circuit didn't fire)
    # ------------------------------------------------------------------
    if decision is None:
        try:
            from langchain_openai import ChatOpenAI  # type: ignore[import]

            model_name = os.environ.get("REMEDY_LLM_MODEL", _DEFAULT_MODEL)
            llm = ChatOpenAI(model=model_name, temperature=0)
            structured_llm = llm.with_structured_output(SupervisorDecision, method="function_calling")
            prompt_text = build_supervisor_prompt(prompt_state)
            logger.info("supervisor: invoking structured LLM for routing decision.")
            decision = structured_llm.invoke(prompt_text)
            logger.info(
                "supervisor: LLM decision → next_node=%s targets=%s reason=%s",
                decision.next_node,
                decision.target_task_ids,
                decision.decision_reason,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "supervisor: LLM call failed (%s) — using deterministic fallback.", exc
            )

    # Re-apply short-circuit post-LLM (LLM may have overridden it incorrectly)
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
    # 8. Validate and clamp the LLM decision (or use deterministic fallback)
    # ------------------------------------------------------------------
    if decision is None:
        decision = _deterministic_routing(
            task_queue,
            group_by_id,
            qa_evaluations,
            active_target_task_ids=active_target_task_ids,
            current_status=str(state.get("status") or ""),
        )
        logger.info(
            "supervisor: deterministic fallback → next_node=%s", decision.next_node
        )
    else:
        known_task_ids = set(task_queue.keys())

        # Remove unknown, terminal, or optimistically_fixed task IDs from targets
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

        # Enforce cardinality constraints — fall back if violated
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
                active_target_task_ids=active_target_task_ids,
                current_status=str(state.get("status") or ""),
            )
        else:
            try:
                decision = SupervisorDecision(
                    next_node=decision.next_node,
                    updated_task_strategies=decision.updated_task_strategies,
                    target_task_ids=valid_target_ids,
                    unfixable_task_ids=valid_unfixable_ids,
                    new_constraints=decision.new_constraints,
                    feedback_by_task={
                        k: v
                        for k, v in decision.feedback_by_task.items()
                        if k in known_task_ids
                    },
                    instructions=decision.instructions,
                    decision_reason=decision.decision_reason,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "supervisor: decision rebuild failed (%s) — falling back.", exc
                )
                decision = _deterministic_routing(
                    task_queue,
                    group_by_id,
                    qa_evaluations,
                    active_target_task_ids=active_target_task_ids,
                    current_status=str(state.get("status") or ""),
                )

    # ------------------------------------------------------------------
    # 9. Apply strategy updates and unfixable marks from decision
    # ------------------------------------------------------------------
    for t_id, new_strategy in decision.updated_task_strategies.items():
        if t_id in task_queue:
            task_queue[t_id].strategy = new_strategy

    for t_id in decision.unfixable_task_ids:
        if t_id in task_queue:
            task_queue[t_id].status = TaskStatus.UNFIXABLE

    logger.info(
        "supervisor: routing to '%s' with targets=%s",
        decision.next_node,
        decision.target_task_ids,
    )

    returned_constraints: List[str] = list(auto_new_constraints)
    for constraint in decision.new_constraints:
        if (
            constraint
            and constraint not in existing_constraints
            and constraint not in returned_constraints
        ):
            returned_constraints.append(constraint)

    # Build feedback_by_group for backward compat with subagent bridge nodes
    # (maps parent_group_id → feedback so bridge nodes can still look up by group)
    feedback_by_task = dict(decision.feedback_by_task)
    feedback_by_group: Dict[str, str] = {}
    for t_id, fb in feedback_by_task.items():
        if t_id in task_queue:
            gid = task_queue[t_id].parent_group_id
            feedback_by_group[gid] = fb

    # ------------------------------------------------------------------
    # 10. Return state patch
    # ------------------------------------------------------------------
    return {
        "status": "supervisor_routed",
        "next_routing_step": decision.next_node,
        "active_target_task_ids": list(decision.target_task_ids),
        # Keep active_target_group_ids populated for bridge nodes that still use it
        "active_target_group_ids": [
            task_queue[t].parent_group_id
            for t in decision.target_task_ids
            if t in task_queue
        ],
        "feedback_by_task": feedback_by_task,
        "feedback_by_group": feedback_by_group,
        "supervisor_instructions": decision.instructions,
        "task_queue": task_queue,
        # constraints_ledger uses operator.add — return only the NEW entries
        "constraints_ledger": returned_constraints,
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
