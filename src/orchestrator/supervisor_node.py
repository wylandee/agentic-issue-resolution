"""
supervisor_node.py - Agentic Supervisor Node for Phase 5 hub-and-spoke orchestration.

Phase 2 architecture: Two-Phase Commander
-----------------------------------------
The Supervisor now runs as an explicit two-phase commander:

  Phase A — Planner (conditional):
    A bounded ReAct loop with access to ``view_npm_package_versions``.
    Runs only when there are NEEDS_RETRY or PENDING tasks with empty instructions.
    Outputs a free-form "Strategy Scratchpad" with observations, playbook selections,
    registry findings, instruction revisions, spawn recommendations, and routing notes.

  Phase B — Router (always):
    A zero-shot ``ChatOpenAI.with_structured_output(SupervisorDecision)`` call
    that reads the scratchpad and emits the strict typed routing decision.

  Guardrails (Python):
    Validate and apply the decision: reject unknown task IDs, clamp cardinality,
    apply copy-on-write task updates, materialize spawn requests, enforce depth
    and queue-size caps.

Public API
----------
MAX_RETRIES : int
    Maximum number of QA-fail-retry cycles before a task is marked unfixable.
build_supervisor_prompt(state) -> str
    Builds the Router prompt text (includes planner scratchpad when available).
run_supervisor_node(state) -> Dict[str, Any]
    LangGraph node callable.
supervisor_router(state) -> str
    Conditional-edge callable: reads ``next_routing_step`` from state.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Set

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
    VulnerabilityGroup,
)
from src.orchestrator.state import OrchestratorState
from src.orchestrator.subagent_runtime import run_bounded_subagent_loop
from src.orchestrator.task_utils import build_initial_remediation_task
from src.tools.registry_tools import view_npm_package_versions

logger = logging.getLogger(__name__)

MAX_RETRIES: int = 3
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
# Planner phase helpers
# ---------------------------------------------------------------------------


def _needs_planner(
    task_queue: Dict[str, RemediationTask],
) -> bool:
    """Return True when the planner should be invoked.

    Conditions:
    - Any task is NEEDS_RETRY (requires failure investigation), OR
    - Any PENDING task has an empty instruction (requires initial instruction writing).
    """
    for task in task_queue.values():
        if task.status == TaskStatus.NEEDS_RETRY:
            return True
        if task.status == TaskStatus.PENDING and not task.instruction.strip():
            return True
    return False


def _build_planner_prompt(
    task_queue: Dict[str, RemediationTask],
    group_by_id: Dict[str, VulnerabilityGroup],
    qa_evaluations: Dict[str, QAEvaluation],
    action_summaries: List[AgentActionSummary],
    constraints_ledger: List[str],
) -> str:
    """Build the planner system + user messages."""
    lines = [
        "You are the Planner phase of an AppSec remediation Supervisor.",
        "Your job is to investigate failures, query the npm registry, select a playbook,",
        "and write precise remediation instructions for each task that needs attention.",
        "",
        "Use the view_npm_package_versions tool to look up version availability before",
        "writing an instruction for any package. Do not guess version numbers.",
        "",
        "## Task Queue",
    ]

    for task in task_queue.values():
        group = group_by_id.get(task.parent_group_id)
        cves = ", ".join(group.cve_ids) if group and group.cve_ids else "none"
        ghsas = ", ".join(group.ghsa_ids) if group and group.ghsa_ids else "none"
        component = group.vulnerable_component if group else task.parent_group_id
        eval_ = qa_evaluations.get(task.task_id)

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
        "## Queue Caps",
        f"- MAX_TASK_QUEUE_SIZE = {MAX_TASK_QUEUE_SIZE} (current: {len(task_queue)})",
        f"- MAX_ANCESTRY_DEPTH = {MAX_ANCESTRY_DEPTH}",
        "",
        "## 5 Supervisor Playbooks (apply the appropriate one per failed task)",
        "",
        "### Playbook 1: SECURITY_FLAG (Scanner still detects CVE after version bump)",
        "  - Query registry for the package. Find the absolute latest version.",
        "  - If not on latest: write instruction to force an 'overrides' block at repo root.",
        "  - If already on latest and scanner still flags: pivot strategy to CODE_WORKAROUND.",
        "  - Instruction must specify the exact version and manifest file path.",
        "",
        "### Playbook 2: PEER_CONFLICT (ERESOLVE or npm peer tree conflict)",
        "  - Query registry to check if a safe backported patch exists in a lower major.",
        "  - If backport available: instruct update_subagent to use that specific version.",
        "  - If no safe backport: pivot strategy to CODE_WORKAROUND.",
        "  - Instruction must explain what conflict to avoid.",
        "",
        "### Playbook 3: BREAKING_CHANGE (Tests fail after bump)",
        "  - Mark the version bump as successful (use task_status_updates: task-X = QA_PASSED).",
        "  - Add the locked version to new_constraints.",
        "  - Spawn a CODE_WORKAROUND child task (if depth < MAX_ANCESTRY_DEPTH) to fix broken APIs.",
        "  - Spawn instruction must name the specific broken test and API calls to refactor.",
        "",
        "### Playbook 4: Abandoned Package (404 from registry or stale publish date)",
        "  - If registry returns 404 or last publish was 5+ years ago: pivot to CODE_WORKAROUND.",
        "  - Instruction must describe the sanitization or wrapper to add around the library.",
        "",
        "### Playbook 5: EBADENGINE (Node.js version mismatch)",
        "  - Query registry for an older compatible patch that supports the sandbox Node version.",
        "  - If found: write instruction with that version. If not: pivot to CODE_WORKAROUND.",
        "",
        "## Output Format",
        "Write a 'Strategy Scratchpad' with these sections for each task needing attention:",
        "  1. Observations (what failed and why)",
        "  2. Playbook selected",
        "  3. Registry findings (if you queried)",
        "  4. Revised instruction text",
        "  5. Spawn recommendations (if BREAKING_CHANGE requires a child task)",
        "  6. Routing notes",
    ]

    return "\n".join(lines)


def _run_planner_phase(
    task_queue: Dict[str, RemediationTask],
    group_by_id: Dict[str, VulnerabilityGroup],
    qa_evaluations: Dict[str, QAEvaluation],
    action_summaries: List[AgentActionSummary],
    constraints_ledger: List[str],
    llm: Any,
) -> str:
    """Run the bounded planner ReAct loop. Returns the scratchpad text."""
    planner_prompt = _build_planner_prompt(
        task_queue, group_by_id, qa_evaluations, action_summaries, constraints_ledger
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
    eval_status: str = state.get("eval_status", "")

    group_by_id = {g.group_id: g for g in valid_groups}

    lines = [
        "You are the Router phase of an AppSec remediation Supervisor.",
        "Produce a single SupervisorDecision to route the next graph step.",
        "Follow the Routing Rules strictly.",
        "",
    ]

    if scratchpad:
        lines += [
            "## Planner Scratchpad (use these findings to write instructions and spawn tasks)",
            scratchpad,
            "",
        ]

    lines.append("## Remediation Tasks")

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
        f"## Queue Caps: {len(task_queue)}/{MAX_TASK_QUEUE_SIZE} tasks used, depth cap = {MAX_ANCESTRY_DEPTH}",
        "",
        "## Routing Rules (follow strictly)",
        f"1. Send pending/needs_retry VERSION_BUMP tasks to update_subagent in batches of at most {UPDATE_BATCH_SIZE}.",
        "   Use revised_instructions to set the exact instruction per task from the planner scratchpad.",
        "2. Send EXACTLY ONE pending/needs_retry CODE_WORKAROUND task → workaround_subagent.",
        "3. After a subagent succeeds for the current active batch, route that exact optimistically_fixed batch to qa_critic.",
        "4. When ALL non-terminal tasks are qa_passed OR all are unfixable → teardown.",
        "5. PEER_CONFLICT → pivot strategy to CODE_WORKAROUND via updated_task_strategies.",
        "6. BREAKING_CHANGE → mark parent QA_PASSED via task_status_updates, lock version via new_constraints, spawn child task.",
        "7. SECURITY_FLAG → retry with current strategy; use revised_instructions with the exact version from planner.",
        f"8. Any task with {MAX_RETRIES}+ retries → unfixable_task_ids.",
        "9. unfixable, qa_passed, and optimistically_fixed tasks MUST NOT appear in worker target_task_ids.",
        "10. qa_critic target_task_ids MUST contain exactly the batch being evaluated.",
        "11. workaround_subagent MUST have exactly one target_task_id.",
        f"12. update_subagent MUST have between 1 and {UPDATE_BATCH_SIZE} target_task_ids.",
        "13. task_status_updates may only set QA_PASSED or UNFIXABLE; no other values allowed.",
        "14. revised_instructions values must be non-empty strings.",
        f"15. spawn_requests: parent depth must be < {MAX_ANCESTRY_DEPTH}; total queue size must stay ≤ {MAX_TASK_QUEUE_SIZE}.",
        "16. If a task is qa_passed, append its successful version bump or workaround to new_constraints.",
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
) -> Dict[str, RemediationTask]:
    """Validate and materialize spawn requests into new RemediationTask objects.

    Returns a dict of new task_id → RemediationTask to be merged into task_queue.
    Rejected requests are logged to errors.
    """
    next_index = len(task_queue) + 1
    new_tasks: Dict[str, RemediationTask] = {}
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
        logger.info(
            "supervisor: spawned child task '%s' (parent='%s', depth=%d, strategy=%s) — %s",
            child_task_id,
            req.parent_task_id,
            child_depth,
            req.strategy.value,
            req.reason,
        )

    return new_tasks


# ---------------------------------------------------------------------------
# Supervisor node
# ---------------------------------------------------------------------------


def run_supervisor_node(state: OrchestratorState) -> Dict[str, Any]:
    """
    LangGraph node — Supervisor (Phase 2: Two-Phase Commander).

    Execution stages
    ----------------
    1. Normalize task_queue: create initial RemediationTask entries for any
       valid_groups not yet represented (copy-on-write via model_copy).
    2. Ingest subagent action summaries for current active_target_task_ids only.
    3. Ingest QA results for active task IDs only (when status == "qa_completed").
    4. Mark UNFIXABLE any task whose retry_count has reached MAX_RETRIES.
    5. Short-circuit: if active batch is all optimistically_fixed → qa_critic.
    6. Planner phase: run only when NEEDS_RETRY or PENDING tasks with empty
       instructions exist. Uses view_npm_package_versions tool.
    7. Router phase: ChatOpenAI.with_structured_output(SupervisorDecision).
    8. Guardrails: reject unknown IDs, enforce cardinality, fall back to
       deterministic routing if invalid.
    9. Apply guarded: revised_instructions, strategy updates, status overrides,
       unfixable marks, new constraints, and materialized spawn requests.
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

    # Build a reverse lookup: parent_group_id → task
    task_by_group_id: Dict[str, RemediationTask] = {
        t.parent_group_id: t for t in task_queue.values()
    }

    # ------------------------------------------------------------------
    # 2. Ingest subagent action summaries (active targets only)
    # ------------------------------------------------------------------
    active_target_task_ids = list(state.get("active_target_task_ids") or [])
    active_targets = set(active_target_task_ids)
    action_summaries: List[AgentActionSummary] = state.get("action_summaries") or []

    if active_targets and action_summaries:
        recent_summaries = {}
        for summary in action_summaries:
            resolved_t_id = summary.task_id
            if summary.task_id not in task_queue and summary.task_id in task_by_group_id:
                resolved_t_id = task_by_group_id[summary.task_id].task_id
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
        if non_terminal and all(t.status == TaskStatus.OPTIMISTICALLY_FIXED for t in non_terminal):
            decision = SupervisorDecision(
                next_node="qa_critic",
                target_task_ids=[t.task_id for t in non_terminal],
                instructions="Run QA on the remaining optimistically fixed tasks.",
                decision_reason="Routing all remaining optimistically fixed tasks to QA.",
            )

    # ------------------------------------------------------------------
    # 6. Planner phase (only when needed, only when LLM is available)
    # ------------------------------------------------------------------
    scratchpad = ""
    if decision is None:
        planner_needed = _needs_planner(task_queue)
        if planner_needed:
            try:
                from langchain_openai import ChatOpenAI  # type: ignore[import]

                model_name = os.environ.get("REMEDY_LLM_MODEL", _DEFAULT_MODEL)
                planner_llm = ChatOpenAI(model=model_name, temperature=0)
                scratchpad = _run_planner_phase(
                    task_queue=task_queue,
                    group_by_id=group_by_id,
                    qa_evaluations=qa_evaluations,
                    action_summaries=action_summaries,
                    constraints_ledger=existing_constraints,
                    llm=planner_llm,
                )
                logger.info(
                    "supervisor: planner completed (%d chars scratchpad).", len(scratchpad)
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "supervisor: planner LLM initialization failed (%s) — skipping planner.", exc
                )
        else:
            scratchpad = "(No replanning needed — no failed or uninitialized tasks.)"
            logger.info("supervisor: planner skipped (no failed/uninstructed tasks).")

    # ------------------------------------------------------------------
    # 7. Router phase (structured LLM call)
    # ------------------------------------------------------------------
    if decision is None:
        try:
            from langchain_openai import ChatOpenAI  # type: ignore[import]

            model_name = os.environ.get("REMEDY_LLM_MODEL", _DEFAULT_MODEL)
            router_llm = ChatOpenAI(model=model_name, temperature=0)
            structured_llm = router_llm.with_structured_output(
                SupervisorDecision, method="function_calling"
            )
            prompt_state: OrchestratorState = {  # type: ignore[typeddict-item]
                **state,
                "task_queue": task_queue,
                "qa_evaluations": qa_evaluations,
            }
            prompt_text = build_supervisor_prompt(prompt_state, scratchpad=scratchpad)
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
    # 8. Guardrails: validate and clamp (or deterministic fallback)
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

            try:
                decision = SupervisorDecision(
                    next_node=decision.next_node,
                    updated_task_strategies={
                        k: v
                        for k, v in decision.updated_task_strategies.items()
                        if k in known_task_ids
                    },
                    target_task_ids=valid_target_ids,
                    unfixable_task_ids=valid_unfixable_ids,
                    new_constraints=decision.new_constraints,
                    feedback_by_task=clean_feedback,
                    revised_instructions=clean_revised_instructions,
                    spawn_requests=decision.spawn_requests,
                    task_status_updates=clean_status_updates,
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
    # 9. Apply guarded updates to task_queue
    # ------------------------------------------------------------------

    # 9a. Apply revised_instructions (copy-on-write per task)
    for t_id, new_instr in decision.revised_instructions.items():
        if t_id in task_queue and new_instr.strip():
            task_queue[t_id] = task_queue[t_id].model_copy(update={"instruction": new_instr})

    # 9b. Apply strategy pivots
    for t_id, new_strategy in decision.updated_task_strategies.items():
        if t_id in task_queue:
            task_queue[t_id] = task_queue[t_id].model_copy(update={"strategy": new_strategy})

    # 9c. Apply guarded task status overrides (only QA_PASSED and UNFIXABLE)
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

    # 9d. Apply unfixable marks from decision
    for t_id in decision.unfixable_task_ids:
        if t_id in task_queue:
            task_queue[t_id] = task_queue[t_id].model_copy(update={"status": TaskStatus.UNFIXABLE})

    # 9e. Materialize spawn requests
    if decision.spawn_requests:
        new_tasks = _materialize_spawn_requests(
            spawn_requests=list(decision.spawn_requests),
            task_queue=task_queue,
            group_by_id=group_by_id,
            errors=errors,
        )
        task_queue.update(new_tasks)

    logger.info(
        "supervisor: routing to '%s' with targets=%s",
        decision.next_node,
        decision.target_task_ids,
    )

    # ------------------------------------------------------------------
    # 9f. Collect new constraints
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
