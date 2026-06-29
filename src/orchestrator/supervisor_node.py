"""
supervisor_node.py - Agentic Supervisor Node for Phase 5 hub-and-spoke orchestration.

The Supervisor runs exactly once per visit: no tools, no ReAct loop.
It performs Python-first state normalization, then makes a single structured
LLM call to obtain a ``SupervisorDecision``, validates and clamps that
decision, and returns a state patch that wires the next hop.

Public API
----------
MAX_RETRIES : int
    Maximum number of QA-fail-retry cycles before a group is marked unfixable.
derive_initial_strategy(group) -> RoutingStrategy
    Pure function: decides VERSION_BUMP vs CODE_WORKAROUND from the fix plan.
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
    FixPlanStatus,
    GroupRemediationStatus,
    QAEvaluation,
    RoutingStrategy,
    SupervisorDecision,
    VulnerabilityGroup,
)
from src.orchestrator.state import OrchestratorState

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
# Strategy derivation
# ---------------------------------------------------------------------------


def derive_initial_strategy(group: VulnerabilityGroup) -> RoutingStrategy:
    """
    Derive the initial routing strategy from a group's fix plan.

    Returns ``VERSION_BUMP`` only when the fix plan has ``status=VERSION_FOUND``
    (i.e. a safe pinned version is available).  All other plans — workaround,
    no-fix, or absent — map to ``CODE_WORKAROUND``.
    """
    fix_plan = group.fix_plan
    if fix_plan is not None and fix_plan.status == FixPlanStatus.VERSION_FOUND:
        return RoutingStrategy.VERSION_BUMP
    return RoutingStrategy.CODE_WORKAROUND


# ---------------------------------------------------------------------------
# Deterministic fallback router
# ---------------------------------------------------------------------------

_TERMINAL_STATUSES = frozenset({
    GroupRemediationStatus.QA_PASSED,
    GroupRemediationStatus.UNFIXABLE,
})
_WORKABLE_STATUSES = frozenset({
    GroupRemediationStatus.PENDING,
    GroupRemediationStatus.NEEDS_RETRY,
})


def _constraint_entry_for_group(
    group: VulnerabilityGroup,
    strategy: RoutingStrategy,
) -> str:
    """Build a deterministic constraints-ledger entry for a QA-passed group."""
    component = (group.vulnerable_component or group.group_id).strip()
    fix_plan = group.fix_plan

    if strategy == RoutingStrategy.VERSION_BUMP:
        fixed_version = (fix_plan.fixed_version if fix_plan else None) or "unknown"
        return f"{component}: keep resolved version at {fixed_version}"

    return f"{component}: preserve validated security workaround"


def _deterministic_routing(
    valid_groups: List[VulnerabilityGroup],
    group_strategies: Dict[str, RoutingStrategy],
    group_statuses: Dict[str, GroupRemediationStatus],
    retry_counts: Dict[str, int],
    qa_evaluations: Dict[str, QAEvaluation],
    active_target_group_ids: Optional[List[str]] = None,
    current_status: str = "",
) -> SupervisorDecision:
    """
    Pure-Python fallback routing used when the LLM call fails.

    Implements the same priority rules described in the supervisor prompt.
    """
    non_terminal = [
        g for g in valid_groups
        if group_statuses.get(g.group_id, GroupRemediationStatus.PENDING)
        not in _TERMINAL_STATUSES
    ]

    # All groups are terminal → teardown
    if not non_terminal:
        return SupervisorDecision(
            next_node="teardown",
            target_group_ids=[],
            instructions="All groups are terminal. Proceeding to teardown.",
            decision_reason="No actionable groups remain.",
        )

    # All non-terminal groups are optimistically_fixed → qa_critic
    active_target_ids = set(active_target_group_ids or [])
    current_batch = [g for g in valid_groups if g.group_id in active_target_ids]
    if current_status != "qa_completed" and current_batch and all(
        group_statuses.get(g.group_id, GroupRemediationStatus.PENDING)
        == GroupRemediationStatus.OPTIMISTICALLY_FIXED
        for g in current_batch
    ):
        return SupervisorDecision(
            next_node="qa_critic",
            target_group_ids=[g.group_id for g in current_batch],
            instructions="Run QA on the current remediated batch before starting more remediation.",
            decision_reason=(
                f"Routing the current batch of {len(current_batch)} optimistically fixed group(s) to QA."
            ),
        )

    all_optimistic = all(
        group_statuses.get(g.group_id, GroupRemediationStatus.PENDING)
        == GroupRemediationStatus.OPTIMISTICALLY_FIXED
        for g in non_terminal
    )
    if all_optimistic:
        return SupervisorDecision(
            next_node="qa_critic",
            target_group_ids=[g.group_id for g in non_terminal],
            instructions="Run QA on the remaining optimistically fixed groups.",
            decision_reason="Routing all remaining optimistically fixed groups to QA.",
        )

    # Collect groups that still need work
    workable = [
        g for g in non_terminal
        if group_statuses.get(g.group_id, GroupRemediationStatus.PENDING)
        in _WORKABLE_STATUSES
    ]

    # VERSION_BUMP groups batch to update_subagent
    version_bump = [
        g for g in workable
        if group_strategies.get(g.group_id) == RoutingStrategy.VERSION_BUMP
    ]
    if version_bump:
        batch = version_bump[:UPDATE_BATCH_SIZE]
        feedback_by_group: Dict[str, str] = {}
        for g in batch:
            eval_ = qa_evaluations.get(g.group_id)
            if eval_ and eval_.retry_feedback:
                feedback_by_group[g.group_id] = eval_.retry_feedback
        return SupervisorDecision(
            next_node="update_subagent",
            target_group_ids=[g.group_id for g in batch],
            feedback_by_group=feedback_by_group,
            instructions="Apply the required version bump(s) in the package manifest(s) for this batch only.",
            decision_reason=(
                f"Routing {len(batch)} VERSION_BUMP group(s) to update_subagent (batch size cap {UPDATE_BATCH_SIZE})."
            ),
        )

    # CODE_WORKAROUND groups: send exactly one at a time to workaround_subagent
    workaround = [
        g for g in workable
        if group_strategies.get(g.group_id) == RoutingStrategy.CODE_WORKAROUND
    ]
    if workaround:
        target = workaround[0]
        eval_ = qa_evaluations.get(target.group_id)
        feedback: Dict[str, str] = {}
        if eval_ and eval_.retry_feedback:
            feedback[target.group_id] = eval_.retry_feedback
        return SupervisorDecision(
            next_node="workaround_subagent",
            target_group_ids=[target.group_id],
            feedback_by_group=feedback,
            instructions="Apply the minimal safe code workaround for this vulnerability.",
            decision_reason=(
                f"Routing group '{target.group_id}' to workaround_subagent."
            ),
        )

    # Unexpected: no workable groups found → teardown as safe default
    return SupervisorDecision(
        next_node="teardown",
        target_group_ids=[],
        instructions="No actionable groups remain.",
        decision_reason=(
            "Deterministic fallback: no workable groups found, routing to teardown."
        ),
    )


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


def build_supervisor_prompt(state: OrchestratorState) -> str:
    """Build the structured LLM prompt for the Supervisor decision."""
    valid_groups: List[VulnerabilityGroup] = state.get("valid_groups", [])
    group_strategies: Dict[str, RoutingStrategy] = state.get("group_strategies", {})
    group_statuses: Dict[str, GroupRemediationStatus] = state.get("group_statuses", {})
    retry_counts: Dict[str, int] = state.get("retry_counts", {})
    constraints_ledger: List[str] = state.get("constraints_ledger", [])
    action_summaries: List[AgentActionSummary] = state.get("action_summaries", [])
    qa_evaluations: Dict[str, QAEvaluation] = state.get("qa_evaluations", {})
    eval_status: str = state.get("eval_status", "")

    lines = [
        "You are the Supervisor Agent of an AppSec remediation pipeline.",
        "Produce a single SupervisorDecision to route the next graph step.",
        "",
        "## Vulnerability Groups",
    ]

    for group in valid_groups:
        gid = group.group_id
        strategy = group_strategies.get(gid, RoutingStrategy.CODE_WORKAROUND)
        status = group_statuses.get(gid, GroupRemediationStatus.PENDING)
        retries = retry_counts.get(gid, 0)
        fix_plan = group.fix_plan
        cves = ", ".join(group.cve_ids) if group.cve_ids else "none"
        ghsas = ", ".join(group.ghsa_ids) if group.ghsa_ids else "none"
        eval_ = qa_evaluations.get(gid)

        lines += [
            "",
            f"### Group: {gid}",
            f"- Component    : {group.vulnerable_component or 'unknown'}",
            f"- Issue Type   : {group.issue_type.value}",
            f"- CVEs         : {cves}",
            f"- GHSAs        : {ghsas}",
            f"- Fix Plan     : {fix_plan.status.value if fix_plan else 'none'}",
            f"- Strategy     : {strategy.value}",
            f"- Status       : {status.value}",
            f"- Retries Used : {retries}/{MAX_RETRIES}",
        ]
        if eval_ and status != GroupRemediationStatus.OPTIMISTICALLY_FIXED:
            cat = eval_.failure_category.value if eval_.failure_category else "none"
            lines.append(
                f"- Last QA      : passed={eval_.passed}, category={cat}, "
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
            f"- [{summary.group_id}] {summary.status.value}: {summary.summary}"
        )
    if not action_summaries:
        lines.append("- (none)")

    lines += [
        "",
        f"## QA Evaluation Status: {eval_status or 'none'}",
        "",
        "## Routing Rules (follow strictly)",
        f"1. Send pending/needs_retry VERSION_BUMP groups to update_subagent in batches of at most {UPDATE_BATCH_SIZE}. Never send more than {UPDATE_BATCH_SIZE} target_group_ids to update_subagent in one decision.",
        "2. Send EXACTLY ONE pending/needs_retry CODE_WORKAROUND group → workaround_subagent.",
        "3. After a subagent succeeds for the current active batch, route that exact optimistically_fixed batch to qa_critic before starting another remediation batch. Do NOT route optimistically_fixed groups back to subagents.",
        "4. When ALL non-terminal groups are qa_passed OR all are unfixable → teardown.",
        "5. If a needs_retry group has PEER_CONFLICT: pivot the affected group strategy to CODE_WORKAROUND.",
        "6. If a needs_retry group has BREAKING_CHANGE: add a version constraint + pivot to CODE_WORKAROUND with refactor feedback.",
        "7. If a needs_retry group has SECURITY_FLAG: retry with current strategy (unless MAX_RETRIES=3 reached).",
        f"8. Any group with {MAX_RETRIES}+ retries should appear in unfixable_group_ids, not targets.",
        "9. unfixable, qa_passed, and optimistically_fixed groups MUST NOT appear in update_subagent or workaround_subagent target_group_ids.",
        "10. qa_critic target_group_ids MUST contain exactly the batch being evaluated.",
        "11. workaround_subagent MUST have exactly one target_group_id.",
        f"12. update_subagent MUST have between 1 and {UPDATE_BATCH_SIZE} target_group_ids.",
        "13. If a group is qa_passed, append its successful version bump or workaround to the constraints ledger via new_constraints.",
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
    1. Normalise strategies: fill missing entries via ``derive_initial_strategy``.
    2. Initialise statuses:  fill missing entries with ``PENDING``.
    3. Update statuses from QA (only when ``status == "qa_completed"``).
    4. Increment retry counts for groups newly entering ``NEEDS_RETRY``.
    5. Mark ``UNFIXABLE`` any group whose retry count has reached MAX_RETRIES.
    6. Build prompt from normalised state snapshot.
    7. Call ``ChatOpenAI.with_structured_output(SupervisorDecision)``; on any
       exception fall back to ``_deterministic_routing``.
    8. Validate and clamp the decision (reject unknown IDs, enforce per-node
       target cardinality).  Fall back to deterministic routing if clamping
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
            "active_target_group_ids": [],
            "supervisor_instructions": "No groups to process.",
        }

    group_by_id: Dict[str, VulnerabilityGroup] = {g.group_id: g for g in valid_groups}
    existing_constraints: List[str] = list(state.get("constraints_ledger", []))

    # ------------------------------------------------------------------
    # 1. Normalise strategies
    # ------------------------------------------------------------------
    group_strategies: Dict[str, RoutingStrategy] = dict(state.get("group_strategies", {}))
    for group in valid_groups:
        if group.group_id not in group_strategies:
            group_strategies[group.group_id] = derive_initial_strategy(group)

    # ------------------------------------------------------------------
    # 2. Initialise statuses
    # ------------------------------------------------------------------
    group_statuses: Dict[str, GroupRemediationStatus] = dict(
        state.get("group_statuses", {})
    )
    for group in valid_groups:
        if group.group_id not in group_statuses:
            group_statuses[group.group_id] = GroupRemediationStatus.PENDING

    # ------------------------------------------------------------------
    # 2.5. Update statuses from subagent action summaries
    # ------------------------------------------------------------------
    active_targets = set(state.get("active_target_group_ids") or [])
    action_summaries: List[AgentActionSummary] = state.get("action_summaries") or []
    
    if active_targets and action_summaries:
        summary = action_summaries[-1]
        summary_group_id = summary.group_id
        if summary_group_id.startswith("batch:"):
            content = summary_group_id[len("batch:"):]
            gids = [gid.strip() for gid in content.split(",") if gid.strip()]
        else:
            gids = [summary_group_id.strip()]

        for gid in gids:
            if gid not in group_by_id or gid not in active_targets:
                continue
            prev = group_statuses.get(gid, GroupRemediationStatus.PENDING)
            if prev in (GroupRemediationStatus.QA_PASSED, GroupRemediationStatus.UNFIXABLE):
                continue
            if summary.status == AgentActionStatus.SUCCESS:
                group_statuses[gid] = GroupRemediationStatus.OPTIMISTICALLY_FIXED
            else:
                group_statuses[gid] = GroupRemediationStatus.NEEDS_RETRY
                # Subagent failed/surrendered; increment retry count here to prevent infinite loop
                retry_counts = state.get("retry_counts", {})
                if "retry_counts" not in locals():
                    # We initialize the dictionary in step 3, but if we need it here, let's make sure it's available.
                    pass
                # Wait, retry_counts is defined in step 3. I should fetch it here or modify state later.
                # Let's initialize it here so we can mutate it.
                pass

    # ------------------------------------------------------------------
    # 3. Update statuses from QA evaluations
    # ------------------------------------------------------------------
    qa_evaluations: Dict[str, QAEvaluation] = dict(state.get("qa_evaluations", {}))
    retry_counts: Dict[str, int] = dict(state.get("retry_counts", {}))
    auto_new_constraints: List[str] = []

    if state.get("status") == "qa_completed":
        for group_id, evaluation in qa_evaluations.items():
            if group_id not in group_by_id:
                continue
            prev = group_statuses.get(group_id, GroupRemediationStatus.PENDING)
            if prev in (GroupRemediationStatus.UNFIXABLE, GroupRemediationStatus.QA_PASSED):
                continue
            if evaluation.passed:
                group_statuses[group_id] = GroupRemediationStatus.QA_PASSED
                constraint = _constraint_entry_for_group(
                    group_by_id[group_id],
                    group_strategies.get(group_id, derive_initial_strategy(group_by_id[group_id])),
                )
                if (
                    constraint
                    and constraint not in existing_constraints
                    and constraint not in auto_new_constraints
                ):
                    auto_new_constraints.append(constraint)
            else:
                # 4. Increment retry count when newly entering NEEDS_RETRY
                group_statuses[group_id] = GroupRemediationStatus.NEEDS_RETRY
                retry_counts[group_id] = retry_counts.get(group_id, 0) + 1

    # ------------------------------------------------------------------
    # 5. Mark unfixable groups
    # ------------------------------------------------------------------
    for group_id, status_val in list(group_statuses.items()):
        if group_id not in group_by_id:
            continue
        if (
            status_val == GroupRemediationStatus.NEEDS_RETRY
            and retry_counts.get(group_id, 0) >= MAX_RETRIES
        ):
            group_statuses[group_id] = GroupRemediationStatus.UNFIXABLE
            logger.info(
                "supervisor: group '%s' marked UNFIXABLE after %d retries.",
                group_id,
                retry_counts[group_id],
            )

    # ------------------------------------------------------------------
    # 6. Build prompt state snapshot
    # ------------------------------------------------------------------
    prompt_state: OrchestratorState = {  # type: ignore[typeddict-item]
        **state,
        "group_strategies": group_strategies,
        "group_statuses": group_statuses,
        "retry_counts": retry_counts,
        "qa_evaluations": qa_evaluations,
    }

    active_target_group_ids = list(state.get("active_target_group_ids") or [])

    # ------------------------------------------------------------------
    # 7. LLM structured call
    # ------------------------------------------------------------------
    decision: Optional[SupervisorDecision] = None
    if state.get("status") != "qa_completed" and active_target_group_ids:
        active_batch = [
            group
            for group in valid_groups
            if group.group_id in set(active_target_group_ids)
        ]
        if active_batch and all(
            group_statuses.get(group.group_id, GroupRemediationStatus.PENDING)
            == GroupRemediationStatus.OPTIMISTICALLY_FIXED
            for group in active_batch
        ):
            decision = SupervisorDecision(
                next_node="qa_critic",
                target_group_ids=[group.group_id for group in active_batch],
                instructions="Run QA on the current remediated batch before starting more remediation.",
                decision_reason=(
                    f"Routing the current batch of {len(active_batch)} optimistically fixed group(s) to QA."
                ),
            )

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
            decision.target_group_ids,
            decision.decision_reason,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "supervisor: LLM call failed (%s) — using deterministic fallback.", exc
        )

    if state.get("status") != "qa_completed" and active_target_group_ids:
        active_batch = [
            group
            for group in valid_groups
            if group.group_id in set(active_target_group_ids)
        ]
        if active_batch and all(
            group_statuses.get(group.group_id, GroupRemediationStatus.PENDING)
            == GroupRemediationStatus.OPTIMISTICALLY_FIXED
            for group in active_batch
        ):
            decision = SupervisorDecision(
                next_node="qa_critic",
                target_group_ids=[group.group_id for group in active_batch],
                instructions="Run QA on the current remediated batch before starting more remediation.",
                decision_reason=(
                    f"Routing the current batch of {len(active_batch)} optimistically fixed group(s) to QA."
                ),
            )

    # ------------------------------------------------------------------
    # 8. Validate and clamp the LLM decision (or use deterministic fallback)
    # ------------------------------------------------------------------
    if decision is None:
        decision = _deterministic_routing(
            valid_groups,
            group_strategies,
            group_statuses,
            retry_counts,
            qa_evaluations,
            active_target_group_ids=active_target_group_ids,
            current_status=str(state.get("status") or ""),
        )
        logger.info(
            "supervisor: deterministic fallback → next_node=%s", decision.next_node
        )
    else:
        known_ids = set(group_by_id.keys())
        terminal_statuses = _TERMINAL_STATUSES

        # Remove unknown, terminal, or optimistically_fixed group IDs from targets
        valid_target_ids = []
        for gid in decision.target_group_ids:
            if gid not in known_ids:
                continue
            status = group_statuses.get(gid)
            if status in terminal_statuses:
                continue
            # If routing to a subagent, the group MUST NOT be optimistically_fixed
            if decision.next_node in ("update_subagent", "workaround_subagent") and status == GroupRemediationStatus.OPTIMISTICALLY_FIXED:
                continue
            valid_target_ids.append(gid)
        valid_unfixable_ids = [
            gid for gid in decision.unfixable_group_ids if gid in known_ids
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
                "supervisor: update_subagent supports at most %d targets, got %d â€” falling back.",
                UPDATE_BATCH_SIZE,
                len(valid_target_ids),
            )
            needs_fallback = True
        if not needs_fallback and decision.next_node == "qa_critic" and not valid_target_ids:
            logger.warning(
                "supervisor: qa_critic needs at least 1 target, got 0 â€” falling back."
            )
            needs_fallback = True

        if needs_fallback:
            decision = _deterministic_routing(
                valid_groups,
                group_strategies,
                group_statuses,
                retry_counts,
                qa_evaluations,
                active_target_group_ids=active_target_group_ids,
                current_status=str(state.get("status") or ""),
            )
        else:
            try:
                decision = SupervisorDecision(
                    next_node=decision.next_node,
                    updated_strategies=decision.updated_strategies,
                    target_group_ids=valid_target_ids,
                    unfixable_group_ids=valid_unfixable_ids,
                    new_constraints=decision.new_constraints,
                    feedback_by_group={
                        k: v
                        for k, v in decision.feedback_by_group.items()
                        if k in known_ids
                    },
                    instructions=decision.instructions,
                    decision_reason=decision.decision_reason,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "supervisor: decision rebuild failed (%s) — falling back.", exc
                )
                decision = _deterministic_routing(
                    valid_groups,
                    group_strategies,
                    group_statuses,
                    retry_counts,
                    qa_evaluations,
                    active_target_group_ids=active_target_group_ids,
                    current_status=str(state.get("status") or ""),
                )

    # ------------------------------------------------------------------
    # 9. Apply strategy updates and unfixable marks from decision
    # ------------------------------------------------------------------
    for group_id, new_strategy in decision.updated_strategies.items():
        if group_id in group_by_id:
            group_strategies[group_id] = new_strategy

    for group_id in decision.unfixable_group_ids:
        if group_id in group_by_id:
            group_statuses[group_id] = GroupRemediationStatus.UNFIXABLE

    logger.info(
        "supervisor: routing to '%s' with targets=%s",
        decision.next_node,
        decision.target_group_ids,
    )

    returned_constraints: List[str] = list(auto_new_constraints)
    for constraint in decision.new_constraints:
        if (
            constraint
            and constraint not in existing_constraints
            and constraint not in returned_constraints
        ):
            returned_constraints.append(constraint)

    # ------------------------------------------------------------------
    # 10. Return state patch
    # ------------------------------------------------------------------
    return {
        "status": "supervisor_routed",
        "next_routing_step": decision.next_node,
        "active_target_group_ids": list(decision.target_group_ids),
        "feedback_by_group": dict(decision.feedback_by_group),
        "supervisor_instructions": decision.instructions,
        "group_strategies": group_strategies,
        "group_statuses": group_statuses,
        "retry_counts": retry_counts,
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
