"""Supervisor-callable outer portfolio planning transaction."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Mapping
from typing import Any

from remediation_engine.contracts.schemas import IssueType
from remediation_engine.contracts.solver_models import (
    PackageResolutionStatus,
    PortfolioReplanRequest,
)
from remediation_engine.orchestration.portfolio_orchestrator import (
    apply_portfolio_plan,
    build_certified_portfolio_plan,
    prepare_portfolio_inputs,
)
from remediation_engine.orchestration.runtime_context import get_runtime_settings
from remediation_engine.orchestration.state import OrchestratorState
from remediation_engine.orchestration.task_utils import TERMINAL_TASK_STATUSES
from remediation_engine.language import resolve_project_language

log = logging.getLogger(__name__)

MAX_PORTFOLIO_REPLAN_ATTEMPTS = 3
MAX_PORTFOLIO_ITERATIONS = 12


def _fetch_fresh_registry_packument(package_name: str) -> Mapping[str, Any]:
    """Fetch current npm metadata for solver candidate selection."""
    from remediation_engine.tools.registry_tools import _fetch_package_data

    return _fetch_package_data(package_name, cache=None)


def _portfolio_replan_request(state: OrchestratorState) -> PortfolioReplanRequest | None:
    """Normalize typed and legacy outer-replan state for the portfolio tool."""
    request = state.get("portfolio_replan_request")
    if isinstance(request, PortfolioReplanRequest):
        return request

    # ``portfolio_escalation`` was the pre-contract dictionary.  Accept it
    # during migration, but validate only the fields in the typed boundary
    # rather than leaking arbitrary evidence into solver state.
    legacy = request if isinstance(request, dict) else state.get("portfolio_escalation")
    if not isinstance(legacy, dict):
        return None
    fields = (
        "reason",
        "peer_conflict_pairs",
        "forced_singleton_task_ids",
        "triggering_attempt_id",
        "triggering_scan_id",
        "source_portfolio_plan_id",
    )
    payload = {key: legacy[key] for key in fields if key in legacy}
    if not payload.get("reason"):
        payload["reason"] = "PORTFOLIO_REPLAN"
    try:
        return PortfolioReplanRequest.model_validate(payload)
    except Exception as exc:  # noqa: BLE001 - migration input is best effort
        log.warning("Ignoring invalid legacy portfolio escalation: %s", exc)
        return None


def _stable_portfolio_replan_reason(reason: str) -> str:
    """Normalize volatile identifiers before using a replan reason as a key.

    Portfolio IDs and task revisions are expected to change when a plan is
    rebuilt.  They therefore cannot distinguish progress from a repeated
    request.  The user-facing request keeps its full diagnostic text; only the
    internal history key is normalized and sorted.
    """
    normalized = re.sub(
        r"portfolio plan '[^']+' differs from committed '[^']+'",
        "portfolio plan differs from committed",
        reason,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(
        r"is older than committed planned revision \d+ \(current=\d+\)",
        "is older than committed planned revision",
        normalized,
        flags=re.IGNORECASE,
    )
    clauses = [re.sub(r"\s+", " ", clause).strip() for clause in normalized.split(";")]
    return "; ".join(sorted(clause for clause in clauses if clause))


def _record_portfolio_replan_attempt(
    state: OrchestratorState,
    request: PortfolioReplanRequest | None,
) -> tuple[dict[str, int], str | None]:
    """Record a replan request and report an identical-reason loop.

    ``run_portfolio_plan_tool`` is the state boundary that consumes a
    ``PortfolioReplanRequest``.  Keeping the ledger here means every source
    of a replan, including the legacy escalation projection, gets the same
    bounded fail-closed behavior.

    Args:
        state: Current orchestrator state containing the prior ledger.
        request: Normalized outer-replan request, if this is a replan.

    Returns:
        A replacement ledger and an optional diagnostic.  The diagnostic is
        non-empty only when another identical request would exceed the limit.
    """
    history: dict[str, int] = {}
    for raw_reason, raw_count in dict(state.get("portfolio_replan_history", {}) or {}).items():
        if not isinstance(raw_reason, str):
            continue
        reason_key = _stable_portfolio_replan_reason(raw_reason)
        try:
            history[reason_key] = max(history.get(reason_key, 0), int(raw_count))
        except (TypeError, ValueError):
            history.setdefault(reason_key, 0)
    if request is None:
        return history, None

    reason = request.reason.strip() or "PORTFOLIO_REPLAN"
    reason_key = _stable_portfolio_replan_reason(reason)
    attempts = history.get(reason_key, 0)
    portfolio_iteration = int(state.get("portfolio_iteration", 0) or 0)
    if attempts >= MAX_PORTFOLIO_REPLAN_ATTEMPTS or portfolio_iteration >= MAX_PORTFOLIO_ITERATIONS:
        return (
            history,
            "portfolio replan guard: refusing another portfolio iteration after "
            f"{attempts} identical requests or {portfolio_iteration} total iterations "
            f"(reason={reason!r}).",
        )
    history[reason_key] = attempts + 1
    return history, None


def _portfolio_certificate_violations(
    plan: Any,
    task_queue: Mapping[str, Any],
    *,
    accept_feasible: bool = False,
) -> list[str]:
    """Validate the resolver certificate against its committed immutable plan."""
    violations: list[str] = []
    solver_plan = getattr(plan, "solver_plan", None)
    certificate = getattr(plan, "resolution_certificate", None)
    if certificate is None:
        return ["portfolio plan is missing its package-resolution certificate"]
    certificate_status = getattr(getattr(certificate, "status", None), "value", None)
    if (
        str(certificate_status or getattr(certificate, "status", "")).upper()
        != PackageResolutionStatus.CERTIFIED.value
    ):
        violations.append("package-resolution certificate is not CERTIFIED")
    if solver_plan is None:
        return sorted(set([*violations, "portfolio plan is missing its solver plan"]))
    status = getattr(solver_plan, "status", None)
    status = getattr(status, "value", status)
    solver_status = str(status or "").upper()
    if solver_status != "OPTIMAL" and not (accept_feasible and solver_status == "FEASIBLE"):
        violations.append("certified dispatch requires an accepted solver status")
    if not getattr(solver_plan, "candidate_catalog_complete", False):
        violations.append("certified dispatch requires a complete candidate catalog")
    if getattr(certificate, "portfolio_plan_id", None) != getattr(plan, "portfolio_plan_id", None):
        violations.append("package-resolution certificate portfolio plan ID is stale")
    if getattr(certificate, "solver_input_digest", None) != getattr(
        plan, "solver_input_digest", None
    ):
        violations.append("package-resolution certificate solver input digest is stale")
    if getattr(certificate, "repository_fingerprint", None) != getattr(
        plan, "repository_fingerprint", None
    ):
        violations.append("package-resolution certificate host fingerprint is stale")
    if getattr(certificate, "workspace_graph_digest", None) != getattr(
        plan, "workspace_graph_digest", None
    ):
        violations.append("package-resolution certificate workspace graph digest is stale")
    if getattr(certificate, "candidate_catalog_digest", None) != getattr(
        solver_plan, "candidate_catalog_digest", None
    ):
        violations.append("package-resolution certificate candidate catalog digest is stale")
    revisions = dict(getattr(plan, "task_revisions", {}) or {})
    if dict(getattr(certificate, "task_revisions", {}) or {}) != revisions:
        violations.append("package-resolution certificate task revisions are stale")
    for task_id, revision in revisions.items():
        task = task_queue.get(task_id)
        if task is None or task.task_revision != revision:
            violations.append(f"package-resolution certificate revision is stale for {task_id!r}")

    selected = getattr(solver_plan, "selected_plan", None)
    if selected is None:
        violations.append("certified dispatch requires a selected solver assignment")
        return sorted(set(violations))
    if getattr(certificate, "candidate_plan_id", None) != getattr(
        selected, "candidate_plan_id", None
    ):
        violations.append("package-resolution certificate candidate plan ID is stale")
    assignment = dict(sorted((getattr(selected, "selected_candidate_versions", {}) or {}).items()))
    assignment_digest = hashlib.sha256(
        json.dumps(assignment, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
    if getattr(certificate, "candidate_assignment_digest", None) != assignment_digest:
        violations.append("package-resolution certificate selected assignment is stale")
    batches = list(getattr(selected, "batches", ()) or ())
    expected_covered = {
        coverage_id for batch in batches for coverage_id in batch.resolved_coverage_ids
    }
    expected_workaround = {
        coverage_id for batch in batches for coverage_id in batch.workaround_coverage_ids
    }
    expected_all = {
        coverage_id
        for batch in batches
        for coverage_id in (
            batch.resolved_coverage_ids
            + batch.workaround_coverage_ids
            + batch.unresolved_coverage_ids
        )
    }
    certificate_covered = set(getattr(certificate, "covered_coverage_ids", ()) or ())
    certificate_workaround = set(getattr(certificate, "workaround_coverage_ids", ()) or ())
    certificate_unresolved = set(getattr(certificate, "unresolved_coverage_ids", ()) or ())
    if certificate_covered != expected_covered:
        violations.append("package-resolution certificate covered-coverage set is stale")
    if certificate_workaround != expected_workaround:
        violations.append("package-resolution certificate workaround-coverage set is stale")
    if certificate_covered | certificate_workaround | certificate_unresolved != expected_all:
        violations.append("package-resolution certificate coverage universe is stale")
    return sorted(set(violations))


def run_portfolio_plan_tool(state: OrchestratorState) -> dict[str, Any]:
    """Prepare, solve, and commit one copy-on-write outer portfolio plan.

    The deterministic Supervisor invokes this synchronous operation for
    setup/triage entries and existing replan decisions. Planning failures
    inside the transaction are returned as status patches. The input state
    mapping is not mutated.

    Args:
        state: Orchestrator state for the current portfolio iteration.

    Returns:
        A state patch containing the plan result and its diagnostics.

    Notes:
        Certification may fetch registry metadata and use a temporary
        certification workspace.
    """
    iteration = int(state.get("portfolio_iteration", 0) or 0) + 1
    request = _portfolio_replan_request(state)
    replan_history, replan_guard_error = _record_portfolio_replan_attempt(state, request)
    if replan_guard_error:
        log.warning("run_portfolio_plan_tool: %s", replan_guard_error)
        return {
            "status": "portfolio_replan_guarded",
            "next_routing_step": "teardown",
            "portfolio_plan": state.get("portfolio_plan"),
            "portfolio_solver_plan": state.get("portfolio_solver_plan"),
            "portfolio_iteration": iteration,
            "portfolio_replan_request": None,
            "portfolio_replan_history": replan_history,
            "portfolio_dirty": False,
            "portfolio_escalation": None,
            "valid_groups": list(state.get("valid_groups", []) or []),
            "task_queue": dict(state.get("task_queue", {}) or {}),
            "active_target_task_ids": [],
            "active_cluster_id": None,
            "active_dispatch_batch_id": None,
            "active_multi_package_action": None,
            "errors": [replan_guard_error],
        }
    peer_conflict_pairs = request.peer_conflict_pairs if request else ()
    forced_singletons = request.forced_singleton_task_ids if request else ()

    try:
        settings = get_runtime_settings()
        project_language = resolve_project_language(
            state["repo_root"], str(state.get("project_language") or "") or None
        )
        prepared_groups, prepared_queue, prepare_diagnostics = prepare_portfolio_inputs(
            state["repo_root"],
            state.get("valid_groups", []),
            dict(state.get("task_queue", {}) or {}),
            target_packages=state.get("target_packages", []),
            project_language=project_language,
        )
        source_queue = dict(state.get("task_queue", {}) or {})
        portfolio_queue: dict[str, Any] = {}
        for task_id, task in prepared_queue.items():
            source_task = source_queue.get(task_id)
            # Preserve preparation-time triage refreshes, which advance the
            # task revision. Discard same-revision solver candidates until a
            # solver-approved decision commits them.
            portfolio_queue[task_id] = (
                task.model_copy(deep=True)
                if source_task is None or source_task.task_revision != task.task_revision
                else source_task.model_copy(deep=True)
            )
        if prepared_groups and not any(
            group.issue_type == IssueType.SCA for group in prepared_groups
        ):
            return {
                "status": "portfolio_ready",
                "next_routing_step": "supervisor",
                "portfolio_plan": None,
                "portfolio_solver_plan": None,
                "portfolio_iteration": iteration,
                "portfolio_replan_request": None,
                "portfolio_replan_history": replan_history,
                "portfolio_escalation": None,
                "portfolio_dirty": False,
                "valid_groups": prepared_groups,
                "task_queue": prepared_queue,
                "active_target_task_ids": [],
                "active_cluster_id": None,
                "active_dispatch_batch_id": None,
                "errors": sorted(set(prepare_diagnostics)),
            }
        plan = build_certified_portfolio_plan(
            state["repo_root"],
            state.get("workspace_volume") or "",
            prepared_groups,
            prepared_queue,
            target_packages=state.get("target_packages", []),
            peer_conflict_pairs=peer_conflict_pairs,
            forced_singleton_task_ids=forced_singletons,
            settings=settings,
            registry_fetcher=_fetch_fresh_registry_packument,
            portfolio_iteration=iteration,
            portfolio_replan_request=request,
            prior_portfolio_plan=state.get("portfolio_plan") if request is not None else None,
            qa_results_by_attempt=state.get("qa_results_by_attempt") or {},
            attempt_snapshots_by_id=state.get("attempt_snapshots_by_id") or {},
            retry_diagnostics_by_task=state.get("retry_diagnostics_by_task") or {},
            qa_passed_workspace_prefix=state.get("qa_passed_workspace_prefix"),
            project_language=project_language,
        )
    except Exception as exc:  # noqa: BLE001 - fail closed at the transaction boundary
        return {
            "status": "portfolio_failed",
            "next_routing_step": "teardown",
            "portfolio_dirty": False,
            "active_target_task_ids": [],
            "portfolio_iteration": iteration,
            "portfolio_replan_request": request,
            "portfolio_replan_history": replan_history,
            "errors": [f"portfolio node failed: {exc}"],
        }

    solver_plan = getattr(plan, "solver_plan", None)
    solver_plan_diagnostics = list(getattr(solver_plan, "diagnostics", []) or [])
    plan_diagnostics = list(getattr(plan, "diagnostics", []) or [])
    diagnostics = sorted(
        set(prepare_diagnostics) | set(plan_diagnostics) | set(solver_plan_diagnostics)
    )
    diagnostics_text = "\n".join(diagnostics).lower()
    solver_status = getattr(solver_plan, "status", None)
    solver_status = getattr(solver_status, "value", solver_status)
    solver_status = str(solver_status or "").upper()
    accept_feasible = bool(getattr(settings, "solver_accept_feasible", True))
    certificate_violations = _portfolio_certificate_violations(
        plan, portfolio_queue, accept_feasible=accept_feasible
    )
    diagnostics = sorted(set(diagnostics) | set(certificate_violations))
    invalid_dag = any(
        marker in diagnostics_text
        for marker in (
            "invalid dag",
            "invalid-dag",
            "invalid dependency dag",
            "dag validation failed",
            "unknown endpoint",
            "unresolved cycle",
            "partial result",
        )
    )
    all_non_dispatchable_clusters = [
        cluster
        for cluster in getattr(plan, "clusters", [])
        if not getattr(cluster, "dispatchable", True)
    ]
    hard_overflow_clusters = [
        cluster.cluster_id
        for cluster in all_non_dispatchable_clusters
        if str(getattr(cluster, "reason", "")).lower().startswith("hard atomic component exceeds ")
    ]
    if all_non_dispatchable_clusters:
        diagnostics.append(
            "portfolio plan contains retained non-dispatchable cluster(s): "
            f"{sorted(cluster.cluster_id for cluster in all_non_dispatchable_clusters)!r}"
        )
        diagnostics_text = "\n".join(diagnostics).lower()
    multiple_active_tasks = any(
        "multiple nonterminal tasks" in diagnostic for diagnostic in plan_diagnostics
    )
    blocked = multiple_active_tasks or bool(hard_overflow_clusters) or bool(certificate_violations)
    failed_status = solver_status != "OPTIMAL" and not (
        accept_feasible and solver_status == "FEASIBLE"
    )
    if blocked or invalid_dag or failed_status:
        errors = list(diagnostics)
        if multiple_active_tasks:
            errors.append(
                "portfolio node blocked dispatch because a package group has multiple "
                "nonterminal active tasks."
            )
        if hard_overflow_clusters:
            errors.append(
                "portfolio node blocked dispatch because hard atomic component(s) "
                "exceed the multi-package action limit: "
                f"{sorted(hard_overflow_clusters)!r}."
            )
        if failed_status:
            errors.append(f"portfolio solver returned non-dispatchable status {solver_status}.")
        failure_state = (
            "portfolio_native_error"
            if solver_status == "NATIVE_ERROR"
            else "portfolio_unknown"
            if solver_status in {"UNKNOWN", "FEASIBLE"}
            else "portfolio_infeasible"
            if solver_status == "INFEASIBLE"
            else "portfolio_invalid"
            if invalid_dag or solver_status in {"INVALID", "FALLBACK"}
            else "portfolio_unknown"
            if certificate_violations
            else "portfolio_blocked"
        )
        return {
            "status": failure_state,
            "next_routing_step": "teardown",
            "portfolio_plan": plan,
            "portfolio_solver_plan": solver_plan,
            "portfolio_iteration": iteration,
            "portfolio_replan_request": request,
            "portfolio_replan_history": replan_history,
            "portfolio_dirty": False,
            "valid_groups": prepared_groups,
            "task_queue": portfolio_queue,
            "active_target_task_ids": [],
            "active_cluster_id": None,
            "active_dispatch_batch_id": None,
            "active_multi_package_action": None,
            "errors": errors,
        }
    try:
        committed_groups, committed_queue, apply_diagnostics = apply_portfolio_plan(
            plan,
            prepared_groups,
            portfolio_queue,
        )
    except Exception as exc:  # noqa: BLE001 - fail closed at the transaction boundary
        return {
            "status": "portfolio_failed",
            "next_routing_step": "teardown",
            "portfolio_plan": plan,
            "portfolio_solver_plan": solver_plan,
            "portfolio_iteration": iteration,
            "portfolio_replan_request": request,
            "portfolio_replan_history": replan_history,
            "portfolio_dirty": False,
            "valid_groups": prepared_groups,
            "task_queue": portfolio_queue,
            "active_target_task_ids": [],
            "errors": [f"portfolio plan application failed: {exc}"],
        }
    diagnostics = sorted(set(diagnostics) | set(apply_diagnostics))
    deferred_active_task_ids = sorted(
        task_id
        for task_id, task in portfolio_queue.items()
        if task.current_attempt_id is not None
        and any(
            diagnostic == f"task {task_id!r} has an active attempt; plan decision not applied"
            for diagnostic in apply_diagnostics
        )
    )
    if deferred_active_task_ids:
        # Applying the plan to only the tasks without live attempts would
        # create a mixed-generation portfolio: the plan revision would advance
        # while the active tasks still point at the previous plan.  Preserve the
        # previous committed plan and send control back to Supervisor so the
        # attempt-tagged worker/QA envelopes can be reconciled first.
        reconcile_reason = (
            "portfolio plan application deferred until active attempts are reconciled: "
            f"{', '.join(deferred_active_task_ids)}"
        )
        reconcile_request = request or PortfolioReplanRequest(
            reason=reconcile_reason,
            source_portfolio_plan_id=(
                getattr(plan, "portfolio_plan_id", None) or getattr(plan, "plan_id", None)
            ),
        )
        active_attempt_task_ids = [
            task_id
            for task_id, task in portfolio_queue.items()
            if task.current_attempt_id is not None and task.status not in TERMINAL_TASK_STATUSES
        ]
        return {
            "status": "portfolio_reconciliation_required",
            "next_routing_step": "supervisor",
            "portfolio_plan": state.get("portfolio_plan"),
            "portfolio_solver_plan": state.get("portfolio_solver_plan"),
            "portfolio_iteration": iteration,
            "portfolio_replan_request": reconcile_request,
            "portfolio_replan_history": replan_history,
            "portfolio_dirty": True,
            "portfolio_escalation": state.get("portfolio_escalation"),
            "valid_groups": prepared_groups,
            "task_queue": portfolio_queue,
            "active_target_task_ids": active_attempt_task_ids,
            "active_cluster_id": None,
            "active_dispatch_batch_id": None,
            "active_multi_package_action": None,
            "errors": [*diagnostics, reconcile_reason],
        }
    result_state = {
        "status": "portfolio_ready",
        "next_routing_step": "supervisor",
        "portfolio_plan": plan,
        "portfolio_solver_plan": solver_plan,
        "portfolio_iteration": iteration,
        "portfolio_replan_request": None,
        "portfolio_replan_history": replan_history,
        "portfolio_dirty": False,
        "portfolio_escalation": None,
        "valid_groups": committed_groups,
        "task_queue": committed_queue,
        "active_target_task_ids": [],
        "active_cluster_id": None,
        "active_dispatch_batch_id": None,
        "active_multi_package_action": None,
        "errors": diagnostics,
    }
    certified_prefix = getattr(
        getattr(plan, "resolution_certificate", None),
        "workspace_prefix_provenance",
        None,
    )
    result_state["qa_passed_workspace_prefix"] = certified_prefix
    return result_state


def route_after_portfolio(state: OrchestratorState) -> str:
    """Route ready plans and attempt-reconciliation states to the Supervisor."""
    return (
        "supervisor"
        if state.get("status")
        in {
            "portfolio_ready",
            "portfolio_reconciliation_required",
        }
        else "teardown"
    )
