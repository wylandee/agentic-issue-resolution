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
MAX_QA_EVIDENCE_RERUNS : int
    Maximum same-attempt QA reruns for inconclusive evidence or evaluator errors.
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
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from remediation_engine.contracts.decision_codes import (
    DecisionCode,
    validate_transition,
)
from remediation_engine.contracts.schemas import (
    AgentActionStatus,
    AgentActionSummary,
    FailureCategory,
    FinalFullScanResult,
    IncrementalScanRecord,
    IssueType,
    MavenTargetOperation,
    MultiPackageAction,
    NoFixMitigationStage,
    ODCScanEvidence,
    PackageMutation,
    PackageScanOccurrence,
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
    TaskScanRecord,
    TaskSpawnRequest,
    TaskStatus,
    UpdateRetryDiagnostics,
    VulnerabilityGroup,
    VulnerabilityIssue,
    WorkaroundContext,
    WorkaroundPhase,
    WorkaroundReplayPlan,
    WorkerAttemptResult,
)
from remediation_engine.contracts.solver_models import (
    PackageResolutionStatus,
    PortfolioReplanRequest,
)
from remediation_engine.contracts.version_policy import (
    compare_maven_versions,
    is_stable_maven_version,
)
from remediation_engine.language import ProjectLanguage
from remediation_engine.orchestration import _supervisor_execution as _supervisor_execution_helpers
from remediation_engine.orchestration._qa_runtime import (
    _lockfile_paths_for_group,
    _parse_package_occurrence_key,
)
from remediation_engine.orchestration._tool_support import _resolve_maven_manifest_scope
from remediation_engine.orchestration.issue_identity import stable_issue_fingerprint
from remediation_engine.orchestration.runtime_context import get_runtime_settings
from remediation_engine.orchestration.scan_identity import package_scan_occurrence_key
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
    MAX_QA_EVIDENCE_RERUNS,
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
from remediation_engine.orchestration.supervisor_portfolio import (
    route_after_portfolio,
    run_portfolio_plan_tool,
)
from remediation_engine.orchestration.supervisor_routing import (
    _build_consistency_event,
    _build_workaround_retry_instruction,
    _dedupe_consistency_events,
    _deterministic_routing,
    _emit_audit,
    _no_fix_decision_requires_fallback,
)
from remediation_engine.orchestration.supervisor_spawn import (
    _materialize_spawn_requests,
    _plan_initial_transitive_task,
    _reconcile_terminal_pivot_parents,
    _terminalize_pivot_parents,
)
from remediation_engine.orchestration.tactical_supervisor import (
    _approved_candidate_pool,
    _portfolio_escalation_required,
    build_tactical_context,
    classify_diagnostics,
    propose_and_verify_tactical_action,
    registry_candidate_sets_for_context,
)
from remediation_engine.orchestration.task_utils import (
    advance_no_fix_stage,
    build_no_fix_package_removal_instruction,
    build_no_fix_retry_instruction,
    derive_missing_task_qa_policy,
    group_parent_context,
    is_maven_group,
    is_transitive_group,
    select_package_fix_plan,
)
from remediation_engine.runtime.path_policy import WorkspacePathError, normalize_workspace_path
from remediation_engine.tools.npm_graph import (
    load_npm_graph_snapshot,
    lockfile_key_matches_package,
    make_occurrence_id,
    npm_range_contains,
)

logger = logging.getLogger(__name__)

_VERIFIED_ABSENT_REMEDIATION_ACTION = (
    "Removed due to another package upgrade; target absent from every covered package-lock input."
)
_VERIFIED_ABSENT_PORTFOLIO_REPLAN_REASON = (
    "Verified target-package absence changed portfolio membership."
)

# Supervisor dispatch commits one task attempt at a time. Worker and QA
# boundaries consume only the typed task/attempt envelopes.

_VALID_NEXT_NODES: set[str] = {
    "supervisor",
    "update_subagent",
    "workaround_subagent",
    "qa_critic",
    "final_full_scan",
    "triage",
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


def _final_full_scan_due(
    state: OrchestratorState,
    task_queue: Mapping[str, RemediationTask],
) -> bool:
    """Return whether the enabled final full scan still needs to run."""
    if (
        not state.get("final_full_scan_enabled")
        or not state.get("workspace_volume")
        or not task_queue
        or any(task.status not in _TERMINAL_STATUSES for task in task_queue.values())
    ):
        return False
    previous_scan = state.get("final_full_scan_result")
    return _final_full_scan_value(previous_scan, "coverage_scope") != "full_workspace"


def _final_full_scan_value(value: Any, field: str, default: Any = None) -> Any:
    """Read a final scan result field from a model or serialized mapping."""
    if isinstance(value, Mapping):
        return value.get(field, default)
    return getattr(value, field, default)


def _task_scan_record_from_qa_result(
    qa_result: QAAttemptResult,
) -> TaskScanRecord | None:
    """Build a task record only from a complete, passing attempt envelope."""
    evaluation = qa_result.evaluation
    evidence = evaluation.scan_evidence
    if (
        not evaluation.passed
        or evaluation.contract_error
        or evaluation.evidence_inconclusive
        or evidence is None
        or not evidence.complete
    ):
        return None
    if (
        not evidence.scan_performed
        and evidence.scan_skip_reason != "no_matching_target"
        and (evidence.inventory_updates or not evidence.removed_occurrence_keys)
    ):
        return None
    return TaskScanRecord(
        task_id=qa_result.task_id,
        attempt_id=qa_result.attempt_id,
        task_revision=qa_result.task_revision,
        new_identifiers=list(evidence.new_identifiers),
        found_identifiers=list(evidence.found_identifiers),
        remaining_target_identifiers=list(evidence.remaining_target_identifiers),
        found_issues=list(evidence.found_issues),
        closure_package_names=list(evidence.closure_package_names),
        closure_lockfile_keys=list(evidence.closure_lockfile_keys),
        covered_input_paths=list(evidence.covered_input_paths),
        covered_occurrences=list(evidence.covered_occurrences),
        removed_occurrence_keys=list(evidence.removed_occurrence_keys),
        inventory_updates=dict(evidence.inventory_updates),
        complete=evidence.complete,
    )


def _verified_absent_package_tasks(
    task_queue: Mapping[str, RemediationTask],
    group_by_id: Mapping[str, VulnerabilityGroup],
    latest_successful_scan_by_task: Mapping[str, TaskScanRecord],
    accepted_package_inventory_by_lockfile: Mapping[str, str],
) -> dict[str, str]:
    """Find untouched SCA tasks whose target is proven absent from covered lockfiles.

    A candidate requires a complete accepted scan record covering every
    supported package-lock input, a valid target-removal tombstone in those
    inputs, and no current inventory occurrence of the target package.

    Args:
        task_queue: Authoritative tasks eligible for Supervisor reconciliation.
        group_by_id: Vulnerability groups keyed by their stable group IDs.
        latest_successful_scan_by_task: Complete QA-passed scan records accepted
            by the Supervisor.
        accepted_package_inventory_by_lockfile: Current physical package-lock
            occurrence inventory after accepted scan deltas.

    Returns:
        Task IDs mapped to target package names when absence is fully verified.
    """

    def normalized_input_path(value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        normalized = value.replace("\\", "/")
        parts = normalized.split("/")
        if (
            not normalized
            or normalized != value
            or normalized.startswith("/")
            or any(part in {"", ".", ".."} for part in parts)
        ):
            return None
        return normalized

    inventory_packages_by_path: dict[str, set[str]] = {}
    for occurrence_key in accepted_package_inventory_by_lockfile:
        occurrence = _parse_package_occurrence_key(occurrence_key)
        if occurrence is None:
            return {}
        source_path, _lockfile_key, package_name, _version, _index = occurrence
        inventory_packages_by_path.setdefault(source_path, set()).add(package_name)

    verified: dict[str, str] = {}
    for task_id, task in sorted(task_queue.items()):
        if (
            task.is_synthetic
            or task.status not in {TaskStatus.PENDING, TaskStatus.NEEDS_RETRY}
            or task.current_attempt_id is not None
        ):
            continue
        group = group_by_id.get(task.parent_group_id)
        if group is None or group.issue_type != IssueType.SCA:
            continue
        target_package_name = (task.target_package_name or group.vulnerable_component or "").strip()
        if not target_package_name:
            continue
        target_package_key = target_package_name.casefold()

        lockfile_paths = _lockfile_paths_for_group(group)
        if not lockfile_paths:
            continue
        normalized_lockfile_paths: set[str] = set()
        paths_supported = True
        for raw_path in lockfile_paths:
            path = normalized_input_path(raw_path)
            if path is None or path.rsplit("/", 1)[-1].casefold() != "package-lock.json":
                paths_supported = False
                break
            normalized_lockfile_paths.add(path)
        if not paths_supported or not normalized_lockfile_paths:
            continue

        covered_paths: set[str] = set()
        has_target_tombstone = False
        malformed_record = False
        for record_task_id, scan_record in latest_successful_scan_by_task.items():
            if not scan_record.complete or scan_record.task_id != record_task_id:
                continue
            record_paths: set[str] = set()
            for raw_path in scan_record.covered_input_paths:
                path = normalized_input_path(raw_path)
                if path is None:
                    malformed_record = True
                    break
                record_paths.add(path)
            if malformed_record:
                break

            for tombstone in scan_record.removed_occurrence_keys:
                occurrence = _parse_package_occurrence_key(tombstone)
                if occurrence is None:
                    if record_paths.intersection(normalized_lockfile_paths):
                        malformed_record = True
                        break
                    continue
                source_path, _lockfile_key, package_name, _version, _index = occurrence
                if source_path not in normalized_lockfile_paths:
                    continue
                if source_path not in record_paths:
                    malformed_record = True
                    break
                if package_name == target_package_key:
                    has_target_tombstone = True
            if malformed_record:
                break
            covered_paths.update(record_paths.intersection(normalized_lockfile_paths))

        if (
            malformed_record
            or covered_paths != normalized_lockfile_paths
            or not has_target_tombstone
            or any(
                target_package_key in package_names
                for package_names in inventory_packages_by_path.values()
            )
        ):
            continue
        verified[task_id] = target_package_name

    return verified


def _incremental_scan_records_from_evidence(
    evidence: ODCScanEvidence,
    *,
    task_ids: list[str],
    attempt_ids: list[str],
    baseline_scan_identifiers: Iterable[str],
) -> dict[str, IncrementalScanRecord]:
    """Build report-wide snapshots for each fully scanned input path.

    Args:
        evidence: Shared QA scan evidence, including report-wide occurrences.
        task_ids: Passing tasks whose candidate changes were covered.
        attempt_ids: Current QA attempts that produced the shared scan.
        baseline_scan_identifiers: Immutable pre-remediation identifiers.

    Returns:
        Complete snapshots keyed by source input path. Incomplete or skipped
        scans return no snapshots so they cannot erase earlier observations.
    """
    if not evidence.complete or not evidence.scan_performed or evidence.scan_skip_reason:
        return {}

    input_paths = sorted(
        {
            path.strip().replace("\\", "/")
            for path in evidence.covered_input_paths
            if isinstance(path, str) and path.strip()
        }
    )
    if not input_paths:
        return {}

    found_by_path: dict[str, set[str]] = {path: set() for path in input_paths}
    issues_by_path: dict[str, dict[str, VulnerabilityIssue]] = {path: {} for path in input_paths}
    for occurrence in evidence.covered_occurrences:
        input_path = occurrence.source_input_path.strip().replace("\\", "/")
        if input_path not in found_by_path:
            continue
        found_by_path[input_path].update(
            identifier.upper().strip()
            for identifier in occurrence.found_identifiers
            if isinstance(identifier, str) and identifier.strip()
        )
        for issue in occurrence.found_issues:
            issues_by_path[input_path].setdefault(stable_issue_fingerprint(issue), issue)

    if len(input_paths) == 1:
        input_path = input_paths[0]
        found_by_path[input_path].update(
            identifier.upper().strip()
            for identifier in evidence.found_identifiers
            if isinstance(identifier, str) and identifier.strip()
        )
        for issue in evidence.found_issues:
            issues_by_path[input_path].setdefault(stable_issue_fingerprint(issue), issue)
    else:
        mapped_identifiers = set().union(*found_by_path.values())
        unmapped_identifiers = {
            identifier.upper().strip()
            for identifier in evidence.found_identifiers
            if isinstance(identifier, str) and identifier.strip()
        } - mapped_identifiers
        if unmapped_identifiers:
            # Preserve every parsed identifier even if its report occurrence
            # could not be split across multiple scanned inputs.
            found_by_path[input_paths[0]].update(unmapped_identifiers)

    baseline = {
        identifier.upper().strip()
        for identifier in baseline_scan_identifiers
        if isinstance(identifier, str) and identifier.strip()
    }
    task_ids = sorted(set(task_ids))
    attempt_ids = sorted(set(attempt_ids))
    return {
        input_path: IncrementalScanRecord(
            source_input_path=input_path,
            task_ids=task_ids,
            attempt_ids=attempt_ids,
            found_identifiers=sorted(found_by_path[input_path]),
            new_identifiers=sorted(found_by_path[input_path] - baseline),
            found_issues=[
                issues_by_path[input_path][fingerprint]
                for fingerprint in sorted(issues_by_path[input_path])
            ],
            complete=True,
        )
        for input_path in input_paths
    }


def _scan_ledger_state_snapshot(state: OrchestratorState) -> dict[str, Any]:
    """Return complete copies for the four authoritative scan ledger maps."""
    return {
        "accepted_package_inventory_by_lockfile": dict(
            state.get("accepted_package_inventory_by_lockfile", {}) or {}
        ),
        "current_scan_observations_by_occurrence": dict(
            state.get("current_scan_observations_by_occurrence", {}) or {}
        ),
        "latest_successful_scan_by_task": dict(
            state.get("latest_successful_scan_by_task", {}) or {}
        ),
        "latest_incremental_scan_by_input": dict(
            state.get("latest_incremental_scan_by_input", {}) or {}
        ),
        "setup_scan_evidence": state.get("setup_scan_evidence"),
        "setup_scan_committed": bool(state.get("setup_scan_committed", False)),
    }


def _prune_setup_scan_evidence(
    setup_evidence: ODCScanEvidence | None,
    scan_record: TaskScanRecord | None,
    baseline_scan_identifiers: Iterable[str],
    *,
    superseded_input_paths: Iterable[str] = (),
) -> ODCScanEvidence | None:
    """Remove setup findings superseded by a later complete scan."""
    if setup_evidence is None:
        return None
    normalized_superseded_input_paths = {
        path.strip().replace("\\", "/")
        for path in superseded_input_paths
        if isinstance(path, str) and path.strip()
    }
    if not setup_evidence.complete and not normalized_superseded_input_paths:
        return setup_evidence
    superseded_keys = set(scan_record.removed_occurrence_keys) if scan_record is not None else set()
    if scan_record is not None:
        superseded_keys.update(
            package_scan_occurrence_key(
                occurrence.source_input_path,
                occurrence.lockfile_key,
                occurrence.package_name,
                occurrence.version,
                occurrence.occurrence_index,
            )
            for occurrence in scan_record.covered_occurrences
        )
    if not superseded_keys and not normalized_superseded_input_paths:
        return setup_evidence
    remaining_occurrences = [
        occurrence
        for occurrence in setup_evidence.covered_occurrences
        if occurrence.source_input_path.replace("\\", "/") not in normalized_superseded_input_paths
        and package_scan_occurrence_key(
            occurrence.source_input_path,
            occurrence.lockfile_key,
            occurrence.package_name,
            occurrence.version,
            occurrence.occurrence_index,
        )
        not in superseded_keys
    ]
    found_identifiers = sorted(
        {
            identifier.upper().strip()
            for occurrence in remaining_occurrences
            for identifier in occurrence.found_identifiers
            if isinstance(identifier, str) and identifier.strip()
        }
    )
    baseline = {
        identifier.upper().strip()
        for identifier in baseline_scan_identifiers
        if isinstance(identifier, str) and identifier.strip()
    }
    issues_by_fingerprint: dict[str, VulnerabilityIssue] = {}
    for occurrence in remaining_occurrences:
        for issue in occurrence.found_issues:
            issues_by_fingerprint.setdefault(stable_issue_fingerprint(issue), issue)
    remaining_paths = {occurrence.source_input_path for occurrence in remaining_occurrences}
    covered_input_paths = [
        path
        for path in setup_evidence.covered_input_paths
        if path.replace("\\", "/") not in normalized_superseded_input_paths
        and path in remaining_paths
    ]
    if not remaining_occurrences and not covered_input_paths:
        return None
    return setup_evidence.model_copy(
        update={
            "covered_input_paths": covered_input_paths,
            "covered_occurrences": remaining_occurrences,
            "closure_package_names": sorted(
                {occurrence.package_name for occurrence in remaining_occurrences}
            ),
            "closure_lockfile_keys": sorted(
                {
                    occurrence.lockfile_key
                    for occurrence in remaining_occurrences
                    if occurrence.lockfile_key is not None
                }
            ),
            "found_identifiers": found_identifiers,
            "new_identifiers": sorted(set(found_identifiers) - baseline),
            "found_issues": [
                issues_by_fingerprint[fingerprint] for fingerprint in sorted(issues_by_fingerprint)
            ],
        }
    )


def _calculate_cumulative_scan_metrics(
    task_queue: Mapping[str, RemediationTask],
    scan_records: Mapping[str, TaskScanRecord],
    baseline_scan_identifiers: Iterable[str],
    setup_scan_evidence: ODCScanEvidence | None = None,
    incremental_scan_records_by_input: Mapping[str, IncrementalScanRecord] | None = None,
) -> tuple[list[str], list[str], list[str], list[VulnerabilityIssue]]:
    """Aggregate current task and workspace-setup scan observations.

    Args:
        task_queue: Current task state; only ``QA_PASSED`` tasks contribute.
        scan_records: Latest accepted task-scoped QA snapshots.
        baseline_scan_identifiers: Immutable pre-remediation ODC identifier set.
        setup_scan_evidence: Attempt-independent scan evidence from workspace setup.
        incremental_scan_records_by_input: Latest report-wide snapshot per input path.

    Returns:
        Sorted newly discovered identifiers, all observed identifiers,
        remaining task-target identifiers, and deduplicated findings.
    """
    baseline = {
        identifier.upper().strip()
        for identifier in baseline_scan_identifiers
        if isinstance(identifier, str) and identifier.strip()
    }
    new_identifiers: set[str] = set()
    found_identifiers: set[str] = set()
    remaining_target_identifiers: set[str] = set()
    findings_by_fingerprint: dict[str, VulnerabilityIssue] = {}
    for task_id, record in sorted(scan_records.items()):
        task = task_queue.get(task_id)
        if (
            task is None
            or task.status != TaskStatus.QA_PASSED
            or record.task_id != task_id
            or not record.complete
        ):
            continue
        record_paths = {
            path.strip().replace("\\", "/")
            for path in record.covered_input_paths
            if isinstance(path, str) and path.strip()
        }
        superseded_paths = record_paths & set(incremental_scan_records_by_input or {})
        if superseded_paths and record.covered_occurrences:
            current_occurrences = [
                occurrence
                for occurrence in record.covered_occurrences
                if occurrence.source_input_path.replace("\\", "/") not in superseded_paths
            ]
            found = {
                identifier.upper().strip()
                for occurrence in current_occurrences
                for identifier in occurrence.found_identifiers
                if isinstance(identifier, str) and identifier.strip()
            }
            new = found - baseline
            issues = {
                stable_issue_fingerprint(issue): issue
                for occurrence in current_occurrences
                for issue in occurrence.found_issues
            }
        elif superseded_paths:
            found = set()
            new = set()
            issues = {}
        else:
            found = {
                identifier.upper().strip()
                for identifier in record.found_identifiers
                if isinstance(identifier, str) and identifier.strip()
            }
            new = {
                identifier.upper().strip()
                for identifier in record.new_identifiers
                if isinstance(identifier, str) and identifier.strip()
            }
            issues = {stable_issue_fingerprint(issue): issue for issue in record.found_issues}
        found_identifiers.update(found)
        new_identifiers.update((new | found) - baseline)
        remaining_target_identifiers.update(
            identifier.upper().strip()
            for identifier in record.remaining_target_identifiers
            if isinstance(identifier, str) and identifier.strip()
        )
        for fingerprint, issue in issues.items():
            findings_by_fingerprint.setdefault(fingerprint, issue)
    for input_path, record in sorted((incremental_scan_records_by_input or {}).items()):
        if record.source_input_path != input_path or not record.complete:
            continue
        found = {
            identifier.upper().strip()
            for identifier in record.found_identifiers
            if isinstance(identifier, str) and identifier.strip()
        }
        new = {
            identifier.upper().strip()
            for identifier in record.new_identifiers
            if isinstance(identifier, str) and identifier.strip()
        }
        found_identifiers.update(found)
        new_identifiers.update((new | found) - baseline)
        for issue in record.found_issues:
            findings_by_fingerprint.setdefault(stable_issue_fingerprint(issue), issue)
    if setup_scan_evidence is not None:
        setup_found = {
            identifier.upper().strip()
            for identifier in setup_scan_evidence.found_identifiers
            if isinstance(identifier, str) and identifier.strip()
        }
        setup_new = {
            identifier.upper().strip()
            for identifier in setup_scan_evidence.new_identifiers
            if isinstance(identifier, str) and identifier.strip()
        }
        found_identifiers.update(setup_found)
        new_identifiers.update((setup_new | setup_found) - baseline)
        remaining_target_identifiers.update(
            identifier.upper().strip()
            for identifier in setup_scan_evidence.remaining_target_identifiers
            if isinstance(identifier, str) and identifier.strip()
        )
        for issue in setup_scan_evidence.found_issues:
            findings_by_fingerprint.setdefault(stable_issue_fingerprint(issue), issue)

    return (
        sorted(new_identifiers),
        sorted(found_identifiers),
        sorted(remaining_target_identifiers),
        [findings_by_fingerprint[fingerprint] for fingerprint in sorted(findings_by_fingerprint)],
    )


def _incremental_scan_coverage_complete(
    state: OrchestratorState,
    setup_scan_evidence: ODCScanEvidence | None,
) -> bool:
    """Fail closed when baseline or the latest QA scan evidence is incomplete."""
    if not state.get("baseline_scan_complete", False):
        return False
    if not state.get("package_inventory_baseline_complete", False):
        return False
    if setup_scan_evidence is not None and not setup_scan_evidence.complete:
        return False

    snapshots = state.get("attempt_snapshots_by_id", {}) or {}
    latest_by_task: dict[str, tuple[tuple[int, int, int, str], QAAttemptResult]] = {}
    for attempt_key, qa_result in (state.get("qa_results_by_attempt", {}) or {}).items():
        task_id = qa_result.task_id
        attempt_id = qa_result.attempt_id or attempt_key
        snapshot = snapshots.get(attempt_id)
        try:
            attempt_number = int(snapshot.attempt_number) if snapshot is not None else 0
            state_revision = int(snapshot.state_revision) if snapshot is not None else 0
        except (TypeError, ValueError):
            attempt_number = 0
            state_revision = 0
        order = (qa_result.task_revision, attempt_number, state_revision, attempt_id)
        previous = latest_by_task.get(task_id)
        if previous is None or order >= previous[0]:
            latest_by_task[task_id] = (order, qa_result)

    return all(
        qa_result.evaluation.scan_evidence is None or qa_result.evaluation.scan_evidence.complete
        for _order, qa_result in latest_by_task.values()
    )


__all__ = [
    "MAX_QA_EVIDENCE_RERUNS",
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
    "_terminalize_pivot_parents",
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
    allow_attempted_selected_version: bool = False,
    project_language: ProjectLanguage = ProjectLanguage.NODEJS,
    maven_mode: bool | None = None,
) -> tuple[list[str], list[str]]:
    """Return candidates within the task's solver and retry allowlists."""
    maven_mode = project_language == ProjectLanguage.JAVA and (
        maven_mode
        if maven_mode is not None
        else bool(re.fullmatch(r"[^:\s]+:[^:\s]+", task.target_package_name or ""))
    )

    def normalize_version(value: Any) -> str | None:
        if not isinstance(value, str) or not value.strip():
            return None
        version = value.strip()
        if not maven_mode:
            return version.lstrip("vV")
        if not is_stable_maven_version(version):
            return None
        floor = diagnostics.security_floor if diagnostics is not None else None
        if floor:
            try:
                if not is_stable_maven_version(floor) or compare_maven_versions(version, floor) < 0:
                    return None
            except ValueError:
                return None
        return version

    solver_authoritative = (
        bool(task.portfolio_plan_id)
        or bool(task.allowed_target_versions)
        or bool(task.allowed_dependency_types)
    )
    diagnostics_match_target = diagnostics is not None and (
        not diagnostics.target_package_name
        or diagnostics.target_package_name == task.target_package_name
    )
    approved_versions = list(
        dict.fromkeys(
            normalized
            for value in task.allowed_target_versions
            if (normalized := normalize_version(value))
        )
    )
    if not solver_authoritative:
        if plan is not None and plan.candidate_versions_considered:
            fallback_candidates = plan.candidate_versions_considered
        elif diagnostics is not None and diagnostics_match_target:
            fallback_candidates = diagnostics.candidate_versions_considered
        else:
            fallback_candidates = ()
        approved_versions = list(
            dict.fromkeys(
                normalized
                for value in fallback_candidates
                if (normalized := normalize_version(value))
            )
        )

    approved_version_keys = {version.lower() for version in approved_versions}
    if plan is not None and plan.candidate_versions_considered:
        plan_version_keys = {
            normalized.lower()
            for value in plan.candidate_versions_considered
            if (normalized := normalize_version(value))
        }
        approved_version_keys.intersection_update(plan_version_keys)
    if (
        diagnostics is not None
        and diagnostics_match_target
        and diagnostics.candidate_versions_considered
    ):
        diagnostic_version_keys = {
            normalized.lower()
            for value in diagnostics.candidate_versions_considered
            if (normalized := normalize_version(value))
        }
        approved_version_keys.intersection_update(diagnostic_version_keys)
    approved_versions = [
        version for version in approved_versions if version.lower() in approved_version_keys
    ]

    attempted_values: list[str] = []
    if diagnostics is not None:
        attempted_values.extend(diagnostics.attempted_versions)
        attempted_values.extend(diagnostics.executed_versions)
        if task.target_package_name:
            attempted_values.extend(
                diagnostics.attempted_versions_by_target.get(task.target_package_name, [])
            )
    if plan is not None:
        attempted_values.extend(plan.attempted_versions)
    attempted_versions = {
        normalized.lower() for value in attempted_values if (normalized := normalize_version(value))
    }
    selected_version = normalize_version(task.selected_version) if task.selected_version else None
    selected_is_approved = (
        selected_version is not None and selected_version.lower() in approved_version_keys
    )
    ordered_versions = (
        ([selected_version] if selected_is_approved else [])
        if maven_mode and not solver_authoritative
        else ([selected_version] if selected_is_approved else []) + approved_versions
    )
    versions: list[str] = []
    seen_versions: set[str] = set()
    for value in ordered_versions:
        normalized = normalize_version(value)
        if normalized is None:
            continue
        key = normalized.lower()
        attempted_replay = (
            allow_attempted_selected_version
            and selected_version is not None
            and key == selected_version.lower()
        )
        if not key or (key in attempted_versions and not attempted_replay) or key in seen_versions:
            continue
        seen_versions.add(key)
        versions.append(normalized)

    if solver_authoritative:
        approved_types = list(
            dict.fromkeys(
                str(value).strip() for value in task.allowed_dependency_types if str(value).strip()
            )
        )
        if (
            plan is not None
            and plan.candidate_dependency_types
            and (
                not plan.target_package_name or plan.target_package_name == task.target_package_name
            )
        ):
            approved_types = [
                value for value in approved_types if value in set(plan.candidate_dependency_types)
            ]
        if (
            diagnostics is not None
            and diagnostics_match_target
            and diagnostics.candidate_dependency_types
        ):
            approved_types = [
                value
                for value in approved_types
                if value in set(diagnostics.candidate_dependency_types)
            ]
    else:
        selected_type = (
            plan.target_dependency_type
            if plan is not None and plan.target_dependency_type
            else task.target_dependency_type
            or (diagnostics.target_dependency_type if diagnostics is not None else None)
        )
        strategy_stage = plan.strategy_stage if plan is not None else task.strategy_stage
        approved_types = _supervisor_dependency_type_candidates(strategy_stage, selected_type)
        if plan is not None and plan.candidate_dependency_types:
            committed_types = set(plan.candidate_dependency_types)
            approved_types = [value for value in approved_types if value in committed_types]
    attempted_types = {
        str(value).strip().lower()
        for value in (diagnostics.attempted_dependency_types if diagnostics is not None else [])
        if str(value).strip()
    }
    approved_types = [
        value for value in approved_types if value.strip().lower() not in attempted_types
    ]
    approved_type_keys = set(approved_types)
    selected_type = task.target_dependency_type
    if selected_type not in approved_type_keys:
        selected_type = None
    types: list[str] = []
    for value in [selected_type, *approved_types]:
        if value and value not in types:
            types.append(value)
    return versions, types


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
    allow_attempted_selected_version: bool = False,
    project_language: ProjectLanguage = ProjectLanguage.NODEJS,
) -> _UpdateDispatchAuthorization | None:
    """Return a dispatch authorization or reject incomplete update state.

    Update workers may only receive candidates in the task's outer solver
    allowlist, narrowed by any matching committed retry-plan or diagnostic
    candidate pool. A selected task version alone never widens that allowlist.
    """
    allowed_versions, allowed_dependency_types = _ordered_update_candidates(
        task,
        plan=plan,
        diagnostics=diagnostics,
        allow_attempted_selected_version=allow_attempted_selected_version,
        project_language=project_language,
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
    maven_mode = project_language == ProjectLanguage.JAVA and bool(
        re.fullmatch(r"[^:\s]+:[^:\s]+", task.target_package_name or "")
    )
    normalize_version = (
        (lambda value: value.strip()) if maven_mode else (lambda value: value.strip().lstrip("vV"))
    )
    plan_version = (
        normalize_version(plan.selected_version)
        if plan is not None and plan.selected_version
        else None
    )
    task_version = normalize_version(task.selected_version) if task.selected_version else None
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


def _validated_delta_replay_candidate(
    task: RemediationTask,
    delta_isolation_by_cluster: Mapping[str, Any],
    attempt_snapshots_by_id: Mapping[str, TaskAttemptSnapshot],
) -> str | None:
    """Return the exact candidate a validated delta peer may replay once.

    A failed atomic QA batch records each worker-executed version as attempted.
    When the Supervisor validates a unique blame and a restored candidate,
    each non-blamed task is requeued for its own authoritative QA run. Permit
    only that task's original selected version, and only while its immutable
    attempt snapshot and current solver allowlist still authorize it.
    """
    if (
        task.status != TaskStatus.PENDING
        or task.current_attempt_id is not None
        or not task.selected_version
    ):
        return None
    for cluster_id, record in delta_isolation_by_cluster.items():
        if not isinstance(record, Mapping):
            continue
        authorized_task_ids = record.get("authoritative_qa_replay_task_ids")
        attempt_ids_by_task = record.get("attempt_ids_by_task")
        responsible_task_ids = record.get("responsible_task_ids")
        if (
            record.get("status") != "IDENTIFIED"
            or record.get("candidate_restored") is not True
            or record.get("cluster_id") != cluster_id
            or not isinstance(authorized_task_ids, (list, tuple))
            or task.task_id not in authorized_task_ids
            or not isinstance(attempt_ids_by_task, Mapping)
            or task.task_id not in attempt_ids_by_task
            or not isinstance(responsible_task_ids, (list, tuple))
            or len(responsible_task_ids) != 1
            or task.task_id in responsible_task_ids
        ):
            continue

        original_attempt_id = attempt_ids_by_task[task.task_id]
        snapshot = attempt_snapshots_by_id.get(original_attempt_id)
        maven_mode = snapshot is not None and snapshot.maven_target_operation is not None
        normalize_version = (
            (lambda value: value.strip())
            if maven_mode
            else (lambda value: value.strip().lstrip("vV"))
        )
        selected_version = normalize_version(task.selected_version)
        if (
            snapshot is None
            or snapshot.attempt_id != original_attempt_id
            or snapshot.task_id != task.task_id
            or snapshot.dispatch_node != "update_subagent"
            or snapshot.cluster_id != cluster_id
            or snapshot.dispatch_batch_id != record.get("dispatch_batch_id")
            or snapshot.action_digest != record.get("action_digest")
            or snapshot.portfolio_plan_id != record.get("source_portfolio_plan_id")
            or snapshot.selected_version is None
            or normalize_version(snapshot.selected_version) != selected_version
            or selected_version
            not in {normalize_version(value) for value in snapshot.allowed_target_versions}
            or selected_version
            not in {normalize_version(value) for value in task.allowed_target_versions}
        ):
            continue
        return selected_version
    return None


def _validated_delta_replay_task_ids(
    task_queue: Mapping[str, RemediationTask],
    delta_isolation_by_cluster: Mapping[str, Any],
    attempt_snapshots_by_id: Mapping[str, TaskAttemptSnapshot],
) -> set[str]:
    """Return pending peers with a still-valid one-use candidate replay grant."""
    return {
        task_id
        for task_id, task in task_queue.items()
        if _validated_delta_replay_candidate(
            task,
            delta_isolation_by_cluster,
            attempt_snapshots_by_id,
        )
        is not None
    }


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
    target_manifest_paths: Iterable[str] = (),
    maven_target_operation: MavenTargetOperation | None = None,
) -> tuple[RemediationTask, TaskAttemptSnapshot]:
    maven_mode = maven_target_operation is not None
    normalize_version = (
        (lambda value: str(value).strip())
        if maven_mode
        else (lambda value: str(value).strip().lstrip("vV"))
    )
    normalized_allowed_target_versions = list(
        dict.fromkeys(
            normalize_version(value) for value in allowed_target_versions if str(value).strip()
        )
    )
    if dispatch_node == "update_subagent":
        if not normalized_allowed_target_versions:
            raise ValueError(
                "update_subagent snapshots require a non-empty candidate authorization."
            )
        if not task.selected_version:
            raise ValueError("update_subagent snapshots require a committed selected_version.")
        if normalize_version(task.selected_version) not in normalized_allowed_target_versions:
            raise ValueError(
                "update_subagent snapshot selected_version must belong to its candidate authorization."
            )
    normalized_target_manifest_paths: list[str] = []
    for raw_path in target_manifest_paths:
        try:
            normalized_path = normalize_workspace_path(
                raw_path,
                allow_workspace_prefix=False,
            )
        except WorkspacePathError as exc:
            raise ValueError("attempt target manifest paths must be safe workspace paths") from exc
        if normalized_path not in normalized_target_manifest_paths:
            normalized_target_manifest_paths.append(normalized_path)
    if (
        dispatch_node == "update_subagent"
        and normalized_target_manifest_paths
        and maven_target_operation is None
    ):
        raise ValueError("Maven update snapshots require a committed declaration operation.")
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
        allowed_target_versions=normalized_allowed_target_versions,
        target_package_name=task.target_package_name,
        target_dependency_type=task.target_dependency_type,
        maven_target_operation=maven_target_operation,
        target_manifest_paths=normalized_target_manifest_paths,
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
            "portfolio_plan_id": snapshot.portfolio_plan_id,
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
    project_language: ProjectLanguage = ProjectLanguage.NODEJS,
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
    maven_mode = is_maven_group(group, project_language)
    selected = (
        task.selected_version.strip()
        if task.selected_version and maven_mode
        else task.selected_version.strip().lstrip("vV")
        if task.selected_version
        else None
    )
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
                project_language=project_language,
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
        version_scheme="maven" if maven_mode else "semver",
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


def _peer_conflict_referral_decision(
    task: RemediationTask,
    *,
    decision_reason: str,
) -> SupervisorDecision:
    """Return the Supervisor-owned terminal referral for an unresolved peer conflict."""
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
    project_language: ProjectLanguage = ProjectLanguage.NODEJS,
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
            project_language=project_language,
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
                next_node="supervisor",
                target_task_ids=[],
                task_status_updates={task.task_id: TaskStatus.INCONCLUSIVE},
                instructions=(
                    "Registry verification is unavailable; defer remediation and preserve retry budget."
                ),
                decision_reason=detail,
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
            project_language=project_language,
        )
        portfolio_escalation_required = _portfolio_escalation_required(
            context,
            classify_diagnostics(context),
        )
        if (
            not settings.openai_api_key
            and task.strategy == RoutingStrategy.VERSION_BUMP
            and task.status != TaskStatus.NEEDS_RETRY
            and task.retry_count == 0
        ):
            if portfolio_escalation_required:
                return _peer_conflict_referral_decision(
                    task,
                    decision_reason=(
                        "Deterministic Supervisor escalated an unresolved peer conflict "
                        "without a compatible candidate or evidence-backed source workaround."
                    ),
                )
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
                project_language=project_language,
                staged_resolutions=staged_resolutions,
            )
        action, verification = propose_and_verify_tactical_action(
            context,
            settings=settings,
        )
        if action is None or verification is None:
            if portfolio_escalation_required:
                return _peer_conflict_referral_decision(
                    task,
                    decision_reason=(
                        "Deterministic Supervisor escalated an unresolved peer conflict "
                        "after no valid tactical action was returned."
                    ),
                )
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
                project_language=project_language,
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
            if portfolio_escalation_required:
                return _peer_conflict_referral_decision(
                    task,
                    decision_reason=(
                        "Deterministic Supervisor escalated an unresolved peer conflict "
                        "after rejecting an action without an authoritative source workaround."
                    ),
                )
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
            return _peer_conflict_referral_decision(
                task,
                decision_reason=decision_reason,
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
            version_scheme=(
                "maven" if is_maven_group(group, context.project_language) else "semver"
            ),
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
    solver_plan = getattr(plan, "solver_plan", None)
    selected = getattr(solver_plan, "selected_plan", None)
    certificate = getattr(plan, "resolution_certificate", None)
    if certificate is not None:
        certificate_status = getattr(certificate.status, "value", certificate.status)
        if str(certificate_status).upper() != PackageResolutionStatus.CERTIFIED.value:
            violations.append("package-resolution certificate is not CERTIFIED")
        solver_status = getattr(getattr(solver_plan, "status", None), "value", None)
        if str(solver_status or getattr(solver_plan, "status", "")).upper() not in {
            "OPTIMAL",
            "FEASIBLE",
        }:
            violations.append("certified dispatch requires an accepted solver status")
        if not getattr(solver_plan, "candidate_catalog_complete", False):
            violations.append("certified dispatch requires a complete candidate catalog")
        certificate_fields = {
            "portfolio_plan_id": getattr(plan, "portfolio_plan_id", None),
            "solver_input_digest": getattr(plan, "solver_input_digest", None),
            "repository_fingerprint": getattr(plan, "repository_fingerprint", None),
            "workspace_graph_digest": getattr(plan, "workspace_graph_digest", None),
            "candidate_catalog_digest": getattr(solver_plan, "candidate_catalog_digest", None),
        }
        for field_name, expected_value in certificate_fields.items():
            if getattr(certificate, field_name, None) != expected_value:
                violations.append(
                    f"package-resolution certificate {field_name} differs from committed plan"
                )
        plan_revisions = dict(getattr(plan, "task_revisions", {}) or {})
        if dict(getattr(certificate, "task_revisions", {}) or {}) != plan_revisions:
            violations.append("package-resolution certificate task revisions are stale")
        if selected is None:
            violations.append("certified dispatch requires a selected solver assignment")
        else:
            if getattr(certificate, "candidate_plan_id", None) != getattr(
                selected, "candidate_plan_id", None
            ):
                violations.append("package-resolution certificate candidate plan ID is stale")
            assignment = dict(
                sorted((getattr(selected, "selected_candidate_versions", {}) or {}).items())
            )
            assignment_digest = hashlib.sha256(
                json.dumps(
                    assignment,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                ).encode("utf-8")
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
            certificate_covered = set(certificate.covered_coverage_ids)
            certificate_workaround = set(certificate.workaround_coverage_ids)
            certificate_unresolved = set(certificate.unresolved_coverage_ids)
            if certificate_covered != expected_covered:
                violations.append("package-resolution certificate covered-coverage set is stale")
            if certificate_workaround != expected_workaround:
                violations.append("package-resolution certificate workaround-coverage set is stale")
            if (
                certificate_covered | certificate_workaround | certificate_unresolved
                != expected_all
            ):
                violations.append("package-resolution certificate coverage universe is stale")
    decisions = {
        decision.task_id: decision for decision in (getattr(selected, "task_decisions", None) or [])
    }
    planned_ids = set(getattr(plan, "task_ids", ()) or ())
    sca_group_ids = {group.group_id for group in valid_groups if group.issue_type == IssueType.SCA}
    groups_by_id = {group.group_id: group for group in valid_groups}
    tasks_by_group: dict[str, list[RemediationTask]] = {}
    for task in task_queue.values():
        if task.parent_group_id not in sca_group_ids:
            continue
        # A source-workaround child continues its solver-backed package task; it
        # is not a new version-selection task in the immutable portfolio.
        parent = task_queue.get(task.parent_task_id or "")
        is_committed_workaround_child = (
            task.strategy == RoutingStrategy.CODE_WORKAROUND
            and parent is not None
            and parent.task_id in planned_ids
            and parent.parent_group_id == task.parent_group_id
            and parent.strategy == RoutingStrategy.VERSION_BUMP
            and parent.exhausted_update_path
        )
        if not is_committed_workaround_child:
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
        non_mutating_workaround = (
            task.strategy == RoutingStrategy.CODE_WORKAROUND
            and decision_strategy in {"code_workaround", "workaround"}
            and decision.selected_version is None
            and not decision.allowed_alternative_versions
            and bool(decision.selected_plan_issue_ids)
        )
        group = groups_by_id.get(task.parent_group_id)
        expected_package = (
            (task.target_package_name or group.vulnerable_component or "").strip()
            if group is not None
            else ""
        )
        maven_target = group is not None and is_maven_group(group, ProjectLanguage.JAVA)
        expected_manifest = (
            _group_maven_manifest_path(group)
            if maven_target
            else _group_manifest_path(group)
            if group is not None
            else None
        )
        direct_occurrence = None
        if expected_manifest and expected_package:
            direct_occurrence = (
                f"{expected_manifest}::maven::{expected_package}"
                if maven_target
                else make_occurrence_id(expected_manifest, expected_package)
            )
        decision_lockfile_key = str(decision.lockfile_package_key or "").replace("\\", "/")
        if maven_target:
            expected_maven_lock_key = (
                f"{expected_package}@{decision.installed_version}"
                if expected_package and decision.installed_version
                else None
            )
            lockfile_key_valid = (
                expected_maven_lock_key is not None
                and decision_lockfile_key == expected_maven_lock_key
            )
            physical_occurrence = direct_occurrence
        else:
            lockfile_key_valid = bool(
                expected_package
                and lockfile_key_matches_package(decision_lockfile_key, expected_package)
            )
            physical_occurrence = (
                make_occurrence_id(expected_manifest, expected_package, decision_lockfile_key)
                if expected_manifest and expected_package and lockfile_key_valid
                else None
            )
        decision_occurrence = str(decision.target_occurrence_id or "")
        valid_occurrences = {
            occurrence for occurrence in (direct_occurrence, physical_occurrence) if occurrence
        }
        if decision_occurrence not in valid_occurrences:
            violations.append(
                f"task {task_id} solver identity field target_occurrence_id is invalid"
            )
        if not lockfile_key_valid:
            violations.append(
                f"task {task_id} solver identity field lockfile_package_key is invalid"
            )
        elif task.status not in terminal_statuses and (
            decision_occurrence == physical_occurrence
            and physical_occurrence != direct_occurrence
            and not maven_target
            and decision.dependency_type not in _OVERRIDE_DEPENDENCY_TYPES
            and not non_mutating_workaround
        ):
            violations.append(
                f"task {task_id} nested lockfile target requires a package-manager override"
            )
        expected_identity = {
            "target_group_id": task.parent_group_id,
            "target_package_name": expected_package or None,
            "manifest_path": expected_manifest,
            "lockfile_package_key": decision_lockfile_key if lockfile_key_valid else None,
        }
        for field_name, expected_value in expected_identity.items():
            if getattr(decision, field_name, None) != expected_value:
                violations.append(f"task {task_id} solver identity field {field_name} is invalid")
        if task.status in terminal_statuses:
            continue
        try:
            decision_stage = SCARemediationStage(decision.strategy_stage)
            if task.strategy_stage != decision_stage:
                # A committed pivot advances retry stage without changing the
                # certificate-bound package assignment.
                preserves_committed_versions = (
                    expected == RoutingStrategy.VERSION_BUMP
                    and task.strategy == RoutingStrategy.VERSION_BUMP
                    and task.exhausted_update_path
                    and task.strategy_stage == SCARemediationStage.NPM_LATEST
                    and task.selected_version is None
                )
                if not preserves_committed_versions:
                    violations.append(f"task {task_id} strategy stage differs from solver decision")
        except ValueError:
            violations.append(f"task {task_id} has unknown committed strategy stage")
        if non_mutating_workaround:
            if (
                task.selected_version is not None
                or task.allowed_target_versions
                or task.allowed_dependency_types
                or task.target_dependency_type is not None
            ):
                violations.append(
                    f"task {task_id} non-mutating workaround has package mutation inputs"
                )
            if task.selected_plan_issue_ids != decision.selected_plan_issue_ids:
                violations.append(f"task {task_id} workaround plan IDs differ from solver decision")
            continue
        normalize_plan_version = (
            (lambda value: str(value).strip())
            if maven_target
            else (lambda value: str(value).strip().lstrip("vV"))
        )
        approved_versions = [
            normalize_plan_version(value).lower()
            for value in ([decision.selected_version] + list(decision.allowed_alternative_versions))
            if value
        ]
        task_versions = [
            normalize_plan_version(value).lower() for value in task.allowed_target_versions if value
        ]
        if task_versions != approved_versions:
            violations.append(f"task {task_id} allowed_target_versions differ from solver")
        if task.selected_version and normalize_plan_version(
            task.selected_version
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
        planned_ids = set(getattr(plan, "task_ids", ()) or ())
        groups_by_id = {group.group_id: group for group in valid_groups}
        planned_groups = [
            groups_by_id[task_queue[task_id].parent_group_id]
            for task_id in planned_ids
            if task_id in task_queue and task_queue[task_id].parent_group_id in groups_by_id
        ]
        maven_plan = bool(planned_groups) and all(
            is_maven_group(group, ProjectLanguage.JAVA) for group in planned_groups
        )
        try:
            if maven_plan:
                from remediation_engine.orchestration.portfolio_solver import (
                    maven_repository_fingerprint,
                )

                actual_fingerprint = maven_repository_fingerprint(repo_root)
            else:
                actual_fingerprint = load_npm_graph_snapshot(repo_root).repository_fingerprint
        except (OSError, ValueError):
            actual_fingerprint = None
        if not expected_fingerprint or actual_fingerprint != expected_fingerprint:
            violations.append("repository fingerprint differs from committed portfolio plan")
    return bool(violations)


def _nonterminal_sca_task_ids(
    task_queue: dict[str, RemediationTask],
    group_by_id: dict[str, VulnerabilityGroup],
) -> list[str]:
    """Return sorted nonterminal tasks whose parent group is SCA."""
    return sorted(
        task_id
        for task_id, task in task_queue.items()
        if task.status not in _TERMINAL_STATUSES
        and (group := group_by_id.get(task.parent_group_id)) is not None
        and group.issue_type == IssueType.SCA
    )


def _build_multi_package_action(
    cluster_id: str,
    target_task_ids: Iterable[str],
    task_queue: dict[str, RemediationTask],
    group_by_id: dict[str, VulnerabilityGroup],
    *,
    project_language: ProjectLanguage = ProjectLanguage.NODEJS,
    repo_root: str | None = None,
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
        dependency_type = task.target_dependency_type or "dependencies"
        if project_language == ProjectLanguage.JAVA:
            operation = _maven_attempt_target_operation(
                task,
                group,
                project_language,
                repo_root=repo_root,
            )
            maven_paths = _maven_attempt_target_paths(
                group,
                project_language,
                repo_root=repo_root,
                allow_dependency_management_creation=(
                    operation == MavenTargetOperation.ENSURE_DEPENDENCY_MANAGEMENT
                ),
            )
            localized_manifest_paths = sorted(
                {
                    str(localized.manifest_file).replace("\\", "/").lstrip("/")
                    for localized in group.localized_issues
                    if (localized.package_manager or "").strip().lower() == "maven"
                    and localized.manifest_file
                }
                & set(maven_paths)
            )
            manifest_path = localized_manifest_paths[0] if localized_manifest_paths else ""
            if (
                not package_name
                or not target_version
                or not manifest_path
                or not maven_paths
                or operation is None
            ):
                return None
            mutation_values = {
                "ecosystem": "maven",
                "target_manifest_paths": maven_paths,
                "maven_target_operation": operation,
            }
        else:
            manifest_path = _group_manifest_path(group) or ""
            if (
                not package_name
                or not target_version
                or not manifest_path
                or manifest_path.split("/")[-1] != "package.json"
            ):
                return None
            mutation_values = {}
        mutations.append(
            PackageMutation(
                task_id=task_id,
                package_name=package_name,
                manifest_path=manifest_path,
                target_version=target_version,
                dependency_type=dependency_type,
                **mutation_values,
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


def _validated_active_qa_multi_package_action(
    action: Any,
    target_task_ids: Iterable[str],
    task_queue: Mapping[str, RemediationTask],
    attempt_snapshots_by_id: Mapping[str, TaskAttemptSnapshot],
) -> MultiPackageAction | None:
    """Return an active package action only when QA owns its exact committed batch.

    Args:
        action: Active action retained from the worker dispatch.
        target_task_ids: Tasks selected for the current QA dispatch.
        task_queue: Current authoritative task records.
        attempt_snapshots_by_id: Immutable committed attempt snapshots.

    Returns:
        The action if its task set and provenance match the current QA attempts;
        otherwise ``None``.
    """
    if not isinstance(action, MultiPackageAction):
        return None

    target_ids = list(target_task_ids)
    mutation_task_ids = [mutation.task_id for mutation in action.package_mutations]
    cluster_id = action.cluster_id
    dispatch_batch_id = action.dispatch_batch_id
    if (
        not target_ids
        or len(target_ids) != len(set(target_ids))
        or len(mutation_task_ids) != len(set(mutation_task_ids))
        or set(mutation_task_ids) != set(target_ids)
        or not cluster_id
        or not dispatch_batch_id
    ):
        return None

    action_digest = instruction_digest(action.model_dump_json())
    for task_id in target_ids:
        task = task_queue.get(task_id)
        if task is None or not task.current_attempt_id:
            return None
        snapshot = attempt_snapshots_by_id.get(task.current_attempt_id)
        if (
            snapshot is None
            or snapshot.task_id != task_id
            or snapshot.task_revision != task.task_revision
            or snapshot.dispatch_node != "update_subagent"
            or snapshot.cluster_id != cluster_id
            or snapshot.dispatch_batch_id != dispatch_batch_id
            or snapshot.action_digest != action_digest
        ):
            return None
    return action


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


def _group_maven_manifest_path(group: VulnerabilityGroup) -> str | None:
    """Return the unique safe POM path localized for a Maven group."""
    manifest_paths: set[str] = set()
    for localized in group.localized_issues:
        if (localized.package_manager or "").strip().lower() != "maven":
            continue
        raw_path = localized.manifest_file
        if not raw_path:
            continue
        try:
            path = normalize_workspace_path(raw_path, allow_workspace_prefix=False)
        except WorkspacePathError:
            continue
        if path.rsplit("/", 1)[-1].casefold() == "pom.xml":
            manifest_paths.add(path)
    if not manifest_paths:
        for raw_path in [*group.file_paths, group.file_path]:
            if not raw_path:
                continue
            try:
                path = normalize_workspace_path(raw_path, allow_workspace_prefix=False)
            except WorkspacePathError:
                continue
            if path.rsplit("/", 1)[-1].casefold() == "pom.xml":
                manifest_paths.add(path)
    return next(iter(manifest_paths)) if len(manifest_paths) == 1 else None


def _maven_attempt_target_paths(
    group: VulnerabilityGroup,
    project_language: ProjectLanguage,
    *,
    repo_root: str | None = None,
    allow_dependency_management_creation: bool = False,
) -> list[str]:
    """Return safe localized Maven POM targets for one Java attempt.

    Both the finding POM and any distinct property POM used to derive the
    dependency version are committed. Non-Java or non-Maven localized issues
    never contribute a worker mutation target.
    """
    if project_language != ProjectLanguage.JAVA:
        return []
    if repo_root:
        paths, errors = _resolve_maven_manifest_scope(
            group,
            Path(repo_root),
            allow_dependency_management_creation=allow_dependency_management_creation,
        )
        return paths if not errors else []
    paths: list[str] = []
    for localized in group.localized_issues:
        if (localized.package_manager or "").strip().lower() != "maven":
            continue
        for raw_path in (localized.manifest_file, localized.version_property_file):
            if not isinstance(raw_path, str) or not raw_path.strip():
                continue
            try:
                path = normalize_workspace_path(raw_path, allow_workspace_prefix=False)
            except WorkspacePathError:
                continue
            if path.rsplit("/", 1)[-1].casefold() != "pom.xml":
                continue
            if path not in paths:
                paths.append(path)
    return paths


def _maven_attempt_target_operation(
    task: RemediationTask,
    group: VulnerabilityGroup,
    project_language: ProjectLanguage,
    *,
    repo_root: str | None = None,
) -> MavenTargetOperation | None:
    """Return the narrow Maven edit operation authorized for this task."""
    if project_language != ProjectLanguage.JAVA or not _maven_attempt_target_paths(
        group,
        project_language,
        repo_root=repo_root,
        allow_dependency_management_creation=(
            task.target_dependency_type == "dependencyManagement"
        ),
    ):
        return None
    if task.target_dependency_type == "dependencyManagement":
        return MavenTargetOperation.ENSURE_DEPENDENCY_MANAGEMENT
    return MavenTargetOperation.UPDATE_DECLARATION


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
        # ERESOLVE output can include compatible peer edges alongside the
        # actual conflict; do not turn those satisfied edges into task pairs.
        if (
            requester
            and evidence.required_range
            and evidence.observed_version
            and npm_range_contains(evidence.required_range, evidence.observed_version) is True
        ):
            continue
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
    allow_verified_absent_target_pass: bool = False,
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
            allow_verified_absent_target_pass=allow_verified_absent_target_pass,
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


def _validated_delta_isolation_blame(
    portfolio_escalation: Mapping[str, Any] | None,
    delta_isolation_by_cluster: Mapping[str, Any],
    atomic_cluster_task_ids: list[str],
    task_queue: Mapping[str, RemediationTask],
    attempt_snapshots_by_id: Mapping[str, TaskAttemptSnapshot],
) -> str | None:
    """Return one blamed task only when delta provenance matches the attempts."""
    if (
        not isinstance(portfolio_escalation, Mapping)
        or portfolio_escalation.get("reason") != "DELTA_ISOLATION_ATTRIBUTION"
        or len(atomic_cluster_task_ids) <= 1
        or len(atomic_cluster_task_ids) != len(set(atomic_cluster_task_ids))
    ):
        return None

    current_snapshots: list[TaskAttemptSnapshot] = []
    attempt_ids_by_task: dict[str, str] = {}
    for task_id in atomic_cluster_task_ids:
        task = task_queue.get(task_id)
        if task is None or not task.current_attempt_id:
            return None
        snapshot = attempt_snapshots_by_id.get(task.current_attempt_id)
        if (
            snapshot is None
            or snapshot.attempt_id != task.current_attempt_id
            or snapshot.task_id != task_id
            or snapshot.dispatch_node != "update_subagent"
        ):
            return None
        current_snapshots.append(snapshot)
        attempt_ids_by_task[task_id] = task.current_attempt_id

    def common_nonempty_text(attribute: str) -> str | None:
        values = [getattr(snapshot, attribute, None) for snapshot in current_snapshots]
        if any(not isinstance(value, str) or not value.strip() for value in values):
            return None
        unique_values = set(values)
        return next(iter(unique_values)) if len(unique_values) == 1 else None

    cluster_id = common_nonempty_text("cluster_id")
    dispatch_batch_id = common_nonempty_text("dispatch_batch_id")
    action_digest = common_nonempty_text("action_digest")
    source_portfolio_plan_id = common_nonempty_text("portfolio_plan_id")
    if None in {cluster_id, dispatch_batch_id, action_digest, source_portfolio_plan_id}:
        return None

    current_cluster_task_ids = {
        task_id
        for task_id, task in task_queue.items()
        if task.current_attempt_id
        and (snapshot := attempt_snapshots_by_id.get(task.current_attempt_id)) is not None
        and snapshot.cluster_id == cluster_id
        and snapshot.dispatch_node == "update_subagent"
    }
    if current_cluster_task_ids != set(atomic_cluster_task_ids):
        return None

    record = delta_isolation_by_cluster.get(cluster_id)
    if (
        not isinstance(record, Mapping)
        or record.get("status") != "IDENTIFIED"
        or record.get("candidate_restored") is not True
        or record.get("cluster_id") != cluster_id
        or record.get("dispatch_batch_id") != dispatch_batch_id
        or record.get("action_digest") != action_digest
        or record.get("source_portfolio_plan_id") != source_portfolio_plan_id
        or record.get("attempt_ids_by_task") != attempt_ids_by_task
    ):
        return None

    responsible_task_ids = record.get("responsible_task_ids")
    forced_singleton_task_ids = portfolio_escalation.get("forced_singleton_task_ids")
    if (
        not isinstance(responsible_task_ids, (list, tuple))
        or len(responsible_task_ids) != 1
        or not isinstance(forced_singleton_task_ids, (list, tuple))
        or len(forced_singleton_task_ids) != 1
        or forced_singleton_task_ids != responsible_task_ids
    ):
        return None
    blamed_task_id = responsible_task_ids[0]
    if not isinstance(blamed_task_id, str) or blamed_task_id not in set(atomic_cluster_task_ids):
        return None
    if (
        portfolio_escalation.get("triggering_attempt_id") != attempt_ids_by_task[blamed_task_id]
        or portfolio_escalation.get("source_portfolio_plan_id") != source_portfolio_plan_id
    ):
        return None
    return blamed_task_id


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


def _run_supervisor_decision_node(state: OrchestratorState) -> dict[str, Any]:
    """
    Deterministic Supervisor decision body for Phase 5 orchestration.

    Execution stages
    ----------------
    1. Normalize task_queue: create initial RemediationTask entries for any
       valid_groups not yet represented (copy-on-write via model_copy).
    2. Ingest subagent action summaries for current active_target_task_ids only.
    3. Ingest QA results for active task IDs only (when status == "qa_completed").
    4. Resolve no-fix terminal states; defer the retry-cap terminalization
       until tactical and deterministic replanning has had a chance to pivot.
    5. Short-circuit: if an active task is optimistically_fixed â†’ qa_critic.
    6. Project accepted incremental scan results before terminal routing.
    7. Plan retry inputs from deterministic registry facts and committed state.
    8. Route through the deterministic state machine and validate its
       dispatch projection.
    9. Apply guarded: revised_instructions, strategy updates, status overrides,
       unfixable marks, new constraints, and materialized spawn requests.
    10. Return state patch.
    """
    raw_language = state.get("project_language")
    project_language = (
        raw_language
        if isinstance(raw_language, ProjectLanguage)
        else ProjectLanguage(str(raw_language))
        if raw_language is not None
        else ProjectLanguage.NODEJS
    )
    if state.get("post_qa_retriage_limit_reached"):
        scan_due = _final_full_scan_due(state, dict(state.get("task_queue", {}) or {}))
        next_node = "final_full_scan" if scan_due else "teardown"
        decision = SupervisorDecision(
            decision_code=(
                DecisionCode.FINAL_FULL_SCAN_REQUIRED
                if scan_due
                else DecisionCode.NO_ACTIONABLE_TASKS
            ),
            next_node=next_node,
            target_task_ids=[],
            instructions=(
                "Run the enabled final full-workspace scan before teardown."
                if scan_due
                else "Development post-QA re-triage limit reached; proceed to teardown."
            ),
            decision_reason=(
                "The re-triage limit was reached; the configured final scan remains due."
                if scan_due
                else "The configured development post-QA re-triage limit was reached."
            ),
        )
        return {
            "status": "supervisor_routed",
            "next_routing_step": next_node,
            "active_target_task_ids": [],
            "decision_code": decision.decision_code,
            "supervisor_audit": _emit_audit(decision, [], int(state.get("state_revision", 0)) + 1),
            "supervisor_instructions": decision.instructions,
            "post_remediation_scan_identifiers": list(
                state.get("post_remediation_scan_identifiers", []) or []
            ),
            "post_remediation_scan_issues": list(
                state.get("post_remediation_scan_issues", []) or []
            ),
            "new_vulnerability_identifiers": list(
                state.get("new_vulnerability_identifiers", []) or []
            ),
            "new_vulnerability_status": state.get("new_vulnerability_status", "not_scanned"),
            "triage_required": False,
            **_scan_ledger_state_snapshot(state),
        }
    valid_groups: list[VulnerabilityGroup] = list(state.get("valid_groups", []))
    if not valid_groups:
        if _final_full_scan_due(state, dict(state.get("task_queue", {}) or {})):
            decision = SupervisorDecision(
                decision_code=DecisionCode.FINAL_FULL_SCAN_REQUIRED,
                next_node="final_full_scan",
                target_task_ids=[],
                instructions="Run the configured final full-workspace scan before teardown.",
                decision_reason=(
                    "All remediation tasks are terminal and the development final full scan is enabled."
                ),
            )
            return {
                "status": "supervisor_routed",
                "next_routing_step": "final_full_scan",
                "active_target_task_ids": [],
                "decision_code": decision.decision_code,
                "supervisor_audit": _emit_audit(
                    decision, [], int(state.get("state_revision", 0)) + 1
                ),
                "supervisor_instructions": decision.instructions,
                **_scan_ledger_state_snapshot(state),
            }
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
                **_scan_ledger_state_snapshot(state),
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
            **_scan_ledger_state_snapshot(state),
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
    delta_isolation_by_cluster: dict[str, Any] = dict(
        state.get("delta_isolation_by_cluster", {}) or {}
    )
    qa_rerun_counts_by_attempt: dict[str, int] = dict(state.get("qa_rerun_counts_by_attempt", {}))
    accepted_package_inventory_by_lockfile: dict[str, str] = dict(
        state.get("accepted_package_inventory_by_lockfile", {}) or {}
    )
    current_scan_observations_by_occurrence: dict[str, PackageScanOccurrence] = dict(
        state.get("current_scan_observations_by_occurrence", {}) or {}
    )
    latest_successful_scan_by_task: dict[str, TaskScanRecord] = dict(
        state.get("latest_successful_scan_by_task", {}) or {}
    )
    latest_incremental_scan_by_input: dict[str, IncrementalScanRecord] = dict(
        state.get("latest_incremental_scan_by_input", {}) or {}
    )
    setup_scan_evidence: ODCScanEvidence | None = state.get("setup_scan_evidence")
    setup_scan_committed = bool(state.get("setup_scan_committed", False))
    if setup_scan_evidence is not None and not setup_scan_committed:
        if setup_scan_evidence.complete:
            for occurrence_key in setup_scan_evidence.inventory_removals:
                accepted_package_inventory_by_lockfile.pop(occurrence_key, None)
                current_scan_observations_by_occurrence.pop(occurrence_key, None)
            accepted_package_inventory_by_lockfile.update(setup_scan_evidence.inventory_updates)
            for occurrence in setup_scan_evidence.covered_occurrences:
                occurrence_key = package_scan_occurrence_key(
                    occurrence.source_input_path,
                    occurrence.lockfile_key,
                    occurrence.package_name,
                    occurrence.version,
                    occurrence.occurrence_index,
                )
                current_scan_observations_by_occurrence[occurrence_key] = occurrence
        setup_scan_committed = True
    candidate_scan_records_by_task: dict[str, TaskScanRecord] = {}
    candidate_scan_evidence_by_attempt: dict[str, QAAttemptResult] = {}
    current_qa_results_by_task: dict[str, QAAttemptResult] = {}
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
    task_statuses_at_entry = {task_id: task.status for task_id, task in raw_task_queue.items()}
    task_queue: dict[str, RemediationTask] = {
        task_id: task.model_copy() for task_id, task in raw_task_queue.items()
    }
    # Legacy task records may predate Supervisor-owned QA policy provenance.
    # Recover policy only before a portfolio plan exists; planned task inputs
    # remain immutable and are revalidated by the portfolio boundary.
    if state.get("portfolio_plan") is None:
        for task_id, task in list(task_queue.items()):
            if (
                task.qa_policy is not None
                or task.current_attempt_id is not None
                or task.status in _TERMINAL_STATUSES
                or task.portfolio_plan_id is not None
            ):
                continue
            recovered_policy = derive_missing_task_qa_policy(
                task,
                group_by_id.get(task.parent_group_id),
            )
            if recovered_policy is not None:
                task_queue[task_id] = task.model_copy(
                    update={
                        "qa_policy": recovered_policy,
                        "task_revision": task.task_revision + 1,
                    }
                )
    pending_request_at_entry = state.get("portfolio_replan_request")
    unfixable_replan_cleared = bool(
        isinstance(pending_request_at_entry, PortfolioReplanRequest)
        and pending_request_at_entry.reason == "UNFIXABLE_REPLAN"
        and not _nonterminal_sca_task_ids(task_queue, group_by_id)
    )
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
    post_qa_triage_pending = bool(
        state.get("triage_required")
        and state.get("status") in {"qa_completed", "qa_failed", "final_scan_completed"}
    )
    if (
        plan_violations
        and not has_open_attempt
        and not unfixable_replan_cleared
        and not post_qa_triage_pending
    ):
        source_plan_id = (
            getattr(committed_plan, "portfolio_plan_id", None)
            if committed_plan is not None
            else None
        )
        pending_request = state.get("portfolio_replan_request")
        request = (
            pending_request
            if isinstance(pending_request, PortfolioReplanRequest)
            else PortfolioReplanRequest(
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
            "accepted_package_inventory_by_lockfile": accepted_package_inventory_by_lockfile,
            "current_scan_observations_by_occurrence": current_scan_observations_by_occurrence,
            "latest_successful_scan_by_task": latest_successful_scan_by_task,
            "setup_scan_evidence": setup_scan_evidence,
            "setup_scan_committed": setup_scan_committed,
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
        for result_key, stale_result in worker_results_by_attempt.items():
            if (
                result_key != current_attempt_id
                and stale_result.task_id == task_id
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
            if (
                current_attempt_id not in processed_worker_attempt_ids
                and result.attempt_id not in processed_worker_attempt_ids
            ):
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
                processed_worker_attempt_ids.add(result.attempt_id)
                new_worker_attempt_ids.append(result.attempt_id)
            continue
        if (
            result is not None
            and current_attempt_id not in processed_worker_attempt_ids
            and result.attempt_id not in processed_worker_attempt_ids
        ):
            if (
                result.attempt_id != current_attempt_id
                or result.task_id != task_id
                or result.task_revision != task.task_revision
                or snapshot is None
                or (snapshot is not None and snapshot.task_revision != result.task_revision)
                or (
                    snapshot is not None
                    and task.portfolio_plan_id is not None
                    and snapshot.portfolio_plan_id != task.portfolio_plan_id
                )
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
                rejected_attempt_id = (
                    result.attempt_id
                    if result.attempt_id != current_attempt_id
                    else current_attempt_id
                )
                processed_worker_attempt_ids.add(rejected_attempt_id)
                new_worker_attempt_ids.append(rejected_attempt_id)
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
            maven_mode = snapshot.maven_target_operation is not None
            normalize_executed_version = (
                (lambda version: version.strip().lower())
                if maven_mode
                else (lambda version: version.strip().lstrip("vV").lower())
            )
            normalized_executed = {
                normalize_executed_version(version)
                for version in [*raw_executed_versions, reported_effective_version]
                if version
            }
            normalized_allowed = {
                normalize_executed_version(version) for version in allowed_versions if version
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
    qa_inconclusive_exhausted_task_ids: set[str] = set()
    for task_id in active_target_task_ids:
        task = task_queue.get(task_id)
        if task is None or not task.current_attempt_id:
            continue
        qa_result_map_key = task.current_attempt_id
        qa_result = qa_results_by_attempt.get(qa_result_map_key)
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
            or qa_result_map_key != qa_result.attempt_id
            or qa_result.attempt_id != task.current_attempt_id
            or qa_result.task_revision != task.task_revision
            or snapshot is None
            or snapshot.attempt_id != task.current_attempt_id
            or snapshot.task_id != task_id
            or (snapshot is not None and snapshot.task_revision != qa_result.task_revision)
            or (
                snapshot is not None
                and task.portfolio_plan_id is not None
                and snapshot.portfolio_plan_id != task.portfolio_plan_id
            )
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
            scan_record = _task_scan_record_from_qa_result(qa_result)
            if scan_record is not None:
                candidate_scan_records_by_task[task_id] = scan_record
            current_qa_results_by_task[task_id] = qa_result
            if qa_result.batch_scan_evidence is not None:
                candidate_scan_evidence_by_attempt[qa_result.attempt_id] = qa_result
            qa_evaluations[task_id] = qa_result.evaluation
            qa_result_task_ids.add(task_id)
            qa_requires_rerun = (
                qa_result.evaluation.evidence_inconclusive or qa_result.evaluation.contract_error
            )
            if qa_requires_rerun:
                reruns_used = max(
                    0,
                    qa_rerun_counts_by_attempt.get(task.current_attempt_id, 0),
                )
                if reruns_used < MAX_QA_EVIDENCE_RERUNS:
                    qa_rerun_counts_by_attempt[task.current_attempt_id] = reruns_used + 1
                else:
                    # Repeating the same inconclusive QA evidence cannot make
                    # the task progress. Bound the reruns and preserve the
                    # unresolved outcome for the final report.
                    qa_requires_rerun = False
                    qa_inconclusive_exhausted_task_ids.add(task_id)
                    errors.append(
                        f"supervisor: QA remained inconclusive for {task_id} after "
                        f"{MAX_QA_EVIDENCE_RERUNS} same-attempt rerun(s); "
                        "marked the task inconclusive."
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
    validated_delta_blame_task_id: str | None = None
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
        exhausted_cluster_qa = bool(
            qa_inconclusive_exhausted_task_ids.intersection(atomic_cluster_task_ids)
        )
        if exhausted_cluster_qa and not has_real_failure:
            for task_id in atomic_cluster_task_ids:
                task = task_queue[task_id]
                _commit_task_transition(
                    task_queue,
                    task_id,
                    updates={"status": TaskStatus.INCONCLUSIVE},
                    close_attempt=task.current_attempt_id is not None,
                )
            errors.append(
                "supervisor: atomic package-cluster QA exhausted its bounded evidence "
                "reruns; marked every cluster task inconclusive."
            )
        elif not has_real_failure and has_inconclusive:
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
            validated_delta_blame_task_id = _validated_delta_isolation_blame(
                portfolio_escalation if isinstance(portfolio_escalation, Mapping) else None,
                delta_isolation_by_cluster,
                atomic_cluster_task_ids,
                task_queue,
                attempt_snapshots_by_id,
            )
            if (
                isinstance(portfolio_escalation, Mapping)
                and portfolio_escalation.get("reason") == "DELTA_ISOLATION_ATTRIBUTION"
                and validated_delta_blame_task_id is None
            ):
                portfolio_escalation = {
                    **portfolio_escalation,
                    "forced_singleton_task_ids": [],
                    "triggering_attempt_id": None,
                    "source_portfolio_plan_id": None,
                }
            validated_delta_cluster_id = next(
                (
                    snapshot.cluster_id
                    for task_id in atomic_cluster_task_ids
                    if (task := task_queue.get(task_id)) is not None
                    and task.current_attempt_id is not None
                    and (snapshot := attempt_snapshots_by_id.get(task.current_attempt_id))
                    is not None
                ),
                None,
            )
            for task_id in atomic_cluster_task_ids:
                task = task_queue[task_id]
                if task_id == validated_delta_blame_task_id:
                    updates = {
                        "status": TaskStatus.NEEDS_RETRY,
                        "retry_count": task.retry_count + 1,
                    }
                elif validated_delta_blame_task_id is not None:
                    updates = {"status": TaskStatus.PENDING}
                else:
                    updates = {
                        "status": TaskStatus.NEEDS_RETRY,
                        "retry_count": task.retry_count + 1,
                    }
                _commit_task_transition(
                    task_queue,
                    task_id,
                    updates=updates,
                    close_attempt=True,
                )
            if validated_delta_blame_task_id is not None and validated_delta_cluster_id is not None:
                isolation_record = delta_isolation_by_cluster.get(validated_delta_cluster_id)
                if isinstance(isolation_record, Mapping):
                    replay_task_ids = [
                        task_id
                        for task_id in atomic_cluster_task_ids
                        if task_id != validated_delta_blame_task_id
                    ]
                    delta_isolation_by_cluster[validated_delta_cluster_id] = {
                        **isolation_record,
                        "authoritative_qa_replay_task_ids": replay_task_ids,
                    }
                errors.append(
                    "supervisor: delta attribution charged the blamed task and requeued "
                    "untested cluster peers."
                )
            else:
                errors.append(
                    "supervisor: atomic package-cluster QA failure kept every member non-passed."
                )
    if (
        isinstance(portfolio_escalation, Mapping)
        and portfolio_escalation.get("reason") == "DELTA_ISOLATION_ATTRIBUTION"
    ):
        if validated_delta_blame_task_id is None:
            portfolio_escalation = {
                **portfolio_escalation,
                "forced_singleton_task_ids": [],
                "triggering_attempt_id": None,
                "source_portfolio_plan_id": None,
            }
        portfolio_replan_request = None
    else:
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
                        escalation.get("source_portfolio_plan_id")
                        if escalation.get("reason") == "DELTA_ISOLATION_ATTRIBUTION"
                        else (
                            committed_plan.portfolio_plan_id
                            if committed_plan is not None
                            else escalation.get("source_portfolio_plan_id")
                        )
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
            if resolved_t_id in qa_inconclusive_exhausted_task_ids:
                _commit_task_transition(
                    task_queue,
                    resolved_t_id,
                    updates={"status": TaskStatus.INCONCLUSIVE},
                    close_attempt=task.current_attempt_id is not None,
                )
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

    # Commit only records that remain accepted after the cluster-wide QA gate.
    # In an atomic batch, no member reaches QA_PASSED unless every evaluation
    # passes; failed or inconclusive attempts therefore cannot alter ledgers.
    for task_id, scan_record in sorted(candidate_scan_records_by_task.items()):
        task = task_queue.get(task_id)
        if task is None or task.status != TaskStatus.QA_PASSED:
            continue
        latest_successful_scan_by_task[task_id] = scan_record
        for occurrence_key in scan_record.removed_occurrence_keys:
            accepted_package_inventory_by_lockfile.pop(occurrence_key, None)
            current_scan_observations_by_occurrence.pop(occurrence_key, None)
        accepted_package_inventory_by_lockfile.update(scan_record.inventory_updates)
        for occurrence in scan_record.covered_occurrences:
            occurrence_key = package_scan_occurrence_key(
                occurrence.source_input_path,
                occurrence.lockfile_key,
                occurrence.package_name,
                occurrence.version,
                occurrence.occurrence_index,
            )
            current_scan_observations_by_occurrence[occurrence_key] = occurrence
        setup_scan_evidence = _prune_setup_scan_evidence(
            setup_scan_evidence,
            scan_record,
            state.get("baseline_scan_identifiers", []) or [],
        )
    baseline_scan_identifiers = state.get("baseline_scan_identifiers", []) or []
    for qa_result in candidate_scan_evidence_by_attempt.values():
        evidence = qa_result.batch_scan_evidence
        if evidence is None or not qa_result.batch_scan_candidate_retained:
            continue
        covered_task_ids = sorted(set(evidence.covered_task_ids))
        if not covered_task_ids or not set(covered_task_ids).issubset(current_qa_results_by_task):
            continue
        covered_results = [current_qa_results_by_task[task_id] for task_id in covered_task_ids]
        # Retained candidates can still be useful for a workaround, but their
        # report-wide findings are not part of the accepted cumulative ledger.
        if any(not result.evaluation.passed for result in covered_results):
            continue
        covered_tasks = [task_queue.get(task_id) for task_id in covered_task_ids]
        if any(task is None or task.status != TaskStatus.QA_PASSED for task in covered_tasks):
            continue
        input_snapshots = _incremental_scan_records_from_evidence(
            evidence,
            task_ids=covered_task_ids,
            attempt_ids=[result.attempt_id for result in covered_results],
            baseline_scan_identifiers=baseline_scan_identifiers,
        )
        latest_incremental_scan_by_input.update(input_snapshots)
        if input_snapshots:
            setup_scan_evidence = _prune_setup_scan_evidence(
                setup_scan_evidence,
                None,
                baseline_scan_identifiers,
                superseded_input_paths=input_snapshots,
            )

    (
        new_vulnerability_identifiers,
        post_remediation_scan_identifiers,
        remaining_target_identifiers,
        post_remediation_scan_issues,
    ) = _calculate_cumulative_scan_metrics(
        task_queue,
        latest_successful_scan_by_task,
        state.get("baseline_scan_identifiers", []) or [],
        setup_scan_evidence,
        latest_incremental_scan_by_input,
    )
    incremental_scan_coverage_complete = _incremental_scan_coverage_complete(
        state,
        setup_scan_evidence,
    )
    new_vulnerability_status = (
        "detected"
        if new_vulnerability_identifiers or remaining_target_identifiers
        else "none"
        if incremental_scan_coverage_complete
        else "scan_failed"
    )
    triage_required = (
        bool(state.get("post_qa_triage_enabled", True))
        and not bool(state.get("post_qa_retriage_limit_reached", False))
        and bool(new_vulnerability_identifiers or remaining_target_identifiers)
    )
    prior_final_scan = state.get("final_full_scan_result")
    if _final_full_scan_value(prior_final_scan, "coverage_scope") == "full_workspace":
        new_vulnerability_identifiers = list(
            _final_full_scan_value(prior_final_scan, "new_identifiers", []) or []
        )
        post_remediation_scan_identifiers = list(
            _final_full_scan_value(prior_final_scan, "found_identifiers", []) or []
        )
        remaining_target_identifiers = list(
            _final_full_scan_value(prior_final_scan, "remaining_target_identifiers", []) or []
        )
        post_remediation_scan_issues = list(
            _final_full_scan_value(prior_final_scan, "found_issues", []) or []
        )
        new_vulnerability_status = str(
            _final_full_scan_value(prior_final_scan, "status", "scan_failed")
        )
        triage_required = (
            bool(state.get("post_qa_triage_enabled", True))
            and not bool(state.get("post_qa_retriage_limit_reached", False))
            and bool(_final_full_scan_value(prior_final_scan, "triage_required", False))
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
    verified_absent_package_tasks = _verified_absent_package_tasks(
        task_queue,
        group_by_id,
        latest_successful_scan_by_task,
        accepted_package_inventory_by_lockfile,
    )
    verified_absent_portfolio_task_ids: set[str] = set()
    for task_id, package_name in verified_absent_package_tasks.items():
        committed_task = _commit_task_transition(
            task_queue,
            task_id,
            updates={
                "status": TaskStatus.QA_PASSED,
                "remediation_action": _VERIFIED_ABSENT_REMEDIATION_ACTION,
            },
            close_attempt=True,
            clear_selected_version=True,
            allow_verified_absent_target_pass=True,
        )
        if committed_task is None or committed_task.status != TaskStatus.QA_PASSED:
            continue
        logger.info(
            "supervisor: closed task %s without a worker because %s is absent.",
            task_id,
            package_name,
        )
        if committed_plan is not None and task_id in set(committed_plan.task_ids):
            verified_absent_portfolio_task_ids.add(task_id)
    if verified_absent_portfolio_task_ids and committed_plan is not None:
        portfolio_replan_request = PortfolioReplanRequest(
            reason=_VERIFIED_ABSENT_PORTFOLIO_REPLAN_REASON,
            source_portfolio_plan_id=(committed_plan.portfolio_plan_id or committed_plan.plan_id),
        )
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
    all_tasks_terminal = all(task.status in _TERMINAL_STATUSES for task in task_queue.values())
    if triage_required and all_tasks_terminal:
        decision = SupervisorDecision(
            next_node="triage",
            target_task_ids=[],
            instructions="Route cumulative accepted scan findings to post-remediation triage.",
            decision_reason=(
                "No actionable remediation tasks remain and accepted scans contain "
                "baseline-absent or remaining target findings."
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
            portfolio_plan=state.get("portfolio_plan"),
            delta_replay_task_ids=_validated_delta_replay_task_ids(
                task_queue, delta_isolation_by_cluster, attempt_snapshots_by_id
            ),
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
                project_language=project_language,
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
            target_task_ids=[target_task_id] if target_task_id else None,
            project_language=project_language,
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
                    delta_replay_task_ids=_validated_delta_replay_task_ids(
                        task_queue, delta_isolation_by_cluster, attempt_snapshots_by_id
                    ),
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
    # A live attempt anywhere in the queue blocks an outer replan.  Looking
    # only at active_target_task_ids is unsafe because portfolio nodes may
    # clear that projection while leaving the immutable task attempt open.
    active_attempt_is_open = any(
        task.current_attempt_id is not None and task.status not in _TERMINAL_STATUSES
        for task in task_queue.values()
    )
    verified_absent_portfolio_replan_pending = bool(
        committed_plan is not None
        and isinstance(portfolio_replan_request, PortfolioReplanRequest)
        and portfolio_replan_request.reason == _VERIFIED_ABSENT_PORTFOLIO_REPLAN_REASON
        and portfolio_replan_request.source_portfolio_plan_id
        == (
            getattr(committed_plan, "portfolio_plan_id", None)
            or getattr(committed_plan, "plan_id", None)
        )
    )
    portfolio_replan_required = bool(
        not active_attempt_is_open
        and (
            verified_absent_portfolio_replan_pending
            or (
                state.get("portfolio_plan") is not None
                and not unfixable_replan_cleared
                and _portfolio_plan_is_stale(
                    state.get("portfolio_plan"),
                    task_queue,
                    valid_groups,
                    state.get("repo_root"),
                )
            )
        )
    )
    defer_verified_absent_portfolio_replan = (
        verified_absent_portfolio_replan_pending and active_attempt_is_open
    )
    resumable_open_attempt_task_ids: set[str] = set()

    def existing_attempt_targets(next_node: str, target_ids: Iterable[str]) -> list[str]:
        """Keep only targets that can resume their current committed worker attempt."""
        selected: list[str] = []
        for task_id in target_ids:
            if task_id not in resumable_open_attempt_task_ids:
                continue
            task = task_queue.get(task_id)
            if (
                task is None
                or task.current_attempt_id is None
                or task.status not in _WORKABLE_STATUSES
            ):
                continue
            snapshot = attempt_snapshots_by_id.get(task.current_attempt_id)
            if (
                snapshot is None
                or snapshot.task_id != task_id
                or snapshot.dispatch_node != next_node
            ):
                continue
            if next_node == "update_subagent" and task.strategy != RoutingStrategy.VERSION_BUMP:
                continue
            if (
                next_node == "workaround_subagent"
                and task.strategy != RoutingStrategy.CODE_WORKAROUND
            ):
                continue
            selected.append(task_id)
        return selected

    if portfolio_replan_required:
        deterministic_decision = SupervisorDecision(
            decision_code=DecisionCode.PORTFOLIO_PLAN_REQUIRED,
            next_node="portfolio",
            target_task_ids=[],
            instructions="Rebuild the package-group portfolio plan after task reconciliation.",
            decision_reason=(
                _VERIFIED_ABSENT_PORTFOLIO_REPLAN_REASON
                if verified_absent_portfolio_replan_pending
                else "The committed portfolio plan is stale after a completed attempt."
            ),
        )
    elif defer_verified_absent_portfolio_replan:
        open_attempt_task_ids = [
            task_id
            for task_id, task in sorted(task_queue.items())
            if task.current_attempt_id is not None and task.status not in _TERMINAL_STATUSES
        ]
        atomic_open_task_ids = [
            task_id for task_id in atomic_cluster_task_ids if task_id in open_attempt_task_ids
        ]
        atomic_batch_qa_ready = bool(atomic_open_task_ids) and all(
            task_queue[task_id].status == TaskStatus.OPTIMISTICALLY_FIXED
            for task_id in atomic_open_task_ids
        )
        qa_preferred_ids = (
            atomic_open_task_ids
            if atomic_batch_qa_ready
            else [
                task_id for task_id in open_attempt_task_ids if task_id not in atomic_open_task_ids
            ]
        )
        qa_ready_ids = _qa_ready_task_ids(
            task_queue,
            preferred_ids=qa_preferred_ids,
            limit=None if atomic_batch_qa_ready else QA_DISPATCH_LIMIT,
        )
        if qa_ready_ids:
            qa_cluster_id = None
            if atomic_batch_qa_ready and len(qa_ready_ids) > 1:
                qa_cluster_id = attempt_snapshots_by_id[
                    task_queue[qa_ready_ids[0]].current_attempt_id
                ].cluster_id
            deterministic_decision = SupervisorDecision(
                next_node="qa_critic",
                target_task_ids=qa_ready_ids,
                cluster_id=qa_cluster_id,
                instructions="Finish QA for committed attempts before rebuilding the portfolio.",
                decision_reason=_VERIFIED_ABSENT_PORTFOLIO_REPLAN_REASON,
            )
        else:
            resumable_attempts: list[tuple[str, TaskAttemptSnapshot]] = []
            for task_id in open_attempt_task_ids:
                task = task_queue[task_id]
                if task.status == TaskStatus.OPTIMISTICALLY_FIXED:
                    continue
                snapshot = attempt_snapshots_by_id.get(task.current_attempt_id or "")
                if (
                    snapshot is not None
                    and snapshot.task_id == task_id
                    and snapshot.dispatch_node in {"update_subagent", "workaround_subagent"}
                ):
                    resumable_attempts.append((task_id, snapshot))
            if resumable_attempts:
                first_task_id, first_snapshot = resumable_attempts[0]
                if first_snapshot.dispatch_batch_id:
                    resume_ids = [
                        task_id
                        for task_id, snapshot in resumable_attempts
                        if snapshot.dispatch_node == first_snapshot.dispatch_node
                        and snapshot.dispatch_batch_id == first_snapshot.dispatch_batch_id
                    ]
                else:
                    resume_ids = [first_task_id]
                resumable_open_attempt_task_ids.update(resume_ids)
                resume_cluster_id = first_snapshot.cluster_id if len(resume_ids) > 1 else None
                deterministic_decision = SupervisorDecision(
                    next_node=first_snapshot.dispatch_node,
                    target_task_ids=resume_ids,
                    cluster_id=resume_cluster_id,
                    instructions="Continue existing committed attempts before rebuilding the portfolio.",
                    decision_reason=_VERIFIED_ABSENT_PORTFOLIO_REPLAN_REASON,
                )
            else:
                errors.append(
                    "supervisor: verified absence dirtied the portfolio while a live attempt "
                    "had no resumable committed snapshot."
                )
                deterministic_decision = SupervisorDecision(
                    decision_code=DecisionCode.PORTFOLIO_PLAN_REQUIRED,
                    next_node="portfolio",
                    target_task_ids=[],
                    instructions="Rebuild the portfolio after reconciling the invalid live attempt.",
                    decision_reason=_VERIFIED_ABSENT_PORTFOLIO_REPLAN_REASON,
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
            triage_required=triage_required,
            portfolio_plan=state.get("portfolio_plan"),
            delta_replay_task_ids=_validated_delta_replay_task_ids(
                task_queue, delta_isolation_by_cluster, attempt_snapshots_by_id
            ),
        )
    # The deterministic decision is authoritative. QA feedback below is
    # carried only when Python produced it for the selected task; no model
    # can add worker feedback or constraints to this projection.
    if portfolio_replan_required:
        decision = deterministic_decision
    elif portfolio_escalation and portfolio_escalation.get("reason") in {
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
    elif decision is None:
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
            portfolio_plan=state.get("portfolio_plan"),
            delta_replay_task_ids=_validated_delta_replay_task_ids(
                task_queue, delta_isolation_by_cluster, attempt_snapshots_by_id
            ),
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
            valid_target_ids = (
                existing_attempt_targets("update_subagent", decision.target_task_ids)
                if defer_verified_absent_portfolio_replan
                else _update_worker_task_ids(
                    task_queue,
                    retry_diagnostics_by_task,
                    preferred_ids=list(decision.target_task_ids),
                    limit=None if decision.cluster_id else UPDATE_DISPATCH_LIMIT,
                    group_by_id=group_by_id,
                )
            )
        elif decision.next_node == "workaround_subagent":
            valid_target_ids = []
            for t_id in decision.target_task_ids:
                if t_id not in known_task_ids:
                    continue
                task = task_queue[t_id]
                if defer_verified_absent_portfolio_replan:
                    valid_target_ids.extend(existing_attempt_targets("workaround_subagent", [t_id]))
                    continue
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
                "supervisor: qa_critic current policy allows at most %d targets, got %d — falling back.",
                QA_DISPATCH_LIMIT,
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
                delta_replay_task_ids=_validated_delta_replay_task_ids(
                    task_queue, delta_isolation_by_cluster, attempt_snapshots_by_id
                ),
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
                        project_language=project_language,
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
                    triage_required=triage_required,
                    portfolio_plan=state.get("portfolio_plan"),
                    delta_replay_task_ids=_validated_delta_replay_task_ids(
                        task_queue, delta_isolation_by_cluster, attempt_snapshots_by_id
                    ),
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
                    portfolio_plan=state.get("portfolio_plan"),
                    delta_replay_task_ids=_validated_delta_replay_task_ids(
                        task_queue, delta_isolation_by_cluster, attempt_snapshots_by_id
                    ),
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
                    delta_replay_task_ids=_validated_delta_replay_task_ids(
                        task_queue, delta_isolation_by_cluster, attempt_snapshots_by_id
                    ),
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
                    # pivot exists, but it is not a worker target. Let the
                    # deterministic reroute below choose QA, triage,
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

    if _final_full_scan_due(state, task_queue):
        decision = SupervisorDecision(
            decision_code=DecisionCode.FINAL_FULL_SCAN_REQUIRED,
            next_node="final_full_scan",
            target_task_ids=[],
            instructions="Run the configured final full-workspace scan before reconciliation.",
            decision_reason=(
                "All remediation tasks are terminal and the development final full scan is enabled."
            ),
        )
        resolved_target_task_ids = []
        remapped_feedback_by_task = {}
    resolved_next_node = decision.next_node
    if defer_verified_absent_portfolio_replan and resolved_next_node in {
        "update_subagent",
        "workaround_subagent",
    }:
        resolved_target_task_ids = existing_attempt_targets(
            resolved_next_node,
            resolved_target_task_ids,
        )
    else:
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
    if defer_verified_absent_portfolio_replan and resolved_next_node in {
        "update_subagent",
        "workaround_subagent",
    }:
        resolved_target_task_ids = existing_attempt_targets(
            resolved_next_node,
            resolved_target_task_ids,
        )
    else:
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
    update_dispatch_authorizations: dict[str, _UpdateDispatchAuthorization] = {}
    if resolved_next_node == "update_subagent":
        dispatchable_update_ids: list[str] = []
        for task_id in resolved_target_task_ids:
            task = task_queue.get(task_id)
            if task is None:
                continue
            plan = retry_plans_by_task.get(task_id)
            diagnostics = retry_diagnostics_by_task.get(task_id)
            delta_replay_candidate = _validated_delta_replay_candidate(
                task,
                delta_isolation_by_cluster,
                attempt_snapshots_by_id,
            )
            if defer_verified_absent_portfolio_replan and task.current_attempt_id is not None:
                snapshot = attempt_snapshots_by_id.get(task.current_attempt_id)
                if (
                    snapshot is not None
                    and snapshot.attempt_id == task.current_attempt_id
                    and snapshot.task_id == task_id
                    and snapshot.task_revision == task.task_revision
                    and snapshot.dispatch_node == "update_subagent"
                ):
                    # Resume the already committed attempt from its immutable
                    # snapshot; do not rebuild its candidate authorization.
                    dispatchable_update_ids.append(task_id)
                else:
                    errors.append(
                        f"supervisor: could not resume committed update attempt for {task_id}."
                    )
                continue
            authorization = _authorize_update_dispatch(
                task,
                plan=plan,
                diagnostics=diagnostics,
                allow_attempted_selected_version=delta_replay_candidate is not None,
                project_language=project_language,
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
                        project_language=project_language,
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
                            allow_attempted_selected_version=(
                                _validated_delta_replay_candidate(
                                    task,
                                    delta_isolation_by_cluster,
                                    attempt_snapshots_by_id,
                                )
                                is not None
                            ),
                            project_language=project_language,
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
            triage_required=triage_required,
            portfolio_plan=state.get("portfolio_plan"),
            delta_replay_task_ids=_validated_delta_replay_task_ids(
                task_queue, delta_isolation_by_cluster, attempt_snapshots_by_id
            ),
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

    # Replan only on an entry-to-final UNFIXABLE status edge.  Revision changes
    # are deliberately ignored because status-only transitions may not revise
    # the task input.
    newly_unfixable_sca_task_ids = sorted(
        task_id
        for task_id, task in task_queue.items()
        if task.status == TaskStatus.UNFIXABLE
        and task_statuses_at_entry.get(task_id) != TaskStatus.UNFIXABLE
        and (group := group_by_id.get(task.parent_group_id)) is not None
        and group.issue_type == IssueType.SCA
    )
    remaining_nonterminal_sca_task_ids = _nonterminal_sca_task_ids(task_queue, group_by_id)
    unfixable_replan_required = bool(
        newly_unfixable_sca_task_ids and remaining_nonterminal_sca_task_ids
    )
    pending_unfixable_replan = bool(
        isinstance(portfolio_replan_request, PortfolioReplanRequest)
        and portfolio_replan_request.reason == "UNFIXABLE_REPLAN"
    )
    if unfixable_replan_required and not isinstance(
        portfolio_replan_request, PortfolioReplanRequest
    ):
        portfolio_replan_request = PortfolioReplanRequest(
            reason="UNFIXABLE_REPLAN",
            source_portfolio_plan_id=(
                committed_plan.portfolio_plan_id if committed_plan is not None else None
            ),
        )
    if pending_unfixable_replan and not remaining_nonterminal_sca_task_ids:
        portfolio_replan_request = None
        pending_unfixable_replan = False
        unfixable_replan_cleared = True

    replan_after_unfixable = bool(
        remaining_nonterminal_sca_task_ids
        and (unfixable_replan_required or pending_unfixable_replan)
    )
    suppress_unfixable_replan_without_sca = bool(
        not remaining_nonterminal_sca_task_ids
        and (unfixable_replan_cleared or newly_unfixable_sca_task_ids)
    )
    if (
        suppress_unfixable_replan_without_sca
        and resolved_next_node == "portfolio"
        and not portfolio_escalation
        and not isinstance(portfolio_replan_request, PortfolioReplanRequest)
    ):
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
            delta_replay_task_ids=_validated_delta_replay_task_ids(
                task_queue, delta_isolation_by_cluster, attempt_snapshots_by_id
            ),
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
    elif replan_after_unfixable and resolved_next_node != "triage":
        open_attempt_task_ids = sorted(
            task_id
            for task_id, task in task_queue.items()
            if task.current_attempt_id is not None and task.status not in _TERMINAL_STATUSES
        )
        if open_attempt_task_ids:
            if resolved_next_node != "qa_critic":
                cluster_id = (
                    state.get("active_cluster_id") if len(open_attempt_task_ids) > 1 else None
                )
                decision = SupervisorDecision(
                    decision_code=(
                        DecisionCode.QA_READY_BATCH
                        if len(open_attempt_task_ids) > 1
                        else DecisionCode.QA_READY
                    ),
                    next_node="qa_critic",
                    target_task_ids=open_attempt_task_ids,
                    cluster_id=cluster_id,
                    instructions=(
                        "Reconcile the existing open attempt before rebuilding the portfolio."
                    ),
                    decision_reason=(
                        "UNFIXABLE_REPLAN is deferred until the open attempt barrier is clear."
                    ),
                )
                resolved_next_node = "qa_critic"
                resolved_target_task_ids = open_attempt_task_ids
                remapped_feedback_by_task = {}
        else:
            task_diagnostic = (
                f"Newly unfixable SCA task(s) {newly_unfixable_sca_task_ids}; "
                if newly_unfixable_sca_task_ids
                else "A pending UNFIXABLE_REPLAN; "
            )
            decision = SupervisorDecision(
                decision_code=DecisionCode.PORTFOLIO_PLAN_REQUIRED,
                next_node="portfolio",
                target_task_ids=[],
                instructions=("Rebuild the outer portfolio plan before dispatching another task."),
                decision_reason=(
                    f"{task_diagnostic}remaining nonterminal SCA task(s) "
                    f"{remaining_nonterminal_sca_task_ids} require replanning."
                ),
            )
            resolved_next_node = "portfolio"
            resolved_target_task_ids = []
            remapped_feedback_by_task = {}

    # Commit the exact input snapshot before exposing worker targets to the
    # graph. QA reuses the current worker attempt; update/workaround dispatches
    # always receive a new attempt identity.
    active_cluster_id: str | None = None
    active_dispatch_batch_id: str | None = None
    active_multi_package_action: MultiPackageAction | None = None
    portfolio_dispatch_rejected = False
    if (
        defer_verified_absent_portfolio_replan
        and resolved_next_node in {"update_subagent", "workaround_subagent", "qa_critic"}
        and resolved_target_task_ids
    ):
        resumed_snapshots = [
            attempt_snapshots_by_id.get(task_queue[task_id].current_attempt_id or "")
            for task_id in resolved_target_task_ids
        ]
        resumed_snapshots = [snapshot for snapshot in resumed_snapshots if snapshot is not None]
        if resumed_snapshots:
            active_cluster_id = resumed_snapshots[0].cluster_id
            active_dispatch_batch_id = resumed_snapshots[0].dispatch_batch_id
            active_multi_package_action = state.get("active_multi_package_action")
    elif resolved_next_node == "update_subagent" and decision.cluster_id:
        action = _build_multi_package_action(
            decision.cluster_id,
            resolved_target_task_ids,
            task_queue,
            group_by_id,
            project_language=project_language,
            repo_root=state.get("repo_root"),
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
    elif resolved_next_node == "qa_critic":
        if decision.cluster_id:
            active_cluster_id = decision.cluster_id
            active_dispatch_batch_id = state.get("active_dispatch_batch_id")
            active_multi_package_action = state.get("active_multi_package_action")
        else:
            qa_action = _validated_active_qa_multi_package_action(
                state.get("active_multi_package_action"),
                resolved_target_task_ids,
                task_queue,
                attempt_snapshots_by_id,
            )
            if qa_action is not None:
                active_cluster_id = qa_action.cluster_id
                active_dispatch_batch_id = qa_action.dispatch_batch_id
                active_multi_package_action = qa_action
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
            if defer_verified_absent_portfolio_replan and task.current_attempt_id is not None:
                snapshot = attempt_snapshots_by_id.get(task.current_attempt_id)
                if snapshot is None or snapshot.dispatch_node != resolved_next_node:
                    errors.append(
                        f"supervisor: preserved live attempt for {task_id} because its "
                        "committed snapshot does not match the resumed route."
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
            delta_replay_candidate = (
                _validated_delta_replay_candidate(
                    task,
                    delta_isolation_by_cluster,
                    attempt_snapshots_by_id,
                )
                if resolved_next_node == "update_subagent"
                else None
            )
            if resolved_next_node == "update_subagent" and update_authorization is None:
                update_authorization = _authorize_update_dispatch(
                    task,
                    plan=plan,
                    diagnostics=retry_diagnostics_by_task.get(task_id),
                    allow_attempted_selected_version=delta_replay_candidate is not None,
                    project_language=project_language,
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
                portfolio_plan_id=(
                    committed_plan.portfolio_plan_id if committed_plan is not None else None
                ),
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
                target_manifest_paths=(
                    _maven_attempt_target_paths(
                        group_by_id[task.parent_group_id],
                        project_language,
                        repo_root=state.get("repo_root"),
                        allow_dependency_management_creation=(
                            task.target_dependency_type == "dependencyManagement"
                        ),
                    )
                    if task.parent_group_id in group_by_id
                    else ()
                ),
                maven_target_operation=(
                    _maven_attempt_target_operation(
                        task,
                        group_by_id[task.parent_group_id],
                        project_language,
                        repo_root=state.get("repo_root"),
                    )
                    if task.parent_group_id in group_by_id
                    else None
                ),
            )
            task_queue[task_id] = task
            if delta_replay_candidate is not None:
                for cluster_id, record in list(delta_isolation_by_cluster.items()):
                    if not isinstance(record, Mapping):
                        continue
                    replay_task_ids = record.get("authoritative_qa_replay_task_ids")
                    if (
                        not isinstance(replay_task_ids, (list, tuple))
                        or task_id not in replay_task_ids
                    ):
                        continue
                    delta_isolation_by_cluster[cluster_id] = {
                        **record,
                        "authoritative_qa_replay_task_ids": [
                            replay_task_id
                            for replay_task_id in replay_task_ids
                            if replay_task_id != task_id
                        ],
                    }
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

    portfolio_dirty = (
        bool(state.get("portfolio_dirty", False))
        or portfolio_dispatch_rejected
        or unfixable_replan_required
        or pending_unfixable_replan
        or verified_absent_portfolio_replan_pending
    )
    if state.get("portfolio_plan") is not None and not suppress_unfixable_replan_without_sca:
        portfolio_dirty = portfolio_dirty or _portfolio_plan_is_stale(
            state.get("portfolio_plan"),
            task_queue,
            valid_groups,
            state.get("repo_root"),
        )
    if unfixable_replan_cleared and not portfolio_dispatch_rejected and not portfolio_escalation:
        portfolio_dirty = False
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
    final_full_scan_result = (
        prior_final_scan
        if _final_full_scan_value(prior_final_scan, "coverage_scope") == "full_workspace"
        else None
    )
    if final_full_scan_result is None and all(
        task.status in _TERMINAL_STATUSES for task in task_queue.values()
    ):
        final_full_scan_result = FinalFullScanResult(
            completed=incremental_scan_coverage_complete,
            authoritative=False,
            coverage_scope="incremental_dependency_inputs",
            coverage_complete=incremental_scan_coverage_complete,
            found_identifiers=post_remediation_scan_identifiers,
            remaining_target_identifiers=remaining_target_identifiers,
            new_identifiers=new_vulnerability_identifiers,
            found_issues=post_remediation_scan_issues,
            status=new_vulnerability_status,
            triage_required=triage_required,
            error=(
                None
                if incremental_scan_coverage_complete
                else "Incremental dependency scan coverage is incomplete."
            ),
        )
    # Keep the compact per-input snapshots in state; the shared QA report can
    # contain every package occurrence from a large lockfile and is consumed.
    qa_results_by_attempt = {
        attempt_id: (
            result.model_copy(update={"batch_scan_evidence": None})
            if result.batch_scan_evidence is not None
            else result
        )
        for attempt_id, result in qa_results_by_attempt.items()
    }
    return {
        "status": "supervisor_routed",
        "next_routing_step": resolved_next_node,
        "decision_code": decision.decision_code,
        "final_full_scan_result": final_full_scan_result,
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
        "delta_isolation_by_cluster": delta_isolation_by_cluster,
        "qa_rerun_counts_by_attempt": qa_rerun_counts_by_attempt,
        "post_remediation_scan_identifiers": post_remediation_scan_identifiers,
        "post_remediation_scan_issues": post_remediation_scan_issues,
        "new_vulnerability_identifiers": new_vulnerability_identifiers,
        "new_vulnerability_status": new_vulnerability_status,
        "triage_required": triage_required,
        "post_qa_triage_enabled": bool(state.get("post_qa_triage_enabled", True)),
        "processed_worker_attempt_ids": list(new_worker_attempt_ids),
        "processed_qa_attempt_ids": list(new_qa_attempt_ids),
        "accepted_package_inventory_by_lockfile": accepted_package_inventory_by_lockfile,
        "current_scan_observations_by_occurrence": current_scan_observations_by_occurrence,
        "latest_successful_scan_by_task": latest_successful_scan_by_task,
        "latest_incremental_scan_by_input": latest_incremental_scan_by_input,
        "setup_scan_evidence": setup_scan_evidence,
        "setup_scan_committed": setup_scan_committed,
        "consistency_events": consistency_events,
        "state_revision": state_revision,
        # constraints_ledger uses operator.add â€” return only NEW entries
        "constraints_ledger": returned_constraints,
        "errors": errors,
    }


def _invoke_portfolio_plan_tool(
    state: OrchestratorState,
    supervisor_patch: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run portfolio planning with any reconciled Supervisor state.

    Args:
        state: Incoming orchestration state before the Supervisor patch.
        supervisor_patch: Complete fresh Supervisor result to apply before
            planning, when a decision requested a portfolio replan.

    Returns:
        A combined state patch. The portfolio result takes precedence for
        overlapping fields, while Supervisor-only fields and both fresh error
        lists are preserved.
    """
    supervisor_result = dict(supervisor_patch or {})
    supervisor_errors = list(supervisor_result.get("errors", []) or [])
    portfolio_input = {**state, **supervisor_result}
    if supervisor_patch is not None:
        portfolio_input["errors"] = [
            *(state.get("errors", []) or []),
            *supervisor_errors,
        ]
    portfolio_result = run_portfolio_plan_tool(portfolio_input)
    result = {**supervisor_result, **portfolio_result}
    result["errors"] = [
        *supervisor_errors,
        *(portfolio_result.get("errors", []) or []),
    ]
    result["next_routing_step"] = route_after_portfolio(portfolio_result)
    return result


def run_supervisor_node(state: OrchestratorState) -> dict[str, Any]:
    """Invoke portfolio planning at the established Supervisor boundaries."""
    if (
        state.get("status") == "workspace_ready" and state.get("portfolio_plan") is None
    ) or state.get("status") == "triage_completed":
        return _invoke_portfolio_plan_tool(state)

    supervisor_patch = _run_supervisor_decision_node(state)
    if supervisor_patch.get("next_routing_step") == "portfolio":
        return _invoke_portfolio_plan_tool(state, supervisor_patch)
    return supervisor_patch


# ---------------------------------------------------------------------------
# Conditional-edge router
# ---------------------------------------------------------------------------


def supervisor_router(state: OrchestratorState) -> str:
    """Return the committed route, recomputing it when state is invalid."""
    if state.get("post_qa_retriage_limit_reached"):
        if state.get("next_routing_step") == "final_full_scan":
            return "final_full_scan"
        logger.warning("supervisor_router: post-QA re-triage limit reached; routing to teardown.")
        return "teardown"
    step = state.get("next_routing_step", "")
    task_queue = dict(state.get("task_queue", {}) or {})
    if step in _WORKER_NODES and task_queue and not state.get("active_target_task_ids"):
        logger.error(
            "supervisor_router: refusing worker route '%s' without active targets; "
            "failing closed to teardown.",
            step,
        )
        return "teardown"
    if step in _VALID_NEXT_NODES:
        return step
    logger.error("supervisor_router: invalid next_routing_step '%s' - recomputing.", step)

    valid_groups = list(state.get("valid_groups", []) or [])
    group_by_id = {group.group_id: group for group in valid_groups}
    decision = _deterministic_routing(
        task_queue,
        group_by_id,
        dict(state.get("qa_evaluations", {}) or {}),
        dict(state.get("retry_diagnostics_by_task", {}) or {}),
        active_target_task_ids=list(state.get("active_target_task_ids", []) or []),
        current_status=str(state.get("status") or ""),
        triage_required=bool(state.get("triage_required")),
        portfolio_plan=state.get("portfolio_plan"),
        delta_replay_task_ids=_validated_delta_replay_task_ids(
            task_queue,
            state.get("delta_isolation_by_cluster", {}) or {},
            dict(state.get("attempt_snapshots_by_id", {}) or {}),
        ),
    )
    if decision.next_node not in _VALID_NEXT_NODES:
        logger.critical(
            "supervisor_router: deterministic recomputation produced invalid node '%s'.",
            decision.next_node,
        )
        return "teardown"
    return decision.next_node
