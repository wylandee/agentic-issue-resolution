"""
qa_critic.py - Agentic QA evaluator node for the Phase 5 orchestrator.

The QA Critic now follows a single structured-evaluator architecture:

  Step 0 â€” Global Execution (deterministic Python):
    run_dependency_install â†’ run_security_scan â†’ run_unit_tests, called
    exactly once via direct
    Python helpers, with no LLM tools involved.

  Evaluator:
    One bounded read-only tool loop per dispatched task. The model must finish
    by calling the typed emit_qa_evaluation terminal tool; the normal node path
    does not invoke a batch judge.

  Python Guardrails:
    Normalize, validate, and fill missing/duplicate/unknown evaluations
    deterministically.  Enforce scanner and install-error policies.

The node preserves existing per-task QA evaluation semantics while adding
graph-level scan snapshot outputs for vulnerabilities introduced during
remediation.

Heavy QA commands (install, scan, tests) are intentionally *not* exposed as
tools to the update or workaround subagents; they live here only.
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any

from langsmith import traceable

from remediation_engine.contracts.schemas import (
    AgentActionSummary,
    FailureCategory,
    FinalFullScanResult,
    QAEvaluation,
    QAFailureEvidence,
    QAPolicy,
    ScanFallbackReason,
    ScannerExecutionStatus,
    ScanScope,
    VulnerabilityGroup,
)
from remediation_engine.orchestration.state import OrchestratorState
from remediation_engine.orchestration.task_utils import is_no_fix_package_removal_task
from remediation_engine.runtime.sandbox_mgr import DockerSandbox
from remediation_engine.tools.lockfile_closure import TargetPackageResolution

from . import qa_evaluator as _qa_evaluator_module
from . import qa_odc as _qa_odc_module
from . import qa_policy_engine as _qa_policy_engine_module
from . import qa_test_parsing as _qa_test_parsing_module
from ._qa_runtime import (
    _attempt_version_evidence,
    _build_qa_scan_targets,
    _build_qa_task_contexts,
    _cleanup_targeted_artifacts,
    _collect_baseline_identifiers,
    _collect_group_package_state,
    _collect_target_identifiers,
    _derive_qa_task_policies,
    _derive_qa_task_strategies,
    _resolve_action_summary_task_ids,
    _resolve_targeted_packages,
    _scan_evidence,
    _scan_state_projection,
    _store_scan_outcome,
    _targeted_extra_args_conflict,
    _workspace_remediation_fingerprint,
    _write_targeted_artifacts,
)
from .qa_policy_engine import (
    _attach_failure_evidence_to_evaluations,
    _attach_scan_evidence_to_evaluations,
    _extract_deterministic_test_evidence,
)
from .qa_types import (
    QAScanTarget,
    _append_qa_log_records,
    _QAExecutionResults,
    _QALogRecord,
    _QAPackageState,
    _scan_result_value,
    _SecurityScanResult,
)

logger = logging.getLogger(__name__)

# TEMPORARY: disable npm's installed-tree inspection while diagnosing QA results.
_QA_NPM_LS_ENABLED = False


def _run_global_execution(
    sandbox: DockerSandbox,
    workspace_volume: str,
    target_identifiers: set[str],
    baseline_identifiers: set[str] | None = None,
    scan_targets: Sequence[QAScanTarget] | None = None,
    skip_scan: bool = False,
    scan_skip_reason: str | None = None,
    package_state_collector: Callable[[DockerSandbox, bool], Mapping[str, _QAPackageState]]
    | None = None,
) -> _QAExecutionResults:
    """
    Run install, optional package evidence, security scan, and unit tests in order.

    Package evidence is collected immediately after install. No LLM tool wrappers
    are involved; results are stored in a _QAExecutionResults cache for downstream
    use.
    """
    results = _QAExecutionResults()

    logger.info("qa_critic: [Step 0] running npm install.")
    install_outcome = _qa_test_parsing_module._run_install(sandbox)
    _qa_test_parsing_module._store_install_outcome(results, install_outcome)
    install_ok = results.install[0]
    if package_state_collector is not None:
        results.package_state_by_task.update(package_state_collector(sandbox, install_ok))

    if skip_scan:
        results.scan_skipped = True
        results.scan_skip_reason = scan_skip_reason or "explicitly skipped by QA policy"
        _append_qa_log_records(
            results,
            "scan",
            (
                _QALogRecord(
                    phase="scan",
                    label="odc:skipped",
                    exit_code=None,
                    stdout="",
                    stderr="",
                    error=results.scan_skip_reason,
                ),
            ),
        )
        logger.info(
            "qa_critic: [Step 0] skipping security scan (%s); install_ok=%s.",
            results.scan_skip_reason,
            install_ok,
        )
    else:
        logger.info("qa_critic: [Step 0] running security scan (install_ok=%s).", install_ok)
    scan_started = time.monotonic()
    if not skip_scan and scan_targets is None:
        if baseline_identifiers is None:
            _store_scan_outcome(
                results,
                _qa_odc_module._run_security_scan(
                    sandbox,
                    workspace_volume,
                    target_identifiers,
                ),
            )
        else:
            _store_scan_outcome(
                results,
                _qa_odc_module._run_security_scan(
                    sandbox,
                    workspace_volume,
                    target_identifiers,
                    baseline_identifiers,
                ),
            )
    elif not skip_scan:
        baseline = baseline_identifiers or target_identifiers
        target_packages: list[TargetPackageResolution] = []
        covered_task_ids: list[str] = []
        targeted_subdir: str | None = None
        incomplete_reason: ScanFallbackReason | None = None
        incomplete_detail: str | None = None
        targeted_result: _SecurityScanResult | None = None
        scanner_coverage_complete = False
        try:
            if _targeted_extra_args_conflict():
                incomplete_reason = ScanFallbackReason.TARGETED_SCAN_FAILED
                incomplete_detail = "ODC_EXTRA_ARGS conflicts with safe targeted scan scope."
            else:
                resolution_batch = _resolve_targeted_packages(sandbox, scan_targets)
                target_packages = list(resolution_batch.target_packages)
                results.scan_target_failures_by_task = {
                    task_id: (failure.reason, failure.detail)
                    for task_id, failure in resolution_batch.failures_by_task.items()
                }
                if resolution_batch.fallback_detail:
                    logger.info(
                        "qa_critic: some targeted packages could not be resolved: %s",
                        resolution_batch.fallback_detail,
                    )
                if not target_packages:
                    incomplete_reason = (
                        resolution_batch.fallback_reason
                        or ScanFallbackReason.NO_MATCHING_TARGET
                    )
                    incomplete_detail = (
                        resolution_batch.fallback_detail
                        or "No task target resolved to a scannable package."
                    )
            if incomplete_reason is None and target_packages:
                targeted_subdir = ".odc-targeted"
                _write_targeted_artifacts(sandbox, target_packages)
                targeted_result = _qa_odc_module._run_targeted_security_scan(
                    sandbox,
                    workspace_volume,
                    target_identifiers,
                    baseline,
                    targeted_subdir,
                )
                if targeted_result.execution_status != ScannerExecutionStatus.SUCCESS:
                    incomplete_reason = (
                        ScanFallbackReason.TARGETED_REPORT_UNPARSEABLE
                        if targeted_result.execution_status == ScannerExecutionStatus.UNPARSEABLE
                        else ScanFallbackReason.TARGETED_SCAN_FAILED
                    )
                    incomplete_detail = targeted_result.summary
                elif targeted_result.reported_packages is None:
                    incomplete_reason = ScanFallbackReason.TARGETED_REPORT_UNPARSEABLE
                    incomplete_detail = "ODC report did not contain parseable package identities."
                else:
                    expected_packages: list[tuple[str, str]] = []
                    seen_physical_entries: set[tuple[str, str, str]] = set()
                    for resolution in target_packages:
                        node = resolution.target_node
                        if node is None:
                            continue
                        physical_identity = (
                            resolution.source_lockfile,
                            node.lockfile_key,
                            node.version,
                        )
                        if physical_identity in seen_physical_entries:
                            continue
                        seen_physical_entries.add(physical_identity)
                        expected_packages.append((node.package_name.casefold(), node.version))
                    missing_packages = Counter(expected_packages) - Counter(
                        targeted_result.reported_packages
                    )
                    if missing_packages:
                        missing_text = ", ".join(
                            f"{name}@{version} (missing {count})"
                            for (name, version), count in sorted(missing_packages.items())
                        )
                        incomplete_reason = ScanFallbackReason.TARGET_PACKAGE_NOT_REPORTED
                        incomplete_detail = (
                            "ODC report did not enumerate every targeted package occurrence: "
                            f"{missing_text}."
                        )
                    else:
                        scanner_coverage_complete = True
                        covered_task_ids = list(resolution_batch.covered_task_ids)
                        if resolution_batch.failures_by_task:
                            incomplete_reason = resolution_batch.fallback_reason
                            incomplete_detail = resolution_batch.fallback_detail

            if scanner_coverage_complete and targeted_result is not None:
                _store_scan_outcome(results, targeted_result, label="odc:targeted")
                results.scan_evidence = _scan_evidence(
                    targets=scan_targets,
                    scan_result=targeted_result,
                    effective_scope=ScanScope.TARGETED,
                    complete=not results.scan_target_failures_by_task,
                    fallback_reason=(
                        resolution_batch.fallback_reason
                        if results.scan_target_failures_by_task
                        else None
                    ),
                    target_packages=target_packages,
                    covered_task_ids=covered_task_ids,
                )
        except (OSError, RuntimeError, ValueError) as exc:
            logger.warning("qa_critic: targeted scan setup failed — %s", exc)
            incomplete_reason = ScanFallbackReason.TARGETED_SCAN_FAILED
            incomplete_detail = str(exc)
        finally:
            if targeted_subdir is not None:
                _cleanup_targeted_artifacts(sandbox)

        if not scanner_coverage_complete:
            incomplete_reason = incomplete_reason or ScanFallbackReason.TARGETED_SCAN_FAILED
            summary = (
                "INCONCLUSIVE: targeted Dependency-Check evidence is incomplete. "
                f"Reason: {incomplete_detail or incomplete_reason.value}"
            )
            if targeted_result is None:
                incomplete_result = _SecurityScanResult(
                    False,
                    summary,
                    set(),
                    set(),
                    set(),
                    execution_status=ScannerExecutionStatus.NOT_RUN,
                )
            else:
                execution_status = targeted_result.execution_status
                if execution_status == ScannerExecutionStatus.SUCCESS:
                    execution_status = ScannerExecutionStatus.UNPARSEABLE
                incomplete_result = replace(
                    targeted_result,
                    ok=False,
                    summary=summary,
                    remaining_identifiers=set(),
                    found_identifiers=set(),
                    new_identifiers=set(),
                    found_issues=[],
                    execution_status=execution_status,
                )
            _store_scan_outcome(results, incomplete_result, label="odc:targeted")
            results.scan_evidence = _scan_evidence(
                targets=scan_targets,
                scan_result=incomplete_result,
                effective_scope=ScanScope.TARGETED,
                complete=False,
                fallback_reason=incomplete_reason,
                target_packages=target_packages,
                covered_task_ids=[],
            )

    if results.scan_skipped:
        logger.info(
            "qa_critic: scan requested_scope=skipped effective_scope=skipped "
            "reason=%s duration_seconds=%.3f",
            results.scan_skip_reason,
            time.monotonic() - scan_started,
        )
    elif results.scan_evidence is not None:
        logger.info(
            "qa_critic: scan requested_scope=%s effective_scope=%s tasks=%d target_packages=%d "
            "incomplete_reason=%s duration_seconds=%.3f",
            results.scan_evidence.requested_scope.value,
            results.scan_evidence.effective_scope.value,
            len(results.scan_evidence.covered_task_ids),
            len(results.scan_evidence.closure_package_names),
            results.scan_evidence.fallback_reason.value
            if results.scan_evidence.fallback_reason
            else None,
            time.monotonic() - scan_started,
        )
    else:
        logger.info(
            "qa_critic: scan requested_scope=full effective_scope=full tasks=0 "
            "target_packages=0 fallback_reason=None duration_seconds=%.3f",
            time.monotonic() - scan_started,
        )

    logger.info("qa_critic: [Step 0] running unit tests.")
    test_outcome = _qa_test_parsing_module._run_unit_tests(sandbox)
    _qa_test_parsing_module._store_test_outcome(results, test_outcome)

    return results


def _create_skinny_qa_group(group: VulnerabilityGroup) -> VulnerabilityGroup:
    """Create a skinny copy of a group for QA evaluator agents."""
    return group.model_copy(
        update={
            "issues": [],
            "localized_issues": [],
        }
    )


def _filter_recent_action_summaries(
    action_summaries: list[AgentActionSummary], task_ids: list[str]
) -> list[AgentActionSummary]:
    """Extract only the most recently appended summary for each task ID."""
    known_task_ids = set(task_ids)
    recent_summaries = []
    seen_ids = set()
    for summary in reversed(action_summaries):
        resolved_ids = _resolve_action_summary_task_ids(summary, known_task_ids)
        new_ids = set(resolved_ids) - seen_ids
        if new_ids:
            recent_summaries.insert(0, summary)
            seen_ids.update(new_ids)
        if seen_ids == known_task_ids:
            break
    return recent_summaries


def _changed_files_by_task(
    state: OrchestratorState,
    task_contexts: Sequence[Any],
) -> dict[str, list[str]]:
    """Project accepted worker changed files onto each active task."""
    worker_results = state.get("worker_results_by_attempt") or {}
    changed_by_task: dict[str, list[str]] = {}
    for context in task_contexts:
        task = context.task
        attempt_id = getattr(task, "current_attempt_id", None)
        result = worker_results.get(attempt_id) if attempt_id else None
        if isinstance(result, Mapping):
            raw_files = result.get("changed_files") or []
        else:
            raw_files = getattr(result, "changed_files", []) or []
        task_files: list[str] = []
        seen: set[str] = set()
        for path in raw_files:
            if isinstance(path, str) and path.strip() and path not in seen:
                seen.add(path)
                task_files.append(path)
        changed_by_task[context.task_id] = task_files
    return changed_by_task


@traceable(name="qa_critic")
def run_qa_critic_node(state: OrchestratorState) -> dict[str, Any]:
    """
    LangGraph node: run the structured QA evaluator for the supplied group scope.

    Normal Supervisor dispatches supply one task and therefore one parent group.
    Direct callers may still provide multiple groups; each group is evaluated
    independently and returns one structured result.

    Pipeline:
      Step 0 â€” Global Execution (deterministic Python, no LLM tools)
      Evaluator â€” One bounded read-only tool loop with a typed terminal result
      Guards â€” Python guardrails normalize and validate evaluations
    """
    valid_groups: list[VulnerabilityGroup] = state.get("valid_groups") or []
    workspace_volume: str | None = state.get("workspace_volume")
    repo_root: str | None = state.get("repo_root")
    action_summaries: list[AgentActionSummary] = state.get("action_summaries") or []
    task_queue = state.get("task_queue") or {}
    active_task_ids = list(state.get("active_target_task_ids") or [])

    if not valid_groups and not state.get("force_qa"):
        logger.info("qa_critic: no valid groups - skipping QA.")
        return {
            "qa_evaluations": {},
            "eval_status": "all_passed",
            "status": "qa_completed",
            "changed_files": [],
            "qa_investigation_report": "",
            **_scan_state_projection(
                _QAExecutionResults(),
                _collect_baseline_identifiers(state, valid_groups),
                authoritative=False,
            ),
        }

    if not active_task_ids:
        err = "qa_critic: active_target_task_ids is required for scoped QA dispatch."
        logger.error(err)
        return {
            "qa_evaluations": {},
            "eval_status": "failures_detected",
            "status": "qa_failed",
            "errors": [err],
            "changed_files": [],
            "qa_investigation_report": "",
            **_scan_state_projection(
                _QAExecutionResults(),
                _collect_baseline_identifiers(state, valid_groups),
                authoritative=False,
            ),
        }

    try:
        task_contexts = _build_qa_task_contexts(valid_groups, task_queue, active_task_ids)
    except ValueError as exc:
        err = f"qa_critic: invalid task scope: {exc}"
        logger.error(err)
        return {
            "qa_evaluations": {},
            "eval_status": "failures_detected",
            "status": "qa_failed",
            "errors": [err],
            "changed_files": [],
            "qa_investigation_report": "",
        }

    task_strategies = _derive_qa_task_strategies(
        valid_groups,
        state.get("group_strategies") or {},
        task_queue,
        active_task_ids,
    )
    task_policies = _derive_qa_task_policies(
        valid_groups,
        task_queue,
        active_task_ids,
        state.get("attempt_snapshots_by_id"),
    )
    changed_files_by_task = _changed_files_by_task(state, task_contexts)
    candidate_changed_files: list[str] = sorted(
        {path for paths in changed_files_by_task.values() for path in paths}
    )
    baseline_identifiers = _collect_baseline_identifiers(state, valid_groups)
    scan_targets = _build_qa_scan_targets(state, valid_groups)
    active_tasks = [context.task for context in task_contexts]
    skip_scan = all(is_no_fix_package_removal_task(task) for task in active_tasks) and all(
        policy == QAPolicy.NO_FIX_PACKAGE_REMOVAL for policy in task_policies.values()
    )
    scan_is_authoritative = scan_targets is None and not skip_scan
    unscanned_projection = _scan_state_projection(
        _QAExecutionResults(),
        baseline_identifiers,
        authoritative=scan_is_authoritative,
    )

    if not workspace_volume:
        err = "qa_critic: workspace_volume is not set; cannot run QA."
        logger.error(err)
        failed_evals = {
            context.task_id: QAEvaluation(
                task_id=context.task_id,
                passed=False,
                contract_error=True,
                contract_error_reason=err,
                failure_category=FailureCategory.SECURITY_FLAG,
                retry_feedback="QA infrastructure failure: workspace_volume is missing.",
            )
            for context in task_contexts
        }
        return {
            "qa_evaluations": failed_evals,
            "eval_status": "failures_detected",
            "status": "qa_failed",
            "errors": [err],
            "changed_files": [],
            "qa_investigation_report": "",
            **unscanned_projection,
        }

    target_identifiers = _collect_target_identifiers(valid_groups)
    logger.info(
        "qa_critic: evaluating %d groups with %d target identifiers.",
        len(valid_groups),
        len(target_identifiers),
    )

    source_groups_by_id = {group.group_id: group for group in valid_groups}
    skinny_groups_by_id = {group.group_id: _create_skinny_qa_group(group) for group in valid_groups}
    task_contexts = [
        replace(context, group=skinny_groups_by_id[context.group.group_id])
        for context in task_contexts
    ]
    action_summaries = _filter_recent_action_summaries(action_summaries, active_task_ids)

    errors: list[str] = []
    deterministic_test_evidence: QAFailureEvidence | None = None

    def collect_package_states(
        sandbox: DockerSandbox, install_ok: bool
    ) -> Mapping[str, _QAPackageState]:
        package_states: dict[str, _QAPackageState] = {}
        for context in task_contexts:
            task_id = context.task_id
            policy = task_policies.get(task_id)
            if policy not in {
                QAPolicy.VERSION_BUMP,
                QAPolicy.NO_FIX_PACKAGE_REMOVAL,
                QAPolicy.NO_FIX_CODE_REMOVAL,
            }:
                continue
            expected_version = None
            version_evidence_inconclusive = False
            if policy == QAPolicy.VERSION_BUMP:
                expected_version, has_version_evidence = _attempt_version_evidence(
                    state, context.task
                )
                version_evidence_inconclusive = has_version_evidence and expected_version is None
            package_states[task_id] = _collect_group_package_state(
                sandbox,
                source_groups_by_id[context.group.group_id],
                policy,
                task=context.task,
                expected_version=expected_version,
                version_evidence_inconclusive=version_evidence_inconclusive,
                inspect_resolved_graph=install_ok and _QA_NPM_LS_ENABLED,
                npm_ls_disabled=install_ok and not _QA_NPM_LS_ENABLED,
            )
        return package_states

    try:
        with DockerSandbox(repo_root=None, workspace_volume=workspace_volume) as sandbox:
            # ------------------------------------------------------------------
            # Step 0: Global Execution (deterministic Python, exactly once)
            # ------------------------------------------------------------------
            results = _run_global_execution(
                sandbox=sandbox,
                workspace_volume=workspace_volume,
                target_identifiers=target_identifiers,
                baseline_identifiers=baseline_identifiers,
                scan_targets=scan_targets,
                skip_scan=skip_scan,
                scan_skip_reason="no_fix_package_removal" if skip_scan else None,
                package_state_collector=(
                    collect_package_states
                    if _qa_policy_engine_module._QA_PACKAGE_STATE_CHECKS_ENABLED
                    else None
                ),
            )
            scan_projection = _scan_state_projection(
                results,
                baseline_identifiers,
                authoritative=scan_is_authoritative,
            )
            # ------------------------------------------------------------------
            # Pipeline completeness guard
            # ------------------------------------------------------------------
            missing_tools: list[str] = []
            if results.install is None:
                missing_tools.append("run_dependency_install")
            if results.scan is None and not results.scan_skipped:
                missing_tools.append("run_security_scan")
            if results.tests is None:
                missing_tools.append("run_unit_tests")
            if missing_tools:
                missing_list = ", ".join(missing_tools)
                err = (
                    f"qa_critic: global execution did not complete all required steps; "
                    f"missing: {missing_list}."
                )
                logger.error(err)
                errors.append(err)
                for tool_name in missing_tools:
                    errors.append(
                        f"qa_critic: required tool '{tool_name}' did not produce a result."
                    )
                failed_evals = {
                    context.task_id: QAEvaluation(
                        task_id=context.task_id,
                        passed=False,
                        contract_error=True,
                        contract_error_reason=err,
                        failure_category=FailureCategory.SECURITY_FLAG,
                        retry_feedback=(
                            f"QA global execution incomplete: {missing_list} did not run. "
                            "Please retry."
                        ),
                    )
                    for context in task_contexts
                }
                failed_evals = _attach_scan_evidence_to_evaluations(
                    failed_evals,
                    results.scan_evidence,
                    task_contexts,
                    task_failures=results.scan_target_failures_by_task,
                )
                return {
                    "qa_evaluations": failed_evals,
                    "eval_status": "failures_detected",
                    "status": "qa_failed",
                    "scan_skipped": results.scan_skipped,
                    "errors": errors,
                    "changed_files": [],
                    "qa_investigation_report": "",
                    "scan_evidence": results.scan_evidence,
                    **scan_projection,
                }

            if (
                results.scan_evidence is not None
                and not results.scan_evidence.complete
                and not results.scan_evidence.covered_task_ids
            ):
                reason = (
                    results.scan_evidence.fallback_reason.value
                    if results.scan_evidence.fallback_reason
                    else "targeted evidence incomplete"
                )
                detail = (
                    "Targeted ODC did not provide verifiable package coverage "
                    f"({reason}); QA cannot confirm the target CVE status."
                )
                scan_summary = str(_scan_result_value(results.scan, "summary", "") or "").strip()
                if scan_summary:
                    detail = f"{detail} {scan_summary}"
                errors.append(f"qa_critic: {detail}")
                deterministic_gates, _gate_errors = _qa_policy_engine_module._evaluate_policy_gates(
                    task_contexts,
                    results,
                    task_policies,
                )
                inconclusive_evaluations = {
                    context.task_id: QAEvaluation(
                        task_id=context.task_id,
                        passed=False,
                        failure_category=FailureCategory.SECURITY_FLAG,
                        retry_feedback=(
                            f"{detail} Correct the targeted scan evidence and re-run QA; "
                            "this result does not consume a remediation retry."
                        ),
                        evidence_inconclusive=True,
                        deterministic_gates=deterministic_gates.get(context.task_id),
                    )
                    for context in task_contexts
                }
                inconclusive_evaluations = _attach_scan_evidence_to_evaluations(
                    inconclusive_evaluations,
                    results.scan_evidence,
                    task_contexts,
                    task_failures=results.scan_target_failures_by_task,
                )
                report = json.dumps(
                    {
                        "scan_evidence_inconclusive": detail,
                        "evaluations": {
                            task_id: evaluation.model_dump(mode="json")
                            for task_id, evaluation in sorted(inconclusive_evaluations.items())
                        },
                        "scan_projection": scan_projection,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                )
                return {
                    "qa_evaluations": inconclusive_evaluations,
                    "eval_status": "failures_detected",
                    "status": "qa_completed",
                    "scan_skipped": results.scan_skipped,
                    "errors": errors,
                    "changed_files": candidate_changed_files,
                    "qa_investigation_report": report,
                    "scan_evidence": results.scan_evidence,
                    **scan_projection,
                }

            if results.tests and not results.tests[0]:
                deterministic_test_evidence = _extract_deterministic_test_evidence(
                    results,
                    sandbox=sandbox,
                )

            # ------------------------------------------------------------------
            # Structured evaluator: one bounded tool-enabled evaluator per task
            # ------------------------------------------------------------------
            investigations_by_task = _qa_evaluator_module._run_individual_investigations(
                task_contexts=task_contexts,
                task_strategies=task_strategies,
                action_summaries=action_summaries,
                changed_files_by_task=changed_files_by_task,
                sandbox=sandbox,
                repo_root=repo_root,
                results=results,
                task_policies=task_policies,
            )

    except RuntimeError as exc:
        err = f"qa_critic: Docker sandbox unavailable - {exc}"
        logger.error(err)
        errors.append(err)
        failed_evals = {
            context.task_id: QAEvaluation(
                task_id=context.task_id,
                passed=False,
                contract_error=True,
                contract_error_reason=err,
                failure_category=FailureCategory.SECURITY_FLAG,
                retry_feedback="QA infrastructure failure: Docker sandbox could not start.",
            )
            for context in task_contexts
        }
        return {
            "qa_evaluations": failed_evals,
            "eval_status": "failures_detected",
            "status": "qa_failed",
            "errors": errors,
            "changed_files": [],
            "qa_investigation_report": "",
            **unscanned_projection,
        }

    # Collect evaluator errors and normalize the one structured result per task.
    for investigation in investigations_by_task.values():
        errors.extend(investigation.errors)
    llm_evaluations = {
        task_id: investigation.evaluation
        for task_id, investigation in investigations_by_task.items()
        if investigation.evaluation is not None
    }
    qa_evaluations, guardrail_errors = _qa_policy_engine_module._apply_guardrails(
        task_contexts=task_contexts,
        batch_result=llm_evaluations,
        results=results,
        task_policies=task_policies,
        investigations_by_task=investigations_by_task,
    )
    errors.extend(guardrail_errors)
    qa_evaluations = _attach_failure_evidence_to_evaluations(
        qa_evaluations,
        results,
        state,
        deterministic_evidence=deterministic_test_evidence,
    )
    qa_evaluations = _attach_scan_evidence_to_evaluations(
        qa_evaluations,
        results.scan_evidence,
        task_contexts,
        task_failures=results.scan_target_failures_by_task,
    )
    for task_id, (reason, detail) in results.scan_target_failures_by_task.items():
        current = qa_evaluations.get(task_id)
        if current is None:
            continue
        failure_detail = (
            f"Task {task_id} targeted package could not be scanned "
            f"({reason.value}): {detail}"
        )
        errors.append(f"qa_critic: {failure_detail}")
        inconclusive_gates = current.deterministic_gates
        if inconclusive_gates is not None:
            inconclusive_gates = inconclusive_gates.model_copy(
                update={
                    "status": "fail",
                    "scanner_execution_status": ScannerExecutionStatus.NOT_RUN,
                    "target_remaining_identifiers": [],
                    "target_scanner_cleared": None,
                    "diagnostics": [*inconclusive_gates.diagnostics, failure_detail],
                }
            )
        qa_evaluations[task_id] = QAEvaluation(
            task_id=task_id,
            passed=False,
            failure_category=FailureCategory.SECURITY_FLAG,
            retry_feedback=(
                f"{failure_detail} QA is inconclusive and this result does not consume a "
                "remediation retry. Make the target package resolvable for targeted ODC, "
                "or provide valid evidence that the task no longer requires scanning."
            ),
            deterministic_gates=inconclusive_gates,
            evidence_inconclusive=True,
            scan_evidence=current.scan_evidence,
        )
    qa_errors_by_task = {
        task_id: list(investigation.errors)
        for task_id, investigation in investigations_by_task.items()
    }
    for task_id, (_reason, detail) in results.scan_target_failures_by_task.items():
        qa_errors_by_task.setdefault(task_id, []).append(detail)
    qa_investigation_reports_by_task = {
        task_id: json.dumps(
            {
                "task_id": task_id,
                "group_id": investigation.group_id,
                "evaluation": (
                    qa_evaluations[task_id].model_dump(mode="json")
                    if task_id in qa_evaluations
                    else None
                ),
                "tool_transcript": investigation.tool_transcript,
                "errors": qa_errors_by_task.get(task_id, []),
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        for task_id, investigation in sorted(investigations_by_task.items())
    }
    all_passed = all(evaluation.passed for evaluation in qa_evaluations.values())
    eval_status = "all_passed" if all_passed else "failures_detected"
    logger.info(
        "qa_critic: eval_status=%s (%d/%d passed).",
        eval_status,
        sum(1 for evaluation in qa_evaluations.values() if evaluation.passed),
        len(qa_evaluations),
    )

    qa_investigation_report = json.dumps(
        {
            "evaluations": {
                task_id: evaluation.model_dump(mode="json")
                for task_id, evaluation in sorted(qa_evaluations.items())
            },
            "scan_projection": scan_projection,
        },
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )

    return {
        "qa_evaluations": qa_evaluations,
        "eval_status": eval_status,
        "status": "qa_completed",
        "scan_skipped": results.scan_skipped,
        "changed_files": candidate_changed_files,
        "errors": errors,
        "qa_investigation_report": qa_investigation_report,
        "scan_evidence": results.scan_evidence,
        "qa_investigation_reports_by_task": qa_investigation_reports_by_task,
        "qa_errors_by_task": qa_errors_by_task,
        **scan_projection,
    }


@traceable(name="final_full_scan")
def run_final_full_scan_node(state: OrchestratorState) -> dict[str, Any]:
    """Run the authoritative full ODC scan immediately before teardown."""
    workspace_volume = state.get("workspace_volume")
    groups: list[VulnerabilityGroup] = list(state.get("valid_groups") or [])
    baseline = _collect_baseline_identifiers(state, groups)
    target_identifiers = _collect_target_identifiers(groups)
    previous_workspace_fingerprint = state.get("final_scan_workspace_fingerprint")
    if not workspace_volume:
        error = "final_full_scan: workspace_volume is missing."
        result = FinalFullScanResult(
            completed=False,
            found_identifiers=[],
            remaining_target_identifiers=[],
            new_identifiers=[],
            status="scan_failed",
            triage_required=False,
            error=error,
        )
        return {
            "final_full_scan_result": result,
            "final_full_scan_completed": True,
            "new_vulnerability_status": "scan_failed",
            "triage_required": False,
            "status": "final_scan_failed",
            "errors": [error],
        }

    try:
        with DockerSandbox(repo_root=None, workspace_volume=workspace_volume) as sandbox:
            scan = _qa_odc_module._run_security_scan(
                sandbox,
                workspace_volume,
                target_identifiers,
                baseline,
            )
            workspace_fingerprint = _workspace_remediation_fingerprint(
                str(state.get("repo_root")) if state.get("repo_root") else None,
                sandbox,
                list(state.get("changed_files") or []),
            )
    except Exception as exc:  # noqa: BLE001 - scan failures must reach teardown/report
        error = f"final_full_scan: Docker sandbox unavailable - {exc}"
        result = FinalFullScanResult(
            completed=False,
            found_identifiers=[],
            remaining_target_identifiers=[],
            new_identifiers=[],
            status="scan_failed",
            triage_required=False,
            error=error,
        )
        return {
            "final_full_scan_result": result,
            "final_full_scan_completed": True,
            "new_vulnerability_status": "scan_failed",
            "triage_required": False,
            "status": "final_scan_failed",
            "errors": [error],
        }

    found = sorted(scan.found_identifiers)
    remaining = sorted(scan.remaining_identifiers)
    new_identifiers = sorted(scan.new_identifiers)
    hard_failure = not scan.ok and not scan.found_identifiers and not scan.remaining_identifiers
    status = (
        "scan_failed"
        if hard_failure
        else "detected"
        if new_identifiers
        else "unresolved"
        if remaining
        else "none"
    )
    triage_required = bool(new_identifiers or remaining)
    result = FinalFullScanResult(
        completed=not hard_failure,
        found_identifiers=found,
        remaining_target_identifiers=remaining,
        new_identifiers=new_identifiers,
        found_issues=list(scan.found_issues),
        status=status,
        triage_required=triage_required,
        error=scan.summary if hard_failure else None,
    )
    logger.info(
        "final_full_scan: completed=%s found=%d remaining=%d new=%d triage_required=%s",
        result.completed,
        len(found),
        len(remaining),
        len(new_identifiers),
        triage_required,
    )
    return {
        "final_full_scan_result": result,
        "final_full_scan_completed": True,
        "previous_final_scan_workspace_fingerprint": previous_workspace_fingerprint,
        "final_scan_workspace_fingerprint": workspace_fingerprint,
        "baseline_scan_identifiers": sorted(baseline),
        "post_remediation_scan_identifiers": found,
        "post_remediation_scan_issues": list(scan.found_issues),
        "new_vulnerability_identifiers": new_identifiers,
        "new_vulnerability_status": status,
        "triage_required": triage_required,
        "status": "final_scan_completed" if not hard_failure else "final_scan_failed",
        "errors": [scan.summary] if hard_failure else [],
    }
