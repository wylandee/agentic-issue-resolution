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
from pathlib import Path
from typing import Any

from langsmith import traceable

from remediation_engine.contracts.schemas import (
    AgentActionSummary,
    FailureCategory,
    ODCScanEvidence,
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
    _attribute_reported_occurrences,
    _build_qa_scan_targets,
    _build_qa_task_contexts,
    _changed_package_lock_inputs,
    _cleanup_targeted_artifacts,
    _collect_baseline_identifiers,
    _collect_group_package_state,
    _collect_target_identifiers,
    _derive_qa_task_policies,
    _derive_qa_task_strategies,
    _parse_package_occurrence_key,
    _resolve_action_summary_task_ids,
    _resolve_targeted_packages,
    _scan_evidence,
    _scan_evidence_by_task,
    _scan_state_projection,
    _store_scan_outcome,
    _targeted_extra_args_conflict,
    _targeted_project_for_report_path,
    _TargetedPackageBatch,
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


def _missing_targeted_input_paths(
    artifact_manifest: Any,
    reported_paths: Sequence[str],
) -> list[str]:
    """Return staged input paths not represented in ODC dependency file paths."""
    paths_by_project: dict[str, set[str]] = {}
    for raw_path in reported_paths:
        normalized = str(raw_path or "").strip().replace("\\", "/")
        while normalized.startswith("./"):
            normalized = normalized[2:]
        parts = [part for part in normalized.split("/") if part]
        if len(parts) < 3 or parts[0] != ".odc-targeted":
            continue
        relative_path = "/".join(parts[2:])
        if relative_path.rsplit("/", 1)[-1] in {
            _qa_odc_module._ODC_REPORT_NAME,
            _qa_odc_module._ODC_HTML_REPORT_NAME,
        }:
            continue
        project_path = f"{parts[0]}/{parts[1]}"
        paths_by_project.setdefault(project_path, set()).add(relative_path)

    missing_paths: set[str] = set()
    for project_path, input_paths in artifact_manifest.input_paths_by_project.items():
        report_files = paths_by_project.get(project_path, set())
        if not report_files:
            missing_paths.update(input_paths)
            continue
        if artifact_manifest.occurrences_by_project.get(project_path):
            continue
        for input_path in input_paths:
            basename = input_path.replace("\\", "/").rsplit("/", 1)[-1]
            if not any(
                report_file == basename or report_file.startswith(f"{basename}/")
                for report_file in report_files
            ):
                missing_paths.add(input_path)
    return sorted(missing_paths)


def _removed_target_task_ids(batch: Any, targets: Sequence[QAScanTarget]) -> set[str]:
    """Return active tasks whose exact target occurrence has a tombstone."""
    removed_occurrences = [
        parsed
        for occurrence_key in batch.inventory_removals
        if (parsed := _parse_package_occurrence_key(occurrence_key)) is not None
        and parsed[0] in batch.covered_input_paths
    ]
    removed_task_ids: set[str] = set()
    for target in targets:
        target_paths = {str(path).replace("\\", "/") for path in target.manifest_paths}
        target_name = target.target_package.casefold()
        if any(
            source_path in target_paths
            and package_name == target_name
            and (not target.expected_version or version == target.expected_version)
            for source_path, _lockfile_key, package_name, version, _index in removed_occurrences
        ):
            removed_task_ids.add(target.task_id)
    return removed_task_ids


def _clear_expected_removed_target_failures(
    batch: Any,
    targets: Sequence[QAScanTarget],
) -> Any:
    """Ignore target-resolution failures proven to be intentional removals."""
    removed_task_ids = _removed_target_task_ids(batch, targets)
    removed_paths = {
        parsed[0]
        for occurrence_key in batch.inventory_removals
        if (parsed := _parse_package_occurrence_key(occurrence_key)) is not None
        and parsed[0] in batch.covered_input_paths
    }
    target_paths_by_task = {
        target.task_id: {str(path).replace("\\", "/") for path in target.manifest_paths}
        for target in targets
    }
    failures: dict[str, Any] = {}
    for task_id, failure in batch.failures_by_task.items():
        if task_id in removed_task_ids and failure.reason == ScanFallbackReason.NO_MATCHING_TARGET:
            continue
        if (
            task_id in removed_task_ids
            and failure.reason == ScanFallbackReason.MISSING_LOCKFILE
            and any(
                path in target_paths_by_task[task_id] and path in failure.detail
                for path in removed_paths
            )
        ):
            continue
        failures[task_id] = failure
    first_failure = next(iter(failures.values()), None)
    return replace(
        batch,
        failures_by_task=failures,
        fallback_reason=first_failure.reason if first_failure else None,
        fallback_detail=(
            "; ".join(failure.detail for failure in failures.values()) if failures else None
        ),
    )


def _is_deletion_only_no_fix_batch(
    batch: Any,
    targets: Sequence[QAScanTarget],
) -> bool:
    """Return whether a NO_FIX removal attempt has only covered npm tombstones."""
    if (
        not targets
        or not batch.inventory_removals
        or batch.inventory_updates
        or batch.target_packages
        or batch.non_npm_input_projects
        or batch.failures_by_task
    ):
        return False
    return _removed_target_task_ids(batch, targets) == {target.task_id for target in targets}


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
    scan_state: OrchestratorState | None = None,
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

    deletion_only_skip = False
    precomputed_resolution_batch = None
    if skip_scan and scan_skip_reason == "no_fix_package_removal":
        precomputed_resolution_batch = _resolve_targeted_packages(
            sandbox,
            scan_targets or (),
            scan_state,
        )
        precomputed_resolution_batch = _clear_expected_removed_target_failures(
            precomputed_resolution_batch,
            scan_targets or (),
        )
        if _is_deletion_only_no_fix_batch(
            precomputed_resolution_batch,
            scan_targets or (),
        ):
            deletion_only_skip = True
            deletion_task_ids = [target.task_id for target in scan_targets or ()]
            deletion_batch = replace(
                precomputed_resolution_batch,
                covered_task_ids=tuple(deletion_task_ids),
                failures_by_task={},
                fallback_reason=None,
                fallback_detail=None,
            )
            deletion_result = _SecurityScanResult(
                True,
                "Dependency-Check was not run because the accepted input delta contains deletions only.",
                set(),
                set(),
                set(),
                execution_status=ScannerExecutionStatus.NOT_RUN,
            )
            results.scan_skipped = True
            results.scan_skip_reason = "no_fix_package_removal_deletions_only"
            _store_scan_outcome(results, deletion_result, label="odc:deletion-only")
            results.scan_evidence = _scan_evidence(
                targets=scan_targets or (),
                scan_result=deletion_result,
                effective_scope=ScanScope.TARGETED,
                complete=True,
                covered_task_ids=deletion_task_ids,
                batch=deletion_batch,
                scan_performed=False,
            )
            results.scan_evidence_by_task = _scan_evidence_by_task(
                results.scan_evidence,
                targets=scan_targets or (),
                baseline_identifiers=(
                    baseline_identifiers if baseline_identifiers is not None else target_identifiers
                ),
                batch=deletion_batch,
                artifact_manifest=None,
            )
        else:
            skip_scan = False

    if deletion_only_skip:
        logger.info("qa_critic: skipping ODC for deletion-only package-removal attempt.")
    elif skip_scan:
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
    has_active_task_scan = bool(scan_state is not None and scan_state.get("active_target_task_ids"))
    if deletion_only_skip:
        pass
    elif (
        not skip_scan
        and scan_targets is None
        and not has_active_task_scan
        and scan_skip_reason != "no_fix_package_removal"
    ):
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
        active_scan_targets = scan_targets or ()
        targeted_scan_label = "odc:targeted"
        target_packages: list[TargetPackageResolution] = []
        covered_task_ids: list[str] = []
        targeted_subdir: str | None = None
        incomplete_reason: ScanFallbackReason | None = None
        incomplete_detail: str | None = None
        targeted_result: _SecurityScanResult | None = None
        resolution_batch = None
        artifact_manifest = None
        scanner_coverage_complete = False
        has_targeted_inputs = False
        try:
            if _targeted_extra_args_conflict():
                incomplete_reason = ScanFallbackReason.TARGETED_SCAN_FAILED
                incomplete_detail = "ODC_EXTRA_ARGS conflicts with safe targeted scan scope."
            else:
                resolution_batch = precomputed_resolution_batch or _resolve_targeted_packages(
                    sandbox,
                    active_scan_targets,
                    scan_state,
                )
                target_packages = list(resolution_batch.target_packages)
                results.scan_target_skipped_by_task = {
                    task_id: ScanFallbackReason.NO_MATCHING_TARGET.value
                    for task_id in resolution_batch.skipped_task_ids
                }
                has_targeted_inputs = bool(
                    target_packages
                    or resolution_batch.non_npm_input_projects
                    or any(resolution_batch.closures_by_task.values())
                )
                results.scan_target_failures_by_task = {
                    task_id: (failure.reason, failure.detail)
                    for task_id, failure in resolution_batch.failures_by_task.items()
                }
                if resolution_batch.fallback_detail:
                    logger.info(
                        "qa_critic: some targeted packages could not be resolved: %s",
                        resolution_batch.fallback_detail,
                    )
                if not has_targeted_inputs:
                    if resolution_batch.skipped_task_ids and not resolution_batch.failures_by_task:
                        incomplete_detail = (
                            "Targeted Dependency-Check was skipped because the target package "
                            "is absent from the live lockfiles."
                        )
                        targeted_result = _SecurityScanResult(
                            True,
                            incomplete_detail,
                            set(),
                            set(),
                            set(),
                            execution_status=ScannerExecutionStatus.NOT_RUN,
                        )
                        targeted_scan_label = "odc:skipped"
                        covered_task_ids = list(resolution_batch.covered_task_ids)
                        results.scan_skipped = True
                        results.scan_skip_reason = "target_package_absent"
                        scanner_coverage_complete = True
                    else:
                        incomplete_reason = (
                            resolution_batch.fallback_reason
                            or ScanFallbackReason.NO_MATCHING_TARGET
                        )
                        incomplete_detail = (
                            resolution_batch.fallback_detail
                            or "No task target or changed input resolved to a scannable project."
                        )
            if incomplete_reason is None and has_targeted_inputs:
                targeted_subdir = ".odc-targeted"
                artifact_manifest = _write_targeted_artifacts(
                    sandbox,
                    resolution_batch.closures_by_task,
                    resolution_batch.non_npm_input_projects,
                )
                targeted_subdir = artifact_manifest.scan_subdir
                if target_identifiers:
                    targeted_result = _qa_odc_module._run_targeted_security_scan(
                        sandbox,
                        workspace_volume,
                        target_identifiers,
                        baseline,
                        targeted_subdir,
                    )
                else:
                    targeted_scan_label = "odc:incremental"
                    targeted_result = _qa_odc_module._run_incremental_security_scan(
                        sandbox,
                        workspace_volume,
                        targeted_subdir,
                        baseline,
                    )
                if targeted_result.execution_status != ScannerExecutionStatus.SUCCESS:
                    incomplete_reason = (
                        ScanFallbackReason.TARGETED_REPORT_UNPARSEABLE
                        if targeted_result.execution_status == ScannerExecutionStatus.UNPARSEABLE
                        else ScanFallbackReason.TARGETED_SCAN_FAILED
                    )
                    incomplete_detail = targeted_result.summary
                elif targeted_result.reported_paths is None:
                    incomplete_reason = ScanFallbackReason.TARGETED_REPORT_UNPARSEABLE
                    incomplete_detail = (
                        "ODC report did not contain parseable dependency file paths."
                    )
                else:
                    # ODC merges identical npm PURLs across physical lockfile
                    # entries. Validate each identity per source project here;
                    # the lockfile inventory retains the physical multiplicity.
                    expected_by_project: dict[str, set[tuple[str, str]]] = {
                        project_path: {
                            (occurrence.package_name.casefold(), occurrence.version)
                            for occurrence in occurrences
                        }
                        for project_path, occurrences in artifact_manifest.occurrences_by_project.items()
                    }
                    reported_by_project: dict[str, set[tuple[str, str]]] = {
                        project_path: set()
                        for project_path in artifact_manifest.input_paths_by_project
                    }
                    unexpected_packages: Counter[tuple[str, str]] = Counter()
                    for occurrence in targeted_result.reported_package_occurrences or ():
                        if str(_scan_result_value(occurrence, "ecosystem", "")).casefold() != "npm":
                            continue
                        identity = (
                            str(_scan_result_value(occurrence, "package_name", "")).casefold(),
                            str(_scan_result_value(occurrence, "version", "")),
                        )
                        project_path = _targeted_project_for_report_path(
                            _scan_result_value(occurrence, "file_path", None)
                        )
                        if project_path not in expected_by_project:
                            unexpected_packages[identity] += 1
                            continue
                        reported_by_project[project_path].add(identity)

                    missing_packages: Counter[tuple[str, str]] = Counter()
                    for project_path, expected_identities in expected_by_project.items():
                        reported_identities = reported_by_project.get(project_path, set())
                        for identity in expected_identities - reported_identities:
                            missing_packages[identity] += 1
                        for identity in reported_identities - expected_identities:
                            unexpected_packages[identity] += 1
                    missing_inputs = _missing_targeted_input_paths(
                        artifact_manifest,
                        targeted_result.reported_paths,
                    )
                    if missing_packages or unexpected_packages:
                        package_diagnostics: list[str] = []
                        if missing_packages:
                            package_diagnostics.append(
                                "missing "
                                + ", ".join(
                                    f"{name}@{version} ({count})"
                                    for (name, version), count in sorted(missing_packages.items())
                                )
                            )
                        if unexpected_packages:
                            package_diagnostics.append(
                                "unexpected "
                                + ", ".join(
                                    f"{name}@{version} ({count})"
                                    for (name, version), count in sorted(
                                        unexpected_packages.items()
                                    )
                                )
                            )
                        incomplete_reason = ScanFallbackReason.TARGET_PACKAGE_NOT_REPORTED
                        incomplete_detail = (
                            "ODC report package occurrences differed from the targeted "
                            f"closure: {'; '.join(package_diagnostics)}."
                        )
                    elif missing_inputs:
                        incomplete_reason = ScanFallbackReason.TARGETED_SCAN_FAILED
                        incomplete_detail = (
                            "ODC report did not cover every staged changed input: "
                            f"{', '.join(missing_inputs)}."
                        )
                    else:
                        artifact_manifest, occurrences_mapped = _attribute_reported_occurrences(
                            artifact_manifest,
                            targeted_result,
                        )
                        covered_task_ids = list(resolution_batch.covered_task_ids)
                        if not occurrences_mapped:
                            incomplete_reason = ScanFallbackReason.TARGETED_SCAN_FAILED
                            incomplete_detail = (
                                "ODC report package occurrences could not be attributed "
                                "to their original scan inputs."
                            )
                        else:
                            scanner_coverage_complete = True
                            if resolution_batch.failures_by_task:
                                incomplete_reason = resolution_batch.fallback_reason
                                incomplete_detail = resolution_batch.fallback_detail
            if scanner_coverage_complete and targeted_result is not None:
                _store_scan_outcome(results, targeted_result, label=targeted_scan_label)
                results.scan_evidence = _scan_evidence(
                    targets=active_scan_targets,
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
                    batch=resolution_batch,
                    scan_performed=(
                        targeted_result.execution_status != ScannerExecutionStatus.NOT_RUN
                    ),
                    scan_skip_reason=(
                        "no_matching_target"
                        if targeted_result.execution_status == ScannerExecutionStatus.NOT_RUN
                        and resolution_batch.skipped_task_ids
                        else None
                    ),
                    artifact_manifest=artifact_manifest,
                )
                results.scan_evidence_by_task = _scan_evidence_by_task(
                    results.scan_evidence,
                    targets=active_scan_targets,
                    baseline_identifiers=baseline,
                    batch=resolution_batch,
                    artifact_manifest=artifact_manifest,
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
                    remaining_identifiers=targeted_result.remaining_identifiers,
                    found_identifiers=targeted_result.found_identifiers,
                    new_identifiers=targeted_result.new_identifiers,
                    found_issues=targeted_result.found_issues,
                    execution_status=execution_status,
                )
            _store_scan_outcome(results, incomplete_result, label=targeted_scan_label)
            results.scan_evidence = _scan_evidence(
                targets=active_scan_targets,
                scan_result=incomplete_result,
                effective_scope=ScanScope.TARGETED,
                complete=False,
                fallback_reason=incomplete_reason,
                target_packages=target_packages,
                covered_task_ids=[],
                batch=resolution_batch,
                artifact_manifest=artifact_manifest,
                scan_performed=(
                    targeted_result is not None
                    and targeted_result.execution_status != ScannerExecutionStatus.NOT_RUN
                ),
            )
            results.scan_evidence_by_task = _scan_evidence_by_task(
                results.scan_evidence,
                targets=active_scan_targets,
                baseline_identifiers=baseline,
                batch=resolution_batch,
                artifact_manifest=artifact_manifest,
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
    singleton_scope = len(task_contexts) == 1

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
                scan_state=state,
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
                    evidence_by_task=results.scan_evidence_by_task,
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
                    evidence_by_task=results.scan_evidence_by_task,
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
                singleton_scope=singleton_scope,
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
        singleton_scope=singleton_scope,
        deterministic_test_evidence=deterministic_test_evidence,
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
        evidence_by_task=results.scan_evidence_by_task,
    )
    for task_id, (reason, detail) in results.scan_target_failures_by_task.items():
        current = qa_evaluations.get(task_id)
        if current is None:
            continue
        # A no-match skip stays inconclusive, but is distinct from scanner failure
        # so the delta-isolation gate can evaluate the remaining package batch.
        skip_target = reason == ScanFallbackReason.NO_MATCHING_TARGET
        failure_detail = (
            f"Task {task_id} targeted package scan skipped ({reason.value}): {detail}"
            if skip_target
            else f"Task {task_id} targeted package could not be scanned ({reason.value}): {detail}"
        )
        if not skip_target:
            errors.append(f"qa_critic: {failure_detail}")
        inconclusive_gates = current.deterministic_gates
        if inconclusive_gates is not None:
            inconclusive_gates = inconclusive_gates.model_copy(
                update={
                    "status": "fail",
                    "scanner_execution_status": (
                        ScannerExecutionStatus.SKIPPED
                        if skip_target
                        else ScannerExecutionStatus.NOT_RUN
                    ),
                    "target_remaining_identifiers": [],
                    "target_scanner_cleared": None,
                    "diagnostics": [*inconclusive_gates.diagnostics, failure_detail],
                }
            )
        retry_feedback = (
            f"{failure_detail} This skipped target is not evidence that the "
            "vulnerability was cleared. QA remains inconclusive and does not consume "
            "a remediation retry."
            if skip_target
            else f"{failure_detail} QA is inconclusive and this result does not consume a "
            "remediation retry. Make the target package resolvable for targeted ODC, "
            "or provide valid evidence that the task no longer requires scanning."
        )
        qa_evaluations[task_id] = QAEvaluation(
            task_id=task_id,
            passed=False,
            failure_category=FailureCategory.SECURITY_FLAG,
            retry_feedback=retry_feedback,
            deterministic_gates=inconclusive_gates,
            evidence_inconclusive=True,
            scan_evidence=current.scan_evidence,
        )
    qa_errors_by_task = {
        task_id: list(investigation.errors)
        for task_id, investigation in investigations_by_task.items()
    }
    for task_id, (reason, detail) in results.scan_target_failures_by_task.items():
        recorded_detail = (
            f"targeted package scan skipped ({reason.value}): {detail}"
            if reason == ScanFallbackReason.NO_MATCHING_TARGET
            else detail
        )
        qa_errors_by_task.setdefault(task_id, []).append(recorded_detail)
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


def run_workspace_setup_scan_node(state: OrchestratorState) -> dict[str, Any]:
    """Scan package-lock dependency closures changed by workspace installation.

    The source inventory was captured before ``npm install`` by the workspace
    builder. This QA-owned bridge compares that baseline with the live Docker
    volume, scans only complete closures for changed package occurrences, and
    returns attempt-independent evidence for the Supervisor's terminal ledger.
    """
    workspace_volume = state.get("workspace_volume")
    if not workspace_volume:
        evidence = ODCScanEvidence(
            requested_scope=ScanScope.TARGETED,
            effective_scope=ScanScope.TARGETED,
            complete=False,
            scan_performed=False,
            fallback_reason=ScanFallbackReason.TARGETED_SCAN_FAILED,
        )
        return {
            "setup_scan_evidence": evidence,
            "errors": ["workspace setup scan skipped: workspace volume is missing."],
        }

    if not state.get("package_inventory_baseline_complete", False):
        logger.warning(
            "workspace setup scan skipped because the source package inventory is incomplete."
        )
        return {}
    setup_task_id = "workspace_setup"
    source_inventory = state.get("accepted_package_inventory_by_lockfile", {}) or {}
    source_paths = {
        parts[0]
        for occurrence_key in source_inventory
        if isinstance(occurrence_key, str)
        and (parts := _parse_package_occurrence_key(occurrence_key)) is not None
    }
    try:
        from .workspace_builder import _discover_package_directories

        repo_root = Path(str(state.get("repo_root") or "."))
        source_paths.update(
            (package_directory / "package-lock.json").as_posix()
            for package_directory in _discover_package_directories(repo_root)
        )
    except (OSError, ValueError) as exc:
        evidence = ODCScanEvidence(
            requested_scope=ScanScope.TARGETED,
            effective_scope=ScanScope.TARGETED,
            complete=False,
            scan_performed=False,
            fallback_reason=ScanFallbackReason.TARGETED_SCAN_FAILED,
        )
        return {
            "setup_scan_evidence": evidence,
            "errors": [f"workspace setup scan could not enumerate package inputs: {exc}"],
        }

    targets = [
        QAScanTarget(
            task_id=setup_task_id,
            group_id=setup_task_id,
            target_package=setup_task_id,
            expected_version=None,
            manifest_paths=(source_path,),
            dependency_ancestry=(),
            target_identifiers=frozenset(),
        )
        for source_path in sorted(source_paths)
    ]
    scan_state = dict(state)
    scan_state.update(
        {
            "active_target_task_ids": [setup_task_id],
            "task_queue": {},
            "accepted_package_inventory_by_lockfile": source_inventory,
            "worker_results_by_attempt": {},
            "attempt_snapshots_by_id": {},
        }
    )
    manifest = None
    scan_result = None
    failure_reason: ScanFallbackReason | None = None
    failure_details: list[str] = []
    batch: _TargetedPackageBatch | None = None
    try:
        with DockerSandbox(repo_root=None, workspace_volume=workspace_volume) as sandbox:

            def load_lockfile(
                source_path: str,
            ) -> tuple[Mapping[str, Any] | None, int | None, ScanFallbackReason | None, str | None]:
                """Read one installed lockfile using the shared delta resolver contract."""
                try:
                    raw_document = sandbox.read_file(source_path)
                except (OSError, RuntimeError) as exc:
                    return (
                        None,
                        None,
                        ScanFallbackReason.MISSING_LOCKFILE,
                        f"Installed package-lock input {source_path} could not be read: {exc}",
                    )
                if raw_document is None:
                    return (
                        None,
                        None,
                        ScanFallbackReason.MISSING_LOCKFILE,
                        f"Installed package-lock input {source_path} could not be read.",
                    )
                try:
                    document = json.loads(raw_document)
                except (TypeError, ValueError) as exc:
                    return (
                        None,
                        None,
                        ScanFallbackReason.INVALID_LOCKFILE,
                        f"Installed package-lock input {source_path} is invalid JSON: {exc}",
                    )
                packages = document.get("packages") if isinstance(document, Mapping) else None
                lockfile_version = (
                    document.get("lockfileVersion") if isinstance(document, Mapping) else None
                )
                if not isinstance(packages, Mapping) or not isinstance(lockfile_version, int):
                    return (
                        None,
                        None,
                        ScanFallbackReason.INVALID_LOCKFILE,
                        f"Installed package-lock input {source_path} lacks a supported packages map.",
                    )
                return packages, lockfile_version, None, None

            if _targeted_extra_args_conflict():
                failure_reason = ScanFallbackReason.TARGETED_SCAN_FAILED
                failure_details.append("ODC_EXTRA_ARGS conflicts with safe incremental scan scope.")
            else:
                change_batch = _changed_package_lock_inputs(
                    scan_state,
                    targets,
                    load_lockfile,
                )
                if change_batch.failures_by_task:
                    first_failure = next(iter(change_batch.failures_by_task.values()))
                    failure_reason = first_failure.reason
                    failure_details.extend(
                        failure.detail for failure in change_batch.failures_by_task.values()
                    )
                batch = _TargetedPackageBatch(
                    target_packages=change_batch.target_packages,
                    closures_by_task=change_batch.closures_by_task,
                    covered_task_ids=(setup_task_id,),
                    failures_by_task=change_batch.failures_by_task,
                    inventory_updates=change_batch.inventory_updates,
                    inventory_removals=change_batch.inventory_removals,
                    covered_input_paths=change_batch.covered_input_paths,
                    non_npm_input_projects=(),
                    fallback_reason=failure_reason,
                    fallback_detail="; ".join(failure_details) or None,
                )
                if change_batch.closures_by_task.get(setup_task_id):
                    manifest = _write_targeted_artifacts(
                        sandbox,
                        change_batch.closures_by_task,
                    )
                    scan_result = _qa_odc_module._run_incremental_security_scan(
                        sandbox,
                        workspace_volume,
                        manifest.scan_subdir,
                        {
                            identifier.upper().strip()
                            for identifier in (state.get("baseline_scan_identifiers") or [])
                            if isinstance(identifier, str) and identifier.strip()
                        },
                    )
                    if scan_result.execution_status != ScannerExecutionStatus.SUCCESS:
                        failure_reason = (
                            ScanFallbackReason.TARGETED_REPORT_UNPARSEABLE
                            if scan_result.execution_status == ScannerExecutionStatus.UNPARSEABLE
                            else ScanFallbackReason.TARGETED_SCAN_FAILED
                        )
                        failure_details.append(scan_result.summary)
                    elif scan_result.reported_paths is None:
                        failure_reason = ScanFallbackReason.TARGETED_REPORT_UNPARSEABLE
                        failure_details.append(
                            "ODC report did not contain parseable dependency file paths."
                        )
                    else:
                        expected_packages = [
                            (name.casefold(), version)
                            for name, version in manifest.expected_package_occurrences
                        ]
                        reported_package_counts = Counter(scan_result.reported_packages or [])
                        if reported_package_counts != Counter(expected_packages):
                            failure_reason = ScanFallbackReason.TARGET_PACKAGE_NOT_REPORTED
                            failure_details.append(
                                "ODC report package occurrences differed from changed closures."
                            )
                        missing_inputs = _missing_targeted_input_paths(
                            manifest,
                            scan_result.reported_paths,
                        )
                        if missing_inputs:
                            failure_reason = ScanFallbackReason.TARGETED_SCAN_FAILED
                            failure_details.append(
                                "ODC report did not cover installed lockfiles: "
                                + ", ".join(missing_inputs)
                                + "."
                            )
                        manifest, occurrences_mapped = _attribute_reported_occurrences(
                            manifest,
                            scan_result,
                        )
                        if not occurrences_mapped:
                            failure_reason = ScanFallbackReason.TARGETED_SCAN_FAILED
                            failure_details.append(
                                "ODC report package occurrences could not be mapped to "
                                "their source lockfile inputs."
                            )
    except Exception as exc:  # noqa: BLE001 - incomplete setup evidence must reach Supervisor
        failure_reason = failure_reason or ScanFallbackReason.TARGETED_SCAN_FAILED
        failure_details.append(f"workspace setup scan failed: {exc}")
    finally:
        if manifest is not None:
            try:
                with DockerSandbox(repo_root=None, workspace_volume=workspace_volume) as sandbox:
                    _cleanup_targeted_artifacts(sandbox, manifest.scan_subdir)
            except Exception as exc:  # noqa: BLE001
                failure_details.append(f"workspace setup scan cleanup failed: {exc}")
                failure_reason = failure_reason or ScanFallbackReason.TARGETED_SCAN_FAILED

    complete = failure_reason is None
    evidence = _scan_evidence(
        targets=targets,
        scan_result=scan_result,
        effective_scope=ScanScope.TARGETED,
        complete=complete,
        fallback_reason=failure_reason,
        target_packages=batch.target_packages if batch is not None else (),
        covered_task_ids=[],
        batch=batch,
        artifact_manifest=manifest,
        scan_performed=scan_result is not None,
    )
    return {
        "setup_scan_evidence": evidence,
        "errors": (
            [f"workspace setup scan incomplete: {'; '.join(failure_details)}"]
            if failure_details
            else []
        ),
    }
