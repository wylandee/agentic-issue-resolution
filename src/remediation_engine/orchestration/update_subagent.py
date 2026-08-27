"""Dependency Update Subagent for Phase 5 dependency resolution.

The Supervisor normally dispatches one task per invocation, while this module
retains batch-capable helpers for direct callers and future explicit batch mode.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable

from remediation_engine.contracts.schemas import (
    AgentActionStatus,
    AgentActionSummary,
    RemediationTask,
    RoutingStrategy,
    SCARemediationStage,
    TaskStatus,
    UpdateRetryDiagnostics,
    VulnerabilityGroup,
    WorkerAttemptResult,
    WorkerExecutionDiagnostics,
)
from remediation_engine.orchestration.remedy_tools import (
    build_update_toolbelt,
    rollback_pending_package_updates,
)
from remediation_engine.orchestration.runtime_context import get_runtime_settings
from remediation_engine.orchestration.state import SubagentState
from remediation_engine.orchestration.subagent_runtime import run_bounded_subagent_loop
from remediation_engine.orchestration.task_utils import (
    create_skinny_subagent_group,
    filter_constraints_ledger,
    is_transitive_group,
)
from remediation_engine.runtime.path_policy import (
    WorkspacePathError,
    resolve_repository_path,
)
from remediation_engine.runtime.sandbox_mgr import DockerSandbox
from remediation_engine.tools.repository_map import build_repository_map

logger = logging.getLogger(__name__)

_UPDATE_MANIFEST_TOOL_NAME = "modify_and_validate_npm_dependency"

try:
    from langchain_openai import ChatOpenAI  # type: ignore[import]
except ImportError:  # pragma: no cover
    ChatOpenAI = None  # type: ignore[assignment,misc]


def _candidate_manifest_paths(group: VulnerabilityGroup) -> list[str]:
    """Return all candidate manifest paths for one grouped dependency target."""
    candidates: list[str] = []
    seen: set[str] = set()

    def add_candidate(value: str | None) -> None:
        """Add a normalized manifest path once."""
        if not value:
            return
        candidate = value.replace("\\", "/")
        if candidate in seen:
            return
        candidates.append(candidate)
        seen.add(candidate)

    for localized_issue in group.localized_issues:
        add_candidate(localized_issue.manifest_file)

    for file_path in group.file_paths:
        add_candidate(file_path)

    add_candidate(group.file_path)

    for issue in group.issues:
        if issue.file_path and Path(issue.file_path).name == "package.json":
            add_candidate(issue.file_path)

    return candidates


def _create_skinny_subagent_group(group: VulnerabilityGroup) -> VulnerabilityGroup:
    """Create a skinny copy of a group for execution agents."""
    return create_skinny_subagent_group(group)


def _filter_constraints_ledger(
    constraints_ledger: Sequence[str], target_groups: Sequence[VulnerabilityGroup]
) -> list[str]:
    """Filter ledger to only include constraints matching the target components."""
    return filter_constraints_ledger(constraints_ledger, target_groups)


def _resolve_manifest_targets(
    group: VulnerabilityGroup,
    repo_root: Path,
) -> tuple[list[str], list[str]]:
    """Resolve all valid package.json targets for one vulnerability group."""
    candidates = _candidate_manifest_paths(group)
    if not candidates:
        return [], [f"Group '{group.group_id}': no manifest target could be resolved."]

    resolved_paths: list[str] = []
    errors: list[str] = []

    for candidate in candidates:
        try:
            abs_target = resolve_repository_path(repo_root, candidate)
        except WorkspacePathError as exc:
            errors.append(f"Group '{group.group_id}': rejected manifest path '{candidate}': {exc}")
            continue
        if not abs_target.exists():
            errors.append(
                f"Group '{group.group_id}': manifest path '{candidate}' does not exist in repo."
            )
            continue
        if abs_target.is_dir() or abs_target.name != "package.json":
            errors.append(
                f"Group '{group.group_id}': manifest target '{candidate}' must be a package.json file."
            )
            continue

        resolved_paths.append(candidate.replace("\\", "/"))

    return resolved_paths, errors


def _build_package_manifest_map(
    resolved_tasks: Sequence[tuple[RemediationTask, VulnerabilityGroup, Sequence[str]]],
) -> dict[str, list[str]]:
    """Build a per-package allowlist of manifest paths for tool enforcement."""
    package_manifest_map: dict[str, list[str]] = {}
    for task, group, manifest_paths in resolved_tasks:
        package_name = _target_package_name(task, group)
        if not package_name:
            continue
        existing = package_manifest_map.setdefault(package_name, [])
        for manifest_path in manifest_paths:
            if manifest_path not in existing:
                existing.append(manifest_path)
    return package_manifest_map


def _requires_override_remediation(
    task: RemediationTask,
    diagnostics: UpdateRetryDiagnostics | None = None,
    feedback: str = "",
    previous_outcome: str = "",
) -> bool:
    """Return whether the committed stage requires a native package override."""
    del feedback, previous_outcome
    if task.strategy_stage != SCARemediationStage.PACKAGE_OVERRIDE:
        # Preserve explicit legacy/direct override evidence, but never let it
        # override a transitive task that is still editing its parent.
        return bool(diagnostics and diagnostics.used_overrides and not task.parent_package_name)
    return bool(
        diagnostics is None
        or diagnostics.used_overrides
        or task.target_dependency_type in {"overrides", "resolutions", "pnpm_overrides"}
        or diagnostics.target_dependency_type in {"overrides", "resolutions", "pnpm_overrides"}
    )


def _target_package_name(task: RemediationTask, group: VulnerabilityGroup) -> str:
    """Return the Supervisor-owned package target for this task stage."""
    if isinstance(task.target_package_name, str) and task.target_package_name.strip():
        return task.target_package_name.strip()
    if task.strategy_stage != SCARemediationStage.PACKAGE_OVERRIDE:
        if group.parent_package_name:
            return group.parent_package_name.strip()
        for localized in group.localized_issues:
            if localized.parent_package_name:
                return localized.parent_package_name.strip()
    return (group.vulnerable_component or "").strip()


def _target_dependency_type(task: RemediationTask, group: VulnerabilityGroup) -> str | None:
    """Return the Supervisor-owned manifest declaration type for this task."""
    if isinstance(task.target_dependency_type, str) and task.target_dependency_type:
        return task.target_dependency_type
    if task.strategy_stage != SCARemediationStage.PACKAGE_OVERRIDE:
        if group.parent_declaration_type:
            return group.parent_declaration_type
        for localized in group.localized_issues:
            if localized.parent_declaration_type:
                return localized.parent_declaration_type
        for localized in group.localized_issues:
            if localized.declaration_type:
                return localized.declaration_type
    if task.strategy_stage == SCARemediationStage.PACKAGE_OVERRIDE:
        return "overrides"
    # Direct dependency localization normally supplies declaration_type. Keep
    # legacy group-based callers executable when that optional enrichment is
    # absent, while leaving transitive targets fail-closed until their parent
    # declaration policy is committed.
    if task.strategy == RoutingStrategy.VERSION_BUMP and not is_transitive_group(group):
        return "dependencies"
    return None


def _is_retry_task(task: RemediationTask) -> bool:
    return task.retry_count > 0 or task.status == TaskStatus.NEEDS_RETRY


def _is_retry_batch(
    resolved_tasks: Sequence[tuple[RemediationTask, VulnerabilityGroup, Sequence[str]]],
) -> bool:
    return bool(resolved_tasks) and all(_is_retry_task(task) for task, _, _ in resolved_tasks)


def _is_mixed_retry_batch(
    resolved_tasks: Sequence[tuple[RemediationTask, VulnerabilityGroup, Sequence[str]]],
) -> bool:
    saw_retry = False
    saw_first_pass = False
    for task, _, _ in resolved_tasks:
        if _is_retry_task(task):
            saw_retry = True
        else:
            saw_first_pass = True
    return saw_retry and saw_first_pass


def _has_successful_manifest_transaction_for_package(
    task: RemediationTask,
    group: VulnerabilityGroup,
    tool_events: Sequence[Any] | None,
) -> bool:
    """Return whether the package has a successful edit-and-sync transaction."""
    if not tool_events:
        return False
    package_name = _target_package_name(task, group)
    return any(
        getattr(event, "name", "") == _UPDATE_MANIFEST_TOOL_NAME
        and str((getattr(event, "args", {}) or {}).get("package_name", "")).strip() == package_name
        and str(getattr(event, "content", "")).startswith("SUCCESS:")
        for event in tool_events
    )


def _is_executed_manifest_transaction(event: Any) -> bool:
    """Return whether an update event represents an attempted tool execution."""
    return getattr(event, "name", "") == _UPDATE_MANIFEST_TOOL_NAME and not str(
        getattr(event, "content", "")
    ).lstrip().startswith("DEFERRED:")


def _build_update_prompt(
    resolved_tasks: Sequence[tuple[RemediationTask, VulnerabilityGroup, Sequence[str]]],
    constraints_ledger: Sequence[str],
    feedback_by_task: dict[str, str],
    previous_action_summaries_by_task: dict[str, str],
    retry_diagnostics_by_task: dict[str, UpdateRetryDiagnostics] | None = None,
    repository_map: str = "(repository map unavailable)",
    allowed_target_versions_by_task: Mapping[str, Sequence[str]] | None = None,
    allowed_dependency_types_by_task: Mapping[str, Sequence[str]] | None = None,
) -> str:
    """Build an execution-only prompt; strategy selection belongs to Supervisor."""
    allowed_target_versions_by_task = allowed_target_versions_by_task or {}
    allowed_dependency_types_by_task = allowed_dependency_types_by_task or {}
    retry_diagnostics_by_task = retry_diagnostics_by_task or {}
    sections = [
        "You are a dependency-manifest execution worker.",
        "The Supervisor's task instruction is authoritative. Execute it exactly.",
        "Do not search the NPM registry or perform retry planning.",
        "Use only modify_and_validate_npm_dependency for manifest changes.",
        "The repository map below is deterministic read-only context.",
        "Each modify_and_validate_npm_dependency call edits the manifest and immediately synchronizes its package manifests before returning.",
        "Transactions are serialized per-package; call the tool with that package_name fixed while changing only an approved retry candidate.",
        "If a transaction returns ERROR_CODE or FAILURE, call the same tool again for that package with a different Supervisor-approved target_version or dependency_type.",
        "Keep package_name and manifest_path within the committed task allowlists, and never invent a candidate or query the registry.",
        "A package may receive at most three combined transaction attempts. After its limit is exhausted, continue with the next independent package.",
        "A failed transaction is rolled back automatically; continue using the same transaction protocol.",
        "Never edit source-code files in this worker.",
        "The Supervisor-owned dependency-type candidates are the only permitted strategy alternatives.",
        "",
        "Deterministic repository map:",
        repository_map,
        "",
        "Constraints ledger:",
    ]
    sections.extend(f"- {item}" for item in constraints_ledger)
    if not constraints_ledger:
        sections.append("- none")
    for task, group, manifest_paths in resolved_tasks:
        diagnostics = retry_diagnostics_by_task.get(task.task_id)
        allowed_versions = list(allowed_target_versions_by_task.get(task.task_id, ()))
        if not allowed_versions:
            allowed_versions = list(
                dict.fromkeys(
                    value
                    for value in [
                        task.selected_version,
                        *(diagnostics.candidate_versions_considered if diagnostics else []),
                    ]
                    if value
                )
            )
        allowed_types = list(allowed_dependency_types_by_task.get(task.task_id, ()))
        if not allowed_types:
            allowed_types = [
                value
                for value in [
                    _target_dependency_type(task, group),
                    *(diagnostics.candidate_dependency_types if diagnostics else []),
                ]
                if value
            ]
        sections.extend(
            [
                "",
                f"## Task {task.task_id}",
                f"- Component: {group.vulnerable_component or 'unknown'}",
                f"- Edit target: {_target_package_name(task, group)}",
                f"- Declaration type: {_target_dependency_type(task, group) or 'package dependency'}",
                f"- Strategy stage: {task.strategy_stage.value}",
                f"- Parent package: {task.parent_package_name or group.parent_package_name or 'none'}",
                f"- Manifest paths: {', '.join(manifest_paths) or 'none'}",
                f"- Allowed target versions: {', '.join(allowed_versions) or 'none supplied'}",
                f"- Allowed dependency types: {', '.join(dict.fromkeys(allowed_types)) or 'none supplied'}",
                f"- Exact supervisor instruction: {task.instruction or '(missing)'}",
                f"- QA feedback: {feedback_by_task.get(task.task_id, 'none')}",
                f"- Previous outcome: {previous_action_summaries_by_task.get(task.task_id, 'none')}",
            ]
        )
    sections.extend(
        [
            "",
            "Completion rule:",
            "Return control only after every package has one successful combined transaction or has exhausted its three attempts and been surrendered.",
        ]
    )
    return "\n".join(sections)


def _build_action_summaries(
    resolved_tasks: Sequence[tuple[RemediationTask, VulnerabilityGroup, Sequence[str]]],
    changed_files: Sequence[str],
    final_text: str,
    succeeded: bool,
    retry_batch: bool = False,
    tool_events: Sequence[Any] | None = None,
) -> list[AgentActionSummary]:
    """Summarize worker execution without requiring registry evidence."""
    normalized_changed_files = {path.replace("\\", "/") for path in changed_files}
    final_note = (final_text or "").strip()
    summaries: list[AgentActionSummary] = []
    for task, group, manifest_paths in resolved_tasks:
        package_modified = _has_successful_manifest_transaction_for_package(
            task, group, tool_events
        )
        package_validated = package_modified
        task_succeeded = package_modified and package_validated
        if tool_events is None:
            task_succeeded = succeeded and bool(normalized_changed_files)
        changed = [
            path for path in manifest_paths if path.replace("\\", "/") in normalized_changed_files
        ]
        status = AgentActionStatus.SUCCESS if task_succeeded else AgentActionStatus.SURRENDER
        outcome = (
            "Completed validated manifest updates"
            if task_succeeded
            else "Stopped without a validated manifest update"
        )
        changed_label = changed or (manifest_paths if package_modified else ["no files"])
        summary = f"{outcome} for {group.vulnerable_component or 'unknown component'} in {', '.join(manifest_paths) or 'no manifest'}; changed files: {', '.join(changed_label)}."
        if final_note and len(resolved_tasks) == 1:
            summary += f" Final note: {final_note}"
        summaries.append(AgentActionSummary(task_id=task.task_id, status=status, summary=summary))
    return summaries


def _build_surrender_summaries(
    task_ids: Sequence[str],
    message: str,
) -> list[AgentActionSummary]:
    """Build surrender summaries when execution cannot start or complete."""
    return [
        AgentActionSummary(
            task_id=task_id,
            status=AgentActionStatus.SURRENDER,
            summary=message,
        )
        for task_id in task_ids
    ]


def _worker_result_map(
    target_tasks: Sequence[RemediationTask],
    snapshots: dict[str, Any],
    summaries: Sequence[AgentActionSummary],
    *,
    succeeded: bool,
    errors: Sequence[str] = (),
    attempted_versions_by_task: dict[str, list[str]] | None = None,
    executed_versions_by_task: dict[str, list[str]] | None = None,
    effective_target_version_by_task: dict[str, str | None] | None = None,
    effective_dependency_type_by_task: dict[str, str | None] | None = None,
    validation_calls: int = 0,
    manifest_transaction_attempts: int = 0,
    manifest_transaction_attempts_by_task: Mapping[str, int] | None = None,
) -> dict[str, WorkerAttemptResult]:
    """Build attempt-correlated worker envelopes without changing task state."""
    summary_by_task = {summary.task_id: summary for summary in summaries}
    results: dict[str, WorkerAttemptResult] = {}
    attempted_versions_by_task = attempted_versions_by_task or {}
    executed_versions_by_task = executed_versions_by_task or attempted_versions_by_task
    effective_target_version_by_task = effective_target_version_by_task or {}
    effective_dependency_type_by_task = effective_dependency_type_by_task or {}
    manifest_transaction_attempts_by_task = manifest_transaction_attempts_by_task or {}
    for task in target_tasks:
        snapshot = snapshots.get(task.task_id)
        if snapshot is None:
            continue
        summary = summary_by_task.get(task.task_id)
        attempted = list(attempted_versions_by_task.get(task.task_id, []))
        executed = list(executed_versions_by_task.get(task.task_id, []))
        task_succeeded = summary is not None and summary.status == AgentActionStatus.SUCCESS
        results[snapshot.attempt_id] = WorkerAttemptResult(
            attempt_id=snapshot.attempt_id,
            task_id=task.task_id,
            task_revision=snapshot.task_revision,
            status=(
                summary.status
                if summary is not None
                else AgentActionStatus.SUCCESS
                if succeeded
                else AgentActionStatus.SURRENDER
            ),
            executed_versions=executed,
            action_summary=summary,
            execution_diagnostics=WorkerExecutionDiagnostics(
                attempted_versions=attempted,
                executed_versions=executed,
                effective_target_version=effective_target_version_by_task.get(task.task_id),
                effective_dependency_type=effective_dependency_type_by_task.get(task.task_id),
                manifest_transaction_attempts=manifest_transaction_attempts_by_task.get(
                    task.task_id, manifest_transaction_attempts
                ),
                validation_calls=validation_calls,
                validation_passed=task_succeeded,
                failure_reason=" | ".join(errors),
            ),
            instruction_digest=snapshot.instruction_digest,
            errors=list(errors),
        )
    return results


def _attempted_versions_for_current_run(
    resolved_tasks: Sequence[tuple[RemediationTask, VulnerabilityGroup, Sequence[str]]],
    tool_events: Sequence[Any],
) -> dict[str, list[str]]:
    """Collect version targets from combined transactions in this worker run."""
    package_to_task_ids: dict[str, list[str]] = {}
    for task, group, _manifest_paths in resolved_tasks:
        package = _target_package_name(task, group)
        if package:
            package_to_task_ids.setdefault(package, []).append(task.task_id)

    result: dict[str, list[str]] = {task.task_id: [] for task, _, _ in resolved_tasks}
    for event in tool_events:
        if not _is_executed_manifest_transaction(event):
            continue
        args = getattr(event, "args", {}) or {}
        package = str(args.get("package_name", "")).strip()
        target = str(args.get("target_version", "")).strip().lstrip("vV")
        if not package or not target:
            continue
        for task_id in package_to_task_ids.get(package, []):
            if target not in result[task_id]:
                result[task_id].append(target)
    return result


def _executed_versions_for_current_run(
    resolved_tasks: Sequence[tuple[RemediationTask, VulnerabilityGroup, Sequence[str]]],
    tool_events: Sequence[Any],
) -> dict[str, list[str]]:
    """Collect only successful version targets from this worker run."""
    result: dict[str, list[str]] = {task.task_id: [] for task, _, _ in resolved_tasks}
    package_to_task_ids: dict[str, list[str]] = {}
    for task, group, _manifest_paths in resolved_tasks:
        package = _target_package_name(task, group)
        if package:
            package_to_task_ids.setdefault(package, []).append(task.task_id)
    for event in tool_events:
        if not _is_executed_manifest_transaction(event):
            continue
        if not str(getattr(event, "content", "")).startswith("SUCCESS:"):
            continue
        args = getattr(event, "args", {}) or {}
        package = str(args.get("package_name", "")).strip()
        target = str(args.get("target_version", "")).strip().lstrip("vV")
        if not package or not target:
            continue
        for task_id in package_to_task_ids.get(package, []):
            if target not in result[task_id]:
                result[task_id].append(target)
    return result


def _attempted_dependency_types_for_current_run(
    resolved_tasks: Sequence[tuple[RemediationTask, VulnerabilityGroup, Sequence[str]]],
    tool_events: Sequence[Any],
) -> dict[str, list[str]]:
    """Collect dependency declaration types from combined transactions."""
    result: dict[str, list[str]] = {task.task_id: [] for task, _, _ in resolved_tasks}
    package_to_task_ids: dict[str, list[str]] = {}
    for task, group, _manifest_paths in resolved_tasks:
        package = _target_package_name(task, group)
        if package:
            package_to_task_ids.setdefault(package, []).append(task.task_id)
    for event in tool_events:
        if not _is_executed_manifest_transaction(event):
            continue
        args = getattr(event, "args", {}) or {}
        package = str(args.get("package_name", "")).strip()
        dependency_type = str(args.get("dependency_type", "")).strip()
        if not package or not dependency_type:
            continue
        for task_id in package_to_task_ids.get(package, []):
            if dependency_type not in result[task_id]:
                result[task_id].append(dependency_type)
    return result


def _effective_targets_for_current_run(
    resolved_tasks: Sequence[tuple[RemediationTask, VulnerabilityGroup, Sequence[str]]],
    tool_events: Sequence[Any],
) -> tuple[dict[str, str | None], dict[str, str | None]]:
    """Collect effective version and dependency type from successful transactions."""
    package_to_task_ids: dict[str, list[str]] = {}
    for task, group, _manifest_paths in resolved_tasks:
        package = _target_package_name(task, group)
        if package:
            package_to_task_ids.setdefault(package, []).append(task.task_id)

    versions: dict[str, str | None] = {task.task_id: None for task, _, _ in resolved_tasks}
    dependency_types: dict[str, str | None] = {task.task_id: None for task, _, _ in resolved_tasks}
    for event in tool_events:
        if not _is_executed_manifest_transaction(event):
            continue
        if not str(getattr(event, "content", "")).startswith("SUCCESS:"):
            continue
        args = getattr(event, "args", {}) or {}
        package = str(args.get("package_name", "")).strip()
        version = str(args.get("target_version", "")).strip().lstrip("vV") or None
        dependency_type = str(args.get("dependency_type", "")).strip() or None
        for task_id in package_to_task_ids.get(package, []):
            versions[task_id] = version
            dependency_types[task_id] = dependency_type
    return versions, dependency_types


def _build_retry_diagnostics(
    resolved_tasks: Sequence[tuple[RemediationTask, VulnerabilityGroup, Sequence[str]]],
    tool_events: Sequence[Any],
    final_text: str,
    errors: Sequence[str],
    succeeded: bool,
    prior_diagnostics_by_task: dict[str, UpdateRetryDiagnostics],
    constraints_ledger: Sequence[str],
    allowed_target_versions_by_task: Mapping[str, Sequence[str]] | None = None,
    allowed_dependency_types_by_task: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, UpdateRetryDiagnostics]:
    """Record worker evidence while keeping strategy planning in Supervisor."""
    del constraints_ledger
    allowed_target_versions_by_task = allowed_target_versions_by_task or {}
    allowed_dependency_types_by_task = allowed_dependency_types_by_task or {}
    result: dict[str, UpdateRetryDiagnostics] = {}
    joined_errors = " | ".join(error.strip() for error in errors if error.strip())
    lowered_outcome = f"{final_text or ''} {joined_errors}".lower()
    for task, group, _ in resolved_tasks:
        target_package = _target_package_name(task, group)
        target_dependency_type = _target_dependency_type(task, group)
        prior = prior_diagnostics_by_task.get(task.task_id)
        prior_attempts_by_target = dict(prior.attempted_versions_by_target) if prior else {}
        attempted = list(
            prior_attempts_by_target.get(target_package, prior.attempted_versions if prior else [])
        )
        executed = list(prior.executed_versions) if prior else []
        attempted_dependency_types = list(prior.attempted_dependency_types) if prior else []
        candidate_dependency_types = list(prior.candidate_dependency_types) if prior else []
        candidate_dependency_types.extend(
            value
            for value in allowed_dependency_types_by_task.get(task.task_id, ())
            if value not in candidate_dependency_types
        )
        if not candidate_dependency_types and target_dependency_type:
            candidate_dependency_types = [target_dependency_type]
        effective_target_version = prior.effective_target_version if prior else None
        effective_dependency_type = prior.effective_dependency_type if prior else None
        used_overrides = bool(prior.used_overrides) if prior else False
        for event in tool_events:
            if not _is_executed_manifest_transaction(event):
                continue
            if str(event.args.get("package_name", "")).strip() != target_package:
                continue
            target = str(event.args.get("target_version", "")).strip().lstrip("vV")
            if target and target not in attempted:
                attempted.append(target)
            dependency_type = str(event.args.get("dependency_type", "")).strip()
            if dependency_type and dependency_type not in attempted_dependency_types:
                attempted_dependency_types.append(dependency_type)
            used_overrides = used_overrides or dependency_type in {
                "overrides",
                "resolutions",
                "pnpm_overrides",
            }
            if str(event.content).startswith("SUCCESS:"):
                if target and target not in executed:
                    executed.append(target)
                effective_target_version = target or effective_target_version
                effective_dependency_type = dependency_type or effective_dependency_type
        package_abandoned = bool(prior.package_abandoned) if prior else False
        if "package abandoned" in lowered_outcome or "package not found" in lowered_outcome:
            package_abandoned = True
        exhausted_update_path = bool(prior.exhausted_update_path) if prior else False
        if any(
            marker in lowered_outcome
            for marker in (
                "update path is exhausted",
                "update path exhausted",
                "no valid candidate",
                "already attempted the latest",
                "latest version was already attempted",
            )
        ):
            exhausted_update_path = True
        if "retry_limit_reached" in lowered_outcome:
            exhausted_update_path = True
        task_stage = getattr(task, "strategy_stage", SCARemediationStage.OSV_MINIMUM)
        if not isinstance(task_stage, SCARemediationStage):
            task_stage = SCARemediationStage.OSV_MINIMUM
        result[task.task_id] = UpdateRetryDiagnostics(
            task_id=task.task_id,
            committed_attempt_id=prior.committed_attempt_id if prior else None,
            strategy_stage=task_stage,
            security_floor=prior.security_floor
            if prior
            else (group.fix_plan.fixed_version if group.fix_plan else None),
            registry_query_performed=prior.registry_query_performed if prior else False,
            attempted_versions=attempted,
            executed_versions=executed,
            candidate_versions_considered=list(
                dict.fromkeys(
                    [
                        *(prior.candidate_versions_considered if prior else []),
                        *allowed_target_versions_by_task.get(task.task_id, ()),
                    ]
                )
            ),
            attempted_dependency_types=attempted_dependency_types,
            candidate_dependency_types=candidate_dependency_types,
            # Version selection belongs to the Supervisor planner. A worker
            # failure must not resurrect the previous planner selection.
            selected_version=prior.selected_version if prior else None,
            effective_target_version=effective_target_version,
            effective_dependency_type=effective_dependency_type,
            latest_version_seen=prior.latest_version_seen if prior else None,
            used_overrides=used_overrides,
            package_abandoned=package_abandoned,
            exhausted_update_path=exhausted_update_path,
            failure_reason=(joined_errors or final_text.strip())
            if not succeeded
            else (prior.failure_reason if prior else ""),
            reasoning_summary=(final_text or "").strip(),
            instruction_digest=prior.instruction_digest if prior else None,
            target_package_name=target_package,
            target_dependency_type=target_dependency_type,
            parent_package_name=(
                task.parent_package_name
                or group.parent_package_name
                or next(
                    (
                        localized.parent_package_name
                        for localized in group.localized_issues
                        if localized.parent_package_name
                    ),
                    None,
                )
            ),
            parent_minimum_version=(
                task.parent_minimum_version
                if task.parent_minimum_version
                else (prior.parent_minimum_version if prior else None)
            ),
            attempted_versions_by_target={
                **prior_attempts_by_target,
                target_package: attempted,
            },
        )
    return result


@traceable(name="Update_Subagent_Test_Run")  # for langsmith testing
def run_update_subagent_node(state: SubagentState) -> dict[str, Any]:
    """Run the dependency update subagent for one or more committed tasks.

    The Supervisor currently supplies one task per invocation. Direct callers
    may still provide multiple compatible tasks, subject to the existing
    first-pass/retry batch safeguards and per-package rollback behavior.
    """
    repo_root_str = state.get("repo_root", "")
    workspace_volume = state.get("workspace_volume", "")
    target_tasks = list(state.get("target_tasks", []))
    target_groups = list(state.get("target_groups", []))
    constraints_ledger = list(state.get("constraints_ledger", []))
    feedback_by_task = dict(state.get("feedback_by_task", {}))
    previous_action_summaries_by_task = dict(state.get("previous_action_summaries_by_task", {}))
    prior_retry_diagnostics_by_task = dict(state.get("retry_diagnostics_by_task", {}))
    all_task_ids = [t.task_id for t in target_tasks]

    repo_root = Path(repo_root_str)
    if not repo_root_str or not repo_root.is_dir():
        msg = f"Update Subagent: repo_root '{repo_root_str}' is not a valid directory."
        summaries = _build_surrender_summaries(
            all_task_ids, "Stopped before execution because repo_root was invalid."
        )
        return {
            "action_summaries": summaries,
            "action_summary": summaries[0] if summaries else None,
            "changed_files": [],
            "errors": [msg],
        }

    if not workspace_volume:
        msg = "Update Subagent: workspace_volume is missing from state."
        summaries = _build_surrender_summaries(
            all_task_ids, "Stopped before execution because workspace_volume was missing."
        )
        return {
            "action_summaries": summaries,
            "action_summary": summaries[0] if summaries else None,
            "changed_files": [],
            "errors": [msg],
        }

    resolved_tasks: list[tuple[RemediationTask, VulnerabilityGroup, list[str]]] = []
    resolution_errors: list[str] = []
    target_attempt_snapshots = dict(state.get("target_attempt_snapshots", {}))
    allowed_target_versions_by_task: dict[str, list[str]] = {}
    allowed_dependency_types_by_task: dict[str, list[str]] = {}
    groups_by_id = {group.group_id: group for group in target_groups}
    for task in target_tasks:
        group = groups_by_id.get(task.parent_group_id)
        if group is None:
            resolution_errors.append(
                f"Update Subagent: no vulnerability group found for task {task.task_id} "
                f"(parent_group_id={task.parent_group_id})."
            )
            continue
        snapshot = target_attempt_snapshots.get(task.task_id)
        if snapshot is not None:
            if (
                task.current_attempt_id != snapshot.attempt_id
                or task.task_revision != snapshot.task_revision
                or task.instruction != snapshot.instruction
                or task.target_package_name != snapshot.target_package_name
                or task.target_dependency_type != snapshot.target_dependency_type
            ):
                resolution_errors.append(
                    f"Update Subagent: committed attempt snapshot does not match task {task.task_id}."
                )
                continue
            task = task.model_copy(
                update={
                    "strategy_stage": snapshot.strategy_stage,
                    "selected_version": snapshot.selected_version,
                    "target_package_name": snapshot.target_package_name,
                    "target_dependency_type": snapshot.target_dependency_type,
                    "parent_minimum_version": snapshot.parent_minimum_version,
                    "instruction": snapshot.instruction,
                }
            )
            snapshot_versions = list(snapshot.allowed_target_versions)
            if not snapshot_versions and snapshot.selected_version:
                snapshot_versions = [snapshot.selected_version]
            snapshot_dependency_types = list(snapshot.allowed_dependency_types)
            if not snapshot_dependency_types and snapshot.target_dependency_type:
                snapshot_dependency_types = [snapshot.target_dependency_type]
            allowed_target_versions_by_task[task.task_id] = snapshot_versions
            allowed_dependency_types_by_task[task.task_id] = snapshot_dependency_types
        else:
            diagnostics = prior_retry_diagnostics_by_task.get(task.task_id)
            attempted_versions = set(diagnostics.attempted_versions) if diagnostics else set()
            allowed_target_versions_by_task[task.task_id] = list(
                dict.fromkeys(
                    version
                    for version in [
                        task.selected_version,
                        *(diagnostics.candidate_versions_considered if diagnostics else []),
                    ]
                    if version and version not in attempted_versions
                )
            )
            target_type = _target_dependency_type(task, group)
            attempted_types = set(diagnostics.attempted_dependency_types) if diagnostics else set()
            allowed_dependency_types_by_task[task.task_id] = list(
                dict.fromkeys(
                    dependency_type
                    for dependency_type in [
                        target_type,
                        *(diagnostics.candidate_dependency_types if diagnostics else []),
                    ]
                    if dependency_type and dependency_type not in attempted_types
                )
            )
        manifest_paths, errors = _resolve_manifest_targets(group, repo_root)
        resolution_errors.extend(errors)
        if not manifest_paths:
            continue
        resolved_tasks.append((task, group, manifest_paths))

    if not resolved_tasks:
        summaries = _build_surrender_summaries(
            all_task_ids, "Stopped before execution because no manifest targets could be resolved."
        )
        return {
            "action_summaries": summaries,
            "action_summary": summaries[0] if summaries else None,
            "changed_files": [],
            "errors": resolution_errors,
        }

    if _is_mixed_retry_batch(resolved_tasks):
        summaries = _build_surrender_summaries(
            all_task_ids,
            "Stopped before execution because the supervisor mixed first-pass and retry update tasks in one batch.",
        )
        return {
            "action_summaries": summaries,
            "action_summary": summaries[0] if summaries else None,
            "changed_files": [],
            "errors": resolution_errors
            + [
                "Update Subagent: mixed first-pass and retry update tasks are not supported in the same batch."
            ],
        }

    resolved_task_ids = [t.task_id for t, _, _ in resolved_tasks]
    if ChatOpenAI is None:
        msg = "Update Subagent: 'langchain-openai' is not installed."
        summaries = _build_surrender_summaries(
            resolved_task_ids, "Stopped before execution because the LLM client is unavailable."
        )
        return {
            "action_summaries": summaries,
            "action_summary": summaries[0] if summaries else None,
            "changed_files": [],
            "errors": resolution_errors + [msg],
        }

    model_name = get_runtime_settings().update_llm_model
    try:
        llm = ChatOpenAI(model=model_name, temperature=0)
    except Exception as exc:  # noqa: BLE001
        msg = f"Update Subagent: failed to initialize LLM - {exc}."
        summaries = _build_surrender_summaries(
            resolved_task_ids, "Stopped before execution because the LLM failed to initialize."
        )
        return {
            "action_summaries": summaries,
            "action_summary": summaries[0] if summaries else None,
            "changed_files": [],
            "errors": resolution_errors + [msg],
        }

    touched_files: set[str] = set()
    package_checkpoints: dict[str, Any] = {}
    cleanup_errors: list[str] = []
    runtime = None
    execution_state: dict[str, Any] = {
        "edits_started": False,
        "validation_calls": 0,
        "manifest_transaction_attempts": 0,
    }

    filtered_ledger = _filter_constraints_ledger(constraints_ledger, target_groups)
    retry_batch = _is_retry_batch(resolved_tasks)
    skinny_resolved_tasks = [
        (t, _create_skinny_subagent_group(g), paths) for t, g, paths in resolved_tasks
    ]

    prompt = _build_update_prompt(
        skinny_resolved_tasks,
        filtered_ledger,
        feedback_by_task,
        previous_action_summaries_by_task,
        prior_retry_diagnostics_by_task,
        repository_map=build_repository_map(repo_root),
        allowed_target_versions_by_task=allowed_target_versions_by_task,
        allowed_dependency_types_by_task=allowed_dependency_types_by_task,
    )
    initial_messages = [
        SystemMessage(
            content=(
                "You are an execution-only dependency worker. The Supervisor owns candidate generation. "
                "Use only the combined manifest transaction tool, retry failed transactions with a different "
                "Supervisor-approved candidate, and do not query registries."
            )
        ),
        HumanMessage(content=prompt),
    ]

    override_required_packages: set[str] = set()
    allowed_dependency_types_by_package: dict[str, set[str]] = {}
    allowed_target_versions_by_package: dict[str, set[str]] = {}
    for task, group, _ in skinny_resolved_tasks:
        pkg_name = _target_package_name(task, group)
        if pkg_name:
            allowed_versions = allowed_target_versions_by_task.get(task.task_id, [])
            allowed_target_versions_by_package.setdefault(pkg_name, set()).update(allowed_versions)
            allowed_types = allowed_dependency_types_by_task.get(task.task_id, [])
            allowed_dependency_types_by_package.setdefault(pkg_name, set()).update(allowed_types)
        diag = prior_retry_diagnostics_by_task.get(task.task_id)
        if pkg_name and _requires_override_remediation(
            task,
            diag,
            feedback_by_task.get(task.task_id, ""),
            previous_action_summaries_by_task.get(task.task_id, ""),
        ):
            override_required_packages.add(pkg_name)

    try:
        with DockerSandbox(repo_root=None, workspace_volume=workspace_volume) as sandbox:
            package_manifest_map = _build_package_manifest_map(skinny_resolved_tasks)
            toolbelt = build_update_toolbelt(
                sandbox,
                touched_files,
                target_manifest_paths=[
                    manifest_path
                    for _, _, manifest_paths in skinny_resolved_tasks
                    for manifest_path in manifest_paths
                ],
                package_manifest_paths=package_manifest_map,
                allowed_target_versions_by_package=allowed_target_versions_by_package,
                override_required_packages=override_required_packages,
                allowed_dependency_types_by_package=allowed_dependency_types_by_package,
                execution_state=execution_state,
                package_checkpoints=package_checkpoints,
            )
            try:
                runtime = run_bounded_subagent_loop(
                    llm,
                    toolbelt,
                    initial_messages,
                    touched_files,
                    execution_state=execution_state,
                )
            finally:
                rollback_errors = rollback_pending_package_updates(
                    sandbox,
                    package_checkpoints,
                    touched_files,
                )
                cleanup_errors.extend(rollback_errors)
                if rollback_errors and runtime is not None:
                    runtime.errors.extend(rollback_errors)
    except Exception as exc:  # noqa: BLE001
        msg = f"Update Subagent: sandbox or tool loop failed - {exc}"
        summaries = _build_surrender_summaries(
            resolved_task_ids, "Stopped because the sandbox or tool loop failed."
        )
        return {
            "action_summaries": summaries,
            "action_summary": summaries[0] if summaries else None,
            "changed_files": sorted(touched_files),
            "errors": resolution_errors + cleanup_errors + [msg],
        }

    package_names = {
        _target_package_name(task, group)
        for task, group, _ in resolved_tasks
        if _target_package_name(task, group)
    }
    successful_packages = {
        _target_package_name(task, group)
        for task, group, _ in resolved_tasks
        if _has_successful_manifest_transaction_for_package(task, group, runtime.tool_events)
    }
    unvalidated_manifest_paths = {
        path.replace("\\", "/")
        for task, group, manifest_paths in resolved_tasks
        if _target_package_name(task, group) not in successful_packages
        for path in manifest_paths
    }
    committed_changed_files = sorted(touched_files)
    if not committed_changed_files:
        committed_changed_files = sorted(
            path
            for path in runtime.changed_files
            if path.replace("\\", "/") not in unvalidated_manifest_paths
        )
    # A worker succeeds only when every target package has a successful
    # combined edit-and-sync transaction. Failed transactions may precede a
    # later success, so neither raw event counts nor changed-file counts are
    # used as a proxy for completion.
    succeeded = bool(package_names) and all(
        _has_successful_manifest_transaction_for_package(task, group, runtime.tool_events)
        for task, group, _ in resolved_tasks
    )
    attempted_by_task = _attempted_versions_for_current_run(
        resolved_tasks,
        runtime.tool_events,
    )
    attempted_dependency_types_by_task = _attempted_dependency_types_for_current_run(
        resolved_tasks,
        runtime.tool_events,
    )
    executed_by_task = _executed_versions_for_current_run(
        resolved_tasks,
        runtime.tool_events,
    )
    effective_versions_by_task, effective_dependency_types_by_task = (
        _effective_targets_for_current_run(
            resolved_tasks,
            runtime.tool_events,
        )
    )
    runtime_manifest_attempts_by_package = dict(
        execution_state.get("manifest_transaction_attempts_by_package", {})
    )
    runtime_manifest_attempts_by_package.update(
        {
            package: max(
                int(runtime_manifest_attempts_by_package.get(package, 0)),
                int(attempts),
            )
            for package, attempts in execution_state.get(
                "manifest_runtime_attempts_by_package", {}
            ).items()
        }
    )
    manifest_transaction_attempts_by_task = {
        task.task_id: runtime_manifest_attempts_by_package.get(
            _target_package_name(task, group),
            0,
        )
        for task, group, _ in resolved_tasks
    }
    instruction_mismatch_task_ids: set[str] = set()
    instruction_mismatch_errors: list[str] = []
    for task, _group, _manifest_paths in resolved_tasks:
        snapshot = target_attempt_snapshots.get(task.task_id)
        if snapshot is None:
            continue
        allowed_versions = list(getattr(snapshot, "allowed_target_versions", []) or [])
        if not allowed_versions and snapshot.selected_version:
            allowed_versions = [snapshot.selected_version]
        allowed = {version.strip().lstrip("vV").lower() for version in allowed_versions if version}
        observed = {
            version.strip().lstrip("vV").lower()
            for version in attempted_by_task.get(task.task_id, [])
            if version
        }
        unexpected = observed - allowed if allowed else set()
        if unexpected:
            instruction_mismatch_task_ids.add(task.task_id)
            instruction_mismatch_errors.append(
                f"Task {task.task_id}: worker attempted unallowlisted versions "
                f"{', '.join(sorted(unexpected))}; Supervisor-approved versions are "
                f"{', '.join(sorted(allowed))}."
            )
        allowed_dependency_types = list(getattr(snapshot, "allowed_dependency_types", []) or [])
        if not allowed_dependency_types and snapshot.target_dependency_type:
            allowed_dependency_types = [snapshot.target_dependency_type]
        allowed_types = {value.strip().lower() for value in allowed_dependency_types if value}
        observed_types = {
            value.strip().lower()
            for value in attempted_dependency_types_by_task.get(task.task_id, [])
            if value
        }
        unexpected_types = observed_types - allowed_types if allowed_types else set()
        if unexpected_types:
            instruction_mismatch_task_ids.add(task.task_id)
            instruction_mismatch_errors.append(
                f"Task {task.task_id}: worker attempted unallowlisted dependency types "
                f"{', '.join(sorted(unexpected_types))}; Supervisor-approved types are "
                f"{', '.join(sorted(allowed_types))}."
            )

    if instruction_mismatch_task_ids:
        runtime.errors.extend(instruction_mismatch_errors)
        succeeded = False

    retry_diagnostics_by_task = _build_retry_diagnostics(
        resolved_tasks,
        runtime.tool_events,
        runtime.final_text,
        runtime.errors,
        succeeded,
        prior_retry_diagnostics_by_task,
        constraints_ledger=constraints_ledger,
        allowed_target_versions_by_task=allowed_target_versions_by_task,
        allowed_dependency_types_by_task=allowed_dependency_types_by_task,
    )
    summaries = _build_action_summaries(
        resolved_tasks,
        committed_changed_files,
        runtime.final_text,
        succeeded,
        retry_batch=retry_batch,
        tool_events=runtime.tool_events,
    )
    if instruction_mismatch_task_ids:
        summaries = [
            summary.model_copy(
                update={
                    "status": AgentActionStatus.SURRENDER,
                    "summary": (
                        f"{summary.summary} Instruction-mismatch surrender: "
                        f"{next(error for error in instruction_mismatch_errors if summary.task_id in error)}"
                    ),
                }
            )
            if summary.task_id in instruction_mismatch_task_ids
            else summary
            for summary in summaries
        ]
    tagged_summaries = [
        summary.model_copy(
            update={
                "attempt_id": target_attempt_snapshots[summary.task_id].attempt_id,
                "task_revision": target_attempt_snapshots[summary.task_id].task_revision,
                "instruction_digest": target_attempt_snapshots[summary.task_id].instruction_digest,
            }
        )
        if summary.task_id in target_attempt_snapshots
        else summary
        for summary in summaries
    ]
    return {
        "action_summaries": tagged_summaries,
        "action_summary": tagged_summaries[0] if tagged_summaries else None,
        "changed_files": committed_changed_files,
        "retry_diagnostics_by_task": retry_diagnostics_by_task,
        "worker_results_by_attempt": _worker_result_map(
            target_tasks,
            target_attempt_snapshots,
            tagged_summaries,
            succeeded=succeeded,
            errors=runtime.errors,
            attempted_versions_by_task=attempted_by_task,
            executed_versions_by_task=executed_by_task,
            effective_target_version_by_task=effective_versions_by_task,
            effective_dependency_type_by_task=effective_dependency_types_by_task,
            validation_calls=sum(
                1 for event in runtime.tool_events if _is_executed_manifest_transaction(event)
            ),
            manifest_transaction_attempts=int(
                execution_state.get("manifest_transaction_attempts", 0)
            ),
            manifest_transaction_attempts_by_task=manifest_transaction_attempts_by_task,
        ),
        "errors": resolution_errors + runtime.errors,
    }
