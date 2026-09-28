"""Dependency-update worker for Supervisor-committed Phase 5 attempts.

The worker receives one task and its immutable attempt inputs, executes the
focused manifest transaction toolbelt in the Docker workspace, and returns
typed attempt diagnostics for QA and Supervisor reconciliation.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable
from packaging.version import InvalidVersion, Version

try:
    from langchain_openai import ChatOpenAI  # type: ignore[import]
except ImportError:  # pragma: no cover
    ChatOpenAI = None  # type: ignore[assignment,misc]

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
from remediation_engine.language import LANGUAGE_CONFIGS, ProjectLanguage
from remediation_engine.orchestration.remedy_tools import build_update_toolbelt
from remediation_engine.orchestration.runtime_context import get_runtime_settings
from remediation_engine.orchestration.state import SubagentState
from remediation_engine.orchestration.subagent_runtime import run_bounded_subagent_loop
from remediation_engine.orchestration.task_utils import (
    create_skinny_subagent_group,
    filter_constraints_ledger,
    is_transitive_group,
)
from remediation_engine.orchestration.tools_manifest import rollback_pending_package_updates
from remediation_engine.runtime.path_policy import (
    WorkspacePathError,
    resolve_repository_path,
)
from remediation_engine.runtime.sandbox_mgr import DockerSandbox
from remediation_engine.tools.package_identity import normalize_python_package_name
from remediation_engine.tools.repository_map import build_repository_map

logger = logging.getLogger(__name__)


def _group_ecosystem(group: VulnerabilityGroup) -> str:
    for issue in [*group.issues, *(localized.issue for localized in group.localized_issues)]:
        if str(issue.ecosystem or "").strip().casefold() in {"python", "pypi"}:
            return "pypi"
        if str(issue.purl or "").strip().casefold().startswith("pkg:pypi/"):
            return "pypi"
    return "npm"


def _normalize_attempt_version(value: str, ecosystem: str) -> str:
    raw = str(value).strip()
    if ecosystem != "pypi":
        return raw.lstrip("vV")
    try:
        return str(Version(raw))
    except InvalidVersion:
        return raw


def _attempt_version_key(value: str, ecosystem: str) -> str:
    normalized = _normalize_attempt_version(value, ecosystem)
    return normalized if ecosystem == "pypi" else normalized.lower()


_UPDATE_MANIFEST_TOOL_NAMES = frozenset(
    {"modify_and_validate_npm_dependency", "modify_and_validate_python_dependency"}
)
_PYTHON_MANIFEST_TOOL_NAME = "modify_and_validate_python_dependency"
_UPDATE_MANIFEST_TOOL_NAME = "modify_and_validate_npm_dependency"


def _event_ecosystem(event: Any) -> str:
    return "pypi" if getattr(event, "name", "") == _PYTHON_MANIFEST_TOOL_NAME else "npm"


def _event_package_name(event: Any) -> str:
    package = str((getattr(event, "args", {}) or {}).get("package_name", "")).strip()
    ecosystem = _event_ecosystem(event)
    return normalize_python_package_name(package) if ecosystem == "pypi" else package


def _candidate_manifest_paths(
    group: VulnerabilityGroup,
    package_ecosystem: str | None = None,
) -> list[str]:
    """Return candidate manifests, using localization as the only Python source."""
    ecosystem = package_ecosystem or _group_ecosystem(group)
    candidates: list[str] = []
    seen: set[str] = set()

    def add_candidate(value: str | None) -> None:
        if not value:
            return
        candidate = value.replace("\\", "/")
        if candidate not in seen:
            candidates.append(candidate)
            seen.add(candidate)

    if ecosystem == "pypi":
        for localized_issue in group.localized_issues:
            add_candidate(localized_issue.manifest_file)
        return candidates

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
    package_ecosystem: str | None = None,
) -> tuple[list[str], list[str]]:
    """Resolve only the language- and ecosystem-authorized manifest targets."""
    ecosystem = package_ecosystem or _group_ecosystem(group)
    candidates = _candidate_manifest_paths(group, ecosystem)
    if not candidates:
        target = "localized Python manifest" if ecosystem == "pypi" else "package.json"
        return [], [f"Group '{group.group_id}': no {target} target could be resolved."]

    resolved_paths: list[str] = []
    errors: list[str] = []
    requirements_name = re.compile(r"requirements(?:[-_.].+)?\.txt$", re.IGNORECASE)

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
        if abs_target.is_dir():
            errors.append(
                f"Group '{group.group_id}': manifest target '{candidate}' must be a file."
            )
            continue
        basename = abs_target.name
        if ecosystem == "pypi":
            supported = (
                basename in {"pyproject.toml", "setup.cfg", "Pipfile"}
                or requirements_name.fullmatch(basename) is not None
            )
            if not supported:
                errors.append(
                    f"Group '{group.group_id}': Python manifest target '{candidate}' is not editable."
                )
                continue
        elif basename != "package.json":
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
    ecosystem = _group_ecosystem(group)
    if isinstance(task.target_package_name, str) and task.target_package_name.strip():
        package_name = task.target_package_name.strip()
    elif ecosystem == "pypi" and is_transitive_group(group):
        # Pipfile.lock is not parent proof. Never fall back to the vulnerable
        # child or a triage-provided parent for a Python update target.
        package_name = task.parent_package_name or ""
    elif task.strategy_stage != SCARemediationStage.PACKAGE_OVERRIDE:
        package_name = (
            group.parent_package_name
            or next(
                (
                    localized.parent_package_name
                    for localized in group.localized_issues
                    if localized.parent_package_name
                ),
                None,
            )
            or (group.vulnerable_component or "")
        )
    else:
        package_name = group.vulnerable_component or ""
    return (
        normalize_python_package_name(package_name) if ecosystem == "pypi" else package_name.strip()
    )


def _target_dependency_type(task: RemediationTask, group: VulnerabilityGroup) -> str | None:
    """Return the Supervisor-owned manifest declaration type for this task."""
    ecosystem = _group_ecosystem(group)
    if isinstance(task.target_dependency_type, str) and task.target_dependency_type:
        return task.target_dependency_type
    if ecosystem == "pypi" and is_transitive_group(group):
        return None
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
    if task.strategy == RoutingStrategy.VERSION_BUMP and not is_transitive_group(group):
        return None if ecosystem == "pypi" else "dependencies"
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


def _event_matches_task(
    task: RemediationTask,
    group: VulnerabilityGroup,
    manifest_paths: Sequence[str],
    event: Any,
) -> bool:
    ecosystem = _group_ecosystem(group)
    expected_names = (
        {_PYTHON_MANIFEST_TOOL_NAME} if ecosystem == "pypi" else {_UPDATE_MANIFEST_TOOL_NAME}
    )
    if getattr(event, "name", "") not in expected_names:
        return False
    if _event_package_name(event) != _target_package_name(task, group):
        return False
    event_path = str((getattr(event, "args", {}) or {}).get("manifest_path", "")).strip()
    if event_path:
        normalized_event_path = event_path.replace("\\", "/").lstrip("/")
        allowed_paths = {path.replace("\\", "/").lstrip("/") for path in manifest_paths}
        if normalized_event_path not in allowed_paths:
            return False
    return True


def _has_successful_manifest_transaction_for_package(
    task: RemediationTask,
    group: VulnerabilityGroup,
    manifest_paths: Sequence[str],
    tool_events: Sequence[Any] | None,
) -> bool:
    """Return whether this task has a successful edit-and-sync transaction."""
    if not tool_events:
        return False
    return any(
        _event_matches_task(task, group, manifest_paths, event)
        and str(getattr(event, "content", "")).startswith("SUCCESS:")
        for event in tool_events
    )


def _is_executed_manifest_transaction(event: Any) -> bool:
    """Return whether an update event represents an executed transaction."""
    if getattr(event, "name", "") not in _UPDATE_MANIFEST_TOOL_NAMES:
        return False
    content = str(getattr(event, "content", "")).lstrip()
    return not content.startswith(
        ("DEFERRED:", "ERROR_CODE: INVALID_ARGUMENT:", "ERROR_CODE: TARGET_NOT_ALLOWED:")
    )


_UPDATE_WORKER_STATIC_INSTRUCTIONS = """You are a dependency-manifest execution worker.
The Supervisor owns candidate generation, version selection, retry planning, and
task routing. Execute only the Supervisor's task instruction.

Do not search the NPM registry or perform retry planning. Use only
modify_and_validate_npm_dependency for manifest changes. Each call edits the
manifest and immediately synchronizes package manifests before returning.
Transactions are serialized per package. Keep package_name and manifest_path
within the committed task allowlists.

If a transaction returns ERROR_CODE or FAILURE, call the same tool again for that
package with a different Supervisor-approved target_version or dependency_type.
A package may receive at most three combined transaction attempts. Failed
transactions roll back automatically. Continue with the next independent package
after a package succeeds or exhausts its attempts.

Never edit source-code files in this worker. Supervisor-owned dependency-type
candidates are the only permitted strategy alternatives.

Return control only after every package has one successful combined transaction or
has exhausted its three attempts and been surrendered."""


_PYTHON_UPDATE_WORKER_STATIC_INSTRUCTIONS = """You are a Python dependency-manifest
execution worker. The Supervisor owns candidate generation, version selection,
retry planning, and task routing. Execute only the Supervisor's exact task
instruction and committed target version; never choose or substitute a version.

Supported editable dependency declarations are requirements files such as
requirements.txt, static PEP 621 dependencies in pyproject.toml, static setup.cfg
declarations, and Pipfile. setup.py is install-only and must not be edited. Use
only modify_and_validate_python_dependency for dependency-manifest changes. Do
not search PyPI or another package registry, read registry evidence, or perform
version selection. Keep package_name and manifest_path within the committed task
allowlists.

If a transaction fails, retry only when the Supervisor has supplied a revised
exact task instruction and committed target. Do not choose an alternative version
or dependency declaration. A package may receive at most three combined
transaction attempts; failed transactions roll back automatically. Continue
with the next independent package after a package succeeds or exhausts its
attempts.

Never edit source-code files in this worker. Use only the dependency type named
by the Supervisor's committed instruction; do not choose strategy alternatives.

Return control only after every package has one successful combined transaction or
has exhausted its three attempts and been surrendered."""


def _build_update_system_prompt(
    project_language: ProjectLanguage = ProjectLanguage.NODEJS,
) -> str:
    """Return the language-specific dependency worker prompt."""
    if not isinstance(project_language, ProjectLanguage):
        project_language = ProjectLanguage(project_language)
    if project_language == ProjectLanguage.PYTHON:
        return _PYTHON_UPDATE_WORKER_STATIC_INSTRUCTIONS
    return _UPDATE_WORKER_STATIC_INSTRUCTIONS


def _build_update_prompt(
    resolved_tasks: Sequence[tuple[RemediationTask, VulnerabilityGroup, Sequence[str]]],
    constraints_ledger: Sequence[str],
    feedback_by_task: dict[str, str],
    previous_action_summaries_by_task: dict[str, str],
    retry_diagnostics_by_task: dict[str, UpdateRetryDiagnostics] | None = None,
    repository_map: str = "(repository map unavailable)",
    allowed_target_versions_by_task: Mapping[str, Sequence[str]] | None = None,
    allowed_dependency_types_by_task: Mapping[str, Sequence[str]] | None = None,
    project_language: ProjectLanguage = ProjectLanguage.NODEJS,
) -> str:
    """Build the dynamic execution context for the static worker instructions."""
    if not isinstance(project_language, ProjectLanguage):
        project_language = ProjectLanguage(project_language)
    python_project = project_language == ProjectLanguage.PYTHON
    allowed_target_versions_by_task = allowed_target_versions_by_task or {}
    allowed_dependency_types_by_task = allowed_dependency_types_by_task or {}
    retry_diagnostics_by_task = retry_diagnostics_by_task or {}
    sections = [
        "DYNAMIC TASK CONTEXT (Supervisor-owned and authoritative):",
        "",
        "Deterministic repository map:",
        repository_map,
        "",
        "Constraints ledger:",
    ]
    if python_project:
        sections.append("Canonical project language: python (Python).")
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
        if python_project:
            allowed_versions = [task.selected_version] if task.selected_version else []
            dependency_type = _target_dependency_type(task, group)
            allowed_types = [dependency_type] if dependency_type else []
        version_line = (
            f"- Committed target version: {task.selected_version or 'none supplied'}"
            if python_project
            else f"- Allowed target versions: {', '.join(allowed_versions) or 'none supplied'}"
        )
        dependency_type_line = (
            f"- Committed dependency type: {allowed_types[0] if allowed_types else 'none supplied'}"
            if python_project
            else f"- Allowed dependency types: {', '.join(dict.fromkeys(allowed_types)) or 'none supplied'}"
        )
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
                version_line,
                dependency_type_line,
                f"- Exact supervisor instruction: {task.instruction or '(missing)'}",
                f"- QA feedback: {feedback_by_task.get(task.task_id, 'none')}",
                f"- Previous outcome: {previous_action_summaries_by_task.get(task.task_id, 'none')}",
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
    del retry_batch
    normalized_changed_files = {path.replace("\\", "/") for path in changed_files}
    final_note = (final_text or "").strip()
    summaries: list[AgentActionSummary] = []
    for task, group, manifest_paths in resolved_tasks:
        package_modified = _has_successful_manifest_transaction_for_package(
            task,
            group,
            manifest_paths,
            tool_events,
        )
        task_succeeded = package_modified
        if tool_events is None:
            task_succeeded = succeeded and bool(normalized_changed_files)
        changed = [
            path
            for path in _authorized_changed_paths(group, manifest_paths)
            if path.replace("\\", "/") in normalized_changed_files
        ]
        status = AgentActionStatus.SUCCESS if task_succeeded else AgentActionStatus.SURRENDER
        outcome = (
            "Completed validated manifest updates"
            if task_succeeded
            else "Stopped without a validated manifest update"
        )
        changed_label = changed or (manifest_paths if package_modified else ["no files"])
        summary = (
            f"{outcome} for {group.vulnerable_component or 'unknown component'} in "
            f"{', '.join(manifest_paths) or 'no manifest'}; changed files: "
            f"{', '.join(changed_label)}."
        )
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
    changed_files_by_task: Mapping[str, Sequence[str]] | None = None,
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
    changed_files_by_task = changed_files_by_task or {}
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
            changed_files=list(changed_files_by_task.get(task.task_id, [])),
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


def _authorized_changed_paths(
    group: VulnerabilityGroup,
    manifest_paths: Sequence[str],
) -> list[str]:
    """Include generated Pipfile.lock files in each task's file partition."""
    paths = list(manifest_paths)
    if _group_ecosystem(group) == "pypi":
        for manifest_path in manifest_paths:
            if Path(manifest_path).name == "Pipfile":
                lock_path = str(Path(manifest_path).with_name("Pipfile.lock")).replace("\\", "/")
                if lock_path not in paths:
                    paths.append(lock_path)
    return paths


def _changed_files_by_task(
    resolved_tasks: Sequence[tuple[RemediationTask, VulnerabilityGroup, Sequence[str]]],
    changed_files: Sequence[str],
) -> dict[str, list[str]]:
    """Partition committed worker files by each task's authorized files."""
    normalized_files = list(
        dict.fromkeys(
            path.replace("\\", "/").lstrip("/")
            for path in changed_files
            if isinstance(path, str) and path.strip()
        )
    )
    result: dict[str, list[str]] = {}
    for task, group, manifest_paths in resolved_tasks:
        manifest_set = {
            path.replace("\\", "/").lstrip("/")
            for path in _authorized_changed_paths(group, manifest_paths)
        }
        result[task.task_id] = [path for path in normalized_files if path in manifest_set]
    return result


def _attempted_versions_for_current_run(
    resolved_tasks: Sequence[tuple[RemediationTask, VulnerabilityGroup, Sequence[str]]],
    tool_events: Sequence[Any],
) -> dict[str, list[str]]:
    """Collect version targets from transactions correlated to each task."""
    result: dict[str, list[str]] = {task.task_id: [] for task, _, _ in resolved_tasks}
    for task, group, manifest_paths in resolved_tasks:
        ecosystem = _group_ecosystem(group)
        for event in tool_events:
            if not _is_executed_manifest_transaction(event) or not _event_matches_task(
                task, group, manifest_paths, event
            ):
                continue
            target = _normalize_attempt_version(
                str((getattr(event, "args", {}) or {}).get("target_version", "")),
                ecosystem,
            )
            if target and target not in result[task.task_id]:
                result[task.task_id].append(target)
    return result


def _executed_versions_for_current_run(
    resolved_tasks: Sequence[tuple[RemediationTask, VulnerabilityGroup, Sequence[str]]],
    tool_events: Sequence[Any],
) -> dict[str, list[str]]:
    """Collect successful version targets correlated to each current task."""
    result: dict[str, list[str]] = {task.task_id: [] for task, _, _ in resolved_tasks}
    for task, group, manifest_paths in resolved_tasks:
        ecosystem = _group_ecosystem(group)
        for event in tool_events:
            if (
                not _is_executed_manifest_transaction(event)
                or not str(getattr(event, "content", "")).startswith("SUCCESS:")
                or not _event_matches_task(task, group, manifest_paths, event)
            ):
                continue
            target = _normalize_attempt_version(
                str((getattr(event, "args", {}) or {}).get("target_version", "")),
                ecosystem,
            )
            if target and target not in result[task.task_id]:
                result[task.task_id].append(target)
    return result


def _attempted_dependency_types_for_current_run(
    resolved_tasks: Sequence[tuple[RemediationTask, VulnerabilityGroup, Sequence[str]]],
    tool_events: Sequence[Any],
) -> dict[str, list[str]]:
    """Collect declaration types from transactions correlated to each task."""
    result: dict[str, list[str]] = {task.task_id: [] for task, _, _ in resolved_tasks}
    for task, group, manifest_paths in resolved_tasks:
        for event in tool_events:
            if not _is_executed_manifest_transaction(event) or not _event_matches_task(
                task, group, manifest_paths, event
            ):
                continue
            dependency_type = str(
                (getattr(event, "args", {}) or {}).get("dependency_type", "")
            ).strip()
            if dependency_type and dependency_type not in result[task.task_id]:
                result[task.task_id].append(dependency_type)
    return result


def _effective_targets_for_current_run(
    resolved_tasks: Sequence[tuple[RemediationTask, VulnerabilityGroup, Sequence[str]]],
    tool_events: Sequence[Any],
) -> tuple[dict[str, str | None], dict[str, str | None]]:
    """Collect effective version/type from successful transactions per task."""
    versions: dict[str, str | None] = {task.task_id: None for task, _, _ in resolved_tasks}
    dependency_types: dict[str, str | None] = {task.task_id: None for task, _, _ in resolved_tasks}
    for task, group, manifest_paths in resolved_tasks:
        ecosystem = _group_ecosystem(group)
        for event in tool_events:
            if (
                not _is_executed_manifest_transaction(event)
                or not str(getattr(event, "content", "")).startswith("SUCCESS:")
                or not _event_matches_task(task, group, manifest_paths, event)
            ):
                continue
            args = getattr(event, "args", {}) or {}
            versions[task.task_id] = (
                _normalize_attempt_version(
                    str(args.get("target_version", "")),
                    ecosystem,
                )
                or versions[task.task_id]
            )
            dependency_types[task.task_id] = (
                str(args.get("dependency_type", "")).strip() or dependency_types[task.task_id]
            )
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
    for task, group, manifest_paths in resolved_tasks:
        ecosystem = _group_ecosystem(group)
        target_package = _target_package_name(task, group)
        target_dependency_type = _target_dependency_type(task, group)
        prior = prior_diagnostics_by_task.get(task.task_id)
        prior_attempts_by_target = (
            {
                (normalize_python_package_name(package) if ecosystem == "pypi" else package): list(
                    dict.fromkeys(
                        normalized
                        for version in versions
                        if (normalized := _normalize_attempt_version(version, ecosystem))
                    )
                )
                for package, versions in prior.attempted_versions_by_target.items()
            }
            if prior
            else {}
        )
        attempted = list(
            dict.fromkeys(
                normalized
                for version in prior_attempts_by_target.get(
                    target_package,
                    prior.attempted_versions if prior else [],
                )
                if (normalized := _normalize_attempt_version(version, ecosystem))
            )
        )
        executed = [
            normalized
            for version in (prior.executed_versions if prior else [])
            if (normalized := _normalize_attempt_version(version, ecosystem))
        ]
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
        if effective_target_version:
            effective_target_version = (
                _normalize_attempt_version(
                    effective_target_version,
                    ecosystem,
                )
                or None
            )
        effective_dependency_type = prior.effective_dependency_type if prior else None
        used_overrides = bool(prior.used_overrides) if prior else False
        for event in tool_events:
            if not _is_executed_manifest_transaction(event) or not _event_matches_task(
                task, group, manifest_paths, event
            ):
                continue
            event_args = getattr(event, "args", {}) or {}
            target = _normalize_attempt_version(
                str(event_args.get("target_version", "")),
                ecosystem,
            )
            if target and target not in attempted:
                attempted.append(target)
            dependency_type = str(event_args.get("dependency_type", "")).strip()
            if dependency_type and dependency_type not in attempted_dependency_types:
                attempted_dependency_types.append(dependency_type)
            used_overrides = used_overrides or dependency_type in {
                "overrides",
                "resolutions",
                "pnpm_overrides",
            }
            if str(getattr(event, "content", "")).startswith("SUCCESS:"):
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
    """Run the dependency-update worker for Supervisor-committed targets.

    The Supervisor normally supplies one task per invocation. Results are
    correlated to each task's immutable attempt snapshot and include typed
    execution diagnostics for subsequent QA and retry planning.
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
    project_language = state.get("project_language", ProjectLanguage.NODEJS)
    if not isinstance(project_language, ProjectLanguage):
        project_language = ProjectLanguage(project_language)
    package_ecosystems = {_group_ecosystem(group) for _, group, _ in resolved_tasks}
    if len(package_ecosystems) != 1:
        msg = "Update Subagent: one worker invocation cannot mix package ecosystems."
        summaries = _build_surrender_summaries(resolved_task_ids, msg)
        return {
            "action_summaries": summaries,
            "action_summary": summaries[0] if summaries else None,
            "changed_files": [],
            "errors": resolution_errors + [msg],
        }
    package_ecosystem = next(iter(package_ecosystems))
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
        project_language=project_language,
    )
    initial_messages = [SystemMessage(content=_build_update_system_prompt(project_language))]
    if state.get("messages"):
        initial_messages.extend(state["messages"])
    initial_messages.append(HumanMessage(content=prompt))

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
        with DockerSandbox(
            repo_root=None,
            image=LANGUAGE_CONFIGS[project_language].docker_image,
            workspace_volume=workspace_volume,
        ) as sandbox:
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
                language=project_language,
                package_ecosystem=package_ecosystem,
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
    successful_task_ids = {
        task.task_id
        for task, group, manifest_paths in resolved_tasks
        if _has_successful_manifest_transaction_for_package(
            task,
            group,
            manifest_paths,
            runtime.tool_events,
        )
    }
    authorized_changed_paths = {
        path.replace("\\", "/").lstrip("/")
        for _, group, manifest_paths in resolved_tasks
        for path in _authorized_changed_paths(group, manifest_paths)
    }
    successful_authorized_paths = {
        path.replace("\\", "/").lstrip("/")
        for task, group, manifest_paths in resolved_tasks
        if task.task_id in successful_task_ids
        for path in _authorized_changed_paths(group, manifest_paths)
    }
    unvalidated_manifest_paths = {
        path.replace("\\", "/").lstrip("/")
        for task, group, manifest_paths in resolved_tasks
        if task.task_id not in successful_task_ids
        for path in _authorized_changed_paths(group, manifest_paths)
    } - successful_authorized_paths
    committed_changed_files = sorted(
        {
            path.replace("\\", "/").lstrip("/")
            for path in {*touched_files, *runtime.changed_files}
            if path.replace("\\", "/").lstrip("/") in authorized_changed_paths
            and path.replace("\\", "/").lstrip("/") not in unvalidated_manifest_paths
        }
    )
    # A worker succeeds only when every task has its own successful transaction.
    succeeded = bool(package_names) and len(successful_task_ids) == len(resolved_tasks)
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
    for task, group, _manifest_paths in resolved_tasks:
        snapshot = target_attempt_snapshots.get(task.task_id)
        if snapshot is None:
            continue
        ecosystem = _group_ecosystem(group)
        allowed_versions = list(getattr(snapshot, "allowed_target_versions", []) or [])
        if not allowed_versions and snapshot.selected_version:
            allowed_versions = [snapshot.selected_version]
        allowed = {
            key for version in allowed_versions if (key := _attempt_version_key(version, ecosystem))
        }
        observed = {
            key
            for version in attempted_by_task.get(task.task_id, [])
            if (key := _attempt_version_key(version, ecosystem))
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
    changed_files_by_task = _changed_files_by_task(
        resolved_tasks,
        committed_changed_files,
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
            changed_files_by_task=changed_files_by_task,
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
