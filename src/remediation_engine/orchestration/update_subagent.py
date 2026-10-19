"""Dependency-update worker for Supervisor-committed Phase 5 attempts.

The worker receives one task or one Supervisor-committed package cluster and
its immutable attempt inputs, executes the focused manifest transaction
toolbelt in the Docker workspace, and returns typed attempt diagnostics for QA
and Supervisor reconciliation.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable

from remediation_engine.contracts.schemas import (
    AgentActionStatus,
    AgentActionSummary,
    MultiPackageAction,
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
from remediation_engine.orchestration.remedy_tools import (
    build_multi_package_update_toolbelt,
    build_update_toolbelt,
)
from remediation_engine.orchestration.runtime_context import (
    get_runtime_settings,
    openai_api_key_kwargs,
)
from remediation_engine.orchestration.state import SubagentState
from remediation_engine.orchestration.subagent_runtime import run_bounded_subagent_loop
from remediation_engine.orchestration.supervisor_planner import instruction_digest
from remediation_engine.orchestration.task_utils import (
    create_skinny_subagent_group,
    filter_constraints_ledger,
    is_transitive_group,
    select_package_fix_plan,
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

_UPDATE_MANIFEST_TOOL_NAME = "modify_and_validate_npm_dependency"
_PYTHON_UPDATE_MANIFEST_TOOL_NAME = "modify_and_validate_python_dependency"
_UPDATE_MANIFEST_TOOL_NAMES = frozenset(
    {_UPDATE_MANIFEST_TOOL_NAME, _PYTHON_UPDATE_MANIFEST_TOOL_NAME}
)
_BATCH_NPM_DEPENDENCIES_TOOL_NAME = "modify_batch_npm_dependencies"
_BATCH_PYTHON_DEPENDENCIES_TOOL_NAME = "modify_batch_python_dependencies"
_MAJOR_MIGRATION_DIRECTIVE = (
    "Migrate all affected production and test code to the selected package API "
    "while preserving behavior; do not change the solver-approved package or version."
)

try:
    from langchain_openai import ChatOpenAI  # type: ignore[import]
except ImportError:  # pragma: no cover
    ChatOpenAI = None  # type: ignore[assignment,misc]


def _group_ecosystem(group: VulnerabilityGroup) -> str:
    """Return the single package ecosystem identified by a grouped finding."""
    for issue in [*group.issues, *(item.issue for item in group.localized_issues)]:
        if str(issue.ecosystem or "").strip().casefold() in {"python", "pypi"}:
            return "pypi"
        if str(issue.purl or "").strip().casefold().startswith("pkg:pypi/"):
            return "pypi"
    return "npm"


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

    if _group_ecosystem(group) == "pypi":
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
) -> tuple[list[str], list[str]]:
    """Resolve existing, safe declaration paths for one vulnerability group."""
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
        if abs_target.is_dir():
            errors.append(
                f"Group '{group.group_id}': manifest target '{candidate}' is a directory."
            )
            continue
        if _group_ecosystem(group) == "pypi":
            try:
                from remediation_engine.orchestration.tools_manifest import (
                    _normalize_python_manifest_targets,
                )

                _normalize_python_manifest_targets([candidate])
            except ValueError as exc:
                errors.append(
                    f"Group '{group.group_id}': unsupported Python manifest '{candidate}': {exc}"
                )
                continue
        elif abs_target.name != "package.json":
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
        value = task.target_package_name.strip()
        return normalize_python_package_name(value) if ecosystem == "pypi" else value
    if task.strategy_stage != SCARemediationStage.PACKAGE_OVERRIDE:
        if group.parent_package_name:
            value = group.parent_package_name.strip()
            return normalize_python_package_name(value) if ecosystem == "pypi" else value
        for localized in group.localized_issues:
            if localized.parent_package_name:
                value = localized.parent_package_name.strip()
                return normalize_python_package_name(value) if ecosystem == "pypi" else value
    value = (group.vulnerable_component or "").strip()
    return normalize_python_package_name(value) if ecosystem == "pypi" else value


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


def _validate_committed_multi_package_action(
    state: SubagentState,
    action: Any,
    resolved_tasks: Sequence[tuple[RemediationTask, VulnerabilityGroup, Sequence[str]]],
    snapshots: Mapping[str, Any],
) -> list[str]:
    """Validate a committed cluster action immediately before mutation."""
    errors: list[str] = []
    if not isinstance(action, MultiPackageAction):
        return ["Update Subagent: active cluster action is not a validated MultiPackageAction."]
    expected_task_ids = {task.task_id for task, _, _ in resolved_tasks}
    if len(expected_task_ids) < 2:
        errors.append("Update Subagent: atomic action requires at least two resolved tasks.")
    if action.cluster_id != state.get("active_cluster_id"):
        errors.append("Update Subagent: action cluster_id does not match active cluster metadata.")
    if action.dispatch_batch_id != state.get("dispatch_batch_id"):
        errors.append(
            "Update Subagent: action dispatch_batch_id does not match active batch metadata."
        )
    action_digest = instruction_digest(action.model_dump_json())
    expected_batch_id = state.get("dispatch_batch_id")
    mutations_by_task: dict[str, list[Any]] = {}
    for mutation in action.package_mutations:
        mutations_by_task.setdefault(mutation.task_id, []).append(mutation)
    if set(mutations_by_task) != expected_task_ids:
        errors.append("Update Subagent: action mutations do not cover exactly the active tasks.")
    if any(len(mutations) != 1 for mutations in mutations_by_task.values()):
        errors.append("Update Subagent: each active task must map to exactly one package mutation.")

    for task, group, manifest_paths in resolved_tasks:
        snapshot = snapshots.get(task.task_id)
        if snapshot is None:
            errors.append(f"Update Subagent: missing committed snapshot for task {task.task_id}.")
            continue
        if (
            snapshot.cluster_id != state.get("active_cluster_id")
            or snapshot.dispatch_batch_id != expected_batch_id
            or snapshot.action_digest != action_digest
        ):
            errors.append(f"Update Subagent: snapshot provenance mismatch for task {task.task_id}.")
        mutation = mutations_by_task.get(task.task_id, [None])[0]
        if mutation is None:
            continue
        expected_package = _target_package_name(task, group)
        expected_type = _target_dependency_type(task, group)
        allowed_versions = {
            value.strip().lstrip("vV") for value in snapshot.allowed_target_versions if value
        }
        if snapshot.selected_version:
            allowed_versions.add(snapshot.selected_version.strip().lstrip("vV"))
        allowed_types = {
            value.strip().lower() for value in snapshot.allowed_dependency_types if value
        }
        if snapshot.target_dependency_type:
            allowed_types.add(snapshot.target_dependency_type.strip().lower())
        valid_manifest_paths = {path.replace("\\", "/").lstrip("/") for path in manifest_paths}
        if mutation.package_name != expected_package:
            errors.append(f"Update Subagent: package mismatch for task {task.task_id}.")
        if mutation.ecosystem != _group_ecosystem(group):
            errors.append(f"Update Subagent: ecosystem mismatch for task {task.task_id}.")
        if mutation.manifest_path not in valid_manifest_paths:
            errors.append(f"Update Subagent: manifest mismatch for task {task.task_id}.")
        if snapshot.manifest_path and mutation.manifest_path != snapshot.manifest_path:
            errors.append(
                f"Update Subagent: manifest does not match the committed snapshot for task {task.task_id}."
            )
        if mutation.target_version.strip().lstrip("vV") not in allowed_versions:
            errors.append(
                f"Update Subagent: target version is not allowlisted for task {task.task_id}."
            )
        if expected_type and mutation.dependency_type.lower() not in allowed_types:
            errors.append(
                f"Update Subagent: dependency type is not allowlisted for task {task.task_id}."
            )
    return errors


def _has_successful_manifest_transaction_for_package(
    task: RemediationTask,
    group: VulnerabilityGroup,
    tool_events: Sequence[Any] | None,
    manifest_paths: Sequence[str] | None = None,
    resolved_tasks: Sequence[tuple[RemediationTask, VulnerabilityGroup, Sequence[str]]]
    | None = None,
) -> bool:
    """Return whether the package has a successful edit-and-sync transaction."""
    if not tool_events:
        return False
    return any(
        getattr(event, "name", "") in _UPDATE_MANIFEST_TOOL_NAMES
        and (
            task.task_id in _event_matching_task_ids(event, resolved_tasks)
            if resolved_tasks is not None
            else _event_matches_task_target(event, task, group, manifest_paths or ())
        )
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


def _event_package_name(event: Any) -> str:
    """Normalize the package identity using the tool's language-specific semantics."""
    package = str((getattr(event, "args", {}) or {}).get("package_name", "")).strip()
    if getattr(event, "name", "") == _PYTHON_UPDATE_MANIFEST_TOOL_NAME:
        return normalize_python_package_name(package)
    return package


def _event_matches_task_target(
    event: Any,
    task: RemediationTask,
    group: VulnerabilityGroup,
    manifest_paths: Sequence[str],
) -> bool:
    """Match an execution event to its sealed package, declaration, and manifest."""
    ecosystem = _group_ecosystem(group)
    event_name = getattr(event, "name", "")
    if (event_name == _PYTHON_UPDATE_MANIFEST_TOOL_NAME) != (ecosystem == "pypi"):
        return False
    if _event_package_name(event) != _target_package_name(task, group):
        return False
    args = getattr(event, "args", {}) or {}
    event_dependency_type = str(args.get("dependency_type", "")).strip().casefold()
    target_dependency_type = str(_target_dependency_type(task, group) or "").strip().casefold()
    if (
        event_dependency_type
        and target_dependency_type
        and event_dependency_type != target_dependency_type
    ):
        return False
    event_manifest = str(args.get("manifest_path", "")).replace("\\", "/").lstrip("/")
    allowed_manifests = {
        str(path).replace("\\", "/").lstrip("/") for path in manifest_paths if str(path).strip()
    }
    return not event_manifest or not allowed_manifests or event_manifest in allowed_manifests


def _event_matching_task_ids(
    event: Any,
    resolved_tasks: Sequence[tuple[RemediationTask, VulnerabilityGroup, Sequence[str]]],
) -> set[str]:
    """Return unambiguous task owners for a manifest transaction event."""
    matched = [
        (task, group, manifest_paths)
        for task, group, manifest_paths in resolved_tasks
        if _event_matches_task_target(event, task, group, manifest_paths)
    ]
    if not matched:
        return set()

    args = getattr(event, "args", {}) or {}
    event_manifest = str(args.get("manifest_path", "")).replace("\\", "/").lstrip("/")
    if not event_manifest:
        # Older tool traces may omit the manifest. Keep them usable when the
        # package/declaration identifies one target, but do not credit the
        # same package in distinct Python or JavaScript manifests.
        target_identities = {
            (
                _group_ecosystem(group),
                _target_package_name(task, group),
                (_target_dependency_type(task, group) or "").casefold(),
                tuple(sorted(path.replace("\\", "/").lstrip("/") for path in manifest_paths)),
            )
            for task, group, manifest_paths in matched
        }
        if len(target_identities) > 1:
            return set()
    return {task.task_id for task, _group, _manifest_paths in matched}


def _event_target_version(event: Any, ecosystem: str) -> str:
    """Return the exact version target supplied to the manifest tool."""
    del ecosystem
    raw = str((getattr(event, "args", {}) or {}).get("target_version", "")).strip()
    return raw.lstrip("vV")


def _append_attempted_version(versions: list[str], candidate: str, ecosystem: str) -> None:
    """Append an exact version argument once, preserving the execution trace."""
    del ecosystem
    if candidate and candidate not in versions:
        versions.append(candidate)


_UPDATE_WORKER_STATIC_INSTRUCTIONS = """You are a dependency-manifest execution worker.
The Supervisor owns candidate generation, version selection, retry planning, and
task routing. Execute only the Supervisor's task instruction.

Do not search the NPM registry or perform retry planning. Use only
modify_and_validate_npm_dependency for manifest changes. Each call edits the
manifest and immediately synchronizes package manifests before returning.
Transactions are serialized per package. Keep package_name and manifest_path
within the committed task allowlists.

If the exact Supervisor instruction requires a major-version migration, you are
authorized and required to update affected production and test code to the selected
package API while preserving behavior. Do not reselect or change the committed
package or version. Otherwise, do not edit source-code files.

If a transaction returns ERROR_CODE or FAILURE, retry only with a target_version
or dependency_type explicitly allowed by the Supervisor. A major-version migration
instruction never authorizes another package or version. A package may receive at
most three combined transaction attempts. Failed transactions roll back
automatically. Continue with the next independent package after a package succeeds
or exhausts its attempts.

Return control only after every package has one successful combined transaction or
has exhausted its three attempts and been surrendered."""

_PYTHON_UPDATE_WORKER_STATIC_INSTRUCTIONS = """You are a Python dependency-manifest
execution worker. The Supervisor owns candidate generation, version selection,
retry planning, and task routing. Execute only the Supervisor's committed task
instruction, target package, candidate rule, and declaration type.

Supported editable declarations are requirements.txt variants, static PEP 621
dependencies in pyproject.toml, static setup.cfg declarations, and Pipfile.
setup.py is install-only and must not be edited. Use only modify_and_validate_python_dependency
for dependency-manifest changes. Do
not search PyPI or another package registry, read registry evidence, or perform
version selection. Keep package_name and manifest_path within the committed task
allowlists.

If the exact Supervisor instruction requires a major-version migration, you are
authorized and required to update affected production and test code to the
selected package API while preserving behavior. Do not reselect or change the
committed package or version. Otherwise, do not edit source-code files.

If a transaction returns ERROR_CODE or FAILURE, retry only with a target version
or dependency type explicitly allowed by the committed attempt. A major-version
migration instruction never authorizes another package or version. Do not invent
alternatives or expand the candidate set. A package may receive at most three
combined transaction attempts; failed transactions roll back automatically.
Continue with the next independent package after a package succeeds or exhausts
its attempts.

Return control only after every package has one successful combined transaction
or has exhausted its three attempts and been surrendered."""


def _build_update_system_prompt(
    project_language: ProjectLanguage = ProjectLanguage.NODEJS,
) -> str:
    """Return the update worker's language-specific capability instructions."""
    if not isinstance(project_language, ProjectLanguage):
        project_language = ProjectLanguage(project_language)
    if project_language == ProjectLanguage.PYTHON:
        return _PYTHON_UPDATE_WORKER_STATIC_INSTRUCTIONS
    return _UPDATE_WORKER_STATIC_INSTRUCTIONS


_MULTI_PACKAGE_UPDATE_WORKER_STATIC_INSTRUCTIONS = """You are a dependency-manifest
cluster execution worker.

The Supervisor has already selected and validated one atomic multi-package action.
It owns candidate generation, package/version selection, dependency types, retry
planning, task routing, and cluster membership. Your only job is to execute that
committed action in the Docker workspace.

You have exactly one tool: modify_batch_npm_dependencies. Call it exactly
once, with no arguments. The tool applies every package mutation in the committed
action, synchronizes each affected npm manifest, validates the requested results,
and rolls back the complete cluster if any step fails.

When a task's exact Supervisor instruction requires a major-version migration,
source and test migration to the committed package API is authorized and required;
preserve behavior. Do not reselect or change any package name or version in the
committed action. Do not call the singleton dependency tool, search the registry,
or perform retry planning. After the tool returns, report its result and return
control to the Supervisor. A failed or rolled-back action must be surrendered for
a new Supervisor-committed attempt."""

_MULTI_PACKAGE_PYTHON_UPDATE_WORKER_STATIC_INSTRUCTIONS = """You are a Python
dependency-manifest cluster execution worker.

The Supervisor has already selected and validated one atomic multi-package action.
It owns candidate generation, package/version selection, declaration types, retry
planning, task routing, and cluster membership. Your only job is to execute that
committed action in the Docker workspace.

You have exactly one tool: modify_batch_python_dependencies. Call it exactly
once, with no arguments. The tool applies every package mutation in the committed
action, synchronizes each affected Python manifest, validates the requested
results, and rolls back the complete cluster if any step fails.

When a task's exact Supervisor instruction requires a major-version migration,
source and test migration to the committed package API is authorized and required;
preserve behavior. Do not reselect or change any package name or version in the
committed action. Do not call the singleton dependency tool, search PyPI, or
perform retry planning. After the tool returns, report its result and return
control to the Supervisor. A failed or rolled-back action must be surrendered for
a new Supervisor-committed attempt."""


def _build_multi_package_update_prompt(
    resolved_tasks: Sequence[tuple[RemediationTask, VulnerabilityGroup, Sequence[str]]],
    action: MultiPackageAction,
    constraints_ledger: Sequence[str],
    feedback_by_task: dict[str, str],
    previous_action_summaries_by_task: dict[str, str],
    retry_diagnostics_by_task: dict[str, UpdateRetryDiagnostics] | None = None,
    repository_map: str = "(repository map unavailable)",
    allowed_target_versions_by_task: Mapping[str, Sequence[str]] | None = None,
    allowed_dependency_types_by_task: Mapping[str, Sequence[str]] | None = None,
    project_language: ProjectLanguage = ProjectLanguage.NODEJS,
) -> str:
    """Build the authoritative prompt for one committed package cluster.

    The action JSON is included as provenance and context only. The worker tool
    closes over the same immutable action, so model-produced arguments cannot
    alter the mutations that reach the deterministic executor.
    """
    task_context = _build_update_prompt(
        resolved_tasks,
        constraints_ledger,
        feedback_by_task,
        previous_action_summaries_by_task,
        retry_diagnostics_by_task,
        repository_map=repository_map,
        allowed_target_versions_by_task=allowed_target_versions_by_task,
        allowed_dependency_types_by_task=allowed_dependency_types_by_task,
        project_language=project_language,
    )
    if not isinstance(project_language, ProjectLanguage):
        project_language = ProjectLanguage(project_language)
    tool_name = (
        _BATCH_PYTHON_DEPENDENCIES_TOOL_NAME
        if project_language == ProjectLanguage.PYTHON
        else _BATCH_NPM_DEPENDENCIES_TOOL_NAME
    )
    return "\n".join(
        [
            "MULTI-PACKAGE CLUSTER EXECUTION (Supervisor-owned and authoritative):",
            "",
            f"Cluster ID: {action.cluster_id or '(missing)'}",
            f"Dispatch batch ID: {action.dispatch_batch_id or '(missing)'}",
            f"Committed action digest: {instruction_digest(action.model_dump_json())}",
            "",
            "The following immutable action is the only action you may execute:",
            action.model_dump_json(),
            "",
            f"Invoke {tool_name} exactly once with no arguments.",
            "The tool is the only permitted mutation path and applies all listed package mutations atomically.",
            "",
            task_context,
        ]
    )


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
        migration_required = _MAJOR_MIGRATION_DIRECTIVE in str(task.instruction or "")
        if migration_required and task.selected_version:
            allowed_versions = [task.selected_version]
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
                (
                    f"- Committed target version: {task.selected_version or 'none supplied'}"
                    if python_project
                    else f"- Allowed target versions: {', '.join(allowed_versions) or 'none supplied'}"
                ),
                (
                    f"- Committed dependency type: {_target_dependency_type(task, group) or 'none supplied'}"
                    if python_project
                    else f"- Allowed dependency types: {', '.join(dict.fromkeys(allowed_types)) or 'none supplied'}"
                ),
                f"- Exact supervisor instruction: {task.instruction or '(missing)'}",
                f"- QA feedback: {feedback_by_task.get(task.task_id, 'none')}",
                f"- Previous outcome: {previous_action_summaries_by_task.get(task.task_id, 'none')}",
            ]
        )
        if migration_required:
            sections.extend(
                [
                    "- Major-version source/test migration: required.",
                    "- Keep the selected package and version exactly as committed; "
                    "do not reselect or retry another version.",
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
            task, group, tool_events, manifest_paths, resolved_tasks
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
        task_succeeded = (
            succeeded
            and not errors
            and summary is not None
            and summary.status == AgentActionStatus.SUCCESS
        )
        results[snapshot.attempt_id] = WorkerAttemptResult(
            attempt_id=snapshot.attempt_id,
            task_id=task.task_id,
            task_revision=snapshot.task_revision,
            cluster_id=snapshot.cluster_id,
            dispatch_batch_id=snapshot.dispatch_batch_id,
            action_digest=snapshot.action_digest,
            status=(
                summary.status
                if task_succeeded and summary is not None
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
    return results


def _changed_files_by_task(
    resolved_tasks: Sequence[tuple[RemediationTask, VulnerabilityGroup, Sequence[str]]],
    changed_files: Sequence[str],
) -> dict[str, list[str]]:
    """Partition committed worker files by task for attempt envelopes.

    A normal Supervisor dispatch contains one task and receives the complete
    committed file set. When multiple task records are present, manifest
    ownership provides a conservative deterministic partition so a task cannot
    claim another task's package files.
    """
    normalized_files = list(
        dict.fromkeys(
            path.replace("\\", "/").lstrip("/")
            for path in changed_files
            if isinstance(path, str) and path.strip()
        )
    )

    def is_foreign_package_artifact(
        path: str,
        ecosystem: str,
    ) -> bool:
        """Reject incidental manifests and lockfiles from the other ecosystem."""
        name = PurePosixPath(path).name.casefold()
        if ecosystem == "pypi":
            return name in {
                "package.json",
                "package-lock.json",
                "npm-shrinkwrap.json",
                "yarn.lock",
                "pnpm-lock.yaml",
            }
        return name in {
            "pyproject.toml",
            "setup.cfg",
            "pipfile",
            "pipfile.lock",
            "poetry.lock",
            "uv.lock",
        } or (name.startswith("requirements") and name.endswith(".txt"))

    if len(resolved_tasks) == 1:
        task, group, _manifest_paths = resolved_tasks[0]
        ecosystem = _group_ecosystem(group)
        return {
            task.task_id: [
                path
                for path in normalized_files
                if not is_foreign_package_artifact(path, ecosystem)
            ]
        }

    result: dict[str, list[str]] = {}
    for task, _group, manifest_paths in resolved_tasks:
        ecosystem = _group_ecosystem(_group)
        manifest_set = {
            path.replace("\\", "/").lstrip("/")
            for path in manifest_paths
            if isinstance(path, str) and path.strip()
        }
        manifest_parents = {
            PurePosixPath(path).parent
            for path in manifest_set
            if PurePosixPath(path).name == "package.json"
        }
        task_files: list[str] = []
        for path in normalized_files:
            if path in manifest_set:
                task_files.append(path)
                continue

            # npm synchronization can update the lockfile alongside the
            # manifest. A shared package cluster has multiple task records,
            # so exact manifest matching alone would drop the lockfile from
            # every per-task result. Associate package lockfiles with tasks
            # whose manifest lives in the lockfile's directory or below it.
            path_obj = PurePosixPath(path)
            if (
                ecosystem == "pypi"
                and path_obj.name.casefold() == "pipfile.lock"
                and any(
                    PurePosixPath(manifest).name.casefold() == "pipfile"
                    for manifest in manifest_set
                )
                and any(
                    PurePosixPath(manifest).parent == path_obj.parent
                    for manifest in manifest_set
                    if PurePosixPath(manifest).name.casefold() == "pipfile"
                )
            ):
                task_files.append(path)
                continue
            if path_obj.name not in {"package-lock.json", "npm-shrinkwrap.json"}:
                continue
            if any(
                manifest_parent == path_obj.parent
                or manifest_parent.is_relative_to(path_obj.parent)
                for manifest_parent in manifest_parents
            ):
                task_files.append(path)
        result[task.task_id] = task_files
    return result


def _attempted_versions_for_current_run(
    resolved_tasks: Sequence[tuple[RemediationTask, VulnerabilityGroup, Sequence[str]]],
    tool_events: Sequence[Any],
) -> dict[str, list[str]]:
    """Collect version targets from combined transactions in this worker run."""
    result: dict[str, list[str]] = {task.task_id: [] for task, _, _ in resolved_tasks}
    for event in tool_events:
        if not _is_executed_manifest_transaction(event):
            continue
        ecosystem = (
            "pypi" if getattr(event, "name", "") == _PYTHON_UPDATE_MANIFEST_TOOL_NAME else "npm"
        )
        target = _event_target_version(event, ecosystem)
        if not target:
            continue
        for task_id in _event_matching_task_ids(event, resolved_tasks):
            _append_attempted_version(result[task_id], target, ecosystem)
    return result


def _executed_versions_for_current_run(
    resolved_tasks: Sequence[tuple[RemediationTask, VulnerabilityGroup, Sequence[str]]],
    tool_events: Sequence[Any],
) -> dict[str, list[str]]:
    """Collect only successful version targets from this worker run."""
    result: dict[str, list[str]] = {task.task_id: [] for task, _, _ in resolved_tasks}
    for event in tool_events:
        if not _is_executed_manifest_transaction(event):
            continue
        if not str(getattr(event, "content", "")).startswith("SUCCESS:"):
            continue
        ecosystem = (
            "pypi" if getattr(event, "name", "") == _PYTHON_UPDATE_MANIFEST_TOOL_NAME else "npm"
        )
        target = _event_target_version(event, ecosystem)
        if not target:
            continue
        for task_id in _event_matching_task_ids(event, resolved_tasks):
            _append_attempted_version(result[task_id], target, ecosystem)
    return result


def _attempted_dependency_types_for_current_run(
    resolved_tasks: Sequence[tuple[RemediationTask, VulnerabilityGroup, Sequence[str]]],
    tool_events: Sequence[Any],
) -> dict[str, list[str]]:
    """Collect dependency declaration types from combined transactions."""
    result: dict[str, list[str]] = {task.task_id: [] for task, _, _ in resolved_tasks}
    for event in tool_events:
        if not _is_executed_manifest_transaction(event):
            continue
        args = getattr(event, "args", {}) or {}
        dependency_type = str(args.get("dependency_type", "")).strip()
        if not dependency_type:
            continue
        for task_id in _event_matching_task_ids(event, resolved_tasks):
            if dependency_type not in result[task_id]:
                result[task_id].append(dependency_type)
    return result


def _effective_targets_for_current_run(
    resolved_tasks: Sequence[tuple[RemediationTask, VulnerabilityGroup, Sequence[str]]],
    tool_events: Sequence[Any],
) -> tuple[dict[str, str | None], dict[str, str | None]]:
    """Collect effective version and dependency type from successful transactions."""
    versions: dict[str, str | None] = {task.task_id: None for task, _, _ in resolved_tasks}
    dependency_types: dict[str, str | None] = {task.task_id: None for task, _, _ in resolved_tasks}
    for event in tool_events:
        if not _is_executed_manifest_transaction(event):
            continue
        if not str(getattr(event, "content", "")).startswith("SUCCESS:"):
            continue
        args = getattr(event, "args", {}) or {}
        ecosystem = (
            "pypi" if getattr(event, "name", "") == _PYTHON_UPDATE_MANIFEST_TOOL_NAME else "npm"
        )
        version = _event_target_version(event, ecosystem) or None
        dependency_type = str(args.get("dependency_type", "")).strip() or None
        for task_id in _event_matching_task_ids(event, resolved_tasks):
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
            else (
                select_package_fix_plan(group, task.strategy).plan.fixed_version
                if select_package_fix_plan(group, task.strategy).plan
                else None
            ),
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
        result: dict[str, Any] = {
            "action_summaries": summaries,
            "action_summary": summaries[0] if summaries else None,
            "changed_files": [],
            "errors": resolution_errors,
        }
        if state.get("multi_package_action") is not None and len(target_tasks) > 1:
            target_snapshots = {
                task.task_id: target_attempt_snapshots[task.task_id]
                for task in target_tasks
                if task.task_id in target_attempt_snapshots
            }
            result["worker_results_by_attempt"] = _worker_result_map(
                target_tasks,
                target_snapshots,
                summaries,
                succeeded=False,
                errors=resolution_errors,
            )
        return result

    package_ecosystems = {_group_ecosystem(group) for _, group, _ in resolved_tasks}
    if len(package_ecosystems) != 1:
        message = "Update Subagent: one worker invocation cannot mix package ecosystems."
        summaries = _build_surrender_summaries(all_task_ids, message)
        return {
            "action_summaries": summaries,
            "action_summary": summaries[0] if summaries else None,
            "changed_files": [],
            "errors": resolution_errors + [message],
        }
    package_ecosystem = next(iter(package_ecosystems))
    worker_language = state.get("project_language", ProjectLanguage.NODEJS)
    if not isinstance(worker_language, ProjectLanguage):
        worker_language = ProjectLanguage(worker_language)

    committed_action = state.get("multi_package_action")
    if committed_action is None and _is_mixed_retry_batch(resolved_tasks):
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

    if committed_action is not None:
        action_validation_errors = _validate_committed_multi_package_action(
            state,
            committed_action,
            resolved_tasks,
            target_attempt_snapshots,
        )
        if action_validation_errors:
            summaries = _build_surrender_summaries(all_task_ids, action_validation_errors[0])
            target_snapshots = {
                task.task_id: target_attempt_snapshots[task.task_id]
                for task in target_tasks
                if task.task_id in target_attempt_snapshots
            }
            action_errors = resolution_errors + action_validation_errors
            return {
                "action_summaries": summaries,
                "action_summary": summaries[0] if summaries else None,
                "changed_files": [],
                "worker_results_by_attempt": _worker_result_map(
                    target_tasks,
                    target_snapshots,
                    summaries,
                    succeeded=False,
                    errors=action_errors,
                ),
                "errors": action_errors,
            }
        expected_task_ids = {task.task_id for task, _, _ in resolved_tasks}
        action_task_ids = {mutation.task_id for mutation in committed_action.package_mutations}
        if action_task_ids != expected_task_ids:
            message = (
                "Update Subagent: committed multi-package action does not cover exactly "
                "the resolved cluster tasks."
            )
            summaries = _build_surrender_summaries(all_task_ids, message)
            action_errors = resolution_errors + [message]
            target_snapshots = {
                task.task_id: target_attempt_snapshots[task.task_id]
                for task in target_tasks
                if task.task_id in target_attempt_snapshots
            }
            return {
                "action_summaries": summaries,
                "action_summary": summaries[0] if summaries else None,
                "changed_files": [],
                "worker_results_by_attempt": _worker_result_map(
                    target_tasks,
                    target_snapshots,
                    summaries,
                    succeeded=False,
                    errors=action_errors,
                ),
                "errors": action_errors,
            }
        resolved_task_ids = [task.task_id for task, _, _ in resolved_tasks]
        target_snapshots = {
            task.task_id: target_attempt_snapshots[task.task_id]
            for task, _, _ in resolved_tasks
            if task.task_id in target_attempt_snapshots
        }
        if ChatOpenAI is None:
            msg = "Update Subagent: 'langchain-openai' is not installed."
            summaries = _build_surrender_summaries(
                all_task_ids, "Stopped before execution because the LLM client is unavailable."
            )
            action_errors = resolution_errors + [msg]
            return {
                "action_summaries": summaries,
                "action_summary": summaries[0] if summaries else None,
                "changed_files": [],
                "worker_results_by_attempt": _worker_result_map(
                    target_tasks,
                    {
                        task.task_id: target_attempt_snapshots[task.task_id]
                        for task in target_tasks
                        if task.task_id in target_attempt_snapshots
                    },
                    summaries,
                    succeeded=False,
                    errors=action_errors,
                ),
                "errors": action_errors,
            }

        settings = get_runtime_settings()
        model_name = settings.update_llm_model
        try:
            llm = ChatOpenAI(
                model=model_name,
                temperature=0,
                **openai_api_key_kwargs(settings),
            )
        except Exception as exc:  # noqa: BLE001
            msg = f"Update Subagent: failed to initialize LLM - {exc}."
            summaries = _build_surrender_summaries(
                all_task_ids, "Stopped before execution because the LLM failed to initialize."
            )
            action_errors = resolution_errors + [msg]
            return {
                "action_summaries": summaries,
                "action_summary": summaries[0] if summaries else None,
                "changed_files": [],
                "worker_results_by_attempt": _worker_result_map(
                    target_tasks,
                    {
                        task.task_id: target_attempt_snapshots[task.task_id]
                        for task in target_tasks
                        if task.task_id in target_attempt_snapshots
                    },
                    summaries,
                    succeeded=False,
                    errors=action_errors,
                ),
                "errors": action_errors,
            }

        touched_files: set[str] = set()
        package_checkpoints: dict[str, Any] = {}
        execution_state: dict[str, Any] = {
            "edits_started": False,
            "validation_calls": 0,
            "manifest_transaction_attempts": 0,
            "multi_package_action_executed": False,
            "multi_package_action_succeeded": False,
        }
        skinny_resolved_tasks = [
            (task, _create_skinny_subagent_group(group), manifest_paths)
            for task, group, manifest_paths in resolved_tasks
        ]
        filtered_ledger = _filter_constraints_ledger(constraints_ledger, target_groups)
        cluster_prompt = _build_multi_package_update_prompt(
            skinny_resolved_tasks,
            committed_action,
            filtered_ledger,
            feedback_by_task,
            previous_action_summaries_by_task,
            prior_retry_diagnostics_by_task,
            repository_map=build_repository_map(repo_root),
            allowed_target_versions_by_task=allowed_target_versions_by_task,
            allowed_dependency_types_by_task=allowed_dependency_types_by_task,
            project_language=worker_language,
        )
        multi_static_instructions = (
            _MULTI_PACKAGE_PYTHON_UPDATE_WORKER_STATIC_INSTRUCTIONS
            if worker_language == ProjectLanguage.PYTHON
            else _MULTI_PACKAGE_UPDATE_WORKER_STATIC_INSTRUCTIONS
        )
        initial_messages = [SystemMessage(content=multi_static_instructions)]
        if state.get("messages"):
            initial_messages.extend(state["messages"])
        initial_messages.append(HumanMessage(content=cluster_prompt))

        runtime = None
        cleanup_errors: list[str] = []
        sandbox = None
        try:
            with DockerSandbox(
                repo_root=None,
                image=LANGUAGE_CONFIGS[worker_language].docker_image,
                workspace_volume=workspace_volume,
            ) as sandbox:
                toolbelt = build_multi_package_update_toolbelt(
                    sandbox,
                    touched_files,
                    committed_action,
                    execution_state=execution_state,
                    package_checkpoints=package_checkpoints,
                    project_language=worker_language,
                )
                runtime = run_bounded_subagent_loop(
                    llm,
                    toolbelt,
                    initial_messages,
                    touched_files,
                    execution_state=execution_state,
                )
                action_tool_name = (
                    _BATCH_PYTHON_DEPENDENCIES_TOOL_NAME
                    if worker_language == ProjectLanguage.PYTHON
                    else _BATCH_NPM_DEPENDENCIES_TOOL_NAME
                )
                action_tool_events = [
                    event for event in runtime.tool_events if event.name == action_tool_name
                ]
                action_executed = bool(
                    execution_state.get("multi_package_action_executed") or action_tool_events
                )
                action_succeeded = bool(
                    execution_state.get("multi_package_action_succeeded")
                    or any(event.content.startswith("SUCCESS:") for event in action_tool_events)
                )
                succeeded = (
                    action_executed
                    and action_succeeded
                    and not (resolution_errors or runtime.errors)
                )
                if succeeded:
                    package_checkpoints.clear()
                else:
                    if not action_executed:
                        execution_state["multi_package_action_error"] = (
                            "LLM did not invoke the committed multi-package action tool."
                        )
                    cleanup_errors.extend(
                        rollback_pending_package_updates(
                            sandbox,
                            package_checkpoints,
                            touched_files,
                        )
                    )
        except Exception as exc:  # noqa: BLE001 - worker boundary returns typed surrender
            succeeded = False
            if sandbox is not None and package_checkpoints:
                cleanup_errors.extend(
                    rollback_pending_package_updates(
                        sandbox,
                        package_checkpoints,
                        touched_files,
                    )
                )
            cleanup_errors.append(f"Multi-package sandbox or tool loop failed: {exc}")

        runtime_errors = list(resolution_errors)
        if runtime is not None:
            runtime_errors.extend(runtime.errors)
        runtime_errors.extend(cleanup_errors)
        action_error = str(execution_state.get("multi_package_action_error") or "").strip()
        if not succeeded and action_error and action_error not in runtime_errors:
            runtime_errors.append(action_error)
        if not succeeded and not runtime_errors:
            runtime_errors.append(
                "Update Subagent: committed multi-package action did not complete successfully."
            )

        summary_status = AgentActionStatus.SUCCESS if succeeded else AgentActionStatus.SURRENDER
        summary_text = (
            "Supervisor-committed multi-package action completed successfully."
            if succeeded
            else runtime_errors[-1]
        )
        summaries = [
            AgentActionSummary(
                task_id=task.task_id,
                status=summary_status,
                summary=summary_text,
            )
            for task, _, _ in resolved_tasks
        ]
        attempted_by_task = (
            {
                task.task_id: [mutation.target_version]
                for task, _, _ in resolved_tasks
                for mutation in committed_action.package_mutations
                if mutation.task_id == task.task_id
            }
            if execution_state.get("multi_package_action_executed")
            else {}
        )
        executed_by_task = attempted_by_task if succeeded else {}
        changed_by_task = _changed_files_by_task(resolved_tasks, sorted(touched_files))
        tagged_summaries = [
            summary.model_copy(
                update={
                    "attempt_id": target_snapshots[summary.task_id].attempt_id,
                    "task_revision": target_snapshots[summary.task_id].task_revision,
                    "instruction_digest": target_snapshots[summary.task_id].instruction_digest,
                }
            )
            if summary.task_id in target_snapshots
            else summary
            for summary in summaries
        ]
        manifest_count = len(
            {mutation.manifest_path for mutation in committed_action.package_mutations}
        )
        return {
            "action_summaries": tagged_summaries,
            "action_summary": tagged_summaries[0] if tagged_summaries else None,
            "changed_files": sorted(touched_files) if succeeded else [],
            "worker_results_by_attempt": _worker_result_map(
                [task for task, _, _ in resolved_tasks],
                target_snapshots,
                tagged_summaries,
                succeeded=succeeded,
                errors=runtime_errors,
                attempted_versions_by_task=attempted_by_task,
                executed_versions_by_task=executed_by_task,
                changed_files_by_task=changed_by_task if succeeded else {},
                validation_calls=manifest_count
                if execution_state.get("multi_package_action_executed")
                else 0,
                manifest_transaction_attempts=manifest_count
                if execution_state.get("multi_package_action_executed")
                else 0,
            ),
            "errors": runtime_errors,
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

    settings = get_runtime_settings()
    model_name = settings.update_llm_model
    try:
        llm = ChatOpenAI(
            model=model_name,
            temperature=0,
            **openai_api_key_kwargs(settings),
        )
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
        project_language=worker_language,
    )
    initial_messages = [SystemMessage(content=_build_update_system_prompt(worker_language))]
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
            image=LANGUAGE_CONFIGS[worker_language].docker_image,
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
                language=worker_language,
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

    successful_task_ids = {
        task.task_id
        for task, group, manifest_paths in resolved_tasks
        if _has_successful_manifest_transaction_for_package(
            task, group, runtime.tool_events, manifest_paths, resolved_tasks
        )
    }
    unvalidated_manifest_paths = {
        path.replace("\\", "/")
        for task, group, manifest_paths in resolved_tasks
        if task.task_id not in successful_task_ids
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
    succeeded = bool(resolved_tasks) and all(
        task.task_id in successful_task_ids for task, _group, _manifest_paths in resolved_tasks
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
