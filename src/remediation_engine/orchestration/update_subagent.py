"""Dependency Update Subagent for Phase 5 dependency resolution.

The Supervisor normally dispatches one task per invocation, while this module
retains batch-capable helpers for direct callers and future explicit batch mode.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable

from remediation_engine.contracts.schemas import (
    AgentActionStatus,
    AgentActionSummary,
    RemediationTask,
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
)
from remediation_engine.runtime.path_policy import (
    WorkspacePathError,
    resolve_repository_path,
)
from remediation_engine.runtime.sandbox_mgr import DockerSandbox

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "gpt-4o-mini"
_REGISTRY_VERSION_LINE_RE = re.compile(r"^\s+(\d+\.[0-9A-Za-z.+-]+)")
_REGISTRY_MAJOR_LINE_RE = re.compile(r"^\s+v\d+\.x\s+â†’\s+([^\s]+)")
_REGISTRY_LATEST_TAG_RE = re.compile(r"^\s*latest:\s*([^\s]+)")

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


def _format_manifest_paths(manifest_paths: Sequence[str]) -> str:
    if not manifest_paths:
        return "none"
    if len(manifest_paths) == 1:
        return manifest_paths[0]
    return "\n".join(f"- {path}" for path in manifest_paths)


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
    return "overrides" if task.strategy_stage == SCARemediationStage.PACKAGE_OVERRIDE else None


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


def _legacy_build_update_prompt(
    resolved_tasks: Sequence[tuple[RemediationTask, VulnerabilityGroup, Sequence[str]]],
    constraints_ledger: Sequence[str],
    feedback_by_task: dict[str, str],
    previous_action_summaries_by_task: dict[str, str],
    retry_diagnostics_by_task: dict[str, UpdateRetryDiagnostics] | None = None,
) -> str:
    retry_diagnostics_by_task = dict(retry_diagnostics_by_task or {})
    retry_batch = _is_retry_batch(resolved_tasks)
    constraints_text = (
        "\n".join(f"- {item}" for item in constraints_ledger) if constraints_ledger else "- none"
    )
    sections = [
        "\n".join(
            [
                "You are a dependency-resolution specialist operating inside a shared Docker workspace.",
                "You may only modify package manifests through modify_npm_dependency.",
                "You must inspect the repository map before making manifest changes.",
                "Emit at most one modify_npm_dependency or validate_manifest_sync tool call per assistant turn; tool calls in one message cannot observe each other's results.",
                "After completing the exact manifest edits for each package, call validate_manifest_sync for that package before editing the next package.",
                "If validate_manifest_sync fails, you must resolve the peer conflict or invalid manifest state before finishing.",
                "Every modify_npm_dependency call must include the exact manifest_path you intend to edit.",
                "If the vulnerable component appears in multiple manifest paths, call modify_npm_dependency once for each manifest_path that declares it.",
                "Never downgrade a package that is constrained by the constraints ledger.",
                "Use revert_workspace_file if you reach a structurally bad manifest state. For package.json files, specify the package_name parameter to revert only that specific dependency.",
                "Do not search the codebase and do not edit source code files.",
                "For each target below, the Task Instruction is the authoritative directive.",
            ]
        )
    ]

    if retry_batch:
        sections.append(
            "\n".join(
                [
                    "You are an autonomous Dependency Resolution Specialist.",
                    "",
                    "Your previous attempt to mitigate vulnerabilities in this package FAILED during the QA Evaluation phase.",
                    'You are now in the "Smart Planning & Rescue" phase.',
                    "",
                    "For each target below, the Task Instruction remains authoritative.",
                ]
            )
        )
        sections.append(
            "\n".join(
                [
                    "## Standard Operating Procedure (SOP)",
                    "You must act autonomously to find a working solution. Follow these steps strictly:",
                    "",
                    "1. **Reconnaissance:** You MUST use the `view_npm_package_versions` tool on the target component to see the actual, historical versions published to the NPM registry. Do NOT hallucinate version numbers.",
                    "2. **Analysis & Selection:** Evaluate the registry versions against the QA feedback and prior retry diagnostics.",
                    "   - **LATEST-FIRST RULE:** After viewing the NPM registry, you MUST attempt the LATEST version found on the npm registry (`latest` dist-tag or newest published version) instead of selecting the next newest version from the attempted version.",
                    "   - *If PEER_CONFLICT (ERESOLVE):* Only if the latest version was already attempted and caused a peer conflict should you select an untried compatible backported security patch or lower compatible version.",
                    "   - *If SECURITY_FLAG:* Attempt the latest version from the registry unless it was already attempted.",
                    "   - **CRITICAL CEILING CHECK:** If the registry shows you are ALREADY on the absolute latest version and that version was already attempted, do NOT re-apply it. Re-applying the same version will fail QA again.",
                    "3. **Execution:** Use `modify_npm_dependency` to apply the selected version or override. For transitive dependencies, NPM `overrides` or `resolutions` in the root `package.json` may be required.",
                    "4. **Validation:** After completing all required manifest edits for one package, you MUST call `validate_manifest_sync` with that package's exact package_name before moving to the next package.",
                    "5. **Iteration:** If `validate_manifest_sync` fails with an NPM error, do not surrender immediately. Review the error, select a different candidate from the registry, and try again.",
                    "6. **Clean Room Surrender & Exhaustion:**",
                    "   - In a multi-target batch, if one package cannot be resolved due to a peer conflict, you MUST call `revert_workspace_file(file_path, package_name='failing_package')` ONLY for that specific failing package. Preserve and validate the manifest updates for all other successful packages in the batch.",
                    "   - If the latest version on the NPM registry was already attempted and still failed QA due to an unresolved vulnerability (`SECURITY_FLAG`), do NOT re-attempt it or try older versions. Immediately perform a Clean Room Surrender for that package and explicitly state 'update path is exhausted' in your Reasoning Summary.",
                    "",
                    "Do not edit source code files. Your domain is strictly package manifests. Return control to the Supervisor only when every package has a successful per-package `validate_manifest_sync` result, or you have executed a Clean Room Surrender.",
                ]
            )
        )
        sections.append(
            "\n".join(
                [
                    "## Shared Planning Questions (for reference)",
                    "1. What version or override path did the last fix attempt use?",
                    "2. What attempted_versions are already recorded for this task?",
                    "3. What is the latest version currently available according to npm?",
                    "4. Which candidate versions from npm have not been attempted yet?",
                    "5. Is the vulnerable package a direct dependency or does it need npm overrides?",
                    "6. Do the prior QA feedback and previous worker outcome suggest a peer conflict, stale version, or wrong manifest target?",
                    "7. Does the constraints ledger forbid any downgrade or conflicting change?",
                    "8. What version should be attempted next? (You MUST prioritize attempting the latest version found on the npm registry rather than stepping incrementally from the attempted version)",
                    "9. Was the previous manifest update structurally validated, and if so, did QA or scanner findings still remain afterward?",
                    "10. If the latest available version on the NPM registry was already attempted (unless resolving a peer conflict), the update path is exhausted. Explicitly state 'update path is exhausted' in your Reasoning Summary.",
                ]
            )
        )
    else:
        sections.append(
            "\n".join(
                [
                    "First-pass mode:",
                    "- This is an initial execution batch.",
                    "- Execute the Task Instruction exactly as written.",
                    "- Do not use view_npm_package_versions or invent alternate versions.",
                    "- If the exact requested manifest remediation cannot validate, revert and surrender.",
                ]
            )
        )

    sections.append("Constraints ledger:\n" + constraints_text)

    for task, group, manifest_paths in resolved_tasks:
        feedback = feedback_by_task.get(task.task_id, "none")
        previous_outcome = previous_action_summaries_by_task.get(task.task_id, "none")
        diagnostics = retry_diagnostics_by_task.get(task.task_id)
        diagnostics_text = "none"
        if diagnostics is not None:
            attempted = ", ".join(diagnostics.attempted_versions) or "none"
            candidates = ", ".join(diagnostics.candidate_versions_considered[:8]) or "none"
            latest_seen = diagnostics.latest_version_seen or "unknown"
            diagnostics_text = (
                f"attempted={attempted}; candidates={candidates}; "
                f"latest_seen={latest_seen}; exhausted={diagnostics.exhausted_update_path}"
            )
        if retry_batch:
            attempted_set = set(diagnostics.attempted_versions) if diagnostics else set()
            latest_seen = diagnostics.latest_version_seen if diagnostics else None
            candidate_choices = [
                version
                for version in (diagnostics.candidate_versions_considered if diagnostics else [])
                if version not in attempted_set
            ]
            if latest_seen and latest_seen in attempted_set:
                next_version_hint = (
                    f"NONE (latest version '{latest_seen}' already attempted â€” surrender required)"
                    if not candidate_choices
                    else candidate_choices[0]
                )
            elif latest_seen:
                next_version_hint = latest_seen
            elif candidate_choices:
                next_version_hint = candidate_choices[0]
            else:
                next_version_hint = "see registry candidates"

            retry_answers = [
                "1. What version or override path did the last fix attempt use?",
                f"2. What attempted_versions are already recorded for this task? {diagnostics.attempted_versions if diagnostics else 'none'}",
                f"3. What is the latest version currently available according to npm? {diagnostics.latest_version_seen if diagnostics and diagnostics.latest_version_seen else 'unknown'}",
                f"4. Which candidate versions from npm have not been attempted yet? {', '.join(candidate_choices) or 'none'}",
                f"5. Is the vulnerable package a direct dependency or does it need npm overrides? {'npm overrides' if diagnostics and diagnostics.used_overrides else 'direct dependency version bump'}",
                f"6. Do the prior QA feedback and previous worker outcome suggest a peer conflict, stale version, or wrong manifest target? {feedback if feedback != 'none' else previous_outcome}",
                f"7. Does the constraints ledger forbid any downgrade or conflicting change? {'yes' if constraints_ledger else 'no blocking constraints recorded'}",
                f"8. What version should be attempted next? (You MUST prioritize attempting the latest version found on the npm registry rather than stepping incrementally from the attempted version) {next_version_hint}",
                f"9. Was the previous manifest update structurally validated, and if so, did QA or scanner findings still remain afterward? {previous_outcome}",
                "10. If the latest available version was already attempted and no untried candidates remain, the update path is exhausted.",
            ]
            sections.append(
                "\n".join(
                    [
                        "## Task Context",
                        f"- Task ID: {task.task_id}",
                        f"- Component: {group.vulnerable_component or 'unknown'}",
                        f"- Manifest Path: {_format_manifest_paths(manifest_paths)}",
                        f"- Supervisor's Revised Instruction: {task.instruction or 'Derive the safest manifest update.'}",
                        "",
                        "## Why The Previous Attempt Failed",
                        f"- QA Feedback: {feedback}",
                        f"- Previous Worker Outcome: {previous_outcome}",
                        "",
                        "## Prior Retry Diagnostics",
                        diagnostics_text,
                        "",
                        f"## Planning Answers (for Task {task.task_id})",
                        "Provide a short visible answer for each planning question before you make any manifest edit.",
                        "\n".join(retry_answers),
                    ]
                )
            )
        else:
            sections.append(
                "\n".join(
                    [
                        "## Task Context",
                        f"- Task ID: {task.task_id}",
                        f"- Component: {group.vulnerable_component or 'unknown'}",
                        f"- Manifest Path: {_format_manifest_paths(manifest_paths)}",
                        f"- Supervisor's Revised Instruction: {task.instruction or 'Derive the safest manifest update.'}",
                    ]
                )
            )

    sections.append(
        "\n".join(
            [
                "Completion rule:",
                "Return control only after manifest synchronization succeeds and you have completed the required manifest updates.",
            ]
        )
    )
    return "\n\n".join(sections)


def _was_package_reverted(
    task: RemediationTask,
    group: VulnerabilityGroup,
    tool_events: Sequence[Any] | None,
) -> bool:
    if not tool_events:
        return False
    pkg = _target_package_name(task, group)
    reverted = False
    for event in tool_events:
        name = getattr(event, "name", "")
        args = getattr(event, "args", {}) or {}
        content = getattr(event, "content", "")
        if name == "modify_npm_dependency":
            if args.get("package_name") == pkg and content.startswith("SUCCESS:"):
                reverted = False
        elif name == "revert_workspace_file" and content.startswith("SUCCESS:"):
            revert_pkg = args.get("package_name")
            if revert_pkg == pkg or not revert_pkg:
                reverted = True
    return reverted


def _was_package_modified(
    task: RemediationTask,
    group: VulnerabilityGroup,
    tool_events: Sequence[Any] | None,
) -> bool:
    if not tool_events:
        return False
    pkg = _target_package_name(task, group)
    modified = False
    for event in tool_events:
        name = getattr(event, "name", "")
        args = getattr(event, "args", {}) or {}
        content = str(getattr(event, "content", ""))
        if name == "modify_npm_dependency" and args.get("package_name") == pkg:
            modified = content.startswith("SUCCESS:") or content == "SUCCESS"
        elif name == "validate_manifest_sync" and args.get("package_name") == pkg:
            if content.startswith("FAILURE:") or content.startswith("ERROR:"):
                modified = False
        elif name == "revert_workspace_file" and (
            content.startswith("SUCCESS:") or content == "SUCCESS"
        ):
            revert_pkg = args.get("package_name")
            if revert_pkg == pkg or not revert_pkg:
                modified = False
    return modified


def _has_successful_validation_for_package(
    task: RemediationTask,
    group: VulnerabilityGroup,
    tool_events: Sequence[Any] | None,
) -> bool:
    """Return whether a package's final manifest state was validated successfully."""
    if not tool_events:
        return False

    pkg = _target_package_name(task, group)
    last_successful_edit_index = -1

    for index, event in enumerate(tool_events):
        name = getattr(event, "name", "")
        args = getattr(event, "args", {}) or {}
        content = str(getattr(event, "content", ""))
        if (
            name == "modify_npm_dependency"
            and args.get("package_name") == pkg
            and (content.startswith("SUCCESS:") or content == "SUCCESS")
        ):
            last_successful_edit_index = index

    if last_successful_edit_index < 0:
        return False

    for event in tool_events[last_successful_edit_index + 1 :]:
        name = getattr(event, "name", "")
        args = getattr(event, "args", {}) or {}
        content = str(getattr(event, "content", ""))
        if (
            name == "validate_manifest_sync"
            and (not args.get("package_name") or args.get("package_name") == pkg)
            and (content.startswith("SUCCESS:") or content == "SUCCESS")
        ):
            return True

    return False


def _had_registry_lookup_before_package_edit(
    task: RemediationTask,
    group: VulnerabilityGroup,
    tool_events: Sequence[Any] | None,
) -> bool:
    """Return whether retry-mode registry lookup happened before this package's first edit."""
    if not tool_events:
        return False

    pkg = _target_package_name(task, group)
    first_successful_edit_index = -1

    for index, event in enumerate(tool_events):
        name = getattr(event, "name", "")
        args = getattr(event, "args", {}) or {}
        content = str(getattr(event, "content", ""))
        if (
            name == "modify_npm_dependency"
            and args.get("package_name") == pkg
            and (content.startswith("SUCCESS:") or content == "SUCCESS")
        ):
            first_successful_edit_index = index
            break

    if first_successful_edit_index < 0:
        return False

    for event in tool_events[:first_successful_edit_index]:
        name = getattr(event, "name", "")
        args = getattr(event, "args", {}) or {}
        if name == "view_npm_package_versions" and args.get("package_name") == pkg:
            return True

    return False


def _legacy_build_action_summaries(
    resolved_tasks: Sequence[tuple[RemediationTask, VulnerabilityGroup, Sequence[str]]],
    changed_files: Sequence[str],
    final_text: str,
    succeeded: bool,
    retry_batch: bool = False,
    tool_events: Sequence[Any] | None = None,
) -> list[AgentActionSummary]:
    final_note = final_text.strip()
    normalized_changed_files = {path.replace("\\", "/") for path in changed_files}
    include_final_note = bool(final_note) and len(resolved_tasks) == 1
    summaries: list[AgentActionSummary] = []

    for task, group, manifest_paths in resolved_tasks:
        component = (group.vulnerable_component or "unknown component").strip()
        manifest_label = ", ".join(manifest_paths) if manifest_paths else "no manifests"
        relevant_changed_files = [
            path for path in manifest_paths if path.replace("\\", "/") in normalized_changed_files
        ]
        pkg_reverted = _was_package_reverted(task, group, tool_events)
        pkg_modified = (
            _was_package_modified(task, group, tool_events)
            if tool_events is not None
            else (bool(relevant_changed_files) and not pkg_reverted)
        )
        pkg_validated = (
            _has_successful_validation_for_package(task, group, tool_events)
            if tool_events is not None
            else succeeded
        )
        # Registry lookup and version selection are supervisor responsibilities.
        # A retry is successful when the exact committed edit was applied and
        # the final manifest validation passed; requiring a worker-side lookup
        # here would make every retry fail despite the execution-only contract.
        task_succeeded = pkg_modified and pkg_validated
        summary_status = (
            AgentActionStatus.SUCCESS if task_succeeded else AgentActionStatus.SURRENDER
        )
        outcome = (
            "Completed validated manifest updates"
            if task_succeeded
            else "Stopped without a validated manifest update (reverted/surrendered)"
        )

        changed_label = ", ".join(
            relevant_changed_files or (manifest_paths if pkg_modified else ["no files"])
        )
        summary_text = (
            f"{outcome} for {component} in {manifest_label}; changed files: {changed_label}."
        )
        if include_final_note:
            summary_text = f"{summary_text} Final note: {final_note}"
        summaries.append(
            AgentActionSummary(
                task_id=task.task_id,
                status=summary_status,
                summary=summary_text,
            )
        )

    return summaries


def _build_surrender_summaries(task_ids: Sequence[str], message: str) -> list[AgentActionSummary]:
    return [
        AgentActionSummary(task_id=t_id, status=AgentActionStatus.SURRENDER, summary=message)
        for t_id in task_ids
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
    validation_calls: int = 0,
) -> dict[str, WorkerAttemptResult]:
    """Build attempt-correlated worker envelopes without changing task state."""
    summary_by_task = {summary.task_id: summary for summary in summaries}
    results: dict[str, WorkerAttemptResult] = {}
    attempted_versions_by_task = attempted_versions_by_task or {}
    executed_versions_by_task = executed_versions_by_task or attempted_versions_by_task
    for task in target_tasks:
        snapshot = snapshots.get(task.task_id)
        if snapshot is None:
            continue
        summary = summary_by_task.get(task.task_id)
        attempted = list(attempted_versions_by_task.get(task.task_id, []))
        executed = list(executed_versions_by_task.get(task.task_id, []))
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
                validation_calls=validation_calls,
                validation_passed=succeeded,
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
    """Collect version targets from this invocation only.

    ``UpdateRetryDiagnostics`` is intentionally cumulative for compatibility,
    but a ``WorkerAttemptResult`` must describe only the attempt that produced
    it.  In particular, carrying the previous attempt's target here would make
    the supervisor interpret a valid retry as having executed multiple versions.
    Tool events include failed edit calls, so this also preserves attempted
    targets when the edit itself did not succeed.
    """
    package_to_task_ids: dict[str, list[str]] = {}
    for task, group, _manifest_paths in resolved_tasks:
        package = _target_package_name(task, group)
        if package:
            package_to_task_ids.setdefault(package, []).append(task.task_id)

    result: dict[str, list[str]] = {task.task_id: [] for task, _group, _paths in resolved_tasks}
    for event in tool_events:
        if getattr(event, "name", "") != "modify_npm_dependency":
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
    """Collect only successful exact edit targets from this worker run."""
    result: dict[str, list[str]] = {task.task_id: [] for task, _group, _paths in resolved_tasks}
    package_to_task_ids: dict[str, list[str]] = {}
    for task, group, _manifest_paths in resolved_tasks:
        package = _target_package_name(task, group)
        if package:
            package_to_task_ids.setdefault(package, []).append(task.task_id)
    for event in tool_events:
        if getattr(event, "name", "") != "modify_npm_dependency":
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


def _parse_registry_report_versions(report_text: str) -> tuple[Sequence[str], str | None]:
    """Extract candidate versions and the dist-tag latest version from a registry report."""
    candidates: list[str] = []
    seen: set[str] = set()
    latest_version: str | None = None

    for raw_line in report_text.splitlines():
        line = raw_line.rstrip()
        latest_match = _REGISTRY_LATEST_TAG_RE.match(line)
        if latest_match:
            latest_version = latest_match.group(1).strip()
            if latest_version and latest_version not in seen:
                seen.add(latest_version)
                candidates.append(latest_version)
            continue

        version_match = _REGISTRY_VERSION_LINE_RE.match(line)
        if version_match:
            version = version_match.group(1).strip()
            if version and version not in seen:
                seen.add(version)
                candidates.append(version)
            continue

        major_match = _REGISTRY_MAJOR_LINE_RE.match(line)
        if major_match:
            version = major_match.group(1).strip()
            if version and version not in seen:
                seen.add(version)
                candidates.append(version)

    return candidates, latest_version


def _merge_ordered_versions(existing: Sequence[str], new_values: Sequence[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for value in [*existing, *new_values]:
        normalized = value.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        merged.append(normalized)
    return merged


def _extract_reasoning_summary(final_text: str) -> str:
    """Extract a concise visible reasoning summary from the agent final text."""
    cleaned = (final_text or "").strip()
    if not cleaned:
        return ""

    lowered = cleaned.lower()
    for marker in ("reasoning summary", "reasoning", "planning summary"):
        marker_index = lowered.find(marker)
        if marker_index >= 0:
            return cleaned[marker_index:].strip()
    return cleaned


def _legacy_build_retry_diagnostics(
    resolved_tasks: Sequence[tuple[RemediationTask, VulnerabilityGroup, Sequence[str]]],
    tool_events: Sequence[Any],
    final_text: str,
    errors: Sequence[str],
    succeeded: bool,
    prior_diagnostics_by_task: dict[str, UpdateRetryDiagnostics],
    constraints_ledger: Sequence[str],
) -> dict[str, UpdateRetryDiagnostics]:
    """Build per-task retry diagnostics from tool events and prior evidence."""
    diagnostics_by_task: dict[str, UpdateRetryDiagnostics] = {}
    lowered_final_text = final_text.lower()
    joined_errors = " | ".join(error.strip() for error in errors if error.strip())

    for task, group, _manifest_paths in resolved_tasks:
        package_name = (group.vulnerable_component or "").strip()
        prior = prior_diagnostics_by_task.get(task.task_id)
        attempted_versions = list(prior.attempted_versions) if prior else []
        candidate_versions = list(prior.candidate_versions_considered) if prior else []
        registry_query_performed = bool(prior.registry_query_performed) if prior else False
        latest_version_seen = prior.latest_version_seen if prior else None
        used_overrides = bool(prior.used_overrides) if prior else False
        package_abandoned = bool(prior.package_abandoned) if prior else False

        for event in tool_events:
            event_package = str(event.args.get("package_name", "") or "").strip()
            if event.name == "view_npm_package_versions" and event_package == package_name:
                registry_query_performed = True
                parsed_candidates, parsed_latest = _parse_registry_report_versions(event.content)
                candidate_versions = _merge_ordered_versions(
                    candidate_versions,
                    parsed_candidates,
                )
                if parsed_latest:
                    latest_version_seen = parsed_latest
                if event.content.startswith("PACKAGE NOT FOUND:"):
                    package_abandoned = True

            if event.name == "modify_npm_dependency" and event_package == package_name:
                target_version = str(event.args.get("target_version", "") or "").strip()
                if target_version:
                    attempted_versions = _merge_ordered_versions(
                        attempted_versions,
                        [target_version],
                    )
                if str(event.args.get("dependency_type", "") or "").strip() == "overrides":
                    used_overrides = True

        if "package abandoned" in lowered_final_text:
            package_abandoned = True

        selected_version = None
        if succeeded and attempted_versions:
            selected_version = attempted_versions[-1]
        elif prior and prior.selected_version:
            selected_version = prior.selected_version

        attempted_set = set(attempted_versions)
        is_peer_conflict = "peer" in (joined_errors + " " + lowered_final_text) or "eresolve" in (
            joined_errors + " " + lowered_final_text
        )
        reasoning_summary = _extract_reasoning_summary(final_text)
        if not reasoning_summary and prior:
            reasoning_summary = prior.reasoning_summary

        exhausted_update_path = bool(prior.exhausted_update_path) if prior else False
        reasoned_exhausted = any(
            phrase in (lowered_final_text + " " + reasoning_summary.lower())
            for phrase in [
                "exhausted",
                "update path is exhausted",
                "no further version",
                "no other version",
                "no valid candidate",
                "already attempted the latest",
                "already attempted latest",
                "latest version has been attempted",
                "latest version was already attempted",
            ]
        )
        reverted_on_retry = _is_retry_task(task) and _was_package_reverted(task, group, tool_events)
        latest_attempted_on_retry = (
            _is_retry_task(task)
            and latest_version_seen is not None
            and latest_version_seen in attempted_set
        )
        if (
            package_abandoned
            or reasoned_exhausted
            or (reverted_on_retry or latest_attempted_on_retry)
            and not is_peer_conflict
        ):
            exhausted_update_path = True

        failure_reason = ""
        if not succeeded:
            failure_reason = joined_errors or final_text.strip()
        elif prior and prior.failure_reason and not selected_version:
            failure_reason = prior.failure_reason

        diagnostics_by_task[task.task_id] = UpdateRetryDiagnostics(
            task_id=task.task_id,
            registry_query_performed=registry_query_performed,
            attempted_versions=attempted_versions,
            candidate_versions_considered=candidate_versions,
            selected_version=selected_version,
            latest_version_seen=latest_version_seen,
            used_overrides=used_overrides,
            package_abandoned=package_abandoned,
            exhausted_update_path=exhausted_update_path,
            failure_reason=failure_reason,
            reasoning_summary=reasoning_summary,
        )

    return diagnostics_by_task


# ---------------------------------------------------------------------------
# Execution-only worker overrides
# ---------------------------------------------------------------------------


def _build_update_prompt(
    resolved_tasks: Sequence[tuple[RemediationTask, VulnerabilityGroup, Sequence[str]]],
    constraints_ledger: Sequence[str],
    feedback_by_task: dict[str, str],
    previous_action_summaries_by_task: dict[str, str],
    retry_diagnostics_by_task: dict[str, UpdateRetryDiagnostics] | None = None,
) -> str:
    """Build an execution-only prompt; strategy selection belongs to Supervisor."""
    sections = [
        "You are a dependency-manifest execution worker.",
        "The Supervisor's task instruction is authoritative. Execute it exactly.",
        "Do not search the NPM registry, choose alternate versions, or perform retry planning.",
        "Use only read_repository_map, modify_npm_dependency, revert_workspace_file, and validate_manifest_sync.",
        "Inspect the repository map before editing.",
        "Emit at most one modify_npm_dependency or validate_manifest_sync tool call per assistant turn; tool calls in one message cannot observe each other's results.",
        "For each package, apply all of its exact task instructions, then call validate_manifest_sync with that package_name before moving to the next package.",
        "The validator rolls back only the current package to its pre-update checkpoint when synchronization fails; preserve earlier packages whose validations succeeded.",
        "If a package validation fails, do not choose another version or retry inside this worker; surrender to the Supervisor.",
        "Never edit source-code files in this worker.",
        "During parent stages (OSV_MINIMUM, NPM_SAME_MAJOR, NPM_LATEST), edit only the committed parent target and never add an override.",
        "During PACKAGE_OVERRIDE, edit only the committed vulnerable child through the package manager's native override mechanism.",
        "",
        "Constraints ledger:",
    ]
    sections.extend(f"- {item}" for item in constraints_ledger)
    if not constraints_ledger:
        sections.append("- none")
    for task, group, manifest_paths in resolved_tasks:
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
                f"- Exact supervisor instruction: {task.instruction or '(missing)'}",
                f"- QA feedback: {feedback_by_task.get(task.task_id, 'none')}",
                f"- Previous outcome: {previous_action_summaries_by_task.get(task.task_id, 'none')}",
            ]
        )
    sections.extend(
        [
            "",
            "Completion rule:",
            "Return control only after every package has one successful per-package manifest synchronization call, or after surrendering on the first package validation failure.",
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
        package_modified = _was_package_modified(task, group, tool_events)
        package_validated = _has_successful_validation_for_package(task, group, tool_events)
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


def _build_retry_diagnostics(
    resolved_tasks: Sequence[tuple[RemediationTask, VulnerabilityGroup, Sequence[str]]],
    tool_events: Sequence[Any],
    final_text: str,
    errors: Sequence[str],
    succeeded: bool,
    prior_diagnostics_by_task: dict[str, UpdateRetryDiagnostics],
    constraints_ledger: Sequence[str],
) -> dict[str, UpdateRetryDiagnostics]:
    """Record worker evidence while keeping strategy planning in Supervisor."""
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
        used_overrides = bool(prior.used_overrides) if prior else False
        for event in tool_events:
            if event.name != "modify_npm_dependency":
                continue
            if str(event.args.get("package_name", "")).strip() != target_package:
                continue
            target = str(event.args.get("target_version", "")).strip().lstrip("vV")
            if target and target not in attempted:
                attempted.append(target)
            used_overrides = used_overrides or event.args.get("dependency_type") in {
                "overrides",
                "resolutions",
                "pnpm_overrides",
            }
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
        if (
            _is_retry_task(task)
            and _was_package_reverted(task, group, tool_events)
            and "peer" not in lowered_outcome
            and "eresolve" not in lowered_outcome
        ):
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
            executed_versions=attempted,
            candidate_versions_considered=list(prior.candidate_versions_considered)
            if prior
            else [],
            # Version selection belongs to the Supervisor planner. A worker
            # failure must not resurrect the previous planner selection.
            selected_version=prior.selected_version if prior else None,
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
    )
    initial_messages = [
        SystemMessage(
            content=(
                "You are an execution-only dependency worker. The Supervisor owns all version planning. "
                "Execute the exact task instruction, validate manifest synchronization, and do not query registries."
            )
        ),
        HumanMessage(content=prompt),
    ]

    override_required_packages: set[str] = set()
    allowed_dependency_types_by_package: dict[str, set[str]] = {}
    for task, group, _ in skinny_resolved_tasks:
        pkg_name = _target_package_name(task, group)
        target_type = _target_dependency_type(task, group)
        if pkg_name and target_type:
            allowed_dependency_types_by_package.setdefault(pkg_name, set()).add(target_type)
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
                repo_root,
                target_manifest_paths=[
                    manifest_path
                    for _, _, manifest_paths in skinny_resolved_tasks
                    for manifest_path in manifest_paths
                ],
                package_manifest_paths=package_manifest_map,
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

    validation_events = [
        event for event in runtime.tool_events if event.name == "validate_manifest_sync"
    ]
    package_names = {
        _target_package_name(task, group)
        for task, group, _ in resolved_tasks
        if _target_package_name(task, group)
    }
    successful_packages = {
        _target_package_name(task, group)
        for task, group, _ in resolved_tasks
        if _has_successful_validation_for_package(task, group, runtime.tool_events)
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
    succeeded = (
        bool(committed_changed_files)
        and bool(package_names)
        and len(validation_events) == len(package_names)
        and int(execution_state.get("validation_calls", 0)) == len(package_names)
        and all(
            _has_successful_validation_for_package(task, group, runtime.tool_events)
            for _, group, _ in resolved_tasks
        )
    )
    attempted_by_task = _attempted_versions_for_current_run(
        resolved_tasks,
        runtime.tool_events,
    )
    executed_by_task = _executed_versions_for_current_run(
        resolved_tasks,
        runtime.tool_events,
    )
    instruction_mismatch_task_ids: set[str] = set()
    instruction_mismatch_errors: list[str] = []
    for task, _group, _manifest_paths in resolved_tasks:
        snapshot = target_attempt_snapshots.get(task.task_id)
        if snapshot is None or snapshot.selected_version is None:
            continue
        expected = snapshot.selected_version.strip().lstrip("vV").lower()
        observed = {
            version.strip().lstrip("vV").lower()
            for version in attempted_by_task.get(task.task_id, [])
            if version
        }
        if observed and expected not in observed:
            instruction_mismatch_task_ids.add(task.task_id)
            instruction_mismatch_errors.append(
                f"Task {task.task_id}: worker attempted {', '.join(sorted(observed))} "
                f"instead of committed version {snapshot.selected_version}."
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
            validation_calls=sum(
                1 for event in runtime.tool_events if event.name == "validate_manifest_sync"
            ),
        ),
        "errors": resolution_errors + runtime.errors,
    }
