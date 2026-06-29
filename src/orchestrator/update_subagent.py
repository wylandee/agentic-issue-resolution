"""
Batch Update Subagent for Phase 5 dependency resolution.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from src.contracts.schemas import (
    AgentActionStatus,
    AgentActionSummary,
    FixPlanStatus,
    VulnerabilityGroup,
)
from src.orchestrator.remedy_tools import build_update_toolbelt
from src.orchestrator.state import SubagentState
from src.orchestrator.subagent_runtime import (
    has_successful_validation_after_last_edit,
    run_bounded_subagent_loop,
)
from src.runtime.sandbox_mgr import DockerSandbox

from langsmith import traceable

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "gpt-4o-mini"

try:
    from langchain_openai import ChatOpenAI  # type: ignore[import]
except ImportError:  # pragma: no cover
    ChatOpenAI = None  # type: ignore[assignment,misc]


def _candidate_manifest_paths(group: VulnerabilityGroup) -> List[str]:
    """Return all candidate manifest paths for one grouped dependency target."""
    candidates: List[str] = []
    seen: set[str] = set()

    def add_candidate(value: str | None) -> None:
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
    return group.model_copy(update={
        "cve_ids": [],
        "ghsa_ids": [],
        "versions": [],
        "issues": [],
    })


def _filter_constraints_ledger(
    constraints_ledger: Sequence[str],
    target_groups: Sequence[VulnerabilityGroup]
) -> List[str]:
    """Filter ledger to only include constraints matching the target components."""
    components = [g.vulnerable_component for g in target_groups if g.vulnerable_component]
    if not components:
        return list(constraints_ledger)
    
    filtered = []
    for constraint in constraints_ledger:
        if any(comp in constraint for comp in components):
            filtered.append(constraint)
    return filtered


def _resolve_manifest_targets(
    group: VulnerabilityGroup,
    repo_root: Path,
) -> Tuple[List[str], List[str]]:
    """Resolve all valid package.json targets for one vulnerability group."""
    candidates = _candidate_manifest_paths(group)
    if not candidates:
        return [], [f"Group '{group.group_id}': no manifest target could be resolved."]

    resolved_paths: List[str] = []
    errors: List[str] = []

    for candidate in candidates:
        if os.path.isabs(candidate) or candidate.startswith(("/", "\\")):
            errors.append(
                f"Group '{group.group_id}': rejected absolute manifest path '{candidate}'."
            )
            continue
        if ".." in Path(candidate).parts:
            errors.append(
                f"Group '{group.group_id}': rejected path traversal in '{candidate}'."
            )
            continue

        abs_target = (repo_root / candidate).resolve()
        try:
            abs_target.relative_to(repo_root.resolve())
        except ValueError:
            errors.append(
                f"Group '{group.group_id}': manifest path '{candidate}' resolves outside repo_root."
            )
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
    resolved_tasks: Sequence[Tuple[RemediationTask, VulnerabilityGroup, Sequence[str]]],
) -> Dict[str, List[str]]:
    """Build a per-package allowlist of manifest paths for tool enforcement."""
    package_manifest_map: Dict[str, List[str]] = {}
    for _, group, manifest_paths in resolved_tasks:
        package_name = (group.vulnerable_component or "").strip()
        if not package_name:
            continue
        existing = package_manifest_map.setdefault(package_name, [])
        for manifest_path in manifest_paths:
            if manifest_path not in existing:
                existing.append(manifest_path)
    return package_manifest_map


def _build_update_prompt(
    resolved_tasks: Sequence[Tuple[RemediationTask, VulnerabilityGroup, Sequence[str]]],
    constraints_ledger: Sequence[str],
    feedback_by_task: Dict[str, str],
) -> str:
    sections = [
        "\n".join(
            [
                "You are a dependency-resolution specialist operating inside a shared Docker workspace.",
                "You may only modify package manifests through modify_npm_dependency.",
                "You must inspect the repository map before making manifest changes.",
                "Immediately after any manifest change, you must call validate_manifest_sync.",
                "If validate_manifest_sync fails, you must resolve the peer conflict or invalid manifest state before finishing.",
                "Every modify_npm_dependency call must include the exact manifest_path you intend to edit.",
                "If the vulnerable component appears in multiple manifest paths, call modify_npm_dependency once for each manifest_path that declares it.",
                "Never downgrade a package that is constrained by the constraints ledger.",
                "Use revert_workspace_file if you reach a structurally bad manifest state.",
                "Do not search the codebase and do not edit source code files.",
            ]
        )
    ]

    if constraints_ledger:
        sections.append("Constraints ledger:\n" + "\n".join(f"- {item}" for item in constraints_ledger))
    else:
        sections.append("Constraints ledger:\n- none")

    for task, group, manifest_paths in resolved_tasks:
        feedback = feedback_by_task.get(task.task_id, "none")
        sections.append(
            "\n".join(
                [
                    "=== TARGET ===",
                    f"Task ID       : {task.task_id}",
                    f"Manifest Path : {_format_manifest_paths(manifest_paths)}",
                    f"Component     : {group.vulnerable_component or 'unknown'}",
                    f"Instruction   : {task.instruction or 'Derive the safest manifest update.'}",
                    f"QA Feedback   : {feedback}",
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


def _build_action_summaries(
    task_ids: Sequence[str],
    changed_files: Sequence[str],
    final_text: str,
    succeeded: bool,
) -> List[AgentActionSummary]:
    summary_status = AgentActionStatus.SUCCESS if succeeded else AgentActionStatus.SURRENDER
    changed_label = ", ".join(changed_files) if changed_files else "no files"
    outcome = (
        "Completed validated manifest updates"
        if succeeded
        else "Stopped without a validated manifest update"
    )
    final_note = final_text.strip()
    if final_note:
        summary_text = f"{outcome}; changed files: {changed_label}. Final note: {final_note}"
    else:
        summary_text = f"{outcome}; changed files: {changed_label}."
    
    return [
        AgentActionSummary(task_id=t_id, status=summary_status, summary=summary_text)
        for t_id in task_ids
    ]

def _build_surrender_summaries(task_ids: Sequence[str], message: str) -> List[AgentActionSummary]:
    return [
        AgentActionSummary(task_id=t_id, status=AgentActionStatus.SURRENDER, summary=message)
        for t_id in task_ids
    ]

@traceable(name="Update_Subagent_Test_Run") # for langsmith testing
def run_update_subagent_node(state: SubagentState) -> Dict[str, Any]:
    """Run the batch dependency update subagent on ``SubagentState``."""
    repo_root_str = state.get("repo_root", "")
    workspace_volume = state.get("workspace_volume", "")
    target_tasks = list(state.get("target_tasks", []))
    target_groups = list(state.get("target_groups", []))
    constraints_ledger = list(state.get("constraints_ledger", []))
    feedback_by_task = dict(state.get("feedback_by_task", {}))
    all_task_ids = [t.task_id for t in target_tasks]

    repo_root = Path(repo_root_str)
    if not repo_root_str or not repo_root.is_dir():
        msg = f"Update Subagent: repo_root '{repo_root_str}' is not a valid directory."
        summaries = _build_surrender_summaries(all_task_ids, "Stopped before execution because repo_root was invalid.")
        return {"action_summaries": summaries, "changed_files": [], "errors": [msg]}

    if not workspace_volume:
        msg = "Update Subagent: workspace_volume is missing from state."
        summaries = _build_surrender_summaries(all_task_ids, "Stopped before execution because workspace_volume was missing.")
        return {"action_summaries": summaries, "changed_files": [], "errors": [msg]}

    resolved_tasks: List[Tuple[RemediationTask, VulnerabilityGroup, List[str]]] = []
    resolution_errors: List[str] = []
    for task, group in zip(target_tasks, target_groups):
        manifest_paths, errors = _resolve_manifest_targets(group, repo_root)
        resolution_errors.extend(errors)
        if not manifest_paths:
            continue
        resolved_tasks.append((task, group, manifest_paths))

    if not resolved_tasks:
        summaries = _build_surrender_summaries(all_task_ids, "Stopped before execution because no manifest targets could be resolved.")
        return {"action_summaries": summaries, "changed_files": [], "errors": resolution_errors}

    resolved_task_ids = [t.task_id for t, _, _ in resolved_tasks]
    if ChatOpenAI is None:
        msg = "Update Subagent: 'langchain-openai' is not installed."
        summaries = _build_surrender_summaries(resolved_task_ids, "Stopped before execution because the LLM client is unavailable.")
        return {"action_summaries": summaries, "changed_files": [], "errors": resolution_errors + [msg]}

    model_name = os.environ.get("REMEDY_LLM_MODEL", _DEFAULT_MODEL)
    try:
        llm = ChatOpenAI(model=model_name, temperature=0)
    except Exception as exc:  # noqa: BLE001
        msg = f"Update Subagent: failed to initialize LLM - {exc}."
        summaries = _build_surrender_summaries(resolved_task_ids, "Stopped before execution because the LLM failed to initialize.")
        return {"action_summaries": summaries, "changed_files": [], "errors": resolution_errors + [msg]}

    touched_files: set[str] = set()
    
    filtered_ledger = _filter_constraints_ledger(constraints_ledger, target_groups)
    skinny_resolved_tasks = [
        (t, _create_skinny_subagent_group(g), paths) for t, g, paths in resolved_tasks
    ]
    
    prompt = _build_update_prompt(skinny_resolved_tasks, filtered_ledger, feedback_by_task)
    initial_messages = [
        SystemMessage(content="Use only dependency-management tools and validate manifest synchronization after changes."),
        HumanMessage(content=prompt),
    ]

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
            )
            runtime = run_bounded_subagent_loop(llm, toolbelt, initial_messages, touched_files)
    except Exception as exc:  # noqa: BLE001
        msg = f"Update Subagent: sandbox or tool loop failed - {exc}"
        summaries = _build_surrender_summaries(resolved_task_ids, "Stopped because the sandbox or tool loop failed.")
        return {"action_summaries": summaries, "changed_files": sorted(touched_files), "errors": resolution_errors + [msg]}

    succeeded = bool(runtime.changed_files) and has_successful_validation_after_last_edit(
        runtime.tool_events,
        edit_tool_name="modify_npm_dependency",
        validation_tool_name="validate_manifest_sync",
    )
    summaries = _build_action_summaries(
        resolved_task_ids,
        runtime.changed_files,
        runtime.final_text,
        succeeded,
    )
    return {
        "action_summaries": summaries,
        "changed_files": runtime.changed_files,
        "errors": resolution_errors + runtime.errors,
    }
