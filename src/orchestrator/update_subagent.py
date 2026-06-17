"""
Batch Update Subagent for Phase 5 dependency resolution.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


def _resolve_manifest_target(group: VulnerabilityGroup, repo_root: Path) -> Tuple[Optional[str], Optional[str]]:
    candidate: Optional[str] = None
    for localized_issue in group.localized_issues:
        if localized_issue.manifest_file:
            candidate = localized_issue.manifest_file
            break
    if not candidate:
        candidate = group.file_path
    if not candidate and group.issues:
        candidate = group.issues[0].file_path

    if not candidate:
        return None, f"Group '{group.group_id}': no manifest target could be resolved."
    if os.path.isabs(candidate) or candidate.startswith(("/", "\\")):
        return None, f"Group '{group.group_id}': rejected absolute manifest path '{candidate}'."
    if ".." in Path(candidate).parts:
        return None, f"Group '{group.group_id}': rejected path traversal in '{candidate}'."

    abs_target = (repo_root / candidate).resolve()
    try:
        abs_target.relative_to(repo_root.resolve())
    except ValueError:
        return None, f"Group '{group.group_id}': manifest path '{candidate}' resolves outside repo_root."
    if not abs_target.exists():
        return None, f"Group '{group.group_id}': manifest path '{candidate}' does not exist in repo."
    if abs_target.is_dir() or abs_target.name != "package.json":
        return None, f"Group '{group.group_id}': manifest target '{candidate}' must be a package.json file."
    return candidate.replace("\\", "/"), None


def _build_update_prompt(
    resolved_groups: Sequence[Tuple[VulnerabilityGroup, str]],
    constraints_ledger: Sequence[str],
    feedback_by_group: Dict[str, str],
) -> str:
    sections = [
        "\n".join(
            [
                "You are a dependency-resolution specialist operating inside a shared Docker workspace.",
                "You may only modify package manifests through modify_npm_dependency.",
                "You must inspect the repository map before making manifest changes.",
                "Immediately after any manifest change, you must call validate_manifest_sync.",
                "If validate_manifest_sync fails, you must resolve the peer conflict or invalid manifest state before finishing.",
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

    for group, manifest_path in resolved_groups:
        feedback = feedback_by_group.get(group.group_id, "none")
        fix_plan = group.fix_plan
        sections.append(
            "\n".join(
                [
                    "=== TARGET GROUP ===",
                    f"Group ID      : {group.group_id}",
                    f"Manifest Path : {manifest_path}",
                    f"Component     : {group.vulnerable_component or 'unknown'}",
                    f"CVEs          : {', '.join(group.cve_ids) if group.cve_ids else 'none'}",
                    f"GHSAs         : {', '.join(group.ghsa_ids) if group.ghsa_ids else 'none'}",
                    f"Versions      : {', '.join(group.versions) if group.versions else 'unknown'}",
                    f"Fix Status    : {fix_plan.status.value if fix_plan else 'none'}",
                    f"Fixed Version : {fix_plan.fixed_version if fix_plan else 'N/A'}",
                    f"Instruction   : {fix_plan.instruction if fix_plan else 'Derive the safest manifest update.'}",
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


def _build_action_summary(
    group_ids: Sequence[str],
    changed_files: Sequence[str],
    final_text: str,
    succeeded: bool,
) -> AgentActionSummary:
    summary_status = AgentActionStatus.SUCCESS if succeeded else AgentActionStatus.SURRENDER
    group_label = ", ".join(group_ids) if group_ids else "no groups"
    changed_label = ", ".join(changed_files) if changed_files else "no files"
    outcome = (
        "Completed validated manifest updates"
        if succeeded
        else "Stopped without a validated manifest update"
    )
    final_note = final_text.strip()
    if final_note:
        summary = f"{outcome} for {group_label}; changed files: {changed_label}. Final note: {final_note}"
    else:
        summary = f"{outcome} for {group_label}; changed files: {changed_label}."
    return AgentActionSummary(group_id="batch:" + group_label, status=summary_status, summary=summary)

@traceable(name="Update_Subagent_Test_Run") # for langsmith testing
def run_update_subagent_node(state: SubagentState) -> Dict[str, Any]:
    """Run the batch dependency update subagent on ``SubagentState``."""
    repo_root_str = state.get("repo_root", "")
    workspace_volume = state.get("workspace_volume", "")
    target_groups = list(state.get("target_groups", []))
    constraints_ledger = list(state.get("constraints_ledger", []))
    feedback_by_group = dict(state.get("feedback_by_group", {}))

    repo_root = Path(repo_root_str)
    if not repo_root_str or not repo_root.is_dir():
        msg = f"Update Subagent: repo_root '{repo_root_str}' is not a valid directory."
        summary = AgentActionSummary(
            group_id="batch:unknown",
            status=AgentActionStatus.SURRENDER,
            summary="Stopped before execution because repo_root was invalid.",
        )
        return {"action_summary": summary, "changed_files": [], "errors": [msg]}

    if not workspace_volume:
        msg = "Update Subagent: workspace_volume is missing from state."
        summary = AgentActionSummary(
            group_id="batch:unknown",
            status=AgentActionStatus.SURRENDER,
            summary="Stopped before execution because workspace_volume was missing.",
        )
        return {"action_summary": summary, "changed_files": [], "errors": [msg]}

    resolved_groups: List[Tuple[VulnerabilityGroup, str]] = []
    resolution_errors: List[str] = []
    for group in target_groups:
        manifest_path, error = _resolve_manifest_target(group, repo_root)
        if error:
            resolution_errors.append(error)
            continue
        resolved_groups.append((group, manifest_path))

    if not resolved_groups:
        summary = AgentActionSummary(
            group_id="batch:unknown",
            status=AgentActionStatus.SURRENDER,
            summary="Stopped before execution because no manifest targets could be resolved.",
        )
        return {"action_summary": summary, "changed_files": [], "errors": resolution_errors}

    if ChatOpenAI is None:
        msg = "Update Subagent: 'langchain-openai' is not installed."
        summary = AgentActionSummary(
            group_id="batch:" + ",".join(group.group_id for group, _ in resolved_groups),
            status=AgentActionStatus.SURRENDER,
            summary="Stopped before execution because the LLM client is unavailable.",
        )
        return {"action_summary": summary, "changed_files": [], "errors": resolution_errors + [msg]}

    model_name = os.environ.get("REMEDY_LLM_MODEL", _DEFAULT_MODEL)
    try:
        llm = ChatOpenAI(model=model_name, temperature=0)
    except Exception as exc:  # noqa: BLE001
        msg = f"Update Subagent: failed to initialize LLM - {exc}."
        summary = AgentActionSummary(
            group_id="batch:" + ",".join(group.group_id for group, _ in resolved_groups),
            status=AgentActionStatus.SURRENDER,
            summary="Stopped before execution because the LLM failed to initialize.",
        )
        return {"action_summary": summary, "changed_files": [], "errors": resolution_errors + [msg]}

    touched_files: set[str] = set()
    prompt = _build_update_prompt(resolved_groups, constraints_ledger, feedback_by_group)
    initial_messages = [
        SystemMessage(content="Use only dependency-management tools and validate manifest synchronization after changes."),
        HumanMessage(content=prompt),
    ]

    try:
        with DockerSandbox(repo_root=None, workspace_volume=workspace_volume) as sandbox:
            toolbelt = build_update_toolbelt(
                sandbox,
                touched_files,
                repo_root,
                target_manifest_paths=[path for _, path in resolved_groups],
            )
            runtime = run_bounded_subagent_loop(llm, toolbelt, initial_messages, touched_files)
    except Exception as exc:  # noqa: BLE001
        msg = f"Update Subagent: sandbox or tool loop failed - {exc}"
        summary = AgentActionSummary(
            group_id="batch:" + ",".join(group.group_id for group, _ in resolved_groups),
            status=AgentActionStatus.SURRENDER,
            summary="Stopped because the sandbox or tool loop failed.",
        )
        return {"action_summary": summary, "changed_files": sorted(touched_files), "errors": resolution_errors + [msg]}

    succeeded = bool(runtime.changed_files) and has_successful_validation_after_last_edit(
        runtime.tool_events,
        edit_tool_name="modify_npm_dependency",
        validation_tool_name="validate_manifest_sync",
    )
    summary = _build_action_summary(
        [group.group_id for group, _ in resolved_groups],
        runtime.changed_files,
        runtime.final_text,
        succeeded,
    )
    return {
        "action_summary": summary,
        "changed_files": runtime.changed_files,
        "errors": resolution_errors + runtime.errors,
    }
