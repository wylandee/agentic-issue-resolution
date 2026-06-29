"""
Sequential Workaround Subagent for Phase 5 code-security rewrites.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from src.contracts.schemas import AgentActionStatus, AgentActionSummary, VulnerabilityGroup
from src.orchestrator.remedy_tools import build_workaround_toolbelt
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


def _create_skinny_subagent_group(group: VulnerabilityGroup) -> VulnerabilityGroup:
    """Create a skinny copy of a group for execution agents."""
    return group.model_copy(update={
        "cve_ids": [],
        "ghsa_ids": [],
        "versions": [],
        "issues": [],
    })


def _filter_constraints_ledger(
    constraints_ledger: List[str],
    target_group: VulnerabilityGroup
) -> List[str]:
    """Filter ledger to only include constraints matching the target component."""
    comp = target_group.vulnerable_component
    if not comp:
        return list(constraints_ledger)
    
    filtered = []
    for constraint in constraints_ledger:
        if comp in constraint:
            filtered.append(constraint)
    return filtered


def _build_workaround_prompt(
    target_task: Any,  # RemediationTask (imported dynamically or typed properly)
    target_group: VulnerabilityGroup,
    constraints_ledger: List[str],
    previous_feedback: str | None,
) -> str:
    fix_plan = target_group.fix_plan
    sections = [
        "\n".join(
            [
                "You are a code security specialist operating inside a shared Docker workspace.",
                "You must strictly follow this Standard Operating Procedure (SOP):",
                "1. ALWAYS use relative file paths (e.g., 'frontend/src/app.ts'). NEVER use absolute paths starting with '/' or '/workspace'.",
                "2. When using search_codebase_pattern, ALWAYS start your search from the repository root ('.') to ensure you don't miss files, unless you are 100% certain of the directory.",
                "3. NEVER use inspect_ast_symbol on a file unless search_codebase_pattern has explicitly confirmed the file exists and contains the vulnerable logic.",
                "4. After every modified file, you must run validate_code_syntax on that file.",
                "5. If syntax validation fails, you must repair the file or revert it before finishing.",
                "Do not modify package manifests and do not call heavy QA tools."
            ]
        )
    ]

    if constraints_ledger:
        sections.append("Constraints ledger:\n" + "\n".join(f"- {item}" for item in constraints_ledger))
    else:
        sections.append("Constraints ledger:\n- none")

    sections.append(
        "\n".join(
            [
                "=== TARGET ===",
                f"Task ID       : {target_task.task_id}",
                f"Issue Type    : {target_group.issue_type.value}",
                f"Component     : {target_group.vulnerable_component or 'unknown'}",
                f"Initial File  : {target_group.file_path or 'none'}",
                f"Instruction   : {target_task.instruction or 'Derive the smallest safe code change.'}",
                f"QA Feedback   : {previous_feedback or 'none'}",
            ]
        )
    )
    if fix_plan and fix_plan.workaround_snippets:
        sections.append(
            "\n".join(
                [
                    "Workaround snippets:",
                    *[f"- {snippet}" for snippet in fix_plan.workaround_snippets],
                ]
            )
        )
    else:
        sections.append("Workaround snippets:\n- none")
    return "\n\n".join(sections)


def _build_action_summaries(
    task_id: str,
    changed_files: List[str],
    final_text: str,
    succeeded: bool,
) -> List[AgentActionSummary]:
    summary_status = AgentActionStatus.SUCCESS if succeeded else AgentActionStatus.SURRENDER
    changed_label = ", ".join(changed_files) if changed_files else "no files"
    outcome = (
        "Completed validated code workaround edits"
        if succeeded
        else "Stopped without a validated code workaround"
    )
    final_note = final_text.strip()
    if final_note:
        summary_text = f"{outcome}; changed files: {changed_label}. Final note: {final_note}"
    else:
        summary_text = f"{outcome}; changed files: {changed_label}."
    return [AgentActionSummary(task_id=task_id, status=summary_status, summary=summary_text)]

def _build_surrender_summaries(task_id: str, message: str) -> List[AgentActionSummary]:
    return [AgentActionSummary(task_id=task_id, status=AgentActionStatus.SURRENDER, summary=message)]

@traceable(name="Workaround_Subagent_Test_Run") # for langsmith testing
def run_workaround_subagent_node(state: SubagentState) -> Dict[str, Any]:
    """Run the single-group workaround subagent on ``SubagentState``."""
    repo_root_str = state.get("repo_root", "")
    workspace_volume = state.get("workspace_volume", "")
    target_task = state.get("target_task")
    target_group = state.get("target_group")
    constraints_ledger = list(state.get("constraints_ledger", []))
    previous_feedback = state.get("previous_feedback")
    
    t_id = target_task.task_id if target_task else "unknown"

    repo_root = Path(repo_root_str)
    if not repo_root_str or not repo_root.is_dir():
        summaries = _build_surrender_summaries(t_id, "Stopped before execution because repo_root was invalid.")
        return {
            "action_summaries": summaries,
            "changed_files": [],
            "errors": [f"Workaround Subagent: repo_root '{repo_root_str}' is not a valid directory."],
        }

    if not workspace_volume:
        summaries = _build_surrender_summaries(t_id, "Stopped before execution because workspace_volume was missing.")
        return {
            "action_summaries": summaries,
            "changed_files": [],
            "errors": ["Workaround Subagent: workspace_volume is missing from state."],
        }

    if target_task is None or target_group is None:
        summaries = _build_surrender_summaries(t_id, "Stopped before execution because no target task/group was provided.")
        return {
            "action_summaries": summaries,
            "changed_files": [],
            "errors": ["Workaround Subagent: target_task or target_group is missing from state."],
        }

    if ChatOpenAI is None:
        summaries = _build_surrender_summaries(t_id, "Stopped before execution because the LLM client is unavailable.")
        return {
            "action_summaries": summaries,
            "changed_files": [],
            "errors": ["Workaround Subagent: 'langchain-openai' is not installed."],
        }

    model_name = os.environ.get("REMEDY_LLM_MODEL", _DEFAULT_MODEL)
    try:
        llm = ChatOpenAI(model=model_name, temperature=0)
    except Exception as exc:  # noqa: BLE001
        summaries = _build_surrender_summaries(t_id, "Stopped before execution because the LLM failed to initialize.")
        return {
            "action_summaries": summaries,
            "changed_files": [],
            "errors": [f"Workaround Subagent: failed to initialize LLM - {exc}."],
        }

    touched_files: set[str] = set()
    filtered_ledger = _filter_constraints_ledger(constraints_ledger, target_group)
    skinny_group = _create_skinny_subagent_group(target_group)
    prompt = _build_workaround_prompt(target_task, skinny_group, filtered_ledger, previous_feedback)
    initial_messages = [
        SystemMessage(content="Use only source-code tools and validate syntax after each modified file."),
        HumanMessage(content=prompt),
    ]

    try:
        with DockerSandbox(repo_root=None, workspace_volume=workspace_volume) as sandbox:
            toolbelt = build_workaround_toolbelt(sandbox, touched_files, repo_root)
            runtime = run_bounded_subagent_loop(llm, toolbelt, initial_messages, touched_files)
    except Exception as exc:  # noqa: BLE001
        summaries = _build_surrender_summaries(t_id, "Stopped because the sandbox or tool loop failed.")
        return {
            "action_summaries": summaries,
            "changed_files": sorted(touched_files),
            "errors": [f"Workaround Subagent: sandbox or tool loop failed - {exc}"],
        }

    succeeded = bool(runtime.changed_files) and has_successful_validation_after_last_edit(
        runtime.tool_events,
        edit_tool_name="deterministic_search_replace",
        validation_tool_name="validate_code_syntax",
    )
    summaries = _build_action_summaries(
        t_id,
        runtime.changed_files,
        runtime.final_text,
        succeeded,
    )
    return {
        "action_summaries": summaries,
        "changed_files": runtime.changed_files,
        "errors": runtime.errors,
    }
