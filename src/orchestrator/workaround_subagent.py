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


def _build_workaround_prompt(
    target_group: VulnerabilityGroup,
    constraints_ledger: List[str],
    previous_feedback: str | None,
) -> str:
    fix_plan = target_group.fix_plan
    sections = [
        "\n".join(
            [
                "You are a code security specialist operating inside a shared Docker workspace.",
                "You must inspect the repository map first and use AST-guided context gathering before editing.",
                "You may only inspect and edit source files.",
                "After every modified file, you must run validate_code_syntax on that file.",
                "If syntax validation fails, you must repair the file or revert it before finishing.",
                "Do not modify package manifests and do not call heavy QA tools.",
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
                "=== TARGET GROUP ===",
                f"Group ID      : {target_group.group_id}",
                f"Issue Type    : {target_group.issue_type.value}",
                f"Component     : {target_group.vulnerable_component or 'unknown'}",
                f"Initial File  : {target_group.file_path or 'none'}",
                f"CVEs          : {', '.join(target_group.cve_ids) if target_group.cve_ids else 'none'}",
                f"GHSAs         : {', '.join(target_group.ghsa_ids) if target_group.ghsa_ids else 'none'}",
                f"Versions      : {', '.join(target_group.versions) if target_group.versions else 'unknown'}",
                f"Fix Status    : {fix_plan.status.value if fix_plan else 'none'}",
                f"Instruction   : {fix_plan.instruction if fix_plan else 'Derive the smallest safe code change.'}",
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


def _build_action_summary(
    group_id: str,
    changed_files: List[str],
    final_text: str,
    succeeded: bool,
) -> AgentActionSummary:
    summary_status = AgentActionStatus.SUCCESS if succeeded else AgentActionStatus.SURRENDER
    changed_label = ", ".join(changed_files) if changed_files else "no files"
    outcome = (
        "Completed validated code workaround edits"
        if succeeded
        else "Stopped without a validated code workaround"
    )
    final_note = final_text.strip()
    if final_note:
        summary = f"{outcome} for {group_id}; changed files: {changed_label}. Final note: {final_note}"
    else:
        summary = f"{outcome} for {group_id}; changed files: {changed_label}."
    return AgentActionSummary(group_id=group_id, status=summary_status, summary=summary)

@traceable(name="Workaround_Subagent_Test_Run") # for langsmith testing
def run_workaround_subagent_node(state: SubagentState) -> Dict[str, Any]:
    """Run the single-group workaround subagent on ``SubagentState``."""
    repo_root_str = state.get("repo_root", "")
    workspace_volume = state.get("workspace_volume", "")
    target_group = state.get("target_group")
    constraints_ledger = list(state.get("constraints_ledger", []))
    previous_feedback = state.get("previous_feedback")

    repo_root = Path(repo_root_str)
    if not repo_root_str or not repo_root.is_dir():
        summary = AgentActionSummary(
            group_id=target_group.group_id if target_group else "unknown",
            status=AgentActionStatus.SURRENDER,
            summary="Stopped before execution because repo_root was invalid.",
        )
        return {
            "action_summary": summary,
            "changed_files": [],
            "errors": [f"Workaround Subagent: repo_root '{repo_root_str}' is not a valid directory."],
        }

    if not workspace_volume:
        summary = AgentActionSummary(
            group_id=target_group.group_id if target_group else "unknown",
            status=AgentActionStatus.SURRENDER,
            summary="Stopped before execution because workspace_volume was missing.",
        )
        return {
            "action_summary": summary,
            "changed_files": [],
            "errors": ["Workaround Subagent: workspace_volume is missing from state."],
        }

    if target_group is None:
        summary = AgentActionSummary(
            group_id="unknown",
            status=AgentActionStatus.SURRENDER,
            summary="Stopped before execution because no target group was provided.",
        )
        return {
            "action_summary": summary,
            "changed_files": [],
            "errors": ["Workaround Subagent: target_group is missing from state."],
        }

    if ChatOpenAI is None:
        summary = AgentActionSummary(
            group_id=target_group.group_id,
            status=AgentActionStatus.SURRENDER,
            summary="Stopped before execution because the LLM client is unavailable.",
        )
        return {
            "action_summary": summary,
            "changed_files": [],
            "errors": ["Workaround Subagent: 'langchain-openai' is not installed."],
        }

    model_name = os.environ.get("REMEDY_LLM_MODEL", _DEFAULT_MODEL)
    try:
        llm = ChatOpenAI(model=model_name, temperature=0)
    except Exception as exc:  # noqa: BLE001
        summary = AgentActionSummary(
            group_id=target_group.group_id,
            status=AgentActionStatus.SURRENDER,
            summary="Stopped before execution because the LLM failed to initialize.",
        )
        return {
            "action_summary": summary,
            "changed_files": [],
            "errors": [f"Workaround Subagent: failed to initialize LLM - {exc}."],
        }

    touched_files: set[str] = set()
    prompt = _build_workaround_prompt(target_group, constraints_ledger, previous_feedback)
    initial_messages = [
        SystemMessage(content="Use only source-code tools and validate syntax after each modified file."),
        HumanMessage(content=prompt),
    ]

    try:
        with DockerSandbox(repo_root=None, workspace_volume=workspace_volume) as sandbox:
            toolbelt = build_workaround_toolbelt(sandbox, touched_files, repo_root)
            runtime = run_bounded_subagent_loop(llm, toolbelt, initial_messages, touched_files)
    except Exception as exc:  # noqa: BLE001
        summary = AgentActionSummary(
            group_id=target_group.group_id,
            status=AgentActionStatus.SURRENDER,
            summary="Stopped because the sandbox or tool loop failed.",
        )
        return {
            "action_summary": summary,
            "changed_files": sorted(touched_files),
            "errors": [f"Workaround Subagent: sandbox or tool loop failed - {exc}"],
        }

    succeeded = bool(runtime.changed_files) and has_successful_validation_after_last_edit(
        runtime.tool_events,
        edit_tool_name="deterministic_search_replace",
        validation_tool_name="validate_code_syntax",
    )
    summary = _build_action_summary(
        target_group.group_id,
        runtime.changed_files,
        runtime.final_text,
        succeeded,
    )
    return {
        "action_summary": summary,
        "changed_files": runtime.changed_files,
        "errors": runtime.errors,
    }
