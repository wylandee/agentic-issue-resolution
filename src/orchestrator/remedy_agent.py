"""
remedy_agent.py - Phase 5 Remedy Agent node for the AppSec Orchestrator.

The Remedy Agent edits files directly inside the shared Docker workspace by
using native LangChain tools. It also performs dependency install, security
scan, and unit test validation through those tools before finishing.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from src.contracts.schemas import IssueType, VulnerabilityGroup
from src.orchestrator.remedy_tools import build_agent_tools
from src.orchestrator.state import OrchestratorState
from src.runtime.sandbox_mgr import DockerSandbox

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "gpt-4o-mini"
_MAX_TOOL_CALL_ROUNDS = 24

try:
    from langchain_openai import ChatOpenAI  # type: ignore[import]
except ImportError:  # pragma: no cover
    ChatOpenAI = None  # type: ignore[assignment,misc]


def _resolve_target_file(
    group: VulnerabilityGroup,
    repo_root: Path,
) -> Tuple[Optional[str], Optional[str]]:
    """Resolve the repo-relative target file for a vulnerability group."""
    candidate: Optional[str] = None

    if group.issue_type == IssueType.SCA:
        for localized_issue in group.localized_issues:
            if localized_issue.manifest_file:
                candidate = localized_issue.manifest_file
                break
        if not candidate:
            candidate = group.file_path
        if not candidate and group.issues:
            candidate = group.issues[0].file_path
    else:
        candidate = group.file_path
        if not candidate and group.issues:
            candidate = group.issues[0].file_path

    if not candidate:
        return None, f"Group '{group.group_id}': no target file could be resolved."

    if os.path.isabs(candidate) or candidate.startswith(("/", "\\")):
        return None, (
            f"Group '{group.group_id}': rejected absolute file path '{candidate}'."
        )

    parts = Path(candidate).parts
    if ".." in parts:
        return None, (
            f"Group '{group.group_id}': rejected path traversal in '{candidate}'."
        )

    abs_target = (repo_root / candidate).resolve()
    try:
        abs_target.relative_to(repo_root.resolve())
    except ValueError:
        return None, (
            f"Group '{group.group_id}': path '{candidate}' resolves outside repo_root."
        )

    if not abs_target.exists():
        return None, (
            f"Group '{group.group_id}': target file '{candidate}' does not exist in repo."
        )
    if abs_target.is_dir():
        return None, (
            f"Group '{group.group_id}': target path '{candidate}' is a directory."
        )

    return candidate.replace("\\", "/"), None


def _build_group_section(group: VulnerabilityGroup, rel_path: str) -> str:
    """Render a structured description of one vulnerability group."""
    cve_list = ", ".join(group.cve_ids) if group.cve_ids else "none"
    ghsa_list = ", ".join(group.ghsa_ids) if group.ghsa_ids else "none"
    versions = ", ".join(group.versions) if group.versions else "unknown"
    sources = ", ".join(s.value for s in group.sources) if group.sources else "unknown"
    rep_issue = next(
        (
            issue
            for issue in group.issues
            if str(issue.id) == str(group.representative_issue_id)
        ),
        group.issues[0] if group.issues else None,
    )
    rep_msg = (rep_issue.message or "N/A") if rep_issue else "N/A"

    return "\n".join(
        [
            f"Group ID      : {group.group_id}",
            f"Issue Type    : {group.issue_type.value}",
            f"Target File   : {rel_path}",
            f"Component     : {group.vulnerable_component or 'unknown'}",
            f"CVEs          : {cve_list}",
            f"GHSAs         : {ghsa_list}",
            f"Versions      : {versions}",
            f"Sources       : {sources}",
            f"Rep. Message  : {rep_msg}",
        ]
    )


def _build_fix_plan_section(group: VulnerabilityGroup) -> str:
    """Render fix plan details or a note that none is available."""
    fix_plan = group.fix_plan
    if fix_plan is None:
        return (
            "No fix plan is available for this group. Derive the smallest safe fix "
            "from the vulnerability details and the current workspace file content."
        )

    lines = [
        f"Status         : {fix_plan.status.value}",
        f"Strategy       : {fix_plan.strategy_used}",
        f"Fixed Version  : {fix_plan.fixed_version or 'N/A'}",
        f"Instruction    : {fix_plan.instruction}",
    ]
    if fix_plan.workaround_snippets:
        lines.append("Workaround Snippets:")
        for snippet in fix_plan.workaround_snippets[:3]:
            lines.append(f"  ---\n{snippet}\n  ---")
    return "\n".join(lines)


def _build_prompt(
    resolved_groups: List[Tuple[VulnerabilityGroup, str]],
    repo_root: str,
) -> str:
    """Assemble the full prompt for the tool-using Remedy Agent."""
    sections: List[str] = [
        "\n".join(
            [
                "You are an autonomous application security engineer.",
                "Use your tools to inspect and edit files inside the shared Docker workspace.",
                "You may only modify the repo-relative target files listed below.",
                "Always read a file before editing it.",
                "Use modify_npm_dependency to update package.json, and deterministic_search_replace for other source code files.",
                "After each edit, read the file again to verify the result.",
                (
                    f"You have been assigned exactly {len(resolved_groups)} groups to fix. "
                    "When and only when all tools report SUCCESS, stop calling tools and "
                    "return a concise summary."
                ),
            ]
        )
    ]

    sections.append(
        "\n".join(
            [
                "REQUIRED STEP-BY-STEP SEQUENCE",
                "1. Inspect: Read the assigned target files with read_workspace_file.",
                "2. Modify Manifest: Use modify_npm_dependency to apply dependency upgrades and overrides. Do not edit package.json using deterministic_search_replace.",
                "3. Sync: Run run_dependency_install to synchronize the changes.",
                "4. Scan & Test: Run run_security_scan and run_unit_tests to verify the fixes.",
            ]
        )
    )

    sections.append(
        "\n".join(
            [
                "CRITICAL SAFETY RULES",
                "Never edit files outside the allowed target files list.",
                "Do not invent file contents; anchor every change to current workspace text.",
                "Do not leave package.json or package-lock.json in invalid JSON state.",
                "Never introduce trailing commas in package.json.",
                "Never use highly generic single characters (like '}' or '{') as your old_text.",
                "Always ensure your old_text contains unique, context-rich surrounding lines (at least 2-3 lines of context) so that the tool finds exactly one match.",
                "MANIFEST UPGRADE RULES (package.json / package-lock.json):",
                "  - You MUST NOT use deterministic_search_replace to modify package.json or package-lock.json.",
                "  - deterministic_search_replace is reserved strictly for non-manifest files (such as JavaScript/TypeScript source code or configuration scripts).",
                "  - Use modify_npm_dependency to modify manifests with the appropriate dependency_type:",
                "    * To update a direct production dependency (like sanitize-html), set dependency_type to 'dependencies'.",
                "    * To pin a transitive dependency (like lodash or lodash.set) via overrides, set dependency_type to 'overrides'.",
            ]
        )
    )

    sections.append(
        "\n".join(
            [
                "REVERT AND RESCUE PROTOCOLS",
                "If you encounter structural failures (like syntax parser crashes during package installation) or find yourself in a loop of consecutive failures, you must call revert_workspace_file to reset the file to its original host baseline state rather than trying to incrementally repair broken syntax.",
                "Immediately after calling revert_workspace_file, you must call read_workspace_file to inspect the fresh baseline content before attempting any edits again.",
            ]
        )
    )

    sections.append(f"repo_root (host reference only): {repo_root}")
    sections.append(
        "Allowed target files:\n" + "\n".join(
            f"- {rel_path}" for _, rel_path in resolved_groups
        )
    )

    for group, rel_path in resolved_groups:
        sections.append("=== VULNERABILITY GROUP ===\n" + _build_group_section(group, rel_path))
        sections.append("=== FIX PLAN ===\n" + _build_fix_plan_section(group))

    return "\n\n".join(sections)


def _validate_edits(*_args, **_kwargs):
    """
    Compatibility shim for older Phase 5 harnesses that intercepted the
    structured-edit path. The native-tool runtime no longer uses this helper.
    """
    return [], []


def _invoke_bound_tool(tool_map: Dict[str, Any], tool_call: Dict[str, Any]) -> ToolMessage:
    """Execute one bound tool call and wrap the result in a ToolMessage."""
    tool_name = tool_call.get("name", "")
    tool_args = tool_call.get("args", {}) or {}
    tool_call_id = tool_call.get("id")

    if tool_name not in tool_map:
        content = f"ERROR: Unknown tool '{tool_name}'."
    else:
        try:
            result = tool_map[tool_name].invoke(tool_args)
            content = result if isinstance(result, str) else str(result)
        except Exception as exc:  # noqa: BLE001
            content = f"ERROR: Tool '{tool_name}' failed - {exc}"

    return ToolMessage(content=content, tool_call_id=tool_call_id, name=tool_name)


def run_remedy_agent(state: OrchestratorState) -> Dict[str, Any]:
    """
    LangGraph node - Remedy Agent.

    Uses native tools against the shared workspace volume to inspect, edit, and
    validate vulnerable files directly inside Docker.
    """
    repo_root_str: str = state.get("repo_root", "")
    workspace_volume: Optional[str] = state.get("workspace_volume")
    valid_groups: List[VulnerabilityGroup] = state.get("valid_groups", [])

    repo_root = Path(repo_root_str)
    if not repo_root_str or not repo_root.is_dir():
        msg = f"Remedy Agent: repo_root '{repo_root_str}' is not a valid directory."
        logger.error(msg)
        return {"status": "remedy_failed", "errors": [msg]}

    if not workspace_volume:
        msg = "Remedy Agent: workspace_volume is missing from state."
        logger.error(msg)
        return {"status": "remedy_failed", "errors": [msg]}

    resolved_groups: List[Tuple[VulnerabilityGroup, str]] = []
    resolution_errors: List[str] = []
    target_identifiers: Set[str] = set()
    for group in valid_groups:
        rel_path, resolve_err = _resolve_target_file(group, repo_root)
        if resolve_err:
            logger.warning("Remedy Agent: %s", resolve_err)
            resolution_errors.append(resolve_err)
            continue
        resolved_groups.append((group, rel_path))
        for cve in group.cve_ids or []:
            if cve:
                target_identifiers.add(cve.upper().strip())
        for ghsa in group.ghsa_ids or []:
            if ghsa:
                target_identifiers.add(ghsa.upper().strip())

    if not resolved_groups:
        errors = resolution_errors or [
            "Remedy Agent: no valid target files were resolved for any group."
        ]
        return {"status": "remedy_failed", "errors": errors}

    if ChatOpenAI is None:
        msg = (
            "Remedy Agent: 'langchain-openai' is not installed. Run: "
            "pip install langchain-openai"
        )
        logger.error(msg)
        return {"status": "remedy_failed", "errors": [msg]}

    model_name = os.environ.get("REMEDY_LLM_MODEL", _DEFAULT_MODEL)
    try:
        llm = ChatOpenAI(model=model_name, temperature=0)
    except Exception as exc:  # noqa: BLE001
        msg = f"Remedy Agent: failed to initialize LLM - {exc}."
        logger.error(msg)
        return {"status": "remedy_failed", "errors": [msg]}

    touched_files: set[str] = set()
    conversation = list(state.get("messages", []))
    new_messages: List[Any] = []
    prompt = _build_prompt(
        resolved_groups=resolved_groups,
        repo_root=repo_root_str,
    )

    system_message = SystemMessage(
        content=(
            "You must use tools for file inspection, edits, and validation. Only "
            "modify the allowed target files. Never invent file contents."
        )
    )
    human_message = HumanMessage(content=prompt)
    new_messages.extend([system_message, human_message])
    conversation.extend([system_message, human_message])

    try:
        with DockerSandbox(repo_root=None, workspace_volume=workspace_volume) as sandbox:
            tools = build_agent_tools(sandbox, touched_files, target_identifiers, repo_root)
            tool_map = {tool.name: tool for tool in tools}
            llm_with_tools = llm.bind_tools(tools)

            for _ in range(_MAX_TOOL_CALL_ROUNDS):
                response = llm_with_tools.invoke(list(conversation))
                new_messages.append(response)
                conversation.append(response)

                tool_calls = getattr(response, "tool_calls", None) or []
                if not tool_calls:
                    result: Dict[str, Any] = {
                        "messages": new_messages,
                        "changed_files": sorted(touched_files),
                        "status": "edits_completed" if touched_files else "no_changes_made",
                    }
                    if resolution_errors:
                        result["errors"] = resolution_errors
                    return result

                for tool_call in tool_calls:
                    tool_message = _invoke_bound_tool(tool_map, tool_call)
                    new_messages.append(tool_message)
                    conversation.append(tool_message)
    except Exception as exc:  # noqa: BLE001
        msg = f"Remedy Agent: sandbox or tool loop failed - {exc}"
        logger.exception("Remedy Agent: tool loop failed.")
        return {
            "status": "remedy_failed",
            "errors": resolution_errors + [msg],
            "messages": new_messages,
            "changed_files": sorted(touched_files),
        }

    msg = (
        f"Remedy Agent: exceeded maximum tool-call rounds ({_MAX_TOOL_CALL_ROUNDS}) "
        "without reaching a final answer."
    )
    logger.error(msg)
    return {
        "status": "remedy_failed",
        "errors": resolution_errors + [msg],
        "messages": new_messages,
        "changed_files": sorted(touched_files),
    }
