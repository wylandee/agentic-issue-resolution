"""
Sequential Workaround Subagent for Phase 5 code-security rewrites.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from src.contracts.schemas import (
    AgentActionStatus,
    AgentActionSummary,
    WorkerAttemptResult,
    WorkerExecutionDiagnostics,
    VulnerabilityGroup,
)
from src.orchestrator.remedy_tools import build_workaround_toolbelt
from src.orchestrator.state import SubagentState, _derive_legacy_task_from_group
from src.orchestrator.subagent_runtime import (
    has_successful_validation_after_last_edit,
    has_tool_call_before_first_successful_edit,
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
    target_group: VulnerabilityGroup | List[str],
    constraints_ledger: List[str] | None = None,
    previous_feedback: str | None = None,
) -> str:
    if isinstance(target_group, list):
        constraints_ledger = list(target_group)
        target_group = target_task
        target_task = _derive_legacy_task_from_group(target_group)

    constraints_ledger = list(constraints_ledger or [])
    fix_plan = target_group.fix_plan
    is_sast = target_group.issue_type.value == "sast"

    sections = [
        "\n".join(
            [
                "You are a code security specialist operating inside a shared Docker workspace.",
                "Your mission is to apply the smallest, safest source-code change that mitigates the described vulnerability.",
                "",
                "=== STANDARD OPERATING PROCEDURE ===",
                "",
                "PHASE 1 — INVESTIGATE:",
                "  1. Read the TARGET section below to understand the vulnerability and the instruction.",
                "  2. Use read_repository_map to understand the project layout.",
                "  3. Use search_codebase_pattern starting from '.' (repository root) to find all files where the vulnerable code pattern appears.",
                "  4. Use read_workspace_file to read the relevant file(s) and understand the full context around the vulnerable code.",
                "  5. If needed, use inspect_ast_symbol to extract the full body of a specific function or class for deeper analysis.",
                "  6. If workaround snippets are provided, study them to understand the recommended mitigation pattern.",
                "",
                "PHASE 2 — PLAN:",
                "  7. Determine the minimal code change required. Prefer adding validation/sanitization/guards rather than rewriting logic.",
                "     - For SAST issues: Add input validation, output encoding, parameterized queries, or other defensive patterns at the vulnerable sink.",
                "     - For SCA issues: Apply the workaround pattern from the snippets to eliminate the exploitable code path.",
                "  8. Identify every file that needs modification.",
                "  9. Use read_workspace_file on each file to confirm the exact current code before editing.",
                "",
                "PHASE 3 — EXECUTE:",
                "  10. Use deterministic_search_replace for each change. The old_text must be an exact copy of the current code (use read_workspace_file output to get the exact text).",
                "  11. Make one edit per file at a time. Do not batch multiple edits in a single call.",
                "",
                "PHASE 4 — VALIDATE:",
                "  12. After EVERY modified file, immediately run validate_code_syntax and run_typecheck on that file.",
                "  13. If validation fails, either fix the syntax error with another deterministic_search_replace, or revert_workspace_file and retry.",
                "  14. Do NOT finish until every modified file passes syntax validation.",
                "",
                "=== ANTI-PATTERNS (violations will cause immediate QA failure) ===",
                "- ❌ NEVER modify package.json, package-lock.json, yarn.lock, pnpm-lock.yaml, or any dependency manifest.",
                "- ❌ NEVER bump library versions — version selection is strictly the update_subagent's job.",
                "- ❌ NEVER use npm/yarn/pnpm commands.",
                "- ❌ NEVER add new external dependencies.",
                "",
                "=== PRE-EDIT CHECKLIST (mandatory before every deterministic_search_replace) ===",
                "1. Use read_workspace_file to view the EXACT current content of the target file.",
                "2. Copy the EXACT old_text from the file content (character-for-character, carefully checking commas, brackets, and semicolons).",
                "3. Verify your new_text is valid JavaScript/TypeScript syntax before making the tool call.",
                "4. After editing, ALWAYS run validate_code_syntax and run_typecheck on the modified file.",
                "",
                "=== STRICT RULES ===",
                "- ALWAYS use relative file paths (e.g., 'frontend/src/app.ts'). NEVER use absolute paths starting with '/' or '/workspace'.",
                "- NEVER use inspect_ast_symbol on a file unless search_codebase_pattern or read_workspace_file has confirmed it exists and contains relevant code.",
                "- Do NOT modify package.json, package-lock.json, or any dependency manifest.",
                "- Do NOT install new packages or run npm/yarn commands.",
                "- If you cannot find the vulnerable code or determine a safe fix, stop and explain why.",
                ]
        )
    ]

    if previous_feedback:
        retry_lines = [
            "=== RETRY CONTEXT ===",
            "This is a RETRY attempt. A previous code change was rejected by QA.",
            f"QA Feedback: {previous_feedback}",
            "",
            "IMPORTANT: Your previous failed code changes have been safely discarded. You are starting from a clean baseline workspace. Apply a totally new fix that addresses the QA feedback.",
        ]
        sections.append("\n".join(retry_lines))

    if constraints_ledger:
        sections.append("Constraints ledger:\n" + "\n".join(f"- {item}" for item in constraints_ledger))
    else:
        sections.append("Constraints ledger:\n- none")

    target_lines = [
        "=== TARGET ===",
        f"Task ID       : {target_task.task_id}",
        f"Issue Type    : {target_group.issue_type.value}",
        f"Component     : {target_group.vulnerable_component or 'unknown'}",
        f"Initial File  : {target_group.file_path or 'none'}",
    ]

    if is_sast:
        target_lines.append(
            f"Instruction   : {target_task.instruction or 'Apply a defensive code fix at the vulnerable sink identified above.'}"
        )
    else:
        target_lines.append(
            f"Instruction   : {target_task.instruction or 'Apply the workaround code pattern from the snippets below to mitigate the CVE.'}"
        )

    sections.append("\n".join(target_lines))

    if fix_plan and fix_plan.workaround_snippets:
        sections.append(
            "\n".join(
                [
                    "=== WORKAROUND SNIPPETS ===",
                    "These are reference code patterns from security advisories. Adapt them to fit the project's existing code:",
                    *[f"  {i+1}. {snippet}" for i, snippet in enumerate(fix_plan.workaround_snippets)],
                ]
            )
        )
    else:
        sections.append("Workaround snippets:\n- none (derive the fix from the instruction and CVE context)")

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


def _build_attempt_result(
    state: SubagentState,
    summary: AgentActionSummary,
    *,
    succeeded: bool,
    errors: List[str],
    changed_files: List[str],
) -> Dict[str, WorkerAttemptResult]:
    snapshot = state.get("attempt_snapshot")
    if snapshot is None:
        return {}
    tagged_summary = summary.model_copy(
        update={
            "attempt_id": snapshot.attempt_id,
            "task_revision": snapshot.task_revision,
            "instruction_digest": snapshot.instruction_digest,
        }
    )
    return {
        snapshot.attempt_id: WorkerAttemptResult(
            attempt_id=snapshot.attempt_id,
            task_id=snapshot.task_id,
            task_revision=snapshot.task_revision,
            status=tagged_summary.status,
            changed_files=changed_files,
            action_summary=tagged_summary,
            execution_diagnostics=WorkerExecutionDiagnostics(
                validation_passed=succeeded,
                failure_reason=" | ".join(errors),
            ),
            instruction_digest=snapshot.instruction_digest,
            errors=errors,
        )
    }

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

    if os.environ.get("REMEDY_BYPASS_WORKAROUND_SUBAGENT", "false").lower() in ("1", "true", "yes"):
        summaries = _build_surrender_summaries(
            t_id,
            "Workaround subagent bypassed: marked unfixable (workaround functionality currently inactive)."
        )
        return {
            "action_summaries": summaries,
            "action_summary": summaries[0],
            "changed_files": [],
            "worker_results_by_attempt": _build_attempt_result(
                state,
                summaries[0],
                succeeded=False,
                errors=[],
                changed_files=[],
            ),
            "errors": [],
        }

    repo_root = Path(repo_root_str)
    if not repo_root_str or not repo_root.is_dir():
        summaries = _build_surrender_summaries(t_id, "Stopped before execution because repo_root was invalid.")
        return {
            "action_summaries": summaries,
            "action_summary": summaries[0],
            "changed_files": [],
            "errors": [f"Workaround Subagent: repo_root '{repo_root_str}' is not a valid directory."],
        }

    if not workspace_volume:
        summaries = _build_surrender_summaries(t_id, "Stopped before execution because workspace_volume was missing.")
        return {
            "action_summaries": summaries,
            "action_summary": summaries[0],
            "changed_files": [],
            "errors": ["Workaround Subagent: workspace_volume is missing from state."],
        }

    if target_task is None or target_group is None:
        summaries = _build_surrender_summaries(t_id, "Stopped before execution because no target task/group was provided.")
        return {
            "action_summaries": summaries,
            "action_summary": summaries[0],
            "changed_files": [],
            "errors": ["Workaround Subagent: target_task or target_group is missing from state."],
        }

    if ChatOpenAI is None:
        summaries = _build_surrender_summaries(t_id, "Stopped before execution because the LLM client is unavailable.")
        return {
            "action_summaries": summaries,
            "action_summary": summaries[0],
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
            "action_summary": summaries[0],
            "changed_files": [],
            "errors": [f"Workaround Subagent: failed to initialize LLM - {exc}."],
        }

    touched_files: set[str] = set()
    filtered_ledger = _filter_constraints_ledger(constraints_ledger, target_group)
    skinny_group = _create_skinny_subagent_group(target_group)

    try:
        with DockerSandbox(repo_root=None, workspace_volume=workspace_volume) as sandbox:
            prompt = _build_workaround_prompt(
                target_task,
                skinny_group,
                filtered_ledger,
                previous_feedback,
            )
            initial_messages = [
                SystemMessage(content="Use only source-code tools and validate syntax after each modified file."),
                HumanMessage(content=prompt),
            ]

            toolbelt = build_workaround_toolbelt(sandbox, touched_files, repo_root)
            runtime = run_bounded_subagent_loop(llm, toolbelt, initial_messages, touched_files)

            # Programmatic guardrail: revert any dependency manifest files modified by the subagent
            MANIFEST_FILES = {"package.json", "package-lock.json", "yarn.lock", "pnpm-lock.yaml"}
            violated_manifests = {
                f for f in touched_files if os.path.basename(f) in MANIFEST_FILES
            }
            if violated_manifests:
                logger.warning(
                    "Workaround subagent illegally modified manifest files %s. Reverting manifests.",
                    violated_manifests,
                )
                for f in violated_manifests:
                    try:
                        sandbox.revert_file(f)
                    except Exception as exc:  # noqa: BLE001
                        logger.error("Failed to revert manifest file %s: %s", f, exc)
                touched_files -= violated_manifests
                runtime = runtime._replace(changed_files=sorted(touched_files))
    except Exception as exc:  # noqa: BLE001
        summaries = _build_surrender_summaries(t_id, "Stopped because the sandbox or tool loop failed.")
        return {
            "action_summaries": summaries,
            "action_summary": summaries[0],
            "changed_files": sorted(touched_files),
            "errors": [f"Workaround Subagent: sandbox or tool loop failed - {exc}"],
        }

    has_validated = has_successful_validation_after_last_edit(
        runtime.tool_events,
        edit_tool_name="deterministic_search_replace",
        validation_tool_name="validate_code_syntax",
    )
    did_investigate = (
        has_tool_call_before_first_successful_edit(
            runtime.tool_events,
            lookup_tool_name="search_codebase_pattern",
            edit_tool_name="deterministic_search_replace",
        )
        or has_tool_call_before_first_successful_edit(
            runtime.tool_events,
            lookup_tool_name="inspect_ast_symbol",
            edit_tool_name="deterministic_search_replace",
        )
        or has_tool_call_before_first_successful_edit(
            runtime.tool_events,
            lookup_tool_name="read_workspace_file",
            edit_tool_name="deterministic_search_replace",
        )
    )
    succeeded = bool(runtime.changed_files) and has_validated and did_investigate
    summaries = _build_action_summaries(
        t_id,
        runtime.changed_files,
        runtime.final_text,
        succeeded,
    )
    snapshot = state.get("attempt_snapshot")
    tagged_summaries = [
        summaries[0].model_copy(
            update={
                "attempt_id": snapshot.attempt_id,
                "task_revision": snapshot.task_revision,
                "instruction_digest": snapshot.instruction_digest,
            }
        )
        if snapshot is not None
        else summaries[0]
    ]
    return {
        "action_summaries": tagged_summaries,
        "action_summary": tagged_summaries[0],
        "changed_files": runtime.changed_files,
        "worker_results_by_attempt": _build_attempt_result(
            state,
            tagged_summaries[0],
            succeeded=succeeded,
            errors=list(runtime.errors),
            changed_files=list(runtime.changed_files),
        ),
        "errors": runtime.errors,
    }
