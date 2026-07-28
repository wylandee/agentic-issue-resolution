"""
Sequential Workaround Subagent for Phase 5 code-security rewrites.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from remediation_engine.contracts.schemas import (
    AgentActionStatus,
    AgentActionSummary,
    WorkerAttemptResult,
    WorkerExecutionDiagnostics,
    VulnerabilityGroup,
    WorkaroundEdit,
    WorkaroundReplayPlan,
)
from remediation_engine.orchestration.remedy_tools import build_workaround_toolbelt
from remediation_engine.orchestration.state import SubagentState, _derive_legacy_task_from_group
from remediation_engine.orchestration.subagent_runtime import (
    has_all_modified_files_validated_after_last_edit,
    has_tool_call_before_first_successful_edit,
    run_bounded_subagent_loop,
)
from remediation_engine.runtime.sandbox_mgr import DockerSandbox

from langsmith import traceable

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "gpt-4o-mini"

try:
    from langchain_openai import ChatOpenAI  # type: ignore[import]
except ImportError:  # pragma: no cover
    ChatOpenAI = None  # type: ignore[assignment,misc]


def _create_skinny_subagent_group(group: VulnerabilityGroup) -> VulnerabilityGroup:
    """Create a skinny copy of a group for execution agents while preserving compact vulnerability identifiers."""
    return group.model_copy(update={
        "cve_ids": group.cve_ids[:5],
        "ghsa_ids": group.ghsa_ids[:5],
        "versions": group.versions[:5],
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


_QA_ERROR_MARKER = re.compile(
    r"(?:error|exception|failed|failure|not\s+a\s+function|undefined|cannot|invalid|missing)",
    re.IGNORECASE,
)


def _clean_prompt_snippet(value: str, max_chars: int = 240) -> str:
    """Normalize one extracted QA or vulnerability snippet for prompt use."""
    cleaned = re.sub(r"\s+", " ", value).strip(" `\"'")
    cleaned = cleaned.replace("`", "").replace('"', "")
    return cleaned[:max_chars].strip()


def _extract_qa_error_snippet(previous_feedback: str) -> str:
    """Extract the most diagnostic error text from QA feedback for web search."""
    feedback = str(previous_feedback or "")

    explicit_error = re.search(
        r"\b[A-Za-z_][\w.]*(?:Error|Exception)\s*:\s*[^;\n]{1,240}",
        feedback,
        re.IGNORECASE,
    )
    if explicit_error:
        return _clean_prompt_snippet(explicit_error.group(0))

    for match in re.finditer(r"`([^`\n]{1,240})`|\"([^\"\n]{1,240})\"", feedback):
        candidate = match.group(1) or match.group(2) or ""
        if _QA_ERROR_MARKER.search(candidate):
            return _clean_prompt_snippet(candidate)

    not_a_function = re.search(
        r"\([^\n)]{1,240}\)\s+is\s+not\s+a\s+function",
        feedback,
        re.IGNORECASE,
    )
    if not_a_function:
        return _clean_prompt_snippet(not_a_function.group(0))

    for line in feedback.splitlines():
        if _QA_ERROR_MARKER.search(line):
            return _clean_prompt_snippet(line)

    return _clean_prompt_snippet(feedback, max_chars=240)


def _extract_vulnerability_mechanism(group: VulnerabilityGroup) -> str:
    """Extract a compact vulnerability mechanism before the detailed fix guidance."""
    for issue in getattr(group, "issues", []) or []:
        message = getattr(issue, "message", None)
        if not isinstance(message, str) or not message.strip():
            continue

        mechanism = message
        for section_marker in ("### Am I affected?", "### How to fix that?"):
            mechanism = mechanism.split(section_marker, 1)[0]
        mechanism = _clean_prompt_snippet(mechanism, max_chars=1200)
        if mechanism:
            return mechanism

    fix_plan = getattr(group, "fix_plan", None)
    instruction = getattr(fix_plan, "instruction", None)
    if isinstance(instruction, str) and instruction.strip():
        return _clean_prompt_snippet(instruction, max_chars=1200)
    return ""


def _build_workaround_prompt(
    target_task: Any,  # RemediationTask
    target_group: VulnerabilityGroup | List[str],
    constraints_ledger: List[str] | None = None,
    previous_feedback: str | None = None,
    current_replay_plan: WorkaroundReplayPlan | None = None,
    vulnerability_mechanism: str | None = None,
) -> str:
    if isinstance(target_group, list):
        constraints_ledger = list(target_group)
        target_group = target_task
        target_task = _derive_legacy_task_from_group(target_group)

    constraints_ledger = list(constraints_ledger or [])
    fix_plan = getattr(target_group, "fix_plan", None)
    is_sast = getattr(getattr(target_group, "issue_type", None), "value", "") == "sast"
    vulnerability_mechanism = (
        _clean_prompt_snippet(vulnerability_mechanism, max_chars=1200)
        if vulnerability_mechanism
        else _extract_vulnerability_mechanism(target_group)
    )

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
                "  2. For SCA issues: Use search_web to search for the library's migration guide, changelog, or breaking changes for the target version. Include error details if provided.",
                "  3. Use read_repository_map to understand the project layout.",
                "  4. Use search_codebase_pattern starting from '.' (repository root) to find all files where the vulnerable code pattern appears.",
                "  5. Use read_workspace_file to read the relevant file(s) and understand the full context around the vulnerable code.",
                "  6. For relevant JS/TS functions or classes, use inspect_ast_symbol to extract the symbol body before editing (or document a fallback if no AST symbol exists).",
                "  7. If workaround snippets are provided, study them to understand the recommended mitigation pattern.",
                "",
                "PHASE 2 — PLAN:",
                "  8. Determine the minimal code change required. Prefer adding validation/sanitization/guards rather than rewriting logic.",
                "  9. Identify every file that needs modification.",
                "  10. Use read_workspace_file on each file to confirm the exact current code before editing.",
                "",
                "PHASE 3 — EXECUTE:",
                "  11. CRITICAL: Call `record_plan` BEFORE executing any `deterministic_search_replace` to describe affected files, symbols, and intended changes.",
                "  12. Use `deterministic_search_replace` for each change. The old_text must be an exact copy of the current code.",
                "  13. Make one edit per file at a time.",
                "",
                "PHASE 4 — VALIDATE:",
                "  14. After EVERY modified file, run validate_code_syntax on that file.",
                "  15. If validation fails, either fix the syntax error with another deterministic_search_replace, or revert_workspace_file and retry.",
                "",
                "PHASE 5 — CLOSURE REVIEW (mandatory before finishing):",
                "  16. Re-search the vulnerable pattern with search_codebase_pattern to confirm no vulnerable call sites remain.",
                "  17. Inspect planned symbols with inspect_ast_symbol or re-read modified files with read_workspace_file.",
                "  18. Re-validate syntax on EVERY modified file after its final edit.",
                "  19. Run run_typecheck when TypeScript compilation is supported by the workspace.",
                "",
                "=== ANTI-PATTERNS (violations will cause immediate QA failure) ===",
                "- ❌ NEVER modify package.json, package-lock.json, yarn.lock, pnpm-lock.yaml, or any dependency manifest.",
                "- ❌ NEVER bump library versions — version selection is strictly the update_subagent's job.",
                "- ❌ NEVER use npm/yarn/pnpm commands.",
                "- ❌ NEVER add new external dependencies.",
                "",
                "=== STRICT RULES ===",
                "- CRITICAL: You MUST use the `record_plan` tool to explicitly write out your investigation findings and your exact code changes BEFORE executing any code edit tools.",
                "- ALWAYS use relative file paths (e.g., 'lib/insecurity.ts'). NEVER use absolute paths starting with '/'.",
                "- Do NOT modify package.json, package-lock.json, or any dependency manifest.",
                "- If you cannot find the vulnerable code or determine a safe fix, stop and explain why.",
            ]
        )
    ]

    if current_replay_plan and current_replay_plan.successful_edits:
        replay_lines = [
            "=== CUMULATIVE REPLAY CONTEXT ===",
            f"The following {len(current_replay_plan.successful_edits)} valid code workaround edit(s) from prior attempt(s) have been automatically replayed onto your pre-task baseline:",
        ]
        for edit in current_replay_plan.successful_edits:
            replay_lines.append(f"  - File: {edit.file_path} (edit #{edit.edit_index})")
        replay_lines.append("Build directly on top of these replayed edits to address the remaining QA feedback.")
        sections.append("\n".join(replay_lines))

    if previous_feedback:
        comp_name = getattr(target_group, "vulnerable_component", "") or "component"
        cve_label = (
            target_group.cve_ids[0]
            if getattr(target_group, "cve_ids", None)
            else (
                target_group.ghsa_ids[0]
                if getattr(target_group, "ghsa_ids", None)
                else ""
            )
        )
        err_snippet = _extract_qa_error_snippet(previous_feedback)
        suggested_query = f'{comp_name} {cve_label} "{err_snippet}" workaround migration breaking change'.strip()

        retry_lines = [
            "=== RETRY CONTEXT ===",
            "This is a RETRY attempt. A previous code change was rejected by QA.",
            f"QA Feedback: {previous_feedback}",
            f"SUGGESTED TARGETED SEARCH QUERY: {suggested_query}",
            "",
            "IMPORTANT: Prior valid edits have been restored and replayed. Address the specific failure reason above.",
        ]
        sections.append("\n".join(retry_lines))

    if constraints_ledger:
        sections.append("Constraints ledger:\n" + "\n".join(f"- {item}" for item in constraints_ledger))
    else:
        sections.append("Constraints ledger:\n- none")

    target_lines = [
        "=== TARGET ===",
        f"Task ID       : {target_task.task_id}",
        f"Issue Type    : {getattr(target_group.issue_type, 'value', str(target_group.issue_type))}",
        f"Component     : {getattr(target_group, 'vulnerable_component', '') or 'unknown'}",
        f"Initial File  : {getattr(target_group, 'file_path', '') or 'none'}",
        f"Vulnerability Mechanism: {vulnerability_mechanism or 'not provided'}",
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

    if fix_plan and getattr(fix_plan, "workaround_snippets", None):
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
    replay_plan: Optional[WorkaroundReplayPlan] = None,
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
            replay_plan=replay_plan,
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
    current_replay_plan = state.get("current_replay_plan")
    
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
    vulnerability_mechanism = _extract_vulnerability_mechanism(target_group)
    skinny_group = _create_skinny_subagent_group(target_group)
    plan_state = {"recorded": False}

    pre_attempt_snapshots: Dict[str, str] = {}
    replayed_edits: List[WorkaroundEdit] = []
    if current_replay_plan is not None:
        pre_attempt_snapshots = dict(current_replay_plan.pre_attempt_snapshots)
        replayed_edits = list(current_replay_plan.successful_edits)

    try:
        with DockerSandbox(repo_root=None, workspace_volume=workspace_volume) as sandbox:
            # 1. Restore pre-attempt snapshots if replay plan present
            if pre_attempt_snapshots:
                for rel_p, orig_content in pre_attempt_snapshots.items():
                    try:
                        sandbox.write_file(rel_p, orig_content)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Workaround subagent: failed to restore snapshot for %s: %s", rel_p, exc)

            # 2. Replay prior successful edits sequentially
            for redit in replayed_edits:
                try:
                    curr = sandbox.read_file(redit.file_path)
                    if curr and redit.old_text in curr:
                        updated = curr.replace(redit.old_text, redit.new_text, 1)
                        sandbox.write_file(redit.file_path, updated)
                        touched_files.add(redit.file_path)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Workaround subagent: failed to replay edit on %s: %s", redit.file_path, exc)

            prompt = _build_workaround_prompt(
                target_task,
                skinny_group,
                filtered_ledger,
                previous_feedback,
                current_replay_plan,
                vulnerability_mechanism=vulnerability_mechanism,
            )
            initial_messages = [
                SystemMessage(content="Use only source-code tools and validate syntax after each modified file."),
                HumanMessage(content=prompt),
            ]

            toolbelt = build_workaround_toolbelt(sandbox, touched_files, repo_root, plan_state=plan_state)
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

            # Build cumulative WorkaroundEdit list & capture new pre-edit snapshots
            all_edits: List[WorkaroundEdit] = list(replayed_edits)
            for event in runtime.tool_events:
                if event.name == "deterministic_search_replace" and event.content.startswith("SUCCESS:"):
                    f_path = event.args.get("file_path", "")
                    old_t = event.args.get("old_text", "")
                    new_t = event.args.get("new_text", "")
                    if f_path:
                        norm_f = f_path.replace("\\", "/").strip()
                        if norm_f not in pre_attempt_snapshots:
                            pre_attempt_snapshots[norm_f] = old_t
                        all_edits.append(
                            WorkaroundEdit(
                                file_path=norm_f,
                                old_text=old_t,
                                new_text=new_t,
                                edit_index=len(all_edits) + 1,
                            )
                        )
    except Exception as exc:  # noqa: BLE001
        summaries = _build_surrender_summaries(t_id, "Stopped because the sandbox or tool loop failed.")
        return {
            "action_summaries": summaries,
            "action_summary": summaries[0],
            "changed_files": sorted(touched_files),
            "errors": [f"Workaround Subagent: sandbox or tool loop failed - {exc}"],
        }

    snapshot = state.get("attempt_snapshot")

    new_replay_plan = WorkaroundReplayPlan(
        task_id=t_id,
        pre_attempt_snapshots=pre_attempt_snapshots,
        successful_edits=all_edits,
        investigation_findings={"changed_files": list(touched_files)},
        source_attempt_id=snapshot.attempt_id if snapshot else "",
    )

    has_all_validated = has_all_modified_files_validated_after_last_edit(
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
    is_record_plan_in_toolbelt = any(
        getattr(t, "name", "") == "record_plan" for t in toolbelt
    )
    has_recorded_plan = (
        (not is_record_plan_in_toolbelt)
        or plan_state.get("recorded", False)
        or has_tool_call_before_first_successful_edit(
            runtime.tool_events,
            lookup_tool_name="record_plan",
            edit_tool_name="deterministic_search_replace",
        )
    )

    succeeded = bool(runtime.changed_files) and has_all_validated and did_investigate and has_recorded_plan

    summaries = _build_action_summaries(
        t_id,
        runtime.changed_files,
        runtime.final_text,
        succeeded,
    )
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
            replay_plan=new_replay_plan,
        ),
        "errors": runtime.errors,
    }


