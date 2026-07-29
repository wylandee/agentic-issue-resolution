"""
Shared bounded ReAct runtime for Phase 5 specialist subagents.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage, ToolMessage

from remediation_engine.orchestration.trajectory_exporter import (
    invoke_with_trajectory,
)

MAX_SUBAGENT_TOOL_CALL_ROUNDS = 24
SANDBOX_NOT_RUNNING_MARKER = "Sandbox is not running,"
STAGNATION_REPETITION_THRESHOLD = 2


@dataclass(frozen=True)
class ToolEvent:
    """Record one tool invocation observed by a specialist worker."""

    name: str
    args: dict[str, Any]
    content: str


@dataclass(frozen=True)
class SubagentRuntimeResult:
    """Summarize a bounded worker run and its observable side effects."""

    final_text: str
    tool_events: list[ToolEvent]
    changed_files: list[str]
    errors: list[str]


def _infer_changed_file(tool_event: ToolEvent) -> str | None:
    """Infer a changed file path from a successful edit-like tool event."""
    if not tool_event.content.startswith("SUCCESS:"):
        return None

    if tool_event.name == "modify_npm_dependency":
        manifest_path = tool_event.args.get("manifest_path", "package.json")
        if isinstance(manifest_path, str) and manifest_path.strip():
            return manifest_path.replace("\\", "/")
        return "package.json"

    if tool_event.name in {
        "deterministic_search_replace",
        "deterministic_replace_ast_symbol",
        "revert_workspace_file",
    }:
        file_path = tool_event.args.get("file_path")
        if isinstance(file_path, str) and file_path.strip():
            return file_path.replace("\\", "/")

    return None


def _invoke_bound_tool(tool_map: dict[str, Any], tool_call: dict[str, Any]) -> ToolMessage:
    """Execute one bound tool call and wrap the result in a ToolMessage."""
    tool_name = tool_call.get("name", "")
    tool_args = tool_call.get("args", {}) or {}
    tool_call_id = tool_call.get("id")
    if not isinstance(tool_call_id, str) or not tool_call_id.strip():
        tool_call_id = "invalid-tool-call-id"

    if tool_name not in tool_map:
        content = f"ERROR: Unknown tool '{tool_name}'."
    else:
        try:
            result = invoke_with_trajectory(
                f"tool.{tool_name}",
                lambda: tool_map[tool_name].invoke(tool_args),
                tool_args,
                run_type="tool",
            )
            content = result if isinstance(result, str) else str(result)
        except Exception as exc:  # noqa: BLE001
            content = f"ERROR: Tool '{tool_name}' failed - {exc}"

    return ToolMessage(content=content, tool_call_id=tool_call_id, name=tool_name)


def _tool_call_signature(tool_call: dict[str, Any]) -> tuple[str, str]:
    """Return a stable signature for comparing repeated tool calls."""
    tool_name = str(tool_call.get("name", ""))
    tool_args = tool_call.get("args", {}) or {}

    if tool_name in {"validate_workaround", "run_targeted_test"}:
        return tool_name, str(tool_call.get("id", id(tool_call)))

    if tool_name in {
        "deterministic_search_replace",
        "deterministic_replace_ast_symbol",
        "revert_workspace_file",
    }:
        return tool_name, f"file:{tool_args.get('file_path', '')}"

    try:
        serialized_args = json.dumps(tool_args, sort_keys=True, default=str)
    except (TypeError, ValueError):
        serialized_args = repr(tool_args)
    return tool_name, serialized_args


def _is_failed_tool_result(content: str) -> bool:
    """Return whether a tool response indicates that the requested operation failed."""
    lowered = content.lstrip().lower()
    return lowered.startswith(("error:", "not found:", "failed:", "failure:"))


def _stagnation_recovery_instruction(
    tool_name: str,
    tool_args: dict[str, Any],
    tool_content: str = "",
) -> str:
    """Build a detailed recovery instruction after a repeated failed tool call."""
    formatted_args = json.dumps(tool_args)
    reason = tool_content.strip()[:300] or "Unknown failure"

    alternative = "Use search_codebase_pattern or read_workspace_file to inspect code context."
    if tool_name == "inspect_ast_symbol":
        alternative = (
            "An imported identifier or package binding is not an AST symbol. Use search_codebase_pattern "
            "to find the call site, then inspect the enclosing declared function/class, or use "
            "read_workspace_file and document the fallback in record_plan."
        )
    elif tool_name == "run_targeted_test":
        alternative = (
            "Verify the test path is relative and the runner is supported, or proceed with AST "
            "inspection and syntax validation."
        )
    elif tool_name in {"deterministic_search_replace", "deterministic_replace_ast_symbol"}:
        alternative = (
            "Inspect the file using read_workspace_file or inspect_ast_symbol to verify the exact "
            "anchor lines or line_hint before editing."
        )

    return (
        f"Recovery instruction: tool '{tool_name}' with arguments {formatted_args} has failed repeatedly.\n"
        f"Concrete failure reason: {reason}\n"
        f"Confirmed next action / alternative path: {alternative}\n"
        f"PROHIBITION: You are explicitly prohibited from retrying tool '{tool_name}' with the exact same signature. Reassess your hypothesis and choose a different valid tool or argument set."
    )


def _targeted_test_recovery_instruction(tool_content: str) -> str:
    """Build a structured next-step instruction from a failed targeted test."""
    evidence = tool_content.strip()[:1800] or "The targeted test failed without diagnostic output."
    return (
        "Targeted test feedback requires a hypothesis update before the next edit.\n"
        "Exact targeted-test result:\n"
        f"{evidence}\n"
        "Trace the failure to the cumulative replayed patch and record a revised complete plan "
        "before editing. Preserve all prior security fixes, and include every causally related "
        "import, declaration, call site, and control-flow change in that plan. Do not fix an "
        "unrelated package or file merely because it appears in the stack trace."
    )


def _validation_gate_recovery_instruction(tool_content: str) -> str:
    """Build a recovery message from a failed combined validation gate."""
    evidence = (
        tool_content.strip()[:2400] or "The validation gate failed without diagnostic output."
    )
    return (
        "The combined workaround validation gate failed; do not declare success.\n"
        "Exact validation-gate result:\n"
        f"{evidence}\n"
        "Treat the first failed gate as the current root cause. Reconcile it with the cumulative "
        "patch, re-read every related file, and record one complete revised plan before editing. "
        "Preserve prior security fixes and do not apply an isolated unrelated change. After the "
        "repair, call validate_workaround again with the complete modified-file list."
    )


def run_bounded_subagent_loop(
    llm: Any,
    tools: Sequence[Any],
    initial_messages: Sequence[Any],
    touched_files: set[str],
    planning_state: dict[str, bool] | None = None,
) -> SubagentRuntimeResult:
    """Run a bounded tool-calling loop for one specialized subagent."""
    tool_map = {tool.name: tool for tool in tools}
    llm_with_tools = llm.bind_tools(list(tools))
    conversation = list(initial_messages)
    tool_events: list[ToolEvent] = []
    final_text = ""
    errors: list[str] = []
    observed_changed_files = set(touched_files)
    failed_tool_call_counts: dict[tuple[str, str], int] = {}
    recovery_signatures: set[tuple[str, str]] = set()
    last_failed_tool_content: dict[tuple[str, str], str] = {}
    consecutive_validation_failures = 0

    for _ in range(MAX_SUBAGENT_TOOL_CALL_ROUNDS):
        try:
            response = invoke_with_trajectory(
                "react.llm",
                lambda: llm_with_tools.invoke(list(conversation)),
                list(conversation),
                run_type="llm",
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Subagent LLM/tool-loop invocation failed: {exc}")
            return SubagentRuntimeResult(
                final_text=final_text,
                tool_events=tool_events,
                changed_files=sorted(observed_changed_files | set(touched_files)),
                errors=errors,
            )
        conversation.append(response)
        final_text = str(getattr(response, "content", "") or "")
        if planning_state is not None:
            lowered = final_text.lower()
            if "planning answers" in lowered or "1. what version" in lowered:
                planning_state["submitted"] = True

        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            return SubagentRuntimeResult(
                final_text=final_text,
                tool_events=tool_events,
                changed_files=sorted(observed_changed_files | set(touched_files)),
                errors=errors,
            )

        recovery_instruction: str | None = None
        sandbox_stopped = False
        malformed_tool_call = False
        for tool_call in tool_calls:
            call_signature = _tool_call_signature(tool_call)
            tool_call_id = tool_call.get("id")
            if not isinstance(tool_call_id, str) or not tool_call_id.strip():
                malformed_tool_call = True
                tool_call_id = "invalid-tool-call-id"
            if call_signature in recovery_signatures:
                prior_failure = last_failed_tool_content.get(call_signature, "")[:500]
                tool_message = ToolMessage(
                    content=(
                        "ERROR: Repeated failed tool call suppressed. Choose a different "
                        "tool or argument set; the exact signature has already received a "
                        f"recovery instruction. Prior failure: {prior_failure}"
                    ),
                    tool_call_id=tool_call_id,
                    name=tool_call.get("name", ""),
                )
            else:
                tool_message = _invoke_bound_tool(tool_map, tool_call)
            conversation.append(tool_message)
            event = ToolEvent(
                name=tool_call.get("name", ""),
                args=tool_call.get("args", {}) or {},
                content=tool_message.content,
            )
            tool_events.append(event)

            if _is_failed_tool_result(tool_message.content):
                failed_tool_call_counts[call_signature] = (
                    failed_tool_call_counts.get(call_signature, 0) + 1
                )
                last_failed_tool_content[call_signature] = tool_message.content
                if (
                    failed_tool_call_counts[call_signature] >= STAGNATION_REPETITION_THRESHOLD
                    and call_signature not in recovery_signatures
                ):
                    recovery_instruction = _stagnation_recovery_instruction(
                        event.name,
                        event.args,
                        event.content,
                    )
                    recovery_signatures.add(call_signature)
            else:
                failed_tool_call_counts.pop(call_signature, None)

            if event.name == "run_targeted_test" and event.content.startswith("FAILURE:"):
                conversation.append(
                    HumanMessage(content=_targeted_test_recovery_instruction(event.content))
                )
            if event.name == "validate_workaround":
                if event.content.startswith("FAILURE:"):
                    consecutive_validation_failures += 1
                    if consecutive_validation_failures >= 3:
                        conversation.append(
                            HumanMessage(
                                content=(
                                    "CRITICAL: You have failed the validation gate 3 times in a row. "
                                    "You are in a failure loop. STOP guessing syntax. You MUST use search_web "
                                    "to find the exact error message or migration guide before making further edits."
                                )
                            )
                        )
                        consecutive_validation_failures = 0
                    else:
                        conversation.append(
                            HumanMessage(
                                content=_validation_gate_recovery_instruction(event.content)
                            )
                        )
                else:
                    consecutive_validation_failures = 0

            inferred_path = _infer_changed_file(event)
            if inferred_path:
                if event.name == "revert_workspace_file":
                    if inferred_path not in touched_files:
                        observed_changed_files.discard(inferred_path)
                else:
                    observed_changed_files.add(inferred_path)

            if SANDBOX_NOT_RUNNING_MARKER in tool_message.content:
                # Do not return from inside this loop. The model may have
                # emitted several tool calls in one assistant message and
                # every call needs a matching ToolMessage before the next LLM
                # request. Returning here was the source of malformed OpenAI
                # transcripts in which later tool_call_ids had no response.
                sandbox_stopped = True

        if malformed_tool_call:
            errors.append(
                "Subagent stopped because the model emitted a tool call without a valid tool_call_id."
            )
            return SubagentRuntimeResult(
                final_text=final_text,
                tool_events=tool_events,
                changed_files=sorted(observed_changed_files | set(touched_files)),
                errors=errors,
            )

        if sandbox_stopped:
            errors.extend(
                [
                    event.content
                    for event in tool_events[-len(tool_calls) :]
                    if SANDBOX_NOT_RUNNING_MARKER in event.content
                ]
            )
            errors.append("Subagent stopped because the sandbox is no longer running.")
            return SubagentRuntimeResult(
                final_text=final_text,
                tool_events=tool_events,
                changed_files=sorted(observed_changed_files | set(touched_files)),
                errors=list(dict.fromkeys(errors)),
            )

        if recovery_instruction:
            conversation.append(HumanMessage(content=recovery_instruction))

    errors.append("Subagent exceeded the maximum tool-call rounds without reaching a final answer.")
    return SubagentRuntimeResult(
        final_text=final_text,
        tool_events=tool_events,
        changed_files=sorted(observed_changed_files | set(touched_files)),
        errors=errors,
    )


def has_successful_validation_after_last_edit(
    tool_events: Sequence[ToolEvent],
    edit_tool_name: str,
    validation_tool_name: str,
) -> bool:
    """Return whether the last successful edit was followed by a successful validation."""
    last_successful_edit_index = -1
    for index, event in enumerate(tool_events):
        if event.name == edit_tool_name and event.content.startswith("SUCCESS:"):
            last_successful_edit_index = index

    if last_successful_edit_index < 0:
        return False

    for event in tool_events[last_successful_edit_index + 1 :]:
        if event.name == validation_tool_name and event.content.startswith("SUCCESS:"):
            return True
    return False


def has_all_modified_files_validated_after_last_edit(
    tool_events: Sequence[ToolEvent],
    edit_tool_names: Sequence[str] = (
        "deterministic_search_replace",
        "deterministic_replace_ast_symbol",
    ),
    validation_tool_name: str = "validate_code_syntax",
) -> bool:
    """Return whether every file modified by a successful edit had a successful syntax validation after its last edit."""
    last_edit_by_file: dict[str, int] = {}
    for index, event in enumerate(tool_events):
        if event.name in edit_tool_names and event.content.startswith("SUCCESS:"):
            file_path = event.args.get("file_path")
            if isinstance(file_path, str) and file_path.strip():
                norm_path = file_path.replace("\\", "/").strip()
                last_edit_by_file[norm_path] = index

    if not last_edit_by_file:
        return False

    for norm_path, last_edit_idx in last_edit_by_file.items():
        validated = False
        for event in tool_events[last_edit_idx + 1 :]:
            if event.name == validation_tool_name and event.content.startswith("SUCCESS:"):
                val_path = event.args.get("file_path")
                if not val_path or val_path.replace("\\", "/").strip() == norm_path:
                    validated = True
                    break
        if not validated:
            return False

    return True


def has_successful_validation_gate(
    tool_events: Sequence[ToolEvent],
    validation_tool_name: str = "validate_workaround",
    edit_tool_names: Sequence[str] = (
        "deterministic_search_replace",
        "deterministic_replace_ast_symbol",
        "revert_workspace_file",
    ),
    has_prior_edits: bool = False,
) -> bool:
    """Return whether the combined gate passed after the final cumulative edit.

    A gate result from before a later edit is stale and cannot establish worker
    success. ``has_prior_edits`` covers replayed edits, which are restored into
    the sandbox before the current attempt and therefore do not have edit events
    in the current runtime trace.
    """
    last_edit_index = max(
        (
            index
            for index, event in enumerate(tool_events)
            if event.name in edit_tool_names and event.content.startswith("SUCCESS:")
        ),
        default=-1,
    )
    if last_edit_index < 0 and not has_prior_edits:
        return False

    return any(
        index > last_edit_index
        and event.name == validation_tool_name
        and event.content.startswith("SUCCESS:")
        for index, event in enumerate(tool_events)
    )


def has_single_final_successful_validation(
    tool_events: Sequence[ToolEvent],
    edit_tool_name: str,
    validation_tool_name: str,
) -> bool:
    """Require exactly one successful validation after the final successful edit."""
    validation_indices = [
        index for index, event in enumerate(tool_events) if event.name == validation_tool_name
    ]
    if len(validation_indices) != 1:
        return False

    last_successful_edit_index = max(
        (
            index
            for index, event in enumerate(tool_events)
            if event.name == edit_tool_name and event.content.startswith("SUCCESS:")
        ),
        default=-1,
    )
    validation_index = validation_indices[0]
    return (
        last_successful_edit_index >= 0
        and validation_index > last_successful_edit_index
        and tool_events[validation_index].content.startswith("SUCCESS:")
    )


def has_tool_call_before_first_successful_edit(
    tool_events: Sequence[ToolEvent],
    lookup_tool_name: str,
    edit_tool_names: Sequence[str] | str = (
        "deterministic_search_replace",
        "deterministic_replace_ast_symbol",
    ),
) -> bool:
    """Return whether a given tool was called before the first successful edit."""
    names = (edit_tool_names,) if isinstance(edit_tool_names, str) else tuple(edit_tool_names)
    first_successful_edit_index = -1
    for index, event in enumerate(tool_events):
        if event.name in names and event.content.startswith("SUCCESS:"):
            first_successful_edit_index = index
            break

    if first_successful_edit_index < 0:
        return False

    for event in tool_events[:first_successful_edit_index]:
        if event.name == lookup_tool_name:
            return True
    return False
