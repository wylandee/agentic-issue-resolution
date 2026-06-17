"""
Shared bounded ReAct runtime for Phase 5 specialist subagents.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Set

from langchain_core.messages import ToolMessage

MAX_SUBAGENT_TOOL_CALL_ROUNDS = 24
SANDBOX_NOT_RUNNING_MARKER = "Sandbox is not running,"


@dataclass(frozen=True)
class ToolEvent:
    name: str
    args: Dict[str, Any]
    content: str


@dataclass(frozen=True)
class SubagentRuntimeResult:
    final_text: str
    tool_events: List[ToolEvent]
    changed_files: List[str]
    errors: List[str]


def _infer_changed_file(tool_event: ToolEvent) -> str | None:
    """Infer a changed file path from a successful edit-like tool event."""
    if not tool_event.content.startswith("SUCCESS:"):
        return None

    if tool_event.name == "modify_npm_dependency":
        manifest_path = tool_event.args.get("manifest_path", "package.json")
        if isinstance(manifest_path, str) and manifest_path.strip():
            return manifest_path.replace("\\", "/")
        return "package.json"

    if tool_event.name in {"deterministic_search_replace", "revert_workspace_file"}:
        file_path = tool_event.args.get("file_path")
        if isinstance(file_path, str) and file_path.strip():
            return file_path.replace("\\", "/")

    return None


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


def run_bounded_subagent_loop(
    llm: Any,
    tools: Sequence[Any],
    initial_messages: Sequence[Any],
    touched_files: Set[str],
) -> SubagentRuntimeResult:
    """Run a bounded tool-calling loop for one specialized subagent."""
    tool_map = {tool.name: tool for tool in tools}
    llm_with_tools = llm.bind_tools(list(tools))
    conversation = list(initial_messages)
    tool_events: List[ToolEvent] = []
    final_text = ""
    errors: List[str] = []
    observed_changed_files = set(touched_files)

    for _ in range(MAX_SUBAGENT_TOOL_CALL_ROUNDS):
        response = llm_with_tools.invoke(list(conversation))
        conversation.append(response)
        final_text = str(getattr(response, "content", "") or "")

        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            return SubagentRuntimeResult(
                final_text=final_text,
                tool_events=tool_events,
                changed_files=sorted(observed_changed_files | set(touched_files)),
                errors=errors,
            )

        for tool_call in tool_calls:
            tool_message = _invoke_bound_tool(tool_map, tool_call)
            conversation.append(tool_message)
            event = ToolEvent(
                name=tool_call.get("name", ""),
                args=tool_call.get("args", {}) or {},
                content=tool_message.content,
            )
            tool_events.append(event)

            inferred_path = _infer_changed_file(event)
            if inferred_path:
                if event.name == "revert_workspace_file":
                    observed_changed_files.discard(inferred_path)
                else:
                    observed_changed_files.add(inferred_path)

            if SANDBOX_NOT_RUNNING_MARKER in tool_message.content:
                errors.extend(
                    [
                        tool_message.content,
                        "Subagent stopped immediately because the sandbox is no longer running.",
                    ]
                )
                return SubagentRuntimeResult(
                    final_text=final_text,
                    tool_events=tool_events,
                    changed_files=sorted(observed_changed_files | set(touched_files)),
                    errors=errors,
                )

    errors.append(
        "Subagent exceeded the maximum tool-call rounds without reaching a final answer."
    )
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

    for event in tool_events[last_successful_edit_index + 1:]:
        if event.name == validation_tool_name and event.content.startswith("SUCCESS:"):
            return True
    return False
