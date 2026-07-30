"""Regression tests for the shared specialist-subagent runtime."""

from __future__ import annotations

from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage

from remediation_engine.orchestration.subagent_runtime import (
    ToolEvent,
    has_successful_validation_gate,
    run_bounded_subagent_loop,
)


def test_repeated_failed_ast_call_injects_recovery_without_terminating() -> None:
    """A repeated failed AST lookup should guide the next LLM turn."""
    ast_tool = MagicMock()
    ast_tool.name = "inspect_ast_symbol"
    ast_tool.invoke.return_value = (
        "NOT FOUND: No declared function, class, or method named "
        "'expressJwt' was found in 'lib/insecurity.ts'."
    )

    tool_call_args = {
        "file_path": "lib/insecurity.ts",
        "symbol_name": "expressJwt",
    }
    responses = [
        AIMessage(
            content="",
            tool_calls=[{"name": ast_tool.name, "args": tool_call_args, "id": "call-1"}],
        ),
        AIMessage(
            content="",
            tool_calls=[{"name": ast_tool.name, "args": tool_call_args, "id": "call-2"}],
        ),
        AIMessage(content="I changed strategy and continued the investigation."),
    ]
    bound_llm = MagicMock()
    bound_llm.invoke.side_effect = responses
    llm = MagicMock()
    llm.bind_tools.return_value = bound_llm

    result = run_bounded_subagent_loop(
        llm,
        [ast_tool],
        [HumanMessage(content="Investigate the vulnerability.")],
        set(),
    )

    assert result.errors == []
    assert len(result.tool_events) == 2
    assert bound_llm.invoke.call_count == 3
    third_turn = bound_llm.invoke.call_args_list[2].args[0]
    recovery_messages = [
        message
        for message in third_turn
        if isinstance(message, HumanMessage) and "Recovery instruction" in message.content
    ]
    assert len(recovery_messages) == 1
    assert "imported identifier" in recovery_messages[0].content
    assert "read_workspace_file" in recovery_messages[0].content


def test_exact_failed_signature_is_suppressed_after_recovery() -> None:
    """A worker can continue after recovery without executing the bad call again."""
    ast_tool = MagicMock()
    ast_tool.name = "inspect_ast_symbol"
    ast_tool.invoke.return_value = "NOT FOUND: imported identifier is not a declared symbol."
    args = {"file_path": "lib/insecurity.ts", "symbol_name": "expressJwt"}
    responses = [
        AIMessage(content="", tool_calls=[{"name": ast_tool.name, "args": args, "id": "1"}]),
        AIMessage(content="", tool_calls=[{"name": ast_tool.name, "args": args, "id": "2"}]),
        AIMessage(content="", tool_calls=[{"name": ast_tool.name, "args": args, "id": "3"}]),
        AIMessage(content="I used the enclosing function instead."),
    ]
    bound_llm = MagicMock()
    bound_llm.invoke.side_effect = responses
    llm = MagicMock()
    llm.bind_tools.return_value = bound_llm

    result = run_bounded_subagent_loop(
        llm,
        [ast_tool],
        [HumanMessage(content="Investigate the vulnerability.")],
        set(),
    )

    assert result.errors == []
    assert bound_llm.invoke.call_count == 4
    assert ast_tool.invoke.call_count == 2
    assert "Repeated failed tool call suppressed" in result.tool_events[2].content


def test_sandbox_failure_drains_all_tool_calls_before_returning() -> None:
    """A multi-tool assistant turn must receive a response for every call."""
    stopped_tool = MagicMock()
    stopped_tool.name = "read_workspace_file"
    stopped_tool.invoke.return_value = "ERROR: Sandbox is not running, so the file cannot be read."
    second_tool = MagicMock()
    second_tool.name = "search_codebase_pattern"
    second_tool.invoke.return_value = "search result"

    response = AIMessage(
        content="",
        tool_calls=[
            {"name": stopped_tool.name, "args": {"file_path": "src/a.ts"}, "id": "call-1"},
            {"name": second_tool.name, "args": {"search_pattern": "needle"}, "id": "call-2"},
        ],
    )
    bound_llm = MagicMock()
    bound_llm.invoke.return_value = response
    llm = MagicMock()
    llm.bind_tools.return_value = bound_llm

    result = run_bounded_subagent_loop(
        llm,
        [stopped_tool, second_tool],
        [HumanMessage(content="Inspect the source.")],
        set(),
    )

    assert bound_llm.invoke.call_count == 1
    assert stopped_tool.invoke.call_count == 1
    assert second_tool.invoke.call_count == 1
    assert [event.name for event in result.tool_events] == [
        "read_workspace_file",
        "search_codebase_pattern",
    ]
    assert any("Sandbox is not running" in error for error in result.errors)


def test_validation_gate_must_follow_the_final_edit() -> None:
    """A gate passed before a later edit must not establish success."""
    events = [
        ToolEvent(
            name="deterministic_search_replace",
            args={"file_path": "src/a.ts"},
            content="SUCCESS: Modified",
        ),
        ToolEvent(
            name="validate_workaround",
            args={},
            content="SUCCESS: Workaround validation gate passed",
        ),
        ToolEvent(
            name="deterministic_search_replace",
            args={"file_path": "src/a.ts"},
            content="SUCCESS: Modified again",
        ),
    ]

    assert has_successful_validation_gate(events) is False
    assert (
        has_successful_validation_gate(
            events[:2],
        )
        is True
    )


def test_validation_gate_can_validate_replayed_edits() -> None:
    """A retry may validate edits replayed before the current runtime trace."""
    events = [
        ToolEvent(
            name="validate_workaround",
            args={},
            content="SUCCESS: Workaround validation gate passed",
        )
    ]

    assert has_successful_validation_gate(events) is False
    assert has_successful_validation_gate(events, has_prior_edits=True) is True


def test_successful_workaround_edit_injects_immediate_validation_instruction() -> None:
    edit_tool = MagicMock()
    edit_tool.name = "deterministic_search_replace"
    edit_tool.invoke.return_value = "SUCCESS: File modified: src/auth.ts"

    bound_llm = MagicMock()
    bound_llm.invoke.side_effect = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": edit_tool.name,
                    "args": {"file_path": "src/auth.ts"},
                    "id": "edit-1",
                }
            ],
        ),
        AIMessage(content="I will validate the patch now."),
    ]
    llm = MagicMock()
    llm.bind_tools.return_value = bound_llm

    result = run_bounded_subagent_loop(
        llm,
        [edit_tool],
        [HumanMessage(content="Execute the planned patch.")],
        set(),
        execution_state={"phase": "VALIDATE"},
    )

    assert result.errors == []
    second_turn = bound_llm.invoke.call_args_list[1].args[0]
    assert any(
        isinstance(message, HumanMessage)
        and "next action must be validate_workaround" in message.content
        for message in second_turn
    )
