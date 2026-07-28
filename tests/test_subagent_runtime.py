"""Regression tests for the shared specialist-subagent runtime."""

from __future__ import annotations

from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage

from remediation_engine.orchestration.subagent_runtime import (
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
