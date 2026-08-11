"""Regression tests for the shared specialist-subagent runtime."""

from __future__ import annotations

from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage

from remediation_engine.orchestration.subagent_runtime import (
    ToolEvent,
    _validation_gate_recovery_instruction,
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


def test_manifest_tool_calls_are_serialized_and_deferred_within_one_turn() -> None:
    """A batch continues after parallel-looking manifest calls are deferred."""
    modify_tool = MagicMock()
    modify_tool.name = "modify_npm_dependency"
    modify_tool.invoke.side_effect = [
        "SUCCESS: Natively updated dependencies.lodash to 4.17.21 in package.json.",
        "SUCCESS: Natively updated dependencies.axios to 1.7.4 in package.json.",
    ]
    validate_tool = MagicMock()
    validate_tool.name = "validate_manifest_sync"
    validate_tool.invoke.side_effect = [
        "SUCCESS: Manifest synchronization succeeded for package 'lodash'.",
        "SUCCESS: Manifest synchronization succeeded for package 'axios'.",
    ]

    responses = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": modify_tool.name,
                    "args": {"package_name": "lodash"},
                    "id": "modify-lodash",
                },
                {
                    "name": modify_tool.name,
                    "args": {"package_name": "axios"},
                    "id": "modify-axios-deferred",
                },
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": validate_tool.name,
                    "args": {"package_name": "lodash"},
                    "id": "validate-lodash",
                }
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": modify_tool.name,
                    "args": {"package_name": "axios"},
                    "id": "modify-axios",
                }
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": validate_tool.name,
                    "args": {"package_name": "axios"},
                    "id": "validate-axios",
                }
            ],
        ),
        AIMessage(content="Batch complete."),
    ]
    bound_llm = MagicMock()
    bound_llm.invoke.side_effect = responses
    llm = MagicMock()
    llm.bind_tools.return_value = bound_llm

    result = run_bounded_subagent_loop(
        llm,
        [modify_tool, validate_tool],
        [HumanMessage(content="Process the dependency update batch.")],
        set(),
    )

    assert result.errors == []
    assert [event.name for event in result.tool_events] == [
        "modify_npm_dependency",
        "validate_manifest_sync",
        "modify_npm_dependency",
        "validate_manifest_sync",
    ]
    assert modify_tool.invoke.call_count == 2
    assert validate_tool.invoke.call_count == 2
    assert bound_llm.invoke.call_count == 5
    llm.bind_tools.assert_called_once_with(
        [modify_tool, validate_tool],
        parallel_tool_calls=False,
    )

    second_turn = bound_llm.invoke.call_args_list[1].args[0]
    assert any(
        isinstance(message, HumanMessage)
        and "Manifest operation sequencing barrier" in message.content
        for message in second_turn
    )
    first_turn_tool_messages = [
        message
        for message in bound_llm.invoke.call_args_list[1].args[0]
        if message.__class__.__name__ == "ToolMessage"
    ]
    assert any("DEFERRED" in message.content for message in first_turn_tool_messages)


def test_manifest_validation_failure_does_not_stop_later_batch_items() -> None:
    """A failed package validation can be rolled back while the batch advances."""
    modify_tool = MagicMock()
    modify_tool.name = "modify_npm_dependency"
    modify_tool.invoke.side_effect = [
        "SUCCESS: Natively updated dependencies.lodash to 4.17.21 in package.json.",
        "SUCCESS: Natively updated dependencies.axios to 1.7.4 in package.json.",
    ]
    validate_tool = MagicMock()
    validate_tool.name = "validate_manifest_sync"
    validate_tool.invoke.side_effect = [
        "FAILURE: Manifest sync failed for package 'lodash'. Rolled back package 'lodash'.",
        "SUCCESS: Manifest synchronization succeeded for package 'axios'.",
    ]

    responses = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": modify_tool.name,
                    "args": {"package_name": "lodash"},
                    "id": "modify-lodash",
                }
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": validate_tool.name,
                    "args": {"package_name": "lodash"},
                    "id": "validate-lodash",
                }
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": modify_tool.name,
                    "args": {"package_name": "axios"},
                    "id": "modify-axios",
                }
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": validate_tool.name,
                    "args": {"package_name": "axios"},
                    "id": "validate-axios",
                }
            ],
        ),
        AIMessage(content="Lodash surrendered; axios completed."),
    ]
    bound_llm = MagicMock()
    bound_llm.invoke.side_effect = responses
    llm = MagicMock()
    llm.bind_tools.return_value = bound_llm

    result = run_bounded_subagent_loop(
        llm,
        [modify_tool, validate_tool],
        [HumanMessage(content="Process the dependency update batch.")],
        set(),
    )

    assert result.errors == []
    assert modify_tool.invoke.call_count == 2
    assert validate_tool.invoke.call_count == 2
    assert [event.name for event in result.tool_events] == [
        "modify_npm_dependency",
        "validate_manifest_sync",
        "modify_npm_dependency",
        "validate_manifest_sync",
    ]
    assert "Rolled back package 'lodash'" in result.tool_events[1].content
    assert result.tool_events[-1].content.startswith("SUCCESS:")


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


def test_validation_recovery_explains_checkpoint_behavior() -> None:
    """Tell the worker which edit state each validation outcome leaves behind."""
    instruction = _validation_gate_recovery_instruction(
        'FAILURE: targeted test failed\nJSON: {"overall_status":"CODE_FAILURE"}'
    )

    assert "entire pending edit set has been reverted" in instruction
    assert "re-include every required change" in instruction
    assert "Previously validated edits remain" in instruction
    assert "pending edit set is retained" in instruction
    assert "do not re-apply it" in instruction


def test_invalid_validation_requests_do_not_consume_gate_budget() -> None:
    """Preflight-invalid validation requests are tracked separately from gate runs."""
    validate_tool = MagicMock()
    validate_tool.name = "validate_workaround"

    responses = [
        AIMessage(
            content="",
            tool_calls=[{"name": validate_tool.name, "args": {}, "id": "invalid-1"}],
        ),
        AIMessage(
            content="",
            tool_calls=[{"name": validate_tool.name, "args": {}, "id": "invalid-2"}],
        ),
        AIMessage(
            content="",
            tool_calls=[{"name": validate_tool.name, "args": {}, "id": "gate-1"}],
        ),
        AIMessage(content="I will revise the patch after the gate failure."),
    ]
    validate_tool.invoke.side_effect = [
        "ERROR: [INVALID_RUNTIME_SMOKE] [INVALID_VALIDATION_INPUT] choose a source module",
        "ERROR: [INVALID_VALIDATION_INPUT] targeted test path could not be verified",
        'FAILURE: Workaround validation gate \'syntax\' failed.\nJSON: {"overall_status":"CODE_FAILURE"}',
    ]
    bound_llm = MagicMock()
    bound_llm.invoke.side_effect = responses
    llm = MagicMock()
    llm.bind_tools.return_value = bound_llm
    execution_state = {"validation_calls": 0, "validation_input_errors": 0}

    result = run_bounded_subagent_loop(
        llm,
        [validate_tool],
        [HumanMessage(content="Validate the patch.")],
        set(),
        execution_state=execution_state,
    )

    assert result.errors == []
    assert execution_state["validation_calls"] == 1
    assert execution_state["validation_input_errors"] == 2


def test_repeated_invalid_validation_requests_have_separate_bound() -> None:
    """Malformed validation requests cannot loop forever without gate attempts."""
    validate_tool = MagicMock()
    validate_tool.name = "validate_workaround"
    bound_llm = MagicMock()
    bound_llm.invoke.side_effect = [
        AIMessage(
            content="",
            tool_calls=[{"name": validate_tool.name, "args": {}, "id": f"invalid-{idx}"}],
        )
        for idx in range(1, 4)
    ]
    llm = MagicMock()
    llm.bind_tools.return_value = bound_llm
    validate_tool.invoke.return_value = (
        "ERROR: [INVALID_VALIDATION_INPUT] targeted test path could not be verified"
    )
    execution_state = {"validation_calls": 0, "validation_input_errors": 0}

    result = run_bounded_subagent_loop(
        llm,
        [validate_tool],
        [HumanMessage(content="Validate the patch.")],
        set(),
        execution_state=execution_state,
    )

    assert any("VALIDATION_INPUT_LIMIT_REACHED" in error for error in result.errors)
    assert execution_state["validation_calls"] == 0
    assert execution_state["validation_input_errors"] == 3


def test_repeated_invalid_validation_target_stops_before_third_retry() -> None:
    """An identical invalid smoke/test selection cannot consume more loop turns."""
    validate_tool = MagicMock()
    validate_tool.name = "validate_workaround"
    bound_llm = MagicMock()
    bound_llm.invoke.side_effect = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": validate_tool.name,
                    "args": {
                        "modified_files": ["routes/order.ts"],
                        "runtime_smoke_file": "app.ts",
                        "targeted_test_file": "test/server",
                    },
                    "id": f"invalid-target-{idx}",
                }
            ],
        )
        for idx in range(1, 4)
    ]
    llm = MagicMock()
    llm.bind_tools.return_value = bound_llm
    validate_tool.invoke.return_value = (
        "ERROR: [INVALID_VALIDATION_INPUT] Targeted test 'test/server' must be a source test file"
    )
    execution_state = {"validation_calls": 0, "validation_input_errors": 0}

    result = run_bounded_subagent_loop(
        llm,
        [validate_tool],
        [HumanMessage(content="Validate the patch.")],
        set(),
        execution_state=execution_state,
    )

    assert any("VALIDATION_INPUT_RETRY_LOOP" in error for error in result.errors)
    assert execution_state["validation_calls"] == 0
    assert execution_state["validation_input_errors"] == 2
    assert bound_llm.invoke.call_count == 2


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
