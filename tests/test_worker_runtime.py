"""Focused tests for worker runtime."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from remediation_engine.contracts.schemas import (
    QACriticLLMOutput,
    ScratchpadEntry,
    ScratchpadScope,
    WorkaroundExecutionPhase,
)
from remediation_engine.runtime.agent_runtime import (
    DEFAULT_COMPACTION_INTERVAL,
    MAX_SCRATCHPAD_CHARS,
    PHASE_TOOL_REGISTRY,
    ContextManager,
    ScratchpadMemory,
    ToolEvent,
    _infer_changed_files,
    _manifest_retry_recovery_instruction,
    _preferred_removal_tool_name,
    _response_text_content,
    _validation_gate_recovery_instruction,
    compact_conversation,
    create_chat_model,
    get_phase_prompt,
    get_tools_for_phase,
    has_successful_validation_gate,
    run_bounded_subagent_loop,
)
from remediation_engine.settings import AppSettings


class _FakeTool:
    def __init__(self, name: str, result: str, mutate=None) -> None:
        self.name = name
        self.result = result
        self.mutate = mutate
        self.calls: list[dict] = []

    def invoke(self, args: dict) -> str:
        self.calls.append(dict(args))
        if self.mutate is not None:
            self.mutate(args)
        return self.result


class _BoundModel:
    def __init__(self, owner: _ScriptedModel, names: list[str]) -> None:
        self.owner = owner
        self.names = names

    def invoke(self, messages):
        self.owner.seen_messages.append(list(messages))
        if not self.owner.responses:
            raise AssertionError("scripted model exhausted")
        return self.owner.responses.pop(0)


class _ScriptedModel:
    def __init__(self, responses: list[AIMessage]) -> None:
        self.responses = list(responses)
        self.bind_history: list[list[str]] = []
        self.seen_messages: list[list] = []

    def bind_tools(self, tools, **_kwargs):
        names = [str(tool.name) for tool in tools]
        self.bind_history.append(names)
        return _BoundModel(self, names)


def _call(name: str, args: dict | None = None, call_id: str | None = None) -> dict:
    return {
        "name": name,
        "args": args or {},
        "id": call_id or f"call-{name}",
        "type": "tool_call",
    }


def _response(*calls: dict) -> AIMessage:
    return AIMessage(content="", tool_calls=list(calls))


def _toolbelt(*tools: _FakeTool) -> list[_FakeTool]:
    return list(tools)


def test_scratchpad_entry_is_frozen_and_has_safe_defaults() -> None:
    entry = ScratchpadEntry(phase=WorkaroundExecutionPhase.INVESTIGATE, round_number=1)

    assert entry.key_findings == []
    assert entry.files_inspected == []
    with pytest.raises((TypeError, ValueError)):
        entry.round_number = 2  # type: ignore[misc]
    with pytest.raises(ValueError):
        ScratchpadEntry.model_validate(
            {"phase": "INVESTIGATE", "round_number": 1, "unexpected": True}
        )


def test_maven_transaction_events_keep_exact_pom_and_gav_in_scratchpad() -> None:
    memory = ScratchpadMemory()
    memory.update_from_tool_event(
        ToolEvent(
            "modify_and_validate_maven_dependency",
            {
                "package_name": "org.example:library",
                "target_version": "2.4.1",
                "dependency_type": "dependencyManagement",
                "manifest_path": "pom.xml",
            },
            "SUCCESS: Updated org.example:library and synchronized authorized POMs.",
        ),
        WorkaroundExecutionPhase.EXECUTE,
        1,
    )
    memory.update_from_tool_event(
        ToolEvent(
            "remove_no_fix_dependency",
            {
                "requested_package": "org.example:library",
                "manifest_path": "modules/service/pom.xml",
            },
            "SUCCESS: Removed direct Maven dependency org.example:library.",
        ),
        WorkaroundExecutionPhase.EXECUTE,
        2,
    )

    rendered = memory.render()

    assert "org.example:library" in rendered
    assert "dependencyManagement" in rendered
    assert "modules/service/pom.xml" in rendered
    assert "Maven direct dependency removal" in rendered


def test_qa_scratchpad_is_phase_free_and_retains_terminal_evidence() -> None:
    memory = ScratchpadMemory(ScratchpadScope.QA)
    memory.update_from_tool_event(
        ToolEvent(
            "query_qa_logs",
            {"log_type": "install", "view": "errors"},
            "ERESOLVE peer conflict",
        ),
        WorkaroundExecutionPhase.INVESTIGATE,
        1,
    )
    memory.update_from_tool_event(
        ToolEvent(
            "emit_qa_evaluation",
            {
                "task_id": "task-1",
                "passed": False,
                "failure_category": "INSTALL_FAILURE",
                "retry_feedback": "peer conflict",
                "semantic_security_review": None,
                "test_attribution": None,
            },
            (
                "QA evaluation: task=task-1 passed=False "
                "category=INSTALL_FAILURE retry_required=True"
            ),
        ),
        WorkaroundExecutionPhase.INVESTIGATE,
        2,
    )

    assert len(memory.entries) == 2
    assert all(entry.scope == ScratchpadScope.QA for entry in memory.entries)
    assert all(entry.phase is None for entry in memory.entries)
    assert "QA_REVIEW" in memory.render()
    assert "INSTALL_FAILURE" in memory.render()


def test_qa_context_manager_bypasses_workaround_phase_gating() -> None:
    tools = _toolbelt(
        _FakeTool("record_plan", "plan"),
        _FakeTool("read_repository_map", "map"),
        _FakeTool("emit_qa_evaluation", "evaluation"),
    )
    manager = ContextManager(
        tools,
        compaction_interval=3,
        skip_phase_gating=True,
        scratchpad_scope=ScratchpadScope.QA,
    )

    assert manager.skip_phase_gating is True
    assert [tool.name for tool in manager.get_tools_for_phase(WorkaroundExecutionPhase.PLAN)] == [
        "record_plan",
        "read_repository_map",
        "emit_qa_evaluation",
    ]


def test_qa_compaction_occurs_at_round_three_and_preserves_scratchpad() -> None:
    tool = _FakeTool("query_qa_logs", "ERESOLVE peer conflict")
    manager = ContextManager(
        [tool],
        compaction_interval=3,
        skip_phase_gating=True,
        scratchpad_scope=ScratchpadScope.QA,
    )
    conversation = [
        SystemMessage(content="static"),
        HumanMessage(content="review"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "query_qa_logs",
                    "args": {"log_type": "install"},
                    "id": "qa-call",
                }
            ],
        ),
        ToolMessage(content="ERESOLVE peer conflict", name="query_qa_logs", tool_call_id="qa-call"),
    ]
    manager.update_scratchpad(
        ToolEvent("query_qa_logs", {"log_type": "install"}, "ERESOLVE peer conflict"),
        WorkaroundExecutionPhase.INVESTIGATE,
        1,
    )

    before_boundary = manager.compact_conversation(conversation, 2)
    at_boundary = manager.compact_conversation(conversation, 3)
    assert at_boundary[4].content.startswith("[COMPACTED] query_qa_logs:")
    assert before_boundary[3].content == conversation[3].content
    assert at_boundary[1].additional_kwargs["remediation_engine_scratchpad"] is True
    assert "QA_REVIEW" in at_boundary[1].content
    at_round_six = manager.compact_conversation(at_boundary, 6)
    assert at_round_six[1].content == at_boundary[1].content
    assert at_round_six[4].content == at_boundary[4].content


def test_qa_scratchpad_extracts_bounded_file_diff_and_log_facts() -> None:
    memory = ScratchpadMemory(ScratchpadScope.QA)
    events = [
        ToolEvent(
            "list_changed_files",
            {},
            "- package.json\n- src/auth.ts\n",
        ),
        ToolEvent(
            "generate_workspace_diff",
            {},
            "diff --git a/src/auth.ts b/src/auth.ts\n--- a/src/auth.ts\n+++ b/src/auth.ts",
        ),
        ToolEvent(
            "read_file_context",
            {"file_path": "src/auth.ts", "start_line": 1, "end_line": 8},
            "const secret = 'do-not-retain'\nexport function auth() {}",
        ),
    ]
    for round_number, event in enumerate(events, start=1):
        memory.update_from_tool_event(event, None, round_number)

    rendered = memory.render()

    assert "package.json" in rendered
    assert "src/auth.ts" in rendered
    assert "do-not-retain" not in rendered
    assert "read src/auth.ts lines 1-8" in rendered


def test_registry_filters_in_builder_order_and_missing_optional_tools() -> None:
    names = [
        "record_plan",
        "read_repository_map",
        "read_workspace_file",
        "search_codebase_pattern",
        "inspect_ast_symbol",
        "search_web",
        "read_web_page",
        "deterministic_apply_edit_set",
        "revert_workspace_file",
        "validate_workaround",
        "record_targeted_test_substitution",
        "remove_no_fix_dependency",
    ]
    tools = [SimpleNamespace(name=name) for name in names]

    assert set(PHASE_TOOL_REGISTRY) == set(WorkaroundExecutionPhase)
    for phase in WorkaroundExecutionPhase:
        expected = [name for name in names if name in PHASE_TOOL_REGISTRY[phase]]
        assert [tool.name for tool in get_tools_for_phase(phase, tools)] == expected

    assert "remove_no_fix_dependency" in [
        tool.name for tool in get_tools_for_phase(WorkaroundExecutionPhase.EXECUTE, tools)
    ]
    assert "remove_no_fix_dependency" in [
        tool.name for tool in get_tools_for_phase(WorkaroundExecutionPhase.VALIDATE, tools)
    ]
    assert [tool.name for tool in get_tools_for_phase(None, tools)] == [
        name for name in names if name in PHASE_TOOL_REGISTRY[WorkaroundExecutionPhase.INVESTIGATE]
    ]


def test_python_manifest_tools_are_registered_for_their_execution_phases() -> None:
    python_removal = SimpleNamespace(name="remove_no_fix_python_dependency")
    python_update = SimpleNamespace(name="modify_and_validate_python_dependency")

    assert get_tools_for_phase(
        WorkaroundExecutionPhase.EXECUTE,
        [python_removal, python_update],
    ) == [python_removal, python_update]
    assert get_tools_for_phase(
        WorkaroundExecutionPhase.VALIDATE,
        [python_removal, python_update],
    ) == [python_removal]
    assert (
        get_tools_for_phase(
            WorkaroundExecutionPhase.INVESTIGATE,
            [python_removal, python_update],
        )
        == []
    )
    node_workaround_tools = [SimpleNamespace(name="remove_no_fix_dependency")]
    assert (
        get_tools_for_phase(
            WorkaroundExecutionPhase.EXECUTE,
            node_workaround_tools,
        )
        == node_workaround_tools
    )


def test_phase_prompts_are_action_oriented_and_preserve_optional_context() -> None:
    assert DEFAULT_COMPACTION_INTERVAL == 4
    assert "do not edit" in get_phase_prompt("INVESTIGATE").lower()
    assert "record_plan" in get_phase_prompt("PLAN")
    assert "atomic edit" in get_phase_prompt("EXECUTE")
    assert "read_repository_map" in get_phase_prompt("VALIDATE")
    assert "either operation may happen first" in get_phase_prompt("VALIDATE")
    assert "next action must be validate_workaround" in get_phase_prompt("VALIDATE")
    assert get_phase_prompt("PLAN", "original task") == (
        "original task\n\n" + get_phase_prompt("PLAN")
    )
    with pytest.raises(ValueError):
        get_phase_prompt("not-a-phase")


def test_scratchpad_extracts_bounded_facts_and_deduplicates() -> None:
    memory = ScratchpadMemory()
    event = ToolEvent(
        name="read_workspace_file",
        args={"file_path": "\\src\\app.js", "start_line": 2, "end_line": 8},
        content="const password = super-secret\n" + ("body " * 500),
    )
    memory.update_from_tool_event(event, WorkaroundExecutionPhase.INVESTIGATE, 1)
    memory.update_from_tool_event(event, WorkaroundExecutionPhase.INVESTIGATE, 2)
    memory.update_from_tool_event(
        ToolEvent(
            name="inspect_ast_symbol",
            args={
                "file_path": "src/app.js",
                "symbol_name": "handle",
                "node_type": "Function",
                "line": 12,
            },
            content="location 12",
        ),
        WorkaroundExecutionPhase.INVESTIGATE,
        3,
    )
    memory.update_from_tool_event(
        ToolEvent(
            name="inspect_ast_symbol",
            args={
                "file_path": "src/app.js",
                "symbol_name": "handle",
                "node_type": "Function",
                "line": 99,
            },
            content="new location",
        ),
        WorkaroundExecutionPhase.INVESTIGATE,
        4,
    )

    rendered = memory.render()
    assert "src/app.js" in rendered
    assert "lines 2-8" in rendered
    assert "super-secret" not in rendered
    assert rendered.count("AST src/app.js::handle") == 1
    assert len(memory.entries) == 2
    assert len(rendered) <= MAX_SCRATCHPAD_CHARS


def test_scratchpad_retains_latest_critical_outcome_within_cap() -> None:
    memory = ScratchpadMemory()
    for index in range(80):
        pattern = f"pattern-{index}-" + ("x" * 150)
        memory.update_from_tool_event(
            ToolEvent(
                name="search_codebase_pattern",
                args={"directory": "src", "search_pattern": pattern},
                content=f"src/file-{index}.js: " + ("result " * 40),
            ),
            WorkaroundExecutionPhase.INVESTIGATE,
            index + 1,
        )
    memory.update_from_tool_event(
        ToolEvent(
            name="validate_workaround",
            args={},
            content=(
                "SUCCESS: validation\n"
                'JSON: {"overall_status":"PASS","validated_files":["src/latest.js"],'
                '"infrastructure_diagnostics":"latest-critical-diagnostic"}'
            ),
        ),
        WorkaroundExecutionPhase.VALIDATE,
        81,
    )

    rendered = memory.render()
    assert len(rendered) <= MAX_SCRATCHPAD_CHARS
    assert "latest-critical-diagnostic" in rendered
    assert "src/latest.js" in rendered
    assert "older scratchpad facts omitted" in rendered


def test_compaction_preserves_rounds_metadata_and_does_not_mutate_input() -> None:
    system = SystemMessage(content="system instructions")
    human = HumanMessage(content="task")
    old = ToolMessage(
        content="old file body " * 100,
        name="read_workspace_file",
        tool_call_id="old-call",
        additional_kwargs={"trace": "keep"},
    )
    old_search = ToolMessage(
        content="old search body " * 100,
        name="search_codebase_pattern",
        tool_call_id="search-call",
    )
    critical = ToolMessage(
        content="plan must remain complete",
        name="record_plan",
        tool_call_id="plan-call",
    )
    previous = ToolMessage(
        content="previous round body " * 100,
        name="read_web_page",
        tool_call_id="previous-call",
    )
    current = ToolMessage(
        content="current round body " * 100,
        name="read_workspace_file",
        tool_call_id="current-call",
    )
    conversation = [
        system,
        human,
        AIMessage(content="reason-1"),
        old,
        critical,
        AIMessage(content="reason-2"),
        old_search,
        AIMessage(content="reason-3"),
        previous,
        AIMessage(content="reason-4"),
        current,
    ]
    compacted = compact_conversation(conversation, current_round=4)

    assert len(compacted) == len(conversation)
    assert compacted[0] is system
    assert compacted[1] is human
    assert compacted[2].content == "reason-1"
    assert old.content.startswith("old file body")
    assert compacted[3].content.startswith("[COMPACTED] read_workspace_file:")
    assert compacted[3].tool_call_id == "old-call"
    assert compacted[3].additional_kwargs["trace"] == "keep"
    assert compacted[4] is critical
    assert compacted[6].content.startswith("[COMPACTED] search_codebase_pattern:")
    assert compacted[8] is previous
    assert compacted[10] is current
    assert sum(len(str(message.content)) for message in compacted) < sum(
        len(str(message.content)) for message in conversation
    )
    again = compact_conversation(compacted, current_round=4)
    assert [message.content for message in again] == [message.content for message in compacted]


def test_manager_inserts_and_replaces_only_its_scratchpad_marker() -> None:
    manager = ContextManager([], compaction_interval=1)
    conversation = [SystemMessage(content="static"), HumanMessage(content="task")]
    manager.update_scratchpad(
        ToolEvent("read_repository_map", {}, "SUCCESS: map"),
        WorkaroundExecutionPhase.INVESTIGATE,
        1,
    )
    first = manager.compact_conversation(conversation, 1)
    assert [type(message) for message in first[:2]] == [SystemMessage, SystemMessage]
    assert first[0].content == "static"
    assert "SUCCESS: map" in first[1].content

    manager.update_scratchpad(
        ToolEvent("record_plan", {"security_invariant": "safe"}, "SUCCESS: plan"),
        WorkaroundExecutionPhase.PLAN,
        2,
    )
    second = manager.compact_conversation(first, 2)
    assert len(second) == len(first)
    assert second[0].content == "static"
    assert second[1].content != first[1].content
    assert second[1].additional_kwargs["remediation_engine_scratchpad"] is True


def test_runtime_rebinds_through_lifecycle_and_keeps_tool_events_raw() -> None:
    state: dict[str, object] = {"phase": "INVESTIGATE", "local_investigation_complete": False}

    def mark_local(_args):
        state["local_investigation_complete"] = True

    def record(_args):
        state.update({"phase": "EXECUTE", "planned_replacements": [{"file_path": "src/a.js"}]})

    def edit(_args):
        state["phase"] = "VALIDATE"

    tools = _toolbelt(
        _FakeTool("record_plan", "SUCCESS: plan", record),
        _FakeTool("read_repository_map", "SUCCESS: repository map"),
        _FakeTool("read_workspace_file", "SUCCESS: file", mark_local),
        _FakeTool("search_codebase_pattern", "src/a.js:1: match", mark_local),
        _FakeTool(
            "deterministic_apply_edit_set",
            'SUCCESS: edit\nJSON: {"affected_files":["src/a.js"]}',
            edit,
        ),
        _FakeTool(
            "validate_workaround",
            'SUCCESS: validation\nJSON: {"overall_status":"PASS","validated_files":["src/a.js"]}',
        ),
    )
    model = _ScriptedModel(
        [
            _response(_call("read_repository_map")),
            _response(_call("read_workspace_file", {"file_path": "src/a.js"})),
            _response(_call("record_plan", {"affected_files": ["src/a.js"]})),
            _response(_call("deterministic_apply_edit_set", {"replacements": []})),
            _response(_call("validate_workaround", {"modified_files": ["src/a.js"]})),
            AIMessage(content="done"),
        ]
    )
    manager = ContextManager(tools, compaction_interval=1)
    result = run_bounded_subagent_loop(
        model,
        tools,
        [SystemMessage(content="system"), HumanMessage(content="task")],
        set(),
        execution_state=state,
        context_manager=manager,
    )

    assert model.bind_history == [
        ["record_plan", "read_repository_map", "read_workspace_file", "search_codebase_pattern"],
        ["record_plan", "read_workspace_file", "search_codebase_pattern"],
        ["read_workspace_file", "search_codebase_pattern", "deterministic_apply_edit_set"],
        ["read_repository_map", "read_workspace_file", "validate_workaround"],
    ]
    assert [event.name for event in result.tool_events] == [
        "read_repository_map",
        "read_workspace_file",
        "record_plan",
        "deterministic_apply_edit_set",
        "validate_workaround",
    ]
    assert result.tool_events[-1].content.startswith("SUCCESS: validation")
    assert any(
        any(
            getattr(message, "additional_kwargs", {}).get("remediation_engine_scratchpad")
            for message in messages
        )
        for messages in model.seen_messages
    )
    assert state["phase"] == "VALIDATE"
    assert result.errors == []


def test_same_response_cannot_bypass_phase_snapshot() -> None:
    state = {"phase": "PLAN", "local_investigation_complete": True}
    plan = _FakeTool(
        "record_plan", "SUCCESS: plan", lambda _args: state.update({"phase": "EXECUTE"})
    )
    edit = _FakeTool("deterministic_apply_edit_set", "SUCCESS: edit")
    tools = _toolbelt(plan, edit)
    model = _ScriptedModel(
        [
            _response(_call("record_plan"), _call("deterministic_apply_edit_set")),
            AIMessage(content="done"),
        ]
    )

    result = run_bounded_subagent_loop(
        model,
        tools,
        [HumanMessage(content="task")],
        set(),
        execution_state=state,
        context_manager=ContextManager(tools),
    )

    assert edit.calls == []
    assert any("[PHASE_VIOLATION]" in event.content for event in result.tool_events)
    assert [event.name for event in result.tool_events] == [
        "record_plan",
        "deterministic_apply_edit_set",
    ]


def test_validation_code_failure_rebinds_to_investigate_for_fresh_inspection() -> None:
    state = {"phase": "EXECUTE", "local_investigation_complete": True}

    def failed_validation(_args):
        state.update({"phase": "INVESTIGATE", "local_investigation_complete": False})

    read = _FakeTool("read_repository_map", "SUCCESS: fresh map")
    edit = _FakeTool(
        "deterministic_apply_edit_set",
        "SUCCESS: edit",
        lambda _args: state.update({"phase": "VALIDATE"}),
    )
    validate = _FakeTool("validate_workaround", "FAILURE: CODE_FAILURE", failed_validation)
    tools = _toolbelt(read, edit, validate)
    model = _ScriptedModel(
        [
            _response(_call("deterministic_apply_edit_set")),
            _response(_call("validate_workaround")),
            _response(_call("read_repository_map")),
            AIMessage(content="stop"),
        ]
    )

    result = run_bounded_subagent_loop(
        model,
        tools,
        [HumanMessage(content="task")],
        set(),
        execution_state=state,
        context_manager=ContextManager(tools),
    )

    assert read.calls == [{}]
    assert state["phase"] == "INVESTIGATE"
    assert state["local_investigation_complete"] is False
    assert model.bind_history[-1] == ["read_repository_map"]
    assert any(event.name == "read_repository_map" for event in result.tool_events)


def test_blocker_cleanup_uses_full_tool_map_when_validate_hides_revert() -> None:
    state = {"phase": "EXECUTE"}
    revert = _FakeTool("revert_workspace_file", "SUCCESS: reverted")
    edit = _FakeTool(
        "deterministic_apply_edit_set",
        'SUCCESS: edit\nJSON: {"affected_files":["src/a.js"]}',
        lambda _args: state.update({"phase": "VALIDATE"}),
    )
    validate = _FakeTool("validate_workaround", "BLOCKED: Sandbox is not running")
    tools = _toolbelt(edit, validate, revert)
    model = _ScriptedModel(
        [_response(_call("deterministic_apply_edit_set")), _response(_call("validate_workaround"))]
    )

    result = run_bounded_subagent_loop(
        model,
        tools,
        [HumanMessage(content="task")],
        set(),
        execution_state=state,
        context_manager=ContextManager(tools),
    )

    assert revert.calls == [{"file_path": "src/a.js"}]
    assert ["revert_workspace_file"] not in model.bind_history
    assert result.changed_files == []
    assert result.errors


def test_python_workaround_selects_python_removal_tool_name():
    assert (
        _preferred_removal_tool_name({"validate_python_workaround": object()})
        == "remove_no_fix_python_dependency"
    )


def test_response_content_extracts_text_without_serializing_reasoning_blocks():
    """Responses content preserves plain text and drops opaque provider blocks."""
    content = [
        {"type": "reasoning", "reasoning": "PRIVATE REASONING PAYLOAD"},
        {"type": "encrypted_content", "encrypted_content": "OPAQUE ENCRYPTED PAYLOAD"},
        {"type": "text", "text": "Planning answers: update lodash to 4.17.21."},
    ]

    extracted = _response_text_content(content)

    assert extracted == "Planning answers: update lodash to 4.17.21."
    assert "PRIVATE REASONING PAYLOAD" not in extracted
    assert "OPAQUE ENCRYPTED PAYLOAD" not in extracted


def test_list_response_text_still_marks_planning_answers_as_submitted():
    """Text extraction keeps planning-state detection working for Responses blocks."""
    bound_llm = MagicMock()
    bound_llm.invoke.return_value = AIMessage(
        content=[
            {"type": "reasoning", "reasoning": "PRIVATE REASONING PAYLOAD"},
            {"type": "text", "text": "Planning answers: update lodash to 4.17.21."},
        ]
    )
    llm = MagicMock()
    llm.bind_tools.return_value = bound_llm
    planning_state = {"submitted": False}

    result = run_bounded_subagent_loop(
        llm,
        [],
        [HumanMessage(content="Plan the update.")],
        set(),
        planning_state=planning_state,
    )

    assert result.final_text == "Planning answers: update lodash to 4.17.21."
    assert planning_state["submitted"] is True


def test_structured_terminal_output_is_returned_without_free_text() -> None:
    """The QA evaluator returns only validated structured data."""
    terminal_tool = MagicMock()
    terminal_tool.name = "emit_qa_evaluation"

    bound_llm = MagicMock()
    bound_llm.invoke.return_value = AIMessage(
        content="",
        tool_calls=[
            {
                "name": terminal_tool.name,
                "args": {"task_id": "g1", "passed": True},
                "id": "final-1",
            }
        ],
    )
    llm = MagicMock()
    llm.bind_tools.return_value = bound_llm

    result = run_bounded_subagent_loop(
        llm,
        [terminal_tool],
        [HumanMessage(content="Review the task.")],
        set(),
        structured_output_model=QACriticLLMOutput,
        structured_output_tool_name=terminal_tool.name,
    )

    assert isinstance(result.structured_output, QACriticLLMOutput)
    assert result.structured_output.task_id == "g1"
    assert result.final_text == ""
    terminal_tool.invoke.assert_not_called()


def test_qa_bypass_binds_all_tools_and_records_terminal_without_invoking_body() -> None:
    review_tool = MagicMock()
    review_tool.name = "query_qa_logs"
    terminal_tool = MagicMock()
    terminal_tool.name = "emit_qa_evaluation"
    bound_llm = MagicMock()
    bound_llm.invoke.return_value = AIMessage(
        content="",
        tool_calls=[
            {
                "name": terminal_tool.name,
                "args": {"task_id": "g1", "passed": True},
                "id": "terminal-1",
            }
        ],
    )
    llm = MagicMock()
    llm.bind_tools.return_value = bound_llm
    manager = ContextManager(
        [review_tool, terminal_tool],
        compaction_interval=3,
        skip_phase_gating=True,
        scratchpad_scope=ScratchpadScope.QA,
    )

    result = run_bounded_subagent_loop(
        llm,
        [review_tool, terminal_tool],
        [HumanMessage(content="Review the task.")],
        set(),
        execution_state={"phase": WorkaroundExecutionPhase.VALIDATE.value},
        structured_output_model=QACriticLLMOutput,
        structured_output_tool_name=terminal_tool.name,
        context_manager=manager,
    )

    assert [tool.name for tool in llm.bind_tools.call_args.args[0]] == [
        "query_qa_logs",
        "emit_qa_evaluation",
    ]
    assert result.errors == []
    assert [event.name for event in result.tool_events] == ["emit_qa_evaluation"]
    assert terminal_tool.invoke.call_count == 0
    assert manager.scratchpad.entries[-1].scope == ScratchpadScope.QA
    assert "task_id=g1" in manager.scratchpad.entries[-1].critical_outcome


def test_invalid_structured_terminal_output_is_rejected_and_retried() -> None:
    """Malformed terminal fields never become a QA verdict."""
    terminal_tool = MagicMock()
    terminal_tool.name = "emit_qa_evaluation"

    bound_llm = MagicMock()
    bound_llm.invoke.side_effect = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": terminal_tool.name,
                    "args": {"task_id": "g1", "passed": False},
                    "id": "invalid-1",
                }
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": terminal_tool.name,
                    "args": {
                        "task_id": "g1",
                        "passed": False,
                        "failure_category": "breaking_change",
                        "retry_feedback": "Fix the failing test.",
                    },
                    "id": "final-1",
                }
            ],
        ),
    ]
    llm = MagicMock()
    llm.bind_tools.return_value = bound_llm

    result = run_bounded_subagent_loop(
        llm,
        [terminal_tool],
        [HumanMessage(content="Review the task.")],
        set(),
        structured_output_model=QACriticLLMOutput,
        structured_output_tool_name=terminal_tool.name,
    )

    assert isinstance(result.structured_output, QACriticLLMOutput)
    assert result.structured_output.failure_category.value == "breaking_change"
    assert any("validation failed" in error for error in result.errors)
    assert bound_llm.invoke.call_count == 2
    terminal_tool.invoke.assert_not_called()


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


def test_validation_pass_terminates_without_a_max_round_error() -> None:
    """A valid workaround PASS closes the bounded worker loop immediately."""
    edit_tool = MagicMock()
    edit_tool.name = "deterministic_search_replace"
    edit_tool.invoke.return_value = "SUCCESS: File modified: src/auth.ts"
    validate_tool = MagicMock()
    validate_tool.name = "validate_workaround"
    validate_tool.invoke.return_value = (
        "SUCCESS: Workaround validation gate passed. JSON: "
        '{"overall_status":"PASS","validated_files":["src/auth.ts"]}'
    )

    bound_llm = MagicMock()
    bound_llm.invoke.side_effect = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": edit_tool.name,
                    "args": {"file_path": "src/auth.ts"},
                    "id": "edit-pass-1",
                }
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": validate_tool.name,
                    "args": {
                        "modified_files": ["src/auth.ts"],
                        "runtime_smoke_file": "src/auth.ts",
                        "targeted_test_file": "tests/auth.test.ts",
                    },
                    "id": "validate-pass-1",
                }
            ],
        ),
    ]
    llm = MagicMock()
    llm.bind_tools.return_value = bound_llm

    result = run_bounded_subagent_loop(
        llm,
        [edit_tool, validate_tool],
        [HumanMessage(content="Apply and validate the patch.")],
        set(),
        execution_state={"validation_calls": 0},
    )

    assert bound_llm.invoke.call_count == 2
    assert result.terminal_validation_passed is True
    assert result.errors == []
    assert "MAX_SUBAGENT_TOOL_CALL_ROUNDS" not in " ".join(result.errors)


def test_python_validation_pass_is_terminal_and_counts_the_executed_gate() -> None:
    edit_tool = MagicMock()
    edit_tool.name = "deterministic_search_replace"
    edit_tool.invoke.return_value = "SUCCESS: File modified: src/auth.py"
    validate_tool = MagicMock()
    validate_tool.name = "validate_python_workaround"
    validate_tool.invoke.return_value = (
        "SUCCESS: Workaround validation gate passed. JSON: "
        '{"overall_status":"PASS","validated_files":["src/auth.py"]}'
    )
    bound_llm = MagicMock()
    bound_llm.invoke.side_effect = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": edit_tool.name,
                    "args": {"file_path": "src/auth.py"},
                    "id": "python-edit-pass",
                }
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": validate_tool.name,
                    "args": {
                        "modified_files": ["src/auth.py"],
                        "runtime_smoke_file": "src/auth.py",
                        "targeted_test_file": "tests/test_auth.py",
                    },
                    "id": "python-validate-pass",
                }
            ],
        ),
    ]
    llm = MagicMock()
    llm.bind_tools.return_value = bound_llm
    execution_state = {"validation_calls": 0}

    result = run_bounded_subagent_loop(
        llm,
        [edit_tool, validate_tool],
        [HumanMessage(content="Apply and validate the Python patch.")],
        set(),
        execution_state=execution_state,
    )

    assert bound_llm.invoke.call_count == 2
    assert result.terminal_validation_passed is True
    assert result.errors == []
    assert execution_state["validation_calls"] == 1
    assert execution_state["validation_passed"] is True
    assert has_successful_validation_gate(result.tool_events) is True


def test_python_validation_names_share_preflight_and_gate_attempt_limits() -> None:
    validate_tool = MagicMock()
    validate_tool.name = "validate_python_workaround"
    validate_tool.invoke.side_effect = [
        "ERROR: [INVALID_VALIDATION_INPUT] choose a valid source module",
        "FAILURE: [INVALID_RUNTIME_SMOKE] smoke import failed",
        'FAILURE: syntax failed\nJSON: {"overall_status":"CODE_FAILURE"}',
        'FAILURE: pytest failed\nJSON: {"overall_status":"CODE_FAILURE"}',
    ]
    args = {
        "modified_files": ["src/app.py"],
        "runtime_smoke_file": "src/app.py",
        "targeted_test_file": "tests/test_app.py",
    }
    bound_llm = MagicMock()
    bound_llm.invoke.side_effect = [
        AIMessage(
            content="",
            tool_calls=[{"name": validate_tool.name, "args": {}, "id": "python-invalid"}],
        ),
        *[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": validate_tool.name,
                        "args": args,
                        "id": f"python-gate-{index}",
                    }
                ],
            )
            for index in range(1, 4)
        ],
    ]
    llm = MagicMock()
    llm.bind_tools.return_value = bound_llm
    execution_state = {"validation_calls": 0, "validation_input_errors": 0}

    result = run_bounded_subagent_loop(
        llm,
        [validate_tool],
        [HumanMessage(content="Validate the Python patch.")],
        set(),
        execution_state=execution_state,
    )

    assert bound_llm.invoke.call_count == 4
    assert execution_state["validation_calls"] == 3
    assert execution_state["validation_input_errors"] == 1
    assert any("VALIDATION_LIMIT_REACHED" in error for error in result.errors)
    assert any(
        isinstance(message, HumanMessage) and "validate_python_workaround" in message.content
        for message in bound_llm.invoke.call_args_list[1].args[0]
    )
    assert any(
        isinstance(message, HumanMessage) and "validate_python_workaround" in message.content
        for message in bound_llm.invoke.call_args_list[2].args[0]
    )


def test_python_targeted_test_failure_uses_shared_recovery_feedback() -> None:
    targeted_test = MagicMock()
    targeted_test.name = "run_targeted_python_test"
    targeted_test.invoke.return_value = "FAILURE: Targeted test failed (pytest): tests/test_app.py"
    bound_llm = MagicMock()
    bound_llm.invoke.side_effect = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": targeted_test.name,
                    "args": {"test_file": "tests/test_app.py", "test_name": "test_route"},
                    "id": "python-targeted-failure",
                }
            ],
        ),
        AIMessage(content="I will revise the hypothesis."),
    ]
    llm = MagicMock()
    llm.bind_tools.return_value = bound_llm

    result = run_bounded_subagent_loop(
        llm,
        [targeted_test],
        [HumanMessage(content="Run the targeted Python test.")],
        set(),
    )

    assert result.errors == []
    second_turn = bound_llm.invoke.call_args_list[1].args[0]
    assert any(
        isinstance(message, HumanMessage) and "Exact targeted-test result:" in message.content
        for message in second_turn
    )


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
            tool_calls=[
                {
                    "name": validate_tool.name,
                    "args": {"source_module": "src/app.js"},
                    "id": "invalid-1",
                }
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": validate_tool.name,
                    "args": {"targeted_test": "tests/app.test.js"},
                    "id": "invalid-2",
                }
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": validate_tool.name,
                    "args": {
                        "source_module": "src/app.js",
                        "targeted_test": "tests/app.test.js",
                        "runtime_smoke_file": "src/app.js",
                    },
                    "id": "gate-1",
                }
            ],
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


def test_manifest_tool_calls_are_serialized_and_deferred_within_one_turn() -> None:
    """Only the first combined transaction in a model turn is executed."""
    combined_tool = MagicMock()
    combined_tool.name = "modify_and_validate_npm_dependency"
    combined_tool.invoke.side_effect = [
        "SUCCESS: updated and synchronized lodash",
        "SUCCESS: updated and synchronized axios",
    ]

    bound_llm = MagicMock()
    bound_llm.invoke.side_effect = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": combined_tool.name,
                    "args": {
                        "package_name": "lodash",
                        "target_version": "4.17.21",
                        "dependency_type": "dependencies",
                    },
                    "id": "combined-1",
                },
                {
                    "name": combined_tool.name,
                    "args": {
                        "package_name": "axios",
                        "target_version": "1.7.4",
                        "dependency_type": "dependencies",
                    },
                    "id": "combined-2",
                },
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": combined_tool.name,
                    "args": {
                        "package_name": "axios",
                        "target_version": "1.7.4",
                        "dependency_type": "dependencies",
                    },
                    "id": "combined-3",
                }
            ],
        ),
        AIMessage(content="complete"),
    ]
    llm = MagicMock()
    llm.bind_tools.return_value = bound_llm

    result = run_bounded_subagent_loop(
        llm,
        [combined_tool],
        [HumanMessage(content="Update the dependencies.")],
        set(),
    )

    assert result.errors == []
    assert [event.name for event in result.tool_events] == [
        "modify_and_validate_npm_dependency",
        "modify_and_validate_npm_dependency",
        "modify_and_validate_npm_dependency",
    ]
    assert result.tool_events[1].content.startswith("DEFERRED: manifest tools")
    assert combined_tool.invoke.call_count == 2
    assert combined_tool.invoke.call_args_list[0].args[0] == {
        "package_name": "lodash",
        "target_version": "4.17.21",
        "dependency_type": "dependencies",
    }
    llm.bind_tools.assert_called_once_with([combined_tool], parallel_tool_calls=False)
    assert any(
        isinstance(message, HumanMessage)
        and "Manifest transaction sequencing barrier" in message.content
        for message in bound_llm.invoke.call_args_list[1].args[0]
    )


def test_manifest_transaction_failure_does_not_stop_later_batch_items() -> None:
    """A failed package transaction can be retried without blocking another package."""
    combined_tool = MagicMock()
    combined_tool.name = "modify_and_validate_npm_dependency"
    combined_tool.invoke.side_effect = [
        "ERROR_CODE: MANIFEST_SYNC_FAILED: rolled back lodash",
        "SUCCESS: updated and synchronized lodash",
        "SUCCESS: updated and synchronized axios",
    ]

    bound_llm = MagicMock()
    bound_llm.invoke.side_effect = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": combined_tool.name,
                    "args": {
                        "package_name": "lodash",
                        "target_version": "4.17.21",
                        "dependency_type": "dependencies",
                    },
                    "id": "combined-1",
                }
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": combined_tool.name,
                    "args": {
                        "package_name": "lodash",
                        "target_version": "4.17.22",
                        "dependency_type": "dependencies",
                    },
                    "id": "combined-2",
                }
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": combined_tool.name,
                    "args": {
                        "package_name": "axios",
                        "target_version": "1.7.4",
                        "dependency_type": "dependencies",
                    },
                    "id": "combined-3",
                }
            ],
        ),
        AIMessage(content="lodash retried; axios completed"),
    ]
    llm = MagicMock()
    llm.bind_tools.return_value = bound_llm

    result = run_bounded_subagent_loop(
        llm,
        [combined_tool],
        [HumanMessage(content="Update the dependencies.")],
        set(),
    )

    assert result.errors == []
    assert combined_tool.invoke.call_count == 3
    assert "rolled back lodash" in result.tool_events[0].content
    assert any(
        isinstance(message, HumanMessage)
        and "different value from the Supervisor-approved candidate lists" in message.content
        for message in bound_llm.invoke.call_args_list[1].args[0]
    )


def test_manifest_retry_limit_is_scoped_to_one_package() -> None:
    """Three failed calls surrender one package while another package continues."""
    combined_tool = MagicMock()
    combined_tool.name = "modify_and_validate_npm_dependency"
    combined_tool.invoke.side_effect = [
        "ERROR_CODE: EDIT_FAILED: first package attempt failed",
        "ERROR_CODE: MANIFEST_SYNC_FAILED: second package attempt failed",
        "FAILURE: third package attempt failed",
        "SUCCESS: updated and synchronized axios",
    ]

    bound_llm = MagicMock()
    bound_llm.invoke.side_effect = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": combined_tool.name,
                    "args": {
                        "package_name": "lodash",
                        "target_version": "4.17.21",
                        "dependency_type": "dependencies",
                    },
                    "id": "lodash-1",
                }
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": combined_tool.name,
                    "args": {
                        "package_name": "lodash",
                        "target_version": "4.17.22",
                        "dependency_type": "dependencies",
                    },
                    "id": "lodash-2",
                }
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": combined_tool.name,
                    "args": {
                        "package_name": "lodash",
                        "target_version": "4.17.23",
                        "dependency_type": "dependencies",
                    },
                    "id": "lodash-3",
                }
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": combined_tool.name,
                    "args": {
                        "package_name": "lodash",
                        "target_version": "4.17.24",
                        "dependency_type": "dependencies",
                    },
                    "id": "lodash-4",
                },
                {
                    "name": combined_tool.name,
                    "args": {
                        "package_name": "axios",
                        "target_version": "1.7.4",
                        "dependency_type": "dependencies",
                    },
                    "id": "axios-deferred",
                },
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": combined_tool.name,
                    "args": {
                        "package_name": "axios",
                        "target_version": "1.7.4",
                        "dependency_type": "dependencies",
                    },
                    "id": "axios-1",
                }
            ],
        ),
        AIMessage(content="lodash surrendered; axios completed"),
    ]
    llm = MagicMock()
    llm.bind_tools.return_value = bound_llm

    result = run_bounded_subagent_loop(
        llm,
        [combined_tool],
        [HumanMessage(content="Update the dependencies.")],
        set(),
    )

    assert result.errors == []
    assert combined_tool.invoke.call_count == 4
    assert result.tool_events[3].content.startswith("ERROR_CODE: RETRY_LIMIT_REACHED:")
    assert result.tool_events[4].content.startswith("DEFERRED: manifest tools")
    assert result.tool_events[5].content.startswith("SUCCESS:")
    assert any(
        isinstance(message, HumanMessage) and "exhausted its 3-attempt limit" in message.content
        for message in bound_llm.invoke.call_args_list[4].args[0]
    )


def test_maven_update_transactions_are_serialized_and_retry_by_exact_gav() -> None:
    maven_tool = MagicMock()
    maven_tool.name = "modify_and_validate_maven_dependency"
    maven_tool.invoke.side_effect = [
        "ERROR_CODE: MANIFEST_SYNC_FAILED: Maven dependency resolution failed",
        "SUCCESS: Updated org.example:library to 2.4.1 through Maven target dependencies.",
    ]
    first_args = {
        "package_name": "org.example:library",
        "target_version": "2.4.0",
        "dependency_type": "dependencies",
        "manifest_path": "modules/service/pom.xml",
    }
    retry_args = {
        **first_args,
        "target_version": "2.4.1",
        "dependency_type": "dependencies",
    }
    bound_llm = MagicMock()
    bound_llm.invoke.side_effect = [
        AIMessage(
            content="",
            tool_calls=[
                {"name": maven_tool.name, "args": first_args, "id": "maven-first"},
                {"name": maven_tool.name, "args": retry_args, "id": "maven-deferred"},
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[{"name": maven_tool.name, "args": retry_args, "id": "maven-retry"}],
        ),
        AIMessage(content="Maven transaction completed."),
    ]
    llm = MagicMock()
    llm.bind_tools.return_value = bound_llm
    execution_state = {}

    result = run_bounded_subagent_loop(
        llm,
        [maven_tool],
        [HumanMessage(content="Update the committed Maven dependency.")],
        set(),
        execution_state=execution_state,
    )

    assert [event.name for event in result.tool_events] == [
        "modify_and_validate_maven_dependency",
        "modify_and_validate_maven_dependency",
        "modify_and_validate_maven_dependency",
    ]
    assert result.tool_events[1].content.startswith("DEFERRED: manifest tools")
    assert maven_tool.invoke.call_count == 2
    assert result.changed_files == ["modules/service/pom.xml"]
    assert execution_state["manifest_runtime_attempts_by_package"] == {"org.example:library": 2}
    second_turn = bound_llm.invoke.call_args_list[1].args[0]
    assert any(
        isinstance(message, HumanMessage)
        and "modify_and_validate_maven_dependency" in message.content
        for message in second_turn
    )
    assert any(
        isinstance(message, HumanMessage)
        and "Manifest transaction sequencing barrier" in message.content
        for message in second_turn
    )


def test_only_maven_no_fix_removal_is_serialized() -> None:
    removal_tool = MagicMock()
    removal_tool.name = "remove_no_fix_dependency"
    removal_tool.invoke.return_value = "SUCCESS: removed configured direct dependency"
    maven_args = {
        "requested_package": "org.example:library",
        "manifest_path": "pom.xml",
    }
    bound_llm = MagicMock()
    bound_llm.invoke.side_effect = [
        AIMessage(
            content="",
            tool_calls=[
                {"name": removal_tool.name, "args": maven_args, "id": "maven-remove-1"},
                {"name": removal_tool.name, "args": maven_args, "id": "maven-remove-2"},
            ],
        ),
        AIMessage(content="Maven removal attempted."),
    ]
    llm = MagicMock()
    llm.bind_tools.return_value = bound_llm

    maven_result = run_bounded_subagent_loop(
        llm,
        [removal_tool],
        [HumanMessage(content="Remove the committed direct Maven dependency.")],
        set(),
    )

    assert maven_result.tool_events[1].content.startswith("DEFERRED:")
    assert removal_tool.invoke.call_count == 1

    node_removal = MagicMock()
    node_removal.name = "remove_no_fix_dependency"
    node_removal.invoke.return_value = "SUCCESS: removed configured direct dependency"
    node_llm = MagicMock()
    node_bound = MagicMock()
    node_bound.invoke.side_effect = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": node_removal.name,
                    "args": {"requested_package": "lodash", "manifest_path": "package.json"},
                    "id": "node-remove-1",
                },
                {
                    "name": node_removal.name,
                    "args": {"requested_package": "axios", "manifest_path": "package.json"},
                    "id": "node-remove-2",
                },
            ],
        ),
        AIMessage(content="Node removals attempted."),
    ]
    node_llm.bind_tools.return_value = node_bound

    node_result = run_bounded_subagent_loop(
        node_llm,
        [node_removal],
        [HumanMessage(content="Remove the configured npm dependencies.")],
        set(),
    )

    assert [event.content for event in node_result.tool_events] == [
        "SUCCESS: removed configured direct dependency",
        "SUCCESS: removed configured direct dependency",
    ]
    assert node_removal.invoke.call_count == 2


def test_python_manifest_success_projects_manifest_and_regenerated_lockfile() -> None:
    event = ToolEvent(
        name="modify_and_validate_python_dependency",
        args={"package_name": "zope.interface", "manifest_path": "Pipfile"},
        content=(
            'SUCCESS: updated dependency JSON: {"changed_files": ["Pipfile", "Pipfile.lock"]}'
        ),
    )

    assert _infer_changed_files(event) == ["Pipfile", "Pipfile.lock"]
    recovery = _manifest_retry_recovery_instruction(
        ToolEvent(
            name=event.name,
            args={
                "package_name": "zope.interface",
                "target_version": "6.0",
                "dependency_type": "packages",
            },
            content="ERROR_CODE: SYNC_FAILED: restored both Pipfile files",
        ),
        attempts=1,
    )
    assert "modify_and_validate_python_dependency" in recovery
    assert "modify_and_validate_npm_dependency" not in recovery


def test_python_update_and_removal_transactions_share_turn_serialization() -> None:
    update_tool = MagicMock()
    update_tool.name = "modify_and_validate_python_dependency"
    update_tool.invoke.return_value = (
        'SUCCESS: updated package JSON: {"changed_files": ["Pipfile", "Pipfile.lock"]}'
    )
    removal_tool = MagicMock()
    removal_tool.name = "remove_no_fix_python_dependency"
    removal_tool.invoke.return_value = (
        'SUCCESS: removed package JSON: {"changed_files": ["Pipfile", "Pipfile.lock"]}'
    )

    bound_llm = MagicMock()
    bound_llm.invoke.side_effect = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": update_tool.name,
                    "args": {
                        "package_name": "zope.interface",
                        "target_version": "6.0",
                        "dependency_type": "packages",
                        "manifest_path": "Pipfile",
                    },
                    "id": "python-update",
                },
                {
                    "name": removal_tool.name,
                    "args": {
                        "package_name": "zope.interface",
                        "manifest_path": "Pipfile",
                    },
                    "id": "python-remove-deferred",
                },
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": removal_tool.name,
                    "args": {
                        "package_name": "zope.interface",
                        "manifest_path": "Pipfile",
                    },
                    "id": "python-remove",
                }
            ],
        ),
        AIMessage(content="complete"),
    ]
    llm = MagicMock()
    llm.bind_tools.return_value = bound_llm

    result = run_bounded_subagent_loop(
        llm,
        [update_tool, removal_tool],
        [HumanMessage(content="Update or remove the authorized dependency.")],
        set(),
    )

    assert update_tool.invoke.call_count == 1
    assert removal_tool.invoke.call_count == 1
    assert result.tool_events[1].content.startswith("DEFERRED:")
    assert set(result.changed_files) == {"Pipfile", "Pipfile.lock"}


def test_python_manifest_retry_limit_uses_normalized_package_identity() -> None:
    update_tool = MagicMock()
    update_tool.name = "modify_and_validate_python_dependency"
    update_tool.invoke.side_effect = [
        "ERROR_CODE: EDIT_FAILED: first attempt",
        "ERROR_CODE: EDIT_FAILED: second attempt",
        "ERROR_CODE: EDIT_FAILED: third attempt",
    ]

    bound_llm = MagicMock()
    bound_llm.invoke.side_effect = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": update_tool.name,
                    "args": {
                        "package_name": package,
                        "target_version": version,
                        "dependency_type": "dependencies",
                    },
                    "id": f"python-attempt-{index}",
                }
            ],
        )
        for index, (package, version) in enumerate(
            [
                ("Zope.Interface", "6.0.0"),
                ("zope_interface", "6.0.1"),
                ("zope-interface", "6.0.2"),
                ("ZOPE-INTERFACE", "6.0.3"),
            ],
            start=1,
        )
    ] + [AIMessage(content="complete")]
    llm = MagicMock()
    llm.bind_tools.return_value = bound_llm

    result = run_bounded_subagent_loop(
        llm,
        [update_tool],
        [HumanMessage(content="Update the authorized Python dependency.")],
        set(),
    )

    assert update_tool.invoke.call_count == 3
    assert result.tool_events[3].content.startswith("ERROR_CODE: RETRY_LIMIT_REACHED:")
    assert any(
        isinstance(message, HumanMessage)
        and "modify_and_validate_python_dependency" in message.content
        for message in bound_llm.invoke.call_args_list[4].args[0]
    )


class _StructuredResult(BaseModel):
    """Small response schema for testing LangChain's offline payload conversion."""

    answer: str


@pytest.mark.parametrize(
    ("model", "effort", "expected_effort", "expected_temperature"),
    [
        ("gpt-4o-mini", None, None, 0),
        ("gpt-5.4-mini", None, "none", 0),
        ("gpt-5.4-nano", None, "none", 0),
        ("gpt-6-luna", None, "medium", None),
        ("gpt-6-luna", "none", "none", None),
        ("gpt-6-luna", "low", "low", None),
        ("gpt-6-luna", "medium", "medium", None),
        ("gpt-6-luna", "high", "high", None),
        ("gpt-6-luna", "xhigh", "xhigh", None),
        ("gpt-6-luna", "max", "max", None),
    ],
)
def test_responses_payload_uses_model_compatible_parameters(
    model, effort, expected_effort, expected_temperature
):
    settings = AppSettings(openai_api_key="test-key")
    llm = create_chat_model(
        model,
        settings=settings,
        reasoning_effort=effort,
        chat_openai_class=ChatOpenAI,
    )

    payload = llm._get_request_payload("test prompt")

    assert payload["model"] == model
    assert payload["input"]
    assert llm.use_responses_api is True
    if expected_effort is None:
        assert "reasoning" not in payload
    else:
        assert payload["reasoning"] == {"effort": expected_effort}
    if expected_temperature is None:
        assert "temperature" not in payload
    else:
        assert payload["temperature"] == expected_temperature


def test_factory_rejects_effort_for_non_luna_model():
    with pytest.raises(ValueError, match="only applies when the model is gpt-6-luna"):
        create_chat_model("gpt-4o-mini", reasoning_effort="low", chat_openai_class=ChatOpenAI)


def test_factory_preserves_explicit_request_bounds_without_exposing_arbitrary_overrides():
    captured = {}

    def fake_chat_openai(**kwargs):
        captured.update(kwargs)
        return kwargs

    create_chat_model(
        "gpt-6-luna",
        settings=AppSettings(openai_api_key="test-key"),
        reasoning_effort="medium",
        request_timeout=60,
        max_retries=0,
        max_tokens=2048,
        chat_openai_class=fake_chat_openai,
    )

    assert captured["model"] == "gpt-6-luna"
    assert captured["reasoning_effort"] == "medium"
    assert captured["request_timeout"] == 60
    assert captured["max_retries"] == 0
    assert captured["max_tokens"] == 2048
    assert "api_key" in captured
    assert "arbitrary_kwargs" not in captured


def test_luna_output_budget_maps_to_responses_output_limit():
    llm = create_chat_model(
        "gpt-6-luna",
        settings=AppSettings(openai_api_key="test-key"),
        reasoning_effort="medium",
        max_tokens=2048,
        chat_openai_class=ChatOpenAI,
    )

    assert llm._get_request_payload("short summary")["max_output_tokens"] == 2048


@pytest.mark.parametrize(
    ("bound", "value", "message"),
    [
        ("request_timeout", 0, "request_timeout must be positive"),
        ("max_retries", -1, "max_retries cannot be negative"),
        ("max_tokens", 0, "max_tokens must be positive"),
    ],
)
def test_factory_rejects_invalid_request_bounds(bound, value, message):
    with pytest.raises(ValueError, match=message):
        create_chat_model("gpt-6-luna", **{bound: value}, chat_openai_class=ChatOpenAI)


@pytest.mark.parametrize("model", ["gpt-4o-mini", "gpt-5.4-mini", "gpt-5.4-nano", "gpt-6-luna"])
def test_responses_client_converts_function_tools(model):
    llm = create_chat_model(
        model,
        settings=AppSettings(openai_api_key="test-key"),
        chat_openai_class=ChatOpenAI,
    )
    bound_llm = llm.bind_tools(
        [
            {
                "name": "lookup_item",
                "description": "Look up an item.",
                "parameters": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                    "additionalProperties": False,
                },
            }
        ],
        parallel_tool_calls=False,
    )

    payload = llm._get_request_payload("test prompt", **bound_llm.kwargs)

    assert payload["tools"][0]["type"] == "function"
    assert payload["tools"][0]["name"] == "lookup_item"
    assert payload["parallel_tool_calls"] is False


@pytest.mark.parametrize("model", ["gpt-4o-mini", "gpt-5.4-mini", "gpt-5.4-nano", "gpt-6-luna"])
def test_responses_client_converts_structured_output(model):
    llm = create_chat_model(
        model,
        settings=AppSettings(openai_api_key="test-key"),
        chat_openai_class=ChatOpenAI,
    )
    structured_llm = llm.with_structured_output(_StructuredResult)

    payload = llm._get_request_payload("test prompt", **structured_llm.first.kwargs)

    assert payload["text_format"] is _StructuredResult
