"""Contract and runtime tests for workaround context management."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from remediation_engine.contracts.schemas import ScratchpadEntry, WorkaroundExecutionPhase
from remediation_engine.orchestration.context_manager import (
    DEFAULT_COMPACTION_INTERVAL,
    MAX_SCRATCHPAD_CHARS,
    PHASE_TOOL_REGISTRY,
    ContextManager,
    ScratchpadMemory,
    compact_conversation,
    get_phase_prompt,
    get_tools_for_phase,
)
from remediation_engine.orchestration.subagent_runtime import ToolEvent, run_bounded_subagent_loop


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
    ]
    tools = [SimpleNamespace(name=name) for name in names]

    assert set(PHASE_TOOL_REGISTRY) == set(WorkaroundExecutionPhase)
    for phase in WorkaroundExecutionPhase:
        expected = [name for name in names if name in PHASE_TOOL_REGISTRY[phase]]
        assert [tool.name for tool in get_tools_for_phase(phase, tools)] == expected

    assert "remove_no_fix_dependency" not in [
        tool.name for tool in get_tools_for_phase(WorkaroundExecutionPhase.EXECUTE, tools)
    ]
    assert [tool.name for tool in get_tools_for_phase(None, tools)] == [
        name for name in names if name in PHASE_TOOL_REGISTRY[WorkaroundExecutionPhase.INVESTIGATE]
    ]


def test_phase_prompts_are_action_oriented_and_preserve_optional_context() -> None:
    assert DEFAULT_COMPACTION_INTERVAL == 4
    assert "do not edit" in get_phase_prompt("INVESTIGATE").lower()
    assert "record_plan" in get_phase_prompt("PLAN")
    assert "atomic edit" in get_phase_prompt("EXECUTE")
    assert "read_repository_map" in get_phase_prompt("VALIDATE")
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
