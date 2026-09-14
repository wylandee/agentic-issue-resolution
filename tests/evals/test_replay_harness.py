"""Unit tests for the deterministic Phase 5 replay harness."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from remediation_engine.contracts.schemas import CommandResult
from remediation_engine.evals.models import EvalTestCaseRecord
from remediation_engine.orchestration.subagent_runtime import ToolEvent
from remediation_engine.runtime.path_policy import WorkspacePathError
from tests.evals.conftest import _deduplicate_token_records
from tests.evals.eval_case_helpers import case_metadata
from tests.evals.replay_harness import (
    ReplayCapture,
    ReplaySandbox,
    ScriptedReplayModel,
    cached_replay,
    serialize_result,
    serialize_tool_events,
)


def test_tool_event_serialization_uses_canonical_shape() -> None:
    """Runtime events serialize independently of the golden expected trace."""
    events = [ToolEvent(name="read_file_context", args={"file_path": "src/app.js"}, content="ok")]
    assert serialize_tool_events(events) == [
        {"name": "read_file_context", "args": {"file_path": "src/app.js"}, "output": "ok"}
    ]


def test_typed_result_serialization() -> None:
    """Pydantic command results become JSON-compatible mappings."""
    result = CommandResult(exit_code=0, stdout="ok", stderr="", duration_seconds=0.1)
    assert serialize_result(result)["exit_code"] == 0
    assert serialize_result(result)["stdout"] == "ok"


def test_path_traversal_is_rejected() -> None:
    """Sandbox reads and writes share production path validation."""
    with ReplaySandbox({"package.json": "{}"}) as sandbox:
        with pytest.raises(WorkspacePathError):
            sandbox.read_file("../outside.txt")
        with pytest.raises(WorkspacePathError):
            sandbox.write_file("/workspace/../outside.txt", "bad")


def test_command_routing_and_unknown_commands() -> None:
    """Registered commands return typed results and unknown commands fail."""
    with ReplaySandbox(
        {"package.json": '{"dependencies": {"demo": "1.0.0"}}'},
        command_responses={"npm test": {"exit_code": 0, "stdout": "passed"}},
    ) as sandbox:
        routed = sandbox.run("npm test")
        updated = sandbox.run("npm pkg set 'dependencies[demo]=2.0.0'")
        unknown = sandbox.run("curl https://example.invalid")
        assert isinstance(routed, CommandResult)
        assert routed.exit_code == 0
        assert updated.exit_code == 0
        assert '"demo": "2.0.0"' in (sandbox.read_file("package.json") or "")
        assert unknown.exit_code == 127
        assert sandbox.commands == [
            "npm test",
            "npm pkg set 'dependencies[demo]=2.0.0'",
            "curl https://example.invalid",
        ]


def test_rollback_restores_memory_only() -> None:
    """Rollback returns the replay workspace to its captured baseline."""
    with ReplaySandbox({"package.json": "{}", "src/app.js": "before"}) as sandbox:
        sandbox.write_file("src/app.js", "after")
        sandbox.write_file("generated.txt", "temporary")
        sandbox.restore_baseline()
        assert sandbox.read_file("src/app.js") == "before"
        assert sandbox.read_file("generated.txt") is None
        assert sandbox.writes == []


def test_no_host_repository_writes(tmp_path: Path) -> None:
    """Replay writes stay in memory and never alter a host baseline."""
    host_file = tmp_path / "package.json"
    host_file.write_text("{}\n", encoding="utf-8")
    with ReplaySandbox({"package.json": "{}\n"}) as sandbox:
        sandbox.write_file("package.json", '{"changed": true}\n')
    assert host_file.read_text(encoding="utf-8") == "{}\n"


def test_no_external_process_or_network_boundary_is_exposed() -> None:
    """The harness records commands without Docker, HTTP, or subprocess calls."""
    with ReplaySandbox(
        {"package.json": "{}"},
        command_responses={"node -c": {"exit_code": 0, "stdout": "syntax ok"}},
    ) as sandbox:
        result = sandbox.run("node -c src/app.js")
        assert result.exit_code == 0
        assert sandbox.external_calls == []


def test_cache_reuses_one_capture() -> None:
    """Two metric consumers share one production replay result."""
    cache: dict[tuple[str, str], ReplayCapture] = {}
    calls = 0

    def factory() -> ReplayCapture:
        nonlocal calls
        calls += 1
        return ReplayCapture(case_id="case", component="test")

    first = cached_replay(cache, "test", "case", factory)
    second = cached_replay(cache, "test", "case", factory)
    assert first is second
    assert calls == 1


def test_replay_capture_token_usage_is_added_to_case_metadata() -> None:
    """Persist provider token usage alongside a live replay observation."""
    capture = ReplayCapture(
        case_id="case",
        component="workaround_subagent",
        input_tokens=100,
        output_tokens=25,
        total_tokens=125,
        cached_input_tokens=80,
        token_cost=0.0003,
    )
    metadata = case_metadata(
        {"case_id": "case"},
        component="workaround_subagent",
        replay_source="production_live",
        actual_tools=[],
        capture=capture,
    )

    assert metadata["input_tokens"] == 100
    assert metadata["output_tokens"] == 25
    assert metadata["total_tokens"] == 125
    assert metadata["token_cost"] == 0.0003
    assert metadata["token_usage_available"] is True
    assert metadata["cached_input_tokens"] == 80
    assert metadata["cache_usage_available"] is True


def test_duplicate_metric_records_count_one_replay_once() -> None:
    """Deduplicate identical metric records without merging distinct replays."""
    metadata = {
        "case_id": "case-1",
        "component": "workaround_subagent",
        "replay_source": "production_live",
    }
    first = EvalTestCaseRecord(
        case_id="case-1",
        test_name="tool-correctness",
        input_tokens=100,
        output_tokens=25,
        total_tokens=125,
        token_cost=0.0002,
        token_usage_available=True,
        additional_metadata=metadata,
    )
    duplicate = first.model_copy(update={"test_name": "task-completion"})
    distinct = EvalTestCaseRecord(
        case_id="case-1",
        test_name="separate-replay",
        input_tokens=110,
        output_tokens=30,
        total_tokens=140,
        token_cost=0.0003,
        token_usage_available=True,
        additional_metadata=metadata,
    )

    deduplicated = _deduplicate_token_records([first, duplicate, distinct])

    assert [record.test_name for record in deduplicated] == [
        "tool-correctness",
        "separate-replay",
    ]
    assert sum(record.total_tokens or 0 for record in deduplicated) == 265
    assert sum(record.token_cost or 0.0 for record in deduplicated) == pytest.approx(0.0005)


def test_scripted_model_consumes_bound_tool_trace_and_final_response() -> None:
    """The scripted model validates tools and fails when its script is exhausted."""
    model = ScriptedReplayModel.from_tool_trace(
        [{"name": "read_file_context", "args": {"file_path": "src/app.js"}, "output": "ignored"}],
        final_text="done",
    )
    model.bind_tools([SimpleNamespace(name="read_file_context")])

    first = model.invoke([SystemMessage(content="static")])
    second = model.invoke([SystemMessage(content="static"), HumanMessage(content="dynamic")])

    assert first.tool_calls[0]["name"] == "read_file_context"
    assert second.tool_calls == []
    assert model.invocation_count == 2
    assert model.consumed_tool_calls == [
        {"name": "read_file_context", "args": {"file_path": "src/app.js"}, "id": "call-1"}
    ]
    assert model.invocation_messages[0][0]["type"] == "system"
    assert model.invocation_messages[0][0]["content"] == "static"
    assert model.invocation_messages[1][-1]["content"] == "dynamic"
    with pytest.raises(AssertionError, match="more scripted"):
        model.invoke([])


def test_scripted_model_rejects_unbound_tools() -> None:
    """A stale fixture tool fails instead of being silently synthesized."""
    model = ScriptedReplayModel.from_tool_trace([{"name": "unexpected_tool", "args": {}}])
    model.bind_tools([SimpleNamespace(name="allowed_tool")])

    with pytest.raises(AssertionError, match="not bound"):
        model.invoke([])


def test_scripted_model_can_opt_out_for_historical_unbound_tool_trace() -> None:
    """Historical workaround traces may preserve an unbound call as an event."""
    model = ScriptedReplayModel.from_tool_trace(
        [{"name": "historical_tool", "args": {}}],
        enforce_bound_tools=False,
    )
    model.bind_tools([SimpleNamespace(name="allowed_tool")])

    response = model.invoke([])

    assert response.tool_calls[0]["name"] == "historical_tool"
    assert model.consumed_tool_calls[0]["name"] == "historical_tool"


def test_scripted_structured_output_is_one_shot() -> None:
    """Structured triage responses are returned once and invocation is recorded."""
    result = CommandResult(exit_code=0, stdout="ok", stderr="", duration_seconds=0.0)
    model = ScriptedReplayModel.from_structured_result(result)
    structured = model.with_structured_output(CommandResult)

    assert structured.invoke("prompt") is result
    assert model.structured_invocation_count == 1
    with pytest.raises(AssertionError, match="more than once"):
        structured.invoke("prompt")
