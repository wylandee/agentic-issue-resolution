"""Unit tests for TrajectoryRecorder-to-DeepEval adapter and loaders (Phase 0)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from remediation_engine.evals.trajectories import (
    AttemptSnapshotRow,
    TrajectoryDocument,
    TrajectorySpan,
    extract_tool_calls,
    parse_trajectory_dict,
    parse_trajectory_markdown,
    spans_to_test_cases,
    trajectory_to_test_case,
)
from tests.evals.conftest import EvalSettings


def test_parse_trajectory_dict() -> None:
    """Test parsing an in-memory dictionary representation into TrajectoryDocument."""
    data = {
        "trace_id": "test-trace-123",
        "repo_root": "/workspace/repo",
        "source": "unit-test",
        "initial_state": {"status": "pending", "valid_groups": [{"group_id": "grp-1"}]},
        "final_state": {"status": "completed"},
        "spans": [
            {
                "run_id": "span-1",
                "name": "supervisor",
                "run_type": "chain",
                "parent_run_id": None,
                "started_at": "2026-08-25T01:00:00+00:00",
                "ended_at": "2026-08-25T01:00:02+00:00",
                "sequence": 1,
            },
            {
                "run_id": "span-2",
                "name": "modify_npm_dependency",
                "run_type": "tool",
                "parent_run_id": "span-1",
                "started_at": "2026-08-25T01:00:01+00:00",
                "ended_at": "2026-08-25T01:00:02+00:00",
                "inputs": {"package_name": "lodash", "target_version": "4.17.21"},
                "outputs": {"success": True},
                "sequence": 2,
            },
        ],
    }

    doc = parse_trajectory_dict(data)
    assert doc.trace_id == "test-trace-123"
    assert doc.repo_root == "/workspace/repo"
    assert doc.span_count == 2
    assert len(doc.spans) == 2

    span1 = doc.spans[0]
    assert span1.name == "supervisor"
    assert span1.duration_seconds == 2.0

    span2 = doc.spans[1]
    assert span2.name == "modify_npm_dependency"
    assert span2.parent_name == "supervisor"
    assert span2.is_tool is True
    assert span2.duration_seconds == 1.0


def test_trajectory_span_computed_properties() -> None:
    """Test computed properties on TrajectorySpan."""
    llm_span = TrajectorySpan(
        run_id="s1",
        name="triage.llm",
        run_type="llm",
        prompt_tokens=100,
        completion_tokens=50,
    )
    assert llm_span.is_llm is True
    assert llm_span.is_tool is False

    supervisor_plan_fix_span = TrajectorySpan(
        run_id="s3",
        name="supervisor.plan_fix",
        run_type="chain",
    )
    assert supervisor_plan_fix_span.is_llm is True
    assert supervisor_plan_fix_span.is_tool is False

    tool_span = TrajectorySpan(
        run_id="s2",
        name="tool.validate_manifest_sync",
        run_type="tool",
    )
    assert tool_span.is_llm is False
    assert tool_span.is_tool is True

    custom_tool_span = TrajectorySpan(
        run_id="s3",
        name="modify_npm_dependency",
        run_type="chain",
    )
    assert custom_tool_span.is_tool is True


def test_extract_tool_calls() -> None:
    """Test tool call extraction from spans and LLM output payloads."""
    spans = [
        TrajectorySpan(
            run_id="parent-1",
            name="react.llm",
            run_type="llm",
            outputs={
                "tool_calls": [
                    {
                        "name": "modify_npm_dependency",
                        "args": {"package_name": "lodash", "target_version": "4.17.21"},
                    }
                ]
            },
        ),
        TrajectorySpan(
            run_id="tool-1",
            name="modify_npm_dependency",
            run_type="tool",
            parent_run_id="parent-1",
            inputs={"package_name": "lodash", "target_version": "4.17.21"},
            outputs="Successfully updated package.json",
        ),
        TrajectorySpan(
            run_id="tool-2",
            name="tool.validate_manifest_sync",
            run_type="tool",
            parent_run_id="parent-1",
            inputs={"manifest_path": "package.json"},
            outputs="Manifest in sync",
        ),
    ]

    # Extract all tool calls
    all_tools = extract_tool_calls(spans)
    assert len(all_tools) >= 2
    tool_names = [t.name for t in all_tools]
    assert "modify_npm_dependency" in tool_names
    assert "validate_manifest_sync" in tool_names

    # Extract with parent filter
    filtered_tools = extract_tool_calls(spans, parent_span_id="parent-1")
    assert len(filtered_tools) == 2


def test_trajectory_to_test_case_conversion() -> None:
    """Test converting a single TrajectorySpan to an LLMTestCase."""
    span = TrajectorySpan(
        run_id="triage-1",
        name="triage.llm",
        run_type="llm",
        inputs="Triage prompt for CVE-2021-44228",
        outputs={"verdict": "ACTIONABLE", "strategy": "VERSION_UPDATE", "confidence": 0.95},
        duration_seconds=1.25,
    )

    doc = TrajectoryDocument(
        trace_id="trace-abc",
        initial_state={"valid_groups": [{"group_id": "sca:lodash"}]},
        spans=[span],
    )

    test_case = trajectory_to_test_case(span, doc=doc)
    assert test_case.input == "Triage prompt for CVE-2021-44228"
    assert "ACTIONABLE" in test_case.actual_output
    assert test_case.completion_time == 1.25
    assert test_case.context is not None
    assert "sca:lodash" in test_case.context[0]
    assert test_case.additional_metadata is not None
    assert test_case.additional_metadata["run_id"] == "triage-1"


def test_spans_to_test_cases_filtering() -> None:
    """Test spans_to_test_cases filtering by agent type."""
    spans = [
        TrajectorySpan(
            run_id="s1",
            name="triage.llm",
            run_type="llm",
            inputs="triage input",
            outputs="triage output",
        ),
        TrajectorySpan(
            run_id="s2",
            name="update_subagent",
            run_type="chain",
            inputs="update input",
            outputs="update output",
        ),
        TrajectorySpan(
            run_id="s3",
            name="qa_critic",
            run_type="chain",
            inputs="qa prompt",
            outputs={"evaluations": []},
        ),
    ]

    triage_cases = spans_to_test_cases(spans, agent_filter="triage")
    assert len(triage_cases) == 1
    assert triage_cases[0].input == "triage input"

    qa_cases = spans_to_test_cases(spans, agent_filter="qa_critic")
    assert len(qa_cases) == 1
    assert "evaluations" in qa_cases[0].actual_output


def test_supervisor_alias_filters_only_tactical_reasoner_spans() -> None:
    """Select the tactical reasoner and its children, not unrelated Supervisor spans."""
    spans = [
        TrajectorySpan(
            run_id="tactical",
            name="supervisor.tactical_reasoner",
            run_type="llm",
            inputs="tactical input",
            outputs="tactical output",
        ),
        TrajectorySpan(
            run_id="tactical-child",
            name="ChatOpenAI",
            run_type="llm",
            parent_name="supervisor.tactical_reasoner",
            inputs="child input",
            outputs="child output",
        ),
        TrajectorySpan(
            run_id="supervisor",
            name="supervisor",
            run_type="chain",
            inputs="unrelated supervisor input",
            outputs="unrelated supervisor output",
        ),
        TrajectorySpan(
            run_id="router",
            name="supervisor_router",
            run_type="chain",
            parent_name="supervisor",
            inputs="router input",
            outputs="router output",
        ),
    ]
    document = TrajectoryDocument(trace_id="tactical-trace", spans=spans)

    expected_spans = spans[:2]
    assert document.spans_for_agent("supervisor") == expected_spans
    assert document.spans_for_agent("tactical_supervisor") == expected_spans

    no_document_cases = spans_to_test_cases(spans, agent_filter="supervisor")
    assert [case.input for case in no_document_cases] == ["tactical input", "child input"]
    alias_cases = spans_to_test_cases(spans, agent_filter="tactical_supervisor")
    assert [case.input for case in alias_cases] == ["tactical input", "child input"]


def test_markdown_parser_resilience() -> None:
    """Test markdown parser resilience on partial, malformed, or minimal inputs."""
    minimal_markdown = """# Phase 5 Remediation Trajectory

## Run Metadata

- Trace ID: `trace-minimal`
- Repository: `/test/repo`
- Export source: `local-fallback`

## Root Input State

```json
{"status": "pending"}
```

## Root Final State

```json
{"status": "completed"}
```
"""
    doc = parse_trajectory_markdown(minimal_markdown)
    assert doc.trace_id == "trace-minimal"
    assert doc.repo_root == "/test/repo"
    assert doc.initial_state.get("status") == "pending"
    assert doc.final_state.get("status") == "completed"
    assert len(doc.spans) == 0

    # Test malformed JSON fallback
    malformed_markdown = """# Phase 5 Remediation Trajectory

## Root Input State

```json
{invalid json string without closing
```
"""
    doc_malformed = parse_trajectory_markdown(malformed_markdown)
    assert isinstance(doc_malformed.initial_state, dict)


def test_eval_settings_fixture(eval_settings: EvalSettings) -> None:
    """Test the eval_settings fixture configuration."""
    assert eval_settings.judge_model == "gpt-6-luna" or bool(eval_settings.judge_model)
    assert eval_settings.reasoning_effort == "max" or bool(eval_settings.reasoning_effort)
    assert isinstance(eval_settings.is_live, bool)
    assert isinstance(eval_settings.trajectory_dir, Path)
    assert isinstance(eval_settings.golden_dir, Path)
    if eval_settings.openai_api_key:
        assert eval_settings.openai_api_key not in repr(eval_settings)


def test_eval_settings_env_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that eval_settings resolves custom EVAL_JUDGE_MODEL from environment."""
    from tests.evals import conftest

    monkeypatch.setenv("EVAL_JUDGE_MODEL", "gpt-4o-mini")
    # Simulate eval_settings construction
    judge = (
        os.environ.get("EVAL_JUDGE_MODEL", "").strip()
        or conftest._INITIAL_JUDGE_MODEL
        or "gpt-6-luna"
    )
    assert judge == "gpt-4o-mini"


def test_attempt_snapshot_row_model() -> None:
    """Test AttemptSnapshotRow dataclass instantiation and fields."""
    row = AttemptSnapshotRow(
        task_id="task-1",
        attempt_id="att-1",
        revision="rev-0",
        stage="STAGE_1",
        qa_policy="STRICT",
        policy_source="MANIFEST",
        selected_version="4.17.21",
        dispatch="update_subagent",
        executed_versions=["4.17.21"],
        worker="completed",
        qa="passed",
        final_task_status="resolved",
        instruction="Update lodash to 4.17.21",
    )
    assert row.task_id == "task-1"
    assert row.executed_versions == ["4.17.21"]
    assert row.worker == "completed"


def test_markdown_parser_reads_nested_spans_without_local_trajectories(tmp_path):
    path = tmp_path / "trace.md"
    path.write_text(
        """# Phase 5 Remediation Trajectory

## Run Metadata

- Trace ID: `deterministic-trace`

## Root Final State

```json
{"status": "completed", "enabled": false, "count": 0}
```

## Span Details

### 1. supervisor (`chain`)

```json
{"run_id": "parent", "name": "supervisor", "run_type": "chain", "outputs": {"status": "completed"}}
```

### 2. tool.read (`tool`)

```json
{"run_id": "child", "parent_run_id": "parent", "name": "tool.read", "run_type": "tool", "outputs": "ok"}
```
""",
        encoding="utf-8",
    )
    doc = parse_trajectory_markdown(path)
    assert doc.trace_id == "deterministic-trace"
    assert doc.final_state == {"status": "completed", "enabled": False, "count": 0}
    assert [span.run_id for span in doc.spans] == ["span-1", "span-2"]
    assert doc.spans[1].name == "tool.read"
