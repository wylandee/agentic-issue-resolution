"""
Unit and integration tests for cumulative workaround retries and replay plans.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4

from remediation_engine.contracts.schemas import (
    AgentActionStatus,
    IssueType,
    RemediationTask,
    RoutingStrategy,
    TaskAttemptSnapshot,
    TaskStatus,
    VulnerabilityGroup,
    WorkaroundEdit,
    WorkaroundExecutionPhase,
    WorkaroundReplayPlan,
    WorkerAttemptResult,
)
from remediation_engine.orchestration.remedy_tools import (
    _is_authoritative_evidence_source,
    _make_deterministic_search_replace_tool,
    _make_record_plan_tool,
    _make_validate_workaround_tool,
)
from remediation_engine.orchestration.subagent_runtime import (
    ToolEvent,
    has_all_modified_files_validated_after_last_edit,
    run_bounded_subagent_loop,
)
from remediation_engine.orchestration.supervisor_node import run_supervisor_node
from remediation_engine.orchestration.workaround_subagent import (
    _build_workaround_prompt,
)


def test_workaround_schemas_and_serialization() -> None:
    """Verify WorkaroundEdit and WorkaroundReplayPlan contracts."""
    edit1 = WorkaroundEdit(
        file_path="lib/insecurity.ts",
        old_text="expressJwt({",
        new_text="expressjwt({",
        edit_index=1,
    )
    assert edit1.file_path == "lib/insecurity.ts"
    assert edit1.edit_index == 1

    plan = WorkaroundReplayPlan(
        task_id="task-1",
        pre_attempt_snapshots={"lib/insecurity.ts": "const x = 1;"},
        successful_edits=[edit1],
        source_attempt_id="att-100",
    )
    assert plan.task_id == "task-1"
    assert len(plan.successful_edits) == 1
    assert plan.pre_attempt_snapshots["lib/insecurity.ts"] == "const x = 1;"

    worker_result = WorkerAttemptResult(
        attempt_id="att-100",
        task_id="task-1",
        task_revision=1,
        status=AgentActionStatus.SUCCESS,
        instruction_digest="digest-123",
        replay_plan=plan,
    )
    assert worker_result.replay_plan is not None
    assert worker_result.replay_plan.task_id == "task-1"


def test_has_all_modified_files_validated_after_last_edit() -> None:
    """Verify per-file syntax validation check."""
    events_valid = [
        ToolEvent(
            name="deterministic_search_replace",
            args={"file_path": "a.js"},
            content="SUCCESS: Modified",
        ),
        ToolEvent(
            name="validate_code_syntax", args={"file_path": "a.js"}, content="SUCCESS: Validated"
        ),
        ToolEvent(
            name="deterministic_search_replace",
            args={"file_path": "b.js"},
            content="SUCCESS: Modified",
        ),
        ToolEvent(
            name="validate_code_syntax", args={"file_path": "b.js"}, content="SUCCESS: Validated"
        ),
    ]
    assert has_all_modified_files_validated_after_last_edit(events_valid) is True

    events_invalid = [
        ToolEvent(
            name="deterministic_search_replace",
            args={"file_path": "a.js"},
            content="SUCCESS: Modified",
        ),
        ToolEvent(
            name="validate_code_syntax", args={"file_path": "a.js"}, content="SUCCESS: Validated"
        ),
        ToolEvent(
            name="deterministic_search_replace",
            args={"file_path": "b.js"},
            content="SUCCESS: Modified",
        ),
        # Missing validation for b.js
    ]
    assert has_all_modified_files_validated_after_last_edit(events_invalid) is False


def test_targeted_search_query_generation_in_prompt() -> None:
    """Verify that QA error context gives the worker evidence for query selection."""
    task = RemediationTask(
        task_id="task-1",
        parent_group_id="grp-1",
        strategy=RoutingStrategy.CODE_WORKAROUND,
        instruction="Fix express-jwt missing algorithms",
    )
    group = VulnerabilityGroup(
        group_id="grp-1",
        issue_type=IssueType.SCA,
        vulnerable_component="express-jwt",
        cve_ids=["CVE-2020-15084"],
        representative_issue_id=str(uuid4()),
    )

    prompt = _build_workaround_prompt(
        target_task=task,
        target_group=group,
        previous_feedback='TypeError: "algorithms should be set"',
    )

    assert "=== RECOMMENDED INITIAL SEARCH QUERY ===" in prompt
    assert "Scenario: update_mitigates_cve_but_breaks_tests" in prompt
    assert "express-jwt" in prompt
    assert "algorithms should be set" in prompt


def test_supervisor_commits_matching_replay_plan_and_ignores_stale() -> None:
    """Verify Supervisor only commits replay plans for matching active attempt snapshots."""
    task = RemediationTask(
        task_id="task-1",
        parent_group_id="grp-1",
        strategy=RoutingStrategy.CODE_WORKAROUND,
        instruction="Fix express-jwt missing algorithms",
        status=TaskStatus.PENDING,
        current_attempt_id="att-1",
        task_revision=1,
    )
    snapshot = TaskAttemptSnapshot(
        attempt_id="att-1",
        task_id="task-1",
        task_revision=1,
        instruction="Fix express-jwt missing algorithms",
        instruction_digest="digest-1",
        dispatch_node="workaround_subagent",
    )
    replay_plan = WorkaroundReplayPlan(
        task_id="task-1",
        pre_attempt_snapshots={"file.js": "const orig = 1;"},
        successful_edits=[
            WorkaroundEdit(
                file_path="file.js",
                old_text="const orig = 1;",
                new_text="const orig = 2;",
                edit_index=1,
            )
        ],
        source_attempt_id="att-1",
    )
    matching_result = WorkerAttemptResult(
        attempt_id="att-1",
        task_id="task-1",
        task_revision=1,
        status=AgentActionStatus.SUCCESS,
        instruction_digest="digest-1",
        replay_plan=replay_plan,
        changed_files=["file.js"],
    )

    group = VulnerabilityGroup(
        group_id="grp-1",
        issue_type=IssueType.SCA,
        vulnerable_component="express-jwt",
        representative_issue_id=str(uuid4()),
    )

    state: dict[str, Any] = {
        "repo_root": "/fake/repo",
        "valid_groups": [group],
        "task_queue": {"task-1": task},
        "active_target_task_ids": ["task-1"],
        "attempt_snapshots_by_id": {"att-1": snapshot},
        "worker_results_by_attempt": {"att-1": matching_result},
    }

    out = run_supervisor_node(state)
    assert "workaround_replay_plans_by_task" in out
    assert "task-1" in out["workaround_replay_plans_by_task"]
    committed_plan = out["workaround_replay_plans_by_task"]["task-1"]
    assert len(committed_plan.successful_edits) == 1
    assert committed_plan.successful_edits[0].new_text == "const orig = 2;"

    # Stale result test: mismatch in instruction_digest
    stale_result = WorkerAttemptResult(
        attempt_id="att-2",
        task_id="task-1",
        task_revision=1,
        status=AgentActionStatus.SUCCESS,
        instruction_digest="wrong-digest",
        replay_plan=WorkaroundReplayPlan(task_id="task-1", source_attempt_id="att-2"),
    )
    stale_state: dict[str, Any] = {
        "repo_root": "/fake/repo",
        "valid_groups": [group],
        "task_queue": {"task-1": task},
        "active_target_task_ids": ["task-1"],
        "attempt_snapshots_by_id": {"att-1": snapshot},
        "worker_results_by_attempt": {"att-2": stale_result},
    }
    stale_out = run_supervisor_node(stale_state)
    assert "task-1" not in stale_out.get("workaround_replay_plans_by_task", {})


def test_is_authoritative_evidence_source() -> None:
    """Verify classification of authoritative vs non-authoritative evidence sources."""
    assert (
        _is_authoritative_evidence_source("https://github.com/advisories/GHSA-6g6m-m6h5-w9gf")
        is True
    )
    assert _is_authoritative_evidence_source("https://registry.npmjs.org/express-jwt") is True
    assert _is_authoritative_evidence_source("node_modules/express-jwt/README.md") is True
    assert _is_authoritative_evidence_source("https://stackoverflow.com/questions/12345") is False
    assert _is_authoritative_evidence_source("search snippet only") is False
    assert _is_authoritative_evidence_source("") is False


def test_blocked_vs_failure_validation_classification() -> None:
    """Verify BLOCKED vs FAILURE returns in workaround validation tool."""
    mock_sandbox = MagicMock()

    # Case 1: missing node_modules / module -> BLOCKED
    def mock_run_no_nm(cmd, **kw):
        if "tsc" in cmd or "node" in cmd:
            return MagicMock(exit_code=1, stdout="", stderr="Cannot find module 'express-jwt'")
        return MagicMock(exit_code=0, stdout="", stderr="")

    mock_sandbox.run.side_effect = mock_run_no_nm
    mock_sandbox.read_file.return_value = "const x = 1;"
    touched: set[str] = set()
    tool = _make_validate_workaround_tool(mock_sandbox, touched)
    res = tool.invoke({"modified_files": ["lib/insecurity.ts"]})
    assert res.startswith("BLOCKED:")
    assert "Cannot find module" in res

    # Case 2: node_modules exists, typecheck fails with missing command -> BLOCKED
    def mock_run(cmd, **kw):
        if "test -d node_modules" in cmd:
            return MagicMock(exit_code=0, stdout="", stderr="")
        if "tsc" in cmd:
            return MagicMock(exit_code=127, stdout="", stderr="npx: command not found")
        return MagicMock(exit_code=0, stdout="", stderr="")

    mock_sandbox.run.side_effect = mock_run
    mock_sandbox.read_file.return_value = "const x = 1;"
    res2 = tool.invoke({"modified_files": ["lib/insecurity.ts"]})
    assert res2.startswith("BLOCKED:")
    assert "npx: command not found" in res2

    # Case 3: typecheck fails with code type error -> FAILURE
    def mock_run_type_err(cmd, **kw):
        if "test -d node_modules" in cmd:
            return MagicMock(exit_code=0, stdout="", stderr="")
        if "tsc" in cmd:
            return MagicMock(
                exit_code=2,
                stdout="lib/insecurity.ts(5,1): error TS2304: Cannot find name 'foo'.",
                stderr="",
            )
        return MagicMock(exit_code=0, stdout="", stderr="")

    mock_sandbox.run.side_effect = mock_run_type_err
    res3 = tool.invoke({"modified_files": ["lib/insecurity.ts"]})
    assert res3.startswith("FAILURE:")
    assert "TS2304" in res3


def test_subagent_runtime_drains_turn_and_stops_on_infrastructure_blocker() -> None:
    """Verify subagent loop drains all tool calls in turn before exiting on BLOCKED."""
    mock_llm = MagicMock()
    mock_llm.bind_tools.return_value = mock_llm

    tool1 = MagicMock()
    tool1.name = "validate_workaround"
    tool1.invoke.return_value = "BLOCKED: Workaround validation gate 'environment' blocked.\nDiagnostic: node_modules missing"

    tool2 = MagicMock()
    tool2.name = "read_repository_map"
    tool2.invoke.return_value = "tree output"

    ai_msg = MagicMock(
        content="",
        tool_calls=[
            {
                "name": "validate_workaround",
                "args": {"modified_files": ["lib/insecurity.ts"]},
                "id": "call_1",
            },
            {"name": "read_repository_map", "args": {}, "id": "call_2"},
        ],
    )
    mock_llm.invoke.return_value = ai_msg

    touched: set[str] = set()
    result = run_bounded_subagent_loop(
        llm=mock_llm,
        tools=[tool1, tool2],
        initial_messages=[],
        touched_files=touched,
    )

    # LLM should only be invoked once
    assert mock_llm.invoke.call_count == 1
    # Both tool calls should be recorded
    assert len(result.tool_events) == 2
    assert result.tool_events[0].name == "validate_workaround"
    assert result.tool_events[1].name == "read_repository_map"
    # Errors should contain INFRASTRUCTURE_BLOCKER
    assert any("INFRASTRUCTURE_BLOCKER:" in err for err in result.errors)


def test_authoritative_evidence_enforcement_in_edit_tools() -> None:
    """Verify edit tools require authoritative evidence when require_authoritative_evidence is set."""
    mock_sandbox = MagicMock()
    mock_sandbox.read_file.return_value = "const x = 1;"
    mock_sandbox.run.return_value = MagicMock(exit_code=0, stdout="", stderr="")
    touched: set[str] = set()
    plan_state: dict[str, Any] = {
        "recorded": True,
        "phase": WorkaroundExecutionPhase.EXECUTE.value,
        "planned_files": ["lib/insecurity.ts"],
        "inspected_files": {"lib/insecurity.ts"},
        "require_authoritative_evidence": True,
        "has_authoritative_evidence": False,
    }

    edit_tool = _make_deterministic_search_replace_tool(mock_sandbox, touched, plan_state)
    res = edit_tool.invoke(
        {
            "file_path": "lib/insecurity.ts",
            "old_text": "const x = 1;",
            "new_text": "const x = 2;",
        }
    )

    assert res.startswith("ERROR:")
    assert "Authoritative evidence required" in res

    # Record plan with authoritative evidence source
    rec_tool = _make_record_plan_tool(plan_state)
    rec_tool.invoke(
        {
            "affected_files": ["lib/insecurity.ts"],
            "affected_symbols": ["x"],
            "security_invariant": "test invariant",
            "causal_hypothesis": "test hypothesis",
            "exact_intended_edits": "change 1 to 2",
            "evidence_source": "https://github.com/expressjs/express-jwt",
        }
    )

    assert plan_state.get("has_authoritative_evidence") is True

    # Now edit should succeed
    res2 = edit_tool.invoke(
        {
            "file_path": "lib/insecurity.ts",
            "old_text": "const x = 1;",
            "new_text": "const x = 2;",
        }
    )
    assert res2.startswith("SUCCESS:")
