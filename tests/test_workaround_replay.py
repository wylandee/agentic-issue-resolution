"""
Unit and integration tests for cumulative workaround retries and replay plans.
"""

from __future__ import annotations

from typing import Any
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
    WorkaroundReplayPlan,
    WorkerAttemptResult,
)
from remediation_engine.orchestration.subagent_runtime import (
    ToolEvent,
    has_all_modified_files_validated_after_last_edit,
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
        ToolEvent(name="deterministic_search_replace", args={"file_path": "a.js"}, content="SUCCESS: Modified"),
        ToolEvent(name="validate_code_syntax", args={"file_path": "a.js"}, content="SUCCESS: Validated"),
        ToolEvent(name="deterministic_search_replace", args={"file_path": "b.js"}, content="SUCCESS: Modified"),
        ToolEvent(name="validate_code_syntax", args={"file_path": "b.js"}, content="SUCCESS: Validated"),
    ]
    assert has_all_modified_files_validated_after_last_edit(events_valid) is True

    events_invalid = [
        ToolEvent(name="deterministic_search_replace", args={"file_path": "a.js"}, content="SUCCESS: Modified"),
        ToolEvent(name="validate_code_syntax", args={"file_path": "a.js"}, content="SUCCESS: Validated"),
        ToolEvent(name="deterministic_search_replace", args={"file_path": "b.js"}, content="SUCCESS: Modified"),
        # Missing validation for b.js
    ]
    assert has_all_modified_files_validated_after_last_edit(events_invalid) is False


def test_targeted_search_query_generation_in_prompt() -> None:
    """Verify that QA error context generates targeted search query guidance."""
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

    assert "SUGGESTED TARGETED SEARCH QUERY" in prompt
    assert "express-jwt" in prompt
    assert "CVE-2020-15084" in prompt


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
