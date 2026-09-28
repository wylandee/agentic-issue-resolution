"""
tests/test_execution_nodes.py - Unit tests for the remaining Phase 5 execution nodes.

All Docker SDK interactions are mocked. No real Docker daemon is required.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from remediation_engine.contracts.schemas import (
    CommandResult,
    RemediationTask,
    RoutingStrategy,
    SCARemediationStage,
    TaskAttemptSnapshot,
    TaskStatus,
)
from remediation_engine.language import ProjectLanguage
from remediation_engine.orchestration.graph_wrappers import (
    run_update_subagent_from_orchestrator,
    run_workaround_subagent_from_orchestrator,
)
from remediation_engine.orchestration.state import (
    ChangedFilesProjection,
    initial_orchestrator_state,
    merge_changed_files_reducer,
)
from remediation_engine.orchestration.supervisor_node import instruction_digest
from remediation_engine.orchestration.teardown_node import _build_diff, run_teardown_node
from remediation_engine.orchestration.workspace_builder import run_workspace_builder_node


def _sandbox_mock() -> MagicMock:
    mock = MagicMock()
    mock.__enter__ = MagicMock(return_value=mock)
    mock.__exit__ = MagicMock(return_value=None)
    mock.run.return_value = CommandResult(exit_code=0, duration_seconds=0.0)
    return mock


def _maven_dispatch_state():
    task = SimpleNamespace(
        task_id="task-1",
        parent_group_id="group-1",
        current_attempt_id=None,
    )
    group = SimpleNamespace(group_id="group-1")
    return {
        "task_queue": {"task-1": task},
        "active_target_task_ids": ["task-1"],
        "valid_groups": [group],
        "repo_root": "/tmp/repository",
        "workspace_volume": "agent_workspace_test",
        "maven_cache_volume": "agent_maven_repository_test",
        "project_language": ProjectLanguage.JAVA,
        "attempt_snapshots_by_id": {},
        "action_summaries": [],
        "constraints_ledger": [],
        "feedback_by_task": {},
        "retry_diagnostics_by_task": {},
        "workaround_replay_plans_by_task": {},
        "workspace_rollback_anchors_by_task": {},
    }


def _capture_maven_subagent_state(dispatcher, worker_node_name):
    state = _maven_dispatch_state()
    worker_node = MagicMock(return_value={})
    graph = SimpleNamespace(**{worker_node_name: worker_node})
    with (
        patch(
            "remediation_engine.orchestration.graph_wrappers._dispatch_boundary_rejection",
            return_value=None,
        ),
        patch(
            "remediation_engine.orchestration.graph_wrappers._prepare_workspace_for_dispatch",
            return_value=(state, []),
        ),
        patch(
            "remediation_engine.orchestration.graph_wrappers._create_workspace_attempt_snapshot",
            return_value=("snapshot", []),
        ),
        patch(
            "remediation_engine.orchestration.graph_wrappers._finalize_worker_workspace_snapshot",
            return_value=[],
        ),
        patch(
            "remediation_engine.orchestration.graph_wrappers._graph_module",
            return_value=graph,
        ),
    ):
        dispatcher(state)
    return worker_node.call_args.args[0]


class TestStateDefaults:
    def test_orchestrator_state_initializes_master_state_fields(self, tmp_path):
        state = initial_orchestrator_state(str(tmp_path), [])

        assert state["constraints_ledger"] == []
        assert state["retry_counts"] == {}
        assert state["group_strategies"] == {}
        assert state["qa_evaluations"] == {}
        assert state["action_summaries"] == []
        assert state["changed_files"] == []
        assert state["workspace_volume"] is None
        assert state["status"] == "pending"
        assert "messages" not in state
        assert state["maven_cache_volume"] is None


class TestMavenWorkerCacheRouting:
    def test_update_dispatch_passes_run_owned_maven_cache(self):
        state = _capture_maven_subagent_state(
            run_update_subagent_from_orchestrator,
            "run_update_subagent_node",
        )

        assert state["project_language"] == ProjectLanguage.JAVA
        assert state["maven_cache_volume"] == "agent_maven_repository_test"

    def test_workaround_dispatch_passes_run_owned_maven_cache(self):
        state = _capture_maven_subagent_state(
            run_workaround_subagent_from_orchestrator,
            "run_workaround_subagent_node",
        )

        assert state["project_language"] == ProjectLanguage.JAVA
        assert state["maven_cache_volume"] == "agent_maven_repository_test"


class TestWorkspaceBuilderNode:
    def test_invalid_repo_root_returns_workspace_build_failed(self):
        result = run_workspace_builder_node({"repo_root": "/nonexistent/xyz"})

        assert result["status"] == "workspace_build_failed"
        assert result["workspace_volume"] is None

    def test_success_creates_volume_and_copies_repo(self, tmp_path):
        state = initial_orchestrator_state(str(tmp_path), [])
        (tmp_path / "package.json").write_text("{}", encoding="utf-8")
        (tmp_path / "frontend").mkdir()
        (tmp_path / "frontend" / "package.json").write_text("{}", encoding="utf-8")
        client = MagicMock()
        sandbox = _sandbox_mock()

        with (
            patch(
                "remediation_engine.orchestration.workspace_builder.get_docker_client",
                return_value=client,
            ),
            patch(
                "remediation_engine.orchestration.workspace_builder.DockerSandbox",
                return_value=sandbox,
            ) as mock_sandbox,
        ):
            result = run_workspace_builder_node(state)

        client.volumes.create.assert_called_once()
        mock_sandbox.assert_called_once()
        assert [call.args[0] for call in sandbox.run.call_args_list] == [
            "npm install --package-lock=true",
            "cd frontend && npm install --package-lock=true",
        ]
        assert result["status"] == "workspace_ready"
        assert result["workspace_volume"].startswith("agent_workspace_")

    def test_no_package_repository_skips_install(self, tmp_path):
        state = initial_orchestrator_state(str(tmp_path), [])
        client = MagicMock()
        sandbox = _sandbox_mock()

        with (
            patch(
                "remediation_engine.orchestration.workspace_builder.get_docker_client",
                return_value=client,
            ),
            patch(
                "remediation_engine.orchestration.workspace_builder.DockerSandbox",
                return_value=sandbox,
            ),
        ):
            result = run_workspace_builder_node(state)

        sandbox.run.assert_not_called()
        assert result["status"] == "workspace_ready"

    def test_install_failure_preserves_volume_and_diagnostics(self, tmp_path):
        state = initial_orchestrator_state(str(tmp_path), [])
        (tmp_path / "package.json").write_text("{}", encoding="utf-8")
        client = MagicMock()
        sandbox = _sandbox_mock()
        sandbox.run.return_value = CommandResult(
            exit_code=1,
            duration_seconds=0.0,
            stdout="install output",
            stderr="npm ERR! dependency failure",
        )

        with (
            patch(
                "remediation_engine.orchestration.workspace_builder.get_docker_client",
                return_value=client,
            ),
            patch(
                "remediation_engine.orchestration.workspace_builder.DockerSandbox",
                return_value=sandbox,
            ),
        ):
            result = run_workspace_builder_node(state)

        assert result["status"] == "workspace_build_failed"
        assert result["workspace_volume"].startswith("agent_workspace_")
        assert "npm install failed in ." in result["errors"][0]
        assert "npm ERR! dependency failure" in result["errors"][0]

    def test_copy_failure_preserves_volume_for_teardown(self, tmp_path):
        state = initial_orchestrator_state(str(tmp_path), [])
        client = MagicMock()

        with (
            patch(
                "remediation_engine.orchestration.workspace_builder.get_docker_client",
                return_value=client,
            ),
            patch(
                "remediation_engine.orchestration.workspace_builder.DockerSandbox",
                side_effect=RuntimeError("copy failed"),
            ),
        ):
            result = run_workspace_builder_node(state)

        assert result["status"] == "workspace_build_failed"
        assert result["workspace_volume"].startswith("agent_workspace_")


class TestTeardownNode:
    def test_final_changed_file_projection_replaces_historical_candidates(self):
        assert merge_changed_files_reducer(["stale.ts"], ChangedFilesProjection(["actual.ts"])) == [
            "actual.ts"
        ]

    def test_diff_marks_missing_final_newline(self):
        diff = _build_diff("src/app.js", "const before = true;", "const after = true;")

        assert "\\ No newline at end of file" in diff

    def test_final_state_barrier_detaches_terminal_worker_attempt(self, tmp_path):
        instruction = "Apply the workaround."
        snapshot = TaskAttemptSnapshot(
            attempt_id="attempt-terminal",
            task_id="task-1",
            state_revision=1,
            task_revision=1,
            strategy_stage=SCARemediationStage.CODE_WORKAROUND,
            instruction=instruction,
            instruction_digest=instruction_digest(instruction),
            dispatch_node="workaround_subagent",
        )
        state = initial_orchestrator_state(str(tmp_path), [])
        state["task_queue"] = {
            "task-1": RemediationTask(
                task_id="task-1",
                parent_group_id="g1",
                strategy=RoutingStrategy.CODE_WORKAROUND,
                strategy_stage=SCARemediationStage.CODE_WORKAROUND,
                instruction=instruction,
                status=TaskStatus.UNFIXABLE,
                task_revision=1,
                current_attempt_id=snapshot.attempt_id,
            )
        }
        state["attempt_snapshots_by_id"] = {snapshot.attempt_id: snapshot}

        result = run_teardown_node(state)

        assert result["task_queue"]["task-1"].current_attempt_id is None
        assert result["active_target_task_ids"] == []
        assert any(
            event.error_code == "TERMINAL_TASK_FIELDS_NORMALIZED"
            for event in result["consistency_events"]
        )

    def test_final_state_barrier_repairs_passed_parent_with_failed_pivot_child(self, tmp_path):
        parent = RemediationTask(
            task_id="task-parent",
            parent_group_id="group-parent",
            strategy=RoutingStrategy.VERSION_BUMP,
            strategy_stage=SCARemediationStage.NPM_LATEST,
            instruction="Update the package.",
            status=TaskStatus.QA_PASSED,
        )
        child = RemediationTask(
            task_id="task-child",
            parent_group_id="group-child",
            parent_task_id="task-parent",
            strategy=RoutingStrategy.CODE_WORKAROUND,
            strategy_stage=SCARemediationStage.CODE_WORKAROUND,
            instruction="Apply a workaround.",
            status=TaskStatus.UNFIXABLE,
        )
        state = initial_orchestrator_state(str(tmp_path), [])
        state["task_queue"] = {parent.task_id: parent, child.task_id: child}

        result = run_teardown_node(state)

        assert result["task_queue"]["task-parent"].status == TaskStatus.PIVOTED
        assert result["status"] == "completed_with_errors"
        assert any(
            event.error_code == "PIVOT_PARENT_STATUS_REPAIRED"
            for event in result["consistency_events"]
        )

    def test_changed_files_are_deduplicated_diffed_and_volume_removed(self, tmp_path):
        route_dir = tmp_path / "routes"
        route_dir.mkdir()
        (route_dir / "login.ts").write_text("const x = 1;\n", encoding="utf-8")

        state = initial_orchestrator_state(str(tmp_path), [])
        state["status"] = "edits_completed"
        state["workspace_volume"] = "agent_workspace_deadbeef"
        state["changed_files"] = ["routes/login.ts", "routes/login.ts"]

        sandbox = _sandbox_mock()
        sandbox.read_file.return_value = "const x = 2;\n"
        client = MagicMock()

        with (
            patch(
                "remediation_engine.orchestration.teardown_node.DockerSandbox",
                return_value=sandbox,
            ),
            patch(
                "remediation_engine.orchestration.teardown_node.get_docker_client",
                return_value=client,
            ),
        ):
            result = run_teardown_node(state)

        sandbox.read_file.assert_called_once_with("routes/login.ts")
        client.volumes.get.assert_called_once_with("agent_workspace_deadbeef")
        client.volumes.get.return_value.remove.assert_called_once_with(force=True)
        assert result["status"] == "completed"
        assert result["workspace_volume"] is None
        assert result["changed_files"] == ["routes/login.ts"]
        assert "a/routes/login.ts" in result["diff"]
        assert "b/routes/login.ts" in result["diff"]

    def test_teardown_removes_workspace_and_maven_cache_volumes(self, tmp_path):
        state = initial_orchestrator_state(str(tmp_path), [])
        state["workspace_volume"] = "agent_workspace_deadbeef"
        state["maven_cache_volume"] = "agent_maven_repository_deadbeef"
        client = MagicMock()
        client.containers.list.return_value = []
        volumes = {
            state["workspace_volume"]: MagicMock(),
            state["maven_cache_volume"]: MagicMock(),
        }
        client.volumes.get.side_effect = volumes.__getitem__

        with patch(
            "remediation_engine.orchestration.teardown_node.get_docker_client",
            return_value=client,
        ):
            result = run_teardown_node(state)

        assert client.volumes.get.call_args_list[0].args == ("agent_workspace_deadbeef",)
        assert client.volumes.get.call_args_list[1].args == ("agent_maven_repository_deadbeef",)
        for volume in volumes.values():
            volume.remove.assert_called_once_with(force=True)
        assert result["workspace_volume"] is None
        assert result["maven_cache_volume"] is None

    def test_teardown_reports_cache_failure_without_claiming_workspace_live(self, tmp_path):
        state = initial_orchestrator_state(str(tmp_path), [])
        state["workspace_volume"] = "agent_workspace_deadbeef"
        state["maven_cache_volume"] = "agent_maven_repository_deadbeef"
        client = MagicMock()
        client.containers.list.return_value = []
        workspace = MagicMock()
        maven_cache = MagicMock()
        maven_cache.remove.side_effect = RuntimeError("permission denied")
        client.volumes.get.side_effect = {
            state["workspace_volume"]: workspace,
            state["maven_cache_volume"]: maven_cache,
        }.__getitem__

        with patch(
            "remediation_engine.orchestration.teardown_node.get_docker_client",
            return_value=client,
        ):
            result = run_teardown_node(state)

        assert [call.args[0] for call in client.volumes.get.call_args_list] == [
            "agent_workspace_deadbeef",
            "agent_maven_repository_deadbeef",
        ]
        workspace.remove.assert_called_once_with(force=True)
        maven_cache.remove.assert_called_once_with(force=True)
        assert result["workspace_volume"] is None
        assert result["maven_cache_volume"] == "agent_maven_repository_deadbeef"
        assert result["status"] == "completed_with_errors"
        assert any("failed to remove Maven cache volume" in error for error in result["errors"])

    def test_teardown_still_removes_cache_when_workspace_removal_fails(self, tmp_path):
        state = initial_orchestrator_state(str(tmp_path), [])
        state["workspace_volume"] = "agent_workspace_deadbeef"
        state["maven_cache_volume"] = "agent_maven_repository_deadbeef"
        client = MagicMock()
        client.containers.list.return_value = []
        workspace = MagicMock()
        workspace.remove.side_effect = RuntimeError("workspace volume locked")
        maven_cache = MagicMock()
        client.volumes.get.side_effect = {
            state["workspace_volume"]: workspace,
            state["maven_cache_volume"]: maven_cache,
        }.__getitem__

        with patch(
            "remediation_engine.orchestration.teardown_node.get_docker_client",
            return_value=client,
        ):
            result = run_teardown_node(state)

        assert [call.args[0] for call in client.volumes.get.call_args_list] == [
            "agent_workspace_deadbeef",
            "agent_maven_repository_deadbeef",
        ]
        workspace.remove.assert_called_once_with(force=True)
        maven_cache.remove.assert_called_once_with(force=True)
        assert result["workspace_volume"] == "agent_workspace_deadbeef"
        assert result["maven_cache_volume"] is None
        assert any("failed to remove workspace volume" in error for error in result["errors"])

    def test_unexpected_teardown_failure_still_cleans_both_volumes(self, tmp_path):
        state = initial_orchestrator_state(str(tmp_path), [])
        state["workspace_volume"] = "agent_workspace_deadbeef"
        state["maven_cache_volume"] = "agent_maven_repository_deadbeef"
        client = MagicMock()
        client.containers.list.return_value = []
        volumes = {
            state["workspace_volume"]: MagicMock(),
            state["maven_cache_volume"]: MagicMock(),
        }
        client.volumes.get.side_effect = volumes.__getitem__

        with (
            patch(
                "remediation_engine.orchestration.supervisor_node.reconcile_phase5_state_before_teardown",
                side_effect=RuntimeError("terminal graph failure"),
            ),
            patch(
                "remediation_engine.orchestration.teardown_node.get_docker_client",
                return_value=client,
            ),
        ):
            result = run_teardown_node(state)

        for volume in volumes.values():
            volume.remove.assert_called_once_with(force=True)
        assert result["workspace_volume"] is None
        assert result["maven_cache_volume"] is None
        assert any("terminal graph failure" in error for error in result["errors"])

    def test_changed_files_are_derived_from_actual_final_content(self, tmp_path):
        route_dir = tmp_path / "routes"
        route_dir.mkdir()
        (route_dir / "login.ts").write_text("const x = 1;\n", encoding="utf-8")
        (tmp_path / "lib").mkdir()
        (tmp_path / "lib" / "insecurity.ts").write_text("const safe = true;\n", encoding="utf-8")

        state = initial_orchestrator_state(str(tmp_path), [])
        state["workspace_volume"] = "agent_workspace_deadbeef"
        state["changed_files"] = ["routes/login.ts", "lib/insecurity.ts"]

        sandbox = _sandbox_mock()
        sandbox.read_file.side_effect = lambda path: {
            "routes/login.ts": "const x = 2;\n",
            "lib/insecurity.ts": "const safe = true;\n",
        }.get(path)
        client = MagicMock()

        with (
            patch(
                "remediation_engine.orchestration.teardown_node.DockerSandbox",
                return_value=sandbox,
            ),
            patch(
                "remediation_engine.orchestration.teardown_node.get_docker_client",
                return_value=client,
            ),
        ):
            result = run_teardown_node(state)

        assert result["changed_files"] == ["routes/login.ts"]
        assert "lib/insecurity.ts" not in result["diff"]

    def test_no_changed_files_still_removes_volume(self, tmp_path):
        state = initial_orchestrator_state(str(tmp_path), [])
        state["workspace_volume"] = "agent_workspace_deadbeef"

        client = MagicMock()

        with (
            patch(
                "remediation_engine.orchestration.teardown_node.get_docker_client",
                return_value=client,
            ),
            patch(
                "remediation_engine.orchestration.teardown_node.DockerSandbox",
            ) as mock_sandbox,
        ):
            result = run_teardown_node(state)

        mock_sandbox.assert_not_called()
        client.volumes.get.assert_called_once_with("agent_workspace_deadbeef")
        client.volumes.get.return_value.remove.assert_called_once_with(force=True)
        assert result["changed_files"] == []
        assert result["diff"] == ""

    def test_invalid_changed_file_type_is_reported(self, tmp_path):
        state = initial_orchestrator_state(str(tmp_path), [])
        state["changed_files"] = [None]

        result = run_teardown_node(state)

        assert result["changed_files"] == []
        assert any("path must be a string" in error for error in result["errors"])

    def test_attached_container_is_removed_before_volume(self, tmp_path):
        state = initial_orchestrator_state(str(tmp_path), [])
        state["workspace_volume"] = "agent_workspace_deadbeef"
        client = MagicMock()
        attached = MagicMock()
        attached.name = "/adoring_hertz"
        client.containers.list.return_value = [attached]

        with patch(
            "remediation_engine.orchestration.teardown_node.get_docker_client",
            return_value=client,
        ):
            result = run_teardown_node(state)

        client.containers.list.assert_called_once_with(
            all=True,
            filters={"volume": "agent_workspace_deadbeef"},
        )
        attached.remove.assert_called_once_with(force=True)
        client.volumes.get.return_value.remove.assert_called_once_with(force=True)
        assert result["workspace_volume"] is None

    def test_volume_conflict_is_retried_after_attached_container_cleanup(self, tmp_path):
        state = initial_orchestrator_state(str(tmp_path), [])
        state["workspace_volume"] = "agent_workspace_deadbeef"
        client = MagicMock()
        client.containers.list.return_value = []
        client.volumes.get.return_value.remove.side_effect = [
            RuntimeError("409 Client Error: Conflict (volume is in use)"),
            None,
        ]

        with (
            patch(
                "remediation_engine.orchestration.teardown_node.get_docker_client",
                return_value=client,
            ),
            patch("remediation_engine.orchestration.teardown_node.time.sleep") as sleep,
        ):
            result = run_teardown_node(state)

        assert client.volumes.get.return_value.remove.call_count == 2
        sleep.assert_called_once()
        assert result["status"] == "completed"
        assert result["workspace_volume"] is None

    def test_persistent_volume_conflict_is_reported_without_raising(self, tmp_path):
        state = initial_orchestrator_state(str(tmp_path), [])
        state["workspace_volume"] = "agent_workspace_deadbeef"
        client = MagicMock()
        client.containers.list.return_value = []
        client.volumes.get.return_value.remove.side_effect = RuntimeError(
            "409 Client Error: Conflict (volume is in use)"
        )

        with (
            patch(
                "remediation_engine.orchestration.teardown_node.get_docker_client",
                return_value=client,
            ),
            patch("remediation_engine.orchestration.teardown_node.time.sleep"),
        ):
            result = run_teardown_node(state)

        assert result["status"] == "completed_with_errors"
        assert result["workspace_volume"] == "agent_workspace_deadbeef"
        assert any("failed to remove workspace volume" in error for error in result["errors"])

    def test_existing_errors_produce_completed_with_errors_status(self, tmp_path):
        """Teardown preserves a failed worker outcome in its terminal status."""
        state = initial_orchestrator_state(str(tmp_path), [])
        state["errors"] = ["worker surrendered"]

        result = run_teardown_node(state)

        assert result["status"] == "completed_with_errors"
