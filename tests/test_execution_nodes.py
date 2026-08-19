"""
tests/test_execution_nodes.py - Unit tests for the remaining Phase 5 execution nodes.

All Docker SDK interactions are mocked. No real Docker daemon is required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from remediation_engine.contracts.schemas import (
    CommandResult,
    RemediationTask,
    RoutingStrategy,
    SCARemediationStage,
    TaskAttemptSnapshot,
    TaskStatus,
)
from remediation_engine.orchestration.state import initial_orchestrator_state
from remediation_engine.orchestration.supervisor_node import _instruction_digest
from remediation_engine.orchestration.teardown_node import run_teardown_node
from remediation_engine.orchestration.workspace_builder import run_workspace_builder_node


def _sandbox_mock() -> MagicMock:
    mock = MagicMock()
    mock.__enter__ = MagicMock(return_value=mock)
    mock.__exit__ = MagicMock(return_value=None)
    mock.run.return_value = CommandResult(exit_code=0, duration_seconds=0.0)
    return mock


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
    def test_final_state_barrier_detaches_terminal_worker_attempt(self, tmp_path):
        instruction = "Apply the workaround."
        snapshot = TaskAttemptSnapshot(
            attempt_id="attempt-terminal",
            task_id="task-1",
            state_revision=1,
            task_revision=1,
            strategy_stage=SCARemediationStage.CODE_WORKAROUND,
            instruction=instruction,
            instruction_digest=_instruction_digest(instruction),
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
