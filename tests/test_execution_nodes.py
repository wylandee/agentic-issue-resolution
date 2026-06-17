"""
tests/test_execution_nodes.py - Unit tests for the remaining Phase 5 execution nodes.

All Docker SDK interactions are mocked. No real Docker daemon is required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.orchestrator.editor_node import run_workspace_builder_node
from src.orchestrator.state import initial_orchestrator_state
from src.orchestrator.teardown_node import run_teardown_node


def _sandbox_mock() -> MagicMock:
    mock = MagicMock()
    mock.__enter__ = MagicMock(return_value=mock)
    mock.__exit__ = MagicMock(return_value=None)
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
        client = MagicMock()
        sandbox = _sandbox_mock()

        with patch(
            "src.orchestrator.editor_node.get_docker_client",
            return_value=client,
        ), patch(
            "src.orchestrator.editor_node.DockerSandbox",
            return_value=sandbox,
        ) as mock_sandbox:
            result = run_workspace_builder_node(state)

        client.volumes.create.assert_called_once()
        mock_sandbox.assert_called_once()
        assert result["status"] == "workspace_ready"
        assert result["workspace_volume"].startswith("agent_workspace_")

    def test_copy_failure_preserves_volume_for_teardown(self, tmp_path):
        state = initial_orchestrator_state(str(tmp_path), [])
        client = MagicMock()

        with patch(
            "src.orchestrator.editor_node.get_docker_client",
            return_value=client,
        ), patch(
            "src.orchestrator.editor_node.DockerSandbox",
            side_effect=RuntimeError("copy failed"),
        ):
            result = run_workspace_builder_node(state)

        assert result["status"] == "workspace_build_failed"
        assert result["workspace_volume"].startswith("agent_workspace_")


class TestTeardownNode:
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

        with patch(
            "src.orchestrator.teardown_node.DockerSandbox",
            return_value=sandbox,
        ), patch(
            "src.orchestrator.teardown_node.get_docker_client",
            return_value=client,
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

        with patch(
            "src.orchestrator.teardown_node.get_docker_client",
            return_value=client,
        ), patch(
            "src.orchestrator.teardown_node.DockerSandbox",
        ) as mock_sandbox:
            result = run_teardown_node(state)

        mock_sandbox.assert_not_called()
        client.volumes.get.assert_called_once_with("agent_workspace_deadbeef")
        client.volumes.get.return_value.remove.assert_called_once_with(force=True)
        assert result["changed_files"] == []
        assert result["diff"] == ""
