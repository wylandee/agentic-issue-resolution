"""Characterization tests for the Docker-backed workspace builder node.

Docker client and sandbox boundaries are mocked so these tests exercise the node's
observable state projections without requiring a Docker daemon.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from remediation_engine.contracts.schemas import CommandResult
from remediation_engine.orchestration.workspace_builder import run_workspace_builder_node

_VOLUME_HEX = "deadbeefcafebabe"


def _sandbox_mock(*, result: CommandResult | None = None) -> MagicMock:
    sandbox = MagicMock()
    sandbox.__enter__ = MagicMock(return_value=sandbox)
    sandbox.__exit__ = MagicMock(return_value=None)
    sandbox.run.return_value = result or CommandResult(exit_code=0, duration_seconds=0.0)
    return sandbox


def _volume_name_patch():
    return patch(
        "remediation_engine.orchestration.workspace_builder.uuid.uuid4",
        return_value=SimpleNamespace(hex=_VOLUME_HEX),
    )


class TestWorkspaceBuilderNode:
    def test_success_creates_named_volume_copies_repo_and_returns_ready_projection(self, tmp_path):
        """The builder provisions one volume and hands the repository to the sandbox."""
        (tmp_path / "package.json").write_text("{}", encoding="utf-8")
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "index.js").write_text("console.log('ready');", encoding="utf-8")
        (tmp_path / "packages" / "web").mkdir(parents=True)
        (tmp_path / "packages" / "web" / "package.json").write_text("{}", encoding="utf-8")

        client = MagicMock()
        sandbox = _sandbox_mock()
        with (
            _volume_name_patch(),
            patch(
                "remediation_engine.orchestration.workspace_builder.get_docker_client",
                return_value=client,
            ),
            patch(
                "remediation_engine.orchestration.workspace_builder.DockerSandbox",
                return_value=sandbox,
            ) as sandbox_type,
        ):
            result = run_workspace_builder_node({"repo_root": str(tmp_path)})

        volume_name = f"agent_workspace_{_VOLUME_HEX[:8]}"
        client.volumes.create.assert_called_once_with(name=volume_name)
        client.close.assert_called_once_with()
        sandbox_type.assert_called_once_with(str(tmp_path), workspace_volume=volume_name)
        sandbox.__enter__.assert_called_once_with()
        sandbox.__exit__.assert_called_once()
        assert [call.args[0] for call in sandbox.run.call_args_list] == [
            "npm install --package-lock=true",
            "cd packages/web && npm install --package-lock=true",
        ]
        assert all(call.kwargs == {"timeout": 900} for call in sandbox.run.call_args_list)
        assert result == {"workspace_volume": volume_name, "status": "workspace_ready"}

    def test_volume_creation_failure_closes_client_and_does_not_start_sandbox(self, tmp_path):
        """A failed volume transaction reports failure and releases its client."""
        client = MagicMock()
        client.volumes.create.side_effect = RuntimeError("volume quota exceeded")
        sandbox_type = MagicMock()

        with (
            _volume_name_patch(),
            patch(
                "remediation_engine.orchestration.workspace_builder.get_docker_client",
                return_value=client,
            ),
            patch(
                "remediation_engine.orchestration.workspace_builder.DockerSandbox",
                sandbox_type,
            ),
        ):
            result = run_workspace_builder_node({"repo_root": str(tmp_path)})

        client.close.assert_called_once_with()
        sandbox_type.assert_not_called()
        assert result["status"] == "workspace_build_failed"
        assert result["workspace_volume"] is None
        assert "volume quota exceeded" in result["errors"][0]

    def test_sandbox_setup_failure_preserves_volume_for_teardown(self, tmp_path):
        """Once provisioned, a setup error retains the volume name for cleanup."""
        client = MagicMock()
        sandbox_type = MagicMock(side_effect=RuntimeError("repository copy failed"))

        with (
            _volume_name_patch(),
            patch(
                "remediation_engine.orchestration.workspace_builder.get_docker_client",
                return_value=client,
            ),
            patch(
                "remediation_engine.orchestration.workspace_builder.DockerSandbox",
                sandbox_type,
            ),
        ):
            result = run_workspace_builder_node({"repo_root": str(tmp_path)})

        volume_name = f"agent_workspace_{_VOLUME_HEX[:8]}"
        client.volumes.create.assert_called_once_with(name=volume_name)
        client.close.assert_called_once_with()
        assert result["status"] == "workspace_build_failed"
        assert result["workspace_volume"] == volume_name
        assert "repository copy failed" in result["errors"][0]

    def test_dependency_install_failure_returns_diagnostics_and_keeps_volume(self, tmp_path):
        """A failed npm command is surfaced while retaining the initialized volume."""
        (tmp_path / "package.json").write_text("{}", encoding="utf-8")
        client = MagicMock()
        sandbox = _sandbox_mock(
            result=CommandResult(
                exit_code=1,
                stdout="npm output",
                stderr="npm ERR! install failed",
                duration_seconds=0.0,
            )
        )

        with (
            _volume_name_patch(),
            patch(
                "remediation_engine.orchestration.workspace_builder.get_docker_client",
                return_value=client,
            ),
            patch(
                "remediation_engine.orchestration.workspace_builder.DockerSandbox",
                return_value=sandbox,
            ),
        ):
            result = run_workspace_builder_node({"repo_root": str(tmp_path)})

        volume_name = f"agent_workspace_{_VOLUME_HEX[:8]}"
        assert result["status"] == "workspace_build_failed"
        assert result["workspace_volume"] == volume_name
        assert "npm install failed in ." in result["errors"][0]
        assert "npm ERR! install failed" in result["errors"][0]
        sandbox.__exit__.assert_called_once()
