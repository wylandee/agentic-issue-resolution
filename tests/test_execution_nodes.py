"""
tests/test_execution_nodes.py - Unit tests for the Phase 5 execution nodes.

All Docker SDK and subprocess interactions are mocked. No real Docker daemon or
Dependency-Check installation is required.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.contracts.schemas import (
    CommandResult,
    IssueSource,
    IssueType,
    Severity,
    VulnerabilityGroup,
    VulnerabilityIssue,
)
from src.orchestrator.editor_node import run_workspace_builder_node
from src.orchestrator.scanner_node import run_scanner_node
from src.orchestrator.state import initial_orchestrator_state
from src.orchestrator.teardown_node import run_teardown_node
from src.orchestrator.tester_node import run_tester_node
from src.orchestrator.workspace_sync_node import run_workspace_sync_node


def _sca_group(
    cve_id: str = "CVE-2021-44228",
    ghsa_id: str | None = None,
) -> VulnerabilityGroup:
    issue = VulnerabilityIssue(
        source=IssueSource.ODC,
        issue_type=IssueType.SCA,
        severity=Severity.HIGH,
        cve_id=cve_id,
        ghsa_id=ghsa_id,
        package_name="lodash",
        package_version="4.17.15",
    )
    return VulnerabilityGroup(
        group_id="sca:package.json:lodash",
        issue_type=IssueType.SCA,
        vulnerable_component="lodash",
        cve_ids=[cve_id] if cve_id else [],
        ghsa_ids=[ghsa_id] if ghsa_id else [],
        versions=["4.17.15"],
        sources=[IssueSource.ODC],
        representative_issue_id=issue.id,
        issues=[issue],
    )


def _sandbox_mock() -> MagicMock:
    mock = MagicMock()
    mock.__enter__ = MagicMock(return_value=mock)
    mock.__exit__ = MagicMock(return_value=None)
    return mock


class TestStateDefaults:
    def test_orchestrator_state_initializes_tool_loop_fields(self, tmp_path):
        state = initial_orchestrator_state(str(tmp_path), [])
        assert state["messages"] == []
        assert state["changed_files"] == []
        assert state["workspace_volume"] is None
        assert state["install_failures"] is None


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


class TestWorkspaceSyncNode:
    def test_missing_workspace_volume_returns_dependency_sync_failed(self, tmp_path):
        state = initial_orchestrator_state(str(tmp_path), [])
        result = run_workspace_sync_node(state)
        assert result["status"] == "dependency_sync_failed"

    def test_success_runs_npm_install_package_lock_true(self, tmp_path):
        state = initial_orchestrator_state(str(tmp_path), [])
        state["workspace_volume"] = "agent_workspace_deadbeef"

        sandbox = _sandbox_mock()
        sandbox.run.return_value = CommandResult(
            exit_code=0,
            stdout="ok",
            stderr="",
            duration_seconds=1.0,
        )

        with patch(
            "src.orchestrator.workspace_sync_node.DockerSandbox",
            return_value=sandbox,
        ) as mock_cls:
            result = run_workspace_sync_node(state)

        mock_cls.assert_called_once_with(
            repo_root=None,
            workspace_volume="agent_workspace_deadbeef",
        )
        sandbox.run.assert_called_once_with(
            "npm install --package-lock=true",
            timeout=600,
        )
        assert result["status"] == "dependencies_ready"
        assert result["install_failures"] is None

    def test_install_failure_returns_dependency_sync_failed(self, tmp_path):
        state = initial_orchestrator_state(str(tmp_path), [])
        state["workspace_volume"] = "agent_workspace_deadbeef"

        sandbox = _sandbox_mock()
        sandbox.run.return_value = CommandResult(
            exit_code=1,
            stdout="",
            stderr="dependency conflict",
            duration_seconds=1.0,
        )

        with patch(
            "src.orchestrator.workspace_sync_node.DockerSandbox",
            return_value=sandbox,
        ):
            result = run_workspace_sync_node(state)

        assert result["status"] == "dependency_sync_failed"
        assert "dependency conflict" in result["install_failures"]


class TestScannerNode:
    def test_missing_workspace_volume_returns_scan_failed(self, tmp_path):
        state = initial_orchestrator_state(str(tmp_path), [_sca_group()])
        result = run_scanner_node(state)
        assert result["status"] == "scan_failed"

    def test_docker_missing_returns_scanned(self, tmp_path):
        state = initial_orchestrator_state(str(tmp_path), [_sca_group()])
        state["workspace_volume"] = "agent_workspace_deadbeef"

        with patch("src.orchestrator.scanner_node.shutil.which", return_value=None):
            result = run_scanner_node(state)

        assert result["status"] == "scanned"
        assert result["scan_failures"] is None

    def test_odc_command_mounts_named_volume_and_cache(self, tmp_path):
        state = initial_orchestrator_state(str(tmp_path), [_sca_group()])
        state["workspace_volume"] = "agent_workspace_deadbeef"

        proc = MagicMock(spec=subprocess.CompletedProcess)
        proc.returncode = 0
        proc.stdout = ""
        proc.stderr = ""

        sandbox = _sandbox_mock()
        sandbox.read_file.return_value = json.dumps({"dependencies": []})

        with patch(
            "src.orchestrator.scanner_node.shutil.which",
            return_value="docker",
        ), patch(
            "src.orchestrator.scanner_node.subprocess.run",
            return_value=proc,
        ) as mock_run, patch(
            "src.orchestrator.scanner_node.DockerSandbox",
            return_value=sandbox,
        ):
            run_scanner_node(state)

        cmd = mock_run.call_args[0][0]
        assert "agent_workspace_deadbeef:/scan" in cmd
        expected_cache = str((Path(__file__).resolve().parents[1] / "data" / "cache").resolve())
        assert f"{expected_cache}:/usr/share/dependency-check/data" in cmd
        assert "--noupdate" in cmd

    def test_odc_command_falls_back_to_legacy_named_volume(self, tmp_path):
        state = initial_orchestrator_state(str(tmp_path), [_sca_group()])
        state["workspace_volume"] = "agent_workspace_deadbeef"

        proc = MagicMock(spec=subprocess.CompletedProcess)
        proc.returncode = 0
        proc.stdout = ""
        proc.stderr = ""

        sandbox = _sandbox_mock()
        sandbox.read_file.return_value = json.dumps({"dependencies": []})

        with patch(
            "src.orchestrator.scanner_node._resolve_odc_cache_source",
            return_value="odc-cache",
        ), patch(
            "src.orchestrator.scanner_node.shutil.which",
            return_value="docker",
        ), patch(
            "src.orchestrator.scanner_node.subprocess.run",
            return_value=proc,
        ) as mock_run, patch(
            "src.orchestrator.scanner_node.DockerSandbox",
            return_value=sandbox,
        ):
            run_scanner_node(state)

        cmd = mock_run.call_args[0][0]
        assert "odc-cache:/usr/share/dependency-check/data" in cmd

    def test_report_with_remaining_target_cve_returns_scan_failed(self, tmp_path):
        state = initial_orchestrator_state(str(tmp_path), [_sca_group("CVE-2021-44228")])
        state["workspace_volume"] = "agent_workspace_deadbeef"

        proc = MagicMock(spec=subprocess.CompletedProcess)
        proc.returncode = 0
        proc.stdout = ""
        proc.stderr = ""

        report = {
            "dependencies": [
                {
                    "fileName": "lodash.tgz",
                    "packages": [{"id": "pkg:npm/lodash@4.17.15"}],
                    "vulnerabilities": [{"name": "CVE-2021-44228"}],
                }
            ]
        }
        sandbox = _sandbox_mock()
        sandbox.read_file.return_value = json.dumps(report)

        with patch(
            "src.orchestrator.scanner_node.shutil.which",
            return_value="docker",
        ), patch(
            "src.orchestrator.scanner_node.subprocess.run",
            return_value=proc,
        ), patch(
            "src.orchestrator.scanner_node.DockerSandbox",
            return_value=sandbox,
        ):
            result = run_scanner_node(state)

        assert result["status"] == "scan_failed"
        assert "CVE-2021-44228" in result["scan_failures"]

    def test_report_with_remaining_target_ghsa_returns_scan_failed(self, tmp_path):
        state = initial_orchestrator_state(
            str(tmp_path),
            [_sca_group(cve_id=None, ghsa_id="GHSA-VPQ2-C234-7XJ6")],
        )
        state["workspace_volume"] = "agent_workspace_deadbeef"

        proc = MagicMock(spec=subprocess.CompletedProcess)
        proc.returncode = 0
        proc.stdout = ""
        proc.stderr = ""

        report = {
            "dependencies": [
                {
                    "fileName": "once.tgz",
                    "packages": [{"id": "pkg:npm/once@1.1.2"}],
                    "vulnerabilities": [{"name": "GHSA-vpq2-c234-7xj6"}],
                }
            ]
        }
        sandbox = _sandbox_mock()
        sandbox.read_file.return_value = json.dumps(report)

        with patch(
            "src.orchestrator.scanner_node.shutil.which",
            return_value="docker",
        ), patch(
            "src.orchestrator.scanner_node.subprocess.run",
            return_value=proc,
        ), patch(
            "src.orchestrator.scanner_node.DockerSandbox",
            return_value=sandbox,
        ):
            result = run_scanner_node(state)

        assert result["status"] == "scan_failed"
        assert "GHSA-VPQ2-C234-7XJ6" in result["scan_failures"]


class TestTesterNode:
    def test_missing_workspace_volume_returns_test_failed(self, tmp_path):
        state = initial_orchestrator_state(str(tmp_path), [])
        result = run_tester_node(state)
        assert result["status"] == "test_failed"

    def test_success_reuses_volume_and_runs_only_npm_test(self, tmp_path):
        state = initial_orchestrator_state(str(tmp_path), [])
        state["workspace_volume"] = "agent_workspace_deadbeef"

        sandbox = _sandbox_mock()
        sandbox.run.return_value = CommandResult(
            exit_code=0,
            stdout="passing",
            stderr="",
            duration_seconds=2.0,
        )

        with patch("src.orchestrator.tester_node.DockerSandbox", return_value=sandbox) as mock_cls:
            result = run_tester_node(state)

        mock_cls.assert_called_once_with(repo_root=None, workspace_volume=state["workspace_volume"])
        sandbox.run.assert_called_once_with("npm test", timeout=600)
        assert result["status"] == "tested"


class TestTeardownNode:
    def test_tested_status_returns_diff_for_changed_files_and_removes_volume(self, tmp_path):
        route_dir = tmp_path / "routes"
        route_dir.mkdir()
        (route_dir / "login.ts").write_text("const x = 1;\n", encoding="utf-8")

        state = initial_orchestrator_state(str(tmp_path), [])
        state["status"] = "tested"
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
