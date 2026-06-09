"""
tests/test_execution_nodes.py — Unit tests for the Phase 5 execution nodes.

All Docker SDK calls and subprocess calls are mocked; no real Docker daemon or
Dependency-Check installation is required.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.contracts.schemas import (
    CommandResult,
    EditRequest,
    IssueSource,
    IssueType,
    Severity,
    VulnerabilityGroup,
    VulnerabilityIssue,
)
from src.orchestrator.editor_node import run_editor_node
from src.orchestrator.scanner_node import run_scanner_node
from src.orchestrator.state import initial_orchestrator_state
from src.orchestrator.teardown_node import run_teardown_node
from src.orchestrator.tester_node import run_tester_node


def _edit(tmp_path: Path, file_path: str, old_text: str, new_text: str) -> EditRequest:
    return EditRequest(
        repo_root=str(tmp_path),
        file_path=file_path,
        old_text=old_text,
        new_text=new_text,
        dry_run=False,
    )


def _sca_group(cve_id: str = "CVE-2021-44228") -> VulnerabilityGroup:
    issue = VulnerabilityIssue(
        source=IssueSource.ODC,
        issue_type=IssueType.SCA,
        severity=Severity.HIGH,
        cve_id=cve_id,
        package_name="lodash",
        package_version="4.17.15",
    )
    return VulnerabilityGroup(
        group_id="sca:package.json:lodash",
        issue_type=IssueType.SCA,
        vulnerable_component="lodash",
        cve_ids=[cve_id],
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
    def test_workspace_volume_initialized_to_none(self, tmp_path):
        state = initial_orchestrator_state(str(tmp_path), [])
        assert "workspace_volume" in state
        assert state["workspace_volume"] is None


class TestEditorNode:
    def test_no_edits_returns_no_edits(self, tmp_path):
        state = initial_orchestrator_state(str(tmp_path), [])
        result = run_editor_node(state)
        assert result["status"] == "no_edits"

    def test_invalid_repo_root_returns_edit_failed(self):
        state = initial_orchestrator_state("/nonexistent/xyz", [])
        state["edit_requests"] = [_edit(Path("."), "package.json", "a", "b")]
        result = run_editor_node(state)
        assert result["status"] == "edit_failed"
        assert result["workspace_volume"] is None

    def test_success_creates_volume_applies_edit_and_installs(self, tmp_path):
        state = initial_orchestrator_state(str(tmp_path), [])
        state["edit_requests"] = [
            _edit(tmp_path, "package.json", '"lodash": "^4.17.15"', '"lodash": "^4.17.21"')
        ]

        client = MagicMock()
        sandbox = _sandbox_mock()
        sandbox.read_file.return_value = '{"dependencies": {"lodash": "^4.17.15"}}'
        sandbox.run.return_value = CommandResult(
            exit_code=0,
            stdout="ok",
            stderr="",
            duration_seconds=1.0,
        )

        with patch("src.orchestrator.editor_node.get_docker_client", return_value=client), patch(
            "src.orchestrator.editor_node.DockerSandbox",
            return_value=sandbox,
        ):
            result = run_editor_node(state)

        client.volumes.create.assert_called_once()
        sandbox.write_file.assert_called_once()
        sandbox.run.assert_called_once_with("npm install --no-audit --no-fund", timeout=600)
        assert result["status"] == "edited"
        assert result["workspace_volume"].startswith("agent_workspace_")

    def test_old_text_not_found_returns_edit_failed_and_preserves_volume(self, tmp_path):
        state = initial_orchestrator_state(str(tmp_path), [])
        state["edit_requests"] = [_edit(tmp_path, "package.json", "missing", "new")]

        client = MagicMock()
        sandbox = _sandbox_mock()
        sandbox.read_file.return_value = '{"dependencies":{"lodash":"^4.17.15"}}'

        with patch("src.orchestrator.editor_node.get_docker_client", return_value=client), patch(
            "src.orchestrator.editor_node.DockerSandbox",
            return_value=sandbox,
        ):
            result = run_editor_node(state)

        assert result["status"] == "edit_failed"
        assert result["workspace_volume"].startswith("agent_workspace_")

    def test_npm_install_failure_returns_edit_failed(self, tmp_path):
        state = initial_orchestrator_state(str(tmp_path), [])
        state["edit_requests"] = [_edit(tmp_path, "package.json", "old", "new")]

        client = MagicMock()
        sandbox = _sandbox_mock()
        sandbox.read_file.return_value = "old"
        sandbox.run.return_value = CommandResult(
            exit_code=1,
            stdout="",
            stderr="dependency conflict",
            duration_seconds=1.0,
        )

        with patch("src.orchestrator.editor_node.get_docker_client", return_value=client), patch(
            "src.orchestrator.editor_node.DockerSandbox",
            return_value=sandbox,
        ):
            result = run_editor_node(state)

        assert result["status"] == "edit_failed"
        assert "dependency conflict" in result["test_failures"]
        assert result["workspace_volume"].startswith("agent_workspace_")


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

        with patch("src.orchestrator.scanner_node.shutil.which", return_value="docker"), patch(
            "src.orchestrator.scanner_node.subprocess.run",
            return_value=proc,
        ) as mock_run, patch(
            "src.orchestrator.scanner_node.DockerSandbox",
            return_value=sandbox,
        ):
            run_scanner_node(state)

        cmd = mock_run.call_args[0][0]
        assert f"{state['workspace_volume']}:/scan" in cmd
        assert "odc-cache:/usr/share/dependency-check/data" in cmd
        assert "--noupdate" in cmd
        assert sandbox.read_file.called

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

        with patch("src.orchestrator.scanner_node.shutil.which", return_value="docker"), patch(
            "src.orchestrator.scanner_node.subprocess.run",
            return_value=proc,
        ), patch(
            "src.orchestrator.scanner_node.DockerSandbox",
            return_value=sandbox,
        ):
            result = run_scanner_node(state)

        assert result["status"] == "scan_failed"
        assert "CVE-2021-44228" in result["scan_failures"]

    def test_nonzero_exit_without_report_returns_scan_failed(self, tmp_path):
        state = initial_orchestrator_state(str(tmp_path), [_sca_group()])
        state["workspace_volume"] = "agent_workspace_deadbeef"

        proc = MagicMock(spec=subprocess.CompletedProcess)
        proc.returncode = 1
        proc.stdout = "warn"
        proc.stderr = "crashed"

        sandbox = _sandbox_mock()
        sandbox.read_file.return_value = None

        with patch("src.orchestrator.scanner_node.shutil.which", return_value="docker"), patch(
            "src.orchestrator.scanner_node.subprocess.run",
            return_value=proc,
        ), patch(
            "src.orchestrator.scanner_node.DockerSandbox",
            return_value=sandbox,
        ):
            result = run_scanner_node(state)

        assert result["status"] == "scan_failed"
        assert "produced no parseable report" in result["scan_failures"]


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

    def test_test_failure_returns_test_failed(self, tmp_path):
        state = initial_orchestrator_state(str(tmp_path), [])
        state["workspace_volume"] = "agent_workspace_deadbeef"

        sandbox = _sandbox_mock()
        sandbox.run.return_value = CommandResult(
            exit_code=1,
            stdout="1 failing",
            stderr="AssertionError",
            duration_seconds=2.0,
        )

        with patch("src.orchestrator.tester_node.DockerSandbox", return_value=sandbox):
            result = run_tester_node(state)

        assert result["status"] == "test_failed"
        assert "AssertionError" in result["test_failures"]


class TestTeardownNode:
    def test_tested_status_returns_diff_and_removes_volume(self, tmp_path):
        (tmp_path / "package.json").write_text('{"name":"app","version":"1.0.0"}\n', encoding="utf-8")
        state = initial_orchestrator_state(str(tmp_path), [])
        state["status"] = "tested"
        state["workspace_volume"] = "agent_workspace_deadbeef"

        sandbox = _sandbox_mock()
        sandbox.read_file.side_effect = [
            '{"name":"app","version":"1.1.0"}\n',
            None,
        ]
        client = MagicMock()

        with patch("src.orchestrator.teardown_node.DockerSandbox", return_value=sandbox), patch(
            "src.orchestrator.teardown_node.get_docker_client",
            return_value=client,
        ):
            result = run_teardown_node(state)

        sandbox.read_file.assert_any_call("package.json")
        client.volumes.get.assert_called_once_with("agent_workspace_deadbeef")
        client.volumes.get.return_value.remove.assert_called_once_with(force=True)
        assert result["status"] == "completed"
        assert result["workspace_volume"] is None
        assert result["changed_files"] == ["package.json"]
        assert "a/package.json" in result["diff"]
        assert "b/package.json" in result["diff"]

    def test_cleanup_failure_is_returned_in_errors(self, tmp_path):
        state = initial_orchestrator_state(str(tmp_path), [])
        state["workspace_volume"] = "agent_workspace_deadbeef"

        client = MagicMock()
        client.volumes.get.side_effect = RuntimeError("remove failed")

        with patch("src.orchestrator.teardown_node.get_docker_client", return_value=client):
            result = run_teardown_node(state)

        assert result["status"] == "completed"
        assert result["workspace_volume"] is None
        assert any("remove failed" in err for err in result["errors"])
