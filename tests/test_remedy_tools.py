"""
tests/test_remedy_tools.py - Direct unit tests for Phase 5 Remedy Agent tools.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.contracts.schemas import CommandResult
from src.orchestrator.remedy_tools import build_agent_tools


def _tool_map(sandbox, touched_files=None, target_identifiers=None):
    if touched_files is None:
        touched_files = set()
    if target_identifiers is None:
        target_identifiers = set()
    tools = build_agent_tools(sandbox, touched_files, target_identifiers)
    return {tool.name: tool for tool in tools}


class TestFactory:
    def test_factory_returns_all_expected_tools(self):
        sandbox = MagicMock()
        sandbox._workspace_volume = "agent_workspace_deadbeef"

        tools = _tool_map(sandbox)

        assert set(tools) == {
            "read_workspace_file",
            "deterministic_search_replace",
            "run_dependency_install",
            "run_security_scan",
            "run_unit_tests",
        }


class TestReadWorkspaceFile:
    def test_absolute_path_rejected(self):
        sandbox = MagicMock()
        tools = _tool_map(sandbox)

        result = tools["read_workspace_file"].invoke({"file_path": "/etc/passwd"})

        assert "ERROR:" in result
        assert "absolute" in result.lower()

    def test_missing_file_returns_error(self):
        sandbox = MagicMock()
        sandbox.read_file.return_value = None
        tools = _tool_map(sandbox)

        result = tools["read_workspace_file"].invoke({"file_path": "package.json"})

        assert "ERROR:" in result
        sandbox.read_file.assert_called_once_with("package.json")


class TestDeterministicSearchReplace:
    def test_zero_match_returns_error(self):
        sandbox = MagicMock()
        sandbox.read_file.return_value = "hello world"
        tools = _tool_map(sandbox)

        result = tools["deterministic_search_replace"].invoke(
            {
                "file_path": "app.ts",
                "old_text": "missing",
                "new_text": "new",
            }
        )

        assert "ERROR:" in result
        assert "not found" in result.lower()
        sandbox.write_file.assert_not_called()

    def test_multiple_match_returns_error(self):
        sandbox = MagicMock()
        sandbox.read_file.return_value = "dup\ndup\n"
        tools = _tool_map(sandbox)

        result = tools["deterministic_search_replace"].invoke(
            {
                "file_path": "app.ts",
                "old_text": "dup",
                "new_text": "safe",
            }
        )

        assert "ERROR:" in result
        assert "multiple times" in result.lower()
        sandbox.write_file.assert_not_called()

    def test_success_writes_file_and_tracks_changed_path(self):
        sandbox = MagicMock()
        sandbox.read_file.return_value = "const x = 1;\r\n"
        touched_files = set()
        tools = _tool_map(sandbox, touched_files=touched_files)

        result = tools["deterministic_search_replace"].invoke(
            {
                "file_path": "routes/login.ts",
                "old_text": "const x = 1;\n",
                "new_text": "const x = 2;\n",
            }
        )

        assert result == "SUCCESS: File modified: routes/login.ts"
        sandbox.write_file.assert_called_once_with("routes/login.ts", "const x = 2;\r\n")
        assert touched_files == {"routes/login.ts"}


class TestRunDependencyInstall:
    def test_success_runs_npm_install(self):
        sandbox = MagicMock()
        sandbox.run.return_value = CommandResult(
            exit_code=0,
            stdout="ok",
            stderr="",
            duration_seconds=1.0,
        )
        tools = _tool_map(sandbox)

        result = tools["run_dependency_install"].invoke({})

        sandbox.run.assert_called_once_with("npm install --package-lock=true", timeout=600)
        assert result.startswith("SUCCESS:")

    def test_failure_includes_stdout_and_stderr(self):
        sandbox = MagicMock()
        sandbox.run.return_value = CommandResult(
            exit_code=1,
            stdout="partial",
            stderr="dependency conflict",
            duration_seconds=1.0,
        )
        tools = _tool_map(sandbox)

        result = tools["run_dependency_install"].invoke({})

        assert result.startswith("FAILURE:")
        assert "dependency conflict" in result
        assert "partial" in result


class TestRunSecurityScan:
    def test_missing_workspace_volume_returns_failure(self):
        sandbox = MagicMock()
        tools = _tool_map(sandbox, target_identifiers={"CVE-2021-44228"})

        result = tools["run_security_scan"].invoke({})

        assert result.startswith("FAILURE:")
        assert "workspace_volume" in result

    def test_odc_command_mounts_workspace_and_cache_dir(self):
        sandbox = MagicMock()
        sandbox._workspace_volume = "agent_workspace_deadbeef"
        sandbox.read_file.return_value = json.dumps({"dependencies": []})
        tools = _tool_map(sandbox, target_identifiers={"CVE-2021-44228"})

        proc = MagicMock(spec=subprocess.CompletedProcess)
        proc.returncode = 0
        proc.stdout = ""
        proc.stderr = ""

        with patch("src.orchestrator.remedy_tools.shutil.which", return_value="docker"), patch(
            "src.orchestrator.remedy_tools.subprocess.run",
            return_value=proc,
        ) as mock_run:
            tools["run_security_scan"].invoke({})

        cmd = mock_run.call_args[0][0]
        assert "agent_workspace_deadbeef:/scan" in cmd
        expected_cache = str((Path(__file__).resolve().parents[1] / "data" / "cache").resolve())
        assert f"{expected_cache}:/usr/share/dependency-check/data" in cmd
        assert "--noupdate" in cmd

    def test_odc_command_can_fallback_to_named_cache_volume(self):
        sandbox = MagicMock()
        sandbox._workspace_volume = "agent_workspace_deadbeef"
        sandbox.read_file.return_value = json.dumps({"dependencies": []})
        tools = _tool_map(sandbox, target_identifiers={"CVE-2021-44228"})

        proc = MagicMock(spec=subprocess.CompletedProcess)
        proc.returncode = 0
        proc.stdout = ""
        proc.stderr = ""

        with patch("src.orchestrator.remedy_tools._resolve_odc_cache_source", return_value="odc-cache"), patch(
            "src.orchestrator.remedy_tools.shutil.which",
            return_value="docker",
        ), patch(
            "src.orchestrator.remedy_tools.subprocess.run",
            return_value=proc,
        ) as mock_run:
            tools["run_security_scan"].invoke({})

        cmd = mock_run.call_args[0][0]
        assert "odc-cache:/usr/share/dependency-check/data" in cmd

    def test_remaining_target_cve_returns_failure(self):
        sandbox = MagicMock()
        sandbox._workspace_volume = "agent_workspace_deadbeef"
        sandbox.read_file.return_value = json.dumps(
            {
                "dependencies": [
                    {
                        "fileName": "lodash.tgz",
                        "packages": [{"id": "pkg:npm/lodash@4.17.15"}],
                        "vulnerabilities": [{"name": "CVE-2021-44228"}],
                    }
                ]
            }
        )
        tools = _tool_map(sandbox, target_identifiers={"CVE-2021-44228"})

        proc = MagicMock(spec=subprocess.CompletedProcess)
        proc.returncode = 0
        proc.stdout = ""
        proc.stderr = ""

        with patch("src.orchestrator.remedy_tools.shutil.which", return_value="docker"), patch(
            "src.orchestrator.remedy_tools.subprocess.run",
            return_value=proc,
        ):
            result = tools["run_security_scan"].invoke({})

        assert result.startswith("FAILURE:")
        assert "CVE-2021-44228" in result

    def test_remaining_target_ghsa_returns_failure(self):
        sandbox = MagicMock()
        sandbox._workspace_volume = "agent_workspace_deadbeef"
        sandbox.read_file.return_value = json.dumps(
            {
                "dependencies": [
                    {
                        "fileName": "once.tgz",
                        "packages": [{"id": "pkg:npm/once@1.1.2"}],
                        "vulnerabilities": [{"name": "GHSA-vpq2-c234-7xj6"}],
                    }
                ]
            }
        )
        tools = _tool_map(sandbox, target_identifiers={"GHSA-VPQ2-C234-7XJ6"})

        proc = MagicMock(spec=subprocess.CompletedProcess)
        proc.returncode = 0
        proc.stdout = ""
        proc.stderr = ""

        with patch("src.orchestrator.remedy_tools.shutil.which", return_value="docker"), patch(
            "src.orchestrator.remedy_tools.subprocess.run",
            return_value=proc,
        ):
            result = tools["run_security_scan"].invoke({})

        assert result.startswith("FAILURE:")
        assert "GHSA-VPQ2-C234-7XJ6" in result

    def test_no_remaining_targets_returns_success(self):
        sandbox = MagicMock()
        sandbox._workspace_volume = "agent_workspace_deadbeef"
        sandbox.read_file.return_value = json.dumps({"dependencies": []})
        tools = _tool_map(sandbox, target_identifiers={"CVE-2021-44228"})

        proc = MagicMock(spec=subprocess.CompletedProcess)
        proc.returncode = 0
        proc.stdout = ""
        proc.stderr = ""

        with patch("src.orchestrator.remedy_tools.shutil.which", return_value="docker"), patch(
            "src.orchestrator.remedy_tools.subprocess.run",
            return_value=proc,
        ):
            result = tools["run_security_scan"].invoke({})

        assert result.startswith("SUCCESS:")


class TestRunUnitTests:
    def test_success_runs_npm_test(self):
        sandbox = MagicMock()
        sandbox.run.return_value = CommandResult(
            exit_code=0,
            stdout="passing",
            stderr="",
            duration_seconds=2.0,
        )
        tools = _tool_map(sandbox)

        result = tools["run_unit_tests"].invoke({})

        sandbox.run.assert_called_once_with("npm test", timeout=600)
        assert result.startswith("SUCCESS:")

    def test_failure_includes_stdout_and_stderr(self):
        sandbox = MagicMock()
        sandbox.run.return_value = CommandResult(
            exit_code=1,
            stdout="failing output",
            stderr="AssertionError",
            duration_seconds=2.0,
        )
        tools = _tool_map(sandbox)

        result = tools["run_unit_tests"].invoke({})

        assert result.startswith("FAILURE:")
        assert "AssertionError" in result
        assert "failing output" in result
