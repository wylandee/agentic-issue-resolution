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


def _tool_map(sandbox, touched_files=None, target_identifiers=None, host_repo_root=None):
    if touched_files is None:
        touched_files = set()
    if target_identifiers is None:
        target_identifiers = set()
    if host_repo_root is None:
        host_repo_root = Path("/dummy/repo/root")
    tools = build_agent_tools(sandbox, touched_files, target_identifiers, host_repo_root)
    return {tool.name: tool for tool in tools}


class TestFactory:
    def test_factory_returns_all_expected_tools(self):
        sandbox = MagicMock()
        sandbox._workspace_volume = "agent_workspace_deadbeef"

        tools = _tool_map(sandbox)

        assert set(tools) == {
            "read_workspace_file",
            "deterministic_search_replace",
            "revert_workspace_file",
            "modify_npm_dependency",
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

class TestRevertWorkspaceFile:
    def test_absolute_path_rejected(self):
        sandbox = MagicMock()
        tools = _tool_map(sandbox)

        result = tools["revert_workspace_file"].invoke({"file_path": "/etc/passwd"})

        assert "ERROR:" in result
        assert "absolute" in result.lower()

    def test_path_traversal_rejected(self):
        sandbox = MagicMock()
        tools = _tool_map(sandbox)

        result = tools["revert_workspace_file"].invoke({"file_path": "foo/../../bar"})

        assert "ERROR:" in result
        assert "traversal" in result.lower()

    def test_missing_host_file_returns_error(self, tmp_path):
        sandbox = MagicMock()
        tools = _tool_map(sandbox, host_repo_root=tmp_path)

        result = tools["revert_workspace_file"].invoke({"file_path": "nonexistent.json"})

        assert "ERROR:" in result
        assert "does not exist on host" in result.lower()

    def test_success_reverts_file_and_untracks(self, tmp_path):
        sandbox = MagicMock()
        baseline_file = tmp_path / "package.json"
        baseline_content = '{"dependencies": {"lodash": "^4.17.15"}}'
        baseline_file.write_text(baseline_content, encoding="utf-8")

        touched_files = {"package.json", "other_file.js"}
        tools = _tool_map(sandbox, touched_files=touched_files, host_repo_root=tmp_path)

        result = tools["revert_workspace_file"].invoke({"file_path": "package.json"})

        assert result == "SUCCESS: Reverted workspace file 'package.json' to its baseline state."
        sandbox.write_file.assert_called_once_with("package.json", baseline_content)
        assert touched_files == {"other_file.js"}


class TestModifyNpmDependency:
    def test_invalid_package_name_rejected(self):
        sandbox = MagicMock()
        tools = _tool_map(sandbox)

        result = tools["modify_npm_dependency"].invoke(
            {
                "package_name": "lodash; rm -rf /",
                "target_version": "4.17.21",
                "dependency_type": "dependencies",
            }
        )
        assert "ERROR: Invalid package_name" in result
        sandbox.run.assert_not_called()

    def test_invalid_version_rejected(self):
        sandbox = MagicMock()
        tools = _tool_map(sandbox)

        result = tools["modify_npm_dependency"].invoke(
            {
                "package_name": "lodash",
                "target_version": "4.17.21 | echo hacked",
                "dependency_type": "dependencies",
            }
        )
        assert "ERROR: Invalid target_version" in result
        sandbox.run.assert_not_called()

    def test_invalid_dependency_type_rejected(self):
        sandbox = MagicMock()
        tools = _tool_map(sandbox)

        result = tools["modify_npm_dependency"].invoke(
            {
                "package_name": "lodash",
                "target_version": "4.17.21",
                "dependency_type": "invalid_type",
            }
        )
        assert "ERROR: dependency_type must be strictly" in result
        sandbox.run.assert_not_called()

    def test_success_runs_command_and_tracks(self):
        sandbox = MagicMock()
        sandbox.run.return_value = CommandResult(
            exit_code=0,
            stdout="ok",
            stderr="",
            duration_seconds=0.5,
        )
        touched_files = set()
        tools = _tool_map(sandbox, touched_files=touched_files)

        result = tools["modify_npm_dependency"].invoke(
            {
                "package_name": "@scope/package-name",
                "target_version": "1.2.3-beta.1",
                "dependency_type": "overrides",
            }
        )

        assert "SUCCESS:" in result
        assert "overrides.@scope/package-name" in result
        sandbox.run.assert_called_once_with("npm pkg set overrides.@scope/package-name=1.2.3-beta.1")
        assert touched_files == {"package.json"}

    def test_failure_returns_error_details(self):
        sandbox = MagicMock()
        sandbox.run.return_value = CommandResult(
            exit_code=1,
            stdout="some stdout",
            stderr="some error",
            duration_seconds=0.5,
        )
        touched_files = set()
        tools = _tool_map(sandbox, touched_files=touched_files)

        result = tools["modify_npm_dependency"].invoke(
            {
                "package_name": "lodash",
                "target_version": "4.17.21",
                "dependency_type": "dependencies",
            }
        )

        assert "FAILURE:" in result
        assert "exit 1" in result
        assert "some error" in result
        sandbox.run.assert_called_once_with("npm pkg set dependencies.lodash=4.17.21")
        assert "package.json" not in touched_files


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

    def test_odc_command_mounts_workspace_and_cache_volume(self):
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
        assert "odc-cache:/usr/share/dependency-check/data" in cmd
        assert "--noupdate" in cmd

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
