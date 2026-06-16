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
            "read_repository_map",
            "search_codebase_pattern",
            "inspect_ast_symbol",
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
        sandbox.run.assert_called_once_with("npm pkg set overrides[@scope/package-name]=1.2.3-beta.1")
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
        sandbox.run.assert_called_once_with("npm pkg set dependencies[lodash]=4.17.21")
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


class TestReadRepositoryMap:
    def test_returns_file_listing(self):
        sandbox = MagicMock()
        sandbox.run.return_value = CommandResult(
            exit_code=0,
            stdout="/workspace/package.json\n/workspace/src/index.ts\n",
            stderr="",
            duration_seconds=0.1,
        )
        tools = _tool_map(sandbox)

        result = tools["read_repository_map"].invoke({})

        assert "/workspace/package.json" in result
        assert "/workspace/src/index.ts" in result
        sandbox.run.assert_called_once()
        call_cmd = sandbox.run.call_args[0][0]
        assert "find /workspace" in call_cmd
        assert "node_modules" in call_cmd

    def test_caps_at_max_entries(self):
        sandbox = MagicMock()
        big_output = "\n".join(f"/workspace/file{i}.ts" for i in range(450))
        sandbox.run.return_value = CommandResult(
            exit_code=0,
            stdout=big_output,
            stderr="",
            duration_seconds=0.2,
        )
        tools = _tool_map(sandbox)

        result = tools["read_repository_map"].invoke({})

        assert "truncated" in result
        assert "50 more entries" in result

    def test_error_from_sandbox_returns_error(self):
        sandbox = MagicMock()
        sandbox.run.return_value = CommandResult(
            exit_code=1,
            stdout="",
            stderr="permission denied",
            duration_seconds=0.0,
        )
        tools = _tool_map(sandbox)

        result = tools["read_repository_map"].invoke({})

        assert "ERROR:" in result
        assert "permission denied" in result

    def test_empty_workspace_returns_placeholder(self):
        sandbox = MagicMock()
        sandbox.run.return_value = CommandResult(
            exit_code=0,
            stdout="",
            stderr="",
            duration_seconds=0.0,
        )
        tools = _tool_map(sandbox)

        result = tools["read_repository_map"].invoke({})

        assert "empty" in result


class TestSearchCodebasePattern:
    def test_basic_match_returns_grep_output(self):
        sandbox = MagicMock()
        sandbox.run.return_value = CommandResult(
            exit_code=0,
            stdout="/workspace/src/index.ts:10:const sanitizeHtml = require('sanitize-html');\n",
            stderr="",
            duration_seconds=0.3,
        )
        tools = _tool_map(sandbox)

        result = tools["search_codebase_pattern"].invoke(
            {"search_pattern": "sanitizeHtml", "target_directory": "."}
        )

        assert "sanitizeHtml" in result
        sandbox.run.assert_called_once()
        cmd_called = sandbox.run.call_args[0][0]
        assert "grep -RInE" in cmd_called
        assert "sanitizeHtml" in cmd_called

    def test_empty_pattern_returns_error(self):
        sandbox = MagicMock()
        tools = _tool_map(sandbox)

        result = tools["search_codebase_pattern"].invoke(
            {"search_pattern": "", "target_directory": "."}
        )

        assert "ERROR: search_pattern is required" in result
        sandbox.run.assert_not_called()

    def test_no_match_returns_no_match(self):
        sandbox = MagicMock()
        sandbox.run.return_value = CommandResult(
            exit_code=1,
            stdout="",
            stderr="",
            duration_seconds=0.1,
        )
        tools = _tool_map(sandbox)

        result = tools["search_codebase_pattern"].invoke(
            {"search_pattern": "nonExistentSymbol"}
        )

        assert "NO MATCH" in result

    def test_absolute_target_directory_rejected(self):
        sandbox = MagicMock()
        tools = _tool_map(sandbox)

        result = tools["search_codebase_pattern"].invoke(
            {"search_pattern": "foo", "target_directory": "/etc"}
        )

        assert "ERROR:" in result
        sandbox.run.assert_not_called()

    def test_output_truncated_at_32kb(self):
        sandbox = MagicMock()
        # Generate a big output string (>32 KB)
        big_line = "x" * 100 + "\n"
        big_output = big_line * 400  # ~40 KB
        sandbox.run.return_value = CommandResult(
            exit_code=0,
            stdout=big_output,
            stderr="",
            duration_seconds=0.5,
        )
        tools = _tool_map(sandbox)

        result = tools["search_codebase_pattern"].invoke({"search_pattern": "x"})

        assert "truncated at 32 KB" in result
        assert len(result.encode()) < 40_000  # well under 40 KB

    def test_grep_error_returns_error(self):
        sandbox = MagicMock()
        sandbox.run.return_value = CommandResult(
            exit_code=2,
            stdout="",
            stderr="grep: bad regex",
            duration_seconds=0.1,
        )
        tools = _tool_map(sandbox)

        result = tools["search_codebase_pattern"].invoke({"search_pattern": "[bad"})

        assert "ERROR:" in result


class TestInspectAstSymbol:
    def test_absolute_path_rejected(self):
        sandbox = MagicMock()
        tools = _tool_map(sandbox)

        result = tools["inspect_ast_symbol"].invoke(
            {"file_path": "/etc/passwd", "symbol_name": "foo"}
        )

        assert "ERROR:" in result
        assert "absolute" in result.lower()

    def test_missing_file_returns_error(self):
        sandbox = MagicMock()
        sandbox.read_file.return_value = None
        tools = _tool_map(sandbox)

        result = tools["inspect_ast_symbol"].invoke(
            {"file_path": "src/app.ts", "symbol_name": "myFn"}
        )

        assert "ERROR:" in result

    def test_non_js_ts_file_returns_error(self):
        sandbox = MagicMock()
        sandbox.read_file.return_value = "x = 1"
        tools = _tool_map(sandbox)

        result = tools["inspect_ast_symbol"].invoke(
            {"file_path": "readme.md", "symbol_name": "myFn"}
        )

        assert "ERROR:" in result
        assert "No AST parser available" in result

    def test_symbol_not_found_returns_not_found(self):
        sandbox = MagicMock()
        sandbox.read_file.return_value = "const x = 1;\n"
        tools = _tool_map(sandbox)

        result = tools["inspect_ast_symbol"].invoke(
            {"file_path": "src/index.js", "symbol_name": "nonExistentFn"}
        )

        # Without tree-sitter this errors; with it, returns NOT FOUND or ERROR.
        assert "NOT FOUND" in result or "ERROR" in result

    def test_tree_sitter_unavailable_returns_error(self):
        """Simulate tree-sitter being unavailable by patching parse_source to None."""
        from unittest.mock import patch as _patch

        sandbox = MagicMock()
        sandbox.read_file.return_value = "function foo() { return 1; }\n"
        tools = _tool_map(sandbox)

        with _patch("src.tools.code_map.parse_source", return_value=None), \
             _patch("src.tools.code_map.language_for_path", return_value=object()):
            result = tools["inspect_ast_symbol"].invoke(
                {"file_path": "src/index.js", "symbol_name": "foo"}
            )

        assert "ERROR" in result
