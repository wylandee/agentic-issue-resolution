"""
tests/test_remedy_tools.py - Direct unit tests for Phase 5 specialist toolbelts.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from src.contracts.schemas import CommandResult
from src.orchestrator.remedy_tools import (
    build_update_toolbelt,
    build_workaround_toolbelt,
)


def _update_tool_map(
    sandbox,
    touched_files=None,
    host_repo_root=None,
    manifest_paths=None,
    package_manifest_paths=None,
):
    if touched_files is None:
        touched_files = set()
    if host_repo_root is None:
        host_repo_root = Path("/dummy/repo/root")
    if manifest_paths is None:
        manifest_paths = ["package.json"]
    if package_manifest_paths is None:
        package_manifest_paths = {"lodash": manifest_paths}
    tools = build_update_toolbelt(
        sandbox,
        touched_files,
        host_repo_root,
        manifest_paths,
        package_manifest_paths,
    )
    return {tool.name: tool for tool in tools}


def _workaround_tool_map(sandbox, touched_files=None, host_repo_root=None):
    if touched_files is None:
        touched_files = set()
    if host_repo_root is None:
        host_repo_root = Path("/dummy/repo/root")
    tools = build_workaround_toolbelt(sandbox, touched_files, host_repo_root)
    return {tool.name: tool for tool in tools}


class TestToolbeltFactories:
    def test_update_toolbelt_is_strictly_scoped(self):
        sandbox = MagicMock()
        tools = _update_tool_map(sandbox)

        assert set(tools) == {
            "read_repository_map",
            "modify_npm_dependency",
            "revert_workspace_file",
            "validate_manifest_sync",
        }

    def test_workaround_toolbelt_is_strictly_scoped(self):
        sandbox = MagicMock()
        tools = _workaround_tool_map(sandbox)

        assert set(tools) == {
            "read_repository_map",
            "search_codebase_pattern",
            "inspect_ast_symbol",
            "deterministic_search_replace",
            "revert_workspace_file",
            "validate_code_syntax",
        }


class TestModifyNpmDependency:
    def test_success_tracks_manifest_path(self):
        sandbox = MagicMock()
        sandbox.run.return_value = CommandResult(
            exit_code=0,
            stdout="ok",
            stderr="",
            duration_seconds=0.5,
        )
        touched_files = set()
        tools = _update_tool_map(
            sandbox,
            touched_files=touched_files,
            manifest_paths=["frontend/package.json"],
        )

        result = tools["modify_npm_dependency"].invoke(
            {
                "package_name": "lodash",
                "target_version": "4.17.21",
                "dependency_type": "dependencies",
                "manifest_path": "frontend/package.json",
            }
        )

        assert result.startswith("SUCCESS:")
        assert "frontend/package.json" in result
        assert touched_files == {"frontend/package.json"}
        sandbox.run.assert_called_once()
        assert "cd /workspace/frontend" in sandbox.run.call_args[0][0]

    def test_rejects_manifest_outside_allowed_batch_targets(self):
        sandbox = MagicMock()
        tools = _update_tool_map(
            sandbox,
            manifest_paths=["package.json", "frontend/package.json"],
            package_manifest_paths={"lodash": ["package.json"]},
        )

        result = tools["modify_npm_dependency"].invoke(
            {
                "package_name": "lodash",
                "target_version": "4.17.21",
                "dependency_type": "dependencies",
                "manifest_path": "frontend/package.json",
            }
        )

        assert result.startswith("ERROR:")
        assert "not an allowed target for package 'lodash'" in result
        assert "package.json" in result
        sandbox.run.assert_not_called()

    def test_rejects_manifest_not_allowed_for_package_in_mixed_batch(self):
        sandbox = MagicMock()
        tools = _update_tool_map(
            sandbox,
            manifest_paths=["package.json", "frontend/package.json"],
            package_manifest_paths={
                "lodash": ["package.json"],
                "axios": ["frontend/package.json"],
            },
        )

        result = tools["modify_npm_dependency"].invoke(
            {
                "package_name": "lodash",
                "target_version": "4.17.21",
                "dependency_type": "dependencies",
                "manifest_path": "frontend/package.json",
            }
        )

        assert result.startswith("ERROR:")
        assert "not an allowed target for package 'lodash'" in result
        assert "package.json" in result
        sandbox.run.assert_not_called()

    def test_allows_package_specific_manifest_in_mixed_batch(self):
        sandbox = MagicMock()
        sandbox.run.return_value = CommandResult(
            exit_code=0,
            stdout="ok",
            stderr="",
            duration_seconds=0.5,
        )
        touched_files = set()
        tools = _update_tool_map(
            sandbox,
            touched_files=touched_files,
            manifest_paths=["package.json", "frontend/package.json"],
            package_manifest_paths={
                "lodash": ["package.json"],
                "axios": ["frontend/package.json"],
            },
        )

        result = tools["modify_npm_dependency"].invoke(
            {
                "package_name": "axios",
                "target_version": "1.7.4",
                "dependency_type": "dependencies",
                "manifest_path": "frontend/package.json",
            }
        )

        assert result.startswith("SUCCESS:")
        assert "frontend/package.json" in result
        assert touched_files == {"frontend/package.json"}
        sandbox.run.assert_called_once()

    def test_rejects_unknown_package_name_for_batch(self):
        sandbox = MagicMock()
        tools = _update_tool_map(
            sandbox,
            manifest_paths=["package.json"],
            package_manifest_paths={"lodash": ["package.json"]},
        )

        result = tools["modify_npm_dependency"].invoke(
            {
                "package_name": "express",
                "target_version": "4.21.0",
                "dependency_type": "dependencies",
                "manifest_path": "package.json",
            }
        )

        assert result.startswith("ERROR:")
        assert "package_name 'express' is not an allowed target for this batch" in result
        sandbox.run.assert_not_called()

    def test_invalid_manifest_path_rejected(self):
        sandbox = MagicMock()
        tools = _update_tool_map(sandbox)

        result = tools["modify_npm_dependency"].invoke(
            {
                "package_name": "lodash",
                "target_version": "4.17.21",
                "dependency_type": "dependencies",
                "manifest_path": "frontend/package-lock.json",
            }
        )

        assert result == "ERROR: manifest_path must point to a package.json file."


class TestValidateManifestSync:
    def test_success_runs_ignore_scripts_for_each_manifest_directory(self):
        sandbox = MagicMock()
        sandbox.run.return_value = CommandResult(
            exit_code=0,
            stdout="ok",
            stderr="",
            duration_seconds=1.0,
        )
        tools = _update_tool_map(
            sandbox,
            manifest_paths=["package.json", "frontend/package.json"],
        )

        result = tools["validate_manifest_sync"].invoke({})

        assert result.startswith("SUCCESS:")
        assert sandbox.run.call_count == 2
        commands = [call.args[0] for call in sandbox.run.call_args_list]
        assert all("--package-lock-only --ignore-scripts" in cmd for cmd in commands)
        assert any("cd /workspace/frontend" in cmd for cmd in commands)

    def test_failure_surfaces_stderr(self):
        sandbox = MagicMock()
        sandbox.run.return_value = CommandResult(
            exit_code=1,
            stdout="partial",
            stderr="ERESOLVE unable to resolve dependency tree",
            duration_seconds=1.0,
        )
        tools = _update_tool_map(sandbox)

        result = tools["validate_manifest_sync"].invoke({})

        assert result.startswith("FAILURE:")
        assert "ERESOLVE" in result
        assert "partial" in result


class TestValidateCodeSyntax:
    def test_js_uses_node_check(self):
        sandbox = MagicMock()
        sandbox.run.return_value = CommandResult(
            exit_code=0,
            stdout="",
            stderr="",
            duration_seconds=0.1,
        )
        tools = _workaround_tool_map(sandbox)

        result = tools["validate_code_syntax"].invoke({"file_path": "src/index.js"})

        assert result.startswith("SUCCESS:")
        sandbox.run.assert_called_once()
        assert "node -c" in sandbox.run.call_args[0][0]

    def test_typescript_uses_tsc(self):
        sandbox = MagicMock()
        sandbox.run.return_value = CommandResult(
            exit_code=0,
            stdout="",
            stderr="",
            duration_seconds=0.1,
        )
        tools = _workaround_tool_map(sandbox)

        result = tools["validate_code_syntax"].invoke({"file_path": "src/index.ts"})

        assert result.startswith("SUCCESS:")
        assert "esbuild" in sandbox.run.call_args[0][0]

    def test_failure_surfaces_syntax_output(self):
        sandbox = MagicMock()
        sandbox.run.return_value = CommandResult(
            exit_code=1,
            stdout="routes/login.ts(10,2): error TS1005",
            stderr="}",
            duration_seconds=0.1,
        )
        tools = _workaround_tool_map(sandbox)

        result = tools["validate_code_syntax"].invoke({"file_path": "routes/login.ts"})

        assert result.startswith("FAILURE:")
        assert "TS1005" in result

    def test_unsupported_extension_returns_error(self):
        sandbox = MagicMock()
        tools = _workaround_tool_map(sandbox)

        result = tools["validate_code_syntax"].invoke({"file_path": "README.md"})

        assert result.startswith("ERROR:")

    def test_absolute_path_rejected(self):
        sandbox = MagicMock()
        tools = _workaround_tool_map(sandbox)

        result = tools["validate_code_syntax"].invoke({"file_path": "/etc/passwd"})

        assert result.startswith("ERROR:")
