"""
tests/test_remedy_tools.py - Direct unit tests for Phase 5 specialist toolbelts.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

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
    enable_registry_lookup=False,
    attempted_versions_by_package=None,
    override_required_packages=None,
    require_planning_answers=False,
    planning_state=None,
    execution_state=None,
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
        enable_registry_lookup=enable_registry_lookup,
        attempted_versions_by_package=attempted_versions_by_package,
        override_required_packages=override_required_packages,
        require_planning_answers=require_planning_answers,
        planning_state=planning_state,
        execution_state=execution_state,
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
            "read_workspace_file",
            "search_codebase_pattern",
            "inspect_ast_symbol",
            "deterministic_search_replace",
            "revert_workspace_file",
            "validate_code_syntax",
            "run_typecheck",
        }

    def test_update_toolbelt_excludes_registry_lookup(self):
        sandbox = MagicMock()
        tools = _update_tool_map(sandbox, enable_registry_lookup=True)

        assert "view_npm_package_versions" not in tools


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

    def test_rejects_direct_dependency_edit_when_overrides_required(self):
        sandbox = MagicMock()
        tools = _update_tool_map(
            sandbox,
            package_manifest_paths={"cookie": ["package.json"]},
            override_required_packages={"cookie"},
        )

        result = tools["modify_npm_dependency"].invoke(
            {
                "package_name": "cookie",
                "target_version": "0.7.0",
                "dependency_type": "dependencies",
                "manifest_path": "package.json",
            }
        )

        assert result.startswith("ERROR:")
        assert "constrained to npm overrides" in result
        assert "dependency_type='overrides'" in result
        sandbox.run.assert_not_called()

    def test_allows_override_edit_when_overrides_required(self):
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
            package_manifest_paths={"cookie": ["package.json"]},
            override_required_packages={"cookie"},
        )

        result = tools["modify_npm_dependency"].invoke(
            {
                "package_name": "cookie",
                "target_version": "0.7.0",
                "dependency_type": "overrides",
                "manifest_path": "package.json",
            }
        )

        assert result.startswith("SUCCESS:")
        assert "overrides.cookie" in result
        assert touched_files == {"package.json"}
        sandbox.run.assert_called_once()


class TestRevertWorkspaceFile:
    def test_revert_entire_file_success(self, tmp_path):
        sandbox = MagicMock()
        touched_files = {"src/index.js"}

        host_repo = tmp_path / "host"
        host_repo.mkdir()
        baseline_file = host_repo / "src" / "index.js"
        baseline_file.parent.mkdir(parents=True, exist_ok=True)
        baseline_file.write_text("console.log('original');", encoding="utf-8")

        tools = _workaround_tool_map(sandbox, touched_files=touched_files, host_repo_root=host_repo)

        result = tools["revert_workspace_file"].invoke({"file_path": "src/index.js"})

        assert result.startswith("SUCCESS:")
        sandbox.write_file.assert_called_once_with("src/index.js", "console.log('original');")
        assert "src/index.js" not in touched_files

    def test_revert_package_json_package_name_provided(self, tmp_path):
        sandbox = MagicMock()
        touched_files = {"package.json"}

        host_repo = tmp_path / "host"
        host_repo.mkdir()
        baseline_file = host_repo / "package.json"
        baseline_json = {
            "dependencies": {
                "lodash": "4.17.20",
                "axios": "0.21.1"
            },
            "devDependencies": {
                "jest": "26.6.3"
            }
        }
        import json
        baseline_file.write_text(json.dumps(baseline_json, indent=2), encoding="utf-8")

        sandbox_json = {
            "dependencies": {
                "lodash": "4.17.21",
                "axios": "0.22.0",
                "newpkg": "1.0.0"
            },
            "devDependencies": {
                "jest": "27.0.0"
            }
        }
        sandbox.read_file.return_value = json.dumps(sandbox_json, indent=2)

        tools = _update_tool_map(sandbox, touched_files=touched_files, host_repo_root=host_repo)

        result = tools["revert_workspace_file"].invoke({
            "file_path": "package.json",
            "package_name": "axios"
        })

        assert result.startswith("SUCCESS:")
        assert sandbox.write_file.called
        written_content = sandbox.write_file.call_args[0][1]
        written_json = json.loads(written_content)

        assert written_json["dependencies"]["lodash"] == "4.17.21"
        assert written_json["dependencies"]["axios"] == "0.21.1"
        assert written_json["devDependencies"]["jest"] == "27.0.0"
        assert written_json["dependencies"]["newpkg"] == "1.0.0"
        assert "package.json" in touched_files

    def test_revert_package_json_package_name_removes_added_package(self, tmp_path):
        sandbox = MagicMock()
        touched_files = {"package.json"}

        host_repo = tmp_path / "host"
        host_repo.mkdir()
        baseline_file = host_repo / "package.json"
        baseline_json = {
            "dependencies": {
                "lodash": "4.17.20"
            }
        }
        import json
        baseline_file.write_text(json.dumps(baseline_json, indent=2), encoding="utf-8")

        sandbox_json = {
            "dependencies": {
                "lodash": "4.17.20",
                "newpkg": "1.0.0"
            }
        }
        sandbox.read_file.return_value = json.dumps(sandbox_json, indent=2)

        tools = _update_tool_map(sandbox, touched_files=touched_files, host_repo_root=host_repo)

        result = tools["revert_workspace_file"].invoke({
            "file_path": "package.json",
            "package_name": "newpkg"
        })

        assert result.startswith("SUCCESS:")
        written_content = sandbox.write_file.call_args[0][1]
        written_json = json.loads(written_content)

        assert "newpkg" not in written_json["dependencies"]
        assert "package.json" not in touched_files

    def test_revert_non_package_json_with_package_name_rejected(self, tmp_path):
        sandbox = MagicMock()
        host_repo = tmp_path / "host"
        host_repo.mkdir()
        baseline_file = host_repo / "src" / "index.js"
        baseline_file.parent.mkdir(parents=True, exist_ok=True)
        baseline_file.write_text("console.log('original');", encoding="utf-8")

        tools = _workaround_tool_map(sandbox, host_repo_root=host_repo)

        result = tools["revert_workspace_file"].invoke({
            "file_path": "src/index.js",
            "package_name": "axios"
        })

        assert result.startswith("ERROR:")
        assert "only be specified for package.json files" in result


class TestValidateManifestSync:
    def test_validation_is_rejected_after_the_first_call_in_worker_run(self):
        sandbox = MagicMock()
        sandbox.run.return_value = CommandResult(
            exit_code=0,
            stdout="ok",
            stderr="",
            duration_seconds=1.0,
        )
        execution_state = {"edits_started": False, "validation_calls": 0}
        tools = _update_tool_map(sandbox, execution_state=execution_state)

        edit_result = tools["modify_npm_dependency"].invoke(
            {
                "package_name": "lodash",
                "target_version": "4.17.22",
                "dependency_type": "dependencies",
                "manifest_path": "package.json",
            }
        )
        first_validation = tools["validate_manifest_sync"].invoke({})
        second_validation = tools["validate_manifest_sync"].invoke({})

        assert edit_result.startswith("SUCCESS:")
        assert first_validation.startswith("SUCCESS:")
        assert second_validation.startswith("ERROR:")
        assert "only be called once" in second_validation

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
