"""
tests/test_remedy_tools.py - Direct unit tests for Phase 5 specialist toolbelts.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from remediation_engine.contracts.schemas import CommandResult
from remediation_engine.orchestration.remedy_tools import (
    _make_deterministic_replace_ast_symbol_tool,
    _make_run_targeted_test_tool,
    _make_validate_code_syntax_tool,
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


def _workaround_tool_map(sandbox, touched_files=None, host_repo_root=None, plan_state=None):
    if touched_files is None:
        touched_files = set()
    if host_repo_root is None:
        host_repo_root = Path("/dummy/repo/root")
    if plan_state is None:
        plan_state = {"local_investigation_complete": True, "web_search_performed": True}
    tools = build_workaround_toolbelt(sandbox, touched_files, host_repo_root, plan_state=plan_state)
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
            "record_plan",
            "record_targeted_test_substitution",
            "search_web",
            "read_web_page",
            "read_repository_map",
            "read_workspace_file",
            "search_codebase_pattern",
            "inspect_ast_symbol",
            "deterministic_apply_edit_set",
            "revert_workspace_file",
            "validate_workaround",
        }

    def test_targeted_test_uses_matching_npm_script_and_qa_target(self):
        sandbox = MagicMock()
        package_json = {
            "scripts": {
                "test": "npm run test:frontend && npm run test:api",
                "test:frontend": "cd frontend && npm run test",
                "test:api": 'node --test "test/api/**/*.test.ts"',
            }
        }

        def read_file(path):
            if path == "package.json":
                return json.dumps(package_json)
            if path == "src/auth.ts":
                return "export const auth = true;"
            if path == "test/api/2fa.test.ts":
                return "test('auth', () => {});"
            return None

        sandbox.read_file.side_effect = read_file
        sandbox.run.return_value = CommandResult(
            exit_code=0,
            stdout="targeted test passed",
            stderr="",
            duration_seconds=0.1,
        )
        tools = {
            tool.name: tool
            for tool in build_workaround_toolbelt(
                sandbox,
                set(),
                Path("/dummy/repo/root"),
                preferred_test_files=["test/api/2fa.test.ts"],
            )
        }

        corrected = tools["validate_workaround"].invoke(
            {
                "modified_files": ["src/auth.ts"],
                "targeted_test_file": "test/api/file-serving.test.ts",
                "runtime_smoke_file": "src/auth.ts",
            }
        )
        assert corrected.startswith("SUCCESS: Workaround validation gate passed")
        assert "canonical QA test 'test/api/2fa.test.ts'" in corrected

        compiled = tools["validate_workaround"].invoke(
            {"modified_files": ["src/auth.ts"], "targeted_test_file": "build/test/api/2fa.test.js"}
        )
        assert "Compiled test path" in compiled

        result = tools["validate_workaround"].invoke(
            {
                "modified_files": ["src/auth.ts"],
                "runtime_smoke_file": "src/auth.ts",
                "targeted_test_file": "test/api/2fa.test.ts",
            }
        )
        assert result.startswith("SUCCESS: Workaround validation gate passed")
        assert any(
            "node --test test/api/2fa.test.ts" in call.args[0]
            for call in sandbox.run.call_args_list
        )

    def test_targeted_mocha_test_uses_workspace_local_runner(self):
        sandbox = MagicMock()
        package_json = {
            "scripts": {
                "test": "npm run test:server",
                "test:server": "mocha -r tsx --recursive test/server/**/*.ts",
            }
        }

        def read_file(path):
            if path == "package.json":
                return json.dumps(package_json)
            if path == "src/auth.ts":
                return "export const auth = true;"
            if path == "test/server/insecuritySpec.ts":
                return "describe('security', () => {});"
            return None

        sandbox.read_file.side_effect = read_file
        sandbox.run.return_value = CommandResult(
            exit_code=0,
            stdout=json.dumps(
                {
                    "stats": {"tests": 1, "passes": 1, "failures": 0, "pending": 0},
                    "tests": [
                        {
                            "title": "cannot be bypassed by exploiting lack of recursive sanitization",
                            "fullTitle": "insecurity sanitizeSecure cannot be bypassed by exploiting lack of recursive sanitization",
                        }
                    ],
                    "passes": [
                        {
                            "title": "cannot be bypassed by exploiting lack of recursive sanitization",
                            "fullTitle": "insecurity sanitizeSecure cannot be bypassed by exploiting lack of recursive sanitization",
                        }
                    ],
                    "failures": [],
                    "pending": [],
                }
            ),
            stderr="",
            duration_seconds=0.1,
        )
        tools = {
            tool.name: tool
            for tool in build_workaround_toolbelt(
                sandbox,
                set(),
                Path("/dummy/repo/root"),
                preferred_test_files=["test/server/insecuritySpec.ts"],
            )
        }

        result = tools["validate_workaround"].invoke(
            {
                "modified_files": ["src/auth.ts"],
                "runtime_smoke_file": "src/auth.ts",
                "targeted_test_file": "test/server/insecuritySpec.ts",
                "targeted_test_name": "sanitizeSecure - cannot be bypassed by exploiting lack of recursive sanitization",
            }
        )

        assert result.startswith("SUCCESS: Workaround validation gate passed")
        assert any(
            "npx --no-install mocha -r tsx --recursive test/server/insecuritySpec.ts --grep "
            in call.args[0]
            and call.args[0].endswith("--reporter json")
            for call in sandbox.run.call_args_list
        )

    def test_validate_workaround_short_circuits_on_first_failed_gate(self):
        sandbox = MagicMock()
        sandbox.read_file.side_effect = lambda path: (
            "const value = 1;" if path == "src/auth.ts" else None
        )
        sandbox.run.return_value = CommandResult(
            exit_code=1,
            stdout="",
            stderr="SyntaxError: unexpected token",
            duration_seconds=0.1,
        )
        tools = _workaround_tool_map(
            sandbox,
            plan_state={"targeted_test_required": False},
        )

        result = tools["validate_workaround"].invoke(
            {"modified_files": ["src/auth.ts"], "runtime_smoke_file": "src/auth.ts"}
        )

        assert result.startswith("FAILURE: Workaround validation gate 'syntax'")
        assert sandbox.run.call_count == 1

    def test_targeted_mocha_rejects_zero_tests_as_invalid_selection(self):
        sandbox = MagicMock()
        sandbox.read_file.side_effect = lambda path: (
            json.dumps({"scripts": {"test": "mocha test/**/*.ts"}})
            if path == "package.json"
            else "describe('security', () => {});"
        )
        sandbox.run.return_value = CommandResult(
            exit_code=0,
            stdout=json.dumps(
                {
                    "stats": {"tests": 0, "passes": 0, "failures": 0, "pending": 0},
                    "tests": [],
                    "passes": [],
                    "failures": [],
                    "pending": [],
                }
            ),
            stderr="",
            duration_seconds=0.1,
        )
        tool = _make_run_targeted_test_tool(
            sandbox,
            preferred_test_files=["test/server/insecuritySpec.ts"],
        )

        result = tool.invoke(
            {
                "test_file": "test/server/insecuritySpec.ts",
                "test_name": "sanitizeSecure - cannot be bypassed by exploiting lack of recursive sanitization",
            }
        )

        assert result.startswith("ERROR: [INVALID_VALIDATION_INPUT]")
        assert (
            "did not match any test" in result
            or "no parseable test titles" in result
            or "no executed tests" in result
        )

    def test_targeted_mocha_accepts_suite_name_hint(self):
        sandbox = MagicMock()
        sandbox.read_file.side_effect = lambda path: (
            json.dumps({"scripts": {"test": "mocha test/**/*.ts"}})
            if path == "package.json"
            else "describe('b2bOrder', () => {});"
        )
        full_title = "b2bOrder deserializing arbitrary JSON should not solve rceChallenge"
        sandbox.run.return_value = CommandResult(
            exit_code=0,
            stdout=json.dumps(
                {
                    "stats": {"tests": 1, "passes": 1, "failures": 0, "pending": 0},
                    "tests": [
                        {
                            "title": "deserializing arbitrary JSON should not solve rceChallenge",
                            "fullTitle": full_title,
                            "file": "test/server/b2bOrderSpec.ts",
                        }
                    ],
                    "passes": [
                        {
                            "title": "deserializing arbitrary JSON should not solve rceChallenge",
                            "fullTitle": full_title,
                            "file": "test/server/b2bOrderSpec.ts",
                        }
                    ],
                    "failures": [],
                    "pending": [],
                }
            ),
            stderr="",
            duration_seconds=0.1,
        )
        tool = _make_run_targeted_test_tool(
            sandbox,
            preferred_test_files=["test/server/b2bOrderSpec.ts"],
        )

        result = tool.invoke(
            {
                "test_file": "test/server/b2bOrderSpec.ts",
                "test_name": "b2bOrder",
            }
        )

        assert result.startswith("SUCCESS: Targeted test passed (mocha)")
        assert "Tests executed: 1" in result

    def test_validation_does_not_count_unmatched_mocha_hint_as_gate_attempt(self):
        sandbox = MagicMock()

        def read_file(path):
            if path == "package.json":
                return json.dumps({"scripts": {"test": "mocha test/**/*.ts"}})
            if path == "tsconfig.json":
                return "{}"
            if path == "src/auth.ts":
                return "export const auth = true;"
            if path == "test/server/b2bOrderSpec.ts":
                return "describe('b2bOrder', () => {});"
            return None

        def run(command, timeout=None):
            if "mocha" in command and "--reporter json" in command:
                return CommandResult(
                    exit_code=0,
                    stdout=json.dumps(
                        {
                            "stats": {"tests": 0, "passes": 0, "failures": 0, "pending": 0},
                            "tests": [],
                            "passes": [],
                            "failures": [],
                            "pending": [],
                        }
                    ),
                    stderr="",
                    duration_seconds=0.1,
                )
            return CommandResult(exit_code=0, stdout="", stderr="", duration_seconds=0.1)

        sandbox.read_file.side_effect = read_file
        sandbox.run.side_effect = run
        plan_state = {"targeted_test_required": True}
        tools = _workaround_tool_map(sandbox, plan_state=plan_state)

        result = tools["validate_workaround"].invoke(
            {
                "modified_files": ["src/auth.ts"],
                "runtime_smoke_file": "src/auth.ts",
                "targeted_test_file": "test/server/b2bOrderSpec.ts",
                "targeted_test_name": "does-not-exist",
            }
        )

        assert str(result).startswith("ERROR: [INVALID_VALIDATION_INPUT]")
        assert plan_state["validation_calls"] == 0
        assert plan_state["validation_input_errors"] == 1

    def test_targeted_mocha_reports_selected_test_failure(self):
        sandbox = MagicMock()
        sandbox.read_file.side_effect = lambda path: (
            json.dumps({"scripts": {"test": "mocha test/**/*.ts"}})
            if path == "package.json"
            else "describe('security', () => {});"
        )
        full_title = "insecurity sanitizeSecure cannot be bypassed by exploiting lack of recursive sanitization"
        failure = {
            "title": "cannot be bypassed by exploiting lack of recursive sanitization",
            "fullTitle": full_title,
            "file": "test/server/insecuritySpec.ts",
            "err": {
                "message": "expected 'unsafe' to equal 'safe'",
                "stack": "AssertionError: expected 'unsafe' to equal 'safe'\n    at Context.<anonymous> (test/server/insecuritySpec.ts:184:112)",
            },
        }
        sandbox.run.return_value = CommandResult(
            exit_code=1,
            stdout=json.dumps(
                {
                    "stats": {"tests": 1, "passes": 0, "failures": 1, "pending": 0},
                    "tests": [failure],
                    "passes": [],
                    "failures": [failure],
                    "pending": [],
                }
            ),
            stderr="",
            duration_seconds=0.1,
        )
        tool = _make_run_targeted_test_tool(
            sandbox,
            preferred_test_files=["test/server/insecuritySpec.ts"],
        )

        result = tool.invoke(
            {
                "test_file": "test/server/insecuritySpec.ts",
                "test_name": "sanitizeSecure - cannot be bypassed by exploiting lack of recursive sanitization",
            }
        )

        assert result.startswith("FAILURE: Targeted test failed")
        assert "test/server/insecuritySpec.ts:184:112" in result

    def test_validate_workaround_runtime_smoke_catches_import_errors(self):
        sandbox = MagicMock()
        sandbox.read_file.side_effect = lambda path: (
            "const value = 1;" if path == "src/auth.ts" else None
        )
        sandbox.run.side_effect = [
            CommandResult(exit_code=0, stdout="", stderr="", duration_seconds=0.1),
            CommandResult(exit_code=1, stdout="", stderr="", duration_seconds=0.1),
            CommandResult(
                exit_code=1,
                stdout="",
                stderr="ReferenceError: jwksRsa is not defined",
                duration_seconds=0.1,
            ),
        ]
        tools = _workaround_tool_map(
            sandbox,
            plan_state={"targeted_test_required": False},
        )

        result = tools["validate_workaround"].invoke(
            {"modified_files": ["src/auth.ts"], "runtime_smoke_file": "src/auth.ts"}
        )

        assert result.startswith("FAILURE: Workaround validation gate 'runtime_smoke'")
        assert "jwksRsa is not defined" in result
        assert sandbox.run.call_count == 3

    def test_ast_tool_accepts_complete_arrow_declaration_replacement(self):
        """Arrow symbols may be replaced with their complete exported declaration."""

        class FakeNode:
            def __init__(self, node_type, start_byte, end_byte, text, children=None):
                self.type = node_type
                self.start_byte = start_byte
                self.end_byte = end_byte
                self.text = text
                self.children = children or []

        source = "export const isAuthorized = () => expressjwt({});\n"
        arrow_start = source.index("()")
        arrow = FakeNode("arrow_function", arrow_start, len(source) - 1, "() => expressjwt({});")
        export = FakeNode("export_statement", 0, len(source) - 1, source, [arrow])
        root = FakeNode("program", 0, len(source), source, [export])

        sandbox = MagicMock()
        sandbox.read_file.return_value = source
        sandbox.run.return_value.exit_code = 0
        touched = set()
        plan_state = {
            "recorded": True,
            "planned_files": ["lib/insecurity.ts"],
            "inspected_files": set(),
            "fallback_files": set(),
        }
        tool = _make_deterministic_replace_ast_symbol_tool(sandbox, touched, plan_state)

        with (
            patch("remediation_engine.tools.code_map.language_for_path", return_value="typescript"),
            patch(
                "remediation_engine.tools.code_map.parse_source",
                return_value=MagicMock(root_node=root),
            ),
            patch(
                "remediation_engine.tools.code_map.find_named_symbol",
                return_value={
                    "symbol_name": "isAuthorized",
                    "node_type": "arrow_function",
                    "start_line": 1,
                    "end_line": 1,
                    "start_byte": arrow_start,
                    "end_byte": len(source) - 1,
                    "text": arrow.text,
                },
            ),
        ):
            result = tool.invoke(
                {
                    "file_path": "lib/insecurity.ts",
                    "symbol_name": "isAuthorized",
                    "replacement": "const isAuthorized = () => expressjwt({ algorithms: ['RS256'] });",
                }
            )

        assert result.startswith("SUCCESS:")
        assert "lib/insecurity.ts" in touched
        written = sandbox.write_file.call_args.args[1]
        assert "export const isAuthorized = const" not in written
        assert "const isAuthorized = () => expressjwt" in written


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
            "dependencies": {"lodash": "4.17.20", "axios": "0.21.1"},
            "devDependencies": {"jest": "26.6.3"},
        }
        import json

        baseline_file.write_text(json.dumps(baseline_json, indent=2), encoding="utf-8")

        sandbox_json = {
            "dependencies": {"lodash": "4.17.21", "axios": "0.22.0", "newpkg": "1.0.0"},
            "devDependencies": {"jest": "27.0.0"},
        }
        sandbox.read_file.return_value = json.dumps(sandbox_json, indent=2)

        tools = _update_tool_map(sandbox, touched_files=touched_files, host_repo_root=host_repo)

        result = tools["revert_workspace_file"].invoke(
            {"file_path": "package.json", "package_name": "axios"}
        )

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
        baseline_json = {"dependencies": {"lodash": "4.17.20"}}
        import json

        baseline_file.write_text(json.dumps(baseline_json, indent=2), encoding="utf-8")

        sandbox_json = {"dependencies": {"lodash": "4.17.20", "newpkg": "1.0.0"}}
        sandbox.read_file.return_value = json.dumps(sandbox_json, indent=2)

        tools = _update_tool_map(sandbox, touched_files=touched_files, host_repo_root=host_repo)

        result = tools["revert_workspace_file"].invoke(
            {"file_path": "package.json", "package_name": "newpkg"}
        )

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

        result = tools["revert_workspace_file"].invoke(
            {"file_path": "src/index.js", "package_name": "axios"}
        )

        assert result.startswith("ERROR:")
        assert "only be specified for package.json files" in result


class TestValidateManifestSync:
    def test_validation_runs_once_per_package_in_worker_run(self):
        sandbox = MagicMock()
        sandbox.run.return_value = CommandResult(
            exit_code=0,
            stdout="ok",
            stderr="",
            duration_seconds=1.0,
        )
        execution_state = {"edits_started": False, "validation_calls": 0}
        tools = _update_tool_map(
            sandbox,
            manifest_paths=["package.json", "frontend/package.json"],
            package_manifest_paths={
                "lodash": ["package.json"],
                "axios": ["frontend/package.json"],
            },
            execution_state=execution_state,
        )

        edit_result = tools["modify_npm_dependency"].invoke(
            {
                "package_name": "lodash",
                "target_version": "4.17.22",
                "dependency_type": "dependencies",
                "manifest_path": "package.json",
            }
        )
        first_validation = tools["validate_manifest_sync"].invoke({"package_name": "lodash"})
        second_edit_result = tools["modify_npm_dependency"].invoke(
            {
                "package_name": "axios",
                "target_version": "1.7.4",
                "dependency_type": "dependencies",
                "manifest_path": "frontend/package.json",
            }
        )
        second_validation = tools["validate_manifest_sync"].invoke({"package_name": "axios"})

        assert edit_result.startswith("SUCCESS:")
        assert second_edit_result.startswith("SUCCESS:")
        assert first_validation.startswith("SUCCESS:")
        assert second_validation.startswith("SUCCESS:")
        assert execution_state["validation_calls"] == 2

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

        result = tools["validate_manifest_sync"].invoke({"package_name": "lodash"})

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

        result = tools["validate_manifest_sync"].invoke({"package_name": "lodash"})

        assert result.startswith("FAILURE:")
        assert "ERESOLVE" in result
        assert "partial" in result

    def test_failed_package_validation_restores_its_checkpoint(self):
        sandbox = MagicMock()
        success = CommandResult(exit_code=0, stdout="ok", stderr="", duration_seconds=1.0)
        failure = CommandResult(
            exit_code=1,
            stdout="partial",
            stderr="ERESOLVE unable to resolve dependency tree",
            duration_seconds=1.0,
        )
        sandbox.run.side_effect = [success, failure, success, success]
        baseline = '{"dependencies": {"lodash": "4.17.20"}}\n'
        sandbox.read_file.side_effect = lambda path: baseline if path == "package.json" else None
        touched_files: set[str] = set()
        execution_state = {"edits_started": False, "validation_calls": 0}
        tools = _update_tool_map(
            sandbox,
            touched_files=touched_files,
            execution_state=execution_state,
        )

        edit_result = tools["modify_npm_dependency"].invoke(
            {
                "package_name": "lodash",
                "target_version": "4.17.22",
                "dependency_type": "dependencies",
                "manifest_path": "package.json",
            }
        )
        validation_result = tools["validate_manifest_sync"].invoke({"package_name": "lodash"})

        assert edit_result.startswith("SUCCESS:")
        assert validation_result.startswith("FAILURE:")
        assert "Rolled back package 'lodash'" in validation_result
        assert touched_files == set()
        sandbox.write_file.assert_called_once_with("package.json", baseline)


class TestValidateCodeSyntax:
    def test_js_uses_node_check(self):
        sandbox = MagicMock()
        sandbox.run.return_value = CommandResult(
            exit_code=0,
            stdout="",
            stderr="",
            duration_seconds=0.1,
        )
        result = _make_validate_code_syntax_tool(sandbox).invoke({"file_path": "src/index.js"})

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
        result = _make_validate_code_syntax_tool(sandbox).invoke({"file_path": "src/index.ts"})

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
        result = _make_validate_code_syntax_tool(sandbox).invoke({"file_path": "routes/login.ts"})

        assert result.startswith("FAILURE:")
        assert "TS1005" in result

    def test_unsupported_extension_returns_error(self):
        sandbox = MagicMock()
        result = _make_validate_code_syntax_tool(sandbox).invoke({"file_path": "README.md"})

        assert result.startswith("ERROR:")

    def test_absolute_path_rejected(self):
        sandbox = MagicMock()
        result = _make_validate_code_syntax_tool(sandbox).invoke({"file_path": "/etc/passwd"})

        assert result.startswith("ERROR:")


class TestSearchWeb:
    def test_returns_formatted_results(self, monkeypatch):
        monkeypatch.setenv("SERPER_API_KEY", "test-key")
        body = {
            "organic": [
                {"snippet": "Snippet 1", "title": "Title 1", "link": "https://example.com/1"},
                {"snippet": "Snippet 2", "title": "Title 2", "link": "https://example.com/2"},
                {"snippet": "Snippet 3", "title": "Title 3", "link": "https://example.com/3"},
                {"snippet": "Snippet 4", "title": "Title 4", "link": "https://example.com/4"},
            ]
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = body
        mock_resp.raise_for_status = MagicMock()

        sandbox = MagicMock()
        tools = _workaround_tool_map(sandbox)

        with patch(
            "remediation_engine.orchestration.remedy_tools.requests.post", return_value=mock_resp
        ):
            res = tools["search_web"].invoke({"query": "express-jwt v8 migration"})

        assert "Found 3 results" in res
        assert "Title 1" in res
        assert "Title 2" in res
        assert "Title 3" in res
        assert "Title 4" not in res

    def test_returns_error_when_no_api_key(self, monkeypatch):
        monkeypatch.delenv("SERPER_API_KEY", raising=False)
        sandbox = MagicMock()
        tools = _workaround_tool_map(sandbox)

        res = tools["search_web"].invoke({"query": "express-jwt v8"})
        assert "ERROR: SERPER_API_KEY is not set" in res

    def test_returns_error_on_network_failure(self, monkeypatch):
        monkeypatch.setenv("SERPER_API_KEY", "test-key")
        sandbox = MagicMock()
        tools = _workaround_tool_map(sandbox)

        with patch(
            "remediation_engine.orchestration.remedy_tools.requests.post",
            side_effect=Exception("timeout"),
        ):
            res = tools["search_web"].invoke({"query": "express-jwt v8"})

        assert "ERROR: Web search failed" in res

    def test_respects_call_limit(self, monkeypatch):
        monkeypatch.setenv("SERPER_API_KEY", "test-key")
        body = {"organic": [{"snippet": "S1", "title": "T1", "link": "https://a.com"}]}
        mock_resp = MagicMock()
        mock_resp.json.return_value = body
        mock_resp.raise_for_status = MagicMock()

        sandbox = MagicMock()
        tools = _workaround_tool_map(sandbox)

        with patch(
            "remediation_engine.orchestration.remedy_tools.requests.post", return_value=mock_resp
        ):
            for _ in range(3):
                res = tools["search_web"].invoke({"query": "query"})
                assert "Found 1 results" in res

            res_exceeded = tools["search_web"].invoke({"query": "query 4"})
            assert "ERROR: search_web call limit reached (max 3 per session)" in res_exceeded

    def test_handles_empty_results(self, monkeypatch):
        monkeypatch.setenv("SERPER_API_KEY", "test-key")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"organic": []}
        mock_resp.raise_for_status = MagicMock()

        sandbox = MagicMock()
        tools = _workaround_tool_map(sandbox)

        with patch(
            "remediation_engine.orchestration.remedy_tools.requests.post", return_value=mock_resp
        ):
            res = tools["search_web"].invoke({"query": "query"})

        assert "No results found for this query" in res


class TestReadWebPage:
    def test_returns_markdown_content(self):
        mock_resp = MagicMock()
        mock_resp.text = "# Migration Guide\nUse expressjwt({ ... })"
        mock_resp.raise_for_status = MagicMock()

        sandbox = MagicMock()
        tools = _workaround_tool_map(sandbox)

        with patch(
            "remediation_engine.orchestration.remedy_tools.requests.get", return_value=mock_resp
        ) as mock_get:
            res = tools["read_web_page"].invoke({"url": "https://example.com/guide"})

        mock_get.assert_called_once_with(
            "https://r.jina.ai/https://example.com/guide",
            headers={"Accept": "text/plain"},
            timeout=15,
        )
        assert "# Migration Guide" in res
        assert "Use expressjwt" in res

    def test_truncates_oversized_page(self):
        mock_resp = MagicMock()
        mock_resp.text = "A" * 20_000
        mock_resp.raise_for_status = MagicMock()

        sandbox = MagicMock()
        tools = _workaround_tool_map(sandbox)

        with patch(
            "remediation_engine.orchestration.remedy_tools.requests.get", return_value=mock_resp
        ):
            res = tools["read_web_page"].invoke({"url": "https://example.com/huge"})

        assert "[Content truncated at 16000 characters...]" in res

    def test_handles_network_failure(self):
        sandbox = MagicMock()
        tools = _workaround_tool_map(sandbox)

        with patch(
            "remediation_engine.orchestration.remedy_tools.requests.get",
            side_effect=Exception("Connection reset"),
        ):
            res = tools["read_web_page"].invoke({"url": "https://example.com/error"})

        assert "ERROR: Failed to read web page" in res

    def test_github_urls_use_github_api_instead_of_jina(self):
        mock_resp = MagicMock()
        mock_resp.text = "# Migration Guide\nUse expressjwt from the named export."
        mock_resp.raise_for_status = MagicMock()
        sandbox = MagicMock()
        tools = _workaround_tool_map(sandbox)

        with patch(
            "remediation_engine.orchestration.remedy_tools.requests.get",
            return_value=mock_resp,
        ) as mock_get:
            res = tools["read_web_page"].invoke(
                {"url": "https://github.com/auth0/express-jwt/blob/v8.5.1/README.md"}
            )

        mock_get.assert_called_once_with(
            "https://api.github.com/repos/auth0/express-jwt/contents/README.md?ref=v8.5.1",
            headers={
                "Accept": "application/vnd.github.raw+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            timeout=15,
        )
        assert "Use expressjwt from the named export" in res
