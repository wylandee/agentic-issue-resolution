"""Focused tests for QA execution and review toolbelts."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from remediation_engine.contracts.schemas import CommandResult
from remediation_engine.orchestration._qa_runtime import _generate_workspace_diff
from remediation_engine.orchestration.qa_evaluator import (
    _LOG_QUERY_MAX_CHARS,
    _query_qa_logs,
    build_qa_review_toolbelt,
)
from remediation_engine.orchestration.qa_types import (
    _QAExecutionResults,
    _QALogRecord,
    _SecurityScanResult,
    _validate_qa_path,
)


class TestGenerateWorkspaceDiff:
    def test_detects_modified_file(self):
        tmp_path = Path.cwd()
        rel_path = "src/remediation_engine/orchestration/qa_critic.py"
        sandbox = MagicMock()
        sandbox.read_file.side_effect = lambda path: "const x = 2;\n" if path == rel_path else None
        diff_text, changed = _generate_workspace_diff(str(tmp_path), sandbox, [rel_path])
        assert rel_path in changed
        assert rel_path in diff_text

    def test_detects_deleted_file(self):
        tmp_path = Path.cwd()
        rel_path = "src/remediation_engine/orchestration/qa_critic.py"

        sandbox = MagicMock()
        # read_file returns None â†’ file deleted in workspace
        sandbox.read_file.return_value = None
        diff_text, changed = _generate_workspace_diff(str(tmp_path), sandbox, [rel_path])

        assert rel_path in changed
        assert "deleted" in diff_text.lower()

    def test_ignores_node_modules(self):
        tmp_path = Path.cwd()

        sandbox = MagicMock()
        sandbox.read_file.return_value = None
        _, changed = _generate_workspace_diff(str(tmp_path), sandbox, ["node_modules/evil.js"])

        assert not any("node_modules" in f for f in changed)

    def test_diff_text_is_capped(self, tmp_path):
        from remediation_engine.orchestration._qa_runtime import _DIFF_CHAR_BUDGET

        rel_path = "src/remediation_engine/orchestration/qa_critic.py"

        big_content = "x" * (_DIFF_CHAR_BUDGET * 3)
        sandbox = MagicMock()
        sandbox.read_file.side_effect = lambda path: (
            big_content + "\nextra line\n" if path == rel_path else None
        )
        diff_text, _ = _generate_workspace_diff(str(tmp_path), sandbox, [rel_path])
        assert len(diff_text) <= _DIFF_CHAR_BUDGET + len("\n... (diff truncated)")

    def test_optimized_changed_files_path(self):
        tmp_path = Path.cwd()
        rel_path = "src/remediation_engine/orchestration/qa_critic.py"
        sandbox = MagicMock()
        sandbox.read_file.side_effect = lambda path: (
            "const x = 2;\n" if path == rel_path else "const y = 2;\n"
        )

        diff_text, changed = _generate_workspace_diff(
            str(tmp_path), sandbox, candidate_changed_files=[rel_path]
        )

        assert rel_path in changed
        assert "other.js" not in changed
        assert rel_path in diff_text
        assert "other.js" not in diff_text

    def test_empty_candidates_returns_empty_diff_note(self, tmp_path):
        """When no candidate files are given, diff returns an empty-diff note."""
        sandbox = MagicMock()
        diff_text, changed = _generate_workspace_diff(str(tmp_path), sandbox, [])
        assert changed == []
        assert "empty" in diff_text.lower() or "no changed files" in diff_text.lower()
        sandbox.read_file.assert_not_called()

    def test_blocked_candidate_path_is_reported_explicitly(self, tmp_path):
        """Traversal candidates are errors, not unchanged-file observations."""
        sandbox = MagicMock()

        diff_text, changed = _generate_workspace_diff(str(tmp_path), sandbox, ["../outside.js"])

        assert changed == []
        assert "ERROR: blocked candidate path" in diff_text
        sandbox.read_file.assert_not_called()


class TestQAReviewToolSafety:
    """Verify read_file_context rejects dangerous paths."""

    def _get_read_file_tool(self, sandbox=None):
        if sandbox is None:
            sandbox = MagicMock()
        results = _QAExecutionResults()
        tools = build_qa_review_toolbelt(
            sandbox=sandbox,
            candidate_changed_files=[],
            host_repo_root=None,
            results=results,
        )
        results.install = (True, "install ok")
        results.scan = (True, "scan ok", set())
        results.tests = (True, "tests ok")
        for t in tools:
            if t.name == "read_file_context":
                return t
        raise KeyError("read_file_context not found")

    def test_rejects_absolute_path(self):
        tool = self._get_read_file_tool()
        result = tool.invoke({"file_path": "/etc/passwd"})
        assert "ERROR" in result
        assert "absolute" in result.lower()

    def test_rejects_path_traversal(self):
        tool = self._get_read_file_tool()
        result = tool.invoke({"file_path": "../../etc/passwd"})
        assert "ERROR" in result

    def test_returns_file_content_for_valid_path(self):
        sandbox = MagicMock()
        sandbox.read_file.return_value = "const x = 1;\n"
        tool = self._get_read_file_tool(sandbox)

        result = tool.invoke({"file_path": "src/app.js"})
        assert "const x = 1;" in result
        sandbox.read_file.assert_called_once_with("src/app.js")

    def test_returns_error_when_file_not_found(self):
        sandbox = MagicMock()
        sandbox.read_file.return_value = None
        tool = self._get_read_file_tool(sandbox)

        result = tool.invoke({"file_path": "nonexistent.js"})
        assert "ERROR" in result
        assert "not found" in result.lower()


class TestValidateQaPath:
    """Unit tests for the _validate_qa_path helper."""

    def test_accepts_relative_path(self):
        assert _validate_qa_path("src/app.js") == "src/app.js"

    def test_accepts_nested_relative_path(self):
        assert _validate_qa_path("a/b/c.ts") == "a/b/c.ts"

    def test_rejects_empty_string(self):
        with pytest.raises(ValueError, match="required"):
            _validate_qa_path("")

    def test_rejects_absolute_path(self):
        with pytest.raises(ValueError, match="absolute"):
            _validate_qa_path("/etc/passwd")

    def test_rejects_path_traversal(self):
        with pytest.raises(ValueError, match="traversal"):
            _validate_qa_path("../../secret")

    def test_normalizes_backslashes(self):
        assert _validate_qa_path("src\\app.js") == "src/app.js"


class TestQADiffToolNoCandidates:
    """Verify generate_workspace_diff tool returns an informative note when no candidates exist."""

    def test_empty_candidates_returns_informative_note(self):
        sandbox = MagicMock()
        results = _QAExecutionResults()
        tools = build_qa_review_toolbelt(
            sandbox=sandbox,
            candidate_changed_files=[],
            host_repo_root="/tmp/repo",
            results=results,
        )
        results.install = (True, "install ok")
        results.scan = (True, "scan ok", set())
        results.tests = (True, "tests ok")
        diff_tool = next(t for t in tools if t.name == "generate_workspace_diff")
        result = diff_tool.invoke({})
        # Should mention no changed files, not crash
        assert "empty" in result.lower() or "no changed files" in result.lower()
        sandbox.read_file.assert_not_called()

    def test_no_host_repo_root_returns_error(self):
        sandbox = MagicMock()
        results = _QAExecutionResults()
        tools = build_qa_review_toolbelt(
            sandbox=sandbox,
            candidate_changed_files=["src/app.js"],
            host_repo_root=None,
            results=results,
        )
        results.install = (True, "install ok")
        results.scan = (True, "scan ok", set())
        results.tests = (True, "tests ok")
        diff_tool = next(t for t in tools if t.name == "generate_workspace_diff")
        result = diff_tool.invoke({})
        assert "ERROR" in result


class TestQueryQaLogs:
    """Verify query_qa_logs returns cached data and handles missing runs correctly."""

    def _get_query_tool(self, prepopulate: bool = False):
        sandbox = MagicMock()
        results = _QAExecutionResults()
        tools = build_qa_review_toolbelt(
            sandbox=sandbox,
            candidate_changed_files=[],
            host_repo_root=None,
            results=results,
        )
        if prepopulate:
            results.install = (True, "install ready")
            results.scan = _SecurityScanResult(True, "scan ready", set(), set(), set())
            results.tests = (True, "tests ready")
        tool = next(t for t in tools if t.name == "query_qa_logs")
        return tool, results

    def test_returns_error_when_install_not_run(self):
        tool, _ = self._get_query_tool()
        result = tool.invoke({"log_type": "install"})
        assert "ERROR" in result

    def test_returns_cached_install_log(self):
        tool, results = self._get_query_tool(prepopulate=True)
        results.install = (True, "npm install output here")
        result = tool.invoke({"log_type": "install"})
        assert "npm install output here" in result

    def test_returns_cached_scan_log(self):
        tool, results = self._get_query_tool(prepopulate=True)
        results.scan = _SecurityScanResult(True, "scan output here", set(), set(), set())
        result = tool.invoke({"log_type": "scan"})
        assert "scan output here" in result

    def test_returns_cached_tests_log(self):
        tool, results = self._get_query_tool(prepopulate=True)
        results.tests = (True, "test output here")
        result = tool.invoke({"log_type": "tests"})
        assert "test output here" in result

    def test_invalid_log_type_returns_error(self):
        tool, _ = self._get_query_tool(prepopulate=True)
        result = tool.invoke({"log_type": "unknown"})
        assert "ERROR" in result


class TestQAToolsNotInSubagentToolbelts:
    """
    Verify that the heavy QA commands are not present in the update or
    workaround toolbelts that are given to subagents.

    This duplicates the guards in test_remedy_tools.py but is kept here for
    documentation purposes and test isolation.
    """

    def test_run_dependency_install_not_in_update_toolbelt(self):
        from remediation_engine.orchestration.remedy_tools import build_update_toolbelt

        sandbox = MagicMock()
        tools = build_update_toolbelt(
            sandbox,
            touched_files=set(),
            target_manifest_paths=["package.json"],
            package_manifest_paths={"lodash": ["package.json"]},
        )
        tool_names = {t.name for t in tools}
        assert "run_dependency_install" not in tool_names
        assert "run_security_scan" not in tool_names
        assert "run_unit_tests" not in tool_names

    def test_run_dependency_install_not_in_workaround_toolbelt(self):
        from remediation_engine.orchestration.remedy_tools import build_workaround_toolbelt

        sandbox = MagicMock()
        tools = build_workaround_toolbelt(
            sandbox,
            touched_files=set(),
            host_repo_root=Path("/repo"),
        )
        tool_names = {t.name for t in tools}
        assert "run_dependency_install" not in tool_names
        assert "run_security_scan" not in tool_names
        assert "run_unit_tests" not in tool_names


class TestBuildQaReviewToolbelt:
    def _build(self, prepopulate=True):
        sandbox = MagicMock()
        sandbox.read_file.return_value = "file content"
        results = _QAExecutionResults()
        if prepopulate:
            results.install = (True, "ok")
            results.scan = _SecurityScanResult(True, "ok", set(), set(), set())
            results.tests = (True, "ok")
        return build_qa_review_toolbelt(
            sandbox=sandbox,
            candidate_changed_files=["src/app.ts"],
            host_repo_root="/tmp/repo",
            results=results,
        ), results

    def test_no_execution_tools_present(self):
        tools, _ = self._build()
        names = {t.name for t in tools}
        assert "run_dependency_install" not in names
        assert "run_security_scan" not in names
        assert "run_unit_tests" not in names

    def test_all_six_review_tools_present(self):
        tools, _ = self._build()
        names = {t.name for t in tools}
        for n in [
            "list_changed_files",
            "generate_workspace_diff",
            "read_file_context",
            "search_codebase_pattern",
            "inspect_ast_symbol",
            "query_qa_logs",
        ]:
            assert n in names

    def test_review_tools_locked_when_results_empty(self):
        tools, _ = self._build(prepopulate=False)
        tool = next(t for t in tools if t.name == "list_changed_files")
        result = tool.invoke({})
        assert "ERROR" in result

    def test_list_changed_files_works_when_populated(self):
        tools, _ = self._build(prepopulate=True)
        tool = next(t for t in tools if t.name == "list_changed_files")
        assert "src/app.ts" in tool.invoke({})

    def test_query_qa_logs_returns_install_log(self):
        tools, results = self._build(prepopulate=True)
        results.install = (True, "npm install output here")
        tool = next(t for t in tools if t.name == "query_qa_logs")
        assert "npm install output here" in tool.invoke({"log_type": "install"})

    def test_query_qa_logs_returns_condensed_test_summary(self):
        tools, results = self._build(prepopulate=True)
        results.tests = (
            False,
            "npm test FAILED (exit 1).\nDetected Failures: 1\n\nFailing Tests:\n1. jwt challenge\nAssertionError: boom",
        )
        tool = next(t for t in tools if t.name == "query_qa_logs")
        result = tool.invoke({"log_type": "tests"})
        assert "Detected Failures: 1" in result
        assert "jwt challenge" in result

    def test_query_qa_logs_supports_tail_errors_and_filter_views(self):
        tools, results = self._build(prepopulate=True)
        record = _QALogRecord(
            phase="install",
            label="npm install",
            exit_code=1,
            stdout="first stdout line\nsecond stdout line",
            stderr="ERESOLVE peer conflict\nsecondary diagnostic",
            error="install failed",
        )
        results.install = (False, "install failed")
        results.log_records["install"] = (record,)
        tool = next(t for t in tools if t.name == "query_qa_logs")

        tail = tool.invoke({"log_type": "install", "view": "tail", "tail_lines": 1})
        errors = tool.invoke({"log_type": "install", "view": "errors"})
        filtered = tool.invoke(
            {
                "log_type": "install",
                "view": "filter",
                "filter_pattern": "secondary diagnostic",
            }
        )
        full = tool.invoke({"log_type": "install", "view": "full"})

        assert "[npm install stdout]" in full
        assert "exit_code=1" in full
        assert "ERESOLVE peer conflict" in full
        assert "secondary diagnostic" in tail
        assert "first stdout line" not in tail
        assert "ERESOLVE peer conflict" in errors
        assert "install failed" in errors
        assert "secondary diagnostic" in filtered
        assert "first stdout line" not in filtered

    def test_query_qa_logs_caps_full_output_and_validates_view(self):
        results = _QAExecutionResults(
            install=(True, "install ok"),
            log_records={
                "install": (
                    _QALogRecord(
                        phase="install",
                        label="npm install",
                        exit_code=0,
                        stdout="x" * (_LOG_QUERY_MAX_CHARS + 100),
                        stderr="",
                    ),
                )
            },
        )

        capped = _query_qa_logs(results, "install", view="full")
        invalid = _query_qa_logs(results, "install", view="unknown")
        missing = _query_qa_logs(_QAExecutionResults(), "install", view="tail")

        assert len(capped) <= _LOG_QUERY_MAX_CHARS
        assert capped.endswith("(log output truncated)")
        assert invalid.startswith("ERROR: view")
        assert missing.startswith("ERROR: run_dependency_install")

    def test_query_qa_logs_rejects_invalid_filter_regex(self):
        tools, _ = self._build(prepopulate=True)
        tool = next(t for t in tools if t.name == "query_qa_logs")

        result = tool.invoke(
            {
                "log_type": "install",
                "view": "filter",
                "filter_pattern": "[unterminated",
            }
        )

        assert result.startswith("ERROR:")

    def test_search_codebase_pattern_forwards_shared_argument_contract(self):
        sandbox = MagicMock()
        sandbox.run.return_value = CommandResult(
            exit_code=0,
            stdout="src/app.ts:1:foo\n",
            stderr="",
            duration_seconds=0.1,
        )
        results = _QAExecutionResults(
            install=(True, "install ok"),
            scan=(True, "scan ok", set()),
            tests=(True, "tests ok"),
        )
        tools = build_qa_review_toolbelt(
            sandbox=sandbox,
            candidate_changed_files=["src/app.ts"],
            host_repo_root="/tmp/repo",
            results=results,
        )
        search_tool = next(t for t in tools if t.name == "search_codebase_pattern")

        assert (
            search_tool.invoke({"search_pattern": "foo", "target_directory": "src"})
            == "src/app.ts:1:foo"
        )
        command = sandbox.run.call_args.args[0]
        assert "'foo'" in command
        assert "'src'" in command
        assert set(search_tool.args) == {"search_pattern", "target_directory"}
