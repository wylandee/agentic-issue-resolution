"""
tests/test_remedy_tools.py - Direct unit tests for Phase 5 Remedy Agent tools.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from src.orchestrator.remedy_tools import build_agent_tools


def _tool_map(sandbox, touched_files=None):
    if touched_files is None:
        touched_files = set()
    tools = build_agent_tools(sandbox, touched_files)
    return {tool.name: tool for tool in tools}


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
