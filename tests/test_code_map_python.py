"""Tests for Python AST localization through the shared tree-sitter helpers."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from remediation_engine.contracts.schemas import ASTNodeType, CommandResult
from remediation_engine.orchestration.tools_edit import (
    _make_deterministic_replace_ast_symbol_tool,
)
from remediation_engine.tools.code_map import (
    _PYTHON_TREE_SITTER_AVAILABLE,
    extract_imports,
    extract_sink_expression,
    find_enclosing_symbol,
    find_named_symbol,
    language_for_path,
    parse_source,
)

pytestmark = pytest.mark.skipif(
    not _PYTHON_TREE_SITTER_AVAILABLE,
    reason="tree-sitter Python grammar is unavailable",
)


def _parse_python(source: str):
    language = language_for_path("module.py")
    assert language is not None
    tree = parse_source(source.encode(), language)
    assert tree is not None
    return tree.root_node, source.encode()


def test_python_function_and_class_symbols_are_searchable():
    source = "class Handler:\n    def handle(self):\n        return 1\n"
    root, source_bytes = _parse_python(source)

    class_symbol = find_named_symbol(root, "Handler", source_bytes)
    function_symbol = find_named_symbol(root, "handle", source_bytes)

    assert class_symbol is not None
    assert class_symbol["node_type"] == "class_definition"
    assert class_symbol["text"].startswith("class Handler:")
    assert "return 1" in class_symbol["text"]
    assert function_symbol is not None
    assert function_symbol["node_type"] == "function_definition"
    assert function_symbol["text"].startswith("def handle(self):")
    assert find_enclosing_symbol(root, 3) == ("handle", ASTNodeType.METHOD)


def test_decorated_definition_uses_one_outer_replacement_span():
    source = "@authorize\n@cache\ndef load_user(user_id):\n    return user_id\n"
    root, source_bytes = _parse_python(source)

    symbol = find_named_symbol(root, "load_user", source_bytes)

    assert symbol is not None
    assert symbol["node_type"] == "decorated_definition"
    assert symbol["start_line"] == 1
    assert symbol["end_line"] == 4
    assert symbol["start_byte"] == 0
    assert source_bytes[symbol["start_byte"] : symbol["end_byte"]].decode() == symbol["text"]
    assert symbol["text"].startswith("@authorize\n@cache\n")
    assert symbol["text"].count("@authorize") == 1
    assert symbol["text"].count("@cache") == 1
    assert symbol["text"].rstrip().endswith("return user_id")
    assert find_enclosing_symbol(root, 4) == ("load_user", ASTNodeType.FUNCTION)


def test_decorated_class_lookup_includes_its_decorator():
    source = "@register\nclass User:\n    role = 'member'\n"
    root, source_bytes = _parse_python(source)

    symbol = find_named_symbol(root, "User", source_bytes)

    assert symbol is not None
    assert symbol["node_type"] == "decorated_definition"
    assert symbol["start_line"] == 1
    assert symbol["text"].startswith("@register\nclass User:")
    assert source_bytes[symbol["start_byte"] : symbol["end_byte"]].decode() == symbol["text"]
    assert find_enclosing_symbol(root, 3) == ("User", ASTNodeType.CLASS)


def test_python_imports_and_call_expression_are_extracted():
    source = (
        "import os\nfrom pathlib import Path\ndef read_file():\n    return open('sample.txt')\n"
    )
    root, source_bytes = _parse_python(source)

    imports = extract_imports(root, source_bytes)
    call = extract_sink_expression(root, 4, source_bytes)

    assert imports == ["import os", "from pathlib import Path"]
    assert call == "open('sample.txt')"
    assert find_enclosing_symbol(root, 4) == ("read_file", ASTNodeType.FUNCTION)


def test_decorated_python_ast_replacement_replaces_outer_definition_once():
    source = "@cache\ndef load_user(user_id):\n    return user_id\n"
    replacement = "@audit\ndef load_user(user_id):\n    return None"
    sandbox = MagicMock()
    sandbox.read_file.return_value = source
    sandbox.run.return_value = CommandResult(
        exit_code=0,
        stdout="",
        stderr="",
        duration_seconds=0.1,
    )
    plan_state = {
        "recorded": True,
        "phase": "EXECUTE",
        "planned_files": ["src/module.py"],
        "inspected_files": {"src/module.py"},
        "fallback_files": set(),
    }
    tool = _make_deterministic_replace_ast_symbol_tool(
        sandbox,
        set(),
        plan_state,
    )

    result = tool.invoke(
        {
            "file_path": "src/module.py",
            "symbol_name": "load_user",
            "replacement": replacement,
        }
    )

    assert result.startswith("SUCCESS:")
    updated = sandbox.write_file.call_args.args[1]
    assert updated == replacement + "\n"
    assert updated.count("@cache") == 0
    assert updated.count("@audit") == 1
