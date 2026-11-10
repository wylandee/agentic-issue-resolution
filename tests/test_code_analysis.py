"""Focused tests for code analysis."""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from remediation_engine.tools.source import (
    _PYTHON_TREE_SITTER_AVAILABLE,
    _TREE_SITTER_AVAILABLE,
    build_repository_map,
    extract_all_imports,
    extract_imports,
    find_named_symbol,
    language_for_path,
    load_source_bytes,
    parse_source,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# TestCodeMapHelpers
# ---------------------------------------------------------------------------


class TestCodeMapHelpers:
    """Unit tests for the pure helper functions in code_map.py."""

    # --- resolve_repo_file ---

    # --- load_source_bytes / load_source_lines ---

    def test_load_source_bytes_reads_file(self, tmp_path: Path):
        f = tmp_path / "x.js"
        f.write_bytes(b"hello world")
        assert load_source_bytes(f) == b"hello world"

    def test_load_source_bytes_missing_returns_none(self, tmp_path: Path):
        assert load_source_bytes(tmp_path / "ghost.js") is None

    # --- language_for_path ---

    @pytest.mark.parametrize(
        "path, expected_none",
        [
            ("app.js", False),
            ("app.jsx", False),
            ("app.ts", False),
            ("app.tsx", False),
            ("app.mjs", False),
            ("app.py", False),
            ("app.java", True),
            ("README.md", True),
        ],
    )
    def test_language_for_path(self, path: str, expected_none: bool):
        lang = language_for_path(path)
        if path.endswith(".py"):
            assert (lang is not None) == _PYTHON_TREE_SITTER_AVAILABLE
        elif expected_none:
            assert lang is None
        elif _TREE_SITTER_AVAILABLE:
            assert lang is not None

    # --- extract_snippet ---


# ---------------------------------------------------------------------------
# TestASTHelpers (only run when tree-sitter is available)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _TREE_SITTER_AVAILABLE, reason="tree-sitter not installed")
class TestASTHelpers:
    """Tests for AST-based helpers (tree-sitter required)."""

    def _parse_js(self, code: str):
        import tree_sitter_javascript as tsjs
        from tree_sitter import Language, Parser

        lang = Language(tsjs.language())
        parser = Parser(lang)
        tree = parser.parse(code.encode())
        return tree.root_node, lang

    # --- parse_source ---

    def test_parse_source_returns_tree(self):
        lang = language_for_path("app.js")
        tree = parse_source(b"function f() {}", lang)
        assert tree is not None
        assert tree.root_node.type == "program"

    def test_parse_source_invalid_language_returns_none(self):
        result = parse_source(b"function f() {}", None)
        assert result is None

    # --- find_enclosing_symbol ---

    # --- extract_imports ---

    def test_extract_imports_esm(self):
        code = textwrap.dedent("""\
            import express from 'express';
            import { Router } from 'express';
            const morgan = require('morgan');
            function app() {}
        """)
        lang = language_for_path("app.js")
        tree = parse_source(code.encode(), lang)
        root = tree.root_node
        imports = extract_imports(root, code.encode())
        assert any("express" in i for i in imports)
        assert any("morgan" in i for i in imports)
        # function declaration should NOT appear
        assert not any("function" in i for i in imports)

    def test_extract_imports_empty_file(self):
        lang = language_for_path("app.js")
        tree = parse_source(b"", lang)
        root = tree.root_node
        imports = extract_imports(root, b"")
        assert imports == []

    def test_extract_imports_capped_at_20(self):
        lines = "\n".join(f"import x{i} from 'mod{i}';" for i in range(30))
        lang = language_for_path("app.js")
        tree = parse_source(lines.encode(), lang)
        root = tree.root_node
        imports = extract_imports(root, lines.encode())
        assert len(imports) <= 20

    def test_extract_all_imports_is_uncapped_for_analysis(self):
        lines = "\n".join(f"import x{i} from 'mod{i}';" for i in range(30))
        lang = language_for_path("app.js")
        tree = parse_source(lines.encode(), lang)

        assert len(extract_all_imports(tree.root_node, lines.encode())) == 30
        assert len(extract_imports(tree.root_node, lines.encode())) == 20

    # --- extract_sink_expression ---


# ---------------------------------------------------------------------------
# TestDataFlowHints
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# TestScoreConfidence
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# TestLocateSast — happy path (real files + tree-sitter)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _TREE_SITTER_AVAILABLE, reason="tree-sitter not installed")

# ---------------------------------------------------------------------------
# TestLocateSastFallback — graceful degradation
# ---------------------------------------------------------------------------


def _parse_python(source: str):
    language = language_for_path("module.py")
    assert language is not None
    tree = parse_source(source.encode(), language)
    assert tree is not None
    return tree.root_node, source.encode()


@pytest.mark.skipif(
    not _PYTHON_TREE_SITTER_AVAILABLE,
    reason="tree-sitter Python grammar is unavailable",
)
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


@pytest.mark.skipif(
    not _PYTHON_TREE_SITTER_AVAILABLE,
    reason="tree-sitter Python grammar is unavailable",
)
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


@pytest.mark.skipif(
    not _PYTHON_TREE_SITTER_AVAILABLE,
    reason="tree-sitter Python grammar is unavailable",
)
def test_decorated_class_lookup_includes_its_decorator():
    source = "@register\nclass User:\n    role = 'member'\n"
    root, source_bytes = _parse_python(source)

    symbol = find_named_symbol(root, "User", source_bytes)

    assert symbol is not None
    assert symbol["node_type"] == "decorated_definition"
    assert symbol["start_line"] == 1
    assert symbol["text"].startswith("@register\nclass User:")
    assert source_bytes[symbol["start_byte"] : symbol["end_byte"]].decode() == symbol["text"]


@pytest.mark.skipif(
    not _PYTHON_TREE_SITTER_AVAILABLE,
    reason="tree-sitter Python grammar is unavailable",
)
def test_python_imports_are_extracted():
    source = (
        "import os\nfrom pathlib import Path\ndef read_file():\n    return open('sample.txt')\n"
    )
    root, source_bytes = _parse_python(source)

    imports = extract_imports(root, source_bytes)

    assert imports == ["import os", "from pathlib import Path"]


@pytest.mark.skipif(
    not _PYTHON_TREE_SITTER_AVAILABLE,
    reason="tree-sitter Python grammar is unavailable",
)
def test_repository_map_is_sorted_and_excludes_generated_or_dependency_paths(tmp_path):
    (tmp_path / "z.txt").write_text("z", encoding="utf-8")
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    (source_dir / "b.py").write_text("b", encoding="utf-8")
    (source_dir / "a.py").write_text("a", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("ignored", encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "ignored.js").write_text("ignored", encoding="utf-8")
    (tmp_path / "bundle.js.map").write_text("ignored", encoding="utf-8")
    (tmp_path / "generated.map").mkdir()

    assert build_repository_map(tmp_path) == "\n".join(
        ["a.txt", "src", "src/a.py", "src/b.py", "z.txt"]
    )


def test_repository_map_preserves_empty_and_truncation_behavior(tmp_path):
    assert build_repository_map(tmp_path) == "(workspace is empty)"

    for name in ("a", "b", "c", "d"):
        (tmp_path / f"{name}.txt").write_text(name, encoding="utf-8")

    assert build_repository_map(tmp_path, max_entries=2) == (
        "a.txt\nb.txt\n... (truncated, 2 more entries)"
    )


def test_repository_map_rejects_non_positive_limit(tmp_path):
    with pytest.raises(ValueError, match="max_entries"):
        build_repository_map(tmp_path, max_entries=0)


def test_repository_map_returns_stable_error_when_read_fails(tmp_path):
    with patch(
        "remediation_engine.tools.source.os.walk",
        side_effect=PermissionError("permission denied"),
    ):
        result = build_repository_map(tmp_path)

    assert result.startswith("ERROR_CODE: REPOSITORY_MAP_UNAVAILABLE:")
    assert "permission denied" in result
