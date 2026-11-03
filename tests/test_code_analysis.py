"""Focused tests for code analysis."""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from remediation_engine.contracts.schemas import (
    ASTNodeType,
    CommandResult,
    IssueSource,
    IssueType,
    LineRange,
    LocalizedIssue,
    Severity,
    VulnerabilityIssue,
)
from remediation_engine.tools.source import (
    _PYTHON_TREE_SITTER_AVAILABLE,
    _TREE_SITTER_AVAILABLE,
    _extract_data_flow_hints,
    _score_confidence,
    build_repository_map,
    extract_all_imports,
    extract_imports,
    extract_sink_expression,
    extract_snippet,
    find_enclosing_symbol,
    find_named_symbol,
    language_for_path,
    load_source_bytes,
    load_source_lines,
    locate_sast,
    parse_source,
    resolve_repo_file,
)
from remediation_engine.tools.workers import _make_deterministic_replace_ast_symbol_tool

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_issue(
    *,
    issue_type: IssueType = IssueType.SAST,
    file_path: str | None = "src/app.js",
    line_start: int = 5,
    line_end: int = 5,
) -> VulnerabilityIssue:
    lr = LineRange(start=line_start, end=line_end) if line_start else None
    return VulnerabilityIssue(
        source=IssueSource.SEMGREP,
        issue_type=issue_type,
        severity=Severity.HIGH,
        file_path=file_path,
        line_range=lr,
        message="Test finding",
    )


# ---------------------------------------------------------------------------
# TestCodeMapHelpers
# ---------------------------------------------------------------------------


class TestCodeMapHelpers:
    """Unit tests for the pure helper functions in code_map.py."""

    # --- resolve_repo_file ---

    def test_resolve_repo_file_found(self, tmp_path: Path):
        (tmp_path / "src").mkdir()
        target = tmp_path / "src" / "app.js"
        target.write_text("console.log('hi')")
        result = resolve_repo_file(str(tmp_path), "src/app.js")
        assert result == target

    def test_resolve_repo_file_leading_slash_stripped(self, tmp_path: Path):
        target = tmp_path / "index.js"
        target.write_text("x")
        result = resolve_repo_file(str(tmp_path), "/index.js")
        assert result == target

    def test_resolve_repo_file_missing_returns_none(self, tmp_path: Path):
        result = resolve_repo_file(str(tmp_path), "does/not/exist.js")
        assert result is None

    def test_resolve_repo_file_rejects_traversal(self, tmp_path: Path):
        outside = tmp_path.parent / f"{tmp_path.name}-outside"
        outside.mkdir()
        target = outside / "secret.js"
        target.write_text("secret")

        assert resolve_repo_file(str(tmp_path), "../" + target.name) is None

    # --- load_source_bytes / load_source_lines ---

    def test_load_source_bytes_reads_file(self, tmp_path: Path):
        f = tmp_path / "x.js"
        f.write_bytes(b"hello world")
        assert load_source_bytes(f) == b"hello world"

    def test_load_source_bytes_missing_returns_none(self, tmp_path: Path):
        assert load_source_bytes(tmp_path / "ghost.js") is None

    def test_load_source_lines_basic(self, tmp_path: Path):
        f = tmp_path / "x.js"
        f.write_text("line1\nline2\nline3")
        lines = load_source_lines(f)
        assert lines == ["line1", "line2", "line3"]

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

    def test_extract_snippet_basic(self):
        lines = [f"line{i}" for i in range(1, 101)]  # 100 lines
        snippet = extract_snippet(lines, 50, 50)
        assert "line50" in snippet

    def test_extract_snippet_respects_max_lines(self):
        lines = [f"L{i}" for i in range(1, 1001)]
        snippet = extract_snippet(lines, 500, 500, max_lines=30)
        assert len(snippet.splitlines()) <= 30

    def test_extract_snippet_empty_lines(self):
        assert extract_snippet([], 1, 1) == ""

    def test_extract_snippet_at_file_start(self):
        lines = ["import x", "const y = 1", "x()"]
        snippet = extract_snippet(lines, 1, 1)
        assert "import x" in snippet

    def test_extract_snippet_at_file_end(self):
        lines = ["a", "b", "c"]
        snippet = extract_snippet(lines, 3, 3)
        assert "c" in snippet


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

    def test_find_enclosing_symbol_function_declaration(self, tmp_path):
        code = textwrap.dedent("""\
            function handleLogin(req, res) {
              const user = req.body.username;
              return user;
            }
        """)
        lang = language_for_path("app.js")
        tree = parse_source(code.encode(), lang)
        root = tree.root_node
        name, node_type = find_enclosing_symbol(root, 2)  # line 2: const user = ...
        assert name == "handleLogin"
        assert node_type == ASTNodeType.FUNCTION

    def test_find_enclosing_symbol_class_method(self):
        code = textwrap.dedent("""\
            class UserController {
              login(req, res) {
                const id = req.params.id;
                res.send(id);
              }
            }
        """)
        lang = language_for_path("app.js")
        tree = parse_source(code.encode(), lang)
        root = tree.root_node
        name, node_type = find_enclosing_symbol(root, 3)
        assert name == "login"
        assert node_type == ASTNodeType.METHOD

    def test_find_enclosing_symbol_arrow_function(self):
        code = textwrap.dedent("""\
            const processInput = (data) => {
              eval(data);
            };
        """)
        lang = language_for_path("app.js")
        tree = parse_source(code.encode(), lang)
        root = tree.root_node
        name, node_type = find_enclosing_symbol(root, 2)
        assert name == "processInput"
        assert node_type == ASTNodeType.ARROW_FUNCTION

    def test_find_enclosing_symbol_none_at_top_level(self):
        code = "const x = require('fs');\n"
        lang = language_for_path("app.js")
        tree = parse_source(code.encode(), lang)
        root = tree.root_node
        name, node_type = find_enclosing_symbol(root, 1)
        # Top-level code has no enclosing function
        assert name is None
        assert node_type == ASTNodeType.UNKNOWN

    def test_find_enclosing_symbol_nested_prefers_innermost(self):
        code = textwrap.dedent("""\
            function outer() {
              function inner() {
                eval('bad');
              }
            }
        """)
        lang = language_for_path("app.js")
        tree = parse_source(code.encode(), lang)
        root = tree.root_node
        name, node_type = find_enclosing_symbol(root, 3)
        assert name == "inner"

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

    def test_extract_sink_expression_call(self):
        code = textwrap.dedent("""\
            function f(x) {
              eval(x);
            }
        """)
        lang = language_for_path("app.js")
        tree = parse_source(code.encode(), lang)
        root = tree.root_node
        sink = extract_sink_expression(root, 2, code.encode())
        assert sink is not None
        assert "eval" in sink

    def test_extract_sink_expression_fallback_to_raw_line(self):
        code = "const x = 1;\nthis_is_not_a_call = something;\n"
        lang = language_for_path("app.js")
        tree = parse_source(code.encode(), lang)
        root = tree.root_node
        sink = extract_sink_expression(root, 2, code.encode())
        # Should return the raw source line as fallback
        assert sink is not None


# ---------------------------------------------------------------------------
# TestDataFlowHints
# ---------------------------------------------------------------------------


class TestDataFlowHints:
    def test_detects_req_body(self):
        hints = _extract_data_flow_hints("const user = req.body.username;")
        assert any("req.body" in h for h in hints)

    def test_detects_eval_sink(self):
        hints = _extract_data_flow_hints("eval(userInput);")
        assert any("eval" in h for h in hints)

    def test_detects_innerHTML(self):
        hints = _extract_data_flow_hints("el.innerHTML = value;")
        assert any("innerHTML" in h for h in hints)

    def test_detects_db_query(self):
        hints = _extract_data_flow_hints("db.query('SELECT * FROM users WHERE id=' + id);")
        assert any("DB query" in h or "db.query" in h.lower() for h in hints)

    def test_empty_snippet(self):
        hints = _extract_data_flow_hints("")
        assert hints == []

    def test_capped_at_8(self):
        # Craft a snippet hitting many patterns
        snippet = (
            "req.body.x; req.params.y; req.query.z; req.headers.h; "
            "eval(x); innerHTML=y; exec(z); spawn(w); db.query(q); new Function(s);"
        )
        hints = _extract_data_flow_hints(snippet)
        assert len(hints) <= 8


# ---------------------------------------------------------------------------
# TestScoreConfidence
# ---------------------------------------------------------------------------


class TestScoreConfidence:
    def test_file_unreadable_returns_zero(self):
        score = _score_confidence(
            file_readable=False,
            ast_available=True,
            enclosing_found=True,
            sink_found=True,
            imports_found=True,
        )
        assert score == 0.0

    def test_full_score_is_one(self):
        score = _score_confidence(
            file_readable=True,
            ast_available=True,
            enclosing_found=True,
            sink_found=True,
            imports_found=True,
        )
        assert score == 1.0

    def test_text_only_no_ast(self):
        score = _score_confidence(
            file_readable=True,
            ast_available=False,
            enclosing_found=False,
            sink_found=True,
            imports_found=False,
        )
        # 0.1 base + 0.2 sink = 0.3
        assert score == pytest.approx(0.3)

    def test_score_between_zero_and_one(self):
        for fr in [True, False]:
            for aa in [True, False]:
                for ef in [True, False]:
                    for sf in [True, False]:
                        for im in [True, False]:
                            s = _score_confidence(
                                file_readable=fr,
                                ast_available=aa,
                                enclosing_found=ef,
                                sink_found=sf,
                                imports_found=im,
                            )
                            assert 0.0 <= s <= 1.0


# ---------------------------------------------------------------------------
# TestLocateSast â€” happy path (real files + tree-sitter)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _TREE_SITTER_AVAILABLE, reason="tree-sitter not installed")
class TestLocateSast:
    """Integration tests that spin up real JS files in tmp_path."""

    def test_basic_js_file(self, tmp_path: Path):
        src = textwrap.dedent("""\
            const express = require('express');
            const router = require('./router');

            function handleLogin(req, res) {
                const user = req.body.username;
                res.send(user);
            }

            module.exports = { handleLogin };
        """)
        (tmp_path / "auth.js").write_text(src)

        issue = _make_issue(file_path="auth.js", line_start=5, line_end=5)
        result = locate_sast(issue, str(tmp_path))

        assert isinstance(result, LocalizedIssue)
        assert result.issue is issue
        assert result.enclosing_symbol == "handleLogin"
        assert result.enclosing_node_type == ASTNodeType.FUNCTION
        assert result.snippet is not None and "handleLogin" in result.snippet
        assert any("express" in i or "router" in i for i in result.imports)
        assert result.localization_confidence > 0.5

    def test_class_method_localization(self, tmp_path: Path):
        src = textwrap.dedent("""\
            class ProductService {
                search(req, res) {
                    const term = req.query.q;
                    db.query('SELECT * FROM products WHERE name LIKE ' + term);
                }
            }
        """)
        (tmp_path / "product.js").write_text(src)

        issue = _make_issue(file_path="product.js", line_start=4, line_end=4)
        result = locate_sast(issue, str(tmp_path))

        assert result.enclosing_symbol == "search"
        assert result.enclosing_node_type == ASTNodeType.METHOD
        assert any("DB query" in h or "db.query" in h.lower() for h in result.data_flow_hints)

    def test_typescript_file(self, tmp_path: Path):
        src = textwrap.dedent("""\
            import { Request, Response } from 'express';

            function getUser(req: Request, res: Response): void {
                const id: string = req.params.id;
                res.send(id);
            }
        """)
        (tmp_path / "user.ts").write_text(src)

        issue = _make_issue(file_path="user.ts", line_start=4, line_end=4)
        result = locate_sast(issue, str(tmp_path))

        assert result.enclosing_symbol == "getUser"
        assert result.localization_confidence > 0.5

    def test_arrow_function_localization(self, tmp_path: Path):
        src = textwrap.dedent("""\
            const runQuery = (input) => {
                eval(input);
            };
        """)
        (tmp_path / "utils.js").write_text(src)

        issue = _make_issue(file_path="utils.js", line_start=2, line_end=2)
        result = locate_sast(issue, str(tmp_path))

        assert result.enclosing_symbol == "runQuery"
        assert result.enclosing_node_type == ASTNodeType.ARROW_FUNCTION

    def test_sink_expression_extracted(self, tmp_path: Path):
        src = "function f(x) {\n  eval(x + ' extra');\n}\n"
        (tmp_path / "f.js").write_text(src)

        issue = _make_issue(file_path="f.js", line_start=2, line_end=2)
        result = locate_sast(issue, str(tmp_path))

        assert result.sink_expression is not None
        assert "eval" in result.sink_expression

    def test_returns_localized_issue_not_vulnerability_issue(self, tmp_path: Path):
        (tmp_path / "a.js").write_text("const x = 1;\n")
        issue = _make_issue(file_path="a.js", line_start=1, line_end=1)
        result = locate_sast(issue, str(tmp_path))
        assert isinstance(result, LocalizedIssue)
        assert result.issue is issue


# ---------------------------------------------------------------------------
# TestLocateSastFallback â€” graceful degradation
# ---------------------------------------------------------------------------


class TestLocateSastFallback:
    """Edge-case and graceful-degradation tests that do NOT require tree-sitter."""

    def test_sca_issue_returns_stub(self, tmp_path: Path):
        issue = _make_issue(issue_type=IssueType.SCA)
        result = locate_sast(issue, str(tmp_path))
        assert result.localization_confidence == 0.0
        assert result.enclosing_symbol is None

    def test_missing_file_path_returns_stub(self, tmp_path: Path):
        issue = _make_issue(file_path=None)
        result = locate_sast(issue, str(tmp_path))
        assert result.localization_confidence == 0.0

    def test_file_not_on_disk_returns_stub(self, tmp_path: Path):
        issue = _make_issue(file_path="does/not/exist.js")
        result = locate_sast(issue, str(tmp_path))
        assert result.localization_confidence == 0.0

    def test_always_returns_localized_issue(self, tmp_path: Path):
        issue = _make_issue(file_path="ghost.js")
        result = locate_sast(issue, str(tmp_path))
        assert isinstance(result, LocalizedIssue)

    def test_issue_object_not_mutated(self, tmp_path: Path):
        (tmp_path / "x.js").write_text("const a = 1;\n")
        issue = _make_issue(file_path="x.js", line_start=1, line_end=1)
        original_id = issue.id
        locate_sast(issue, str(tmp_path))
        assert issue.id == original_id
        assert issue.file_path == "x.js"

    def test_python_file_uses_structural_localization_without_js_taint_hints(self, tmp_path: Path):
        src = "def login(username):\n    return eval(req.body.username)\n"
        (tmp_path / "app.py").write_text(src)
        issue = _make_issue(file_path="app.py", line_start=2, line_end=2)
        result = locate_sast(issue, str(tmp_path))

        assert result.snippet is not None
        assert result.data_flow_hints == []
        if _PYTHON_TREE_SITTER_AVAILABLE:
            assert result.enclosing_symbol == "login"
            assert result.enclosing_node_type == ASTNodeType.FUNCTION
            assert result.sink_expression == "eval(req.body.username)"
            assert result.localization_confidence >= 0.5
        else:
            assert result.enclosing_symbol is None
            assert 0.0 < result.localization_confidence < 0.5

    def test_empty_file_returns_low_confidence(self, tmp_path: Path):
        (tmp_path / "empty.js").write_bytes(b"")
        issue = _make_issue(file_path="empty.js", line_start=1, line_end=1)
        result = locate_sast(issue, str(tmp_path))
        # Empty file: readable=False (no lines), so confidence = 0
        assert isinstance(result, LocalizedIssue)

    def test_issue_without_line_range(self, tmp_path: Path):
        """Issues without a line_range should still not crash."""
        (tmp_path / "app.js").write_text("console.log('hi');\n")
        issue = VulnerabilityIssue(
            source=IssueSource.SEMGREP,
            issue_type=IssueType.SAST,
            severity=Severity.LOW,
            file_path="app.js",
            line_range=None,
        )
        result = locate_sast(issue, str(tmp_path))
        assert isinstance(result, LocalizedIssue)

    def test_localization_confidence_is_float_in_range(self, tmp_path: Path):
        (tmp_path / "b.js").write_text("const x = eval(y);\n")
        issue = _make_issue(file_path="b.js", line_start=1, line_end=1)
        result = locate_sast(issue, str(tmp_path))
        assert 0.0 <= result.localization_confidence <= 1.0


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
    assert find_enclosing_symbol(root, 3) == ("handle", ASTNodeType.METHOD)


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
    assert find_enclosing_symbol(root, 4) == ("load_user", ASTNodeType.FUNCTION)


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
    assert find_enclosing_symbol(root, 3) == ("User", ASTNodeType.CLASS)


@pytest.mark.skipif(
    not _PYTHON_TREE_SITTER_AVAILABLE,
    reason="tree-sitter Python grammar is unavailable",
)
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


@pytest.mark.skipif(
    not _PYTHON_TREE_SITTER_AVAILABLE,
    reason="tree-sitter Python grammar is unavailable",
)
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
