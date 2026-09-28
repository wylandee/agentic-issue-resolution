"""
code_map.py â€” Reusable AST & file helpers for the SAST code locator.

Responsibilities
----------------
* Resolve repo-relative file paths to absolute paths within a repo root.
* Load raw source bytes from disk (no network, no side effects).
* Build and cache tree-sitter ``Language`` / ``Parser`` objects for JS/TS and Python.
* Parse source bytes into a ``tree_sitter.Tree``, with a safe fallback.
* Walk an AST to find the enclosing function/class/method at a given line.
* Extract import statements from the top-level of a file.
* Extract a bounded code snippet around a line range.

Design constraints
------------------
* No mutation of external state.
* No network calls.
* No file writes.
* Graceful degradation: every public function returns ``None`` / empty list on
  failure rather than raising, unless the caller explicitly opts into exceptions.

Language support
----------------
JavaScript (.js, .jsx), TypeScript (.ts, .tsx), and Python (.py) use tree-sitter.
The Python grammar is loaded independently so its failure does not disable JS/TS.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from remediation_engine.runtime.path_policy import (
    WorkspacePathError,
    resolve_repository_path,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy language / parser construction
# ---------------------------------------------------------------------------

_JS_LANG: object = None
_TS_LANG: object = None
_TSX_LANG: object = None
_PY_LANG: object = None
_TREE_SITTER_CORE_AVAILABLE: bool = False
_TREE_SITTER_AVAILABLE: bool = False
_PYTHON_TREE_SITTER_AVAILABLE: bool = False

try:
    from tree_sitter import Language, Node, Parser, Tree

    _TREE_SITTER_CORE_AVAILABLE = True
except Exception as _ts_err:  # pragma: no cover
    log.warning(
        "tree-sitter core unavailable â€” AST localization will be skipped: %s",
        _ts_err,
    )

if _TREE_SITTER_CORE_AVAILABLE:
    try:
        import tree_sitter_javascript as _tsjs
        import tree_sitter_typescript as _tsts

        _JS_LANG = Language(_tsjs.language())
        _TS_LANG = Language(_tsts.language_typescript())
        _TSX_LANG = Language(_tsts.language_tsx())
        _TREE_SITTER_AVAILABLE = True
    except Exception as _js_ts_err:  # pragma: no cover
        log.warning("tree-sitter JavaScript/TypeScript grammars unavailable: %s", _js_ts_err)

    try:
        import tree_sitter_python as _tspy

        _PY_LANG = Language(_tspy.language())
        _PYTHON_TREE_SITTER_AVAILABLE = True
    except Exception as _py_err:  # pragma: no cover
        log.warning("tree-sitter Python grammar unavailable: %s", _py_err)


# Mapping of file extension â†’ Language object
_EXT_TO_LANG = {}
if _TREE_SITTER_AVAILABLE:
    _EXT_TO_LANG.update(
        {
            ".js": _JS_LANG,
            ".jsx": _JS_LANG,
            ".ts": _TS_LANG,
            ".tsx": _TSX_LANG,
            ".mjs": _JS_LANG,
            ".cjs": _JS_LANG,
        }
    )
if _PYTHON_TREE_SITTER_AVAILABLE:
    _EXT_TO_LANG[".py"] = _PY_LANG


def language_for_path(file_path: str) -> object | None:
    """Return the tree-sitter ``Language`` for a file extension, or ``None``."""
    ext = Path(file_path).suffix.lower()
    return _EXT_TO_LANG.get(ext)


# ---------------------------------------------------------------------------
# File resolution
# ---------------------------------------------------------------------------


def resolve_repo_file(repo_root: str, repo_relative_path: str) -> Path | None:
    """Resolve a repo-relative path to an absolute ``Path``, or ``None`` if missing.

    Args:
        repo_root: Absolute path to the repository workspace root.
        repo_relative_path: Path as stored in the finding (may have a leading slash stripped).

    Returns:
        An absolute ``Path`` pointing to the file, or ``None`` when it does not exist.
    """
    # Scanner paths historically arrived with a leading slash even when they
    # were repository-relative. Preserve that compatibility spelling while
    # routing the actual resolution through the shared traversal/symlink
    # containment policy.
    rel = str(repo_relative_path).replace("\\", "/").lstrip("/")
    try:
        candidate = resolve_repository_path(repo_root, rel)
    except (OSError, WorkspacePathError) as exc:
        log.warning("resolve_repo_file: blocked path %r: %s", repo_relative_path, exc)
        return None
    if candidate.is_file():
        return candidate
    log.debug("resolve_repo_file: not found: %s", candidate)
    return None


# ---------------------------------------------------------------------------
# Source loading
# ---------------------------------------------------------------------------


def load_source_bytes(path: Path) -> bytes | None:
    """Read raw bytes from *path*, returning ``None`` on any IO error.

    We do **not** decode here â€” tree-sitter works natively on bytes, and
    preserving bytes lets us avoid encoding surprises.
    """
    try:
        return path.read_bytes()
    except OSError as exc:
        log.warning("load_source_bytes: cannot read %s: %s", path, exc)
        return None


def load_source_lines(path: Path) -> list[str] | None:
    """Return the file as a list of text lines, or ``None`` on error.

    Uses UTF-8 with ``errors='replace'`` to avoid crashing on binary content.
    """
    raw = load_source_bytes(path)
    if raw is None:
        return None
    return raw.decode("utf-8", errors="replace").splitlines()


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_source(source_bytes: bytes, language: object) -> Tree | None:  # type: ignore[name-defined]
    """Parse *source_bytes* with *language*, returning a ``Tree`` or ``None``.

    A parse result is always returned by tree-sitter (even for invalid code), so
    ``None`` is only returned when tree-sitter is unavailable.
    """
    if not _TREE_SITTER_CORE_AVAILABLE:
        return None
    try:
        parser = Parser(language)  # type: ignore[arg-type]
        return parser.parse(source_bytes)
    except Exception as exc:  # pragma: no cover
        log.warning("parse_source: tree-sitter parse failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------

_ENCLOSING_TYPES: tuple[str, ...] = (
    "function_declaration",
    "function_expression",
    "generator_function_declaration",
    "generator_function",
    "arrow_function",
    "method_definition",
    "class_declaration",
    "class_expression",
    "function_definition",
    "class_definition",
    "decorated_definition",
)


def _node_contains_line(node: Node, zero_indexed_line: int) -> bool:  # type: ignore[name-defined]
    """Return True when *zero_indexed_line* falls within *node*'s extent."""
    return node.start_point[0] <= zero_indexed_line <= node.end_point[0]


def _definition_node(node: Node) -> Node:  # type: ignore[name-defined]
    """Return the function/class declaration wrapped by a Python decorator node."""
    if node.type == "decorated_definition":
        for child in node.children:
            if child.type in ("function_definition", "class_definition"):
                return child
    return node


def _normalized_definition_node(node: Node) -> Node:  # type: ignore[name-defined]
    """Use the outer Python decorated definition as the symbol's source span."""
    if (
        node.type in ("function_definition", "class_definition")
        and node.parent is not None
        and node.parent.type == "decorated_definition"
    ):
        return node.parent
    return node


def _extract_node_name(node: Node) -> str | None:  # type: ignore[name-defined]
    """Best-effort extraction of a human-readable name for *node*."""
    node = _definition_node(node)

    # Standard named fields used by JS/TS/Python grammars
    for field in ("name", "key"):
        child = node.child_by_field_name(field)
        if child is not None and child.text:
            text = child.text
            return text.decode("utf-8", errors="replace") if isinstance(text, bytes) else str(text)

    # Arrow functions assigned to a variable: check parent
    if node.type == "arrow_function" and node.parent is not None:
        parent = node.parent
        # variable_declarator â†’ identifier
        if parent.type == "variable_declarator":
            id_child = parent.child_by_field_name("name")
            if id_child is not None and id_child.text:
                text = id_child.text
                return (
                    text.decode("utf-8", errors="replace") if isinstance(text, bytes) else str(text)
                )
    return None


# Schema enum labels for each enclosing node type
_TYPE_ENUM_MAP: dict = {}
if _TREE_SITTER_CORE_AVAILABLE:
    from remediation_engine.contracts.schemas import ASTNodeType

    _TYPE_ENUM_MAP = {
        "function_declaration": ASTNodeType.FUNCTION,
        "function_expression": ASTNodeType.FUNCTION,
        "generator_function_declaration": ASTNodeType.FUNCTION,
        "generator_function": ASTNodeType.FUNCTION,
        "arrow_function": ASTNodeType.ARROW_FUNCTION,
        "method_definition": ASTNodeType.METHOD,
        "class_declaration": ASTNodeType.CLASS,
        "class_expression": ASTNodeType.CLASS,
        "function_definition": ASTNodeType.FUNCTION,
        "class_definition": ASTNodeType.CLASS,
    }


def _symbol_enum_type(node: Node) -> ASTNodeType:  # type: ignore[name-defined]
    """Map a symbol node to the stable, language-neutral AST node enum."""
    definition = _definition_node(node)
    if definition.type == "function_definition":
        scope = definition.parent
        if scope is not None and scope.type == "decorated_definition":
            scope = scope.parent
        if (
            scope is not None
            and scope.type == "block"
            and scope.parent is not None
            and scope.parent.type == "class_definition"
        ):
            return ASTNodeType.METHOD
    return _TYPE_ENUM_MAP.get(definition.type, ASTNodeType.UNKNOWN)


def find_enclosing_symbol(
    root: Node,  # type: ignore[name-defined]
    one_indexed_line: int,
) -> tuple[str | None, object]:
    """Walk the AST from *root* to find the innermost enclosing symbol at *one_indexed_line*.

    Args:
        root: The ``tree_sitter.Node`` root of the parsed tree.
        one_indexed_line: 1-indexed line number from the Semgrep finding.

    Returns:
        A ``(symbol_name, ASTNodeType)`` tuple.  Both values are ``None`` /
        ``ASTNodeType.UNKNOWN`` when nothing is found.
    """
    if not _TREE_SITTER_CORE_AVAILABLE:
        from remediation_engine.contracts.schemas import ASTNodeType

        return None, ASTNodeType.UNKNOWN

    from remediation_engine.contracts.schemas import ASTNodeType

    target = one_indexed_line - 1  # convert to 0-indexed
    best_name: str | None = None
    best_type: ASTNodeType = ASTNodeType.UNKNOWN
    best_size: int = -1  # track smallest enclosing node

    stack = list(root.children)
    while stack:
        node = stack.pop()
        if not _node_contains_line(node, target):
            continue
        stack.extend(node.children)

        if node.type not in _ENCLOSING_TYPES:
            continue

        symbol_node = _normalized_definition_node(node)
        node_size = symbol_node.end_point[0] - symbol_node.start_point[0]
        # Prefer innermost (smallest) enclosing node
        if best_size == -1 or node_size <= best_size:
            name = _extract_node_name(symbol_node)
            if name:
                best_name = name
                best_type = _symbol_enum_type(symbol_node)
                best_size = node_size

    return best_name, best_type


# Node types that are directly searchable by name in `find_named_symbol`.
_NAMED_TYPES: tuple[str, ...] = (
    "function_declaration",
    "function_expression",
    "generator_function_declaration",
    "generator_function",
    "arrow_function",
    "method_definition",
    "class_declaration",
    "class_expression",
    "function_definition",
    "class_definition",
)


def find_named_symbol(
    root: Node,  # type: ignore[name-defined]
    symbol_name: str,
    source_bytes: bytes,
    *,
    line_hint: int | None = None,
) -> dict[str, Any] | None:
    """Find an AST node whose resolved name exactly matches *symbol_name*.

    Args:
        root: The ``tree_sitter.Node`` root of the parsed tree.
        symbol_name: Exact name to search for (case-sensitive).
        source_bytes: Raw source bytes â€” used to extract node text.
        line_hint: 1-indexed line number. When multiple candidates are found
            this is used to select the one whose range encloses or is nearest
            to *line_hint*.

    Returns:
        A ``dict`` with keys:

        - ``symbol_name`` (str) â€” matched name
        - ``node_type`` (str) â€” tree-sitter node type string
        - ``start_line`` (int) â€” 1-indexed
        - ``end_line`` (int)   â€” 1-indexed
        - ``text`` (str)       â€” raw node text (may be truncated in the caller)

        Returns ``None`` when no match is found.
        Raises ``ValueError`` on ambiguous matches that cannot be resolved
        by *line_hint*.
    """
    if not _TREE_SITTER_CORE_AVAILABLE:
        raise RuntimeError("tree-sitter is unavailable.")

    candidates: list[dict[str, Any]] = []

    stack = list(root.children)
    while stack:
        node = stack.pop()
        stack.extend(node.children)

        if node.type not in _NAMED_TYPES:
            continue

        name = _extract_node_name(node)
        if name != symbol_name:
            continue

        symbol_node = _normalized_definition_node(node)
        raw = symbol_node.text
        node_text = (
            raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw or "")
        )
        candidates.append(
            {
                "symbol_name": name,
                "node_type": symbol_node.type,
                "start_line": symbol_node.start_point[0] + 1,  # 1-indexed
                "end_line": symbol_node.end_point[0] + 1,  # 1-indexed
                "start_byte": symbol_node.start_byte,
                "end_byte": symbol_node.end_byte,
                "text": node_text,
            }
        )

    if not candidates:
        return None

    if len(candidates) == 1:
        return candidates[0]

    # Multiple exact name matches â€” try to resolve by line_hint.
    if line_hint is not None:
        # Prefer a candidate whose range encloses the hint line.
        enclosing = [c for c in candidates if c["start_line"] <= line_hint <= c["end_line"]]
        if len(enclosing) == 1:
            return enclosing[0]
        if len(enclosing) > 1:
            # Among enclosing, prefer the innermost (smallest range).
            return min(enclosing, key=lambda c: c["end_line"] - c["start_line"])

        # No enclosing match â€” pick the nearest by start_line.
        return min(candidates, key=lambda c: abs(c["start_line"] - line_hint))

    # Unresolvable ambiguity: surface deterministic error with all ranges.
    ranges = ", ".join(
        f"L{c['start_line']}-{c['end_line']}"
        for c in sorted(candidates, key=lambda c: c["start_line"])
    )
    raise ValueError(
        f"Ambiguous symbol '{symbol_name}': found {len(candidates)} definitions at "
        f"{ranges}. Provide line_hint to disambiguate."
    )


def extract_imports(root: Node, source_bytes: bytes) -> list[str]:  # type: ignore[name-defined]
    """Return top-level JS/Python imports and CommonJS require statements.

    Captures ES6 ``import`` declarations, Python ``import``/``from`` statements,
    and ``const x = require(...)`` expression statements.
    """
    if not _TREE_SITTER_CORE_AVAILABLE:
        return []

    results: list[str] = []

    for child in root.children:
        node_type = child.type
        if node_type in ("import_statement", "import_from_statement"):
            text = child.text
            if isinstance(text, bytes):
                text = text.decode("utf-8", errors="replace")
            results.append(text.strip())
        elif node_type in ("expression_statement", "lexical_declaration", "variable_declaration"):
            # Capture require() calls at the top level
            raw = child.text
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="replace")
            if "require(" in raw:
                results.append(raw.strip())

    return results[:20]  # cap at 20 to stay context-sized


def extract_sink_expression(
    root: Node,  # type: ignore[name-defined]
    one_indexed_line: int,
    source_bytes: bytes,
) -> str | None:
    """Return the innermost call expression on *one_indexed_line*, if any.

    Recognizes JavaScript/TypeScript ``call_expression`` and Python ``call``
    nodes. Falls back to the stripped source line when no call is present.
    """
    if not _TREE_SITTER_CORE_AVAILABLE:
        return None

    target = one_indexed_line - 1  # 0-indexed

    best_node: Node | None = None  # type: ignore[name-defined]
    best_size = -1

    stack = list(root.children)
    while stack:
        node = stack.pop()
        if not _node_contains_line(node, target):
            continue
        stack.extend(node.children)

        if node.type not in ("call_expression", "call"):
            continue

        size = node.end_point[0] - node.start_point[0]
        # Find the innermost call_expression on that line
        if best_size == -1 or size <= best_size:
            best_node = node
            best_size = size

    if best_node is not None:
        raw = best_node.text
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        # Truncate to a single line to avoid multi-line blobs
        return raw.split("\n")[0].strip()[:200]

    # Fallback: return raw source line
    lines = source_bytes.decode("utf-8", errors="replace").splitlines()
    if 1 <= one_indexed_line <= len(lines):
        return lines[one_indexed_line - 1].strip()[:200]

    return None


# ---------------------------------------------------------------------------
# Snippet extraction (pure text, no AST required)
# ---------------------------------------------------------------------------

_SNIPPET_RADIUS = 10  # lines of context above and below the finding


def extract_snippet(
    lines: list[str],
    start_line: int,
    end_line: int,
    *,
    max_lines: int = 30,
) -> str:
    """Return a bounded code snippet centred on ``[start_line, end_line]``.

    Args:
        lines: Full source file as a list of text lines.
        start_line: 1-indexed start of the finding range.
        end_line: 1-indexed end of the finding range (inclusive).
        max_lines: Hard cap on snippet length (default 30).

    Returns:
        A multi-line string with the relevant code, or an empty string if
        *lines* is empty or the range is invalid.
    """
    if not lines:
        return ""

    n = len(lines)
    # Convert to 0-indexed
    lo = max(0, start_line - 1 - _SNIPPET_RADIUS)
    hi = min(n, end_line + _SNIPPET_RADIUS)

    # Apply max_lines cap symmetrically around the finding centre.  Compute
    # the final window explicitly because the previous inclusive/exclusive
    # arithmetic could return max_lines + 1 rows for even caps.
    if max_lines <= 0:
        return ""
    centre = (start_line + end_line) // 2 - 1
    if hi - lo > max_lines:
        lo = max(0, min(n - max_lines, centre - max_lines // 2))
        hi = lo + max_lines

    return "\n".join(lines[lo:hi])
