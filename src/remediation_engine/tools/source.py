"""Source parsing and deterministic repository map helpers."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

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
except Exception as _core_err:  # pragma: no cover
    log.warning("tree-sitter core unavailable: %s", _core_err)

try:
    import tree_sitter_javascript as _tsjs
    import tree_sitter_typescript as _tsts
    from tree_sitter import Language, Node, Parser, Tree

    _TREE_SITTER_CORE_AVAILABLE = True
    _JS_LANG = Language(_tsjs.language())
    _TS_LANG = Language(_tsts.language_typescript())
    _TSX_LANG = Language(_tsts.language_tsx())
    _TREE_SITTER_AVAILABLE = True
except Exception as _ts_err:  # pragma: no cover
    log.warning("tree-sitter unavailable — source inspection will be skipped: %s", _ts_err)

if _TREE_SITTER_CORE_AVAILABLE:
    try:
        import tree_sitter_python as _tspy

        _PY_LANG = Language(_tspy.language())
        _PYTHON_TREE_SITTER_AVAILABLE = True
    except Exception as _py_err:  # pragma: no cover
        log.warning("tree-sitter Python grammar unavailable: %s", _py_err)


# Mapping of file extension → Language object
_EXT_TO_LANG = {}
if _TREE_SITTER_AVAILABLE:
    _EXT_TO_LANG = {
        ".js": _JS_LANG,
        ".jsx": _JS_LANG,
        ".ts": _TS_LANG,
        ".tsx": _TSX_LANG,
        ".mjs": _JS_LANG,
        ".cjs": _JS_LANG,
    }
if _PYTHON_TREE_SITTER_AVAILABLE:
    _EXT_TO_LANG[".py"] = _PY_LANG


def language_for_path(file_path: str) -> object | None:
    """Return the tree-sitter ``Language`` for a file extension, or ``None``."""
    ext = Path(file_path).suffix.lower()
    return _EXT_TO_LANG.get(ext)


# ---------------------------------------------------------------------------
# File resolution
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Source loading
# ---------------------------------------------------------------------------


def load_source_bytes(path: Path) -> bytes | None:
    """Read raw bytes from *path*, returning ``None`` on any IO error.

    We do **not** decode here — tree-sitter works natively on bytes, and
    preserving bytes lets us avoid encoding surprises.
    """
    try:
        return path.read_bytes()
    except OSError as exc:
        log.warning("load_source_bytes: cannot read %s: %s", path, exc)
        return None


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


def _definition_node(node: Node) -> Node:  # type: ignore[name-defined]
    """Return a Python declaration wrapped in a decorated-definition node."""
    if node.type == "decorated_definition":
        for child in node.children:
            if child.type in ("function_definition", "class_definition"):
                return child
    return node


def _normalized_definition_node(node: Node) -> Node:  # type: ignore[name-defined]
    """Use the outer Python decorator node as the replacement source span."""
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
    # Standard named fields used by JS/TS grammars
    for field in ("name", "key"):
        child = node.child_by_field_name(field)
        if child is not None and child.text:
            text = child.text
            return text.decode("utf-8", errors="replace") if isinstance(text, bytes) else str(text)

    # Arrow functions assigned to a variable: check parent
    if node.type == "arrow_function" and node.parent is not None:
        parent = node.parent
        # variable_declarator → identifier
        if parent.type == "variable_declarator":
            id_child = parent.child_by_field_name("name")
            if id_child is not None and id_child.text:
                text = id_child.text
                return (
                    text.decode("utf-8", errors="replace") if isinstance(text, bytes) else str(text)
                )
    return None


# Schema enum labels for each enclosing node type


# Node types directly searchable by name.
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
        source_bytes: Raw source bytes — used to extract node text.
        line_hint: 1-indexed line number. When multiple candidates are found
            this is used to select the one whose range encloses or is nearest
            to *line_hint*.

    Returns:
        A ``dict`` with keys:

        - ``symbol_name`` (str) — matched name
        - ``node_type`` (str) — tree-sitter node type string
        - ``start_line`` (int) — 1-indexed
        - ``end_line`` (int)   — 1-indexed
        - ``text`` (str)       — raw node text (may be truncated in the caller)

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

    # Multiple exact name matches — try to resolve by line_hint.
    if line_hint is not None:
        # Prefer a candidate whose range encloses the hint line.
        enclosing = [c for c in candidates if c["start_line"] <= line_hint <= c["end_line"]]
        if len(enclosing) == 1:
            return enclosing[0]
        if len(enclosing) > 1:
            # Among enclosing, prefer the innermost (smallest range).
            return min(enclosing, key=lambda c: c["end_line"] - c["start_line"])

        # No enclosing match — pick the nearest by start_line.
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
    """Return top-level import/require statements as stripped text strings.

    Captures:
    - ES6 ``import`` declarations
    - ``const x = require(...)`` expression statements
    """
    return _extract_import_statements(root)[:20]


def extract_all_imports(root: Node, source_bytes: bytes) -> list[str]:  # type: ignore[name-defined]
    """Return every top-level import/require statement for deterministic analysis."""
    return _extract_import_statements(root)


def _extract_import_statements(root: Node) -> list[str]:  # type: ignore[name-defined]
    """Extract top-level JS/Python imports and CommonJS requires without a cap."""
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

    return results


# ---------------------------------------------------------------------------
# Bounded repository map
# ---------------------------------------------------------------------------

DEFAULT_REPOSITORY_MAP_MAX_ENTRIES = 400
_EXCLUDED_DIRECTORY_NAMES = frozenset({".git", "node_modules"})


def build_repository_map(
    repo_root: str | Path,
    max_entries: int = DEFAULT_REPOSITORY_MAP_MAX_ENTRIES,
) -> str:
    """Return a deterministic, bounded list of repository entries.

    Args:
        repo_root: Host repository directory to inspect.
        max_entries: Maximum number of relative entries to include before
            appending a truncation marker.

    Returns:
        A sorted newline-delimited list of repository-relative POSIX paths,
        ``"(workspace is empty)"`` for an empty repository, or a bounded
        ``ERROR_CODE: REPOSITORY_MAP_UNAVAILABLE`` message when the root
        cannot be inspected.

    Raises:
        ValueError: If ``max_entries`` is not positive.
    """
    if max_entries <= 0:
        raise ValueError("max_entries must be positive")

    try:
        root = Path(repo_root).resolve()
        if not root.is_dir():
            return "ERROR_CODE: REPOSITORY_MAP_UNAVAILABLE: repository root is not a directory"
    except (OSError, RuntimeError) as exc:
        return f"ERROR_CODE: REPOSITORY_MAP_UNAVAILABLE: {str(exc)[:500]}"

    entries: set[str] = set()
    errors: list[str] = []

    def on_walk_error(error: OSError) -> None:
        """Collect a bounded filesystem error without aborting the walk."""
        errors.append(str(error))

    try:
        for current_dir, dirnames, filenames in os.walk(
            root,
            topdown=True,
            onerror=on_walk_error,
            followlinks=False,
        ):
            current_path = Path(current_dir)
            dirnames[:] = sorted(
                dirname
                for dirname in dirnames
                if dirname not in _EXCLUDED_DIRECTORY_NAMES and not dirname.endswith(".map")
            )

            relative_dir = current_path.relative_to(root)
            if relative_dir != Path("."):
                entries.add(relative_dir.as_posix())

            for filename in filenames:
                if filename.endswith(".map"):
                    continue
                relative_file = (current_path / filename).relative_to(root)
                entries.add(relative_file.as_posix())
    except OSError as exc:
        errors.append(str(exc))

    if errors:
        detail = "; ".join(errors)[:500]
        return f"ERROR_CODE: REPOSITORY_MAP_UNAVAILABLE: {detail}"

    ordered_entries = sorted(entries)
    if not ordered_entries:
        return "(workspace is empty)"

    capped = ordered_entries[:max_entries]
    output = "\n".join(capped)
    if len(ordered_entries) > max_entries:
        output += f"\n... (truncated, {len(ordered_entries) - max_entries} more entries)"
    return output
