"""
code_locator.py â€” SAST finding localization using tree-sitter.

Public API
----------
``locate_sast(issue, repo_root) -> LocalizedIssue``
    Primary entry point.  Given a SAST ``VulnerabilityIssue`` and the absolute
    path to the target repository's workspace root, returns a ``LocalizedIssue``
    enriched with:

    - ``enclosing_symbol`` / ``enclosing_node_type``  â€” the nearest enclosing
      function, method, class, or arrow function (via tree-sitter AST walk).
    - ``sink_expression`` â€” the call-expression at the finding line.
    - ``imports`` â€” top-level import / require statements from the file.
    - ``snippet`` â€” a â‰¤ 30-line code excerpt centred on the finding.
    - ``localization_confidence`` â€” 0.0â€“1.0 score based on what was found.
    - ``data_flow_hints`` â€” simple taint-style hints derived from the snippet.

Design constraints
------------------
* No mutation of the input ``issue`` object.
* No network calls.
* No file writes.
* Graceful degradation at every step â€” a ``LocalizedIssue`` is always returned,
  even when tree-sitter is unavailable or the file cannot be read.

Language support (v1)
---------------------
JavaScript (.js, .jsx), TypeScript (.ts, .tsx), and their module variants.
Other languages will produce a text-only fallback localization.
"""

from __future__ import annotations

import logging
import re

from remediation_engine.contracts.schemas import (
    ASTNodeType,
    IssueType,
    LocalizedIssue,
    VulnerabilityIssue,
)
from remediation_engine.tools.code_map import (
    extract_imports,
    extract_sink_expression,
    extract_snippet,
    find_enclosing_symbol,
    language_for_path,
    load_source_bytes,
    load_source_lines,
    parse_source,
    resolve_repo_file,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Taint / data-flow hint patterns
# ---------------------------------------------------------------------------

# Common sources of untrusted data in JS/TS web apps
_TAINT_SOURCE_PATTERNS = [
    (re.compile(r"\breq\.(body|params|query|headers|cookies)\b"), "taint source: req.{group}"),
    (re.compile(r"\bprocess\.env\b"), "taint source: process.env"),
    (re.compile(r"\bwindow\.(location|search|hash|name)\b"), "taint source: window.{group}"),
    (re.compile(r"\bdocument\.(URL|referrer|cookie)\b"), "taint source: document.{group}"),
    (re.compile(r"\blocalStorage\.getItem\b"), "taint source: localStorage.getItem"),
    (re.compile(r"\bsessionStorage\.getItem\b"), "taint source: sessionStorage.getItem"),
    (re.compile(r"\bJSON\.parse\b"), "taint source: JSON.parse (deserialization)"),
]

# Common dangerous sinks
_TAINT_SINK_PATTERNS = [
    (re.compile(r"\beval\s*\("), "sink: eval()"),
    (re.compile(r"\binnerHTML\s*="), "sink: innerHTML assignment"),
    (re.compile(r"\bdocument\.write\s*\("), "sink: document.write()"),
    (re.compile(r"\bexec\s*\("), "sink: exec()"),
    (re.compile(r"\bexecSync\s*\("), "sink: execSync()"),
    (re.compile(r"\bspawn\s*\("), "sink: spawn()"),
    (re.compile(r"\bcreateElement\s*\("), "sink: createElement()"),
    (re.compile(r"\bsetTimeout\s*\(", re.IGNORECASE), "sink: setTimeout() with string arg"),
    (re.compile(r"\bsetInterval\s*\(", re.IGNORECASE), "sink: setInterval()"),
    (re.compile(r"\bnew\s+Function\s*\("), "sink: new Function() constructor"),
    (re.compile(r"\bres\.(send|json|render|end)\s*\("), "sink: HTTP response write"),
    (re.compile(r"\bdb\.(query|execute|run)\s*\("), "sink: raw DB query"),
    (re.compile(r"\bsequelize\.query\s*\("), "sink: Sequelize raw query"),
    (re.compile(r"\bknex\.raw\s*\("), "sink: Knex raw query"),
]


def _extract_data_flow_hints(snippet: str) -> list[str]:
    """Scan *snippet* for taint-source and sink patterns, returning brief hints."""
    hints: list[str] = []
    seen: set = set()

    for pattern, template in _TAINT_SOURCE_PATTERNS:
        m = pattern.search(snippet)
        if m:
            try:
                label = template.format(group=m.group(1))
            except (IndexError, KeyError):
                label = template
            if label not in seen:
                hints.append(label)
                seen.add(label)

    for pattern, label in _TAINT_SINK_PATTERNS:
        if pattern.search(snippet) and label not in seen:
            hints.append(label)
            seen.add(label)

    return hints[:8]  # cap to keep prompt context manageable


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------


def _score_confidence(
    *,
    file_readable: bool,
    ast_available: bool,
    enclosing_found: bool,
    sink_found: bool,
    imports_found: bool,
) -> float:
    """Compute a 0â€“1 localization confidence score.

    Weights:
    - file_readable: prerequisite â€” if False, score = 0.0
    - ast_available: +0.3 (tree-sitter worked)
    - enclosing_found: +0.3 (we know which function contains the issue)
    - sink_found: +0.2 (we identified the call expression)
    - imports_found: +0.1 (we extracted dependency context)
    """
    if not file_readable:
        return 0.0
    score = 0.1  # base: we at least have a snippet
    if ast_available:
        score += 0.3
    if enclosing_found:
        score += 0.3
    if sink_found:
        score += 0.2
    if imports_found:
        score += 0.1
    return round(min(score, 1.0), 2)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def locate_sast(
    issue: VulnerabilityIssue,
    repo_root: str,
) -> LocalizedIssue:
    """Localize a SAST ``VulnerabilityIssue`` within the repository at *repo_root*.

    Args:
        issue:     The ``VulnerabilityIssue`` to localize.  Must have
                   ``issue_type == IssueType.SAST``.  The object is not mutated.
        repo_root: Absolute path to the repository workspace root on disk.

    Returns:
        A ``LocalizedIssue`` â€” always.  On any failure the result will have
        ``localization_confidence == 0.0`` and all AST fields set to their
        defaults.

    Raises:
        Nothing â€” all errors are caught and logged.
    """
    if issue.issue_type != IssueType.SAST:
        log.warning(
            "locate_sast called with non-SAST issue %s (type=%s) â€” returning stub.",
            issue.id,
            issue.issue_type,
        )
        return LocalizedIssue(
            issue=issue,
            localization_confidence=0.0,
        )

    # ------------------------------------------------------------------
    # Step 1: Resolve the file path
    # ------------------------------------------------------------------
    if not issue.file_path:
        log.debug("locate_sast: issue %s has no file_path", issue.id)
        return LocalizedIssue(issue=issue, localization_confidence=0.0)

    abs_path = resolve_repo_file(repo_root, issue.file_path)
    if abs_path is None:
        log.debug(
            "locate_sast: file not found for issue %s: %s/%s",
            issue.id,
            repo_root,
            issue.file_path,
        )
        return LocalizedIssue(issue=issue, localization_confidence=0.0)

    # ------------------------------------------------------------------
    # Step 2: Load source
    # ------------------------------------------------------------------
    source_bytes = load_source_bytes(abs_path)
    if source_bytes is None:
        return LocalizedIssue(issue=issue, localization_confidence=0.0)

    source_lines = load_source_lines(abs_path) or []
    file_readable = bool(source_lines)

    # ------------------------------------------------------------------
    # Step 3: Determine line range
    # ------------------------------------------------------------------
    lr = issue.line_range
    start_line = lr.start if lr else 1
    end_line = lr.end if lr else start_line

    # ------------------------------------------------------------------
    # Step 4: Extract plain-text snippet (no AST needed)
    # ------------------------------------------------------------------
    snippet = extract_snippet(source_lines, start_line, end_line) if source_lines else None

    # ------------------------------------------------------------------
    # Step 5: Attempt AST-based enrichment
    # ------------------------------------------------------------------
    language = language_for_path(issue.file_path)
    ast_available = False
    enclosing_symbol: str | None = None
    enclosing_node_type: ASTNodeType = ASTNodeType.UNKNOWN
    sink_expression: str | None = None
    imports: list[str] = []

    if language is not None:
        tree = parse_source(source_bytes, language)
        if tree is not None:
            ast_available = True
            root = tree.root_node

            # 5a. Enclosing symbol
            try:
                enclosing_symbol, enclosing_node_type = find_enclosing_symbol(root, start_line)
            except Exception as exc:
                log.warning("locate_sast: find_enclosing_symbol failed: %s", exc)

            # 5b. Sink expression
            try:
                sink_expression = extract_sink_expression(root, start_line, source_bytes)
            except Exception as exc:
                log.warning("locate_sast: extract_sink_expression failed: %s", exc)

            # 5c. Imports
            try:
                imports = extract_imports(root, source_bytes)
            except Exception as exc:
                log.warning("locate_sast: extract_imports failed: %s", exc)
    else:
        # Text-only fallback: try to extract a sink from the raw line
        if source_lines and 1 <= start_line <= len(source_lines):
            sink_expression = source_lines[start_line - 1].strip()[:200]

    # ------------------------------------------------------------------
    # Step 6: Data-flow hints
    # ------------------------------------------------------------------
    data_flow_hints: list[str] = []
    if snippet:
        try:
            data_flow_hints = _extract_data_flow_hints(snippet)
        except Exception as exc:
            log.warning("locate_sast: _extract_data_flow_hints failed: %s", exc)

    # ------------------------------------------------------------------
    # Step 7: Confidence
    # ------------------------------------------------------------------
    confidence = _score_confidence(
        file_readable=file_readable,
        ast_available=ast_available,
        enclosing_found=bool(enclosing_symbol),
        sink_found=bool(sink_expression),
        imports_found=bool(imports),
    )

    log.info(
        "locate_sast: issue=%s file=%s symbol=%s confidence=%.2f",
        issue.id,
        issue.file_path,
        enclosing_symbol,
        confidence,
    )

    return LocalizedIssue(
        issue=issue,
        enclosing_symbol=enclosing_symbol,
        enclosing_node_type=enclosing_node_type,
        sink_expression=sink_expression,
        imports=imports,
        data_flow_hints=data_flow_hints,
        snippet=snippet,
        localization_confidence=confidence,
    )
