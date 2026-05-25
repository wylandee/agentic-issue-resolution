"""
Edit Tools — strict file editor for the Agentic AppSec Remediation Engine.

Separation-of-Concerns role
----------------------------
This module sits *after* the Fix Planner in the pipeline.  It receives a
fully-typed ``EditRequest`` (produced by the Remedy agent from a ``FixPlan``)
and applies exactly one exact-anchor string replacement to the target file.

It does **not**:
  * Decide what to edit.
  * Parse JSON, TypeScript, Python, or any other language.
  * Call any network API.
  * Mutate any Pydantic model.

Public API
----------
``apply_edit(request: EditRequest) -> EditResult``
    Validates and optionally applies a single exact-string replacement.
    Always returns a typed ``EditResult``; never raises.

Safety guarantees
-----------------
* The resolved target path must be inside ``repo_root`` (no escapes).
* ``old_text`` must appear *exactly once* in the file content.
* A CRLF→LF normalisation pass is attempted as a last resort before
  rejecting a "not found" result — so an agent that supplies LF text
  against a CRLF-encoded file will still succeed.
* Net line deletions are capped by ``request.max_deletion_lines``.
* Writes are atomic: a temp file is written then ``os.replace``d.
"""

from __future__ import annotations

import difflib
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple, Union

from src.contracts import EditRequest, EditResult, EditStatus

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Error message constants (exact strings asserted by tests)
# ---------------------------------------------------------------------------

_MSG_NOT_FOUND = (
    "old_text not found in file. Ensure exact spacing and indentation."
)
_MSG_AMBIGUOUS = (
    "old_text is ambiguous (found multiple times). Provide more context lines."
)
_MSG_TOO_MANY_DELETIONS = "Attempted to delete too many lines."
_MSG_BINARY = "Cannot edit binary or non-UTF-8 files."


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _reject(request: EditRequest, reason: str) -> EditResult:
    """Build a REJECTED ``EditResult`` with the given human-readable reason."""
    return EditResult(
        request=request,
        status=EditStatus.REJECTED,
        rejection_reason=reason,
    )


def _resolve_target(
    request: EditRequest,
) -> Union[Tuple[Path, Path], EditResult]:
    """
    Validate ``repo_root`` and ``file_path``, returning ``(repo_root_path,
    target_path)`` on success or a REJECTED ``EditResult`` on failure.

    Checks (in order):
    1. ``repo_root`` exists and is a directory.
    2. ``file_path`` is not absolute.
    3. Resolved target is strictly inside ``repo_root``.
    4. Target exists and is a regular file.
    """
    repo_root_path = Path(request.repo_root)

    if not repo_root_path.exists():
        return _reject(request, f"repo_root does not exist: {request.repo_root}")
    if not repo_root_path.is_dir():
        return _reject(request, f"repo_root is not a directory: {request.repo_root}")

    if Path(request.file_path).is_absolute():
        return _reject(
            request,
            f"file_path must be repo-relative, not absolute: {request.file_path}",
        )

    target_path = (repo_root_path / request.file_path).resolve()
    repo_root_resolved = repo_root_path.resolve()

    # Guard against path traversal that slipped past the Pydantic validator
    try:
        target_path.relative_to(repo_root_resolved)
    except ValueError:
        return _reject(
            request,
            f"file_path resolves outside repo_root: {request.file_path}",
        )

    if not target_path.exists():
        return _reject(request, f"Target file does not exist: {request.file_path}")
    if not target_path.is_file():
        return _reject(
            request,
            f"Target path is not a regular file: {request.file_path}",
        )

    return repo_root_resolved, target_path


def _read_utf8(path: Path, request: EditRequest) -> Union[str, EditResult]:
    """
    Read *path* and decode as UTF-8.

    Returns the file content string on success, or a REJECTED ``EditResult``
    if the file cannot be decoded (binary / non-UTF-8).
    """
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return _reject(request, _MSG_BINARY)


def _find_unique_anchor(
    content: str,
    old_text: str,
    request: EditRequest,
) -> Union[Tuple[str, str], EditResult]:
    """
    Locate ``old_text`` exactly once inside ``content``.

    Falls back to a conservative CRLF→LF normalisation pass when the literal
    ``old_text`` is absent but the normalised content matches once.

    Returns ``(matched_content, matched_old_text)`` — the versions of content
    and old_text that should be used for the replacement — or a REJECTED
    ``EditResult``.
    """
    count = content.count(old_text)

    if count == 0:
        # Conservative CRLF→LF fallback: only when the file uses CRLF
        if "\r\n" in content or "\r\n" in old_text:
            normalised_content = content.replace("\r\n", "\n")
            normalised_old = old_text.replace("\r\n", "\n")
            fallback_count = normalised_content.count(normalised_old)
            if fallback_count == 1:
                return normalised_content, normalised_old
            if fallback_count > 1:
                return _reject(request, _MSG_AMBIGUOUS)
        return _reject(request, _MSG_NOT_FOUND)

    if count > 1:
        return _reject(request, _MSG_AMBIGUOUS)

    return content, old_text


def _line_count(text: str) -> int:
    """Return the number of lines in *text* (empty string → 0)."""
    if not text:
        return 0
    return text.count("\n") + (0 if text.endswith("\n") else 1)


def _build_diff(
    file_path: str,
    old_content: str,
    new_content: str,
) -> Tuple[str, int, int]:
    """
    Generate a unified diff and count added/removed lines.

    Header lines (``--- a/...``, ``+++ b/...``) are excluded from the counts.

    Returns ``(diff_text, lines_added, lines_removed)``.
    """
    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)

    diff_lines = list(
        difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
        )
    )

    lines_added = 0
    lines_removed = 0
    for line in diff_lines:
        if line.startswith("+") and not line.startswith("+++"):
            lines_added += 1
        elif line.startswith("-") and not line.startswith("---"):
            lines_removed += 1

    return "".join(diff_lines), lines_added, lines_removed


def _atomic_write(path: Path, content: str) -> None:
    """
    Write *content* to *path* atomically.

    Creates a sibling temp file named ``.{name}.tmp.{pid}``, preserves the
    original file's permission bits, then uses ``os.replace`` for an atomic
    rename.  Cleans up the temp file on write failure.
    """
    tmp_path = path.parent / f".{path.name}.tmp.{os.getpid()}"
    try:
        tmp_path.write_bytes(content.encode("utf-8"))
        # Preserve original file mode
        original_mode = path.stat().st_mode
        os.chmod(tmp_path, original_mode)
        os.replace(tmp_path, path)
    except Exception:
        # Best-effort cleanup of the temp file
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_edit(request: EditRequest) -> EditResult:
    """
    Validate and apply (or dry-run) a single exact-anchor file replacement.

    This function never raises.  All error conditions are returned as
    ``EditResult`` objects with the appropriate ``EditStatus``.

    Parameters
    ----------
    request:
        A fully-typed ``EditRequest`` produced by the Remedy agent.

    Returns
    -------
    EditResult
        ``status=APPLIED``  — edit written to disk successfully.
        ``status=DRY_RUN``  — validation passed; no disk write (request.dry_run=True).
        ``status=REJECTED`` — a pre-write validation check failed.
        ``status=ERROR``    — an unexpected filesystem error occurred during write.
    """
    # ------------------------------------------------------------------
    # 1. Resolve and validate paths
    # ------------------------------------------------------------------
    resolved = _resolve_target(request)
    if isinstance(resolved, EditResult):
        return resolved
    _repo_root, target_path = resolved

    # ------------------------------------------------------------------
    # 2. Read file as UTF-8
    # ------------------------------------------------------------------
    content_result = _read_utf8(target_path, request)
    if isinstance(content_result, EditResult):
        return content_result
    content: str = content_result

    # ------------------------------------------------------------------
    # 3. Locate the unique anchor (with optional CRLF fallback)
    # ------------------------------------------------------------------
    anchor_result = _find_unique_anchor(content, request.old_text, request)
    if isinstance(anchor_result, EditResult):
        return anchor_result
    working_content, matched_old = anchor_result

    # ------------------------------------------------------------------
    # 4. Deletion guard
    # ------------------------------------------------------------------
    old_line_count = _line_count(matched_old)
    new_line_count = _line_count(request.new_text)
    net_deleted = max(0, old_line_count - new_line_count)
    if net_deleted > request.max_deletion_lines:
        return _reject(request, _MSG_TOO_MANY_DELETIONS)

    # ------------------------------------------------------------------
    # 5. Compute new content and generate diff
    # ------------------------------------------------------------------
    new_content = working_content.replace(matched_old, request.new_text, 1)
    diff_text, lines_added, lines_removed = _build_diff(
        request.file_path, working_content, new_content
    )

    # ------------------------------------------------------------------
    # 6. Dry-run: return stats without writing
    # ------------------------------------------------------------------
    if request.dry_run:
        log.info(
            "[edit_tools] DRY_RUN %s (+%d -%d)",
            request.file_path,
            lines_added,
            lines_removed,
        )
        return EditResult(
            request=request,
            status=EditStatus.DRY_RUN,
            unified_diff=diff_text or None,
            lines_added=lines_added,
            lines_removed=lines_removed,
            applied_at=None,
        )

    # ------------------------------------------------------------------
    # 7. Atomic write
    # ------------------------------------------------------------------
    try:
        _atomic_write(target_path, new_content)
    except Exception as exc:
        log.error("[edit_tools] write failed for %s: %s", request.file_path, exc)
        return EditResult(
            request=request,
            status=EditStatus.ERROR,
            rejection_reason=f"Write failed: {exc}",
        )

    log.info(
        "[edit_tools] APPLIED %s (+%d -%d)",
        request.file_path,
        lines_added,
        lines_removed,
    )
    return EditResult(
        request=request,
        status=EditStatus.APPLIED,
        unified_diff=diff_text or None,
        lines_added=lines_added,
        lines_removed=lines_removed,
        applied_at=datetime.now(timezone.utc),
    )
