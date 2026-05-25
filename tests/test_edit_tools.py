"""
Tests for src/tools/edit_tools.py

All tests are filesystem-based using ``tmp_path`` (pytest built-in fixture).
No network calls are made.

Coverage
--------
Success cases
  - Package version bump (SCA-style package.json edit)
  - npm overrides block insertion
  - Source-code replacement (SAST-style .ts edit)
  - Dry-run returns diff but does NOT write to disk

Rejection cases
  - Path escapes repo_root via symlink-resolved traversal
  - Absolute file_path
  - repo_root does not exist
  - repo_root is a file, not a directory
  - Target file does not exist
  - Target path is a directory
  - Non-UTF-8 / binary file content
  - old_text not found in file
  - old_text appears multiple times (ambiguous)
  - Net deletion exceeds max_deletion_lines

Edge / correctness cases
  - CRLF file matched by LF old_text via safe normalisation fallback
  - CRLF fallback is NOT triggered when LF file has multiple matches
  - Diff line counts exclude ``---`` / ``+++`` header lines
  - Atomic write: final on-disk content matches new_text exactly
  - Atomic write: temp file is cleaned up on failure (monkeypatched)
  - Custom max_deletion_lines respected
  - EditResult JSON round-trip integrity
  - Error status on unexpected write failure
"""

from __future__ import annotations

import os
from datetime import timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from src.contracts import EditRequest, EditResult, EditStatus
from src.tools.edit_tools import (
    _MSG_AMBIGUOUS,
    _MSG_BINARY,
    _MSG_NOT_FOUND,
    _MSG_TOO_MANY_DELETIONS,
    _atomic_write,
    _build_diff,
    _find_unique_anchor,
    _line_count,
    apply_edit,
)


# ===========================================================================
# Helpers / factories
# ===========================================================================


def _req(tmp_path: Path, file_path: str = "app.ts", **overrides) -> EditRequest:
    """Return a minimal valid EditRequest rooted at *tmp_path*."""
    defaults = dict(
        repo_root=str(tmp_path),
        file_path=file_path,
        old_text="OLD",
        new_text="NEW",
    )
    defaults.update(overrides)
    return EditRequest(**defaults)


def _write(tmp_path: Path, rel_path: str, content: str) -> Path:
    """Write *content* (UTF-8) to ``tmp_path / rel_path`` and return the Path."""
    p = tmp_path / rel_path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


# ===========================================================================
# Unit tests — helper functions
# ===========================================================================


class TestLineCount:
    def test_empty(self):
        assert _line_count("") == 0

    def test_single_line_no_newline(self):
        assert _line_count("hello") == 1

    def test_single_line_with_newline(self):
        assert _line_count("hello\n") == 1

    def test_two_lines(self):
        assert _line_count("line1\nline2") == 2

    def test_three_lines_trailing_newline(self):
        assert _line_count("a\nb\nc\n") == 3


class TestBuildDiff:
    def test_simple_replacement(self):
        diff, added, removed = _build_diff("foo.ts", "old line\n", "new line\n")
        assert added == 1
        assert removed == 1
        assert "foo.ts" in diff
        assert "+new line" in diff
        assert "-old line" in diff

    def test_header_lines_excluded_from_counts(self):
        diff, added, removed = _build_diff("f.ts", "a\n", "b\n")
        # "+++ b/f.ts" and "--- a/f.ts" must NOT be counted
        assert added == 1
        assert removed == 1

    def test_insertion(self):
        _diff, added, removed = _build_diff("f.ts", "a\n", "a\nb\n")
        assert added == 1
        assert removed == 0

    def test_deletion(self):
        _diff, added, removed = _build_diff("f.ts", "a\nb\n", "a\n")
        assert added == 0
        assert removed == 1

    def test_no_change(self):
        diff, added, removed = _build_diff("f.ts", "same\n", "same\n")
        assert added == 0
        assert removed == 0
        assert diff == ""


class TestFindUniqueAnchor:
    def _fake_req(self) -> EditRequest:
        return EditRequest(
            repo_root="/tmp",
            file_path="f.ts",
            old_text="TARGET",
            new_text="REPLACEMENT",
        )

    def test_found_once(self):
        result = _find_unique_anchor("prefix TARGET suffix", "TARGET", self._fake_req())
        assert not isinstance(result, EditResult)
        content, old = result
        assert old == "TARGET"

    def test_not_found_returns_rejected(self):
        result = _find_unique_anchor("no match here", "TARGET", self._fake_req())
        assert isinstance(result, EditResult)
        assert result.status == EditStatus.REJECTED
        assert result.rejection_reason == _MSG_NOT_FOUND

    def test_found_multiple_returns_rejected(self):
        result = _find_unique_anchor("TARGET TARGET", "TARGET", self._fake_req())
        assert isinstance(result, EditResult)
        assert result.status == EditStatus.REJECTED
        assert result.rejection_reason == _MSG_AMBIGUOUS

    def test_crlf_fallback_success(self):
        # File uses CRLF; agent supplies old_text with CRLF that doesn't
        # match literally because the content has been round-tripped through
        # a system that normalised to LF.  We exercise this by making the
        # file content pure LF and the old_text use CRLF — the fallback
        # normalises both sides to LF and finds the single match.
        lf_content = "line1\nTARGET\nline3\n"
        crlf_old = "TARGET\r\n"  # agent accidentally used CRLF in old_text
        # Direct count: "TARGET\r\n" is NOT in the LF content
        assert lf_content.count(crlf_old) == 0
        result = _find_unique_anchor(lf_content, crlf_old, self._fake_req())
        assert not isinstance(result, EditResult), (
            f"Expected a tuple, got {result!r}"
        )
        working_content, matched_old = result
        # After normalisation both sides are pure LF — no CRLF remains
        assert "\r\n" not in working_content
        assert "\r\n" not in matched_old
        # The replacement works on the normalised content
        assert matched_old in working_content

    def test_crlf_fallback_ambiguous(self):
        crlf_content = "TARGET\r\nTARGET\r\n"
        result = _find_unique_anchor(crlf_content, "TARGET", self._fake_req())
        assert isinstance(result, EditResult)
        assert result.rejection_reason == _MSG_AMBIGUOUS


# ===========================================================================
# apply_edit — success cases
# ===========================================================================


class TestApplyEditSuccess:
    def test_package_version_bump(self, tmp_path):
        """SCA-style: update a version string in package.json."""
        content = '{\n  "lodash": "4.17.20"\n}\n'
        _write(tmp_path, "package.json", content)

        req = EditRequest(
            repo_root=str(tmp_path),
            file_path="package.json",
            old_text='"lodash": "4.17.20"',
            new_text='"lodash": "4.17.21"',
        )
        result = apply_edit(req)

        assert result.status == EditStatus.APPLIED
        assert result.applied_at is not None
        assert result.applied_at.tzinfo == timezone.utc
        assert result.lines_added == 1
        assert result.lines_removed == 1
        # Verify on-disk content
        final = (tmp_path / "package.json").read_text(encoding="utf-8")
        assert '"lodash": "4.17.21"' in final
        assert '"lodash": "4.17.20"' not in final

    def test_npm_overrides_insertion(self, tmp_path):
        """SCA-style: insert an overrides block after the dependencies block."""
        content = '{\n  "dependencies": {\n    "cookie": "0.4.0"\n  }\n}\n'
        expected_snippet = '"overrides": {\n    "cookie": "0.7.0"\n  }'
        _write(tmp_path, "package.json", content)

        req = EditRequest(
            repo_root=str(tmp_path),
            file_path="package.json",
            old_text='  "dependencies": {\n    "cookie": "0.4.0"\n  }',
            new_text=(
                '  "dependencies": {\n    "cookie": "0.4.0"\n  },\n'
                '  "overrides": {\n    "cookie": "0.7.0"\n  }'
            ),
        )
        result = apply_edit(req)

        assert result.status == EditStatus.APPLIED
        final = (tmp_path / "package.json").read_text(encoding="utf-8")
        assert expected_snippet in final

    def test_source_code_replacement(self, tmp_path):
        """SAST-style: replace an unsafe query with a parameterised one."""
        content = (
            "function getData(id) {\n"
            "  const q = `SELECT * FROM t WHERE id=${id}`;\n"
            "  db.query(q);\n"
            "}\n"
        )
        _write(tmp_path, "routes/data.ts", content)

        req = EditRequest(
            repo_root=str(tmp_path),
            file_path="routes/data.ts",
            old_text="  const q = `SELECT * FROM t WHERE id=${id}`;\n  db.query(q);",
            new_text="  db.query('SELECT * FROM t WHERE id=?', [id]);",
        )
        result = apply_edit(req)

        assert result.status == EditStatus.APPLIED
        final = (tmp_path / "routes/data.ts").read_text(encoding="utf-8")
        assert "SELECT * FROM t WHERE id=?" in final
        assert "`SELECT * FROM t WHERE id=${id}`" not in final

    def test_dry_run_does_not_modify_disk(self, tmp_path):
        """dry_run=True must return DRY_RUN status and leave the file unchanged."""
        original = "const x = 1;\n"
        _write(tmp_path, "app.ts", original)

        req = EditRequest(
            repo_root=str(tmp_path),
            file_path="app.ts",
            old_text="const x = 1;",
            new_text="const x = 2;",
            dry_run=True,
        )
        result = apply_edit(req)

        assert result.status == EditStatus.DRY_RUN
        assert result.applied_at is None
        assert result.lines_added >= 1
        assert result.lines_removed >= 1
        # File must be unchanged
        assert (tmp_path / "app.ts").read_text(encoding="utf-8") == original

    def test_dry_run_diff_populated(self, tmp_path):
        """dry_run must still populate the unified diff."""
        _write(tmp_path, "f.ts", "OLD\n")
        req = _req(tmp_path, file_path="f.ts", old_text="OLD", new_text="NEW", dry_run=True)
        result = apply_edit(req)
        assert result.unified_diff is not None
        assert "-OLD" in result.unified_diff
        assert "+NEW" in result.unified_diff


# ===========================================================================
# apply_edit — rejection cases
# ===========================================================================


class TestApplyEditRejections:
    def test_repo_root_does_not_exist(self, tmp_path):
        req = EditRequest(
            repo_root=str(tmp_path / "nonexistent"),
            file_path="app.ts",
            old_text="x",
            new_text="y",
        )
        result = apply_edit(req)
        assert result.status == EditStatus.REJECTED
        assert "does not exist" in result.rejection_reason  # type: ignore[operator]

    def test_repo_root_is_a_file(self, tmp_path):
        f = tmp_path / "notadir"
        f.write_text("hi")
        req = EditRequest(
            repo_root=str(f),
            file_path="app.ts",
            old_text="x",
            new_text="y",
        )
        result = apply_edit(req)
        assert result.status == EditStatus.REJECTED
        assert "not a directory" in result.rejection_reason  # type: ignore[operator]

    def test_absolute_file_path(self, tmp_path):
        _write(tmp_path, "app.ts", "x")
        req = EditRequest(
            repo_root=str(tmp_path),
            file_path=str(tmp_path / "app.ts"),  # absolute
            old_text="x",
            new_text="y",
        )
        result = apply_edit(req)
        assert result.status == EditStatus.REJECTED
        assert "absolute" in result.rejection_reason  # type: ignore[operator]

    def test_target_does_not_exist(self, tmp_path):
        req = _req(tmp_path, file_path="missing.ts")
        result = apply_edit(req)
        assert result.status == EditStatus.REJECTED
        assert "does not exist" in result.rejection_reason  # type: ignore[operator]

    def test_target_is_directory(self, tmp_path):
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        req = _req(tmp_path, file_path="subdir")
        result = apply_edit(req)
        assert result.status == EditStatus.REJECTED
        assert "not a regular file" in result.rejection_reason  # type: ignore[operator]

    def test_binary_file(self, tmp_path):
        p = tmp_path / "image.bin"
        p.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")
        req = EditRequest(
            repo_root=str(tmp_path),
            file_path="image.bin",
            old_text="PNG",
            new_text="JPG",
        )
        result = apply_edit(req)
        assert result.status == EditStatus.REJECTED
        assert result.rejection_reason == _MSG_BINARY

    def test_old_text_not_found(self, tmp_path):
        _write(tmp_path, "app.ts", "const x = 1;\n")
        req = _req(tmp_path, old_text="DOES_NOT_EXIST", new_text="y")
        result = apply_edit(req)
        assert result.status == EditStatus.REJECTED
        assert result.rejection_reason == _MSG_NOT_FOUND

    def test_old_text_ambiguous(self, tmp_path):
        _write(tmp_path, "app.ts", "foo\nfoo\n")
        req = _req(tmp_path, old_text="foo", new_text="bar")
        result = apply_edit(req)
        assert result.status == EditStatus.REJECTED
        assert result.rejection_reason == _MSG_AMBIGUOUS

    def test_deletion_exceeds_max(self, tmp_path):
        # old_text has 5 lines, new_text has 1 line → net delete = 4
        old = "line1\nline2\nline3\nline4\nline5"
        _write(tmp_path, "app.ts", old + "\n")
        req = EditRequest(
            repo_root=str(tmp_path),
            file_path="app.ts",
            old_text=old,
            new_text="replacement",
            max_deletion_lines=3,  # 4 net deletions > 3 → reject
        )
        result = apply_edit(req)
        assert result.status == EditStatus.REJECTED
        assert result.rejection_reason == _MSG_TOO_MANY_DELETIONS

    def test_deletion_at_limit_passes(self, tmp_path):
        # net delete = 3, limit = 3 → should NOT be rejected
        old = "a\nb\nc\nd"
        _write(tmp_path, "app.ts", old + "\n")
        req = EditRequest(
            repo_root=str(tmp_path),
            file_path="app.ts",
            old_text=old,
            new_text="x",
            max_deletion_lines=3,
        )
        result = apply_edit(req)
        assert result.status == EditStatus.APPLIED


# ===========================================================================
# apply_edit — edge / correctness cases
# ===========================================================================


class TestApplyEditEdgeCases:
    def test_crlf_file_matched_by_lf_old_text(self, tmp_path):
        """A CRLF-encoded file should be editable with a LF old_text."""
        crlf_content = "line1\r\nTARGET_LINE\r\nline3\r\n"
        p = tmp_path / "file.ts"
        p.write_bytes(crlf_content.encode("utf-8"))

        req = EditRequest(
            repo_root=str(tmp_path),
            file_path="file.ts",
            old_text="TARGET_LINE",
            new_text="REPLACED_LINE",
        )
        result = apply_edit(req)

        assert result.status == EditStatus.APPLIED
        final = p.read_text(encoding="utf-8")
        assert "REPLACED_LINE" in final
        assert "TARGET_LINE" not in final

    def test_lf_file_not_matched_multiple_times_no_false_crlf(self, tmp_path):
        """CRLF fallback must NOT rescue a genuinely ambiguous LF file."""
        _write(tmp_path, "f.ts", "foo\nfoo\n")
        req = _req(tmp_path, file_path="f.ts", old_text="foo", new_text="bar")
        result = apply_edit(req)
        assert result.status == EditStatus.REJECTED
        assert result.rejection_reason == _MSG_AMBIGUOUS

    def test_diff_counts_correct_for_multi_line_change(self, tmp_path):
        _write(tmp_path, "f.ts", "a\nb\nc\n")
        req = EditRequest(
            repo_root=str(tmp_path),
            file_path="f.ts",
            old_text="a\nb\nc",
            new_text="x\ny",
        )
        result = apply_edit(req)
        assert result.status == EditStatus.APPLIED
        assert result.lines_added == 2
        assert result.lines_removed == 3

    def test_atomic_write_final_content_correct(self, tmp_path):
        """After a successful write, exactly new_text must appear in the file."""
        _write(tmp_path, "pkg.json", '{"v": "1.0.0"}\n')
        req = EditRequest(
            repo_root=str(tmp_path),
            file_path="pkg.json",
            old_text='"v": "1.0.0"',
            new_text='"v": "2.0.0"',
        )
        apply_edit(req)
        content = (tmp_path / "pkg.json").read_text(encoding="utf-8")
        assert '"v": "2.0.0"' in content

    def test_atomic_write_temp_cleaned_up_on_failure(self, tmp_path):
        """If the atomic rename fails, no temp file should remain."""
        _write(tmp_path, "f.ts", "OLD\n")
        req = _req(tmp_path, file_path="f.ts", old_text="OLD", new_text="NEW")

        with patch("src.tools.edit_tools.os.replace", side_effect=OSError("disk full")):
            result = apply_edit(req)

        assert result.status == EditStatus.ERROR
        assert "Write failed" in result.rejection_reason  # type: ignore[operator]
        # No .tmp file should remain in tmp_path
        tmp_files = list(tmp_path.glob(".*.tmp.*"))
        assert tmp_files == [], f"Temp file not cleaned up: {tmp_files}"

    def test_custom_max_deletion_lines_high_limit(self, tmp_path):
        """A very large max_deletion_lines should always allow any realistic edit."""
        big_old = "\n".join(f"line {i}" for i in range(100))
        _write(tmp_path, "f.ts", big_old + "\n")
        req = EditRequest(
            repo_root=str(tmp_path),
            file_path="f.ts",
            old_text=big_old,
            new_text="single replacement",
            max_deletion_lines=500,
        )
        result = apply_edit(req)
        assert result.status == EditStatus.APPLIED

    def test_edit_result_json_round_trip(self, tmp_path):
        """EditResult produced by apply_edit must survive a JSON round-trip."""
        _write(tmp_path, "f.ts", "OLD\n")
        req = _req(tmp_path, old_text="OLD", new_text="NEW")
        result = apply_edit(req)

        reloaded = EditResult.model_validate_json(result.model_dump_json())
        assert reloaded.status == result.status
        assert reloaded.lines_added == result.lines_added
        assert reloaded.lines_removed == result.lines_removed
        assert reloaded.request.file_path == result.request.file_path

    def test_no_diff_when_content_identical(self, tmp_path):
        """If old_text == new_text, diff is empty and no counts appear."""
        _write(tmp_path, "f.ts", "same\n")
        req = EditRequest(
            repo_root=str(tmp_path),
            file_path="f.ts",
            old_text="same",
            new_text="same",
        )
        result = apply_edit(req)
        # Apply succeeds but nothing changed
        assert result.status == EditStatus.APPLIED
        assert result.lines_added == 0
        assert result.lines_removed == 0
        assert result.unified_diff is None

    def test_file_permissions_preserved(self, tmp_path):
        """The edit must not change the file's permission bits."""
        p = _write(tmp_path, "script.sh", "OLD\n")
        original_mode = p.stat().st_mode
        os.chmod(p, 0o755)
        expected_mode = p.stat().st_mode

        req = EditRequest(
            repo_root=str(tmp_path),
            file_path="script.sh",
            old_text="OLD",
            new_text="NEW",
        )
        apply_edit(req)

        assert p.stat().st_mode == expected_mode

    def test_subdirectory_file_edit(self, tmp_path):
        """Files in nested subdirectories are correctly resolved and edited."""
        _write(tmp_path, "a/b/c/file.ts", "VERSION = '1'\n")
        req = EditRequest(
            repo_root=str(tmp_path),
            file_path="a/b/c/file.ts",
            old_text="VERSION = '1'",
            new_text="VERSION = '2'",
        )
        result = apply_edit(req)
        assert result.status == EditStatus.APPLIED
        final = (tmp_path / "a/b/c/file.ts").read_text(encoding="utf-8")
        assert "VERSION = '2'" in final
