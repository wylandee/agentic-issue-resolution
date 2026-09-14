"""Characterization tests for atomic report persistence."""

from __future__ import annotations

from pathlib import Path

from remediation_engine.orchestration.report_persistence import write_report_atomic


def test_write_report_atomic_writes_exact_content_and_removes_temporary_file(
    tmp_path: Path,
) -> None:
    """A successful write creates the report and leaves no sibling temporary file."""
    report_path = tmp_path / "reports" / "remediation_run-42.md"
    markdown = "# Remediation report\n\nPackage: lodash\nStatus: complete\n"

    write_report_atomic(report_path, markdown)

    assert report_path.is_file()
    assert report_path.read_text(encoding="utf-8") == markdown
    temporary_path = report_path.with_name(f".{report_path.name}.tmp")
    assert not temporary_path.exists()
