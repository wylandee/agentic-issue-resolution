"""Characterization tests for atomic report persistence."""

from __future__ import annotations

import threading
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
    assert not list(report_path.parent.glob(f".{report_path.name}.*.tmp"))


def test_concurrent_report_writers_keep_independent_atomic_temporaries(
    tmp_path: Path, monkeypatch
) -> None:
    """Interleaved writers do not consume each other's temporary file."""
    report_path = tmp_path / "report.md"
    first_writer_at_replace = threading.Event()
    second_writer_replaced = threading.Event()
    original_replace = Path.replace
    failures: list[BaseException] = []

    def controlled_replace(source: Path, target: Path) -> Path:
        if threading.current_thread().name == "report-writer-a":
            first_writer_at_replace.set()
            assert second_writer_replaced.wait(timeout=5)
        result = original_replace(source, target)
        if threading.current_thread().name == "report-writer-b":
            second_writer_replaced.set()
        return result

    monkeypatch.setattr(Path, "replace", controlled_replace)

    def write(markdown: str) -> None:
        try:
            write_report_atomic(report_path, markdown)
        except BaseException as exc:  # capture thread failures for the assertion
            failures.append(exc)

    first = threading.Thread(target=write, args=("writer a\n",), name="report-writer-a")
    second = threading.Thread(target=write, args=("writer b\n",), name="report-writer-b")
    first.start()
    assert first_writer_at_replace.wait(timeout=5)
    second.start()
    first.join(timeout=5)
    second.join(timeout=5)

    assert not first.is_alive() and not second.is_alive()
    assert failures == []
    assert report_path.read_text(encoding="utf-8") in {"writer a\n", "writer b\n"}
    assert not list(tmp_path.glob(".report.md.*.tmp"))
